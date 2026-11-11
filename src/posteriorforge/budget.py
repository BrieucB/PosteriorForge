"""Fail-closed cumulative allocated-GPU accounting for the fixed 200 hour cap.

This ledger belongs in private SCRATCH, never a shared Git checkout. Reservations
are durable before sbatch is called. Unknown submission outcomes keep their full
reservation until reconciled; process failures never make capacity reappear.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal, ROUND_CEILING
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import uuid

CAP_GPU_SECONDS = 200 * 3600
SCHEMA = "posteriorforge.gpu_budget.v1"
TERMINAL_STATES = frozenset({"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL",
                             "OUT_OF_MEMORY", "PREEMPTED", "BOOT_FAIL", "REVOKED", "DEADLINE"})
HELD_STATES = frozenset({"reserved", "submitting", "submitted", "uncertain"})
SACCT_FIELDS = ("JobIDRaw", "State", "ElapsedRaw", "AllocTRES", "Account", "Partition", "JobName")


class BudgetError(RuntimeError):
    """A reservation or accounting invariant was not satisfied."""


class SubmissionUncertain(BudgetError):
    """A submission might exist; its durable reservation remains held."""


def _now():
    return datetime.now(timezone.utc).isoformat()


def _positive_int(value, label):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _gpu_count(tres):
    counts = {}
    for field in tres.split(","):
        key, separator, value = field.partition("=")
        if separator and (key == "gres/gpu" or key.startswith("gres/gpu:")):
            if not value.isdecimal():
                raise BudgetError("noninteger allocated GPU TRES")
            counts[key] = int(value)
    if "gres/gpu" in counts:
        typed = sum(value for key, value in counts.items() if key != "gres/gpu")
        if typed and typed != counts["gres/gpu"]:
            raise BudgetError("generic and typed allocated GPU counts disagree")
        return counts["gres/gpu"]
    return sum(counts.values())


def parse_sacct(text):
    """Parse only allocation records; steps and array tasks are unsupported."""
    records = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        fields = line.split("|")
        if len(fields) == len(SACCT_FIELDS) + 1 and not fields[-1]:
            fields.pop()
        if len(fields) != len(SACCT_FIELDS):
            raise BudgetError("unexpected sacct field count")
        row = dict(zip(SACCT_FIELDS, fields))
        if not row["JobIDRaw"].isdigit():
            continue
        if row["JobIDRaw"] in records:
            raise BudgetError("duplicate Slurm allocation accounting record")
        records[row["JobIDRaw"]] = row
    return records


class BudgetLedger:
    """Atomic ledger; cap is immutable and failed runs count at actual allocation.

    Methods return JSON-safe dictionaries. All call sites must use this same
    ledger for references, qualification, inference, retries and profiling.
    """
    def __init__(self, path, cap_gpu_hours=200):
        if cap_gpu_hours != 200:
            raise ValueError("the authorized hard cap is exactly 200 GPU-hours")
        self.path = Path(path)
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")

    @contextmanager
    def _locked(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            if self.path.exists():
                value = json.loads(self.path.read_text())
            else:
                value = {"schema": SCHEMA, "cap_gpu_seconds": CAP_GPU_SECONDS,
                         "created_at": _now(), "reservations": [], "events": []}
            self._validate(value)
            try:
                yield value
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _validate(value):
        if value.get("schema") != SCHEMA or value.get("cap_gpu_seconds") != CAP_GPU_SECONDS:
            raise BudgetError("ledger schema or immutable hard cap mismatch")
        entries = value.get("reservations")
        if not isinstance(entries, list):
            raise BudgetError("invalid reservation records")
        ids, jobs = set(), set()
        for entry in entries:
            identity = entry.get("reservation_id")
            job = entry.get("job_id")
            if not identity or identity in ids or (job is not None and job in jobs):
                raise BudgetError("duplicate or missing reservation/job identity")
            ids.add(identity)
            if job is not None:
                jobs.add(job)
            if entry.get("state") not in HELD_STATES | {"reconciled", "released"}:
                raise BudgetError("unknown reservation state")
            if type(entry.get("gpus")) is not int or entry["gpus"] not in (1, 2, 4):
                raise BudgetError("invalid reserved GPU count")
            wall = entry.get("walltime_seconds")
            if type(wall) is not int or wall <= 0 or entry.get("reserved_gpu_seconds") != entry["gpus"] * wall:
                raise BudgetError("reservation arithmetic mismatch")
            if type(entry.get("actual_gpu_seconds", 0)) is not int or entry.get("actual_gpu_seconds", 0) < 0:
                raise BudgetError("invalid actual GPU allocation accounting")
            if entry["state"] in ("reconciled", "submitted") and not job:
                raise BudgetError("submitted/reconciled reservation lacks job identity")

    def _save(self, value):
        self._validate(value)
        descriptor, temporary = tempfile.mkstemp(prefix=self.path.name + ".", dir=self.path.parent)
        try:
            os.chmod(temporary, 0o600)
            with os.fdopen(descriptor, "w") as handle:
                json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
            directory = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @staticmethod
    def _summary(value):
        actual = sum(entry.get("actual_gpu_seconds", 0) for entry in value["reservations"])
        reserved = sum(entry["reserved_gpu_seconds"] for entry in value["reservations"] if entry["state"] in HELD_STATES)
        return {"cap_gpu_hours": 200, "actual_gpu_hours": actual / 3600,
                "reserved_gpu_hours": reserved / 3600,
                "remaining_gpu_hours": (CAP_GPU_SECONDS - actual - reserved) / 3600,
                "actual_gpu_seconds": actual, "reserved_gpu_seconds": reserved,
                "remaining_gpu_seconds": CAP_GPU_SECONDS - actual - reserved,
                "cap_exceeded": actual + reserved > CAP_GPU_SECONDS,
                "reservation_count": len(value["reservations"])}

    def summary(self):
        with self._locked() as value:
            return self._summary(value)

    def inspect(self):
        with self._locked() as value:
            return json.loads(json.dumps(value))

    @staticmethod
    def _entry(value, reservation_id):
        for entry in value["reservations"]:
            if entry["reservation_id"] == reservation_id:
                return entry
        raise BudgetError("unknown reservation")

    def _reserve_locked(self, value, *, gpus, walltime_seconds, purpose, metadata=None):
        if type(gpus) is not int or gpus not in (1, 2, 4):
            raise ValueError("same-node qgpu allocation requires 1, 2 or 4 GPUs")
        walltime_seconds = _positive_int(walltime_seconds, "walltime_seconds")
        if walltime_seconds > 2 * 86400:
            raise ValueError("qgpu walltime exceeds the 48 hour limit")
        if not isinstance(purpose, str) or not purpose.strip():
            raise ValueError("a nonempty purpose is required")
        cost = gpus * walltime_seconds
        if cost > self._summary(value)["remaining_gpu_seconds"]:
            raise BudgetError("reservation would exceed cumulative 200 allocated GPU-hour cap")
        entry = {"reservation_id": uuid.uuid4().hex, "gpus": gpus,
                 "walltime_seconds": walltime_seconds, "reserved_gpu_seconds": cost,
                 "purpose": purpose, "metadata": metadata or {}, "created_at": _now(),
                 "state": "reserved", "job_id": None, "actual_gpu_seconds": 0}
        value["reservations"].append(entry)
        value["events"].append({"time": _now(), "event": "reserve", "reservation_id": entry["reservation_id"]})
        self._save(value)
        return entry

    def reserve(self, *, gpus, walltime_seconds, purpose, metadata=None):
        with self._locked() as value:
            entry = self._reserve_locked(value, gpus=gpus, walltime_seconds=walltime_seconds,
                                         purpose=purpose, metadata=metadata)
            return dict(entry)

    def release_unsubmitted(self, reservation_id, *, reason):
        """Release a reservation only when submission has never been attempted."""
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("release reason required")
        with self._locked() as value:
            entry = self._entry(value, reservation_id)
            if entry["state"] != "reserved" or entry["job_id"] is not None:
                raise BudgetError("attempted or uncertain submissions cannot be released")
            entry.update(state="released", release_reason=reason, released_at=_now())
            value["events"].append({"time": _now(), "event": "release_unsubmitted", "reservation_id": reservation_id})
            self._save(value)
            return dict(entry)

    def submit(self, script, *, gpus, walltime_seconds, purpose, job_name,
               log_directory, script_args=(), metadata=None, timeout=60):
        """Reserve then submit a single-node allocation; never retries sbatch.

        Scripts must not contain #SBATCH options. Resources are fixed here, so a
        script cannot silently override the reserved GPU count or walltime.
        Automatic requeue is disabled: every retry requires its own reservation.
        A nonzero exit or timeout holds capacity because acceptance is unknown.
        """
        script = Path(script).resolve(strict=True)
        if any(line.lstrip().startswith("#SBATCH") for line in script.read_text().splitlines()):
            raise ValueError("SBATCH directives are forbidden; use budget-controlled arguments")
        if not re.fullmatch(r"pf-[a-zA-Z0-9_-]{1,80}", job_name):
            raise ValueError("job_name must begin pf- and contain simple characters")
        if not all(isinstance(arg, str) for arg in script_args):
            raise ValueError("script_args must be strings")
        directory = Path(log_directory).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        walltime_seconds = _positive_int(walltime_seconds, "walltime_seconds")
        hours, remainder = divmod(walltime_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        argv = ["sbatch", "--parsable", "--no-requeue", "--account=eu-26-17", "--partition=qgpu", "--nodes=1",
                "--ntasks=1", f"--gpus-per-node={gpus}", f"--cpus-per-task={16 * gpus}",
                f"--time={hours:02d}:{minutes:02d}:{seconds:02d}", f"--job-name={job_name}",
                f"--output={directory}/%x-%j.out", f"--error={directory}/%x-%j.err",
                str(script), *script_args]
        with self._locked() as value:
            entry = self._reserve_locked(value, gpus=gpus, walltime_seconds=walltime_seconds,
                                         purpose=purpose, metadata=metadata)
            identity = entry["reservation_id"]
            entry.update(state="submitting", submitted_argv=argv, submission_started_at=_now())
            self._save(value)
            try:
                result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
            except (OSError, subprocess.TimeoutExpired) as error:
                entry.update(state="uncertain", submission_error=type(error).__name__)
                self._save(value)
                raise SubmissionUncertain(f"submission outcome uncertain; reservation {identity} remains held") from error
            entry["submission_stdout"] = result.stdout.strip()
            entry["submission_stderr"] = result.stderr.strip()
            entry["submission_returncode"] = result.returncode
            match = re.fullmatch(r"([0-9]+)(?:;[a-zA-Z0-9_.-]+)?", result.stdout.strip())
            if result.returncode != 0 or match is None:
                entry["state"] = "uncertain"
                self._save(value)
                raise SubmissionUncertain(f"submission outcome uncertain; reservation {identity} remains held")
            entry.update(state="submitted", job_id=match.group(1), submitted_at=_now())
            value["events"].append({"time": _now(), "event": "submitted", "reservation_id": identity,
                                    "job_id": entry["job_id"]})
            self._save(value)
            return dict(entry)

    def attach_job(self, reservation_id, job_id):
        """Recover a known job ID after an uncertain submission, never free it."""
        if not isinstance(job_id, str) or not job_id.isdecimal():
            raise ValueError("a plain allocation job ID is required")
        with self._locked() as value:
            entry = self._entry(value, reservation_id)
            if entry["state"] not in ("reserved", "submitting", "uncertain") or entry["job_id"]:
                raise BudgetError("reservation cannot receive a job ID")
            entry.update(state="submitted", job_id=job_id, attached_at=_now())
            self._save(value)
            return dict(entry)

    def reconcile(self, reservation_id, record):
        """Reconcile one authenticated sacct allocation row; nonterminal stays held."""
        with self._locked() as value:
            entry = self._entry(value, reservation_id)
            if entry["state"] == "reconciled":
                return dict(entry)
            if entry["state"] not in HELD_STATES or not entry["job_id"]:
                raise BudgetError("only a known submitted allocation can be reconciled")
            if record.get("JobIDRaw") != entry["job_id"]:
                raise BudgetError("accounting job ID mismatch")
            if record.get("Account") != "eu-26-17" or record.get("Partition") != "qgpu":
                raise BudgetError("accounting account/partition mismatch")
            state = record.get("State", "").split()[0].rstrip("+")
            if state not in TERMINAL_STATES:
                entry.update(last_observed_state=state, last_observed_at=_now())
                self._save(value)
                return dict(entry)
            elapsed_text = record.get("ElapsedRaw", "")
            try:
                elapsed = int(Decimal(elapsed_text).to_integral_value(rounding=ROUND_CEILING))
            except Exception as error:
                raise BudgetError("invalid accounting elapsed seconds") from error
            if elapsed < 0:
                raise BudgetError("negative accounting elapsed seconds")
            allocated = _gpu_count(record.get("AllocTRES", ""))
            if elapsed > 0 and allocated == 0:
                raise BudgetError("nonzero elapsed allocation lacks GPU TRES; capacity remains held")
            actual = allocated * elapsed
            entry.update(state="reconciled", actual_gpu_seconds=actual,
                         accounted_gpus=allocated, accounted_elapsed_seconds=elapsed,
                         slurm_state=state, accounting_record=dict(record), reconciled_at=_now())
            entry["accounting_exceeded_reservation"] = actual > entry["reserved_gpu_seconds"]
            value["events"].append({"time": _now(), "event": "reconciled", "reservation_id": reservation_id,
                                    "actual_gpu_seconds": actual, "slurm_state": state})
            self._save(value)
            return dict(entry)

    def reconcile_all(self, *, timeout=60):
        snapshot = self.inspect()
        entries = [entry for entry in snapshot["reservations"] if entry["state"] in HELD_STATES and entry["job_id"]]
        if not entries:
            return {"updated": [], "missing_job_ids": [], "summary": self.summary()}
        ids = ",".join(entry["job_id"] for entry in entries)
        result = subprocess.run(["sacct", "-X", "-n", "-P", "-j", ids,
                                 "--format=JobIDRaw%30,State%40,ElapsedRaw,AllocTRES%300,Account%60,Partition%60,JobName%100"], capture_output=True,
                                text=True, timeout=timeout, check=True)
        records = parse_sacct(result.stdout)
        updated, missing = [], []
        for entry in entries:
            if entry["job_id"] not in records:
                missing.append(entry["job_id"])
                continue
            updated.append(self.reconcile(entry["reservation_id"], records[entry["job_id"]]))
        return {"updated": updated, "missing_job_ids": missing, "summary": self.summary()}
