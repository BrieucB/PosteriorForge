import json
import multiprocessing
import subprocess

import pytest

from posteriorforge.budget import BudgetError, BudgetLedger, SubmissionUncertain, parse_sacct


def _reserve_worker(path, output):
    try:
        BudgetLedger(path).reserve(gpus=4, walltime_seconds=90000, purpose="concurrent test")
        output.put(True)
    except BudgetError:
        output.put(False)


def _row(job="123", state="COMPLETED", elapsed="900", tres="cpu=16,gres/gpu=1,gres/gpu:a100=1"):
    return {"JobIDRaw": job, "State": state, "ElapsedRaw": elapsed, "AllocTRES": tres,
            "Account": "eu-26-17", "Partition": "qgpu", "JobName": "pf-test"}


def test_hard_cap_includes_concurrent_reservations(tmp_path):
    context = multiprocessing.get_context("fork")
    queue = context.Queue()
    path = str(tmp_path / "ledger.json")
    workers = [context.Process(target=_reserve_worker, args=(path, queue)) for _ in range(6)]
    for worker in workers:
        worker.start()
    outcomes = [queue.get(timeout=20) for _ in workers]
    for worker in workers:
        worker.join(timeout=20)
        assert worker.exitcode == 0
    assert sum(outcomes) == 2
    summary = BudgetLedger(path).summary()
    assert summary["reserved_gpu_hours"] == 200
    assert summary["remaining_gpu_hours"] == 0
    with pytest.raises(BudgetError, match="exceed"):
        BudgetLedger(path).reserve(gpus=1, walltime_seconds=1, purpose="overflow")


def test_failed_run_counts_actual_and_releases_unused_reservation(tmp_path):
    ledger = BudgetLedger(tmp_path / "ledger.json")
    entry = ledger.reserve(gpus=1, walltime_seconds=3600, purpose="failed qualification")
    ledger.attach_job(entry["reservation_id"], "123")
    ledger.reconcile(entry["reservation_id"], _row(state="FAILED"))
    summary = ledger.summary()
    assert summary["actual_gpu_hours"] == .25
    assert summary["reserved_gpu_hours"] == 0
    assert summary["remaining_gpu_hours"] == 199.75
    assert ledger.reconcile(entry["reservation_id"], _row())["actual_gpu_seconds"] == 900


def test_nonterminal_missing_tres_and_wrong_job_keep_reservation(tmp_path):
    ledger = BudgetLedger(tmp_path / "ledger.json")
    entry = ledger.reserve(gpus=2, walltime_seconds=3600, purpose="reference")
    ledger.attach_job(entry["reservation_id"], "123")
    ledger.reconcile(entry["reservation_id"], _row(state="RUNNING"))
    assert ledger.summary()["reserved_gpu_hours"] == 2
    with pytest.raises(BudgetError):
        ledger.reconcile(entry["reservation_id"], _row(job="other"))
    with pytest.raises(BudgetError, match="lacks GPU TRES"):
        ledger.reconcile(entry["reservation_id"], _row(tres="cpu=32"))
    assert ledger.summary()["reserved_gpu_hours"] == 2


def test_pending_cancelled_allocation_consumes_zero(tmp_path):
    ledger = BudgetLedger(tmp_path / "ledger.json")
    entry = ledger.reserve(gpus=1, walltime_seconds=3600, purpose="queue")
    ledger.attach_job(entry["reservation_id"], "123")
    ledger.reconcile(entry["reservation_id"], _row(state="CANCELLED by 1234", elapsed="0", tres=""))
    assert ledger.summary()["remaining_gpu_hours"] == 200


def test_unknown_submission_never_frees_capacity(tmp_path, monkeypatch):
    ledger = BudgetLedger(tmp_path / "ledger.json")
    script = tmp_path / "run.sh"
    script.write_text("#!/bin/bash\ntrue\n")
    def timeout(*args, **kwargs):
        # Simulate a crash after Slurm may already have accepted the allocation.
        raise subprocess.TimeoutExpired(args[0], 1)
    monkeypatch.setattr(subprocess, "run", timeout)
    with pytest.raises(SubmissionUncertain):
        ledger.submit(script, gpus=4, walltime_seconds=3600, purpose="uncertain",
                      job_name="pf-test", log_directory=tmp_path / "logs")
    assert ledger.summary()["reserved_gpu_hours"] == 4
    entry = ledger.inspect()["reservations"][0]
    assert entry["state"] == "uncertain"
    with pytest.raises(BudgetError):
        ledger.release_unsubmitted(entry["reservation_id"], reason="no response")


def test_submit_saves_before_sbatch_and_forbids_resource_override(tmp_path, monkeypatch):
    ledger = BudgetLedger(tmp_path / "ledger.json")
    script = tmp_path / "run.sh"
    script.write_text("#!/bin/bash\ntrue\n")
    def accepted(argv, **kwargs):
        state = json.loads(ledger.path.read_text())
        assert state["reservations"][0]["state"] == "submitting"
        assert state["reservations"][0]["reserved_gpu_seconds"] == 2 * 3600
        assert "--gpus-per-node=2" in argv
        assert "--nodes=1" in argv
        assert "--no-requeue" in argv
        return subprocess.CompletedProcess(argv, 0, "123;karolina\n", "")
    monkeypatch.setattr(subprocess, "run", accepted)
    entry = ledger.submit(script, gpus=2, walltime_seconds=3600, purpose="qualification",
                          job_name="pf-test", log_directory=tmp_path / "logs")
    assert entry["job_id"] == "123"
    script.write_text("#!/bin/bash\n#SBATCH --gpus=8\n")
    with pytest.raises(ValueError, match="directives"):
        ledger.submit(script, gpus=1, walltime_seconds=60, purpose="bad", job_name="pf-test",
                      log_directory=tmp_path / "logs")


def test_release_only_before_attempt_and_immutable_cap(tmp_path):
    ledger = BudgetLedger(tmp_path / "ledger.json")
    entry = ledger.reserve(gpus=1, walltime_seconds=3600, purpose="not needed")
    ledger.release_unsubmitted(entry["reservation_id"], reason="replaced before submission")
    assert ledger.summary()["remaining_gpu_hours"] == 200
    with pytest.raises(ValueError, match="exactly 200"):
        BudgetLedger(tmp_path / "other.json", cap_gpu_hours=201)
    payload = json.loads(ledger.path.read_text())
    payload["cap_gpu_seconds"] += 1
    ledger.path.write_text(json.dumps(payload))
    with pytest.raises(BudgetError, match="immutable"):
        ledger.summary()


def test_sacct_steps_ignored_and_gpu_types_not_double_counted(tmp_path):
    text = "123|FAILED|900|cpu=16,gres/gpu=1,gres/gpu:a100=1|eu-26-17|qgpu|pf-test|\n123.batch|FAILED|900||eu-26-17|qgpu|batch|\n"
    rows = parse_sacct(text)
    assert list(rows) == ["123"]
    ledger = BudgetLedger(tmp_path / "ledger.json")
    entry = ledger.reserve(gpus=1, walltime_seconds=3600, purpose="test")
    ledger.attach_job(entry["reservation_id"], "123")
    result = ledger.reconcile(entry["reservation_id"], rows["123"])
    assert result["actual_gpu_seconds"] == 900


def test_actual_overrun_recorded_truthfully_and_blocks_more_work(tmp_path):
    ledger = BudgetLedger(tmp_path / "ledger.json")
    first = ledger.reserve(gpus=4, walltime_seconds=90000, purpose="one")
    ledger.reserve(gpus=4, walltime_seconds=90000, purpose="two")
    ledger.attach_job(first["reservation_id"], "123")
    ledger.reconcile(first["reservation_id"], _row(elapsed="90001", tres="gres/gpu=4"))
    assert ledger.summary()["cap_exceeded"]
    with pytest.raises(BudgetError, match="exceed"):
        ledger.reserve(gpus=1, walltime_seconds=1, purpose="no room")


def test_inconsistent_tres_keeps_capacity_reserved(tmp_path):
    ledger = BudgetLedger(tmp_path / "ledger.json")
    entry = ledger.reserve(gpus=4, walltime_seconds=3600, purpose="test")
    ledger.attach_job(entry["reservation_id"], "123")
    with pytest.raises(BudgetError, match="disagree"):
        ledger.reconcile(entry["reservation_id"], _row(tres="gres/gpu=1,gres/gpu:a100=4"))
    assert ledger.summary()["reserved_gpu_hours"] == 4


@pytest.mark.parametrize("gpus", [1, 2, 4])
def test_literal_node_placement_preserves_reserved_resources(tmp_path, monkeypatch, gpus):
    ledger = BudgetLedger(tmp_path / "ledger.json")
    script = tmp_path / "run.sh"
    script.write_text("#!/bin/bash\ntrue\n")
    def accepted(argv, **kwargs):
        state = json.loads(ledger.path.read_text())
        entry = state["reservations"][0]
        assert entry["state"] == "submitting"
        assert entry["reserved_gpu_seconds"] == gpus * 600
        assert argv.count("--nodelist=acn65") == 1
        assert "--nodes=1" in argv and "--no-requeue" in argv
        assert f"--gpus-per-node={gpus}" in argv
        assert f"--cpus-per-task={16*gpus}" in argv
        assert argv[-1] == str(script)
        return subprocess.CompletedProcess(argv, 0, "123;karolina\n", "")
    monkeypatch.setattr(subprocess, "run", accepted)
    entry = ledger.submit(script, gpus=gpus, walltime_seconds=600,
        purpose="fixed-total same-node scaling", job_name="pf-placement",
        log_directory=tmp_path / "logs", node_name="acn65")
    assert entry["job_id"] == "123"
    assert ledger.summary()["reserved_gpu_seconds"] == gpus * 600
    ledger.reconcile(entry["reservation_id"], _row(elapsed="90", tres=f"gres/gpu={gpus}"))
    assert ledger.summary()["actual_gpu_seconds"] == gpus * 90
    assert ledger.summary()["reserved_gpu_seconds"] == 0


@pytest.mark.parametrize("node", ["", "acn65,acn66", "acn[1-4]", "--gpus=8", "acn65\n--nodes=2", 65])
def test_invalid_node_placement_never_reserves_or_submits(tmp_path, monkeypatch, node):
    ledger = BudgetLedger(tmp_path / "ledger.json")
    script = tmp_path / "run.sh"
    script.write_text("#!/bin/bash\ntrue\n")
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid placement must not call sbatch")
    monkeypatch.setattr(subprocess, "run", forbidden)
    with pytest.raises(ValueError, match="literal Karolina"):
        ledger.submit(script, gpus=1, walltime_seconds=600, purpose="invalid",
            job_name="pf-placement", log_directory=tmp_path / "logs", node_name=node)
    assert ledger.summary()["reservation_count"] == 0
