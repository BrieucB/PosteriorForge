"""Additive deterministic CDF query refinement using retained reference nodes.

No posterior samples, changed observed cases, or new quadrature nodes enter
query selection. Completed parent reference artifacts remain immutable.
"""
from __future__ import annotations

from hashlib import sha256
import os
from pathlib import Path
import time

import numpy as np

from .artifacts import runtime_identity, sha256_file, write_json
from .banks import derive_seed, load_heldout
from .budget import BudgetLedger
from .reference import load_reference_distribution, reference_moments
from .science.analytical_benchmark_v2 import array_descriptor, build_problem, unflatten_observations
from .science.analytical_benchmark_reference_v2 import (
    FullHierarchyReference, ReferenceNodes, refinement_witnesses, scramble_witness_summary,
)
from .validation import _reference_arrays, make_reference_input

SCHEMA = "posteriorforge.reference_grid.v1"
GRID_CONTRACT_SCHEMA = "posteriorforge.reference_grid_contract.v1"
CDF_PRECISION_LIMIT = .005
BASE_QUERIES = np.linspace(-8., 8., 513)


def _require_budgeted_gpu(budget_ledger=None):
    job = os.getenv("SLURM_JOB_ID")
    nodes = os.getenv("SLURM_JOB_NUM_NODES", os.getenv("SLURM_NNODES", "0"))
    if not job or not job.isdecimal() or nodes != "1":
        raise RuntimeError("reference grid refinement requires a single-node budgeted Slurm GPU allocation")
    path = Path(budget_ledger or os.getenv("PF_BUDGET_LEDGER") or
                (Path(os.environ.get("PF_SCRATCH", "/scratch/project/eu-26-17/eubrieucb/posteriorforge")) / "budget/ledger.json"))
    if not path.is_file():
        raise RuntimeError("reserved cumulative GPU budget ledger is missing")
    ledger = BudgetLedger(path)
    value = ledger.inspect()
    candidates = [entry for entry in value["reservations"] if entry.get("job_id") == job]
    if len(candidates) != 1 or candidates[0]["state"] not in ("submitted", "uncertain"):
        raise RuntimeError("current Slurm job lacks an active GPU budget reservation")
    entry = candidates[0]
    summary = ledger.summary()
    if summary["cap_exceeded"] or entry["gpus"] not in (1, 2, 4):
        raise RuntimeError("cumulative GPU budget invariant failed")
    return {"ledger": str(path.resolve()), "reservation_id": entry["reservation_id"],
            "job_id": job, "reserved_gpu_seconds": entry["reserved_gpu_seconds"],
            "gpus": entry["gpus"], "account": "eu-26-17", "partition": "qgpu"}


def _table_checks(queries, cdfs):
    if (queries.ndim != 1 or len(queries) < 2 or not np.isfinite(queries).all()
            or np.any(np.diff(queries) <= 0) or cdfs.shape != (53, len(queries))
            or not np.isfinite(cdfs).all()):
        raise ValueError("finite increasing queries and CDFs[53,q] required")
    if np.any(cdfs < -1e-12) or np.any(cdfs > 1 + 1e-12) or np.any(np.diff(cdfs, axis=1) < -1e-12):
        raise ValueError("retained reference CDF table is not monotone probabilities")


def _adaptive_grid(queries, cdfs, precision, evaluate, *, max_cdf_step,
                   max_rounds, max_queries, deadline):
    """Pure deterministic refinement engine; evaluate only previously unseen midpoints."""
    queries, cdfs = np.asarray(queries, dtype=np.float64), np.asarray(cdfs, dtype=np.float64)
    _table_checks(queries, cdfs)
    if not np.isfinite(precision) or not 0 <= precision <= CDF_PRECISION_LIMIT:
        raise ValueError("qualified parent CDF precision required")
    if (not np.isfinite(max_cdf_step) or not 0 < max_cdf_step <= .005
            or type(max_rounds) is not int or max_rounds < 0
            or type(max_queries) is not int or max_queries < len(queries)):
        raise ValueError("positive max CDF step<=.005 and valid refinement limits required")
    # The cache is authoritative: existing query values are never reevaluated.
    cache = {float(query): (cdfs[:, index].copy(), np.full(53, precision))
             for index, query in enumerate(queries)}
    history, reason, qualified = [], None, False
    for iteration in range(max_rounds + 1):
        queries = np.asarray(sorted(cache), dtype=np.float64)
        cdfs = np.column_stack([cache[float(query)][0] for query in queries])
        errors = np.column_stack([cache[float(query)][1] for query in queries])
        _table_checks(queries, cdfs)
        increments = np.diff(cdfs, axis=1)
        interval_max = increments.max(axis=0)
        selected = np.flatnonzero(interval_max > max_cdf_step)
        tails = float(max(cdfs[:, 0].max(), (1 - cdfs[:, -1]).max()))
        record = {"iteration": iteration, "point_count": len(queries),
                  "queries_sha256": array_descriptor(queries)["sha256"],
                  "maximum_cdf_step": float(increments.max()),
                  "maximum_marginal_cdf_step": float(increments[:21].max()),
                  "maximum_joint_cdf_step": float(increments[21:].max()),
                  "maximum_tail_mass": tails, "cdf_precision": float(errors.max()),
                  "selected_interval_indices": selected.tolist(), "added_queries": []}
        history.append(record)
        if time.monotonic() >= deadline:
            reason = "deadline exhausted"
            break
        if errors.max() > CDF_PRECISION_LIMIT:
            reason = "new-point retained scramble or consecutive-power precision failed"
            break
        if tails > CDF_PRECISION_LIMIT:
            reason = "fixed query endpoints have excessive reference tail mass"
            break
        if not len(selected):
            reason, qualified = "all deterministic grid requirements passed", True
            break
        if iteration == max_rounds:
            reason = "maximum bisection rounds exhausted"
            break
        if len(queries) + len(selected) > max_queries:
            reason = "maximum query count exhausted"
            break
        midpoints = queries[selected] + (queries[selected + 1] - queries[selected]) / 2
        if any(float(query) in cache for query in midpoints):
            raise RuntimeError("bisection requested an already evaluated query")
        record["added_queries"] = midpoints.tolist()
        try:
            values, uncertainty = evaluate(midpoints)
        except TimeoutError:
            reason = "deadline exhausted during retained-node evaluation"
            break
        values, uncertainty = np.asarray(values, dtype=np.float64), np.asarray(uncertainty, dtype=np.float64)
        if (values.shape != (53, len(midpoints)) or uncertainty.shape != values.shape
                or not np.isfinite(values).all() or not np.isfinite(uncertainty).all()
                or np.any(uncertainty < 0)):
            raise ValueError("retained-node evaluator returned invalid CDF/precision arrays")
        for index, query in enumerate(midpoints):
            cache[float(query)] = (values[:, index].copy(), uncertainty[:, index].copy())
    # A timeout during evaluation may leave a planned addition outside the cache.
    queries = np.asarray(sorted(cache), dtype=np.float64)
    cdfs = np.column_stack([cache[float(query)][0] for query in queries])
    precision = max(float(np.max(cache[float(query)][1])) for query in queries)
    return queries, cdfs, {"passed": qualified, "reason": reason,
            "cdf_precision": precision, "history": history,
            "maximum_cdf_step": float(np.diff(cdfs, axis=1).max()),
            "maximum_tail_mass": float(max(cdfs[:, 0].max(), (1 - cdfs[:, -1]).max()))}


def _authenticate_parent(banks_root, case_index, reference_root):
    root = Path(reference_root)
    manifest_sha = sha256_file(root / "manifest.json")
    arrays, parent = load_reference_distribution(root)
    _, _, _, _, qualified = _reference_arrays(make_reference_input(arrays, parent))
    if not qualified or parent.get("case_index") != case_index:
        raise ValueError("qualified parent reference for this case required")
    observed = load_heldout(banks_root, case_index)
    problem = build_problem()
    expected_binding = {"scientific_fingerprint": problem.scientific_fingerprint,
        "parameter_order": list(problem.parameter_order),
        "observation_sha256": array_descriptor(observed.obs)["sha256"]}
    if parent["binding"] != expected_binding:
        raise ValueError("parent reference does not match held-out science/observation")
    if not np.array_equal(arrays["queries"], BASE_QUERIES):
        raise ValueError("parent query grid must be the original513point[-8,8] grid")
    # Construction and all retained-node CDF evaluations use real GPU float64.
    reference = FullHierarchyReference(problem, unflatten_observations(observed.obs), backend="jax", device="gpu")
    directions, _ = refinement_witnesses(reference)
    if not np.array_equal(arrays["directions"], directions):
        raise ValueError("parent directions differ from the53predefined witnesses")
    combined_sha = parent["combined_node_manifest_sha256"]
    combined = ReferenceNodes.load(root / "combined", reference, manifest_sha256=combined_sha)
    mean, sd = reference_moments(reference, combined)
    if not np.array_equal(mean, arrays["mean"]) or not np.array_equal(sd, arrays["sd"]):
        raise ValueError("parent physical moments differ from retained combined nodes")
    del combined
    history = parent["refinement"]["history"]
    powers = [entry["power"] for entry in history]
    if powers != [16, 18, 20, 22][:len(powers)] or len(powers) < 2:
        raise ValueError("parent must retain at least two approved consecutive refinement powers")
    if parent["derived_seed"] != derive_seed(parent["base_seed"], "reference", case=case_index):
        raise ValueError("parent reference seed derivation mismatch")
    banks, custody = [], []
    for entry in history[-2:]:
        level = []
        for j in range(4):
            path = root / f"nodes-p{entry['power']}-s{j}"
            digest = sha256_file(path / "manifest.json")
            bank = ReferenceNodes.load(path, reference, manifest_sha256=digest)
            expected_seed = int(np.random.SeedSequence([parent["derived_seed"], entry["power"], j]).generate_state(1)[0])
            if (bank.metadata["power"] != entry["power"] or bank.metadata["seed"] != expected_seed
                    or entry["seeds"][j] != expected_seed):
                raise ValueError("retained bank power/scramble seed mismatch")
            custody.append({"directory": path.name, "manifest_sha256": digest,
                "power": entry["power"], "scramble": j, "seed": expected_seed})
            level.append(bank)
        banks.append(level)
    return arrays, parent, reference, banks, custody, manifest_sha


def refine_reference_grid(banks_root, case_index, reference_root, output_root, *,
                          max_cdf_step=.005, deadline_seconds=7200, max_rounds=12,
                          max_queries=32769, witness_batch_size=256, budget_ledger=None):
    """Save a separate immutable dense-query reference; never alter its parent.

    The accepted parent moments/independent checks are retained exactly. New CDF
    points require final-four-scramble and previous-power agreement<=.005. Limits
    and timeouts yield an inconclusive manifest, never a partial pass.
    """
    if type(case_index) is not int or not 0 <= case_index < 8:
        raise ValueError("case index must be in[0,7]")
    if not np.isfinite(deadline_seconds) or not 0 < deadline_seconds <= 7200:
        raise ValueError("deadline must be positive and no more than two hours")
    if type(witness_batch_size) is not int or witness_batch_size < 1:
        raise ValueError("positive witness batch size required")
    budget = _require_budgeted_gpu(budget_ledger)
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    deadline = started + deadline_seconds
    result = {**runtime_identity(), "schema": SCHEMA, "artifact_identity": "PosteriorForge",
              "state": "inconclusive", "converged": False, "independent_passed": False,
              "case_index": case_index, "budget_reservation": budget,
              "source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
              "reference_root": str(Path(reference_root).resolve()),
              "qualification": {"state": "inconclusive", "precision": None,
                                "grid_gate_passed": False}}
    try:
        arrays, parent, reference, banks, custody, parent_sha = _authenticate_parent(banks_root, case_index, reference_root)
        result.update(parent_manifest_sha256=parent_sha, binding=parent["binding"],
                      scientific_fingerprint=parent["binding"]["scientific_fingerprint"],
                      retained_node_manifests=custody, parent_distribution_sha256=parent["distribution"]["sha256"],
                      moment_precision=parent["moment_precision"],
                      parent_reference_qualification=parent["qualification"])
        def evaluate(queries):
            level_centers, final_spread = [], None
            for level in banks:
                values = np.asarray([bank.projected_cdfs(reference, arrays["directions"], queries,
                    batch_size=witness_batch_size, deadline=deadline) for bank in level])
                center, spread = scramble_witness_summary(level, values)
                level_centers.append(center)
                final_spread = spread
            return level_centers[-1], np.maximum(final_spread, np.abs(level_centers[-1] - level_centers[-2]))
        queries, cdfs, grid = _adaptive_grid(arrays["queries"], arrays["cdfs"],
            parent["cdf_precision"], evaluate, max_cdf_step=max_cdf_step,
            max_rounds=max_rounds, max_queries=max_queries, deadline=deadline)
        if sha256_file(Path(reference_root) / "manifest.json") != parent_sha:
            raise ValueError("parent reference manifest changed during grid refinement")
        result["grid_history"] = grid["history"]
        result["grid_contract"] = {"schema": GRID_CONTRACT_SCHEMA,
            "method": "deterministic_common_query_midpoint_bisection_by_maximum_53_reference_CDF_increment",
            "reference_only_selection": True, "fixed_direction_count": 53,
            "base_point_count": 513, "base_query_low": -8., "base_query_high": 8.,
            "base_queries_sha256": array_descriptor(BASE_QUERIES)["sha256"],
            "directions_sha256": array_descriptor(arrays["directions"])["sha256"],
            "maximum_allowed_cdf_step": max_cdf_step,
            "maximum_allowed_tail_mass": CDF_PRECISION_LIMIT,
            "maximum_allowed_reference_precision": CDF_PRECISION_LIMIT,
            "maximum_rounds": max_rounds, "maximum_queries": max_queries,
            "final_point_count": len(queries), "maximum_cdf_step": grid["maximum_cdf_step"],
            "maximum_tail_mass": grid["maximum_tail_mass"],
            "query_additions": "each round bisects every current interval whose maximum53direction reference CDF increment exceeds the fixed threshold"}
        result.update(cdf_precision=grid["cdf_precision"], reason=grid["reason"])
        payload = root / "distribution.npz"
        values = {"directions": arrays["directions"], "queries": queries,
                  "cdfs": cdfs, "mean": arrays["mean"], "sd": arrays["sd"]}
        with payload.open("xb") as handle:
            np.savez(handle, **values)
        result["distribution"] = {"file": payload.name, "sha256": sha256_file(payload),
            "arrays": {name: array_descriptor(value) for name, value in values.items()}}
        result["qualification"] = dict(parent["qualification"],
            state="qualified" if grid["passed"] else "inconclusive", precision=grid["cdf_precision"],
            grid_gate_passed=grid["passed"], grid_cdf_step_passed=grid["maximum_cdf_step"] <= max_cdf_step,
            grid_tails_passed=grid["maximum_tail_mass"] <= CDF_PRECISION_LIMIT,
            joint_reference_passed=grid["passed"] and parent["qualification"]["joint_reference_passed"])
        if grid["passed"]:
            result.update(state="verified", converged=True, independent_passed=True)
    except TimeoutError as error:
        result.update(reason=str(error), timed_out=True)
    except Exception as error:
        result.update(state="failed", reason=str(error), error_type=type(error).__name__)
        result["elapsed_seconds"] = time.monotonic() - started
        write_json(root / "manifest.json", result)
        raise
    result["elapsed_seconds"] = time.monotonic() - started
    write_json(root / "manifest.json", result)
    return result
