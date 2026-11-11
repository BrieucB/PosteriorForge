"""Retained float64 reference refinement with independent order checks.

Production reference calculations require a Slurm allocation. Their allocated
GPU-hours are charged by the caller's budget ledger, including CPU-side nested
quadrature while a GPU allocation is held.
"""
from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import time

import numpy as np

from .banks import derive_seed, load_heldout
from .science.analytical_benchmark_v2 import array_descriptor, build_problem, unflatten_observations
from .science.analytical_benchmark_reference_v2 import (
    FullHierarchyReference, ReferenceNodes, compare_nested, nested_global_noise_check,
    refine_reference, refinement_witnesses, scramble_witness_summary,
)

SCHEMA = "posteriorforge.reference.v1"
CDF_TOLERANCE = .005
MOMENT_RELATIVE_TOLERANCE = .005
DEFAULT_NESTED_CORNERS = ((20, 128), (40, 64), (40, 128))


def _json(path, value):
    with Path(path).open("w") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def reference_moments(reference, nodes):
    """Exact mixture first/second marginal moments in declared parameter order."""
    nodes.authenticate(reference)
    mean, variance = np.empty(21), np.empty(21)
    ppos = [reference.problem.parameter_order.index(name) for name in reference.names]
    npos = [reference.problem.parameter_order.index(name) for name in reference.problem.prior.noise_names]
    mean[ppos] = nodes.metadata["physics_mean"]
    variance[ppos] = np.diag(nodes.metadata["physics_covariance"])
    mean[npos] = nodes.metadata["noise_mean"]
    variance[npos] = np.asarray(nodes.metadata["noise_second_moment"]) - mean[npos] ** 2
    if not np.isfinite(mean).all() or not np.isfinite(variance).all() or np.any(variance <= 0):
        raise ValueError("nonpositive/nonfinite posterior marginal variance")
    return mean, np.sqrt(variance)


def _save_node(path, nodes):
    manifest = nodes.save(path)
    # Mathematical metadata retains its source schema; artifact ownership is new.
    manifest.update(schema=SCHEMA + ".nodes", artifact_identity="PosteriorForge")
    _json(Path(path) / "manifest.json", manifest)
    return manifest


def _load_node(path, reference):
    digest = sha256((Path(path) / "manifest.json").read_bytes()).hexdigest()
    return ReferenceNodes.load(path, reference, manifest_sha256=digest)


def _dense_witnesses(reference, root, history, *, points, batch_size, deadline):
    """Recheck all 53 directions at a denser retained two-level grid.

    A finite witness grid is a numerical diagnostic, not a universal proof of
    convergence for arbitrary multivariate functionals. The validation layer
    conservatively bounds candidate CDF differences between these grid points.
    """
    if type(points) is not int or points < 129:
        raise ValueError("at least129 dense witness points required")
    directions, _ = refinement_witnesses(reference)
    queries = np.linspace(-8., 8., points)
    centers, spreads, retained_moments = [], [], []
    if len(history) < 2:
        raise ValueError("two retained refinement levels required")
    for entry in history[-2:]:
        banks = [_load_node(root / f"nodes-p{entry['power']}-s{j}", reference) for j in range(4)]
        moments = [reference_moments(reference, bank) for bank in banks]
        evidence = np.asarray([bank.metadata["log_evidence"] for bank in banks])
        evidence_weight = np.exp(evidence - evidence.max())
        evidence_weight /= evidence_weight.sum()
        means = np.asarray([moment[0] for moment in moments])
        sds = np.asarray([moment[1] for moment in moments])
        combined_mean = evidence_weight @ means
        combined_variance = evidence_weight @ (sds * sds + means * means) - combined_mean * combined_mean
        if np.any(combined_variance <= 0):
            raise ValueError("nonpositive retained mixture variance")
        retained_moments.append({"power": entry["power"], "scramble_means": means,
            "scramble_sds": sds, "combined_mean": combined_mean,
            "combined_sd": np.sqrt(combined_variance)})
        values = np.asarray([bank.projected_cdfs(reference, directions, queries,
                            batch_size=batch_size, deadline=deadline) for bank in banks])
        center, spread = scramble_witness_summary(banks, values)
        centers.append(center)
        spreads.append(spread)
    delta = np.abs(centers[-1] - centers[-2])
    error = np.maximum(delta, spreads[-1])
    result = {"points": points, "query_low": -8., "query_high": 8.,
              "compared_powers": [h["power"] for h in history[-2:]],
              "marginal_cdf_precision": float(error[:21].max()),
              "joint_projection_cdf_precision": float(error[21:].max()),
              "passed": bool(np.max(error) <= CDF_TOLERANCE),
              "contract": "maximum final scramble spread and consecutive-power CDF difference at retained witness grid"}
    return directions, queries, centers[-1], result, retained_moments



def _moment_precision(reference, nodes, retained_moments, nested_checks):
    """Retained all-coordinate moment convergence, including independent corners.

    Errors are normalized by the final mixture SD. They are empirical numerical
    uncertainty diagnostics under the declared retained-refinement contract.
    No moment check changes the frozen quadrature powers or observed cases.
    """
    if len(retained_moments) != 2 or len(nested_checks) < 3:
        raise ValueError("two retained levels and three independent nested corners required")
    mean, sd = reference_moments(reference, nodes)
    previous, final = retained_moments
    mean_errors = [np.abs(np.asarray(final["scramble_means"]) - mean).max(axis=0),
                   np.abs(np.asarray(previous["combined_mean"]) - mean)]
    sd_errors = [np.abs(np.asarray(final["scramble_sds"]) - sd).max(axis=0),
                 np.abs(np.asarray(previous["combined_sd"]) - sd)]
    mean_error = np.max(mean_errors, axis=0)
    sd_error = np.max(sd_errors, axis=0)
    ppos = [reference.problem.parameter_order.index(name) for name in reference.names[:3]]
    npos = [reference.problem.parameter_order.index(name) for name in reference.problem.prior.noise_names]
    # Source compare_nested already verifies separate-axis corners and binding.
    nested_final = nested_checks[-1]
    nested_global_mean = np.asarray(nested_final["global_mean"], dtype=np.float64)
    nested_variance = np.diag(np.asarray(nested_final["global_covariance"], dtype=np.float64))
    if np.any(nested_variance <= 0):
        raise ValueError("nonpositive independent global posterior variance")
    nested_global_sd = np.sqrt(nested_variance)
    independent_global_mean = np.abs(nested_global_mean - mean[ppos])
    independent_global_sd = np.abs(nested_global_sd - sd[ppos])
    independent_noise_mean = np.abs(np.asarray(nested_final["noise_mean"]) - mean[npos])
    for corner in nested_checks[:-1]:
        variance = np.diag(np.asarray(corner["global_covariance"], dtype=np.float64))
        if np.any(variance <= 0):
            raise ValueError("nonpositive independent-corner global posterior variance")
        independent_global_mean = np.maximum(independent_global_mean,
            np.abs(np.asarray(corner["global_mean"]) - nested_global_mean))
        independent_global_sd = np.maximum(independent_global_sd,
            np.abs(np.sqrt(variance) - nested_global_sd))
        independent_noise_mean = np.maximum(independent_noise_mean,
            np.abs(np.asarray(corner["noise_mean"]) - nested_final["noise_mean"]))
    mean_error[ppos] = np.maximum(mean_error[ppos], independent_global_mean)
    sd_error[ppos] = np.maximum(sd_error[ppos], independent_global_sd)
    mean_error[npos] = np.maximum(mean_error[npos], independent_noise_mean)
    mean_relative, sd_relative = mean_error / sd, sd_error / sd
    passed = (np.isfinite(mean_relative).all() and np.isfinite(sd_relative).all()
              and mean_relative.max() <= MOMENT_RELATIVE_TOLERANCE
              and sd_relative.max() <= MOMENT_RELATIVE_TOLERANCE)
    return {"passed": bool(passed), "threshold_reference_sd": MOMENT_RELATIVE_TOLERANCE,
            "mean_error_reference_sd": mean_relative.tolist(),
            "sd_error_reference_sd": sd_relative.tolist(),
            "mean_error_physical": mean_error.tolist(), "sd_error_physical": sd_error.tolist(),
            "max_mean_error_reference_sd": float(mean_relative.max()),
            "max_sd_error_reference_sd": float(sd_relative.max()),
            "compared_powers": [entry["power"] for entry in retained_moments],
            "checks": ["final_four_scramble_moments", "previous_power_combined_moments",
                       "independent_global_mean_and_sd", "independent_noise_mean",
                       "separate_axis_independent_corner_changes"],
            "contract": "retained empirical numerical convergence diagnostics; validation propagates these uncertainties"}


def _prepare_reference(banks_root, case_index: int, output_root, *, backend="jax", device="gpu",
                      seed=20261114, batch_size=512, witness_batch_size=256,
                      dense_points=513, nested_corners=DEFAULT_NESTED_CORNERS,
                      deadline_seconds=7200) -> dict:
    """Compute/refine one reusable reference; never promote exhaustion to pass.

    Cached authenticated scramble banks are reusable after interruption. A
    timeout produces an inconclusive public manifest and is never a pass.
    """
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("production references require an accounted Slurm GPU allocation")
    if type(case_index) is not int or not 0 <= case_index < 8:
        raise ValueError("case index must be in[0,7]")
    if deadline_seconds <= 0:
        raise ValueError("positive deadline required")
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "manifest.json").exists():
        raise FileExistsError("reference final manifest exists; use a new attempt directory")
    started = time.monotonic()
    deadline = started + deadline_seconds
    heldout = load_heldout(banks_root, case_index)
    observed = unflatten_observations(heldout.obs)
    reference = FullHierarchyReference(build_problem(), observed, backend=backend, device=device)
    case_seed = derive_seed(seed, "reference", case=case_index)
    result = {"schema": SCHEMA, "artifact_identity": "PosteriorForge",
              "case_index": case_index, "binding": reference.binding,
              "base_seed": seed, "derived_seed": case_seed,
              "backend": backend, "device": device,
              "state": "inconclusive", "converged": False, "independent_passed": False,
              "cdf_tolerance": CDF_TOLERANCE,
              "slurm_job_id": os.environ["SLURM_JOB_ID"],
              "qualification": {"witness_passed": False, "dense_witness_passed": False,
                                "independent_nested_passed": False, "moment_gate_passed": False},
              "reference_precision_contract": "finite-grid retained scramble/refinement diagnostic plus independent nested-order check; not global calibration certification"}
    def bank_path(power, j):
        return root / f"nodes-p{power}-s{j}"
    def provider(power, j, scramble_seed):
        path = bank_path(power, j)
        return _load_node(path, reference) if path.exists() else None
    def retain(bank, power, j):
        path = bank_path(power, j)
        if not path.exists():
            _save_node(path, bank)
    try:
        nodes, refinement = refine_reference(reference, marginal_resolution=.05,
                    joint_resolution=.05, seed=case_seed, batch_size=batch_size,
                    witness_batch_size=witness_batch_size, deadline=deadline,
                    bank_provider=provider, bank_callback=retain)
        result["refinement"] = refinement
        _json(root / "refinement.json", refinement)
        result["qualification"]["witness_passed"] = refinement["witness_gate_passed"]
        if not refinement["witness_gate_passed"]:
            result["reason"] = "retained four-scramble reference exhausted approved powers without adequate CDF precision"
        else:
            # Dense retained checks protect table-based candidate decisions.
            directions, queries, cdfs, dense, retained_moments = _dense_witnesses(reference, root,
                        refinement["history"], points=dense_points,
                        batch_size=witness_batch_size, deadline=deadline)
            result["dense_witness"] = dense
            result["qualification"]["dense_witness_passed"] = dense["passed"]
            mean, sd = reference_moments(reference, nodes)
            payload = root / "distribution.npz"
            with payload.open("xb") as handle:
                np.savez(handle, directions=directions, queries=queries, cdfs=cdfs,
                         mean=mean, sd=sd)
            result["distribution"] = {"file": payload.name,
                "sha256": sha256(payload.read_bytes()).hexdigest(),
                "arrays": {name: array_descriptor(value) for name, value in
                           {"directions": directions, "queries": queries, "cdfs": cdfs,
                            "mean": mean, "sd": sd}.items()}}
            checks = []
            for global_order, noise_order in nested_corners:
                path = root / f"nested-g{global_order}-n{noise_order}.json"
                if path.exists():
                    check = json.loads(path.read_text())
                else:
                    check = nested_global_noise_check(reference, global_order=global_order,
                                noise_order=noise_order, batch_size=128, deadline=deadline)
                    _json(path, check)
                checks.append(check)
            independent = compare_nested(reference, nodes, checks,
                         cdf_tolerance=CDF_TOLERANCE, log_evidence_tolerance=.01,
                         global_mean_tolerance=.001, global_covariance_tolerance=.001,
                         noise_mean_tolerance=.001)
            result["independent_nested"] = independent
            result["qualification"]["independent_nested_passed"] = independent["passed"]
            moment_precision = _moment_precision(reference, nodes, retained_moments, checks)
            result["moment_precision"] = moment_precision
            result["qualification"]["moment_gate_passed"] = moment_precision["passed"]
            result["cdf_precision"] = max(dense["marginal_cdf_precision"],
                     dense["joint_projection_cdf_precision"], independent["noise_cdf_max_error"])
            if all(result["qualification"].values()):
                result.update(state="verified", converged=True, independent_passed=True)
            else:
                result["reason"] = "reference numerical checks remain inconclusive"
        final_path = root / "combined"
        if not final_path.exists():
            _save_node(final_path, nodes)
        result["combined_node_manifest_sha256"] = sha256((final_path / "manifest.json").read_bytes()).hexdigest()
    except TimeoutError as error:
        result["reason"] = str(error)
        result["timed_out"] = True
    result["qualification"].update(
        state="qualified" if result["state"] == "verified" else "inconclusive",
        precision=result.get("cdf_precision"),
        witness_gate_passed=result["qualification"]["witness_passed"],
        independent_nested_passed=result["qualification"]["independent_nested_passed"],
        cpu_check_passed=result.get("refinement", {}).get("cpu_check", {}).get("passed", False),
        joint_reference_passed=result["qualification"]["dense_witness_passed"],
        moment_precision=result.get("moment_precision"),
    )
    result["elapsed_seconds"] = time.monotonic() - started
    _json(root / "manifest.json", result)
    return result


def load_reference_distribution(root):
    """Return authenticated dense table arrays and its qualification manifest."""
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    entry = manifest["distribution"]
    path = root / entry["file"]
    if entry["file"] != "distribution.npz" or sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
        raise ValueError("reference table payload hash mismatch")
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != {"directions", "queries", "cdfs", "mean", "sd"}:
            raise ValueError("unexpected reference table arrays")
        arrays = {name: payload[name].copy() for name in payload.files}
    if any(array_descriptor(value) != entry["arrays"][name] for name, value in arrays.items()):
        raise ValueError("reference table array hash mismatch")
    return arrays, manifest


def prepare_reference(banks_root, case_index: int, output_root, **kwargs) -> dict:
    """Run retained refinement, preserving a failed manifest on unexpected errors."""
    started = time.monotonic()
    try:
        return _prepare_reference(banks_root, case_index, output_root, **kwargs)
    except Exception as error:
        # Existing completed evidence is immutable. Record new initialization or
        # runtime failures without fabricating convergence or qualification.
        root = Path(output_root)
        if os.environ.get("SLURM_JOB_ID") and not (root / "manifest.json").exists():
            root.mkdir(parents=True, exist_ok=True)
            _json(root / "manifest.json", {"schema": SCHEMA,
                "artifact_identity": "PosteriorForge", "state": "failed",
                "converged": False, "independent_passed": False,
                "case_index": case_index, "error_type": type(error).__name__,
                "reason": str(error), "elapsed_seconds": time.monotonic() - started,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "qualification": {"state": "failed", "precision": None,
                    "witness_gate_passed": False, "independent_nested_passed": False,
                    "cpu_check_passed": False, "joint_reference_passed": False,
                    "moment_gate_passed": False}})
        raise


def qualify_reference(banks_root, case_index: int, output_root, *, atol=1e-9, rtol=1e-9) -> dict:
    """Small real GPU float64 qualification against two independent CPU formulas."""
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("GPU reference qualification requires an accounted Slurm allocation")
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = {"schema": SCHEMA + ".qualification", "artifact_identity": "PosteriorForge",
              "case_index": case_index, "slurm_job_id": os.environ["SLURM_JOB_ID"],
              "state": "failed", "passed": False, "atol": atol, "rtol": rtol}
    try:
        heldout = load_heldout(banks_root, case_index)
        problem = build_problem()
        observation = unflatten_observations(heldout.obs)
        cpu = FullHierarchyReference(problem, observation, backend="numpy", device="cpu")
        gpu = FullHierarchyReference(problem, observation, backend="jax", device="gpu")
        sigma = np.asarray([[.08] * 6, [.15] * 6, [.255] * 6, [.49] * 6,
                            [.1, .15, .2, .25, .35, .45]], dtype=np.float64)
        actual = gpu.conditional(sigma)
        checks = []
        for name, expected in (("numpy_precision_space", cpu.conditional(sigma)),
                               ("numpy_observation_covariance_space", cpu.covariance_space_conditional(sigma))):
            checks.append({"formula": name,
                "float64": all(value.dtype == np.float64 for value in actual),
                "mean_covariance_log_evidence_max_errors": [float(np.max(np.abs(a - b))) for a, b in zip(actual, expected)],
                "passed": bool(all(np.allclose(a, b, atol=atol, rtol=rtol) for a, b in zip(actual, expected)))})
        result.update(binding=gpu.binding, checks=checks, sigma=sigma.tolist(),
                      gpu_device=str(gpu._jax_device),
                      passed=all(check["passed"] and check["float64"] for check in checks))
        result["state"] = "qualified" if result["passed"] else "failed"
    except Exception as error:
        result.update(error_type=type(error).__name__, reason=str(error))
        raise
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        _json(root / "manifest.json", result)
    return result
