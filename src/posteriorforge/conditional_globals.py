"""One prior-derived correlated Gaussian refresh of ALL three global means.

Built with PriorLabs-TabPFN. Globals enter the approved model only through the
Gaussian hierarchy. Given the local physics, the likelihood and independent
noise priors cancel, so p(G | locals, noise, observation) = p(G | locals).
This separate hybrid NPE/Gibbs kernel replaces all three globals once and keeps
the18 local/noise coordinates unchanged. No observation/reference/truth input,
calibration, iteration, clipping, redraw, filtering or accuracy claim is used.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import numpy as np
from .science.analytical_benchmark_v2 import build_problem

SCHEMA = "posteriorforge.conditional_globals.v1"
CONDITIONAL_ID = "one_exact_prior_global_block_v1"
GLOBAL_NAMES = ("mu_a", "mu_b", "mu_d0")
PINNED_SCIENCE_SHA256 = {
    "analytical_benchmark_v2.py": "8d5f286f6d8512391dea2ae036313bec05782d28c7803f336490cbcdce3bd99f",
    "hierarchical_noise.py": "cbf0b8c956362a5eaa31923e9b94a3db3e3ede2072603add51e78105b8c0cbf0",
    "hierarchical_exact.py": "db001593cc7307b1290b981622705961c5953001a6c1245fae5ec779938e9687",
    "hierarchical_problem.py": "03aa2d8a3d279783cf6c599f1400f253f3db2e1e7eb027aff5b7cd1b48fe12d0",
}


def _require(value, message):
    if not value:
        raise ValueError(message)


def source_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _geometry():
    root = Path(__file__).with_name("science")
    sources = {name: hashlib.sha256((root/name).read_bytes()).hexdigest()
               for name in PINNED_SCIENCE_SHA256}
    _require(sources == PINNED_SCIENCE_SHA256, "approved scientific source changed")
    problem = build_problem()
    prior = problem.prior.base
    order = tuple(problem.parameter_order)
    _require(tuple(prior.global_names) == GLOBAL_NAMES and len(order) == 21
             and tuple(order.index(name) for name in GLOBAL_NAMES) == (0, 1, 2),
             "approved three-global physical order required")
    blocks = tuple(prior.conditional_blocks)
    observations = tuple(problem.observation_model.blocks)
    _require(len(blocks) == len(observations) == len(problem.prior.noise_names) == 6,
             "six independent local/noise blocks required")
    precision0 = np.linalg.solve(prior.global_covariance, np.eye(3))
    natural0 = precision0 @ prior.global_mean
    precision = precision0.copy()
    natural_intercept = natural0.copy()
    terms = []
    locals_ = []
    for block, observation in zip(blocks, observations):
        indices = tuple(order.index(name) for name in block.local_names)
        _require(block.dataset_id == observation.dataset_id
                 and block.local_names == observation.local_names
                 and not set(block.local_names).intersection(GLOBAL_NAMES),
                 "likelihood must refer only to its matching local physical block")
        weighted_map = np.linalg.solve(block.covariance, block.population_map)
        precision += block.population_map.T @ weighted_map
        natural_intercept -= block.offset @ weighted_map
        terms.append((indices, weighted_map))
        locals_.extend(indices)
    noise_indices = tuple(order.index(name) for name in problem.prior.noise_names)
    _require(len(locals_) == len(set(locals_)) == 12
             and set((0, 1, 2) + tuple(locals_) + noise_indices) == set(range(21)),
             "global/local/independent-noise coordinate partition changed")
    _require(np.isfinite(precision).all() and np.isfinite(natural_intercept).all()
             and np.allclose(precision, precision.T, rtol=0, atol=1e-14),
             "finite symmetric Gaussian precision required")
    np.linalg.cholesky(precision)
    covariance = np.linalg.solve(precision, np.eye(3))
    # Symmetric Float64 representation of the real-arithmetic inverse.
    covariance = (covariance + covariance.T) / 2.
    factor = np.linalg.cholesky(covariance)
    coefficients = np.zeros((21, 3), dtype=np.float64)
    for indices, weighted_map in terms:
        coefficients[list(indices)] += weighted_map @ covariance
    intercept = natural_intercept @ covariance
    return {"problem": problem, "physical_order": order, "local_indices": tuple(locals_),
            "noise_indices": noise_indices, "prior_precision": precision0,
            "prior_natural": natural0, "precision": precision,
            "covariance": covariance, "factor": factor,
            "intercept": intercept, "coefficients": coefficients,
            "science_sources": sources}


def _values(values, geometry):
    _require(isinstance(values, np.ndarray) and values.dtype == np.dtype(np.float64),
             "original float64 physical array required; no implicit cast")
    _require(values.ndim == 2 and values.shape[1] == 21 and len(values) > 0,
             "nonempty complete physical samples[n,21] required")
    _require(np.isfinite(values).all(), "nonfinite rows cannot be refreshed")
    _require(geometry["problem"].contains(values).all(),
             "all input rows must already be in original physical prior support")
    return values


def conditional_parameters(values):
    """Return mean[n,3], covariance[3,3] for G | local physics, in Float64.

    Positive n is available for tests; the scientific producer must enforce50000.
    Fixed per-row reductions do not depend on batch size or input memory layout.
    """
    geometry = _geometry()
    rows = _values(values, geometry)
    mean = np.repeat(geometry["intercept"][None, :], len(rows), axis=0)
    for index in geometry["local_indices"]:
        mean += rows[:, index, None] * geometry["coefficients"][index]
    _require(np.isfinite(mean).all(), "nonfinite conditional means; no repair allowed")
    return mean, geometry["covariance"].copy()


def refresh(values, rng):
    """Draw a single correlated block using Z[n,3] @ chol(covariance).T once."""
    _require(isinstance(rng, np.random.Generator), "explicit caller-owned NumPy Generator required")
    mean, covariance = conditional_parameters(values)
    factor = np.linalg.cholesky(covariance)
    draws = mean + rng.standard_normal((len(values), 3)) @ factor.T
    _require(np.isfinite(draws).all(), "nonfinite Gaussian draw; no redraw or repair allowed")
    output = np.array(values, copy=True, order="C")
    output[:, :3] = draws
    return output


def metadata():
    geometry = _geometry()
    return {
        "schema": SCHEMA, "conditional_id": CONDITIONAL_ID,
        "module_sha256": source_sha256(), "attribution": "Built with PriorLabs-TabPFN",
        "scientific_fingerprint": geometry["problem"].scientific_fingerprint,
        "scientific_sources_sha256": geometry["science_sources"],
        "physical_parameter_order": list(geometry["physical_order"]),
        "global_names": list(GLOBAL_NAMES), "target_indices": [0, 1, 2],
        "unchanged_indices": list(range(3, 21)),
        "local_physics_indices": list(geometry["local_indices"]),
        "noise_indices": list(geometry["noise_indices"]),
        "global_prior_precision": geometry["prior_precision"].tolist(),
        "global_prior_natural": geometry["prior_natural"].tolist(),
        "global_fullconditional_precision": geometry["precision"].tolist(),
        "conditional_covariance": geometry["covariance"].tolist(),
        "conditional_covariance_lower_cholesky": geometry["factor"].tolist(),
        "conditional_mean_intercept": geometry["intercept"].tolist(),
        "conditional_mean_coefficients_physical_order": geometry["coefficients"].tolist(),
        "conditional": "p(G | local physics, independent noise, observation) = p(G | local physics)",
        "mean_derivation": "P^-1 eta from actual Gaussian prior and conditional blocks; no rounded coefficients",
        "normal_transform": "one row-major Z[n,3] standard_normal draw; Z @ chol(covariance).T",
        "input_output_dtype": "original float64; no implicit conversion",
        "noise_likelihood_cancel": True, "all_globals_unbounded": True,
        "sweeps": 1, "scalar_calibration": False, "target_or_feature_order_changed": False,
        "observation_reference_truth_arguments": False,
        "clipping_redraw_filtering_or_repair": False,
        "exactness_scope": "real-arithmetic prior-derived Gaussian conditional; NumPy Float64 execution",
        "joint_KL_identity_scope": "ideal distributions, when KL exists: KL(q_local * p(G|local) || p) = KL(q_local || p_local)",
        "local_marginals_corrected": False, "finite_cohort_gate_improvement_guaranteed": False,
        "posterior_accuracy_guaranteed": False, "scientific_count_required_of_producer": 50000,
    }
