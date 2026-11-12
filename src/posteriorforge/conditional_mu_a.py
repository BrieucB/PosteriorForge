"""One exact prior-derived Gaussian full-conditional refresh of physical mu_a.

Built with PriorLabs-TabPFN. This is a separate hybrid NPE/Gibbs candidate:
given complete supported physical rows, draw mu_a | other20 once. Observations,
reference distributions and truth labels are not inputs. The frozen scientific
model puts globals only in the Gaussian hierarchy, so the likelihood and six
independent noise priors cancel from this conditional. Exactness describes the
real-arithmetic conditional; evaluation and normal draws use NumPy float64.

Refreshing an approximate joint need not repair its other20 marginal law or
pass any posterior accuracy gate. No row is filtered, clipped or repaired.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .science.analytical_benchmark_v2 import build_problem

SCHEMA = "posteriorforge.conditional_mu_a.v1"
CONDITIONAL_ID = "one_exact_prior_mu_a_fullconditional_v1"
TARGET = "mu_a"
PINNED_SCIENCE_SHA256 = {
    "analytical_benchmark_v2.py": "8d5f286f6d8512391dea2ae036313bec05782d28c7803f336490cbcdce3bd99f",
    "hierarchical_noise.py": "cbf0b8c956362a5eaa31923e9b94a3db3e3ede2072603add51e78105b8c0cbf0",
    "hierarchical_exact.py": "db001593cc7307b1290b981622705961c5953001a6c1245fae5ec779938e9687",
    "hierarchical_problem.py": "03aa2d8a3d279783cf6c599f1400f253f3db2e1e7eb027aff5b7cd1b48fe12d0",
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def source_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _geometry():
    """Derive P,h from the actual Gaussian hierarchy, never rounded coefficients."""
    root = Path(__file__).with_name("science")
    actual = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
              for name in PINNED_SCIENCE_SHA256}
    _require(actual == PINNED_SCIENCE_SHA256, "approved scientific source changed")
    problem = build_problem()
    prior = problem.prior.base
    order = tuple(problem.parameter_order)
    globals_ = tuple(prior.global_names)
    _require(globals_ == ("mu_a", "mu_b", "mu_d0") and len(order) == 21,
             "approved three-global/21-physical-coordinate geometry required")
    global_indices = tuple(order.index(name) for name in globals_)
    _require(global_indices == (0, 1, 2), "approved physical global order required")
    blocks = tuple(prior.conditional_blocks)
    observations = tuple(problem.observation_model.blocks)
    _require(len(blocks) == len(observations) == len(problem.prior.noise_names) == 6,
             "six independent local/noise blocks required")
    used = []
    terms = []
    prior_precision = np.linalg.solve(prior.global_covariance, np.eye(3))
    natural = prior_precision @ prior.global_mean
    precision = prior_precision.copy()
    for block, observation in zip(blocks, observations):
        indices = tuple(order.index(name) for name in block.local_names)
        _require(block.dataset_id == observation.dataset_id
                 and block.local_names == observation.local_names
                 and not set(block.local_names).intersection(globals_),
                 "likelihood may use only its matching local physical block")
        _require(block.population_map.shape == (len(indices), 3)
                 and block.offset.shape == (len(indices),)
                 and block.covariance.shape == (len(indices), len(indices)),
                 "Gaussian conditional block dimensions changed")
        weighted_map = np.linalg.solve(block.covariance, block.population_map)
        precision += block.population_map.T @ weighted_map
        terms.append((indices, block.offset, weighted_map))
        used.extend(indices)
    noise_indices = tuple(order.index(name) for name in problem.prior.noise_names)
    _require(len(used) == len(set(used)) == 12
             and set(global_indices + tuple(used) + noise_indices) == set(range(21)),
             "local/global/independent-noise coordinate partition changed")
    _require(np.isfinite(precision).all() and np.isfinite(natural).all()
             and np.allclose(precision, precision.T, rtol=0, atol=1e-14),
             "finite symmetric Gaussian precision required")
    np.linalg.cholesky(precision)
    target = globals_.index(TARGET)
    coefficients = np.zeros(21, dtype=np.float64)
    for indices, _, weighted_map in terms:
        coefficients[list(indices)] += weighted_map[:, target] / precision[target, target]
    for k, physical in enumerate(global_indices):
        if k != target:
            coefficients[physical] -= precision[target, k] / precision[target, target]
    intercept = natural[target] / precision[target, target]
    for _, offset, weighted_map in terms:
        intercept -= (offset @ weighted_map)[target] / precision[target, target]
    return {
        "problem": problem, "physical_order": order, "global_order": globals_,
        "global_indices": global_indices, "target_index": order.index(TARGET),
        "noise_indices": noise_indices, "precision": precision,
        "prior_precision": prior_precision, "prior_natural": natural,
        "coefficients": coefficients, "intercept": float(intercept),
        "variance": float(1. / precision[target, target]), "science_sources": actual,
    }


def _values(values, geometry):
    _require(isinstance(values, np.ndarray) and values.dtype == np.dtype(np.float64),
             "original float64 physical sample array required; no implicit cast")
    _require(values.ndim == 2 and values.shape[1] == 21 and len(values) > 0,
             "nonempty complete physical samples[n,21] required")
    _require(np.isfinite(values).all(), "nonfinite input rows cannot be refreshed")
    _require(geometry["problem"].contains(values).all(),
             "all input rows must already lie in original physical prior support")
    return values


def conditional_parameters(values):
    """Return (mean[n], variance) for mu_a | other20, both evaluated in float64.

    No observation or reference is needed. Positive n is allowed for algebra
    tests; a scientific producer must independently enforce n == 50000.
    """
    geometry = _geometry()
    rows = _values(values, geometry)
    mean = (np.sum(rows * geometry["coefficients"], axis=1, dtype=np.float64)
            + geometry["intercept"])
    _require(np.isfinite(mean).all() and np.isfinite(geometry["variance"])
             and geometry["variance"] > 0, "nonfinite conditional parameters")
    return mean, geometry["variance"]


def refresh(values, rng):
    """Return a new array after ONE mu_a draw per row using the supplied Generator.

    The input and other20 logical columns retain their exact float64 bytes.
    Only this explicit Generator advances; NumPy's default/global RNG is unused.
    """
    _require(isinstance(rng, np.random.Generator),
             "an explicit caller-owned numpy.random.Generator is required")
    mean, variance = conditional_parameters(values)
    output = np.array(values, copy=True, order="C")
    draws = mean + np.sqrt(variance) * rng.standard_normal(len(values))
    _require(np.isfinite(draws).all(), "nonfinite Gaussian draw; no repair is permitted")
    output[:, 0] = draws
    return output


def metadata():
    """Source/geometry identity; not a claim of posterior accuracy or calibration."""
    geometry = _geometry()
    return {
        "schema": SCHEMA, "conditional_id": CONDITIONAL_ID,
        "module_sha256": source_sha256(), "attribution": "Built with PriorLabs-TabPFN",
        "scientific_fingerprint": geometry["problem"].scientific_fingerprint,
        "scientific_sources_sha256": geometry["science_sources"],
        "physical_parameter_order": list(geometry["physical_order"]),
        "global_order": list(geometry["global_order"]),
        "global_indices": list(geometry["global_indices"]),
        "target_name": TARGET, "target_index": geometry["target_index"],
        "unchanged_indices": [i for i in range(21) if i != geometry["target_index"]],
        "noise_indices": list(geometry["noise_indices"]),
        "global_prior_precision": geometry["prior_precision"].tolist(),
        "global_prior_natural": geometry["prior_natural"].tolist(),
        "global_fullconditional_precision": geometry["precision"].tolist(),
        "conditional_mean_intercept": geometry["intercept"],
        "conditional_mean_coefficients_physical_order": geometry["coefficients"].tolist(),
        "conditional_variance": geometry["variance"],
        "conditional": "p(mu_a | other20, observation) = p(mu_a | other20)",
        "noise_and_likelihood_cancel": True,
        "input_output_dtype": "float64; no implicit conversion",
        "rng": "one standard_normal draw per row from caller-owned numpy.random.Generator",
        "sweeps": 1, "target_order_changed": False, "feature_order_changed": False,
        "observation_reference_truth_arguments": False, "scalar_calibration": False,
        "clipping_filtering_or_repair": False,
        "exactness_scope": "prior-derived Gaussian full conditional in real arithmetic; NumPy float64 execution",
        "posterior_accuracy_guaranteed": False, "scientific_count_enforced_by_producer": 50000,
    }
