"""Frozen quadratic/scalar hierarchy; independent, indexed prior-predictive streams.

The central-world filter changes only the held-out population.  The problem's
inference prior, normalization stream and validation stream retain full support.
"""
from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm, truncnorm

from .hierarchical_noise import build_inferred_noise_problem
from .hierarchical_problem import HierarchicalObservations

SCHEMA = "bayesarena.analytical_benchmark_v2.v1"
SEED = 20261016
STREAM_BLOCK_SIZE = 512
DATASET_IDS = ("mechanical_1", "mechanical_2", "mechanical_3",
               "acoustic_1", "acoustic_2", "acoustic_3")
OBSERVATION_COUNTS = (10, 10, 10, 1, 1, 1)
ROLE_COUNTS = {"normalization": 131072, "validation": 65536,
               "test": 256, "diagnostic": 8}
APPROVED_CONTRACT_SHA256 = "0eba0da9fec78f270a7d3c376f6168f5541ba2524304d2a7cadb8ac5f1eb6fc7"


def build_problem():
    """Return the unchanged 21D prior with quadratic mechanics and scalar acoustics."""
    old = build_inferred_noise_problem(noise_gaussian_priors=((.255, .1225),) * 6)
    x = np.linspace(0., 1., 10)
    design = np.column_stack((x, x * x, np.ones_like(x)))
    datasets, blocks = [], []
    for index, (dataset, block) in enumerate(zip(old.datasets, old.observation_model.blocks)):
        if index < 3:
            dataset = replace(dataset, observation_size=10, controls=tuple(x))
            block = replace(block, design=design, offset=np.zeros(10), covariance=np.eye(10))
        datasets.append(dataset)
        blocks.append(block)
    forward = replace(old.observation_model.base, blocks=tuple(blocks))
    problem = replace(old, name="quadratic_scalar_gaussian_hierarchy_v2", version=SCHEMA,
                      datasets=tuple(datasets),
                      observation_model=replace(old.observation_model, base=forward))
    _verify_prior(problem)
    return problem


def _verify_prior(problem):
    """Fail on drift of the approved numerical Gaussian hierarchy or SD priors."""
    base = problem.prior.base
    expected_covariance = np.asarray([[1., .12, .02], [.12, 1.2, -.03], [.02, -.03, .4]])
    if (base.global_names != ("mu_a", "mu_b", "mu_d0")
            or not np.array_equal(base.global_mean, [.3, -.2, .1])
            or not np.array_equal(base.global_covariance, expected_covariance)
            or problem.prior.bounds != (.01, .5)
            or problem.prior.noise_gaussian_priors != ((.255, .1225),) * 6):
        raise ValueError("approved numerical prior drift")
    for i, block in enumerate(base.conditional_blocks):
        mapping = np.eye(3) if i < 3 else np.asarray([[1., 0., 0.]])
        covariance = np.diag(np.square([.45, .70, .25])) if i < 3 else np.asarray([[.45**2]])
        if (not np.array_equal(block.population_map, mapping)
                or not np.array_equal(block.covariance, covariance)
                or not np.array_equal(block.offset, np.zeros(len(block.local_names)))):
            raise ValueError("approved conditional Gaussian prior drift")
    return {"status": "exact_numeric_equality_passed", "global_mean": base.global_mean.tolist(),
            "global_covariance": base.global_covariance.tolist(),
            "conditional_blocks": [block.manifest() for block in base.conditional_blocks],
            "noise_bounds": [.01, .5], "noise_underlying_mean": .255,
            "noise_underlying_sd": .1225, "noise_independent_coordinates": 6}


def flatten_observations(observations):
    """Flatten in the frozen six-dataset order, always retaining float64."""
    if not isinstance(observations, HierarchicalObservations):
        raise TypeError("expected HierarchicalObservations")
    if set(observations.values) != set(DATASET_IDS):
        raise ValueError("observation dataset identity mismatch")
    for name, count in zip(DATASET_IDS, OBSERVATION_COUNTS):
        if observations.values[name].shape[1] != count:
            raise ValueError("observation point count mismatch")
    return np.concatenate([observations.values[name] for name in DATASET_IDS], axis=1)


def unflatten_observations(flat):
    values = np.asarray(flat, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 33 or len(values) < 1:
        raise ValueError("flat observations must have shape [n,33]")
    offsets = np.r_[0, np.cumsum(OBSERVATION_COUNTS)]
    return HierarchicalObservations({name: values[:, offsets[i]:offsets[i + 1]]
                                    for i, name in enumerate(DATASET_IDS)})


def _rng(seed, role, index, purpose):
    if not isinstance(role, str) or not role.strip():
        raise ValueError("role must be a nonempty stream name")
    if not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    tag = np.frombuffer(sha256((SCHEMA + ":" + role).encode()).digest(), dtype="<u4")
    return np.random.default_rng(np.random.SeedSequence([int(seed), *map(int, tag), index, purpose]))


def central_world_mask(parameters, problem=None):
    """Whole-world central 90% coordinate rule, with conditional local residuals."""
    problem = build_problem() if problem is None else problem
    theta = np.asarray(parameters, dtype=np.float64)
    if theta.ndim != 2 or theta.shape[1] != 21:
        raise ValueError("parameters must have shape [n,21]")
    prior = problem.prior.base
    position = {name: i for i, name in enumerate(problem.parameter_order)}
    h = theta[:, [position[name] for name in prior.global_names]]
    z = (h - prior.global_mean) / np.sqrt(np.diag(prior.global_covariance))
    limit = norm.ppf(.95)
    keep = np.isfinite(theta).all(axis=1) & (np.abs(z) <= limit).all(axis=1)
    for block in prior.conditional_blocks:
        local = theta[:, [position[name] for name in block.local_names]]
        residual = local - h @ block.population_map.T - block.offset
        keep &= (np.abs(residual / np.sqrt(np.diag(block.covariance))) <= limit).all(axis=1)
    low, high = problem.prior.bounds
    bounds = truncnorm.ppf([.05, .95], (low - .255) / .1225, (high - .255) / .1225,
                          loc=.255, scale=.1225)
    noise = theta[:, [position[name] for name in problem.prior.noise_names]]
    return keep & ((noise >= bounds[0]) & (noise <= bounds[1])).all(axis=1)


def draw_worlds(size, *, role, seed=SEED, start=0, central=False):
    """Draw complete worlds with exact random access and batching invariance.

    Full-prior streams use fixed 512-world blocks.  Central streams use an
    independent rejection stream per accepted world, rejecting the entire
    correlated prior draw; observation errors are sampled only after acceptance.
    These algorithms and namespaces are part of the frozen stream definition.
    """
    if type(size) is not int or size < 1 or type(start) is not int or start < 0:
        raise ValueError("size must be positive and start nonnegative integers")
    if type(central) is not bool:
        raise TypeError("central must be boolean")
    problem = build_problem()
    theta_parts, obs_parts = [], []
    if central:
        for index in range(start, start + size):
            rng = _rng(seed, role, index, 2)
            while True:
                candidates = problem.sample_prior(32, rng)
                accepted = np.flatnonzero(central_world_mask(candidates, problem))
                if len(accepted):
                    theta = candidates[accepted[:1]]
                    break
            theta_parts.append(theta)
            obs_parts.append(flatten_observations(problem.simulate(theta, _rng(seed, role, index, 3))))
    else:
        first = start // STREAM_BLOCK_SIZE
        last = (start + size - 1) // STREAM_BLOCK_SIZE
        for block in range(first, last + 1):
            theta = problem.sample_prior(STREAM_BLOCK_SIZE, _rng(seed, role, block, 0))
            obs = flatten_observations(problem.simulate(theta, _rng(seed, role, block, 1)))
            lo = max(start - block * STREAM_BLOCK_SIZE, 0)
            hi = min(start + size - block * STREAM_BLOCK_SIZE, STREAM_BLOCK_SIZE)
            theta_parts.append(theta[lo:hi])
            obs_parts.append(obs[lo:hi])
    return np.concatenate(theta_parts), unflatten_observations(np.concatenate(obs_parts))


def array_descriptor(array):
    value = np.ascontiguousarray(array)
    return {"shape": list(value.shape), "dtype": value.dtype.str,
            "sha256": sha256(value.tobytes()).hexdigest()}


def prepare_data(output_root):
    """Create immutable NPZ role artifacts and their scientific/hash manifest.

    Diagnostic worlds are a separate full-prior stream. Only primary test worlds
    are centrally selected. An existing output directory is never overwritten.
    """
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=False)
    problem = build_problem()
    source_dir = Path(__file__).resolve().parent
    source_paths = [source_dir / name for name in ("analytical_benchmark_v2.py", "hierarchical_noise.py",
                                                  "hierarchical_exact.py", "hierarchical_problem.py")]
    sources = {path.name: sha256(path.read_bytes()).hexdigest() for path in source_paths}
    manifest = {"schema": SCHEMA + ".data", "seed": SEED,
                "approved_contract_sha256": APPROVED_CONTRACT_SHA256,
                "scientific_fingerprint": problem.scientific_fingerprint,
                "specification": problem.scientific_manifest(),
                "numeric_prior_verification": _verify_prior(problem), "sources": sources,
                "stream_block_size": STREAM_BLOCK_SIZE, "roles": {}}
    for role, size in ROLE_COUNTS.items():
        theta, obs = draw_worlds(size, role=role, central=role == "test")
        flat = flatten_observations(obs)
        path = root / (role + ".npz")
        with path.open("xb") as handle:
            np.savez(handle, parameters=theta, observations=flat)
        manifest["roles"][role] = {"file": path.name, "worlds": size,
            "central": role == "test", "sha256": sha256(path.read_bytes()).hexdigest(),
            "parameters": array_descriptor(theta), "observations": array_descriptor(flat)}
    if any(sha256(path.read_bytes()).hexdigest() != sources[path.name] for path in source_paths):
        raise ValueError("data-generation source changed during preparation")
    with (root / "manifest.json").open("x") as handle:
        json.dump(manifest, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    return manifest
