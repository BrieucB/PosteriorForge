"""Independent Gaussian algebra and CPU-only isolation tests for one mu_a refresh."""
from copy import deepcopy
import inspect
import json

import numpy as np
import pytest

from posteriorforge import conditional_mu_a as conditional
from posteriorforge.science.analytical_benchmark_v2 import build_problem, unflatten_observations


def rows(count=17):
    problem = build_problem()
    return problem.sample_prior(count, np.random.default_rng(719))


def joint_gaussian():
    """Assemble a joint covariance independently from generative loadings."""
    prior = build_problem().prior.base
    loadings = [np.eye(3)] + [b.population_map for b in prior.conditional_blocks]
    mapping = np.concatenate(loadings, axis=0)
    offset = np.concatenate([np.zeros(3)] + [b.offset for b in prior.conditional_blocks])
    independent = np.zeros((len(mapping), len(mapping)))
    start = 3
    for block in prior.conditional_blocks:
        n = len(block.local_names)
        independent[start:start+n, start:start+n] = block.covariance
        start += n
    mean = mapping @ prior.global_mean + offset
    covariance = mapping @ prior.global_covariance @ mapping.T + independent
    names = prior.parameter_order
    return mean, covariance, names


def test_scalar_conditional_matches_independent_full_joint_schur_conditioning():
    values = rows()
    mean, covariance, names = joint_gaussian()
    physical = build_problem().parameter_order
    observed_indices = list(range(1, len(names)))
    measured = values[:, [physical.index(name) for name in names[1:]]]
    cross = covariance[0, observed_indices]
    weights = np.linalg.solve(covariance[np.ix_(observed_indices, observed_indices)], cross)
    independent_mean = mean[0] + (measured - mean[1:]) @ weights
    independent_variance = covariance[0, 0] - cross @ weights
    actual_mean, actual_variance = conditional.conditional_parameters(values)
    np.testing.assert_allclose(actual_mean, independent_mean, rtol=0, atol=2e-14)
    np.testing.assert_allclose(actual_variance, independent_variance, rtol=0, atol=4e-15)


def test_prior_log_density_ratios_match_gaussian_fullconditional_without_noise_or_data():
    problem = build_problem()
    values = rows()
    left = values.copy()
    right = values.copy()
    left[:, 0] = np.linspace(-1.8, 2.2, len(values))
    right[:, 0] = np.linspace(.8, -.7, len(values))
    mean, variance = conditional.conditional_parameters(values)
    expected = -.5 * ((left[:, 0] - mean)**2 - (right[:, 0] - mean)**2) / variance
    actual = problem.log_prior(left) - problem.log_prior(right)
    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=1e-12)


def test_actual_frozen_likelihood_is_independent_of_all_three_globals():
    problem = build_problem()
    values = rows()
    changed = values.copy()
    changed[:, :3] += np.array([31., -19., 23.])
    observations = unflatten_observations(np.linspace(-.7, .9, 33, dtype=np.float64)[None, :])
    original = problem.log_likelihood(values, observations)
    np.testing.assert_array_equal(problem.log_likelihood(changed, observations), original)
    for name in problem.prior.noise_names:
        values[:, problem.parameter_order.index(name)] = .25
    changed = values.copy()
    changed[:, 0] = np.linspace(-100., 100., len(values))
    np.testing.assert_array_equal(problem.log_likelihood(changed, observations),
                                  problem.log_likelihood(values, observations))


def test_conditional_ignores_original_mu_a_and_six_independent_noise_coordinates():
    values = rows()
    first = conditional.conditional_parameters(values)
    changed = values.copy()
    changed[:, 0] = np.linspace(-1e6, 1e6, len(values))
    for j in build_problem().prior.noise_names:
        changed[:, build_problem().parameter_order.index(j)] = np.linspace(.01, .5, len(values))
    second = conditional.conditional_parameters(changed)
    np.testing.assert_array_equal(first[0], second[0])
    assert first[1] == second[1]


@pytest.mark.parametrize("layout", ["C", "F", "strided", "readonly"])
def test_one_refresh_preserves_input_and_all_other20_column_bytes(layout):
    values = rows()
    values[0, 1] = -0.0
    if layout == "F":
        values = np.asfortranarray(values)
    elif layout == "strided":
        values = np.repeat(values, 2, axis=0)[::2]
    elif layout == "readonly":
        values.setflags(write=False)
    before = values.copy()
    generator = np.random.default_rng(192)
    expected_generator = np.random.default_rng(192)
    mean, variance = conditional.conditional_parameters(values)
    normals = expected_generator.standard_normal(len(values))
    output = conditional.refresh(values, generator)
    assert output.shape == values.shape and output.dtype == values.dtype == np.float64
    assert not np.shares_memory(output, values)
    assert values.tobytes() == before.tobytes()
    assert output[:, 1:].tobytes() == before[:, 1:].tobytes()
    assert output[:, 0].tobytes() == (mean + np.sqrt(variance) * normals).tobytes()
    assert generator.bit_generator.state == expected_generator.bit_generator.state
    assert build_problem().contains(output).all()


def test_explicit_generator_is_reproducible_does_not_consume_default_rng_or_other_generator():
    values = rows()
    np.random.seed(519)
    default_before = deepcopy(np.random.get_state())
    other = np.random.default_rng(91)
    other_before = deepcopy(other.bit_generator.state)
    left = conditional.refresh(values, np.random.default_rng(82))
    right = conditional.refresh(values, np.random.default_rng(82))
    assert left.tobytes() == right.tobytes()
    assert other.bit_generator.state == other_before
    after = np.random.get_state()
    assert default_before[0] == after[0] and default_before[2:] == after[2:]
    np.testing.assert_array_equal(default_before[1], after[1])


@pytest.mark.parametrize("fault", ["nan", "inf", "shape", "empty", "float32", "int", "list",
                                  "below_support", "above_support"])
def test_invalid_input_rejected_without_rng_advance_mutation_or_repair(fault):
    values = rows()
    if fault == "nan":
        values[0, 0] = np.nan
    elif fault == "inf":
        values[0, 2] = np.inf
    elif fault == "shape":
        values = values[:, :20]
    elif fault == "empty":
        values = values[:0]
    elif fault == "float32":
        values = values.astype(np.float32)
    elif fault == "int":
        values = values.astype(np.int64)
    elif fault == "list":
        values = values.tolist()
    elif fault == "below_support":
        values[0, 6] = np.nextafter(.01, -np.inf)
    elif fault == "above_support":
        values[0, 20] = np.nextafter(.5, np.inf)
    before = deepcopy(values)
    generator = np.random.default_rng(12)
    state = deepcopy(generator.bit_generator.state)
    with np.errstate(over="ignore", invalid="ignore"), pytest.raises(ValueError):
        conditional.refresh(values, generator)
    assert generator.bit_generator.state == state
    if isinstance(values, np.ndarray):
        assert values.tobytes() == before.tobytes()
    else:
        assert values == before


@pytest.mark.parametrize("bad_rng", [None, 13, np.random.RandomState(13)])
def test_refresh_requires_explicit_modern_generator(bad_rng):
    with pytest.raises(ValueError):
        conditional.refresh(rows(), bad_rng)


def test_source_geometry_guard_rejects_mutated_pin(monkeypatch):
    monkeypatch.setitem(conditional.PINNED_SCIENCE_SHA256, "hierarchical_exact.py", "0" * 64)
    with pytest.raises(ValueError):
        conditional.conditional_parameters(rows())


def test_metadata_binds_actual_derived_matrices_indices_and_honest_scope():
    record = conditional.metadata()
    problem = build_problem()
    assert record["physical_parameter_order"] == list(problem.parameter_order)
    assert record["scientific_fingerprint"] == problem.scientific_fingerprint
    assert record["module_sha256"] == conditional.source_sha256()
    assert record["target_index"] == 0 and record["unchanged_indices"] == list(range(1, 21))
    assert record["sweeps"] == 1 and not record["posterior_accuracy_guaranteed"]
    assert not record["scalar_calibration"] and not record["clipping_filtering_or_repair"]
    assert record["scientific_count_enforced_by_producer"] == 50000
    precision = np.asarray(record["global_fullconditional_precision"])
    assert record["conditional_variance"] == 1 / precision[0, 0]
    assert record["conditional_mean_coefficients_physical_order"][0] == 0
    assert set(inspect.signature(conditional.refresh).parameters) == {"values", "rng"}
    assert set(inspect.signature(conditional.conditional_parameters).parameters) == {"values"}
    json.dumps(record, allow_nan=False)


def test_supported_extreme_finite_physics_is_not_silently_clipped_or_rejected():
    values = rows()
    values[:, [3, 7, 11, 15, 17, 19]] = np.finfo(np.float64).max
    before = values.copy()
    mean, variance = conditional.conditional_parameters(values)
    assert np.isfinite(mean).all() and variance > 0
    output = conditional.refresh(values, np.random.default_rng(12))
    assert np.isfinite(output).all()
    assert output[:, 1:].tobytes() == before[:, 1:].tobytes()
    assert values.tobytes() == before.tobytes()
