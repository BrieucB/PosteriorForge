"""Independent joint Gaussian/Schur and one-block RNG/isolation CPU tests."""
from copy import deepcopy
import inspect
import json
import random
import numpy as np
import pytest
from posteriorforge import conditional_globals as kernel
from posteriorforge.science.analytical_benchmark_v2 import build_problem, unflatten_observations


def rows(count=19):
    return build_problem().sample_prior(count, np.random.default_rng(783))


def independent_joint():
    prior = build_problem().prior.base
    maps = [np.eye(3)] + [b.population_map for b in prior.conditional_blocks]
    mapping = np.concatenate(maps)
    offset = np.concatenate([np.zeros(3)] + [b.offset for b in prior.conditional_blocks])
    noise = np.zeros((len(mapping), len(mapping)))
    position = 3
    for block in prior.conditional_blocks:
        n = len(block.local_names)
        noise[position:position+n, position:position+n] = block.covariance
        position += n
    return (mapping @ prior.global_mean + offset,
            mapping @ prior.global_covariance @ mapping.T + noise,
            prior.parameter_order)


def test_all_global_conditional_matches_independent_joint_gaussian_schur_complement():
    values = rows()
    joint_mean, joint_covariance, names = independent_joint()
    observed = list(range(3, len(names)))
    cross = joint_covariance[:3, 3:]
    regression = np.linalg.solve(joint_covariance[np.ix_(observed, observed)], cross.T)
    physical = build_problem().parameter_order
    physics = values[:, [physical.index(name) for name in names[3:]]]
    expected_mean = joint_mean[:3] + (physics - joint_mean[3:]) @ regression
    expected_covariance = joint_covariance[:3, :3] - cross @ regression
    mean, covariance = kernel.conditional_parameters(values)
    np.testing.assert_allclose(mean, expected_mean, atol=3e-14, rtol=0)
    np.testing.assert_allclose(covariance, expected_covariance, atol=4e-15, rtol=0)
    assert np.any(np.abs(covariance - np.diag(np.diag(covariance))) > 1e-5)


def test_joint_prior_logdensity_ratios_equal_derived_multivariate_gaussian_ratios():
    problem = build_problem()
    values = rows()
    left, right = values.copy(), values.copy()
    left[:, :3] = np.arange(len(values)*3).reshape(-1, 3) / 9 - 2
    right[:, :3] = np.arange(len(values)*3).reshape(-1, 3)[::-1] / 13 - 1
    mean, covariance = kernel.conditional_parameters(values)
    l, r = left[:, :3] - mean, right[:, :3] - mean
    expected = -.5 * (np.sum(l * np.linalg.solve(covariance, l.T).T, axis=1)
                      - np.sum(r * np.linalg.solve(covariance, r.T).T, axis=1))
    actual = problem.log_prior(left) - problem.log_prior(right)
    np.testing.assert_allclose(actual, expected, rtol=4e-14, atol=2e-12)


def test_actual_likelihood_and_conditional_do_not_use_original_globals_or_noise_values():
    values = rows()
    changed = values.copy()
    changed[:, :3] += np.array([19., -8., 5.])
    observation = unflatten_observations(np.linspace(-1, 2, 33, dtype=np.float64)[None])
    p = build_problem()
    np.testing.assert_array_equal(p.log_likelihood(changed, observation), p.log_likelihood(values, observation))
    original = kernel.conditional_parameters(values)
    for index in kernel.metadata()["noise_indices"]:
        changed[:, index] = np.linspace(.01, .5, len(values))
    repeated = kernel.conditional_parameters(changed)
    assert original[0].tobytes() == repeated[0].tobytes()
    assert original[1].tobytes() == repeated[1].tobytes()


@pytest.mark.parametrize("layout", ["C", "F", "strided", "readonly"])
def test_one_joint_correlated_draw_replays_and_keeps_all18_column_bytes(layout):
    values = rows()
    values[0, 4] = -0.
    if layout == "F": values = np.asfortranarray(values)
    elif layout == "strided": values = np.repeat(values, 2, axis=0)[::2]
    elif layout == "readonly": values.setflags(write=False)
    before = values.copy()
    generator = np.random.default_rng(531)
    replay = np.random.default_rng(531)
    mean, covariance = kernel.conditional_parameters(values)
    normals = replay.standard_normal((len(values), 3))
    expected = mean + normals @ np.linalg.cholesky(covariance).T
    output = kernel.refresh(values, generator)
    assert output[:, :3].tobytes() == expected.tobytes()
    assert output[:, 3:].tobytes() == before[:, 3:].tobytes()
    assert output.dtype == np.float64 and not np.shares_memory(output, values)
    assert values.tobytes() == before.tobytes()
    assert generator.bit_generator.state == replay.bit_generator.state
    assert build_problem().contains(output).all()
    # Incorrectly treating the block as three independent normals is distinguishable.
    independent = mean + normals * np.sqrt(np.diag(covariance))
    assert not np.array_equal(output[:, :3], independent)


def test_conditional_mean_is_row_batch_and_layout_invariant_without_default_RNG_use():
    values = rows()
    mean, covariance = kernel.conditional_parameters(values)
    pieces = np.concatenate([kernel.conditional_parameters(values[:3])[0],
                             kernel.conditional_parameters(values[3:])[0]])
    singles = np.concatenate([kernel.conditional_parameters(v[None])[0] for v in values])
    assert mean.tobytes() == pieces.tobytes() == singles.tobytes()
    assert mean.tobytes() == kernel.conditional_parameters(np.asfortranarray(values))[0].tobytes()
    np.random.seed(91)
    default_before = deepcopy(np.random.get_state())
    python_before = random.getstate()
    other = np.random.default_rng(82)
    other_before = deepcopy(other.bit_generator.state)
    result = kernel.refresh(values, np.random.default_rng(12))
    assert result.shape == values.shape
    after = np.random.get_state()
    assert default_before[0] == after[0] and default_before[2:] == after[2:]
    np.testing.assert_array_equal(default_before[1], after[1])
    assert random.getstate() == python_before and other.bit_generator.state == other_before


@pytest.mark.parametrize("fault", ["nan", "inf", "float32", "list", "empty", "shape", "low_noise", "high_noise"])
def test_invalid_input_fails_without_generator_advance_input_mutation_or_repair(fault):
    values = rows()
    if fault == "nan": values[0, 1] = np.nan
    elif fault == "inf": values[0, 7] = np.inf
    elif fault == "float32": values = values.astype(np.float32)
    elif fault == "list": values = values.tolist()
    elif fault == "empty": values = values[:0]
    elif fault == "shape": values = values[:, :20]
    elif fault == "low_noise": values[0, 6] = np.nextafter(.01, -np.inf)
    else: values[0, 20] = np.nextafter(.5, np.inf)
    before = deepcopy(values)
    generator = np.random.default_rng(81)
    state = deepcopy(generator.bit_generator.state)
    with pytest.raises(ValueError): kernel.refresh(values, generator)
    assert generator.bit_generator.state == state
    if isinstance(values, np.ndarray): assert values.tobytes() == before.tobytes()
    else: assert values == before


@pytest.mark.parametrize("rng", [None, 1, np.random.RandomState(1)])
def test_explicit_modern_generator_required(rng):
    with pytest.raises(ValueError): kernel.refresh(rows(), rng)


def test_science_source_guard_and_metadata_have_full_block_not_scalar_scope(monkeypatch):
    metadata = kernel.metadata()
    assert metadata["target_indices"] == [0, 1, 2] and metadata["unchanged_indices"] == list(range(3, 21))
    assert metadata["sweeps"] == 1 and not metadata["scalar_calibration"]
    assert not metadata["posterior_accuracy_guaranteed"] and not metadata["local_marginals_corrected"]
    assert not metadata["finite_cohort_gate_improvement_guaranteed"]
    assert set(inspect.signature(kernel.refresh).parameters) == {"values", "rng"}
    coefficients = np.array(metadata["conditional_mean_coefficients_physical_order"])
    assert np.array_equal(coefficients[:3], np.zeros((3, 3)))
    assert np.array_equal(coefficients[metadata["noise_indices"]], np.zeros((6, 3)))
    json.dumps(metadata, allow_nan=False)
    monkeypatch.setitem(kernel.PINNED_SCIENCE_SHA256, "hierarchical_exact.py", "0" * 64)
    with pytest.raises(ValueError): kernel.conditional_parameters(rows())
