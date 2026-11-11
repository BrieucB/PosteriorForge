import copy

import numpy as np
import pytest
from scipy.special import ndtr

from posteriorforge.validation import (assess_samples, cdf_distance_bounds, compare_to_upstream,
                                       make_reference_input, sample_integrity)


def _reference(points=2049):
    rng = np.random.default_rng(763019)
    directions = rng.normal(size=(32, 21))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    directions = np.concatenate((np.eye(21), directions))
    queries = np.linspace(-8, 8, points)
    return {"directions": directions, "queries": queries,
            "cdfs": np.repeat(ndtr(queries)[None, :], 53, axis=0),
            "mean": np.zeros(21), "sd": np.ones(21),
            "qualification": {"state": "qualified", "precision": .001,
                              "witness_gate_passed": True, "independent_nested_passed": True,
                              "cpu_check_passed": True, "joint_reference_passed": True}}


@pytest.fixture(scope="module")
def samples():
    return np.random.default_rng(42).normal(size=(10000, 21))


def test_complete_finite_rows_and_exact_support_required(samples):
    assert not sample_integrity(samples, expected_count=len(samples))["passed"]
    mask = np.ones(len(samples), dtype=bool)
    assert sample_integrity(samples, expected_count=len(samples), support_mask=mask)["passed"]
    assert not sample_integrity(samples[:-1], expected_count=len(samples), support_mask=mask)["passed"]
    invalid = samples.copy()
    invalid[0, 0] = np.nan
    assert not sample_integrity(invalid, expected_count=len(samples), support_mask=mask)["passed"]
    assert not sample_integrity(samples, expected_count=len(samples), lower_bounds=np.full(21, -.1),
                                upper_bounds=np.full(21, .1))["passed"]
    assert np.isnan(invalid[0, 0])  # integrity never clips or repairs rows


def test_absolute_gate_passes_consistent_analytic_reference(samples):
    result = assess_samples(samples, _reference(), expected_count=len(samples),
                            support_mask=np.ones(len(samples), dtype=bool))
    assert result["passed"]
    assert result["status"] == "pass"
    assert result["marginal_cdf"]["max_upper"] < .10
    assert result["joint_projection_cdf"]["max_upper"] < .10


def test_coarse_grid_never_promotes_diagnostic_lower_bound_to_pass(samples):
    reference = _reference(points=3)
    result = assess_samples(samples, reference, expected_count=len(samples),
                            support_mask=np.ones(len(samples), dtype=bool))
    assert result["marginal_cdf"]["max_lower"] < .1
    assert result["marginal_cdf"]["max_upper"] > .1
    assert not result["passed"]
    assert result["status"] == "cdf_inconclusive"


def test_reference_uncertainty_and_independent_failures_never_pass(samples):
    for key, value in (("precision", .005001), ("independent_nested_passed", False),
                       ("joint_reference_passed", False), ("cpu_check_passed", False)):
        reference = _reference()
        reference["qualification"][key] = value
        result = assess_samples(samples, reference, expected_count=len(samples),
                                support_mask=np.ones(len(samples), dtype=bool))
        assert result["status"] == "reference_inconclusive"
        assert not result["passed"]


def test_moment_failure_is_model_failure_only_when_reference_qualified(samples):
    shifted = samples + .3
    reference = _reference()
    result = assess_samples(shifted, reference, expected_count=len(samples),
                            support_mask=np.ones(len(samples), dtype=bool))
    assert result["status"] == "model_failure"
    assert not result["moments"]["passed"]
    reference["qualification"]["state"] = "inconclusive"
    result = assess_samples(shifted, reference, expected_count=len(samples),
                            support_mask=np.ones(len(samples), dtype=bool))
    assert result["status"] == "reference_inconclusive"


def test_whole_line_cdf_bound_contains_exact_normal_ks():
    values = np.random.default_rng(1).normal(size=3000)
    queries = np.linspace(-2, 2, 57)
    bound = cdf_distance_bounds(values[:, None], queries, ndtr(queries)[None, :])
    ordered = np.sort(values)
    exact = max(np.max(np.abs(np.arange(1, len(values) + 1) / len(values) - ndtr(ordered))),
                np.max(np.abs(np.arange(len(values)) / len(values) - ndtr(ordered))))
    assert bound["lower"][0] <= exact <= bound["upper"][0]
    assert bound["upper"][0] - bound["lower"][0] < .04


def test_tail_mass_included_and_malformed_reference_rejected(samples):
    queries = np.array([-.1, .1])
    bound = cdf_distance_bounds(samples[:, :1], queries, ndtr(queries)[None, :])
    assert bound["upper"][0] > .45
    with pytest.raises(ValueError, match="monotone"):
        cdf_distance_bounds(samples[:, :1], queries, np.array([[.6, .5]]))
    with pytest.raises(ValueError, match="increasing"):
        cdf_distance_bounds(samples[:, :1], queries[::-1], np.array([[.5, .6]]))


def test_incomplete_samples_fail_even_if_reference_bad(samples):
    result = assess_samples(samples[:-1], _reference(), expected_count=len(samples),
                            support_mask=np.ones(len(samples), dtype=bool))
    assert result["status"] == "invalid_samples"
    assert not result["passed"]


def test_optimized_comparison_flags_and_same_cohort_is_zero(samples):
    reference = _reference()
    same = compare_to_upstream(samples, samples, reference_sd=reference["sd"], directions=reference["directions"])
    assert not same["investigation_required"]
    assert same["max_cdf_difference"] == 0
    changed = compare_to_upstream(samples + .1, samples, reference_sd=reference["sd"], directions=reference["directions"])
    assert changed["investigation_required"]
    assert changed["max_mean_difference_reference_sd"] > .05
    assert changed["absolute_gate_still_required"]


def test_adapter_outer_inconclusive_cannot_be_overridden(samples):
    reference = _reference()
    arrays = {name: value for name, value in reference.items() if name != "qualification"}
    manifest = {"state": "inconclusive", "converged": False, "independent_passed": False,
                "qualification": reference["qualification"]}
    adapted = make_reference_input(arrays, manifest)
    result = assess_samples(samples, adapted, expected_count=len(samples),
                            support_mask=np.ones(len(samples), dtype=bool))
    assert not result["passed"]
    assert result["status"] == "reference_inconclusive"
    assert reference["qualification"]["state"] == "qualified"
