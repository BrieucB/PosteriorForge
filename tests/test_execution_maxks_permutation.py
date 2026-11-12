"""Synthetic unit fixtures for the whole-vector random-label diagnostic."""
import importlib.util
import itertools
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import permutation_test

PATH = Path(__file__).resolve().parents[1]/"scripts/calibrate_execution_maxks.py"
spec = importlib.util.spec_from_file_location("maxks_permutation_tests", PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def brute(first, second, directions):
    a, b = np.asarray(first)@directions.T, np.asarray(second)@directions.T
    return np.array([max(abs(np.searchsorted(np.sort(a[:, j]), x, side="right")/len(a)
                            - np.searchsorted(np.sort(b[:, j]), x, side="right")/len(b))
                        for x in np.unique(np.r_[a[:, j], b[:, j]]))
                     for j in range(len(directions))])


@pytest.mark.parametrize("n,m", [(4, 4), (3, 5)])
def test_tied_integer_counts_match_brute_for_every_whole_row_allocation(n, m):
    pool = np.array([[0, 0], [0, 1], [1, 0], [1, 1],
                     [2, 1], [2, 2], [0, 0], [1, 1]], dtype=float)
    directions = np.array([[1, 0], [0, 1], [1, 1], [1, -1]])
    cache = module.ProjectionRankCache(pool[:n], pool[n:n+m], directions)
    assert cache.tied_adjacent_count > 0
    for selected in itertools.combinations(range(n+m), n):
        labels = np.zeros(n+m, dtype=bool)
        labels[list(selected)] = True
        actual = cache.counts(labels)/cache.denominator
        np.testing.assert_allclose(actual, brute(pool[labels], pool[~labels], directions),
                                   atol=1e-15, rtol=0)


def test_tie_interiors_never_create_spurious_difference():
    a = np.zeros((3, 2))
    cache = module.ProjectionRankCache(a, a.copy(), np.eye(2))
    np.testing.assert_array_equal(cache.observed_counts(), [0, 0])


def test_shared_row_labels_preserve_duplicate_and_opposite_projection_statistics():
    rng = np.random.default_rng(7)
    a, b = rng.normal(size=(12, 2)), rng.normal(size=(12, 2))
    directions = np.array([[1, 0], [1, 0], [-1, 0], [0, 1]])
    result = module.calibrate(a, b, directions, permutations=19,
                              seed_sequence=np.random.SeedSequence(11))
    null = np.array(result["null_axis_ks_integer_numerators"])
    np.testing.assert_array_equal(null[:, 0], null[:, 1])
    np.testing.assert_array_equal(null[:, 0], null[:, 2])


def test_randomized_result_reproducible_plus_one_and_exact_integer_ties():
    a = np.arange(12.).reshape(6, 2)
    b = a+.3
    kwargs = dict(permutations=31)
    first = module.calibrate(a, b, np.eye(2),
                             seed_sequence=np.random.SeedSequence(81), **kwargs)
    second = module.calibrate(a, b, np.eye(2),
                              seed_sequence=np.random.SeedSequence(81), **kwargs)
    assert first["null_axis_ks_integer_numerators"] == second["null_axis_ks_integer_numerators"]
    count = sum(x >= first["observed_max53_integer_numerator"]
                for x in first["null_max53_integer_numerators"])
    assert first["randomized_p_plus_one"] == (1+count)/32
    assert first["p_value_resolution"] == 1/32


def test_tiny_complete_enumeration_matches_official_scipy_greater_tail():
    pool = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.], [2., 0.], [2., 2.]])
    directions = np.array([[1., 0.], [0., 1.], [1., -1.]])
    cache = module.ProjectionRankCache(pool[:3], pool[3:], directions)
    observed = cache.observed_counts().max()
    exact = []
    for selected in itertools.combinations(range(6), 3):
        labels = np.zeros(6, dtype=bool)
        labels[list(selected)] = True
        exact.append(cache.counts(labels).max())
    def statistic(first, second):
        return float(brute(pool[np.asarray(first, dtype=int)],
                           pool[np.asarray(second, dtype=int)], directions).max())
    result = permutation_test((np.arange(3), np.arange(3, 6)), statistic,
                              permutation_type="independent", alternative="greater",
                              vectorized=False, n_resamples=np.inf)
    assert result.pvalue == pytest.approx(np.mean(np.asarray(exact) >= observed))


@pytest.mark.parametrize("k,n", [(0, 127), (1, 127), (63, 127), (127, 127)])
def test_mc_interval_reports_uncertainty_and_preserves_plus_one_bounds(k, n):
    interval = module.monte_carlo_interval(k, n)
    low, high = interval["null_exceedance_probability_interval"]
    assert low <= k/n <= high
    adjusted = interval["plus_one_probability_mc_interval"]
    assert adjusted[0] >= 1/(n+1)
    assert adjusted[1] <= 1


@pytest.mark.parametrize("case", ["nonfinite", "unbalanced", "nonboolean"])
def test_invalid_inputs_rejected(case):
    a = np.zeros((4, 2)); b = np.ones((4, 2))
    if case == "nonfinite":
        a[0, 0] = np.nan
        with pytest.raises(ValueError):
            module.ProjectionRankCache(a, b, np.eye(2))
        return
    cache = module.ProjectionRankCache(a, b, np.eye(2))
    labels = np.zeros(8, dtype=bool) if case == "unbalanced" else np.arange(8) < 4
    if case == "nonboolean":
        labels = labels.astype(int)
    with pytest.raises(ValueError):
        cache.counts(labels)
