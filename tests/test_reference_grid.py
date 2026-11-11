"""Small pure/mock grid tests; no real retained reference or accelerator work."""
import time

import numpy as np
import pytest

from posteriorforge.reference_grid import _adaptive_grid, _require_budgeted_gpu, refine_reference_grid
from posteriorforge.budget import BudgetLedger


def _uniform_cdfs(queries):
    return np.tile((np.asarray(queries) + 1) / 2, (53, 1))


def _run(evaluator, *, threshold=.005, max_rounds=12, max_queries=1025, deadline=None):
    queries = np.array([-1., 0., 1.])
    return _adaptive_grid(queries, _uniform_cdfs(queries), .001, evaluator,
        max_cdf_step=threshold, max_rounds=max_rounds, max_queries=max_queries,
        deadline=time.monotonic() + 30 if deadline is None else deadline)


def test_deterministic_bisection_caches_queries_and_retains_base():
    evaluated = []
    def evaluate(queries):
        evaluated.extend(queries.tolist())
        return _uniform_cdfs(queries), np.full((53, len(queries)), .0005)
    queries, cdfs, result = _run(evaluate)
    assert result["passed"]
    assert result["maximum_cdf_step"] <= .005
    assert result["maximum_tail_mass"] == 0
    assert result["cdf_precision"] == .001
    assert np.array_equal(queries, np.linspace(-1., 1., 257))
    assert np.array_equal(cdfs, _uniform_cdfs(queries))
    assert len(evaluated) == len(set(evaluated)) == 254
    assert not ({-1., 0., 1.} & set(evaluated))
    assert set([-1., 0., 1.]).issubset(queries)
    # Replay selected midpoints from the retained history, independent of calls.
    current = np.array([-1., 0., 1.])
    for history in result["history"]:
        selected = np.flatnonzero(np.diff(_uniform_cdfs(current), axis=1).max(axis=0) > .005)
        assert history["selected_interval_indices"] == selected.tolist()
        if not history["added_queries"]:
            break
        expected = current[selected] + np.diff(current)[selected] / 2
        np.testing.assert_array_equal(history["added_queries"], expected)
        current = np.sort(np.r_[current, expected])
    np.testing.assert_array_equal(current, queries)


@pytest.mark.parametrize("limit,expected", [("rounds", "rounds"), ("queries", "query count"), ("deadline", "deadline")])
def test_exhausted_limits_are_inconclusive(limit, expected):
    def evaluate(queries):
        return _uniform_cdfs(queries), np.zeros((53, len(queries)))
    kwargs = {"max_rounds": 0} if limit == "rounds" else {"max_queries": 3} if limit == "queries" else {"deadline": time.monotonic() - 1}
    _, _, result = _run(evaluate, **kwargs)
    assert not result["passed"]
    assert expected in result["reason"]


def test_new_point_precision_failure_is_not_promoted_to_pass():
    def evaluate(queries):
        return _uniform_cdfs(queries), np.full((53, len(queries)), .006)
    _, _, result = _run(evaluate)
    assert not result["passed"]
    assert result["cdf_precision"] == .006
    assert "precision failed" in result["reason"]


def test_reference_tail_mass_prevents_pass():
    queries = np.array([-1., 0., 1.])
    _, _, result = _adaptive_grid(queries, .1 + .8 * _uniform_cdfs(queries), .001,
        lambda queries: (_uniform_cdfs(queries), np.zeros((53, len(queries)))),
        max_cdf_step=.005, max_rounds=12, max_queries=1025, deadline=time.monotonic() + 30)
    assert not result["passed"]
    assert "tail mass" in result["reason"]


def test_evaluation_timeout_retains_inconclusive_base_grid():
    def evaluate(queries):
        raise TimeoutError("mock GPU deadline")
    queries, _, result = _run(evaluate)
    assert not result["passed"]
    assert "deadline" in result["reason"]
    np.testing.assert_array_equal(queries, [-1., 0., 1.])


def test_real_operation_requires_matching_active_budget(monkeypatch, tmp_path):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(RuntimeError, match="Slurm"):
        refine_reference_grid(tmp_path, 0, tmp_path, tmp_path / "out")
    ledger = BudgetLedger(tmp_path / "ledger.json")
    entry = ledger.reserve(gpus=1, walltime_seconds=3600, purpose="unit fixture no job submission")
    ledger.attach_job(entry["reservation_id"], "123")
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_JOB_NUM_NODES", "1")
    reservation = _require_budgeted_gpu(ledger.path)
    assert reservation["job_id"] == "123"
    assert reservation["reserved_gpu_seconds"] == 3600
    monkeypatch.setenv("SLURM_JOB_ID", "124")
    with pytest.raises(RuntimeError, match="active GPU budget"):
        _require_budgeted_gpu(ledger.path)
