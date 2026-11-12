"""Engineering-only fixtures; no case data or scientific cohort is refreshed."""
import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/refresh_ss15_campaign_mu_a.py"
spec = importlib.util.spec_from_file_location("campaign_mu_a_refresh_test", PATH)
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


def fixture_parent():
    metadata = p.kernel.metadata()
    rows = np.zeros((50000, 21), np.float64)
    rows[:, metadata["noise_indices"]] = .25
    def desc(shape, dtype):
        return {"shape": shape, "dtype": dtype, "sha256": "a" * 64}
    raw = {"checkpoint_sha256": p.CHECKPOINT_SHA,
           "scientific_fingerprint": metadata["scientific_fingerprint"],
           "parameter_order": metadata["physical_parameter_order"],
           "context_theta": desc([4096, 21], "<f8"),
           "context_obs": desc([4096, 33], "<f8"), "query": desc([1, 33], "<f8")}
    doc = {"schema": p.PARENT_SCHEMA, "variant": p.PARENT_VARIANT, "status": "complete",
           **p.request_spec(20261112, 3), "runner_source_sha256": p.CAMPAIGN_SHA,
           "frozen_pilot_source_sha256": p.PILOT_SHA, "rank": 0, "world_size": 1,
           "sources_unchanged": True, "default_rng_preserved": True,
           "physical_parameter_order_unchanged": True, "clipping_or_repair_performed": False,
           "partial_vector_filtering": False, "original_gate_unchanged": True,
           "actual_request_fixed_conditional_screen_performed": False,
           "completed_stage_indices": list(range(21)), **raw, "raw_scientific_bindings": copy.deepcopy(raw),
           "model_inputs": {"context_theta": desc([4096, 21], "<f4"),
                            "context_obs": desc([4096, 15], "<f4"), "query": desc([1, 15], "<f4")},
           "samples": p.array_descriptor(rows)}
    return doc, rows, metadata


def test_exact23_grid_and_seed_namespaces():
    assert len(p.PENDING_GRID) == len(set(p.PENDING_GRID)) == 23
    seeds = []
    for context_seed, case in p.PENDING_GRID:
        spec = p.request_spec(context_seed, case)
        assert spec["requested_count"] == 50000 and spec["context_size"] == 4096
        seeds.append(p.derive_seed(context_seed, p.ROLE, case=case, repetition=0))
    assert len(set(seeds)) == 23
    assert p.ROLE == p.frozen_refresh().ROLE == "ss15_mu_a_gibbs_once_v1"


@pytest.mark.parametrize("seed,case", [(20261111, 0), (20261114, 0), (20261112, 8), (True, 1)])
def test_anchor_or_outside_grid_refused(seed, case):
    with pytest.raises(ValueError):
        p.request_spec(seed, case)


@pytest.mark.parametrize("changes", [{"count": 49999}, {"context_size": 1024},
                                      {"repetition": 1}, {"count": True}])
def test_scientific_quota_cannot_be_overridden(changes):
    with pytest.raises(ValueError):
        p.request_spec(20261112, 1, **changes)


def test_full_parent_shape_and_scientific_binding():
    doc, rows, metadata = fixture_parent()
    assert p.validate_parent(doc, rows, metadata) == p.request_spec(20261112, 3)
    mutations = [
        lambda d: d.update(runner_source_sha256="f" * 64),
        lambda d: d.update(status="failed"),
        lambda d: d.update(requested_count=49999),
        lambda d: d["raw_scientific_bindings"]["query"].update(sha256="b" * 64),
        lambda d: d["model_inputs"]["context_obs"].update(shape=[4096, 33]),
        lambda d: d.update(parameter_order=list(reversed(metadata["physical_parameter_order"]))),
        lambda d: d["samples"].update(sha256="b" * 64)]
    for mutate in mutations:
        bad = copy.deepcopy(doc)
        mutate(bad)
        with pytest.raises(ValueError):
            p.validate_parent(bad, rows, metadata)


def test_no_unfrozen_source_execution(monkeypatch, tmp_path):
    monkeypatch.setattr(p, "CAMPAIGN_SHA", "UNFROZEN_FAIL_CLOSED")
    with pytest.raises(ValueError, match="not frozen"):
        p.load_parent(tmp_path / "missing.json", "a" * 64, p.frozen_refresh(), p.kernel.metadata())


def test_wrong_parent_custody_hash_refused(monkeypatch, tmp_path):
    monkeypatch.setattr(p, "CAMPAIGN_SHA", "b" * 64)
    path = tmp_path / "manifest.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="custody hash"):
        p.load_parent(path, "a" * 64, p.frozen_refresh(), p.kernel.metadata())


def test_reuses_exact_frozen_one_pass_not_reimplementation():
    base = p.frozen_refresh()
    assert base.KERNEL_SHA == p.KERNEL_SHA
    assert p.sha256_file(Path(base.__file__)) == p.BASE_SHA
    assert p.kernel.source_sha256() == p.KERNEL_SHA
    # The actual full-quota algebra and private-RNG tests live with frozen5c.


def test_no_observation_reference_evaluation_inputs():
    import ast
    tree = ast.parse(PATH.read_text())
    imports = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    assert not any(name and ("reference" in name or "validation" in name or "evaluat" in name
                             or "torch" in name or "tabpfn" in name) for name in imports)
    assert not hasattr(p, "load_context") and not hasattr(p, "load_heldout")
