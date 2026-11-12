"""CPU-only routing, anchor honesty and source ownership for pending SS15 requests."""
import ast
import builtins
from copy import deepcopy
import inspect
import json
from pathlib import Path
import subprocess
import sys
import symtable
from types import SimpleNamespace

import numpy as np
import pytest

from posteriorforge import performance_ss15_campaign as campaign
from posteriorforge import performance_ss15 as pilot
from posteriorforge.artifacts import sha256_file


def toy_inputs(seed, case):
    """Engineering fixtures only: no simulator, posterior or new scientific worlds."""
    offset = (seed - campaign.ANCHOR[0]) * .1
    raw = np.linspace(-1, 2, 4096 * 33, dtype=np.float64).reshape(4096, 33) + offset
    theta = np.linspace(-2, 3, 4096 * 21, dtype=np.float64).reshape(4096, 21) + offset
    observed = (raw[case:case+1] + case * .3).copy()
    return SimpleNamespace(theta=theta, obs=raw), observed


def install_toy_banks(monkeypatch):
    calls = []
    def context(banks, seed, size):
        assert banks == "toy" and size == 4096
        calls.append(("context", seed))
        return toy_inputs(seed, 0)[0]
    def observed(banks, case):
        assert banks == "toy"
        calls.append(("query", case))
        # Heldout observations depend on case, not the context-bank seed.
        return SimpleNamespace(obs=toy_inputs(campaign.ANCHOR[0], case)[1][0])
    monkeypatch.setattr(campaign, "load_context", context)
    monkeypatch.setattr(campaign, "load_heldout", observed)
    return calls


def test_predefined24_and_exact_pending23_keep_existing_pilot_out():
    assert len(campaign.PREDEFINED_GRID) == 24 and len(set(campaign.PREDEFINED_GRID)) == 24
    assert len(campaign.PENDING_GRID) == 23 and campaign.ANCHOR not in campaign.PENDING_GRID
    for seed, case in campaign.PENDING_GRID:
        spec = campaign.request_spec(context_seed=seed, case=case)
        assert spec == {"context_seed": seed, "case": case, "context_size": 4096,
                        "requested_count": 50000, "repetition": 0}
    with pytest.raises(RuntimeError, match="already exists"):
        campaign.request_spec(context_seed=campaign.ANCHOR[0], case=0)


@pytest.mark.parametrize("change", [{"context_seed": 20261110}, {"context_seed": True},
                                  {"case": -1}, {"case": 8}, {"case": True},
                                  {"context_size": 1024}, {"context_size": 4096.},
                                  {"count": 49999}, {"count": 50000.},
                                  {"repetition": 1}, {"repetition": False}])
def test_campaign_rejects_outside_grid_or_reduced_requests_before_output_model_or_budget(tmp_path, change):
    spec = {"context_seed": 20261112, "case": 3, **change}
    with pytest.raises(RuntimeError):
        campaign.run(banks="unused", output=tmp_path / "fresh", checkpoint="unused",
                     ledger="unused", reference="unused", reference_grid="unused",
                     qualification="unused", **spec)
    assert not (tmp_path / "fresh").exists()


@pytest.mark.parametrize("actual_seed,actual_case", [(20261111, 1), (20261112, 0), (20261113, 7)])
def test_anchor_replay_always_receives_separately_loaded_anchor_inputs_not_actual_query(monkeypatch, actual_seed, actual_case):
    calls = install_toy_banks(monkeypatch)
    actual = campaign.input_bundle("toy", actual_seed, actual_case)
    validated = []
    def replay(path, *, raw_bindings, transformed_bindings):
        assert path == "authentic-anchor-path"
        validated.append((deepcopy(raw_bindings), deepcopy(transformed_bindings)))
        return {"matching_FP32_execution_passed_all21": True,
                "FP16_hard_validity_passed_all21": True,
                "FP16_strict_failed_stages": list(range(21)),
                "FP16_matching_uncached_strict_failed_stages": [0, 1]}, {"passed": True}
    # Mock validator observes plumbing only; no fake fixture is inference evidence.
    monkeypatch.setattr(pilot, "qualification_receipt", replay)
    anchor = campaign.conditional_anchor("toy", "authentic-anchor-path")
    original, query = toy_inputs(*campaign.ANCHOR)
    expected_raw = campaign.raw_bindings(original, query)
    assert validated[0][0] == expected_raw == anchor["raw_scientific_bindings"]
    assert validated[0][0] != actual["raw_bindings"]
    assert calls == [("context", actual_seed), ("query", actual_case),
                     ("context", campaign.ANCHOR[0]), ("query", 0)]
    assert not anchor["qualifies_other_queries_or_contexts"]
    assert anchor["case"] == 0 and anchor["context_seed"] == campaign.ANCHOR[0]
    scope = campaign.numerical_scope(actual, anchor)
    assert not scope["actual_request_fixed_conditional_screen_performed"]
    assert not scope["anchor_qualifies_actual_request"] and not scope["posterior_accuracy_claimed"]
    assert scope["same_context_as_anchor"] is (actual_seed == campaign.ANCHOR[0])
    assert scope["same_query_as_anchor"] is (actual_case == 0)
    assert actual["transformed_bindings"]["query_SS15_model"] == actual["model_inputs"]["query"]


def test_real_fixed_anchor_replays_without_qualifying_any_new_request():
    root = Path("/scratch/project/eu-26-17/eubrieucb/posteriorforge/results")
    qualification = root / "accuracy-ss15-v1/conditional-qualification-map2-01/manifest.json"
    if not qualification.is_file():
        # Locate the existing composite, never synthesize an authentic proof.
        candidates = list((root / "accuracy-ss15-v1").glob("*/manifest.json"))
        qualification = next((p for p in candidates if sha256_file(p) ==
                              campaign.CONDITIONAL_QUALIFICATION_SHA256), None)
    if qualification is None or not qualification.is_file():
        pytest.skip("immutable fixed anchor artifact unavailable")
    campaign.require_sources()
    anchor = campaign.conditional_anchor(root / "banks-v1", qualification)
    assert anchor["qualification_sha256"] == "b192d05e68d59a3518ea9df5c6c8c4f36845fa89418b30d404261615a25d5116"
    assert anchor["replay"]["passed"] and anchor["matching_FP32_execution_passed_all21"]
    assert anchor["raw_scientific_bindings"]["query"]["shape"] == [1, 33]
    assert anchor["model_inputs"]["query"]["shape"] == [1, 15]
    assert not anchor["qualifies_other_queries_or_contexts"]


@pytest.mark.parametrize("name", ["engine_scope", "model_bindings", "validate_prediction", "qualification_receipt"])
def test_replaced_frozen_pilot_function_is_rejected_before_use(monkeypatch, name):
    monkeypatch.setattr(pilot, name, lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError, match="frozen SS15 function changed"):
        campaign.require_sources()


def test_policy_and_original_producer_identity_are_honest_and_torch_free():
    command = ("import sys;from posteriorforge import performance_ss15_campaign as c;"
               "c.require_sources();m=c.execution_metadata();assert 'torch' not in sys.modules;"
               "assert m['pending_new_grid_points']==23;"
               "assert not m['new_case_numerical_qualification_claimed'];"
               "assert m['engine_policy']['module_sha256']==c.PILOT_SHA256")
    subprocess.run([sys.executable, "-c", command], check=True)
    result = subprocess.run([sys.executable, "-m", "posteriorforge.performance_ss15_campaign",
                             "--help"], check=True, capture_output=True, text=True)
    assert "--case" in result.stdout and "--context-seed" in result.stdout
    assert "--count" not in result.stdout and "--repetition" not in result.stdout
    assert sha256_file(pilot.__file__) == campaign.PILOT_SHA256
    assert campaign.SCHEMA != pilot.SCHEMA and campaign.VARIANT != pilot.VARIANT


def test_no_unresolved_module_globals_in_run_or_preflight_helpers():
    table = symtable.symtable(Path(campaign.__file__).read_text(), campaign.__file__, "exec")
    for name in ("run", "input_bundle", "conditional_anchor", "numerical_scope"):
        scope = next(child for child in table.get_children() if child.get_name() == name)
        missing = [s.get_name() for s in scope.get_symbols() if s.is_global() and s.is_referenced()
                   and s.get_name() not in vars(campaign) and not hasattr(builtins, s.get_name())]
        assert missing == []


def test_run_routes_actual_case_seed_and_unchanged_sampler_without_mutating_private_aliases():
    tree = ast.parse(inspect.getsource(campaign.run))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    seeds = next(node for node in calls if isinstance(node.func, ast.Attribute)
                 and node.func.attr == "worker_stream_seeds")
    assert [ast.unparse(arg) for arg in seeds.args] == ["context_seed", "case", "repetition", "0", "1"]
    refs = next(node for node in calls if isinstance(node.func, ast.Name) and node.func.id == "reference_binding")
    assert next(k.value for k in refs.keywords if k.arg == "case").id == "case"
    sampler = next(node for node in calls if isinstance(node.func, ast.Name) and node.func.id == "accept_reject")
    keywords = {k.arg: ast.unparse(k.value) for k in sampler.keywords}
    assert keywords["num_samples"] == keywords["max_sampling_batch_size"] == "COUNT"
    assert keywords["max_iter_rejection"] == "None" and keywords["accept_reject_fn"] == "prior.support.check"
    wave = next(node for node in calls if isinstance(node.func, ast.Attribute) and node.func.attr == "stage_major_wave")
    assert ast.unparse(wave.func) == "wave.stage_major_wave" and ast.unparse(wave.args[5]) == "1024"
    scopes = [node for node in ast.walk(tree) if isinstance(node, ast.With)
              and any("finite_cdf_compatibility" in ast.unparse(item.context_expr) for item in node.items)]
    assert len(scopes) == 1
    assert [ast.unparse(item.context_expr.func) for item in scopes[0].items] == [
        "finite_cdf_compatibility", "pilot.engine_scope"]
    # All original callback/class/function aliases remain owned by the frozen module.
    assert campaign.pilot.SS15StageAdapter is campaign._ADAPTER_CLASS
    assert "matching_FP32_execution_passed_all21" not in inspect.getsource(campaign.run)
