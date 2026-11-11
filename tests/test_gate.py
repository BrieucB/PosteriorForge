"""Small CPU-only tamper tests for the scientific run authentication boundary."""
import copy
import json
import subprocess
import sys

import numpy as np
import pytest

from posteriorforge.banks import AnalyticalBankProvider, CONTEXT_SEEDS, derive_seed
from posteriorforge.gate import evaluate_gate, verify_run_binding
from posteriorforge.inference import (CHECKPOINT_SHA256, UPSTREAM_REVISION,
                                      frozen_regressor_settings, upstream_source_hashes)
from posteriorforge.science.analytical_benchmark_v2 import array_descriptor, build_problem


@pytest.fixture
def binding():
    # The helper authenticates descriptors, so three complete rows suffice.
    # Full 4096-row selection remains the responsibility of evaluate_gate.
    seed, case = CONTEXT_SEEDS[0], 2
    context = AnalyticalBankProvider().get_bank(seed=seed, count=3)
    query = np.linspace(-.5, .5, 33, dtype=np.float64).reshape(1, 33)
    manifest = {
        "schema": "posteriorforge.inference.v1", "variant": "upstream_unmodified",
        "context_seed": seed, "case": case, "context_size": 4096,
        "requested_count": 50000, "status": "complete",
        "scientific_fingerprint": build_problem().scientific_fingerprint,
        "parameter_order": list(build_problem().parameter_order),
        "context_theta": array_descriptor(context.theta),
        "context_obs": array_descriptor(context.obs),
        "checkpoint_sha256": CHECKPOINT_SHA256, "upstream_revision": UPSTREAM_REVISION,
        "upstream_files": upstream_source_hashes(), "context_filter": "no_filtering",
        "max_iter_rejection": None, "scientific_run": True, "microbatch": 10000,
        "seed": derive_seed(seed, "posterior", case=case), "repetition": 0,
        "rank": 0, "world_size": 1,
        "regressor_settings": frozen_regressor_settings("fixture-not-a-real-checkpoint.ckpt"),
        "query": array_descriptor(query),
    }
    return manifest, context, query, seed, case


def _verify(binding, manifest):
    _, context, query, seed, case = binding
    return verify_run_binding(manifest, context, query, seed, case)


def test_valid_contract_accepts_without_checkpoint_or_model_loading(binding):
    manifest = binding[0]
    assert _verify(binding, manifest) is None
    moved = copy.deepcopy(manifest)
    moved["regressor_settings"]["model_path"] = "/another-machine/cache/same-checkpoint.ckpt"
    assert _verify(binding, moved) is None  # immutable weight hash binds weights; path is portable


@pytest.mark.parametrize("field", ["context_theta", "context_obs"])
@pytest.mark.parametrize("component", ["sha256", "shape", "dtype"])
def test_context_array_descriptors_cannot_be_swapped_or_retyped(binding, field, component):
    manifest = copy.deepcopy(binding[0])
    value = manifest[field]
    if component == "sha256":
        value[component] = "0" * 64
    elif component == "shape":
        value[component] = [value["shape"][0] + 1, value["shape"][1]]
    else:
        value[component] = "<f4"
    with pytest.raises(ValueError, match=field):
        _verify(binding, manifest)


@pytest.mark.parametrize("field,value", [
    ("scientific_fingerprint", "another-scientific-problem"),
    ("checkpoint_sha256", "0" * 64),
    ("upstream_revision", "0" * 40),
    ("seed", 17),
    ("microbatch", 4096),
    ("repetition", 1),
    ("rank", 1),
    ("world_size", 2),
    ("context_filter", "distance_filtering"),
    ("max_iter_rejection", 1000),
    ("scientific_run", False),
])
def test_fixed_execution_contract_tampering_is_rejected(binding, field, value):
    manifest = copy.deepcopy(binding[0])
    manifest[field] = value
    with pytest.raises(ValueError, match=field):
        _verify(binding, manifest)


def test_parameter_order_and_vendored_source_tampering_is_rejected(binding):
    manifest = copy.deepcopy(binding[0])
    manifest["parameter_order"][0], manifest["parameter_order"][1] = manifest["parameter_order"][1], manifest["parameter_order"][0]
    with pytest.raises(ValueError, match="parameter_order"):
        _verify(binding, manifest)
    manifest = copy.deepcopy(binding[0])
    source = next(iter(manifest["upstream_files"]))
    manifest["upstream_files"][source] = "0" * 64
    with pytest.raises(ValueError, match="upstream_files"):
        _verify(binding, manifest)
    manifest = copy.deepcopy(binding[0])
    manifest["upstream_files"].pop(source)
    with pytest.raises(ValueError, match="upstream_files"):
        _verify(binding, manifest)


@pytest.mark.parametrize("setting,value", [
    ("n_estimators", 4), ("softmax_temperature", 1.), ("average_before_softmax", True),
    ("categorical_features_indices", [0]), ("device", "cpu"),
    ("inference_precision", "torch.float16"), ("fit_mode", "fit_with_cache"),
    ("memory_saving_mode", False), ("random_state", 123), ("n_jobs", 2),
    ("ignore_pretraining_limits", True), ("inference_config", {"changed": True}),
])
def test_each_frozen_regressor_setting_is_authenticated(binding, setting, value):
    manifest = copy.deepcopy(binding[0])
    manifest["regressor_settings"][setting] = value
    with pytest.raises(ValueError, match="frozen regressor settings"):
        _verify(binding, manifest)


def test_queries_cannot_be_replaced_by_labels_or_other_observations(binding):
    manifest = copy.deepcopy(binding[0])
    context, query = binding[1], binding[2]
    for substitute in (context.theta[:1], query + .01, query.reshape(33), np.concatenate((query, context.theta[:1]), axis=1)):
        manifest["query"] = array_descriptor(substitute)
        with pytest.raises(ValueError, match="held-out observation"):
            _verify(binding, manifest)


def test_changed_expected_seed_or_case_cannot_reuse_a_run(binding):
    manifest, context, query, seed, case = binding
    with pytest.raises(ValueError, match="seed"):
        verify_run_binding(manifest, context, query, seed + 1, case)
    with pytest.raises(ValueError, match="seed"):
        verify_run_binding(manifest, context, query, seed, case + 1)


def test_missing_reference_and_run_matrix_is_incomplete_and_never_authorizes_engineering(tmp_path):
    output = tmp_path / "gate.json"
    result = evaluate_gate(banks=tmp_path / "absent-banks", references=tmp_path / "absent-references",
                           runs=tmp_path / "absent-runs", output=output)
    assert result["status"] == "incomplete"
    assert not result["passed"]
    assert not result["optimization_authorized_by_gate"]
    assert len(result["runs"]) == result["expected_runs"] == 24
    assert all(entry["status"] == "missing" and entry["passed"] is False for entry in result["runs"])
    assert len(result["references"]) == 8
    assert all(entry["state"] == "missing" for entry in result["references"])
    assert json.loads(output.read_text()) == result


def test_gate_import_does_not_import_torch_or_initialize_model():
    subprocess.run([sys.executable, "-c", "import sys; import posteriorforge.gate; assert 'torch' not in sys.modules"],
                   check=True, capture_output=True, text=True)
