"""Fail-closed provenance tests for the explicitly labelled CDF repair evaluator.

These fixtures are protocol records only, never posterior data or pilot evidence.
"""
import copy
import importlib.util
from pathlib import Path
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/evaluate_cdf_pilot.py"
spec = importlib.util.spec_from_file_location("evaluate_cdf_pilot", SCRIPT)
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


@pytest.fixture
def descriptor():
    from posteriorforge import compat_cdf as adapter
    return {
        "schema": "posteriorforge.compat.finite_cdf.v1",
        "adapter_id": adapter.ADAPTER_ID,
        "adapter_version": adapter.ADAPTER_VERSION,
        "adapter_sha256": adapter.adapter_source_sha256(),
        "tabpfn_version": adapter.PINNED_TABPFN_VERSION,
        "sdk_source_sha256": dict(adapter.PINNED_SDK_SHA256),
        "sdk_source_sha256_after": dict(adapter.PINNED_SDK_SHA256),
        "sdk_sources_unchanged": True,
        "sdk_aliases_unchanged": True,
        "translator_code_unchanged": True,
        "correction": "finite_cdf_parentheses_only",
        "source_guard_passed": True,
        "sdk_files_modified": False,
        "original_alias_restored": True,
        "activation": "explicit",
        "scope": "finite_ensemble_cdf_parentheses_only",
        "tail_family_changed": False,
        "execution_optimized": False,
        "patched_call_count": 1008,
    }


@pytest.fixture
def manifest(descriptor):
    from posteriorforge.inference_cdf import guard_baseline_source
    return {
        "schema": evaluation.RUN_SCHEMA, "variant": "cdf_arithmetic_v1", "status": "complete",
        "context_seed": 20261111, "case": 0, "context_size": 4096,
        "requested_count": 50000, "scientific_run": True,
        "observation_preprocessing": "raw; TabPFN frozen preprocessing",
        "log_density": "not_used", "compatibility_adapter": descriptor,
        "metadata_writer_restored": True, "explicit_variant_activation": True,
        "tail_family_changed": False, "execution_optimized": False,
        "baseline_execution_source": guard_baseline_source(),
        "expected_patched_call_count": 1008, "rejection": {"support_calls": 6},
    }


def test_exact_approved_descriptor_can_be_assessed_without_sdk_import(descriptor):
    assert evaluation.validate_adapter_descriptor(descriptor) == descriptor


@pytest.mark.parametrize("field,value", [
    ("tail_family_changed", True),
    ("execution_optimized", True),
    ("sdk_files_modified", True),
    ("sdk_sources_unchanged", False),
    ("sdk_aliases_unchanged", False),
    ("translator_code_unchanged", False),
    ("source_guard_passed", False),
    ("original_alias_restored", False),
    ("activation", "implicit"),
    ("scope", "also_whiten_targets"),
    ("adapter_sha256", "0" * 64),
    ("patched_call_count", 0),
    ("patched_call_count", True),
    ("patched_call_count", -1),
])
def test_unapproved_or_unexecuted_repair_is_rejected(descriptor, field, value):
    descriptor[field] = value
    with pytest.raises(ValueError):
        evaluation.validate_adapter_descriptor(descriptor)


def test_different_installed_sdk_is_not_accepted_by_label_alone(descriptor):
    descriptor["sdk_source_sha256"]["tabpfn/utils.py"] = "0" * 64
    with pytest.raises(ValueError, match="sdk_source_sha256"):
        evaluation.validate_adapter_descriptor(descriptor)


@pytest.mark.parametrize("field,value", [
    ("variant", "upstream_unmodified"),
    ("schema", "posteriorforge.inference.v1"),
    ("status", "running"),
    ("requested_count", 10000),
    ("scientific_run", False),
    ("observation_preprocessing", "whitened"),
    ("context_size", 1024),
    ("context_seed", 20261112),
    ("case", 1),
    ("metadata_writer_restored", False),
    ("explicit_variant_activation", False),
    ("tail_family_changed", True),
    ("execution_optimized", True),
    ("expected_patched_call_count", 1007),
    ("expected_patched_call_count", True),
])
def test_wrong_cohort_or_statistical_changes_fail_before_assessment(manifest, field, value):
    manifest[field] = value
    with pytest.raises(ValueError):
        evaluation.validate_pilot_header(manifest, seed=20261111, case=0, context_size=4096)


def test_complete_new_schema_is_distinct_from_original_gate(manifest):
    evaluation.validate_pilot_header(manifest, seed=20261111, case=0, context_size=4096)
    assert manifest["schema"] != "posteriorforge.inference.v1"


def test_samples_payload_hash_is_checked_before_loading(tmp_path):
    payload = tmp_path / "samples.npz"
    payload.write_bytes(b"not a posterior payload")
    with pytest.raises(ValueError, match="payload SHA256"):
        evaluation.load_samples(tmp_path / "manifest.json",
                                {"samples_file": "samples.npz", "samples_sha256": "0" * 64})


def test_path_traversal_cannot_select_an_unbound_original_run(tmp_path):
    with pytest.raises(ValueError, match="relative artifact"):
        evaluation.within(tmp_path, "../other-run/manifest.json")



def test_claimed_calls_must_cover_all_rejection_batches(manifest):
    manifest["rejection"]["support_calls"] = 7
    with pytest.raises(ValueError, match="every ensemble/stage"):
        evaluation.validate_pilot_header(manifest, seed=20261111, case=0, context_size=4096)


def test_positive_counter_alone_cannot_hide_partial_activation(manifest):
    manifest["compatibility_adapter"]["patched_call_count"] = 1
    with pytest.raises(ValueError, match="every ensemble/stage"):
        evaluation.validate_pilot_header(manifest, seed=20261111, case=0, context_size=4096)


def test_original_execution_source_is_independently_authenticated(manifest):
    manifest["baseline_execution_source"]["baseline_inference_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="execution-source"):
        evaluation.validate_pilot_header(manifest, seed=20261111, case=0, context_size=4096)


@pytest.mark.parametrize("reference,field", [
    (reference, field) for reference in ("parent", "refined_grid")
    for field in ("manifest_sha256", "distribution_sha256", "cdf_precision", "binding")
])
def test_run_cannot_silently_use_another_reference(reference, field):
    expected = {
        name: {"manifest_sha256": name + "-manifest", "distribution_sha256": name + "-payload",
               "cdf_precision": .001, "binding": {"scientific_fingerprint": "fixture-only"}}
        for name in ("parent", "refined_grid")
    }
    manifest = {"reference_binding": copy.deepcopy(expected)}
    evaluation.validate_run_reference_binding(manifest, {"runner_reference_binding": expected})
    manifest["reference_binding"][reference][field] = "wrong"
    with pytest.raises(ValueError, match="reference binding"):
        evaluation.validate_run_reference_binding(manifest, {"runner_reference_binding": expected})
