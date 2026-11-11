"""Performance-first evaluator tests; synthetic arrays here are unit fixtures only."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.stats import norm

from posteriorforge.artifacts import sha256_file
from posteriorforge.banks import derive_seed
from posteriorforge.inference import (CHECKPOINT_SHA256, UPSTREAM_REVISION,
                                     frozen_regressor_settings, upstream_source_hashes)
from posteriorforge.science.analytical_benchmark_v2 import array_descriptor, build_problem

PATH = Path(__file__).resolve().parents[1]/"scripts/evaluate_execution_variant.py"
spec = importlib.util.spec_from_file_location("execution_variant_review_tests", PATH)
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


@pytest.fixture
def cohort():
    rng = np.random.default_rng(402)
    values = rng.normal(size=(256, 21))
    noise = [build_problem().parameter_order.index(name) for name in build_problem().prior.noise_names]
    values[:, noise] = rng.uniform(.12, .38, size=(len(values), 6))
    values[:, 0] += 1.0  # Explicit physical failure; identical cohorts still have no relative drift.
    return values


@pytest.fixture
def reference():
    rng = np.random.default_rng(881)
    joint = rng.normal(size=(32, 21))
    joint /= np.linalg.norm(joint, axis=1, keepdims=True)
    directions = np.concatenate((np.eye(21), joint))
    mean, sd = np.zeros(21), np.ones(21)
    noise = [build_problem().parameter_order.index(name) for name in build_problem().prior.noise_names]
    mean[noise] = .25
    sd[noise] = .26/np.sqrt(12)
    queries = np.linspace(-8, 8, 513)
    cdfs = norm.cdf((queries[None, :] - (directions@mean)[:, None]) /
                    np.sqrt((directions**2)@(sd**2))[:, None])
    qualified = {"state": "qualified", "precision": .001, "moment_gate_passed": True,
                 "witness_gate_passed": True, "independent_nested_passed": True,
                 "cpu_check_passed": True, "joint_reference_passed": True,
                 "moment_precision": {"passed": True, "mean_error_reference_sd": [0.]*21,
                                      "sd_error_reference_sd": [0.]*21}}
    return {"directions": directions, "queries": queries, "cdfs": cdfs,
            "mean": mean, "sd": sd, "qualification": qualified}


def test_physical_failure_does_not_veto_identical_execution_cohort(cohort, reference):
    result = review.comparative_assessment(cohort, cohort.copy(), reference, expected_count=len(cohort))
    assert result["sample_integrity"]["passed"]
    assert not result["absolute_physical_assessment"]["passed"]
    assert result["numeric_review_status"] == "no_empirical_drift_flag"
    relative = result["relative_distribution_review"]
    assert not relative["absolute_gate_still_required"]
    assert relative["original_api_absolute_gate_policy"] is True
    assert relative["thresholds_are_investigation_flags"]
    assert not relative["cdf_is_reference_grid_bound"]
    assert not relative["population_confidence_claim"]
    assert relative["max_cdf_difference"] == 0


def test_mean_shift_is_investigation_even_while_physical_gate_remains_separate(cohort, reference):
    candidate = cohort.copy()
    candidate[:, 0] += .08
    result = review.comparative_assessment(candidate, cohort, reference, expected_count=len(cohort))
    assert result["numeric_review_status"] == "distribution_investigation_required"
    assert 0 in result["relative_distribution_review"]["flags"]["mean_reference_sd"]
    assert result["relative_distribution_review"]["max_mean_difference_reference_sd"] == pytest.approx(.08)


def test_relative_sd_change_has_its_own_investigation_flag(cohort, reference):
    candidate = cohort.copy()
    candidate[:, 1] *= 1.08
    result = review.comparative_assessment(candidate, cohort, reference, expected_count=len(cohort))
    assert 1 in result["relative_distribution_review"]["flags"]["sd_fraction"]


@pytest.mark.parametrize("kind", ["short", "wrong_dimension", "nonfinite", "unsupported"])
def test_invalid_complete_vector_cohort_never_reaches_relative_review(cohort, reference, kind):
    candidate = cohort.copy()
    if kind == "short": candidate = candidate[:-1]
    if kind == "wrong_dimension": candidate = candidate[:, :-1]
    if kind == "nonfinite": candidate[0, 0] = np.nan
    if kind == "unsupported": candidate[0, 6] = .5001
    result = review.comparative_assessment(candidate, cohort, reference, expected_count=len(cohort))
    assert result["numeric_review_status"] == "invalid_samples"
    assert result["relative_distribution_review"] is None
    assert not result["absolute_physical_assessment"]["passed"]


def test_corrupt_control_is_not_a_valid_relative_comparator(cohort, reference):
    control = cohort.copy()
    control[0, 6] = .001
    with pytest.raises(ValueError, match="corrected control"):
        review.comparative_assessment(cohort, control, reference, expected_count=len(cohort))


def test_rejection_final_trim_is_visible_but_not_automatically_invalid():
    value = review.rejection_review({"support_calls": 6, "proposed": 60000,
                                    "accepted_before_final_trim": 51000, "returned": 50000},
                                   returned=50000)
    assert not value["structural_oddities"]
    assert value["final_trim_count"] == 1000
    assert value["proposal_support_acceptance_fraction"] == .85


@pytest.mark.parametrize("rejection", [
    None, {"support_calls": True},
    {"support_calls": 1, "proposed": 49000, "accepted_before_final_trim": 50000, "returned": 50000},
    {"support_calls": 0, "proposed": 50000, "accepted_before_final_trim": 50000, "returned": 50000},
])
def test_rejection_telemetry_oddities_are_not_silently_discarded(rejection):
    assert review.rejection_review(rejection, returned=50000)["structural_oddities"]


@pytest.fixture
def header(monkeypatch):
    context = SimpleNamespace(theta=np.zeros((4096, 21)), obs=np.zeros((4096, 33)))
    query = np.zeros((1, 33))
    settings = frozen_regressor_settings("unit-fixture.ckpt")
    manifest = {
        "schema": review.RUN_SCHEMA, "variant": "execution_cache_v1", "status": "complete",
        "context_seed": 20261111, "case": 0, "context_size": 4096, "requested_count": 50000,
        "scientific_run": True, "performance_first_authorized": True,
        "absolute_accuracy_gate_required_for_performance": False, "tail_family_changed": False,
        "observation_preprocessing": "raw; TabPFN frozen preprocessing", "log_density": "not_used",
        "scientific_fingerprint": build_problem().scientific_fingerprint,
        "parameter_order": list(build_problem().parameter_order),
        "context_theta": array_descriptor(context.theta), "context_obs": array_descriptor(context.obs),
        "query": array_descriptor(query), "checkpoint_sha256": CHECKPOINT_SHA256,
        "upstream_revision": UPSTREAM_REVISION, "upstream_files": upstream_source_hashes(),
        "context_filter": "no_filtering", "max_iter_rejection": None, "microbatch": 10000,
        "seed": derive_seed(20261111, "posterior", case=0), "repetition": 0, "rank": 0, "world_size": 1,
        "regressor_settings": dict(settings), "compatibility_adapter": {"unit_fixture": True},
        "sampling_rng_policy": {"policy_id": "unit_fixture_isolated_sampler_v1"},
        "model": {"dtype": "torch.float32", "autocast": False},
    }
    monkeypatch.setattr(review.bindings, "validate_adapter_descriptor", lambda d: {"patched_call_count": 1})
    return manifest, context, query, {"regressor_settings": settings}


def test_only_authorized_fit_mode_is_normalized_for_original_binding(header):
    manifest, context, query, control = header
    manifest["regressor_settings"]["fit_mode"] = "fit_with_cache"
    review.validate_execution_header(manifest, context=context, query=query, control=control)
    assert manifest["regressor_settings"]["fit_mode"] == "fit_with_cache"  # No mutation of actual record.


@pytest.mark.parametrize("key,value", [
    ("softmax_temperature", .5), ("n_estimators", 1), ("inference_precision", "torch.bfloat16"),
])
def test_scientific_or_precision_setting_change_is_not_hidden_as_execution_only(header, key, value):
    manifest, context, query, control = header
    manifest["regressor_settings"][key] = value
    with pytest.raises(ValueError):
        review.validate_execution_header(manifest, context=context, query=query, control=control)


@pytest.mark.parametrize("key,value", [
    ("performance_first_authorized", False), ("absolute_accuracy_gate_required_for_performance", True),
    ("tail_family_changed", True), ("requested_count", 10000), ("observation_preprocessing", "sufficient statistics"),
])
def test_policy_and_raw_observation_contract_is_explicit(header, key, value):
    manifest, context, query, control = header
    manifest[key] = value
    with pytest.raises(ValueError):
        review.validate_execution_header(manifest, context=context, query=query, control=control)


@pytest.fixture
def qualification_case(tmp_path, header):
    from posteriorforge.execution_cache import execution_metadata
    manifest, context, query, _ = header
    manifest["execution_engine"] = execution_metadata()
    coverage = [0, 6, 14, 20]
    rows = [{"stage_index": j, "passed": True, "stress_comparison": {"passed": True},
             "comparisons": [{"query_rows": n, "cache_vs_same_chunk_control": {"passed": True},
                              "control_chunk_vs_1024": {"passed": True}} for n in (1, 16, 128, 1024)],
             "baseline_model_contract_after": {"input_normalization_train_only_verified": True},
             "cache_model_contract_after": {"input_normalization_train_only_verified": True}}
            for j in coverage]
    q = {"schema": review.QUALIFICATION_SCHEMA, "qualified": True, "status": "screen_pass",
         "engine_id": manifest["execution_engine"]["engine_id"],
         "module_sha256": manifest["execution_engine"]["module_sha256"],
         "control_variant": "cdf_arithmetic_v1", "stage_coverage": coverage,
         "requested_stages": coverage, "full21_qualified": False, "stage_results": rows,
         "thresholds": dict(review.QUALIFICATION_THRESHOLDS), "row_counts": [1,16,128,1024],
         "sources_unchanged": True, "prefix_manifest_sha256": review.CONTROL_MANIFEST_SHA256,
         "prefix_samples_sha256": review.CONTROL_SAMPLES_SHA256,
         "sources": {"qualification_script_sha256": sha256_file(PATH.with_name("qualify_execution_cache.py"))},
         "compatibility_adapter": {"unit_fixture": True}}
    for key in ("checkpoint_sha256", "scientific_fingerprint", "parameter_order", "context_theta", "context_obs", "query"):
        q[key] = manifest[key]
    path = tmp_path/"qualification.json"
    def save():
        path.write_text(json.dumps(q))
        manifest["qualification_manifest_sha256"] = sha256_file(path)
    save()
    return q, manifest, context, query, path, save


def test_four_stage_screen_cannot_be_upgraded_to_full21(qualification_case):
    q, manifest, context, query, path, save = qualification_case
    result = review.qualification_review(path, manifest, context=context, query=query)
    assert not result["full21_qualified"]
    assert result["scope"] == "coverage_limited_fixed_conditional_screen"
    q["full21_qualified"] = True
    save()
    with pytest.raises(ValueError, match="coverage claim"):
        review.qualification_review(path, manifest, context=context, query=query)


@pytest.mark.parametrize("mutation", ["threshold", "failed_stage", "normalization", "source", "prefix"])
def test_source_or_numeric_qualification_cannot_be_silently_weakened(qualification_case, mutation):
    q, manifest, context, query, path, save = qualification_case
    if mutation == "threshold": q["thresholds"]["max_cdf_absolute"] = .01
    if mutation == "failed_stage": q["stage_results"][0]["passed"] = False
    if mutation == "normalization": q["stage_results"][0]["cache_model_contract_after"]["input_normalization_train_only_verified"] = False
    if mutation == "source": q["sources"]["qualification_script_sha256"] = "0"*64
    if mutation == "prefix": q["prefix_manifest_sha256"] = "0"*64
    save()
    with pytest.raises(ValueError):
        review.qualification_review(path, manifest, context=context, query=query)


def test_qualification_payload_sha_is_authenticated(qualification_case):
    q, manifest, context, query, path, save = qualification_case
    path.write_text(path.read_text()+" ")
    with pytest.raises(ValueError, match="SHA256"):
        review.qualification_review(path, manifest, context=context, query=query)


def test_engine_configuration_and_sdk_pins_are_bound():
    from posteriorforge.execution_cache import execution_metadata
    engine = execution_metadata()
    review.validate_engine_metadata(engine)
    engine["config"]["normalize_on_train_only"] = False
    with pytest.raises(ValueError, match="configuration invariant"):
        review.validate_engine_metadata(engine)
