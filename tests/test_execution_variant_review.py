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
    from posteriorforge.inference_cdf import guard_baseline_source
    from posteriorforge import execution_cache
    from posteriorforge.performance_runner import STREAM_ROLES, STREAM_DTYPES
    stream_seeds = {key: derive_seed(20261111, role, case=0, rank=0, repetition=0)
                    for key, role in STREAM_ROLES.items()}
    streams = {key: {"role": STREAM_ROLES[key], "seed": seed, "dtype": STREAM_DTYPES[key], "device": "cpu"}
               for key, seed in stream_seeds.items()}
    manifest = {
        "schema": review.RUN_SCHEMA, "variant": "execution_cache_v1", "status": "complete",
        "context_seed": 20261111, "case": 0, "context_size": 4096, "requested_count": 50000,
        "scientific_run": True, "performance_first_authorized": True,
        "end_to_end": True, "run_purpose": "performance_execution", "sources_unchanged": True,
        "partial_vector_filtering": False, "clipping_or_repair_performed": False,
        "rank": 0, "world_size": 1, "completed_stage_indices": list(range(21)),
        "baseline_execution_source": guard_baseline_source(),
        "runner_source_sha256": sha256_file(PATH.parents[1]/"src/posteriorforge/performance_runner.py"),
        "execution_engine": execution_cache.execution_metadata(), "seed_streams": stream_seeds,
        "expected_patched_call_count": 1, "query_batch_size": 1024,
        "absolute_accuracy_gate_required_for_performance": False, "tail_family_changed": False,
        "observation_preprocessing": "raw; TabPFN frozen preprocessing", "log_density": "not_used",
        "scientific_fingerprint": build_problem().scientific_fingerprint,
        "parameter_order": list(build_problem().parameter_order),
        "context_theta": array_descriptor(context.theta), "context_obs": array_descriptor(context.obs),
        "query": array_descriptor(query), "checkpoint_sha256": CHECKPOINT_SHA256,
        "upstream_revision": UPSTREAM_REVISION, "upstream_files": upstream_source_hashes(),
        "context_filter": "no_filtering", "max_iter_rejection": None, "microbatch": 50000,
        "seed": stream_seeds["component_uniform"], "repetition": 0, "rank": 0, "world_size": 1,
         "regressor_settings": {**settings, "fit_mode": "fit_with_cache"}, "compatibility_adapter": {"unit_fixture": True},
        "sampling_rng_policy": "independent_cpu_sampling_streams_model_rng_isolated",
        "sampling_policy": {"head_policy": "not_used", "mass_envelope": None, "streams": streams,
                            "stage0_dedup": {"enabled": False}},
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
    inverse = {"icdf_contract_version": 2, "passed": True,
               "same_logits_and_borders_exact_failures": 0, "independent_fp32_replay_failures": 0}
    passed_comparison = {"passed": True, "inverse_cdf": inverse}
    rows = [{"stage_index": j, "passed": True, "stress_comparison": copy.deepcopy(passed_comparison),
             "comparisons": [{"query_rows": n, "cache_vs_same_chunk_control": copy.deepcopy(passed_comparison),
                              "control_chunk_vs_1024": copy.deepcopy(passed_comparison)} for n in (1, 16, 128, 1024)],
             "baseline_model_contract_after": {"input_normalization_train_only_verified": True},
             "cache_model_contract_after": {"input_normalization_train_only_verified": True}}
            for j in coverage]
    q = {"schema": review.QUALIFICATION_SCHEMA, "qualified": True, "status": "screen_pass",
         "engine_id": manifest["execution_engine"]["engine_id"],
         "module_sha256": manifest["execution_engine"]["module_sha256"],
         "control_variant": "cdf_arithmetic_v1", "icdf_contract_version": 2, "stage_coverage": coverage,
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


@pytest.fixture
def sampling_policy():
    from posteriorforge import execution_mixture as mixture
    return {**mixture.execution_metadata(), "head_policy": "raw_mass_rejection_v1",
            "mass_envelope": mixture.HEAD_MASS_ENVELOPE,
            "stage0_dedup": {"enabled": True, "scope": "full_mixture_only", "component_distribution_dedup": False},
            "streams": {
                 "head_proposal": {"role": "execution-head-proposal", "seed": derive_seed(20261111,"execution-head-proposal",case=0,rank=0,repetition=0), "dtype": "int64", "device": "cpu"},
                "head_acceptance": {"role": "execution-head-acceptance", "seed": derive_seed(20261111,"execution-head-acceptance",case=0,rank=0,repetition=0), "dtype": "float64", "device": "cpu"},
                "component_uniform": {"role": "execution-component-uniform", "seed": derive_seed(20261111,"execution-component-uniform",case=0,rank=0,repetition=0), "dtype": "float32", "device": "cpu"}}}


def test_explicit_mass_corrected_independent_stream_policy_is_accepted(sampling_policy):
    review.validate_sampling_policy(sampling_policy)


@pytest.mark.parametrize("mutation", ["uniform_heads", "envelope", "shared_seed", "wrong_dtype", "component_dedup", "source"])
def test_biased_or_unqualified_mixture_policy_is_rejected(sampling_policy, mutation):
    if mutation == "uniform_heads": sampling_policy["head_policy"] = "uniform_normalized_components"
    if mutation == "envelope": sampling_policy["mass_envelope"] = 1.1
    if mutation == "shared_seed": sampling_policy["streams"]["component_uniform"]["seed"] = sampling_policy["streams"]["head_acceptance"]["seed"]
    if mutation == "wrong_dtype": sampling_policy["streams"]["head_acceptance"]["dtype"] = "float32"
    if mutation == "component_dedup": sampling_policy["stage0_dedup"]["component_distribution_dedup"] = True
    if mutation == "source": sampling_policy["module_sha256"] = "0"*64
    with pytest.raises(ValueError):
        review.validate_sampling_policy(sampling_policy)


def test_mixture_header_requires_its_own_policy_not_cache_rng_label(header, sampling_policy):
    manifest, context, query, control = header
    manifest["variant"] = "execution_mixture_v1"
    del manifest["sampling_policy"]
    with pytest.raises(ValueError, match="sampling"):
        review.validate_execution_header(manifest, context=context, query=query, control=control)
    manifest["sampling_policy"] = sampling_policy
    review.validate_execution_header(manifest, context=context, query=query, control=control)


def test_old_icdf_value_contract_cannot_be_relabelled_as_v2(qualification_case):
    q, manifest, context, query, path, save = qualification_case
    q["icdf_contract_version"] = 1
    save()
    with pytest.raises(ValueError, match="version2"):
        review.qualification_review(path, manifest, context=context, query=query)


def test_each_base_comparison_must_include_independent_v2_replay(qualification_case):
    q, manifest, context, query, path, save = qualification_case
    q["stage_results"][0]["comparisons"][0]["cache_vs_same_chunk_control"]["inverse_cdf"]["independent_fp32_replay_failures"] = 1
    save()
    with pytest.raises(ValueError, match="ICDFv2"):
        review.qualification_review(path, manifest, context=context, query=query)


@pytest.fixture
def full_component_proof(qualification_case):
    from posteriorforge import execution_mixture as mixture
    q, manifest, context, query, path, save = qualification_case
    manifest["variant"] = "execution_mixture_v1"
    q["stage_coverage"] = list(range(21))
    q["requested_stages"] = list(range(21))
    q.update(full21_qualified=True, full21_component_qualified=True, full21_sampler_qualified=True,
             component_qualified=True, component_stage_coverage=list(range(21)),
             mixture_module_sha256=mixture.execution_source_sha256())
    row_template = copy.deepcopy(q["stage_results"][0])
    component = {
        "schema": "posteriorforge.execution_component_qualification.v1", "passed": True,
        "module_sha256": mixture.execution_source_sha256(), "engine_id": mixture.ENGINE_ID,
        "fast_icdf_same_logits_failures": 0, "uniform_head_without_mass_correction_is_not_authorized": True,
        "executor_metadata": {"full8_member_lists_restored": True},
        "roundoff": {"component_mass_guard_passed": True, "mass_corrected_reconstruction_passed": True},
        "normalized_mean_captured_raw_vs_full_fp32": {"passed": True},
        "normalized_mean_observed_raw_vs_full_fp32": {"passed": True},
        "mass_corrected_observed_outputs_vs_full_fp32": {"passed": True},
        "stage0_dedup_scope": "full_mixture_only", "component_distribution_dedup_qualified": False,
        "stage0_shared_distribution": {"passed": True, "shared_icdf_same_logits_exact": True},
        "members": [
            {"ensemble_index": i, "passed": True, "configuration_identity_equal": True,
             "exact_borders": True, "float32": True, "component_raw_mass_matches_fp64_sum": True,
             "normalized_component": {"passed": True}, "corrected_cdf_call_delta": 1,
             "executed_corrected_helper_fingerprint_matches": True, "fast_icdf_same_logits_exact": True,
             "component_head_mass_envelope": mixture.HEAD_MASS_ENVELOPE}
            for i in range(8)],
    }
    q["stage_results"] = [{**copy.deepcopy(row_template), "stage_index": i,
                            "component_qualification": copy.deepcopy(component)} for i in range(21)]
    save()
    return q, manifest, context, query, path, save


def test_complete_mixture_proof_does_not_compare_selected_head_to_full_mixture(full_component_proof):
    q, manifest, context, query, path, save = full_component_proof
    result = review.qualification_review(path, manifest, context=context, query=query)
    assert result["full21_qualified"]
    assert result["component_and_sampler_proof"]["full21_component_qualified"]
    assert result["component_and_sampler_proof"]["selected_head_is_not_compared_as_the_full_mixture"]


@pytest.mark.parametrize("mutation", [
    "partial", "sampler_unqualified", "stale_source", "stale_translator", "missing_head",
    "no_list_restore", "mass", "raw_reconstruction", "actual_api_reconstruction", "unqualified_shared_sampler",
])
def test_mixture_cannot_omit_component_sampler_or_reconstruction_proof(full_component_proof, mutation):
    q, manifest, context, query, path, save = full_component_proof
    component = q["stage_results"][0]["component_qualification"]
    if mutation == "partial": q["full21_component_qualified"] = False
    if mutation == "sampler_unqualified": q["full21_sampler_qualified"] = False
    if mutation == "stale_source": q["mixture_module_sha256"] = "0"*64
    if mutation == "stale_translator": component["members"][0]["executed_corrected_helper_fingerprint_matches"] = False
    if mutation == "missing_head": component["members"] = component["members"][:-1]
    if mutation == "no_list_restore": component["executor_metadata"]["full8_member_lists_restored"] = False
    if mutation == "mass": component["roundoff"]["component_mass_guard_passed"] = False
    if mutation == "raw_reconstruction": component["normalized_mean_observed_raw_vs_full_fp32"]["passed"] = False
    if mutation == "actual_api_reconstruction": component["mass_corrected_observed_outputs_vs_full_fp32"]["passed"] = False
    if mutation == "unqualified_shared_sampler": component["stage0_shared_distribution"]["shared_icdf_same_logits_exact"] = False
    save()
    with pytest.raises(ValueError):
        review.qualification_review(path, manifest, context=context, query=query)


def test_actual_runner_private_seed_and_wave_metadata_bind_without_claiming_old_rng_identity(header):
    manifest, context, query, control = header
    assert manifest["seed"] != derive_seed(20261111, "posterior", case=0)
    assert manifest["microbatch"] == 50000
    review.validate_execution_header(manifest, context=context, query=query, control=control)


def test_50000_row_worker_cannot_masquerade_as_end_to_end_run(header):
    manifest, context, query, control = header
    manifest.update(end_to_end=False, run_purpose="performance_worker", scientific_run=False)
    with pytest.raises(ValueError):
        review.validate_execution_header(manifest, context=context, query=query, control=control)


def test_stream_seed_label_cannot_hide_different_actual_seed(header):
    manifest, context, query, control = header
    manifest["seed"] += 1
    with pytest.raises(ValueError, match="seed binding"):
        review.validate_execution_header(manifest, context=context, query=query, control=control)


def test_global_stage0_memo_is_not_qualified_by_wave_local_proof(sampling_policy):
    sampling_policy["stage0_dedup"]["density_scope"] = "global case0 cache"
    with pytest.raises(ValueError, match="cross-request"):
        review.validate_sampling_policy(sampling_policy)


@pytest.fixture
def assembled_hybrid_case(tmp_path,monkeypatch,header,sampling_policy):
    # Reuse the assembler's explicitly synthetic protocol fixtures; no model or
    # posterior cohort is generated, and no fixture is exported as run evidence.
    fixture_path=PATH.parents[1]/"tests/test_hybrid_qualification_assembly.py"
    fixture_spec=importlib.util.spec_from_file_location("hybrid_protocol_fixture_for_evaluator",fixture_path)
    fixture_module=importlib.util.module_from_spec(fixture_spec);fixture_spec.loader.exec_module(fixture_module)
    actual_sources=fixture_module.assembly.current_sources()
    monkeypatch.setattr(fixture_module,"SOURCES",actual_sources)
    cached,hybrids,fallback=fixture_module.fixture(tmp_path/"producer-fixtures",monkeypatch)
    manifest,context,query,control=header
    for proof_path in [cached,*hybrids]:
        value=json.loads(proof_path.read_text())
        for key in ("checkpoint_sha256","scientific_fingerprint","parameter_order","context_theta","context_obs","query"):
            value[key]=manifest[key]
        value.update(prefix_manifest_sha256=review.CONTROL_MANIFEST_SHA256,
                     prefix_samples_sha256=review.CONTROL_SAMPLES_SHA256,
                     fixed_prefix_row_indices=np.linspace(0,49999,1024,dtype=int).tolist())
        proof_path.write_text(json.dumps(value))
    output=tmp_path/"assembled"
    fixture_module.assembly.assemble(cache_proofs=[cached],hybrid_proofs=hybrids,fallback_map=fallback,output=output)
    from posteriorforge import execution_hybrid
    manifest.update(variant="execution_hybrid_v1",mode="component_hybrid",sampling_policy=sampling_policy,
                    hybrid_execution=execution_hybrid.execution_metadata(fallback))
    path=output/"manifest.json";manifest["qualification_manifest_sha256"]=sha256_file(path)
    return manifest,context,query,control,path


def test_hybrid_metadata_end_to_end_preflight_preserves_failed_cache_statuses(assembled_hybrid_case):
    manifest,context,query,control,path=assembled_hybrid_case
    review.validate_execution_header(manifest,context=context,query=query,control=control)
    result=review.qualification_review(path,manifest,context=context,query=query)
    assert result["full21_hybrid_qualified"]
    assert result["full21_qualified"] is False
    assert result["failed_cache_producer_statuses_preserved"]
    assert result["qualification"]["source_inputs"][0]["producer_status"]=="conditional_mismatch"
    assert result["qualification"]["source_inputs"][0]["producer_qualified"] is False


@pytest.mark.parametrize("issue",["cache_upgrade","fallback_policy","numeric","archive_status","hybrid_source"])
def test_hybrid_assembly_cannot_hide_or_reclassify_source_failures(assembled_hybrid_case,issue):
    manifest,context,query,control,path=assembled_hybrid_case
    value=json.loads(path.read_text())
    if issue=="cache_upgrade":value["full21_qualified"]=True
    if issue=="fallback_policy":value["fallback_policy_sha256"]="0"*64
    if issue=="numeric":
        value["stage_results"][0]["comparisons"][0]["cache_vs_same_chunk_control"]["final"]["max_cdf_absolute_error"]=.01
    if issue=="hybrid_source":value["hybrid_module_sha256"]="0"*64
    if issue=="archive_status":
        archive=path.parent/value["source_inputs"][0]["archived_manifest"]
        original=json.loads(archive.read_text());original["status"]="screen_pass";archive.write_text(json.dumps(original))
    path.write_text(json.dumps(value));manifest["qualification_manifest_sha256"]=sha256_file(path)
    with pytest.raises(ValueError):
        review.qualification_review(path,manifest,context=context,query=query)


@pytest.fixture
def aggregate_case(tmp_path,monkeypatch,header,sampling_policy):
    # Synthetic 50k-shaped custody fixtures only. No model, speed benchmark,
    # physical-gate verdict or posterior evidence is generated here.
    worker_test=PATH.parents[1]/"tests/test_execution_workers.py"
    fixture_spec=importlib.util.spec_from_file_location("worker_protocol_fixture_for_review",worker_test)
    wf=importlib.util.module_from_spec(fixture_spec);fixture_spec.loader.exec_module(wf)
    root=tmp_path/"aggregate";root.mkdir()
    contracts,paths=wf._payloads(root,mode="component_hybrid")
    manifest,context,query,control=header
    from posteriorforge import execution_cache,execution_hybrid,performance_runner
    custody={"runner_reference_binding":{"parent":{"manifest_sha256":"synthetic-parent"},
                                          "refined_grid":{"manifest_sha256":"synthetic-grid"}}}
    proof=tmp_path/"qualification-only-fixture.json";proof.write_text("protocol fixture, not a real proof")
    qsha=sha256_file(proof)
    for rank,(contract,path) in enumerate(zip(contracts,paths)):
        contract.update(runner_source_sha256=sha256_file(performance_runner.__file__),
                        orchestrator_source_sha256=sha256_file(PATH.with_name("run_execution_workers.py")))
        path.write_text(json.dumps(contract))
        child_path=Path(contract["expected_output"])/"manifest.json"
        payload_manifest=json.loads(child_path.read_text())
        descriptor=payload_manifest["compatibility_adapter"]
        child=copy.deepcopy(manifest)
        child.update(schema=review.RUN_SCHEMA,variant="execution_hybrid_v1",mode="component_hybrid",
            rank=rank,world_size=2,requested_count=25000,scientific_run=False,
            end_to_end=False,run_purpose="performance_worker",benchmark_total_count=50000,
            worker_contract_sha256=sha256_file(path),qualification_full21=True,
            qualification_manifest_sha256=qsha,reference_binding=custody["runner_reference_binding"],
            execution_engine={**execution_cache.execution_metadata(),"query_batch_size":1024,"posterior_count":25000,
                              "sampling_rng_policy":"independent_cpu_sampling_streams_model_rng_isolated"},
            hybrid_execution=execution_hybrid.execution_metadata({10:[5],13:[6]}),
            compatibility_adapter=descriptor,expected_patched_call_count=1,
            timings={"posterior_request_seconds":1.0},memory={"peak_allocated_bytes":1,"peak_reserved_bytes":1},
            rejection={"support_calls":1,"proposed":25000,"accepted_before_final_trim":25000,"returned":25000})
        child["sampling_policy"]=copy.deepcopy(sampling_policy)
        for name,seed in contract["seed_streams"].items():child["sampling_policy"]["streams"][name]["seed"]=seed
        child["seed_streams"]=contract["seed_streams"];child["seed"]=contract["seed_streams"]["component_uniform"]
        child["samples"]=payload_manifest["samples"];child["samples_sha256"]=payload_manifest["samples_sha256"]
        child_path.write_text(json.dumps(child))
    expected={key:control[key] for key in ("regressor_settings",)}
    # The real control metadata has these bindings; fixture control is deliberately
    # minimal, so copy the same unit-fixture scientific/header identities.
    control.update({key:manifest[key] for key in ("scientific_fingerprint","parameter_order","context_seed","case",
                   "context_size","context_theta","context_obs","query","checkpoint_sha256","baseline_execution_source")})
    bindings={key:control[key] for key in ("scientific_fingerprint","parameter_order","context_seed","case","context_size",
                                         "context_theta","context_obs","query","checkpoint_sha256","baseline_execution_source")}
    bindings.update(mode="component_hybrid",variant="execution_hybrid_v1",reference_binding=custody["runner_reference_binding"])
    combined,records=wf.workers.validate_worker_payloads(contracts,paths,expected_bindings=bindings,
            driver_schema=review.RUN_SCHEMA,qualification_sha256=qsha)
    with (root/"samples.npz").open("wb") as stream:np.savez(stream,samples=combined)
    m={**bindings,"schema":"posteriorforge.inference.execution.aggregate.v1","variant":"execution_workers_v1",
       "worker_execution_variant":"execution_hybrid_v1","status":"complete","requested_count":50000,"returned_count":50000,
       "world_size":2,"rank":None,"end_to_end":True,"run_purpose":"performance_e2e_aggregate",
       "aggregate_published":True,"worker_counts":[25000,25000],"worker_return_codes":[0,0],
       "budget":{"job_id":"123","reservation_id":"reservation"},
       "orchestrator_source_sha256":sha256_file(PATH.with_name("run_execution_workers.py")),
       "runner_source_sha256":sha256_file(performance_runner.__file__),"qualification_manifest_sha256":qsha,
       "worker_contracts":[{"path":str(path),"sha256":sha256_file(path)} for path in paths],
       "rank_seed_streams":[contract["seed_streams"] for contract in contracts],
       "visible_device_tokens":["0","1"],"workers":records,
       "samples":array_descriptor(combined),"samples_sha256":sha256_file(root/"samples.npz"),
       "execution_engine":{**execution_cache.execution_metadata(),"query_batch_size":1024,"posterior_count":50000,
                           "sampling_rng_policy":"independent_cpu_sampling_streams_model_rng_isolated"},
       "hybrid_execution":execution_hybrid.execution_metadata({10:[5],13:[6]})}
    path=root/"manifest.json";path.write_text(json.dumps(m))
    return path,m,context,query,control,custody,proof,combined


def test_aggregate_authenticates_exact_complete_rank_order_and_private_worker_streams(aggregate_case):
    path,m,context,query,control,custody,proof,expected=aggregate_case
    values,effective,detail=review.authenticate_worker_aggregate(path,m,context=context,query=query,control=control,
                                                               custody=custody,qualification_path=proof)
    np.testing.assert_array_equal(values,expected)
    assert detail["rank_order_concatenation_exact"]
    assert detail["summed_rejection"]["returned"]==50000
    assert effective["variant"]=="execution_hybrid_v1"
    assert effective["end_to_end"] is False


@pytest.mark.parametrize("issue",["rankorder","seed_record","partial_worker","device","contract"])
def test_aggregate_cannot_hide_incomplete_reordered_or_misbound_worker_data(aggregate_case,issue):
    path,m,context,query,control,custody,proof,expected=aggregate_case
    if issue=="rankorder":
        swapped=np.concatenate((expected[25000:],expected[:25000]))
        with (path.parent/"samples.npz").open("wb") as stream:np.savez(stream,samples=swapped)
        m["samples"]=array_descriptor(swapped);m["samples_sha256"]=sha256_file(path.parent/"samples.npz")
    if issue=="seed_record":m["rank_seed_streams"][1]["component_uniform"]+=1
    if issue=="partial_worker":
        child=path.parent/"worker-1/manifest.json";value=json.loads(child.read_text());value["completed_stage_indices"]=list(range(20));child.write_text(json.dumps(value))
    if issue=="device":m["visible_device_tokens"]=["0","0"]
    if issue=="contract":m["worker_contracts"][0]["sha256"]="0"*64
    with pytest.raises(ValueError):
        review.authenticate_worker_aggregate(path,m,context=context,query=query,control=control,
                                             custody=custody,qualification_path=proof)


def test_actual_driver_can_bind_guarded_upstream_identity_without_redundant_top_fields(header):
    manifest,context,query,control=header
    del manifest["upstream_revision"];del manifest["upstream_files"]
    review.validate_execution_header(manifest,context=context,query=query,control=control)
    assert "upstream_revision" not in manifest and "upstream_files" not in manifest


def test_supplied_wrong_upstream_identity_is_never_overridden_by_supplement(header):
    manifest,context,query,control=header
    manifest["upstream_revision"]="incorrect"
    with pytest.raises(ValueError,match="upstream_revision"):
        review.validate_execution_header(manifest,context=context,query=query,control=control)
