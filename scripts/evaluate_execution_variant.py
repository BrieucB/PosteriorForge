#!/usr/bin/env python3
"""Independent 50k execution review against the immutable CDF-corrected control.

Built with PriorLabs-TabPFN. Performance-first authorization is explicit:
absolute physical failure remains visible but is not a performance-phase veto.
No model calls, reference integrations, clipping, row filtering or gate edits.
"""
from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import json
from pathlib import Path
import numpy as np

from posteriorforge.artifacts import sha256_file, write_json
from posteriorforge.banks import derive_seed, load_context, load_heldout
from posteriorforge.gate import verify_run_binding
from posteriorforge.inference import support_mask
from posteriorforge.validation import assess_samples, compare_to_upstream, sample_integrity

_BINDING_PATH = Path(__file__).with_name("evaluate_cdf_pilot.py")
_spec = importlib.util.spec_from_file_location("cdf_execution_review_bindings", _BINDING_PATH)
bindings = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bindings)
require = bindings.require

SCHEMA = "posteriorforge.execution_variant_review.v1"
RUN_SCHEMA = "posteriorforge.inference.execution.v1"
QUALIFICATION_SCHEMA = "posteriorforge.execution_cache_qualification.v1"
ATTRIBUTION = "Built with PriorLabs-TabPFN"
CONTROL_MANIFEST_SHA256 = "a0114fd74721cf9bed31e9131045e841f262db123c8f849e5cb3f15814a14f4b"
CONTROL_SAMPLES_SHA256 = "27e51ad1b1ee615fc921ee4f5c237155139304da3e7c193f7781a389cee6d53e"
QUALIFICATION_THRESHOLDS = {"normalization_absolute": 1e-5, "max_cdf_absolute": 2e-5,
                            "max_row_total_variation": 1e-4, "quantile_coordinate_span_fraction": 2e-6}


def rejection_review(rejection, *, returned):
    """Structural telemetry checks, never silently convert anomalies into pass."""
    oddities = []
    if not isinstance(rejection, dict):
        return {"structural_oddities": ["missing_rejection_mapping"], "telemetry": rejection}
    for key in ("support_calls", "proposed", "accepted_before_final_trim", "returned"):
        value = rejection.get(key)
        if type(value) is not int or value < (1 if key == "support_calls" else 0):
            oddities.append("invalid_" + key)
    result = {"structural_oddities": oddities, "telemetry": dict(rejection)}
    if oddities:
        return result
    proposed, accepted, actual_returned = (rejection[key] for key in
                                         ("proposed", "accepted_before_final_trim", "returned"))
    if not proposed >= accepted >= actual_returned == returned:
        oddities.append("inconsistent_proposed_accepted_returned_counts")
    if proposed:
        result.update(proposal_support_acceptance_fraction=accepted/proposed,
                      returned_fraction_of_proposals=actual_returned/proposed,
                      final_trim_count=accepted-actual_returned)
    else:
        oddities.append("zero_proposals")
    return result


def comparative_assessment(candidate, control, reference, *, expected_count=50000):
    """Preserve original assessment math; separate physical gate and drift flags."""
    value = np.asarray(candidate)
    mask = support_mask(value) if value.ndim == 2 and value.shape[1] == 21 else None
    integrity = sample_integrity(value, expected_count=expected_count, dimension=21, support_mask=mask)
    absolute = assess_samples(value, reference, support_mask=mask, expected_count=expected_count)
    result = {
        "sample_integrity": integrity, "absolute_physical_assessment": absolute,
        "absolute_physical_gate_required_for_performance": False,
        "relative_distribution_review": None,
        "scope": "finite_cohort_diagnostics_not_population_equivalence_or_global_calibration",
    }
    if not integrity["passed"]:
        result["numeric_review_status"] = "invalid_samples"
        return result
    control = np.asarray(control)
    control_mask = support_mask(control) if control.ndim == 2 and control.shape[1] == 21 else None
    control_integrity = sample_integrity(control, expected_count=expected_count, dimension=21,
                                        support_mask=control_mask)
    require(control_integrity["passed"], "complete finite supported corrected control required")
    relative = compare_to_upstream(value, control, reference_sd=reference["sd"],
                                   directions=reference["directions"])
    relative["original_api_absolute_gate_policy"] = relative["absolute_gate_still_required"]
    relative["absolute_gate_still_required"] = False
    relative["policy_override"] = "Explicit human performance-first authorization; physical assessment is retained separately."
    relative["cdf_statistic"] = "exact two-sample empirical CDF supremum at all pooled sample values, for the same 53 physical projections"
    relative["cdf_is_reference_grid_bound"] = False
    relative["population_confidence_claim"] = False
    relative["thresholds_are_investigation_flags"] = True
    relative["mean_difference_interpretation"] = "absolute change in signed bias, equivalently abs(candidate mean - control mean)/reference SD; not difference of absolute biases"
    thresholds = relative["investigation_thresholds"]
    relative["flags"] = {
        "marginal_cdf": [index for index, value in enumerate(relative["cdf_difference"][:21])
                         if value > thresholds["cdf"]],
        "joint_projection_cdf": [index for index, value in enumerate(relative["cdf_difference"][21:])
                                 if value > thresholds["cdf"]],
        "mean_reference_sd": [index for index, value in enumerate(relative["mean_difference_reference_sd"])
                              if value > thresholds["mean_reference_sd"]],
        "sd_fraction": [index for index, value in enumerate(relative["sd_fraction_difference"])
                        if value > thresholds["sd_fraction"]],
    }
    result["relative_distribution_review"] = relative
    result["numeric_review_status"] = ("distribution_investigation_required"
                                       if relative["investigation_required"] else "no_empirical_drift_flag")
    return result


def validate_sampling_policy(policy):
    from posteriorforge import execution_mixture as mixture
    require(isinstance(policy, dict), "explicit sampling_policy mapping required")
    for key, value in mixture.execution_metadata().items():
        actual = policy.get(key)
        require(actual is value if isinstance(value, bool) else actual == value,
                "mixture sampling metadata mismatch: " + key)
    require(policy.get("head_policy") == "raw_mass_rejection_v1",
            "uniform normalized-component selection is not the approved mixture law")
    require(policy.get("mass_envelope") == mixture.HEAD_MASS_ENVELOPE,
            "mixture head mass envelope changed")
    dedup = policy.get("stage0_dedup", {})
    require(isinstance(dedup, dict) and dedup.get("enabled") is True
            and dedup.get("scope") == "full_mixture_only"
            and dedup.get("component_distribution_dedup") is False,
            "unqualified stage0/component dedup policy")
    require(dedup.get("density_scope", "proposal-wave-local; no cross-request/case reuse")
            == "proposal-wave-local; no cross-request/case reuse", "cross-request stage0 memo is unqualified")
    streams = policy.get("streams")
    roles = {"head_proposal": ("execution-head-proposal", "int64"),
             "head_acceptance": ("execution-head-acceptance", "float64"),
             "component_uniform": ("execution-component-uniform", "float32")}
    require(isinstance(streams, dict) and set(streams) == set(roles),
            "three separate head/acceptance/component RNG streams required")
    seeds = []
    for key, (role, dtype) in roles.items():
        stream = streams[key]
        require(isinstance(stream, dict) and stream.get("role") == role
                and str(stream.get("dtype")).removeprefix("torch.") == dtype,
                "sampling stream role/dtype mismatch: " + key)
        require(type(stream.get("seed")) is int and 0 <= stream["seed"] < 2**64,
                "explicit integer sampling stream seed required: " + key)
        require(stream.get("device", "cpu") == "cpu", "CPU sampling stream required: " + key)
        seeds.append(stream["seed"])
    require(len(set(seeds)) == 3, "head/acceptance/component stream seeds must differ")
    return dict(policy)


def validate_execution_header(manifest, *, context, query, control):
    from posteriorforge.inference_cdf import guard_baseline_source
    from posteriorforge.performance_runner import STREAM_ROLES, STREAM_DTYPES
    variant = manifest.get("variant")
    require(variant in ("execution_cache_v1", "execution_mixture_v1"), "unknown execution variant")
    expected = {
        "schema": RUN_SCHEMA, "variant": variant, "status": "complete",
        "context_seed": 20261111, "case": 0, "context_size": 4096,
        "requested_count": 50000, "scientific_run": True,
        "end_to_end": True, "run_purpose": "performance_execution",
        "performance_first_authorized": True,
        "absolute_accuracy_gate_required_for_performance": False,
        "tail_family_changed": False, "sources_unchanged": True,
        "partial_vector_filtering": False, "clipping_or_repair_performed": False,
        "observation_preprocessing": "raw; TabPFN frozen preprocessing",
        "log_density": "not_used", "microbatch": 50000, "query_batch_size": 1024,
        "rank": 0, "world_size": 1, "completed_stage_indices": list(range(21)),
        "sampling_rng_policy": "independent_cpu_sampling_streams_model_rng_isolated",
    }
    for key, expected_value in expected.items():
        actual = manifest.get(key)
        require(actual is expected_value if isinstance(expected_value, bool) else actual == expected_value,
                "execution manifest contract mismatch: " + key)
    require(type(manifest.get("repetition")) is int and manifest["repetition"] >= 0,
            "explicit nonnegative execution repetition required")
    require(manifest.get("baseline_execution_source") == guard_baseline_source(),
            "original execution-source guard changed")
    runner_path = Path(__file__).resolve().parents[1]/"src/posteriorforge/performance_runner.py"
    require(manifest.get("runner_source_sha256") == sha256_file(runner_path),
            "execution runner differs from the recorded source")
    descriptor = bindings.validate_adapter_descriptor(manifest.get("compatibility_adapter"))
    validate_engine_metadata(manifest.get("execution_engine"))
    original_settings = dict(control["regressor_settings"])
    actual_settings = dict(manifest["regressor_settings"])
    require(actual_settings.get("fit_mode") == "fit_with_cache", "resident executor requires actual fit_with_cache metadata")
    actual_settings["fit_mode"] = original_settings["fit_mode"]
    policy = manifest.get("sampling_policy", {})
    if variant == "execution_mixture_v1":
        validate_sampling_policy(policy)
    else:
        require(policy.get("head_policy") == "not_used" and policy.get("mass_envelope") is None
                and policy.get("stage0_dedup", {}).get("enabled") is False,
                "resident-full policy must not claim component sampling")
    streams = policy.get("streams", {})
    require(set(streams) == set(STREAM_ROLES), "complete sampling stream policy required")
    seeds = {}
    for name, role in STREAM_ROLES.items():
        expected_seed = derive_seed(20261111, role, case=0, rank=0, repetition=manifest["repetition"])
        stream = streams[name]
        require(stream.get("role") == role and stream.get("seed") == expected_seed
                and str(stream.get("dtype")).removeprefix("torch.") == STREAM_DTYPES[name].removeprefix("torch.")
                and stream.get("device") == "cpu", "execution stream derivation mismatch: " + name)
        seeds[name] = expected_seed
    require(len(set(seeds.values())) == 3 and manifest.get("seed_streams") == seeds
            and manifest.get("seed") == seeds["component_uniform"], "execution sampling seed binding mismatch")
    # The old binder is reused only for unchanged science/context/checkpoint/settings.
    # Its original RNG/microbatch metadata are normalized on a private copy AFTER
    # the actual new versioned execution policy has been independently checked.
    adjusted = copy.deepcopy(manifest)
    adjusted.update(regressor_settings=actual_settings, microbatch=10000, repetition=0,
                    seed=derive_seed(20261111, "posterior", case=0))
    verify_run_binding(adjusted, context, query, 20261111, 0)
    require(manifest.get("model", {}).get("dtype") == "torch.float32"
            and manifest.get("model", {}).get("autocast") is False,
            "execution changed frozen FP32/no-autocast contract")
    expected_calls = manifest.get("expected_patched_call_count")
    require(type(expected_calls) is int and expected_calls > 0
            and expected_calls == descriptor["patched_call_count"],
            "execution CDF call coverage differs from its declared plan")
    return descriptor


def validate_engine_metadata(engine):
    from posteriorforge.execution_cache import execution_metadata
    approved = execution_metadata()
    require(isinstance(engine, dict), "execution_engine mapping required")
    for key in ("module_sha256", "engine_id", "engine_version", "SDKsourcehashes"):
        require(engine.get(key) == approved[key], "unqualified execution engine metadata: " + key)
    require(isinstance(engine.get("config"), dict), "execution configuration mapping required")
    for key, value in approved["config"].items():
        require(engine["config"].get(key) == value, "execution configuration invariant changed: " + key)
    return approved


def validate_component_qualification(qualification):
    from posteriorforge import execution_mixture as mixture
    require(qualification.get("full21_qualified") is True
            and qualification.get("full21_component_qualified") is True
            and qualification.get("full21_sampler_qualified") is True
            and qualification.get("component_qualified") is True,
            "full21 base/component/production-sampler qualification required for mixture")
    require(sorted(qualification.get("component_stage_coverage", [])) == list(range(21)),
            "mixture component qualification does not cover all21 stages")
    require(qualification.get("mixture_module_sha256") == mixture.execution_source_sha256(),
            "mixture source differs from the qualified component module")
    for row in qualification["stage_results"]:
        component = row.get("component_qualification", {})
        require(component.get("schema") == "posteriorforge.execution_component_qualification.v1"
                and component.get("passed") is True
                and component.get("module_sha256") == mixture.execution_source_sha256()
                and component.get("engine_id") == mixture.ENGINE_ID,
                "invalid or stale component qualification")
        require(component.get("fast_icdf_same_logits_failures") == 0
                and component.get("uniform_head_without_mass_correction_is_not_authorized") is True,
                "production sampler or mass-corrected head law was not qualified")
        members = component.get("members", [])
        require([m.get("ensemble_index") for m in members] == list(range(8)),
                "all eight component paths must be qualified")
        for member in members:
            require(member.get("passed") is True
                    and member.get("configuration_identity_equal") is True
                    and member.get("exact_borders") is True and member.get("float32") is True
                    and member.get("component_raw_mass_matches_fp64_sum") is True
                    and member.get("normalized_component", {}).get("passed") is True
                    and member.get("corrected_cdf_call_delta") == 1
                    and member.get("executed_corrected_helper_fingerprint_matches") is True
                    and member.get("fast_icdf_same_logits_exact") is True
                    and member.get("component_head_mass_envelope") == mixture.HEAD_MASS_ENVELOPE,
                    "component source/CDF/mass/production-sampler parity failed")
        require(component.get("executor_metadata", {}).get("full8_member_lists_restored") is True,
                "component filtering did not restore all eight member lists")
        roundoff = component.get("roundoff", {})
        require(roundoff.get("component_mass_guard_passed") is True
                and roundoff.get("mass_corrected_reconstruction_passed") is True,
                "mass-corrected component reconstruction failed")
        for key in ("normalized_mean_captured_raw_vs_full_fp32",
                    "normalized_mean_observed_raw_vs_full_fp32",
                    "mass_corrected_observed_outputs_vs_full_fp32"):
            require(component.get(key, {}).get("passed") is True,
                    "direct actual-mixture reconstruction failed: " + key)
        require(component.get("stage0_dedup_scope") == "full_mixture_only"
                and component.get("component_distribution_dedup_qualified") is False,
                "component dedup scope was changed")
        if row["stage_index"] == 0:
            dedup = component.get("stage0_shared_distribution", {})
            require(dedup.get("passed") is True and dedup.get("shared_icdf_same_logits_exact") is True,
                    "shared full-mixture stage0 sampler was not qualified")
    return {"full21_component_qualified": True, "full21_sampler_qualified": True,
            "mixture_module_sha256": mixture.execution_source_sha256(),
            "selected_head_is_not_compared_as_the_full_mixture": True}


def qualification_review(path, manifest, *, context, query):
    require(sha256_file(path) == manifest.get("qualification_manifest_sha256"),
            "execution qualification manifest SHA256 mismatch")
    q = json.loads(Path(path).read_text())
    require(q.get("schema") == QUALIFICATION_SCHEMA, "unknown pointwise qualification schema")
    require(q.get("qualified") is True and q.get("status") == "screen_pass",
            "completed passing fixed-conditional qualification required")
    engine = manifest.get("execution_engine")
    validate_engine_metadata(engine)
    module_path = Path(__file__).resolve().parents[1] / "src/posteriorforge/execution_cache.py"
    require(engine.get("module_sha256") == sha256_file(module_path)
            and q.get("module_sha256") == engine.get("module_sha256"),
            "execution engine source differs from the qualified engine")
    require(isinstance(engine.get("engine_id"), str) and bool(engine["engine_id"])
            and q.get("engine_id") == engine["engine_id"], "qualified engine identity mismatch")
    expected = {"control_variant": "cdf_arithmetic_v1",
                "checkpoint_sha256": manifest["checkpoint_sha256"],
                "scientific_fingerprint": manifest["scientific_fingerprint"],
                "parameter_order": manifest["parameter_order"],
                "context_theta": manifest["context_theta"], "context_obs": manifest["context_obs"],
                "query": manifest["query"]}
    for key, value in expected.items():
        require(q.get(key) == value, "pointwise qualification binding mismatch: " + key)
    require(q.get("icdf_contract_version") == 2, "ICDF numerical contract version2 required")
    require(q.get("thresholds") == QUALIFICATION_THRESHOLDS, "pointwise numerical thresholds changed")
    require(q.get("row_counts") == [1, 16, 128, 1024], "pointwise query-count contract changed")
    require(q.get("sources_unchanged") is True, "qualification sources were not restored/unchanged")
    require(q.get("prefix_manifest_sha256") == CONTROL_MANIFEST_SHA256
            and q.get("prefix_samples_sha256") == CONTROL_SAMPLES_SHA256,
            "qualification prefixes differ from the corrected control")
    qualifier_path = Path(__file__).with_name("qualify_execution_cache.py")
    require(q.get("sources", {}).get("qualification_script_sha256") == sha256_file(qualifier_path),
            "qualification script differs from the recorded source")
    bindings.validate_adapter_descriptor(q.get("compatibility_adapter"))
    coverage = q.get("stage_coverage")
    require(isinstance(coverage, list) and coverage and all(type(v) is int and 0 <= v < 21 for v in coverage)
            and len(set(coverage)) == len(coverage), "valid completed stage coverage required")
    require(q.get("requested_stages") == coverage, "qualified requested/completed stage coverage mismatch")
    rows = q.get("stage_results")
    require(isinstance(rows, list) and [row.get("stage_index") for row in rows] == coverage,
            "qualification stage results do not match completed coverage")
    for row in rows:
        require(row.get("passed") is True and row.get("stress_comparison", {}).get("passed") is True,
                "qualification has a failed stage or stress comparison")
        comparisons = row.get("comparisons", [])
        require([item.get("query_rows") for item in comparisons] == [1, 16, 128, 1024]
                and all(item.get("cache_vs_same_chunk_control", {}).get("passed") is True
                        and item.get("control_chunk_vs_1024", {}).get("passed") is True for item in comparisons),
                "qualification has missing/failed pointwise or batching comparisons")
        for item in [part[key] for part in comparisons
                     for key in ("cache_vs_same_chunk_control", "control_chunk_vs_1024")] + [row["stress_comparison"]]:
            require(item.get("inverse_cdf", {}).get("icdf_contract_version") == 2
                    and item["inverse_cdf"].get("passed") is True
                    and item["inverse_cdf"].get("same_logits_and_borders_exact_failures") == 0
                    and item["inverse_cdf"].get("independent_fp32_replay_failures") == 0,
                    "missing/failed probability-space ICDFv2 arithmetic proof")
        require(row.get("baseline_model_contract_after", {}).get("input_normalization_train_only_verified") is True
                and row.get("cache_model_contract_after", {}).get("input_normalization_train_only_verified") is True,
                "training-only normalization was not demonstrated")
    full = sorted(coverage) == list(range(21))
    require(q.get("full21_qualified") is full, "qualification coverage claim is inconsistent")
    component_proof = (validate_component_qualification(q)
                       if manifest["variant"] == "execution_mixture_v1" else None)
    return {"manifest_file": str(Path(path).resolve()), "manifest_sha256": sha256_file(path),
            "icdf_contract_version": 2, "component_and_sampler_proof": component_proof,
            "stage_coverage": coverage, "full21_qualified": full,
            "scope": "full21_fixed_conditional_diagnostic" if full else "coverage_limited_fixed_conditional_screen",
            "thresholds": q.get("thresholds"), "qualification": q,
            "not_a_proof_of_posterior_equivalence": True}


def write_csv(path, rows):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    root = Path(args.root).resolve()
    results = root / "results"
    gate_path = results / "gate-final.json"
    require(sha256_file(gate_path) == bindings.BASELINE_GATE_SHA256, "immutable original gate changed")
    gate = json.loads(gate_path.read_text())
    banks = results / "banks-v1"
    context = load_context(banks, 20261111, 4096)
    query = load_heldout(banks, 0).obs
    arrays, _, reference, custody = bindings.reference_for_case(results, gate, query, 0)
    control_path = results / "cdf-arithmetic-v1/context-4096/seed-20261111-case-0/manifest.json"
    require(sha256_file(control_path) == CONTROL_MANIFEST_SHA256, "canonical corrected control manifest changed")
    control_manifest = json.loads(control_path.read_text())
    require(control_manifest["samples_sha256"] == CONTROL_SAMPLES_SHA256, "canonical corrected control sample identity changed")
    bindings.validate_pilot_header(control_manifest, seed=20261111, case=0, context_size=4096)
    bindings.validate_run_reference_binding(control_manifest, custody)
    verify_run_binding(control_manifest, context, query, 20261111, 0)
    control = bindings.load_samples(control_path, control_manifest)
    control_assessment = assess_samples(control, reference, support_mask=support_mask(control))
    require(control_assessment["integrity"]["passed"] and control_assessment["reference_qualified"],
            "canonical corrected control or qualified reference invalid")
    if args.preflight_only:
        value = {"schema": SCHEMA + ".preflight", "attribution": ATTRIBUTION,
                 "control_manifest_sha256": sha256_file(control_path),
                 "control_samples_sha256": CONTROL_SAMPLES_SHA256,
                 "control_absolute_physical_status": control_assessment["status"],
                 "control_absolute_physical_passed": control_assessment["passed"],
                 "control_integrity_passed": True, "reference_qualified": True,
                 "absolute_accuracy_gate_required_for_performance": False,
                 "original_gate_sha256": sha256_file(gate_path)}
        print(json.dumps(value, indent=2))
        return value
    require(args.variant_manifest and args.qualification_manifest and args.output,
            "--variant-manifest, --qualification-manifest and --output required")
    manifest_path = Path(args.variant_manifest).resolve()
    manifest = json.loads(manifest_path.read_text())
    validate_execution_header(manifest, context=context, query=query, control=control_manifest)
    bindings.validate_run_reference_binding(manifest, custody)
    qualification = qualification_review(args.qualification_manifest, manifest, context=context, query=query)
    candidate = bindings.load_samples(manifest_path, manifest)
    assessment = comparative_assessment(candidate, control, reference)
    output = Path(args.output).resolve()
    for protected in ("banks-v1", "upstream-v1", "references-v1", "reference-grids-v1", "cdf-arithmetic-v1"):
        require(not output.is_relative_to(results / protected), "review may not alter original/control artifacts")
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "schema": SCHEMA, "attribution": ATTRIBUTION,
        "performance_first_authorized": True, "absolute_accuracy_gate_required_for_performance": False,
        "no_model_calls": True, "no_reference_integrations": True, "original_gate_unchanged": True,
        "fixed_case": 0, "context_seed": 20261111, "context_size": 4096, "expected_count": 50000,
        "parameter_order": manifest["parameter_order"], **assessment,
        "pointwise_qualification": qualification,
        "performance_review_requires_qualification_scope_awareness": True,
        "control": {"manifest_file": str(control_path), "manifest_sha256": sha256_file(control_path),
                    "samples_sha256": CONTROL_SAMPLES_SHA256,
                    "absolute_physical_assessment": control_assessment,
                    "timings": control_manifest["timings"],
                    "sampling_rng_policy": "original shared Torch RNG; seed role posterior, repetition0"},
        "candidate": {"manifest_file": str(manifest_path), "manifest_sha256": sha256_file(manifest_path),
                      "samples_sha256": manifest["samples_sha256"], "samples_descriptor": manifest["samples"],
                      "execution_policy": manifest.get("execution_policy"), "execution_engine": manifest["execution_engine"],
                      "sampling_rng_policy": manifest.get("sampling_rng_policy"), "sampling_policy": manifest.get("sampling_policy"),
                      "timings": manifest["timings"], "memory": manifest["memory"]},
        "reference": custody,
        "rejection_review": {
            "control": rejection_review(control_manifest.get("rejection"), returned=50000),
            "candidate": rejection_review(manifest.get("rejection"), returned=50000)},
        "source": {"evaluation_script_sha256": sha256_file(__file__),
                   "binding_helper_sha256": sha256_file(_BINDING_PATH),
                   "original_validation_sha256": sha256_file(Path(__file__).resolve().parents[1]/"src/posteriorforge/validation.py")},
        "interpretation": [
            "Relative CDF differences are exact two-sample ECDF distances, not reference-grid bounds; no reference numerical error is added to them.",
            "Absolute physical CDF bounds include the unchanged saved-reference uncertainty and whole-line monotonicity/tail bounds.",
            "The .01 CDF, .05 reference-SD mean and .05 SD-fraction thresholds flag investigation; they neither establish population equivalence nor waive absolute scientific failure.",
            "Changed forward/batch/RNG consumption can change finite Monte Carlo cohorts even if fixed-conditional distributions agree.",
            "A fixed-conditional screen only covers its declared prefixes, stages and chunks; accepted-prefix probes do not establish the unfiltered proposal or rejection-conditioned joint law.",
            "Performance-first authorization explicitly permits engineering review while the original physical accuracy criteria remain failed and visible.",
        ],
    }
    relative = result["relative_distribution_review"]
    if relative is not None:
        rows = []
        for index, name in enumerate(manifest["parameter_order"]):
            rows.append({"parameter": name, "two_sample_ecdf_distance": relative["cdf_difference"][index],
                         "mean_difference_reference_sd": relative["mean_difference_reference_sd"][index],
                         "sd_fraction_difference": relative["sd_fraction_difference"][index],
                         "control_physical_cdf_upper": control_assessment["marginal_cdf"]["upper"][index],
                         "candidate_physical_cdf_upper": assessment["absolute_physical_assessment"]["marginal_cdf"]["upper"][index]})
        write_csv(output/"execution-parameters.csv", rows)
        write_csv(output/"execution-projections.csv",
                  [{"joint_witness": index+1, "two_sample_ecdf_distance": relative["cdf_difference"][index+21],
                    "control_physical_cdf_upper": control_assessment["joint_projection_cdf"]["upper"][index],
                    "candidate_physical_cdf_upper": assessment["absolute_physical_assessment"]["joint_projection_cdf"]["upper"][index]}
                   for index in range(32)])
    write_json(output/"execution-review.json", result)
    write_json(output/"execution-checksums.json", {p.name:sha256_file(p) for p in sorted(output.iterdir()) if p.is_file()})
    require(sha256_file(gate_path) == bindings.BASELINE_GATE_SHA256
            and sha256_file(control_path) == CONTROL_MANIFEST_SHA256,
            "immutable control/gate changed during review")
    print(json.dumps({"output":str(output), "numeric_review_status":result["numeric_review_status"],
                      "integrity_passed":result["sample_integrity"]["passed"],
                      "full21_fixed_conditional_qualified":qualification["full21_qualified"],
                      "absolute_physical_passed":result["absolute_physical_assessment"]["passed"],
                      "absolute_accuracy_gate_required_for_performance":False}, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--variant-manifest")
    parser.add_argument("--qualification-manifest")
    parser.add_argument("--output")
    parser.add_argument("--preflight-only", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
