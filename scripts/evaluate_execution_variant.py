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
import hashlib
import re
import subprocess
from types import ModuleType
from pathlib import Path
import numpy as np

from posteriorforge.artifacts import sha256_file, write_json
from posteriorforge.banks import derive_seed, load_context, load_heldout
from posteriorforge.gate import verify_run_binding
from posteriorforge.inference import support_mask, UPSTREAM_REVISION
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
RUNNER_V1_SHA256 = "c121e897ab583dc846fcb04da6c3c74930713f5d9025411d66e042b8d817899b"
RUNNER_V1_GIT_REVISION = "ea2d673c30c3d8c047c9d1b3f349d0b142b38652"
RUNNER_V2_SHA256 = "0a071ae7d016b4e99b75940da878d6c37eeab7f25a92b921efb2a39d66fd429d"
RUNNER_V2_GIT_REVISION = "2bb6f3809a32b6ebe8933669172c681058151f50"
ORCHESTRATOR_SHA256 = "6731e392a63c21a0dbcd2f5df585619691f604fde9ed23df852a3bfaf512df99"
RUNNER_V3_SHA256 = "be9a7d38413f797b72377f048d69db8f948953e0cd8d54e361ee690893337ff1"
RUNNER_V3_GIT_REVISION = "a768ba06af054e68e2c478d690c43b05f3299f80"
ORCHESTRATOR_V3_SHA256 = "87bfdce4716f8899e964d457a96372ddca788a951eb1d3004506d1e1c3bdee54"
RUNNER_V4_SHA256 = "23ccad8b5bbcdd7a48340b349d515ca870763359cfeb71333e02fc2b6a8aca3b"
RUNNER_V4_GIT_REVISION = "36594f45debeaf1921b10b7bc12a1daa8539ee37"
ORCHESTRATOR_V4_SHA256 = "2136a19ca8d8d081b03dfe0a0069c271f800a0a236fcc76bcd740ba9694365a2"
FIFO_MODULE_SHA256 = "8fb2ba1fb63cd917378ff9bc8b6f70be7410862c509cd22cb2fb6f454d3d35fa"
FIFO_POLICY_SHA256 = "dbeca85d7107cd9a5d9cf9f967aa02807cc38d10b7493ecc42d2181a503391af"
GQA_VARIANTS = ("execution_gqa_hybrid_v1","execution_gqa_hybrid_fifo_v1")
_REVIEWED_V4_RUNNER = None
GQA_BATCH_QUALIFIER_SHA256 = "31422927077911879b6cd155605a6a668796305ee97ccf6585819d8fe494cfde"
GQA_BATCH_SCHEMA = "posteriorforge.execution_gqa_batch_qualification.v1"
LARGE_BATCHES = (4096,6250,10000)
GQA_WRAPPER_SHA256 = "ac22c67c23403723902805666d1239bd30a8af355554df58a8638a8fd1f21e82"
GQA_MODULE_SHA256 = "8b2cff752855e1b1966d2e374322602fa71368b1926da9d3b0261edd111aee6b"
GQA_RECOVERY_SHA256 = "83064f54712014b5c4c66569e854872caff03593cfdc337f2472a7b5723aea02"
_HISTORICAL_RUNNER_VERIFIED = False
_HISTORICAL_V2_RUNNER_VERIFIED = False
_HISTORICAL_V3_RUNNER_VERIFIED = False
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


def reviewed_v4_runner():
    """Read the exact committed CPU validator; never trust a future current file."""
    global _REVIEWED_V4_RUNNER
    if _REVIEWED_V4_RUNNER is None:
        repo=Path(__file__).resolve().parents[1]
        content=subprocess.run(["git","-C",str(repo),"show",
            RUNNER_V4_GIT_REVISION+":src/posteriorforge/performance_runner.py"],check=True,capture_output=True).stdout
        require(hashlib.sha256(content).hexdigest()==RUNNER_V4_SHA256,"historical pinned version4 runner Git blob mismatch")
        module=ModuleType("posteriorforge._reviewed_execution_runner_v4")
        module.__file__=str(repo/"src/posteriorforge/performance_runner.py");module.__package__="posteriorforge"
        exec(compile(content,module.__file__,"exec"),module.__dict__)
        require(module.RUNNER_VERSION==4 and module.PROPOSAL_BUFFER_SHA256==FIFO_MODULE_SHA256,
                "reviewed version4 FIFO source constant mismatch")
        _REVIEWED_V4_RUNNER=module
    return _REVIEWED_V4_RUNNER


def validate_fifo_runtime_metadata(manifest,*,rank,world,count):
    require(manifest.get("runner_source_sha256")==RUNNER_V4_SHA256 and manifest.get("runner_version")==4,
            "FIFO requires reviewed version4 runner")
    runner=reviewed_v4_runner();lookahead=manifest.get("proposal_lookahead")
    runner.validate_proposal_lookahead(manifest.get("mode"),lookahead)
    policy=runner.proposal_buffer_policy(lookahead)
    require(manifest.get("proposal_buffer_policy")==policy,"FIFO declared source/policy mismatch")
    if lookahead==0:
        require(manifest.get("variant")!="execution_gqa_hybrid_fifo_v1" and manifest.get("proposal_buffer") is None,
                "zero lookahead cannot claim FIFO execution")
        return None
    require(lookahead==1200 and manifest.get("variant")=="execution_gqa_hybrid_fifo_v1"
            and policy.get("module_sha256")==FIFO_MODULE_SHA256 and policy.get("policy_sha256")==FIFO_POLICY_SHA256,
            "explicit reviewed FIFO family/global lookahead required")
    scope=runner.proposal_buffer_scope(manifest,context_seed=manifest["context_seed"],case=manifest["case"],
        repetition=manifest["repetition"],rank=rank,world_size=world,seeds=manifest["seed_streams"])
    record=manifest.get("proposal_buffer",{})
    runner.validate_proposal_buffer_record(record,world_size=world,rank=rank,scope=scope)
    counts=record["original_sampler_counts"]
    require(counts["returned_count"]==count and manifest.get("rejection")=={
        "support_calls":counts["original_support_calls"],"proposed":counts["original_checked_count"],
        "accepted_before_final_trim":counts["accepted_before_trim"],"returned":count},
        "FIFO delivery/support/accepted/trim counts mismatch")
    waves=record["actual_generated_wave_sizes"];preparations=21*len(waves)
    require(manifest.get("proposal_wave_sizes")==waves and manifest.get("generated_proposal_wave_count")==len(waves)
            and manifest.get("expected_stage_prepare_count")==preparations
            and manifest.get("execution_counters",{}).get("prepare_count")==preparations,
            "FIFO actual generated waves/stage preparations mismatch")
    events=manifest.get("stage_events",[])
    require(len(events)==preparations,"FIFO actual all21 stage event coverage missing")
    for index,event in enumerate(events):
        wave,stage=divmod(index,21)
        require(all(type(event.get(key)) is int for key in ("proposal_round","stage_index","proposed_rows"))
                and event.get("proposal_round")==wave+1 and event.get("stage_index")==stage
                and event.get("proposed_rows")==waves[wave],
                "FIFO stage order/full-wave row coverage mismatch")
    return {"policy":policy,"record":record,"all21_per_generated_wave_verified":True,
        "original_support_calls_are_not_generated_waves":True,
        "accepted_final_trim_count":counts["accepted_before_trim"]-count,
        "seeded_rowwise_equality_claim":False}


def validate_runner_source(manifest, *, aggregate=False):
    """Authenticate a versioned source; historical artifacts never use current SHA."""
    global _HISTORICAL_RUNNER_VERIFIED, _HISTORICAL_V2_RUNNER_VERIFIED, _HISTORICAL_V3_RUNNER_VERIFIED
    value, version = manifest.get("runner_source_sha256"), manifest.get("runner_version")
    repo = Path(__file__).resolve().parents[1]
    if value == RUNNER_V1_SHA256:
        require(version is None or type(version) is int and version == 1,
                "historical runner version/source mismatch")
        require(manifest.get("variant") not in GQA_VARIANTS,
                "GQA hybrid requires reviewed runner version2")
        if not _HISTORICAL_RUNNER_VERIFIED:
            content = subprocess.run(["git", "-C", str(repo), "show",
                RUNNER_V1_GIT_REVISION+":src/posteriorforge/performance_runner.py"],
                check=True, capture_output=True).stdout
            require(hashlib.sha256(content).hexdigest() == RUNNER_V1_SHA256,
                    "historical pinned runner Git blob mismatch")
            _HISTORICAL_RUNNER_VERIFIED = True
        return {"version": 1, "sha256": value, "historical_git_revision": RUNNER_V1_GIT_REVISION,
                "version_field_absent_in_original": version is None,
                "authentication": "SHA256 of the exact pinned historical Git blob"}
    if value == RUNNER_V3_SHA256:
        require(type(version) is int and version == 3, "runner version3/source mismatch")
        if not _HISTORICAL_V3_RUNNER_VERIFIED:
            content=subprocess.run(["git","-C",str(repo),"show",
                RUNNER_V3_GIT_REVISION+":src/posteriorforge/performance_runner.py"],check=True,capture_output=True).stdout
            require(hashlib.sha256(content).hexdigest()==RUNNER_V3_SHA256,"historical pinned version3 runner Git blob mismatch")
            _HISTORICAL_V3_RUNNER_VERIFIED=True
        return {"version":3,"sha256":value,"historical_git_revision":RUNNER_V3_GIT_REVISION,
                "authentication":"SHA256 of the exact pinned historical version3 Git blob"}
    if value == RUNNER_V4_SHA256:
        require(type(version) is int and version==4,"runner version4/source mismatch")
        reviewed_v4_runner()
        return {"version":4,"sha256":value,"historical_git_revision":RUNNER_V4_GIT_REVISION,
                "authentication":"SHA256 of the exact pinned historical version4 Git blob"}
    require(value == RUNNER_V2_SHA256, "unknown/unreviewed execution runner source")
    require((aggregate and version is None) or type(version) is int and version == 2,
            "runner version2/source mismatch")
    if not _HISTORICAL_V2_RUNNER_VERIFIED:
        content = subprocess.run(["git", "-C", str(repo), "show",
            RUNNER_V2_GIT_REVISION+":src/posteriorforge/performance_runner.py"],
            check=True, capture_output=True).stdout
        require(hashlib.sha256(content).hexdigest() == RUNNER_V2_SHA256,
                "historical pinned version2 runner Git blob mismatch")
        _HISTORICAL_V2_RUNNER_VERIFIED = True
    return {"version": 2, "sha256": value, "historical_git_revision": RUNNER_V2_GIT_REVISION,
            "authentication": "SHA256 of the exact pinned historical version2 Git blob"}


def validate_gqa_runtime_metadata(manifest):
    from posteriorforge import execution_gqa_hybrid as execution, execution_hybrid as hybrid
    execution.require_frozen_sources()
    require(execution.execution_source_sha256() == GQA_WRAPPER_SHA256
            and execution.PINNED_GQA_SHA256 == GQA_MODULE_SHA256,
            "unreviewed GQA wrapper/attention source")
    approved = execution.execution_metadata()
    for field in ("gqa_hybrid_execution", "gqa_execution_counters"):
        actual = manifest.get(field)
        require(isinstance(actual, dict), "GQA runtime mapping missing: "+field)
        for key, value in approved.items():
            require(actual.get(key) is value if isinstance(value, bool) else actual.get(key) == value,
                    "GQA runtime source/policy mismatch: "+field+"/"+key)
    require(manifest.get("mode") == "component_gqa_hybrid"
            and manifest.get("qualification_full21") is True
            and manifest.get("qualification_full21_gqa_hybrid") is True
            and manifest.get("qualification_cache_only_full21") is False,
            "explicit GQA hybrid qualification/cache-only distinction required")
    require("hybrid_execution" not in manifest,
            "GQA hybrid must not carry the different legacy top-level hybrid policy")
    require(manifest.get("metadata_writer_restored") is True, "GQA metadata writer restoration missing")
    checks, counters = manifest.get("gqa_prediction_checks", {}), manifest["gqa_execution_counters"]
    sampling, hybrid_counters = manifest.get("sampling_counters", {}), manifest.get("hybrid_execution_counters", {})
    numbers = [sampling.get(key) for key in ("stage0_shared_predictions", "component_predictions")]
    require(all(type(n) is int and n > 0 for n in numbers), "actual stage0/component prediction counts required")
    expected_calls = sum(numbers)
    require(type(checks.get("verified_calls")) is int and checks["verified_calls"] == expected_calls
            and counters.get("contexts") == expected_calls,
            "actual GQA context/check coverage differs from sampling")
    require(manifest.get("expected_patched_call_count") == 8*numbers[0]+numbers[1],
            "GQA corrected CDF coverage differs from actual predictions")
    require(all(checks.get(key) is True for key in ("all_aliases_restored", "all_sources_unchanged"))
            and all(counters.get(key) is True for key in ("all_contexts_restored", "all_sources_unchanged")),
            "GQA actual source/alias restoration failed")
    for key in ("attention_call_count", "expanded_attention_calls", "expanded_tensor_views"):
        require(type(counters.get(key)) is int and counters[key] > 0,
                "actual positive GQA attention counter required: "+key)
    require(counters["attention_call_count"] >= counters["expanded_attention_calls"]
            and counters["expanded_tensor_views"] >= counters["expanded_attention_calls"],
            "inconsistent actual GQA attention/view counters")
    fallback = checks.get("uncached_fallback_calls")
    require(type(fallback) is int and fallback > 0
            and fallback == hybrid_counters.get("uncached_predict_count"),
            "actual original stage10/member5 fallback coverage mismatch")
    for key, value in hybrid.execution_metadata({10:[5]}).items():
        require(hybrid_counters.get(key) is value if isinstance(value, bool) else hybrid_counters.get(key) == value,
                "GQA underlying hybrid policy/source mismatch: "+key)
    require(hybrid_counters.get("original_preprocessing_lists_restored") is True
            and hybrid_counters.get("original_translator_alias_restored") is True
            and manifest.get("component_execution_counters", {}).get("full8_member_lists_restored") is True,
            "GQA component/preprocessor/CDF aliases were not restored")
    fingerprints = checks.get("attention_fingerprints", {})
    keys = ("original_attention_code_sha256", "active_attention_code_sha256", "executed_original_attention_code_sha256")
    require(set(fingerprints) == set(keys) and all(isinstance(fingerprints[key], str)
            and re.fullmatch("[0-9a-f]{64}", fingerprints[key]) for key in keys),
            "well-formed actual attention fingerprints required")
    require(fingerprints[keys[0]] == fingerprints[keys[2]] != fingerprints[keys[1]],
            "actual original attention execution fingerprint mismatch")
    if manifest.get("runner_source_sha256") in (RUNNER_V3_SHA256,RUNNER_V4_SHA256):
        chunk=manifest.get("query_batch_size")
        native=manifest.get("execution_counters",{})
        validate_engine_metadata(native)
        cap=manifest.get("engine_query_batch_cap")
        require(native.get("config",{}).get("query_batch_size")==cap,
                "actual prepared engine config differs from declared cap")
        support_calls=manifest.get("rejection",{}).get("support_calls")
        if manifest.get("proposal_lookahead",0):
            record=manifest.get("proposal_buffer",{})
            generated=record.get("generated_wave_count")
            require(type(generated) is int and generated>0 and native.get("prepare_count")==21*generated,
                    "version4 FIFO actual generated-stage coverage mismatch")
        else:
            require(type(support_calls) is int and support_calls>0
                    and native.get("prepare_count")==21*support_calls,
                    "version3/4 actual stage prepare coverage mismatch")
        large=chunk in LARGE_BATCHES
        expected={"qualification_family":"large_batch_v1" if large else "gqa_hybrid_1024_v2",
                  "qualification_large_batch_size":chunk if large else None,
                  "qualification_large_batch_producer_sha256":GQA_BATCH_QUALIFIER_SHA256 if large else None,
                  "qualification_portable_source_verified":large,"historical_install_ctime_replayed":not large}
        for key,value in expected.items():
            require(manifest.get(key) is value if isinstance(value,bool) else manifest.get(key)==value,
                    "version3 qualification family/portability mismatch: "+key)
    policy = manifest.get("sampling_policy", {})
    require(policy.get("uniform_assignment") == "preassigned_per_stage_by_physical_row"
            and policy.get("head_retry_preserves_row_uniform") is True,
            "GQA head retries must preserve the preassigned row uniform")
    return dict(fingerprints)


def validate_execution_header(manifest, *, context, query, control, worker_partition=None):
    from posteriorforge.inference_cdf import guard_baseline_source
    from posteriorforge.performance_runner import STREAM_ROLES, STREAM_DTYPES
    variant = manifest.get("variant")
    require(variant in ("execution_cache_v1", "execution_mixture_v1", "execution_hybrid_v1", *GQA_VARIANTS), "unknown execution variant")
    partition = worker_partition or {"rank":0,"world_size":1,"requested_count":50000}
    rank,world,count = (partition[key] for key in ("rank","world_size","requested_count"))
    require(type(world) is int and world in (1,2,4) and type(rank) is int and 0<=rank<world
            and count==50000//world, "invalid declared execution partition")
    is_worker = worker_partition is not None
    expected = {
        "schema": RUN_SCHEMA, "variant": variant, "status": "complete",
        "context_seed": 20261111, "case": 0, "context_size": 4096,
        "requested_count": count, "scientific_run": not is_worker,
        "end_to_end": not is_worker, "run_purpose": "performance_worker" if is_worker else "performance_execution",
        "performance_first_authorized": True,
        "absolute_accuracy_gate_required_for_performance": False,
        "tail_family_changed": False, "sources_unchanged": True,
        "partial_vector_filtering": False, "clipping_or_repair_performed": False,
        "observation_preprocessing": "raw; TabPFN frozen preprocessing",
        "log_density": "not_used", "microbatch": 50000, "query_batch_size": manifest.get("query_batch_size") if manifest.get("runner_source_sha256") in (RUNNER_V3_SHA256,RUNNER_V4_SHA256) else 1024,
        "rank": rank, "world_size": world, "completed_stage_indices": list(range(21)),
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
    source_identity = validate_runner_source(manifest)
    chunk = manifest.get("query_batch_size")
    if source_identity["version"] in (3,4):
        expected_mode={"execution_cache_v1":"resident_full","execution_mixture_v1":"component_mass",
                       "execution_hybrid_v1":"component_hybrid","execution_gqa_hybrid_v1":"component_gqa_hybrid","execution_gqa_hybrid_fifo_v1":"component_gqa_hybrid"}[variant]
        require(manifest.get("mode") == expected_mode, "version3 mode/variant mismatch")
        require(type(chunk) is int and (chunk == 1024 or variant in GQA_VARIANTS and chunk in LARGE_BATCHES),
                "unqualified version3 sampler chunk")
        require(manifest.get("engine_query_batch_cap") == (10000 if chunk in LARGE_BATCHES else 1024),
                "actual engine query cap differs from sampler/qualified cap")
        require(manifest.get("torch_float32_matmul_precision") == "highest"
                and manifest.get("cuda_matmul_allow_tf32") is False, "version3 frozen FP32/TF32-off flags changed")
    else:
        require(chunk == 1024, "historical runner only qualified sampler chunk1024")
    if source_identity["version"]==4:
        validate_fifo_runtime_metadata(manifest,rank=rank,world=world,count=count)
    else:
        require(variant!="execution_gqa_hybrid_fifo_v1" and manifest.get("proposal_lookahead",0)==0
                and manifest.get("proposal_buffer") is None,"historical runner cannot claim FIFO")
    if variant in GQA_VARIANTS:
        validate_gqa_runtime_metadata(manifest)
    descriptor = bindings.validate_adapter_descriptor(manifest.get("compatibility_adapter"))
    validate_engine_metadata(manifest.get("execution_engine"))
    if source_identity["version"] in (3,4):
        actual_engine=manifest["execution_engine"]
        require(actual_engine.get("query_batch_size")==chunk and actual_engine.get("posterior_count")==count
                and actual_engine.get("sampling_rng_policy")=="independent_cpu_sampling_streams_model_rng_isolated",
                "version3 sampler engine chunk/count/RNG metadata mismatch")
    original_settings = dict(control["regressor_settings"])
    actual_settings = dict(manifest["regressor_settings"])
    require(actual_settings.get("fit_mode") == "fit_with_cache", "resident executor requires actual fit_with_cache metadata")
    actual_settings["fit_mode"] = original_settings["fit_mode"]
    policy = manifest.get("sampling_policy", {})
    if variant in ("execution_mixture_v1", "execution_hybrid_v1", *GQA_VARIANTS):
        validate_sampling_policy(policy)
    else:
        require(policy.get("head_policy") == "not_used" and policy.get("mass_envelope") is None
                and policy.get("stage0_dedup", {}).get("enabled") is False,
                "resident-full policy must not claim component sampling")
    if variant == "execution_hybrid_v1":
        from posteriorforge import execution_hybrid as hybrid
        approved = hybrid.execution_metadata({10:[5],13:[6]})
        actual = manifest.get("hybrid_execution", {})
        require(isinstance(actual, dict) and all(actual.get(key) == value for key,value in approved.items()),
                "declared hybrid source/member fallback policy mismatch")
    streams = policy.get("streams", {})
    require(set(streams) == set(STREAM_ROLES), "complete sampling stream policy required")
    seeds = {}
    for name, role in STREAM_ROLES.items():
        expected_seed = derive_seed(20261111, role, case=0, rank=rank, repetition=manifest["repetition"])
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
    adjusted.setdefault("upstream_revision",UPSTREAM_REVISION)
    adjusted.setdefault("upstream_files",manifest["baseline_execution_source"]["upstream_files"])
    adjusted.update(regressor_settings=actual_settings, microbatch=10000, repetition=0,
                    rank=0,world_size=1,scientific_run=True,
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


def hybrid_qualification_review(path, manifest, qualification):
    from posteriorforge import execution_hybrid as hybrid
    require(manifest.get("variant") == "execution_hybrid_v1", "hybrid proof cannot qualify an all-cache engine")
    require(qualification.get("status") == "hybrid_screen_pass" and qualification.get("qualified") is True
            and qualification.get("full21_hybrid_qualified") is True
            and qualification.get("full21_component_qualified") is True
            and qualification.get("full21_sampler_qualified") is True
            and qualification.get("full21_qualified") is False
            and qualification.get("cache_only_full21_qualified") is False,
            "explicit complete hybrid proof required; failed all-cache qualification must stay false")
    script=Path(__file__).with_name("assemble_hybrid_qualification.py")
    require(qualification.get("assembler_source_sha256")==sha256_file(script),
            "hybrid proof assembler source changed")
    spec=importlib.util.spec_from_file_location("execution_review_hybrid_assembler",script)
    assembler=importlib.util.module_from_spec(spec);spec.loader.exec_module(assembler)
    sources=assembler.current_sources()
    for key,value in sources.items():
        require(qualification.get(key)==value,"assembled hybrid source mismatch: "+key)
    approved=hybrid.execution_metadata({10:[5],13:[6]})
    actual=manifest.get("hybrid_execution",{})
    require(all(actual.get(key)==value for key,value in approved.items()),
            "execution hybrid source/policy differs from qualified policy")
    require(qualification.get("fallback_map")==approved["fallback_map"]
            and qualification.get("fallback_policy_sha256")==approved["fallback_policy_sha256"],
            "assembled fallback map/policy identity mismatch")
    require(qualification.get("thresholds")==QUALIFICATION_THRESHOLDS
            and qualification.get("row_counts")==[1,16,128,1024]
            and qualification.get("icdf_contract_version")==2
            and qualification.get("sources_unchanged") is True
            and qualification.get("input_statuses_preserved") is True,
            "hybrid numerical/source/input-status contract changed")
    expected={"control_variant":"cdf_arithmetic_v1","prefix_manifest_sha256":CONTROL_MANIFEST_SHA256,
              "prefix_samples_sha256":CONTROL_SAMPLES_SHA256,
              "fixed_prefix_row_indices":np.linspace(0,49999,1024,dtype=int).tolist()}
    for key in ("checkpoint_sha256","scientific_fingerprint","parameter_order","context_theta","context_obs","query"):
        expected[key]=manifest[key]
    for key,value in expected.items():
        require(qualification.get(key)==value,"hybrid scientific/query/prefix binding mismatch: "+key)
    validate_engine_metadata(manifest.get("execution_engine"))
    require(qualification.get("stage_coverage")==list(range(21))
            and qualification.get("component_stage_coverage")==list(range(21)),
            "complete hybrid stage coverage required")
    common={key:qualification[key] for key in assembler.COMMON}
    inputs={}
    root=Path(path).resolve().parent
    for identity in qualification.get("source_inputs",[]):
        archive=bindings.within(root,identity["archived_manifest"])
        require(sha256_file(archive)==identity["sha256"],"archived producer proof checksum mismatch")
        value=json.loads(archive.read_text())
        require(value.get("status")==identity["producer_status"]
                and value.get("qualified")==identity["producer_qualified"]
                and value.get("full21_qualified")==identity["producer_full21_qualified"],
                "failed/partial producer proof was reclassified")
        assembler.header_check(value,identity["kind"],common,sources)
        require(identity["sha256"] not in inputs,"duplicate archived source proof")
        inputs[identity["sha256"]]=(identity,value)
    require(inputs,"authenticated original stage proofs required")
    rows=qualification.get("stage_results",[])
    require([row.get("stage_index") for row in rows]==list(range(21)),"assembled hybrid rows incomplete")
    for stage,row in enumerate(rows):
        is_hybrid=str(stage) in approved["fallback_map"]
        assembler.stage_check(row,stage,sources,hybrid=is_hybrid)
        selected=row.get("selected_source",{})
        require(selected.get("sha256") in inputs,"selected stage source is not archived")
        identity,value=inputs[selected["sha256"]]
        require(identity["kind"]==("hybrid" if is_hybrid else "cache")
                and selected.get("producer_status")==identity["producer_status"]
                and row.get("assembled_route")==("hybrid" if is_hybrid else "cached"),
                "assembled stage/source route mismatch")
        original=value if is_hybrid else next((r for r in value.get("stage_results",[]) if r.get("stage_index")==stage),None)
        copied={key:value for key,value in row.items() if key not in ("assembled_route","selected_source")}
        require(original is not None and copied==original,"assembled numerical stage record differs from original producer")
        expected_routes={str(i):("uncached" if i in approved["fallback_map"].get(str(stage),[]) else "cached") for i in range(8)}
        require(qualification.get("stage_routes",{}).get(str(stage),{}).get("member_routes")==expected_routes,
                "assembled per-member route map changed")
        if is_hybrid:
            require(original.get("fallback_members")==approved["fallback_map"][str(stage)]
                    and original.get("member_routes")==expected_routes,"fresh fallback member policy changed")
            for member in original["component_qualification"]["members"]:
                require(member.get("execution_route")==expected_routes[str(member["ensemble_index"])]
                        and member.get("route_counter_delta")==1 and member.get("other_route_counter_delta")==0,
                        "fallback did not execute the declared real route")
    return {"manifest_file":str(Path(path).resolve()),"manifest_sha256":sha256_file(path),
            "stage_coverage":list(range(21)),"full21_qualified":False,"full21_hybrid_qualified":True,
            "full21_component_qualified":True,"full21_sampler_qualified":True,
            "scope":"full21_declared_hybrid_fixed_conditional_diagnostic","icdf_contract_version":2,
            "qualification":qualification,"not_a_proof_of_posterior_equivalence":True,
            "failed_cache_producer_statuses_preserved":True}


def gqa_hybrid_qualification_review(path, manifest, qualification):
    from posteriorforge import performance_runner as runner, execution_cache as cache
    from posteriorforge import execution_mixture as mixture, execution_hybrid as hybrid
    from posteriorforge import execution_gqa_hybrid as execution
    require(manifest.get("variant") in GQA_VARIANTS,
            "GQA proof cannot qualify another execution family")
    require(validate_runner_source(manifest)["version"] in (2,3,4) and manifest.get("query_batch_size")==1024,
            "legacy GQA proof requires original1024 sampler path")
    bindings_value = {key:manifest[key] for key in
        ("scientific_fingerprint", "parameter_order", "context_seed", "case", "context_size",
         "context_theta", "context_obs", "query", "checkpoint_sha256")}
    require(sha256_file(Path(__file__).with_name("recover_gqa_hybrid_qualification.py")) == GQA_RECOVERY_SHA256,
            "reviewed recovery source changed")
    runner.validate_gqa_hybrid_qualification(qualification, bindings=bindings_value,
        cache_sha256=cache.execution_source_sha256(), mixture_sha256=mixture.execution_source_sha256(),
        hybrid_sha256=hybrid.execution_source_sha256(), query_batch_size=1024, qualification_path=path)
    require(qualification.get("prefix_manifest_sha256") == CONTROL_MANIFEST_SHA256
            and qualification.get("prefix_samples_sha256") == CONTROL_SAMPLES_SHA256
            and qualification.get("fixed_prefix_row_indices") == np.linspace(0,49999,1024,dtype=int).tolist(),
            "GQA original fixed-prefix binding changed")
    bindings.validate_adapter_descriptor(qualification.get("compatibility_adapter"))
    require(manifest.get("qualification_producer_schema") == qualification["schema"]
            and manifest.get("qualification_recovery_source_sha256") == qualification.get("recovery_source_sha256"),
            "runner did not consume the declared GQA proof/recovery")
    actual = validate_gqa_runtime_metadata(manifest)
    helper = runner.gqa_recovery_helper()
    fingerprints = []
    for stage, row in enumerate(qualification["stage_results"]):
        for comparison in row["comparisons"]:
            descriptors = helper.gqa_report(comparison["actual_gqa_execution"], stage, GQA_MODULE_SHA256)
            fingerprints.extend({key:descriptor[key] for key in actual} for descriptor in descriptors)
    require(fingerprints and all(value == actual for value in fingerprints),
            "actual attention code fingerprints differ from qualified executions")
    require(execution.execution_source_sha256() == GQA_WRAPPER_SHA256, "GQA wrapper changed after proof validation")
    return {"manifest_file":str(Path(path).resolve()), "manifest_sha256":sha256_file(path),
            "stage_coverage":list(range(21)), "full21_qualified":False,
            "full21_gqa_hybrid_qualified":True, "full21_component_qualified":True,
            "full21_sampler_qualified":True, "scope":"full21_GQA_hybrid_fixed_conditional_diagnostic",
            "icdf_contract_version":2, "qualification":qualification,
            "raw_producer_status_preserved":qualification.get("raw_status_preserved", False),
            "recovery_scope":qualification.get("recovery_scope"),
            "not_a_proof_of_posterior_equivalence":True}


def portable_sdk_root(qualification_path, explicit=None):
    """A local locator, never an identity; all22 bytes are authenticated afterward."""
    if explicit is not None:
        path=Path(explicit).resolve()
        require(path.is_dir(), "explicit current SDK source directory missing")
        return path
    for parent in Path(qualification_path).resolve().parents:
        path=parent/"envs/inference/lib/python3.11/site-packages/tabpfn"
        if path.is_dir():return path
    raise ValueError("current SDK source tree required: pass --sdk-root; historical paths are not identities")


def gqa_batch_qualification_review(path,manifest,qualification,*,sdk_root=None):
    from posteriorforge import performance_runner as runner, execution_cache as cache
    from posteriorforge import execution_mixture as mixture, execution_hybrid as hybrid
    require(manifest.get("variant") in GQA_VARIANTS
            and validate_runner_source(manifest)["version"] in (3,4), "portable large-batch proof requires version3 GQA hybrid")
    chunk=manifest.get("query_batch_size")
    require(type(chunk) is int and chunk in LARGE_BATCHES,"declared qualified neural sampler chunk required")
    require(manifest.get("engine_query_batch_cap")==10000,"large sampler chunk requires actual10000 engine cap")
    root=portable_sdk_root(path,sdk_root)
    require(sha256_file(Path(__file__).with_name("qualify_execution_gqa_batches.py"))==GQA_BATCH_QUALIFIER_SHA256,
            "reviewed large-batch qualification producer changed")
    expected={key:manifest[key] for key in ("checkpoint_sha256","scientific_fingerprint","parameter_order",
                                           "context_theta","context_obs","query")}
    runner.validate_gqa_batch_qualification(qualification,bindings=expected,
        cache_sha256=cache.execution_source_sha256(),mixture_sha256=mixture.execution_source_sha256(),
        hybrid_sha256=hybrid.execution_source_sha256(),query_batch_size=chunk,qualification_path=path,sdk_root=root)
    require(qualification.get("prefix_manifest_sha256")==CONTROL_MANIFEST_SHA256
            and qualification.get("prefix_samples_sha256")==CONTROL_SAMPLES_SHA256
            and qualification.get("fixed_prefix_row_indices")==np.linspace(0,49999,1024,dtype=int).tolist(),
            "portable batch fixed-prefix source/order changed")
    bundle=qualification["portable_source_authentication"]
    require(bundle.get("marshal_code_fingerprints_are_within_process_execution_witnesses_not_cross_install_path_identifiers") is True
            and bundle.get("verification_requires_original_machine_paths") is False
            and bundle.get("verification_uses_file_ctime") is False,
            "portable source/marshal interpretation changed")
    require(manifest.get("qualification_producer_schema")==GQA_BATCH_SCHEMA
            and manifest.get("qualification_recovery_source_sha256") is None,
            "runner did not consume the portable batch proof independently of historical recovery")
    bindings.validate_adapter_descriptor(qualification.get("compatibility_adapter"))
    # Marshaled code includes co_filename. Authenticate cross-install identity
    # through source bytes/RECORD, and runtime execution through its own process
    # original==executed!=active plus stable counters. Never compare to old paths.
    runtime_fingerprints=validate_gqa_runtime_metadata(manifest)
    return {"manifest_file":str(Path(path).resolve()),"manifest_sha256":sha256_file(path),
            "stage_coverage":list(range(21)),"full21_qualified":False,
            "full21_gqa_hybrid_qualified":True,"full21_component_qualified":True,"full21_sampler_qualified":True,
            "full21_large_batch_qualified":True,"qualified_sampler_chunk":chunk,"engine_query_batch_cap":10000,
            "scope":"full21 fixed-conditionals at declared larger batch; repeated1032-row pool is not a posterior",
            "icdf_contract_version":2,"qualification":qualification,
            "portable_source_verification":{"current_sdk_tree":str(root),"tree_path_is_identity":False,
                "sdk_python_source_count":len(bundle["sdk_source_sha256"]),"wheel_RECORD_checked":True,
                "historical_install_ctime_replayed":False,"historical_base_role":"authenticated provenance only",
                "marshal_role":"within-process witnesses; not cross-install path identities",
                "runtime_fingerprints_self_consistent":runtime_fingerprints},
            "not_a_proof_of_posterior_equivalence":True}


def qualification_review(path, manifest, *, context, query, sdk_root=None):
    require(sha256_file(path) == manifest.get("qualification_manifest_sha256"),
            "execution qualification manifest SHA256 mismatch")
    q = json.loads(Path(path).read_text())
    if q.get("schema")==GQA_BATCH_SCHEMA:
        return gqa_batch_qualification_review(path,manifest,q,sdk_root=sdk_root)
    if manifest.get("runner_source_sha256") in (RUNNER_V3_SHA256,RUNNER_V4_SHA256) and manifest.get("query_batch_size") in LARGE_BATCHES:
        raise ValueError("version3 large sampler chunk requires portable all21 batch proof")
    if q.get("schema") in ("posteriorforge.execution_gqa_hybrid_qualification.v1",
                            "posteriorforge.execution_gqa_hybrid_recovery.v1"):
        return gqa_hybrid_qualification_review(path,manifest,q)
    require(manifest.get("variant") not in GQA_VARIANTS,
            "GQA hybrid requires its own complete source-bound proof")
    if q.get("schema")=="posteriorforge.execution_hybrid_qualification.v1":
        return hybrid_qualification_review(path,manifest,q)
    require(manifest.get("variant")!="execution_hybrid_v1","hybrid engine requires its separately assembled hybrid proof")
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


def reviewed_orchestrator(manifest):
    script=Path(__file__).with_name("run_execution_workers.py")
    identity=manifest.get("orchestrator_source_sha256")
    if identity==ORCHESTRATOR_SHA256:
        content=subprocess.run(["git","-C",str(script.parents[1]),"show",
            RUNNER_V2_GIT_REVISION+":scripts/run_execution_workers.py"],check=True,capture_output=True).stdout
        require(hashlib.sha256(content).hexdigest()==ORCHESTRATOR_SHA256,"historical orchestrator Git blob changed")
        module=ModuleType("reviewed_historical_worker_orchestrator")
        module.__file__=str(script);exec(compile(content,str(script),"exec"),module.__dict__)
        return module,{"sha256":identity,"historical_git_revision":RUNNER_V2_GIT_REVISION}
    if identity==ORCHESTRATOR_V4_SHA256:
        content=subprocess.run(["git","-C",str(script.parents[1]),"show",
            RUNNER_V4_GIT_REVISION+":scripts/run_execution_workers.py"],check=True,capture_output=True).stdout
        require(hashlib.sha256(content).hexdigest()==ORCHESTRATOR_V4_SHA256,"historical version4 orchestrator Git blob changed")
        module=ModuleType("reviewed_version4_worker_orchestrator");module.__file__=str(script)
        exec(compile(content,str(script),"exec"),module.__dict__)
        return module,{"sha256":identity,"historical_git_revision":RUNNER_V4_GIT_REVISION}
    require(identity==ORCHESTRATOR_V3_SHA256,"unreviewed aggregate orchestrator source")
    content=subprocess.run(["git","-C",str(script.parents[1]),"show",
        RUNNER_V3_GIT_REVISION+":scripts/run_execution_workers.py"],check=True,capture_output=True).stdout
    require(hashlib.sha256(content).hexdigest()==ORCHESTRATOR_V3_SHA256,"historical version3 orchestrator Git blob changed")
    module=ModuleType("reviewed_version3_worker_orchestrator");module.__file__=str(script)
    exec(compile(content,str(script),"exec"),module.__dict__)
    return module,{"sha256":identity,"historical_git_revision":RUNNER_V3_GIT_REVISION}


def authenticate_worker_aggregate(path, manifest, *, context, query, control, custody, qualification_path):
    """Authenticate all complete leaf cohorts and exact rank-order concatenation."""
    script=Path(__file__).with_name("run_execution_workers.py")
    orchestrator,orchestrator_identity=reviewed_orchestrator(manifest)
    orchestrator_sha=orchestrator_identity["sha256"]
    from posteriorforge import performance_runner
    root=Path(path).resolve().parent
    world=manifest.get("world_size")
    require(type(world) is int and world in (1,2,4),"aggregate world size must be1/2/4")
    required={"schema":"posteriorforge.inference.execution.aggregate.v1","variant":"execution_workers_v1",
              "status":"complete","requested_count":50000,"returned_count":50000,
              "end_to_end":True,"rank":None,"run_purpose":"performance_e2e_aggregate",
              "aggregate_published":True,"worker_counts":orchestrator.partition_counts(world),
              "worker_return_codes":[0]*world}
    for key,value in required.items():
        require(manifest.get(key) is value if isinstance(value,bool) else manifest.get(key)==value,
                "aggregate contract mismatch: "+key)
    aggregate_runner = validate_runner_source(manifest, aggregate=True)
    qsha=sha256_file(qualification_path)
    require(manifest.get("qualification_manifest_sha256")==qsha,"aggregate qualification SHA mismatch")
    underlying=manifest.get("worker_execution_variant")
    modes={"execution_cache_v1":"resident_full","execution_mixture_v1":"component_mass",
           "execution_hybrid_v1":"component_hybrid","execution_gqa_hybrid_v1":"component_gqa_hybrid","execution_gqa_hybrid_fifo_v1":"component_gqa_hybrid"}
    require(underlying in modes and manifest.get("mode")==modes[underlying],"aggregate execution family mismatch")
    chunk=manifest.get("query_batch_size",1024)
    if aggregate_runner["version"] in (3,4):
        require(type(chunk) is int and (chunk==1024 or underlying in GQA_VARIANTS and chunk in LARGE_BATCHES),
                "unqualified aggregate sampler chunk")
        require(manifest.get("engine_query_batch_cap")==(10000 if chunk in LARGE_BATCHES else 1024),
                "aggregate actual query cap mismatch")
    expected={key:control[key] for key in ("scientific_fingerprint","parameter_order","context_seed","case","context_size",
                                          "context_theta","context_obs","query","checkpoint_sha256","baseline_execution_source")}
    expected.update(mode=modes[underlying],variant=underlying,reference_binding=custody["runner_reference_binding"])
    for key,value in expected.items():
        if key!="variant":require(manifest.get(key)==value,"aggregate science/source binding mismatch: "+key)
    if aggregate_runner["version"]==4:
        v4=reviewed_v4_runner();lookahead=manifest.get("proposal_lookahead")
        v4.validate_proposal_lookahead(modes[underlying],lookahead)
        require(underlying==v4.execution_variant(modes[underlying],lookahead)
                and manifest.get("proposal_buffer_policy")==v4.proposal_buffer_policy(lookahead),
                "aggregate FIFO source/variant/policy mismatch")
        expected.update(proposal_lookahead=lookahead,proposal_buffer_policy=v4.proposal_buffer_policy(lookahead))
    else:
        require(underlying!="execution_gqa_hybrid_fifo_v1" and manifest.get("proposal_lookahead",0)==0,
                "historical aggregate cannot claim FIFO")
    contracts=[];paths=[]
    devices=manifest.get("visible_device_tokens",[])
    require(len(devices)==world and len(set(devices))==world,"distinct aggregate GPU device assignments required")
    identities=manifest.get("worker_contracts",[])
    require(len(identities)==world,"all worker contracts required")
    for rank,identity in enumerate(identities):
        contract_path=Path(identity["path"])
        require(sha256_file(contract_path)==identity["sha256"],"worker contract SHA mismatch")
        contract=json.loads(contract_path.read_text())
        require(contract.get("schema")=="posteriorforge.execution_worker_contract.v1"
                and contract.get("rank")==rank and contract.get("world_size")==world
                and contract.get("worker_count")==50000//world
                and contract.get("expected_output")==str((root/f"worker-{rank}").resolve())
                and contract.get("aggregate_output")==str(root)
                and contract.get("orchestrator_source_sha256")==orchestrator_sha
                and contract.get("parent_job_id")==str(manifest["budget"]["job_id"])
                and contract.get("budget_reservation_id")==manifest["budget"]["reservation_id"]
                and contract.get("visible_device_token")==devices[rank],
                "worker contract rank/output/source binding changed")
        if aggregate_runner["version"]==4:
            require(contract.get("proposal_lookahead")==manifest["proposal_lookahead"],
                    "worker contract FIFO lookahead mismatch")
        contracts.append(contract);paths.append(contract_path)
    combined,records=orchestrator.validate_worker_payloads(contracts,paths,expected_bindings=expected,
        driver_schema=RUN_SCHEMA,qualification_sha256=qsha)
    require(manifest.get("workers")==records,"aggregate worker hash/count records differ from actual children")
    workers=[];all_seeds=[];rejections=[]
    for rank,contract in enumerate(contracts):
        worker_path=Path(contract["expected_output"])/"manifest.json"
        worker=json.loads(worker_path.read_text())
        validate_execution_header(worker,context=context,query=query,control=control,
            worker_partition={"rank":rank,"world_size":world,"requested_count":50000//world})
        bindings.validate_run_reference_binding(worker,custody)
        require(worker.get("qualification_manifest_sha256")==qsha
                and worker.get("variant")==underlying,"worker qualification/execution family mismatch")
        require(worker.get("runner_source_sha256") == manifest.get("runner_source_sha256")
                and worker.get("query_batch_size")==chunk,
                "aggregate/worker runner source or sampler chunk mismatch")
        if aggregate_runner["version"] in (3,4):
            require(contract.get("query_batch_size",1024)==chunk,"worker contract sampler chunk mismatch")
            require(worker.get("engine_query_batch_cap")==manifest.get("engine_query_batch_cap"),
                    "aggregate/worker native query cap mismatch")
        all_seeds.extend(worker["seed_streams"].values());rejections.append(worker["rejection"]);workers.append(worker)
    require(len(set(all_seeds))==len(all_seeds),"worker/rank sampling streams collide")
    require(manifest.get("rank_seed_streams")==[worker["seed_streams"] for worker in workers],
            "aggregate rank-seed metadata differs from actual children")
    values=bindings.load_samples(Path(path),manifest)
    require(np.array_equal(values,combined),"aggregate cohort is not exact rank-order worker concatenation")
    validate_engine_metadata(manifest.get("execution_engine"))
    if underlying=="execution_hybrid_v1":
        require(manifest.get("hybrid_execution")==workers[0].get("hybrid_execution"),
                "aggregate hybrid policy differs from authenticated workers")
    if underlying in GQA_VARIANTS:
        require(manifest.get("gqa_hybrid_execution") == workers[0].get("gqa_hybrid_execution")
                and all(worker.get("gqa_hybrid_execution")==workers[0]["gqa_hybrid_execution"] for worker in workers),
                "aggregate GQA policy differs across authenticated workers")
        portable=aggregate_runner["version"] in (3,4) and chunk in LARGE_BATCHES
        if portable:
            require(manifest.get("qualification_family")=="large_batch_v1"
                    and all(worker.get("qualification_family")=="large_batch_v1" for worker in workers),
                    "aggregate portable proof family differs from workers")
        else:
            require(all(worker["gqa_prediction_checks"]["attention_fingerprints"]
                        ==workers[0]["gqa_prediction_checks"]["attention_fingerprints"] for worker in workers),
                    "aggregate GQA fingerprints differ across authenticated workers")
    if aggregate_runner["version"]==4:
        if manifest["proposal_lookahead"]:
            buffers=[worker["proposal_buffer"] for worker in workers]
            expected_buffer={"schema":"posteriorforge.execution.proposal_buffer.aggregate.v1",**manifest["proposal_buffer_policy"],
                "world_size":world,"worker_initial_lookahead":[row["worker_initial_lookahead"] for row in buffers],
                "generated_count":sum(row["generated_count"] for row in buffers),
                "original_checked_count":sum(row["original_sampler_counts"]["original_checked_count"] for row in buffers),
                "accepted_before_trim":sum(row["original_sampler_counts"]["accepted_before_trim"] for row in buffers),
                "returned_count":50000,"remaining_unused_buffer_count":sum(row["remaining_unused_buffer_count"] for row in buffers),
                "actual_generated_wave_count":sum(row["generated_wave_count"] for row in buffers),"worker_records":buffers}
            require(manifest.get("proposal_buffer")==expected_buffer
                    and sum(expected_buffer["worker_initial_lookahead"])==1200
                    and expected_buffer["generated_count"]==expected_buffer["original_checked_count"]+expected_buffer["remaining_unused_buffer_count"],
                    "aggregate FIFO counts/conservation differ from authenticated workers")
        else:require(manifest.get("proposal_buffer") is None,"zero-lookahead aggregate cannot contain FIFO")
    summed={key:sum(item[key] for item in rejections) for key in
            ("support_calls","proposed","accepted_before_final_trim","returned")}
    detail={"orchestrator_source_sha256":orchestrator_sha,"orchestrator_source_authentication":orchestrator_identity,"runner_source_authentication":aggregate_runner,"worker_count":world,
            "worker_records":records,"rank_seed_streams":[worker["seed_streams"] for worker in workers],
            "rank_order_concatenation_exact":True,"partial_worker_fallback_performed":False,
            "worker_timings":[worker["timings"] for worker in workers],
            "worker_memory":[worker["memory"] for worker in workers],
            "sampling_policies_by_rank":[worker["sampling_policy"] for worker in workers],
            "GQA_runtime_checks_by_rank":[worker.get("gqa_prediction_checks") for worker in workers],
            "GQA_runtime_counters_by_rank":[worker.get("gqa_execution_counters") for worker in workers],
            "proposal_buffer_policy":manifest.get("proposal_buffer_policy"),"proposal_buffer":manifest.get("proposal_buffer"),
            "summed_rejection":summed,
            "rejection_scope":"sum of independent authenticated worker counters; not a single serial rejection process"}
    return values,workers[0],detail


def control_repeat_diagnostics(folder, candidate, primary, reference, *, custody, context, query):
    """Independent corrected controls contextualize flags without removing them."""
    from posteriorforge import performance_control
    from posteriorforge.inference_cdf import guard_baseline_source
    folder=Path(folder)
    paths=[folder/f"rep-{rep}/seed-20261111-case-0/manifest.json" for rep in range(3)]
    if not all(path.is_file() for path in paths):
        return {"status":"not_available","cohorts_fabricated":False,"expected_manifest_files":[str(p) for p in paths]}
    records=[];seeds=[]
    for rep,path in enumerate(paths):
        manifest=json.loads(path.read_text())
        expected={"schema":"posteriorforge.inference.execution.v1","variant":"performance_control_v1","status":"complete",
                  "run_purpose":"performance_control","context_seed":20261111,"case":0,"context_size":4096,
                  "requested_count":50000,"scientific_run":True,"repetition":rep,
                  "performance_first_authorized":True,"absolute_accuracy_gate_required_for_performance":False,
                  "execution_optimized":False,"tail_family_changed":False,"metadata_writer_restored":True}
        for key,value in expected.items():
            require(manifest.get(key) is value if isinstance(value,bool) else manifest.get(key)==value,
                    "independent corrected-control contract mismatch: "+key)
        require(manifest.get("baseline_execution_source")==guard_baseline_source()
                and manifest.get("execution_engine",{}).get("module_sha256")==sha256_file(performance_control.__file__)
                and manifest["execution_engine"].get("engine_id")=="corrected_unoptimized"
                and manifest["execution_engine"].get("cache_policy")=="original_fit_preprocessors"
                and manifest["execution_engine"].get("rng_role")=="posterior",
                "independent corrected-control source/path changed")
        expected_seed=derive_seed(20261111,"posterior",case=0,repetition=rep)
        require(manifest.get("seed")==expected_seed,"independent control posterior RNG repetition mismatch")
        seeds.append(expected_seed)
        adjusted=copy.deepcopy(manifest);adjusted.update(repetition=0,seed=derive_seed(20261111,"posterior",case=0))
        verify_run_binding(adjusted,context,query,20261111,0)
        descriptor=bindings.validate_adapter_descriptor(manifest["compatibility_adapter"])
        require(descriptor["patched_call_count"]==manifest.get("expected_patched_call_count")
                ==21*8*manifest["rejection"]["support_calls"],"corrected-control call coverage changed")
        bindings.validate_run_reference_binding(manifest,custody)
        samples=bindings.load_samples(path,manifest)
        own=comparative_assessment(samples,primary,reference)
        candidate_review=comparative_assessment(candidate,samples,reference)
        require(own["sample_integrity"]["passed"],"independent control samples are invalid")
        records.append({"repetition":rep,"seed":expected_seed,"manifest_file":str(path),
                        "manifest_sha256":sha256_file(path),"samples_sha256":manifest["samples_sha256"],
                        "identical_to_primary_payload":manifest["samples_sha256"]==CONTROL_SAMPLES_SHA256,
                        "control_vs_primary":own["relative_distribution_review"],
                        "control_absolute_physical_assessment":own["absolute_physical_assessment"],
                        "candidate_vs_this_control":candidate_review["relative_distribution_review"],
                        "timings":manifest["timings"],"wrapper_total_seconds":manifest["wrapper_total_seconds"]})
    require(len(set(seeds))==3,"control posterior RNG repetitions are not distinct")
    return {"status":"available","distinct_posterior_rng_repetitions":3,"cohorts":records,
            "primary_old_control_is_not_a_fourth_independent_repetition":True,
            "flags_suppressed_or_thresholds_changed":False,
            "interpretation":"Control-to-control flags are diagnostic evidence of finite-cohort variation in the unchanged implemented sampler. They do not waive candidate flags or establish population equivalence."}


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
    aggregate_detail=None
    if manifest.get("schema")=="posteriorforge.inference.execution.aggregate.v1":
        candidate,effective_manifest,aggregate_detail=authenticate_worker_aggregate(
            manifest_path,manifest,context=context,query=query,control=control_manifest,
            custody=custody,qualification_path=args.qualification_manifest)
        qualification=qualification_review(args.qualification_manifest,effective_manifest,context=context,query=query,sdk_root=getattr(args,"sdk_root",None))
    else:
        validate_execution_header(manifest,context=context,query=query,control=control_manifest)
        bindings.validate_run_reference_binding(manifest,custody)
        qualification=qualification_review(args.qualification_manifest,manifest,context=context,query=query,sdk_root=getattr(args,"sdk_root",None))
        candidate=bindings.load_samples(manifest_path,manifest)
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
                      "samples_sha256": manifest["samples_sha256"], "samples_descriptor": manifest["samples"], "hybrid_execution": manifest.get("hybrid_execution"),
                      "runner_source_authentication":validate_runner_source(effective_manifest if aggregate_detail else manifest),
                      "gqa_hybrid_execution":manifest.get("gqa_hybrid_execution"),
                      "proposal_lookahead":manifest.get("proposal_lookahead",0),"proposal_buffer_policy":manifest.get("proposal_buffer_policy"),
                      "proposal_buffer":manifest.get("proposal_buffer"),
                      "gqa_prediction_checks":manifest.get("gqa_prediction_checks"),
                      "gqa_execution_counters":manifest.get("gqa_execution_counters"),
                      "execution_policy": manifest.get("execution_policy"), "execution_engine": manifest["execution_engine"],
                      "sampling_rng_policy": manifest.get("sampling_rng_policy"), "sampling_policy": manifest.get("sampling_policy"),
                      "timings": manifest["timings"], "memory": manifest.get("memory"), "aggregate_authentication": aggregate_detail},
        "reference": custody,
        "rejection_review": {
            "control": rejection_review(control_manifest.get("rejection"), returned=50000),
            "candidate": rejection_review(aggregate_detail["summed_rejection"] if aggregate_detail else manifest.get("rejection"), returned=50000)},
        "source": {"evaluation_script_sha256": sha256_file(__file__),
                   "binding_helper_sha256": sha256_file(_BINDING_PATH),
                   "original_validation_sha256": sha256_file(Path(__file__).resolve().parents[1]/"src/posteriorforge/validation.py"),
                   "legacy_binding_copy_supplements": {
                       "missing_upstream_revision":"constant authenticated by the frozen original inference source hash",
                       "missing_upstream_files":"independently authenticated baseline_execution_source.upstream_files",
                       "actual_manifest_changed":False}},
        "interpretation": [
            "Relative CDF differences are exact two-sample ECDF distances, not reference-grid bounds; no reference numerical error is added to them.",
            "Absolute physical CDF bounds include the unchanged saved-reference uncertainty and whole-line monotonicity/tail bounds.",
            "The .01 CDF, .05 reference-SD mean and .05 SD-fraction thresholds flag investigation; they neither establish population equivalence nor waive absolute scientific failure.",
            "Changed forward/batch/RNG consumption can change finite Monte Carlo cohorts even if fixed-conditional distributions agree.",
            "FIFO lookahead generates extra complete unfiltered vectors; original API deliveries, support checks, final accepted trim and unused rows remain separately counted. Extra stream consumption changes seeded row mapping; no pointwise cohort equality is claimed.",
            "A fixed-conditional screen only covers its declared prefixes, stages and chunks; accepted-prefix probes do not establish the unfiltered proposal or rejection-conditioned joint law.",
            "Performance-first authorization explicitly permits engineering review while the original physical accuracy criteria remain failed and visible.",
        ],
    }
    result["control_repetition_context"]=control_repeat_diagnostics(
        results/"performance-first-v1/controls",candidate,control,reference,
        custody=custody,context=context,query=query)
    tail_path=results/"performance-first-v1/control-tail-diagnostic/science-control-repeat-tails.json"
    result["existing_control_tail_diagnostic"] = ({"file":str(tail_path),"sha256":sha256_file(tail_path),
        "role":"existing diagnostic context only; no row removed, criterion waived or artifact cause assumed"}
        if tail_path.is_file() else None)
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
                      "full21_hybrid_fixed_conditional_qualified":qualification.get("full21_hybrid_qualified",False),
                      "full21_gqa_hybrid_fixed_conditional_qualified":qualification.get("full21_gqa_hybrid_qualified",False),
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
    parser.add_argument("--sdk-root", help="current SDK source tree for portable byte verification; path is not identity")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
