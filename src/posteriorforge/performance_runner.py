"""Explicit 50k execution experiments with unchanged whole-vector rejection.

Built with PriorLabs-TabPFN. This module is opt-in and import-safe: no Torch or
model import occurs before the actual budgeted GPU entry point. Physical
accuracy remains visible, but is not a prerequisite for the authorized
performance-first phase. Worker outputs never claim to be E2E benchmarks.
"""
from __future__ import annotations
import argparse
from importlib import import_module
import json
import os
from pathlib import Path
import time
import numpy as np
from .artifacts import runtime_identity,sha256_file,write_json
from .banks import derive_seed

SCHEMA="posteriorforge.inference.execution.v1"
COUNT=50000
RUNNER_VERSION=4
PROPOSAL_BUFFER_SHA256="8fb2ba1fb63cd917378ff9bc8b6f70be7410862c509cd22cb2fb6f454d3d35fa"
FIFO_VARIANT="execution_gqa_hybrid_fifo_v1"
BASE_VARIANTS={"resident_full":"execution_cache_v1","component_mass":"execution_mixture_v1","component_hybrid":"execution_hybrid_v1","component_gqa_hybrid":"execution_gqa_hybrid_v1"}
GQA_QUALIFIER_SHA256="ab0c289ad051aa3553ca47402da07ac2643c576e660a297be9242c357787848a"
GQA_RECOVERY_SHA256="83064f54712014b5c4c66569e854872caff03593cfdc337f2472a7b5723aea02"
_GQA_RECOVERY_MODULE=None
GQA_BATCH_QUALIFIER_SHA256="31422927077911879b6cd155605a6a668796305ee97ccf6585819d8fe494cfde"
GQA_LARGE_BATCHES=(4096,6250,10000)
_GQA_BATCH_MODULE=None
QUALIFIER_SHA256="6c927a5c278c7fc54b556ac557ecef64f5e6338f5ae13bf466f3b1e6a4640f90"
HYBRID_FALLBACK_MAP={"10":[5],"13":[6]}
HYBRID_QUALIFIER_SHA256="15e8e9ce844d3352e49534714f0e0569f42e970a3d9876630f877a346eaaf9b6"
HYBRID_ASSEMBLER_SHA256="29dcbc62f273fba0dad50f3a4063b445dbe0329f30371b164237c31a55fb7ae9"
STREAM_ROLES={"head_proposal":"execution-head-proposal","head_acceptance":"execution-head-acceptance","component_uniform":"execution-component-uniform"}
STREAM_DTYPES={"head_proposal":"torch.int64","head_acceptance":"torch.float64","component_uniform":"torch.float32"}
QUALIFICATION_THRESHOLDS={"normalization_absolute":1e-5,"max_cdf_absolute":2e-5,"max_row_total_variation":1e-4,"quantile_coordinate_span_fraction":2e-6}


def validate_proposal_lookahead(mode,proposal_lookahead):
    if type(proposal_lookahead) is not int or proposal_lookahead not in (0,1200) or (proposal_lookahead and mode!="component_gqa_hybrid"):
        raise ValueError("global proposal lookahead0/1200; nonzero requires component_gqa_hybrid")
    return proposal_lookahead


def execution_variant(mode,proposal_lookahead=0):
    validate_proposal_lookahead(mode,proposal_lookahead)
    if mode not in BASE_VARIANTS:raise ValueError("explicit execution mode required")
    return FIFO_VARIANT if proposal_lookahead else BASE_VARIANTS[mode]


def guarded_proposal_buffer():
    from . import execution_proposal_buffer as fifo
    if fifo.source_sha256()!=PROPOSAL_BUFFER_SHA256:raise RuntimeError("reviewed FIFO complete-vector buffer source changed")
    return fifo


def proposal_buffer_policy(proposal_lookahead=0):
    import hashlib
    if type(proposal_lookahead) is not int or proposal_lookahead not in (0,1200):raise ValueError("explicit global lookahead0/1200 required")
    if not proposal_lookahead:return {"enabled":False,"global_lookahead":0,"policy_id":"none"}
    fifo=guarded_proposal_buffer()
    policy={"enabled":True,"global_lookahead":1200,"policy_id":fifo.POLICY_ID,"module_sha256":PROPOSAL_BUFFER_SHA256,
        "original_sampler_source_sha256":fifo.ORIGINAL_SAMPLER_SHA256,"worker_lookahead":"1200/world_size",
        "first_wave":"original_requested_count plus worker lookahead","later_wave":"exact FIFO deficit only",
        "delivery":"original requested count; FIFO complete vectors; unfiltered","rng_draws_by_buffer":0,
        "seed_coupling":"extra rowU/head-stream consumption changes seeded vector mapping; no rowwise equality claim"}
    return {**policy,"policy_sha256":hashlib.sha256(json.dumps(policy,sort_keys=True,separators=(",",":")).encode()).hexdigest()}


def proposal_buffer_scope(bindings,*,context_seed,case,repetition,rank,world_size,seeds):
    return {"case":case,"query":bindings["query"],"context_theta":bindings["context_theta"],"context_obs":bindings["context_obs"],
        "checkpoint_sha256":bindings["checkpoint_sha256"],"scientific_fingerprint":bindings["scientific_fingerprint"],
        "context_seed":context_seed,"repetition":repetition,"rank":rank,"world_size":world_size,"seed_streams":dict(seeds),
        "scope":"one posterior request; no cross-query/case/context reuse"}


def validate_proposal_buffer_record(record,*,world_size,rank,scope=None):
    """Portable CPU validation of actual FIFO deliveries, waves and output counts."""
    import hashlib
    fifo=guarded_proposal_buffer();policy=fifo.worker_policy(world_size,rank)
    required={"enabled":True,"policy_sha256":proposal_buffer_policy(1200)["policy_sha256"],"schema":fifo.SCHEMA,"policy_id":fifo.POLICY_ID,"module_sha256":PROPOSAL_BUFFER_SHA256,**policy,
        "finalized":True,"failed":False,"failure":None,"live_buffer_rows":0,"buffer_reset_count":0,
        "support_filtering_performed_by_buffer":False,"row_repair_or_reordering":False,"partial_vector_filtering":False,
        "all_rows_are_complete_vectors":True,"vector_width":21,"rng_draws_by_buffer":0,"log_density":"not_used",
        "original_sampler_source_sha256":fifo.ORIGINAL_SAMPLER_SHA256,"unused_rows_discarded_only_after_original_sampler_completion":True,
        "source_unchanged":True}
    for key,expected in required.items():
        if type(record.get(key)) is not type(expected) or record.get(key)!=expected:raise ValueError("FIFO record mismatch: "+key)
    saved_scope=record.get("request_scope")
    if not isinstance(saved_scope,dict) or (scope is not None and saved_scope!=scope):raise ValueError("FIFO request scope differs")
    encoded=json.dumps(saved_scope,sort_keys=True,separators=(",",":"),allow_nan=False)
    if hashlib.sha256(encoded.encode()).hexdigest()!=record.get("request_scope_sha256"):raise ValueError("FIFO request scope hash differs")
    api=record.get("api_requested_sizes")
    if not isinstance(api,list) or not api or any(type(n) is not int or n<=0 for n in api) or api[0]!=policy["worker_expected_return_count"]:
        raise ValueError("FIFO original API requests/count contract differ")
    expected_waves=[];remaining=0
    for n in api:
        if n<=remaining:remaining-=n
        else:
            lookahead=policy["worker_initial_lookahead"] if not expected_waves else 0
            expected_waves.append(n-remaining+lookahead);remaining=lookahead
    count=sum(expected_waves);checked=sum(api)
    expected={"actual_generated_wave_sizes":expected_waves,"generated_wave_attempt_sizes":expected_waves,
        "generated_wave_count":len(expected_waves),"required_stage_preparations_for_generated_waves":21*len(expected_waves),
        "generated_count":count,"original_api_delivered_count":checked,"remaining_unused_buffer_count":remaining}
    for key,value in expected.items():
        if type(record.get(key)) is not type(value) or record.get(key)!=value:raise ValueError("FIFO conservation/wave mismatch: "+key)
    result=record.get("original_sampler_counts",{})
    if not isinstance(result,dict) or any(type(result.get(key)) is not int for key in ("original_checked_count","original_support_calls","returned_count","accepted_before_trim")):
        raise ValueError("FIFO typed original support counts required")
    if result.get("original_checked_count")!=checked or result.get("original_support_calls")!=len(api) or result.get("returned_count")!=policy["worker_expected_return_count"]:
        raise ValueError("FIFO original support/output counts differ")
    accepted=result.get("accepted_before_trim")
    if type(accepted) is not int or not policy["worker_expected_return_count"]<=accepted<=checked:raise ValueError("FIFO accepted-before-trim count differs")
    return True


def worker_stream_seeds(context_seed,case,repetition,rank=0,world_size=1):
    if type(world_size) is not int or world_size not in (1,2,4) or type(rank) is not int or not 0<=rank<world_size:
        raise ValueError("rank and world_size1/2/4 required")
    return {name:derive_seed(context_seed,role,case=case,rank=rank,repetition=repetition) for name,role in STREAM_ROLES.items()}


def worker_partition(contract,*,output,job_id,context_seed,case,repetition,runner_sha256):
    """A strict private subcount contract; only an aggregate can claim 50k E2E."""
    if contract is None:return {"rank":0,"world_size":1,"requested_count":COUNT,"run_purpose":"performance_execution","end_to_end":True}
    if contract.get("schema")!="posteriorforge.execution_worker_contract.v1" or contract.get("benchmark_total_count")!=COUNT:
        raise ValueError("explicit aggregate50000 worker contract required")
    world,rank=contract.get("world_size"),contract.get("rank")
    if type(world) is not int or world not in (1,2,4) or type(rank) is not int or not 0<=rank<world or contract.get("worker_count")!=COUNT//world:
        raise ValueError("worker partition must sum to50000 across1/2/4 ranks")
    expected={"parent_job_id":job_id,"context_seed":context_seed,"case":case,"repetition":repetition,
              "runner_source_sha256":runner_sha256,"expected_output":str(Path(output).resolve()),
              "seed_streams":worker_stream_seeds(context_seed,case,repetition,rank,world)}
    if any(contract.get(key)!=value for key,value in expected.items()):raise ValueError("worker contract binding mismatch")
    if not isinstance(contract.get("aggregate_output"),str) or not contract.get("orchestrator_source_sha256"):
        raise ValueError("aggregate output and orchestrator identity required")
    if contract.get("visible_device_token")!=os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise ValueError("worker visible device differs from parent contract")
    return {"rank":rank,"world_size":world,"requested_count":COUNT//world,"run_purpose":"performance_worker","end_to_end":False}


def validate_hybrid_qualification(value,*,bindings,cache_sha256,mixture_sha256,hybrid_sha256,query_batch_size):
    """Validate the distinct hybrid composition without changing any cache flag."""
    required={"schema":"posteriorforge.execution_hybrid_qualification.v1","status":"hybrid_screen_pass","qualified":True,
        "full21_qualified":False,"cache_only_full21_qualified":False,"full21_hybrid_qualified":True,
        "full21_component_qualified":True,"full21_sampler_qualified":True,"sources_unchanged":True,"input_statuses_preserved":True,
        "control_variant":"cdf_arithmetic_v1","icdf_contract_version":2,"cache_module_sha256":cache_sha256,
        "mixture_module_sha256":mixture_sha256,"hybrid_module_sha256":hybrid_sha256,
        "cache_qualifier_sha256":QUALIFIER_SHA256,"hybrid_qualifier_sha256":HYBRID_QUALIFIER_SHA256,
        "assembler_source_sha256":HYBRID_ASSEMBLER_SHA256,"fallback_map":HYBRID_FALLBACK_MAP,
        "thresholds":QUALIFICATION_THRESHOLDS,"row_counts":[1,16,128,1024]}
    for key,expected in required.items():
        if type(value.get(key)) is not type(expected) or value.get(key)!=expected:raise ValueError("hybrid qualification mismatch: "+key)
    from .execution_hybrid import policy_metadata
    if value.get("fallback_policy_sha256")!=policy_metadata(HYBRID_FALLBACK_MAP)["fallback_policy_sha256"]:raise ValueError("hybrid fallback policy hash mismatch")
    if query_batch_size!=1024 or sorted(value.get("stage_coverage",[]))!=list(range(21)):raise ValueError("hybrid all21/query1024 qualification required")
    for key,expected in bindings.items():
        if value.get(key)!=expected:raise ValueError("hybrid qualification binding mismatch: "+key)
    def probability(item):
        if item.get("passed") is not True:raise ValueError("hybrid probability comparison failed")
        for key,limit in (("max_normalization_error",1e-5),("max_cdf_absolute_error",2e-5),("max_row_total_variation",1e-4)):
            number=item.get(key)
            if type(number) not in (int,float) or not np.isfinite(number) or not 0<=number<=limit:raise ValueError("hybrid numeric threshold changed: "+key)
        if not np.isfinite(item.get("minimum_probability",float("nan"))) or item["minimum_probability"]<0:raise ValueError("hybrid nonnegative probabilities required")
    def prediction(item):
        if not all(item.get(key) is True for key in ("passed","eight_members","exact_final_borders","final_float32")):raise ValueError("hybrid full prediction proof incomplete")
        probability(item["final"])
        if len(item.get("members",[]))!=8:raise ValueError("hybrid eight-member comparison required")
        for member in item["members"]:
            probability(member)
            if not all(member.get(key) is True for key in ("configuration_identity_equal","exact_borders","float32")):raise ValueError("hybrid member identity changed")
        inverse=item.get("inverse_cdf",{})
        if inverse.get("passed") is not True or inverse.get("icdf_contract_version")!=2 or inverse.get("same_logits_and_borders_exact_failures")!=0 or inverse.get("independent_fp32_replay_failures")!=0 or inverse.get("probability_space_failures"):raise ValueError("hybrid original inverse-CDF proof incomplete")
    rows=value.get("stage_results",[])
    if sorted(row.get("stage_index",-1) for row in rows)!=list(range(21)):raise ValueError("hybrid all21 stage records required")
    for row in rows:
        stage=row["stage_index"];fallback=HYBRID_FALLBACK_MAP.get(str(stage),[])
        routes={str(member):"uncached" if member in fallback else "cached" for member in range(8)}
        route=value.get("stage_routes",{}).get(str(stage),{})
        if row.get("passed") is not True or route.get("fallback_members")!=fallback or route.get("member_routes")!=routes or row.get("assembled_route")!=("hybrid" if fallback else "cached"):raise ValueError("declared hybrid stage/member routes differ")
        comparisons=row.get("comparisons",[])
        if [c.get("query_rows") for c in comparisons]!=[1,16,128,1024]:raise ValueError("hybrid query coverage incomplete")
        for comparison in comparisons:
            prediction(comparison["cache_vs_same_chunk_control"]);prediction(comparison["control_chunk_vs_1024"])
        prediction(row["stress_comparison"])
        component=row.get("component_qualification",{})
        expected_component=hybrid_sha256 if fallback else mixture_sha256
        if component.get("passed") is not True or component.get("module_sha256")!=expected_component or component.get("fast_icdf_same_logits_failures")!=0 or component.get("executor_metadata",{}).get("full8_member_lists_restored") is not True:raise ValueError("hybrid component source/sampler/restoration mismatch")
        for name in ("normalized_mean_captured_raw_vs_full_fp32","normalized_mean_observed_raw_vs_full_fp32","mass_corrected_observed_outputs_vs_full_fp32"):probability(component[name])
        if component.get("roundoff",{}).get("mass_corrected_reconstruction_passed") is not True or component["roundoff"].get("component_mass_guard_passed") is not True:raise ValueError("hybrid raw-mass law proof failed")
        members=component.get("members",[])
        if sorted(m.get("ensemble_index",-1) for m in members)!=list(range(8)):raise ValueError("hybrid all8 component witnesses required")
        for member in members:
            probability(member);probability(member["normalized_component"])
            if not all(member.get(key) is True for key in ("configuration_identity_equal","exact_borders","float32","component_raw_mass_matches_fp64_sum","fast_icdf_same_logits_exact","executed_corrected_helper_fingerprint_matches")) or member.get("corrected_cdf_call_delta")!=1:raise ValueError("hybrid component source/math proof failed")
            if fallback and (member.get("execution_route")!=routes[str(member["ensemble_index"])] or member.get("route_counter_delta")!=1 or member.get("other_route_counter_delta")!=0):raise ValueError("hybrid actual member route mismatch")
        if stage==0:
            probability(component["stage0_shared_distribution"])
            if component.get("stage0_dedup_scope")!="full_mixture_only" or component["stage0_shared_distribution"].get("shared_icdf_same_logits_exact") is not True:raise ValueError("hybrid stage0 shared density unqualified")
    return True


def gqa_recovery_helper():
    import importlib.util
    global _GQA_RECOVERY_MODULE
    path=Path(__file__).resolve().parents[2]/"scripts/recover_gqa_hybrid_qualification.py"
    if sha256_file(path)!=GQA_RECOVERY_SHA256:raise ValueError("reviewed GQA recovery source changed")
    if _GQA_RECOVERY_MODULE is None:
        spec=importlib.util.spec_from_file_location("_reviewed_execution_gqa_recovery",path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);_GQA_RECOVERY_MODULE=module
    return _GQA_RECOVERY_MODULE


def validate_gqa_hybrid_qualification(value,*,bindings,cache_sha256,mixture_sha256,hybrid_sha256,query_batch_size,qualification_path):
    from . import execution_gqa_hybrid as execution
    from .compat_cdf import adapter_source_sha256
    metadata=execution.execution_metadata();helper=gqa_recovery_helper()
    if value.get("schema")==helper.SCHEMA:
        if qualification_path is None:raise ValueError("source-bound recovered proof file required")
        if helper.verify_recovery_artifact(qualification_path)!=value:raise ValueError("recovered proof differs from archived predicate replay")
    elif value.get("schema")!="posteriorforge.execution_gqa_hybrid_qualification.v1":raise ValueError("explicit GQA hybrid proof schema required")
    required={"status":"gqa_hybrid_screen_pass","qualified":True,"full21_gqa_hybrid_qualified":True,
        "full21_qualified":False,"cache_only_full21_qualified":False,"full21_component_qualified":True,"full21_sampler_qualified":True,
        "sources_unchanged":True,"qualifier_source_sha256":GQA_QUALIFIER_SHA256,"module_sha256":execution.execution_source_sha256(),
        "gqa_module_sha256":execution.PINNED_GQA_SHA256,"hybrid_module_sha256":hybrid_sha256,
        "cache_module_sha256":cache_sha256,"mixture_module_sha256":mixture_sha256,
        "thresholds":QUALIFICATION_THRESHOLDS,"row_counts":[1,16,128,1024],"icdf_contract_version":2,
        "fallback_map":{"10":[5]},"fallback_policy_sha256":metadata["fallback_policy_sha256"],"execution_engine":metadata,
        "posterior_vectors_generated":0,"scientific_gate_evaluated":False}
    for key,expected in required.items():
        if type(value.get(key)) is not type(expected) or value.get(key)!=expected:raise ValueError("GQA qualification mismatch: "+key)
    if query_batch_size!=1024 or value.get("stage_coverage")!=list(range(21)):raise ValueError("all21 original1024 GQA coverage required")
    for key,expected in bindings.items():
        if value.get(key)!=expected:raise ValueError("GQA qualification scientific binding changed: "+key)
    coverage=value.get("producer_coverage",{})
    if coverage.get("fresh_original_control")!={"rows":1024,"members":8,"member_rows":1024,"stress_rows":8,"calls_per_stage_main":1} or coverage.get("saved_golden",{}).get("member_rows")!=128:
        raise ValueError("fresh all8 member1024 and separate saved128 coverage required")
    adapter=value.get("compatibility_adapter",{})
    if adapter.get("adapter_sha256")!=adapter_source_sha256() or adapter.get("original_alias_restored") is not True or adapter.get("sdk_aliases_unchanged") is not True or adapter.get("executed_corrected_cdf_code_sha256")!=adapter.get("corrected_cdf_code_sha256") or adapter.get("corrected_cdf_code_sha256")==adapter.get("original_cdf_code_sha256"):
        raise ValueError("qualified actual corrected CDF source/restoration required")
    rows=value.get("stage_results",[])
    if [row.get("stage_index") for row in rows]!=list(range(21)):raise ValueError("all21 ordered GQA stage records required")
    for stage,row in enumerate(rows):
        fallback=[5] if stage==10 else [];routes={str(i):"uncached" if i in fallback else "cached" for i in range(8)}
        if row.get("passed") is not True or row.get("fallback_members")!=fallback or row.get("member_routes")!=routes:raise ValueError("GQA actual member routes changed")
        if not all(row.get(k) is True for k in ("normalization_train_only_verified","preprocessor_identity","golden_fit_context_equal","prepared_state_unchanged","rng_unchanged")):raise ValueError("GQA state/preprocessing/RNG proof incomplete")
        if not row.get("rng") or not all(v.get("before")==v.get("after") for v in row.get("rng",[])):raise ValueError("GQA measured RNG state changed")
        comparisons=row.get("comparisons",[])
        if [v.get("query_rows") for v in comparisons]!=[1,16,128,1024]:raise ValueError("GQA candidate chunk coverage incomplete")
        for comparison in comparisons:
            helper.prediction(comparison["candidate_vs_fresh_control1024_slice"]);helper.gqa_report(comparison["actual_gqa_execution"],stage,execution.PINNED_GQA_SHA256)
        for key in ("fresh_control_vs_saved_golden","stress_control_vs_saved_golden","stress_comparison"):helper.prediction(row[key])
        helper.gqa_report(row["stress_gqa_execution"],stage,execution.PINNED_GQA_SHA256)
        for key in ("component_qualification","stress_component_qualification"):
            helper.component(row[key],stage,wrapper_sha=execution.execution_source_sha256(),hybrid_sha=hybrid_sha256,mixture_sha=mixture_sha256)
        for outlier in row.get("outlier_checks",[]):
            if outlier.get("passed") is not True or outlier.get("target_excluded_from_query") is not True:raise ValueError("GQA outlier proof incomplete")
            helper.prediction(outlier["prediction_comparison"]);helper.gqa_report(outlier["actual_gqa_execution"],stage,execution.PINNED_GQA_SHA256)
            helper.component(outlier["component_qualification"],stage,wrapper_sha=execution.execution_source_sha256(),hybrid_sha=hybrid_sha256,mixture_sha=mixture_sha256,outlier=True)
        if stage==0:
            dedup=row.get("stage0_shared_distribution",{});helper.probability(dedup);helper.prediction(dedup["full1024_comparison"])
            if dedup.get("passed") is not True or dedup.get("shared_icdf_same_logits_exact") is not True:raise ValueError("GQA stage0 full-mixture dedup proof failed")
    return True


def valid_execution_chunk(mode,query_batch_size):
    return type(query_batch_size) is int and (query_batch_size==1024 or (mode=="component_gqa_hybrid" and query_batch_size in GQA_LARGE_BATCHES))


def execution_query_cap(mode,query_batch_size):
    return 10000 if mode=="component_gqa_hybrid" and query_batch_size in GQA_LARGE_BATCHES else query_batch_size


def gqa_batch_helper():
    import importlib.util
    global _GQA_BATCH_MODULE
    path=Path(__file__).resolve().parents[2]/"scripts/qualify_execution_gqa_batches.py"
    if sha256_file(path)!=GQA_BATCH_QUALIFIER_SHA256:raise ValueError("reviewed large-batch producer source changed")
    if _GQA_BATCH_MODULE is None:
        spec=importlib.util.spec_from_file_location("_reviewed_gqa_batch_qualification",path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);_GQA_BATCH_MODULE=module
    return _GQA_BATCH_MODULE


def validate_gqa_batch_qualification(value,*,bindings,cache_sha256,mixture_sha256,hybrid_sha256,query_batch_size,qualification_path,sdk_root=None):
    """Fresh same-process batch proof; portable source bytes, no historic ctime."""
    from . import execution_gqa_hybrid as execution
    from .compat_cdf import adapter_source_sha256
    from .science.analytical_benchmark_v2 import array_descriptor
    helper=gqa_batch_helper();math=gqa_recovery_helper();metadata=execution.execution_metadata()
    if type(query_batch_size) is not int or query_batch_size not in GQA_LARGE_BATCHES or qualification_path is None:raise ValueError("exact source-bound4096/6250/10000 batch proof required")
    required={"schema":"posteriorforge.execution_gqa_batch_qualification.v1","status":"batch_screen_pass","qualified":True,
        "qualifier_source_sha256":GQA_BATCH_QUALIFIER_SHA256,"full21_qualified":False,"cache_only_full21_qualified":False,
        "posterior_vectors_generated":0,"scientific_gate_evaluated":False,"sources_unchanged":True,
        "requested_batch_sizes":list(GQA_LARGE_BATCHES),"candidate_query_batch_cap":10000,"comparison_block_cap":1024,
        "thresholds":QUALIFICATION_THRESHOLDS,"module_sha256":execution.execution_source_sha256(),"gqa_module_sha256":execution.PINNED_GQA_SHA256,
        "hybrid_module_sha256":hybrid_sha256,"cache_module_sha256":cache_sha256,"mixture_module_sha256":mixture_sha256,
        "execution_engine":metadata,"fallback_map":{"10":[5]},"fallback_policy_sha256":metadata["fallback_policy_sha256"],
        "base_full21_gqa_hybrid_qualified":True,"base_qualification_schema":"posteriorforge.execution_gqa_hybrid_recovery.v1","recovered_producer_sha256":GQA_RECOVERY_SHA256,
        "reference_pool":{"main_rows":1024,"stress_rows":8,"pool_rows":1032,"main_member_count":8,"stress_member_count":8,"original_calls_per_stage":2}}
    for key,expected in required.items():
        if type(value.get(key)) is not type(expected) or value.get(key)!=expected:raise ValueError("large-batch qualification mismatch: "+key)
    for key,expected in bindings.items():
        if value.get(key)!=expected:raise ValueError("large-batch scientific binding mismatch: "+key)
    if value.get("stage_coverage")!=list(range(21)) or [r.get("stage_index") for r in value.get("stage_results",[])]!=list(range(21)):raise ValueError("all21 fresh batch stages required")
    outcomes=helper.full21_batch_outcomes(value["stage_results"],value["stage_coverage"])
    if value.get("full21_batch_qualified")!=outcomes or outcomes.get(str(query_batch_size)) is not True:raise ValueError("requested neural batch size is not full21 qualified")
    if value.get("qualified_batch_sizes")!=[n for n in GQA_LARGE_BATCHES if outcomes[str(n)]]:raise ValueError("per-size qualification summary differs")
    root=Path(qualification_path).resolve().parent;bundle=value.get("portable_source_authentication",{})
    if sdk_root is None:sdk_root=Path(import_module("tabpfn").__file__).parent
    helper.verify_portable_sources(root,bundle,sdk_root=sdk_root)
    if bundle.get("sdk_distribution")!="tabpfn==2.0.9" or bundle.get("all_wheel_sdk_python_sources_checked") is not True or bundle.get("verification_uses_file_ctime") is not False or bundle.get("verification_requires_original_machine_paths") is not False:raise ValueError("portable proof must not require historical install paths/ctime")
    base_entry=bundle["archives"]["base_qualification"]
    if base_entry["sha256"]!=value.get("base_qualification_manifest_sha256"):raise ValueError("archived base proof hash differs")
    base=json.loads((root/base_entry["file"]).read_text())
    if base.get("schema")!="posteriorforge.execution_gqa_hybrid_recovery.v1" or base.get("status")!="gqa_hybrid_screen_pass" or base.get("recovery_source_sha256")!=GQA_RECOVERY_SHA256 or base.get("raw_status_preserved") is not True or base.get("raw_producer_status")!="conditional_mismatch" or base.get("full21_gqa_hybrid_qualified") is not True:
        raise ValueError("archived historical base qualification lineage differs")
    if base.get("execution_engine")!=metadata or base.get("fallback_map")!={"10":[5]} or base.get("module_sha256")!=execution.execution_source_sha256() or base.get("gqa_module_sha256")!=execution.PINNED_GQA_SHA256 or base.get("hybrid_module_sha256")!=hybrid_sha256 or base.get("cache_module_sha256")!=cache_sha256 or base.get("mixture_module_sha256")!=mixture_sha256 or base.get("raw_qualifier_source_sha256")!=GQA_QUALIFIER_SHA256 or base.get("full21_qualified") is not False or base.get("cache_only_full21_qualified") is not False:
        raise ValueError("archived base source/fallback graph changed")
    for key in ("checkpoint_sha256","scientific_fingerprint","parameter_order","context_theta","context_obs","query","prefix_manifest_sha256","prefix_samples_sha256"):
        if value.get(key)!=base.get(key):raise ValueError("fresh batch/historical base binding differs: "+key)
    adapter=value.get("compatibility_adapter",{})
    if adapter.get("adapter_sha256")!=adapter_source_sha256() or not all(adapter.get(k) is True for k in ("original_alias_restored","sdk_sources_unchanged","sdk_aliases_unchanged","source_guard_passed","original_alias_identity_verified","translator_code_unchanged")) or adapter.get("patched_call_count",0)<=0 or adapter.get("executed_corrected_cdf_code_sha256")!=adapter.get("corrected_cdf_code_sha256") or adapter.get("corrected_cdf_code_sha256")==adapter.get("original_cdf_code_sha256"):
        raise ValueError("large-batch actual corrected CDF source/restoration failed")
    expected_graph={"wrapper":execution.execution_source_sha256(),"gqa":execution.PINNED_GQA_SHA256,"hybrid":hybrid_sha256,"cache":cache_sha256,"mixture":mixture_sha256,"recovery":GQA_RECOVERY_SHA256}
    if value.get("source_graph_at_end")!=expected_graph:raise ValueError("large-batch source graph changed")
    def direct(item,rows):
        math.probability(item)
        if item.get("passed") is not True or item.get("rows_compared")!=rows or item.get("comparison_block_cap")!=1024 or item.get("blocks")!=(rows+1023)//1024 or type(item.get("worst_cdf_row")) is not int or not 0<=item["worst_cdf_row"]<rows:
            raise ValueError("complete direct all-row probability comparison required")
    for row in value["stage_results"]:
        stage=row["stage_index"]
        if row.get("reference_main_rows")!=1024 or row.get("reference_stress_rows")!=8 or row.get("reference_pool_query",{}).get("shape")!=[1032,33+stage]:raise ValueError("fresh main1024 plusstress8 pool required")
        if not all(row.get(k) is True for k in ("prepare_rng_unchanged","normalization_train_only_verified","all8_preprocessor_identity","final_prepared_state_unchanged")):raise ValueError("fresh batch state/normalization/RNG failed")
        contracts=row.get("model_contracts",[])
        if len(contracts)!=(3 if stage==10 else 2) or not all(c.get("input_normalization_train_only_verified") is True and c.get("autocast") is False and c.get("forced_inference_dtype")=="torch.float32" for c in contracts):raise ValueError("actual FP32 model normalizer contracts required")
        if not all(c.get("preprocessor_sha256")==contracts[0].get("preprocessor_sha256") and len(c.get("preprocessor_sha256",[]))==8 for c in contracts):raise ValueError("all8 fresh fitted preprocessors differ")
        if [b.get("query_rows") for b in row.get("batch_results",[])]!=list(GQA_LARGE_BATCHES):raise ValueError("all requested batch outcomes must be preserved")
        for batch in row["batch_results"]:
            n=batch["query_rows"]
            if not outcomes[str(n)]:continue  # Preserve failing/unsupported sizes; never select them.
            if batch.get("status")!="batch_pass" or batch.get("passed") is not True or batch.get("actual_input_query_shape")!=[n,33+stage] or batch.get("explicit_query_batch_cap")!=10000:raise ValueError("actual qualified batch shape/cap differs")
            indices=helper.repeated_pool_indices(n)
            if batch.get("reference_row_indices")!=array_descriptor(indices):raise ValueError("repeated-pool row identity differs")
            if not all(batch.get(k) is True for k in ("rng_unchanged","prepared_state_unchanged","normalized_mixture_comparison_is_direct_not_error_bound")):raise ValueError("batch state/RNG/direct mixture proof failed")
            direct(batch["ideal_raw_mixture_vs_original"],n);direct(batch["actual_mass_weighted_component_mixture_vs_original"],n)
            if [m.get("ensemble_index") for m in batch.get("members",[])]!=list(range(8)):raise ValueError("all8 full actual neural batches required")
            for member in batch["members"]:
                index=member["ensemble_index"];fallback=stage==10 and index==5
                direct(member["direct_raw_probability_comparison"],n)
                if not all(member.get(k) is True for k in ("passed","configuration_identity_equal","exact_borders","float32","component_raw_mass_matches_fp64_sum","normalized_component_passed")):raise ValueError("actual member distribution/state failed")
                if member.get("reference_configuration_identity")!=member.get("candidate_configuration_identity"):raise ValueError("fresh same-process original/candidate identity differs")
                if member.get("actual_execution_route")!=("original_fit_preprocessors" if fallback else "resident_cache") or member.get("corrected_cdf_call_delta")!=1 or member.get("rng_before")!=member.get("rng_after"):raise ValueError("actual member route/CDF/RNG differs")
                math.context(member["actual_gqa_descriptor"],expanded=not fallback,gqa_sha=execution.PINNED_GQA_SHA256)
                if member.get("actual_neural_shape")!=helper.shape_evidence(member["actual_gqa_descriptor"],n) or member["actual_neural_shape"].get("passed") is not True:raise ValueError("observed neural query length differs")
                error,bound=member.get("component_raw_mass_sum_error"),member.get("component_raw_mass_fp64_roundoff_bound")
                if type(error) not in (int,float) or type(bound) not in (int,float) or not np.isfinite(error) or not np.isfinite(bound) or not 0<=error<=bound<=8*5000*np.finfo(np.float64).eps or not 0<member.get("mass_min",0)<=member.get("mass_max",0)<=1.00001 or member.get("head_mass_envelope")!=1.00001:raise ValueError("actual FP64 mass/envelope bound failed")
                replay=member.get("stock_fixed_uniform_replay",{})
                if replay.get("passed") is not True or replay.get("exact_same_logits_finite_bar_semantics") is not True or replay.get("row_indices")!=np.linspace(0,n-1,8,dtype=np.int64).tolist():raise ValueError("actual stock finite-bar common-uniform replay failed")
    return True


def validate_gqa_prediction(prediction,stage,member,*,cdf_descriptor,report):
    from . import execution_gqa_hybrid as execution
    helper=gqa_recovery_helper();descriptor=prediction.get("gqa_attention_descriptor",{})
    fallback=stage==10 and member==5
    helper.context(descriptor,expanded=not fallback,gqa_sha=execution.PINNED_GQA_SHA256)
    if member is not None and prediction.get("hybrid_execution_path")!=("original_fit_preprocessors" if fallback else "resident_cache"):
        raise RuntimeError("actual GQA member route differs from qualified policy")
    if cdf_descriptor.get("executed_corrected_cdf_code_sha256")!=cdf_descriptor.get("corrected_cdf_code_sha256") or cdf_descriptor.get("corrected_cdf_code_sha256")==cdf_descriptor.get("original_cdf_code_sha256"):
        raise RuntimeError("actual GQA call did not execute private corrected CDF")
    fingerprint={k:descriptor[k] for k in ("original_attention_code_sha256","active_attention_code_sha256","executed_original_attention_code_sha256")}
    if report.get("attention_fingerprints") not in (None,fingerprint):raise RuntimeError("actual GQA attention code fingerprint changed")
    report["attention_fingerprints"]=fingerprint;report["verified_calls"]=report.get("verified_calls",0)+1
    report["uncached_fallback_calls"]=report.get("uncached_fallback_calls",0)+int(fallback)
    report["all_aliases_restored"]=True;report["all_sources_unchanged"]=True


def validate_qualification(value,*,bindings,cache_sha256,mixture_sha256,mode,query_batch_size,hybrid_sha256=None,qualification_path=None,sdk_root=None):
    if mode=="component_gqa_hybrid":
        if query_batch_size in GQA_LARGE_BATCHES:return validate_gqa_batch_qualification(value,bindings=bindings,cache_sha256=cache_sha256,mixture_sha256=mixture_sha256,hybrid_sha256=hybrid_sha256,query_batch_size=query_batch_size,qualification_path=qualification_path,sdk_root=sdk_root)
        return validate_gqa_hybrid_qualification(value,bindings=bindings,cache_sha256=cache_sha256,mixture_sha256=mixture_sha256,hybrid_sha256=hybrid_sha256,query_batch_size=query_batch_size,qualification_path=qualification_path)
    if mode=="component_hybrid":return validate_hybrid_qualification(value,bindings=bindings,cache_sha256=cache_sha256,mixture_sha256=mixture_sha256,hybrid_sha256=hybrid_sha256,query_batch_size=query_batch_size)
    if (value.get("schema")!="posteriorforge.execution_cache_qualification.v1" or value.get("status")!="screen_pass"
            or value.get("qualified") is not True or value.get("full21_qualified") is not True
            or value.get("icdf_contract_version")!=2 or value.get("sources_unchanged") is not True):
        raise ValueError("complete full21 contract2 cache qualification required")
    if value.get("sources",{}).get("qualification_script_sha256")!=QUALIFIER_SHA256:raise ValueError("reviewed qualification producer source required")
    if value.get("engine_id")!="resident_stage_kv_fp32_v1":raise ValueError("qualified engine identity mismatch")
    if value.get("control_variant")!="cdf_arithmetic_v1" or value.get("module_sha256")!=cache_sha256:
        raise ValueError("qualified corrected control/cache source mismatch")
    if value.get("thresholds")!=QUALIFICATION_THRESHOLDS or query_batch_size not in value.get("row_counts",[]):
        raise ValueError("unqualified query batch size or numerical thresholds")
    if sorted(value.get("stage_coverage",[]))!=list(range(21)):
        raise ValueError("all21 unique stages must be qualified")
    for key,expected in bindings.items():
        if value.get(key)!=expected:raise ValueError("qualification binding mismatch: "+key)
    adapter=value.get("compatibility_adapter",{})
    if not (adapter.get("original_alias_restored") is True and adapter.get("sdk_sources_unchanged") is True and adapter.get("sdk_aliases_unchanged") is True and adapter.get("patched_call_count",0)>0 and adapter.get("executed_corrected_cdf_code_sha256")==adapter.get("corrected_cdf_code_sha256") and adapter.get("corrected_cdf_code_sha256")!=adapter.get("original_cdf_code_sha256")):
        raise ValueError("qualified corrected CDF execution and restoration required")
    rows=value.get("stage_results",[])
    if sorted(row.get("stage_index",-1) for row in rows)!=list(range(21)) or not all(row.get("passed") is True for row in rows):
        raise ValueError("all21 stage result records must pass")
    for row in rows:
        if sorted(c.get("query_rows",-1) for c in row.get("comparisons",[]))!=[1,16,128,1024]:raise ValueError("all declared query chunks must be qualified")
        for comparison in row.get("comparisons",[]):
            for key in ("cache_vs_same_chunk_control","control_chunk_vs_1024"):
                item=comparison[key]
                if item.get("passed") is not True or item.get("inverse_cdf",{}).get("icdf_contract_version")!=2:
                    raise ValueError("pointwise and actual original sampler proof required")
        if row.get("stress_comparison",{}).get("passed") is not True or row["stress_comparison"].get("inverse_cdf",{}).get("icdf_contract_version")!=2:
            raise ValueError("stress inputs must retain qualified arithmetic")
    if mode=="component_mass":
        if value.get("mixture_module_sha256")!=mixture_sha256 or value.get("full21_component_qualified") is not True or value.get("full21_sampler_qualified") is not True:
            raise ValueError("full21 component and fast-sampler source qualification required")
        for row in rows:
            component=row.get("component_qualification",{})
            if component.get("passed") is not True or component.get("fast_icdf_same_logits_failures")!=0 or component.get("executor_metadata",{}).get("full8_member_lists_restored") is not True:
                raise ValueError("component capture/restoration/sampler qualification required")
            for name in ("normalized_mean_captured_raw_vs_full_fp32","normalized_mean_observed_raw_vs_full_fp32","mass_corrected_observed_outputs_vs_full_fp32"):
                if component.get(name,{}).get("passed") is not True:raise ValueError("actual mixture reconstruction must pass")
            members=component.get("members",[])
            if len(members)!=8 or not all(m.get("passed") is True and m.get("corrected_cdf_call_delta")==1 and m.get("executed_corrected_helper_fingerprint_matches") is True for m in members):
                raise ValueError("every component must execute the corrected translator")
            if row["stage_index"]==0 and (component.get("stage0_dedup_scope")!="full_mixture_only" or component.get("stage0_shared_distribution",{}).get("passed") is not True):
                raise ValueError("stage0 shared full-mixture distribution is unqualified")
    return True


def compact_cache(value):
    result={key:item for key,item in value.items() if key!="tensor_records"}
    by_category={};by_device={};parameters={};parameter_dtypes=set();kv=0;plain=0
    for row in value.get("tensor_records",[]):
        size=row["logical_bytes"];by_category[row["category"]]=by_category.get(row["category"],0)+size
        by_device[row["device"]]=by_device.get(row["device"],0)+size
        if any(token in row["name"] for token in ("_k_cache","_v_cache","_kv_cache")):kv+=size
        if row["category"]=="plain_tensor":plain+=size
        if row["category"]=="parameter":
            member=str(row["ensemble_index"]);parameters[member]=parameters.get(member,0)+int(np.prod(row["shape"]));parameter_dtypes.add(row["dtype"])
    result.update(logical_bytes_by_category=by_category,logical_bytes_by_device=by_device,
                  logical_kv_bytes=kv,logical_plain_tensor_bytes=plain,parameter_count_by_member=parameters,parameter_dtypes=sorted(parameter_dtypes),
                  logical_byte_counts_can_include_shared_views=True)
    return result


def stage_major_wave(engine,theta_context,obs_context,observation,proposed_count,query_batch_size,draw,*,stage_draw=None,progress=None):
    """Internal variable-sized proposal wave, always all21 targets, no filtering."""
    if type(proposed_count) is not int or proposed_count<1 or query_batch_size<1:raise ValueError("positive wave/chunk sizes required")
    observation=np.asarray(observation,dtype=np.float32).reshape(1,33)
    values=np.empty((proposed_count,21),dtype=np.float32)
    for stage in range(21):
        start=time.perf_counter();prepared=engine.prepare(theta_context,obs_context,stage)
        prepared_at=time.perf_counter();before=compact_cache(engine.inspect_cache());chunks=0
        def query_for(rows):
            rows=np.asarray(rows,dtype=np.int64)
            return np.concatenate((np.repeat(observation,len(rows),axis=0),values[rows,:stage]),axis=1)
        try:
            if stage_draw is None:
                for first in range(0,proposed_count,query_batch_size):
                    rows=np.arange(first,min(first+query_batch_size,proposed_count))
                    prediction=engine.predict(query_for(rows),output_type="full",capture_members=False)
                    sample=np.asarray(draw(prediction),dtype=np.float32)
                    if sample.shape!=(len(rows),):raise RuntimeError("sampler returned an incomplete stage")
                    values[rows,stage]=sample;chunks+=1
                extra={}
            else:
                sample,extra=stage_draw(engine,stage,query_for,proposed_count,query_batch_size)
                sample=np.asarray(sample,dtype=np.float32)
                if sample.shape!=(proposed_count,):raise RuntimeError("stage sampler returned incomplete rows")
                values[:,stage]=sample;chunks=extra.get("model_query_chunks",0)
            after=compact_cache(engine.inspect_cache())
            event={"stage_index":stage,"proposed_rows":proposed_count,"query_batch_size":query_batch_size,"model_query_chunks":chunks,
                   "prepare_seconds":prepared_at-start,"predict_and_sample_seconds":time.perf_counter()-prepared_at,
                   "cache_before":before,"cache_after":after,"sampler":extra,
                   "fit_stock_cache_seconds":prepared.get("fit_stock_cache_seconds"),"resident_move_seconds":prepared.get("resident_move_seconds"),
                   "hybrid_fallback_members":prepared.get("hybrid_fallback_members",[]),"uncached_fit_seconds":prepared.get("uncached_fit_seconds",0.)}
            regressor=getattr(engine,"regressor",None)
            event["precision"]={"autocast":bool(regressor.use_autocast_),"forced_inference_dtype":str(regressor.forced_inference_dtype_)} if regressor is not None else {}
            if progress:progress(event)
        finally:engine.close()
    return values


def component_stage_sampler(component,torch,generators,counters,*,prediction_check=None):
    from .execution_mixture import finite_bar_icdf,head_acceptance
    def sample_stage(engine,stage,query_for,count,chunk_size):
        chunks=0;head_candidates=0;head_rejected=0;draws=np.empty(count,dtype=np.float32)
        neural_sizes={str(i):{} for i in range(8)};assignment_counts=[0]*8;rejection_counts=[0]*8
        def note_size(member,rows):
            key=str(rows);neural_sizes[str(member)][key]=neural_sizes[str(member)].get(key,0)+1
        stage_uniforms=torch.rand(count,generator=generators["component_uniform"],dtype=torch.float32,device="cpu")
        if stage==0:
            prediction=engine.predict(query_for([0]),output_type="full",capture_members=False)
            if prediction_check:prediction_check(prediction,stage,None)
            draws[:]=finite_bar_icdf(prediction["logits"],prediction["criterion"].borders,stage_uniforms).numpy()
            for member in range(8):note_size(member,1)
            counters["stage0_shared_predictions"]+=1
            return draws,{"model_query_chunks":1,"stage0_full_mixture_shared":True,"density_scope":"proposal-wave-local; no cross-request/case reuse","component_distribution_dedup":False,"uniform_draws":count,"neural_query_batch_sizes_by_member":neural_sizes,"head_assignment_counts_by_member":assignment_counts,"head_rejection_counts_by_member":rejection_counts}
        pending=np.arange(count)
        while len(pending):
            assignments=torch.randint(0,8,(len(pending),),generator=generators["head_proposal"],device="cpu").numpy()
            acceptance=torch.rand(len(pending),generator=generators["head_acceptance"],dtype=torch.float64,device="cpu").numpy()
            retry=[];head_candidates+=len(pending)
            for member in range(8):
                locations=np.flatnonzero(assignments==member);assignment_counts[member]+=len(locations)
                for first in range(0,len(locations),chunk_size):
                    selected=locations[first:first+chunk_size];rows=pending[selected]
                    prediction=component.predict_component(member,query_for(rows),capture_member=False);note_size(member,len(rows))
                    if prediction_check:prediction_check(prediction,stage,member)
                    mass=prediction["component_raw_mass"].numpy();accepted=head_acceptance(mass,acceptance[selected])
                    chunks+=1;counters["component_predictions"]+=1
                    if np.any(accepted):
                        uniforms=stage_uniforms[torch.from_numpy(rows[accepted])]
                        values=finite_bar_icdf(prediction["logits"][torch.from_numpy(accepted)],prediction["criterion"].borders,uniforms)
                        draws[rows[accepted]]=values.numpy()
                    if np.any(~accepted):retry.append(rows[~accepted]);head_rejected+=int((~accepted).sum());rejection_counts[member]+=int((~accepted).sum())
            pending=np.concatenate(retry) if retry else np.empty(0,dtype=np.int64)
        counters["head_candidates"]+=head_candidates;counters["head_rejections"]+=head_rejected
        return draws,{"model_query_chunks":chunks,"head_candidates":head_candidates,"head_rejections":head_rejected,
                      "head_policy":"raw_mass_rejection_v1","uniform_draws":count,"neural_query_batch_sizes_by_member":neural_sizes,"head_assignment_counts_by_member":assignment_counts,"head_rejection_counts_by_member":rejection_counts,"partial_vector_filtering":False,"uniform_assignment":"preassigned per stage by physical row; retained across head retries"}
    return sample_stage


def run_execution(*,banks,output,checkpoint,ledger,reference,reference_grid,qualification,context_seed=20261111,case=0,repetition=0,query_batch_size=1024,mode="resident_full",worker_contract=None,proposal_lookahead=0):
    output=Path(output)
    if output.exists():raise FileExistsError("fresh execution output required")
    if mode not in ("resident_full","component_mass","component_hybrid","component_gqa_hybrid") or not valid_execution_chunk(mode,query_batch_size) or type(repetition) is not int or repetition<0:
        raise ValueError("explicit qualified mode/query chunk and nonnegative repetition required")
    validate_proposal_lookahead(mode,proposal_lookahead)
    if proposal_lookahead:guarded_proposal_buffer()
    from .reference_grid import _require_budgeted_gpu
    from .inference_cdf import guard_baseline_source,reference_binding
    from .inference import CHECKPOINT_SHA256,frozen_regressor_settings,support_mask,_prior
    from .banks import load_context,load_heldout
    from .science.analytical_benchmark_v2 import array_descriptor,build_problem
    start=time.perf_counter();runner_sha=sha256_file(__file__);source=guard_baseline_source();budget=_require_budgeted_gpu(ledger)
    contract=json.loads(Path(worker_contract).read_text()) if worker_contract else None
    partition=worker_partition(contract,output=output,job_id=os.getenv("SLURM_JOB_ID"),context_seed=context_seed,case=case,repetition=repetition,runner_sha256=runner_sha)
    if contract and contract.get("proposal_lookahead",0)!=proposal_lookahead:raise ValueError("worker FIFO lookahead differs from aggregate contract")
    if budget["gpus"]!=partition["world_size"]:raise RuntimeError("reserved GPU count must equal declared worker world")
    if contract:
        if contract.get("budget_reservation_id")!=budget["reservation_id"]:raise ValueError("worker reservation mismatch")
        orchestrator=Path(__file__).resolve().parents[2]/"scripts/run_execution_workers.py"
        if not orchestrator.is_file() or sha256_file(orchestrator)!=contract["orchestrator_source_sha256"]:raise ValueError("orchestrator source mismatch")
    output.mkdir(parents=True,exist_ok=False)
    status={**runtime_identity(),"schema":SCHEMA,"variant":execution_variant(mode,proposal_lookahead),
        "runner_version":RUNNER_VERSION,"proposal_lookahead":proposal_lookahead,"proposal_buffer_policy":proposal_buffer_policy(proposal_lookahead),"proposal_buffer":None,"status":"running","attribution":"Built with PriorLabs-TabPFN","performance_first_authorized":True,
        "absolute_accuracy_gate_required_for_performance":False,"known_physical_control_status":"model_failure",
        "count_contract":"50000 total complete physical vectors per E2E benchmark","benchmark_total_count":COUNT,
        **partition,"scientific_run":partition["requested_count"]==COUNT and partition["end_to_end"],"sampling_rng_policy":"independent_cpu_sampling_streams_model_rng_isolated","context_seed":context_seed,"case":case,"repetition":repetition,"context_size":4096,
        "query_batch_size":query_batch_size,"engine_query_batch_cap":execution_query_cap(mode,query_batch_size),"microbatch":COUNT,"max_iter_rejection":None,"log_density":"not_used",
        "context_filter":"no_filtering","observation_preprocessing":"raw; TabPFN frozen preprocessing",
        "budget_binding":budget,"baseline_execution_source":source,"runner_source_sha256":runner_sha,
        "worker_contract_sha256":sha256_file(worker_contract) if worker_contract else None,
        "completed_stage_indices":[],"stage_events":[],"proposal_wave_sizes":[],"partial_vector_filtering":False,"original_gate_unchanged":True,
        "clipping_or_repair_performed":False,"mode":mode,"execution_optimized":True,"tail_family_changed":False}
    write_json(output/"manifest.json",status);engine=None;descriptor=None;proposal_adapter=None
    try:
        import torch
        from . import execution_cache as cache
        from . import execution_mixture as mixture
        from . import execution_hybrid as hybrid
        from . import execution_gqa_hybrid as gqa_hybrid
        from .compat_cdf import finite_cdf_compatibility
        from .upstream import load_baseline
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError("one visible CUDA GPU required per process")
        if torch.get_default_dtype()!=torch.float32 or torch.get_float32_matmul_precision()!="highest" or torch.backends.cuda.matmul.allow_tf32:
            raise RuntimeError("FP32 default/highest/TF32False contract required")
        if sha256_file(checkpoint)!=CHECKPOINT_SHA256:raise ValueError("immutable checkpoint mismatch")
        context=load_context(banks,context_seed,4096);observed=load_heldout(banks,case);query=np.asarray(observed.obs).reshape(1,33)
        problem=build_problem();bindings={"checkpoint_sha256":CHECKPOINT_SHA256,"scientific_fingerprint":problem.scientific_fingerprint,
            "parameter_order":list(problem.parameter_order),"context_theta":array_descriptor(context.theta),"context_obs":array_descriptor(context.obs),"query":array_descriptor(query)}
        qpath=Path(qualification);qsha=sha256_file(qpath);q=json.loads(qpath.read_text())
        validate_qualification(q,bindings=bindings,cache_sha256=cache.execution_source_sha256(),mixture_sha256=mixture.execution_source_sha256(),mode=mode,query_batch_size=query_batch_size,hybrid_sha256=hybrid.execution_source_sha256(),qualification_path=qpath)
        status.update(**bindings,qualification_manifest_sha256=qsha,qualification_scope=q["scope"],qualification_full21=True,
            reference_binding=reference_binding(banks=banks,reference=reference,reference_grid=reference_grid,case=case),
            execution_engine={**cache.execution_metadata(),"sampling_rng_policy":"independent_cpu_sampling_streams_model_rng_isolated","query_batch_size":query_batch_size,"posterior_count":partition["requested_count"]},
            regressor_settings={**frozen_regressor_settings(checkpoint),"fit_mode":"fit_with_cache"},
            gpu=torch.cuda.get_device_name(),torch_cuda=torch.version.cuda,torch_float32_matmul_precision=torch.get_float32_matmul_precision(),cuda_matmul_allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32))
        if mode=="component_hybrid":
            status["hybrid_execution"]=hybrid.execution_metadata(HYBRID_FALLBACK_MAP)
            status["qualification_full21_hybrid"]=True;status["qualification_cache_only_full21"]=False
        if mode=="component_gqa_hybrid":
            status["gqa_hybrid_execution"]=gqa_hybrid.execution_metadata()
            status["qualification_full21_gqa_hybrid"]=True;status["qualification_cache_only_full21"]=False
            status["qualification_producer_schema"]=q["schema"];status["qualification_recovery_source_sha256"]=q.get("recovery_source_sha256")
            status["qualification_family"]="large_batch_v1" if query_batch_size in GQA_LARGE_BATCHES else "gqa_hybrid_1024_v2"
            status["qualification_large_batch_size"]=query_batch_size if query_batch_size in GQA_LARGE_BATCHES else None
            status["qualification_large_batch_producer_sha256"]=GQA_BATCH_QUALIFIER_SHA256 if query_batch_size in GQA_LARGE_BATCHES else None
            status["qualification_portable_source_verified"]=query_batch_size in GQA_LARGE_BATCHES
            status["historical_install_ctime_replayed"]=query_batch_size not in GQA_LARGE_BATCHES
            if np.__version__!="1.26.4":raise RuntimeError("qualified NumPy1.26.4 required for GQA hybrid")
        from . import inference as baseline_module
        if baseline_module.write_json is not write_json:raise RuntimeError("original metadata writer must remain unchanged")
        load_baseline();accept_reject=import_module("npe_pfn.accept_reject_sampler").accept_reject_sample
        # All imports and qualification finish before explicitly seeding the independent sampling streams.
        seeds=worker_stream_seeds(context_seed,case,repetition,partition["rank"],partition["world_size"])
        torch.manual_seed(seeds["component_uniform"]);torch.cuda.manual_seed_all(seeds["component_uniform"])
        generators={name:torch.Generator(device="cpu").manual_seed(seed) for name,seed in seeds.items()}
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS","16")))
        streams={name:{"role":STREAM_ROLES[name],"seed":seed,"dtype":STREAM_DTYPES[name],"device":"cpu","generator":"torch.Generator"} for name,seed in seeds.items()}
        if mode=="resident_full":streams["component_uniform"]["generator"]="torch.default_generator; original criterion.sample"
        status["sampling_policy"]={**(mixture.execution_metadata() if mode in ("component_mass","component_hybrid","component_gqa_hybrid") else {}),
            "head_policy":"raw_mass_rejection_v1" if mode in ("component_mass","component_hybrid","component_gqa_hybrid") else "not_used",
            "uniform_assignment":"preassigned_per_stage_by_physical_row" if mode in ("component_mass","component_hybrid","component_gqa_hybrid") else "original_chunk_sampler_global_cpu","head_retry_preserves_row_uniform":mode in ("component_mass","component_hybrid","component_gqa_hybrid"),
            "streams":streams,"model_rng_isolation":"CPU and selected CUDA fork_rng in frozen executor",
            "mass_envelope":1.00001 if mode in ("component_mass","component_hybrid","component_gqa_hybrid") else None,
            "stage0_dedup":{"enabled":mode in ("component_mass","component_hybrid","component_gqa_hybrid"),"scope":"full_mixture_only","component_distribution_dedup":False,"density_scope":"proposal-wave-local; no cross-request/case reuse"}}
        status["seed_streams"]=seeds;status["seed"]=seeds["component_uniform"]
        theta=context.theta.astype(np.float32);obs=context.obs.astype(np.float32);prior=_prior(torch)
        counters={"stage0_shared_predictions":0,"component_predictions":0,"head_candidates":0,"head_rejections":0}
        with finite_cdf_compatibility() as descriptor:
            if mode=="component_gqa_hybrid":
                engine=gqa_hybrid.GQAHybridStageExecutor(checkpoint,cdf_descriptor=descriptor,query_batch_size=execution_query_cap(mode,query_batch_size));component=engine
            elif mode=="component_hybrid":
                engine=hybrid.HybridStageExecutor(checkpoint,cdf_descriptor=descriptor,fallback_map=HYBRID_FALLBACK_MAP,query_batch_size=query_batch_size);component=engine
            else:
                engine=cache.ResidentStageExecutor(checkpoint,cdf_descriptor=descriptor,query_batch_size=query_batch_size)
                component=mixture.ComponentMixtureExecutor(engine) if mode=="component_mass" else None
            gqa_checks={}
            prediction_check=(lambda prediction,stage,member:validate_gqa_prediction(prediction,stage,member,cdf_descriptor=descriptor,report=gqa_checks)) if mode=="component_gqa_hybrid" else None
            stage_draw=component_stage_sampler(component,torch,generators,counters,prediction_check=prediction_check) if component else None
            def progress(event):
                status["stage_events"].append({"proposal_round":len(status["proposal_wave_sizes"]),**event})
                status["elapsed_seconds"]=time.perf_counter()-start;write_json(output/"manifest.json",status)
            def proposal(size):
                status["proposal_wave_sizes"].append(int(size))
                values=stage_major_wave(engine,theta,obs,query,int(size),query_batch_size,
                    lambda prediction:prediction["criterion"].sample(prediction["logits"]).detach().cpu().numpy(),stage_draw=stage_draw,progress=progress)
                return torch.from_numpy(values),None
            selected_proposal=proposal
            if proposal_lookahead:
                fifo=guarded_proposal_buffer()
                scope=proposal_buffer_scope(bindings,context_seed=context_seed,case=case,repetition=repetition,
                    rank=partition["rank"],world_size=partition["world_size"],seeds=seeds)
                proposal_adapter=fifo.ProposalBuffer(proposal,scope=scope,concatenate=lambda parts:torch.cat(parts,dim=0),
                    world_size=partition["world_size"],rank=partition["rank"])
                selected_proposal=proposal_adapter
                status["proposal_buffer"]={**proposal_adapter.metadata(),"enabled":True,"policy_sha256":status["proposal_buffer_policy"]["policy_sha256"]}
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();request_start=time.perf_counter()
            with torch.no_grad():samples,_,_=accept_reject(proposal=selected_proposal,accept_reject_fn=prior.support.check,
                num_samples=partition["requested_count"],max_sampling_batch_size=COUNT,max_iter_rejection=None,show_progress_bars=False,proposal_sampling_kwargs={})
            if proposal_adapter is not None:
                record=proposal_adapter.finalize(original_checked_count=prior.support.proposed,accepted_before_trim=prior.support.accepted,
                    returned_count=len(samples),original_support_calls=prior.support.calls)
                guarded_proposal_buffer();record.update(source_unchanged=True,enabled=True,policy_sha256=status["proposal_buffer_policy"]["policy_sha256"])
                validate_proposal_buffer_record(record,world_size=partition["world_size"],rank=partition["rank"],scope=scope)
                status["proposal_buffer"]=record
            torch.cuda.synchronize();finished=time.perf_counter()
            values=samples.detach().cpu().numpy().astype(np.float64)
            if values.shape!=(partition["requested_count"],21) or not support_mask(values).all():raise RuntimeError("incomplete/nonfinite/unsupported output; no row repair")
            status["execution_counters"]={key:value for key,value in engine.metadata.items() if key!="stages"}
            status["sampling_counters"]=dict(counters)
            if mode in ("component_hybrid","component_gqa_hybrid"):
                status["component_execution_counters"]=engine.cached_component.metadata
                status["hybrid_execution_counters"]={key:value for key,value in engine.hybrid_counters.items() if key!="stages"}
            elif component:status["component_execution_counters"]=component.metadata
            if mode=="component_gqa_hybrid":
                status["gqa_execution_counters"]=dict(engine.gqa_counters);status["gqa_prediction_checks"]=dict(gqa_checks)
                expected_contexts=counters["stage0_shared_predictions"]+counters["component_predictions"]
                if gqa_checks.get("verified_calls")!=expected_contexts or engine.gqa_counters["contexts"]!=expected_contexts or not engine.gqa_counters["all_contexts_restored"] or not engine.gqa_counters["all_sources_unchanged"] or engine.gqa_counters["expanded_attention_calls"]<=0:
                    raise RuntimeError("GQA actual per-call source/restoration coverage failed")
                if gqa_checks.get("uncached_fallback_calls")!=engine.hybrid_counters["uncached_predict_count"]:raise RuntimeError("GQA original fallback count mismatch")
            expected_calls=8*engine.metadata["predict_count"] if mode=="resident_full" else 8*counters["stage0_shared_predictions"]+counters["component_predictions"]
            generated_waves=len(status["proposal_wave_sizes"])
            if proposal_adapter is not None and status["proposal_buffer"]["actual_generated_wave_sizes"]!=status["proposal_wave_sizes"]:raise RuntimeError("FIFO/provider generated-wave mismatch")
            if descriptor["patched_call_count"]!=expected_calls or engine.metadata["prepare_count"]!=21*generated_waves:
                raise RuntimeError("model/CDF stage coverage mismatch")
            status["expected_patched_call_count"]=expected_calls
            status["expected_stage_prepare_count"]=21*generated_waves;status["generated_proposal_wave_count"]=generated_waves
            actual_cache=status["stage_events"][-1]["cache_after"];actual_precision=status["stage_events"][-1]["precision"]
            if len(actual_cache["parameter_count_by_member"])!=8 or set(actual_cache["parameter_count_by_member"].values())!={11081864} or actual_cache["parameter_dtypes"]!=["torch.float32"] or actual_precision!={"autocast":False,"forced_inference_dtype":"torch.float32"}:raise RuntimeError("actual frozen FP32 model/parameter contract mismatch")
            status["model"]={"dtype":"torch.float32","autocast":False,"forced_inference_dtype":"torch.float32","parameter_count":next(iter(status["stage_events"][-1]["cache_after"]["parameter_count_by_member"].values())),"parameter_count_by_member":status["stage_events"][-1]["cache_after"]["parameter_count_by_member"],"prepared_member_count":8,"post_request_residency":"stage caches released"}
        if not descriptor.get("original_alias_restored") or not descriptor.get("sdk_sources_unchanged"):raise RuntimeError("CDF context failed restoration")
        if baseline_module.write_json is not write_json:raise RuntimeError("original metadata writer changed")
        status["metadata_writer_restored"]=True;status["metadata_writer_policy"]="never replaced by execution driver"
        if proposal_lookahead:
            guarded_proposal_buffer()
            validate_proposal_buffer_record(status["proposal_buffer"],world_size=partition["world_size"],rank=partition["rank"],scope=scope)
        if guard_baseline_source()!=source or sha256_file(__file__)!=runner_sha or sha256_file(qpath)!=qsha:
            raise RuntimeError("source/qualification changed during execution")
        expected_cache_sha=q["cache_module_sha256"] if mode in ("component_hybrid","component_gqa_hybrid") else q["module_sha256"]
        if cache.execution_source_sha256()!=expected_cache_sha or (mode in ("component_mass","component_hybrid","component_gqa_hybrid") and mixture.execution_source_sha256()!=q["mixture_module_sha256"]) or (mode in ("component_hybrid","component_gqa_hybrid") and hybrid.execution_source_sha256()!=q["hybrid_module_sha256"]):
            raise RuntimeError("qualified executor source changed during execution")
        if mode=="component_gqa_hybrid":
            if query_batch_size in GQA_LARGE_BATCHES:
                helper=gqa_batch_helper();helper.verify_portable_sources(qpath.parent,q["portable_source_authentication"],sdk_root=Path(import_module("tabpfn").__file__).parent)
            gqa_hybrid.require_frozen_sources()
            if gqa_hybrid.execution_source_sha256()!=q["module_sha256"]:raise RuntimeError("qualified GQA wrapper changed during execution")
        with (output/"samples.npz").open("xb") as stream:np.savez(stream,samples=values)
        status.update(status="complete",samples=array_descriptor(values),samples_file="samples.npz",samples_sha256=sha256_file(output/"samples.npz"),
            completed_stage_indices=list(range(21)),compatibility_adapter=dict(descriptor),sources_unchanged=True,
            rejection={"support_calls":prior.support.calls,"proposed":prior.support.proposed,"accepted_before_final_trim":prior.support.accepted,"returned":len(values)},
            timings={"posterior_request_seconds":finished-request_start,"total_seconds":finished-start,"accepted_samples_per_second":len(values)/(finished-request_start),"includes_stage_preparation":True,"includes_CPU_sampling":True},
            memory={"peak_allocated_bytes":torch.cuda.max_memory_allocated(),"peak_reserved_bytes":torch.cuda.max_memory_reserved()})
    except BaseException as error:
        status.update(status="failed",error_type=type(error).__name__,error=str(error))
        if descriptor is not None:status["compatibility_adapter"]=dict(descriptor)
        if proposal_adapter is not None:status["proposal_buffer"]={**proposal_adapter.metadata(),"enabled":True,"policy_sha256":status["proposal_buffer_policy"]["policy_sha256"]}
        raise
    finally:
        if engine is not None:engine.close()
        status["wrapper_total_seconds"]=time.perf_counter()-start;write_json(output/"manifest.json",status)
    return status


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("banks","output","checkpoint","ledger","reference","reference-grid","qualification"):parser.add_argument("--"+name,required=True)
    parser.add_argument("--context-seed",type=int,default=20261111);parser.add_argument("--case",type=int,default=0)
    parser.add_argument("--repetition",type=int,default=0);parser.add_argument("--query-batch-size",type=int,default=1024)
    parser.add_argument("--proposal-lookahead",type=int,choices=(0,1200),default=0)
    parser.add_argument("--mode",choices=("resident_full","component_mass","component_hybrid","component_gqa_hybrid"),default="resident_full");parser.add_argument("--worker-contract")
    result=run_execution(**vars(parser.parse_args()));print(json.dumps({key:result.get(key) for key in ("schema","variant","status","requested_count","rank","world_size","end_to_end","timings")},indent=2))


if __name__=="__main__":main()
