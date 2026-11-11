#!/usr/bin/env python3
"""Assemble a NEW full21 hybrid conditional proof from immutable stage evidence.

Built with PriorLabs-TabPFN. Failed/partial cache inputs retain their statuses;
only individually passing cached stages are selected. Declared fallback stages
require fresh all-eight-member hybrid proofs. No GPU or model is invoked.
"""
from __future__ import annotations
import argparse
import hashlib
from copy import deepcopy
from importlib import import_module
import json
import math
from pathlib import Path

from posteriorforge.artifacts import sha256_file,write_json
from posteriorforge.compat_cdf import adapter_source_sha256,PINNED_SDK_SHA256
from posteriorforge.inference import CHECKPOINT_SHA256
from posteriorforge.science.analytical_benchmark_v2 import build_problem

SCHEMA="posteriorforge.execution_hybrid_qualification.v1"
CACHE_SCHEMA="posteriorforge.execution_cache_qualification.v1"
STAGE_SCHEMA="posteriorforge.execution_hybrid_stage_qualification.v1"
THRESHOLDS={"normalization_absolute":1e-5,"max_cdf_absolute":2e-5,
            "max_row_total_variation":1e-4,"quantile_coordinate_span_fraction":2e-6}
ROWS=[1,16,128,1024]
COMMON=("checkpoint_sha256","scientific_fingerprint","parameter_order","context_theta","context_obs",
        "query","prefix_manifest_sha256","prefix_samples_sha256","fixed_prefix_row_indices")
CACHE_PIN="45cd3107c50dcf73d03167e2b6dd045cccdb798337e8d3c47b5210cc30dc1a03"
MIXTURE_PIN="e71f43b73ae8259c20b5d7b52feccc7c851e9a2fbb86d869bc0a5f7f3af792a1"


def require(condition,message):
    if not condition:raise ValueError(message)


def current_sources():
    from posteriorforge import execution_cache,execution_mixture
    hybrid=import_module("posteriorforge.execution_hybrid")
    root=Path(__file__).resolve().parent
    result={"cache_module_sha256":execution_cache.execution_source_sha256(),
        "mixture_module_sha256":execution_mixture.execution_source_sha256(),
        "hybrid_module_sha256":sha256_file(hybrid.__file__),
        "cache_qualifier_sha256":sha256_file(root/"qualify_execution_cache.py"),
        "hybrid_qualifier_sha256":sha256_file(root/"qualify_execution_hybrid.py")}
    require(result["cache_module_sha256"]==CACHE_PIN and result["mixture_module_sha256"]==MIXTURE_PIN,
            "frozen parent executor source changed")
    return result


def normalize_fallback_map(value):
    require(isinstance(value,dict) and bool(value),"explicit nonempty declared fallback map required")
    result={}
    for key,members in value.items():
        stage=int(key)
        require(str(stage)==str(key) and 0<=stage<21,"invalid fallback stage")
        require(isinstance(members,list) and members and all(type(member) is int and 0<=member<8 for member in members),
                "fallback members must be nonempty integer indices0..7")
        require(len(set(members))==len(members),"duplicate fallback member")
        result[str(stage)]=sorted(members)
    return dict(sorted(result.items(),key=lambda item:int(item[0])))


def fallback_policy_sha256(value):
    canonical=normalize_fallback_map(value)
    return hashlib.sha256(json.dumps(canonical,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def validate_adapter(value):
    require(isinstance(value,dict),"missing corrected-CDF execution descriptor")
    require(value.get("adapter_sha256")==adapter_source_sha256(),"CDF adapter source mismatch")
    for key in ("source_guard_passed","original_alias_restored","sdk_sources_unchanged",
                "sdk_aliases_unchanged","original_alias_identity_verified","translator_code_unchanged"):
        require(value.get(key) is True,"CDF descriptor flag failed: "+key)
    require(value.get("patched_call_count",0)>0,"corrected CDF never executed")
    require(value.get("executed_corrected_cdf_code_sha256")==value.get("corrected_cdf_code_sha256")
            and value.get("corrected_cdf_code_sha256")!=value.get("original_cdf_code_sha256"),"wrong executed CDF fingerprint")
    require(value.get("active_translator_code_sha256")==value.get("original_translator_code_sha256"),"translator arithmetic changed")
    require(value.get("sdk_source_sha256")==dict(PINNED_SDK_SHA256)
            and value.get("sdk_source_sha256_after")==dict(PINNED_SDK_SHA256),"SDK source descriptor changed")


def probability_check(item):
    require(isinstance(item,dict) and item.get("passed") is True,"probability comparison did not pass")
    metrics=(("minimum_probability",0,math.inf),("max_normalization_error",0,THRESHOLDS["normalization_absolute"]),
             ("max_cdf_absolute_error",0,THRESHOLDS["max_cdf_absolute"]),
             ("max_row_total_variation",0,THRESHOLDS["max_row_total_variation"]))
    for key,low,high in metrics:
        value=item.get(key)
        require(type(value) in (int,float) and math.isfinite(value) and low<=value<=high,"probability threshold violated: "+key)


def prediction_check(item):
    require(item.get("passed") is True and item.get("eight_members") is True
            and item.get("exact_final_borders") is True and item.get("final_float32") is True,
            "full canonical prediction comparison incomplete")
    probability_check(item["final"])
    members=item.get("members",[])
    require(len(members)==8 and sorted(member.get("index",-1) for member in members)==list(range(8)),
            "all8 unique comparison members required")
    for member in members:
        probability_check(member)
        require(all(member.get(key) is True for key in ("configuration_identity_equal","exact_borders","float32")),
                "member configuration/borders/precision mismatch")
    inverse=item.get("inverse_cdf",{})
    require(inverse.get("icdf_contract_version")==2 and inverse.get("passed") is True
            and inverse.get("same_logits_and_borders_exact_failures")==0
            and inverse.get("independent_fp32_replay_failures")==0
            and not inverse.get("probability_space_failures"),"actual original ICDF-v2 proof missing")


def component_check(value,stage,sources,hybrid=False):
    mixture_sha256=sources["mixture_module_sha256"]
    component_source=sources["hybrid_module_sha256"] if hybrid else mixture_sha256
    require(value.get("schema")=="posteriorforge.execution_component_qualification.v1"
            and value.get("module_sha256")==component_source and value.get("passed") is True,"component proof source/status mismatch")
    if hybrid:
        require(value.get("component_mixture_module_sha256")==mixture_sha256 and value.get("sampler_module_sha256")==mixture_sha256,"hybrid component/sampler primitive source changed")
    require(value.get("fast_icdf_same_logits_failures")==0 and value.get("executor_metadata",{}).get("full8_member_lists_restored") is True,
            "component sampler/restoration failed")
    require(value.get("uniform_head_without_mass_correction_is_not_authorized") is True
            and value.get("component_distribution_dedup_qualified") is False,"unqualified component sampling shortcut")
    for key in ("normalized_mean_captured_raw_vs_full_fp32","normalized_mean_observed_raw_vs_full_fp32",
                "mass_corrected_observed_outputs_vs_full_fp32"):probability_check(value[key])
    roundoff=value.get("roundoff",{})
    require(roundoff.get("mass_corrected_reconstruction_passed") is True
            and roundoff.get("component_mass_guard_passed") is True,"raw-mass reconstruction/envelope failed")
    members=value.get("members",[])
    require(len(members)==8 and sorted(member.get("ensemble_index",-1) for member in members)==list(range(8)),
            "all8 independent component proofs required")
    for member in members:
        probability_check(member)
        require(all(member.get(key) is True for key in ("configuration_identity_equal","exact_borders","float32",
                "component_raw_mass_matches_fp64_sum","fast_icdf_same_logits_exact",
                "executed_corrected_helper_fingerprint_matches")),"component arithmetic/source proof incomplete")
        require(member.get("corrected_cdf_call_delta")==1,"component did not call the private corrected translator exactly once")
        require(0<member.get("mass_min",-1)<=member.get("mass_max",-1)<=1.00001,"component raw mass outside declared envelope")
        require(member.get("component_head_mass_envelope")==1.00001,"component head envelope changed")
        error=member.get("component_raw_mass_sum_error",math.inf);bound=member.get("component_raw_mass_fp64_roundoff_bound",-1)
        require(type(error) in (int,float) and type(bound) in (int,float) and math.isfinite(error) and math.isfinite(bound) and 0<=error<=bound,"FP64 raw mass disagreement")
        probability_check(member["normalized_component"])
    if stage==0:
        require(value.get("stage0_dedup_scope")=="full_mixture_only","stage0 deduplication scope changed")
        probability_check(value["stage0_shared_distribution"])
        require(value["stage0_shared_distribution"].get("shared_icdf_same_logits_exact") is True,"shared first-stage ICDF changed")


def stage_check(row,stage,sources,hybrid=False):
    require(row.get("stage_index")==stage and row.get("passed") is True,"selected stage did not pass")
    if hybrid:
        require(row.get("normalization_train_only_verified") is True,"hybrid train-only normalization declaration missing")
        contracts=("control_model_contract","resident_model_contract","uncached_model_contract")
    else:
        require(row.get("preprocessor_pickle_identity") is True,"cached preprocessor state identity failed")
        contracts=("baseline_model_contract_after","cache_model_contract_after")
    for key in contracts:
        require(row.get(key,{}).get("input_normalization_train_only_verified") is True,"actual train-only normalizer flag failed: "+key)
    comparisons=row.get("comparisons",[])
    require([item.get("query_rows") for item in comparisons]==ROWS,"fixed query rows incomplete")
    for item in comparisons:
        prediction_check(item["cache_vs_same_chunk_control"])
        prediction_check(item["control_chunk_vs_1024"])
    prediction_check(row["stress_comparison"])
    component_check(row["component_qualification"],stage,sources,hybrid)


def header_check(value,kind,common,sources):
    require(value.get("schema")==(CACHE_SCHEMA if kind=="cache" else STAGE_SCHEMA),"wrong proof schema")
    require(value.get("thresholds")==THRESHOLDS and value.get("row_counts")==ROWS
            and value.get("icdf_contract_version")==2,"numerical qualification contract changed")
    require(value.get("sources_unchanged") is True,"producer source-restoration proof missing")
    validate_adapter(value.get("compatibility_adapter"))
    for key in COMMON:
        require(key in value and value[key]==common[key],"proof scientific/context/prefix binding mismatch: "+key)
    if kind=="cache":
        require(value.get("status") in ("screen_pass","conditional_mismatch","incomplete","failed"),"cache proof still running or unknown")
        require(value.get("module_sha256")==sources["cache_module_sha256"]
                and value.get("mixture_module_sha256")==sources["mixture_module_sha256"],"cache/mixture module source changed")
        require(value.get("sources",{}).get("qualification_script_sha256")==sources["cache_qualifier_sha256"],"cache producer source changed")
    else:
        require(value.get("status")=="stage_pass" and value.get("passed") is True,"fresh hybrid stage proof did not pass")
        for key in ("cache_module_sha256","mixture_module_sha256","hybrid_module_sha256"):
            require(value.get(key)==sources[key],"hybrid source mismatch: "+key)
        require(value.get("qualifier_source_sha256")==sources["hybrid_qualifier_sha256"],"hybrid producer source mismatch")
        require(value.get("fullstage8_component_qualified") is True
                and value.get("fullstage_sampler_qualified") is True,"fresh hybrid all8/sampler proof incomplete")


def assemble(*,cache_proofs,hybrid_proofs,fallback_map,output):
    fallback=normalize_fallback_map(fallback_map);sources=current_sources()
    require(cache_proofs and hybrid_proofs,"both cached and fresh hybrid proof inputs required")
    loaded=[]
    for kind,paths in (("cache",cache_proofs),("hybrid",hybrid_proofs)):
        for path in paths:
            path=Path(path).resolve();raw=path.read_bytes();value=json.loads(raw)
            loaded.append({"kind":kind,"path":path,"raw":raw,"value":value,"sha256":sha256_file(path)})
    common={key:loaded[0]["value"][key] for key in COMMON}
    problem=build_problem()
    require(common["checkpoint_sha256"]==CHECKPOINT_SHA256
            and common["scientific_fingerprint"]==problem.scientific_fingerprint
            and common["parameter_order"]==list(problem.parameter_order),"declared scientific problem/checkpoint changed")
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    result={"schema":SCHEMA,"status":"assembling","qualified":False,"full21_qualified":False,
        "cache_only_full21_qualified":False,"full21_hybrid_qualified":False,
        "full21_component_qualified":False,"full21_sampler_qualified":False,
        "attribution":"Built with PriorLabs-TabPFN","posterior_vectors_generated":0,
        "scientific_gate_evaluated":False,"performance_first_authorized":True,
        "absolute_accuracy_gate_required_for_performance":False,"known_physical_control_status":"model_failure",
        "scope":"fixed_conditional_hybrid_screen_not_posterior_equivalence",
        "control_variant":"cdf_arithmetic_v1","icdf_contract_version":2,"row_counts":ROWS,
        "thresholds":dict(THRESHOLDS),"fallback_map":fallback,
        "fallback_policy_sha256":fallback_policy_sha256(fallback),
        "hybrid_input_policy_composition":"Stage-local composition: each selected input policy is hash-validated, but only that stage fallback_members and member_routes define its route; unrelated producer stage routes do not alter the final declared map.",
        "assembler_source_sha256":sha256_file(__file__),**sources,**common,"stage_results":[],"source_inputs":[]}
    write_json(root/"manifest.json",result)
    try:
        cache_candidates={};hybrid_candidates={}
        for index,item in enumerate(loaded):
            value=item["value"];kind=item["kind"];header_check(value,kind,common,sources)
            archive=root/"inputs"/f"{index:02d}-{kind}.json";archive.parent.mkdir(parents=True,exist_ok=True);archive.write_bytes(item["raw"])
            identity={"kind":kind,"path":str(item["path"]),"sha256":item["sha256"],
                "producer_status":value.get("status"),"producer_qualified":value.get("qualified"),
                "producer_full21_qualified":value.get("full21_qualified"),"archived_manifest":str(archive.relative_to(root))}
            result["source_inputs"].append(identity)
            if kind=="cache":
                for row in value.get("stage_results",[]):
                    stage=row.get("stage_index")
                    require(type(stage) is int and 0<=stage<21,"invalid cached stage record")
                    if str(stage) in fallback or row.get("passed") is not True:continue
                    stage_check(row,stage,sources)
                    cache_candidates.setdefault(stage,(deepcopy(row),identity))
            else:
                stage=value.get("stage_index")
                require(str(stage) in fallback,"hybrid stage not in declared fallback map")
                input_policy=normalize_fallback_map(value.get("fallback_map"))
                require(value.get("fallback_policy_sha256")==fallback_policy_sha256(input_policy),"hybrid input policy hash mismatch")
                members=value.get("fallback_members")
                require(input_policy.get(str(stage))==members,"hybrid local policy declaration mismatch")
                require(members==fallback[str(stage)],"hybrid fallback member declaration changed")
                stage_check(value,stage,sources,hybrid=True)
                routes={str(member):("uncached" if member in members else "cached") for member in range(8)}
                require(value.get("member_routes")==routes,"fresh hybrid member route map mismatch")
                for member in value["component_qualification"]["members"]:
                    require(member.get("execution_route")==routes[str(member["ensemble_index"])] and member.get("route_counter_delta")==1 and member.get("other_route_counter_delta")==0,"actual hybrid component route/counter mismatch")
                require(stage not in hybrid_candidates,"duplicate hybrid stage proof")
                hybrid_candidates[stage]=(deepcopy(value),identity)
        stage_routes={}
        for stage in range(21):
            is_hybrid=str(stage) in fallback
            candidates=hybrid_candidates if is_hybrid else cache_candidates
            require(stage in candidates,"missing passing "+("hybrid" if is_hybrid else "cached")+" stage"+str(stage))
            row,identity=candidates[stage]
            routes={str(member):("uncached" if member in fallback.get(str(stage),[]) else "cached") for member in range(8)}
            row["assembled_route"]="hybrid" if is_hybrid else "cached"
            row["selected_source"]={key:identity[key] for key in ("path","sha256","producer_status")}
            result["stage_results"].append(row)
            stage_routes[str(stage)]={"fallback_members":fallback.get(str(stage),[]),"member_routes":routes,
                 "source_sha256":identity["sha256"],"source_producer_status":identity["producer_status"]}
        require(all(sha256_file(item["path"])==item["sha256"] for item in loaded),"input proof changed during assembly")
        require(current_sources()==sources and sha256_file(__file__)==result["assembler_source_sha256"],"source changed during assembly")
        result.update(status="hybrid_screen_pass",qualified=True,full21_hybrid_qualified=True,
            full21_component_qualified=True,full21_sampler_qualified=True,
            stage_coverage=list(range(21)),component_stage_coverage=list(range(21)),stage_routes=stage_routes,
            sources_unchanged=True,input_statuses_preserved=True)
        write_json(root/"manifest.json",result)
        return result
    except BaseException as error:
        result.update(status="assembly_failed",error_type=type(error).__name__,error=str(error),
                      qualified=False,full21_hybrid_qualified=False,full21_component_qualified=False,full21_sampler_qualified=False)
        write_json(root/"manifest.json",result);raise


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-proof",action="append",required=True,dest="cache_proofs")
    parser.add_argument("--hybrid-proof",action="append",required=True,dest="hybrid_proofs")
    parser.add_argument("--fallback-map",required=True);parser.add_argument("--output",required=True)
    args=vars(parser.parse_args());args["fallback_map"]=json.loads(Path(args["fallback_map"]).read_text())
    result=assemble(**args)
    print(json.dumps({key:result[key] for key in ("schema","status","full21_qualified","full21_hybrid_qualified","stage_coverage")}))
