#!/usr/bin/env python3
"""CPU-only semantic configuration identity recovery; original proof immutable.

Built with PriorLabs-TabPFN. No model import/forward, GPU computation, numerical
threshold change or output replacement. Only recognized function addresses in
repr are process-local; exact configuration type and pickle SHA stay required.
"""
from __future__ import annotations
import argparse
import csv
import base64
import subprocess
import ast
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re

SCHEMA="posteriorforge.execution_gqa_hybrid_recovery.v1"
RAW_SCHEMA="posteriorforge.execution_gqa_hybrid_qualification.v1"
RAW_PRODUCER_SHA="ab0c289ad051aa3553ca47402da07ac2643c576e660a297be9242c357787848a"
FUNCTION_SOURCE_SHA="7da6b7c6d060a396b4aa7024fd5ac2c9bfddbc7d62029b113962cecbdaca6787"
FUNCTIONS=("_inf_to_nan_func","_identity")
ADDRESS=re.compile(r"<function (_inf_to_nan_func|_identity) at 0x[0-9a-fA-F]+>")
THRESHOLDS={"max_cdf_absolute":2e-5,"max_row_total_variation":1e-4,"normalization_absolute":1e-5,"quantile_coordinate_span_fraction":2e-6}
ROWS=[1,16,128,1024]


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def bytes_sha(value):return hashlib.sha256(value).hexdigest()


def semantic_identity(actual,golden):
    require(actual.get("type")==golden.get("type") and actual.get("sha256")==golden.get("pickle_sha256"),"configuration type/pickle hash differs")
    left,right=actual.get("repr"),golden.get("repr")
    require(isinstance(left,str) and isinstance(right,str),"configuration repr missing")
    normalize=lambda text:ADDRESS.sub(lambda m:"<function "+m.group(1)+" at <process-address>>",text)
    require(normalize(left)==normalize(right),"repr differs beyond recognized pinned function addresses")
    return {"type":actual["type"],"pickle_sha256":actual["sha256"],"repr_equal":left==right,
        "normalized_repr_sha256":bytes_sha(normalize(left).encode()),"actual_repr_sha256":bytes_sha(left.encode()),
        "golden_repr_sha256":bytes_sha(right.encode()),"recognized_functions":sorted(set(ADDRESS.findall(left))),
        "recognized_address_occurrences":len(ADDRESS.findall(left))}


def sdk_evidence(root,*,raw,golden):
    from posteriorforge import execution_cache as cache
    root=Path(root).resolve();sources={}
    for name,expected in cache.PINNED_SDK_SHA256.items():
        path=root/Path(name).relative_to("tabpfn");require(sha(path)==expected,"pinned SDK source changed: "+name);sources[name]=expected
    function_file=root/"model/preprocessing.py";require(sha(function_file)==FUNCTION_SOURCE_SHA,"function repr SDK source changed")
    tree=ast.parse(function_file.read_text());definitions={n.name:n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in FUNCTIONS}
    require(set(definitions)==set(FUNCTIONS),"exact pinned repr function definitions required")
    repo=Path(__file__).resolve().parents[1];auth_path="evidence/diagnostics/model-sdk-source-authenticity.json"
    auth_bytes=(repo/auth_path).read_bytes();auth=json.loads(auth_bytes);lock_bytes=(repo/"environments/inference.lock").read_bytes()
    history=[]
    for document in (golden,raw):
        revision=document.get("git_revision","");require(re.fullmatch(r"[0-9a-f]{40}",revision),"recorded source revision missing")
        require(subprocess.check_output(["git","show",revision+":"+auth_path],cwd=repo)==auth_bytes,"historically recorded wheel authentication changed")
        require(subprocess.check_output(["git","show",revision+":environments/inference.lock"],cwd=repo)==lock_bytes,"historical inference lock changed")
        require(document.get("versions",{}).get("numpy")=="1.26.4" and document["versions"].get("tabpfn")=="2.0.9","recorded callable dependency versions changed")
        history.append({"git_revision":revision,"created_unix":document["created_unix"],"wheel_authentication_bytes_equal":True,"inference_lock_bytes_equal":True})
    cached=Path(auth["cached_extracted_wheel_root"]);record=cached/"tabpfn-2.0.9.dist-info/RECORD"
    require(sha(record)==auth["cached_record_sha256"],"historically recorded complete wheel RECORD changed")
    cached_function=cached/"tabpfn/model/preprocessing.py"
    require(cached_function.read_bytes()==function_file.read_bytes(),"installed function source differs from authenticated cached wheel")
    expected="sha256="+base64.urlsafe_b64encode(hashlib.sha256(function_file.read_bytes()).digest()).decode().rstrip("=")
    require(any(row[0]=="tabpfn/model/preprocessing.py" and row[1]==expected for row in csv.reader(record.open())),"callable file not bound by historical wheel RECORD")
    stat=function_file.stat();first=min(v["created_unix"] for v in history)
    require(stat.st_ctime<first and record.stat().st_ctime<first,"SDK source/RECORD was written after historical GPU evidence began")
    callable_audit={}
    for name,node in definitions.items():
        require(not node.decorator_list and not node.args.defaults and all(v is None for v in node.args.kw_defaults),"callable decorators/defaults changed")
        require(not any(isinstance(v,(ast.FunctionDef,ast.Lambda,ast.Nonlocal)) for v in ast.walk(ast.Module(body=node.body,type_ignores=[]))),"callable closure/binding changed")
        callable_audit[name]={"module_level_function":True,"defaults":None,"kwdefaults":None,"closure":None,"bound_or_partial":False,
            "source_sha256":bytes_sha(ast.get_source_segment(function_file.read_text(),node).encode()),
            "runtime_global_policy":"_identity has no runtime globals; _inf_to_nan_func uses NumPy nan_to_num/nan from recorded1.26.4 hash-locked environment"}
    return {"sdk_root":str(root),"sdk_source_sha256":sources,"function_source_sha256":FUNCTION_SOURCE_SHA,
        "historical_authentication":{"method":"recorded Git authentication -> full cached-wheel RECORD -> callable source bytes; not a previously recorded individual file hash",
            "auth_manifest_sha256":bytes_sha(auth_bytes),"cached_record_sha256":auth["cached_record_sha256"],"inference_lock_sha256":bytes_sha(lock_bytes),
            "recorded_source_revisions":history,"installed_function_ctime_ns":stat.st_ctime_ns,"installed_function_mtime_ns":stat.st_mtime_ns,
            "installed_and_cached_same_inode":(stat.st_dev,stat.st_ino)==(cached_function.stat().st_dev,cached_function.stat().st_ino),
            "callable_FILE_RECORD_hash_match":True,"limitation":auth["limitation"]},"callable_source_audit":callable_audit,
        "function_ast_sha256":{name:bytes_sha(ast.dump(node,include_attributes=False).encode()) for name,node in definitions.items()},
        "normalization_scope":"only <function _inf_to_nan_func|_identity at 0xHEX> process addresses"}


def probability(item):
    require(type(item.get("passed")) is bool,"missing probability decision")
    for key,limit in (("max_normalization_error",1e-5),("max_cdf_absolute_error",2e-5),("max_row_total_variation",1e-4)):
        value=item.get(key);require(type(value) in (int,float) and math.isfinite(value) and 0<=value<=limit,"probability threshold failed: "+key)
    require(type(item.get("minimum_probability")) in (int,float) and math.isfinite(item["minimum_probability"]) and item["minimum_probability"]>=0,"negative/nonfinite probability")


def prediction(item,*,repair_identity=False,changes=None,path="",exact_numeric=False):
    require(all(item.get(k) is True for k in ("eight_members","exact_final_borders","final_float32")),"prediction shape/borders/dtype failed")
    probability(item["final"]);require(item["final"].get("passed") is True,"original final numeric comparison failed");members=item.get("members",[]);had_identity_error=False
    require([m.get("index") for m in members]==list(range(8)),"all8 ordered member comparisons required")
    for member in members:
        probability(member);require(member.get("exact_borders") is True and member.get("float32") is True,"member borders/dtype failed")
        if repair_identity and member.get("configuration_identity_equal") is False:
            had_identity_error=True;member["configuration_identity_equal"]=True;changes.append(path+f"/members/{member['index']}/configuration_identity_equal")
        elif repair_identity:require(member.get("passed") is True,"unexplained nonidentity member failure")
        require(member.get("configuration_identity_equal") is True,"uncorrected member identity mismatch")
        if exact_numeric:
            require(member["max_cdf_absolute_error"]==member["max_row_total_variation"]==0 and member.get("logits",{}).get("max_centered_logit_error_on_common_finite_bins")==0,"golden/fresh numeric output differs")
        if repair_identity:member["passed"]=True
        else:require(member.get("passed") is True,"member comparison already failed")
    inverse=item.get("inverse_cdf",{})
    require(inverse.get("passed") is True and inverse.get("icdf_contract_version")==2 and inverse.get("independent_fp32_replay_failures")==0
        and inverse.get("same_logits_and_borders_exact_failures")==0 and inverse.get("probability_space_failures")==[],"ICDFv2 independent replay failed")
    if exact_numeric:require(item["final"]["max_cdf_absolute_error"]==item["final"]["max_row_total_variation"]==0,"golden/fresh final output differs")
    if "saved_full1024_final" in item:probability(item["saved_full1024_final"])
    if repair_identity:
        require(item.get("passed") is True or had_identity_error,"unexplained nonidentity prediction failure");item["passed"]=True
    else:require(item.get("passed") is True,"prediction already failed")


def context(item,*,expanded,gqa_sha):
    keys=("original_alias_restored","original_function_restored","original_code_unchanged","active_code_unchanged","gqa_source_unchanged","sdk_sources_unchanged","source_guard_passed")
    require(all(item.get(k) is True for k in keys) and item.get("module_sha256")==gqa_sha,"GQA alias/source restoration failed")
    require(item.get("executed_original_attention_code_sha256")==item.get("original_attention_code_sha256") and bool(item.get("original_attention_code_sha256"))
        and item.get("original_attention_code_sha256")!=item.get("active_attention_code_sha256"),"actual original attention fingerprint missing")
    require(type(item.get("attention_call_count")) is int and item["attention_call_count"]>0,"actual attention did not execute")
    require((item.get("expanded_attention_calls",0)>0 if expanded else item.get("expanded_attention_calls")==0),"incorrect expanded/uncached GQA route")
    require(item.get("prediction_math_dtype")=="torch.float32" and all(item.get(k) is False for k in ("prefill_projection_changed","dtype_changed","dropout_changed","softmax_scale_changed","sdk_files_modified")),"GQA precision/arithmetic scope changed")
    for view in item.get("view_examples",[]):require(view.get("shares_storage") is True and view["after_stride"][-2]==0,"expanded view copied storage")


def gqa_report(item,stage,gqa_sha):
    require(item.get("passed") is True and [m.get("member_index") for m in item.get("members",[])]==list(range(8)),"all8 actual GQA witnesses missing")
    descriptors=[]
    for member in item["members"]:
        index=member["member_index"];fallback=stage==10 and index==5
        require(member.get("passed") is True and member.get("expected_route")==("uncached" if fallback else "expanded_cached"),"GQA member route failed")
        context(member["descriptor"],expanded=not fallback,gqa_sha=gqa_sha);descriptors.append(member["descriptor"])
    return descriptors


def component(item,stage,*,wrapper_sha,hybrid_sha,mixture_sha,repair_identity=False,changes=None,path="",outlier=False):
    expected=hybrid_sha if outlier else wrapper_sha
    require(item.get("module_sha256")==expected and item.get("component_mixture_module_sha256")==mixture_sha
        and item.get("sampler_module_sha256")==mixture_sha,"component source binding failed")
    require(item.get("fast_icdf_same_logits_failures")==0 and item.get("executor_metadata",{}).get("full8_member_lists_restored") is True
        and item["executor_metadata"].get("original_translator_alias_restored") is True,"component sampler/restoration failed")
    require(item.get("roundoff",{}).get("mass_corrected_reconstruction_passed") is True and item["roundoff"].get("component_mass_guard_passed") is True,"raw mass reconstruction failed")
    for key in ("normalized_mean_captured_raw_vs_full_fp32","normalized_mean_observed_raw_vs_full_fp32","mass_corrected_observed_outputs_vs_full_fp32","hybrid_diagnostic_mean_roundoff"):probability(item[key]);require(item[key].get("passed") is True,"mixture reconstruction failed")
    members=item.get("members",[]);had_identity_error=False;require([m.get("ensemble_index") for m in members]==list(range(8)),"all8 actual components required")
    routes={str(i):"uncached" if stage==10 and i==5 else "cached" for i in range(8)}
    require(item["executor_metadata"].get("member_routes")==routes,"component route policy changed")
    for member in members:
        probability(member);probability(member["normalized_component"])
        require(member["normalized_component"].get("passed") is True,"component normalization failed")
        if repair_identity and member.get("configuration_identity_equal") is False:
            had_identity_error=True;member["configuration_identity_equal"]=True;changes.append(path+f"/members/{member['ensemble_index']}/configuration_identity_equal")
        elif repair_identity:require(member.get("passed") is True,"unexplained nonidentity component failure")
        keys=("configuration_identity_equal","exact_borders","float32","component_raw_mass_matches_fp64_sum","fast_icdf_same_logits_exact","executed_corrected_helper_fingerprint_matches")
        require(all(member.get(k) is True for k in keys) and member.get("corrected_cdf_call_delta")==1,"component identity/CDF/math failed")
        require(member.get("execution_route")==routes[str(member["ensemble_index"])] and member.get("route_counter_delta")==1 and member.get("other_route_counter_delta")==0,"actual component route counters failed")
        error,bound=member.get("component_raw_mass_sum_error"),member.get("component_raw_mass_fp64_roundoff_bound")
        require(type(error) in (int,float) and type(bound) in (int,float) and math.isfinite(error) and math.isfinite(bound)
            and 0<=error<=bound<=8*5000*2.220446049250313e-16,"FP64 raw mass error/bound failed")
        require(0<member.get("mass_min",0)<=member.get("mass_max",0)<=1.00001 and member.get("component_head_mass_envelope")==1.00001,"raw head mass envelope changed")
        if repair_identity:member["passed"]=True
        else:require(member.get("passed") is True,"actual component already failed")
    if repair_identity:
        require(item.get("passed") is True or had_identity_error,"unexplained nonidentity component report failure");item["passed"]=True
    else:require(item.get("passed") is True,"actual component qualification failed")


def recover_document(raw,golden,*,raw_sha,golden_sha,source_evidence):
    from posteriorforge import execution_gqa_hybrid as execution
    from posteriorforge import execution_cache as cache,execution_mixture as mixture
    execution.require_frozen_sources();metadata=execution.execution_metadata()
    require(raw.get("schema")==RAW_SCHEMA and raw.get("status")=="conditional_mismatch" and raw.get("qualifier_source_sha256")==RAW_PRODUCER_SHA,"exact immutable failed GPU producer required")
    require(raw.get("qualified") is False and raw.get("full21_gqa_hybrid_qualified") is False and raw.get("full21_qualified") is False
        and raw.get("cache_only_full21_qualified") is False and raw.get("sources_unchanged") is True,"raw producer state changed")
    require(raw.get("execution_engine")==metadata and raw.get("module_sha256")==execution.execution_source_sha256()
        and raw.get("cache_module_sha256")==cache.execution_source_sha256() and raw.get("mixture_module_sha256")==mixture.execution_source_sha256()
        and raw.get("gqa_module_sha256")==execution.PINNED_GQA_SHA256 and raw.get("hybrid_module_sha256")==execution.PINNED_HYBRID_SHA256,"raw source graph changed")
    require(raw.get("thresholds")==THRESHOLDS and raw.get("row_counts")==ROWS and raw.get("icdf_contract_version")==2
        and raw.get("stage_coverage")==list(range(21)) and raw.get("fallback_map")=={"10":[5]},"raw threshold/coverage/fallback changed")
    require(raw.get("fallback_policy_sha256")==metadata["fallback_policy_sha256"] and raw.get("posterior_vectors_generated")==0
        and raw.get("scientific_gate_evaluated") is False,"raw scope changed")
    require(raw.get("golden_manifest_sha256")==golden_sha and golden.get("status")=="complete" and golden.get("full21_original_control_captured") is True
        and golden.get("sources_unchanged") is True and golden.get("member_rows_retained")==128,"golden source/coverage changed")
    for key in ("checkpoint_sha256","scientific_fingerprint","parameter_order","context_theta","context_obs","query","prefix_manifest_sha256","prefix_samples_sha256"):
        require(raw.get(key)==golden.get(key),"golden/raw binding changed: "+key)
    require(raw.get("fixed_prefix_row_indices")==golden.get("logical_row_indices") and raw.get("original_sources")==golden["sources"]["baseline"],"raw control cohort/source changed")
    require(source_evidence["sdk_source_sha256"]==golden["sources"]["SDKsourcehashes"] and source_evidence["function_source_sha256"]==FUNCTION_SOURCE_SHA,"SDK function source identity changed")
    coverage={"fresh_original_control":{"rows":1024,"members":8,"member_rows":1024,"stress_rows":8,"calls_per_stage_main":1},
        "saved_golden":{"rows":1024,"members":8,"member_rows":128,"same_as_fresh1024_coverage":False},
        "candidate":{"chunks":ROWS,"members_per_chunk":8,"member_rows_at_largest_chunk":1024,"stress_rows":8}}
    require(raw.get("producer_coverage")==coverage,"original all8 member1024 coverage changed")
    control=raw.get("original_control_chunk_invariance",{});require([v.get("stage_index") for v in control.get("stage_evidence",[])]==list(range(21)),"all21 original control invariance evidence required")
    for row in control["stage_evidence"]:
        require(row.get("original_control_only") is True and row.get("candidate_proof_inherited") is False,"old candidate qualification inherited")
        require(set(row.get("comparisons",{}))==set(map(str,ROWS)),"original control chunk coverage changed")
        for comparison in row["comparisons"].values():prediction(comparison)
    adapter=raw.get("compatibility_adapter",{})
    require(all(adapter.get(k) is True for k in ("original_alias_restored","sdk_sources_unchanged","sdk_aliases_unchanged","source_guard_passed","original_alias_identity_verified","translator_code_unchanged"))
        and adapter.get("patched_call_count",0)>0 and adapter.get("executed_corrected_cdf_code_sha256")==adapter.get("corrected_cdf_code_sha256")
        and adapter.get("corrected_cdf_code_sha256")!=adapter.get("original_cdf_code_sha256"),"raw actual corrected CDF execution/restoration failed")
    result=deepcopy(raw);changes=[];evidence=[]
    require([r.get("stage_index") for r in result["stage_results"]]==list(range(21)),"all21 ordered stage records required")
    for stage,row in enumerate(result["stage_results"]):
        prepared=row["prepared"];actual=prepared.get("configuration_identities",[]);record=golden["stages"][stage]
        require(len(actual)==8 and record["stage_index"]==stage,"actual prepared/golden8member order changed")
        if stage==10:require(prepared.get("fallback_configuration_identities")==actual and prepared.get("fallback_configurations_match") is True,"uncached fallback configuration identity differs")
        stage_witnesses=[]
        for payload in record["payloads"]:
            require([m["ensemble_index"] for m in payload["metadata"]["members"]]==list(range(8)),"golden member order changed")
            stage_witnesses.append({"payload":payload["file"],"payload_sha256":payload["sha256"],"members":[semantic_identity(a,m["configuration"]) for a,m in zip(actual,payload["metadata"]["members"])]})
        evidence.append({"stage_index":stage,"actual_prepared_member_count":8,"payload_witnesses":stage_witnesses,"fresh_to_candidate_identity_transitivity_required":True})
        prefix=f"/stage_results/{stage}"
        require(row.get("fallback_members")==([5] if stage==10 else []) and row.get("member_routes")=={str(i):"uncached" if stage==10 and i==5 else "cached" for i in range(8)},"stage route changed")
        require([v.get("query_rows") for v in row.get("comparisons",[])]==ROWS,"candidate chunk coverage changed")
        descriptors=[]
        for value in row["comparisons"]:
            prediction(value["candidate_vs_fresh_control1024_slice"])
            require(value.get("reference_is_fresh1024_slice") is True and value.get("same_chunk_control_recomputed") is False,"original control reference policy changed")
            descriptors+=gqa_report(value["actual_gqa_execution"],stage,execution.PINNED_GQA_SHA256)
        prediction(row["stress_comparison"]);descriptors+=gqa_report(row["stress_gqa_execution"],stage,execution.PINNED_GQA_SHA256)
        for name in ("component_qualification","stress_component_qualification"):
            component(row[name],stage,wrapper_sha=execution.execution_source_sha256(),hybrid_sha=execution.PINNED_HYBRID_SHA256,mixture_sha=mixture.execution_source_sha256())
        for name in ("fresh_control_vs_saved_golden","stress_control_vs_saved_golden"):
            prediction(row[name],repair_identity=True,changes=changes,path=prefix+"/"+name,exact_numeric=True)
        require(row.get("normalization_train_only_verified") is True and row.get("preprocessor_identity") is True and row.get("golden_fit_context_equal") is True
            and row.get("prepared_state_unchanged") is True and row.get("rng_unchanged") is True,"previous state/preprocessing/RNG invariant failed")
        require(row["fresh_original_fit_context"]==record["fit_transform_context"],"saved/fresh fitted context differs")
        for key in ("control_model_contract","resident_model_contract","uncached_model_contract"):
            contract=row.get(key)
            if key=="uncached_model_contract" and stage!=10:require(contract is None,"unexpected uncached model");continue
            require(contract and contract.get("input_normalization_train_only_verified") is True and contract.get("autocast") is False and contract.get("forced_inference_dtype")=="torch.float32","actual model precision/normalizer failed")
            normalizers=[m for m in contract["modules"] if m.get("type")=="InputNormalizationEncoderStep"]
            require(normalizers and all(m["settings"].get("normalize_on_train_only") is True for m in normalizers),"actual normalizer setting differs")
        require(all(v.get("before")==v.get("after") for v in row["rng"]),"actual RNG state changed")
        outliers=row.get("outlier_checks",[]);require([v.get("label") for v in outliers]==(["rep2_outlier_prefix_1","rep2_outlier_prefix_16"] if stage==19 else []),"predeclared outlier coverage changed")
        for i,outlier in enumerate(outliers):
            require(outlier.get("target_excluded_from_query") is True and outlier.get("rows")==[1,16][i],"outlier query scope changed")
            prediction(outlier["prediction_comparison"],repair_identity=True,changes=changes,path=prefix+f"/outlier_checks/{i}/prediction_comparison")
            descriptors+=gqa_report(outlier["actual_gqa_execution"],stage,execution.PINNED_GQA_SHA256)
            component(outlier["component_qualification"],stage,wrapper_sha=execution.execution_source_sha256(),hybrid_sha=execution.PINNED_HYBRID_SHA256,
                mixture_sha=mixture.execution_source_sha256(),repair_identity=True,changes=changes,path=prefix+f"/outlier_checks/{i}/component_qualification",outlier=True)
            outlier["passed"]=True
        if stage==0:
            dedup=row["stage0_shared_distribution"];probability(dedup);prediction(dedup["full1024_comparison"])
            require(dedup.get("passed") is True and dedup.get("shared_icdf_same_logits_exact") is True,"original stage0 dedup failed")
            for key in ("single_actual_gqa_context","full_actual_gqa_context"):context(dedup[key],expanded=True,gqa_sha=execution.PINNED_GQA_SHA256);descriptors.append(dedup[key])
        else:require(row.get("stage0_shared_distribution") is None,"cross-stage density memo unexpected")
        counters=row["execution_counters"]
        require(counters.get("all_contexts_restored") is True and counters.get("all_sources_unchanged") is True and counters.get("contexts")==len(descriptors),"actual GQA context coverage/restoration failed")
        for key in ("attention_call_count","expanded_attention_calls","expanded_tensor_views"):
            require(counters.get(key)==sum(d[key] for d in descriptors),"actual GQA counters differ: "+key)
        row["passed"]=True
    require(changes,"no recognized erroneous identity decisions to recover")
    result.update(schema=SCHEMA,status="gqa_hybrid_screen_pass",qualified=True,full21_gqa_hybrid_qualified=True,
        full21_component_qualified=True,full21_sampler_qualified=True,failed_stage_indices=[],
        raw_producer_status=raw["status"],raw_producer_schema=raw["schema"],raw_status_preserved=True,
        raw_qualification_manifest_sha256=raw_sha,raw_qualifier_source_sha256=RAW_PRODUCER_SHA,
        recovery_source_sha256=sha(__file__),recovery_id="pinned_config_pickle_type_and_function_address_repr_v1",
        recovery_cause="Cross-process function addresses in config repr caused identity false negatives; exact type/pickle SHA and all numeric checks retained.",
        configuration_identity_evidence=evidence,configuration_identity_corrected_paths=changes,
        configuration_identity_sdk_evidence=source_evidence,
        recovery_inputs={"raw_gpu_proof":{"file":"inputs/raw-gpu-proof.json","sha256":raw_sha},"golden_manifest":{"file":"inputs/golden-manifest.json","sha256":golden_sha}},
        recovery_scope="CPU metadata identity and derived predicates only; no model/GPU calls or numerical tolerance changes")
    return result


def recover(*,raw,goldens,output,sdk_root):
    from posteriorforge.artifacts import write_json
    raw_path=Path(raw);golden_path=Path(goldens)/"manifest.json";raw_bytes=raw_path.read_bytes();golden_bytes=golden_path.read_bytes()
    raw_document=json.loads(raw_bytes);golden_document=json.loads(golden_bytes)
    evidence=sdk_evidence(sdk_root,raw=raw_document,golden=golden_document);document=recover_document(raw_document,golden_document,raw_sha=bytes_sha(raw_bytes),golden_sha=bytes_sha(golden_bytes),source_evidence=evidence)
    # Recovery never changes a source manifest. Archive exact bytes before publishing.
    root=Path(output);root.mkdir(parents=True,exist_ok=False);(root/"inputs").mkdir()
    (root/"inputs/raw-gpu-proof.json").write_bytes(raw_bytes);(root/"inputs/golden-manifest.json").write_bytes(golden_bytes)
    require(raw_path.read_bytes()==raw_bytes and golden_path.read_bytes()==golden_bytes,"input changed during CPU recovery")
    write_json(root/"manifest.json",document);verify_recovery_artifact(root/"manifest.json")
    return document


def verify_recovery_artifact(path):
    path=Path(path);document=json.loads(path.read_text());require(document.get("schema")==SCHEMA and document.get("recovery_source_sha256")==sha(__file__),"reviewed recovery source mismatch")
    inputs=[]
    for name in ("raw_gpu_proof","golden_manifest"):
        entry=document["recovery_inputs"][name];source=(path.parent/entry["file"]).resolve()
        require(source.is_relative_to(path.parent.resolve()) and sha(source)==entry["sha256"],"archived recovery input checksum mismatch")
        inputs.append(json.loads(source.read_text()))
    evidence=sdk_evidence(document["configuration_identity_sdk_evidence"]["sdk_root"],raw=inputs[0],golden=inputs[1])
    expected=recover_document(inputs[0],inputs[1],raw_sha=document["raw_qualification_manifest_sha256"],golden_sha=document["golden_manifest_sha256"],source_evidence=evidence)
    require(document==expected,"CPU recovery artifact differs from complete predicate replay")
    return document


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("raw","goldens","output","sdk-root"):parser.add_argument("--"+name,required=True)
    result=recover(**vars(parser.parse_args()));print(json.dumps({k:result[k] for k in ("schema","status","raw_producer_status","raw_qualification_manifest_sha256","recovery_source_sha256")}))
