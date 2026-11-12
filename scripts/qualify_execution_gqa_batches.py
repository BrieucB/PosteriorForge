#!/usr/bin/env python3
"""Bounded all21 large-batch GQA engineering qualification; no posterior draws.

Built with PriorLabs-TabPFN. Every row is compared directly with fresh original
1024-main +8-stress predictions. A repeated reference pool is not a posterior.
"""
from __future__ import annotations
import argparse
import base64
import csv
import ast
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import time
import numpy as np

SCHEMA="posteriorforge.execution_gqa_batch_qualification.v1"
BATCH_SIZES=(4096,6250,10000)
COMPARISON_BLOCK_ROWS=1024
RECOVERY_SHA="83064f54712014b5c4c66569e854872caff03593cfdc337f2472a7b5723aea02"
BASE_SHA="6c927a5c278c7fc54b556ac557ecef64f5e6338f5ae13bf466f3b1e6a4640f90"
PROBE_SHA="62d624c4f4d2b3231b16a932e5818597d0718d2bfb4ac6137c7d7b00fe93bf00"


def require(value,message):
    if not value:raise RuntimeError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_helper(name,expected):
    path=Path(__file__).with_name(name);require(sha(path)==expected,"frozen helper source changed: "+name)
    spec=importlib.util.spec_from_file_location("_large_batch_"+path.stem,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def verify_portable_sources(root,bundle,*,sdk_root):
    """Relocatable cryptographic source check; no ctime or original-machine path."""
    root=Path(root).resolve();sdk_root=Path(sdk_root).resolve()
    for entry in bundle["archives"].values():
        path=(root/entry["file"]).resolve()
        require(path.is_relative_to(root) and sha(path)==entry["sha256"],"portable source archive changed")
    record=root/bundle["archives"]["wheel_record"]["file"];record_rows=list(csv.reader(record.open()))
    recorded={name:digest.removeprefix("sha256=") for name,digest,*_ in record_rows if digest.startswith("sha256=")}
    expected_sources={name:base64.urlsafe_b64decode(value+"="*((4-len(value)%4)%4)).hex() for name,value in recorded.items() if name.startswith("tabpfn/") and name.endswith(".py")}
    require(bundle["sdk_source_sha256"]==expected_sources and bool(expected_sources),"complete archived SDK Python source set required")
    for name,expected in bundle["sdk_source_sha256"].items():
        require(name.startswith("tabpfn/") and Path(name).suffix==".py","only declared SDK Python sources are mapped")
        path=sdk_root/Path(name).relative_to("tabpfn");require(path.resolve().is_relative_to(sdk_root) and sha(path)==expected,"relocated SDK source mismatch: "+name)
        digest=base64.urlsafe_b64encode(bytes.fromhex(expected)).decode().rstrip("=")
        require(recorded.get(name)==digest,"SDK source is not bound by archived wheel RECORD")
    return True


def portable_source_bundle(root,*,sdk_root,qualification):
    root=Path(root);repo=Path(__file__).resolve().parents[1]
    auth_bytes=(repo/"evidence/diagnostics/model-sdk-source-authenticity.json").read_bytes();auth=json.loads(auth_bytes)
    cached=Path(auth["cached_extracted_wheel_root"]);record=cached/"tabpfn-2.0.9.dist-info/RECORD"
    require(sha(record)==auth["cached_record_sha256"],"authenticated complete wheel RECORD changed")
    target=root/"source-auth";target.mkdir(exist_ok=False)
    contents={"wheel_record":("tabpfn-wheel-RECORD.csv",record.read_bytes()),"historical_wheel_authentication":("historical-wheel-authentication.json",auth_bytes),
        "inference_lock":("inference.lock",(repo/"environments/inference.lock").read_bytes()),"base_qualification":("base-gqa-qualification.json",Path(qualification).read_bytes())}
    archives={}
    for name,(filename,content) in contents.items():
        path=target/filename;path.write_bytes(content);archives[name]={"file":str(path.relative_to(root)),"sha256":sha(path)}
    sources={}
    for name,digest,*_ in csv.reader(record.open()):
        if name.startswith("tabpfn/") and name.endswith(".py") and digest.startswith("sha256="):
            encoded=digest.removeprefix("sha256=");sources[name]=base64.urlsafe_b64decode(encoded+"="*((4-len(encoded)%4)%4)).hex()
    bundle={"schema":"posteriorforge.portable_sdk_source_authentication.v1","archives":archives,"sdk_source_sha256":sources,
        "sdk_distribution":"tabpfn==2.0.9","all_wheel_sdk_python_sources_checked":True,"verification_requires_original_machine_paths":False,"verification_uses_file_ctime":False,
        "historical_absolute_paths_are_informational":True,"marshal_code_fingerprints_are_within_process_execution_witnesses_not_cross_install_path_identifiers":True,
        "base_recovery_scope":"historic provenance; fresh same-process batch controls and source bytes independently establish this new proof",
        "wheel_zip_limitation":auth["limitation"]}
    verify_portable_sources(root,bundle,sdk_root=sdk_root);return bundle


def repeated_pool_indices(rows):
    require(type(rows) is int and rows in BATCH_SIZES,"exact predeclared neural batches4096/6250/10000 required")
    return np.arange(rows,dtype=np.int64)%1032


def compare_blocks(reference,candidate,indices,utilities,*,block_rows=COMPARISON_BLOCK_ROWS):
    """Compare all raw row masses, never repair normalization or sum error bounds."""
    reference=np.asarray(reference);candidate=np.asarray(candidate);indices=np.asarray(indices)
    require(type(block_rows) is int and 1<=block_rows<=1024,"comparison blocks must have at most1024rows")
    require(reference.ndim==candidate.ndim==2 and candidate.shape[1]==reference.shape[1] and indices.shape==(len(candidate),)
        and indices.dtype.kind in "iu" and np.all((indices>=0)&(indices<len(reference))),"direct reference row mapping invalid")
    result={"passed":True,"rows_compared":0,"comparison_block_cap":block_rows,"blocks":0,"worst_cdf_row":None,
        "max_cdf_absolute_error":0.,"max_row_total_variation":0.,"max_normalization_error":0.,"max_probability_absolute_error":0.,"minimum_probability":float("inf")}
    for first in range(0,len(candidate),block_rows):
        last=min(first+block_rows,len(candidate));a=reference[indices[first:last]];b=candidate[first:last]
        metric=utilities.compare_probabilities(a,b);result["rows_compared"]+=last-first;result["blocks"]+=1
        result["passed"] &= bool(metric["passed"])
        if "reason" in metric:result.setdefault("failure_reason",metric["reason"]);continue
        if metric["max_cdf_absolute_error"]>=result["max_cdf_absolute_error"]:
            delta=np.abs(np.cumsum(a.astype(np.float64)-b.astype(np.float64),axis=-1)).max(-1)
            result["worst_cdf_row"]=first+int(np.argmax(delta))
        for key in ("max_cdf_absolute_error","max_row_total_variation","max_normalization_error","max_probability_absolute_error"):
            result[key]=max(result[key],metric[key])
        result["minimum_probability"]=min(result["minimum_probability"],metric["minimum_probability"])
    if not np.isfinite(result["minimum_probability"]):result["minimum_probability"]=None
    return result


def shape_evidence(descriptor,rows):
    shapes=[ast.literal_eval(key) for key in descriptor.get("head_shape_histogram",{})]
    lengths=sorted({shape["q"][-3] for shape in shapes if shape.get("q") is not None and len(shape["q"])>=4})
    return {"passed":rows in lengths,"requested_neural_query_rows":rows,"observed_attention_query_lengths":lengths,
        "head_shape_histogram":descriptor.get("head_shape_histogram",{}),"effective_head_shape_histogram":descriptor.get("effective_head_shape_histogram",{})}


def concatenate_reference(main,stress,torch):
    require(len(main["members"])==len(stress["members"])==8,"all8 fresh reference members required")
    members=[]
    for left,right in zip(main["members"],stress["members"]):
        require(left["configuration_identity"]==right["configuration_identity"] and torch.equal(left["from_borders"],right["from_borders"])
            and torch.equal(left["to_borders"],right["to_borders"]),"main/stress original member identity differs")
        members.append({**left,"raw_logits":torch.cat((left["raw_logits"],right["raw_logits"])),
            "translated_probabilities":torch.cat((left["translated_probabilities"],right["translated_probabilities"]))})
    require(torch.equal(main["criterion"].borders,stress["criterion"].borders),"reference canonical borders differ")
    return {"criterion":main["criterion"],"logits":torch.cat((main["logits"],stress["logits"])),"members":members}


def probability_accumulators(rows,bins):
    return {"raw_sum":np.zeros((rows,bins),dtype=np.float64),"weighted_output_sum":np.zeros((rows,bins),dtype=np.float64),"mass_sum":np.zeros(rows,dtype=np.float64)}


def accumulate_component(accumulator,raw,normalized,mass):
    for first in range(0,len(raw),1024):
        last=min(first+1024,len(raw));r=raw[first:last].astype(np.float64);m=mass[first:last]
        accumulator["raw_sum"][first:last]+=r
        accumulator["weighted_output_sum"][first:last]+=normalized[first:last].astype(np.float64)*m[:,None]
        accumulator["mass_sum"][first:last]+=m


def fixed_uniform_sampler_check(prediction,torch,mixture):
    n=len(prediction["logits"]);indices=np.linspace(0,n-1,8,dtype=np.int64)
    logits=prediction["logits"][torch.from_numpy(indices)]
    uniforms=torch.tensor([0.,np.finfo(np.float32).eps/2,np.finfo(np.float32).eps,.1,.5,.9,1-np.finfo(np.float32).eps,np.nextafter(np.float32(1),np.float32(0))],dtype=torch.float32)
    fast=mixture.finite_bar_icdf(logits,prediction["criterion"].borders,uniforms)
    stock=torch.stack([prediction["criterion"].icdf(logits[i],float(uniforms[i])).reshape(()) for i in range(8)])
    return {"passed":bool(((fast==stock)|(torch.isnan(fast)&torch.isnan(stock))).all()),"row_indices":indices.tolist(),
        "uniforms":uniforms.numpy().tolist(),"stock_nonfinite_rows":int((~torch.isfinite(stock)).sum()),"exact_same_logits_finite_bar_semantics":True}


def full21_batch_outcomes(stage_results,stage_coverage):
    return {str(count):bool(stage_coverage==list(range(21)) and [r.get("stage_index") for r in stage_results]==list(range(21)) and all(
        stage.get("normalization_train_only_verified") is True and stage.get("all8_preprocessor_identity") is True and stage.get("final_prepared_state_unchanged") is True
        and next((value.get("passed") is True for value in stage.get("batch_results",[]) if value.get("query_rows")==count),False)
        for stage in stage_results)) for count in BATCH_SIZES}


def qualify(*,banks,prefix_run,checkpoint,ledger,qualification,output,max_seconds=2250):
    require(60<=max_seconds<=2250,"bounded60..2250seconds required")
    from posteriorforge.artifacts import runtime_identity,write_json
    from posteriorforge.reference_grid import _require_budgeted_gpu
    from posteriorforge import execution_gqa_hybrid as execution,execution_cache as cache,execution_mixture as mixture
    budget=_require_budgeted_gpu(ledger);require(budget["gpus"]==1,"one registered GPU allocation required")
    root=Path(output);root.mkdir(parents=True,exist_ok=False);start=time.perf_counter();deadline=start+max_seconds
    result={**runtime_identity(),"schema":SCHEMA,"status":"running","attribution":"Built with PriorLabs-TabPFN","budget_binding":budget,
        "posterior_vectors_generated":0,"scientific_gate_evaluated":False,"full21_qualified":False,"cache_only_full21_qualified":False,
        "requested_batch_sizes":list(BATCH_SIZES),"candidate_query_batch_cap":max(BATCH_SIZES),"comparison_block_cap":1024,
        "reference_pool":{"main_rows":1024,"stress_rows":8,"pool_rows":1032,"main_member_count":8,"stress_member_count":8,"original_calls_per_stage":2},
        "repeated_pool_mapping":"candidate row i -> fresh reference pool row i modulo1032; first1024 main then8 stress",
        "changed_execution_setting":"explicit query_batch_size cap only; same frozen GQA/hybrid/sampler sources",
        "qualifier_source_sha256":sha(__file__),"execution_engine":execution.execution_metadata(),"stage_results":[],"stage_coverage":[],
        "full21_batch_qualified":{str(n):False for n in BATCH_SIZES},"qualified_batch_sizes":[],"qualified":False,
        "scope":"bounded repeated-pool neural batch parity; no posterior draws, new-world coverage or end-to-end speed claim"}
    engine=None;active=None;failed_rows={};write_json(root/"manifest.json",result)
    try:
        import torch
        from tabpfn import TabPFNRegressor
        from importlib import import_module
        from posteriorforge.inference import CHECKPOINT_SHA256,frozen_regressor_settings,support_mask
        from posteriorforge.inference_cdf import guard_baseline_source
        from posteriorforge.banks import load_context,load_heldout,derive_seed
        from posteriorforge.gate import verify_run_binding
        from posteriorforge.compat_cdf import finite_cdf_compatibility
        from posteriorforge.science.analytical_benchmark_v2 import array_descriptor,build_problem
        recovery=load_helper("recover_gqa_hybrid_qualification.py",RECOVERY_SHA)
        utilities=load_helper("qualify_execution_cache.py",BASE_SHA);probe=load_helper("boundedprobe_gqa_expansion.py",PROBE_SHA)
        qpath=Path(qualification);qsha=sha(qpath);proof=recovery.verify_recovery_artifact(qpath)
        require(proof["full21_gqa_hybrid_qualified"] is True and proof["fallback_map"]=={"10":[5]},"recovered full21 GQA hybrid proof required")
        execution.require_frozen_sources();engine_sha=execution.execution_source_sha256();original_source=guard_baseline_source()
        sdk_root=Path(import_module("tabpfn").__file__).parent
        result["portable_source_authentication"]=portable_source_bundle(root,sdk_root=sdk_root,qualification=qpath)
        require(torch.cuda.is_available() and torch.cuda.device_count()==1,"one visible CUDA device required")
        require(torch.get_default_dtype()==torch.float32 and torch.get_float32_matmul_precision()=="highest" and not torch.backends.cuda.matmul.allow_tf32,"unchanged FP32 highest/TF32False required")
        require(sha(checkpoint)==CHECKPOINT_SHA256 and np.__version__=="1.26.4","immutable checkpoint/NumPy1.26.4 required")
        context=load_context(banks,20261111,4096);heldout=load_heldout(banks,0);observation=heldout.obs.reshape(1,33)
        prefix=Path(prefix_run);manifest=json.loads((prefix/"manifest.json").read_text());verify_run_binding(manifest,context,heldout.obs,20261111,0)
        require(manifest.get("status")=="complete" and manifest.get("variant")=="cdf_arithmetic_v1" and sha(prefix/"samples.npz")==manifest["samples_sha256"],"immutable corrected-control prefixes required")
        with np.load(prefix/"samples.npz",allow_pickle=False) as data:samples=data["samples"].copy()
        require(samples.shape==(50000,21) and support_mask(samples).all(),"50000 complete supported saved prefix source required")
        positions=np.linspace(0,49999,1024,dtype=np.int64);theta=context.theta.astype(np.float32);obs=context.obs.astype(np.float32)
        problem=build_problem();joint=np.concatenate((obs,theta),axis=1);noise=[problem.parameter_order.index(name) for name in problem.prior.noise_names]
        binding={"checkpoint_sha256":CHECKPOINT_SHA256,"scientific_fingerprint":problem.scientific_fingerprint,"parameter_order":list(problem.parameter_order),
            "context_theta":array_descriptor(context.theta),"context_obs":array_descriptor(context.obs),"query":array_descriptor(observation),
            "prefix_manifest_sha256":sha(prefix/"manifest.json"),"prefix_samples_sha256":manifest["samples_sha256"]}
        require(all(proof.get(k)==v for k,v in binding.items()) and proof["fixed_prefix_row_indices"]==positions.tolist(),"batch/base proof scientific binding changed")
        seed=derive_seed(20261111,"execution-gqa-batch-qualification",case=0)
        np.random.seed(seed);random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS","16")))
        result.update(**binding,base_qualification_manifest_sha256=qsha,base_qualification_schema=proof["schema"],base_full21_gqa_hybrid_qualified=True,
            recovered_producer_sha256=RECOVERY_SHA,thresholds=dict(utilities.THRESHOLDS),fixed_prefix_row_indices=positions.tolist(),seed=seed,
            module_sha256=engine_sha,gqa_module_sha256=execution.PINNED_GQA_SHA256,hybrid_module_sha256=execution.PINNED_HYBRID_SHA256,
            cache_module_sha256=cache.execution_source_sha256(),mixture_module_sha256=mixture.execution_source_sha256(),original_sources=original_source,
            **execution.hybrid.policy_metadata({10:[5]}))
        kwargs=frozen_regressor_settings(checkpoint);kwargs["inference_precision"]=torch.float32
        with finite_cdf_compatibility() as active:
            for stage in range(21):
                if time.perf_counter()+60>=deadline:result["status"]="incomplete";break
                result["active_stage"]=stage;write_json(root/"manifest.json",result)
                queries,stress=utilities.fixed_queries(observation,samples[positions],theta,stage,noise,problem.prior.bounds)
                pool=np.concatenate((queries,stress),axis=0);baseline=TabPFNRegressor(**kwargs);initial=utilities.rng_state(torch)
                baseline.fit(joint[:,:33+stage],joint[:,33+stage])
                main=cache.capture_corrected_prediction(baseline,queries,cdf_descriptor=active)
                stress_reference=cache.capture_corrected_prediction(baseline,stress,cdf_descriptor=active)
                reference=concatenate_reference(main,stress_reference,torch);original_contract=utilities.model_contract(baseline)
                reference_final=reference["logits"].softmax(-1).numpy();baseline.model_.cpu();utilities.restore_rng(torch,initial)
                engine=execution.GQAHybridStageExecutor(checkpoint,cdf_descriptor=active,query_batch_size=max(BATCH_SIZES))
                before_prepare=utilities.rng_identity(utilities.rng_state(torch));prepared=engine.prepare(theta,obs,stage)
                prepare_rng_unchanged=before_prepare==utilities.rng_identity(utilities.rng_state(torch));state=probe.prepared_state(engine.base)
                stage_row={"stage_index":stage,"parameter":problem.parameter_order[stage],"prepared":prepared,"batch_results":[],
                    "reference_main_rows":len(main["logits"]),"reference_stress_rows":len(stress_reference["logits"]),"reference_pool_query":array_descriptor(pool),
                    "prepare_rng_unchanged":prepare_rng_unchanged,"original_control_contract":original_contract}
                for count in BATCH_SIZES:
                    if time.perf_counter()+20>=deadline:result["status"]="incomplete";break
                    indices=repeated_pool_indices(count);query=pool[indices];bins=reference_final.shape[-1];acc=probability_accumulators(count,bins)
                    batch={"query_rows":count,"actual_input_query_shape":list(query.shape),"explicit_query_batch_cap":engine.query_batch_size,
                        "reference_row_indices":array_descriptor(indices),"status":"running","passed":False,"members":[]}
                    try:
                        torch.cuda.reset_peak_memory_stats();started=time.perf_counter();before_total=utilities.rng_identity(utilities.rng_state(torch))
                        for member in range(8):
                            calls=active["patched_call_count"];before=utilities.rng_identity(utilities.rng_state(torch));torch.cuda.synchronize();t0=time.perf_counter()
                            prediction=engine.predict_component(member,query,capture_member=True);torch.cuda.synchronize();wall=time.perf_counter()-t0
                            after=utilities.rng_identity(utilities.rng_state(torch));descriptor=prediction["gqa_attention_descriptor"]
                            recovery.context(descriptor,expanded=not(stage==10 and member==5),gqa_sha=execution.PINNED_GQA_SHA256)
                            expected_route="original_fit_preprocessors" if stage==10 and member==5 else "resident_cache"
                            require(prediction["hybrid_execution_path"]==expected_route and active["patched_call_count"]-calls==1
                                and active.get("executed_corrected_cdf_code_sha256")==active.get("corrected_cdf_code_sha256")
                                and active.get("corrected_cdf_code_sha256")!=active.get("original_cdf_code_sha256"),"actual route/private corrected CDF changed")
                            observed=prediction["member"];raw=observed["translated_probabilities"].numpy();normalized=prediction["logits"].softmax(-1).numpy()
                            require(raw.shape==normalized.shape==(count,bins) and observed["raw_logits"].shape[0]==count,"actual component neural output row count changed")
                            direct=compare_blocks(reference["members"][member]["translated_probabilities"].numpy(),raw,indices,utilities)
                            identity=observed["configuration_identity"]==reference["members"][member]["configuration_identity"]
                            borders=utilities._same(observed["from_borders"],reference["members"][member]["from_borders"]) and utilities._same(observed["to_borders"],reference["members"][member]["to_borders"])
                            dtype=str(observed["raw_logits"].dtype)==str(prediction["logits"].dtype)=="torch.float32"
                            mass=prediction["component_raw_mass"].numpy();mass_error=0.;normalization_ok=True
                            for first in range(0,count,1024):
                                last=min(first+1024,count);r=raw[first:last].astype(np.float64);m=r.sum(-1)
                                mass_error=max(mass_error,float(np.abs(mass[first:last]-m).max()))
                                normalization_ok &= bool(utilities.compare_probabilities(r/m[:,None],normalized[first:last])["passed"])
                            mass_bound=8*bins*np.finfo(np.float64).eps
                            mass_ok=bool(np.isfinite(mass).all() and np.all((mass>0)&(mass<=mixture.HEAD_MASS_ENVELOPE)) and mass_error<=mass_bound and prediction["head_mass_envelope"]==mixture.HEAD_MASS_ENVELOPE)
                            shape=shape_evidence(descriptor,count);sampler=fixed_uniform_sampler_check(prediction,torch,mixture)
                            observed_row={"ensemble_index":member,"passed":bool(direct["passed"] and identity and borders and dtype and mass_ok and normalization_ok and before==after and shape["passed"] and sampler["passed"]),
                                "direct_raw_probability_comparison":direct,"configuration_identity_equal":identity,"reference_configuration_identity":reference["members"][member]["configuration_identity"],
                                "candidate_configuration_identity":observed["configuration_identity"],"identity_comparison_scope":"same process fresh original vs actual candidate; no cross-process repr comparison",
                                "exact_borders":borders,"float32":dtype,
                                "component_raw_mass_matches_fp64_sum":mass_ok,"component_raw_mass_sum_error":mass_error,"component_raw_mass_fp64_roundoff_bound":mass_bound,
                                "mass_min":float(mass.min()),"mass_max":float(mass.max()),"head_mass_envelope":float(prediction["head_mass_envelope"]),
                                "normalized_component_passed":normalization_ok,"actual_execution_route":expected_route,"corrected_cdf_call_delta":active["patched_call_count"]-calls,
                                "rng_before":before,"rng_after":after,"actual_neural_shape":shape,"actual_gqa_descriptor":descriptor,
                                "stock_fixed_uniform_replay":sampler,"captured_predict_wall_seconds":wall,"timing_is_not_capture_free_or_end_to_end":True}
                            batch["members"].append(observed_row);accumulate_component(acc,raw,normalized,mass)
                            if not direct["passed"] and direct["worst_cdf_row"] is not None:
                                row=direct["worst_cdf_row"];key=f"stage{stage}_N{count}_member{member}"
                                failed_rows[key+"_candidate_raw_probability"]=raw[row].copy();failed_rows[key+"_reference_raw_probability"]=reference["members"][member]["translated_probabilities"][indices[row]].numpy().copy()
                                failed_rows[key+"_query"]=query[row].copy();failed_rows[key+"_physical_borders"]=prediction["criterion"].borders.numpy().copy()
                            del prediction,observed,raw,normalized,mass
                        ideal=acc["raw_sum"]/acc["mass_sum"][:,None];weighted=acc["weighted_output_sum"]/acc["mass_sum"][:,None]
                        batch.update(ideal_raw_mixture_vs_original=compare_blocks(reference_final,ideal,indices,utilities),
                            actual_mass_weighted_component_mixture_vs_original=compare_blocks(reference_final,weighted,indices,utilities),
                            normalized_mixture_comparison_is_direct_not_error_bound=True,rng_unchanged=before_total==utilities.rng_identity(utilities.rng_state(torch)),
                            prepared_state_unchanged=state==probe.prepared_state(engine.base),
                            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),elapsed_seconds=time.perf_counter()-started)
                        batch["passed"]=bool(all(m["passed"] for m in batch["members"]) and batch["ideal_raw_mixture_vs_original"]["passed"]
                            and batch["actual_mass_weighted_component_mixture_vs_original"]["passed"] and batch["rng_unchanged"] and batch["prepared_state_unchanged"] and prepare_rng_unchanged)
                        batch["status"]="batch_pass" if batch["passed"] else "conditional_mismatch"
                    except torch.cuda.OutOfMemoryError as error:
                        batch.update(status="cuda_out_of_memory",passed=False,error_type=type(error).__name__,error=str(error),
                            automatic_smaller_batch_retry=False,completed_member_count=len(batch["members"]),
                            actual_gqa_after_failure=dict(engine.last_gqa_descriptor or {}),
                            prepared_state_unchanged_after_oom=state==probe.prepared_state(engine.base),
                            rng_unchanged_after_oom=before_total==utilities.rng_identity(utilities.rng_state(torch)))
                        gc.collect();torch.cuda.empty_cache()
                    stage_row["batch_results"].append(batch);del acc,query;gc.collect()
                    result["active_batch_size"]=count;write_json(root/"manifest.json",{**result,"active_stage_result":stage_row})
                contracts=[original_contract,utilities.model_contract(engine.regressor)]
                if engine.uncached_regressor is not None:contracts.append(utilities.model_contract(engine.uncached_regressor))
                stage_row["model_contracts"]=contracts
                stage_row["normalization_train_only_verified"]=all(v["input_normalization_train_only_verified"] and not v["autocast"] and v["forced_inference_dtype"]=="torch.float32" for v in contracts)
                stage_row["all8_preprocessor_identity"]=all(v["preprocessor_sha256"]==original_contract["preprocessor_sha256"] for v in contracts)
                stage_row["final_prepared_state_unchanged"]=state==probe.prepared_state(engine.base)
                stage_row["cache_telemetry"]=engine.inspect_cache();result["stage_results"].append(stage_row);result["stage_coverage"].append(stage)
                engine.close();engine=None;write_json(root/"manifest.json",result)
                del baseline,main,stress_reference,reference,pool,reference_final
                if result["status"]=="incomplete":break
            else:result["status"]="completed_pending_restoration"
        result["compatibility_adapter"]=dict(active)
        require(active["original_alias_restored"] and active["sdk_sources_unchanged"],"corrected CDF restoration failed")
        require(guard_baseline_source()==original_source and execution.execution_source_sha256()==engine_sha and sha(__file__)==result["qualifier_source_sha256"]
            and sha(qpath)==qsha,"batch qualification source graph changed")
        execution.require_frozen_sources();cache._load_sdk();recovery.verify_recovery_artifact(qpath)
        verify_portable_sources(root,result["portable_source_authentication"],sdk_root=sdk_root)
        result["full21_batch_qualified"]=full21_batch_outcomes(result["stage_results"],result["stage_coverage"])
        qualified=[n for n in BATCH_SIZES if result["full21_batch_qualified"][str(n)]]
        result.update(qualified_batch_sizes=qualified,qualified=bool(qualified),all_requested_batch_sizes_qualified=len(qualified)==len(BATCH_SIZES),
            status="incomplete" if result["status"]=="incomplete" else "batch_screen_pass" if qualified else "conditional_mismatch",
            sources_unchanged=True,source_graph_at_end={"wrapper":execution.execution_source_sha256(),"gqa":execution.gqa.execution_source_sha256(),
                "hybrid":execution.hybrid.execution_source_sha256(),"cache":cache.execution_source_sha256(),"mixture":mixture.execution_source_sha256(),"recovery":sha(Path(__file__).with_name("recover_gqa_hybrid_qualification.py"))},
            elapsed_seconds=time.perf_counter()-start)
        if failed_rows:
            with (root/"failed-rows.npz").open("xb") as stream:np.savez_compressed(stream,**failed_rows)
            result["failed_rows_payload_sha256"]=sha(root/"failed-rows.npz")
        write_json(root/"manifest.json",result);return result
    except BaseException as error:
        result.update(status="failed",qualified=False,error_type=type(error).__name__,error=str(error),elapsed_seconds=time.perf_counter()-start)
        if active is not None:result["compatibility_adapter"]=dict(active)
        write_json(root/"manifest.json",result);raise
    finally:
        if engine is not None:engine.close()


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("banks","prefix-run","checkpoint","ledger","qualification","output"):parser.add_argument("--"+name,required=True)
    parser.add_argument("--max-seconds",type=int,default=2250);result=qualify(**vars(parser.parse_args()))
    print(json.dumps({k:result[k] for k in ("schema","status","qualified_batch_sizes","full21_batch_qualified","elapsed_seconds")}))
