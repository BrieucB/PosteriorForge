#!/usr/bin/env python3
"""Bounded GQA layout diagnostic on stages6/10/13; no posterior sampling.

Built with PriorLabs-TabPFN. Reuses the EXACT same prepared cache for ordinary
and expanded query predictions. This is not a full21 or physical accuracy pass.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time
import numpy as np

from posteriorforge.artifacts import runtime_identity,sha256_file,write_json
from posteriorforge import execution_cache as cache,execution_gqa as gqa

UTILITIES_SHA256="6c927a5c278c7fc54b556ac557ecef64f5e6338f5ae13bf466f3b1e6a4640f90"
DEFAULT_STAGES=(6,10,13)


def load_utilities():
    path=Path(__file__).with_name("qualify_execution_cache.py")
    if sha256_file(path)!=UTILITIES_SHA256:raise RuntimeError("frozen comparison helper source mismatch")
    spec=importlib.util.spec_from_file_location("_gqa_comparisons",path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def validate_plan(stages,row_counts,max_seconds):
    if not stages or len(set(stages))!=len(stages) or any(stage not in DEFAULT_STAGES for stage in stages):
        raise ValueError("unique bounded stages6/10/13 only")
    if tuple(row_counts) not in ((16,128),(16,128,1024)) or not 60<=max_seconds<=900:
        raise ValueError("fixed rows16/128 optionally1024 and60..900seconds required")


def tensor_signature(tensor):
    array=tensor.detach().cpu().contiguous().numpy()
    return {"shape":list(array.shape),"dtype":str(array.dtype),
        "sha256":hashlib.sha256(array.tobytes()).hexdigest()}


def prepared_state(engine):
    records=[];embeddings=[]
    for member,model in enumerate(engine.regressor.executor_.models):
        tensor=getattr(model,"cached_embeddings",None)
        if tensor is not None:embeddings.append({"member":member,**tensor_signature(tensor)})
        for category,name,value in cache._tensor_records(model,engine._torch):
            if value.numel():
                records.append({"member":member,"category":category,"name":name,"shape":list(value.shape),
                    "stride":list(value.stride()),"device":str(value.device),"dtype":str(value.dtype),
                    "storage_pointer":value.untyped_storage().data_ptr(),
                    "storage_bytes":value.untyped_storage().nbytes()})
    return {"tensor_storage_records":records,"all_cached_embedding_hashes":embeddings}


def observed_prediction(call,*,expanded):
    with gqa.expanded_kv_attention(enabled=expanded) as descriptor:
        prediction=call()
    if not descriptor["original_alias_restored"] or not descriptor["sdk_sources_unchanged"]:
        raise RuntimeError("attention observer/expansion failed restoration")
    if expanded and descriptor["expanded_attention_calls"]<=0:raise RuntimeError("requested GQA path did not execute")
    return prediction,descriptor


def capture_compact(captures,stage,path,prediction,control,positions):
    """Keep first/last/worst-CDF rows; full logits stay outside Git."""
    p=prediction["logits"].softmax(-1).detach().cpu().numpy()
    reference=control["logits"].softmax(-1).detach().cpu().numpy()
    delta=np.abs(np.cumsum(p.astype(np.float64)-reference.astype(np.float64),axis=-1)).max(-1)
    selected=np.asarray(sorted({0,len(p)-1,int(np.argmax(delta))}),dtype=int)
    prefix=f"stage{stage}_{path}"
    captures[prefix+"_original_sample_rows"]=positions[selected]
    captures[prefix+"_logits"]=prediction["logits"][selected].detach().cpu().numpy()
    captures[prefix+"_physical_borders"]=prediction["criterion"].borders.detach().cpu().numpy()
    for member,value in enumerate(prediction["members"]):
        captures[f"{prefix}_member{member}_raw_logits"]=value["raw_logits"][selected].numpy()
        captures[f"{prefix}_member{member}_from_borders"]=value["from_borders"].numpy()
        captures[f"{prefix}_member{member}_translated_probabilities"]=value["translated_probabilities"][selected].numpy()


def profile_path(call,*,expanded,output,name,torch,utilities):
    with gqa.expanded_kv_attention(enabled=expanded) as descriptor:
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        before=torch.cuda.memory_allocated()
        profile=utilities.profile_prediction(call,output,name,torch)
        peak=torch.cuda.max_memory_allocated();reserved=torch.cuda.max_memory_reserved()
    return {**profile,"attention_descriptor":descriptor,"allocated_before":before,
        "peak_allocated_bytes":peak,"peak_working_bytes_above_before":peak-before,
        "peak_reserved_bytes":reserved,"includes_passive_shape_observer":True}


def run(*,banks,prefix_run,checkpoint,ledger,output,stages=DEFAULT_STAGES,
        row_counts=(16,128),max_seconds=900,profile=True):
    validate_plan(stages,row_counts,max_seconds)
    from posteriorforge.reference_grid import _require_budgeted_gpu
    budget=_require_budgeted_gpu(ledger)
    if budget["gpus"]!=1:raise RuntimeError("one registered GPU allocation required")
    import torch
    from tabpfn import TabPFNRegressor
    from posteriorforge.inference import CHECKPOINT_SHA256,frozen_regressor_settings,support_mask
    from posteriorforge.inference_cdf import guard_baseline_source
    from posteriorforge.compat_cdf import finite_cdf_compatibility
    from posteriorforge.banks import load_context,load_heldout,derive_seed
    from posteriorforge.gate import verify_run_binding
    from posteriorforge.science.analytical_benchmark_v2 import build_problem,array_descriptor
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError("one visible CUDA device required")
    if torch.get_default_dtype()!=torch.float32 or torch.get_float32_matmul_precision()!="highest" or torch.backends.cuda.matmul.allow_tf32:
        raise RuntimeError("FP32 highest/TF32False required")
    utilities=load_utilities();source=guard_baseline_source();gqa_sha=gqa.execution_source_sha256()
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter();deadline=start+max_seconds;engine=None;active=None;captures={}
    result={**runtime_identity(),"schema":"posteriorforge.gqa_expansion_probe.v1","status":"running",
        "attribution":"Built with PriorLabs-TabPFN","budget_binding":budget,"diagnostic_only":True,
        "posterior_vectors_generated":0,"full21_qualified":False,"scientific_gate_evaluated":False,
        "scope":"bounded fixed-prefix query-layout/backend experiment; no posterior or full21 claim",
        "requested_stages":list(stages),"row_counts":list(row_counts),"internal_deadline_seconds":max_seconds,"thresholds":utilities.THRESHOLDS,
        "gqa_module_sha256":gqa_sha,"cache_module_sha256":cache.execution_source_sha256(),
        "probe_source_sha256":sha256_file(__file__),"comparison_helper_sha256":UTILITIES_SHA256,
        "prefill_projection_changed":False,"same_prepared_cache_reused":True,
        "profile_scope":"one capture-free predict128 per path/stage; no fit/sampler/memory-event tracing",
        "stage_results":[],"limitations":["Finite accepted-prefix diagnostic does not cover all proposal prefixes.",
           "Ordinary and expanded paths share exact prepared caches; cache prefill arithmetic is not changed.",
           "All profile paths retain the same resident stage cache; incremental query peak is reported separately.",
           "Full21 golden proof and genuine50000-vector E2E would be separate steps."]}
    write_json(root/"manifest.json",result)
    try:
        if sha256_file(checkpoint)!=CHECKPOINT_SHA256:raise ValueError("immutable checkpoint required")
        context=load_context(banks,20261111,4096);heldout=load_heldout(banks,0);obs=heldout.obs.reshape(1,33)
        prefix_run=Path(prefix_run);manifest=json.loads((prefix_run/"manifest.json").read_text())
        if manifest.get("variant")!="cdf_arithmetic_v1" or manifest.get("status")!="complete":
            raise ValueError("complete corrected-control accepted prefixes required")
        verify_run_binding(manifest,context,heldout.obs,20261111,0)
        if sha256_file(prefix_run/"samples.npz")!=manifest["samples_sha256"]:raise ValueError("prefix payload hash mismatch")
        with np.load(prefix_run/"samples.npz",allow_pickle=False) as data:samples=data["samples"].copy()
        if samples.shape!=(50000,21) or not support_mask(samples).all():raise ValueError("complete finite supported saved prefixes required")
        positions=np.linspace(0,len(samples)-1,1024,dtype=int);prefixes=samples[positions]
        problem=build_problem();theta=context.theta.astype(np.float32);x=context.obs.astype(np.float32)
        joint=np.concatenate((x,theta),axis=1);noise=[problem.parameter_order.index(name) for name in problem.prior.noise_names]
        seed=derive_seed(20261111,"gqa-layout-probe",case=0);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS","16")))
        result.update(checkpoint_sha256=CHECKPOINT_SHA256,scientific_fingerprint=problem.scientific_fingerprint,
            parameter_order=list(problem.parameter_order),context_theta=array_descriptor(context.theta),
            context_obs=array_descriptor(context.obs),query=array_descriptor(obs),
            prefix_manifest_sha256=sha256_file(prefix_run/"manifest.json"),prefix_samples_sha256=manifest["samples_sha256"],
            fixed_prefix_row_indices=positions.tolist(),seed=seed,device=torch.cuda.get_device_name(),original_sources=source)
        kwargs=frozen_regressor_settings(checkpoint);kwargs["inference_precision"]=torch.float32
        with finite_cdf_compatibility() as active:
            for stage in stages:
                if time.perf_counter()+120>=deadline:result["status"]="bounded_incomplete";break
                result["active_stage"]=stage;write_json(root/"manifest.json",result)
                baseline=TabPFNRegressor(**kwargs);torch.cuda.synchronize();t0=time.perf_counter()
                baseline.fit(joint[:,:33+stage],joint[:,33+stage]);torch.cuda.synchronize()
                control_fit=time.perf_counter()-t0
                queries,stress=utilities.fixed_queries(obs,prefixes,theta,stage,noise,problem.prior.bounds)
                controls={};control_observers={}
                for count in row_counts:
                    controls[count],control_observers[str(count)]=observed_prediction(
                        lambda count=count:cache.capture_corrected_prediction(baseline,queries[:count],cdf_descriptor=active),expanded=False)
                stress_control,_=observed_prediction(lambda:cache.capture_corrected_prediction(baseline,stress,cdf_descriptor=active),expanded=False)
                baseline.model_.cpu()
                engine=cache.ResidentStageExecutor(checkpoint,cdf_descriptor=active,query_batch_size=max(row_counts))
                prepared=engine.prepare(theta,x,stage);before=prepared_state(engine)
                row={"stage_index":stage,"parameter":problem.parameter_order[stage],"control_fit_seconds":control_fit,
                     "prepared":prepared,"comparisons":[],"control_attention_observers":control_observers}
                ordinary={};expanded={}
                for count in row_counts:
                    ordinary[count],ordinary_observer=observed_prediction(
                        lambda count=count:engine.predict(queries[:count],capture_members=True),expanded=False)
                    expanded[count],expanded_observer=observed_prediction(
                        lambda count=count:engine.predict(queries[:count],capture_members=True),expanded=True)
                    row["comparisons"].append({"query_rows":count,
                        "ordinary_vs_control":utilities.compare_predictions(controls[count],ordinary[count]),
                        "expanded_vs_control":utilities.compare_predictions(controls[count],expanded[count]),
                        "expanded_vs_same_cache_ordinary":utilities.compare_predictions(ordinary[count],expanded[count]),
                        "ordinary_attention_observer":ordinary_observer,"expanded_attention_descriptor":expanded_observer})
                ordinary_stress,_=observed_prediction(lambda:engine.predict(stress,capture_members=True),expanded=False)
                expanded_stress,stress_descriptor=observed_prediction(lambda:engine.predict(stress,capture_members=True),expanded=True)
                row["stress_comparisons"]={"ordinary_vs_control":utilities.compare_predictions(stress_control,ordinary_stress),
                    "expanded_vs_control":utilities.compare_predictions(stress_control,expanded_stress),
                    "expanded_attention_descriptor":stress_descriptor}
                after=prepared_state(engine)
                row["cache_tensor_storage_and_embeddings_unchanged"]=before==after
                row["cached_embedding_hashes_before"]=before["all_cached_embedding_hashes"]
                row["cached_embedding_hashes_after"]=after["all_cached_embedding_hashes"]
                if not row["cache_tensor_storage_and_embeddings_unchanged"]:raise RuntimeError("query layout experiment mutated prepared cache state")
                if profile:
                    row["profiles"]={
                        "uncached":profile_path(lambda:baseline.predict(queries[:128],output_type="full",quantiles=[]),
                            expanded=False,output=root,name=f"stage-{stage}-uncached128",torch=torch,utilities=utilities),
                        "ordinary":profile_path(lambda:engine.predict(queries[:128],capture_members=False),
                            expanded=False,output=root,name=f"stage-{stage}-ordinary128",torch=torch,utilities=utilities),
                        "expanded":profile_path(lambda:engine.predict(queries[:128],capture_members=False),
                            expanded=True,output=root,name=f"stage-{stage}-expanded128",torch=torch,utilities=utilities)}
                row["cache_state_unchanged_after_profiles"]=before==prepared_state(engine)
                if not row["cache_state_unchanged_after_profiles"]:raise RuntimeError("profiling mutated prepared cache state")
                largest=max(row_counts)
                for name,prediction in (("control",controls[largest]),("ordinary",ordinary[largest]),("expanded",expanded[largest])):
                    capture_compact(captures,stage,name,prediction,controls[largest],positions[:largest])
                row["ordinary_within_fixed_screen"]=all(item["ordinary_vs_control"]["passed"] for item in row["comparisons"]) and row["stress_comparisons"]["ordinary_vs_control"]["passed"]
                row["expanded_within_fixed_screen"]=all(item["expanded_vs_control"]["passed"] for item in row["comparisons"]) and row["stress_comparisons"]["expanded_vs_control"]["passed"]
                result["stage_results"].append(row);engine.close();engine=None
                write_json(root/"manifest.json",result)
                del baseline,controls,ordinary,expanded,stress_control,ordinary_stress,expanded_stress
            else:result["status"]="bounded_diagnostic_complete"
        result["compatibility_adapter"]=dict(active)
        if not active["original_alias_restored"] or not active["sdk_sources_unchanged"]:raise RuntimeError("CDF adapter restoration failed")
        if gqa.execution_source_sha256()!=gqa_sha or sha256_file(__file__)!=result["probe_source_sha256"] or guard_baseline_source()!=source:
            raise RuntimeError("diagnostic source changed during probe")
        with (root/"compact-captures.npz").open("xb") as stream:np.savez_compressed(stream,**captures)
        result.update(capture_sha256=sha256_file(root/"compact-captures.npz"),
            completed_stages=[row["stage_index"] for row in result["stage_results"]],
            all_requested_expanded_conditions_pass=bool(len(result["stage_results"])==len(stages) and all(row["expanded_within_fixed_screen"] for row in result["stage_results"])),
            sources_unchanged=True,elapsed_seconds=time.perf_counter()-start)
        write_json(root/"manifest.json",result);return result
    except BaseException as error:
        result.update(status="failed_diagnostic",error_type=type(error).__name__,error=str(error),elapsed_seconds=time.perf_counter()-start)
        if active is not None:result["compatibility_adapter"]=dict(active)
        write_json(root/"manifest.json",result);raise
    finally:
        if engine is not None:engine.close()


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("banks","prefix-run","checkpoint","ledger","output"):parser.add_argument("--"+name,required=True)
    parser.add_argument("--stages",default="6,10,13");parser.add_argument("--include-1024",action="store_true")
    parser.add_argument("--max-seconds",type=int,default=900);parser.add_argument("--no-profile",action="store_true")
    args=vars(parser.parse_args());args["stages"]=tuple(int(value) for value in args["stages"].split(","))
    args["row_counts"]=(16,128,1024) if args.pop("include_1024") else (16,128);args["profile"]=not args.pop("no_profile")
    result=run(**args);print(json.dumps({key:result.get(key) for key in ("schema","status","completed_stages","all_requested_expanded_conditions_pass","elapsed_seconds")}))
