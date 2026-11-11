#!/usr/bin/env python3
"""Budgeted fixed-conditional cache screen; no complete posterior sampling.

Built with PriorLabs-TabPFN. Import/help are CPU-safe. This diagnostic never
changes the original runner, arithmetic adapter, scientific gate, or saved data.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import random
import time
import numpy as np

SCHEMA = "posteriorforge.execution_cache_qualification.v1"
ATTRIBUTION = "Built with PriorLabs-TabPFN"
DEFAULT_STAGES = (0, 6, 14, 20) + tuple(i for i in range(21) if i not in (0, 6, 14, 20))
DEFAULT_ROWS = (1, 16, 128, 1024)
THRESHOLDS = {"normalization_absolute": 1e-5, "max_cdf_absolute": 2e-5,
              "max_row_total_variation": 1e-4, "quantile_coordinate_span_fraction": 2e-6}


def array_signature(value):
    array = np.ascontiguousarray(value)
    return {"shape": list(array.shape), "dtype": str(array.dtype),
            "sha256": hashlib.sha256(array.tobytes()).hexdigest()}


def compare_probabilities(reference, candidate, thresholds=THRESHOLDS):
    """Compare raw probability masses without silently repairing normalization."""
    a, b = np.asarray(reference), np.asarray(candidate)
    if a.shape != b.shape or a.ndim != 2 or not a.size:
        return {"passed": False, "reason": "probability shape mismatch", "reference_shape": list(a.shape), "candidate_shape": list(b.shape)}
    finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
    if not finite:
        return {"passed": False, "reason": "nonfinite probabilities"}
    a, b = a.astype(np.float64), b.astype(np.float64)
    normalization = float(max(np.abs(a.sum(-1)-1).max(), np.abs(b.sum(-1)-1).max()))
    minimum = float(min(a.min(), b.min()))
    difference = np.abs(a-b)
    cdf_error = float(np.abs(np.cumsum(a, axis=-1)-np.cumsum(b, axis=-1)).max())
    tv = float((difference.sum(-1)*.5).max())
    return {"passed": bool(minimum >= 0 and normalization <= thresholds["normalization_absolute"]
                           and cdf_error <= thresholds["max_cdf_absolute"] and tv <= thresholds["max_row_total_variation"]),
            "minimum_probability": minimum, "max_normalization_error": normalization,
            "max_probability_absolute_error": float(difference.max()),
            "max_cdf_absolute_error": cdf_error, "max_row_total_variation": tv}


def compare_log_probabilities(reference, candidate):
    """Record gauge-independent logit differences; CDF/TV decide numerical parity."""
    a, b = np.asarray(reference, np.float64), np.asarray(candidate, np.float64)
    if a.shape != b.shape: return {"shape_equal": False}
    if np.isnan(a).any() or np.isnan(b).any() or np.isposinf(a).any() or np.isposinf(b).any():
        return {"shape_equal": True, "invalid_logits": True}
    with np.errstate(invalid="ignore", divide="ignore"):
        a = a-np.logaddexp.reduce(a, axis=-1, keepdims=True)
        b = b-np.logaddexp.reduce(b, axis=-1, keepdims=True)
    finite = np.isfinite(a) & np.isfinite(b)
    return {"shape_equal": True, "negative_infinity_pattern_equal": bool(np.array_equal(np.isneginf(a), np.isneginf(b))),
            "max_centered_logit_error_on_common_finite_bins": float(np.abs(a[finite]-b[finite]).max()) if finite.any() else None}


def slice_prediction(prediction, rows):
    result = dict(prediction)
    result["logits"] = prediction["logits"][:rows]
    result["members"] = [{**member,
        "raw_logits": member["raw_logits"][:rows],
        "translated_probabilities": member["translated_probabilities"][:rows]}
        for member in prediction["members"]]
    return result


def _numpy(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def _same(value, other):
    return np.array_equal(_numpy(value), _numpy(other))


def compare_predictions(reference, candidate):
    """CPU comparisons of captured corrected predictions and every ensemble member."""
    a = reference["logits"].softmax(-1).detach().cpu().numpy()
    b = candidate["logits"].softmax(-1).detach().cpu().numpy()
    final = compare_probabilities(a, b)
    borders_equal = _same(reference["criterion"].borders, candidate["criterion"].borders)
    dtype_equal = str(reference["logits"].dtype) == str(candidate["logits"].dtype) == "torch.float32"
    members = []
    count_equal = len(reference["members"]) == len(candidate["members"]) == 8
    for index, (left, right) in enumerate(zip(reference["members"], candidate["members"])):
        metrics = compare_probabilities(_numpy(left["translated_probabilities"]), _numpy(right["translated_probabilities"]))
        identity_equal = left["configuration_identity"] == right["configuration_identity"]
        exact_borders = _same(left["from_borders"], right["from_borders"]) and _same(left["to_borders"], right["to_borders"])
        float32 = str(left["raw_logits"].dtype) == str(right["raw_logits"].dtype) == "torch.float32"
        metrics.update(index=index, configuration_identity_equal=bool(identity_equal), exact_borders=bool(exact_borders),
                       float32=float32, logits=compare_log_probabilities(_numpy(left["raw_logits"]), _numpy(right["raw_logits"])))
        metrics["passed"] = metrics["passed"] and identity_equal and exact_borders and float32
        members.append(metrics)
    inverse = inverse_cdf_probe(reference, candidate, final.get("max_cdf_absolute_error", 1.))
    return {"passed": bool(final["passed"] and borders_equal and dtype_equal and count_equal and all(m["passed"] for m in members) and inverse["passed"]),
            "exact_final_borders": bool(borders_equal), "final_float32": dtype_equal, "eight_members": count_equal,
            "final": final, "members": members, "inverse_cdf": inverse,
            "final_logits": compare_log_probabilities(_numpy(reference["logits"]), _numpy(candidate["logits"]))}


def inverse_cdf_probe(reference, candidate, cdf_delta):
    """Use original icdf at common uniforms, with probability-error quantile brackets.

    u=1 is stress only. Nonfinite original endpoint behavior is retained/reported,
    never clipped or replaced. This is not a new production sampler.
    """
    left, right = reference["criterion"], candidate["criterion"]
    a, b = reference["logits"], candidate["logits"]
    rows = sorted(set([0, len(a)//2, len(a)-1]))
    span = float((left.borders[-1]-left.borders[0]).item())
    allowance = max(abs(span)*THRESHOLDS["quantile_coordinate_span_fraction"], np.finfo(np.float32).eps)
    failures = []; errors = []; nonfinite = []; tested = 0; same_input_failures = 0
    for row in rows:
        probabilities = a[row].softmax(-1)
        knots = probabilities.cumsum(-1)
        selected = [int(len(knots)*fraction) for fraction in (.1, .5, .9)]
        uniforms = sorted(set([0., float(np.finfo(np.float32).eps), .1, .5, .9, 1.-float(np.finfo(np.float32).eps), 1., *[float(knots[index].item()) for index in selected]]))
        for u in uniforms:
            if not 0 <= u <= 1: continue
            ref = float(left.icdf(a[row], u).item()); value = float(right.icdf(b[row], u).item())
            identical = float(right.icdf(a[row], u).item())
            if not ((np.isnan(ref) and np.isnan(identical)) or ref == identical):same_input_failures += 1
            low = float(left.icdf(a[row], max(0.,u-cdf_delta)).item())
            high = float(left.icdf(a[row], min(1.,u+cdf_delta)).item())
            tested += 1
            if np.isfinite(ref) and np.isfinite(value):errors.append(abs(ref-value)/max(abs(span),1e-30))
            else:nonfinite.append({"row":row,"u":u,"reference":str(ref),"candidate":str(value)})
            if np.isfinite(ref) and not np.isfinite(value):failures.append({"row":row,"u":u,"reason":"candidate nonfinite for finite control"})
            elif np.isfinite(value) and np.isfinite(low) and np.isfinite(high) and not low-allowance <= value <= high+allowance:
                failures.append({"row":row,"u":u,"reason":"outside CDF-error quantile bracket"})
    return {"passed":not failures and same_input_failures==0,"tested_common_uniforms":tested,
            "same_logits_and_borders_exact_failures":same_input_failures,"bracket_failures":failures[:12],
            "max_normalized_common_uniform_value_difference":max(errors) if errors else None,
            "nonfinite_original_endpoint_behavior":nonfinite[:12],"u_one_is_stress_not_production":True,
            "cdf_delta_for_bracket":cdf_delta,"coordinate_allowance":allowance}


def rng_state(torch):
    return {"torch_cpu":torch.get_rng_state().clone(),"torch_cuda":[s.clone() for s in torch.cuda.get_rng_state_all()],
            "numpy":np.random.get_state(),"python":random.getstate()}


def restore_rng(torch, state):
    torch.set_rng_state(state["torch_cpu"]);torch.cuda.set_rng_state_all(state["torch_cuda"])
    np.random.set_state(state["numpy"]);random.setstate(state["python"])


def rng_identity(state):
    return {key:hashlib.sha256(pickle.dumps(value,protocol=5) if key in ("numpy","python") else
            b"".join(s.cpu().numpy().tobytes() for s in value) if isinstance(value,list) else value.cpu().numpy().tobytes()).hexdigest()
            for key,value in state.items()}


def model_contract(regressor):
    roots = getattr(regressor.executor_,"models",None) or [regressor.model_]
    modules=[]
    scalar_fields=("normalize_on_train_only","normalize_x","normalize_to_ranking","normalize_by_used_features",
                   "remove_outliers","remove_outliers_sigma","cache_trainset_representation","feature_positional_embedding",
                   "keep_nans","num_features","normalize_by_sqrt","features_per_group","seed")
    tensor_fields=("cached_embeddings","mean_for_normalization","std_for_normalization","lower_for_outlier_removal","upper_for_outlier_removal","number_of_used_features_","feature_means_","sel")
    for member,root in enumerate(roots):
        for name,module in root.named_modules():
            values={key:getattr(module,key) for key in scalar_fields if isinstance(getattr(module,key,None),(bool,int,float,str))}
            tensors={}
            for key in tensor_fields:
                tensor=getattr(module,key,None)
                if hasattr(tensor,"numel"):
                    item={"shape":list(tensor.shape),"dtype":str(tensor.dtype),"device":str(tensor.device),"bytes":tensor.numel()*tensor.element_size()}
                    if tensor.numel()<=4096:item["sha256"]=hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
                    if tensor.numel()<=64:item["values"]=[float(v) if np.isfinite(v) else str(v) for v in tensor.detach().cpu().numpy().reshape(-1)]
                    tensors[key]=item
            if values or tensors:modules.append({"member":member,"module":name,"type":type(module).__name__,"settings":values,"tensors":tensors})
    normalizers=[row for row in modules if row["type"]=="InputNormalizationEncoderStep"]
    return {"modules":modules,"input_normalization_train_only_verified":bool(normalizers) and all(row["settings"].get("normalize_on_train_only") is True for row in normalizers),
            "preprocessor_sha256":[hashlib.sha256(pickle.dumps(p,protocol=5)).hexdigest() for p in regressor.executor_.preprocessors],
            "autocast":bool(regressor.use_autocast_),"forced_inference_dtype":str(regressor.forced_inference_dtype_)}


def fixed_queries(obs, prefixes, theta_context, stage, noise_indices, bounds):
    main=np.concatenate((np.repeat(obs,len(prefixes),axis=0),prefixes[:,:stage]),axis=1).astype(np.float32)
    stress=np.concatenate((np.repeat(obs,8,axis=0),theta_context[np.linspace(0,len(theta_context)-1,8,dtype=int),:stage]),axis=1).astype(np.float32)
    if stage:
        eligible=[i for i in noise_indices if i<stage]
        if eligible:
            column=33+eligible[0];low,high=bounds;epsilon=(high-low)*1e-4
            stress[4:,column]=[low,high,low-epsilon,high+epsilon]
        else:
            stress[4,33:]=0;stress[5,33]=np.nan;stress[6,33:]=np.nan
            stress[7,33:]=theta_context[:,:stage].mean(0)+10*theta_context[:,:stage].std(0)
    return main,stress


def _trace_summary(path):
    """Reuse the already-tested bounded-memory reader; no key_averages retention."""
    helper=Path(__file__).with_name("diagnose_runtime.py")
    spec=importlib.util.spec_from_file_location("_cache_trace_reader",helper);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.summarize_trace(path)


def profile_prediction(call, output, name, torch):
    # One small prediction only: no criterion.sample, shapes, stacks or memory events.
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
                                record_shapes=False,profile_memory=False,with_stack=False) as prof:
        with torch.profiler.record_function("pfdiagnostic/profile/request"):
            with torch.profiler.record_function("execution_cache/"+name):call()
            torch.cuda.synchronize()
    trace_path=output/(name+".trace.json");prof.export_chrome_trace(str(trace_path));del prof
    summary=_trace_summary(trace_path)
    from posteriorforge.artifacts import sha256_file,write_json
    write_json(output/(name+"-trace-summary.json"),summary)
    compressed=trace_path.with_suffix(trace_path.suffix+".gz")
    with trace_path.open("rb") as source,gzip.open(compressed,"wb",compresslevel=5) as target:
        while block:=source.read(1024*1024):target.write(block)
    trace_path.unlink()
    return {"trace_file":compressed.name,"trace_sha256":sha256_file(compressed),"summary":summary,
            "profile_memory":False,"record_shapes":False,"sampler_calls":0,"query_rows":128}


def qualify(*,banks,prefix_run,checkpoint,ledger,output,stages=DEFAULT_STAGES,row_counts=DEFAULT_ROWS,max_seconds=1050,profile=True):
    from posteriorforge.artifacts import runtime_identity,sha256_file,write_json
    from posteriorforge.reference_grid import _require_budgeted_gpu
    if tuple(row_counts)!=DEFAULT_ROWS or any(stage not in range(21) for stage in stages) or len(set(stages))!=len(stages):
        raise ValueError("declared fixed stages and row counts1/16/128/1024 required")
    if not 60<=max_seconds<=1050:raise ValueError("qualification deadline must be60..1050seconds")
    budget=_require_budgeted_gpu(ledger)
    if budget["gpus"]!=1:raise RuntimeError("one reserved GPU required")
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();deadline=started+max_seconds
    result={**runtime_identity(),"schema":SCHEMA,"attribution":ATTRIBUTION,"status":"running","qualified":False,"full21_qualified":False,
        "control_variant":"cdf_arithmetic_v1","scope":"fixed_conditional_screen_not_posterior_equivalence",
        "requested_stages":list(stages),"stage_coverage":[],"row_counts":list(row_counts),"thresholds":dict(THRESHOLDS),
        "budget_binding":budget,"internal_deadline_seconds":max_seconds,"stage_results":[],"posterior_vectors_generated":0,
        "scientific_gate_evaluated":False,"rng_stream_bitwise_equivalence_required":False,
        "limitations":["A finite fixed-prefix screen is not a joint posterior-law proof or physical accuracy assessment.","Initial accepted prefixes omit some intermediate proposal states; a separate tiny stress block is diagnostic only.","Different model-forward counts and LayerStack RNG consumption can change final RNG streams; conditional law and common uniforms are compared separately.","Captured member CPU copies synchronize; their timings are excluded from capture-free performance measurements."]}
    write_json(output/"manifest.json",result)
    active=None;engine=None
    try:
        import torch
        from tabpfn import TabPFNRegressor
        from posteriorforge.inference import CHECKPOINT_SHA256,frozen_regressor_settings,support_mask
        from posteriorforge.inference_cdf import guard_baseline_source
        from posteriorforge.compat_cdf import finite_cdf_compatibility
        from posteriorforge.execution_cache import ResidentStageExecutor,capture_corrected_prediction,ENGINE_ID
        from posteriorforge.banks import derive_seed,load_context,load_heldout
        from posteriorforge.gate import verify_run_binding
        from posteriorforge.science.analytical_benchmark_v2 import array_descriptor,build_problem
        import posteriorforge.execution_cache as cache_module
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError("one visible CUDA GPU required")
        if torch.get_float32_matmul_precision()!="highest" or torch.backends.cuda.matmul.allow_tf32:raise RuntimeError("FP32 highest/TF32False required")
        source=guard_baseline_source()
        if sha256_file(checkpoint)!=CHECKPOINT_SHA256:raise ValueError("pinned checkpoint required")
        prefix_run=Path(prefix_run);prefix_manifest=json.loads((prefix_run/"manifest.json").read_text())
        if prefix_manifest.get("status")!="complete" or prefix_manifest.get("variant")!="cdf_arithmetic_v1":raise ValueError("authenticated completed corrected-pilot prefixes required")
        if prefix_manifest["case"]!=0 or prefix_manifest["context_seed"]!=20261111:raise ValueError("fixed case0/seed20261111 required")
        context=load_context(banks,20261111,4096);heldout=load_heldout(banks,0);obs=heldout.obs.reshape(1,33)
        verify_run_binding(prefix_manifest,context,heldout.obs,20261111,0)
        if sha256_file(prefix_run/"samples.npz")!=prefix_manifest["samples_sha256"]:raise ValueError("prefix payload hash mismatch")
        with np.load(prefix_run/"samples.npz",allow_pickle=False) as saved:values=saved["samples"].copy()
        if values.shape!=(50000,21) or not support_mask(values).all():raise ValueError("complete accepted prefix source required")
        positions=np.linspace(0,len(values)-1,max(row_counts),dtype=int);prefixes=values[positions]
        theta=context.theta.astype(np.float32);observations=context.obs.astype(np.float32);joint=np.concatenate((observations,theta),axis=1)
        problem=build_problem();noise_indices=[problem.parameter_order.index(n) for n in problem.prior.noise_names]
        seed=derive_seed(20261111,"execution-cache-qualification",case=0);np.random.seed(seed);random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS","16")))
        result.update(engine_id=ENGINE_ID,module_sha256=sha256_file(cache_module.__file__),
            checkpoint_sha256=CHECKPOINT_SHA256,scientific_fingerprint=problem.scientific_fingerprint,parameter_order=list(problem.parameter_order),
            context_theta=array_descriptor(context.theta),context_obs=array_descriptor(context.obs),query=array_descriptor(obs),
            prefix_manifest_sha256=sha256_file(prefix_run/"manifest.json"),prefix_samples_sha256=prefix_manifest["samples_sha256"],
            fixed_prefix_row_indices=positions.tolist(),seed=seed,device=torch.cuda.get_device_name(),
            sources={**source,"execution_cache_module_sha256":sha256_file(cache_module.__file__),"qualification_script_sha256":sha256_file(__file__)})
        kwargs=frozen_regressor_settings(checkpoint);kwargs["inference_precision"]=torch.float32
        with finite_cdf_compatibility() as active:
            for stage in stages:
                if time.perf_counter()+90>=deadline:result["status"]="incomplete";break
                result["active_stage"]=stage;result["elapsed_seconds"]=time.perf_counter()-started;write_json(output/"manifest.json",result)
                cdf_calls_initial=active["patched_call_count"]
                baseline=TabPFNRegressor(**kwargs);initial_rng=rng_state(torch)
                torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t0=time.perf_counter()
                baseline.fit(joint[:,:33+stage],joint[:,33+stage]);torch.cuda.synchronize();fit_seconds=time.perf_counter()-t0
                after_control_fit_rng=rng_identity(rng_state(torch))
                queries,stress=fixed_queries(obs,prefixes,theta,stage,noise_indices,problem.prior.bounds)
                controls={};rng_rows=[]
                baseline_contract=model_contract(baseline)
                for count in row_counts:
                    before=rng_identity(rng_state(torch));controls[count]=capture_corrected_prediction(baseline,queries[:count],cdf_descriptor=active)
                    rng_rows.append({"path":"control","query_rows":count,"before":before,"after":rng_identity(rng_state(torch))})
                control_stress=capture_corrected_prediction(baseline,stress,cdf_descriptor=active)
                torch.cuda.synchronize();t0=time.perf_counter();baseline.predict(queries[:max(row_counts)],output_type="full",quantiles=[]);torch.cuda.synchronize()
                baseline_predict_seconds=time.perf_counter()-t0
                baseline_after=model_contract(baseline)
                profile_this=profile and stage==6
                if profile_this:
                    result.setdefault("profiles",{})["control-stage6-predict128"]=profile_prediction(lambda:baseline.predict(queries[:128],output_type="full",quantiles=[]),output,"control-stage6-predict128",torch)
                cdf_calls_after_control=active["patched_call_count"]
                # Release baseline GPU allocations before resident preparation; CPU captures remain authoritative.
                baseline.model_.cpu();restore_rng(torch,initial_rng)
                engine=ResidentStageExecutor(checkpoint,device="cuda",query_batch_size=max(row_counts),cdf_descriptor=active)
                torch.cuda.synchronize();t0=time.perf_counter()
                prepared_metadata=engine.prepare(theta,observations,stage);torch.cuda.synchronize();prepare_seconds=time.perf_counter()-t0
                after_cache_prepare_rng=rng_identity(rng_state(torch))
                cache_before=engine.inspect_cache();cached_contract=model_contract(engine.regressor)
                if profile_this:
                    result.setdefault("profiles",{})["resident-stage6-predict128"]=profile_prediction(lambda:engine.predict(queries[:128],output_type="full",capture_members=False),output,"resident-stage6-predict128",torch)
                comparisons=[]
                for count in row_counts:
                    before=rng_identity(rng_state(torch));candidate=engine.predict(queries[:count],output_type="full",capture_members=True)
                    rng_rows.append({"path":"cache","query_rows":count,"before":before,"after":rng_identity(rng_state(torch))})
                    comparisons.append({"query_rows":count,"cache_vs_same_chunk_control":compare_predictions(controls[count],candidate),
                        "control_chunk_vs_1024":compare_predictions(slice_prediction(controls[max(row_counts)],count),controls[count])})
                candidate_stress=engine.predict(stress,output_type="full",capture_members=True)
                stress_comparison=compare_predictions(control_stress,candidate_stress)
                cache_after=engine.inspect_cache();cached_after=model_contract(engine.regressor)
                torch.cuda.synchronize();t0=time.perf_counter();engine.predict(queries,output_type="full",capture_members=False);torch.cuda.synchronize()
                predict_seconds=time.perf_counter()-t0
                row={"stage_index":stage,"parameter":problem.parameter_order[stage],"query_shape":list(queries.shape),"comparisons":comparisons,
                    "stress_comparison":stress_comparison,"stress_query":array_signature(stress),"stress_is_not_posterior_sampling":True,
                    "baseline_model_contract_before":baseline_contract,"baseline_model_contract_after":baseline_after,
                    "cache_model_contract_before":cached_contract,"cache_model_contract_after":cached_after,
                    "preprocessor_pickle_identity":baseline_contract["preprocessor_sha256"]==cached_contract["preprocessor_sha256"],
                    "cache_before":cache_before,"cache_after":cache_after,"engine_prepared":prepared_metadata,"engine_counters":{key:value for key,value in engine.metadata.items() if key!="stages"},"rng":rng_rows,
                    "cdf_call_deltas":{"control":cdf_calls_after_control-cdf_calls_initial,"cache":active["patched_call_count"]-cdf_calls_after_control},
                    "rng_initial_fit_state":rng_identity(initial_rng),"rng_after_control_fit":after_control_fit_rng,"rng_after_cache_prepare":after_cache_prepare_rng,
                    "timings_seconds":{"control_fit":fit_seconds,"cache_prepare_capture_free":prepare_seconds,
                        "control_predict1024_capture_free":baseline_predict_seconds,"cache_predict1024_capture_free":predict_seconds},
                    "peak_allocated_bytes":torch.cuda.max_memory_allocated(),"peak_reserved_bytes":torch.cuda.max_memory_reserved(),
                    "cache_prepare_profiled":False,"predict128_profiled_separately":profile_this}
                row["passed"]=all(c["cache_vs_same_chunk_control"]["passed"] and c["control_chunk_vs_1024"]["passed"] for c in comparisons) and stress_comparison["passed"] and baseline_after["input_normalization_train_only_verified"] and cached_after["input_normalization_train_only_verified"]
                result["stage_results"].append(row);result["stage_coverage"].append(stage);engine.close();engine=None
                result["elapsed_seconds"]=time.perf_counter()-started;write_json(output/"manifest.json",result)
                del baseline,controls,control_stress,candidate,candidate_stress
                if not row["passed"]:result["status"]="conditional_mismatch";break
            else:result["status"]="screen_pass"
        result["compatibility_adapter"]=dict(active)
        result["qualified"]=result["status"]=="screen_pass" and len(result["stage_coverage"])==len(stages)
        result["full21_qualified"]=result["qualified"] and set(result["stage_coverage"])==set(range(21))
        if guard_baseline_source()!=source:raise RuntimeError("original sources changed")
        if sha256_file(cache_module.__file__)!=result["module_sha256"] or sha256_file(__file__)!=result["sources"]["qualification_script_sha256"]:raise RuntimeError("qualification or engine source changed")
        result["sources_unchanged"]=True
    except BaseException as error:
        result.update(status="failed",qualified=False,full21_qualified=False,error_type=type(error).__name__,error=str(error))
        if active is not None:result["compatibility_adapter"]=dict(active)
        raise
    finally:
        if engine is not None:engine.close()
        result["elapsed_seconds"]=time.perf_counter()-started;write_json(output/"manifest.json",result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("banks","prefix-run","checkpoint","ledger","output"):parser.add_argument("--"+name,required=True)
    parser.add_argument("--stages",default=",".join(map(str,DEFAULT_STAGES)))
    parser.add_argument("--max-seconds",default=1050,type=int)
    parser.add_argument("--no-profile",action="store_true")
    args=vars(parser.parse_args());args["stages"]=tuple(int(value) for value in args["stages"].split(","));args["profile"]=not args.pop("no_profile")
    result=qualify(**args);print(json.dumps({key:result.get(key) for key in ("schema","status","qualified","full21_qualified","stage_coverage","elapsed_seconds")},indent=2,allow_nan=False))
    if not result["qualified"]:raise SystemExit(2)


if __name__=="__main__":main()
