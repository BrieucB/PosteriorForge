#!/usr/bin/env python3
"""Bounded approximate precision screen, stages6/10/13; no posterior generation.

Built with PriorLabs-TabPFN. One separately named policy per allocation, strict
unchanged FP32 CDF/TV/normalization thresholds, and early rejection on failure.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import importlib.util
import json
import os
import resource
import random
import sys
from pathlib import Path
import time
import numpy as np

SCHEMA="posteriorforge.precision_conditional_probe.v1"
UTILITY_SHA="6c927a5c278c7fc54b556ac557ecef64f5e6338f5ae13bf466f3b1e6a4640f90"
RECOVERY_SHA="83064f54712014b5c4c66569e854872caff03593cfdc337f2472a7b5723aea02"
STAGES=(6,10,13)
ROWS=(128,1024)
THRESHOLDS={"max_cdf_absolute":2e-5,"max_row_total_variation":1e-4,
            "normalization_absolute":1e-5,"quantile_coordinate_span_fraction":2e-6}


def require(value,message):
    if not value:raise RuntimeError(message)


def load_helper(name,expected):
    path=Path(__file__).with_name(name)
    require(hashlib.sha256(path.read_bytes()).hexdigest()==expected,"pinned helper changed: "+name)
    spec=importlib.util.spec_from_file_location("_precision_"+path.stem,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def exact_prediction(left,right,torch):
    def same(a,b):
        return tuple(a.shape)==tuple(b.shape) and a.dtype==b.dtype and torch.equal(
            a.detach().cpu().contiguous().view(torch.uint8),b.detach().cpu().contiguous().view(torch.uint8))
    return same(left["logits"],right["logits"]) and len(left["members"])==len(right["members"])==8 and all(
        same(a[key],b[key]) for a,b in zip(left["members"],right["members"])
        for key in ("raw_logits","translated_probabilities","from_borders","to_borders"))


def comparison_arms(u0,up,cp,g0,utility):
    return {"uncached_FP32_to_uncached_precision":utility.compare_predictions(u0,up),
            "uncached_precision_to_cached_precision_and_declared_views":utility.compare_predictions(up,cp),
            "qualified_FP32_GQA_hybrid_to_cached_precision":utility.compare_predictions(g0,cp),
            "uncached_FP32_to_qualified_FP32_GQA_hybrid":utility.compare_predictions(u0,g0)}


def reconstruction_checks(components,reference,utility):
    raw=np.stack([utility._numpy(member["translated_probabilities"]) for member in components["members"]]).astype(np.float64)
    masses=raw.sum(-1);weights=masses/masses.sum(0,keepdims=True)
    ideal=raw.mean(0);ideal/=ideal.sum(-1,keepdims=True)
    actual=np.stack([utility._numpy(prediction["logits"].softmax(-1)) for prediction in components["component_predictions"]]).astype(np.float64)
    target=utility._numpy(reference["logits"].softmax(-1))
    return {"normalized_actual_raw_mean_vs_reference":utility.compare_probabilities(ideal,target),
            "mass_corrected_actual_component_outputs_vs_reference":utility.compare_probabilities((actual*weights[:,:,None]).sum(0),target),
            "scope":"direct probability comparisons, not sums of separate CDF error bounds"}


def bound_sources(qualification,context,query,checkpoint):
    from posteriorforge.artifacts import sha256_file
    from posteriorforge.science.analytical_benchmark_v2 import build_problem,array_descriptor
    from posteriorforge import execution_gqa_hybrid as gqa
    recovery=load_helper("recover_gqa_hybrid_qualification.py",RECOVERY_SHA)
    q=recovery.verify_recovery_artifact(qualification)
    expected={"status":"gqa_hybrid_screen_pass","full21_gqa_hybrid_qualified":True,
              "full21_component_qualified":True,"full21_sampler_qualified":True,"full21_qualified":False,
              "cache_only_full21_qualified":False,"thresholds":THRESHOLDS,"fallback_map":{"10":[5]},
              "module_sha256":gqa.execution_source_sha256(),"checkpoint_sha256":sha256_file(checkpoint),
              "context_theta":array_descriptor(context.theta),"context_obs":array_descriptor(context.obs),
              "query":array_descriptor(query),"scientific_fingerprint":build_problem().scientific_fingerprint,
              "parameter_order":list(build_problem().parameter_order)}
    for key,value in expected.items():require(q.get(key)==value,"qualified FP32 comparator binding mismatch: "+key)
    return q


def model_summary(regressor,utility):
    contract=utility.model_contract(regressor)
    require(regressor.softmax_temperature==.9
            and str(regressor.bardist_.borders.dtype)=="torch.float32"
            and str(regressor.renormalized_criterion_.borders.dtype)=="torch.float32",
            "frozen FP32 border/temperature postprocessing contract changed")
    require(contract["input_normalization_train_only_verified"],"actual train-only normalization required")
    for module in contract["modules"]:
        if module["type"]=="InputNormalizationEncoderStep":
            for name,item in module["tensors"].items():
                if item.get("dtype")!="torch.float32":
                    raise RuntimeError("input normalization statistics left FP32: "+name)
    # Under AMP, forced_dtype=None is intentional, not an FP32 guarantee.
    return {"contract":contract,"target_mean":regressor.y_train_mean_,"target_sd":regressor.y_train_std_,
            "canonical_borders_dtype":str(regressor.bardist_.borders.dtype),
            "physical_borders_dtype":str(regressor.renormalized_criterion_.borders.dtype),
            "temperature":regressor.softmax_temperature}


def run(*,banks,prefix_run,checkpoint,qualification,ledger,output,mode,max_seconds=900,trace=True):
    from posteriorforge.artifacts import runtime_identity,sha256_file,write_json
    from posteriorforge.reference_grid import _require_budgeted_gpu
    from posteriorforge import execution_precision_probe as precision
    from posteriorforge import execution_gqa_hybrid as gqa,execution_cache as cache
    from posteriorforge.banks import load_context,load_heldout,derive_seed
    from posteriorforge.science.analytical_benchmark_v2 import build_problem,array_descriptor
    from posteriorforge.inference import CHECKPOINT_SHA256,frozen_regressor_settings,support_mask
    from posteriorforge.compat_cdf import finite_cdf_compatibility
    from posteriorforge.gate import verify_run_binding
    require(mode in precision.MODES,"one explicitly named approximate policy required")
    require(type(max_seconds) is int and 300<=max_seconds<=1050,"bounded deadline300..1050s required")
    budget=_require_budgeted_gpu(ledger);require(budget["gpus"]==1,"one registered reserved GPU required")
    output=Path(output);output.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    active=None
    report={**runtime_identity(),"schema":SCHEMA,"status":"running","mode":mode,
        "attribution":"Built with PriorLabs-TabPFN","approximate_experiment":True,"qualified_fp32":False,
        "full21_qualified":False,"posterior_vectors_generated":0,"reference_integrations":0,
        "scientific_gate_evaluated":False,"requested_stages":list(STAGES),"completed_stages":[],
        "thresholds":THRESHOLDS,"row_counts":list(ROWS),"stress_rows":8,"stage_results":[],
        "budget_binding":budget,"precision_module":precision.metadata(mode),
        "probe_source_sha256":sha256_file(__file__),"scope":"fixed-conditionals only; no equivalence or end-to-end speed claim",
        "cache_comparison_scope":"cache plus declared query-view layout; do not attribute the composite difference solely to KV reuse"}
    write_json(output/"manifest.json",report);engine=None
    try:
        import torch
        require(torch.cuda.is_available() and torch.cuda.device_count()==1,"one visible CUDA GPU required")
        require(torch.get_float32_matmul_precision()=="highest" and not torch.backends.cuda.matmul.allow_tf32 and not torch.is_autocast_enabled("cuda"),"start from strict original FP32 flags")
        seed=derive_seed(20261111,"precision-conditional-probe",case=0)
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS","16")))
        report.update(seed=seed,rng_role="precision-conditional-probe",
                      posterior_sampling_rng_used=False)
        require(sha256_file(checkpoint)==CHECKPOINT_SHA256,"immutable checkpoint required")
        precision.require_sources()
        utility=load_helper("qualify_execution_cache.py",UTILITY_SHA)
        require(utility.THRESHOLDS==THRESHOLDS,"unchanged strict FP32 thresholds required")
        context=load_context(banks,20261111,4096);case=load_heldout(banks,0);query=case.obs.reshape(1,33)
        qualified=bound_sources(qualification,context,query,checkpoint)
        prefix_run=Path(prefix_run);pm=json.loads((prefix_run/"manifest.json").read_text())
        require(sha256_file(prefix_run/"manifest.json")==qualified["prefix_manifest_sha256"]
                and sha256_file(prefix_run/"samples.npz")==qualified["prefix_samples_sha256"],"fixed-prefix source changed")
        verify_run_binding(pm,context,case.obs,20261111,0)
        with np.load(prefix_run/"samples.npz",allow_pickle=False) as data:values=data["samples"].copy()
        require(values.shape==(50000,21) and support_mask(values).all(),"original complete supported prefix source required")
        positions=np.asarray(qualified["fixed_prefix_row_indices"],dtype=int)
        prefixes=values[positions];theta=context.theta.astype(np.float32);obs=context.obs.astype(np.float32)
        problem=build_problem();noise=[problem.parameter_order.index(n) for n in problem.prior.noise_names]
        report.update(checkpoint_sha256=CHECKPOINT_SHA256,qualification_sha256=sha256_file(qualification),
            context_theta=array_descriptor(context.theta),context_obs=array_descriptor(context.obs),query=array_descriptor(query),
            parameter_order=list(problem.parameter_order),prefix_samples_sha256=pm["samples_sha256"],fixed_prefix_rows=positions.tolist())
        original_sources={"precision":precision.source_sha256(),"probe":sha256_file(__file__),"qualification":sha256_file(qualification)}
        active=None
        with finite_cdf_compatibility() as active:
            for stage in STAGES:
                require(time.perf_counter()+180<start+max_seconds,"bounded stage deadline; no partial result promoted")
                row={"stage_index":stage,"parameter":problem.parameter_order[stage],"passed":False,"comparisons":{}}
                report["stage_results"].append(row);report["active_stage"]=stage;write_json(output/"manifest.json",report)
                queries,stress=utility.fixed_queries(query,prefixes,theta,stage,noise,problem.prior.bounds)
                sets={"128":queries[:128],"1024":queries,"stress":stress}
                # Each arm is prepared independently. All retained outputs are CPU captures.
                initial_rng=utility.rng_identity(utility.rng_state(torch));t0=time.perf_counter()
                engine=gqa.GQAHybridStageExecutor(checkpoint,cdf_descriptor=active,query_batch_size=1024)
                prepared=engine.prepare(theta,obs,stage)
                g0={name:precision.capture_members(engine,x) for name,x in sets.items()}
                row["qualified_GQA_model"]=model_summary(engine.regressor,utility)
                row["qualified_GQA_prepare"]=prepared;row["qualified_GQA_total_seconds"]=time.perf_counter()-t0
                engine.close();engine=None;gc.collect()
                # Private FP32 arm is a hard prerequisite before lower precision.
                with precision.PrecisionSession(precision.CONTROL_MODE) as transparent:
                    engine=precision.PrecisionStageExecutor(checkpoint,cdf_descriptor=active)
                    engine.prepare(theta,obs,stage)
                    replay={name:precision.capture_members(engine,x) for name,x in sets.items()}
                    row["private_FP32_transparency"]={name:{"byte_exact":exact_prediction(g0[name],replay[name],torch),
                        "comparison":utility.compare_predictions(g0[name],replay[name])} for name in sets}
                    engine.close();engine=None
                row["private_FP32_session"]=transparent.report()
                require(all(r["byte_exact"] and r["comparison"]["passed"] for r in row["private_FP32_transparency"].values()),
                        "private FP32 adapter did not reproduce qualified FP32 output exactly")
                with precision.PrecisionSession(precision.CONTROL_MODE,query_views=False) as stock:
                    uncached=precision.UncachedPrecisionStage(checkpoint,cdf_descriptor=active,stage_index=stage,theta=theta,obs=obs)
                    u0_components={name:precision.capture_members(uncached,x) for name,x in sets.items()}
                    row["uncached_FP32_model"]=model_summary(uncached.regressor,utility)
                    original_regressor=uncached.regressor;original_base=uncached.base
                row["uncached_FP32_session"]=stock.report()
                u0={}
                for name,x in sets.items():
                    with original_base._model_rng():
                        u0[name]=cache.capture_corrected_prediction(original_regressor,x,cdf_descriptor=active)
                row["FP32_component_reconstruction_vs_actual_original_full_API"]={
                    name:reconstruction_checks(u0_components[name],u0[name],utility) for name in sets}
                require(all(value["passed"] for checks in row["FP32_component_reconstruction_vs_actual_original_full_API"].values()
                            for key,value in checks.items() if key!="scope"),
                        "FP32 component reconstruction differs from actual original full API")
                uncached.close();del original_regressor,original_base

                t0=time.perf_counter()
                with precision.PrecisionSession(mode,query_views=False) as low_stock:
                    uncached=precision.UncachedPrecisionStage(checkpoint,cdf_descriptor=active,stage_index=stage,theta=theta,obs=obs)
                    up={name:precision.capture_members(uncached,x) for name,x in sets.items()}
                    row["uncached_precision_model"]=model_summary(uncached.regressor,utility);uncached.close()
                row["uncached_precision_total_seconds"]=time.perf_counter()-t0;row["uncached_precision_session"]=low_stock.report()
                torch.cuda.reset_peak_memory_stats()
                with precision.PrecisionSession(mode) as candidate:
                    row["candidate_allocator_baseline_bytes"]=torch.cuda.memory_allocated()
                    row["candidate_allocator_reserved_baseline_bytes"]=torch.cuda.memory_reserved()
                    row["reserved_memory_scope"]="allocator pool retained from preceding control arms; not isolated candidate capacity"
                    engine=precision.PrecisionStageExecutor(checkpoint,cdf_descriptor=active)
                    torch.cuda.synchronize();t0=time.perf_counter();prepared=engine.prepare(theta,obs,stage);torch.cuda.synchronize()
                    row["cached_precision_prepare_seconds"]=time.perf_counter()-t0;row["cached_precision_prepare"]=prepared
                    row["computed_prefill_witness"]=precision.prefill_witness_passed(candidate.report(),stage)
                    require(row["computed_prefill_witness"]["passed"],"declared precision prefill not actually witnessed")
                    row["cache_after_computed_prefill"]=engine.inspect_cache()
                    row["cached_precision_model"]=model_summary(engine.regressor,utility)
                    fields=("target_mean","target_sd","canonical_borders_dtype","physical_borders_dtype","temperature")
                    row["target_statistics_and_border_types_preserved"]=all(
                        all(model[field]==row["uncached_FP32_model"][field] for field in fields)
                        for model in (row["qualified_GQA_model"],row["uncached_precision_model"],row["cached_precision_model"]))
                    require(row["target_statistics_and_border_types_preserved"],"target statistics/border types changed")
                    cp={name:precision.capture_members(engine,x) for name,x in sets.items()}
                    row["computed_query_witness"]=precision.query_witness_passed(candidate.report(),stage)
                    require(row["computed_query_witness"]["passed"],"actual query precision activation not witnessed")
                    for name in sets:row["comparisons"][name]=comparison_arms(u0[name],up[name],cp[name],g0[name],utility)
                    row["actual_precision_component_reconstruction"]={
                        name:reconstruction_checks(cp[name],g0[name],utility) for name in sets}
                    # Timing excludes raw member snapshots; dtype observer remains
                    # active and its small instrumentation scope is disclosed.
                    times={}
                    for name in ("128","1024"):
                        torch.cuda.synchronize();t0=time.perf_counter()
                        for member in range(8):engine.predict_component(member,sets[name],capture_member=False)
                        torch.cuda.synchronize();times[name]=time.perf_counter()-t0
                    row["capture_free_all8_query_seconds"]=times
                    row["timing_instrumentation"]="no raw snapshots; bounded projection-dtype observer remains active"
                    row["TF32_instruction_precision_claim"]="flags and actual kernels are recorded; no instruction-level precision is inferred from flags alone"
                    if trace:row["bounded_trace"]=utility.profile_prediction(
                        lambda:engine.predict_component(0,queries[:128],capture_member=False),output,
                        "precision-stage-"+str(stage),torch)
                    row["peak_allocated_bytes"]=torch.cuda.max_memory_allocated()
                    row["peak_reserved_bytes"]=torch.cuda.max_memory_reserved()
                    engine.close();engine=None
                row["cached_precision_session"]=candidate.report()
                row["rng_preserved"]=initial_rng==utility.rng_identity(utility.rng_state(torch))
                row["passed"]=bool(row["rng_preserved"] and row["computed_prefill_witness"]["passed"]
                    and all(value["passed"] for comparisons in row["comparisons"].values() for value in comparisons.values())
                    and all(value["passed"] for checks in row["actual_precision_component_reconstruction"].values()
                            for key,value in checks.items() if key!="scope")
                    and all(session["aliases_and_flags_restored"] for session in
                            (row["private_FP32_session"],row["uncached_FP32_session"],
                             row["uncached_precision_session"],row["cached_precision_session"])))
                report["completed_stages"].append(stage);write_json(output/"manifest.json",report)
                del g0,replay,u0,u0_components,up,cp;gc.collect()
                if not row["passed"]:
                    report["status"]="rejected_by_unchanged_strict_conditional_screen";break
            else:report["status"]="three_stage_screen_pass_not_full21_or_posterior_qualification"
        report["compatibility_adapter"]=dict(active)
        require(active["original_alias_restored"] is True and active["sdk_sources_unchanged"] is True,"CDF context restoration failed")
        require(original_sources=={"precision":precision.source_sha256(),"probe":sha256_file(__file__),
                                   "qualification":sha256_file(qualification)},"source/proof changed during precision screen")
        report["sources_unchanged"]=True
    except BaseException as error:
        report.update(status="failed",error_type=type(error).__name__,error=str(error))
        if active is not None:report["compatibility_adapter"]=dict(active)
        raise
    finally:
        if engine is not None:engine.close()
        report["peak_cpu_rss_bytes"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=="darwin" else 1024)
        report["elapsed_seconds"]=time.perf_counter()-start;write_json(output/"manifest.json",report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("banks","prefix-run","checkpoint","qualification","ledger","output","mode"):parser.add_argument("--"+name,required=True)
    parser.add_argument("--max-seconds",type=int,default=900);parser.add_argument("--no-trace",action="store_true")
    args=vars(parser.parse_args());args["trace"]=not args.pop("no_trace")
    result=run(**args);print(json.dumps({k:result[k] for k in ("status","mode","completed_stages","elapsed_seconds")},indent=2))


if __name__=="__main__":main()
