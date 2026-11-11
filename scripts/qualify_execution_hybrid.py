#!/usr/bin/env python3
"""Budgeted declared-fallback stage qualification; no posterior generation.

Built with PriorLabs-TabPFN. Each output stage is a separate, source-bound proof.
Original cache failures remain immutable. No all-cache/full21 claim is made.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import time
import numpy as np

BASE_QUALIFIER_SHA256="6c927a5c278c7fc54b556ac557ecef64f5e6338f5ae13bf466f3b1e6a4640f90"
SCHEMA="posteriorforge.execution_hybrid_stage_qualification.v1"
COLLECTION_SCHEMA="posteriorforge.execution_hybrid_stage_collection.v1"


def frozen_utilities():
    path=Path(__file__).with_name("qualify_execution_cache.py")
    if hashlib.sha256(path.read_bytes()).hexdigest()!=BASE_QUALIFIER_SHA256:raise RuntimeError("frozen numerical qualification utility source changed")
    spec=importlib.util.spec_from_file_location("_frozen_cache_qualification",path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def parse_fallbacks(values):
    from posteriorforge.execution_hybrid import canonical_fallback_map
    mapping={}
    for value in values:
        stage,members=value.split(":",1);stage=int(stage)
        if stage in mapping:raise ValueError("duplicate fallback stage")
        mapping[stage]=[int(member) for member in members.split(",")]
    return canonical_fallback_map(mapping)


def hybrid_prediction(engine,query):
    """Capture actual routed components; FP32 mean is diagnostic reconstruction."""
    import torch
    predictions=[];members=[];routes=[]
    for index in range(8):
        before=engine.cdf_descriptor["patched_call_count"]
        cached_before=engine.hybrid_counters["resident_component_predict_count"];uncached_before=engine.hybrid_counters["uncached_predict_count"]
        prediction=engine.predict_component(index,query,capture_member=True)
        cached_delta=engine.hybrid_counters["resident_component_predict_count"]-cached_before
        uncached_delta=engine.hybrid_counters["uncached_predict_count"]-uncached_before
        path=prediction["hybrid_execution_path"]
        route="uncached" if path=="original_fit_preprocessors" else "cached" if path=="resident_cache" else "unknown"
        expected="uncached" if index in engine.fallback_map.get(engine.stage_index,()) else "cached"
        if route!=expected:raise RuntimeError("actual hybrid execution path differs from declared member route")
        prediction["qualification_route"]={"execution_route":route,"route_counter_delta":uncached_delta if route=="uncached" else cached_delta,
            "other_route_counter_delta":cached_delta if route=="uncached" else uncached_delta,
            "corrected_cdf_call_delta":engine.cdf_descriptor["patched_call_count"]-before,
            "executed_corrected_helper_fingerprint_matches":engine.cdf_descriptor.get("executed_corrected_cdf_code_sha256")==engine.cdf_descriptor.get("corrected_cdf_code_sha256") and engine.cdf_descriptor.get("corrected_cdf_code_sha256")!=engine.cdf_descriptor.get("original_cdf_code_sha256")}
        predictions.append(prediction);members.append(prediction["member"]);routes.append(route)
    logits=torch.stack([member["translated_probabilities"] for member in members],dim=0).mean(0).log()
    return {"criterion":predictions[0]["criterion"],"logits":logits,"members":members,"component_predictions":predictions,
            "member_routes":{str(index):route for index,route in enumerate(routes)},"full_mixture_is_diagnostic_reconstruction":True}


def component_report(reference,candidate,engine,utilities):
    import torch
    from posteriorforge import execution_hybrid as hybrid,execution_mixture as mixture
    metrics=utilities.compare_predictions(reference,candidate);members=[];fast_failures=0
    raw=np.stack([utilities._numpy(member["translated_probabilities"]) for member in candidate["members"]]).astype(np.float64)
    roundoff=mixture.mixture_roundoff_report(raw);masses=[];actual_outputs=[]
    for index,prediction in enumerate(candidate["component_predictions"]):
        row={**metrics["members"][index],"ensemble_index":index,**prediction["qualification_route"]}
        mass=utilities._numpy(prediction["component_raw_mass"]);expected=raw[index].sum(-1)
        bound=8*raw.shape[-1]*np.finfo(np.float64).eps;error=float(np.abs(mass-expected).max())
        normalized=utilities.compare_probabilities(raw[index]/expected[:,None],utilities._numpy(prediction["logits"].softmax(-1)))
        uvalues=np.array([0.,np.finfo(np.float32).eps/2,np.finfo(np.float32).eps,.1,.5,.9,1-np.finfo(np.float32).eps,np.nextafter(np.float32(1),np.float32(0))],np.float32)
        uniforms=torch.from_numpy(np.resize(uvalues,len(prediction["logits"])).copy())
        fast=mixture.finite_bar_icdf(prediction["logits"],prediction["criterion"].borders,uniforms)
        stock=torch.stack([prediction["criterion"].icdf(prediction["logits"][j],float(uniforms[j])).reshape(()) for j in range(len(uniforms))])
        exact=bool(((fast==stock)|(torch.isnan(fast)&torch.isnan(stock))).all());fast_failures+=int(not exact)
        row.update(component_raw_mass_matches_fp64_sum=bool(error<=bound),component_raw_mass_sum_error=error,
            component_raw_mass_fp64_roundoff_bound=bound,mass_min=float(mass.min()),mass_max=float(mass.max()),
            component_head_mass_envelope=float(prediction["head_mass_envelope"]),normalized_component=normalized,
            fast_icdf_same_logits_exact=exact,fast_icdf_nonfinite_matches=int((~torch.isfinite(stock)).sum()))
        row["passed"]=bool(row["passed"] and error<=bound and normalized["passed"] and exact and row["route_counter_delta"]==1 and row["other_route_counter_delta"]==0 and row["corrected_cdf_call_delta"]==1 and row["executed_corrected_helper_fingerprint_matches"])
        members.append(row);masses.append(mass);actual_outputs.append(utilities._numpy(prediction["logits"].softmax(-1)).astype(np.float64))
    control=utilities._numpy(reference["logits"].softmax(-1));candidate_final=utilities._numpy(candidate["logits"].softmax(-1))
    ideal=raw.mean(0);ideal/=ideal.sum(-1,keepdims=True)
    control_raw=np.stack([utilities._numpy(member["translated_probabilities"]) for member in reference["members"]]).astype(np.float64)
    control_ideal=control_raw.mean(0);control_ideal/=control_ideal.sum(-1,keepdims=True)
    reconstructed=utilities.compare_probabilities(control_ideal,control)
    observed=utilities.compare_probabilities(ideal,control)
    hybrid_mean_roundoff=utilities.compare_probabilities(ideal,candidate_final)
    mass=np.stack(masses);weights=mass/mass.sum(0,keepdims=True)
    weighted=utilities.compare_probabilities((np.stack(actual_outputs)*weights[:,:,None]).sum(0),control)
    restored=bool(engine.hybrid_counters["original_preprocessing_lists_restored"] and engine.hybrid_counters["original_translator_alias_restored"] and engine.cached_component.metadata["full8_member_lists_restored"])
    return {"schema":"posteriorforge.execution_component_qualification.v1","engine_id":hybrid.ENGINE_ID,
        "module_sha256":hybrid.execution_source_sha256(),"component_mixture_module_sha256":mixture.execution_source_sha256(),"sampler_module_sha256":mixture.execution_source_sha256(),
        "passed":bool(metrics["passed"] and all(row["passed"] for row in members) and roundoff["mass_corrected_reconstruction_passed"] and roundoff["component_mass_guard_passed"] and reconstructed["passed"] and observed["passed"] and hybrid_mean_roundoff["passed"] and weighted["passed"] and restored),
        "members":members,"roundoff":roundoff,"normalized_mean_captured_raw_vs_full_fp32":reconstructed,
        "normalized_mean_observed_raw_vs_full_fp32":observed,"mass_corrected_observed_outputs_vs_full_fp32":weighted,
        "hybrid_diagnostic_mean_roundoff":hybrid_mean_roundoff,"fast_icdf_same_logits_failures":fast_failures,"stage0_shared_distribution":None,"stage0_dedup_scope":"full_mixture_only",
        "executor_metadata":{"full8_member_lists_restored":restored,"original_translator_alias_restored":engine.hybrid_counters["original_translator_alias_restored"],
                             "member_routes":candidate["member_routes"]},"component_distribution_dedup_qualified":False,
        "production_uniform_domain":"FP32[0,1);u=1 stock-only stress","uniform_head_without_mass_correction_is_not_authorized":True}


def qualify(*,banks,prefix_run,checkpoint,ledger,output,fallback,stages=None,max_seconds=2250):
    from posteriorforge.artifacts import runtime_identity,sha256_file,write_json
    from posteriorforge.reference_grid import _require_budgeted_gpu
    from posteriorforge.execution_hybrid import canonical_fallback_map
    mapping=canonical_fallback_map(fallback);stages=tuple(mapping) if stages is None else tuple(stages)
    if not stages or len(set(stages))!=len(stages) or any(stage not in mapping for stage in stages):raise ValueError("qualify only explicitly declared fallback stages")
    if not 60<=max_seconds<=2250:raise ValueError("bounded qualification deadline60..2250seconds required")
    budget=_require_budgeted_gpu(ledger)
    if budget["gpus"]!=1:raise RuntimeError("one reserved GPU required")
    output=Path(output);output.mkdir(parents=True,exist_ok=False);start=time.perf_counter();deadline=start+max_seconds
    collection={**runtime_identity(),"schema":COLLECTION_SCHEMA,"status":"running","attribution":"Built with PriorLabs-TabPFN", "requested_stages":list(stages),"completed_stages":[],"budget_binding":budget,
                "posterior_vectors_generated":0,"stage_proofs":[],"full21_qualified":False,"full21_hybrid_qualified":False}
    write_json(output/"manifest.json",collection);proofs=[];engine=None;active=None
    try:
        import torch
        from tabpfn import TabPFNRegressor
        from posteriorforge import execution_hybrid as hybrid,execution_cache as cache,execution_mixture as mixture
        from posteriorforge.banks import load_context,load_heldout,derive_seed
        from posteriorforge.inference import CHECKPOINT_SHA256,frozen_regressor_settings,support_mask
        from posteriorforge.inference_cdf import guard_baseline_source
        from posteriorforge.gate import verify_run_binding
        from posteriorforge.compat_cdf import finite_cdf_compatibility
        from posteriorforge.science.analytical_benchmark_v2 import array_descriptor,build_problem
        utilities=frozen_utilities();source=guard_baseline_source();hybrid.require_frozen_sources()
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError("one visible CUDA GPU required")
        if torch.get_float32_matmul_precision()!="highest" or torch.backends.cuda.matmul.allow_tf32:raise RuntimeError("unchanged FP32 highest/TF32False required")
        if sha256_file(checkpoint)!=CHECKPOINT_SHA256:raise ValueError("immutable checkpoint required")
        context=load_context(banks,20261111,4096);observed=load_heldout(banks,0);query=observed.obs.reshape(1,33)
        prefix_run=Path(prefix_run);manifest=json.loads((prefix_run/"manifest.json").read_text())
        if manifest.get("status")!="complete" or manifest.get("variant")!="cdf_arithmetic_v1":raise ValueError("complete original corrected-pilot prefixes required")
        verify_run_binding(manifest,context,observed.obs,20261111,0)
        if sha256_file(prefix_run/"samples.npz")!=manifest["samples_sha256"]:raise ValueError("prefix payload hash mismatch")
        with np.load(prefix_run/"samples.npz",allow_pickle=False) as data:values=data["samples"].copy()
        if values.shape!=(50000,21) or not support_mask(values).all():raise ValueError("authenticated complete supported prefix source required")
        positions=np.linspace(0,len(values)-1,1024,dtype=int);prefixes=values[positions]
        theta=context.theta.astype(np.float32);obs=context.obs.astype(np.float32);joint=np.concatenate((obs,theta),axis=1)
        problem=build_problem();noise=[problem.parameter_order.index(name) for name in problem.prior.noise_names]
        seed=derive_seed(20261111,"execution-hybrid-qualification",case=0);np.random.seed(seed);random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS","16")))
        common={"case":0,"context_seed":20261111,"context_size":4096,"seed":seed,"rng_role":"execution-hybrid-qualification","control_variant":"cdf_arithmetic_v1","cache_module_sha256":cache.execution_source_sha256(),"mixture_module_sha256":mixture.execution_source_sha256(),
            "hybrid_module_sha256":hybrid.execution_source_sha256(),"qualifier_source_sha256":sha256_file(__file__),"base_qualifier_source_sha256":BASE_QUALIFIER_SHA256,
            "checkpoint_sha256":CHECKPOINT_SHA256,"scientific_fingerprint":problem.scientific_fingerprint,"parameter_order":list(problem.parameter_order),
            "context_theta":array_descriptor(context.theta),"context_obs":array_descriptor(context.obs),"query":array_descriptor(query),
            "prefix_manifest_sha256":sha256_file(prefix_run/"manifest.json"),"prefix_samples_sha256":manifest["samples_sha256"],"fixed_prefix_row_indices":positions.tolist(),
            "thresholds":dict(utilities.THRESHOLDS),"row_counts":list(utilities.DEFAULT_ROWS),"icdf_contract_version":2,
            "sources":{"original_execution":source,"hybrid_module_sha256":hybrid.execution_source_sha256(),"cache_module_sha256":cache.execution_source_sha256(),"mixture_module_sha256":mixture.execution_source_sha256()},
            "scope":"fresh all8 member hybrid fixed-conditional stage proof; not a full21 all-cache or posterior-accuracy pass",**hybrid.policy_metadata(mapping)}
        kwargs=frozen_regressor_settings(checkpoint);kwargs["inference_precision"]=torch.float32
        with finite_cdf_compatibility() as active:
            for stage in stages:
                if time.perf_counter()+90>=deadline:collection["status"]="incomplete";break
                stage_root=output/f"stage-{stage}";stage_root.mkdir(exist_ok=False)
                proof={**runtime_identity(),**common,"schema":SCHEMA,"status":"running","passed":False,"stage_index":stage,"fallback_members":list(mapping[stage]),
                       "member_routes":{str(index):"uncached" if index in mapping[stage] else "cached" for index in range(8)},"full21_qualified":False,
                       "fullstage8_component_qualified":False,"fullstage_sampler_qualified":False,"budget_binding":budget,"posterior_vectors_generated":0}
                proofs.append((stage_root,proof));collection["active_stage"]=stage;write_json(stage_root/"manifest.json",proof);write_json(output/"manifest.json",collection)
                baseline=TabPFNRegressor(**kwargs);initial=utilities.rng_state(torch)
                torch.cuda.synchronize();t0=time.perf_counter();baseline.fit(joint[:,:33+stage],joint[:,33+stage]);torch.cuda.synchronize();control_fit=time.perf_counter()-t0
                queries,stress=utilities.fixed_queries(query,prefixes,theta,stage,noise,problem.prior.bounds)
                controls={count:cache.capture_corrected_prediction(baseline,queries[:count],cdf_descriptor=active) for count in utilities.DEFAULT_ROWS}
                stress_control=cache.capture_corrected_prediction(baseline,stress,cdf_descriptor=active)
                control_contract=utilities.model_contract(baseline);baseline.model_.cpu();utilities.restore_rng(torch,initial)
                engine=hybrid.HybridStageExecutor(checkpoint,cdf_descriptor=active,fallback_map=mapping,query_batch_size=1024)
                torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t0=time.perf_counter();prepared=engine.prepare(theta,obs,stage);torch.cuda.synchronize();prepare=time.perf_counter()-t0
                comparisons=[];candidates={}
                for count in utilities.DEFAULT_ROWS:
                    candidates[count]=hybrid_prediction(engine,queries[:count])
                    comparisons.append({"query_rows":count,"cache_vs_same_chunk_control":utilities.compare_predictions(controls[count],candidates[count]),
                                        "control_chunk_vs_1024":utilities.compare_predictions(utilities.slice_prediction(controls[1024],count),controls[count])})
                stress_candidate=hybrid_prediction(engine,stress);stress_comparison=utilities.compare_predictions(stress_control,stress_candidate)
                component=component_report(controls[1024],candidates[1024],engine,utilities)
                proof.update(comparisons=comparisons,stress_comparison=stress_comparison,component_qualification=component,
                    control_model_contract=control_contract,resident_model_contract=utilities.model_contract(engine.regressor),
                    uncached_model_contract=utilities.model_contract(engine.uncached_regressor),prepared=prepared,cache_telemetry=engine.inspect_cache(),
                    hybrid_execution_counters={key:value for key,value in engine.hybrid_counters.items() if key!="stages"},
                    timings_seconds={"control_fit":control_fit,"hybrid_prepare":prepare},
                    memory={"peak_allocated_bytes":torch.cuda.max_memory_allocated(),"peak_reserved_bytes":torch.cuda.max_memory_reserved()})
                proof["normalization_train_only_verified"]=all(proof[key]["input_normalization_train_only_verified"] for key in ("control_model_contract","resident_model_contract","uncached_model_contract"))
                proof["fullstage8_component_qualified"]=component["passed"]
                proof["fullstage_sampler_qualified"]=component["fast_icdf_same_logits_failures"]==0 and component["passed"]
                proof["passed"]=all(c["cache_vs_same_chunk_control"]["passed"] and c["control_chunk_vs_1024"]["passed"] for c in comparisons) and stress_comparison["passed"] and component["passed"] and proof["normalization_train_only_verified"]
                proof["status"]="stage_complete_pending_adapter_restore";write_json(stage_root/"manifest.json",proof)
                engine.close();engine=None;collection["completed_stages"].append(stage);collection["stage_proofs"].append({"stage_index":stage,"file":str(stage_root.relative_to(output)/"manifest.json"),"numerical_passed":proof["passed"]})
                collection["elapsed_seconds"]=time.perf_counter()-start;write_json(output/"manifest.json",collection)
                del baseline,controls,candidates,stress_control,stress_candidate
        if guard_baseline_source()!=source or hybrid.execution_source_sha256()!=common["hybrid_module_sha256"] or sha256_file(__file__)!=common["qualifier_source_sha256"]:raise RuntimeError("hybrid qualification execution source changed")
        hybrid.require_frozen_sources();cache._load_sdk()
        for stage_root,proof in proofs:
            proof.update(status="stage_pass" if proof["passed"] else "stage_failure",sources_unchanged=True,compatibility_adapter=dict(active))
            write_json(stage_root/"manifest.json",proof)
        if collection["status"]!="incomplete":collection["status"]="stages_pass" if len(proofs)==len(stages) and all(p["passed"] for _,p in proofs) else "stage_failure"
    except BaseException as error:
        collection.update(status="failed",error_type=type(error).__name__,error=str(error))
        for stage_root,proof in proofs:
            proof.update(status="failed",passed=False,error_type=type(error).__name__,error=str(error))
            if active is not None:proof["compatibility_adapter"]=dict(active)
            write_json(stage_root/"manifest.json",proof)
        raise
    finally:
        if engine is not None:engine.close()
        collection["elapsed_seconds"]=time.perf_counter()-start;write_json(output/"manifest.json",collection)
    return collection


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("banks","prefix-run","checkpoint","ledger","output"):parser.add_argument("--"+name,required=True)
    parser.add_argument("--fallback",action="append",required=True,help="Declared stage:member[,member] repeated per stage")
    parser.add_argument("--stages");parser.add_argument("--max-seconds",type=int,default=2250)
    args=vars(parser.parse_args());args["fallback"]=parse_fallbacks(args["fallback"]);args["stages"]=tuple(map(int,args["stages"].split(","))) if args["stages"] else None
    result=qualify(**args);print(json.dumps(result,indent=2))
    if result["status"]!="stages_pass":raise SystemExit(2)


if __name__=="__main__":main()
