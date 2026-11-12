#!/usr/bin/env python3
"""Fresh all21 FP32 GQA-hybrid qualification; no posterior generation.

Built with PriorLabs-TabPFN. Fresh original1024/all8 outputs retain the previous
member coverage. Saved goldens retain128 member rows, explicitly separately.
Original small-chunk invariance is authenticated, never silently re-assumed.
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

SCHEMA="posteriorforge.execution_gqa_hybrid_qualification.v1"
BASE_SHA="6c927a5c278c7fc54b556ac557ecef64f5e6338f5ae13bf466f3b1e6a4640f90"
HYBRID_HELPER_SHA="15e8e9ce844d3352e49534714f0e0569f42e970a3d9876630f877a346eaaf9b6"
GOLDEN_HELPER_SHA="02e389e317f7689ad232134475603807f06db361d306a1c46687af01863a66d6"
GQA_PROBE_SHA="62d624c4f4d2b3231b16a932e5818597d0718d2bfb4ac6137c7d7b00fe93bf00"
ASSEMBLER_SHA="29dcbc62f273fba0dad50f3a4063b445dbe0329f30371b164237c31a55fb7ae9"
THRESHOLDS={"max_cdf_absolute":2e-5,"max_row_total_variation":1e-4,"normalization_absolute":1e-5,"quantile_coordinate_span_fraction":2e-6}
BINDINGS=("checkpoint_sha256","scientific_fingerprint","parameter_order","context_theta","context_obs","query","prefix_manifest_sha256","prefix_samples_sha256")
ROWS=(1,16,128,1024)


def require(value,message):
    if not value:raise RuntimeError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_helper(name,expected):
    path=Path(__file__).with_name(name)
    require(sha(path)==expected,"frozen helper source mismatch: "+name)
    spec=importlib.util.spec_from_file_location("_gqa_hybrid_"+path.stem,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def prediction_passes(value):
    inverse=value.get("inverse_cdf",{})
    return (value.get("passed") is True and inverse.get("passed") is True
        and inverse.get("icdf_contract_version")==2 and inverse.get("independent_fp32_replay_failures")==0
        and inverse.get("same_logits_and_borders_exact_failures")==0
        and len(value.get("members",[]))==8 and all(m.get("passed") is True for m in value["members"]))


def bind_control_proof(path,golden):
    """Use only original-control chunk evidence; old candidate passes confer none."""
    path=Path(path);proof=json.loads(path.read_text())
    require(proof.get("schema")=="posteriorforge.execution_hybrid_qualification.v1"
        and proof.get("status")=="hybrid_screen_pass" and proof.get("assembler_source_sha256")==ASSEMBLER_SHA,
        "authenticated frozen control-chunk proof assembly required")
    require(proof.get("sources_unchanged") is True and proof.get("thresholds")==THRESHOLDS
        and proof.get("row_counts")==list(ROWS) and proof.get("icdf_contract_version")==2,"control proof settings changed")
    require(all(proof.get(k)==golden.get(k) for k in BINDINGS),"control proof/golden binding mismatch")
    require(proof.get("fixed_prefix_row_indices")==golden.get("logical_row_indices"),"control proof prefix cohort changed")
    archives={}
    for entry in proof["source_inputs"]:
        archive=(path.parent/entry["archived_manifest"]).resolve()
        require(archive.is_relative_to(path.parent.resolve()) and sha(archive)==entry["sha256"],"control proof source archive mismatch")
        source=json.loads(archive.read_text());require(source.get("status")==entry["producer_status"],"original producer status changed")
        adapter=source.get("compatibility_adapter",{})
        require(source.get("sources_unchanged") is True and adapter.get("original_alias_restored") is True
            and adapter.get("sdk_sources_unchanged") is True,"control evidence sources/adapter not restored")
        archives[entry["sha256"]]=(source,entry)
    selected={};records=[]
    for row in proof["stage_results"]:
        stage=row["stage_index"];route=proof["stage_routes"][str(stage)];source,entry=archives[route["source_sha256"]]
        original=source if source.get("stage_index")==stage else next(r for r in source["stage_results"] if r["stage_index"]==stage)
        require(row["comparisons"]==original["comparisons"],"selected original control comparison changed")
        comparisons={c["query_rows"]:c["control_chunk_vs_1024"] for c in original["comparisons"]}
        require(set(comparisons)==set(ROWS) and all(prediction_passes(v) for v in comparisons.values()),"original control chunk invariance failed")
        contract=original.get("control_model_contract",original.get("baseline_model_contract_after",{}))
        require(contract.get("input_normalization_train_only_verified") is True,"original control normalizer unverified")
        require(stage not in selected,"duplicate control proof stage")
        selected[stage]=comparisons
        records.append({"stage_index":stage,"source_sha256":entry["sha256"],"producer_status_preserved":entry["producer_status"],
            "original_control_only":True,"candidate_proof_inherited":False,"comparisons":comparisons})
    require(set(selected)==set(range(21)),"all21 original chunk-invariance stages required")
    return {"path":str(path.resolve()),"sha256":sha(path),"producer_status":proof["status"],"stage_evidence":records},proof


def validate_golden_header(golden):
    require(golden.get("schema")=="posteriorforge.execution_control_goldens.v1" and golden.get("status")=="complete"
        and golden.get("full21_original_control_captured") is True and golden.get("sources_unchanged") is True,
        "complete immutable original-control goldens required")
    require(golden.get("producer_sha256")==GOLDEN_HELPER_SHA and golden.get("member_rows_retained")==128
        and golden.get("stage_coverage")==list(range(21)),"golden producer/member coverage changed")
    adapter=golden.get("compatibility_adapter",{})
    require(adapter.get("original_alias_restored") is True and adapter.get("sdk_sources_unchanged") is True
        and adapter.get("executed_corrected_cdf_code_sha256")==adapter.get("corrected_cdf_code_sha256")
        and adapter.get("patched_call_count",0)>0,"golden CDF correction execution unverified")


def golden_prediction(arrays,entry,criterion,torch):
    keep=entry["metadata"]["member_rows"];members=[]
    require(np.array_equal(arrays["physical_borders"],criterion.borders.cpu().numpy()),"golden canonical physical borders changed")
    for index,metadata in enumerate(entry["metadata"]["members"]):
        config=metadata["configuration"];prefix=f"member_{index}_"
        members.append({"ensemble_index":index,"raw_logits":torch.from_numpy(arrays[prefix+"translation_input_logits"]),
            "translated_probabilities":torch.from_numpy(arrays[prefix+"translated_probabilities"]),
            "from_borders":torch.from_numpy(arrays[prefix+"from_borders"]),"to_borders":torch.from_numpy(arrays[prefix+"to_borders"]),
            "configuration_identity":{"type":config["type"],"repr":config["repr"],"sha256":config["pickle_sha256"]}})
    return {"criterion":criterion,"logits":torch.from_numpy(arrays["final_logits"][:keep]),"members":members}


def load_payload(root,entry,helper):
    helper.verify_payload(root,entry)
    with np.load(Path(root)/entry["file"],allow_pickle=False) as data:return {k:data[k].copy() for k in data.files}


def context_passes(d,*,expanded):
    keys=("original_alias_restored","original_function_restored","original_code_unchanged","active_code_unchanged",
        "gqa_source_unchanged","sdk_sources_unchanged","source_guard_passed")
    fingerprint=(d.get("executed_original_attention_code_sha256")==d.get("original_attention_code_sha256")
        and d.get("original_attention_code_sha256") is not None and d.get("original_attention_code_sha256")!=d.get("active_attention_code_sha256"))
    return bool(all(d.get(k) is True for k in keys) and fingerprint and d.get("attention_call_count",0)>0
        and (d.get("expanded_attention_calls",0)>0 if expanded else d.get("expanded_attention_calls")==0))


def gqa_checks(candidate,stage):
    records=[]
    for index,prediction in enumerate(candidate["component_predictions"]):
        d=prediction["gqa_attention_descriptor"];fallback=stage==10 and index==5
        verified=context_passes(d,expanded=not fallback)
        records.append({"member_index":index,"expected_route":"uncached" if fallback else "expanded_cached",
            "passed":verified,"descriptor":d})
    return {"passed":all(r["passed"] for r in records),"members":records}


def stage_passes(row):
    return bool(row.get("normalization_train_only_verified") and row.get("preprocessor_identity")
        and row.get("prepared_state_unchanged") and row.get("rng_unchanged")
        and prediction_passes(row.get("fresh_control_vs_saved_golden",{}))
        and prediction_passes(row.get("stress_control_vs_saved_golden",{}))
        and all(prediction_passes(c["candidate_vs_fresh_control1024_slice"]) and c["actual_gqa_execution"]["passed"] for c in row.get("comparisons",[]))
        and tuple(c["query_rows"] for c in row.get("comparisons",[]))==ROWS and prediction_passes(row.get("stress_comparison",{}))
        and row.get("stress_gqa_execution",{}).get("passed") and row.get("component_qualification",{}).get("passed")
        and row.get("stress_component_qualification",{}).get("passed")
        and all(v.get("passed") for v in row.get("outlier_checks",[]))
        and (row.get("stage_index")!=0 or row.get("stage0_shared_distribution",{}).get("passed")))


def qualify(*,banks,prefix_run,checkpoint,ledger,output,goldens,control_proof,max_seconds=1050):
    require(60<=max_seconds<=2250,"bounded60..2250seconds required")
    from posteriorforge.artifacts import runtime_identity,sha256_file,write_json
    from posteriorforge.reference_grid import _require_budgeted_gpu
    from posteriorforge import execution_gqa_hybrid as execution,execution_cache as cache,execution_mixture as mixture
    from posteriorforge.inference_cdf import guard_baseline_source
    budget=_require_budgeted_gpu(ledger);require(budget["gpus"]==1,"one registered GPU allocation required")
    root=Path(output);root.mkdir(parents=True,exist_ok=False);start=time.perf_counter();deadline=start+max_seconds
    result={**runtime_identity(),"schema":SCHEMA,"status":"running","attribution":"Built with PriorLabs-TabPFN",
        "posterior_vectors_generated":0,"scientific_gate_evaluated":False,"full21_qualified":False,"cache_only_full21_qualified":False,
        "qualified":False,"full21_gqa_hybrid_qualified":False,"full21_component_qualified":False,"full21_sampler_qualified":False,
        "budget_binding":budget,"row_counts":list(ROWS),"thresholds":dict(THRESHOLDS),"icdf_contract_version":2,
        "execution_engine":execution.execution_metadata(),"qualifier_source_sha256":sha(__file__),"stage_results":[],"stage_coverage":[],
        "producer_coverage":{"fresh_original_control":{"rows":1024,"members":8,"member_rows":1024,"stress_rows":8,"calls_per_stage_main":1},
            "saved_golden":{"rows":1024,"members":8,"member_rows":128,"same_as_fresh1024_coverage":False},
            "candidate":{"chunks":list(ROWS),"members_per_chunk":8,"member_rows_at_largest_chunk":1024,"stress_rows":8}},
        "control_small_chunk_reference":"fresh1024 firstN, supported by authenticated previous original chunk-invariance proof",
        "scope":"all21 fixed-conditional GQA-hybrid proof; no posterior-accuracy or end-to-end speed claim"}
    active=None;engine=None;write_json(root/"manifest.json",result)
    try:
        import torch
        from tabpfn import TabPFNRegressor
        from posteriorforge.compat_cdf import finite_cdf_compatibility
        from posteriorforge.banks import load_context,load_heldout,derive_seed
        from posteriorforge.inference import CHECKPOINT_SHA256,frozen_regressor_settings,support_mask
        from posteriorforge.gate import verify_run_binding
        from posteriorforge.science.analytical_benchmark_v2 import array_descriptor,build_problem
        utilities=load_helper("qualify_execution_cache.py",BASE_SHA)
        routed=load_helper("qualify_execution_hybrid.py",HYBRID_HELPER_SHA)
        golden_helper=load_helper("build_execution_goldens.py",GOLDEN_HELPER_SHA)
        probe=load_helper("boundedprobe_gqa_expansion.py",GQA_PROBE_SHA)
        execution.require_frozen_sources();original_source=guard_baseline_source();engine_sha=execution.execution_source_sha256()
        require(torch.cuda.is_available() and torch.cuda.device_count()==1,"one visible CUDA GPU required")
        require(torch.get_default_dtype()==torch.float32 and torch.get_float32_matmul_precision()=="highest" and not torch.backends.cuda.matmul.allow_tf32,"unchanged FP32 highest/TF32False required")
        require(sha(checkpoint)==CHECKPOINT_SHA256,"immutable checkpoint required")
        golden_root=Path(goldens);golden_path=golden_root/"manifest.json";golden=json.loads(golden_path.read_text());validate_golden_header(golden)
        control_evidence,oldproof=bind_control_proof(control_proof,golden)
        golden_sha=sha(golden_path);result.update(golden_manifest_sha256=golden_sha,original_control_chunk_invariance=control_evidence)
        context=load_context(banks,20261111,4096);heldout=load_heldout(banks,0);query=heldout.obs.reshape(1,33)
        prefix_root=Path(prefix_run);manifest=json.loads((prefix_root/"manifest.json").read_text())
        require(manifest.get("status")=="complete" and manifest.get("variant")=="cdf_arithmetic_v1","complete corrected-control prefixes required")
        verify_run_binding(manifest,context,heldout.obs,20261111,0)
        require(sha(prefix_root/"samples.npz")==manifest["samples_sha256"],"prefix payload hash mismatch")
        with np.load(prefix_root/"samples.npz",allow_pickle=False) as data:samples=data["samples"].copy()
        require(samples.shape==(50000,21) and support_mask(samples).all(),"50000 complete supported saved prefix vectors required")
        indices=np.linspace(0,49999,1024,dtype=np.int64);theta=context.theta.astype(np.float32);obs=context.obs.astype(np.float32)
        problem=build_problem();joint=np.concatenate((obs,theta),axis=1);noise=[problem.parameter_order.index(n) for n in problem.prior.noise_names]
        binding={"checkpoint_sha256":CHECKPOINT_SHA256,"scientific_fingerprint":problem.scientific_fingerprint,"parameter_order":list(problem.parameter_order),
            "context_theta":array_descriptor(context.theta),"context_obs":array_descriptor(context.obs),"query":array_descriptor(query),
            "prefix_manifest_sha256":sha(prefix_root/"manifest.json"),"prefix_samples_sha256":manifest["samples_sha256"]}
        require(all(golden.get(k)==v for k,v in binding.items()) and golden["logical_row_indices"]==indices.tolist(),"golden/fresh scientific binding changed")
        seed=derive_seed(20261111,"execution-gqa-hybrid-qualification",case=0)
        np.random.seed(seed);random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS","16")))
        result.update(**binding,fixed_prefix_row_indices=indices.tolist(),context_seed=20261111,case=0,context_size=4096,seed=seed,
            original_sources=original_source,base_qualifier_source_sha256=BASE_SHA,hybrid_qualifier_source_sha256=HYBRID_HELPER_SHA,
            cache_module_sha256=cache.execution_source_sha256(),mixture_module_sha256=mixture.execution_source_sha256(),
            module_sha256=engine_sha,gqa_module_sha256=execution.PINNED_GQA_SHA256,hybrid_module_sha256=execution.PINNED_HYBRID_SHA256,
            **execution.hybrid.policy_metadata({10:[5]}))
        kwargs=frozen_regressor_settings(checkpoint);kwargs["inference_precision"]=torch.float32
        with finite_cdf_compatibility() as active:
            for stage in range(21):
                if time.perf_counter()+45>=deadline:result["status"]="incomplete";break
                result["active_stage"]=stage;write_json(root/"manifest.json",result)
                record=golden["stages"][stage];require(record["stage_index"]==stage,"golden stage ordering changed")
                main_entry=next(v for v in record["payloads"] if v["metadata"]["label"]=="main")
                stress_entry=next(v for v in record["payloads"] if v["metadata"]["label"]=="stress")
                main_arrays=load_payload(golden_root,main_entry,golden_helper);stress_arrays=load_payload(golden_root,stress_entry,golden_helper)
                queries,stress=utilities.fixed_queries(query,samples[indices],theta,stage,noise,problem.prior.bounds)
                require(np.array_equal(queries,main_arrays["query"],equal_nan=True) and np.array_equal(stress,stress_arrays["query"],equal_nan=True),"golden/fresh query bytes changed")
                baseline=TabPFNRegressor(**kwargs);before_control=utilities.rng_state(torch);t0=time.perf_counter()
                baseline.fit(joint[:,:33+stage],joint[:,33+stage])
                control=cache.capture_corrected_prediction(baseline,queries,cdf_descriptor=active)
                stress_control=cache.capture_corrected_prediction(baseline,stress,cdf_descriptor=active)
                contract=utilities.model_contract(baseline);control_seconds=time.perf_counter()-t0
                fit_context=[{"member_index":i,"X_train":golden_helper.array_info(x),"y_train":golden_helper.array_info(y),
                    "preprocessor_pickle_sha256":contract["preprocessor_sha256"][i],"categorical_indices":list(cat)}
                    for i,(x,y,cat) in enumerate(zip(baseline.executor_.X_trains,baseline.executor_.y_trains,baseline.executor_.cat_ixs))]
                golden_fit_context_equal=fit_context==record["fit_transform_context"]
                golden_reference=golden_prediction(main_arrays,main_entry,control["criterion"],torch)
                golden_main_check=utilities.compare_predictions(golden_reference,utilities.slice_prediction(control,128))
                full_golden_check=utilities.compare_probabilities(torch.from_numpy(main_arrays["final_logits"]).softmax(-1).numpy(),control["logits"].softmax(-1).numpy())
                golden_main_check["saved_full1024_final"]=full_golden_check;golden_main_check["passed"] &= full_golden_check["passed"]
                golden_stress=golden_prediction(stress_arrays,stress_entry,control["criterion"],torch)
                golden_stress_check=utilities.compare_predictions(golden_stress,stress_control)
                baseline.model_.cpu();utilities.restore_rng(torch,before_control)
                engine=execution.GQAHybridStageExecutor(checkpoint,cdf_descriptor=active,query_batch_size=1024)
                initial_rng=utilities.rng_identity(utilities.rng_state(torch));t0=time.perf_counter();prepared=engine.prepare(theta,obs,stage);prepare_seconds=time.perf_counter()-t0
                rng_records=[{"path":"prepare","before":initial_rng,"after":utilities.rng_identity(utilities.rng_state(torch))}]
                state=probe.prepared_state(engine.base);comparisons=[];candidates={}
                for count in ROWS:
                    before=utilities.rng_identity(utilities.rng_state(torch));candidate=routed.hybrid_prediction(engine,queries[:count]);candidates[count]=candidate
                    rng_records.append({"path":"component_all8","rows":count,"before":before,"after":utilities.rng_identity(utilities.rng_state(torch))})
                    comparisons.append({"query_rows":count,"candidate_vs_fresh_control1024_slice":utilities.compare_predictions(utilities.slice_prediction(control,count),candidate),
                        "actual_gqa_execution":gqa_checks(candidate,stage),"reference_is_fresh1024_slice":True,"same_chunk_control_recomputed":False})
                before=utilities.rng_identity(utilities.rng_state(torch));stress_candidate=routed.hybrid_prediction(engine,stress)
                rng_records.append({"path":"stress","before":before,"after":utilities.rng_identity(utilities.rng_state(torch))})
                component=routed.component_report(control,candidates[1024],engine,utilities)
                stress_component=routed.component_report(stress_control,stress_candidate,engine,utilities)
                for report in (component,stress_component):report.update(engine_id=execution.ENGINE_ID,module_sha256=engine_sha,gqa_module_sha256=execution.PINNED_GQA_SHA256,base_hybrid_module_sha256=execution.PINNED_HYBRID_SHA256)
                cached_contract=utilities.model_contract(engine.regressor)
                uncached_contract=utilities.model_contract(engine.uncached_regressor) if engine.uncached_regressor is not None else None
                outliers=[]
                for entry in record["payloads"]:
                    if not entry["metadata"]["label"].startswith("rep2_outlier"):continue
                    arrays=load_payload(golden_root,entry,golden_helper);reference=golden_prediction(arrays,entry,control["criterion"],torch)
                    before=utilities.rng_identity(utilities.rng_state(torch));prediction=routed.hybrid_prediction(engine,arrays["query"])
                    rng_records.append({"path":entry["metadata"]["label"],"before":before,"after":utilities.rng_identity(utilities.rng_state(torch))})
                    check=utilities.compare_predictions(reference,prediction);execution_check=gqa_checks(prediction,stage)
                    outlier_component=routed.component_report(reference,prediction,engine,utilities)
                    outliers.append({"label":entry["metadata"]["label"],"rows":entry["metadata"]["rows"],"target_excluded_from_query":entry["metadata"]["target_excluded_from_query"],
                        "prediction_comparison":check,"actual_gqa_execution":execution_check,"component_qualification":outlier_component,
                        "passed":prediction_passes(check) and execution_check["passed"] and outlier_component["passed"]})
                dedup=None
                if stage==0:
                    before=utilities.rng_identity(utilities.rng_state(torch));single=engine.predict(queries[:1],capture_members=True)
                    full=engine.predict(queries,capture_members=True)
                    rng_records.append({"path":"stage0_full_mixture_dedup","before":before,"after":utilities.rng_identity(utilities.rng_state(torch))})
                    dedup=utilities.compare_probabilities(np.repeat(single["logits"].softmax(-1).numpy(),1024,axis=0),full["logits"].softmax(-1).numpy())
                    full_compare=utilities.compare_predictions(control,full);u=torch.tensor([0.,np.finfo(np.float32).eps/2,.1,.5,.9,np.nextafter(np.float32(1),np.float32(0))],dtype=torch.float32)
                    fast=mixture.finite_bar_icdf(single["logits"],single["criterion"].borders,u)
                    stock=torch.stack([single["criterion"].icdf(single["logits"][0],float(v)).reshape(()) for v in u])
                    exact=bool(((fast==stock)|(torch.isnan(fast)&torch.isnan(stock))).all())
                    dedup.update(shared_icdf_same_logits_exact=exact,full1024_comparison=full_compare,
                        single_actual_gqa_context=single["gqa_attention_descriptor"],full_actual_gqa_context=full["gqa_attention_descriptor"])
                    dedup["passed"] &= exact and prediction_passes(full_compare) and context_passes(single["gqa_attention_descriptor"],expanded=True) and context_passes(full["gqa_attention_descriptor"],expanded=True)
                normal=contract["input_normalization_train_only_verified"] and cached_contract["input_normalization_train_only_verified"] and (uncached_contract is None or uncached_contract["input_normalization_train_only_verified"])
                preprocess=golden_fit_context_equal and contract["preprocessor_sha256"]==cached_contract["preprocessor_sha256"] and (uncached_contract is None or contract["preprocessor_sha256"]==uncached_contract["preprocessor_sha256"])
                rng_records.append({"path":"candidate_total_including_arithmetic_checks","before":initial_rng,"after":utilities.rng_identity(utilities.rng_state(torch))})
                row={"stage_index":stage,"parameter":problem.parameter_order[stage],"comparisons":comparisons,
                    "fresh_control_vs_saved_golden":golden_main_check,"stress_control_vs_saved_golden":golden_stress_check,
                    "stress_comparison":utilities.compare_predictions(stress_control,stress_candidate),"stress_gqa_execution":gqa_checks(stress_candidate,stage),
                    "component_qualification":component,"stress_component_qualification":stress_component,"outlier_checks":outliers,"stage0_shared_distribution":dedup,
                    "member_routes":candidates[1024]["member_routes"],"fallback_members":[5] if stage==10 else [],
                    "control_model_contract":contract,"resident_model_contract":cached_contract,"uncached_model_contract":uncached_contract,
                    "normalization_train_only_verified":bool(normal),"preprocessor_identity":bool(preprocess),"golden_fit_context_equal":golden_fit_context_equal,
                    "fresh_original_fit_context":fit_context,"prepared":prepared,
                    "prepared_state_unchanged":state==probe.prepared_state(engine.base),"all_cached_embedding_hashes":state["all_cached_embedding_hashes"],
                    "rng":rng_records,"rng_unchanged":all(v["before"]==v["after"] for v in rng_records),"execution_counters":dict(engine.gqa_counters),
                    "hybrid_counters":{k:v for k,v in engine.hybrid_counters.items() if k!="stages"},"cache_telemetry":engine.inspect_cache(),
                    "timings_seconds":{"fresh_control_fit_and_predict":control_seconds,"candidate_prepare":prepare_seconds}}
                row["passed"]=stage_passes(row);result["stage_results"].append(row);result["stage_coverage"].append(stage)
                engine.close();engine=None;result["elapsed_seconds"]=time.perf_counter()-start;write_json(root/"manifest.json",result)
                del baseline,control,stress_control,candidates,stress_candidate,main_arrays,stress_arrays
            else:result["status"]="complete_pending_restoration"
        result["compatibility_adapter"]=dict(active)
        require(active["original_alias_restored"] and active["sdk_sources_unchanged"],"CDF adapter restoration failed")
        require(guard_baseline_source()==original_source and execution.execution_source_sha256()==engine_sha and sha(__file__)==result["qualifier_source_sha256"],"qualification execution source changed")
        execution.require_frozen_sources();cache._load_sdk()
        require(sha(golden_path)==golden_sha and sha(control_proof)==control_evidence["sha256"],"reference manifest changed during qualification")
        passed=result["stage_coverage"]==list(range(21)) and all(row["passed"] for row in result["stage_results"])
        result.update(status="gqa_hybrid_screen_pass" if passed else "incomplete" if result["status"]=="incomplete" else "conditional_mismatch",
            qualified=passed,full21_gqa_hybrid_qualified=passed,full21_component_qualified=passed,full21_sampler_qualified=passed,
            failed_stage_indices=[row["stage_index"] for row in result["stage_results"] if not row["passed"]],sources_unchanged=True,
            elapsed_seconds=time.perf_counter()-start)
        write_json(root/"manifest.json",result);return result
    except BaseException as error:
        result.update(status="failed",qualified=False,error_type=type(error).__name__,error=str(error),elapsed_seconds=time.perf_counter()-start)
        if active is not None:result["compatibility_adapter"]=dict(active)
        write_json(root/"manifest.json",result);raise
    finally:
        if engine is not None:engine.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("banks","prefix-run","checkpoint","ledger","output","goldens","control-proof"):parser.add_argument("--"+name,required=True)
    parser.add_argument("--max-seconds",type=int,default=1050);result=qualify(**vars(parser.parse_args()))
    print(json.dumps({k:result[k] for k in ("schema","status","stage_coverage","failed_stage_indices","elapsed_seconds")}))
    if not result["qualified"]:raise SystemExit(2)


if __name__=="__main__":main()
