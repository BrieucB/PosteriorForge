#!/usr/bin/env python3
"""Fresh21-stage all-FP32 cached/native screen. Built with PriorLabs-TabPFN.

Exactly two arms, no FP16. Same frozen prior-latent inputs and fixtures as550.
Numeric-only failures are collected across all21; hard failures stop. No old
pass or fallback policy is inherited. No posterior sample/reference integration.
"""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,os,random,time
from types import SimpleNamespace
from contextlib import ExitStack
from pathlib import Path
import numpy as np
from posteriorforge.artifacts import sha256_file,write_json

SCHEMA="posteriorforge.prior_native_control_conditional_screen.v1"
NATIVE_BACKEND_SHA256="e5a2733e767b187e573315547ec9253267e43d1e3700596f61e97116c53f3256"
DATA_HELPER_SHA256="3ae723d4748f23af40b89b9508713e270730985b6a8b5d4ccca1959e60ae17f9"
FAILED_SCREEN_SHA256="550b91af4a031a3393f59146e1aa5360c75b0e716362ee727aeb1397af149593"
PREREG_SHA256='82b731edc9d57410e5343bb82833658931de728bd99940707908438dbbcdfc7b'
NATIVE_STAGES=(6,10)
ARM="candidate_FP32_vs_original_FP32"
STAGES=tuple(range(21))
QUERIES=("1","1024","stress")
THRESHOLDS={"normalization_absolute":1e-5,"max_cdf_absolute":2e-5,"max_row_total_variation":1e-4,"quantile_coordinate_span_fraction":2e-6}
GUARDS=("sources_unchanged","identical_latent_inputs","latent_target_bytes_unchanged","FP32_statistics_normalizers_borders",
 "target_statistics_preserved","preprocessor_identity_preserved","state_contract_preserved","immutable_native_state_preserved",
 "model_RNG_preserved","actual_dtype_witnesses","corrected_CDF_executed","aliases_restored")

def require(value,message):
 if not value:raise ValueError(message)

def load(name,digest):
 path=Path(__file__).with_name(name)
 require(isinstance(digest,str) and len(digest)==64 and sha256_file(path)==digest,"unfrozen/changed helper: "+name)
 s=importlib.util.spec_from_file_location("_prior_native_q_"+path.stem,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def data_helper():return load("qualify_prior_conditionals_v1.py",DATA_HELPER_SHA256)

def source_pins():
 from posteriorforge import execution_prior_native_v1 as native
 require(isinstance(NATIVE_BACKEND_SHA256,str) and native.source_sha256()==NATIVE_BACKEND_SHA256,"new native policy not frozen")
 native.require_sources();data=data_helper();base=data.source_pins()
 doc=Path(__file__).resolve().parents[1]/"docs/accuracy-prior-coordinate-native-control-v1.md"
 require(isinstance(PREREG_SHA256,str) and sha256_file(doc)==PREREG_SHA256,"new explicit native protocol changed")
 return {"qualifier":sha256_file(__file__),"execution":NATIVE_BACKEND_SHA256,"input_numeric_ancestry":DATA_HELPER_SHA256,
  "ancestral_sources":base,"failed_empty_cache_screen":FAILED_SCREEN_SHA256,"native_protocol":PREREG_SHA256}

def failed_screen_receipt(path,raw,transformed):
 require(sha256_file(path)==FAILED_SCREEN_SHA256,"exact failed empty-cache inventory required")
 d=json.loads(Path(path).read_bytes());data=data_helper();old=data.load("qualify_ss15_conditionals.py",data.SS15_QUALIFIER_SHA256)
 require(d["status"]=="prior_coordinates_matching_FP32_structural_or_hard_failure" and d["sources"]==data.source_pins()
  and d["raw_bindings"]==raw and d["transformed_bindings"]==transformed,"failed original screen/input custody differs")
 rows=d["stage_results"];require([r["stage_index"] for r in rows]==list(STAGES),"complete original numerical inventory required")
 for row in rows:
  derived=data.stage_decision(row,old)
  require(all(row[k]==v for k,v in derived.items()) and derived["common_guards_passed"] and derived["control_hard_validity_passed"],"original inventory has hard/custody failure")
 failed=[r["stage_index"] for r in rows if not r["matching_FP32_execution_passed"]]
 require(failed==list(NATIVE_STAGES),"explicit native correction selection differs; no adaptive stage changes")
 return {"passed_as_failure_inventory":True,"qualification_passed":False,"manifest_sha256":FAILED_SCREEN_SHA256,
  "predeclared_all8_native_stages":list(NATIVE_STAGES),"source_status_preserved":d["status"]}

def witness_guard(report,stage,native):
 if stage not in NATIVE_STAGES:
  data=data_helper();return data.witness_guard(report,native.CONTROL_MODE,stage)
 cached=native.cached
 require(report["mode"]==native.CONTROL_MODE and report["module_sha256"]==native.CACHED_SHA256
  and report["codec_sha256"]==native.CODEC_SHA256 and report["fallback_map"]=={}
  and report["aliases_and_flags_restored"] is True,"original native precision owner/source/restoration differs")
 require(report["counters"].get("witness_histogram_overflow",0)==0
  and sum(w["count"] for w in report["witnesses"])==report["counters"]["qkv_calls"]==report["counters"]["attention_calls"]
  and report["counters"].get("expanded_views",0)==0,"native QKV/attention/original-view coverage differs")
 require(not any(w["phase"]=="prefill" for w in report["witnesses"]),"stock original preparation cannot claim neural cached prefill")
 cached._phase_rows(report,stage,"query",range(8),native.CONTROL_MODE)
 return True

def stage_decision(row,old):
 require(set(row["comparisons"])==set(QUERIES),"all N1/N1024/stress comparisons required")
 hard=set(row["guards"])==set(GUARDS) and all(v is True for v in row["guards"].values())
 require(set(row["hard_predictions"])=={"original_"+k for k in QUERIES}|{"candidate_"+k for k in QUERIES},"both arms/three complete query outputs required")
 hard=hard and all(old.probability_hard(v) for v in row["hard_predictions"].values())
 for c in row["comparisons"].values():
  require(c["passed"] is old.comparison_decision(c),"original numeric predicate relabelled")
  hard=hard and old.comparison_hard(c)
 return {"hard_validity_passed":bool(hard),"matching_FP32_execution_passed":bool(hard and all(v["passed"] for v in row["comparisons"].values()))}

def validate_screen(d,*,expected_sources,raw_bindings,transformed_bindings,coordinate_policy,native_policy,inventory_replay):
 from posteriorforge import execution_prior_native_v1 as native
 data=data_helper();old=data.load("qualify_ss15_conditionals.py",data.SS15_QUALIFIER_SHA256)
 expected={"schema":SCHEMA,"status":"prior_native_control_screen_pass","sources":expected_sources,"raw_bindings":raw_bindings,
  "transformed_bindings":transformed_bindings,"coordinate_policy":coordinate_policy,"native_execution_policy":native_policy,
  "codec_inventory":inventory_replay,"thresholds":THRESHOLDS,"row_counts":[1,1024],"stress_rows":8,"FP16_activated":False,
  "posterior_vectors_generated":0,"reference_integrations":0,"old_numerical_pass_inherited":False,
  "sources_unchanged":True,"global_RNG_preserved":True}
 for k,v in expected.items():require(d.get(k)==v,"new original-stage screen binding differs: "+k)
 require(d["completed_grid"]["passed"] and d["completed_grid"]["manifest_sha256"]==data.COMPLETED_GRID_SHA256,"closed baseline24 receipt differs")
 require(d["failed_empty_cache_inventory"]["manifest_sha256"]==FAILED_SCREEN_SHA256
  and d["failed_empty_cache_inventory"]["qualification_passed"] is False,"failed original screen was promoted")
 rows=d["stage_results"];require([r["stage_index"] for r in rows]==list(STAGES),"all21 fresh comparisons required")
 for row in rows:
  j=row["stage_index"];rule=native.stage_policy(j);b=row["mask_inputs"];t=b["training"]
  require(row["route"]==native.route(j) and row["target_codec_coordinate"]==rule["codec_coordinate"] and t["mask"]==rule
   and t["original_latent_target"]==t["prepared_latent_target"]==b["actual_latent_target"]
   and t["masked_training"]==b["actual_training"] and t["same_fixed_mask_for_training_query"] and t["target_bytes_unchanged"],"actual latent predictors/target/route differ")
  calculated=stage_decision(row,old);require(all(row[k]==v for k,v in calculated.items()) and calculated["matching_FP32_execution_passed"],"fresh exact-FP32 structural/hard failure")
  require(row["CDF_call_counts"]=={"original_FP32":24,"candidate_FP32":24} and row["corrected_CDF_calls"]==48,"actual two-arm 3x8 CDF-call coverage differs")
  require(row["RNG_identity_before"]==row["RNG_identity_after"],"actual model RNG changed")
  witness_guard(row["precision_session"],j,native)
  if j in NATIVE_STAGES:
   require(row["preparation_witness"]["native_neural_prefill_calls"]==0
    and row["state_scope"]=="native cold fit guarded; warmed baseline after planned N1, then N1024/stress"
    and len(row["immutable_native_snapshots"])==4 and all(s==row["immutable_native_snapshots"][0] for s in row["immutable_native_snapshots"])
    and row["immutable_native_snapshots"][0]==row["original_control_immutable_state"]
    and set(row["native_warmed_contracts"])=={"1024","stress"}
    and all(s==row["model_after_planned_N1"] for s in row["native_warmed_contracts"].values()),
    "native cold/warm or immutable-state claim differs")
  else:require(row["state_scope"]=="unchanged88 prepared-before/after cached contract","cached19 state rule changed")
 footer=d["compatibility_adapter"]
 require(footer["patched_call_count"]==1008 and footer["original_alias_restored"] and footer["sdk_sources_unchanged"]
  and footer["executed_corrected_cdf_code_sha256"]==footer["corrected_cdf_code_sha256"]!=footer["original_cdf_code_sha256"],"actual corrected-CDF source/restoration differs")
 require(d["matching_FP32_execution_passed_all21"] is True and not d["FP32_failed_stages"],"all21 exact-FP32 comparisons required")
 return {"passed":True,"matching_FP32_execution_passed_all21":True,"fresh_stages":21,"native_all8_stages":[6,10],
  "cached_stages":19,"FP16_activated":False,"posterior_equivalence_claimed":False,"scope":"exact saved finite-prefix screen only"}

def run(*,banks,prefix_run,checkpoint,codec_inventory,codec_reference_inventory,codec_cpu_receipt,completed_grid_audit,failed_screen,output,ledger,max_seconds=1050):
 from posteriorforge.banks import derive_seed
 from posteriorforge.inference import CHECKPOINT_SHA256,frozen_regressor_settings
 from posteriorforge.inference_cdf import guard_baseline_source
 from posteriorforge import execution_cache as cache,execution_prior_native_v1 as native
 pins=source_pins();data=data_helper();completion=data.grid_receipt(completed_grid_audit)
 inventory,inventory_replay=data.inventory_receipt(codec_inventory,codec_reference_inventory,codec_cpu_receipt)
 runtime_audit={};theta,stats,query,prefixes,raw,transformed,custody=data.input_bindings(banks,prefix_run,inventory,runtime_audit=runtime_audit)
 failed=failed_screen_receipt(failed_screen,raw,transformed)
 require(type(max_seconds) is int and 600<=max_seconds<=2250 and sha256_file(checkpoint)==CHECKPOINT_SHA256,"bounded screen and immutable checkpoint required")
 old=data.load("qualify_ss15_conditionals.py",data.SS15_QUALIFIER_SHA256);utility=old.load("qualify_execution_cache.py",old.UTILITY_SHA);base=old.load("characterize_precision_approximation.py",old.BASE_SHA)
 require(THRESHOLDS==data.THRESHOLDS==old.THRESHOLDS==utility.THRESHOLDS,"numeric limits changed")
 from posteriorforge.reference_grid import _require_budgeted_gpu
 budget=_require_budgeted_gpu(ledger);require(budget["gpus"]==1,"one reserved GPU required")
 started=time.perf_counter();deadline=started+max_seconds;output=Path(output);output.mkdir(parents=True,exist_ok=False)
 policy=native.policy_metadata();coordinates=native.cached.policy_metadata();baseline=guard_baseline_source()
 report={"schema":SCHEMA,"status":"running","attribution":"Built with PriorLabs-TabPFN","sources":pins,"raw_bindings":raw,"transformed_bindings":transformed,
  "coordinate_policy":coordinates,"native_execution_policy":policy,"codec_inventory":inventory_replay,"completed_grid":completion,
  "failed_empty_cache_inventory":failed,"runtime_input_audit":runtime_audit,"prefix_custody":custody,"thresholds":THRESHOLDS,
  "row_counts":[1,1024],"stress_rows":8,"FP16_activated":False,"posterior_vectors_generated":0,"reference_integrations":0,
  "old_numerical_pass_inherited":False,"stage_results":[],"matching_FP32_execution_passed_all21":False,"FP32_failed_stages":[],
  "numeric_failure_policy":"collect all21 numerical tolerance failures; stop hard integrity/source/state/validity failure",
  "budget_binding":budget,"baseline_execution_source":baseline}
 active=None;write_json(output/"manifest.json",report)
 try:
  import torch
  from posteriorforge.compat_cdf import finite_cdf_compatibility
  require(torch.cuda.is_available() and torch.cuda.device_count()==1,"one visible GPU required")
  require(torch.get_default_dtype()==torch.float32 and torch.get_float32_matmul_precision()=="highest" and not torch.backends.cuda.matmul.allow_tf32 and not torch.is_autocast_enabled("cuda"),"strict outer FP32 flags required")
  seed=derive_seed(20261111,"prior-coordinate-fixed-conditionals-v1",case=2);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
  torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS","16")));initial_rng=utility.rng_identity(utility.rng_state(torch));report["seed"]=seed
  with finite_cdf_compatibility() as active:
   for j in STAGES:
    require(time.perf_counter()+25<deadline,"bounded deadline, incomplete cannot qualify")
    main,stress,noise=data.fixed_queries(query,prefixes,theta,j,native);mt,ms,mq,sq,binding=data.masked_inputs(theta,stats,main,stress,j,native)
    row={"stage_index":j,"route":native.route(j),"target_codec_coordinate":native.stage_policy(j)["codec_coordinate"],"stress_noise_stage_indices":noise,
     "mask_inputs":binding,"comparisons":{},"hard_predictions":{},"guards":{}};report["stage_results"].append(row);write_json(output/"manifest.json",report)
    before_rng=utility.rng_identity(utility.rng_state(torch));before_cdf=active["patched_call_count"]
    settings=frozen_regressor_settings(checkpoint);settings["inference_precision"]=torch.float32;_,sdk,_=cache._load_sdk();control=sdk.TabPFNRegressor(**settings)
    train=np.concatenate((ms,mt[:,:j]),axis=1);target=mt[:,j]
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):control.fit(train,target)
    queries={"1":mq[:1],"1024":mq,"stress":sq};refs={}
    for key,features in queries.items():
     with torch.random.fork_rng(devices=[torch.cuda.current_device()]):prediction=cache.capture_corrected_prediction(control,features,cdf_descriptor=active)
     refs[key]=prediction;row["hard_predictions"]["original_"+key]=old.hard_prediction(prediction,base,utility)
    original_model=base.model_summary(control,utility);row["original_model"]=original_model;original_calls=active["patched_call_count"]-before_cdf
    with native.PriorNativeSession() as session:
     with ExitStack() as cleanup:
      engine=native.PriorNativeStageExecutor(checkpoint,cdf_descriptor=active);cleanup.callback(engine.close)
      prepared=engine.prepare(mt,ms,j);row["prepared"]=prepared;row["preparation_witness"]=native.preparation_witness(session.report(),j,prepared)
      cold=base.model_summary(engine.regressor,utility);row["model_cold"]=cold;anchor=cold;normalizers=old.state_guard(original_model) and old.state_guard(cold)
      immutable=[];immutable_ok=True;warmed_state_equal=True;row["native_warmed_contracts"]={}
      if j in NATIVE_STAGES:
       immutable.append(native.immutable_native_snapshot(engine))
       row["original_control_immutable_state"]=native.immutable_native_snapshot(SimpleNamespace(regressor=control))
       require(immutable[0]==row["original_control_immutable_state"],"native cold immutable parameters/targets/configs/fitted context differ from original control")
       require(normalizers and all(cold[k]==original_model[k] for k in ("target_mean","target_sd","temperature"))
        and cold["contract"]["preprocessor_sha256"]==original_model["contract"]["preprocessor_sha256"],"native cold settings/target/borders/train-only contract invalid")
       require(utility.rng_identity(utility.rng_state(torch))==before_rng,"native initialization changed default RNG")
      for key,features in queries.items():
       candidate=old.capture_components(engine,features,torch);row["hard_predictions"]["candidate_"+key]=old.hard_prediction(candidate,base,utility)
       row["comparisons"][key]=old.compare(refs[key],candidate,utility)
       observed=base.model_summary(engine.regressor,utility);normalizers=normalizers and old.state_guard(observed)
       if j in NATIVE_STAGES:
        immutable.append(native.immutable_native_snapshot(engine));immutable_ok=immutable_ok and immutable[-1]==immutable[0]
        require(immutable_ok and old.state_guard(observed),"native immutable values/dtype/context invalid after planned query")
        require(utility.rng_identity(utility.rng_state(torch))==before_rng,"native planned query changed default RNG")
        require(engine.child.hybrid_counters["original_preprocessing_lists_restored"] and engine.child.hybrid_counters["original_translator_alias_restored"]
         and engine.child.hybrid_counters["native_query_view_setting_restored"],"native query did not restore preprocessing/translator/view state")
        if key=="1":anchor=observed;row["model_after_planned_N1"]=observed
        else:
         row["native_warmed_contracts"][key]=observed;warmed_state_equal=warmed_state_equal and observed==anchor
         require(warmed_state_equal,"native fitted/context warm contract changed after planned query")
      after=base.model_summary(engine.regressor,utility);row["model_after"]=after
      state_equal=warmed_state_equal and anchor==after
      row["immutable_native_snapshots"]=immutable
      row["warm_contract_scope"]={"protected":"all observed frozen utility model_contract fields after planned N1, including normalization/outlier/train-feature/cached-embedding/number-of-used-features",
       "all_observed_fields_claimed_query_independent":False,"number_of_used_features_source_scope":"SDK _fit uses full supplied x; field remains protected observationally, not exempted",
       "ephemeral_forward_query_tensors_inventoried":False,"native_KV_cache_used":False,"cold_to_warm_equality_claimed":False} if j in NATIVE_STAGES else {"protected":"unchanged88 prepared before/after contract"}
      row["state_scope"]="native cold fit guarded; warmed baseline after planned N1, then N1024/stress" if j in NATIVE_STAGES else "unchanged88 prepared-before/after cached contract"
      target_equal=all(cold[k]==original_model[k]==after[k] for k in ("target_mean","target_sd","temperature"))
      preprocessing_equal=cold["contract"]["preprocessor_sha256"]==original_model["contract"]["preprocessor_sha256"]==after["contract"]["preprocessor_sha256"]
    row["precision_session"]=session.report();witness_guard(row["precision_session"],j,native)
    row["CDF_call_counts"]={"original_FP32":original_calls,"candidate_FP32":active["patched_call_count"]-before_cdf-original_calls}
    row["corrected_CDF_calls"]=active["patched_call_count"]-before_cdf
    require(row["CDF_call_counts"]=={"original_FP32":24,"candidate_FP32":24},"actual two-arm head coverage differs")
    after_rng=utility.rng_identity(utility.rng_state(torch));row["RNG_identity_before"]=before_rng;row["RNG_identity_after"]=after_rng
    row["guards"]={k:True for k in GUARDS};row["guards"].update(sources_unchanged=source_pins()==pins,
     identical_latent_inputs=data.masked_inputs(theta,stats,main,stress,j,native)[4]==binding,
     latent_target_bytes_unchanged=target.tobytes()==theta[:,j].tobytes(),FP32_statistics_normalizers_borders=normalizers,
     target_statistics_preserved=target_equal,preprocessor_identity_preserved=preprocessing_equal,state_contract_preserved=state_equal,
     immutable_native_state_preserved=immutable_ok,model_RNG_preserved=before_rng==after_rng,
     corrected_CDF_executed=active["executed_corrected_cdf_code_sha256"]==active["corrected_cdf_code_sha256"],aliases_restored=row["precision_session"]["aliases_and_flags_restored"])
    row.update(stage_decision(row,old));write_json(output/"manifest.json",report)
    del control,refs,candidate,prediction,engine;torch.cuda.synchronize()
    if not row["hard_validity_passed"]:break
  report["compatibility_adapter"]=dict(active)
  require(active["original_alias_restored"] and active["sdk_sources_unchanged"] and active["patched_call_count"]==48*len(report["stage_results"]),"actual CDF source/restoration/coverage differs")
  require(utility.rng_identity(utility.rng_state(torch))==initial_rng,"global model RNG changed")
  require(source_pins()==pins and native.policy_metadata()==policy and guard_baseline_source()==baseline,"source/policy changed")
  require(sha256_file(failed_screen)==FAILED_SCREEN_SHA256 and data.grid_receipt(completed_grid_audit)==completion,"closed selection/completion changed")
  require(data.inventory_receipt(codec_inventory,codec_reference_inventory,codec_cpu_receipt)[1]==inventory_replay,"codec CPU inputs changed")
  rows=report["stage_results"];complete=[r["stage_index"] for r in rows]==list(STAGES);failed_stages=[r["stage_index"] for r in rows if not r["matching_FP32_execution_passed"]]
  report.update(matching_FP32_execution_passed_all21=bool(complete and not failed_stages),FP32_failed_stages=failed_stages,sources_unchanged=True,global_RNG_preserved=True)
  report["status"]="prior_native_control_screen_pass" if report["matching_FP32_execution_passed_all21"] else "prior_native_control_structural_or_hard_failure"
  if report["status"]=="prior_native_control_screen_pass":report["replay"]=validate_screen(report,expected_sources=pins,raw_bindings=raw,transformed_bindings=transformed,coordinate_policy=coordinates,native_policy=policy,inventory_replay=inventory_replay)
 except BaseException as error:
  report.update(status="failed_or_incomplete_prior_native_screen",error_type=type(error).__name__,error=str(error))
  if active is not None:report["compatibility_adapter"]=dict(active)
  raise
 finally:report["elapsed_seconds"]=time.perf_counter()-started;write_json(output/"manifest.json",report)
 return report

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for key in ("banks","prefix-run","checkpoint","codec-inventory","codec-reference-inventory","codec-cpu-receipt","completed-grid-audit","failed-screen","output","ledger"):p.add_argument("--"+key,required=True)
 p.add_argument("--max-seconds",type=int,default=1050);report=run(**vars(p.parse_args()))
 print(json.dumps({k:report.get(k) for k in ("status","matching_FP32_execution_passed_all21","FP32_failed_stages","elapsed_seconds")},sort_keys=True))
if __name__=="__main__":main()
