#!/usr/bin/env python3
"""Fresh sigma-first fixed-conditional execution screen. Built with PriorLabs-TabPFN.

Zero posterior samples. Original numeric limits are unchanged. All21 stages get fresh original FP32 controls. No anchor is reused.
FP16 drift is retained as approximation data, never called FP32 qualification.
"""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,os,random,time
from pathlib import Path
import numpy as np
from posteriorforge.artifacts import sha256_file,write_json
from posteriorforge.science.analytical_benchmark_v2 import array_descriptor

SCHEMA="posteriorforge.ss15_sigma_first_conditional_screen.v1"
ORDER_SHA="d48d1638ef9e540bdd629d321fba657af4db757700b1092fc58fb9ded1f79ad2"
SS15_QUALIFIER_SHA="0c3dd7274d6741bba2ba0ae2744542865f48b2f887a6831e1d2aa6bce47f78c0"
ENGINE_SHA="1732f7269286f1629f6e09b1b7ea14cfadbaea84cdae9ee4ab5221cf5b78fbbb"
THRESHOLDS={"normalization_absolute":1e-5,"max_cdf_absolute":2e-5,
 "max_row_total_variation":1e-4,"quantile_coordinate_span_fraction":2e-6}
FALLBACK_MAP={"10":[5],"20":[4]}
STAGES=tuple(range(21))
ORDER=(0,1,2,6,3,4,5,10,7,8,9,14,11,12,13,16,15,18,17,20,19)
NOISE_STAGE_INDICES=(3,7,11,15,17,19)
ROW_COUNTS=(1,1024)
ARMS=("cached_FP32_vs_original_FP32","cached_FP16_vs_original_FP32")
GUARDS=("sources_unchanged","identical_masked_training_inputs","original_targets_unchanged",
 "same_train_query_mask","FP32_statistics_normalizers_borders","target_statistics_preserved",
 "cached_contract_preserved","model_RNG_preserved","actual_dtype_witnesses","corrected_CDF_executed","aliases_restored")

def require(value,message):
 if not value:raise ValueError(message)

def load(name,digest):
 path=Path(__file__).with_name(name)
 require(sha256_file(path)==digest,"frozen helper source differs: "+name)
 spec=importlib.util.spec_from_file_location("_sigma_first_screen_"+path.stem,path)
 m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def source_pins():
 from posteriorforge import execution_ss15_sigma_first as graph,execution_ss15 as execution
 require(isinstance(ORDER_SHA,str) and len(ORDER_SHA)==64 and graph.source_sha256()==ORDER_SHA,
  "sigma-first order source must be frozen before screen execution")
 graph.require_sources();require(execution.source_sha256()==ENGINE_SHA,"frozen1732 execution changed")
 old=load("qualify_ss15_conditionals.py",SS15_QUALIFIER_SHA)
 return {"producer":sha256_file(__file__),"order_adapter":ORDER_SHA,"execution":ENGINE_SHA,
  "numeric_comparison":SS15_QUALIFIER_SHA,"utility":old.UTILITY_SHA,
  "hard_predictions":old.BASE_SHA}

def masked_inputs(theta,statistics,main,stress,stage,graph):
 """Bind actual full-width features and unchanged target, not just the mask."""
 mt,ms=graph.mask_training(theta,statistics,stage)
 mq=graph.mask_query(main,stage);sq=graph.mask_query(stress,stage)
 binding=graph.training_bindings(theta,statistics,mt,ms,stage)
 original=np.concatenate((statistics,theta[:,:stage]),axis=1)
 expected=original.copy();expected[:,graph.stage_policy(stage)["feature_columns_zeroed"]]=np.float32(0)
 actual=np.concatenate((ms,mt[:,:stage]),axis=1)
 require(actual.tobytes()==expected.tobytes() and theta[:,stage:].tobytes()==mt[:,stage:].tobytes(),
  "same fixed full-width mask/original labels must feed every arm")
 for before,after in ((main,mq),(stress,sq)):
  wanted=before.copy();wanted[:,graph.stage_policy(stage)["feature_columns_zeroed"]]=np.float32(0)
  require(after.tobytes()==wanted.tobytes(),"actual masked query differs from fixed policy")
 return mt,ms,mq,sq,{"training":binding,
  "unmasked_queries":{"main":array_descriptor(main),"stress":array_descriptor(stress)},
  "actual_masked_queries":{"main":array_descriptor(mq),"stress":array_descriptor(sq)},
  "actual_training_predictors":array_descriptor(actual),"actual_target":array_descriptor(mt[:,stage])}

def fixed_ordered_queries(observed,prefixes,theta,stage,graph,old,bounds):
 """Stress the retained own-block sigma prefix, using ORDERED noise indices."""
 kept=graph.stage_policy(stage)["ordered_prefix_indices_kept"]
 eligible=[index for index in NOISE_STAGE_INDICES if index<stage and index in kept]
 main,stress=old.fixed_queries(observed,prefixes,theta,stage,eligible,bounds)
 return main,stress,eligible

def stage_decision(row,old):
 comparisons=row["comparisons"]
 require(set(comparisons)==set(ARMS),"two matching masked execution arms required")
 hard=set(row.get("guards",{}))==set(GUARDS) and all(v is True for v in row["guards"].values())
 hard=hard and bool(row.get("hard_predictions")) and all(old.probability_hard(v) for v in row["hard_predictions"].values())
 for arm in ARMS:
  require(set(comparisons[arm])=={"1","1024","stress"},"N1,N1024,stress8 observed coverage required")
  for record in comparisons[arm].values():
   require(record["passed"] is old.comparison_decision(record),"saved numeric strict decision differs")
   hard=hard and old.comparison_hard(record)
 return {"hard_validity_passed":bool(hard),
  "matching_FP32_execution_passed":bool(hard and all(v["passed"] for v in comparisons[ARMS[0]].values())),
  "FP16_hard_validity_passed":bool(hard),
  "FP16_strict_passed":bool(hard and all(v["passed"] for v in comparisons[ARMS[1]].values())),
  "FP16_qualified_as_FP32":False,"posterior_equivalence_claimed":False}

def validate_screen(document,*,expected_sources,raw_bindings,transformed_bindings,order_policy):
 old=load("qualify_ss15_conditionals.py",SS15_QUALIFIER_SHA)
 expected={"schema":SCHEMA,"status":"sigma_first_conditional_screen_pass","sources":expected_sources,
  "raw_bindings":raw_bindings,"transformed_bindings":transformed_bindings,"order_policy":order_policy,
  "thresholds":THRESHOLDS,"row_counts":list(ROW_COUNTS),"stress_rows":8,"fallback_map":FALLBACK_MAP,
  "posterior_vectors_generated":0,"reference_integrations":0,"sources_unchanged":True,"global_RNG_preserved":True,
  "FP16_qualified_as_FP32":False,"posterior_equivalence_claimed":False,
  "stage_to_physical_order":list(ORDER),"noise_stage_indices":list(NOISE_STAGE_INDICES),
  "anchor_reused":False,"prior_target_order_qualification_inherited":False}
 for key,value in expected.items():require(document.get(key)==value,"sigma-first screen binding differs: "+key)
 rows=document["stage_results"];require([r["stage_index"] for r in rows]==list(STAGES),"all21 fresh ordered stages required")
 for row in rows:
  binding=row["mask_inputs"];training=binding["training"];stage=row["stage_index"]
  require(row["target_physical_index"]==ORDER[stage]
   and row["target_parameter"]==raw_bindings["parameter_order"][ORDER[stage]]
   and training["target_physical_index"]==ORDER[stage],"ordered stage-to-physical target binding differs")
  require(row["stress_noise_stage_indices"]==[i for i in NOISE_STAGE_INDICES
   if i<stage and i in order_policy["stages"][stage]["ordered_prefix_indices_kept"]],
   "stress bounds were applied to a removed or wrong noise prefix")
  require(training["mask"]==order_policy["stages"][stage]
   and training["policy_sha256"]==order_policy["policy_sha256"]
   and training["original_target"]==training["prepared_target"]==binding["actual_target"]
   and training["prepared_training_predictors"]==binding["actual_training_predictors"]
   and training["training_query_same_mask"] is True and training["target_bytes_unchanged"] is True,
   "actual masked features/target binding differs")
  calculated=stage_decision(row,old)
  require(all(row[k]==v for k,v in calculated.items()) and calculated["hard_validity_passed"]
   and calculated["matching_FP32_execution_passed"],"fresh masked structural/hard comparison failed")
  require(row["corrected_CDF_calls"]==72,"actual9x8 original/cached head calls required")
  for label,mode in (("FP32",_execution_modes()[0]),("FP16",_execution_modes()[1])):
   old.witness_guard(row["precision_sessions"][label],mode=mode,stage=row["stage_index"],fallback_map=FALLBACK_MAP)
 descriptor=document["compatibility_adapter"]
 require(descriptor["patched_call_count"]==72*21 and descriptor["original_alias_restored"] is True
  and descriptor["sdk_sources_unchanged"] is True
  and descriptor["executed_corrected_cdf_code_sha256"]==descriptor["corrected_cdf_code_sha256"]
  and descriptor["corrected_cdf_code_sha256"]!=descriptor["original_cdf_code_sha256"],"actual CDF coverage/restoration differs")
 failed=[r["stage_index"] for r in rows if not r["FP16_strict_passed"]]
 require(document["FP16_strict_failed_stages"]==failed and document["matching_FP32_execution_passed_all21"] is True
  and document["FP16_hard_validity_passed_all21"] is True,"full coverage or approximation flags relabelled")
 return {"passed":True,"fresh_stages":21,"anchor_reused":False,
  "matching_FP32_execution_passed_all21":True,"FP16_hard_validity_passed_all21":True,
  "FP16_strict_failed_stages":failed,"FP16_qualified_as_FP32":False,"posterior_equivalence_claimed":False,
  "scope":"fixed prefixes only; no posterior-law or accuracy claim"}

def _execution_modes():
 from posteriorforge import execution_ss15 as adapter
 return adapter.CONTROL_MODE,adapter.FP16_MODE

def run(*,banks,prefix_run,checkpoint,output,ledger,max_seconds=1050):
 from contextlib import ExitStack
 from posteriorforge.banks import load_context,load_heldout,derive_seed
 from posteriorforge.science.analytical_benchmark_v2 import build_problem
 from posteriorforge.inference import CHECKPOINT_SHA256,frozen_regressor_settings,support_mask
 from posteriorforge.inference_cdf import guard_baseline_source
 from posteriorforge.reference_grid import _require_budgeted_gpu
 from posteriorforge.compat_cdf import finite_cdf_compatibility
 from posteriorforge import execution_cache as cache,execution_ss15 as adapter,representation_ss15 as rep
 from posteriorforge import execution_ss15_sigma_first as graph
 require(type(max_seconds) is int and 600<=max_seconds<=2250,"bounded600..2250s fixed-conditional screen required")
 pins=source_pins();old=load("qualify_ss15_conditionals.py",SS15_QUALIFIER_SHA)
 utility=old.load("qualify_execution_cache.py",old.UTILITY_SHA);base=old.load("characterize_precision_approximation.py",old.BASE_SHA)
 require(utility.THRESHOLDS==old.THRESHOLDS==THRESHOLDS,"original numeric limits changed")
 budget=_require_budgeted_gpu(ledger);require(budget["gpus"]==1,"one reserved GPU required")
 require(sha256_file(checkpoint)==CHECKPOINT_SHA256,"immutable checkpoint differs")
 context=load_context(banks,20261111,4096);observed=load_heldout(banks,0);raw_query=observed.obs.reshape(1,33)
 physical_theta,stats,query,raw,transformed=old.transformed_inputs(context.theta,context.obs,raw_query,rep);problem=build_problem()
 raw.update(checkpoint_sha256=CHECKPOINT_SHA256,scientific_fingerprint=problem.scientific_fingerprint,parameter_order=list(problem.parameter_order))
 theta=graph.order_theta(physical_theta)
 require(tuple(graph.ORDER)==ORDER and graph.restore_physical(theta).tobytes()==physical_theta.tobytes(),"exact declared order/inverse required")
 initial=old.model_input_descriptors(theta,stats,query)
 prefix_run=Path(prefix_run);pm=prefix_run/"manifest.json";pp=prefix_run/"samples.npz"
 require(sha256_file(pm)=="a0114fd74721cf9bed31e9131045e841f262db123c8f849e5cb3f15814a14f4b"
  and sha256_file(pp)=="27e51ad1b1ee615fc921ee4f5c237155139304da3e7c193f7781a389cee6d53e",
  "exact frozen original50000 prefix cohort required")
 with np.load(pp,allow_pickle=False) as saved:
  require(saved.files==["samples"],"existing complete physical prefixes only");prefix_values=saved["samples"].copy()
 require(prefix_values.shape==(50000,21) and support_mask(prefix_values).all(),"existing50000 supported physical prefixes required")
 positions=np.linspace(0,49999,1024,dtype=int)
 prefixes=graph.order_theta(prefix_values[positions].astype(np.float32))
 noise=[ORDER.index(problem.parameter_order.index(n)) for n in problem.prior.noise_names]
 require(tuple(noise)==NOISE_STAGE_INDICES,"stress noise bounds must use ordered stage indices")
 started=time.perf_counter();deadline=started+max_seconds;output=Path(output);output.mkdir(parents=True,exist_ok=False)
 source=guard_baseline_source();policy=graph.policy_metadata()
 report={"schema":SCHEMA,"status":"running","attribution":"Built with PriorLabs-TabPFN","sources":pins,
  "raw_bindings":raw,"transformed_bindings":transformed,"order_policy":policy,"thresholds":THRESHOLDS,
  "row_counts":list(ROW_COUNTS),"stress_rows":8,"fallback_map":FALLBACK_MAP,
  "stage_to_physical_order":list(ORDER),"noise_stage_indices":list(NOISE_STAGE_INDICES),
  "ordered_model_inputs":initial,"ordered_fixed_prefixes":array_descriptor(prefixes),
  "physical_model_inputs":old.model_input_descriptors(physical_theta,stats,query),"stage_results":[],"posterior_vectors_generated":0,"reference_integrations":0,
  "prefix_manifest_sha256":sha256_file(pm),"prefix_samples_sha256":sha256_file(pp),"fixed_prefix_row_indices":positions.tolist(),
  "matching_FP32_execution_passed_all21":False,"FP16_hard_validity_passed_all21":False,
  "FP16_strict_failed_stages":[],"FP16_qualified_as_FP32":False,"posterior_equivalence_claimed":False,
  "anchor_reused":False,"prior_target_order_qualification_inherited":False,"failfast_on_matching_FP32_structural_failure":True,
  "budget_binding":budget,"baseline_execution_source":source}
 write_json(output/"manifest.json",report);active=None
 try:
  import torch
  require(torch.cuda.is_available() and torch.cuda.device_count()==1,"one visible GPU required")
  require(torch.get_default_dtype()==torch.float32 and torch.get_float32_matmul_precision()=="highest"
   and not torch.backends.cuda.matmul.allow_tf32 and not torch.is_autocast_enabled("cuda"),"strict outer FP32 flags required")
  seed=derive_seed(20261111,"sigma-first-SS15-fixed-conditionals",case=0)
  random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
  torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS","16")))
  initial_rng=utility.rng_identity(utility.rng_state(torch));report.update(seed=seed,RNG_role="sigma-first-SS15-fixed-conditionals")
  with finite_cdf_compatibility() as active:
   for stage in STAGES:
    require(time.perf_counter()+25<deadline,"bounded deadline; incomplete coverage cannot qualify")
    main,stress,stress_noise=fixed_ordered_queries(query,prefixes,theta,stage,graph,old,problem.prior.bounds)
    mt,ms,mq,sq,binding=masked_inputs(theta,stats,main,stress,stage,graph)
    row={"stage_index":stage,"stress_noise_stage_indices":stress_noise,"target_physical_index":ORDER[stage],"target_parameter":problem.parameter_order[ORDER[stage]],"mask_inputs":binding,"comparisons":{a:{} for a in ARMS},"hard_predictions":{},"guards":{}}
    report["stage_results"].append(row);report["active_stage"]=stage;write_json(output/"manifest.json",report)
    before_rng=utility.rng_identity(utility.rng_state(torch));before_cdf=active["patched_call_count"]
    settings=frozen_regressor_settings(checkpoint);settings["inference_precision"]=torch.float32
    _,sdk,_=cache._load_sdk();control=sdk.TabPFNRegressor(**settings)
    train=np.concatenate((ms,mt[:,:stage]),axis=1);target=mt[:,stage]
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):control.fit(train,target)
    queries={"1":mq[:1],"1024":mq,"stress":sq};references={}
    for key,features in queries.items():
     with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
      prediction=cache.capture_corrected_prediction(control,features,cdf_descriptor=active)
     references[key]=prediction;row["hard_predictions"]["original_FP32_"+key]=old.hard_prediction(prediction,base,utility)
    row["original_FP32_model"]=base.model_summary(control,utility)
    target_equal=True;normalizers=True;state_equal=True;preprocessors_equal=True;session_reports={}
    for mode,label in ((adapter.CONTROL_MODE,"FP32"),(adapter.FP16_MODE,"FP16")):
     with adapter.SS15PrecisionSession(mode,fallback_map={10:[5],20:[4]}) as session:
      with ExitStack() as cleanup:
       engine=adapter.SS15StageExecutor(checkpoint,cdf_descriptor=active,query_batch_size=1024,fallback_map={10:[5],20:[4]})
       cleanup.callback(engine.close);prepared=engine.prepare(mt,ms,stage)
       model=old.cached_model_summary(engine,base,utility);row[label+"_prepared"]=prepared;row[label+"_model_before"]=model
       normalizers=normalizers and old.state_guard(model)
       target_equal=target_equal and all(model[k]==row["original_FP32_model"][k] for k in ("target_mean","target_sd","temperature"))
       preprocessors_equal=preprocessors_equal and model["contract"]["preprocessor_sha256"]==row["original_FP32_model"]["contract"]["preprocessor_sha256"]
       for key,features in queries.items():
        candidate=old.capture_components(engine,features,torch)
        row["hard_predictions"]["cached_"+label+"_"+key]=old.hard_prediction(candidate,base,utility)
        row["comparisons"][ARMS[0] if label=="FP32" else ARMS[1]][key]=old.compare(references[key],candidate,utility)
       after=old.cached_model_summary(engine,base,utility);row[label+"_model_after"]=after
       state_equal=state_equal and model==after;normalizers=normalizers and old.state_guard(after)
     session_reports[label]=session.report()
     old.witness_guard(session_reports[label],mode=mode,stage=stage,fallback_map=FALLBACK_MAP)
    row["precision_sessions"]=session_reports;row["corrected_CDF_calls"]=active["patched_call_count"]-before_cdf
    require(row["corrected_CDF_calls"]==72,"actual9x8 corrected head coverage differs")
    after_rng=utility.rng_identity(utility.rng_state(torch))
    row["RNG_identity_before"]=before_rng;row["RNG_identity_after"]=after_rng
    row["cached_contract_evidence_scope"]={"KV_contents_hashed":False,"large_embedding_bytes_hashed":False,
     "scope":"frozen model metadata/normalizers/configuration before-after; no retention proof"}
    row["guards"]={k:True for k in GUARDS}
    row["guards"].update(sources_unchanged=source_pins()==pins,
     identical_masked_training_inputs=old.model_input_descriptors(theta,stats,query)==initial
      and masked_inputs(theta,stats,main,stress,stage,graph)[4]==binding,
     original_targets_unchanged=target.tobytes()==theta[:,stage].tobytes()==physical_theta[:,ORDER[stage]].tobytes(),
     FP32_statistics_normalizers_borders=normalizers,target_statistics_preserved=target_equal,
     cached_contract_preserved=state_equal,model_RNG_preserved=before_rng==after_rng,
     corrected_CDF_executed=active.get("executed_corrected_cdf_code_sha256")==active.get("corrected_cdf_code_sha256"),
     aliases_restored=all(r["aliases_and_flags_restored"] for r in session_reports.values()))
    row["guards"]["same_train_query_mask"]=binding["training"]["training_query_same_mask"]
    row["guards"]["sources_unchanged"]=row["guards"]["sources_unchanged"] and preprocessors_equal
    row.update(stage_decision(row,old));write_json(output/"manifest.json",report)
    del control,references,candidate,prediction,engine
    torch.cuda.synchronize()
    if not row["hard_validity_passed"] or not row["matching_FP32_execution_passed"]:break
  report["compatibility_adapter"]=dict(active)
  require(active["original_alias_restored"] and active["sdk_sources_unchanged"]
   and active["patched_call_count"]==72*len(report["stage_results"]),"observed corrected-CDF restoration/coverage differs")
  require(utility.rng_identity(utility.rng_state(torch))==initial_rng,"global model RNG changed")
  require(source_pins()==pins and guard_baseline_source()==source and graph.policy_metadata()==policy,"source/input policy changed")
  require(sha256_file(pm)==report["prefix_manifest_sha256"] and sha256_file(pp)==report["prefix_samples_sha256"]
   and sha256_file(checkpoint)==CHECKPOINT_SHA256,"qualified inputs changed")
  rows=report["stage_results"];complete=[r["stage_index"] for r in rows]==list(STAGES)
  report.update(matching_FP32_execution_passed_all21=bool(complete and all(r["matching_FP32_execution_passed"] for r in rows)),
   FP16_hard_validity_passed_all21=bool(complete and all(r["hard_validity_passed"] for r in rows)),
   FP16_strict_failed_stages=[r["stage_index"] for r in rows if not r["FP16_strict_passed"]],
   sources_unchanged=True,global_RNG_preserved=True)
  report["status"]="sigma_first_conditional_screen_pass" if report["matching_FP32_execution_passed_all21"] else "sigma_first_matching_FP32_structural_or_hard_failure"
  if report["status"]=="sigma_first_conditional_screen_pass":
   report["replay"]=validate_screen(report,expected_sources=pins,raw_bindings=raw,transformed_bindings=transformed,order_policy=policy)
 except BaseException as error:
  report.update(status="failed_or_incomplete_sigma_first_conditional_screen",error_type=type(error).__name__,error=str(error))
  if active is not None:report["compatibility_adapter"]=dict(active)
  raise
 finally:
  report["elapsed_seconds"]=time.perf_counter()-started;write_json(output/"manifest.json",report)
 return report

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 for name in ("banks","prefix-run","checkpoint","output","ledger"):parser.add_argument("--"+name,required=True)
 parser.add_argument("--max-seconds",type=int,default=1050)
 report=run(**vars(parser.parse_args()))
 print(json.dumps({k:report.get(k) for k in ("status","matching_FP32_execution_passed_all21","FP16_hard_validity_passed_all21","FP16_strict_failed_stages","elapsed_seconds")},indent=2))
if __name__=="__main__":main()
