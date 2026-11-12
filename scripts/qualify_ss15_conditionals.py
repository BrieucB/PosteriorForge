#!/usr/bin/env python3
"""Matching SS15 conditional execution screen. Built with PriorLabs-TabPFN.

Fresh original FP32 predictions and reusable all8 member probabilities use the
same transformed context/query and physical prefixes as the new cache engine.
FP16 differences remain explicit approximation diagnostics. This screen draws
zero posterior vectors and does not establish joint posterior equivalence.
"""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,os,random,time
from pathlib import Path
import numpy as np
from posteriorforge.artifacts import sha256_file,write_json
from posteriorforge.science.analytical_benchmark_v2 import array_descriptor

SCHEMA="posteriorforge.ss15_conditional_qualification.v1"
REPRESENTATION_SHA="46323c0b94301c2f7e678538881d5cbbdd8bc52c62a60021798176bffe81eb85"
ENGINE_SHA="1732f7269286f1629f6e09b1b7ea14cfadbaea84cdae9ee4ab5221cf5b78fbbb"
UTILITY_SHA="6c927a5c278c7fc54b556ac557ecef64f5e6338f5ae13bf466f3b1e6a4640f90"
BASE_SHA="c55c00a1697f8d4341880af18676c29b4d99551e131c89ff79bea238c693f8bd"
THRESHOLDS={"normalization_absolute":1e-5,"max_cdf_absolute":2e-5,
 "max_row_total_variation":1e-4,"quantile_coordinate_span_fraction":2e-6}
ROW_COUNTS=(1,16,128,1024)
FALLBACK_MAP={"10":[5]}
COMPARISON_KEYS=("cached_FP32_vs_original_FP32","cached_FP16_vs_original_FP32","cached_FP16_vs_uncached_FP16")
GUARDS=("sources_unchanged","identical_transformed_training_inputs","identical_physical_targets",
 "frozen_regressor_settings","member_configuration_identity","FP32_postprocessing","FP32_normalizers",
 "train_only_normalization","target_statistics_preserved","cached_contract_preserved",
 "model_RNG_preserved","corrected_CDF_executed","aliases_restored","actual_dtype_witnesses")

def require(value,message):
 if not value:raise ValueError(message)

def load(name,digest):
 path=Path(__file__).with_name(name)
 require(sha256_file(path)==digest,"pinned helper source changed: "+name)
 spec=importlib.util.spec_from_file_location("_ss15_"+path.stem,path)
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def source_pins():
 from posteriorforge import representation_ss15 as rep,execution_ss15 as engine
 require(rep.source_sha256()==REPRESENTATION_SHA,"reviewed SS15 representation changed")
 require(type(ENGINE_SHA) is str and len(ENGINE_SHA)==64 and engine.source_sha256()==ENGINE_SHA,
   "new SS15 execution source is not yet frozen or has changed")
 engine.require_sources()
 return {"qualifier":sha256_file(__file__),"representation":REPRESENTATION_SHA,"execution":ENGINE_SHA,
   "utility":UTILITY_SHA,"comparison_hard_guards":BASE_SHA}

def transformed_inputs(theta,raw_observations,raw_query,representation):
 """Original float64 observations are transformed before any model cast."""
 theta=np.asarray(theta);raw=np.asarray(raw_observations);query=np.asarray(raw_query)
 require(theta.shape==(4096,21) and raw.shape==(4096,33) and query.shape==(1,33),
   "exact physical bank4096x21, raw4096x33 and observedquery1x33 required")
 require(raw.dtype==query.dtype==np.dtype(np.float64),"original float64 bank/query observations required")
 require(np.isfinite(theta).all() and np.isfinite(raw).all() and np.isfinite(query).all(),"finite unchanged inputs required")
 train64=representation.transform_observations(raw);query64=representation.transform_observations(query)
 train=representation.transform_for_model(raw);observed=representation.transform_for_model(query)
 require(train.shape==(4096,15) and observed.shape==(1,15) and train.dtype==observed.dtype==np.float32,
   "source-bound model statistics must be float32 with exactly15 columns")
 require(np.array_equal(train,train64.astype(np.float32)) and np.array_equal(observed,query64.astype(np.float32)),
   "one final representation cast required")
 return theta.astype(np.float32),train,observed,{
  "context_theta":array_descriptor(theta),"context_obs":array_descriptor(raw),"query":array_descriptor(query)},{
  "context_obs_SS15_float64":array_descriptor(train64),"query_SS15_float64":array_descriptor(query64),
  "context_obs_SS15_model":array_descriptor(train),"query_SS15_model":array_descriptor(observed)}

def model_input_descriptors(theta,statistics,query):
 return {"physical_targets_float32":array_descriptor(theta),"context_statistics_float32":array_descriptor(statistics),
   "query_statistics_float32":array_descriptor(query)}

def fixed_queries(observed,prefixes,theta,stage,noise_indices,bounds):
 """Append physical prefixes at column15; observed sufficient statistics stay fixed."""
 obs=np.asarray(observed,np.float32);prefixes=np.asarray(prefixes,np.float32);theta=np.asarray(theta,np.float32)
 require(obs.shape==(1,15) and prefixes.shape==(1024,21) and theta.shape==(4096,21)
   and type(stage) is int and 0<=stage<21,"fixed SS15/prefix/stage shapes required")
 main=np.concatenate((np.repeat(obs,1024,axis=0),prefixes[:,:stage]),axis=1).astype(np.float32)
 stress=np.concatenate((np.repeat(obs,8,axis=0),theta[np.linspace(0,4095,8,dtype=int),:stage]),axis=1).astype(np.float32)
 if stage:
  eligible=[i for i in noise_indices if i<stage]
  if eligible:
   column=15+eligible[0];low,high=bounds;epsilon=(high-low)*1e-4
   stress[0,column]=low;stress[1,column]=low+epsilon;stress[2,column]=high-epsilon;stress[3,column]=high
  stress[4,15:]=0;stress[5,15]=np.nan;stress[6,15:]=np.nan
  stress[7,15:]=theta[:,:stage].mean(0)+10*theta[:,:stage].std(0)
 require(np.array_equal(main[:,:15],np.repeat(obs,1024,axis=0))
   and np.array_equal(stress[:,:15],np.repeat(obs,8,axis=0)),"stress must not alter observation statistics")
 return main,stress

def configuration_equal(left,right):
 """Pickle/type are semantic identity; repr callable addresses are audit only."""
 return (left.get("type")==right.get("type") and left.get("sha256")==right.get("sha256")
   and isinstance(left.get("sha256"),str) and len(left["sha256"])==64)

def inverse_valid(record):
 return bool(record.get("passed") is True and record.get("icdf_contract_version")==2
   and record.get("same_logits_and_borders_exact_failures")==0
   and record.get("independent_fp32_replay_failures")==0
   and record.get("production_sampler_changed") is False)

def comparison_hard(record):
 return bool(record.get("exact_final_borders") is True and record.get("final_float32") is True
  and record.get("eight_members") is True and len(record.get("members",[]))==8
  and all(m.get("configuration_identity_equal") is True and m.get("exact_borders") is True
      and m.get("float32") is True for m in record["members"]) and inverse_valid(record.get("inverse_cdf",{})))

def probability_hard(record):
 if record.get("passed") is not True:return False
 numeric=record.get("max_normalization_error",float("inf"))
 masses=record.get("raw_head_masses",[]);actual=record.get("actual_component_mass_checks",[])
 return bool(np.isfinite(numeric) and 0<=numeric<=1e-5 and record.get("mass_envelope")==1.00001
  and record.get("normalization_absolute")==1e-5 and len(masses)==8
  and all(0<m["min"]<=m["max"]<=1.00001 for m in masses) and len(actual) in (0,8)
  and all(m.get("passed") is True and np.isfinite(m.get("max_fp64_sum_error",np.nan))
      and 0<=m["max_fp64_sum_error"]<=m["fp64_roundoff_bound"]
      and m["fp64_roundoff_bound"]==8*5000*np.finfo(np.float64).eps for m in actual))

def strict_probability(record):
 for key,limit in (("max_normalization_error",1e-5),("max_cdf_absolute_error",2e-5),("max_row_total_variation",1e-4)):
  value=record.get(key)
  if type(value) not in (int,float) or not np.isfinite(value) or not 0<=value<=limit:return False
 minimum=record.get("minimum_probability")
 return bool(type(minimum) in (int,float) and np.isfinite(minimum) and minimum>=0)

def comparison_decision(record):
 return bool(comparison_hard(record) and strict_probability(record.get("final",{}))
   and all(strict_probability(member) for member in record.get("members",[])))

def stage_decision(row):
 require(set(row.get("guards",{}))==set(GUARDS),"complete source/state/dtype/RNG guard set required")
 comparisons=row["comparisons"]
 require(set(comparisons)==set(COMPARISON_KEYS),"three distinct execution/precision comparison arms required")
 for arm,expected in ((COMPARISON_KEYS[0],{"1","16","128","1024","stress"}),
                      (COMPARISON_KEYS[1],{"1","16","128","1024","stress"}),
                      (COMPARISON_KEYS[2],{"128","stress"})):
  require(set(comparisons[arm])==expected,"exact observed conditional coverage required: "+arm)
  for comparison in comparisons[arm].values():
   require(comparison.get("passed") is comparison_decision(comparison),"saved strict probability decision differs")
 hard=all(v is True for v in row["guards"].values())
 hard=hard and bool(row.get("hard_predictions")) and all(probability_hard(v) for v in row["hard_predictions"].values())
 hard=hard and all(comparison_hard(v) for arm in comparisons.values() for v in arm.values())
 strict32=all(v["passed"] is True for v in comparisons[COMPARISON_KEYS[0]].values())
 strict16=all(v["passed"] is True for v in comparisons[COMPARISON_KEYS[1]].values())
 matching16=all(v["passed"] is True for v in comparisons[COMPARISON_KEYS[2]].values())
 return {"hard_validity_passed":bool(hard),"matching_FP32_execution_passed":bool(hard and strict32),
  "FP16_hard_validity_passed":bool(hard),"FP16_strict_passed":bool(hard and strict16),
  "FP16_matching_uncached_strict_passed":bool(hard and matching16),
  "FP16_qualified_as_FP32":False,"posterior_equivalence_claimed":False}

def validate_qualification(document,*,raw_bindings,transformed_bindings,representation_descriptor,expected_sources,fallback_map=None):
 """Pure CPU replay for the SS15 producer; no imports of Torch/SDK or model calls."""
 expected={"schema":SCHEMA,"status":"ss15_conditional_screen_pass","thresholds":THRESHOLDS,
  "row_counts":list(ROW_COUNTS),"stress_rows":8,"posterior_vectors_generated":0,"reference_integrations":0,
  "raw_bindings":raw_bindings,"transformed_bindings":transformed_bindings,
  "representation_descriptor":representation_descriptor,"sources":expected_sources,
  "fallback_map":FALLBACK_MAP if fallback_map is None else fallback_map,"sources_unchanged":True,
  "FP16_qualified_as_FP32":False,"posterior_equivalence_claimed":False}
 for key,value in expected.items():require(document.get(key)==value,"SS15 qualification binding differs: "+key)
 rows=document.get("stage_results",[])
 require([r.get("stage_index") for r in rows]==list(range(21)),"all21 matching SS15 stage records required")
 for row in rows:
  calculated=stage_decision(row)
  require(all(row.get(k)==v for k,v in calculated.items()),"saved SS15 stage predicates differ")
  require(calculated["hard_validity_passed"] and calculated["matching_FP32_execution_passed"],"new SS15 structural/hard screen failed")
  require(row.get("corrected_CDF_calls")==112,"actual source-bound all8 control/query CDF coverage differs")
  from posteriorforge import execution_ss15 as adapter
  for label,mode in (("FP32",adapter.CONTROL_MODE),("FP16",adapter.FP16_MODE)):
   witness_guard(row["precision_sessions"][label],mode=mode,stage=row["stage_index"],
      fallback_map=FALLBACK_MAP if fallback_map is None else fallback_map)
 descriptor=document.get("compatibility_adapter",{})
 require(descriptor.get("patched_call_count")==2352 and descriptor.get("original_alias_restored") is True
   and descriptor.get("sdk_sources_unchanged") is True
   and descriptor.get("executed_corrected_cdf_code_sha256")==descriptor.get("corrected_cdf_code_sha256")
   and descriptor.get("corrected_cdf_code_sha256")!=descriptor.get("original_cdf_code_sha256"),
   "new SS15 corrected-CDF execution/restoration footer differs")
 require(document.get("matching_FP32_execution_passed_all21") is True and document.get("FP16_hard_validity_passed_all21") is True,
   "all21 direct matching structural/hard checks required")
 strict=[r["stage_index"] for r in rows if not r["FP16_strict_passed"]]
 matched=[r["stage_index"] for r in rows if not r["FP16_matching_uncached_strict_passed"]]
 require(document.get("FP16_strict_failed_stages")==strict
   and document.get("FP16_matching_uncached_strict_failed_stages")==matched,"FP16 strict failures relabelled")
 return {"passed":True,"schema":SCHEMA+".replay","stages":21,"matching_FP32_execution_passed_all21":True,
  "FP16_hard_validity_passed_all21":True,"FP16_strict_failed_stages":strict,
  "FP16_matching_uncached_strict_failed_stages":matched,"FP16_qualified_as_FP32":False,
  "posterior_equivalence_claimed":False,"raw33_numerical_qualification_inherited":False}

def normalize_configuration_metadata(prediction):
 result=dict(prediction);result["members"]=[]
 for member in prediction["members"]:
  identity=member["configuration_identity"]
  result["members"].append({**member,"configuration_identity":{"type":identity["type"],"sha256":identity["sha256"]},
    "configuration_repr_audit":identity.get("repr")})
 return result

def compare(reference,candidate,utility):
 result=utility.compare_predictions(normalize_configuration_metadata(reference),normalize_configuration_metadata(candidate))
 result["configuration_identity_basis"]="exact pickle SHA256 and type; repr addresses are audit only"
 require(comparison_hard(result),"conditional border/config/dtype/unchanged inverse hard guard failed")
 require(result["passed"] is comparison_decision(result),"comparison numeric predicate differs")
 return result

def capture_components(engine,query,torch):
 from posteriorforge import execution_ss15 as adapter
 result=adapter.capture_members(engine,query)
 raw=[member["translated_probabilities"] for member in result["members"]]
 result["logits"]=torch.stack([p.to(device=engine.base.device) for p in raw],dim=0).mean(0).log().cpu()
 result["full_mixture_is_diagnostic_reconstruction"]=True
 return result

def cached_model_summary(engine,base,utility):
 return base.model_summary(engine.regressor,utility)

def state_guard(summary):
 c=summary["contract"]
 return bool(c["input_normalization_train_only_verified"]
  and summary["temperature"]==.9 and summary["canonical_borders_dtype"]==summary["physical_borders_dtype"]=="torch.float32"
  and all(t["dtype"]=="torch.float32" for m in c["modules"] if m["type"]=="InputNormalizationEncoderStep" for t in m["tensors"].values()))

def witness_guard(report,*,mode,stage,fallback_map):
 """Observe actual projections; the old raw33 numerical proof is not reused."""
 require(report.get("mode")==mode and report.get("module_sha256")==ENGINE_SHA,"actual SS15 precision source/mode differs")
 require(report["aliases_and_flags_restored"] is True and report["counters"].get("witness_histogram_overflow",0)==0,
   "SS15 precision alias restoration or witness capacity failed")
 require(sum(w["count"] for w in report["witnesses"])==report["counters"]["qkv_calls"]==report["counters"]["attention_calls"],
   "actual projection/attention witness coverage differs")
 from posteriorforge import execution_ss15 as adapter
 for member in range(8):
  strict=mode==adapter.CONTROL_MODE or member in fallback_map.get(str(stage),[])
  for phase in ("prefill","query"):
   selected=[w for w in report["witnesses"] if w["stage"]==stage and w["member"]==member and w["phase"]==phase]
   require(selected,"all8 actual prefill/query witnesses required")
   for w in selected:
    projected=[v for v in w["projection_dtypes"].values() if v is not None]
    require(w["allow_tf32"] is False and w["matmul_precision"]=="highest"
      and w["autocast_enabled"] is (not strict),"actual SS15 mode/fallback precision flags differ")
    require(projected and all(v in ("torch.float32",) if strict else v in ("torch.float16","torch.float32") for v in projected),
      "actual projected dtype outside declared precision family")
    if not strict:require("torch.float16" in projected and w["autocast_dtype"]=="torch.float16","FP16 computation was not observed")
 return True

def hard_prediction(prediction,base,utility):
 result=base.hard_prediction_validity(prediction)
 require(probability_hard(result),"raw mass/FP32 normalization hard guard failed")
 if prediction.get("component_predictions"):
  reconstruction=base.reconstruction_checks(prediction,prediction,utility)
  require(all(v["passed"] for k,v in reconstruction.items() if k!="scope"),"actual raw-mass/normalized-component reconstruction failed")
 return result

def save_golden(output,stage,label,prediction,queries,utility):
 """Save all member probabilities for1024; raw-logit diagnostic coverage128."""
 members=prediction["members"];rows=len(queries)
 arrays={"queries":queries.copy(),"canonical_raw_probabilities":np.stack([utility._numpy(m["translated_probabilities"]) for m in members]),
  "full_mixture_logits":utility._numpy(prediction["logits"]).copy(),
  "full_mixture_probabilities":utility._numpy(prediction["logits"].softmax(-1)).copy(),
  "physical_borders":utility._numpy(prediction["criterion"].borders).copy(),
  "from_borders":np.stack([utility._numpy(m["from_borders"]) for m in members]),
  "to_borders":np.stack([utility._numpy(m["to_borders"]) for m in members]),
  "raw_member_logits_first128":np.stack([utility._numpy(m["raw_logits"][:128]) for m in members])}
 require(arrays["canonical_raw_probabilities"].shape==(8,rows,5000),"all8 canonical probability bins required")
 path=output/f"original-FP32-stage-{stage:02d}-{label}.npz"
 with path.open("xb") as stream:np.savez(stream,**arrays)
 return {"file":path.name,"sha256":sha256_file(path),"observed_member_probability_rows":rows,
  "archived_member_probability_rows":rows,"archived_raw_member_logit_rows":min(rows,128),
  "all8_members":True,"source":"fresh original corrected FP32 fit_preprocessors API",
  "configurations":[m["configuration_identity"] for m in members],
  "arrays":{key:array_descriptor(value) for key,value in arrays.items()}}

def run(*,banks,prefix_run,checkpoint,output,ledger,max_seconds=2250):
 from contextlib import ExitStack
 from posteriorforge.banks import load_context,load_heldout,derive_seed
 from posteriorforge.science.analytical_benchmark_v2 import build_problem
 from posteriorforge.gate import verify_run_binding
 from posteriorforge.inference import CHECKPOINT_SHA256,frozen_regressor_settings,support_mask
 from posteriorforge.inference_cdf import guard_baseline_source
 from posteriorforge.reference_grid import _require_budgeted_gpu
 from posteriorforge.compat_cdf import finite_cdf_compatibility
 from posteriorforge import execution_cache as cache,execution_ss15 as adapter,representation_ss15 as rep
 require(type(max_seconds) is int and 900<=max_seconds<=2250,"bounded900..2250 second diagnostic required")
 pins=source_pins();utility=load("qualify_execution_cache.py",UTILITY_SHA);base=load("characterize_precision_approximation.py",BASE_SHA)
 require(utility.THRESHOLDS==THRESHOLDS,"strict thresholds changed")
 budget=_require_budgeted_gpu(ledger);require(budget["gpus"]==1,"one reserved GPU required")
 require(sha256_file(checkpoint)==CHECKPOINT_SHA256,"frozen checkpoint required")
 started=time.perf_counter();deadline=started+max_seconds;output=Path(output);output.mkdir(parents=True,exist_ok=False)
 context=load_context(banks,20261111,4096);observed=load_heldout(banks,0);raw_query=observed.obs.reshape(1,33)
 theta,stats,query,raw,transformed=transformed_inputs(context.theta,context.obs,raw_query,rep);problem=build_problem()
 initial_model_inputs=model_input_descriptors(theta,stats,query)
 raw.update(checkpoint_sha256=CHECKPOINT_SHA256,scientific_fingerprint=problem.scientific_fingerprint,parameter_order=list(problem.parameter_order))
 prefix_run=Path(prefix_run);prefix_manifest=prefix_run/"manifest.json";prefix_payload=prefix_run/"samples.npz"
 prefix_header=json.loads(prefix_manifest.read_bytes());verify_run_binding(prefix_header,context,raw_query,20261111,0)
 require(sha256_file(prefix_payload)==prefix_header["samples_sha256"],"existing physical prefix payload differs")
 with np.load(prefix_payload,allow_pickle=False) as saved:prefix_values=saved["samples"].copy()
 require(prefix_values.shape==(50000,21) and support_mask(prefix_values).all(),"existing complete supported physical prefixes required")
 positions=np.linspace(0,49999,1024,dtype=int);prefixes=prefix_values[positions].astype(np.float32)
 baseline_source=guard_baseline_source();descriptor=rep.representation_metadata()
 report={"schema":SCHEMA,"status":"running","attribution":"Built with PriorLabs-TabPFN","sources":pins,
  "raw_bindings":raw,"transformed_bindings":transformed,"representation_descriptor":descriptor,
  "row_counts":list(ROW_COUNTS),"stress_rows":8,"thresholds":THRESHOLDS,"fallback_map":FALLBACK_MAP,
  "posterior_vectors_generated":0,"reference_integrations":0,"stage_results":[],
  "matching_FP32_execution_passed_all21":False,"FP16_hard_validity_passed_all21":False,
  "FP16_qualified_as_FP32":False,"posterior_equivalence_claimed":False,
  "prefix_manifest_sha256":sha256_file(prefix_manifest),"prefix_samples_sha256":sha256_file(prefix_payload),
  "fixed_prefix_row_indices":positions.tolist(),"prefix_scope":"accepted original raw33-run physical vectors; fixed diagnostic prefixes, no truth labels or new posterior draw",
  "FP16_strict_failed_stages":[],"FP16_matching_uncached_strict_failed_stages":[],
  "raw33_numerical_qualification_inherited":False,"budget_binding":budget,"baseline_execution_source":baseline_source}
 write_json(output/"manifest.json",report);active=None
 try:
  import torch
  require(torch.cuda.is_available() and torch.cuda.device_count()==1,"one visible GPU required")
  require(torch.get_default_dtype()==torch.float32 and torch.get_float32_matmul_precision()=="highest"
    and not torch.backends.cuda.matmul.allow_tf32 and not torch.is_autocast_enabled("cuda"),"strict FP32 outer flags required")
  seed=derive_seed(20261111,"matching-SS15-conditionals",case=0)
  random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
  torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS","16")))
  report.update(seed=seed,RNG_role="matching-SS15-conditionals",torch_version=str(torch.__version__),numpy_version=np.__version__)
  global_rng=utility.rng_identity(utility.rng_state(torch));noise=[problem.parameter_order.index(n) for n in problem.prior.noise_names]
  with finite_cdf_compatibility() as active:
   for stage in range(21):
    require(time.perf_counter()+25<deadline,"bounded deadline; partial coverage cannot qualify")
    row={"stage_index":stage,"comparisons":{k:{} for k in COMPARISON_KEYS},"hard_predictions":{},"guards":{}}
    row["model_inputs_before"]=model_input_descriptors(theta,stats,query)
    require(row["model_inputs_before"]==initial_model_inputs,"transformed training/query inputs or physical targets changed")
    report["stage_results"].append(row);report["active_stage"]=stage;write_json(output/"manifest.json",report)
    before_rng=utility.rng_identity(utility.rng_state(torch));before_cdf=active["patched_call_count"]
    main,stress=fixed_queries(query,prefixes,theta,stage,noise,problem.prior.bounds)
    row["queries"]={"main":array_descriptor(main),"stress":array_descriptor(stress)}
    settings=frozen_regressor_settings(checkpoint);settings["inference_precision"]=torch.float32
    _,sdk,_=cache._load_sdk();control=sdk.TabPFNRegressor(**settings)
    joint=np.concatenate((stats,theta),axis=1)
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
     control.fit(joint[:,:15+stage],joint[:,15+stage])
    references={}
    for label,queries in (("1024",main),("stress",stress)):
     with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
      prediction=cache.capture_corrected_prediction(control,queries,cdf_descriptor=active)
     references[label]=prediction
     row["hard_predictions"]["original_FP32_"+label]=hard_prediction(prediction,base,utility)
    row["goldens"]={label:save_golden(output,stage,label,prediction,main if label=="1024" else stress,utility)
      for label,prediction in references.items()}
    row["original_FP32_model"]=base.model_summary(control,utility)
    cached_results={};session_reports={};state_equal=True;target_equal=True;normalizers=True;preprocessors_equal=True
    for mode,label in ((adapter.CONTROL_MODE,"FP32"),(adapter.FP16_MODE,"FP16")):
     with adapter.SS15PrecisionSession(mode,fallback_map={10:[5]}) as session:
      with ExitStack() as cleanup:
       matching={}
       if label=="FP16":
        uncached=adapter.SS15UncachedStage(checkpoint,cdf_descriptor=active,stage_index=stage,theta=theta,obs=stats)
        cleanup.callback(uncached.close)
        for key,queries in (("128",main[:128]),("stress",stress)):
         prediction=capture_components(uncached,queries,torch);matching[key]=prediction
         row["hard_predictions"]["uncached_FP16_"+key]=hard_prediction(prediction,base,utility)
       engine=adapter.SS15StageExecutor(checkpoint,cdf_descriptor=active,query_batch_size=1024,fallback_map={10:[5]})
       cleanup.callback(engine.close)
       prepared=engine.prepare(theta,stats,stage)
       before_model=cached_model_summary(engine,base,utility);normalizers=normalizers and state_guard(before_model)
       target_equal=target_equal and all(before_model[k]==row["original_FP32_model"][k] for k in ("target_mean","target_sd","temperature"))
       preprocessors_equal=preprocessors_equal and before_model["contract"]["preprocessor_sha256"]==row["original_FP32_model"]["contract"]["preprocessor_sha256"]
       row[label+"_prepared"]=prepared;row[label+"_model_before"]=before_model;cached_results[label]={}
       for key in ("1","16","128","1024","stress"):
        queries=stress if key=="stress" else main[:int(key)]
        candidate=capture_components(engine,queries,torch);cached_results[label][key]=candidate
        row["hard_predictions"]["cached_"+label+"_"+key]=hard_prediction(candidate,base,utility)
        reference=references["stress"] if key=="stress" else utility.slice_prediction(references["1024"],int(key))
        arm=COMPARISON_KEYS[0] if label=="FP32" else COMPARISON_KEYS[1]
        row["comparisons"][arm][key]=compare(reference,candidate,utility)
        if label=="FP16" and key in matching:
         row["comparisons"][COMPARISON_KEYS[2]][key]=compare(matching[key],candidate,utility)
       after_model=cached_model_summary(engine,base,utility)
       state_equal=state_equal and before_model==after_model
       row[label+"_model_after"]=after_model;normalizers=normalizers and state_guard(after_model)
     session_reports[label]=session.report()
     witness_guard(session_reports[label],mode=mode,stage=stage,fallback_map=FALLBACK_MAP)
    row["precision_sessions"]=session_reports
    row["model_inputs_after"]=model_input_descriptors(theta,stats,query)
    row["cached_contract_evidence_scope"]={"all8_normalizers_configurations_and_embedding_metadata":True,
      "KV_contents_hashed":False,"large_cached_embedding_bytes_hashed":False,
      "scope":"observed frozen utility contract before/after queries; not a full KV byte replay or retention proof"}
    after_rng=utility.rng_identity(utility.rng_state(torch));row["RNG_identity_before"]=before_rng;row["RNG_identity_after"]=after_rng
    row["corrected_CDF_calls"]=active["patched_call_count"]-before_cdf
    require(row["corrected_CDF_calls"]==112,"actual all8 original/cached/matched control coverage differs")
    row["guards"]={k:True for k in GUARDS}
    row["guards"].update(sources_unchanged=source_pins()==pins,FP32_normalizers=normalizers,
      identical_transformed_training_inputs=row["model_inputs_before"]==row["model_inputs_after"]==initial_model_inputs,
      identical_physical_targets=row["model_inputs_after"]["physical_targets_float32"]==initial_model_inputs["physical_targets_float32"],
      target_statistics_preserved=target_equal,cached_contract_preserved=state_equal,frozen_regressor_settings=preprocessors_equal,model_RNG_preserved=before_rng==after_rng,
      corrected_CDF_executed=active.get("executed_corrected_cdf_code_sha256")==active.get("corrected_cdf_code_sha256"),
      aliases_restored=all(r["aliases_and_flags_restored"] is True for r in session_reports.values()))
    row.update(stage_decision(row))
    require(row["hard_validity_passed"],"source/state/mass/normalization/dtype hard failure; accuracy attribution blocked")
    del control,references,cached_results,matching,candidate,prediction,engine
    torch.cuda.synchronize()
    write_json(output/"manifest.json",report)
  report["compatibility_adapter"]=dict(active)
  require(active["original_alias_restored"] and active["sdk_sources_unchanged"] and active["patched_call_count"]==2352,
    "actual corrected CDF final coverage/restoration failed")
  require(utility.rng_identity(utility.rng_state(torch))==global_rng,"global model RNG changed")
  require(source_pins()==pins and guard_baseline_source()==baseline_source and rep.representation_metadata()==descriptor,"source changed")
  require(sha256_file(prefix_manifest)==report["prefix_manifest_sha256"] and sha256_file(prefix_payload)==report["prefix_samples_sha256"]
    and sha256_file(checkpoint)==CHECKPOINT_SHA256,"frozen physical inputs changed")
  rows=report["stage_results"];report.update(
   matching_FP32_execution_passed_all21=all(r["matching_FP32_execution_passed"] for r in rows),
   FP16_hard_validity_passed_all21=all(r["FP16_hard_validity_passed"] for r in rows),
   FP16_strict_failed_stages=[r["stage_index"] for r in rows if not r["FP16_strict_passed"]],
   FP16_matching_uncached_strict_failed_stages=[r["stage_index"] for r in rows if not r["FP16_matching_uncached_strict_passed"]],
   sources_unchanged=True,global_RNG_preserved=True)
  report["status"]="ss15_conditional_screen_pass" if report["matching_FP32_execution_passed_all21"] else "ss15_matching_FP32_structural_failure"
  if report["status"]=="ss15_conditional_screen_pass":
   report["replay"]=validate_qualification(report,raw_bindings=raw,transformed_bindings=transformed,
    representation_descriptor=descriptor,expected_sources=pins,fallback_map=FALLBACK_MAP)
 except BaseException as error:
  report.update(status="failed_or_incomplete_SS15_conditionals",error_type=type(error).__name__,error=str(error))
  if active is not None:report["compatibility_adapter"]=dict(active)
  raise
 finally:
  report["elapsed_seconds"]=time.perf_counter()-started;write_json(output/"manifest.json",report)
 return report

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 for name in ("banks","prefix-run","checkpoint","output","ledger"):parser.add_argument("--"+name,required=True)
 parser.add_argument("--max-seconds",type=int,default=2250)
 report=run(**vars(parser.parse_args()))
 print(json.dumps({k:report.get(k) for k in ("status","matching_FP32_execution_passed_all21",
  "FP16_hard_validity_passed_all21","FP16_strict_failed_stages","elapsed_seconds")},indent=2))
if __name__=="__main__":main()
