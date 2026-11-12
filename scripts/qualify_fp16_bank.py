#!/usr/bin/env python3
"""All21 FP16 bank-retention roundtrip, not FP32/posterior qualification.

Built with PriorLabs-TabPFN. No posterior draws, new worlds, reference integration,
precision/sampler repair, old-residual allowance, or private CUDA cleanup.
"""
from __future__ import annotations
import argparse,gc,gzip,hashlib,importlib.util,json,os,pickle,random,time
from pathlib import Path
import numpy as np

SCHEMA="posteriorforge.fp16_bank_retention_qualification.v1"
OWNER_SHA="4cc2367598ffdbbf2ffb56b7227c2fdd5b96d4d623f0961f20668f89424eeb3e"
ALL21_SHA="20e64a64357de7862c20b2fa846c6d128c181ca564c261800b626e6c4bd32208"
BASE_SHA="c55c00a1697f8d4341880af18676c29b4d99551e131c89ff79bea238c693f8bd"
UTILITY_SHA="6c927a5c278c7fc54b556ac557ecef64f5e6338f5ae13bf466f3b1e6a4640f90"
STAGES=tuple(range(21))
WARM_SWEEPS=2
MAX_SECONDS=2250

def require(value,message):
 if not value:raise RuntimeError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(name,expected):
 path=Path(__file__).with_name(name);require(sha(path)==expected,"frozen source changed: "+name)
 spec=importlib.util.spec_from_file_location("_fp16_bank_"+path.stem,path)
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def sources():
 from posteriorforge import execution_fp16_bank as owner
 require(owner.execution_source_sha256()==OWNER_SHA,"new owner source changed");owner.require_sources()
 return {"producer":sha(__file__),"bank_owner":OWNER_SHA,"precision":owner.PRECISION_SHA,
         "migration":owner.MIGRATION_SHA,"clean_experiment":owner.CLEAN_SHA,
         "all21":ALL21_SHA,"base":BASE_SHA,"utility":UTILITY_SHA}

def tensor_descriptor(tensor,torch):
 value=tensor.detach().cpu().contiguous()
 return {"dtype":str(value.dtype),"shape":list(value.shape),
         "sha256":hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()}

def prediction_fingerprint(prediction,torch):
 """Hash actual whole batches, not a scalar-row recomputation of their CDF."""
 from posteriorforge.execution_mixture import finite_bar_icdf
 logits=prediction["logits"];borders=prediction["criterion"].borders
 require(logits.device.type==borders.device.type=="cpu","saved physical sampler inputs must be CPU")
 uniforms=torch.linspace(0,float(np.nextafter(np.float32(1),np.float32(0))),len(logits),dtype=torch.float32)
 inverse=finite_bar_icdf(logits,borders,uniforms)
 return {"logits":tensor_descriptor(logits,torch),"physical_borders":tensor_descriptor(borders,torch),
   "members":[{**{k:tensor_descriptor(m[k],torch) for k in ("raw_logits","translated_probabilities","from_borders","to_borders")},
     "configuration_identity":m["configuration_identity"]} for m in prediction["members"]],
   "same_wholebatch_fixed_uniform_inverse":tensor_descriptor(inverse,torch),
   "uniforms":tensor_descriptor(uniforms,torch),
   "existing_inverse_nonfinite":int((~torch.isfinite(inverse)).sum()),
   "existing_inverse_outside_physical_borders":int(((inverse<borders[0])|(inverse>borders[-1])).sum()),
   "sampler_math_changed":False,"native_or_scalar_inverse_replay_performed":False}

def shared_fingerprint(prediction,torch):
 from posteriorforge.execution_mixture import finite_bar_icdf
 logits=prediction["logits"];borders=prediction["criterion"].borders
 uniforms=torch.tensor([0.,.5,float(np.nextafter(np.float32(1),np.float32(0)))],dtype=torch.float32)
 values=finite_bar_icdf(logits,borders,uniforms)
 return {"logits":tensor_descriptor(logits,torch),"physical_borders":tensor_descriptor(borders,torch),
         "same_wholebatch_shared_inverse":tensor_descriptor(values,torch),
         "uniforms":tensor_descriptor(uniforms,torch),
         "existing_inverse_nonfinite":int((~torch.isfinite(values)).sum()),
         "existing_inverse_outside_borders":int(((values<borders[0])|(values>borders[-1])).sum())}

def library_baseline(torch,rng_identity):
 """Warm this policy's libraries with deterministic public operations only."""
 require(torch.cuda.memory_allocated()==0,"fresh GPU allocated baseline must be exactly zero")
 before=rng_identity()
 started=time.perf_counter()
 for dtype in (torch.float16,torch.float32): # FP32 is the declared10:5 fallback.
  left=torch.ones((1024,1024),dtype=dtype,device="cuda")
  right=torch.ones((1024,1024),dtype=dtype,device="cuda")
  product=left@right
  query=torch.ones((1,8,128,64),dtype=dtype,device="cuda")
  key=torch.ones((1,1,128,64),dtype=dtype,device="cuda").expand(1,8,128,64)
  value=torch.ones((1,1,128,64),dtype=dtype,device="cuda").expand(1,8,128,64)
  attention=torch.nn.functional.scaled_dot_product_attention(query,key,value,dropout_p=0.)
  torch.cuda.synchronize()
  require(bool(torch.isfinite(product).all()) and bool(torch.isfinite(attention).all()),"library warmup produced nonfinite values")
  del left,right,product,query,key,value,attention
 gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache();torch.cuda.synchronize()
 require(before==rng_identity(),"deterministic library-only warmup changed RNG state")
 return {"fresh_allocated_bytes":0,"allocated_bytes":int(torch.cuda.memory_allocated()),
   "reserved_bytes":int(torch.cuda.memory_reserved()),"elapsed_seconds":time.perf_counter()-started,
   "operations":"FP16 and declared-fallback FP32 ones GEMM1024 and SDPA(1,8,128,64),dropout0",
   "computed_dtypes":["torch.float16","torch.float32"],"model_fit_or_forward_calls":0,
   "rng_preserved":True,"private_cuda_clear_called":False,"old_residual_allowance_used":False,
   "required_residual":"exact equality to this newly measured allocated-byte baseline"}

def assert_residual(torch,baseline,label):
 torch.cuda.synchronize();gc.collect()
 allocated=int(torch.cuda.memory_allocated())
 require(allocated==baseline["allocated_bytes"],"new library-only CUDA residual differs after "+label)
 return {"label":label,"allocated_bytes":allocated,"new_library_baseline_bytes":baseline["allocated_bytes"],"exact_equal":True}

def archive(output,name,value):
 path=output/(name+".json.gz")
 with gzip.open(path,"wt",encoding="utf-8",compresslevel=3) as handle:
  json.dump(value,handle,sort_keys=True,allow_nan=False)
 return {"file":path.name,"sha256":sha(path)}

def logical_values(snapshot):
 result=[]
 for row in snapshot:
  for path in row["paths"]:
   if "/<view_base>" not in path and "/<grad>" not in path:
    result.append({k:row[k] for k in ("dtype","shape","sha256")}|{"path":path})
 return sorted(result,key=lambda r:r["path"])

def preprocessing_identity(stage):
 from posteriorforge.execution_hybrid import PREPROCESSING_FIELDS
 fields={"resident":("preprocessors","configs","cat_ixs","n_train_samples")}
 if stage.uncached_regressor is not None:fields["uncached"]=PREPROCESSING_FIELDS
 result={}
 for name,names in fields.items():
  reg=stage.regressor if name=="resident" else stage.uncached_regressor
  result[name]={key:hashlib.sha256(pickle.dumps(getattr(reg.executor_,key),protocol=5)).hexdigest() for key in names}
 return result

def model_state(stage,base,utility):
 state={"resident":base.model_summary(stage.regressor,utility)}
 if stage.uncached_regressor is not None:state["uncached"]=base.model_summary(stage.uncached_regressor,utility)
 return state

def check_model_state(state):
 for name,value in state.items():
  contract=value["contract"]
  require(contract["input_normalization_train_only_verified"],"train-only normalizer missing")
  require(value["canonical_borders_dtype"]==value["physical_borders_dtype"]=="torch.float32"
     and value["temperature"]==.9,"FP32 physical/canonical borders or temperature changed")
  for module in contract["modules"]:
   if module["type"]=="InputNormalizationEncoderStep":
    require(all(t["dtype"]=="torch.float32" for t in module["tensors"].values()),"normalizer statistic tensor left FP32")
  if name=="uncached":require(contract["autocast"] is False and contract["forced_inference_dtype"]=="torch.float32","fallback is not strictly FP32")
  else:require(contract["autocast"] is True and contract["forced_inference_dtype"]=="None","actual cached FP16 AMP policy changed")

def validate_counts(counters,windows):
 expected={"logical_acquires":63,"misses":21,"hits":42,"reload_count":42,"offload_count":63}
 for name,value in expected.items():require(counters.get(name)==value,"all21 cold/warm count differs: "+name)
 require(counters.get("physical_head_calls")==2056,"actual total physical head/CDF context count differs")
 require(len(windows)==63,"exact63 observation windows required")
 require([w["stage_index"] for w in windows]==list(STAGES)*3,"cold21 then cyclic21+21 required")
 require(all(w["cold"] is (i<21) for i,w in enumerate(windows)),"cold/warm window labels differ")
 require(sum(w["counter_delta"].get("witness_histogram_overflow",0) for w in windows)==0,"observer overflow")
 require(all(w["counter_delta"].get("qkv_calls",0)==w["counter_delta"].get("attention_calls",0) for w in windows),"window QKV/attention count differs")
 require(all(not any(r["phase"]=="prefill" for r in w["precision_window"]["witnesses"]) for w in windows[21:]),"warm prefill occurred")
 return {"passed":True,"cold_fits":21,"warm_actual_fits":0,"warm_actual_preprocess_refits":0,
         "warm_new_prefill_events":0,"reloads":42,"offloads":63,"total_migrations":105}

def validate_observation_window(window,original,stage,sweep,precision):
 report=window["precision_window"];query=precision.query_witness_passed(report,stage)
 require(query["passed"],"window actual all8 query witness failed")
 original_calls=3 if stage==0 else 2
 actual_calls=(3 if stage==0 else 2) if sweep==0 else (7 if stage==0 else 5)
 for old,new in zip(original["computed_query_witness"]["members"],query["members"]):
  require(old["member"]==new["member"] and old["query_event_count"]%original_calls==0,
          "original per-head query architecture factor unavailable")
  require(new["query_event_count"]==old["query_event_count"]//original_calls*actual_calls,
          "actual per-member query projection count differs")
 require(all(w["stage"]==stage and 0<=w["member"]<8 for w in report["witnesses"]),
         "window stage/member witness leaked across acquisition")
 require(window["counter_delta"].get("qkv_calls",0)==window["counter_delta"].get("attention_calls",0),
         "QKV/attention event count differs")
 if sweep==0:
  prefill=precision.prefill_witness_passed(report,stage)
  require(prefill==original["computed_prefill_witness"],"cold prefill architecture/dtype/storage witness differs from immutable FP16 screen")
 else:
  require(not any(w["phase"]=="prefill" for w in report["witnesses"]),"warm acquisition did prefill")
 return {"passed":True,"actual_query_calls_per_member":actual_calls,
   "cold_prefill_provenance":original["computed_prefill_witness"],
   "warm_prefill_witness_claimed":False,"actual_warm_prefill_events":0 if sweep else None,
   "all21_ancestry_source":"immutable f01; original strict drift failures preserved"}

def read_bindings(*,banks,prefix_run,checkpoint,qualification,batch_qualification,sdk_root,all21_artifact):
 from posteriorforge.banks import load_context,load_heldout
 from posteriorforge.gate import verify_run_binding
 from posteriorforge.inference import CHECKPOINT_SHA256,support_mask
 from posteriorforge.science.analytical_benchmark_v2 import build_problem,array_descriptor
 from posteriorforge import performance_fp16_experiment as clean
 require(sha(checkpoint)==CHECKPOINT_SHA256,"checkpoint changed")
 context=load_context(banks,20261111,4096);case0=load_heldout(banks,0);case1=load_heldout(banks,1);problem=build_problem()
 query=case0.obs.reshape(1,33);other=case1.obs.reshape(1,33)
 bindings={"checkpoint_sha256":CHECKPOINT_SHA256,"scientific_fingerprint":problem.scientific_fingerprint,
   "parameter_order":list(problem.parameter_order),"context_theta":array_descriptor(context.theta),
   "context_obs":array_descriptor(context.obs),"query":array_descriptor(query)}
 old=load("characterize_fp16_all_stages.py",ALL21_SHA)
 q,receipt=old.portable_qualification(qualification,batch_qualification,sdk_root,bindings)
 approximation=clean.validate_all21(all21_artifact,bindings)
 prefix_run=Path(prefix_run)
 require(sha(prefix_run/"manifest.json")==q["prefix_manifest_sha256"] and sha(prefix_run/"samples.npz")==q["prefix_samples_sha256"],"prefix custody changed")
 pm=json.loads((prefix_run/"manifest.json").read_bytes());verify_run_binding(pm,context,case0.obs,20261111,0)
 with np.load(prefix_run/"samples.npz",allow_pickle=False) as saved:values=saved["samples"].copy()
 require(values.shape==(50000,21) and support_mask(values).all(),"complete existing supported prefix vectors required")
 positions=np.asarray(q["fixed_prefix_row_indices"],int)
 require(positions.shape==(1024,) and len(np.unique(positions))==1024,"authenticated1024 prefix rows required")
 require(not np.array_equal(query,other),"A/B saved observations must differ")
 return context,query,other,values[positions],positions,problem,bindings,q,receipt,approximation

def run(*,banks,prefix_run,checkpoint,qualification,batch_qualification,sdk_root,all21_artifact,ledger,output,max_seconds=2250):
 from posteriorforge.artifacts import runtime_identity,write_json
 from posteriorforge.reference_grid import _require_budgeted_gpu
 from posteriorforge.banks import derive_seed
 from posteriorforge.science.analytical_benchmark_v2 import array_descriptor
 from posteriorforge.compat_cdf import finite_cdf_compatibility
 from posteriorforge import execution_fp16_bank as owner,execution_precision_probe as precision
 require(type(max_seconds) is int and 900<=max_seconds<=MAX_SECONDS,"bounded900..2250 seconds required")
 original_sources=sources();budget=_require_budgeted_gpu(ledger);require(budget["gpus"]==1,"one budgeted GPU required")
 start=time.perf_counter();deadline=start+max_seconds;output=Path(output);output.mkdir(parents=True,exist_ok=False)
 utility=load("qualify_execution_cache.py",UTILITY_SHA);base=load("characterize_precision_approximation.py",BASE_SHA)
 all21=load("characterize_fp16_all_stages.py",ALL21_SHA)
 context,query,other,prefixes,positions,problem,bindings,q,receipt,approximation=read_bindings(
   banks=banks,prefix_run=prefix_run,checkpoint=checkpoint,qualification=qualification,batch_qualification=batch_qualification,
   sdk_root=sdk_root,all21_artifact=all21_artifact)
 theta=context.theta.astype(np.float32);obs=context.obs.astype(np.float32)
 report={**runtime_identity(),"schema":SCHEMA,"status":"running","attribution":"Built with PriorLabs-TabPFN",
   "retention_qualified":False,"qualified_FP32":False,"posterior_equivalence_claimed":False,
   "posterior_vectors_generated":0,"reference_integrations":0,"sources":original_sources,"budget_binding":budget,
   "scope":"new FP16 bank residency/state retention; not old FP32 host/GQA qualification",
   "engine_policy":owner.execution_metadata(),"all21_approximation_manifest_sha256":sha(all21_artifact),
   "all21_strict_failed_stages_preserved":approximation["strict_failed_stages"],
   "portable_recovery_verification":receipt,"qualification_sha256":sha(qualification),
   "batch_qualification_sha256":sha(batch_qualification),"prefix_manifest_sha256":q["prefix_manifest_sha256"],
   "prefix_samples_sha256":q["prefix_samples_sha256"],"fixed_prefix_row_indices":positions.tolist(),
   "query_B":array_descriptor(other),"cold_stages":[],"warm_sweeps":[[],[]],**bindings,
   "limits":{"rss_bytes":100*owner.GIB,"slack_bytes":8*owner.GIB,"effective_current_RSS_limit_bytes":92*owner.GIB},
   "limitations":["Fixed accepted prefixes and eight stress rows are finite diagnostic coverage, not a joint posterior bound.",
     "Exact reuse is relative to this FP16 approximate composite; all original strict FP32 drift failures remain visible.",
     "No full50000 warm request or performance claim is made by this conditional retention proof."]}
 write_json(output/"manifest.json",report)
 bank=None;session=None;active=None;baseline=None
 try:
  import torch
  require(torch.cuda.is_available() and torch.cuda.device_count()==1,"one visible GPU required")
  require(torch.get_float32_matmul_precision()=="highest" and not torch.backends.cuda.matmul.allow_tf32
    and not torch.is_autocast_enabled("cuda"),"highest/TF32off/outerAMPoff required")
  seed=derive_seed(20261111,"fp16-bank-retention-proof",case=0)
  random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
  torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS","16")))
  rng=lambda:utility.rng_identity(utility.rng_state(torch))
  global_rng=rng();baseline=library_baseline(torch,rng)
  report.update(seed=seed,rng_role="fp16-bank-retention-proof",library_only_cuda_baseline=baseline,
    device=torch.cuda.get_device_name(),torch_version=str(torch.__version__),torch_cuda_version=torch.version.cuda,
    environment={k:os.environ.get(k) for k in ("CUBLAS_WORKSPACE_CONFIG","PYTORCH_CUDA_ALLOC_CONF","OMP_NUM_THREADS")})
  torch.cuda.reset_peak_memory_stats();noise=[problem.parameter_order.index(n) for n in problem.prior.noise_names]
  cold={}
  with finite_cdf_compatibility() as active:
   with owner.fp16_bank_scope(checkpoint,cdf_descriptor=active,verify_copy_hashes=True) as bank:
    session=bank.session
    report["initial_admission"]=bank.initial_admission
    for sweep in range(3):
     rows=report["cold_stages"] if sweep==0 else report["warm_sweeps"][sweep-1]
     for stage in STAGES:
      require(time.perf_counter()+20<deadline,"bounded deadline; incomplete bank proof never promoted")
      report.update(active_stage=stage,active_sweep=sweep,elapsed_seconds=time.perf_counter()-start);write_json(output/"manifest.json",report)
      before_rng=rng();before_calls=active["patched_call_count"];before_counters=dict(bank.counters)
      row={"stage_index":stage,"parameter":problem.parameter_order[stage],"passed":False}
      rows.append(row);t=time.perf_counter();prepared=bank.prepare(theta,obs,stage);torch.cuda.synchronize()
      row["acquire_seconds"]=time.perf_counter()-t;row["prepared"]=prepared
      require(prepared["bank_hit"] is (sweep>0) and prepared["actual_fit_count"]==(0 if sweep else 1)
        and prepared["actual_preprocess_refit_count"]==(0 if sweep else 1),"actual warm fit/preprocess count differs")
      model=bank.entries[bank.active]["engine"]
      queries,stress=utility.fixed_queries(query,prefixes,theta,stage,noise,problem.prior.bounds)
      bqueries=np.concatenate((np.repeat(other,16,axis=0),prefixes[:16,:stage]),axis=1).astype(np.float32)
      require(not np.array_equal(queries[:16],bqueries),"queryB must differ; no query density is retained")
      a=precision.capture_members(bank,queries);a_stress=precision.capture_members(bank,stress)
      require(base.hard_prediction_validity(a)["passed"] and base.hard_prediction_validity(a_stress)["passed"],"A/stress raw mass/probability hard guard failed")
      fingerprint={"A":prediction_fingerprint(a,torch),"stress":prediction_fingerprint(a_stress,torch)}
      if stage==0:
       shared=bank.predict(queries[:1],output_type="full")
       fingerprint["fresh_shared_stage0"]=shared_fingerprint(shared,torch);del shared
      scalar=model_state(model,base,utility);check_model_state(scalar);preprocess=preprocessing_identity(model)
      logical=owner.logical_snapshot(owner.stage_roots(model),torch)
      if sweep==0:
       cold[stage]={"prediction":fingerprint,"model":scalar,"preprocessing":preprocess,"logical":logical_values(logical)}
       row["cold_prefill"]=precision.prefill_witness_passed(session.report(),stage)
       require(row["cold_prefill"]["passed"],"all8 computed cold prefill witness required")
      else:
       require(fingerprint==cold[stage]["prediction"],"reloaded A/stress/shared density differs at whole-batch bit level")
       require(scalar==cold[stage]["model"] and preprocess==cold[stage]["preprocessing"],"cold/warm normalizer/config/target/preprocessing changed")
       require(logical_values(logical)==cold[stage]["logical"],"cold/warm logical model state changed")
       b=precision.capture_members(bank,bqueries)
       require(base.hard_prediction_validity(b)["passed"],"B raw mass/probability guard failed")
       row["B_prediction"]=prediction_fingerprint(b,torch)
       again=precision.capture_members(bank,queries);again_stress=precision.capture_members(bank,stress)
       require(base.hard_prediction_validity(again)["passed"] and base.hard_prediction_validity(again_stress)["passed"],"A-after-B probability guard failed")
       repeated={"A":prediction_fingerprint(again,torch),"stress":prediction_fingerprint(again_stress,torch)}
       if stage==0:
        shared=bank.predict(queries[:1],output_type="full")
        repeated["fresh_shared_stage0"]=shared_fingerprint(shared,torch);del shared
       require(repeated==fingerprint,"A→B→A whole-batch predictions/inverse/borders/config differ")
       after_scalar=model_state(model,base,utility);check_model_state(after_scalar)
       require(after_scalar==scalar and preprocessing_identity(model)==preprocess,"B/A changed normalization/config/preprocessing/target statistics")
       after_logical=owner.logical_snapshot(owner.stage_roots(model),torch)
       require(logical_values(after_logical)==logical_values(logical),"B/A changed logical tensor paths/shapes/dtypes/values")
       row["after_B_A_state_archive"]=archive(output,f"sweep-{sweep}-stage-{stage:02d}-after-B-A-logical",after_logical)
       row["A_B_A_exact"]=True;del b,again,again_stress
      row["prediction"]=fingerprint;row["model"]=scalar;row["preprocessing"]=preprocess
      row["logical_state_archive"]=archive(output,f"sweep-{sweep}-stage-{stage:02d}-logical",logical)
      row["computed_query_witness"]=precision.query_witness_passed(session.report(),stage)
      require(row["computed_query_witness"]["passed"],"all8 actual FP16/fallback query witness required")
      row["root_coverage"]=owner.verify_root_coverage(model,torch)
      row["model_rng_preserved"]=before_rng==rng();require(row["model_rng_preserved"],"model/retention changed default RNG")
      row["corrected_CDF_calls"]=active["patched_call_count"]-before_calls
      expected=(24 if stage==0 else 16) if sweep==0 else (56 if stage==0 else 40)
      require(row["corrected_CDF_calls"]==expected,"physical8head/shared-density CDF coverage differs")
      del a,a_stress,model
      bank.close()
      row["copy_report_archive"]=archive(output,f"sweep-{sweep}-stage-{stage:02d}-migration",bank.copy_events[-1 if sweep==0 else -2:])
      row["window"]=len(bank.windows.windows)-1
      row["observation_window_validation"]=validate_observation_window(bank.windows.windows[-1],
         approximation["stage_results"][stage],stage,sweep,precision)
      wanted_calls=(3 if stage==0 else 2) if sweep==0 else (7 if stage==0 else 5)
      for member in range(8):
       name=f"stage_{stage}_member_{member}_head_calls"
       require(bank.counters[name]-before_counters.get(name,0)==wanted_calls,"actual physical member context coverage differs")
      row["counter_delta"]={k:v-before_counters.get(k,0) for k,v in bank.counters.items()}
      require(row["counter_delta"]["physical_head_calls"]==row["corrected_CDF_calls"]==expected,
         "actual physical head/CDF context conservation failed")
      row["passed"]=True;write_json(output/"manifest.json",report)
     inspection=bank.inspect_cache();report[f"sweep_{sweep}_all21_CPU_inventory"]=archive(output,f"sweep-{sweep}-whole-bank",inspection)
     require(inspection["retained_stage_count"]==21 and all(not s["state"]["cuda_storage_bytes"] for s in inspection["stages"]),"all21 roots must be retained CPU")
     report[f"sweep_{sweep}_CUDA_residual"]=assert_residual(torch,baseline,f"sweep{sweep}all21CPU")
     require(owner.migration.process_memory()["rss_peak_bytes"]<=100*owner.GIB,"measured peak RSS exceeds100GiB")
     del inspection;gc.collect()
    report["training_binding"]=bank.training_binding
    report["counters_before_release"]=dict(bank.counters)
    report["count_validation"]=validate_counts(bank.counters,bank.windows.windows)
    report["diagnostic_windows"]=archive(output,"all-observation-windows",bank.windows.windows)
    report["root_coverage_events"]=bank.coverage_events
    report["migration_count"]=len(bank.copy_events)
    require(report["migration_count"]==105 and all(e["comparison"]["passed"] and e["comparison"]["full_storage_bytes_equal"] for e in bank.copy_events),"all105 byte/layout/alias migration proofs required")
    report["final_active_precision_report"]=session.report()
    require(report["final_active_precision_report"]["aliases_and_flags_restored"] is False,"continuous active session cannot claim original alias restoration")
   report["window_counter_partition"]=bank.window_accounting
   report["after_release_CUDA_residual"]=assert_residual(torch,baseline,"all-stage-release-and-precision-exit")
   report["final_precision_report"]=session.report()
   require(report["final_precision_report"]["aliases_and_flags_restored"],"final051 alias/flag restoration failed")
  report["compatibility_adapter"]=dict(active)
  expected_calls=21*16+8+2*(21*40+16)
  require(active["patched_call_count"]==expected_calls and active["original_alias_restored"] and active["sdk_sources_unchanged"],"corrected CDF coverage/restoration failed")
  require(sources()==original_sources and sha(all21_artifact)==report["all21_approximation_manifest_sha256"],"source or ancestry changed")
  for path,wanted in ((qualification,report["qualification_sha256"]),(batch_qualification,report["batch_qualification_sha256"]),
       (Path(prefix_run)/"manifest.json",report["prefix_manifest_sha256"]),
       (Path(prefix_run)/"samples.npz",report["prefix_samples_sha256"]),(checkpoint,report["checkpoint_sha256"])):
   require(sha(path)==wanted,"authenticated scientific/source input changed before proof completion")
  require(global_rng==rng(),"global RNG changed across entire proof")
  report["final_CUDA_residual"]=assert_residual(torch,baseline,"CDFexit")
  report["terminal_memory_before_pass"]=owner.migration.process_memory()
  require(report["terminal_memory_before_pass"]["rss_peak_bytes"]<=100*owner.GIB,"terminal peak RSS exceeds100GiB")
  report.update(status="all21_fp16_bank_retention_screen_pass",retention_qualified=True,
    all21_retention_qualified=True,all21_fixed_conditional_bits_equal=True,warm_fits=0,warm_preprocess_refits=0,
    expected_patched_call_count=expected_calls,sources_unchanged=True,global_rng_preserved=True,
    peak_GPU_allocated_bytes=int(torch.cuda.max_memory_allocated()),peak_GPU_reserved_bytes=int(torch.cuda.max_memory_reserved()))
 except BaseException as error:
  report.update(status="failed_or_incomplete_fp16_bank_retention",error_type=type(error).__name__,error=str(error))
  if active is not None:report["compatibility_adapter"]=dict(active)
  if session is not None:report["final_precision_report"]=session.report()
  if bank is not None:report["bank_released_on_error"]=bool(bank.closed and not bank.entries)
  raise
 finally:
  report["elapsed_seconds"]=time.perf_counter()-start
  report["final_memory"]=owner.migration.process_memory()
  write_json(output/"manifest.json",report)
 return report

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 for name in ("banks","prefix-run","checkpoint","qualification","batch-qualification","sdk-root","all21-artifact","ledger","output"):
  parser.add_argument("--"+name,required=True)
 parser.add_argument("--max-seconds",type=int,default=2250)
 result=run(**vars(parser.parse_args()))
 print(json.dumps({k:result[k] for k in ("status","retention_qualified","elapsed_seconds")},indent=2))
if __name__=="__main__":main()
