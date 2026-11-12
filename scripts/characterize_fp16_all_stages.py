#!/usr/bin/env python3
"""All21 FP16 approximate conditional characterization, not qualification.

Built with PriorLabs-TabPFN. Original FP32 tolerances/failed flags remain visible.
No posterior vectors, new observations, reference integration, or sampler repair.
"""
from __future__ import annotations
import argparse,copy,gc,hashlib,importlib.util,json,os,random,resource,sys,time
from pathlib import Path
import numpy as np

SCHEMA="posteriorforge.fp16_all21_approximation_characterization.v1"
MODE="amp_fp16_prefill_query_v1"
BASE_SHA="c55c00a1697f8d4341880af18676c29b4d99551e131c89ff79bea238c693f8bd"
PORTABLE_SHA="3234694dc1d2bb242d0995d465ddc69250cf67fb05e9127e7380fe0eb70ea6ee"
PRECISION_SHA="051f477ff4b9ab8bbcfbaf3c1afa4dcf6b111d2f2fbcf9efb9c94a00d2b2ad3d"
UTILITY_SHA="6c927a5c278c7fc54b556ac557ecef64f5e6338f5ae13bf466f3b1e6a4640f90"
STAGES=tuple(range(21))
THRESHOLDS={"normalization_absolute":1e-5,"max_cdf_absolute":2e-5,"max_row_total_variation":1e-4,
            "quantile_coordinate_span_fraction":2e-6}
STAGE0_POLICY={"policy_id":"fp16_shared_raw_translated_probability_stack_mean_log_v1",
 "input":"8 actual corrected canonical translated FP32 probability tensors from declared singlehead calls",
 "order":"original ensemble order0..7","math":"torch.stack(rawP,dim=0).mean(dim=0).log().cpu()",
 "device":"original CUDA prediction device; captured CPU probability tensors transferred back without dtype/value conversion",
 "criterion":"actual physical criterion from the same prepared regressor",
 "original_full_API_identity_claimed":False,"sampler":"unchanged e71 finite_bar_icdf with caller-owned CPU FP32 uniforms"}

def require(value,message):
 if not value:raise RuntimeError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(name,expected):
 path=Path(__file__).with_name(name);require(sha(path)==expected,"pinned source changed: "+name)
 spec=importlib.util.spec_from_file_location("_fp16_all21_"+path.stem,path)
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def sources():
 from posteriorforge import execution_precision_probe as precision
 require(precision.source_sha256()==PRECISION_SHA,"frozen precision source changed")
 precision.require_sources()
 pins={"characterize_precision_approximation.py":BASE_SHA,"verify_portable_gqa_recovery.py":PORTABLE_SHA,
       "qualify_execution_cache.py":UTILITY_SHA}
 for name,expected in pins.items():require(sha(Path(__file__).with_name(name))==expected,"frozen helper source changed: "+name)
 return {"characterizer":sha(__file__),"precision":PRECISION_SHA,"base":BASE_SHA,"portable":PORTABLE_SHA,"utility":UTILITY_SHA}

def portable_qualification(qualification,batch_qualification,sdk_root,bindings):
 verifier=load("verify_portable_gqa_recovery.py",PORTABLE_SHA)
 receipt=verifier.verify_portable_recovery(qualification,batch_qualification,sdk_root=sdk_root)
 raw=Path(qualification).read_bytes()
 require(hashlib.sha256(raw).hexdigest()==verifier.RECOVERY_MANIFEST_SHA256,"verified a83 bytes changed before scientific binding")
 q=json.loads(raw)
 from posteriorforge import execution_gqa_hybrid as gqa
 expected={"status":"gqa_hybrid_screen_pass","full21_gqa_hybrid_qualified":True,
   "full21_component_qualified":True,"full21_sampler_qualified":True,"full21_qualified":False,
   "cache_only_full21_qualified":False,"thresholds":THRESHOLDS,"fallback_map":{"10":[5]},
   "module_sha256":gqa.execution_source_sha256(),**bindings}
 for key,value in expected.items():require(q.get(key)==value,"qualified FP32 scientific/source binding changed: "+key)
 return q,receipt

def finite_borders(prediction):
 b=prediction["criterion"].borders.detach().cpu().numpy()
 return bool(b.ndim==1 and len(b)>1 and np.isfinite(b).all() and np.all(np.diff(b)>0))

def assemble_shared_stage0(components,torch,device):
 require(len(components["members"])==8,"all8 actual raw components required")
 raw=[member["translated_probabilities"] for member in components["members"]]
 require(all(p.dtype==torch.float32 and tuple(p.shape)==tuple(raw[0].shape) and len(p)==1 for p in raw),
         "actual singleton FP32 canonical raw probabilities required")
 # No normalization/mass weighting/softmax of component logits here. These are
 # the actual corrected rawP, with the SDK's exact ensemble operation/order.
 logits=torch.stack([p.to(device=device) for p in raw],dim=0).mean(dim=0).log().cpu()
 return {**components,"logits":logits,"stage0_shared_policy":STAGE0_POLICY}

def repeat_singleton(prediction,count):
 result=dict(prediction);result["logits"]=prediction["logits"].repeat(count,1)
 result["members"]=[{**m,"raw_logits":m["raw_logits"].repeat(count,1),
                    "translated_probabilities":m["translated_probabilities"].repeat(count,1)}
                    for m in prediction["members"]]
 return result

def sampler_distribution_parity(prediction,torch):
 from posteriorforge import execution_mixture as mixture
 logits=prediction["logits"];criterion=prediction["criterion"];points=[]
 for u in (0.,.5,float(np.nextafter(np.float32(1),np.float32(0)))):
  uniforms=torch.full((len(logits),),u,dtype=torch.float32)
  actual=mixture.finite_bar_icdf(logits,criterion.borders,uniforms)
  native=criterion.icdf(logits,u)
  exact=torch.equal(actual.contiguous().view(torch.uint8),native.contiguous().view(torch.uint8))
  require(exact,"same-wholebatch native/e71 production inverse differs")
  points.append({"u":u,"rows":len(logits),"bitwise_equal":exact,
      "nonfinite_existing_arithmetic":int((~torch.isfinite(actual)).sum()),
      "outside_physical_borders":int(((actual<criterion.borders[0])|(actual>criterion.borders[-1])).sum())})
 return {"passed":True,"points":points,"same_whole_batch_logits_borders_uniforms":True,
         "u_one_not_called":True,"production_sampler_changed":False}

def sampler_parity(components,torch):
 result=[{"member":member,**sampler_distribution_parity(pred,torch)}
         for member,pred in enumerate(components["component_predictions"])]
 return {"passed":all(row["passed"] for row in result),"members":result,
         "uniform_domain":"production CPU FP32[0,1);0/.5/1-2^-24",
         "same_whole_batch_logits_borders_uniforms":True,"u_one_not_called":True,"production_sampler_changed":False}

def outcome(row,base):
 strict=all(c["passed"] for c in row["comparisons"].values()) and all(
   v["passed"] for checks in row["actual_mass_corrected_reconstruction"].values() for k,v in checks.items() if k!="scope")
 if row.get("stage0"):
  strict=strict and all(v["passed"] for v in row["stage0"]["comparisons"].values())
 hard=bool(row["rng_preserved"] and row["target_statistics_preserved"] and row["computed_prefill_witness"]["passed"]
   and row["computed_query_witness"]["passed"] and row["precision_session"]["aliases_and_flags_restored"]
   and row["G0_source_restoration_valid"] and all(v["passed"] for q in row["probability_validity"].values() for v in q.values())
   and all(base.comparison_hard_validity(c) for c in row["comparisons"].values())
   and all(row["border_validity"].values()) and all(v["passed"] for v in row["sampler_parity"].values())
   and all(v["passed"] for checks in row["Cp_self_reconstruction"].values() for k,v in checks.items() if k!="scope"))
 if row.get("stage0"):hard=hard and row["stage0"]["hard_validity_passed"]
 return {"hard_validity_passed":hard,"strict_numeric_comparisons_passed":bool(strict),
         "strict_screen_passed":bool(strict and hard),"strict_failure_preserved":not (strict and hard),
         "qualified":False,"approximation_accepted":False}

def run(*,banks,prefix_run,checkpoint,qualification,batch_qualification,sdk_root,ledger,output,
        max_seconds=1050,timing_repetitions=0):
 from posteriorforge.artifacts import runtime_identity,write_json
 from posteriorforge.reference_grid import _require_budgeted_gpu
 from posteriorforge import execution_precision_probe as precision,execution_gqa_hybrid as gqa
 from posteriorforge.banks import load_context,load_heldout,derive_seed
 from posteriorforge.science.analytical_benchmark_v2 import build_problem,array_descriptor
 from posteriorforge.inference import CHECKPOINT_SHA256,support_mask
 from posteriorforge.compat_cdf import finite_cdf_compatibility
 from posteriorforge.gate import verify_run_binding
 require(type(max_seconds) is int and 300<=max_seconds<=1050,"bounded max_seconds300..1050 required")
 require(type(timing_repetitions) is int and timing_repetitions in (0,1,3),"timing repetitions0/1/3 required")
 original_sources=sources();base=load("characterize_precision_approximation.py",BASE_SHA)
 utility=load("qualify_execution_cache.py",UTILITY_SHA);require(utility.THRESHOLDS==THRESHOLDS,"strict thresholds changed")
 budget=_require_budgeted_gpu(ledger);require(budget["gpus"]==1,"one reserved GPU required")
 start=time.perf_counter();output=Path(output);output.mkdir(parents=True,exist_ok=False);engine=None;active=None
 report={**runtime_identity(),"schema":SCHEMA,"status":"running","mode":MODE,"approximate_experiment":True,
   "attribution":"Built with PriorLabs-TabPFN","qualified":False,"full21_qualified":False,
   "all21_hard_validity_passed":False,"posterior_vectors_generated":0,"reference_integrations":0,
   "scientific_gate_evaluated":False,"requested_stages":list(STAGES),"completed_stages":[],
   "thresholds":THRESHOLDS,"row_counts":[1024],"stress_rows":8,"stage_results":[],"sources":original_sources,
   "budget_binding":budget,"precision_module":precision.metadata(MODE),"stage0_policy":STAGE0_POLICY,
   "timing_repetitions":timing_repetitions,"timing_order":"G0 first; Cp later, no randomized/interleaved timing",
   "approximate_full_run_review_policy":base.APPROXIMATE_FULL_RUN_REVIEW,
   "scope":"all21 approximate cache plus declared view-layout conditionals; not KV-only/exact/posterior qualification",
   "negative_infinite_logits":"original zero/cancelled bins retained; NaN/+inf logits or invalid probability rows rejected"}
 write_json(output/"manifest.json",report)
 try:
  import torch
  require(torch.cuda.is_available() and torch.cuda.device_count()==1,"one visible CUDA GPU required")
  require(torch.get_float32_matmul_precision()=="highest" and not torch.backends.cuda.matmul.allow_tf32 and not torch.is_autocast_enabled("cuda"),"strict FP32 initial flags required")
  seed=derive_seed(20261111,"fp16-all21-conditional-characterizer",case=0)
  random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
  torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS","16")))
  report.update(seed=seed,rng_role="fp16-all21-conditional-characterizer",posterior_sampling_rng_used=False)
  require(sha(checkpoint)==CHECKPOINT_SHA256,"immutable checkpoint required")
  context=load_context(banks,20261111,4096);case=load_heldout(banks,0);query=case.obs.reshape(1,33);problem=build_problem()
  bindings={"checkpoint_sha256":CHECKPOINT_SHA256,"scientific_fingerprint":problem.scientific_fingerprint,
    "parameter_order":list(problem.parameter_order),"context_theta":array_descriptor(context.theta),
    "context_obs":array_descriptor(context.obs),"query":array_descriptor(query)}
  q,receipt=portable_qualification(qualification,batch_qualification,sdk_root,bindings)
  report.update(**bindings,qualification_sha256=sha(qualification),batch_qualification_sha256=sha(batch_qualification),
                portable_recovery_verification=receipt)
  prefix_run=Path(prefix_run);pm=json.loads((prefix_run/"manifest.json").read_text())
  require(sha(prefix_run/"manifest.json")==q["prefix_manifest_sha256"] and sha(prefix_run/"samples.npz")==q["prefix_samples_sha256"],"prefix custody changed")
  verify_run_binding(pm,context,case.obs,20261111,0)
  with np.load(prefix_run/"samples.npz",allow_pickle=False) as saved:values=saved["samples"].copy()
  require(values.shape==(50000,21) and support_mask(values).all(),"immutable supported complete prefixes required")
  positions=np.asarray(q["fixed_prefix_row_indices"],int);prefixes=values[positions]
  theta=context.theta.astype(np.float32);obs=context.obs.astype(np.float32)
  noise=[problem.parameter_order.index(n) for n in problem.prior.noise_names]
  report["fixed_prefix_rows"]=positions.tolist();expected_calls=0
  with finite_cdf_compatibility() as active:
   for stage in STAGES:
    require(time.perf_counter()+25<start+max_seconds,"bounded deadline; partial coverage never promoted")
    queries,stress=utility.fixed_queries(query,prefixes,theta,stage,noise,problem.prior.bounds)
    sets={"1024":queries,"stress":stress};timing_sets={"128":queries[:128],"1024":queries}
    row={"stage_index":stage,"parameter":problem.parameter_order[stage],"hard_validity_passed":False}
    report["stage_results"].append(row);report["active_stage"]=stage;write_json(output/"manifest.json",report)
    rng=utility.rng_identity(utility.rng_state(torch));before_calls=active["patched_call_count"]
    engine=gqa.GQAHybridStageExecutor(checkpoint,cdf_descriptor=active,query_batch_size=1024)
    torch.cuda.synchronize();t=time.perf_counter();row["G0_prepare"]=engine.prepare(theta,obs,stage);torch.cuda.synchronize()
    row["G0_prepare_seconds"]=time.perf_counter()-t
    g0={name:precision.capture_members(engine,x) for name,x in sets.items()}
    row["G0_model"]=base.model_summary(engine.regressor,utility)
    if stage==10:row["G0_fallback_model"]=base.model_summary(engine.uncached_regressor,utility)
    g0single=engine.predict(queries[:1],output_type="full",capture_members=True) if stage==0 else None
    if timing_repetitions:row["G0_capture_free_timings"]=base.timed_all8(engine,timing_sets,torch,repetitions=timing_repetitions)
    gcounters=dict(engine.gqa_counters);row["G0_execution_counters"]=gcounters
    row["G0_source_restoration_valid"]=bool(gcounters["all_contexts_restored"] and gcounters["all_sources_unchanged"] and gcounters["expanded_attention_calls"]>0)
    engine.close();engine=None;gc.collect()
    with precision.PrecisionSession(MODE) as session:
     engine=precision.PrecisionStageExecutor(checkpoint,cdf_descriptor=active,query_batch_size=1024)
     torch.cuda.synchronize();t=time.perf_counter();row["Cp_prepare"]=engine.prepare(theta,obs,stage);torch.cuda.synchronize()
     row["Cp_prepare_seconds"]=time.perf_counter()-t
     row["computed_prefill_witness"]=precision.prefill_witness_passed(session.report(),stage)
     require(row["computed_prefill_witness"]["passed"],"actual computed FP16 prefill/fallback/storage not witnessed")
     cp={name:precision.capture_members(engine,x) for name,x in sets.items()}
     row["Cp_model"]=base.model_summary(engine.regressor,utility)
     if stage==10:row["Cp_fallback_model"]=base.model_summary(engine.uncached_regressor,utility)
     fields=("target_mean","target_sd","canonical_borders_dtype","physical_borders_dtype","temperature")
     row["target_statistics_preserved"]=all(row["G0_model"][k]==row["Cp_model"][k] for k in fields)
     if stage==10:row["target_statistics_preserved"] &= all(row["G0_fallback_model"][k]==row["Cp_fallback_model"][k] for k in fields)
     row["comparisons"]={name:utility.compare_predictions(g0[name],cp[name]) for name in sets}
     row["actual_mass_corrected_reconstruction"]={name:base.reconstruction_checks(cp[name],g0[name],utility) for name in sets}
     row["Cp_self_reconstruction"]={name:base.reconstruction_checks(cp[name],cp[name],utility) for name in sets}
     row["probability_validity"]={name:{"G0":base.hard_prediction_validity(g0[name]),"Cp":base.hard_prediction_validity(cp[name])} for name in sets}
     row["border_validity"]={name:finite_borders(g0[name]) and finite_borders(cp[name]) for name in sets}
     row["sampler_parity"]={name:sampler_parity(cp[name],torch) for name in sets}
     if stage==0:
      require(np.array_equal(queries,np.repeat(queries[:1],len(queries),axis=0)),"stage0 queries must be identical")
      singleton=assemble_shared_stage0(precision.capture_members(engine,queries[:1]),torch,engine.base.device)
      comparisons={"G0_actual_original_full_singleton_vs_Cp_constructed":utility.compare_predictions(g0single,singleton),
        "Cp_singleton_repeated_vs_Cp_N1024":utility.compare_predictions(repeat_singleton(singleton,len(queries)),cp["1024"])}
      shared_parity=sampler_distribution_parity(singleton,torch)
      shared_self_reconstruction=base.reconstruction_checks(singleton,singleton,utility)
      row["stage0"]={"policy":STAGE0_POLICY,"comparisons":comparisons,"shared_final_sampler_parity":shared_parity,
        "shared_self_reconstruction":shared_self_reconstruction,
        "hard_validity_passed":bool(base.hard_prediction_validity(singleton)["passed"] and finite_borders(singleton)
          and all(base.comparison_hard_validity(v) for v in comparisons.values())
          and shared_parity["passed"] and all(v["passed"] for k,v in shared_self_reconstruction.items() if k!="scope")),
        "G0_reference_is_actual_original_full_API":True,"Cp_original_full_API_identity_claimed":False}
     row["computed_query_witness"]=precision.query_witness_passed(session.report(),stage)
     require(row["computed_query_witness"]["passed"],"actual FP16 query/F32 fallback not witnessed")
     if timing_repetitions:row["Cp_capture_free_timings"]=base.timed_all8(engine,timing_sets,torch,repetitions=timing_repetitions)
     row["hybrid_execution_counters"]=dict(engine.hybrid_counters);engine.close();engine=None
    row["precision_session"]=session.report();row["rng_preserved"]=rng==utility.rng_identity(utility.rng_state(torch))
    row.update(outcome(row,base));row["corrected_CDF_calls"]=active["patched_call_count"]-before_calls
    wanted=32+(16 if stage==0 else 0)+(32*(1+timing_repetitions) if timing_repetitions else 0)
    require(row["corrected_CDF_calls"]==wanted,"exact corrected CDF call coverage changed")
    expected_calls+=wanted;report["completed_stages"].append(stage);write_json(output/"manifest.json",report)
    require(row["hard_validity_passed"],"hard probability/state/source/arithmetic condition failed")
    del g0,cp,g0single;gc.collect()
    require(time.perf_counter()<start+max_seconds,"bounded deadline; partial coverage never promoted")
  report["compatibility_adapter"]=dict(active)
  require(active["original_alias_restored"] and active["sdk_sources_unchanged"] and active["patched_call_count"]==expected_calls,"actual CDF coverage/restoration failed")
  require(sources()==original_sources,"frozen source changed during characterization")
  require(sha(qualification)==report["qualification_sha256"] and sha(batch_qualification)==report["batch_qualification_sha256"],"proof changed")
  report.update(status="all21_fp16_approximation_characterized_not_qualified",all21_hard_validity_passed=True,
      all21_strict_screen_passed=all(r["strict_screen_passed"] for r in report["stage_results"]),
      strict_failed_stages=[r["stage_index"] for r in report["stage_results"] if not r["strict_screen_passed"]],
      expected_patched_call_count=expected_calls,sources_unchanged=True)
 except BaseException as error:
  report.update(status="hard_failure_or_incomplete_characterization",error_type=type(error).__name__,error=str(error))
  if active is not None:report["compatibility_adapter"]=dict(active)
  raise
 finally:
  if engine is not None:engine.close()
  report["elapsed_seconds"]=time.perf_counter()-start
  report["peak_cpu_rss_bytes"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=="darwin" else 1024)
  write_json(output/"manifest.json",report)
 return report

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 for name in ("banks","prefix-run","checkpoint","qualification","batch-qualification","sdk-root","ledger","output"):parser.add_argument("--"+name,required=True)
 parser.add_argument("--max-seconds",type=int,default=1050)
 parser.add_argument("--timing-repetitions",type=int,choices=(0,1,3),default=0)
 result=run(**vars(parser.parse_args()))
 print(json.dumps({k:result[k] for k in ("status","completed_stages","all21_hard_validity_passed","elapsed_seconds")},indent=2))
if __name__=="__main__":main()
