#!/usr/bin/env python3
"""Independent CPU review of the explicitly approximate FP16/FIFO experiment.

Built with PriorLabs-TabPFN. Saved hard-screen and production witnesses are
replayed without model calls. Physical failures and both same-case comparison
flags remain visible; no exact-FP32 or joint-equivalence qualification is issued.
"""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,subprocess
from pathlib import Path
import numpy as np
from posteriorforge.artifacts import sha256_file,write_json
from posteriorforge.banks import load_context,load_heldout,derive_seed
from posteriorforge.inference import CHECKPOINT_SHA256,frozen_regressor_settings,support_mask
from posteriorforge.science.analytical_benchmark_v2 import array_descriptor,build_problem
from posteriorforge.validation import sample_integrity

SCHEMA="posteriorforge.fp16_experiment_review.v1"
PRODUCER_SHA="2280e1ceef807f05a264b5318be18eb8bee7c52c8bab21884d6682f84cafcba9"
PRODUCER_COMMIT="2b27bea496b4b99b85a4436a6e2cf9cbcc3ef5b7"
ALL21_SHA="f01bfddb87dcd16c9bd7c9df15f482ed3ea46756b1d84f3c04f6584644a0687a"
ALL21_SOURCE="20e64a64357de7862c20b2fa846c6d128c181ca564c261800b626e6c4bd32208"
PRECISION_SHA="051f477ff4b9ab8bbcfbaf3c1afa4dcf6b111d2f2fbcf9efb9c94a00d2b2ad3d"
BASE_SHA="c55c00a1697f8d4341880af18676c29b4d99551e131c89ff79bea238c693f8bd"
METRICS_SHA="1c9a185aa5ce28a1d5fa56dbd012522d46c4a2f5c55faf4e0997f63ad0285753"
MODE="amp_fp16_prefill_query_v1"
ROLES={"head_proposal":"execution-head-proposal","head_acceptance":"execution-head-acceptance","component_uniform":"execution-component-uniform"}
DTYPES={"head_proposal":"torch.int64","head_acceptance":"torch.float64","component_uniform":"torch.float32"}
FIFO_CONTROL_SHA="c0937b4b08c23d78829695b1207775cefbd3d161b592cb0139d25241927f1e42"

def require(value,message):
 if not value:raise ValueError(message)

def load(name,digest):
 path=Path(__file__).with_name(name);require(sha256_file(path)==digest,"reviewed helper source changed: "+name)
 spec=importlib.util.spec_from_file_location("_fp16_review_"+path.stem,path)
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def sources():
 from posteriorforge import performance_fp16_experiment as producer,execution_precision_probe as precision
 require(producer.source_sha256()==PRODUCER_SHA and precision.source_sha256()==PRECISION_SHA,"reviewed experimental source changed")
 repo=Path(__file__).resolve().parents[1]
 content=subprocess.run(["git","-C",str(repo),"show",PRODUCER_COMMIT+":src/posteriorforge/performance_fp16_experiment.py"],check=True,capture_output=True).stdout
 require(hashlib.sha256(content).hexdigest()==PRODUCER_SHA,"pinned producer Git blob differs")
 precision.require_sources()
 metrics=load("evaluate_execution_variant.py",METRICS_SHA)
 all21=load("characterize_fp16_all_stages.py",ALL21_SOURCE);base=load("characterize_precision_approximation.py",BASE_SHA)
 return producer,precision,metrics,all21,base

def probability(item):
 require(item.get("passed") is True,"saved hard probability self-check failed")
 for key,limit in (("max_normalization_error",1e-5),("max_cdf_absolute_error",2e-5),("max_row_total_variation",1e-4)):
  value=item.get(key);require(type(value) in (int,float) and np.isfinite(value) and 0<=value<=limit,"hard self-check numeric bound differs: "+key)
 require(np.isfinite(item.get("minimum_probability",np.nan)) and item["minimum_probability"]>=0,"negative/nonfinite saved probabilities")

def raw_probability(item):
 require(item.get("passed") is True and item.get("mass_envelope")==1.00001
   and item.get("normalization_absolute")==1e-5 and 0<=item.get("max_normalization_error",np.inf)<=1e-5,
   "all21 raw mass/normalization bound differs")
 require(len(item.get("raw_head_masses",[]))==len(item.get("actual_component_mass_checks",[]))==8,"all8 actual mass records required")
 for mass in item["raw_head_masses"]:
  require(0<mass["min"]<=mass["max"]<=1.00001,"all21 mass outside unchanged envelope")
 for mass in item["actual_component_mass_checks"]:
  require(mass["passed"] is True and np.isfinite(mass["max_fp64_sum_error"])
    and 0<=mass["max_fp64_sum_error"]<=mass["fp64_roundoff_bound"]
    and mass["fp64_roundoff_bound"]==8*5000*np.finfo(np.float64).eps,"all21 actual FP64 mass arithmetic differs")

def inverse_parity(item,rows,*,members=True):
 require(item.get("passed") is True and item.get("same_whole_batch_logits_borders_uniforms") is True
   and item.get("u_one_not_called") is True and item.get("production_sampler_changed") is False,
   "same-batch native/production inverse scope differs")
 if members:
  records=item.get("members",[])
  require([r.get("member") for r in records]==list(range(8)),"all8 inverse distributions required")
  for record in records:inverse_parity(record,rows,members=False)
  return
 points=item.get("points",[])
 require([p.get("u") for p in points]==[0.,.5,float(np.nextafter(np.float32(1),np.float32(0)))],
   "central/production-max inverse evidence differs")
 for point in points:
  require(point.get("rows")==rows and point.get("bitwise_equal") is True,"same-wholebatch inverse parity failed")
  for key in ("nonfinite_existing_arithmetic","outside_physical_borders"):
   require(type(point.get(key)) is int and 0<=point[key]<=rows,"original inverse pathology count invalid")

def all21_replay(path,bindings,producer,precision,all21,base):
 require(sha256_file(path)==ALL21_SHA,"immutable all21 hard-screen artifact differs")
 q=producer.validate_all21(path,bindings)
 require(q["timing_repetitions"]==0,"fixed architectural witness factors require untimed all21 screen")
 for j,row in enumerate(q["stage_results"]):
  calculated=all21.outcome(row,base)
  require(calculated["hard_validity_passed"] and all(row.get(k)==v for k,v in calculated.items()),"all21 hard/strict decision replay differs")
  require(row["computed_prefill_witness"]==precision.prefill_witness_passed(row["precision_session"],j)
    and row["computed_query_witness"]==precision.query_witness_passed(row["precision_session"],j),"all21 actual dtype evidence differs")
  for name,rows in (("1024",1024),("stress",8)):
   for family in ("Cp","G0"):raw_probability(row["probability_validity"][name][family])
   inverse_parity(row["sampler_parity"][name],rows)
  for checks in row["Cp_self_reconstruction"].values():
   for key,item in checks.items():
    if key!="scope":probability(item)
  if j==0:
   inverse_parity(row["stage0"]["shared_final_sampler_parity"],1,members=False)
   for key,item in row["stage0"]["shared_self_reconstruction"].items():
    if key!="scope":probability(item)
 require(q["strict_failed_stages"]==list(range(21)) and q["all21_strict_screen_passed"] is False
   and q["qualified"] is False and q["full21_qualified"] is False,"existing strict failures or unqualified scope relabeled")
 return q

def histogram_counts(sample,stage,rows):
 if stage==0:
  require(sample.get("stage0_full_mixture_shared") is True and sample.get("density_scope")=="proposal-wave-local; no cross-request/case reuse"
    and sample.get("model_query_chunks")==1,"shared stage0 density scope differs")
  require(sample["neural_query_batch_sizes_by_member"]=={str(i):{"1":1} for i in range(8)},"stage0 must evaluate8 real singleton heads")
  return [1]*8,[1]*8,0,0
 calls=[];counts=[]
 assignments=sample["head_assignment_counts_by_member"];rejects=sample["head_rejection_counts_by_member"]
 require(len(assignments)==len(rejects)==8 and sum(assignments)==sample["head_candidates"]
   and sum(rejects)==sample["head_rejections"] and sum(assignments)-sum(rejects)==rows,"per-row head retry conservation differs")
 for member in range(8):
  hist=sample["neural_query_batch_sizes_by_member"][str(member)]
  require(all(k.isdigit() and 1<=int(k)<=1024 and type(v) is int and v>0 for k,v in hist.items()),"actual member batches invalid")
  calls.append(sum(hist.values()));counts.append(sum(int(k)*v for k,v in hist.items()))
  require(counts[-1]==assignments[member],"actual queried rows differ from head assignments")
 require(sum(calls)==sample["model_query_chunks"],"model chunks differ from all8 member histograms")
 return calls,counts,sample["head_candidates"],sample["head_rejections"]

def scope_review(scope,event,screen_row,*,precision):
 j=scope["stage_index"];report=scope["precision_session"]
 require(scope.get("failed") is False and scope["queried_members"]==list(range(8)),"successful production stage/all8 members required")
 for key,value in precision.metadata(MODE).items():require(report.get(key)==value,"actual051 descriptor source/policy differs: "+key)
 require(report["aliases_and_flags_restored"] is True and report["computed_kv_not_inferred_from_storage"] is True
   and report["query_views"] is True and report["counters"].get("witness_histogram_overflow",0)==0,"actual precision aliases/flags/witness capacity failed")
 require(report["private_clones"]==screen_row["precision_session"]["private_clones"]
   and report["torch_autocast_source_sha256"]==screen_row["precision_session"]["torch_autocast_source_sha256"]
   and report["torch_version"]==screen_row["precision_session"]["torch_version"],"actual private/runtime source lineage differs")
 prefill=precision.prefill_witness_passed(report,j);query=precision.query_witness_passed(report,j)
 require(prefill["passed"] and query["passed"] and prefill==scope["prefill"],"actual computed prefill/query FP16/F32 fallback witnesses failed")
 calls,rows,heads,rejects=histogram_counts(event["sampler"],j,event["proposed_rows"])
 require(scope["actual_head_contexts"]==sum(calls),"stage physical head count differs from sampler callbacks")
 routes=scope["hybrid_route_counters"];uncached=calls[5] if j==10 else 0;uncached_rows=rows[5] if j==10 else 0
 require(routes["module_sha256"]==PRECISION_SHA and routes["forward_precision_policy"]==MODE
   and routes["postprocessing_math_dtype"]=="torch.float32" and routes["qualified_fp32"] is False
   and routes["resident_component_predict_count"]==sum(calls)-uncached and routes.get("uncached_predict_count",0)==uncached
   and routes.get("uncached_query_rows",0)==uncached_rows and routes.get("uncached_fit_count",0)==int(j==10)
   and routes["original_preprocessing_lists_restored"] is True and routes["original_translator_alias_restored"] is True,
   "actual cached/10:5 original FP32 route counters differ")
 model=scope["model"];contract=model["contract"];expected=screen_row["Cp_model"]
 for key in ("target_mean","target_sd","canonical_borders_dtype","physical_borders_dtype","temperature"):
  require(model[key]==expected[key],"frozen target statistics/FP32 postprocessing differs: "+key)
 require(contract["input_normalization_train_only_verified"] is True and contract["autocast"] is True and contract["forced_inference_dtype"]=="None"
   and contract["preprocessor_sha256"]==expected["contract"]["preprocessor_sha256"] and len(contract["preprocessor_sha256"])==8,
   "production fitted preprocessor/train-only AMP contract differs")
 normalizers=[m for m in contract["modules"] if m["type"]=="InputNormalizationEncoderStep"]
 require(normalizers and all(m["settings"].get("normalize_on_train_only") is True
   and all(v["dtype"]=="torch.float32" for v in m["tensors"].values()) for m in normalizers),"normalization left original FP32/train-only policy")
 require(event["precision"]=={"autocast":True,"forced_inference_dtype":"None"},"actual cached stage AMP metadata differs")
 cache=event["cache_after"]
 require(set(cache["parameter_count_by_member"])==set(map(str,range(8)))
   and set(cache["parameter_count_by_member"].values())=={11081864} and cache["parameter_dtypes"]==["torch.float32"],"all8 frozen FP32 parameter banks required")
 nscreen=3 if j==0 else 2
 for member in range(8):
  observed=[w for w in report["witnesses"] if w["phase"]=="query" and w["stage"]==j and w["member"]==member]
  old=[w for w in screen_row["precision_session"]["witnesses"] if w["phase"]=="query" and w["stage"]==j and w["member"]==member]
  factor=sum(w["count"] for w in old)
  require(factor%nscreen==0 and sum(w["count"] for w in observed)==calls[member]*(factor//nscreen),
          "per-member actual projection-event coverage differs from physical calls")
  strict=j==10 and member==5
  for witness in observed:
   values=[v for v in witness["projection_dtypes"].values() if v is not None]
   require(witness["allow_tf32"] is False and witness["matmul_precision"]=="highest"
     and witness["autocast_enabled"] is (not strict),"actual query flags differ")
   require(values and all(v in ("torch.float32",) if strict else v in ("torch.float32","torch.float16") for v in values),
           "actual query projected dtype outside declared family")
   if not strict:require("torch.float16" in values and witness["autocast_dtype"]=="torch.float16","actual FP16 computation absent")
 counts=report["counters"]
 require(sum(w["count"] for w in report["witnesses"])==counts["qkv_calls"]==counts["attention_calls"],"computed projection/attention histogram totals differ")
 require(prefill==precision.prefill_witness_passed(screen_row["precision_session"],j),"production prefill/stored-cache witness differs from hard screen")
 require(counts["expanded_views"]>0,"actual cached view expansion absent")
 return {"stage_index":j,"actual_heads":sum(calls),"cached_heads":sum(calls)-uncached,"original_FP32_fallback_heads":uncached,
   "sampler_callbacks":1 if j==0 else sum(calls),"head_proposals":heads,"head_rejections":rejects}

def authenticate(manifest_path,*,context,query,approximation,qualification,batch_qualification,sdk_root):
 producer,precision,metrics,all21,base=sources();path=Path(manifest_path);m=json.loads(path.read_text())
 expected={"schema":"posteriorforge.inference.fp16_fifo_experiment.v1","variant":"execution_fp16_fifo_experiment_v1","status":"complete",
   "runner_source_sha256":PRODUCER_SHA,"approximate_experiment":True,"qualified_fp32":False,"qualification_full21":False,
   "qualification_cache_only_full21":False,"approximation_all21_hard_screen_passed":True,"all21_strict_screen_passed":False,
   "strict_failed_stages":list(range(21)),"approximation_manifest_sha256":ALL21_SHA,"approximation_producer_sha256":ALL21_SOURCE,
   "context_seed":20261111,"case":0,"context_size":4096,"query_batch_size":1024,"engine_query_batch_cap":1024,"requested_count":50000,
   "benchmark_total_count":50000,"rank":0,"world_size":1,"end_to_end":True,"scientific_run":True,"proposal_lookahead":1200,
   "microbatch":50000,"max_iter_rejection":None,"log_density":"not_used","context_filter":"no_filtering",
   "observation_preprocessing":"raw; TabPFN frozen preprocessing","performance_first_authorized":True,"absolute_accuracy_gate_required_for_performance":False,
   "partial_vector_filtering":False,"clipping_or_repair_performed":False,"tail_family_changed":False,"original_gate_unchanged":True,
   "sources_unchanged":True,"completed_stage_indices":list(range(21)),
   "known_physical_control_status":"model_failure","source_checkout_required":True,
   "relative_distribution_review_thresholds":{"CDF_change":0.01,"mean_change_refSD":0.05,"SD_relative_change":0.05}}
 for key,value in expected.items():require(type(m.get(key)) is type(value) and m.get(key)==value,"FP16 experimental manifest differs: "+key)
 require(m["git_revision"]==PRODUCER_COMMIT and type(m.get("repetition")) is int and m["repetition"]>=0,"explicit producer Git/repetition binding required")
 from posteriorforge.inference_cdf import guard_baseline_source
 require(m["baseline_execution_source"]==guard_baseline_source(),"six upstream/default science source guard differs")
 problem=build_problem();bindings={"checkpoint_sha256":CHECKPOINT_SHA256,"scientific_fingerprint":problem.scientific_fingerprint,
   "parameter_order":list(problem.parameter_order),"context_theta":array_descriptor(context.theta),"context_obs":array_descriptor(context.obs),"query":array_descriptor(query)}
 for key,value in bindings.items():require(m.get(key)==value,"actual frozen science/bank/query binding differs: "+key)
 q=all21_replay(approximation,bindings,producer,precision,all21,base)
 _,receipt=all21.portable_qualification(qualification,batch_qualification,sdk_root,bindings)
 require(sha256_file(qualification)==q["qualification_sha256"] and sha256_file(batch_qualification)==q["batch_qualification_sha256"],"FP32 source ancestry differs")
 require(m["execution_policy"]==producer.execution_metadata(),"separate FP16 execution policy/source differs")
 settings=m["regressor_settings"]
 require(settings=={**frozen_regressor_settings(settings["model_path"]),"fit_mode":"fit_with_cache","inference_precision":"autocast"}
   and sha256_file(settings["model_path"])==CHECKPOINT_SHA256,"actual frozen weights/scientific regressor settings differ")
 frozen=producer.frozen_sampling();seeds=frozen.worker_stream_seeds(20261111,0,m["repetition"],0,1)
 require(m["seed_streams"]==seeds and m["seed"]==seeds["component_uniform"],"sampling stream seed derivation differs")
 policy=m["sampling_policy"]
 from posteriorforge import execution_mixture as mixture
 require(policy["frozen_component_policy_provenance"]==mixture.execution_metadata()
   and policy["forward_precision_policy"]==MODE and policy["postprocessing_math_dtype"]=="torch.float32"
   and policy["head_policy"]=="raw_mass_rejection_v1" and policy["mass_envelope"]==1.00001
   and policy["uniform_assignment"]=="preassigned_per_stage_by_physical_row" and policy["head_retry_preserves_row_uniform"] is True,
   "FP16 experiment primitive sampling policy differs")
 require(policy["stage0_dedup"]=={"enabled":True,"scope":"constructed_shared_raw_probability_mixture","policy":all21.STAGE0_POLICY,
   "component_distribution_dedup":False,"density_scope":"one proposal wave; no cross-request/case reuse"},"new constructed stage0 policy differs")
 require(set(policy["streams"])==set(ROLES),"three separate RNG roles required")
 for name,role in ROLES.items():require(policy["streams"][name]=={"role":role,"seed":seeds[name],"dtype":DTYPES[name],"device":"cpu","generator":"torch.Generator"},"actual sampling generator policy differs")
 record=m["proposal_buffer"]
 require(m["proposal_buffer_policy"]==frozen.proposal_buffer_policy(1200),"FIFO policy/source differs")
 scope=frozen.proposal_buffer_scope(bindings,context_seed=20261111,case=0,repetition=m["repetition"],rank=0,world_size=1,seeds=seeds)
 frozen.validate_proposal_buffer_record(record,world_size=1,rank=0,scope=scope)
 require(m["proposal_wave_sizes"]==record["actual_generated_wave_sizes"],"provider/FIFO actual generation differs")
 counts=record["original_sampler_counts"]
 require(m["rejection"]=={"support_calls":counts["original_support_calls"],"proposed":counts["original_checked_count"],
   "accepted_before_final_trim":counts["accepted_before_trim"],"returned":50000},"original whole-vector support/trim counts differ")
 events=m["stage_events"];execution=m["execution_counters"];scopes=execution["completed_stage_scopes"];waves=len(m["proposal_wave_sizes"])
 require(len(events)==len(scopes)==execution["prepare_count"]==m["expected_stage_prepare_count"]==21*waves
   and m["generated_proposal_wave_count"]==waves and execution["all_stage_aliases_restored"] is True,"all21 production scope/generation coverage differs")
 reviewed=[]
 for index,(event,production) in enumerate(zip(events,scopes)):
  wave,j=divmod(index,21)
  require(event["proposal_round"]==wave+1 and event["stage_index"]==production["stage_index"]==j
    and event["proposed_rows"]==m["proposal_wave_sizes"][wave] and event["query_batch_size"]==1024
    and event["sampler"]["uniform_draws"]==event["proposed_rows"],"actual production stage/row order differs")
  reviewed.append(scope_review(production,event,q["stage_results"][j],precision=precision))
 physical=sum(r["actual_heads"] for r in reviewed);callbacks=sum(r["sampler_callbacks"] for r in reviewed)
 sampling={"stage0_shared_predictions":waves,"component_predictions":callbacks-waves,"head_candidates":sum(r["head_proposals"] for r in reviewed),"head_rejections":sum(r["head_rejections"] for r in reviewed)}
 require(m["sampling_counters"]==sampling and execution["actual_head_contexts"]==physical and execution["shared_densities"]==waves,"actual physical/shared/sampler counts differ")
 require(m["actual_prediction_checks"]=={"sampler_callback_calls":callbacks,"actual_head_contexts":physical,"shared_densities":waves,"original_full_API_identity_claimed":False},"per-call actual precision proof coverage differs")
 descriptor=metrics.bindings.validate_adapter_descriptor(m["compatibility_adapter"])
 require(descriptor["patched_call_count"]==physical==m["expected_patched_call_count"],"physical heads and corrected-CDF counter differ")
 values=metrics.bindings.load_samples(path,m)
 require(sample_integrity(values,expected_count=50000,dimension=21,support_mask=support_mask(values))["passed"],"invalid full50000 supported finite cohort")
 return m,values,{"source_commit":PRODUCER_COMMIT,"producer_sha256":PRODUCER_SHA,"all21_manifest_sha256":ALL21_SHA,
   "all21_hard_predicates_replayed":True,"strict_failed_stages_preserved":q["strict_failed_stages"],"production_scope_reviews":reviewed,
   "actual_physical_heads":physical,"sampler_callbacks":callbacks,"corrected_CDF_calls":descriptor["patched_call_count"],
   "portable_FP32_ancestry":receipt,"all50000_integrity_passed":True,"new_exact_FP32_or_joint_qualification":False}

def run(*,root,manifest,approximation,qualification,batch_qualification,sdk_root,output):
 root=Path(root);results=root/"results";producer,precision,metrics,all21,base=sources();binder=metrics.bindings
 input_paths=[Path(manifest),Path(approximation),Path(qualification),Path(batch_qualification)]
 initial_inputs={str(p.resolve()):sha256_file(p) for p in input_paths}
 gatepath=results/"gate-final.json";require(sha256_file(gatepath)==binder.BASELINE_GATE_SHA256,"immutable original gate differs")
 gate=json.loads(gatepath.read_text());context=load_context(results/"banks-v1",20261111,4096);query=load_heldout(results/"banks-v1",0).obs.reshape(1,33)
 _,_,reference,custody=binder.reference_for_case(results,gate,query,0)
 m,values,authentication=authenticate(manifest,context=context,query=query,approximation=approximation,qualification=qualification,batch_qualification=batch_qualification,sdk_root=sdk_root)
 binder.validate_run_reference_binding(m,custody)
 primary_path=results/"cdf-arithmetic-v1/context-4096/seed-20261111-case-0/manifest.json"
 require(sha256_file(primary_path)==metrics.CONTROL_MANIFEST_SHA256,"immutable corrected primary differs")
 primary=json.loads(primary_path.read_text());binder.validate_pilot_header(primary,seed=20261111,case=0,context_size=4096)
 binder.validate_run_reference_binding(primary,custody);primary_values=binder.load_samples(primary_path,primary)
 fifo_path=results/"performance-first-v1/gqa-fifo-fp32-batch-1024/rep-0/seed-20261111-case-0/manifest.json"
 require(sha256_file(fifo_path)==FIFO_CONTROL_SHA,"immutable strict FP32 FIFO1024 control differs")
 fifo=json.loads(fifo_path.read_text());metrics.validate_execution_header(fifo,context=context,query=query,control=primary)
 binder.validate_run_reference_binding(fifo,custody);fifo_values=binder.load_samples(fifo_path,fifo)
 comparisons={"corrected_primary":metrics.comparative_assessment(values,primary_values,reference),
   "strict_FP32_FIFO1024":metrics.comparative_assessment(values,fifo_values,reference)}
 require(all(c["sample_integrity"]["passed"] and c["absolute_physical_assessment"]["reference_qualified"] for c in comparisons.values()),"same-case reference/control integrity failed")
 for p in (primary_path,fifo_path,gatepath):initial_inputs[str(p.resolve())]=sha256_file(p)
 sources()
 require(all(sha256_file(p)==digest for p,digest in initial_inputs.items()),"immutable review input changed during computation")
 output=Path(output);output.mkdir(parents=True,exist_ok=False)
 result={"schema":SCHEMA,"attribution":"Built with PriorLabs-TabPFN","manifest_file":str(Path(manifest).resolve()),"manifest_sha256":sha256_file(manifest),
   "authentication":authentication,"comparisons":comparisons,"reference":custody,
   "input_sha256":initial_inputs,"frozen_inputs_and_sources_unchanged":True,
   "controls":{"corrected_primary":{"manifest_sha256":sha256_file(primary_path),"samples_sha256":primary["samples_sha256"]},
    "strict_FP32_FIFO1024":{"manifest_sha256":sha256_file(fifo_path),"samples_sha256":fifo["samples_sha256"]}},
   "timings":m["timings"],"wrapper_total_seconds":m["wrapper_total_seconds"],"memory":m["memory"],"rejection":m["rejection"],"proposal_buffer":m["proposal_buffer"],
   "source":{"reviewer_sha256":sha256_file(__file__),"metric_math_source_sha256":METRICS_SHA,"producer_source_sha256":PRODUCER_SHA},
   "performance_first_authorized":True,"absolute_accuracy_gate_required_for_performance":False,"approximate_experiment":True,
   "qualified_FP32":False,"joint_equivalence_inferred":False,"old_permutation_pvalues_transferred":False,
   "no_model_calls":True,"no_GPU_calls":True,"no_reference_integrations":True,"original_gate_unchanged":True}
 write_json(output/"fp16-experiment-review.json",result)
 write_json(output/"checksums.json",{p.name:sha256_file(p) for p in output.iterdir() if p.is_file()})
 require(sha256_file(gatepath)==binder.BASELINE_GATE_SHA256,"immutable gate changed during review")
 print(json.dumps({"output":str(output),"integrity":True,"physical_status":comparisons["corrected_primary"]["absolute_physical_assessment"]["status"],
   "relative_status":{k:v["numeric_review_status"] for k,v in comparisons.items()}},indent=2))
 return result

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ("root","manifest","approximation","qualification","batch-qualification","sdk-root","output"):p.add_argument("--"+name,required=True)
 run(**vars(p.parse_args()))
if __name__=="__main__":main()
