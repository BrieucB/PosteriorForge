"""Cold FP16 replica worker; only a verified 50,000-vector aggregate is E2E.

Built with PriorLabs-TabPFN. The frozen2280 stage adapter and authenticated23cc
sampling functions perform all model and sampling computation. This new wrapper
adds a fixed-total contract, per-rank quota and honest worker provenance only.
"""
from __future__ import annotations
import argparse,json,os,time
from importlib import import_module
from pathlib import Path
import numpy as np
from . import performance_fp16_experiment as clean
from .artifacts import runtime_identity,sha256_file,write_json

SCHEMA="posteriorforge.inference.fp16_replica_worker.v1"
VARIANT="execution_fp16_fifo_replica_worker_v1"
CONTRACT_SCHEMA="posteriorforge.fp16_replica_contract.v1"
TOTAL_COUNT=50000
MODE=clean.MODE
CLEAN_SHA="2280e1ceef807f05a264b5318be18eb8bee7c52c8bab21884d6682f84cafcba9"
ALL21_MANIFEST_SHA256=clean.ALL21_MANIFEST_SHA256
ALL21_SOURCE_SHA=clean.ALL21_SOURCE_SHA
require=clean.require
frozen_sampling=clean.frozen_sampling
load_script=clean.load_script
validate_all21=clean.validate_all21
execution_metadata=clean.execution_metadata
precision_engine_scope=clean.precision_engine_scope
validate_prediction=clean.validate_prediction

def source_sha256():return sha256_file(__file__)

def validate_contract(contract,*,budget,output,repetition):
 world=contract.get("world_size");rank=contract.get("rank")
 require(type(world) is int and world in (1,2,4),"one/two/four worker world required")
 require(type(rank) is int and 0<=rank<world,"invalid worker rank")
 count=TOTAL_COUNT//world
 expected={"schema":CONTRACT_SCHEMA,"benchmark_total_count":TOTAL_COUNT,"worker_count":count,
  "parent_job_id":str(budget["job_id"]),"budget_reservation_id":budget["reservation_id"],
  "context_seed":20261111,"case":0,"repetition":repetition,
  "expected_output":str(Path(output).resolve()),"worker_source_sha256":source_sha256(),
  "canonical_clean_producer_sha256":CLEAN_SHA,"individual_worker_is_end_to_end":False,
  "query_batch_size":1024,"global_proposal_lookahead":1200,"worker_proposal_lookahead":1200//world,
  "seed_streams":frozen_sampling().worker_stream_seeds(20261111,0,repetition,rank,world)}
 require(clean.source_sha256()==CLEAN_SHA,"canonical2280 producer changed")
 require(all(contract.get(k)==v for k,v in expected.items()),"FP16 worker contract binding differs")
 require(budget["gpus"]==world,"allocation GPU count must equal world")
 for key,want in (("PF_EXECUTION_WORKER_RANK",str(rank)),("PF_EXECUTION_WORKER_WORLD_SIZE",str(world)),
                  ("CUDA_VISIBLE_DEVICES",contract.get("visible_device_token")),("OMP_NUM_THREADS","16")):
  require(os.environ.get(key)==want,"worker environment differs: "+key)
 affinity=contract.get("cpu_affinity")
 require(isinstance(affinity,list) and len(affinity)==16 and len(set(affinity))==16
   and all(type(i) is int and i>=0 for i in affinity),"sixteen distinct allocated worker CPUs required")
 require(sorted(os.sched_getaffinity(0))==sorted(affinity),"actual worker CPU affinity differs")
 orchestrator=clean.repository()/"scripts/run_fp16_workers.py"
 require(orchestrator.is_file() and sha256_file(orchestrator)==contract.get("orchestrator_source_sha256"),
   "reviewed FP16 orchestrator source changed")
 return rank,world,count

def run_worker(*,banks,output,checkpoint,ledger,reference,reference_grid,approximation,
                   qualification,batch_qualification,sdk_root,worker_contract,repetition=0):
 start=time.perf_counter();output=Path(output);require(not output.exists(),"fresh FP16 experimental output required")
 require(ALL21_MANIFEST_SHA256 is not None,"all21 artifact has not been pinned; posterior launch is disabled")
 require(type(repetition) is int and repetition>=0,"nonnegative repetition required")
 from .inference_cdf import guard_baseline_source,reference_binding
 from .inference import CHECKPOINT_SHA256,frozen_regressor_settings,support_mask,_prior
 from .banks import load_context,load_heldout
 from .science.analytical_benchmark_v2 import array_descriptor,build_problem
 from .reference_grid import _require_budgeted_gpu
 contract_path=Path(worker_contract).resolve(strict=True)
 contract=json.loads(contract_path.read_text())
 budget=_require_budgeted_gpu(ledger)
 rank,world,count=validate_contract(contract,budget=budget,output=output,repetition=repetition)
 frozen=frozen_sampling();policy=execution_metadata();all21=load_script("characterize_fp16_all_stages.py",ALL21_SOURCE_SHA)
 baseline_source=guard_baseline_source();initial_source=source_sha256()
 require(sha256_file(checkpoint)==CHECKPOINT_SHA256,"immutable checkpoint mismatch")
 context=load_context(banks,20261111,4096);observed=load_heldout(banks,0);query=np.asarray(observed.obs).reshape(1,33)
 problem=build_problem();bindings={"checkpoint_sha256":CHECKPOINT_SHA256,"scientific_fingerprint":problem.scientific_fingerprint,
  "parameter_order":list(problem.parameter_order),"context_theta":array_descriptor(context.theta),
  "context_obs":array_descriptor(context.obs),"query":array_descriptor(query)}
 screen=validate_all21(approximation,bindings)
 _,portable_receipt=all21.portable_qualification(qualification,batch_qualification,sdk_root,bindings)
 require(sha256_file(qualification)==screen["qualification_sha256"] and sha256_file(batch_qualification)==screen["batch_qualification_sha256"],"screen FP32 ancestry differs")
 refs=reference_binding(banks=banks,reference=reference,reference_grid=reference_grid,case=0)
 require(budget["gpus"]==world,"worker world must equal the reserved GPU allocation")
 output.mkdir(parents=True,exist_ok=False);engine=None;descriptor=None;buffer=None
 status={**runtime_identity(),"schema":SCHEMA,"variant":VARIANT,"status":"running","attribution":"Built with PriorLabs-TabPFN",
   "approximate_experiment":True,"qualified_fp32":False,"qualification_full21":False,"qualification_cache_only_full21":False,
   "approximation_all21_hard_screen_passed":True,"all21_strict_screen_passed":screen["all21_strict_screen_passed"],
   "strict_failed_stages":screen["strict_failed_stages"],"approximation_manifest_sha256":ALL21_MANIFEST_SHA256,
   "approximation_producer_sha256":ALL21_SOURCE_SHA,"approximation_scope":screen["scope"],
   "source_checkout_required":True,"runner_source_sha256":initial_source,"execution_policy":policy,
   "context_seed":20261111,"case":0,"repetition":repetition,"context_size":4096,"query_batch_size":1024,"engine_query_batch_cap":1024,
   "requested_count":count,"benchmark_total_count":TOTAL_COUNT,"rank":rank,"world_size":world,"end_to_end":False,"scientific_run":False,
   "run_purpose":"performance_worker","worker_contract_sha256":sha256_file(contract_path),
   "canonical_clean_producer_sha256":CLEAN_SHA,"sampling_wrapper_only_changes":["contract","quota","rank","world","provenance","per_rank_CPU_affinity_and_OMP_threads"],
   "proposal_lookahead":1200,"proposal_buffer_policy":frozen.proposal_buffer_policy(1200),"proposal_buffer":None,
   "microbatch":count,"max_iter_rejection":None,"log_density":"not_used","context_filter":"no_filtering",
   "observation_preprocessing":"raw; TabPFN frozen preprocessing","performance_first_authorized":True,
   "absolute_accuracy_gate_required_for_performance":False,"known_physical_control_status":"model_failure",
   "candidate_physical_assessment":"pending independent original physical gate; no accuracy pass claim",
   "relative_distribution_review_thresholds":{"CDF_change":.01,"mean_change_refSD":.05,"SD_relative_change":.05},
   "partial_vector_filtering":False,"clipping_or_repair_performed":False,"tail_family_changed":False,"execution_optimized":True,
   "original_gate_unchanged":True,"baseline_execution_source":baseline_source,"budget_binding":budget,
   "reference_binding":refs,"portable_FP32_ancestry_verification":portable_receipt,
   "sampling_rng_policy":"independent_cpu_sampling_streams_model_rng_isolated",
   "timing_boundaries":{"posterior_request":"first proposal through final support/FIFO validation and CUDA synchronization",
     "wrapper":"function entry through NPZ/hash completion; final manifest write/interpreter startup reported by outer job timer"},
   "completed_stage_indices":[],"stage_events":[],"proposal_wave_sizes":[],**bindings}
 write_json(output/"manifest.json",status)
 try:
  import torch
  from . import execution_mixture as mixture
  from .compat_cdf import finite_cdf_compatibility
  from .upstream import load_baseline
  require(torch.cuda.is_available() and torch.cuda.device_count()==1,"one visible CUDA GPU required")
  require(torch.get_default_dtype()==torch.float32 and torch.get_float32_matmul_precision()=="highest"
      and not torch.backends.cuda.matmul.allow_tf32 and not torch.is_autocast_enabled("cuda"),"original strict initial flags required")
  require(np.__version__=="1.26.4","recorded NumPy1.26.4 inference environment required")
  load_baseline();accept_reject=import_module("npe_pfn.accept_reject_sampler").accept_reject_sample
  seeds=frozen.worker_stream_seeds(20261111,0,repetition,rank,world)
  torch.manual_seed(seeds["component_uniform"]);torch.cuda.manual_seed_all(seeds["component_uniform"])
  generators={name:torch.Generator(device="cpu").manual_seed(seed) for name,seed in seeds.items()}
  torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS","16")))
  streams={name:{"role":frozen.STREAM_ROLES[name],"seed":seed,"dtype":frozen.STREAM_DTYPES[name],
    "device":"cpu","generator":"torch.Generator"} for name,seed in seeds.items()}
  status.update(seed_streams=seeds,seed=seeds["component_uniform"],gpu=torch.cuda.get_device_name(),torch_cuda=torch.version.cuda,
      regressor_settings={**frozen_regressor_settings(checkpoint),"fit_mode":"fit_with_cache","inference_precision":"autocast"},
      sampling_policy={"schema":"posteriorforge.fp16_sampling_policy.v1","frozen_component_policy_provenance":mixture.execution_metadata(),
       "forward_precision_policy":MODE,"postprocessing_math_dtype":"torch.float32","head_policy":"raw_mass_rejection_v1",
       "uniform_assignment":"preassigned_per_stage_by_physical_row","head_retry_preserves_row_uniform":True,"streams":streams,
       "model_rng_isolation":"CPU and selected CUDA fork_rng in frozen executor","mass_envelope":1.00001,
       "stage0_dedup":{"enabled":True,"scope":"constructed_shared_raw_probability_mixture",
         "policy":all21.STAGE0_POLICY,"component_distribution_dedup":False,"density_scope":"one proposal wave; no cross-request/case reuse"}})
  theta=context.theta.astype(np.float32);obs=context.obs.astype(np.float32);prior=_prior(torch)
  counters={"stage0_shared_predictions":0,"component_predictions":0,"head_candidates":0,"head_rejections":0}
  actual_checks={}
  with finite_cdf_compatibility() as descriptor, precision_engine_scope(checkpoint,cdf_descriptor=descriptor) as engine:
   callback=lambda prediction,stage,member:validate_prediction(prediction,stage,member,report=actual_checks)
   # These two functions are the EXACT authenticated23cc code objects.
   stage_draw=frozen.component_stage_sampler(engine,torch,generators,counters,prediction_check=callback)
   def progress(event):
    status["stage_events"].append({"proposal_round":len(status["proposal_wave_sizes"]),**event})
    status["elapsed_seconds"]=time.perf_counter()-start;write_json(output/"manifest.json",status)
   def proposal(size):
    status["proposal_wave_sizes"].append(int(size))
    values=frozen.stage_major_wave(engine,theta,obs,query,int(size),1024,lambda _:None,stage_draw=stage_draw,progress=progress)
    return torch.from_numpy(values),None
   scope=frozen.proposal_buffer_scope(bindings,context_seed=20261111,case=0,repetition=repetition,rank=rank,world_size=world,seeds=seeds)
   fifo=frozen.guarded_proposal_buffer()
   buffer=fifo.ProposalBuffer(proposal,scope=scope,concatenate=lambda parts:torch.cat(parts,dim=0),world_size=world,rank=rank)
   torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();request_start=time.perf_counter()
   with torch.no_grad():
    samples,_,_=accept_reject(proposal=buffer,accept_reject_fn=prior.support.check,num_samples=count,
      max_sampling_batch_size=count,max_iter_rejection=None,show_progress_bars=False,proposal_sampling_kwargs={})
   record=buffer.finalize(original_checked_count=prior.support.proposed,accepted_before_trim=prior.support.accepted,
       returned_count=len(samples),original_support_calls=prior.support.calls)
   record.update(enabled=True,source_unchanged=True,policy_sha256=status["proposal_buffer_policy"]["policy_sha256"])
   frozen.validate_proposal_buffer_record(record,world_size=world,rank=rank,scope=scope);status["proposal_buffer"]=record
   torch.cuda.synchronize();finished=time.perf_counter()
   values=samples.detach().cpu().numpy().astype(np.float64)
   require(values.shape==(count,21) and support_mask(values).all(),"exact contracted quota of complete finite prior-supported vectors required")
   generated=len(status["proposal_wave_sizes"]);wanted_heads=8*counters["stage0_shared_predictions"]+counters["component_predictions"]
   require(engine.prepare_count==21*generated and len(engine.stages)==21*generated,"actual generated-wave stage coverage differs")
   require(engine.actual_heads==actual_checks["actual_head_contexts"]==wanted_heads==descriptor["patched_call_count"],"actual8-head/shared/CDF counters differ")
   require(engine.shared_densities==counters["stage0_shared_predictions"]==generated,"one shared stage0 density per generated wave required")
   require(record["actual_generated_wave_sizes"]==status["proposal_wave_sizes"],"FIFO/provider wave sizes differ")
   status.update(execution_counters=engine.metadata(),sampling_counters=dict(counters),actual_prediction_checks=actual_checks,
       expected_patched_call_count=wanted_heads,expected_stage_prepare_count=21*generated,generated_proposal_wave_count=generated)
  require(descriptor["original_alias_restored"] and descriptor["sdk_sources_unchanged"],"CDF adapter restoration failed")
  require(source_sha256()==initial_source and clean.source_sha256()==CLEAN_SHA and sha256_file(contract_path)==status["worker_contract_sha256"] and frozen_sampling() is frozen and guard_baseline_source()==baseline_source,"source changed during experiment")
  require(sha256_file(approximation)==ALL21_MANIFEST_SHA256,"all21 proof changed during experiment")
  require(sha256_file(qualification)==screen["qualification_sha256"] and sha256_file(batch_qualification)==screen["batch_qualification_sha256"],"FP32 ancestry changed")
  with (output/"samples.npz").open("xb") as stream:np.savez(stream,samples=values)
  status.update(status="complete",samples=array_descriptor(values),samples_file="samples.npz",samples_sha256=sha256_file(output/"samples.npz"),
    completed_stage_indices=list(range(21)),compatibility_adapter=dict(descriptor),sources_unchanged=True,
    rejection={"support_calls":prior.support.calls,"proposed":prior.support.proposed,"accepted_before_final_trim":prior.support.accepted,"returned":len(values)},
    timings={"posterior_request_seconds":finished-request_start,"total_seconds":time.perf_counter()-start,
      "accepted_samples_per_second":count/(finished-request_start),"includes_stage_preparation":True,"includes_CPU_sampling":True},
    memory={"peak_allocated_bytes":torch.cuda.max_memory_allocated(),"peak_reserved_bytes":torch.cuda.max_memory_reserved()})
 except BaseException as error:
  status.update(status="failed",error_type=type(error).__name__,error=str(error))
  if descriptor is not None:status["compatibility_adapter"]=dict(descriptor)
  if buffer is not None:status["proposal_buffer"]=buffer.metadata()
  raise
 finally:
  if engine is not None:engine.close()
  status["wrapper_total_seconds"]=time.perf_counter()-start;write_json(output/"manifest.json",status)
 return status

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 for name in ("banks","output","checkpoint","ledger","reference","reference-grid","approximation","qualification","batch-qualification","sdk-root","worker-contract"):
  parser.add_argument("--"+name,required=True)
 parser.add_argument("--repetition",type=int,default=0)
 result=run_worker(**vars(parser.parse_args()))
 print(json.dumps({k:result.get(k) for k in ("schema","variant","status","requested_count","rank","world_size","timings")},indent=2))
if __name__=="__main__":main()
