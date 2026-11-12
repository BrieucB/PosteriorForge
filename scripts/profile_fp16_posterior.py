#!/usr/bin/env python3
"""Instrumented full50000 FP16 posterior; frozen algorithms remain unchanged.

Built with PriorLabs-TabPFN. A NEW orchestration calls pinned2280 APIs and exact
23cc sampling functions through local pass-through timing wrappers. This is not
an uninstrumented2280 repetition. Run from the reviewed source checkout with
PYTHONPATH=<checkout>/src. Raw trace bytes are never rewritten.
"""
from __future__ import annotations
import argparse
from bisect import bisect_right
from collections import defaultdict
from contextlib import contextmanager,nullcontext
import gzip
import hashlib
import importlib.util
import inspect
import json
import marshal
import os
from pathlib import Path
import pickle
from threading import get_ident
from types import CodeType
import time
import numpy as np

SCHEMA="posteriorforge.fp16_full50000_runtime_profile.v1"
VARIANT="execution_fp16_fifo_profile_v1"
COUNT=50000
BOUNDARY="pfdiagnostic/profile/request"
CANDIDATE_SHA="2280e1ceef807f05a264b5318be18eb8bee7c52c8bab21884d6682f84cafcba9"
PARSER_SHA="4707b75c3866d31688031da704eeeffe5ab8fed9a2dd85195626028a0c08b021"
READER_SHA="1852980b507649ecf960d0f84b58d15e24073639c7482cb7a433e4e17b51139b"
CLEAN_MANIFEST_SHA="3546de7f867d3aeaefbbf4e1dab03d90fe867e8355506629319a6e9eed492133"
CLEAN_SAMPLES_SHA="61d0a058233ab2ff6e12d1ad29b17ef9d80c3fe10cf1f474be4d33bd716f1f11"
PRIORITY=("query_head","query_shared_stage0","prepare","inspect_guard","close","prediction_check","support","sampling_stage")
PINS={"src/posteriorforge/performance_fp16_experiment.py":CANDIDATE_SHA,
 "src/posteriorforge/execution_precision_probe.py":"051f477ff4b9ab8bbcfbaf3c1afa4dcf6b111d2f2fbcf9efb9c94a00d2b2ad3d",
 "src/posteriorforge/performance_runner.py":"23ccad8b5bbcdd7a48340b349d515ca870763359cfeb71333e02fc2b6a8aca3b",
 "src/posteriorforge/execution_mixture.py":"e71f43b73ae8259c20b5d7b52feccc7c851e9a2fbb86d869bc0a5f7f3af792a1",
 "src/posteriorforge/execution_proposal_buffer.py":"8fb2ba1fb63cd917378ff9bc8b6f70be7410862c509cd22cb2fb6f454d3d35fa",
 "scripts/characterize_fp16_all_stages.py":"20e64a64357de7862c20b2fa846c6d128c181ca564c261800b626e6c4bd32208",
 "scripts/diagnose_prefill_allocator.py":PARSER_SHA,"scripts/diagnose_runtime.py":READER_SHA}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_script(name,expected):
 path=Path(__file__).with_name(name)
 if sha(path)!=expected:raise RuntimeError("pinned helper changed: "+name)
 spec=importlib.util.spec_from_file_location("_fp16_profile_"+path.stem,path)
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def source_pins():
 root=Path(__file__).resolve().parents[1]
 for name,expected in PINS.items():
  if sha(root/name)!=expected:raise RuntimeError("frozen profile dependency changed: "+name)
 return dict(PINS)


def code_sha(function):return hashlib.sha256(marshal.dumps(function.__code__)).hexdigest()


class Seals:
 """Observe exact callable identities; no original alias is replaced."""
 def __init__(self,candidate,frozen):
  self.candidate=candidate;self.frozen=frozen;self.rows=[]
  for owner,names in ((candidate,("frozen_sampling","validate_all21","execution_metadata","precision_engine_scope","validate_prediction","FP16StageAdapter")),
      (frozen,("stage_major_wave","component_stage_sampler","worker_stream_seeds","guarded_proposal_buffer","validate_proposal_buffer_record")),
      (candidate.FP16StageAdapter,("prepare","predict","predict_component","inspect_cache","close","metadata"))):
   for name in names:
    value=getattr(owner,name);self.rows.append((owner,name,value,getattr(value,"__code__",None)))
 def verify(self):
  source_pins()
  if self.candidate.frozen_sampling() is not self.frozen:raise RuntimeError("original private sampling module identity changed")
  for owner,name,value,code in self.rows:
   if getattr(owner,name) is not value or getattr(value,"__code__",None) is not code:raise RuntimeError("original callable alias/code changed: "+name)
 def descriptor(self):
  return {"original_alias_replacements":[],"original_aliases_unchanged":True,
   "functions":[{"name":getattr(value,"__qualname__",name),"code_sha256":None if code is None else code_sha(value)} for _,name,value,code in self.rows]}


def union_seconds(intervals):
 intervals=sorted((left,right) for left,right in intervals if right>left)
 if not intervals:return 0.
 left,right=intervals[0];total=0.
 for a,b in intervals[1:]:
  if a>right:total+=right-left;left,right=a,b
  else:right=max(right,b)
 return total+right-left


class HostRanges:
 def __init__(self,record_function=None):
  self.annotate=record_function or (lambda name:nullcontext());self.owner=get_ident();self.wave=0;self.events=[];self.stack=[]
 @contextmanager
 def span(self,operation,stage=-1,member=-1):
  if get_ident()!=self.owner:raise RuntimeError("profile ranges require original owner thread")
  row={"id":len(self.events),"operation":operation,"proposal_wave":self.wave,"stage_index":stage,"member_index":member,
       "parent_id":self.stack[-1] if self.stack else None,"start_seconds":time.perf_counter()}
  self.events.append(row);self.stack.append(row["id"])
  try:
   with self.annotate("pfdiagnostic/profile/"+operation+"/w"+str(self.wave)+"/s"+str(stage)+"/m"+str(member)):yield row
  except BaseException as error:row["error_type"]=type(error).__name__;raise
  finally:
   row["stop_seconds"]=time.perf_counter();row["host_wall_seconds"]=row["stop_seconds"]-row["start_seconds"];self.stack.pop()
 def exclusive_sampling(self):
  result=[]
  for event in self.events:
   if event["operation"]!="sampling_stage":continue
   children=[(max(event["start_seconds"],row["start_seconds"]),min(event["stop_seconds"],row["stop_seconds"]))
    for row in self.events if row["operation"] in ("query_head","query_shared_stage0","prediction_check") and row["start_seconds"]>=event["start_seconds"] and row["stop_seconds"]<=event["stop_seconds"]]
   removed=union_seconds(children);exclusive=max(0.,event["host_wall_seconds"]-removed)
   result.append({"proposal_wave":event["proposal_wave"],"stage_index":event["stage_index"],"inclusive_host_seconds":event["host_wall_seconds"],
                  "nested_query_and_callback_union_seconds":removed,"exclusive_host_seconds":exclusive})
  return {"stages":result,"exclusive_host_seconds":sum(row["exclusive_host_seconds"] for row in result),
   "scope":"owner-thread sampler bookkeeping/RNG/head selection/inverse arithmetic/query assembly, excluding union of original query and callback spans; host wall, not CPU compute",
   "overlap_caution":"Never add exclusive host intervals to GPU kernel/copy/synchronization unions as disjoint costs."}


class RangeAdapter:
 """One original call with exact arguments/return object; no SDK/model mutation."""
 def __init__(self,original,ranges):self.original=original;self.ranges=ranges
 def __getattr__(self,name):return getattr(self.original,name)
 @property
 def regressor(self):return self.original.regressor
 def prepare(self,*args,**kwargs):
  stage=kwargs.get("stage",args[2] if len(args)>2 else -1)
  with self.ranges.span("prepare",stage):return self.original.prepare(*args,**kwargs)
 def predict_component(self,member,*args,**kwargs):
  with self.ranges.span("query_head",self.original.stage_index,member):return self.original.predict_component(member,*args,**kwargs)
 def predict(self,*args,**kwargs):
  with self.ranges.span("query_shared_stage0",self.original.stage_index):return self.original.predict(*args,**kwargs)
 def inspect_cache(self,*args,**kwargs):
  with self.ranges.span("inspect_guard",self.original.stage_index):return self.original.inspect_cache(*args,**kwargs)
 def close(self,*args,**kwargs):
  with self.ranges.span("close",self.original.stage_index):return self.original.close(*args,**kwargs)


def measured_stage_sampler(factory,*args,ranges,**kwargs):
 original=factory(*args,**kwargs);code=original.__code__
 def wrapped(*call_args,**call_kwargs):
  if original.__code__ is not code:raise RuntimeError("actual original stage sampler code changed")
  stage=call_kwargs.get("stage",call_args[1] if len(call_args)>1 else -1)
  with ranges.span("sampling_stage",stage):return original(*call_args,**call_kwargs)
 return wrapped,{"original_factory_code_sha256":code_sha(factory),"original_returned_sampler_code_sha256":code_sha(original),
  "factory_original_calls":1,"private_function_aliases_replaced":False,"call_arguments_returns_order_preserved":True}


def trace_attribution(path,reader):
 """Attribute GPU events via original CUDA launch correlations, not host waits."""
 annotations=[];runtime=[];sdpa=defaultdict(int);boundary=[]
 for event in reader.iter_trace_events(path):
  if event.get("ph")!="X" or not isinstance(event.get("dur"),(int,float)):continue
  name=event.get("name","");cat=event.get("cat","")
  if cat=="user_annotation" and name==BOUNDARY:boundary.append((event["ts"],event["ts"]+event["dur"]))
  if cat=="user_annotation" and name.startswith("pfdiagnostic/profile/"):
   operation=name.split("/")[2]
   if operation in PRIORITY:annotations.append((event.get("pid"),event.get("tid"),operation,event["ts"],event["ts"]+event["dur"]))
  if name.startswith("aten::") and "scaled_dot_product" in name:sdpa[name]+=1
  if cat in ("cuda_runtime","cuda_driver"):runtime.append((event.get("pid"),event.get("tid"),event["ts"],event.get("args",{}).get("correlation")))
 if len(boundary)!=1:raise ValueError("one owner host request boundary required for attribution")
 start,stop=boundary[0];ranges=defaultdict(list)
 for pid,tid,operation,left,right in annotations:ranges[(pid,tid,operation)].append((left,right))
 for key in ranges:ranges[key].sort()
 starts={key:[left for left,_ in value] for key,value in ranges.items()};correlations={};conflicts=set()
 for pid,tid,ts,correlation in runtime:
  if not isinstance(correlation,int):continue
  kind="unmatched"
  for operation in PRIORITY:
   key=(pid,tid,operation);index=bisect_right(starts.get(key,[]),ts)-1
   if index>=0 and ranges[key][index][0]<=ts<ranges[key][index][1]:kind=operation;break
  if correlation in correlations and correlations[correlation]!=kind:conflicts.add(correlation)
  correlations[correlation]=kind
 groups=defaultdict(lambda:{"kernel_count":0,"kernel_sum_seconds":0.,"kernel_intervals":[],"copy_count":0,"copy_sum_seconds":0.,"copy_intervals":[],"copy_bytes":0})
 copies=defaultdict(lambda:{"count":0,"bytes":0,"sum_seconds":0.});attention=defaultdict(lambda:{"count":0,"sum_seconds":0.})
 for event in reader.iter_trace_events(path):
  if event.get("ph")!="X" or not isinstance(event.get("dur"),(int,float)):continue
  cat=event.get("cat","").lower();kernel="kernel" in cat;copy="memcpy" in cat
  if not kernel and not copy:continue
  left=max(start,event["ts"]);right=min(stop,event["ts"]+event["dur"])
  if right<=left:continue
  correlation=event.get("args",{}).get("correlation");kind=correlations.get(correlation,"unmatched") if correlation not in conflicts else "unmatched"
  group=groups[kind];seconds=(right-left)/1e6
  if kernel:
   group["kernel_count"]+=1;group["kernel_sum_seconds"]+=seconds;group["kernel_intervals"].append((left/1e6,right/1e6))
   if any(word in event.get("name","").lower() for word in ("attention","fmha","flash")):
    attention[event["name"]]["count"]+=1;attention[event["name"]]["sum_seconds"]+=seconds
  if copy:
   group["copy_count"]+=1;group["copy_sum_seconds"]+=seconds;group["copy_intervals"].append((left/1e6,right/1e6))
   amount=event.get("args",{}).get("bytes",0);amount=amount if isinstance(amount,(int,float)) else 0;group["copy_bytes"]+=amount
   row=copies[event.get("name","")];row["count"]+=1;row["bytes"]+=amount;row["sum_seconds"]+=seconds
 for group in groups.values():
  group["kernel_interval_union_seconds"]=union_seconds(group.pop("kernel_intervals"));group["copy_interval_union_seconds"]=union_seconds(group.pop("copy_intervals"))
 return {"method":"CUDA runtime launch correlation to most-specific observed owner-thread range; GPU durations clipped to host request boundary",
  "groups":dict(groups),"correlation_conflicts":len(conflicts),"unmatched_work_is_retained":True,"observed_SDPA_operator_counts":dict(sdpa),
  "observed_attention_kernel_names":[{"name":name,**row} for name,row in sorted(attention.items(),key=lambda pair:pair[1]["sum_seconds"],reverse=True)],
  "copy_names_and_bytes":dict(copies),"flash_attention_inferred_from_flags":False,
  "caution":"Kernel sums can overlap. Per-kind unions can overlap one another; host synchronization waits are not independent GPU compute."}


def payload_comparison(candidate,control):
 same=candidate.shape==control.shape and candidate.dtype==control.dtype and candidate.tobytes()==control.tobytes()
 report={"passed":same,"exact_array_bytes_equal":same,"scope":"coupled complete50000-array diagnostic against clean5406772; NPZ encoding bytes are not the scientific payload"}
 if candidate.shape==control.shape:
  difference=np.abs(candidate-control);report.update(changed_rows=int(np.count_nonzero(np.any(candidate!=control,axis=1))),
    changed_entries=int(np.count_nonzero(candidate!=control)),max_absolute=float(difference.max()))
 return report


def require_original_sampling_code(accept_reject,prior_factory):
 from posteriorforge.inference_cdf import UPSTREAM_SOURCE_SHA256,BASELINE_INFERENCE_SHA256
 for function,name,expected in ((inspect.unwrap(accept_reject),"accept_reject_sample",UPSTREAM_SOURCE_SHA256["accept_reject_sampler.py"]),
                                (prior_factory,"_prior",BASELINE_INFERENCE_SHA256)):
  path=Path(inspect.getfile(function))
  if sha(path)!=expected:raise RuntimeError("original algorithm source differs: "+name)
  compiled=compile(path.read_text(),str(path),"exec",dont_inherit=True)
  code=next(value for value in compiled.co_consts if isinstance(value,CodeType) and value.co_name==name)
  if function.__code__!=code:raise RuntimeError("active original algorithm differs from pinned body: "+name)


def rng_snapshot(torch):
 import random
 return {"torch_cpu":hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
  "torch_cuda":hashlib.sha256(b"".join(value.numpy().tobytes() for value in torch.cuda.get_rng_state_all())).hexdigest(),
  "numpy":hashlib.sha256(pickle.dumps(np.random.get_state(),protocol=5)).hexdigest(),"python":hashlib.sha256(pickle.dumps(random.getstate(),protocol=5)).hexdigest()}


def run(*,banks,output,checkpoint,ledger,reference,reference_grid,approximation,qualification,batch_qualification,sdk_root,clean_run,max_seconds=780):
 from posteriorforge.artifacts import runtime_identity,write_json
 from posteriorforge.reference_grid import _require_budgeted_gpu
 if max_seconds!=780:raise ValueError("fixed780-second soft deadline required")
 start=time.perf_counter();deadline=start+780;budget=_require_budgeted_gpu(ledger)
 if budget["gpus"]!=1:raise RuntimeError("one budgeted GPU required")
 output=Path(output);output.mkdir(parents=True,exist_ok=False);source=sha(__file__)
 report={**runtime_identity(),"schema":SCHEMA,"variant":VARIANT,"status":"running","instrumented":True,"clean_repetition":False,
  "original_run_experiment_called":False,"orchestration_scope":"new profile driver calling pinned2280 APIs and exact23cc algorithms through local pass-through wrappers",
  "helper_source_sha256":source,"budget_binding":budget,"requested_count":COUNT,"returned_count":None,"case":0,"context_seed":20261111,"repetition":0,
  "query_batch_size":1024,"proposal_lookahead":1200,"approximate_experiment":True,"qualified_FP32":False,"original_physical_failure_remains":True,
  "trace_configuration":{"record_shapes":False,"profile_memory":False,"with_stack":False},"startup_costs":{},
  "stage_events":[],"proposal_wave_sizes":[],"attribution":"Built with PriorLabs-TabPFN"}
 write_json(output/"manifest.json",report);descriptor=None;ranges=None;original_engine=None;trace=None
 try:
  t0=time.perf_counter();report["source_pins"]=source_pins();parser=load_script("diagnose_prefill_allocator.py",PARSER_SHA);reader=load_script("diagnose_runtime.py",READER_SHA)
  from posteriorforge import performance_fp16_experiment as candidate,execution_mixture as mixture
  from posteriorforge.inference_cdf import guard_baseline_source,reference_binding
  from posteriorforge.inference import CHECKPOINT_SHA256,frozen_regressor_settings,support_mask,_prior
  from posteriorforge.banks import load_context,load_heldout
  from posteriorforge.science.analytical_benchmark_v2 import build_problem,array_descriptor
  from posteriorforge.compat_cdf import finite_cdf_compatibility
  from posteriorforge.upstream import load_baseline
  frozen=candidate.frozen_sampling();seals=Seals(candidate,frozen);seals.verify();policy=candidate.execution_metadata()
  all21=candidate.load_script("characterize_fp16_all_stages.py",candidate.ALL21_SOURCE_SHA)
  baseline=guard_baseline_source();report["startup_costs"]["imports_and_source_auth_seconds"]=time.perf_counter()-t0;t0=time.perf_counter()
  if sha(checkpoint)!=CHECKPOINT_SHA256:raise ValueError("immutable checkpoint mismatch")
  context=load_context(banks,20261111,4096);observed=load_heldout(banks,0);query=observed.obs.reshape(1,33);problem=build_problem()
  bindings={"checkpoint_sha256":CHECKPOINT_SHA256,"scientific_fingerprint":problem.scientific_fingerprint,"parameter_order":list(problem.parameter_order),
   "context_theta":array_descriptor(context.theta),"context_obs":array_descriptor(context.obs),"query":array_descriptor(query)}
  screen=candidate.validate_all21(approximation,bindings);_,receipt=all21.portable_qualification(qualification,batch_qualification,sdk_root,bindings)
  if sha(qualification)!=screen["qualification_sha256"] or sha(batch_qualification)!=screen["batch_qualification_sha256"]:raise ValueError("FP32 ancestry differs")
  refs=reference_binding(banks=banks,reference=reference,reference_grid=reference_grid,case=0)
  clean_run=Path(clean_run)
  if sha(clean_run/"manifest.json")!=CLEAN_MANIFEST_SHA or sha(clean_run/"samples.npz")!=CLEAN_SAMPLES_SHA:raise ValueError("exact clean5406772 comparator required")
  clean=json.loads((clean_run/"manifest.json").read_text())
  if clean.get("status")!="complete" or clean.get("runner_source_sha256")!=CANDIDATE_SHA or clean.get("execution_policy")!=policy or any(clean.get(k)!=v for k,v in bindings.items()):raise ValueError("clean comparator source/policy/bank binding differs")
  inputs={str(Path(path).resolve()):sha(path) for path in (checkpoint,approximation,qualification,batch_qualification,clean_run/"manifest.json",clean_run/"samples.npz")}
  report.update(**bindings,execution_policy=policy,approximation_manifest_sha256=sha(approximation),strict_failed_stages=screen["strict_failed_stages"],
   all21_strict_screen_passed=screen["all21_strict_screen_passed"],portable_FP32_ancestry_receipt=receipt,reference_binding=refs,
   clean_comparator={"job_id":"5406772","manifest_sha256":CLEAN_MANIFEST_SHA,"samples_npz_sha256":CLEAN_SAMPLES_SHA,"request_seconds":clean["timings"]["posterior_request_seconds"]},
   startup_auth_inputs_sha256=inputs)
  report["startup_costs"]["checkpoint_bank_screen_reference_and_clean_auth_seconds"]=time.perf_counter()-t0;t0=time.perf_counter()
  import torch
  if not torch.cuda.is_available() or torch.cuda.device_count()!=1 or torch.get_default_dtype()!=torch.float32 or torch.get_float32_matmul_precision()!="highest" or torch.backends.cuda.matmul.allow_tf32 or torch.is_autocast_enabled("cuda"):raise RuntimeError("original strict initial GPU/dtype flags required")
  if np.__version__!="1.26.4":raise RuntimeError("recorded inference NumPy version required")
  load_baseline()
  from importlib import import_module
  accept_reject=import_module("npe_pfn.accept_reject_sampler").accept_reject_sample
  require_original_sampling_code(accept_reject,_prior)
  sampler_identity=(accept_reject,accept_reject.__code__,inspect.unwrap(accept_reject).__code__)
  seeds=frozen.worker_stream_seeds(20261111,0,0,0,1);torch.manual_seed(seeds["component_uniform"]);torch.cuda.manual_seed_all(seeds["component_uniform"])
  generators={name:torch.Generator(device="cpu").manual_seed(seed) for name,seed in seeds.items()};torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS","16")))
  prior=_prior(torch);support_identity=(type(prior.support).check,type(prior.support).check.__code__)
  theta=context.theta.astype(np.float32);obs=context.obs.astype(np.float32)
  counters={"stage0_shared_predictions":0,"component_predictions":0,"head_candidates":0,"head_rejections":0};actual_checks={}
  ranges=HostRanges(torch.profiler.record_function);global_before=rng_snapshot(torch)
  report.update(seed_streams=seeds,gpu=torch.cuda.get_device_name(),torch_cuda=torch.version.cuda,
   regressor_settings={**frozen_regressor_settings(checkpoint),"fit_mode":"fit_with_cache","inference_precision":"autocast"},
   stage0_host_scope="one aggregate query span containing8 actual original head calls; individual stage0 head host spans not claimed")
  report["startup_costs"]["torch_CUDA_seed_and_thread_setup_seconds"]=time.perf_counter()-t0
  with finite_cdf_compatibility() as descriptor,candidate.precision_engine_scope(checkpoint,cdf_descriptor=descriptor) as original_engine:
   engine=RangeAdapter(original_engine,ranges)
   def callback(prediction,stage,member):
    with ranges.span("prediction_check",stage,-1 if member is None else member):return candidate.validate_prediction(prediction,stage,member,report=actual_checks)
   stage_draw,factory_proof=measured_stage_sampler(frozen.component_stage_sampler,engine,torch,generators,counters,prediction_check=callback,ranges=ranges)
   report["sampler_factory_instrumentation"]=factory_proof
   def progress(event):
    report["stage_events"].append({"proposal_round":len(report["proposal_wave_sizes"]),**event});report["elapsed_seconds"]=time.perf_counter()-start;write_json(output/"manifest.json",report)
    if time.perf_counter()>=deadline:raise TimeoutError("soft780-second deadline; no reduced-count posterior is accepted")
   def proposal(size):
    report["proposal_wave_sizes"].append(int(size));ranges.wave=len(report["proposal_wave_sizes"])
    values=frozen.stage_major_wave(engine,theta,obs,query,int(size),1024,lambda _:None,stage_draw=stage_draw,progress=progress)
    return torch.from_numpy(values),None
   scope=frozen.proposal_buffer_scope(bindings,context_seed=20261111,case=0,repetition=0,rank=0,world_size=1,seeds=seeds)
   buffer=frozen.guarded_proposal_buffer().ProposalBuffer(proposal,scope=scope,concatenate=lambda parts:torch.cat(parts,dim=0),world_size=1,rank=0)
   def support(values):
    with ranges.span("support"):return prior.support.check(values)
   holder={}
   def profiled_body():
    try:
     with torch.no_grad():
      samples,_,_=accept_reject(proposal=buffer,accept_reject_fn=support,num_samples=COUNT,max_sampling_batch_size=COUNT,max_iter_rejection=None,show_progress_bars=False,proposal_sampling_kwargs={})
     record=buffer.finalize(original_checked_count=prior.support.proposed,accepted_before_trim=prior.support.accepted,returned_count=len(samples),original_support_calls=prior.support.calls)
     record.update(enabled=True,source_unchanged=True,policy_sha256=frozen.proposal_buffer_policy(1200)["policy_sha256"])
     frozen.validate_proposal_buffer_record(record,world_size=1,rank=0,scope=scope);holder["buffer"]=record
     return samples
    except BaseException as error:
     holder["error"]=error
     try:engine.close()
     except BaseException as cleanup:holder["cleanup_error"]={"type":type(cleanup).__name__,"error":str(cleanup)}
     return None
   if time.perf_counter()+240>=deadline:raise TimeoutError("insufficient predeclared request/export allowance")
   report["startup_costs"]["total_before_profile_seconds"]=time.perf_counter()-start
   torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t0=time.perf_counter()
   samples,trace=parser.profile_call(profiled_body,output/"full50000-fp16.trace.json",torch)
   report["profile_call_including_export_summary_gzip_seconds"]=time.perf_counter()-t0
   report["profiled_posterior_request_seconds"]=trace["summary"]["profiled_request_trace_span_seconds"]
   report["trace"]=trace;trace.update(sampling_calls=1,sampling_scope="one actual50000-return original rejection request; instrumental timing",posterior_vectors_returned=None)
   if "cleanup_error" in holder:report["profile_cleanup_error"]=holder["cleanup_error"]
   if "error" in holder:raise holder["error"]
   values=samples.detach().cpu().numpy().astype(np.float64)
   if values.shape!=(COUNT,21) or not support_mask(values).all():raise RuntimeError("exact50000 complete supported vectors required")
   wanted=8*counters["stage0_shared_predictions"]+counters["component_predictions"];generated=len(report["proposal_wave_sizes"])
   if original_engine.prepare_count!=21*generated or len(original_engine.stages)!=21*generated or original_engine.actual_heads!=actual_checks["actual_head_contexts"] or original_engine.actual_heads!=wanted or descriptor["patched_call_count"]!=wanted or original_engine.shared_densities!=generated or counters["stage0_shared_predictions"]!=generated or holder["buffer"]["actual_generated_wave_sizes"]!=report["proposal_wave_sizes"]:raise RuntimeError("actual full-request head/CDF/stage/FIFO counters differ")
   if original_engine.actual_heads!=wanted or actual_checks["sampler_callback_calls"]!=counters["stage0_shared_predictions"]+counters["component_predictions"]:raise RuntimeError("physical head versus sampler callback coverage differs")
   report.update(returned_count=COUNT,proposal_buffer=holder["buffer"],sampling_counters=dict(counters),actual_prediction_checks=actual_checks,
    execution_counters=original_engine.metadata(),expected_patched_call_count=wanted,global_rng_preserved=global_before==rng_snapshot(torch),
    rejection={"support_calls":prior.support.calls,"proposed":prior.support.proposed,"accepted_before_final_trim":prior.support.accepted,"returned":len(values)},
    memory={"peak_allocated_bytes":torch.cuda.max_memory_allocated(),"peak_reserved_bytes":torch.cuda.max_memory_reserved()})
   trace["posterior_vectors_returned"]=COUNT
  report["compatibility_adapter"]=dict(descriptor);report["owner_thread_ranges"]=ranges.events;report["exclusive_sampling_host"]=ranges.exclusive_sampling()
  report["gpu_launch_attribution"]=trace_attribution(output/"full50000-fp16.trace.json",reader)
  with (output/"samples.npz").open("xb") as stream:np.savez(stream,samples=values)
  with np.load(clean_run/"samples.npz",allow_pickle=False) as payload:control=payload["samples"].copy()
  report["clean_array_comparison"]=payload_comparison(values,control);report["samples"]=array_descriptor(values);report["samples_npz_sha256"]=sha(output/"samples.npz")
  seals.verify()
  if import_module("npe_pfn.accept_reject_sampler").accept_reject_sample is not sampler_identity[0] or accept_reject.__code__ is not sampler_identity[1] or inspect.unwrap(accept_reject).__code__ is not sampler_identity[2] or type(prior.support).check is not support_identity[0] or type(prior.support).check.__code__ is not support_identity[1]:raise RuntimeError("original rejection/support identities changed")
  report["original_callable_seals"]=seals.descriptor()
  report["sources_unchanged"]=sha(__file__)==source and all(sha(path)==expected for path,expected in inputs.items()) and guard_baseline_source()==baseline
  passed=all((report["sources_unchanged"],report["global_rng_preserved"],report["clean_array_comparison"]["passed"],descriptor["original_alias_restored"],descriptor["sdk_sources_unchanged"],
              trace["summary"]["cuda_activity_status"]=="recorded",report["execution_counters"]["all_stage_aliases_restored"]))
  report.update(status="complete" if passed else "failed",all_guards_passed=bool(passed),completed_stage_indices=list(range(21)),
   timing_interpretation="profiled request span is instrumented; original clean195.18s request is a separate measurement; no overhead-corrected timing is invented",
   cautions=["Host range wall includes waits and transfers, not CPU compute.","Nested exclusive sampling is bookkeeping/query assembly/inverse arithmetic, not a standalone compute partition.","GPU kernel sums/union/copy/sync host waits overlap and cannot be added.","Observed SDPA operators and kernels determine backend evidence; flags alone do not establish FlashAttention.","Any clean-array mismatch is retained and fails the coupled diagnostic; original physical failure is not changed."])
 except BaseException as error:
  report.update(status="failed",error_type=type(error).__name__,error=str(error))
  if descriptor is not None:report["compatibility_adapter"]=dict(descriptor)
  if ranges is not None:report["owner_thread_ranges"]=ranges.events
  raise
 finally:
  report["total_seconds_including_profile_export_and_checks"]=time.perf_counter()-start;write_json(output/"manifest.json",report)
 return report


def main():
 parser=argparse.ArgumentParser(description=__doc__)
 for name in ("banks","output","checkpoint","ledger","reference","reference-grid","approximation","qualification","batch-qualification","sdk-root","clean-run"):parser.add_argument("--"+name,required=True)
 parser.add_argument("--max-seconds",type=int,default=780);result=run(**vars(parser.parse_args()))
 print(json.dumps({key:result.get(key) for key in ("status","returned_count","all_guards_passed","total_seconds_including_profile_export_and_checks")},indent=2))
 if result["status"]!="complete":raise SystemExit(2)


if __name__=="__main__":main()
