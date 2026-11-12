"""CPU-only pass-through, exact full-count, interval and source guard tests."""
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import numpy as np
import pytest


def helper():
 path=Path(__file__).parents[1]/"scripts/profile_fp16_posterior.py"
 spec=importlib.util.spec_from_file_location("_fp16_profile_test",path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_import_is_torch_safe_and_sources_pinned():
 h=helper();subprocess.run([sys.executable,"-c","import runpy,sys;runpy.run_path("+repr(str(h.__file__))+",run_name='import_only');assert 'torch' not in sys.modules"],check=True)
 assert len(h.source_pins())==8


def test_adapter_calls_original_once_preserves_argument_return_identity_and_closes_on_error():
 h=helper();token=object();seen=[]
 class Original:
  stage_index=20;regressor=token
  def prepare(self,*a,**k):seen.append(("prepare",a,k));return token
  def predict_component(self,*a,**k):seen.append(("head",a,k));return token
  def predict(self,*a,**k):seen.append(("shared",a,k));return token
  def inspect_cache(self,*a,**k):seen.append(("inspect",a,k));return token
  def close(self,*a,**k):seen.append(("close",a,k));return token
 original=Original();ranges=h.HostRanges();adapter=h.RangeAdapter(original,ranges)
 assert adapter.regressor is token
 assert adapter.prepare(token,token,20,flag=token) is token
 assert adapter.predict_component(5,token,flag=token) is token
 assert adapter.predict(token,flag=token) is token and adapter.inspect_cache(flag=token) is token and adapter.close() is token
 assert seen==[("prepare",(token,token,20),{"flag":token}),("head",(5,token),{"flag":token}),("shared",(token,),{"flag":token}),("inspect",(),{"flag":token}),("close",(),{})]
 assert [row["operation"] for row in ranges.events]==["prepare","query_head","query_shared_stage0","inspect_guard","close"]
 assert original.prepare.__func__ is Original.prepare
 def fail(*a,**k):raise RuntimeError("injected original query failure")
 original.predict_component=fail
 with pytest.raises(RuntimeError,match="injected"):
  try:adapter.predict_component(5,token)
  finally:adapter.close()
 assert seen[-1][0]=="close" and ranges.events[-2]["error_type"]=="RuntimeError" and not ranges.stack


def test_factory_wraps_original_once_and_preserves_return_object_code_alias_and_rng():
 torch=pytest.importorskip("torch");h=helper();ranges=h.HostRanges();token=object();calls=[]
 def factory(*args,**kwargs):
  calls.append(("factory",args,kwargs))
  def stage(*args,**kwargs):calls.append(("stage",args,kwargs));return token
  return stage
 identity=factory;code=factory.__code__;before=torch.get_rng_state().clone()
 wrapped,proof=h.measured_stage_sampler(factory,token,ranges=ranges,flag=token)
 assert wrapped(token,20,token,50000,1024) is token and factory is identity and factory.__code__ is code
 assert calls==[("factory",(token,),{"flag":token}),("stage",(token,20,token,50000,1024),{})]
 assert proof["factory_original_calls"]==1 and not proof["private_function_aliases_replaced"] and torch.equal(before,torch.get_rng_state())


def test_exclusive_host_sampling_subtracts_child_union_without_double_counting():
 h=helper();ranges=h.HostRanges()
 ranges.events=[{"operation":"sampling_stage","start_seconds":0.,"stop_seconds":10.,"host_wall_seconds":10.,"stage_index":20,"proposal_wave":1},
  {"operation":"query_head","start_seconds":1.,"stop_seconds":6.},
  {"operation":"prediction_check","start_seconds":4.,"stop_seconds":7.},
  {"operation":"query_shared_stage0","start_seconds":8.,"stop_seconds":9.}]
 result=ranges.exclusive_sampling()
 assert result["exclusive_host_seconds"]==3. and result["stages"][0]["nested_query_and_callback_union_seconds"]==7.


def test_nonowner_ranges_are_rejected_without_mutation(monkeypatch):
 h=helper();ranges=h.HostRanges();monkeypatch.setattr(h,"get_ident",lambda:ranges.owner+1)
 with pytest.raises(RuntimeError,match="owner"):
  with ranges.span("prepare",0):pass
 assert not ranges.events and not ranges.stack


def test_correlation_attribution_filters_gpu_duplicates_and_retains_unmatched_copies(tmp_path):
 h=helper();reader=h.load_script("diagnose_runtime.py",h.READER_SHA);parser=h.load_script("diagnose_prefill_allocator.py",h.PARSER_SHA)
 events=[{"ph":"X","cat":"user_annotation","name":h.BOUNDARY,"ts":0,"dur":1000,"pid":1,"tid":2},
  {"ph":"X","cat":"gpu_user_annotation","name":h.BOUNDARY,"ts":10,"dur":900,"pid":0,"tid":7},
  {"ph":"X","cat":"user_annotation","name":"pfdiagnostic/profile/prepare/w1/s0/m-1","ts":10,"dur":100,"pid":1,"tid":2},
  {"ph":"X","cat":"user_annotation","name":"pfdiagnostic/profile/query_head/w1/s0/m5","ts":200,"dur":100,"pid":1,"tid":2},
  {"ph":"X","cat":"cuda_runtime","name":"cudaLaunchKernel","ts":20,"dur":1,"pid":1,"tid":2,"args":{"correlation":1}},
  {"ph":"X","cat":"cuda_runtime","name":"cudaLaunchKernel","ts":220,"dur":1,"pid":1,"tid":2,"args":{"correlation":2}},
  {"ph":"X","cat":"kernel","name":"prefill_kernel","ts":90,"dur":100,"args":{"correlation":1}},
  {"ph":"X","cat":"kernel","name":"fmha_cutlassF","ts":290,"dur":200,"args":{"correlation":2}},
  {"ph":"X","cat":"gpu_memcpy","name":"Memcpy DtoH (Device -> Pageable)","ts":700,"dur":100,"args":{"bytes":4096,"correlation":999}},
  {"ph":"X","cat":"cpu_op","name":"aten::_scaled_dot_product_efficient_attention","ts":210,"dur":80,"pid":1,"tid":2}]
 path=tmp_path/"trace.json";path.write_text(json.dumps({"traceEvents":events}));raw=path.read_bytes()
 result=h.trace_attribution(path,reader);summary=parser.summarize_profile_trace(path)
 assert path.read_bytes()==raw and result["groups"]["prepare"]["kernel_count"]==1 and result["groups"]["query_head"]["kernel_count"]==1
 assert result["groups"]["query_head"]["kernel_interval_union_seconds"]==pytest.approx(.0002)
 assert result["groups"]["unmatched"]["copy_bytes"]==4096 and result["observed_SDPA_operator_counts"]=={"aten::_scaled_dot_product_efficient_attention":1}
 assert not result["flash_attention_inferred_from_flags"] and summary["profiled_request_trace_span_seconds"]==.001


def test_full50000_fake_stream_uses_original_reject_fifo_and_array_comparison():
 torch=pytest.importorskip("torch");h=helper()
 from posteriorforge import performance_fp16_experiment as candidate
 from posteriorforge.inference import _prior
 from posteriorforge.upstream import load_baseline
 from importlib import import_module
 load_baseline();accept=import_module("npe_pfn.accept_reject_sampler").accept_reject_sample
 h.require_original_sampling_code(accept,_prior);frozen=candidate.frozen_sampling();before_codes={name:getattr(frozen,name).__code__ for name in ("component_stage_sampler","stage_major_wave")}
 outputs=[]
 for _ in range(2):
  gen=torch.Generator().manual_seed(13);counts={"calls":0,"checked":0,"accepted":0}
  def provider(n):
   values=torch.zeros((n,21),dtype=torch.float32);values[:,0]=torch.rand(n,generator=gen);return values,None
  def support(values):
   mask=torch.ones(len(values),dtype=torch.bool)
   if counts["calls"]==0:mask[:10]=False
   counts["calls"]+=1;counts["checked"]+=len(values);counts["accepted"]+=int(mask.sum());return mask
  fifo=frozen.guarded_proposal_buffer().ProposalBuffer(provider,scope={"case":0,"query":{"marker":0}},concatenate=lambda parts:torch.cat(parts),world_size=1,rank=0)
  samples,_,_=accept(proposal=fifo,accept_reject_fn=support,num_samples=50000,max_sampling_batch_size=50000,max_iter_rejection=None,show_progress_bars=False)
  assert samples.shape==(50000,21) and counts=={"calls":2,"checked":50100,"accepted":50090}
  outputs.append(samples.numpy().astype(np.float64))
 assert h.payload_comparison(*outputs)["passed"]
 outputs[1][0,0]+=1
 assert not h.payload_comparison(*outputs)["passed"]
 assert all(getattr(frozen,name).__code__ is code for name,code in before_codes.items()) and not torch.cuda._initialized


def test_seals_reject_original_alias_change_without_accepting_wrapper_as_original(monkeypatch):
 h=helper()
 def function(*a,**k):return None
 names=("stage_major_wave","component_stage_sampler","worker_stream_seeds","guarded_proposal_buffer","validate_proposal_buffer_record")
 frozen=SimpleNamespace(**{name:function for name in names})
 cls=type("FakeStage",(),{name:function for name in ("prepare","predict","predict_component","inspect_cache","close","metadata")})
 candidate=SimpleNamespace(frozen_sampling=lambda:frozen,validate_all21=function,execution_metadata=function,precision_engine_scope=function,validate_prediction=function,FP16StageAdapter=cls)
 seals=h.Seals(candidate,frozen);seals.verify();original=frozen.component_stage_sampler
 frozen.component_stage_sampler=lambda *a,**k:None
 with pytest.raises(RuntimeError,match="alias/code changed"):seals.verify()
 frozen.component_stage_sampler=original;seals.verify()
 assert seals.descriptor()["original_alias_replacements"]==[]
