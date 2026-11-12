"""CPU-only FP16 experiment governance/law tests; no model/CUDA execution."""
from copy import deepcopy
import hashlib,importlib.util,json,subprocess,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from posteriorforge import performance_fp16_experiment as runner

ROOT=Path(__file__).parents[1]
PHASE=Path("/scratch/project/eu-26-17/eubrieucb/posteriorforge/results/performance-first-v1")
SCREEN=PHASE/"fp16-all21-approx-characterization-01/manifest.json"

def test_import_help_metadata_and_authenticated_git_functions_are_torch_free():
 subprocess.run([sys.executable,"-c",
  "import sys;from posteriorforge.performance_fp16_experiment import execution_metadata; m=execution_metadata();"
  "assert 'torch' not in sys.modules;assert not m['qualified_fp32'];assert m['source_checkout_required']"],check=True,cwd=ROOT)
 result=subprocess.run([sys.executable,"-m","posteriorforge.performance_fp16_experiment","--help"],capture_output=True,text=True,check=True)
 assert "--approximation" in result.stdout and "--batch-qualification" in result.stdout
 frozen=runner.frozen_sampling()
 assert frozen.RUNNER_VERSION==4 and frozen.COUNT==50000
 assert runner.execution_metadata()["stage0_real_head_contexts_per_shared_density"]==8

def test_unpinned_artifact_disables_posterior_before_budget_or_cuda(tmp_path,monkeypatch):
 monkeypatch.setattr(runner,"ALL21_MANIFEST_SHA256",None)
 with pytest.raises(RuntimeError,match="not been pinned"):
  runner.run_experiment(banks="none",output=tmp_path/"fresh",checkpoint="none",ledger="none",
   reference="none",reference_grid="none",approximation="none",qualification="none",batch_qualification="none",sdk_root="none")
 assert not (tmp_path/"fresh").exists()

def test_changed_private_sampling_function_is_rejected(monkeypatch):
 frozen=runner.frozen_sampling()
 with monkeypatch.context() as patch:
  patch.setattr(frozen,"stage_major_wave",lambda *a,**k:None)
  with pytest.raises(RuntimeError,match="private frozen"):runner.frozen_sampling()
 runner.frozen_sampling()

def event(stage=1,member=2,strict=False):
 return {"phase":"query","stage":stage,"member":member,"autocast_enabled":not strict,
  "autocast_dtype":"torch.float16","allow_tf32":False,"matmul_precision":"highest",
  "projection_dtypes":{"q":"torch.float32" if strict else "torch.float16","k":None,"v":None,"kv":None,"qkv":None}}

def test_actual_query_witness_binds_member_stage_forward_dtype_and_scope_delta():
 key=json.dumps(event(),sort_keys=True)
 actual=runner.query_witness_delta({key:3},{key:5},stage=1,member=2)
 assert actual["computed_projection_dtype"]=="torch.float16" and actual["passed"]
 strict=json.dumps(event(10,5,True),sort_keys=True)
 assert runner.query_witness_delta({}, {strict:1},stage=10,member=5)["strict_FP32_fallback"]

@pytest.mark.parametrize("bad",["member","stage","dtype","autocast","TF32","empty"])
def test_actual_query_scope_cannot_be_replaced_by_precision_labels(bad):
 value=event()
 if bad=="member":value["member"]=3
 elif bad=="stage":value["stage"]=2
 elif bad=="dtype":value["projection_dtypes"]["q"]="torch.float32"
 elif bad=="autocast":value["autocast_enabled"]=False
 elif bad=="TF32":value["allow_tf32"]=True
 hist={} if bad=="empty" else {json.dumps(value,sort_keys=True):1}
 with pytest.raises(RuntimeError):runner.query_witness_delta({},hist,stage=1,member=2)

@pytest.mark.parametrize("failure",["enter","constructor","prepare"])
def test_prepare_failures_close_only_successfully_entered_precision_scopes(monkeypatch,failure):
 from posteriorforge import execution_precision_probe as precision
 track={"exit":0,"close":0}
 class Session:
  def __init__(self,mode):pass
  def __enter__(self):
   if failure=="enter":raise RuntimeError("synthetic enter")
   return self
  def __exit__(self,*args):track["exit"]+=1
  def report(self):return {"aliases_and_flags_restored":True,"counters":{}}
 class Engine:
  def __init__(self,*args,**kwargs):
   if failure=="constructor":raise RuntimeError("synthetic constructor")
  def prepare(self,*args):raise RuntimeError("synthetic prepare")
  def close(self):track["close"]+=1
 monkeypatch.setattr(precision,"PrecisionSession",Session);monkeypatch.setattr(precision,"PrecisionStageExecutor",Engine)
 adapter=runner.FP16StageAdapter("none",cdf_descriptor={})
 with pytest.raises(RuntimeError,match="synthetic"):adapter.prepare(None,None,0)
 assert adapter.session is None and adapter.engine is None
 assert track["exit"]==(0 if failure=="enter" else 1)
 assert track["close"]==(1 if failure=="prepare" else 0)
 adapter.close()

def test_shared_density_and_physical_head_context_counts_are_separate():
 proof={"mode":runner.MODE,"precision_module_sha256":runner.PRECISION_SHA,"module_sha256":runner.source_sha256(),
   "stage":0,"member":None,"actual_head_contexts":8,"corrected_CDF_calls":8}
 report={};runner.validate_prediction({"fp16_execution_proof":proof},0,None,report=report)
 assert report["sampler_callback_calls"]==1 and report["actual_head_contexts"]==8 and report["shared_densities"]==1
 assert not report["original_full_API_identity_claimed"]
 bad={**proof,"actual_head_contexts":1}
 with pytest.raises(RuntimeError):runner.validate_prediction({"fp16_execution_proof":bad},0,None,report={})

def test_frozen_support_fifo_trimming_returns_exact50000_complete_vectors():
 torch=pytest.importorskip("torch")
 from posteriorforge.inference import _prior
 from posteriorforge.science.analytical_benchmark_v2 import build_problem
 from posteriorforge.upstream import load_baseline
 from importlib import import_module
 frozen=runner.frozen_sampling();fifo=frozen.guarded_proposal_buffer();prior=_prior(torch);load_baseline()
 accept=import_module("npe_pfn.accept_reject_sampler").accept_reject_sample
 indices=[build_problem().parameter_order.index(n) for n in build_problem().prior.noise_names]
 sizes=[]
 def provider(n):
  sizes.append(n);x=torch.zeros(n,21);x[:,indices]=.25;x[:,0]=torch.arange(n,dtype=torch.float32)
  x[:2,indices[0]]=.6
  return x,None
 scope={"case":0,"query":{"sha256":"fixed"}}
 buffer=fifo.ProposalBuffer(provider,scope=scope,concatenate=lambda p:torch.cat(p),world_size=1,rank=0)
 samples,_,_=accept(proposal=buffer,accept_reject_fn=prior.support.check,num_samples=50000,
   max_sampling_batch_size=50000,max_iter_rejection=None,show_progress_bars=False,proposal_sampling_kwargs={})
 record=buffer.finalize(original_checked_count=prior.support.proposed,accepted_before_trim=prior.support.accepted,
    returned_count=len(samples),original_support_calls=prior.support.calls)
 record.update(enabled=True,source_unchanged=True,policy_sha256=frozen.proposal_buffer_policy(1200)["policy_sha256"])
 assert frozen.validate_proposal_buffer_record(record,world_size=1,rank=0,scope=scope)
 assert sizes==[51200] and samples.shape==(50000,21) and bool(prior.support.check(samples).all())
 assert samples[:,0].tolist()==list(range(2,50002))
 assert record["generated_count"]==record["original_api_delivered_count"]+record["remaining_unused_buffer_count"]

def actual_bindings():
 from posteriorforge.banks import load_context,load_heldout
 from posteriorforge.science.analytical_benchmark_v2 import build_problem,array_descriptor
 from posteriorforge.inference import CHECKPOINT_SHA256
 if not SCREEN.exists():pytest.skip("authenticated HPC all21 artifact unavailable")
 banks=PHASE.parent/"banks-v1";context=load_context(banks,20261111,4096);observed=load_heldout(banks,0);problem=build_problem()
 return {"checkpoint_sha256":CHECKPOINT_SHA256,"scientific_fingerprint":problem.scientific_fingerprint,
  "parameter_order":list(problem.parameter_order),"context_theta":array_descriptor(context.theta),
  "context_obs":array_descriptor(context.obs),"query":array_descriptor(observed.obs.reshape(1,33))}

def test_real_all21_artifact_and_scientific_bindings_preserve_strict_failure():
 q=runner.validate_all21(SCREEN,actual_bindings())
 assert q["all21_hard_validity_passed"] and not q["all21_strict_screen_passed"]
 assert q["strict_failed_stages"]==list(range(21)) and q["expected_patched_call_count"]==688

@pytest.mark.parametrize("bad",["checksum","order","strict_relabel","self_numeric","mass"])
def test_pinned_artifact_checks_reject_tamper_and_independently_check_hard_numerics(tmp_path,monkeypatch,bad):
 bindings=actual_bindings();q=json.loads(SCREEN.read_text())
 if bad=="checksum":
  p=tmp_path/"bad.json";p.write_bytes(SCREEN.read_bytes()+b"\n")
 elif bad=="order":
  bindings["parameter_order"]=bindings["parameter_order"][::-1];p=SCREEN
 else:
  if bad=="strict_relabel":q["strict_failed_stages"]=[]
  elif bad=="self_numeric":q["stage_results"][0]["Cp_self_reconstruction"]["1024"]["normalized_actual_raw_mean_vs_reference"]["max_cdf_absolute_error"]=2.1e-5
  else:q["stage_results"][0]["probability_validity"]["1024"]["Cp"]["raw_head_masses"][0]["max"]=1.00002
  p=tmp_path/"changed.json";p.write_text(json.dumps(q))
  monkeypatch.setattr(runner,"ALL21_MANIFEST_SHA256",runner.sha256_file(p))
 with pytest.raises(RuntimeError):runner.validate_all21(p,bindings)


@pytest.mark.parametrize("failure",["inspect","compact"])
def test_frozen_pretry_cache_failure_closes_precision_before_outer_CDF(monkeypatch,failure):
 frozen=runner.frozen_sampling();events=[];state={"active":False}
 class FakeCDF:
  def __enter__(self):state["active"]=True;events.append("cdf_enter");return {}
  def __exit__(self,*args):
   assert "precision_close" in events
   events.append("cdf_exit");state["active"]=False
 class FakeEngine:
  def __init__(self,*args,**kwargs):pass
  def prepare(self,*args):events.append("prepare");return {}
  def inspect_cache(self):
   events.append("inspect")
   if failure=="inspect":raise RuntimeError("synthetic inspect")
   return {"tensor_records":[{"category":"parameter"}]}
  def close(self):
   assert state["active"]
   events.append("precision_close")
 monkeypatch.setattr(runner,"FP16StageAdapter",FakeEngine)
 with pytest.raises((RuntimeError,KeyError)):
  with FakeCDF() as descriptor,runner.precision_engine_scope("none",cdf_descriptor=descriptor) as engine:
   frozen.stage_major_wave(engine,None,None,np.zeros(33),1,1,lambda _:None)
 assert events==["cdf_enter","prepare","inspect","precision_close","cdf_exit"]
