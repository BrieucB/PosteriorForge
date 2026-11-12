"""CPU-only contract and fault tests for the NEW FP16 retention proof."""
from pathlib import Path
from types import SimpleNamespace
from collections import Counter
from contextlib import contextmanager
import copy,hashlib,importlib.util,json,subprocess,sys
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("fp16_bank_proof",ROOT/"scripts/qualify_fp16_bank.py")
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)

def test_sources_help_and_all21_preflight_remain_torch_free():
 helper.sources()
 subprocess.run([sys.executable,"-c","import importlib.util,sys;s=importlib.util.spec_from_file_location('p','scripts/qualify_fp16_bank.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.sources();assert 'torch' not in sys.modules"],cwd=ROOT,check=True)
 result=subprocess.run([sys.executable,str(ROOT/"scripts/qualify_fp16_bank.py"),"--help"],capture_output=True,text=True,check=True)
 assert "--all21-artifact" in result.stdout and "--max-seconds" in result.stdout

def valid_counts():
 counters={"logical_acquires":63,"misses":21,"hits":42,"reload_count":42,"offload_count":63,"physical_head_calls":2056}
 windows=[{"stage_index":stage,"cold":sweep==0,"counter_delta":{"qkv_calls":1,"attention_calls":1},
           "precision_window":{"witnesses":[{"phase":"prefill" if sweep==0 else "query"}]}}
          for sweep in range(3) for stage in range(21)]
 return counters,windows

def test_exact_cyclic_coverage_and_zero_warm_prefill():
 counters,windows=valid_counts();result=helper.validate_counts(counters,windows)
 assert result["total_migrations"]==105 and result["warm_actual_fits"]==result["warm_actual_preprocess_refits"]==0

@pytest.mark.parametrize("kind",["refit","order","prefill","overflow","missing","attention","heads"])
def test_partial_or_wrong_coverage_cannot_pass(kind):
 counters,windows=valid_counts()
 if kind=="refit":counters["misses"]+=1
 elif kind=="order":windows[-1]["stage_index"]=0
 elif kind=="prefill":windows[-1]["precision_window"]["witnesses"][0]["phase"]="prefill"
 elif kind=="overflow":windows[-1]["counter_delta"]["witness_histogram_overflow"]=1
 elif kind=="attention":windows[-1]["counter_delta"]["attention_calls"]=0
 elif kind=="heads":counters["physical_head_calls"]=2055
 else:windows.pop()
 with pytest.raises(RuntimeError):helper.validate_counts(counters,windows)

def prediction(torch):
 b=torch.tensor([-4.,0.,2.,99.],dtype=torch.float32)
 logits=torch.tensor([[-float("inf"),0.,-2.],[1.,0.,-35.]],dtype=torch.float32)
 members=[{"raw_logits":logits.clone(),"translated_probabilities":logits.softmax(-1),
           "from_borders":b.clone(),"to_borders":b.clone(),"configuration_identity":{"member":i}} for i in range(8)]
 return {"logits":logits,"criterion":SimpleNamespace(borders=b),"members":members}

def test_prediction_wholebatch_hash_and_actual_e71_inverse_preserve_nan_rng_and_detect_single_bit():
 torch=pytest.importorskip("torch");p=prediction(torch);rng=torch.get_rng_state().clone()
 first=helper.prediction_fingerprint(p,torch)
 assert first["existing_inverse_nonfinite"]==1 and not first["native_or_scalar_inverse_replay_performed"]
 assert torch.equal(rng,torch.get_rng_state())
 assert first==helper.prediction_fingerprint(p,torch)
 p["members"][7]["translated_probabilities"][0,1]=torch.nextafter(p["members"][7]["translated_probabilities"][0,1],torch.tensor(0.))
 assert first!=helper.prediction_fingerprint(p,torch)

def test_shared_singleton_uses_actual_production_domain_u_zero_half_max():
 torch=pytest.importorskip("torch");p=prediction(torch);p["logits"]=p["logits"][:1]
 r=helper.shared_fingerprint(p,torch)
 assert r["same_wholebatch_shared_inverse"]["shape"]==[3]
 assert r["existing_inverse_nonfinite"]>=1

def test_new_library_baseline_no_old_residual_or_private_clear_cpu_api_surrogate():
 torch=pytest.importorskip("torch");previous=torch.get_num_threads();torch.set_num_threads(1)
 class CUDA:
  bytes=0;clears=0
  def memory_allocated(self):return self.bytes
  def memory_reserved(self):return self.bytes+40
  def synchronize(self):self.bytes=12345
  def empty_cache(self):self.clears+=1
 cuda=CUDA()
 def ones(*args,**kw):
  assert kw.pop("device")=="cuda"
  return torch.ones(*args,**kw)
 fake=SimpleNamespace(cuda=cuda,ones=ones,float16=torch.float16,float32=torch.float32,nn=torch.nn,isfinite=torch.isfinite)
 rng=lambda:hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
 try:r=helper.library_baseline(fake,rng)
 finally:torch.set_num_threads(previous)
 assert r["fresh_allocated_bytes"]==0 and r["allocated_bytes"]==12345 and cuda.clears==1
 assert not r["old_residual_allowance_used"] and not r["private_cuda_clear_called"] and r["rng_preserved"]

def test_existing_gpu_allocated_state_refuses_library_baseline():
 fake=SimpleNamespace(cuda=SimpleNamespace(memory_allocated=lambda:1))
 with pytest.raises(RuntimeError,match="fresh GPU"):helper.library_baseline(fake,lambda:"rng")

def test_exact_new_baseline_residual_never_subtracts_allowance(monkeypatch):
 fake=SimpleNamespace(cuda=SimpleNamespace(memory_allocated=lambda:10,synchronize=lambda:None))
 assert helper.assert_residual(fake,{"allocated_bytes":10},"allCPU")["exact_equal"]
 with pytest.raises(RuntimeError,match="residual differs"):helper.assert_residual(fake,{"allocated_bytes":9},"allCPU")

def test_logical_values_ignore_predictor_buffer_id_layout_but_not_values_shapes_dtype():
 a=[{"paths":["stage/a","stage/a/<view_base>"],"dtype":"torch.float16","shape":[2],
     "sha256":"ab","stride":[1],"tensor_id":1,"storage_offset":0}]
 b=copy.deepcopy(a);b[0]["stride"]=[2];b[0]["tensor_id"]=999
 assert helper.logical_values(a)==helper.logical_values(b)
 b[0]["sha256"]="cd"
 assert helper.logical_values(a)!=helper.logical_values(b)

def test_archive_roundtrip_and_checksums(tmp_path):
 import gzip
 r=helper.archive(tmp_path,"state",{"a":[1,2],"label":"Built with PriorLabs-TabPFN"})
 path=tmp_path/r["file"]
 assert hashlib.sha256(path.read_bytes()).hexdigest()==r["sha256"]
 with gzip.open(path,"rt") as f:assert json.load(f)["a"]==[1,2]

def test_window_member_projection_coverage_cold_provenance_and_no_fake_warm_prefill():
 old={"computed_query_witness":{"members":[{"member":i,"query_event_count":24} for i in range(8)]},
      "computed_prefill_witness":{"passed":True,"members":[]}}
 rows=[{"phase":"query","stage":2,"member":i} for i in range(8)]
 window={"precision_window":{"witnesses":rows},"counter_delta":{"qkv_calls":480,"attention_calls":480}}
 class Precision:
  def query_witness_passed(self,r,s):return {"passed":True,"members":[{"member":i,"query_event_count":60} for i in range(8)]}
 result=helper.validate_observation_window(window,old,2,1,Precision())
 assert result["actual_query_calls_per_member"]==5 and result["actual_warm_prefill_events"]==0
 window["precision_window"]["witnesses"][0]["stage"]=3
 with pytest.raises(RuntimeError,match="leaked"):helper.validate_observation_window(window,old,2,1,Precision())

def test_bank_release_before_continuous_precision_then_cdf_even_fault(monkeypatch):
 from posteriorforge import execution_fp16_bank as bank
 events=[]
 class Session:
  def __init__(self,*a):pass
  def __enter__(self):events.append("051enter");return self
  def __exit__(self,*a):events.append("051exit")
 class Owner:
  def __init__(self,*a,**kw):events.append("bankcreate")
  def shutdown(self):events.append("bankrelease")
 @contextmanager
 def cdf():
  events.append("cdfenter")
  try:yield {}
  finally:events.append("cdfexit")
 monkeypatch.setattr(bank.precision,"PrecisionSession",Session)
 monkeypatch.setattr(bank,"FP16BankOwner",Owner)
 with pytest.raises(RuntimeError,match="postprepare"):
  with cdf() as active:
   with bank.fp16_bank_scope("checkpoint",cdf_descriptor=active):
    raise RuntimeError("postprepare inspect fault")
 assert events==["cdfenter","051enter","bankcreate","bankrelease","051exit","cdfexit"]

def test_shutdown_archive_error_still_exits_precision_before_cdf(monkeypatch):
 from posteriorforge import execution_fp16_bank as bank
 events=[]
 class Session:
  def __init__(self,*a):pass
  def __enter__(self):return self
  def __exit__(self,*a):events.append("051exit")
 class Owner:
  def __init__(self,*a,**kw):pass
  def shutdown(self):
   events.append("release-before-archive-error");raise RuntimeError("archive failed after release")
 monkeypatch.setattr(bank.precision,"PrecisionSession",Session);monkeypatch.setattr(bank,"FP16BankOwner",Owner)
 try:
  with bank.fp16_bank_scope("checkpoint",cdf_descriptor={}):pass
 except RuntimeError:
  events.append("outer-cdfexit")
 assert events==["release-before-archive-error","051exit","outer-cdfexit"]

def test_actual_authentic_scientific_bindings_without_model_or_reference_calls():
 phase=Path("/scratch/project/eu-26-17/eubrieucb/posteriorforge/results/performance-first-v1")
 root=phase.parents[1]
 if not phase.is_dir():pytest.skip("remote artifacts unavailable")
 args=dict(banks=phase.parent/"banks-v1",prefix_run=phase.parent/"cdf-arithmetic-v1/context-4096/seed-20261111-case-0",
  checkpoint=root/"cache/model/tabpfn-v2-regressor.ckpt",qualification=phase/"gqa-hybrid-qualification-recovered-01/manifest.json",
  batch_qualification=phase/"gqa-large-batch-qualification-01/manifest.json",
  sdk_root=root/"envs/inference/lib/python3.11/site-packages/tabpfn",
  all21_artifact=phase/"fp16-all21-approx-characterization-01/manifest.json")
 context,a,b,prefixes,positions,problem,bindings,q,receipt,approx=helper.read_bindings(**args)
 assert context.theta.shape==(4096,21) and prefixes.shape==(1024,21)
 assert not np.array_equal(a,b) and approx["strict_failed_stages"]==list(range(21))
 assert receipt["replay"]["existing_narrow_identity_decisions_replayed"]==184
 assert approx["all21_hard_validity_passed"] and not approx["all21_strict_screen_passed"]

@pytest.mark.parametrize("bad",["normalizer","cachedamp","fallbackamp","borders","temperature"])
def test_model_stats_policy_cannot_mask_low_precision_normalizers_or_fallback(bad):
 resident={"contract":{"input_normalization_train_only_verified":True,"autocast":True,
   "forced_inference_dtype":"None","modules":[{"type":"InputNormalizationEncoderStep",
      "tensors":{"mean":{"dtype":"torch.float32"}}}]},
   "canonical_borders_dtype":"torch.float32","physical_borders_dtype":"torch.float32","temperature":.9}
 fallback=copy.deepcopy(resident);fallback["contract"].update(autocast=False,forced_inference_dtype="torch.float32")
 state={"resident":resident,"uncached":fallback}
 helper.check_model_state(state)
 if bad=="normalizer":resident["contract"]["modules"][0]["tensors"]["mean"]["dtype"]="torch.float16"
 elif bad=="cachedamp":resident["contract"]["autocast"]=False
 elif bad=="fallbackamp":fallback["contract"]["autocast"]=True
 elif bad=="borders":resident["physical_borders_dtype"]="torch.float16"
 else:resident["temperature"]=1.
 with pytest.raises(RuntimeError):helper.check_model_state(state)
