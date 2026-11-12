"""CPU-only ownership, bounded observation and exact mixed-dtype migration tests."""
from collections import Counter
from types import SimpleNamespace
import gc,json,weakref
import numpy as np
import pytest
from posteriorforge import execution_fp16_bank as bank

class Session:
 def __init__(self):
  self.counters=Counter();self.hist=Counter();self.examples=[];self.current=None
 def report(self):
  return {"counters":dict(self.counters),"witnesses":[{"count":v,**json.loads(k)} for k,v in self.hist.items()],
          "witness_examples":list(self.examples),"aliases_and_flags_restored":False}
 def event(self,stage,phase="query",kind=0):
  event={"stage":stage,"phase":phase,"member":kind%8,"kind":kind}
  self.hist[json.dumps(event,sort_keys=True)]+=1
  self.counters["qkv_calls"]+=1;self.counters["attention_calls"]+=1
  if len(self.examples)<16:self.examples.append(event)

def test_import_and_identity_seals_are_torch_free():
 import subprocess,sys
 subprocess.run([sys.executable,"-c","import sys;from posteriorforge import execution_fp16_bank as b;b.require_sources();assert 'torch' not in sys.modules"],check=True)
 bank.require_identity_seals()

@pytest.mark.parametrize("kind",["factory","method","migration"])
def test_in_process_replacement_rejected_even_with_unchanged_files(monkeypatch,kind):
 if kind=="factory":monkeypatch.setattr(bank.precision,"PrecisionStageExecutor",object)
 elif kind=="method":monkeypatch.setattr(bank._FACTORY,"predict_component",lambda *a:None)
 else:monkeypatch.setattr(bank.migration,"move_tensor_state",lambda *a:None)
 with pytest.raises(bank.BankError,match="callable identity"):bank.require_sources()

def test_21_window_rotation_preserves_counters_without_672_key_overflow():
 session=Session();windows=bank.DiagnosticWindows(session)
 for stage in range(21):
  windows.begin(stage,stage+1,True)
  for k in range(32):session.event(stage,"prefill" if k<8 else "query",k)
  row=windows.finish()
  assert len(row["precision_window"]["witnesses"])==32
  assert row["precision_window"]["aliases_and_flags_restored"] is False
 assert windows.verify_complete()["sum_window_deltas"]["qkv_calls"]==672
 assert session.counters["attention_calls"]==672 and len(session.hist)==32
 assert sum(len(w["precision_window"]["witnesses"]) for w in windows.windows)==672

def test_between_window_events_cannot_be_silently_cleared():
 s=Session();w=bank.DiagnosticWindows(s);w.begin(0,1,True);s.event(0);w.finish();s.event(0,kind=3)
 with pytest.raises(bank.BankError,match="between observation"):w.begin(1,2,True)
 assert s.hist
 with pytest.raises(bank.BankError,match="after final"):w.verify_complete()

@pytest.mark.parametrize("kind",["prefill","overflow","count"])
def test_warm_windows_reject_hidden_prefill_or_missing_witnesses(kind):
 s=Session();w=bank.DiagnosticWindows(s);w.begin(1,1,False)
 s.event(1,"prefill" if kind=="prefill" else "query")
 if kind=="overflow":s.counters["witness_histogram_overflow"]+=1
 if kind=="count":s.counters["qkv_calls"]+=1
 with pytest.raises(bank.BankError):w.finish()

def geometry(torch):
 module=torch.nn.Module()
 module.weight=torch.nn.Parameter(torch.arange(24,dtype=torch.float32).reshape(4,6))
 module.register_buffer("norm",torch.tensor([1.,2.],dtype=torch.float32))
 with torch.inference_mode():
  storage=torch.arange(72,dtype=torch.float16).reshape(8,9)
  module.kv={"base":storage,"slice":storage[1:7:2,2:8],"zero_stride":storage[2:3,:].expand(5,9),
    "mixed_dtype":storage.view(torch.uint8),"empty":torch.empty(0,dtype=torch.float16)}
 module.plain=storage.t()
 return module

def test_actual_eff_mover_preserves_half_views_aliases_offsets_and_original_cpu_dtype():
 torch=pytest.importorskip("torch")
 session=SimpleNamespace()
 stage=SimpleNamespace(base=SimpleNamespace(regressor=geometry(torch)),uncached_regressor=geometry(torch),session=session)
 coverage=bank.verify_root_coverage(stage,torch)
 assert coverage["passed"] and coverage["session_tensor_objects"]==0
 roots=bank.stage_roots(stage);before=bank.migration.state_manifest(roots,torch,hash_bytes=True)
 logical=bank.logical_snapshot(roots,torch);rng=torch.get_rng_state().clone()
 report=bank.migration.move_tensor_state(roots,torch,target="cpu",force_copy=True,verify_bytes=True)
 back=bank.migration.move_tensor_state(roots,torch,restore_devices=report["original_devices"],force_copy=True,verify_bytes=True)
 assert report["comparison"]["passed"] and back["comparison"]["full_storage_bytes_equal"]
 assert bank.migration.compare_state(before,back["after"],require_hashes=True)["passed"]
 assert logical==bank.logical_snapshot(roots,torch) and torch.equal(rng,torch.get_rng_state())
 assert {"torch.float16","torch.float32","torch.uint8"}<={r["dtype"] for r in before["tensors"]}
 assert any(0 in r["stride"] for r in before["tensors"])
 assert all(d=="cpu" for d in back["original_devices"].values())

def test_projection_refuses_session_tensor_or_foreign_registry():
 torch=pytest.importorskip("torch")
 stage=SimpleNamespace(base=SimpleNamespace(regressor=geometry(torch)),uncached_regressor=None,
                        session=SimpleNamespace(foreign=torch.ones(1)))
 with pytest.raises(bank.BankError,match="cover every stage"):bank.verify_root_coverage(stage,torch)

class Stage:
 def __init__(self,stage,session,torch,fail_prepare=False,fail_close=False):
  self.stage_index=stage;self.session=session;self.torch=torch
  self.base=SimpleNamespace(regressor=geometry(torch));self.uncached_regressor=None
  self.closed=0;self.fits=0;self.fail_prepare=fail_prepare;self.fail_close=fail_close
 @property
 def regressor(self):return self.base.regressor
 def prepare(self,theta,obs,stage):
  self.fits+=1;self.session.event(stage,"prefill")
  if self.fail_prepare:raise RuntimeError("prepare injected")
  return {"fit_stock_cache_seconds":1.,"resident_move_seconds":.5,"uncached_fit_seconds":0.}
 def predict_component(self,member,query,**kwargs):
  self.session.event(self.stage_index,kind=member)
  return {"logits":self.torch.from_numpy(np.asarray(query[:,:2],dtype=np.float32).copy())}
 def close(self):
  self.closed+=1;self.base.regressor=None;self.uncached_regressor=None
  if self.fail_close:raise RuntimeError("close injected")

def harness(*,bad_prepare=None,bad_close=None,memory=None):
 torch=pytest.importorskip("torch");session=Session();made=[]
 def factory(stage):
  value=Stage(stage,session,torch,stage==bad_prepare,stage==bad_close);made.append(value);return value
 owner=bank.FP16BankOwner._for_cpu_tests(session,torch,factory=factory,
  memory_probe=lambda:memory or {"rss_bytes":0,"cgroup_limit_bytes":None,"cgroup_current_bytes":None})
 return owner,made,torch

def arrays():
 return np.zeros((8,21),np.float32),np.zeros((8,33),np.float32)

def test_cold_offload_warm_hit_does_no_fit_and_queries_are_fresh():
 owner,made,torch=harness();theta,obs=arrays()
 first=owner.prepare(theta,obs,0)
 a=owner.predict_component(0,np.ones((3,33),np.float32))["logits"].clone()
 owner.close();warm=owner.prepare(theta,obs,0)
 b=owner.predict_component(0,np.full((3,33),2,np.float32))["logits"]
 again=owner.predict_component(0,np.ones((3,33),np.float32))["logits"]
 assert not first["bank_hit"] and warm["bank_hit"]
 assert warm["actual_fit_count"]==warm["actual_preprocess_refit_count"]==0
 assert len(made)==1 and made[0].fits==1 and torch.equal(a,again) and not torch.equal(a,b)
 assert owner.training_binding["query_excluded_from_training_key"]
 owner.close();owner.shutdown()
 assert owner.counters["misses"]==1 and owner.counters["hits"]==1
 assert owner.window_accounting["passed"] and made[0].closed==1

def test_changed_training_context_is_terminal_and_releases_all():
 owner,made,_=harness();theta,obs=arrays();owner.prepare(theta,obs,0);owner.close()
 obs[0,0]=1
 with pytest.raises(bank.BankError,match="context changed"):owner.prepare(theta,obs,1)
 assert owner.failed and not owner.entries and all(s.closed==1 for s in made)
 owner.shutdown()

def test_failed_cold_prepare_before_insertion_closes_local_engine():
 owner,made,_=harness(bad_prepare=0)
 with pytest.raises(RuntimeError,match="prepare injected"):owner.prepare(*arrays(),0)
 assert not owner.entries and made[0].closed==1 and owner.failed
 owner.shutdown()

def test_retained_corruption_cannot_reload():
 owner,made,_=harness();owner.prepare(*arrays(),0);owner.close()
 made[0].base.regressor.norm[0]=19
 with pytest.raises(bank.BankError,match="CPU stage changed"):owner.prepare(*arrays(),0)
 assert owner.failed and not owner.entries and made[0].closed==1
 owner.shutdown()

def test_cross_stage_half_storage_and_partial_overlaps_rejected():
 owner,_,torch=harness();theta,obs=arrays();owner.prepare(theta,obs,0);owner.close()
 shared=owner.entries[(owner.context,0)]["engine"].base.regressor.plain
 roots={"different_wrapper":torch.from_numpy(shared.numpy()[1:])}
 with pytest.raises(bank.BankError,match="overlapping storage"):owner._other_storage_guard(roots)
 owner.shutdown()

def test_zero_size_separate_storages_do_not_false_positive():
 owner,_,torch=harness();owner.prepare(*arrays(),0);owner.close()
 owner._other_storage_guard({"empty":torch.empty(0,dtype=torch.float16)})
 owner.shutdown()

def test_memory_policy_exact_100gib_plus8_and_cgroup_without_rss_subtraction():
 admit=bank.migration.memory_admission
 assert admit({"rss_bytes":91*bank.GIB},bank.GIB,100*bank.GIB,8*bank.GIB)["passed"]
 assert not admit({"rss_bytes":92*bank.GIB},1,100*bank.GIB,8*bank.GIB)["passed"]
 assert not admit({"rss_bytes":0,"cgroup_limit_bytes":8*bank.GIB,"cgroup_current_bytes":1},0,100*bank.GIB,8*bank.GIB)["passed"]
 with pytest.raises(bank.BankError,match="initial fixed"):harness(memory={"rss_bytes":93*bank.GIB})

def test_shutdown_archive_fault_still_releases_every_engine(monkeypatch):
 owner,made,_=harness();owner.prepare(*arrays(),0);owner.close();owner.prepare(*arrays(),1)
 def broken(**kw):raise RuntimeError("archive injected")
 monkeypatch.setattr(owner.windows,"finish",broken)
 with pytest.raises(bank.BankError,match="archive injected"):owner.shutdown()
 assert owner.closed and not owner.entries and all(s.closed==1 for s in made)

def test_close_failure_removes_owned_registry_and_root_references():
 owner,made,_=harness(bad_close=0);owner.prepare(*arrays(),0);owner.close()
 entry=next(iter(owner.entries.values()))
 with pytest.raises(bank.BankError,match="close injected"):owner.shutdown()
 assert not owner.entries and not entry["roots"] and "engine" not in entry and owner.closed

def test_context_key_requires_shapes_and_same_float32_bytes():
 owner,_,_=harness();theta,obs=arrays();k=owner._key(theta,obs)
 assert k==owner._key(theta.astype(np.float64),obs.astype(np.float64))
 theta[0,0]=np.nextafter(np.float32(0),np.float32(1))
 assert k!=owner._key(theta,obs)
 with pytest.raises(bank.BankError):owner._key(theta[:,:20],obs)
 owner.shutdown()

def test_many_disjoint_storage_intervals_are_accepted_without_pairwise_guard():
 owner,_,torch=harness()
 existing=[torch.empty(4,dtype=torch.float16) for _ in range(3000)]
 incoming=[torch.empty(5,dtype=torch.float16) for _ in range(3000)]
 owner.entries[("test",0)]={"roots":{"states":existing},"engine":SimpleNamespace(close=lambda:None)}
 owner._other_storage_guard({"states":incoming})
 owner.shutdown()
