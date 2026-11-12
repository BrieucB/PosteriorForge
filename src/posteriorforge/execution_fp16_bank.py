"""FP16 prepared-bank retention with one continuous frozen051 precision session.

Built with PriorLabs-TabPFN. New ownership only: original051 computation,
e71 probabilities/inverse arithmetic and generic eff storage migration unchanged.
No query density, posterior prefix, uniform or head assignment is retained.
"""
from __future__ import annotations
from collections import defaultdict
from contextlib import ExitStack,contextmanager
import copy,gc,hashlib,importlib.util,inspect,json,os,time
from pathlib import Path
from threading import Lock,get_ident
import weakref
import numpy as np
from . import execution_precision_probe as precision,execution_host_cache as migration
from . import performance_fp16_experiment as clean
from .artifacts import sha256_file

SCHEMA="posteriorforge.execution.fp16_prepared_bank.v1"
ENGINE_ID="continuous_fp16_prepared_bank_v1"
MODE="amp_fp16_prefill_query_v1"
GIB=1024**3
PRECISION_SHA="051f477ff4b9ab8bbcfbaf3c1afa4dcf6b111d2f2fbcf9efb9c94a00d2b2ad3d"
MIGRATION_SHA="effc86e17e33adb96fa8130c476a143b378a63c6d81bf71a31e3e2cd34e9bbde"
CLEAN_SHA="2280e1ceef807f05a264b5318be18eb8bee7c52c8bab21884d6682f84cafcba9"
BASE_SHA="c55c00a1697f8d4341880af18676c29b4d99551e131c89ff79bea238c693f8bd"
ALL21_SHA="20e64a64357de7862c20b2fa846c6d128c181ca564c261800b626e6c4bd32208"
_OWNER_LOCK=Lock()
_OWNER=None
_FACTORY=precision.PrecisionStageExecutor
_RESIDENT=precision.PrecisionResident
def _callable_seal(owner,name):
 value=inspect.getattr_static(owner,name)
 function=value.__func__ if isinstance(value,(classmethod,staticmethod)) else value
 return owner,name,value,getattr(function,"__code__",None)
_SEALS=[_callable_seal(owner,name) for owner,names in (
 (precision,("PrecisionStageExecutor","PrecisionResident","PrecisionSession","capture_members")),
 (_FACTORY,("__init__","prepare","predict_component","close","inspect_cache")),
 (_RESIDENT,("__init__","prepare","predict","close")),
 (migration,("tensor_inventory","state_manifest","move_tensor_state","compare_state","memory_admission","process_memory")))
 for name in names]

def require_identity_seals():
 for owner,name,original,code in _SEALS:
  current=inspect.getattr_static(owner,name)
  function=current.__func__ if isinstance(current,(classmethod,staticmethod)) else current
  require(current is original and getattr(function,"__code__",None) is code,
          "frozen in-process callable identity changed: "+name)

class BankError(RuntimeError):pass
def require(value,message):
 if not value:raise BankError(message)
def execution_source_sha256():return sha256_file(__file__)
def require_sources():
 require_identity_seals()
 require(precision.source_sha256()==PRECISION_SHA,"frozen051 source changed");precision.require_sources()
 require(migration.execution_source_sha256()==MIGRATION_SHA,"generic migration source changed")
 require(clean.source_sha256()==CLEAN_SHA,"clean2280 source changed")

def execution_metadata():
 require_sources()
 return {"schema":SCHEMA,"engine_id":ENGINE_ID,"module_sha256":execution_source_sha256(),
  "precision_module_sha256":PRECISION_SHA,"migration_primitives_source_sha256":MIGRATION_SHA,"clean_source_sha256":CLEAN_SHA,
  "factory":"frozen051 PrecisionStageExecutor (same exact active Session)","forward_precision":MODE,
  "weights_and_normalizers_dtype":"torch.float32","cached_projection_dtype":"computed torch.float16; stage10/member5 FP32",
  "residency":"fixed admitted bank; all inactive stage roots CPU, at most one stage CUDA",
  "root_projection":["resident_regressor","uncached_regressor"],"session_and_bank_registry_excluded_from_migration":True,
  "migration":"full untyped-storage byte copy; same Tensor identities/alias/layout/inference/version within each migration",
  "diagnostic_windows":"archive/reset hist and examples only; cumulative051 counters never reset",
  "window_histogram_limit":512,"window_examples_limit":16,
  "active_alias_policy":"owned private051 SDK aliases remain active until final session exit",
  "temporary_close":"offload only","shutdown":"release all entries before precision then CDF exits",
  "query_density_memoization":False,"request_sampling_state_retained":False,
  "rss_limit_bytes":100*GIB,"safety_slack_bytes":8*GIB,"tail_family_changed":False,
  "qualified_FP32":False,"attribution":"Built with PriorLabs-TabPFN"}

def stage_roots(stage):
 """Only these original regressor objects own current stage tensor state."""
 require(stage.base.regressor is not None,"prepared resident regressor required")
 return {"resident_regressor":stage.base.regressor,"uncached_regressor":stage.uncached_regressor}

def tensor_ids(root,torch):return {id(row["tensor"]) for row in migration.tensor_inventory(root,torch)}

def verify_root_coverage(stage,torch):
 roots=stage_roots(stage);projected=tensor_ids(roots,torch);full=tensor_ids(stage,torch)
 require(projected==full,"explicit migration roots do not cover every stage-owned Tensor")
 require(not tensor_ids(stage.session,torch),"continuous Session unexpectedly owns Tensor state/backreferences")
 return {"passed":True,"projected_tensor_objects":len(projected),"full_engine_tensor_objects":len(full),
         "session_tensor_objects":0,"root_names":list(roots)}

def logical_snapshot(root,torch):
 """Tensor logical bytes across original predictor lifecycle; ignores padding/IDs."""
 inventory=migration.tensor_inventory(root,torch);result=[]
 for item in inventory:
  tensor=item["tensor"];raw=tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
  result.append({"paths":item["paths"],"dtype":str(tensor.dtype),"shape":list(tensor.shape),"stride":list(tensor.stride()),
    "storage_offset":tensor.storage_offset(),"requires_grad":tensor.requires_grad,"inference":tensor.is_inference(),
    "sha256":hashlib.sha256(raw).hexdigest()})
 return sorted(result,key=lambda row:row["paths"][0])

def active_session_guard(session):
 require(precision._session() is session,"exact original continuous Session required")
 require(session.mode==MODE and session.query_views is True and session.current is None,"quiescent declared FP16 session required")
 require(session.owner==get_ident(),"same owner thread required")
 require(session._flags()==session.initial_flags,"per-forward flags not restored")
 require(session.attention_cls.compute_qkv is session.qkv_private
   and inspect.getattr_static(session.attention_cls,"compute_attention_heads") is session.attention_private
   and inspect.getattr_static(session.inference_module.InferenceEngineCacheKV,"prepare") is session.private_prepare_descriptor,
   "active051 private alias identity changed")

class DiagnosticWindows:
 """Bounded instance observation state; never touch model/RNG/source aliases."""
 def __init__(self,session):
  self.session=session;self.active=None;self.windows=[];self.initial_counters=dict(session.counters)
  self.expected_counters=dict(session.counters)
 def begin(self,stage,acquire,cold):
  require(self.active is None and self.session.current is None,"close previous diagnostic window first")
  require(dict(self.session.counters)==self.expected_counters,"unaccounted051 events between observation windows")
  # This is explicitly diagnostics-only instance state, not frozen051 code.
  self.session.hist.clear();self.session.examples.clear()
  self.active={"stage_index":stage,"logical_acquire":acquire,"cold":bool(cold),"counter_start":dict(self.session.counters)}
 def finish(self,*,error=None):
  if self.active is None:return None
  row=self.active;self.active=None;report=self.session.report()
  delta={key:report["counters"].get(key,0)-row["counter_start"].get(key,0) for key in set(report["counters"])|set(row["counter_start"])}
  row={**row,"counter_end":dict(report["counters"]),"counter_delta":delta,
       "precision_window":report,"session_still_active":True,"original_aliases_restored_here":False,
       "error":error,"window_scope":"hist/examples since acquisition; counters are whole-session cumulative"}
  self.windows.append(row);self.expected_counters=dict(report["counters"])
  require(all(type(v) is int and v>=0 for v in delta.values()),"cumulative051 counters regressed")
  require(delta.get("witness_histogram_overflow",0)==0,"bounded witness histogram overflow")
  require(sum(w["count"] for w in report["witnesses"])==delta.get("qkv_calls",0),"window/cumulative QKV accounting differs")
  if error is None:require(delta.get("qkv_calls",0)==delta.get("attention_calls",0),"successful window QKV/attention accounting differs")
  require(len(report["witnesses"])<=512 and len(report["witness_examples"])<=16,"bounded window record limits violated")
  require(report["aliases_and_flags_restored"] is False,"active Session must not claim original alias restoration")
  prefill=[w for w in report["witnesses"] if w["phase"]=="prefill"]
  if not row["cold"]:require(not prefill,"warm hit unexpectedly performed prefill")
  return row


 def verify_complete(self):
  require(self.active is None,"diagnostic window still active")
  final=dict(self.session.counters);require(final==self.expected_counters,"unaccounted051 events after final window")
  keys=set(final)|set(self.initial_counters)
  totals={k:sum(w["counter_delta"].get(k,0) for w in self.windows) for k in keys}
  require(all(self.initial_counters.get(k,0)+totals[k]==final.get(k,0) for k in keys),"whole-session051 counter partition differs")
  return {"passed":True,"initial_counters":self.initial_counters,"final_cumulative_counters":final,
    "sum_window_deltas":totals,"windows":len(self.windows),
    "final051_report_witnesses_scope":"last acquisition only; cumulative counters cover all windows"}

class FP16BankOwner:
 """One owner of exact051 stages, independent of the old FP32 HostStageCache."""
 def __init__(self,checkpoint,*,cdf_descriptor,precision_session,device="cuda",verify_copy_hashes=False,
              expected_all_state_bytes=56*GIB):
  require_sources();active_session_guard(precision_session)
  require(not precision_session.counters and not precision_session.hist and not precision_session.examples,
          "new bank requires unused continuous051 observation state")
  from .inference import CHECKPOINT_SHA256
  require(sha256_file(checkpoint)==CHECKPOINT_SHA256,"immutable checkpoint required")
  require(device in ("cuda","cuda:0"),"one CUDA device required")
  self._initialize(checkpoint,cdf_descriptor,precision_session,precision_session.torch,device,
      verify_copy_hashes,expected_all_state_bytes,production=True)
  self.checkpoint_sha256=CHECKPOINT_SHA256
  global _OWNER
  with _OWNER_LOCK:
   require(_OWNER is None or _OWNER() is None or _OWNER().closed,"another FP16 bank owner is active")
   _OWNER=weakref.ref(self)

 def _initialize(self,checkpoint,cdf,session,torch,device,hashes,expected,*,production,memory_probe=None):
  require(type(expected) is int and expected>0 and type(hashes) is bool,"explicit positive admission and hash policy required")
  self.checkpoint=str(Path(checkpoint).resolve());self.cdf_descriptor=cdf;self.session=session;self.torch=torch
  self.device=device;self.verify_copy_hashes=hashes;self.production=production;self.checkpoint_sha256="cpu-test-only"
  self.owner_pid=os.getpid();self.owner_thread=get_ident();self._busy=Lock();self.closed=False;self.failed=False;self.failure=None
  self.entries={};self.active=None;self.context=None;self.last_prepared=None;self.counters=defaultdict(int)
  self.copy_events=[];self.coverage_events=[];self.windows=DiagnosticWindows(session);self.request_events=[];self.training_binding=None
  self.memory_probe=memory_probe or migration.process_memory
  self.initial_admission=migration.memory_admission(self.memory_probe(),expected,100*GIB,8*GIB)
  require(self.initial_admission["passed"],"initial fixed100GiB+8GiB memory admission failed")

 @classmethod
 def _for_cpu_tests(cls,session,torch,*,factory,memory_probe):
  """Explicit nonproduction harness; public production constructor has no factory injection."""
  value=cls.__new__(cls)
  value._initialize("cpu-test-only",{},session,torch,"cpu",True,1,production=False,memory_probe=memory_probe)
  value.test_factory=factory;value.memory_probe=memory_probe
  return value

 def _guard(self):
  require(not self.closed and not self.failed,"closed/failed bank cannot be reused")
  require(os.getpid()==self.owner_pid and get_ident()==self.owner_thread,"same process/thread required")
  if self.production:
   require_sources();active_session_guard(self.session)
   from tabpfn import regressor
   precision.cache._require_active_cdf(regressor,self.cdf_descriptor)

 def _memory(self):return self.memory_probe()
 def _sync(self):
  if self.device.startswith("cuda"):self.torch.cuda.synchronize()
 def _key(self,theta,obs):
  arrays=[np.ascontiguousarray(value,dtype=np.float32) for value in (theta,obs)]
  require(arrays[0].ndim==arrays[1].ndim==2 and arrays[0].shape[1]==21 and arrays[1].shape[1]==33
    and len(arrays[0])==len(arrays[1]) and all(np.isfinite(a).all() for a in arrays),"exact finite theta21/obs33 training arrays required")
  if self.production:require(len(arrays[0])==4096,"fixed4096 bank required")
  from .science.analytical_benchmark_v2 import build_problem
  problem=build_problem()
  digest=hashlib.sha256()
  for a in arrays:digest.update(str(a.shape).encode());digest.update(a.tobytes())
  digest.update(json.dumps({"checkpoint":self.checkpoint_sha256,"mode":MODE,"query_batch_size":1024,
    "device":self.device,"precision_source":PRECISION_SHA,"bank_source":execution_source_sha256(),"parameter_order":list(problem.parameter_order),"scientific_fingerprint":problem.scientific_fingerprint},sort_keys=True).encode())
  return digest.hexdigest()

 @property
 def stage_index(self):return None if self.active is None else self.entries[self.active]["engine"].stage_index
 @property
 def regressor(self):return None if self.active is None else self.entries[self.active]["engine"].regressor
 @property
 def query_batch_size(self):return 1024

 def _coverage(self,engine):
  start=time.perf_counter()
  result=(verify_root_coverage(engine,self.torch) if self.production else
    {"passed":True,"projected_tensor_objects":len(tensor_ids(stage_roots(engine),self.torch)),"cpu_test_only":True})
  result["inventory_wall_seconds"]=time.perf_counter()-start
  return result

 def _other_storage_guard(self,roots):
  def storage_ranges(root):
   result=set()
   for row in migration.tensor_inventory(root,self.torch):
    t=row["tensor"];storage=t.untyped_storage()
    if storage.nbytes():result.add((str(t.device),storage.data_ptr(),storage.data_ptr()+storage.nbytes()))
   return result
  mine_ids=tensor_ids(roots,self.torch);mine_ranges=storage_ranges(roots)
  for entry in self.entries.values():
   require(not mine_ids.intersection(tensor_ids(entry["roots"],self.torch)),"cross-stage Tensor sharing unsupported")
   other=sorted(storage_ranges(entry["roots"]));mine=sorted(mine_ranges);i=j=0
   while i<len(mine) and j<len(other):
    a,b=mine[i],other[j]
    require(not (a[0]==b[0] and a[1]<b[2] and b[1]<a[2]),"cross-stage overlapping storage sharing unsupported")
    if a[0]<b[0] or (a[0]==b[0] and a[2]<=b[1]):i+=1
    else:j+=1

 def _release_all(self):
  errors=[]
  for entry in list(self.entries.values()):
   engine=entry.pop("engine",None)
   try:
    if engine is not None:engine.close()
   except BaseException as error:errors.append(type(error).__name__+": "+str(error))
   finally:
    entry["roots"].clear()
    del engine
  self.entries.clear();self.active=None;self.last_prepared=None;gc.collect()
  if errors:raise BankError("release errors: "+"; ".join(errors))

 def _abort(self,error):
  self.failed=True;self.failure=type(error).__name__+": "+str(error)
  try:self.windows.finish(error=self.failure)
  finally:self._release_all()

 def _offload(self):
  if self.active is None:return
  entry=self.entries[self.active];roots=stage_roots(entry["engine"]);entry["roots"]=roots
  self.coverage_events.append(self._coverage(entry["engine"]))
  state=migration.state_manifest(roots,self.torch)
  admission=migration.memory_admission(self._memory(),state["cuda_storage_bytes"],100*GIB,8*GIB)
  require(admission["passed"],"measured pre-offload100GiB+8GiB admission failed")
  self._sync()
  report=migration.move_tensor_state(roots,self.torch,target="cpu",verify_bytes=self.verify_copy_hashes,
      force_copy=not self.production)
  self._sync()
  require(report["after"]["cuda_storage_bytes"]==0,"stage roots retain CUDA storage after offload")
  entry["original_devices"]=report["original_devices"];entry["cpu_state"]=report["after"]
  self.copy_events.append({"operation":"offload","stage_index":self.stage_index,"admission":admission,**report})
  self.counters["offload_count"]+=1;self.counters["offload_bytes"]+=report["copied_bytes"];self.active=None
  self.windows.finish();gc.collect()
  require(migration.memory_admission(self._memory(),0,100*GIB,8*GIB)["passed"],"measured post-offload memory admission failed")

 def prepare(self,theta,obs,stage):
  require(type(stage) is int and 0<=stage<21,"stage0..20 required")
  require(self._busy.acquire(blocking=False),"bank operation already active")
  local=None
  try:
   self._guard();self._offload();self.counters["logical_acquires"]+=1
   key=self._key(theta,obs)
   require(self.context in (None,key),"training bank/context changed; explicit new bank required")
   self.context=key
   from .science.analytical_benchmark_v2 import array_descriptor
   self.training_binding={"context_key":key,"theta_float32":array_descriptor(np.ascontiguousarray(theta,dtype=np.float32)),
      "obs_float32":array_descriptor(np.ascontiguousarray(obs,dtype=np.float32)),"query_excluded_from_training_key":True}
   index=(key,stage);cold=index not in self.entries
   self.windows.begin(stage,self.counters["logical_acquires"],cold)
   if self.production:require(sha256_file(self.checkpoint)==self.checkpoint_sha256,"checkpoint changed during bank lifetime")
   if not cold:
    entry=self.entries[index];current=migration.state_manifest(entry["roots"],self.torch,hash_bytes=self.verify_copy_hashes)
    require(not current["cuda_storage_bytes"] and migration.compare_state(entry["cpu_state"],current,require_hashes=self.verify_copy_hashes)["passed"],
            "retained CPU stage changed")
    require(migration.memory_admission(self._memory(),0,100*GIB,8*GIB)["passed"],"pre-reload memory admission failed")
    self._sync();report=migration.move_tensor_state(entry["roots"],self.torch,restore_devices=entry["original_devices"],
      verify_bytes=self.verify_copy_hashes,force_copy=not self.production);self._sync()
    self.active=index;self.copy_events.append({"operation":"reload","stage_index":stage,**report})
    self.counters["hits"]+=1;self.counters["reload_count"]+=1;self.counters["reload_bytes"]+=report["copied_bytes"]
    prepared={**entry["prepared"],"fit_stock_cache_seconds":0.,"resident_move_seconds":0.,"uncached_fit_seconds":0.,
      "bank_hit":True,"actual_fit_count":0,"actual_preprocess_refit_count":0,
      "cold_prepared_provenance":entry["prepared"],"continuous_session_identity":id(self.session)}
   else:
    require(migration.memory_admission(self._memory(),0,100*GIB,8*GIB)["passed"],"pre-fit memory admission failed")
    local=(precision.PrecisionStageExecutor(self.checkpoint,cdf_descriptor=self.cdf_descriptor,device=self.device,query_batch_size=1024)
      if self.production else self.test_factory(stage))
    if self.production:require(type(local) is _FACTORY,"exact original051 factory instance required")
    prepared=local.prepare(theta,obs,stage);roots=stage_roots(local)
    self.coverage_events.append(self._coverage(local));self._other_storage_guard(roots)
    if self.production:
     require(local.session is self.session,"new stage did not retain the continuous051 Session")
     require(precision.prefill_witness_passed(self.session.report(),stage)["passed"],"actual coldFP16/F32 fallback prefill not witnessed")
    self.entries[index]={"engine":local,"roots":roots,"prepared":copy.deepcopy(prepared)};local=None;self.active=index
    self.counters["misses"]+=1
    prepared={**prepared,"bank_hit":False,"actual_fit_count":1,"actual_preprocess_refit_count":1,
      "continuous_session_identity":id(self.session)}
   self.last_prepared=prepared;return prepared
  except BaseException as error:
   try:
    if local is not None:local.close()
   finally:self._abort(error)
   raise
  finally:self._busy.release()

 def predict_component(self,member,query,*,capture_member=False,with_log_prob=False):
  require(self._busy.acquire(blocking=False),"bank operation already active")
  try:
   self._guard();require(self.active is not None,"active GPU stage required")
   engine=self.entries[self.active]["engine"];before=dict(self.session.hist);cdf_before=self.cdf_descriptor.get("patched_call_count",0)
   value=engine.predict_component(member,query,capture_member=capture_member,with_log_prob=with_log_prob)
   if self.production:
    clean.query_witness_delta(before,dict(self.session.hist),stage=self.stage_index,member=member)
    require(self.cdf_descriptor["patched_call_count"]-cdf_before==1,"actual corrected-CDF head call differs")
    expected="original_fit_preprocessors" if self.stage_index==10 and member==5 else "resident_cache"
    require(value.get("hybrid_execution_path")==expected,"actual cached/FP32 fallback route differs")
   if self.production:
    torch=self.torch;logits=value["logits"];mass=value["component_raw_mass"];borders=value["criterion"].borders
    require(logits.dtype==torch.float32 and logits.device.type=="cpu"
       and bool((torch.isfinite(logits)|torch.isneginf(logits)).all()) and bool(torch.isfinite(logits).any(-1).all()),
       "invalid original FP32 postprocessing logits")
    require(mass.dtype==torch.float64 and bool(torch.isfinite(mass).all()) and bool((mass>0).all())
       and bool((mass<=1.00001).all()) and value["head_mass_envelope"]==1.00001,"original FP64 mass envelope changed")
    require(borders.dtype==torch.float32 and bool(torch.isfinite(borders).all()),"physical borders left finiteFP32")
   self.counters["physical_head_calls"]+=1
   self.counters[f"stage_{self.stage_index}_member_{member}_head_calls"]+=1
   self.counters[f"stage_{self.stage_index}_member_{member}_query_rows"]+=len(query)
   return value
  except BaseException as error:self._abort(error);raise
  finally:self._busy.release()

 def predict(self,query,*,output_type="full",capture_members=False):
  require(self.stage_index==0 and len(query)==1 and output_type=="full","only freshstage0 singleton aggregate allowed")
  # The original per-head scopes are taken by this owner's predict_component.
  values=precision.capture_members(self,query);helper=clean.load_script("characterize_fp16_all_stages.py",ALL21_SHA)
  self.counters["fresh_stage0_densities"]+=1
  return helper.assemble_shared_stage0(values,self.torch,self.device)

 def inspect_cache(self):
  started=time.perf_counter()
  self._guard();items=[];cuda=0
  for index,entry in self.entries.items():
   state=migration.state_manifest(entry["roots"],self.torch);active=index==self.active
   require(active or not state["cuda_storage_bytes"],"inactive stage owns CUDA tensors")
   cuda+=int(state["cuda_storage_bytes"]>0);items.append({"stage_index":index[1],"active":active,"state":state})
  require(cuda<=1,"multiple CUDA stages resident")
  return {"schema":SCHEMA+".whole_bank_inventory","scope":"new whole-bank inventory, not old compact-cache/GQA proof",
      "inventory_wall_seconds":time.perf_counter()-started,
      "policy":execution_metadata(),"production":self.production,"counters":dict(self.counters),"stages":items,
      "retained_stage_count":len(self.entries),"one_active_GPU_stage":True,"training_binding":self.training_binding,"memory":self._memory(),
      "global_cuda_allocated_bytes":self.torch.cuda.memory_allocated() if self.device.startswith("cuda") else 0,
      "active_stage_telemetry":None if self.active is None else self.entries[self.active]["engine"].inspect_cache()}

 def close(self):
  require(self._busy.acquire(blocking=False),"bank operation already active")
  try:self._guard();self._offload()
  except BaseException as error:self._abort(error);raise
  finally:self._busy.release()

 def shutdown(self):
  global _OWNER
  require(self._busy.acquire(blocking=False),"bank operation already active")
  errors=[]
  try:
   if self.closed:return
   try:
    self.windows.finish(error="shutdown" if self.failed else None)
    self.window_accounting=self.windows.verify_complete()
   except BaseException as error:errors.append(error)
   try:self._release_all()
   except BaseException as error:errors.append(error)
   finally:self.closed=True
   with _OWNER_LOCK:
    if _OWNER is not None and _OWNER() is self:_OWNER=None
   if errors:raise BankError("shutdown cleanup errors: "+"; ".join(str(e) for e in errors))
  finally:self._busy.release()

@contextmanager
def fp16_bank_scope(checkpoint,*,cdf_descriptor,verify_copy_hashes=False):
 """Caller owns CDF; release bank before continuous051 exits on every path."""
 with precision.PrecisionSession(MODE) as session:
  with ExitStack() as cleanup:
   owner=FP16BankOwner(checkpoint,cdf_descriptor=cdf_descriptor,precision_session=session,verify_copy_hashes=verify_copy_hashes)
   cleanup.callback(owner.shutdown)
   yield owner
