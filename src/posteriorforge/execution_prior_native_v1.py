"""Explicit all-FP32 correction: original all8 stages6/10, cached other19.

Built with PriorLabs-TabPFN. Coordinate/input/mask/sampler/decoder math and
frozen88/1732/cache/mixture dictionaries remain unchanged. No old native map
is inherited. A fresh complete21-stage numerical screen is required.
"""
from __future__ import annotations
from contextlib import contextmanager
import copy,gc,hashlib,json,pickle
from pathlib import Path
from threading import Lock
from types import MethodType,SimpleNamespace
import numpy as np
from . import execution_prior_coordinates_v1 as cached
from . import execution_cache as cache
from . import prior_coordinates_v1 as codec

SCHEMA="posteriorforge.execution.prior_native_control.v1"
ENGINE_ID="prior_latent_original_stage6_stage10_fp32_v1"
CACHED_SHA256="88f28c5a1f01c086661243c5d3bfb59631eb3c95fe3567b7f1ede325ddba0aea"
CODEC_SHA256="d8ce6c49e7b65c6ee55c59b7243d4ce943cbea28c92ebed0329798074aaa70c5"
FAILED_SCREEN_SHA256="550b91af4a031a3393f59146e1aa5360c75b0e716362ee727aeb1397af149593"
CONTROL_MODE=cached.CONTROL_MODE
MODE=CONTROL_MODE
NATIVE_STAGES=(6,10)
NATIVE_PHYSICAL_SLOTS=(5,9)
MEMBERS=tuple(range(8))
_CACHED_CLASS=cached.PriorCoordinateStageExecutor
_SESSION_CLASS=cached.PriorCoordinateSession
_BODY=cached._CLONES["uncached_predict_body"]
_BODY_CODE=_BODY.__code__
_MODEL_RNG=cache.ResidentStageExecutor._model_rng
_SEALS=[(owner,name,getattr(owner,name),getattr(owner,name).__code__) for owner,names in
 ((_CACHED_CLASS,("__init__","prepare","predict_component","predict","inspect_cache","close")),
  (_SESSION_CLASS,("__init__","__enter__","__exit__","forward_scope","strict_fallback"))) for name in names]

def require(value,message):
 if not value:raise RuntimeError(message)

def source_sha256():return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

def require_sources():
 require(cached.source_sha256()==CACHED_SHA256 and codec.source_sha256()==CODEC_SHA256,"frozen cached/codec source changed")
 cached.require_sources()
 require(cached._CLONES["uncached_predict_body"] is _BODY and _BODY.__code__ is _BODY_CODE,"original native prediction body changed")
 require(cache.ResidentStageExecutor._model_rng is _MODEL_RNG,"original model RNG context changed")
 for owner,name,function,code in _SEALS:
  require(getattr(owner,name) is function and function.__code__ is code,"frozen callable identity changed: "+name)
 require(tuple(codec.ORDER[j] for j in NATIVE_STAGES)==NATIVE_PHYSICAL_SLOTS,"new native-stage physical labels differ")
 return {"cached":CACHED_SHA256,"codec":CODEC_SHA256,"failed_empty_cache_screen":FAILED_SCREEN_SHA256}

def policy_metadata():
 require_sources()
 data={"native_stages":list(NATIVE_STAGES),"members":list(MEMBERS),
  "stage_bindings":[codec.coordinate_metadata(codec.ORDER[j]) for j in NATIVE_STAGES],
  "other_stages":[j for j in range(21) if j not in NATIVE_STAGES],
  "candidate_mode":CONTROL_MODE,"cached_source_sha256":CACHED_SHA256,
  "failed_empty_cache_screen_sha256":FAILED_SCREEN_SHA256}
 return {**data,"module_sha256":source_sha256(),"policy_sha256":hashlib.sha256(json.dumps(data,sort_keys=True,separators=(",",":")).encode()).hexdigest(),
  "native_stage_rule":"new explicit all8 originalFP32 ordered stages6/10 only",
  "old_sigma_native_policy_inherited":False,"empty_cache_screen_promoted_to_pass":False,
  "coordinate_policy":cached.policy_metadata(),"FP16_activated":False}

def execution_metadata():
 return {"schema":SCHEMA,"engine_id":ENGINE_ID,"module_sha256":source_sha256(),"policy":policy_metadata(),
  "mode":CONTROL_MODE,"original_native_fit_mode":"fit_preprocessors","native_members":8,
  "native_neural_prefill_calls":0,"native_KV_cache_used":False,
  "other19_cached_numerical_code":"EXACT frozen88 delegation",
  "native_query_math":"exact frozen1732 selected-original-member body; original attention; FP32",
  "sampler_decoder_or_coordinate_math_changed":False,"old_numeric_pass_inherited":False,
  "qualified":False,"attribution":"Built with PriorLabs-TabPFN"}

def PriorNativeSession(mode=CONTROL_MODE):
 require(mode==CONTROL_MODE,"this new policy is FP32 only; no FP16 candidate/comparison")
 require_sources()
 return _SESSION_CLASS(CONTROL_MODE)

def _session():
 session=cached._session()
 require(type(session) is _SESSION_CLASS and session.mode==CONTROL_MODE and session.fallback_map=={},"exact frozen88 CONTROL Session required")
 return session

@contextmanager
def original_native_query_scope(session,stage,member):
 require(stage in NATIVE_STAGES and type(member) is int and member in MEMBERS,"declared all8 native stage/member required")
 require(session.query_views is True,"cached query-view setting unexpectedly changed before native query")
 try:
  session.query_views=False
  with session.forward_scope("query",stage,member):yield
 finally:
  session.query_views=True

class _NativeStage:
 def __init__(self,checkpoint,*,cdf_descriptor,query_batch_size=1024,device="cuda"):
  require_sources();require(query_batch_size==1024,"exact qualified cap1024 required")
  self.session=_session();self.checkpoint=checkpoint;self.cdf_descriptor=cdf_descriptor
  self.query_batch_size=query_batch_size;self.device=device;self.stage_index=None;self.uncached_regressor=None
  self.base=SimpleNamespace(_torch=self.session.torch,device=device,metadata={"model_rng_isolation_count":0},
   _sdk_regressor=self.session.regressor_module,cdf_descriptor=cdf_descriptor)
  self.base._model_rng=MethodType(_MODEL_RNG,self.base);self._busy=Lock()
  self.hybrid_counters={"uncached_predict_count":0,"uncached_query_rows":0,
   "original_preprocessing_lists_restored":True,"original_translator_alias_restored":True,
   "native_query_view_setting_restored":True}
 @property
 def regressor(self):return self.uncached_regressor
 def prepare(self,theta,obs,stage):
  require_sources();require(stage in NATIVE_STAGES,"only declared stages use native executor")
  theta,obs=cached._numpy_training(theta,obs);self.close();self.stage_index=stage;self.session.stage=stage
  torch=self.base._torch
  from .inference import CHECKPOINT_SHA256,frozen_regressor_settings
  require(cache._sha(self.checkpoint)==CHECKPOINT_SHA256,"immutable checkpoint differs")
  settings=frozen_regressor_settings(self.checkpoint);settings["inference_precision"]=torch.float32
  regressor=self.session.regressor_module.TabPFNRegressor(**settings);self.uncached_regressor=regressor
  try:
   train=np.concatenate((obs,theta[:,:stage]),axis=1);target=theta[:,stage]
   with self.base._model_rng():regressor.fit(train,target)
   require(type(regressor.executor_) is self.session.inference_module.InferenceEngineCachePreprocessing,"exact original all8 preprocessing executor required")
   require(regressor.fit_mode=="fit_preprocessors" and not regressor.use_autocast_ and regressor.forced_inference_dtype_==torch.float32
    and regressor.model_.cache_trainset_representation is False,"original native FP32 execution required")
   require(all(len(getattr(regressor.executor_,name))==8 for name in cached.ss15.hybrid.PREPROCESSING_FIELDS),"all8 original preprocessing/configuration lists required")
   require(regressor.renormalized_criterion_.borders.dtype==torch.float32,"native latent criterion must remain FP32")
   self.prepared={"stage_index":stage,"route":"original_fit_preprocessors_all8","native_neural_prefill_calls":0,
    "native_KV_cache_used":False,"fit_mode":regressor.fit_mode,"autocast":False,"forced_dtype":str(regressor.forced_inference_dtype_),
    "target_codec_coordinate":codec.coordinate_metadata(codec.ORDER[stage]),
    "configuration_identities":[cache._configuration_identity(c) for c in regressor.executor_.ensemble_configs],
    "preprocessor_sha256":[hashlib.sha256(pickle.dumps(p,protocol=5)).hexdigest() for p in regressor.executor_.preprocessors],
    "original_all8_lists_verified":True}
   return self.prepared
  except BaseException:self.close();raise
 def predict_component(self,member,query,*,capture_member=False,with_log_prob=False):
  require_sources();require(_session() is self.session and self.regressor is not None,"native session/model unavailable")
  require(not with_log_prob,"component is not a full-mixture log density")
  require(type(member) is int and member in MEMBERS,"original member0..7 required")
  if not self._busy.acquire(blocking=False):raise RuntimeError("native operation already active")
  try:
   self.hybrid_counters["native_query_view_setting_restored"]=False
   try:
    with original_native_query_scope(self.session,self.stage_index,member):
     result=_BODY(self,member,query,capture_member)
   finally:self.hybrid_counters["native_query_view_setting_restored"]=self.session.query_views is True
   require(result["logits"].dtype==self.base._torch.float32,"native postprocessing must remain FP32")
   result["prior_native_execution"]={"module_sha256":source_sha256(),"policy_sha256":policy_metadata()["policy_sha256"],
    "route":"original_fit_preprocessors_all8","stage":self.stage_index,"member":member,"target":codec.coordinate_metadata(codec.ORDER[self.stage_index]),
    "original_query_views_disabled":True,"query_view_setting_restored":True,"native_KV_cache_used":False}
   return result
  finally:self._busy.release()
 def inspect_cache(self):
  return {"schema":SCHEMA+".original_native_state","route":"original_fit_preprocessors_all8","KV_cache_used":False,
   "neural_prefill_calls":0,"stock_all8_preprocessing_fit":self.regressor is not None,"tensor_records":[],
   "tensor_inventory_scope":"none requested; NOT a cache-retention or byte-inventory claim"}
 def close(self):
  self.uncached_regressor=None;gc.collect()

class PriorNativeStageExecutor:
 def __init__(self,checkpoint,*,cdf_descriptor,query_batch_size=1024,device="cuda"):
  require_sources();require(query_batch_size==1024,"cap1024 required");_session()
  self.checkpoint=checkpoint;self.cdf_descriptor=cdf_descriptor;self.query_batch_size=query_batch_size;self.device=device
  self.child=None;self.stage_index=None
 def prepare(self,theta,obs,stage):
  require_sources();require(type(stage) is int and 0<=stage<21,"ordered stage0..20 required");self.close();self.stage_index=stage
  cls=_NativeStage if stage in NATIVE_STAGES else _CACHED_CLASS
  self.child=cls(self.checkpoint,cdf_descriptor=self.cdf_descriptor,query_batch_size=self.query_batch_size,device=self.device)
  return self.child.prepare(theta,obs,stage)
 @property
 def regressor(self):return self.child.regressor if self.child is not None else None
 @property
 def base(self):return self.child.base
 @property
 def metadata(self):
  return {"execution":execution_metadata(),"stage":self.stage_index,"route":route(self.stage_index),
   "child_metadata":self.child.metadata if self.child is not None and hasattr(self.child,"metadata") else getattr(self.child,"hybrid_counters",{})}
 def predict_component(self,member,query,*,capture_member=False,with_log_prob=False):
  require(self.child is not None,"prepare a stage first")
  result=self.child.predict_component(member,query,capture_member=capture_member,with_log_prob=with_log_prob)
  if self.stage_index not in NATIVE_STAGES:
   result["prior_native_execution"]={"module_sha256":source_sha256(),"policy_sha256":policy_metadata()["policy_sha256"],
    "route":"frozen88_resident_cache","stage":self.stage_index,"member":member,"target":codec.coordinate_metadata(codec.ORDER[self.stage_index])}
  return result
 def predict(self,*args,**kwargs):
  require(self.stage_index not in NATIVE_STAGES,"native full-mixture API not declared; reconstruct captured all8 components explicitly")
  return self.child.predict(*args,**kwargs)
 def inspect_cache(self):return self.child.inspect_cache()
 def close(self):
  child=self.child;self.child=None
  if child is not None:child.close()

def route(stage):return "original_fit_preprocessors_all8" if stage in NATIVE_STAGES else "frozen88_resident_cache"

def preparation_witness(report,stage,prepared):
 if stage not in NATIVE_STAGES:return cached.prefill_witness_passed(report,stage,mode=CONTROL_MODE)
 require(report["mode"]==CONTROL_MODE and report["module_sha256"]==CACHED_SHA256 and report["codec_sha256"]==CODEC_SHA256,"native original Session source differs")
 require(prepared["route"]=="original_fit_preprocessors_all8" and prepared["original_all8_lists_verified"]
  and prepared["native_neural_prefill_calls"]==0 and prepared["native_KV_cache_used"] is False,"native preparation cannot claim cached prefill")
 require(report["counters"].get("qkv_calls",0)==report["counters"].get("attention_calls",0)==0 and not report["witnesses"],"stock fit_preprocessors must not contain a neural prefill")
 return {"passed":True,"stage":stage,"route":route(stage),"native_neural_prefill_calls":0,"cached_prefill_claimed":False}

def query_witness_delta(before,after,stage,member):
 if stage not in NATIVE_STAGES:return cached.query_witness_delta(before,after,stage,member,mode=CONTROL_MODE)
 def hist(r):return {json.dumps({k:v for k,v in w.items() if k!="count"},sort_keys=True):w["count"] for w in r["witnesses"]}
 first=hist(before);last=hist(after);rows=[]
 for k,v in first.items():require(last.get(k,0)>=v,"native histogram decreased")
 for k,v in last.items():
  n=v-first.get(k,0)
  if n:
   event=json.loads(k);require((event["phase"],event["stage"],event["member"])==("query",stage,member),"native unrelated query event")
   rows.append({**event,"count":n})
 count=after["counters"].get("qkv_calls",0)-before["counters"].get("qkv_calls",0)
 require(count==after["counters"].get("attention_calls",0)-before["counters"].get("attention_calls",0)==sum(w["count"] for w in rows)>0,"native actual query count differs")
 require(after["counters"].get("expanded_views",0)==before["counters"].get("expanded_views",0),"original native attention cannot use expanded query views")
 delta={**after,"witnesses":rows,"counters":{"qkv_calls":count,"attention_calls":count,"witness_histogram_overflow":0}}
 cached._phase_rows(delta,stage,"query",[member],CONTROL_MODE)
 return {"passed":True,"stage":stage,"member":member,"route":route(stage),"qkv_calls":count,"attention_calls":count,"original_query_views_disabled":True}

# Coordinate/mask functions are the exact immutable88 objects.
stage_policy=cached.stage_policy
mask_training=cached.mask_training
mask_query=cached.mask_query
training_bindings=cached.training_bindings
codec_module=cached.codec_module

def immutable_native_snapshot(engine):
 """Diagnostic immutable-value checks, excluding legitimate cold encoder state."""
 regressor=engine.regressor
 require(regressor is not None,"prepared original native regressor required")
 digest=hashlib.sha256();count=0
 for name,parameter in regressor.model_.named_parameters():
  require(str(parameter.dtype)=="torch.float32","original native parameter dtype differs")
  value=parameter.detach().cpu().contiguous().numpy()
  digest.update(name.encode()+str(value.shape).encode()+value.tobytes());count+=value.size
 borders=regressor.renormalized_criterion_.borders.detach().cpu().numpy()
 return {"weight_values_sha256":digest.hexdigest(),"parameter_count":int(count),
  "target_mean":float(regressor.y_train_mean_),"target_sd":float(regressor.y_train_std_),
  "criterion_borders":cached.array_descriptor(borders),
  "configuration_identities":[{k:v for k,v in cache._configuration_identity(c).items() if k in ("type","sha256")} for c in regressor.executor_.ensemble_configs],
  "preprocessor_sha256":[hashlib.sha256(pickle.dumps(p,protocol=5)).hexdigest() for p in regressor.executor_.preprocessors],
  "fitted_context_X":[cached.array_descriptor(value) for value in regressor.executor_.X_trains],
  "fitted_context_y":[cached.array_descriptor(value) for value in regressor.executor_.y_trains],
  "scope":"immutable weights/targets/criterion/configs/preprocessors/all8 fitted X/y; excludes legitimate cold-to-warm normalization/embedding initialization"}
