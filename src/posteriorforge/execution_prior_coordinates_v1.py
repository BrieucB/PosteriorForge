"""Separate prior-latent SS15 execution policy over frozen numerical methods.

Built with PriorLabs-TabPFN. No old module/global dictionary is edited.
Inputs are already encoded, ordered prior-latent Float32 targets/prefixes.
There are NO inherited physical-stage native fallbacks and NO numerical pass
is inherited. Sampling and inverse coordinate decoding are outside this module.
"""
from __future__ import annotations
import copy,hashlib,importlib
from pathlib import Path
from types import FunctionType,SimpleNamespace
import numpy as np
from . import execution_ss15 as ss15
from . import execution_precision_probe as precision
from .science.analytical_benchmark_v2 import array_descriptor,build_problem

SCHEMA="posteriorforge.execution.prior_coordinates.v1"
ENGINE_ID="prior_latent_ss15_resident_v1"
SS15_SHA256="1732f7269286f1629f6e09b1b7ea14cfadbaea84cdae9ee4ab5221cf5b78fbbb"
CODEC_SHA256="d8ce6c49e7b65c6ee55c59b7243d4ce943cbea28c92ebed0329798074aaa70c5"
CONTROL_MODE=ss15.CONTROL_MODE
FP16_MODE=ss15.FP16_MODE
MODES=(CONTROL_MODE,FP16_MODE)
ORDER=(0,1,2,6,3,4,5,10,7,8,9,14,11,12,13,16,15,18,17,20,19)
BLOCKS=((3,4,5,6),(7,8,9,10),(11,12,13,14),(15,16),(17,18),(19,20))
OBSERVATIONS=((0,1,2,3),(4,5,6,7),(8,9,10,11),(12,),(13,),(14,))
_SEALED=[]

class PriorExecutionError(RuntimeError):pass

def source_sha256():return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

def require(value,message):
 if not value:raise PriorExecutionError(message)

def codec_module():
 require(isinstance(CODEC_SHA256,str) and len(CODEC_SHA256)==64,"codec source not frozen; execution disabled")
 module=importlib.import_module(".prior_coordinates_v1",__package__)
 require(module.source_sha256()==CODEC_SHA256,"frozen prior codec source changed")
 module.require_sources()
 require(tuple(module.ORDER)==ORDER,"codec/backend latent stage order differs")
 return module

def canonical_fallback_map(value=None):
 require(value is None or value=={},"initial prior-coordinate policy has NO native fallback; review any failure before new policy")
 return {}

def require_sources():
 require(ss15.source_sha256()==SS15_SHA256,"frozen numerical SS15 ancestry changed");ss15.require_sources();codec_module()
 for owner,name,function,code in _SEALED:
  require(getattr(owner,name) is function and function.__code__ is code,"source-qualified method alias changed: "+name)
 for label,(function,code) in _CLONE_SEALS.items():
  require(_CLONES[label] is function and function.__code__ is code,"private numerical clone changed: "+label)
 for label,(function,code) in globals().get("_METHOD_SEALS",{}).items():
  require(_METHODS[label] is function and function.__code__ is code,"private method dictionary changed: "+label)

def stage_policy(stage):
 require(type(stage) is int and 0<=stage<21,"latent stage0..20 required")
 target=ORDER[stage]
 if stage<3:obs=tuple(range(15));prefix=tuple(range(stage))
 else:
  block=next(i for i,indices in enumerate(BLOCKS) if target in indices);obs=OBSERVATIONS[block]
  prefix=tuple(k for k in range(stage) if k<3 or ORDER[k] in BLOCKS[block])
 keep=tuple(obs)+tuple(15+k for k in prefix);width=15+stage
 names=build_problem().parameter_order
 return {"stage_index":stage,"target_original_physical_slot":target,"target_coordinate":"prior_latent_"+names[target],
  "target_is_original_physical_value":False,"codec_coordinate":codec_module().coordinate_metadata(target),"observation_columns_kept":list(obs),
  "ordered_latent_prefix_indices_kept":list(prefix),"feature_columns_kept":list(keep),
  "feature_columns_zeroed":[k for k in range(width) if k not in keep],"input_width":width,
  "target_index_in_ordered_training":stage,"all_three_globals_condition_local_stages":stage>=3}

def policy_metadata():
 codec=codec_module();rules=[stage_policy(j) for j in range(21)]
 payload={"stage_to_original_physical_slot":list(ORDER),"stages":rules,"codec_sha256":CODEC_SHA256}
 import json
 return {**payload,"policy_sha256":hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":")).encode()).hexdigest(),
  "module_sha256":source_sha256(),"codec":codec.metadata(),"fallback_map":{},"native_stage_policy_inherited":False,
  "target_or_prefix_physical_label_identity_claimed":False}

def execution_metadata(mode=CONTROL_MODE,fallback_map=None):
 require(mode in MODES,"explicit cachedFP32 or FP16 comparison mode required");canonical_fallback_map(fallback_map);require_sources()
 return {"schema":SCHEMA,"engine_id":ENGINE_ID,"module_sha256":source_sha256(),"codec_sha256":CODEC_SHA256,
  "numerical_ancestry_ss15_sha256":SS15_SHA256,"observation_representation":ss15.representation_descriptor(),
  "policy":policy_metadata(),"mode":mode,"observation_features":15,"stage_count":21,"query_batch_size":1024,
  "fallback_map":{},"fallback_fit_mode":"none; no native stage inherited","fallback_prepared_members":0,
  "forward_precision":"FP32" if mode==CONTROL_MODE else "FP16 approximation; separately report all strict drift",
  "criterion_coordinate":"prior-latent target, NOT original physical parameter",
  "weights_normalizers_borders_postprocessing_dtype":"torch.float32","component_mass_dtype":"torch.float64",
  "component_selector_and_probability_math":"unchanged e71; raw-mass head rejection",
  "stage0":"8 actual latent rawP heads; original GPU stack/mean/log then CPU",
  "stage0_original_full_API_identity_claimed":False,"codec_decode_inside_executor":False,
  "old_numerical_qualification_inherited":False,"qualified":False,"posterior_equivalence_claimed":False,
  "original_global_dictionaries_modified":False,"attribution":"Built with PriorLabs-TabPFN"}

def _numpy_training(theta,obs):
 require(isinstance(theta,np.ndarray) and theta.dtype==np.float32 and theta.shape==(4096,21) and theta.flags.c_contiguous,
  "already ordered contiguous latent Float32[4096,21] required; no implicit cast")
 require(isinstance(obs,np.ndarray) and obs.dtype==np.float32 and obs.shape==(4096,15) and obs.flags.c_contiguous,
  "already transformed contiguous SS15 Float32[4096,15] required")
 require(np.isfinite(theta).all() and np.isfinite(obs).all(),"finite latent training labels/features required")
 return theta,obs

def _numpy_query(query,stage,max_rows):
 if hasattr(query,"detach"):query=query.detach().cpu().numpy()
 value=np.asarray(query)
 require(type(stage) is int and 0<=stage<21 and value.dtype==np.float32 and value.ndim==2 and value.shape[1]==15+stage and 1<=len(value)<=max_rows
  and np.isfinite(value).all(),"finite prior-latent query Float32[rows,15+stage] required; no implicit cast")
 return value

def mask_training(theta,obs,stage):
 _numpy_training(theta,obs);rule=stage_policy(stage)
 if stage<3:return theta,obs
 mt=theta.copy();ms=obs.copy()
 ms[:,[j for j in range(15) if j not in rule["observation_columns_kept"]]]=np.float32(0)
 mt[:,[j for j in range(stage) if j not in rule["ordered_latent_prefix_indices_kept"]]]=np.float32(0)
 return mt,ms

def mask_query(query,stage):
 value=_numpy_query(query,stage,1024);rule=stage_policy(stage)
 if stage<3:return value
 result=value.copy();result[:,rule["feature_columns_zeroed"]]=np.float32(0);return result

def training_bindings(theta,obs,stage):
 mt,ms=mask_training(theta,obs,stage);rule=stage_policy(stage)
 return {"module_sha256":source_sha256(),"codec_sha256":CODEC_SHA256,"policy_sha256":policy_metadata()["policy_sha256"],
  "stage":stage,"target_coordinate":rule["target_coordinate"],"mask":rule,
  "unmasked_training":array_descriptor(np.concatenate((obs,theta[:,:stage]),axis=1)),
  "masked_training":array_descriptor(np.concatenate((ms,mt[:,:stage]),axis=1)),
  "original_latent_target":array_descriptor(theta[:,stage]),"prepared_latent_target":array_descriptor(mt[:,stage]),
  "same_fixed_mask_for_training_query":True,"target_bytes_unchanged":theta[:,stage].tobytes()==mt[:,stage].tobytes()}

def _session():
 current=precision._session()
 require(type(current) is PriorCoordinateSession,"explicit prior-coordinate Session owner required")
 current.require_policy();return current

def _guard():
 require_sources();session=_session()
 require(session.backend_source==source_sha256() and session.codec_descriptor==codec_module().metadata(),"source/codec changed within session")
 return session

class PriorCoordinateSession(precision.PrecisionSession):
 def __init__(self,mode=CONTROL_MODE,*,fallback_map=None):
  require(mode in MODES,"explicit cachedFP32 or FP16 comparison mode required");canonical_fallback_map(fallback_map);require_sources()
  super().__init__(mode)
  self.fallback_map={};self._declared_mode=mode;self.backend_source=source_sha256();self.codec_descriptor=codec_module().metadata()
  # Names are plumbing expected by the unchanged1732 Resident body; semantics
  # and public identity are replaced in this separate policy owner's descriptor.
  self.ss15_source=self.backend_source;self.ss15_representation=ss15.representation_descriptor()
  self.descriptor={**self.descriptor,**execution_metadata(mode),"schema":SCHEMA+".precision_session"}
 def strict_fallback(self,stage,member):return False
 def require_policy(self):
  require(self.mode==self._declared_mode and self.fallback_map=={},"declared prior mode or empty-fallback policy mutated")
 def __enter__(self):
  self.require_policy();require_sources()
  require(self.backend_source==source_sha256(),"backend source changed before entry")
  return super().__enter__()

def _hybrid_metadata(mode):
 _session();return {**execution_metadata(mode),"with_log_prob_supported":False}

def _policy_only(_):
 return {"fallback_map":{},"fallback_policy_sha256":hashlib.sha256(b"{}").hexdigest(),
  "policy_scope":"NEW prior-coordinate cached-only policy; no inherited native stage"}

def _bind(function,namespace):
 globals_copy=dict(function.__globals__);globals_copy.update(namespace)
 clone=FunctionType(function.__code__,globals_copy,function.__name__,function.__defaults__,function.__closure__)
 clone.__kwdefaults__=copy.copy(function.__kwdefaults__);clone.__module__=__name__
 return clone

# The arithmetic code objects and their closure cells remain the frozen originals.
# Only lookups of validation/policy/metadata/classes in NEW dictionaries differ.
_CLONES={}
_proxy=SimpleNamespace(**{k:v for k,v in vars(ss15.hybrid).items() if not k.startswith("__")})
_proxy.policy_metadata=_policy_only
_namespace={"_ss15_session":_session,"_ss15_numpy_training":_numpy_training,"_ss15_numpy_query":_numpy_query,
 "_ss15_hybrid_metadata":_hybrid_metadata,"_session":_session,"_guard_ss15":_guard,
 "execution_metadata":execution_metadata,"source_sha256":source_sha256,"canonical_fallback_map":canonical_fallback_map,
 "_hybrid_metadata":_hybrid_metadata,"hybrid":_proxy,"_CLONES":_CLONES}
for name in ("resident_prepare","resident_predict","hybrid_prepare","uncached_predict_body"):
 _CLONES[name]=_bind(ss15._CLONES[name],_namespace)
_CLONE_SEALS={name:(f,f.__code__) for name,f in _CLONES.items()}

class PriorCoordinateResident(ss15.SS15Resident):pass
class PriorCoordinateStageExecutor(ss15.SS15StageExecutor):
 def predict_component(self,*args,**kwargs):
  result=_METHODS["stage_predict_component"](self,*args,**kwargs)
  result["prior_coordinate_execution"]=result.pop("ss15_execution")
  return result
 def predict(self,*args,**kwargs):
  result=_METHODS["stage_predict"](self,*args,**kwargs)
  if "ss15_execution" in result:result["prior_coordinate_execution"]=result.pop("ss15_execution")
  return result
 def inspect_cache(self):
  result=_METHODS["stage_inspect"](self)
  if "ss15_execution" in result:result["prior_coordinate_execution"]=result.pop("ss15_execution")
  return result

_namespace.update(SS15Resident=PriorCoordinateResident)
_METHODS={}
for label,owner,name,dest in (
 ("resident_init",ss15.SS15Resident,"__init__",PriorCoordinateResident),
 ("resident_prepare",ss15.SS15Resident,"prepare",PriorCoordinateResident),
 ("resident_predict",ss15.SS15Resident,"predict",PriorCoordinateResident),
 ("stage_init",ss15.SS15StageExecutor,"__init__",PriorCoordinateStageExecutor),
 ("stage_prepare",ss15.SS15StageExecutor,"prepare",PriorCoordinateStageExecutor),
 ("stage_uncached",ss15.SS15StageExecutor,"_predict_uncached",PriorCoordinateStageExecutor),
 ("stage_predict_component",ss15.SS15StageExecutor,"predict_component",None),
 ("stage_predict",ss15.SS15StageExecutor,"predict",None),
 ("stage_inspect",ss15.SS15StageExecutor,"inspect_cache",None)):
 original=getattr(owner,name);clone=_bind(original,_namespace);_METHODS[label]=clone
 _SEALED.append((owner,name,original,original.__code__))
 if dest is not None:setattr(dest,name,clone)
_METHOD_SEALS={name:(f,f.__code__) for name,f in _METHODS.items()}
for owner,names in ((PriorCoordinateSession,("__init__","__enter__","strict_fallback","require_policy")),
 (PriorCoordinateResident,("__init__","prepare","predict")),
 (PriorCoordinateStageExecutor,("__init__","prepare","predict_component","predict","inspect_cache"))):
 for name in names:
  f=getattr(owner,name);_SEALED.append((owner,name,f,f.__code__))

def clone_descriptors():
 require_sources()
 for name,(function,code) in _METHOD_SEALS.items():
  require(_METHODS[name] is function and function.__code__ is code,"private method clone changed: "+name)
 return {"numerical_private_functions":{name:{"code_object_is_frozen_1732":f.__code__ is ss15._CLONES[name].__code__,
  "original_global_dictionary_modified":False} for name,f in _CLONES.items()},
  "method_code_objects_reused":list(_METHODS),"SS15_width_AST_ancestry":ss15.clone_descriptors(),
  "globals_scope":"new dictionaries only; original1732/051/cache/mixture dictionaries unchanged",
  "target_identity":"new prior-latent coordinate labels and cache provenance; no raw physical label claim"}

def _phase_rows(report,stage,phase,members,mode):
 require(mode in MODES and report.get("mode")==mode and report.get("module_sha256")==source_sha256()
  and report.get("codec_sha256")==CODEC_SHA256 and report.get("fallback_map")=={},"wrong prior-coordinate precision report")
 require(report["counters"].get("witness_histogram_overflow",0)==0,"precision witness capacity exceeded")
 require(sum(w["count"] for w in report["witnesses"])==report["counters"].get("qkv_calls",0)
  ==report["counters"].get("attention_calls",0),"QKV/attention witness counts differ")
 result=[]
 for member in members:
  selected=[w for w in report["witnesses"] if w["stage"]==stage and w["member"]==member and w["phase"]==phase]
  require(selected,"actual phase/stage/member witness missing")
  for event in selected:
   dtype=[d for d in event["projection_dtypes"].values() if d is not None]
   strict=mode==CONTROL_MODE
   require(event["allow_tf32"] is False and event["matmul_precision"]=="highest"
    and event["autocast_enabled"] is (not strict) and event["strict_fp32_fallback"] is False,"actual precision/fallback flags differ")
   require(dtype and all(d in (("torch.float32",) if strict else ("torch.float32","torch.float16")) for d in dtype),"projection dtype outside declared family")
   if not strict:require("torch.float16" in dtype and event["autocast_dtype"]=="torch.float16","FP16 was not actually computed")
   storage=[d for d in event.get("supplied_cache_dtypes",[]) if d is not None]
   require(all(d in (("torch.float32",) if strict else ("torch.float32","torch.float16")) for d in storage),"cache storage dtype outside declared family")
  if phase=="prefill":require(any(w["cache_kv"] for w in selected),"computed cached-prefill witness missing")
  result.append({"member":member,"events":sum(w["count"] for w in selected),"passed":True})
 return result

def prefill_witness_passed(report,stage,*,mode=CONTROL_MODE):
 members=_phase_rows(report,stage,"prefill",range(8),mode)
 return {"passed":True,"stage":stage,"mode":mode,"members":members,"fallback_map":{},
  "scope":"actual computed prefill; active aliases are not claimed restored"}

def query_witness_delta(before,after,stage,member,*,mode=CONTROL_MODE):
 import json
 require(type(member) is int and 0<=member<8,"member0..7 required")
 # Report hashes/types apply to both endpoints, even when there were no queries.
 prefill_witness_passed(before,stage,mode=mode);prefill_witness_passed(after,stage,mode=mode)
 def hist(report):return {json.dumps({k:v for k,v in row.items() if k!="count"},sort_keys=True):row["count"] for row in report["witnesses"]}
 first=hist(before);last=hist(after);rows=[]
 for key,count in first.items():require(last.get(key,0)>=count,"precision histogram decreased or rotated mid-query")
 for key,count in last.items():
  difference=count-first.get(key,0)
  if difference:
   event=json.loads(key);require(event["phase"]=="query" and event["stage"]==stage and event["member"]==member,"unrelated query witness changed")
   rows.append({**event,"count":difference})
 delta_q=after["counters"].get("qkv_calls",0)-before["counters"].get("qkv_calls",0)
 delta_a=after["counters"].get("attention_calls",0)-before["counters"].get("attention_calls",0)
 require(delta_q==delta_a==sum(w["count"] for w in rows) and delta_q>0,"positive actual query QKV/attention delta required")
 delta={**after,"witnesses":rows,"counters":{"qkv_calls":delta_q,"attention_calls":delta_a,"witness_histogram_overflow":0}}
 _phase_rows(delta,stage,"query",[member],mode)
 return {"passed":True,"stage":stage,"member":member,"mode":mode,"qkv_calls":delta_q,"attention_calls":delta_a,
  "fallback_map":{},"scope":"same active session; aliases restored only on later context exit"}
