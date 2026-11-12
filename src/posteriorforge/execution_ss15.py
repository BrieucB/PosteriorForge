"""Explicit SS15 dimension adapter over frozen precision/cache computation.

Built with PriorLabs-TabPFN. Guarded private clones change observation-width
plumbing and validation only; original modules, class methods, private globals,
probability translation, component selection and sampler are never edited.
Transform original float64 raw33 banks/query with representation_ss15 BEFORE
calling this adapter. Original raw33 numerical qualifications do not cover SS15.
"""
from __future__ import annotations
import ast
import __future__
import copy
import hashlib
import inspect
import json
from pathlib import Path
from threading import Lock
import textwrap
import numpy as np
from . import execution_cache as cache,execution_hybrid as hybrid,execution_mixture as mixture
from . import execution_precision_probe as precision

SCHEMA="posteriorforge.execution.ss15.v1"
ENGINE_ID="ss15_precision_stage_v1"
OBSERVATION_FEATURES=15
QUERY_CAP=1024
REPRESENTATION_SHA256="46323c0b94301c2f7e678538881d5cbbdd8bc52c62a60021798176bffe81eb85"
CONTROL_MODE=precision.CONTROL_MODE
FP16_MODE="amp_fp16_prefill_query_v1"
MODES=(CONTROL_MODE,FP16_MODE)
PINS={"cache":"45cd3107c50dcf73d03167e2b6dd045cccdb798337e8d3c47b5210cc30dc1a03",
 "hybrid":"5dfdefa870489ded594de1db13a55a3da5948c720d7a9fd587ebe3a8f8be00a2",
 "mixture":"e71f43b73ae8259c20b5d7b52feccc7c851e9a2fbb86d869bc0a5f7f3af792a1",
 "precision":"051f477ff4b9ab8bbcfbaf3c1afa4dcf6b111d2f2fbcf9efb9c94a00d2b2ad3d"}
_ORIGINAL_METHODS=[(owner,name,getattr(owner,name),getattr(owner,name).__code__) for owner,names in (
 (cache.ResidentStageExecutor,("__init__","prepare","predict","close")),
 (hybrid.HybridStageExecutor,("prepare","_predict_uncached","predict_component","close")),
 (precision.PrecisionSession,("__enter__","__exit__","forward_scope")),
 (precision.PrecisionStageExecutor,("predict_component",)),
 (precision.UncachedPrecisionStage,("__init__","predict_component","close"))) for name in names]


class SS15Error(RuntimeError):pass


def source_sha256():return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def require_sources():
 for name,module in (("cache",cache),("hybrid",hybrid),("mixture",mixture),("precision",precision)):
  actual=module.source_sha256() if name=="precision" else module.execution_source_sha256()
  if actual!=PINS[name]:raise SS15Error("frozen source changed: "+name)
 precision.require_sources()
 for owner,name,function,code in _ORIGINAL_METHODS:
  if getattr(owner,name) is not function or function.__code__ is not code:raise SS15Error("original method identity changed: "+name)


def canonical_fallback_map(value=None):
 result=hybrid.canonical_fallback_map({10:[5]} if value is None else value)
 if 5 not in result.get(10,()):raise ValueError("required declared FP32 stage10/member5 fallback cannot be removed")
 return result


def representation_descriptor():
 from . import representation_ss15 as representation
 if representation.source_sha256()!=REPRESENTATION_SHA256:raise SS15Error("frozen exact SS15 representation source changed")
 return representation.representation_metadata()


def execution_metadata(mode=FP16_MODE,fallback_map=None):
 if mode not in MODES:raise ValueError("explicit SS15 transparentFP32 or FP16 mode required")
 require_sources();fallback=canonical_fallback_map(fallback_map)
 return {"schema":SCHEMA,"engine_id":ENGINE_ID,"module_sha256":source_sha256(),"representation":representation_descriptor(),
  "observation_features":15,"stage_count":21,"query_batch_size":1024,"mode":mode,"source_ancestry":dict(PINS),
  **hybrid.policy_metadata(fallback),"raw33_numerical_qualification_applies":False,"qualified":False,
  "representation_transform_inside_executor":False,"model_inputs":"already transformed contiguous SS15 float32 observations plus original physical prefix",
  "forward_precision":"FP32" if mode==CONTROL_MODE else "FP16 AMP except explicitly declared original FP32 fallback",
  "weights_normalizers_borders_postprocessing_dtype":"torch.float32","component_mass_dtype":"torch.float64",
  "component_selector_and_probability_math":"unchanged e71","stage0":"8 actual raw canonical FP32 probabilities, stack/mean/log on original prediction CUDA device",
  "sampler_changed":False,"tail_family_changed":False,"original_globals_mutated":False,"attribution":"Built with PriorLabs-TabPFN"}


def _numpy_training(theta,observations):
 theta=np.asarray(theta,dtype=np.float32);observations=np.asarray(observations,dtype=np.float32)
 if theta.shape!=(4096,21) or observations.shape!=(4096,15) or not np.isfinite(theta).all() or not np.isfinite(observations).all():
  raise ValueError("finite4096x21 theta and separately transformed4096x15 SS15 required")
 return theta,observations


def _numpy_query(query,stage,max_rows):
 if hasattr(query,"detach"):query=query.detach().cpu().numpy()
 value=np.asarray(query,dtype=np.float32)
 if type(stage) is not int or not 0<=stage<21 or value.ndim!=2 or value.shape[1]!=15+stage or not 1<=len(value)<=max_rows or np.isinf(value).any():
  raise ValueError("SS15 query needs15+stage columns, NaN passed unchanged, infinity rejected, explicit row cap")
 return value


def _session():
 session=precision._session()
 if not isinstance(session,SS15PrecisionSession):raise SS15Error("caller-owned explicit SS15PrecisionSession required")
 session.require_policy()
 return session


class SS15PrecisionSession(precision.PrecisionSession):
 """Original051 lifetime/AMP mechanics; new source-bound representation/policy."""
 def __init__(self,mode=FP16_MODE,*,fallback_map=None):
  if mode not in MODES:raise ValueError("SS15 mode must be transparentFP32 or FP16")
  require_sources();self.fallback_map=canonical_fallback_map(fallback_map);super().__init__(mode)
  self._declared_mode=mode;self._declared_fallback_map=copy.deepcopy(self.fallback_map)
  self.ss15_source=source_sha256();self.ss15_representation=representation_descriptor()
  self.descriptor={**self.descriptor,"schema":SCHEMA+".precision_session","module_sha256":self.ss15_source,
   "frozen_precision_module_sha256":PINS["precision"],"representation":self.ss15_representation,"observation_features":15,
   **hybrid.policy_metadata(self.fallback_map),"original_session_resident_prepare":"original raw33 clone created but UNUSED by SS15Resident"}
 def require_policy(self):
  if self.mode!=self._declared_mode or self.fallback_map!=self._declared_fallback_map:raise SS15Error("declared SS15 mode/fallback policy mutated; new explicit session/qualification required")
 def strict_fallback(self,stage,member):return member in self.fallback_map.get(stage,())
 def __enter__(self):
  self.require_policy()
  if source_sha256()!=self.ss15_source or representation_descriptor()!=self.ss15_representation:raise SS15Error("SS15 source/representation changed before entry")
  return super().__enter__()


def _hybrid_metadata(mode):
 session=_session()
 return {**execution_metadata(mode,session.fallback_map),"with_log_prob_supported":False,
  "fallback_prepared_members":8,"fallback_fit_mode":"fit_preprocessors"}


def _ast(text):
 tree=ast.parse(text)
 return tree.body[0].value if len(tree.body)==1 and isinstance(tree.body[0],ast.Expr) else tree.body[0]


def _dump(node):return ast.dump(node,include_attributes=False)


def _clone(function,rules,namespace,label):
 original=ast.parse(textwrap.dedent(inspect.getsource(function)));original.body[0].decorator_list=[]
 expected=copy.deepcopy(original);counts={name:0 for name,*_ in rules};reverse={}
 parsed=[(name,_dump(_ast(before)),_ast(after),count) for name,before,after,count in rules]
 class Change(ast.NodeTransformer):
  def generic_visit(self,node):
   node=super().generic_visit(node)
   for name,before,after,count in parsed:
    if _dump(node)==before:
     replacement=copy.deepcopy(after);reverse[id(replacement)]=copy.deepcopy(node);counts[name]+=1
     return ast.copy_location(replacement,node)
   return node
 changed=Change().visit(copy.deepcopy(original))
 if counts!={name:count for name,_,_,count in parsed}:raise SS15Error("limited SS15 AST edit count differs: "+label+" "+repr(counts))
 class Undo(ast.NodeTransformer):
  def visit(self,node):
   if id(node) in reverse:return copy.deepcopy(reverse[id(node)])
   return super().visit(node)
 # Undo must use the actual replaced-node identities, not a copied tree.
 restored=Undo().visit(changed)
 if _dump(restored)!=_dump(expected):raise SS15Error("SS15 AST reverse equality failed: "+label)
 # Reapply after the reverse audit to obtain the separately compiled new tree.
 reverse.clear();counts={name:0 for name,*_ in rules}
 changed=Change().visit(copy.deepcopy(original));ast.fix_missing_locations(changed)
 scope=dict(function.__globals__);scope.update(namespace);scope["__name__"]=__name__
 exec(compile(changed,"<posteriorforge-ss15:"+label+">","exec",flags=__future__.annotations.compiler_flag,dont_inherit=True),scope)
 clone=scope[function.__name__]
 descriptor={"original_module":function.__module__,"original_qualname":function.__qualname__,"change_counts":dict(counts),
  "original_AST_sha256":hashlib.sha256(_dump(expected).encode()).hexdigest(),"SS15_AST_sha256":hashlib.sha256(_dump(changed).encode()).hexdigest(),
  "reverse_AST_exact":True,"original_global_dictionary_modified":False,"compiled_source_label":clone.__code__.co_filename}
 return clone,descriptor


def build_clones():
 require_sources()
 assumption,assumption_descriptor=precision._assumption_clone()
 namespace={"_ss15_numpy_training":_numpy_training,"_ss15_numpy_query":_numpy_query,"_ss15_session":_session,
  "_ss15_inference_setting":precision._inference_setting,"_ss15_cache_assumptions":assumption,
  "_ss15_hybrid_metadata":_hybrid_metadata}
 width=("observation_width","33+stage_index","15+stage_index")
 funcs={};descriptors={}
 specs=[
 ("resident_prepare",cache.ResidentStageExecutor.prepare,[
  (*width,3),("training_validator","_numpy_training","_ss15_numpy_training",1),
  ("precision_setting",'settings["inference_precision"]=torch.float32','settings["inference_precision"]=_ss15_inference_setting()',1),
  ("precision_assumption","_cache_assumptions","_ss15_cache_assumptions",1)]),
 ("resident_predict",cache.ResidentStageExecutor.predict,[("query_validator","_numpy_query","_ss15_numpy_query",1)]),
 ("hybrid_prepare",hybrid.HybridStageExecutor.prepare,[(*width,2),("training_validator","cache._numpy_training","_ss15_numpy_training",1)]),
 ("uncached_predict_body",hybrid.HybridStageExecutor._predict_uncached,[("query_validator","cache._numpy_query","_ss15_numpy_query",1)]),
 ("uncached_init",precision.UncachedPrecisionStage.__init__,[(*width,4),
  ("strict_stage","stage_index==10","stage_index in _ss15_session().fallback_map",1),
  ("session","_session","_ss15_session",1),("policy_metadata","_hybrid_counter_metadata","_ss15_hybrid_metadata",1)]),
 ("uncached_component",precision.UncachedPrecisionStage.predict_component,[
  ("strict_member","self.stage_index==10 and member_index==5","_ss15_session().strict_fallback(self.stage_index,member_index)",1),
  ("query_body","hybrid.HybridStageExecutor._predict_uncached","_ss15_uncached_predict_body",1),("session","_session","_ss15_session",1)])]
 for label,function,rules in specs:
  funcs[label],descriptors[label]=_clone(function,rules,{**namespace,"_ss15_uncached_predict_body":funcs.get("uncached_predict_body")},label)
 descriptors["resident_prepare"]["unchanged_051_assumption_clone"]=assumption_descriptor
 return funcs,descriptors


_CLONES,_AST_DESCRIPTORS=build_clones()
_CLONE_SEALS={name:(function,function.__code__) for name,function in _CLONES.items()}


def clone_descriptors():return copy.deepcopy(_AST_DESCRIPTORS)


def _guard_ss15():
 require_sources();session=_session()
 if source_sha256()!=session.ss15_source or representation_descriptor()!=session.ss15_representation:raise SS15Error("SS15 source/representation changed during scope")
 for name,(function,code) in _CLONE_SEALS.items():
  if _CLONES[name] is not function or function.__code__ is not code:raise SS15Error("SS15 private clone identity changed: "+name)
 return session


class SS15Resident(precision.PrecisionResident):
 def __init__(self,*args,**kwargs):
  if kwargs.get("query_batch_size",1024)!=1024:raise ValueError("SS15 initial cap1024 required")
  session=_guard_ss15();super().__init__(*args,**kwargs)
  self.metadata.update(execution_metadata(session.mode,session.fallback_map))
  self.metadata["config"]={**dict(cache.CONFIG_INVARIANTS),"observation_features":15,"query_batch_size":1024,
    "inference_precision":"torch.float32" if session.mode==CONTROL_MODE else "autocast","representation":session.ss15_representation}
 def prepare(self,theta_context,obs_context,stage_index):
  session=_guard_ss15();session.stage=stage_index
  value=_CLONES["resident_prepare"](self,theta_context,obs_context,stage_index)
  self.metadata["actual_use_autocast"]=bool(self.regressor.use_autocast_)
  self.metadata["actual_forced_inference_dtype"]=str(self.regressor.forced_inference_dtype_)
  return value
 def predict(self,*args,**kwargs):
  _guard_ss15();return _CLONES["resident_predict"](self,*args,**kwargs)


class SS15StageExecutor(precision.PrecisionStageExecutor):
 def __init__(self,checkpoint,*,cdf_descriptor,device="cuda",query_batch_size=1024,fallback_map=None):
  session=_guard_ss15()
  if query_batch_size!=1024:raise ValueError("SS15 initial explicitly qualified cap1024 required")
  fallback=canonical_fallback_map(session.fallback_map if fallback_map is None else fallback_map)
  if fallback!=session.fallback_map:raise ValueError("engine/session declared fallback maps differ")
  self.session=session;self.fallback_map=fallback
  self.base=SS15Resident(checkpoint,cdf_descriptor=cdf_descriptor,device=device,query_batch_size=query_batch_size)
  self.cached_component=mixture.ComponentMixtureExecutor(self.base);self.uncached_regressor=None;self._busy=Lock();self._prepared=None
  self.hybrid_counters={**_hybrid_metadata(session.mode),"uncached_fit_count":0,"uncached_predict_count":0,"uncached_query_rows":0,
   "resident_component_predict_count":0,"original_preprocessing_lists_restored":True,"original_translator_alias_restored":True,"stages":[]}
 def prepare(self,*args,**kwargs):
  _guard_ss15();return _CLONES["hybrid_prepare"](self,*args,**kwargs)
 def _predict_uncached(self,*args,**kwargs):
  _guard_ss15();return _CLONES["uncached_predict_body"](self,*args,**kwargs)
 def predict_component(self,*args,**kwargs):
  _guard_ss15();value=super().predict_component(*args,**kwargs)
  value["ss15_execution"]={"module_sha256":source_sha256(),"observation_features":15,"mode":self.session.mode,
   "stage_index":self.stage_index,"member_index":args[0] if args else kwargs["member_index"],**hybrid.policy_metadata(self.fallback_map)}
  return value
 def predict(self,query,*,output_type="full",capture_members=False):
  _guard_ss15()
  if self.stage_index==0 and len(query)==1 and output_type=="full":
   components=precision.capture_members(self,query)
   from .performance_fp16_experiment import load_script,ALL21_SOURCE_SHA
   helper=load_script("characterize_fp16_all_stages.py",ALL21_SOURCE_SHA)
   result=helper.assemble_shared_stage0(components,self.base._torch,self.base.device)
   result["ss15_execution"]={**execution_metadata(self.session.mode,self.fallback_map),"shared_actual_heads":8,"original_full_API_identity_claimed":False}
   return result
  return super().predict(query,output_type=output_type,capture_members=capture_members)
 def inspect_cache(self):
  return {**super().inspect_cache(),"ss15_execution":execution_metadata(self.session.mode,self.fallback_map)}


class SS15UncachedStage(precision.UncachedPrecisionStage):
 def __init__(self,checkpoint,*,cdf_descriptor,stage_index,theta,obs):
  if type(stage_index) is not int or not 0<=stage_index<21:raise ValueError("stage index0..20 required")
  _guard_ss15();theta,obs=_numpy_training(theta,obs)
  _CLONES["uncached_init"](self,checkpoint,cdf_descriptor=cdf_descriptor,stage_index=stage_index,theta=theta,obs=obs)
 def predict_component(self,*args,**kwargs):
  _guard_ss15();return _CLONES["uncached_component"](self,*args,**kwargs)


capture_members=precision.capture_members
