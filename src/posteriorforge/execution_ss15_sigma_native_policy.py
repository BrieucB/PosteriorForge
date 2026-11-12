"""Explicit native-FP32 whole-stage execution correction for frozen sigma-first.

Built with PriorLabs-TabPFN. Frozen4099 method/helper code objects execute with
private fallback-policy globals. Frozen d48 order/mask methods are reused via
diamond MRO. Original source files, module dictionaries and SDK aliases are not
edited. Native stages are declared before inference, never selected from output.
"""
from __future__ import annotations
import copy
import hashlib
import json
import marshal
from pathlib import Path
from types import FunctionType,SimpleNamespace
from . import performance_ss15 as pilot,performance_ss15_sigma_first as producer
from . import execution_ss15 as execution,execution_ss15_sigma_first as order
from .artifacts import sha256_file

SCHEMA="posteriorforge.execution.ss15_sigma_native_policy.v1"
POLICY_ID="sigma_first_predetermined_failed_FP32_stage_native_all8_v1"
# Frozen before the final screen/cohort by unchanged-MAP2 strict numerical
# inventory: stage14 failed; all remaining local stages15..20 passed.
NATIVE_STAGES=(14,)
SELECTION_ARTIFACT_SHA256={
 "failed_MAP2_screen":"06bdcc69a189ba90ea121afef8eeb997e760a5c63221ae739859c0ac01094983",
 "remaining_MAP2_inventory":"504494557f6bb01362a3c939bf39c2a95be85e7c8c67d0959ce27d5feb4455d8"}
SOURCES={"adapter":"4099da3c19366997ab0800e5430b5bb1c129b4b3e535f678b3d4652233303480",
 "ordered_producer":"128c200a82f1252b2ad971ed43f10f534b16b9e02f8882cfd62d222cb88442cd",
 "order_mask":"d48d1638ef9e540bdd629d321fba657af4db757700b1092fc58fb9ded1f79ad2",
 "execution":"1732f7269286f1629f6e09b1b7ea14cfadbaea84cdae9ee4ab5221cf5b78fbbb"}
ANCESTRAL_MAP={10:[5],20:[4]}
_METHOD_NAMES=("__init__","prepare","predict_component","predict","inspect_cache","close","metadata")
_HELPER_NAMES=("is_fallback","prefill_witness_passed","query_witness_delta","validate_prediction","execution_metadata")
_ORIGINAL_CLASS=pilot.SS15StageAdapter
_ORIGINAL_METHODS={name:(getattr(_ORIGINAL_CLASS,name),getattr(_ORIGINAL_CLASS,name).__code__) for name in _METHOD_NAMES}
_ORIGINAL_HELPERS={name:(getattr(pilot,name),getattr(pilot,name).__code__) for name in _HELPER_NAMES}
_ORDER_CLASS=order.SigmaFirstStageAdapter
_ORDER_SEALS={name:(getattr(_ORDER_CLASS,name),getattr(_ORDER_CLASS,name).__code__)
 for name in ("__init__","prepare","predict_component","predict","inspect_cache","metadata")}
_VALIDATOR=(producer.validate_prediction,producer.validate_prediction.__code__)
_BUNDLES={}


def require(value,message):
 if not value:raise ValueError(message)


def source_sha256():return sha256_file(__file__)


def require_sources():
 require(pilot.source_sha256()==SOURCES["adapter"] and producer.source_sha256()==SOURCES["ordered_producer"]
  and order.source_sha256()==SOURCES["order_mask"] and execution.source_sha256()==SOURCES["execution"],
  "frozen sigma-first ancestry source changed")
 require(pilot.FALLBACK_MAP==ANCESTRAL_MAP and producer.FALLBACK_MAP==ANCESTRAL_MAP,
  "original MAP2 module globals changed")
 require(pilot.SS15StageAdapter is _ORIGINAL_CLASS and order.SigmaFirstStageAdapter is _ORDER_CLASS,
  "original adapter class identity changed")
 for name,(function,code) in _ORIGINAL_METHODS.items():
  require(getattr(_ORIGINAL_CLASS,name) is function and function.__code__ is code,"original method changed: "+name)
 for name,(function,code) in _ORIGINAL_HELPERS.items():
  require(getattr(pilot,name) is function and function.__code__ is code,"original helper changed: "+name)
 for name,(function,code) in _ORDER_SEALS.items():
  require(getattr(_ORDER_CLASS,name) is function and function.__code__ is code,"original order/mask method changed: "+name)
 require(producer.validate_prediction is _VALIDATOR[0] and producer.validate_prediction.__code__ is _VALIDATOR[1],
  "original sigma-first validator changed")
 pilot.require_sources();order.require_sources()


def canonical_native_stages(native_stages=None):
 selected=NATIVE_STAGES if native_stages is None else tuple(native_stages)
 require(selected and all(type(j) is int and 3<=j<21 for j in selected)
  and len(selected)==len(set(selected)) and 14 in selected,"declared distinct local native stages including14 required")
 return tuple(sorted(selected))


def fallback_map(native_stages=None):
 stages=canonical_native_stages(native_stages)
 result=copy.deepcopy(ANCESTRAL_MAP)
 for j in stages:result[j]=list(range(8))
 return {int(j):list(members) for j,members in execution.canonical_fallback_map(result).items()}


def policy_metadata(native_stages=None):
 stages=canonical_native_stages(native_stages);mapping=fallback_map(stages)
 physical_names=order._ORDER
 definition={"policy_id":POLICY_ID,"native_stage_indices":list(stages),
  "selection_artifact_sha256":dict(SELECTION_ARTIFACT_SHA256),
  "active_fallback_map":{str(j):members for j,members in mapping.items()},
  "active_fallback_stage_member_bindings":[{"stage":j,"member":m,"target_physical_index":order.ORDER[j],
    "target_physical_name":physical_names[order.ORDER[j]],"coordinate_system":"sigma-first stage index"}
    for j,members in mapping.items() for m in members],
  "whole_stage_all8_native_rule":True,"native_selection_source":"unchanged fixed-conditional FP32 member limits; never posterior metrics",
  "scientific_inputs_order_sampler_and_global_kernel_changed":False}
 digest=hashlib.sha256(json.dumps(definition,sort_keys=True,separators=(",",":")).encode()).hexdigest()
 return {"schema":SCHEMA,"module_sha256":source_sha256(),"policy_sha256":digest,**definition,
  "ancestral_MAP2":{str(j):list(members) for j,members in ANCESTRAL_MAP.items()},"ancestral_sources":dict(SOURCES),
  "order_mask_ancestry":order.policy_metadata(),
  "order_mask_ancestry_fallback_metadata_scope":"ancestral MAP2 only; superseded by active_fallback_map",
  "actual_numerical_qualification_claimed":False,"attribution":"Built with PriorLabs-TabPFN"}


def _clone(function,scope):
 require(function.__closure__ is None,"only audited closure-free4099/128c functions may be cloned")
 result=FunctionType(function.__code__,scope,function.__name__,function.__defaults__,function.__closure__)
 result.__kwdefaults__=copy.deepcopy(function.__kwdefaults__)
 result.__annotations__=dict(function.__annotations__)
 result.__module__=__name__;result.__doc__=function.__doc__
 return result


def _build_bundle(stages):
 require_sources()
 mapping=fallback_map(stages);original_namespace=dict(vars(pilot));scope=dict(original_namespace);scope["FALLBACK_MAP"]=copy.deepcopy(mapping)
 scope["source_sha256"]=source_sha256;scope["require_sources"]=require_sources
 clones={}
 for name in _HELPER_NAMES:
  clones[name]=_clone(getattr(pilot,name),scope)
  scope[name]=clones[name]
 original_execution_metadata=clones["execution_metadata"]

 def metadata():
  value=original_execution_metadata()
  return {**value,"schema":SCHEMA,"module_sha256":source_sha256(),"native_execution_policy":policy_metadata(stages),
   "ancestral_4099_policy_scope":"backend code ancestry; current fallback policy is active_fallback_map",
   "input_order_mask_proof_scope":"frozen d48 scientific inputs/order; its MAP2 fallback metadata is superseded"}
 scope["execution_metadata"]=metadata

 methods={name:_clone(getattr(_ORIGINAL_CLASS,name),scope) for name in _METHOD_NAMES}
 private_class=type("_PrivateNative4099StageAdapter",(_ORIGINAL_CLASS,),{"__module__":__name__,**methods})
 # d48's unchanged super() calls now resolve to PRIVATE native4099 bodies.
 def composite_metadata(self):
  value=_ORDER_CLASS.metadata(self)
  ancestry=value.pop("sigma_first_input_policy")
  return {**value,"ancestral_order_mask_policy":ancestry,
   "active_native_execution_policy":policy_metadata(stages),
   "order_mask_proof_precision_metadata_scope":"ancestral MAP2 only; current execution uses active native policy",
   "current_prediction_proof_module_sha256":source_sha256(),
   "input_order_mask_proof_module_sha256":SOURCES["order_mask"],
   "private_clone_descriptors":clone_descriptors(stages)}
 composite=type("SigmaNativeStageAdapter",(_ORDER_CLASS,private_class),
  {"__module__":__name__,"metadata":composite_metadata})
 original_validator_namespace=dict(vars(producer));validator_scope=dict(original_validator_namespace)
 proxy=SimpleNamespace(validate_prediction=clones["validate_prediction"])
 validator_scope["pilot"]=proxy
 validator=_clone(producer.validate_prediction,validator_scope)
 sealed_helpers={name:function for name,function in clones.items() if name!="execution_metadata"}
 original_map=copy.deepcopy(mapping)

 def guard():
  require_sources()
  require(scope["FALLBACK_MAP"]==original_map and scope["source_sha256"] is source_sha256
   and scope["require_sources"] is require_sources and scope["execution_metadata"] is metadata,
   "private native execution globals changed")
  changed={"FALLBACK_MAP","source_sha256","require_sources","execution_metadata",*_HELPER_NAMES}
  for name,value in original_namespace.items():
   if name not in changed:
    require(scope.get(name) is value and vars(pilot).get(name) is value,
     "private/original unmodified global alias changed: "+name)
  for name,value in original_validator_namespace.items():
   if name!="pilot":
    require(validator_scope.get(name) is value and vars(producer).get(name) is value,
     "private/original validator global alias changed: "+name)
  for name,function in sealed_helpers.items():
   require(scope[name] is function and function.__code__ is _ORIGINAL_HELPERS[name][1],
    "private helper identity/code changed: "+name)
  for name,function in methods.items():
   require(getattr(private_class,name) is function and function.__code__ is _ORIGINAL_METHODS[name][1],
    "private method identity/code changed: "+name)
  require(validator_scope["pilot"] is proxy and proxy.validate_prediction is clones["validate_prediction"]
   and validator.__code__ is _VALIDATOR[1],"private sigma validator proxy/code changed")
  require(composite.__mro__[:4]==(composite,_ORDER_CLASS,private_class,_ORIGINAL_CLASS)
   and composite.metadata is composite_metadata,"native/order diamond MRO or metadata changed")
  return True

 guard()
 return SimpleNamespace(adapter_class=composite,private_class=private_class,scope=scope,clones=clones,
  validator=validator,guard=guard,native_stages=stages)


def _bundle(native_stages=None):
 stages=canonical_native_stages(native_stages)
 if stages not in _BUNDLES:_BUNDLES[stages]=_build_bundle(stages)
 result=_BUNDLES[stages];result.guard();return result


def adapter_class(native_stages=None):return _bundle(native_stages).adapter_class


def validate_prediction(prediction,stage,member,*,report,native_stages=None):
 bundle=_bundle(native_stages)
 bundle.validator(prediction,stage,member,report=report)


def clone_descriptors(native_stages=None):
 bundle=_bundle(native_stages);rows=[]
 for name,function in bundle.clones.items():
  original=getattr(pilot,name)
  rows.append({"name":name,"original":"4099."+name,"code_sha256":hashlib.sha256(marshal.dumps(function.__code__)).hexdigest(),
   "same_code_object":function.__code__ is original.__code__,"private_global_dictionary":function.__globals__ is bundle.scope})
 for name in _METHOD_NAMES:
  function=getattr(bundle.private_class,name);original=getattr(_ORIGINAL_CLASS,name)
  rows.append({"name":"adapter."+name,"original":"4099.SS15StageAdapter."+name,
   "code_sha256":hashlib.sha256(marshal.dumps(function.__code__)).hexdigest(),
   "same_code_object":function.__code__ is original.__code__,"private_global_dictionary":function.__globals__ is bundle.scope})
 rows.append({"name":"ordered_validator","original":"128c.validate_prediction",
  "code_sha256":hashlib.sha256(marshal.dumps(bundle.validator.__code__)).hexdigest(),
  "same_code_object":bundle.validator.__code__ is _VALIDATOR[1],
  "private_global_dictionary":bundle.validator.__globals__ is not producer.__dict__})
 return rows
