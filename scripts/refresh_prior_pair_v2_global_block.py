#!/usr/bin/env python3
"""V2 pair-bound physical G3 caller. Built with PriorLabs-TabPFN.

Only all8 original FP32 stages6/10/14 plus cached18 v2 lineage is accepted.
Frozen v1 functions are reused in THIS new namespace; old globals/files remain
untouched. G3 arithmetic, seed roles, physical18 bytes and timing are unchanged.
"""
from __future__ import annotations
import ast,copy,importlib.util,inspect
from pathlib import Path
from types import FunctionType
import hashlib

V1_WRAPPER_SHA256="22e57084351631f36c6989cc01fe8bf4f2827095e587c496cda2e7f368d3cdb9"
_path=Path(__file__).with_name("refresh_prior_pair_global_block_v1.py")
if hashlib.sha256(_path.read_bytes()).hexdigest()!=V1_WRAPPER_SHA256:raise ValueError("frozen v1 validation ancestry changed")
_spec=importlib.util.spec_from_file_location("_v2_G3_frozen_v1",_path);_ANCESTOR=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_ANCESTOR)
for _key,_value in vars(_ANCESTOR).items():
 if not _key.startswith("__"):globals()[_key]=_value
# Functions defined by v1 are copied with their exact CODE OBJECT into NEW globals.
# Imported numerical functions/classes are retained unchanged.
for _key,_value in vars(_ANCESTOR).items():
 if isinstance(_value,FunctionType) and _value.__globals__ is vars(_ANCESTOR):
  _clone=FunctionType(_value.__code__,globals(),_value.__name__,_value.__defaults__,_value.__closure__)
  _clone.__kwdefaults__=_value.__kwdefaults__;globals()[_key]=_clone

SCHEMA="posteriorforge.prior_coordinate_pair_global_refresh.v2"
PRODUCER_SHA256="9260a7bdb3bbd3e593f8e0c754fdc8c3f0ca32d87978d14e3394209e1d33866d"
PRODUCER_SCHEMA="posteriorforge.inference.prior_coordinate_pair.v2"
PRODUCER_VARIANT="accuracy_prior_coordinates_pair_native_fp32_physical_f64_fifo_v2"
PAIR_QUALIFIER_SHA256="a9f4d6818c030ac29587642fa744101c49b9f4a5fd192ab224c0f273f770e875"
NATIVE_BACKEND_SHA256="0199b16e9cc34203fa0d5645d161742fe9ed244633d7beee047fbcbf2528661c"
PROTOCOL_SHA256="8640a4e157698862823521d3fd14396ecc3ebf121dffffabc987cc937d3ad2fc"
PROTOCOL_PATH="docs/accuracy-prior-coordinate-native-control-v2.md"
PAIR_DATA_PROTOCOL_SHA256="9fc6a91568864cc996dafbcfaa41e77280e4825c63f61d455bdadf3703cd0a66"
V1_CASE0_FAILED_SCREEN_SHA256="fbed99d9fb256bb72a49cb03bebb820d805ac2cbafaa4ebac4deb08bf3160734"
INELIGIBLE_V1_PROOFS=(OLD_CASE2_SCREEN_SHA256,V1_CASE0_FAILED_SCREEN_SHA256)

def _routing_clone(function,strings=None,native_list=False):
 """Only enumerated literals change; exact reverse AST must match v1."""
 original=ast.parse(inspect.getsource(function)).body[0];candidate=copy.deepcopy(original)
 strings={} if strings is None else dict(strings);counts={key:0 for key in strings};native_count=0
 for node in ast.walk(candidate):
  if isinstance(node,ast.Constant) and type(node.value) is str and node.value in strings:
   old=node.value;node.value=strings[old];counts[old]+=1
  elif native_list and isinstance(node,ast.List) and [getattr(x,"value",None) for x in node.elts]==[6,10]:
   node.elts.append(ast.Constant(14));native_count+=1
 require(all(n==1 for n in counts.values()) and native_count==(1 if native_list else 0),"undeclared/missing v2 routing literal delta")
 reverse=copy.deepcopy(candidate);reverse_strings={v:k for k,v in strings.items()}
 for node in ast.walk(reverse):
  if isinstance(node,ast.Constant) and type(node.value) is str and node.value in reverse_strings:node.value=reverse_strings[node.value]
  elif native_list and isinstance(node,ast.List) and [getattr(x,"value",None) for x in node.elts]==[6,10,14]:node.elts.pop()
 require(ast.dump(reverse,include_attributes=False)==ast.dump(original,include_attributes=False),"v2 routing reverse AST differs")
 private=dict(globals());exec(compile(ast.fix_missing_locations(ast.Module(body=[candidate],type_ignores=[])),__file__+"::declared-routing-only","exec"),private)
 generated=private[function.__name__]
 result=FunctionType(generated.__code__,globals(),function.__name__,function.__defaults__,function.__closure__)
 result.__kwdefaults__=function.__kwdefaults__
 return result

_HEADER_SCOPE_ANCESTRY="INNER frozen88 precision owner only; outer native all8 stages6/10 are explicit"
_HEADER_SCOPE_V2="INNER frozen88 precision owner only; outer native all8 stages6/10/14 are explicit"
_validate_v2_header=_routing_clone(_ANCESTOR.validate_header,{_HEADER_SCOPE_ANCESTRY:_HEADER_SCOPE_V2},native_list=True)
load_parent=_routing_clone(_ANCESTOR.load_parent,{"src/posteriorforge/performance_prior_pair_v1.py":"src/posteriorforge/performance_prior_pair_v2.py"})
_HEADER_CODE=_validate_v2_header.__code__
_LOAD_PARENT_CODE=load_parent.__code__

def validate_header(document,spec,proof_sha256):
 require(proof_sha256 not in INELIGIBLE_V1_PROOFS,"v1 proof cannot qualify v2")
 result=_validate_v2_header(document,spec,proof_sha256)
 require(document.get("numerical_candidate_changed") is True
  and document.get("execution_routing_changed_relative_v1") is True
  and document.get("fixed_case2_qualification_inherited") is False
  and document.get("native_protocol_sha256")==PROTOCOL_SHA256,
  "v2 is an explicit routing change, never unchanged v1 qualification")
 return result

_verify_v1_canonical_policy=verify_canonical_policy
def verify_canonical_policy(recorded,current):
 result=_verify_v1_canonical_policy(recorded,current)
 require(recorded.get("numerical_candidate_changed") is True
  and recorded.get("execution_routing_changed_relative_v1") is True
  and recorded.get("neural_weights_codec_sampler_support_math_changed") is False
  and recorded.get("inherited_v1_numerical_qualification") is False,
  "active v2 route/candidate change and no v1 qualification inheritance required")
 expected={"top_run_codeobject_identity_with_a30":False,"only_two_run_metadata_literals_changed":True,
  "original_run_controlflow_bytecode_equal":True,"original_run_names_varnames_freevars_equal":True,
  "all_nested_run_codeobjects_identical":True,"metadata_literals_reversible":True,
  "original_adapter_method_codeobjects_identical":True,"private_adapter_backend_source_policy_globals":True}
 require(all(recorded.get("derivation",{}).get(k) is v for k,v in expected.items()),
  "v2 code/source derivation must guard exact control flow and declared metadata-only literal changes")
 return {**result,"v2_routing_change_relative_v1_explicit":True,"v1_numerical_qualification_inherited":False}


def derivation_descriptor():
 return {"schema":"posteriorforge.prior_pair_v2_downstream_derivation.v1","v1_wrapper_sha256":V1_WRAPPER_SHA256,
  "unchanged_functions_same_code_objects":all(globals()[n].__code__ is getattr(_ANCESTOR,n).__code__ for n in
   ("execution_checks","refresh_with_timing","refresh","_safe_sibling"))
   and _verify_v1_canonical_policy.__code__ is _ANCESTOR.verify_canonical_policy.__code__,
  "AST_deltas_reverse_exact":["header native list6/10->6/10/14","header native scope text","recorded producer Git pathv1->v2"],
  "old_globals_unmodified":_ANCESTOR.SCHEMA=="posteriorforge.prior_coordinate_pair_global_refresh.v1"
   and _ANCESTOR.PRODUCER_SHA256=="52eaa638d09358765fcbb1d229265a5679f05a5fe0a5866f513412c7dc572a0b",
  "routing_function_code_identity":_validate_v2_header.__code__ is _HEADER_CODE and load_parent.__code__ is _LOAD_PARENT_CODE,
  "numerical_G3_seed_or_metric_math_changed":False,"serialized_code_fingerprint_used_as_identity":False}

def source_guard():
 require(all(type(x) is str and len(x)==64 for x in (PRODUCER_SHA256,PAIR_QUALIFIER_SHA256,NATIVE_BACKEND_SHA256)),
  "v2 producer/qualifier/backend unpinned; actual G3 disabled")
 from posteriorforge import performance_prior_pair_v2 as producer,prior_coordinate_pair_v1 as pairs
 require(producer.source_sha256()==PRODUCER_SHA256 and pairs.source_sha256()==PAIR_SOURCE_SHA256,"v2 producer/data source changed")
 producer.require_sources();pairs.require_sources()
 require(producer.backend.source_sha256()==NATIVE_BACKEND_SHA256 and tuple(producer.backend.NATIVE_STAGES)==(6,10,14),
  "only separately frozen v2 native6/10/14 route accepted")
 require(producer.PAIR_CPU_RECEIPT_SHA256==PAIR_CPU_RECEIPT_SHA256 and producer.physical_fifo.source_sha256()==F64_HELPER_SHA256,
  "unchanged pair CPU/F64 FIFO ancestry differs")
 require(kernel.source_sha256()==KERNEL_SHA256 and codec.source_sha256()==CODEC_SHA256,"unchanged scientific G3/codec differs")
 require(_RECORDING_GENERATOR is DRAW.RecordingGenerator and _RECORDING_GENERATOR.standard_normal.__code__ is _RECORDING_CODE
  and kernel.refresh is _KERNEL_REFRESH and _KERNEL_REFRESH.__code__ is _KERNEL_REFRESH_CODE,"original draw observer/G3 callable changed")
 derivation=derivation_descriptor()
 require(derivation["unchanged_functions_same_code_objects"] and derivation["old_globals_unmodified"] and derivation["routing_function_code_identity"],"v1 validation/G3 ancestry changed")
 root=Path(__file__).resolve().parents[1];require(sha256_file(root/PROTOCOL_PATH)==PROTOCOL_SHA256,"v2 protocol changed")
 revision=subprocess.run(["git","-C",str(root),"rev-parse","HEAD"],check=True,capture_output=True,text=True).stdout.strip()
 BASE.verify_git_sources(root,revision,{"scripts/"+Path(__file__).name:source_sha256(),
  "scripts/refresh_prior_pair_global_block_v1.py":V1_WRAPPER_SHA256,
  "src/posteriorforge/performance_prior_pair_v2.py":PRODUCER_SHA256,
  "src/posteriorforge/prior_coordinate_pair_v1.py":PAIR_SOURCE_SHA256,
  "src/posteriorforge/execution_prior_native_v2.py":NATIVE_BACKEND_SHA256,
  "scripts/qualify_prior_coordinate_pair_v2.py":PAIR_QUALIFIER_SHA256,
  "src/posteriorforge/conditional_globals.py":KERNEL_SHA256,PROTOCOL_PATH:PROTOCOL_SHA256})
 return producer,pairs,{"wrapper":source_sha256(),"pair_producer":PRODUCER_SHA256,"pair_data":PAIR_SOURCE_SHA256,
  "pair_qualifier":PAIR_QUALIFIER_SHA256,"native_backend":NATIVE_BACKEND_SHA256,
  "pair_CPU_receipt":PAIR_CPU_RECEIPT_SHA256,"kernel":KERNEL_SHA256,"codec":CODEC_SHA256,"F64_helper":F64_HELPER_SHA256,
  "protocol":PROTOCOL_SHA256,"pair_data_only_protocol_ancestry":PAIR_DATA_PROTOCOL_SHA256,
  "v1_validation_ancestry":V1_WRAPPER_SHA256,"guard_primitive_source":BASE_WRAPPER_SHA256,
  "actual_draw_observer_source":DRAW_SOURCE_SHA256,"derivation":derivation,"git_revision":revision}

if __name__=="__main__":main()
