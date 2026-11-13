#!/usr/bin/env python3
"""V2 same-pair original physical gates/plots. Built with PriorLabs-TabPFN.

A new v2 family:18 cached stages and all8 original FP32 at6/10/14.
Unchanged original21+32 criteria, fixed focus and actual saved-Z algebra.
No neural/GPU/world/reference integration or actual global refresh here.
NEW G3 verification uses actual saved Z. Frozen OLD baseline custody replays
its already-recorded private normal stream, without a new posterior update.
"""
from __future__ import annotations
import ast,copy,hashlib,importlib.util,inspect
from pathlib import Path
from types import FunctionType
V1_EVALUATOR_SHA256="f16e13bb251ff195ed1c7f23a2045c806f7c1bc027665671d2d58e6a98246f1d"
_path=Path(__file__).with_name("evaluate_prior_pair_v1.py")
if hashlib.sha256(_path.read_bytes()).hexdigest()!=V1_EVALUATOR_SHA256:raise ValueError("frozen v1 metric/plot ancestry changed")
_spec=importlib.util.spec_from_file_location("_v2_evaluation_frozen_v1",_path);_ANCESTOR=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_ANCESTOR)
for _key,_value in vars(_ANCESTOR).items():
 if not _key.startswith("__"):globals()[_key]=_value
for _key,_value in vars(_ANCESTOR).items():
 if isinstance(_value,FunctionType) and _value.__globals__ is vars(_ANCESTOR):
  _clone=FunctionType(_value.__code__,globals(),_value.__name__,_value.__defaults__,_value.__closure__)
  _clone.__kwdefaults__=_value.__kwdefaults__;globals()[_key]=_clone
SCHEMA="posteriorforge.prior_coordinate_pair_assessment.v2"
WRAPPER_SHA256='74cb52b451b1f99987445e73de7f27232acff6e8fb6de875d398e0002524ebc3'
LABELS=("OLD matched physical-target neural mixed FP16/FP32 + physical G3",
 "NEW v2 prior coordinates FP32\n18 cached; all8 original at6/10/14",
 "NEW same v2 parent + ONE SEPARATE physical G3","qualified same-case reference")

def _literal_clone(function,strings):
 original=ast.parse(inspect.getsource(function)).body[0];candidate=copy.deepcopy(original);counts={k:0 for k in strings}
 for node in ast.walk(candidate):
  if isinstance(node,ast.Constant) and type(node.value) is str and node.value in strings:
   old=node.value;node.value=strings[old];counts[old]+=1
 require(all(counts[k]==(2 if k=="pair-density-focus.png" else 1) for k in strings),"missing/undeclared v2 path/display literal")
 reverse=copy.deepcopy(candidate);back={v:k for k,v in strings.items()}
 for node in ast.walk(reverse):
  if isinstance(node,ast.Constant) and type(node.value) is str and node.value in back:node.value=back[node.value]
 require(ast.dump(reverse,include_attributes=False)==ast.dump(original,include_attributes=False),"v2 metric/plot reverse AST differs")
 private=dict(globals());exec(compile(ast.fix_missing_locations(ast.Module(body=[candidate],type_ignores=[])),__file__+"::v2-path-labels-only","exec"),private)
 f=private[function.__name__];result=FunctionType(f.__code__,globals(),function.__name__,function.__defaults__,function.__closure__)
 result.__kwdefaults__=function.__kwdefaults__;return result

authenticate_hybrid=_literal_clone(_ANCESTOR.authenticate_hybrid,{
 "scripts/refresh_prior_pair_global_block_v1.py":"scripts/refresh_prior_pair_v2_global_block.py",
 "src/posteriorforge/performance_prior_pair_v1.py":"src/posteriorforge/performance_prior_pair_v2.py",
 "scripts/qualify_prior_coordinate_pair_v1.py":"scripts/qualify_prior_coordinate_pair_v2.py"})

plots=_literal_clone(_ANCESTOR.plots,{
 ": FP32 / cached19 / original all8 at6+10; physical G3 separate":": FP32 / cached18 / original all8 at6+10+14; physical G3 separate",
 "Pair-qualified prior-coordinate cohorts: four predeclared physical coordinates":"V2 pair-qualified cohorts: four predeclared physical coordinates",
 "Pair-qualified prior-coordinate cohorts: all21 physical densities":"V2 pair-qualified cohorts: all21 physical densities",
 "Pair-qualified all21 physical marginal CDFs":"V2 all21 physical marginal CDFs",
 "Pair-qualified all32 fixed physical projection CDFs":"V2 all32 fixed physical projection CDFs",
 "pair-density-focus.png":"pair-v2-density-focus.png","pair-density.png":"pair-v2-density.png","pair-":"pair-v2-"})

# Metadata-only key rename scopes the zero to NEW-parent draws. OLD baseline
# deterministic stream verification is described separately above/in source pins.
# Verify the one-key source delta, then replace only that constant while keeping
# actual control-flow bytecode and every nested code object EXACTLY original.
_literal_clone(_ANCESTOR.run,{"new_global_or_normal_draws_during_assessment":"new_parent_global_or_normal_draws_during_assessment"})
_old_key="new_global_or_normal_draws_during_assessment"
_new_key="new_parent_global_or_normal_draws_during_assessment"
_key_edits=[0]
def _metadata_constants(values,reverse=False):
 result=[]
 for value in values:
  before,after=(_new_key,_old_key) if reverse else (_old_key,_new_key)
  if type(value) is str and value==before:
   value=after
   if not reverse:_key_edits[0]+=1
  elif type(value) is tuple:value=_metadata_constants(value,reverse)
  result.append(value)
 return tuple(result)
_old_run=_ANCESTOR.run
_RUN_CODE=_old_run.__code__.replace(co_consts=_metadata_constants(_old_run.__code__.co_consts))
require(_key_edits==[1] and _metadata_constants(_RUN_CODE.co_consts,True)==_old_run.__code__.co_consts
 and _RUN_CODE.co_code==_old_run.__code__.co_code,"v2 run metadata-only constant change differs")
run=FunctionType(_RUN_CODE,globals(),_old_run.__name__,_old_run.__defaults__,_old_run.__closure__)
run.__kwdefaults__=_old_run.__kwdefaults__
_AUTHENTICATE_CODE=authenticate_hybrid.__code__
_PLOTS_CODE=plots.__code__

def source_guard():
 w=load("refresh_prior_pair_v2_global_block.py",WRAPPER_SHA256)
 producer,pairs,pins=w.source_guard();root=Path(__file__).resolve().parents[1]
 w.BASE.verify_git_sources(root,pins["git_revision"],{"scripts/"+Path(__file__).name:sha256_file(__file__),
  "scripts/refresh_prior_pair_v2_global_block.py":WRAPPER_SHA256,"scripts/evaluate_prior_pair_v1.py":V1_EVALUATOR_SHA256})
 require(authenticate_hybrid.__code__ is _AUTHENTICATE_CODE and plots.__code__ is _PLOTS_CODE
  and run.__code__ is _RUN_CODE and run.__code__.co_code==_ANCESTOR.run.__code__.co_code
  and validate_reference_custody.__code__ is _ANCESTOR.validate_reference_custody.__code__
  and load_baseline.__code__ is _ANCESTOR.load_baseline.__code__,"v1 original gate/reference/run guards changed")
 return w,producer,pairs,{**pins,"pair_v2_evaluator":sha256_file(__file__),"pair_v2_G3_wrapper":WRAPPER_SHA256,
  "v1_metric_plot_ancestry":V1_EVALUATOR_SHA256,"metric_or_G3_math_changed":False,
  "assessment_RNG_scope":"NEW G3 uses saved actual Z only; matched OLD baseline verification deterministically replays its already-recorded PCG64 stream, with no new posterior/G3 update"}

if __name__=="__main__":main()
