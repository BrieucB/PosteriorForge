#!/usr/bin/env python3
"""V3 same-pair original physical assessment; Built with PriorLabs-TabPFN.

The original50000-row/21marginal/32joint/CDF/mean/SD/reference gates are
unchanged. Only complete v3 physical parents and one separately authenticated
physical G3 are eligible. No model, GPU, new posterior or reference integration
runs here. NEW G3 validation uses its actual saved Z; OLD baseline custody
replays its already-recorded private stream, without another posterior/G3.
"""
from __future__ import annotations
import ast,copy,hashlib,importlib.util,inspect
from pathlib import Path
from types import FunctionType
from posteriorforge import validation as _validation

V1_EVALUATOR_SHA256="f16e13bb251ff195ed1c7f23a2045c806f7c1bc027665671d2d58e6a98246f1d"
V2_EVALUATOR_SHA256="61eb7b6c1546a0401cc25518adaba0a444ae2a09a83dd5a23745b8c337ceb1d7"
_path=Path(__file__).with_name("evaluate_prior_pair_v1.py")
if hashlib.sha256(_path.read_bytes()).hexdigest()!=V1_EVALUATOR_SHA256:raise ValueError("frozen original assessment math changed")
_v2_path=Path(__file__).with_name("evaluate_prior_pair_v2.py")
if hashlib.sha256(_v2_path.read_bytes()).hexdigest()!=V2_EVALUATOR_SHA256:raise ValueError("frozen v2 assessment ancestry changed")
_spec=importlib.util.spec_from_file_location("_v3_assessment_frozen_v1",_path);_ANCESTOR=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_ANCESTOR)
for _key,_value in vars(_ANCESTOR).items():
 if not _key.startswith("__"):globals()[_key]=_value
for _key,_value in vars(_ANCESTOR).items():
 if isinstance(_value,FunctionType) and _value.__globals__ is vars(_ANCESTOR):
  _clone=FunctionType(_value.__code__,globals(),_value.__name__,_value.__defaults__,_value.__closure__)
  _clone.__kwdefaults__=_value.__kwdefaults__;globals()[_key]=_clone

SCHEMA="posteriorforge.prior_coordinate_pair_assessment.v3"
WRAPPER_SHA256='0720eb75a10550f2486fd1d77f3a5ba4e20496c3812194c878bec5633fc60ff3'  # New v3 physical-parent wrapper frozen after CPU review.
LABELS=("OLD matched physical-target neural mixed FP16/FP32 + physical G3",
 "NEW v3 F32 forward / Policy C F64 probability\n18 cached; all8 original at6/10/14; physical F64",
 "NEW same v3 physical parent + ONE SEPARATE physical G3","qualified same-case reference")
METRIC_SOURCES={
 "src/posteriorforge/validation.py":"92ddfd79a1c42cdbc56dfc9ad0f2c987fa70ee69387b0c5decff53c20b9aa694",
 "scripts/evaluate_prior_coordinates_pilot_v1.py":"6c49e59ccee9bebeecd53990556a6e00a9f41b2fc53a2e8ecf9827014efc2d56",
 "scripts/evaluate_ss15_sigma_native_campaign.py":"146c2ca35e15b54b7f68134d383574485ed1cc228f38e6fc611624f00b411924",
 "scripts/evaluate_cdf_pilot.py":"284d3b061696d4d823666e12d3f79c720f87692b40f45743296cfb8e93865424",
 "scripts/evaluate_ss15_pilot.py":"3156d54b7d29f7aec29c2846782729a8c73da3bfcda16bb2e03cec3084899552",
 "scripts/evaluate_ss15_sigma_first_global_block.py":"80337f6e9b0014a5c4b0086548871b917dc4bec36fb373dc3a016b911bf02386",
}

def source_sha256():return sha256_file(__file__)
_RECORDED_GIT_FIELDS=(
 ('src/posteriorforge/execution_prior_native_v3.py','NATIVE_BACKEND_SHA256'),
 ('src/posteriorforge/execution_probability_v3.py','PROBABILITY_SHA256'),
 ('src/posteriorforge/prior_proposals_v3.py','PROPOSAL_SHA256'),
 ('src/posteriorforge/execution_prior_native_v2.py','NEURAL_SOURCE_SHA256'),
 ('src/posteriorforge/prior_coordinates_v1.py','CODEC_SHA256'),
 ('docs/accuracy-prior-coordinate-fp64-proposals-v3.md','PROTOCOL_SHA256'),
 ('scripts/refresh_prior_pair_global_block_v1.py','V1_WRAPPER_SHA256'),
)
def _literal_clone(function,strings,*,recorded_sources=False):
 """Enumerated source/display literals only; all numerical AST is reversible."""
 original=ast.parse(inspect.getsource(function)).body[0];candidate=copy.deepcopy(original);counts={k:0 for k in strings}
 for node in ast.walk(candidate):
  if isinstance(node,ast.Constant) and type(node.value) is str and node.value in strings:
   old=node.value;node.value=strings[old];counts[old]+=1
 require(all(counts[k]==(2 if k=="pair-density-focus.png" else 1) for k in strings),"missing/undeclared v3 source/display literal")
 if recorded_sources:
  calls=[n for n in ast.walk(candidate) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='verify_git_sources']
  require(len(calls)==1 and len(calls[0].args)==3 and isinstance(calls[0].args[2],ast.Dict),'one original recorded-Git source dictionary required')
  d=calls[0].args[2]
  for path,attribute in _RECORDED_GIT_FIELDS:
   d.keys.append(ast.Constant(path));d.values.append(ast.Attribute(value=ast.Name(id='w',ctx=ast.Load()),attr=attribute,ctx=ast.Load()))
 reverse=copy.deepcopy(candidate);back={v:k for k,v in strings.items()}
 if recorded_sources:
  calls=[n for n in ast.walk(reverse) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='verify_git_sources']
  d=calls[0].args[2];count=len(_RECORDED_GIT_FIELDS)
  for key,value,(path,attribute) in zip(d.keys[-count:],d.values[-count:],_RECORDED_GIT_FIELDS):
   require(isinstance(key,ast.Constant) and key.value==path and isinstance(value,ast.Attribute) and isinstance(value.value,ast.Name)
    and value.value.id=='w' and value.attr==attribute,'undeclared v3 recorded-source delta')
  del d.keys[-count:];del d.values[-count:]
 for node in ast.walk(reverse):
  if isinstance(node,ast.Constant) and type(node.value) is str and node.value in back:node.value=back[node.value]
 require(ast.dump(reverse,include_attributes=False)==ast.dump(original,include_attributes=False),"v3 metric/algebra reverse AST differs")
 private=dict(globals());exec(compile(ast.fix_missing_locations(ast.Module(body=[candidate],type_ignores=[])),__file__+"::v3-source-display-only","exec"),private)
 f=private[function.__name__];result=FunctionType(f.__code__,globals(),function.__name__,function.__defaults__,function.__closure__)
 result.__kwdefaults__=function.__kwdefaults__;return result

authenticate_hybrid=_literal_clone(_ANCESTOR.authenticate_hybrid,{
 "scripts/refresh_prior_pair_global_block_v1.py":"scripts/refresh_prior_pair_v3_global_block.py",
 "src/posteriorforge/performance_prior_pair_v1.py":"src/posteriorforge/performance_prior_pair_v3.py",
 "scripts/qualify_prior_coordinate_pair_v1.py":"scripts/qualify_prior_coordinate_pair_v3.py"},recorded_sources=True)
plots=_literal_clone(_ANCESTOR.plots,{
 ": FP32 / cached19 / original all8 at6+10; physical G3 separate":": v3 F32 forward / Policy C F64 probability / physical F64; cached18; original all8 at6+10+14; physical G3 separate",
 "Pair-qualified prior-coordinate cohorts: four predeclared physical coordinates":"V3 pair-qualified physical cohorts: four predeclared coordinates",
 "Pair-qualified prior-coordinate cohorts: all21 physical densities":"V3 pair-qualified cohorts: all21 physical densities",
 "Pair-qualified all21 physical marginal CDFs":"V3 all21 physical marginal CDFs",
 "Pair-qualified all32 fixed physical projection CDFs":"V3 all32 fixed physical projection CDFs",
 "pair-density-focus.png":"pair-v3-density-focus.png","pair-density.png":"pair-v3-density.png","pair-":"pair-v3-"})
# The zero concerns NEW-parent draws only; OLD baseline stream verification is
# inherited verbatim. No numerical expression or control-flow byte changes.
_old_key="new_global_or_normal_draws_during_assessment";_new_key="new_parent_global_or_normal_draws_during_assessment";_key_edits=[0]
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
_RUN_CODE=_ANCESTOR.run.__code__.replace(co_consts=_metadata_constants(_ANCESTOR.run.__code__.co_consts))
require(_key_edits==[1] and _metadata_constants(_RUN_CODE.co_consts,True)==_ANCESTOR.run.__code__.co_consts
 and _RUN_CODE.co_code==_ANCESTOR.run.__code__.co_code,"v3 original scientific run/control flow differs")
run=FunctionType(_RUN_CODE,globals(),_ANCESTOR.run.__name__,_ANCESTOR.run.__defaults__,_ANCESTOR.run.__closure__);run.__kwdefaults__=_ANCESTOR.run.__kwdefaults__
_AUTHENTICATE_CODE=authenticate_hybrid.__code__;_PLOTS_CODE=plots.__code__

def _configuration():return (SCHEMA,WRAPPER_SHA256,V1_EVALUATOR_SHA256,V2_EVALUATOR_SHA256,COUNT,FOCUS,COLORS,LABELS,tuple(sorted(METRIC_SOURCES.items())),_RECORDED_GIT_FIELDS)
def _metric_limits():return (_validation.CDF_LIMIT,_validation.BIAS_SD_LIMIT,_validation.SD_RATIO_LIMITS,_validation.REFERENCE_PRECISION_LIMIT)
def _check_sources():
 require(_configuration()==_CONFIGURATION and source_sha256()==_IMPORTED_SHA256,"v3 assessment source/configuration changed")
 require(_metric_limits()==(.10,.20,(.8,1.2),.005),"original scientific limits changed")
 for name,fn,code in _FUNCTIONS:
  require(globals().get(name) is fn and fn.__code__ is code,"v3 assessment callable changed: "+name)
 for name,value in _OBJECTS.items():require(globals().get(name) is value,"v3 assessment dependency changed: "+name)
 for name,fn,code in _METRIC_FUNCTIONS:
  require(getattr(_validation,name) is fn and fn.__code__ is code,"original scientific gate callable changed: "+name)
 require(authenticate_hybrid.__code__ is _AUTHENTICATE_CODE and plots.__code__ is _PLOTS_CODE and run.__code__ is _RUN_CODE,
  "v3 assessment derived function changed")
 require(validate_reference_custody.__code__ is _ANCESTOR.validate_reference_custody.__code__
  and load_baseline.__code__ is _ANCESTOR.load_baseline.__code__,"original reference/baseline custody changed")
 root=Path(__file__).resolve().parents[1]
 for path,digest in {**METRIC_SOURCES,"scripts/evaluate_prior_pair_v1.py":V1_EVALUATOR_SHA256,"scripts/evaluate_prior_pair_v2.py":V2_EVALUATOR_SHA256}.items():
  require(sha256_file(root/path)==digest,"original scientific assessment source changed: "+path)
 return True

def derivation_descriptor():
 _check_sources()
 return {"schema":"posteriorforge.prior_pair_v3_assessment_derivation.v1","v1_evaluator_sha256":V1_EVALUATOR_SHA256,
  "v2_evaluator_sha256":V2_EVALUATOR_SHA256,"source_display_AST_deltas_reverse_exact":True,"recorded_Git_source_dictionary_additions":list(_RECORDED_GIT_FIELDS),
  "run_controlflow_bytecode_equal":run.__code__.co_code==_ANCESTOR.run.__code__.co_code,
  "run_names_varnames_freevars_equal":(run.__code__.co_names,run.__code__.co_varnames,run.__code__.co_freevars)==(_ANCESTOR.run.__code__.co_names,_ANCESTOR.run.__code__.co_varnames,_ANCESTOR.run.__code__.co_freevars),
  "run_metadata_key_only_reversible":_metadata_constants(run.__code__.co_consts,True)==_ANCESTOR.run.__code__.co_consts,
  "original_reference_and_baseline_codeobjects_identical":validate_reference_custody.__code__ is _ANCESTOR.validate_reference_custody.__code__ and load_baseline.__code__ is _ANCESTOR.load_baseline.__code__,
  "old_module_globals_unmodified":_ANCESTOR.SCHEMA=="posteriorforge.prior_coordinate_pair_assessment.v1" and _ANCESTOR.WRAPPER_SHA256=="22e57084351631f36c6989cc01fe8bf4f2827095e587c496cda2e7f368d3cdb9",
  "metric_G3_seed_or_reference_math_changed":False,"scientific_metric_sources":dict(METRIC_SOURCES),
  "expected_complete_count":COUNT,"marginals":21,"joint_projections":32,"CDF_limit":.10,"mean_bias_reference_SD_limit":.20,"SD_ratio_limits":[.8,1.2],"reference_precision_limit":.005,
  "assessment_RNG_scope":"NEW G3 uses saved actual Z only; matched OLD baseline verification replays already-recorded private PCG64 stream, no new posterior/G3"}

def source_guard(*,require_committed=True):
 _check_sources()
 require(type(require_committed) is bool,"explicit source commitment mode required")
 require(type(WRAPPER_SHA256) is str and len(WRAPPER_SHA256)==64,"v3 refresh wrapper unpinned; actual assessment disabled")
 w=load("refresh_prior_pair_v3_global_block.py",WRAPPER_SHA256)
 producer,pairs,pins=w.source_guard(require_committed=require_committed);root=Path(__file__).resolve().parents[1]
 if require_committed:
  w.BASE.verify_git_sources(root,pins["git_revision"],{"scripts/"+Path(__file__).name:source_sha256(),
   "scripts/refresh_prior_pair_v3_global_block.py":WRAPPER_SHA256,"scripts/evaluate_prior_pair_v1.py":V1_EVALUATOR_SHA256,
   "scripts/evaluate_prior_pair_v2.py":V2_EVALUATOR_SHA256,**METRIC_SOURCES})
 return w,producer,pairs,{**pins,"pair_v3_evaluator":source_sha256(),"pair_v3_G3_wrapper":WRAPPER_SHA256,
  "v1_metric_plot_ancestry":V1_EVALUATOR_SHA256,"v2_source_display_ancestry":V2_EVALUATOR_SHA256,
  "metric_or_G3_math_changed":False,"assessment_derivation":derivation_descriptor(),"assessment_metric_sources":dict(METRIC_SOURCES),
  "assessment_RNG_scope":"NEW G3 uses saved actual Z only; matched OLD baseline verification replays already-recorded PCG64 stream, no new posterior/G3"}

_METRIC_FUNCTIONS=tuple((n,f,f.__code__) for n,f in vars(_validation).items() if inspect.isfunction(f) and f.__module__==_validation.__name__)
_FUNCTIONS=tuple((n,f,f.__code__) for n,f in list(globals().items()) if inspect.isfunction(f) and f.__globals__ is globals())
_OBJECTS={n:globals()[n] for n in ('_ANCESTOR','_validation','ast','copy','hashlib','importlib','inspect','Path','FunctionType','np','sha256_file','require','load')}
_CONFIGURATION=_configuration();_IMPORTED_SHA256=source_sha256()
if __name__=="__main__":main()
