#!/usr/bin/env python3
"""One explicit SS15 fallback repair and source-bound CPU proof composition.

Built with PriorLabs-TabPFN. The original FAILED all21 artifact is preserved.
Only stage20 is rerun. Private source edits affect loop selection, declared
fallback arguments, bounded deadline and stage-local metadata/validation.
Original conditional probability, preprocessing, sampler and comparison math
remain byte-for-byte AST-identical after reversing those declared edits.
"""
from __future__ import annotations
import argparse,ast,copy,hashlib,importlib.util,inspect,json,textwrap
from pathlib import Path
from posteriorforge.artifacts import sha256_file,write_json

ANCESTOR_SHA="0c3dd7274d6741bba2ba0ae2744542865f48b2f887a6831e1d2aa6bce47f78c0"
ORIGINAL_SHA="5dc422946763156d9f876b2f07d9781e7b350265746ffa1024854d579f819deb"
STAGE_SCHEMA="posteriorforge.ss15_conditional_stage_qualification.v1"
SCHEMA="posteriorforge.ss15_conditional_assembly.v1"
MAP={"10":[5],"20":[4]}
INT_MAP={10:[5],20:[4]}
STAGE=20
_CLONE=None
_CLONE_CODE=None
_DESCRIPTOR=None

def require(value,message):
 if not value:raise ValueError(message)

def ancestor():
 path=Path(__file__).with_name("qualify_ss15_conditionals.py")
 require(sha256_file(path)==ANCESTOR_SHA,"frozen original SS15 collector changed")
 spec=importlib.util.spec_from_file_location("_ss15_repair_ancestor",path)
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def source_pins():
 old=ancestor().source_pins()
 return {"assembly_verifier":sha256_file(__file__),"ancestor_qualifier":ANCESTOR_SHA,
  **{key:old[key] for key in ("representation","execution","utility","comparison_hard_guards")}}

def dump(node):return ast.dump(node,include_attributes=False)

def private_stage_collector():
 global _CLONE,_CLONE_CODE,_DESCRIPTOR
 q=ancestor()
 if _CLONE is not None:
  require(_CLONE.__code__ is _CLONE_CODE,"private stage collector code changed")
  return _CLONE,copy.deepcopy(_DESCRIPTOR)
 tree=ast.parse(textwrap.dedent(inspect.getsource(q.run)));original=copy.deepcopy(tree)
 changes={};reverse={}
 def replace(node,new,label):
  reverse[id(new)]=copy.deepcopy(node);changes[label]=changes.get(label,0)+1
  return ast.copy_location(new,node)
 class Select(ast.NodeTransformer):
  def visit_For(self,node):
   node=self.generic_visit(node)
   if isinstance(node.target,ast.Name) and node.target.id=="stage" and dump(node.iter)==dump(ast.parse("range(21)",mode="eval").body):
    node.iter=replace(node.iter,ast.parse("(20,)",mode="eval").body,"stage_loop")
   return node
  def visit_keyword(self,node):
   node=self.generic_visit(node)
   if node.arg=="fallback_map" and dump(node.value)==dump(ast.parse("{10:[5]}",mode="eval").body):
    node.value=replace(node.value,ast.Name(id="_REPAIR_INT_MAP",ctx=ast.Load()),"declared_fallback_argument")
   if node.arg in ("matching_FP32_execution_passed_all21","FP16_hard_validity_passed_all21"):
    new=copy.deepcopy(node);new.arg=node.arg.replace("_all21","_stage")
    return replace(node,new,"stage_only_flag_"+node.arg)
   return node
  def visit_Compare(self,node):
   node=self.generic_visit(node)
   if dump(node)==dump(ast.parse("900<=max_seconds<=2250",mode="eval").body):
    return replace(node,ast.parse("60<=max_seconds<=450",mode="eval").body,"bounded_stage_deadline")
   return node
  def visit_Subscript(self,node):
   node=self.generic_visit(node)
   if dump(node)==dump(ast.parse('report["matching_FP32_execution_passed_all21"]',mode="eval").body):
    return replace(node,ast.parse('report["matching_FP32_execution_passed_stage"]',mode="eval").body,"stage_status_predicate")
   return node
  def visit_Constant(self,node):
   if type(node.value) is int and node.value==2352:return replace(node,ast.Constant(112),"stage_CDF_count")
   if node.value=="ss15_conditional_screen_pass":return replace(node,ast.Constant("ss15_conditional_stage_screen_pass"),"stage_success_status")
   if node.value=="ss15_matching_FP32_structural_failure":return replace(node,ast.Constant("ss15_conditional_stage_structural_failure"),"stage_failure_status")
   return node
 changed=Select().visit(copy.deepcopy(original))
 expected={"stage_loop":1,"declared_fallback_argument":2,"bounded_stage_deadline":1,"stage_status_predicate":1,
  "stage_CDF_count":1,"stage_success_status":2,"stage_failure_status":1,
  "stage_only_flag_matching_FP32_execution_passed_all21":1,"stage_only_flag_FP16_hard_validity_passed_all21":1}
 require(changes==expected,"explicit stage-only source edit counts differ: "+repr(changes))
 class Undo(ast.NodeTransformer):
  def visit(self,node):
   if id(node) in reverse:return copy.deepcopy(reverse[id(node)])
   return super().visit(node)
 restored=Undo().visit(changed)
 require(dump(restored)==dump(original),"repair contains undeclared computational changes")
 changes.clear();reverse.clear();changed=Select().visit(copy.deepcopy(original));ast.fix_missing_locations(changed)
 descriptor={"ancestor_source_sha256":ANCESTOR_SHA,"declared_edit_counts":dict(changes),"reverse_AST_exact":True,
  "original_AST_sha256":hashlib.sha256(dump(original).encode()).hexdigest(),
  "stage_AST_sha256":hashlib.sha256(dump(changed).encode()).hexdigest(),
  "scope":"stage loop, explicit map arguments, deadline and stage metadata/predicate only; original numeric pipeline unchanged"}
 def stage_writer(path,value):
  if Path(path).name=="manifest.json":
   value={**value,"requested_stages":[20],
    "completed_stages":[r["stage_index"] for r in value.get("stage_results",[]) if "matching_FP32_execution_passed" in r],
    "full21_qualified":False,"matching_FP32_execution_passed_all21":False,"FP16_hard_validity_passed_all21":False,
    "private_collector":descriptor}
  q.write_json(path,value)
 def stage_validation(document,**kwargs):
  document.update(requested_stages=[20],completed_stages=[20],private_collector=descriptor,full21_qualified=False)
  return validate_stage(document,**kwargs)
 scope=dict(q.run.__globals__)
 scope.update(SCHEMA=STAGE_SCHEMA,FALLBACK_MAP=MAP,_REPAIR_INT_MAP=INT_MAP,source_pins=source_pins,
  validate_qualification=stage_validation,write_json=stage_writer)
 exec(compile(changed,"<posteriorforge-SS15-stage20-only>","exec",dont_inherit=True),scope)
 _CLONE=scope["run"];_CLONE_CODE=_CLONE.__code__;_DESCRIPTOR=descriptor
 return _CLONE,copy.deepcopy(_DESCRIPTOR)

def cdf_context(descriptor,count):
 require(descriptor.get("patched_call_count")==count and descriptor.get("original_alias_restored") is True
  and descriptor.get("sdk_sources_unchanged") is True and descriptor.get("sdk_aliases_unchanged") is True
  and descriptor.get("executed_corrected_cdf_code_sha256")==descriptor.get("corrected_cdf_code_sha256")
  and descriptor.get("corrected_cdf_code_sha256")!=descriptor.get("original_cdf_code_sha256"),
  "actual corrected-CDF source/restoration/count context differs")

def check_row(row,map_,*,must_pass):
 q=ancestor();calculated=q.stage_decision(row)
 require(all(row.get(key)==value for key,value in calculated.items()),"source-bound saved row decision differs")
 require(calculated["hard_validity_passed"] and (calculated["matching_FP32_execution_passed"] or not must_pass),
   "selected conditional row failed hard/FP32 screen")
 require(row["corrected_CDF_calls"]==112,"selected stage CDF coverage differs")
 require(isinstance(row.get("RNG_identity_before"),dict) and row["RNG_identity_before"]
  and row["RNG_identity_before"]==row.get("RNG_identity_after"),"recorded conditional RNG identity changed")
 from posteriorforge import execution_ss15 as engine
 for label,mode in (("FP32",engine.CONTROL_MODE),("FP16",engine.FP16_MODE)):
  q.witness_guard(row["precision_sessions"][label],mode=mode,stage=row["stage_index"],fallback_map=map_)
 return calculated

def validate_stage(document,*,raw_bindings,transformed_bindings,representation_descriptor,expected_sources,fallback_map):
 q=ancestor()
 expected={"schema":STAGE_SCHEMA,"status":"ss15_conditional_stage_screen_pass","fallback_map":MAP,
  "sources":source_pins(),"raw_bindings":raw_bindings,"transformed_bindings":transformed_bindings,
  "representation_descriptor":representation_descriptor,"thresholds":q.THRESHOLDS,"row_counts":list(q.ROW_COUNTS),"stress_rows":8,
  "posterior_vectors_generated":0,"reference_integrations":0,"sources_unchanged":True,"global_RNG_preserved":True,
  "matching_FP32_execution_passed_all21":False,"FP16_hard_validity_passed_all21":False,
  "matching_FP32_execution_passed_stage":True,"FP16_hard_validity_passed_stage":True,
  "requested_stages":[20],"completed_stages":[20],"full21_qualified":False,
  "private_collector":private_stage_collector()[1]}
 for key,value in expected.items():require(document.get(key)==value,"stage20 proof binding differs: "+key)
 require(fallback_map==MAP and expected_sources==source_pins(),"explicit stage20 policy/source required")
 require([r["stage_index"] for r in document["stage_results"]]==[20],"stage-local proof must contain only20")
 check_row(document["stage_results"][0],MAP,must_pass=True);cdf_context(document["compatibility_adapter"],112)
 return {"passed":True,"stage_index":20,"all21_claimed":False,"fallback_map":MAP,
  "FP16_qualified_as_FP32":False,"posterior_equivalence_claimed":False}

def run_stage(**kwargs):
 require(60<=kwargs.get("max_seconds",450)<=450,"bounded60..450 second stage-only execution required")
 kwargs.setdefault("max_seconds",450)
 function,_=private_stage_collector();function(**kwargs)
 path=Path(kwargs["output"])/"manifest.json";document=json.loads(path.read_bytes())
 require(document["requested_stages"]==document["completed_stages"]==[20],"stage-only completion required")
 validate_stage(document,raw_bindings=document["raw_bindings"],transformed_bindings=document["transformed_bindings"],
  representation_descriptor=document["representation_descriptor"],expected_sources=source_pins(),fallback_map=MAP)
 return document

def _component(path,expected_sha=None):
 raw=Path(path).read_bytes();digest=hashlib.sha256(raw).hexdigest()
 require(expected_sha is None or digest==expected_sha,"component bytes changed")
 return json.loads(raw),raw,digest

def row_sha(row):return hashlib.sha256(json.dumps(row,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()

def compose(original,stage):
 q=ancestor()
 require(original.get("global_RNG_preserved") is True,"original conditional RNG assertion absent/failed")
 require(original["status"]=="ss15_matching_FP32_structural_failure" and original["sources"]==q.source_pins()
  and original["fallback_map"]=={"10":[5]},"original failed proof/source/policy must stay unchanged")
 require([r["stage_index"] for r in original["stage_results"]]==list(range(21)),"original all21 coverage required")
 for key in ("raw_bindings","transformed_bindings","representation_descriptor","thresholds","row_counts","stress_rows",
             "prefix_manifest_sha256","prefix_samples_sha256","fixed_prefix_row_indices"):
  require(original[key]==stage[key],"matching scientific/conditional inputs differ: "+key)
 decisions=[check_row(row,{"10":[5]},must_pass=False) for row in original["stage_results"]]
 require([i for i,d in enumerate(decisions) if not d["matching_FP32_execution_passed"]]==[20],
  "repair must select the sole failed original stage20")
 validate_stage(stage,raw_bindings=original["raw_bindings"],transformed_bindings=original["transformed_bindings"],
  representation_descriptor=original["representation_descriptor"],expected_sources=source_pins(),fallback_map=MAP)
 cdf_context(original["compatibility_adapter"],2352)
 selected=copy.deepcopy(original["stage_results"][:20])+copy.deepcopy(stage["stage_results"])
 for row in selected:
  index=row["stage_index"]
  if index<20:
   require({"10":[5]}.get(str(index),[])==MAP.get(str(index),[]),"unreplayed stage fallback semantics changed")
  check_row(row,MAP,must_pass=True)
 return selected

def validate_qualification(document,*,raw_bindings,transformed_bindings,representation_descriptor,expected_sources,fallback_map=None,artifact_root=None):
 """Torch-free source-bound composition replay; no model/numeric re-execution."""
 require(artifact_root is not None,"composite artifact_root is required for authenticated component replay")
 root=Path(artifact_root).resolve();q=ancestor()
 expected={"schema":SCHEMA,"status":"ss15_conditional_screen_pass","sources":source_pins(),
  "fallback_map":MAP,"raw_bindings":raw_bindings,"transformed_bindings":transformed_bindings,
  "representation_descriptor":representation_descriptor,"thresholds":q.THRESHOLDS,"row_counts":list(q.ROW_COUNTS),"stress_rows":8,
  "matching_FP32_execution_passed_all21":True,"FP16_hard_validity_passed_all21":True,
  "FP16_qualified_as_FP32":False,"posterior_equivalence_claimed":False,"sources_unchanged":True,
  "posterior_vectors_generated":0,"reference_integrations":0}
 for key,value in expected.items():require(document.get(key)==value,"composite binding differs: "+key)
 require(expected_sources==source_pins() and fallback_map==MAP,"explicit composite source/map required")
 components=document["components"];loaded={}
 for name,filename in (("original_failed","original-failed-manifest.json"),("stage20","stage20-manifest.json")):
  entry=components[name];require(entry["file"]=="inputs/"+filename,"component manifest path differs")
  path=(root/entry["file"]).resolve();require(path.is_relative_to(root),"component path escapes artifact root")
  value,_,digest=_component(path,ORIGINAL_SHA if name=="original_failed" else entry["sha256"])
  require(digest==entry["sha256"] and value["status"]==entry["producer_status"],"component source/status custody differs")
  loaded[name]=value
 selected=compose(loaded["original_failed"],loaded["stage20"])
 require(document["stage_results"]==selected,"selected stage records differ from authenticated inputs")
 expected_rows=[{"stage_index":row["stage_index"],"component":"original_failed" if row["stage_index"]<20 else "stage20",
  "row_sha256":row_sha(row)} for row in selected]
 require(document["selected_stage_sources"]==expected_rows,"per-stage provenance differs")
 contexts=[{"component":"original_failed","actual_observed_CDF_calls":2352,"selected_stage_CDF_calls":2240,
  "descriptor":loaded["original_failed"]["compatibility_adapter"]},
  {"component":"stage20","actual_observed_CDF_calls":112,"selected_stage_CDF_calls":112,
   "descriptor":loaded["stage20"]["compatibility_adapter"]}]
 require(document["corrected_CDF_contexts"]==contexts and document["selected_stage_CDF_calls"]==2352
  and document["single_CDF_context_claimed"] is False and "compatibility_adapter" not in document,
  "distinct actual CDF contexts were conflated")
 failures=[r["stage_index"] for r in selected if not r["FP16_strict_passed"]]
 matched=[r["stage_index"] for r in selected if not r["FP16_matching_uncached_strict_passed"]]
 require(document["FP16_strict_failed_stages"]==failures and document["FP16_matching_uncached_strict_failed_stages"]==matched,
  "FP16 approximation flags relabelled")
 return {"passed":True,"schema":SCHEMA+".replay","stages":21,
  "matching_FP32_execution_passed_all21":True,"FP16_hard_validity_passed_all21":True,
  "FP16_strict_failed_stages":failures,"FP16_matching_uncached_strict_failed_stages":matched,
  "original_failed_manifest_sha256":ORIGINAL_SHA,"original_status_preserved":True,"selected_stage_CDF_calls":2352,
  "distinct_CDF_contexts":2,"source_bound_saved_predicates_replayed":True,"model_predictions_reexecuted":False,
  "FP16_qualified_as_FP32":False,"posterior_equivalence_claimed":False}

def assemble(*,original,stage_proof,output):
 old,old_bytes,old_sha=_component(original,ORIGINAL_SHA);stage,stage_bytes,stage_sha=_component(stage_proof)
 selected=compose(old,stage);q=ancestor();output=Path(output);output.mkdir(parents=True,exist_ok=False)
 (output/"inputs").mkdir()
 for name,content in (("original-failed-manifest.json",old_bytes),("stage20-manifest.json",stage_bytes)):
  with (output/"inputs"/name).open("xb") as stream:stream.write(content)
 document={key:copy.deepcopy(old[key]) for key in ("raw_bindings","transformed_bindings","representation_descriptor",
  "thresholds","row_counts","stress_rows","prefix_manifest_sha256","prefix_samples_sha256","fixed_prefix_row_indices")}
 document.update(schema=SCHEMA,status="ss15_conditional_screen_pass",sources=source_pins(),fallback_map=MAP,
  attribution="Built with PriorLabs-TabPFN",sources_unchanged=True,stage_results=selected,
  matching_FP32_execution_passed_all21=True,FP16_hard_validity_passed_all21=True,
  FP16_strict_failed_stages=[r["stage_index"] for r in selected if not r["FP16_strict_passed"]],
  FP16_matching_uncached_strict_failed_stages=[r["stage_index"] for r in selected if not r["FP16_matching_uncached_strict_passed"]],
  FP16_qualified_as_FP32=False,posterior_equivalence_claimed=False,posterior_vectors_generated=0,reference_integrations=0,
  components={"original_failed":{"file":"inputs/original-failed-manifest.json","sha256":old_sha,"producer_status":old["status"]},
   "stage20":{"file":"inputs/stage20-manifest.json","sha256":stage_sha,"producer_status":stage["status"]}},
  selected_stage_sources=[{"stage_index":r["stage_index"],"component":"original_failed" if r["stage_index"]<20 else "stage20",
   "row_sha256":row_sha(r)} for r in selected],
  corrected_CDF_contexts=[{"component":"original_failed","actual_observed_CDF_calls":2352,"selected_stage_CDF_calls":2240,
   "descriptor":old["compatibility_adapter"]},
   {"component":"stage20","actual_observed_CDF_calls":112,"selected_stage_CDF_calls":112,"descriptor":stage["compatibility_adapter"]}],
  selected_stage_CDF_calls=2352,single_CDF_context_claimed=False,
  policy_composition={"same_engine_source":old["sources"]["execution"],
   "stage_local_only_delta":{"stage_index":20,"old_members":[],"new_members":[4]},
   "untouched_stage_indices":list(range(20)),"failed_input_never_reclassified":True})
 document["replay"]=validate_qualification(document,raw_bindings=document["raw_bindings"],
  transformed_bindings=document["transformed_bindings"],representation_descriptor=document["representation_descriptor"],
  expected_sources=source_pins(),fallback_map=MAP,artifact_root=output)
 write_json(output/"manifest.json",document)
 return document

def main():
 parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest="command",required=True)
 stage=sub.add_parser("stage20")
 for key in ("banks","prefix-run","checkpoint","output","ledger"):stage.add_argument("--"+key,required=True)
 stage.add_argument("--max-seconds",type=int,default=450)
 build=sub.add_parser("assemble")
 for key in ("original","stage-proof","output"):build.add_argument("--"+key,required=True)
 args=vars(parser.parse_args());command=args.pop("command")
 result=run_stage(**args) if command=="stage20" else assemble(**args)
 print(json.dumps({"schema":result["schema"],"status":result["status"]}))
if __name__=="__main__":main()
