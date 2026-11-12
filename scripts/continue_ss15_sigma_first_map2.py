#!/usr/bin/env python3
"""Selected15..20 MAP2 continuation. Built with PriorLabs-TabPFN.

Six single-stage executions of the EXACT frozen dde09 run code, each with
private STAGES and a partial-only writer/validator. No model math, old globals,
old files or numeric decisions change. No full21 qualification is claimed.
"""
from __future__ import annotations
import argparse,copy,hashlib,importlib.util,json,time,types
from pathlib import Path
from posteriorforge.artifacts import sha256_file,write_json

SCHEMA="posteriorforge.ss15_sigma_first_map2_inventory.v1"
CHILD_SCHEMA=SCHEMA+".single_stage"
HELPER_SHA="dde09d9592e8d46c124b14360b8aed78ec699412ba2c5327175e0dc6adda72d8"
FAILED_SHA="06bdcc69a189ba90ea121afef8eeb997e760a5c63221ae739859c0ac01094983"
REQUESTED_STAGES=(15,16,17,18,19,20)
COMMON_KEYS=("sources","raw_bindings","transformed_bindings","order_policy","thresholds","row_counts","stress_rows",
 "fallback_map","stage_to_physical_order","noise_stage_indices","ordered_model_inputs","ordered_fixed_prefixes",
 "physical_model_inputs","prefix_manifest_sha256","prefix_samples_sha256","fixed_prefix_row_indices",
 "posterior_vectors_generated","reference_integrations","anchor_reused","prior_target_order_qualification_inherited")

def require(value,message):
 if not value:raise ValueError(message)

def frozen_helper():
 p=Path(__file__).with_name("qualify_ss15_sigma_first.py")
 require(sha256_file(p)==HELPER_SHA,"frozen failed MAP2 helper changed")
 s=importlib.util.spec_from_file_location("_frozen_sigma_MAP2_inventory",p)
 m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def source_pins(original=None):
 original=frozen_helper() if original is None else original
 return {"wrapper":sha256_file(__file__),"frozen_MAP2_helper":HELPER_SHA,
  "original_sources":original.source_pins(),"run_code_object_policy":"identical original code; private globals only",
  "private_overrides":["STAGES","write_json","validate_screen"]}

def partial_document(document,stage):
 """Normalize a COPY at every write; internal loop-completion flags are not qualification."""
 result=copy.deepcopy(document)
 result["schema"]=CHILD_SCHEMA
 result["original_internal_schema"]=document.get("schema")
 result["original_internal_loop_status"]=document.get("status")
 result["requested_stage_indices"]=[stage]
 rows=result.get("stage_results",[])
 row_complete=len(rows)==1 and rows[0].get("stage_index")==stage and "hard_validity_passed" in rows[0]
 complete=bool(row_complete and document.get("status")!="running"
  and document.get("sources_unchanged") is True and document.get("global_RNG_preserved") is True
  and document.get("compatibility_adapter",{}).get("original_alias_restored") is True)
 result["selected_stage_numerical_row_complete"]=row_complete
 result["selected_stage_complete"]=complete
 result["selected_stage_matching_FP32_execution_passed"]=bool(complete and rows[0]["matching_FP32_execution_passed"])
 result["selected_stage_hard_validity_passed"]=bool(complete and rows[0]["hard_validity_passed"])
 result["status"]=("partial_stage_complete" if complete else
  "partial_stage_running" if document.get("status")=="running" else "partial_stage_failed_or_incomplete")
 result["matching_FP32_execution_passed_all21"]=False
 result["FP16_hard_validity_passed_all21"]=False
 result["full21_qualified"]=False
 result["selected_stage_scope_only"]=True
 result["single_child_CDF_context_only"]=True
 result["control_flow_scope"]="original single-stage loop complete flags never mean all21"
 return result

def validate_child(document,stage,original,failed):
 rows=document.get("stage_results",[])
 require([r.get("stage_index") for r in rows]==[stage],"exact single requested stage required")
 require(all(document.get(k)==failed[k] for k in COMMON_KEYS),"same failed MAP2 scientific/model/query/prefix bindings required")
 old=original.load("qualify_ss15_conditionals.py",original.SS15_QUALIFIER_SHA)
 row=rows[0];decision=original.stage_decision(row,old)
 training=row["mask_inputs"]["training"];binding=row["mask_inputs"]
 require(row["target_physical_index"]==original.ORDER[stage]
  and row["target_parameter"]==failed["raw_bindings"]["parameter_order"][original.ORDER[stage]]
  and training["target_physical_index"]==original.ORDER[stage]
  and training["mask"]==failed["order_policy"]["stages"][stage]
  and training["policy_sha256"]==failed["order_policy"]["policy_sha256"]
  and training["original_target"]==training["prepared_target"]==binding["actual_target"]
  and training["prepared_training_predictors"]==binding["actual_training_predictors"],
  "same ordered physical target/masked training bindings required")
 require(all(row.get(k)==v for k,v in decision.items()),"actual per-stage numeric decisions differ")
 require(decision["hard_validity_passed"],"hard/source/state failure blocks valid numerical-stage inventory")
 require(row["corrected_CDF_calls"]==72,"actual single-stage9x8 CDF calls required")
 for label,mode in zip(("FP32","FP16"),original._execution_modes()):
  old.witness_guard(row["precision_sessions"][label],mode=mode,stage=stage,fallback_map=original.FALLBACK_MAP)
 d=document.get("compatibility_adapter",{})
 require(d.get("patched_call_count")==72 and d.get("original_alias_restored") is True
  and d.get("sdk_sources_unchanged") is True
  and d.get("executed_corrected_cdf_code_sha256")==d.get("corrected_cdf_code_sha256")
  and d.get("corrected_cdf_code_sha256")!=d.get("original_cdf_code_sha256"),
  "actual single-stage CDF restoration/execution footer differs")
 require(document.get("sources_unchanged") is True and document.get("global_RNG_preserved") is True,
  "actual stage source/global model RNG restoration required")
 return {"passed":True,"stage_index":stage,"matching_FP32_execution_passed":decision["matching_FP32_execution_passed"],
  "FP16_hard_validity_passed":decision["FP16_hard_validity_passed"],"FP16_strict_passed":decision["FP16_strict_passed"],
  "full21_qualified":False,"scope":"one actual stage and one actual72-call CDF context"}

def private_run(original,stage,failed):
 require(stage in REQUESTED_STAGES,"only predetermined unseen stages15..20 allowed")
 globals_=dict(original.run.__globals__)
 def writer(path,document):
  if isinstance(document,dict) and document.get("schema")==original.SCHEMA:
   document=partial_document(document,stage)
  write_json(path,document)
 def validator(document,**unused):
  return validate_child(document,stage,original,failed)
 globals_.update(STAGES=(stage,),write_json=writer,validate_screen=validator)
 clone=types.FunctionType(original.run.__code__,globals_,original.run.__name__,original.run.__defaults__,original.run.__closure__)
 clone.__kwdefaults__=copy.copy(original.run.__kwdefaults__)
 require(clone.__code__ is original.run.__code__ and clone.__globals__ is not original.run.__globals__,
  "exact run code object/private globals required")
 return clone

def failure_locations(row,original):
 old=original.load("qualify_ss15_conditionals.py",original.SS15_QUALIFIER_SHA)
 records=[]
 for query,comparison in row["comparisons"][original.ARMS[0]].items():
  for member,record in enumerate(comparison["members"]):
   if not old.strict_probability(record):
    records.append({"stage":row["stage_index"],"target_physical_index":row["target_physical_index"],
     "target_parameter":row["target_parameter"],"query":query,"member":member,
     "max_cdf_absolute_error":record["max_cdf_absolute_error"],"max_row_total_variation":record["max_row_total_variation"]})
  if not old.strict_probability(comparison["final"]):
   records.append({"stage":row["stage_index"],"query":query,"member":"final_mixture",**comparison["final"]})
 return records

def run(*,banks,prefix_run,checkpoint,failed_manifest,output,ledger,max_seconds=600):
 require(type(max_seconds) is int and 240<=max_seconds<=900,"outer continuation bound240..900s required")
 original=frozen_helper();pins=source_pins(original)
 original_globals={name:getattr(original,name) for name in ("STAGES","write_json","validate_screen")}
 original_run=original.run;code=original_run.__code__
 failed_manifest=Path(failed_manifest);failed_bytes=failed_manifest.read_bytes()
 require(hashlib.sha256(failed_bytes).hexdigest()==FAILED_SHA,"exact immutable failed MAP2 proof required")
 failed=json.loads(failed_bytes)
 require(failed["status"]=="sigma_first_matching_FP32_structural_or_hard_failure"
  and [r["stage_index"] for r in failed["stage_results"]]==list(range(15))
  and failed["stage_results"][-1]["hard_validity_passed"] is True
  and failed["stage_results"][-1]["matching_FP32_execution_passed"] is False,
  "original hard-valid numerical-stage14 failure required")
 output=Path(output);output.mkdir(parents=True,exist_ok=False)
 (output/"inputs").mkdir();(output/"inputs"/"failed-MAP2-manifest.json").write_bytes(failed_bytes)
 started=time.perf_counter();deadline=started+max_seconds
 report={"schema":SCHEMA,"status":"running","attribution":"Built with PriorLabs-TabPFN","sources":pins,
  "failed_MAP2_manifest_sha256":FAILED_SHA,"failed_MAP2_status_preserved":failed["status"],
  "requested_stage_indices":list(REQUESTED_STAGES),"children":[],"stage_results":[],
  "matching_FP32_execution_passed_all21":False,"FP16_hard_validity_passed_all21":False,"full21_qualified":False,
  "posterior_vectors_generated":0,"reference_integrations":0,"single_CDF_context_claimed":False,
  "corrected_CDF_contexts":[],"actual_corrected_CDF_call_count":0,"outer_max_seconds":max_seconds,
  "no_model_math_old_file_or_original_global_changes":True,"numeric_failures_do_not_stop_later_requested_stages":True,
  "repair_rule":"failed local stage -> all8 native original FP32; no posterior-dependent selection",
  "native_policy_implemented":False,"original_source_run_code_object_identical":True}
 write_json(output/"manifest.json",report)
 try:
  for stage in REQUESTED_STAGES:
   require(time.perf_counter()+25<deadline,"outer deadline; incomplete inventory cannot determine final native policy")
   child=output/f"stage-{stage:02d}";function=private_run(original,stage,failed)
   raw=function(banks=banks,prefix_run=prefix_run,checkpoint=checkpoint,output=child,ledger=ledger,max_seconds=600)
   validation=validate_child(raw,stage,original,failed)
   normalized=partial_document(raw,stage)
   normalized["continuation_source_sha256"]=pins["wrapper"];normalized["single_stage_replay"]=validation
   write_json(child/"manifest.json",normalized)
   report["children"].append({"stage":stage,"file":str((child/"manifest.json").relative_to(output)),
    "sha256":sha256_file(child/"manifest.json"),"validation":validation})
   row=raw["stage_results"][0];report["stage_results"].append(row)
   report["corrected_CDF_contexts"].append({"stage":stage,"descriptor":raw["compatibility_adapter"],
    "RNG_identity_before":row["RNG_identity_before"],"RNG_identity_after":row["RNG_identity_after"],
    "global_RNG_preserved":raw["global_RNG_preserved"]})
   report["actual_corrected_CDF_call_count"]+=raw["compatibility_adapter"]["patched_call_count"]
   require(all(getattr(original,k) is v for k,v in original_globals.items())
    and original.run is original_run and original.run.__code__ is code,
    "original helper function/globals changed")
   report["elapsed_seconds"]=time.perf_counter()-started;write_json(output/"manifest.json",report)
  require([r["stage_index"] for r in report["stage_results"]]==list(REQUESTED_STAGES)
   and report["actual_corrected_CDF_call_count"]==432,"all6 separate actual72-call contexts required")
  require(source_pins(original)==pins and sha256_file(failed_manifest)==FAILED_SHA,"original source/failure artifact changed")
  failures=[r["stage_index"] for r in report["stage_results"] if not r["matching_FP32_execution_passed"]]
  report.update(status="partial_inventory_complete",selected6_hard_validity_passed=True,
   selected6_matching_FP32_failed_stages=failures,
   deterministic_native_stage_inventory=[14]+failures,
   numeric_failure_locations=[item for row in report["stage_results"] for item in failure_locations(row,original)],
   sources_unchanged=True,original_globals_unchanged=True)
 except BaseException as error:
  report.update(status="partial_inventory_failed_or_incomplete",error_type=type(error).__name__,error=str(error))
  raise
 finally:
  report["elapsed_seconds"]=time.perf_counter()-started
  report["matching_FP32_execution_passed_all21"]=False;report["FP16_hard_validity_passed_all21"]=False;report["full21_qualified"]=False
  write_json(output/"manifest.json",report)
 return report

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 for name in ("banks","prefix-run","checkpoint","failed-manifest","output","ledger"):parser.add_argument("--"+name,required=True)
 parser.add_argument("--max-seconds",type=int,default=600)
 report=run(**vars(parser.parse_args()))
 print(json.dumps({key:report.get(key) for key in ("status","selected6_matching_FP32_failed_stages","actual_corrected_CDF_call_count","elapsed_seconds")},indent=2))
if __name__=="__main__":main()
