#!/usr/bin/env python3
"""Read-only baseline24 completion custody/accounting audit. No metric recomputation.

Built with PriorLabs-TabPFN. Physical model failures are retained as completed
results. No ledger mutation, refresh, inference, integration, or GPU submission.
"""
from __future__ import annotations
import argparse
from decimal import Decimal,ROUND_CEILING
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import numpy as np
from posteriorforge.budget import BudgetLedger,HELD_STATES
from posteriorforge.inference import support_mask

SCHEMA="posteriorforge.sigma_native_baseline_grid_completion.v1"
SEEDS=(20261111,20261112,20261113)
CASES=tuple(range(8))
BASELINE_SOURCES={
 "campaign":"3965e36206d7f750c66350df54b6d855d6c9f8f4f81a0e069aeeb478492cd453",
 "first":"e322d56492476be1acf5fcdb9afd30b57e2a69a373098417a2bf8944b51ba73b",
 "native":"cf135637d073439039cea3e82a98e533cd161b095aac488ea92588aacb9a1461",
 "refresh":"fc3c99732c39e96b0e704d28b90323a523337c9c560828656b0bb5972d747a4f",
 "evaluator":"146c2ca35e15b54b7f68134d383574485ed1cc228f38e6fc611624f00b411924",
 "qualification":"bedfa67dc8a1e7f41a8536e4c8091643dc05d309b0da0367dc92136c19603705"}
SOURCE_PATHS={
 "campaign":"src/posteriorforge/performance_ss15_scientific_campaign.py",
 "first":"src/posteriorforge/performance_ss15_sigma_native.py",
 "native":"src/posteriorforge/execution_ss15_sigma_native_policy.py",
 "refresh":"scripts/refresh_ss15_campaign_global_block.py",
 "evaluator":"scripts/evaluate_ss15_sigma_native_campaign.py"}

def require(value,message):
 if not value:raise ValueError(message)

def digest(path):
 return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def point_paths(phase,seed,case):
 require(type(seed) is int and seed in SEEDS and type(case) is int and case in CASES,"predefined grid point required")
 root=Path(phase)
 if (seed,case)==(20261111,0):
  return (root/"sigma-native-pilot-01/manifest.json",
          root/"global-block-sigma-native-01/manifest.json",
          root/"global-block-sigma-native-assessment-01/assessment.json")
 name=f"seed-{seed}-case-{case}"
 return (root/"sigma-native-grid-01"/name/"manifest.json",
         root/"sigma-native-grid-refresh-01"/name/"manifest.json",
         root/"sigma-native-grid-assessment-01"/name/"assessment.json")

def read_ledger(path):
 raw=Path(path).read_bytes();document=json.loads(raw)
 # These static functions validate/summarize only; no BudgetLedger instance,
 # lock, snapshot, write or reconciliation method is called.
 BudgetLedger._validate(document)
 return document,BudgetLedger._summary(document),hashlib.sha256(raw).hexdigest()

def parse_scheduler(text):
 columns=("JobIDRaw","Account","Partition","State","ElapsedRaw","AllocTRES","Restarts")
 records={}
 for line in text.splitlines():
  values=line.split("|")
  if not values or not values[0] or "." in values[0]:continue
  require(len(values)>=len(columns),"incomplete sacct row")
  row=dict(zip(columns,values));job=row["JobIDRaw"]
  require(job.isdigit() and job not in records,"duplicate/nonliteral Slurm job")
  records[job]=row
 return records

def scheduler_records(jobs):
 require(jobs and all(re.fullmatch("[0-9]+",j) for j in jobs),"literal Slurm jobs required")
 result=subprocess.run(["sacct","-j",",".join(jobs),"--format=JobIDRaw,Account,Partition,State,ElapsedRaw,AllocTRES,Restarts",
  "--parsable2","--noheader"],check=True,capture_output=True,text=True)
 return parse_scheduler(result.stdout)

def audit_allocation(parent,ledger,record):
 binding=parent["budget_binding"];job=parent["slurm_job_id"]
 require(binding["job_id"]==job and binding["account"]=="eu-26-17" and binding["partition"]=="qgpu"
  and binding["gpus"]==1,"original oneGPU budget binding differs")
 candidates=[e for e in ledger["reservations"] if e.get("job_id")==job]
 require(len(candidates)==1,"unique actual grid allocation required")
 entry=candidates[0]
 require(entry["reservation_id"]==binding["reservation_id"] and entry["state"]=="reconciled"
  and entry["gpus"]==entry["accounted_gpus"]==1 and entry["slurm_state"]=="COMPLETED",
  "root must reconcile completed oneGPU allocation first")
 argv=entry["submitted_argv"]
 require(argv.count("--no-requeue")==1 and "--requeue" not in argv
  and "--account=eu-26-17" in argv and "--partition=qgpu" in argv
  and "--gpus-per-node=1" in argv and "--nodes=1" in argv and "--cpus-per-task=16" in argv,
  "actual submitted scheduler policy differs")
 require(record["JobIDRaw"]==job and record["Account"]=="eu-26-17" and record["Partition"]=="qgpu"
  and record["State"].split()[0].rstrip("+")=="COMPLETED" and int(record["Restarts"])==0,
  "actual scheduler state/account/restarts differs")
 elapsed=int(Decimal(record["ElapsedRaw"]).to_integral_value(rounding=ROUND_CEILING))
 matches=re.findall(r"(?:^|,)gres/gpu(?:[:][^=,]+)?=([0-9]+)",record["AllocTRES"])
 require(matches and all(int(x)==1 for x in matches),"actual oneGPU TRES required")
 require(elapsed==entry["accounted_elapsed_seconds"]==entry["actual_gpu_seconds"]
  and elapsed>0,"actual elapsed/allocation seconds not reconciled")
 old=entry["accounting_record"]
 require(all(old.get(k)==record[k] for k in ("JobIDRaw","Account","Partition","State","ElapsedRaw","AllocTRES")),
  "stored ledger accounting differs from current sacct")
 return {"job_id":job,"gpus":1,"elapsed_seconds":elapsed,"actual_gpu_seconds":elapsed,
         "restarts":0,"no_requeue_verified_from_submitted_argv":True,"ledger_state":"reconciled",
         "account":"eu-26-17","partition":"qgpu"}

def tools(phase):
 repo=Path(__file__).resolve().parents[1]
 for key,name in SOURCE_PATHS.items():require(digest(repo/name)==BASELINE_SOURCES[key],"baseline source changed: "+key)
 require(digest(Path(phase)/"sigma-native-conditionals-01/manifest.json")==BASELINE_SOURCES["qualification"],
  "original finite numerical proof changed")
 path=repo/SOURCE_PATHS["evaluator"]
 spec=importlib.util.spec_from_file_location("_completion_frozen_evaluator",path)
 evaluator=importlib.util.module_from_spec(spec);spec.loader.exec_module(evaluator)
 wrapper,_=evaluator.source_guard()
 binder=evaluator.load("evaluate_cdf_pilot.py",evaluator.BINDER_SHA)
 pure=evaluator.load("evaluate_ss15_pilot.py",evaluator.PURE_HELPER_SHA)
 gatepath=Path(phase).parent/"gate-final.json"
 require(digest(gatepath)==binder.BASELINE_GATE_SHA256,"unchanged original physical gate required")
 return evaluator,wrapper,binder,pure,json.loads(gatepath.read_bytes())

def audit_point(seed,case,*,phase,ledger,scheduler=None,_tools=None,_ledger=None):
 """Audit a completed point even while other baseline points are pending."""
 parent_path,hybrid_path,assessment_path=point_paths(phase,seed,case)
 require(all(p.is_file() for p in (parent_path,hybrid_path,assessment_path)),"point is not fully closed")
 ev,w,binder,pure,gate=tools(phase) if _tools is None else _tools
 ledger_document=read_ledger(ledger)[0] if _ledger is None else _ledger
 a_raw=assessment_path.read_bytes();a=json.loads(a_raw)
 p,h,before,after,joint=ev.authenticate(w,parent_path,hybrid_path,digest(hybrid_path))
 require((p["context_seed"],p["case"])==(seed,case) and p["requested_count"]==h["requested_count"]==50000,
  "actual completed request binding/count differs")
 require(Path(p["budget_binding"]["ledger"]).resolve()==Path(ledger).resolve(),"actual recorded ledger path differs")
 expected=BASELINE_SOURCES["first"] if (seed,case)==(20261111,0) else BASELINE_SOURCES["campaign"]
 require(p["runner_source_sha256"]==expected and p["sources_unchanged"] and h["sources_unchanged"]
  and p["default_rng_preserved"] and p["default_rng_before"]==p["default_rng_after"],
  "frozen producer or recorded production RNG/source invariant differs")
 require(p["original_gate_unchanged"] and not p["clipping_or_repair_performed"],"scientific gate/policy changed")
 require(before.shape==after.shape==(50000,21) and before.dtype==after.dtype==np.float64
  and support_mask(before).all() and support_mask(after).all()
  and before[:,3:].tobytes()==after[:,3:].tobytes(),"supported50000 or all18 physical byte identity differs")
 require(a["schema"]==ev.SCHEMA and (a["context_seed"],a["case"])==(seed,case)
  and a["parent_manifest_sha256"]==digest(parent_path) and a["hybrid_manifest_sha256"]==digest(hybrid_path)
  and a["hybrid_samples_sha256"]==h["samples_sha256"] and a["all50000_rows_used"]
  and a["scientific_limits_unchanged"] and a["original_gate_sha256"]==binder.BASELINE_GATE_SHA256
  and a["plots_pending"] is False,"closed unchanged assessment custody differs")
 require(a["source_pins"]["evaluator_sha256"]==BASELINE_SOURCES["evaluator"]
  and a["source_pins"]["wrapper_sha256"]==BASELINE_SOURCES["refresh"] and a["authentication"]==joint,
  "assessment source/private-normal replay differs")
 actual=ev.actual_inputs(Path(phase).parent,seed,case,w)
 anchor=actual if (seed,case)==w.ANCHOR else ev.actual_inputs(Path(phase).parent,*w.ANCHOR,w)
 execution=ev.execution_checks(w,p,actual,anchor,Path(phase)/"sigma-native-conditionals-01/manifest.json",binder,pure)
 require(execution==a["execution_checks"] and execution["passed"],"saved technical/mask/count proof differs")
 _,_,_,reference=binder.reference_for_case(Path(phase).parent,gate,actual["query"],case)
 binder.validate_run_reference_binding(p,reference)
 require(reference==a["reference"] and a["unchanged18_control"]["passed"]
  and a["unchanged18_control"]["all18_local_noise_columns_and_marginal_assessments_unchanged"],
  "case-specific reference/control custody differs")
 require(all(s["all_rows_used"]==50000 and s["reference_qualified"] for s in a["summary"].values()),
  "qualified reference or full-row assessment missing")
 require(a["status"]==a["summary"]["after"]["status"] and a["passed"]==a["summary"]["after"]["passed"]
  and a["status"] in ("pass","model_failure"),"physical outcome labels inconsistent")
 closure=json.loads(assessment_path.with_name("checksums.json").read_bytes())
 figures={v["file"]:v["sha256"] for v in a["figures"].values()}
 require(len(figures)>1 and set(closure)=={"assessment.json",*figures},"closed figure set differs")
 require(closure["assessment.json"]==hashlib.sha256(a_raw).hexdigest(),"assessment checksum differs")
 for name,want in closure.items():
  require(Path(name).name==name and digest(assessment_path.with_name(name))==want,"figure/assessment bytes differ")
 require(all(closure[name]==want for name,want in figures.items()),"declared figure identity differs")
 job=p["slurm_job_id"]
 records=scheduler_records([job]) if scheduler is None else scheduler
 require(job in records,"scheduler record missing")
 allocation=audit_allocation(p,ledger_document,records[job])
 core=dict(context_seed=seed,case=case,parent_manifest_sha256=digest(parent_path),
   hybrid_manifest_sha256=digest(hybrid_path),assessment_sha256=hashlib.sha256(a_raw).hexdigest(),
   count=50000,technical_passed=True,physical_status=a["status"])
 detail=dict(parent_payload_sha256=p["samples_sha256"],hybrid_payload_sha256=h["samples_sha256"],
   finite_supported_parent_rows=50000,finite_supported_hybrid_rows=50000,all18_columns_byte_identical=True,
   parent_summary=a["summary"]["before"],hybrid_summary=a["summary"]["after"],
   closed_figures=figures,checksums_sha256=digest(assessment_path.with_name("checksums.json")),
   reference_custody=reference,original_gate_sha256=binder.BASELINE_GATE_SHA256,
   allocation=allocation,posterior_request_seconds=p["timings"]["posterior_request_seconds"],
   wrapper_seconds=p["wrapper_total_seconds"],joint_refresh_CPU_seconds=h["refresh"]["refresh_seconds"],
   numerical_screen_scope=execution["screen_scope"],private_normal_replay_performed=True,new_refresh_performed=False,
   point_paths=dict(parent=str(parent_path),hybrid=str(hybrid_path),assessment=str(assessment_path)))
 return {"point":core,"details":detail}

def completion_document(audits,ledger_summary,ledger_sha256):
 points=[r["point"] for r in audits]
 expected=[(s,c) for s in SEEDS for c in CASES]
 require([(r["context_seed"],r["case"]) for r in points]==expected,"all24 sorted pairs required, no omissions or duplicates")
 core_keys={"context_seed","case","parent_manifest_sha256","hybrid_manifest_sha256","assessment_sha256","count","technical_passed","physical_status"}
 require(all(set(r)==core_keys and r["count"]==50000 and r["technical_passed"] is True
             and r["physical_status"] in ("pass","model_failure") for r in points),"all24 exact-core full-count technical passes required")
 for record in audits:
  allocation=record["details"]["allocation"]
  require(allocation["gpus"]==1 and allocation["restarts"]==0
          and allocation["ledger_state"]=="reconciled" and allocation["no_requeue_verified_from_submitted_argv"] is True,
          "all24 singleGPU closed allocations required")
 require(ledger_summary["reserved_gpu_seconds"]==0 and ledger_summary["cap_exceeded"] is False,
  "root must reconcile all allocations and leave zero reserved before completion")
 return dict(schema=SCHEMA,status="complete",attribution="Built with PriorLabs-TabPFN",
   context_seeds=list(SEEDS),cases=list(CASES),point_count=24,requested_count_per_point=50000,
   completed_pairs=[list(x) for x in expected],all_requests_complete=True,all_assessments_closed=True,
   all_integrity_and_reference_checks_passed=True,all_GPU_allocations_reconciled=True,
   reserved_GPU_seconds_at_completion=0,baseline_sources=BASELINE_SOURCES,points=points,
   physical_accuracy_gate_passed=all(p["physical_status"]=="pass" for p in points),
   physical_pass_count=sum(p["physical_status"]=="pass" for p in points),
   physical_failure_count=sum(p["physical_status"]=="model_failure" for p in points),
   details=[r["details"] for r in audits],ledger_readonly_summary=ledger_summary,ledger_sha256=ledger_sha256,
   baseline_grid_actual_GPU_seconds=sum(r["details"]["allocation"]["actual_gpu_seconds"] for r in audits),
   no_new_metrics_or_reference_integrations=True,no_new_posterior_refresh_or_model_calls=True,
   ledger_mutated_or_reconciled_by_helper=False,torch_imported="torch" in sys.modules,
   auditor_source_sha256=digest(__file__))

def run(*,phase,ledger,output):
 output=Path(output);output.mkdir(parents=True,exist_ok=False)
 ledger_document,summary,ledger_sha=read_ledger(ledger)
 bundle=tools(phase);jobs=[]
 for seed in SEEDS:
  for case in CASES:
   paths=point_paths(phase,seed,case)
   require(all(p.is_file() for p in paths),"all24 parent/refresh/assessment files must close before final receipt")
   jobs.append(json.loads(paths[0].read_bytes())["slurm_job_id"])
 require(len(set(jobs))==24,"one distinct scientific GPU allocation per grid point required")
 scheduler=scheduler_records(jobs)
 audits=[audit_point(seed,case,phase=phase,ledger=ledger,scheduler=scheduler,_tools=bundle,_ledger=ledger_document)
         for seed in SEEDS for case in CASES]
 result=completion_document(audits,summary,ledger_sha)
 require(digest(ledger)==ledger_sha,"ledger changed during read-only completion audit")
 require("torch" not in sys.modules,"CPU completion audit imported Torch")
 target=output/"completion.json";target.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n")
 (output/"checksums.json").write_text(json.dumps({"completion.json":digest(target)},sort_keys=True,indent=2)+"\n")
 return result

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 for name in ("phase","ledger","output"):parser.add_argument("--"+name,required=True)
 args=parser.parse_args();result=run(phase=args.phase,ledger=args.ledger,output=args.output)
 print(json.dumps({"status":result["status"],"points":24,"physical_gate":result["physical_accuracy_gate_passed"],
  "receipt_sha256":digest(Path(args.output)/"completion.json")}))
if __name__=="__main__":main()
