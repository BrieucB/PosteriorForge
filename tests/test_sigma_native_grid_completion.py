"""Completion custody/accounting tests; physical failure must not block closure."""
import copy
import importlib.util
import json
from pathlib import Path
import pytest
from posteriorforge import budget
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("baseline_completion",ROOT/"scripts/audit_sigma_native_grid_completion.py")
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)

def accounting():
 return {"JobIDRaw":"12345","Account":"eu-26-17","Partition":"qgpu","State":"COMPLETED",
         "ElapsedRaw":"209","AllocTRES":"billing=1,cpu=16,gres/gpu=1,gres/gpu:a100=1","Restarts":"0"}

def allocation_inputs():
 row=accounting()
 entry={"reservation_id":"rid","job_id":"12345","state":"reconciled","gpus":1,
  "walltime_seconds":1200,"reserved_gpu_seconds":1200,"actual_gpu_seconds":209,
  "accounted_gpus":1,"accounted_elapsed_seconds":209,"slurm_state":"COMPLETED",
  "submitted_argv":["sbatch","--no-requeue","--account=eu-26-17","--partition=qgpu",
                    "--gpus-per-node=1","--nodes=1","--cpus-per-task=16"],
  "accounting_record":dict(row)}
 ledger={"schema":budget.SCHEMA,"cap_gpu_seconds":720000,"reservations":[entry],"events":[]}
 parent={"slurm_job_id":"12345","budget_binding":{"job_id":"12345","account":"eu-26-17",
         "partition":"qgpu","gpus":1,"reservation_id":"rid"}}
 return parent,ledger,row

def audits():
 records=[]
 for seed in a.SEEDS:
  for case in a.CASES:
   records.append({"point":{"context_seed":seed,"case":case,"parent_manifest_sha256":"1"*64,
    "hybrid_manifest_sha256":"2"*64,"assessment_sha256":"3"*64,"count":50000,
    "technical_passed":True,"physical_status":"model_failure"},
    "details":{"allocation":{"actual_gpu_seconds":209,"gpus":1,"restarts":0,
      "ledger_state":"reconciled","no_requeue_verified_from_submitted_argv":True}}})
 return records

def test_complete_all_failures_is_still_complete_receipt():
 d=a.completion_document(audits(),{"reserved_gpu_seconds":0,"cap_exceeded":False},"4"*64)
 assert d["status"]=="complete" and d["point_count"]==24
 assert d["physical_accuracy_gate_passed"] is False and d["physical_failure_count"]==24
 assert d["completed_pairs"]==[[s,c] for s in a.SEEDS for c in a.CASES]
 assert all(set(p)=={"context_seed","case","parent_manifest_sha256","hybrid_manifest_sha256",
  "assessment_sha256","count","technical_passed","physical_status"} for p in d["points"])

@pytest.mark.parametrize("change",[
 lambda r:r.pop(),lambda r:r.append(r[-1]),lambda r:r.reverse(),
 lambda r:r[0]["point"].update(count=49999),lambda r:r[0]["point"].update(technical_passed=False),
 lambda r:r[0]["point"].update(physical_status="ignored"),lambda r:r[0]["details"]["allocation"].update(restarts=1),
 lambda r:r[0]["details"]["allocation"].update(gpus=2)])
def test_incomplete_changed_or_restarted_grid_refused(change):
 rows=audits();change(rows)
 with pytest.raises(ValueError):a.completion_document(rows,{"reserved_gpu_seconds":0,"cap_exceeded":False},"4"*64)

@pytest.mark.parametrize("summary",[{"reserved_gpu_seconds":1200,"cap_exceeded":False},
                                     {"reserved_gpu_seconds":0,"cap_exceeded":True}])
def test_held_or_overcap_ledger_refused(summary):
 with pytest.raises(ValueError):a.completion_document(audits(),summary,"4"*64)

def test_read_ledger_has_no_lock_write_or_reconcile(tmp_path,monkeypatch):
 parent,ledger,record=allocation_inputs()
 path=tmp_path/"ledger.json";path.write_text(json.dumps(ledger));before=path.read_bytes()
 def forbidden(*args,**kwargs):raise AssertionError("mutating BudgetLedger API called")
 for name in ("_locked","_save","snapshot","reconcile","reconcile_all"):monkeypatch.setattr(budget.BudgetLedger,name,forbidden,raising=False)
 read,summary,digest=a.read_ledger(path)
 assert summary["actual_gpu_seconds"]==209 and summary["reserved_gpu_seconds"]==0 and len(digest)==64
 assert path.read_bytes()==before and list(tmp_path.iterdir())==[path]
 assert a.audit_allocation(parent,read,record)["actual_gpu_seconds"]==209

@pytest.mark.parametrize("change",[
 lambda p,l,r:r.update(Restarts="1"),lambda p,l,r:r.update(AllocTRES="cpu=16,gres/gpu=2"),
 lambda p,l,r:r.update(State="FAILED"),lambda p,l,r:r.update(ElapsedRaw="210"),
 lambda p,l,r:l["reservations"][0].update(state="submitted"),
 lambda p,l,r:l["reservations"][0]["submitted_argv"].remove("--no-requeue"),
 lambda p,l,r:l["reservations"][0].update(actual_gpu_seconds=208),
 lambda p,l,r:p["budget_binding"].update(reservation_id="other")])
def test_actual_allocation_invariant_mutations_fail(change):
 p,l,r=allocation_inputs();change(p,l,r)
 with pytest.raises(ValueError):a.audit_allocation(p,l,r)

def test_scheduler_parser_ignores_steps_and_rejects_duplicates():
 line="12345|eu-26-17|qgpu|COMPLETED|209|cpu=16,gres/gpu=1|0\n"
 assert list(a.parse_scheduler(line+"12345.batch|eu-26-17|qgpu|COMPLETED|208||0\n"))==["12345"]
 with pytest.raises(ValueError):a.parse_scheduler(line+line)

def test_first_corrected_paths_not_prototype_and_current_othercase0(tmp_path):
 p,h,d=a.point_paths(tmp_path,20261111,0)
 assert p==tmp_path/"sigma-native-pilot-01/manifest.json"
 p,h,d=a.point_paths(tmp_path,20261112,0)
 assert p==tmp_path/"sigma-native-grid-01/seed-20261112-case-0/manifest.json"
 with pytest.raises(ValueError):a.point_paths(tmp_path,20261114,0)

def test_missing_point_cannot_be_closed(tmp_path):
 with pytest.raises(ValueError,match="not fully closed"):
  a.audit_point(20261111,0,phase=tmp_path,ledger=tmp_path/"absent.json")
