"""CPU-only proof of private single-stage continuation and partial labeling."""
from copy import deepcopy
import importlib.util,json,types
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
def imported(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
w=imported("_map2_inventory_tests",ROOT/"scripts/continue_ss15_sigma_first_map2.py")
fixtures=imported("_sigma_row_fixtures",ROOT/"tests/test_ss15_sigma_first_conditionals.py")
original=w.frozen_helper()
FAILED=Path("/scratch/project/eu-26-17/eubrieucb/posteriorforge/results/accuracy-ss15-v1/sigma-first-conditionals-01/manifest.json")

def failed():
 if not FAILED.is_file():pytest.skip("actual Karolina failed proof unavailable")
 d=json.loads(FAILED.read_bytes())
 assert w.sha256_file(FAILED)==w.FAILED_SHA
 return d

def child(stage,passed=True):
 d={k:deepcopy(v) for k,v in failed().items()}
 r=fixtures.row(stage)
 if not passed:
  r["comparisons"][original.ARMS[0]]["1024"]=fixtures.fixtures.comparison(False)
  r.update(original.stage_decision(r,fixtures.old))
 r["RNG_identity_before"]={"same":"state"};r["RNG_identity_after"]={"same":"state"}
 d.update(schema=original.SCHEMA,status="sigma_first_conditional_screen_pass" if passed else "sigma_first_matching_FP32_structural_or_hard_failure",
  stage_results=[r],matching_FP32_execution_passed_all21=passed,FP16_hard_validity_passed_all21=True,
  sources_unchanged=True,global_RNG_preserved=True)
 d["compatibility_adapter"]["patched_call_count"]=72
 return d

def test_real_run_code_and_original_globals_are_preserved():
 before={k:getattr(original,k) for k in ("STAGES","write_json","validate_screen","run")}
 f=w.private_run(original,16,failed())
 assert f.__code__ is original.run.__code__ and f.__globals__ is not original.run.__globals__
 assert f.__globals__["STAGES"]==(16,)
 assert all(getattr(original,k) is v for k,v in before.items())
 with pytest.raises(ValueError):w.private_run(original,14,failed())

def test_writer_copies_and_never_claims_full21_even_for_a_passing_child(tmp_path):
 f=w.private_run(original,15,failed());raw=child(15);saved=deepcopy(raw)
 path=tmp_path/"manifest.json";f.__globals__["write_json"](path,raw)
 d=json.loads(path.read_bytes())
 assert d["schema"]==w.CHILD_SCHEMA and d["selected_stage_complete"]
 assert d["selected_stage_matching_FP32_execution_passed"]
 assert not d["full21_qualified"] and not d["matching_FP32_execution_passed_all21"] and not d["FP16_hard_validity_passed_all21"]
 assert raw==saved

def test_no_premature_complete_before_actual_CDF_and_RNG_footer():
 d=child(15);d["status"]="running";d.pop("compatibility_adapter")
 partial=w.partial_document(d,15)
 assert partial["selected_stage_numerical_row_complete"] and not partial["selected_stage_complete"]
 assert partial["status"]=="partial_stage_running" and not partial["full21_qualified"]

def test_real_numeric_failure_is_preserved_and_is_not_a_hard_failure():
 d=child(16,False);r=w.validate_child(d,16,original,failed())
 assert r["passed"] and not r["matching_FP32_execution_passed"] and not r["full21_qualified"]
 assert w.failure_locations(d["stage_results"][0],original)

@pytest.mark.parametrize("fault",["source","target","mask","mass","CDF","RNG","restoration"])
def test_hard_state_or_binding_failure_blocks_valid_inventory(fault):
 d=child(15)
 if fault=="source":d["sources"]["producer"]="wrong"
 elif fault=="target":d["stage_results"][0]["target_physical_index"]=0
 elif fault=="mask":d["stage_results"][0]["mask_inputs"]["training"]["mask"]["feature_columns_zeroed"]=[]
 elif fault=="mass":d["stage_results"][0]["hard_predictions"]["candidate"]["raw_head_masses"][0]["max"]=1.1
 elif fault=="CDF":d["compatibility_adapter"]["patched_call_count"]=71
 elif fault=="RNG":d["global_RNG_preserved"]=False
 else:d["compatibility_adapter"]["original_alias_restored"]=False
 with pytest.raises(ValueError):w.validate_child(d,15,original,failed())

def test_outer_loop_collects_all6_even_when_numeric_only_children_fail(tmp_path,monkeypatch):
 # Synthetic engineering execution only; no real model calls in this test.
 def fake_run(*,banks,prefix_run,checkpoint,output,ledger,max_seconds):
  stage=STAGES[0]
  p=Path(output);p.mkdir(parents=True)
  d=child(stage,stage not in (16,19))
  write_json(p/"manifest.json",d)
  return d
 fake=types.SimpleNamespace(**{k:getattr(original,k) for k in ("STAGES","write_json","validate_screen",
  "load","stage_decision","SS15_QUALIFIER_SHA","_execution_modes","FALLBACK_MAP","ORDER","ARMS","SCHEMA")},run=fake_run)
 monkeypatch.setattr(w,"frozen_helper",lambda:fake)
 monkeypatch.setattr(w,"source_pins",lambda unused=None:{"wrapper":"CPU-synthetic-engineering-test"})
 out=tmp_path/"inventory"
 report=w.run(banks="unused",prefix_run="unused",checkpoint="unused",ledger="unused",
  failed_manifest=FAILED,output=out,max_seconds=600)
 assert report["status"]=="partial_inventory_complete"
 assert report["selected6_matching_FP32_failed_stages"]==[16,19]
 assert report["deterministic_native_stage_inventory"]==[14,16,19]
 assert len(report["corrected_CDF_contexts"])==6 and report["actual_corrected_CDF_call_count"]==432
 assert not report["matching_FP32_execution_passed_all21"] and not report["full21_qualified"]
 assert all(not json.loads((out/c["file"]).read_bytes())["full21_qualified"] for c in report["children"])

def test_actual_frozen_source_preflight_is_Torch_free():
 import subprocess,sys
 subprocess.run([sys.executable,"-c","import importlib.util,sys;s=importlib.util.spec_from_file_location('w','scripts/continue_ss15_sigma_first_map2.py');w=importlib.util.module_from_spec(s);s.loader.exec_module(w);w.source_pins();assert 'torch' not in sys.modules"],cwd=ROOT,check=True)
