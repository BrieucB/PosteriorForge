"""Synthetic CPU proof-composition tests; fixtures are never inference evidence."""
from copy import deepcopy
import ast,importlib.util,json,shutil,subprocess,sys
from pathlib import Path
import pytest
from posteriorforge.artifacts import write_json,sha256_file

ROOT=Path(__file__).parents[1]
def module(path,name):
 spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value
r=module(ROOT/"scripts/repair_ss15_conditionals.py","repair_tests")
fixture=module(ROOT/"tests/test_ss15_conditional_qualification.py","old_test_fixture_only")

def test_private_clone_is_cpu_only_with_exact_declared_reverse_ast_and_stage_namespace(tmp_path):
 subprocess.run([sys.executable,"-c","import importlib.util,sys;s=importlib.util.spec_from_file_location('r','scripts/repair_ss15_conditionals.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.private_stage_collector();m.source_pins();assert 'torch' not in sys.modules"],cwd=ROOT,check=True)
 function,description=r.private_stage_collector()
 assert description["reverse_AST_exact"] and description["declared_edit_counts"]["stage_loop"]==1
 assert description["declared_edit_counts"]["declared_fallback_argument"]==2
 assert function.__globals__["SCHEMA"]==r.STAGE_SCHEMA
 assert function.__globals__["FALLBACK_MAP"]==r.MAP and function.__globals__["_REPAIR_INT_MAP"]==r.INT_MAP
 assert function.__globals__["validate_qualification"] is not r.ancestor().validate_qualification
 path=tmp_path/"manifest.json"
 function.__globals__["write_json"](path,{"schema":r.STAGE_SCHEMA,"stage_results":[]})
 data=json.loads(path.read_text())
 assert data["requested_stages"]==[20] and data["completed_stages"]==[]
 assert not data["full21_qualified"] and not data["matching_FP32_execution_passed_all21"]

def synthetic_components():
 q=r.ancestor();old=fixture.document();old["status"]="ss15_matching_FP32_structural_failure"
 old["sources"]=q.source_pins();old["matching_FP32_execution_passed_all21"]=False;old["global_RNG_preserved"]=True
 for record in old["stage_results"]:
  record.update(RNG_identity_before={"CPU":"synthetic-engineering-fixture"},RNG_identity_after={"CPU":"synthetic-engineering-fixture"})
 old.update(prefix_manifest_sha256="prefix-m",prefix_samples_sha256="prefix-p",fixed_prefix_row_indices=list(range(1024)))
 old["compatibility_adapter"]["sdk_aliases_unchanged"]=True
 failure=old["stage_results"][20]
 bad=failure["comparisons"][q.COMPARISON_KEYS[0]]["1024"];bad["final"]["max_cdf_absolute_error"]=3e-5;bad["passed"]=False
 failure.update(q.stage_decision(failure))
 row=fixture.stage(20)
 row.update(RNG_identity_before={"CPU":"synthetic-engineering-fixture"},RNG_identity_after={"CPU":"synthetic-engineering-fixture"})
 report=row["precision_sessions"]["FP16"]
 for witness in report["witnesses"]:
  if witness["member"]==4:
   witness["autocast_enabled"]=False;witness["projection_dtypes"]["qkv"]="torch.float32"
 stage={key:deepcopy(old[key]) for key in ("raw_bindings","transformed_bindings","representation_descriptor",
  "thresholds","row_counts","stress_rows","prefix_manifest_sha256","prefix_samples_sha256","fixed_prefix_row_indices")}
 stage.update(schema=r.STAGE_SCHEMA,status="ss15_conditional_stage_screen_pass",sources=r.source_pins(),fallback_map=r.MAP,
  stage_results=[row],sources_unchanged=True,global_RNG_preserved=True,posterior_vectors_generated=0,reference_integrations=0,
  matching_FP32_execution_passed_all21=False,FP16_hard_validity_passed_all21=False,
  matching_FP32_execution_passed_stage=True,FP16_hard_validity_passed_stage=True,
  requested_stages=[20],completed_stages=[20],full21_qualified=False,private_collector=r.private_stage_collector()[1],
  compatibility_adapter={**deepcopy(old["compatibility_adapter"]),"patched_call_count":112})
 return old,stage

def create(tmp_path,monkeypatch):
 old,stage=synthetic_components()
 original=tmp_path/"original.json";proof=tmp_path/"stage20.json"
 write_json(original,old);write_json(proof,stage)
 monkeypatch.setattr(r,"ORIGINAL_SHA",sha256_file(original)) # Synthetic provenance, not real5dc.
 return original,proof

def verify(document,root):
 return r.validate_qualification(document,raw_bindings={"raw":"sha"},transformed_bindings={"features":"sha"},
  representation_descriptor={"module":"sha"},expected_sources=r.source_pins(),fallback_map=r.MAP,artifact_root=root)

def test_composition_preserves_failed_input_and_selects_only_new_stage20_with_distinct_contexts(tmp_path,monkeypatch):
 original,proof=create(tmp_path,monkeypatch);before=original.read_bytes();output=tmp_path/"assembled"
 result=r.assemble(original=original,stage_proof=proof,output=output)
 assert original.read_bytes()==before and json.loads(before)["status"]=="ss15_matching_FP32_structural_failure"
 assert (output/"inputs/original-failed-manifest.json").read_bytes()==before
 assert len(result["stage_results"])==21 and result["matching_FP32_execution_passed_all21"]
 assert result["selected_stage_sources"][-1]["component"]=="stage20"
 assert all(row["component"]=="original_failed" for row in result["selected_stage_sources"][:-1])
 assert result["corrected_CDF_contexts"][0]["actual_observed_CDF_calls"]==2352
 assert result["corrected_CDF_contexts"][0]["selected_stage_CDF_calls"]==2240
 assert result["corrected_CDF_contexts"][1]["actual_observed_CDF_calls"]==112
 assert result["selected_stage_CDF_calls"]==2352 and "compatibility_adapter" not in result
 assert verify(result,output)["original_status_preserved"]
 relocated=tmp_path/"relocated";shutil.copytree(output,relocated)
 assert verify(json.loads((relocated/"manifest.json").read_text()),relocated)["passed"]

@pytest.mark.parametrize("fault",["inputbytes","row","source","map","fakefooter","CDF","status","provenance","FP16","missing"])
def test_composite_custody_and_predicate_tampering_is_rejected(tmp_path,monkeypatch,fault):
 original,proof=create(tmp_path,monkeypatch);output=tmp_path/"assembled"
 value=r.assemble(original=original,stage_proof=proof,output=output)
 if fault=="inputbytes":
  path=output/"inputs/original-failed-manifest.json";path.write_bytes(path.read_bytes()+b"\n")
 elif fault=="row":value["stage_results"][1]["corrected_CDF_calls"]=111
 elif fault=="source":value["sources"]["execution"]="wrong"
 elif fault=="map":value["fallback_map"]={"10":[5]}
 elif fault=="fakefooter":value["compatibility_adapter"]=value["corrected_CDF_contexts"][0]["descriptor"]
 elif fault=="CDF":value["corrected_CDF_contexts"][0]["selected_stage_CDF_calls"]=2352
 elif fault=="status":value["components"]["original_failed"]["producer_status"]="ss15_conditional_screen_pass"
 elif fault=="provenance":value["selected_stage_sources"][19]["component"]="stage20"
 elif fault=="FP16":value["FP16_strict_failed_stages"]=[]
 else:value["stage_results"].pop()
 with pytest.raises(ValueError):verify(value,output)

def test_stage20_cannot_claim_full21_or_skip_original_f32_precision(tmp_path):
 old,stage=synthetic_components()
 kwargs=dict(raw_bindings=old["raw_bindings"],transformed_bindings=old["transformed_bindings"],
  representation_descriptor=old["representation_descriptor"],expected_sources=r.source_pins(),fallback_map=r.MAP)
 assert r.validate_stage(stage,**kwargs)["passed"]
 for key in ("full21_qualified","matching_FP32_execution_passed_all21","FP16_hard_validity_passed_all21"):
  changed=deepcopy(stage);changed[key]=True
  with pytest.raises(ValueError):r.validate_stage(changed,**kwargs)
 changed=deepcopy(stage)
 for w in changed["stage_results"][0]["precision_sessions"]["FP16"]["witnesses"]:
  if w["member"]==4:w["autocast_enabled"]=True;w["projection_dtypes"]["qkv"]="torch.float16"
 with pytest.raises(ValueError):r.validate_stage(changed,**kwargs)

def test_real_failed_source_is_recognized_without_promoting_or_mutating():
 path=Path("/scratch/project/eu-26-17/eubrieucb/posteriorforge/results/accuracy-ss15-v1/conditional-qualification-02/manifest.json")
 if not path.is_file():pytest.skip("actual source artifact unavailable")
 value=json.loads(path.read_bytes());before=path.read_bytes()
 assert sha256_file(path)==r.ORIGINAL_SHA and value["status"]=="ss15_matching_FP32_structural_failure"
 failed=[]
 for row in value["stage_results"]:
  decision=r.check_row(row,{"10":[5]},must_pass=False)
  if not decision["matching_FP32_execution_passed"]:failed.append(row["stage_index"])
 assert failed==[20] and path.read_bytes()==before
 with pytest.raises(ValueError):r.compose(value,value)


@pytest.mark.parametrize("field",["global_RNG_preserved","requested_stages","completed_stages","private_collector"])
def test_stage_replay_requires_observed_rng_and_explicit_partial_scope(field):
 old,stage=synthetic_components()
 stage[field]=False if field=="global_RNG_preserved" else [] if field!="private_collector" else {}
 with pytest.raises(ValueError):
  r.validate_stage(stage,raw_bindings=old["raw_bindings"],transformed_bindings=old["transformed_bindings"],
   representation_descriptor=old["representation_descriptor"],expected_sources=r.source_pins(),fallback_map=r.MAP)
