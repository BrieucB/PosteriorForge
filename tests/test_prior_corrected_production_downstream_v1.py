"""Closed production family/normal footer and one saved-G3 binding tests."""
from copy import deepcopy
import importlib.util,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from posteriorforge import prior_coordinates_v1 as codec,conditional_globals as kernel
from posteriorforge.science.analytical_benchmark_v2 import array_descriptor
from posteriorforge.artifacts import sha256_file
ROOT=Path(__file__).resolve().parents[1]
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
w=load("_corrected_production_wrapper_tests",ROOT/"scripts/refresh_prior_corrected_production_v1.py")
e=load("_corrected_production_eval_tests",ROOT/"scripts/evaluate_prior_corrected_production_v1.py")
observer=w.load("assess_prior_forensic_replay_v1.py",w.DRAW_OBSERVER_SOURCE_SHA256)

def footer():
 rng={k:"same" for k in ("numpy","python","torch_CPU","torch_CUDA")}
 return {"schema":w.PRODUCER_SCHEMA,"variant":w.PRODUCER_VARIANT,"status":"complete",
  "runner_source_sha256":w.PRODUCER_SHA256,"context_seed":20261111,"case":2,"context_size":4096,"requested_count":50000,
  "repetition":0,"rank":0,"world_size":1,"mode":"fp32_transparent_control_v1","native_all8_stages":[6,10],"FP16_activated":False,
  "sources_unchanged":True,"default_rng_preserved":True,"completed_stage_indices":list(range(21)),
  "samples_in_original_physical_order_before_FIFO_support":True,"all_generated_rows_strictly_decoded_including_unused_FIFO_tail":True,
  "model_coordinate_decode_after_global_refresh":False,"sampler_math_changed":False,"partial_vector_filtering":False,
  "clipping_or_repair_performed":False,"physical_log_density_claimed":False,"learned_targets_are_original_physical_values":False,
  "fallback_map":{},"fallback_map_scope":"INNER frozen88 precision owner only; outer native all8 stages6/10 are explicit",
  "matching_cached_native_FP32_execution_passed_all21":True,"fresh_same_policy_all21_screen_replayed":True,
  "actual_fixed_conditional_screen_performed_during_request":False,"default_rng_before":rng.copy(),"default_rng_after":rng.copy(),
  "matching_conditional_qualification_sha256":w.FINITE_SCREEN_SHA256,"execution_counters":{},"sampling_counters":{},
  "actual_prediction_checks":{"actual_head_contexts":1138,"sampler_callback_calls":1131,"shared_densities":1},
  "timings":{"posterior_request_seconds":100.,"includes_stage_preparation":True,"includes_CPU_sampling":True,"includes_strict_all_generated_wave_decode":True},
  "memory":{"peak_allocated_bytes":100,"peak_reserved_bytes":200},"expected_patched_call_count":1138,
  "compatibility_adapter":{"original_alias_restored":True,"sdk_sources_unchanged":True,"sdk_aliases_unchanged":True,"patched_call_count":1138}}

def test_exact_recorded_counts_footer_has_no_invented_pass_field():
 d=footer();assert "passed" not in d["actual_prediction_checks"]
 assert w.validate_success_header(d)["closed_successful_production_request"]

@pytest.mark.parametrize("fault",["forensic","oldfailed","case","count","RNGmissing","RNGchanged","source","engine","CDF","timing","memory","headcount","stage","FP16","fixture"])
def test_closed_production_footer_and_family_failclosed(fault):
 d=footer()
 if fault=="forensic":d["schema"]="posteriorforge.prior_coordinate_forensic_cohort.v1";d["status"]="forensic_complete"
 elif fault=="oldfailed":d["schema"]="posteriorforge.inference.prior_coordinates_native.v1";d["status"]="failed"
 elif fault=="case":d["case"]=0
 elif fault=="count":d["requested_count"]=49999
 elif fault=="RNGmissing":d.pop("default_rng_after")
 elif fault=="RNGchanged":d["default_rng_after"]["torch_CUDA"]="changed"
 elif fault=="source":d["runner_source_sha256"]="0"*64
 elif fault=="engine":d.pop("execution_counters")
 elif fault=="CDF":d["compatibility_adapter"]["original_alias_restored"]=False
 elif fault=="timing":d["timings"].pop("posterior_request_seconds")
 elif fault=="memory":d["memory"]["peak_allocated_bytes"]=0
 elif fault=="headcount":d["actual_prediction_checks"]["sampler_callback_calls"]-=1
 elif fault=="stage":d["completed_stage_indices"].pop()
 elif fault=="fixture":d["test_fixture"]={"mocked":"neural telemetry"}
 else:d["FP16_activated"]=True
 with pytest.raises(ValueError):w.validate_success_header(d)

def hybrid_fixture(tmp_path,monkeypatch):
 monkeypatch.setattr(e,"COUNT",17)
 monkeypatch.setattr(w,"verify_git_sources",lambda *args,**kwargs:{"passed":True})
 before=codec.decode_ordered(np.zeros((17,21),np.float32))
 seed=e.derive_seed(20261111,w.ROLE,case=2,repetition=0);after,z,receipt=observer.refresh_with_actual_draws(before,seed)
 np.savez(tmp_path/"samples.npz",samples=after);np.savez(tmp_path/"Z.npz",standard_normal_draws=z)
 m={"schema":w.SCHEMA,"status":"complete","forensic":False,"original_parent_status":"complete",
  "parent":{"parent_manifest_sha256":"parent"},"source_pins":{"actual":"pins","git_revision":"a"*40,"wrapper":"b"*64},"conditional_kernel":kernel.metadata(),
  "coordinate_decode_after_global_refresh":False,"samples_file":"samples.npz","samples_sha256":sha256_file(tmp_path/"samples.npz"),
  "actual_draws_file":"Z.npz","actual_draws_sha256":sha256_file(tmp_path/"Z.npz"),"samples":array_descriptor(after),"refresh":receipt}
 return before,m

def auth(tmp_path,before,m):
 p=tmp_path/"manifest.json";p.write_text(json.dumps(m))
 return e.authenticate_hybrid(p,sha256_file(p),"parent",before,w,{"actual":"pins","git_revision":"c"*40,"wrapper":"b"*64})

def test_saved_actualZ_production_G3_auth_no_reroll(tmp_path,monkeypatch):
 before,m=hybrid_fixture(tmp_path,monkeypatch)
 def forbidden(*args,**kwargs):raise AssertionError("assessment must never draw another actual normal matrix")
 monkeypatch.setattr(observer.RecordingGenerator,"standard_normal",forbidden)
 _,after=auth(tmp_path,before,m)
 assert after[:,3:].tobytes()==before[:,3:].tobytes()

@pytest.mark.parametrize("fault",["forensic","oldparent","role","seed","initialstate","defaultRNG","physical18","kernel","source","Zdescriptor"])
def test_saved_G3_prod_source_seed_draw_and_physical18_failclosed(tmp_path,monkeypatch,fault):
 before,m=hybrid_fixture(tmp_path,monkeypatch)
 if fault=="forensic":m["forensic"]=True
 elif fault=="oldparent":m["original_parent_status"]="failed"
 elif fault=="role":m["refresh"]["role"]="different"
 elif fault=="seed":m["refresh"]["seed"]+=1
 elif fault=="initialstate":m["refresh"]["initial_private_state"]["state"]["state"]+=1
 elif fault=="defaultRNG":m["refresh"]["default_numpy_python_RNG_after_sha256"]="changed"
 elif fault=="physical18":
  with np.load(tmp_path/"samples.npz") as a:rows=a["samples"]
  rows[0,3]+=1;np.savez(tmp_path/"samples.npz",samples=rows)
  m["samples_sha256"]=sha256_file(tmp_path/"samples.npz");m["samples"]=array_descriptor(rows)
 elif fault=="kernel":m["conditional_kernel"]["module_sha256"]="0"*64
 elif fault=="source":m["source_pins"]={"different":"pins"}
 else:m["refresh"]["standard_normal_draws"]["sha256"]="0"*64
 with pytest.raises(ValueError):auth(tmp_path,before,m)

def test_production_labels_focus_and_reused_math_origins():
 assert e.FOCUS==(0,7,8,10) and not any("FORENSIC" in x for x in e.LABELS)
 assert "OLD physical-target" in e.LABELS[0] and "all8 original at6/10" in e.LABELS[1]
 assert w.KERNEL_SHA256==kernel.source_sha256() and w.DRAW_OBSERVER_SOURCE_SHA256==sha256_file(ROOT/"scripts/assess_prior_forensic_replay_v1.py")


def test_actual_CPU_success_fixture_never_qualifies_as_production_when_available():
 path=Path("/scratch/project/eu-26-17/eubrieucb/posteriorforge/results/accuracy-prior-coordinates-v1/physical-fifo-archived-fixture-cpu-01/inference-pytest-temp/test_archived_neural_fixture_r0/EXPLICIT_CPU_ARCHIVED_NEURAL_TEST_FIXTURE/manifest.json")
 if not path.exists():pytest.skip("independent Karolina test-only completed fixture unavailable")
 assert sha256_file(path)=="5238d0d9745d7d5ef4248483d0f3d518a50e5b57a116d3d95be7ceef8ef99bdd"
 d=json.loads(path.read_bytes())
 assert "passed" not in d["actual_prediction_checks"]
 assert set(d["actual_prediction_checks"])=={"actual_head_contexts","sampler_callback_calls","shared_densities"}
 assert d["schema"]==w.PRODUCER_SCHEMA and d["status"]=="complete"
 assert "execution_policy" in d and "execution_engine" not in d
 with pytest.raises(ValueError):w.validate_success_header(d)


def test_immutable_pin_comparison_allows_later_evidence_commit():
 assert w.immutable_pins({"wrapper":"same","git_revision":"a"*40})==w.immutable_pins({"wrapper":"same","git_revision":"c"*40})

def test_recorded_git_revision_and_actual_source_bytes_when_available():
 root=ROOT
 import subprocess
 rev=subprocess.run(["git","-C",str(root),"rev-parse","HEAD"],check=True,capture_output=True,text=True).stdout.strip()
 expected={"src/posteriorforge/performance_prior_physical_fifo_v1.py":w.PRODUCER_SHA256}
 assert w.verify_git_sources(root,rev,expected)["passed"]
 with pytest.raises(ValueError):w.verify_git_sources(root,"not-a-revision",expected)
 with pytest.raises(ValueError):w.verify_git_sources(root,rev,{next(iter(expected)):"0"*64})


def execution_policy():
 return {"module_sha256":"a"*64,"routing":{"native":[6,10],"cached_stages":19},
  "runtime_source_inventory":{"inventory_sha256":"b"*64,"committed_HEAD_exact":True},
  "derivation":{"run_code_sha256":"c"*64,"reverse_AST_exact":True,"original_function_globals_unmodified":True,
   "original_neural_engine_codec_stage_draw_objects_used":True,"declared_AST_changes":["F64 completion","failure footer"]}}

def test_process_local_serialized_fingerprint_difference_preserved_without_data_mutation():
 left=execution_policy();right=deepcopy(left);right["derivation"]["run_code_sha256"]="d"*64
 original=deepcopy(left);current=deepcopy(right)
 result=w.verify_execution_policy(left,right)
 assert result["passed"] and result["all_other_metadata_fields_exact"] and not result["serialized_code_fingerprints_equal"]
 assert result["recorded_execution_serialized_code_SHA256"]=="c"*64
 assert result["current_verifier_serialized_code_SHA256"]=="d"*64
 assert left==original and right==current and not result["original_producer_manifest_changed"]

@pytest.mark.parametrize("fault",["recorded_hash","current_hash","source","source_inventory","routing","AST","globals","type"])
def test_only_wellformed_fingerprint_telemetry_may_differ(fault):
 left=execution_policy();right=deepcopy(left);right["derivation"]["run_code_sha256"]="d"*64
 if fault=="recorded_hash":left["derivation"]["run_code_sha256"]="invalid"
 elif fault=="current_hash":right["derivation"]["run_code_sha256"]="x"*64
 elif fault=="source":left["module_sha256"]="e"*64
 elif fault=="source_inventory":left["runtime_source_inventory"]["inventory_sha256"]="e"*64
 elif fault=="routing":left["routing"]["native"]=[14]
 elif fault=="AST":left["derivation"]["reverse_AST_exact"]=False
 elif fault=="type":left["routing"]["cached_stages"]=19.0
 else:left["derivation"]["original_function_globals_unmodified"]=False
 with pytest.raises(ValueError):w.verify_execution_policy(left,right)


def test_actual_closed_GPU_manifest_and_fresh_process_policy_regression(tmp_path):
 import os,subprocess,sys
 parent=Path("/scratch/project/eu-26-17/eubrieucb/posteriorforge/results/accuracy-prior-coordinates-v1/native-pilot-corrected-01/manifest.json")
 interpreter=Path("/scratch/project/eu-26-17/eubrieucb/posteriorforge/envs/inference/bin/python")
 if not parent.is_file() or not interpreter.exists():pytest.skip("actual closed Karolina production metadata/interpreter unavailable")
 code="""from pathlib import Path
import hashlib,importlib.util,json,sys
path=Path('/scratch/project/eu-26-17/eubrieucb/posteriorforge/results/accuracy-prior-coordinates-v1/native-pilot-corrected-01/manifest.json')
raw=path.read_bytes()
assert hashlib.sha256(raw).hexdigest()=='62aa8bf71ddbf922ab9543141c4ada5cb438e5af162ede76cffe60d1dd3819b5'
d=json.loads(raw)
from posteriorforge.artifacts import versions
actual_versions={k:versions()[k] for k in ('python','numpy','scipy')}
assert actual_versions=={k:d['versions'][k] for k in actual_versions}
spec=importlib.util.spec_from_file_location('fresh_consumer','scripts/refresh_prior_corrected_production_v1.py')
w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
from posteriorforge import performance_prior_physical_fifo_v1 as p
before=p._RUN_BODY.__code__
fresh=p.execution_metadata()
differences=w.policy_differences(d['execution_policy'],fresh)
result=w.verify_execution_policy(d['execution_policy'],fresh)
expected_sources=dict(fresh['runtime_source_inventory']['sources'])
expected_sources['src/posteriorforge/performance_prior_physical_fifo_v1.py']=p.source_sha256()
recorded_sources=w.verify_git_sources(Path.cwd(),d['git_revision'],expected_sources)
p.require_sources()
assert p._RUN_BODY.__code__ is before and 'torch' not in sys.modules
assert all(x['path']=='.derivation.run_code_sha256' and x.get('recorded_type')==x.get('current_type')=='str' for x in differences)
assert path.read_bytes()==raw
print(json.dumps(dict(actual_closed_manifest_sha256=hashlib.sha256(raw).hexdigest(),actual_status=d['status'],actual_field_value_TYPE_differences=differences,policy_authentication=result,producer_source_sha256=p.source_sha256(),same_runtime_versions=actual_versions,recorded_executed_Git_source_inventory=recorded_sources,same_code_object_identity_preserved=True,Torch_imported=False,new_neural_G3_GPU_calls=0),sort_keys=True))
"""
 result=subprocess.run([str(interpreter),"-c",code],cwd=ROOT,env=dict(os.environ,PYTHONPATH="src",CUDA_VISIBLE_DEVICES=""),capture_output=True,text=True,check=True)
 report=json.loads(result.stdout)
 assert report["policy_authentication"]["passed"] and report["same_code_object_identity_preserved"]
 assert report["new_neural_G3_GPU_calls"]==0 and not report["Torch_imported"]
 (tmp_path/"actual-closed-fresh-process-metadata-regression.json").write_text(result.stdout)
 print("ACTUAL_CLOSED_FRESH_PROCESS_REGRESSION="+result.stdout.strip())
