"""Forensic dtype integration, exact actual-Z observation and missing-footer semantics."""
from copy import deepcopy
import importlib.util,json
from pathlib import Path
import numpy as np
import pytest
from posteriorforge import prior_coordinates_v1 as codec,conditional_globals as kernel
from posteriorforge.artifacts import sha256_file
from posteriorforge.science.analytical_benchmark_v2 import array_descriptor
ROOT=Path(__file__).resolve().parents[1]
def imported(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
r=imported("_forensic_replay_tests",ROOT/"scripts/replay_prior_forensic_cohort_v1.py")
a=imported("_forensic_assess_tests",ROOT/"scripts/assess_prior_forensic_replay_v1.py")

def physical(n):return codec.decode_ordered(np.zeros((n,21),np.float32))

def test_missing_failed_request_fields_remain_absent_or_exactly_recorded():
 doc={"status":"failed","default_rng_preserved":None,"sources_unchanged":False}
 result=r.missing_footer_observations(doc)
 assert result["execution_counters"]=={"present":False,"recorded_value":None}
 assert result["default_rng_preserved"]=={"present":True,"recorded_value":None}
 assert result["sources_unchanged"]=={"present":True,"recorded_value":False}
 assert "actual_prediction_checks" in result and "expected_patched_call_count" in result

def test_recording_generator_exact_ordinary_matrix_state_and_no_default_RNG():
 first=np.random.Generator(np.random.PCG64(903));second=a.RecordingGenerator(np.random.PCG64(903))
 default=a.state_hash();expected=first.standard_normal((17,3));observed=second.standard_normal((17,3))
 assert observed.tobytes()==expected.tobytes()==second.actual_draws.tobytes()
 assert second.actual_draws is not observed and first.bit_generator.state==second.bit_generator.state
 assert second.standard_normal_calls==1 and a.state_hash()==default
 before=deepcopy(second.bit_generator.state)
 with pytest.raises(ValueError,match="single actual"):second.standard_normal((1,3))
 assert second.bit_generator.state==before

def test_one_unchanged_kernel_call_actual_Z_and_physical18_identity():
 values=physical(17);before=values.tobytes()
 expected=kernel.refresh(values,np.random.Generator(np.random.PCG64(278)))
 result,normals,receipt=a.refresh_with_actual_draws(values,278)
 assert result.tobytes()==expected.tobytes() and values.tobytes()==before
 assert result[:,3:].tobytes()==values[:,3:].tobytes()
 assert normals.shape==(17,3) and receipt["standard_normal_method_calls"]==1
 assert receipt["standard_normal_scalar_count"]==51 and receipt["draws_are_actual_returned_matrix_not_rerolled"]
 assert receipt["global_block_refresh_count_per_row"]==1 and not receipt["coordinate_decode_after_global_refresh"]

def hybrid_fixture(tmp_path,monkeypatch):
 monkeypatch.setattr(a,"COUNT",17)
 before=physical(17);seed=a.derive_seed(20261111,a.ROLE,case=2,repetition=0)
 after,normals,receipt=a.refresh_with_actual_draws(before,seed)
 np.savez(tmp_path/"samples.npz",samples=after);np.savez(tmp_path/"Z.npz",standard_normal_draws=normals)
 m={"schema":a.G3_SCHEMA,"status":"forensic_complete","source_pins":{"actual":"sources"},
  "parent_manifest_sha256":"parent","original_neural_request_status":"failed",
  "production_qualification_claimed":False,"original_missing_footers_recovered":False,
  "conditional_kernel":kernel.metadata(),"samples_file":"samples.npz","samples_sha256":sha256_file(tmp_path/"samples.npz"),
  "actual_draws_file":"Z.npz","actual_draws_sha256":sha256_file(tmp_path/"Z.npz"),"samples":array_descriptor(after),"refresh":receipt}
 return m,before

def validate_hybrid(tmp_path,m,before):
 path=tmp_path/"manifest.json";path.write_text(json.dumps(m))
 return a.authenticate_hybrid(path,sha256_file(path),before,"parent",{"actual":"sources"})

def test_saved_G3_auth_uses_actual_Z_without_another_normal_draw(tmp_path,monkeypatch):
 m,before=hybrid_fixture(tmp_path,monkeypatch)
 def forbidden(*args,**kwargs):raise AssertionError("no actual draw during saved artifact assessment")
 monkeypatch.setattr(a.RecordingGenerator,"standard_normal",forbidden)
 got,after=validate_hybrid(tmp_path,m,before)
 assert got["refresh"]["standard_normal_method_calls"]==1 and after[:,3:].tobytes()==before[:,3:].tobytes()

@pytest.mark.parametrize("fault",["seed","role","bit_generator","initial_state","state_hash","default_RNG","refresh_count","kernel","production","missing_footers"])
def test_G3_source_seed_states_scope_failclosed(tmp_path,monkeypatch,fault):
 m,before=hybrid_fixture(tmp_path,monkeypatch);v=m["refresh"]
 if fault=="seed":v["seed"]+=1
 elif fault=="role":v["role"]="different"
 elif fault=="bit_generator":v["bit_generator"]="different"
 elif fault=="initial_state":v["initial_private_state"]["state"]["state"]+=1
 elif fault=="state_hash":v["final_private_state_sha256"]="0"*64
 elif fault=="default_RNG":v["default_numpy_python_RNG_after_sha256"]="0"*64
 elif fault=="refresh_count":v["global_block_refresh_count_per_row"]=2
 elif fault=="kernel":m["conditional_kernel"]["module_sha256"]="0"*64
 elif fault=="production":m["production_qualification_claimed"]=True
 else:m["original_missing_footers_recovered"]=True
 with pytest.raises(ValueError):validate_hybrid(tmp_path,m,before)

def test_real_codec_F64_to_same_production_helper_without_model_or_cast():
 pytest.importorskip("torch")
 from posteriorforge import execution_physical_fifo_v1 as helper
 values=physical(51200);before=values.tobytes()
 chosen,record,rng=r.replay_saved(values,{"case":2,"query":{"synthetic":"deterministic codec-zero fixture"}},helper)
 assert chosen.shape==(50000,21) and chosen.dtype==np.float64
 assert chosen.tobytes()==values[:50000].tobytes() and values.tobytes()==before
 assert record["generated_count"]==51200 and record["original_api_delivered_count"]==50000
 assert record["remaining_unused_buffer_count"]==1200 and record["original_sampler_counts"]["original_checked_count"]==50000
 assert rng["new_CPU_replay_default_RNG_preserved"] and not rng["failed_GPU_request_RNG_footers_recovered"]

def test_immutable_actual_failure_archive_auth_when_available():
 path=Path("/scratch/project/eu-26-17/eubrieucb/posteriorforge/results/accuracy-prior-coordinates-v1/native-pilot-01/manifest.json")
 if not path.exists():pytest.skip("actual Karolina failed archive unavailable")
 # Exact F64 decode is intentionally tied to the original inference versions.
 from posteriorforge.artifacts import versions
 doc=json.loads(path.read_bytes())
 if any(versions()[k]!=doc["versions"][k] for k in ("python","numpy","scipy")):pytest.skip("exact-runtime F64 replay belongs to inference environment")
 parent,values,receipt=r.authenticate_failed(path)
 assert parent["status"]=="failed" and values.shape==(51200,21)
 assert not receipt["missing_original_success_footers_recovered"] and receipt["new_CPU_posthoc_verification"]["passed"]
 assert receipt["missing_failed_request_success_footers"]["actual_prediction_checks"]["recorded_value"] is None

def test_predeclared_focus_and_forensic_labels():
 assert a.FOCUS==(0,7,8,10)
 assert "OLD physical-target" in a.LABELS[0]
 assert all("FORENSIC" in label for label in a.LABELS[1:3]) and "SEPARATE physical G3" in a.LABELS[2]
