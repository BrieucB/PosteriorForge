"""Fresh latent proof bindings and independent FP32/FP16 decisions; no models."""
from copy import deepcopy
import importlib.util,inspect,json
from pathlib import Path
import numpy as np
import pytest
from posteriorforge import execution_prior_coordinates_v1 as backend
ROOT=Path(__file__).resolve().parents[1]
def imported(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
q=imported("_prior_q_tests",ROOT/"scripts/qualify_prior_conditionals_v1.py")
fixtures=imported("_prior_numeric_fixtures",ROOT/"tests/test_ss15_graph_conditionals.py")
old=q.load("qualify_ss15_conditionals.py",q.SS15_QUALIFIER_SHA256)

@pytest.fixture(autouse=True)
def pins(monkeypatch):
 monkeypatch.setattr(q,"BACKEND_SHA256",backend.source_sha256())
 monkeypatch.setattr(q,"COMPLETED_GRID_SHA256","a"*64)

def session(mode,stage):
 s=fixtures.session(mode,stage);s.update(module_sha256=backend.source_sha256(),codec_sha256=q.CODEC_SHA256,fallback_map={})
 for event in s["witnesses"]:
  strict=mode==backend.CONTROL_MODE;event.update(strict_fp32_fallback=False,autocast_enabled=not strict)
  event["projection_dtypes"]["qkv"]="torch.float32" if strict else "torch.float16"
 return s

def row(stage):
 rule=backend.stage_policy(stage)
 r={"stage_index":stage,"target_codec_coordinate":rule["codec_coordinate"],"guards":{k:True for k in q.GUARDS},
  "arm_guards":{label:{k:True for k in q.ARM_GUARDS} for label in ("FP32","FP16")},
  "comparisons":{arm:{key:fixtures.comparison(arm==q.ARMS[0]) for key in ("1","1024","stress")} for arm in q.ARMS},
  "hard_predictions":{label+key:fixtures.mass() for label in ("original_FP32_","cached_FP32_","cached_FP16_") for key in ("1","1024","stress")},
  "corrected_CDF_call_counts":{"original_FP32":24,"cached_FP32":24,"cached_FP16":24},"corrected_CDF_calls":72,
  "RNG_identity_before":"same","RNG_identity_after":"same"}
 r["precision_sessions"]={label:session(mode,stage) for label,mode in (("FP32",backend.CONTROL_MODE),("FP16",backend.FP16_MODE))}
 r["mask_inputs"]={"training":{"mask":rule,"module_sha256":backend.source_sha256(),"codec_sha256":q.CODEC_SHA256,
  "original_latent_target":{"same":"target"},"prepared_latent_target":{"same":"target"},
  "masked_training":{"same":"features"},"same_fixed_mask_for_training_query":True,"target_bytes_unchanged":True},
  "actual_latent_target":{"same":"target"},"actual_training":{"same":"features"}}
 r.update(q.stage_decision(r,old));return r

def document():
 return {"schema":q.SCHEMA,"status":"prior_coordinates_conditional_screen_pass","sources":{"fixed":"sources"},
  "raw_bindings":{"actual":"case2"},"transformed_bindings":{"latentF32":"fixed"},"coordinate_policy":backend.policy_metadata(),
  "codec_inventory":{"passed":True},"thresholds":q.THRESHOLDS.copy(),"row_counts":[1,1024],"stress_rows":8,"fallback_map":{},
  "posterior_vectors_generated":0,"reference_integrations":0,"old_numerical_qualification_inherited":False,
  "sources_unchanged":True,"global_RNG_preserved":True,"FP16_qualified_as_FP32":False,"posterior_equivalence_claimed":False,
  "completed_grid":{"manifest_sha256":"a"*64,"passed":True},"stage_results":[row(j) for j in range(21)],
  "matching_FP32_execution_passed_all21":True,"FP16_hard_validity_passed_all21":True,"FP16_strict_failed_stages":list(range(21)),
  "compatibility_adapter":{"patched_call_count":1512,"original_alias_restored":True,"sdk_sources_unchanged":True,
   "executed_corrected_cdf_code_sha256":"corrected","corrected_cdf_code_sha256":"corrected","original_cdf_code_sha256":"original"}}

def update(d):
 d["FP16_strict_failed_stages"]=[]
 for row in d["stage_results"]:
  row.update(q.stage_decision(row,old))
  if not row["FP16_strict_passed"]:d["FP16_strict_failed_stages"].append(row["stage_index"])
 d["FP16_hard_validity_passed_all21"]=all(r["FP16_hard_validity_passed"] for r in d["stage_results"])
 d["compatibility_adapter"]["patched_call_count"]=sum(r["corrected_CDF_calls"] for r in d["stage_results"])
 return d

def validate(d):
 return q.validate_screen(d,expected_sources={"fixed":"sources"},raw_bindings={"actual":"case2"},
  transformed_bindings={"latentF32":"fixed"},coordinate_policy=backend.policy_metadata(),inventory_replay={"passed":True})

def test_FP32_pass_with_FP16_strict_drift_is_separate():
 r=validate(document());assert r["matching_FP32_execution_passed_all21"] and r["FP16_strict_failed_stages"]==list(range(21))
 assert not r["FP16_qualified_as_FP32"] and not r["posterior_equivalence_claimed"]

@pytest.mark.parametrize("fault",["mass","dtype_guard","evaluation_error"])
def test_FP16_hard_failure_does_not_veto_independent_FP32(fault):
 d=document();r=d["stage_results"][10]
 if fault=="mass":r["hard_predictions"]["cached_FP16_1024"]["raw_head_masses"][0]["max"]=2
 elif fault=="dtype_guard":r["arm_guards"]["FP16"]["actual_dtype_witnesses"]=False
 else:
  r["comparisons"][q.ARMS[1]]["1024"]=q.evaluation_error(RuntimeError("observed approximate failure"))
  r["corrected_CDF_call_counts"]["cached_FP16"]=8;r["corrected_CDF_calls"]=56
 result=validate(update(d));assert result["matching_FP32_execution_passed_all21"] and not result["FP16_hard_validity_passed_all21"]

@pytest.mark.parametrize("fault",["numeric","mass","alias","target","source","coverage","fallback","RNG","CDF"])
def test_control_or_common_failure_cannot_be_promoted(fault):
 d=document();r=d["stage_results"][4]
 if fault=="numeric":r["comparisons"][q.ARMS[0]]["1024"]=fixtures.comparison(False)
 elif fault=="mass":r["hard_predictions"]["cached_FP32_1024"]["raw_head_masses"][0]["max"]=2
 elif fault=="alias":r["precision_sessions"]["FP16"]["aliases_and_flags_restored"]=False
 elif fault=="target":r["mask_inputs"]["training"]["prepared_latent_target"]={"wrong":True}
 elif fault=="source":d["sources"]={"different":True}
 elif fault=="coverage":d["stage_results"].pop()
 elif fault=="fallback":d["fallback_map"]={"10":[5]}
 elif fault=="RNG":r["RNG_identity_after"]="different"
 else:r["corrected_CDF_call_counts"]["cached_FP32"]=16;r["corrected_CDF_calls"]=64
 with pytest.raises(ValueError):validate(update(d))

def grid():
 pairs=[[seed,case] for seed in (20261111,20261112,20261113) for case in range(8)]
 return {"schema":q.COMPLETION_SCHEMA,"status":"complete","context_seeds":[20261111,20261112,20261113],"cases":list(range(8)),
  "point_count":24,"requested_count_per_point":50000,"completed_pairs":pairs,"all_requests_complete":True,
  "all_assessments_closed":True,"all_integrity_and_reference_checks_passed":True,"all_GPU_allocations_reconciled":True,
  "reserved_GPU_seconds_at_completion":0,"baseline_sources":q.BASELINE_SOURCES.copy(),"physical_accuracy_gate_passed":False,
  "points":[{"context_seed":s,"case":c,"parent_manifest_sha256":"1"*64,"hybrid_manifest_sha256":"2"*64,
   "assessment_sha256":"3"*64,"count":50000,"technical_passed":True,"physical_status":"model_failure"} for s,c in pairs]}

def test_fixed24_completion_accepts_retained_scientific_failures():
 assert q.validate_grid_document(grid())=={"passed":True,"points":24,"physical_accuracy_gate_passed":False}

@pytest.mark.parametrize("fault",["duplicate","missing","reserved","source","count","unclosed","faked_pass"])
def test_grid_completion_failclosed(fault):
 d=grid()
 if fault=="duplicate":d["completed_pairs"][-1]=d["completed_pairs"][0]
 elif fault=="missing":d["points"].pop()
 elif fault=="reserved":d["reserved_GPU_seconds_at_completion"]=1
 elif fault=="source":d["baseline_sources"]["campaign"]="0"*64
 elif fault=="count":d["points"][0]["count"]=49999
 elif fault=="unclosed":d["all_assessments_closed"]=False
 else:d["physical_accuracy_gate_passed"]=True
 with pytest.raises(ValueError):q.validate_grid_document(d)

def test_unpinned_completion_file_disables_new_GPU(tmp_path,monkeypatch):
 path=tmp_path/"receipt.json";path.write_text(json.dumps(grid()));monkeypatch.setattr(q,"COMPLETED_GRID_SHA256",None)
 with pytest.raises(ValueError,match="new GPU disabled"):q.grid_receipt(path)

def test_collect_all_numeric_failures_until_hard_fault_source_rule():
 source=inspect.getsource(q.run)
 assert 'if not row["control_hard_validity_passed"]:break' in source
 assert 'if not row["matching_FP32_execution_passed"]:break' not in source
 assert source.index("grid_receipt(")<source.index("import torch")

def test_latent_stress_is_finite_ownnoise_and_decoder_valid():
 theta=np.zeros((4096,21),np.float32);stats=np.zeros((1,15),np.float32);prefix=np.zeros((1024,21),np.float32)
 for j in range(21):
  main,stress,noise=q.fixed_queries(stats,prefix,theta,j,backend)
  assert main.shape==(1024,15+j) and stress.shape==(8,15+j) and np.isfinite(stress).all()
  assert noise==[s for s in q.NOISE_STAGES if s<j and s in backend.stage_policy(j)["ordered_latent_prefix_indices_kept"]]

def test_actual_CPU_inventory_replay_and_F32_input_preflight_when_available():
 base=Path("/scratch/project/eu-26-17/eubrieucb/posteriorforge/results")
 runtime=base/"accuracy-prior-coordinates-v1/codec-inventory-inference-01/manifest.json"
 reference=base/"accuracy-prior-coordinates-v1/codec-inventory-reference-01/manifest.json"
 combined=ROOT/"evidence/accuracy-prior-coordinates-v1/codec-inventory-01/CPU-receipt.json"
 if not all(p.exists() for p in (runtime,reference,combined)):pytest.skip("actual CPU inventory artifacts unavailable")
 inventory,receipt=q.inventory_receipt(runtime,reference,combined)
 assert receipt["model_Float32_bytes_match_across_environments"] and not receipt["Float64_latent_byte_identity_claimed"]
 runtime_audit={}
 values=q.input_bindings(base/"banks-v1",base/"accuracy-ss15-v1/sigma-native-grid-01/seed-20261111-case-2",inventory,runtime_audit=runtime_audit)
 assert runtime_audit["context_latent_physical_slots_Float64"]["shape"]==[4096,21]
 assert runtime_audit["fixed_prefix_latent_physical_slots_Float64"]["shape"]==[1024,21]
 assert set(runtime_audit["environment"])=={"python","numpy","scipy"}
 assert not runtime_audit["Float64_latent_cross_environment_byte_identity_claimed"]
 theta,stats,query,prefixes,raw,transformed,custody=values
 assert theta.dtype==stats.dtype==query.dtype==prefixes.dtype==np.float32
 assert raw["query"]["shape"]==[1,33] and transformed["fixed_prefix_ordered_latent_model_Float32"]["shape"]==[1024,21]
 assert custody["selected_indices"]==list(range(1024))


def test_actual_completed_grid_receipt_when_available(monkeypatch):
 path=Path("/scratch/project/eu-26-17/eubrieucb/posteriorforge/results/accuracy-ss15-v1/sigma-native-grid-completion-01/completion.json")
 if not path.exists():pytest.skip("final actual24 completion unavailable")
 monkeypatch.setattr(q,"COMPLETED_GRID_SHA256","6136487ee78375a802c8a0cb0c225a7f1b207063b4083514614c0d97a35c2425")
 receipt=q.grid_receipt(path)
 assert receipt["passed"] and receipt["points"]==24 and receipt["physical_accuracy_gate_passed"] is False


def test_hard_control_primary_failure_preserved_before_FP16():
 r=row(4)
 r["hard_predictions"]["original_FP32_1"]["raw_head_masses"][0]["max"]=2
 assert not q.control_hard_decision(r,old,True)
 # This is the actual failure writer invoked before any uninitialized FP16 data
 # is inspected; no secondary KeyError can replace the primary fault.
 r["arm_guards"].pop("FP16");r["comparisons"][q.ARMS[1]]={}
 r["corrected_CDF_call_counts"]["cached_FP16"]=0
 with pytest.raises(q.ControlHardFailure,match="FP16 arm not activated"):q.control_hard_abort(r)
 assert r["failure"]["kind"]=="original_or_cached_FP32_hard_failure"
 assert r["failure"]["FP16_arm_activated"] is False and r["corrected_CDF_calls"]==48


def test_control_numeric_failure_is_not_a_hard_failure():
 r=row(4);r["comparisons"][q.ARMS[0]]["1024"]=fixtures.comparison(False)
 assert q.control_hard_decision(r,old,True)
 assert not q.stage_decision(r,old)["matching_FP32_execution_passed"]
 assert q.stage_decision(r,old)["control_hard_validity_passed"]
