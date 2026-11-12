"""CPU-only fresh native/cached FP32 bindings and strict-versus-hard replay."""
from copy import deepcopy
import ast,importlib.util,inspect,json,subprocess,sys
from pathlib import Path
import pytest
from posteriorforge import execution_prior_native_v1 as backend
ROOT=Path(__file__).resolve().parents[1]
def imported(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
q=imported("_prior_native_q_tests",ROOT/"scripts/qualify_prior_native_conditionals_v1.py")
fixtures=imported("_prior_native_numeric_fixtures",ROOT/"tests/test_ss15_graph_conditionals.py")
data=q.data_helper()
old=data.load("qualify_ss15_conditionals.py",data.SS15_QUALIFIER_SHA256)

@pytest.fixture(autouse=True)
def pins(monkeypatch):
 monkeypatch.setattr(q,"NATIVE_BACKEND_SHA256",backend.source_sha256())

def session(stage):
 records=[]
 for phase in (("query",) if stage in q.NATIVE_STAGES else ("prefill","query")):
  for member in range(8):
   records.append({"count":1,"stage":stage,"member":member,"phase":phase,"allow_tf32":False,
    "matmul_precision":"highest","autocast_enabled":False,"autocast_dtype":"torch.float16",
    "strict_fp32_fallback":False,"projection_dtypes":{"q":None,"k":None,"v":None,"kv":None,"qkv":"torch.float32"}})
 return {"mode":backend.CONTROL_MODE,"module_sha256":backend.CACHED_SHA256,"codec_sha256":backend.CODEC_SHA256,
  "fallback_map":{},"aliases_and_flags_restored":True,
  "counters":{"qkv_calls":len(records),"attention_calls":len(records),"expanded_views":0,"witness_histogram_overflow":0},
  "witnesses":records}

def row(stage):
 rule=backend.stage_policy(stage)
 r={"stage_index":stage,"route":backend.route(stage),"target_codec_coordinate":rule["codec_coordinate"],
  "guards":{key:True for key in q.GUARDS},"comparisons":{key:fixtures.comparison() for key in q.QUERIES},
  "hard_predictions":{label+key:fixtures.mass() for label in ("original_","candidate_") for key in q.QUERIES},
  "CDF_call_counts":{"original_FP32":24,"candidate_FP32":24},"corrected_CDF_calls":48,
  "RNG_identity_before":"same","RNG_identity_after":"same","precision_session":session(stage)}
 r["mask_inputs"]={"training":{"mask":rule,"original_latent_target":{"same":"target"},
  "prepared_latent_target":{"same":"target"},"masked_training":{"same":"features"},
  "same_fixed_mask_for_training_query":True,"target_bytes_unchanged":True},
  "actual_latent_target":{"same":"target"},"actual_training":{"same":"features"}}
 if stage in q.NATIVE_STAGES:
  r.update(preparation_witness={"native_neural_prefill_calls":0},
   state_scope="native cold fit guarded; warmed baseline after planned N1, then N1024/stress",
   immutable_native_snapshots=[{"fixed":"weights/context"} for _ in range(4)],
   original_control_immutable_state={"fixed":"weights/context"},
   model_cold={"legitimate":"cold encoder initialization"},
   model_after_planned_N1={"known_context":"warm protected"},
   native_warmed_contracts={"1024":{"known_context":"warm protected"},"stress":{"known_context":"warm protected"}})
 else:r["state_scope"]="unchanged88 prepared-before/after cached contract"
 r.update(q.stage_decision(r,old));return r

def document():
 return {"schema":q.SCHEMA,"status":"prior_native_control_screen_pass","sources":{"fixed":"sources"},
  "raw_bindings":{"actual":"case2"},"transformed_bindings":{"latentF32":"fixed"},
  "coordinate_policy":backend.cached.policy_metadata(),"native_execution_policy":backend.policy_metadata(),
  "codec_inventory":{"passed":True},"completed_grid":{"passed":True,"manifest_sha256":data.COMPLETED_GRID_SHA256},
  "failed_empty_cache_inventory":{"manifest_sha256":q.FAILED_SCREEN_SHA256,"qualification_passed":False},
  "thresholds":q.THRESHOLDS.copy(),"row_counts":[1,1024],"stress_rows":8,"FP16_activated":False,
  "posterior_vectors_generated":0,"reference_integrations":0,"old_numerical_pass_inherited":False,
  "sources_unchanged":True,"global_RNG_preserved":True,"stage_results":[row(j) for j in range(21)],
  "matching_FP32_execution_passed_all21":True,"FP32_failed_stages":[],
  "compatibility_adapter":{"patched_call_count":1008,"original_alias_restored":True,"sdk_sources_unchanged":True,
   "executed_corrected_cdf_code_sha256":"corrected","corrected_cdf_code_sha256":"corrected","original_cdf_code_sha256":"original"}}

def validate(d):
 return q.validate_screen(d,expected_sources={"fixed":"sources"},raw_bindings={"actual":"case2"},
  transformed_bindings={"latentF32":"fixed"},coordinate_policy=backend.cached.policy_metadata(),
  native_policy=backend.policy_metadata(),inventory_replay={"passed":True})

def test_fresh21_two_arm_pass_with_legitimate_cold_initialization():
 d=document();result=validate(d)
 assert result["passed"] and result["native_all8_stages"]==[6,10] and result["cached_stages"]==19
 assert not result["FP16_activated"] and not result["posterior_equivalence_claimed"]
 assert d["stage_results"][6]["model_cold"]!=d["stage_results"][6]["model_after_planned_N1"]

@pytest.mark.parametrize("fault",["numeric","mass","state","target","source","coverage","route","RNG","CDF","dtype",
 "native_prefill","immutable","control_immutable","intermediate_state_restored","threshold","FP16","failed_promoted","globalCDF"])
def test_hard_numeric_binding_route_and_intermediate_failures_never_promoted(fault):
 d=document();r=d["stage_results"][6]
 if fault=="numeric":r["comparisons"]["1024"]=fixtures.comparison(False)
 elif fault=="mass":r["hard_predictions"]["candidate_1024"]["raw_head_masses"][0]["max"]=2
 elif fault=="state":r["guards"]["state_contract_preserved"]=False
 elif fault=="target":r["mask_inputs"]["actual_latent_target"]={"wrong":True}
 elif fault=="source":d["sources"]={"different":True}
 elif fault=="coverage":d["stage_results"].pop()
 elif fault=="route":r["route"]="frozen88_resident_cache"
 elif fault=="RNG":r["RNG_identity_after"]="changed"
 elif fault=="CDF":r["CDF_call_counts"]["candidate_FP32"]=16
 elif fault=="dtype":r["precision_session"]["witnesses"][0]["projection_dtypes"]["qkv"]="torch.float16"
 elif fault=="native_prefill":r["preparation_witness"]["native_neural_prefill_calls"]=8
 elif fault=="immutable":r["immutable_native_snapshots"][2]["fixed"]="changed"
 elif fault=="control_immutable":r["original_control_immutable_state"]["fixed"]="different original context"
 elif fault=="intermediate_state_restored":
  r["native_warmed_contracts"]["1024"]={"known_context":"changed then restored"}
  assert r["native_warmed_contracts"]["stress"]==r["model_after_planned_N1"]
 elif fault=="threshold":d["thresholds"]["max_cdf_absolute"]=.001
 elif fault=="FP16":d["FP16_activated"]=True
 elif fault=="failed_promoted":d["failed_empty_cache_inventory"]["qualification_passed"]=True
 else:d["compatibility_adapter"]["patched_call_count"]=1512
 with pytest.raises((ValueError,RuntimeError)):validate(d)

def test_numeric_failure_stays_hard_valid_but_blocks_pass():
 r=row(10);r["comparisons"]["1024"]=fixtures.comparison(False)
 decision=q.stage_decision(r,old)
 assert decision=={"hard_validity_passed":True,"matching_FP32_execution_passed":False}

def test_collects_all_numeric_stages_and_native_hardstop_before_more_queries():
 source=inspect.getsource(q.run)
 assert 'if not row["hard_validity_passed"]:break' in source
 assert 'if not row["matching_FP32_execution_passed"]:break' not in source
 assert source.index("grid_receipt(")<source.index("import torch")
 assert 'require(warmed_state_equal,"native fitted/context warm contract changed after planned query")' in source
 assert 'require(immutable_ok and old.state_guard(observed)' in source
 tree=ast.parse(source)
 queries=[ast.unparse(n) for n in ast.walk(tree) if isinstance(n,ast.Call) and ast.unparse(n.func)=="old.capture_components"]
 assert queries==["old.capture_components(engine, features, torch)"]
 assert 'queries={"1":mq[:1],"1024":mq,"stress":sq}' in source
 assert 'control.fit(train,target)' in source and "FP16_MODE" not in source

def test_native_query_witness_has_no_invented_prefill_and_cached_keeps_old_contract():
 for j in range(21):assert q.witness_guard(session(j),j,backend)
 d=session(6);d["witnesses"][0]["phase"]="prefill"
 with pytest.raises((ValueError,RuntimeError)):q.witness_guard(d,6,backend)

def test_source_and_actual_receipt_preflight_without_Torch():
 code="import importlib.util,sys;s=importlib.util.spec_from_file_location('q','scripts/qualify_prior_native_conditionals_v1.py');q=importlib.util.module_from_spec(s);s.loader.exec_module(q);from posteriorforge import execution_prior_native_v1 as n;q.NATIVE_BACKEND_SHA256=n.source_sha256();q.source_pins();assert 'torch' not in sys.modules"
 subprocess.run([sys.executable,"-c",code],cwd=ROOT,check=True)
 result=subprocess.run([sys.executable,str(ROOT/"scripts/qualify_prior_native_conditionals_v1.py"),"--help"],cwd=ROOT,capture_output=True,text=True,check=True)
 assert "--failed-screen" in result.stdout and "--completed-grid-audit" in result.stdout

def test_actual_failed_inventory_and_all_CPU_prerequisites_when_available():
 root=Path("/scratch/project/eu-26-17/eubrieucb/posteriorforge")
 phase=root/"results/accuracy-prior-coordinates-v1"
 failed=phase/"conditionals-01/manifest.json"
 if not failed.is_file():pytest.skip("actual immutable Karolina failed inventory unavailable")
 inventory,receipt=data.inventory_receipt(phase/"codec-inventory-inference-01/manifest.json",
  phase/"codec-inventory-reference-01/manifest.json",ROOT/"evidence/accuracy-prior-coordinates-v1/codec-inventory-01/CPU-receipt.json")
 audit={};theta,stats,query,prefix,raw,transformed,custody=data.input_bindings(root/"results/banks-v1",
  root/"results/accuracy-ss15-v1/sigma-native-grid-01/seed-20261111-case-2",inventory,runtime_audit=audit)
 selection=q.failed_screen_receipt(failed,raw,transformed)
 assert selection["predeclared_all8_native_stages"]==[6,10] and not selection["qualification_passed"]
 assert data.grid_receipt(root/"results/accuracy-ss15-v1/sigma-native-grid-completion-01/completion.json")["passed"]
 assert q.source_pins()["execution"]==backend.source_sha256()
 assert receipt["model_Float32_bytes_match_across_environments"] and audit["environment"]
 assert query.shape==(1,15) and prefix.shape==(1024,21) and custody["selected_indices"]==list(range(1024))
