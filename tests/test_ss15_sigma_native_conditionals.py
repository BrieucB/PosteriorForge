"""Final native-policy screen guards, source pins and deterministic inventory."""
from copy import deepcopy
import importlib.util,json,subprocess,sys
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
def imported(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
q=imported("_sigma_native_q_tests",ROOT/"scripts/qualify_ss15_sigma_native.py")
base=imported("_sigma_first_q_fixtures",ROOT/"tests/test_ss15_sigma_first_conditionals.py")

def document():
 from posteriorforge import execution_ss15_sigma_native_policy as native
 d=base.document()
 d.update(schema=q.SCHEMA,status="sigma_native_conditional_screen_pass",fallback_map=deepcopy(q.FALLBACK_MAP),
  native_policy=native.policy_metadata(),native_stage_indices=[14],native_inventory_sha256=q.INVENTORY_SHA)
 for row in d["stage_results"]:
  if row["stage_index"] in q.NATIVE_STAGES:
   row["comparisons"][q.ARMS[1]]={key:base.fixtures.comparison() for key in ("1","1024","stress")}
   report=row["precision_sessions"]["FP16"]
   for w in report["witnesses"]:
    w["autocast_enabled"]=False;w["projection_dtypes"]["qkv"]="torch.float32"
   row.update(q.stage_decision(row,base.old))
 d["FP16_strict_failed_stages"]=[r["stage_index"] for r in d["stage_results"] if not r["FP16_strict_passed"]]
 return d

def validate(d):
 from posteriorforge import execution_ss15_sigma_first as order,execution_ss15_sigma_native_policy as native
 from posteriorforge.science.analytical_benchmark_v2 import build_problem
 return q.validate_screen(d,expected_sources={"fixed":"sources"},
  raw_bindings={"parameter_order":list(build_problem().parameter_order)},transformed_bindings={"SS15":"fixed"},
  order_policy=order.policy_metadata(),native_policy=native.policy_metadata())

def test_all21_native_policy_pass_retains_other_FP16_drift_without_promotion():
 result=validate(document())
 assert result["passed"] and result["fresh_stages"]==21 and not result["anchor_reused"]
 assert 14 not in result["FP16_strict_failed_stages"]
 assert not result["FP16_qualified_as_FP32"] and not result["posterior_equivalence_claimed"]

@pytest.mark.parametrize("fault",["native_map","native_stages","inventory","native_policy","native_numeric",
 "native_dtype","FP32_numeric","incomplete","posterior","old_pass"])
def test_changed_selection_policy_or_native_precision_cannot_qualify(fault):
 d=document();row=d["stage_results"][14]
 if fault=="native_map":d["fallback_map"]["14"].pop()
 elif fault=="native_stages":d["native_stage_indices"]=[14,15]
 elif fault=="inventory":d["native_inventory_sha256"]="0"*64
 elif fault=="native_policy":d["native_policy"]["active_fallback_map"]["14"].pop()
 elif fault=="native_numeric":
  row["comparisons"][q.ARMS[1]]["1024"]=base.fixtures.comparison(False)
  row.update(q.stage_decision(row,base.old));d["FP16_strict_failed_stages"].append(14);d["FP16_strict_failed_stages"].sort()
 elif fault=="native_dtype":row["precision_sessions"]["FP16"]["witnesses"][0]["projection_dtypes"]["qkv"]="torch.float16"
 elif fault=="FP32_numeric":row["comparisons"][q.ARMS[0]]["1024"]["final"]["max_cdf_absolute_error"]=3e-5
 elif fault=="incomplete":d["stage_results"].pop()
 elif fault=="posterior":d["posterior_vectors_generated"]=50000
 else:d["prior_target_order_qualification_inherited"]=True
 with pytest.raises(ValueError):validate(d)

def test_frozen_native_source_preflight_and_JSON_metadata_are_Torch_free():
 from posteriorforge import execution_ss15_sigma_native_policy as native
 pins=q.source_pins()
 assert pins["native_policy"]==q.NATIVE_MODULE_SHA and pins["native_stage_inventory"]==q.INVENTORY_SHA
 assert json.loads(json.dumps(native.policy_metadata()))==native.policy_metadata()
 subprocess.run([sys.executable,"-c","import importlib.util,sys;s=importlib.util.spec_from_file_location('q','scripts/qualify_ss15_sigma_native.py');q=importlib.util.module_from_spec(s);s.loader.exec_module(q);q.source_pins();assert 'torch' not in sys.modules"],cwd=ROOT,check=True)

def test_runtime_uses_active_native_map_for_both_original1732_modes():
 import ast,inspect,textwrap
 tree=ast.parse(textwrap.dedent(inspect.getsource(q.run)))
 maps=[]
 for call in ast.walk(tree):
  if isinstance(call,ast.Call) and ast.unparse(call.func) in ("adapter.SS15PrecisionSession","adapter.SS15StageExecutor"):
   maps.extend(ast.unparse(k.value) for k in call.keywords if k.arg=="fallback_map")
 assert maps==["native.fallback_map(NATIVE_STAGES)"]*2

def test_actual_six_child_inventory_and_same_failed_MAP2_custody_when_available():
 from posteriorforge import execution_ss15_sigma_native_policy as native
 path=Path("/scratch/project/eu-26-17/eubrieucb/posteriorforge/results/accuracy-ss15-v1/sigma-first-map2-inventory-01/manifest.json")
 if not path.is_file():pytest.skip("actual Karolina deterministic inventory unavailable")
 receipt=q.validate_inventory(path,native.policy_metadata())
 assert receipt["passed"] and receipt["native_stage_indices"]==[14]
 assert receipt["manifest_sha256"]==q.INVENTORY_SHA


def test_numerical_comparison_RNG_role_and_seed_match_frozen_MAP2_helper():
 import ast,inspect,textwrap
 def call(function):
  tree=ast.parse(textwrap.dedent(inspect.getsource(function)))
  matches=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and ast.unparse(n.func)=="derive_seed"]
  assert len(matches)==1
  return ast.dump(matches[0],include_attributes=False)
 assert call(q.run)==call(base.q.run)
 assert '"sigma-first-SS15-fixed-conditionals"' in inspect.getsource(q.run)
