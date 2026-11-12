"""CPU mask/target binding and strict-versus-hard fixed-conditional screen tests."""
from copy import deepcopy
import importlib.util,json,subprocess,sys
from pathlib import Path
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location("_graph_q",ROOT/"scripts/qualify_ss15_graph_conditionals.py")
q=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(q)
old=q.load("qualify_ss15_conditionals.py",q.SS15_QUALIFIER_SHA)

def comparison(passed=True):
 p={"minimum_probability":0.,"max_normalization_error":0.,
  "max_cdf_absolute_error":0. if passed else 3e-5,"max_row_total_variation":0.}
 return {"passed":passed,"exact_final_borders":True,"final_float32":True,"eight_members":True,"final":p.copy(),
  "members":[{**p,"configuration_identity_equal":True,"exact_borders":True,"float32":True} for _ in range(8)],
  "inverse_cdf":{"passed":True,"icdf_contract_version":2,"same_logits_and_borders_exact_failures":0,
   "independent_fp32_replay_failures":0,"production_sampler_changed":False}}

def mass():
 return {"passed":True,"max_normalization_error":0.,"mass_envelope":1.00001,"normalization_absolute":1e-5,
  "raw_head_masses":[{"min":1.,"max":1.} for _ in range(8)],"actual_component_mass_checks":[]}

def session(mode,stage):
 from posteriorforge import execution_ss15 as e
 records=[]
 for phase in ("prefill","query"):
  for member in range(8):
   strict=mode==e.CONTROL_MODE or member in q.FALLBACK_MAP.get(str(stage),[])
   records.append({"count":1,"stage":stage,"member":member,"phase":phase,"allow_tf32":False,
    "matmul_precision":"highest","autocast_enabled":not strict,"autocast_dtype":"torch.float16",
    "projection_dtypes":{"q":None,"k":None,"v":None,"kv":None,"qkv":"torch.float32" if strict else "torch.float16"}})
 return {"mode":mode,"module_sha256":q.ENGINE_SHA,"aliases_and_flags_restored":True,
  "counters":{"qkv_calls":16,"attention_calls":16},"witnesses":records}

def row(stage):
 from posteriorforge import execution_ss15_graph as graph
 r={"stage_index":stage,"guards":{key:True for key in q.GUARDS},
  "comparisons":{arm:{key:comparison(arm==q.ARMS[0]) for key in ("1","1024","stress")} for arm in q.ARMS},
  "hard_predictions":{"original":mass(),"candidate":mass()},"corrected_CDF_calls":72}
 r["precision_sessions"]={label:session(mode,stage) for label,mode in zip(("FP32","FP16"),q._execution_modes())}
 r["mask_inputs"]={"training":{"mask":graph.stage_policy(stage),"policy_sha256":graph.policy_metadata()["policy_sha256"],
  "original_target":{"target":"same"},"prepared_target":{"target":"same"},"prepared_training_predictors":{"x":"same"},
  "training_query_same_mask":True,"target_bytes_unchanged":True},
  "actual_target":{"target":"same"},"actual_training_predictors":{"x":"same"}}
 r.update(q.stage_decision(r,old));return r

def document():
 from posteriorforge import execution_ss15_graph as graph
 d={"schema":q.SCHEMA,"status":"graph_conditional_screen_pass","sources":{"source":"fixed"},
  "raw_bindings":{"raw":"fixed"},"transformed_bindings":{"SS15":"fixed"},"graph_policy":graph.policy_metadata(),
  "thresholds":q.THRESHOLDS.copy(),"row_counts":list(q.ROW_COUNTS),"stress_rows":8,"fallback_map":deepcopy(q.FALLBACK_MAP),
  "posterior_vectors_generated":0,"reference_integrations":0,"sources_unchanged":True,"global_RNG_preserved":True,
  "FP16_qualified_as_FP32":False,"posterior_equivalence_claimed":False,"global_anchor_sha256":q.ANCHOR_SHA,
  "reused_global_stages":[{"stage_index":j,"mask_identity":True,"exact_model_input_bytes_unchanged":True,
   "exact_queries_unchanged":True,"anchor_sha256":q.ANCHOR_SHA,"FP16_strict_passed":False} for j in range(3)],
  "stage_results":[row(j) for j in q.LOCAL_STAGES],"FP16_strict_failed_stages":list(range(21)),
  "matching_FP32_execution_passed_all21":True,"FP16_hard_validity_passed_all21":True,
  "compatibility_adapter":{"patched_call_count":72*18,"original_alias_restored":True,"sdk_sources_unchanged":True,
   "executed_corrected_cdf_code_sha256":"corrected","corrected_cdf_code_sha256":"corrected","original_cdf_code_sha256":"original"}}
 return d

def validate(d):
 from posteriorforge import execution_ss15_graph as graph
 return q.validate_screen(d,expected_sources={"source":"fixed"},raw_bindings={"raw":"fixed"},
  transformed_bindings={"SS15":"fixed"},graph_policy=graph.policy_metadata())

def test_import_source_preflight_and_help_do_not_import_Torch():
 subprocess.run([sys.executable,"-c",
  "import importlib.util,sys;s=importlib.util.spec_from_file_location('q','scripts/qualify_ss15_graph_conditionals.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.source_pins();assert 'torch' not in sys.modules"],cwd=ROOT,check=True)
 result=subprocess.run([sys.executable,str(ROOT/"scripts/qualify_ss15_graph_conditionals.py"),"--help"],
  capture_output=True,text=True,check=True)
 assert "--anchor-qualification" in result.stdout

def test_actual_masks_bind_full_width_labels_and_unused_nan_deletion():
 from posteriorforge import execution_ss15_graph as graph
 theta=np.arange(4096*21,dtype=np.float32).reshape(4096,21)
 stats=np.arange(4096*15,dtype=np.float32).reshape(4096,15)
 main=np.concatenate((stats[:1024],theta[:1024,:12]),axis=1)
 stress=main[:8].copy();stress[0,0]=np.nan;stress[1,15]=np.nan
 before=(theta.tobytes(),stats.tobytes(),main.tobytes(),stress.tobytes())
 mt,ms,mq,sq,binding=q.masked_inputs(theta,stats,main,stress,12,graph)
 assert binding["training"]["target_bytes_unchanged"] and mt[:,12:].tobytes()==theta[:,12:].tobytes()
 assert np.isfinite(sq[0,0]) and np.isnan(sq[1,15])
 assert before==(theta.tobytes(),stats.tobytes(),main.tobytes(),stress.tobytes())
 assert mq.shape==main.shape and ms.shape==stats.shape

def test_structural_pass_retains_all_FP16_strict_drift_flags():
 result=validate(document())
 assert result["passed"] and result["FP16_strict_failed_stages"]==list(range(21))
 assert not result["FP16_qualified_as_FP32"] and not result["posterior_equivalence_claimed"]

@pytest.mark.parametrize("fault",["source","raw","feature","mask","target","threshold","missing18","globalidentity",
 "CDFcount","mass","FP32value","FP16flag","dtype","RNG","inverse","posterior","fallback"])
def test_binding_coverage_numeric_and_hard_failures_never_promoted(fault):
 d=document();r=d["stage_results"][0]
 if fault=="source":d["sources"]["source"]="other"
 elif fault=="raw":d["raw_bindings"]["raw"]="other"
 elif fault=="feature":d["transformed_bindings"]["SS15"]="other"
 elif fault=="mask":r["mask_inputs"]["training"]["mask"]["feature_columns_zeroed"]=[]
 elif fault=="target":r["mask_inputs"]["actual_target"]={"target":"changed"}
 elif fault=="threshold":d["thresholds"]["max_cdf_absolute"]=.001
 elif fault=="missing18":d["stage_results"].pop()
 elif fault=="globalidentity":d["reused_global_stages"][1]["exact_queries_unchanged"]=False
 elif fault=="CDFcount":r["corrected_CDF_calls"]=71
 elif fault=="mass":r["hard_predictions"]["candidate"]["raw_head_masses"][0]["max"]=1.1
 elif fault=="FP32value":r["comparisons"][q.ARMS[0]]["1024"]["final"]["max_cdf_absolute_error"]=3e-5
 elif fault=="FP16flag":d["FP16_strict_failed_stages"]=[]
 elif fault=="dtype":r["precision_sessions"]["FP16"]["witnesses"][0]["projection_dtypes"]["qkv"]="torch.float32"
 elif fault=="RNG":r["guards"]["model_RNG_preserved"]=False
 elif fault=="inverse":r["comparisons"][q.ARMS[1]]["1024"]["inverse_cdf"]["independent_fp32_replay_failures"]=1
 elif fault=="posterior":d["posterior_vectors_generated"]=50000
 else:d["fallback_map"]={"10":[5],"13":[6],"20":[4]}
 with pytest.raises(ValueError):validate(d)

def test_same_N1024_control_and_feature_arguments_are_direct_not_sliced():
 import ast,inspect,textwrap
 tree=ast.parse(textwrap.dedent(inspect.getsource(q.run)))
 fits=[ast.unparse(n) for n in ast.walk(tree) if isinstance(n,ast.Call) and ast.unparse(n.func)=="control.fit"]
 assert fits==["control.fit(train, target)"]
 calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and ast.unparse(n.func)=="old.compare"]
 assert len(calls)==1 and ast.unparse(calls[0].args[0])=="references[key]"
 assert "slice_prediction" not in inspect.getsource(q.run)

def test_actual_global_prefix_and_model_byte_identity_preflight_when_available():
 from posteriorforge.banks import load_context,load_heldout
 from posteriorforge import representation_ss15 as rep,execution_ss15_graph as graph
 from posteriorforge.science.analytical_benchmark_v2 import build_problem
 root=Path("/scratch/project/eu-26-17/eubrieucb/posteriorforge/results")
 path=root/"accuracy-ss15-v1/conditional-qualification-map2-assembled-01/manifest.json"
 if not path.is_file():pytest.skip("saved Karolina anchor unavailable")
 anchor=json.loads(path.read_bytes());context=load_context(root/"banks-v1",20261111,4096)
 theta,stats,observed,_,_=old.transformed_inputs(context.theta,context.obs,load_heldout(root/"banks-v1",0).obs.reshape(1,33),rep)
 with np.load(root/"cdf-arithmetic-v1/context-4096/seed-20261111-case-0/samples.npz") as saved:
  prefix=saved["samples"][np.asarray(anchor["fixed_prefix_row_indices"])].astype(np.float32)
 problem=build_problem();noise=[problem.parameter_order.index(n) for n in problem.prior.noise_names]
 records=q.reuse_globals(anchor,theta,stats,observed,prefix,noise,problem.prior.bounds,old,graph)
 assert [r["stage_index"] for r in records]==[0,1,2]
 prefix[0,0]+=1
 with pytest.raises(ValueError,match="queries changed"):
  q.reuse_globals(anchor,theta,stats,observed,prefix,noise,problem.prior.bounds,old,graph)
