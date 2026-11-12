"""CPU engineering guards and pure decision replay; no posterior generation."""
from copy import deepcopy
import importlib.util,json,subprocess,sys
from pathlib import Path
import numpy as np
import pytest

ROOT=Path(__file__).parents[1]
spec=importlib.util.spec_from_file_location("ss15_q_tests",ROOT/"scripts/qualify_ss15_conditionals.py")
q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)

def test_import_help_has_no_Torch_or_CUDA_dependency():
 subprocess.run([sys.executable,"-c","import importlib.util,sys;s=importlib.util.spec_from_file_location('q','scripts/qualify_ss15_conditionals.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);assert 'torch' not in sys.modules"],cwd=ROOT,check=True)
 result=subprocess.run([sys.executable,str(ROOT/"scripts/qualify_ss15_conditionals.py"),"--help"],check=True,text=True,capture_output=True)
 assert "--prefix-run" in result.stdout

def test_exact_transform_uses_saved_float64_before_one_final_cast_without_mutating_inputs():
 from posteriorforge import representation_ss15 as rep
 theta=np.zeros((4096,21),np.float64);raw=np.linspace(-1,1,4096*33).reshape(4096,33);observed=raw[:1].copy()
 saved=raw.copy();t,x,y,binding,features=q.transformed_inputs(theta,raw,observed,rep)
 assert t.shape==(4096,21) and x.shape==(4096,15) and y.shape==(1,15)
 assert t.dtype==x.dtype==y.dtype==np.float32 and np.array_equal(raw,saved)
 np.testing.assert_array_equal(x,rep.transform_observations(raw).astype(np.float32))
 assert binding["context_obs"]["shape"]==[4096,33] and features["context_obs_SS15_model"]["shape"]==[4096,15]
 with pytest.raises(ValueError,match="original float64"):q.transformed_inputs(theta,raw.astype(np.float32),observed,rep)

def test_physical_prefixes_start_at15_and_stress_never_changes_statistics():
 observed=np.arange(15,dtype=np.float32)[None]
 theta=np.linspace(-1,1,4096*21,dtype=np.float32).reshape(4096,21);prefix=theta[:1024].copy()
 main,stress=q.fixed_queries(observed,prefix,theta,5,[2,10],(.01,.5))
 assert main.shape==(1024,20) and stress.shape==(8,20)
 np.testing.assert_array_equal(main[:,:15],np.repeat(observed,1024,axis=0))
 np.testing.assert_array_equal(main[:,15:],prefix[:,:5])
 assert stress[0,17]==np.float32(.01) and stress[3,17]==np.float32(.5)
 assert np.isnan(stress[5,15]) and np.isnan(stress[6,15:]).all()
 np.testing.assert_array_equal(stress[:,:15],np.repeat(observed,8,axis=0))

def test_config_identity_does_not_compare_ASLR_repr_but_requires_pickle_identity():
 a={"type":"module.Config","sha256":"a"*64,"repr":"<function x at0x123>"}
 b={**a,"repr":"<function x at0x456>"}
 assert q.configuration_equal(a,b)
 b["sha256"]="b"*64
 assert not q.configuration_equal(a,b)

def comparison(passed=True):
 probability={"minimum_probability":0.,"max_normalization_error":0.,
  "max_cdf_absolute_error":0. if passed else 3e-5,"max_row_total_variation":0.}
 return {"passed":passed,"exact_final_borders":True,"final_float32":True,"eight_members":True,"final":probability.copy(),
  "members":[{**probability,"configuration_identity_equal":True,"exact_borders":True,"float32":True} for _ in range(8)],
  "inverse_cdf":{"passed":True,"icdf_contract_version":2,"same_logits_and_borders_exact_failures":0,
   "independent_fp32_replay_failures":0,"production_sampler_changed":False}}

def hard_prediction():
 return {"passed":True,"max_normalization_error":0.,"mass_envelope":1.00001,"normalization_absolute":1e-5,
  "raw_head_masses":[{"min":1.,"max":1.} for _ in range(8)],"actual_component_mass_checks":[]}

def session_fixture(mode,stage):
 from posteriorforge import execution_ss15 as adapter
 records=[]
 for phase in ("prefill","query"):
  for member in range(8):
   strict=mode==adapter.CONTROL_MODE or stage==10 and member==5
   records.append({"count":1,"stage":stage,"member":member,"phase":phase,"allow_tf32":False,
    "matmul_precision":"highest","autocast_enabled":not strict,"autocast_dtype":"torch.float16",
    "projection_dtypes":{"q":None,"k":None,"v":None,"kv":None,"qkv":"torch.float32" if strict else "torch.float16"}})
 return {"mode":mode,"module_sha256":q.ENGINE_SHA,"aliases_and_flags_restored":True,
  "counters":{"qkv_calls":16,"attention_calls":16},"witnesses":records}

def stage(index):
 arms={name:{key:comparison() for key in (("128","stress") if name==q.COMPARISON_KEYS[2] else ("1","16","128","1024","stress"))}
  for name in q.COMPARISON_KEYS}
 arms[q.COMPARISON_KEYS[1]]["1024"]=comparison(False)
 result={"stage_index":index,"guards":{key:True for key in q.GUARDS},"comparisons":arms,
  "hard_predictions":{"reference":hard_prediction(),"cached":hard_prediction()}}
 from posteriorforge import execution_ss15 as adapter
 result.update(corrected_CDF_calls=112,precision_sessions={
  "FP32":session_fixture(adapter.CONTROL_MODE,index),"FP16":session_fixture(adapter.FP16_MODE,index)})
 result.update(q.stage_decision(result));return result

def document():
 result={"schema":q.SCHEMA,"status":"ss15_conditional_screen_pass","thresholds":q.THRESHOLDS.copy(),
  "row_counts":list(q.ROW_COUNTS),"stress_rows":8,"posterior_vectors_generated":0,"reference_integrations":0,
  "raw_bindings":{"raw":"sha"},"transformed_bindings":{"features":"sha"},"representation_descriptor":{"module":"sha"},
  "sources":{"new_engine":"sha"},"fallback_map":{"10":[5]},"sources_unchanged":True,"FP16_qualified_as_FP32":False,
  "posterior_equivalence_claimed":False,"stage_results":[stage(i) for i in range(21)],
  "matching_FP32_execution_passed_all21":True,"FP16_hard_validity_passed_all21":True,
  "FP16_strict_failed_stages":list(range(21)),"FP16_matching_uncached_strict_failed_stages":[]}
 result["compatibility_adapter"]={"patched_call_count":2352,"original_alias_restored":True,
  "sdk_sources_unchanged":True,"executed_corrected_cdf_code_sha256":"corrected",
  "corrected_cdf_code_sha256":"corrected","original_cdf_code_sha256":"original"}
 return result

def validate(value):
 return q.validate_qualification(value,raw_bindings={"raw":"sha"},transformed_bindings={"features":"sha"},
  representation_descriptor={"module":"sha"},expected_sources={"new_engine":"sha"},fallback_map={"10":[5]})

def test_pure_replay_accepts_structural_pass_and_retains_FP16_strict_failures():
 result=validate(document())
 assert result["matching_FP32_execution_passed_all21"] and result["FP16_strict_failed_stages"]==list(range(21))
 assert not result["FP16_qualified_as_FP32"] and not result["raw33_numerical_qualification_inherited"]

@pytest.mark.parametrize("problem",["incomplete","threshold","representation","raw","features","source","promotion","FP16_flags",
 "FP32_numeric","hard_mass","normalizer","config","inverse","rows","posterior","fallback"])
def test_tampered_source_coverage_numeric_and_hard_state_cannot_pass(problem):
 value=document();row=value["stage_results"][4]
 if problem=="incomplete":value["stage_results"].pop()
 elif problem=="threshold":value["thresholds"]["max_cdf_absolute"]=.1
 elif problem=="representation":value["representation_descriptor"]["module"]="other"
 elif problem=="raw":value["raw_bindings"]["raw"]="other"
 elif problem=="features":value["transformed_bindings"]["features"]="other"
 elif problem=="source":value["sources"]["new_engine"]="other"
 elif problem=="promotion":value["FP16_qualified_as_FP32"]=True
 elif problem=="FP16_flags":value["FP16_strict_failed_stages"]=[]
 elif problem=="FP32_numeric":row["comparisons"][q.COMPARISON_KEYS[0]]["1024"]["final"]["max_cdf_absolute_error"]=1e-3
 elif problem=="hard_mass":row["hard_predictions"]["cached"]["raw_head_masses"][0]["max"]=1.001
 elif problem=="normalizer":row["guards"]["FP32_normalizers"]=False
 elif problem=="config":row["comparisons"][q.COMPARISON_KEYS[0]]["128"]["members"][2]["configuration_identity_equal"]=False
 elif problem=="inverse":row["comparisons"][q.COMPARISON_KEYS[1]]["1024"]["inverse_cdf"]["independent_fp32_replay_failures"]=1
 elif problem=="rows":row["comparisons"][q.COMPARISON_KEYS[0]].pop("1024")
 elif problem=="posterior":value["posterior_vectors_generated"]=50000
 else:value["fallback_map"]={"10":[5],"13":[6]}
 with pytest.raises(ValueError):validate(value)


@pytest.mark.parametrize("fault",["head_count","CDFfooter","dtype","source","attention","AMP"])
def test_actual_observation_counter_and_dtype_records_are_replayed(fault):
 value=document();row=value["stage_results"][3];report=row["precision_sessions"]["FP16"]
 if fault=="head_count":row["corrected_CDF_calls"]=111
 elif fault=="CDFfooter":value["compatibility_adapter"]["patched_call_count"]=2351
 elif fault=="dtype":report["witnesses"][0]["projection_dtypes"]["qkv"]="torch.float32"
 elif fault=="source":report["module_sha256"]="wrong"
 elif fault=="attention":report["counters"]["attention_calls"]=15
 else:report["witnesses"][0]["autocast_enabled"]=False
 with pytest.raises(ValueError):validate(value)


def test_fast_model_input_descriptors_detect_in_place_changes():
 theta=np.zeros((4096,21),np.float32);stats=np.zeros((4096,15),np.float32);query=np.zeros((1,15),np.float32)
 before=q.model_input_descriptors(theta,stats,query)
 stats[5,7]=1
 after=q.model_input_descriptors(theta,stats,query)
 assert before["physical_targets_float32"]==after["physical_targets_float32"]
 assert before["context_statistics_float32"]!=after["context_statistics_float32"]
 theta[3,11]=2
 assert after["physical_targets_float32"]!=q.model_input_descriptors(theta,stats,query)["physical_targets_float32"]

def test_real_source_pins_and_training_slices_are_Torch_free_and_width15():
 import ast,inspect,textwrap
 pins=q.source_pins()
 assert pins["execution"]==q.ENGINE_SHA and pins["representation"]==q.REPRESENTATION_SHA
 tree=ast.parse(textwrap.dedent(inspect.getsource(q.run)))
 fits=[ast.unparse(node) for node in ast.walk(tree) if isinstance(node,ast.Call) and ast.unparse(node.func)=="control.fit"]
 assert fits==["control.fit(joint[:, :15 + stage], joint[:, 15 + stage])"]
