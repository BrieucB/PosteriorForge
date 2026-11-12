"""Fresh all21 ordered-target screen tests; no model or posterior calls."""
from copy import deepcopy
import importlib.util,json,subprocess,sys
from pathlib import Path
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
def imported(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
q=imported("_sigma_first_q",ROOT/"scripts/qualify_ss15_sigma_first.py")
fixtures=imported("_shared_numeric_fixtures",ROOT/"tests/test_ss15_graph_conditionals.py")
old=q.load("qualify_ss15_conditionals.py",q.SS15_QUALIFIER_SHA)

def row(stage):
 from posteriorforge import execution_ss15_sigma_first as order
 from posteriorforge.science.analytical_benchmark_v2 import build_problem
 physical=q.ORDER[stage]
 r={"stage_index":stage,"target_physical_index":physical,"target_parameter":build_problem().parameter_order[physical],
  "stress_noise_stage_indices":[i for i in q.NOISE_STAGE_INDICES if i<stage and i in order.stage_policy(stage)["ordered_prefix_indices_kept"]],
  "guards":{k:True for k in q.GUARDS},
  "comparisons":{arm:{key:fixtures.comparison(arm==q.ARMS[0]) for key in ("1","1024","stress")} for arm in q.ARMS},
  "hard_predictions":{"control":fixtures.mass(),"candidate":fixtures.mass()},"corrected_CDF_calls":72}
 r["precision_sessions"]={label:fixtures.session(mode,stage) for label,mode in zip(("FP32","FP16"),q._execution_modes())}
 r["mask_inputs"]={"training":{"mask":order.stage_policy(stage),"policy_sha256":order.policy_metadata()["policy_sha256"],
  "original_target":{"same":"target"},"prepared_target":{"same":"target"},"prepared_training_predictors":{"same":"features"},
  "target_physical_index":physical,"training_query_same_mask":True,"target_bytes_unchanged":True},
  "actual_target":{"same":"target"},"actual_training_predictors":{"same":"features"}}
 r.update(q.stage_decision(r,old));return r

def document():
 from posteriorforge import execution_ss15_sigma_first as order
 from posteriorforge.science.analytical_benchmark_v2 import build_problem
 return {"schema":q.SCHEMA,"status":"sigma_first_conditional_screen_pass","sources":{"fixed":"sources"},
  "raw_bindings":{"parameter_order":list(build_problem().parameter_order)},"transformed_bindings":{"SS15":"fixed"},
  "order_policy":order.policy_metadata(),"thresholds":q.THRESHOLDS.copy(),"row_counts":list(q.ROW_COUNTS),
  "stress_rows":8,"fallback_map":deepcopy(q.FALLBACK_MAP),"posterior_vectors_generated":0,"reference_integrations":0,
  "sources_unchanged":True,"global_RNG_preserved":True,"FP16_qualified_as_FP32":False,"posterior_equivalence_claimed":False,
  "stage_to_physical_order":list(q.ORDER),"noise_stage_indices":list(q.NOISE_STAGE_INDICES),
  "anchor_reused":False,"prior_target_order_qualification_inherited":False,
  "stage_results":[row(j) for j in q.STAGES],"FP16_strict_failed_stages":list(range(21)),
  "matching_FP32_execution_passed_all21":True,"FP16_hard_validity_passed_all21":True,
  "compatibility_adapter":{"patched_call_count":72*21,"original_alias_restored":True,"sdk_sources_unchanged":True,
   "executed_corrected_cdf_code_sha256":"new","corrected_cdf_code_sha256":"new","original_cdf_code_sha256":"old"}}

def validate(d):
 from posteriorforge import execution_ss15_sigma_first as order
 from posteriorforge.science.analytical_benchmark_v2 import build_problem
 return q.validate_screen(d,expected_sources={"fixed":"sources"},
  raw_bindings={"parameter_order":list(build_problem().parameter_order)},transformed_bindings={"SS15":"fixed"},
  order_policy=order.policy_metadata())

def test_fresh_all21_pass_never_promotes_FP16_or_inherits_an_anchor():
 result=validate(document())
 assert result["fresh_stages"]==21 and not result["anchor_reused"]
 assert result["FP16_strict_failed_stages"]==list(range(21)) and not result["FP16_qualified_as_FP32"]

@pytest.mark.parametrize("fault",["missing_global","order","target","noise_stage","wrong_stress_sigma","anchor","old_pass",
 "raw","features","source","mask","threshold","CDFcalls","mass","FP32value","FP16flag","dtype","RNG","inverse","posterior"])
def test_invalid_order_binding_or_numeric_guard_is_never_qualified(fault):
 d=document();r=d["stage_results"][8]
 if fault=="missing_global":d["stage_results"].pop(0)
 elif fault=="order":d["stage_to_physical_order"][3]=3
 elif fault=="target":r["target_physical_index"]=8
 elif fault=="noise_stage":d["noise_stage_indices"]=[6,10,14,16,18,20]
 elif fault=="wrong_stress_sigma":r["stress_noise_stage_indices"]=[3]
 elif fault=="anchor":d["anchor_reused"]=True
 elif fault=="old_pass":d["prior_target_order_qualification_inherited"]=True
 elif fault=="raw":d["raw_bindings"]["parameter_order"][3]="other"
 elif fault=="features":d["transformed_bindings"]["SS15"]="other"
 elif fault=="source":d["sources"]["fixed"]="other"
 elif fault=="mask":r["mask_inputs"]["training"]["mask"]["feature_columns_zeroed"]=[]
 elif fault=="threshold":d["thresholds"]["max_cdf_absolute"]=1e-3
 elif fault=="CDFcalls":r["corrected_CDF_calls"]=64
 elif fault=="mass":r["hard_predictions"]["candidate"]["raw_head_masses"][0]["max"]=1.1
 elif fault=="FP32value":r["comparisons"][q.ARMS[0]]["1024"]["final"]["max_cdf_absolute_error"]=3e-5
 elif fault=="FP16flag":d["FP16_strict_failed_stages"]=[]
 elif fault=="dtype":r["precision_sessions"]["FP16"]["witnesses"][0]["projection_dtypes"]["qkv"]="torch.float32"
 elif fault=="RNG":r["guards"]["model_RNG_preserved"]=False
 elif fault=="inverse":r["comparisons"][q.ARMS[1]]["1024"]["inverse_cdf"]["independent_fp32_replay_failures"]=1
 else:d["posterior_vectors_generated"]=1
 with pytest.raises(ValueError):validate(d)

def test_ordered_targets_masks_and_noise_boundary_stress_all6_blocks():
 from posteriorforge import execution_ss15_sigma_first as order
 physical=np.arange(4096*21,dtype=np.float32).reshape(4096,21)
 theta=order.order_theta(physical);stats=np.arange(4096*15,dtype=np.float32).reshape(4096,15)
 observed=stats[:1].copy();prefixes=theta[:1024].copy()
 assert order.restore_physical(theta).tobytes()==physical.tobytes()
 bounds=(.01,.5);tested=set()
 for stage in range(21):
  main,stress,noise=q.fixed_ordered_queries(observed,prefixes,theta,stage,order,old,bounds)
  mt,ms,mq,sq,binding=q.masked_inputs(theta,stats,main,stress,stage,order)
  assert mt[:,stage].tobytes()==physical[:,q.ORDER[stage]].tobytes()
  assert binding["training"]["target_physical_index"]==q.ORDER[stage]
  if noise:
   column=15+noise[0]
   assert sq[0,column]==np.float32(bounds[0]) and sq[3,column]==np.float32(bounds[1])
   tested.add(noise[0])
 assert tested==set(q.NOISE_STAGE_INDICES)

def test_same_N_references_are_direct_and_no_anchor_API_is_present():
 import ast,inspect,textwrap
 tree=ast.parse(textwrap.dedent(inspect.getsource(q.run)))
 fits=[ast.unparse(n) for n in ast.walk(tree) if isinstance(n,ast.Call) and ast.unparse(n.func)=="control.fit"]
 assert fits==["control.fit(train, target)"]
 assert "anchor_qualification" not in inspect.signature(q.run).parameters
 assert "slice_prediction" not in inspect.getsource(q.run)

def test_actual_saved_bank_ordered_prefix_and_all21_masks_when_available():
 from posteriorforge import execution_ss15_sigma_first as order,representation_ss15 as rep
 from posteriorforge.banks import load_context,load_heldout
 root=Path("/scratch/project/eu-26-17/eubrieucb/posteriorforge/results")
 p=root/"cdf-arithmetic-v1/context-4096/seed-20261111-case-0/samples.npz"
 if not p.is_file():pytest.skip("saved Karolina prefix cohort unavailable")
 context=load_context(root/"banks-v1",20261111,4096)
 physical,stats,observed,raw,_=old.transformed_inputs(context.theta,context.obs,load_heldout(root/"banks-v1",0).obs.reshape(1,33),rep)
 theta=order.order_theta(physical)
 with np.load(p) as archive:prefixes=order.order_theta(archive["samples"][np.linspace(0,49999,1024,dtype=int)].astype(np.float32))
 assert raw["context_theta"]["shape"]==[4096,21] and observed.shape==(1,15)
 for stage in q.STAGES:
  main,stress,_=q.fixed_ordered_queries(observed,prefixes,theta,stage,order,old,(.01,.5))
  mt,_,_,_,_=q.masked_inputs(theta,stats,main,stress,stage,order)
  assert mt[:,stage].tobytes()==physical[:,q.ORDER[stage]].tobytes()

def test_import_help_and_source_preflight_have_no_Torch_dependency():
 subprocess.run([sys.executable,"-c","import importlib.util,sys;s=importlib.util.spec_from_file_location('q','scripts/qualify_ss15_sigma_first.py');q=importlib.util.module_from_spec(s);s.loader.exec_module(q);q.source_pins();assert 'torch' not in sys.modules"],cwd=ROOT,check=True)
 result=subprocess.run([sys.executable,str(ROOT/"scripts/qualify_ss15_sigma_first.py"),"--help"],capture_output=True,text=True,check=True)
 assert "--prefix-run" in result.stdout and "--anchor" not in result.stdout
