"""CPU-only all21 FP16 experiment contracts; no model/GPU calls."""
import copy,importlib.util,json,subprocess,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
ROOT=Path(__file__).parents[1]
spec=importlib.util.spec_from_file_location("fp16_all21_tests",ROOT/"scripts/characterize_fp16_all_stages.py")
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)

def test_sources_cli_import_and_strict_limits_without_torch():
 helper.sources()
 subprocess.run([sys.executable,"-c",
  "import importlib.util,sys;s=importlib.util.spec_from_file_location('a','scripts/characterize_fp16_all_stages.py');"
  "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);assert 'torch' not in sys.modules;m.sources()"],
  check=True,cwd=ROOT)
 result=subprocess.run([sys.executable,str(ROOT/"scripts/characterize_fp16_all_stages.py"),"--help"],check=True,capture_output=True,text=True)
 assert "--batch-qualification" in result.stdout and "--sdk-root" in result.stdout
 assert helper.STAGES==tuple(range(21)) and helper.MODE=="amp_fp16_prefill_query_v1"
 assert helper.THRESHOLDS=={"normalization_absolute":1e-5,"max_cdf_absolute":2e-5,"max_row_total_variation":1e-4,
                           "quantile_coordinate_span_fraction":2e-6}

def valid_row():
 return {"comparisons":{"main":{"passed":False,"hard":True}},
 "actual_mass_corrected_reconstruction":{"main":{"raw":{"passed":False},"scope":"direct"}},
 "Cp_self_reconstruction":{"main":{"raw":{"passed":True},"weighted":{"passed":True},"scope":"self"}},
 "rng_preserved":True,"target_statistics_preserved":True,"computed_prefill_witness":{"passed":True},
 "computed_query_witness":{"passed":True},"precision_session":{"aliases_and_flags_restored":True},
 "G0_source_restoration_valid":True,"probability_validity":{"main":{"G0":{"passed":True},"Cp":{"passed":True}}},
 "border_validity":{"main":True},"sampler_parity":{"main":{"passed":True}}}

def test_approximate_drift_is_preserved_and_never_labeled_qualification():
 row=valid_row();base=SimpleNamespace(comparison_hard_validity=lambda c:c["hard"])
 result=helper.outcome(row,base)
 assert result["hard_validity_passed"] and not result["strict_screen_passed"]
 assert result["strict_failure_preserved"] and not result["qualified"] and not result["approximation_accepted"]
 assert row["comparisons"]["main"]["passed"] is False

@pytest.mark.parametrize("field",["rng","source","mass","borders","sampler","identity","singleton"])
def test_hard_failure_cannot_be_masked_by_approximate_status(field):
 row=valid_row();base=SimpleNamespace(comparison_hard_validity=lambda c:c["hard"])
 if field=="rng":row["rng_preserved"]=False
 elif field=="source":row["G0_source_restoration_valid"]=False
 elif field=="mass":row["probability_validity"]["main"]["Cp"]["passed"]=False
 elif field=="borders":row["border_validity"]["main"]=False
 elif field=="sampler":row["sampler_parity"]["main"]["passed"]=False
 elif field=="identity":row["comparisons"]["main"]["hard"]=False
 else:row["stage0"]={"hard_validity_passed":False,"comparisons":{"main":{"passed":False}}}
 assert not helper.outcome(row,base)["hard_validity_passed"]

def synthetic_components():
 torch=pytest.importorskip("torch")
 from tabpfn.model.bar_distribution import BarDistribution
 borders=torch.tensor([-3.,0.,2.,9.]);criterion=BarDistribution(borders)
 raw=[torch.tensor([[0.,.2+i*.02,.8-i*.02]],dtype=torch.float32)*(1+float(i)*1e-7) for i in range(8)]
 return {"criterion":criterion,"members":[{"translated_probabilities":p,"raw_logits":p.log(),
   "from_borders":borders.clone(),"to_borders":borders.clone()} for p in raw],
  "component_predictions":[{"criterion":criterion,"logits":p.log()} for p in raw],
  "logits":torch.zeros(1,3)}

def test_stage0_uses_actual_raw_probability_stack_mean_log_not_normalized_component_logits():
 torch=pytest.importorskip("torch");components=synthetic_components()
 # Deliberately different component logits would expose use of the wrong data.
 for p in components["component_predictions"]:p["logits"]=torch.tensor([[8.,0.,-8.]])
 before=[m["translated_probabilities"].clone() for m in components["members"]]
 rng=torch.get_rng_state().clone()
 result=helper.assemble_shared_stage0(components,torch,"cpu")
 expected=torch.stack(before,dim=0).mean(dim=0).log()
 assert torch.equal(result["logits"].view(torch.uint8),expected.view(torch.uint8))
 assert result["criterion"] is components["criterion"] and torch.equal(torch.get_rng_state(),rng)
 assert all(torch.equal(a,m["translated_probabilities"]) for a,m in zip(before,components["members"]))
 assert not result["stage0_shared_policy"]["original_full_API_identity_claimed"]

@pytest.mark.parametrize("bad",["seven","dtype","rows"])
def test_stage0_requires_all8_actual_singleton_fp32_raw_heads(bad):
 torch=pytest.importorskip("torch");components=synthetic_components()
 if bad=="seven":components["members"].pop()
 elif bad=="dtype":components["members"][0]["translated_probabilities"]=components["members"][0]["translated_probabilities"].half()
 else:components["members"][0]["translated_probabilities"]=components["members"][0]["translated_probabilities"].repeat(2,1)
 with pytest.raises(RuntimeError):helper.assemble_shared_stage0(components,torch,"cpu")

def test_stage0_repetition_preserves_physical_criterion_and_actual_row_order():
 torch=pytest.importorskip("torch");p=synthetic_components()
 p=helper.assemble_shared_stage0(p,torch,"cpu");r=helper.repeat_singleton(p,4)
 assert r["criterion"] is p["criterion"] and r["logits"].shape==(4,3)
 for row in r["logits"]:assert torch.equal(row,p["logits"][0])
 assert len(r["members"])==8 and all(m["raw_logits"].shape==(4,3) for m in r["members"])

def test_same_wholebatch_production_icdf_bits_and_existing_endpoint_nan_are_retained():
 torch=pytest.importorskip("torch");p=synthetic_components()
 rng=torch.get_rng_state().clone();before=[m["translated_probabilities"].clone() for m in p["members"]]
 report=helper.sampler_parity(p,torch)
 assert report["passed"] and report["u_one_not_called"] and not report["production_sampler_changed"]
 assert all(r["points"][0]["nonfinite_existing_arithmetic"]==1 for r in report["members"])
 assert torch.equal(torch.get_rng_state(),rng)
 assert all(torch.equal(a,m["translated_probabilities"]) for a,m in zip(before,p["members"]))

def test_actual_qualified_context_and_fresh_portable_binding_is_torch_free():
 from posteriorforge.banks import load_context,load_heldout
 from posteriorforge.science.analytical_benchmark_v2 import build_problem,array_descriptor
 from posteriorforge.inference import CHECKPOINT_SHA256
 phase=Path("/scratch/project/eu-26-17/eubrieucb/posteriorforge/results/performance-first-v1")
 root=phase.parents[1];banks=phase.parent/"banks-v1"
 sdk=phase.parents[1]/"envs/inference/lib/python3.11/site-packages/tabpfn"
 if not banks.is_dir() or not sdk.is_dir():pytest.skip("authenticated HPC artifacts unavailable")
 context=load_context(banks,20261111,4096);case=load_heldout(banks,0);problem=build_problem()
 bindings={"checkpoint_sha256":CHECKPOINT_SHA256,"scientific_fingerprint":problem.scientific_fingerprint,
  "parameter_order":list(problem.parameter_order),"context_theta":array_descriptor(context.theta),
  "context_obs":array_descriptor(context.obs),"query":array_descriptor(case.obs.reshape(1,33))}
 q,receipt=helper.portable_qualification(phase/"gqa-hybrid-qualification-recovered-01/manifest.json",
   phase/"gqa-large-batch-qualification-01/manifest.json",sdk,bindings)
 assert q["full21_gqa_hybrid_qualified"] and not q["full21_qualified"]
 assert receipt["replay"]["existing_narrow_identity_decisions_replayed"]==184
 assert not receipt["current_install_ctime_checked"] and not receipt["new_qualification_or_numeric_waiver"]
 bad={**bindings,"parameter_order":list(reversed(bindings["parameter_order"]))}
 with pytest.raises(RuntimeError,match="parameter_order"):
  helper.portable_qualification(phase/"gqa-hybrid-qualification-recovered-01/manifest.json",
   phase/"gqa-large-batch-qualification-01/manifest.json",sdk,bad)

def test_authenticated_all21_existing_physical_borders_are_finite_and_strictly_increasing():
 base=Path("/scratch/project/eu-26-17/eubrieucb/posteriorforge/results/performance-first-v1/control-goldens-01")
 if not base.is_dir():pytest.skip("authenticated goldens unavailable")
 assert helper.sha(base/"manifest.json")=="7de3549d1ad0c90c11b81ed751b4c9a28dd51441e17ae841f9ff372a8cc63642"
 manifest=(base/"manifest.json").read_text()
 for j in range(21):
  path=base/f"stage-{j:02d}-main.npz";assert helper.sha(path) in manifest
  with np.load(path,allow_pickle=False) as saved:b=saved["physical_borders"]
  assert np.isfinite(b).all() and (np.diff(b)>0).all()


def test_hard_invalidity_never_receives_a_strict_screen_pass():
 row=valid_row();row["comparisons"]["main"]["passed"]=True
 row["actual_mass_corrected_reconstruction"]["main"]["raw"]["passed"]=True
 row["rng_preserved"]=False
 result=helper.outcome(row,SimpleNamespace(comparison_hard_validity=lambda c:c["hard"]))
 assert result["strict_numeric_comparisons_passed"]
 assert not result["hard_validity_passed"] and not result["strict_screen_passed"]


def test_component_sampler_self_reconstruction_is_hard_even_when_drift_is_expected():
 row=valid_row();row["Cp_self_reconstruction"]["main"]["weighted"]["passed"]=False
 result=helper.outcome(row,SimpleNamespace(comparison_hard_validity=lambda c:c["hard"]))
 assert not result["hard_validity_passed"] and not result["qualified"]

def test_constructed_stage0_final_has_its_own_native_e71_production_parity():
 torch=pytest.importorskip("torch");components=synthetic_components()
 singleton=helper.assemble_shared_stage0(components,torch,"cpu")
 result=helper.sampler_distribution_parity(singleton,torch)
 assert result["passed"] and result["u_one_not_called"]
 assert [p["u"] for p in result["points"]]==[0.,.5,float(np.nextafter(np.float32(1),np.float32(0)))]
 assert all(p["bitwise_equal"] and p["rows"]==1 for p in result["points"])
