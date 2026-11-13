"""Deterministic CPU-only v3 qualifier contracts; no neural constructors/draws."""
from pathlib import Path
from types import FunctionType,SimpleNamespace
import copy,importlib.util,json,sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
def load():
 spec=importlib.util.spec_from_file_location("prior_q3_cpu",ROOT/"scripts/qualify_prior_coordinate_pair_v3.py")
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
q=load()
def probability_record():
 return {"actual_probability_dtypes":{k:"torch.float64" for k in ("raw_weights64","probabilities64","sampling_cdf64","sampling_interval_weights64","criterion_borders")},
  "all_components_probability_dtype64":True,"all_required_values_finite":True,"nonnegative_weights":True,
  "positive_raw_row_mass":True,"raw_mass_envelope":1.00001,"max_raw_mass_deviation":0.,
  "API_normalization_error":0.,"CDF_monotonic":True,"CDF_terminal_exactly_one":True,
  "all8_actual_components":True,"fitted_F32_border_bits_promoted_unchanged":True,"temperature_reapplied":False,
  "independent_postprocessing_replay_passed":True,
  "inverse_same_input_replay":{"passed":True,"uniform_bits":list(q.UNIFORM_BITS),"u1_rejected":True,"stored_stage_dtype":"<f4","u0_policy":"first_positive_CDF_interval_left_border"},
  "proposal_precision":{"hard_passed":True}}
def metric():
 return {"minimum_probability":0.,"max_normalization_error":0.,"max_cdf_absolute_error":0.,"max_row_total_variation":0.}
def comparison():
 layers={k:metric() for k in ("raw_weights","API_probabilities","sampling_interval_weights")}
 members=[{"member":i,"configuration_identity_equal":True,"exact_source_borders":True,**copy.deepcopy(layers)} for i in range(8)]
 return {"same_actual_model_inputs":True,"exact_F64_criterion_borders":True,"all8_components":True,"members":members,**layers,"passed":True}
def row():
 return {"guards":dict.fromkeys(q.GUARDS,True),
  "hard_predictions":{a+"_"+k:probability_record() for a in ("original","candidate") for k in q.QUERIES},
  "comparisons":{k:comparison() for k in q.QUERIES}}
def test_frozen_limits_and_exact_metric_math():
 assert q.THRESHOLDS==dict(q._utility.THRESHOLDS)
 a=np.asarray([[.2,.3,.5]],np.float64);b=np.asarray([[.2,.300001,.499999]],np.float64)
 assert q.compare_weights(a,b)==q._utility.compare_probabilities(a,b)
def test_raw_mass_is_not_silently_normalized():
 result=q.compare_weights(np.asarray([[.2,.8]]),np.asarray([[.18,.72]]))
 assert not result["passed"] and result["max_normalization_error"]>0.09
@pytest.mark.parametrize("bad",[
 np.asarray([[np.nan,1.]]),np.asarray([[-1e-9,1.000000001]]),
 np.asarray([[.500021,.499979]])])
def test_frozen_CDF_finite_nonnegative_limits_fail(bad):
 assert not q.compare_weights(np.asarray([[.5,.5]]),bad)["passed"]
def test_independent_reference_identity_borders():
 logits=np.asarray([[0.,-1.,-2.],[-np.inf,0.,-2.]],np.float32)
 borders=np.asarray([-1.,0.,1.,2.],np.float32)
 got=q.reference_translation64(logits,borders,borders)
 assert got["raw_weights64"].dtype==np.float64
 np.testing.assert_allclose(got["raw_weights64"],got["softmax64"],rtol=0,atol=3e-16)
def test_inverse_zero_extension_positive_plateau_and_u1():
 cdf=np.asarray([[0.,.25,.25,1.]],np.float64);b=np.arange(5,dtype=np.float64)
 assert q.independent_inverse64(cdf,b,np.float32(0))[0]==1.
 assert q.independent_inverse64(cdf,b,np.float32(.25))[0]==2.
 assert q.independent_inverse64(cdf,b,np.nextafter(np.float32(.25),np.float32(1)))[0]>3.
 with pytest.raises(ValueError):q.independent_inverse64(cdf,b,np.float32(1))
def test_shared_probability_owner_is_passed_to_session():
 descriptor={"active":True};seen=[]
 class CPUFixtureSession:
  def __init__(self,*,probability_descriptor):
   assert probability_descriptor is descriptor;seen.append(probability_descriptor)
 assert isinstance(q.make_session(SimpleNamespace(PriorNativeSession=CPUFixtureSession),descriptor),CPUFixtureSession)
 assert seen==[descriptor]
def test_probability_and_stage_positive_metadata_fixture():
 assert q.probability_hard(probability_record())
 assert q.stage_decision(row())=={"hard_validity_passed":True,"matching_v3_execution_passed":True}
@pytest.mark.parametrize("field,value",[
 ("all_components_probability_dtype64",False),("all_required_values_finite",False),
 ("nonnegative_weights",False),("raw_mass_envelope",1.01),("max_raw_mass_deviation",1.1e-5),
 ("CDF_terminal_exactly_one",False),("temperature_reapplied",True),
 ("independent_postprocessing_replay_passed",False)])
def test_actual_dtype_mass_probability_contract_faults(field,value):
 bad=probability_record();bad[field]=value
 assert not q.probability_hard(bad)
def test_float32_postprocessing_cannot_be_relabelled_v3():
 bad=probability_record();bad["actual_probability_dtypes"]["probabilities64"]="torch.float32"
 assert not q.probability_hard(bad)
def test_u0_legacy_NaN_policy_cannot_be_relabelled():
 bad=probability_record();bad["inverse_same_input_replay"]["u0_policy"]="legacy_0_over_0"
 assert not q.probability_hard(bad)
def test_raw_API_and_interval_comparisons_are_separate():
 r=row();r["comparisons"]["1024"]["sampling_interval_weights"]["max_cdf_absolute_error"]=2.1e-5
 r["comparisons"]["1024"]["passed"]=False
 result=q.stage_decision(r)
 assert result["hard_validity_passed"] is True and result["matching_v3_execution_passed"] is False
def test_saved_passed_flag_replay_rejects_numeric_relabel():
 r=row();r["comparisons"]["1"]["members"][0]["raw_weights"]["max_cdf_absolute_error"]=2.1e-5
 with pytest.raises(ValueError,match="relabelled"):q.stage_decision(r)
def test_missing_member_and_missing_query_fail_closed():
 r=row();r["comparisons"]["1"]["members"].pop()
 r["comparisons"]["1"]["passed"]=False
 assert not q.stage_decision(r)["hard_validity_passed"]
 r=row();r["comparisons"].pop("stress")
 with pytest.raises(ValueError):q.stage_decision(r)
def test_legacy_qualification_credit_is_hard_failure():
 r=row();r["guards"]["no_legacy_qualification_inherited"]=False
 assert not q.stage_decision(r)["hard_validity_passed"]
def test_bool_numeric_metrics_cannot_pass_as_zero():
 value=metric();value["max_cdf_absolute_error"]=False
 assert not q.strict_probability(value)
def test_unfrozen_protocol_disables_neural_source_entry_without_Torch():
 scope=dict(q.source_pins.__globals__);scope["V3_PROTOCOL_SHA256"]=None
 clone=FunctionType(q.source_pins.__code__,scope)
 with pytest.raises(ValueError,match="not frozen"):clone()
def test_original_numeric_helper_globals_and_function_unchanged():
 function=q._utility.compare_probabilities;before=(function.__code__,id(function.__globals__),copy.deepcopy(q._utility.THRESHOLDS))
 q.compare_weights(np.asarray([[.4,.6]]),np.asarray([[.4,.6]]))
 assert before==(function.__code__,id(function.__globals__),q._utility.THRESHOLDS)
@pytest.mark.parametrize("schema,status",[
 ("posteriorforge.prior_coordinate_pair_conditionals.v2","prior_coordinate_pair_v2_screen_pass"),
 ("posteriorforge.prior_native_control_conditional_screen.v1","prior_native_control_screen_pass")])
def test_old_proof_schema_has_no_v3_credit(schema,status):
 with pytest.raises(ValueError,match="schema"):
  q.validate_screen({"schema":schema,"status":status},spec=q.pairs.FIRST_PAIR,expected_sources={},
   raw_bindings={},transformed_bindings={},coordinate_policy={},native_policy={},
   probability_policy={},proposal_policy={},inventory_replay={},pair_cpu_receipt={})
HPC=Path("/scratch/project/eu-26-17/eubrieucb/posteriorforge")
@pytest.mark.skipif(not (HPC/"results/accuracy-prior-coordinates-v1/pair-preflight-combined-01/receipt.json").exists(),reason="authenticated saved HPC data not installed")
def test_actual_panel_input_bytes_and_wrong_pair_prefix_rejection():
 phase=HPC/"results/accuracy-prior-coordinates-v1";data=HPC/"results/accuracy-ss15-v1"
 kwargs={"banks":HPC/"results/banks-v1","completion_receipt":data/"sigma-native-grid-completion-01/completion.json",
  "training_inventory":phase/"codec-inventory-inference-01/manifest.json",
  "pair_cpu_receipt":phase/"pair-preflight-combined-01/receipt.json","prefix_run":data/"sigma-native-pilot-01"}
 # Resolve original inventory from saved CPU provenance rather than invent a fixture bank.
 cpu=json.loads(kwargs["pair_cpu_receipt"].read_bytes())
 original=cpu.get("training_inventory")
 if isinstance(original,dict) and "path" in original:kwargs["training_inventory"]=original["path"]
 if not Path(kwargs["training_inventory"]).exists():
  pytest.skip("actual original training inventory path supplied by integration preflight")
 bound=q.bound_inputs(q.pairs.FIRST_PAIR,**kwargs)
 assert bound.theta.dtype==bound.stats.dtype==bound.query.dtype==np.float32
 assert bound.theta.shape==(4096,21) and bound.prefixes.shape==(1024,21)
 with pytest.raises(ValueError):q.bound_inputs(q.pairs.PairSpec(20261111,1),**kwargs)


def test_control_hard_failure_is_separate_from_numeric_tolerance_failure():
 r=row();r["comparisons"]["1024"]["passed"]=False
 assert q.control_hard_decision(r) is True
 r["hard_predictions"]["original_1"]["CDF_terminal_exactly_one"]=False
 assert q.control_hard_decision(r) is False

def test_actual_F32_input_fingerprint_detects_mutation():
 mt=np.zeros((2,21),np.float32);ms=np.zeros((2,15),np.float32)
 mq=np.zeros((2,18),np.float32);sq=np.zeros((8,18),np.float32)
 before=q.input_fingerprint(mt,ms,mq,sq)
 mt[0,3]=np.float32(1)
 assert q.input_fingerprint(mt,ms,mq,sq)!=before

def test_actual_v3_F64_probability_inverse_decoder_CPU_fixture():
 torch=pytest.importorskip("torch",reason="actual Torch CPU probability primitives unavailable in reference environment")
 from posteriorforge import execution_probability_v3 as probability,prior_proposals_v3 as proposal
 assert not torch.cuda.is_initialized()
 before=torch.get_rng_state().clone()
 x=torch.tensor([[0.,-1.,-2.],[-2.,0.,-1.]],dtype=torch.float32)
 b=torch.tensor([-1.,0.,1.,2.],dtype=torch.float32)
 with probability.production_scope() as descriptor:
  parts=[]
  for member in range(8):
   p=probability.from_adjusted_logits(x,b,b,b)
   p["member"]={"configuration_identity":{"type":"CPU fixture, no neural model","sha256":"0"*64},
     "from_borders":b,"to_borders":b}
   parts.append(p)
  whole=probability.assemble_shared_raw64(parts,stage0=False)
  record=q.prediction_record(whole,np.zeros((2,15),np.float32),probability,proposal,torch)
  assert q.probability_hard(record),record
  assert record["inverse_same_input_replay"]["outside_or_nonfinite_outputs"]==0
  assert descriptor["translation_call_count"]==8
  assert descriptor["inverse_call_count"]==27
 assert descriptor["closed"] and descriptor["source_unchanged"]
 assert torch.equal(before,torch.get_rng_state()) and not torch.cuda.is_initialized()

def test_v3_U0_policy_differs_from_legacy_without_extra_neural_arm():
 torch=pytest.importorskip("torch",reason="actual Torch CPU probability primitives unavailable in reference environment")
 from posteriorforge import execution_probability_v3 as probability
 x=torch.tensor([[-float("inf"),0.,-1.]],dtype=torch.float32)
 b=torch.tensor([-1.,0.,1.,2.],dtype=torch.float32)
 parts=[]
 for member in range(8):
  p=probability.from_adjusted_logits(x,b,b,b)
  p["member"]={"configuration_identity":{"type":"CPU fixture, no neural model","sha256":"0"*64},"from_borders":b,"to_borders":b}
  parts.append(p)
 whole=probability.assemble_shared_raw64(parts,stage0=False)
 got=probability.finite_bar_icdf(whole["logits"],whole["criterion"].borders,torch.zeros(1,dtype=torch.float32))
 assert got.item()==0.
 legacy=q.legacy_FP32_diagnostic(whole,torch)
 assert legacy["qualifies_v3"] is False and legacy["forward_calls_added"]==0
 assert legacy["old_API_inverse_outputs"][0]["nonfinite"]==1
 assert not torch.cuda.is_initialized()


def test_explicit_checkout_only_source_receipt_has_no_activation_credit():
 receipt=q.checkout_source_receipt()
 assert receipt["checkout_bytes_verified"] is True and receipt["committed_bytes_verified"] is False
 assert set(receipt["sources"])==set(q.new_source_files())

def test_malformed_recorded_revision_rejected_before_Git():
 with pytest.raises(ValueError,match="valid recorded Git revision"):
  q.committed_source_receipt(revision="not-a-recorded-commit")


@pytest.mark.skipif(not (HPC/"results/accuracy-prior-coordinates-v1/pair-preflight-combined-01/receipt.json").exists(),reason="authenticated saved HPC data not installed")
def test_actual_qualifier_query_builder_all21_exact_6f35_constructor_free():
 """Use the ACTUAL run helper and real v3 backend, including failed stage1 seam."""
 import hashlib,pickle,random
 from posteriorforge import execution_prior_native_v3 as backend
 from posteriorforge.science.analytical_benchmark_v2 import array_descriptor
 assert "build_stage_inputs" in q.run.__code__.co_names
 phase=HPC/"results/accuracy-prior-coordinates-v1";prior=HPC/"results/accuracy-ss15-v1"
 cpu_path=phase/"pair-preflight-combined-01/receipt.json"
 assert hashlib.sha256(cpu_path.read_bytes()).hexdigest()==q.PAIR_CPU_RECEIPT_SHA256
 before=pickle.dumps((np.random.get_state(),random.getstate()),protocol=5)
 modules_before=set(sys.modules)
 torch_before=sys.modules.get("torch")
 torch_rng_before=torch_before.get_rng_state().numpy().tobytes() if torch_before is not None else None
 cuda_before=torch_before.cuda.is_initialized() if torch_before is not None else False
 original_sources={name:hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() for name,module in
  (("backend",backend),("neural",backend.neural),("codec",backend.neural.codec_module()))}
 assert original_sources=={"backend":q.V3_BACKEND_SHA256,"neural":backend.NEURAL_SHA,"codec":"d8ce6c49e7b65c6ee55c59b7243d4ce943cbea28c92ebed0329798074aaa70c5"}
 bound=q.bound_inputs(q.pairs.FIRST_PAIR,banks=HPC/"results/banks-v1",
  completion_receipt=prior/"sigma-native-grid-completion-01/completion.json",
  training_inventory=phase/"codec-inventory-inference-01/manifest.json",pair_cpu_receipt=cpu_path,
  prefix_run=prior/"sigma-native-pilot-01")
 source_bytes=tuple(a.tobytes() for a in (bound.theta,bound.stats,bound.query,bound.prefixes))
 data=q.data_helper();fixed=data.fixed_queries;masked=data.masked_inputs
 helper_identity=(fixed.__code__,id(fixed.__globals__),masked.__code__,id(masked.__globals__))
 assert backend.stage_policy is backend.neural.stage_policy
 assert backend.neural.codec_module() is backend.codec
 assert not hasattr(backend,"codec_module")  # The production seam is genuinely exercised.
 records=[]
 for stage in q.STAGES:
  main,stress,noise,mt,ms,mq,sq,binding=q.build_stage_inputs(bound,stage,backend,data)
  old_main,old_stress,old_noise=fixed(bound.query,bound.prefixes,bound.theta,stage,backend.neural)
  old_mt,old_ms,old_mq,old_sq,old_binding=masked(bound.theta,bound.stats,old_main,old_stress,stage,backend.neural)
  assert main.tobytes()==old_main.tobytes() and stress.tobytes()==old_stress.tobytes() and noise==old_noise
  assert all(a.dtype==np.float32 and a.flags.c_contiguous for a in (main,stress,mt,ms,mq,sq))
  assert main.shape==(1024,15+stage) and stress.shape==(8,15+stage)
  assert all(a.tobytes()==b.tobytes() for a,b in ((mt,old_mt),(ms,old_ms),(mq,old_mq),(sq,old_sq)))
  assert binding==old_binding==bound.cpu_row["inference"]["all21_mask_target_query_bindings"][stage]
  assert binding["actual_latent_target"]==array_descriptor(bound.theta[:,stage])
  assert binding["actual_latent_target"]==binding["training"]["prepared_latent_target"]==binding["training"]["original_latent_target"]
  assert mt[:,stage:].tobytes()==bound.theta[:,stage:].tobytes()
  records.append({"stage":stage,"raw_main":array_descriptor(main),"raw_stress":array_descriptor(stress),
   "model_training":binding["actual_training"],"masked_main":binding["main"],"masked_stress":binding["stress"],
   "target":binding["actual_latent_target"],"exact_6f35_binding":True})
 assert [r["stage"] for r in records]==list(range(21))
 assert source_bytes==tuple(a.tobytes() for a in (bound.theta,bound.stats,bound.query,bound.prefixes))
 assert helper_identity==(fixed.__code__,id(fixed.__globals__),masked.__code__,id(masked.__globals__))
 assert original_sources=={name:hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() for name,module in
  (("backend",backend),("neural",backend.neural),("codec",backend.neural.codec_module()))}
 assert pickle.dumps((np.random.get_state(),random.getstate()),protocol=5)==before
 assert not ({"torch","tabpfn"} & (set(sys.modules)-modules_before))
 if torch_before is not None:
  assert torch_before.get_rng_state().numpy().tobytes()==torch_rng_before
  assert torch_before.cuda.is_initialized()==cuda_before
 print("ALL21_ACTUAL_QUERY_BUILDER="+json.dumps({"stages":records,"source_sha256":original_sources,
  "default_RNG_unchanged":True,"NN_constructor_or_forward_GPU_or_draw_calls":0},sort_keys=True))


@pytest.mark.parametrize("value",[60,1050,1650])
def test_actual_run_deadline_guard_accepts_bounded_integer(value):
 import inspect
 assert "validate_max_seconds" in q.run.__code__.co_names
 assert inspect.signature(q.run).parameters["max_seconds"].default==1050
 assert q.validate_max_seconds(value) is value

@pytest.mark.parametrize("value",[59,1651,True,False,1050.0])
def test_actual_run_deadline_guard_rejects_outside_or_noninteger(value):
 with pytest.raises(ValueError,match="bounded helper deadline60..1650"):
  q.validate_max_seconds(value)
