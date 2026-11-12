"""CPU-only approximation diagnostics: flags retained, original math unchanged."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest

ROOT=Path(__file__).parents[1]
spec=importlib.util.spec_from_file_location("approximation_characterizer",ROOT/"scripts/characterize_precision_approximation.py")
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)

def prediction(logits,borders):
    torch=pytest.importorskip("torch")
    from tabpfn.model.bar_distribution import BarDistribution
    criterion=BarDistribution(borders.clone())
    member={"raw_logits":logits.clone(),"translated_probabilities":logits.softmax(-1),
            "from_borders":borders.clone(),"to_borders":borders.clone()}
    return {"logits":logits,"criterion":criterion,"members":[copy.deepcopy(member) for _ in range(8)],
            "component_predictions":[{"logits":logits.clone(),"criterion":criterion,
                "component_raw_mass":logits.softmax(-1).double().sum(-1),"head_mass_envelope":1.00001} for _ in range(8)]}

def test_import_cli_and_original_bindings_without_torch():
    helper.characterize_sources()
    subprocess.run([sys.executable,"-c",
        "import importlib.util,sys; s=importlib.util.spec_from_file_location('c','scripts/characterize_precision_approximation.py');"
        "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);assert 'torch' not in sys.modules;m.characterize_sources()"],cwd=ROOT,check=True)
    assert helper.STAGES==(6,10,13) and helper.ROWS==(128,1024)
    assert helper.APPROXIMATE_FULL_RUN_REVIEW["three_stages_do_not_qualify_all21_or_a_posterior"]
    assert helper.THRESHOLDS=={"max_cdf_absolute":2e-5,"max_row_total_variation":1e-4,
                              "normalization_absolute":1e-5,"quantile_coordinate_span_fraction":2e-6}

def outcome_row():
    comparison={"passed":False,"exact_final_borders":True,"final_float32":True,"eight_members":True,
                "inverse_cdf":{"passed":True},"members":[{"configuration_identity_equal":True,"exact_borders":True,"float32":True}]*8}
    row={"rng_preserved":True,"computed_prefill_witness":{"passed":True},"computed_query_witness":{"passed":True},
         "target_statistics_and_border_types_preserved":True,"comparisons":{"1024":{"arm":comparison}},
         "hard_probability_validity":{"1024":{"arm":{"passed":True}}},
         "actual_precision_component_reconstruction":{"1024":{"raw":{"passed":False},"scope":"diagnostic"}}}
    for name in ("private_FP32_session","uncached_FP32_session","uncached_precision_session","cached_precision_session"):
        row[name]={"aliases_and_flags_restored":True}
    return row

def test_large_numeric_drift_keeps_strict_failure_without_automatic_approximation_rejection():
    row=outcome_row();result=helper.strict_and_hard_outcome(row)
    assert not result["strict_screen_passed"] and result["strict_failure_preserved"]
    assert result["hard_validity_passed"] and not result["approximation_accepted"]
    assert row["comparisons"]["1024"]["arm"]["passed"] is False

@pytest.mark.parametrize("field",["rng","inverse","identity","restoration","probability"])
def test_hard_invalidity_stops_separately_from_drift(field):
    row=outcome_row()
    if field=="rng":row["rng_preserved"]=False
    elif field=="inverse":row["comparisons"]["1024"]["arm"]["inverse_cdf"]["passed"]=False
    elif field=="identity":row["comparisons"]["1024"]["arm"]["members"][0]["configuration_identity_equal"]=False
    elif field=="restoration":row["cached_precision_session"]["aliases_and_flags_restored"]=False
    else:row["hard_probability_validity"]["1024"]["arm"]["passed"]=False
    assert not helper.strict_and_hard_outcome(row)["hard_validity_passed"]

@pytest.mark.parametrize("uniform",[0.,.1,.5,.9,float(np.nextafter(np.float32(1),np.float32(0))),1.])
def test_guarded_stock_return_observer_exact_batch_arithmetic_and_rng(uniform):
    torch=pytest.importorskip("torch")
    logits=torch.tensor([[-float("inf"),0.,-2.],[0.,1.,-55.]],dtype=torch.float32)
    p=prediction(logits,torch.tensor([-3.,0.,5.,100.]))
    before=torch.get_rng_state().clone();bits=logits.view(torch.uint8).clone()
    actual,capture=helper._observed_stock(p["criterion"],logits,uniform)
    stock=p["criterion"].icdf(logits,float(np.float32(uniform)))
    assert torch.equal(actual.view(torch.uint8),stock.view(torch.uint8))
    replay=helper.replay_batch32(capture,p["criterion"].borders.numpy())
    values=actual.numpy()
    assert np.all((values==replay)|(np.isnan(values)&np.isnan(replay)))
    assert torch.equal(torch.get_rng_state(),before) and torch.equal(logits.view(torch.uint8),bits)
    assert capture["arithmetic_ast_restoration_exact"]

def test_installed_cpu_uniform_production_endpoint_is_not_u_one():
    torch=pytest.importorskip("torch")
    meta=helper.uniform_lattice_evidence(torch)
    assert meta["production_max"]==float(np.nextafter(np.float32(1),np.float32(0)))
    assert meta["u_zero_in_production_lattice"] and not meta["u_one_in_production_lattice"]
    assert meta["single_lattice_point_probability_under_uniform_random_word"]==2**-24

def test_probability_mass_guard_and_negative_infinite_cancellation_logits():
    torch=pytest.importorskip("torch")
    p=prediction(torch.tensor([[0.,-float("inf"),1.]]),torch.tensor([-1.,0.,1.,4.]))
    assert helper.hard_prediction_validity(p)["passed"]
    p["members"][2]["translated_probabilities"]*=1.00002
    assert not helper.hard_prediction_validity(p)["passed"]

def test_forensics_are_bounded_checksums_and_serialize_nonfinite_inputs(tmp_path):
    torch=pytest.importorskip("torch")
    p=prediction(torch.tensor([[-float("inf"),0.]]),torch.tensor([-1.,0.,1.]))
    _,c=helper._observed_stock(p["criterion"],p["logits"],0.)
    archive=helper.ForensicArchive(maximum=1)
    for _ in range(2):
        archive.add(p,"final",c,0,stage=6,arm="toy",side="reference",category="production_endpoint",other_value=float("nan"))
    metadata=archive.save(tmp_path)
    assert metadata["record_count"]==1 and len(metadata["sha256"])==64
    json.dumps(metadata,allow_nan=False)
    assert metadata["records"][0]["actual_intermediates"]["value"]=="nan"
    with np.load(tmp_path/"quantile-forensics.npz",allow_pickle=False) as arrays:
        assert len(arrays["record_0_logits"])==2 and arrays["record_0_physical_borders"].dtype==np.float32
        assert np.array_equal(arrays["record_0_actual_probabilities"],c["probabilities"][0])
        assert np.array_equal(arrays["record_0_actual_cumulative"],c["cumulative"][0])

def test_central_and_production_endpoint_errors_have_explicit_equal_span_and_all_heads(tmp_path):
    torch=pytest.importorskip("torch")
    a=prediction(torch.tensor([[0.,-1.,-2.],[1.,0.,-1.]]),torch.tensor([-5.,0.,2.,100.]))
    b=prediction(torch.tensor([[.002,-1.,-2.],[1.,.003,-1.]]),a["criterion"].borders)
    predictions={"U0":a,"Up":b,"Cp":b,"G0":a}
    archive=helper.ForensicArchive()
    result=helper.quantile_characterization(predictions,predictions,stage=6,label="1024",target_sd=3.,archive=archive,utility=None)
    arm=result["arms"]["uncached_FP32_to_uncached_precision"]
    assert set(arm["central"])=={"final",*[f"member_{i}" for i in range(8)]}
    assert [x["u"] for x in arm["central"]["final"]]==[float(np.float32(x)) for x in (.1,.5,.9)]
    assert [x["u"] for x in arm["endpoints"]["final"]]==list(helper.ENDPOINT_U)
    assert arm["central"]["final"][0]["exact_shared_physical_border_span"]==105.
    assert arm["endpoints"]["final"][1]["production_lattice_point"] and arm["endpoints"]["final"][2]["u_one_stress_only"]
    assert result["same_logit_stock_finite_and_independent_replay_exact"]
    saved=archive.save(tmp_path)
    assert saved["record_count"]==18
    production=[r for r in saved["records"] if r["category"]=="production_endpoint"]
    assert production and all(r["native_vs_production_whole_batch_bitwise_equal"] is True for r in production)
    assert all(r["production_inverse"] is not None for r in production)
    stress=[r for r in saved["records"] if r["category"]=="u_one_stress"]
    assert stress and all(r["production_inverse"] is None and r["production_finite_sampler_value"] is None for r in stress)
    with np.load(tmp_path/"quantile-forensics.npz",allow_pickle=False) as arrays:
        for r in production:
            assert arrays[r["key"]+"_native_icdf_value"].tobytes()==arrays[r["key"]+"_production_finite_sampler_value"].tobytes()

def test_capture_free_timing_calls_are_identical_for_both_policies(monkeypatch):
    class Cuda:
        def synchronize(self):pass
    class Torch:cuda=Cuda()
    class Engine:
        def __init__(self):self.calls=[]
        def predict_component(self,member,x,*,capture_member):self.calls.append((member,x,capture_member))
    first=Engine();second=Engine();sets={"128":"a","1024":"b"}
    helper.timed_all8(first,sets,Torch(),repetitions=3);helper.timed_all8(second,sets,Torch(),repetitions=3)
    assert first.calls==second.calls and len(first.calls)==64 and all(not v[2] for v in first.calls)

def test_real_5000_bin_golden_batch_observer_no_gpu():
    torch=pytest.importorskip("torch")
    base=Path("/scratch/project/eu-26-17/eubrieucb/posteriorforge/results/performance-first-v1/control-goldens-01")
    if not (base/"stage-19-main.npz").exists():pytest.skip("authenticated HPC golden unavailable")
    import hashlib
    manifest=json.loads((base/"manifest.json").read_text())
    assert hashlib.sha256((base/"manifest.json").read_bytes()).hexdigest()=="7de3549d1ad0c90c11b81ed751b4c9a28dd51441e17ae841f9ff372a8cc63642"
    # The manifest pins the individual artifact; find its descriptor without assuming layout.
    raw=(base/"manifest.json").read_text()
    file_sha=hashlib.sha256((base/"stage-19-main.npz").read_bytes()).hexdigest()
    assert file_sha in raw
    with np.load(base/"stage-19-main.npz",allow_pickle=False) as arrays:
        logits=torch.from_numpy(arrays["final_logits"].copy());borders=torch.from_numpy(arrays["physical_borders"].copy())
    assert logits.shape==(1024,5000)
    p=prediction(logits,borders)
    for u in (0.,.5,float(np.nextafter(np.float32(1),np.float32(0))),1.):
        value,capture=helper._observed_stock(p["criterion"],logits,u)
        expected=p["criterion"].icdf(logits,u)
        assert torch.equal(value.view(torch.uint8),expected.view(torch.uint8))
        replay=helper.replay_batch32(capture,borders.numpy())
        assert np.all((value.numpy()==replay)|(np.isnan(value.numpy())&np.isnan(replay)))


def test_production_tail_extrapolation_is_decomposed_from_same_intermediates(tmp_path):
    torch=pytest.importorskip("torch")
    generator=torch.Generator().manual_seed(13)
    for _ in range(6):
        logits=torch.randn(1,1000,generator=generator);logits[:,-1]=-50
    p=prediction(logits,torch.arange(1001,dtype=torch.float32))
    u=float(np.nextafter(np.float32(1),np.float32(0)))
    value,capture=helper._observed_stock(p["criterion"],logits,u)
    assert value.item()>1e10 and capture["cdf_terminal"][0]<u
    assert capture["index"][0]==999 and capture["rest"][0]>capture["mass"][0]
    assert capture["mass"][0]>0 and capture["left"][0]==999 and capture["right"][0]==1000
    archive=helper.ForensicArchive();archive.logical_rows=np.array([17])
    archive.add(p,"final",capture,0,stage=6,arm="unchanged_FP32",side="reference",
                category="production_endpoint",other_value=value.item())
    metadata=archive.save(tmp_path);record=metadata["records"][0]
    assert record["u"]==u and record["outside_physical_borders"] and record["original_prefix_sample_row"]==17
    assert record["border_span"]==1000
    assert record["actual_intermediates"]["value"]/record["border_span"]>1e7

def test_active_stock_code_monkeypatch_is_rejected(monkeypatch):
    torch=pytest.importorskip("torch")
    from tabpfn.model.bar_distribution import BarDistribution
    p=prediction(torch.zeros(1,2),torch.tensor([-1.,0.,1.]))
    original=BarDistribution.icdf
    # Preserve source location/name but change ACTIVE arithmetic bytecode.
    replacement=copy.copy(original)
    namespace={};exec("def changed(self,logits,left_prob): return logits.sum(-1)",namespace)
    changed=namespace["changed"].__code__.replace(co_filename=original.__code__.co_filename,
        co_firstlineno=original.__code__.co_firstlineno,co_name=original.__code__.co_name)
    import types
    fake=types.FunctionType(changed,original.__globals__,original.__name__)
    monkeypatch.setattr(BarDistribution,"icdf",fake)
    with pytest.raises(RuntimeError,match="active stock ICDF code"):
        helper._observed_stock(p["criterion"],p["logits"],.5)


def test_actual_reported_component_mass_is_checked_against_captured_probabilities():
    torch=pytest.importorskip("torch")
    p=prediction(torch.tensor([[0.,1.]]),torch.tensor([-1.,0.,1.]))
    p["component_predictions"][3]["component_raw_mass"]*=.9
    assert not helper.hard_prediction_validity(p)["passed"]

def _fake_contract_regressor(cached):
    from types import SimpleNamespace
    torch=pytest.importorskip("torch")
    normalizer_type=type("InputNormalizationEncoderStep",(),{})
    normalizer=normalizer_type();normalizer.normalize_on_train_only=True
    normalizer.mean_for_normalization=torch.tensor([0.,1.],dtype=torch.float32)
    normalizer.std_for_normalization=torch.tensor([1.,2.],dtype=torch.float32)
    class Model:
        def __init__(self):self.cached_embeddings=cached
        def named_modules(self):return [("",self),("normalizer",normalizer)]
    model=Model()
    return SimpleNamespace(executor_=SimpleNamespace(models=[model],preprocessors=["frozen-preprocessor"]),
                           model_=model,use_autocast_=False,forced_inference_dtype_=torch.float32)


def test_private_model_metadata_clone_has_exact_two_expression_ast_delta_without_torch():
    utility=helper.load_helper("qualify_execution_cache.py",helper.UTILITY_SHA)
    function,descriptor=helper.private_model_contract(utility)
    assert function is not utility.model_contract
    assert descriptor["ast_restoration_exact"] and descriptor["declared_changes"]=={
        "byte_hash_conversion":1,"scalar_values_conversion":1}
    assert descriptor["model_tensor_casts"]==descriptor["model_tensor_writes"]==descriptor["rng_calls"]==0
    assert descriptor["original_utility_modified"] is False


@pytest.mark.parametrize("dtype_name",["float32","float16"])
@pytest.mark.parametrize("shape",["scalar","noncontiguous","empty"])
def test_fp32_and_fp16_model_metadata_are_byte_identical_to_original(dtype_name,shape):
    torch=pytest.importorskip("torch");dtype=getattr(torch,dtype_name)
    tensor=(torch.tensor(1.25,dtype=dtype) if shape=="scalar" else
            torch.tensor([[1.,-2.],[float("nan"),float("inf")]],dtype=dtype).T if shape=="noncontiguous" else
            torch.empty((0,2),dtype=dtype))
    regressor=_fake_contract_regressor(tensor);utility=helper.load_helper("qualify_execution_cache.py",helper.UTILITY_SHA)
    private,_=helper.private_model_contract(utility)
    bits=tensor.detach().contiguous().reshape(-1).view(torch.uint8).clone()
    version=tensor._version;rng=torch.get_rng_state().clone()
    assert private(regressor)==utility.model_contract(regressor)
    assert tensor.dtype==dtype and tensor._version==version and torch.equal(torch.get_rng_state(),rng)
    assert torch.equal(tensor.detach().contiguous().reshape(-1).view(torch.uint8),bits)


@pytest.mark.parametrize("scalar",[True,False])
def test_bf16_model_metadata_scalar_and_values_preserve_dtype_bytes_rng_and_version(scalar):
    import hashlib
    torch=pytest.importorskip("torch")
    tensor=(torch.tensor(1.25,dtype=torch.bfloat16) if scalar else
            torch.tensor([[1.25,-2.5],[float("nan"),float("inf")]],dtype=torch.bfloat16).T)
    regressor=_fake_contract_regressor(tensor);utility=helper.load_helper("qualify_execution_cache.py",helper.UTILITY_SHA)
    with pytest.raises(TypeError,match="BFloat16"):utility.model_contract(regressor)
    private,descriptor=helper.private_model_contract(utility)
    version=tensor._version;rng=torch.get_rng_state().clone()
    expected_bytes=tensor.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    result=private(regressor);item=result["modules"][0]["tensors"]["cached_embeddings"]
    assert item["dtype"]=="torch.bfloat16" and item["shape"]==list(tensor.shape)
    assert item["sha256"]==hashlib.sha256(expected_bytes).hexdigest()
    assert item["values"]==([1.25] if scalar else [1.25,"nan",-2.5,"inf"])
    assert tensor.dtype==torch.bfloat16 and tensor._version==version and torch.equal(torch.get_rng_state(),rng)
    assert tensor.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()==expected_bytes
    assert result["input_normalization_train_only_verified"] and descriptor["ast_restoration_exact"]


def test_private_model_metadata_rejects_modified_active_original_function(monkeypatch):
    utility=helper.load_helper("qualify_execution_cache.py",helper.UTILITY_SHA)
    monkeypatch.setattr(utility,"model_contract",lambda regressor:{"forged":True})
    with pytest.raises(RuntimeError,match="active model_contract"):helper.private_model_contract(utility)

