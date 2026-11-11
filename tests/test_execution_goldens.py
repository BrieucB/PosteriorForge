"""CPU-only golden custody/capture tests; all data below are synthetic unit fixtures."""
import copy
import importlib.util
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import numpy as np
import pytest

PATH=Path(__file__).resolve().parents[1]/"scripts/build_execution_goldens.py"
spec=importlib.util.spec_from_file_location("build_goldens_test_module",PATH)
gold=importlib.util.module_from_spec(spec);spec.loader.exec_module(gold)


@pytest.fixture
def payload(tmp_path):
    rows,bins=3,4
    arrays={"query":np.zeros((rows,34),np.float32),"uniforms":np.array([0,.5,.75],np.float32),
            "component_indices":np.array([0,1,7],np.int64),"logical_row_indices":np.array([10,20,30],np.int64),
            "final_logits":np.zeros((rows,bins),np.float32),"physical_borders":np.arange(bins+1,dtype=np.float32)}
    for i in range(8):
        prefix=f"member_{i}_"
        for key in ("raw_model_logits","translation_input_logits","translated_probabilities"):
            arrays[prefix+key]=np.zeros((rows,bins),np.float32)
        for key in ("from_borders","to_borders"):arrays[prefix+key]=np.arange(bins+1,dtype=np.float32)
        arrays[prefix+"logit_cancel_mask"]=np.zeros(bins,bool)
    meta={"rows":rows,"stage_index":1,"member_rows":rows,"member_row_indices":[10,20,30],
          "members":[{"ensemble_index":i} for i in range(8)]}
    entry=gold.save_payload(tmp_path/"fixture.npz",arrays,meta)
    return tmp_path,arrays,meta,entry


def rewrite(root,arrays,meta):
    path=root/"changed.npz"
    return gold.save_payload(path,arrays,meta)


def test_complete_arrays_validate_with_no_pickle(payload):
    root,arrays,meta,entry=payload
    assert gold.verify_payload(root,entry)


def test_changed_payload_bytes_fail_before_replay(payload):
    root,arrays,meta,entry=payload
    with (root/entry["file"]).open("ab") as stream:stream.write(b"x")
    with pytest.raises(ValueError,match="checksum"):gold.verify_payload(root,entry)


@pytest.mark.parametrize("mutation",["dtype","missing_head","wrong_stage","row_order","uniform_one"])
def test_manifest_cannot_hide_structural_or_order_corruption(payload,mutation):
    root,arrays,meta,entry=payload
    if mutation=="dtype":arrays["final_logits"]=arrays["final_logits"].astype(np.float64)
    if mutation=="missing_head":del arrays["member_7_raw_model_logits"]
    if mutation=="wrong_stage":meta["stage_index"]=0
    if mutation=="row_order":arrays["logical_row_indices"]=arrays["logical_row_indices"][::-1]
    if mutation=="uniform_one":arrays["uniforms"][0]=1.
    changed=rewrite(root,arrays,meta)
    with pytest.raises(ValueError):gold.verify_payload(root,changed)


def test_array_descriptor_detects_logical_permutation_and_dtype():
    a=np.array([1.,2.,3.],np.float32)
    assert gold.array_info(a)!=gold.array_info(a[::-1])
    assert gold.array_info(a)!=gold.array_info(a.astype(np.float64))


def test_object_arrays_are_never_goldens():
    with pytest.raises(ValueError,match="numeric"):
        gold.array_info(np.array([object()],dtype=object))


def test_import_is_cpu_safe_and_has_no_model_side_effects():
    code=f"import runpy,sys;runpy.run_path({str(PATH)!r},run_name='golden_import_test');assert 'torch' not in sys.modules"
    subprocess.run([sys.executable,"-c",code],check=True)


def test_builder_refuses_unbudgeted_execution_without_creating_outputs(tmp_path,monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID",raising=False)
    with pytest.raises(RuntimeError):
        gold.build(banks="none",prefix_run="none",checkpoint="none",ledger="none",output=tmp_path/"out")
    assert not (tmp_path/"out").exists()


def test_independent_unequal_width_zero_mass_and_same_border_math():
    torch=pytest.importorskip("torch")
    pytest.importorskip("tabpfn")
    from posteriorforge.compat_cdf import finite_cdf_compatibility
    from posteriorforge.execution_cache import _require_active_cdf
    import tabpfn.regressor as module
    with finite_cdf_compatibility() as descriptor:
        active=_require_active_cdf(module,descriptor)
        result=gold.mathematical_self_checks(torch,active,descriptor)
        assert result["unequal_width_translation_passed"]
        assert result["same_border_translation_passed"]
        assert result["zero_mass_u0_original_nan_preserved"]
    assert descriptor["original_alias_restored"]


@pytest.fixture
def fake_regressor():
    torch=pytest.importorskip("torch")
    pytest.importorskip("tabpfn")
    import tabpfn.regressor as module
    from tabpfn.model.bar_distribution import FullSupportBarDistribution
    class Model:
        training=False
        def named_modules(self):return []
    class Executor:
        def iter_outputs(self,query,**kwargs):
            raw=torch.tensor([[-1.,-.5,.1]],dtype=torch.float32).repeat(len(query),1)
            for i in range(8):yield raw.clone(),SimpleNamespace(target_transform=None,index=i)
    class Regressor:
        def __init__(self):
            self.executor_=Executor();self.model_=Model();self.use_autocast_=False
            self.forced_inference_dtype_=torch.float32;self.softmax_temperature=.9;self.fail=False
        def predict(self,query,**kwargs):
            outputs=[];borders=torch.tensor([-2.,-.5,0.,3.])
            for raw,config in self.executor_.iter_outputs(query):
                if self.fail:raise RuntimeError("deliberate original-path failure")
                outputs.append(module.translate_probs_across_borders(raw/self.softmax_temperature,frm=borders,to=borders))
            return {"logits":torch.stack(outputs).mean(0).log(),
                    "criterion":FullSupportBarDistribution(borders),"untouched_sentinel":42}
    return torch,module,Regressor()


def test_capture_observes_pre_temperature_raw_logits_and_restores_aliases(fake_regressor):
    torch,module,regressor=fake_regressor
    from posteriorforge.compat_cdf import finite_cdf_compatibility
    query=np.zeros((4,34),np.float32)
    original_iter=regressor.executor_.iter_outputs;original_transform=module._transform_borders_one
    with finite_cdf_compatibility() as descriptor:
        active=module.translate_probs_across_borders
        result=gold.capture_original_prediction(regressor,query,descriptor=descriptor)
        assert result["untouched_sentinel"]==42
        assert result["corrected_cdf_call_delta"]==8
        assert len(result["golden_members"])==8
        for member in result["golden_members"]:
            assert torch.equal(member["translation_input_logits"],member["raw_model_logits"]/.9)
        assert module.translate_probs_across_borders is active
        assert module._transform_borders_one is original_transform
        assert regressor.executor_.iter_outputs==original_iter
        assert "iter_outputs" not in vars(regressor.executor_)
        arrays,metadata=gold.pack_prediction(result,query,np.array([0,.1,.5,.9],np.float32),
                                             np.array([0,1,2,3],np.int64),np.arange(4,dtype=np.int64))
        assert arrays["member_0_raw_model_logits"].shape==(4,3)
        assert not arrays["member_0_logit_cancel_mask"].any()
        bad=np.array([0,1,2,8],np.int64)
        with pytest.raises(ValueError,match="member indices"):
            gold.pack_prediction(result,query,np.zeros(4,np.float32),bad,np.arange(4,dtype=np.int64))


def test_capture_failure_restores_methods_and_corrected_alias(fake_regressor):
    torch,module,regressor=fake_regressor
    from posteriorforge.compat_cdf import finite_cdf_compatibility
    original_iter=regressor.executor_.iter_outputs;original_transform=module._transform_borders_one
    regressor.fail=True
    with finite_cdf_compatibility() as descriptor:
        active=module.translate_probs_across_borders
        with pytest.raises(RuntimeError,match="deliberate"):
            gold.capture_original_prediction(regressor,np.zeros((2,34),np.float32),descriptor=descriptor)
        assert module.translate_probs_across_borders is active
        assert module._transform_borders_one is original_transform
        assert regressor.executor_.iter_outputs==original_iter
    assert descriptor["original_alias_restored"]
