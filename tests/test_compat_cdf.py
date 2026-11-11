"""Explicit CDF-only adapter contracts; pure CPU fake needs no Torch package."""
from __future__ import annotations
import random
from types import ModuleType, SimpleNamespace
import numpy as np
import pytest
from posteriorforge import compat_cdf as compat


class Tensor(np.ndarray):
    def __new__(cls, value, device="cpu"):
        obj=np.asarray(value).view(cls);obj._device=device;return obj
    def __array_finalize__(self,source):
        self._device=getattr(source,"_device","cpu")
    @property
    def device(self):return self._device
    def repeat(self,repeats):return Tensor(np.tile(np.asarray(self),repeats),self.device)
    def clamp(self,min=None,max=None):
        return Tensor(np.clip(np.asarray(self),min,max),self.device)
    def clamp_min(self,value):return self.clamp(min=value)
    def clip(self,min=None,max=None,**kwargs):return self.clamp(min,max)
    def to(self,device):return Tensor(np.asarray(self),device)
    def gather(self,*,index,dim):
        return Tensor(np.take_along_axis(np.asarray(self),np.asarray(index),axis=dim),self.device)


def fake_softmax(value,dim):
    a=np.asarray(value);p=np.exp(a-np.max(a,axis=dim,keepdims=True))
    return Tensor(p/p.sum(axis=dim,keepdims=True),value.device)


FAKE_UTILS = '''
from __future__ import annotations

def _map_to_bucket_ix(y, borders):
    ix = torch.searchsorted(sorted_sequence=borders, input=y) - 1
    ix[y == borders[0]] = 0
    ix[y == borders[-1]] = len(borders) - 2
    return ix

def _cdf(logits: torch.Tensor, borders: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
    ys = ys.repeat(logits.shape[:-1] + (1,))
    n_bars = len(borders) - 1
    y_buckets = _map_to_bucket_ix(ys, borders).clamp(0, n_bars - 1).to(logits.device)
    probs = torch.softmax(logits, dim=-1)
    prob_so_far = torch.cumsum(probs, dim=-1) - probs
    prob_left_of_bucket = prob_so_far.gather(index=y_buckets, dim=-1)
    bucket_widths = borders[1:] - borders[:-1]
    share_of_bucket_left = ys - borders[y_buckets] / bucket_widths[y_buckets]
    share_of_bucket_left = share_of_bucket_left.clamp(0.0, 1.0)
    prob_in_bucket = probs.gather(index=y_buckets, dim=-1) * share_of_bucket_left
    prob_left_of_ys = prob_left_of_bucket + prob_in_bucket
    prob_left_of_ys[ys <= borders[0]] = 0.0
    prob_left_of_ys[ys >= borders[-1]] = 1.0
    return prob_left_of_ys.clip(0.0, 1.0)

def translate_probs_across_borders(logits, *, frm, to):
    prob_left = _cdf(logits, borders=frm, ys=to)
    prob_left[..., 0] = 0.0
    prob_left[..., -1] = 1.0
    return (prob_left[..., 1:] - prob_left[..., :-1]).clamp_min(0.0)
'''


@pytest.fixture
def sdk(tmp_path,monkeypatch):
    utils=ModuleType("tabpfn.utils");utils.__file__=str(tmp_path/"utils.py")
    (tmp_path/"utils.py").write_text(FAKE_UTILS)
    utils.torch=SimpleNamespace(
        Tensor=Tensor,softmax=fake_softmax,
        cumsum=lambda a,dim:Tensor(np.cumsum(np.asarray(a),axis=dim),a.device),
        searchsorted=lambda *,sorted_sequence,input:Tensor(
            np.searchsorted(np.asarray(sorted_sequence),np.asarray(input)),input.device))
    exec(compile(FAKE_UTILS,utils.__file__,"exec"),utils.__dict__)
    regressor=ModuleType("tabpfn.regressor");regressor.__file__=str(tmp_path/"regressor.py")
    (tmp_path/"regressor.py").write_text("from tabpfn.utils import translate_probs_across_borders\n")
    regressor.translate_probs_across_borders=utils.translate_probs_across_borders
    hashes={"tabpfn/utils.py":compat._sha256(utils.__file__),
            "tabpfn/regressor.py":compat._sha256(regressor.__file__)}
    monkeypatch.setattr(compat,"PINNED_SDK_SHA256",hashes)
    monkeypatch.setattr(compat,"_load_sdk",lambda:("2.0.9",utils,regressor))
    return utils,regressor


def expected_translation(probabilities,borders,new_borders):
    expected=np.zeros((*probabilities.shape[:-1],len(new_borders)-1))
    for j,(low,high) in enumerate(zip(new_borders[:-1],new_borders[1:])):
        for k,(left,right) in enumerate(zip(borders[:-1],borders[1:])):
            expected[...,j]+=probabilities[...,k]*max(0,min(high,right)-max(low,left))/(right-left)
    return expected


def test_explicit_activation_changes_only_context_alias_and_restores(sdk):
    utils,regressor=sdk;original=regressor.translate_probs_across_borders
    helper=utils._cdf
    borders=Tensor([-4.,-2.,0.]);logits=Tensor(np.log([[.5,.5]]))
    np.testing.assert_allclose(original(logits,frm=borders,to=borders),[[0,1]])
    with compat.finite_cdf_compatibility() as descriptor:
        assert regressor.translate_probs_across_borders is not original
        assert regressor.translate_probs_across_borders.__code__ is original.__code__
        assert utils.translate_probs_across_borders is original
        assert utils._cdf is helper
        actual=regressor.translate_probs_across_borders(logits,frm=borders,to=borders)
        np.testing.assert_allclose(actual,[[.5,.5]],atol=0)
        assert descriptor["patched_call_count"]==1
        assert descriptor["original_alias_restored"] is False
    assert regressor.translate_probs_across_borders is original
    assert descriptor["original_alias_restored"] is True
    assert descriptor["sdk_sources_unchanged"] is True
    assert descriptor["sdk_aliases_unchanged"] is True
    assert descriptor["scope"]=="finite_ensemble_cdf_parentheses_only"
    assert descriptor["tail_family_changed"] is False
    assert descriptor["execution_optimized"] is False
    assert descriptor["adapter_sha256"]==compat.adapter_source_sha256()
    np.testing.assert_allclose(original(logits,frm=borders,to=borders),[[0,1]])


@pytest.mark.parametrize("dtype",[np.float32,np.float64])
@pytest.mark.parametrize("leading",[(2,),(2,3)])
def test_nonuniform_shifted_borders_broadcast_dtype_and_device(sdk,dtype,leading):
    _,regressor=sdk
    p=np.broadcast_to(np.array([.2,.3,.5],dtype=dtype),(*leading,3)).copy()
    borders=np.array([-3.,-1.,.5,4.],dtype=dtype)
    new=np.array([-4.,-2.,0.,1.,5.],dtype=dtype)
    with compat.finite_cdf_compatibility():
        actual=regressor.translate_probs_across_borders(
            Tensor(np.log(p),"fake:7"),frm=Tensor(borders,"fake:7"),to=Tensor(new,"fake:7"))
    np.testing.assert_allclose(actual,expected_translation(p,borders,new),
                               atol=2e-7 if dtype==np.float32 else 2e-15,rtol=0)
    assert actual.dtype==dtype
    assert actual.device=="fake:7"
    np.testing.assert_allclose(actual.sum(-1),1,atol=2e-7,rtol=0)


def test_body_exception_restores_alias(sdk):
    _,regressor=sdk;original=regressor.translate_probs_across_borders
    with pytest.raises(ValueError,match="body"):
        with compat.finite_cdf_compatibility() as descriptor:raise ValueError("body")
    assert regressor.translate_probs_across_borders is original
    assert descriptor["original_alias_restored"]


def test_nested_context_rejected_without_breaking_outer(sdk):
    _,regressor=sdk;original=regressor.translate_probs_across_borders
    with compat.finite_cdf_compatibility():
        patched=regressor.translate_probs_across_borders
        with pytest.raises(compat.CompatibilityError,match="already active"):
            with compat.finite_cdf_compatibility():pass
        assert regressor.translate_probs_across_borders is patched
    assert regressor.translate_probs_across_borders is original


def test_wrong_sdk_version_rejected_before_activation(sdk,monkeypatch):
    utils,regressor=sdk;original=regressor.translate_probs_across_borders
    monkeypatch.setattr(compat,"_load_sdk",lambda:("9.9",utils,regressor))
    with pytest.raises(compat.CompatibilityError,match="exactly"):
        with compat.finite_cdf_compatibility():pass
    assert regressor.translate_probs_across_borders is original


def test_source_hash_change_rejected_before_activation(sdk):
    utils,regressor=sdk;original=regressor.translate_probs_across_borders
    with open(utils.__file__,"a") as stream:stream.write("\n# unrelated source change\n")
    with pytest.raises(compat.CompatibilityError,match="hash mismatch"):
        with compat.finite_cdf_compatibility():pass
    assert regressor.translate_probs_across_borders is original


def test_preexisting_alias_patch_rejected(sdk):
    utils,regressor=sdk;regressor.translate_probs_across_borders=lambda *a,**k:None
    with pytest.raises(compat.CompatibilityError,match="conflicting SDK aliases"):
        with compat.finite_cdf_compatibility():pass
    assert utils.translate_probs_across_borders is not regressor.translate_probs_across_borders


def test_conflicting_alias_during_context_fails_and_restores_original(sdk):
    _,regressor=sdk;original=regressor.translate_probs_across_borders
    with pytest.raises(compat.CompatibilityError,match="conflicting alias change"):
        with compat.finite_cdf_compatibility() as descriptor:
            regressor.translate_probs_across_borders=lambda *a,**k:None
    assert regressor.translate_probs_across_borders is original
    assert descriptor["original_alias_restored"]


def test_source_change_during_context_fails_after_restoration(sdk):
    utils,regressor=sdk;original=regressor.translate_probs_across_borders
    with pytest.raises(compat.CompatibilityError,match="sources changed"):
        with compat.finite_cdf_compatibility() as descriptor:
            with open(utils.__file__,"a") as stream:stream.write("\n# conflict\n")
    assert regressor.translate_probs_across_borders is original
    assert descriptor["sdk_sources_unchanged"] is False


def test_actual_pinned_torch_cpu_contract_and_rng_preservation():
    torch=pytest.importorskip("torch")
    pytest.importorskip("tabpfn")
    version,utils,regressor=compat._load_sdk()
    if version!="2.0.9":pytest.skip("actual pinned SDK integration only")
    original=regressor.translate_probs_across_borders;helper=utils._cdf
    before_torch=torch.get_rng_state().clone();before_numpy=np.random.get_state();before_python=random.getstate()
    with compat.finite_cdf_compatibility() as descriptor:
        for dtype in (torch.float32,torch.float64):
            p=torch.tensor([[[.2,.3,.5],[.4,.1,.5]]],dtype=dtype)
            borders=torch.tensor([-3.,-1.,.5,4.],dtype=dtype)
            new=torch.tensor([-4.,-2.,0.,1.,5.],dtype=dtype)
            actual=regressor.translate_probs_across_borders(p.log(),frm=borders,to=new)
            expected=expected_translation(p.numpy(),borders.numpy(),new.numpy())
            np.testing.assert_allclose(actual.numpy(),expected,atol=2e-7,rtol=0)
            assert actual.dtype==dtype and actual.device.type=="cpu"
    assert descriptor["patched_call_count"]==2 and descriptor["original_alias_restored"]
    assert utils._cdf is helper and regressor.translate_probs_across_borders is original
    assert torch.equal(torch.get_rng_state(),before_torch)
    after_numpy=np.random.get_state()
    assert before_numpy[0]==after_numpy[0]
    np.testing.assert_array_equal(before_numpy[1],after_numpy[1])
    assert before_numpy[2:]==after_numpy[2:]
    assert random.getstate()==before_python
