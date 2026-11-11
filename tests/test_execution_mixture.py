"""CPU exact-law, restoration and finite-bar sampling contracts."""
from contextlib import contextmanager
from types import SimpleNamespace
import numpy as np
import pytest
from posteriorforge import execution_cache as cache
from posteriorforge import execution_mixture as mixture


def test_independent_row_stage_members_preserve_product_of_mixtures():
    probabilities=np.array([.9]*4+[.1]*4)
    independent=sum(a*b for a in probabilities for b in probabilities)/64
    wrong_one_member_for_vector=np.mean(probabilities**2)
    assert abs(independent-.25)<1e-15
    assert abs(wrong_one_member_for_vector-.41)<1e-15
    assert abs(independent-wrong_one_member_for_vector)>.1


def test_row_mass_weight_tv_bound_matches_disjoint_components():
    p=np.zeros((8,2,2))
    p[:4,:,0]=1+1e-6;p[4:,:,1]=1-1e-6
    report=mixture.mixture_roundoff_report(p)
    assert report["component_mass_guard_passed"]
    np.testing.assert_allclose(report["row_actual_normalization_tv"],[.5e-6]*2,atol=2e-16,rtol=0)
    np.testing.assert_allclose(report["row_weight_tv_bound"],report["row_actual_normalization_tv"],atol=2e-16,rtol=0)


def test_normalized_components_have_no_ideal_mixture_discrepancy():
    p=np.arange(1,8*3*5+1,dtype=float).reshape(8,3,5)
    p/=p.sum(-1,keepdims=True)
    report=mixture.mixture_roundoff_report(p)
    assert report["maximum_actual_normalization_tv"]<1e-15


@pytest.mark.parametrize("bad",[np.zeros((8,2,2)),np.full((8,2,2),np.nan),-np.ones((8,2,2))])
def test_invalid_probability_components_fail_closed(bad):
    with pytest.raises(ValueError):mixture.mixture_roundoff_report(bad)


def test_shared_finite_bar_icdf_matches_piecewise_inverse_without_expansion():
    logits=np.log([[.2,.3,.5]]);borders=np.array([-3.,-1.,.5,4.])
    u=np.array([0.,.1,.2,.35,.5,.9])
    expected=np.array([-3.,-2.,-1.,-.25,.5,3.3])
    actual=mixture.finite_bar_icdf_numpy(logits,borders,u)
    np.testing.assert_allclose(actual,expected,atol=1e-14,rtol=0)


def test_finite_inverse_preserves_leading_zero_mass_nan_at_uniform_zero():
    value=mixture.finite_bar_icdf_numpy([[-np.inf,0.]],[-1.,0.,1.],[0.])
    assert np.isnan(value[0])


class Tensor(np.ndarray):
    def __new__(cls,value):return np.asarray(value,dtype=np.float32).view(cls)
    def detach(self):return self
    def cpu(self):return self
    def clone(self):return Tensor(self.copy())
    def numpy(self):return np.asarray(self)


def fake_base(monkeypatch,*,fail=False,mass=1.):
    descriptor={"variant":"cdf_arithmetic_v1","scope":"finite_ensemble_cdf_parentheses_only",
                "source_guard_passed":True,"original_alias_restored":False}
    def cdf(logits,*,borders,ys):
        if descriptor["original_alias_restored"]:raise RuntimeError("inactive")
        return logits
    namespace={"_cdf":cdf}
    exec("def translate(logits,*,frm,to):\n return _cdf(logits,borders=frm,ys=to)\n",namespace)
    active=namespace["translate"];module=SimpleNamespace(translate_probs_across_borders=active)
    descriptor["active_translator_code_sha256"]=cache._code_sha(active)
    executor=SimpleNamespace(preprocessors=list(range(8)),models=list(range(8)),
         configs=[SimpleNamespace(index=i) for i in range(8)],cat_ixs=list(range(8)),n_train_samples=[4096]*8)
    observed=[]
    criterion=object()
    class Regressor:
        executor_=executor
        def predict(self,query,*,output_type,quantiles):
            for preprocessor,model,config,cat_ix,length in zip(executor.preprocessors,executor.models,executor.configs,executor.cat_ixs,executor.n_train_samples):
                observed.append(model)
                assert preprocessor==model==config.index==cat_ix and length==4096
                if fail:raise ValueError("model failure")
                p=Tensor([[mass*(model+1)/10,mass*(1-(model+1)/10)]]*len(query))
                translated=module.translate_probs_across_borders(p,frm=Tensor([-1,0,1]),to=Tensor([-1,0,1]))
            return {"criterion":criterion,"logits":Tensor(np.log(translated))}
    base=object.__new__(cache.ResidentStageExecutor)
    base.regressor=Regressor();base.stage_index=6;base.cdf_descriptor=descriptor
    def predict(query,*,output_type,capture_members):
        cache._require_active_cdf(module,descriptor)
        assert capture_members is False
        return base.regressor.predict(query,output_type=output_type,quantiles=[])
    base.predict=predict
    monkeypatch.setattr(mixture,"_regressor_module",lambda:module)
    monkeypatch.setattr(mixture,"_component_mass_cpu",lambda output:np.asarray(output,dtype=np.float64).sum(-1).view(Tensor))
    return base,module,observed,criterion


def test_component_filters_before_compute_and_restores_all_original_list_objects(monkeypatch):
    base,module,observed,criterion=fake_base(monkeypatch)
    originals={name:getattr(base.regressor.executor_,name) for name in mixture._MEMBER_FIELDS}
    active=module.translate_probs_across_borders
    executor=mixture.ComponentMixtureExecutor(base)
    result=executor.predict_component(3,np.ones((2,39)),capture_member=True)
    assert observed==[3] and result["criterion"] is criterion
    assert result["component_raw_mass"].dtype==np.float64
    assert result["head_mass_envelope"]==mixture.HEAD_MASS_ENVELOPE
    assert result["member"]["ensemble_index"]==3
    assert result["member"]["raw_logits"].shape==(2,2)
    assert module.translate_probs_across_borders is active
    assert "predict" not in vars(base.regressor)
    assert all(getattr(base.regressor.executor_,name) is value for name,value in originals.items())
    assert executor.metadata["full8_member_lists_restored"]
    assert executor.metadata["component_query_rows"][3]==2


def test_component_error_restores_model_lists_proxy_and_active_cdf_alias(monkeypatch):
    base,module,observed,_=fake_base(monkeypatch,fail=True)
    originals={name:getattr(base.regressor.executor_,name) for name in mixture._MEMBER_FIELDS}
    active=module.translate_probs_across_borders
    executor=mixture.ComponentMixtureExecutor(base)
    with pytest.raises(ValueError,match="model failure"):
        executor.predict_component(5,np.ones((1,39)))
    assert observed==[5] and module.translate_probs_across_borders is active
    assert "predict" not in vars(base.regressor)
    assert all(getattr(base.regressor.executor_,name) is value for name,value in originals.items())


def test_component_mass_guard_does_not_silently_renormalize_bad_source_mass(monkeypatch):
    base,module,_,_=fake_base(monkeypatch,mass=1.01)
    executor=mixture.ComponentMixtureExecutor(base)
    with pytest.raises(cache.CacheExecutionError,match="mass exceeds"):
        executor.predict_component(0,np.ones((1,39)))
    assert len(base.regressor.executor_.models)==8
    with pytest.raises(NotImplementedError):executor.predict_component(0,np.ones((1,39)),with_log_prob=True)


def test_exact_resident_source_guard_is_not_optional(monkeypatch):
    monkeypatch.setattr(mixture,"PINNED_CACHE_SHA256","0"*64)
    with pytest.raises(cache.CacheExecutionError,match="frozen resident"):
        mixture._require_cache_source()


def test_actual_torch_vectorized_icdf_matches_pinned_inherited_icdf_common_uniforms():
    torch=pytest.importorskip("torch")
    pytest.importorskip("tabpfn")
    from tabpfn.model.bar_distribution import FullSupportBarDistribution
    for dtype in (torch.float32,torch.float64):
        borders=torch.tensor([-3.,-1.,.5,4.],dtype=dtype)
        criterion=FullSupportBarDistribution(borders)
        logits=torch.tensor([[.2,-.7,1.],[2.,-1.,-.5],[.1,.3,.7]],dtype=dtype)
        uniforms=torch.tensor([0.,.37,.999],dtype=torch.float32)
        expected=torch.stack([criterion.icdf(row,float(u)) for row,u in zip(logits,uniforms)])
        actual=mixture.finite_bar_icdf(logits,borders,uniforms)
        torch.testing.assert_close(actual,expected,atol=1e-6 if dtype==torch.float32 else 1e-12,rtol=0)
        shared=mixture.finite_bar_icdf(logits[:1],borders,uniforms)
        shared_expected=torch.stack([criterion.icdf(logits[0],float(u)) for u in uniforms])
        torch.testing.assert_close(shared,shared_expected,atol=1e-6 if dtype==torch.float32 else 1e-12,rtol=0)
    invalid=torch.tensor([[-float("inf"),0.]],dtype=torch.float32)
    result=mixture.finite_bar_icdf(invalid,torch.tensor([-1.,0.,1.]),torch.tensor([0.]))
    assert torch.isnan(result[0])


def test_vectorized_icdf_does_not_draw_or_reseed_cpu_rng():
    torch=pytest.importorskip("torch")
    before=torch.get_rng_state().clone()
    mixture.finite_bar_icdf(torch.tensor([[0.,1.]]),torch.tensor([-1.,0.,1.]),
                            torch.tensor([.1,.2,.8],dtype=torch.float32))
    assert torch.equal(before,torch.get_rng_state())


def test_raw_mass_head_rejection_reconstructs_normalized_sum_exactly():
    masses=np.linspace(.2,.9,8)
    components=np.array([[.05+.1*i,.95-.1*i] for i in range(8)])
    raw=masses[:,None]*components
    proposal_accept_mass=masses/8/mixture.HEAD_MASS_ENVELOPE
    accepted_head_weights=proposal_accept_mass/proposal_accept_mass.sum()
    head_law=(accepted_head_weights[:,None]*components).sum(0)
    full_law=raw.sum(0)/raw.sum()
    np.testing.assert_allclose(head_law,full_law,atol=2e-16,rtol=0)
    report=mixture.mixture_roundoff_report(raw[:,None,:])
    assert report["mass_corrected_reconstruction_passed"]
    assert report["maximum_mass_corrected_weighted_tv"]<2e-16
    assert report["maximum_actual_normalization_tv"]>.01


def test_head_acceptance_uses_declared_envelope_without_drawing_random_values():
    masses=np.array([.2,.8,mixture.HEAD_MASS_ENVELOPE])
    u=np.array([.1,.9,.999999])
    np.testing.assert_array_equal(mixture.head_acceptance(masses,u),[True,False,True])
    with pytest.raises(ValueError):mixture.head_acceptance([1.01],[.5])
    with pytest.raises(ValueError):mixture.head_acceptance([1.],[1.])


def test_actual_component_mass_accumulates_raw_fp32_in_fp64():
    torch=pytest.importorskip("torch")
    p=torch.full((3,5000),1/5000,dtype=torch.float32)
    result=mixture._component_mass_cpu(p)
    assert result.dtype==torch.float64 and result.device.type=="cpu"
    torch.testing.assert_close(result,p.sum(-1,dtype=torch.float64),rtol=0,atol=0)
