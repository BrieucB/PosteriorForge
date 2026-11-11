"""Numerical qualification contracts; model execution is never needed by tests."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest

SCRIPT=Path(__file__).parents[1]/"scripts/qualify_execution_cache.py"
spec=importlib.util.spec_from_file_location("cache_qualification",SCRIPT)
qual=importlib.util.module_from_spec(spec);spec.loader.exec_module(qual)


def test_probability_comparison_accepts_declared_roundoff_without_renormalizing():
    a=np.array([[.2,.3,.5]],dtype=np.float32)
    b=np.array([[.200001,.299999,.5]],dtype=np.float32)
    result=qual.compare_probabilities(a,b)
    assert result["passed"] and result["max_cdf_absolute_error"]<2e-5
    original=b.copy();qual.compare_probabilities(a,b);np.testing.assert_array_equal(b,original)


def test_total_variation_detects_alternating_error_hidden_by_cdf_bound():
    a=np.full((1,100),.01)
    b=a+np.tile([3e-6,-3e-6],50)
    result=qual.compare_probabilities(a,b)
    assert result["max_cdf_absolute_error"]<2e-5
    assert result["max_row_total_variation"]>1e-4 and not result["passed"]


@pytest.mark.parametrize("candidate",[np.array([[.21,.29,.5]]),np.array([[.2,.3,.49]]),np.array([[.2,np.nan,.8]]),np.array([[.2,-.1,.9]]),np.array([[.5,.5]])])
def test_probability_invalidity_and_declared_tolerance_fail_closed(candidate):
    assert not qual.compare_probabilities(np.array([[.2,.3,.5]]),candidate)["passed"]


def test_logit_comparison_ignores_additive_gauge_and_records_zero_mass():
    a=np.log(np.array([[.2,.3,.5]]));result=qual.compare_log_probabilities(a,a+4)
    assert result["max_centered_logit_error_on_common_finite_bins"]<1e-14
    with np.errstate(divide="ignore"):z=np.log([[0.,.5,.5]])
    assert qual.compare_log_probabilities(z,z)["negative_infinity_pattern_equal"]


def test_fixed_prefixes_preserve_raw_observation_and_sources_with_bound_stress():
    obs=np.arange(33,dtype=float).reshape(1,33);theta=np.arange(32*21,dtype=float).reshape(32,21)
    prefixes=theta[:16].copy();original=theta.copy()
    main,stress=qual.fixed_queries(obs,prefixes,theta,14,[6,10,14],(.01,.5))
    np.testing.assert_array_equal(main[:,:33],np.repeat(obs,16,axis=0))
    np.testing.assert_array_equal(stress[:,:33],np.repeat(obs,8,axis=0))
    assert stress[6,39]<.01 and stress[7,39]>.5
    np.testing.assert_array_equal(theta,original)


def test_import_and_help_never_import_torch():
    code="import runpy,sys;runpy.run_path("+repr(str(SCRIPT))+",run_name="+repr("qual_import")+");assert "+repr("torch")+" not in sys.modules"
    subprocess.run([sys.executable,"-c",code],check=True)
    result=subprocess.run([sys.executable,str(SCRIPT),"--help"],capture_output=True,text=True,check=True)
    assert "--prefix-run" in result.stdout


def test_unbudgeted_invocation_rejected_before_output_or_torch(tmp_path,monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID",raising=False)
    with pytest.raises(RuntimeError,match="budgeted Slurm"):
        qual.qualify(banks="none",prefix_run="none",checkpoint="none",ledger="none",output=tmp_path/"out")
    assert not (tmp_path/"out").exists()


def test_actual_stock_icdf_endpoint_semantics_are_preserved_on_identical_inputs():
    torch=pytest.importorskip("torch")
    pytest.importorskip("tabpfn")
    from tabpfn.model.bar_distribution import FullSupportBarDistribution
    p=torch.tensor([[0.,.5,.5]],dtype=torch.float32)
    criterion=FullSupportBarDistribution(torch.tensor([-3.,-1.,0.,2.]))
    prediction={"criterion":criterion,"logits":p.log()}
    result=qual.inverse_cdf_probe(prediction,prediction,0.)
    assert result["passed"] and result["same_logits_and_borders_exact_failures"]==0
    assert result["u_one_is_stress_not_production"]
    assert any(row["u"]==0 for row in result["nonfinite_original_endpoint_behavior"])


def test_probability_space_cdf_handles_finite_support_without_repairing_values():
    p=np.array([.2,.3,.5]);b=np.array([-3.,-1.,0.,2.])
    assert qual.finite_bar_cdf64(p,b,-4.)==0
    assert qual.finite_bar_cdf64(p,b,3.)==1
    assert qual.finite_bar_cdf64(p,b,-2.)==pytest.approx(.1)
    assert qual.finite_bar_cdf64(p,b,1.)==pytest.approx(.75)
    assert np.isnan(qual.finite_bar_cdf64(p,b,np.nan))


def test_independent_fp32_replay_preserves_original_endpoint_extrapolation():
    p=np.array([1-3*np.finfo(np.float32).eps,1e-14],dtype=np.float32)
    c=np.cumsum(p,dtype=np.float32);b=np.array([-3.,-1.,2.],dtype=np.float32)
    u=np.float32(1-np.finfo(np.float32).eps)
    value=qual.replay_stock_icdf32(p,c,b,u)
    assert value>b[-1]+1e6
    assert qual.finite_bar_cdf64(p,b,value)==1
    assert 1-float(u)<1e-6


def test_contract_v2_reports_endpoint_artifact_instead_of_false_value_bracket_failure():
    class Array(np.ndarray):
        def __new__(cls,value):return np.asarray(value,dtype=np.float32).view(cls)
        def softmax(self,axis):return self
        def cumsum(self,axis):return Array(np.cumsum(np.asarray(self),axis=axis,dtype=np.float32))
        def detach(self):return self
        def cpu(self):return self
        def numpy(self):return np.asarray(self)
    class Criterion:
        borders=Array([-3.,-1.,2.])
        def icdf(self,p,u):return np.float32(qual.replay_stock_icdf32(p,p.cumsum(-1),self.borders,u))
    epsilon=np.finfo(np.float32).eps
    left={"criterion":Criterion(),"logits":Array([[1-3*epsilon,1e-14]])}
    right={"criterion":Criterion(),"logits":Array([[1-2*epsilon,1e-14]])}
    result=qual.inverse_cdf_probe(left,right,float(epsilon))
    assert result["passed"] and result["icdf_contract_version"]==2
    assert result["same_logits_and_borders_exact_failures"]==0
    assert result["independent_fp32_replay_failures"]==0
    assert result["max_normalized_common_uniform_value_difference"]>1
    assert result["max_probability_residual"]<1e-6
    assert result["production_sampler_changed"] is False


def test_actual_normalized_softmax_production_grid_counterexample_uses_probability_contract():
    torch=pytest.importorskip("torch")
    pytest.importorskip("tabpfn")
    from tabpfn.model.bar_distribution import FullSupportBarDistribution
    generator=torch.Generator().manual_seed(13)
    for _ in range(6):
        logits=torch.randn((1,1000),generator=generator)
        logits[0,-1]=-50.
    criterion=FullSupportBarDistribution(torch.arange(1001,dtype=torch.float32))
    u=float(np.float32(1-np.finfo(np.float32).eps/2))
    value=float(criterion.icdf(logits[0],u).item())
    probabilities=logits[0].softmax(-1).numpy()
    assert value>1e15
    assert abs(qual.finite_bar_cdf64(probabilities,criterion.borders.numpy(),value)-u)==pytest.approx(np.finfo(np.float32).eps/2)
    prediction={"criterion":criterion,"logits":logits}
    result=qual.inverse_cdf_probe(prediction,prediction,0.)
    assert result["passed"] and result["icdf_contract_version"]==2
    assert result["independent_fp32_replay_failures"]==0
    assert any(point["u"]==u and point["production_uniform"] for point in result["diagnostic_points"])
