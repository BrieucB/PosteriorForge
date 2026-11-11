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
