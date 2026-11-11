"""Small synthetic algebra checks only; real conditional references require GPU jobs."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
from scipy.stats import truncnorm

spec=importlib.util.spec_from_file_location('noise_diagnostic',Path(__file__).parents[1]/'scripts/diagnose_noise_conditionals.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_conditional_cdf_recovers_prior_when_likelihood_constant():
    sigma=np.array([.01,.03,.16,.3,.49,.5])
    result=module.conditional_cdf_numpy(sigma,np.zeros_like(sigma),0,nodes=128)
    expected=truncnorm.cdf(sigma,(.01-.255)/.1225,(.5-.255)/.1225,loc=.255,scale=.1225)
    np.testing.assert_allclose(result,expected,atol=2e-13,rtol=0)


def test_conditional_noise_quadrature_refines_and_matches_independent_adaptive_quad():
    sigma=np.array([.01,.05,.12,.25,.49,.5]);rss=np.full_like(sigma,.2)
    q64=module.conditional_cdf_numpy(sigma,rss,10,nodes=64)
    q128=module.conditional_cdf_numpy(sigma,rss,10,nodes=128)
    np.testing.assert_allclose(q64,q128,atol=1e-9,rtol=0)
    assert np.all(np.diff(q128)>=0)
    assert q128[0]==0 and q128[-1]==1
    for s,r,value in zip(sigma,rss,q128):
        check,error=module.conditional_cdf_quad(float(s),float(r),10)
        assert error<1e-8
        assert abs(check-value)<1e-10


@pytest.mark.parametrize('sigma,rss,count', [([.005],[.1],10),([.2],[-1],1),([np.nan],[.1],1),([.2],[.1],-1)])
def test_invalid_inputs_fail_without_repair(sigma,rss,count):
    with pytest.raises(ValueError):module.conditional_cdf_numpy(sigma,rss,count)
