"""Synthetic interpolation checks only: no checkpoint, learned model or GPU."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('translation_diagnostic',Path(__file__).parents[1]/'scripts/diagnose_ensemble_translation.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_correct_finite_cdf_nonunit_negative_borders_and_identity():
    p=np.array([[.5,.5]])
    b=np.array([-4.,-2.,0.])
    np.testing.assert_allclose(module.finite_bar_cdf(p,b,[-4,-3,-2,-1,0]),[[0,.25,.5,.75,1]],atol=0)
    translated,_=module.translate_finite(p,b,b)
    np.testing.assert_allclose(translated,p,atol=0)
    broken,_=module.translate_finite(p,b,b,original_precedence=True)
    np.testing.assert_allclose(broken,[[0,1]],atol=0)


def test_correct_translation_preserves_mass_and_matches_uniform_interval_overlap():
    p=np.array([[.2,.3,.5],[.4,.5,.1]])
    b=np.array([-3.,-1.,.5,4.])
    new=np.array([-4.,-2.,0.,1.,5.])
    actual,cdf=module.translate_finite(p,b,new)
    expected=np.zeros_like(actual)
    for row in range(len(p)):
        for target,(low,high) in enumerate(zip(new[:-1],new[1:])):
            for mass,left,right in zip(p[row],b[:-1],b[1:]):
                expected[row,target]+=mass*max(0,min(high,right)-max(low,left))/(right-left)
    np.testing.assert_allclose(actual,expected,atol=2e-16)
    np.testing.assert_allclose(actual.sum(-1),1,atol=2e-16)
    assert np.all(np.diff(cdf,axis=-1)>=0)


def test_buggy_cdf_can_still_be_monotone_and_probability_normalized():
    p=np.array([[.25,.25,.25,.25]])
    b=np.array([-3.,-2.,-1.,0.,2.])
    query=np.array([-3.,-2.,-1.5,-1.,-.5,0.,.1,1.,2.])
    bad=module.finite_bar_cdf(p,b,query,original_precedence=True)
    assert np.all(np.diff(bad,axis=-1)>=0)
    broken,_=module.translate_finite(p,b,query,original_precedence=True)
    np.testing.assert_allclose(broken.sum(-1),1,atol=0)
    good=module.finite_bar_cdf(p,b,query)
    assert np.max(np.abs(bad-good))>.01
    assert np.all(np.diff(good,axis=-1)>=0)


@pytest.mark.parametrize('borders',[[0,0,1],[0,2,1],[0,np.nan,1]])
def test_bad_borders_fail_closed(borders):
    with pytest.raises(ValueError):module.finite_bar_cdf([[.5,.5]],borders,[0,1])
