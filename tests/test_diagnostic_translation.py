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

summary_spec=importlib.util.spec_from_file_location('model_summary',Path(__file__).parents[1]/'scripts/summarize_model_diagnostics.py')
summary=importlib.util.module_from_spec(summary_spec);summary_spec.loader.exec_module(summary)


def test_tail_tv_ks_formula_matches_independent_density_integrals():
    from scipy.integrate import quad
    from scipy.special import erf
    from scipy.stats import norm
    borders=np.array([-3.,-1.,4.,5.])
    masses=np.array([.3,.4,.3])
    scales=np.array([2.,1.])/norm.ppf(.75)
    def finite_density(y):
        if y<borders[0] or y>borders[-1]:return 0.
        i=min(np.searchsorted(borders,y,side='right')-1,2)
        return masses[i]/np.diff(borders)[i]
    def full_density(y):
        if y<=borders[1]:return masses[0]*np.sqrt(2/np.pi)/scales[0]*np.exp(-.5*((borders[1]-y)/scales[0])**2)
        if y>=borders[-2]:return masses[-1]*np.sqrt(2/np.pi)/scales[1]*np.exp(-.5*((y-borders[-2])/scales[1])**2)
        return masses[1]/(borders[2]-borders[1])
    numerical_tv=.5*sum(quad(lambda y:abs(full_density(y)-finite_density(y)),a,b,epsabs=1e-12)[0] for a,b in zip([-np.inf,*borders],[*borders,np.inf]))
    tv,ks=summary.tail_distances(masses[0],masses[-1])
    assert abs(numerical_tv-tv)<1e-12
    def full_cdf(y):
        if y<=borders[1]:return masses[0]*(1-erf((borders[1]-y)/(scales[0]*np.sqrt(2))))
        if y<borders[-2]:return masses[0]+masses[1]*(y-borders[1])/(borders[-2]-borders[1])
        return 1-masses[-1]+masses[-1]*erf((y-borders[-2])/(scales[1]*np.sqrt(2)))
    points=np.unique(np.r_[np.linspace(-20,20,10001),borders])
    finite_cdf=module.finite_bar_cdf(masses[None,:],borders,points)[0]
    numerical_ks=np.max(np.abs(np.array([full_cdf(y) for y in points])-finite_cdf))
    assert abs(numerical_ks-ks)<1e-12


def test_tail_tv_ks_are_different_with_asymmetric_endpoint_masses():
    tv,ks=summary.tail_distances(np.array([.8,.1]),np.array([.1,.1]))
    np.testing.assert_allclose(tv,[.45,.1])
    np.testing.assert_allclose(ks,[.4,.05])
