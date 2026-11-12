"""Focused deterministic CPU checks; no posterior draws or saved cohorts needed."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
P=Path(__file__).resolve().parents[1]/"scripts/diagnose_case2_conditional_mean.py"
s=importlib.util.spec_from_file_location("_test_case2_attribution",P);d=importlib.util.module_from_spec(s);s.loader.exec_module(d)

def fixture_rows(n=9):
 x=np.zeros((n,21),dtype=np.float64)
 x[:,:3]=np.arange(n*3).reshape(n,3)/17-.4
 x[:,7:10]=np.arange(n*3).reshape(n,3)/23-.2
 x[:,10]=np.linspace(.021,.49,n)
 return x,np.linspace(-.7,.4,10)

def test_actual_geometry_conditional_not_marginal():
 g=d.geometry();assert g["global_indices"]==[0,1,2] and g["local_indices"]==[7,8,9] and g["sigma_index"]==10
 assert np.array_equal(g["prior"].covariance,np.diag(np.square([.45,.70,.25])))
 assert np.array_equal(g["forward"].design,np.column_stack((np.linspace(0,1,10),np.linspace(0,1,10)**2,np.ones(10))))
 assert not np.array_equal(g["prior"].covariance,g["prior"].covariance+g["prior"].population_map@g["problem"].prior.base.global_covariance@g["prior"].population_map.T)

def test_precision_matches_independent_covariance_space():
 x,y=fixture_rows();g=d.geometry();m,c=d.conditional_moments(x,y,g);a,b=d.covariance_space(x,y,g)
 np.testing.assert_allclose(m,a,rtol=0,atol=3e-12);np.testing.assert_allclose(c,b,rtol=0,atol=3e-12)
 assert np.all(np.linalg.eigvalsh(c)>0)

def test_actual_prior_likelihood_log_ratio():
 x,y=fixture_rows();g=d.geometry();m,c=d.conditional_moments(x,y,g)
 for k in (0,4,8):
  beta=np.array([.2,-.8,.13]);alt=np.array([-.1,.5,-.07]);h=x[k:k+1,:3]
  def f(v):
   mapping={name:np.array([v[j]]) for j,name in enumerate(g["prior"].local_names)};mapping["sigma_m2"]=x[k:k+1,10]
   return (g["problem"].prior.base.conditional_log_prob("mechanical_2",h,v[None,:])+g["problem"].observation_model.local_log_prob("mechanical_2",mapping,y[None,:]))[0]
  a=beta-m[k];b=alt-m[k];p=np.linalg.solve(c[k],np.eye(3))
  assert abs(f(beta)-f(alt)+.5*(a@p@a-b@p@b))<1e-10

def test_offsets_and_population_map_are_used():
 from dataclasses import replace
 g=d.geometry();g["prior"]=replace(g["prior"],population_map=np.array([[1,.2,0],[0,1,.3],[.1,0,1]]),offset=np.array([.3,-.1,.02]))
 g["forward"]=replace(g["forward"],offset=np.linspace(-.2,.3,10));g["V0_inverse"]=np.linalg.solve(g["prior"].covariance,np.eye(3))
 x,y=fixture_rows();m,c=d.conditional_moments(x,y,g);a,b=d.covariance_space(x,y,g)
 np.testing.assert_allclose(m,a,rtol=0,atol=4e-12);np.testing.assert_allclose(c,b,rtol=0,atol=4e-12)

def test_block_mean_independent_of_sampled_local_values_and_other_blocks():
 x,y=fixture_rows();a,b=d.conditional_moments(x,y);z=x.copy();z[:,3:10]+=50;z[:,11:]=-25
 c,e=d.conditional_moments(z,y);assert np.array_equal(a,c) and np.array_equal(b,e)

def test_pairing_and_signed_error_identity():
 beta=np.array([.2,.7,-.3,1.1,1.4]);m=np.array([.1,.9,.4,.3,.5]);a=d.paired_decomposition(beta,m,.8,2.,.0004,.0007)
 c=a["components"];r=c["accepted_sampler_conditional_residual"]["signed_refSD"];s=c["conditional_mean_mixing_discrepancy"]["signed_refSD"];t=c["total_signed_error"]["signed_refSD"]
 assert t<0 and abs(r+s-t)<1e-15 and a["MCSE_total_variance_identity_abs_error"]<1e-15
 paired=np.array(a["paired_mean_covariance_refSD2"]);assert abs(paired[0,1])>0
 assert c["accepted_sampler_conditional_residual"]["approximate_iid_row_MCSE_refSD"]>0

@pytest.mark.parametrize("value",[-.5,0.,.5])
def test_reference_signed_intervals(value):
 low,high=d.signed_numerical_interval(value,.002,.003)
 for mu in np.linspace(-.002,.002,5):
  for sd in np.linspace(-.003,.003,5):assert low-1e-15<=(value+mu)/(1+sd)<=high+1e-15
 assert low<=value<=high

def test_no_mutation_or_default_RNG_consumption():
 x,y=fixture_rows();before=x.copy();state=d.state_sha();d.conditional_moments(x,y)
 assert np.array_equal(x,before) and d.state_sha()==state

@pytest.mark.parametrize("kind",["float32","nan","negative_sigma"])
def test_invalid_inputs_rejected(kind):
 x,y=fixture_rows()
 if kind=="float32":x=x.astype(np.float32)
 elif kind=="nan":x[0,0]=np.nan
 else:x[0,10]=-.1
 with pytest.raises(ValueError):d.conditional_moments(x,y)
