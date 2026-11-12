"""Exact sufficient-statistic algebra/measure and existing-data CPU checks."""
import hashlib,json
from pathlib import Path
import numpy as np
import pytest
from scipy.stats import norm,chi2
from posteriorforge import representation_ss15 as ss
from posteriorforge.science.analytical_benchmark_v2 import build_problem,unflatten_observations

def test_fixed_integer_basis_is_orthonormal_and_spans_frozen_rank3_design():
 p=build_problem();q=ss.orthonormal_basis();x=p.observation_model.blocks[0].design
 np.testing.assert_allclose(q.T@q,np.eye(3),atol=5e-16,rtol=0)
 np.testing.assert_allclose(q@(q.T@x),x,atol=8e-16,rtol=0)
 assert not q.flags.writeable
 assert np.linalg.matrix_rank(x)==3 and ss.RESIDUAL_DOF==10-3
 assert ss.FEATURE_ORDER[:4]==("mechanical_1.signal_constant","mechanical_1.signal_linear",
  "mechanical_1.signal_quadratic","mechanical_1.residual_energy")
 assert ss.FEATURE_ORDER[-3:]==("acoustic_1.raw","acoustic_2.raw","acoustic_3.raw")

def observations():
 return (np.arange(11*33,dtype=np.float64).reshape(11,33)/40
         +np.sin(np.arange(11*33,dtype=np.float64).reshape(11,33)))

def test_transform_has_no_fitted_state_keeps_acoustics_and_single_final_cast():
 raw=observations();before=raw.copy();stats=ss.transform_observations(raw)
 assert stats.shape==(11,15) and stats.dtype==np.float64 and stats.flags.c_contiguous
 assert np.array_equal(stats[:,-3:],raw[:,-3:]) and np.array_equal(raw,before)
 model=ss.transform_for_model(raw)
 assert model.dtype==np.float32 and np.array_equal(model,stats.astype(np.float32))
 assert np.all(stats[:,[3,7,11]]>=0)
 assert not np.shares_memory(stats,raw)

def test_feature_bytes_batch_chunk_row_order_and_memory_layout_invariant():
 raw=observations();whole=ss.transform_observations(raw)
 pieces=np.concatenate([ss.transform_observations(raw[:1]),ss.transform_observations(raw[1:5]),ss.transform_observations(raw[5:])])
 singles=np.concatenate([ss.transform_observations(row[None,:]) for row in raw])
 assert whole.tobytes()==pieces.tobytes()==singles.tobytes()
 assert ss.transform_observations(np.asfortranarray(raw)).tobytes()==whole.tobytes()
 order=np.arange(len(raw))[::-1]
 assert np.array_equal(ss.transform_observations(raw[order]),whole[order])

@pytest.mark.parametrize("bad",["float32","columns","empty","nan","inf","vector"])
def test_original_float64_shape_and_finite_inputs_are_required_before_model_cast(bad):
 raw=observations()
 if bad=="float32":raw=raw.astype(np.float32)
 elif bad=="columns":raw=raw[:,:32]
 elif bad=="empty":raw=raw[:0]
 elif bad=="vector":raw=raw[0]
 else:raw[0,0]=np.nan if bad=="nan" else np.inf
 with pytest.raises(ValueError):ss.transform_observations(raw)

def theta_rows():
 p=build_problem();theta=np.zeros((9,21),np.float64)
 for j,name in enumerate(p.parameter_order):
  theta[:,j]=np.linspace(-.2,.3,len(theta))+j*.013
 for k,name in enumerate(p.prior.noise_names):theta[:,p.parameter_order.index(name)]=np.linspace(.01,.5,len(theta))
 return theta

def test_raw_normalized_likelihood_and_ratios_reconstructed_including_all6_noise_scales():
 p=build_problem();theta=theta_rows();obs=observations()[:1];stats=ss.transform_observations(obs)
 raw=p.log_likelihood(theta,unflatten_observations(obs));reduced=ss.raw_likelihood_factor(theta,stats)
 np.testing.assert_allclose(reduced,raw,rtol=5e-14,atol=1e-9)
 np.testing.assert_allclose(reduced-reduced[-1],raw-raw[-1],rtol=5e-14,atol=1e-9)

def test_summary_density_matches_gaussian_signal_times_scaled_chi_square_energy():
 p=build_problem();theta=theta_rows();raw=observations()[:1];s=ss.transform_observations(raw);q=ss.orthonormal_basis()
 independent=np.zeros(len(theta))
 for i,(block,name) in enumerate(zip(p.observation_model.blocks,p.prior.noise_names)):
  beta=theta[:,[p.parameter_order.index(n) for n in block.local_names]]
  sigma=theta[:,p.parameter_order.index(name)]
  if i<3:
   mean=beta@(q.T@block.design).T
   independent+=norm.logpdf(s[:,4*i:4*i+3],mean,sigma[:,None]).sum(1)
   independent+=chi2.logpdf(s[:,4*i+3]/sigma**2,7)-2*np.log(sigma)
  else:
   mean=beta@block.design.T+block.offset
   independent+=norm.logpdf(s[:,12+i-3:13+i-3],mean,sigma[:,None]).sum(1)
 actual=ss.summary_log_density(theta,s)
 np.testing.assert_allclose(actual,independent,rtol=5e-14,atol=1e-9)
 factor=ss.residual_measure_log_factor(s)
 np.testing.assert_allclose(actual-ss.raw_likelihood_factor(theta,s),np.repeat(factor,len(theta)),rtol=0,atol=1e-9)

def test_residual_energy_carries_noise_information_that_signal_alone_loses():
 q=ss.orthonormal_basis();v=np.array([1,-3,3,-1,0,0,0,0,0,0],np.float64)/np.sqrt(20)
 np.testing.assert_allclose(q.T@v,0,atol=2e-16,rtol=0)
 raw=np.zeros((2,33),np.float64);raw[0,:10]=v*.1;raw[1,:10]=v*.7
 s=ss.transform_observations(raw)
 np.testing.assert_allclose(s[0,:3],s[1,:3],atol=5e-17,rtol=0)
 np.testing.assert_allclose(s[:,3],[.01,.49],atol=3e-16,rtol=0)
 theta=theta_rows()[:2];p=build_problem();theta[:,p.parameter_order.index("sigma_m1")]=[.1,.4]
 first=ss.raw_likelihood_factor(theta,s[:1]);second=ss.raw_likelihood_factor(theta,s[1:])
 expected=-(.49-.01)/(2*theta[:,p.parameter_order.index("sigma_m1")]**2)
 np.testing.assert_allclose(second-first,expected,atol=1e-10,rtol=1e-13)
 assert not np.isclose(second[0]-first[0],second[1]-first[1])

def test_zero_energy_boundary_retained_without_clip_or_nan():
 raw=np.zeros((1,33),np.float64);s=ss.transform_observations(raw)
 assert np.array_equal(s[:,[3,7,11]],np.zeros((1,3)))
 assert np.isfinite(ss.raw_likelihood_factor(theta_rows(),s)).all()
 assert np.isneginf(ss.summary_log_density(theta_rows(),s)).all()

@pytest.mark.parametrize("bad",["negative_energy","wrongshape","nonfinite","pairing"])
def test_invalid_summaries_do_not_receive_silent_repairs(bad):
 s=ss.transform_observations(observations()[:1]);theta=theta_rows()
 if bad=="negative_energy":s[0,3]=-1e-100
 elif bad=="wrongshape":s=s[:,:14]
 elif bad=="nonfinite":s[0,3]=np.inf
 else:s=np.repeat(s,2,axis=0)
 with pytest.raises(ValueError):ss.raw_likelihood_factor(theta,s)

def test_nonpositive_noise_returns_negative_infinite_like_original_likelihood():
 p=build_problem();theta=theta_rows();theta[:,p.parameter_order.index("sigma_m1")]=0
 obs=observations()[:1];stats=ss.transform_observations(obs)
 assert np.isneginf(ss.raw_likelihood_factor(theta,stats)).all()
 assert np.isneginf(p.log_likelihood(theta,unflatten_observations(obs))).all()

def test_metadata_cache_identity_has_geometry_source_order_dtype_and_all_globals():
 metadata=ss.representation_metadata();p=build_problem();raw=observations()
 assert metadata["raw_dimension"]==33 and metadata["observation_dimension"]==15
 assert metadata["physical_parameter_order"]==list(p.parameter_order)
 assert metadata["global_parameters_retained"]==["mu_a","mu_b","mu_d0"]
 assert metadata["noise_parameters_retained"]==list(p.prior.noise_names)
 assert metadata["module_sha256"]==ss.source_sha256()
 assert metadata["statistics_dtype"]=="float64" and len(metadata["basis_float64"]["sha256"])==64
 assert not metadata["prior_changed"] and not metadata["target_transform"] and not metadata["sampler_changed"]
 key=ss.representation_cache_key(raw)
 assert key==ss.representation_cache_key(np.asfortranarray(raw))
 changed=raw.copy();changed[0,0]=np.nextafter(changed[0,0],np.inf)
 assert key!=ss.representation_cache_key(changed)
 json.dumps(metadata,allow_nan=False)

def test_all_saved_banks_and8_heldout_cases_keep_likelihood_ratios_and_acoustic_features():
 from posteriorforge.banks import load_context,load_heldout
 root=Path("/scratch/project/eu-26-17/eubrieucb/posteriorforge/results/banks-v1")
 if not root.is_dir():pytest.skip("immutable HPC banks unavailable")
 p=build_problem()
 for seed in (20261111,20261112,20261113):
  bank=load_context(root,seed,4096);s=ss.transform_observations(bank.obs)
  assert np.array_equal(s[:,-3:],bank.obs[:,-3:])
  paired_raw=p.log_likelihood(bank.theta,unflatten_observations(bank.obs))
  paired_ss=ss.raw_likelihood_factor(bank.theta,s)
  np.testing.assert_allclose(paired_raw,paired_ss,rtol=8e-14,atol=2e-10)
  for case in range(8):
   obs=load_heldout(root,case).obs.reshape(1,33)
   raw=p.log_likelihood(bank.theta,unflatten_observations(obs))
   reduced=ss.raw_likelihood_factor(bank.theta,ss.transform_observations(obs))
   np.testing.assert_allclose(raw,reduced,rtol=8e-14,atol=2e-9)
