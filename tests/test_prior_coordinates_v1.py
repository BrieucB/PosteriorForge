"""CPU-only causal transport checks; no posterior or reference fitting."""
import hashlib
import json
from pathlib import Path
import pickle
import random
import subprocess
import sys
import numpy as np
import pytest
from scipy.stats import norm,truncnorm
from posteriorforge import prior_coordinates_v1 as c

def physical_rows(n=7):
    t=np.arange(n,dtype=np.float64)
    z=np.column_stack((.17*t-.4, .3-.11*t, -.2+.07*t))
    covariance=np.array([[1.,.12,.02],[.12,1.2,-.03],[.02,-.03,.4]])
    g=np.array([.3,-.2,.1])+z@np.linalg.cholesky(covariance).T
    values=np.zeros((n,21),dtype=np.float64);values[:,:3]=g
    residual=np.column_stack((.2+.03*t,-.4+.01*t,.1-.02*t))
    for start in (3,7,11):
        values[:,start:start+3]=g+residual*np.array([.45,.7,.25])
    for start in (15,17,19):
        values[:,start]=g[:,0]+.45*(.15-.02*t)
    for offset,index in enumerate(c.NOISE_INDICES):
        values[:,index]=.08+.035*t+.003*offset
    return values

def test_global_cholesky_matches_independent_linear_solve():
    p=physical_rows();z=c.encode_physical(p)
    cov=np.array([[1.,.12,.02],[.12,1.2,-.03],[.02,-.03,.4]])
    expected=np.linalg.solve(np.linalg.cholesky(cov),(p[:,:3]-[.3,-.2,.1]).T).T
    np.testing.assert_allclose(z[:,:3],expected,rtol=0,atol=5e-16)
    assert c.metadata()['global_lower_cholesky'][1][0]!=0
    np.testing.assert_allclose(z[:,0],p[:,0]-.3,rtol=0,atol=0)

def test_conditional_local_residual_uses_D_not_marginal_covariance():
    p=physical_rows();z=c.encode_physical(p)
    expected=(p[:,3:6]-p[:,:3])/np.array([.45,.7,.25])
    np.testing.assert_allclose(z[:,3:6],expected,rtol=0,atol=8e-16)
    marginal=np.sqrt(np.array([1.,1.2,.4])+np.array([.45,.7,.25])**2)
    assert not np.allclose(z[:,3:6],(p[:,3:6]-p[:,:3])/marginal)
    for start in (7,11):np.testing.assert_array_equal(z[:,start:start+3],z[:,3:6])

def test_noise_pit_matches_independent_public_cdf_inverse():
    p=physical_rows()
    p[:,6]=[.03,.1,.2,.255,.31,.4,.49]
    z=c.encode_physical(p)
    expected=norm.ppf(truncnorm.cdf(p[:,6],-2.,2.,loc=.255,scale=.1225))
    np.testing.assert_allclose(z[:,6],expected,rtol=0,atol=3e-14)

def test_noise_inverse_matches_independent_public_ppf():
    z=np.zeros((7,21),dtype=np.float64)
    # values are in STAGE order: first local sigma is stage3.
    z[:,3]=np.linspace(-3,3,7)
    physical=c.decode_ordered(z)
    expected=truncnorm.ppf(norm.cdf(z[:,3]),-2.,2.,loc=.255,scale=.1225)
    np.testing.assert_allclose(physical[:,6],expected,rtol=0,atol=4e-16)

def test_all_causal_prefixes_match_complete_decode_without_future_values():
    values=physical_rows();ordered=c.order_latents(c.encode_physical(values))
    complete=c.decode_ordered(ordered)
    for width in range(1,22):
        partial=c.decode_ordered_prefix(ordered[:,:width])
        np.testing.assert_array_equal(partial,complete[:,c.ORDER[:width]])
    future_bad=ordered.copy();future_bad[:,20]=np.nan
    np.testing.assert_array_equal(c.decode_ordered_prefix(future_bad[:,:20]),complete[:,c.ORDER[:20]])
    with pytest.raises(c.CoordinateError):c.decode_ordered(future_bad)

def test_chunk_invariance_and_inputs_unmodified():
    physical=physical_rows(11);original=physical.tobytes()
    latent=c.encode_physical(physical);model=c.encode_for_model(physical)
    np.testing.assert_array_equal(latent,np.concatenate([c.encode_physical(physical[:3]),c.encode_physical(physical[3:])]))
    np.testing.assert_array_equal(model,np.concatenate([c.encode_for_model(physical[:3]),c.encode_for_model(physical[3:])]))
    assert physical.tobytes()==original and model.dtype==np.float32 and model.flags.c_contiguous
    report=c.roundtrip_report(physical)
    assert report['Float64_roundtrip_passed'] and report['model_Float32_cast_roundtrip_passed']

@pytest.mark.parametrize('index',c.NOISE_INDICES)
@pytest.mark.parametrize('endpoint',[.01,.5])
def test_training_endpoints_fail_whole_input(index,endpoint):
    p=physical_rows();p[2,index]=endpoint
    with pytest.raises(c.CoordinateError,match='endpoint'):c.encode_for_model(p)

@pytest.mark.parametrize('value',[float('nan'),float('inf'),float('-inf')])
@pytest.mark.parametrize('stage',[0,3,4])
def test_nonfinite_latents_cannot_be_decoded_to_accepted_endpoints(stage,value):
    z=np.zeros((3,21),dtype=np.float32);z[1,stage]=value
    with pytest.raises(c.CoordinateError,match='nonfinite'):c.decode_ordered(z)

@pytest.mark.parametrize('value',[-100.,100.])
def test_finite_tail_physical_saturation_fails_without_clipping(value):
    z=np.zeros((1,21),dtype=np.float64);z[0,3]=value
    with pytest.raises(c.CoordinateError,match='saturated'):c.decode_ordered(z)

def test_interior_CDF_saturation_is_not_repaired():
    p=physical_rows();p[0,6]=np.nextafter(np.float64(.01),np.float64(.5))
    with pytest.raises(c.CoordinateError):c.encode_physical(p)

def test_implicit_training_cast_and_wrong_prefix_shape_refused():
    with pytest.raises(c.CoordinateError):c.encode_physical(physical_rows().astype(np.float32))
    with pytest.raises(c.CoordinateError):c.decode_ordered(np.zeros((3,20),dtype=np.float32))
    with pytest.raises(c.CoordinateError):c.decode_ordered_prefix(np.empty((3,0),dtype=np.float64))

def test_roundtrip_guard_catches_corrupted_decoder(monkeypatch):
    original=c._noise_decode
    monkeypatch.setattr(c,'_noise_decode',lambda values,spec:original(values,spec)+1e-5)
    with pytest.raises(c.CoordinateError,match='physical roundtrip'):c.encode_physical(physical_rows())

def test_source_guard_rejects_different_scientific_source(monkeypatch):
    monkeypatch.setattr(c,'PINNED_SOURCES',{'science/analytical_benchmark_v2.py':'0'*64})
    with pytest.raises(c.CoordinateError,match='source changed'):c.encode_physical(physical_rows())

def test_metadata_units_tolerances_meaning_and_json_serializability():
    m=c.metadata();json.dumps(m,allow_nan=False)
    assert m['physical_roundtrip_prior_units'][:3]==[1.,np.sqrt(1.2),np.sqrt(.4)]
    assert m['physical_roundtrip_prior_units'][3:6]==[.45,.7,.25]
    assert m['physical_roundtrip_centers'][3:6]==[.3,-.2,.1]
    assert all(m['physical_roundtrip_prior_units'][k]==.1225 for k in c.NOISE_INDICES)
    assert m['tolerances']['F64_physical_prior_units']==256*np.finfo(np.float64).eps
    assert m['tolerances']['F64_latent_conditioning_admissibility']==8*np.sqrt(np.finfo(np.float64).eps)
    assert 'NOT an ulp' in m['tolerances']['F64_latent_scope']
    assert m['first_global_density_correction_claimed'] is False
    assert m['old_native_fallback_policy_transferred'] is False
    assert len(m['coordinates'])==21 and m['coordinates'][6]['stage']==3

def test_no_default_rng_consumption():
    before=(pickle.dumps(np.random.get_state()),pickle.dumps(random.getstate()))
    c.roundtrip_report(physical_rows())
    assert before==(pickle.dumps(np.random.get_state()),pickle.dumps(random.getstate()))

def test_G3_refresh_preserves_decoded_physical_locals_and_requires_reencoding():
    from posteriorforge import conditional_globals
    physical=c.decode_ordered(c.order_latents(c.encode_physical(physical_rows())))
    latent=c.encode_physical(physical)
    refreshed=conditional_globals.refresh(physical,np.random.default_rng(1729))
    assert refreshed[:,3:].tobytes()==physical[:,3:].tobytes()
    reencoded=c.encode_physical(refreshed)
    assert not np.array_equal(reencoded[:,3:6],latent[:,3:6])
    wrong=latent.copy();wrong[:,:3]=reencoded[:,:3]
    assert c.decode_ordered(c.order_latents(wrong))[:,3:].tobytes()!=physical[:,3:].tobytes()

def test_no_torch_import_in_codec_subprocess():
    code="import sys; import posteriorforge.prior_coordinates_v1; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable,'-c',code],check=True)
