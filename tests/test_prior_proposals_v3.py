"""Deterministic scalar/engineering proposal checks; no model/RNG/posterior."""
import gzip
import hashlib
import json
import math
import pickle
import random
from pathlib import Path
import numpy as np
import pytest
from posteriorforge import prior_proposals_v3 as p
from posteriorforge import prior_coordinates_v1 as old
from posteriorforge.inference import support_mask

def latent(n=7,dtype=np.float64):
    t=np.arange(n,dtype=np.float64)
    z=np.empty((n,21),dtype=np.float64)
    for j in range(21): z[:,j]=.013*t-.027*j
    return z.astype(dtype)

def physical(n=7):
    z=latent(n)
    return old.decode_ordered(z)

def test_exact_private_codeobjects_and_strict_training_aliases():
    assert p._NOISE.__code__ is old._noise_decode.__code__
    assert p._PREFIX.__code__ is old._decode_prefix.__code__
    assert p._NOISE.__globals__ is not old._noise_decode.__globals__
    assert p._PREFIX.__globals__ is not old._decode_prefix.__globals__
    assert old._noise_decode.__globals__["_strict_noise"] is old._strict_noise
    assert p.training_encode_for_model is old.encode_for_model
    assert p.training_encode_physical is old.encode_physical
    assert p.require_sources()["original_namespace_mutated"] is False

def test_training_theta_to_latent_to_theta_and_finalF32bytes_unchanged():
    x=physical();saved=x.tobytes()
    a=p.training_encode_physical(x);b=old.encode_physical(x)
    assert a.tobytes()==b.tobytes()
    assert p.training_encode_for_model(x).tobytes()==old.encode_for_model(x).tobytes()
    assert p.training_encode_for_model(x).dtype==np.float32
    assert p.decode_ordered_proposals(p.training_encode_for_model(x)[:,old.ORDER]).shape==x.shape
    assert x.tobytes()==saved

@pytest.mark.parametrize("slot",old.NOISE_INDICES)
@pytest.mark.parametrize("endpoint",[.01,.5])
def test_training_endpoints_remain_strict(slot,endpoint):
    x=physical();x[0,slot]=endpoint
    with pytest.raises(old.CoordinateError):p.training_encode_for_model(x)

@pytest.mark.parametrize("dtype",[np.float32,np.float64])
def test_moderate_decode_bytes_identical_and_chunk_invariant(dtype):
    z=latent(9,dtype);before=z.tobytes()
    x=p.decode_ordered_proposals(z)
    assert x.dtype==np.float64 and x.flags.c_contiguous
    assert x.tobytes()==old.decode_ordered(z).tobytes()
    assert x.tobytes()==np.concatenate([p.decode_ordered_proposals(z[:3]),p.decode_ordered_proposals(z[3:])]).tobytes()
    assert z.tobytes()==before

def test_all21_prefixes_no_future_columns():
    z=latent();full=p.decode_ordered_proposals(z)
    for width in range(1,22):
        q=p.decode_ordered_prefix_proposals(z[:,:width])
        assert q.tobytes()==full[:,old.ORDER[:width]].copy().tobytes()
    z[:,20]=np.nan
    assert p.decode_ordered_prefix_proposals(z[:,:20]).shape==(len(z),20)

def test_computed_endpoints_and_below_support_retained_not_replaced():
    z=np.zeros((7,21),np.float64)
    z[:,3]=[-205.2462615966797,-13.876729011535645,-8.,0.,8.,13.876729011535645,205.2462615966797]
    original=z.tobytes();x=p.decode_ordered_proposals(z)
    g=old._geometry();expected=p._NOISE(z[:,3],g["noise"][6])
    assert x[:,6].tobytes()==expected.tobytes() and z.tobytes()==original
    assert x[-1,6]==.5 and x[-2,6]==.5
    assert x[0,6]<.01 and x[1,6]<.01
    mask=support_mask(x)
    assert mask.tolist()==[False,False,True,True,True,True,True]
    # The decoder returns all7 rows; original whole-row support rejects2 later.
    assert len(x)==7 and len(mask)==7
    report=p.precision_report(z,x)
    assert report["hard_passed"] and not report["acceptance_applied"]
    assert report["original_closed_support_finite_proposals_only"]["noise_outside_rows"]==2
    with pytest.raises(old.CoordinateError):old.decode_ordered(z)

@pytest.mark.parametrize("value",[float("nan"),float("inf"),float("-inf")])
@pytest.mark.parametrize("stage",[0,3,20])
def test_nonfinite_latent_never_converted_to_finite_endpoint(value,stage):
    z=latent(2);z[0,stage]=value
    with pytest.raises(old.CoordinateError,match="nonfinite"):p.decode_ordered_proposals(z)
    report=p.precision_report(z)
    assert report["hard_passed"] is False and report["rows_filtered"]==0

def test_bad_shapes_and_implicit_casts_refused():
    with pytest.raises(old.CoordinateError):p.decode_ordered_proposals(np.zeros((2,20),np.float64))
    with pytest.raises(old.CoordinateError):p.decode_ordered_proposals(np.zeros((2,21),np.int64))
    with pytest.raises(old.CoordinateError):p.decode_ordered_prefix_proposals(np.empty((2,0),np.float64))

def test_genuine_arithmetic_overflow_hard_fails():
    z=np.full((1,21),np.finfo(np.float64).max,np.float64)
    with np.errstate(over="ignore",invalid="ignore"):
        with pytest.raises(old.CoordinateError):p.decode_ordered_proposals(z)

def test_precision_validator_catches_corrupt_physical_noise_and_affine():
    z=latent();x=p.decode_ordered_proposals(z)
    a=x.copy();a[:,6]+=1e-8
    assert p.precision_report(z,a)["hard_passed"] is False
    b=x.copy();b[:,3]+=1e-4
    assert p.precision_report(z,b)["hard_passed"] is False

def test_directed_validation_edges_satisfy_exact_physical_bound_no_output_mutation():
    spec=old._geometry()["noise"][6]
    y=np.array([.01,.255,.5],np.float64);before=y.tobytes()
    edges=p._physical_error_interval(y,spec)
    c=p._fraction(old.F64_PHYSICAL_UNITS);mu=p._fraction(spec["location"]);s=p._fraction(spec["scale"])
    for value,pair in zip(y,edges):
        v=p._fraction(value)
        for probe in pair:
            expected=p._fraction(probe)
            assert abs(v-expected)/s <= c*(1+abs((expected-mu)/s))
    assert y.tobytes()==before
    assert p.metadata()["physical_error_validation"]["directed_edges_never_modify_proposals"]

def test_source_guard_rejects_private_namespace_mutation(monkeypatch):
    with monkeypatch.context() as m:
        m.setitem(p._NOISE_NS,"_strict_noise",lambda *args:None)
        with pytest.raises(old.CoordinateError,match="namespace"):p.require_sources()
    assert p.require_sources()

def test_no_default_RNG_draw_or_input_mutation():
    before=(pickle.dumps(np.random.get_state()),pickle.dumps(random.getstate()))
    z=latent();saved=z.tobytes()
    x=p.decode_ordered_proposals(z);r=p.precision_report(z,x)
    assert r["hard_passed"] and saved==z.tobytes()
    assert before==(pickle.dumps(np.random.get_state()),pickle.dumps(random.getstate()))

def saved_policyC():
    root=Path(__file__).resolve().parents[1]
    f=root/"evidence/accuracy-prior-coordinates-v1/tail-postprocessing-offline-01/exact-report.json.gz"
    raw=gzip.decompress(f.read_bytes())
    assert hashlib.sha256(raw).hexdigest()=="cbc8e44e62780d48e594845bd63a9c72faa76eaf7a1459d9e4dbb2210e989a5a"
    d=json.loads(raw)
    return [next(p0 for p0 in r["variants"]["C_softmax64_adjusted_F32logits_F64log_CPUsoftmax64"]["points"]
                 if p0["U_bits"]=="f6ff7f3f")["value"]
            for r in d["records"] if r["stage"]==19 and r["label"]=="N1024"]

def test_all16_saved_C_interiors_validate_physical_precision_without_unattainable_latent_RT():
    values=saved_policyC();assert len(values)==16
    # Tiny engineering vectors only: one saved scalar with zero other latent coords.
    z=np.zeros((16,21),np.float64);z[:,19]=values
    x=p.decode_ordered_proposals(z)
    assert np.all((x[:,20]>.01)&(x[:,20]<.5))
    assert p.precision_report(z,x)["hard_passed"]
    spec=old._geometry()["noise"][20]
    replay=old._noise_encode(x[:,20],spec)
    assert np.all(np.abs(replay-np.array(values))>old.F32_LATENT_ADMISSIBILITY*(1+np.abs(values)))
    assert p.metadata()["noise_latent_to_physical_not_lossless_near_bounds"]

def test_source_guard_rejects_new_callable_replacement(monkeypatch):
    with monkeypatch.context() as m:
        m.setattr(p,"_physical_error_interval",lambda y,s:np.column_stack((y-1.,y+1.)))
        with pytest.raises(old.CoordinateError,match="callable"):p.require_sources()
    assert p.require_sources()

def test_readonly_inputs_preserved_and_precision_output_never_used_as_acceptance():
    z=latent(2);z.setflags(write=False)
    x=p.decode_ordered_proposals(z)
    r=p.precision_report(z,x)
    assert r["hard_passed"] and r["acceptance_applied"] is False
    assert z.flags.writeable is False and len(x)==len(z)


def test_near_physical_budget_cells_fail_closed_without_false_acceptance():
    from fractions import Fraction
    z=np.zeros((1,21),np.float64);base=p.decode_ordered_proposals(z)
    spec=old._geometry()["noise"][6]
    mu=Fraction.from_float(spec["location"]);sd=Fraction.from_float(spec["scale"])
    budget=Fraction.from_float(old.F64_PHYSICAL_UNITS)*sd
    rejected_outside=0;accepted_inside=0
    for direction in (-1,1):
        center=float(mu+direction*budget)
        for steps in (-2,-1,0,1,2):
            value=center
            for _ in range(abs(steps)):
                value=math.nextafter(value,math.inf if steps>0 else -math.inf)
            within=abs(Fraction.from_float(value)-mu)<=budget
            x=base.copy();x[0,6]=value
            result=p.precision_report(z,x)
            if result["hard_passed"]:
                assert within   # No observed widening by numerical logCDF.
                accepted_inside+=1
            if not within:
                assert result["hard_passed"] is False
                rejected_outside+=1
            if steps==-direction:
                # One inward physical cell from the nearest boundary: margin
                # avoids confusing conservative inward probe rounding with a
                # universal exact-interval acceptance claim.
                assert within and result["hard_passed"]
    assert rejected_outside==4 and accepted_inside>=4
