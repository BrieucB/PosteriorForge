"""Deterministic exact-normalized captured-weight reference checks; no RNG tests."""
from fractions import Fraction
import hashlib,importlib.util,pickle,random
from pathlib import Path
import numpy as np
import pytest
path=Path(__file__).resolve().parents[1]/"scripts/prior_tail_icdf_reference_v1.py"
spec=importlib.util.spec_from_file_location("_tail_exact_reference",path);r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
def frac(d):return Fraction(int(d["numerator"]),int(d["denominator"]))
def arrays(p,b,u):return np.array(p,np.float32),np.array(b,np.float32),np.array(u,np.float32)

def test_manual_unequal_bar_quantiles_exact_bins_fractions_CDF():
 p,b,u=arrays([.25,.5,.25],[-2,0,1,5],[0,.125,.25,.5,.75,.875,1])
 out=r.reference_row(p,b,u)
 assert [q["bin"] for q in out["points"]]==[0,0,0,1,1,2,2]
 assert [q["inverse_nearest_Float64"] for q in out["points"]]==[-2,-1,0,.5,1,3,5]
 assert all(frac(q["exact_inverse_CDF_residual"])==0 for q in out["points"])
 assert out["points"][-1]["u1_out_of_contract_stress"] and not out["points"][-1]["inside_production_uniform_contract"]

def test_zero_mass_bins_and_plateau_ties_never_divide_by_zero():
 p,b,u=arrays([0,1,0,1,0],[0,1,2,5,8,9],[0,.5,.75,1])
 out=r.reference_row(p,b,u)
 assert out["essential_support_endpoints"]==[1,8] and out["outer_borders"]==[0,9]
 assert [q["bin"] for q in out["points"]]==[1,1,3,3]
 assert [q["inverse_nearest_Float64"] for q in out["points"]]==[1,2,6.5,8]
 assert all(q["selected_positive_weight"] for q in out["points"])

def test_existing_sum_is_normalized_not_assumed_one_and_scaling_invariant():
 p,b,u=arrays([.25,1,.75],[-90,-1,3,87],[.1,.5,.9999980926513672,.9999994039535522,1])
 a=r.reference_row(p,b,u);c=r.reference_row(p/np.float32(2),b,u)
 assert frac(a["input_weight_sum_exact"])==2 and frac(c["input_weight_sum_exact"])==1
 assert [q["inverse_exact"] for q in a["points"]]==[q["inverse_exact"] for q in c["points"]]

def test_actual_Float32_near_one_lattice_and_5000_bins_support_residual():
 p=np.full(5000,np.float32(1/5000));b=np.linspace(-90,87,5001,dtype=np.float32)
 u=np.array([np.float32(.9999980926513672),np.float32(.9999994039535522),np.nextafter(np.float32(1),np.float32(0)),np.float32(1)],np.float32)
 out=r.reference_row(p,b,u)
 assert frac(out["input_weight_sum_exact"])!=1
 for q in out["points"]:
  assert q["bin"]==4999 and q["fraction_in_closed_unit_interval"] and q["inside_outer_borders"]
  assert -90<=q["inverse_nearest_Float64"]<=87
  assert frac(q["exact_inverse_CDF_residual"])==0
  assert abs(frac(q["rounded_Float64_CDF_residual_exact"]))<=frac(q["rounded_CDF_residual_bound_exact"])

def test_subnormal_weights_are_exact_dyadic_integers():
 tiny=np.nextafter(np.float32(0),np.float32(1))
 p,b,u=arrays([tiny,2*tiny],[0,2,5],[.5])
 q=r.reference_row(p,b,u)["points"][0]
 assert q["bin"]==1 and frac(q["interpolation_fraction_exact"])==Fraction(1,4)
 assert frac(q["inverse_exact"])==Fraction(11,4)

def test_tiny_weight_is_not_lost_by_Float64_normalization():
 tiny=np.nextafter(np.float32(0),np.float32(1))
 p,b,u=arrays([tiny,1],[0,1,2],[tiny])
 out=r.reference_row(p,b,u);q=out["points"][0]
 assert frac(out["input_weight_sum_exact"])==1+Fraction(1,1<<149)
 assert q["bin"]==1 and frac(q["interpolation_fraction_exact"])>0
 assert frac(q["inverse_exact"])>1 and q["inverse_nearest_Float64"]==1
 assert frac(q["rounded_Float64_CDF_residual_exact"])<0

def test_observed_extrapolation_is_not_repaired_and_CDF_residual_is_not_support_test():
 p,b,u=arrays([.5,.5],[-1,0,1],[.9999994039535522,1])
 observed=np.array([205.2462615966797,205.2462615966797],np.float32)
 before=observed.tobytes();out=r.reference_row(p,b,u,observed_values=observed)
 assert observed.tobytes()==before
 for q in out["points"]:
  x=q["observed_frozen_output"];assert not x["inside_outer_borders"] and x["unchanged_observed_value"]>1
 assert frac(out["points"][0]["observed_frozen_output"]["normalized_reference_CDF_residual_exact"])==1-Fraction.from_float(float(u[0]))
 assert frac(out["points"][1]["observed_frozen_output"]["normalized_reference_CDF_residual_exact"])==0
 assert not out["points"][1]["observed_frozen_output"]["inside_positive_mass_support"]

def test_zero_mass_support_holes_are_distinguished():
 p,b,u=arrays([1,0,1],[0,1,2,3],[.5])
 out=r.reference_row(p,b,u,observed_values=np.array([1.5],np.float32))
 assert out["points"][0]["observed_frozen_output"]["inside_outer_borders"]
 assert not out["points"][0]["observed_frozen_output"]["inside_positive_mass_support"]

@pytest.mark.parametrize("bad",["negative","nan","inf","zero","dtype","borders","u_negative","u_high","u_nan","u_dtype"])
def test_invalid_inputs_fail_without_clipping(bad):
 p,b,u=arrays([.25,.75],[0,1,2],[.5])
 if bad=="negative":p[0]=-.1
 elif bad=="nan":p[0]=np.nan
 elif bad=="inf":p[0]=np.inf
 elif bad=="zero":p[:]=0
 elif bad=="dtype":p=p.astype(np.float64)
 elif bad=="borders":b[1]=b[0]
 elif bad=="u_negative":u[0]=-.1
 elif bad=="u_high":u[0]=1.1
 elif bad=="u_nan":u[0]=np.nan
 else:u=u.astype(np.float64)
 with pytest.raises(ValueError):r.reference_row(p,b,u)

def test_no_RNG_softmax_or_input_mutation(monkeypatch):
 p,b,u=arrays([.1,.7,.2],[-3,-1,2,10],[0,.5,.9999994039535522,1])
 saved=[x.tobytes() for x in (p,b,u)]
 state=pickle.dumps((np.random.get_state(),random.getstate()),protocol=4)
 def forbidden(*args,**kwargs):raise AssertionError("RNG/softmax exponential forbidden")
 for name in ["rand","random","normal","default_rng"]:monkeypatch.setattr(np.random,name,forbidden)
 monkeypatch.setattr(random,"random",forbidden);monkeypatch.setattr(np,"exp",forbidden)
 out=r.reference_row(p,b,u)
 assert state==pickle.dumps((np.random.get_state(),random.getstate()),protocol=4)
 assert saved==[x.tobytes() for x in (p,b,u)]
 assert out["no_clipping_or_repair_or_RNG"] and out["not_the_frozen_Float32_cumulative_roundoff_law"]
