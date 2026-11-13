#!/usr/bin/env python3
"""Diagnostic exact-normalized inverse of EXISTING captured Float32 bar weights.

Built with PriorLabs-TabPFN. Independent CPU arithmetic only: no softmax, neural
model, frozen sampler, RNG or production-policy import/call. This is NOT the
frozen Float32 cumulative-roundoff law and not softmax recomputed in Float64.
"""
from __future__ import annotations
from bisect import bisect_left,bisect_right
from fractions import Fraction
import hashlib,math
from pathlib import Path
import numpy as np

SCHEMA="posteriorforge.captured_Float32_bar_exact_normalized_reference.v1"

def source_sha256():return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
def require(value,message):
 if not value:raise ValueError(message)
def ratio(value):return {"numerator":str(value.numerator),"denominator":str(value.denominator)}
def exact(value):return Fraction.from_float(float(value))
def _digest(values):return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()

def _mass_units(probabilities):
 """Every finite nonnegative binary32 value is an integer multiple of2^-149."""
 units=[]
 for word in np.ascontiguousarray(probabilities).view(np.uint32):
  bits=int(word);exponent=(bits>>23)&255;mantissa=bits&((1<<23)-1)
  units.append(mantissa if exponent==0 else ((1<<23)|mantissa)<<(exponent-1))
 return units

def _cdf(value,borders,units,prefix,total):
 if value<=borders[0]:return Fraction(0)
 if value>=borders[-1]:return Fraction(1)
 j=bisect_right(borders,value)-1
 before=prefix[j]-units[j]
 return (Fraction(before)+units[j]*(value-borders[j])/(borders[j+1]-borders[j]))/total

def _support(value,borders,units):
 if value<borders[0] or value>borders[-1]:return False
 k=bisect_left(borders,value)
 if k<len(borders) and borders[k]==value:
  return bool((k>0 and units[k-1]>0) or (k<len(units) and units[k]>0))
 j=bisect_right(borders,value)-1
 return 0<=j<len(units) and units[j]>0

def _quantization_bound(real,rounded):
 value=exact(rounded)
 if real==value:return Fraction(0)
 toward=math.inf if real>value else -math.inf
 adjacent=math.nextafter(rounded,toward)
 require(math.isfinite(adjacent),"reference Float64 rounding neighbor is nonfinite")
 return abs(exact(adjacent)-value)/2

def reference_row(probabilities_F32,borders_F32,uniforms_F32,*,observed_values=None):
 """Normalize captured p EXACTLY, then invert its finite uniform-bar mixture.

 For0<u<1 use the left generalized inverse, including exact plateau ties.
 u0/u1 are explicit essential-support endpoint extensions. u1 is OUTSIDE the
 frozen production[0,1) contract. Zero-mass bars are skipped, never divided by.
 Inputs/observed outputs are not clipped, repaired, reordered or mutated.
 """
 require(isinstance(probabilities_F32,np.ndarray) and probabilities_F32.dtype==np.float32
  and probabilities_F32.ndim==1 and len(probabilities_F32)>0,"captured nonempty Float32 weight row required")
 require(isinstance(borders_F32,np.ndarray) and borders_F32.dtype==np.float32
  and borders_F32.shape==(len(probabilities_F32)+1,),"captured Float32 borders[bin+1] required")
 require(np.isfinite(probabilities_F32).all() and (probabilities_F32>=0).all(),"weights must be finite/nonnegative")
 require(np.isfinite(borders_F32).all() and (borders_F32[1:]>borders_F32[:-1]).all(),"finite strictly increasing borders required")
 require(isinstance(uniforms_F32,np.ndarray) and uniforms_F32.dtype==np.float32 and uniforms_F32.ndim==1
  and len(uniforms_F32)>0 and np.isfinite(uniforms_F32).all()
  and (uniforms_F32>=0).all() and (uniforms_F32<=1).all(),"provided Float32 uniforms in[0,1] required; no clipping")
 if observed_values is not None:
  require(isinstance(observed_values,np.ndarray) and observed_values.dtype in (np.dtype("float32"),np.dtype("float64"))
   and observed_values.shape==uniforms_F32.shape,"observed outputs must match providedU shape and Float32/64 dtype")
 units=_mass_units(probabilities_F32);total=sum(units)
 require(total>0,"positive existing weight sum required")
 prefix=[];running=0
 for mass in units:running+=mass;prefix.append(running)
 borders=[exact(x) for x in borders_F32];first=next(i for i,x in enumerate(units) if x>0)
 last=len(units)-1-next(i for i,x in enumerate(reversed(units)) if x>0)
 mass_sum=Fraction(total,1<<149);points=[]
 for number,provided in enumerate(uniforms_F32):
  u=exact(provided)
  if u==0:j=first
  else:j=bisect_left([v*u.denominator for v in prefix],u.numerator*total)
  require(j<len(units) and units[j]>0,"exact inverse selected no positive bin")
  before=prefix[j]-units[j]
  fraction=Fraction(u.numerator*total-before*u.denominator,u.denominator*units[j])
  require(0<=fraction<=1,"exact inverse interpolation fraction outside[0,1]")
  value=borders[j]+fraction*(borders[j+1]-borders[j]);rounded=float(value)
  require(math.isfinite(rounded) and borders[j]<=exact(rounded)<=borders[j+1],"reference Float64 value outside selected bar")
  cdf_exact=_cdf(value,borders,units,prefix,total);cdf_rounded=_cdf(exact(rounded),borders,units,prefix,total)
  require(cdf_exact==u,"exact normalized inverse CDF residual is nonzero")
  value_error=abs(exact(rounded)-value);round_bound=_quantization_bound(value,rounded)
  require(value_error<=round_bound,"reference Float64 result not within nearest-rounding bound")
  slope=Fraction(units[j],total)/(borders[j+1]-borders[j])
  residual=cdf_rounded-u
  require(abs(residual)<=slope*round_bound,"rounded inverse CDF residual exceeds exact local rounding bound")
  point={"provided_U_Float32":float(provided),"provided_U_hex":float(provided).hex(),"provided_U_exact":ratio(u),
   "inside_production_uniform_contract":bool(0<=u<1),"u1_out_of_contract_stress":bool(u==1),
   "bin":j,"selected_positive_weight":True,"CDF_before_exact":ratio(Fraction(before,total)),
   "CDF_after_exact":ratio(Fraction(prefix[j],total)),"bin_mass_exact_normalized":ratio(Fraction(units[j],total)),
   "interpolation_fraction_exact":ratio(fraction),"fraction_in_closed_unit_interval":True,
   "left_border":float(borders_F32[j]),"right_border":float(borders_F32[j+1]),
   "inverse_exact":ratio(value),"inverse_nearest_Float64":rounded,"inside_outer_borders":True,
   "inverse_rounding_absolute_error_exact":ratio(value_error),"inverse_rounding_bound_exact":ratio(round_bound),
   "exact_inverse_CDF_residual":ratio(cdf_exact-u),"rounded_Float64_CDF_residual_exact":ratio(residual),
   "rounded_Float64_CDF_residual":float(residual),"rounded_CDF_residual_bound_exact":ratio(slope*round_bound)}
  if observed_values is not None:
   observed=float(observed_values[number])
   observation={"unchanged_observed_value":observed if math.isfinite(observed) else str(observed),
    "finite":math.isfinite(observed),"inside_outer_borders":False,"inside_positive_mass_support":False}
   if not math.isnan(observed):
    cdf=Fraction(1 if observed>0 else 0) if math.isinf(observed) else _cdf(exact(observed),borders,units,prefix,total)
    observation.update(normalized_reference_CDF_exact=ratio(cdf),normalized_reference_CDF_residual_exact=ratio(cdf-u),
     normalized_reference_CDF_residual=float(cdf-u))
   if math.isfinite(observed):
    observation.update(inside_outer_borders=bool(borders[0]<=exact(observed)<=borders[-1]),
     inside_positive_mass_support=_support(exact(observed),borders,units),
     absolute_error_from_exact_normalized_inverse_exact=ratio(abs(exact(observed)-value)))
   point["observed_frozen_output"]=observation
  points.append(point)
 return {"schema":SCHEMA,"reference_source_sha256":source_sha256(),
  "semantics":"exactly normalize EXISTING captured Float32 weights by their exact sum; piecewise-uniform finite-bar inverse; no softmax recomputation",
  "not_the_frozen_Float32_cumulative_roundoff_law":True,"no_clipping_or_repair_or_RNG":True,
  "input_weights_Float32_sha256":_digest(probabilities_F32),"input_borders_Float32_sha256":_digest(borders_F32),
  "provided_U_Float32_sha256":_digest(uniforms_F32),"bin_count":len(units),"positive_bin_count":sum(x>0 for x in units),
  "input_weight_sum_exact":ratio(mass_sum),"input_weight_sum_nearest_Float64":float(mass_sum),
  "input_weight_sum_error_from_one_exact":ratio(mass_sum-1),
  "essential_support_endpoints":[float(borders_F32[first]),float(borders_F32[last+1])],
  "outer_borders":[float(borders_F32[0]),float(borders_F32[-1])],
  "endpoint_semantics":"u0:first positive bar left; u1:last positive bar right; u1 is out-of-contract stress, not a production input",
  "historical_batch_or_member_reconstruction_claimed":False,"points":points}
