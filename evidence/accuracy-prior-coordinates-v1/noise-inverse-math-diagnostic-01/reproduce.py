#!/usr/bin/env python3
"""Read-only scalar noise inverse diagnosis; Built with PriorLabs-TabPFN.

This script never produces a physical sample array or recovered cohort. It reads
the two predeclared scalar cells from the authenticated failed latent archive.
No model, sampler, RNG draw, G3, reference integration or source edit occurs.
"""
from __future__ import annotations
import argparse,hashlib,json,math,platform
from pathlib import Path
import mpmath as mp
import numpy as np
import scipy
from scipy.special import log_ndtr,ndtri_exp,ndtr
from posteriorforge import prior_coordinates_v1 as codec

MANIFEST_SHA256="92755b788fc894d4ca16157e573bd1b5c9d8a4468b71cf184fa16303bced7bce"
LATENT_SHA256="1ba89c1a4e53c5895ad1623b5a396e4086154c3dc830f8036da388f234d90349"
CODEC_SHA256="d8ce6c49e7b65c6ee55c59b7243d4ce943cbea28c92ebed0329798074aaa70c5"
POINTS=((4355,15,16,13.876729011535645),(41765,19,20,205.2462615966797))
DPS=180
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(value,message):
 if not value:raise ValueError(message)
def exact(value):
 n,d=float(value).as_integer_ratio();return mp.mpf(n)/d
def text(value):return mp.nstr(value,70)
def Q(value):return mp.erfc(value/mp.sqrt(2))/2
def phi(value):return mp.exp(-value*value/2)/mp.sqrt(2*mp.pi)
def factor(t,b):
 # Rescaled integral avoids the unresolvable endpoint subtraction b-t.
 return mp.quad(lambda v:mp.exp(b*t*v-t*t*v*v/2),[0,1])

def gap_from_latent(z,spec):
 mu,sd,low,high=(exact(spec[k]) for k in ("location","scale","low","high"))
 a=(low-mu)/sd;b=(high-mu)/sd;mass=1-Q(-a)-Q(b)
 rhs=mass*Q(z)/phi(b)
 t=rhs
 for _ in range(4):t-=(t*factor(t,b)-rhs)/mp.exp(b*t-t*t/2)
 # t solves int_0^t exp(bu-u^2/2) du = M*Q(z)/phi(b).
 # Gap, not high-gap, is represented at arbitrary precision.
 gap=sd*t
 # Mills: Q(z)<phi(z)/z. For our tiny t<1 and b=2, integrand>=1;
 # hence gap < sd*exp((b^2-z^2)/2)/z, independently of quadrature.
 upper_log=mp.log(sd)+(b*b-z*z)/2-mp.log(z)
 require(rhs<1 and b==2,"predeclared bound proof assumes b2 and root t<1")
 require(gap<mp.exp(upper_log),"independent Mills/integral upper bound failed")
 return gap,{"exact_binary_a":text(a),"exact_binary_b":text(b),"normalization_mass":text(mass),
  "rescaled_t":text(t),"rhs":text(rhs),"normal_survival":text(Q(z)),
  "log10_normal_survival":text(mp.log10(Q(z))),"analytic_upper_gap":text(mp.exp(upper_log)),
  "log_analytic_upper_gap":text(upper_log),
  "linearized_gap_relative_error_upper_log10":text(mp.log10(b*rhs)),
  "linearized_bound_scope":"t is between rhs*exp(-b*rhs) and rhs; log10(b*rhs) bounds 1-exp(-b*rhs) from above"}

def reproduce(parent):
 require(sha(parent/"manifest.json")==MANIFEST_SHA256,"failed manifest bytes differ")
 require(sha(parent/"latent-proposal-waves.npz")==LATENT_SHA256,"failed raw latent archive differs")
 require(codec.source_sha256()==CODEC_SHA256,"frozen codec differs")
 with np.load(parent/"latent-proposal-waves.npz",allow_pickle=False) as saved:
  require(saved.files==["wave_0000"],"one original proposal wave required");latent=saved["wave_0000"]
 require(latent.dtype==np.float32 and latent.shape==(51200,21),"actual raw Float32 wave shape differs")
 mp.mp.dps=DPS
 spec=codec._geometry()["noise"][16]
 require(spec==codec._geometry()["noise"][20],"two target priors differ")
 mu,sd,low,high=(exact(spec[k]) for k in ("location","scale","low","high"))
 predecessor=float(np.nextafter(spec["high"],spec["low"]))
 spacing=high-exact(predecessor);half=spacing/2;b=(high-mu)/sd;a=(low-mu)/sd;mass=1-Q(-a)-Q(b)
 qpred=phi(b)*(spacing/sd)*factor(spacing/sd,b)/mass
 qmid=phi(b)*(half/sd)*factor(half/sd,b)/mass
 invert=lambda q:mp.findroot(lambda z:mp.log(Q(z))-mp.log(q),(mp.mpf(8),mp.mpf(9)))
 zpred,zmid=invert(qpred),invert(qmid)
 actual_pred_z=float(codec._noise_encode(np.array([predecessor],np.float64),spec)[0])
 base=float(log_ndtr(-spec["b"]));base_root=mp.findroot(lambda x:mp.log(Q(-x))-exact(base),(-mp.mpf("2.01"),-mp.mpf("1.99")))
 rows=[]
 for row,stage,slot,expected in POINTS:
  z=float(latent[row,stage]);require(z==expected and codec.ORDER[stage]==slot,"predeclared actual scalar differs")
  zmp=exact(z);gap,details=gap_from_latent(zmp,spec)
  add=float(log_ndtr(-z)+spec["log_mass"]);summed=float(np.logaddexp(base,add))
  inverse=float(ndtri_exp(summed));physical=float(spec["location"]-spec["scale"]*inverse)
  lost=mp.log1p(mp.exp(exact(add)-exact(base)))
  require(gap<half and mp.mpf(details["analytic_upper_gap"])<half,"correct-rounding conclusion not proven")
  min_bits=int(mp.floor(-mp.log(gap,2)-2)+1)
  rows.append({"row_zero_based":row,"ordered_stage":stage,"physical_slot":slot,"physical_name":codec.PHYSICAL_ORDER[slot],
   "inside_first50000":row<50000,"actual_finite_latent":z,"actual_latent_hex":z.hex(),
   "exact_upper_gap":text(gap),"log10_exact_upper_gap":text(mp.log10(gap)),"method":details,
   "gap_to_half_predecessor_gap":text(gap/half),
   "correct_rounding_to_nearest_Float64":"0.5 endpoint; no correctly rounded STRICT interior value for THIS inverse",
   "other_interior_Float64_values_exist":True,"minimum_local_significand_bits_for_correct_rounding_to_interior":min_bits,
   "precision_bits_scope":"only enough to separate this real quantile from endpoint after rounding; not sufficient by itself for all roundtrip bounds",
   "exact_prior_roundtrip_error_if_interior_predecessor_substituted":text(zmp-zpred),
   "frozen_roundtrip_error_if_interior_predecessor_substituted":abs(z-actual_pred_z),
   "current_Float32_latent_roundtrip_bound":float(codec.F32_LATENT_ADMISSIBILITY*(1+abs(z))),
   "frozen_scalar_intermediates":{"log_survival_z":float(log_ndtr(-z)),"log_mass_plus_log_survival":add,
    "logaddexp":summed,"base_log_tail":base,"logaddexp_equals_base_bits":summed.hex()==base.hex(),
    "lost_logaddexp_increment_from_actual_rounded_inputs":text(lost),"base_log_ulp":abs(float(np.spacing(base))),
    "ndtri_exp":inverse,"ndtri_exp_hex":inverse.hex(),"physical_scalar":physical,"physical_scalar_hex":physical.hex(),
    "strict_noise_predicate":spec["low"]<physical<spec["high"]},
   "naive_diagnostics_NOT_used_by_frozen_codec":{"Phi_z":float(ndtr(z)),"exp_log_survival":float(np.exp(log_ndtr(-z)))}})
 require(sha(parent/"manifest.json")==MANIFEST_SHA256 and sha(parent/"latent-proposal-waves.npz")==LATENT_SHA256,"inputs changed")
 return {"schema":"posteriorforge.scalar_noise_inverse_math_reproduction.v1","status":"diagnostic_only",
  "attribution":"Built with PriorLabs-TabPFN","versions":{"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,"mpmath":mp.__version__},
  "working_decimal_precision":DPS,"manifest_sha256":MANIFEST_SHA256,"latent_archive_sha256":LATENT_SHA256,"codec_source_sha256":CODEC_SHA256,
  "source_constants_float":dict(spec),"source_constants_hex":{k:float(v).hex() for k,v in spec.items()},
  "source_constants_binary_rational":{k:list(float(spec[k]).as_integer_ratio()) for k in ("location","scale","low","high")},
  "source_geometry_rounded_a_b":[spec["a"],spec["b"]],"ideal_binary_source_a":text(a),"ideal_binary_source_b":text(b),
  "upper_predecessor":predecessor,"upper_predecessor_hex":predecessor.hex(),"upper_predecessor_gap":text(spacing),
  "rounding_midpoint_gap":text(half),"exact_prior_z_at_predecessor":text(zpred),"frozen_z_at_predecessor":actual_pred_z,
  "exact_prior_z_above_which_nearest_Float64_is_endpoint":text(zmid),
  "rounded_log_tail_input":base,"ideal_ndtri_of_rounded_log_tail_input":text(base_root),
  "correctly_rounded_ndtri_of_rounded_log_tail_input":float(base_root),
  "actual_frozen_ndtri_of_rounded_log_tail_input":float(ndtri_exp(base)),
  "rows":rows,
  "method_and_error_scope":{"equation":"sigma=U-s*t; integral_0^t exp(bu-u^2/2)du=M*Q(z)/phi(b)",
   "gap_evaluated_without_subtracting_from_boundary":True,"ultra_small_gap":"stored as arbitrary-precision distance/log distance, not as .5 minus an invisible Float64 delta",
   "180_digits_is_not_certified_interval_arithmetic":True,
   "independent_bound":"Q(z)<phi(z)/z and integral>=t imply gap < s*exp((b^2-z^2)/2)/z; this bound is also far below half a Float64 predecessor gap",
   "source_prior_vs_cached_geometry":"binary physical constants imply a slightly below -2; stored geometry a=-2. Original raw report also preserves both masses/gaps. Upper b is exactly2 for both; conclusion unchanged",
   "latent_tail_cause_not_identified":"no actual logits/u/bin/interpolation inputs used or reconstructed"},
  "new_model_sampler_RNG_G3_or_reference_calls":0,"repaired_physical_values_or_cohort_saved":False,"source_or_policy_modified":False}

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument("--parent",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
 args=parser.parse_args();require(not args.output.exists(),"fresh scalar report path required")
 report=reproduce(args.parent)
 with args.output.open("x") as stream:json.dump(report,stream,indent=2,sort_keys=True,allow_nan=False);stream.write("\n")
 print(json.dumps({"output":str(args.output),"sha256":sha(args.output),"new_draws_or_cohorts":0},sort_keys=True))
if __name__=="__main__":main()
