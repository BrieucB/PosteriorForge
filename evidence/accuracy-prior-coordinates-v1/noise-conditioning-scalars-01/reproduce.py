#!/usr/bin/env python3
"""Saved Policy-C scalar conditioning diagnosis; no samples, RNG or policy change."""
from __future__ import annotations
import argparse, hashlib, json, math, platform, struct, time
from pathlib import Path
import mpmath as mp
import numpy as np
import scipy
from posteriorforge import prior_coordinates_v1 as codec

REPORT_SHA="cbc8e44e62780d48e594845bd63a9c72faa76eaf7a1459d9e4dbb2210e989a5a"
MATH_SHA="d5a22a8a617023ced6d1b043ecca6185bf384a5dcd33bb2ff862050f430028e5"
CODEC_SHA="d8ce6c49e7b65c6ee55c59b7243d4ce943cbea28c92ebed0329798074aaa70c5"
VARIANT="C_softmax64_adjusted_F32logits_F64log_CPUsoftmax64"
OWN_U_BITS="f6ff7f3f"
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(ok,msg):
    if not ok: raise ValueError(msg)
def exact(x):
    n,d=float(x).as_integer_ratio()
    return mp.mpf(n)/d
def txt(x): return mp.nstr(x,85)
def Q(x): return mp.erfc(x/mp.sqrt(2))/2
def bits(x): return struct.unpack(">Q",struct.pack(">d",float(x)))[0]
def number(n): return struct.unpack(">d",struct.pack(">Q",int(n)))[0]
def setup(spec,rounded_geometry=False):
    mu,sd,low,high=[exact(spec[k]) for k in ("location","scale","low","high")]
    a=exact(spec["a"]) if rounded_geometry else (low-mu)/sd
    b=exact(spec["b"]) if rounded_geometry else (high-mu)/sd
    mass=1-Q(-a)-Q(b)
    return mu,sd,low,high,a,b,mass
def inv(z,g):
    mu,sd,low,high,a,b,mass=g
    sf=Q(b)+mass*Q(z)
    return mu+sd*mp.sqrt(2)*mp.erfinv(1-2*sf)
def enc(x,g):
    mu,sd,low,high,a,b,mass=g
    sf=(Q((x-mu)/sd)-Q(b))/mass
    require(0<sf<1,"diagnostic encode must remain strictly interior")
    return mp.sqrt(2)*mp.erfinv(1-2*sf)
def floor64(x):
    f=float(x)
    return math.nextafter(f,-math.inf) if exact(f)>x else f
def ceil64(x):
    f=float(x)
    return math.nextafter(f,math.inf) if exact(f)<x else f
def scalar_frozen(z,spec):
    x=float(codec._noise_decode(np.array([z],dtype=np.float64),spec)[0])
    y=float(codec._noise_encode(np.array([x],dtype=np.float64),spec)[0])
    back=float(codec._noise_decode(np.array([y],dtype=np.float64),spec)[0])
    return x,y,back
def exists_for_tolerance(z,bound,g):
    lo,hi=inv(z-exact(bound),g),inv(z+exact(bound),g)
    first=ceil64(lo);last=floor64(hi)
    count=max(0,bits(last)-bits(first)+1)
    require(count<2,"predeclared narrow interval unexpectedly contains multiple floats")
    return dict(latent_interval=[txt(z-exact(bound)),txt(z+exact(bound))],
        exact_physical_interval=[txt(lo),txt(hi)],physical_interval_width=txt(hi-lo),
        first_F64_at_or_above_lower=first,first_F64_hex=first.hex(),
        last_F64_at_or_below_upper=last,last_F64_hex=last.hex(),
        representable_F64_count=count,any_representable_F64=count>0,
        lower_distance_to_first_F64=txt(exact(first)-lo),
        first_F64_distance_above_upper=txt(exact(first)-hi))
def row_record(r,spec):
    p=next(x for x in r["variants"][VARIANT]["points"] if x["U_bits"]==OWN_U_BITS)
    zf=float(p["value"]);require(7.8<zf<8.3,"fixed stage19 own-U range changed")
    mp.mp.dps=100;g100=setup(spec);x100=inv(exact(zf),g100)
    x100str=mp.nstr(x100,100)
    mp.mp.dps=180;g=setup(spec);z=exact(zf);x=inv(z,g);xr=inv(z,setup(spec,True))
    stability=abs(x-mp.mpf(x100str));eps=mp.mpf("1e-80")
    require(stability<eps,"100/180-digit inverse disagreement exceeds diagnostic envelope")
    # Independent residual/sign check, not certified interval arithmetic.
    require(enc(x-eps,g)<z<enc(x+eps,g),"high-precision inverse numerical bracket failed")
    lo,hi=floor64(x),ceil64(x);nearest=float(x)
    candidates=[]
    for f in sorted(set([lo,hi])):
        ze=enc(exact(f),g)
        frozen=float(codec._noise_encode(np.array([f],dtype=np.float64),spec)[0])
        candidates.append(dict(physical=f,physical_hex=f.hex(),
            exact_prior_latent=txt(ze),exact_prior_latent_abs_error=txt(abs(ze-z)),
            frozen_d8_latent=frozen,frozen_d8_latent_abs_error=abs(frozen-zf)))
    best=min(candidates,key=lambda v:mp.mpf(v["exact_prior_latent_abs_error"]))
    fx,fz,fb=scalar_frozen(zf,spec)
    require(fx==p["d8_noise"]["physical"] and fz==p["d8_noise"]["reencode"],
            "unchanged frozen scalar does not reproduce closed CBC8 record")
    physical_scale=g[1];standard=(x-g[0])/g[1]
    physical64_abs=exact(codec.F64_PHYSICAL_UNITS)*(1+abs(standard))*physical_scale
    physical32_abs=exact(codec.F32_PHYSICAL_UNITS)*(1+abs(standard))*physical_scale
    bound32=float(codec.F32_LATENT_ADMISSIBILITY*(1+abs(zf)))
    bound64=float(codec.F64_LATENT_ADMISSIBILITY*(1+abs(zf)))
    e32=exists_for_tolerance(z,bound32,g);e64=exists_for_tolerance(z,bound64,g)
    pred=math.nextafter(nearest,-math.inf);succ=math.nextafter(nearest,math.inf)
    cell_lo=(exact(pred)+exact(nearest))/2;cell_hi=(exact(nearest)+exact(succ))/2
    ulp=exact(succ)-exact(nearest)
    frozen_error=abs(exact(fx)-x);nearest_error=abs(exact(nearest)-x)
    minerr=mp.mpf(best["exact_prior_latent_abs_error"])
    # Conservative finite grid around every physically admissible output for this scalar.
    # It is diagnostic only, no returned sample/cohort; no values saved as a sample array.
    fmin=max(ceil64(x-physical64_abs),math.nextafter(spec["low"],math.inf))
    fmax=min(floor64(x+physical64_abs),math.nextafter(spec["high"],-math.inf))
    require(bits(fmax)-bits(fmin)<1200,"bounded scalar-cell scan exceeded predeclared limit")
    physical_cells=np.array([number(k) for k in range(bits(fmin),bits(fmax)+1)],dtype=np.float64)
    frozen_enc=codec._noise_encode(physical_cells,spec)
    errors=np.abs(frozen_enc-zf);ix=int(np.argmin(errors))
    return dict(stage=19,arm=r["arm"],member=r["member"],label=r["label"],
        input_z=zf,input_z_hex=zf.hex(),C_point_exact_rational=p["value_exact"],
        input_scope="saved C nearest-F64 finite-bar reference quantile, not a new posterior draw",
        exact_prior_inverse=txt(x),rounded_geometry_prior_inverse=txt(xr),
        source_vs_rounded_geometry_inverse_difference=txt(abs(x-xr)),
        inverse_precision=dict(decimal_dps=[100,180],absolute_disagreement=txt(stability),
            numerical_envelope_absolute="1e-80",envelope_certified=False,
            inverse_reencode_residual=txt(abs(enc(x,g)-z)),
            independent_sign_bracket=[txt(x-eps),txt(x+eps)],
            scope="mpmath high-precision stability and residual/sign checks; NOT a rigorous interval-arithmetic error certificate"),
        nearest_F64=nearest,nearest_F64_hex=nearest.hex(),
        nearest_F64_physical_error=txt(nearest_error),
        nearest_F64_physical_error_in_ULPs=txt(nearest_error/ulp),
        nearest_F64_rounding_cell_physical=[txt(cell_lo),txt(cell_hi)],
        nearest_F64_cell_latent_preimage=[txt(enc(cell_lo,g)),txt(enc(cell_hi,g))],
        bracketing_F64_candidates=candidates,
        minimum_exact_prior_latent_error_among_all_F64=txt(minerr),
        monotonic_existence_argument="Exact prior transport is strictly increasing; the two physical floats bracketing the real inverse minimize latent error. Equivalently count floats inside inverse([z-tolerance,z+tolerance]).",
        latent_tolerance32=bound32,latent_tolerance64=bound64,
        minimum_exact_latent_error_divided_by_RT32_bound=txt(minerr/exact(bound32)),
        minimum_exact_latent_error_divided_by_RT64_bound=txt(minerr/exact(bound64)),
        RT32_physical_interval=e32,RT64_physical_interval=e64,
        frozen_d8=dict(physical=fx,physical_hex=fx.hex(),reencode=fz,
            latent_error=abs(fz-zf),RT32_pass=abs(fz-zf)<=bound32,RT64_pass=abs(fz-zf)<=bound64,
            physical_inverse_error=txt(frozen_error),physical_inverse_error_in_ULPs=txt(frozen_error/ulp),
            ULP_distance_from_correct_nearest_F64=abs(bits(fx)-bits(nearest)),
            physical_inverse_error_divided_by_F64_physical_bound=txt(frozen_error/physical64_abs),
            physical_inverse_error_divided_by_F32_physical_bound=txt(frozen_error/physical32_abs),
            frozen_encode_decode_physical_roundtrip_error=abs(fb-fx)),
        unchanged_physical_tolerance=dict(F64_absolute_at_exact_inverse=txt(physical64_abs),
            F32_absolute_at_exact_inverse=txt(physical32_abs),
            F64_prior_units_coefficient=codec.F64_PHYSICAL_UNITS,
            F32_prior_units_coefficient=codec.F32_PHYSICAL_UNITS,
            physical_unit=float(spec["scale"]),physical_center=float(spec["location"]),
            comparison_scope="The original codec physical-bound formula evaluated at the high-precision scalar inverse is an accuracy diagnostic; it is not a new qualification or replacement for latent admissibility."),
        frozen_encoder_search_within_F64_physical_bound=dict(count=len(physical_cells),
            physical_range=[fmin,fmax],monotone_non_decreasing=bool(np.all(np.diff(frozen_enc)>=0)),
            minimum_latent_error=float(errors[ix]),minimizing_physical=float(physical_cells[ix]),
            RT32_satisfying_count=int(np.sum(errors<=bound32)),
            RT64_satisfying_count=int(np.sum(errors<=bound64)),
            scope="Exhaustive finite scalar-cell check inside the unchanged F64 physical-accuracy interval intersected with the frozen strict codec domain; endpoint/outside cells are not candidates. No global numerical-monotonicity claim outside this interval."),
        diagnostic_only=True)
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report",type=Path,required=True);parser.add_argument("--math-report",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True);a=parser.parse_args()
    require(not a.output.exists(),"fresh output directory required")
    require(sha(a.report)==REPORT_SHA and sha(a.math_report)==MATH_SHA,"closed input report bytes differ")
    require(codec.source_sha256()==CODEC_SHA,"frozen codec bytes differ")
    require("torch" not in __import__("sys").modules,"Torch not used in this CPU scalar diagnosis")
    start=time.perf_counter();d=json.loads(a.report.read_text());mathdoc=json.loads(a.math_report.read_text())
    spec=mathdoc["source_constants_float"]
    rows=[r for r in d["records"] if r["stage"]==19 and r["label"]=="N1024"]
    require(len(rows)==16 and {(r["arm"],r["member"]) for r in rows}==
        {(arm,k) for arm in ("original","candidate") for k in range(8)},"all16 stage19 rows required")
    # State observation only; no random-method invocation.
    np_before=hashlib.sha256(__import__("pickle").dumps(np.random.get_state())).hexdigest()
    py_before=hashlib.sha256(__import__("pickle").dumps(__import__("random").getstate())).hexdigest()
    records=[row_record(r,spec) for r in rows]
    np_after=hashlib.sha256(__import__("pickle").dumps(np.random.get_state())).hexdigest()
    py_after=hashlib.sha256(__import__("pickle").dumps(__import__("random").getstate())).hexdigest()
    require(np_before==np_after and py_before==py_after,"default RNG states changed")
    require(sha(a.report)==REPORT_SHA and sha(a.math_report)==MATH_SHA and codec.source_sha256()==CODEC_SHA,"immutable sources changed")
    result=dict(schema="posteriorforge.prior_noise_conditioning_scalars.v1",status="diagnostic_only",
        source_sha256=sha(__file__),input_report_sha256=REPORT_SHA,input_math_report_sha256=MATH_SHA,
        codec_sha256=CODEC_SHA,source_constants_float=spec,
        source_constants_binary_rational=mathdoc["source_constants_binary_rational"],
        versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,mpmath=mp.__version__),
        input_scope="All16 stage19 N1024 original/candidate saved Policy-C own-U scalars; no density recomputation.",
        mathematical_prior_scope="Exact real truncated normal using binary source mu,sd,low,high. Rounded a=-2,b=2 geometry ideal also crosschecked separately.",
        records=records,wall_seconds=time.perf_counter()-start,
        default_RNG_state_hashes=dict(numpy_before=np_before,numpy_after=np_after,python_before=py_before,python_after=py_after),
        new_GPU_NN_model_constructor_world_posterior_G3_or_RNG_draw_calls=0,
        physical_samples_or_cohort_created=False,production_policy_or_accuracy_gate_changed=False,
        conclusion_scope="Physical inverse accuracy, Float64 representability and latent conditioning are separate. No endpoint acceptance, tolerance waiver, cell substitution or policy selection occurs.")
    a.output.mkdir(parents=True)
    p=a.output/"report.json";p.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
    concise=dict(report_sha256=sha(p),source_sha256=sha(__file__),count=len(records),wall_seconds=result["wall_seconds"],
        no_exact_prior_F64_for_RT32=sum(not r["RT32_physical_interval"]["any_representable_F64"] for r in records),
        no_exact_prior_F64_for_RT64=sum(not r["RT64_physical_interval"]["any_representable_F64"] for r in records),
        no_frozen_encoder_F64_within_physical_bound_RT32=sum(r["frozen_encoder_search_within_F64_physical_bound"]["RT32_satisfying_count"]==0 for r in records),
        no_frozen_encoder_F64_within_physical_bound_RT64=sum(r["frozen_encoder_search_within_F64_physical_bound"]["RT64_satisfying_count"]==0 for r in records),
        frozen_inverse_ULP_distance_range=[min(r["frozen_d8"]["ULP_distance_from_correct_nearest_F64"] for r in records),max(r["frozen_d8"]["ULP_distance_from_correct_nearest_F64"] for r in records)],
        frozen_inverse_physical64_ratio_range=[min(float(r["frozen_d8"]["physical_inverse_error_divided_by_F64_physical_bound"]) for r in records),max(float(r["frozen_d8"]["physical_inverse_error_divided_by_F64_physical_bound"]) for r in records)],
        minimum_achievable_exact_latent_error_range=[min(float(r["minimum_exact_prior_latent_error_among_all_F64"]) for r in records),max(float(r["minimum_exact_prior_latent_error_among_all_F64"]) for r in records)])
    (a.output/"summary.json").write_text(json.dumps(concise,indent=2,sort_keys=True)+"\n")
    print(json.dumps(concise,sort_keys=True))
if __name__=="__main__":main()
