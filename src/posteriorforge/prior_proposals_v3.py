"""V3 finite physical proposals; strict training transport remains frozen d8.

No clipping, replacement, row deletion or acceptance occurs here. Validation
probe edges may be directed inward; decoded proposal bytes are never adjusted.
"""
from __future__ import annotations
from fractions import Fraction
import hashlib
import math
from pathlib import Path
from types import FunctionType
import numpy as np
from scipy.special import log_ndtr
from scipy.stats import truncnorm
from . import prior_coordinates_v1 as training

SCHEMA="posteriorforge.prior_proposals.v3"
CODEC_SHA256="d8ce6c49e7b65c6ee55c59b7243d4ce943cbea28c92ebed0329798074aaa70c5"
SUPPORT_SHA256="95366efc4871b96c3a1a3f19414ed1fd77d058f02135f4afd95a64b5918b3d3e"
ORDER=training.ORDER
INVERSE_ORDER=training.INVERSE_ORDER
CoordinateError=training.CoordinateError
training_encode_for_model=training.encode_for_model
training_encode_physical=training.encode_physical
_ORIGINALS={name:getattr(training,name) for name in (
    "_noise_decode","_decode_prefix","_encode_prefix","_rows","_geometry",
    "_physical_check","_latent_check","_local_mean","encode_for_model","encode_physical",
    "require_sources","source_sha256")}
_CODES={k:v.__code__ for k,v in _ORIGINALS.items() if hasattr(v,"__code__")}
def _require(ok,message):
    if not ok: raise CoordinateError(message)
def source_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
def _finite_noise_output(values,spec,label):
    _require(np.isfinite(values).all(),label+": nonfinite physical proposal")
def _noise_not_reencoded(values,spec):
    # Diagnostic affine replay only. Local Cholesky laws do not depend on sigma.
    return np.zeros_like(values,dtype=np.float64)
_NOISE_NS=dict(_ORIGINALS["_noise_decode"].__globals__)
_NOISE_NS["_strict_noise"]=_finite_noise_output
_NOISE=FunctionType(_CODES["_noise_decode"],_NOISE_NS,"_finite_noise_inverse")
_PREFIX_NS=dict(_ORIGINALS["_decode_prefix"].__globals__)
_PREFIX_NS["_noise_decode"]=_NOISE
_PREFIX=FunctionType(_CODES["_decode_prefix"],_PREFIX_NS,"_proposal_prefix")
_AFFINE_NS=dict(_ORIGINALS["_encode_prefix"].__globals__)
_AFFINE_NS["_noise_encode"]=_noise_not_reencoded
_AFFINE=FunctionType(_CODES["_encode_prefix"],_AFFINE_NS,"_affine_replay")
_PRIVATE=((_NOISE,_CODES["_noise_decode"],_NOISE_NS),
          (_PREFIX,_CODES["_decode_prefix"],_PREFIX_NS),
          (_AFFINE,_CODES["_encode_prefix"],_AFFINE_NS))
_NAMESPACE_SEALS=tuple((ns,dict(ns)) for _,_,ns in _PRIVATE)
def require_sources():
    _require(hashlib.sha256(Path(training.__file__).read_bytes()).hexdigest()==CODEC_SHA256,
             "frozen training codec source changed")
    support=Path(__file__).with_name("inference.py")
    _require(hashlib.sha256(support.read_bytes()).hexdigest()==SUPPORT_SHA256,
             "original authoritative support source changed")
    for name,fn in _ORIGINALS.items():
        _require(getattr(training,name) is fn,"frozen codec alias changed: "+name)
        if name in _CODES: _require(fn.__code__ is _CODES[name],"frozen codec body changed: "+name)
    _ORIGINALS["require_sources"]()
    for name,fn,code in _PUBLIC_SEALS:
        _require(globals().get(name) is fn and fn.__code__ is code,"proposal decoder callable changed: "+name)
    for fn,code,ns in _PRIVATE:
        _require(fn.__code__ is code and fn.__globals__ is ns,"private decoder body changed")
    for ns,sealed in _NAMESPACE_SEALS:
        _require(ns.keys()==sealed.keys() and all(ns[k] is v for k,v in sealed.items()),
                 "private decoder namespace changed")
    _require(training_encode_for_model is _ORIGINALS["encode_for_model"]
             and training_encode_physical is _ORIGINALS["encode_physical"],
             "strict training aliases changed")
    return dict(training_codec_sha256=CODEC_SHA256,original_support_sha256=SUPPORT_SHA256,
                original_namespace_mutated=False,private_decode_codeobjects_identical=True)
def _fraction(value): return Fraction.from_float(float(value))
def _inward_float(value,lower):
    result=float(value)
    exact=_fraction(result)
    if (lower and exact<value) or (not lower and exact>value):
        # A VALIDATION EDGE only, never a proposal/output adjustment.
        result=math.nextafter(result,math.inf if lower else -math.inf)
    return result
def _physical_error_interval(y,spec):
    """Exact original physical inequality solved for ideal x, with inward probes.

    |y-x|/s <= c*(1+|(x-mu)/s|), c=original256eps64. The two
    Fraction roots use the expected ideal x in the bound, not observed y.
    """
    mu,s,c=_fraction(spec["location"]),_fraction(spec["scale"]),_fraction(training.F64_PHYSICAL_UNITS)
    result=np.empty((len(y),2),dtype=np.float64)
    for i,value in enumerate(y):
        v=(_fraction(value)-mu)/s
        minus,plus=v-c,v+c
        sign_minus=1 if minus>=0 else -1
        sign_plus=1 if plus>=0 else -1
        lo=mu+s*minus/(1+c*sign_minus)
        hi=mu+s*plus/(1-c*sign_plus)
        result[i,0]=_inward_float(lo,True);result[i,1]=_inward_float(hi,False)
        _require(result[i,0]<=result[i,1],"empty physical precision validation interval")
    return result
def _prior_logprob(x,spec,lower):
    # CDF extension beyond ORIGINAL closed support, for validation probes only.
    result=(truncnorm.logcdf if lower else truncnorm.logsf)(
        x,spec["a"],spec["b"],loc=spec["location"],scale=spec["scale"])
    result=np.asarray(result,dtype=np.float64)
    result[x<=spec["low"]]=-np.inf if lower else 0.
    result[x>=spec["high"]]=0. if lower else -np.inf
    return result
def _validate_precision(latent,physical,geo,input_dtype):
    stage_indices=tuple(ORDER[:latent.shape[1]])
    affine=[j for j,p in enumerate(stage_indices) if p not in geo["noise"]]
    replay=_AFFINE(physical,geo)
    coefficient=(training.F32_LATENT_ADMISSIBILITY if input_dtype==np.dtype("float32")
                 else training.F64_LATENT_ADMISSIBILITY)
    if affine:
        _ORIGINALS["_latent_check"](replay[:,affine],latent[:,affine],coefficient)
        back=_PREFIX(replay,geo)
        _ORIGINALS["_physical_check"](back[:,affine],physical[:,affine],
            [stage_indices[j] for j in affine],training.F64_PHYSICAL_UNITS,geo)
    noise=[]
    for j,p in enumerate(stage_indices):
        if p not in geo["noise"]: continue
        spec=geo["noise"][p];x=physical[:,j];z=latent[:,j]
        edges=_physical_error_interval(x,spec)
        left=z<=0
        target=log_ndtr(np.where(left,z,-z))
        # For lower tail CDF increases; for upper tail SF decreases.
        low=np.empty(len(z));high=np.empty(len(z))
        low[left]=_prior_logprob(edges[left,0],spec,True)
        high[left]=_prior_logprob(edges[left,1],spec,True)
        low[~left]=_prior_logprob(edges[~left,1],spec,False)
        high[~left]=_prior_logprob(edges[~left,0],spec,False)
        passed=(target>=low)&(target<=high)
        _require(np.isfinite(target).all() and passed.all(),
                 "noise inverse exceeds unchanged physical-error probability interval: physical"+str(p))
        noise.append(dict(stage=j,physical_slot=p,count=len(z),hard_passed=True,
            lower_endpoint_count=int(np.sum(x==spec["low"])),
            upper_endpoint_count=int(np.sum(x==spec["high"])),
            below_original_support_count=int(np.sum(x<spec["low"])),
            above_original_support_count=int(np.sum(x>spec["high"])),
            physical_validation_interval_width_min=float(np.min(edges[:,1]-edges[:,0])),
            physical_validation_interval_width_max=float(np.max(edges[:,1]-edges[:,0])),
            universal_absolute_latent_roundtrip_required=False))
    return dict(hard_passed=True,count=len(latent),available_prefix_width=latent.shape[1],
        affine_original_latent_and_physical_checks_retained=True,noise=noise,
        validation="input Normal lower/upper log-tail probability lies in prior-CDF image of unchanged physical-error interval",
        probability_tolerance_added=0.,physical_prior_unit_coefficient=training.F64_PHYSICAL_UNITS,
        source_identity=require_sources())
def _rows(values,widths,label):
    return _ORIGINALS["_rows"](values,widths,(np.float64,np.float32),label)
def decode_ordered_prefix_proposals(values):
    """All available causal columns, finite F64 stage-order proposals; no acceptance."""
    require_sources()
    rows=_rows(values,tuple(range(1,22)),"decode_ordered_prefix_proposals")
    latent=np.ascontiguousarray(rows,dtype=np.float64)
    geo=_ORIGINALS["_geometry"]()
    physical=_PREFIX(latent,geo)
    _validate_precision(latent,physical,geo,rows.dtype)
    return physical
def decode_ordered_proposals(values):
    """Every complete latent row -> every physical F64 row, BEFORE original FIFO/support."""
    _rows(values,(21,),"decode_ordered_proposals")
    return np.ascontiguousarray(decode_ordered_prefix_proposals(values)[:,INVERSE_ORDER])
def precision_report(values,physical=None):
    """Inspect scalar/engineering proposals; returns failure evidence, never filters rows.

    Optional physical is F64 ORIGINAL physical order for a21-column input;
    for a shorter prefix it must be stage order. No sample acceptance is done.
    """
    require_sources()
    try:
        rows=_rows(values,tuple(range(1,22)),"precision_report")
        latent=np.ascontiguousarray(rows,dtype=np.float64);geo=_ORIGINALS["_geometry"]()
        if physical is None: staged=_PREFIX(latent,geo)
        else:
            _require(isinstance(physical,np.ndarray) and physical.dtype==np.float64
                and physical.shape==rows.shape and np.isfinite(physical).all(),
                "precision_report requires matching finite physicalF64")
            staged=physical[:,ORDER] if rows.shape[1]==21 else physical
        result=_validate_precision(latent,staged,geo,rows.dtype)
        result["acceptance_applied"]=False
        result["original_closed_support_finite_proposals_only"]={
            "full_vector_scope":rows.shape[1]==21,
            "noise_outside_rows":int(np.sum(np.any(np.column_stack([
                (staged[:,j]<geo["noise"][p]["low"])|(staged[:,j]>geo["noise"][p]["high"])
                for j,p in enumerate(ORDER[:rows.shape[1]]) if p in geo["noise"]]),axis=1)))
                if any(p in geo["noise"] for p in ORDER[:rows.shape[1]]) else 0,
            "authority":"original inference._prior.Support.check; this is a diagnostic count, not acceptance"}
        return result
    except (CoordinateError,FloatingPointError,OverflowError) as error:
        return dict(hard_passed=False,error_type=type(error).__name__,reason=str(error),
                    acceptance_applied=False,rows_filtered=0,source_identity=require_sources())
proposal_decode_report=precision_report
def metadata():
    identity=require_sources()
    return dict(schema=SCHEMA,module_sha256=source_sha256(),source_identity=identity,
        ordered_latent_to_physical=list(ORDER),input_dtypes=["float32","float64"],output_dtype="float64",
        all_rows_retained_before_original_F64_FIFO_and_support=True,
        proposal_endpoint_policy="retain computed finite endpoints; retain finite slight-outside proposals for original whole-row rejection",
        nonfinite_input_or_arithmetic="hard fail whole request; no finite endpoint substitution",
        inverse_math="exact frozen d8 noise and causal decode codeobjects in NEW private namespaces; only strict noise-output domain validator replaced by finiteness",
        training_encoding="EXACT original d8 strict encode_physical/encode_for_model aliases; unchanged finalF32 cast",
        physical_error_validation=dict(coefficient=training.F64_PHYSICAL_UNITS,
            formula="|computed-ideal|/unit <= coefficient*(1+abs((ideal-center)/unit))",
            noise_method="exact rational interval roots, inward F64 VALIDATION edges, original prior lower/upper log-tail CDF image",
            probability_tolerance_added=0.,directed_edges_never_modify_proposals=True),
        affine_original_checks_retained=True,
        universal_noise_absolute_latent_roundtrip_replaced=True,
        noise_latent_to_physical_not_lossless_near_bounds=True,
        original_closed_support_sha256=SUPPORT_SHA256,acceptance_performed_here=False,
        clipping_nextafter_replacement_or_redraw_of_proposals=False,
        original_CDF_moment_SD_gate_changed=False,new_numerical_qualification_inherited=False)
contract_metadata=metadata

_PUBLIC_SEALS=tuple((name,globals()[name],globals()[name].__code__) for name in (
    "_require","source_sha256","_finite_noise_output","_noise_not_reencoded",
    "require_sources","_fraction","_inward_float","_physical_error_interval","_prior_logprob",
    "_validate_precision","_rows","decode_ordered_prefix_proposals","decode_ordered_proposals",
    "precision_report","proposal_decode_report","metadata","contract_metadata"))
