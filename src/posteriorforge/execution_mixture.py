"""Qualified latent-member evaluation and finite-bar inverse CDF primitives.

Built with PriorLabs-TabPFN. All eight cached members remain prepared. A caller
must propose a member uniformly and independently for each row at each stage,
then accept that head with its raw canonical mass divided by the envelope;
one member for a whole vector is a different joint distribution. Component
outputs are not full-mixture log densities. No RNG is drawn in this module.
"""
from __future__ import annotations
import hashlib
from importlib import import_module
from pathlib import Path
from threading import Lock

from . import execution_cache as cache

ENGINE_ID="resident_member_mixture_fp32_v1"
ENGINE_VERSION=1
SCHEMA="posteriorforge.execution.mixture.v1"
PINNED_CACHE_SHA256="45cd3107c50dcf73d03167e2b6dd045cccdb798337e8d3c47b5210cc30dc1a03"
COMPONENT_MASS_TOLERANCE=1e-5
HEAD_MASS_ENVELOPE=1+COMPONENT_MASS_TOLERANCE
_MEMBER_FIELDS=("preprocessors","models","configs","cat_ixs","n_train_samples")
_MISSING=object()


def execution_source_sha256():return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def execution_metadata():
    return {"schema":SCHEMA,"engine_id":ENGINE_ID,"engine_version":ENGINE_VERSION,
        "module_sha256":execution_source_sha256(),"resident_cache_module_sha256":PINNED_CACHE_SHA256,
        "prepared_ensemble_members":8,"evaluated_members_per_row":1,
        "member_selection":"caller-owned uniform independent head proposal per row per autoregressive stage, accepted with rawmass/envelope",
        "canonical_component_path":"unchanged TabPFN.predict: member temperature, inverse target borders, corrected canonical translation, original physical criterion",
        "component_mass_tolerance":COMPONENT_MASS_TOLERANCE,"head_mass_envelope":HEAD_MASS_ENVELOPE,
        "component_mass_guard":"0 < FP64sum(rawFP32canonicalprobabilities) <= head_mass_envelope",
        "prediction_math_dtype":"torch.float32","component_mass_accumulation_dtype":"torch.float64",
        "head_rejection_corrects_component_normalization":True,
        "roundoff_scope":"raw-mass head rejection reproduces normalize(sum(rawP)) in real arithmetic; residual FP32 log/softmax output roundoff is qualified separately",
        "uncorrected_uniform_head_tv_bound_if_all8_masses_close_to_one":COMPONENT_MASS_TOLERANCE/(1-COMPONENT_MASS_TOLERANCE),
        "bound_limitation":"Selected-member runtime guards do not observe unselected members at the same row and do not establish a global joint bound.",
        "with_log_prob_supported":False,"tail_family_changed":False,
        "attribution":"Built with PriorLabs-TabPFN"}


def _require_cache_source():
    if cache.execution_source_sha256()!=PINNED_CACHE_SHA256:
        raise cache.CacheExecutionError("latent-member engine requires the qualified frozen resident-cache source")


def _regressor_module():return import_module("tabpfn.regressor")


def _component_mass_cpu(output):
    torch=import_module("torch")
    return output.sum(dim=-1,dtype=torch.float64).detach().cpu()


def _observe_component_prediction(base,query,*,member_index,capture_member,report):
    """Install capture only after base.predict's active-CDF guard has run."""
    regressor=base.regressor;module=_regressor_module()
    active=cache._require_active_cdf(module,base.cdf_descriptor)
    original_predict=regressor.predict
    previous_instance_predict=vars(regressor).get("predict",_MISSING)
    captures=[];mass_reports=[];raw_masses=[]

    def predict_proxy(*args,**kwargs):
        def record(logits,*,frm,to):
            output=active(logits,frm=frm,to=to)
            mass_tensor=_component_mass_cpu(output)
            mass=mass_tensor.numpy()
            raw_masses.append(mass_tensor)
            import numpy as np
            if not np.isfinite(mass).all() or np.any(mass<=0):
                raise cache.CacheExecutionError("undefined component probability mass; no mixture-law claim")
            deviation=float(np.max(np.abs(mass.astype(np.float64)-1)))
            if bool((mass>HEAD_MASS_ENVELOPE).any()):
                raise cache.CacheExecutionError("component canonical mass exceeds qualified head envelope")
            mass_reports.append({"rows":len(mass),"min":float(mass.min()),"max":float(mass.max()),
                                 "max_abs_deviation_from_one":deviation})
            if capture_member:
                captures.append({"ensemble_index":member_index,
                    "raw_logits":logits.detach().cpu().clone(),
                    "from_borders":frm.detach().cpu().clone(),"to_borders":to.detach().cpu().clone(),
                    "translated_probabilities":output.detach().cpu().clone(),
                    "component_probability_mass":mass.copy(),
                    "configuration_identity":cache._configuration_identity(regressor.executor_.configs[0])})
            return output
        module.translate_probs_across_borders=record
        try:
            prediction=original_predict(*args,**kwargs)
        finally:
            conflict=module.translate_probs_across_borders is not record
            module.translate_probs_across_borders=active
            if conflict:raise cache.CacheExecutionError("conflicting component translator capture")
        return prediction

    regressor.predict=predict_proxy
    try:
        prediction=base.predict(query,output_type="full",capture_members=False)
    finally:
        conflict=vars(regressor).get("predict") is not predict_proxy
        if previous_instance_predict is _MISSING:vars(regressor).pop("predict",None)
        else:regressor.predict=previous_instance_predict
        if conflict:raise cache.CacheExecutionError("conflicting component predict proxy")
    if len(mass_reports)!=1 or (capture_member and len(captures)!=1):
        raise cache.CacheExecutionError("exactly one original component translation required")
    observed=mass_reports[0]
    report["mass_observed_rows"]+=observed["rows"]
    report["maximum_observed_mass_deviation"]=max(report["maximum_observed_mass_deviation"],observed["max_abs_deviation_from_one"])
    report["minimum_observed_mass"]=observed["min"] if report["minimum_observed_mass"] is None else min(report["minimum_observed_mass"],observed["min"])
    report["maximum_observed_mass"]=observed["max"] if report["maximum_observed_mass"] is None else max(report["maximum_observed_mass"],observed["max"])
    prediction["component_raw_mass"]=raw_masses[0]
    prediction["head_mass_envelope"]=HEAD_MASS_ENVELOPE
    if capture_member:prediction["member"]=captures[0]
    return prediction


class ComponentMixtureExecutor:
    """Evaluate one assigned component before model computation, restoring lists.

    This object shares its base executor and requires exclusive sequential use.
    It does not generate assignments or uniforms and does not reject proposals.
    """
    def __init__(self,base_executor):
        _require_cache_source()
        if not isinstance(base_executor,cache.ResidentStageExecutor):
            raise TypeError("explicit ResidentStageExecutor required")
        self.base=base_executor;self._busy=Lock()
        self.metadata={**execution_metadata(),"component_predict_count":[0]*8,
             "component_query_rows":[0]*8,"mass_observed_rows":0,
             "maximum_observed_mass_deviation":0.,"minimum_observed_mass":None,
             "maximum_observed_mass":None,"full8_member_lists_restored":True}

    @property
    def regressor(self):return self.base.regressor
    @property
    def stage_index(self):return self.base.stage_index

    def predict_component(self,member_index,query,*,capture_member=False,with_log_prob=False):
        if with_log_prob:raise NotImplementedError("component output is not full-mixture log probability")
        if type(member_index) is not int or not 0<=member_index<8:raise ValueError("member index0..7 required")
        if self.regressor is None:raise cache.CacheExecutionError("prepare the base stage first")
        if not self._busy.acquire(blocking=False):raise cache.CacheExecutionError("concurrent component execution is unsupported")
        executor=self.regressor.executor_;original={}
        try:
            for name in _MEMBER_FIELDS:
                value=getattr(executor,name)
                if len(value)!=8:raise cache.CacheExecutionError("all8prepared member lists required before filtering")
                original[name]=value
            self.metadata["full8_member_lists_restored"]=False
            for name,value in original.items():setattr(executor,name,[value[member_index]])
            prediction=_observe_component_prediction(self.base,query,member_index=member_index,
                capture_member=capture_member,report=self.metadata)
            self.metadata["component_predict_count"][member_index]+=1
            self.metadata["component_query_rows"][member_index]+=len(query)
            return prediction
        finally:
            for name,value in original.items():setattr(executor,name,value)
            self.metadata["full8_member_lists_restored"]=(len(original)==len(_MEMBER_FIELDS) and all(getattr(executor,name) is value and len(value)==8 for name,value in original.items()))
            self._busy.release()

    def inspect_cache(self):return self.base.inspect_cache()
    def log_prob(self,*args,**kwargs):
        raise NotImplementedError("evaluate all8members for the full-mixture density")


def mixture_roundoff_report(component_translated_probabilities):
    """Fixed-row finite-family discrepancy and weight-TV upper bound, CPU only.

    Inputs are raw canonical member probabilities [member,row,bin]. Calculations
    use float64 on captured values; actual FP32 API-output parity is a separate
    qualification check, not silently included in this analytic report.
    """
    import numpy as np
    p=np.asarray(component_translated_probabilities,dtype=np.float64)
    if p.ndim!=3 or not p.shape[0] or not np.isfinite(p).all() or np.any(p<0):
        raise ValueError("finite nonnegative member,row,bin probabilities required")
    mass=p.sum(-1)
    if np.any(mass<=0):raise ValueError("positive component row masses required")
    full=p.mean(0);full/=full.sum(-1,keepdims=True)
    latent=(p/mass[...,None]).mean(0)
    weights=mass/mass.sum(0,keepdims=True)
    bound=.5*np.abs(weights-1/len(p)).sum(0)
    actual=.5*np.abs(full-latent).sum(-1)
    cdf=np.abs(np.cumsum(full-latent,axis=-1)).max(-1)
    if np.any(actual>bound+1e-12):raise RuntimeError("normalization weight-TV bound failed")
    weighted=((mass/mass.sum(0,keepdims=True))[...,None]*(p/mass[...,None])).sum(0)
    weighted_tv=.5*np.abs(full-weighted).sum(-1)
    weighted_cdf=np.abs(np.cumsum(full-weighted,axis=-1)).max(-1)
    return {"member_count":len(p),"rows":p.shape[1],"component_mass_min":float(mass.min()),
        "component_mass_max":float(mass.max()),"component_mass_max_abs_deviation":float(np.abs(mass-1).max()),
        "component_mass_guard_passed":bool(np.all(mass<=HEAD_MASS_ENVELOPE)),
        "component_mass_closeness_to_one_passed":bool(np.all(np.abs(mass-1)<=COMPONENT_MASS_TOLERANCE)),
        "row_mass_corrected_weighted_tv":weighted_tv.tolist(),
        "row_mass_corrected_weighted_cdf_distance":weighted_cdf.tolist(),
        "maximum_mass_corrected_weighted_tv":float(weighted_tv.max()),
        "maximum_mass_corrected_weighted_cdf_distance":float(weighted_cdf.max()),
        "mass_corrected_reconstruction_passed":bool(np.all(weighted_tv<=1e-12)),
        "row_weight_tv_bound":bound.tolist(),"row_actual_normalization_tv":actual.tolist(),
        "row_actual_normalization_cdf_distance":cdf.tolist(),
        "maximum_weight_tv_bound":float(bound.max()),"maximum_actual_normalization_tv":float(actual.max()),
        "maximum_actual_normalization_cdf_distance":float(cdf.max()),
        "bound_scope":"fixed captured rows, allmembermasses observed; no global or accepted-joint bound"}


def head_acceptance(component_raw_mass,uniforms):
    """RNG-free raw-mass head rejection; caller retries rejected uniform heads."""
    import numpy as np
    if hasattr(component_raw_mass,"detach"):component_raw_mass=component_raw_mass.detach().cpu().numpy()
    if hasattr(uniforms,"detach"):uniforms=uniforms.detach().cpu().numpy()
    mass=np.asarray(component_raw_mass,dtype=np.float64)
    u=np.asarray(uniforms,dtype=np.float64)
    if mass.ndim!=1 or u.shape!=mass.shape or not len(mass):
        raise ValueError("paired nonempty head mass and uniform vectors required")
    if not np.isfinite(mass).all() or np.any(mass<=0) or np.any(mass>HEAD_MASS_ENVELOPE):
        raise ValueError("head mass outside declared positive envelope")
    if not np.isfinite(u).all() or np.any(u<0) or np.any(u>=1):
        raise ValueError("head acceptance uniforms must be in[0,1)")
    return u < mass/HEAD_MASS_ENVELOPE


def finite_bar_icdf(logits,borders,uniforms):
    """Vectorized inherited finite-uniform-bar ICDF with caller-owned FP32 u.

    No clipping or repair, including u=0 with leading zero probability: the
    original 0/0 NaN is retained for complete-vector support rejection. A single
    shared distribution uses one CDF and never expands an N-by-bin matrix.
    """
    torch=import_module("torch")
    if logits.ndim!=2 or borders.ndim!=1 or uniforms.ndim!=1 or logits.shape[-1]+1!=len(borders):
        raise ValueError("logits[row,bin], borders[bin+1], uniforms[row] required")
    if len(logits) not in (1,len(uniforms)) or uniforms.dtype!=torch.float32:
        raise ValueError("one shared or oneperrow distribution; caller-owned FP32 uniforms required")
    if not bool(torch.isfinite(uniforms).all()) or bool((uniforms<0).any()) or bool((uniforms>=1).any()):
        raise ValueError("uniforms must be finite in[0,1)")
    if logits.device!=borders.device or logits.device!=uniforms.device:
        raise ValueError("logits, borders and uniforms must share a device")
    probabilities=logits.softmax(-1);cdf=torch.cumsum(probabilities,-1)
    if len(logits)==1:
        index=torch.searchsorted(cdf[0],uniforms).clamp(0,probabilities.shape[-1]-1)
        before=torch.cat((torch.zeros(1,dtype=cdf.dtype,device=cdf.device),cdf[0]),dim=0)
        rest=uniforms-before[index]
        probability=probabilities[0,index]
    else:
        index=torch.searchsorted(cdf,uniforms[:,None].contiguous()).squeeze(-1).clamp(0,probabilities.shape[-1]-1)
        before=torch.cat((torch.zeros((*cdf.shape[:-1],1),dtype=cdf.dtype,device=cdf.device),cdf),dim=-1)
        rest=uniforms-before.gather(-1,index[:,None]).squeeze(-1)
        probability=probabilities.gather(-1,index[:,None]).squeeze(-1)
    left=borders[index];right=borders[index+1]
    return left+(right-left)*rest/probability


def finite_bar_icdf_numpy(logits,borders,uniforms):
    """Independent CPU math companion for tests, with the same finite-bin law."""
    import numpy as np
    logits=np.asarray(logits);b=np.asarray(borders);u=np.asarray(uniforms)
    if logits.ndim!=2 or b.ndim!=1 or u.ndim!=1 or logits.shape[1]+1!=len(b) or len(logits) not in (1,len(u)):
        raise ValueError("invalid finite-bar shapes")
    p=np.exp(logits-logits.max(-1,keepdims=True));p/=p.sum(-1,keepdims=True)
    cdf=np.cumsum(p,axis=-1);result=[]
    for row,value in enumerate(u):
        i=0 if len(logits)==1 else row
        index=min(np.searchsorted(cdf[i],value,side="left"),len(b)-2)
        before=0 if index==0 else cdf[i,index-1]
        with np.errstate(divide="ignore",invalid="ignore"):
            result.append(b[index]+(b[index+1]-b[index])*(value-before)/p[i,index])
    return np.asarray(result)
