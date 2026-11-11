"""Explicit declared per-member uncached fallbacks over frozen resident execution.

Built with PriorLabs-TabPFN. No SDK, cache engine, mixture implementation or
sampling law is edited. Only declared stage/member pairs use a fresh stock
fit_preprocessors regressor. Every fallback fits all eight original configs,
then temporarily selects one original member for the unchanged predict API.
"""
from __future__ import annotations
from contextlib import contextmanager
import gc
import hashlib
import json
from pathlib import Path
import pickle
from threading import Lock
import time
from . import execution_cache as cache
from . import execution_mixture as mixture

ENGINE_ID="resident_declared_member_fallback_fp32_v1"
ENGINE_VERSION=1
SCHEMA="posteriorforge.execution.hybrid.v1"
PINNED_CACHE_SHA256="45cd3107c50dcf73d03167e2b6dd045cccdb798337e8d3c47b5210cc30dc1a03"
PINNED_MIXTURE_SHA256="e71f43b73ae8259c20b5d7b52feccc7c851e9a2fbb86d869bc0a5f7f3af792a1"
PREPROCESSING_FIELDS=("preprocessors","X_trains","y_trains","ensemble_configs","cat_ixs")


def execution_source_sha256():return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def canonical_fallback_map(value):
    if not isinstance(value,dict) or not value:raise ValueError("explicit nonempty stage-to-member fallback map required")
    result={}
    for stage,members in value.items():
        if isinstance(stage,str) and stage.isdecimal():stage=int(stage)
        if type(stage) is not int or not 0<=stage<21 or stage in result:raise ValueError("unique fallback stages0..20 required")
        if not isinstance(members,(list,tuple)) or not members or any(type(member) is not int or not 0<=member<8 for member in members) or len(set(members))!=len(members):
            raise ValueError("unique fallback member indices0..7 required")
        result[stage]=tuple(sorted(members))
    return dict(sorted(result.items()))


def policy_metadata(fallback_map):
    canonical={str(stage):list(members) for stage,members in canonical_fallback_map(fallback_map).items()}
    return {"fallback_map":canonical,"fallback_policy_sha256":hashlib.sha256(json.dumps(canonical,sort_keys=True,separators=(",",":")).encode()).hexdigest(),
            "policy_scope":"declared stage/member pairs only; no adaptive fallback or cross-stage state reuse"}


def execution_metadata(fallback_map):
    return {"schema":SCHEMA,"engine_id":ENGINE_ID,"engine_version":ENGINE_VERSION,"module_sha256":execution_source_sha256(),
            "resident_cache_module_sha256":PINNED_CACHE_SHA256,"component_mixture_module_sha256":PINNED_MIXTURE_SHA256,
            **policy_metadata(fallback_map),"fallback_fit_mode":"fit_preprocessors","fallback_prepared_members":8,
            "fallback_predict_path":"original TabPFN.predict on one selected original preprocessing member",
            "component_mass_accumulation_dtype":"torch.float64","prediction_math_dtype":"torch.float32",
            "model_rng_isolation":"frozen base executor CPU and selected CUDA fork_rng",
            "with_log_prob_supported":False,"full_mixture_predict_at_fallback_stage_supported":False,
            "attribution":"Built with PriorLabs-TabPFN"}


def require_frozen_sources():
    if cache.execution_source_sha256()!=PINNED_CACHE_SHA256 or mixture.execution_source_sha256()!=PINNED_MIXTURE_SHA256:
        raise cache.CacheExecutionError("hybrid requires the exact frozen resident and component sources")


@contextmanager
def selected_preprocessing_member(executor,member_index,report):
    """Select one original member and restore every exact original container."""
    if type(member_index) is not int or not 0<=member_index<8:raise ValueError("original member index0..7 required")
    originals={name:getattr(executor,name) for name in PREPROCESSING_FIELDS}
    if any(len(value)!=8 for value in originals.values()):raise cache.CacheExecutionError("all8 original preprocessing members required")
    selected={name:[value[member_index]] for name,value in originals.items()}
    report["original_preprocessing_lists_restored"]=False
    try:
        for name,value in selected.items():setattr(executor,name,value)
        yield
    finally:
        conflict=any(getattr(executor,name) is not value for name,value in selected.items())
        for name,value in originals.items():setattr(executor,name,value)
        report["original_preprocessing_lists_restored"]=all(getattr(executor,name) is value and len(value)==8 for name,value in originals.items())
        if conflict:raise cache.CacheExecutionError("preprocessing member lists changed during fallback")


class HybridStageExecutor:
    """One resident stage plus an uncached regressor only at declared stages.

    The caller owns one repaired-CDF context and all sampling/head decisions.
    Each prepare drops previous stage state. No density memo is retained across
    queries or cases. Full-mixture evaluation at fallback stages is unsupported;
    qualification reconstructs all eight captured canonical member outputs.
    """
    def __init__(self,checkpoint,*,cdf_descriptor,fallback_map,device="cuda",query_batch_size=1024):
        require_frozen_sources();self.fallback_map=canonical_fallback_map(fallback_map)
        self.base=cache.ResidentStageExecutor(checkpoint,cdf_descriptor=cdf_descriptor,device=device,query_batch_size=query_batch_size)
        self.cached_component=mixture.ComponentMixtureExecutor(self.base)
        self.uncached_regressor=None;self._busy=Lock();self._prepared=None
        self.hybrid_counters={**execution_metadata(self.fallback_map),"uncached_fit_count":0,"uncached_predict_count":0,"uncached_query_rows":0,
            "resident_component_predict_count":0,"original_preprocessing_lists_restored":True,"original_translator_alias_restored":True,
            "stages":[]}

    @property
    def regressor(self):return self.base.regressor
    @property
    def cdf_descriptor(self):return self.base.cdf_descriptor
    @property
    def stage_index(self):return self.base.stage_index
    @property
    def query_batch_size(self):return self.base.query_batch_size
    @property
    def metadata(self):
        return {**self.base.metadata,"hybrid_execution":self.hybrid_counters,
                "uncached_predict_count":self.hybrid_counters["uncached_predict_count"],"uncached_fit_count":self.hybrid_counters["uncached_fit_count"]}

    def _clear(self):
        self.uncached_regressor=None;self._prepared=None;self.base.close();gc.collect()

    def prepare(self,theta_context,obs_context,stage_index):
        if not self._busy.acquire(blocking=False):raise cache.CacheExecutionError("hybrid operation already active")
        try:
            self._clear();prepared=self.base.prepare(theta_context,obs_context,stage_index)
            result={**prepared,"hybrid_fallback_members":list(self.fallback_map.get(stage_index,())),"uncached_fit_seconds":0.}
            if stage_index in self.fallback_map:
                theta,obs=cache._numpy_training(theta_context,obs_context)
                torch=self.base._torch
                from .inference import frozen_regressor_settings
                settings=frozen_regressor_settings(self.base.checkpoint);settings["inference_precision"]=torch.float32
                regressor=self.base._sdk_regressor.TabPFNRegressor(**settings)
                joint=torch.cat((torch.from_numpy(obs.copy()),torch.from_numpy(theta.copy())),dim=1)
                torch.cuda.synchronize();start=time.perf_counter()
                with self.base._model_rng():regressor.fit(joint[:,:33+stage_index],joint[:,33+stage_index])
                torch.cuda.synchronize();result["uncached_fit_seconds"]=time.perf_counter()-start
                if type(regressor.executor_) is not self.base._sdk_inference.InferenceEngineCachePreprocessing:
                    raise cache.CacheExecutionError("fallback must be the exact stock preprocessing engine")
                if regressor.use_autocast_ or regressor.forced_inference_dtype_!=torch.float32 or regressor.model_.cache_trainset_representation:
                    raise cache.CacheExecutionError("fallback must preserve original uncached FP32 execution")
                if not torch.equal(regressor.renormalized_criterion_.borders,self.base.regressor.renormalized_criterion_.borders):
                    raise cache.CacheExecutionError("fallback canonical physical borders differ")
                config_ids=[cache._configuration_identity(config) for config in regressor.executor_.ensemble_configs]
                resident_ids=[cache._configuration_identity(config) for config in self.base.regressor.executor_.configs]
                if config_ids!=resident_ids:raise cache.CacheExecutionError("all8 fallback configurations must match original member identities")
                preprocessing=[hashlib.sha256(pickle.dumps(p,protocol=5)).hexdigest() for p in regressor.executor_.preprocessors]
                resident_preprocessing=[hashlib.sha256(pickle.dumps(p,protocol=5)).hexdigest() for p in self.base.regressor.executor_.preprocessors]
                if preprocessing!=resident_preprocessing:raise cache.CacheExecutionError("fallback fitted preprocessing differs")
                self.uncached_regressor=regressor;self.hybrid_counters["uncached_fit_count"]+=1
                result.update(fallback_configuration_identities=config_ids,fallback_preprocessor_sha256=preprocessing,
                    fallback_fit_mode=regressor.fit_mode,fallback_autocast=bool(regressor.use_autocast_),
                    fallback_forced_dtype=str(regressor.forced_inference_dtype_),fallback_configurations_match=True,
                    fallback_canonical_borders_exact=True)
            self._prepared=result;self.hybrid_counters["stages"].append(result)
            return result
        except BaseException:
            self._clear();raise
        finally:self._busy.release()

    def predict(self,query,*,output_type="full",capture_members=False):
        if self.stage_index in self.fallback_map:raise NotImplementedError("full cached mixture is not the declared hybrid at fallback stages")
        if not self._busy.acquire(blocking=False):raise cache.CacheExecutionError("hybrid operation already active")
        try:return self.base.predict(query,output_type=output_type,capture_members=capture_members)
        finally:self._busy.release()

    def predict_component(self,member_index,query,*,capture_member=False,with_log_prob=False):
        if with_log_prob:raise NotImplementedError("a component is not a full-mixture log density")
        if type(member_index) is not int or not 0<=member_index<8:raise ValueError("member index0..7 required")
        if self.stage_index is None:raise cache.CacheExecutionError("prepare a stage first")
        if not self._busy.acquire(blocking=False):raise cache.CacheExecutionError("hybrid operation already active")
        try:
            if member_index not in self.fallback_map.get(self.stage_index,()):
                result=self.cached_component.predict_component(member_index,query,capture_member=capture_member)
                result["hybrid_execution_path"]="resident_cache";self.hybrid_counters["resident_component_predict_count"]+=1
                return result
            return self._predict_uncached(member_index,query,capture_member)
        finally:self._busy.release()

    def _predict_uncached(self,member_index,query,capture_member):
        import numpy as np
        regressor=self.uncached_regressor
        if regressor is None:raise cache.CacheExecutionError("declared uncached fallback was not prepared")
        value=cache._numpy_query(query,self.stage_index,self.query_batch_size)
        module=self.base._sdk_regressor;active=cache._require_active_cdf(module,self.cdf_descriptor)
        captures=[];masses=[];calls_before=self.cdf_descriptor["patched_call_count"]
        def record(logits,*,frm,to):
            output=active(logits,frm=frm,to=to);mass=mixture._component_mass_cpu(output)
            array=mass.numpy()
            if not np.isfinite(array).all() or np.any(array<=0) or np.any(array>mixture.HEAD_MASS_ENVELOPE):
                raise cache.CacheExecutionError("fallback raw probability mass outside the unchanged envelope")
            masses.append(mass)
            if capture_member:captures.append({"ensemble_index":member_index,"raw_logits":logits.detach().cpu().clone(),
                "from_borders":frm.detach().cpu().clone(),"to_borders":to.detach().cpu().clone(),
                "translated_probabilities":output.detach().cpu().clone(),"component_probability_mass":array.copy(),
                "configuration_identity":cache._configuration_identity(regressor.executor_.ensemble_configs[0])})
            return output
        with selected_preprocessing_member(regressor.executor_,member_index,self.hybrid_counters):
            self.hybrid_counters["original_translator_alias_restored"]=False
            module.translate_probs_across_borders=record
            try:
                with self.base._model_rng():prediction=regressor.predict(value,output_type="full",quantiles=[])
            finally:
                conflict=module.translate_probs_across_borders is not record
                module.translate_probs_across_borders=active
                self.hybrid_counters["original_translator_alias_restored"]=module.translate_probs_across_borders is active
                if conflict:raise cache.CacheExecutionError("corrected translator alias changed during uncached fallback")
        if len(masses)!=1 or (capture_member and len(captures)!=1) or self.cdf_descriptor["patched_call_count"]-calls_before!=1:
            raise cache.CacheExecutionError("fallback must execute exactly one corrected canonical translation")
        if self.cdf_descriptor.get("executed_corrected_cdf_code_sha256")!=self.cdf_descriptor.get("corrected_cdf_code_sha256") or self.cdf_descriptor.get("corrected_cdf_code_sha256")==self.cdf_descriptor.get("original_cdf_code_sha256"):
            raise cache.CacheExecutionError("fallback did not execute the declared corrected CDF")
        prediction.update(component_raw_mass=masses[0],head_mass_envelope=mixture.HEAD_MASS_ENVELOPE,hybrid_execution_path="original_fit_preprocessors")
        if capture_member:prediction["member"]=captures[0]
        self.hybrid_counters["uncached_predict_count"]+=1;self.hybrid_counters["uncached_query_rows"]+=len(value)
        return prediction

    def inspect_cache(self):
        value=self.base.inspect_cache()
        if self.uncached_regressor is None:return {**value,"uncached_fallback_prepared":False}
        torch=self.base._torch;records=[]
        for category,name,tensor in cache._tensor_records(self.uncached_regressor.model_,torch):
            records.append({"category":category,"name":name,"shape":list(tensor.shape),"dtype":str(tensor.dtype),"device":str(tensor.device),"logical_bytes":tensor.numel()*tensor.element_size()})
        arrays=self.uncached_regressor.executor_.X_trains+self.uncached_regressor.executor_.y_trains
        return {**value,"uncached_fallback_prepared":True,"uncached_fallback_members":list(self.fallback_map[self.stage_index]),
            "uncached_tensor_records":records,"uncached_preprocessed_training_numpy_bytes":sum(array.nbytes for array in arrays)}

    def close(self):
        if not self._busy.acquire(blocking=False):raise cache.CacheExecutionError("cannot close during a hybrid operation")
        try:self._clear()
        finally:self._busy.release()

    def log_prob(self,*args,**kwargs):raise NotImplementedError("hybrid component output is not a full-mixture log density")
