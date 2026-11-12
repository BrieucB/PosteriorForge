"""Explicit request-scoped composition of frozen GQA and CPU stage retention.

Built with PriorLabs-TabPFN. The normal production host constructor keeps its
checkpoint/CUDA/CDF guards. Only the stage factory is replaced, before any fit,
by a sealed adapter for the exact frozen GQA one-fallback class. No SDK, stage
preparation, sampler, query density or migration implementation is modified.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from threading import get_ident
from . import execution_host_cache as retention
from . import execution_gqa_hybrid as stage_execution
from . import execution_gqa as gqa
from . import execution_cache as cache, execution_hybrid as hybrid, execution_mixture as mixture

ENGINE_ID="gqa_hybrid_cpu_stage_retention_fp32_v1"
SCHEMA="posteriorforge.execution.gqa_host_cache.v1"
PINNED_HOST_SHA256="effc86e17e33adb96fa8130c476a143b378a63c6d81bf71a31e3e2cd34e9bbde"
PINNED_STAGE_SHA256="ac22c67c23403723902805666d1239bd30a8af355554df58a8638a8fd1f21e82"
PINNED_HYBRID_SHA256=retention.PINNED_HYBRID_SHA256
FALLBACK_MAP={10:(5,)}
GIB=retention.GIB
process_memory=retention.process_memory
tensor_inventory=retention.tensor_inventory
state_manifest=retention.state_manifest
compare_state=retention.compare_state
_STAGE_CLASS=stage_execution.GQAHybridStageExecutor
_SEALED_STAGE_CLASS=_STAGE_CLASS
_SEALED_METHODS={name:(getattr(_STAGE_CLASS,name),getattr(_STAGE_CLASS,name).__code__) for name in ("__init__","prepare","predict","predict_component","_query","close")}


def execution_source_sha256():return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def require_frozen_sources():
    retention.require_frozen_sources();stage_execution.require_frozen_sources()
    if _STAGE_CLASS is not _SEALED_STAGE_CLASS or stage_execution.GQAHybridStageExecutor is not _SEALED_STAGE_CLASS:
        raise retention.HostCacheError("frozen GQA class alias changed")
    for name,(function,code) in _SEALED_METHODS.items():
        if getattr(_STAGE_CLASS,name) is not function or function.__code__ is not code:raise retention.HostCacheError("sealed GQA callable changed: "+name)
    if _stage_factory is not _FACTORY_IDENTITY or _stage_factory.__code__ is not _FACTORY_CODE:raise retention.HostCacheError("sealed stage factory callable changed")
    if retention.execution_source_sha256()!=PINNED_HOST_SHA256 or stage_execution.execution_source_sha256()!=PINNED_STAGE_SHA256:
        raise retention.HostCacheError("exact frozen host retention and GQA stage sources required")


def execution_metadata():
    return {"schema":SCHEMA,"engine_id":ENGINE_ID,"engine_version":1,"module_sha256":execution_source_sha256(),
        "host_retention_module_sha256":PINNED_HOST_SHA256,"gqa_hybrid_module_sha256":PINNED_STAGE_SHA256,
        "gqa_module_sha256":stage_execution.PINNED_GQA_SHA256,"hybrid_module_sha256":PINNED_HYBRID_SHA256,
        "cache_module_sha256":hybrid.PINNED_CACHE_SHA256,"mixture_module_sha256":hybrid.PINNED_MIXTURE_SHA256,
        **hybrid.policy_metadata({10:[5]}),"prediction_math_dtype":"torch.float32","query_density_memoization":False,
        "host_constructor":"normal production constructor; no private test-factory injection",
        "factory_policy":"exact source/class/function/code identities sealed and checked at every operation",
        "residency":"one active GPU stage; retained stage objects CPU within one caller-owned CDF session",
        "memory_policy":"fixed all-or-disabled;100GiB inclusive of8GiB transient slack; no LRU",
        "process_scope":"private per-process retained stages; no cross-process host-cache sharing",
        "production_validation":"CPU inventory identity/layout/aliases/source/owner/residency; no tensor byte hashing",
        "proof_validation":"every full storage hashed plus exact migration identity/layout/aliases",
        "sampling_algorithm_changed":False,"attribution":"Built with PriorLabs-TabPFN"}


def _stage_factory(checkpoint,*,cdf_descriptor,fallback_map,device,query_batch_size):
    if hybrid.canonical_fallback_map(fallback_map)!=FALLBACK_MAP:raise retention.HostCacheError("only declared stage10/member5 fallback permitted")
    stage=_STAGE_CLASS(checkpoint,cdf_descriptor=cdf_descriptor,device=device,query_batch_size=query_batch_size)
    if type(stage) is not _STAGE_CLASS:raise retention.HostCacheError("exact frozen GQA stage class required")
    return stage


_FACTORY_IDENTITY=_stage_factory
_FACTORY_CODE=_stage_factory.__code__


def descriptor_restored(descriptor):
    keys=("original_alias_restored","original_function_restored","original_code_unchanged","active_code_unchanged","gqa_source_unchanged","sdk_sources_unchanged")
    return bool(descriptor) and all(descriptor.get(name) is True for name in keys)


def _inventory_summary(state):
    raw=json.dumps({key:value for key,value in state.items() if key!="timings_seconds"},sort_keys=True,separators=(",",":"))
    return {key:state[key] for key in ("tensor_objects","unique_storage_bytes","cpu_storage_bytes","cuda_storage_bytes")} | {
        "identity_layout_inventory_sha256":hashlib.sha256(raw.encode()).hexdigest(),"tensor_byte_hashes":False}


class GQAHostStageCache(retention.HostStageCache):
    """One-fallback GQA stage state retained on CPU between proposal waves.

    verify_copy_hashes=True is mandatory for a new GPU qualification. Production
    explicitly uses False and still validates scope, source, exact CPU inventory,
    single-stage residency, admission and all GQA alias restoration on every call.
    """
    def __init__(self,checkpoint,*,cdf_descriptor,fallback_map=None,device="cuda",query_batch_size=1024,
                 verify_copy_hashes=False,expected_all_state_bytes=87*GIB,rss_limit_bytes=100*GIB,safety_slack_bytes=8*GIB):
        require_frozen_sources()
        if fallback_map is not None and hybrid.canonical_fallback_map(fallback_map)!=FALLBACK_MAP:raise retention.HostCacheError("only stage10/member5 fallback permitted")
        if (expected_all_state_bytes,rss_limit_bytes,safety_slack_bytes)!=(87*GIB,100*GIB,8*GIB):raise retention.HostCacheError("fixed87GiB estimate/100GiB inclusive limit/8GiB slack required")
        if type(verify_copy_hashes) is not bool:raise ValueError("explicit boolean byte-hash policy required")
        self._composition_ready=False;self._composition_source=execution_source_sha256();self._owner_thread=get_ident()
        self._factory_identity=_stage_factory;self._factory_code=_stage_factory.__code__;self._class_identity=_STAGE_CLASS
        self._method_identities=dict(_SEALED_METHODS)
        self._attention_identity=gqa._load_guarded_attention()[:3]
        self._gqa_totals={"prediction_calls":0,"restored_prediction_calls":0,"expanded_attention_calls":0,"offloads_with_attention_restored":0}
        # Deliberately omit _stage_factory: the immutable production guards run.
        super().__init__(checkpoint,cdf_descriptor=cdf_descriptor,fallback_map={10:[5]},device=device,query_batch_size=query_batch_size,
                         expected_all_state_bytes=expected_all_state_bytes,rss_limit_bytes=rss_limit_bytes,safety_slack_bytes=safety_slack_bytes,
                         verify_copy_hashes=verify_copy_hashes)
        self.factory=self._factory_identity;self._composition_ready=True
        try:self._ensure_owner()
        except BaseException:self.shutdown();raise

    def _ensure_owner(self):
        super()._ensure_owner();require_frozen_sources()
        if get_ident()!=self._owner_thread:raise retention.HostCacheError("host/GQA session accessed from nonowner thread")
        if execution_source_sha256()!=self._composition_source:raise retention.HostCacheError("composition source changed during session")
        if not self._composition_ready:return
        if not self.production or self.factory is not self._factory_identity or _stage_factory is not self._factory_identity or _stage_factory.__code__ is not self._factory_code:
            raise retention.HostCacheError("sealed production stage factory identity changed")
        if _STAGE_CLASS is not self._class_identity or stage_execution.GQAHybridStageExecutor is not self._class_identity:
            raise retention.HostCacheError("sealed GQA class identity changed")
        for name,(function,code) in self._method_identities.items():
            if getattr(self._class_identity,name) is not function or function.__code__ is not code:raise retention.HostCacheError("sealed GQA callable changed: "+name)
        self._require_quiescent_attention()

    def _require_quiescent_attention(self):
        if gqa._LOCK.locked():raise retention.HostCacheError("GQA attention context active outside one scoped prediction")
        actual=gqa._load_guarded_attention()[:3]
        if any(left is not right for left,right in zip(actual,self._attention_identity)):
            raise retention.HostCacheError("original attention alias/function/descriptor identity changed")

    def _context_key(self,theta,obs):
        original=super()._context_key(theta,obs)
        return hashlib.sha256((original+self._composition_source+PINNED_STAGE_SHA256+stage_execution.PINNED_GQA_SHA256).encode()).hexdigest()

    @property
    def active_executor(self):
        if self._active is None:raise retention.HostCacheError("prepare an active stage first")
        return self._entries[self._active]["stage"]
    @property
    def hybrid_counters(self):return self.active_executor.hybrid_counters
    @property
    def cached_component(self):return self.active_executor.cached_component
    @property
    def uncached_regressor(self):return self.active_executor.uncached_regressor
    @property
    def base(self):return self.active_executor.base
    @property
    def metadata(self):
        original=super().metadata
        return {**execution_metadata(),"host_retention":original,"counters":dict(self.counters),"retention_enabled":self.retention_enabled,
                "disabled_reason":self.disabled_reason,"retained_stages":len(self._entries),"context_key":self._context,
                "checkpoint_sha256":self.checkpoint_sha256,"gqa_lifecycle":dict(self._gqa_totals),"copy_validation":original["copy_validation"]}

    def _compact_events(self):
        if self.verify_copy_hashes:return
        for event in self.copy_events:
            for key in ("before_state","state"):
                if key in event and "tensors" in event[key]:event[key]=_inventory_summary(event[key])

    def _offload(self):
        self._require_quiescent_attention()
        if self._active is not None:
            stage=self.active_executor
            if stage.last_gqa_descriptor is not None and not descriptor_restored(stage.last_gqa_descriptor):raise retention.HostCacheError("GQA alias not restored before CPU offload")
            self._gqa_totals["offloads_with_attention_restored"]+=1
        super()._offload();self._compact_events()

    def prepare(self,*args,**kwargs):
        result=super().prepare(*args,**kwargs);self._compact_events();return result

    def _predict(self,name,*args,**kwargs):
        prediction=super()._predict(name,*args,**kwargs)
        descriptor=prediction.get("gqa_attention_descriptor")
        try:
            self._require_quiescent_attention()
            if not descriptor_restored(descriptor):raise retention.HostCacheError("GQA alias/code/source restoration not proven after prediction")
            self._gqa_totals["prediction_calls"]+=1;self._gqa_totals["restored_prediction_calls"]+=1
            self._gqa_totals["expanded_attention_calls"]+=descriptor["expanded_attention_calls"]
            return prediction
        except BaseException as error:
            self._abort_operation(error);raise

    def inspect_cache(self):
        report=super().inspect_cache()
        if not self.verify_copy_hashes:
            report["stages"]=[{**row,"state":_inventory_summary(row["state"])} for row in report["stages"]]
            report["inventory_policy"]="full stored CPU inventory guards retained internally; public inactive-stage telemetry compact; no byteSHA"
        return report


HostStageCache=GQAHostStageCache
