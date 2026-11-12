"""Explicit FP32 expanded-KV execution with one declared original fallback.

Built with PriorLabs-TabPFN. This module composes frozen implementations;
all context preparation, eight member identities and sampling laws are retained.
Only stage10/member5 uses the original corrected uncached prediction path.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
from . import execution_cache as cache, execution_mixture as mixture
from . import execution_hybrid as hybrid, execution_gqa as gqa

ENGINE_ID="expanded_kv_declared_member_fallback_fp32_v1"
ENGINE_VERSION=1
SCHEMA="posteriorforge.execution.gqa_hybrid.v1"
PINNED_HYBRID_SHA256="5dfdefa870489ded594de1db13a55a3da5948c720d7a9fd587ebe3a8f8be00a2"
PINNED_GQA_SHA256="8b2cff752855e1b1966d2e374322602fa71368b1926da9d3b0261edd111aee6b"
FALLBACK_MAP={10:(5,)}


def execution_source_sha256():return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def execution_metadata():
    return {"schema":SCHEMA,"engine_id":ENGINE_ID,"engine_version":ENGINE_VERSION,
        "module_sha256":execution_source_sha256(),"base_hybrid_module_sha256":PINNED_HYBRID_SHA256,
        "gqa_module_sha256":PINNED_GQA_SHA256,
        "resident_cache_module_sha256":hybrid.PINNED_CACHE_SHA256,
        "component_mixture_module_sha256":hybrid.PINNED_MIXTURE_SHA256,
        **hybrid.policy_metadata({10:[5]}),"prediction_math_dtype":"torch.float32",
        "scope":"frozen stage preparation; expanded shared KV query views; original stage10/member5 fallback",
        "prefill_projection_changed":False,"sampling_algorithm_changed":False,
        "execution_optimized":True,"with_log_prob_supported":False,
        "qualified":False,"attribution":"Built with PriorLabs-TabPFN"}


def require_frozen_sources():
    hybrid.require_frozen_sources()
    if hybrid.execution_source_sha256()!=PINNED_HYBRID_SHA256 or gqa.execution_source_sha256()!=PINNED_GQA_SHA256:
        raise cache.CacheExecutionError("GQA hybrid requires exact frozen hybrid and attention-view adapter sources")


class GQAHybridStageExecutor(hybrid.HybridStageExecutor):
    """Caller owns the repaired-CDF context and independent sampling streams."""
    def __init__(self,checkpoint,*,cdf_descriptor,device="cuda",query_batch_size=1024):
        require_frozen_sources()
        super().__init__(checkpoint,cdf_descriptor=cdf_descriptor,fallback_map={10:[5]},
                         device=device,query_batch_size=query_batch_size)
        self.gqa_counters={**execution_metadata(),"contexts":0,"attention_call_count":0,
            "expanded_attention_calls":0,"expanded_tensor_views":0,"all_contexts_restored":True,
            "all_sources_unchanged":True}
        self.last_gqa_descriptor=None

    @property
    def metadata(self):
        return {**super().metadata,"gqa_hybrid_execution":self.gqa_counters}

    def prepare(self,*args,**kwargs):
        require_frozen_sources()
        return super().prepare(*args,**kwargs)

    def _query(self,call):
        require_frozen_sources();descriptor=None
        try:
            with gqa.expanded_kv_attention() as descriptor:prediction=call()
            prediction["gqa_attention_descriptor"]=dict(descriptor)
            return prediction
        finally:
            if descriptor is not None:
                self.last_gqa_descriptor=descriptor
                counters=self.gqa_counters;counters["contexts"]+=1
                for name in ("attention_call_count","expanded_attention_calls","expanded_tensor_views"):
                    counters[name]+=descriptor[name]
                counters["all_contexts_restored"] &= bool(descriptor["original_alias_restored"] and descriptor["original_function_restored"])
                counters["all_sources_unchanged"] &= bool(descriptor["sdk_sources_unchanged"] and descriptor["gqa_source_unchanged"])
            require_frozen_sources()

    def predict(self,query,*,output_type="full",capture_members=False):
        parent=super().predict
        return self._query(lambda:parent(query,output_type=output_type,capture_members=capture_members))

    def predict_component(self,member_index,query,*,capture_member=False,with_log_prob=False):
        parent=super().predict_component
        return self._query(lambda:parent(member_index,query,capture_member=capture_member,with_log_prob=with_log_prob))

    def inspect_cache(self):
        return {**super().inspect_cache(),"gqa_hybrid_execution":dict(self.gqa_counters)}
