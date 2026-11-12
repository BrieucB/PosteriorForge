"""Honest multi-request accounting adapter over frozen conditional-trim stages.

No sampling or query-density memo is implemented here. One caller-owned active
CDF context and immutable training bank span a sequential process-local session.
close offloads; shutdown releases every retained object. Built with PriorLabs-TabPFN.
"""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from threading import get_ident
import numpy as np
from . import execution_gqa_host_trim as engine_module

SCHEMA="posteriorforge.execution.host_session_adapter.v1"
ENGINE_SHA256="1868d0e22906773fe62baeadbdb3124f9f2086ffdfd8cb918caca6aee3cb3e99"
_BASE_KEYS=("prepare_count","predict_count","query_rows","model_rng_isolation_count")
_HYBRID_KEYS=("uncached_fit_count","uncached_predict_count","uncached_query_rows","resident_component_predict_count")
_GQA_KEYS=("contexts","attention_call_count","expanded_attention_calls","expanded_tensor_views")


def source_sha256():return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def require_sources():
    engine_module.require_frozen_sources()
    if engine_module.execution_source_sha256()!=ENGINE_SHA256:raise RuntimeError("frozen qualified host-trim engine source changed")


def _subtract(current,previous):
    result={}
    for key,value in current.items():
        before=previous.get(key,0)
        if isinstance(value,dict):result[key]=_subtract(value,before if isinstance(before,dict) else {})
        elif isinstance(value,list):
            prior=before if isinstance(before,list) else [0]*len(value)
            if len(prior)!=len(value):raise RuntimeError("counter vector length changed")
            result[key]=[a-b for a,b in zip(value,prior)]
        else:result[key]=value-before
    return result


def counter_delta(current,previous):
    result=_subtract(current,previous)
    def valid(value):
        if isinstance(value,dict):return all(valid(item) for item in value.values())
        if isinstance(value,list):return all(valid(item) for item in value)
        return type(value) is int and value>=0
    if not valid(result):raise RuntimeError("session counters decreased or became noninteger")
    return result


class HostExecutionSession:
    def __init__(self,checkpoint,*,cdf_descriptor,allocator_evidence,theta_context,obs_context,query_batch_size=1024):
        require_sources()
        if query_batch_size!=1024:raise ValueError("initial host session requires directly qualified1024 query cap")
        self._source=source_sha256();self._pid=os.getpid();self._thread=get_ident();self._closed=False;self._request=None
        self.theta=np.asarray(theta_context,dtype=np.float32);self.obs=np.asarray(obs_context,dtype=np.float32)
        self._bank_sha=self._bank_identity(self.theta,self.obs);self.cdf_descriptor=cdf_descriptor;self.query_batch_size=query_batch_size
        self.host=engine_module.TrimGQAHostStageCache(checkpoint,allocator_evidence=allocator_evidence,cdf_descriptor=cdf_descriptor,
                                                     query_batch_size=query_batch_size,verify_copy_hashes=False)
        self._seen={};self._totals={"base":{key:0 for key in _BASE_KEYS},"hybrid":{key:0 for key in _HYBRID_KEYS},
            "gqa":{key:0 for key in _GQA_KEYS},"component":{"component_predict_count":[0]*8,"component_query_rows":[0]*8,"mass_observed_rows":0},
            "routes":{"cached_calls_by_member":[0]*8,"uncached_calls_by_member":[0]*8,"cached_rows_by_member":[0]*8,"uncached_rows_by_member":[0]*8}}
        self._restored=True;self.last_acquire=None;self.acquisitions=[];self.request_records=[];self._shutdown_summary=None;self._shutdown_error=None

    @staticmethod
    def _bank_identity(theta,obs):
        if theta.shape!=(4096,21) or obs.shape!=(4096,33) or not np.isfinite(theta).all() or not np.isfinite(obs).all():raise ValueError("unchanged finite4096 paired training bank required")
        digest=hashlib.sha256()
        for array in (theta,obs):digest.update(np.ascontiguousarray(array,dtype=np.float32).tobytes())
        return digest.hexdigest()

    def _ensure(self,*,request=True):
        if self._closed:raise RuntimeError("host session is shut down")
        if os.getpid()!=self._pid or get_ident()!=self._thread:raise RuntimeError("host session belongs to its original process/thread")
        if source_sha256()!=self._source:raise RuntimeError("host session adapter source changed")
        require_sources();self.host._ensure_owner()
        if request and self._request is None:raise RuntimeError("begin an explicit request before model acquisition")

    @property
    def regressor(self):return self.host.regressor
    @property
    def stage_index(self):return self.host.stage_index

    def _observe_active(self):
        if self.host._active is None:return
        stage=self.host.active_executor;identity=(stage.stage_index,id(stage))
        current={"base":{key:int(stage.base.metadata[key]) for key in _BASE_KEYS},
            "hybrid":{key:int(stage.hybrid_counters[key]) for key in _HYBRID_KEYS},
            "gqa":{key:int(stage.gqa_counters[key]) for key in _GQA_KEYS},
            "component":{"component_predict_count":list(stage.cached_component.metadata["component_predict_count"]),
                "component_query_rows":list(stage.cached_component.metadata["component_query_rows"]),"mass_observed_rows":int(stage.cached_component.metadata["mass_observed_rows"])}}
        delta=counter_delta(current,self._seen.get(identity,{}));self._seen[identity]=deepcopy(current)
        for group,values in delta.items():
            for key,value in values.items():
                if isinstance(value,list):self._totals[group][key]=[a+b for a,b in zip(self._totals[group][key],value)]
                else:self._totals[group][key]+=value
        self._restored &= bool(stage.cached_component.metadata["full8_member_lists_restored"] and stage.hybrid_counters["original_preprocessing_lists_restored"]
            and stage.hybrid_counters["original_translator_alias_restored"] and stage.gqa_counters["all_contexts_restored"] and stage.gqa_counters["all_sources_unchanged"])
        if not self._restored:raise RuntimeError("session observed un-restored original member/CDF/GQA state")

    def counter_snapshot(self):
        if not self._closed:self._observe_active()
        return {**deepcopy(self._totals),"host":dict(self.host.counters)}

    def begin_request(self,*,case,query,seed_streams):
        self._ensure(request=False)
        if self._request is not None or type(case) is not int or not 0<=case<8:raise ValueError("one explicit heldout-case request at a time")
        array=np.asarray(query,dtype=np.float32)
        if array.shape not in ((33,),(1,33)) or not np.isfinite(array).all():raise ValueError("raw finite observed33-vector required")
        if len(seed_streams)!=3 or len(set(seed_streams.values()))!=3:raise ValueError("three distinct per-request sampling streams required")
        self._request_observation=array.reshape(1,33).copy()
        self._request={"case":case,"query_fp32_sha256":hashlib.sha256(array.reshape(1,33).tobytes()).hexdigest(),
                       "seed_streams":dict(seed_streams),"counter_start":self.counter_snapshot(),"acquisition_start":len(self.acquisitions)}
        return deepcopy(self._request)

    def prepare(self,theta_context,obs_context,stage_index):
        self._ensure()
        if self._bank_identity(np.asarray(theta_context,dtype=np.float32),np.asarray(obs_context,dtype=np.float32))!=self._bank_sha:raise RuntimeError("session training bank changed")
        self._observe_active();before=dict(self.host.counters);prepared=self.host.prepare(theta_context,obs_context,stage_index);self._observe_active()
        after=dict(self.host.counters);delta=counter_delta(after,before)
        actual_miss=delta.get("actual_prepare_miss_count",0);hit=delta.get("cache_hit_count",0)
        if actual_miss+hit!=1:raise RuntimeError("one logical acquire must be exactly a measured cold fit or retained hit")
        actual={**prepared,"fit_stock_cache_seconds":float(prepared.get("fit_stock_cache_seconds",0.)) if actual_miss else 0.,
                "resident_move_seconds":float(prepared.get("resident_move_seconds",0.)) if actual_miss else 0.,
                "uncached_fit_seconds":float(prepared.get("uncached_fit_seconds",0.)) if actual_miss else 0.}
        self.last_acquire={"case":self._request["case"],"stage_index":stage_index,"logical_acquire_index":after["logical_prepare_count"],
            "host_counter_delta":delta,"actual_resident_stage_fit_count":actual_miss,"retained_stage_hit_count":hit,
            "actual_fit_seconds":{"stock_cache":actual["fit_stock_cache_seconds"],"resident_move":actual["resident_move_seconds"],"uncached_fallback":actual["uncached_fit_seconds"]},
            "original_cold_prepare_provenance":{key:prepared.get(key) for key in ("fit_stock_cache_seconds","resident_move_seconds","uncached_fit_seconds")},
            "hit_provenance_times_are_not_charged_as_new_work":bool(hit)}
        self.acquisitions.append(deepcopy(self.last_acquire));return actual

    def _check_query_scope(self,query):
        value=np.asarray(query,dtype=np.float32)
        if value.ndim!=2 or value.shape[1]<33 or not np.array_equal(value[:,:33],np.broadcast_to(self._request_observation,(len(value),33))):
            raise RuntimeError("query observations differ from active request; no cross-case density reuse")

    def predict(self,query,**kwargs):
        self._ensure();self._check_query_scope(query);result=self.host.predict(query,**kwargs);self._observe_active();return result

    def predict_component(self,member_index,query,**kwargs):
        self._ensure();self._check_query_scope(query);result=self.host.predict_component(member_index,query,**kwargs)
        path=result.get("hybrid_execution_path");route="uncached" if path=="original_fit_preprocessors" else "cached" if path=="resident_cache" else None
        if route is None:raise RuntimeError("actual component route missing")
        self._totals["routes"][route+"_calls_by_member"][member_index]+=1;self._totals["routes"][route+"_rows_by_member"][member_index]+=len(query)
        self._observe_active();return result

    def inspect_cache(self):
        self._ensure();return self.host.active_executor.inspect_cache()

    def close(self):
        self._ensure();self._observe_active();self.host.close()
        if not self.host.retention_enabled:raise RuntimeError("qualified all21 retention disabled by unchanged memory guard: "+str(self.host.disabled_reason))

    def inspect_session(self):
        self._ensure(request=False)
        report=self.host.inspect_cache()
        return {"host":report,"aggregate_counters":self.counter_snapshot(),"all_member_cdf_gqa_states_restored":self._restored,
                "trim_events":deepcopy(self.host.trim_events),"adapter_source_sha256":self._source,"bank_fp32_sha256":self._bank_sha,
                "density_memoization":False,"process_scope":"one process/CDF session; no sharing between ranks"}

    def end_request(self):
        self._ensure()
        if self.host.stage_index is not None:raise RuntimeError("all stage objects must be CPU-offloaded before request completion")
        result={**deepcopy(self._request),"counter_end":self.counter_snapshot()}
        result["counter_delta"]=counter_delta(result["counter_end"],result.pop("counter_start"))
        result["acquisitions"]=deepcopy(self.acquisitions[result.pop("acquisition_start"):])
        self.request_records.append(deepcopy(result));self._request=None;return result

    def shutdown(self):
        if self._closed:
            if self._shutdown_error is not None:raise self._shutdown_error
            return deepcopy(self._shutdown_summary)
        summary={"retained_stages_before_release":len(self.host._entries),"trim_event_count":len(self.host.trim_events),
                 "active_request_aborted":self._request is not None}
        audit_error=None;release_error=None
        try:
            summary["aggregate_counters"]=self.counter_snapshot()
        except BaseException as error:
            audit_error=error;summary["accounting_audit_error"]={"type":type(error).__name__,"error":str(error)}
        finally:
            # Audit failures must never prevent stage release inside the CDF scope.
            try:self.host.shutdown()
            except BaseException as error:
                release_error=error;summary["stage_release_error"]={"type":type(error).__name__,"error":str(error)}
            self._closed=True
            summary.update(accounting_audit_passed=audit_error is None,
                all_member_cdf_gqa_states_restored=self._restored if audit_error is None else False,
                retained_stages_after_release=len(self.host._entries),
                shutdown_released_all_stages=not self.host._entries and self.host._active is None,
                stage_release_call_succeeded=release_error is None)
            self._shutdown_summary=summary
        self._shutdown_error=audit_error if audit_error is not None else release_error
        if audit_error is not None:
            if release_error is not None:raise audit_error from release_error
            raise audit_error
        if release_error is not None:raise release_error
        return deepcopy(summary)
