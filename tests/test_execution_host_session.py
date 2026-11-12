"""CPU-only retained-stage accounting and cross-request query isolation."""
from collections import defaultdict
from types import SimpleNamespace
import numpy as np
import pytest
from posteriorforge import execution_host_session as session


def child(index):
    return SimpleNamespace(stage_index=index,regressor=SimpleNamespace(),
        base=SimpleNamespace(metadata={"prepare_count":1,"predict_count":0,"query_rows":0,"model_rng_isolation_count":1}),
        hybrid_counters={"uncached_fit_count":int(index==10),"uncached_predict_count":0,"uncached_query_rows":0,"resident_component_predict_count":0,
                         "original_preprocessing_lists_restored":True,"original_translator_alias_restored":True},
        cached_component=SimpleNamespace(metadata={"component_predict_count":[0]*8,"component_query_rows":[0]*8,"mass_observed_rows":0,"full8_member_lists_restored":True}),
        gqa_counters={"contexts":0,"attention_call_count":0,"expanded_attention_calls":0,"expanded_tensor_views":0,"all_contexts_restored":True,"all_sources_unchanged":True})


class FakeHost:
    def __init__(self,checkpoint,**kwargs):
        assert kwargs["verify_copy_hashes"] is False
        self.counters=defaultdict(int);self._entries={};self._active=None;self.retention_enabled=True;self.disabled_reason=None;self.trim_events=[]
    def _ensure_owner(self):pass
    @property
    def stage_index(self):return None if self._active is None else self._active
    @property
    def active_executor(self):return self._entries[self._active]["stage"]
    @property
    def regressor(self):return self.active_executor.regressor
    def prepare(self,theta,obs,index):
        self.counters["logical_prepare_count"]+=1
        if index in self._entries:self.counters["cache_hit_count"]+=1;self.counters["reload_count"]+=1
        else:self._entries[index]={"stage":child(index)};self.counters["actual_prepare_miss_count"]+=1
        self._active=index
        return {"stage_index":index,"fit_stock_cache_seconds":3.,"resident_move_seconds":1.,"uncached_fit_seconds":.5 if index==10 else 0.,"host_cache_hit":self._entries[index]["stage"].base.metadata["predict_count"]>0}
    def predict_component(self,index,query,**kwargs):
        value=self.active_executor;value.gqa_counters["contexts"]+=1;value.gqa_counters["attention_call_count"]+=12
        value.gqa_counters["expanded_attention_calls"]+=12;value.gqa_counters["expanded_tensor_views"]+=12;value.base.metadata["model_rng_isolation_count"]+=1
        uncached=value.stage_index==10 and index==5
        if uncached:value.hybrid_counters["uncached_predict_count"]+=1;value.hybrid_counters["uncached_query_rows"]+=len(query)
        else:
            value.hybrid_counters["resident_component_predict_count"]+=1;value.base.metadata["predict_count"]+=1;value.base.metadata["query_rows"]+=len(query)
            value.cached_component.metadata["component_predict_count"][index]+=1;value.cached_component.metadata["component_query_rows"][index]+=len(query);value.cached_component.metadata["mass_observed_rows"]+=len(query)
        return {"hybrid_execution_path":"original_fit_preprocessors" if uncached else "resident_cache","marker":np.asarray(query)[:,0].copy()}
    def inspect_cache(self):return {"retained_stages":len(self._entries)}
    def close(self):
        if self._active is not None:self.counters["offload_count"]+=1;self._active=None
    def shutdown(self):self._entries.clear();self._active=None


def make_session(monkeypatch):
    monkeypatch.setattr(session,"require_sources",lambda:None);monkeypatch.setattr(session.engine_module,"TrimGQAHostStageCache",FakeHost)
    return session.HostExecutionSession("checkpoint",cdf_descriptor={},allocator_evidence="evidence",theta_context=np.zeros((4096,21),np.float32),obs_context=np.zeros((4096,33),np.float32))


def test_cold_hit_times_and_all_stage_counters_are_actual_not_replayed(monkeypatch):
    value=make_session(monkeypatch);seeds={"head_proposal":1,"head_acceptance":2,"component_uniform":3};theta=value.theta;obs=value.obs
    for visit,case in enumerate((0,1,0)):
        query=np.full((1,33),case,dtype=np.float32);value.begin_request(case=case,query=query,seed_streams=seeds)
        for index,member in ((0,2),(10,5),(20,7)):
            prepared=value.prepare(theta,obs,index)
            assert prepared["fit_stock_cache_seconds"]==(3. if visit==0 else 0.)
            assert prepared["resident_move_seconds"]==(1. if visit==0 else 0.)
            actual=value.predict_component(member,query);assert actual["marker"][0]==case;value.close()
        record=value.end_request();delta=record["counter_delta"]
        assert delta["host"].get("actual_prepare_miss_count",0)==(3 if visit==0 else 0)
        assert delta["host"].get("cache_hit_count",0)==(0 if visit==0 else 3)
        assert delta["gqa"]["contexts"]==3
    counters=value.counter_snapshot()
    assert counters["base"]["prepare_count"]==3 and counters["hybrid"]["uncached_fit_count"]==1
    assert counters["component"]["component_predict_count"]==[0,0,3,0,0,0,0,3]
    assert counters["routes"]["uncached_calls_by_member"][5]==3
    assert counters["host"]["logical_prepare_count"]==9 and counters["host"]["cache_hit_count"]==6
    summary=value.shutdown();assert summary["shutdown_released_all_stages"] and summary["retained_stages_before_release"]==3


def test_new_case_cannot_reuse_old_query_or_mutate_bank(monkeypatch):
    value=make_session(monkeypatch);value.begin_request(case=1,query=np.ones((1,33)),seed_streams={"a":1,"b":2,"c":3});value.prepare(value.theta,value.obs,0)
    with pytest.raises(RuntimeError,match="active request"):value.predict_component(0,np.zeros((1,33)))
    changed=value.theta.copy();changed[0,0]=1
    with pytest.raises(RuntimeError,match="bank changed"):value.prepare(changed,value.obs,0)
    value.shutdown()


def test_counter_length_or_monotonicity_change_is_rejected():
    with pytest.raises(RuntimeError):session.counter_delta({"a":[1,2]},{"a":[0]})
    with pytest.raises(RuntimeError):session.counter_delta({"a":1},{"a":2})


@pytest.mark.parametrize("failure",["counter","restoration"])
def test_audit_failure_still_releases_stages_before_cdf_exit(monkeypatch,failure):
    from contextlib import ExitStack,contextmanager
    value=make_session(monkeypatch);events=[]
    value.begin_request(case=0,query=np.zeros((1,33)),seed_streams={"a":1,"b":2,"c":3})
    value.prepare(value.theta,value.obs,0)
    if failure=="counter":value.host.active_executor.base.metadata["prepare_count"]=-1
    else:value.host.active_executor.gqa_counters["all_contexts_restored"]=False
    original=value.host.shutdown
    def release():events.append("release");original()
    monkeypatch.setattr(value.host,"shutdown",release)
    @contextmanager
    def cdf_scope():
        try:yield
        finally:
            assert not value.host._entries and value.host._active is None
            events.append("cdf_exit")
    with pytest.raises(RuntimeError):
        with cdf_scope(),ExitStack() as owned:
            owned.callback(value.shutdown)
    assert events==["release","cdf_exit"] and value._closed
    summary=value._shutdown_summary
    assert not summary["accounting_audit_passed"] and summary["stage_release_call_succeeded"] and summary["shutdown_released_all_stages"]
    assert summary["accounting_audit_error"]["type"]=="RuntimeError" and "stage_release_error" not in summary


def test_audit_and_release_errors_remain_distinct_and_audit_is_primary(monkeypatch):
    value=make_session(monkeypatch)
    value.begin_request(case=0,query=np.zeros((1,33)),seed_streams={"a":1,"b":2,"c":3});value.prepare(value.theta,value.obs,0)
    value.host.active_executor.base.metadata["prepare_count"]=-1
    def failed_release():raise OSError("injected release failure")
    monkeypatch.setattr(value.host,"shutdown",failed_release)
    with pytest.raises(RuntimeError,match="counters decreased") as caught:value.shutdown()
    summary=value._shutdown_summary
    assert isinstance(caught.value.__cause__,OSError)
    assert summary["accounting_audit_error"]["type"]=="RuntimeError" and summary["stage_release_error"]=={"type":"OSError","error":"injected release failure"}
    assert not summary["stage_release_call_succeeded"] and not summary["shutdown_released_all_stages"] and value._closed
    with pytest.raises(RuntimeError,match="counters decreased"):value.shutdown()
