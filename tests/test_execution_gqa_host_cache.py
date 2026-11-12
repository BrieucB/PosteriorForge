"""CPU-only source, factory, lifecycle and telemetry checks; no real model calls."""
from collections import defaultdict
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import pytest
from posteriorforge import execution_gqa_host_cache as comp


def constructed(monkeypatch,*,verify=False):
    calls=[];attention=(object(),object(),object())
    monkeypatch.setattr(comp.gqa,"_load_guarded_attention",lambda:(*attention,{}))
    def normal_init(self,checkpoint,**kwargs):
        calls.append(kwargs);assert "_stage_factory" not in kwargs
        self.production=True;self.factory=comp.hybrid.HybridStageExecutor;self._closed=False;self._failed=False;self.failure_reason=None
        self.cdf_descriptor=kwargs["cdf_descriptor"];self.cdf_validator=lambda d:None
        self.fallback_map={10:(5,)};self._entries={};self._active=None;self._context=None;self.counters=defaultdict(int)
        self.checkpoint_sha256="checkpoint";self.retention_enabled=True;self.disabled_reason=None;self.initial_admission={"passed":True}
        self.verify_copy_hashes=kwargs["verify_copy_hashes"];self.copy_events=[]
        self._ensure_owner()
    monkeypatch.setattr(comp.retention.HostStageCache,"__init__",normal_init)
    value=comp.GQAHostStageCache("checkpoint",cdf_descriptor={},verify_copy_hashes=verify)
    return value,calls,attention


def restored():
    return {key:True for key in ("original_alias_restored","original_function_restored","original_code_unchanged","active_code_unchanged","gqa_source_unchanged","sdk_sources_unchanged")} | {"expanded_attention_calls":12}


def test_cpu_safe_import_and_exact_source_pins():
    comp.require_frozen_sources()
    subprocess.run([sys.executable,"-c","import posteriorforge.execution_gqa_host_cache,sys;assert "+repr("torch")+" not in sys.modules"],check=True)


def test_constructor_uses_normal_production_guards_and_one_fallback(monkeypatch):
    value,calls,_=constructed(monkeypatch)
    assert calls[0]["fallback_map"]=={10:[5]} and value.production
    assert value.factory is comp._stage_factory and value.metadata["copy_validation"]=="identity_layout_aliases_without_byte_hash"
    assert value.metadata["process_scope"].startswith("private per-process")


def test_mutated_factory_or_class_is_rejected_at_every_acquire(monkeypatch):
    value,_,_=constructed(monkeypatch);value.factory=lambda:None
    with pytest.raises(comp.retention.HostCacheError,match="factory identity"):value._ensure_owner()
    value.factory=comp._stage_factory
    monkeypatch.setattr(comp.stage_execution,"GQAHybridStageExecutor",object())
    with pytest.raises(comp.retention.HostCacheError,match="class alias"):value._ensure_owner()


def test_callables_and_attention_descriptor_are_identity_guarded(monkeypatch):
    value,_,attention=constructed(monkeypatch)
    monkeypatch.setattr(comp.gqa,"_load_guarded_attention",lambda:(attention[0],attention[1],object(),{}))
    with pytest.raises(comp.retention.HostCacheError,match="descriptor identity"):value._ensure_owner()
    monkeypatch.setattr(comp.gqa,"_load_guarded_attention",lambda:(*attention,{}))
    monkeypatch.setattr(comp._STAGE_CLASS,"predict",lambda self,*a,**k:None)
    with pytest.raises(comp.retention.HostCacheError,match="callable changed"):value._ensure_owner()


def test_nonowner_thread_or_wrong_memory_policy_fails_closed(monkeypatch):
    value,_,_=constructed(monkeypatch);monkeypatch.setattr(comp,"get_ident",lambda:value._owner_thread+1)
    with pytest.raises(comp.retention.HostCacheError,match="nonowner thread"):value._ensure_owner()
    with pytest.raises(comp.retention.HostCacheError,match="fixed87GiB"):comp.GQAHostStageCache("x",cdf_descriptor={},rss_limit_bytes=200*comp.GIB)


def test_prediction_requires_restored_alias_before_return_and_offload(monkeypatch):
    value,_,_=constructed(monkeypatch)
    descriptor=restored();monkeypatch.setattr(comp.retention.HostStageCache,"_predict",lambda *a,**k:{"gqa_attention_descriptor":descriptor})
    result=value._predict("predict",None)
    assert result["gqa_attention_descriptor"] is descriptor and value._gqa_totals["restored_prediction_calls"]==1
    stage=SimpleNamespace(last_gqa_descriptor=descriptor);value._active=("context",0);value._entries={value._active:{"stage":stage}}
    called=[];monkeypatch.setattr(comp.retention.HostStageCache,"_offload",lambda self:called.append(True))
    value._offload();assert called==[True]
    descriptor["original_alias_restored"]=False
    with pytest.raises(comp.retention.HostCacheError,match="before CPU offload"):value._offload()


def test_missing_prediction_descriptor_aborts_session(monkeypatch):
    value,_,_=constructed(monkeypatch);aborted=[]
    monkeypatch.setattr(comp.retention.HostStageCache,"_predict",lambda *a,**k:{})
    monkeypatch.setattr(value,"_abort_operation",lambda error:aborted.append(error))
    with pytest.raises(comp.retention.HostCacheError,match="not proven"):value._predict("predict",None)
    assert len(aborted)==1


def test_production_compacts_history_but_retains_cpu_inventory_and_proof_is_full(monkeypatch):
    value,_,_=constructed(monkeypatch)
    state={"tensor_objects":1,"unique_storage_bytes":24,"cpu_storage_bytes":24,"cuda_storage_bytes":0,
           "tensors":[{"shape":[6],"tensor_id":42,"stride":[1]}],"storages":[],"timings_seconds":{}}
    value._entries={0:{"cpu_state":state}};value.copy_events=[{"state":state,"before_state":state}]
    value._compact_events()
    assert "tensors" not in value.copy_events[0]["state"] and "tensors" in value._entries[0]["cpu_state"]
    assert value.copy_events[0]["state"]["tensor_byte_hashes"] is False
    value.verify_copy_hashes=True;value.copy_events=[{"state":state}];value._compact_events()
    assert value.copy_events[0]["state"] is state


def test_callable_mutated_before_construction_is_not_sealed_as_valid(monkeypatch):
    monkeypatch.setattr(comp._STAGE_CLASS,"predict_component",lambda *args,**kwargs:None)
    with pytest.raises(comp.retention.HostCacheError,match="callable changed"):
        comp.GQAHostStageCache("checkpoint",cdf_descriptor={})
