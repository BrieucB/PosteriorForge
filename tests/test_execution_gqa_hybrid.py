"""CPU-only composition proof for the explicit GQA-hybrid executor."""
from contextlib import contextmanager
from types import SimpleNamespace
import random
import pytest
from posteriorforge import execution_gqa_hybrid as module


@pytest.fixture
def executor(monkeypatch):
    calls=[]
    def init(self,checkpoint,**kwargs):
        calls.append(("init",checkpoint,kwargs));self.base=SimpleNamespace(metadata={"base":True})
        self.hybrid_counters={};self.fallback_map=kwargs["fallback_map"]
    def predict(self,query,**kwargs):calls.append(("predict",query,kwargs));return {"logits":"unchanged"}
    def component(self,index,query,**kwargs):
        calls.append(("component",index,query,kwargs))
        if kwargs["with_log_prob"]:raise NotImplementedError("density")
        return {"logits":"same-component","hybrid_execution_path":"original_fit_preprocessors" if index==5 else "resident_cache"}
    monkeypatch.setattr(module.hybrid.HybridStageExecutor,"__init__",init)
    monkeypatch.setattr(module.hybrid.HybridStageExecutor,"predict",predict)
    monkeypatch.setattr(module.hybrid.HybridStageExecutor,"predict_component",component)
    monkeypatch.setattr(module.hybrid.HybridStageExecutor,"prepare",lambda self,*a,**k:{"args":a,"kwargs":k})
    monkeypatch.setattr(module.hybrid.HybridStageExecutor,"inspect_cache",lambda self:{"storage":"same"})
    monkeypatch.setattr(module.hybrid.HybridStageExecutor,"metadata",property(lambda self:{"original":"metadata"}))
    @contextmanager
    def context():
        calls.append(("enter",));descriptor={"attention_call_count":12,"expanded_attention_calls":12,"expanded_tensor_views":12,
            "original_alias_restored":False,"original_function_restored":False,"sdk_sources_unchanged":True,"gqa_source_unchanged":True}
        try:yield descriptor
        finally:
            descriptor.update(original_alias_restored=True,original_function_restored=True);calls.append(("exit",))
    monkeypatch.setattr(module.gqa,"expanded_kv_attention",context)
    return module.GQAHybridStageExecutor("checkpoint",cdf_descriptor={"same":"active"}),calls


def test_only_authorized_fallback_and_original_constructor_settings(executor):
    engine,calls=executor
    assert calls[0][2]=={"cdf_descriptor":{"same":"active"},"fallback_map":{10:[5]},"device":"cuda","query_batch_size":1024}
    assert 13 not in engine.fallback_map
    with pytest.raises(TypeError):module.GQAHybridStageExecutor("x",cdf_descriptor={},fallback_map={13:[6]})


def test_predict_keeps_original_value_and_records_restored_actual_context(executor):
    engine,calls=executor;query=object();before=random.getstate()
    result=engine.predict(query,capture_members=True)
    assert result["logits"]=="unchanged" and calls[2][1] is query
    assert result["gqa_attention_descriptor"]["original_alias_restored"] is True
    assert engine.gqa_counters["contexts"]==1 and engine.gqa_counters["expanded_attention_calls"]==12
    assert random.getstate()==before and calls[-1]==("exit",)


def test_component_path_is_original_hybrid_call_with_no_member_reduction(executor):
    engine,calls=executor;query=object();result=engine.predict_component(5,query,capture_member=True)
    assert result["hybrid_execution_path"]=="original_fit_preprocessors"
    assert calls[2]==("component",5,query,{"capture_member":True,"with_log_prob":False})
    assert engine.gqa_counters["all_contexts_restored"] is True


def test_exception_restores_scope_and_records_counters(executor):
    engine,calls=executor
    with pytest.raises(NotImplementedError):engine.predict_component(0,object(),with_log_prob=True)
    assert calls[-1]==("exit",) and engine.last_gqa_descriptor["original_alias_restored"]
    assert engine.gqa_counters["contexts"]==1


def test_preparation_delegates_without_attention_override(executor):
    engine,calls=executor;initial=list(calls)
    result=engine.prepare("theta","obs",13)
    assert result=={"args":("theta","obs",13),"kwargs":{}} and calls==initial
    assert engine.metadata["original"]=="metadata"
    assert engine.inspect_cache()["storage"]=="same"


def test_frozen_dependency_guard_fail_closed(monkeypatch):
    monkeypatch.setattr(module.gqa,"execution_source_sha256",lambda:"changed")
    with pytest.raises(module.cache.CacheExecutionError,match="exact frozen"):
        module.require_frozen_sources()


def test_metadata_is_explicit_fp32_and_not_a_qualification():
    value=module.execution_metadata()
    assert value["fallback_map"]=={"10":[5]} and value["qualified"] is False
    assert value["prediction_math_dtype"]=="torch.float32" and value["sampling_algorithm_changed"] is False
    assert value["gqa_module_sha256"]==module.PINNED_GQA_SHA256
