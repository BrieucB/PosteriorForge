"""CPU-only contracts for the source-guarded resident execution adapter."""
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import inspect
import numpy as np
import pytest
from posteriorforge import execution_cache as cache


def stock_iterator(self,X,*,device,autocast,only_return_standard_out=True):
    for model in self.models:
        model=model.to(device)
        output=model(X)
        model=model.cpu()
        yield output,model.identity


class FakeModel:
    def __init__(self,identity):self.identity=identity;self.device="cpu";self.calls=[]
    def to(self,device):self.device=device;self.calls.append(("to",device));return self
    def cpu(self):self.device="cpu";self.calls.append(("cpu",));return self
    def __call__(self,X):return np.asarray(X)*2+self.identity


def test_resident_iterator_retains_original_outputs_and_removes_only_offload():
    original=SimpleNamespace(models=[FakeModel(0),FakeModel(1)])
    resident=SimpleNamespace(models=[FakeModel(0),FakeModel(1)])
    patched=cache._resident_iter(stock_iterator)
    left=list(stock_iterator(original,[1,2],device="cuda",autocast=False))
    right=list(patched(resident,[1,2],device="cuda",autocast=False))
    for (a,ia),(b,ib) in zip(left,right):
        np.testing.assert_array_equal(a,b);assert ia==ib
    assert all(model.device=="cpu" for model in original.models)
    assert all(model.device=="cuda" for model in resident.models)
    assert all(not any(call[0]=="cpu" for call in model.calls) for model in resident.models)
    again=list(patched(resident,[3],device="cuda",autocast=False))
    np.testing.assert_array_equal(again[1][0],[7])


def without_expected_offload(self,X):
    yield X,None


def test_resident_iterator_refuses_unknown_source_expression():
    with pytest.raises(cache.CacheExecutionError,match="offload expression"):
        cache._resident_iter(without_expected_offload)


def test_training_validation_preserves_raw_fp32_values():
    theta=np.arange(4096*21,dtype=np.float64).reshape(4096,21)
    obs=np.arange(4096*33,dtype=np.float64).reshape(4096,33)
    t,x=cache._numpy_training(theta,obs)
    assert t.dtype==x.dtype==np.float32
    np.testing.assert_array_equal(t,theta.astype(np.float32))
    np.testing.assert_array_equal(x,obs.astype(np.float32))


@pytest.mark.parametrize("theta_shape,obs_shape",[((4095,21),(4095,33)),((4096,20),(4096,33)),((4096,21),(4096,32))])
def test_wrong_training_shapes_fail(theta_shape,obs_shape):
    with pytest.raises(ValueError):cache._numpy_training(np.zeros(theta_shape),np.zeros(obs_shape))


def test_nonfinite_training_and_infinite_queries_fail_without_repair():
    theta=np.zeros((4096,21));theta[0,0]=np.nan
    with pytest.raises(ValueError):cache._numpy_training(theta,np.zeros((4096,33)))
    with pytest.raises(ValueError):cache._numpy_query([[np.inf]*39],6,1024)


@pytest.mark.parametrize("shape,stage,maximum",[((2,38),6,8),((9,39),6,8),((0,39),6,8)])
def test_queries_require_exact_stage_prefix_and_explicit_chunk_limit(shape,stage,maximum):
    with pytest.raises(ValueError):cache._numpy_query(np.zeros(shape),stage,maximum)


def test_query_conversion_does_not_reorder_rows_or_columns():
    x=np.arange(5*39,dtype=np.float64).reshape(5,39)
    actual=cache._numpy_query(x,6,8)
    assert actual.dtype==np.float32
    np.testing.assert_array_equal(actual,x.astype(np.float32))


def active_sdk():
    descriptor={"variant":"cdf_arithmetic_v1","scope":"finite_ensemble_cdf_parentheses_only",
                "source_guard_passed":True,"original_alias_restored":False}
    def helper(logits,*,borders,ys):
        if descriptor["original_alias_restored"]:raise RuntimeError("inactive")
        return logits
    namespace={"_cdf":helper}
    exec("def translate(logits,*,frm,to):\n return _cdf(logits,borders=frm,ys=to)\n",namespace)
    active=namespace["translate"]
    descriptor["active_translator_code_sha256"]=cache._code_sha(active)
    return SimpleNamespace(translate_probs_across_borders=active),descriptor


def test_active_cdf_context_identity_is_required():
    module,descriptor=active_sdk()
    assert cache._require_active_cdf(module,descriptor) is module.translate_probs_across_borders
    descriptor["original_alias_restored"]=True
    with pytest.raises(cache.CacheExecutionError,match="active CDF"):cache._require_active_cdf(module,descriptor)


def test_other_context_descriptor_cannot_borrow_matching_translator_code():
    module,descriptor=active_sdk()
    other=dict(descriptor)
    with pytest.raises(cache.CacheExecutionError,match="does not own"):cache._require_active_cdf(module,other)


def test_closing_explicitly_releases_engine_model_list_even_with_regressor_reference():
    executor=object.__new__(cache.ResidentStageExecutor)
    models=[FakeModel(0),FakeModel(1)];regressor=SimpleNamespace(executor_=SimpleNamespace(models=models))
    executor.regressor=regressor;executor.stage_index=6;executor._prepared={}
    executor.close()
    assert models==[] and regressor.executor_ is None
    assert executor.regressor is None and executor.stage_index is None
    executor.close()


def test_model_rng_context_wraps_calls_and_restores_on_exception():
    events=[]
    @contextmanager
    def fork_rng(*,devices,enabled):
        events.append(("enter",devices,enabled))
        try:yield
        finally:events.append(("restore",))
    obj=object.__new__(cache.ResidentStageExecutor)
    obj.device="cuda";obj.metadata={"model_rng_isolation_count":0}
    obj._torch=SimpleNamespace(device=lambda value:SimpleNamespace(index=None),
                              cuda=SimpleNamespace(current_device=lambda:0),
                              random=SimpleNamespace(fork_rng=fork_rng))
    with pytest.raises(ValueError):
        with obj._model_rng():raise ValueError("failure")
    assert events==[("enter",[0],True),("restore",)]


def test_cpu_safe_execution_identity_is_complete():
    identity=cache.execution_metadata()
    assert identity["engine_id"]=="resident_stage_kv_fp32_v1"
    assert len(identity["module_sha256"])==64
    assert identity["config"]["inference_precision"]=="torch.float32"
    assert identity["config"]["query_batch_axis"]==1
    assert len(identity["SDKsourcehashes"])==9


class CaptureTensor(np.ndarray):
    def __new__(cls,value):return np.asarray(value,dtype=np.float32).view(cls)
    def detach(self):return self
    def cpu(self):return self
    def clone(self):return CaptureTensor(self.copy())


def test_member_capture_returns_original_outputs_and_restores_active_alias(monkeypatch):
    module,descriptor=active_sdk();original=module.translate_probs_across_borders
    stock={"logits":CaptureTensor([[1,2,3]]),"criterion":object()}
    class Regressor:
        executor_=SimpleNamespace(configs=[SimpleNamespace(index=i) for i in range(8)])
        def predict(self,query,*,output_type,quantiles):
            assert output_type=="full" and quantiles==[]
            for _ in range(8):
                module.translate_probs_across_borders(CaptureTensor([[1,2,3]]),
                     frm=CaptureTensor([-3,-1,1,4]),to=CaptureTensor([-3,-1,1,4]))
            return stock
    monkeypatch.setattr(cache,"import_module",lambda name:module)
    actual=cache.capture_corrected_prediction(Regressor(),[[1]],cdf_descriptor=descriptor)
    assert actual is stock and module.translate_probs_across_borders is original
    assert len(actual["members"])==8
    member=actual["members"][3]
    assert member["ensemble_index"]==3
    np.testing.assert_array_equal(member["raw_logits"],[[1,2,3]])
    np.testing.assert_array_equal(member["translated_probabilities"],[[1,2,3]])
    assert len(member["configuration_identity"]["sha256"])==64


def test_member_capture_restores_active_alias_after_prediction_error(monkeypatch):
    module,descriptor=active_sdk();original=module.translate_probs_across_borders
    class Regressor:
        executor_=SimpleNamespace(configs=[SimpleNamespace(index=i) for i in range(8)])
        def predict(self,*args,**kwargs):raise ValueError("model failed")
    monkeypatch.setattr(cache,"import_module",lambda name:module)
    with pytest.raises(ValueError,match="model failed"):
        cache.capture_corrected_prediction(Regressor(),[[1]],cdf_descriptor=descriptor)
    assert module.translate_probs_across_borders is original


class Storage:
    def __init__(self,pointer,size):self.pointer=pointer;self.size=size
    def data_ptr(self):return self.pointer
    def nbytes(self):return self.size


class StoredTensor:
    dtype="torch.float32";device="cuda:0"
    def __init__(self,storage,count):self.storage=storage;self.count=count;self.shape=(count,)
    def numel(self):return self.count
    def element_size(self):return 4
    def untyped_storage(self):return self.storage


def test_cache_telemetry_deduplicates_shared_storage_and_covers_plain_attributes():
    weights=StoredTensor(Storage(100,40),10)
    shared=Storage(200,128);kv=StoredTensor(shared,32);view=StoredTensor(shared,16)
    class Model:
        cached_embeddings=view
        def __init__(self):self.cached_embeddings=view
        def named_parameters(self):return iter([("weight",weights)])
        def named_buffers(self):return iter([("_kv_cache",kv)])
        def named_modules(self):return iter([("",self)])
    obj=object.__new__(cache.ResidentStageExecutor)
    obj.regressor=SimpleNamespace(executor_=SimpleNamespace(models=[Model()]))
    obj.stage_index=6
    obj._torch=SimpleNamespace(is_tensor=lambda value:isinstance(value,StoredTensor),
         cuda=SimpleNamespace(memory_allocated=lambda:168,memory_reserved=lambda:256))
    result=obj.inspect_cache()
    assert result["unique_storage_bytes"]==168
    assert result["unique_cuda_storage_bytes"]==168
    assert len(result["tensor_records"])==3
    assert {row["category"] for row in result["tensor_records"]}=={"parameter","buffer","plain_tensor"}
    assert result["resident_ensemble_count"]==1


def test_nan_queries_pass_unchanged_to_original_sdk_missing_feature_handling():
    query=np.arange(3*39,dtype=np.float64).reshape(3,39)
    query[0,0]=np.nan;query[1,35]=np.nan
    original=query.copy()
    actual=cache._numpy_query(query,6,1024)
    assert actual.dtype==np.float32
    np.testing.assert_array_equal(np.isnan(actual),np.isnan(original))
    np.testing.assert_array_equal(actual,original.astype(np.float32))
    np.testing.assert_array_equal(query,original)
