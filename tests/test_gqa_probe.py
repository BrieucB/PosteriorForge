"""CPU-safe contracts for the bounded query-layout diagnostic."""
from contextlib import contextmanager
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

spec=importlib.util.spec_from_file_location("gqa_probe",Path(__file__).parents[1]/"scripts/boundedprobe_gqa_expansion.py")
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)


def test_bounded_plan_is_not_an_accidental_full21_screen():
    probe.validate_plan((6,10,13),(16,128),900)
    probe.validate_plan((10,),(16,128,1024),300)
    for stages,rows,seconds in (((0,),(16,128),900),((6,6),(16,128),900),((6,),(1,16),900),((6,),(16,128),901)):
        with pytest.raises(ValueError):probe.validate_plan(stages,rows,seconds)


def test_comparison_helper_is_source_guarded_and_cpu_safe():
    helper=probe.load_utilities()
    assert helper.THRESHOLDS["max_cdf_absolute"]==2e-5
    assert helper.compare_probabilities(np.array([[.5,.5]]),np.array([[.5,.5]]))["passed"]


def fake_context(expanded_calls):
    @contextmanager
    def context(*,enabled):
        result={"original_alias_restored":False,"sdk_sources_unchanged":True,
                "expanded_attention_calls":expanded_calls if enabled else 0}
        try:yield result
        finally:result["original_alias_restored"]=True
    return context


def test_prediction_observer_returns_exact_original_result(monkeypatch):
    monkeypatch.setattr(probe.gqa,"expanded_kv_attention",fake_context(3))
    result=object();prediction,descriptor=probe.observed_prediction(lambda:result,expanded=True)
    assert prediction is result and descriptor["expanded_attention_calls"]==3
    assert descriptor["original_alias_restored"]


def test_requested_expansion_must_actually_execute(monkeypatch):
    monkeypatch.setattr(probe.gqa,"expanded_kv_attention",fake_context(0))
    with pytest.raises(RuntimeError,match="did not execute"):
        probe.observed_prediction(lambda:object(),expanded=True)


class Tensor(np.ndarray):
    def __new__(cls,value):return np.asarray(value,dtype=np.float32).view(cls)
    def detach(self):return self
    def cpu(self):return self
    def contiguous(self):return Tensor(np.ascontiguousarray(np.asarray(self)))
    def numpy(self):return np.asarray(self)
    def softmax(self,dim):
        a=np.asarray(self);p=np.exp(a-a.max(axis=dim,keepdims=True))
        return Tensor(p/p.sum(axis=dim,keepdims=True))
    def stride(self):return tuple(step//self.itemsize for step in self.strides)
    def numel(self):return self.size
    def element_size(self):return self.itemsize
    @property
    def device(self):return "cpu"
    def untyped_storage(self):
        return SimpleNamespace(data_ptr=lambda:self.ctypes.data,nbytes=lambda:self.nbytes)


def prediction(logits):
    raw=Tensor(logits);borders=Tensor([-2,-1,0,3])
    return {"logits":raw,"criterion":SimpleNamespace(borders=borders),
        "members":[{"raw_logits":raw,"from_borders":borders,
                    "translated_probabilities":raw.softmax(-1)} for _ in range(8)]}


def test_compact_capture_keeps_real_source_rows_including_worst_cdf():
    control=prediction(np.zeros((8,3)));values=np.zeros((8,3));values[4]=[8,-8,-8]
    candidate=prediction(values);captures={};positions=np.arange(8)*7
    probe.capture_compact(captures,10,"expanded",candidate,control,positions)
    np.testing.assert_array_equal(captures["stage10_expanded_original_sample_rows"],[0,28,49])
    assert captures["stage10_expanded_logits"].shape==(3,3)
    assert len([key for key in captures if key.endswith("_raw_logits")])==8


def test_cached_embedding_hash_covers_tensor_larger_than_old4096_cutoff():
    embedding=Tensor(np.arange(22*192).reshape(22,192));kv=Tensor(np.ones((2,4,2,1,2)))
    class Model:
        cached_embeddings=embedding
        def __init__(self):self.cached_embeddings=embedding
        def named_parameters(self):return iter([])
        def named_buffers(self):return iter([("_kv_cache",kv)])
        def named_modules(self):return iter([("",self)])
    engine=SimpleNamespace(regressor=SimpleNamespace(executor_=SimpleNamespace(models=[Model()])),
        _torch=SimpleNamespace(is_tensor=lambda value:isinstance(value,Tensor)))
    before=probe.prepared_state(engine);embedding[21,191]+=1
    after=probe.prepared_state(engine)
    assert before["tensor_storage_records"]==after["tensor_storage_records"]
    assert before["all_cached_embedding_hashes"]!=after["all_cached_embedding_hashes"]


def test_profile_reports_working_peak_above_existing_cache(monkeypatch,tmp_path):
    monkeypatch.setattr(probe.gqa,"expanded_kv_attention",fake_context(4))
    cuda=SimpleNamespace(synchronize=lambda:None,reset_peak_memory_stats=lambda:None,
        memory_allocated=lambda:100,max_memory_allocated=lambda:180,max_memory_reserved=lambda:256)
    utilities=SimpleNamespace(profile_prediction=lambda call,output,name,torch:{"summary":{"kernel":call()}})
    result=probe.profile_path(lambda:"actual-kernel-name",expanded=True,output=tmp_path,
        name="fake",torch=SimpleNamespace(cuda=cuda),utilities=utilities)
    assert result["peak_working_bytes_above_before"]==80
    assert result["summary"]["kernel"]=="actual-kernel-name"
