"""CPU-only static-alias, zero-copy-layout and exact-restoration tests."""
from pathlib import Path
from threading import Thread
from types import SimpleNamespace
import inspect
import numpy as np
import pytest
from posteriorforge import execution_gqa as gqa


class FakeTensor:
    def __init__(self,array,*,owner=None,dtype="torch.float32"):
        self.array=np.asarray(array,dtype=np.float32);self.owner=self.array if owner is None else owner
        self.dtype=dtype
    @property
    def shape(self):return self.array.shape
    def expand(self,*shape):
        return FakeTensor(np.broadcast_to(self.array,shape),owner=self.owner,dtype=self.dtype)
    def stride(self):return tuple(n//self.array.itemsize for n in self.array.strides)
    def numel(self):return self.array.size
    def element_size(self):return self.array.itemsize
    def untyped_storage(self):
        return SimpleNamespace(data_ptr=lambda:self.owner.ctypes.data,nbytes=lambda:self.owner.nbytes)


@pytest.fixture
def attention(tmp_path,monkeypatch):
    calls=[]
    class Attention:
        @staticmethod
        def compute_attention_heads(q,k,v,kv,qkv,dropout_p=None,softmax_scale=None):
            result=(q,k,v,kv,qkv,dropout_p,softmax_scale);calls.append(result);return result
    original=Attention.compute_attention_heads;descriptor=inspect.getattr_static(Attention,"compute_attention_heads")
    source=tmp_path/"attention.py";source.write_text("original-source")
    monkeypatch.setattr(gqa,"PINNED_ATTENTION_SHA256",gqa._sha(source))
    monkeypatch.setattr(gqa,"_load_guarded_attention",lambda:(Attention,original,descriptor,
         {"attention_source":str(source),"attention_source_sha256":gqa._sha(source),
          "resident_cache_module_sha256":gqa.cache.execution_source_sha256()}))
    return Attention,original,descriptor,calls


def q_tensor(heads=4):return FakeTensor(np.arange(2*3*heads*2).reshape(2,3,heads,2))
def packed(heads=1):return FakeTensor(np.arange(2*5*2*heads*2).reshape(2,5,2,heads,2))


def test_packed_kv_is_stride_zero_view_and_original_function_executes(attention):
    cls,original,descriptor,calls=attention;q=q_tensor();kv=packed()
    with gqa.expanded_kv_attention() as report:
        assert cls.compute_attention_heads is not original
        assert "executed_original_attention_code_sha256" not in report
        output=cls.compute_attention_heads(q,None,None,kv,None,dropout_p=.125,softmax_scale=.37)
        assert output[0] is q and output[3] is not kv
        assert output[3].shape==(2,5,2,4,2) and output[3].stride()[-2]==0
        assert output[3].untyped_storage().data_ptr()==kv.untyped_storage().data_ptr()
        np.testing.assert_array_equal(output[3].array,np.repeat(kv.array,4,axis=-2))
        assert output[-2:]==(.125,.37)
        assert report["expanded_attention_calls"]==1 and report["expanded_tensor_views"]==1
        assert report["executed_original_attention_code_sha256"]==report["original_attention_code_sha256"]
    assert inspect.getattr_static(cls,"compute_attention_heads") is descriptor
    assert cls.compute_attention_heads is original and report["original_alias_restored"]
    assert report["original_code_unchanged"] and report["sdk_sources_unchanged"]
    assert len(calls)==1


def test_separate_k_and_v_are_both_expanded_without_storage_copy(attention):
    cls,_,_,_=attention;q=q_tensor()
    k=FakeTensor(np.ones((2,5,1,2)));v=FakeTensor(np.ones((2,5,1,3)))
    with gqa.expanded_kv_attention() as report:
        output=cls.compute_attention_heads(q,k,v,None,None)
        assert output[1].shape==(2,5,4,2) and output[2].shape==(2,5,4,3)
        assert output[1].owner is k.owner and output[2].owner is v.owner
        assert report["expanded_tensor_views"]==2


@pytest.mark.parametrize("kind",["qkv","already_full","multi_kv","single_query_head"])
def test_noneligible_attention_preserves_every_original_object(attention,kind):
    cls,_,_,_=attention
    if kind=="qkv":
        q=None;kv=None;qkv=FakeTensor(np.zeros((2,3,3,4,2)))
    else:
        q=q_tensor(1 if kind=="single_query_head" else 4);qkv=None
        kv=packed(4 if kind=="already_full" else 2 if kind=="multi_kv" else 1)
    with gqa.expanded_kv_attention() as report:
        output=cls.compute_attention_heads(q,None,None,kv,qkv)
    assert output[0] is q and output[3] is kv and output[4] is qkv
    assert report["expanded_attention_calls"]==0


def test_passive_observer_keeps_single_head_layout_unchanged(attention):
    cls,_,_,_=attention;q=q_tensor();kv=packed()
    with gqa.expanded_kv_attention(enabled=False) as report:
        output=cls.compute_attention_heads(q,None,None,kv,None)
    assert output[3] is kv and report["attention_call_count"]==1
    assert report["expanded_attention_calls"]==0 and report["head_shape_histogram"]


def test_body_exception_restores_exact_staticmethod_descriptor(attention):
    cls,original,descriptor,_=attention
    with pytest.raises(ValueError):
        with gqa.expanded_kv_attention() as report:raise ValueError("body")
    assert cls.compute_attention_heads is original
    assert inspect.getattr_static(cls,"compute_attention_heads") is descriptor
    assert report["original_alias_restored"]


def test_nested_context_rejected_without_damaging_outer(attention):
    cls,original,_,_=attention
    with gqa.expanded_kv_attention():
        active=cls.compute_attention_heads
        with pytest.raises(gqa.GQAExecutionError,match="already active"):
            with gqa.expanded_kv_attention():pass
        assert cls.compute_attention_heads is active
    assert cls.compute_attention_heads is original


def test_alias_conflict_fails_after_restoring_original(attention):
    cls,original,descriptor,_=attention
    with pytest.raises(gqa.GQAExecutionError,match="conflicting"):
        with gqa.expanded_kv_attention():
            cls.compute_attention_heads=staticmethod(lambda *args,**kwargs:None)
    assert cls.compute_attention_heads is original
    assert inspect.getattr_static(cls,"compute_attention_heads") is descriptor


def test_other_thread_cannot_silently_use_scoped_attention(attention):
    cls,_,_,_=attention;errors=[]
    def call():
        try:cls.compute_attention_heads(q_tensor(),None,None,packed(),None)
        except Exception as error:errors.append(error)
    with gqa.expanded_kv_attention():
        thread=Thread(target=call);thread.start();thread.join()
    assert len(errors)==1 and isinstance(errors[0],gqa.GQAExecutionError)


def test_no_implicit_precision_change_is_accepted(attention):
    cls,_,_,_=attention;q=q_tensor();q.dtype="torch.float16"
    with pytest.raises(gqa.GQAExecutionError,match="FP32"):
        with gqa.expanded_kv_attention():cls.compute_attention_heads(q,None,None,packed(),None)


def test_actual_pinned_cpu_attention_matches_and_does_not_consume_rng():
    torch=pytest.importorskip("torch")
    pytest.importorskip("tabpfn")
    cls,original,descriptor,_=gqa._load_guarded_attention()
    q=torch.arange(2*3*4*2,dtype=torch.float32).reshape(2,3,4,2)/100
    kv=torch.arange(2*5*2*1*2,dtype=torch.float32).reshape(2,5,2,1,2)/100
    before=torch.get_rng_state().clone()
    expected=original(q,None,None,kv,None,dropout_p=0.,softmax_scale=.37)
    with gqa.expanded_kv_attention() as report:
        actual=cls.compute_attention_heads(q,None,None,kv,None,dropout_p=0.,softmax_scale=.37)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    assert torch.equal(before,torch.get_rng_state())
    assert inspect.getattr_static(cls,"compute_attention_heads") is descriptor
    assert report["expanded_attention_calls"]==1 and report["original_alias_restored"]


def test_metadata_import_never_claims_gpu_or_full21_qualification():
    value=gqa.execution_metadata()
    assert value["qualified"] is False
    assert value["prefill_projection_changed"] is False
    assert value["dtype_changed"] is False
