"""CPU-only observer semantics and source/allocator witnesses; no model/GPU calls."""
import ast
import importlib.util
import inspect
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import pytest


def helper():
    path=Path(__file__).parents[1]/"scripts"/"diagnose_prefill_allocator.py"
    spec=importlib.util.spec_from_file_location("prefill_probe_test",path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_import_is_torch_safe_and_no_empty_cache_or_forced_factor_call_exists():
    h=helper();path=Path(h.__file__)
    code="import runpy,sys;runpy.run_path("+repr(str(path))+",run_name=\"import_only\");assert "+repr("torch")+" not in sys.modules"
    subprocess.run([sys.executable,"-c",code],check=True)
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):assert node.func.attr not in ("empty_cache","reset_save_peak_mem_factor")


def test_classmethod_observer_preserves_original_arguments_return_and_descriptor():
    h=helper();seen=[];token=object()
    class Estimator:
        @classmethod
        def get_max_free_memory(cls,device,*,unit,default_gb_cpu_if_failed_to_calculate):
            seen.append((cls,device,unit,default_gb_cpu_if_failed_to_calculate));return 12.5
    original=inspect.getattr_static(Estimator,"get_max_free_memory");observer=h.PassiveObserver()
    observer.patch(Estimator,"get_max_free_memory","free_memory")
    assert Estimator.get_max_free_memory(token,unit="gb",default_gb_cpu_if_failed_to_calculate=8)==12.5
    observer.close()
    assert seen==[(Estimator,token,"gb",8)] and inspect.getattr_static(Estimator,"get_max_free_memory") is original
    assert observer.records[0]["actual_original_return"]==12.5 and observer.restorations[0]["exact_descriptor_restored"]


def test_effective_factor_observer_returns_same_object_and_restores_after_error():
    h=helper();token=SimpleNamespace(shape=(2,3));seen=[]
    class Module:
        def _compute(self,x,*,save_peak_mem_factor,allow_inplace):
            seen.append((x,save_peak_mem_factor,allow_inplace));return x
    original=Module._compute;observer=h.PassiveObserver();observer.patch(Module,"_compute","effective_compute")
    assert Module()._compute(token,save_peak_mem_factor=None,allow_inplace=True) is token
    observer.close();assert Module._compute is original and seen==[(token,None,True)]
    assert observer.records[0]["effective_save_peak_mem_factor"] is None


def test_conflict_is_reported_after_exact_restoration_and_nonowner_rejected(monkeypatch):
    h=helper()
    class Module:
        def work(self):return 1
    original=Module.work;observer=h.PassiveObserver();observer.patch(Module,"work","plain")
    monkeypatch.setattr(h,"get_ident",lambda:observer.owner+1)
    with pytest.raises(RuntimeError,match="nonowner"):Module().work()
    Module.work=lambda self:2
    with pytest.raises(RuntimeError,match="alias changed"):observer.close()
    assert Module.work is original


def test_allocator_snapshot_distinguishes_driver_free_from_reusable_pool():
    h=helper();cuda=SimpleNamespace(memory_stats=lambda:{"num_alloc_retries":3,"inactive_split_bytes.all.current":200},
        mem_get_info=lambda:(100,1000),memory_allocated=lambda:200,memory_reserved=lambda:900,
        get_device_properties=lambda index:SimpleNamespace(total_memory=1000),
        memory=SimpleNamespace(_snapshot=lambda:{"allocator_settings":{"max_split_size":512},"segments":[{"blocks":[{"state":"inactive","size":300},{"state":"active_allocated","size":200}]}]}))
    result=h.allocator_snapshot(SimpleNamespace(cuda=cuda),"phase")
    assert result["driver_free_bytes"]==100 and result["estimator_total_minus_allocated_bytes"]==800
    assert result["inactive_block_bytes"]==300 and result["allocator_settings"]=={"max_split_size":512}


def test_actual_pinned_estimator_uses_total_minus_allocated_not_reserved_or_driver(monkeypatch):
    torch=pytest.importorskip("torch");h=helper()
    from tabpfn.model import memory
    assert h.sha(memory.__file__)==h.MEMORY_SHA256
    monkeypatch.setattr(torch.cuda,"get_device_properties",lambda index:SimpleNamespace(total_memory=1_000_000_000))
    monkeypatch.setattr(torch.cuda,"memory_allocated",lambda index:100_000_000)
    monkeypatch.setattr(torch.cuda,"memory_reserved",lambda index:990_000_000)
    monkeypatch.setattr(torch.cuda,"mem_get_info",lambda *args:(_ for _ in ()).throw(AssertionError("driver free must not be used")))
    actual=memory.MemoryUsageEstimator.get_max_free_memory(SimpleNamespace(type="cuda"),unit="gb",default_gb_cpu_if_failed_to_calculate=8)
    assert actual==.9


def test_real_sdk_scope_restores_exact_descriptors_without_model_or_cuda_calls():
    torch=pytest.importorskip("torch");h=helper()
    from tabpfn.model import memory,transformer,layer,multi_head_attention,mlp
    from posteriorforge import execution_gqa as gqa
    targets=((memory.MemoryUsageEstimator,"reset_peak_memory_if_required"),(transformer.PerFeatureTransformer,"forward"),
             (layer.PerFeatureEncoderLayer,"forward"),(multi_head_attention.MultiHeadAttention,"_compute"),(mlp.MLP,"_compute"))
    before={(cls,name):inspect.getattr_static(cls,name) for cls,name in targets};cuda_before=bool(torch.cuda._initialized)
    with h.observe_sdk(memory,transformer,layer,multi_head_attention,mlp,torch) as observer:
        gqa._load_guarded_attention()
        assert not observer.records
    assert all(inspect.getattr_static(cls,name) is descriptor for (cls,name),descriptor in before.items())
    assert len(observer.restorations)==8 and all(row["exact_descriptor_restored"] for row in observer.restorations)
    assert bool(torch.cuda._initialized)==cuda_before
