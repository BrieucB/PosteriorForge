"""CPU-only exact projection and conditional-trim dispatch; no real trim/GPU."""
from collections import defaultdict
import ctypes
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import numpy as np
import pytest
from posteriorforge import execution_gqa_host_trim as trim


def fixture(monkeypatch,*,rss_before=95, rss_after=85,hashes=True):
    torch=pytest.importorskip("torch")
    value=object.__new__(trim.TrimGQAHostStageCache);value.torch=torch;value.device="cpu"
    value.verify_copy_hashes=hashes;value.rss_limit_bytes=100*trim.GIB;value.safety_slack_bytes=8*trim.GIB
    value.retention_enabled=True;value.trim_events=[];value._active=("context",20)
    stage=SimpleNamespace(stage_index=20,tensor=torch.arange(16,dtype=torch.float32));value._entries={value._active:{"stage":stage}}
    memory={"rss_bytes":rss_before*trim.GIB,"rss_peak_bytes":96*trim.GIB,"cgroup_current_bytes":rss_before*trim.GIB,"cgroup_limit_bytes":121*trim.GIB}
    value.memory_probe=lambda:dict(memory);value._libc_identity={"version":"fake"};calls=[]
    def reclaim(pad):calls.append(pad);memory["rss_bytes"]=rss_after*trim.GIB;memory["cgroup_current_bytes"]=rss_after*trim.GIB;return 1
    value._trim_function=reclaim
    monkeypatch.setattr(value,"_ensure_owner",lambda:None);monkeypatch.setattr(value,"_require_trim_seal",lambda **kwargs:None)
    original=trim.state_manifest
    def inventory(root,torch,**kwargs):
        result=original(root,torch,**kwargs);result["cuda_storage_bytes"]=5*trim.GIB;return result
    monkeypatch.setattr(trim,"state_manifest",inventory)
    def parent_offload(self):
        decision=trim.retention.memory_admission(self.memory_probe(),5*trim.GIB,self.rss_limit_bytes,self.safety_slack_bytes)
        calls.append(("unchanged_parent",decision));self.retention_enabled=decision["passed"]
    monkeypatch.setattr(trim.composition.GQAHostStageCache,"_offload",parent_offload)
    return value,stage,memory,calls


def test_source_pins_and_import_are_cpu_safe():
    trim.require_frozen_sources()
    subprocess.run([sys.executable,"-c","import posteriorforge.execution_gqa_host_trim,sys;assert "+repr("torch")+" not in sys.modules"],check=True)


def test_passing_original_projection_never_calls_gc_or_trim(monkeypatch):
    value,stage,memory,calls=fixture(monkeypatch,rss_before=80)
    monkeypatch.setattr(trim.gc,"collect",lambda:(_ for _ in ()).throw(AssertionError("gc must not run")))
    value._offload()
    assert len(calls)==1 and calls[0][0]=="unchanged_parent" and not value.trim_events
    assert calls[0][1]["additional_bytes_not_in_current_rss"]==5*trim.GIB


def test_failed_projection_trims_then_calls_exact_unchanged_parent_guard(monkeypatch):
    value,stage,memory,calls=fixture(monkeypatch)
    before=trim.retention.state_manifest(stage,value.torch,hash_bytes=True)
    value._offload();assert calls[0]==0 and calls[-1][0]=="unchanged_parent"
    event=value.trim_events[0]
    assert not event["trigger_projection"]["passed"] and event["parent_projection_after_trim"]["passed"]
    assert event["active_state_comparison"]["full_storage_bytes_equal"] and event["rng_preserved"]
    assert event["retention_enabled_after_parent"] and event["rss_reclaimed_by_trim_bytes"]==10*trim.GIB
    assert event["trigger_projection"]["rss_limit_bytes"]==event["parent_projection_after_trim"]["rss_limit_bytes"]==100*trim.GIB
    assert trim.retention.compare_state(before,trim.retention.state_manifest(stage,value.torch,hash_bytes=True),require_hashes=True)["passed"]


def test_ineffective_trim_still_invokes_original_disable_policy(monkeypatch):
    value,stage,memory,calls=fixture(monkeypatch,rss_after=95);value._offload()
    assert len(value.trim_events)==1 and not value.retention_enabled
    assert not value.trim_events[0]["parent_projection_after_trim"]["passed"]
    assert calls[-1][1]["projected_rss_with_slack_bytes"]==108*trim.GIB


def test_changed_sentinel_or_numpy_rng_is_rejected_before_parent_offload(monkeypatch):
    value,stage,memory,calls=fixture(monkeypatch)
    def corrupt(pad):stage.tensor.add_(1);return 1
    value._trim_function=corrupt
    with pytest.raises(trim.retention.HostCacheError,match="changed live stage"):value._offload()
    assert not any(isinstance(call,tuple) for call in calls)
    value,stage,memory,calls=fixture(monkeypatch)
    value._trim_function=lambda pad:(np.random.rand(),1)[1]
    with pytest.raises(trim.retention.HostCacheError,match="or RNG"):value._offload()


def test_production_inventory_mode_is_explicit_and_keeps_state_guards(monkeypatch):
    value,stage,memory,calls=fixture(monkeypatch,hashes=False);value._offload()
    comparison=value.trim_events[0]["active_state_comparison"]
    assert comparison["passed"] and comparison["full_storage_bytes_equal"] is None
    assert comparison["tensor_identity_layout_aliases_equal"] and comparison["full_storage_alias_graph_equal"]


def test_signature_or_function_address_mutation_rejects(monkeypatch):
    value=object.__new__(trim.TrimGQAHostStageCache)
    class Fn:
        argtypes=[ctypes.c_size_t];restype=ctypes.c_int
    function=Fn();value._libc=SimpleNamespace(malloc_trim=function);value._trim_function=function
    value._libc_identity={"malloc_trim_address":7};monkeypatch.setattr(trim,"_function_address",lambda fn:7)
    value._require_trim_seal(check_files=False)
    function.restype=ctypes.c_void_p
    with pytest.raises(trim.retention.HostCacheError,match="signature changed"):value._require_trim_seal(check_files=False)
