"""CPU-only storage, ownership and retention-policy checks; no model calls."""
from types import SimpleNamespace
import random
import subprocess
import sys
import numpy as np
import pytest
from posteriorforge import execution_host_cache as host


def test_import_and_frozen_sources_are_cpu_safe():
    host.require_frozen_sources()
    subprocess.run([sys.executable, "-c", "import posteriorforge.execution_host_cache,sys;assert 'torch' not in sys.modules"], check=True)


def owned_state(torch):
    module = torch.nn.Module()
    module.weight = torch.nn.Parameter(torch.arange(24, dtype=torch.float32).reshape(4, 6))
    module.register_buffer("nonpersistent", module.weight.detach()[:, 1:5], persistent=False)
    module.plain = {"nested": [module.weight[:, ::2], module.weight[1:2, :].expand(3, 6)], "same": module.weight}
    module.reinterpret = module.weight.detach().view(torch.uint8)
    module.empty = torch.empty(0)
    with torch.inference_mode():
        module.inference = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        module.inference_view = module.inference[:, ::2]
    return SimpleNamespace(models=[module], other=SimpleNamespace(model_=module))


def test_full_storage_copy_preserves_tensor_identity_views_mixed_dtype_and_rng():
    torch = pytest.importorskip("torch"); root = owned_state(torch)
    before = host.state_manifest(root, torch, hash_bytes=True)
    states = (torch.get_rng_state().clone(), np.random.get_state(), random.getstate())
    report = host.move_tensor_state(root, torch, target="cpu", verify_bytes=True, force_copy=True)
    assert report["comparison"]["passed"] and report["comparison"]["full_storage_bytes_equal"]
    assert report["copied_bytes"] == before["unique_storage_bytes"]
    assert host.compare_state(before, host.state_manifest(root, torch, hash_bytes=True), require_hashes=True)["passed"]
    model = root.models[0]
    assert model.plain["same"] is model.weight
    assert model.plain["nested"][1].stride() == (0, 1)
    assert model.plain["nested"][0]._base is model.weight
    assert model.inference.is_inference() and model.inference_view.is_inference()
    assert torch.equal(torch.get_rng_state(), states[0])
    assert np.array_equal(np.random.get_state()[1], states[1][1]) and random.getstate() == states[2]


def test_distinct_overlapping_numpy_storages_fail_before_mutation():
    torch = pytest.importorskip("torch"); array = np.arange(20, dtype=np.float32)
    root = [torch.from_numpy(array), torch.from_numpy(array[3:])]
    original = [tensor.data_ptr() for tensor in root]
    with pytest.raises(host.HostCacheError, match="overlapping distinct"):
        host.move_tensor_state(root, torch, target="cpu", force_copy=True)
    assert [tensor.data_ptr() for tensor in root] == original


def test_storage_alias_device_split_is_rejected():
    torch = pytest.importorskip("torch"); root = owned_state(torch)
    origins = {id(row["tensor"]): "cpu" for row in host.tensor_inventory(root, torch)}
    origins[id(root.models[0].weight)] = "cuda:0"
    with pytest.raises(host.HostCacheError, match="different devices"):
        host.move_tensor_state(root, torch, restore_devices=origins)


def test_admission_counts_only_additional_bytes_and_checks_cgroup():
    memory = {"rss_bytes": 50, "cgroup_current_bytes": 60, "cgroup_limit_bytes": 110}
    assert host.memory_admission(memory, 20, 100, 10)["passed"]
    assert host.memory_admission(memory, 20, 100, 10)["projected_rss_with_slack_bytes"] == 80
    assert not host.memory_admission(memory, 50, 100, 10)["passed"]
    assert not host.memory_admission(memory, 45, 200, 10)["passed"]


class FakeStage:
    instances = []
    def __init__(self, checkpoint, **kwargs):
        import torch
        self.state = owned_state(torch); self.cdf_descriptor = kwargs["cdf_descriptor"]
        self.stage_index = None; self.regressor = SimpleNamespace(); self.closed = False
        self.instances.append(self)
    def prepare(self, theta, obs, stage_index):
        self.stage_index = stage_index; self.offset = float(np.asarray(theta).sum()); return {"stage_index": stage_index}
    def predict_component(self, member, query, **kwargs):
        return np.asarray(query).sum(axis=1) + self.stage_index + member + self.offset
    def predict(self, query, **kwargs): return self.predict_component(0, query)
    def inspect_cache(self): return {"stage_index": self.stage_index}
    def close(self): self.closed = True; self.state = None; self.stage_index = None


def make_cache(**overrides):
    torch = pytest.importorskip("torch"); descriptor = {"active": True}
    def check(owner):
        if owner is not descriptor or not owner["active"]: raise host.HostCacheError("owner inactive")
    options = dict(cdf_descriptor=descriptor, fallback_map={10: [5], 13: [6]}, device="cpu",
                   expected_all_state_bytes=100, rss_limit_bytes=1000, safety_slack_bytes=10,
                   verify_copy_hashes=True, _stage_factory=FakeStage, _torch=torch,
                   _memory_probe=lambda: {"rss_bytes": 10}, _cdf_validator=check)
    options.update(overrides)
    return host.HostStageCache("fake-checkpoint", **options), descriptor


def test_prepare_hits_and_query_A_B_A_never_memoize_density():
    value, descriptor = make_cache()
    try:
        theta = np.zeros((2, 2)); obs = np.zeros((2, 3)); a = np.array([[1., 2.]]); b = np.array([[9., -3.]])
        assert not value.prepare(theta, obs, 10)["host_cache_hit"]
        first = value.predict_component(5, a)
        value.close()
        assert value.inspect_cache()["stages"][0]["state"]["cuda_storage_bytes"] == 0
        assert value.prepare(theta, obs, 13)["host_cache_hit"] is False
        value.close(); assert value.prepare(theta, obs, 10)["host_cache_hit"]
        np.testing.assert_array_equal(value.predict_component(5, a), first)
        assert not np.array_equal(value.predict_component(5, b), first)
        np.testing.assert_array_equal(value.predict_component(5, a), first)
        assert value.counters["actual_prepare_miss_count"] == 2 and value.counters["cache_hit_count"] == 1
        assert value.counters["logical_prepare_count"] == 3 and value.counters["reload_count"] == 1
        assert value.inspect_cache()["one_active_gpu_stage_verified"]
        value.prepare(theta+1, obs, 10)
        assert value.counters["context_invalidation_count"] == 1 and len(value._entries) == 1
    finally: value.shutdown()


def test_inactive_owner_concurrent_session_and_operations_reject():
    value, descriptor = make_cache()
    try:
        with pytest.raises(host.HostCacheError, match="another host-cache"): make_cache()
        value._busy.acquire()
        try:
            with pytest.raises(host.HostCacheError, match="already active"): value.prepare(np.zeros(1), np.zeros(1), 0)
            with pytest.raises(host.HostCacheError, match="already active"): value.predict(np.zeros((1, 1)))
        finally: value._busy.release()
        descriptor["active"] = False
        with pytest.raises(host.HostCacheError, match="owner inactive"): value.prepare(np.zeros(1), np.zeros(1), 0)
    finally: value.shutdown()
    with pytest.raises(host.HostCacheError, match="closed"): value.prepare(np.zeros(1), np.zeros(1), 0)


def test_failed_admission_disables_all_retention_without_lru():
    memory = {"rss_bytes": 10}; value, _ = make_cache(_memory_probe=lambda: dict(memory))
    try:
        value.prepare(np.zeros(1), np.zeros(1), 0); value.close()
        value.prepare(np.zeros(1), np.zeros(1), 1)
        memory["rss_bytes"] = 995; value.close()
        assert not value.retention_enabled and len(value._entries) == 0
        memory["rss_bytes"] = 10
        value.prepare(np.zeros(1), np.zeros(1), 0); value.close()
        assert len(value._entries) == 0 and value.counters["actual_prepare_miss_count"] == 3
        assert value.counters["cache_hit_count"] == 0
    finally: value.shutdown()


def test_cpu_mutation_is_detected_before_reload():
    torch = pytest.importorskip("torch"); value, _ = make_cache()
    try:
        value.prepare(np.zeros(1), np.zeros(1), 0); value.close()
        with torch.no_grad(): value._entries[next(iter(value._entries))]["stage"].state.models[0].weight.add_(1)
        with pytest.raises(host.HostCacheError, match="retained CPU stage changed"):
            value.prepare(np.zeros(1), np.zeros(1), 0)
    finally: value.shutdown()


def test_slots_only_owned_tensors_are_scanned_and_moved():
    torch=pytest.importorskip("torch")
    class Slotted:
        __slots__=("nested",)
    root=Slotted();root.nested={"tensor":torch.arange(8).reshape(2,4)[:,::2]}
    before=host.state_manifest(root,torch,hash_bytes=True)
    assert before["tensor_objects"]==2
    assert host.move_tensor_state(root,torch,target="cpu",force_copy=True,verify_bytes=True)["comparison"]["passed"]


def test_migration_failure_releases_all_and_makes_session_terminal(monkeypatch):
    value,_=make_cache()
    try:
        value.prepare(np.zeros(1),np.zeros(1),0)
        def broken(*args,**kwargs):raise RuntimeError("mid-copy failure")
        monkeypatch.setattr(host,"move_tensor_state",broken)
        with pytest.raises(RuntimeError,match="mid-copy"):value.close()
        assert value._failed and not value._entries
        with pytest.raises(host.HostCacheError,match="session failed"):value.prepare(np.zeros(1),np.zeros(1),0)
    finally:value.shutdown()
