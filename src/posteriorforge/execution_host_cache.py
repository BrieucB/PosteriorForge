"""Request-scoped CPU retention of exactly prepared hybrid stage state.

Built with PriorLabs-TabPFN. This candidate changes tensor residency only.
It retains no query outputs, densities, uniforms or head assignments. One caller
owns one active corrected-CDF context and one host-cache session at a time.
"""
from __future__ import annotations
from collections import defaultdict
import functools
import gc
import hashlib
import json
from pathlib import Path
from threading import Lock
import time
from types import FunctionType, MethodType, ModuleType
import weakref
import numpy as np
from . import execution_cache as cache, execution_hybrid as hybrid, execution_mixture as mixture

ENGINE_ID = "hybrid_cpu_stage_retention_fp32_v1"
SCHEMA = "posteriorforge.execution.host_stage_cache.v1"
PINNED_HYBRID_SHA256 = "5dfdefa870489ded594de1db13a55a3da5948c720d7a9fd587ebe3a8f8be00a2"
GIB = 1024**3
_OWNER_LOCK = Lock()
_OWNER = None


class HostCacheError(RuntimeError):
    pass


def execution_source_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def require_frozen_sources():
    hybrid.require_frozen_sources()
    if hybrid.execution_source_sha256() != PINNED_HYBRID_SHA256:
        raise HostCacheError("exact frozen hybrid source required")


def execution_metadata(fallback_map):
    return {"schema": SCHEMA, "engine_id": ENGINE_ID, "engine_version": 1,
            "module_sha256": execution_source_sha256(), "hybrid_module_sha256": PINNED_HYBRID_SHA256,
            "cache_module_sha256": cache.execution_source_sha256(), "mixture_module_sha256": mixture.execution_source_sha256(),
            **hybrid.policy_metadata(fallback_map), "residency": "all retained stages CPU; at most one active GPU stage",
            "migration": "full untyped-storage copy preserving Tensor identity, aliases, offset and stride",
            "admission_policy": "fixed admission; permanently disable and release all on measured memory limit",
            "cdf_ownership": "one active caller-owned descriptor for entire session; no rebinding",
            "query_density_memoization": False, "attribution": "Built with PriorLabs-TabPFN"}


def process_memory():
    """Current RSS and own cgroup hard limit; no peak-RSS substitution."""
    import os
    rss = int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    import resource
    result = {"rss_bytes": rss, "rss_peak_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
              "cgroup_limit_bytes": None, "cgroup_current_bytes": None}
    for line in Path("/proc/self/cgroup").read_text().splitlines():
        hierarchy, controllers, relative = line.split(":", 2)
        if hierarchy == "0":
            root = Path("/sys/fs/cgroup") / relative.lstrip("/")
            names = ("memory.max", "memory.current")
        elif "memory" in controllers.split(","):
            root = Path("/sys/fs/cgroup/memory") / relative.lstrip("/")
            names = ("memory.limit_in_bytes", "memory.usage_in_bytes")
        else:
            continue
        try:
            limit = (root / names[0]).read_text().strip()
            result["cgroup_limit_bytes"] = None if limit == "max" or int(limit) >= 2**60 else int(limit)
            result["cgroup_current_bytes"] = int((root / names[1]).read_text())
        except (OSError, ValueError):
            pass
    return result


def memory_admission(memory, additional_bytes, rss_limit_bytes, safety_slack_bytes):
    """additional_bytes excludes every stage already counted in current RSS."""
    rss_projection = memory["rss_bytes"] + additional_bytes + safety_slack_bytes
    limit, current = memory.get("cgroup_limit_bytes"), memory.get("cgroup_current_bytes")
    cgroup_projection = None if current is None else current + additional_bytes + safety_slack_bytes
    passed = rss_projection <= rss_limit_bytes and (limit is None or cgroup_projection is None or cgroup_projection <= limit)
    return {"passed": bool(passed), "current_memory": dict(memory), "additional_bytes_not_in_current_rss": additional_bytes,
            "projected_rss_with_slack_bytes": rss_projection, "projected_cgroup_with_slack_bytes": cgroup_projection,
            "rss_limit_bytes": rss_limit_bytes, "safety_slack_bytes": safety_slack_bytes}


def tensor_inventory(root, torch):
    """Walk owned Python state, including nested plain tensors and view bases.

    Module globals are deliberately excluded; functions expose only closures and
    defaults, modules/classes expose no global object graph. Every stage-owned
    regressor/model/config/preprocessor/container is reached through its attrs.
    """
    found, expanded = {}, set()
    def walk(value, path):
        if torch.is_tensor(value):
            identity = id(value)
            if identity in found:
                found[identity]["paths"].append(path)
                return
            found[identity] = {"tensor": value, "paths": [path]}
            if value._base is not None:
                walk(value._base, path + "/<view_base>")
            if value.is_leaf and value.grad is not None:
                walk(value.grad, path + "/<grad>")
            for name,item in vars(value).items():walk(item,path+"/<tensor_attribute>"+name)
            return
        if value is None or isinstance(value, (str, bytes, int, float, complex, bool, np.ndarray, np.generic, type, ModuleType)):
            return
        identity = id(value)
        if identity in expanded:
            return
        expanded.add(identity)
        if isinstance(value, MethodType):
            walk(value.__self__, path + "/<self>"); walk(value.__func__, path + "/<function>")
        elif isinstance(value, FunctionType):
            walk(value.__defaults__, path + "/<defaults>"); walk(value.__kwdefaults__, path + "/<kwdefaults>")
            for index, cell in enumerate(value.__closure__ or ()):
                try: walk(cell.cell_contents, path + f"/<closure:{index}>")
                except ValueError: pass
        elif isinstance(value, functools.partial):
            walk(value.func, path + "/<function>"); walk(value.args, path + "/<args>"); walk(value.keywords, path + "/<kwargs>")
        elif isinstance(value, dict):
            for index, (key, item) in enumerate(value.items()):
                walk(key, path + f"/<key:{index}>"); walk(item, path + "/" + str(key))
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value): walk(item, path + f"/{index}")
        elif isinstance(value, (set, frozenset)):
            for index, item in enumerate(sorted(value, key=id)): walk(item, path + f"/<set:{index}>")
        else:
            try: attributes = vars(value)
            except TypeError: attributes = {}
            for name, item in attributes.items(): walk(item, path + "/" + name)
            for cls in type(value).__mro__:
                slots = getattr(cls, "__slots__", ())
                if isinstance(slots, str): slots = (slots,)
                for name in slots:
                    if name not in attributes and name not in ("__dict__", "__weakref__") and hasattr(value, name):
                        walk(getattr(value, name), path + "/<slot>" + name)
    walk(root, "stage")
    for row in found.values(): row["paths"] = sorted(set(row["paths"]))
    return sorted(found.values(), key=lambda row: row["paths"][0])


def _groups(inventory):
    groups = {}
    for row in inventory:
        tensor = row["tensor"]
        if tensor.layout != __import__("torch").strided or tensor.is_quantized or tensor.is_conj() or tensor.is_neg():
            raise HostCacheError("only real strided, nonquantized, materialized tensor views supported")
        storage = tensor.untyped_storage()
        key = (str(tensor.device), storage.data_ptr() if storage.nbytes() else storage._cdata, storage.nbytes())
        group = groups.setdefault(key, {"storage": storage, "rows": [], "label": row["paths"][0], "device": str(tensor.device)})
        group["rows"].append(row)
    return sorted(groups.values(), key=lambda group: group["label"])


def _raw(group, torch):
    return torch.empty(0, dtype=torch.uint8, device=group["device"]).set_(group["storage"], 0, (group["storage"].nbytes(),), (1,))


def state_manifest(root, torch, *, hash_bytes=False):
    started=time.perf_counter();hash_seconds=0.
    inventory = tensor_inventory(root, torch); groups = _groups(inventory); rows = []; storages = []
    for group in groups:
        size = group["storage"].nbytes()
        record = {"alias_group": group["label"], "device": group["device"], "bytes": size,
                  "tensor_ids": sorted(id(row["tensor"]) for row in group["rows"])}
        if hash_bytes:
            t0=time.perf_counter();record["sha256"] = hashlib.sha256(_raw(group, torch).cpu().numpy().tobytes()).hexdigest();hash_seconds+=time.perf_counter()-t0
        storages.append(record)
        for row in group["rows"]:
            tensor = row["tensor"]
            rows.append({"paths": row["paths"], "tensor_id": id(tensor), "alias_group": group["label"], "device": str(tensor.device),
                         "dtype": str(tensor.dtype), "shape": list(tensor.shape), "stride": list(tensor.stride()),
                         "storage_offset": tensor.storage_offset(), "contiguous": tensor.is_contiguous(),
                         "requires_grad": tensor.requires_grad, "inference": tensor.is_inference(),
                         "view_base_id": None if tensor._base is None else id(tensor._base),
                         "version": None if tensor.is_inference() else tensor._version})
    return {"tensors": sorted(rows, key=lambda row: row["paths"][0]), "storages": storages,
            "tensor_objects": len(rows), "unique_storage_bytes": sum(row["bytes"] for row in storages),
            "cpu_storage_bytes": sum(row["bytes"] for row in storages if row["device"] == "cpu"),
            "cuda_storage_bytes": sum(row["bytes"] for row in storages if row["device"].startswith("cuda")),
            "timings_seconds":{"hash_and_cpu_download":hash_seconds,"inventory_without_hash":time.perf_counter()-started-hash_seconds}}


def compare_state(before, after, *, require_hashes=False):
    def strip(row, excluded): return {key: value for key, value in row.items() if key not in excluded}
    tensors_equal = [strip(row, {"device"}) for row in before["tensors"]] == [strip(row, {"device"}) for row in after["tensors"]]
    storage_layout_equal = [strip(row, {"device", "sha256"}) for row in before["storages"]] == [strip(row, {"device", "sha256"}) for row in after["storages"]]
    hashes_equal = all("sha256" in row for row in before["storages"] + after["storages"]) and [row.get("sha256") for row in before["storages"]] == [row.get("sha256") for row in after["storages"]]
    return {"passed": bool(tensors_equal and storage_layout_equal and (hashes_equal or not require_hashes)),
            "tensor_identity_layout_aliases_equal": tensors_equal, "full_storage_alias_graph_equal": storage_layout_equal,
            "full_storage_bytes_equal": hashes_equal if require_hashes else None}


def move_tensor_state(root, torch, *, target=None, restore_devices=None, verify_bytes=False, force_copy=False):
    """Migrate full storages; retain original Tensor/Parameter objects and views."""
    if (target is None) == (restore_devices is None): raise ValueError("one target or original-device map required")
    before = state_manifest(root, torch, hash_bytes=verify_bytes)
    inventory = tensor_inventory(root, torch); groups = _groups(inventory)
    origins = {id(row["tensor"]): str(row["tensor"].device) for row in inventory}; planned = []
    for group in groups:
        desired = {str(torch.device(target)) if target is not None else restore_devices[id(row["tensor"])] for row in group["rows"]}
        if len(desired) != 1: raise HostCacheError("one aliased storage cannot restore to different devices")
        destination = next(iter(desired)); planned.append((group, destination))
    # Distinct Storage wrappers over overlapping physical bytes are unsupported.
    ranges = sorted((group["device"], group["storage"].data_ptr(), group["storage"].data_ptr()+group["storage"].nbytes())
                    for group, destination in planned if group["storage"].nbytes() and (destination != group["device"] or force_copy))
    for previous, current in zip(ranges, ranges[1:]):
        if previous[0] == current[0] and current[1] < previous[2]: raise HostCacheError("overlapping distinct storages cannot be migrated safely")
    copied = 0;copy_started=time.perf_counter()
    with torch.inference_mode():
        for group, destination in planned:
            if destination == group["device"] and not force_copy: continue
            copied_raw = _raw(group, torch).to(device=destination, copy=True, non_blocking=False)
            # Tensor inference status follows the assigned data Tensor. Create
            # each typed view under its original inference mode, including when
            # normal and inference tensors share the same physical storage.
            for row in group["rows"]:
                original = row["tensor"]
                with torch.inference_mode(original.is_inference()):
                    replacement = torch.empty(0, dtype=original.dtype, device=destination).set_(
                        copied_raw.untyped_storage(), original.storage_offset(), tuple(original.shape), tuple(original.stride()))
                    original.data = replacement
            copied += group["storage"].nbytes()
    copy_seconds=time.perf_counter()-copy_started
    after = state_manifest(root, torch, hash_bytes=verify_bytes); comparison = compare_state(before, after, require_hashes=verify_bytes)
    if not comparison["passed"]: raise HostCacheError("tensor migration changed values, identities, layout or aliases")
    return {"before": before, "after": after, "original_devices": origins, "copied_bytes": copied, "comparison": comparison,
            "timings_seconds":{"storage_copy_and_tensor_rebinding":copy_seconds,"before":before["timings_seconds"],"after":after["timings_seconds"]}}


class HostStageCache:
    """Fixed-admission stage retention within one request/CDF ownership session."""
    def __init__(self, checkpoint, *, cdf_descriptor, fallback_map, device="cuda", query_batch_size=1024,
                 expected_all_state_bytes=87*GIB, rss_limit_bytes=100*GIB, safety_slack_bytes=8*GIB,
                 verify_copy_hashes=False, _stage_factory=None, _torch=None, _memory_probe=None, _cdf_validator=None):
        global _OWNER
        require_frozen_sources()
        for number in (query_batch_size, expected_all_state_bytes, rss_limit_bytes, safety_slack_bytes):
            if type(number) is not int or number <= 0: raise ValueError("positive integer memory/batch limits required")
        self.checkpoint = str(Path(checkpoint).resolve()); self.cdf_descriptor = cdf_descriptor
        self.fallback_map = hybrid.canonical_fallback_map(fallback_map); self.device = device; self.query_batch_size = query_batch_size
        self.torch = _torch if _torch is not None else __import__("torch")
        self.factory = _stage_factory or hybrid.HybridStageExecutor; self.memory_probe = _memory_probe or process_memory
        self.cdf_validator = _cdf_validator; self.production = _stage_factory is None
        if self.production:
            from .inference import CHECKPOINT_SHA256
            self.checkpoint_sha256 = hashlib.sha256(Path(self.checkpoint).read_bytes()).hexdigest()
            if self.checkpoint_sha256 != CHECKPOINT_SHA256: raise HostCacheError("immutable checkpoint required")
            if not str(device).startswith("cuda"): raise HostCacheError("production host retention requires CUDA")
        else: self.checkpoint_sha256 = "private-cpu-test-factory"
        self.rss_limit_bytes = rss_limit_bytes; self.safety_slack_bytes = safety_slack_bytes; self.verify_copy_hashes = verify_copy_hashes
        self._busy = Lock(); self._entries = {}; self._active = None; self._context = None; self._closed = False; self._failed = False; self.failure_reason = None
        self.counters = defaultdict(int); self.copy_events = []; self.last_prepared = None
        self.initial_admission = memory_admission(self.memory_probe(), expected_all_state_bytes, rss_limit_bytes, safety_slack_bytes)
        self.retention_enabled = self.initial_admission["passed"]; self.disabled_reason = None if self.retention_enabled else "initial projected memory admission failed"
        self._ensure_owner()
        with _OWNER_LOCK:
            if _OWNER is not None and _OWNER() is not None and not _OWNER()._closed: raise HostCacheError("another host-cache session is active")
            _OWNER = weakref.ref(self)

    def _ensure_owner(self):
        if self._closed: raise HostCacheError("host-cache session is closed")
        if self._failed: raise HostCacheError("host-cache session failed and cannot be reused: "+self.failure_reason)
        if self.cdf_validator is not None: self.cdf_validator(self.cdf_descriptor)
        else:
            from tabpfn import regressor
            cache._require_active_cdf(regressor, self.cdf_descriptor)

    def _sync(self):
        if str(self.device).startswith("cuda"): self.torch.cuda.synchronize()

    def _context_key(self, theta, obs):
        digest = hashlib.sha256()
        for value in (theta, obs):
            array = np.ascontiguousarray(value, dtype=np.float32); digest.update(str(array.shape).encode()); digest.update(array.tobytes())
        digest.update(json.dumps({"checkpoint_sha256": self.checkpoint_sha256, "policy": hybrid.policy_metadata(self.fallback_map),
                                  "query_batch_size": self.query_batch_size, "hybrid_sha256": PINNED_HYBRID_SHA256,
                                  "dtype": "torch.float32", "device": str(self.device)}, sort_keys=True).encode())
        return digest.hexdigest()

    @property
    def stage_index(self): return None if self._active is None else self._entries[self._active]["stage"].stage_index
    @property
    def regressor(self): return None if self._active is None else self._entries[self._active]["stage"].regressor
    @property
    def metadata(self):
        return {**execution_metadata(self.fallback_map), "retention_enabled": self.retention_enabled, "disabled_reason": self.disabled_reason,
                "initial_admission": self.initial_admission, "counters": dict(self.counters), "retained_stages": len(self._entries),
                "context_key": self._context, "checkpoint_sha256": self.checkpoint_sha256,
                "failed": self._failed, "failure_reason": self.failure_reason,
                "copy_validation":"full_storage_sha256_identity_layout_aliases" if self.verify_copy_hashes else "identity_layout_aliases_without_byte_hash"}

    def _release_all(self):
        for entry in self._entries.values(): entry["stage"].close()
        self._entries.clear(); self._active = None; gc.collect()

    def _abort_operation(self, error):
        # A group-wise migration can fail after some storages have moved.
        # Such a session is terminal; never expose partially migrated state.
        self._failed = True; self.failure_reason = type(error).__name__+": "+str(error)
        self._release_all()

    def _disable(self, reason):
        self.retention_enabled = False; self.disabled_reason = reason; self.counters["retention_disable_count"] += 1
        self._release_all()

    def _offload(self):
        if self._active is None: return
        key = self._active; entry = self._entries[key]
        if not self.retention_enabled:
            entry["stage"].close(); del self._entries[key]; self._active = None; gc.collect(); return
        inventory = state_manifest(entry["stage"], self.torch)
        decision = memory_admission(self.memory_probe(), inventory["cuda_storage_bytes"], self.rss_limit_bytes, self.safety_slack_bytes)
        if not decision["passed"]:
            self._disable("measured memory before offload exceeded fixed limit"); return
        self._sync(); start = time.perf_counter()
        report = move_tensor_state(entry["stage"], self.torch, target="cpu", verify_bytes=self.verify_copy_hashes)
        self._sync(); entry["original_devices"] = report["original_devices"]; entry["cpu_state"] = report["after"]
        self._active = None; gc.collect()
        if report["after"]["cuda_storage_bytes"]: raise HostCacheError("stage-owned CUDA reference survived offload")
        self.counters["offload_count"] += 1; self.counters["offload_bytes"] += report["copied_bytes"]
        self.copy_events.append({"operation": "offload", "stage_index": entry["stage"].stage_index, "seconds": time.perf_counter()-start,
                                 "copied_bytes": report["copied_bytes"], "comparison": report["comparison"], "timings_seconds":report["timings_seconds"], "before_state": report["before"], "state": report["after"], "admission": decision,
                                 "admission_inventory_seconds":inventory["timings_seconds"]})
        if not memory_admission(self.memory_probe(), 0, self.rss_limit_bytes, self.safety_slack_bytes)["passed"]:
            self._disable("measured RSS after offload exceeded fixed limit")

    def prepare(self, theta_context, obs_context, stage_index):
        if type(stage_index) is not int or not 0 <= stage_index < 21: raise ValueError("stage index0..20 required")
        if not self._busy.acquire(blocking=False): raise HostCacheError("host-cache operation already active")
        try:
            self._ensure_owner(); self.counters["logical_prepare_count"] += 1
            context = self._context_key(theta_context, obs_context)
            if self.production and hashlib.sha256(Path(self.checkpoint).read_bytes()).hexdigest() != self.checkpoint_sha256:
                raise HostCacheError("checkpoint changed during retained session")
            if self._context is not None and context != self._context:
                self._release_all(); self.counters["context_invalidation_count"] += 1
            self._context = context; key = (context, stage_index)
            if self._active == key:
                self.counters["cache_hit_count"] += 1
                return {**self.last_prepared, "host_cache_hit": True, "host_cache_already_active": True}
            self._offload()
            if key in self._entries:
                entry = self._entries[key]; current = state_manifest(entry["stage"], self.torch, hash_bytes=self.verify_copy_hashes)
                if current["cuda_storage_bytes"] or not compare_state(entry["cpu_state"], current, require_hashes=self.verify_copy_hashes)["passed"]:
                    raise HostCacheError("retained CPU stage changed before reload")
                self._sync(); start = time.perf_counter()
                report = move_tensor_state(entry["stage"], self.torch, restore_devices=entry["original_devices"], verify_bytes=self.verify_copy_hashes)
                self._sync(); self._active = key; self.counters["cache_hit_count"] += 1
                self.counters["reload_count"] += 1; self.counters["reload_bytes"] += report["copied_bytes"]
                self.copy_events.append({"operation": "reload", "stage_index": stage_index, "seconds": time.perf_counter()-start,
                                         "copied_bytes": report["copied_bytes"], "comparison": report["comparison"], "timings_seconds":report["timings_seconds"], "before_state": report["before"], "state": report["after"],
                                         "retained_cpu_verification_seconds":current["timings_seconds"]})
                self.last_prepared = {**entry["prepared"], "host_cache_hit": True, "host_cache_already_active": False}
                return self.last_prepared
            if not memory_admission(self.memory_probe(), 0, self.rss_limit_bytes, self.safety_slack_bytes)["passed"]:
                self._disable("measured memory before prepare exceeded fixed limit")
            stage = self.factory(self.checkpoint, cdf_descriptor=self.cdf_descriptor, fallback_map=self.fallback_map,
                                 device=self.device, query_batch_size=self.query_batch_size)
            try: prepared = stage.prepare(theta_context, obs_context, stage_index)
            except BaseException:
                stage.close(); raise
            self._entries[key] = {"stage": stage, "prepared": prepared}; self._active = key
            self.counters["actual_prepare_miss_count"] += 1
            self.last_prepared = {**prepared, "host_cache_hit": False, "host_cache_already_active": False}
            return self.last_prepared
        except BaseException as error:
            self._abort_operation(error); raise
        finally: self._busy.release()

    def _predict(self, name, *args, **kwargs):
        if not self._busy.acquire(blocking=False): raise HostCacheError("host-cache operation already active")
        try:
            self._ensure_owner()
            if self._active is None: raise HostCacheError("prepare an active stage first")
            self.counters[name+"_count"] += 1
            return getattr(self._entries[self._active]["stage"], name)(*args, **kwargs)
        except BaseException as error:
            self._abort_operation(error); raise
        finally: self._busy.release()

    def predict(self, query, **kwargs): return self._predict("predict", query, **kwargs)
    def predict_component(self, member_index, query, **kwargs): return self._predict("predict_component", member_index, query, **kwargs)

    def inspect_cache(self):
        if not self._busy.acquire(blocking=False): raise HostCacheError("host-cache operation already active")
        try:
            self._ensure_owner(); entries = []; cuda_stages = 0
            for key, entry in self._entries.items():
                state = state_manifest(entry["stage"], self.torch); active = key == self._active
                if not active and state["cuda_storage_bytes"]: raise HostCacheError("inactive retained stage owns CUDA storage")
                cuda_stages += int(state["cuda_storage_bytes"] > 0)
                entries.append({"stage_index": key[1], "active": active, "state": state})
            if cuda_stages > 1: raise HostCacheError("more than one GPU stage resident")
            return {**self.metadata, "stages": entries, "one_active_gpu_stage_verified": cuda_stages <= 1,
                    "memory": self.memory_probe(), "global_cuda_allocated_bytes": self.torch.cuda.memory_allocated() if str(self.device).startswith("cuda") else 0,
                    "global_cuda_reserved_bytes": self.torch.cuda.memory_reserved() if str(self.device).startswith("cuda") else 0, "retained_cpu_storage_bytes": sum(row["state"]["cpu_storage_bytes"] for row in entries),
                    "active_stage_telemetry": None if self._active is None else self._entries[self._active]["stage"].inspect_cache()}
        finally: self._busy.release()

    def close(self):
        if not self._busy.acquire(blocking=False): raise HostCacheError("host-cache operation already active")
        try: self._ensure_owner(); self._offload()
        except BaseException as error:
            self._abort_operation(error); raise
        finally: self._busy.release()

    def shutdown(self):
        global _OWNER
        if not self._busy.acquire(blocking=False): raise HostCacheError("host-cache operation already active")
        try:
            if self._closed: return
            self._release_all(); self._closed = True
            with _OWNER_LOCK:
                if _OWNER is not None and _OWNER() is self: _OWNER = None
        finally: self._busy.release()
