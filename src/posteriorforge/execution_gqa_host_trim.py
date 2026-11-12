"""Conditional libc page reclaim before the unchanged host admission guard.

Built with PriorLabs-TabPFN. Only an admission projection that would reject may
trigger gc+malloc_trim(0). Actual RSS is then remeasured; no byte subtraction,
limit increase, cache eviction, tensor arithmetic or neural path is changed.
"""
from __future__ import annotations
import ctypes
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import time
import numpy as np
from . import execution_gqa_host_cache as composition
from . import execution_host_cache as retention

ENGINE_ID="gqa_hybrid_cpu_stage_retention_conditional_trim_fp32_v1"
SCHEMA="posteriorforge.execution.gqa_host_trim.v1"
PINNED_COMPOSITION_SHA256="e0f8031bef5085187096d7b30a42310fd8892cad87f6f5d9a443ee5249f506d9"
ALLOCATOR_EVIDENCE_SHA256="0ab24c37e27467b20da531bfea91d1216e46501f925178592957eea9f494b75b"
ALLOCATOR_DIAGNOSTIC_SHA256="9a2a7a600d145c64ef808ba34b47441f59047cd2a7a080da04746eafd6296eaa"
PINNED_HYBRID_SHA256=composition.PINNED_HYBRID_SHA256
PINNED_STAGE_SHA256=composition.PINNED_STAGE_SHA256
GIB=retention.GIB
process_memory=retention.process_memory
tensor_inventory=retention.tensor_inventory
state_manifest=retention.state_manifest
compare_state=retention.compare_state
cache=composition.cache
hybrid=composition.hybrid
mixture=composition.mixture
stage_execution=composition.stage_execution
gqa=composition.gqa


def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def execution_source_sha256():return _sha(__file__)


def require_frozen_sources():
    composition.require_frozen_sources()
    if composition.execution_source_sha256()!=PINNED_COMPOSITION_SHA256:raise retention.HostCacheError("frozen GQA-host composition source required")
    for name,(function,code) in _OWN_METHODS.items():
        if getattr(TrimGQAHostStageCache,name) is not function or function.__code__ is not code:raise retention.HostCacheError("conditional trim callable identity changed: "+name)


def policy_metadata():
    policy={"trigger":"original RSS/cgroup plus incoming unique CUDA storage plus8GiB slack would reject",
            "action":"gc.collect then sealed libc malloc_trim(0), once per rejected boundary",
            "parent_guard":"unchanged original measured-memory admission;100GiB inclusive of8GiB slack",
            "ineffective_action":"unchanged parent disables all retention; no bypass or LRU",
            "rss_bytes_subtracted":0,"rss_limit_bytes":100*GIB,"safety_slack_bytes":8*GIB,
            "allocator_evidence_sha256":ALLOCATOR_EVIDENCE_SHA256}
    return {**policy,"policy_sha256":hashlib.sha256(json.dumps(policy,sort_keys=True,separators=(",",":")).encode()).hexdigest()}


def execution_metadata():
    return {**composition.execution_metadata(),"schema":SCHEMA,"engine_id":ENGINE_ID,"module_sha256":execution_source_sha256(),
            "base_gqa_host_module_sha256":PINNED_COMPOSITION_SHA256,"allocator_diagnostic_source_sha256":ALLOCATOR_DIAGNOSTIC_SHA256,
            "allocator_policy":policy_metadata(),"neural_predict_path_changed":False,"tensor_migration_changed":False}


def authenticate_allocator_evidence(path):
    if _sha(path)!=ALLOCATOR_EVIDENCE_SHA256:raise retention.HostCacheError("exact archived CPU allocator diagnostic required")
    evidence=json.loads(Path(path).read_text())
    for key,value in {"schema":"posteriorforge.cpu_allocator_diagnostic.v1","status":"complete","helper_source_sha256":ALLOCATOR_DIAGNOSTIC_SHA256,
        "source_unchanged":True,"meaningful_free_page_reclaim_observed":True,"cuda_initialized_before":False,"cuda_initialized_after":False,
        "gpu_kernel_calls":0,"model_calls":0,"actual_unique_storage_bytes":5231169160}.items():
        if evidence.get(key)!=value:raise retention.HostCacheError("allocator evidence mismatch: "+key)
    cycles=evidence.get("cycles",[])
    if len(cycles)!=2 or not all(row.get("rng_preserved") and row.get("sentinel_full_storage_identity_layout_aliases",{}).get("passed") for row in cycles):
        raise retention.HostCacheError("both archived allocator sentinel/RNG checks required")
    return evidence


def _function_address(function):return ctypes.cast(function,ctypes.c_void_p).value



class _DlInfo(ctypes.Structure):
    _fields_=[("filename",ctypes.c_char_p),("base",ctypes.c_void_p),("symbol",ctypes.c_char_p),("symbol_address",ctypes.c_void_p)]


def resolve_function(library,function):
    resolver=getattr(library,"dladdr",None)
    if resolver is None:return {"available":False,"reason":"dladdr symbol unavailable"}
    resolver.argtypes=[ctypes.c_void_p,ctypes.POINTER(_DlInfo)];resolver.restype=ctypes.c_int
    info=_DlInfo();success=resolver(ctypes.c_void_p(_function_address(function)),ctypes.byref(info))
    if not success or not info.filename:raise retention.HostCacheError("malloc_trim address did not resolve to a loaded object")
    return {"available":True,"path":str(Path(info.filename.decode()).resolve()),"symbol":None if info.symbol is None else info.symbol.decode()}


def open_trim_runtime():
    library=ctypes.CDLL(None);version=getattr(library,"gnu_get_libc_version",None);trim=getattr(library,"malloc_trim",None)
    if version is None or trim is None:raise retention.HostCacheError("glibc version and malloc_trim symbols required")
    version.restype=ctypes.c_char_p;version=version().decode()
    trim.argtypes=[ctypes.c_size_t];trim.restype=ctypes.c_int
    paths=[]
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields=line.split()
        if len(fields)<6:continue
        candidate=Path(fields[-1])
        if candidate.is_absolute() and (candidate.name.startswith("libc-") or candidate.name=="libc.so.6") and candidate.is_file():paths.append(candidate.resolve())
    paths=sorted(set(paths))
    if not paths:raise retention.HostCacheError("loaded libc object paths required")
    identity={"glibc_version":version,"libc_objects":[{"path":str(path),"sha256":_sha(path)} for path in paths],
              "LD_PRELOAD":os.environ.get("LD_PRELOAD"),"malloc_trim_address":_function_address(trim),
              "malloc_trim_signature":"int malloc_trim(size_t)","malloc_trim_argument":0}
    identity["malloc_trim_dladdr"]=resolve_function(library,trim)
    if identity["malloc_trim_dladdr"]["available"] and identity["malloc_trim_dladdr"]["path"] not in {str(path) for path in paths}:
        raise retention.HostCacheError("malloc_trim resolves to an unrecorded interposer, not loaded libc")
    return library,trim,identity


class TrimGQAHostStageCache(composition.GQAHostStageCache):
    def __init__(self,checkpoint,*,allocator_evidence,**kwargs):
        require_frozen_sources();self._trim_source=execution_source_sha256();self._trim_ready=False
        self._allocator_evidence_path=str(Path(allocator_evidence).resolve());self._allocator_evidence=authenticate_allocator_evidence(allocator_evidence)
        self._libc,self._trim_function,self._libc_identity=open_trim_runtime()
        if self._libc_identity["glibc_version"]!=self._allocator_evidence["libc"]["glibc_version"]:
            raise retention.HostCacheError("actual glibc version differs from archived allocator diagnostic")
        self._libc_identity["exact_binary_match_to_diagnostic"]=self._libc_identity["libc_objects"]==self._allocator_evidence["libc"]["libc_objects"]
        self.trim_events=[]
        super().__init__(checkpoint,**kwargs)
        self._trim_ready=True
        try:self._ensure_owner()
        except BaseException:self.shutdown();raise

    def _ensure_owner(self):
        super()._ensure_owner();require_frozen_sources()
        if execution_source_sha256()!=self._trim_source:raise retention.HostCacheError("conditional trim source changed during session")
        if self._trim_ready:self._require_trim_seal(check_files=False)

    def _require_trim_seal(self,*,check_files):
        if getattr(self._libc,"malloc_trim",None) is not self._trim_function or _function_address(self._trim_function)!=self._libc_identity["malloc_trim_address"]:
            raise retention.HostCacheError("sealed libc malloc_trim callable/address changed")
        if self._trim_function.argtypes!=[ctypes.c_size_t] or self._trim_function.restype is not ctypes.c_int:
            raise retention.HostCacheError("sealed libc malloc_trim signature changed")
        if check_files:
            if _sha(self._allocator_evidence_path)!=ALLOCATOR_EVIDENCE_SHA256:raise retention.HostCacheError("archived allocator evidence changed")
            if resolve_function(self._libc,self._trim_function)!=self._libc_identity["malloc_trim_dladdr"]:raise retention.HostCacheError("malloc_trim loaded-object resolution changed")
            if any(_sha(row["path"])!=row["sha256"] for row in self._libc_identity["libc_objects"]):raise retention.HostCacheError("loaded libc file identity changed")

    def _context_key(self,theta,obs):
        original=super()._context_key(theta,obs)
        identity=json.dumps({key:value for key,value in self._libc_identity.items() if key!="malloc_trim_address"},sort_keys=True,separators=(",",":"))
        return hashlib.sha256((original+self._trim_source+policy_metadata()["policy_sha256"]+identity).encode()).hexdigest()

    @property
    def metadata(self):
        original=super().metadata
        return {**original,**execution_metadata(),"base_gqa_host_execution":original,"trim_event_count":len(self.trim_events),
                "libc_identity":dict(self._libc_identity),"allocator_evidence_sha256":ALLOCATOR_EVIDENCE_SHA256,
                "copy_validation":original["copy_validation"]}

    def _rng_snapshot(self):
        torch=self.torch
        return (torch.get_rng_state().clone(),[state.clone() for state in torch.cuda.get_rng_state_all()] if str(self.device).startswith("cuda") else [],random.getstate(),np.random.get_state())

    def _rng_equal(self,before):
        after=self._rng_snapshot()
        return bool(self.torch.equal(before[0],after[0]) and len(before[1])==len(after[1]) and all(self.torch.equal(a,b) for a,b in zip(before[1],after[1])) and before[2]==after[2] and before[3][0]==after[3][0] and np.array_equal(before[3][1],after[3][1]) and before[3][2:]==after[3][2:])

    def _live_storage(self):
        states=[state_manifest(entry["stage"],self.torch) for entry in self._entries.values()]
        return {"stage_count":len(states),"cpu_storage_bytes":sum(row["cpu_storage_bytes"] for row in states),
                "cuda_storage_bytes":sum(row["cuda_storage_bytes"] for row in states),
                "scope":"sum of per-stage unique storage inventories; no RSS subtraction"}

    def _conditional_trim(self,active_inventory,projection):
        self._require_trim_seal(check_files=True);start=time.perf_counter();rng=self._rng_snapshot()
        before=state_manifest(self.active_executor,self.torch,hash_bytes=self.verify_copy_hashes)
        event={"stage_index":self.stage_index,"trigger_projection":projection,"incoming_unique_cuda_storage_bytes":active_inventory["cuda_storage_bytes"],
            "live_cache_storage_before":self._live_storage(),"memory_before_gc":self.memory_probe(),"libc_identity":dict(self._libc_identity),
            "allocator_evidence_sha256":ALLOCATOR_EVIDENCE_SHA256,"policy_sha256":policy_metadata()["policy_sha256"]}
        t0=time.perf_counter();event["gc_collected_objects"]=gc.collect();event["gc_seconds"]=time.perf_counter()-t0
        event["memory_immediately_before_trim"]=self.memory_probe();t0=time.perf_counter()
        event["malloc_trim_return"]=int(self._trim_function(0));event["malloc_trim_seconds"]=time.perf_counter()-t0
        event["memory_immediately_after_trim"]=self.memory_probe()
        event["rss_reclaimed_by_trim_bytes"]=event["memory_immediately_before_trim"]["rss_bytes"]-event["memory_immediately_after_trim"]["rss_bytes"]
        after=state_manifest(self.active_executor,self.torch,hash_bytes=self.verify_copy_hashes)
        event["active_state_comparison"]=compare_state(before,after,require_hashes=self.verify_copy_hashes)
        if self.verify_copy_hashes:event["active_state_before"]=before;event["active_state_after"]=after
        event["rng_preserved"]=self._rng_equal(rng);event["live_cache_storage_after"]=self._live_storage()
        event["parent_projection_after_trim"]=retention.memory_admission(self.memory_probe(),after["cuda_storage_bytes"],self.rss_limit_bytes,self.safety_slack_bytes)
        event["seconds"]=time.perf_counter()-start;self.trim_events.append(event)
        self._require_trim_seal(check_files=True)
        if not event["active_state_comparison"]["passed"] or not event["rng_preserved"] or event["live_cache_storage_before"]!=event["live_cache_storage_after"]:
            raise retention.HostCacheError("conditional trim changed live stage values/layout/aliases/storage or RNG")

    def _offload(self):
        self._ensure_owner();events_before=len(self.trim_events)
        if self._active is not None and self.retention_enabled:
            inventory=state_manifest(self.active_executor,self.torch)
            projection=retention.memory_admission(self.memory_probe(),inventory["cuda_storage_bytes"],self.rss_limit_bytes,self.safety_slack_bytes)
            if not projection["passed"]:self._conditional_trim(inventory,projection)
        # Always recomputes the ORIGINAL measured projection. A failed reclaim
        # still invokes the ORIGINAL disable-all policy without any allowance.
        super()._offload()
        if len(self.trim_events)>events_before:self.trim_events[-1]["retention_enabled_after_parent"]=self.retention_enabled


_OWN_METHODS={name:(getattr(TrimGQAHostStageCache,name),getattr(TrimGQAHostStageCache,name).__code__) for name in ("__init__","_ensure_owner","_require_trim_seal","_context_key","_rng_snapshot","_rng_equal","_live_storage","_conditional_trim","_offload")}
HostStageCache=TrimGQAHostStageCache
