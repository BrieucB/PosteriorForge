#!/usr/bin/env python3
"""Bounded CPU allocator replay from authenticated stage20 storage sizes.

No CUDA initialization, GPU kernels, model calls or posterior sampling. The
single allocated GPU is deliberately idle and remains fully budget-accounted.
Built with PriorLabs-TabPFN.
"""
from __future__ import annotations
import argparse
import ctypes
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import resource
import time
from types import SimpleNamespace

SCHEMA="posteriorforge.cpu_allocator_diagnostic.v1"
FAILED_PROOF_SHA256="1c849f74784937c69e8d76f769eeb0f7785b1a6bbccfdb7f6e12903e10b6defc"
HOST_SOURCE_SHA256="effc86e17e33adb96fa8130c476a143b378a63c6d81bf71a31e3e2cd34e9bbde"
EXPECTED_STAGE20_BYTES=5231169160
MAX_REPLAY_BYTES=5240000000
MEANINGFUL_RECLAIM_BYTES=256*1024**2


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def storage_sizes(manifest,cap_bytes=MAX_REPLAY_BYTES):
    if manifest.get("status")!="failed":raise ValueError("preserved failed source required")
    stages=manifest.get("all21_cpu_residency",{}).get("stages",[])
    selected=[row for row in stages if row.get("stage_index")==20]
    if len(selected)!=1:raise ValueError("exactly one stage20 state required")
    state=selected[0]["state"];storages=state.get("storages",[])
    if state.get("cuda_storage_bytes")!=0:raise ValueError("source stage20 must be entirely CPU resident")
    if any(row.get("device")!="cpu" or type(row.get("bytes")) is not int or row["bytes"]<0 for row in storages):raise ValueError("nonnegative unique CPU-storage sizes required")
    labels=[row["alias_group"] for row in storages]
    if len(labels)!=len(set(labels)):raise ValueError("source storage aliases must be counted only once")
    sizes=[row["bytes"] for row in storages if row["bytes"]]
    if not sizes or sum(sizes)!=state.get("unique_storage_bytes") or sum(sizes)!=state.get("cpu_storage_bytes"):
        raise ValueError("source unique-storage accounting mismatch")
    if sum(sizes)>cap_bytes:raise ValueError("bounded replay byte cap exceeded")
    return sizes


def cgroup_memory():
    records=[]
    try:memberships=Path("/proc/self/cgroup").read_text().splitlines();mounts=Path("/proc/self/mountinfo").read_text().splitlines()
    except OSError as error:return {"readable":False,"error":str(error)}
    for member in memberships:
        hierarchy,controllers,relative=member.split(":",2)
        version=2 if hierarchy=="0" else 1
        if version==1 and "memory" not in controllers.split(","):continue
        for mount in mounts:
            left,right=mount.split(" - ",1);fields=left.split();kind,source,options=right.split()[:3]
            if (version==2 and kind!="cgroup2") or (version==1 and (kind!="cgroup" or "memory" not in options.split(","))):continue
            root=fields[3];point=fields[4]
            if root!="/" and relative!=root and not relative.startswith(root.rstrip("/")+"/"):continue
            tail=relative.lstrip("/") if root=="/" else relative[len(root):].lstrip("/")
            directory=Path(point)/tail
            names=("memory.current","memory.peak","memory.max") if version==2 else ("memory.usage_in_bytes","memory.max_usage_in_bytes","memory.limit_in_bytes")
            record={"version":version,"membership":relative,"mount_root":root,"mount_point":point,"directory":str(directory),"files":{},"scope":"own process memory cgroup; membership labels show whether Slurm-scoped"}
            for label,name in zip(("current_bytes","peak_bytes","limit_bytes"),names):
                try:
                    raw=(directory/name).read_text().strip();record["files"][label]={"path":str(directory/name),"raw":raw}
                    record[label]=None if raw=="max" or (label=="limit_bytes" and int(raw)>=2**60) else int(raw)
                except (OSError,ValueError) as error:record[label]=None;record["files"][label]={"path":str(directory/name),"error":str(error)}
            record["readable"]=record["current_bytes"] is not None;record["finite_hard_limit_readable"]=record["limit_bytes"] is not None;records.append(record)
    return {"readable":any(row["readable"] for row in records),"records":records}


def memory_snapshot(label):
    rss=int(Path("/proc/self/statm").read_text().split()[1])*os.sysconf("SC_PAGE_SIZE")
    return {"label":label,"current_rss_bytes":rss,"process_peak_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            "cgroup":cgroup_memory()}


def libc_trim():
    library=ctypes.CDLL(None);version=getattr(library,"gnu_get_libc_version",None)
    if version is not None:version.restype=ctypes.c_char_p;version=version().decode()
    paths=[]
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields=line.split()
        if len(fields)<6:continue
        candidate=Path(fields[-1]);name=candidate.name
        if candidate.is_absolute() and (name.startswith("libc-") or name=="libc.so.6") and candidate.is_file():paths.append(candidate.resolve())
    paths=sorted(set(paths))
    descriptor={"glibc_version":version,"libc_objects":[{"path":str(path),"sha256":sha(path)} for path in paths],"LD_PRELOAD":os.environ.get("LD_PRELOAD")}
    trim=getattr(library,"malloc_trim",None)
    descriptor["malloc_trim_available"]=trim is not None
    if trim is not None:trim.argtypes=[ctypes.c_size_t];trim.restype=ctypes.c_int
    return trim,descriptor


def sentinel_state(torch):
    base=torch.arange(256*1024,dtype=torch.float32,device="cpu")
    return SimpleNamespace(base=base,view=base.reshape(512,512)[:,::2],expanded=base[:512].reshape(1,512).expand(4,512),byte_view=base.view(torch.uint8))


def replay_cycle(torch,sizes,sentinel,host,trim,cycle,memory=memory_snapshot):
    start=time.perf_counter();before=host.state_manifest(sentinel,torch,hash_bytes=True);rng=torch.get_rng_state().clone();python_rng=random.getstate()
    rows=[memory("before_allocate")]
    allocations=[]
    for size in sizes:
        allocations.append(torch.empty(size,dtype=torch.uint8,device="cpu"));allocations[-1].fill_(cycle+1)
    rows.append(memory("all_replay_storages_live"));del allocations
    gc.collect();rows.append(memory("after_delete_gc"))
    trim_start=time.perf_counter();returned=None if trim is None else int(trim(0));trim_seconds=time.perf_counter()-trim_start
    rows.append(memory("after_malloc_trim"));after=host.state_manifest(sentinel,torch,hash_bytes=True)
    comparison=host.compare_state(before,after,require_hashes=True)
    preserved=bool(torch.equal(rng,torch.get_rng_state()) and python_rng==random.getstate())
    delta=rows[-2]["current_rss_bytes"]-rows[-1]["current_rss_bytes"]
    if not comparison["passed"] or not preserved:raise RuntimeError("CPU allocator replay changed sentinel values/layout/aliases or RNG")
    return {"cycle":cycle,"allocated_unique_storage_bytes":sum(sizes),"allocated_storage_count":len(sizes),"snapshots":rows,
            "malloc_trim_return":returned,"malloc_trim_seconds":trim_seconds,"rss_reclaimed_by_trim_bytes":delta,
            "post_delete_above_cycle_baseline_bytes":rows[-2]["current_rss_bytes"]-rows[0]["current_rss_bytes"],
            "meaningful_reclaim_threshold_bytes":MEANINGFUL_RECLAIM_BYTES,"meaningful_free_page_reclaim":returned is not None and delta>=MEANINGFUL_RECLAIM_BYTES,
            "sentinel_full_storage_identity_layout_aliases":comparison,"rng_preserved":preserved,"elapsed_seconds":time.perf_counter()-start}


def run(*,failed_host_proof,ledger,output,max_seconds=90):
    from posteriorforge.artifacts import runtime_identity,write_json
    from posteriorforge.reference_grid import _require_budgeted_gpu
    from posteriorforge import execution_host_cache as host
    if type(max_seconds) is not int or not 10<=max_seconds<=90:raise ValueError("bounded max-seconds10..90 required")
    budget=_require_budgeted_gpu(ledger)
    if budget["gpus"]!=1:raise RuntimeError("one budget-accounted intentionally idle GPU allocation required")
    output=Path(output);output.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    report={**runtime_identity(),"schema":SCHEMA,"status":"running","budget_binding":budget,"helper_source_sha256":sha(__file__),
            "gpu_reservation_purpose":"intentionally idle GPU; all allocated seconds remain budget-accounted",
            "cuda_initialization_requested":False,"gpu_kernel_calls":0,"model_calls":0,"posterior_vectors_generated":0,
            "cycles":[],"attribution":"Built with PriorLabs-TabPFN"};write_json(output/"manifest.json",report)
    try:
        if sha(failed_host_proof)!=FAILED_PROOF_SHA256:raise ValueError("exact preserved original failed host proof required")
        if host.execution_source_sha256()!=HOST_SOURCE_SHA256:raise RuntimeError("frozen sentinel inventory/migration utility changed")
        source=json.loads(Path(failed_host_proof).read_text());sizes=storage_sizes(source)
        if sum(sizes)!=EXPECTED_STAGE20_BYTES:raise ValueError("authenticated stage20 bytes differ from predeclared replay")
        import torch
        if getattr(torch.cuda,"_initialized",False):raise RuntimeError("fresh CPU-only process required; CUDA already initialized")
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS","16")))
        trim,identity=libc_trim();sentinel=sentinel_state(torch)
        report.update(torch_version=torch.__version__,source_failed_host_proof_sha256=FAILED_PROOF_SHA256,source_status=source["status"],stage_index=20,
            actual_unique_storage_sizes_bytes=sizes,actual_unique_storage_bytes=sum(sizes),max_replay_bytes=MAX_REPLAY_BYTES,
            libc=identity,cuda_initialized_before=False,host_inventory_source_sha256=HOST_SOURCE_SHA256)
        for cycle in range(2):
            if time.perf_counter()-start>=max_seconds:raise TimeoutError("bounded CPU allocator replay deadline")
            report["cycles"].append(replay_cycle(torch,sizes,sentinel,host,trim,cycle));write_json(output/"manifest.json",report)
        report["cuda_initialized_after"]=bool(getattr(torch.cuda,"_initialized",False))
        if report["cuda_initialized_after"]:raise RuntimeError("CPU-only allocator diagnostic unexpectedly initialized CUDA")
        report["source_unchanged"]=sha(__file__)==report["helper_source_sha256"] and sha(failed_host_proof)==FAILED_PROOF_SHA256 and host.execution_source_sha256()==HOST_SOURCE_SHA256
        if not report["source_unchanged"]:raise RuntimeError("allocator diagnostic source/evidence custody changed")
        report.update(status="complete",meaningful_free_page_reclaim_observed=any(row["meaningful_free_page_reclaim"] for row in report["cycles"]),
                      interprets_actual_host_failure=False,changes_memory_policy=False,
                      scope="bounded standalone CPU allocation replay; not proof of live model-process reclaim or host-cache qualification")
    except BaseException as error:
        report.update(status="failed",error_type=type(error).__name__,error=str(error));raise
    finally:
        report["elapsed_seconds"]=time.perf_counter()-start;write_json(output/"manifest.json",report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("failed-host-proof","ledger","output"):parser.add_argument("--"+name,required=True)
    parser.add_argument("--max-seconds",type=int,default=90)
    result=run(**vars(parser.parse_args()));print(json.dumps({key:result[key] for key in ("status","meaningful_free_page_reclaim_observed","elapsed_seconds")},indent=2))


if __name__=="__main__":main()
