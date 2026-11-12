#!/usr/bin/env python3
"""Bounded FP32 GEMM workspace accounting; no model or posterior calls.

Built with PriorLabs-TabPFN. A private workspace-clear API is probed only when
present. This diagnostic never changes or upgrades a preserved host proof.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

SCHEMA="posteriorforge.cuda_workspace_diagnostic.v1"
FIXED_SHAPES=((128,512,1024),(1024,1024,1024))


def snapshot(torch,label,start):
    torch.cuda.synchronize()
    stats=torch.cuda.memory_stats()
    blocks=[]
    if hasattr(torch.cuda,"memory_snapshot"):
        for segment in torch.cuda.memory_snapshot():
            for block in segment.get("blocks",[]):
                if block.get("state")=="active_allocated":
                    blocks.append({name:block.get(name) for name in ("address","size","requested_size","state")})
    return {"label":label,"elapsed_seconds":time.perf_counter()-start,
            "allocated_bytes":torch.cuda.memory_allocated(),"reserved_bytes":torch.cuda.memory_reserved(),
            "active_bytes":stats.get("active_bytes.all.current"),"allocated_allocator_bytes":stats.get("allocated_bytes.all.current"),
            "active_blocks":blocks,"allocation_retries":stats.get("num_alloc_retries"),"oom_count":stats.get("num_ooms")}


def summarize_cycle(snapshots,clear_available,checksum,expected_checksum):
    by={row["label"]:row for row in snapshots}
    baseline=by["before_inputs"]["allocated_bytes"]
    deleted=by["after_delete_sync_gc"]["allocated_bytes"]
    empty=by["after_empty_cache"]["allocated_bytes"]
    cleared=by.get("after_clear_cublas_workspaces",{}).get("allocated_bytes")
    residual=deleted-baseline
    delta=None if cleared is None else empty-cleared
    reproduced=bool(clear_available and residual>0 and empty==deleted and cleared==baseline and delta==residual and checksum==expected_checksum)
    return {"baseline_allocated_bytes":baseline,"post_delete_residual_allocated_bytes":residual,
            "empty_cache_allocated_delta_bytes":deleted-empty,"workspace_clear_allocated_delta_bytes":delta,
            "after_workspace_clear_allocated_bytes":cleared,"clear_api_available":bool(clear_available),
            "matmul_checksum":checksum,"expected_checksum":expected_checksum,"matmul_checksum_exact":checksum==expected_checksum,
            "workspace_mechanism_reproduced":reproduced,"snapshots":snapshots}


def measure_cycle(torch,shape):
    import gc
    m,k,n=shape;start=time.perf_counter();rows=[snapshot(torch,"before_inputs",start)]
    left=torch.ones((m,k),device="cuda",dtype=torch.float32)
    right=torch.ones((k,n),device="cuda",dtype=torch.float32)
    rows.append(snapshot(torch,"inputs_live",start))
    output=torch.mm(left,right)
    if output.dtype!=torch.float32:raise RuntimeError("FP32 output required")
    rows.append(snapshot(torch,"matmul_output_live",start));checksum=float(output[0,0].item())
    del output,left,right
    torch.cuda.synchronize();gc.collect();torch.cuda.synchronize()
    rows.append(snapshot(torch,"after_delete_sync_gc",start))
    torch.cuda.empty_cache();rows.append(snapshot(torch,"after_empty_cache",start))
    clear=getattr(torch._C,"_cuda_clearCublasWorkspaces",None)
    if callable(clear):
        clear();torch.cuda.synchronize();gc.collect();rows.append(snapshot(torch,"after_clear_cublas_workspaces",start))
    return {"shape_m_k_n":list(shape),"seconds":time.perf_counter()-start,
            **summarize_cycle(rows,callable(clear),checksum,float(k))}


def analyze(cycles,failed_host_residual=None):
    reproduced=bool(len(cycles)==len(FIXED_SHAPES) and all(row["workspace_mechanism_reproduced"] for row in cycles))
    deltas=[row["workspace_clear_allocated_delta_bytes"] for row in cycles]
    stable=bool(reproduced and len(set(deltas))==1)
    match=None if failed_host_residual is None else bool(stable and deltas[0]==failed_host_residual)
    return {"residual_workspace_mechanism_reproduced":reproduced,"stable_exact_workspace_delta":stable,
            "workspace_clear_deltas_bytes":deltas,"failed_host_residual_allocated_bytes":failed_host_residual,
            "exactly_matches_preserved_host_residual":match,
            "interpretation":"fresh no-model GEMM residual attributable to clearable cuBLAS workspace" if reproduced else "workspace attribution not established",
            "preserved_host_proof_qualified":False,"authorizes_automatic_host_pass":False,
            "scope":"fresh-process allocator mechanism; matching failed-host bytes is supporting evidence, not a completed all21 host qualification"}


def run(*,ledger,output,failed_host_proof=None,max_seconds=60):
    from posteriorforge.artifacts import runtime_identity,sha256_file,write_json
    from posteriorforge.reference_grid import _require_budgeted_gpu
    if type(max_seconds) is not int or not 10<=max_seconds<=120:raise ValueError("bounded max-seconds10..120 required")
    budget=_require_budgeted_gpu(ledger)
    if budget["gpus"]!=1:raise RuntimeError("one reserved GPU required")
    output=Path(output);output.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    report={**runtime_identity(),"schema":SCHEMA,"status":"running","budget_binding":budget,
            "helper_source_sha256":sha256_file(__file__),"model_calls":0,"posterior_vectors_generated":0,
            "attribution":"Built with PriorLabs-TabPFN","cycles":[],"fixed_shapes_m_k_n":[list(shape) for shape in FIXED_SHAPES]}
    write_json(output/"manifest.json",report)
    try:
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError("one visible CUDA GPU required")
        if torch.get_float32_matmul_precision()!="highest" or torch.backends.cuda.matmul.allow_tf32:raise RuntimeError("unchanged FP32 highest/TF32False required")
        torch.cuda.init();torch.cuda.synchronize()
        cpu_rng=torch.get_rng_state().clone();cuda_rng=[state.clone() for state in torch.cuda.get_rng_state_all()]
        report.update(torch_version=torch.__version__,torch_cuda_version=torch.version.cuda,
            device_name=torch.cuda.get_device_name(0),device_total_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
            dtype="torch.float32",autocast=False,float32_matmul_precision=torch.get_float32_matmul_precision(),allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            clear_api_available=callable(getattr(torch._C,"_cuda_clearCublasWorkspaces",None)),
            environment={name:os.environ.get(name) for name in ("CUBLAS_WORKSPACE_CONFIG","PYTORCH_CUDA_ALLOC_CONF","OMP_NUM_THREADS","CUDA_VISIBLE_DEVICES")})
        failed_residual=None
        if failed_host_proof is not None:
            path=Path(failed_host_proof);old=json.loads(path.read_text())
            if old.get("status")!="failed" or old.get("error")!="unexplained live CUDA allocation with all21 stages on CPU":raise ValueError("preserved specific failed host proof required")
            residency=old.get("all21_cpu_residency",{})
            if residency.get("retained_stages")!=21 or any(stage["state"]["cuda_storage_bytes"] for stage in residency.get("stages",[])):raise ValueError("failed host proof stage-owned CUDA state differs")
            failed_residual=residency["global_cuda_allocated_bytes"]-old["initial_global_cuda_allocated_bytes"]
            report["failed_host_proof"]={"path":str(path.resolve()),"sha256":sha256_file(path),"status":old["status"],"error":old["error"],"residual_allocated_bytes":failed_residual}
        for shape in FIXED_SHAPES:
            if time.perf_counter()-start>=max_seconds:raise TimeoutError("bounded workspace probe deadline")
            report["cycles"].append(measure_cycle(torch,shape));write_json(output/"manifest.json",report)
        report.update(analyze(report["cycles"],failed_residual))
        report["rng_preserved"]=bool(torch.equal(cpu_rng,torch.get_rng_state()) and all(torch.equal(a,b) for a,b in zip(cuda_rng,torch.cuda.get_rng_state_all())))
        report["source_unchanged"]=report["helper_source_sha256"]==sha256_file(__file__)
        if not report["source_unchanged"] or not report["rng_preserved"]:raise RuntimeError("diagnostic source or RNG preservation failed")
        report["status"]="complete"
    except BaseException as error:
        report.update(status="failed",error_type=type(error).__name__,error=str(error));raise
    finally:
        report["elapsed_seconds"]=time.perf_counter()-start;write_json(output/"manifest.json",report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("ledger","output"):parser.add_argument("--"+name,required=True)
    parser.add_argument("--failed-host-proof");parser.add_argument("--max-seconds",type=int,default=60)
    result=run(**vars(parser.parse_args()))
    print(json.dumps({key:result[key] for key in ("status","residual_workspace_mechanism_reproduced","workspace_clear_deltas_bytes","exactly_matches_preserved_host_residual","elapsed_seconds")},indent=2))


if __name__=="__main__":main()
