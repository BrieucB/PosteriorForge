#!/usr/bin/env python3
"""Measured cuBLAS warm baseline followed by the unchanged all21 host proof.

Built with PriorLabs-TabPFN. No private workspace clear occurs here or inside the
measured proof. The prior failed proof is immutable and remains failed.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time

SCHEMA="posteriorforge.execution_host_cache_warmed_qualification.v1"
WORKSPACE_PROBE_SHA256="7cd658461c3dfbba28639e2738adf2b002575149fcc7ec4b7949e6d6a6702223"
WORKSPACE_EVIDENCE_SHA256="3781229c5930d654eb628b002950eba1a239d50dee2215c088d7048899bb4602"
FAILED_HOST_PROOF_SHA256="1c849f74784937c69e8d76f769eeb0f7785b1a6bbccfdb7f6e12903e10b6defc"
HOST_QUALIFIER_SHA256="0dc55766d5896e71bd2d9b5e0df6c9a7f09b459928317cac3c44557ab2a310f4"
HOST_MODULE_SHA256="effc86e17e33adb96fa8130c476a143b378a63c6d81bf71a31e3e2cd34e9bbde"
WORKSPACE_BYTES=8519680
OFFICIAL_SOURCES=[
 {"title":"PyTorch2.6 CUDA semantics: cuBLAS workspaces","url":"https://docs.pytorch.org/docs/2.6/notes/cuda.html#cublas-workspaces",
  "supports":"workspace persists per handle/stream until explicit clearing; default size formula"},
 {"title":"PyTorchv2.6.0 CublasHandlePool.cpp","url":"https://github.com/pytorch/pytorch/blob/v2.6.0/aten/src/ATen/cuda/CublasHandlePool.cpp",
  "supports":"default non-Hopper CUDA workspace is4096*1024*2+16*1024*8 bytes; clearing releases workspace map"}]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_frozen(filename,expected):
    path=Path(__file__).with_name(filename)
    if sha(path)!=expected:raise RuntimeError("frozen source changed: "+filename)
    spec=importlib.util.spec_from_file_location("_warmed_"+path.stem,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def authenticate_workspace_evidence(evidence_path,failed_host_path):
    if sha(evidence_path)!=WORKSPACE_EVIDENCE_SHA256:raise ValueError("exact completed workspace diagnostic SHA required")
    if sha(failed_host_path)!=FAILED_HOST_PROOF_SHA256:raise ValueError("preserved failed host proof SHA mismatch")
    evidence=json.loads(Path(evidence_path).read_text());failed=json.loads(Path(failed_host_path).read_text())
    required={"schema":"posteriorforge.cuda_workspace_diagnostic.v1","status":"complete","helper_source_sha256":WORKSPACE_PROBE_SHA256,
              "residual_workspace_mechanism_reproduced":True,"stable_exact_workspace_delta":True,
              "workspace_clear_deltas_bytes":[WORKSPACE_BYTES,WORKSPACE_BYTES],"exactly_matches_preserved_host_residual":True,
              "rng_preserved":True,"source_unchanged":True,"model_calls":0,"posterior_vectors_generated":0,
              "torch_version":"2.6.0+cu124","torch_cuda_version":"12.4","dtype":"torch.float32","allow_tf32":False}
    for key,value in required.items():
        if evidence.get(key)!=value:raise ValueError("workspace evidence contract mismatch: "+key)
    if evidence.get("failed_host_proof",{}).get("sha256")!=FAILED_HOST_PROOF_SHA256:raise ValueError("workspace evidence failed-proof binding mismatch")
    if failed.get("status")!="failed" or failed.get("error")!="unexplained live CUDA allocation with all21 stages on CPU":raise ValueError("original failed proof status must remain failed")
    residency=failed["all21_cpu_residency"]
    if residency["global_cuda_allocated_bytes"]-failed["initial_global_cuda_allocated_bytes"]!=WORKSPACE_BYTES:raise ValueError("preserved residual bytes mismatch")
    if residency["retained_stages"]!=21 or len(residency["stages"])!=21 or any(row["state"]["cuda_storage_bytes"] for row in residency["stages"]):
        raise ValueError("preserved all21 CPU-state evidence mismatch")
    return evidence


def warm_runtime(torch,snapshot):
    """Fresh-process deterministic FP32 GEMM; leave measured workspace live."""
    start=time.perf_counter();rows=[snapshot(torch,"fresh_runtime",start)]
    if rows[0]["allocated_bytes"]!=0:raise RuntimeError("fresh-process zero allocated tensor baseline required")
    cpu_rng=torch.get_rng_state().clone();cuda_rng=[state.clone() for state in torch.cuda.get_rng_state_all()]
    left=torch.ones((1024,1024),dtype=torch.float32,device="cuda")
    right=torch.ones((1024,1024),dtype=torch.float32,device="cuda")
    output=torch.mm(left,right);torch.cuda.synchronize();checksum=float(output[0,0].item())
    if output.dtype!=torch.float32 or checksum!=1024.:raise RuntimeError("deterministic FP32 warmup checksum failed")
    del output,left,right
    torch.cuda.synchronize();gc.collect();torch.cuda.synchronize();rows.append(snapshot(torch,"after_delete_sync_gc",start))
    torch.cuda.empty_cache();rows.append(snapshot(torch,"warmed_runtime_after_empty_cache",start))
    if rows[-2]["allocated_bytes"]!=WORKSPACE_BYTES or rows[-1]["allocated_bytes"]!=WORKSPACE_BYTES:
        raise RuntimeError("actual warmed baseline differs from the exactly authenticated workspace bytes")
    same=bool(torch.equal(cpu_rng,torch.get_rng_state()) and all(torch.equal(a,b) for a,b in zip(cuda_rng,torch.cuda.get_rng_state_all())))
    if not same:raise RuntimeError("deterministic warmup changed CPU/CUDA RNG")
    return {"shape_m_k_n":[1024,1024,1024],"dtype":"torch.float32","checksum":checksum,"rng_preserved":same,
            "seconds":time.perf_counter()-start,"snapshots":rows,"measured_live_workspace_bytes":rows[-1]["allocated_bytes"],
            "workspace_clear_called":False,"workspace_retained_through_proof":True,
            "baseline_policy":"exact measured authenticated FP32 runtime allocation; no arbitrary residual allowance"}


def validate_completed_proof(proof):
    required={"schema":"posteriorforge.execution_host_cache_qualification.v1","status":"host_screen_pass","qualified":True,
              "full21_host_residency_qualified":True,"full21_cache_only_qualified":False,
              "initial_global_cuda_allocated_bytes":WORKSPACE_BYTES,"all21_cpu_no_unexplained_cuda_allocations":True,
              "session_rng_preserved":True,"zero_refits_verified":True,"memory_contract_passed":True,"sources_unchanged":True}
    for key,value in required.items():
        if proof.get(key)!=value:raise RuntimeError("unchanged host qualifier did not pass: "+key)
    if proof.get("sources",{}).get("qualifier_source_sha256")!=HOST_QUALIFIER_SHA256 or proof.get("sources",{}).get("host_module_sha256")!=HOST_MODULE_SHA256:
        raise RuntimeError("inner host proof source custody mismatch")
    counters=proof.get("host_execution",{}).get("counters",{})
    for key,value in (("actual_prepare_miss_count",21),("cache_hit_count",42),("reload_count",42)):
        if counters.get(key)!=value:raise RuntimeError("full21 two warm scans counter mismatch: "+key)
    if proof.get("cyclic_full_scans",{}).get("evictions")!=0:raise RuntimeError("zero-eviction proof required")
    if proof.get("all21_cpu_residency",{}).get("global_cuda_allocated_bytes")!=WORKSPACE_BYTES:
        raise RuntimeError("all21 CPU residual must equal measured workspace exactly")
    if sorted(row["stage_index"] for row in proof.get("stage_results",[]) if row.get("passed"))!=list(range(21)):
        raise RuntimeError("all21 stage proofs required")
    if proof.get("compatibility_adapter",{}).get("original_alias_restored") is not True:raise RuntimeError("CDF alias exact restoration required")


def run(*,banks,prefix_run,checkpoint,ledger,qualification,workspace_evidence,failed_host_proof,output,max_seconds=2250):
    from posteriorforge.artifacts import runtime_identity,write_json
    from posteriorforge.reference_grid import _require_budgeted_gpu
    from posteriorforge import execution_host_cache as host
    if type(max_seconds) is not int or not 600<=max_seconds<=2250:raise ValueError("bounded max-seconds600..2250 required")
    budget=_require_budgeted_gpu(ledger)
    if budget["gpus"]!=1:raise RuntimeError("one reserved GPU required")
    output=Path(output);output.mkdir(parents=True,exist_ok=False);start=time.perf_counter();source_sha=sha(__file__)
    report={**runtime_identity(),"schema":SCHEMA,"status":"running","qualified":False,"full21_warmed_host_residency_qualified":False,
            "full21_cache_only_qualified":False,"posterior_vectors_generated":0,"budget_binding":budget,
            "wrapper_source_sha256":source_sha,"frozen_host_qualifier_sha256":HOST_QUALIFIER_SHA256,"host_module_sha256":HOST_MODULE_SHA256,
            "workspace_evidence_sha256":WORKSPACE_EVIDENCE_SHA256,"preserved_failed_host_proof_sha256":FAILED_HOST_PROOF_SHA256,
            "official_sources":OFFICIAL_SOURCES,"attribution":"Built with PriorLabs-TabPFN",
            "scope":"new warmed-runtime all21 host residency proof; prior failed proof remains immutable; no posterior accuracy claim",
            "proof_manifest":"proof/manifest.json"}
    write_json(output/"manifest.json",report)
    try:
        evidence=authenticate_workspace_evidence(workspace_evidence,failed_host_proof)
        workspace=load_frozen("diagnose_cuda_workspace.py",WORKSPACE_PROBE_SHA256)
        qualifier=load_frozen("qualify_execution_host_cache.py",HOST_QUALIFIER_SHA256)
        if host.execution_source_sha256()!=HOST_MODULE_SHA256:raise RuntimeError("frozen host module changed")
        host.require_frozen_sources()
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError("one visible CUDA GPU required")
        if str(torch.__version__)!=evidence["torch_version"] or torch.version.cuda!=evidence["torch_cuda_version"] or torch.cuda.get_device_name(0)!=evidence["device_name"]:
            raise RuntimeError("runtime Torch/CUDA/GPU identity differs from workspace evidence")
        if torch.get_float32_matmul_precision()!="highest" or torch.backends.cuda.matmul.allow_tf32:raise RuntimeError("unchanged FP32 highest/TF32False required")
        for name in ("CUBLAS_WORKSPACE_CONFIG","PYTORCH_CUDA_ALLOC_CONF"):
            if os.environ.get(name)!=evidence["environment"].get(name):raise RuntimeError("workspace allocator environment mismatch: "+name)
        torch.cuda.init();torch.cuda.synchronize()
        report["warmup"]=warm_runtime(torch,workspace.snapshot);report["status"]="warmed_baseline_verified"
        write_json(output/"warmup.json",report["warmup"]);write_json(output/"manifest.json",report)
        remaining=int(max_seconds-(time.perf_counter()-start))
        if remaining<300:raise TimeoutError("insufficient bounded time for unchanged host proof")
        inner=qualifier.qualify(banks=banks,prefix_run=prefix_run,checkpoint=checkpoint,ledger=ledger,qualification=qualification,
                                output=output/"proof",max_seconds=remaining)
        validate_completed_proof(inner)
        report["proof_manifest_sha256"]=sha(output/"proof"/"manifest.json")
        unchanged=(sha(__file__)==source_sha and sha(Path(__file__).with_name("qualify_execution_host_cache.py"))==HOST_QUALIFIER_SHA256
            and sha(Path(__file__).with_name("diagnose_cuda_workspace.py"))==WORKSPACE_PROBE_SHA256 and host.execution_source_sha256()==HOST_MODULE_SHA256
            and sha(workspace_evidence)==WORKSPACE_EVIDENCE_SHA256 and sha(failed_host_proof)==FAILED_HOST_PROOF_SHA256)
        if not unchanged:raise RuntimeError("warmed proof source/evidence custody changed")
        report.update(status="warmed_host_screen_pass",qualified=True,full21_warmed_host_residency_qualified=True,
                      sources_and_evidence_unchanged=True,preserved_failed_host_proof_status="failed",
                      counters=inner["host_execution"]["counters"],all21_cpu_residual_bytes=inner["all21_cpu_residency"]["global_cuda_allocated_bytes"])
    except BaseException as error:
        report.update(status="failed",qualified=False,full21_warmed_host_residency_qualified=False,error_type=type(error).__name__,error=str(error))
        if (output/"proof"/"manifest.json").exists():report["partial_proof_manifest_sha256"]=sha(output/"proof"/"manifest.json")
        raise
    finally:
        report["elapsed_seconds"]=time.perf_counter()-start;write_json(output/"manifest.json",report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("banks","prefix-run","checkpoint","ledger","qualification","workspace-evidence","failed-host-proof","output"):
        parser.add_argument("--"+name,required=True)
    parser.add_argument("--max-seconds",type=int,default=2250)
    result=run(**vars(parser.parse_args()));print(json.dumps({key:result[key] for key in ("status","qualified","elapsed_seconds")},indent=2))


if __name__=="__main__":main()
