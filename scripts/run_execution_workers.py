#!/usr/bin/env python3
"""Run independent full-posterior workers in one already registered allocation.

Built with PriorLabs-TabPFN. This script never submits jobs. Only the aggregate
50000 accepted complete vectors is an E2E result; every child is a worker.
"""
from __future__ import annotations
import time
_PROGRAM_START=time.perf_counter()
import argparse
from importlib import import_module
import inspect
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

from posteriorforge.artifacts import runtime_identity,sha256_file,write_json
from posteriorforge.banks import derive_seed

TOTAL_COUNT=50000
CONTRACT_SCHEMA="posteriorforge.execution_worker_contract.v1"
AGGREGATE_SCHEMA="posteriorforge.inference.execution.aggregate.v1"
STREAM_ROLES={"head_proposal":"execution-head-proposal","head_acceptance":"execution-head-acceptance",
              "component_uniform":"execution-component-uniform"}


def partition_counts(workers,total=TOTAL_COUNT):
    if type(workers) is not int or workers not in (1,2,4) or type(total) is not int or total!=TOTAL_COUNT:
        raise ValueError("exactly50000 complete vectors across1,2or4workers required")
    return [total//workers]*workers


def worker_seed_streams(context_seed,case,repetition,rank,world_size):
    partition_counts(world_size)
    if type(rank) is not int or not 0<=rank<world_size:raise ValueError("invalid worker rank")
    return {name:derive_seed(context_seed,role,case=case,rank=rank,repetition=repetition)
            for name,role in STREAM_ROLES.items()}


def visible_device_tokens(value,workers):
    partition_counts(workers)
    tokens=[token.strip() for token in value.split(",")] if value else []
    if len(tokens)!=workers or any(not token or token=="NoDevFiles" for token in tokens) or len(set(tokens))!=workers:
        raise ValueError("parent must expose exactly one distinct allocated CUDA device per worker")
    return tokens


def worker_thread_plan(workers,allocated_cpus=None,requested=None):
    partition_counts(workers)
    allocated=16*workers if allocated_cpus in (None,"") else int(allocated_cpus)
    if allocated<workers:raise ValueError("allocation needs at least one CPU per independent worker")
    maximum=allocated//workers
    if requested is not None and (type(requested) is not int or not 1<=requested<=maximum):
        raise ValueError("per-worker threads must be positive and fit allocated CPUs")
    return (maximum if requested is None else requested),allocated


def build_contract(*,rank,workers,output,aggregate_output,budget,context_seed,case,repetition,
                   runner_source_sha256,visible_device_token):
    count=partition_counts(workers)[rank]
    return {"schema":CONTRACT_SCHEMA,"benchmark_total_count":TOTAL_COUNT,"world_size":workers,
        "rank":rank,"worker_count":count,"parent_job_id":str(budget["job_id"]),
        "budget_reservation_id":budget["reservation_id"],"context_seed":context_seed,"case":case,
        "repetition":repetition,"seed_streams":worker_seed_streams(context_seed,case,repetition,rank,workers),
        "expected_output":str(Path(output).resolve()),"aggregate_output":str(Path(aggregate_output).resolve()),
        "visible_device_token":visible_device_token,"runner_source_sha256":runner_source_sha256,
        "orchestrator_source":str(Path(__file__).resolve()),"orchestrator_source_sha256":sha256_file(__file__),
        "individual_worker_is_end_to_end":False}


def _stream_seed_map(manifest):
    streams=manifest["sampling_policy"]["streams"]
    return {name:(value["seed"] if isinstance(value,dict) else value) for name,value in streams.items()}


def scientific_bindings(banks,context_seed,case):
    from posteriorforge.banks import load_context,load_heldout
    from posteriorforge.science.analytical_benchmark_v2 import build_problem,array_descriptor
    from posteriorforge.inference import CHECKPOINT_SHA256,validate_query
    from posteriorforge.inference_cdf import guard_baseline_source
    context=load_context(banks,context_seed,4096);observed=load_heldout(banks,case).obs
    return {"scientific_fingerprint":build_problem().scientific_fingerprint,
        "parameter_order":list(build_problem().parameter_order),"context_seed":context_seed,"case":case,
        "context_size":4096,"context_theta":array_descriptor(context.theta),"context_obs":array_descriptor(context.obs),
        "query":array_descriptor(validate_query(observed)),"checkpoint_sha256":CHECKPOINT_SHA256,
        "baseline_execution_source":guard_baseline_source()}


def validate_worker_payloads(contracts,contract_paths,*,expected_bindings,driver_schema,qualification_sha256):
    import numpy as np
    from posteriorforge.inference import support_mask
    from posteriorforge.science.analytical_benchmark_v2 import array_descriptor
    from posteriorforge.compat_cdf import adapter_source_sha256,PINNED_SDK_SHA256
    from posteriorforge import execution_cache
    arrays=[];records=[];sample_hashes=set();all_streams=[]
    for contract,path in zip(contracts,contract_paths):
        root=Path(contract["expected_output"]);manifest_path=root/"manifest.json"
        manifest=json.loads(manifest_path.read_text())
        expected={"schema":driver_schema,"status":"complete","run_purpose":"performance_worker",
            "rank":contract["rank"],"world_size":contract["world_size"],
            "benchmark_total_count":TOTAL_COUNT,"requested_count":contract["worker_count"],
            "worker_contract_sha256":sha256_file(path),"end_to_end":False,
            "completed_stage_indices":list(range(21)),"runner_source_sha256":contract["runner_source_sha256"],
            "sources_unchanged":True,"qualification_full21":True}
        for key,value in {**expected_bindings,**expected}.items():
            if manifest.get(key)!=value:raise ValueError("worker artifact binding mismatch: "+key)
        if _stream_seed_map(manifest)!=contract["seed_streams"]:raise ValueError("worker sampling seeds differ from contract")
        all_streams.append(contract["seed_streams"])
        engine=manifest.get("execution_engine",{})
        expected_engine={**execution_cache.execution_metadata(),
            "sampling_rng_policy":"independent_cpu_sampling_streams_model_rng_isolated",
            "query_batch_size":contract.get("query_batch_size",1024),
            "posterior_count":contract["worker_count"]}
        if any(engine.get(key)!=value for key,value in expected_engine.items()):
            raise ValueError("worker base cache engine identity mismatch")
        if manifest.get("qualification_manifest_sha256")!=qualification_sha256:
            raise ValueError("worker qualification artifact mismatch")
        adapter=manifest["compatibility_adapter"]
        if (adapter.get("adapter_sha256")!=adapter_source_sha256()
            or adapter.get("variant")!="cdf_arithmetic_v1"
            or adapter.get("scope")!="finite_ensemble_cdf_parentheses_only"
            or adapter.get("source_guard_passed") is not True
            or adapter.get("sdk_source_sha256")!=dict(PINNED_SDK_SHA256)
            or adapter.get("sdk_source_sha256_after")!=dict(PINNED_SDK_SHA256)
            or adapter.get("active_translator_code_sha256")!=adapter.get("original_translator_code_sha256")
            or adapter.get("original_alias_restored") is not True
            or adapter.get("sdk_sources_unchanged") is not True
            or adapter.get("sdk_aliases_unchanged") is not True
            or adapter.get("original_alias_identity_verified") is not True
            or adapter.get("translator_code_unchanged") is not True
            or adapter.get("patched_call_count",0)<=0
            or adapter.get("executed_corrected_cdf_code_sha256")!=adapter.get("corrected_cdf_code_sha256")
            or adapter.get("corrected_cdf_code_sha256")==adapter.get("original_cdf_code_sha256")):
            raise ValueError("worker CDF execution/restoration not authenticated")
        payload=root/"samples.npz";digest=sha256_file(payload)
        if digest!=manifest.get("samples_sha256") or digest in sample_hashes:
            raise ValueError("worker samples hash mismatch or duplicated rank payload")
        sample_hashes.add(digest)
        with np.load(payload,allow_pickle=False) as value:samples=value["samples"].copy()
        if samples.shape!=(contract["worker_count"],21) or not support_mask(samples).all():
            raise ValueError("worker returned incomplete, nonfinite or unsupported vectors")
        if manifest.get("samples")!=array_descriptor(samples):raise ValueError("worker sample descriptor mismatch")
        arrays.append(samples.astype(np.float64,copy=False))
        records.append({"rank":contract["rank"],"manifest":str(manifest_path),
            "manifest_sha256":sha256_file(manifest_path),"samples_sha256":digest,
            "accepted_count":len(samples),"seed_streams":contract["seed_streams"]})
    if len(contracts)!=len(contract_paths) or len(arrays) not in (1,2,4):
        raise ValueError("worker artifact set incomplete")
    flat=[seed for streams in all_streams for seed in streams.values()]
    if len(set(flat))!=len(flat):raise ValueError("rank or role sampling seed streams collide")
    for name in STREAM_ROLES:
        if len({streams[name] for streams in all_streams})!=len(all_streams):
            raise ValueError("rank sampling seed streams collide")
    combined=np.concatenate(arrays,axis=0)
    if combined.shape!=(TOTAL_COUNT,21) or not support_mask(combined).all():
        raise ValueError("aggregate must contain50000 complete supported physical vectors")
    return combined,records


def terminate_workers(processes,grace_seconds=3):
    """Stop live process groups, preserving every diagnostic/output file."""
    for process in processes:
        if process.poll() is None:
            try:os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError:pass
    deadline=time.monotonic()+grace_seconds
    while any(process.poll() is None for process in processes) and time.monotonic()<deadline:
        time.sleep(.05)
    for process in processes:
        if process.poll() is None:
            try:os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:pass
    for process in processes:process.wait()


def execute_workers(commands,environments,logs,*,timeout_seconds,on_progress=None,cwd=None):
    if not commands or len(commands)!=len(environments) or len(commands)!=len(logs):
        raise ValueError("complete command/environment/log set required")
    processes=[];handles=[];started=time.monotonic()
    try:
        for command,environment,(stdout_path,stderr_path) in zip(commands,environments,logs):
            stdout=open(stdout_path,"xb");handles.append(stdout)
            stderr=open(stderr_path,"xb");handles.append(stderr)
            processes.append(subprocess.Popen(command,env=environment,stdout=stdout,stderr=stderr,
                                               cwd=cwd,start_new_session=True))
        last=None
        while True:
            codes=[process.poll() for process in processes]
            if codes!=last and on_progress is not None:on_progress(codes)
            last=codes
            if any(code is not None and code!=0 for code in codes):
                raise RuntimeError("at least one posterior worker failed; no partial aggregate")
            if all(code==0 for code in codes):return codes
            if time.monotonic()-started>=timeout_seconds:raise TimeoutError("aggregate worker deadline exhausted")
            time.sleep(.1)
    except BaseException:
        terminate_workers(processes)
        raise
    finally:
        for handle in handles:handle.close()


def run(*,banks,output,checkpoint,ledger,reference,reference_grid,qualification,workers=1,
        context_seed=20261111,case=0,repetition=0,query_batch_size=1024,
        mode="component_mass",timeout_seconds=3000,threads_per_worker=None):
    started=_PROGRAM_START if __name__=="__main__" else time.perf_counter()
    from posteriorforge.reference_grid import _require_budgeted_gpu
    from posteriorforge.science.analytical_benchmark_v2 import array_descriptor
    import numpy as np
    counts=partition_counts(workers)
    budget=_require_budgeted_gpu(ledger)
    if budget["gpus"]!=workers:raise RuntimeError("registered allocation GPU count must equal worker count")
    tokens=visible_device_tokens(os.getenv("CUDA_VISIBLE_DEVICES"),workers)
    effective_threads,allocated_cpus=worker_thread_plan(workers,os.getenv("SLURM_CPUS_PER_TASK"),threads_per_worker)
    if type(timeout_seconds) is not int or timeout_seconds<1:raise ValueError("positive aggregate deadline required")
    driver=import_module("posteriorforge.performance_runner")
    driver_source=sha256_file(inspect.getfile(driver))
    driver_schema=getattr(driver,"SCHEMA","posteriorforge.inference.execution.v1")
    root=Path(output).resolve();root.mkdir(parents=True,exist_ok=False)
    result={**runtime_identity(),"schema":AGGREGATE_SCHEMA,"status":"running",
        "variant":"execution_workers_v1","run_purpose":"performance_e2e_aggregate",
        "attribution":"Built with PriorLabs-TabPFN","requested_count":TOTAL_COUNT,"world_size":workers,
        "rank":None,"end_to_end":True,"budget":budget,
        "threads_per_worker":effective_threads,"allocation_cpu_count":allocated_cpus,"orchestrator_source_sha256":sha256_file(__file__),
        "runner_source_sha256":driver_source,"worker_counts":counts,"worker_return_codes":[None]*workers,
        "accuracy_acceptance":"evaluated separately after genuine performance run",
        "performance_first_authorized":True,"absolute_accuracy_gate_required_for_performance":False,
        "known_physical_control_status":"model_failure",
        "qualification_manifest_sha256":sha256_file(qualification),"mode":mode}
    write_json(root/"manifest.json",result)
    try:
        from posteriorforge.inference_cdf import reference_binding
        bindings=scientific_bindings(banks,context_seed,case)
        from posteriorforge import execution_cache,execution_mixture
        worker_engine={**execution_cache.execution_metadata(),
            "sampling_rng_policy":"independent_cpu_sampling_streams_model_rng_isolated",
            "query_batch_size":query_batch_size,"posterior_count":TOTAL_COUNT}
        bindings.update(mode=mode,variant="execution_cache_v1" if mode=="resident_full" else "execution_mixture_v1",
            reference_binding=reference_binding(banks=banks,reference=reference,reference_grid=reference_grid,case=case))
        result.update({key:value for key,value in bindings.items() if key!="variant"})
        result["worker_execution_variant"]=bindings["variant"]
        result["execution_engine"]=worker_engine
        contracts=[];contract_paths=[];commands=[];environments=[];logs=[]
        for rank in range(workers):
            worker_output=root/f"worker-{rank}"
            contract=build_contract(rank=rank,workers=workers,output=worker_output,aggregate_output=root,
                budget=budget,context_seed=context_seed,case=case,repetition=repetition,
                runner_source_sha256=driver_source,visible_device_token=tokens[rank])
            if contract["seed_streams"]!=driver.worker_stream_seeds(context_seed,case,repetition,rank,workers):
                raise RuntimeError("orchestrator and driver seed contracts disagree")
            contract.update(mode=mode,query_batch_size=query_batch_size,omp_threads_per_worker=effective_threads)
            path=root/"contracts"/f"worker-{rank}.json";write_json(path,contract)
            contracts.append(contract);contract_paths.append(path)
            command=[sys.executable,"-m","posteriorforge.performance_runner","--banks",str(Path(banks).resolve()),
                "--output",str(worker_output),"--checkpoint",str(Path(checkpoint).resolve()),"--ledger",str(Path(ledger).resolve()),
                "--reference",str(Path(reference).resolve()),"--reference-grid",str(Path(reference_grid).resolve()),
                "--qualification",str(Path(qualification).resolve()),"--context-seed",str(context_seed),
                "--case",str(case),"--repetition",str(repetition),"--query-batch-size",str(query_batch_size),
                "--mode",mode,"--worker-contract",str(path)]
            commands.append(command)
            environment=os.environ.copy();environment["CUDA_VISIBLE_DEVICES"]=tokens[rank]
            environment["PF_EXECUTION_WORKER_RANK"]=str(rank);environment["PF_EXECUTION_WORKER_WORLD_SIZE"]=str(workers)
            environment["OMP_NUM_THREADS"]=str(effective_threads)
            environments.append(environment);logs.append((root/f"worker-{rank}.stdout.log",root/f"worker-{rank}.stderr.log"))
        result["worker_contracts"]=[{"path":str(path),"sha256":sha256_file(path)} for path in contract_paths]
        result["worker_commands"]=commands;result["visible_device_tokens"]=tokens
        result["rank_seed_streams"]=[contract["seed_streams"] for contract in contracts]
        observed_env=("CUDA_VISIBLE_DEVICES","OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS","OMP_PROC_BIND","OMP_PLACES")
        result["worker_environment"]=[{key:environment.get(key) for key in observed_env} for environment in environments]
        write_json(root/"manifest.json",result)
        def progress(codes):
            result["worker_return_codes"]=codes;write_json(root/"manifest.json",result)
        execute_workers(commands,environments,logs,timeout_seconds=timeout_seconds,
                        on_progress=progress,cwd=Path(__file__).resolve().parents[1])
        values,records=validate_worker_payloads(contracts,contract_paths,expected_bindings=bindings,
             driver_schema=driver_schema,qualification_sha256=result["qualification_manifest_sha256"])
        if sha256_file(inspect.getfile(driver))!=driver_source or sha256_file(__file__)!=result["orchestrator_source_sha256"] or sha256_file(qualification)!=result["qualification_manifest_sha256"]:
            raise RuntimeError("runner/orchestrator/qualification source changed during aggregate execution")
        pending=root/"aggregate-samples.pending.npz"
        with pending.open("xb") as stream:np.savez(stream,samples=values)
        os.replace(pending,root/"samples.npz")
        descriptor=array_descriptor(values);payload_sha256=sha256_file(root/"samples.npz");elapsed=time.perf_counter()-started
        result.update(status="complete",samples=descriptor,samples_file="samples.npz",
            samples_sha256=payload_sha256,workers=records,aggregate_published=True,
            timings={"end_to_end_seconds":elapsed,
                "accepted_samples_per_second":TOTAL_COUNT/elapsed,
                "includes_budget_binding_context_auth_child_startup_all_workers_aggregation_hashing_and_payload_write":True},
            returned_count=len(values))
        write_json(root/"manifest.json",result)
        return result
    except BaseException as error:
        result.update(status="failed",error_type=type(error).__name__,error=str(error),
                      elapsed_seconds=time.perf_counter()-started,aggregate_published=False)
        write_json(root/"manifest.json",result)
        raise


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("banks","output","checkpoint","ledger","reference","reference-grid","qualification"):
        parser.add_argument("--"+name,required=True)
    parser.add_argument("--workers",type=int,choices=[1,2,4],default=1)
    parser.add_argument("--context-seed",type=int,default=20261111);parser.add_argument("--case",type=int,default=0)
    parser.add_argument("--repetition",type=int,default=0);parser.add_argument("--query-batch-size",type=int,default=1024)
    parser.add_argument("--mode",choices=["resident_full","component_mass"],default="component_mass")
    parser.add_argument("--timeout-seconds",type=int,default=3000);parser.add_argument("--threads-per-worker",type=int)
    print(json.dumps(run(**vars(parser.parse_args())),indent=2))
