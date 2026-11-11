#!/usr/bin/env python3
"""Read-only baseline timing evidence and reversible failure-diagnostic profiling.

Built with PriorLabs-TabPFN. This never changes or reclassifies the scientific gate.
Accelerator imports happen only in the explicit, budgeted ``profile`` operation.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from array import array
from contextlib import contextmanager, nullcontext
import csv
import gzip
import hashlib
import json
from pathlib import Path
import os
import resource
import shutil
import statistics
import time

from posteriorforge.artifacts import runtime_identity, sha256_file, write_json

SCHEMA = "posteriorforge.failure_runtime_diagnostic.v1"
ATTRIBUTION = "Built with PriorLabs-TabPFN"
BASELINE_COMMIT = "89404ac9e43bf9fb72d3e51a7113dbd207871221"


def descriptive(values):
    values = list(values)
    if not values:
        return None
    return {"minimum": min(values), "median_across_distinct_cases_and_banks": statistics.median(values),
            "maximum": max(values)}


def interval_union(intervals):
    """Duration of a union, without double-counting concurrent GPU intervals."""
    intervals = sorted((float(a), float(b)) for a, b in intervals if b >= a)
    if not intervals:
        return 0.
    start, stop = intervals[0]
    total = 0.
    for left, right in intervals[1:]:
        if left > stop:
            total += stop - start
            start, stop = left, right
        else:
            stop = max(stop, right)
    return total + stop - start


def summarize_saved(runs, ledger, output):
    """Summarize actual completed 50k requests, not diagnostic extrapolations."""
    root, output = Path(runs), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifests = [(path, json.loads(path.read_text())) for path in sorted(root.glob("*/manifest.json"))]
    if len(manifests) != 24:
        raise ValueError("exactly 24 baseline manifests required")
    ledger_value = json.loads(Path(ledger).read_text())
    allocations = {entry["job_id"]: entry for entry in ledger_value["reservations"]}
    rows, inputs = [], []
    for path, value in manifests:
        if (value["status"] != "complete" or value["variant"] != "upstream_unmodified"
                or value["requested_count"] != 50000 or value["context_size"] != 4096):
            raise ValueError("completed original 50000-vector upstream workload required")
        t, r, m = value["timings"], value["rejection"], value["memory"]
        allocation = allocations[value["slurm_job_id"]]
        rows.append({"context_seed": value["context_seed"], "case": value["case"],
            "job_id": value["slurm_job_id"], **t, **r, **m,
            "allocated_gpus": allocation["accounted_gpus"],
            "slurm_elapsed_seconds": allocation["accounted_elapsed_seconds"],
            "candidate_acceptance_rate": r["accepted_before_final_trim"] / r["proposed"],
            "valid_candidates_trimmed": r["accepted_before_final_trim"] - r["returned"],
            "invalid_candidates_rejected": r["proposed"] - r["accepted_before_final_trim"],
            "amortized_seconds_per_returned_vector": t["posterior_request_seconds"] / r["returned"],
            "autoregressive_fit_predict_stages": 21 * r["support_calls"]})
        inputs.append({"manifest_file": str(path.relative_to(root)), "manifest_sha256": sha256_file(path)})
    numeric = ("posterior_request_seconds", "total_seconds", "constructor_and_import_seconds",
        "append_context_seconds", "accepted_samples_per_second", "slurm_elapsed_seconds",
        "support_calls", "proposed", "accepted_before_final_trim", "valid_candidates_trimmed", "invalid_candidates_rejected",
        "candidate_acceptance_rate", "amortized_seconds_per_returned_vector",
        "peak_allocated_bytes", "peak_reserved_bytes", "autoregressive_fit_predict_stages")
    result = {"schema": SCHEMA + ".saved_baseline", "attribution": ATTRIBUTION,
        "baseline_commit": BASELINE_COMMIT, "run_count": len(rows),
        "workload": "50000 complete accepted vectors, 4096 context, 21 stages, max query 10000, FP32, 8 ensembles",
        "summary": {name: descriptive(row[name] for row in rows) for name in numeric},
        "support_call_counts": dict(Counter(row["support_calls"] for row in rows)),
        "inputs": inputs, "ledger_sha256": sha256_file(ledger),
        "interpretation": {"median_is_not_three_repeat_benchmark": True,
            "request_includes_lazy_checkpoint_load_and_all_stage_preparation": True,
            "append_context_is_only_bank_attachment_not_stage_preprocessing": True,
            "amortized_vector_time_is_not_isolated_draw_latency": True,
            "isolated_draw_latency_measured_in_saved_baseline": False,
            "fit_predict_call_count_is_source_plus_support_counter_not_a_trace": True,
            "candidate_rejection_alone_does_not_establish_runtime_bottleneck": True}}
    write_json(output / "runtime-baseline-summary.json", result)
    with (output / "runtime-baseline-runs.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), layout="constrained")
    for seed in (20261111, 20261112, 20261113):
        selected = [row for row in rows if row["context_seed"] == seed]
        axes[0].plot([row["case"] for row in selected],
                     [row["posterior_request_seconds"]/60 for row in selected], marker="o", label=str(seed))
        axes[1].plot([row["case"] for row in selected],
                     [100*row["candidate_acceptance_rate"] for row in selected], marker="o", label=str(seed))
    axes[0].set(xlabel="Shared held-out case", ylabel="Complete 50k request (minutes)", title="Measured original requests")
    axes[1].set(xlabel="Shared held-out case", ylabel="Accepted candidates before trim (%)", title="Rejection work, not a bottleneck verdict")
    for ax in axes:
        ax.set_xticks(range(8)); ax.grid(alpha=.2); ax.legend(title="Context seed", fontsize=8)
    fig.suptitle(ATTRIBUTION + "\nFrozen baseline evidence; distinct cases/banks, not timing repeats", fontsize=12)
    fig.savefig(output / "runtime-baseline-timing.png", dpi=160, bbox_inches="tight"); plt.close(fig)
    return result


class CallRecorder:
    """Nested host durations and profiler annotations; no inserted synchronization."""
    def __init__(self, record_function=None):
        self.record_function = record_function or (lambda _: nullcontext())
        self.events = []
        self.stack = []
        self.request = "none"
        self.proposal = 0
        self.stage = -1
        self.features = 0
        self.rows = 0

    @contextmanager
    def measure(self, operation):
        event = {"operation": operation, "request": self.request, "proposal": self.proposal,
                 "stage_index": self.stage, "features": self.features, "query_rows": self.rows,
                 "parent_operation": self.stack[-1] if self.stack else None}
        label = f"pfdiagnostic/{self.request}/p{self.proposal}/s{self.stage}/{operation}"
        self.stack.append(operation)
        started = time.perf_counter()
        try:
            with self.record_function(label):
                yield
        except BaseException as error:
            event["error"] = type(error).__name__
            raise
        finally:
            event["host_wall_seconds"] = time.perf_counter() - started
            self.events.append(event)
            self.stack.pop()

    @contextmanager
    def instrument(self, posterior, cuda, observation_dimension=33):
        replacements = []
        def replace(owner, name, wrapper):
            original = getattr(owner, name)
            replacements.append((owner, name, original, name in getattr(owner, "__dict__", {})))
            setattr(owner, name, wrapper(original))
        def proposal(original):
            def wrapped(*args, **kwargs):
                self.proposal += 1
                self.stage = -1
                self.rows = int(kwargs.get("sampling_batch_size", args[0] if args else 0))
                with self.measure("proposal"):
                    return original(*args, **kwargs)
            return wrapped
        def fit(original):
            def wrapped(*args, **kwargs):
                x = args[0] if args else kwargs["X"]
                self.features = int(x.shape[1]); self.stage = self.features - observation_dimension
                with self.measure("fit"):
                    return original(*args, **kwargs)
            return wrapped
        def simple(operation):
            def maker(original):
                def wrapped(*args, **kwargs):
                    with self.measure(operation):
                        return original(*args, **kwargs)
                return wrapped
            return maker
        replace(posterior, "_sample", proposal)
        replace(posterior._model, "fit", fit)
        replace(posterior._model, "predict", simple("predict"))
        replace(posterior, "_within_support", simple("support"))
        replace(cuda, "empty_cache", simple("empty_cache"))
        replace(cuda, "ipc_collect", simple("ipc_collect"))
        try:
            yield self
        finally:
            for owner, name, original, owned in reversed(replacements):
                if owned:
                    setattr(owner, name, original)
                else:
                    delattr(owner, name)


def iter_trace_events(path, chunk_bytes=1024*1024):
    decoder=json.JSONDecoder()
    with Path(path).open(encoding="utf-8") as stream:
        buffer=""
        while True:
            chunk=stream.read(chunk_bytes)
            if not chunk: raise ValueError("traceEvents array not found")
            buffer+=chunk
            key=buffer.find(chr(34)+"traceEvents"+chr(34))
            start=buffer.find("[",key+13) if key>=0 else -1
            if start>=0:
                buffer=buffer[start+1:]; position=0; break
            if len(buffer)>2*chunk_bytes: raise ValueError("oversized trace header")
        while True:
            while position<len(buffer) and (buffer[position].isspace() or buffer[position]==","):
                position+=1
            if position<len(buffer) and buffer[position]=="]":return
            try:
                event, end=decoder.raw_decode(buffer,position)
            except json.JSONDecodeError:
                chunk=stream.read(chunk_bytes)
                if not chunk:raise ValueError("truncated or invalid trace event array")
                buffer=buffer[position:]+chunk;position=0;continue
            position=end
            if not isinstance(event,dict):raise ValueError("non-object trace event")
            yield event
            if position>=chunk_bytes:
                buffer=buffer[position:];position=0

def union_seconds(flat):
    pairs=sorted(zip(flat[::2],flat[1::2]))
    if not pairs:return 0.
    left,right=pairs[0];total=0.
    for start,stop in pairs[1:]:
        if start>right:total+=right-left;left,right=start,stop
        else:right=max(right,stop)
    return (total+right-left)/1e6

def summarize_trace(path):
    start_clock=time.perf_counter();path=Path(path)
    groups={key:defaultdict(lambda:{"count":0,"summed_duration_seconds":0.,"bytes":None,"events_with_byte_count":0}) for key in ("kernel","copy","memset","runtime","sync")}
    counts=defaultdict(int); sums=defaultdict(float);request=[];labels=[];all_events=0
    for event in iter_trace_events(path):
        all_events+=1
        if event.get("ph")!="X" or not isinstance(event.get("dur"),(float,int)):continue
        name=event.get("name","");cat=event.get("cat","").lower()
        if name=="pfdiagnostic/profile/request":request.append((event["ts"],event["ts"]+event["dur"]))
        if name.startswith("pfdiagnostic/profile/"):
            labels.append({"name":name,"start_us":event["ts"],"duration_seconds":event["dur"]/1e6})
        tags=[]
        if "kernel" in cat:tags.append("kernel")
        if "memcpy" in cat:tags.append("copy")
        if "memset" in cat:tags.append("memset")
        if "cuda_runtime" in cat or "cuda_driver" in cat:
            tags.append("runtime")
            if "synchroniz" in name.lower():tags.append("sync")
        for tag in tags:
            counts[tag]+=1;sums[tag]+=event["dur"]/1e6
            row=groups[tag][name];row["count"]+=1;row["summed_duration_seconds"]+=event["dur"]/1e6
            for key in ("Bytes","bytes","size"):
                value=event.get("args",{}).get(key)
                if isinstance(value,(int,float)):
                    row["bytes"]=(row["bytes"] or 0)+value;row["events_with_byte_count"]+=1;break
    if len(request)!=1:raise ValueError("exactly one annotated profile request required")
    start,stop=request[0];kernels=array("d");active=array("d");copies=array("d");sync=array("d")
    for event in iter_trace_events(path):
        if event.get("ph")!="X" or not isinstance(event.get("dur"),(float,int)):continue
        cat=event.get("cat","").lower();name=event.get("name","")
        is_kernel="kernel" in cat;is_copy="memcpy" in cat;is_memset="memset" in cat
        is_sync=("cuda_runtime" in cat or "cuda_driver" in cat) and "synchroniz" in name.lower()
        if not (is_kernel or is_copy or is_memset or is_sync):continue
        left=max(start,event["ts"]);right=min(stop,event["ts"]+event["dur"])
        if left>=right:continue
        if is_kernel:kernels.extend((left,right))
        if is_kernel or is_copy or is_memset:active.extend((left,right))
        if is_copy:copies.extend((left,right))
        if is_sync:sync.extend((left,right))
    def grouped(tag,limit=None):
        rows=sorted(({"name":key,**value} for key,value in groups[tag].items()),key=lambda r:r["summed_duration_seconds"],reverse=True)
        return rows if limit is None else rows[:limit]
    span=(stop-start)/1e6;kernel_union=union_seconds(kernels)
    result={"schema":"posteriorforge.failure_runtime_diagnostic.v1.streaming_trace","attribution":"Built with PriorLabs-TabPFN","diagnostic_only":True,"method":"Two passes with stdlib JSONDecoder; bounded buffer, only compact GPU/sync intervals retained.","trace_file":str(path),"trace_bytes":path.stat().st_size,"trace_event_count":all_events,"profiled_request_trace_span_seconds":span,"counts_and_summed_times_scope":"entire profiler session; interval unions clipped to annotated request","cuda_activity_status":"recorded" if counts["kernel"] else "missing_or_unavailable","cuda_kernel_event_count":counts["kernel"],"cuda_kernel_summed_seconds":sums["kernel"],"cuda_kernel_interval_union_seconds":kernel_union,"cuda_kernel_and_copy_interval_union_seconds":union_seconds(active),"cuda_copy_interval_union_seconds":union_seconds(copies),"cuda_runtime_sync_interval_union_seconds":union_seconds(sync),"cuda_copy_event_count":counts["copy"],"cuda_copy_summed_seconds":sums["copy"],"cuda_runtime_sync_event_count":counts["sync"],"cuda_runtime_sync_host_seconds":sums["sync"],"kernel_busy_fraction_of_profile_span":kernel_union/max(span,1e-30),"top_kernels":grouped("kernel",25),"copies":grouped("copy"),"memset":grouped("memset"),"sync_calls":grouped("sync"),"runtime_calls":grouped("runtime",20),"annotated_host_spans":labels,"cautions":["Kernel sums can overlap; interval union avoids that double count.","Prediction host wall includes CPU work, GPU waits and transfers; it is not CPU compute time.","Host sync can overlap active kernels; never add it to GPU time as independent cost.","This profiled diagnostic differs in query count and overhead from the original 50000-vector requests; no exact original decomposition is established."],"elapsed_summary_seconds":time.perf_counter()-start_clock,"maxrss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    return result


def benchmark_captured_sampler(*,capture_dir,output):
    """CPU only: unchanged sample() at fixed captured logits, without model calls."""
    capture_dir=Path(capture_dir)
    manifest=json.loads((capture_dir/"manifest.json").read_text())
    if manifest.get("state")!="completed_diagnostic":
        raise ValueError("completed authenticated bar capture required")
    capture=capture_dir/"bar-logits-capture.npz"
    if sha256_file(capture)!=manifest["capture_sha256"]:
        raise ValueError("bar capture hash mismatch")
    if Path(output).exists():raise FileExistsError(output)
    import inspect
    import platform
    import numpy as np
    import torch
    from tabpfn.model.bar_distribution import FullSupportBarDistribution
    torch.set_num_threads(16);torch.manual_seed(20261111)
    criterion_source=Path(inspect.getfile(FullSupportBarDistribution))
    original_hash=sha256_file(criterion_source)
    if original_hash!=manifest["criterion_source_sha256"]:raise ValueError("captured criterion source changed")
    result={**runtime_identity(),"schema":SCHEMA+".sampler_cpu","attribution":ATTRIBUTION,
      "diagnostic_only":True,"model_calls":0,"gpu_calls":0,"genuine_inference_request":False,
      "original_gate_changed":False,"input_capture_sha256":manifest["capture_sha256"],
      "criterion_source_sha256":original_hash,"torch":torch.__version__,"python":platform.python_version(),
      "cpu_threads":torch.get_num_threads(),
      "procedure":"Actual unchanged FullSupportBarDistribution.sample; cyclic repetitions of captured actual FP32 CPU logits to 10000 rows; constructor, tile setup and 128-row warmup excluded; 3 timed calls per stage.",
      "limitation":"Fixed captured-logit CPU microbenchmark, not a complete model request or an exact original 50000-vector decomposition. Selected captured rows are not representative of all proposal prefixes.","stages":[]}
    with np.load(capture,allow_pickle=False) as values:
        for stage in (0,6,9,14,16,20):
            logits=torch.from_numpy(values[f"stage{stage}_logits"].copy())
            borders=torch.from_numpy(values[f"stage{stage}_borders"].copy())
            criterion=FullSupportBarDistribution(borders)
            tiled=logits.repeat((10000+len(logits)-1)//len(logits),1)[:10000].contiguous()
            if tiled.dtype!=torch.float32 or tiled.device.type!="cpu":
                raise ValueError("captured FP32 CPU logits required")
            with torch.no_grad():
                criterion.sample(tiled[:128]);durations=[]
                for repeat in range(3):
                    started=time.perf_counter();draw=criterion.sample(tiled);durations.append(time.perf_counter()-started)
                    if draw.shape!=(10000,) or not draw.isfinite().all():
                        raise RuntimeError("CPU sampler microbenchmark integrity failure")
            result["stages"].append({"stage_index":stage,"parameter":manifest["parameter_order"][stage],
                "captured_rows":len(logits),"timed_rows":10000,"logits_shape":list(tiled.shape),
                "wall_seconds":durations,"median_wall_seconds":statistics.median(durations),"finite_outputs":True})
    if sha256_file(criterion_source)!=original_hash:raise RuntimeError("criterion source changed")
    result["source_hash_unchanged"]=True
    with Path(output).open("x") as stream:json.dump(result,stream,indent=2)
    return result


def _budget_binding(path):
    from posteriorforge.budget import BudgetLedger, HELD_STATES
    job = os.environ.get("SLURM_JOB_ID")
    if not job:
        raise RuntimeError("diagnostic profile requires a reserved Slurm allocation")
    ledger = BudgetLedger(path).inspect()
    matching = [entry for entry in ledger["reservations"] if entry.get("job_id") == job]
    if len(matching) != 1 or matching[0]["state"] not in HELD_STATES or matching[0]["gpus"] != 1:
        raise RuntimeError("one active one-GPU reservation for this job required")
    return {key: matching[0][key] for key in ("reservation_id", "job_id", "gpus", "walltime_seconds", "purpose")}


def profile_baseline(*, banks, checkpoint, output, ledger, profile_count=10000):
    if type(profile_count) is not int or not 1 <= profile_count <= 10000:
        raise ValueError("profile_count must be 1..10000; full scientific sweeps forbidden")
    binding = _budget_binding(ledger)
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    result = {**runtime_identity(), "schema": SCHEMA, "attribution": ATTRIBUTION,
        "baseline_commit": BASELINE_COMMIT, "diagnostic_only": True,
        "scientific_gate_reclassified": False, "optimization_performed": False,
        "budget_binding": binding, "case": 0, "context_seed": 20261111, "context_size": 4096,
        "profile_count": profile_count, "status": "running", "requests": []}
    write_json(output/"manifest.json", result)
    try:
        import numpy as np
        import torch
        from posteriorforge.banks import derive_seed, load_context, load_heldout
        from posteriorforge.inference import (CHECKPOINT_SHA256, UPSTREAM_REVISION, _prior,
            validate_query, frozen_regressor_settings, upstream_source_hashes, support_mask)
        from posteriorforge.science.analytical_benchmark_v2 import array_descriptor, build_problem
        from posteriorforge.upstream import load_baseline
        if sha256_file(checkpoint) != CHECKPOINT_SHA256:
            raise ValueError("pinned checkpoint hash mismatch")
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("one visible CUDA device required")
        context=load_context(banks,20261111,4096); heldout=load_heldout(banks,0)
        query=validate_query(heldout.obs)
        hashes_before=upstream_source_hashes()
        provenance=json.loads((Path(__file__).resolve().parents[1]/"provenance/upstream-npe-pfn.json").read_text())
        expected={Path(name).name: value["sha256"] for name,value in provenance["python_files"].items()}
        if hashes_before != expected or provenance["revision"] != UPSTREAM_REVISION:
            raise ValueError("upstream scientific baseline was modified")
        seed=derive_seed(20261111,"failure-runtime-diagnostic",case=0)
        np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS","16")))
        prior=_prior(torch)
        kwargs=frozen_regressor_settings(checkpoint); kwargs["inference_precision"]=torch.float32
        started=time.perf_counter()
        posterior=load_baseline()(prior=prior,filter_type="no_filtering",show_progress_bars=False,
                                  regressor_init_kwargs=kwargs)
        posterior.append_simulations(torch.from_numpy(context.theta.copy()).float(),torch.from_numpy(context.obs.copy()).float())
        torch.cuda.synchronize()
        result.update(constructor_and_bank_attachment_seconds=time.perf_counter()-started,
            checkpoint_sha256=CHECKPOINT_SHA256,upstream_revision=UPSTREAM_REVISION,upstream_files=hashes_before,
            regressor_settings=frozen_regressor_settings(checkpoint),seed=seed,
            context_theta=array_descriptor(context.theta),context_obs=array_descriptor(context.obs),query=array_descriptor(query),
            parameter_order=list(build_problem().parameter_order),scientific_fingerprint=build_problem().scientific_fingerprint,
            device=torch.cuda.get_device_name(),torch_cuda=torch.version.cuda,
            torch_float32_matmul_precision=torch.get_float32_matmul_precision(),
            cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            cudnn_allow_tf32=torch.backends.cudnn.allow_tf32)
        recorder=CallRecorder(torch.profiler.record_function)
        def request(name,count,profile=False):
            recorder.request=name; recorder.proposal=0
            before=(prior.support.calls,prior.support.proposed,prior.support.accepted)
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            session_started=time.perf_counter()
            if profile:
                session=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
                    record_shapes=False,profile_memory=True,with_stack=False)
            else:session=nullcontext()
            with session as prof:
                with torch.profiler.record_function(f"pfdiagnostic/{name}/request"):
                    request_started=time.perf_counter()
                    with torch.no_grad():
                        samples=posterior.sample((count,),x=torch.from_numpy(query).float(),max_sampling_batch_size=10000,
                                                 max_iter_rejection=None,with_log_prob=False)
                    torch.cuda.synchronize()
                    request_seconds=time.perf_counter()-request_started
            session_seconds=time.perf_counter()-session_started
            values=samples.detach().cpu().numpy().astype(np.float64)
            if values.shape != (count,21) or not support_mask(values).all():
                raise RuntimeError("diagnostic sample integrity failed; never repair rows")
            after=(prior.support.calls,prior.support.proposed,prior.support.accepted)
            selected=[event for event in recorder.events if event["request"]==name]
            totals=defaultdict(float);calls=Counter()
            for event in selected:
                operation=event["operation"];totals[operation]+=event["host_wall_seconds"];calls[operation]+=1
            external_cache=sum(event["host_wall_seconds"] for event in selected if event["operation"] in ("empty_cache","ipc_collect") and event["parent_operation"]=="proposal")
            row={"name":name,"requested_complete_count":count,"profiler_enabled":profile,
                "request_seconds":request_seconds,"session_seconds_including_profiler_setup_flush":session_seconds,
                "support_calls":after[0]-before[0],"proposed":after[1]-before[1],
                "accepted_before_trim":after[2]-before[2],"returned":count,
                "candidate_acceptance_rate":(after[2]-before[2])/(after[1]-before[1]),
                "proposal_sizes":[event["query_rows"] for event in selected if event["operation"]=="proposal"],
                "invalid_candidates_rejected":(after[1]-before[1])-(after[2]-before[2]),
                "valid_candidates_trimmed":(after[2]-before[2])-count,
                "host_seconds_by_operation_inclusive":dict(totals),"call_counts":dict(calls),
                "external_cache_host_seconds":external_cache,
                "proposal_other_host_seconds":totals["proposal"]-totals["fit"]-totals["predict"]-external_cache,
                "request_other_host_seconds":request_seconds-totals["proposal"]-totals["support"],
                "peak_allocated_bytes":torch.cuda.max_memory_allocated(),"peak_reserved_bytes":torch.cuda.max_memory_reserved(),
                "samples":array_descriptor(values),"complete_finite_in_support":True}
            result["requests"].append(row)
            write_json(output/"manifest.json",result)
            if profile:
                trace_path=output/"runtime.trace.json"
                prof.export_chrome_trace(str(trace_path))
                operators=[]
                for item in prof.key_averages():
                    operators.append({"name":item.key,"count":item.count,
                        "self_cpu_seconds":float(item.self_cpu_time_total)/1e6,
                        "inclusive_cpu_seconds":float(item.cpu_time_total)/1e6,
                        "self_cuda_seconds":float(getattr(item,"self_device_time_total",0.))/1e6,
                        "inclusive_cuda_seconds":float(getattr(item,"device_time_total",0.))/1e6})
                operator_summary={"scope":"PyTorch key averages; self times avoid nested operator double count",
                    "top_self_cpu":sorted(operators,key=lambda item:item["self_cpu_seconds"],reverse=True)[:40],
                    "top_self_cuda":sorted(operators,key=lambda item:item["self_cuda_seconds"],reverse=True)[:40]}
                write_json(output/"runtime-operator-summary.json",operator_summary)
                result["operator_summary"]=operator_summary
                trace_summary=summarize_trace(trace_path)
                write_json(output/"runtime-trace-summary.json",trace_summary)
                with trace_path.open("rb") as source,gzip.open(output/"runtime.trace.json.gz","wb",compresslevel=5) as target:
                    shutil.copyfileobj(source,target,1024*1024)
                result["trace"]={"file":"runtime.trace.json.gz","sha256":sha256_file(output/"runtime.trace.json.gz"),
                    "uncompressed_sha256":sha256_file(trace_path),"compressed_bytes":(output/"runtime.trace.json.gz").stat().st_size}
                trace_path.unlink()
                result["trace_summary"]=trace_summary
                write_json(output/"manifest.json",result)
        with recorder.instrument(posterior,torch.cuda):
            request("cold_single",1)
            request("profile",profile_count,True)
            request("warm_single",1)
        if upstream_source_hashes()!=hashes_before:raise RuntimeError("upstream source mutated")
        result.update(status="complete",upstream_source_unchanged=True,
            model={"parameter_count":sum(p.numel() for p in posterior._model.model_.parameters()),
                   "dtype":str(next(posterior._model.model_.parameters()).dtype),
                   "parameter_device_after_request":str(next(posterior._model.model_.parameters()).device),
                   "autocast":bool(posterior._model.use_autocast_)},
            instrumentation={"vendor_source_modified":False,"extra_per_method_cuda_synchronization":False,
                "outer_request_cuda_synchronization_matches_baseline":True,
                "host_method_durations_include_device_waits":True,
                "profiler_record_shapes":False,"stage_input_shapes_recorded_by_wrapper":True,
                "profiler_memory_events":True,"profiler_stack_capture":False,
                "diagnostic_workload_not_a_50000_request":True,"no_exact_50000_breakdown_extrapolation":True})
        fields=["request","proposal","stage_index","features","query_rows","operation","parent_operation","host_wall_seconds","error"]
        with (output/"runtime-stage-events.csv").open("w",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=fields,extrasaction="ignore");writer.writeheader();writer.writerows(recorder.events)
        write_json(output/"runtime-stage-events.json",recorder.events)
        _plot_profile(result,output)
    except BaseException as error:
        result.update(status="failed",error_type=type(error).__name__,error=str(error))
        write_json(output/"manifest.json",result)
        raise
    write_json(output/"manifest.json",result)
    return result


def _plot_profile(result,output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout="constrained")
    requests=result["requests"]
    axes[0].bar([r["name"] for r in requests],[r["request_seconds"] for r in requests],color=["#425775","#c76b1b","#268079"])
    axes[0].set(ylabel="Actual request seconds",title="Diagnostic counts differ; profile includes overhead")
    profile=next(r for r in requests if r["profiler_enabled"])
    totals=profile["host_seconds_by_operation_inclusive"]
    labels=["fit","predict","external cache","other"]
    values=[totals["fit"],totals["predict"],profile["external_cache_host_seconds"],
            profile["proposal_other_host_seconds"]+totals["support"]+profile["request_other_host_seconds"]]
    axes[1].bar(labels,values,color=["#425775","#c76b1b","#268079","#9da8b8"])
    axes[1].set(ylabel="Host wall seconds (disjoint top-level groups)",title="Prediction includes GPU compute/waits, not CPU-only")
    axes[1].tick_params(axis="x",labelrotation=15)
    fig.suptitle(ATTRIBUTION+"\nInstrumentation only: same 4096-context frozen FP32 baseline; no optimized variant",fontsize=12)
    fig.savefig(output/"runtime-profile-breakdown.png",dpi=160,bbox_inches="tight");plt.close(fig)


def publish_runtime_report(*,profile_dir,baseline_summary,trace_summary,controlled_stop,sampler,output):
    """Publish recovered partial-profile evidence without marking it complete."""
    import re
    from posteriorforge.inference import upstream_source_hashes
    profile_dir=Path(profile_dir);output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((profile_dir/"manifest.json").read_text())
    baseline=json.loads(Path(baseline_summary).read_text())
    trace=json.loads(Path(trace_summary).read_text())
    stop=json.loads(Path(controlled_stop).read_text())
    sample=json.loads(Path(sampler).read_text())
    if manifest["status"]!="running" or stop.get("warm_single_measured") is not False:
        raise ValueError("this recovery report expects the preserved partial controlled-stop run")
    names=[r["name"] for r in manifest["requests"]]
    if names!=["cold_single","profile"] or not stop["request_completed"]:
        raise ValueError("the two actual completed requests are required")
    after=upstream_source_hashes()
    if after!=manifest["upstream_files"]:raise RuntimeError("upstream source differs from pre-wrapper record")
    cold,profile=manifest["requests"]
    copies={"runtime-original-partial-manifest.json":profile_dir/"manifest.json",
            "runtime-controlled-stop.json":Path(controlled_stop),"runtime-trace-summary.json":Path(trace_summary),
            "runtime-sampler-cpu.json":Path(sampler)}
    inputs={}
    for name,path in copies.items():
        shutil.copyfile(path,output/name)
        inputs[name]={"source":str(path),"sha256":sha256_file(path)}
    inputs["runtime-baseline-summary.json"]={"source":str(baseline_summary),"sha256":sha256_file(baseline_summary)}
    partitions=[]
    for request in (cold,profile):
        totals=request["host_seconds_by_operation_inclusive"]
        other=request["proposal_other_host_seconds"]+totals["support"]+request["request_other_host_seconds"]
        values={"fit":totals["fit"],"predict":totals["predict"],"external_cache":request["external_cache_host_seconds"],"other":other}
        if abs(sum(values.values())-request["request_seconds"])>1e-6:raise ValueError("host partition mismatch")
        partitions.append({"request":request["name"],"requested_vectors":request["requested_complete_count"],"proposal_sizes":request["proposal_sizes"],"request_seconds":request["request_seconds"],"profiler_enabled":request["profiler_enabled"],"host_partition_seconds":values,"host_partition_fractions":{key:value/request["request_seconds"] for key,value in values.items()}})
    attention=sum(row["summed_duration_seconds"] for row in trace["top_kernels"] if "AttentionKernel" in row["name"])
    result={"schema":SCHEMA+".recovered_evidence","attribution":ATTRIBUTION,"diagnostic_only":True,
      "original_gate_unchanged":True,"scientific_gate_reclassified":False,"optimization_performed":False,
      "status":"completed_sampling_and_raw_trace_recovered_after_controlled_postprocessing_stop",
      "original_profile_manifest_status_preserved":manifest["status"],"warm_single":"not_measured",
      "baseline_commit":BASELINE_COMMIT,"profile_execution_git_revision":manifest["git_revision"],
      "case":manifest["case"],"context_seed":manifest["context_seed"],"context_size":manifest["context_size"],
      "checkpoint_sha256":manifest["checkpoint_sha256"],"settings":manifest["regressor_settings"],
      "device":manifest["device"],"float32_matmul_precision":manifest["torch_float32_matmul_precision"],
      "upstream_hashes_before":manifest["upstream_files"],"upstream_hashes_after":after,"upstream_source_unchanged":True,
      "all_original_24_requests":baseline["summary"],"actual_diagnostic_requests":manifest["requests"],
      "host_partition":partitions,"trace_metrics":{key:trace[key] for key in ("trace_event_count","profiled_request_trace_span_seconds","cuda_kernel_event_count","cuda_kernel_summed_seconds","cuda_kernel_interval_union_seconds","cuda_kernel_and_copy_interval_union_seconds","cuda_copy_interval_union_seconds","cuda_copy_event_count","cuda_copy_summed_seconds","cuda_runtime_sync_event_count","cuda_runtime_sync_host_seconds","cuda_runtime_sync_interval_union_seconds","kernel_busy_fraction_of_profile_span")},
      "dominant_attention_summed_seconds":attention,"attention_fraction_of_kernel_sum":attention/trace["cuda_kernel_summed_seconds"],
      "attention_summed_fraction_of_profile_span":attention/trace["profiled_request_trace_span_seconds"],
      "copies":trace["copies"],"sync_calls":trace["sync_calls"],"top_kernels":trace["top_kernels"][:8],
      "cpu_sampler_median_seconds_range":[min(r["median_wall_seconds"] for r in sample["stages"]),max(r["median_wall_seconds"] for r in sample["stages"])],
      "source_call_graph":{"each_proposal":"21 autoregressive fit/predict stages; predict iterates 8 ensembles and moves model to CUDA then CPU; final CPU logits are sampled by per-row icdf.","each_fit":"Pinned SDK reloads/reconstructs checkpoint and fits stage preprocessors; measured fit wall does not isolate physical disk I/O.","original_24_calls":"6 proposals/support calls per original request =>126 fit/predict pairs; first 5 full 10000, final adaptive 387–1994, not 60000 total."},
      "measured_conclusions":["For this 10000-accepted-vector diagnostic, prediction host spans and actual GPU kernels dominate; FP32 attention is the largest recorded kernel group.","Host synchronization and host memcpy API waits overlap GPU execution and must not be added to GPU time as independent overhead.","Device copy durations are below 1 second in this trace; host memcpy API duration includes waits and is not device transfer time.","Single accepted vector took 149.243 seconds after constructor/bank attachment; original ≈47 ms/vector is amortized throughput, not isolated latency.","The CPU scalar sampler costs about 1.3–2.3 seconds per 10000-row stage in this fixed-logit microbenchmark; its per-row operators generate a large trace, but it is secondary to GPU prediction in the measured request."],
      "limitations":["Cold-single, profiled 10000 and original 50000 requests use different counts; no exact original 50000 cost decomposition or overhead factor is extrapolated.","One unprofiled single-vector timing is not a latency distribution; warm-single was not measured.","Controlled stop occurred after complete request and raw trace export during CPU-only profiler key averaging; the original partial manifest is preserved.","Trace-derived GPU interval unions and host partitions are overlapping views, not additive categories.","Rejection and final valid overshoot are distinct; measured proposal sizes are retained for every completed diagnostic request.","No scientific gate was rerun or passed by this evidence."],"inputs":inputs}
    write_json(output/"runtime-evidence.json",result)
    with (output/"runtime-stage-spans.csv").open("w",newline="") as stream:
        fields=["proposal","stage_index","parameter","query_rows","operation","host_seconds"]
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for row in trace["annotated_host_spans"]:
            match=re.match(r"pfdiagnostic/profile/p(\d+)/s(-?\d+)/(.*)$",row["name"])
            if not match:continue
            proposal,stage,operation=int(match[1]),int(match[2]),match[3]
            writer.writerow({"proposal":proposal,"stage_index":stage,"parameter":manifest["parameter_order"][stage] if stage>=0 else "all stages","query_rows":profile["proposal_sizes"][proposal-1],"operation":operation,"host_seconds":row["duration_seconds"]})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(1,2,figsize=(13.5,5.0),layout="constrained")
    colors={"fit":"#4b6382","predict":"#cb7b31","external_cache":"#267d75","other":"#a8b2c0"}
    y=np.arange(2);left=np.zeros(2)
    for key in ("fit","predict","external_cache","other"):
        values=[r["host_partition_seconds"][key] for r in partitions]
        axes[0].barh(y,values,left=left,color=colors[key],label=key.replace("_"," "));left+=values
    axes[0].set_yticks(y,["Cold: 1 vector\nproposals [1]","Profile: 10000 vectors\nproposals [10000,101]"])
    axes[0].set(xlabel="Measured request wall seconds",title="Actual host spans, separate workloads",xlim=(0,680))
    axes[0].invert_yaxis();axes[0].legend(loc="lower right",fontsize=9)
    for index,row in enumerate(partitions):axes[0].text(row["request_seconds"]+7,index,format(row["request_seconds"],".2f")+"s",va="center",fontsize=10)
    span=trace["profiled_request_trace_span_seconds"]
    labels=["GPU kernel union","GPU kernel + copy union","Host sync wait union"]
    values=[trace["cuda_kernel_interval_union_seconds"],trace["cuda_kernel_and_copy_interval_union_seconds"],trace["cuda_runtime_sync_interval_union_seconds"]]
    axes[1].barh(range(3),[span]*3,color="#e9edf2",label="Profile request span")
    axes[1].barh(range(3),values,color=["#cb7b31","#267d75","#4b6382"])
    axes[1].set_yticks(range(3),labels);axes[1].invert_yaxis()
    axes[1].set(xlabel="Interval union seconds inside profiled request",title="GPU activity and host waits overlap",xlim=(0,680))
    for index,value in enumerate(values):axes[1].text(value+7,index,f"{value:.2f}s",va="center",fontsize=10)
    axes[1].legend(loc="lower right",fontsize=9)
    for ax in axes:ax.grid(axis="x",alpha=.15);ax.set_axisbelow(True)
    fig.suptitle(ATTRIBUTION+"\nFrozen FP32 baseline diagnostic; no optimization or gate change",fontsize=13)
    fig.supxlabel("Prediction host time includes GPU compute/waits. Profiler overhead included; warm-single unmeasured. No exact 50k extrapolation.",fontsize=9)
    fig.savefig(output/"runtime-profile-breakdown.png",dpi=170,bbox_inches="tight");plt.close(fig)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest="operation",required=True)
    summary=commands.add_parser("summary")
    for name in ("runs","ledger","output"):summary.add_argument("--"+name,required=True)
    profile=commands.add_parser("profile")
    for name in ("banks","checkpoint","output","ledger"):profile.add_argument("--"+name,required=True)
    profile.add_argument("--profile-count",type=int,default=10000)
    trace=commands.add_parser("trace-summary")
    trace.add_argument("--trace",required=True);trace.add_argument("--output",required=True)
    sampler=commands.add_parser("sampler-cpu")
    sampler.add_argument("--capture-dir",required=True);sampler.add_argument("--output",required=True)
    report=commands.add_parser("report")
    for name in ("profile-dir","baseline-summary","trace-summary","controlled-stop","sampler","output"):report.add_argument("--"+name,required=True)
    args=vars(parser.parse_args());operation=args.pop("operation")
    if operation=="summary":result=summarize_saved(**args)
    elif operation=="profile":result=profile_baseline(**args)
    elif operation=="sampler-cpu":result=benchmark_captured_sampler(**args)
    elif operation=="report":result=publish_runtime_report(**args)
    else:
        output=Path(args["output"])
        if output.exists():raise FileExistsError(output)
        result=summarize_trace(args["trace"])
        with output.open("x") as stream:json.dump(result,stream,indent=2)
    print(json.dumps(result,indent=2,sort_keys=True,allow_nan=False))


if __name__=="__main__":main()
