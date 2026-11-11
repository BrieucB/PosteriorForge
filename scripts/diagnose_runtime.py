#!/usr/bin/env python3
"""Read-only baseline timing evidence and reversible failure-diagnostic profiling.

Built with PriorLabs-TabPFN. This never changes or reclassifies the scientific gate.
Accelerator imports happen only in the explicit, budgeted ``profile`` operation.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager, nullcontext
import csv
import gzip
import hashlib
import json
from pathlib import Path
import os
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


def summarize_trace(path):
    """Summarize CPU/CUDA trace intervals without adding overlapping durations."""
    with Path(path).open() as stream:
        trace = json.load(stream)
    events = [event for event in trace.get("traceEvents", [])
              if event.get("ph") == "X" and isinstance(event.get("dur"), (float, int))]
    kernels = [event for event in events if "kernel" in event.get("cat", "").lower()]
    copies = [event for event in events if "memcpy" in event.get("cat", "").lower()]
    memset = [event for event in events if "memset" in event.get("cat", "").lower()]
    runtime = [event for event in events if any(tag in event.get("cat", "").lower() for tag in ("cuda_runtime", "cuda_driver"))]
    sync = [event for event in runtime if "synchroniz" in event.get("name", "").lower()]
    request = [event for event in events if event.get("name") == "pfdiagnostic/profile/request"]
    if len(request) != 1:
        raise ValueError("exactly one annotated profiled request required")
    start, stop = request[0]["ts"], request[0]["ts"] + request[0]["dur"]
    def duration(items): return sum(event["dur"] for event in items)/1e6
    def union(items):
        return interval_union([(max(start,event["ts"]), min(stop,event["ts"]+event["dur"]))
                               for event in items if event["ts"] < stop and event["ts"]+event["dur"] > start])/1e6
    def grouped(items):
        values = defaultdict(lambda: {"count": 0, "summed_duration_seconds": 0., "bytes": None, "events_with_byte_count": 0})
        for event in items:
            row = values[event.get("name", "unknown")]
            row["count"] += 1; row["summed_duration_seconds"] += event["dur"]/1e6
            args = event.get("args", {})
            for key in ("Bytes", "bytes", "size"):
                if isinstance(args.get(key), (int,float)):
                    row["bytes"] = (row["bytes"] or 0) + args[key]
                    row["events_with_byte_count"] += 1; break
        return sorted([{"name": key, **value} for key,value in values.items()],
                      key=lambda row: row["summed_duration_seconds"], reverse=True)
    result = {"schema": SCHEMA + ".trace", "trace_event_count": len(trace.get("traceEvents", [])),
        "profiled_request_trace_span_seconds": (stop-start)/1e6,
        "counts_and_summed_times_scope": "entire profiler session; GPU interval unions clipped to annotated request",
        "cuda_kernel_event_count": len(kernels), "cuda_kernel_summed_seconds": duration(kernels),
        "cuda_kernel_interval_union_seconds": union(kernels),
        "cuda_kernel_and_copy_interval_union_seconds": union(kernels+copies+memset),
        "cuda_copy_event_count": len(copies), "cuda_copy_summed_seconds": duration(copies),
        "cuda_runtime_sync_event_count": len(sync), "cuda_runtime_sync_host_seconds": duration(sync),
        "top_kernels": grouped(kernels)[:25], "copies": grouped(copies), "sync_calls": grouped(sync),
        "runtime_calls": grouped(runtime)[:20],
        "cautions": ["Kernel sums can overlap across streams; interval union avoids that double count.",
            "Host prediction durations include device waits and copies; they are not CPU compute durations.",
            "Host synchronization can overlap active kernels; never add it to GPU time as independent cost.",
            "A diagnostic query count and profiler overhead do not supply an exact original 50k breakdown."]}
    if not kernels:
        result["cuda_activity_status"] = "missing_or_unavailable"
    else:
        result["cuda_activity_status"] = "recorded"
    result["kernel_busy_fraction_of_profile_span"] = result["cuda_kernel_interval_union_seconds"] / max((stop-start)/1e6, 1e-30)
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


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest="operation",required=True)
    summary=commands.add_parser("summary")
    for name in ("runs","ledger","output"):summary.add_argument("--"+name,required=True)
    profile=commands.add_parser("profile")
    for name in ("banks","checkpoint","output","ledger"):profile.add_argument("--"+name,required=True)
    profile.add_argument("--profile-count",type=int,default=10000)
    args=vars(parser.parse_args());operation=args.pop("operation")
    result=summarize_saved(**args) if operation=="summary" else profile_baseline(**args)
    print(json.dumps(result,indent=2,sort_keys=True,allow_nan=False))


if __name__=="__main__":main()
