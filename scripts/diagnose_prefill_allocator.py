#!/usr/bin/env python3
"""Passive stage20 memory-factor/allocator witness; no policy changes or samples.

Built with PriorLabs-TabPFN. The driver-free estimator hypothesis is explicitly
refuted by pinned source: availability is total minus allocated, KV prefill does
not auto-reset, and cached-query layers locally disable the saving factor.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager,nullcontext
import functools
import gc
import gzip
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import random
import sys
from threading import Lock,get_ident
import time
import numpy as np

SCHEMA="posteriorforge.prefill_allocator_diagnostic.v1"
MEMORY_SHA256="80783faf3a29e8bbafeaeedfc069bfd1ae275cb9dc607c234548aeaba2aa2050"
BATCH_PROOF_SHA256="7bc0e4449abe4aaf0a62b0df55fb25eda423f821524955c9266f8f91761aa77c"
STAGE_SHA256="ac22c67c23403723902805666d1239bd30a8af355554df58a8638a8fd1f21e82"
BASE_QUALIFIER_SHA256="6c927a5c278c7fc54b556ac557ecef64f5e6338f5ae13bf466f3b1e6a4640f90"
QUERY_SIZES=(1024,6250,10000,10000,6250,1024)
_LOCK=Lock()


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_numerics():
    path=Path(__file__).with_name("qualify_execution_cache.py")
    if sha(path)!=BASE_QUALIFIER_SHA256:raise RuntimeError("frozen numerical utility source changed")
    spec=importlib.util.spec_from_file_location("_prefill_numerics",path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def factors(model):
    layers=getattr(getattr(model,"transformer_encoder",None),"layers",())
    return [getattr(layer,"save_peak_mem_factor","missing") for layer in layers]


def shape(value):return list(value.shape) if hasattr(value,"shape") else None


class PassiveObserver:
    """Scoped original-call-once wrappers; all arguments/returns/order preserved."""
    def __init__(self,record_function=None,max_events=20000):
        self.records=[];self.restorations=[];self.patches=[];self.phase="unlabelled";self.owner=get_ident();self.forward_stack=[]
        self.record_function=record_function or (lambda label:nullcontext());self.max_events=max_events;self.overflow=0
    def append(self,row):
        if len(self.records)<self.max_events:self.records.append(row)
        else:self.overflow+=1
    @contextmanager
    def label(self,label):
        old=self.phase;self.phase=label
        try:yield
        finally:self.phase=old
    def patch(self,cls,name,kind):
        original_descriptor=inspect.getattr_static(cls,name)
        original=original_descriptor.__func__ if isinstance(original_descriptor,classmethod) else original_descriptor
        if not callable(original):raise TypeError("plain method/classmethod required")
        observer=self
        @functools.wraps(original)
        def wrapped(receiver,*args,**kwargs):
            if get_ident()!=observer.owner:raise RuntimeError("passive observer called from nonowner thread")
            row={"phase":observer.phase,"kind":kind,"class":cls.__name__,"method":name,
                 "receiver_id":id(receiver),"forward_model_id":observer.forward_stack[-1] if observer.forward_stack else None}
            model=None
            if kind=="reset_decision":
                model=kwargs.get("model",args[1] if len(args)>1 else None);x=kwargs.get("X",args[2] if len(args)>2 else None)
                row.update(model_id=id(model),before_layer_factors=factors(model),save_peak_mem=kwargs.get("save_peak_mem",args[0] if args else None),
                    X_shape=shape(x),cache_kv=kwargs.get("cache_kv"),dtype_byte_size=kwargs.get("dtype_byte_size"),
                    safety_factor=kwargs.get("safety_factor",5.),n_train_samples=kwargs.get("n_train_samples"))
            elif kind=="estimate":
                row.update(X_shape=shape(kwargs.get("X",args[0] if args else None)),cache_kv=kwargs.get("cache_kv"),n_train_samples=kwargs.get("n_train_samples"),unit=kwargs.get("unit","gb"))
            elif kind=="free_memory":row.update(device=str(kwargs.get("device",args[0] if args else None)),unit=kwargs.get("unit","gb"))
            elif kind=="set_factor":row.update(before_layer_factors=factors(receiver),requested_factor=kwargs.get("factor",args[0] if args else None))
            elif kind=="model_forward":
                row.update(layer_factors=factors(receiver),cache_trainset_representation=bool(getattr(receiver,"cache_trainset_representation",False)),single_eval_pos=kwargs.get("single_eval_pos"))
                observer.forward_stack.append(id(receiver))
            elif kind=="layer_forward":row.update(stored_factor=getattr(receiver,"save_peak_mem_factor","missing"),single_eval_pos=kwargs.get("single_eval_pos"),cache_trainset_representation=kwargs.get("cache_trainset_representation"))
            elif kind=="effective_compute":row.update(effective_save_peak_mem_factor=kwargs.get("save_peak_mem_factor"),input_shape=shape(args[0] if args else None),allow_inplace=kwargs.get("allow_inplace"))
            t0=time.perf_counter()
            try:
                with observer.record_function("pfpassive/"+observer.phase+"/"+kind):result=original(receiver,*args,**kwargs)
                if kind in ("free_memory","estimate"):row["actual_original_return"]=float(result)
                if kind=="reset_decision":row["after_layer_factors"]=factors(model)
                if kind=="set_factor":row["after_layer_factors"]=factors(receiver)
                return result
            except BaseException as error:row["error_type"]=type(error).__name__;raise
            finally:
                row["host_call_seconds"]=time.perf_counter()-t0;observer.append(row)
                if kind=="model_forward":observer.forward_stack.pop()
        replacement=classmethod(wrapped) if isinstance(original_descriptor,classmethod) else wrapped
        setattr(cls,name,replacement);self.patches.append((cls,name,original_descriptor,replacement))
    def close(self):
        conflict=False
        for cls,name,original,replacement in reversed(self.patches):
            conflict|=inspect.getattr_static(cls,name) is not replacement;setattr(cls,name,original)
            self.restorations.append({"class":cls.__name__,"method":name,"exact_descriptor_restored":inspect.getattr_static(cls,name) is original})
        self.patches.clear()
        if conflict:raise RuntimeError("passive observer alias changed during scope")


@contextmanager
def observe_sdk(memory,transformer,layer,attention,mlp,torch):
    if not _LOCK.acquire(blocking=False):raise RuntimeError("passive SDK observer already active")
    observer=PassiveObserver(torch.profiler.record_function)
    try:
        for cls,name,kind in ((memory.MemoryUsageEstimator,"get_max_free_memory","free_memory"),
            (memory.MemoryUsageEstimator,"estimate_memory_of_one_batch","estimate"),(memory.MemoryUsageEstimator,"reset_peak_memory_if_required","reset_decision"),
            (transformer.PerFeatureTransformer,"reset_save_peak_mem_factor","set_factor"),(transformer.PerFeatureTransformer,"forward","model_forward"),
            (layer.PerFeatureEncoderLayer,"forward","layer_forward"),(attention.MultiHeadAttention,"_compute","effective_compute"),(mlp.MLP,"_compute","effective_compute")):
            observer.patch(cls,name,kind)
        yield observer
    finally:
        try:observer.close()
        finally:_LOCK.release()


def allocator_snapshot(torch,label):
    stats=torch.cuda.memory_stats();free,total=torch.cuda.mem_get_info()
    result={"label":label,"allocated_bytes":torch.cuda.memory_allocated(),"reserved_bytes":torch.cuda.memory_reserved(),
        "driver_free_bytes":free,"driver_total_bytes":total,"estimator_total_minus_allocated_bytes":torch.cuda.get_device_properties(0).total_memory-torch.cuda.memory_allocated(),
        "stats":{key:value for key,value in stats.items() if key in ("num_alloc_retries","num_ooms","num_sync_all_streams","num_device_alloc","num_device_free") or key in ("inactive_split_bytes.all.current","active_bytes.all.current","allocated_bytes.all.current")}}
    try:
        snapshot=torch.cuda.memory._snapshot();segments=snapshot.get("segments",[])
        inactive=[block["size"] for segment in segments for block in segment.get("blocks",[]) if block.get("state")=="inactive"]
        result.update(allocator_settings=snapshot.get("allocator_settings"),allocator_snapshot_keys=list(snapshot),
            inactive_block_count=len(inactive),inactive_block_bytes=sum(inactive),largest_inactive_block_bytes=max(inactive,default=0))
    except (AttributeError,RuntimeError) as error:result["allocator_settings_unavailable"]=type(error).__name__+": "+str(error)
    return result


def prefix_prediction(prediction,rows=128):
    return {"logits":prediction["logits"][:rows].detach().cpu().clone(),"borders":prediction["criterion"].borders.detach().cpu().clone(),
            "mass":prediction["component_raw_mass"][:rows].detach().cpu().clone()}


def compare_prefix(reference,candidate,utilities,torch):
    metric=utilities.compare_probabilities(reference["logits"].softmax(-1).numpy(),candidate["logits"].softmax(-1).numpy())
    borders=bool(torch.equal(reference["borders"],candidate["borders"]))
    metric.update(exact_borders=borders,float32=reference["logits"].dtype==candidate["logits"].dtype==torch.float32)
    metric["passed"]=bool(metric["passed"] and borders and metric["float32"])
    return metric


def check_prediction(prediction,cdf,before):
    descriptor=prediction.get("gqa_attention_descriptor",{})
    keys=("original_alias_restored","original_function_restored","original_code_unchanged","active_code_unchanged","gqa_source_unchanged","sdk_sources_unchanged")
    if not all(descriptor.get(key) is True for key in keys) or cdf["patched_call_count"]-before!=1:raise RuntimeError("GQA/CDF actual execution/restoration witness failed")


def summarize_profile_trace(path,*,boundary="pfdiagnostic/profile/request"):
    """Use the existing reader with one host request, retaining GPU range events."""
    reader_path=Path(__file__).with_name("diagnose_runtime.py")
    spec=importlib.util.spec_from_file_location("_passive_trace_reader",reader_path)
    reader=importlib.util.module_from_spec(spec);spec.loader.exec_module(reader)
    original=reader.iter_trace_events
    def events(trace_path):
        for event in original(trace_path):
            if event.get("name")==boundary:
                event={**event,"name":"pfdiagnostic/profile/request" if event.get("cat")=="user_annotation" else "passive_gpu_range/request"}
            yield event
    reader.iter_trace_events=events
    summary=reader.summarize_trace(path)
    summary["host_request_boundary_policy"]={"source_name":boundary,"category":"user_annotation","gpu_range_events_retained_under_distinct_name":True,"raw_trace_modified":False}
    summary["reader_source_sha256"]=sha(reader_path)
    return summary


def profile_call(call,path,torch):
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=False,profile_memory=False,with_stack=False) as profiler:
        with torch.profiler.record_function("pfdiagnostic/profile/request"):result=call();torch.cuda.synchronize()
    profiler.export_chrome_trace(str(path));del profiler
    # Existing streaming reader avoids retaining the trace JSON in memory.
    summary=summarize_profile_trace(path)
    compressed=path.with_suffix(path.suffix+".gz")
    with path.open("rb") as source,gzip.open(compressed,"wb",compresslevel=5) as target:
        while block:=source.read(1024*1024):target.write(block)
    return result,{"trace":compressed.name,"trace_sha256":sha(compressed),"uncompressed_trace_bytes":path.stat().st_size,"summary":summary,
                   "profile_memory":False,"record_shapes":False,"sampling_calls":0}


def run(*,banks,prefix_run,checkpoint,qualification,ledger,output,max_seconds=240):
    from posteriorforge.artifacts import runtime_identity,write_json
    from posteriorforge.reference_grid import _require_budgeted_gpu
    if max_seconds!=240:raise ValueError("predeclared240-second diagnostic deadline required")
    budget=_require_budgeted_gpu(ledger)
    if budget["gpus"]!=1:raise RuntimeError("one budgeted GPU required")
    output=Path(output);output.mkdir(parents=True,exist_ok=False);start=time.perf_counter();deadline=start+max_seconds;engine=None;observer=None;active=None
    report={**runtime_identity(),"schema":SCHEMA,"status":"running","budget_binding":budget,"source_sha256":sha(__file__),
        "posterior_vectors_generated":0,"stage_index":20,"query_sizes_in_order":list(QUERY_SIZES),"query_timings":[],"allocator":[],"traces":[],"timing_scope":"Python passive wrappers present; no profiler or capture clones inside timed calls; first128 parity performed after timing; no end-to-end speed claim",
        "estimator_hypothesis":"REFUTED: pinned SDK uses total minus allocated; KV prefill skips auto reset; cached-query layer effective factor is forcedNone",
        "policy_changes":[],"empty_cache_calls":0,"forced_factor_calls":0,"attribution":"Built with PriorLabs-TabPFN"}
    write_json(output/"manifest.json",report)
    try:
        import torch
        report["before_sdk_import"]={"allocator_environment":os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),"cuda_initialized":bool(getattr(torch.cuda,"_initialized",False)),"tabpfn_imported":"tabpfn" in sys.modules}
        from tabpfn.model import memory,transformer,layer,multi_head_attention,mlp
        report["after_sdk_import"]={"allocator_environment":os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),"cuda_initialized":bool(getattr(torch.cuda,"_initialized",False))}
        from posteriorforge import execution_gqa_hybrid as execution,execution_cache as cache
        from posteriorforge.banks import load_context,load_heldout,derive_seed
        from posteriorforge.inference import CHECKPOINT_SHA256,support_mask
        from posteriorforge.compat_cdf import finite_cdf_compatibility
        from posteriorforge.inference_cdf import guard_baseline_source
        from posteriorforge.science.analytical_benchmark_v2 import array_descriptor
        execution.require_frozen_sources();utilities=load_numerics()
        if sha(memory.__file__)!=MEMORY_SHA256 or execution.execution_source_sha256()!=STAGE_SHA256:raise RuntimeError("pinned memory/GQA source mismatch")
        if sha(checkpoint)!=CHECKPOINT_SHA256 or sha(qualification)!=BATCH_PROOF_SHA256:raise ValueError("authenticated checkpoint/large-batch qualification required")
        q=json.loads(Path(qualification).read_text())
        sdk_root=Path(memory.__file__).parents[1];portable=q["portable_source_authentication"]["sdk_source_sha256"]
        for name,expected in portable.items():
            if not name.startswith("tabpfn/") or sha(sdk_root/Path(name).relative_to("tabpfn"))!=expected:raise RuntimeError("authenticated full SDK source changed: "+name)
        report["authenticated_sdk_python_sources"]=len(portable)
        if q.get("status")!="batch_screen_pass" or q.get("qualified_batch_sizes")!=[4096,6250,10000]:raise ValueError("complete frozen large-batch proof required")
        context=load_context(banks,20261111,4096);case=load_heldout(banks,0);query=case.obs.reshape(1,33)
        if q["context_theta"]!=array_descriptor(context.theta) or q["context_obs"]!=array_descriptor(context.obs) or q["query"]!=array_descriptor(query):raise ValueError("qualification bank/query binding mismatch")
        prefix_run=Path(prefix_run)
        if sha(prefix_run/"manifest.json")!=q["prefix_manifest_sha256"] or sha(prefix_run/"samples.npz")!=q["prefix_samples_sha256"]:raise ValueError("authenticated accepted-prefix source required")
        with np.load(prefix_run/"samples.npz",allow_pickle=False) as source:values=source["samples"].copy()
        if values.shape!=(50000,21) or not support_mask(values).all():raise ValueError("supported50k prefix source required")
        indices=np.linspace(0,49999,10000,dtype=int)
        queries=np.concatenate((np.repeat(query,10000,axis=0),values[indices,:20]),axis=1).astype(np.float32)
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1 or torch.get_float32_matmul_precision()!="highest" or torch.backends.cuda.matmul.allow_tf32:raise RuntimeError("one visible FP32 highest/TF32False GPU required")
        seed=derive_seed(20261111,"passive-prefill-allocator",case=0);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);np.random.seed(seed);random.seed(seed)
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS","16")))
        initial_rng=utilities.rng_identity(utilities.rng_state(torch));baseline=guard_baseline_source()
        source_hashes={str(Path(module.__file__).name):sha(module.__file__) for module in (memory,transformer,layer,multi_head_attention,mlp)}
        report.update(seed=seed,qualification_sha256=sha(qualification),checkpoint_sha256=CHECKPOINT_SHA256,context_theta=array_descriptor(context.theta),context_obs=array_descriptor(context.obs),
            query=array_descriptor(query),query_prefix_indices=indices.tolist(),source_hashes=source_hashes,gqa_module_sha256=execution.execution_source_sha256(),
            cpu_threads=torch.get_num_threads(),cpu_affinity=sorted(os.sched_getaffinity(0)),device_name=torch.cuda.get_device_name(0),
            after_cuda_init={"allocator_environment":os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),"cuda_initialized":bool(getattr(torch.cuda,"_initialized",False))})
        def ensure_time(seconds=15):
            if time.perf_counter()+seconds>=deadline:raise TimeoutError("bounded passive diagnostic deadline")
        def capture128(engine):
            captured=[]
            for member in range(8):
                before=active["patched_call_count"];prediction=engine.predict_component(member,queries[:128],capture_member=False);check_prediction(prediction,active,before);captured.append(prefix_prediction(prediction));del prediction
            return captured
        with finite_cdf_compatibility() as active,observe_sdk(memory,transformer,layer,multi_head_attention,mlp,torch) as observer:
            engine=execution.GQAHybridStageExecutor(checkpoint,cdf_descriptor=active,query_batch_size=10000)
            report["allocator"].append(allocator_snapshot(torch,"before_initial_prepare"));ensure_time(30)
            with observer.label("initial_prepare_unprofiled"):
                torch.cuda.synchronize();t0=time.perf_counter();prepared=engine.prepare(context.theta.astype(np.float32),context.obs.astype(np.float32),20);torch.cuda.synchronize()
            report["initial_prepare"]={"wall_seconds":time.perf_counter()-t0,"prepared":prepared,"layer_factors_by_member":[factors(model) for model in engine.regressor.executor_.models]}
            report["allocator"].append(allocator_snapshot(torch,"after_initial_prepare"))
            with observer.label("reference128_capture_separate"):reference=capture128(engine)
            for position,rows in enumerate(QUERY_SIZES):
                ensure_time(20);entry={"order_index":position,"query_rows_per_member":rows,"members":[]};phase="query-"+str(position)+"-N"+str(rows)
                with observer.label(phase):
                    for member in range(8):
                        before=active["patched_call_count"];torch.cuda.synchronize();t0=time.perf_counter()
                        prediction=engine.predict_component(member,queries[:rows],capture_member=False);torch.cuda.synchronize();seconds=time.perf_counter()-t0
                        check_prediction(prediction,active,before);candidate=prefix_prediction(prediction)
                        entry["members"].append({"ensemble_index":member,"wall_seconds_capture_free":seconds,"prefix128_parity":compare_prefix(reference[member],candidate,utilities,torch)})
                        del prediction,candidate
                entry["wall_seconds_sum_members"]=sum(row["wall_seconds_capture_free"] for row in entry["members"])
                entry["all8_prefix128_passed"]=all(row["prefix128_parity"]["passed"] for row in entry["members"]);report["query_timings"].append(entry)
                report["allocator"].append(allocator_snapshot(torch,phase+"-after"));report["elapsed_seconds"]=time.perf_counter()-start;write_json(output/"manifest.json",report)
            engine.close();gc.collect();report["allocator"].append(allocator_snapshot(torch,"closed_after_large_history_no_cache_release"));ensure_time(30)
            with observer.label("fresh_prepare_after_query_history_unprofiled"):
                torch.cuda.synchronize();t0=time.perf_counter();prepared=engine.prepare(context.theta.astype(np.float32),context.obs.astype(np.float32),20);torch.cuda.synchronize()
            report["fresh_prepare_after_query_history"]={"wall_seconds":time.perf_counter()-t0,"prepared":prepared,"layer_factors_by_member":[factors(model) for model in engine.regressor.executor_.models]}
            with observer.label("fresh_reference128_capture_separate"):fresh=capture128(engine)
            report["fresh_prefill_prefix128_parity"]=[compare_prefix(a,b,utilities,torch) for a,b in zip(reference,fresh)]
            report["allocator"].append(allocator_snapshot(torch,"after_fresh_prepare_and_reference"));engine.close();gc.collect();ensure_time(35)
            with observer.label("prefill_profiled_separate"):
                _,trace=profile_call(lambda:engine.prepare(context.theta.astype(np.float32),context.obs.astype(np.float32),20),output/"prefill.trace.json",torch)
            report["traces"].append({"kind":"prefill_profiled_separate",**trace});ensure_time(15)
            with observer.label("query10000_member0_profiled_separate"):
                before=active["patched_call_count"];prediction,trace=profile_call(lambda:engine.predict_component(0,queries,capture_member=False),output/"query10000-member0.trace.json",torch)
                check_prediction(prediction,active,before);del prediction
            report["traces"].append({"kind":"query10000_member0_profiled_separate",**trace});engine.close();engine=None
            report["allocator"].append(allocator_snapshot(torch,"after_all_models_closed"))
        report["observer_restorations"]=observer.restorations;report["observer_overflow"]=observer.overflow
        write_json(output/"passive-events.json",{"records":observer.records,"restorations":observer.restorations,"overflow":observer.overflow})
        report["passive_events_sha256"]=sha(output/"passive-events.json");report["compatibility_adapter"]=dict(active)
        report["rng_preserved"]=initial_rng==utilities.rng_identity(utilities.rng_state(torch))
        report["sources_unchanged"]=all(sha(sdk_root/Path(name).relative_to("tabpfn"))==expected for name,expected in portable.items()) and sha(__file__)==report["source_sha256"] and baseline==guard_baseline_source() and source_hashes=={str(Path(module.__file__).name):sha(module.__file__) for module in (memory,transformer,layer,multi_head_attention,mlp)}
        passed=report["rng_preserved"] and report["sources_unchanged"] and active["original_alias_restored"] and observer.overflow==0 and all(row["exact_descriptor_restored"] for row in observer.restorations) and all(row["all8_prefix128_passed"] for row in report["query_timings"]) and all(row["passed"] for row in report["fresh_prefill_prefix128_parity"])
        report.update(status="complete" if passed else "failed",all_guards_passed=bool(passed))
    except BaseException as error:
        report.update(status="failed",error_type=type(error).__name__,error=str(error))
        if observer is not None:
            write_json(output/"passive-events.json",{"records":observer.records,"restorations":observer.restorations,"overflow":observer.overflow})
        raise
    finally:
        if engine is not None:engine.close()
        report["elapsed_seconds"]=time.perf_counter()-start;write_json(output/"manifest.json",report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("banks","prefix-run","checkpoint","qualification","ledger","output"):parser.add_argument("--"+name,required=True)
    parser.add_argument("--max-seconds",type=int,default=240)
    result=run(**vars(parser.parse_args()));print(json.dumps({key:result[key] for key in ("status","all_guards_passed","elapsed_seconds")},indent=2))
    if result["status"]!="complete":raise SystemExit(2)


if __name__=="__main__":main()
