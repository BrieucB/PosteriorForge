"""New one-GPU multi-request host-retention benchmark with frozen FIFO math.

Every request returns exactly50000 complete supported vectors. One CDF context
and immutable bank span the session; fresh FIFO and independent generators span
only one request. This entry point has its own source/schema/accounting identity.
Built with PriorLabs-TabPFN.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from contextlib import ExitStack
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import pickle
import subprocess
import time
from types import CodeType,ModuleType,SimpleNamespace
import numpy as np
from .artifacts import runtime_identity,sha256_file,write_json

SCHEMA="posteriorforge.inference.host_session.v1"
REQUEST_SCHEMA="posteriorforge.inference.host_request.v1"
VARIANT="execution_gqa_host_trim_fifo_v1"
COUNT=50000
SAMPLING_SOURCE_SHA256="23ccad8b5bbcdd7a48340b349d515ca870763359cfeb71333e02fc2b6a8aca3b"
SAMPLING_COMMIT="36594f45debeaf1921b10b7bc12a1daa8539ee37"
SAMPLING_BLOB="a910d353032575a3cd0193b9582d771955baf913"
HOST_PROOF_SHA256="18179d2d2161c7713050be99c03d3b8c8a92d82cd1b310fb69134e112a3bf8ba"
HOST_PRODUCER_SHA256="9e4043f5ab06b1f5bdfb4d30a91992420f368170a40151bdc9556ae56b80d443"
HOST_ENGINE_SHA256="1868d0e22906773fe62baeadbdb3124f9f2086ffdfd8cb918caca6aee3cb3e99"
PORTABLE_VERIFIER_SHA256="3234694dc1d2bb242d0995d465ddc69250cf67fb05e9127e7380fe0eb70ea6ee"
RUNTIME_HELPER_SHA256={
    "scripts/recover_gqa_hybrid_qualification.py":"83064f54712014b5c4c66569e854872caff03593cfdc337f2472a7b5723aea02",
    "src/posteriorforge/execution_proposal_buffer.py":"8fb2ba1fb63cd917378ff9bc8b6f70be7410862c509cd22cb2fb6f454d3d35fa"}
FROZEN_EXPORTS=("stage_major_wave","component_stage_sampler","worker_stream_seeds","proposal_buffer_policy","proposal_buffer_scope",
                "validate_proposal_buffer_record","guarded_proposal_buffer","validate_gqa_prediction")


def source_sha256():return sha256_file(__file__)


def selected_cases(cases=None,*,isolation_aba=False):
    if isolation_aba:
        if cases is not None:raise ValueError("isolation-aba selects fixed visits0,1,0; do not also supply cases")
        return (0,1,0)
    if cases is None:return tuple(range(8))
    if isinstance(cases,str):
        try:cases=tuple(int(value) for value in cases.split(","))
        except ValueError as error:raise ValueError("ordered unique heldout cases0..7 required") from error
    cases=tuple(cases)
    if not cases or any(type(value) is not int or not 0<=value<8 for value in cases) or cases!=tuple(sorted(set(cases))):
        raise ValueError("ordered unique heldout cases0..7 required")
    return cases


def load_frozen_sampling(archive,*,repo=None):
    """Execute immutable Git bytes, never the future mutable working-tree file."""
    repo=Path(repo or Path(__file__).resolve().parents[2]);archive=Path(archive)
    blob=subprocess.run(["git","-C",str(repo),"cat-file","blob",SAMPLING_BLOB],check=True,capture_output=True).stdout
    if hashlib.sha256(blob).hexdigest()!=SAMPLING_SOURCE_SHA256:raise RuntimeError("authenticated V4 sampling Git blob differs")
    archive.parent.mkdir(parents=True,exist_ok=True)
    with archive.open("xb") as stream:stream.write(blob)
    module=ModuleType("posteriorforge._host_frozen_sampling_v4");module.__package__="posteriorforge"
    # __file__ is solely a relative-helper locator. Executed bytecode comes from
    # the authenticated archive named by co_filename, not this working-tree path.
    module.__file__=str(repo/"src/posteriorforge/performance_runner.py")
    exec(compile(blob,str(archive.resolve()),"exec",dont_inherit=True),module.__dict__)
    helper_paths={name:repo/name for name in RUNTIME_HELPER_SHA256}
    for name,path in helper_paths.items():
        if not path.is_file() or sha256_file(path)!=RUNTIME_HELPER_SHA256[name]:raise RuntimeError("frozen runtime helper missing/changed; this entry point requires the reviewed source checkout: "+name)
    recovery=module.gqa_recovery_helper();recovery_context=recovery.context;recovery_context_code=recovery_context.__code__
    functions={name:getattr(module,name) for name in FROZEN_EXPORTS};identities={name:(fn,fn.__code__) for name,fn in functions.items()}
    def verify():
        if sha256_file(archive)!=SAMPLING_SOURCE_SHA256 or any(getattr(module,name) is not fn or fn.__code__ is not code for name,(fn,code) in identities.items()):
            raise RuntimeError("frozen sampling archive/callable changed")
        for name,path in helper_paths.items():
            if not path.is_file() or sha256_file(path)!=RUNTIME_HELPER_SHA256[name]:raise RuntimeError("frozen runtime helper source changed: "+name)
        if module.gqa_recovery_helper() is not recovery or recovery.context is not recovery_context or recovery_context.__code__ is not recovery_context_code:raise RuntimeError("runtime GQA descriptor guard identity changed")
        module.guarded_proposal_buffer()
    return SimpleNamespace(**functions,stream_roles=deepcopy(module.STREAM_ROLES),stream_dtypes=deepcopy(module.STREAM_DTYPES),verify=verify,source={"sha256":SAMPLING_SOURCE_SHA256,"git_commit":SAMPLING_COMMIT,"git_blob":SAMPLING_BLOB,
        "archive":str(archive.resolve()),"executed_source":"authenticated Git blob; working-tree file never read for execution",
        "requires_reviewed_source_checkout":True,"relative_runtime_helpers":dict(RUNTIME_HELPER_SHA256),"recovery_helper_scope":"runtime descriptor context predicate only; old live-ctime qualification verifier never called",
        "exported_helpers":list(FROZEN_EXPORTS),"old_run_execution_not_called":True})


def load_portable_verifier():
    path=Path(__file__).resolve().parents[2]/"scripts/verify_portable_gqa_recovery.py"
    if sha256_file(path)!=PORTABLE_VERIFIER_SHA256:raise RuntimeError("reviewed portable qualification verifier changed")
    spec=importlib.util.spec_from_file_location("_host_portable_gqa",path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def authenticate_host_proof(path,*,context,query0,query1,checkpoint_sha):
    from .science.analytical_benchmark_v2 import array_descriptor
    if sha256_file(path)!=HOST_PROOF_SHA256:raise ValueError("exact full21 conditional-trim proof required")
    proof=json.loads(Path(path).read_text())
    expected={"schema":"posteriorforge.execution_gqa_host_trim_qualification.v1","status":"gqa_host_trim_screen_pass","qualified":True,
        "full21_gqa_host_trim_qualified":True,"full21_cache_only_qualified":False,"checkpoint_sha256":checkpoint_sha,
        "context_theta":array_descriptor(context.theta),"context_obs":array_descriptor(context.obs),"query_a":array_descriptor(query0),"query_b":array_descriptor(query1),
        "initial_global_cuda_allocated_bytes":8519680}
    for key,value in expected.items():
        if proof.get(key)!=value:raise ValueError("host qualification binding mismatch: "+key)
    if len(proof.get("final_predicates",{}))!=14 or not all(value is True for value in proof["final_predicates"].values()) or proof.get("failed_predicates")!=[]:
        raise ValueError("all14 measured host-proof predicates required")
    if proof["sources"].get("host_module_sha256")!=HOST_ENGINE_SHA256 or proof["sources"].get("qualifier_source_sha256")!=HOST_PRODUCER_SHA256:
        raise ValueError("qualified host source/producer changed")
    counts=proof["host_execution"]["counters"]
    if any(counts.get(key)!=value for key,value in (("actual_prepare_miss_count",21),("cache_hit_count",42),("reload_count",42))):raise ValueError("qualified host cold/warm counters differ")
    return proof


def guarded_sampler(torch):
    from .upstream import load_baseline
    from importlib import import_module
    from .inference_cdf import UPSTREAM_SOURCE_SHA256
    load_baseline();module=import_module("npe_pfn.accept_reject_sampler");function=module.accept_reject_sample
    path=Path(module.__file__);raw=inspect.unwrap(function)
    if sha256_file(path)!=UPSTREAM_SOURCE_SHA256["accept_reject_sampler.py"]:raise RuntimeError("original whole-vector rejection source changed")
    compiled=compile(path.read_text(),str(path),"exec",dont_inherit=True)
    expected=next(code for code in compiled.co_consts if isinstance(code,CodeType) and code.co_name=="accept_reject_sample")
    if raw.__code__!=expected:raise RuntimeError("active original rejection body differs from pinned source")
    wrapper_code=function.__code__;raw_code=raw.__code__
    def verify():
        if module.accept_reject_sample is not function or function.__code__ is not wrapper_code or inspect.unwrap(function) is not raw or raw.__code__ is not raw_code:
            raise RuntimeError("original rejection callable identity changed")
    return function,verify



def guarded_prior(torch):
    from . import inference as original
    from .inference_cdf import BASELINE_INFERENCE_SHA256
    path=Path(original.__file__)
    if sha256_file(path)!=BASELINE_INFERENCE_SHA256:raise RuntimeError("original physical support source changed")
    compiled=compile(path.read_text(),str(path),"exec",dont_inherit=True)
    factory_code=next(code for code in compiled.co_consts if isinstance(code,CodeType) and code.co_name=="_prior")
    if original._prior.__code__!=factory_code:raise RuntimeError("active original prior factory differs")
    support_class=next(code for code in factory_code.co_consts if isinstance(code,CodeType) and code.co_name=="Support")
    expected=next(code for code in support_class.co_consts if isinstance(code,CodeType) and code.co_name=="check")
    prior=original._prior(torch);function=type(prior.support).check
    if function.__code__!=expected:raise RuntimeError("active original whole-vector support body differs")
    code=function.__code__
    def verify():
        if type(prior.support).check is not function or function.__code__ is not code:raise RuntimeError("original support callable identity changed")
    return prior,verify


def allocator_observation(torch):
    """Read actual allocator state without changing its configuration."""
    import os
    snapshot=torch.cuda.memory._snapshot()
    return {"PYTORCH_CUDA_ALLOC_CONF":os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
            "CUBLAS_WORKSPACE_CONFIG":os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "allocator_settings":deepcopy(snapshot.get("allocator_settings")),
            "observed_with":"torch.cuda.memory._snapshot().allocator_settings",
            "allocator_policy_changed_by_observer":False}


def rng_identity(torch,generators):
    import random
    result={name:hashlib.sha256(generator.get_state().numpy().tobytes()).hexdigest() for name,generator in generators.items()}
    result["default_torch_cpu"]=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
    result["default_torch_cuda"]=hashlib.sha256(b"".join(state.numpy().tobytes() for state in torch.cuda.get_rng_state_all())).hexdigest()
    result["numpy_global"]=hashlib.sha256(pickle.dumps(np.random.get_state(),protocol=5)).hexdigest()
    result["python_global"]=hashlib.sha256(pickle.dumps(random.getstate(),protocol=5)).hexdigest();return result


def isolation_comparison(requests,*,output):
    if len(requests)!=3 or [row["case"] for row in requests]!=[0,1,0]:raise ValueError("exact three-visit ABA result required")
    first,last=requests[0],requests[2]
    keys=("seed_streams","rejection","proposal_wave_sizes","sampling_counters","rng_before","rng_after")
    same={key:first[key]==last[key] for key in keys}
    fifo_keys=("request_scope_sha256","api_requested_sizes","actual_generated_wave_sizes","generated_count","original_api_delivered_count","remaining_unused_buffer_count","original_sampler_counts")
    same["fifo_scopes_and_counts"]=all(first["proposal_buffer"][key]==last["proposal_buffer"][key] for key in fifo_keys)
    same["array_payload_sha256"]=first["samples"]["sha256"]==last["samples"]["sha256"]
    same["npz_payload_sha256"]=first["samples_sha256"]==last["samples_sha256"]
    same["fresh_fifo_each_request"]=len({row["fifo_instance_identity"] for row in requests})==3
    same["fresh_sampling_generators_each_request"]=len({identity for row in requests for identity in row["generator_identities"].values()})==9
    result={"passed":all(same.values()),"predicates":same,"scope":"same case/repetition/streams and original counts; host cold/hit counts intentionally differ and remain separately reported"}
    if not same["array_payload_sha256"]:
        with np.load(Path(output)/first["relative_samples_file"],allow_pickle=False) as data:a=data["samples"].copy()
        with np.load(Path(output)/last["relative_samples_file"],allow_pickle=False) as data:b=data["samples"].copy()
        difference=np.abs(a-b);result["differences"]={"changed_entries":int(np.count_nonzero(a!=b)),"changed_rows":int(np.count_nonzero(np.any(a!=b,axis=1))),
            "max_absolute":float(difference.max()),"max_absolute_by_parameter":difference.max(axis=0).tolist()}
    return result


def run_session(*,banks,checkpoint,ledger,host_qualification,gqa_qualification,portable_batch_qualification,allocator_evidence,workspace_evidence,failed_host_proof,
                references,reference_grids,output,cases=None,isolation_aba=False,repetition=0):
    from .reference_grid import _require_budgeted_gpu
    from .inference import CHECKPOINT_SHA256,_prior,support_mask,frozen_regressor_settings
    from .inference_cdf import guard_baseline_source,reference_binding
    from .banks import load_context,load_heldout
    from .science.analytical_benchmark_v2 import array_descriptor,build_problem
    from .execution_host_session import HostExecutionSession,source_sha256 as adapter_sha
    from . import execution_gqa_host_trim as host_engine,execution_mixture as mixture,execution_cache as cache
    from .compat_cdf import finite_cdf_compatibility
    selected=selected_cases(cases,isolation_aba=isolation_aba)
    if type(repetition) is not int or repetition<0:raise ValueError("nonnegative repetition required")
    start=time.perf_counter();budget=_require_budgeted_gpu(ledger)
    if budget["gpus"]!=1:raise RuntimeError("host session is single-process one-GPU only")
    output=Path(output);output.mkdir(parents=True,exist_ok=False);initial_source=source_sha256();initial_adapter=adapter_sha();baseline=guard_baseline_source()
    report={**runtime_identity(),"schema":SCHEMA,"variant":VARIANT,"status":"running","session_complete":False,"all_eight_cases_complete":False,
        "requested_cases":list(selected),"isolation_aba_mode":isolation_aba,"requested_vectors_per_case":COUNT,"requested_session_total_vectors":COUNT*len(selected),
        "context_seed":20261111,"context_size":4096,"repetition":repetition,"query_batch_size":1024,"world_size":1,"rank":0,
        "runner_source_sha256":initial_source,"adapter_source_sha256":initial_adapter,"budget_binding":budget,"request_manifests":[],
        "performance_first_authorized":True,"absolute_accuracy_gate_required_for_performance":False,"original_gate_unchanged":True,
        "attribution":"Built with PriorLabs-TabPFN"};write_json(output/"manifest.json",report)
    adapter=None;descriptor=None;requests=[];buffers=[];generators_kept=[];shutdown=None;active_request=None;active_folder=None
    try:
        ops=load_frozen_sampling(output/"sources/sampling-v4.py");ops.verify();report["frozen_sampling_utility"]=ops.source
        if sha256_file(checkpoint)!=CHECKPOINT_SHA256:raise ValueError("immutable checkpoint mismatch")
        context=load_context(banks,20261111,4096);observed={case:load_heldout(banks,case) for case in set(selected)|{0,1}}
        qhost=authenticate_host_proof(host_qualification,context=context,query0=observed[0].obs.reshape(1,33),query1=observed[1].obs.reshape(1,33),checkpoint_sha=CHECKPOINT_SHA256)
        import torch
        if torch.is_autocast_enabled() or torch.is_autocast_enabled("cpu"):raise RuntimeError("outer autocast must be disabled")
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1 or torch.get_default_dtype()!=torch.float32 or torch.get_float32_matmul_precision()!="highest" or torch.backends.cuda.matmul.allow_tf32:
            raise RuntimeError("one visible strict FP32 GPU required")
        portable=load_portable_verifier();sdk_spec=importlib.util.find_spec("tabpfn")
        if sdk_spec is None or not sdk_spec.submodule_search_locations:raise RuntimeError("installed TabPFN source directory required")
        recovered=portable.load_verified_recovery(gqa_qualification,portable_batch_qualification,sdk_root=Path(next(iter(sdk_spec.submodule_search_locations))))
        source_inputs={name:{"path":str(Path(path).resolve()),"sha256":sha256_file(path)} for name,path in {"host_qualification":host_qualification,"gqa_qualification":gqa_qualification,"portable_batch_qualification":portable_batch_qualification,"allocator_evidence":allocator_evidence,"workspace_evidence":workspace_evidence,"failed_host_proof":failed_host_proof}.items()}
        report["source_inputs"]=source_inputs
        report["portable_qualification_verifier"]={"source_sha256":PORTABLE_VERIFIER_SHA256,"recovery_sha256":sha256_file(gqa_qualification),
            "source_bundle_manifest_sha256":sha256_file(portable_batch_qualification),"fresh_install_ctime_checked":False,"old_live_ctime_verifier_called":False}
        # Warm only the independently measured library workspace, preserving RNG.
        warm_path=Path(__file__).resolve().parents[2]/"scripts/qualify_execution_host_cache_warmed.py"
        if sha256_file(warm_path)!="b923906f24d53decd37c7a4e421d1fc44d34663863ce8d662c9c14a4dcff3ea3":raise RuntimeError("frozen runtime warm-baseline helper changed")
        spec=importlib.util.spec_from_file_location("_host_session_warm",warm_path);warm=importlib.util.module_from_spec(spec);spec.loader.exec_module(warm)
        evidence=warm.authenticate_workspace_evidence(workspace_evidence,failed_host_proof)
        if str(torch.__version__)!=evidence["torch_version"] or torch.version.cuda!=evidence["torch_cuda_version"] or torch.cuda.get_device_name()!=evidence["device_name"]:raise RuntimeError("workspace runtime identity differs")
        import os
        for name in ("CUBLAS_WORKSPACE_CONFIG","PYTORCH_CUDA_ALLOC_CONF"):
            if os.environ.get(name)!=evidence["environment"].get(name):raise RuntimeError("workspace allocator environment differs: "+name)
        workspace=warm.load_frozen("diagnose_cuda_workspace.py",warm.WORKSPACE_PROBE_SHA256)
        report["runtime_warmup"]=warm.warm_runtime(torch,workspace.snapshot)
        report["allocator_after_runtime_warmup"]=allocator_observation(torch)
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS","16")))
        prior_sampler,verify_sampler=guarded_sampler(torch);theta=context.theta.astype(np.float32);obs=context.obs.astype(np.float32);problem=build_problem()
        bindings={"checkpoint_sha256":CHECKPOINT_SHA256,"scientific_fingerprint":problem.scientific_fingerprint,"parameter_order":list(problem.parameter_order),
                  "context_theta":array_descriptor(context.theta),"context_obs":array_descriptor(context.obs)}
        reference_bindings={case:reference_binding(banks=banks,reference=Path(references)/f"case-{case}",reference_grid=Path(reference_grids)/f"case-{case}",case=case) for case in set(selected)}
        report["session_id"]=hashlib.sha256(json.dumps({"job":budget["job_id"],"output":str(output.resolve()),"cases":list(selected),"repetition":repetition,"source":initial_source,"context":bindings["context_theta"]},sort_keys=True,separators=(",",":")).encode()).hexdigest()
        report.update(**bindings,host_qualification_manifest_sha256=HOST_PROOF_SHA256,host_qualification_producer_sha256=HOST_PRODUCER_SHA256,
            host_execution=host_engine.execution_metadata(),direct_host_query_cap=1024,large_batch_composition_claimed=False)
        for key in ("checkpoint_sha256","scientific_fingerprint","parameter_order","context_theta","context_obs"):
            if recovered.get(key)!=bindings[key]:raise ValueError("recovered qualification actual-bank binding differs: "+key)
        with finite_cdf_compatibility() as descriptor, ExitStack() as owned_stages:
            adapter=HostExecutionSession(checkpoint,cdf_descriptor=descriptor,allocator_evidence=allocator_evidence,theta_context=theta,obs_context=obs)
            def shutdown_owned_stages():
                try:report["shutdown"]=adapter.shutdown()
                except BaseException as cleanup_error:
                    if adapter._shutdown_summary is not None:report["shutdown"]=deepcopy(adapter._shutdown_summary)
                    report["shutdown_error"]={"type":type(cleanup_error).__name__,"error":str(cleanup_error)}
                    raise
            owned_stages.callback(shutdown_owned_stages)  # Runs before CDF exit, including body failures.
            report["allocator_after_SDK_and_cache_construction"]=allocator_observation(torch)
            report["session_setup_seconds"]=time.perf_counter()-start
            report["setup_actual_stage_fits"]=adapter.counter_snapshot()["host"].get("actual_prepare_miss_count",0)
            for visit,case in enumerate(selected):
                report["active_visit"]=visit;report["active_case"]=case;write_json(output/"manifest.json",report)
                case_start=time.perf_counter();folder=output/f"visit-{visit:02d}-case-{case}";folder.mkdir(exist_ok=False)
                query=observed[case].obs.reshape(1,33);seeds=ops.worker_stream_seeds(20261111,case,repetition,0,1)
                torch.manual_seed(seeds["component_uniform"]);torch.cuda.manual_seed_all(seeds["component_uniform"])
                generators={name:torch.Generator(device="cpu").manual_seed(seed) for name,seed in seeds.items()};generators_kept.append(generators)
                rng_before=rng_identity(torch,generators);adapter.begin_request(case=case,query=query,seed_streams=seeds)
                prior,verify_support=guarded_prior(torch);cdf_before=descriptor["patched_call_count"]
                counters={"stage0_shared_predictions":0,"component_predictions":0,"head_candidates":0,"head_rejections":0};gqa_checks={};waves=[];stage_events=[]
                request={**runtime_identity(),**bindings,"schema":REQUEST_SCHEMA,"variant":VARIANT,"session_id":report["session_id"],"status":"running","visit_index":visit,"case":case,"repetition":repetition,
                    "requested_count":COUNT,"benchmark_total_count":COUNT,"end_to_end":True,"scientific_run":True,"count_contract":"exactly50000 complete returned vectors for this request",
                    "context_seed":20261111,"context_size":4096,"query":array_descriptor(query),"query_batch_size":1024,"microbatch":50000,"world_size":1,"rank":0,
                    "observation_preprocessing":"raw; TabPFN frozen preprocessing","context_filter":"no_filtering","max_iter_rejection":None,"log_density":"not_used",
                    "partial_vector_filtering":False,"clipping_or_repair_performed":False,"seed_streams":seeds,"rng_before":rng_before,"sampling_rng_policy":"independent_cpu_sampling_streams_model_rng_isolated",
                    "sampling_policy":{**mixture.execution_metadata(),"head_policy":"raw_mass_rejection_v1","uniform_assignment":"preassigned_per_stage_by_physical_row","head_retry_preserves_row_uniform":True,
                        "streams":{name:{"seed":seed,"role":ops.stream_roles[name],"dtype":ops.stream_dtypes[name],"device":"cpu","generator":"torch.Generator"} for name,seed in seeds.items()}},
                    "reference_binding":reference_bindings[case],"runner_source_sha256":initial_source,"adapter_source_sha256":initial_adapter,"frozen_sampling_utility_sha256":SAMPLING_SOURCE_SHA256,
                    "execution_engine":cache.execution_metadata(),"host_engine":host_engine.execution_metadata(),"host_qualification_manifest_sha256":HOST_PROOF_SHA256,"direct_host_query_cap":1024,"large_batch_composition_claimed":False,
                    "regressor_settings":{**frozen_regressor_settings(checkpoint),"fit_mode":"fit_with_cache"},"performance_first_authorized":True,"absolute_accuracy_gate_required_for_performance":False,
                    "original_gate_unchanged":True,"proposal_buffer_policy":ops.proposal_buffer_policy(1200),"fifo_stage_field_interpretation":"required_stage_preparations_for_generated_waves counts LOGICAL stage acquisitions here; actual fit/hit/reload counters are separate",
                    "attribution":"Built with PriorLabs-TabPFN"}
                active_request=request;active_folder=folder
                def progress(event,_events=stage_events,_waves=waves,_request=request,_folder=folder,_case_start=case_start):
                    _events.append({"proposal_round":len(_waves),**event,"host_acquire":deepcopy(adapter.last_acquire)})
                    _request.update(stage_events=_events,proposal_wave_sizes=_waves,elapsed_seconds=time.perf_counter()-_case_start);write_json(_folder/"manifest.json",_request)
                stage_draw=ops.component_stage_sampler(adapter,torch,generators,counters,prediction_check=lambda prediction,stage,member,_checks=gqa_checks:ops.validate_gqa_prediction(prediction,stage,member,cdf_descriptor=descriptor,report=_checks))
                def provider(size,_waves=waves,_query=query,_stage_draw=stage_draw,_progress=progress):
                    _waves.append(int(size));values=ops.stage_major_wave(adapter,theta,obs,_query,int(size),1024,
                        lambda prediction:prediction["criterion"].sample(prediction["logits"]).detach().cpu().numpy(),stage_draw=_stage_draw,progress=_progress)
                    return torch.from_numpy(values),None
                scope=ops.proposal_buffer_scope({**bindings,"query":array_descriptor(query)},context_seed=20261111,case=case,repetition=repetition,rank=0,world_size=1,seeds=seeds)
                fifo=ops.guarded_proposal_buffer().ProposalBuffer(provider,scope=scope,concatenate=lambda parts:torch.cat(parts,dim=0),world_size=1,rank=0);buffers.append(fifo)
                request["fifo_instance_identity"]=id(fifo);request["generator_identities"]={name:id(generator) for name,generator in generators.items()}
                write_json(folder/"manifest.json",request);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();request_start=time.perf_counter()
                samples,_,_=prior_sampler(proposal=fifo,accept_reject_fn=prior.support.check,num_samples=COUNT,max_sampling_batch_size=COUNT,max_iter_rejection=None,show_progress_bars=False,proposal_sampling_kwargs={})
                record=fifo.finalize(original_checked_count=prior.support.proposed,accepted_before_trim=prior.support.accepted,returned_count=len(samples),original_support_calls=prior.support.calls)
                record.update(source_unchanged=True,enabled=True,policy_sha256=request["proposal_buffer_policy"]["policy_sha256"]);ops.validate_proposal_buffer_record(record,world_size=1,rank=0,scope=scope)
                torch.cuda.synchronize();sample_finished=time.perf_counter();values=samples.detach().cpu().numpy().astype(np.float64)
                if values.shape!=(COUNT,21) or not support_mask(values).all():raise RuntimeError("incomplete/invalid whole-vector output; no repair")
                verify_support()
                verify_sampler();ops.verify();request_counters=adapter.end_request();expected_cdf=8*counters["stage0_shared_predictions"]+counters["component_predictions"]
                if descriptor["patched_call_count"]-cdf_before!=expected_cdf:raise RuntimeError("actual request corrected-CDF count mismatch")
                delta=request_counters["counter_delta"]
                if delta["host"].get("logical_prepare_count",0)!=21*len(waves) or delta["gqa"]["contexts"]!=counters["stage0_shared_predictions"]+counters["component_predictions"] or gqa_checks.get("verified_calls")!=delta["gqa"]["contexts"]:raise RuntimeError("host logical acquisitions/GQA execution counters mismatch")
                with (folder/"samples.npz").open("xb") as stream:np.savez(stream,samples=values)
                after=adapter.inspect_session();rss=after["host"]["memory"];rng_after=rng_identity(torch,generators)
                globals_preserved=all(rng_before[name]==rng_after[name] for name in ("default_torch_cpu","default_torch_cuda","numpy_global","python_global"))
                if not globals_preserved:raise RuntimeError("model/retention work changed global RNG outside independent sampling generators")
                if after["host"]["retained_stages"]!=21 or after["host"]["global_cuda_allocated_bytes"]!=8519680:raise RuntimeError("request did not retain all21 CPU stages with measured library-only GPU residual")
                request.update(status="sampling_complete_pending_session_footer",samples=array_descriptor(values),samples_file="samples.npz",relative_samples_file=str((folder/"samples.npz").relative_to(output)),samples_sha256=sha256_file(folder/"samples.npz"),
                    proposal_buffer=record,proposal_wave_sizes=waves,stage_events=stage_events,sampling_counters=dict(counters),gqa_prediction_checks=gqa_checks,
                    cdf_call_count_delta=descriptor["patched_call_count"]-cdf_before,expected_request_cdf_calls=expected_cdf,host_request_accounting=request_counters,
                    host_session_after_request=after,rng_after=rng_after,global_rng_preserved=globals_preserved,completed_stage_indices=list(range(21)),
                    rejection={"support_calls":prior.support.calls,"proposed":prior.support.proposed,"accepted_before_final_trim":prior.support.accepted,"returned":len(values)},
                    timings={"posterior_request_seconds":sample_finished-request_start,"case_total_seconds_so_far":time.perf_counter()-case_start,"actual_stage_fit_count":delta["host"].get("actual_prepare_miss_count",0),
                        "retained_stage_hits":delta["host"].get("cache_hit_count",0),"cold_request":delta["host"].get("actual_prepare_miss_count",0)>0,
                        "includes_all_logical_acquires_transfers_CPU_sampling_and_rejection":True},
                    memory={"request_gpu_peak_allocated_bytes":torch.cuda.max_memory_allocated(),"request_gpu_peak_reserved_bytes":torch.cuda.max_memory_reserved(),
                        "current_rss_bytes":rss["rss_bytes"],"process_cumulative_peak_rss_bytes":rss["rss_peak_bytes"],"cpu_peak_is_not_request_resettable":True})
                write_json(folder/"manifest.json",request);requests.append(request);report["request_manifests"].append(str((folder/"manifest.json").relative_to(output)))
                report.update(completed_sampling_requests=len(requests),elapsed_seconds=time.perf_counter()-start);write_json(output/"manifest.json",report)
                del samples,values
            report["session_counters"]=adapter.counter_snapshot();report["retention_before_shutdown"]=adapter.inspect_session();shutdown=adapter.shutdown();report["shutdown"]=shutdown
        if not descriptor["original_alias_restored"] or not descriptor["sdk_sources_unchanged"]:raise RuntimeError("CDF session restoration failed")
        if isolation_aba:
            report["isolation_aba"]=isolation_comparison(requests,output=output)
            if not report["isolation_aba"]["passed"]:raise RuntimeError("full50k A-to-B-to-A isolation failed; payloads and differences preserved for review")
        if source_sha256()!=initial_source or adapter_sha()!=initial_adapter or guard_baseline_source()!=baseline or sha256_file(host_qualification)!=HOST_PROOF_SHA256 or sha256_file(Path(portable.__file__))!=PORTABLE_VERIFIER_SHA256:raise RuntimeError("session source/qualification changed")
        for name,binding in source_inputs.items():
            if sha256_file(binding["path"])!=binding["sha256"]:raise RuntimeError("session authenticated input changed: "+name)
        ops.verify();verify_sampler();host_engine.require_frozen_sources()
        report.update(status="complete",session_complete=True,completed_sampling_requests=len(requests),returned_session_total_vectors=COUNT*len(requests),
            all_eight_cases_complete=not isolation_aba and selected==tuple(range(8)),compatibility_adapter=dict(descriptor),sources_unchanged=True,
            total_seconds=time.perf_counter()-start,amortized_selected_requests_seconds=(time.perf_counter()-start)/len(requests))
        if report["all_eight_cases_complete"]:report["amortized_eight_case_seconds"]=report["total_seconds"]/8
        for request,path in zip(requests,report["request_manifests"]):
            request.update(status="complete",session_footer="../manifest.json",session_cdf_call_count_total_at_footer=descriptor["patched_call_count"],compatibility_adapter=dict(descriptor),sources_unchanged=True,shutdown_released_all_stages=shutdown["shutdown_released_all_stages"])
            write_json(output/path,request)
        report["request_manifest_sha256"]={path:sha256_file(output/path) for path in report["request_manifests"]}
    except BaseException as error:
        report.update(status="failed",session_complete=False,error_type=type(error).__name__,error=str(error))
        if descriptor is not None:report["compatibility_adapter"]=dict(descriptor)
        if active_request is not None and active_request.get("status")=="running":
            active_request.update(status="failed",error_type=type(error).__name__,error=str(error));write_json(active_folder/"manifest.json",active_request)
        raise
    finally:
        if adapter is not None and not adapter._closed:
            try:report["shutdown"]=adapter.shutdown()
            except BaseException as cleanup_error:
                if adapter._shutdown_summary is not None:report["shutdown"]=deepcopy(adapter._shutdown_summary)
                report["shutdown_error"]={"type":type(cleanup_error).__name__,"error":str(cleanup_error)}
        report["elapsed_seconds"]=time.perf_counter()-start;write_json(output/"manifest.json",report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("banks","checkpoint","ledger","host-qualification","gqa-qualification","portable-batch-qualification","allocator-evidence","workspace-evidence","failed-host-proof","references","reference-grids","output"):
        parser.add_argument("--"+name,required=True)
    parser.add_argument("--cases");parser.add_argument("--isolation-aba",action="store_true");parser.add_argument("--repetition",type=int,default=0)
    result=run_session(**vars(parser.parse_args()));print(json.dumps({key:result.get(key) for key in ("schema","status","requested_cases","returned_session_total_vectors","total_seconds")},indent=2))


if __name__=="__main__":main()
