#!/usr/bin/env python3
"""Fresh all21 conditional-trim GQA/host proof with original measured-memory guard.

Built with PriorLabs-TabPFN. This is residency qualification of the frozen
hybrid policy, not another cache-only proof or posterior-accuracy claim.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import random
import time
import numpy as np

SCHEMA = "posteriorforge.execution_gqa_host_trim_qualification.v1"
PINNED_QUALIFICATION_SHA256 = "a83e4cfb925317aea7a4d576deee9e0337fb66bd3d3e80470f7e7102b4c4ea48"
HYBRID_QUALIFIER_SHA256 = "15e8e9ce844d3352e49534714f0e0569f42e970a3d9876630f877a346eaaf9b6"
BASE_QUALIFIER_SHA256 = "6c927a5c278c7fc54b556ac557ecef64f5e6338f5ae13bf466f3b1e6a4640f90"
FALLBACK_MAP = {10: [5]}
BASE_GQA_HOST_QUALIFIER_SHA256="4d76c9f2c5e95a69362ceb2044d050813bfdcc73fd0682acaccab3892cc604e3"
BASE_HOST_QUALIFIER_SHA256="0dc55766d5896e71bd2d9b5e0df6c9a7f09b459928317cac3c44557ab2a310f4"
WARM_HELPER_SHA256="b923906f24d53decd37c7a4e421d1fc44d34663863ce8d662c9c14a4dcff3ea3"


def frozen_helpers():
    loaded = []
    for filename, expected in (("qualify_execution_cache.py", BASE_QUALIFIER_SHA256), ("qualify_execution_hybrid.py", HYBRID_QUALIFIER_SHA256)):
        path = Path(__file__).with_name(filename)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected: raise RuntimeError("frozen qualification source changed")
        spec = importlib.util.spec_from_file_location("_host_"+path.stem, path)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); loaded.append(module)
    return tuple(loaded)


def validate_gqa_qualification(proof):
    required = {"schema": "posteriorforge.execution_gqa_hybrid_recovery.v1", "status": "gqa_hybrid_screen_pass", "qualified": True,
                "full21_gqa_hybrid_qualified": True, "full21_component_qualified": True, "full21_sampler_qualified": True,
                "full21_qualified": False, "cache_only_full21_qualified": False, "icdf_contract_version": 2,
                "fallback_map": {"10": [5]}}
    for key, expected in required.items():
        if proof.get(key) != expected: raise ValueError("full21 frozen hybrid qualification required: "+key)
    if sorted(row["stage_index"] for row in proof.get("stage_results", [])) != list(range(21)):
        raise ValueError("exact all21 selected proof rows required")



def prepare_warmed_baseline(torch,workspace_evidence,failed_host_proof):
    path=Path(__file__).with_name("qualify_execution_host_cache_warmed.py")
    if hashlib.sha256(path.read_bytes()).hexdigest()!=WARM_HELPER_SHA256:raise RuntimeError("frozen warm-baseline helper changed")
    original=Path(__file__).with_name("qualify_execution_host_cache.py")
    if hashlib.sha256(original.read_bytes()).hexdigest()!=BASE_HOST_QUALIFIER_SHA256:raise RuntimeError("frozen host proof spine changed")
    gqa_original=Path(__file__).with_name("qualify_execution_gqa_host_cache.py")
    if hashlib.sha256(gqa_original.read_bytes()).hexdigest()!=BASE_GQA_HOST_QUALIFIER_SHA256:raise RuntimeError("frozen GQA host proof spine changed")
    spec=importlib.util.spec_from_file_location("_composition_warm_baseline",path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    evidence=module.authenticate_workspace_evidence(workspace_evidence,failed_host_proof)
    if str(torch.__version__)!=evidence["torch_version"] or torch.version.cuda!=evidence["torch_cuda_version"] or torch.cuda.get_device_name(0)!=evidence["device_name"]:
        raise RuntimeError("workspace evidence Torch/CUDA/device identity mismatch")
    for name in ("CUBLAS_WORKSPACE_CONFIG","PYTORCH_CUDA_ALLOC_CONF"):
        if os.environ.get(name)!=evidence["environment"].get(name):raise RuntimeError("workspace environment differs")
    workspace=module.load_frozen("diagnose_cuda_workspace.py",module.WORKSPACE_PROBE_SHA256)
    warmup=module.warm_runtime(torch,workspace.snapshot)
    return warmup,{"warm_helper_source_sha256":WARM_HELPER_SHA256,"base_host_qualifier_source_sha256":BASE_HOST_QUALIFIER_SHA256,
                   "workspace_evidence_sha256":module.WORKSPACE_EVIDENCE_SHA256,"preserved_failed_host_proof_sha256":module.FAILED_HOST_PROOF_SHA256,
                   "official_sources":module.OFFICIAL_SOURCES}


def gqa_prediction(engine,routed,query):
    from posteriorforge.execution_gqa_host_cache import descriptor_restored
    result=routed.hybrid_prediction(engine,query)
    descriptors=[prediction["gqa_attention_descriptor"] for prediction in result["component_predictions"]]
    restored=all(descriptor_restored(descriptor) for descriptor in descriptors)
    if not restored:raise RuntimeError("GQA alias must be restored after every component prediction")
    result["gqa_execution"]={"all_predictions_restored":restored,"component_predictions":len(descriptors),
        "expanded_attention_calls":sum(descriptor["expanded_attention_calls"] for descriptor in descriptors)}
    return result



def final_predicates(report,active):
    stages=report.get("stage_results",[]);migrations=report.get("migrations",[])
    final=report.get("final_retention",{});initial=report.get("initial_global_cuda_allocated_bytes")
    return {"sources_unchanged":report.get("sources_unchanged") is True,
        "all21_initial_cpu_no_extra_cuda":report.get("all21_cpu_no_unexplained_cuda_allocations") is True,
        "cdf_alias_restored":active.get("original_alias_restored") is True,
        "zero_refits_42_warm_hits":report.get("zero_refits_verified") is True,
        "session_rng_preserved":report.get("session_rng_preserved") is True,
        "conditional_trim_exercised":report.get("conditional_trim_exercised") is True,
        "all_trim_events_valid":report.get("all_trim_events_valid") is True,
        "allocator_source_and_evidence_unchanged":report.get("allocator_source_and_evidence_unchanged") is True,
        "all_gqa_lifecycles_restored":report.get("all_gqa_lifecycles_restored") is True,
        "memory_contract_passed":report.get("memory_contract_passed") is True,
        "two_full_warm_scans":report.get("cyclic_full_scans",{}).get("passed") is True,
        "all21_stage_passes":len(stages)==21 and sorted(row["stage_index"] for row in stages if row.get("passed"))==list(range(21)),
        "all105_migration_hashes_aliases_layout":len(migrations)==105 and all(event.get("comparison",{}).get("passed") and event.get("comparison",{}).get("full_storage_bytes_equal") for event in migrations),
        "all21_final_cpu_no_extra_cuda":initial==8519680 and final.get("global_cuda_allocated_bytes")==initial and final.get("retained_stages")==21
            and len(final.get("stages",[]))==21 and all(row.get("state",{}).get("cuda_storage_bytes")==0 for row in final.get("stages",[]))}


def prediction_exact(left, right, torch):
    """Exact value/NaN replay diagnostic in addition to unchanged CDF/TV bounds."""
    def exact(a, b):
        if hasattr(a, "detach"): a = a.detach().cpu()
        if hasattr(b, "detach"): b = b.detach().cpu()
        if hasattr(a, "shape") and tuple(a.shape) != tuple(b.shape): return False
        if torch.is_tensor(a): return bool(((a == b) | (torch.isnan(a) & torch.isnan(b))).all())
        return np.array_equal(a, b, equal_nan=True)
    return bool(exact(left["logits"], right["logits"]) and all(
        exact(a[field], b[field]) for a, b in zip(left["members"], right["members"])
        for field in ("raw_logits", "from_borders", "to_borders", "translated_probabilities")))


def preprocessing_identity(stage):
    """Hash CPU fitted preprocessing/configs/arrays; never transform train twice."""
    from posteriorforge.execution_hybrid import PREPROCESSING_FIELDS
    executor = stage.regressor.executor_
    result = {"cached": {name: hashlib.sha256(pickle.dumps(getattr(executor, name), protocol=5)).hexdigest()
                          for name in ("preprocessors", "configs", "cat_ixs", "n_train_samples")}}
    if stage.uncached_regressor is not None:
        executor = stage.uncached_regressor.executor_
        result["uncached"] = {name: hashlib.sha256(pickle.dumps(getattr(executor, name), protocol=5)).hexdigest() for name in PREPROCESSING_FIELDS}
    return result





def logical_tensor_state(stage, torch):
    # Original stock fallback .to/.cpu() may replace buffer objects. Compare
    # logical named values across predictions; exact identities/aliases remain
    # mandatory across each individual migration of its fixed current state.
    from posteriorforge import execution_gqa_host_trim as host
    started=time.perf_counter();rows=[]
    for entry in host.tensor_inventory(stage,torch):
        tensor=entry["tensor"]
        paths=[path for path in entry["paths"] if "/<view_base>" not in path and "/<grad>" not in path]
        if not paths:continue
        value=tensor.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy().tobytes()
        digest=hashlib.sha256(value).hexdigest()
        rows.extend({"path":path,"dtype":str(tensor.dtype),"shape":list(tensor.shape),"sha256":digest,
                     "stride":list(tensor.stride()),"device":str(tensor.device)} for path in paths)
    return {"tensors":sorted(rows,key=lambda row:row["path"]),"snapshot_and_hash_seconds":time.perf_counter()-started}


def compare_logical_state(before,after):
    def values(state):return [{key:value for key,value in row.items() if key not in ("stride","device")} for row in state["tensors"]]
    same=values(before)==values(after)
    return {"passed":same,"logical_paths_shapes_dtypes_values_equal":same,
            "layout_and_device_equal_diagnostic":before["tensors"]==after["tensors"],
            "before_tensor_paths":len(before["tensors"]),"after_tensor_paths":len(after["tensors"]),
            "after_snapshot_and_hash_seconds":after["snapshot_and_hash_seconds"]}


def archive_logical_state(state,output,stage,label):
    from posteriorforge.artifacts import sha256_file
    path=output/("logical-state-stage-{stage}-{label}.json.gz".format(stage=stage,label=label))
    with gzip.open(path,"wt",encoding="utf-8",compresslevel=5) as handle:json.dump(state,handle,sort_keys=True,allow_nan=False)
    return {"file":path.name,"sha256":sha256_file(path),"snapshot_and_hash_seconds":state["snapshot_and_hash_seconds"]}


def scalar_model_contract(stage, utilities):
    result={"resident":utilities.model_contract(stage.regressor)}
    if stage.uncached_regressor is not None:result["uncached"]=utilities.model_contract(stage.uncached_regressor)
    return result


def valid_model_contract(contracts):
    return bool(contracts) and all(row.get("input_normalization_train_only_verified") is True
        and row.get("autocast") is False and row.get("forced_inference_dtype")=="torch.float32" for row in contracts.values())


def valid_prediction_probabilities(prediction, utilities):
    rows=[utilities.compare_probabilities(utilities._numpy(prediction["logits"].softmax(-1)),utilities._numpy(prediction["logits"].softmax(-1)))]
    rows.extend(utilities.compare_probabilities(utilities._numpy(row["translated_probabilities"]),utilities._numpy(row["translated_probabilities"])) for row in prediction["members"])
    return {"passed":all(row["passed"] for row in rows),"final":rows[0],"members":rows[1:]}


def audit_tensor_coverage(stage, torch):
    from posteriorforge import execution_cache as cache, execution_host_cache as host
    scanned={id(row["tensor"]) for row in host.tensor_inventory(stage,torch)}
    roots=list(stage.regressor.executor_.models)+[stage.regressor.model_,stage.regressor.bardist_,stage.regressor.renormalized_criterion_]
    if stage.uncached_regressor is not None:
        roots.extend([stage.uncached_regressor.model_,stage.uncached_regressor.bardist_,stage.uncached_regressor.renormalized_criterion_])
    known={id(tensor) for root in roots for _,_,tensor in cache._tensor_records(root,torch)}
    return {"passed":known.issubset(scanned),"known_registered_and_direct_model_tensor_objects":len(known),
            "recursively_scanned_tensor_objects":len(scanned),"known_tensor_objects_missing":len(known-scanned),
            "audit_scope":"actual all8 models, original base model, criteria and declared uncached model; nested Python dict/slots/containers scanned"}


def compact_copy_events(engine, output, start_index):
    """Archive complete identity/layout/full-storage-hash inventories outside Git."""
    from posteriorforge.artifacts import sha256_file
    records = []
    for index in range(start_index, len(engine.copy_events)):
        event = engine.copy_events[index]; path = output / "migration-{index:03d}-{operation}-stage-{stage_index}.json.gz".format(index=index, **event)
        with gzip.open(path, "wt", encoding="utf-8", compresslevel=5) as handle: json.dump(event, handle, sort_keys=True, allow_nan=False)
        records.append({"operation": event["operation"], "stage_index": event["stage_index"], "seconds": event["seconds"],
                        "copied_bytes": event["copied_bytes"], "comparison": event["comparison"], "timings_seconds":event.get("timings_seconds"), "inventory_file": path.name,
                        "inventory_sha256": sha256_file(path)})
    return records


def qualify(*, banks, prefix_run, checkpoint, ledger, qualification, workspace_evidence, failed_host_proof, allocator_evidence, output, max_seconds=2250,
            rss_limit_gib=100, safety_slack_gib=8, expected_all_state_gib=87):
    from posteriorforge.artifacts import runtime_identity, sha256_file, write_json
    from posteriorforge.reference_grid import _require_budgeted_gpu
    from posteriorforge import execution_gqa_host_trim as host
    if type(max_seconds) is not int or not 300 <= max_seconds <= 2250: raise ValueError("bounded deadline300..2250seconds required")
    if (rss_limit_gib, safety_slack_gib, expected_all_state_gib) != (100, 8, 87): raise ValueError("predeclared100GiB limit/8GiB slack/87GiB initial estimate required")
    budget = _require_budgeted_gpu(ledger)
    if budget["gpus"] != 1: raise RuntimeError("one reserved GPU required")
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter(); deadline = start+max_seconds; engine = None; active = None; initial_sources = None
    report = {**runtime_identity(), "schema": SCHEMA, "status": "running", "qualified": False,
              "full21_gqa_host_trim_qualified": False, "full21_cache_only_qualified": False, "posterior_vectors_generated": 0,
              "budget_binding": budget, "stage_results": [], "migrations": [], "attribution": "Built with PriorLabs-TabPFN",
              "scope": "new conditional-trim policy over GQA one-fallback plus host residency; full21 same-state device roundtrip and query A-to-B-to-A; no posterior accuracy claim"}
    write_json(output/"manifest.json", report)
    try:
        import torch
        from posteriorforge.banks import load_context, load_heldout, derive_seed
        from posteriorforge.science.analytical_benchmark_v2 import array_descriptor, build_problem
        from posteriorforge.inference import CHECKPOINT_SHA256, support_mask
        from posteriorforge.inference_cdf import guard_baseline_source
        from posteriorforge.gate import verify_run_binding
        from posteriorforge.compat_cdf import finite_cdf_compatibility
        utilities, hybrid_utilities = frozen_helpers(); host.require_frozen_sources()
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1: raise RuntimeError("exactly one visible CUDA GPU required")
        if torch.get_float32_matmul_precision() != "highest" or torch.backends.cuda.matmul.allow_tf32: raise RuntimeError("frozen FP32 highest/TF32False required")
        if sha256_file(checkpoint) != CHECKPOINT_SHA256: raise ValueError("immutable checkpoint required")
        if sha256_file(qualification) != PINNED_QUALIFICATION_SHA256: raise ValueError("exact authenticated hybrid full21 proof required")
        proof = json.loads(Path(qualification).read_text()); validate_gqa_qualification(proof)
        if proof["hybrid_module_sha256"] != host.PINNED_HYBRID_SHA256: raise ValueError("qualification hybrid source mismatch")
        context = load_context(banks, 20261111, 4096); case_a = load_heldout(banks, 0); case_b = load_heldout(banks, 1)
        query_a = case_a.obs.reshape(1, 33); query_b = case_b.obs.reshape(1, 33)
        for key, actual in (("context_theta", array_descriptor(context.theta)), ("context_obs", array_descriptor(context.obs)), ("query", array_descriptor(query_a))):
            if proof.get(key) != actual: raise ValueError("hybrid proof bank/query binding mismatch: "+key)
        prefix_run = Path(prefix_run); prefix_manifest = json.loads((prefix_run/"manifest.json").read_text())
        if prefix_manifest.get("status") != "complete" or prefix_manifest.get("variant") != "cdf_arithmetic_v1": raise ValueError("original completed corrected pilot prefixes required")
        verify_run_binding(prefix_manifest, context, case_a.obs, 20261111, 0)
        if sha256_file(prefix_run/"manifest.json") != proof["prefix_manifest_sha256"] or sha256_file(prefix_run/"samples.npz") != proof["prefix_samples_sha256"]:
            raise ValueError("hybrid proof prefix hashes mismatch")
        with np.load(prefix_run/"samples.npz", allow_pickle=False) as data: values = data["samples"].copy()
        if values.shape != (50000, 21) or not support_mask(values).all(): raise ValueError("complete supported prefix source required")
        positions = np.asarray(proof["fixed_prefix_row_indices"], dtype=int); prefixes = values[positions[:128]]
        theta, obs = context.theta.astype(np.float32), context.obs.astype(np.float32)
        problem = build_problem(); noise = [problem.parameter_order.index(name) for name in problem.prior.noise_names]
        seed = derive_seed(20261111, "execution-gqa-host-trim-roundtrip-qualification", case=0)
        np.random.seed(seed); random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "16")))
        initial_sources = {"baseline": guard_baseline_source(), "host_module_sha256": host.execution_source_sha256(),"base_gqa_host_module_sha256":host.composition.execution_source_sha256(),
                           "qualifier_source_sha256": sha256_file(__file__), "hybrid_module_sha256": host.hybrid.execution_source_sha256(),
                           "gqa_hybrid_module_sha256":host.stage_execution.execution_source_sha256(),"gqa_module_sha256":host.gqa.execution_source_sha256(),
                           "cache_module_sha256": host.cache.execution_source_sha256(), "mixture_module_sha256": host.mixture.execution_source_sha256()}
        report.update(sources=initial_sources, checkpoint_sha256=CHECKPOINT_SHA256, qualification_sha256=sha256_file(qualification),
                      qualification=str(Path(qualification).resolve()), scientific_fingerprint=problem.scientific_fingerprint,
                      parameter_order=list(problem.parameter_order), context_theta=array_descriptor(context.theta), context_obs=array_descriptor(context.obs),
                      query_a=array_descriptor(query_a), query_b=array_descriptor(query_b), prefix_manifest_sha256=proof["prefix_manifest_sha256"],
                      prefix_samples_sha256=proof["prefix_samples_sha256"], fixed_prefix_row_indices=positions[:128].tolist(),
                      seed=seed, rng_role="execution-gqa-host-trim-roundtrip-qualification", thresholds=dict(utilities.THRESHOLDS), row_counts=[128,16,8],
                      memory_contract={"rss_limit_bytes":100*host.GIB,"safety_slack_bytes":8*host.GIB,"initial_all_state_estimate_bytes":87*host.GIB,
                                       "estimate_is_not_measured_fit_evidence":True,"rss_limit_includes_slack":True,"maximum_measured_rss_before_slack_bytes":92*host.GIB}, **host.hybrid.policy_metadata(FALLBACK_MAP))
        expected = {}; stage_rows = {}; preprocessing = {}; initial_states = {}; initial_contracts = {}; event_index = 0
        report["warmup"],warm_sources=prepare_warmed_baseline(torch,workspace_evidence,failed_host_proof)
        report.update(warm_sources);write_json(output/"warmup.json",report["warmup"])
        with finite_cdf_compatibility() as active:
            initial_rng = utilities.rng_identity(utilities.rng_state(torch))
            initial_cuda_allocated=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
            report["initial_global_cuda_allocated_bytes"]=initial_cuda_allocated
            if initial_cuda_allocated!=report["warmup"]["measured_live_workspace_bytes"]:raise RuntimeError("initial proof baseline differs from measured warm runtime")
            engine = host.HostStageCache(checkpoint, allocator_evidence=allocator_evidence,cdf_descriptor=active, fallback_map=FALLBACK_MAP, verify_copy_hashes=True)
            report["base_gqa_host_qualifier_source_sha256"]=BASE_GQA_HOST_QUALIFIER_SHA256;report["allocator_diagnostic_source_sha256"]=host.ALLOCATOR_DIAGNOSTIC_SHA256;report["allocator_evidence_sha256"]=host.ALLOCATOR_EVIDENCE_SHA256;report["allocator_policy"]=host.policy_metadata();report["libc_identity"]=dict(engine._libc_identity)
            report["initial_admission"] = engine.initial_admission
            if not engine.retention_enabled: raise RuntimeError("fixed all-stage admission refused; no all21 retention claim")
            # First sweep materializes every stage once, authenticates query A,
            # then retains all21 simultaneously on CPU before any second sweep.
            for stage in range(21):
                if time.perf_counter()+75 >= deadline: raise TimeoutError("bounded first-sweep deadline")
                report.update(active_stage=stage, phase="initial_prepare_and_offload"); write_json(output/"manifest.json", report)
                rng_before = utilities.rng_identity(utilities.rng_state(torch)); t0 = time.perf_counter()
                prepared = engine.prepare(theta, obs, stage)
                owned = engine._entries[engine._active]["stage"]
                preprocessing[stage] = preprocessing_identity(owned)
                coverage=audit_tensor_coverage(owned,torch)
                if not coverage["passed"]:raise RuntimeError("known model tensors missing from owned-state scanner")
                queries, stress = utilities.fixed_queries(query_a, prefixes, theta, stage, noise, problem.prior.bounds)
                expected[stage] = {"a": gqa_prediction(engine,hybrid_utilities, queries), "stress": gqa_prediction(engine,hybrid_utilities, stress)}
                expected[stage]["physical_borders"]=expected[stage]["a"]["criterion"].borders.detach().cpu().clone()
                row = {"stage_index":stage, "prepared":prepared, "first_prepare_and_prediction_seconds":time.perf_counter()-t0,"tensor_scanner_coverage":coverage,
                       "rng_preserved_initial_prepare_predict":rng_before==utilities.rng_identity(utilities.rng_state(torch))}
                initial_contracts[stage]=scalar_model_contract(owned,utilities)
                row["initial_model_contract"]=initial_contracts[stage]
                row["initial_model_contract_valid"]=valid_model_contract(initial_contracts[stage])
                initial_states[stage]=logical_tensor_state(owned,torch)
                row["logical_state_initial"]=archive_logical_state(initial_states[stage],output,stage,"initial")
                engine.close(); rng_after = utilities.rng_identity(utilities.rng_state(torch))
                if not engine.retention_enabled:raise RuntimeError("measured memory disabled retention: "+str(engine.disabled_reason))
                row["rng_preserved_initial_offload"] = rng_after == rng_before
                row["offload_cpu_only"] = engine._entries[(engine._context,stage)]["cpu_state"]["cuda_storage_bytes"] == 0
                row["cpu_fitted_state_unchanged"] = preprocessing[stage] == preprocessing_identity(owned)
                row["retention_still_enabled"] = engine.retention_enabled
                report["migrations"].extend(compact_copy_events(engine, output, event_index)); event_index = len(engine.copy_events)
                stage_rows[stage]=row; report["stage_results"].append(row); report["elapsed_seconds"]=time.perf_counter()-start
                write_json(output/"manifest.json", report)
                if not engine.retention_enabled: raise RuntimeError("measured memory refused retention; preserved partial proof only")
            report["all21_cpu_residency"] = engine.inspect_cache()
            report["all21_cpu_no_unexplained_cuda_allocations"]=report["all21_cpu_residency"]["global_cuda_allocated_bytes"]==initial_cuda_allocated
            if not report["all21_cpu_no_unexplained_cuda_allocations"]:raise RuntimeError("unexplained live CUDA allocation with all21 stages on CPU")
            if len(engine._entries) != 21 or report["all21_cpu_residency"]["retained_stages"] != 21: raise RuntimeError("all21 simultaneous retention required")
            # Each retained stage is restored without fitting. Query B is a
            # different existing observed case; query A is evaluated again.
            for stage in range(21):
                if time.perf_counter()+55 >= deadline: raise TimeoutError("bounded reload-sweep deadline")
                report.update(active_stage=stage, phase="reload_A_B_A"); write_json(output/"manifest.json", report)
                rng_before = utilities.rng_identity(utilities.rng_state(torch)); t0=time.perf_counter()
                prepared=engine.prepare(theta, obs, stage); owned=engine._entries[engine._active]["stage"]
                queries,stress=utilities.fixed_queries(query_a,prefixes,theta,stage,noise,problem.prior.bounds)
                changed=np.concatenate((np.repeat(query_b,16,axis=0),prefixes[:16,:stage]),axis=1).astype(np.float32)
                reloaded_contract=scalar_model_contract(owned,utilities)
                first=gqa_prediction(engine,hybrid_utilities,queries)
                b=gqa_prediction(engine,hybrid_utilities,changed)
                after_b_contract=scalar_model_contract(owned,utilities)
                after_b_snapshot=logical_tensor_state(owned,torch)
                after_b_state=compare_logical_state(initial_states[stage],after_b_snapshot)
                again=gqa_prediction(engine,hybrid_utilities,queries)
                stress_after=gqa_prediction(engine,hybrid_utilities,stress)
                comparisons={"A_before_offload_vs_after_reload":utilities.compare_predictions(expected[stage]["a"],first),
                             "A_after_B_vs_after_reload":utilities.compare_predictions(first,again),
                             "stress_before_offload_vs_after_reload":utilities.compare_predictions(expected[stage]["stress"],stress_after)}
                component=hybrid_utilities.component_report(expected[stage]["a"],first,owned,utilities)
                component.update(engine_id=host.ENGINE_ID,module_sha256=host.execution_source_sha256(),gqa_hybrid_module_sha256=host.PINNED_STAGE_SHA256)
                after_a_contract=scalar_model_contract(owned,utilities)
                after_a_snapshot=logical_tensor_state(owned,torch)
                after_a_state=compare_logical_state(initial_states[stage],after_a_snapshot)
                b_probabilities=valid_prediction_probabilities(b,utilities)
                borders_exact=bool(torch.equal(expected[stage]["physical_borders"],first["criterion"].borders.detach().cpu()) and torch.equal(expected[stage]["physical_borders"],again["criterion"].borders.detach().cpu()) and torch.equal(expected[stage]["physical_borders"],b["criterion"].borders.detach().cpu()))
                gqa_ok=all(prediction["gqa_execution"]["all_predictions_restored"] for prediction in (expected[stage]["a"],expected[stage]["stress"],first,b,again,stress_after))
                row=stage_rows[stage]; row.update(all_gqa_predictions_restored=gqa_ok,gqa_execution=first["gqa_execution"],physical_border_snapshot_exact=borders_exact,cache_hit=prepared["host_cache_hit"],comparisons=comparisons,
                     component_qualification=component, query_A_exact_replay=prediction_exact(expected[stage]["a"],first,torch),
                     query_A_after_B_exact_replay=prediction_exact(first,again,torch), query_B_distinct_input=not np.array_equal(changed,queries[:16],equal_nan=True),
                     query_B=array_descriptor(changed),query_B_valid_probabilities=b_probabilities,
                     model_contract_after_reload=reloaded_contract,model_contract_after_B=after_b_contract,model_contract_after_A_again=after_a_contract,
                     model_contracts_exact_and_valid=bool(row["initial_model_contract_valid"] and reloaded_contract==initial_contracts[stage] and after_b_contract==initial_contracts[stage] and after_a_contract==initial_contracts[stage]),
                     logical_tensor_state_after_B=after_b_state,logical_tensor_state_after_A_again=after_a_state,
                     logical_state_after_B=archive_logical_state(after_b_snapshot,output,stage,"after-B"),
                     logical_state_after_A_again=archive_logical_state(after_a_snapshot,output,stage,"after-A-again"),
                     reload_and_A_B_A_seconds=time.perf_counter()-t0,
                     cpu_fitted_state_unchanged_after_reload=preprocessing[stage]==preprocessing_identity(owned))
                engine.close(); report["trim_events"]=list(engine.trim_events)
                if not engine.retention_enabled:raise RuntimeError("unchanged parent admission disabled retention after conditional reclaim: "+str(engine.disabled_reason))
                row["rng_preserved_reload_A_B_A_offload"]=rng_before==utilities.rng_identity(utilities.rng_state(torch))
                row["passed"]=bool(row["cache_hit"] and all(c["passed"] for c in comparisons.values()) and component["passed"]
                    and row["rng_preserved_initial_prepare_predict"] and row["rng_preserved_initial_offload"] and row["rng_preserved_reload_A_B_A_offload"]
                    and row["all_gqa_predictions_restored"] and row["physical_border_snapshot_exact"] and row["query_A_exact_replay"] and row["query_A_after_B_exact_replay"] and row["query_B_distinct_input"] and b_probabilities["passed"]
                    and row["model_contracts_exact_and_valid"] and after_b_state["passed"] and after_a_state["passed"]
                    and row["offload_cpu_only"] and row["cpu_fitted_state_unchanged"] and row["cpu_fitted_state_unchanged_after_reload"] and row["retention_still_enabled"])
                report["migrations"].extend(compact_copy_events(engine,output,event_index));event_index=len(engine.copy_events)
                report["elapsed_seconds"]=time.perf_counter()-start;write_json(output/"manifest.json",report)
                del first,b,again,stress_after
                if not row["passed"]: raise RuntimeError("host roundtrip stage numerical/state/RNG proof failed")
            # Explicit second full scan after the first all21 reload scan.
            # Both follow cold preload:42 hits, zero new fits and zero evictions.
            cycle=[];misses_before=engine.counters["actual_prepare_miss_count"];hits_before=engine.counters["cache_hit_count"]
            for stage in range(21):
                if time.perf_counter()+30>=deadline:raise TimeoutError("bounded cyclic proof deadline")
                rng_before=utilities.rng_identity(utilities.rng_state(torch));t0=time.perf_counter()
                prepared=engine.prepare(theta,obs,stage);owned=engine._entries[engine._active]["stage"]
                queries,_=utilities.fixed_queries(query_a,prefixes[:16],theta,stage,noise,problem.prior.bounds)
                actual=gqa_prediction(engine,hybrid_utilities,queries)
                comparison=utilities.compare_predictions(utilities.slice_prediction(expected[stage]["a"],16),actual)
                source_row=next(row for row in proof["stage_results"] if row["stage_index"]==stage)
                measured=source_row.get("cache_after",source_row.get("cache_telemetry",{}))["unique_cuda_storage_bytes"]
                estimate=sum(float(prepared.get(name,0)) for name in ("fit_stock_cache_seconds","resident_move_seconds","uncached_fit_seconds"))
                engine.close();report["trim_events"]=list(engine.trim_events)
                if not engine.retention_enabled:raise RuntimeError("unchanged parent admission disabled retention in second warm scan")
                rng_equal=rng_before==utilities.rng_identity(utilities.rng_state(torch))
                cycle.append({"stage_index":stage,"cache_hit":prepared["host_cache_hit"],"comparison":comparison,
                              "measured_source_base_cuda_storage_bytes":measured,"seconds":time.perf_counter()-t0,
                              "avoided_actual_first_prepare_seconds":estimate,"rng_preserved":rng_equal,"gqa_restored":actual["gqa_execution"]["all_predictions_restored"]})
                report["migrations"].extend(compact_copy_events(engine,output,event_index));event_index=len(engine.copy_events)
                del actual
            report["cyclic_full_scans"]={"sequence_after_cold_preload":list(range(21))*2,"first_scan_steps":[{"stage_index":row["stage_index"],"cache_hit":row["cache_hit"],"passed":row["passed"]} for row in report["stage_results"]],"second_scan_steps":cycle,
                "actual_prepare_miss_delta":engine.counters["actual_prepare_miss_count"]-misses_before,
                "total_hits_after_cold_preload":engine.counters["cache_hit_count"],"total_prepare_misses_after_cold_preload":engine.counters["actual_prepare_miss_count"]-21,
                "cache_hit_delta":engine.counters["cache_hit_count"]-hits_before,"evictions":0,
                "avoided_actual_first_prepare_seconds":sum(row["avoided_actual_first_prepare_seconds"] for row in cycle),
                "passed":all(row["cache_hit"] and row["comparison"]["passed"] and row["rng_preserved"] and row["gqa_restored"] for row in cycle)
                    and engine.counters["actual_prepare_miss_count"]==misses_before and engine.counters["cache_hit_count"]-hits_before==21}
            report["final_retention"]=engine.inspect_cache();report["host_execution"]=engine.metadata
            report["peak_cuda_allocated_bytes"]=torch.cuda.max_memory_allocated();report["peak_cuda_reserved_bytes"]=torch.cuda.max_memory_reserved()
            report["peak_rss_bytes"]=host.process_memory()["rss_peak_bytes"]
            report["memory_contract_passed"]=bool(report["peak_rss_bytes"]<=100*host.GIB and engine.retention_enabled and len(engine._entries)==21)
            report["trim_events"]=list(engine.trim_events)
            report["conditional_trim_exercised"]=len(engine.trim_events)>0
            report["all_trim_events_valid"]=all(not event["trigger_projection"]["passed"] and event["active_state_comparison"]["passed"] and event["rng_preserved"] and event["retention_enabled_after_parent"] for event in engine.trim_events)
            report["allocator_source_and_evidence_unchanged"]=host._sha(allocator_evidence)==host.ALLOCATOR_EVIDENCE_SHA256
            engine._require_trim_seal(check_files=True)
            report["gqa_lifecycle"]=dict(engine._gqa_totals)
            report["all_gqa_lifecycles_restored"]=engine._gqa_totals["prediction_calls"]==engine._gqa_totals["restored_prediction_calls"] and engine._gqa_totals["expanded_attention_calls"]>0
            report["session_rng_preserved"]=initial_rng==utilities.rng_identity(utilities.rng_state(torch))
            counters=engine.counters
            report["zero_refits_verified"]=counters["actual_prepare_miss_count"]==21 and counters["cache_hit_count"]==42 and counters["reload_count"]==42
            engine.shutdown();engine=None
            report["status"]="complete_pending_adapter_restore"
            write_json(output/"manifest.json",report)
        current={"baseline":guard_baseline_source(),"host_module_sha256":host.execution_source_sha256(),"base_gqa_host_module_sha256":host.composition.execution_source_sha256(),"qualifier_source_sha256":sha256_file(__file__),
                 "hybrid_module_sha256":host.hybrid.execution_source_sha256(),
                 "gqa_hybrid_module_sha256":host.stage_execution.execution_source_sha256(),"gqa_module_sha256":host.gqa.execution_source_sha256(),"cache_module_sha256":host.cache.execution_source_sha256(),"mixture_module_sha256":host.mixture.execution_source_sha256()}
        host.require_frozen_sources()
        report["sources_unchanged"]=current==initial_sources;report["compatibility_adapter"]=dict(active)
        report["final_predicates"]=final_predicates(report,active)
        report["failed_predicates"]=[key for key,value in report["final_predicates"].items() if not value]
        passed=all(report["final_predicates"].values())
        report.update(status="gqa_host_trim_screen_pass" if passed else "failed",qualified=passed,full21_gqa_host_trim_qualified=passed)
    except BaseException as error:
        report.update(status="failed",qualified=False,full21_gqa_host_trim_qualified=False,error_type=type(error).__name__,error=str(error))
        if active is not None: report["compatibility_adapter"]=dict(active)
        if engine is not None:
            report["host_execution_at_failure"]=engine.metadata;report["memory_at_failure"]=host.process_memory();report["trim_events"]=list(engine.trim_events)
        raise
    finally:
        if engine is not None: engine.shutdown()
        report["elapsed_seconds"]=time.perf_counter()-start;write_json(output/"manifest.json",report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("banks","prefix-run","checkpoint","ledger","qualification","workspace-evidence","failed-host-proof","allocator-evidence","output"):parser.add_argument("--"+name,required=True)
    parser.add_argument("--max-seconds",type=int,default=2250)
    args=vars(parser.parse_args());result=qualify(**args);print(json.dumps({key:result[key] for key in ("status","qualified","elapsed_seconds")},indent=2))
    if result["status"]!="gqa_host_trim_screen_pass":raise SystemExit(2)


if __name__=="__main__":main()
