#!/usr/bin/env python3
"""Retain corrected, unoptimized fixed-conditional control outputs, never posterior draws.

Built with PriorLabs-TabPFN. These CPU-replay fixtures do not qualify a changed
GPU backend, cache, precision or sampler. Such changes still need fresh GPU
pointwise qualification. Raw artifacts stay outside Git.
"""
from __future__ import annotations
import argparse
import hashlib
from importlib import import_module
import importlib.util
import json
import os
from pathlib import Path
import pickle
import threading
import time
import numpy as np

SCHEMA = "posteriorforge.execution_control_goldens.v1"
ATTRIBUTION = "Built with PriorLabs-TabPFN"
PREFIX_MANIFEST_SHA256 = "a0114fd74721cf9bed31e9131045e841f262db123c8f849e5cb3f15814a14f4b"
PREFIX_SAMPLES_SHA256 = "27e51ad1b1ee615fc921ee4f5c237155139304da3e7c193f7781a389cee6d53e"
_MISSING = object()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def array_info(value):
    a = np.ascontiguousarray(value)
    require(a.dtype.kind in "fiub", "numeric/bool golden arrays required")
    return {"shape": list(a.shape), "dtype": a.dtype.str,
            "sha256": hashlib.sha256(a.tobytes()).hexdigest()}


def config_info(config):
    return {"type": type(config).__module__+"."+type(config).__qualname__,
            "repr": repr(config), "pickle_sha256": hashlib.sha256(pickle.dumps(config, protocol=5)).hexdigest()}


def model_state_info(regressor):
    """Actual retained per-forward normalization/embedding state; absent is explicit."""
    model = regressor.model_
    fields = ("normalize_on_train_only", "normalize_x", "normalize_to_ranking",
              "normalize_by_used_features", "remove_outliers", "remove_outliers_sigma",
              "cache_trainset_representation", "feature_positional_embedding", "seed",
              "min_num_layers_layer_dropout", "num_layers")
    tensors = ("cached_embeddings", "mean_for_normalization", "std_for_normalization",
               "lower_for_outlier_removal", "upper_for_outlier_removal", "used_features")
    records = []
    for name, module in model.named_modules():
        scalars = {key: getattr(module, key) for key in fields
                   if isinstance(getattr(module, key, None), (bool, int, float, str))}
        state = {}
        for key in tensors:
            value = getattr(module, key, None)
            if hasattr(value, "detach"):
                state[key] = array_info(value.detach().cpu().contiguous().numpy())
            elif hasattr(module, key):
                state[key] = None
        if scalars or state:
            records.append({"module": name, "type": type(module).__name__, "settings": scalars, "tensors": state})
    return {"modules": records, "model_training": bool(model.training),
            "autocast": bool(regressor.use_autocast_),
            "forced_inference_dtype": str(regressor.forced_inference_dtype_),
            "embedding_hash_limitation": "Only actually retained tensor state is hashed; seed/mode/source bind embeddings that the unoptimized path does not retain."}


def capture_original_prediction(regressor, query, *, descriptor):
    """Observe original output tensors/transform results, restoring every alias."""
    from posteriorforge import execution_cache as capture_guard
    require(threading.current_thread() is threading.main_thread(), "exclusive main-thread capture required")
    module = import_module("tabpfn.regressor")
    active = capture_guard._require_active_cdf(module, descriptor)
    executor = regressor.executor_
    original_iter = executor.iter_outputs
    previous_iter = vars(executor).get("iter_outputs", _MISSING)
    original_transform = module._transform_borders_one
    raw = []; transforms = {}; translated = []
    def observed_iter(*args, **kwargs):
        for output, config in original_iter(*args, **kwargs):
            require(hasattr(output, "detach"), "original tensor output required")
            raw.append({"raw_model_logits": output.detach().cpu().clone(),
                        "configuration": config_info(config), "model_state": model_state_info(regressor)})
            yield output, config
    def observed_transform(*args, **kwargs):
        result = original_transform(*args, **kwargs)
        mask, descending, borders = result
        transforms[len(raw)-1] = {"mask": None if mask is None else np.asarray(mask).copy(),
                                 "descending": bool(descending), "returned_borders": np.asarray(borders).copy()}
        return result
    def observed_translate(logits, *, frm, to):
        result = active(logits, frm=frm, to=to)
        index = len(translated)
        item = raw[index]
        translated.append({"ensemble_index": index, **item,
                           "translation_input_logits": logits.detach().cpu().clone(),
                           "from_borders": frm.detach().cpu().clone(),
                           "to_borders": to.detach().cpu().clone(),
                           "translated_probabilities": result.detach().cpu().clone(),
                           "target_transform": transforms.get(index)})
        return result
    executor.iter_outputs = observed_iter
    module._transform_borders_one = observed_transform
    module.translate_probs_across_borders = observed_translate
    before = descriptor["patched_call_count"]
    try:
        prediction = regressor.predict(query, output_type="full", quantiles=[])
    finally:
        conflicts = (executor.iter_outputs is not observed_iter
                     or module._transform_borders_one is not observed_transform
                     or module.translate_probs_across_borders is not observed_translate)
        if previous_iter is _MISSING: vars(executor).pop("iter_outputs", None)
        else: executor.iter_outputs = previous_iter
        module._transform_borders_one = original_transform
        module.translate_probs_across_borders = active
        if conflicts: raise RuntimeError("golden observation aliases changed concurrently")
    require(len(raw) == len(translated) == 8 and descriptor["patched_call_count"]-before == 8,
            "all eight original corrected member paths must be observed")
    require(descriptor.get("executed_corrected_cdf_code_sha256") == descriptor.get("corrected_cdf_code_sha256"),
            "corrected helper execution fingerprint mismatch")
    prediction["golden_members"] = translated
    prediction["capture_aliases_restored"] = True
    prediction["corrected_cdf_call_delta"] = descriptor["patched_call_count"]-before
    return prediction


def pack_prediction(prediction, query, uniforms, heads, logical_rows, *, retained_member_rows=128):
    """No numerical recomputation of model output; only bounded CPU snapshots."""
    arrays = {"query": np.asarray(query), "uniforms": np.asarray(uniforms),
              "component_indices": np.asarray(heads), "logical_row_indices": np.asarray(logical_rows),
              "final_logits": prediction["logits"].detach().cpu().numpy(),
              "physical_borders": prediction["criterion"].borders.detach().cpu().numpy()}
    rows = len(query); keep = min(rows, retained_member_rows)
    require(arrays["final_logits"].shape[0] == rows and arrays["uniforms"].shape == (rows,)
            and arrays["component_indices"].shape == (rows,), "golden row ordering/shape mismatch")
    require(arrays["final_logits"].dtype == np.float32 and arrays["uniforms"].dtype == np.float32
            and arrays["query"].dtype == np.float32, "FP32 control queries/logits and fixture uniforms required")
    require(arrays["component_indices"].dtype == np.int64
            and arrays["logical_row_indices"].shape == (rows,)
            and np.all((arrays["component_indices"] >= 0) & (arrays["component_indices"] < 8)),
            "ordered int64 member indices0..7 required")
    require(np.isfinite(arrays["uniforms"]).all()
            and np.all((arrays["uniforms"] >= 0) & (arrays["uniforms"] < 1)),
            "production-domain fixture uniforms required")
    metadata = {"rows": rows, "member_rows": keep, "member_row_indices": np.asarray(logical_rows)[:keep].tolist(),
                "capture_aliases_restored": prediction["capture_aliases_restored"],
                "corrected_cdf_call_delta": prediction["corrected_cdf_call_delta"], "members": []}
    for index, member in enumerate(prediction["golden_members"]):
        require(member["ensemble_index"] == index, "golden member order changed")
        prefix = f"member_{index}_"
        for key in ("raw_model_logits", "translation_input_logits", "translated_probabilities"):
            value = member[key].detach().cpu().numpy()
            require(value.shape[0] == rows and value.dtype == np.float32, "FP32 ordered member output required")
            arrays[prefix+key] = value[:keep].copy()
        for key in ("from_borders", "to_borders"):
            arrays[prefix+key] = member[key].detach().cpu().numpy()
        transform = member["target_transform"]
        mask = np.zeros(arrays[prefix+"raw_model_logits"].shape[-1], dtype=bool) if transform is None or transform["mask"] is None else transform["mask"]
        arrays[prefix+"logit_cancel_mask"] = np.asarray(mask, dtype=bool)
        metadata["members"].append({"ensemble_index": index, "configuration": member["configuration"],
            "model_state": member["model_state"], "target_transform_observed": transform is not None,
            "descending_borders": False if transform is None else transform["descending"],
            "cancel_mask_present": transform is not None and transform["mask"] is not None})
    return arrays, metadata


def save_payload(path, arrays, metadata):
    from posteriorforge.artifacts import sha256_file
    for value in arrays.values(): array_info(value)
    with Path(path).open("xb") as stream: np.savez(stream, **arrays)
    return {"file": Path(path).name, "sha256": sha256_file(path),
            "arrays": {name: array_info(value) for name, value in arrays.items()}, "metadata": metadata}


def verify_payload(root, entry):
    from posteriorforge.artifacts import sha256_file
    path = Path(root)/entry["file"]
    require(path.parent.resolve() == Path(root).resolve(), "golden payload must stay inside declared directory")
    require(sha256_file(path) == entry["sha256"], "golden payload checksum mismatch")
    with np.load(path, allow_pickle=False) as saved:
        expected = {"query","uniforms","component_indices","logical_row_indices","final_logits","physical_borders"}
        expected.update(f"member_{i}_{key}" for i in range(8) for key in
                        ("raw_model_logits","translation_input_logits","translated_probabilities",
                         "from_borders","to_borders","logit_cancel_mask"))
        require(set(saved.files) == set(entry["arrays"]) == expected, "unexpected/missing golden payload arrays")
        for name in saved.files:
            require(array_info(saved[name]) == entry["arrays"][name], "golden array descriptor mismatch: "+name)
        rows = entry["metadata"]["rows"]
        require(saved["query"].shape[0] == rows and saved["final_logits"].shape[0] == rows
                and saved["uniforms"].shape == saved["component_indices"].shape == (rows,),
                "golden row order/shape mismatch")
        require(saved["final_logits"].dtype == saved["uniforms"].dtype == saved["query"].dtype == np.float32,
                "golden control output/query or uniform dtype changed")
        require(saved["query"].shape[1] == 33+entry["metadata"]["stage_index"],
                "golden query feature/target order mismatch")
        require(saved["component_indices"].dtype == saved["logical_row_indices"].dtype == np.int64
                and np.all((saved["component_indices"] >= 0) & (saved["component_indices"] < 8)),
                "golden member/logical-row indices changed")
        require(np.isfinite(saved["uniforms"]).all()
                and np.all((saved["uniforms"] >= 0) & (saved["uniforms"] < 1)),
                "golden uniform domain changed")
        keep = entry["metadata"]["member_rows"]
        require(type(keep) is int and 1 <= keep <= min(rows,128)
                and entry["metadata"]["member_row_indices"] == saved["logical_row_indices"][:keep].tolist()
                and [m["ensemble_index"] for m in entry["metadata"]["members"]] == list(range(8)),
                "golden member row/configuration order mismatch")
        for i in range(8):
            prefix=f"member_{i}_"
            raw=saved[prefix+"raw_model_logits"];effective=saved[prefix+"translation_input_logits"]
            require(raw.ndim==2 and raw.shape[0]==keep and effective.shape==raw.shape
                    and raw.dtype==effective.dtype==np.float32,
                    "golden member raw/effective logits mismatch")
            require(saved[prefix+"translated_probabilities"].shape==(keep,saved["final_logits"].shape[1])
                    and saved[prefix+"translated_probabilities"].dtype==np.float32
                    and len(saved[prefix+"from_borders"])==raw.shape[1]+1
                    and len(saved[prefix+"to_borders"])==saved["final_logits"].shape[1]+1
                    and saved[prefix+"logit_cancel_mask"].shape==(raw.shape[1],)
                    and saved[prefix+"logit_cancel_mask"].dtype==np.bool_,
                    "golden member probability/border/mask mismatch")
    return True


def mathematical_self_checks(torch, active, descriptor):
    """Unequal-width/zero-mass/same-border fixtures; no model forward."""
    from tabpfn.model.bar_distribution import FullSupportBarDistribution
    borders = torch.tensor([-2., -.5, 0., 3.], dtype=torch.float32)
    p = torch.tensor([[.2, 0., .8]], dtype=torch.float32)
    logits = p.log()
    target = torch.tensor([-2., -1., -.5, 0., 1., 3.], dtype=torch.float32)
    expected_cdf = ((target.numpy()[:,None]-borders.numpy()[:-1])/
                    np.diff(borders.numpy())).clip(0,1)@p.numpy()[0]
    expected = np.diff(expected_cdf)
    unequal = active(logits, frm=borders, to=target).detach().cpu().numpy()[0]
    same = active(logits, frm=borders, to=borders).detach().cpu().numpy()[0]
    require(np.allclose(unequal, expected, rtol=0, atol=1e-6)
            and np.allclose(same, p.numpy()[0], rtol=0, atol=1e-6),
            "independent unequal-width corrected translation math failed")
    zero = torch.tensor([[0., .2, .8]], dtype=torch.float32).log()
    criterion = FullSupportBarDistribution(borders)
    endpoint = float(criterion.icdf(zero[0], 0.))
    require(np.isnan(endpoint), "stock leading-zero/u0 behavior was silently repaired")
    require(descriptor.get("executed_corrected_cdf_code_sha256") == descriptor.get("corrected_cdf_code_sha256"),
            "math probe did not execute the corrected helper")
    return {"unequal_width_translation_passed": True, "same_border_translation_passed": True,
            "zero_mass_u0_original_nan_preserved": True, "production_sampler_changed": False}


def build(*, banks, prefix_run, checkpoint, ledger, output, max_seconds=1050, outlier_run=None):
    from posteriorforge.artifacts import sha256_file,write_json,runtime_identity
    from posteriorforge.reference_grid import _require_budgeted_gpu
    require(60 <= max_seconds <= 1050, "explicit 60..1050-second golden budget required")
    budget = _require_budgeted_gpu(ledger)
    require(budget["gpus"] == 1, "one reserved GPU required for original-control capture")
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter(); source_sha = sha256_file(__file__)
    result = {**runtime_identity(), "schema": SCHEMA, "status": "running", "attribution": ATTRIBUTION,
              "budget_binding": budget, "producer_sha256": source_sha, "stages": [], "stage_coverage": [],
              "full21_original_control_captured": False, "posterior_vectors_generated": 0,
              "cache_or_backend_qualified": False, "member_rows_retained": 128,
              "limitation": "CPU replay never replaces fresh GPU qualification after cache/backend/precision changes."}
    write_json(output/"manifest.json", result)
    descriptor = None
    try:
        import torch
        from tabpfn import TabPFNRegressor
        from posteriorforge import execution_cache as guard
        from posteriorforge.compat_cdf import finite_cdf_compatibility
        from posteriorforge.inference import CHECKPOINT_SHA256,frozen_regressor_settings,support_mask
        from posteriorforge.inference_cdf import guard_baseline_source
        from posteriorforge.banks import load_context,load_heldout,derive_seed
        from posteriorforge.gate import verify_run_binding
        from posteriorforge.science.analytical_benchmark_v2 import build_problem,array_descriptor
        require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "one visible CUDA device required")
        require(torch.get_default_dtype() == torch.float32 and torch.get_float32_matmul_precision() == "highest"
                and not torch.backends.cuda.matmul.allow_tf32, "original FP32/highest/TF32False required")
        require(sha256_file(checkpoint) == CHECKPOINT_SHA256, "pinned checkpoint required")
        guard._load_sdk()
        baseline_source = guard_baseline_source()
        context = load_context(banks, 20261111, 4096)
        query = load_heldout(banks, 0).obs.reshape(1,33)
        prefix_run = Path(prefix_run); pm = json.loads((prefix_run/"manifest.json").read_text())
        require(sha256_file(prefix_run/"manifest.json") == PREFIX_MANIFEST_SHA256
                and sha256_file(prefix_run/"samples.npz") == PREFIX_SAMPLES_SHA256, "immutable corrected-control prefixes required")
        verify_run_binding(pm, context, query, 20261111, 0)
        with np.load(prefix_run/"samples.npz", allow_pickle=False) as saved: prefixes = saved["samples"].copy()
        require(array_descriptor(prefixes) == pm["samples"] and support_mask(prefixes).all(), "invalid corrected prefix source")
        problem = build_problem(); indices = np.linspace(0,49999,1024,dtype=np.int64)
        theta = context.theta.astype(np.float32); observations = context.obs.astype(np.float32)
        joint = np.concatenate((observations,theta),axis=1)
        helper = Path(__file__).with_name("qualify_execution_cache.py")
        spec = importlib.util.spec_from_file_location("golden_original_control_helpers", helper)
        diagnostic = importlib.util.module_from_spec(spec);spec.loader.exec_module(diagnostic)
        uniform_seed = derive_seed(20261111,"golden-component-uniform",case=0)
        head_seed = derive_seed(20261111,"golden-component-index",case=0)
        uniforms = torch.rand((21,1024),generator=torch.Generator().manual_seed(uniform_seed),dtype=torch.float32).numpy()
        heads = torch.randint(0,8,(21,1024),generator=torch.Generator().manual_seed(head_seed)).numpy()
        global_seed = derive_seed(20261111,"golden-original-control",case=0)
        np.random.seed(global_seed);torch.manual_seed(global_seed);torch.cuda.manual_seed_all(global_seed)
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS","16")))
        optional = None
        if outlier_run:
            op = Path(outlier_run); om = json.loads((op/"manifest.json").read_text())
            require(om["case"] == 0 and om["context_seed"] == 20261111
                    and om["context_theta"] == pm["context_theta"] and om["context_obs"] == pm["context_obs"]
                    and om["query"] == pm["query"] and om["parameter_order"] == pm["parameter_order"]
                    and sha256_file(op/"samples.npz") == om["samples_sha256"], "outlier-prefix source mismatch")
            with np.load(op/"samples.npz",allow_pickle=False) as f: oa=f["samples"].copy()
            require(array_descriptor(oa)==om["samples"] and support_mask(oa).all(), "invalid outlier-prefix cohort")
            optional = {"values":oa, "row":43246, "manifest_sha256":sha256_file(op/"manifest.json"),
                        "samples_sha256":om["samples_sha256"], "target_excluded_from_query":True}
        result.update(checkpoint_sha256=CHECKPOINT_SHA256, scientific_fingerprint=problem.scientific_fingerprint,
            parameter_order=list(problem.parameter_order), context_theta=array_descriptor(context.theta),
            context_obs=array_descriptor(context.obs), query=array_descriptor(query), logical_row_indices=indices.tolist(),
            prefix_manifest_sha256=PREFIX_MANIFEST_SHA256,prefix_samples_sha256=PREFIX_SAMPLES_SHA256,
            fixed_uniform_seed=uniform_seed,fixed_member_index_seed=head_seed,global_model_seed=global_seed,
            fixed_random_role_scope="diagnostic replay only; not posterior draws",
            sources={"baseline":baseline_source,"SDKsourcehashes":dict(guard.PINNED_SDK_SHA256),
                     "capture_guard_module_sha256":guard.execution_source_sha256(),"helper_sha256":sha256_file(helper)},
            regressor_settings=frozen_regressor_settings(checkpoint),
            outlier_prefix_source=None if optional is None else {k:v for k,v in optional.items() if k!="values"})
        noise = [problem.parameter_order.index(n) for n in problem.prior.noise_names]
        kwargs = frozen_regressor_settings(checkpoint); kwargs["inference_precision"] = torch.float32
        with finite_cdf_compatibility() as descriptor:
            active = guard._require_active_cdf(import_module("tabpfn.regressor"),descriptor)
            result["mathematical_self_checks"] = mathematical_self_checks(torch,active,descriptor)
            for stage in range(21):
                if time.perf_counter()-started+30 >= max_seconds:
                    result["status"] = "incomplete"; break
                result["active_stage"] = stage;write_json(output/"manifest.json",result)
                regressor = TabPFNRegressor(**kwargs)
                regressor.fit(joint[:,:33+stage],joint[:,33+stage])
                main, stress = diagnostic.fixed_queries(query,prefixes[indices],theta,stage,noise,problem.prior.bounds)
                model_before = model_state_info(regressor)
                executor = regressor.executor_
                fit_context = [{"member_index":i,"X_train":array_info(x),"y_train":array_info(y),
                    "preprocessor_pickle_sha256":hashlib.sha256(pickle.dumps(pre,protocol=5)).hexdigest(),
                    "categorical_indices":list(cat)}
                    for i,(x,y,pre,cat) in enumerate(zip(executor.X_trains,executor.y_trains,executor.preprocessors,executor.cat_ixs))]
                require(len(fit_context)==8, "all eight original fit-transform contexts required")
                stage_record = {"stage_index":stage,"parameter":problem.parameter_order[stage],
                    "target_training_mean":regressor.y_train_mean_,"target_training_sd":regressor.y_train_std_,
                    "raw_feature_count":33+stage,"fit_transform_context":fit_context,"model_before":model_before,
                    "softmax_temperature":regressor.softmax_temperature,"payloads":[]}
                for label, q, logical, u, h in (
                    ("main",main,indices,uniforms[stage],heads[stage]),
                    ("stress",stress,np.arange(-len(stress),0,dtype=np.int64),uniforms[stage,:len(stress)],heads[stage,:len(stress)])):
                    prediction = capture_original_prediction(regressor,q,descriptor=descriptor)
                    arrays, metadata = pack_prediction(prediction,q,u,h,logical)
                    metadata.update(label=label,stage_index=stage)
                    stage_record["payloads"].append(save_payload(output/f"stage-{stage:02d}-{label}.npz",arrays,metadata))
                    del prediction, arrays
                if stage==19 and optional is not None:
                    prefix=optional["values"][optional["row"],:19]
                    one=np.concatenate((query[0],prefix)).astype(np.float32)[None,:]
                    for count in (1,16):
                        q=np.repeat(one,count,axis=0)
                        prediction=capture_original_prediction(regressor,q,descriptor=descriptor)
                        arrays,metadata=pack_prediction(prediction,q,uniforms[stage,:count],heads[stage,:count],
                                                      np.full(count,optional["row"],dtype=np.int64))
                        metadata.update(label=f"rep2_outlier_prefix_{count}",stage_index=19,target_excluded_from_query=True)
                        stage_record["payloads"].append(save_payload(output/f"stage-19-outlier-{count}.npz",arrays,metadata))
                        del prediction,arrays
                stage_record["model_after"] = model_state_info(regressor)
                for entry in stage_record["payloads"]: verify_payload(output,entry)
                result["stages"].append(stage_record);result["stage_coverage"].append(stage)
                result["elapsed_seconds"]=time.perf_counter()-started;write_json(output/"manifest.json",result)
                regressor.model_.cpu();del regressor
            else:result["status"]="complete"
        result["compatibility_adapter"]=dict(descriptor)
        require(descriptor["original_alias_restored"] is True and descriptor["sdk_sources_unchanged"] is True,
                "corrected-control aliases/sources were not restored")
        require(guard_baseline_source()==baseline_source and sha256_file(__file__)==source_sha
                and sha256_file(helper)==result["sources"]["helper_sha256"], "golden producer/source changed")
        result["full21_original_control_captured"] = result["status"]=="complete" and result["stage_coverage"]==list(range(21))
        result["sources_unchanged"]=True
    except BaseException as error:
        result.update(status="failed",error_type=type(error).__name__,error=str(error))
        if descriptor is not None: result["compatibility_adapter"]=dict(descriptor)
        raise
    finally:
        result["elapsed_seconds"]=time.perf_counter()-started;write_json(output/"manifest.json",result)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("banks","prefix-run","checkpoint","ledger","output"):parser.add_argument("--"+key,required=True)
    parser.add_argument("--max-seconds",type=int,default=1050);parser.add_argument("--outlier-run")
    arguments=vars(parser.parse_args())
    result=build(**arguments)
    print(json.dumps({"schema":result["schema"],"status":result["status"],
                      "output":arguments["output"],"stage_coverage":result["stage_coverage"],
                      "full21_original_control_captured":result["full21_original_control_captured"],
                      "posterior_vectors_generated":0,"cache_or_backend_qualified":False,
                      "elapsed_seconds":result["elapsed_seconds"]},indent=2))
