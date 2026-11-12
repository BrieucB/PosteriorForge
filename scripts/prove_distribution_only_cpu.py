#!/usr/bin/env python3
"""CPU-only saved-neural-output proof for the private distribution-only API.

Built with PriorLabs-TabPFN. Fake executors yield authenticated retained neural
tensors; fitted target transforms are represented transparently by their saved
inverse-border outputs. No fit, model forward, checkpoint load or CUDA call.
Repeated rows test postprocessing shape only, never new neural coverage.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import pickle
import random
import statistics
import sys
import time
from types import SimpleNamespace

import numpy as np

from posteriorforge.artifacts import sha256_file, write_json
from posteriorforge.execution_distribution_only import execution_metadata, predict_distribution

REPO = Path(__file__).resolve().parents[1]
GOLDEN_MANIFEST_SHA256 = "7de3549d1ad0c90c11b81ed751b4c9a28dd51441e17ae841f9ff372a8cc63642"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(value).hexdigest()


def tensor_identity(value):
    array = value.detach().contiguous().numpy()
    return {"dtype": array.dtype.str, "shape": list(array.shape),
            "sha256": digest(array.tobytes()), "version": value._version}


class IdentityNumericPreprocessor:
    """Declared fake prepared feature transform: executor ignores features."""
    def transform(self, value):
        return value.to_numpy(dtype=np.float64, copy=True)


class RetainedInverseBorders:
    """Replay the observed inverse-border output, not a refitted transformer."""
    def __init__(self, standard, transformed):
        self.standard = np.array(standard, copy=True)
        self.transformed = np.array(transformed, copy=True)

    def inverse_transform(self, value):
        require(np.array_equal(value[:, 0], self.standard),
                "replayed target transform received different standard borders")
        return self.transformed.reshape(-1, 1).copy()


class RetainedNeuralExecutor:
    def __init__(self, query, outputs, configs):
        self.query = query
        self.outputs = outputs
        self.configs = configs
        self.calls = 0

    def iter_outputs(self, query, *, device, autocast):
        require(device == "cpu" and autocast is False, "CPU-only replay required")
        require(np.array_equal(query, self.query.astype(np.float64), equal_nan=True),
                "query feature values/order changed before retained executor")
        self.calls += 1
        yield from zip(self.outputs, self.configs)


def make_regressor(arrays, metadata, *, members=None, rows=None, repeated=False,
                   average_before_softmax=False):
    import torch
    import tabpfn.regressor as sdk
    from tabpfn.model.bar_distribution import FullSupportBarDistribution
    members = list(range(8)) if members is None else list(members)
    retained = metadata["member_rows"]
    count = retained if rows is None else rows
    require(repeated or count <= retained, "uncaptured neural rows cannot be invented")
    indices = np.arange(count) % retained if repeated else np.arange(count)
    query = arrays["query"][indices].copy()
    standard = arrays["member_0_to_borders"].copy()
    outputs = []
    configs = []
    for member in members:
        prefix = f"member_{member}_"
        require(np.array_equal(arrays[prefix + "to_borders"], standard),
                "different canonical standard borders")
        observed = metadata["members"][member]
        require(not observed["descending_borders"] and
                not arrays[prefix + "logit_cancel_mask"].any(),
                "this retained inverse-border replay requires observed nondecreasing unmasked fixtures")
        config = object.__new__(sdk.RegressorEnsembleConfig)
        config.target_transform = (RetainedInverseBorders(
            standard, arrays[prefix + "from_borders"])
            if observed["target_transform_observed"] else None)
        if config.target_transform is None:
            require(np.array_equal(arrays[prefix + "from_borders"], standard),
                    "untransformed member borders changed")
        outputs.append(torch.from_numpy(
            arrays[prefix + "raw_model_logits"][indices].copy()))
        configs.append(config)
    model = object.__new__(sdk.TabPFNRegressor)
    model.n_features_in_ = query.shape[1]
    model.categorical_features_indices = None
    model.preprocessor_ = IdentityNumericPreprocessor()
    model.bardist_ = FullSupportBarDistribution(torch.from_numpy(standard))
    model.renormalized_criterion_ = FullSupportBarDistribution(
        torch.from_numpy(arrays["physical_borders"].copy()))
    model.executor_ = RetainedNeuralExecutor(query, outputs, configs)
    model.device_ = "cpu"
    model.use_autocast_ = False
    model.forced_inference_dtype_ = torch.float32
    model.softmax_temperature = .9
    model.average_before_softmax = average_before_softmax
    model.interface_config_ = SimpleNamespace(FIX_NAN_BORDERS_AFTER_TARGET_TRANSFORM=True)
    return model, query


def rng_identity(torch, generators):
    return {
        "python": digest(pickle.dumps(random.getstate(), protocol=5)),
        "numpy": digest(pickle.dumps(np.random.get_state(), protocol=5)),
        "torch_cpu": tensor_identity(torch.get_rng_state())["sha256"],
        "named_cpu_generators": {name: tensor_identity(g.get_state())["sha256"]
                                 for name, g in generators.items()},
        # Reading this module attribute does not initialize/query a device.
        "cuda_initialized": vars(torch.cuda).get("_initialized", False),
    }


def input_identity(model, query):
    tensors = {f"raw_{i}": tensor_identity(value)
               for i, value in enumerate(model.executor_.outputs)}
    for owner, criterion in (("standard", model.bardist_),
                             ("physical", model.renormalized_criterion_)):
        for name, value in criterion.named_buffers():
            tensors[owner + "." + name] = tensor_identity(value)
        for name, value in vars(criterion).items():
            if hasattr(value, "_version"):
                tensors[owner + ".plain." + name] = tensor_identity(value)
    return {"query": {"dtype": query.dtype.str, "shape": list(query.shape),
                       "sha256": digest(query.tobytes())}, "tensors": tensors}


@contextmanager
def forbid_summaries(criterion):
    """Profile tripwire: no SDK/class/criterion method is replaced."""
    codes = {getattr(criterion, name).__func__.__code__: name
             for name in ("mean", "median", "mode", "icdf")}
    previous = sys.getprofile()
    def profile(frame, event, argument):
        if event == "call" and frame.f_code in codes and frame.f_locals.get("self") is criterion:
            raise AssertionError("forbidden criterion summary entered: " + codes[frame.f_code])
    require(previous is None, "exclusive profiling scope required")
    sys.setprofile(profile)
    try:
        yield
    finally:
        sys.setprofile(previous)


@contextmanager
def observe_translation(module):
    """The harness scopes a real capture alias; the private API never patches it."""
    active = module.translate_probs_across_borders
    records = []
    def record(logits, *, frm, to):
        output = active(logits, frm=frm, to=to)
        records.append({
            "translation_input": tensor_identity(logits),
            "from_borders": tensor_identity(frm),
            "to_borders": tensor_identity(to),
            "translated_probability": tensor_identity(output),
            "raw_mass": tensor_identity(output.sum(dim=-1, dtype=__import__("torch").float64)),
        })
        return output
    module.translate_probs_across_borders = record
    try:
        yield records
    finally:
        conflict = module.translate_probs_across_borders is not record
        module.translate_probs_across_borders = active
        require(not conflict, "capture alias changed concurrently")


def prove_case(model, query, *, stage, name, members, repeated, torch, sdk, generators):
    before_rng = rng_identity(torch, generators)
    before_inputs = input_identity(model, query)
    original_alias = type(model).predict
    original_code = original_alias.__code__
    with observe_translation(sdk) as original_capture:
        original = model.predict(query, output_type="full", quantiles=[])
    with observe_translation(sdk) as private_capture:
        with forbid_summaries(model.renormalized_criterion_):
            private = predict_distribution(model, query)
    require(set(private) == {"criterion", "logits"}, "unexpected private summary keys")
    require(original["criterion"] is private["criterion"] is model.renormalized_criterion_,
            "actual physical criterion identity changed")
    require(tensor_identity(original["logits"]) == tensor_identity(private["logits"]),
            "original/private logits differ bitwise")
    require(original_capture == private_capture and len(private_capture) == len(members),
            "live canonical-CDF/component-mass capture differs")
    require(input_identity(model, query) == before_inputs,
            "input tensor bits/version or criterion state changed")
    require(rng_identity(torch, generators) == before_rng and not before_rng["cuda_initialized"],
            "RNG state changed or CUDA initialized")
    require(type(model).predict is original_alias and original_alias.__code__ is original_code,
            "original SDK prediction alias/code changed")
    return {"stage_index": stage, "fixture": name, "rows": len(query),
            "members": list(members), "retained_neural_rows_repeated": repeated,
            "new_neural_coverage": False,
            "logits_bitwise_identical": True, "actual_criterion_same_object": True,
            "all_cpu_rng_states_unchanged": True, "cuda_never_initialized": True,
            "input_bits_and_tensor_versions_unchanged": True,
            "summary_entry_tripwire_passed": True,
            "translation_and_component_mass_capture_calls": len(private_capture),
            "logits": tensor_identity(private["logits"])}


def benchmark(model, query, *, repetitions):
    methods = [("sdk_full", lambda: model.predict(query, output_type="full", quantiles=[])),
               ("distribution_only_guarded", lambda: predict_distribution(model, query))]
    for _, method in methods:
        method()
    results = {name: [] for name, _ in methods}
    for repetition in range(repetitions):
        for name, method in (methods if repetition % 2 == 0 else methods[::-1]):
            started = time.perf_counter()
            method()
            results[name].append(time.perf_counter() - started)
    return [{"query_rows": len(query), "operation": name,
             "median_seconds": statistics.median(values),
             "minimum_seconds": min(values), "maximum_seconds": max(values),
             "repetitions": repetitions, "raw_samples_seconds": values}
            for name, values in results.items()]


def prove(*, goldens, output, threads=16, repetitions=11, pilot_manifest=None):
    import torch
    import tabpfn.regressor as sdk
    from posteriorforge.compat_cdf import finite_cdf_compatibility
    goldens = Path(goldens).resolve()
    output = Path(output).resolve()
    require(sha256_file(goldens / "manifest.json") == GOLDEN_MANIFEST_SHA256,
            "immutable original-control goldens required")
    manifest = json.loads((goldens / "manifest.json").read_text())
    require(manifest["status"] == "complete" and manifest["full21_original_control_captured"],
            "complete full21 original-control goldens required")
    builder_path = REPO / "scripts/build_execution_goldens.py"
    require(sha256_file(builder_path) == manifest["producer_sha256"],
            "golden verifier producer source changed")
    spec = importlib.util.spec_from_file_location("retained_golden_verifier", builder_path)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    identity = execution_metadata()
    require(not vars(torch.cuda).get("_initialized", False), "fresh CPU-only process required")
    torch.set_num_threads(threads)
    generators = {name: torch.Generator(device="cpu").manual_seed(seed)
                  for name, seed in (("component_uniform", 3175593578),
                                     ("head_proposal", 1681898578),
                                     ("head_acceptance", 179600625))}
    before_rng = rng_identity(torch, generators)
    before_sources = {name: sha256_file(Path(sdk.__file__).parent / name)
                      for name in identity["sdk_sources"]}
    original_alias = sdk.TabPFNRegressor.predict
    output.mkdir(parents=True, exist_ok=False)
    cases = []
    timings = []
    benchmark_arrays = benchmark_metadata = None
    with finite_cdf_compatibility() as descriptor:
        for stage in manifest["stages"]:
            index = stage["stage_index"]
            for entry in stage["payloads"]:
                builder.verify_payload(goldens, entry)
                with np.load(goldens / entry["file"], allow_pickle=False) as saved:
                    arrays = {name: saved[name].copy() for name in saved.files}
                metadata = entry["metadata"]
                selections = [list(range(8))] + [[member] for member in range(8)]
                for members in selections:
                    model, query = make_regressor(arrays, metadata, members=members)
                    cases.append(prove_case(
                        model, query, stage=index, name=entry["file"], members=members,
                        repeated=False, torch=torch, sdk=sdk, generators=generators))
                if index == 6 and entry["file"].endswith("main.npz"):
                    benchmark_arrays, benchmark_metadata = arrays, metadata
                print(json.dumps({"stage": index, "fixture": entry["file"],
                                  "proof_cases_completed": len(cases)}), flush=True)
        # Explicit postprocessing shape tests, using 128 actual retained neural
        # rows repeated eight times. This is not 1024 distinct neural queries.
        for members in ([0], [2], [5], list(range(8))):
            model, query = make_regressor(benchmark_arrays, benchmark_metadata,
                                          members=members, rows=1024, repeated=True)
            cases.append(prove_case(
                model, query, stage=6, name="stage-06-retained-rows-repeated-1024",
                members=members, repeated=True, torch=torch, sdk=sdk, generators=generators))
        # The tripwire must actually detect the SDK's real summary path.
        model, query = make_regressor(benchmark_arrays, benchmark_metadata, members=[0], rows=8)
        try:
            with forbid_summaries(model.renormalized_criterion_):
                model.predict(query, output_type="full", quantiles=[])
        except AssertionError as error:
            require("forbidden criterion summary entered" in str(error), "wrong tripwire failure")
        else:
            raise ValueError("summary tripwire failed to detect original SDK summaries")
        for count in (1, 16, 128, 1024):
            model, query = make_regressor(benchmark_arrays, benchmark_metadata,
                                          members=[0], rows=count, repeated=count > 128)
            timings.extend(benchmark(model, query, repetitions=repetitions))
    require(descriptor["original_alias_restored"] and
            sdk.TabPFNRegressor.predict is original_alias, "SDK aliases not restored")
    require(rng_identity(torch, generators) == before_rng, "proof changed RNG state")
    require({name: sha256_file(Path(sdk.__file__).parent / name) for name in before_sources} ==
            before_sources, "SDK sources changed")
    with (output / "cpu-timings.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(timings[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(timings)
    estimate = None
    if pilot_manifest:
        pilot = json.loads(Path(pilot_manifest).read_text())
        histogram = Counter()
        for event in pilot["stage_events"]:
            sampler = event["sampler"]
            if sampler.get("stage0_full_mixture_shared"):
                histogram[1] += 1
            else:
                for counts in sampler["neural_query_batch_sizes_by_member"].values():
                    histogram.update({int(rows): count for rows, count in counts.items()})
        require(sum(histogram.values()) == pilot["gqa_prediction_checks"]["verified_calls"],
                "actual prediction call histogram mismatch")
        baseline = {r["query_rows"]: r["median_seconds"] for r in timings
                    if r["operation"] == "sdk_full"}
        private = {r["query_rows"]: r["median_seconds"] for r in timings
                   if r["operation"] == "distribution_only_guarded"}
        grid = sorted(baseline)
        by_operation = {(r["query_rows"], r["operation"]): r["raw_samples_seconds"]
                        for r in timings}
        paired = {n: [a - b for a, b in zip(by_operation[n, "sdk_full"],
                                           by_operation[n, "distribution_only_guarded"])]
                  for n in grid}
        paired_medians = {n: statistics.median(values) for n, values in paired.items()}
        net = sum(count * float(np.interp(rows, grid,
                  [paired_medians[n] for n in grid])) for rows, count in histogram.items())
        separate_median_net = sum(count * float(np.interp(rows, grid,
                  [baseline[n] - private[n] for n in grid])) for rows, count in histogram.items())
        estimate = {"pilot_manifest_sha256": sha256_file(pilot_manifest),
                    "call_count": sum(histogram.values()), "full1024_calls": histogram[1024],
                    "call_row_histogram": dict(sorted(histogram.items())),
                    "interpolated_cpu_only_net_savings_seconds": net,
                    "interpolated_difference_of_separate_medians_seconds": separate_median_net,
                    "paired_cpu_savings_by_query_rows": {
                        n: {"median_seconds": paired_medians[n],
                            "q25_seconds": float(np.quantile(values, .25)),
                            "q75_seconds": float(np.quantile(values, .75)),
                            "minimum_seconds": min(values), "maximum_seconds": max(values),
                            "positive_pairs": sum(v > 0 for v in values),
                            "pair_count": len(values), "raw_difference_seconds": values}
                        for n, values in paired.items()},
                    "measurement_limits": "Alternating CPU pairs are timing diagnostics, not independent "
                                          "population samples or an E2E performance guarantee.",
                    "old_request_seconds": pilot["timings"]["posterior_request_seconds"],
                    "prior_summary_only_hypothesis_seconds": [16, 18],
                    "scope": "CPU fake-executor postprocessing medians interpolated by actual row count; "
                             "includes private guards, excludes neural/GPU execution; not an E2E measurement."}
    result = {
        "schema": "posteriorforge.distribution_only_cpu_proof.v1",
        "attribution": "Built with PriorLabs-TabPFN",
        "status": "passed_cpu_replay_only_not_integrated_or_gpu_qualified",
        "module_source_sha256": sha256_file(REPO / "src/posteriorforge/execution_distribution_only.py"),
        "proof_source_sha256": sha256_file(__file__),
        "golden_manifest_sha256": GOLDEN_MANIFEST_SHA256,
        "identity": identity, "case_count": len(cases), "cases": cases,
        "retained_raw_neural_row_coverage": "first128/main, stress8 and separately retained outlier fixtures",
        "repeated1024_is_postprocessing_shape_only": True,
        "target_transform_replay": "actual saved inverse-border outputs; fitted transformer objects were not retained",
        "feature_preprocessor": "declared identity numeric fixture; fake executor checks features but yields saved neural tensors",
        "summary_tripwire_negative_control_passed": True,
        "original_sdk_aliases_and_sources_restored": True,
        "original_sdk_class_method_never_replaced": True,
        "all_cpu_rng_states_unchanged": True, "cuda_never_initialized": True,
        "new_model_calls": 0, "new_gpu_calls": 0,
        "threads": threads, "cpu_measurements": timings,
        "timings_sha256": sha256_file(output / "cpu-timings.csv"),
        "potential_savings": estimate,
        "existing_engines_runner_or_evaluator_changed": False,
        "demonstrated_end_to_end_speedup": False,
        "qualified_for_production": False,
    }
    write_json(output / "manifest.json", result)
    return {"output": str(output), "case_count": len(cases),
            "manifest_sha256": sha256_file(output / "manifest.json"),
            "cpu_measurements": timings, "potential_savings": estimate}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goldens", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pilot-manifest")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--repetitions", type=int, default=11)
    print(json.dumps(prove(**vars(parser.parse_args())), indent=2))
