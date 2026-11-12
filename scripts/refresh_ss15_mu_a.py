#!/usr/bin/env python3
"""One separately labeled hybrid NPE-PFN/Gibbs refresh; Built with PriorLabs-TabPFN.

CPU only. The frozen SS15 50000-row cohort is an input, not regenerated.
Observations, references, truth and evaluation APIs are never loaded.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import pickle
import random
import re
import subprocess
import time

import numpy as np

from posteriorforge import conditional_mu_a as kernel
from posteriorforge.artifacts import runtime_identity, sha256_file, write_json
from posteriorforge.banks import derive_seed
from posteriorforge.science.analytical_benchmark_v2 import array_descriptor

SCHEMA = "posteriorforge.ss15_mu_a_refresh.v1"
VARIANT = "hybrid_ss15_mu_a_gibbs_once_v1"
ROLE = "ss15_mu_a_gibbs_once_v1"
COUNT = 50000
KERNEL_SHA = "850df8d6498f78c12500ffdf60c4c9126f5c1ab67bc29867002d4c991373d2e8"
CONTRACT_SHA = "932bbcb7efb95a9298af6668cf5f4a550c78bb52b6af3d42d667d031c24ca1a5"
CONTRACT_ORIGIN = "e600df6"
PARENT_MANIFEST_SHA = "28da396e0af6cf26009b4eb8feab8b75ed7a1d7a5add30440a35b38539da4f94"
PARENT_PAYLOAD_SHA = "f263f712d943d7916205b2f2b77f9243ac4ba3819f5f739c45f00710d5b4be47"
PARENT_ARRAY_SHA = "8c832a13961dc8f1c9b85dbda56b65ce3315730071c04b2d6ae3d5aff07386f1"
PARENT_PRODUCER_SHA = "4099da3c19366997ab0800e5430b5bb1c129b4b3e535f678b3d4652233303480"
PARENT_PRODUCER = "src/posteriorforge/performance_ss15.py"


def require(value, message):
    if not value:
        raise ValueError(message)


def state_sha(value):
    return hashlib.sha256(pickle.dumps(value, protocol=5)).hexdigest()


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def git_bytes(root, revision, name):
    require(isinstance(revision, str) and re.fullmatch("[0-9a-f]{40}", revision) is not None,
            "full40hex Git revision required")
    return subprocess.run(["git", "-C", str(root), "show", revision + ":" + name],
                          check=True, capture_output=True).stdout


def source_guard():
    root = Path(__file__).resolve().parents[1]
    contract = root / "docs/accuracy-ss15-mu-a-gibbs-v1.md"
    require(kernel.source_sha256() == KERNEL_SHA, "conditional kernel source changed")
    require(sha256_file(contract) == CONTRACT_SHA, "frozen mathematical contract changed")
    origin = subprocess.run(["git", "-C", str(root), "rev-parse", CONTRACT_ORIGIN],
                            check=True, capture_output=True, text=True).stdout.strip()
    require(hashlib.sha256(git_bytes(root, origin, str(contract.relative_to(root)))).hexdigest()
            == CONTRACT_SHA, "pre-cohort mathematical contract Git bytes differ")
    revision = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                             check=True, capture_output=True, text=True).stdout.strip()
    source = Path(__file__).resolve()
    require(hashlib.sha256(git_bytes(root, revision, str(source.relative_to(root)))).hexdigest()
            == sha256_file(source), "producer must be committed before cohort execution")
    return {"producer_sha256": sha256_file(source), "git_revision": revision,
            "kernel_sha256": KERNEL_SHA, "math_contract_sha256": CONTRACT_SHA,
            "math_contract_origin_commit": origin, "producer_committed_before_run": True}


def validate_parent(manifest, values, metadata):
    expected = {"schema": "posteriorforge.inference.ss15_fp16.v1",
                "variant": "accuracy_ss15_fp16_fifo_v1", "status": "complete",
                "requested_count": COUNT, "context_seed": 20261111, "case": 0,
                "repetition": 0, "runner_source_sha256": PARENT_PRODUCER_SHA,
                "scientific_fingerprint": metadata["scientific_fingerprint"],
                "parameter_order": metadata["physical_parameter_order"]}
    for key, value in expected.items():
        actual = manifest.get(key)
        require(type(actual) is type(value) and actual == value, "parent binding differs: " + key)
    require(isinstance(values, np.ndarray) and values.dtype == np.dtype("float64")
            and values.shape == (COUNT, 21), "strict50000 complete originalFloat64 rows required")
    descriptor = array_descriptor(values)
    require(descriptor == manifest.get("samples") and descriptor["sha256"] == PARENT_ARRAY_SHA,
            "parent physical array descriptor differs")
    require(metadata["target_index"] == 0 and metadata["unchanged_indices"] == list(range(1, 21)),
            "only physical mu_a may change")


def load_parent(path, metadata):
    path = Path(path).resolve()
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == PARENT_MANIFEST_SHA, "frozen parent manifest differs")
    manifest = json.loads(raw)
    require(manifest.get("samples_file") == "samples.npz", "fixed parent samples filename required")
    payload = path.with_name("samples.npz")
    require(sha256_file(payload) == PARENT_PAYLOAD_SHA == manifest.get("samples_sha256"),
            "frozen parent payload differs")
    with np.load(payload, allow_pickle=False) as archive:
        require(archive.files == ["samples"], "only complete physical samples expected")
        values = archive["samples"]
    validate_parent(manifest, values, metadata)
    root = Path(__file__).resolve().parents[1]
    require(hashlib.sha256(git_bytes(root, manifest.get("git_revision"), PARENT_PRODUCER)).hexdigest()
            == PARENT_PRODUCER_SHA, "executed parent producer Git bytes differ")
    return manifest, values, {"manifest_sha256": PARENT_MANIFEST_SHA,
                            "samples_sha256": PARENT_PAYLOAD_SHA,
                            "samples": array_descriptor(values),
                            "producer_sha256": PARENT_PRODUCER_SHA,
                            "git_revision": manifest["git_revision"]}


def refresh_once(values, seed):
    """Engineering-testable strict-count wrapper; never evaluates posterior accuracy."""
    require(isinstance(values, np.ndarray) and values.dtype == np.dtype("float64")
            and values.shape == (COUNT, 21), "strict50000 originalFloat64 quota required")
    original = array_descriptor(values)
    default_before = (copy.deepcopy(np.random.get_state()), random.getstate())
    rng = np.random.Generator(np.random.PCG64(seed))
    initial = copy.deepcopy(rng.bit_generator.state)
    mean, variance = kernel.conditional_parameters(values)
    started = time.perf_counter()
    result = kernel.refresh(values, rng)
    seconds = time.perf_counter() - started
    final = copy.deepcopy(rng.bit_generator.state)
    replay = np.random.Generator(np.random.PCG64(0))
    replay.bit_generator.state = copy.deepcopy(initial)
    normals = replay.standard_normal(COUNT)
    require(np.array_equal(result[:, 0], mean + np.sqrt(variance) * normals),
            "one-pass standard-normal replay differs")
    require(final == replay.bit_generator.state, "private RNG consumed more than one draw per row")
    require(not np.shares_memory(values, result) and result.shape == (COUNT, 21)
            and result.dtype == np.dtype("float64"), "fresh complete Float64 output required")
    require(values[:, 1:].tobytes() == result[:, 1:].tobytes(),
            "one of the other20 physical coordinates changed")
    require(array_descriptor(values) == original, "input cohort was mutated")
    # Reusing the source-derived validator checks all finite rows and original support.
    kernel.conditional_parameters(result)
    default_after = (np.random.get_state(), random.getstate())
    require(state_sha(default_before) == state_sha(default_after), "default NumPy/Python RNG changed")
    z_before = (values[:, 0] - mean) / np.sqrt(variance)
    z_after = (result[:, 0] - mean) / np.sqrt(variance)
    receipt = {
        "seed": int(seed), "role": ROLE, "bit_generator": "PCG64",
        "initial_private_state": initial, "final_private_state": final,
        "initial_private_state_sha256": json_sha(initial),
        "final_private_state_sha256": json_sha(final),
        "standard_normal_draws": array_descriptor(normals),
        "standard_normal_draw_count": COUNT, "independent_draw_replay_passed": True,
        "default_numpy_python_RNG_before_sha256": state_sha(default_before),
        "default_numpy_python_RNG_after_sha256": state_sha(default_after),
        "default_numpy_python_RNG_preserved": True,
        "torch_RNG_observation": "not imported or observed; CPU NumPy-only producer",
        "other20_columns_byte_identical": True, "input_array_unchanged": True,
        "finite_supported_complete_count": COUNT, "refresh_count_per_row": 1,
        "conditional_residual_diagnostic": {
            "scope": "model-only standardized full-conditional residual; not an accuracy gate",
            "before": {"mean": float(z_before.mean()), "sd": float(z_before.std(ddof=0))},
            "after": {"mean": float(z_after.mean()), "sd": float(z_after.std(ddof=0))}},
        "refresh_seconds": seconds}
    return result, receipt


def run(parent_manifest, output):
    producer_rng_before = state_sha((np.random.get_state(), random.getstate()))
    sources = source_guard()
    metadata = kernel.metadata()
    parent, values, lineage = load_parent(parent_manifest, metadata)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    record = {"schema": SCHEMA, "variant": VARIANT, "status": "running",
              "attribution": "Built with PriorLabs-TabPFN",
              "method": "hybrid NPE-PFN/Gibbs; ONE scalar mu_a full-conditional refresh",
              "pure_TFM_inference": False, "requested_count": COUNT,
              "context_seed": 20261111, "case": 0, "repetition": 0,
              "scientific_fingerprint": metadata["scientific_fingerprint"],
              "parameter_order": metadata["physical_parameter_order"],
              "parent": lineage, "source_pins": sources, "conditional_kernel": metadata,
              "runtime": runtime_identity(), "posterior_accuracy_guaranteed": False,
              "observations_references_truth_loaded": False, "original_cohort_preserved": True,
              "support_filtering_clipping_redraw_or_trimming": False,
              "raw_scientific_bindings": parent["raw_scientific_bindings"],
              "raw_scientific_bindings_scope": "inherited parent provenance only; no bound observation files loaded",
              "end_to_end_posterior_request_timing_claimed": False}
    write_json(output / "manifest.json", record)
    try:
        seed = derive_seed(20261111, ROLE, case=0, repetition=0)
        result, receipt = refresh_once(values, seed)
        np.savez(output / "samples.npz", samples=result)
        record.update(status="complete", samples=array_descriptor(result),
                      samples_file="samples.npz", samples_sha256=sha256_file(output / "samples.npz"),
                      refresh=receipt)
        require(source_guard() == sources, "producer/kernel/contract source changed during refresh")
        require(sha256_file(parent_manifest) == PARENT_MANIFEST_SHA,
                "parent manifest changed during refresh")
        require(sha256_file(Path(parent_manifest).with_name("samples.npz")) == PARENT_PAYLOAD_SHA,
                "parent payload changed during refresh")
        producer_rng_after = state_sha((np.random.get_state(), random.getstate()))
        require(producer_rng_before == producer_rng_after, "producer changed default RNG")
        record["default_RNG_observation"] = {
            "before_sha256": producer_rng_before, "after_sha256": producer_rng_after,
            "preserved": True, "observed_generators": ["NumPy legacy default", "Python random"],
            "scope": "run entry through samples save and final source/parent authentication; JSON footer follows",
            "torch_CPU_CUDA_RNG_hashes_observed": False}
        record["sources_unchanged"] = True
    except BaseException as error:
        record.update(status="failed", error_type=type(error).__name__, error=str(error))
        write_json(output / "manifest.json", record)
        raise
    write_json(output / "manifest.json", record)
    write_json(output / "checksums.json",
               {"manifest.json": sha256_file(output / "manifest.json"),
                "samples.npz": record["samples_sha256"]})
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record = run(args.parent_manifest, args.output)
    print(json.dumps({"status": record["status"], "variant": VARIANT,
                      "returned": record["samples"]["shape"][0],
                      "samples_sha256": record["samples_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
