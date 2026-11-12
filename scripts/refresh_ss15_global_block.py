#!/usr/bin/env python3
"""One separately labeled prior-only correlated-global refresh of a pure SS15 cohort.

Built with PriorLabs-TabPFN. CPU only; no observations, references, truth,
simulator, neural model or evaluation API is loaded. Exactly50000 saved rows
are retained, with only their three globals refreshed once.
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

from posteriorforge import conditional_globals as kernel
from posteriorforge.artifacts import runtime_identity, sha256_file, write_json
from posteriorforge.banks import derive_seed
from posteriorforge.science.analytical_benchmark_v2 import array_descriptor

SCHEMA = "posteriorforge.ss15_global_block_refresh.v1"
VARIANT = "hybrid_ss15_global_block_gibbs_once_v1"
ROLE = "ss15_global_block_gibbs_once_v1"
COUNT = 50000
KERNEL_SHA = "e346e25bc9825abf016c2bd75addaf2b025d3e3d94e1a0ea6f6e31dd02997a43"
CONTRACT_SHA = "bc95d42f25f560c8ca404fee28b4367582a4323404ac3ff03310ff70fc6a2e5d"
CONTRACT_ORIGIN = "f71e93f"
CONTROL_MANIFEST_SHA = "28da396e0af6cf26009b4eb8feab8b75ed7a1d7a5add30440a35b38539da4f94"
CONTROL_PAYLOAD_SHA = "f263f712d943d7916205b2f2b77f9243ac4ba3819f5f739c45f00710d5b4be47"
CONTROL_ARRAY_SHA = "8c832a13961dc8f1c9b85dbda56b65ce3315730071c04b2d6ae3d5aff07386f1"
CONTROL_SOURCE_SHA = "4099da3c19366997ab0800e5430b5bb1c129b4b3e535f678b3d4652233303480"
GRAPH_SOURCE_SHA = "c2f20a8dbf1f4e2bf48294be038962d0c3e4e00ec8b4a7cab29ea259d91d1818"
GRAPH_MASK_SHA = "dfa096c314fdc3855d0a45a4bcb3f9858675747edc2403f917e32e19837eb110"
CHECKPOINT_SHA = "2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736"
PARENT_FAMILIES = {
    "posteriorforge.inference.ss15_fp16.v1":
        ("accuracy_ss15_fp16_fifo_v1", CONTROL_SOURCE_SHA, "src/posteriorforge/performance_ss15.py", "pure_ss15_control"),
    "posteriorforge.inference.ss15_graph_local.v1":
        ("accuracy_ss15_graph_local_fp16_fifo_v1", GRAPH_SOURCE_SHA, "src/posteriorforge/performance_ss15_graph.py", "pure_graph_local"),
}


def require(value, message):
    if not value:
        raise ValueError(message)


def state_sha(value):
    return hashlib.sha256(pickle.dumps(value, protocol=5)).hexdigest()


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def digest(value):
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None


def git_bytes(root, revision, name):
    require(isinstance(revision, str) and re.fullmatch("[0-9a-f]{40}", revision) is not None,
            "full40hex Git revision required")
    return subprocess.run(["git", "-C", str(root), "show", revision + ":" + name],
                          check=True, capture_output=True).stdout


def source_guard():
    root = Path(__file__).resolve().parents[1]
    source = Path(__file__).resolve()
    contract = root / "docs/accuracy-ss15-graph-local-v1.md"
    require(digest(KERNEL_SHA) and kernel.source_sha256() == KERNEL_SHA, "joint kernel source unpinned/changed")
    require(sha256_file(contract) == CONTRACT_SHA, "preregistered mathematical rule changed")
    origin = subprocess.run(["git", "-C", str(root), "rev-parse", CONTRACT_ORIGIN],
                            check=True, capture_output=True, text=True).stdout.strip()
    require(hashlib.sha256(git_bytes(root, origin, str(contract.relative_to(root)))).hexdigest()
            == CONTRACT_SHA, "pre-cohort mathematical rule Git bytes differ")
    revision = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                             check=True, capture_output=True, text=True).stdout.strip()
    require(hashlib.sha256(git_bytes(root, revision, str(source.relative_to(root)))).hexdigest()
            == sha256_file(source), "producer must be committed before cohort execution")
    kernel_path = "src/posteriorforge/conditional_globals.py"
    require(hashlib.sha256(git_bytes(root, revision, kernel_path)).hexdigest() == KERNEL_SHA,
            "joint kernel must be committed before cohort execution")
    return {"producer_sha256": sha256_file(source), "git_revision": revision,
            "kernel_sha256": KERNEL_SHA, "math_contract_sha256": CONTRACT_SHA,
            "math_contract_origin_commit": origin, "producer_and_kernel_committed_before_run": True}


def validate_parent(manifest, values, metadata):
    """Validate provenance only; no observation/reference files are read."""
    schema = manifest.get("schema")
    require(schema in PARENT_FAMILIES, "only pure SS15 or pure graph-local parent is permitted")
    variant, source, filename, family = PARENT_FAMILIES[schema]
    require(digest(source), "pure parent producer source is not frozen")
    expected = {"variant": variant, "status": "complete", "requested_count": COUNT,
                "context_seed": 20261111, "case": 0, "context_size": 4096, "repetition": 0,
                "runner_source_sha256": source, "scientific_fingerprint": metadata["scientific_fingerprint"],
                "parameter_order": metadata["physical_parameter_order"], "sources_unchanged": True,
                "default_rng_preserved": True, "completed_stage_indices": list(range(21)),
                "clipping_or_repair_performed": False, "partial_vector_filtering": False}
    for key, value in expected.items():
        require(type(manifest.get(key)) is type(value) and manifest[key] == value,
                "pure parent binding differs: " + key)
    require(isinstance(values, np.ndarray) and values.dtype == np.dtype("float64")
            and values.shape == (COUNT, 21), "exact50000 complete originalFloat64 rows required")
    require(array_descriptor(values) == manifest.get("samples"), "parent array descriptor differs")
    require(metadata["target_indices"] == [0, 1, 2]
            and metadata["unchanged_indices"] == list(range(3, 21)), "joint global-only kernel required")
    raw = manifest.get("raw_scientific_bindings", {})
    keys = {"checkpoint_sha256", "scientific_fingerprint", "parameter_order", "context_theta", "context_obs", "query"}
    require(set(raw) == keys and all(manifest.get(k) == v for k, v in raw.items()),
            "all six inherited raw scientific bindings required")
    require(raw["checkpoint_sha256"] == CHECKPOINT_SHA
            and raw["scientific_fingerprint"] == metadata["scientific_fingerprint"]
            and raw["parameter_order"] == metadata["physical_parameter_order"],
            "physical checkpoint/problem/order binding differs")
    for key, shape in (("context_theta", [4096, 21]), ("context_obs", [4096, 33]), ("query", [1, 33])):
        d = raw[key]
        require(d.get("shape") == shape and isinstance(d.get("dtype"), str) and np.dtype(d["dtype"]) == np.dtype("float64")
                and digest(d.get("sha256")), "inherited raw array descriptor differs: " + key)
    if family == "pure_ss15_control":
        require(manifest.get("samples_sha256") == CONTROL_PAYLOAD_SHA
                and manifest["samples"]["sha256"] == CONTROL_ARRAY_SHA,
                "original pure-SS15 control array/payload changed")
    else:
        require(manifest.get("preregistration_sha256") == CONTRACT_SHA
                and manifest.get("graph_local_input_policy", {}).get("module_sha256") == GRAPH_MASK_SHA
                and digest(manifest["graph_local_input_policy"].get("policy_sha256")),
                "pure graph-local mask/preregistration source differs")
        require(manifest.get("actual_graph_local_fixed_conditional_screen_performed") is False,
                "graph inputs were relabelled as numerically qualified")
    # Enforce finite complete original support before creating/advancing a private RNG.
    kernel.conditional_parameters(values)
    return {"family": family, "producer_sha256": source, "producer_path": filename}


def load_parent(path, expected_sha256, metadata):
    require(digest(expected_sha256), "explicit expected parent manifest SHA256 required")
    path = Path(path).resolve()
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected_sha256, "parent manifest differs from expected SHA")
    manifest = json.loads(raw)
    require(manifest.get("samples_file") == "samples.npz", "fixed parent samples filename required")
    payload = path.with_name("samples.npz")
    require(digest(manifest.get("samples_sha256"))
            and sha256_file(payload) == manifest["samples_sha256"], "parent payload hash mismatch")
    with np.load(payload, allow_pickle=False) as archive:
        require(archive.files == ["samples"], "only complete physical sample array expected")
        values = archive["samples"]
    identity = validate_parent(manifest, values, metadata)
    if identity["family"] == "pure_ss15_control":
        require(expected_sha256 == CONTROL_MANIFEST_SHA, "original pure control manifest changed")
    root = Path(__file__).resolve().parents[1]
    require(hashlib.sha256(git_bytes(root, manifest.get("git_revision"), identity["producer_path"])).hexdigest()
            == identity["producer_sha256"], "executed pure parent producer Git bytes differ")
    if identity["family"] == "pure_graph_local":
        require(hashlib.sha256(git_bytes(root, manifest["git_revision"],
                "src/posteriorforge/execution_ss15_graph.py")).hexdigest() == GRAPH_MASK_SHA,
                "executed graph-mask Git bytes differ")
    return manifest, values, {**identity, "manifest_sha256": expected_sha256,
                             "samples_sha256": manifest["samples_sha256"],
                             "samples": array_descriptor(values), "git_revision": manifest["git_revision"]}


def refresh_once(values, seed):
    """Strict-quota engineering seam; one independent correlated block per row."""
    require(isinstance(values, np.ndarray) and values.dtype == np.dtype("float64")
            and values.shape == (COUNT, 21), "strict50000 originalFloat64 quota required")
    original = array_descriptor(values)
    default_before = (copy.deepcopy(np.random.get_state()), random.getstate())
    rng = np.random.Generator(np.random.PCG64(seed))
    initial = copy.deepcopy(rng.bit_generator.state)
    mean, covariance = kernel.conditional_parameters(values)
    started = time.perf_counter()
    result = kernel.refresh(values, rng)
    seconds = time.perf_counter() - started
    final = copy.deepcopy(rng.bit_generator.state)
    replay = np.random.Generator(np.random.PCG64(0))
    replay.bit_generator.state = copy.deepcopy(initial)
    normals = replay.standard_normal((COUNT, 3))
    expected = mean + normals @ np.linalg.cholesky(covariance).T
    require(result[:, :3].tobytes() == expected.tobytes(), "joint Gaussian draw replay differs")
    require(final == replay.bit_generator.state, "private RNG consumed more than one[n,3] draw")
    require(result.shape == (COUNT, 21) and result.dtype == np.dtype("float64")
            and not np.shares_memory(result, values), "fresh completeFloat64 output required")
    require(result[:, 3:].tobytes() == values[:, 3:].tobytes(), "one of18 local/noise coordinates changed")
    require(array_descriptor(values) == original, "input cohort was mutated")
    kernel.conditional_parameters(result)
    default_after = (np.random.get_state(), random.getstate())
    require(state_sha(default_before) == state_sha(default_after), "default NumPy/Python RNG changed")
    return result, {
        "seed": int(seed), "role": ROLE, "bit_generator": "PCG64",
        "initial_private_state": initial, "final_private_state": final,
        "initial_private_state_sha256": json_sha(initial), "final_private_state_sha256": json_sha(final),
        "standard_normal_draws": array_descriptor(normals),
        "standard_normal_draw_shape": [COUNT, 3], "standard_normal_draw_count": COUNT*3,
        "independent_whole_matrix_draw_replay_passed": True,
        "default_numpy_python_RNG_before_sha256": state_sha(default_before),
        "default_numpy_python_RNG_after_sha256": state_sha(default_after),
        "default_numpy_python_RNG_preserved": True, "all18_columns_byte_identical": True,
        "input_array_unchanged": True, "finite_supported_complete_count": COUNT,
        "global_block_refresh_count_per_row": 1, "refresh_seconds": seconds,
        "torch_RNG_observation": "not imported or observed; NumPy-only producer"}


def run(parent_manifest, expected_parent_sha256, output):
    before = state_sha((np.random.get_state(), random.getstate()))
    sources = source_guard()
    metadata = kernel.metadata()
    parent, values, lineage = load_parent(parent_manifest, expected_parent_sha256, metadata)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    record = {"schema": SCHEMA, "variant": VARIANT, "status": "running",
              "attribution": "Built with PriorLabs-TabPFN",
              "method": "hybrid NPE-PFN/Gibbs; ONE correlated3-global conditional block",
              "pure_TFM_inference": False, "requested_count": COUNT,
              "context_seed": parent["context_seed"], "case": parent["case"], "repetition": 0,
              "scientific_fingerprint": metadata["scientific_fingerprint"],
              "parameter_order": metadata["physical_parameter_order"],
              "parent": lineage, "source_pins": sources, "conditional_kernel": metadata,
              "runtime": runtime_identity(), "posterior_accuracy_guaranteed": False,
              "observations_references_truth_loaded": False, "original_cohort_preserved": True,
              "support_filtering_clipping_redraw_or_trimming": False,
              "raw_scientific_bindings": parent["raw_scientific_bindings"],
              "raw_scientific_bindings_scope": "inherited provenance only; no bound observation files loaded",
              "end_to_end_posterior_request_timing_claimed": False}
    write_json(output/"manifest.json", record)
    try:
        seed = derive_seed(parent["context_seed"], ROLE, case=parent["case"], repetition=0)
        result, receipt = refresh_once(values, seed)
        with (output/"samples.npz").open("xb") as stream: np.savez(stream, samples=result)
        require(source_guard() == sources, "producer/kernel/preregistration source changed")
        require(sha256_file(parent_manifest) == expected_parent_sha256
                and sha256_file(Path(parent_manifest).with_name("samples.npz")) == lineage["samples_sha256"],
                "parent input changed during refresh")
        after = state_sha((np.random.get_state(), random.getstate()))
        require(before == after, "producer changed default RNG")
        record.update(status="complete", samples=array_descriptor(result),
                      samples_file="samples.npz", samples_sha256=sha256_file(output/"samples.npz"),
                      refresh=receipt, sources_unchanged=True,
                      default_RNG_observation={"before_sha256": before, "after_sha256": after,
                          "preserved": True, "observed_generators": ["NumPy legacy default", "Python random"],
                          "torch_CPU_CUDA_RNG_hashes_observed": False})
    except BaseException as error:
        record.update(status="failed", error_type=type(error).__name__, error=str(error))
        write_json(output/"manifest.json", record)
        raise
    write_json(output/"manifest.json", record)
    write_json(output/"checksums.json", {"manifest.json": sha256_file(output/"manifest.json"),
                                        "samples.npz": record["samples_sha256"]})
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--expected-parent-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record = run(args.parent_manifest, args.expected_parent_sha256, args.output)
    print(json.dumps({"status": record["status"], "variant": VARIANT,
                      "returned": record["samples"]["shape"][0],
                      "samples_sha256": record["samples_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
