#!/usr/bin/env python3
"""One saved-row mu_a refresh for a pending predefined SS15 grid point.

Built with PriorLabs-TabPFN. Separately labeled hybrid NPE-PFN/Gibbs.
No observations, reference, truth, model calls or new posterior requests.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import re

import numpy as np

from posteriorforge import conditional_mu_a as kernel
from posteriorforge.artifacts import runtime_identity, sha256_file, write_json
from posteriorforge.banks import derive_seed
from posteriorforge.science.analytical_benchmark_v2 import array_descriptor

SCHEMA = "posteriorforge.ss15_campaign_mu_a_refresh.v1"
VARIANT = "hybrid_ss15_mu_a_gibbs_campaign_once_v1"
PARENT_SCHEMA = "posteriorforge.inference.ss15_campaign.v1"
PARENT_VARIANT = "accuracy_ss15_fp16_fifo_campaign_v1"
# Exact frozen campaign source; no numerical-screen claim is transferred across cases.
CAMPAIGN_SHA = "f847d3e5154fae444de36a63c6d359e58280315d9c548d809da0d6e53f5c7c76"
CAMPAIGN_SOURCE = "src/posteriorforge/performance_ss15_campaign.py"
BASE_SHA = "5c1997f2b5e830f88aca1e29a4871c47c7b51940740e866acc7c8801e8a42feb"
KERNEL_SHA = "850df8d6498f78c12500ffdf60c4c9126f5c1ab67bc29867002d4c991373d2e8"
PILOT_SHA = "4099da3c19366997ab0800e5430b5bb1c129b4b3e535f678b3d4652233303480"
CHECKPOINT_SHA = "2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736"
ROLE = "ss15_mu_a_gibbs_once_v1"
COUNT = 50000
SEEDS = (20261111, 20261112, 20261113)
PENDING_GRID = tuple((s, c) for s in SEEDS for c in range(8) if (s, c) != (20261111, 0))


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(value):
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None


def frozen_refresh():
    path = Path(__file__).with_name("refresh_ss15_mu_a.py")
    require(sha256_file(path) == BASE_SHA, "frozen one-pass wrapper source changed")
    spec = importlib.util.spec_from_file_location("_campaign_mu_a_frozen_refresh", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(module.ROLE == ROLE and module.KERNEL_SHA == KERNEL_SHA and module.COUNT == COUNT,
            "frozen one-pass policy changed")
    return module


def source_guard(base):
    require(digest(CAMPAIGN_SHA), "campaign source is not frozen; execution prohibited")
    ancestor = base.source_guard()
    root = Path(__file__).resolve().parents[1]
    require(sha256_file(root / CAMPAIGN_SOURCE) == CAMPAIGN_SHA, "campaign producer source changed")
    source = Path(__file__).resolve()
    revision = ancestor["git_revision"]
    require(hashlib.sha256(base.git_bytes(root, revision, str(source.relative_to(root)))).hexdigest()
            == sha256_file(source), "campaign refresh producer must be committed before execution")
    return {"producer_sha256": sha256_file(source), "git_revision": revision,
            "campaign_producer_sha256": CAMPAIGN_SHA, "frozen_refresh_producer_sha256": BASE_SHA,
            "kernel_sha256": KERNEL_SHA, "math_contract_sha256": ancestor["math_contract_sha256"],
            "math_contract_origin_commit": ancestor["math_contract_origin_commit"],
            "producer_committed_before_run": True}


def request_spec(context_seed, case, *, context_size=4096, count=COUNT, repetition=0):
    require(type(context_seed) is int and type(case) is int
            and (context_seed, case) in PENDING_GRID,
            "one of the remaining23 predefined grid points required; anchor is already complete")
    require(type(context_size) is int and context_size == 4096, "exact4096 context required")
    require(type(count) is int and count == COUNT, "exact50000 accepted complete vectors required")
    require(type(repetition) is int and repetition == 0, "only repetition0 is declared")
    return {"context_seed": context_seed, "case": case, "context_size": context_size,
            "requested_count": count, "repetition": repetition}


def descriptor(value, shape, dtype):
    return (isinstance(value, dict) and set(value) == {"shape", "dtype", "sha256"}
            and value["shape"] == shape and value["dtype"] == dtype and digest(value["sha256"]))


def validate_parent(manifest, values, metadata):
    request = request_spec(manifest.get("context_seed"), manifest.get("case"),
                           context_size=manifest.get("context_size"),
                           count=manifest.get("requested_count"), repetition=manifest.get("repetition"))
    expected = {"schema": PARENT_SCHEMA, "variant": PARENT_VARIANT, "status": "complete",
                "runner_source_sha256": CAMPAIGN_SHA, "frozen_pilot_source_sha256": PILOT_SHA,
                "scientific_fingerprint": metadata["scientific_fingerprint"],
                "parameter_order": metadata["physical_parameter_order"],
                "rank": 0, "world_size": 1, "sources_unchanged": True,
                "default_rng_preserved": True, "physical_parameter_order_unchanged": True,
                "clipping_or_repair_performed": False, "partial_vector_filtering": False,
                "original_gate_unchanged": True, "actual_request_fixed_conditional_screen_performed": False}
    for key, value in expected.items():
        actual = manifest.get(key)
        require(type(actual) is type(value) and actual == value, "campaign parent differs: " + key)
    require(manifest.get("completed_stage_indices") == list(range(21)), "all21 original stages required")
    raw = manifest.get("raw_scientific_bindings")
    keys = {"checkpoint_sha256", "scientific_fingerprint", "parameter_order",
            "context_theta", "context_obs", "query"}
    require(isinstance(raw, dict) and set(raw) == keys, "full6 raw scientific provenance required")
    require(all(manifest.get(k) == raw[k] for k in raw), "top/nested raw bindings differ")
    require(raw["checkpoint_sha256"] == CHECKPOINT_SHA
            and raw["scientific_fingerprint"] == metadata["scientific_fingerprint"]
            and raw["parameter_order"] == metadata["physical_parameter_order"],
            "scientific model/raw physical order differs")
    for key, shape in [("context_theta", [4096, 21]), ("context_obs", [4096, 33]), ("query", [1, 33])]:
        require(descriptor(raw[key], shape, "<f8"), "originalFloat64 raw descriptor differs: " + key)
    model = manifest.get("model_inputs")
    require(isinstance(model, dict) and set(model) == {"context_theta", "context_obs", "query"},
            "actual model input descriptors required")
    for key, shape in [("context_theta", [4096, 21]), ("context_obs", [4096, 15]), ("query", [1, 15])]:
        require(descriptor(model[key], shape, "<f4"), "actual SS15Float32 descriptor differs: " + key)
    require(isinstance(values, np.ndarray) and values.dtype == np.dtype("float64")
            and values.shape == (COUNT, 21), "original50000 complete Float64 vectors required")
    require(array_descriptor(values) == manifest.get("samples"), "parent array bytes/descriptor differ")
    return request


def load_parent(path, expected_manifest_sha256, base, metadata):
    require(digest(CAMPAIGN_SHA), "campaign source is not frozen; execution prohibited")
    require(digest(expected_manifest_sha256), "explicit parent manifest SHA256 required")
    path = Path(path).resolve()
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected_manifest_sha256, "parent manifest custody hash differs")
    manifest = json.loads(raw)
    require(manifest.get("samples_file") == "samples.npz", "fixed samples filename required")
    payload = path.with_name("samples.npz")
    payload_sha = sha256_file(payload)
    require(digest(manifest.get("samples_sha256")) and payload_sha == manifest["samples_sha256"],
            "parent payload checksum differs")
    with np.load(payload, allow_pickle=False) as archive:
        require(archive.files == ["samples"], "complete physical samples only")
        values = archive["samples"]
    request = validate_parent(manifest, values, metadata)
    root = Path(__file__).resolve().parents[1]
    require(hashlib.sha256(base.git_bytes(root, manifest.get("git_revision"), CAMPAIGN_SOURCE)).hexdigest()
            == CAMPAIGN_SHA, "executed campaign producer Git bytes differ")
    lineage = {"manifest_sha256": expected_manifest_sha256, "samples_sha256": payload_sha,
               "samples": array_descriptor(values), "producer_sha256": CAMPAIGN_SHA,
               "git_revision": manifest["git_revision"]}
    return manifest, values, request, lineage


def run(parent_manifest, parent_manifest_sha256, output):
    base = frozen_refresh()
    default_before = base.state_sha((np.random.get_state(), random.getstate()))
    sources = source_guard(base)
    metadata = kernel.metadata()
    parent, values, request, lineage = load_parent(parent_manifest, parent_manifest_sha256, base, metadata)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    record = {"schema": SCHEMA, "variant": VARIANT, "status": "running",
              "attribution": "Built with PriorLabs-TabPFN", **request,
              "method": "hybrid NPE-PFN/Gibbs; one identical frozen scalar mu_a kernel",
              "pure_TFM_inference": False, "parent": lineage, "source_pins": sources,
              "conditional_kernel": metadata, "runtime": runtime_identity(),
              "scientific_fingerprint": metadata["scientific_fingerprint"],
              "parameter_order": metadata["physical_parameter_order"],
              "raw_scientific_bindings": parent["raw_scientific_bindings"],
              "raw_scientific_bindings_scope": "authenticated parent provenance only; no observation files loaded",
              "original_cohort_preserved": True, "observations_references_truth_loaded": False,
              "posterior_accuracy_guaranteed": False, "support_filtering_clipping_redraw_or_trimming": False,
              "end_to_end_posterior_request_timing_claimed": False}
    write_json(output / "manifest.json", record)
    try:
        seed = derive_seed(request["context_seed"], ROLE, case=request["case"], repetition=0)
        result, receipt = base.refresh_once(values, seed)
        with (output / "samples.npz").open("xb") as stream:
            np.savez(stream, samples=result)
        require(source_guard(base) == sources, "source changed during refresh")
        require(sha256_file(parent_manifest) == parent_manifest_sha256
                and sha256_file(Path(parent_manifest).with_name("samples.npz")) == lineage["samples_sha256"],
                "parent changed during refresh")
        default_after = base.state_sha((np.random.get_state(), random.getstate()))
        require(default_before == default_after, "producer default RNG changed")
        record.update(status="complete", samples=array_descriptor(result), samples_file="samples.npz",
                      samples_sha256=sha256_file(output / "samples.npz"), refresh=receipt,
                      sources_unchanged=True,
                      default_RNG_observation={"before_sha256": default_before, "after_sha256": default_after,
                          "preserved": True, "observed_generators": ["NumPy legacy default", "Python random"],
                          "torch_CPU_CUDA_RNG_hashes_observed": False,
                          "scope": "run entry through save and final source/parent checks; JSON footer follows"})
    except BaseException as error:
        record.update(status="failed", error_type=type(error).__name__, error=str(error))
        write_json(output / "manifest.json", record)
        raise
    write_json(output / "manifest.json", record)
    write_json(output / "checksums.json", {"manifest.json": sha256_file(output / "manifest.json"),
                                          "samples.npz": record["samples_sha256"]})
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--parent-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record = run(args.parent_manifest, args.parent_manifest_sha256, args.output)
    print(json.dumps({k: record[k] for k in ("schema", "variant", "status", "context_seed", "case")},
                     sort_keys=True))


if __name__ == "__main__":
    main()
