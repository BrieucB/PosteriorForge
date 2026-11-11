"""Explicit CDF-only variant using the unchanged, source-guarded baseline runner.

Built with PriorLabs-TabPFN. The default CLI/baseline is never patched. Only this
entry point activates the versioned compatibility adapter; scientific sampling,
RNG derivation, rejection and model settings remain in the original runner.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import threading
import time

from . import inference as baseline
from .artifacts import runtime_identity, sha256_file, write_json
from .banks import load_heldout
from .reference_grid import _require_budgeted_gpu
from .science.analytical_benchmark_v2 import array_descriptor, build_problem

VARIANT = "cdf_arithmetic_v1"
SCHEMA = "posteriorforge.inference.cdf_arithmetic.v1"
BASELINE_RELEASE = "89404ac9e43bf9fb72d3e51a7113dbd207871221"
BASELINE_INFERENCE_SHA256 = "95366efc4871b96c3a1a3f19414ed1fd77d058f02135f4afd95a64b5918b3d3e"
UPSTREAM_PROVENANCE_SHA256 = "4caf84b49b04c79f28bcbe9f88f9da6b64ac0b99a2c60f53e01b2fc1c6833c8b"
_HOOK_ACTIVE = False


UPSTREAM_SOURCE_SHA256 = {
    "__init__.py": "a32a437d7910cec8c251ceb6849fece6b61ece32c16e8a346e4afae82bf7bc08",
    "accept_reject_sampler.py": "e6f862a176eae4cf0269c49655cf2edefc854428461445c74aa2b02f3b144512",
    "npe_pfn.py": "29517328b3ff23e2e8040d19f33e79adef5fc7f91db2cffccf60234344399a85",
    "restricted_prior.py": "13026cf8a4618aa4ca6406e0596a7ec97c0664de5b7c2b4dd7f31e4c32ece160",
    "support_posterior.py": "022e5b22b9397e9faea08b73df76b5a0c5cf0c012b665decfe28c5b725c362ee",
    "tsnpe_pfn.py": "f959662adf6ac6ed5aecefcebad907ef951c22bf05bbcf2c8e0e6312cb054034",
}


def guard_baseline_source():
    """Guard original sources in both a checkout and a public installed wheel."""
    if sha256_file(baseline.__file__) != BASELINE_INFERENCE_SHA256:
        raise RuntimeError("original inference runner source hash mismatch")
    if baseline.upstream_source_hashes() != UPSTREAM_SOURCE_SHA256:
        raise RuntimeError("six original upstream sources changed")
    external = Path(__file__).resolve().parents[2] / "provenance/upstream-npe-pfn.json"
    if external.is_file() and sha256_file(external) != UPSTREAM_PROVENANCE_SHA256:
        raise RuntimeError("original upstream provenance hash mismatch")
    return {"baseline_inference_sha256": BASELINE_INFERENCE_SHA256,
            "upstream_provenance_sha256": UPSTREAM_PROVENANCE_SHA256,
            "upstream_files": dict(UPSTREAM_SOURCE_SHA256)}


def reference_binding(*, banks, reference, reference_grid, case):
    """Authenticate existing numerical reference metadata; never integrate it."""
    problem = build_problem()
    query = load_heldout(banks, case).obs
    expected = {"scientific_fingerprint": problem.scientific_fingerprint,
                "parameter_order": list(problem.parameter_order),
                "observation_sha256": array_descriptor(query)["sha256"]}
    result = {}
    for name, folder in (("parent", reference), ("refined_grid", reference_grid)):
        root = Path(folder)
        path = root / "manifest.json"
        manifest = json.loads(path.read_text())
        if manifest.get("state") != "verified" or manifest.get("binding") != expected:
            raise ValueError(f"qualified {name} reference binding mismatch")
        precision = manifest.get("cdf_precision")
        if not isinstance(precision, (int, float)) or not 0 <= precision <= .005:
            raise ValueError(f"unqualified {name} reference precision")
        distribution = manifest["distribution"]
        if sha256_file(root / distribution["file"]) != distribution["sha256"]:
            raise ValueError(f"{name} reference distribution payload mismatch")
        result[name] = {"manifest_sha256": sha256_file(path),
                        "distribution_sha256": distribution["sha256"],
                        "cdf_precision": precision, "binding": expected}
        if name == "refined_grid" and manifest.get("parent_manifest_sha256") != result["parent"]["manifest_sha256"]:
            raise ValueError("refined grid is not bound to the selected parent")
    return result


def label_manifest(value, *, descriptor, common, pending=False):
    """Copy only metadata; never mutate the original runner's status object."""
    result = dict(value)
    result.update(common)
    result.update(schema=SCHEMA, variant=VARIANT, compatibility_adapter=dict(descriptor))
    if pending and result.get("status") == "complete":
        result["status"] = "sampling_complete_pending_adapter_restore"
    return result


@contextmanager
def labelled_manifest_writes(output, descriptor, common):
    """A process-local metadata hook around the unchanged runner's writer alias."""
    global _HOOK_ACTIVE
    if _HOOK_ACTIVE or threading.current_thread() is not threading.main_thread():
        raise RuntimeError("manifest hook must be non-reentrant on the main thread")
    if baseline.write_json is not write_json:
        raise RuntimeError("original inference writer alias was already replaced")
    expected = Path(output).resolve() / "manifest.json"
    original = baseline.write_json
    state = {"owns_output": False, "writer_restored": False}

    def recording_writer(path, value):
        if Path(path).resolve() != expected or value.get("schema") != "posteriorforge.inference.v1":
            raise RuntimeError("metadata hook refused an unrelated artifact")
        state["owns_output"] = True  # original runner already created it exclusively
        original(path, label_manifest(value, descriptor=descriptor, common=common, pending=True))

    _HOOK_ACTIVE = True
    baseline.write_json = recording_writer
    try:
        yield state
    finally:
        was_expected = baseline.write_json is recording_writer
        baseline.write_json = original
        _HOOK_ACTIVE = False
        state["writer_restored"] = baseline.write_json is original
        if not was_expected:
            raise RuntimeError("inference manifest writer changed during explicit run")


def run_cdf_variant(*, banks, output, context_seed, case, checkpoint, ledger,
                    reference, reference_grid, context_size=4096, count=50000,
                    microbatch=10000, repetition=0):
    """Run one immutable-input scientific case; no later gate is implied."""
    output = Path(output)
    if output.exists():
        raise FileExistsError("fresh output required; existing runs are never overwritten")
    if context_size not in (256, 1024, 4096) or count != 50000 or microbatch != 10000 or repetition != 0:
        raise ValueError("this unoptimized scientific variant requires50k, batch10k, repetition0")
    outer_started = time.perf_counter()
    source = guard_baseline_source()
    budget = _require_budgeted_gpu(ledger)
    if budget["gpus"] != 1:
        raise RuntimeError("CDF-only scientific variant requires one reserved GPU")
    refs = reference_binding(banks=banks, reference=reference, reference_grid=reference_grid, case=case)
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("one visible CUDA device is required")
    if torch.get_float32_matmul_precision() != "highest" or torch.backends.cuda.matmul.allow_tf32:
        raise RuntimeError("original FP32 matmul precision must remain highest/TF32False")
    from .compat_cdf import finite_cdf_compatibility
    common = {"baseline_release": BASELINE_RELEASE, "baseline_execution_source": source,
              "budget_binding": budget, "reference_binding": refs,
              "attribution": "Built with PriorLabs-TabPFN", "explicit_variant_activation": True,
              "tail_family_changed": False, "execution_optimized": False,
              "original_rng_role": "posterior", "adapter_setup_outside_original_timing": True,
              "torch_float32_matmul_precision": torch.get_float32_matmul_precision(),
              "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
              "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32)}
    status = None
    descriptor = {}
    hook = {"owns_output": False, "writer_restored": False}
    setup_started = time.perf_counter()
    try:
        with finite_cdf_compatibility() as active_descriptor:
            descriptor = active_descriptor
            ready = time.perf_counter()
            common["wrapper_preexecution_setup_seconds"] = ready - outer_started
            common["adapter_activation_seconds"] = ready - setup_started
            with labelled_manifest_writes(output, descriptor, common) as hook:
                status = baseline.run_baseline(
                    banks=banks, output=output, context_seed=context_seed, case=case,
                    context_size=context_size, count=count, microbatch=microbatch,
                    checkpoint=checkpoint, checkpoint_sha256=baseline.CHECKPOINT_SHA256,
                    repetition=repetition, device="cuda")
        if not hook["writer_restored"] or baseline.write_json is not write_json:
            raise RuntimeError("original inference writer was not restored")
        if descriptor.get("original_alias_restored") is not True or descriptor.get("patched_call_count", 0) <= 0:
            raise RuntimeError("CDF adapter activation/restoration was not demonstrated")
        expected_calls = 21 * 8 * status["rejection"]["support_calls"]
        if descriptor["patched_call_count"] != expected_calls:
            raise RuntimeError("CDF adapter did not cover every ensemble/stage prediction")
        common["expected_patched_call_count"] = expected_calls
        if guard_baseline_source() != source:
            raise RuntimeError("original execution sources changed during the run")
        if status["model"]["dtype"] != "torch.float32" or status["model"]["autocast"] is not False:
            raise RuntimeError("variant changed the original FP32 execution contract")
        status = label_manifest(status, descriptor=descriptor, common=common)
        status["wrapper_total_seconds"] = time.perf_counter() - outer_started
        status["metadata_writer_restored"] = True
        write_json(output / "manifest.json", status)
        return status
    except BaseException as error:
        if hook["owns_output"]:
            if status is None:
                path = output / "manifest.json"
                status = json.loads(path.read_text()) if path.is_file() else runtime_identity()
            status = label_manifest(status, descriptor=descriptor, common=common)
            status.update(status="failed", error_type=type(error).__name__, error=str(error),
                          wrapper_total_seconds=time.perf_counter()-outer_started,
                          metadata_writer_restored=baseline.write_json is write_json)
            write_json(output / "manifest.json", status)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("banks", "output", "checkpoint", "ledger", "reference", "reference-grid"):
        parser.add_argument("--"+name, required=True)
    parser.add_argument("--context-seed", required=True, type=int)
    parser.add_argument("--case", required=True, type=int)
    parser.add_argument("--context-size", default=4096, type=int, choices=(256, 1024, 4096))
    parser.add_argument("--count", default=50000, type=int)
    parser.add_argument("--microbatch", default=10000, type=int)
    parser.add_argument("--repetition", default=0, type=int)
    print(json.dumps(run_cdf_variant(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
