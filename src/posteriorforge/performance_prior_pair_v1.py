"""Exact frozen FP32/F64 production math with explicit pair-specific proof custody.

Built with PriorLabs-TabPFN. No numerical source, source globals, model settings,
codec, mask, order, stream rule, support, FIFO, or success/footer algorithm changes.
A new pair must have its own successful all21 matching execution artifact.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import inspect
import json
import re
from pathlib import Path
from types import FunctionType, SimpleNamespace

from . import performance_prior_physical_fifo_v1 as ancestor
from . import prior_coordinate_pair_v1 as pairs

SCHEMA = "posteriorforge.inference.prior_coordinate_pair.v1"
VARIANT = "accuracy_prior_coordinates_pair_native_fp32_physical_f64_fifo_v1"
ANCESTOR_SHA256 = "a30e0c15513eb0c1a6ed3c68f0781327edef4714a244a119a3a4d4c68f8780bf"
PREREGISTRATION_PATH = "docs/accuracy-prior-coordinate-fixed-panel-v1.md"
PREREGISTRATION_SHA256 = "9fc6a91568864cc996dafbcfaa41e77280e4825c63f61d455bdadf3703cd0a66"
PAIR_MODULE_SHA256 = "402117cad65af190352cb0b14977fbbdc896f196d069749b0be28608fdef400e"
PAIR_CPU_HELPER_SHA256 = "b69a92b59fa504bce22e34dc31895f9162870425118764293f959c46390110e4"
QUALIFIER_FILENAME = "qualify_prior_coordinate_pair_v1.py"
QUALIFICATION_SCHEMA = "posteriorforge.prior_coordinate_pair_conditionals.v1"
QUALIFICATION_STATUS = "prior_coordinate_pair_screen_pass"
QUALIFIER_SHA256 = "29ca88203b81f240769a6bb38b18b68aff0fe6ed2df36003729b17e0116163a0"
PAIR_CPU_RECEIPT_SHA256 = "6f35d9d090a955604186c5fbf3312b82465eff6fdae7e4cd24a23b646dbdf193"
# Numerical proof sources exclude this data/orchestration-only registry.
# Root inserts only a completed same-pair proof after its independent audit.
QUALIFICATION_SHA256_BY_PAIR = {}
COUNT = 50000
MODE = ancestor.failed.MODE
backend = ancestor.failed.backend
codec = ancestor.failed.codec
model_bindings = ancestor.failed.model_bindings
physical_fifo = ancestor.physical_fifo
clean = ancestor._CLEAN_PROXY
require = ancestor.failed.require
sha256_file = ancestor.failed.sha256_file
load_script = ancestor.failed.load_script
_FIXED_RUN_CODE = ancestor._RUN_BODY.__code__
_SAMPLING_FACTORY = ancestor.physical_sampling_namespace
_DECODE = ancestor.failed.decode_complete_wave
_MODEL_BINDINGS = model_bindings


def source_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _sha(value, label):
    require(type(value) is str and re.fullmatch("[0-9a-f]{64}", value) is not None,
            label + " must be a pinned SHA256")
    return value


def _spec(value):
    require(type(value) is pairs.PairSpec, "explicit immutable PairSpec required")
    return value


def _configuration_state():
    return (ANCESTOR_SHA256, PREREGISTRATION_PATH, PREREGISTRATION_SHA256,
            PAIR_MODULE_SHA256, PAIR_CPU_HELPER_SHA256, QUALIFIER_FILENAME, QUALIFIER_SHA256,
            PAIR_CPU_RECEIPT_SHA256, COUNT, MODE, SCHEMA, VARIANT,
            tuple(sorted(QUALIFICATION_SHA256_BY_PAIR.items())))


def _check_configuration():
    require(_configuration_state() == _IMPORTED_CONFIGURATION,
            "pair source/proof registry configuration changed within process")
    for name, value in _OBJECT_SEALS.items():
        require(globals()[name] is value, "pair numerical module/object binding changed: " + name)


def _dependencies():
    return {
        "src/posteriorforge/performance_prior_physical_fifo_v1.py": ANCESTOR_SHA256,
        "src/posteriorforge/prior_coordinate_pair_v1.py": _sha(PAIR_MODULE_SHA256, "pair module"),
        "scripts/" + QUALIFIER_FILENAME: _sha(QUALIFIER_SHA256, "pair qualifier"),
        "scripts/preflight_prior_coordinate_pairs_v1.py": _sha(PAIR_CPU_HELPER_SHA256, "pair CPU helper"),
        PREREGISTRATION_PATH: PREREGISTRATION_SHA256,
    }


def runtime_source_inventory(*, require_committed):
    inherited = ancestor.runtime_source_inventory(require_committed=require_committed)
    sources = {**inherited["sources"], **_dependencies()}
    root = Path(__file__).resolve().parents[2]
    for name, digest in sources.items():
        require(sha256_file(root / name) == digest, "pair runtime source changed: " + name)
    if require_committed:
        committed = ancestor._committed_source_hashes(root, list(sources))
        require(committed == sources, "pair runtime source inventory differs from committed HEAD")
        own_name = "src/posteriorforge/performance_prior_pair_v1.py"
        require(hashlib.sha256(ancestor._git_bytes(root, own_name)).hexdigest() == source_sha256(),
                "exact pair producer must be committed")
    return {
        "schema": "posteriorforge.prior_coordinate_pair_runtime_sources.v1",
        "sources": sources, "source_count": len(sources),
        "inventory_sha256": hashlib.sha256(json.dumps(
            sources, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "current_checkout_exact": True, "committed_HEAD_exact": bool(require_committed),
        "own_producer_source_sha256": source_sha256(),
        "own_source_checked_separately_to_avoid_recursive_hash": True,
        "external_libraries": inherited["external_libraries"],
    }


def _integrity(*, require_committed):
    _check_configuration()
    ancestor.require_sources()
    require(ancestor.source_sha256() == ANCESTOR_SHA256, "frozen a30 producer changed")
    require(pairs.source_sha256() == _sha(PAIR_MODULE_SHA256, "pair module"),
            "explicit pair binding source changed")
    pairs.require_sources()
    require(ancestor._RUN_BODY.__code__ is _FIXED_RUN_CODE,
            "frozen production code object changed")
    require(ancestor.physical_sampling_namespace is _SAMPLING_FACTORY
            and ancestor.failed.decode_complete_wave is _DECODE
            and model_bindings is _MODEL_BINDINGS,
            "numerical codec/FIFO/model binding identity changed")
    require(source_sha256() == _IMPORTED_SOURCE_SHA256, "pair producer changed within process")
    for name, (function, code) in _FUNCTION_SEALS.items():
        require(globals()[name] is function and function.__code__ is code,
                "pair orchestration source closure changed: " + name)
    runtime_source_inventory(require_committed=require_committed)
    return True


def require_sources():
    return _integrity(require_committed=True)


def qualification_sha256(spec):
    spec = _spec(spec)
    value = QUALIFICATION_SHA256_BY_PAIR.get((spec.context_seed, spec.case))
    return _sha(value, "own-pair all21 qualification")


def _stable_ancestry_policy():
    policy = copy.deepcopy(ancestor.execution_metadata())
    # This field is explicitly process-context marshal telemetry, not source identity.
    telemetry = policy["derivation"].pop("run_code_sha256")
    _sha(telemetry, "ancestral process-local code telemetry")
    policy["derivation"]["run_code_telemetry_scope"] = (
        "sha256(marshal.dumps(code)); excluded from cross-process policy identity")
    return policy


def derivation_descriptor():
    original_ast = ast.dump(ast.parse(inspect.getsource(ancestor.failed.run)),
                            include_attributes=False)
    return {
        "frozen_F64_producer_sha256": ANCESTOR_SHA256,
        "unchanged_run_code_object_identity": ancestor._RUN_BODY.__code__ is _FIXED_RUN_CODE,
        "private_FunctionType_namespace": True,
        "original_namespaces_unmodified": True,
        "original71_run_source_AST_sha256": hashlib.sha256(original_ast.encode()).hexdigest(),
        "ancestral_F64_reverse_AST_exact": ancestor.derivation_descriptor()["reverse_AST_exact"],
        "declared_pair_binding_changes": [
            "explicit immutable context_seed/case",
            "own same-pair all21 proof/input gate",
            "pair-local FIFO scope and source/header custody"],
        "marshal_fingerprint_used_as_cross_process_source_identity": False,
    }


def execution_metadata(spec, proof_sha256):
    spec = _spec(spec)
    _sha(proof_sha256, "own-pair all21 qualification")
    require_sources()
    return {
        "schema": "posteriorforge.prior_coordinate_pair_execution.v1",
        "module_sha256": source_sha256(), "pair_spec": spec.as_dict(),
        "pair_binding_policy": spec.metadata(),
        "pair_binding_module_sha256": PAIR_MODULE_SHA256,
        "pair_CPU_helper_source_sha256": PAIR_CPU_HELPER_SHA256,
        "qualifier_source_sha256": QUALIFIER_SHA256,
        "same_pair_qualification_sha256": proof_sha256,
        "same_pair_CPU_panel_receipt_sha256": PAIR_CPU_RECEIPT_SHA256,
        "inherited_numerical_execution": _stable_ancestry_policy(),
        "runtime_source_inventory": runtime_source_inventory(require_committed=True),
        "derivation": derivation_descriptor(),
        "postprocessing_dtype_scope": "latent-bin probabilities Float32; decoded physical FIFO/output Float64",
        "numerical_candidate_changed": False, "FP16_activated": False,
        "wrapper_clock_scope": "unchanged a30 run-body entry; outer PairSpec/source/registry preflight precedes it",
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "old_case2_screen_inherited": False, "physical_G3_in_this_neural_request": False,
        "attribution": "Built with PriorLabs-TabPFN",
    }


def proposal_scope(spec, frozen, raw_bindings, model_inputs, seeds):
    spec = _spec(spec)
    scope = frozen.proposal_buffer_scope(
        raw_bindings, context_seed=spec.context_seed, case=spec.case, repetition=0,
        rank=0, world_size=1, seeds=seeds)
    scope.update(
        pair_spec=spec.as_dict(), pair_binding_module_sha256=PAIR_MODULE_SHA256,
        prior_coordinate_producer_sha256=source_sha256(),
        neural_adapter_source_sha256=ancestor.ORIGINAL_PRODUCER_SHA256,
        physical_fifo_source_sha256=ancestor.PHYSICAL_FIFO_SHA256,
        codec_sha256=ancestor.failed.CODEC_SHA256, backend_sha256=ancestor.failed.BACKEND_SHA256,
        coordinate_policy_sha256=backend.cached.policy_metadata()["policy_sha256"],
        native_execution_policy_sha256=backend.policy_metadata()["policy_sha256"],
        model_inputs=model_inputs, stage_to_physical_order=list(codec.ORDER),
        physical_to_stage_order=list(codec.INVERSE_ORDER),
        strict_full_wave_decode_before_FIFO_and_support=True,
        post_decode_fifo_dtype="Float64 physical; no cast")
    return scope


def qualification_receipt(path, *, spec, banks, pair_cpu_receipt, raw_bindings, model_inputs,
                          codec_inventory, codec_reference_inventory, codec_combined_inventory,
                          completed_grid, prefix_run, failed_screen):
    spec = _spec(spec)
    proof_sha = qualification_sha256(spec)
    require(sha256_file(path) == proof_sha, "own-pair conditional proof SHA differs")
    require(sha256_file(pair_cpu_receipt) == _sha(PAIR_CPU_RECEIPT_SHA256, "pair CPU panel"),
            "all23 pair CPU panel receipt differs")
    helper = load_script(QUALIFIER_FILENAME, _sha(QUALIFIER_SHA256, "pair qualifier"))
    document, receipt = helper.qualification_receipt(
        spec, path, banks=banks, completion_receipt=completed_grid,
        training_inventory=codec_inventory,
        reference_training_inventory=codec_reference_inventory,
        combined_training_inventory=codec_combined_inventory,
        pair_cpu_receipt=pair_cpu_receipt, raw_bindings=raw_bindings,
        model_inputs=model_inputs, prefix_run=prefix_run, failed_screen=failed_screen)
    require(document.get("schema") == QUALIFICATION_SCHEMA
            and document.get("status") == QUALIFICATION_STATUS
            and document.get("matching_FP32_execution_passed_all21") is True,
            "own-pair successful all21 proof schema/status required; no old case2 transfer")
    require(type(document.get("pair")) is dict
            and pairs.PairSpec.from_dict(document["pair"]) == spec,
            "conditional screen is for another pair")
    require(receipt.get("replay", {}).get("passed") is True, "own-pair all21 proof failed")
    return document, receipt


def bound_validation_namespace(spec, proof_sha256):
    spec = _spec(spec)
    return SimpleNamespace(
        backend=backend, MODE=MODE, clean=clean, source_sha256=source_sha256,
        execution_metadata=lambda: execution_metadata(spec, proof_sha256),
        proposal_scope=lambda frozen, raw, model, seeds: proposal_scope(spec, frozen, raw, model, seeds),
        model_bindings=model_bindings, codec=codec)


def _failure_observer(source_guard=None):
    namespace = dict(ancestor._capture_failure_footer.__globals__)
    namespace.update(source_sha256=source_sha256, require_sources=source_guard or require_sources)
    capture = FunctionType(ancestor._capture_failure_footer.__code__, namespace,
                           "capture_actual_pair_failure", ancestor._capture_failure_footer.__defaults__)
    namespace["_capture_failure_footer"] = capture
    safe = FunctionType(ancestor._safe_capture_failure_footer.__code__, namespace,
                        "safe_capture_actual_pair_failure",
                        ancestor._safe_capture_failure_footer.__defaults__)
    return safe


def _bind_run(spec, proof_sha256, pair_cpu_receipt):
    """Only configuration/global binding changes; the exact reviewed run code is reused."""
    spec = _spec(spec)
    _sha(proof_sha256, "own-pair proof")
    namespace = dict(ancestor._RUN_BODY.__globals__)
    original_identity = namespace["runtime_identity"]

    def identity():
        return {
            **original_identity(), "pair_spec": spec.as_dict(),
            "pair_binding_policy": spec.metadata(),
            "pair_binding_module_sha256": PAIR_MODULE_SHA256,
            "pair_cpu_receipt_sha256": PAIR_CPU_RECEIPT_SHA256,
            "pair_preregistration_sha256": PREREGISTRATION_SHA256,
            "fixed_case2_qualification_inherited": False,
            "numerical_candidate_changed": False,
            "actual_request_matches_its_own_finite_screen": True,
            "ownblock_mask_and_target_algorithm_unchanged": True,
        }

    def screen(path, **kwargs):
        return qualification_receipt(
            path, spec=spec, pair_cpu_receipt=pair_cpu_receipt, **kwargs)

    def bound_sources():
        require_sources()
        for key, value in bound_seals.items():
            require(namespace[key] is value, "private pair run global changed: " + key)
        require(function.__code__ is _FIXED_RUN_CODE, "private pair run code changed")
        return True

    namespace.update(
        SCHEMA=SCHEMA, VARIANT=VARIANT, CONTEXT_SEED=spec.context_seed, CASE=spec.case,
        CONDITIONAL_QUALIFICATION_SHA256=proof_sha256,
        PREREGISTRATION_SHA256=PREREGISTRATION_SHA256,
        source_sha256=source_sha256, require_sources=bound_sources,
        execution_metadata=lambda: execution_metadata(spec, proof_sha256),
        proposal_scope=lambda frozen, raw, model, seeds: proposal_scope(spec, frozen, raw, model, seeds),
        qualification_receipt=screen, runtime_identity=identity,
        _safe_capture_failure_footer=_failure_observer(bound_sources))
    function = FunctionType(_FIXED_RUN_CODE, namespace, "run_pair_bound_exact_F64_body",
                            ancestor._RUN_BODY.__defaults__, ancestor._RUN_BODY.__closure__)
    function.__kwdefaults__ = ancestor._RUN_BODY.__kwdefaults__
    bound_seals = dict(namespace)
    require(function.__code__ is ancestor._RUN_BODY.__code__, "pair run algorithm changed")
    return function


def run(*, context_seed, case, pair_cpu_receipt, **paths):
    spec = pairs.PairSpec(context_seed, case)
    require_sources()
    proof_sha = qualification_sha256(spec)
    require(sha256_file(pair_cpu_receipt) == _sha(PAIR_CPU_RECEIPT_SHA256, "pair CPU panel"),
            "pair CPU panel not pinned")
    return _bind_run(spec, proof_sha, pair_cpu_receipt)(**paths)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context-seed", type=int, required=True)
    parser.add_argument("--case", type=int, required=True)
    for name in ("banks", "output", "checkpoint", "ledger", "reference", "reference-grid",
                 "qualification", "codec-inventory", "codec-reference-inventory",
                 "codec-combined-inventory", "completed-grid", "prefix-run", "failed-screen",
                 "pair-cpu-receipt"):
        parser.add_argument("--" + name, required=True)
    result = run(**vars(parser.parse_args()))
    print(json.dumps({k: result[k] for k in
          ("schema", "variant", "status", "case", "context_seed", "samples_sha256")}, sort_keys=True))


_IMPORTED_SOURCE_SHA256 = source_sha256()
_IMPORTED_CONFIGURATION = _configuration_state()
_OBJECT_SEALS = {name: globals()[name] for name in (
    "ancestor", "pairs", "backend", "codec", "physical_fifo", "clean", "model_bindings")}
_FUNCTION_SEALS = {name: (globals()[name], globals()[name].__code__) for name in (
    "source_sha256", "_sha", "_spec", "_configuration_state", "_check_configuration", "_dependencies", "runtime_source_inventory", "_integrity",
    "require_sources", "qualification_sha256", "_stable_ancestry_policy", "derivation_descriptor",
    "execution_metadata", "proposal_scope", "qualification_receipt", "bound_validation_namespace",
    "_failure_observer", "_bind_run", "run")}

if __name__ == "__main__":
    main()
