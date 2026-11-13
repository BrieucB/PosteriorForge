"""Isolated v2 all8-native6/10/14 routing with fresh pair-specific proof custody.

Built with PriorLabs-TabPFN. Checkpoint, codec, masks, order, stream rules,
support and FIFO algorithms stay frozen. Stage14 joins6/10 on the original
stock all8 FP32 path. Existing sources, classes and global dictionaries are
untouched; no previous numerical qualification activates this new policy.
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
from contextlib import contextmanager
from . import performance_prior_pair_v1 as prior_v1
from . import execution_prior_native_v2 as native_v2

from . import performance_prior_physical_fifo_v1 as ancestor
from . import prior_coordinate_pair_v1 as pairs

SCHEMA = "posteriorforge.inference.prior_coordinate_pair.v2"
VARIANT = "accuracy_prior_coordinates_pair_native_fp32_physical_f64_fifo_v2"
ANCESTOR_SHA256 = "a30e0c15513eb0c1a6ed3c68f0781327edef4714a244a119a3a4d4c68f8780bf"
PREREGISTRATION_PATH = "docs/accuracy-prior-coordinate-native-control-v2.md"
PREREGISTRATION_SHA256 = "8640a4e157698862823521d3fd14396ecc3ebf121dffffabc987cc937d3ad2fc"
V1_PRODUCER_SHA256 = "52eaa638d09358765fcbb1d229265a5679f05a5fe0a5866f513412c7dc572a0b"
NATIVE_BACKEND_SHA256 = "0199b16e9cc34203fa0d5645d161742fe9ed244633d7beee047fbcbf2528661c"
PAIR_MODULE_SHA256 = "402117cad65af190352cb0b14977fbbdc896f196d069749b0be28608fdef400e"
PAIR_CPU_HELPER_SHA256 = "b69a92b59fa504bce22e34dc31895f9162870425118764293f959c46390110e4"
QUALIFIER_FILENAME = "qualify_prior_coordinate_pair_v2.py"
QUALIFICATION_SCHEMA = "posteriorforge.prior_coordinate_pair_conditionals.v2"
QUALIFICATION_STATUS = "prior_coordinate_pair_v2_screen_pass"
QUALIFIER_SHA256 = "a9f4d6818c030ac29587642fa744101c49b9f4a5fd192ab224c0f273f770e875"
PAIR_CPU_RECEIPT_SHA256 = "6f35d9d090a955604186c5fbf3312b82465eff6fdae7e4cd24a23b646dbdf193"
# Numerical proof sources exclude this data/orchestration-only registry.
# Root inserts only a completed same-pair proof after its independent audit.
QUALIFICATION_SHA256_BY_PAIR = {(20261111, 0): "aa8aff311d8f7fab6d774337510bb5f66abed309300c995267e98fa7f3e2a88e"}
COUNT = 50000
MODE = ancestor.failed.MODE
backend = native_v2
codec = ancestor.failed.codec
model_bindings = ancestor.failed.model_bindings
physical_fifo = ancestor.physical_fifo
clean = ancestor._CLEAN_PROXY
require = ancestor.failed.require
sha256_file = ancestor.failed.sha256_file
load_script = ancestor.failed.load_script
_ANCESTRAL_RUN_CODE = ancestor._RUN_BODY.__code__
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
    return (ANCESTOR_SHA256, V1_PRODUCER_SHA256, NATIVE_BACKEND_SHA256, PREREGISTRATION_PATH, PREREGISTRATION_SHA256,
            PAIR_MODULE_SHA256, PAIR_CPU_HELPER_SHA256, QUALIFIER_FILENAME, QUALIFIER_SHA256,
            PAIR_CPU_RECEIPT_SHA256, COUNT, MODE, SCHEMA, VARIANT,
            tuple(sorted(QUALIFICATION_SHA256_BY_PAIR.items())),
            tuple(sorted(_RUN_METADATA_EDITS.items())))


def _check_configuration():
    require(_configuration_state() == _IMPORTED_CONFIGURATION,
            "pair source/proof registry configuration changed within process")
    for name, value in _OBJECT_SEALS.items():
        require(globals()[name] is value, "pair numerical module/object binding changed: " + name)


def _dependencies():
    return {
        "src/posteriorforge/performance_prior_physical_fifo_v1.py": ANCESTOR_SHA256,
        "src/posteriorforge/performance_prior_pair_v1.py": V1_PRODUCER_SHA256,
        "src/posteriorforge/execution_prior_native_v2.py": _sha(NATIVE_BACKEND_SHA256, "v2 native backend"),
        "src/posteriorforge/prior_coordinate_pair_v1.py": _sha(PAIR_MODULE_SHA256, "pair module"),
        "scripts/" + QUALIFIER_FILENAME: _sha(QUALIFIER_SHA256, "pair qualifier"),
        "scripts/preflight_prior_coordinate_pairs_v1.py": _sha(PAIR_CPU_HELPER_SHA256, "pair CPU helper"),
        PREREGISTRATION_PATH: PREREGISTRATION_SHA256,
    }


def runtime_source_inventory(*, require_committed):
    inherited = prior_v1.runtime_source_inventory(require_committed=require_committed)
    sources = {**inherited["sources"], **_dependencies()}
    root = Path(__file__).resolve().parents[2]
    for name, digest in sources.items():
        require(sha256_file(root / name) == digest, "pair runtime source changed: " + name)
    if require_committed:
        committed = ancestor._committed_source_hashes(root, list(sources))
        require(committed == sources, "pair runtime source inventory differs from committed HEAD")
        own_name = "src/posteriorforge/performance_prior_pair_v2.py"
        require(hashlib.sha256(ancestor._git_bytes(root, own_name)).hexdigest() == source_sha256(),
                "exact pair producer must be committed")
    return {
        "schema": "posteriorforge.prior_coordinate_pair_runtime_sources.v2",
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
    prior_v1.require_sources()
    require(prior_v1.source_sha256() == V1_PRODUCER_SHA256, "frozen v1 pair producer changed")
    require(backend.source_sha256() == _sha(NATIVE_BACKEND_SHA256, "v2 native backend"), "v2 native backend changed")
    backend.require_sources()
    require(_FIXED_RUN_CODE is _IMPORTED_RUN_CODE, "private v2 run code object changed")
    for name, (function, code) in _ADAPTER_SOURCE_SEALS.items():
        original = getattr(ancestor.failed.PriorCoordinateStageAdapter, name)
        original = original.fget if isinstance(original, property) else original
        require(original is function and original.__code__ is code,
                "frozen adapter source method changed: " + name)
    require(ancestor.failed.engine_scope.__wrapped__ is _SCOPE_SOURCE
            and _SCOPE_SOURCE.__code__ is _SCOPE_SOURCE_CODE
            and ancestor.failed.validate_prediction is _PREDICTION_SOURCE
            and _PREDICTION_SOURCE.__code__ is _PREDICTION_SOURCE_CODE,
            "frozen context/prediction source code changed")
    require(ancestor.source_sha256() == ANCESTOR_SHA256, "frozen a30 producer changed")
    require(pairs.source_sha256() == _sha(PAIR_MODULE_SHA256, "pair module"),
            "explicit pair binding source changed")
    pairs.require_sources()
    require(ancestor._RUN_BODY.__code__ is _ANCESTRAL_RUN_CODE,
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
    old, new = _ANCESTRAL_RUN_CODE, _FIXED_RUN_CODE
    assert new.co_consts == tuple(_RUN_METADATA_EDITS.get(x, x) if type(x) is str else x
                                  for x in old.co_consts)
    return {
        "frozen_F64_producer_sha256": ANCESTOR_SHA256,
        "frozen_v1_pair_producer_sha256": V1_PRODUCER_SHA256,
        "top_run_codeobject_identity_with_a30": False,
        "only_two_run_metadata_literals_changed": True,
        "original_run_controlflow_bytecode_equal": new.co_code == old.co_code,
        "original_run_names_varnames_freevars_equal":
            (new.co_names, new.co_varnames, new.co_freevars) == (old.co_names, old.co_varnames, old.co_freevars),
        "all_nested_run_codeobjects_identical": all(
            new.co_consts[i] is value for i, value in enumerate(old.co_consts)
            if inspect.iscode(value)),
        "metadata_literals_reversible": tuple(
            _RUN_METADATA_REVERSE.get(x, x) if type(x) is str else x
            for x in new.co_consts) == old.co_consts,
        "metadata_literal_edits": dict(_RUN_METADATA_EDITS),
        "original_adapter_method_codeobjects_identical": True,
        "private_adapter_backend_source_policy_globals": True,
        "original71_run_source_AST_sha256": hashlib.sha256(original_ast.encode()).hexdigest(),
        "ancestral_F64_reverse_AST_exact": ancestor.derivation_descriptor()["reverse_AST_exact"],
        "original_namespaces_unmodified": True,
        "marshal_fingerprint_used_as_cross_process_source_identity": False,
    }


def execution_metadata(spec, proof_sha256):
    spec = _spec(spec)
    _sha(proof_sha256, "own-pair all21 qualification")
    require_sources()
    return {
        "schema": "posteriorforge.prior_coordinate_pair_execution.v2",
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
        "numerical_candidate_changed": True, "execution_routing_changed_relative_v1": True,
        "neural_weights_codec_sampler_support_math_changed": False,
        "inherited_v1_numerical_qualification": False,
        "active_native_backend_source_sha256": NATIVE_BACKEND_SHA256,
        "active_native_execution_policy": backend.policy_metadata(),
        "native_all8_stages": [6, 10, 14], "cached_stages": 18,
        "FP16_activated": False,
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
        codec_sha256=ancestor.failed.CODEC_SHA256, backend_sha256=NATIVE_BACKEND_SHA256,
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


def _clone_function(function, namespace):
    clone = FunctionType(function.__code__, namespace, function.__name__,
                         function.__defaults__, function.__closure__)
    clone.__kwdefaults__ = function.__kwdefaults__
    clone.__annotations__ = dict(function.__annotations__)
    return clone


def _adapter_bindings(spec, proof_sha256):
    """Exact71 methods in a new class; only policy/backend/source globals differ."""
    namespace = dict(ancestor.failed.__dict__)
    namespace.update(backend=backend, BACKEND_SHA256=NATIVE_BACKEND_SHA256,
        source_sha256=source_sha256, require_sources=require_sources,
        execution_metadata=lambda: execution_metadata(spec, proof_sha256))
    original = ancestor.failed.PriorCoordinateStageAdapter
    attrs = {"__module__": __name__, "__doc__": original.__doc__}
    for name, value in vars(original).items():
        if inspect.isfunction(value):
            attrs[name] = _clone_function(value, namespace)
            require(attrs[name].__code__ is value.__code__, "adapter method arithmetic changed")
        elif isinstance(value, property):
            attrs[name] = property(*[
                _clone_function(x, namespace) if x is not None else None
                for x in (value.fget, value.fset, value.fdel)], doc=value.__doc__)
        elif name not in ("__dict__", "__weakref__", "__module__", "__doc__"):
            attrs[name] = value
    adapter = type("PriorPairV2StageAdapter", (object,), attrs)
    namespace["PriorCoordinateStageAdapter"] = adapter
    generator = _clone_function(ancestor.failed.engine_scope.__wrapped__, namespace)
    scope = contextmanager(generator)
    prediction_check = _clone_function(ancestor.failed.validate_prediction, namespace)
    require(generator.__code__ is ancestor.failed.engine_scope.__wrapped__.__code__
            and prediction_check.__code__ is ancestor.failed.validate_prediction.__code__,
            "adapter context or prediction arithmetic changed")
    return adapter, scope, prediction_check, namespace


def _bind_run(spec, proof_sha256, pair_cpu_receipt):
    """Only configuration/global binding changes; the exact reviewed run code is reused."""
    spec = _spec(spec)
    _sha(proof_sha256, "own-pair proof")
    namespace = dict(ancestor._RUN_BODY.__globals__)
    adapter, scope, check, adapter_namespace = _adapter_bindings(spec, proof_sha256)
    namespace.update(backend=backend, BACKEND_SHA256=NATIVE_BACKEND_SHA256,
        engine_scope=scope, validate_prediction=check)
    original_identity = namespace["runtime_identity"]

    def identity():
        return {
            **original_identity(), "pair_spec": spec.as_dict(),
            "pair_binding_policy": spec.metadata(),
            "pair_binding_module_sha256": PAIR_MODULE_SHA256,
            "pair_cpu_receipt_sha256": PAIR_CPU_RECEIPT_SHA256,
            "pair_preregistration_sha256": PREREGISTRATION_SHA256,
            "fixed_case2_qualification_inherited": False,
            "numerical_candidate_changed": True,
            "execution_routing_changed_relative_v1": True,
            "neural_weights_codec_sampler_support_math_changed": False,
            "inherited_v1_numerical_qualification": False,
            "pair_data_preregistration_sha256": pairs.PREREGISTRATION_SHA256,
            "actual_request_matches_its_own_finite_screen": True,
            "ownblock_mask_and_target_algorithm_unchanged": True,
        }

    def screen(path, **kwargs):
        return qualification_receipt(
            path, spec=spec, pair_cpu_receipt=pair_cpu_receipt, **kwargs)

    def bound_sources():
        require_sources()
        require(set(vars(adapter)) == adapter_class_keys, "private v2 adapter class fields changed")
        for name, value in adapter_class_attributes.items():
            require(vars(adapter)[name] is value, "private v2 adapter attribute changed: " + name)
        for name, adapter_method, code in adapter_method_seals:
            require(adapter_method.__code__ is code and adapter_method.__globals__ is adapter_namespace,
                    "private v2 adapter method code/globals changed: " + name)
        require(scope.__code__ is scope_code and scope.__wrapped__ is scope_generator
                and scope_generator.__code__ is scope_generator_code
                and scope_generator.__globals__ is adapter_namespace,
                "private v2 context generator/code/globals changed")
        require(check.__code__ is prediction_code and check.__globals__ is adapter_namespace,
                "private v2 prediction checker code/globals changed")
        for key, value in adapter_seals.items():
            require(adapter_namespace[key] is value, "private v2 adapter global changed: " + key)
        for key, value in bound_seals.items():
            require(namespace[key] is value, "private pair run global changed: " + key)
        require(function.__code__ is _FIXED_RUN_CODE, "private pair run code changed")
        return True

    adapter_namespace["require_sources"] = bound_sources
    namespace.update(
        SCHEMA=SCHEMA, VARIANT=VARIANT, CONTEXT_SEED=spec.context_seed, CASE=spec.case,
        CONDITIONAL_QUALIFICATION_SHA256=proof_sha256,
        PREREGISTRATION_SHA256=PREREGISTRATION_SHA256,
        NATIVE_PROTOCOL_SHA256=PREREGISTRATION_SHA256,
        source_sha256=source_sha256, require_sources=bound_sources,
        execution_metadata=lambda: execution_metadata(spec, proof_sha256),
        proposal_scope=lambda frozen, raw, model, seeds: proposal_scope(spec, frozen, raw, model, seeds),
        qualification_receipt=screen, runtime_identity=identity,
        _safe_capture_failure_footer=_failure_observer(bound_sources))
    function = FunctionType(_FIXED_RUN_CODE, namespace, "run_pair_bound_exact_F64_body",
                            ancestor._RUN_BODY.__defaults__, ancestor._RUN_BODY.__closure__)
    function.__kwdefaults__ = ancestor._RUN_BODY.__kwdefaults__
    bound_seals = dict(namespace)
    adapter_seals = dict(adapter_namespace)
    adapter_class_attributes = dict(vars(adapter))
    adapter_class_keys = set(adapter_class_attributes)
    adapter_method_seals = []
    for name, value in adapter_class_attributes.items():
        methods = (value.fget, value.fset, value.fdel) if isinstance(value, property) else (value,)
        for method in methods:
            if inspect.isfunction(method):
                adapter_method_seals.append((name, method, method.__code__))
    scope_code, scope_generator = scope.__code__, scope.__wrapped__
    scope_generator_code, prediction_code = scope_generator.__code__, check.__code__
    require(function.__code__ is _FIXED_RUN_CODE, "v2 pair run code changed")
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


_RUN_METADATA_EDITS = {
    "FP32; 19 cached stages, all8 original-control at6/10":
        "FP32; 18 cached stages, all8 original-control at6/10/14",
    "INNER frozen88 precision owner only; outer native all8 stages6/10 are explicit":
        "INNER frozen88 precision owner only; outer native all8 stages6/10/14 are explicit",
}
_RUN_METADATA_REVERSE = {value: key for key, value in _RUN_METADATA_EDITS.items()}
_require_hits = {key: sum(type(x) is str and x == key for x in _ANCESTRAL_RUN_CODE.co_consts)
                 for key in _RUN_METADATA_EDITS}
require(all(x == 1 for x in _require_hits.values()) and sum(_require_hits.values()) == 2,
        "exact two run metadata literal replacements required")
_FIXED_RUN_CODE = _ANCESTRAL_RUN_CODE.replace(co_consts=tuple(
    _RUN_METADATA_EDITS.get(x, x) if type(x) is str else x
    for x in _ANCESTRAL_RUN_CODE.co_consts))
require(_FIXED_RUN_CODE.co_code == _ANCESTRAL_RUN_CODE.co_code
        and (_FIXED_RUN_CODE.co_names, _FIXED_RUN_CODE.co_varnames, _FIXED_RUN_CODE.co_freevars) ==
            (_ANCESTRAL_RUN_CODE.co_names, _ANCESTRAL_RUN_CODE.co_varnames, _ANCESTRAL_RUN_CODE.co_freevars)
        and all(_FIXED_RUN_CODE.co_consts[i] is x for i, x in enumerate(_ANCESTRAL_RUN_CODE.co_consts)
                if inspect.iscode(x))
        and tuple(_RUN_METADATA_REVERSE.get(x, x) if type(x) is str else x
                  for x in _FIXED_RUN_CODE.co_consts) == _ANCESTRAL_RUN_CODE.co_consts,
        "undeclared run controlflow/constant change")
_IMPORTED_RUN_CODE = _FIXED_RUN_CODE
_ADAPTER_SOURCE_SEALS = {}
for _name, _value in vars(ancestor.failed.PriorCoordinateStageAdapter).items():
    _method = _value.fget if isinstance(_value, property) else _value
    if inspect.isfunction(_method):
        _ADAPTER_SOURCE_SEALS[_name] = (_method, _method.__code__)
_SCOPE_SOURCE = ancestor.failed.engine_scope.__wrapped__
_SCOPE_SOURCE_CODE = _SCOPE_SOURCE.__code__
_PREDICTION_SOURCE = ancestor.failed.validate_prediction
_PREDICTION_SOURCE_CODE = _PREDICTION_SOURCE.__code__
_IMPORTED_SOURCE_SHA256 = source_sha256()
_IMPORTED_CONFIGURATION = _configuration_state()
_OBJECT_SEALS = {name: globals()[name] for name in (
    "ancestor", "prior_v1", "pairs", "backend", "codec", "physical_fifo", "clean", "model_bindings",
    "_FIXED_RUN_CODE", "_IMPORTED_RUN_CODE", "_ANCESTRAL_RUN_CODE", "_RUN_METADATA_EDITS", "_RUN_METADATA_REVERSE")}
_FUNCTION_SEALS = {name: (globals()[name], globals()[name].__code__) for name in (
    "source_sha256", "_sha", "_spec", "_configuration_state", "_check_configuration", "_dependencies", "runtime_source_inventory", "_integrity",
    "require_sources", "qualification_sha256", "_stable_ancestry_policy", "derivation_descriptor",
    "execution_metadata", "proposal_scope", "qualification_receipt", "bound_validation_namespace",
    "_failure_observer", "_clone_function", "_adapter_bindings", "_bind_run", "run")}

if __name__ == "__main__":
    main()
