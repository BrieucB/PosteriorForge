"""Pair orchestration/source/proof isolation tests; no models or GPU."""
from types import SimpleNamespace
import copy
import hashlib
import json
import pickle
import random

import numpy as np
import pytest

from posteriorforge import performance_prior_pair_v2 as p


def state():
    return pickle.dumps((np.random.get_state(), random.getstate()), protocol=5)


def spec(case=0, seed=20261111):
    return p.pairs.PairSpec(seed, case)


def test_private_request_keeps_exact_reviewed_run_code_and_old_namespaces():
    before = dict(p.ancestor._RUN_BODY.__globals__)
    rng = state()
    first = p._bind_run(spec(), "a" * 64, "panel")
    other = p._bind_run(spec(7, 20261113), "b" * 64, "panel")
    assert first.__code__ is other.__code__ is p._FIXED_RUN_CODE
    assert first.__code__ is not p.ancestor._RUN_BODY.__code__
    assert first.__code__.co_code == p.ancestor._RUN_BODY.__code__.co_code
    assert first.__globals__["CASE"] == 0 and first.__globals__["CONTEXT_SEED"] == 20261111
    assert other.__globals__["CASE"] == 7 and other.__globals__["CONTEXT_SEED"] == 20261113
    for name in ("wave", "model_bindings", "record_decode_wave", "physical_fifo", "rng_hashes"):
        assert first.__globals__[name] is before[name]
    assert first.__globals__["physical_fifo"].complete_request is p.physical_fifo.complete_request
    assert all(p.ancestor._RUN_BODY.__globals__[k] is v for k, v in before.items())
    assert state() == rng


def test_derivation_uses_source_AST_and_live_identity_not_marshaled_code():
    result = p.derivation_descriptor()
    assert result["top_run_codeobject_identity_with_a30"] is False
    assert result["only_two_run_metadata_literals_changed"] and result["ancestral_F64_reverse_AST_exact"]
    for name in ("original_run_controlflow_bytecode_equal", "original_run_names_varnames_freevars_equal",
                 "all_nested_run_codeobjects_identical", "metadata_literals_reversible",
                 "original_adapter_method_codeobjects_identical", "private_adapter_backend_source_policy_globals"):
        assert result[name] is True
    assert len(result["original71_run_source_AST_sha256"]) == 64
    assert result["marshal_fingerprint_used_as_cross_process_source_identity"] is False
    assert "run_code_sha256" not in result


def test_scope_separates_pair_query_and_actual_original_seed_streams():
    frozen = p.ancestor.physical_sampling_namespace()
    raw = {"context_theta": {"sha256": "t"}, "context_obs": {"sha256": "o"},
           "query": {"sha256": "q"}, "checkpoint_sha256": "c",
           "scientific_fingerprint": "s", "parameter_order": list(range(21))}
    a = spec()
    b = spec(1)
    seed_a = frozen.worker_stream_seeds(a.context_seed, a.case, 0, 0, 1)
    seed_b = frozen.worker_stream_seeds(b.context_seed, b.case, 0, 0, 1)
    scope_a = p.proposal_scope(a, frozen, raw, {}, seed_a)
    scope_b = p.proposal_scope(b, frozen, raw, {}, seed_b)
    assert seed_a != seed_b and scope_a != scope_b
    assert scope_a["pair_spec"] == a.as_dict()
    assert scope_b["pair_spec"] == b.as_dict()
    assert scope_a["strict_full_wave_decode_before_FIFO_and_support"]
    assert scope_a["model_inputs"] == {}
    assert scope_a["stage_to_physical_order"] == list(p.codec.ORDER)


def test_nonimmutable_pair_like_object_is_rejected():
    with pytest.raises(RuntimeError, match="immutable PairSpec"):
        p._bind_run(SimpleNamespace(context_seed=20261111, case=0), "a" * 64, "panel")


def test_empty_own_pair_registry_cannot_inherit_case2_proof(monkeypatch):
    monkeypatch.setattr(p, "QUALIFICATION_SHA256_BY_PAIR", {})
    with pytest.raises(RuntimeError, match="own-pair all21 qualification"):
        p.qualification_sha256(spec())
    with pytest.raises(RuntimeError, match="own-pair all21 qualification"):
        p.qualification_sha256(spec(2))


def test_case2_registry_entry_does_not_authorize_case0(monkeypatch):
    old = "5ec69eb7aadbd74c7d96ee534b71d0d8bee2b6dd6f6e3d2f44e74607452f682b"
    monkeypatch.setattr(p, "QUALIFICATION_SHA256_BY_PAIR", {(20261111, 2): old})
    with pytest.raises(RuntimeError, match="own-pair"):
        p.qualification_sha256(spec())


def gate_setup(monkeypatch, tmp_path, document, receipt=None):
    artifact = tmp_path / "proof.json"
    artifact.write_text(json.dumps(document))
    panel = tmp_path / "panel.json"
    panel.write_text('{"panel":"explicit test"}')
    proof_sha = hashlib.sha256(artifact.read_bytes()).hexdigest()
    monkeypatch.setattr(p, "QUALIFICATION_SHA256_BY_PAIR", {(20261111, 0): proof_sha})
    monkeypatch.setattr(p, "PAIR_CPU_RECEIPT_SHA256", hashlib.sha256(panel.read_bytes()).hexdigest())
    monkeypatch.setattr(p, "QUALIFIER_SHA256", "a" * 64)
    def helper(*args, **kwargs):
        return document, receipt or {"replay": {"passed": True}}
    monkeypatch.setattr(p, "load_script", lambda *args: SimpleNamespace(qualification_receipt=helper))
    kwargs = dict(spec=spec(), banks="banks", pair_cpu_receipt=panel,
        raw_bindings={"query": "actual"}, model_inputs={"query": "actual"},
        codec_inventory="i", codec_reference_inventory="r", codec_combined_inventory="c",
        completed_grid="g", prefix_run="prefix", failed_screen="old")
    return artifact, panel, kwargs


def valid_document():
    return {"schema": p.QUALIFICATION_SCHEMA, "status": p.QUALIFICATION_STATUS,
            "pair": spec().as_dict(), "matching_FP32_execution_passed_all21": True}


def test_old_case2_proof_schema_rejected_even_if_helper_claimed_pass(monkeypatch, tmp_path):
    doc = valid_document()
    doc["schema"] = "posteriorforge.prior_native_conditionals.v1"
    path, _, kwargs = gate_setup(monkeypatch, tmp_path, doc)
    with pytest.raises(RuntimeError, match="no old case2 transfer"):
        p.qualification_receipt(path, **kwargs)


def test_own_pair_proof_for_another_case_is_rejected(monkeypatch, tmp_path):
    doc = valid_document()
    doc["pair"] = spec(2).as_dict()
    path, _, kwargs = gate_setup(monkeypatch, tmp_path, doc)
    with pytest.raises(RuntimeError, match="another pair"):
        p.qualification_receipt(path, **kwargs)


@pytest.mark.parametrize("field,value", [("status", "failed"),
                                        ("matching_FP32_execution_passed_all21", False)])
def test_failed_own_pair_proof_cannot_activate(monkeypatch, tmp_path, field, value):
    doc = valid_document()
    doc[field] = value
    path, _, kwargs = gate_setup(monkeypatch, tmp_path, doc)
    with pytest.raises(RuntimeError, match="successful all21 proof"):
        p.qualification_receipt(path, **kwargs)


def test_modified_qualification_file_rejected(monkeypatch, tmp_path):
    path, _, kwargs = gate_setup(monkeypatch, tmp_path, valid_document())
    path.write_text("{}")
    with pytest.raises(RuntimeError, match="proof SHA"):
        p.qualification_receipt(path, **kwargs)


def test_modified_CPU_panel_file_rejected(monkeypatch, tmp_path):
    path, panel, kwargs = gate_setup(monkeypatch, tmp_path, valid_document())
    panel.write_text("{}")
    with pytest.raises(RuntimeError, match="CPU panel"):
        p.qualification_receipt(path, **kwargs)


def test_actual_binding_arguments_are_forwarded_without_anchor_substitution(monkeypatch, tmp_path):
    path, _, kwargs = gate_setup(monkeypatch, tmp_path, valid_document())
    seen = {}
    def helper(actual_spec, artifact_path, **binding):
        seen.update(spec=actual_spec, path=artifact_path, binding=binding)
        return valid_document(), {"replay": {"passed": True}}
    monkeypatch.setattr(p, "load_script", lambda *args: SimpleNamespace(qualification_receipt=helper))
    p.qualification_receipt(path, **kwargs)
    assert seen["spec"] == kwargs["spec"]
    assert seen["binding"]["raw_bindings"] is kwargs["raw_bindings"]
    assert seen["binding"]["model_inputs"] is kwargs["model_inputs"]
    assert seen["binding"]["completion_receipt"] == kwargs["completed_grid"]
    assert seen["binding"]["prefix_run"] == kwargs["prefix_run"]


def test_actual_wrongbank_or_query_gate_error_not_suppressed(monkeypatch, tmp_path):
    path, _, kwargs = gate_setup(monkeypatch, tmp_path, valid_document())
    def wrong_binding(*args, **binding):
        raise RuntimeError("actual bank/query/model binding differs from own pair")
    monkeypatch.setattr(p, "load_script", lambda *args: SimpleNamespace(qualification_receipt=wrong_binding))
    with pytest.raises(RuntimeError, match="bank/query/model"):
        p.qualification_receipt(path, **kwargs)


def test_private_failure_observer_keeps_algorithm_and_new_source_guard():
    safe = p._failure_observer()
    assert safe.__code__ is p.ancestor._safe_capture_failure_footer.__code__
    capture = safe.__globals__["_capture_failure_footer"]
    assert capture.__code__ is p.ancestor._capture_failure_footer.__code__
    assert capture.__globals__["require_sources"] is p.require_sources
    assert capture.__globals__["source_sha256"] is p.source_sha256
    assert p.ancestor._safe_capture_failure_footer.__globals__["_capture_failure_footer"] is p.ancestor._capture_failure_footer


def test_model_and_mask_target_algorithms_are_frozen_objects():
    assert p.model_bindings is p.ancestor.failed.model_bindings
    assert p.backend is not p.ancestor.failed.backend
    assert p.backend.cached is p.ancestor.failed.backend.cached
    assert p.codec is p.ancestor.failed.codec
    assert p.backend.NATIVE_STAGES == (6, 10, 14)
    rules = [p.backend.stage_policy(i) for i in range(21)]
    assert len(rules) == 21
    assert rules[6]["stage_index"] == 6 and rules[10]["stage_index"] == 10


def test_all21_train_query_masks_and_current_target_no_leakage():
    # Deterministic engineering arrays only; no model or posterior draws.
    owner = p.backend.cached
    theta = (np.arange(4096 * 21, dtype=np.float32).reshape(4096, 21) + 1) / np.float32(128)
    stats = (np.arange(4096 * 15, dtype=np.float32).reshape(4096, 15) + 1) / np.float32(64)
    theta_bytes, stats_bytes = theta.tobytes(), stats.tobytes()
    for stage in range(21):
        rule = owner.stage_policy(stage)
        prepared, observations = owner.mask_training(theta, stats, stage)
        query = np.concatenate((stats[:8], theta[:8, :stage]), axis=1)
        masked = owner.mask_query(query, stage)
        predictors = np.concatenate((observations[:8], prepared[:8, :stage]), axis=1)
        assert masked.tobytes() == predictors.tobytes()
        assert prepared[:, stage].tobytes() == theta[:, stage].tobytes()
        assert rule["target_original_physical_slot"] == p.codec.ORDER[stage]
        assert rule["target_coordinate"] == "prior_latent_" + p.codec.metadata()["physical_parameter_order"][p.codec.ORDER[stage]]
        changed_target = theta.copy()
        changed_target[:, stage:] += np.float32(100)
        target_prepared, target_obs = owner.mask_training(changed_target, stats, stage)
        # Current/future labels never enter current predictors.
        assert np.concatenate((target_obs, target_prepared[:, :stage]), axis=1).tobytes() == np.concatenate((observations, prepared[:, :stage]), axis=1).tobytes()
        assert target_prepared[:, stage].tobytes() != prepared[:, stage].tobytes()
        if stage < 3:
            assert prepared is theta and observations is stats
        else:
            assert set(range(3)).issubset(rule["ordered_latent_prefix_indices_kept"])
            changed_query = query.copy()
            changed_query[:, rule["feature_columns_zeroed"]] += np.float32(1000)
            assert owner.mask_query(changed_query, stage).tobytes() == masked.tobytes()
            assert np.all(masked[:, rule["feature_columns_zeroed"]] == 0)
    assert theta.tobytes() == theta_bytes and stats.tobytes() == stats_bytes


def test_withinprocess_registry_or_module_rebinding_is_rejected(monkeypatch):
    p._check_configuration()
    monkeypatch.setattr(p, "QUALIFICATION_SHA256_BY_PAIR", {(20261111, 0): "a" * 64})
    with pytest.raises(RuntimeError, match="configuration changed"):
        p._check_configuration()


def test_private_bound_source_guard_detects_tampered_pair_namespace(monkeypatch):
    function = p._bind_run(spec(), "a" * 64, "panel")
    # The outer source check is separately exercised with committed bytes.
    monkeypatch.setattr(p, "require_sources", lambda: True)
    function.__globals__["CASE"] = 2
    with pytest.raises(RuntimeError, match="private pair run global"):
        function.__globals__["require_sources"]()


def test_numeric_replay_cannot_be_false_under_pass_header(monkeypatch, tmp_path):
    path, _, kwargs = gate_setup(monkeypatch, tmp_path, valid_document(),
                                {"replay": {"passed": False}})
    with pytest.raises(RuntimeError, match="all21 proof failed"):
        p.qualification_receipt(path, **kwargs)


def test_bool_pair_field_cannot_match_integer_case(monkeypatch, tmp_path):
    doc = valid_document()
    doc["pair"]["case"] = False
    path, _, kwargs = gate_setup(monkeypatch, tmp_path, doc)
    with pytest.raises((ValueError, RuntimeError), match="PairSpec|fixed case"):
        p.qualification_receipt(path, **kwargs)


def test_actual_case2_saved_prefix_path_rejected_for_case0_before_encoding():
    from pathlib import Path
    root = Path("/scratch/project/eu-26-17/eubrieucb/posteriorforge")
    completion = root / "results/accuracy-ss15-v1/sigma-native-grid-completion-01/completion.json"
    wrong_parent = root / "results/accuracy-ss15-v1/sigma-native-grid-01/seed-20261111-case-2"
    if not completion.is_file() or not (wrong_parent / "manifest.json").is_file():
        pytest.skip("Karolina authenticated old24 custody fixture unavailable")
    with pytest.raises(ValueError, match="different actual pair"):
        p.pairs.load_prefix(spec(), completion_receipt=completion, parent=wrong_parent)


@pytest.mark.parametrize("wrong_actual_inputs", [True, False])
@pytest.mark.parametrize("old_family", ["case2_v1_passed", "case0_v1_failed"])
def test_real_production_gate_real_qualifier_rejects_actual_v1_proof_for_v2(wrong_actual_inputs, old_family):
    """Real input/qualifier path; only private registry lookup presents forbidden5ec."""
    from pathlib import Path
    from types import FunctionType
    root = Path("/scratch/project/eu-26-17/eubrieucb/posteriorforge")
    phase = root / "results/accuracy-prior-coordinates-v1"
    repo = Path(p.__file__).resolve().parents[2]
    actual_old_proof = phase / ("native-conditionals-01/manifest.json" if old_family == "case2_v1_passed" else "pair-conditionals-seed20261111-case0-01/manifest.json")
    cpu = phase / "pair-preflight-combined-01/receipt.json"
    failed = phase / "native-pilot-01/manifest.json"
    if p.QUALIFIER_SHA256 is None:
        pytest.skip("real new qualifier source has not yet frozen; no interface substituted")
    if not all(x.is_file() for x in (actual_old_proof, cpu, failed)):
        pytest.skip("Karolina authenticated actual proof/input fixture unavailable")
    old_sha = ("5ec69eb7aadbd74c7d96ee534b71d0d8bee2b6dd6f6e3d2f44e74607452f682b" if old_family == "case2_v1_passed" else "fbed99d9fb256bb72a49cb03bebb820d805ac2cbafaa4ebac4deb08bf3160734")
    assert p.sha256_file(actual_old_proof) == old_sha
    assert p.sha256_file(failed) == "c81001f878bc1cd04d144c4d98a22e2ab5a089698b58978e10773c33654bd488"
    actual = spec()
    if wrong_actual_inputs:
        header = json.loads(failed.read_bytes())
        raw, model = header["raw_scientific_bindings"], header["model_inputs"]
    else:
        context, query, raw = p.pairs.raw_inputs(actual, banks=root / "results/banks-v1")
        _, _, _, model, _ = p.model_bindings(context, query)
    original_globals = dict(p.qualification_receipt.__globals__)
    namespace = dict(original_globals)
    namespace["qualification_sha256"] = lambda owner: old_sha
    gate = FunctionType(p.qualification_receipt.__code__, namespace,
                        "explicit_negative_registry_fixture_only",
                        p.qualification_receipt.__defaults__, p.qualification_receipt.__closure__)
    gate.__kwdefaults__ = p.qualification_receipt.__kwdefaults__
    assert gate.__code__ is p.qualification_receipt.__code__
    assert namespace["load_script"] is p.load_script
    expected = "actual producer pair/raw/model inputs differ" if wrong_actual_inputs else "v2|new pair proof schema/pass"
    with pytest.raises(ValueError, match=expected):
        gate(actual_old_proof, spec=actual, banks=root / "results/banks-v1",
            pair_cpu_receipt=cpu, raw_bindings=raw, model_inputs=model,
            codec_inventory=phase / "codec-inventory-inference-01/manifest.json",
            codec_reference_inventory=phase / "codec-inventory-reference-01/manifest.json",
            codec_combined_inventory=repo / "evidence/accuracy-prior-coordinates-v1/codec-inventory-01/CPU-receipt.json",
            completed_grid=root / "results/accuracy-ss15-v1/sigma-native-grid-completion-01/completion.json",
            prefix_run=root / "results/accuracy-ss15-v1/sigma-native-pilot-01",
            failed_screen=phase / "conditionals-01/manifest.json")
    assert all(p.qualification_receipt.__globals__[key] is value
               for key, value in original_globals.items())


def test_new_adapter_method_context_and_prediction_codeobjects_exact_original():
    import inspect
    adapter, scope, check, namespace = p._adapter_bindings(spec(), "a" * 64)
    original = p.ancestor.failed.PriorCoordinateStageAdapter
    assert adapter is not original and namespace["backend"] is p.backend
    assert namespace["BACKEND_SHA256"] is p.NATIVE_BACKEND_SHA256
    for name, value in vars(original).items():
        if inspect.isfunction(value):
            assert vars(adapter)[name] is not value
            assert vars(adapter)[name].__code__ is value.__code__
            assert vars(adapter)[name].__globals__ is namespace
        elif isinstance(value, property):
            assert vars(adapter)[name].fget.__code__ is value.fget.__code__
            assert vars(adapter)[name].fget.__globals__ is namespace
    assert scope.__wrapped__.__code__ is p.ancestor.failed.engine_scope.__wrapped__.__code__
    assert check.__code__ is p.ancestor.failed.validate_prediction.__code__
    assert original.prepare.__globals__["backend"] is p.ancestor.failed.backend


def test_new_prediction_checker_accepts_only_new_backend_and_producer_sources():
    _, _, check, _ = p._adapter_bindings(spec(), "a" * 64)
    proof = {"module_sha256": p.source_sha256(), "backend_source_sha256": p.backend.source_sha256(),
             "codec_sha256": p.ancestor.failed.CODEC_SHA256, "mode": p.MODE, "stage": 14,
             "member": 0, "actual_head_contexts": 1, "corrected_CDF_calls": 1,
             "witness": {"passed": True}}
    report = {}
    check({"prior_coordinate_execution_proof": proof}, 14, 0, report=report)
    assert report["actual_head_contexts"] == 1
    for key, bad in (("module_sha256", p.prior_v1.source_sha256()),
                     ("backend_source_sha256", p.ancestor.failed.backend.source_sha256())):
        wrong = dict(proof);wrong[key] = bad
        with pytest.raises(RuntimeError, match="provenance differs"):
            check({"prior_coordinate_execution_proof": wrong}, 14, 0, report={})


@pytest.mark.parametrize("mutation", ["adapter_global", "method_code", "property", "scope_generator", "prediction_code"])
def test_new_private_adapter_code_global_seals_reject_mutation(monkeypatch, mutation):
    fn = p._bind_run(spec(), "a" * 64, "panel")
    monkeypatch.setattr(p, "require_sources", lambda: True)
    scope = fn.__globals__["engine_scope"]
    namespace = scope.__wrapped__.__globals__
    adapter = namespace["PriorCoordinateStageAdapter"]
    if mutation == "adapter_global":
        namespace["backend"] = SimpleNamespace()
    elif mutation == "method_code":
        adapter.prepare.__code__ = (lambda self, *a, **kw: None).__code__
    elif mutation == "property":
        adapter.regressor = property(lambda self: None)
    elif mutation == "scope_generator":
        scope.__wrapped__ = lambda *a, **kw: None
    else:
        fn.__globals__["validate_prediction"].__code__ = (lambda *a, **kw: None).__code__
    with pytest.raises(RuntimeError, match="private v2"):
        fn.__globals__["require_sources"]()


def test_v2_run_two_header_literals_and_protocol_bindings_correct():
    fn = p._bind_run(spec(), "a" * 64, "panel")
    assert fn.__globals__["NATIVE_PROTOCOL_SHA256"] == p.PREREGISTRATION_SHA256
    assert fn.__globals__["backend"] is p.backend
    assert "FP32; 18 cached stages, all8 original-control at6/10/14" in fn.__code__.co_consts
    assert "INNER frozen88 precision owner only; outer native all8 stages6/10/14 are explicit" in fn.__code__.co_consts
    assert "FP32; 19 cached stages, all8 original-control at6/10" not in fn.__code__.co_consts


def test_untampered_private_v2_run_guard_passes_with_only_source_commit_boundary_stub(monkeypatch):
    before = dict(p.ancestor._RUN_BODY.__globals__)
    fn = p._bind_run(spec(), "a" * 64, "panel")
    monkeypatch.setattr(p, "require_sources", lambda: True)
    assert fn.__globals__["require_sources"]() is True
    assert fn.__globals__["engine_scope"].__wrapped__.__globals__["require_sources"]() is True
    assert all(p.ancestor._RUN_BODY.__globals__[key] is value for key, value in before.items())


def test_actual_registered_same_pair_v2_proof_positive_replay():
    """Actual positive proof/interface replay; no mocks, NN, GPU or new draws."""
    from pathlib import Path
    root = Path("/scratch/project/eu-26-17/eubrieucb/posteriorforge")
    phase = root / "results/accuracy-prior-coordinates-v1"
    proof = phase / "pair-conditionals-v2-seed20261111-case0-01/manifest.json"
    if not proof.is_file():
        pytest.skip("actual completed v2 case0 conditional artifact unavailable")
    owner = spec()
    assert p.qualification_sha256(owner) == "aa8aff311d8f7fab6d774337510bb5f66abed309300c995267e98fa7f3e2a88e"
    context, query, raw = p.pairs.raw_inputs(owner, banks=root / "results/banks-v1")
    _, _, _, model, _ = p.model_bindings(context, query)
    repo = Path(p.__file__).resolve().parents[2]
    before = state()
    document, receipt = p.qualification_receipt(
        proof, spec=owner, banks=root / "results/banks-v1",
        pair_cpu_receipt=phase / "pair-preflight-combined-01/receipt.json",
        raw_bindings=raw, model_inputs=model,
        codec_inventory=phase / "codec-inventory-inference-01/manifest.json",
        codec_reference_inventory=phase / "codec-inventory-reference-01/manifest.json",
        codec_combined_inventory=repo / "evidence/accuracy-prior-coordinates-v1/codec-inventory-01/CPU-receipt.json",
        completed_grid=root / "results/accuracy-ss15-v1/sigma-native-grid-completion-01/completion.json",
        prefix_run=root / "results/accuracy-ss15-v1/sigma-native-pilot-01",
        failed_screen=phase / "conditionals-01/manifest.json")
    assert document["schema"] == p.QUALIFICATION_SCHEMA and document["status"] == p.QUALIFICATION_STATUS
    assert document["matching_FP32_execution_passed_all21"] is True
    assert document["FP32_failed_stages"] == [] and receipt["replay"]["passed"] is True
    assert receipt["replay"]["native_all8_stages"] == [6, 10, 14]
    assert document["compatibility_adapter"]["patched_call_count"] == 1008
    assert state() == before
