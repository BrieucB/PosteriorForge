"""CPU-only v2 physical-boundary fixture from saved V1 neural values only.

6b6c equality is not a claim about any v2 neural cohort.

No model forward, CUDA initialization, new neural draw, or reconstructed c810
footer. The synthetic engine/callback journal is visibly test-only. Codec,
Float64 FIFO, original support/sampler, finalization, source guards and output
success code are the actual reviewed production objects.
"""
from contextlib import contextmanager
from importlib import import_module
from pathlib import Path
from types import FunctionType, SimpleNamespace
import hashlib
import json
import pickle
import random

import numpy as np
import pytest

ROOT = Path("/scratch/project/eu-26-17/eubrieucb/posteriorforge")
PHASE = ROOT / "results/accuracy-prior-coordinates-v1"
FAILED_SHA = "c81001f878bc1cd04d144c4d98a22e2ab5a089698b58978e10773c33654bd488"
PRODUCER_SHA = "a30e0c15513eb0c1a6ed3c68f0781327edef4714a244a119a3a4d4c68f8780bf"
HELPER_SHA = "f8823ae75e7ae2eea6f1aa667af48ecf5c892450361b5abd490ccda36a28f7e0"


def test_v2_physical_boundary_only_v1_saved_case2_archived_fixture_real_decode_FIFO_support_and_success_footer(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    required = [
        PHASE / "native-pilot-01/manifest.json",
        PHASE / "native-pilot-01/latent-proposal-waves.npz",
        PHASE / "native-pilot-01/decoded-physical-proposal-waves.npz",
        PHASE / "forensic-replay-01/manifest.json",
        ROOT / "cache/model/tabpfn-v2-regressor.ckpt",
    ]
    if not all(path.is_file() for path in required):
        pytest.skip("Karolina archived-neural engineering fixture unavailable")
    from posteriorforge import performance_prior_physical_fifo_v1 as p
    from posteriorforge import performance_prior_pair_v2 as pair

    from posteriorforge.artifacts import sha256_file
    from posteriorforge.science.analytical_benchmark_v2 import array_descriptor
    from posteriorforge import reference_grid

    assert p.source_sha256() == PRODUCER_SHA
    assert p.physical_fifo.source_sha256() == HELPER_SHA
    # These remain REAL committed source/closure guards, not test substitutes.
    p.require_sources()
    assert p.runtime_source_inventory(require_committed=True)["committed_HEAD_exact"]
    failed_bytes = required[0].read_bytes()
    assert hashlib.sha256(failed_bytes).hexdigest() == FAILED_SHA
    original = json.loads(failed_bytes)
    assert sha256_file(required[1]) == original["latent_wave_archive"]["sha256"]
    assert sha256_file(required[2]) == original["decoded_wave_archive"]["sha256"]
    with np.load(required[1], allow_pickle=False) as archive:
        latent = archive["wave_0000"]
    with np.load(required[2], allow_pickle=False) as archive:
        physical = archive["wave_0000"]
    assert latent.dtype == np.float32 and latent.shape == (51200, 21)
    assert physical.dtype == np.float64 and physical.shape == (51200, 21)
    expected = json.loads(required[3].read_bytes())["samples"]
    assert array_descriptor(physical[:50000]) == expected
    assert expected["sha256"] == "6b6c36843d6c213d086c6712aacc68dc1cb11148163ebe24bc74119a102b54eb"
    latent_bytes, physical_bytes = latent.tobytes(), physical.tobytes()

    # Import actual SDK translation/original sampler BEFORE mocking CUDA boundary.
    sdk = import_module("tabpfn.regressor")
    p.physical_fifo._original_sampler()
    assert not torch.cuda.is_initialized()
    original_threads = torch.get_num_threads()
    original_precision = torch.get_float32_matmul_precision()
    original_tf32 = torch.backends.cuda.matmul.allow_tf32
    lifecycle, engines, callback_sources = [], [], []
    spec = pair.pairs.PairSpec(20261111, 2)
    fixture_proof = tmp_path / "EXPLICIT_TEST_ONLY_OWN_PAIR_GATE.json"
    fixture_proof.write_text('{"test_fixture":true,"not_an_actual_numerical_qualification":true}\n')
    fixture_proof_sha = sha256_file(fixture_proof)
    bound = pair._bind_run(spec, fixture_proof_sha, tmp_path / "EXPLICIT_TEST_ONLY_CPU_PANEL.json")
    actual_source_globals = dict(bound.__globals__)

    repo = Path(pair.__file__).resolve().parents[2]
    new_fixture_sources = {
        str(Path(pair.__file__).resolve()): sha256_file(pair.__file__),
        str(Path(pair.pairs.__file__).resolve()): sha256_file(pair.pairs.__file__),
        str(Path(pair.backend.__file__).resolve()): sha256_file(pair.backend.__file__),
        str(Path(pair.prior_v1.__file__).resolve()): sha256_file(pair.prior_v1.__file__),
        str(repo / "scripts/qualify_prior_coordinate_pair_v2.py"): sha256_file(repo / "scripts/qualify_prior_coordinate_pair_v2.py"),
        str(repo / pair.PREREGISTRATION_PATH): pair.PREREGISTRATION_SHA256,
    }

    def current_pair_sources():
        # Existing a30/146 committed closure stays REAL. The NEW source pins/
        # proof/commit boundary are explicitly fixture-only before activation.
        p.require_sources()
        pair.pairs.require_sources()
        pair._check_configuration()
        assert all(sha256_file(path) == digest for path, digest in new_fixture_sources.items())
        assert pair._bind_run.__code__ is pair._FUNCTION_SEALS["_bind_run"][1]
        return True

    def fixture_inventory(*, require_committed):
        inherited = pair.prior_v1.runtime_source_inventory(require_committed=True)
        current_pair_sources()
        sources = dict(inherited["sources"])
        sources.update({str(Path(name).relative_to(repo)): digest
                        for name, digest in new_fixture_sources.items()
                        if Path(name) != Path(pair.__file__).resolve()})
        sources["src/posteriorforge/performance_prior_physical_fifo_v1.py"] = p.source_sha256()
        return {
            **inherited, "sources": sources, "source_count": len(sources),
            "committed_HEAD_exact": False, "test_fixture_new_sources_current_bytes_only": True,
            "own_producer_source_sha256": pair.source_sha256(),
            "inventory_sha256": hashlib.sha256(json.dumps(
                sources, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}

    policy_globals = dict(pair.execution_metadata.__globals__)
    policy_globals.update(require_sources=current_pair_sources,
        runtime_source_inventory=fixture_inventory,
        PAIR_MODULE_SHA256=sha256_file(pair.pairs.__file__),
        QUALIFIER_SHA256=sha256_file(repo / "scripts/qualify_prior_coordinate_pair_v2.py"))
    policy_function = FunctionType(pair.execution_metadata.__code__, policy_globals,
                                   "exact_pair_policy_current_source_fixture")
    def fixture_policy():
        return policy_function(spec, fixture_proof_sha)

    def fixture_pair_qualification(path, **kwargs):
        assert path == fixture_proof
        assert kwargs["raw_bindings"] == original["raw_scientific_bindings"]
        assert kwargs["model_inputs"] == original["model_inputs"]
        return {"schema": pair.QUALIFICATION_SCHEMA, "status": pair.QUALIFICATION_STATUS,
                "pair": spec.as_dict(), "matching_FP32_execution_passed_all21": True,
                "test_only_not_an_actual_numerical_qualification": True}, {
                    "replay": {"passed": True}, "test_only_gate_mocked": True,
                    "numerical_qualification_claimed": False}


    class FixtureEngine:
        """Only neural generation/cache telemetry is mocked, never physical math."""
        regressor = None

        def __init__(self, descriptor):
            self.descriptor = descriptor
            self.prepare_count = self.actual_heads = self.shared_densities = 0
            self.stages, self.training_inputs, self.query_calls = [], [], []
            self.stage_index = None

        def prepare(self, theta, obs, stage):
            assert theta.shape == (4096, 21) and obs.shape == (4096, 15)
            assert theta.dtype == obs.dtype == np.float32
            assert self.stage_index is None
            self.stage_index = stage
            self.prepare_count += 1
            self.training_inputs.append({"fixture_neural_prepare": True, "stage": stage})
            return {"fixture_only": True, "fit_stock_cache_seconds": 0, "resident_move_seconds": 0}

        def inspect_cache(self):
            return {"fixture_neural_engine": True, "tensor_records": []}

        def close(self):
            if self.stage_index is not None:
                self.stages.append({"fixture_only": True, "stage_index": self.stage_index})
                self.stage_index = None

        def metadata(self):
            return {"fixture_only": True, "model_calls": 0,
                    "not_recovered_failed_request_counters": True,
                    "prepare_count": self.prepare_count,
                    "actual_head_contexts": self.actual_heads,
                    "shared_densities": self.shared_densities,
                    "completed_stage_scopes": list(self.stages),
                    "actual_latent_training_inputs": list(self.training_inputs),
                    "actual_latent_query_calls": list(self.query_calls)}

    @contextmanager
    def fixture_engine_scope(checkpoint, *, cdf_descriptor):
        assert checkpoint == required[4]
        engine = FixtureEngine(cdf_descriptor)
        engines.append(engine)
        lifecycle.append("fixture_engine_enter")
        try:
            yield engine
        finally:
            engine.close()
            lifecycle.append("fixture_engine_exit")

    frozen = p.physical_sampling_namespace()

    def fixture_stage_factory(component, torch_argument, generators, counters, *, prediction_check=None):
        assert torch_argument is torch and prediction_check is not None
        assert all(g.device.type == "cpu" for g in generators.values())
        callback_sources.append(prediction_check)

        def draw(engine, stage, query_for, count, chunk_size):
            assert engine is component and count == 51200 and chunk_size == 1024
            assert query_for([0]).shape == (1, 15 + stage)
            heads = 8 if stage == 0 else 1
            # Actual tiny deterministic CDF translations exercise alias/counter
            # restoration. They are NOT model predictions or c810's1138 calls.
            borders = torch.tensor([-2., .5, 3.], dtype=torch.float32, device="cpu")
            logits = torch.zeros((1, 2), dtype=torch.float32, device="cpu")
            for member in range(heads):
                sdk.translate_probs_across_borders(logits, frm=borders, to=borders)
                engine.query_calls.append({"fixture_only": True, "stage": stage, "member": member})
            engine.actual_heads += heads
            if stage == 0:
                counters["stage0_shared_predictions"] += 1
                engine.shared_densities += 1
            else:
                counters["component_predictions"] += 1
            proof = {
                "module_sha256": pair.source_sha256(),
                "backend_source_sha256": pair.backend.source_sha256(),
                "codec_sha256": p.failed.CODEC_SHA256, "mode": p.failed.MODE,
                "stage": stage, "member": None if stage == 0 else 0,
                "actual_head_contexts": heads, "corrected_CDF_calls": heads,
                "witness": {"passed": True, "fixture_neural_witness": True},
            }
            # Execute the actual producer71 callback/footer accounting code.
            prediction_check({"prior_coordinate_execution_proof": proof},
                             stage, None if stage == 0 else 0)
            return latent[:, stage].copy(), {"model_query_chunks": 1,
                   "fixture_neural_returns": True, "new_neural_or_uniform_draws": 0}
        return draw

    fixture_frozen = SimpleNamespace(**vars(frozen))
    fixture_frozen.component_stage_sampler = fixture_stage_factory
    fixture_clean = SimpleNamespace(**vars(p._CLEAN_PROXY))
    fixture_clean.frozen_sampling = lambda: fixture_frozen
    identity = bound.__globals__["runtime_identity"]

    def fixture_identity():
        return {**identity(), "test_fixture": {
            "archived_neural_returns_only": True, "model_calls": 0,
            "CUDA_boundary_mocked": True, "budget_authorization_mocked": True,
            "own_pair_proof_gate_explicitly_mocked": True, "new_source_commit_boundary_fixture_only": True,
            "new_pair_source_pins_current_bytes_fixture_only": True,
            "not_a_production_or_numerical_qualification": True,
            "neural_returns_are_saved_v1_case2_NOT_v2_neural_outputs": True,
            "6b6c_equality_only_physical_boundary_regression_NOT_v2_NN_prediction": True,
            "synthetic_neural_callback_count": 28,
            "not_recovered_c810_execution_or_RNG_footer": True,
            "physical_decode_FIFO_support_success_code_real": True}}

    # Local import inside real run uses this explicitly fixture-labelled budget.
    monkeypatch.setattr(reference_grid, "_require_budgeted_gpu",
                        lambda ledger: {"gpus": 1, "fixture_only": True, "job_id": "CPU_TEST_NO_GPU"})

    def forbidden(*args, **kwargs):
        raise AssertionError("no model loading, CUDA initialization or new random draws in fixture")

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    monkeypatch.setattr(torch, "load", forbidden)
    for name in ("rand", "randint", "randn", "normal"):
        monkeypatch.setattr(torch, name, forbidden)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)  # test boundary only
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda seed: None)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "CPU_TEST_FIXTURE_NO_GPU")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_rng_state",
                        lambda: torch.zeros(16, dtype=torch.uint8, device="cpu"))

    namespace = dict(actual_source_globals)
    namespace.update(clean=fixture_clean, engine_scope=fixture_engine_scope,
                     runtime_identity=fixture_identity, require_sources=current_pair_sources,
                     execution_metadata=fixture_policy,
                     qualification_receipt=fixture_pair_qualification)
    run_fixture = FunctionType(bound.__code__, namespace, "run_pair_archived_neural_CPU_fixture",
                               bound.__defaults__, bound.__closure__)
    run_fixture.__kwdefaults__ = p._RUN_BODY.__kwdefaults__
    assert run_fixture.__code__ is pair._FIXED_RUN_CODE
    assert run_fixture.__code__.co_code == p._RUN_BODY.__code__.co_code
    assert namespace["physical_fifo"] is p.physical_fifo
    assert namespace["backend"] is pair.backend
    assert namespace["validate_prediction"] is bound.__globals__["validate_prediction"]
    assert namespace["require_sources"] is current_pair_sources
    assert namespace["execution_metadata"] is fixture_policy
    assert namespace["wave"] is p.failed.wave

    output = tmp_path / "EXPLICIT_CPU_ARCHIVED_NEURAL_TEST_FIXTURE"
    defaults = pickle.dumps((np.random.get_state(), random.getstate()), protocol=5)
    try:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        with torch.random.fork_rng(devices=[]):
            status = run_fixture(
                banks=ROOT / "results/banks-v1", output=output, checkpoint=required[4],
                ledger=ROOT / "budget/ledger.json", reference=ROOT / "results/references-v1/case-2",
                reference_grid=ROOT / "results/reference-grids-v1/case-2",
                qualification=fixture_proof,
                codec_inventory=PHASE / "codec-inventory-inference-01/manifest.json",
                codec_reference_inventory=PHASE / "codec-inventory-reference-01/manifest.json",
                codec_combined_inventory=Path(p.__file__).resolve().parents[2] / "evidence/accuracy-prior-coordinates-v1/codec-inventory-01/CPU-receipt.json",
                completed_grid=ROOT / "results/accuracy-ss15-v1/sigma-native-grid-completion-01/completion.json",
                prefix_run=ROOT / "results/accuracy-ss15-v1/sigma-native-grid-01/seed-20261111-case-2",
                failed_screen=PHASE / "conditionals-01/manifest.json")
    finally:
        torch.set_num_threads(original_threads)
        torch.set_float32_matmul_precision(original_precision)
        torch.backends.cuda.matmul.allow_tf32 = original_tf32
    assert not torch.cuda.is_initialized()
    assert pickle.dumps((np.random.get_state(), random.getstate()), protocol=5) == defaults
    assert status["status"] == "complete" and status["schema"] == pair.SCHEMA
    assert status["variant"] == pair.VARIANT and status["pair_spec"] == spec.as_dict()
    assert status["context_seed"] == 20261111 and status["case"] == 2
    assert status["native_all8_stages"] == [6, 10, 14]
    assert status["native_protocol_sha256"] == pair.PREREGISTRATION_SHA256
    assert status["neural_execution_description"] == "FP32; 18 cached stages, all8 original-control at6/10/14"
    assert status["fallback_map_scope"].endswith("stages6/10/14 are explicit")
    assert status["test_fixture"]["neural_returns_are_saved_v1_case2_NOT_v2_neural_outputs"]
    assert status["matching_conditional_qualification_sha256"] == fixture_proof_sha
    assert status["matched_execution_screen"]["test_only_gate_mocked"]
    assert status["test_fixture"]["not_a_production_or_numerical_qualification"] is True
    assert status["test_fixture"]["not_recovered_c810_execution_or_RNG_footer"]
    assert status["default_rng_preserved"] and status["default_rng_before"] == status["default_rng_after"]
    assert status["sources_unchanged"] and status["completed_stage_indices"] == list(range(21))
    assert status["expected_stage_prepare_count"] == 21 and status["generated_proposal_wave_count"] == 1
    assert status["execution_counters"]["fixture_only"]
    assert status["execution_counters"]["prepare_count"] == 21
    assert status["actual_prediction_checks"]["actual_head_contexts"] == 28
    assert status["expected_patched_call_count"] == status["compatibility_adapter"]["patched_call_count"] == 28
    assert status["compatibility_adapter"]["original_alias_restored"]
    assert status["compatibility_adapter"]["sdk_sources_unchanged"]
    assert status["compatibility_adapter"]["executed_corrected_cdf_code_sha256"] == status["compatibility_adapter"]["corrected_cdf_code_sha256"]
    assert status["proposal_buffer"]["generated_count"] == 51200
    assert status["proposal_buffer"]["original_api_delivered_count"] == 50000
    assert status["proposal_buffer"]["remaining_unused_buffer_count"] == 1200
    assert status["rejection"] == {"support_calls": 1, "proposed": 50000,
                                  "accepted_before_final_trim": 50000, "returned": 50000}
    assert status["proposal_buffer"]["original_support_check_code_and_closure_verified"]
    assert status["proposal_buffer"]["data_dtype"] == p.physical_fifo.DATA_DTYPE
    assert status["samples"] == expected
    with np.load(output / "samples.npz", allow_pickle=False) as archive:
        values = archive["samples"]
        assert values.dtype == np.float64 and values.shape == (50000, 21)
        assert values.tobytes() == physical[:50000].tobytes()
    assert len(status["stage_events"]) == len(status["execution_counters"]["completed_stage_scopes"]) == 21
    assert all(e["sampler"]["fixture_neural_returns"] for e in status["stage_events"])
    assert lifecycle == ["fixture_engine_enter", "fixture_engine_exit"] and len(callback_sources) == 1
    assert original["status"] == "failed" and original["completed_stage_indices"] == []
    assert required[0].read_bytes() == failed_bytes and not required[0].with_name("samples.npz").exists()
    assert latent.tobytes() == latent_bytes and physical.tobytes() == physical_bytes
    assert all(bound.__globals__[key] is value for key, value in actual_source_globals.items())
    p.require_sources()
