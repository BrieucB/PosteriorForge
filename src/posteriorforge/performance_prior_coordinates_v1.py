"""Cached-FP32 prior-coordinate pilot; new source and qualification identity.

Built with PriorLabs-TabPFN. Exactly 50,000 complete physical outputs for
seed 20261111 / case 2. Full generated latent waves are decoded strictly before
the unchanged FIFO and original whole-vector support. No density in physical
units is claimed. Execution stays disabled until fresh proof pins are set.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from contextlib import ExitStack, contextmanager
from importlib import import_module
from pathlib import Path

import numpy as np

from . import performance_ss15 as ancestry
from . import execution_prior_coordinates_v1 as backend
from . import prior_coordinates_v1 as codec
from . import representation_ss15 as representation
from . import execution_ss15_sampling as wave
from .artifacts import runtime_identity, sha256_file, write_json

SCHEMA = "posteriorforge.inference.prior_coordinates.v1"
VARIANT = "accuracy_prior_coordinates_cached_fp32_fifo_v1"
MODE = backend.CONTROL_MODE
COUNT = 50000
CONTEXT_SEED = 20261111
CASE = 2
ANCESTRY_SHA256 = "4099da3c19366997ab0800e5430b5bb1c129b4b3e535f678b3d4652233303480"
CODEC_SHA256 = "d8ce6c49e7b65c6ee55c59b7243d4ce943cbea28c92ebed0329798074aaa70c5"
BACKEND_SHA256 = None
QUALIFIER_FILENAME = "qualify_prior_conditionals_v1.py"
QUALIFIER_SHA256 = None
CONDITIONAL_QUALIFICATION_SHA256 = None
INVENTORY_HELPER_SHA256 = None
INVENTORY_SHA256 = None
PREREGISTRATION_SHA256 = "7c4aca04c4571960108c0296ec5d4770940ce0b51fa0a77cdd0c27182b4ed849"
clean = ancestry.clean
require = ancestry.require
load_script = ancestry.load_script
rng_hashes = ancestry.rng_hashes


def source_sha256():
    return sha256_file(__file__)


def require_sources():
    require(ancestry.source_sha256() == ANCESTRY_SHA256, "frozen orchestration ancestry changed")
    ancestry.require_sources()
    require(codec.source_sha256() == CODEC_SHA256, "prior codec changed")
    codec.require_sources()
    require(isinstance(BACKEND_SHA256, str) and backend.source_sha256() == BACKEND_SHA256,
            "fresh prior-coordinate backend is not pinned or changed")
    backend.require_sources()
    require(MODE == backend.CONTROL_MODE and tuple(codec.ORDER) == tuple(backend.ORDER),
            "cached FP32 or latent order changed")
    doc = Path(__file__).resolve().parents[2] / "docs/accuracy-prior-coordinate-pilot-v1.md"
    require(sha256_file(doc) == PREREGISTRATION_SHA256, "prior-coordinate preregistration changed")


def execution_metadata():
    require_sources()
    return {
        "schema": "posteriorforge.prior_coordinate_pilot.execution.v1",
        "module_sha256": source_sha256(),
        "backend": backend.execution_metadata(MODE),
        "codec": codec.metadata(),
        "observation_representation": representation.representation_metadata(),
        "sampler_source_sha256": clean.SAMPLING_SHA,
        "sampling_wave": wave.execution_metadata(),
        "stage0_assembler_source_sha256": ancestry.ALL21_SOURCE_SHA,
        "mode": MODE, "query_batch_size": 1024, "fallback_map": {},
        "postprocessing_dtype": "torch.float32",
        "decode": "actual complete latent wave, without an extra cast, before FIFO/support",
        "all_generated_latent_rows_archived": True,
        "unused_FIFO_tail_also_decoded": True,
        "old_numerical_qualification_inherited": False,
        "comparison_to_old_grid_changes_both_coordinates_and_precision": True,
        "isolated_causal_effect_of_coordinates_claimed": False,
        "physical_log_density_claimed": False,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "attribution": "Built with PriorLabs-TabPFN",
    }


def model_bindings(context, raw_query):
    """Encode ORIGINAL saved Float64 parameters before the single model cast."""
    from .science.analytical_benchmark_v2 import array_descriptor
    physical = context.theta
    require(isinstance(physical, np.ndarray) and physical.dtype == np.float64,
            "original Float64 physical context required")
    latent64 = codec.encode_physical(physical)
    latent32 = codec.encode_for_model(physical)
    require(latent32.tobytes() == latent64.astype(np.float32).tobytes(),
            "one-final-cast latent model bytes differ")
    ordered = codec.order_latents(latent32)
    obs64 = representation.transform_observations(context.obs)
    query64 = representation.transform_observations(raw_query)
    obs = representation.transform_for_model(context.obs)
    query = representation.transform_for_model(raw_query)
    model = {"context_theta": array_descriptor(ordered),
             "context_obs": array_descriptor(obs), "query": array_descriptor(query)}
    receipt = {
        "physical_theta_original_Float64": array_descriptor(physical),
        "latent_physical_slot_Float64_runtime": array_descriptor(latent64),
        "latent_physical_slot_model_Float32": array_descriptor(latent32),
        "ordered_latent_model_Float32": array_descriptor(ordered),
        "stage_to_physical_order": list(codec.ORDER),
        "physical_to_stage_order": list(codec.INVERSE_ORDER),
        "model_target_coordinate_system": "causal_prior_latent_sigma_first_stage_order",
        "ordered_equals_single_cast_latent_columns_pi": True,
        "runtime_Float64_provenance_not_cross_environment_byte_equality": True,
        "context_SS15_Float64_runtime": array_descriptor(obs64),
        "query_SS15_Float64_runtime": array_descriptor(query64),
        "model_inputs": model,
    }
    return ordered, obs, query, model, receipt


def decode_complete_wave(values):
    require(isinstance(values, np.ndarray) and values.ndim == 2 and values.shape[1] == 21,
            "complete ordered latent matrix required")
    require(values.dtype in (np.dtype("float32"), np.dtype("float64")),
            "declared Float32/Float64 latent samples only; no implicit cast")
    # The codec rejects NaN/Inf, endpoint saturation and failed declared RT.
    # Do not replace its failure by a rejected vector, clipping, or a redraw.
    physical = codec.decode_ordered(values)
    require(physical.dtype == np.float64 and physical.shape == values.shape
            and bool(np.isfinite(physical).all()), "strict complete-wave decode failed")
    return physical


def record_decode_wave(latent, latent_waves, status):
    """Retain every complete actual wave BEFORE strict decoding can fail."""
    from .science.analytical_benchmark_v2 import array_descriptor
    index = len(latent_waves)
    latent_waves.append(latent.copy(order="C"))
    record = {"wave_index": index, "row_count": len(latent),
              "actual_latent_draws": array_descriptor(latent),
              "input_dtype_unchanged_before_decode": True,
              "includes_unused_FIFO_tail": True, "strict_decode_passed": False}
    status["decode_wave_records"].append(record)
    status["actual_complete_generated_wave_count"] = len(latent_waves)
    try:
        physical = decode_complete_wave(latent)
    except BaseException as error:
        record.update(decode_error_type=type(error).__name__, decode_error=str(error))
        status.update(failed_decode_wave_index=index,
                      failed_decode_wave_descriptor=array_descriptor(latent))
        raise
    record.update(decoded_physical=array_descriptor(physical), strict_decode_passed=True)
    return physical


def archive_latent_waves(output, latent_waves):
    path = Path(output) / "latent-proposal-waves.npz"
    with path.open("xb") as stream:
        np.savez(stream, **{f"wave_{j:04d}": value for j, value in enumerate(latent_waves)})
    return {"file": path.name, "sha256": sha256_file(path),
            "keys": [f"wave_{j:04d}" for j in range(len(latent_waves))],
            "all_actual_complete_generated_waves_preserved": True,
            "all_generated_rows_including_unused_tail": True,
            "generated_row_count": sum(map(len, latent_waves)),
            "complete_generated_wave_count": len(latent_waves)}


class PriorCoordinateStageAdapter:
    """Per-stage Session and source-bound records around unchanged model math."""
    def __init__(self, checkpoint, *, cdf_descriptor):
        self.checkpoint = checkpoint
        self.cdf_descriptor = cdf_descriptor
        self.engine = self.session = None
        self.stage_index = None
        self.actual_heads = self.shared_densities = self.prepare_count = 0
        self.stages = []
        self.training_inputs = []
        self.query_calls = []
        self.failure = self.entered = False

    @property
    def regressor(self):
        return self.engine.regressor if self.engine is not None else None

    def prepare(self, theta, obs, stage):
        require(self.session is None, "previous prior-coordinate stage not closed")
        self.stage_index = stage
        self.failure = False
        self.stage_heads = 0
        self.members = set()
        self.prefill = self.model = None
        self.entered = False
        entry = backend.training_bindings(theta, obs, stage)
        masked_theta, masked_obs = backend.mask_training(theta, obs, stage)
        self.session = backend.PriorCoordinateSession(MODE)
        try:
            self.session.__enter__()
            self.entered = True
            self.engine = backend.PriorCoordinateStageExecutor(
                self.checkpoint, cdf_descriptor=self.cdf_descriptor, query_batch_size=1024)
            result = self.engine.prepare(masked_theta, masked_obs, stage)
            self.prepare_count += 1
            self.prefill = backend.prefill_witness_passed(self.session.report(), stage, mode=MODE)
            require(self.prefill["passed"], "actual FP32 prefill/cache witness failed")
            base = load_script("characterize_precision_approximation.py", ancestry.BASE_SHA)
            self.model = base.model_summary(
                self.engine.regressor, load_script("qualify_execution_cache.py", base.UTILITY_SHA))
            self.training_inputs.append(entry)
            return result
        except BaseException:
            self.failure = True
            self.close()
            raise

    def predict_component(self, member, query, *, capture_member=False, with_log_prob=False):
        from .science.analytical_benchmark_v2 import array_descriptor
        import torch
        require(self.engine is not None and self.session is not None, "prepared latent stage required")
        masked = backend.mask_query(query, self.stage_index)
        before = self.session.report()
        cdf_before = self.cdf_descriptor["patched_call_count"]
        try:
            prediction = self.engine.predict_component(
                member, masked, capture_member=capture_member, with_log_prob=with_log_prob)
            witness = backend.query_witness_delta(before, self.session.report(),
                stage=self.stage_index, member=member, mode=MODE)
            require(prediction.get("hybrid_execution_path") == "resident_cache",
                    "initial prior-coordinate policy must use cached-only route")
            require(self.cdf_descriptor["patched_call_count"] - cdf_before == 1,
                    "one actual corrected CDF call per head required")
            require(self.cdf_descriptor.get("executed_corrected_cdf_code_sha256")
                    == self.cdf_descriptor.get("corrected_cdf_code_sha256"),
                    "corrected CDF helper was not executed")
            logits, mass = prediction["logits"], prediction["component_raw_mass"]
            require(logits.dtype == torch.float32 and logits.device.type == "cpu"
                    and bool((torch.isfinite(logits) | torch.isneginf(logits)).all())
                    and bool(torch.isfinite(logits).any(-1).all()), "invalid FP32 latent logits")
            require(mass.dtype == torch.float64 and bool(torch.isfinite(mass).all())
                    and bool((mass > 0).all()) and bool((mass <= 1.00001).all())
                    and prediction["head_mass_envelope"] == 1.00001, "raw component mass guard failed")
            require(prediction["criterion"].borders.dtype == torch.float32
                    and bool(torch.isfinite(prediction["criterion"].borders).all()),
                    "invalid latent criterion borders")
            self.actual_heads += 1
            self.stage_heads += 1
            self.members.add(member)
            rule = backend.stage_policy(self.stage_index)
            self.query_calls.append({
                "stage": self.stage_index, "member": member,
                "query_unmasked": array_descriptor(np.asarray(query)),
                "query_masked": array_descriptor(masked),
                "mask": rule, "policy_sha256": backend.policy_metadata()["policy_sha256"],
                "actual_target_coordinate": rule["target_coordinate"],
                "execution_route": "resident_cache", "witness": witness,
            })
            prediction["prior_coordinate_execution_proof"] = {
                "module_sha256": source_sha256(), "backend_source_sha256": backend.source_sha256(),
                "codec_sha256": CODEC_SHA256, "mode": MODE, "stage": self.stage_index,
                "member": member, "actual_head_contexts": 1, "corrected_CDF_calls": 1,
                "witness": witness, "criterion_coordinate": rule["target_coordinate"],
            }
            return prediction
        except BaseException:
            self.failure = True
            raise

    def predict(self, query, *, output_type="full", capture_members=False):
        import torch
        require(self.stage_index == 0 and len(query) == 1 and output_type == "full",
                "only stage0 singleton full aggregate is declared")
        start = self.actual_heads
        # Capture helper arithmetic is not needed here: the original1732 helper
        # merely invokes this adapter's eight predict_component calls.
        from .execution_ss15 import capture_members
        captured = capture_members(self, query)
        all21 = load_script("characterize_fp16_all_stages.py", ancestry.ALL21_SOURCE_SHA)
        result = all21.assemble_shared_stage0(captured, torch, self.engine.base.device)
        require(self.actual_heads - start == 8, "stage0 aggregate needs eight actual head contexts")
        self.shared_densities += 1
        result["prior_coordinate_execution_proof"] = {
            "module_sha256": source_sha256(), "backend_source_sha256": backend.source_sha256(),
            "codec_sha256": CODEC_SHA256, "mode": MODE, "stage": 0, "member": None,
            "actual_head_contexts": 8, "corrected_CDF_calls": 8,
            "criterion_coordinate": backend.stage_policy(0)["target_coordinate"],
            "stage0_policy": all21.STAGE0_POLICY,
        }
        return result

    def inspect_cache(self):
        return self.engine.inspect_cache()

    def close(self):
        if self.session is None:
            return
        if not self.entered:
            self.engine = self.session = None
            return
        session, engine, routes = self.session, self.engine, {}
        audit_error = release_error = None
        released = engine is None
        try:
            if engine is not None:
                try:
                    routes = {k: v for k, v in getattr(engine, "hybrid_counters", {}).items() if k != "stages"}
                except BaseException as error:
                    audit_error = error
                finally:
                    try:
                        engine.close()
                        released = True
                    except BaseException as error:
                        release_error = error
                if audit_error is None and release_error is None and not self.failure:
                    require(routes.get("resident_component_predict_count", 0) == self.stage_heads
                            and routes.get("uncached_predict_count", 0) == 0,
                            "actual cached-only route counts differ")
                if audit_error is not None:
                    if release_error is not None:
                        raise audit_error from release_error
                    raise audit_error
                if release_error is not None:
                    raise release_error
        finally:
            try:
                session.__exit__(None, None, None)
            finally:
                try:
                    report = session.report()
                    self.stages.append({
                        "stage_index": self.stage_index, "actual_head_contexts": self.stage_heads,
                        "queried_members": sorted(self.members), "failed": self.failure,
                        "prefill": self.prefill, "model": self.model,
                        "route_counters": routes, "precision_session": report,
                        "engine_release_attempted": engine is not None,
                        "engine_release_succeeded": released,
                        "accounting_audit_error": repr(audit_error) if audit_error else None,
                        "engine_release_error": repr(release_error) if release_error else None,
                    })
                finally:
                    self.engine = self.session = None
                    self.entered = False
            require(report["aliases_and_flags_restored"]
                    and report["counters"].get("witness_histogram_overflow", 0) == 0,
                    "precision alias/flag restoration or witness capacity failed")

    def metadata(self):
        return {
            "policy": execution_metadata(), "prepare_count": self.prepare_count,
            "actual_head_contexts": self.actual_heads, "shared_densities": self.shared_densities,
            "completed_stage_scopes": self.stages,
            "actual_latent_training_inputs": self.training_inputs,
            "actual_latent_query_calls": self.query_calls,
            "all_stage_aliases_restored": all(
                r["precision_session"]["aliases_and_flags_restored"] for r in self.stages),
        }


@contextmanager
def engine_scope(checkpoint, *, cdf_descriptor):
    with ExitStack() as cleanup:
        engine = PriorCoordinateStageAdapter(checkpoint, cdf_descriptor=cdf_descriptor)
        cleanup.callback(engine.close)
        yield engine


def validate_prediction(prediction, stage, member, *, report):
    proof = prediction.get("prior_coordinate_execution_proof", {})
    count = 8 if member is None else 1
    require((member is None) == (stage == 0)
            and proof.get("module_sha256") == source_sha256()
            and proof.get("backend_source_sha256") == backend.source_sha256()
            and proof.get("codec_sha256") == CODEC_SHA256 and proof.get("mode") == MODE
            and proof.get("stage") == stage and proof.get("member") == member,
            "actual prior-coordinate prediction provenance differs")
    require(proof.get("actual_head_contexts") == proof.get("corrected_CDF_calls") == count,
            "actual latent head/CDF count differs")
    if member is not None:
        require(proof.get("witness", {}).get("passed") is True, "actual FP32 head witness missing")
    report["sampler_callback_calls"] = report.get("sampler_callback_calls", 0) + 1
    report["actual_head_contexts"] = report.get("actual_head_contexts", 0) + count
    report["shared_densities"] = report.get("shared_densities", 0) + int(member is None)


def qualification_receipt(path, *, banks, prefix_run, codec_inventory, codec_reference_inventory,
                          codec_combined_inventory, completed_grid,
                          raw_bindings, model_inputs):
    for value in (QUALIFIER_SHA256, CONDITIONAL_QUALIFICATION_SHA256,
                  INVENTORY_HELPER_SHA256, INVENTORY_SHA256):
        require(isinstance(value, str) and len(value) == 64,
                "fresh codec inventory and conditional screen must be pinned before inference")
    require(sha256_file(path) == CONDITIONAL_QUALIFICATION_SHA256,
            "fresh conditional screen artifact differs")
    helper = load_script(QUALIFIER_FILENAME, QUALIFIER_SHA256)
    require(helper.INVENTORY_HELPER_SHA256 == INVENTORY_HELPER_SHA256
            and helper.INVENTORY_SHA256 == INVENTORY_SHA256,
            "fresh screen codec inventory bindings differ")
    grid = helper.grid_receipt(completed_grid)
    inventory, inventory_replay = helper.inventory_receipt(
        codec_inventory, codec_reference_inventory, codec_combined_inventory)
    theta, obs, query, _, raw, transformed, prefix_custody = helper.input_bindings(
        banks, prefix_run, inventory)
    require(raw == raw_bindings, "actual physical bank/query differs from the fresh screen")
    from .science.analytical_benchmark_v2 import array_descriptor
    require(model_inputs == {
        "context_theta": array_descriptor(theta), "context_obs": array_descriptor(obs),
        "query": array_descriptor(query)}, "actual latent model inputs differ from fresh screen")
    document = json.loads(Path(path).read_bytes())
    receipt = helper.validate_screen(
        document, expected_sources=helper.source_pins(), raw_bindings=raw,
        transformed_bindings=transformed, coordinate_policy=backend.policy_metadata(),
        inventory_replay=inventory_replay)
    require(receipt["passed"] and document["matching_FP32_execution_passed_all21"],
            "fresh matching cached FP32 all21 screen did not pass")
    # FP16 is a separately retained comparison, never this candidate or a veto.
    return document, {
        "screen_sha256": CONDITIONAL_QUALIFICATION_SHA256, "replay": receipt,
        "codec_inventory": inventory_replay, "completed_baseline_grid": grid,
        "fixed_prefix_custody": prefix_custody,
        "raw_scientific_bindings": raw, "transformed_bindings": transformed,
        "scope": "separate same-case finite prefixes; request only replays artifact",
        "FP16_activated": False,
    }


def proposal_scope(frozen, raw_bindings, model_inputs, seeds):
    scope = frozen.proposal_buffer_scope(
        raw_bindings, context_seed=CONTEXT_SEED, case=CASE, repetition=0,
        rank=0, world_size=1, seeds=seeds)
    scope.update(
        prior_coordinate_producer_sha256=source_sha256(),
        codec_sha256=CODEC_SHA256, backend_sha256=BACKEND_SHA256,
        coordinate_policy_sha256=backend.policy_metadata()["policy_sha256"],
        model_inputs=model_inputs, stage_to_physical_order=list(codec.ORDER),
        physical_to_stage_order=list(codec.INVERSE_ORDER),
        strict_full_wave_decode_before_FIFO_and_support=True)
    return scope


def run(*, banks, output, checkpoint, ledger, reference, reference_grid, qualification,
        codec_inventory, codec_reference_inventory, codec_combined_inventory, completed_grid, prefix_run):
    start = time.perf_counter()
    output = Path(output)
    require(not output.exists(), "fresh prior-coordinate pilot output required")
    require_sources()
    from .inference_cdf import guard_baseline_source, reference_binding
    from .inference import CHECKPOINT_SHA256, support_mask, _prior
    from .banks import load_context, load_heldout
    from .science.analytical_benchmark_v2 import array_descriptor, build_problem
    from .reference_grid import _require_budgeted_gpu

    frozen = clean.frozen_sampling()
    initial_source = source_sha256()
    baseline_source = guard_baseline_source()
    require(sha256_file(checkpoint) == CHECKPOINT_SHA256, "immutable checkpoint differs")
    context = load_context(banks, CONTEXT_SEED, 4096)
    raw_query = np.asarray(load_heldout(banks, CASE).obs).reshape(1, 33)
    problem = build_problem()
    raw = {
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "scientific_fingerprint": problem.scientific_fingerprint,
        "parameter_order": list(problem.parameter_order),
        "context_theta": array_descriptor(context.theta),
        "context_obs": array_descriptor(context.obs), "query": array_descriptor(raw_query),
    }
    theta, obs, query, model_inputs, encoding = model_bindings(context, raw_query)
    screen, receipt = qualification_receipt(
        qualification, banks=banks, prefix_run=prefix_run, codec_inventory=codec_inventory,
        codec_reference_inventory=codec_reference_inventory, codec_combined_inventory=codec_combined_inventory,
        completed_grid=completed_grid, raw_bindings=raw, model_inputs=model_inputs)
    refs = reference_binding(banks=banks, reference=reference,
                             reference_grid=reference_grid, case=CASE)
    budget = _require_budgeted_gpu(ledger)
    require(budget["gpus"] == 1, "single GPU prior-coordinate pilot only")
    output.mkdir(parents=True, exist_ok=False)
    engine = descriptor = buffer = None
    latent_waves = []
    status = {
        **runtime_identity(), "schema": SCHEMA, "variant": VARIANT, "status": "running",
        "attribution": "Built with PriorLabs-TabPFN", "runner_source_sha256": initial_source,
        "execution_policy": execution_metadata(), "codec": codec.metadata(),
        "coordinate_policy": backend.policy_metadata(), "model_inputs": model_inputs,
        "model_target_descriptors": encoding, "raw_scientific_bindings": raw,
        "physical_parameter_order": list(problem.parameter_order),
        "learned_target_coordinate_system": "causal_prior_latent_sigma_first_stage_order",
        "learned_targets_are_original_physical_values": False,
        "samples_in_original_physical_order_before_FIFO_support": True,
        "all_generated_rows_strictly_decoded_including_unused_FIFO_tail": True,
        "decode_failure_policy": "FAIL request; no rejection substitute, cast, clipping, redraw or repair",
        "model_coordinate_decode_after_global_refresh": False,
        "stage_to_physical_order": list(codec.ORDER),
        "physical_to_stage_order": list(codec.INVERSE_ORDER),
        "matching_conditional_qualification_sha256": CONDITIONAL_QUALIFICATION_SHA256,
        "matched_execution_screen": receipt,
        "actual_fixed_conditional_screen_performed_during_request": False,
        "fresh_same_policy_all21_screen_replayed": True,
        "matching_cached_FP32_execution_passed_all21": True,
        "FP16_comparison_hard_validity": screen.get("FP16_hard_validity_passed_all21"),
        "FP16_comparison_strict_failed_stages": screen.get("FP16_strict_failed_stages"),
        "FP16_activated": False, "mode": MODE, "fallback_map": {},
        "requested_count": COUNT, "benchmark_total_count": COUNT, "context_seed": CONTEXT_SEED,
        "context_size": 4096, "case": CASE, "repetition": 0, "rank": 0, "world_size": 1,
        "end_to_end": True, "scientific_run": True,
        "observation_preprocessing": "fixed SS15 Float32 with source-defined own-block zero mask",
        "context_filter": "no_filtering", "sampler_math_changed": False,
        "partial_vector_filtering": False, "tail_family_changed": False,
        "clipping_or_repair_performed": False, "physical_log_density_claimed": False,
        "original_gate_unchanged": True, "posterior_accuracy_claimed": False,
        "proposal_buffer_policy": frozen.proposal_buffer_policy(1200), "proposal_lookahead": 1200,
        "query_batch_size": 1024, "engine_query_batch_cap": 1024,
        "max_iter_rejection": None, "microbatch": COUNT,
        "budget_binding": budget, "reference_binding": refs,
        "baseline_execution_source": baseline_source, "preregistration_sha256": PREREGISTRATION_SHA256,
        "completed_stage_indices": [], "stage_events": [], "proposal_wave_sizes": [],
        "decode_wave_records": [], **raw,
    }
    write_json(output / "manifest.json", status)
    try:
        import torch
        from .compat_cdf import finite_cdf_compatibility
        from .upstream import load_baseline
        require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "one visible GPU required")
        require(torch.get_default_dtype() == torch.float32
                and torch.get_float32_matmul_precision() == "highest"
                and not torch.backends.cuda.matmul.allow_tf32
                and not torch.is_autocast_enabled("cuda"), "strict outer FP32 flags required")
        load_baseline()
        accept_reject = import_module("npe_pfn.accept_reject_sampler").accept_reject_sample
        seeds = frozen.worker_stream_seeds(CONTEXT_SEED, CASE, 0, 0, 1)
        torch.manual_seed(seeds["component_uniform"])
        torch.cuda.manual_seed_all(seeds["component_uniform"])
        generators = {name: torch.Generator(device="cpu").manual_seed(seed)
                      for name, seed in seeds.items()}
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "16")))
        status.update(
            seed=seeds["component_uniform"], seed_streams=seeds,
            gpu=torch.cuda.get_device_name(), torch_cuda=torch.version.cuda,
            sampling_policy={
                "forward_precision": MODE,
                "streams": {name: {"role": frozen.STREAM_ROLES[name], "seed": seed,
                           "dtype": frozen.STREAM_DTYPES[name], "device": "cpu",
                           "generator": "torch.Generator"} for name, seed in seeds.items()},
                "head_policy": "raw_mass_rejection_v1", "mass_envelope": 1.00001,
                "uniform_assignment": "preassigned_per_stage_by_physical_row",
                "head_retry_preserves_row_uniform": True,
                "stage0_density_scope": "one latent proposal wave; eight actual rawP heads; no cross-query reuse"})
        prior = _prior(torch)
        counters = {"stage0_shared_predictions": 0, "component_predictions": 0,
                    "head_candidates": 0, "head_rejections": 0}
        actual = {}
        with finite_cdf_compatibility() as descriptor, engine_scope(
                checkpoint, cdf_descriptor=descriptor) as engine:
            callback = lambda prediction, stage, member: validate_prediction(
                prediction, stage, member, report=actual)
            stage_draw = frozen.component_stage_sampler(
                engine, torch, generators, counters, prediction_check=callback)

            def progress(event):
                status["stage_events"].append({"proposal_round": len(status["proposal_wave_sizes"]), **event})
                write_json(output / "manifest.json", status)

            def proposal(size):
                status["proposal_wave_sizes"].append(int(size))
                latent = wave.stage_major_wave(
                    engine, theta, obs, query, int(size), 1024, lambda _: None,
                    stage_draw=stage_draw, progress=progress)
                physical = record_decode_wave(latent, latent_waves, status)
                return torch.from_numpy(physical), None

            scope = proposal_scope(frozen, raw, model_inputs, seeds)
            fifo = frozen.guarded_proposal_buffer()
            buffer = fifo.ProposalBuffer(
                proposal, scope=scope, concatenate=lambda parts: torch.cat(parts, dim=0),
                world_size=1, rank=0)
            before_rng = rng_hashes(torch)
            status["default_rng_before"] = before_rng
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            request_start = time.perf_counter()
            with torch.no_grad():
                samples, _, _ = accept_reject(
                    proposal=buffer, accept_reject_fn=prior.support.check, num_samples=COUNT,
                    max_sampling_batch_size=COUNT, max_iter_rejection=None,
                    show_progress_bars=False, proposal_sampling_kwargs={})
            record = buffer.finalize(
                original_checked_count=prior.support.proposed,
                accepted_before_trim=prior.support.accepted, returned_count=len(samples),
                original_support_calls=prior.support.calls)
            record.update(enabled=True, source_unchanged=True,
                          policy_sha256=status["proposal_buffer_policy"]["policy_sha256"])
            frozen.validate_proposal_buffer_record(record, world_size=1, rank=0, scope=scope)
            status["proposal_buffer"] = record
            torch.cuda.synchronize()
            finished = time.perf_counter()
            values = samples.detach().cpu().numpy()
            require(values.dtype == np.float64 and values.shape == (COUNT, 21)
                    and support_mask(values).all(), "exactly50000 complete supported physical vectors required")
            waves = len(status["proposal_wave_sizes"])
            heads = 8 * counters["stage0_shared_predictions"] + counters["component_predictions"]
            require(engine.prepare_count == len(engine.stages) == 21 * waves, "all21 stage scopes differ")
            require(engine.actual_heads == actual["actual_head_contexts"]
                    == heads == descriptor["patched_call_count"], "actual head/CDF counters differ")
            require(engine.shared_densities == counters["stage0_shared_predictions"] == waves,
                    "stage0 singleton density count differs")
            require(record["actual_generated_wave_sizes"] == status["proposal_wave_sizes"]
                    == [len(v) for v in latent_waves], "actual generated latent/FIFO waves differ")
            require(len(engine.training_inputs) == 21 * waves and len(engine.query_calls) == heads,
                    "actual latent train/query journal coverage differs")
            status.update(
                execution_counters=engine.metadata(), sampling_counters=dict(counters),
                actual_prediction_checks=actual, expected_patched_call_count=heads,
                expected_stage_prepare_count=21 * waves, generated_proposal_wave_count=waves)
        after_rng = rng_hashes(torch)
        status.update(default_rng_after=after_rng, default_rng_preserved=before_rng == after_rng)
        require(before_rng == after_rng, "model/sampler/decode changed default RNG streams")
        require(descriptor["original_alias_restored"] and descriptor["sdk_sources_unchanged"],
                "CDF alias/source restoration failed")
        require_sources()
        require(source_sha256() == initial_source and guard_baseline_source() == baseline_source,
                "execution/scientific sources changed")
        require(sha256_file(qualification) == CONDITIONAL_QUALIFICATION_SHA256,
                "fresh screen artifact changed")
        archive_start = time.perf_counter()
        wave_archive = archive_latent_waves(output, latent_waves)
        with (output / "samples.npz").open("xb") as stream:
            np.savez(stream, samples=values)
        status.update(
            status="complete", samples=array_descriptor(values), samples_file="samples.npz",
            samples_sha256=sha256_file(output / "samples.npz"),
            latent_wave_archive=wave_archive,
            compatibility_adapter=dict(descriptor), sources_unchanged=True,
            completed_stage_indices=list(range(21)),
            rejection={"support_calls": prior.support.calls, "proposed": prior.support.proposed,
                       "accepted_before_final_trim": prior.support.accepted, "returned": len(values)},
            timings={"posterior_request_seconds": finished - request_start,
                     "total_seconds": time.perf_counter() - start,
                     "accepted_samples_per_second": COUNT / (finished - request_start),
                     "includes_stage_preparation": True, "includes_CPU_sampling": True,
                     "includes_strict_all_generated_wave_decode": True,
                     "latent_and_physical_archive_seconds": time.perf_counter() - archive_start},
            memory={"peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved()})
    except BaseException as error:
        status.update(status="failed", error_type=type(error).__name__, error=str(error))
        if descriptor is not None:
            status["compatibility_adapter"] = dict(descriptor)
        if buffer is not None:
            status["proposal_buffer"] = buffer.metadata()
        if latent_waves and not (output / "latent-proposal-waves.npz").exists():
            try:
                status["latent_wave_archive"] = archive_latent_waves(output, latent_waves)
                status["latent_wave_archive"]["preserved_after_request_failure"] = True
            except BaseException as archive_error:
                status["latent_wave_archive_preservation_error"] = {
                    "type": type(archive_error).__name__, "error": str(archive_error)}
        # An archive error is secondary evidence; retain the original failure.
        raise
    finally:
        if engine is not None:
            engine.close()
        status["wrapper_total_seconds"] = time.perf_counter() - start
        write_json(output / "manifest.json", status)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("banks", "output", "checkpoint", "ledger", "reference", "reference-grid",
                 "qualification", "codec-inventory", "codec-reference-inventory",
                 "codec-combined-inventory", "completed-grid", "prefix-run"):
        parser.add_argument("--" + name, required=True)
    record = run(**vars(parser.parse_args()))
    print(json.dumps({key: record[key] for key in ("schema", "variant", "status", "case", "context_seed", "samples_sha256")}, sort_keys=True))


if __name__ == "__main__":
    main()
