"""Unmodified upstream baseline execution. Importing this module is CPU-safe."""
from __future__ import annotations
from pathlib import Path
import json
import os
import time
import numpy as np
from .artifacts import runtime_identity, sha256_file, write_json
from .banks import derive_seed, load_context, load_heldout
from .science.analytical_benchmark_v2 import array_descriptor, build_problem


def validate_query(observation):
    """Only the observed 33-vector is accepted; held-out labels are not inputs."""
    value = np.asarray(observation, dtype=np.float64)
    if value.shape not in ((33,), (1, 33)) or not np.isfinite(value).all():
        raise ValueError("one finite raw observation[33] required")
    return value.reshape(1, 33).copy()


def support_mask(theta):
    value = np.asarray(theta)
    if value.ndim != 2 or value.shape[1] != 21:
        raise ValueError("complete theta[n,21] required")
    finite = np.isfinite(value).all(axis=1)
    result = np.zeros(len(value), dtype=bool)
    if finite.any():
        result[finite] = build_problem().contains(value[finite])
    return result


def frozen_regressor_settings(checkpoint, device="cuda"):
    return dict(n_estimators=8, categorical_features_indices=None,
                softmax_temperature=0.9, average_before_softmax=False,
                model_path=str(Path(checkpoint).resolve()), device=device,
                ignore_pretraining_limits=False, fit_mode="fit_preprocessors",
                memory_saving_mode="auto", random_state=0, n_jobs=1,
                inference_config=None, inference_precision="torch.float32")


def _prior(torch):
    problem = build_problem()
    indices = [problem.parameter_order.index(name) for name in problem.prior.noise_names]
    low, high = problem.prior.bounds
    class Support(torch.distributions.constraints.Constraint):
        event_dim = 1
        def __init__(self):
            self.proposed, self.accepted, self.calls = 0, 0, 0
        def check(self, value):
            mask = (torch.isfinite(value).all(dim=-1)
                    & ((value[..., indices] >= low) & (value[..., indices] <= high)).all(dim=-1))
            self.calls += 1
            self.proposed += len(value)
            self.accepted += int(mask.sum().item())
            return mask
    class Prior(torch.distributions.Distribution):
        arg_constraints = {}
        support = Support()
        def __init__(self):
            super().__init__(event_shape=torch.Size([21]), validate_args=False)
        def log_prob(self, value):
            result = problem.log_prior(value.detach().cpu().numpy())
            return torch.as_tensor(result, dtype=value.dtype, device=value.device)
    return Prior()


def run_baseline(*, banks, output, context_seed, case, context_size=4096,
                 count=50000, microbatch=10000, checkpoint, checkpoint_sha256,
                 repetition=0, device="cuda"):
    """Run complete accepted rows; external limits fail the run, never fill rows."""
    if context_size not in (256, 1024, 4096) or type(count) is not int or count < 1:
        raise ValueError("declared context size and positive count required")
    if microbatch not in (1024, 4096, 10000):
        raise ValueError("declared microbatch required")
    if device != "cuda" or not os.getenv("SLURM_JOB_ID"):
        raise RuntimeError("real inference must run in a budgeted Slurm CUDA allocation")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    status = {**runtime_identity(), "schema": "posteriorforge.inference.v1",
              "artifact_identity": "PosteriorForge", "status": "running",
              "variant": "upstream_unmodified", "context_seed": context_seed,
              "case": case, "context_size": context_size, "requested_count": count,
              "scientific_run": count == 50000, "microbatch": microbatch,
              "repetition": repetition, "checkpoint_sha256": checkpoint_sha256,
              "observation_preprocessing": "raw; TabPFN frozen preprocessing",
              "context_filter": "no_filtering", "log_density": "not_used",
              "max_iter_rejection": None, "rank": 0, "world_size": 1}
    write_json(output / "manifest.json", status)
    try:
        if sha256_file(checkpoint) != checkpoint_sha256:
            raise ValueError("checkpoint SHA256 mismatch")
        context = load_context(banks, context_seed, context_size)
        heldout = load_heldout(banks, case)
        query = validate_query(heldout.obs)
        status.update(scientific_fingerprint=build_problem().scientific_fingerprint,
                      parameter_order=list(build_problem().parameter_order),
                      context_theta=array_descriptor(context.theta), context_obs=array_descriptor(context.obs),
                      query=array_descriptor(query))
        import torch
        from .upstream import load_baseline
        baseline_class = load_baseline()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        seed = derive_seed(context_seed, "posterior", case=case, repetition=repetition)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS", "8")))
        torch.cuda.reset_peak_memory_stats()
        status.update(seed=seed, gpu=torch.cuda.get_device_name(), torch_cuda=torch.version.cuda,
                      regressor_settings=frozen_regressor_settings(checkpoint, device))
        kwargs = frozen_regressor_settings(checkpoint, device)
        kwargs["inference_precision"] = torch.float32
        prior = _prior(torch)
        posterior = baseline_class(
            prior=prior, filter_type="no_filtering", show_progress_bars=False,
            regressor_init_kwargs=kwargs)
        torch.cuda.synchronize()
        initialized = time.perf_counter()
        posterior.append_simulations(torch.from_numpy(context.theta.copy()).float(),
                                     torch.from_numpy(context.obs.copy()).float())
        torch.cuda.synchronize()
        prepared = time.perf_counter()
        with torch.no_grad():
            samples = posterior.sample((count,), x=torch.from_numpy(query).float(),
                                       max_sampling_batch_size=microbatch,
                                       max_iter_rejection=None, with_log_prob=False)
        torch.cuda.synchronize()
        finished = time.perf_counter()
        values = samples.detach().cpu().numpy().astype(np.float64)
        if values.shape != (count, 21) or not support_mask(values).all():
            raise RuntimeError("upstream returned incomplete, nonfinite or unsupported rows")
        with (output / "samples.npz").open("xb") as stream:
            np.savez(stream, samples=values)
        model = posterior._model.model_
        status.update(model={"parameter_count": sum(p.numel() for p in model.parameters()),
                             "dtype": str(next(model.parameters()).dtype),
                             "device": str(next(model.parameters()).device),
                             "autocast": bool(posterior._model.use_autocast_),
                             "forced_inference_dtype": str(posterior._model.forced_inference_dtype_)},
                      status="complete", samples=array_descriptor(values),
                      samples_file="samples.npz", samples_sha256=sha256_file(output / "samples.npz"),
                      timings={"constructor_and_import_seconds": initialized-started,
                               "cold_checkpoint_load_included_in_request": True,
                               "append_context_seconds": prepared-initialized,
                               "posterior_request_seconds": finished-prepared,
                               "total_seconds": finished-started,
                               "accepted_samples_per_second": count/(finished-prepared),
                               "includes_repeated_stage_context_preparation": True},
                      rejection={"support_calls": prior.support.calls,
                                 "proposed": prior.support.proposed,
                                 "accepted_before_final_trim": prior.support.accepted,
                                 "returned": count},
                      memory={"peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                              "peak_reserved_bytes": torch.cuda.max_memory_reserved()},
                      stage_timings="unmeasured; profiling gated on scientific feasibility",
                      transfer_and_sync="unmeasured; profiling gated on scientific feasibility")
    except BaseException as error:
        status.update(status="failed", error_type=type(error).__name__, error=str(error),
                      elapsed_seconds=time.perf_counter()-started)
        write_json(output / "manifest.json", status)
        raise
    write_json(output / "manifest.json", status)
    return status
