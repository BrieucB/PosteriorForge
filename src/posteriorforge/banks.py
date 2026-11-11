"""Immutable paired prior-predictive banks and deterministic role streams."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import time
from typing import Protocol

import numpy as np

from .science.analytical_benchmark_v2 import (
    array_descriptor, build_problem, central_world_mask, draw_worlds,
    flatten_observations, unflatten_observations,
)

SCHEMA = "posteriorforge.banks.v1"
CONTEXT_SEEDS = (20261111, 20261112, 20261113)
HELDOUT_SEED = 20261114
CONTEXT_SIZES = (256, 1024, 4096)


def derive_seed(seed: int, role: str, *, case: int = 0, rank: int = 0,
                repetition: int = 0) -> int:
    """Derive an explicit independent uint32 seed with NumPy SeedSequence."""
    values = (seed, case, rank, repetition)
    if any(type(x) is not int or x < 0 for x in values):
        raise ValueError("seed, case, rank and repetition must be nonnegative integers")
    if not isinstance(role, str) or not role.strip():
        raise ValueError("nonempty role required")
    tag = np.frombuffer(sha256(("posteriorforge:" + role).encode()).digest(), dtype="<u4")
    return int(np.random.SeedSequence([seed, *map(int, tag), case, rank, repetition])
               .generate_state(1, dtype=np.uint32)[0])


@dataclass(frozen=True)
class PairedBank:
    """Complete ordered physical parameter/observation rows, never shuffled separately."""
    theta: np.ndarray
    obs: np.ndarray
    manifest: dict

    def __post_init__(self):
        theta = np.ascontiguousarray(self.theta, dtype=np.float64)
        obs = np.ascontiguousarray(self.obs, dtype=np.float64)
        if theta.ndim != 2 or theta.shape[1] != 21 or obs.shape != (len(theta), 33) or len(theta) < 1:
            raise ValueError("paired theta[n,21] and obs[n,33] required")
        if not np.isfinite(theta).all() or not np.isfinite(obs).all():
            raise ValueError("paired bank must be finite")
        if not build_problem().contains(theta).all():
            raise ValueError("paired bank outside joint prior support")
        # Read-only copies prevent callers mutating a bank after authentication.
        theta = np.frombuffer(theta.tobytes(), dtype=np.float64).reshape(theta.shape)
        obs = np.frombuffer(obs.tobytes(), dtype=np.float64).reshape(obs.shape)
        object.__setattr__(self, "theta", theta)
        object.__setattr__(self, "obs", obs)
        object.__setattr__(self, "manifest", json.loads(json.dumps(self.manifest, allow_nan=False)))

    def prefix(self, count: int) -> "PairedBank":
        if type(count) is not int or not 1 <= count <= len(self.theta):
            raise ValueError("prefix must be a positive count no larger than the bank")
        return PairedBank(self.theta[:count], self.obs[:count],
                          dict(self.manifest, prefix_count=count,
                               theta_prefix=array_descriptor(self.theta[:count]),
                               obs_prefix=array_descriptor(self.obs[:count])))


class PairedBankProvider(Protocol):
    """Future physical surrogate/C/CUDA adapters must return paired full-prior rows."""
    def get_bank(self, *, seed: int, count: int) -> PairedBank: ...


class AnalyticalBankProvider:
    def get_bank(self, *, seed: int, count: int) -> PairedBank:
        theta, observed = draw_worlds(count, role="posteriorforge.context", seed=seed)
        problem = build_problem()
        return PairedBank(theta, flatten_observations(observed), {
            "schema": SCHEMA, "provider": "analytical21x33", "seed": seed,
            "role": "context", "central": False,
            "scientific_fingerprint": problem.scientific_fingerprint,
            "parameter_order": list(problem.parameter_order),
        })


def _write_bank(path: Path, bank: PairedBank) -> dict:
    with path.open("xb") as handle:
        np.savez(handle, theta=bank.theta, obs=bank.obs)
    result = dict(bank.manifest, file=path.name, worlds=len(bank.theta),
                  sha256=sha256(path.read_bytes()).hexdigest(),
                  theta=array_descriptor(bank.theta), obs=array_descriptor(bank.obs))
    with path.with_suffix(".json").open("x") as handle:
        json.dump(result, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    return result


def prepare_banks(output_root, *, context_seeds=CONTEXT_SEEDS,
                  heldout_seed=HELDOUT_SEED) -> dict:
    """Create exactly three4096-world banks and eight shared central90% cases.

    Existing roots are never overwritten. Small nested contexts are prefixes of
    each complete bank. No normalization or validation banks are created.
    """
    seeds = tuple(context_seeds)
    if len(seeds) != 3 or len(set(seeds)) != 3 or any(type(x) is not int or x < 0 for x in seeds):
        raise ValueError("three distinct nonnegative context seeds required")
    if type(heldout_seed) is not int or heldout_seed < 0 or heldout_seed in seeds:
        raise ValueError("held-out seed must be nonnegative and distinct")
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    problem = build_problem()
    provider = AnalyticalBankProvider()
    contexts = []
    for seed in seeds:
        bank = provider.get_bank(seed=seed, count=4096)
        entry = _write_bank(root / f"context-{seed}.npz", bank)
        entry["nested_prefixes"] = {str(n): {
            "theta": array_descriptor(bank.theta[:n]), "obs": array_descriptor(bank.obs[:n])}
            for n in CONTEXT_SIZES}
        contexts.append(entry)
    theta, observations = draw_worlds(8, role="posteriorforge.heldout", seed=heldout_seed,
                                      central=True)
    heldout = _write_bank(root / "heldout.npz", PairedBank(theta, flatten_observations(observations), {
        "schema": SCHEMA, "provider": "analytical21x33", "role": "heldout",
        "seed": heldout_seed, "central": True,
        "central_rule": "whole-world rejection; each global coordinate, conditional local residual and truncated-noise quantile lies in coordinate-wise central90%",
        "scientific_fingerprint": problem.scientific_fingerprint,
        "parameter_order": list(problem.parameter_order),
    }))
    result = {"schema": SCHEMA, "artifact_identity": "PosteriorForge",
              "scientific_fingerprint": problem.scientific_fingerprint,
              "specification": problem.scientific_manifest(),
              "contexts": contexts, "heldout": heldout,
              "shared_cases": 8, "context_sizes": list(CONTEXT_SIZES),
              "stream_definition": {"algorithm": "NumPyPCG64 with indexed SeedSequence streams",
                  "context_role": "posteriorforge.context", "heldout_role": "posteriorforge.heldout",
                  "bank_seed_inputs": "explicit seeds, unchanged; source draw_worlds hashes role plus retained science schema",
                  "full_prior_block_size": 512, "heldout_selection": "entire joint draw rejection before observation simulation"},
              "elapsed_seconds": time.perf_counter() - started,
              "compute": {"gpu_hours": 0, "operation": "analytical prior-predictive data generation"}}
    with (root / "manifest.json").open("x") as handle:
        json.dump(result, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    return result


def load_bank(path, *, expected_sha256=None) -> PairedBank:
    """Authenticate NPZ bytes, array bytes, shape, support, and science identity."""
    path = Path(path)
    metadata = json.loads(path.with_suffix(".json").read_text())
    actual = sha256(path.read_bytes()).hexdigest()
    if actual != metadata["sha256"] or (expected_sha256 is not None and actual != expected_sha256):
        raise ValueError("bank payload SHA256 mismatch")
    problem = build_problem()
    if metadata["scientific_fingerprint"] != problem.scientific_fingerprint:
        raise ValueError("scientific fingerprint mismatch")
    if metadata["parameter_order"] != list(problem.parameter_order):
        raise ValueError("parameter order mismatch")
    with np.load(path, allow_pickle=False) as arrays:
        if set(arrays.files) != {"theta", "obs"}:
            raise ValueError("bank must contain exactly theta and obs")
        for name in arrays.files:
            if arrays[name].dtype != np.float64 or array_descriptor(arrays[name]) != metadata[name]:
                raise ValueError("bank array dtype/hash mismatch")
        return PairedBank(arrays["theta"], arrays["obs"], metadata)


def load_context(root, seed: int, size: int = 4096) -> PairedBank:
    if size not in CONTEXT_SIZES:
        raise ValueError("approved context sizes are256,1024,4096")
    return load_bank(Path(root) / f"context-{seed}.npz").prefix(size)


def load_heldout(root, case_index: int) -> PairedBank:
    bank = load_bank(Path(root) / "heldout.npz")
    if type(case_index) is not int or not 0 <= case_index < len(bank.theta):
        raise ValueError("case index outside held-out bank")
    return PairedBank(bank.theta[case_index:case_index + 1], bank.obs[case_index:case_index + 1],
                      dict(bank.manifest, case_index=case_index))
