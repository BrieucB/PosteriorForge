"""Float64 references for the frozen quadratic/scalar Gaussian hierarchy.

Six-dimensional scrambled Sobol integration integrates the Gaussian physics
analytically. Independent observation-space algebra and a three-global by six
one-dimensional-noise quadrature provide separate checks. No learned model is
imported. Node artifacts are reusable; posterior cohorts are drawn in bounded
batches, preserving coherent complete-vector rows and all duplicate node draws.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import itertools
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.special import logsumexp, ndtr
from scipy.stats import qmc

from .analytical_benchmark_v2 import array_descriptor, flatten_observations
from .analytical_independent_reference import references_from_problem
from .hierarchical_noise import GaussianIntegratedNoiseReference

SCHEMA = "bayesarena.analytical_benchmark_reference_v2.v1"
POWERS = (16, 18, 20, 22)
SCRAMBLES = 4
ERROR_FRACTION = .1
_JAX_KERNELS = None


def _jax_kernels(jax, jnp):
    """One generic executable cache: observed-world constants are dynamic inputs."""
    global _JAX_KERNELS
    if _JAX_KERNELS is not None:
        return _JAX_KERNELS

    def calculate(sigma, precision0, natural0, matrices, rhs, squared, counts, prior_quad, prior_logdet):
        weight = jnp.reciprocal(jnp.square(sigma))
        precision = precision0 + jnp.einsum("ni,ijk->njk", weight, matrices)
        natural = natural0 + weight @ rhs
        chol = jnp.linalg.cholesky(precision)
        identity = jnp.broadcast_to(jnp.eye(15, dtype=jnp.float64), precision.shape)
        inverse = jnp.linalg.solve(chol, identity)
        covariance = jnp.swapaxes(inverse, -1, -2) @ inverse
        mean = jnp.einsum("nij,nj->ni", covariance, natural)
        quad = prior_quad + weight @ squared - jnp.sum(natural * mean, axis=1)
        logdet = 2 * jnp.log(jnp.diagonal(chol, axis1=1, axis2=2)).sum(axis=1)
        evidence = (-.5 * (counts.sum() * math.log(2 * math.pi) + prior_logdet + logdet + quad)
                    - jnp.log(sigma) @ counts)
        return mean, covariance, evidence

    def projection(sigma, weights, directions, queries, ppos, npos, *constants):
        from jax.scipy.special import ndtr as jax_ndtr
        mean, covariance, _ = calculate(sigma, *constants)
        dp, dn = directions[:, ppos], directions[:, npos]
        location = mean @ dp.T + sigma @ dn.T
        variance = jnp.einsum("ki,nij,kj->nk", dp, covariance, dp)
        sd = jnp.sqrt(jnp.maximum(variance, 0))
        nonzero = sd > 0
        z = (queries[None, None, :] - location[:, :, None]) / jnp.where(nonzero, sd, 1)[:, :, None]
        cdf = jnp.where(nonzero[:, :, None], jax_ndtr(z), location[:, :, None] <= queries[None, None, :])
        return jnp.einsum("n,nkq->kq", weights, cdf), jnp.min(variance)

    def sample(sigma, standard_normal, *constants):
        mean, covariance, _ = calculate(sigma, *constants)
        return mean + jnp.einsum("nij,nj->ni", jnp.linalg.cholesky(covariance), standard_normal)

    _JAX_KERNELS = jax.jit(calculate), jax.jit(projection), jax.jit(sample)
    return _JAX_KERNELS


def _json_hash(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _check_sigma(sigma):
    value = np.asarray(sigma, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 6 or len(value) < 1 or not np.isfinite(value).all() or np.any(value <= 0):
        raise ValueError("finite positive sigma[n,6] required")
    return value


class FullHierarchyReference(GaussianIntegratedNoiseReference):
    """Native float64 conditional Gaussian calculations on NumPy or JAX."""

    def __init__(self, problem, observations, *, backend="numpy", device="cpu"):
        super().__init__(problem, observations)
        self.observations = observations
        self.observation_sha256 = array_descriptor(flatten_observations(observations))["sha256"]
        self.binding = {"scientific_fingerprint": problem.scientific_fingerprint,
                        "observation_sha256": self.observation_sha256,
                        "parameter_order": list(problem.parameter_order)}
        self.backend, self.device = backend, device
        if backend not in ("numpy", "jax") or device not in ("cpu", "gpu"):
            raise ValueError("backend numpy/jax and device cpu/gpu required")
        if backend == "numpy" and device != "cpu":
            raise ValueError("NumPy reference is CPU only")
        self._jax_conditional = None
        if backend == "jax":
            # Lazy import: importing this module never initializes accelerator or MPI.
            import jax
            jax.config.update("jax_enable_x64", True)
            import jax.numpy as jnp
            devices = jax.devices(device)
            if not devices:
                raise RuntimeError("requested native JAX device unavailable")
            self._jax_device = devices[0]
            self._jax_constants = [jax.device_put(jnp.asarray(value, dtype=jnp.float64), self._jax_device)
                         for value in (self.precision, self.natural, self.matrices, self.rhs,
                                       self.squared, self.counts, self.prior_quadratic, self.prior_logdet)]
            self._jax_conditional, self._jax_projection, self._jax_sample = _jax_kernels(jax, jnp)
            self._jax_positions = (jax.device_put(jnp.asarray([problem.parameter_order.index(name) for name in self.names]), self._jax_device),
                                   jax.device_put(jnp.asarray([problem.parameter_order.index(name) for name in problem.prior.noise_names]), self._jax_device))
            probe = self._jax_conditional(jax.device_put(jnp.ones((1, 6), dtype=jnp.float64) * .255,
                                                        self._jax_device), *self._jax_constants)
            if any(value.dtype != jnp.float64 for value in probe):
                raise RuntimeError("native JAX float64 reference unavailable")

    def conditional(self, sigma):
        sigma = _check_sigma(sigma)
        if self.backend == "numpy":
            return super().conditional(sigma)
        import jax
        result = self._jax_conditional(jax.device_put(sigma, self._jax_device), *self._jax_constants)
        return tuple(np.asarray(value, dtype=np.float64) for value in result)

    def covariance_space_conditional(self, sigma):
        """Independent CPU formula using the 33D observation covariance."""
        sigma = _check_sigma(sigma)
        a, c = self.design, self.prior_covariance
        cross = c @ a.T
        centered = self.centered - a @ self.prior_mean
        means, covariances, evidences = [], [], []
        for row in sigma:
            covariance_y = a @ c @ a.T + np.diag(np.repeat(row * row, self.counts))
            chol = np.linalg.cholesky(covariance_y)
            solve = np.linalg.solve(chol.T, np.linalg.solve(chol, centered))
            solved_cross = np.linalg.solve(chol.T, np.linalg.solve(chol, cross.T))
            means.append(self.prior_mean + cross @ solve)
            covariances.append(c - cross @ solved_cross)
            evidences.append(-.5 * (len(centered) * math.log(2 * math.pi)
                                   + 2 * np.log(np.diag(chol)).sum() + centered @ solve))
        return np.asarray(means), np.asarray(covariances), np.asarray(evidences)

    def conditional_draw(self, sigma, standard_normal):
        """Transform caller-owned iid normals; native JAX transfers only physics draws."""
        sigma = _check_sigma(sigma)
        standard_normal = np.asarray(standard_normal, dtype=np.float64)
        if standard_normal.shape != (len(sigma), 15) or not np.isfinite(standard_normal).all():
            raise ValueError("finite independent standard normals[n,15] required")
        if self.backend == "jax":
            import jax
            physics = self._jax_sample(jax.device_put(sigma, self._jax_device),
                                       jax.device_put(standard_normal, self._jax_device), *self._jax_constants)
            return np.asarray(physics, dtype=np.float64)
        mean, covariance, _ = self.conditional(sigma)
        return mean + np.einsum("nij,nj->ni", np.linalg.cholesky(covariance), standard_normal)

    def verify_cpu(self, sigma=None, *, atol=2e-8, rtol=2e-8):
        if sigma is None:
            sigma = np.asarray([[.02] * 6, [.1] * 6, [.255] * 6, [.49] * 6,
                                [.03, .08, .17, .23, .36, .48]])
        actual = self.conditional(sigma)
        independent = self.covariance_space_conditional(sigma)
        errors = []
        for value, expected in zip(actual, independent):
            if not np.allclose(value, expected, atol=atol, rtol=rtol):
                raise ValueError("independent observation-space Gaussian check failed")
            errors.append(float(np.max(np.abs(value - expected))))
        return {"passed": True, "mean_covariance_log_evidence_max_errors": errors,
                "atol": atol, "rtol": rtol, "sigma": np.asarray(sigma).tolist()}

    def integrate(self, *, power=16, seed=20261016, batch_size=512, deadline=None):
        """Return an authenticated node bank; never draw a posterior implicitly."""
        if type(power) is not int or not 4 <= power <= 22 or type(batch_size) is not int or batch_size < 1:
            raise ValueError("power in[4,22] and positive batch_size required")
        uniform = qmc.Sobol(6, scramble=True, seed=seed).random_base2(power)
        sigma = np.empty_like(uniform)
        log_proposal = np.zeros(len(uniform))
        for index, cdf in enumerate(self.proposal_cdfs):
            cells = np.clip(np.searchsorted(cdf, uniform[:, index], side="right") - 1, 0, len(cdf) - 2)
            mass = cdf[cells + 1] - cdf[cells]
            width = self.proposal_grid[cells + 1] - self.proposal_grid[cells]
            sigma[:, index] = self.proposal_grid[cells] + (uniform[:, index] - cdf[cells]) / mass * width
            log_proposal += np.log(mass / width)
        del uniform
        log_prior = sum(self.problem.prior.noise_log_prob(name, sigma[:, i])
                        for i, name in enumerate(self.problem.dataset_ids))
        log_weights = np.empty(len(sigma))
        anchor, mass = -np.inf, 0.
        first, second = np.zeros(15), np.zeros((15, 15))
        for start in range(0, len(sigma), batch_size):
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("bounded Sobol reference deadline exhausted")
            stop = min(start + batch_size, len(sigma))
            mean, covariance, log_y = self.conditional(sigma[start:stop])
            logs = log_y + log_prior[start:stop] - log_proposal[start:stop]
            if not np.isfinite(logs).all():
                raise ValueError("nonfinite quadrature log weight")
            log_weights[start:stop] = logs
            new_anchor = max(anchor, float(logs.max()))
            factor = math.exp(anchor - new_anchor)
            weights = np.exp(logs - new_anchor)
            mass = mass * factor + weights.sum()
            first = first * factor + weights @ mean
            second = second * factor + np.einsum("n,nij->ij", weights,
                        covariance + mean[:, :, None] * mean[:, None, :])
            anchor = new_anchor
        total = float(logsumexp(log_weights))
        weights = np.exp(log_weights - total)
        mean = first / mass
        metadata = {"schema": SCHEMA, "binding": self.binding, "power": power, "seed": int(seed),
                    "backend": self.backend, "device": self.device,
                    "log_evidence": total - math.log(len(sigma)),
                    "effective_nodes": float(1 / np.square(weights).sum()),
                    "physics_mean": mean.tolist(),
                    "physics_covariance": (second / mass - np.outer(mean, mean)).tolist(),
                    "noise_mean": (weights @ sigma).tolist(),
                    "noise_second_moment": (weights @ (sigma * sigma)).tolist(),
                    "proposal": "residual_scatter_piecewise_constant_80percent_plus_uniform20percent_exact_density_correction",
                    "proposal_residual_dof": self.residual_dof}
        return ReferenceNodes(sigma, weights, metadata)


@dataclass
class ReferenceNodes:
    sigma: np.ndarray
    weights: np.ndarray
    metadata: dict

    def __post_init__(self):
        sigma = np.ascontiguousarray(_check_sigma(self.sigma))
        weights = np.ascontiguousarray(self.weights, dtype=np.float64)
        self.sigma = np.frombuffer(sigma.tobytes(), dtype=np.float64).reshape(sigma.shape)
        self.weights = np.frombuffer(weights.tobytes(), dtype=np.float64).reshape(weights.shape)
        if self.weights.shape != (len(self.sigma),) or not np.isfinite(self.weights).all() or np.any(self.weights < 0):
            raise ValueError("invalid reference weights")
        if not np.isclose(self.weights.sum(), 1., atol=2e-14, rtol=0):
            raise ValueError("reference weights must sum to one")
        self.metadata = json.loads(json.dumps(self.metadata, allow_nan=False))
        self.sigma.flags.writeable = self.weights.flags.writeable = False
        cumulative = np.cumsum(self.weights)
        cumulative /= cumulative[-1]
        self.cdf = np.frombuffer(cumulative.tobytes(), dtype=np.float64)

    def authenticate(self, reference):
        if self.metadata.get("binding") != reference.binding:
            raise ValueError("reference nodes belong to another problem or observed world")
        low, high = reference.problem.prior.bounds
        if np.any(self.sigma < low) or np.any(self.sigma > high):
            raise ValueError("reference nodes outside original prior support")

    def save(self, output_root):
        root = Path(output_root)
        root.mkdir(parents=True, exist_ok=False)
        path = root / "nodes.npz"
        with path.open("xb") as handle:
            np.savez(handle, sigma=self.sigma, weights=self.weights)
        manifest = {"metadata": self.metadata, "file": path.name,
                    "sha256": sha256(path.read_bytes()).hexdigest(),
                    "sigma": array_descriptor(self.sigma), "weights": array_descriptor(self.weights)}
        with (root / "manifest.json").open("x") as handle:
            json.dump(manifest, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
        return manifest

    @classmethod
    def load(cls, output_root, reference, *, manifest_sha256):
        root = Path(output_root)
        manifest_bytes = (root / "manifest.json").read_bytes()
        if sha256(manifest_bytes).hexdigest() != manifest_sha256:
            raise ValueError("node manifest hash mismatch")
        manifest = json.loads(manifest_bytes)
        if manifest["file"] != "nodes.npz":
            raise ValueError("unexpected node payload path")
        path = root / manifest["file"]
        if sha256(path.read_bytes()).hexdigest() != manifest["sha256"]:
            raise ValueError("node payload hash mismatch")
        with np.load(path, allow_pickle=False) as bank:
            if set(bank.files) != {"sigma", "weights"}:
                raise ValueError("unexpected node arrays")
            for name in bank.files:
                if array_descriptor(bank[name]) != manifest[name]:
                    raise ValueError("node array hash mismatch")
            result = cls(bank["sigma"], bank["weights"], manifest["metadata"])
        result.authenticate(reference)
        return result

    def sample_cohort(self, reference, *, count=2048, seed=20261016, cohort=0, batch_size=256):
        """Independent iid mixture rows; batching does not change RNG consumption."""
        self.authenticate(reference)
        if (type(count) is not int or count < 1 or type(cohort) is not int or cohort < 0
                or type(batch_size) is not int or batch_size < 1 or type(seed) is not int or seed < 0):
            raise ValueError("positive count/batch and nonnegative cohort required")
        tag = np.frombuffer(bytes.fromhex(_json_hash(reference.binding)), dtype="<u4").tolist()
        rng = np.random.default_rng(np.random.SeedSequence([seed, 0x52454632, *tag, cohort]))
        chosen = np.searchsorted(self.cdf, rng.random(count), side="right")
        # Separate RNG stream fixes per-row Gaussian variates independent of batching.
        gaussian = np.random.default_rng(np.random.SeedSequence([seed, 0x47415532, *tag, cohort]))
        joint = np.empty((count, 21), dtype=np.float64)
        physics_positions = [reference.problem.parameter_order.index(name) for name in reference.names]
        noise_positions = [reference.problem.parameter_order.index(name) for name in reference.problem.prior.noise_names]
        for start in range(0, count, batch_size):
            stop = min(start + batch_size, count)
            sigma = self.sigma[chosen[start:stop]]
            physics = reference.conditional_draw(sigma, gaussian.standard_normal((len(sigma), 15)))
            joint[start:stop, physics_positions] = physics
            joint[start:stop, noise_positions] = sigma
        if not np.isfinite(joint).all():
            raise ValueError("nonfinite unfiltered reference posterior draw")
        return joint

    def cohorts(self, reference, *, count=2048, additional=199, seed=20261016, batch_size=256):
        """Yield the primary plus 199 additional cohorts without retaining them."""
        for cohort in range(additional + 1):
            yield self.sample_cohort(reference, count=count, seed=seed, cohort=cohort, batch_size=batch_size)

    def projected_cdfs(self, reference, directions, queries, *, batch_size=256, deadline=None):
        """Analytic mixture CDFs for physical coordinates and joint projections."""
        self.authenticate(reference)
        directions = np.asarray(directions, dtype=np.float64)
        queries = np.asarray(queries, dtype=np.float64)
        if directions.ndim != 2 or directions.shape[1] != 21 or queries.ndim != 1:
            raise ValueError("directions[k,21] and queries[q] required")
        ppos = [reference.problem.parameter_order.index(name) for name in reference.names]
        npos = [reference.problem.parameter_order.index(name) for name in reference.problem.prior.noise_names]
        dp, dn = directions[:, ppos], directions[:, npos]
        result = np.zeros((len(directions), len(queries)))
        if reference.backend == "jax":
            import jax
            device = reference._jax_device
            device_directions = jax.device_put(directions, device)
            device_queries = jax.device_put(queries, device)
            for start in range(0, len(self.sigma), batch_size):
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError("bounded reference witness deadline exhausted")
                stop = min(start + batch_size, len(self.sigma))
                cdf, minimum_variance = reference._jax_projection(
                    jax.device_put(self.sigma[start:stop], device),
                    jax.device_put(self.weights[start:stop], device), device_directions, device_queries,
                    *reference._jax_positions, *reference._jax_constants)
                if float(minimum_variance) < -1e-12:
                    raise ValueError("negative projected Gaussian variance")
                result += np.asarray(cdf, dtype=np.float64)
            return result
        for start in range(0, len(self.sigma), batch_size):
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("bounded reference witness deadline exhausted")
            stop = min(start + batch_size, len(self.sigma))
            mean, covariance, _ = reference.conditional(self.sigma[start:stop])
            location = mean @ dp.T + self.sigma[start:stop] @ dn.T
            variance = np.einsum("ki,nij,kj->nk", dp, covariance, dp)
            if np.any(variance < -1e-12):
                raise ValueError("negative projected Gaussian variance")
            sd = np.sqrt(np.maximum(variance, 0))
            for j, query in enumerate(queries):
                cdf = np.empty_like(location)
                nonzero = sd > 0
                cdf[nonzero] = ndtr((query - location[nonzero]) / sd[nonzero])
                cdf[~nonzero] = location[~nonzero] <= query
                result[:, j] += self.weights[start:stop] @ cdf
        return result


def combine_scrambles(banks):
    if len(banks) != SCRAMBLES:
        raise ValueError("exactly four independently scrambled banks required")
    first = banks[0].metadata
    if any(bank.metadata["binding"] != first["binding"] or bank.metadata["power"] != first["power"]
           for bank in banks):
        raise ValueError("scramble binding/power mismatch")
    seeds = [bank.metadata["seed"] for bank in banks]
    if len(set(seeds)) != SCRAMBLES:
        raise ValueError("scramble seeds must be distinct")
    logs = np.asarray([bank.metadata["log_evidence"] for bank in banks])
    mixture_weights = np.exp(logs - logsumexp(logs))
    metadata = dict(first, seeds=seeds, scrambles=SCRAMBLES,
                    log_evidence=float(logsumexp(logs) - math.log(SCRAMBLES)),
                    scramble_log_evidences=logs.tolist())
    for key in ("physics_mean", "noise_mean", "noise_second_moment"):
        metadata[key] = np.einsum("n,n...->...", mixture_weights,
                                 np.asarray([bank.metadata[key] for bank in banks])).tolist()
    mean = np.asarray(metadata["physics_mean"])
    second = sum(weight * (np.asarray(bank.metadata["physics_covariance"])
                           + np.outer(bank.metadata["physics_mean"], bank.metadata["physics_mean"]))
                 for weight, bank in zip(mixture_weights, banks))
    metadata["physics_covariance"] = (second - np.outer(mean, mean)).tolist()
    weights = np.concatenate([weight * bank.weights for weight, bank in zip(mixture_weights, banks)])
    metadata["effective_nodes"] = float(1 / np.square(weights).sum())
    return ReferenceNodes(np.concatenate([bank.sigma for bank in banks]), weights, metadata)


def scramble_witness_summary(banks, values):
    """Use the exact evidence mixture of the returned combined node bank."""
    values = np.asarray(values, dtype=np.float64)
    if len(banks) != SCRAMBLES or values.shape[0] != SCRAMBLES:
        raise ValueError("four scramble witness arrays required")
    evidence = np.asarray([bank.metadata["log_evidence"] for bank in banks])
    center = np.einsum("n,n...->...", np.exp(evidence - logsumexp(evidence)), values)
    return center, np.max(np.abs(values - center), axis=0)


def refinement_witnesses(reference):
    """Fixed prior-scaled physical axes and 32 complete-vector joint projections."""
    ppos = [reference.problem.parameter_order.index(name) for name in reference.names]
    npos = [reference.problem.parameter_order.index(name) for name in reference.problem.prior.noise_names]
    scales = np.empty(21)
    scales[ppos] = np.sqrt(np.diag(reference.prior_covariance))
    # Exact truncated-prior SD; these scales never use the observed posterior.
    from scipy.stats import truncnorm
    noise_sd = float(truncnorm.std((.01 - .255) / .1225, (.5 - .255) / .1225, loc=.255, scale=.1225))
    scales[npos] = noise_sd
    random = np.random.default_rng(763019).normal(size=(32, 21))
    random /= np.linalg.norm(random, axis=1)[:, None]
    directions = np.concatenate([np.eye(21), random]) / scales
    return directions, np.linspace(-8., 8., 129)


def refine_reference(reference, *, marginal_resolution, joint_resolution=None, powers=POWERS,
                     seed=20261016, batch_size=512, witness_batch_size=256,
                     deadline=None, bank_provider=None, bank_callback=None):
    """Four-scramble refinement with explicit resolution-relative witness gates.

    Projected CDF witnesses are joint-sensitive diagnostics, not a mathematical
    uniform bound on every multivariate functional. The caller must additionally
    validate the joint metric and independent nested check before a verdict.
    Exhausting powers never converts inadequate precision into a pass.
    """
    if not np.isfinite(marginal_resolution) or marginal_resolution <= 0:
        raise ValueError("positive marginal sampling resolution required")
    joint_resolution = marginal_resolution if joint_resolution is None else joint_resolution
    if not np.isfinite(joint_resolution) or joint_resolution <= 0:
        raise ValueError("positive joint witness resolution required")
    if tuple(powers) != POWERS:
        raise ValueError("approved reference powers are fixed at16,18,20,22")
    directions, queries = refinement_witnesses(reference)
    history, previous = [], None
    cpu_check = reference.verify_cpu()
    final = None
    for power in powers:
        banks = []
        for j in range(SCRAMBLES):
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("bounded reference refinement deadline exhausted")
            scramble_seed = int(np.random.SeedSequence([seed, power, j]).generate_state(1)[0])
            bank = None if bank_provider is None else bank_provider(power, j, scramble_seed)
            if bank is None:
                bank = reference.integrate(power=power, seed=scramble_seed,
                                            batch_size=batch_size, deadline=deadline)
            bank.authenticate(reference)
            if bank.metadata["power"] != power or bank.metadata["seed"] != scramble_seed:
                raise ValueError("provided quadrature bank power/seed mismatch")
            if bank_callback is not None:
                bank_callback(bank, power, j)
            banks.append(bank)
        values = np.asarray([bank.projected_cdfs(reference, directions, queries,
                             batch_size=witness_batch_size, deadline=deadline) for bank in banks])
        final = combine_scrambles(banks)
        center, spread = scramble_witness_summary(banks, values)
        delta = np.full_like(center, np.inf) if previous is None else np.abs(center - previous)
        marginal_error = float(max(spread[:21].max(), delta[:21].max()))
        joint_error = float(max(spread[21:].max(), delta[21:].max()))
        passed = marginal_error <= ERROR_FRACTION * marginal_resolution and joint_error <= ERROR_FRACTION * joint_resolution
        history.append({"power": power, "seeds": final.metadata["seeds"],
                        "marginal_witness_error": None if not np.isfinite(marginal_error) else marginal_error,
                        "joint_projection_witness_error": None if not np.isfinite(joint_error) else joint_error,
                        "scramble_log_evidences": final.metadata["scramble_log_evidences"],
                        "resolution_gate_passed": bool(passed)})
        previous = center
        if passed:
            break
    result = {"history": history, "cpu_check": cpu_check,
              "marginal_resolution": marginal_resolution, "joint_projection_resolution": joint_resolution,
              "error_fraction": ERROR_FRACTION, "state": "inconclusive",
              "witness_gate_passed": history[-1]["resolution_gate_passed"],
              "pending": ["independent_nested_check", "joint_metric_reference_check"],
              "directions": directions.tolist(), "queries": queries.tolist()}
    final.metadata["refinement"] = result
    return final, result


def _legendre_cdf_ratios(nodes, queries):
    """Integrate the Gauss-node interpolating polynomial up to each CDF query.

    Ratios multiply ordinary Gauss weights. This avoids a discontinuous step
    integrand; doubled-order comparison still controls the interpolation error.
    No clipping of the resulting CDF or quadrature contributions is performed.
    """
    order = len(nodes)
    polynomial = np.polynomial.legendre.legvander(nodes, order - 1)
    at_query = np.polynomial.legendre.legvander(queries, order)
    integrals = np.empty((len(queries), order))
    integrals[:, 0] = np.asarray(queries) + 1
    for k in range(1, order):
        integrals[:, k] = (at_query[:, k + 1] - at_query[:, k - 1]) / (2 * k + 1)
    return (polynomial * ((2 * np.arange(order) + 1) / 2)) @ integrals.T


def nested_global_noise_check(reference, *, global_order=20, noise_order=64, batch_size=128, deadline=None):
    """Independent 3D global Gauss-Hermite × six 1D noise Gauss-Legendre integrals.

    Local physics is integrated in observation space. The Gaussian global
    proposal uses only the known-noise(.255) observation-space posterior, widened
    by factor two in SD; no Sobol nodes, weights or posterior draws are consumed.
    Orders must be refined independently by the caller; a single order is not a
    convergence certificate.
    """
    if global_order < 4 or noise_order < 16 or batch_size < 1:
        raise ValueError("nested orders global>=4,noise>=16 required")
    prior = reference.problem.prior.base
    sigma = np.full((1, 6), .255)
    mean, covariance, _ = reference.covariance_space_conditional(sigma)
    proposal_mean = mean[0, :3]
    proposal_covariance = covariance[0, :3, :3] * 4
    proposal_chol = np.linalg.cholesky(proposal_covariance)
    prior_chol = np.linalg.cholesky(prior.global_covariance)
    gh_x, gh_w = np.polynomial.hermite.hermgauss(global_order)
    gl_x, gl_w = np.polynomial.legendre.leggauss(noise_order)
    low, high = reference.problem.prior.bounds
    noise = (gl_x + 1) * (high - low) / 2 + low
    log_noise_weight = np.log(gl_w * (high - low) / 2)
    local = []
    for block, conditional in zip(reference.problem.observation_model.blocks, prior.conditional_blocks):
        signal = block.design @ conditional.covariance @ block.design.T
        eig, rotation = np.linalg.eigh(signal)
        eig = np.maximum(eig, 0)
        variance = eig[None, :] + noise[:, None] ** 2
        local.append((block, conditional, rotation, variance,
                      np.log(variance).sum(axis=1),
                      log_noise_weight + reference.problem.prior.noise_log_prob(block.dataset_id, noise)))
    total_log = -np.inf
    global_first = np.zeros(3)
    global_second = np.zeros((3, 3))
    noise_first = np.zeros(6)
    noise_cdf = np.zeros((6, 9))
    queries = np.linspace(low, high, 11)[1:-1]
    cdf_ratios = _legendre_cdf_ratios(gl_x, (queries - low) * 2 / (high - low) - 1)
    iterator = iter(itertools.product(range(global_order), repeat=3))
    while True:
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("bounded independent nested reference deadline exhausted")
        indices = np.asarray(list(itertools.islice(iterator, batch_size)), dtype=int)
        if not len(indices):
            break
        z = math.sqrt(2) * gh_x[indices]
        h = proposal_mean + z @ proposal_chol.T
        prior_z = np.linalg.solve(prior_chol, (h - prior.global_mean).T).T
        log_ratio = (-.5 * np.square(prior_z).sum(axis=1) - np.log(np.diag(prior_chol)).sum()
                     + .5 * np.square(z).sum(axis=1) + np.log(np.diag(proposal_chol)).sum())
        log_weight = np.log(gh_w[indices]).sum(axis=1) - 1.5 * math.log(math.pi) + log_ratio
        conditional_noise, conditional_cdf = [], []
        for block, conditional, rotation, variance, logdet, prior_log_weight in local:
            pred = (h @ conditional.population_map.T + conditional.offset) @ block.design.T + block.offset
            residual = (reference.observations.values[block.dataset_id] - pred) @ rotation
            logs = (-.5 * (block.observation_size * math.log(2 * math.pi) + logdet[None, :]
                          + np.einsum("bm,sm->bs", residual * residual, 1 / variance))
                    + prior_log_weight)
            log_integral = logsumexp(logs, axis=1)
            weights = np.exp(logs - log_integral[:, None])
            log_weight += log_integral
            conditional_noise.append(weights @ noise)
            conditional_cdf.append(weights @ cdf_ratios)
        batch_log = float(logsumexp(log_weight))
        merged = float(np.logaddexp(total_log, batch_log))
        old_factor = math.exp(total_log - merged)
        weight = np.exp(log_weight - merged)
        global_first = global_first * old_factor + weight @ h
        global_second = global_second * old_factor + np.einsum("n,ni,nj->ij", weight, h, h)
        noise_first = noise_first * old_factor + weight @ np.asarray(conditional_noise).T
        noise_cdf = noise_cdf * old_factor + np.einsum("n,inq->iq", weight, np.asarray(conditional_cdf))
        total_log = merged
    return {"schema": SCHEMA + ".nested", "binding": reference.binding,
            "global_order": global_order, "noise_order": noise_order,
            "global_dimensions": 3, "noise_integral_dimensions": 1,
            "log_evidence": total_log, "global_mean": global_first.tolist(),
            "global_covariance": (global_second - np.outer(global_first, global_first)).tolist(),
            "noise_mean": noise_first.tolist(), "noise_cdf_queries": queries.tolist(),
            "noise_cdfs": noise_cdf.tolist(), "state": "requires_independent_order_refinement"}


def compare_nested(reference, nodes, checks, *, cdf_tolerance, log_evidence_tolerance,
                   global_mean_tolerance, global_covariance_tolerance, noise_mean_tolerance):
    """Require three nested corners, refining each axis separately, and Sobol agreement."""
    nodes.authenticate(reference)
    if len(checks) < 3 or any(check["binding"] != reference.binding for check in checks):
        raise ValueError("three bound nested corners required")
    final = checks[-1]
    global_corners = [check for check in checks[:-1]
                      if check["global_order"] < final["global_order"] and check["noise_order"] == final["noise_order"]]
    noise_corners = [check for check in checks[:-1]
                     if check["global_order"] == final["global_order"] and check["noise_order"] < final["noise_order"]]
    if not global_corners or not noise_corners:
        raise ValueError("separate global-coarse and noise-coarse nested corners required")
    corners = [max(global_corners, key=lambda check: check["global_order"]),
               max(noise_corners, key=lambda check: check["noise_order"])]
    tolerances = (cdf_tolerance, log_evidence_tolerance, global_mean_tolerance,
                  global_covariance_tolerance, noise_mean_tolerance)
    if any(not np.isfinite(tolerance) or tolerance <= 0 for tolerance in tolerances):
        raise ValueError("positive independent-check tolerances required")
    queries = np.asarray(final["noise_cdf_queries"])
    if any(check["noise_cdf_queries"] != final["noise_cdf_queries"] for check in corners):
        raise ValueError("nested CDF query mismatch")
    qmc_cdf = np.asarray([nodes.weights @ (nodes.sigma[:, i, None] <= queries) for i in range(6)])
    cdf_delta = float(max([np.max(np.abs(np.asarray(final["noise_cdfs"]) - qmc_cdf))]
                         + [np.max(np.abs(np.asarray(final["noise_cdfs"]) - check["noise_cdfs"])) for check in corners]))
    evidence_delta = float(max([abs(final["log_evidence"] - nodes.metadata["log_evidence"])]
                              + [abs(final["log_evidence"] - check["log_evidence"]) for check in corners]))
    global_mean = np.asarray(final["global_mean"])
    global_covariance = np.asarray(final["global_covariance"])
    noise_mean = np.asarray(final["noise_mean"])
    global_mean_error = float(max([np.max(np.abs(global_mean - nodes.metadata["physics_mean"][:3]))]
                                 + [np.max(np.abs(global_mean - check["global_mean"])) for check in corners]))
    global_covariance_error = float(max([np.max(np.abs(global_covariance - np.asarray(nodes.metadata["physics_covariance"])[:3, :3]))]
                                       + [np.max(np.abs(global_covariance - check["global_covariance"])) for check in corners]))
    noise_mean_error = float(max([np.max(np.abs(noise_mean - nodes.metadata["noise_mean"]))]
                                + [np.max(np.abs(noise_mean - check["noise_mean"])) for check in corners]))
    errors = (cdf_delta, evidence_delta, global_mean_error, global_covariance_error, noise_mean_error)
    passed = all(np.isfinite(error) and error <= tolerance for error, tolerance in zip(errors, tolerances))
    return {"passed": bool(passed), "state": "verified" if passed else "inconclusive",
            "noise_cdf_max_error": cdf_delta, "log_evidence_max_error": evidence_delta,
            "cdf_tolerance": cdf_tolerance, "log_evidence_tolerance": log_evidence_tolerance,
            "global_mean_max_error": global_mean_error, "global_mean_tolerance": global_mean_tolerance,
            "global_covariance_max_error": global_covariance_error, "global_covariance_tolerance": global_covariance_tolerance,
            "noise_mean_max_error": noise_mean_error, "noise_mean_tolerance": noise_mean_tolerance,
            "moment_units": "physical coordinates; covariance in physical squared units",
            "checks": checks}


def standalone_references(problem):
    """Unchanged own-data references, with true local marginal Gaussian priors."""
    return references_from_problem(problem)


def sampling_resolution_pilot(problem, observations, *, data_provenance, seed=20261016, deadline=None):
    """Measured common coordinate resolution from 199 independent reference pairs.

    Uses diagnostic world0, mechanical_1 noise SD only. Its continuous 1D KS
    law is distribution-free; it is conservative relative to the later maximum
    over worlds/targets/coordinates. No candidate output enters this calculation.
    """
    if observations.batch_size != 1 or not isinstance(data_provenance, dict):
        raise ValueError("one observed diagnostic world and bound data provenance required")
    if data_provenance.get("role") != "diagnostic" or data_provenance.get("index") != 0:
        raise ValueError("numerical pilot is predeclared diagnostic world0")
    if not isinstance(data_provenance.get("manifest_sha256"), str) or len(data_provenance["manifest_sha256"]) != 64:
        raise ValueError("bound diagnostic data manifest SHA256 required")
    def check_deadline():
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("bounded numerical resolution pilot deadline exhausted")
    check_deadline()
    from scipy.stats import ks_2samp
    reference = standalone_references(problem)["mechanical_1"]
    observation = observations.values["mechanical_1"]
    prepared = reference.prepare_observed(observation, deadline=deadline)
    controls = []
    # Dedicated numerical namespace; none of the calibration cohort RNGs reuse it.
    for pair in range(199):
        check_deadline()
        samples = []
        for side in range(2):
            check_deadline()
            rng = np.random.default_rng(np.random.SeedSequence([seed, 0x4E554D32, pair, side]))
            samples.append(reference.sample_posterior(observation, prepared, draws=2048, rng=rng)[:, -1])
        controls.append(float(ks_2samp(*samples, method="asymp").statistic))
    envelope = float(np.quantile(controls, .95, method="higher"))
    sources = {str(Path(__file__).resolve()): sha256(Path(__file__).read_bytes()).hexdigest()}
    dependency = Path(references_from_problem.__code__.co_filename).resolve()
    sources[str(dependency)] = sha256(dependency.read_bytes()).hexdigest()
    return {"schema": SCHEMA + ".numerical_resolution_pilot",
            "binding": {"scientific_fingerprint": problem.scientific_fingerprint,
                        "observation_sha256": array_descriptor(flatten_observations(observations))["sha256"],
                        "data_provenance": data_provenance},
            "seed": seed, "rng_namespace": int(0x4E554D32), "rng": "numpy.PCG64.SeedSequence[seed,namespace,pair,side]",
            "pairs": 199, "samples_per_side": 2048, "dataset": "mechanical_1", "coordinate": "sigma_m1",
            "control_KS": controls, "quantile": .95, "quantile_method": "higher",
            "marginal_resolution": envelope, "numerical_CDF_tolerance": ERROR_FRACTION * envelope,
            "standalone_controls": prepared["convergence"], "sources": sources}
