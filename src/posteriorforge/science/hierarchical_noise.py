"""Unknown absolute Gaussian noise, with an independent Gaussian-integrated reference.

The physics/hierarchy prior and linear forwards are the known-noise fixture's
unchanged ones. Only six unpooled noise scales are added. Conditional physics
posteriors are exact Gaussian distributions; integration over noise is numerical.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np
from scipy.special import logsumexp
from scipy.stats import qmc, truncnorm

from .hierarchical_exact import (
    GaussianObservationBlock, LinearGaussianHierarchyPrior, LinearGaussianObservationModel,
    _rows, build_exact_heterogeneous_problem, draw_exact_fixture_case,
)
from .hierarchical_problem import HierarchicalCoordinate, HierarchicalObservations, HierarchicalProblemSpec


SCHEMA = "bayesarena.inference.absolute_noise_hierarchy.v1"


@dataclass(frozen=True)
class IndependentNoiseHierarchyPrior:
    """Independent physical SD priors, uniform unless explicitly configured.

    ``noise_gaussian_priors`` follows ``noise_names`` order. Each non-None
    entry is the underlying Gaussian (mean, standard deviation), truncated to
    ``bounds``; None entries keep their uniform prior. These scales are not
    pooled, and the parameters are not moments of the truncated distribution.
    """

    base: LinearGaussianHierarchyPrior
    noise_names: tuple[str, ...]
    bounds: tuple[float, float] = (.01, .50)
    noise_gaussian_priors: tuple[tuple[float, float] | None, ...] | None = None

    def __post_init__(self):
        names = tuple(self.noise_names)
        bounds = tuple(float(v) for v in self.bounds)
        if len(names) != len(self.base.conditional_blocks) or len(set(names)) != len(names):
            raise ValueError("one distinct noise coordinate is required per dataset")
        if set(names).intersection(self.base.parameter_order):
            raise ValueError("noise and physics coordinate names overlap")
        if len(bounds) != 2 or not np.isfinite(bounds).all() or not 0 < bounds[0] < bounds[1]:
            raise ValueError("noise prior bounds must be positive, finite and increasing")
        object.__setattr__(self, "noise_names", names)
        object.__setattr__(self, "bounds", bounds)
        if self.noise_gaussian_priors is not None:
            entries = tuple(self.noise_gaussian_priors)
            if len(entries) != len(names):
                raise ValueError("one Gaussian prior entry is required per noise coordinate")
            gaussian = []
            for entry in entries:
                if entry is None:
                    gaussian.append(None)
                    continue
                pair = tuple(float(value) for value in entry)
                if len(pair) != 2 or not np.isfinite(pair).all() or pair[1] <= 0:
                    raise ValueError("noise Gaussian priors require a finite mean and positive standard deviation")
                standardized = np.array([(bound - pair[0]) / pair[1] for bound in bounds])
                if not np.isfinite(standardized).all() or standardized[0] >= standardized[1]:
                    raise ValueError("noise Gaussian standardized bounds must be finite and increasing")
                gaussian.append(pair)
            object.__setattr__(self, "noise_gaussian_priors", tuple(gaussian) if any(
                entry is not None for entry in gaussian) else None)

    @property
    def global_parameter_order(self):
        return self.base.global_parameter_order

    @property
    def global_names(self):
        return self.base.global_names

    @property
    def parameter_order(self):
        return self.global_names + tuple(name for block, noise in zip(self.base.conditional_blocks, self.noise_names)
                                         for name in (*block.local_names, noise))

    def noise_name(self, dataset_id):
        index = [block.dataset_id for block in self.base.conditional_blocks].index(dataset_id)
        return self.noise_names[index]

    def conditional_parameter_order(self, dataset_id):
        return (*self.base.conditional_parameter_order(dataset_id), self.noise_name(dataset_id))

    def manifest(self):
        manifest = {"schema": SCHEMA + ".prior", "base": self.base.manifest(),
                "parameter_order": list(self.parameter_order), "noise_names": list(self.noise_names),
                "noise_prior": {"family": "independent_unpooled_uniform_standard_deviation", "bounds": list(self.bounds)}}
        if self.noise_gaussian_priors is not None:
            entries = []
            for block, name, entry in zip(self.base.conditional_blocks, self.noise_names, self.noise_gaussian_priors):
                item = {"dataset_id": block.dataset_id, "noise_name": name,
                        "family": "uniform" if entry is None else "truncated_gaussian",
                        "bounds": list(self.bounds)}
                if entry is not None:
                    item.update(underlying_mean=entry[0], underlying_standard_deviation=entry[1])
                entries.append(item)
            family = ("independent_unpooled_truncated_gaussian_standard_deviation"
                      if all(entry is not None for entry in self.noise_gaussian_priors)
                      else "independent_unpooled_mixed_standard_deviation")
            manifest["noise_prior"] = {"family": family, "bounds": list(self.bounds), "coordinates": entries}
        return manifest

    def sample(self, size, rng):
        physics = self.base.sample(size, rng)
        noise = rng.uniform(*self.bounds, size=(size, len(self.noise_names)))
        if self.noise_gaussian_priors is not None:
            # Reuse exactly the uniform route's draws, including mixed priors.
            # This leaves subsequent physics/observation RNG streams paired.
            for index, entry in enumerate(self.noise_gaussian_priors):
                if entry is not None:
                    mean, sd = entry
                    uniform = (noise[:, index] - self.bounds[0]) / (self.bounds[1] - self.bounds[0])
                    quantiles = truncnorm.ppf(uniform, (self.bounds[0] - mean) / sd,
                                              (self.bounds[1] - mean) / sd, loc=mean, scale=sd)
                    # PPF endpoint arithmetic can stray outside by one ulp.
                    noise[:, index] = np.clip(quantiles, *self.bounds)
        columns = {name: physics[:, i] for i, name in enumerate(self.base.parameter_order)}
        columns.update({name: noise[:, i] for i, name in enumerate(self.noise_names)})
        return np.column_stack([columns[name] for name in self.parameter_order])

    def global_log_prob(self, values):
        return self.base.global_log_prob(values)

    def _noise_log_prob(self, values, index=0):
        if self.noise_gaussian_priors is not None and self.noise_gaussian_priors[index] is not None:
            mean, sd = self.noise_gaussian_priors[index]
            return truncnorm.logpdf(values, (self.bounds[0] - mean) / sd,
                                    (self.bounds[1] - mean) / sd, loc=mean, scale=sd)
        return np.where((values >= self.bounds[0]) & (values <= self.bounds[1]),
                        -math.log(self.bounds[1] - self.bounds[0]), -np.inf)

    def noise_log_prob(self, dataset_id, values):
        """Normalized physical noise-SD log density for the named dataset."""
        return self._noise_log_prob(np.asarray(values, dtype=float), self.noise_names.index(self.noise_name(dataset_id)))

    def conditional_log_prob(self, dataset_id, global_values, local_values):
        local = _rows(local_values, len(self.conditional_parameter_order(dataset_id)))
        return (self.base.conditional_log_prob(dataset_id, global_values, local[:, :-1])
                + self.noise_log_prob(dataset_id, local[:, -1]))

    def log_prob(self, values):
        values = _rows(values, len(self.parameter_order))
        physics = values[:, [self.parameter_order.index(name) for name in self.base.parameter_order]]
        result = self.base.log_prob(physics)
        for index, name in enumerate(self.noise_names):
            result += self._noise_log_prob(values[:, self.parameter_order.index(name)], index)
        return result


@dataclass(frozen=True)
class InferredGaussianNoiseModel:
    parameter_names: tuple[str, ...]
    base: LinearGaussianObservationModel
    noise_names: tuple[str, ...]

    def __post_init__(self):
        object.__setattr__(self, "parameter_names", tuple(self.parameter_names))
        object.__setattr__(self, "noise_names", tuple(self.noise_names))
        if len(self.noise_names) != len(self.base.blocks) or len(set(self.parameter_names)) != len(self.parameter_names):
            raise ValueError("invalid noise-model coordinate membership")
        if not set((*self.base.parameter_order, *self.noise_names)).issubset(self.parameter_names):
            raise ValueError("noise model refers to undeclared coordinates")

    @property
    def parameter_order(self):
        return self.parameter_names

    @property
    def dataset_ids(self):
        return self.base.dataset_ids

    @property
    def blocks(self):
        return self.base.blocks

    def manifest(self):
        return {"schema": SCHEMA + ".likelihood", "parameter_order": list(self.parameter_order),
                "base_forward": self.base.manifest(), "noise_names": list(self.noise_names),
                "noise": "independent_points; standard_deviation=sigma_i; absolute_not_relative"}

    def predict(self, values):
        values = _rows(values, len(self.parameter_order))
        physics = values[:, [self.parameter_order.index(name) for name in self.base.parameter_order]]
        return self.base.predict(physics)

    def sample(self, values, rng):
        values = _rows(values, len(self.parameter_order))
        predictions = self.predict(values)
        result = {}
        for block, noise_name in zip(self.blocks, self.noise_names):
            sigma = values[:, self.parameter_order.index(noise_name)]
            if np.any(sigma <= 0):
                raise ValueError("noise standard deviation must be positive")
            result[block.dataset_id] = predictions[block.dataset_id] + sigma[:, None] * rng.normal(size=predictions[block.dataset_id].shape)
        return HierarchicalObservations(result)

    def local_log_prob(self, dataset_id, local_parameters, observed):
        index = self.dataset_ids.index(dataset_id)
        block = self.blocks[index]
        physics = np.column_stack([local_parameters[name] for name in block.local_names])
        sigma = np.asarray(local_parameters[self.noise_names[index]])
        data = np.asarray(observed)
        if data.ndim != 2 or data.shape[1] != block.observation_size or len(data) not in (1, len(physics)):
            raise ValueError("observation shape differs from the declared noise block")
        valid = sigma > 0
        safe = np.where(valid, sigma, 1.)
        residual = data - (physics @ block.design.T + block.offset)
        logs = (-.5 * np.square(residual / safe[:, None]).sum(axis=1)
                - block.observation_size * (np.log(safe) + .5 * math.log(2 * math.pi)))
        return np.where(valid, logs, -np.inf)

    def log_prob(self, values, observations):
        values = _rows(values, len(self.parameter_order))
        result = np.zeros(len(values))
        for block, noise_name in zip(self.blocks, self.noise_names):
            names = (*block.local_names, noise_name)
            mapping = {name: values[:, self.parameter_order.index(name)] for name in names}
            result += self.local_log_prob(block.dataset_id, mapping, observations.values[block.dataset_id])
        return result


def build_inferred_noise_problem(*, acoustic_repeats=1, noise_bounds=(.01, .50), noise_gaussian_priors=None):
    if type(acoustic_repeats) is not int or acoustic_repeats < 1:
        raise ValueError("acoustic repeats must be a positive integer")
    base = build_exact_heterogeneous_problem()
    names = tuple(f"sigma_m{i}" for i in range(1, 4)) + tuple(f"sigma_a{i}" for i in range(1, 4))
    prior = IndependentNoiseHierarchyPrior(base.prior, names, noise_bounds, noise_gaussian_priors)
    coordinates, datasets, blocks = list(base.coordinates[:3]), [], []
    for index, (dataset, block, noise) in enumerate(zip(base.datasets, base.observation_model.blocks, names)):
        coordinates.extend(coordinate for coordinate in base.coordinates if coordinate.dataset_id == dataset.dataset_id)
        coordinates.append(HierarchicalCoordinate(noise, "synthetic_response", "local", dataset.dataset_id,
                                                   lower=noise_bounds[0], upper=noise_bounds[1]))
        repeats = acoustic_repeats if index >= 3 else 1
        points = block.observation_size * repeats
        datasets.append(replace(dataset, local_coordinates=(*dataset.local_coordinates, noise), observation_size=points,
                                controls=tuple(np.repeat(dataset.controls, repeats)),
                                kind="curve" if repeats > 1 else dataset.kind))
        blocks.append(GaussianObservationBlock(block.dataset_id, block.local_names,
                      np.repeat(block.design, repeats, axis=0), np.repeat(block.offset, repeats),
                      np.eye(points) * block.covariance[0, 0]))
    model = InferredGaussianNoiseModel(prior.parameter_order,
              LinearGaussianObservationModel(base.prior.parameter_order, tuple(blocks)), names)
    return HierarchicalProblemSpec(name=f"absolute_noise_hierarchy_acoustic_repeats_{acoustic_repeats}",
             version=SCHEMA, coordinates=tuple(coordinates), families=base.families,
             datasets=tuple(datasets), prior=prior, observation_model=model)


def draw_inferred_noise_case(*, seed=20260930, acoustic_repeats=1, noise_bounds=(.01, .50), noise_gaussian_priors=None):
    problem = build_inferred_noise_problem(acoustic_repeats=acoustic_repeats, noise_bounds=noise_bounds,
                                         noise_gaussian_priors=noise_gaussian_priors)
    known, old_truth, old_observed = draw_exact_fixture_case(seed=seed)
    physics = {name: old_truth[:, i] for i, name in enumerate(known.parameter_order)}
    physics.update({name: np.array([.12 if index < 3 else .08]) for index, name in enumerate(problem.prior.noise_names)})
    truth = np.column_stack([physics[name] for name in problem.parameter_order])
    values = dict(old_observed.values)
    if acoustic_repeats > 1:
        predicted = problem.observation_model.predict(truth)
        rng = np.random.default_rng(np.random.SeedSequence([seed, 416017]))
        for dataset_id in problem.dataset_ids[3:]:
            extra = predicted[dataset_id][:, 1:] + .08 * rng.normal(size=(1, acoustic_repeats - 1))
            values[dataset_id] = np.concatenate((old_observed.values[dataset_id], extra), axis=1)
    return problem, truth, HierarchicalObservations(values)


def draw_benchmark_case(*, fixture="exact", seed=20260930, acoustic_repeats=1):
    if fixture == "exact":
        if acoustic_repeats != 1:
            raise ValueError("acoustic repeats apply only to the inferred-noise fixture")
        return draw_exact_fixture_case(seed=seed)
    if fixture == "inferred-noise":
        return draw_inferred_noise_case(seed=seed, acoustic_repeats=acoustic_repeats)
    raise ValueError("unknown benchmark fixture")


class GaussianIntegratedNoiseReference:
    """Independent matrix algebra, never TMCMC/HNPE densities or samples."""

    def __init__(self, problem, observations):
        if not isinstance(problem.prior, IndependentNoiseHierarchyPrior):
            raise TypeError("noise reference requires the declared Gaussian physics/independent noise prior")
        problem.validate_observations(observations)
        if observations.batch_size != 1:
            raise ValueError("reference accepts one observed world")
        self.problem = problem
        self.prior = problem.prior.base
        self.names = self.prior.parameter_order
        d = len(self.names)
        # Assemble the complete prior covariance without calling prior.log_prob.
        self.prior_mean = np.zeros(d)
        mapping = np.zeros((d, len(self.prior.global_names)))
        globals_count = len(self.prior.global_names)
        mapping[:globals_count] = np.eye(globals_count)
        self.prior_mean[:globals_count] = self.prior.global_mean
        independent = np.zeros((d, d))
        position = {name: i for i, name in enumerate(self.names)}
        for block in self.prior.conditional_blocks:
            indices = [position[name] for name in block.local_names]
            mapping[indices] = block.population_map
            self.prior_mean[indices] = block.population_map @ self.prior.global_mean + block.offset
            independent[np.ix_(indices, indices)] = block.covariance
        self.prior_covariance = mapping @ self.prior.global_covariance @ mapping.T + independent
        self.precision = np.linalg.inv(self.prior_covariance)
        self.natural = self.precision @ self.prior_mean
        self.prior_quadratic = self.prior_mean @ self.natural
        self.prior_logdet = np.linalg.slogdet(self.prior_covariance)[1]
        matrices, rhs, squared, counts, designs, centered = [], [], [], [], [], []
        for block in problem.observation_model.blocks:
            design = np.zeros((block.observation_size, d))
            design[:, [position[name] for name in block.local_names]] = block.design
            residual = observations.values[block.dataset_id][0] - block.offset
            matrices.append(design.T @ design)
            rhs.append(design.T @ residual)
            squared.append(residual @ residual)
            counts.append(block.observation_size)
            designs.append(design)
            centered.append(residual)
        self.matrices, self.rhs = np.array(matrices), np.array(rhs)
        self.squared, self.counts = np.array(squared), np.array(counts)
        self.design = np.concatenate(designs)
        self.centered = np.concatenate(centered)
        # Reference-only proposal: integrate efficiently where residual scatter
        # puts noise mass. These summaries never enter either inference backend.
        low, high = problem.prior.bounds
        self.proposal_grid = np.linspace(low, high, 16385)
        self.proposal_cdfs, self.residual_dof, self.residual_sse = [], [], []
        for design, residual in zip(designs, centered):
            left, singular, _ = np.linalg.svd(design, full_matrices=False)
            rank = int(np.sum(singular > singular[0] * 1e-10))
            remainder = residual - left[:, :rank] @ (left[:, :rank].T @ residual)
            dof, sse = len(residual) - rank, float(remainder @ remainder)
            logs = -dof * np.log(self.proposal_grid) - .5 * sse / np.square(self.proposal_grid)
            density = np.exp(logs - logs.max())
            delta = np.diff(self.proposal_grid)
            masses = .5 * (density[1:] + density[:-1]) * delta
            masses = .8 * masses / masses.sum() + .2 * delta / (high - low)
            self.proposal_cdfs.append(np.r_[0., np.cumsum(masses)])
            self.proposal_cdfs[-1][-1] = 1.
            self.residual_dof.append(dof)
            self.residual_sse.append(sse)

    def conditional(self, sigma):
        sigma = _rows(sigma, len(self.problem.datasets))
        if np.any(sigma <= 0):
            raise ValueError("reference noise scales must be positive")
        weight = np.reciprocal(np.square(sigma))
        precision = self.precision + np.einsum("ni,ijk->njk", weight, self.matrices)
        natural = self.natural + weight @ self.rhs
        chol = np.linalg.cholesky(precision)
        identity = np.broadcast_to(np.eye(len(self.names)), precision.shape)
        inverse_chol = np.linalg.solve(chol, identity)
        covariance = np.swapaxes(inverse_chol, -1, -2) @ inverse_chol
        mean = np.einsum("nij,nj->ni", covariance, natural)
        quadratic = self.prior_quadratic + weight @ self.squared - np.sum(natural * mean, axis=1)
        logdet = 2 * np.log(np.diagonal(chol, axis1=1, axis2=2)).sum(axis=1)
        log_evidence = (-.5 * (self.counts.sum() * math.log(2 * math.pi) + self.prior_logdet + logdet + quadratic)
                        - np.log(sigma) @ self.counts)
        return mean, covariance, log_evidence

    def integrate(self, *, power=16, seed=603091, draws=32768, batch_size=512):
        if power < 4 or power > 22 or draws < 8 or batch_size < 1:
            raise ValueError("invalid bounded noise quadrature settings")
        low, high = self.problem.prior.bounds
        uniform = qmc.Sobol(len(self.problem.datasets), scramble=True, seed=seed).random_base2(power)
        sigma = np.empty_like(uniform)
        log_proposal = np.zeros(len(uniform))
        for index, cdf in enumerate(self.proposal_cdfs):
            cells = np.clip(np.searchsorted(cdf, uniform[:, index], side="right") - 1, 0, len(cdf) - 2)
            masses = cdf[cells + 1] - cdf[cells]
            widths = self.proposal_grid[cells + 1] - self.proposal_grid[cells]
            fraction = (uniform[:, index] - cdf[cells]) / masses
            sigma[:, index] = self.proposal_grid[cells] + fraction * widths
            log_proposal += np.log(masses / widths)
        log_prior = -len(self.problem.datasets) * math.log(high - low)
        if self.problem.prior.noise_gaussian_priors is not None:
            log_prior = sum(self.problem.prior.noise_log_prob(dataset_id, sigma[:, index])
                            for index, dataset_id in enumerate(self.problem.dataset_ids))
        logs = []
        log_scale, mass = -np.inf, 0.
        first = np.zeros(len(self.names))
        second = np.zeros((len(self.names), len(self.names)))
        for start in range(0, len(sigma), batch_size):
            mean, covariance, evidence = self.conditional(sigma[start:start + batch_size])
            prior_batch = log_prior if np.ndim(log_prior) == 0 else log_prior[start:start + len(evidence)]
            evidence = evidence + prior_batch - log_proposal[start:start + len(evidence)]
            logs.append(evidence)
            new_scale = max(log_scale, float(evidence.max()))
            rescale = np.exp(log_scale - new_scale)
            batch_weights = np.exp(evidence - new_scale)
            mass = mass * rescale + batch_weights.sum()
            first = first * rescale + batch_weights @ mean
            second = second * rescale + np.einsum("n,nij->ij", batch_weights,
                       covariance + mean[:, :, None] * mean[:, None, :])
            log_scale = new_scale
        logs = np.concatenate(logs)
        # Stable streaming accumulation retains no bank of 15x15 covariances.
        weights = np.exp(logs - logsumexp(logs))
        mean_physics = first / mass
        covariance_physics = second / mass - np.outer(mean_physics, mean_physics)
        rng = np.random.default_rng(seed + 1)
        chosen = rng.choice(len(sigma), size=draws, p=weights)
        mean, covariance, _ = self.conditional(sigma[chosen])
        physics = mean + np.einsum("nij,nj->ni", np.linalg.cholesky(covariance), rng.normal(size=mean.shape))
        columns = {name: physics[:, i] for i, name in enumerate(self.names)}
        columns.update({name: sigma[chosen, i] for i, name in enumerate(self.problem.prior.noise_names)})
        joint = np.column_stack([columns[name] for name in self.problem.parameter_order])
        return {"joint": joint, "sigma_nodes": sigma, "weights": weights,
                "physics_mean": mean_physics, "physics_covariance": covariance_physics,
                "log_evidence": float(logsumexp(logs) - math.log(len(logs))),
                "effective_nodes": float(1 / np.square(weights).sum()), "power": power, "seed": seed,
                "noise_mean": weights @ sigma,
                "noise_second_moment": weights @ np.square(sigma),
                "proposal": "reference_only_residual_scatter_importance_proposal_with_exact_piecewise_density_correction_and_20_percent_uniform_support",
                "proposal_residual_dof": self.residual_dof}
