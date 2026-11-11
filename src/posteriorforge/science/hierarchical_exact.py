"""Exact heterogeneous linear-Gaussian hierarchy for backend validation.

This fixture has the same graph used by a general calibration problem: shared
population quantities, heterogeneous local parameter vectors, multi-point and
scalar observations, and conditionally independent data sets.  Its fixed
Gaussian scales make the complete posterior analytically available, providing
an inference-engine check that is independent of TMCMC and HNPE.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np

from .hierarchical_problem import (
    HierarchicalCoordinate,
    HierarchicalDataset,
    HierarchicalObservations,
    HierarchicalProblemSpec,
    PopulationFamily,
)


LINEAR_GAUSSIAN_PRIOR_SCHEMA = "bayesarena.inference.linear_gaussian_hierarchy_prior.v1"
LINEAR_GAUSSIAN_OBSERVATION_SCHEMA = (
    "bayesarena.inference.linear_gaussian_hierarchy_observation.v1"
)
EXACT_HETEROGENEOUS_FIXTURE_SCHEMA = (
    "bayesarena.inference.exact_heterogeneous_hierarchy_fixture.v1"
)
_LOG_2PI = math.log(2.0 * math.pi)


def _names(values: tuple[str, ...], *, field: str) -> tuple[str, ...]:
    result = tuple(str(value).strip() for value in values)
    if not result or any(not value for value in result) or len(set(result)) != len(result):
        raise ValueError(f"{field} must contain unique non-empty names")
    return result


def _vector(value: Any, size: int, *, field: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{field} must be a finite vector of length {size}")
    return np.frombuffer(result.tobytes(order="C"), dtype=np.float64).reshape(result.shape)


def _matrix(value: Any, shape: tuple[int, int], *, field: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{field} must be a finite matrix with shape {shape}")
    return np.frombuffer(result.tobytes(order="C"), dtype=np.float64).reshape(result.shape)


def _covariance(value: Any, size: int, *, field: str) -> np.ndarray:
    result = _matrix(value, (size, size), field=field)
    if not np.allclose(result, result.T, rtol=0.0, atol=1e-13):
        raise ValueError(f"{field} must be symmetric")
    try:
        np.linalg.cholesky(result)
    except np.linalg.LinAlgError as error:
        raise ValueError(f"{field} must be positive definite") from error
    return result


def _rows(parameters: Any, dimension: int) -> np.ndarray:
    result = np.asarray(parameters, dtype=np.float64)
    if result.ndim == 1:
        result = result[None, :]
    if result.ndim != 2 or result.shape[1] != dimension or not np.isfinite(result).all():
        raise ValueError(f"parameters must be finite with shape [N,{dimension}]")
    return result


def _mvn_logpdf(residual: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    chol = np.linalg.cholesky(covariance)
    whitened = np.linalg.solve(chol, residual.T).T
    log_determinant = 2.0 * np.log(np.diag(chol)).sum()
    return -0.5 * (
        residual.shape[1] * _LOG_2PI
        + log_determinant
        + np.square(whitened).sum(axis=1)
    )


@dataclass(frozen=True)
class GaussianConditionalBlock:
    """Local Gaussian ``theta_i | h ~ N(B_i h + b_i, Q_i)``."""

    dataset_id: str
    local_names: tuple[str, ...]
    population_map: np.ndarray
    offset: np.ndarray
    covariance: np.ndarray

    def __post_init__(self) -> None:
        dataset_id = str(self.dataset_id).strip()
        if not dataset_id:
            raise ValueError("conditional block requires a dataset id")
        object.__setattr__(self, "dataset_id", dataset_id)
        names = _names(self.local_names, field=f"local names for {dataset_id}")
        object.__setattr__(self, "local_names", names)
        raw_map = np.asarray(self.population_map, dtype=np.float64)
        if raw_map.ndim != 2 or raw_map.shape[0] != len(names):
            raise ValueError("population_map rows must match local_names")
        object.__setattr__(
            self, "population_map",
            _matrix(raw_map, raw_map.shape, field=f"population map for {dataset_id}"),
        )
        object.__setattr__(
            self, "offset", _vector(self.offset, len(names), field=f"offset for {dataset_id}")
        )
        object.__setattr__(
            self, "covariance",
            _covariance(self.covariance, len(names), field=f"covariance for {dataset_id}"),
        )

    def manifest(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "local_names": list(self.local_names),
            "population_map": self.population_map.tolist(),
            "offset": self.offset.tolist(),
            "covariance": self.covariance.tolist(),
        }


@dataclass(frozen=True)
class GaussianObservationBlock:
    """Known linear observation ``y_i | theta_i ~ N(A_i theta_i+c_i,R_i)``."""

    dataset_id: str
    local_names: tuple[str, ...]
    design: np.ndarray
    offset: np.ndarray
    covariance: np.ndarray

    def __post_init__(self) -> None:
        dataset_id = str(self.dataset_id).strip()
        if not dataset_id:
            raise ValueError("observation block requires a dataset id")
        object.__setattr__(self, "dataset_id", dataset_id)
        names = _names(self.local_names, field=f"observation locals for {dataset_id}")
        object.__setattr__(self, "local_names", names)
        raw_design = np.asarray(self.design, dtype=np.float64)
        if raw_design.ndim != 2 or raw_design.shape[1] != len(names) or raw_design.shape[0] < 1:
            raise ValueError("design must have one column per local coordinate")
        points = raw_design.shape[0]
        object.__setattr__(
            self, "design", _matrix(raw_design, raw_design.shape, field=f"design for {dataset_id}")
        )
        object.__setattr__(
            self, "offset", _vector(self.offset, points, field=f"observation offset for {dataset_id}")
        )
        object.__setattr__(
            self, "covariance",
            _covariance(self.covariance, points, field=f"observation covariance for {dataset_id}"),
        )

    @property
    def observation_size(self) -> int:
        return self.design.shape[0]

    def manifest(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "local_names": list(self.local_names),
            "design": self.design.tolist(),
            "offset": self.offset.tolist(),
            "covariance": self.covariance.tolist(),
        }


@dataclass(frozen=True)
class LinearGaussianHierarchyPrior:
    global_names: tuple[str, ...]
    global_mean: np.ndarray
    global_covariance: np.ndarray
    conditional_blocks: tuple[GaussianConditionalBlock, ...]

    def __post_init__(self) -> None:
        names = _names(self.global_names, field="global names")
        object.__setattr__(self, "global_names", names)
        object.__setattr__(
            self, "global_mean", _vector(self.global_mean, len(names), field="global mean")
        )
        object.__setattr__(
            self, "global_covariance",
            _covariance(self.global_covariance, len(names), field="global covariance"),
        )
        blocks = tuple(self.conditional_blocks)
        if not blocks or len({block.dataset_id for block in blocks}) != len(blocks):
            raise ValueError("conditional blocks require unique datasets")
        if any(block.population_map.shape[1] != len(names) for block in blocks):
            raise ValueError("every population map must use the complete global vector")
        local_names = tuple(name for block in blocks for name in block.local_names)
        if len(local_names) != len(set(local_names)) or set(local_names).intersection(names):
            raise ValueError("global and local coordinate names must be unique across blocks")
        object.__setattr__(self, "conditional_blocks", blocks)

    @property
    def parameter_order(self) -> tuple[str, ...]:
        return self.global_names + tuple(
            name for block in self.conditional_blocks for name in block.local_names
        )

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": LINEAR_GAUSSIAN_PRIOR_SCHEMA,
            "parameter_order": list(self.parameter_order),
            "global_names": list(self.global_names),
            "global_mean": self.global_mean.tolist(),
            "global_covariance": self.global_covariance.tolist(),
            "conditional_blocks": [block.manifest() for block in self.conditional_blocks],
        }

    def sample(self, size: int, rng: np.random.Generator) -> np.ndarray:
        if type(size) is not int or size < 1 or not isinstance(rng, np.random.Generator):
            raise ValueError("sample requires a positive size and numpy Generator")
        global_values = rng.multivariate_normal(
            self.global_mean, self.global_covariance, size=size
        )
        pieces = [global_values]
        for block in self.conditional_blocks:
            mean = global_values @ block.population_map.T + block.offset
            noise = rng.multivariate_normal(
                np.zeros(len(block.local_names)), block.covariance, size=size
            )
            pieces.append(mean + noise)
        return np.concatenate(pieces, axis=1)

    def log_prob(self, parameters: np.ndarray) -> np.ndarray:
        values = _rows(parameters, len(self.parameter_order))
        global_dimension = len(self.global_names)
        global_values = values[:, :global_dimension]
        result = self.global_log_prob(global_values)
        start = global_dimension
        for block in self.conditional_blocks:
            stop = start + len(block.local_names)
            local = values[:, start:stop]
            result += self.conditional_log_prob(block.dataset_id, global_values, local)
            start = stop
        return result

    @property
    def global_parameter_order(self) -> tuple[str, ...]:
        return self.global_names

    def conditional_parameter_order(self, dataset_id: str) -> tuple[str, ...]:
        return self._block(dataset_id).local_names

    def _block(self, dataset_id: str) -> GaussianConditionalBlock:
        for block in self.conditional_blocks:
            if block.dataset_id == dataset_id:
                return block
        raise KeyError(f"unknown prior dataset: {dataset_id}")

    def global_log_prob(self, global_parameters: Any) -> np.ndarray:
        values = _rows(global_parameters, len(self.global_names))
        return _mvn_logpdf(values - self.global_mean, self.global_covariance)

    def conditional_log_prob(
        self, dataset_id: str, global_parameters: Any, local_parameters: Any
    ) -> np.ndarray:
        block = self._block(dataset_id)
        global_values = _rows(global_parameters, len(self.global_names))
        local_values = _rows(local_parameters, len(block.local_names))
        if len(global_values) != len(local_values) and min(len(global_values), len(local_values)) != 1:
            raise ValueError("global/local prior batches must match or have one broadcast row")
        mean = global_values @ block.population_map.T + block.offset
        return _mvn_logpdf(local_values - mean, block.covariance)


@dataclass(frozen=True)
class LinearGaussianObservationModel:
    parameter_names: tuple[str, ...]
    blocks: tuple[GaussianObservationBlock, ...]

    def __post_init__(self) -> None:
        names = _names(self.parameter_names, field="observation-model parameter names")
        object.__setattr__(self, "parameter_names", names)
        blocks = tuple(self.blocks)
        if not blocks or len({block.dataset_id for block in blocks}) != len(blocks):
            raise ValueError("observation blocks require unique datasets")
        known = set(names)
        if any(not set(block.local_names).issubset(known) for block in blocks):
            raise ValueError("observation block refers to an unknown coordinate")
        object.__setattr__(self, "blocks", blocks)

    @property
    def parameter_order(self) -> tuple[str, ...]:
        return self.parameter_names

    @property
    def dataset_ids(self) -> tuple[str, ...]:
        return tuple(block.dataset_id for block in self.blocks)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": LINEAR_GAUSSIAN_OBSERVATION_SCHEMA,
            "parameter_order": list(self.parameter_order),
            "dataset_ids": list(self.dataset_ids),
            "blocks": [block.manifest() for block in self.blocks],
        }

    def _indices(self, block: GaussianObservationBlock) -> tuple[int, ...]:
        positions = {name: index for index, name in enumerate(self.parameter_names)}
        return tuple(positions[name] for name in block.local_names)

    def predict(self, parameters: np.ndarray) -> Mapping[str, np.ndarray]:
        values = _rows(parameters, len(self.parameter_names))
        return {
            block.dataset_id: values[:, self._indices(block)] @ block.design.T + block.offset
            for block in self.blocks
        }

    def sample(
        self, parameters: np.ndarray, rng: np.random.Generator
    ) -> HierarchicalObservations:
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be numpy.random.Generator")
        predictions = self.predict(parameters)
        sampled = {}
        for block in self.blocks:
            noise = rng.multivariate_normal(
                np.zeros(block.observation_size),
                block.covariance,
                size=len(predictions[block.dataset_id]),
            )
            sampled[block.dataset_id] = predictions[block.dataset_id] + noise
        return HierarchicalObservations(sampled)

    def log_prob(
        self, parameters: np.ndarray, observations: HierarchicalObservations
    ) -> np.ndarray:
        values = _rows(parameters, len(self.parameter_names))
        result = np.zeros(len(values), dtype=np.float64)
        if tuple(observations.values) != self.dataset_ids:
            raise ValueError("observation blocks differ from the model")
        for block in self.blocks:
            observed = observations.values[block.dataset_id]
            if observed.shape[1] != block.observation_size or len(observed) not in (1, len(values)):
                raise ValueError(f"invalid observation shape for {block.dataset_id}")
            local_values = {name: values[:, self.parameter_names.index(name)] for name in block.local_names}
            result += self.local_log_prob(block.dataset_id, local_values, observed)
        return result

    def local_log_prob(
        self, dataset_id: str, local_parameters: Mapping[str, np.ndarray], observed: np.ndarray
    ) -> np.ndarray:
        for block in self.blocks:
            if block.dataset_id == dataset_id:
                values = np.column_stack([local_parameters[name] for name in block.local_names])
                values = _rows(values, len(block.local_names))
                data = np.asarray(observed, dtype=np.float64)
                if data.ndim != 2 or data.shape[1] != block.observation_size or len(data) not in (1, len(values)):
                    raise ValueError(f"invalid observation shape for {dataset_id}")
                prediction = values @ block.design.T + block.offset
                return _mvn_logpdf(data - prediction, block.covariance)
        raise KeyError(f"unknown observation dataset: {dataset_id}")


@dataclass(frozen=True)
class GaussianPosterior:
    parameter_order: tuple[str, ...]
    mean: np.ndarray
    covariance: np.ndarray

    def __post_init__(self) -> None:
        names = _names(self.parameter_order, field="posterior parameter order")
        object.__setattr__(self, "parameter_order", names)
        object.__setattr__(self, "mean", _vector(self.mean, len(names), field="posterior mean"))
        object.__setattr__(
            self, "covariance",
            _covariance(self.covariance, len(names), field="posterior covariance"),
        )

    def log_prob(self, parameters: Any) -> np.ndarray:
        values = _rows(parameters, len(self.parameter_order))
        return _mvn_logpdf(values - self.mean, self.covariance)

    def sample(self, size: int, rng: np.random.Generator) -> np.ndarray:
        if type(size) is not int or size < 1 or not isinstance(rng, np.random.Generator):
            raise ValueError("sample requires a positive size and numpy Generator")
        return rng.multivariate_normal(self.mean, self.covariance, size=size)


def exact_linear_gaussian_posterior(
    problem: HierarchicalProblemSpec,
    observations: HierarchicalObservations,
) -> GaussianPosterior:
    """Derive the complete posterior from the declared matrices.

    This path does not call the problem's log-density functions.  It assembles
    the Gaussian natural parameters directly, providing an independent check
    of the canonical prior and likelihood implementation.
    """

    if not isinstance(problem.prior, LinearGaussianHierarchyPrior):
        raise TypeError("exact posterior requires LinearGaussianHierarchyPrior")
    if not isinstance(problem.observation_model, LinearGaussianObservationModel):
        raise TypeError("exact posterior requires LinearGaussianObservationModel")
    if any(coordinate.lower is not None for coordinate in problem.coordinates):
        raise ValueError("the exact Gaussian posterior requires unbounded coordinates")
    if any(family.boundary_semantics != "unbounded" for family in problem.families):
        raise ValueError("the exact Gaussian posterior requires unbounded population families")
    problem.validate_observations(observations)
    if observations.batch_size != 1:
        raise ValueError("exact posterior accepts one observed hierarchical world")
    prior = problem.prior
    model = problem.observation_model
    dimension = problem.dimension
    precision = np.zeros((dimension, dimension), dtype=np.float64)
    natural = np.zeros(dimension, dtype=np.float64)
    global_dimension = len(prior.global_names)
    global_slice = slice(0, global_dimension)
    global_precision = np.linalg.inv(prior.global_covariance)
    precision[global_slice, global_slice] += global_precision
    natural[global_slice] += global_precision @ prior.global_mean
    positions = {name: index for index, name in enumerate(problem.parameter_order)}
    conditional_by_dataset = {block.dataset_id: block for block in prior.conditional_blocks}
    observation_by_dataset = {block.dataset_id: block for block in model.blocks}
    if set(conditional_by_dataset) != set(problem.dataset_ids):
        raise ValueError("conditional prior blocks must match the declared datasets")
    if prior.global_names != tuple(
        coordinate.name for coordinate in problem.coordinates if coordinate.scope == "global"
    ):
        raise ValueError("Gaussian global names must match declared global coordinates")
    for dataset in problem.datasets:
        conditional = conditional_by_dataset[dataset.dataset_id]
        if conditional.local_names != dataset.local_coordinates:
            raise ValueError("conditional local names must match the dataset contract")
        local_indices = np.asarray([positions[name] for name in conditional.local_names])
        local_precision = np.linalg.inv(conditional.covariance)
        population_map = conditional.population_map
        precision[global_slice, global_slice] += population_map.T @ local_precision @ population_map
        cross = -population_map.T @ local_precision
        precision[np.ix_(range(global_dimension), local_indices)] += cross
        precision[np.ix_(local_indices, range(global_dimension))] += cross.T
        precision[np.ix_(local_indices, local_indices)] += local_precision
        natural[global_slice] += -population_map.T @ local_precision @ conditional.offset
        natural[local_indices] += local_precision @ conditional.offset

        observed = observations.values[dataset.dataset_id][0]
        observation = observation_by_dataset[dataset.dataset_id]
        if not set(observation.local_names).issubset(dataset.local_coordinates):
            raise ValueError("an observation block may only use its own local coordinates")
        if observation.observation_size != dataset.observation_size:
            raise ValueError("observation block size must match the dataset contract")
        observation_indices = np.asarray([positions[name] for name in observation.local_names])
        observation_precision = np.linalg.inv(observation.covariance)
        design = observation.design
        precision[np.ix_(observation_indices, observation_indices)] += (
            design.T @ observation_precision @ design
        )
        natural[observation_indices] += (
            design.T @ observation_precision @ (observed - observation.offset)
        )
    covariance = np.linalg.inv(precision)
    covariance = 0.5 * (covariance + covariance.T)
    mean = np.linalg.solve(precision, natural)
    return GaussianPosterior(problem.parameter_order, mean, covariance)


def _mechanical_design(controls: np.ndarray, case: int) -> np.ndarray:
    # a and b are intentionally similar, producing the ridge that a scalar
    # observation of a can resolve through population pooling.
    return np.column_stack(
        (
            0.95 + (0.20 + 0.01 * case) * controls,
            0.90 + (0.18 - 0.01 * case) * controls,
            np.ones_like(controls),
        )
    )


def build_exact_heterogeneous_problem() -> HierarchicalProblemSpec:
    """Return the six-dataset exact fixture used to validate generic backends."""

    global_names = ("mu_a", "mu_b", "mu_d0")
    dataset_ids = (
        "mechanical_1", "mechanical_2", "mechanical_3",
        "acoustic_1", "acoustic_2", "acoustic_3",
    )
    mechanical_names = tuple(
        (f"a_m{index}", f"b_m{index}", f"d0_m{index}") for index in range(1, 4)
    )
    acoustic_names = tuple((f"a_a{index}",) for index in range(1, 4))
    local_names = mechanical_names + acoustic_names
    coordinates = [
        HierarchicalCoordinate(name, "scaled", "global") for name in global_names
    ]
    for dataset_id, names in zip(dataset_ids, local_names):
        for name in names:
            family = "a" if name.startswith("a_") else "b" if name.startswith("b_") else "d0"
            coordinates.append(
                HierarchicalCoordinate(name, "scaled", "local", dataset_id, family)
            )
    a_members = tuple(names[0] for names in local_names)
    b_members = tuple(names[1] for names in mechanical_names)
    d0_members = tuple(names[2] for names in mechanical_names)
    families = (
        PopulationFamily("a", "mu_a", a_members, fixed_scale=0.45),
        PopulationFamily("b", "mu_b", b_members, fixed_scale=0.70),
        PopulationFamily("d0", "mu_d0", d0_members, fixed_scale=0.25),
    )
    conditional_blocks = []
    for dataset_id, names in zip(dataset_ids[:3], mechanical_names):
        conditional_blocks.append(
            GaussianConditionalBlock(
                dataset_id, names, np.eye(3), np.zeros(3),
                np.diag(np.square((0.45, 0.70, 0.25))),
            )
        )
    for dataset_id, names in zip(dataset_ids[3:], acoustic_names):
        conditional_blocks.append(
            GaussianConditionalBlock(
                dataset_id, names, np.asarray(((1.0, 0.0, 0.0),)), np.zeros(1),
                np.asarray(((0.45**2,),)),
            )
        )
    prior = LinearGaussianHierarchyPrior(
        global_names,
        np.asarray((0.30, -0.20, 0.10)),
        np.asarray(((1.00, 0.12, 0.02), (0.12, 1.20, -0.03), (0.02, -0.03, 0.40))),
        tuple(conditional_blocks),
    )
    datasets = []
    observation_blocks = []
    point_counts = (5, 4, 3)
    for case, (dataset_id, names, points) in enumerate(
        zip(dataset_ids[:3], mechanical_names, point_counts), start=1
    ):
        controls = np.linspace(0.0, 1.0, points)
        design = _mechanical_design(controls, case)
        datasets.append(
            HierarchicalDataset(
                dataset_id, "curve", names, points, tuple(controls), "control", "response"
            )
        )
        observation_blocks.append(
            GaussianObservationBlock(
                dataset_id, names, design, np.full(points, 4.0 + 0.1 * case),
                np.eye(points) * 0.12**2,
            )
        )
    for case, (dataset_id, names) in enumerate(zip(dataset_ids[3:], acoustic_names), start=1):
        coefficient = 1.25 + 0.15 * case
        datasets.append(
            HierarchicalDataset(
                dataset_id, "scalar", names, 1, (2.0 + 0.5 * case,), "setting", "response"
            )
        )
        observation_blocks.append(
            GaussianObservationBlock(
                dataset_id, names, np.asarray(((coefficient,),)),
                np.asarray((3.0 + 0.1 * case,)), np.asarray(((0.08**2,),)),
            )
        )
    model = LinearGaussianObservationModel(prior.parameter_order, tuple(observation_blocks))
    return HierarchicalProblemSpec(
        name="exact_heterogeneous_six_dataset_hierarchy",
        version=EXACT_HETEROGENEOUS_FIXTURE_SCHEMA,
        coordinates=tuple(coordinates),
        families=families,
        datasets=tuple(datasets),
        prior=prior,
        observation_model=model,
    )


def draw_exact_fixture_case(
    *, seed: int = 20260930,
) -> tuple[HierarchicalProblemSpec, np.ndarray, HierarchicalObservations]:
    """Draw one deterministic prior-predictive world for matched backend tests."""

    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    problem = build_exact_heterogeneous_problem()
    seed_sequence = np.random.SeedSequence(seed)
    parameter_seed, observation_seed = seed_sequence.spawn(2)
    truth = problem.sample_prior(1, np.random.default_rng(parameter_seed))
    observations = problem.simulate(truth, np.random.default_rng(observation_seed))
    return problem, truth, observations


__all__ = [
    "EXACT_HETEROGENEOUS_FIXTURE_SCHEMA",
    "GaussianConditionalBlock",
    "GaussianObservationBlock",
    "GaussianPosterior",
    "LinearGaussianHierarchyPrior",
    "LinearGaussianObservationModel",
    "build_exact_heterogeneous_problem",
    "draw_exact_fixture_case",
    "exact_linear_gaussian_posterior",
]
