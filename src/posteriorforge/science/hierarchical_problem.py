"""Backend-neutral contracts for heterogeneous hierarchical inference.

The scientific model is declared once and consumed by inference backends.  A
problem owns a flat, deterministic parameter order while retaining the
population-family and per-dataset structure needed by Korali and HNPE.
Neither backend is permitted to replace the declared prior or observation
model with a private copy.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np


HIERARCHICAL_PROBLEM_SCHEMA = "bayesarena.inference.hierarchical_problem.v1"


def _name(value: str, *, field: str) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{field} must be non-empty")
    return result


def _json_value(value: Any) -> Any:
    """Convert a scientific manifest to strict, stable JSON-compatible data."""

    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("scientific manifests may not contain NaN or infinity")
        return value
    raise TypeError(f"unsupported scientific-manifest value: {type(value).__name__}")


def _fingerprint(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        _json_value(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class HierarchicalCoordinate:
    """One reported or latent coordinate in the canonical flat state."""

    name: str
    unit: str
    scope: str
    dataset_id: str | None = None
    family: str | None = None
    lower: float | None = None
    upper: float | None = None
    reported: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _name(self.name, field="coordinate name"))
        object.__setattr__(self, "unit", _name(self.unit, field=f"unit for {self.name}"))
        if self.scope not in ("global", "local"):
            raise ValueError(f"coordinate {self.name} scope must be global or local")
        if self.scope == "global" and self.dataset_id is not None:
            raise ValueError(f"global coordinate {self.name} may not name a dataset")
        if self.scope == "local" and self.dataset_id is None:
            raise ValueError(f"local coordinate {self.name} requires a dataset")
        if self.dataset_id is not None:
            object.__setattr__(self, "dataset_id", _name(self.dataset_id, field="dataset id"))
        if self.family is not None:
            object.__setattr__(self, "family", _name(self.family, field="family"))
        if (self.lower is None) != (self.upper is None):
            raise ValueError(f"coordinate {self.name} must declare both bounds or neither")
        if self.lower is not None:
            lower, upper = float(self.lower), float(self.upper)
            if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
                raise ValueError(f"coordinate {self.name} bounds must be finite and increasing")
            object.__setattr__(self, "lower", lower)
            object.__setattr__(self, "upper", upper)
        if type(self.reported) is not bool:
            raise TypeError("reported must be Boolean")

    def contains(self, value: np.ndarray) -> np.ndarray:
        result = np.isfinite(value)
        if self.lower is not None:
            result &= value >= self.lower
            result &= value <= self.upper
        return result

    def manifest(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "unit": self.unit,
            "scope": self.scope,
            "dataset_id": self.dataset_id,
            "family": self.family,
            "bounds": None if self.lower is None else [self.lower, self.upper],
            "density_transform": "identity" if self.lower is None else "logit",
            "reported": self.reported,
        }


@dataclass(frozen=True)
class PopulationFamily:
    """Population relationship shared by a declared subset of local coordinates."""

    name: str
    location_coordinate: str
    member_coordinates: tuple[str, ...]
    scale_coordinate: str | None = None
    fixed_scale: float | None = None
    distribution: str = "normal"
    boundary_semantics: str = "unbounded"

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _name(self.name, field="population family"))
        object.__setattr__(
            self, "location_coordinate",
            _name(self.location_coordinate, field=f"location coordinate for {self.name}"),
        )
        members = tuple(_name(value, field=f"member of {self.name}") for value in self.member_coordinates)
        if not members or len(set(members)) != len(members):
            raise ValueError(f"population family {self.name} requires unique members")
        object.__setattr__(self, "member_coordinates", members)
        if (self.scale_coordinate is None) == (self.fixed_scale is None):
            raise ValueError(
                f"population family {self.name} requires exactly one of scale_coordinate or fixed_scale"
            )
        if self.scale_coordinate is not None:
            object.__setattr__(
                self, "scale_coordinate",
                _name(self.scale_coordinate, field=f"scale coordinate for {self.name}"),
            )
        else:
            scale = float(self.fixed_scale)
            if not math.isfinite(scale) or scale <= 0.0:
                raise ValueError(f"fixed scale for {self.name} must be positive")
            object.__setattr__(self, "fixed_scale", scale)
        object.__setattr__(self, "distribution", _name(self.distribution, field="distribution"))
        object.__setattr__(
            self, "boundary_semantics",
            _name(self.boundary_semantics, field="boundary semantics"),
        )

    def manifest(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "location_coordinate": self.location_coordinate,
            "scale_coordinate": self.scale_coordinate,
            "fixed_scale": self.fixed_scale,
            "member_coordinates": list(self.member_coordinates),
            "distribution": self.distribution,
            "boundary_semantics": self.boundary_semantics,
        }


@dataclass(frozen=True)
class HierarchicalDataset:
    """Schema for one conditionally independent observation block."""

    dataset_id: str
    kind: str
    local_coordinates: tuple[str, ...]
    observation_size: int
    controls: tuple[float, ...]
    control_unit: str
    observation_unit: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_id", _name(self.dataset_id, field="dataset id"))
        object.__setattr__(self, "kind", _name(self.kind, field="dataset kind"))
        local = tuple(_name(value, field=f"local coordinate for {self.dataset_id}")
                      for value in self.local_coordinates)
        if not local or len(local) != len(set(local)):
            raise ValueError(f"dataset {self.dataset_id} requires unique local coordinates")
        object.__setattr__(self, "local_coordinates", local)
        if type(self.observation_size) is not int or self.observation_size < 1:
            raise ValueError("observation_size must be a positive integer")
        controls = tuple(float(value) for value in self.controls)
        if len(controls) != self.observation_size or not np.isfinite(controls).all():
            raise ValueError(
                f"dataset {self.dataset_id} controls must be finite and match observation_size"
            )
        object.__setattr__(self, "controls", controls)
        object.__setattr__(self, "control_unit", _name(self.control_unit, field="control unit"))
        object.__setattr__(
            self, "observation_unit", _name(self.observation_unit, field="observation unit")
        )

    def manifest(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "kind": self.kind,
            "local_coordinates": list(self.local_coordinates),
            "observation_size": self.observation_size,
            "controls": list(self.controls),
            "control_unit": self.control_unit,
            "observation_unit": self.observation_unit,
        }


@dataclass(frozen=True)
class HierarchicalObservations:
    """A batch of observations keyed by dataset id."""

    values: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        converted: dict[str, np.ndarray] = {}
        batch_size: int | None = None
        for key, raw in self.values.items():
            dataset_id = _name(key, field="observation dataset id")
            value = np.asarray(raw, dtype=np.float64)
            if value.ndim != 2 or value.shape[0] < 1 or value.shape[1] < 1:
                raise ValueError("each observation array must have shape [batch, points]")
            if not np.isfinite(value).all():
                raise ValueError("observations must be finite")
            if batch_size is None:
                batch_size = len(value)
            elif len(value) != batch_size:
                raise ValueError("all observation blocks must have the same batch size")
            # An immutable bytes owner prevents both caller aliasing and
            # re-enabling write access on the stored scientific observations.
            value = np.frombuffer(value.tobytes(order="C"), dtype=np.float64).reshape(value.shape)
            if dataset_id in converted:
                raise ValueError("observation dataset ids must be unique after stripping whitespace")
            converted[dataset_id] = value
        if not converted:
            raise ValueError("at least one observation block is required")
        object.__setattr__(self, "values", MappingProxyType(converted))

    @property
    def batch_size(self) -> int:
        return len(next(iter(self.values.values())))

    def manifest(self) -> dict[str, Any]:
        arrays = {}
        for dataset_id, value in self.values.items():
            arrays[dataset_id] = {
                "shape": list(value.shape),
                "dtype": value.dtype.str,
                "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
            }
        return {"batch_size": self.batch_size, "arrays": arrays}


@runtime_checkable
class HierarchicalPrior(Protocol):
    @property
    def parameter_order(self) -> tuple[str, ...]: ...

    def manifest(self) -> Mapping[str, Any]: ...

    def sample(self, size: int, rng: np.random.Generator) -> np.ndarray: ...

    def log_prob(self, parameters: np.ndarray) -> np.ndarray: ...


@runtime_checkable
class HierarchicalObservationModel(Protocol):
    @property
    def parameter_order(self) -> tuple[str, ...]: ...

    @property
    def dataset_ids(self) -> tuple[str, ...]: ...

    def manifest(self) -> Mapping[str, Any]: ...

    def predict(self, parameters: np.ndarray) -> Mapping[str, np.ndarray]: ...

    def sample(
        self, parameters: np.ndarray, rng: np.random.Generator
    ) -> HierarchicalObservations: ...

    def log_prob(
        self, parameters: np.ndarray, observations: HierarchicalObservations
    ) -> np.ndarray: ...


@dataclass(frozen=True)
class HierarchicalProblemSpec:
    """Single scientific source compiled by every inference backend."""

    name: str
    version: str
    coordinates: tuple[HierarchicalCoordinate, ...]
    families: tuple[PopulationFamily, ...]
    datasets: tuple[HierarchicalDataset, ...]
    prior: HierarchicalPrior
    observation_model: HierarchicalObservationModel

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _name(self.name, field="problem name"))
        object.__setattr__(self, "version", _name(self.version, field="problem version"))
        coordinates = tuple(self.coordinates)
        datasets = tuple(self.datasets)
        families = tuple(self.families)
        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "datasets", datasets)
        object.__setattr__(self, "families", families)
        if not coordinates or not datasets:
            raise ValueError("a hierarchical problem requires coordinates and datasets")
        coordinate_names = tuple(value.name for value in coordinates)
        dataset_ids = tuple(value.dataset_id for value in datasets)
        family_names = tuple(value.name for value in families)
        for label, values in (
            ("coordinate", coordinate_names), ("dataset", dataset_ids), ("family", family_names)
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{label} names must be unique")
        global_names = {value.name for value in coordinates if value.scope == "global"}
        local_by_dataset = {
            dataset_id: tuple(
                value.name for value in coordinates
                if value.scope == "local" and value.dataset_id == dataset_id
            )
            for dataset_id in dataset_ids
        }
        unknown_dataset = [
            value.name for value in coordinates
            if value.scope == "local" and value.dataset_id not in set(dataset_ids)
        ]
        if unknown_dataset:
            raise ValueError(f"local coordinates reference unknown datasets: {unknown_dataset}")
        for dataset in datasets:
            if dataset.local_coordinates != local_by_dataset[dataset.dataset_id]:
                raise ValueError(
                    f"dataset {dataset.dataset_id} local order differs from the canonical coordinate order"
                )
        assigned_members: set[str] = set()
        coordinates_by_name = {value.name: value for value in coordinates}
        for family in families:
            if family.location_coordinate not in global_names:
                raise ValueError(f"family {family.name} location is not a global coordinate")
            if family.scale_coordinate is not None and family.scale_coordinate not in global_names:
                raise ValueError(f"family {family.name} scale is not a global coordinate")
            for member in family.member_coordinates:
                if member not in coordinates_by_name or coordinates_by_name[member].scope != "local":
                    raise ValueError(f"family {family.name} member {member} is not a local coordinate")
                if coordinates_by_name[member].family != family.name:
                    raise ValueError(f"coordinate {member} does not declare family {family.name}")
                if member in assigned_members:
                    raise ValueError(f"local coordinate {member} belongs to multiple families")
                assigned_members.add(member)
        for coordinate in coordinates:
            if coordinate.scope == "local" and coordinate.family is not None:
                if coordinate.family not in set(family_names) or coordinate.name not in assigned_members:
                    raise ValueError(f"local coordinate {coordinate.name} has an unresolved family")
        if not isinstance(self.prior, HierarchicalPrior):
            raise TypeError("prior does not satisfy HierarchicalPrior")
        if not isinstance(self.observation_model, HierarchicalObservationModel):
            raise TypeError("observation_model does not satisfy HierarchicalObservationModel")
        if tuple(self.prior.parameter_order) != coordinate_names:
            raise ValueError("prior parameter order differs from the canonical coordinate order")
        if tuple(self.observation_model.parameter_order) != coordinate_names:
            raise ValueError("observation-model parameter order differs from the canonical coordinate order")
        if tuple(self.observation_model.dataset_ids) != dataset_ids:
            raise ValueError("observation-model dataset order differs from the canonical dataset order")
        # Force strict-JSON validation while constructing the immutable contract.
        _json_value(self.prior.manifest())
        _json_value(self.observation_model.manifest())

    @property
    def parameter_order(self) -> tuple[str, ...]:
        return tuple(value.name for value in self.coordinates)

    @property
    def dataset_ids(self) -> tuple[str, ...]:
        return tuple(value.dataset_id for value in self.datasets)

    @property
    def dimension(self) -> int:
        return len(self.coordinates)

    def _parameters(self, values: Any) -> np.ndarray:
        result = np.asarray(values, dtype=np.float64)
        if result.ndim == 1:
            result = result[None, :]
        if result.ndim != 2 or result.shape[1] != self.dimension:
            raise ValueError(f"parameters must have shape [N,{self.dimension}]")
        if not np.isfinite(result).all():
            raise ValueError("parameters must be finite")
        return result

    def contains(self, values: Any) -> np.ndarray:
        parameters = self._parameters(values)
        result = np.ones(len(parameters), dtype=np.bool_)
        for index, coordinate in enumerate(self.coordinates):
            result &= coordinate.contains(parameters[:, index])
        return result

    def to_unconstrained(self, parameters: Any) -> np.ndarray:
        """Map physical coordinates to identity/logit density coordinates.

        Bounds are not clipped: a point on a finite boundary has no finite
        logit, and accepting it by clipping would change the density target.
        """

        values = self._parameters(parameters).copy()
        for index, coordinate in enumerate(self.coordinates):
            if coordinate.lower is not None:
                point = values[:, index]
                if np.any((point <= coordinate.lower) | (point >= coordinate.upper)):
                    raise ValueError(f"{coordinate.name} must be strictly inside its bounds")
                values[:, index] = np.log(point - coordinate.lower) - np.log(coordinate.upper - point)
        return values

    def from_unconstrained(self, parameters: Any) -> np.ndarray:
        """Map identity/logit coordinates back to the declared physical units."""

        values = self._parameters(parameters).copy()
        for index, coordinate in enumerate(self.coordinates):
            if coordinate.lower is not None:
                logistic = np.exp(-np.logaddexp(0.0, -values[:, index]))
                values[:, index] = coordinate.lower + (coordinate.upper - coordinate.lower) * logistic
        return values

    def log_abs_det_from_unconstrained(self, parameters: Any) -> np.ndarray:
        """Return log|dx/dz|, needed by a density expressed in logit space."""

        values = self._parameters(parameters)
        result = np.zeros(len(values), dtype=np.float64)
        for index, coordinate in enumerate(self.coordinates):
            if coordinate.lower is not None:
                z = values[:, index]
                result += (
                    math.log(coordinate.upper - coordinate.lower)
                    - np.logaddexp(0.0, -z) - np.logaddexp(0.0, z)
                )
        return result

    def validate_observations(self, observations: HierarchicalObservations) -> None:
        if not isinstance(observations, HierarchicalObservations):
            raise TypeError("observations must be HierarchicalObservations")
        if tuple(observations.values) != self.dataset_ids:
            raise ValueError("observation dataset order differs from the problem contract")
        for dataset in self.datasets:
            if observations.values[dataset.dataset_id].shape[1] != dataset.observation_size:
                raise ValueError(f"observation size mismatch for {dataset.dataset_id}")

    def sample_prior(self, size: int, rng: np.random.Generator) -> np.ndarray:
        if type(size) is not int or size < 1:
            raise ValueError("size must be a positive integer")
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be numpy.random.Generator")
        values = self._parameters(self.prior.sample(size, rng))
        if len(values) != size or not self.contains(values).all():
            raise ValueError("prior returned invalid or out-of-support samples")
        return values

    def simulate(self, parameters: Any, rng: np.random.Generator) -> HierarchicalObservations:
        values = self._parameters(parameters)
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be numpy.random.Generator")
        if not self.contains(values).all():
            raise ValueError("cannot simulate out-of-support parameters")
        observations = self.observation_model.sample(values, rng)
        self.validate_observations(observations)
        if observations.batch_size != len(values):
            raise ValueError("observation model returned the wrong batch size")
        return observations

    def log_prior(self, parameters: Any) -> np.ndarray:
        values = self._parameters(parameters)
        result = np.asarray(self.prior.log_prob(values), dtype=np.float64)
        if result.shape != (len(values),) or np.any(np.isnan(result) | np.isposinf(result)):
            raise ValueError("prior returned an invalid log-density vector")
        return np.where(self.contains(values), result, -np.inf)

    def log_likelihood(
        self, parameters: Any, observations: HierarchicalObservations
    ) -> np.ndarray:
        values = self._parameters(parameters)
        self.validate_observations(observations)
        if observations.batch_size not in (1, len(values)):
            raise ValueError("observation batch must have one row or match parameter rows")
        result = np.asarray(
            self.observation_model.log_prob(values, observations), dtype=np.float64
        )
        if result.shape != (len(values),) or np.any(np.isnan(result) | np.isposinf(result)):
            raise ValueError("observation model returned an invalid log-likelihood vector")
        return result

    def log_joint(
        self, parameters: Any, observations: HierarchicalObservations
    ) -> np.ndarray:
        values = self._parameters(parameters)
        return self.log_prior(values) + self.log_likelihood(values, observations)

    def scientific_manifest(self) -> dict[str, Any]:
        return {
            "schema": HIERARCHICAL_PROBLEM_SCHEMA,
            "name": self.name,
            "version": self.version,
            "parameter_order": list(self.parameter_order),
            "coordinates": [value.manifest() for value in self.coordinates],
            "families": [value.manifest() for value in self.families],
            "datasets": [value.manifest() for value in self.datasets],
            "prior": _json_value(self.prior.manifest()),
            "observation_model": _json_value(self.observation_model.manifest()),
        }

    @property
    def scientific_fingerprint(self) -> str:
        return _fingerprint(self.scientific_manifest())


__all__ = [
    "HIERARCHICAL_PROBLEM_SCHEMA",
    "HierarchicalCoordinate",
    "HierarchicalDataset",
    "HierarchicalObservationModel",
    "HierarchicalObservations",
    "HierarchicalPrior",
    "HierarchicalProblemSpec",
    "PopulationFamily",
]
