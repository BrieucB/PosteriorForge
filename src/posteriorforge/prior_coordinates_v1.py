"""Source-derived causal prior coordinates. Built with PriorLabs-TabPFN.

No posterior/reference fitting, clipping, row deletion, endpoint repair, or RNG.
Float64 physical arithmetic and model Float32 casting are distinct contracts.
"""
from __future__ import annotations
from functools import lru_cache
import hashlib
from pathlib import Path
from types import MappingProxyType
import numpy as np
from scipy.special import log_ndtr, ndtri_exp
from scipy.stats import truncnorm
from .science.analytical_benchmark_v2 import build_problem

SCHEMA = "posteriorforge.prior_coordinates.v1"
COORDINATE_ID = "causal_gaussian_hierarchy_truncnoise_normal_v1"
ORDER = (0,1,2,6,3,4,5,10,7,8,9,14,11,12,13,16,15,18,17,20,19)
INVERSE_ORDER = tuple(int(i) for i in np.argsort(ORDER))
PHYSICAL_ORDER = ("mu_a","mu_b","mu_d0","a_m1","b_m1","d0_m1","sigma_m1",
                  "a_m2","b_m2","d0_m2","sigma_m2","a_m3","b_m3","d0_m3","sigma_m3",
                  "a_a1","sigma_a1","a_a2","sigma_a2","a_a3","sigma_a3")
NOISE_INDICES = (6,10,14,16,18,20)
EPS64 = np.finfo(np.float64).eps
EPS32 = np.finfo(np.float32).eps
F64_PHYSICAL_UNITS = float(256 * EPS64)
F64_LATENT_ADMISSIBILITY = float(8 * np.sqrt(EPS64))
F32_PHYSICAL_UNITS = float(32 * EPS32)
F32_LATENT_ADMISSIBILITY = float(32 * EPS32)
PINNED_SOURCES = MappingProxyType({
    "science/analytical_benchmark_v2.py":"8d5f286f6d8512391dea2ae036313bec05782d28c7803f336490cbcdce3bd99f",
    "science/hierarchical_noise.py":"cbf0b8c956362a5eaa31923e9b94a3db3e3ede2072603add51e78105b8c0cbf0",
    "science/hierarchical_exact.py":"db001593cc7307b1290b981622705961c5953001a6c1245fae5ec779938e9687",
    "science/hierarchical_problem.py":"03aa2d8a3d279783cf6c599f1400f253f3db2e1e7eb027aff5b7cd1b48fe12d0",
    "execution_ss15_sigma_first.py":"d48d1638ef9e540bdd629d321fba657af4db757700b1092fc58fb9ded1f79ad2",
})

class CoordinateError(ValueError):
    """The entire coordinate qualification/request must fail; no row repair."""

def _require(value, message):
    if not value:
        raise CoordinateError(message)

def source_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

def require_sources():
    root = Path(__file__).parent
    for name, expected in PINNED_SOURCES.items():
        _require(hashlib.sha256((root/name).read_bytes()).hexdigest() == expected,
                 "frozen prior/order source changed: " + name)
    return dict(PINNED_SOURCES)

@lru_cache(maxsize=1)
def _geometry():
    require_sources()
    problem = build_problem()
    _require(tuple(problem.parameter_order) == PHYSICAL_ORDER, "physical parameter order changed")
    base = problem.prior.base
    mean = np.asarray(base.global_mean, dtype=np.float64)
    covariance = np.asarray(base.global_covariance, dtype=np.float64)
    global_factor = np.linalg.cholesky(covariance)
    centers = np.zeros(21, dtype=np.float64)
    units = np.zeros(21, dtype=np.float64)
    centers[:3] = mean
    units[:3] = np.sqrt(np.diag(covariance))
    blocks = []
    by_physical = {}
    for number, block in enumerate(base.conditional_blocks):
        indices = tuple(PHYSICAL_ORDER.index(name) for name in block.local_names)
        mapping = np.array(block.population_map, dtype=np.float64, copy=True)
        offset = np.array(block.offset, dtype=np.float64, copy=True)
        local_cov = np.array(block.covariance, dtype=np.float64, copy=True)
        factor = np.linalg.cholesky(local_cov)
        center = offset.copy()
        for k in range(3):
            center += mapping[:, k] * mean[k]
        centers[list(indices)] = center
        units[list(indices)] = np.sqrt(np.diag(local_cov))
        item = dict(dataset_id=block.dataset_id, indices=indices, mapping=mapping,
                    offset=offset, covariance=local_cov, factor=factor)
        blocks.append(item)
        for component, physical in enumerate(indices):
            by_physical[physical] = (item, component)
    low, high = problem.prior.bounds
    noise = {}
    for name, entry in zip(problem.prior.noise_names, problem.prior.noise_gaussian_priors):
        physical = PHYSICAL_ORDER.index(name)
        location, scale = entry
        a, b = (low-location)/scale, (high-location)/scale
        log_mass = np.log1p(-np.exp(log_ndtr(a)) - np.exp(log_ndtr(-b)))
        _require(np.isfinite(log_mass), "invalid source noise normalization")
        noise[physical] = dict(location=location, scale=scale, low=low, high=high,
                               a=a, b=b, log_mass=float(log_mass))
        centers[physical], units[physical] = location, scale
    _require(tuple(noise) == NOISE_INDICES and np.all(units > 0), "noise order or prior units changed")
    return dict(problem=problem, mean=mean, covariance=covariance, factor=global_factor,
                centers=centers, units=units, blocks=blocks, by_physical=by_physical, noise=noise)

def _rows(values, widths, dtypes, label):
    _require(isinstance(values, np.ndarray) and values.dtype in tuple(np.dtype(x) for x in dtypes),
             label + ": explicit original dtype required")
    _require(values.ndim == 2 and len(values) > 0 and values.shape[1] in widths,
             label + ": nonempty matrix with declared width required")
    _require(np.isfinite(values).all(), label + ": nonfinite input; no repair")
    return values

def _strict_noise(values, spec, label):
    _require(np.all((values > spec["low"]) & (values < spec["high"])),
             label + ": endpoint, out-of-support, or saturated noise; no clipping")

def _noise_encode(values, spec):
    _strict_noise(values, spec, "encode")
    lower = truncnorm.logcdf(values, spec["a"], spec["b"], loc=spec["location"], scale=spec["scale"])
    upper = truncnorm.logsf(values, spec["a"], spec["b"], loc=spec["location"], scale=spec["scale"])
    _require(np.isfinite(lower).all() and np.isfinite(upper).all()
             and np.all(lower < 0) and np.all(upper < 0),
             "prior CDF/log-survival saturation; no inward move")
    output = np.empty_like(values, dtype=np.float64)
    left = lower <= -np.log(2.)
    output[left] = ndtri_exp(lower[left])
    output[~left] = -ndtri_exp(upper[~left])
    _require(np.isfinite(output).all(), "nonfinite noise normal coordinate")
    return output

def _noise_decode(values, spec):
    _require(np.isfinite(values).all(), "nonfinite noise latent before CDF")
    output = np.empty_like(values, dtype=np.float64)
    left = values <= 0
    # Retain lower/upper tail logs instead of forming a rounded Phi(z)==1.
    lower = np.logaddexp(log_ndtr(spec["a"]), log_ndtr(values[left]) + spec["log_mass"])
    upper = np.logaddexp(log_ndtr(-spec["b"]), log_ndtr(-values[~left]) + spec["log_mass"])
    output[left] = spec["location"] + spec["scale"] * ndtri_exp(lower)
    output[~left] = spec["location"] - spec["scale"] * ndtri_exp(upper)
    _require(np.isfinite(output).all(), "nonfinite decoded noise")
    _strict_noise(output, spec, "decode")
    return output

def _local_mean(g, item, component):
    output = np.full(len(g), item["offset"][component], dtype=np.float64)
    for k in range(3):
        output += item["mapping"][component, k] * g[:, k]
    return output

def _encode_prefix(physical, geo):
    latent = np.empty_like(physical, dtype=np.float64)
    for stage, physical_index in enumerate(ORDER[:physical.shape[1]]):
        if physical_index < 3:
            residual = physical[:, stage] - geo["mean"][physical_index]
            for k in range(physical_index):
                residual -= geo["factor"][physical_index, k] * latent[:, k]
            latent[:, stage] = residual / geo["factor"][physical_index, physical_index]
        elif physical_index in geo["noise"]:
            latent[:, stage] = _noise_encode(physical[:, stage], geo["noise"][physical_index])
        else:
            item, component = geo["by_physical"][physical_index]
            residual = physical[:, stage] - _local_mean(physical[:, :3], item, component)
            for k in range(component):
                previous = INVERSE_ORDER[item["indices"][k]]
                _require(previous < stage, "noncausal local Cholesky ordering")
                residual -= item["factor"][component, k] * latent[:, previous]
            latent[:, stage] = residual / item["factor"][component, component]
    _require(np.isfinite(latent).all(), "nonfinite affine encoding")
    return np.ascontiguousarray(latent)

def _decode_prefix(latent, geo):
    physical = np.empty_like(latent, dtype=np.float64)
    for stage, physical_index in enumerate(ORDER[:latent.shape[1]]):
        if physical_index < 3:
            value = np.full(len(latent), geo["mean"][physical_index], dtype=np.float64)
            for k in range(physical_index + 1):
                value += geo["factor"][physical_index, k] * latent[:, k]
            physical[:, stage] = value
        elif physical_index in geo["noise"]:
            physical[:, stage] = _noise_decode(latent[:, stage], geo["noise"][physical_index])
        else:
            item, component = geo["by_physical"][physical_index]
            value = _local_mean(physical[:, :3], item, component)
            for k in range(component + 1):
                previous = INVERSE_ORDER[item["indices"][k]]
                _require(previous <= stage, "noncausal local decoding")
                value += item["factor"][component, k] * latent[:, previous]
            physical[:, stage] = value
    _require(np.isfinite(physical).all(), "nonfinite affine decoding; no row removal")
    return np.ascontiguousarray(physical)

def _physical_check(actual, expected, indices, coefficient, geo):
    units = geo["units"][list(indices)]
    centers = geo["centers"][list(indices)]
    error = np.abs(actual-expected) / units
    bound = coefficient * (1 + np.abs((expected-centers)/units))
    _require(np.all(error <= bound), "physical roundtrip exceeds declared prior-unit bound")

def _latent_check(actual, expected, coefficient):
    _require(np.all(np.abs(actual-expected) <= coefficient*(1+np.abs(expected))),
             "latent conditioning/roundtrip admissibility failed")

def encode_physical(values):
    """Original physical Float64[n,21] to Float64 latent physical-index slots."""
    require_sources()
    rows = _rows(values, (21,), (np.float64,), "encode_physical")
    geo = _geometry()
    stage = _encode_prefix(rows[:, ORDER], geo)
    decoded = _decode_prefix(stage, geo)
    _physical_check(decoded, rows[:, ORDER], ORDER, F64_PHYSICAL_UNITS, geo)
    _latent_check(_encode_prefix(decoded, geo), stage, F64_LATENT_ADMISSIBILITY)
    return np.ascontiguousarray(stage[:, INVERSE_ORDER])

def encode_for_model(values):
    """Exactly one final Float32 cast, checked separately from Float64 transport."""
    latent = encode_physical(values)
    model = np.ascontiguousarray(latent.astype(np.float32))
    _require(np.isfinite(model).all(), "latent model cast overflow")
    _latent_check(model.astype(np.float64), latent, F32_LATENT_ADMISSIBILITY)
    decoded = decode_ordered(model[:, ORDER])
    _physical_check(decoded, values, range(21), F32_PHYSICAL_UNITS, _geometry())
    return model

def order_latents(values):
    _rows(values, (21,), (np.float64,np.float32), "order_latents")
    return np.ascontiguousarray(values[:, ORDER])

def decode_ordered_prefix(values):
    """Decode only the available causal prefix; returns physical stage-order columns.

    Invalid or saturated model output fails the entire request. It is not clipped,
    converted into accepted finite bounds, filtered, or independently redrawn.
    """
    require_sources()
    rows = _rows(values, tuple(range(1,22)), (np.float64,np.float32), "decode_ordered_prefix")
    geo = _geometry()
    latent = np.ascontiguousarray(rows, dtype=np.float64)
    physical = _decode_prefix(latent, geo)
    coefficient = F32_LATENT_ADMISSIBILITY if rows.dtype == np.dtype("float32") else F64_LATENT_ADMISSIBILITY
    replay = _encode_prefix(physical, geo)
    _latent_check(replay, latent, coefficient)
    _physical_check(_decode_prefix(replay,geo), physical, ORDER[:rows.shape[1]], F64_PHYSICAL_UNITS, geo)
    return physical

def decode_ordered(values):
    """Complete ordered latent vector to original physical Float64[n,21]."""
    _rows(values, (21,), (np.float64,np.float32), "decode_ordered")
    return np.ascontiguousarray(decode_ordered_prefix(values)[:, INVERSE_ORDER])

def coordinate_metadata(physical_slot):
    require_sources()
    _require(type(physical_slot) is int and 0 <= physical_slot < 21, "physical slot0..20 required")
    geo = _geometry()
    base = dict(physical_slot=physical_slot, physical_name=PHYSICAL_ORDER[physical_slot],
                stage=INVERSE_ORDER[physical_slot], latent_name="z_"+PHYSICAL_ORDER[physical_slot])
    if physical_slot < 3:
        base.update(kind="global_lower_cholesky_residual",
                    encode_physical_dependencies=list(range(physical_slot+1)),
                    decode_latent_dependencies=list(range(physical_slot+1)))
    elif physical_slot in geo["noise"]:
        base.update(kind="independent_truncated_gaussian_prior_CDF_to_normal",
                    encode_physical_dependencies=[physical_slot], decode_latent_dependencies=[physical_slot])
    else:
        item, component = geo["by_physical"][physical_slot]
        base.update(kind="local_conditional_gaussian_residual_given_all_globals",
                    dataset_id=item["dataset_id"],
                    encode_physical_dependencies=list(range(3))+list(item["indices"][:component+1]),
                    decode_latent_dependencies=list(range(3))+list(item["indices"][:component+1]))
    return base

def metadata():
    require_sources()
    geo = _geometry()
    return dict(schema=SCHEMA, coordinate_id=COORDINATE_ID, module_sha256=source_sha256(),
        attribution="Built with PriorLabs-TabPFN", scientific_fingerprint=geo["problem"].scientific_fingerprint,
        scientific_and_order_sources_sha256=dict(PINNED_SOURCES),
        physical_parameter_order=list(PHYSICAL_ORDER), stage_to_physical_order=list(ORDER),
        physical_to_stage_order=list(INVERSE_ORDER), coordinates=[coordinate_metadata(k) for k in range(21)],
        global_prior_mean=geo["mean"].tolist(), global_prior_covariance=geo["covariance"].tolist(),
        global_lower_cholesky=geo["factor"].tolist(),
        conditional_blocks=[dict(dataset_id=b["dataset_id"],physical_indices=list(b["indices"]),
            mean_map=b["mapping"].tolist(),mean_offset=b["offset"].tolist(),
            conditional_covariance=b["covariance"].tolist(),lower_cholesky=b["factor"].tolist()) for b in geo["blocks"]],
        noise_priors={str(k):{a:float(b) for a,b in spec.items()} for k,spec in geo["noise"].items()},
        physical_roundtrip_centers=geo["centers"].tolist(),physical_roundtrip_prior_units=geo["units"].tolist(),
        tolerances=dict(F64_physical_prior_units=F64_PHYSICAL_UNITS,
            F64_latent_conditioning_admissibility=F64_LATENT_ADMISSIBILITY,
            F32_model_cast_physical_prior_units=F32_PHYSICAL_UNITS,
            F32_model_cast_latent_admissibility=F32_LATENT_ADMISSIBILITY,
            bound="coefficient*(1+abs(expected standardized coordinate))",
            F64_latent_scope="declared conditioning/roundtrip admissibility; NOT an ulp-Float64 accuracy guarantee",
            qualification_CDF_TV_quantile_thresholds_changed=False),
        input_dtype="original saved Float64 physical; no implicit conversion",
        model_cast="one final Float32 cast of Float64 latent coordinates",
        normal_prior_scope="ideal real-arithmetic prior only; no posterior normality or accuracy guarantee",
        first_global_affine_cancels_after_empirical_target_standardization=True,
        first_global_density_correction_claimed=False, references_truth_or_fitted_constants=False,
        endpoint_nonfinite_saturation_roundtrip_policy="FAIL whole qualification/request; no clipping, discard, repair or atoms",
        decoded_before_original_FIFO_and_support=True,
        global_refresh_contract="decode first; exact G3 replaces only physical0:3 and keeps physical3:21 byte-identical",
        old_numerical_qualification_transferred=False, old_native_fallback_policy_transferred=False)

def roundtrip_report(values):
    """Deterministic codec-only scalar inventory; no RNG, references, or fitting."""
    geo = _geometry()
    latent64 = encode_physical(values)
    latent32 = encode_for_model(values)
    back64 = decode_ordered(order_latents(latent64))
    back32 = decode_ordered(order_latents(latent32))
    units = geo["units"]
    physical_factor = 1 + np.abs((values-geo["centers"])/units)
    replay64 = encode_physical(back64)
    replay32 = encode_physical(back32)
    ratios = dict(
        physical64=np.max((np.abs(back64-values)/units)/(F64_PHYSICAL_UNITS*physical_factor),axis=0).tolist(),
        model32_physical=np.max((np.abs(back32-values)/units)/(F32_PHYSICAL_UNITS*physical_factor),axis=0).tolist(),
        latent64=np.max(np.abs(replay64-latent64)/(F64_LATENT_ADMISSIBILITY*(1+np.abs(latent64))),axis=0).tolist(),
        model32_latent=np.max(np.abs(replay32-latent32)/(F32_LATENT_ADMISSIBILITY*(1+np.abs(latent32))),axis=0).tolist(),
        final_model_cast=np.max(np.abs(latent32.astype(np.float64)-latent64)/(F32_LATENT_ADMISSIBILITY*(1+np.abs(latent64))),axis=0).tolist())
    return dict(count=len(values), input_dtype=str(values.dtype), error_to_declared_bound_max_per_coordinate=ratios,
        latent_min_per_coordinate=np.min(latent64,axis=0).tolist(),
        latent_max_per_coordinate=np.max(latent64,axis=0).tolist(),
        physical64_max_error_prior_units=np.max(np.abs(back64-values)/units,axis=0).tolist(),
        model32_max_error_prior_units=np.max(np.abs(back32-values)/units,axis=0).tolist(),
        latent64_max_roundtrip_error=np.max(np.abs(replay64-latent64),axis=0).tolist(),
        latent32_max_roundtrip_error=np.max(np.abs(replay32-latent32),axis=0).tolist(),
        training_endpoint_counts={str(k):int(np.sum((values[:,k]<=s["low"])|(values[:,k]>=s["high"]))) for k,s in geo["noise"].items()},
        Float64_roundtrip_passed=True, model_Float32_cast_roundtrip_passed=True,
        no_rows_removed=True, no_clipping_or_repair=True)
