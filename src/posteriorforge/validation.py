"""Absolute posterior gates using retained analytic reference CDF witnesses.

A finite CDF grid is not a Kolmogorov distance. We report both a witnessed
lower bound and a conservative monotonicity upper bound, including tail mass
and the qualified reference numerical precision. Ambiguity never becomes pass.
No function clips, repairs, or selectively discards posterior sample rows.
"""
from __future__ import annotations

import numpy as np

CDF_LIMIT = .10
BIAS_SD_LIMIT = .20
SD_RATIO_LIMITS = (.8, 1.2)
REFERENCE_PRECISION_LIMIT = .005


def sample_integrity(samples, *, expected_count=50000, dimension=21,
                     lower_bounds=None, upper_bounds=None, support_mask=None):
    """Validate complete vectors; caller may supply an exact joint-prior mask."""
    values = np.asarray(samples)
    if type(expected_count) is not int or expected_count < 1:
        raise ValueError("positive expected_count required")
    shape_ok = values.ndim == 2 and values.shape == (expected_count, dimension)
    result = {"passed": False, "expected_count": expected_count, "expected_dimension": dimension,
              "shape": list(values.shape), "complete_shape": bool(shape_ok),
              "finite": False, "in_prior_support": False,
              "clipping_or_repair_performed": False}
    if not shape_ok:
        return result
    if values.dtype.kind not in "fiu":
        raise ValueError("numeric posterior array required")
    finite = np.isfinite(values).all(axis=1)
    supported = np.ones(len(values), dtype=bool)
    for bounds, lower in ((lower_bounds, True), (upper_bounds, False)):
        if bounds is None:
            continue
        bounds = np.asarray(bounds, dtype=np.float64)
        if bounds.shape != (dimension,) or np.isnan(bounds).any():
            raise ValueError("prior bounds must have one value per parameter")
        supported &= ((values >= bounds) if lower else (values <= bounds)).all(axis=1)
    if lower_bounds is not None and upper_bounds is not None:
        if np.any(np.asarray(lower_bounds) >= np.asarray(upper_bounds)):
            raise ValueError("prior lower bounds must be below upper bounds")
    if support_mask is not None:
        mask = np.asarray(support_mask)
        if mask.dtype != np.bool_ or mask.shape != (expected_count,):
            raise ValueError("exact prior support mask must be boolean[n]")
        supported &= mask
    # A caller must actually describe support; finite alone is insufficient.
    support_defined = support_mask is not None or (lower_bounds is not None and upper_bounds is not None)
    result.update(finite=bool(finite.all()), in_prior_support=bool(support_defined and supported.all()),
                  nonfinite_row_count=int((~finite).sum()),
                  out_of_support_row_count=int((~supported).sum()),
                  prior_support_defined=bool(support_defined))
    result["passed"] = result["finite"] and result["in_prior_support"]
    return result


def cdf_distance_bounds(projected_samples, queries, reference_cdfs, *, precision=0.):
    """Bound the supremum over the whole line for every projection.

    Between adjacent query points both CDFs are monotone. Their opposite end
    values bound every discrepancy; below/above the grid the remaining tail
    masses bound it. ``precision`` is added/subtracted conservatively. The
    returned bounds describe empirical-versus-reference-CDF distance, not a
    population confidence interval for the stochastic inference method.
    """
    values = np.asarray(projected_samples, dtype=np.float64)
    queries = np.asarray(queries, dtype=np.float64)
    cdfs = np.asarray(reference_cdfs, dtype=np.float64)
    if values.ndim != 2 or len(values) < 1 or not np.isfinite(values).all():
        raise ValueError("finite projected_samples[n,k] required")
    if queries.ndim != 1 or len(queries) < 2 or not np.isfinite(queries).all() or np.any(np.diff(queries) <= 0):
        raise ValueError("strictly increasing finite query grid required")
    if cdfs.shape != (values.shape[1], len(queries)) or not np.isfinite(cdfs).all():
        raise ValueError("reference CDF grid must have shape[k,q]")
    if np.any(cdfs < -1e-12) or np.any(cdfs > 1 + 1e-12) or np.any(np.diff(cdfs, axis=1) < -1e-12):
        raise ValueError("reference CDFs are not monotone probabilities")
    if not np.isfinite(precision) or not 0 <= precision <= 1:
        raise ValueError("finite nonnegative reference precision required")
    lower, upper, witnessed = [], [], []
    for index in range(values.shape[1]):
        ordered = np.sort(values[:, index])
        left = np.searchsorted(ordered, queries, side="left") / len(ordered)
        right = np.searchsorted(ordered, queries, side="right") / len(ordered)
        cdf = cdfs[index]
        point_error = float(max(np.max(np.abs(left - cdf)), np.max(np.abs(right - cdf))))
        # Empirical CDF just below the right endpoint and just above the left.
        interior = max(float(np.max(cdf[1:] - right[:-1])),
                       float(np.max(left[1:] - cdf[:-1])))
        tails = max(float(cdf[0]), float(left[0]), float(1 - cdf[-1]), float(1 - right[-1]))
        witnessed.append(point_error)
        lower.append(max(0., point_error - precision))
        upper.append(min(1., max(point_error, interior, tails) + precision))
    return {"lower": lower, "upper": upper, "grid_witness_error": witnessed,
            "reference_precision": float(precision), "grid_point_count": len(queries),
            "bound_method": "whole_line_monotonicity_including_tails_plus_reference_precision"}


def _reference_arrays(reference):
    required = ("directions", "queries", "cdfs", "mean", "sd", "qualification")
    missing = [key for key in required if key not in reference]
    if missing:
        raise ValueError("reference is missing: " + ", ".join(missing))
    directions = np.asarray(reference["directions"], dtype=np.float64)
    mean = np.asarray(reference["mean"], dtype=np.float64)
    sd = np.asarray(reference["sd"], dtype=np.float64)
    if directions.shape != (53, 21) or not np.isfinite(directions).all():
        raise ValueError("21 physical axes and 32 full-vector directions required")
    if not np.all(np.diag(directions[:21]) > 0) or not np.array_equal(directions[:21], np.diag(np.diag(directions[:21]))):
        raise ValueError("first 21 witnesses must be positive scaled physical axes")
    if np.any(directions[21:] == 0):
        raise ValueError("32 joint witnesses must include every vector coordinate")
    if mean.shape != (21,) or sd.shape != (21,) or not np.isfinite(mean).all() or not np.isfinite(sd).all() or np.any(sd <= 0):
        raise ValueError("finite analytic reference mean and positive SD[21] required")
    qualification = reference["qualification"]
    if not isinstance(qualification, dict):
        raise ValueError("reference qualification mapping required")
    precision = qualification.get("precision")
    qualified = (qualification.get("state") == "qualified" and type(precision) in (int, float)
                 and np.isfinite(precision) and 0 <= precision <= REFERENCE_PRECISION_LIMIT
                 and all(qualification.get(key) is True for key in
                         ("witness_gate_passed", "independent_nested_passed", "cpu_check_passed", "joint_reference_passed")))
    return directions, mean, sd, qualification, bool(qualified)


def assess_samples(samples, reference, *, lower_bounds=None, upper_bounds=None,
                   support_mask=None, expected_count=50000):
    """Apply the same absolute gate to upstream and any optimized variant.

    ``reference`` combines arrays from load_reference_distribution with its
    manifest qualification. Qualified reference uncertainty is included in CDF
    bounds. If reference precision or independent checks are incomplete, status
    is reference_inconclusive and pass is impossible.
    """
    if type(expected_count) is not int or expected_count < 2:
        raise ValueError("at least two complete samples required for posterior SD")
    directions, reference_mean, reference_sd, qualification, qualified = _reference_arrays(reference)
    integrity = sample_integrity(samples, expected_count=expected_count, lower_bounds=lower_bounds,
                                 upper_bounds=upper_bounds, support_mask=support_mask)
    result = {"schema": "posteriorforge.posterior_gate.v1", "passed": False,
              "status": "reference_inconclusive", "reference_qualified": qualified,
              "reference_qualification": qualification, "integrity": integrity,
              "limits": {"cdf": CDF_LIMIT, "absolute_bias_reference_sd": BIAS_SD_LIMIT,
                         "sd_ratio": list(SD_RATIO_LIMITS), "reference_precision": REFERENCE_PRECISION_LIMIT},
              "scope": "eight_case_diagnostic_screen_not_global_calibration_or_SBC"}
    if not integrity["passed"]:
        result["status"] = "invalid_samples"
        return result
    values = np.asarray(samples, dtype=np.float64)
    sample_mean, sample_sd = values.mean(axis=0), values.std(axis=0, ddof=1)
    bias = np.abs(sample_mean - reference_mean) / reference_sd
    ratio = sample_sd / reference_sd
    moment_passed = bool(np.max(bias) <= BIAS_SD_LIMIT and np.min(ratio) >= SD_RATIO_LIMITS[0]
                         and np.max(ratio) <= SD_RATIO_LIMITS[1])
    result["moments"] = {"sample_mean": sample_mean.tolist(), "sample_sd": sample_sd.tolist(),
                         "sd_ddof": 1, "reference_mean": reference_mean.tolist(),
                         "reference_sd": reference_sd.tolist(), "absolute_bias_reference_sd": bias.tolist(),
                         "max_absolute_bias_reference_sd": float(np.max(bias)), "sd_ratio": ratio.tolist(),
                         "passed": moment_passed}
    precision = qualification.get("precision")
    if type(precision) not in (float, int) or not np.isfinite(precision) or not 0 <= precision <= 1:
        # Retain useful raw witnesses, but do not invent missing precision.
        precision = 0.
    bounds = cdf_distance_bounds(values @ directions.T, reference["queries"], reference["cdfs"], precision=precision)
    for label, positions in (("marginal_cdf", slice(0, 21)), ("joint_projection_cdf", slice(21, 53))):
        lower = np.asarray(bounds["lower"])[positions]
        upper = np.asarray(bounds["upper"])[positions]
        witnessed = np.asarray(bounds["grid_witness_error"])[positions]
        result[label] = {"lower": lower.tolist(), "upper": upper.tolist(),
                         "max_lower": float(lower.max()), "max_upper": float(upper.max()),
                         "max_grid_witness_error": float(witnessed.max()),
                         "passed": bool(upper.max() <= CDF_LIMIT)}
    result["cdf_bound_method"] = bounds["bound_method"]
    result["cdf_grid_point_count"] = bounds["grid_point_count"]
    if not qualified:
        return result
    if not moment_passed or result["marginal_cdf"]["max_lower"] > CDF_LIMIT or result["joint_projection_cdf"]["max_lower"] > CDF_LIMIT:
        result["status"] = "model_failure"
    elif result["marginal_cdf"]["passed"] and result["joint_projection_cdf"]["passed"]:
        result.update(status="pass", passed=True)
    else:
        result["status"] = "cdf_inconclusive"
    return result


def _two_sample_ks(first, second):
    first, second = np.sort(first), np.sort(second)
    grid = np.unique(np.concatenate((first, second)))
    return float(np.max(np.abs(np.searchsorted(first, grid, side="right") / len(first)
                               - np.searchsorted(second, grid, side="right") / len(second))))


def compare_to_upstream(samples, upstream_samples, *, reference_sd, directions):
    """Flag distribution differences for investigation; this is no gate waiver."""
    values, baseline = np.asarray(samples, dtype=np.float64), np.asarray(upstream_samples, dtype=np.float64)
    sd, directions = np.asarray(reference_sd), np.asarray(directions)
    if (values.ndim != 2 or baseline.ndim != 2 or values.shape[1] != 21 or baseline.shape[1] != 21
            or min(len(values), len(baseline)) < 2 or not np.isfinite(values).all() or not np.isfinite(baseline).all()
            or sd.shape != (21,) or not np.isfinite(sd).all() or np.any(sd <= 0)
            or directions.shape != (53, 21) or not np.isfinite(directions).all()):
        raise ValueError("finite complete vector cohorts, SD[21], directions[53,21] required")
    projected, old_projected = values @ directions.T, baseline @ directions.T
    cdf = np.asarray([_two_sample_ks(projected[:, index], old_projected[:, index]) for index in range(53)])
    mean = np.abs(values.mean(0) - baseline.mean(0)) / sd
    old_sd = baseline.std(0, ddof=1)
    if np.any(old_sd == 0):
        raise ValueError("upstream sample SD must be positive")
    sd_fraction = np.abs(values.std(0, ddof=1) / old_sd - 1)
    flagged = bool(np.max(cdf) > .01 or np.max(mean) > .05 or np.max(sd_fraction) > .05)
    return {"schema": "posteriorforge.upstream_comparison.v1", "investigation_required": flagged,
            "cdf_difference": cdf.tolist(), "max_cdf_difference": float(cdf.max()),
            "mean_difference_reference_sd": mean.tolist(), "max_mean_difference_reference_sd": float(mean.max()),
            "sd_fraction_difference": sd_fraction.tolist(), "max_sd_fraction_difference": float(sd_fraction.max()),
            "investigation_thresholds": {"cdf": .01, "mean_reference_sd": .05, "sd_fraction": .05},
            "absolute_gate_still_required": True}


def make_reference_input(arrays, manifest):
    """Adapt authenticated load_reference_distribution output for the gate.

    This is an adapter, not file authentication. Call the reference loader first
    and verify its scientific/observation binding against the inference run.
    Inconsistent outer qualification fields fail closed.
    """
    qualification = dict(manifest.get("qualification", {}))
    outer_qualified = (manifest.get("state") == "verified" and manifest.get("converged") is True
                       and manifest.get("independent_passed") is True)
    if not outer_qualified:
        qualification["state"] = "inconclusive"
    result = {key: arrays[key] for key in ("directions", "queries", "cdfs", "mean", "sd")}
    result["qualification"] = qualification
    return result
