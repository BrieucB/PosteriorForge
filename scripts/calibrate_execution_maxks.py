#!/usr/bin/env python3
"""Whole-vector permutation calibration of the maximum original 53-axis KS.

Built with PriorLabs-TabPFN. Diagnostic only: no inference, reference
integration, row repair, equivalence claim, or investigation-threshold change.
"""
from __future__ import annotations

import argparse
import itertools
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
from scipy.stats import beta

_REVIEW_PATH = Path(__file__).with_name("evaluate_execution_variant.py")
_spec = importlib.util.spec_from_file_location("maxks_execution_review", _REVIEW_PATH)
review = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(review)
from posteriorforge.artifacts import sha256_file, write_json
from posteriorforge.banks import load_heldout
from posteriorforge.validation import compare_to_upstream

SCHEMA = "posteriorforge.execution_maxks_permutation.v1"
ATTRIBUTION = "Built with PriorLabs-TabPFN"
METHOD_URL = "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html"


def require(condition, message):
    if not condition:
        raise ValueError(message)


class ProjectionRankCache:
    """Sort each pooled projection once; whole-row labels are common to axes."""

    def __init__(self, first, second, directions):
        a, b, directions = (np.asarray(x, dtype=np.float64)
                            for x in (first, second, directions))
        require(a.ndim == b.ndim == directions.ndim == 2
                and a.shape[1] == b.shape[1] == directions.shape[1],
                "compatible vector cohorts and projection matrix required")
        require(min(len(a), len(b)) >= 2
                and all(np.isfinite(x).all() for x in (a, b, directions)),
                "finite cohorts with at least two rows each required")
        self.n_first, self.n_second = len(a), len(b)
        self.n = len(a) + len(b)
        require(self.n < np.iinfo(np.int32).max, "rank cache exceeds int32 range")
        pooled = np.concatenate((a @ directions.T, b @ directions.T)).T
        self.order = np.argsort(pooled, axis=1, kind="stable").astype(np.int32)
        sorted_values = np.take_along_axis(pooled, self.order, axis=1)
        self.tie_end = np.ones(sorted_values.shape, dtype=bool)
        self.tie_end[:, :-1] = sorted_values[:, :-1] != sorted_values[:, 1:]
        self.positions = np.arange(1, self.n + 1, dtype=np.int32)[None, :]
        self.denominator = (self.n_first if self.n_first == self.n_second
                            else self.n_first * self.n_second)
        self.tied_adjacent_count = int((~self.tie_end).sum())

    def counts(self, row_labels):
        """Return exact integer KS numerators; evaluate only after whole ties."""
        labels = np.asarray(row_labels)
        require(labels.shape == (self.n,) and labels.dtype == np.bool_
                and int(labels.sum()) == self.n_first,
                "one balanced boolean whole-row allocation required")
        cumulative = np.cumsum(labels[self.order], axis=1, dtype=np.int32)
        if self.n_first == self.n_second:
            delta = cumulative
            delta *= 2
            delta -= self.positions
        else:
            delta = self.n * cumulative.astype(np.int64)
            delta -= self.n_first * self.positions.astype(np.int64)
        np.abs(delta, out=delta)
        return np.max(delta, axis=1, where=self.tie_end, initial=0).astype(np.int64)

    def observed_counts(self):
        labels = np.arange(self.n) < self.n_first
        return self.counts(labels)

    def storage_bytes(self):
        return sum(x.nbytes for x in (self.order, self.tie_end, self.positions))


def monte_carlo_interval(exceedances, permutations, confidence=.95):
    """Exact binomial interval for the random-label exceedance probability."""
    k, n = int(exceedances), int(permutations)
    require(0 <= k <= n and n > 0 and 0 < confidence < 1,
            "valid exceedance count, permutation count and confidence required")
    alpha = (1 - confidence) / 2
    lower = 0. if k == 0 else float(beta.ppf(alpha, k, n-k+1))
    upper = 1. if k == n else float(beta.ppf(1-alpha, k+1, n-k))
    return {
        "confidence": confidence,
        "method": "Clopper-Pearson binomial interval for random-label exceedance probability",
        "null_exceedance_probability_interval": [lower, upper],
        "plus_one_probability_mc_interval": [(1+n*lower)/(n+1), (1+n*upper)/(n+1)],
        "interpretation": "Monte Carlo uncertainty only; not population uncertainty or an equivalence interval",
    }


def calibrate(first, second, directions, *, permutations, seed_sequence):
    require(type(permutations) is int and permutations > 0,
            "positive integer permutation count required")
    started = time.perf_counter()
    cache = ProjectionRankCache(first, second, directions)
    observed = cache.observed_counts()
    rng = np.random.Generator(np.random.PCG64(seed_sequence))
    labels = np.arange(cache.n) < cache.n_first
    null = np.empty((permutations, len(observed)), dtype=np.int64)
    for index in range(permutations):
        # A single row allocation is shared across every axis. Never permute
        # the individual projection columns independently.
        rng.shuffle(labels)
        null[index] = cache.counts(labels)
    null_max, observed_max = null.max(axis=1), int(observed.max())
    exceedances = int((null_max >= observed_max).sum())  # Exact integer ties.
    ks = observed / cache.denominator
    values = null_max / cache.denominator
    return {
        "observed_axis_ks": ks.tolist(),
        "observed_max53_ks": float(ks.max()),
        "observed_max53_integer_numerator": observed_max,
        "integer_denominator": cache.denominator,
        "permutations": permutations,
        "exceedances_greater_or_equal": exceedances,
        "randomized_p_plus_one": (1+exceedances)/(permutations+1),
        "p_value_resolution": 1/(permutations+1),
        "monte_carlo_uncertainty": monte_carlo_interval(exceedances, permutations),
        "null_max53_quantiles": {str(q): float(np.quantile(values, q))
                                 for q in (.5, .9, .95, .99)},
        "null_axis_ks_integer_numerators": null.tolist(),
        "null_max53_integer_numerators": null_max.tolist(),
        "seed_sequence": {"entropy": seed_sequence.entropy,
                          "spawn_key": list(seed_sequence.spawn_key),
                          "bit_generator": "PCG64"},
        "rank_cache": {"sort_once_per_pair": True, "whole_row_labels": True,
                       "preserves_axis_correlations": True,
                       "cdf_ties_evaluated_at_last_unique_value_index": True,
                       "tied_adjacent_count": cache.tied_adjacent_count,
                       "storage_bytes": cache.storage_bytes()},
        "elapsed_cpu_wall_seconds": time.perf_counter()-started,
    }


def authenticate_inputs(root, review_path):
    bindings = review.bindings
    value = json.loads(review_path.read_text())
    require(value.get("schema") == review.SCHEMA
            and value.get("sample_integrity", {}).get("passed") is True,
            "complete authenticated execution review required")
    results = root/"results"
    gate_path = results/"gate-final.json"
    require(sha256_file(gate_path) == bindings.BASELINE_GATE_SHA256,
            "immutable original gate changed")
    query = load_heldout(results/"banks-v1", 0).obs
    _, _, reference, custody = bindings.reference_for_case(
        results, json.loads(gate_path.read_text()), query, 0)
    require(value.get("reference") == custody, "review/reference custody changed")
    records = [("candidate", value["candidate"])]
    cohorts = value["control_repetition_context"]["cohorts"]
    require([c["repetition"] for c in cohorts] == [0, 1, 2],
            "the three original independent controls required")
    records += [("control_rep_"+str(c["repetition"]), c) for c in cohorts]
    arrays, sources = {}, {}
    for name, record in records:
        path = Path(record["manifest_file"]).resolve()
        require(path.is_relative_to(results), "run manifest outside scratch results")
        require(sha256_file(path) == record["manifest_sha256"], "run manifest changed")
        manifest = json.loads(path.read_text())
        require(manifest["samples_sha256"] == record["samples_sha256"],
                "review/payload binding changed")
        bindings.validate_run_reference_binding(manifest, custody)
        samples = bindings.load_samples(path, manifest)
        require(samples.shape == (50000, 21) and np.isfinite(samples).all()
                and review.support_mask(samples).all(),
                "all original 50000 complete supported rows required")
        require(manifest["parameter_order"] == value["parameter_order"],
                "physical parameter order changed")
        arrays[name] = samples
        sources[name] = {"manifest_file": str(path),
                         "manifest_sha256": record["manifest_sha256"],
                         "samples_sha256": record["samples_sha256"]}
    require(sources["control_rep_0"]["samples_sha256"] == review.CONTROL_SAMPLES_SHA256,
            "rep0 is no longer the immutable primary corrected control")
    return value, arrays, sources, reference


def run(args):
    root, review_path = Path(args.root).resolve(), Path(args.review).resolve()
    output = Path(args.output).resolve()
    require(args.permutations > 0 and args.seed >= 0, "positive permutations and nonnegative seed required")
    require(output.is_relative_to(root/"results/performance-first-v1"),
            "diagnostic output must be in the separate performance results tree")
    require(not output.exists(), "fresh diagnostic output directory required")
    value, arrays, sources, reference = authenticate_inputs(root, review_path)
    names = list(arrays)
    pairs = [("candidate", name) for name in names if name != "candidate"]
    pairs += list(itertools.combinations(names[1:], 2))
    seeds = np.random.SeedSequence(args.seed).spawn(len(pairs))
    result = {
        "schema": SCHEMA, "attribution": ATTRIBUTION,
        "scope": "correlation_aware_max53_KS_random_label_diagnostic_only",
        "no_model_calls": True, "no_reference_integrations": True,
        "all_original_rows_used": True, "samples_modified": False,
        "original_investigation_thresholds_and_flags_retained": True,
        "population_equivalence_claim": False, "absolute_physical_gate_waived": False,
        "method": {"official_primary_source": METHOD_URL,
                   "null": "independent exchangeable complete vectors have a common distribution; condition on their pooled vectors",
                   "randomization": "balanced whole-row label shuffles shared across all53 axes, independently repeated with replacement",
                   "statistic": "maximum across the original53 exact two-sample ECDF suprema",
                   "tail": "greater: null max53 >= observed max53, including exact integer ties",
                   "p_value": "(1 + exceedances)/(1 + random permutations)",
                   "not_scipy_two_sided_option": "KS already uses an absolute ECDF difference; upper-tail calibration applies to that nonnegative maximum"},
        "limitations": [
            "Exchangeability and independence of vectors under the common-law null are assumptions, not established by this calibration.",
            "Same fixed bank/query is conditioned upon; this does not assess new contexts, cases, seeds or unfiltered proposal conditional errors.",
            "A large randomized p-value is not evidence of equivalence, absence of drift, or scientific posterior accuracy.",
            "The fixed .01 flags remain; no new threshold replaces them and no pair-selection multiplicity claim is made.",
            "This max53 CDF diagnostic does not resolve mean/SD flags or the control_rep_2 a_a3 tail row.",
            "Monte Carlo intervals quantify finite randomization simulation error only."],
        "review": {"file": str(review_path), "sha256": sha256_file(review_path),
                   "source_evaluator_sha256": value["source"]["evaluation_script_sha256"],
                   "numeric_review_status_unchanged": value["numeric_review_status"]},
        "sources": sources, "reference": value["reference"],
        "parameter_order": value["parameter_order"],
        "source_script_sha256": sha256_file(__file__),
        "numpy_version": np.__version__, "seed": args.seed, "pairs": [],
    }
    output.mkdir(parents=True)
    for index, ((first, second), seed) in enumerate(zip(pairs, seeds)):
        exact = compare_to_upstream(arrays[first], arrays[second],
                                    reference_sd=reference["sd"], directions=reference["directions"])
        row = calibrate(arrays[first], arrays[second], reference["directions"],
                        permutations=args.permutations, seed_sequence=seed)
        difference = float(np.max(np.abs(np.asarray(row["observed_axis_ks"])
                                        - exact["cdf_difference"])))
        require(difference <= 1e-12, "observed rank KS differs from unchanged comparison API")
        if first == "candidate":
            saved = value["control_repetition_context"]["cohorts"][index]["candidate_vs_this_control"]
            require(np.max(np.abs(np.asarray(row["observed_axis_ks"])
                                  - saved["cdf_difference"])) <= 1e-12,
                    "observed53 KS differs from saved evaluator")
        row.update(first=first, second=second, exact_api_max_error=difference,
                   original_relative_investigation=exact,
                   original_cdf_flag_axes=[i for i, x in enumerate(exact["cdf_difference"]) if x > .01])
        result["pairs"].append(row)
        write_json(output/"maxks-permutation.json", result)
        print(json.dumps({"pair": [first, second], "observed_max53_ks": row["observed_max53_ks"],
                          "exceedances": row["exceedances_greater_or_equal"],
                          "permutations": args.permutations,
                          "randomized_p_plus_one": row["randomized_p_plus_one"]}), flush=True)
    result["status"] = "complete_diagnostic_no_equivalence_or_threshold_waiver"
    write_json(output/"maxks-permutation.json", result)
    write_json(output/"checksums.json", {"maxks-permutation.json": sha256_file(output/"maxks-permutation.json")})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--review", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--permutations", type=int, default=127)
    parser.add_argument("--seed", type=int, default=20261112)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
