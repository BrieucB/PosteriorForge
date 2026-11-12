#!/usr/bin/env python3
"""Append authenticated first-50k hybrid pilot details to an existing report.

Built with PriorLabs-TabPFN. CPU-only saved-artifact reporting: no model calls,
reference integrations, qualification decisions, row removal, or gate changes.
Generate the base report with report_execution_performance.py first.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
import re
import statistics
import tempfile

import numpy as np

from posteriorforge.artifacts import sha256_file, write_json
from posteriorforge.reference import load_reference_distribution

REPO = Path(__file__).resolve().parents[1]
PHASE = Path("results/performance-first-v1")
PILOT = PHASE / "hybrid-fp32-batch-1024/rep-0/seed-20261111-case-0"
PRIMARY = Path("results/cdf-arithmetic-v1/context-4096/seed-20261111-case-0")
REFERENCE = Path("results/reference-grids-v1/case-0")
PILOT_SAMPLES_SHA256 = "52c7e0a53e7dd95901ce68338d79cb1966d5d6ce43c4cf78f40998b3910c7683"
PRIMARY_MANIFEST_SHA256 = "a0114fd74721cf9bed31e9131045e841f262db123c8f849e5cb3f15814a14f4b"
REVIEW_SHA256 = "ee081dbd320f2548edfa0ab99c5249852f5b28b357293315511a8be28c17c390"
BEGIN = "<!-- posteriorforge:completed-hybrid-pilot-details:begin -->"
END = "<!-- posteriorforge:completed-hybrid-pilot-details:end -->"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_json(path):
    return json.loads(Path(path).read_text())


def portable_binding(binding, root):
    result = dict(binding)
    if "path" in result:
        result["path"] = str(Path(result["path"]).resolve().relative_to(root))
    return result


def append_section(report_path, section):
    """Replace our block atomically, including sections from the old work script."""
    text = report_path.read_text()
    require(text.count(BEGIN) == text.count(END) <= 1, "malformed pilot report markers")
    if BEGIN in text:
        before, remainder = text.split(BEGIN)
        _, after = remainder.split(END)
        text = before.rstrip() + "\n" + after.lstrip("\n")
    for title in ("Completed pilot interpretation", "Exploratory finite-cohort calibration"):
        text = re.sub(
            r"(?ms)^## " + re.escape(title) + r"\n.*?(?=^## |\Z)", "", text
        )
    text = text.rstrip() + "\n\n" + BEGIN + "\n" + section.rstrip() + "\n" + END + "\n"
    with tempfile.NamedTemporaryFile("w", dir=report_path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(text)
    temporary.replace(report_path)


def render(*, root, output):
    root = Path(root).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    report_path = output / "report.md"
    require(report_path.is_file(), "--output must contain an existing base report.md")
    figure_data = load_json(output / "figure-data.json")
    spec = importlib.util.spec_from_file_location(
        "pilot_execution_plots", REPO / "scripts/report_execution_performance.py"
    )
    plots = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plots)
    candidate, manifest, candidate_binding = plots.load_run(root / PILOT)
    primary, primary_manifest, primary_binding = plots.load_run(root / PRIMARY)
    require(candidate_binding["samples_sha256"] == PILOT_SAMPLES_SHA256,
            "this report is restricted to the reviewed first hybrid pilot")
    require(primary_binding["manifest_sha256"] == PRIMARY_MANIFEST_SHA256,
            "immutable corrected primary control required")
    require(any(a["binding"]["samples_sha256"] == PILOT_SAMPLES_SHA256
                for a in figure_data["assessments"]), "base report does not include this pilot")
    require(figure_data["control"]["manifest_sha256"] == PRIMARY_MANIFEST_SHA256,
            "base report has a different primary control")
    review_path = root / PILOT / "independent-review-v1/execution-review.json"
    require(sha256_file(review_path) == REVIEW_SHA256, "review checksum mismatch")
    review = load_json(review_path)
    for name, binding in (("candidate", candidate_binding), ("control", primary_binding)):
        for key in ("manifest_sha256", "samples_sha256"):
            require(review[name][key] == binding[key], "review binding mismatch: " + name)
    arrays, reference = load_reference_distribution(root / REFERENCE)
    require(reference["binding"]["observation_sha256"] == manifest["query"]["sha256"],
            "reference observation mismatch")
    require(sha256_file(root / REFERENCE / "manifest.json") ==
            figure_data["reference_manifest_sha256"], "base report reference mismatch")
    for key in ("scientific_fingerprint", "parameter_order"):
        require(reference["binding"][key] == manifest[key], "reference mismatch: " + key)
    controls = []
    for repetition in range(3):
        path = root / PHASE / f"controls/rep-{repetition}/seed-20261111-case-0"
        _, control_manifest, binding = plots.load_run(path)
        reviewed = review["control_repetition_context"]["cohorts"][repetition]
        for key in ("manifest_sha256", "samples_sha256"):
            require(reviewed[key] == binding[key], "control repetition binding mismatch")
        controls.append((control_manifest, portable_binding(binding, root)))
    permutation_path = root / PHASE / "maxks-permutation-127-v1/maxks-permutation.json"
    permutation = load_json(permutation_path)
    require(permutation["review"]["sha256"] == REVIEW_SHA256, "permutation review mismatch")
    require(permutation["sources"]["candidate"]["samples_sha256"] == PILOT_SAMPLES_SHA256,
            "permutation candidate mismatch")
    require(permutation["original_investigation_thresholds_and_flags_retained"] is True,
            "permutation diagnostic changed flag policy")
    for repetition, (_, binding) in enumerate(controls):
        require(permutation["sources"][f"control_rep_{repetition}"]["samples_sha256"] ==
                binding["samples_sha256"], "permutation control mismatch")

    events = manifest["stage_events"]
    require(len(events) == 42, "expected two complete 21-stage proposal waves")
    for wave in (1, 2):
        rows = sorted((r for r in events if r["proposal_round"] == wave),
                      key=lambda r: r["stage_index"])
        require([r["stage_index"] for r in rows] == list(range(21)),
                "incomplete stage wave")
        require({r["proposed_rows"] for r in rows} == ({50000} if wave == 1 else {452}),
                "unexpected proposal count")
    request = manifest["timings"]["posterior_request_seconds"]
    preparation = sum(r["prepare_seconds"] for r in events)
    prediction = sum(r["predict_and_sample_seconds"] for r in events)
    other = request - preparation - prediction
    refill_preparation = sum(r["prepare_seconds"] for r in events if r["proposal_round"] == 2)
    require(other >= 0, "stage times exceed complete request time")
    request_median = statistics.median(m["timings"]["posterior_request_seconds"]
                                      for m, _ in controls)
    wrapper_median = statistics.median(m["wrapper_total_seconds"] for m, _ in controls)
    relative = review["relative_distribution_review"]
    require(len(relative["cdf_difference"]) == 53, "incomplete reviewed CDF family")
    family = []
    witnesses = {}
    candidate_projection = candidate @ arrays["directions"].T
    primary_projection = primary @ arrays["directions"].T
    for index in range(53):
        left = np.sort(candidate_projection[:, index])
        right = np.sort(primary_projection[:, index])
        coordinate = np.unique(np.r_[left, right])
        delta = (np.searchsorted(left, coordinate, side="right") -
                 np.searchsorted(right, coordinate, side="right")) / 50000
        at = int(np.argmax(np.abs(delta)))
        distance = float(abs(delta[at]))
        require(abs(distance - relative["cdf_difference"][index]) < 1e-12,
                f"exact CDF witness disagrees with review at axis {index}")
        family.append({
            "axis_index": index,
            "name": manifest["parameter_order"][index] if index < 21 else f"joint{index - 21}",
            "ks": distance,
            "signed_cdf_difference": float(delta[at]),
            "physical_projection_coordinate": float(coordinate[at]),
            "flag": distance > relative["investigation_thresholds"]["cdf"],
        })
        if index in (21, 40):
            witnesses[f"axis_{index}_coordinate"] = coordinate
            witnesses[f"axis_{index}_signed_difference"] = delta

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    figures = {}
    def save(fig, name):
        fig.savefig(output / name, dpi=155, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        figures[name] = sha256_file(output / name)

    fig, axes = plt.subplots(3, 1, figsize=(13, 10),
                             gridspec_kw={"height_ratios": [1.2, 1.2, .65]})
    for wave, ax in zip((1, 2), axes[:2]):
        rows = sorted((r for r in events if r["proposal_round"] == wave),
                      key=lambda r: r["stage_index"])
        x = np.arange(21)
        p = np.array([r["prepare_seconds"] for r in rows])
        s = np.array([r["predict_and_sample_seconds"] for r in rows])
        ax.bar(x, p, color="#315c99", label="Stage preparation")
        ax.bar(x, s, bottom=p, color="#238e8b", label="Prediction + sampling")
        ax.set(xlabel="Autoregressive stage", ylabel="Measured seconds", ylim=(0, 45),
               title=f"Proposal wave {wave}: {rows[0]['proposed_rows']:,} complete vectors before support rejection")
        ax.set_xticks(x)
        ax.grid(axis="y", alpha=.15)
    axes[0].legend(frameon=False, ncol=2)
    axes[2].barh([0], [preparation], height=.35, color="#315c99", label="Preparation")
    axes[2].barh([0], [prediction], left=[preparation], height=.35,
                 color="#238e8b", label="Prediction + sampling")
    axes[2].barh([0], [other], left=[preparation + prediction], height=.35,
                 color="#9ca6b0", label="Other request work")
    axes[2].set_ylim(-.8, .8)
    axes[2].set(yticks=[], xlabel="Full 50,000-accepted-vector request seconds")
    legend = axes[2].legend(frameon=False, ncol=3, loc="upper center")
    fig.suptitle("Built with PriorLabs-TabPFN\nActual FP32 hybrid pilot: repeated preparation dominates the small refill wave",
                 fontsize=15)
    fig.tight_layout()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend_bounds = legend.get_window_extent(renderer)
    require(not any(legend_bounds.overlaps(bar.get_window_extent(renderer))
                    for bar in axes[2].patches), "request legend overlaps measured cost bars")
    save(fig, "actual-stage-costs.png")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for index, ax in zip((21, 40), axes):
        ax.step(witnesses[f"axis_{index}_coordinate"],
                witnesses[f"axis_{index}_signed_difference"], where="post",
                color="#ad3d5a", lw=1.3)
        ax.axhline(0, color="#7d8791", lw=.7)
        for threshold in (.01, -.01):
            ax.axhline(threshold, color="#c56a19", ls="--")
        ax.set(title=f"Fixed joint projection {index - 21}",
               xlabel="Projection coordinate theta dot direction",
               ylabel="Candidate ECDF minus control ECDF")
        ax.grid(alpha=.15)
        ax.text(.03, .94, f"Max absolute difference {family[index]['ks']:.5f}",
                transform=ax.transAxes, va="top")
    fig.suptitle("Built with PriorLabs-TabPFN: retained primary-control investigation flags\n"
                 "50,000 complete vectors per cohort; dashed lines are the unchanged 0.01 threshold",
                 fontsize=13)
    fig.tight_layout()
    save(fig, "signed-cdf-flags.png")
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.bar(np.arange(53), [r["ks"] for r in family],
           color=["#ad3d5a" if r["flag"] else "#315c99" for r in family])
    ax.axhline(.01, color="#c56a19", ls="--", label="Unchanged investigation threshold")
    ax.axvline(20.5, color="#7d8791", lw=.8)
    ax.set_xticks([0, 10, 20, 21, 31, 40, 52],
                  ["M0", "M10", "M20", "J0", "J10", "J19", "J31"], rotation=90)
    ax.set(ylabel="Exact two-sample ECDF distance",
           xlabel="Complete fixed family: 21 marginal + 32 joint projections",
           title="Built with PriorLabs-TabPFN: all 53 primary-control comparisons")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=.15)
    fig.tight_layout()
    fig.canvas.draw()
    tick_bounds = [label.get_window_extent(fig.canvas.get_renderer())
                   for label in ax.get_xticklabels()]
    require(not any(a.overlaps(b) for a, b in zip(tick_bounds, tick_bounds[1:])),
            "CDF family tick labels overlap")
    save(fig, "complete-cdf-family.png")
    np.savez_compressed(output / "signed-cdf-witnesses.npz", **witnesses)
    with (output / "pilot-cdf-family.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(family[0]))
        writer.writeheader()
        writer.writerows(family)
    stage_fields = ("proposal_round", "proposed_rows", "stage_index", "prepare_seconds",
                    "predict_and_sample_seconds", "fit_stock_cache_seconds",
                    "resident_move_seconds", "uncached_fit_seconds")
    with (output / "pilot-stage-events.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=stage_fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in stage_fields} for row in events)
    meta = {
        "schema": "posteriorforge.completed_hybrid_pilot_summary.v1",
        "attribution": "Built with PriorLabs-TabPFN",
        "source_sha256": sha256_file(__file__),
        "candidate": portable_binding(candidate_binding, root),
        "primary": portable_binding(primary_binding, root),
        "reference_manifest_sha256": sha256_file(root / REFERENCE / "manifest.json"),
        "review_sha256": REVIEW_SHA256,
        "permutation_sha256": sha256_file(permutation_path),
        "count": 50000, "one_optimized_repetition_only": True, "control_repetitions": 3,
        "controls": [{"binding": binding, "request_seconds": m["timings"]["posterior_request_seconds"],
                      "wrapper_seconds": m["wrapper_total_seconds"]} for m, binding in controls],
        "control_request_median_seconds": request_median,
        "control_wrapper_median_seconds": wrapper_median,
        "candidate_request_seconds": request,
        "request_speedup": request_median / request,
        "candidate_wrapper_seconds": manifest["wrapper_total_seconds"],
        "wrapper_speedup": wrapper_median / manifest["wrapper_total_seconds"],
        "nominal_request_gpu_hours": request / 3600,
        "nominal_wrapper_gpu_hours": manifest["wrapper_total_seconds"] / 3600,
        "cost_scope": "One GPU wall-products; not scheduler allocation accounting.",
        "memory": manifest["memory"],
        "preparation_seconds": preparation,
        "prediction_and_sampling_seconds": prediction,
        "other_request_seconds": other,
        "refill_proposed_vectors": 452,
        "refill_preparation_seconds": refill_preparation,
        "family": family,
        "relative_distribution_review": relative,
        "numeric_review_status": review["numeric_review_status"],
        "absolute_physical_status": review["absolute_physical_assessment"]["status"],
        "absolute_physical_gate_required_for_performance": False,
        "full21_hybrid_fixed_conditional_qualified": review["pointwise_qualification"]["full21_hybrid_qualified"],
        "cache_only_full21_qualified": review["pointwise_qualification"]["full21_qualified"],
        "rows_or_flags_removed": False, "no_equivalence_claim": True,
        "overall_runtime_phase_complete": False,
        "figures": figures,
        "signed_curves_sha256": sha256_file(output / "signed-cdf-witnesses.npz"),
        "tables": {name: sha256_file(output / name) for name in
                   ("pilot-cdf-family.csv", "pilot-stage-events.csv")},
        "render_checks": {"request_legend_overlaps_cost_bars": False,
                          "cdf_family_tick_labels_overlap": False,
                          "stage_events": len(events), "cdf_family_axes": len(family)},
    }
    write_json(output / "pilot-summary.json", meta)
    # Preserve the original diagnostic byte-for-byte; omit large audit traces.
    (output / "maxks-permutation.json").write_bytes(permutation_path.read_bytes())
    review_extract = {
        "schema": "posteriorforge.completed_hybrid_pilot_review_extract.v1",
        "source_review_path": str(PILOT / "independent-review-v1/execution-review.json"),
        "source_review_sha256": REVIEW_SHA256,
        "attribution": review["attribution"],
        "candidate": portable_binding(candidate_binding, root),
        "primary": portable_binding(primary_binding, root),
        "numeric_review_status": review["numeric_review_status"],
        "sample_integrity": review["sample_integrity"],
        "absolute_physical_assessment": review["absolute_physical_assessment"],
        "relative_distribution_review": relative,
        "pointwise_qualification": {key: value for key, value in
                                   review["pointwise_qualification"].items()
                                   if key not in ("qualification", "manifest_file")},
        "control_repetition_context": {
            "distinct_posterior_rng_repetitions": 3,
            "primary_old_control_is_not_a_fourth_independent_repetition": True,
            "flags_suppressed_or_thresholds_changed": False,
            "cohorts": [{
                "repetition": cohort["repetition"],
                "manifest_sha256": cohort["manifest_sha256"],
                "samples_sha256": cohort["samples_sha256"],
                "identical_to_primary_payload": cohort["identical_to_primary_payload"],
                "candidate_vs_this_control": cohort["candidate_vs_this_control"],
                "control_vs_primary": cohort["control_vs_primary"],
                "absolute_physical_status": cohort["control_absolute_physical_assessment"]["status"],
            } for cohort in review["control_repetition_context"]["cohorts"]],
        },
        "rejection_review": review["rejection_review"],
        "original_gate_unchanged": review["original_gate_unchanged"],
        "absolute_physical_gate_required_for_performance": False,
        "interpretation": review["interpretation"],
        "source": review["source"],
        "raw_arrays_and_large_audit_traces_included": False,
    }
    write_json(output / "execution-review-extract.json", review_extract)
    section = f"""## Completed pilot interpretation

Built with PriorLabs-TabPFN. The strict 50,000-vector FP32 hybrid request took {request:.6f} seconds ({request / 60:.2f} minutes), versus a {request_median:.6f}-second median across three contemporaneous unoptimized controls: **{request_median / request:.3f}x request speedup in one pilot**. Wrapper wall time was {manifest['wrapper_total_seconds']:.6f} seconds, a {wrapper_median / manifest['wrapper_total_seconds']:.3f}x speedup against the three-control wrapper median. Final optimized timing repetitions and overall runtime-phase completion remain pending.

All 50,000 rows are finite, complete and prior-supported. The primary-control maximum exact two-sample empirical CDF difference is {relative['max_cdf_difference']:.5f}, with retained flags at joint projections 0 and 19. Mean change is at most {relative['max_mean_difference_reference_sd']:.5f} reference SD and SD change at most {100 * relative['max_sd_fraction_difference']:.2f}%. Comparisons against control repetitions 1 and 2 retain their additional CDF flags; the separate control-2 tail-sensitive SD flag remains visible. No row, outlier or flag was removed. These finite-cohort checks do not prove population equivalence. The original absolute physical result remains **{review['absolute_physical_assessment']['status']}**; it is retained separately under the authorized performance-first review.

Stage preparation consumed {preparation:.3f} seconds; prediction and sampling consumed {prediction:.3f} seconds; other request work consumed {other:.3f} seconds. The 452-proposal refill repeated {refill_preparation:.3f} seconds of stage preparation. Nominal one-GPU request and wrapper wall-products were {request / 3600:.6f} and {manifest['wrapper_total_seconds'] / 3600:.6f} GPU-hours; scheduler allocated cost is a separate quantity.

![Actual completed request stage costs](actual-stage-costs.png)

The signed empirical-CDF witnesses below use every complete vector at all pooled projection values, preserving ties. Dashed lines retain the 0.01 investigation threshold.

![Signed CDF investigation flags](signed-cdf-flags.png)

All 21 marginal and 32 joint comparisons remain in the complete family plot, pilot-cdf-family.csv and pilot-summary.json. The saved witness NPZ is derived curve data, not a posterior cohort.

![Complete fixed projection family](complete-cdf-family.png)

The declared full-21 hybrid fixed-conditional screen passed; the cache-only full-21 screen did not. Fixed-conditional evidence does not prove the full posterior law. Bounded host reuse and GQA layout are separate candidates; their current probes remain unqualified, with metadata representation and global workspace use under investigation. This pilot claims no gain from either proposal.

## Exploratory finite-cohort calibration

The separate CPU diagnostic preserved whole vectors, all 53 projection correlations and exact ECDF ties. Its 127 balanced label permutations per pair gave candidate-versus-control randomized max53-KS p-values 0.2344, 0.1016 and 0.1328 (resolution 1/128). This is a coarse random-label diagnostic, not an equivalence test, population guarantee or flag waiver. Original CDF investigations and the separate tail-sensitive SD comparison remain unchanged. All six comparisons and their Monte Carlo intervals remain in maxks-permutation.json; the independent numeric review remains distribution_investigation_required.
"""
    append_section(report_path, section)
    return {key: meta[key] for key in
            ("request_speedup", "wrapper_speedup", "preparation_seconds",
             "prediction_and_sampling_seconds", "refill_preparation_seconds",
             "numeric_review_status", "render_checks")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Saved-artifact root containing results/.")
    parser.add_argument("--output", required=True, help="Existing report directory from report_execution_performance.py.")
    print(json.dumps(render(**vars(parser.parse_args())), indent=2))
