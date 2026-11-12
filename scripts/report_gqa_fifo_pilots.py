#!/usr/bin/env python3
"""Render the two authenticated FIFO 50k pilots and separate host-trim proof.

Built with PriorLabs-TabPFN. Saved-artifact CPU reporting only: no model/CUDA
calls, reference integrations, inference edits or qualification changes.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import importlib.util
import json
from pathlib import Path
import shutil
import statistics

import numpy as np

from posteriorforge.artifacts import sha256_file, write_json
from posteriorforge.reference import load_reference_distribution

REPO = Path(__file__).resolve().parents[1]
PHASE = Path("results/performance-first-v1")
PRIMARY = Path("results/cdf-arithmetic-v1/context-4096/seed-20261111-case-0")
REFERENCE = Path("results/reference-grids-v1/case-0")
OLD_GQA = PHASE / "gqa-hybrid-fp32-batch-1024/rep-0/seed-20261111-case-0"
PRIMARY_SHA = "a0114fd74721cf9bed31e9131045e841f262db123c8f849e5cb3f15814a14f4b"
RUNNER_SHA = "23ccad8b5bbcdd7a48340b349d515ca870763359cfeb71333e02fc2b6a8aca3b"
TRIM_PATH = PHASE / "gqa-host-trim-qualification-01/manifest.json"
TRIM_SHA = "18179d2d2161c7713050be99c03d3b8c8a92d82cd1b310fb69134e112a3bf8ba"
PILOTS = {
    1024: {"manifest": "c0937b4b08c23d78829695b1207775cefbd3d161b592cb0139d25241927f1e42",
           "samples": "1bc81c9b2ed04b15c74c1a63fd687aff1a0a86b08d8cd3f915b9f93cdc7cd362",
           "review": "6d93b0243c8203455b0c66dbcdef7ea2832d8ac49019bc6c7ad3d8b1dfad0b84"},
    10000: {"manifest": "bc8316e456bd4757ef6a2d01839a15cb316aadd8d526f0e7b369bdbbf2a6167d",
            "samples": "95c43d89a50030808f625c7a3269eeed302452f6d88823f1a250ebf87f258a6d",
            "review": "6d33770b729662abceaeb3e39729101bec056fa5974bc8f4d3b9f44b95e80be1"},
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_json(path):
    return json.loads(Path(path).read_text())


def scalars(value):
    return {key: item for key, item in value.items()
            if not isinstance(item, (list, dict))}


def binding(value, root):
    result = dict(value)
    result["path"] = str(Path(result["path"]).resolve().relative_to(root))
    return result


def table(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def trim_summary(root):
    path = root / TRIM_PATH
    require(sha256_file(path) == TRIM_SHA, "host-trim proof checksum mismatch")
    raw = load_json(path)
    require(raw["qualified"] and raw["full21_gqa_host_trim_qualified"] and
            raw["posterior_vectors_generated"] == 0 and not raw["failed_predicates"],
            "completed bounded host-trim proof required")
    require(all(raw["final_predicates"].values()) and
            len(raw["stage_results"]) == 21 and
            all(row["passed"] for row in raw["stage_results"]),
            "host-trim predicates/stage coverage changed")
    require(len(raw["migrations"]) == 105 and
            all(row["comparison"]["passed"] for row in raw["migrations"]),
            "all105 full-hash migration comparisons required")
    counters = raw["host_execution"]["counters"]
    require(counters["actual_prepare_miss_count"] == 21 and
            counters["cache_hit_count"] == 42 and
            raw["cyclic_full_scans"]["total_prepare_misses_after_cold_preload"] == 0,
            "host-trim warm hits/refits changed")
    require(raw["memory_contract"]["rss_limit_bytes"] == 100 * 2**30 and
            raw["memory_contract"]["safety_slack_bytes"] == 8 * 2**30 and
            raw["allocator_policy"]["rss_bytes_subtracted"] == 0,
            "unchanged100GiB/8GiB admission guard required")
    require(len(raw["trim_events"]) == 1, "one actual trim event required")
    event = raw["trim_events"][0]
    require(event["rss_reclaimed_by_trim_bytes"] == 4052291584 and
            event["active_state_comparison"]["passed"] and event["rng_preserved"] and
            event["retention_enabled_after_parent"],
            "actual RSS reclaim or trim invariants changed")
    kept_event = scalars(event)
    for key in ("memory_before_gc", "memory_immediately_before_trim",
                "memory_immediately_after_trim", "trigger_projection",
                "parent_projection_after_trim", "active_state_comparison",
                "live_cache_storage_before", "live_cache_storage_after"):
        kept_event[key] = scalars(event[key])
    return {
        "schema": "posteriorforge.host_trim_proof_extract.v1",
        "attribution": raw["attribution"], "source_path": str(TRIM_PATH),
        "source_manifest_sha256": TRIM_SHA, "status": raw["status"],
        "full21_gqa_host_trim_qualified": True, "cache_only_full21_qualified": False,
        "posterior_vectors_generated": 0, "not_a_50000_vector_host_timing": True,
        "stage_coverage": [row["stage_index"] for row in raw["stage_results"]],
        "final_predicates": raw["final_predicates"], "sources": raw["sources"],
        "memory_contract": raw["memory_contract"],
        "allocator_policy": raw["allocator_policy"],
        "counters": counters,
        "warm_scans": scalars(raw["cyclic_full_scans"]),
        "migration_count": 105,
        "migration_operations": dict(Counter(row["operation"] for row in raw["migrations"])),
        "migrations": [{**scalars(row), "comparison": scalars(row["comparison"])}
                       for row in raw["migrations"]],
        "trim_event": kept_event,
        "peak_rss_bytes": raw["peak_rss_bytes"],
        "initial_global_cuda_allocated_bytes": raw["initial_global_cuda_allocated_bytes"],
        "final_global_cuda_allocated_bytes": raw["final_retention"]["global_cuda_allocated_bytes"],
        "retained_stages": raw["final_retention"]["retained_stages"],
        "final_retention_enabled": raw["final_retention"]["retention_enabled"],
        "scope": raw["scope"],
        "raw_cache_storage_json_included": False,
    }


def review_extract(review, expected, path, candidate_binding, primary_binding):
    return {
        "schema": "posteriorforge.fifo_pilot_review_extract.v1",
        "source_review_sha256": expected["review"], "source_review_path": str(path),
        "attribution": review["attribution"], "candidate": candidate_binding,
        "primary": primary_binding, "source": review["source"],
        "sample_integrity": review["sample_integrity"],
        "absolute_physical_assessment": review["absolute_physical_assessment"],
        "relative_distribution_review": review["relative_distribution_review"],
        "control_repetition_context": {
            "distinct_posterior_rng_repetitions": 3,
            "primary_old_control_is_not_a_fourth_independent_repetition": True,
            "flags_suppressed_or_thresholds_changed": False,
            "cohorts": [{
                "repetition": row["repetition"],
                "manifest_sha256": row["manifest_sha256"],
                "samples_sha256": row["samples_sha256"],
                "identical_to_primary_payload": row["identical_to_primary_payload"],
                "candidate_vs_this_control": row["candidate_vs_this_control"],
                "control_vs_primary": row["control_vs_primary"],
                "control_physical_status": row["control_absolute_physical_assessment"]["status"],
            } for row in review["control_repetition_context"]["cohorts"]],
        },
        "pointwise_qualification": {
            key: value for key, value in review["pointwise_qualification"].items()
            if key not in ("qualification", "manifest_file")},
        "rejection_review": review["rejection_review"],
        "numeric_review_status": review["numeric_review_status"],
        "interpretation": review["interpretation"],
        "original_gate_unchanged": review["original_gate_unchanged"],
        "absolute_physical_gate_required_for_performance": False,
        "raw_trace_json_included": False,
    }


def render(*, root, output, package_output=None):
    root = Path(root).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    require(not output.exists(), "use a fresh output directory")
    spec = importlib.util.spec_from_file_location(
        "fifo_base_report", REPO / "scripts/report_execution_performance.py")
    plots = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plots)
    generic_sha = sha256_file(REPO / "scripts/report_execution_performance.py")
    primary, primary_manifest, primary_binding = plots.load_run(root / PRIMARY)
    require(primary_binding["manifest_sha256"] == PRIMARY_SHA,
            "immutable corrected primary required")
    old, old_manifest, old_binding = plots.load_run(root / OLD_GQA)
    require(old_binding["manifest_sha256"] ==
            "87a54e1627cf8d48e9a5bb00daaccdc32671838b9b9573f2f5a81da0e4506abc",
            "immutable earlier GQA pilot required")
    controls = []
    for repetition in range(3):
        values, manifest, item = plots.load_run(
            root / PHASE / f"controls/rep-{repetition}/seed-20261111-case-0")
        controls.append((values, manifest, item))
    arrays, reference = load_reference_distribution(root / REFERENCE)
    require(reference["binding"]["observation_sha256"] == primary_manifest["query"]["sha256"],
            "reference observation mismatch")
    trim = trim_summary(root)
    loaded = []
    for batch, expected in PILOTS.items():
        relative = PHASE / f"gqa-fifo-fp32-batch-{batch}/rep-0/seed-20261111-case-0"
        values, manifest, item = plots.load_run(root / relative)
        require(item["manifest_sha256"] == expected["manifest"] and
                item["samples_sha256"] == expected["samples"] and
                manifest["runner_source_sha256"] == RUNNER_SHA,
                "FIFO candidate/runner binding mismatch")
        review_path = relative / "independent-review-v4/execution-review.json"
        require(sha256_file(root / review_path) == expected["review"],
                "FIFO independent-review-v4 checksum mismatch")
        review = load_json(root / review_path)
        for name, target in (("candidate", item), ("control", primary_binding)):
            require(all(review[name][key] == target[key]
                        for key in ("manifest_sha256", "samples_sha256")),
                    "FIFO review candidate/control binding mismatch")
        require(review["sample_integrity"]["passed"] and
                review["absolute_physical_assessment"]["status"] == "model_failure",
                "retained integrity or physical assessment changed")
        for repetition, (_, _, target) in enumerate(controls):
            row = review["control_repetition_context"]["cohorts"][repetition]
            require(all(row[key] == target[key] for key in
                        ("manifest_sha256", "samples_sha256")),
                    "FIFO review control-repetition binding mismatch")
        events = manifest["stage_events"]
        fifo = manifest["proposal_buffer"]
        require(len(events) == 21 and [row["stage_index"] for row in events] == list(range(21)) and
                {row["proposal_round"] for row in events} == {1} and
                {row["proposed_rows"] for row in events} == {51200} and
                manifest["proposal_wave_sizes"] == [51200],
                "one full51200-vector wave and21 preparations required")
        require(fifo["generated_count"] == 51200 and fifo["generated_wave_count"] == 1 and
                fifo["all_rows_are_complete_vectors"] and not fifo["partial_vector_filtering"] and
                not fifo["row_repair_or_reordering"] and not fifo["support_filtering_performed_by_buffer"] and
                fifo["rng_draws_by_buffer"] == 0 and fifo["source_unchanged"] and
                fifo["unused_rows_discarded_only_after_original_sampler_completion"],
                "FIFO complete-vector/order/support policy changed")
        checked = manifest["rejection"]["proposed"]
        require(manifest["rejection"]["support_calls"] == 2 and
                manifest["rejection"]["accepted_before_final_trim"] == 50150 and
                manifest["rejection"]["returned"] == 50000 and
                fifo["remaining_unused_buffer_count"] == 51200 - checked,
                "original sampler/FIFO accounting changed")
        loaded.append((batch, relative, values, manifest, item, review, review_path))

    plots.render(
        control=root / PRIMARY, reference=root / REFERENCE, output=output,
        runs=[f"Earlier GQA 50k={root / OLD_GQA}"] +
             [f"FIFO {batch:,} (50k)={root / relative}" for batch, relative, *_ in loaded],
        timing_runs=[f"Control rep {rep}={root / PHASE / f'controls/rep-{rep}/seed-20261111-case-0'}"
                     for rep in range(3)],
        tail_diagnostic=root / PHASE / "control-tail-diagnostic/science-control-repeat-tails.json")
    base_data = load_json(output / "figure-data.json")
    require(len(base_data["plotted_posterior_cohorts"]) == 4 and len(base_data["timings"]) == 7,
            "four posterior cohorts and seven distinct timing measurements required")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    figures = {}
    def save(fig, name):
        fig.savefig(output / name, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        figures[name] = sha256_file(output / name)

    request_median = statistics.median(m["timings"]["posterior_request_seconds"]
                                      for _, m, _ in controls)
    wrapper_median = statistics.median(m["wrapper_total_seconds"] for _, m, _ in controls)
    summaries = []
    primary_projection = primary @ arrays["directions"].T
    family = []
    cdf_rows = []
    moment_rows = []
    event_rows = []
    witnesses = {}
    for batch, relative, values, manifest, item, review, review_path in loaded:
        request = manifest["timings"]["posterior_request_seconds"]
        total = manifest["timings"]["total_seconds"]
        wrapper = manifest["wrapper_total_seconds"]
        prep = sum(row["prepare_seconds"] for row in manifest["stage_events"])
        predict = sum(row["predict_and_sample_seconds"] for row in manifest["stage_events"])
        other = request - prep - predict
        require(min(other, total - request, wrapper - total) >= 0, "timing intervals do not reconcile")
        fifo = manifest["proposal_buffer"]
        summary = {
            "query_batch_size": batch, "binding": binding(item, root),
            "review_sha256": PILOTS[batch]["review"],
            "runner_source_sha256": manifest["runner_source_sha256"],
            "git_revision_at_run": manifest["git_revision"],
            "posterior_request_seconds": request, "runner_total_seconds": total,
            "wrapper_seconds": wrapper,
            "runner_pre_request_unattributed_seconds": total - request,
            "outer_wrapper_residual_seconds": wrapper - total,
            "preparation_seconds": prep, "prediction_and_sampling_seconds": predict,
            "other_request_seconds": other,
            "request_speedup_vs_control_median": request_median / request,
            "wrapper_speedup_vs_control_median": wrapper_median / wrapper,
            "request_speedup_vs_earlier_gqa": old_manifest["timings"]["posterior_request_seconds"] / request,
            "nominal_request_gpu_hours": request / 3600,
            "memory": manifest["memory"], "prepared_stages": 21,
            "generated_vectors": 51200, "generated_waves": 1,
            "original_sampler_checked_vectors": manifest["rejection"]["proposed"],
            "support_calls": 2, "accepted_before_trim": 50150, "accepted_final_trim": 150,
            "returned_vectors": 50000,
            "unexamined_fifo_remainder": fifo["remaining_unused_buffer_count"],
            "proposal_buffer": fifo, "proposal_buffer_policy": manifest["proposal_buffer_policy"],
            "gqa_prediction_checks": manifest["gqa_prediction_checks"],
            "relative_distribution_review": review["relative_distribution_review"],
            "absolute_physical_status": "model_failure",
            "numeric_review_status": review["numeric_review_status"],
            "one_timing_repetition_only": True,
        }
        summaries.append(summary)
        write_json(output / f"fifo-{batch}-review-extract.json", review_extract(
            review, PILOTS[batch], review_path, binding(item, root), binding(primary_binding, root)))
        projected = values @ arrays["directions"].T
        for axis in range(53):
            left = np.sort(projected[:, axis])
            right = np.sort(primary_projection[:, axis])
            coordinate = np.unique(np.r_[left, right])
            delta = (np.searchsorted(left, coordinate, side="right") -
                     np.searchsorted(right, coordinate, side="right")) / 50000
            at = int(np.argmax(np.abs(delta)))
            distance = float(abs(delta[at]))
            require(abs(distance - review["relative_distribution_review"]["cdf_difference"][axis]) < 1e-12,
                    "exact primary ECDF disagrees with independent review")
            name = primary_manifest["parameter_order"][axis] if axis < 21 else f"joint{axis - 21}"
            row = {"batch_size": batch, "axis_index": axis, "name": name,
                   "ks": distance, "signed_maximizer": float(delta[at]),
                   "projection_coordinate": float(coordinate[at]), "flag": distance > .01}
            family.append(row)
            if row["flag"]:
                witnesses[f"b{batch}_axis{axis}_coordinate"] = coordinate
                witnesses[f"b{batch}_axis{axis}_signed_difference"] = delta
        comparisons = [("Primary", review["relative_distribution_review"])] + [
            (f"Control rep {row['repetition']}", row["candidate_vs_this_control"])
            for row in review["control_repetition_context"]["cohorts"]]
        for control_label, compared in comparisons:
            for axis, distance in enumerate(compared["cdf_difference"]):
                cdf_rows.append({"batch_size": batch, "control": control_label,
                                 "axis_index": axis,
                                 "name": primary_manifest["parameter_order"][axis] if axis < 21 else f"joint{axis - 21}",
                                 "ks": distance, "flag": distance > .01})
            for axis, parameter in enumerate(primary_manifest["parameter_order"]):
                mean = compared["mean_difference_reference_sd"][axis]
                sd = compared["sd_fraction_difference"][axis]
                moment_rows.append({"batch_size": batch, "control": control_label,
                                    "parameter_index": axis, "parameter": parameter,
                                    "mean_change_reference_sd": mean, "sd_fraction_change": sd,
                                    "mean_flag": mean > .05, "sd_flag": sd > .05})
        for row in manifest["stage_events"]:
            event_rows.append({"batch_size": batch, **{key: row[key] for key in (
                "stage_index", "proposal_round", "proposed_rows",
                "prepare_seconds", "predict_and_sample_seconds",
                "fit_stock_cache_seconds", "resident_move_seconds", "uncached_fit_seconds")}})
    table(output / "fifo-primary-cdf-family.csv", family)
    table(output / "fifo-all-control-cdf-metrics.csv", cdf_rows)
    table(output / "fifo-moment-comparisons.csv", moment_rows)
    table(output / "fifo-stage-events.csv", event_rows)
    np.savez_compressed(output / "fifo-signed-cdf-witnesses.npz", **witnesses)
    write_json(output / "host-trim-proof-extract.json", trim)

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    for ax, (batch, _, _, manifest, *_) in zip(axes, loaded):
        x = np.arange(21)
        p = np.array([row["prepare_seconds"] for row in manifest["stage_events"]])
        s = np.array([row["predict_and_sample_seconds"] for row in manifest["stage_events"]])
        ax.bar(x, p, color="#315c99", label="Preparation")
        ax.bar(x, s, bottom=p, color="#238e8b", label="Prediction + sampling")
        ax.set(title=f"FIFO batch {batch:,}: one 51,200-vector wave, 21 stage preparations",
               ylabel="Measured seconds", ylim=(0, max(30, float(max(p+s))*1.15)))
        ax.grid(axis="y", alpha=.15)
    axes[0].legend(frameon=False, ncol=2)
    axes[-1].set(xlabel="Autoregressive stage", xticks=np.arange(21))
    fig.suptitle("Built with PriorLabs-TabPFN: actual FIFO 50k stage costs", fontsize=14)
    fig.tight_layout()
    save(fig, "fifo-stage-costs.png")

    fig, ax = plt.subplots(figsize=(13, 5.2))
    labels = [f"FIFO {row['query_batch_size']:,}" for row in summaries]
    segments = [
        ("runner_pre_request_unattributed_seconds", "UNATTRIBUTED pre-request", "#c18d36"),
        ("preparation_seconds", "Request: preparation", "#315c99"),
        ("prediction_and_sampling_seconds", "Request: prediction + sampling", "#238e8b"),
        ("other_request_seconds", "Other request work", "#98a5af"),
        ("outer_wrapper_residual_seconds", "Outer wrapper residual", "#b49abf"),
    ]
    left = np.zeros(2)
    for key, label, color in segments:
        values = np.array([row[key] for row in summaries])
        ax.barh(labels, values, left=left, color=color, label=label)
        left += values
    ax.set(xlabel="Measured seconds; stacked extent equals wrapper wall time",
           title="Request and wrapper are separate measurements")
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(.5, -.18))
    ax.grid(axis="x", alpha=.15)
    for i, row in enumerate(summaries):
        ax.text(row["wrapper_seconds"] + 4, i,
                f"request {row['posterior_request_seconds']:.3f}s\nwrapper {row['wrapper_seconds']:.3f}s",
                va="center", fontsize=9)
    ax.set_xlim(0, max(row["wrapper_seconds"] for row in summaries) * 1.25)
    fig.tight_layout()
    save(fig, "fifo-request-wrapper-breakdown.png")

    fig, ax = plt.subplots(figsize=(14, 4.8))
    x = np.arange(53)
    for offset, batch, color in [(-.19, 1024, "#315c99"), (.19, 10000, "#b24d70")]:
        rows = [row for row in family if row["batch_size"] == batch]
        ax.bar(x+offset, [row["ks"] for row in rows], width=.38,
               color=color, label=f"FIFO {batch:,}")
    ax.axhline(.01, color="#c56a19", ls="--", label="Unchanged investigation threshold")
    ax.axvline(20.5, color="#7d8791", lw=.8)
    ax.set_xticks([0, 7, 8, 19, 25, 40, 52], ["M0", "M7", "M8", "M19", "J4", "J19", "J31"], rotation=90)
    ax.set(ylabel="Exact two-sample ECDF distance", xlabel="21 marginals + 32 fixed joint projections",
           title="Built with PriorLabs-TabPFN: every primary-control comparison retained")
    ax.legend(frameon=False, ncol=3)
    ax.grid(axis="y", alpha=.15)
    fig.tight_layout()
    save(fig, "fifo-complete-cdf-family.png")

    flagged = [7, 8, 19, 25, 40]
    fig, axes = plt.subplots(5, 2, figsize=(14, 17))
    for j, axis in enumerate(flagged):
        name = primary_manifest["parameter_order"][axis] if axis < 21 else f"joint {axis-21}"
        for col, batch in enumerate(PILOTS):
            ax = axes[j, col]
            key = f"b{batch}_axis{axis}"
            ax.step(witnesses[key+"_coordinate"], witnesses[key+"_signed_difference"],
                    where="post", color="#a63c60", lw=1)
            for threshold in (-.01, .01):
                ax.axhline(threshold, color="#c56a19", ls="--", lw=.8)
            ax.axhline(0, color="#7d8791", lw=.6)
            distance = next(row["ks"] for row in family
                            if row["batch_size"] == batch and row["axis_index"] == axis)
            ax.set(title=f"FIFO {batch:,}; {name}; max|difference|={distance:.5f}",
                   xlabel="Physical projection coordinate", ylabel="FIFO ECDF minus primary ECDF")
            ax.grid(alpha=.15)
    fig.suptitle("Built with PriorLabs-TabPFN: retained flags M7/M8/M19 and J4/J19\n"
                 "Every signed curve uses 50,000 complete vectors; dashed lines retain ±0.01",
                 fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .97))
    save(fig, "fifo-signed-cdf-flags.png")

    cohorts = [("Primary", primary), ("Earlier GQA", old)] + [
        (f"FIFO {batch:,}", values) for batch, _, values, *_ in loaded]
    density_data = []
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    colors = ["#596573", "#d26c28", "#238e8b", "#8757ad"]
    for axis, ax in zip((7, 8, 19), axes):
        coordinate = arrays["queries"] / arrays["directions"][axis, axis]
        cdf = arrays["cdfs"][axis]
        lo = min(float(np.quantile(values[:, axis], .001)) for _, values in cohorts)
        hi = max(float(np.quantile(values[:, axis], .999)) for _, values in cohorts)
        lo = min(lo, float(coordinate[min(np.searchsorted(cdf, .0005), len(cdf)-1)]))
        hi = max(hi, float(coordinate[min(np.searchsorted(cdf, .9995), len(cdf)-1)]))
        first = max(0, np.searchsorted(coordinate, lo)-1)
        last = min(len(coordinate)-1, np.searchsorted(coordinate, hi)+1)
        select = np.unique(np.linspace(first, last, min(75, last-first+1)).astype(int))
        edges = coordinate[select]
        width = np.diff(edges)
        ax.stairs(np.diff(cdf[select])/width, edges, color="#225ca7",
                  lw=1.6, label="Retained reference mass / width")
        displayed = {}
        for color, (label, values) in zip(colors, cohorts):
            counts, _ = np.histogram(values[:, axis], bins=edges)
            ax.stairs(counts/(50000*width), edges, color=color, lw=1.2, label=label)
            displayed[label] = int(counts.sum())
        ax.set(title=f"M{axis}: {primary_manifest['parameter_order'][axis]}",
               xlabel="Physical value; central display", ylabel="Probability / bin width")
        ax.grid(alpha=.15)
        density_data.append({"axis": axis, "window": [float(edges[0]), float(edges[-1])],
                             "normalization_count": 50000, "displayed_counts": displayed})
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, ncol=3, frameon=False, loc="lower center")
    fig.suptitle("Ordinary posterior densities at the flagged marginals\n"
                 "Full-count normalization retains mass outside the displayed window", fontsize=13)
    fig.tight_layout(rect=(0, .11, 1, .89))
    save(fig, "fifo-flagged-marginal-densities.png")

    event = trim["trim_event"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    before = event["memory_immediately_before_trim"]["rss_bytes"] / 2**30
    after = event["memory_immediately_after_trim"]["rss_bytes"] / 2**30
    axes[0].bar(["Before actual trim", "After actual trim"], [before, after],
                color=["#b24d70", "#238e8b"])
    axes[0].axhline(trim["peak_rss_bytes"]/2**30, color="#596573", ls="--", label="Proof peak RSS")
    axes[0].set(ylabel="Measured RSS GiB", ylim=(0, 105),
                title=f"One actual trim reclaimed {event['rss_reclaimed_by_trim_bytes']/1e9:.3f}GB")
    projections = [event["trigger_projection"], event["parent_projection_after_trim"]]
    values = [max(row["projected_rss_with_slack_bytes"],
                  row["projected_cgroup_with_slack_bytes"])/2**30 for row in projections]
    axes[1].bar(["Rejected before trim", "Admitted after trim"], values,
                color=["#b24d70", "#238e8b"])
    axes[1].axhline(100, color="#c56a19", ls="--", label="Unchanged 100 GiB guard")
    axes[1].set(ylabel="Max projected RSS/cgroup GiB\nincluding incoming CUDA storage + 8 GiB slack",
                ylim=(0, 110), title="Original admission is rerun without an RSS subtraction")
    for ax in axes:
        ax.grid(axis="y", alpha=.15)
        ax.legend(frameon=False)
    fig.suptitle("Separate bounded host-trim proof: 21 stages, 42 warm hits, 0 refits\n"
                 "105 full-hash migrations; zero posterior vectors generated", fontsize=14)
    fig.tight_layout()
    save(fig, "host-trim-proof-memory.png")

    meta = {
        "schema": "posteriorforge.gqa_fifo_pilots_report.v1",
        "attribution": "Built with PriorLabs-TabPFN",
        "report_source_sha256": sha256_file(__file__),
        "generic_renderer_source_sha256": generic_sha,
        "primary": binding(primary_binding, root), "earlier_gqa": binding(old_binding, root),
        "reference_manifest_sha256": sha256_file(root / REFERENCE / "manifest.json"),
        "control_request_median_seconds": request_median,
        "control_wrapper_median_seconds": wrapper_median,
        "controls": [{"binding": binding(item, root),
                      "request_seconds": manifest["timings"]["posterior_request_seconds"],
                      "wrapper_seconds": manifest["wrapper_total_seconds"]}
                     for _, manifest, item in controls],
        "pilots": summaries, "host_trim_manifest_sha256": TRIM_SHA,
        "host_trim_is_not_a_50000_vector_timing": True,
        "density_display": density_data, "figures": figures,
        "signed_curves_sha256": sha256_file(output / "fifo-signed-cdf-witnesses.npz"),
        "all53_primary_ecdf_review_match_tolerance": 1e-12,
        "no_permutation_diagnostic_transferred": True,
        "no_population_equivalence_or_global_calibration_claim": True,
        "no_rows_or_flags_removed": True, "overall_runtime_phase_complete": False,
    }
    write_json(output / "fifo-pilots-summary.json", meta)
    lines = ["## Completed FIFO 50k pilots", "",
             "Built with PriorLabs-TabPFN. Both pilots return 50,000 finite, complete, prior-supported vectors. "
             "Each is a single timing repetition; overall runtime-phase completion remains pending.", "",
             "| FIFO batch | Request s | Wrapper s | Preparation s | Prediction + sampling s | Request speedup vs 3-control median |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in summaries:
        lines.append(f"| {row['query_batch_size']} | {row['posterior_request_seconds']:.6f} | "
                     f"{row['wrapper_seconds']:.6f} | {row['preparation_seconds']:.6f} | "
                     f"{row['prediction_and_sampling_seconds']:.6f} | "
                     f"{row['request_speedup_vs_control_median']:.3f}x |")
    lines += ["", f"The contemporaneous control request median uses all three distinct measurements: "
              f"{request_median:.6f}s. The historical primary retains its own timing bar; its identical "
              "control-repetition 0 posterior payload is omitted only from the curves.", "",
              "Each FIFO provider generated 51,200 complete proposals in one wave and prepared 21 stages. "
              "The unchanged original sampler made two support calls and checked 50,457/50,458 vectors, "
              "accepted 50,150 before its final 150-row trim and returned 50,000. The unexamined FIFO "
              "remainders were 743/742; these are distinct from the accepted rows removed by final truncation. "
              "The FIFO buffer performed no support filtering, row repair, row reordering or RNG draws. "
              "Lookahead changes seeded row mapping; no rowwise-equivalence claim is made.", "",
              "![Actual one-wave FIFO stage preparation and prediction costs](fifo-stage-costs.png)", "",
              "The 10,000-row pilot has 109.467749 seconds in the runner before the posterior request. "
              "This interval is **UNATTRIBUTED pre-request work**, not post-sampling work. The outer "
              "wrapper adds a separate 0.453703-second residual. The 1,024-row equivalents are 10.491280 "
              "and0.511076 seconds. No cause is assigned to the differing startup intervals.", "",
              "![Request and wrapper interval decomposition](fifo-request-wrapper-breakdown.png)", "",
              "Peak allocated/reserved GPU memory is 5.38686/29.8926 GiB for FIFO 1024 and "
              "8.32160/38.9963GiB for FIFO 10000. Reservation is not live tensor storage. Request "
              "wall-product GPU-hours differ from scheduler allocation accounting.", "",
              "## Retained distribution investigations", "",
              "Both primary comparisons retain flags at M7, M8 and M19 and joint projections J4 and J19. "
              "The maximum exact two-sample ECDF distances are 0.01368 and 0.01370. Mean changes are at "
              "most 0.0220494/0.0219952 reference SD and SD fractional changes 1.46949%/1.45697%. "
              "The 0.01 CDF, 0.05 reference-SD mean and 5% SD investigation thresholds are unchanged. "
              "Every control-repetition comparison and all 53 CDF plus 21 moment metrics are retained "
              "in the review extracts and CSVs. The original physical assessment remains **model_failure** "
              "and the numeric status **distribution_investigation_required**. Performance-first "
              "authorization retains the physical failure separately; no row, tail or flag is removed.", "",
              "![Complete 53-axis primary-control comparisons](fifo-complete-cdf-family.png)", "",
              "![Signed witnesses for every retained primary-control flag](fifo-signed-cdf-flags.png)", "",
              "The ordinary densities below use the actual saved complete cohorts and retained reference "
              "bin masses. Counts divide by 50,000 even when a central display window excludes tail mass. "
              "No new reference integration, density correction or sample clipping is performed.", "",
              "![Ordinary posterior densities at flagged marginals](fifo-flagged-marginal-densities.png)", "",
              "These finite-cohort checks do not establish population equivalence or global calibration. "
              "No earlier randomized p-values or permutation diagnostic are transferred to these pilots.", "",
              "## Separate successful host-trim proof", "",
              "The authenticated host-trim screen passed all 21 stages, two full warm scans with 42 warm "
              "cache hits and zero refits after cold preload, and 105 full-storage-hash migrations "
              "preserving tensor identity, layout and aliases. One real trim reclaimed 4.052291584 GB "
              "(decimal) without changing live cache bytes or RNG. The original 100 GiB admission limit "
              "and 8 GiB transient margin were rerun unchanged; no RSS subtraction, eviction, LRU or "
              f"guard bypass was used. Peak process RSS was {trim['peak_rss_bytes']/2**30:.5f} GiB. All 21 cached stages ended on "
              "CPU, with 8.51968 MB of known global CUDA allocation equal to the initial residual.", "",
              "![Bounded host-trim RSS and unchanged admission proof](host-trim-proof-memory.png)", "",
              "This is a fixed-conditional migration and retention proof with **zero posterior vectors "
              "generated**. It is **not a 50k host timing measurement**, and its gains are not added to "
              "the FIFO 50k timings. The cache-only full21 screen remains failed. The full raw host proof "
              "and large independent-review traces remain outside Git; compact extracts bind their "
              "source hashes. All generated figures/tables and source bindings are checksummed."]
    report = output / "report.md"
    report.write_text(report.read_text().rstrip() + "\n\n" + "\n".join(lines) + "\n")
    readme = f"""# FIFO 50k pilots and bounded host-trim evidence

Built with PriorLabs-TabPFN. [NOTICE](NOTICE) and the copied [MIT license](LICENSE) retain attribution; bundled upstream terms remain in the repository.

FIFO 1024 completed its50,000-vector request in 410.963418 s (wrapper 421.965773 s); FIFO 10000 took 448.124877 s (wrapper 558.046330 s). Each generated one51,200-vector wave with 21 preparations. All three contemporaneous control timing measurements, the earlier 607.081521 s GQA request and both FIFO pilots remain visible below. Each optimized configuration has one timing repetition.

![Actual request costs across all control and pilot timings](runtime-and-request-cost.png)

The 10,000-row runner pre-request interval is 109.467749 s and remains UNATTRIBUTED; it is not post-sampling time. Request and wrapper values have distinct scopes.

![Request/wrapper measurements with unattributed pre-request work](fifo-request-wrapper-breakdown.png)

Both cohorts retain M7/M8/M19 and J4/J19 CDF investigation flags; absolute physical status remains model_failure. Complete 53-axis CDF metrics and 21-parameter mean/SD comparisons against primary and all three controls are in the CSVs and hash-bound review extracts. No rows or flags were removed, and no old randomized p-values, equivalence or global-calibration claim is made.

![All 53 primary comparisons](fifo-complete-cdf-family.png)

![Actual ordinary marginal densities](fifo-flagged-marginal-densities.png)

The separate host-trim proof passed21 stages,42 warm hits/0 refits and 105 full-hash migrations. One actual trim reclaimed 4.052 GB while preserving live storage; the 100 GiB/8 GiB guard remained unchanged. Peak RSS was 89.94 GiB and known global GPU residual 8.52 MB. This proof generated zero posterior vectors and is not a 50k host timing benchmark.

![Separate bounded host-trim memory proof](host-trim-proof-memory.png)

[report.md](report.md) embeds all figures. [fifo-pilots-summary.json](fifo-pilots-summary.json), [fifo-1024-review-extract.json](fifo-1024-review-extract.json), [fifo-10000-review-extract.json](fifo-10000-review-extract.json) and [host-trim-proof-extract.json](host-trim-proof-extract.json) retain the metrics and custody. [checksums.json](checksums.json) hashes every other package file. Raw samples, full review/cache JSON and the derived signed-curve witness NPZ remain outside Git.

The report script SHA256 is `{meta['report_source_sha256']}`. Review SHA256 values are `{PILOTS[1024]['review']}` and `{PILOTS[10000]['review']}`; the full host proof SHA256 is `{TRIM_SHA}`. Runner source SHA256 is `{RUNNER_SHA}`. Existing controls are linked from the [controls raw release](https://github.com/BrieucB/PosteriorForge/releases/tag/performance-first-v1-controls); new raw publication remains a separate pending step. Overall runtime-phase completion remains pending.

## Reproduce from saved artifacts

Use the verified Python 3.11 reference environment with NumPy, SciPy and Matplotlib. Preserve the original results layout: gate-final.json, banks-v1, authenticated parent/refined references, corrected primary cohort, all three controls, old GQA cohort, both FIFO cohorts and independent-review-v4 files, control-tail diagnostic, and the full host-trim manifest. The generic renderer is reused unchanged; no model/GPU calls or reference integrations are made.

From the repository root, use fresh output/package directories:

```bash
PF_ROOT=/path/to/unpacked-artifacts
PF_PYTHON=/path/to/verified-python3.11
PF_OUTPUT=/path/to/new-fifo-report
PF_PACKAGE=/path/to/new-portable-package
PYTHONPATH=src OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 "$PF_PYTHON" \\
  scripts/report_gqa_fifo_pilots.py --root "$PF_ROOT" --output "$PF_OUTPUT" \\
  --package-output "$PF_PACKAGE"
```

The script infers its checkout from __file__, verifies frozen sample/manifest/review/proof hashes, recomputes exact signed ECDF witnesses from all 50,000 rows, and copies a bounded package without raw NPZ or large trace files.
"""
    (output / "README.md").write_text(readme)
    for name in ("LICENSE", "NOTICE"):
        shutil.copy2(REPO / name, output / name)
    require(sha256_file(REPO / "scripts/report_execution_performance.py") == generic_sha,
            "generic renderer source changed during render")
    verify = {
        "schema": "posteriorforge.fifo_report_checks.v1",
        "source_sha256": meta["report_source_sha256"],
        "primary_cdf_axes_per_pilot": 53, "total_fifo_stage_events": 42,
        "all_control_cdf_rows": len(cdf_rows), "moment_rows": len(moment_rows),
        "posterior_curve_cohorts": 4, "timing_memory_cohorts": 7,
        "all_primary_ecdf_values_match_review_tolerance": 1e-12,
        "host_trim_source_manifest_authenticated": True,
        "no_new_model_or_gpu_calls": True,
        "parent_visual_review": "pending",
    }
    write_json(output / "render-verification.json", verify)
    checksums(output)
    if package_output:
        target = Path(package_output).expanduser().resolve()
        target.mkdir(parents=True, exist_ok=False)
        for path in output.iterdir():
            if path.is_file() and path.suffix != ".npz":
                shutil.copy2(path, target / path.name)
        checksums(target)
    return {"output": str(output), "package": str(package_output) if package_output else None,
            "source_sha256": meta["report_source_sha256"],
            "report_sha256": sha256_file(output / "report.md"),
            "figures": {**base_data["figures"], **figures}}


def checksums(folder):
    write_json(folder / "checksums.json", {
        "schema": "posteriorforge.evidence_checksums.v1", "algorithm": "sha256",
        "files": {path.name: sha256_file(path) for path in sorted(folder.iterdir())
                  if path.is_file() and path.name not in ("checksums.json", "fifo-signed-cdf-witnesses.npz")}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--package-output")
    print(json.dumps(render(**vars(parser.parse_args())), indent=2))
