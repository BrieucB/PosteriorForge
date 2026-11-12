#!/usr/bin/env python3
"""CPU-only authenticated clean FP16 pilot and host A-B-A report.

Built with PriorLabs-TabPFN. Reads saved cohorts/reviews only; no model/GPU calls,
new reference integration, producer edits or qualification changes.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
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
STRICT = PHASE / "gqa-fifo-fp32-batch-1024/rep-0/seed-20261111-case-0"
LARGE = PHASE / "gqa-fifo-fp32-batch-10000/rep-0/seed-20261111-case-0"
CLEAN = PHASE / "fp16-fifo-50k-01/rep-0/seed-20261111-case-0"
SCREEN = PHASE / "fp16-all21-approx-characterization-01/manifest.json"
HOST = PHASE / "gqa-host-session-aba-01"
CLEAN_SHA = "3546de7f867d3aeaefbbf4e1dab03d90fe867e8355506629319a6e9eed492133"
CLEAN_SAMPLES_SHA = "61d0a058233ab2ff6e12d1ad29b17ef9d80c3fe10cf1f474be4d33bd716f1f11"
REVIEW_SHAS = ["589ed262d9eb97b98b0c78690eb001edc207bbb4b258b5896ec172ae74a50aca",
               "b5e7532473a62dc88eabba5a8b32c5204c789f77bc8ce4be3389a56d33a86f93"]
SCREEN_SHA = "f01bfddb87dcd16c9bd7c9df15f482ed3ea46756b1d84f3c04f6584644a0687a"
HOST_SHA = "8d66ac1a4a8eea1284f93e66f24c4e5ab72b04f8a4ded97516dabe0d2fd62fbb"
HOST_REVIEW_SHA = "eead7c3342e3e57ee8d1d3e31576b94c76e71f2d45eef00a0807249dec413750"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_json(path):
    return json.loads(Path(path).read_text())


def scalars(value):
    return {key: item for key, item in value.items()
            if not isinstance(item, (list, dict))}


def bounded_dict(value):
    """Retain scalar decision/metric fields; large member/row lists stay in source."""
    return {key: bounded_dict(item) if isinstance(item, dict) else item
            for key, item in value.items() if not isinstance(item, list)}


def bind(value, root):
    result = dict(value)
    result["path"] = str(Path(result["path"]).resolve().relative_to(root))
    return result


def table(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def costs(root):
    ledger = load_json(root / "budget/ledger.json")
    records = []
    for job, seconds in [("5406772", 216), ("5406016", 1274), ("5406017", 717)]:
        selected = [row for row in ledger["reservations"] if str(row.get("job_id")) == job]
        require(len(selected) == 1, "completed allocation record missing/ambiguous")
        row = selected[0]
        require(row["state"] == "reconciled" and row["slurm_state"] == "COMPLETED" and
                row["accounted_gpus"] == 1 and row["accounted_elapsed_seconds"] == seconds and
                row["actual_gpu_seconds"] == seconds, "completed allocation scope changed")
        item = {key: row[key] for key in ("job_id", "reservation_id", "state", "slurm_state",
                                         "accounted_elapsed_seconds", "accounted_gpus", "actual_gpu_seconds")}
        item["accounting_record"] = {key: row["accounting_record"][key]
                                     for key in ("JobIDRaw", "ElapsedRaw", "State", "AllocTRES")}
        records.append(item)
    return {"schema": "posteriorforge.completed_allocation_extract.v1",
            "source_path": "budget/ledger.json", "cap_gpu_seconds": ledger["cap_gpu_seconds"],
            "records": records,
            "completed_records_sha256": hashlib.sha256(json.dumps(
                records, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "scope": "Completed scheduler allocations; request/wrapper wall-products are separate."}


def render(*, root, output, package_output=None):
    root = Path(root).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    require(not output.exists(), "use a fresh output directory")
    spec = importlib.util.spec_from_file_location(
        "fp16_base_report", REPO / "scripts/report_execution_performance.py")
    plots = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plots)
    generic_sha = sha256_file(REPO / "scripts/report_execution_performance.py")
    primary, pm, pb = plots.load_run(root / PRIMARY)
    strict, sm, sb = plots.load_run(root / STRICT)
    large, lm, lb = plots.load_run(root / LARGE)
    clean, cm, cb = plots.load_run(root / CLEAN)
    require(cb["manifest_sha256"] == CLEAN_SHA and cb["samples_sha256"] == CLEAN_SAMPLES_SHA,
            "immutable clean FP16 pilot required")
    require(sb["manifest_sha256"] == "c0937b4b08c23d78829695b1207775cefbd3d161b592cb0139d25241927f1e42" and
            pb["manifest_sha256"] == "a0114fd74721cf9bed31e9131045e841f262db123c8f849e5cb3f15814a14f4b",
            "immutable strict FIFO and primary controls required")
    controls = [plots.load_run(root / PHASE / f"controls/rep-{rep}/seed-20261111-case-0")
                for rep in range(3)]
    reviews = []
    for version, expected in enumerate(REVIEW_SHAS, 1):
        path = root / CLEAN / f"independent-review-v{version}/fp16-experiment-review.json"
        require(sha256_file(path) == expected, "independent FP16 review checksum mismatch")
        reviews.append(load_json(path))
    r = reviews[1]
    require(reviews[0]["comparisons"] == r["comparisons"],
            "v1/v2 numerical comparisons changed")
    require(r["manifest_sha256"] == CLEAN_SHA and not r["joint_equivalence_inferred"] and
            not r["old_permutation_pvalues_transferred"], "FP16 review binding/scope changed")
    rng_evidence = r["authentication"]["RNG_evidence"]
    require(rng_evidence["production_default_CPU_CUDA_state_hashes_recorded"] is False and
            rng_evidence["production_default_CPU_CUDA_state_hash_equality_observed"] is False,
            "clean production RNG evidence must retain its recorded scope")
    require(cm["strict_failed_stages"] == list(range(21)) and
            cm["all21_strict_screen_passed"] is False and
            cm["approximation_all21_hard_screen_passed"] is True,
            "original all 21 strict failures/hard passes changed")
    require(sha256_file(root / SCREEN) == SCREEN_SHA, "all21 characterization checksum mismatch")
    screen = load_json(root / SCREEN)
    require(screen["all21_hard_validity_passed"] and not screen["all21_strict_screen_passed"] and
            screen["strict_failed_stages"] == list(range(21)), "characterization scope changed")
    screen_rows = [{**scalars(row), "comparisons": bounded_dict(row["comparisons"]),
                    "computed_prefill_passed": row["computed_prefill_witness"]["passed"],
                    "computed_query_passed": row["computed_query_witness"]["passed"],
                    "sampler_parity": bounded_dict(row["sampler_parity"])}
                   for row in screen["stage_results"]]
    require(all(row["hard_validity_passed"] and row["rng_preserved"] and
                row["strict_failure_preserved"] and not row["strict_screen_passed"]
                for row in screen_rows), "all21 predicates not preserved")
    require(sha256_file(root / HOST / "manifest.json") == HOST_SHA,
            "host ABA session checksum mismatch")
    host = load_json(root / HOST / "manifest.json")
    host_review_path = root / HOST / "independent-review-v1/host-session-review.json"
    require(sha256_file(host_review_path) == HOST_REVIEW_SHA, "host ABA review checksum mismatch")
    hr = load_json(host_review_path)
    require(host["session_complete"] and host["completed_sampling_requests"] == 3 and
            host["returned_session_total_vectors"] == 150000 and
            host["isolation_aba"]["passed"] and len(host["isolation_aba"]["predicates"]) == 11 and
            all(host["isolation_aba"]["predicates"].values()) and
            hr["all150000_integrity_passed"] and
            hr["authentication"]["shared_footer_authenticated"],
            "complete150k ABA/source/footer proof required")
    require(host["shutdown"]["shutdown_released_all_stages"] and
            host["shutdown"]["all_member_cdf_gqa_states_restored"],
            "host shutdown restoration failed")
    visits = []
    host_csv = []
    host_abs_cdf = []
    for index, relative in enumerate(host["request_manifests"]):
        values, manifest, item = plots.load_run(root / HOST / Path(relative).parent)
        require(item["manifest_sha256"] == host["request_manifest_sha256"][relative],
                "host visit manifest binding mismatch")
        reviewed = hr["requests"][index]
        require(manifest["case"] == [0, 1, 0][index] and reviewed["sample_integrity"]["passed"] and
                reviewed["absolute_physical_assessment"]["status"] == "model_failure" and
                manifest["sources_unchanged"] and manifest["global_rng_preserved"],
                "host visit integrity/source/physical status changed")
        for name in ("default_torch_cpu", "default_torch_cuda", "numpy_global", "python_global"):
            require(manifest["rng_before"][name] == manifest["rng_after"][name],
                    "observed host global RNG hash changed")
        if index in (0, 2):
            require(np.array_equal(values, strict) and item["samples_sha256"] == sb["samples_sha256"],
                    "A array/NPZ differs from prior strict FIFO cohort")
        visits.append((values, manifest, item))
        host_csv.append({"visit": index, "case": manifest["case"],
                         "request_seconds": manifest["timings"]["posterior_request_seconds"],
                         "actual_stage_fits": manifest["timings"]["actual_stage_fit_count"],
                         "retained_stage_hits": manifest["timings"]["retained_stage_hits"],
                         "returned_vectors": manifest["rejection"]["returned"],
                         "physical_status": reviewed["absolute_physical_assessment"]["status"],
                         "same_case_unoptimized_control": reviewed["control"]["status"],
                         "global_rng_hashes_observed_preserved": True})
        absolute = reviewed["absolute_physical_assessment"]
        for axis in range(53):
            family = absolute["marginal_cdf"] if axis < 21 else absolute["joint_projection_cdf"]
            offset = axis if axis < 21 else axis - 21
            host_abs_cdf.append({"visit": index, "case": manifest["case"], "axis": axis,
                                 "absolute_cdf_lower": family["lower"][offset],
                                 "absolute_cdf_upper": family["upper"][offset]})
    require(hr["requests"][1]["control"]["status"] == "not_available" and
            hr["requests"][1]["relative_distribution_review"] is None,
            "do not invent an unoptimized same-case B comparison")
    allocated = costs(root)
    arrays, reference = load_reference_distribution(root / "results/reference-grids-v1/case-0")
    b_arrays, b_reference = load_reference_distribution(root / "results/reference-grids-v1/case-1")
    require(sha256_file(root / "results/reference-grids-v1/case-1/manifest.json") ==
            hr["references"]["1"]["grid_manifest_sha256"] and
            b_reference["binding"]["observation_sha256"] == visits[1][1]["query"]["sha256"],
            "B must use its own authenticated reference")

    plots.render(control=root / PRIMARY, reference=root / "results/reference-grids-v1/case-0",
                 output=output,
                 runs=[f"FP32 FIFO 1024={root / STRICT}", f"FP32 FIFO 10000={root / LARGE}",
                       f"FP16 approx clean={root / CLEAN}"],
                 timing_runs=[f"Control rep {rep}={root / PHASE / f'controls/rep-{rep}/seed-20261111-case-0'}"
                              for rep in range(3)],
                 tail_diagnostic=root / PHASE / "control-tail-diagnostic/science-control-repeat-tails.json")
    base = load_json(output / "figure-data.json")
    require(len(base["plotted_posterior_cohorts"]) == 4 and len(base["timings"]) == 7,
            "four curves and seven distinct timing measurements required")
    write_json(output / "allocated-cost-extract.json", allocated)
    write_json(output / "fp16-review-extract.json", {
        "schema": "posteriorforge.fp16_report_review_extract.v1",
        "source_review_v1_sha256": REVIEW_SHAS[0], "source_review_v2_sha256": REVIEW_SHAS[1],
        "v1_v2_comparison_metrics_identical": True, "candidate": bind(cb, root),
        "comparisons": r["comparisons"], "controls": r["controls"], "source": r["source"],
        "authentication": r["authentication"], "reference": r["reference"],
        "physical_failure_retained": True, "qualified_FP32": False,
        "joint_equivalence_inferred": False, "old_pvalues_transferred": False})
    write_json(output / "fp16-all21-characterization-extract.json", {
        "schema": "posteriorforge.fp16_characterization_extract.v1",
        "source_manifest_sha256": SCREEN_SHA, "sources": screen["sources"],
        "hostname": screen["hostname"], "job_id": screen["slurm_job_id"],
        "strict_failed_stages": screen["strict_failed_stages"],
        "all21_hard_validity_passed": True, "all21_strict_screen_passed": False,
        "qualified": False, "scope": screen["scope"], "stage_results": screen_rows,
        "posterior_vectors_generated": screen["posterior_vectors_generated"],
        "posterior_sampling_rng_used": screen["posterior_sampling_rng_used"],
        "rng_role": screen["rng_role"],
        "timing_order": screen["timing_order"],
        "timing_repetitions": screen["timing_repetitions"],
        "end_to_end_benchmark": False,
        "observed_RNG_preservation_scope": "this immutable characterization, not clean production hashes",
        "raw_cache_JSON_included": False})
    host_extract = {
        "schema": "posteriorforge.host_ABA_report_extract.v1",
        "source_manifest_sha256": HOST_SHA, "source_review_sha256": HOST_REVIEW_SHA,
        "runner_source_sha256": host["runner_source_sha256"],
        "adapter_source_sha256": host["adapter_source_sha256"],
        "authentication": hr["authentication"], "review_source": hr["source"],
        "reference_bindings": hr["references"], "requests": hr["requests"],
        "isolation_aba": host["isolation_aba"], "session_counters": host["session_counters"],
        "session_setup_seconds": host["session_setup_seconds"], "session_total_seconds": host["total_seconds"],
        "hostname": host["hostname"], "colocated_characterization_hostname": screen["hostname"],
        "shutdown": host["shutdown"],
        "observed_visit_RNG_hashes": [{"visit": i, "before": m["rng_before"], "after": m["rng_after"],
                                      "global_rng_preserved": m["global_rng_preserved"]}
                                     for i, (_, m, _) in enumerate(visits)],
        "visit_bindings": [bind(item, root) for _, _, item in visits],
        "all150000_integrity_passed": True, "A_arrays_and_NPZ_equal_prior_FIFO1024": True,
        "B_same_case_unoptimized_control_available": False,
        "production_migrations": "identity/layout guards; full-storage hashes belong to the qualification proof",
        "cold_delta_is_not_a_causal_retention_estimate": True}
    write_json(output / "host-aba-extract.json", host_extract)
    table(output / "host-aba-requests.csv", host_csv)
    table(output / "host-aba-physical-cdf-bounds.csv", host_abs_cdf)
    table(output / "fp16-all21-status.csv", [scalars(row) for row in screen_rows])
    event_rows = [{key: row[key] for key in ("stage_index", "proposed_rows", "prepare_seconds",
                  "fit_stock_cache_seconds", "resident_move_seconds", "uncached_fit_seconds",
                  "predict_and_sample_seconds")} for row in cm["stage_events"]]
    table(output / "fp16-stage-events.csv", event_rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    figures = {}
    def save(fig, name):
        fig.savefig(output / name, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        figures[name] = sha256_file(output / name)

    family = []
    moments = []
    witnesses = {}
    candidate = clean @ arrays["directions"].T
    controls_by_name = {"corrected_primary": primary, "strict_FP32_FIFO1024": strict}
    for label, values in controls_by_name.items():
        comparison = r["comparisons"][label]["relative_distribution_review"]
        projected = values @ arrays["directions"].T
        for axis in range(53):
            left = np.sort(candidate[:, axis]); right = np.sort(projected[:, axis])
            coordinate = np.unique(np.r_[left, right])
            delta = (np.searchsorted(left, coordinate, side="right") -
                     np.searchsorted(right, coordinate, side="right")) / 50000
            at = int(np.argmax(np.abs(delta))); distance = float(abs(delta[at]))
            require(abs(distance - comparison["cdf_difference"][axis]) < 1e-12,
                    "exact ECDF differs from authenticated review")
            family.append({"comparison": label, "axis": axis,
                           "name": pm["parameter_order"][axis] if axis < 21 else f"joint{axis-21}",
                           "ks": distance, "signed_maximizer": float(delta[at]),
                           "coordinate": float(coordinate[at]), "flag": distance > .01})
            key = f"{label}_axis{axis}"
            witnesses[key+"_coordinate"] = coordinate
            witnesses[key+"_delta"] = delta
        for axis, name in enumerate(pm["parameter_order"]):
            moments.append({"comparison": label, "parameter_index": axis, "parameter": name,
                            "mean_change_reference_sd": comparison["mean_difference_reference_sd"][axis],
                            "sd_fraction_change": comparison["sd_fraction_difference"][axis]})
    table(output / "fp16-all53-cdf.csv", family)
    table(output / "fp16-moment-comparisons.csv", moments)
    selected = [row["axis"] for row in family if row["comparison"] == "corrected_primary" and row["flag"]]
    strict_rank = sorted([row for row in family if row["comparison"] == "strict_FP32_FIFO1024"],
                         key=lambda row: -row["ks"])[:2]
    keep = {}
    for label, axes in [("corrected_primary", selected),
                        ("strict_FP32_FIFO1024", [row["axis"] for row in strict_rank])]:
        for axis in axes:
            for suffix in ("coordinate", "delta"):
                key = f"{label}_axis{axis}_{suffix}";keep[key] = witnesses[key]
    np.savez_compressed(output / "fp16-signed-cdf-witnesses.npz", **keep)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for ax, label, color in zip(axes, controls_by_name, ["#b24d70", "#238e8b"]):
        rows = [row for row in family if row["comparison"] == label]
        ax.bar(np.arange(53), [row["ks"] for row in rows], color=color)
        ax.axhline(.01, color="#c56a19", ls="--", label="Unchanged 0.01 investigation threshold")
        ax.axvline(20.5, color="#7d8791", lw=.8)
        ax.set(title=label.replace("_", " "), ylabel="Exact ECDF distance")
        ax.grid(axis="y", alpha=.15);ax.legend(frameon=False)
    axes[-1].set_xticks([0,7,8,19,25,40,52], ["M0","M7","M8","M19","J4","J19","J31"], rotation=90)
    axes[-1].set_xlabel("All 21 marginals and 32 fixed joint projections")
    fig.suptitle("Built with PriorLabs-TabPFN: clean FP16 comparison family", fontsize=14)
    fig.tight_layout();save(fig, "fp16-complete-cdf-family.png")

    fig, axes = plt.subplots(len(selected), 1, figsize=(13, 14))
    for axis, ax in zip(selected, axes):
        key = f"corrected_primary_axis{axis}"
        row = next(row for row in family if row["comparison"] == "corrected_primary" and row["axis"] == axis)
        ax.step(witnesses[key+"_coordinate"], witnesses[key+"_delta"], where="post", color="#b24d70", lw=1)
        for threshold in (-.01,.01):ax.axhline(threshold, color="#c56a19", ls="--", lw=.8)
        ax.axhline(0, color="#7d8791", lw=.6)
        ax.set(title=f"{row['name']}; max|difference|={row['ks']:.5f}",
               xlabel="Physical projection coordinate", ylabel="FP16 ECDF minus primary")
        ax.grid(alpha=.15)
    fig.suptitle("Built with PriorLabs-TabPFN: every retained primary-control flag\n"
                 "50,000 complete vectors per cohort; unchanged ±0.01 threshold", fontsize=14)
    fig.tight_layout(rect=(0,0,1,.96));save(fig, "fp16-primary-signed-flags.png")
    fig, axes = plt.subplots(1,2,figsize=(13,4.8))
    for row, ax in zip(strict_rank, axes):
        key = f"strict_FP32_FIFO1024_axis{row['axis']}"
        ax.step(witnesses[key+"_coordinate"], witnesses[key+"_delta"], where="post", color="#238e8b", lw=1.2)
        ax.axhline(0, color="#7d8791", lw=.6)
        ax.set(title=f"{row['name']}; max|difference|={row['ks']:.5f}",
               xlabel="Physical projection coordinate", ylabel="FP16 ECDF minus strict FIFO")
        ax.grid(alpha=.15)
    fig.suptitle("Two largest strict-FIFO comparisons; no empirical flags\n"
                 "Finite-cohort diagnostics; unchanged 0.01 criterion, not equivalence", fontsize=13)
    fig.tight_layout();save(fig,"fp16-strict-fifo-signed.png")

    events = cm["stage_events"]
    prep = sum(row["prepare_seconds"] for row in events)
    predict = sum(row["predict_and_sample_seconds"] for row in events)
    stock = sum(row["fit_stock_cache_seconds"] for row in events)
    move = sum(row["resident_move_seconds"] for row in events)
    fallback = sum(row["uncached_fit_seconds"] for row in events)
    request = cm["timings"]["posterior_request_seconds"]
    fig, axes = plt.subplots(2,1,figsize=(13,8),gridspec_kw={"height_ratios":[1.4,1]})
    x = np.arange(21);bottom=np.zeros(21)
    for key,label,color in [("fit_stock_cache_seconds","Stock cache fit","#315c99"),
                             ("resident_move_seconds","Resident move","#8473b0"),
                             ("uncached_fit_seconds","Original fallback fit","#c18a38"),
                             ("other_prepare","Other preparation","#9aa7b0"),
                             ("predict_and_sample_seconds","Prediction + sampling","#238e8b")]:
        values=np.array([row["prepare_seconds"]-row["fit_stock_cache_seconds"]-
                         row["resident_move_seconds"]-row["uncached_fit_seconds"]
                         if key=="other_prepare" else row[key] for row in events])
        require(values.min()>=0,"overlapping preparation components")
        axes[0].bar(x,values,bottom=bottom,label=label,color=color);bottom+=values
    axes[0].set(xticks=x,xlabel="Autoregressive stage",ylabel="Measured seconds",
                title="Clean FP16: one 51,200-vector proposal wave, 21 preparations")
    axes[0].legend(frameon=False,ncol=3,fontsize=9)
    axes[1].bar(["Posterior request","Wrapper wall","Allocated job"],
                 [request,cm["wrapper_total_seconds"],216],color=["#238e8b","#315c99","#8473b0"])
    axes[1].set(ylabel="Seconds; distinct timing scopes")
    for ax in axes:ax.grid(axis="y",alpha=.15)
    fig.suptitle("Built with PriorLabs-TabPFN: clean FP16 cost and scope",fontsize=14)
    fig.tight_layout();save(fig,"fp16-stage-and-cost-scopes.png")

    times=[row["request_seconds"] for row in host_csv]
    setup=host["session_setup_seconds"];session=host["total_seconds"]
    other=session-setup-sum(times)
    require(other>=0,"host session timing intervals do not reconcile")
    fig,axes=plt.subplots(2,1,figsize=(13,8),gridspec_kw={"height_ratios":[1.2,1]})
    labels=["A cold (case 0)","B retained (case 1)","A retained (case 0)"]
    axes[0].bar(labels,times,color=["#b24d70","#315c99","#238e8b"])
    for i,row in enumerate(host_csv):
        axes[0].text(i,times[i]+5,f"{times[i]:.3f} s\nfits {row['actual_stage_fits']}, hits {row['retained_stage_hits']}",
                     ha="center",va="bottom",fontsize=9)
    axes[0].set(ylabel="Complete 50k request seconds",ylim=(0,max(times)*1.2),
                title="Actual A-B-A requests; cold delta is not a causal retention estimate")
    left=0
    for value,label,color in [(setup,"Setup","#9aa7b0"),(times[0],"A cold","#b24d70"),
                               (times[1],"B retained","#315c99"),(times[2],"A retained","#238e8b"),
                               (other,"Other session intervals","#c18a38")]:
        axes[1].barh(["Session measured"],[value],left=[left],label=label,color=color);left+=value
    axes[1].barh(["Allocated job"],[1274],color="#8473b0")
    axes[1].set(xlabel="Seconds; stacked totals are not a chronology reconstruction")
    axes[1].legend(frameon=False,ncol=3,fontsize=9,loc="upper center",bbox_to_anchor=(.5,-.23))
    for ax in axes:ax.grid(alpha=.15)
    fig.suptitle("Host 5406016 and all 21 FP16 screen 5406017 were colocated on acn53\n"
                 "150,000 valid returned vectors; session/setup/allocated costs stay separate",fontsize=13)
    fig.tight_layout();save(fig,"host-aba-timing-scopes.png")

    b=visits[1][0];absolute=hr["requests"][1]["absolute_physical_assessment"]
    ranked=np.argsort(-np.array(absolute["marginal_cdf"]["upper"]))[:4]
    fig,axes=plt.subplots(2,2,figsize=(13,8))
    b_density=[]
    for axis,ax in zip(ranked,axes.flat):
        coordinate=b_arrays["queries"]/b_arrays["directions"][axis,axis]
        cdf=b_arrays["cdfs"][axis]
        first=max(0,np.searchsorted(cdf,.0005)-1);last=min(len(cdf)-1,np.searchsorted(cdf,.9995)+1)
        select=np.unique(np.linspace(first,last,min(75,last-first+1)).astype(int))
        edges=coordinate[select];width=np.diff(edges);counts,_=np.histogram(b[:,axis],bins=edges)
        ax.stairs(np.diff(cdf[select])/width,edges,color="#225ca7",label="Case 1 reference mass / width")
        ax.stairs(counts/(50000*width),edges,color="#b24d70",label="Host B, 50k complete vectors")
        ax.set(title=pm["parameter_order"][axis],xlabel="Physical value",ylabel="Probability / bin width")
        ax.grid(alpha=.15)
        b_density.append({"axis":int(axis),"displayed_count":int(counts.sum()),
                          "normalization_count":50000,"edges":[float(edges[0]),float(edges[-1])]})
    handles,names=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,names,frameon=False,ncol=2,loc="lower center")
    fig.suptitle("Host B (case 1): physical model_failure against its own qualified reference\n"
                 "No saved same-case unoptimized control; no cross-case relative comparison",fontsize=13)
    fig.tight_layout(rect=(0,.07,1,.91));save(fig,"host-B-own-reference-densities.png")

    median=statistics.median(m["timings"]["posterior_request_seconds"] for _,m,_ in controls)
    summary={
        "schema":"posteriorforge.fp16_pilot_ABA_report.v1","attribution":"Built with PriorLabs-TabPFN",
        "report_source_sha256":sha256_file(__file__),"generic_renderer_source_sha256":generic_sha,
        "clean":bind(cb,root),"clean_runner_sha256":cm["runner_source_sha256"],
        "review_v1_sha256":REVIEW_SHAS[0],"review_v2_sha256":REVIEW_SHAS[1],
        "v1_v2_metrics_identical":True,"all21_manifest_sha256":SCREEN_SHA,
        "request_seconds":request,"wrapper_seconds":cm["wrapper_total_seconds"],"allocated_seconds":216,
        "prepare_seconds":prep,"stock_cache_fit_seconds":stock,"resident_move_seconds":move,
        "original_fallback_fit_seconds":fallback,"predict_and_sample_seconds":predict,
        "other_request_seconds":request-prep-predict,"memory":cm["memory"],
        "control_request_median_seconds":median,"speedup_vs_control_median":median/request,
        "speedup_vs_strict_FIFO1024":sm["timings"]["posterior_request_seconds"]/request,
        "one_clean_optimized_repetition_only":True,"hostname":cm["hostname"],
        "timing_boundaries":cm["timing_boundaries"],"proposal_buffer":cm["proposal_buffer"],
        "RNG_evidence":rng_evidence,"all21_strict_failures_preserved":True,
        "qualified_FP32":False,"approximate_experiment":True,
        "host_manifest_sha256":HOST_SHA,"host_review_sha256":HOST_REVIEW_SHA,
        "host_session_seconds":session,"host_setup_seconds":setup,"host_other_intervals_seconds":other,
        "host_allocated_seconds":1274,"host_requests":host_csv,"host_ABA_predicates":host["isolation_aba"],
        "host_B_density":b_density,"figures":figures,
        "signed_curves_sha256":sha256_file(output/"fp16-signed-cdf-witnesses.npz"),
        "no_old_pvalues_or_equivalence_claim":True,"profiling_is_not_a_clean_repeat":True,
        "overall_runtime_phase_complete":False}
    write_json(output/"fp16-pilot-summary.json",summary)
    lines=["## Clean FP16 approximation pilot","",
           f"Built with PriorLabs-TabPFN. The single clean 50,000-vector FP16 request took **{request:.6f} s**, "
           f"wrapper wall **{cm['wrapper_total_seconds']:.6f} s**, and scheduler allocation **216 GPU-seconds** "
           f"on {cm['hostname']}. The measured request ratio is {median/request:.3f}x against the three-control "
           f"median and {sm['timings']['posterior_request_seconds']/request:.3f}x against the prior strict "
           "FP32 FIFO 1024 pilot. These are single completed-configuration measurements; no clean FP16 "
           "timing repetitions are implied. Request, wrapper and allocation are distinct scopes.","",
           f"Preparation consumed {prep:.6f} s, including stock-cache fitting {stock:.6f} s, resident movement "
           f"{move:.6f} s and original fallback fitting {fallback:.6f} s. Prediction plus sampling consumed "
           f"{predict:.6f} s. Peak allocated/reserved memory was {cm['memory']['peak_allocated_bytes']/2**30:.5f}/"
           f"{cm['memory']['peak_reserved_bytes']/2**30:.5f}GiB. Wrapper scope includes setup and NPZ/hash "
           "completion; its excess over request time is not assigned to one phase.","",
           "![Actual clean FP16 stage and cost scopes](fp16-stage-and-cost-scopes.png)","",
           "This is an **FP16 prefill/query approximation experiment**. The original all 21 hard-validity "
           "passes and all 21 strict-screen failures remain unchanged. No strict FP32/exact/KV-only/joint "
           "qualification is inferred from the new posterior review. The characterization generated zero posterior "
           "vectors; its G0-first/Cp-later measurements were not randomized or interleaved timing repeats. "
           "The original screen-source hash "
           "and every stage decision remain in fp16-all21-characterization-extract.json and its CSV.","",
           "## Complete finite-cohort comparisons","",
           "Review v1 and v2 have identical numerical comparison metrics. Against strict FP32 FIFO 1024, "
           "the maximum 53-axis exact two-sample ECDF distance is 0.00176, mean change at most "
           "0.002883531 reference SD, and SD fractional change 0.001405115 (0.1405115%), with **no empirical flags**. "
           "Against the corrected primary control, flags remain at M7/M8/M19 and J4/J19, with maximum "
           "distance 0.01362. The absolute physical result remains **model_failure**. All original 53 CDF "
           "and 21 mean/SD comparisons are retained in the CSVs and authenticated review extract.","",
           "![Complete FP16 comparison family](fp16-complete-cdf-family.png)","",
           "![Every retained primary-control signed ECDF flag](fp16-primary-signed-flags.png)","",
           "![Largest signed comparisons against strict FP32 FIFO](fp16-strict-fifo-signed.png)","",
           "Every plotted cohort has 50,000 complete finite supported vectors. Ordinary densities use "
           "whole-count normalization and retained reference-bin masses; display windows do not remove "
           "reported draws. The 0.01 CDF/0.05 reference-SD mean/5% SD investigation criteria are unchanged. "
           "No earlier p-values, population-equivalence or global-calibration claim is transferred.","",
           "The clean production manifest records named stream roles/seeds and source-bound isolation "
           "policy, **not before/after default CPU/CUDA RNG hashes**. Observed characterization RNG "
           "preservation belongs to the immutable f01 all 21 screen. It is not relabelled as observed "
           "clean-production state equality.","",
           "## Host A-B-A: a separate 150,000-vector session","",
           "| Visit | Case | Complete 50k request s | Actual fits | Retained hits | Physical result |",
           "|---|---:|---:|---:|---:|---|"]
    for row in host_csv:
        lines.append(f"| {row['visit']} | {row['case']} | {row['request_seconds']:.6f} | "
                     f"{row['actual_stage_fits']} | {row['retained_stage_hits']} | {row['physical_status']} |")
    lines += ["",f"Session total was {session:.6f} s, including setup {setup:.6f} s; allocated duration "
              "was 1274 GPU-seconds. Source/footer authentication, all 150,000 finite supported returned "
              "vectors, shutdown restoration and all 11 A-B-A equality predicates passed. Both A arrays "
              "and their NPZ payloads are exactly equal to the prior FIFO 1024 cohort. These are coupled "
              "repeated-seed checks, not independent posterior/timing repetitions.","",
              "![Actual host A-B-A request and session scopes](host-aba-timing-scopes.png)","",
              "The host job 5406016 and all 21 FP16 screen 5406017 were colocated on **acn53**. The entire "
              "cold difference from the prior FIFO timing is **not a causal retention-cost estimate**. "
              "Cold/warm, case differences, setup, session and allocated scopes remain distinct. No "
              "claim attributes their whole difference to retention.","",
              "B uses its own authenticated case 1 reference and also has physical model_failure. There "
              "is **no saved same-case unoptimized B control**; no cross-case relative comparison or "
              "missing-control zero is invented.","",
              "![Host B ordinary densities against its own reference](host-B-own-reference-densities.png)","",
              "The host requests recorded default Torch CPU/CUDA, NumPy and Python state hashes and "
              "preserved them within each request. Dedicated sampling streams were consumed normally; "
              "their repeated A before/after hashes satisfy the ABA test. Production migrations use "
              "identity/layout guards; full-storage byte-hash evidence belongs to the separate host "
              "qualification. Current FP16 profiling is a separate experiment and is not counted as a "
              "second clean repetition. Overall runtime-phase completion remains pending."]
    report=output/"report.md"
    report.write_text(report.read_text().rstrip()+"\n\n"+"\n".join(lines)+"\n")
    readme=f"""# Clean FP16 pilot and separate host ABA evidence

Built with PriorLabs-TabPFN. [NOTICE](NOTICE) and the copied [MIT license](LICENSE) retain attribution.

The clean FP16 approximation returned 50,000 complete supported vectors in 195.181788 s (wrapper 213.298992 s; allocated 216 GPU-seconds, acn65). It is one clean optimized timing repetition. All three unoptimized control timings and both FP32 FIFO pilots remain visible.

![Actual clean/control request timings](runtime-and-request-cost.png)

The all 21 hard screen passed while all 21 strict failures remain. Against strict FIFO 1024: max 53-axis ECDF 0.00176, mean 0.002883531 reference SD and SD fraction 0.001405115 (0.1405115%), no empirical flags. Against primary, M7/M8/M19 and J4/J19 remain flagged (max 0.01362); absolute physical status remains model_failure. These finite-cohort diagnostics do not establish equivalence or global calibration.

![Complete 53-axis comparison family](fp16-complete-cdf-family.png)

![Actual ordinary posterior densities](posterior-worst-four-densities.png)

The separate host ABA job returned 150,000 valid vectors over three complete requests 554.502019/275.740727/314.327154 s. Session total 1252.439019 s includes 77.859402 s setup; allocated duration 1274 s. All 11 equality predicates passed; A arrays and NPZ bytes equal prior FIFO 1024. B has its own reference/physical failure and no same-case unoptimized control. Host ABA and the all 21 FP16 screen were colocated on acn53; the cold timing difference cannot be assigned wholly to retention.

![Actual ABA/session/allocated scopes](host-aba-timing-scopes.png)

The host has observed default/global RNG hashes; clean FP16 has recorded stream/source policy but no saved production before/after global hashes. Observed f01 RNG preservation belongs to the characterization. Profiling is separate from this clean repetition. All old flags, strict failures and source artifacts are preserved; no old p-values are transferred.

[report.md](report.md) embeds all ten figures. Compact review, screen, allocation and ABA extracts retain original metrics, hashes and scopes. Raw samples, large cache JSON and the derived signed-curve NPZ stay outside Git. [checksums.json](checksums.json) hashes every other package file.

Source script SHA256: `{summary['report_source_sha256']}`. Independent FP16 reviews: `{REVIEW_SHAS[0]}` and `{REVIEW_SHAS[1]}`; their numerical comparisons are identical. All21 source: `{SCREEN_SHA}`. Host session/review: `{HOST_SHA}` / `{HOST_REVIEW_SHA}`. Producer/source hashes are also retained in the extracts. Public raw publication and overall runtime-phase completion are separate pending steps.

## Reproduce

Use the verified Python 3.11 reference environment with NumPy/SciPy/Matplotlib, preserving results/ gate, banks, authenticated case 0/case 1 references, primary and all three controls, both FP32 FIFO cohorts, clean FP16 cohort/reviews, all 21 screen and host ABA session/visits/review. Include budget/ledger.json with the completed 5406772/5406016/5406017 allocation entries. The report reads saved artifacts only and infers the checkout from __file__.

```bash
PF_ROOT=/path/to/unpacked-artifacts
PF_PYTHON=/path/to/verified-python3.11
PF_OUTPUT=/path/to/new-fp16-report
PF_PACKAGE=/path/to/new-portable-package
PYTHONPATH=src OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 "$PF_PYTHON" \\
  scripts/report_fp16_pilot_aba.py --root "$PF_ROOT" --output "$PF_OUTPUT" \\
  --package-output "$PF_PACKAGE"
```
"""
    (output/"README.md").write_text(readme)
    for name in ("LICENSE","NOTICE"):shutil.copy2(REPO/name,output/name)
    write_json(output/"render-verification.json",{
        "schema":"posteriorforge.fp16_ABA_report_checks.v1",
        "report_source_sha256":summary["report_source_sha256"],
        "all106_ECDF_metrics_match_review_tolerance":1e-12,
        "v1_v2_numeric_comparisons_identical":True,"same_actual_A_arrays_and_NPZ_as_prior":True,
        "all150000_host_integrity_passed":True,"all11_ABA_predicates":True,
        "clean_production_global_RNG_observation_claimed":False,
        "posterior_cohorts":4,"timing_measurements":7,"parent_visual_review":"pending",
        "no_new_model_or_GPU_calls":True})
    require(sha256_file(REPO/"scripts/report_execution_performance.py")==generic_sha,
            "generic renderer changed during report")
    checksums(output)
    if package_output:
        target=Path(package_output).expanduser().resolve();target.mkdir(parents=True,exist_ok=False)
        for path in output.iterdir():
            if path.is_file() and path.suffix!=".npz":shutil.copy2(path,target/path.name)
        checksums(target)
    return {"output":str(output),"package":str(package_output) if package_output else None,
            "source_sha256":summary["report_source_sha256"],
            "report_sha256":sha256_file(output/"report.md"),"figures":{**base["figures"],**figures}}


def checksums(folder):
    write_json(folder/"checksums.json",{"schema":"posteriorforge.evidence_checksums.v1",
               "algorithm":"sha256","files":{path.name:sha256_file(path) for path in sorted(folder.iterdir())
               if path.is_file() and path.name not in ("checksums.json","fp16-signed-cdf-witnesses.npz")}})


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",required=True);parser.add_argument("--output",required=True)
    parser.add_argument("--package-output")
    print(json.dumps(render(**vars(parser.parse_args())),indent=2))
