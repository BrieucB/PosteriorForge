#!/usr/bin/env python3
"""Render authenticated saved-run diagnostics without models or reference evaluation.

Built with PriorLabs-TabPFN. Install posteriorforge plus matplotlib, then pass
explicit gate, run, bank, parent-reference and refined-reference roots. All
scientific arrays remain unchanged. Only this plotting script writes outputs.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np

from posteriorforge.artifacts import sha256_file, write_json
from posteriorforge.banks import CONTEXT_SEEDS, load_context, load_heldout
from posteriorforge.gate import verify_refined_grid, verify_run_binding
from posteriorforge.inference import support_mask
from posteriorforge.reference import load_reference_distribution
from posteriorforge.science.analytical_benchmark_v2 import array_descriptor, build_problem
from posteriorforge.validation import _reference_arrays, make_reference_input

ATTRIBUTION = "Built with PriorLabs-TabPFN"
SCHEMA = "posteriorforge.cross_run_diagnostics.v1"
PARAMETERS = ("sigma_m3", "d0_m3", "b_m1", "a_m3")
SEEDS = tuple(CONTEXT_SEEDS)
LIMIT = 0.10


def require(condition, message):
    if not condition:
        raise ValueError(message)


def within(root, relative):
    relative = Path(relative)
    require(not relative.is_absolute() and ".." not in relative.parts,
            "artifact path must remain relative to its declared root")
    path = (root / relative).resolve()
    require(path.is_relative_to(root.resolve()), "artifact path escapes declared root")
    return path


def authenticate(args):
    gate_path = args.gate.resolve()
    gate_bytes = gate_path.read_bytes()
    gate = json.loads(gate_bytes)
    require(gate.get("schema") == "posteriorforge.scientific_gate.v1", "scientific gate required")
    require(gate.get("expected_runs") == 24 and gate.get("expected_count") == 50000
            and gate.get("context_size") == 4096, "complete 24-run scientific gate required")
    order = list(build_problem().parameter_order)
    references = {}
    for entry in gate["references"]:
        case = entry.get("case")
        require(type(case) is int and 0 <= case < 8 and case not in references,
                "duplicate or invalid reference case")
        require(entry.get("state") == "verified" and entry.get("grid_state") == "verified",
                "qualified parent and refined reference required")
        parent_root = within(args.references, f"case-{case}")
        parent_path = parent_root / "manifest.json"
        require(sha256_file(parent_path) == entry["manifest_sha256"],
                "gate-to-parent reference manifest SHA256 mismatch")
        parent_arrays, parent_manifest = load_reference_distribution(parent_root)
        grid_root = within(args.reference_grids, f"case-{case}")
        grid_path = grid_root / "manifest.json"
        require(sha256_file(grid_path) == entry["grid_manifest_sha256"],
                "gate-to-refined reference manifest SHA256 mismatch")
        arrays, manifest = load_reference_distribution(grid_root)
        require(parent_manifest.get("case_index") == manifest.get("case_index") == case,
                "reference case identity mismatch")
        verify_refined_grid(arrays, manifest, parent_arrays, parent_manifest,
                            entry["manifest_sha256"])
        *_, qualified = _reference_arrays(make_reference_input(arrays, manifest))
        require(qualified, "retained reference numerical and independent checks must qualify")
        query = load_heldout(args.banks, case)
        expected_binding = {"scientific_fingerprint": build_problem().scientific_fingerprint,
                            "parameter_order": order,
                            "observation_sha256": array_descriptor(query.obs)["sha256"]}
        require(manifest["binding"] == expected_binding, "reference/query/science binding mismatch")
        directions = arrays["directions"]
        require(directions.shape == (53, 21) and np.isfinite(directions).all()
                and np.array_equal(directions[:21], np.diag(np.diag(directions[:21])))
                and np.all(np.diag(directions[:21]) > 0),
                "saved marginal witnesses must be positive scaled physical axes")
        references[case] = (arrays, manifest, query, {
            "case": case, "parent_manifest_sha256": entry["manifest_sha256"],
            "parent_distribution_sha256": parent_manifest["distribution"]["sha256"],
            "grid_manifest_sha256": entry["grid_manifest_sha256"],
            "grid_distribution_sha256": manifest["distribution"]["sha256"],
            "cdf_precision": manifest["cdf_precision"],
            "maximum_cdf_step": manifest["grid_contract"]["maximum_cdf_step"],
            "binding": expected_binding,
        })
    require(set(references) == set(range(8)), "all eight authenticated references required")
    contexts = {seed: load_context(args.banks, seed, 4096) for seed in SEEDS}
    entries = {}
    samples_case0 = {}
    lower, upper, rows, records = [], [], [], []
    for entry in gate["runs"]:
        key = (entry.get("context_seed"), entry.get("case"))
        require(key not in entries, "duplicate seed/case run in gate")
        entries[key] = entry
    require(set(entries) == {(seed, case) for seed in SEEDS for case in range(8)},
            "gate must contain the exact declared seed/case matrix")
    for seed in SEEDS:
        for case in range(8):
            entry = entries[seed, case]
            path = within(args.runs, entry["manifest_file"])
            require(sha256_file(path) == entry["manifest_sha256"],
                    "gate-to-inference manifest SHA256 mismatch")
            manifest = json.loads(path.read_text())
            require(manifest.get("schema") == "posteriorforge.inference.v1"
                    and manifest.get("status") == "complete"
                    and manifest.get("context_seed") == seed and manifest.get("case") == case
                    and manifest.get("context_size") == 4096
                    and manifest.get("requested_count") == 50000
                    and manifest.get("variant") == "upstream_unmodified",
                    "complete unchanged 4096-context inference contract required")
            verify_run_binding(manifest, contexts[seed], references[case][2].obs, seed, case)
            require(manifest.get("samples_file") == "samples.npz", "unexpected samples filename")
            payload_path = path.parent / "samples.npz"
            require(sha256_file(payload_path) == manifest["samples_sha256"],
                    "inference samples payload SHA256 mismatch")
            with np.load(payload_path, allow_pickle=False) as payload:
                require(set(payload.files) == {"samples"}, "unexpected inference payload arrays")
                samples = payload["samples"].copy()
            require(samples.dtype == np.float64 and samples.shape == (50000, 21)
                    and np.isfinite(samples).all()
                    and array_descriptor(samples) == manifest["samples"]
                    and support_mask(samples).all(),
                    "complete finite prior-supported posterior rows required")
            assessment = entry["assessment"]
            require(assessment["integrity"].get("passed") is True
                    and assessment["integrity"].get("clipping_or_repair_performed") is False,
                    "retained gate integrity must pass without sample repair")
            require(assessment["limits"]["cdf"] == LIMIT,
                    "retained marginal threshold differs from declared 0.10")
            bounds = assessment["marginal_cdf"]
            lo, hi = (np.asarray(bounds[name], dtype=np.float64) for name in ("lower", "upper"))
            require(lo.shape == hi.shape == (21,) and np.isfinite(lo).all()
                    and np.isfinite(hi).all() and np.all((0 <= lo) & (lo <= hi) & (hi <= 1)),
                    "valid marginal CDF discrepancy bounds[21] required")
            require(float(lo.max()) == bounds["max_lower"]
                    and float(hi.max()) == bounds["max_upper"], "retained bound summary mismatch")
            lower.append(lo); upper.append(hi); rows.append({"context_seed": seed, "case": case})
            records.append({"context_seed": seed, "case": case, "gate_status": entry["status"],
                            "manifest_file": entry["manifest_file"],
                            "manifest_sha256": entry["manifest_sha256"],
                            "samples_sha256": manifest["samples_sha256"],
                            "samples_descriptor": manifest["samples"],
                            "context_theta": manifest["context_theta"],
                            "context_obs": manifest["context_obs"],
                            "observation_sha256": manifest["query"]["sha256"]})
            if case == 0:
                samples_case0[seed] = samples
    require(not {"torch", "tabpfn", "sbi", "jax", "jaxlib"}.intersection(sys.modules),
            "saved-data diagnostics must not import accelerator/model packages")
    bank_manifest = args.banks / "manifest.json"
    provenance = {
        "schema": SCHEMA, "attribution": ATTRIBUTION,
        "gate_sha256": sha256(gate_bytes).hexdigest(), "gate_status": gate["status"],
        "gate_verdict_unchanged": True,
        "scientific_scope": "eight_case_diagnostic_screen_not_SBC_or_global_calibration",
        "script_sha256": sha256_file(__file__),
        "bank_manifest_sha256": sha256_file(bank_manifest),
        "heldout_manifest_sha256": sha256_file(args.banks / "heldout.json"),
        "heldout_payload_sha256": sha256_file(args.banks / "heldout.npz"),
        "parameter_order": order, "row_order": rows, "runs": records,
        "references": [references[case][3] for case in range(8)],
        "sample_handling": {"rows_per_run": 50000, "all_complete_rows_used": True,
                            "clipping": False, "repair": False, "new_draws": False},
        "computation": {"retained_data_only": True, "new_reference_evaluations": False,
                        "new_gate_assessment": False, "model_loaded": False,
                        "accelerator_packages_imported": []},
        "regeneration": {"script": "scripts/plot_cross_run_diagnostics.py",
            "arguments": ["--gate gate-final.json", "--runs upstream-v1",
                          "--references references-v1", "--reference-grids reference-grids-v1",
                          "--banks banks-v1", "--output fresh-output-directory"]},
    }
    return (np.asarray(lower), np.asarray(upper), rows, order, samples_case0,
            references[0], provenance)


def heatmap(lower, upper, rows, order, output, provenance):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    from matplotlib.lines import Line2D
    import matplotlib.patheffects as effects

    failed = lower > LIMIT
    ambiguous = (lower <= LIMIT) & (upper > LIMIT)
    meets = upper <= LIMIT
    figure, axis = plt.subplots(figsize=(15.8, 12.2))
    figure.subplots_adjust(left=.165, right=.91, bottom=.28, top=.865)
    palette = LinearSegmentedColormap.from_list("discrepancy", ["#c4dfef", "#fffaf0", "#d35f4e"])
    vmax = max(.50, float(upper.max()))
    image = axis.imshow(upper, aspect="auto", cmap=palette,
                        norm=TwoSlopeNorm(vmin=0, vcenter=LIMIT, vmax=vmax))
    fy, fx = np.where(failed)
    marks = axis.scatter(fx, fy, marker="x", s=25, c="#172b43", linewidths=1.1)
    marks.set_path_effects([effects.Stroke(linewidth=2.8, foreground="white"), effects.Normal()])
    ay, ax = np.where(ambiguous)
    circles = axis.scatter(ax, ay, marker="o", s=33, facecolors="none",
                           edgecolors="#74299b", linewidths=1.4)
    circles.set_path_effects([effects.Stroke(linewidth=2.8, foreground="white"), effects.Normal()])
    axis.set_xticks(np.arange(21), order, rotation=48, ha="right", fontsize=10)
    axis.set_yticks(np.arange(24), [f"{r['context_seed']} · case {r['case']}" for r in rows], fontsize=9)
    axis.set_xlabel("Physical parameter (frozen order)", fontsize=11, labelpad=10)
    axis.set_ylabel("Context-bank seed · shared held-out case", fontsize=11)
    axis.set_xticks(np.arange(-.5, 21), minor=True)
    axis.set_yticks(np.arange(-.5, 24), minor=True)
    axis.grid(which="minor", color="white", linewidth=.65)
    axis.tick_params(which="minor", bottom=False, left=False)
    for boundary in (7.5, 15.5): axis.axhline(boundary, color="#172b43", linewidth=1.7)
    colorbar = figure.colorbar(image, ax=axis, fraction=.028, pad=.02)
    colorbar.set_label("Marginal CDF discrepancy upper bound", fontsize=11)
    colorbar.set_ticks([0, .10, .20, .30, .40, vmax])
    colorbar.ax.axhline(LIMIT, color="#172b43", linewidth=1.5)
    figure.suptitle(ATTRIBUTION + "\n24 complete runs × 21 parameters: marginal CDF discrepancies",
                   fontsize=17, fontweight="bold", color="#172b43", y=.96)
    handles = [
        Line2D([], [], marker="x", linestyle="none", color="#172b43", markersize=7,
               label=f"Definite marginal failure: lower > 0.10 ({failed.sum()} cells)"),
        Line2D([], [], marker="o", linestyle="none", markerfacecolor="none", color="#74299b", markersize=7,
               label=f"Inconclusive marginal criterion: lower ≤ 0.10 < upper ({ambiguous.sum()} cells)"),
        Line2D([], [], marker="s", linestyle="none", markerfacecolor="#c4dfef", color="#91b4c9", markersize=8,
               label=f"Unmarked: upper ≤ 0.10 ({meets.sum()} cells)"),
    ]
    figure.legend(handles=handles, loc="lower center", bbox_to_anchor=(.53, .074),
                  ncol=1, frameon=False, fontsize=9.5)
    figure.text(.53, .025, "Color is the retained discrepancy upper bound; markers use both retained bounds.\n"
                "Threshold applies only to marginal CDF discrepancy, not the full acceptance gate.\n"
                "Eight held-out cases are a diagnostic screen, not SBC or global calibration.",
                ha="center", va="center", fontsize=9, color="#3f526b")
    path = output / "marginal-cdf-discrepancy-heatmap.png"
    figure.savefig(path, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(figure)
    metadata = {**provenance, "figure_kind": "24_by_21_marginal_cdf_discrepancy_heatmap",
                "color_quantity": "retained marginal CDF discrepancy upper bound",
                "threshold": LIMIT, "threshold_scope": "marginal_CDF_only_not_whole_acceptance",
                "marker_contract": {"x": "lower > 0.10: definite failure of this marginal criterion",
                                    "open_circle": "lower <= 0.10 < upper: inconclusive marginal criterion",
                                    "unmarked": "upper <= 0.10: meets this marginal criterion only"},
                "classification_counts": {"definite_marginal_failure": int(failed.sum()),
                    "inconclusive_marginal_criterion": int(ambiguous.sum()),
                    "meets_marginal_criterion": int(meets.sum())},
                "marginal_cdf_discrepancy_lower": lower.tolist(),
                "marginal_cdf_discrepancy_upper": upper.tolist(),
                "figure": {"file": path.name, "sha256": sha256_file(path)}}
    sidecar = path.with_suffix(".json")
    write_json(sidecar, metadata)
    return path, sidecar


def overlays(lower, upper, rows, order, samples_by_seed, reference, output, provenance):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    arrays, manifest, heldout, reference_record = reference
    figure, axes = plt.subplots(2, 2, figsize=(14.6, 10.5), sharey=True)
    figure.subplots_adjust(left=.065, right=.985, bottom=.20, top=.85, hspace=.34, wspace=.20)
    figure.suptitle(ATTRIBUTION + "\nCase 0: the same observation with three independent context banks",
                   fontsize=17, fontweight="bold", color="#172b43", y=.965)
    colors = {SEEDS[0]: "#0072B2", SEEDS[1]: "#D55E00", SEEDS[2]: "#009E73"}
    records = []
    for name, axis in zip(PARAMETERS, axes.flat):
        coordinate = order.index(name)
        scale = float(arrays["directions"][coordinate, coordinate])
        x_ref = arrays["queries"] / scale
        cdf = arrays["cdfs"][coordinate]
        truth = float(heldout.theta[0, coordinate])
        lower_index = max(0, int(np.searchsorted(cdf, .0005)) - 1)
        upper_index = min(len(cdf)-1, int(np.searchsorted(cdf, .9995)))
        left = min(float(x_ref[lower_index]), truth,
                   *(float(samples_by_seed[seed][:, coordinate].min()) for seed in SEEDS))
        right = max(float(x_ref[upper_index]), truth,
                    *(float(samples_by_seed[seed][:, coordinate].max()) for seed in SEEDS))
        padding = .035*(right-left) if right > left else 1.
        axis.plot(x_ref, cdf, color="#172b43", linewidth=2.3,
                  label="Qualified float64 reference", zorder=2)
        details = []
        for seed in SEEDS:
            unique, counts = np.unique(samples_by_seed[seed][:, coordinate], return_counts=True)
            probabilities = np.cumsum(counts, dtype=np.float64)/50000
            x = np.r_[left-padding, unique, right+padding]
            y = np.r_[0., probabilities, 1.]
            axis.step(x, y, where="post", color=colors[seed], linewidth=1.45,
                      label=f"Context seed {seed}", alpha=.95, zorder=3)
            row = next(i for i, r in enumerate(rows) if r == {"context_seed": seed, "case": 0})
            details.append({"context_seed": seed,
                            "cdf_discrepancy_lower": float(lower[row, coordinate]),
                            "cdf_discrepancy_upper": float(upper[row, coordinate])})
        axis.axvline(truth, color="#6b7280", linestyle="--", linewidth=1.5,
                     label="Held-out truth (diagnostic only)", zorder=1)
        annotation = "Saved CDF discrepancy [lower, upper]\n" + "\n".join(
            f"{d['context_seed']}: [{d['cdf_discrepancy_lower']:.3f}, {d['cdf_discrepancy_upper']:.3f}]"
            for d in details)
        axis.text(.025, .965, annotation, transform=axis.transAxes, ha="left", va="top",
                  fontsize=8.4, color="#172b43",
                  bbox={"facecolor":"white", "edgecolor":"#dce3eb", "alpha":.90, "pad":4})
        axis.set(xlim=(left-padding, right+padding), ylim=(0, 1),
                 xlabel=name + " (physical units)", title=name)
        axis.set_yticks([0, .25, .5, .75, 1])
        axis.grid(alpha=.18)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=9)
        if axis in axes[:, 0]: axis.set_ylabel("Posterior cumulative probability", fontsize=10)
        records.append({"parameter": name, "parameter_index": coordinate,
                        "physical_truth": truth, "truth_role": "diagnostic_only_not_accuracy_criterion",
                        "reference_axis_scale": scale, "reference_cdf_precision": manifest["cdf_precision"],
                        "display_xlim": [left-padding, right+padding], "seed_discrepancy_bounds": details})
    handles = [Line2D([], [], color="#172b43", linewidth=2.3, label="Qualified float64 reference")]
    handles += [Line2D([], [], color=colors[s], linewidth=1.7, label=f"Context seed {s}") for s in SEEDS]
    handles += [Line2D([], [], color="#6b7280", linestyle="--", linewidth=1.5,
                       label="Held-out truth (diagnostic only)")]
    figure.legend(handles=handles, loc="lower center", bbox_to_anchor=(.52, .095),
                  ncol=3, frameon=False, fontsize=10)
    figure.text(.52, .035, "50,000 complete accepted rows per seed; exact empirical CDFs, without density smoothing.\n"
                "Truth proximity is diagnostic only and is never an accuracy criterion. Reference lines join saved grid values.\n"
                "Fixed four parameters from the existing case-0 diagnostic. Eight-case screen; not SBC or global calibration.",
                ha="center", va="center", fontsize=9, color="#3f526b")
    path = output / "case0-three-seed-cdf-overlays.png"
    figure.savefig(path, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(figure)
    metadata = {**provenance, "figure_kind": "case0_three_context_seed_exact_ECDF_overlays",
                "case": 0, "context_seeds": list(SEEDS), "selected_parameters": list(PARAMETERS),
                "selection": "same four named parameters as the existing case-0 four-marginal diagnostic; no reselection",
                "truth_role": "diagnostic_only_not_accuracy_criterion",
                "reference_display": "saved qualified CDF values joined for display only; no reference evaluated",
                "empirical_display": "exact ECDF, equal-value rows aggregated by counts; no smoothing",
                "parameters": records, "selected_reference": reference_record,
                "figure": {"file": path.name, "sha256": sha256_file(path)}}
    sidecar = path.with_suffix(".json")
    write_json(sidecar, metadata)
    return path, sidecar


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("gate", "runs", "references", "reference-grids", "banks", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "output must be a fresh directory")
    loaded = authenticate(args)
    import matplotlib
    matplotlib.use("Agg")
    args.output.mkdir(parents=True, exist_ok=False)
    lower, upper, rows, order, samples, reference, provenance = loaded
    first = heatmap(lower, upper, rows, order, args.output, provenance)
    second = overlays(lower, upper, rows, order, samples, reference, args.output, provenance)
    print(json.dumps({"figures": [str(first[0]), str(second[0])],
                      "sidecars": [str(first[1]), str(second[1])],
                      "gate_verdict_unchanged": True,
                      "new_reference_evaluations": False}, indent=2))


if __name__ == "__main__":
    main()
