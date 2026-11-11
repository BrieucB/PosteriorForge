#!/usr/bin/env python3
"""Render retained case-0 posteriors; truth is used only for diagnostic markers."""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path
import numpy as np
from posteriorforge.artifacts import sha256_file, write_json
from posteriorforge.banks import load_heldout
from posteriorforge.reference import load_reference_distribution
from posteriorforge.science.analytical_benchmark_v2 import array_descriptor


def render(*, banks, run, reference, gate, output):
    banks, run, reference, gate, output = map(Path, (banks, run, reference, gate, output))
    gate_bytes = gate.read_bytes()
    gate_value = json.loads(gate_bytes)
    entry = next(item for item in gate_value["runs"] if item["context_seed"] == 20261111 and item["case"] == 0)
    ref_entry = next(item for item in gate_value["references"] if item["case"] == 0)
    manifest_path = run / "manifest.json"
    if sha256_file(manifest_path) != entry["manifest_sha256"]:
        raise ValueError("gate/run manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text())
    if manifest["status"] != "complete" or manifest["requested_count"] != 50000 or manifest["context_seed"] != 20261111 or manifest["case"] != 0:
        raise ValueError("requires completed 50000-row seed20261111 case0")
    if sha256_file(run / "samples.npz") != manifest["samples_sha256"]:
        raise ValueError("sample payload hash mismatch")
    with np.load(run / "samples.npz", allow_pickle=False) as payload:
        samples = payload["samples"].copy()
    if samples.shape != (50000, 21) or not np.isfinite(samples).all() or array_descriptor(samples) != manifest["samples"]:
        raise ValueError("complete sample array mismatch")
    expected_ref_hash = ref_entry.get("grid_manifest_sha256", ref_entry["manifest_sha256"])
    if sha256_file(reference / "manifest.json") != expected_ref_hash:
        raise ValueError("gate/reference manifest hash mismatch")
    arrays, ref = load_reference_distribution(reference)
    if ref["state"] != "verified" or ref["case_index"] != 0:
        raise ValueError("qualified retained case0 reference required")
    heldout = load_heldout(banks, 0)
    if manifest["query"] != array_descriptor(heldout.obs) or ref["binding"]["observation_sha256"] != manifest["query"]["sha256"]:
        raise ValueError("heldout observation mismatch")
    if ref["binding"]["parameter_order"] != manifest["parameter_order"] or ref["binding"]["scientific_fingerprint"] != manifest["scientific_fingerprint"]:
        raise ValueError("reference science/order mismatch")
    truth = heldout.theta[0]
    names = manifest["parameter_order"]
    directions, queries, cdfs = arrays["directions"], arrays["queries"], arrays["cdfs"]
    if not np.array_equal(directions[:21], np.diag(np.diag(directions[:21]))) or np.any(np.diag(directions[:21]) <= 0):
        raise ValueError("positive physical axes required")
    upper = np.asarray(entry["assessment"]["marginal_cdf"]["upper"])
    ranked = np.argsort(-upper, kind="stable")[:4]
    output.mkdir(parents=True, exist_ok=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    colors = {"pfn": "#c56a19", "reference": "#2162b8", "truth": "#263748"}
    windows, diagnostics = [], []
    for index in range(21):
        physical = queries / directions[index, index]
        lo_ref = physical[min(int(np.searchsorted(cdfs[index], .0005)), len(physical) - 1)]
        hi_ref = physical[min(int(np.searchsorted(cdfs[index], .9995)), len(physical) - 1)]
        low = min(float(samples[:, index].min()), float(lo_ref), float(truth[index]))
        high = max(float(samples[:, index].max()), float(hi_ref), float(truth[index]))
        padding = .025 * (high-low) if high > low else 1.
        windows.append((physical, low-padding, high+padding))
        diagnostics.append({"parameter": names[index], "marginal_cdf_upper": float(upper[index]),
                            "true_theta": float(truth[index]), "pfn_min": float(samples[:, index].min()),
                            "pfn_max": float(samples[:, index].max()), "reference_mean": float(arrays["mean"][index]),
                            "reference_sd": float(arrays["sd"][index])})
    figure, axes = plt.subplots(7, 3, figsize=(15, 23.5))
    figure.subplots_adjust(top=.94, bottom=.055, left=.065, right=.985, hspace=.51, wspace=.25)
    figure.suptitle("Built with PriorLabs-TabPFN\nCase 0: all 21 posterior marginals · 50,000 complete PFN draws", fontsize=20, fontweight="bold", color="#172b43", y=.987)
    for index, axis in enumerate(axes.flat):
        physical, low, high = windows[index]
        ordered = np.sort(samples[:, index])
        axis.step(np.r_[low, ordered, high], np.r_[0., np.arange(1, len(ordered)+1)/len(ordered), 1.],
                  where="post", color=colors["pfn"], linewidth=1.35, label="Frozen PFN empirical CDF", zorder=3)
        axis.plot(physical, cdfs[index], color=colors["reference"], linewidth=1.6, label="Qualified reference CDF")
        axis.axvline(truth[index], color=colors["truth"], linestyle="--", linewidth=1.2, label="True held-out parameter")
        axis.set(xlim=(low, high), ylim=(0, 1), title=f"{names[index]}   |   CDF discrepancy upper bound {upper[index]:.3f}")
        axis.set_xlabel("Physical parameter value", fontsize=9)
        if index % 3 == 0:
            axis.set_ylabel("Posterior CDF")
        axis.tick_params(labelsize=8)
        axis.grid(alpha=.17)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, ncol=3, loc="lower center", bbox_to_anchor=(.5, .023), fontsize=11, frameon=False)
    figure.text(.5, .009, "Diagnostic screen, not SBC. Seed 20261111, context 4096. Truth is shown only after inference.\nAll 50,000 PFN values retained; views include all PFN draws, truth, and reference 0.05%–99.95% range. Reference lines join saved CDF values only.", ha="center", fontsize=10, color="#425775")
    cdf_path = output / "case0-all-21-marginal-cdfs.png"
    figure.savefig(cdf_path, dpi=145, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(json.dumps({"cdf_png": str(cdf_path)}), flush=True)

    figure, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))
    figure.subplots_adjust(top=.84, bottom=.15, left=.075, right=.985, hspace=.38, wspace=.25)
    figure.suptitle("Built with PriorLabs-TabPFN\nCase 0: four largest retained marginal CDF errors", fontsize=19, fontweight="bold", color="#172b43", y=.98)
    hist_records = []
    for index, axis in zip(ranked, axes.flat):
        physical, low, high = windows[index]
        first = max(0, int(np.searchsorted(physical, low, side="right")) - 1)
        last = min(len(physical)-1, int(np.searchsorted(physical, high, side="left")) + 1)
        chosen = np.unique(np.linspace(first, last, min(71, last-first+1)).astype(int))
        edges = physical[chosen]
        if len(edges) < 3:
            raise ValueError("insufficient retained reference edges for histogram")
        widths = np.diff(edges)
        counts, _ = np.histogram(samples[:, index], bins=edges)
        pfn_density = counts / (len(samples) * widths)
        ref_mass = np.diff(cdfs[index, chosen])
        if np.any(ref_mass < -1e-12):
            raise ValueError("retained reference CDF is nonmonotone")
        ref_density = ref_mass / widths
        axis.stairs(pfn_density, edges, color=colors["pfn"], linewidth=1.8, label="PFN probability / bin width")
        axis.stairs(ref_density, edges, color=colors["reference"], linewidth=1.8, label="Reference CDF mass / bin width")
        axis.axvline(truth[index], color=colors["truth"], linestyle="--", linewidth=1.3, label="True held-out parameter")
        axis.set(title=f"{names[index]}   |   CDF discrepancy upper bound {upper[index]:.3f}",
                 xlabel="Physical parameter value", ylabel="Bin-averaged posterior density",
                 xlim=(float(edges[0]), float(edges[-1])))
        axis.grid(alpha=.17)
        outside = 1 - counts.sum()/len(samples)
        if outside > 0:
            axis.text(.02, .96, f"PFN mass outside retained grid: {outside:.3%}", transform=axis.transAxes, va="top", fontsize=8)
        hist_records.append({"parameter": names[index], "parameter_index": int(index), "edge_source": "subset of retained physical CDF query edges",
                             "edges": edges.tolist(), "pfn_bin_count": counts.tolist(), "reference_bin_mass": ref_mass.tolist(),
                             "pfn_mass_outside_bins": float(outside), "normalization": "full 50000 PFN draws; no in-window renormalization"})
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, ncol=3, loc="lower center", bbox_to_anchor=(.5, .075), fontsize=10, frameon=False)
    figure.text(.5, .025, "Visualization only: both curves use the same retained query-edge bins. Reference mass is weighted-mixture CDF differences.\nNo unweighted node histogram, KDE, clipping, or new reference evaluation. The scientific gate remains CDF-based; truth is diagnostic only.", ha="center", fontsize=9.5, color="#425775")
    density_path = output / "case0-worst4-marginal-densities.png"
    figure.savefig(density_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(json.dumps({"density_png": str(density_path)}), flush=True)
    metadata = {"schema": "posteriorforge.case_diagnostics.v1", "attribution": "Built with PriorLabs-TabPFN",
                "case": 0, "context_seed": 20261111, "context_size": 4096, "count": 50000,
                "purpose": "retained posterior distribution visualization, diagnostic screen not SBC",
                "truth_usage": "post-inference diagnostic markers only; never an inference input",
                "sample_clipping_or_repair": False, "reference_recomputed": False, "gate_changed": False,
                "gate_status_at_plot": gate_value["status"], "case_status": entry["status"],
                "selection": "four largest marginal CDF discrepancy upper bounds in retained assessment",
                "inputs": {"gate_sha256": sha256(gate_bytes).hexdigest(), "inference_manifest_sha256": entry["manifest_sha256"],
                           "samples_sha256": manifest["samples_sha256"], "reference_manifest_sha256": expected_ref_hash,
                           "reference_distribution_sha256": ref["distribution"]["sha256"],
                           "heldout_npz_sha256": sha256_file(banks/"heldout.npz"),
                           "heldout_manifest_sha256": sha256_file(banks/"heldout.json")},
                "reference_cdf_precision": ref["cdf_precision"], "marginals": diagnostics, "density_bins": hist_records,
                "figures": {cdf_path.name: sha256_file(cdf_path), density_path.name: sha256_file(density_path)},
                "regeneration": {"script": "scripts/plot_case_diagnostics.py",
                                 "arguments": {"banks": str(banks), "run": str(run), "reference": str(reference),
                                               "gate": str(gate), "output": "fresh output directory"}}}
    write_json(output/"case0-diagnostics.json", metadata)
    return {"cdf": str(cdf_path), "density": str(density_path), "sidecar": str(output/"case0-diagnostics.json")}


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("banks", "run", "reference", "gate", "output"):
        parser.add_argument("--"+name, required=True)
    print(json.dumps(render(**vars(parser.parse_args())), indent=2))

