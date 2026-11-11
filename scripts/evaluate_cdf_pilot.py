#!/usr/bin/env python3
"""Assess an explicitly labelled CDF-arithmetic variant with the original physical gate.

Built with PriorLabs-TabPFN. Reads authenticated saved arrays only. No inference,
reference integration, original-gate mutation, sample repair, or row selection.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import numpy as np

from posteriorforge.artifacts import sha256_file, write_json
from posteriorforge.banks import CONTEXT_SEEDS, load_context, load_heldout
from posteriorforge.gate import verify_refined_grid, verify_run_binding
from posteriorforge.inference import support_mask
from posteriorforge.reference import load_reference_distribution
from posteriorforge.science.analytical_benchmark_v2 import array_descriptor, build_problem, unflatten_observations
from posteriorforge.science.analytical_benchmark_reference_v2 import FullHierarchyReference, refinement_witnesses
from posteriorforge.validation import assess_samples, compare_to_upstream, make_reference_input

ATTRIBUTION = "Built with PriorLabs-TabPFN"
SCHEMA = "posteriorforge.cdf_arithmetic_assessment.v1"
RUN_SCHEMA = "posteriorforge.inference.cdf_arithmetic.v1"
BASELINE_GATE_SHA256 = "42163c5d77f0f1f8f7e1bb3a78c81fba43ca36a0682446a0dc533544995e490d"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def within(root, relative):
    root = Path(root).resolve()
    relative = Path(relative)
    require(not relative.is_absolute() and ".." not in relative.parts, "relative artifact path required")
    path = (root / relative).resolve()
    require(path.is_relative_to(root), "artifact path escapes declared root")
    return path


def validate_adapter_descriptor(descriptor):
    """Check the exact approved adapter identity without importing the SDK."""
    from posteriorforge import compat_cdf as adapter
    expected = {
        "schema": "posteriorforge.compat.finite_cdf.v1",
        "adapter_id": adapter.ADAPTER_ID,
        "adapter_version": adapter.ADAPTER_VERSION,
        "adapter_sha256": adapter.adapter_source_sha256(),
        "tabpfn_version": adapter.PINNED_TABPFN_VERSION,
        "sdk_source_sha256": dict(adapter.PINNED_SDK_SHA256),
        "sdk_source_sha256_after": dict(adapter.PINNED_SDK_SHA256),
        "sdk_sources_unchanged": True,
        "sdk_aliases_unchanged": True,
        "translator_code_unchanged": True,
        "correction": "finite_cdf_parentheses_only",
        "source_guard_passed": True,
        "sdk_files_modified": False,
        "original_alias_restored": True,
        "activation": "explicit",
        "scope": "finite_ensemble_cdf_parentheses_only",
        "tail_family_changed": False,
        "execution_optimized": False,
    }
    require(isinstance(descriptor, dict), "compatibility_adapter mapping required")
    for key, value in expected.items():
        actual = descriptor.get(key)
        equal = actual is value if isinstance(value, bool) else actual == value
        require(equal, "CDF adapter contract mismatch: " + key)
    count = descriptor.get("patched_call_count")
    require(type(count) is int and count > 0, "positive patched_call_count required")
    return dict(descriptor)


def validate_pilot_header(manifest, *, seed, case, context_size):
    expected = {
        "schema": RUN_SCHEMA, "variant": "cdf_arithmetic_v1", "status": "complete",
        "context_seed": seed, "case": case, "context_size": context_size,
        "requested_count": 50000, "scientific_run": True,
        "observation_preprocessing": "raw; TabPFN frozen preprocessing",
        "log_density": "not_used",
        "metadata_writer_restored": True, "explicit_variant_activation": True,
        "tail_family_changed": False, "execution_optimized": False,
    }
    for key, value in expected.items():
        actual = manifest.get(key)
        equal = actual is value if isinstance(value, bool) else actual == value
        require(equal, "CDF pilot manifest mismatch: " + key)
    from posteriorforge.inference_cdf import guard_baseline_source
    require(manifest.get("baseline_execution_source") == guard_baseline_source(),
            "pilot original execution-source binding mismatch")
    descriptor = validate_adapter_descriptor(manifest.get("compatibility_adapter"))
    support_calls = manifest.get("rejection", {}).get("support_calls")
    expected_calls = manifest.get("expected_patched_call_count")
    require(type(support_calls) is int and support_calls > 0, "positive rejection support_calls required")
    require(type(expected_calls) is int and expected_calls == descriptor["patched_call_count"]
            and expected_calls == 21 * 8 * support_calls,
            "CDF patch did not cover exactly every ensemble/stage/support proposal batch")
    return descriptor


def validate_run_reference_binding(manifest, custody):
    require(manifest.get("reference_binding") == custody["runner_reference_binding"],
            "pilot reference binding differs from authenticated original parent/refined grid")


def load_samples(manifest_path, manifest):
    require(manifest.get("samples_file", "samples.npz") == "samples.npz", "unexpected sample payload filename")
    path = manifest_path.parent / "samples.npz"
    require(sha256_file(path) == manifest["samples_sha256"], "posterior payload SHA256 mismatch")
    with np.load(path, allow_pickle=False) as payload:
        require(set(payload.files) == {"samples"}, "unexpected posterior payload arrays")
        values = payload["samples"].copy()
    require(array_descriptor(values) == manifest["samples"], "posterior array descriptor mismatch")
    return values


def reference_for_case(results, gate, query, case):
    selected = [entry for entry in gate["references"] if entry["case"] == case]
    require(len(selected) == 1, "unique original gate reference required")
    selected = selected[0]
    parent_root = results / "references-v1" / f"case-{case}"
    grid_root = results / "reference-grids-v1" / f"case-{case}"
    parent_path, grid_path = parent_root / "manifest.json", grid_root / "manifest.json"
    require(sha256_file(parent_path) == selected["manifest_sha256"], "original parent reference changed")
    require(sha256_file(grid_path) == selected["grid_manifest_sha256"], "original refined reference changed")
    parent_arrays, parent = load_reference_distribution(parent_root)
    arrays, manifest = load_reference_distribution(grid_root)
    problem = build_problem()
    expected_binding = {
        "scientific_fingerprint": problem.scientific_fingerprint,
        "parameter_order": list(problem.parameter_order),
        "observation_sha256": array_descriptor(query)["sha256"],
    }
    for key, expected in expected_binding.items():
        require(parent["binding"].get(key) == expected, "reference science/query/order mismatch: " + key)
    require(parent.get("case_index") == case and manifest.get("case_index") == case, "reference case mismatch")
    definition = FullHierarchyReference(problem, unflatten_observations(query))
    directions, _ = refinement_witnesses(definition)
    require(np.array_equal(parent_arrays["directions"], directions), "original 53 projection witnesses changed")
    require(np.array_equal(parent_arrays["queries"], np.linspace(-8., 8., 513)), "original parent query grid changed")
    verify_refined_grid(arrays, manifest, parent_arrays, parent, sha256_file(parent_path))
    require(sha256_file(parent_root / "combined" / "manifest.json") == parent["combined_node_manifest_sha256"],
            "retained combined-node manifest changed")
    for node in manifest["retained_node_manifests"]:
        require(sha256_file(within(parent_root, node["directory"]) / "manifest.json") == node["manifest_sha256"],
                "retained node manifest changed")
    reference = make_reference_input(arrays, manifest)
    custody = {
        "parent_manifest_sha256": sha256_file(parent_path), "grid_manifest_sha256": sha256_file(grid_path),
        "reference_distribution_sha256": manifest["distribution"]["sha256"],
        "directions": array_descriptor(arrays["directions"]), "queries": array_descriptor(arrays["queries"]),
        "cdf_precision": manifest["cdf_precision"], "reference_case": case,
        "binding": manifest["binding"], "grid_state": manifest["state"],
        "runner_reference_binding": {
            "parent": {"manifest_sha256": sha256_file(parent_path),
                       "distribution_sha256": parent["distribution"]["sha256"],
                       "cdf_precision": parent["cdf_precision"], "binding": parent["binding"]},
            "refined_grid": {"manifest_sha256": sha256_file(grid_path),
                             "distribution_sha256": manifest["distribution"]["sha256"],
                             "cdf_precision": manifest["cdf_precision"], "binding": manifest["binding"]},
        },
    }
    return arrays, manifest, reference, custody


def baseline_for_case(results, gate, *, seed, case, context, query, reference):
    entries = [entry for entry in gate["runs"] if entry["context_seed"] == seed and entry["case"] == case]
    require(len(entries) == 1, "unique original baseline run required")
    selected = entries[0]
    path = within(results / "upstream-v1", selected["manifest_file"])
    require(sha256_file(path) == selected["manifest_sha256"], "original baseline manifest changed")
    manifest = json.loads(path.read_text())
    require(manifest.get("schema") == "posteriorforge.inference.v1"
            and manifest.get("variant") == "upstream_unmodified"
            and manifest.get("status") == "complete" and manifest.get("context_size") == 4096
            and manifest.get("requested_count") == 50000
            and manifest.get("context_seed") == seed and manifest.get("case") == case,
            "original baseline identity mismatch")
    verify_run_binding(manifest, context, query, seed, case)
    values = load_samples(path, manifest)
    assessment = assess_samples(values, reference, support_mask=support_mask(values))
    old = selected["assessment"]
    require(assessment["status"] == old["status"] and assessment["passed"] == old["passed"]
            and assessment["limits"] == old["limits"], "baseline reassessment status/limits changed")
    for label in ("marginal_cdf", "joint_projection_cdf"):
        for key in ("lower", "upper"):
            require(np.allclose(assessment[label][key], old[label][key], rtol=0, atol=1e-12),
                    "baseline reassessment CDF differs from original gate")
    for key in ("sample_mean", "sample_sd"):
        require(np.allclose(assessment["moments"][key], old["moments"][key], rtol=0, atol=1e-12),
                "baseline reassessment moments differ from original gate")
    require(assessment["reference_qualified"], "original qualified reference required")
    return values, manifest, assessment, path


def write_csv(path, rows):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def reference_quantile(x, cdf, probability):
    unique, positions = np.unique(cdf, return_index=True)
    return float(np.interp(probability, unique, x[positions]))


def plot_comparisons(output, baseline, pilot, arrays, assessments, order, *, seed, case):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    colors = {"baseline": "#777777", "pilot": "#1479b8", "reference": "#111111"}
    labels = ["original upstream", "CDF arithmetic repair", "qualified reference"]
    legend = [Line2D([0], [0], color=colors[key], lw=1.8, label=label)
              for key, label in zip(("baseline", "pilot", "reference"), labels)]
    precision = assessments["pilot"]["reference_qualification"]["precision"]
    display_details = []
    common = f"seed {seed}, case {case}; 50,000 complete vectors per cohort"
    footer = ATTRIBUTION + " | display ranges selected from saved reference; acceptance uses all samples and whole-line bounds"
    figs = {}
    density, dax = plt.subplots(7, 3, figsize=(14, 19), layout="constrained")
    marginal, maxes = plt.subplots(7, 3, figsize=(14, 19), layout="constrained")
    for j, name in enumerate(order):
        x = arrays["queries"] / arrays["directions"][j, j]
        cdf = arrays["cdfs"][j]
        lo, hi = reference_quantile(x, cdf, .001), reference_quantile(x, cdf, .999)
        require(hi > lo, "nondegenerate reference display range required")
        pad = .03 * (hi - lo)
        lo, hi = lo - pad, hi + pad
        bins = np.linspace(lo, hi, 41)
        widths = np.diff(bins)
        reference_density = np.diff(np.interp(bins, x, cdf)) / widths
        d, m = dax.flat[j], maxes.flat[j]
        d.stairs(reference_density, bins, color=colors["reference"], lw=1.4)
        detail = {"parameter": name, "display_lower": lo, "display_upper": hi, "histogram_bins": 40}
        for key, values in (("baseline", baseline[:, j]), ("pilot", pilot[:, j])):
            counts, _ = np.histogram(values, bins)
            d.stairs(counts / (len(values) * widths), bins, color=colors[key], lw=1.1)
            detail[key + "_mass_below_display"] = float(np.mean(values < lo))
            detail[key + "_mass_above_display"] = float(np.mean(values > hi))
            ordered = np.sort(values)
            grid = np.linspace(lo, hi, 700)
            m.plot(grid, np.searchsorted(ordered, grid, side="right") / len(ordered), color=colors[key], lw=1.2)
        m.plot(x, cdf, color=colors["reference"], lw=1.4)
        m.fill_between(x, np.maximum(0, cdf-precision), np.minimum(1, cdf+precision), color=colors["reference"], alpha=.12)
        before = assessments["baseline"]["marginal_cdf"]["upper"][j]
        after = assessments["pilot"]["marginal_cdf"]["upper"][j]
        title = f"{name}: CDF upper {before:.3f} → {after:.3f}"
        d.set(title=name, xlim=(lo, hi), xlabel="physical coordinate", ylabel="binned density")
        m.set(title=title, xlim=(lo, hi), ylim=(-.02, 1.02), xlabel="physical coordinate", ylabel="CDF")
        display_details.append(detail)
    density.suptitle("Before/after physical marginal densities\n" + common, fontsize=13)
    density.supxlabel("Reference density is saved-CDF bin mass / bin width; illustrative only.\n" + footer, fontsize=9)
    marginal.suptitle("Before/after all 21 physical marginal CDFs\n" + common, fontsize=13)
    marginal.supxlabel(footer, fontsize=9)
    for fig in (density, marginal):
        fig.legend(handles=legend, loc="outside upper center", ncol=3, bbox_to_anchor=(.5, .97))
    figs["density"] = output / "cdf-pilot-density.png"
    figs["marginal_cdf"] = output / "cdf-pilot-marginal-cdf.png"
    density.savefig(figs["density"], dpi=145)
    marginal.savefig(figs["marginal_cdf"], dpi=145)
    plt.close(density)
    plt.close(marginal)
    joint, axes = plt.subplots(8, 4, figsize=(16, 22), layout="constrained")
    for index, ax in enumerate(axes.flat):
        j = index + 21
        cdf = arrays["cdfs"][j]
        x = arrays["queries"]
        lo, hi = reference_quantile(x, cdf, .001), reference_quantile(x, cdf, .999)
        pad = .03 * (hi-lo)
        lo, hi = lo-pad, hi+pad
        grid = np.linspace(lo, hi, 700)
        for key, values in (("baseline", baseline), ("pilot", pilot)):
            projected = np.sort(values @ arrays["directions"][j])
            ax.plot(grid, np.searchsorted(projected, grid, side="right") / len(projected), color=colors[key], lw=1.2)
        ax.plot(x, cdf, color=colors["reference"], lw=1.4)
        ax.fill_between(x, np.maximum(0, cdf-precision), np.minimum(1, cdf+precision), color=colors["reference"], alpha=.12)
        before = assessments["baseline"]["joint_projection_cdf"]["upper"][index]
        after = assessments["pilot"]["joint_projection_cdf"]["upper"][index]
        ax.set(title=f"fixed joint {index+1}: {before:.3f} → {after:.3f}", xlim=(lo, hi),
               ylim=(-.02, 1.02), xlabel="unchanged physical-vector projection", ylabel="CDF")
    joint.suptitle("Before/after all 32 original full-vector projection CDFs\n" + common, fontsize=13)
    joint.supxlabel(footer, fontsize=9)
    joint.legend(handles=legend, loc="outside upper center", ncol=3, bbox_to_anchor=(.5, .975))
    figs["joint_cdf"] = output / "cdf-pilot-joint-cdf.png"
    joint.savefig(figs["joint_cdf"], dpi=145)
    plt.close(joint)
    return {key: {"file": path.name, "sha256": sha256_file(path)} for key, path in figs.items()}, display_details


def run(args):
    root = Path(args.root).resolve()
    results = root / "results"
    gate_path = results / "gate-final.json"
    require(sha256_file(gate_path) == BASELINE_GATE_SHA256, "immutable original gate SHA256 mismatch")
    gate = json.loads(gate_path.read_text())
    require(args.seed in CONTEXT_SEEDS and 0 <= args.case < 8, "original context seed/case required")
    banks = results / "banks-v1"
    query = load_heldout(banks, args.case).obs
    baseline_context = load_context(banks, args.seed, 4096)
    arrays, reference_manifest, reference, reference_custody = reference_for_case(results, gate, query, args.case)
    baseline, original_manifest, original_assessment, original_path = baseline_for_case(
        results, gate, seed=args.seed, case=args.case, context=baseline_context, query=query, reference=reference)
    preflight = {
        "schema": SCHEMA + ".preflight", "attribution": ATTRIBUTION,
        "original_gate_sha256": sha256_file(gate_path), "baseline_reassessment_matches_original_gate": True,
        "baseline_status": original_assessment["status"], "reference": reference_custody,
        "physical_parameter_order": list(build_problem().parameter_order),
        "unchanged_limits": original_assessment["limits"], "no_new_reference_evaluations": True,
    }
    if args.preflight_only:
        print(json.dumps(preflight, indent=2))
        return preflight
    require(args.pilot_manifest is not None and args.output is not None, "--pilot-manifest and --output required")
    manifest_path = Path(args.pilot_manifest).resolve()
    manifest = json.loads(manifest_path.read_text())
    descriptor = validate_pilot_header(manifest, seed=args.seed, case=args.case, context_size=args.context_size)
    validate_run_reference_binding(manifest, reference_custody)
    context = load_context(banks, args.seed, args.context_size)
    verify_run_binding(manifest, context, query, args.seed, args.case)
    require(manifest.get("model", {}).get("dtype") == original_manifest["model"]["dtype"]
            and manifest.get("model", {}).get("parameter_count") == original_manifest["model"]["parameter_count"]
            and manifest.get("model", {}).get("autocast") is False
            and manifest.get("model", {}).get("forced_inference_dtype") == original_manifest["model"]["forced_inference_dtype"],
            "pilot model precision/architecture mismatch")
    pilot = load_samples(manifest_path, manifest)
    assessment = assess_samples(pilot, reference, support_mask=support_mask(pilot))
    require(assessment["reference_qualified"], "qualified original reference required")
    output = Path(args.output).resolve()
    for protected in ("banks-v1", "upstream-v1", "references-v1", "reference-grids-v1"):
        require(not output.is_relative_to(results / protected), "output may not alter original artifacts")
    output.mkdir(parents=True, exist_ok=False)
    assessments = {"baseline": original_assessment, "pilot": assessment}
    result = {
        "schema": SCHEMA, "attribution": ATTRIBUTION,
        "label": "New CDF-arithmetic variant evaluated with unchanged original physical criteria; original gate is immutable",
        "variant": manifest["variant"], "context_seed": args.seed, "case": args.case, "context_size": args.context_size,
        "expected_count": 50000, "parameter_order": list(build_problem().parameter_order),
        "status": assessment["status"], "passed": assessment["passed"],
        "scope": "single_run_diagnostic_not_the_complete_24_run_gate",
        "all_24_original_runs_passed": False, "optimization_authorized_by_this_result": False,
        "initial_pilot_expansion_condition_satisfied": bool(assessment["passed"] and args.context_size == 4096
                                                            and args.seed == 20261111 and args.case == 0),
        "baseline_reassessment_matches_original_gate": True,
        "original_gate_sha256": sha256_file(gate_path),
        "baseline": {"manifest_file": str(original_path), "manifest_sha256": sha256_file(original_path),
                     "samples_sha256": original_manifest["samples_sha256"], "assessment": original_assessment},
        "pilot": {"manifest_file": str(manifest_path), "manifest_sha256": sha256_file(manifest_path),
                  "samples_sha256": manifest["samples_sha256"], "samples_descriptor": manifest["samples"],
                  "assessment": assessment, "compatibility_adapter": descriptor},
        "reference": reference_custody,
        "upstream_comparison_diagnostic": compare_to_upstream(pilot, baseline, reference_sd=arrays["sd"],
                                                             directions=arrays["directions"]),
        "custody": {"evaluation_script_sha256": sha256_file(__file__),
                    "assessment_source_sha256": sha256_file(Path(__file__).resolve().parents[1] / "src/posteriorforge/validation.py"),
                    "binding_source_sha256": sha256_file(Path(__file__).resolve().parents[1] / "src/posteriorforge/gate.py")},
        "limitations": [
            "Reference qualification and its retained numerical-uncertainty contract are unchanged; no reference integration is performed.",
            "CDF bounds compare the finite returned empirical distribution with the reference; they are not population confidence intervals.",
            "Density curves are saved-CDF bin differences; display ranges and binning do not affect acceptance or renormalize sample tails.",
            "One passing run is only the predeclared expansion condition; scientific success requires the entire original 24-run matrix.",
        ],
    }
    rows = []
    for j, name in enumerate(result["parameter_order"]):
        row = {"parameter": name}
        for key in ("baseline", "pilot"):
            a = assessments[key]
            row[key + "_cdf_lower"] = a["marginal_cdf"]["lower"][j]
            row[key + "_cdf_upper"] = a["marginal_cdf"]["upper"][j]
            for metric in ("absolute_bias_reference_sd", "sd_ratio"):
                row[key + "_" + metric] = a["moments"][metric][j]
                for bound in ("lower", "upper"):
                    row[key + "_" + metric + "_" + bound] = a["moments"]["uncertainty_intervals"][metric][bound][j]
        rows.append(row)
    write_csv(output / "cdf-pilot-parameters.csv", rows)
    projections = []
    for j in range(32):
        row = {"joint_witness": j+1, "physical_direction": json.dumps(arrays["directions"][j+21].tolist())}
        for key in ("baseline", "pilot"):
            for bound in ("lower", "upper"):
                row[key + "_cdf_" + bound] = assessments[key]["joint_projection_cdf"][bound][j]
        projections.append(row)
    write_csv(output / "cdf-pilot-projections.csv", projections)
    result["figures"], result["display_details"] = plot_comparisons(
        output, baseline, pilot, arrays, assessments, result["parameter_order"], seed=args.seed, case=args.case)
    write_json(output / "cdf-pilot-assessment.json", result)
    write_json(output / "cdf-pilot-checksums.json", {path.name: sha256_file(path) for path in sorted(output.iterdir()) if path.is_file()})
    require(sha256_file(gate_path) == BASELINE_GATE_SHA256, "original gate changed during assessment")
    print(json.dumps({"output": str(output), "status": result["status"], "passed": result["passed"],
                      "expansion_condition_satisfied": result["initial_pilot_expansion_condition_satisfied"]}, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="scratch PosteriorForge root containing immutable results/")
    parser.add_argument("--pilot-manifest")
    parser.add_argument("--output")
    parser.add_argument("--seed", type=int, default=20261111)
    parser.add_argument("--case", type=int, default=0)
    parser.add_argument("--context-size", type=int, choices=(256, 1024, 4096), default=4096)
    parser.add_argument("--preflight-only", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
