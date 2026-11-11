"""Authenticated empirical-versus-reference posterior CDF diagnostics.

This module reads retained artifacts only. It does not generate posterior draws,
load a transformer, recompute a reference, or alter a scientific gate verdict.
"""
from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np

from .artifacts import sha256_file, write_json
from .reference import load_reference_distribution
from .science.analytical_benchmark_v2 import array_descriptor

SELECTION_SEED = 20261111
SCHEMA = "posteriorforge.posterior_cdf_figure.v1"


def _within(root, relative):
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("artifact path must be relative and remain inside its declared root")
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("artifact path escapes its declared root") from error
    return resolved


def _selection_entries(gate):
    entries = {}
    for entry in gate.get("runs", []):
        if entry.get("context_seed") != SELECTION_SEED:
            continue
        case = entry.get("case")
        if type(case) is not int or not 0 <= case < 8:
            raise ValueError("selected run case must be an integer in[0,7]")
        if case in entries:
            raise ValueError("duplicate fixed-seed case in gate evidence")
        entries[case] = entry
    references = {}
    for entry in gate.get("references", []):
        case = entry.get("case")
        if type(case) is not int or not 0 <= case < 8 or case in references:
            raise ValueError("invalid or duplicate case in gate reference evidence")
        references[case] = entry
    return entries, references


def _authenticate_case(entry, reference_entry, *, runs, references, expected_count):
    """Load one gate-selected case with the complete payload checksum chain."""
    case = entry["case"]
    manifest_path = _within(runs, entry["manifest_file"])
    if sha256_file(manifest_path) != entry["manifest_sha256"]:
        raise ValueError("gate-selected inference manifest SHA256 mismatch")
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("schema") != "posteriorforge.inference.v1" or manifest.get("status") != "complete"
            or manifest.get("variant") != "upstream_unmodified" or manifest.get("context_seed") != SELECTION_SEED
            or manifest.get("case") != case or manifest.get("context_size") != 4096
            or manifest.get("requested_count") != expected_count):
        raise ValueError("gate-selected inference contract differs from the CDF figure contract")
    payload_path = manifest_path.parent / "samples.npz"
    if manifest.get("samples_file", "samples.npz") != "samples.npz":
        raise ValueError("unexpected posterior sample filename")
    if sha256_file(payload_path) != manifest["samples_sha256"]:
        raise ValueError("inference sample payload SHA256 mismatch")
    with np.load(payload_path, allow_pickle=False) as payload:
        if set(payload.files) != {"samples"}:
            raise ValueError("unexpected posterior sample arrays")
        samples = payload["samples"].copy()
    if (samples.shape != (expected_count, 21) or not np.isfinite(samples).all()
            or array_descriptor(samples) != manifest["samples"]):
        raise ValueError("posterior sample rows or array descriptor mismatch")
    ref_root = _within(references, f"case-{case}")
    reference_path = ref_root / "manifest.json"
    if sha256_file(reference_path) != reference_entry["manifest_sha256"]:
        raise ValueError("gate-selected reference manifest SHA256 mismatch")
    arrays, reference_manifest = load_reference_distribution(ref_root)
    if reference_manifest.get("case_index") != case:
        raise ValueError("reference case identity differs from the selected run")
    binding = reference_manifest["binding"]
    if (binding["scientific_fingerprint"] != manifest["scientific_fingerprint"]
            or binding["parameter_order"] != manifest["parameter_order"]
            or binding["observation_sha256"] != manifest["query"]["sha256"]):
        raise ValueError("reference science/order/observation binding differs from inference")
    marginal = entry["assessment"]["marginal_cdf"]
    errors = np.asarray(marginal["upper"], dtype=np.float64)
    if errors.shape != (21,) or not np.isfinite(errors).all() or np.any(errors < 0):
        raise ValueError("retained marginal CDF upper bounds[21] required for declared selection")
    axis = int(np.argmax(errors))  # First axis wins a tie, deterministically.
    directions = np.asarray(arrays["directions"], dtype=np.float64)
    queries = np.asarray(arrays["queries"], dtype=np.float64)
    cdfs = np.asarray(arrays["cdfs"], dtype=np.float64)
    if (directions.shape != (53, 21) or not np.isfinite(directions).all()
            or not np.array_equal(directions[:21], np.diag(np.diag(directions[:21])))
            or np.any(np.diag(directions[:21]) <= 0)):
        raise ValueError("reference marginal witnesses must be positive scaled physical axes")
    if (queries.ndim != 1 or len(queries) < 2 or not np.isfinite(queries).all()
            or np.any(np.diff(queries) <= 0) or cdfs.shape != (53, len(queries))
            or not np.isfinite(cdfs).all() or np.any(cdfs < -1e-12) or np.any(cdfs > 1 + 1e-12)
            or np.any(np.diff(cdfs, axis=1) < -1e-12)):
        raise ValueError("retained reference CDF grid is not a monotone probability table")
    order = manifest["parameter_order"]
    if len(order) != 21:
        raise ValueError("complete physical parameter order[21] required")
    detail = {"case": case, "context_seed": SELECTION_SEED, "status": entry["status"],
              "selected_parameter_index": axis, "selected_parameter": order[axis],
              "selected_marginal_cdf_upper_bound": float(errors[axis]),
              "sample_count": len(samples), "reference_grid_points": len(queries),
              "reference_state": reference_manifest["state"],
              "reference_cdf_precision": reference_manifest.get("cdf_precision"),
              "manifest_file": entry["manifest_file"], "manifest_sha256": entry["manifest_sha256"],
              "samples_sha256": manifest["samples_sha256"], "samples_descriptor": manifest["samples"],
              "reference_manifest_file": f"case-{case}/manifest.json",
              "reference_manifest_sha256": reference_entry["manifest_sha256"],
              "reference_distribution_sha256": reference_manifest["distribution"]["sha256"],
              "scientific_fingerprint": manifest["scientific_fingerprint"],
              "observation_sha256": manifest["query"]["sha256"],
              "coordinate_transform": {"display": "physical_parameter_units", "projected_axis_scale": float(directions[axis, axis]),
                                       "reference_x": "retained_query / projected_axis_scale"}}
    return samples[:, axis], queries / directions[axis, axis], cdfs[axis], detail


def render_posterior_agreement(*, gate, runs, references, output):
    """Render eight empirical/reference CDF panels from a fresh retained gate.

    Fixed seed20261111 is declared before inspecting posterior discrepancies.
    Within each case, the marginal axis with the largest retained upper CDF
    bound is shown; it is intentionally an adverse-case diagnostic selection.
    All posterior rows are used unchanged. Reference grid points are joined for
    display only; no interpolation is used to recompute or alter gate metrics.
    """
    gate_path, runs, references, output = Path(gate), Path(runs), Path(references), Path(output)
    gate_payload = gate_path.read_bytes()
    source = json.loads(gate_payload)
    if source.get("schema") != "posteriorforge.scientific_gate.v1":
        raise ValueError("retained scientific gate evidence required")
    expected_count = source.get("expected_count", 50000)
    if type(expected_count) is not int or expected_count < 2:
        raise ValueError("gate must declare at least two complete posterior rows")
    entries, reference_entries = _selection_entries(source)
    loaded, case_records = {}, []
    for case in range(8):
        entry, reference_entry = entries.get(case), reference_entries.get(case)
        reason = None
        if entry is None or entry.get("status") == "missing":
            reason = "selected upstream run missing"
        elif not entry.get("manifest_file") or not entry.get("manifest_sha256"):
            reason = "selected inference manifest unavailable"
        elif "marginal_cdf" not in entry.get("assessment", {}):
            reason = "retained marginal assessment unavailable"
        elif reference_entry is None or not reference_entry.get("manifest_sha256"):
            reason = "retained reference manifest unavailable"
        if reason is not None:
            case_records.append({"case": case, "plotted": False, "reason": reason,
                                 "run_status": entry.get("status", "missing") if entry else "missing",
                                 "reference_state": reference_entry.get("state", "missing") if reference_entry else "missing"})
            continue
        try:
            values, query_axis, cdf, detail = _authenticate_case(entry, reference_entry, runs=runs,
                                                               references=references, expected_count=expected_count)
        except FileNotFoundError:
            case_records.append({"case": case, "plotted": False, "reason": "declared external artifact unavailable",
                                 "run_status": entry.get("status", "missing"),
                                 "reference_state": reference_entry.get("state", "missing")})
            continue
        detail["plotted"] = True
        case_records.append(detail)
        loaded[case] = (values, query_axis, cdf, detail)
    output.mkdir(parents=True, exist_ok=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 4, figsize=(16, 8.8), sharey=True)
    figure.subplots_adjust(left=.06, right=.99, top=.83, bottom=.17, wspace=.23, hspace=.4)
    figure.suptitle("Built with PriorLabs-TabPFN\nPosterior CDF diagnostic screen — not SBC", fontsize=18, fontweight="bold", color="#172b43")
    for case, axis_plot in enumerate(axes.flat):
        if case not in loaded:
            record = case_records[case]
            axis_plot.text(.5, .5, record["reason"] + "\nRun: " + record["run_status"]
                           + "\nReference: " + record["reference_state"], transform=axis_plot.transAxes,
                           ha="center", va="center", fontsize=9, color="#758397", wrap=True)
            axis_plot.set_title(f"Case {case}: unavailable", fontsize=11)
            axis_plot.set_ylim(0, 1)
            axis_plot.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
            for spine in axis_plot.spines.values():
                spine.set_color("#dce3eb")
            continue
        values, query_axis, cdf, detail = loaded[case]
        ordered = np.sort(values)
        left = min(float(ordered[0]), float(query_axis[0]))
        right = max(float(ordered[-1]), float(query_axis[-1]))
        width = right - left
        padding = .015 * width if width > 0 else 1.
        # Endpoints extend the exact empirical CDF to the displayed tails.
        x = np.concatenate(([left - padding], ordered, [right + padding]))
        y = np.concatenate(([0.], np.arange(1, len(ordered) + 1) / len(ordered), [1.]))
        axis_plot.step(x, y, where="post", color="#c16a1d", linewidth=1.4, label="Upstream empirical CDF", zorder=3)
        axis_plot.plot(query_axis, cdf, color="#2563b8", linewidth=1.8, label="Retained reference grid")
        status_label = {"reference_inconclusive": "reference uncertain", "cdf_inconclusive": "CDF uncertain",
                        "moment_inconclusive": "moment uncertain", "model_failure": "model failure"}.get(detail["status"], detail["status"].replace("_", " "))
        axis_plot.set_title("Case {} · {}\nCDF upper {:.3f} · {}".format(case, detail["selected_parameter"], detail["selected_marginal_cdf_upper_bound"], status_label), fontsize=10.5)
        axis_plot.set(xlabel=detail["selected_parameter"] + " (physical units)", xlim=(left - padding, right + padding), ylim=(0, 1))
        axis_plot.set_yticks([0, .25, .5, .75, 1])
        axis_plot.grid(alpha=.18)
        axis_plot.tick_params(labelsize=8)
        if case % 4 == 0:
            axis_plot.set_ylabel("Posterior cumulative probability", fontsize=9)
        axis_plot.spines[["top", "right"]].set_visible(False)
    if loaded:
        first_axis = axes.flat[next(iter(loaded))]
        handles, labels = first_axis.get_legend_handles_labels()
        figure.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .075), ncol=2, frameon=False, fontsize=10)
    figure.text(.5, .038, "Fixed context seed 20261111. Per case, show the marginal with the largest retained CDF upper bound.\n"
                       "All complete rows are preserved; reference lines join saved grid values for display only. No gate metric is recomputed.",
                       fontsize=9, color="#3f526b", ha="center", va="center")
    figure_path = output / "posterior-cdf-agreement.png"
    figure.savefig(figure_path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    metadata = {"schema": SCHEMA, "attribution": "Built with PriorLabs-TabPFN",
                "scientific_scope": "eight_case_diagnostic_screen_not_SBC_or_global_calibration",
                "gate_status": source.get("status"), "gate_verdict_unchanged": True,
                "selection": {"context_seed": SELECTION_SEED,
                              "within_case_axis": "argmax retained marginal_cdf.upper; first parameter in a tie",
                              "selection_is_adverse_case_not_a_random_marginal": True},
                "sample_handling": {"all_complete_rows_used": True, "clipping": False, "repair": False,
                                    "truncation": False, "gate_interpolation": False},
                "inputs": {"gate_file": gate_path.name, "gate_sha256": sha256(gate_payload).hexdigest(),
                           "runs_root_hint": runs.name, "references_root_hint": references.name},
                "regeneration": {"python_api": "posteriorforge.figures.render_posterior_agreement",
                                 "arguments": ["gate=retained_gate_json", "runs=selected_run_root",
                                               "references=selected_reference_root", "output=fresh_output_directory"]},
                "plotted_case_count": len(loaded), "cases": case_records,
                "figure": {"file": figure_path.name, "sha256": sha256_file(figure_path)}}
    sidecar = output / "posterior-cdf-agreement.json"
    write_json(sidecar, metadata)
    return {"schema": SCHEMA, "figure": str(figure_path), "sidecar": str(sidecar),
            "figure_sha256": metadata["figure"]["sha256"], "sidecar_sha256": sha256_file(sidecar),
            "plotted_case_count": len(loaded), "gate_status": source.get("status")}
