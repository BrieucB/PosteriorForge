"""Portable reports derived solely from retained gate and budget evidence."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from hashlib import sha256
import math
from pathlib import Path

from .artifacts import sha256_file, write_json
from .budget import BudgetLedger, HELD_STATES

SCHEMA = "posteriorforge.evidence_report.v1"
COLORS = {"blue": "#2563b8", "teal": "#087f8c", "green": "#24704a", "orange": "#b56a12",
          "red": "#b83d49", "grey": "#8793a4", "light": "#e6ebf1", "text": "#172b43"}
BUDGET_FIELDS = ("reservation_id", "purpose", "gpus", "walltime_seconds", "reserved_gpu_seconds",
                 "actual_gpu_seconds", "accounted_gpus", "accounted_elapsed_seconds", "allocated_tres",
                 "state", "job_id", "slurm_state", "created_at", "submitted_at",
                 "reconciled_at", "accounting_exceeded_reservation")
RUN_FIELDS = ("context_seed", "case", "status", "passed", "manifest_file", "manifest_sha256",
              "max_marginal_cdf_lower", "max_marginal_cdf_upper", "max_joint_projection_cdf_lower",
              "max_joint_projection_cdf_upper", "max_absolute_bias_reference_sd", "min_sd_ratio", "max_sd_ratio",
              "max_absolute_bias_reference_sd_lower", "max_absolute_bias_reference_sd_upper",
              "min_sd_ratio_lower", "max_sd_ratio_upper", "constructor_and_import_seconds",
              "cold_checkpoint_load_included_in_request", "append_context_seconds", "posterior_request_seconds", "total_seconds",
              "accepted_samples_per_second", "peak_allocated_bytes", "peak_reserved_bytes")


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _csv(path, fields, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _run_rows(gate):
    result = []
    for entry in gate.get("runs", []):
        row = {key: entry.get(key) for key in RUN_FIELDS if key in entry}
        assessment = entry.get("assessment", {})
        for label, target in (("marginal_cdf", "marginal"), ("joint_projection_cdf", "joint_projection")):
            metrics = assessment.get(label, {})
            for bound in ("lower", "upper"):
                value = metrics.get("max_" + bound)
                if _number(value):
                    row[f"max_{target}_cdf_{bound}"] = value
        moments = assessment.get("moments", {})
        if _number(moments.get("max_absolute_bias_reference_sd")):
            row["max_absolute_bias_reference_sd"] = moments["max_absolute_bias_reference_sd"]
        ratios = moments.get("sd_ratio", [])
        if ratios and all(_number(value) for value in ratios):
            row.update(min_sd_ratio=min(ratios), max_sd_ratio=max(ratios))
        intervals = moments.get("uncertainty_intervals")
        if isinstance(intervals, dict):
            for label, bound, target, reduction in (
                    ("absolute_bias_reference_sd", "lower", "max_absolute_bias_reference_sd_lower", max),
                    ("absolute_bias_reference_sd", "upper", "max_absolute_bias_reference_sd_upper", max),
                    ("sd_ratio", "lower", "min_sd_ratio_lower", min),
                    ("sd_ratio", "upper", "max_sd_ratio_upper", max)):
                values = intervals.get(label, {}).get(bound, [])
                if values and all(_number(value) for value in values):
                    row[target] = reduction(values)
        row.update({key: value for key, value in entry.get("timings", {}).items()
                    if key in RUN_FIELDS and (_number(value) or
                    (key == "cold_checkpoint_load_included_in_request" and type(value) is bool))})
        row.update({key: value for key, value in entry.get("memory", {}).items() if key in RUN_FIELDS and _number(value)})
        result.append(row)
    return result


def _figure_tools():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 13, "axes.titleweight": "bold",
                         "axes.spines.top": False, "axes.spines.right": False, "savefig.facecolor": "white",
                         "axes.labelcolor": COLORS["text"], "text.color": COLORS["text"],
                         "figure.dpi": 130})
    return plt


def _save(plt, figure, output, name):
    figure.savefig(output / name, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return name


def _reference_figure(plt, references, output):
    from matplotlib.patches import Patch
    rows = {entry["case"]: entry for entry in references}
    cases = sorted(set(range(8)) | set(rows))
    figure, (axes, states) = plt.subplots(2, 1, figsize=(10, 4.2), gridspec_kw={"height_ratios": [4, 1]}, layout="constrained")
    finite = [(case, rows[case]["cdf_precision"]) for case in cases
              if case in rows and _number(rows[case].get("cdf_precision"))]
    maximum = max([.01] + [value * 1.25 for _, value in finite])
    axes.axhline(.005, color=COLORS["red"], linestyle="--", linewidth=1.3, label="Required precision ≤ 0.005")
    for case, value in finite:
        axes.scatter(case, value, s=55, color=COLORS["teal"], zorder=3)
    for case in cases:
        if not (case in rows and _number(rows[case].get("cdf_precision"))):
            axes.text(case, maximum * .32, "unavailable", ha="center", va="center", rotation=90,
                      color=COLORS["grey"], fontsize=9)
    axes.set(xlim=(min(cases) - .6, max(cases) + .6), ylim=(0, maximum),
             ylabel="Retained CDF numerical precision", title="Reference qualification for the shared held-out cases")
    axes.set_xticks(cases, [str(case) for case in cases])
    axes.grid(axis="y", alpha=.18)
    axes.legend(loc="upper right", fontsize=9)
    state_colors = {"verified": COLORS["green"], "qualified": COLORS["green"], "failed": COLORS["red"],
                    "missing": COLORS["grey"], "inconclusive": COLORS["orange"]}
    observed = []
    for case in cases:
        state = rows.get(case, {}).get("state", "missing")
        if state not in observed:
            observed.append(state)
        states.scatter(case, 0, s=180, marker="s", color=state_colors.get(state, COLORS["orange"]))
    states.set(xlim=axes.get_xlim(), ylim=(-.6, .6))
    states.set_xticks(cases, [f"Case {case}" for case in cases])
    states.set_yticks([])
    states.spines[["left", "bottom"]].set_visible(False)
    states.legend(handles=[Patch(color=state_colors.get(state, COLORS["orange"]), label=state) for state in observed],
                  loc="lower center", bbox_to_anchor=(.5, -.95), ncol=min(5, len(observed)), frameon=False, fontsize=9)
    return _save(plt, figure, output, "reference-qualification.png")


def _budget_figure(plt, summary, output):
    figure, axes = plt.subplots(figsize=(10, 2.8), layout="constrained")
    actual, held = summary["actual_gpu_hours"], summary["reserved_gpu_hours"]
    remaining = max(0, summary["remaining_gpu_hours"])
    start = 0
    for amount, color, label in ((actual, COLORS["blue"], "Actual allocation"),
                                  (held, COLORS["orange"], "Reserved allocation"),
                                  (remaining, COLORS["light"], "Available within cap")):
        axes.barh([0], [amount], left=start, height=.4, color=color, label=f"{label}: {amount:.4f} GPU-h")
        start += amount
    axes.axvline(200, color=COLORS["red"], linestyle="--", label="Hard cap: 200 GPU-h")
    axes.set(xlim=(0, max(205, start * 1.025)), ylim=(-.6, .6), xlabel="Allocated or reserved GPU-hours",
             title="Cumulative compute cost, including failures and reference jobs")
    axes.set_yticks([])
    axes.spines[["left", "bottom"]].set_visible(False)
    axes.legend(loc="lower center", bbox_to_anchor=(.5, -.9), ncol=2, frameon=False, fontsize=9)
    return _save(plt, figure, output, "compute-budget.png")


def _agreement_figure(plt, rows, output):
    selected = [row for row in rows if any(_number(row.get(key)) for key in
                ("max_marginal_cdf_upper", "max_joint_projection_cdf_upper", "max_absolute_bias_reference_sd", "min_sd_ratio"))]
    if not selected:
        return None
    figure, axes = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True, layout="constrained")
    labels = [f"{row.get('context_seed')} / {row.get('case')}" for row in selected]
    for index, row in enumerate(selected):
        for shift, label, color in ((-.12, "marginal", COLORS["blue"]), (.12, "joint_projection", COLORS["teal"])):
            lower, upper = row.get(f"max_{label}_cdf_lower"), row.get(f"max_{label}_cdf_upper")
            if _number(lower) and _number(upper):
                axes[0].plot([index + shift] * 2, [lower, upper], color=color, linewidth=2.2)
                axes[0].scatter(index + shift, upper, color=color, s=22, marker="o")
        bias = row.get("max_absolute_bias_reference_sd")
        if _number(bias):
            axes[1].scatter(index, bias, color=COLORS["grey"], marker="x", s=28, zorder=4)
        bias_low, bias_high = row.get("max_absolute_bias_reference_sd_lower"), row.get("max_absolute_bias_reference_sd_upper")
        if _number(bias_low) and _number(bias_high):
            axes[1].plot([index] * 2, [bias_low, bias_high], color=COLORS["blue"], linewidth=2)
            axes[1].scatter(index, bias_high, color=COLORS["blue"], s=20)
        low, high = row.get("min_sd_ratio_lower"), row.get("max_sd_ratio_upper")
        if _number(low) and _number(high):
            axes[2].plot([index] * 2, [low, high], color=COLORS["teal"], linewidth=2)
            axes[2].scatter([index, index], [low, high], color=COLORS["teal"], s=23)
        raw_low, raw_high = row.get("min_sd_ratio"), row.get("max_sd_ratio")
        if _number(raw_low) and _number(raw_high):
            axes[2].scatter([index, index], [raw_low, raw_high], color=COLORS["grey"], marker="x", s=26, zorder=4)
    axes[0].axhline(.10, color=COLORS["red"], linestyle="--", label="Gate: ≤ 0.10")
    axes[0].plot([], [], color=COLORS["blue"], marker="o", label="Marginal CDF bound")
    axes[0].plot([], [], color=COLORS["teal"], marker="o", label="32 projection CDF bound")
    axes[0].set(ylabel="Max CDF distance", title="Retained posterior agreement and absolute thresholds")
    axes[0].legend(fontsize=8, ncol=3)
    axes[1].axhline(.20, color=COLORS["red"], linestyle="--")
    axes[1].set(ylabel="Max |mean bias| / ref. SD")
    axes[1].plot([], [], color=COLORS["blue"], marker="o", label="Conservative numerical interval")
    axes[1].plot([], [], linestyle="none", color=COLORS["grey"], marker="x", label="Raw point estimate")
    axes[1].legend(fontsize=8, loc="best", ncol=2)
    axes[2].axhspan(.8, 1.2, color=COLORS["green"], alpha=.10)
    axes[2].axhline(.8, color=COLORS["green"], linestyle="--", linewidth=1)
    axes[2].axhline(1.2, color=COLORS["green"], linestyle="--", linewidth=1)
    axes[2].set(ylabel="Predicted / ref. SD\n(min–max across parameters)", xlabel="Context seed / held-out case", xlim=(-.6, len(selected) - .4))
    axes[2].plot([], [], color=COLORS["teal"], marker="o", label="Conservative min–max extent")
    axes[2].plot([], [], linestyle="none", color=COLORS["grey"], marker="x", label="Raw min–max point estimates")
    axes[2].legend(fontsize=8, loc="best", ncol=2)
    axes[2].set_xticks(range(len(selected)), labels, rotation=90, fontsize=8)
    for axis in axes:
        axis.grid(axis="y", alpha=.18)
    return _save(plt, figure, output, "posterior-agreement.png")


def _runtime_figures(plt, rows, output):
    result = {}
    timing = [row for row in rows if _number(row.get("posterior_request_seconds"))]
    if timing:
        figure, axes = plt.subplots(figsize=(11, 4), layout="constrained")
        axes.bar(range(len(timing)), [row["posterior_request_seconds"] for row in timing], color=COLORS["blue"])
        axes.set_xticks(range(len(timing)), [f"{row.get('context_seed')} / {row.get('case')}" for row in timing], rotation=90, fontsize=8)
        axes.set(ylabel="Complete posterior request latency (s)", xlabel="Context seed / held-out case",
                 title="Measured upstream requests; individual scientific runs")
        axes.grid(axis="y", alpha=.18)
        result["latency"] = _save(plt, figure, output, "upstream-latency.png")
    memory = [row for row in rows if _number(row.get("peak_allocated_bytes")) and _number(row.get("peak_reserved_bytes"))]
    if memory:
        figure, axes = plt.subplots(figsize=(11, 4), layout="constrained")
        positions = list(range(len(memory)))
        axes.bar([position - .17 for position in positions], [row["peak_allocated_bytes"] / 2**30 for row in memory],
                 width=.34, color=COLORS["blue"], label="Peak allocated")
        axes.bar([position + .17 for position in positions], [row["peak_reserved_bytes"] / 2**30 for row in memory],
                 width=.34, color=COLORS["teal"], label="Peak reserved")
        axes.set_xticks(positions, [f"{row.get('context_seed')} / {row.get('case')}" for row in memory], rotation=90, fontsize=8)
        axes.set(ylabel="PyTorch CUDA memory (GiB)", xlabel="Context seed / held-out case",
                 title="Measured upstream peak memory")
        axes.grid(axis="y", alpha=.18)
        axes.legend(fontsize=9)
        result["memory"] = _save(plt, figure, output, "upstream-memory.png")
    return result


def render_report(*, gate, ledger, output):
    """Render retained artifacts; no scientific inference or Slurm side effects.

    ``gate`` and ``ledger`` are JSON file paths; ``output`` is a fresh directory.
    Performance figures are conditional on real numeric measurements. Deferred
    profiling and scaling have no fabricated series or inferred speedups.
    """
    gate_path, ledger_path, output = Path(gate), Path(ledger), Path(output)
    gate_payload = gate_path.read_bytes()
    source_gate = json.loads(gate_payload)
    if source_gate.get("schema") != "posteriorforge.scientific_gate.v1":
        raise ValueError("scientific gate evidence schema required")
    if not ledger_path.is_file():
        raise FileNotFoundError("retained budget ledger is required")
    with BudgetLedger(ledger_path)._locked() as ledger_value:
        ledger_hash = sha256_file(ledger_path)
    summary = BudgetLedger._summary(ledger_value)
    if source_gate.get("passed") is True:
        entries = source_gate.get("runs", [])
        refs = source_gate.get("references", [])
        if (source_gate.get("status") != "pass"
                or source_gate.get("optimization_authorized_by_gate") is not True
                or source_gate.get("expected_runs") != 24 or len(entries) != 24
                or any(entry.get("passed") is not True or entry.get("status") != "pass" for entry in entries)
                or len(refs) != 8
                or any(entry.get("state") != "verified" or not _number(entry.get("cdf_precision"))
                       or not 0 <= entry["cdf_precision"] <= .005 for entry in refs)):
            raise ValueError("gate pass claim contradicts retained matrix or reference state")
    output.mkdir(parents=True, exist_ok=False)
    plt = _figure_tools()
    rows = _run_rows(source_gate)
    references = source_gate.get("references", [])
    budget_rows = []
    for entry in ledger_value["reservations"]:
        row = {key: entry[key] for key in BUDGET_FIELDS if key in entry}
        allocated_tres = entry.get("accounting_record", {}).get("AllocTRES")
        if isinstance(allocated_tres, str):
            row["allocated_tres"] = allocated_tres
        budget_rows.append(row)
    figures = {"reference": _reference_figure(plt, references, output),
               "budget": _budget_figure(plt, summary, output)}
    agreement = _agreement_figure(plt, rows, output)
    if agreement:
        figures["agreement"] = agreement
    figures.update(_runtime_figures(plt, rows, output))
    write_json(output / "gate.json", source_gate)
    write_json(output / "budget.json", {"schema": "posteriorforge.public_budget_snapshot.v1", "summary": summary,
                                       "reservations": budget_rows})
    _csv(output / "runs.csv", RUN_FIELDS, rows)
    _csv(output / "references.csv", ("case", "state", "cdf_precision", "manifest_sha256"), references)
    _csv(output / "budget.csv", BUDGET_FIELDS, budget_rows)
    passed_runs = sum(entry.get("passed") is True for entry in source_gate.get("runs", []))
    expected_runs = source_gate.get("expected_runs", 24)
    status = source_gate.get("status", "unknown")
    authorized = source_gate.get("passed") is True and source_gate.get("optimization_authorized_by_gate") is True
    lines = ["# PosteriorForge retained evidence", "", "Built with PriorLabs-TabPFN.", "", f"Scientific gate status: **{status}**. "
             f"{passed_runs}/{expected_runs} expected upstream runs passed the absolute gate.", "",
             "This is an eight-case diagnostic screen of the analytical 21-parameter, 33-observation "
             "inferred-noise hierarchy. It is not full SBC or a global calibration certificate.", "",
             "## Reference precision", "",
             "Each shared held-out case needs retained float64 refinement, independent numerical checks, "
             "and CDF precision at most 0.005. Unavailable precision is shown explicitly and cannot pass.", "",
             f"![Retained reference precision and qualification states]({figures['reference']})", "",
             "## Posterior agreement", ""]
    if agreement:
        lines += ["CDF segments show the witnessed lower and conservative whole-line upper bounds, including "
                  "tail mass and the source reference precision where available. Reference qualification "
                  "is assessed separately; an inconclusive reference cannot pass. A finite-grid lower bound cannot "
                  "establish a pass. Mean and SD panels overlay raw point estimates (crosses) on "
                  "conservative numerical intervals (circles and lines) from the same case reference. "
                  "Crosses without intervals are diagnostic and cannot establish a moment pass.", "",
                  f"![Posterior agreement and absolute acceptance thresholds]({agreement})", ""]
    else:
        lines += ["No retained numeric posterior assessments are available in this gate artifact. "
                  "Posterior agreement remains unmeasured.", ""]
    lines += ["## Measured runtime and memory", ""]
    if "latency" in figures:
        lines += ["Request latency includes upstream repeated stage/context work and support rejection for "
                  "the complete accepted vectors. These are individual scientific runs, not three-repeat "
                  "timing summaries; no p95 or scaling speedup is inferred. Constructor/import timing is "
                  "recorded separately in runs.csv; cold checkpoint loading remains included in the request "
                  "when its retained flag is true.", "",
                  f"![Measured complete upstream posterior request latency]({figures['latency']})", ""]
    else:
        lines += ["No complete numeric request timings are retained in this gate artifact.", ""]
    if "memory" in figures:
        lines += [f"![Measured upstream CUDA peak allocated and reserved memory]({figures['memory']})", ""]
    else:
        lines += ["No numeric CUDA peak memory measurements are retained in this gate artifact.", ""]
    lines += ["## Cumulative compute budget", "",
              f"Actual allocation: **{summary['actual_gpu_hours']:.4f} GPU-hours**. "
              f"Held reservations: **{summary['reserved_gpu_hours']:.4f} GPU-hours**. "
              f"Capacity after reservations: **{summary['remaining_gpu_hours']:.4f} GPU-hours**.", "",
              "The immutable cap is 200 allocated GPU-hours. Failed allocations and reference calculation "
              "are included; unknown submission outcomes retain their full reservation. Reconciled job tables "
              "retain the actual allocated GPU count, elapsed seconds and Slurm allocation TRES separately "
              "from the requested resources.", "",
              f"![Actual, reserved and available cumulative GPU-hours]({figures['budget']})", "",
              "## Engineering gate", "",
              ("The retained gate authorizes subsequent optimization qualification. This report contains only "
               "the measurements listed above; optimization profiles and multi-GPU scaling require their own "
               "retained evidence." if authorized else "Optimization and multi-GPU scaling remain deferred because "
               "the retained scientific gate has not passed. No engineering success is claimed."), "",
              "## Portable evidence", "",
              "`gate.json` retains per-run assessments. `runs.csv`, `references.csv` and `budget.csv` provide "
              "portable tables. `budget.json` omits submission commands and private absolute paths. "
              "`checksums.json` binds this report to its input snapshots and all output files. Raw arrays "
              "remain external; retrieval and verification commands are documented in the repository at "
              "`docs/artifacts.md`.", ""]
    (output / "report.md").write_text("\n".join(lines))
    evidence = {"schema": SCHEMA, "created_utc": datetime.now(timezone.utc).isoformat(),
                "scientific_status": status, "scientific_gate_passed": source_gate.get("passed") is True,
                "optimization_authorized_by_gate": authorized, "passed_runs": passed_runs,
                "expected_runs": expected_runs, "budget": summary, "figures": figures,
                "inputs": {"gate_sha256": sha256(gate_payload).hexdigest(), "ledger_sha256": ledger_hash},
                "unmeasured": [name for name in ("agreement", "latency", "memory") if name not in figures]
                              + ["optimization_profiles", "multi_gpu_scaling"]}
    write_json(output / "evidence.json", evidence)
    hashes = {path.name: sha256_file(path) for path in sorted(output.iterdir()) if path.is_file()}
    write_json(output / "checksums.json", {"schema": SCHEMA + ".checksums", "inputs": evidence["inputs"], "files": hashes})
    return {"schema": SCHEMA, "status": status, "report": str(output / "report.md"),
            "evidence": str(output / "evidence.json"), "budget": summary, "figures": figures}
