import json
import csv
from pathlib import Path

import pytest

from posteriorforge.artifacts import sha256_file, write_json
from posteriorforge.budget import BudgetLedger
from posteriorforge.report import render_report


def _inputs(tmp_path):
    gate = {"schema": "posteriorforge.scientific_gate.v1", "status": "incomplete", "passed": False,
            "expected_runs": 24, "context_size": 4096, "expected_count": 50000,
            "optimization_authorized_by_gate": False, "runs": [],
            "references": [{"case": case, "state": "missing"} for case in range(8)]}
    gate_path = tmp_path / "gate.json"
    write_json(gate_path, gate)
    ledger_path = tmp_path / "ledger.json"
    BudgetLedger(ledger_path).reserve(gpus=1, walltime_seconds=3600, purpose="fixture reservation",
                                      metadata={"private_path": "/private/token-fixture"})
    return gate, gate_path, ledger_path


def test_incomplete_report_has_only_retained_state_figures_and_preserves_budget(tmp_path):
    gate, gate_path, ledger = _inputs(tmp_path)
    output = tmp_path / "report"
    result = render_report(gate=gate_path, ledger=ledger, output=output)
    assert result["status"] == "incomplete"
    assert set(result["figures"]) == {"reference", "budget"}
    evidence = json.loads((output / "evidence.json").read_text())
    assert not evidence["scientific_gate_passed"]
    assert not evidence["optimization_authorized_by_gate"]
    assert evidence["budget"]["reserved_gpu_hours"] == 1
    assert evidence["budget"]["actual_gpu_hours"] == 0
    assert "multi_gpu_scaling" in evidence["unmeasured"]
    report = (output / "report.md").read_text()
    for figure in result["figures"].values():
        assert f"]({figure})" in report
        assert (output / figure).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert "No retained numeric posterior assessments" in report
    assert "remain deferred" in report
    assert "/private/token-fixture" not in (output / "budget.json").read_text()


def test_numeric_assessments_generate_measured_figures_and_portable_tables(tmp_path):
    gate, gate_path, ledger = _inputs(tmp_path)
    gate["runs"] = [{"context_seed": 20261111, "case": 0, "status": "pass", "passed": True,
                     "manifest_file": "seed20261111/case-0/manifest.json", "manifest_sha256": "fixturehash",
                     "assessment": {"marginal_cdf": {"max_lower": .04, "max_upper": .06},
                                    "joint_projection_cdf": {"max_lower": .05, "max_upper": .07},
                                    "moments": {"max_absolute_bias_reference_sd": .1, "sd_ratio": [.9, 1.1],
                                                "uncertainty_intervals": {
                                                    "absolute_bias_reference_sd": {"lower": [.08, .09], "upper": [.1, .105]},
                                                    "sd_ratio": {"lower": [.895, 1.095], "upper": [.905, 1.105]}}}},
                     "timings": {"constructor_and_import_seconds": 19., "cold_checkpoint_load_included_in_request": True,
                                 "posterior_request_seconds": 100., "total_seconds": 120.,
                                 "accepted_samples_per_second": 500.},
                     "memory": {"peak_allocated_bytes": 2**30, "peak_reserved_bytes": 2**31}}]
    gate["references"][0].update(state="verified", cdf_precision=.002, manifest_sha256="fixture-reference")
    write_json(gate_path, gate)
    output = tmp_path / "report"
    result = render_report(gate=gate_path, ledger=ledger, output=output)
    assert set(result["figures"]) == {"reference", "budget", "agreement", "latency", "memory"}
    csv_text = (output / "runs.csv").read_text()
    assert "max_marginal_cdf_upper" in csv_text
    assert "fixturehash" in csv_text
    assert "seed20261111/case-0/manifest.json" in csv_text
    with (output / "runs.csv").open() as stream:
        row = next(csv.DictReader(stream))
    assert float(row["constructor_and_import_seconds"]) == 19.
    assert row["cold_checkpoint_load_included_in_request"] == "True"
    assert "cold_start_seconds" not in row
    assert float(row["max_absolute_bias_reference_sd"]) == .1
    assert float(row["max_absolute_bias_reference_sd_upper"]) == .105
    assert float(row["min_sd_ratio_lower"]) == .895
    assert float(row["max_sd_ratio_upper"]) == 1.105
    report = (output / "report.md").read_text()
    for figure in result["figures"].values():
        assert f"]({figure})" in report
    assert "not three-repeat" in report
    assert "raw point estimates (crosses)" in report
    assert "cold checkpoint loading remains included" in report
    checksums = json.loads((output / "checksums.json").read_text())
    for file, digest in checksums["files"].items():
        assert sha256_file(output / file) == digest
    assert checksums["inputs"]["gate_sha256"] == sha256_file(gate_path)
    assert checksums["inputs"]["ledger_sha256"] == sha256_file(ledger)
    assert result["status"] == "incomplete"  # a single passing case cannot pass the matrix


def test_report_refuses_overwrite_unknown_schema_and_inconsistent_pass(tmp_path):
    gate, gate_path, ledger = _inputs(tmp_path)
    output = tmp_path / "report"
    render_report(gate=gate_path, ledger=ledger, output=output)
    with pytest.raises(FileExistsError):
        render_report(gate=gate_path, ledger=ledger, output=output)
    gate.update(status="pass", passed=True, optimization_authorized_by_gate=True)
    write_json(gate_path, gate)
    with pytest.raises(ValueError, match="contradicts"):
        render_report(gate=gate_path, ledger=ledger, output=tmp_path / "bad-pass")
    gate["schema"] = "other"
    write_json(gate_path, gate)
    with pytest.raises(ValueError, match="schema"):
        render_report(gate=gate_path, ledger=ledger, output=tmp_path / "bad-schema")
