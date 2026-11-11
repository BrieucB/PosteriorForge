"""Bounded artifact-fixture checks; no model or reference computation."""
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.special import ndtr

from posteriorforge.artifacts import sha256_file, write_json
from posteriorforge.figures import render_posterior_agreement
from posteriorforge.science.analytical_benchmark_v2 import array_descriptor, build_problem


def _retained_fixture(tmp_path):
    runs, references = tmp_path / "runs", tmp_path / "references"
    run = runs / "seed-20261111-case-0"
    reference = references / "case-0"
    run.mkdir(parents=True)
    reference.mkdir(parents=True)
    problem = build_problem()
    samples = problem.sample_prior(64, np.random.default_rng(42))
    query = np.zeros((1, 33), dtype=np.float64)
    with (run / "samples.npz").open("wb") as stream:
        np.savez(stream, samples=samples)
    manifest = {"schema": "posteriorforge.inference.v1", "status": "complete", "variant": "upstream_unmodified",
                "context_seed": 20261111, "case": 0, "context_size": 4096, "requested_count": len(samples),
                "scientific_fingerprint": problem.scientific_fingerprint,
                "parameter_order": list(problem.parameter_order), "query": array_descriptor(query),
                "samples_file": "samples.npz", "samples_sha256": sha256_file(run / "samples.npz"),
                "samples": array_descriptor(samples)}
    write_json(run / "manifest.json", manifest)
    queries = np.linspace(-8, 8, 129)
    # Retained fixture tables, explicitly inconclusive; no reference is computed.
    directions = np.concatenate((2 * np.eye(21), np.full((32, 21), 1 / np.sqrt(21))))
    arrays = {"directions": directions, "queries": queries,
              "cdfs": np.repeat(ndtr(queries)[None, :], 53, axis=0),
              "mean": np.zeros(21), "sd": np.ones(21)}
    with (reference / "distribution.npz").open("wb") as stream:
        np.savez(stream, **arrays)
    reference_manifest = {"schema": "posteriorforge.reference.v1", "case_index": 0, "state": "inconclusive",
                          "binding": {"scientific_fingerprint": problem.scientific_fingerprint,
                                      "parameter_order": list(problem.parameter_order),
                                      "observation_sha256": array_descriptor(query)["sha256"]},
                          "distribution": {"file": "distribution.npz", "sha256": sha256_file(reference / "distribution.npz"),
                                           "arrays": {name: array_descriptor(value) for name, value in arrays.items()}}}
    write_json(reference / "manifest.json", reference_manifest)
    upper = [.1] * 21
    upper[3] = .7
    gate = {"schema": "posteriorforge.scientific_gate.v1", "status": "incomplete", "passed": False,
            "expected_count": len(samples),
            "runs": [{"context_seed": 20261111, "case": 0, "status": "reference_inconclusive", "passed": False,
                      "manifest_file": str((run / "manifest.json").relative_to(runs)),
                      "manifest_sha256": sha256_file(run / "manifest.json"),
                      "assessment": {"marginal_cdf": {"upper": upper}}}],
            "references": [{"case": 0, "state": "inconclusive", "manifest_sha256": sha256_file(reference / "manifest.json")}]}
    gate_path = tmp_path / "gate.json"
    write_json(gate_path, gate)
    return gate_path, runs, references, run, reference, gate


def test_cdf_figure_selects_declared_worst_axis_preserves_rows_and_records_unavailable_cases(tmp_path):
    gate, runs, references, run, reference, value = _retained_fixture(tmp_path)
    original_samples_hash = sha256_file(run / "samples.npz")
    original_reference_hash = sha256_file(reference / "distribution.npz")
    result = render_posterior_agreement(gate=gate, runs=runs, references=references, output=tmp_path / "figure")
    metadata = json.loads(Path(result["sidecar"]).read_text())
    assert metadata["attribution"] == "Built with PriorLabs-TabPFN"
    assert metadata["selection"]["context_seed"] == 20261111
    assert metadata["plotted_case_count"] == 1
    selected = metadata["cases"][0]
    assert selected["selected_parameter_index"] == 3
    assert selected["selected_parameter"] == build_problem().parameter_order[3]
    assert selected["selected_marginal_cdf_upper_bound"] == .7
    assert selected["coordinate_transform"]["projected_axis_scale"] == 2.
    assert selected["sample_count"] == 64
    assert selected["reference_state"] == "inconclusive"
    assert all(not entry["plotted"] for entry in metadata["cases"][1:])
    assert metadata["gate_status"] == "incomplete"
    assert metadata["gate_verdict_unchanged"]
    assert metadata["sample_handling"]["all_complete_rows_used"]
    assert not metadata["sample_handling"]["clipping"]
    assert not metadata["sample_handling"]["gate_interpolation"]
    assert sha256_file(run / "samples.npz") == original_samples_hash
    assert sha256_file(reference / "distribution.npz") == original_reference_hash
    assert Path(result["figure"]).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert metadata["figure"]["sha256"] == sha256_file(result["figure"])
    assert result["sidecar_sha256"] == sha256_file(result["sidecar"])


@pytest.mark.parametrize("target", ["inference_manifest", "sample_payload", "reference_manifest", "reference_payload"])
def test_cdf_figure_rejects_broken_checksum_chains(tmp_path, target):
    gate, runs, references, run, reference, value = _retained_fixture(tmp_path)
    selected = {"inference_manifest": run / "manifest.json", "sample_payload": run / "samples.npz",
                "reference_manifest": reference / "manifest.json", "reference_payload": reference / "distribution.npz"}[target]
    with selected.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="SHA256|hash mismatch"):
        render_posterior_agreement(gate=gate, runs=runs, references=references, output=tmp_path / "figure")


def test_cdf_figure_rejects_observation_binding_even_after_manifests_are_rehashed(tmp_path):
    gate, runs, references, run, reference, value = _retained_fixture(tmp_path)
    path = reference / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["binding"]["observation_sha256"] = "0" * 64
    write_json(path, manifest)
    value["references"][0]["manifest_sha256"] = sha256_file(path)
    write_json(gate, value)
    with pytest.raises(ValueError, match="observation binding"):
        render_posterior_agreement(gate=gate, runs=runs, references=references, output=tmp_path / "figure")


def test_cdf_figure_rejects_path_escape_and_marks_absent_external_samples_unavailable(tmp_path):
    gate, runs, references, run, reference, value = _retained_fixture(tmp_path)
    original = value["runs"][0]["manifest_file"]
    value["runs"][0]["manifest_file"] = "../manifest.json"
    write_json(gate, value)
    with pytest.raises(ValueError, match="relative"):
        render_posterior_agreement(gate=gate, runs=runs, references=references, output=tmp_path / "bad")
    value["runs"][0]["manifest_file"] = original
    write_json(gate, value)
    (run / "samples.npz").unlink()
    result = render_posterior_agreement(gate=gate, runs=runs, references=references, output=tmp_path / "missing")
    metadata = json.loads(Path(result["sidecar"]).read_text())
    assert metadata["plotted_case_count"] == 0
    assert metadata["cases"][0]["reason"] == "declared external artifact unavailable"
