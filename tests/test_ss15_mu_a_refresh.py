"""CPU engineering quotas and source/RNG/failure guards; no scientific cohort refresh."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import random

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/refresh_ss15_mu_a.py"
spec = importlib.util.spec_from_file_location("refresh_ss15_mu_a_test", PATH)
producer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(producer)


def toy_rows():
    values = np.zeros((50000, 21), dtype=np.float64)
    values[:, producer.kernel.metadata()["noise_indices"]] = .25
    values[:, 0] = 4.
    values[:, 3] = np.linspace(-1., 1., len(values))
    return values


def test_one_pass_full_quota_replays_noise_and_preserves_other20():
    values = toy_rows()
    original = values.copy()
    result, receipt = producer.refresh_once(values, 12345)
    assert result.shape == (50000, 21) and result.dtype == np.float64
    assert np.array_equal(values, original)
    assert result[:, 1:].tobytes() == original[:, 1:].tobytes()
    assert not np.shares_memory(values, result)
    mean, variance = producer.kernel.conditional_parameters(values)
    rng = np.random.Generator(np.random.PCG64(12345))
    z = rng.standard_normal(50000)
    assert np.array_equal(result[:, 0], mean + np.sqrt(variance) * z)
    assert receipt["standard_normal_draws"] == producer.array_descriptor(z)
    assert receipt["final_private_state"] == rng.bit_generator.state
    assert receipt["other20_columns_byte_identical"]
    assert receipt["conditional_residual_diagnostic"]["before"]["mean"] > 15
    assert abs(receipt["conditional_residual_diagnostic"]["after"]["mean"]) < .02


def test_private_rng_does_not_consume_default_states():
    numpy_before, python_before = copy.deepcopy(np.random.get_state()), random.getstate()
    producer.refresh_once(toy_rows(), 91)
    assert producer.state_sha((numpy_before, python_before)) == producer.state_sha(
        (np.random.get_state(), random.getstate()))


@pytest.mark.parametrize("kind", ["count", "dtype", "nonfinite", "support"])
def test_bad_input_fails_whole_without_filtering(kind):
    values = toy_rows()
    if kind == "count":
        values = values[:-1]
    elif kind == "dtype":
        values = values.astype(np.float32)
    elif kind == "nonfinite":
        values[300, 4] = np.nan
    else:
        values[400, 6] = .8
    with pytest.raises(ValueError):
        producer.refresh_once(values, 5)


def test_parent_binding_rejects_wrong_source_and_physical_order(monkeypatch):
    values = toy_rows()
    metadata = producer.kernel.metadata()
    descriptor = producer.array_descriptor(values)
    monkeypatch.setattr(producer, "PARENT_ARRAY_SHA", descriptor["sha256"])
    doc = {"schema": "posteriorforge.inference.ss15_fp16.v1",
           "variant": "accuracy_ss15_fp16_fifo_v1", "status": "complete",
           "requested_count": 50000, "context_seed": 20261111, "case": 0, "repetition": 0,
           "runner_source_sha256": producer.PARENT_PRODUCER_SHA,
           "scientific_fingerprint": metadata["scientific_fingerprint"],
           "parameter_order": metadata["physical_parameter_order"], "samples": descriptor}
    producer.validate_parent(doc, values, metadata)
    for key, bad in [("runner_source_sha256", "0" * 64),
                     ("parameter_order", list(reversed(doc["parameter_order"]))),
                     ("requested_count", True)]:
        corrupted = copy.deepcopy(doc)
        corrupted[key] = bad
        with pytest.raises(ValueError, match="parent binding"):
            producer.validate_parent(corrupted, values, metadata)


def test_payload_tampering_rejected_before_loading(tmp_path):
    metadata = producer.kernel.metadata()
    path = tmp_path / "manifest.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="parent manifest"):
        producer.load_parent(path, metadata)


def test_modified_kernel_rejected(monkeypatch):
    monkeypatch.setattr(producer.kernel, "source_sha256", lambda: "0" * 64)
    with pytest.raises(ValueError, match="kernel source"):
        producer.source_guard()


def test_no_reference_or_evaluation_dependency_and_only_cpu_imports():
    import ast
    tree = ast.parse(PATH.read_text())
    imports = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    assert not any("evaluat" in name or "validation" in name or "reference" in name
                   for name in imports if name)
    assert not any(name and ("torch" in name or "tabpfn" in name) for name in imports)
    assert producer.COUNT == 50000
