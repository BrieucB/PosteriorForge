import numpy as np
import pytest

from posteriorforge.reference import prepare_reference, reference_moments


def test_real_reference_requires_accounted_slurm(monkeypatch, tmp_path):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(RuntimeError, match="Slurm"):
        prepare_reference(tmp_path / "banks", 0, tmp_path / "reference")


def test_mixture_moments_preserve_parameter_order():
    class Problem:
        parameter_order = tuple(range(21))
        class prior:
            noise_names = tuple(range(15, 21))
    class Reference:
        problem = Problem()
        names = tuple(range(15))
    class Nodes:
        metadata = {"physics_mean": list(range(15)),
                    "physics_covariance": np.eye(15).tolist(),
                    "noise_mean": [.2] * 6,
                    "noise_second_moment": [.05] * 6}
        def authenticate(self, reference):
            pass
    mean, sd = reference_moments(Reference(), Nodes())
    np.testing.assert_array_equal(mean[:15], np.arange(15))
    np.testing.assert_allclose(mean[15:], .2)
    np.testing.assert_allclose(sd[:15], 1)
    np.testing.assert_allclose(sd[15:], .1)


def test_unexpected_runtime_failure_leaves_failed_manifest(monkeypatch, tmp_path):
    import json
    import posteriorforge.reference as module
    monkeypatch.setenv("SLURM_JOB_ID", "unit-test-only-no-compute")
    def fail(*args, **kwargs):
        raise ValueError("synthetic initialization failure")
    monkeypatch.setattr(module, "_prepare_reference", fail)
    root = tmp_path / "failed"
    with pytest.raises(ValueError, match="initialization"):
        module.prepare_reference(tmp_path / "banks", 0, root)
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["state"] == "failed"
    assert manifest["qualification"]["state"] == "failed"
    assert not manifest["converged"]
    assert manifest["qualification"]["precision"] is None
