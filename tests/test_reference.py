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


def _moment_fixture():
    from types import SimpleNamespace
    reference = SimpleNamespace(names=tuple(range(15)), problem=SimpleNamespace(
        parameter_order=tuple(range(21)), prior=SimpleNamespace(noise_names=tuple(range(15, 21)))))
    nodes = SimpleNamespace(metadata={"physics_mean": [0.] * 15,
        "physics_covariance": np.eye(15).tolist(), "noise_mean": [.2] * 6,
        "noise_second_moment": [.05] * 6}, authenticate=lambda reference: None)
    mean, sd = reference_moments(reference, nodes)
    levels = [{"power": power, "scramble_means": np.tile(mean, (4, 1)),
        "scramble_sds": np.tile(sd, (4, 1)), "combined_mean": mean.copy(),
        "combined_sd": sd.copy()} for power in (16, 18)]
    checks = [{"global_mean": [0.] * 3, "global_covariance": np.eye(3).tolist(),
               "noise_mean": [.2] * 6} for _ in range(3)]
    return reference, nodes, levels, checks


def test_all_coordinate_moment_precision_checks_local_and_noise_sd():
    from posteriorforge.reference import _moment_precision
    reference, nodes, levels, checks = _moment_fixture()
    assert _moment_precision(reference, nodes, levels, checks)["passed"]
    # A local Gaussian mean is outside the independent three-global check.
    levels[-1]["scramble_means"][0, 9] += .006
    result = _moment_precision(reference, nodes, levels, checks)
    assert not result["passed"]
    assert result["mean_error_reference_sd"][9] == pytest.approx(.006)
    levels[-1]["scramble_means"][0, 9] -= .006
    # Noise SD precision must be retained, not inferred from its CDF alone.
    levels[0]["combined_sd"][17] += .0006
    result = _moment_precision(reference, nodes, levels, checks)
    assert not result["passed"]
    assert result["sd_error_reference_sd"][17] == pytest.approx(.006)


def test_independent_moment_corner_precision_is_normalized_and_required():
    from posteriorforge.reference import _moment_precision
    reference, nodes, levels, checks = _moment_fixture()
    checks[0]["global_mean"][1] = .006
    result = _moment_precision(reference, nodes, levels, checks)
    assert not result["passed"]
    assert result["mean_error_reference_sd"][1] == pytest.approx(.006)
    with pytest.raises(ValueError, match="two retained"):
        _moment_precision(reference, nodes, levels[-1:], checks)
