"""CPU contracts for the new opt-in API; no neural/model/GPU calls."""
import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sdk = pytest.importorskip("tabpfn.regressor")

from posteriorforge import execution_distribution_only as private
from posteriorforge.compat_cdf import finite_cdf_compatibility

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "distribution_cpu_proof_test", ROOT / "scripts/prove_distribution_only_cpu.py")
proof = importlib.util.module_from_spec(spec)
spec.loader.exec_module(proof)


def fixture(*, average=False):
    rows, bins = 8, 4
    borders = np.array([-2, -1, 0, 1, 2], np.float32)
    arrays = {"query": np.zeros((rows, 33), np.float32),
              "physical_borders": np.array([-4, -2, 0, 2, 4], np.float32)}
    arrays["query"][2, 0] = np.nan
    members = []
    for member in range(8):
        prefix = f"member_{member}_"
        arrays[prefix + "raw_model_logits"] = (
            np.arange(rows * bins, dtype=np.float32).reshape(rows, bins) * .03 + member * .1)
        arrays[prefix + "to_borders"] = borders.copy()
        arrays[prefix + "from_borders"] = borders.copy() + (member * .005 if member % 2 else 0)
        arrays[prefix + "logit_cancel_mask"] = np.zeros(bins, bool)
        members.append({"target_transform_observed": bool(member % 2),
                        "descending_borders": False})
    return proof.make_regressor(
        arrays, {"member_rows": rows, "members": members},
        average_before_softmax=average)


def bitwise(left, right):
    return (left.dtype == right.dtype and left.shape == right.shape and
            left.detach().numpy().tobytes() == right.detach().numpy().tobytes())


@pytest.mark.parametrize("average", [False, True])
def test_original_distribution_bits_actual_criterion_inputs_rng_and_summary_tripwire(average):
    model, query = fixture(average=average)
    generators = {"fixture": torch.Generator(device="cpu").manual_seed(32)}
    with finite_cdf_compatibility():
        result = proof.prove_case(model, query, stage=0, name="synthetic-unit-fixture",
                                  members=range(8), repeated=False, torch=torch,
                                  sdk=sdk, generators=generators)
    assert result["logits_bitwise_identical"]
    assert result["actual_criterion_same_object"]
    assert result["all_cpu_rng_states_unchanged"]
    assert result["input_bits_and_tensor_versions_unchanged"]


def test_real_summary_tripwire_rejects_original_and_private_has_no_fake_keys():
    model, query = fixture()
    with finite_cdf_compatibility():
        with pytest.raises(AssertionError, match="forbidden criterion summary"):
            with proof.forbid_summaries(model.renormalized_criterion_):
                model.predict(query, output_type="full", quantiles=[])
        with proof.forbid_summaries(model.renormalized_criterion_):
            actual = private.predict_distribution(model, query)
    assert set(actual) == {"criterion", "logits"}
    assert actual["criterion"] is model.renormalized_criterion_


def test_live_capture_alias_is_rebound_each_call_and_original_alias_restored():
    model, query = fixture()
    original_predict = sdk.TabPFNRegressor.predict
    with finite_cdf_compatibility():
        active = sdk.translate_probs_across_borders
        with proof.observe_translation(sdk) as first:
            a = private.predict_distribution(model, query)
        assert sdk.translate_probs_across_borders is active
        with proof.observe_translation(sdk) as second:
            b = private.predict_distribution(model, query)
        assert sdk.translate_probs_across_borders is active
    assert len(first) == len(second) == 8
    assert first == second and bitwise(a["logits"], b["logits"])
    assert sdk.TabPFNRegressor.predict is original_predict


def test_exception_preserves_sdk_aliases_and_input_versions():
    model, query = fixture()
    before = proof.input_identity(model, query)
    def fail(*args, **kwargs):
        raise RuntimeError("retained executor failure")
        yield
    model.executor_.iter_outputs = fail
    original = sdk.TabPFNRegressor.predict
    with finite_cdf_compatibility():
        active = sdk.translate_probs_across_borders
        with pytest.raises(RuntimeError, match="retained executor failure"):
            private.predict_distribution(model, query)
        assert sdk.translate_probs_across_borders is active
    assert sdk.TabPFNRegressor.predict is original
    assert proof.input_identity(model, query) == before


def test_fp32_scope_is_explicit():
    model, query = fixture()
    model.forced_inference_dtype_ = torch.float64
    with pytest.raises(private.DistributionOnlyError, match="frozen FP32"):
        private.predict_distribution(model, query)


def test_changed_sdk_method_alias_is_rejected_and_restored(monkeypatch):
    private.execution_metadata()
    model, query = fixture()
    original = sdk.TabPFNRegressor.predict
    with monkeypatch.context() as patch:
        patch.setattr(sdk.TabPFNRegressor, "predict", lambda *a, **k: None)
        with pytest.raises(private.DistributionOnlyError, match="SDK class, method alias"):
            private.predict_distribution(model, query)
    assert sdk.TabPFNRegressor.predict is original


def test_changed_source_binding_is_rejected_without_site_package_edits(monkeypatch):
    model, query = fixture()
    with monkeypatch.context() as patch:
        patch.setitem(private.PINNED_SOURCES, "regressor.py", "0" * 64)
        with pytest.raises(private.DistributionOnlyError, match="source hash"):
            private.predict_distribution(model, query)
    private.execution_metadata()


@pytest.mark.parametrize("before,after", [
    ("logits = logits.cpu()", "logits = logits.float().cpu()"),
    ("mean_out = typing.cast", "different_mean = typing.cast"),
    ('@config_context(transform_output="default")', '@config_context(transform_output="pandas")'),
])
def test_prefix_suffix_or_decorator_ast_change_is_rejected(before, after):
    source = Path(sdk.__file__).read_text()
    assert before in source
    changed = source.replace(before, after)
    with pytest.raises(private.DistributionOnlyError, match="predict AST"):
        private._compile_clone(changed, str(sdk.__file__))


def test_cancel_mask_path_keeps_actual_logits_and_inputs_bitwise():
    model, query = fixture()
    standard = model.bardist_.borders.detach().numpy().copy()
    broken = standard.copy()
    broken[0] = np.nan
    model.executor_.configs[0].target_transform = proof.RetainedInverseBorders(standard, broken)
    generators = {"fixture": torch.Generator(device="cpu").manual_seed(19)}
    with finite_cdf_compatibility():
        result = proof.prove_case(model, query, stage=0, name="synthetic-cancel-mask",
                                  members=range(8), repeated=False, torch=torch,
                                  sdk=sdk, generators=generators)
    assert result["logits_bitwise_identical"]
    assert result["input_bits_and_tensor_versions_unchanged"]


def test_sdk_configuration_context_is_retained():
    from sklearn import config_context, get_config
    model, query = fixture()
    with finite_cdf_compatibility():
        with config_context(transform_output="pandas"):
            before = get_config()
            actual = private.predict_distribution(model, query)
            assert get_config() == before
    assert actual["criterion"] is model.renormalized_criterion_


def test_actual_original_configuration_decorator_is_reused(monkeypatch):
    private.execution_metadata()
    model, query = fixture()
    def unrelated_factory(*args, **kwargs):
        raise AssertionError("a later module alias is not the original method decorator")
    with finite_cdf_compatibility():
        with monkeypatch.context() as patch:
            patch.setattr(sdk, "config_context", unrelated_factory)
            expected = model.predict(query, output_type="full", quantiles=[])
            actual = private.predict_distribution(model, query)
    assert bitwise(expected["logits"], actual["logits"])


def test_import_does_not_import_torch_or_sdk():
    code = ("import sys; import posteriorforge.execution_distribution_only;"
            "assert 'torch' not in sys.modules; assert 'tabpfn' not in sys.modules")
    subprocess.run([sys.executable, "-c", code], check=True)
