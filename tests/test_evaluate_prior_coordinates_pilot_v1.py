"""CPU guard regressions for the new source-distinct read-only evaluator."""
import importlib.util
from pathlib import Path
import pytest

path=Path(__file__).resolve().parents[1]/"scripts/evaluate_prior_coordinates_pilot_v1.py"
spec=importlib.util.spec_from_file_location("test_prior_eval",path)
e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)


def summary():
    return {"canonical_borders_dtype":"torch.float32","physical_borders_dtype":"torch.float32",
        "temperature":.9,"contract":{"input_normalization_train_only_verified":True,
        "modules":[{"type":"InputNormalizationEncoderStep","tensors":{"mean":{"dtype":"torch.float32"}}}]}}


def test_unpinned_parent_cannot_assess():
    with pytest.raises(ValueError,match="not pinned"):
        e.source_guard()


def test_actual_train_only_fp32_contract():
    e.model_guard(summary())


@pytest.mark.parametrize("field,value",[
    ("canonical_borders_dtype","torch.float16"),("physical_borders_dtype","torch.float64"),("temperature",1.)])
def test_model_contract_drift_rejected(field,value):
    model=summary();model[field]=value
    with pytest.raises(ValueError):e.model_guard(model)


def test_normalizer_cast_or_test_normalization_rejected():
    model=summary();model["contract"]["modules"][0]["tensors"]["mean"]["dtype"]="torch.float16"
    with pytest.raises(ValueError,match="normalizer"):e.model_guard(model)
    model=summary();model["contract"]["input_normalization_train_only_verified"]=False
    with pytest.raises(ValueError,match="train-only"):e.model_guard(model)


def test_display_focus_is_parent_preregistered_and_four_curves():
    assert e.FOCUS==(0,7,8,10)
    assert len(e.LABELS)==len(e.COLORS)==4
    assert "OLD physical-target mixed FP16/FP32" in e.LABELS[0]
    assert "NEW causal prior coordinates: cached FP32"==e.LABELS[1]
    assert len(e.DISPLAY_DOC_SHA256)==64
    assert "all8 native stage14" not in path.read_text()
