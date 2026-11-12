"""Saved-cohort binding and unchanged-coordinate controls for the scalar hybrid."""
import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

_PATH=Path(__file__).resolve().parents[1]/"scripts/evaluate_ss15_mu_a.py"
_SPEC=importlib.util.spec_from_file_location("_test_mu_a_assessment",_PATH)
M=importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(M)


def fixture_assessment():
    return {"marginal_cdf":{"lower":[.01]*21,"upper":[.02]*21},
        "moments":{key:[.1]*21 for key in ("sample_mean","sample_sd",
        "absolute_bias_reference_sd","sd_ratio")}}


def test_only_mu_a_change_is_allowed_with_complete_float64_rows():
    before=np.zeros((50000,21),dtype=np.float64)
    after=before.copy();after[:,0]=1.
    assert M.other20_equal(before,after)
    after[-1,12]=1.
    assert not M.other20_equal(before,after)


def test_bit_identity_detects_signed_zero_and_dtype_or_count_changes():
    before=np.zeros((50000,21),dtype=np.float64)
    after=before.copy();after[0,4]=-0.
    assert np.array_equal(before,after)
    assert not M.other20_equal(before,after)
    assert not M.other20_equal(before,before.astype(np.float32))
    assert not M.other20_equal(before[:-1],before[:-1])


def test_mu_a_marginal_assessment_may_change_but_other20_must_not():
    before=fixture_assessment();after=copy.deepcopy(before)
    for values in after["marginal_cdf"].values():values[0]=.8
    for values in after["moments"].values():values[0]=.8
    assert M.unchanged_marginals(before,after)["passed"]


@pytest.mark.parametrize("family,key",[("marginal_cdf","lower"),("marginal_cdf","upper"),
    ("moments","sample_mean"),("moments","sample_sd"),
    ("moments","absolute_bias_reference_sd"),("moments","sd_ratio")])
def test_any_unchanged_local_marginal_change_is_rejected(family,key):
    before=fixture_assessment();after=copy.deepcopy(before)
    after[family][key][12]+=1e-15
    with pytest.raises(ValueError,match="unchanged20"):
        M.unchanged_marginals(before,after)


def test_actual_saved_hybrid_and_private_draw_replay_when_available():
    root=Path("/scratch/project/eu-26-17/eubrieucb/posteriorforge/results/accuracy-ss15-v1")
    manifest=root/"mu-a-gibbs-once-01/manifest.json"
    if not manifest.is_file():pytest.skip("saved Karolina pilot not present")
    parent=root/"pilot-01/seed-20261111-case-0/manifest.json"
    _,_,before,after,receipt=M.authenticate(manifest,parent)
    assert receipt["one_pass_private_normal_replay"]
    assert receipt["other20_byte_identical"]
    assert M.other20_equal(before,after)
    assert not np.array_equal(before[:,0],after[:,0])


def test_modified_hybrid_receipt_is_rejected_before_loading_payload(tmp_path):
    path=tmp_path/"manifest.json";path.write_text("{}")
    with pytest.raises(ValueError,match="hybrid manifest changed"):
        M.authenticate(path,path)
