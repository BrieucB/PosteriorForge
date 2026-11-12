"""CPU saved-cohort provenance and local-control checks for joint-global plots."""
import copy,importlib.util
from pathlib import Path
import numpy as np
import pytest

SPEC=importlib.util.spec_from_file_location("_joint_global_assessment",Path(__file__).resolve().parents[1]/"scripts/evaluate_ss15_global_block.py")
m=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(m)

def assessment():
 return {"marginal_cdf":{"lower":[.01]*21,"upper":[.02]*21},
  "moments":{k:[.1]*21 for k in ("sample_mean","sample_sd","absolute_bias_reference_sd","sd_ratio")}}

def test_only_three_globals_may_change_and_signed_zero_local_counts_as_change():
 before=np.zeros((50000,21),np.float64);after=before.copy();after[:,:3]=1
 assert m.local18_equal(before,after)
 after[0,4]=-0.
 assert not m.local18_equal(before,after)
 assert not m.local18_equal(before,before.astype(np.float32))
 assert not m.local18_equal(before[:-1],before[:-1])

def test_three_global_marginal_assessments_may_change():
 before=assessment();after=copy.deepcopy(before)
 for v in after["marginal_cdf"].values():v[:3]=[.8]*3
 for v in after["moments"].values():v[:3]=[.8]*3
 assert m.unchanged_marginals(before,after)["passed"]

@pytest.mark.parametrize("family,key",[("marginal_cdf","lower"),("marginal_cdf","upper"),
 ("moments","sample_mean"),("moments","sample_sd"),("moments","absolute_bias_reference_sd"),("moments","sd_ratio")])
def test_any_local_marginal_or_moment_change_is_rejected(family,key):
 before=assessment();after=copy.deepcopy(before);after[family][key][4]+=1e-15
 with pytest.raises(ValueError,match="18 unchanged"):m.unchanged_marginals(before,after)

@pytest.mark.parametrize("family",["graph","control"])
def test_actual_both_saved_cohorts_replay_private_draws_and_all18_bytes(family):
 phase=Path("/scratch/project/eu-26-17/eubrieucb/posteriorforge/results/accuracy-ss15-v1")
 path=phase/m.CASES[family]["hybrid_dir"]/"manifest.json"
 if not path.is_file():pytest.skip("saved Karolina joint-global cohort unavailable")
 _,_,before,after,receipt=m.authenticate(phase,family)
 assert receipt["private_matrix_normal_replay"] and receipt["all18_columns_byte_identical"]
 assert not receipt["pure_NPE"]
 assert m.local18_equal(before,after) and not np.array_equal(before[:,:3],after[:,:3])

def test_modified_manifest_is_rejected_before_payload_read(tmp_path):
 path=tmp_path/m.CASES["graph"]["hybrid_dir"];path.mkdir();(path/"manifest.json").write_text("{}")
 with pytest.raises(ValueError,match="hybrid manifest differs"):m.authenticate(tmp_path,"graph")
