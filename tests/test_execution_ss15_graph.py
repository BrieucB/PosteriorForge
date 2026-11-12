"""Synthetic engineering arrays only; no model, posterior, or reference draws."""
import copy
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from posteriorforge import execution_ss15_graph as graph
from posteriorforge import performance_ss15 as pilot


def arrays():
 theta=np.arange(4096*21,dtype=np.float32).reshape(4096,21)/31
 obs=np.arange(4096*15,dtype=np.float32).reshape(4096,15)/17
 return theta,obs


def test_all21_masks_keep_exact_full_block_structure():
 expected=[(range(3,7),range(4)),(range(7,11),range(4,8)),(range(11,15),range(8,12)),
           (range(15,17),[12]),(range(17,19),[13]),(range(19,21),[14])]
 assert [j for block,_ in expected for j in block]==list(range(3,21))
 for physical,ss in expected:
  for stage in physical:
   p=graph.stage_policy(stage)
   assert p["observation_columns_kept"]==list(ss)
   assert p["physical_prefix_indices_kept"]==[0,1,2]+[j for j in physical if j<stage]
   assert p["input_width"]==15+stage
 assert len(graph.policy_metadata()["stages"])==21


def test_train_query_same_mask_targets_and_saved_inputs_unchanged_all21():
 theta,obs=arrays();original_theta=theta.copy();original_obs=obs.copy()
 for stage in range(21):
  mt,mo=graph.mask_training(theta,obs,stage)
  raw=np.concatenate((obs,theta[:,:stage]),axis=1)
  masked=graph.mask_query(raw,stage)
  assert masked.tobytes()==np.concatenate((mo,mt[:,:stage]),axis=1).tobytes()
  assert mt[:,stage:].tobytes()==theta[:,stage:].tobytes()
  kept=graph.stage_policy(stage)["feature_columns_kept"]
  removed=graph.stage_policy(stage)["feature_columns_zeroed"]
  assert masked[:,kept].tobytes()==raw[:,kept].tobytes()
  assert (masked[:,removed]==0).all()
  binding=graph.training_bindings(theta,obs,mt,mo,stage)
  assert binding["original_target"]==binding["prepared_target"]
 assert np.array_equal(theta,original_theta) and np.array_equal(obs,original_obs)


def test_global_stages_are_object_and_byte_identity():
 theta,obs=arrays()
 for stage in range(3):
  mt,mo=graph.mask_training(theta,obs,stage)
  query=np.concatenate((obs[:9],theta[:9,:stage]),axis=1)
  assert mt is theta and mo is obs and graph.mask_query(query,stage) is query


def test_other_block_data_cannot_enter_local_predictors():
 theta,obs=arrays()
 for stage in range(3,21):
  raw=np.concatenate((obs[:5],theta[:5,:stage]),axis=1)
  changed=raw.copy()
  changed[:,graph.stage_policy(stage)["feature_columns_zeroed"]]=np.nan
  assert graph.mask_query(changed,stage).tobytes()==graph.mask_query(raw,stage).tobytes()


def test_used_nan_passed_unchanged_infinity_rejected():
 query=np.zeros((2,24),np.float32);query[0,4]=np.nan
 masked=graph.mask_query(query,9)
 assert np.isnan(masked[0,4])
 query[0,4]=np.inf
 with pytest.raises(ValueError):graph.mask_query(query,9)


@pytest.mark.parametrize("stage",[-1,21,True,1.0])
def test_invalid_stage_rejected(stage):
 with pytest.raises(ValueError):graph.stage_policy(stage)


def test_source_and_frozen_base_method_guard(monkeypatch):
 graph.require_sources()
 monkeypatch.setattr(pilot.SS15StageAdapter,"prepare",lambda *a:None)
 with pytest.raises(ValueError,match="method changed"):graph.require_sources()


def test_adapter_delegates_only_masked_features_with_original_target(monkeypatch):
 theta,obs=arrays();stage=12;seen={}
 engine=object.__new__(graph.GraphLocalStageAdapter)
 engine._graph_source=graph.source_sha256();engine.graph_stage_inputs=[]
 def prepare(self,t,o,j):
  seen.update(theta=t,obs=o,stage=j)
  return {"fake_engineering_prepare":True}
 monkeypatch.setattr(pilot.SS15StageAdapter,"prepare",prepare)
 engine.prepare(theta,obs,stage)
 assert seen["theta"][:,stage].tobytes()==theta[:,stage].tobytes()
 assert seen["stage"]==stage
 assert np.array_equal(seen["obs"][:,:8],np.zeros((4096,8),np.float32))
 assert engine.graph_stage_inputs[0]["target_bytes_unchanged"]


def test_torch_free_import_and_policy():
 command="from posteriorforge import execution_ss15_graph as g;import sys;g.require_sources();g.policy_metadata();assert 'torch' not in sys.modules"
 subprocess.run([sys.executable,"-c",command],check=True)
