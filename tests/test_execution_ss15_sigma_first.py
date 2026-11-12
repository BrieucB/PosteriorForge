"""CPU engineering-only ordering/masking proofs; no model or posterior calls."""
import subprocess
import sys
import numpy as np
import pytest
from posteriorforge import execution_ss15_sigma_first as s


def arrays():
 physical=np.arange(4096*21,dtype=np.float32).reshape(4096,21)/31
 obs=np.arange(4096*15,dtype=np.float32).reshape(4096,15)/17
 return physical,s.order_theta(physical),obs


def test_exact_predeclared_permutation_and_fallback_physical_bindings():
 assert list(s.ORDER)==[0,1,2,6,3,4,5,10,7,8,9,14,11,12,13,16,15,18,17,20,19]
 assert sorted(s.ORDER)==list(range(21))
 assert list(s.INVERSE_ORDER)==list(np.argsort(s.ORDER))
 assert s.fallback_bindings()==[
  {"stage":10,"member":5,"target_physical_index":9,"target_physical_name":"d0_m2","coordinate_system":"sigma-first stage index"},
  {"stage":20,"member":4,"target_physical_index":19,"target_physical_name":"a_a3","coordinate_system":"sigma-first stage index"}]


def test_roundtrip_and_original_physical_bank_unmodified():
 physical,ordered,obs=arrays();original=physical.copy()
 assert ordered.tobytes()==physical[:,s.ORDER].copy().tobytes()
 assert s.restore_physical(ordered).tobytes()==physical.tobytes()
 assert not np.shares_memory(physical,ordered)
 assert np.array_equal(physical,original)


def test_complete_proposal_inverse_preserves_exceptional_bits_without_rejection():
 bits=np.arange(63,dtype=np.uint64).reshape(3,21).copy()
 bits[0,6]=0x7ff8000000000023
 bits[1,16]=0xfff0000000000000
 bits[2,1]=0x8000000000000000
 physical=bits.view(np.float64)
 ordered=np.ascontiguousarray(physical[:,s.ORDER])
 restored=s.restore_physical(ordered)
 assert restored.view(np.uint64).tobytes()==bits.tobytes()
 assert restored.shape==(3,21) and np.isnan(restored[0,6]) and np.isneginf(restored[1,16])
 with pytest.raises(ValueError):s.restore_physical(ordered[:,:20])


def test_globals_used_predictors_and_labels_unchanged_despite_future_permutation():
 physical,ordered,obs=arrays()
 for stage in range(3):
  mt,mo=s.mask_training(ordered,obs,stage)
  assert mt is ordered and mo is obs
  assert mt[:,stage].tobytes()==physical[:,stage].tobytes()
  assert np.concatenate((mo,mt[:,:stage]),axis=1).tobytes()==np.concatenate((obs,physical[:,:stage]),axis=1).tobytes()
 assert not np.array_equal(physical[:,3:],ordered[:,3:])


def test_all6_sigma_first_blocks_all3_globals_and_own_ordered_prefix():
 physical,ordered,obs=arrays()
 noise_stages=[3,7,11,15,17,19]
 for stage in range(21):
  p=s.stage_policy(stage)
  assert p["target_physical_index"]==s.ORDER[stage]
  assert "target_physical_index_unchanged" not in p
  if stage>=3:
   assert p["ordered_prefix_indices_kept"][:3]==[0,1,2]
   if stage in noise_stages:assert p["ordered_prefix_indices_kept"]==[0,1,2]
   else:assert any(s.ORDER[j] in [6,10,14,16,18,20] for j in p["ordered_prefix_indices_kept"])
  mt,mo=s.mask_training(ordered,obs,stage)
  raw=np.concatenate((obs,ordered[:,:stage]),axis=1)
  actual=np.concatenate((mo,mt[:,:stage]),axis=1)
  assert actual.tobytes()==s.mask_query(raw,stage).tobytes()
  assert mt[:,stage:].tobytes()==ordered[:,stage:].tobytes()
  assert mt[:,stage].tobytes()==physical[:,s.ORDER[stage]].tobytes()
  binding=s.training_bindings(ordered,obs,mt,mo,stage)
  assert binding["original_target"]==binding["prepared_target"] and binding["target_bytes_unchanged"]
  assert binding["target_physical_index"]==s.ORDER[stage]


def test_other_block_inputs_deleted_used_nan_passed():
 _,ordered,obs=arrays()
 for stage in range(3,21):
  raw=np.concatenate((obs[:2],ordered[:2,:stage]),axis=1)
  changed=raw.copy();changed[:,s.stage_policy(stage)["feature_columns_zeroed"]]=np.nan
  assert s.mask_query(raw,stage).tobytes()==s.mask_query(changed,stage).tobytes()
 raw=np.zeros((2,25),np.float32);raw[0,4]=np.nan
 assert np.isnan(s.mask_query(raw,10)[0,4])


def test_torch_free_source_and_policy():
 s.require_sources()
 assert s.policy_metadata()["learned_target_order_changed"] is True
 command="from posteriorforge import execution_ss15_sigma_first as s;import sys;s.require_sources();s.policy_metadata();assert 'torch' not in sys.modules"
 subprocess.run([sys.executable,"-c",command],check=True)
