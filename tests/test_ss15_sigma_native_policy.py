"""Private namespace/MRO/dtype classification tests; no model or saved cohort."""
import copy,json,random
import numpy as np
import pytest
from posteriorforge import execution_ss15_sigma_native_policy as n
from posteriorforge import performance_ss15 as pilot,performance_ss15_sigma_first as old
from posteriorforge import execution_ss15_sigma_first as order


def event(stage,member,dtype="torch.float32",phase="prefill"):
 return {"phase":phase,"stage":stage,"member":member,"count":1,"cache_kv":phase=="prefill",
  "projection_dtypes":{"q":dtype,"k":dtype,"v":dtype,"kv":None,"qkv":None},
  "autocast_enabled":dtype=="torch.float16","autocast_dtype":"torch.float16",
  "allow_tf32":False,"matmul_precision":"highest","supplied_cache_dtypes":[dtype],
  "strict_fp32_fallback":dtype=="torch.float32"}


def test_canonical_whole_stage_policy_and_current_physical_bindings():
 assert n.fallback_map()=={10:[5],14:list(range(8)),20:[4]}
 p=n.policy_metadata()
 assert p["native_stage_indices"]==[14] and p["ancestral_MAP2"]=={"10":[5],"20":[4]}
 targets=[r for r in p["active_fallback_stage_member_bindings"] if r["stage"]==14]
 assert len(targets)==8 and all(r["target_physical_index"]==13 and r["target_physical_name"]=="d0_m3" for r in targets)
 assert n.fallback_map((14,15))[15]==list(range(8))


@pytest.mark.parametrize("bad",[(),(15,),(14,14),(14,21),(True,14)])
def test_native_stages_declaration_is_explicit_fixed_local_whole_stage(bad):
 with pytest.raises(ValueError):n.canonical_native_stages(bad)


def test_identical_original_code_objects_private_globals_and_exact_diamond():
 n.require_sources();bundle=n._bundle()
 cls=bundle.adapter_class
 assert cls.__mro__[:4]==(cls,order.SigmaFirstStageAdapter,bundle.private_class,pilot.SS15StageAdapter)
 assert all(r["same_code_object"] and r["private_global_dictionary"] for r in n.clone_descriptors())
 for name in n._METHOD_NAMES:
  clone=getattr(bundle.private_class,name);original=getattr(pilot.SS15StageAdapter,name)
  assert clone.__code__ is original.__code__ and clone.__globals__ is not original.__globals__
 assert cls.prepare is order.SigmaFirstStageAdapter.prepare
 assert cls.predict_component is order.SigmaFirstStageAdapter.predict_component


def test_whole_stage_fp32_prefill_classification_is_private_not_oldmap():
 bundle=n._bundle()
 report={"mode":pilot.MODE,"witnesses":[event(14,m) for m in range(8)]}
 actual=bundle.clones["prefill_witness_passed"](report,14)
 assert actual["passed"] and all(r["passed"] for r in actual["members"])
 assert not pilot.prefill_witness_passed(report,14)["passed"]
 assert all(bundle.clones["is_fallback"](14,m) for m in range(8))
 assert not any(pilot.is_fallback(14,m) for m in range(8))


def test_actual_native_query_requires_fp32_and_rejects_fp16():
 fn=n._bundle().clones["query_witness_delta"]
 before={}
 after={json.dumps(event(14,3,phase="query"),sort_keys=True):1}
 assert fn(before,after,stage=14,member=3)["computed_projection_dtype"]=="torch.float32"
 bad={json.dumps(event(14,3,"torch.float16","query"),sort_keys=True):1}
 with pytest.raises(RuntimeError):fn(before,bad,stage=14,member=3)


def test_private_validator_accepts_new_execution_id_and_frozen_order_proof():
 prediction={"ss15_execution_proof":{"mode":pilot.MODE,"SS15_execution_module_sha256":old.SS15_EXECUTION_SHA256,
  "ancestral_precision_module_sha256":old.pilot.PRECISION_SHA,"module_sha256":n.source_sha256(),
  "stage":14,"member":3,"actual_head_contexts":1,"corrected_CDF_calls":1,"witness":{"passed":True}},
  "sigma_first_input_proof":{"module_sha256":order.source_sha256(),
   "policy_sha256":order.policy_metadata()["policy_sha256"],"stage":14,"member":3,
   "mask":order.stage_policy(14),"training_query_same_fixed_mask":True,
   "ordered_target_binding_verified":True,"target_physical_index":13,"physical_head_contexts":1}}
 report={};n.validate_prediction(prediction,14,3,report=report)
 assert report["actual_head_contexts"]==report["sigma_first_physical_head_contexts"]==1
 bad=copy.deepcopy(prediction);bad["ss15_execution_proof"]["module_sha256"]=pilot.source_sha256()
 with pytest.raises(RuntimeError):n.validate_prediction(bad,14,3,report={})


def test_private_policy_tampering_fails_without_touching_originals():
 bundle=n._bundle((14,15));original=copy.deepcopy(pilot.FALLBACK_MAP)
 bundle.scope["FALLBACK_MAP"][14]=[0]
 try:
  with pytest.raises(ValueError,match="private native execution globals"):bundle.guard()
 finally:bundle.scope["FALLBACK_MAP"][14]=list(range(8))
 bundle.guard()
 assert pilot.FALLBACK_MAP==old.FALLBACK_MAP==original=={10:[5],20:[4]}


def test_factory_does_not_consume_default_rng_or_import_torch():
 import pickle,sys
 before=pickle.dumps((np.random.get_state(),random.getstate()),protocol=5)
 n.adapter_class((14,16));n.clone_descriptors((14,16))
 after=pickle.dumps((np.random.get_state(),random.getstate()),protocol=5)
 assert before==after
 assert "torch" not in sys.modules


def test_real_constructor_metadata_labels_active_and_ancestral_policies():
 cls=n.adapter_class()
 engine=cls("unused-no-model-load",cdf_descriptor={})
 metadata=engine.metadata()
 assert metadata["active_native_execution_policy"]["native_stage_indices"]==[14]
 assert metadata["ancestral_order_mask_policy"]["fallback_stage_member_bindings"]==order.policy_metadata()["fallback_stage_member_bindings"]
 assert "sigma_first_input_policy" not in metadata
 assert metadata["policy"]["native_execution_policy"]["active_fallback_map"]["14"]==list(range(8))
 assert all(r["same_code_object"] for r in metadata["private_clone_descriptors"])
 engine.close()


def test_policy_metadata_replays_exactly_after_json_roundtrip():
 value=n.policy_metadata()
 assert json.loads(json.dumps(value,sort_keys=True))==value
