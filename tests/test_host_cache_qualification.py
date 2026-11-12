"""No-model tests for bounded host qualification custody and fail-closed flags."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest


def helper():
    path=Path(__file__).parents[1]/"scripts"/"qualify_execution_host_cache.py"
    spec=importlib.util.spec_from_file_location("host_qual_test",path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def valid_proof():
    return {"schema":"posteriorforge.execution_hybrid_qualification.v1","status":"hybrid_screen_pass","qualified":True,
            "full21_hybrid_qualified":True,"full21_component_qualified":True,"full21_sampler_qualified":True,
            "full21_qualified":False,"cache_only_full21_qualified":False,"icdf_contract_version":2,
            "fallback_map":{"10":[5],"13":[6]},"stage_results":[{"stage_index":stage} for stage in range(21)]}


def test_authentication_rejects_cache_pass_substitution_and_incomplete_coverage():
    h=helper();h.validate_hybrid_qualification(valid_proof())
    for key,value in (("full21_qualified",True),("full21_hybrid_qualified",False),("icdf_contract_version",1),("fallback_map",{"10":[5]})):
        proof=deepcopy(valid_proof());proof[key]=value
        with pytest.raises(ValueError):h.validate_hybrid_qualification(proof)
    proof=valid_proof();proof["stage_results"].pop()
    with pytest.raises(ValueError,match="all21"):h.validate_hybrid_qualification(proof)


def test_frozen_numerical_utilities_load_without_model_calls():
    utility,hybrid=helper().frozen_helpers()
    assert utility.THRESHOLDS["max_cdf_absolute"]==2e-5
    assert callable(hybrid.hybrid_prediction)


def test_complete_migration_inventory_is_archived_and_hashed(tmp_path):
    h=helper();event={"operation":"offload","stage_index":10,"seconds":.1,"copied_bytes":24,
                     "comparison":{"passed":True,"full_storage_bytes_equal":True},"before_state":{"device":"cuda:0"},"state":{"device":"cpu"}}
    rows=h.compact_copy_events(SimpleNamespace(copy_events=[event]),tmp_path,0)
    import gzip,hashlib
    path=tmp_path/rows[0]["inventory_file"]
    with gzip.open(path,"rt") as handle:assert json.load(handle)==event
    assert hashlib.sha256(path.read_bytes()).hexdigest()==rows[0]["inventory_sha256"]
    assert h.compact_copy_events(SimpleNamespace(copy_events=[event]),tmp_path,1)==[]


def test_exact_replay_reports_nans_and_real_changes():
    torch=pytest.importorskip("torch");h=helper()
    member={name:torch.tensor([[1.,float("nan")]]) for name in ("raw_logits","from_borders","to_borders","translated_probabilities")}
    prediction={"logits":torch.tensor([[2.,3.]]),"members":[member]*8}
    assert h.prediction_exact(prediction,deepcopy(prediction),torch)
    changed=deepcopy(prediction);changed["logits"][0,0]+=1
    assert not h.prediction_exact(prediction,changed,torch)


def test_model_contract_rejects_changed_normalizer_precision_or_autocast():
    h=helper();valid={"resident":{"input_normalization_train_only_verified":True,"autocast":False,"forced_inference_dtype":"torch.float32"}}
    assert h.valid_model_contract(valid)
    for key,value in (("input_normalization_train_only_verified",False),("autocast",True),("forced_inference_dtype","torch.float16")):
        changed=deepcopy(valid);changed["resident"][key]=value
        assert not h.valid_model_contract(changed)


def test_valid_probability_check_allows_log_zero_and_rejects_nan():
    torch=pytest.importorskip("torch");h=helper();u,_=h.frozen_helpers()
    member={"translated_probabilities":torch.tensor([[0.,1.]])}
    prediction={"logits":torch.tensor([[float("-inf"),0.]]),"members":[member]*8}
    assert h.valid_prediction_probabilities(prediction,u)["passed"]
    prediction["logits"][0,0]=float("nan")
    assert not h.valid_prediction_probabilities(prediction,u)["passed"]


def test_logical_state_allows_original_buffer_identity_replacement_but_rejects_values():
    torch=pytest.importorskip("torch");h=helper();state=SimpleNamespace(buffer=torch.arange(8,dtype=torch.float32))
    before=h.logical_tensor_state(state,torch);original_id=id(state.buffer)
    state.buffer=state.buffer.clone()
    assert id(state.buffer)!=original_id and h.compare_logical_state(before,h.logical_tensor_state(state,torch))["passed"]
    state.buffer[0]+=1
    assert not h.compare_logical_state(before,h.logical_tensor_state(state,torch))["passed"]
