"""CPU-only fail-closed qualification metadata and reference-coverage tests."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest

path=Path(__file__).resolve().parents[1]/"scripts/qualify_execution_gqa_hybrid.py"
spec=importlib.util.spec_from_file_location("gqa_hybrid_qualification_tests",path)
q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)


def comparison():
    return {"passed":True,"members":[{"passed":True} for _ in range(8)],"inverse_cdf":{"passed":True,
        "icdf_contract_version":2,"independent_fp32_replay_failures":0,"same_logits_and_borders_exact_failures":0}}


def test_inverse_contract_does_not_trust_passed_boolean():
    value=comparison();assert q.prediction_passes(value)
    for key in ["independent_fp32_replay_failures","same_logits_and_borders_exact_failures"]:
        broken=deepcopy(value);broken["inverse_cdf"][key]=1;assert not q.prediction_passes(broken)
    value["members"][3]["passed"]=False;assert not q.prediction_passes(value)


def valid_row():
    return {"stage_index":13,"normalization_train_only_verified":True,"preprocessor_identity":True,"prepared_state_unchanged":True,"rng_unchanged":True,
        "fresh_control_vs_saved_golden":comparison(),"stress_control_vs_saved_golden":comparison(),
        "comparisons":[{"query_rows":n,"candidate_vs_fresh_control1024_slice":comparison(),"actual_gqa_execution":{"passed":True}} for n in q.ROWS],
        "stress_comparison":comparison(),"stress_gqa_execution":{"passed":True},"component_qualification":{"passed":True},
        "stress_component_qualification":{"passed":True},"outlier_checks":[]}


def test_stage_screen_allows_no_aggregate_cancellation_or_state_rng_excuse():
    row=valid_row();assert q.stage_passes(row)
    for key in ["normalization_train_only_verified","preprocessor_identity","prepared_state_unchanged","rng_unchanged"]:
        broken=deepcopy(row);broken[key]=False;assert not q.stage_passes(broken)
    row["comparisons"][2]["candidate_vs_fresh_control1024_slice"]["members"][6]["passed"]=False
    assert not q.stage_passes(row)


def test_stage_zero_requires_full_mixture_dedup_and_outliers_cannot_be_ignored():
    row=valid_row();row["stage_index"]=0;assert not q.stage_passes(row)
    row["stage0_shared_distribution"]={"passed":True};assert q.stage_passes(row)
    row["outlier_checks"]=[{"passed":False}];assert not q.stage_passes(row)


def context_descriptor(expanded):
    return {**{k:True for k in ("original_alias_restored","original_function_restored","original_code_unchanged","active_code_unchanged",
        "gqa_source_unchanged","sdk_sources_unchanged","source_guard_passed")},"attention_call_count":12,"expanded_attention_calls":12 if expanded else 0,
        "executed_original_attention_code_sha256":"original","original_attention_code_sha256":"original","active_attention_code_sha256":"wrapper"}


def test_actual_gqa_path_requires_seven_expansions_and_one_original_fallback():
    candidate={"component_predictions":[{"gqa_attention_descriptor":context_descriptor(i!=5)} for i in range(8)]}
    assert q.gqa_checks(candidate,10)["passed"]
    candidate["component_predictions"][5]["gqa_attention_descriptor"]["expanded_attention_calls"]=1
    assert not q.gqa_checks(candidate,10)["passed"]
    for item in candidate["component_predictions"]:item["gqa_attention_descriptor"]=context_descriptor(True)
    assert q.gqa_checks(candidate,13)["passed"]
    candidate["component_predictions"][6]["gqa_attention_descriptor"]["original_alias_restored"]=False
    assert not q.gqa_checks(candidate,13)["passed"]


def proof_fixture(tmp_path):
    golden={k:{"bound":k} for k in q.BINDINGS};golden["logical_row_indices"]=[0,48]
    rows=[{"stage_index":i,"comparisons":[{"query_rows":n,"control_chunk_vs_1024":comparison()} for n in q.ROWS],
           "baseline_model_contract_after":{"input_normalization_train_only_verified":True}} for i in range(21)]
    source={"status":"conditional_mismatch","sources_unchanged":True,"compatibility_adapter":{"original_alias_restored":True,"sdk_sources_unchanged":True},"stage_results":rows}
    archive=tmp_path/"original.json";archive.write_text(json.dumps(source));digest=q.sha(archive)
    proof={**golden,"schema":"posteriorforge.execution_hybrid_qualification.v1","status":"hybrid_screen_pass","assembler_source_sha256":q.ASSEMBLER_SHA,
        "sources_unchanged":True,"thresholds":q.THRESHOLDS,"row_counts":list(q.ROWS),"icdf_contract_version":2,
        "fixed_prefix_row_indices":golden["logical_row_indices"],"source_inputs":[{"archived_manifest":archive.name,"sha256":digest,"producer_status":"conditional_mismatch"}],
        "stage_results":deepcopy(rows),"stage_routes":{str(i):{"source_sha256":digest} for i in range(21)}}
    path=tmp_path/"manifest.json";path.write_text(json.dumps(proof));return path,golden,proof,archive


def test_control_proof_authenticates_original_failed_status_without_upgrading_it(tmp_path):
    path,golden,proof,archive=proof_fixture(tmp_path)
    binding,_=q.bind_control_proof(path,golden)
    assert len(binding["stage_evidence"])==21
    assert all(r["producer_status_preserved"]=="conditional_mismatch" and r["candidate_proof_inherited"] is False for r in binding["stage_evidence"])
    archive.write_text(archive.read_text()+" ")
    with pytest.raises(RuntimeError,match="archive mismatch"):q.bind_control_proof(path,golden)


def test_control_proof_rejects_changed_cohort_selected_metrics_or_threshold(tmp_path):
    path,golden,proof,archive=proof_fixture(tmp_path)
    proof["stage_results"][0]["comparisons"][0]["control_chunk_vs_1024"]["passed"]=False;path.write_text(json.dumps(proof))
    with pytest.raises(RuntimeError,match="selected original"):q.bind_control_proof(path,golden)
    proof["stage_results"][0]["comparisons"][0]["control_chunk_vs_1024"]["passed"]=True
    proof["thresholds"]={**q.THRESHOLDS,"max_cdf_absolute":3e-5};path.write_text(json.dumps(proof))
    with pytest.raises(RuntimeError,match="settings"):q.bind_control_proof(path,golden)


def test_golden_metadata_cannot_claim_saved1024_member_coverage():
    g={"schema":"posteriorforge.execution_control_goldens.v1","status":"complete","full21_original_control_captured":True,
        "sources_unchanged":True,"producer_sha256":q.GOLDEN_HELPER_SHA,"member_rows_retained":128,"stage_coverage":list(range(21)),
        "compatibility_adapter":{"original_alias_restored":True,"sdk_sources_unchanged":True,"executed_corrected_cdf_code_sha256":"fixed","corrected_cdf_code_sha256":"fixed","patched_call_count":8}}
    q.validate_golden_header(g);g["member_rows_retained"]=1024
    with pytest.raises(RuntimeError,match="member coverage"):q.validate_golden_header(g)


def test_source_guarded_helpers_import_without_torch():
    assert q.load_helper("qualify_execution_cache.py",q.BASE_SHA).THRESHOLDS==q.THRESHOLDS
    assert q.load_helper("qualify_execution_hybrid.py",q.HYBRID_HELPER_SHA)
    with pytest.raises(RuntimeError,match="helper source"):q.load_helper("qualify_execution_cache.py","wrong")


def test_golden_reconstruction_uses_actual_translation_input_not_raw_model_logits():
    torch=pytest.importorskip("torch")
    criterion=type("Criterion",(),{"borders":torch.tensor([0.,1.,3.])})()
    config={"type":"source.Config","repr":"original8member","pickle_sha256":"config-hash"}
    arrays={"physical_borders":np.array([0.,1.,3.],np.float32),"final_logits":np.zeros((2,2),np.float32)}
    for i in range(8):
        for key,value in [("translation_input_logits",np.ones((2,2),np.float32)*3),("raw_model_logits",np.ones((2,2),np.float32)*17),
            ("translated_probabilities",np.ones((2,2),np.float32)/2),("from_borders",np.array([0.,1.,3.],np.float32)),("to_borders",np.array([0.,1.,3.],np.float32))]:arrays[f"member_{i}_{key}"]=value
    entry={"metadata":{"member_rows":2,"members":[{"configuration":config} for _ in range(8)]}}
    prediction=q.golden_prediction(arrays,entry,criterion,torch)
    assert torch.all(prediction["members"][0]["raw_logits"]==3)
    assert prediction["members"][0]["configuration_identity"]["sha256"]=="config-hash"


def test_executed_attention_fingerprint_is_required_not_just_counters():
    value=context_descriptor(True);assert q.context_passes(value,expanded=True)
    value["executed_original_attention_code_sha256"]="wrong";assert not q.context_passes(value,expanded=True)


def test_fixed_chunk_coverage_is_not_replaceable_by_four_equal_counts():
    row=valid_row();row["comparisons"][0]["query_rows"]=1024;assert not q.stage_passes(row)
