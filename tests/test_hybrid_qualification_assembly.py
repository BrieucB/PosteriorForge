"""Synthetic CPU proof assembly: never executes models or rewrites producers."""
import importlib.util
import json
from pathlib import Path
from copy import deepcopy
import pytest
from posteriorforge.artifacts import sha256_file,write_json
from posteriorforge.compat_cdf import adapter_source_sha256,PINNED_SDK_SHA256
from posteriorforge.inference import CHECKPOINT_SHA256
from posteriorforge.science.analytical_benchmark_v2 import build_problem

spec=importlib.util.spec_from_file_location("hybrid_assembly",Path(__file__).parents[1]/"scripts/assemble_hybrid_qualification.py")
assembly=importlib.util.module_from_spec(spec);spec.loader.exec_module(assembly)
SOURCES={key:str(index)*64 for index,key in enumerate(("cache_module_sha256","mixture_module_sha256",
    "hybrid_module_sha256","cache_qualifier_sha256","hybrid_qualifier_sha256"),1)}


def probability():
    return {"passed":True,"minimum_probability":0.,"max_normalization_error":0.,
            "max_cdf_absolute_error":0.,"max_row_total_variation":0.}


def prediction():
    return {"passed":True,"eight_members":True,"exact_final_borders":True,"final_float32":True,
        "final":probability(),"members":[{**probability(),"index":i,
            "configuration_identity_equal":True,"exact_borders":True,"float32":True} for i in range(8)],
        "inverse_cdf":{"icdf_contract_version":2,"passed":True,
            "same_logits_and_borders_exact_failures":0,"independent_fp32_replay_failures":0,"probability_space_failures":[]}}


def component(stage,hybrid=False):
    members=[]
    for index in range(8):
        members.append({**probability(),"ensemble_index":index,"configuration_identity_equal":True,
          "exact_borders":True,"float32":True,"component_raw_mass_matches_fp64_sum":True,
          "fast_icdf_same_logits_exact":True,"executed_corrected_helper_fingerprint_matches":True,
          "corrected_cdf_call_delta":1,"mass_min":1.,"mass_max":1.,
          "component_head_mass_envelope":1.00001,"component_raw_mass_sum_error":0.,
          "component_raw_mass_fp64_roundoff_bound":1e-10,"normalized_component":probability()})
    result={"schema":"posteriorforge.execution_component_qualification.v1",
       "module_sha256":SOURCES["hybrid_module_sha256" if hybrid else "mixture_module_sha256"],
       "passed":True,"members":members,"fast_icdf_same_logits_failures":0,
       "executor_metadata":{"full8_member_lists_restored":True},
       "uniform_head_without_mass_correction_is_not_authorized":True,"component_distribution_dedup_qualified":False,
       "roundoff":{"mass_corrected_reconstruction_passed":True,"component_mass_guard_passed":True},
       "stage0_dedup_scope":"full_mixture_only",
       "stage0_shared_distribution":{**probability(),"shared_icdf_same_logits_exact":True}}
    for key in ("normalized_mean_captured_raw_vs_full_fp32","normalized_mean_observed_raw_vs_full_fp32",
                "mass_corrected_observed_outputs_vs_full_fp32"):result[key]=probability()
    if hybrid:
        result.update(component_mixture_module_sha256=SOURCES["mixture_module_sha256"],
                      sampler_module_sha256=SOURCES["mixture_module_sha256"])
    return result


def row(stage,hybrid=False):
    return {"stage_index":stage,"passed":True,"parameter":"p"+str(stage),
      "comparisons":[{"query_rows":count,"cache_vs_same_chunk_control":prediction(),
                     "control_chunk_vs_1024":prediction()} for count in assembly.ROWS],
      "stress_comparison":prediction(),"component_qualification":component(stage,hybrid),
      "preprocessor_pickle_identity":True,"baseline_model_contract_after":{"input_normalization_train_only_verified":True},
      "cache_model_contract_after":{"input_normalization_train_only_verified":True},
      "normalization_train_only_verified":True,
      **{name:{"input_normalization_train_only_verified":True} for name in ("control_model_contract","resident_model_contract","uncached_model_contract")}}


def adapter():
    return {"adapter_sha256":adapter_source_sha256(),"source_guard_passed":True,
      "original_alias_restored":True,"sdk_sources_unchanged":True,"sdk_aliases_unchanged":True,
      "original_alias_identity_verified":True,"translator_code_unchanged":True,
      "patched_call_count":8,"original_cdf_code_sha256":"a"*64,"corrected_cdf_code_sha256":"b"*64,
      "executed_corrected_cdf_code_sha256":"b"*64,"active_translator_code_sha256":"c"*64,
      "original_translator_code_sha256":"c"*64,"sdk_source_sha256":dict(PINNED_SDK_SHA256),
      "sdk_source_sha256_after":dict(PINNED_SDK_SHA256)}


def fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(assembly,"current_sources",lambda:dict(SOURCES))
    problem=build_problem()
    header={"thresholds":assembly.THRESHOLDS,"row_counts":assembly.ROWS,"icdf_contract_version":2,
      "sources_unchanged":True,"compatibility_adapter":adapter(),"checkpoint_sha256":CHECKPOINT_SHA256,
      "scientific_fingerprint":problem.scientific_fingerprint,"parameter_order":list(problem.parameter_order),
      "context_theta":{"shape":[4096,21],"sha256":"theta"},"context_obs":{"shape":[4096,33],"sha256":"obs"},
      "query":{"shape":[1,33],"sha256":"query"},"prefix_manifest_sha256":"prefixmanifest",
      "prefix_samples_sha256":"prefixsamples","fixed_prefix_row_indices":list(range(1024))}
    cache={**deepcopy(header),"schema":assembly.CACHE_SCHEMA,"status":"conditional_mismatch",
      "qualified":False,"full21_qualified":False,"module_sha256":SOURCES["cache_module_sha256"],
      "mixture_module_sha256":SOURCES["mixture_module_sha256"],
      "sources":{"qualification_script_sha256":SOURCES["cache_qualifier_sha256"]},
      "stage_results":[row(stage) for stage in range(21)]}
    for stage in (10,13):
        cache["stage_results"][stage]["passed"]=False
        cache["stage_results"][stage]["comparisons"][-1]["cache_vs_same_chunk_control"]["passed"]=False
    cached=tmp_path/"cache-failed.json";write_json(cached,cache)
    hybrid_paths=[]
    fallback={"10":[5],"13":[6]}
    for stage,members in ((10,[5]),(13,[6])):
        routes={str(i):"uncached" if i in members else "cached" for i in range(8)}
        proof={**deepcopy(header),**row(stage,True),"schema":assembly.STAGE_SCHEMA,"status":"stage_pass",
          "fallback_members":members,"member_routes":routes,"fallback_map":fallback,
          "fallback_policy_sha256":assembly.fallback_policy_sha256(fallback),"fullstage8_component_qualified":True,
          "fullstage_sampler_qualified":True,**SOURCES,"qualifier_source_sha256":SOURCES["hybrid_qualifier_sha256"]}
        for member in proof["component_qualification"]["members"]:
            member.update(execution_route=routes[str(member["ensemble_index"])],route_counter_delta=1,other_route_counter_delta=0)
        path=tmp_path/f"hybrid-{stage}.json";write_json(path,proof);hybrid_paths.append(path)
    return cached,hybrid_paths,fallback


def test_assembles_new_hybrid_proof_without_reclassifying_failed_cache(tmp_path,monkeypatch):
    cached,hybrids,fallback=fixture(tmp_path,monkeypatch)
    original=cached.read_bytes();original_hash=sha256_file(cached)
    result=assembly.assemble(cache_proofs=[cached],hybrid_proofs=hybrids,fallback_map=fallback,output=tmp_path/"out")
    assert result["status"]=="hybrid_screen_pass" and result["full21_hybrid_qualified"]
    assert result["full21_component_qualified"] and result["full21_sampler_qualified"]
    assert result["full21_qualified"] is False and result["cache_only_full21_qualified"] is False
    assert result["stage_coverage"]==list(range(21))
    assert result["stage_routes"]["10"]["member_routes"]["5"]=="uncached"
    assert result["stage_routes"]["0"]["source_producer_status"]=="conditional_mismatch"
    assert cached.read_bytes()==original and sha256_file(cached)==original_hash
    archived=tmp_path/"out"/result["source_inputs"][0]["archived_manifest"]
    assert archived.read_bytes()==original
    assert json.loads(archived.read_text())["qualified"] is False


def test_missing_fresh_fallback_proof_never_substitutes_a_cached_row(tmp_path,monkeypatch):
    cached,hybrids,fallback=fixture(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match="missing passing hybrid stage13"):
        assembly.assemble(cache_proofs=[cached],hybrid_proofs=hybrids[:1],fallback_map=fallback,output=tmp_path/"out")
    result=json.loads((tmp_path/"out/manifest.json").read_text())
    assert result["status"]=="assembly_failed" and result["full21_hybrid_qualified"] is False


@pytest.mark.parametrize("change",["threshold","query","source","restoration","numeric","component","route","counter"])
def test_inconsistent_or_unqualified_input_fails_closed(tmp_path,monkeypatch,change):
    cached,hybrids,fallback=fixture(tmp_path,monkeypatch)
    target=cached if change not in ("route","counter") else hybrids[0]
    value=json.loads(target.read_text())
    if change=="threshold":value["thresholds"]["max_cdf_absolute"]=.1
    if change=="query":value["query"]["sha256"]="different"
    if change=="source":value["module_sha256"]="wrong"
    if change=="restoration":value["compatibility_adapter"]["original_alias_restored"]=False
    if change=="numeric":value["stage_results"][0]["comparisons"][0]["cache_vs_same_chunk_control"]["final"]["max_cdf_absolute_error"]=1e-3
    if change=="component":value["stage_results"][0]["component_qualification"]["members"].pop()
    if change=="route":value["member_routes"]["5"]="cached"
    if change=="counter":value["component_qualification"]["members"][5]["route_counter_delta"]=0
    write_json(target,value)
    with pytest.raises(ValueError):
        assembly.assemble(cache_proofs=[cached],hybrid_proofs=hybrids,fallback_map=fallback,output=tmp_path/"out")


def test_existing_assembled_output_is_preserved(tmp_path,monkeypatch):
    cached,hybrids,fallback=fixture(tmp_path,monkeypatch);out=tmp_path/"out";out.mkdir()
    marker=out/"manifest.json";marker.write_text("preserve")
    with pytest.raises(FileExistsError):
        assembly.assemble(cache_proofs=[cached],hybrid_proofs=hybrids,fallback_map=fallback,output=out)
    assert marker.read_text()=="preserve"


@pytest.mark.parametrize("issue",["replay","normalizer","massbound","policyhash"])
def test_strict_replay_normalizer_mass_bound_and_policy_guard(tmp_path,monkeypatch,issue):
    cached,hybrids,fallback=fixture(tmp_path,monkeypatch)
    path=hybrids[0] if issue=="policyhash" else cached
    value=json.loads(path.read_text())
    if issue=="replay":value["stage_results"][0]["comparisons"][0]["cache_vs_same_chunk_control"]["inverse_cdf"]["independent_fp32_replay_failures"]=1
    if issue=="normalizer":value["stage_results"][0]["cache_model_contract_after"]["input_normalization_train_only_verified"]=False
    if issue=="massbound":value["stage_results"][0]["component_qualification"]["members"][0]["component_raw_mass_fp64_roundoff_bound"]=-1
    if issue=="policyhash":value["fallback_policy_sha256"]="wrong"
    write_json(path,value)
    with pytest.raises(ValueError):
        assembly.assemble(cache_proofs=[cached],hybrid_proofs=hybrids,fallback_map=fallback,output=tmp_path/"out")


def test_stage_local_proofs_compose_into_explicit_global_policy(tmp_path,monkeypatch):
    cached,hybrids,fallback=fixture(tmp_path,monkeypatch)
    for path in hybrids:
        value=json.loads(path.read_text());local={str(value["stage_index"]):value["fallback_members"]}
        value["fallback_map"]=local;value["fallback_policy_sha256"]=assembly.fallback_policy_sha256(local)
        write_json(path,value)
    result=assembly.assemble(cache_proofs=[cached],hybrid_proofs=hybrids,fallback_map=fallback,output=tmp_path/"out")
    assert result["fallback_policy_sha256"]==assembly.fallback_policy_sha256(fallback)
    assert result["full21_hybrid_qualified"]
