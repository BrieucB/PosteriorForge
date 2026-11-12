"""CPU fake-law and fail-closed contracts; never execute a real model."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from posteriorforge import performance_runner as runner


class FakeEngine:
    def __init__(self):self.prepared=[];self.queries=[];self.closed=0
    def prepare(self,theta,obs,stage):
        self.stage=stage;self.offset=0;self.prepared.append(stage);return {}
    def predict(self,query,**kwargs):
        self.queries.append((self.stage,query.copy()))
        if self.stage==0:
            draw=np.arange(self.offset,self.offset+len(query),dtype=np.float32)+query[:,0]
            self.offset+=len(query)
        else:draw=query[:,-1]+1
        return {"draw":draw}
    def inspect_cache(self):return {"unique_storage_bytes":0,"tensor_records":[]}
    def close(self):self.closed+=1


def test_stage_major_wave_visits_all_targets_and_preserves_row_prefixes():
    engine=FakeEngine();events=[]
    values=runner.stage_major_wave(engine,np.zeros((4096,21)),np.zeros((4096,33)),np.zeros(33),7,3,lambda prediction:prediction["draw"],progress=events.append)
    np.testing.assert_array_equal(values,np.arange(7)[:,None]+np.arange(21)[None,:])
    assert engine.prepared==list(range(21)) and engine.closed==21
    assert len(events)==21 and all(event["model_query_chunks"]==3 for event in events)
    assert all(query.shape[1]==33+stage for stage,query in engine.queries)


def test_nan_prefix_is_passed_unchanged_and_no_partial_filtering_occurs():
    engine=FakeEngine();first=[True]
    def draw(prediction):
        value=prediction["draw"].copy()
        if first[0]:value[1]=np.nan;first[0]=False
        return value
    values=runner.stage_major_wave(engine,None,None,np.zeros(33),7,3,draw)
    assert values.shape==(7,21) and np.isnan(values[1,0])
    assert any(stage==1 and np.isnan(query[:,33]).any() for stage,query in engine.queries)


def test_incomplete_stage_fails_and_releases_cache():
    engine=FakeEngine()
    with pytest.raises(RuntimeError,match="incomplete stage"):
        runner.stage_major_wave(engine,None,None,np.zeros(33),7,3,lambda _:np.zeros(1))
    assert engine.closed==1


def test_case_a_b_a_queries_are_request_local_without_density_reuse():
    engine=FakeEngine();results=[]
    for value in (3.,8.,3.):
        query=np.full(33,value)
        results.append(runner.stage_major_wave(engine,None,None,query,3,2,lambda prediction:prediction["draw"]))
    np.testing.assert_array_equal(results[0],results[2])
    assert not np.array_equal(results[0],results[1]) and len(engine.prepared)==63


def test_stream_roles_are_distinct_and_rank_derived():
    streams=[runner.worker_stream_seeds(20261111,0,0,rank,4) for rank in range(4)]
    assert len({seed for stream in streams for seed in stream.values()})==12
    assert streams[0]==runner.worker_stream_seeds(20261111,0,0,0,1)
    with pytest.raises(ValueError):runner.worker_stream_seeds(20261111,0,0,4,4)
    with pytest.raises(ValueError):runner.worker_stream_seeds(20261111,0,0,0,True)


def worker_contract(output,world=4,rank=0):
    return {"schema":"posteriorforge.execution_worker_contract.v1","benchmark_total_count":50000,
        "world_size":world,"rank":rank,"worker_count":50000//world,"parent_job_id":"123",
        "context_seed":20261111,"case":0,"repetition":0,"runner_source_sha256":"runner",
        "expected_output":str(output.resolve()),"aggregate_output":"aggregate","orchestrator_source_sha256":"orchestrator",
        "visible_device_token":"GPU-test","seed_streams":runner.worker_stream_seeds(20261111,0,0,rank,world)}


@pytest.mark.parametrize("world,count",[(1,50000),(2,25000),(4,12500)])
def test_worker_subcounts_require_explicit_total50k_contract(tmp_path,monkeypatch,world,count):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","GPU-test")
    contract=worker_contract(tmp_path/"worker",world)
    value=runner.worker_partition(contract,output=tmp_path/"worker",job_id="123",context_seed=20261111,case=0,repetition=0,runner_sha256="runner")
    assert value["requested_count"]==count and value["end_to_end"] is False
    contract["worker_count"]+=1
    with pytest.raises(ValueError,match="partition"):
        runner.worker_partition(contract,output=tmp_path/"worker",job_id="123",context_seed=20261111,case=0,repetition=0,runner_sha256="runner")
    assert runner.worker_partition(None,output=tmp_path,job_id="123",context_seed=20261111,case=0,repetition=0,runner_sha256="runner")["requested_count"]==50000


def qualification():
    item={"passed":True,"inverse_cdf":{"icdf_contract_version":2}}
    return {"schema":"posteriorforge.execution_cache_qualification.v1","status":"screen_pass","qualified":True,
        "full21_qualified":True,"icdf_contract_version":2,"sources_unchanged":True,"control_variant":"cdf_arithmetic_v1",
        "engine_id":"resident_stage_kv_fp32_v1","module_sha256":"cache",
        "compatibility_adapter":{"original_alias_restored":True,"sdk_sources_unchanged":True,"sdk_aliases_unchanged":True,"patched_call_count":1,"executed_corrected_cdf_code_sha256":"corrected","corrected_cdf_code_sha256":"corrected","original_cdf_code_sha256":"original"},"sources":{"qualification_script_sha256":runner.QUALIFIER_SHA256},
        "thresholds":dict(runner.QUALIFICATION_THRESHOLDS),"row_counts":[1,16,128,1024],"stage_coverage":list(range(21)),
        "stage_results":[{"stage_index":stage,"passed":True,"stress_comparison":deepcopy(item),
            "comparisons":[{"query_rows":count,"cache_vs_same_chunk_control":deepcopy(item),"control_chunk_vs_1024":deepcopy(item)} for count in (1,16,128,1024)]} for stage in range(21)]}


def test_qualification_rejects_partial_coverage_changed_source_threshold_or_chunk():
    value=qualification()
    assert runner.validate_qualification(value,bindings={},cache_sha256="cache",mixture_sha256="mixture",mode="resident_full",query_batch_size=1024)
    for mutation in (lambda x:x.update(full21_qualified=False),lambda x:x["stage_coverage"].pop(),lambda x:x["thresholds"].update(max_cdf_absolute=.01),lambda x:x.update(module_sha256="different"),lambda x:x["stage_results"][0]["comparisons"].pop()):
        broken=deepcopy(value);mutation(broken)
        with pytest.raises(ValueError):runner.validate_qualification(broken,bindings={},cache_sha256="cache",mixture_sha256="mixture",mode="resident_full",query_batch_size=1024)
    with pytest.raises(ValueError,match="component"):
        runner.validate_qualification(value,bindings={},cache_sha256="cache",mixture_sha256="mixture",mode="component_mass",query_batch_size=1024)


def test_existing_output_is_preserved_before_any_budget_or_gpu_check(tmp_path):
    path=tmp_path/"manifest.json";path.write_text("untouched")
    with pytest.raises(FileExistsError):runner.run_execution(banks="none",output=tmp_path,checkpoint="none",ledger="none",reference="none",reference_grid="none",qualification="none")
    assert path.read_text()=="untouched"


def test_import_and_help_do_not_import_torch():
    code="import posteriorforge.performance_runner,sys;assert "+repr("torch")+" not in sys.modules"
    subprocess.run([sys.executable,"-c",code],check=True)
    value=subprocess.run([sys.executable,"-m","posteriorforge.performance_runner","--help"],capture_output=True,text=True,check=True)
    assert "--worker-contract" in value.stdout


def test_original_rejection_checks_complete_vectors_and_trims_only_valid_surplus():
    torch=pytest.importorskip("torch")
    from posteriorforge.inference import _prior
    path=Path(__file__).parents[1]/"src/posteriorforge/upstream/npe_pfn/accept_reject_sampler.py"
    spec=importlib.util.spec_from_file_location("original_rejection",path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    prior=_prior(torch);sizes=[]
    from posteriorforge.science.analytical_benchmark_v2 import build_problem
    indices=[build_problem().parameter_order.index(n) for n in build_problem().prior.noise_names]
    def proposal(size):
        sizes.append(size);values=torch.zeros(size,21);values[:,indices]=.25
        if len(sizes)==1:values[:2,indices[0]]=.6
        return values,None
    samples,_,_=module.accept_reject_sample(proposal,prior.support.check,10,max_sampling_batch_size=50000,max_iter_rejection=None)
    assert samples.shape==(10,21) and sizes==[10,100]
    assert prior.support.calls==2 and prior.support.proposed==110 and prior.support.accepted==108
    assert bool((samples[:,indices]==.25).all())


def test_component_uniforms_keep_physical_row_identity_across_chunks_and_head_retries():
    torch=pytest.importorskip("torch")
    from posteriorforge.execution_mixture import HEAD_MASS_ENVELOPE
    class Criterion:
        borders=torch.tensor([-2.,0.,3.])
    class Component:
        def predict_component(self,member,query,**kwargs):
            probability=.1+.08*member+.01*(query[:,0]%3)
            probabilities=np.column_stack((probability,1-probability)).astype(np.float32)
            return {"logits":torch.from_numpy(probabilities).log(),"criterion":Criterion(),
                    "component_raw_mass":torch.full((len(query),),.15+.1*member,dtype=torch.float64),"head_mass_envelope":HEAD_MASS_ENVELOPE}
    def run(chunk):
        generators={name:torch.Generator().manual_seed(seed) for name,seed in zip(runner.STREAM_ROLES,(13,29,41))}
        counters={"stage0_shared_predictions":0,"component_predictions":0,"head_candidates":0,"head_rejections":0}
        def query(rows):return np.column_stack((np.asarray(rows,dtype=np.float32),np.zeros((len(rows),33),np.float32)))
        values,metadata=runner.component_stage_sampler(Component(),torch,generators,counters)(None,1,query,17,chunk)
        return values,metadata,generators["component_uniform"].get_state()
    left,lmeta,lstate=run(2);right,rmeta,rstate=run(5)
    np.testing.assert_array_equal(left,right)
    assert lmeta["head_rejections"]>0 and rmeta["head_rejections"]==lmeta["head_rejections"]
    assert lmeta["uniform_draws"]==rmeta["uniform_draws"]==17 and torch.equal(lstate,rstate)


def hybrid_qualification():
    from posteriorforge.execution_hybrid import policy_metadata
    probability={"passed":True,"minimum_probability":0.,"max_normalization_error":0.,"max_cdf_absolute_error":0.,"max_row_total_variation":0.}
    prediction={"passed":True,"eight_members":True,"exact_final_borders":True,"final_float32":True,"final":deepcopy(probability),
        "members":[{**probability,"index":index,"configuration_identity_equal":True,"exact_borders":True,"float32":True} for index in range(8)],
        "inverse_cdf":{"passed":True,"icdf_contract_version":2,"same_logits_and_borders_exact_failures":0,"independent_fp32_replay_failures":0,"probability_space_failures":[]}}
    value={"schema":"posteriorforge.execution_hybrid_qualification.v1","status":"hybrid_screen_pass","qualified":True,
        "full21_qualified":False,"cache_only_full21_qualified":False,"full21_hybrid_qualified":True,"full21_component_qualified":True,"full21_sampler_qualified":True,
        "sources_unchanged":True,"input_statuses_preserved":True,"control_variant":"cdf_arithmetic_v1","icdf_contract_version":2,
        "cache_module_sha256":"cache","mixture_module_sha256":"mixture","hybrid_module_sha256":"hybrid",
        "cache_qualifier_sha256":runner.QUALIFIER_SHA256,"hybrid_qualifier_sha256":runner.HYBRID_QUALIFIER_SHA256,
        "assembler_source_sha256":runner.HYBRID_ASSEMBLER_SHA256,"fallback_map":deepcopy(runner.HYBRID_FALLBACK_MAP),
        "fallback_policy_sha256":policy_metadata(runner.HYBRID_FALLBACK_MAP)["fallback_policy_sha256"],
        "thresholds":deepcopy(runner.QUALIFICATION_THRESHOLDS),"row_counts":[1,16,128,1024],"stage_coverage":list(range(21)),"stage_results":[],"stage_routes":{}}
    for stage in range(21):
        fallback=runner.HYBRID_FALLBACK_MAP.get(str(stage),[]);routes={str(member):"uncached" if member in fallback else "cached" for member in range(8)}
        component={"passed":True,"module_sha256":"hybrid" if fallback else "mixture","fast_icdf_same_logits_failures":0,
            "executor_metadata":{"full8_member_lists_restored":True},"roundoff":{"mass_corrected_reconstruction_passed":True,"component_mass_guard_passed":True},
            "members":[{**probability,"ensemble_index":index,"configuration_identity_equal":True,"exact_borders":True,"float32":True,
                "component_raw_mass_matches_fp64_sum":True,"fast_icdf_same_logits_exact":True,"executed_corrected_helper_fingerprint_matches":True,
                "corrected_cdf_call_delta":1,"normalized_component":deepcopy(probability),"execution_route":routes[str(index)],"route_counter_delta":1,"other_route_counter_delta":0} for index in range(8)]}
        for name in ("normalized_mean_captured_raw_vs_full_fp32","normalized_mean_observed_raw_vs_full_fp32","mass_corrected_observed_outputs_vs_full_fp32"):component[name]=deepcopy(probability)
        if stage==0:component.update(stage0_dedup_scope="full_mixture_only",stage0_shared_distribution={**probability,"shared_icdf_same_logits_exact":True})
        value["stage_routes"][str(stage)]={"fallback_members":list(fallback),"member_routes":routes}
        value["stage_results"].append({"stage_index":stage,"passed":True,"assembled_route":"hybrid" if fallback else "cached",
            "comparisons":[{"query_rows":count,"cache_vs_same_chunk_control":deepcopy(prediction),"control_chunk_vs_1024":deepcopy(prediction)} for count in (1,16,128,1024)],
            "stress_comparison":deepcopy(prediction),"component_qualification":component})
    return value


def test_hybrid_proof_is_accepted_without_promoting_cache_flags_or_mutating_inputs():
    value=hybrid_qualification();before=deepcopy(value)
    assert runner.validate_qualification(value,bindings={},cache_sha256="cache",mixture_sha256="mixture",hybrid_sha256="hybrid",mode="component_hybrid",query_batch_size=1024)
    assert value==before and value["full21_qualified"] is False
    for mutation in (lambda x:x.update(full21_qualified=True),lambda x:x["fallback_map"]["13"].append(5),
                     lambda x:x["stage_routes"]["10"]["member_routes"].update({"5":"cached"}),
                     lambda x:x["stage_results"][13]["component_qualification"]["members"][6].update(route_counter_delta=0),
                     lambda x:x["stage_results"][2]["comparisons"][0]["cache_vs_same_chunk_control"]["final"].update(max_cdf_absolute_error=3e-5)):
        broken=deepcopy(value);mutation(broken)
        with pytest.raises(ValueError):runner.validate_qualification(broken,bindings={},cache_sha256="cache",mixture_sha256="mixture",hybrid_sha256="hybrid",mode="component_hybrid",query_batch_size=1024)


def gqa_descriptor(expanded=True):
    from posteriorforge.execution_gqa_hybrid import PINNED_GQA_SHA256
    return {**{k:True for k in ("original_alias_restored","original_function_restored","original_code_unchanged","active_code_unchanged",
        "gqa_source_unchanged","sdk_sources_unchanged","source_guard_passed")},"module_sha256":PINNED_GQA_SHA256,
        "original_attention_code_sha256":"original","executed_original_attention_code_sha256":"original","active_attention_code_sha256":"wrapper",
        "attention_call_count":24,"expanded_attention_calls":12 if expanded else 0,"prediction_math_dtype":"torch.float32",
        **{k:False for k in ("prefill_projection_changed","dtype_changed","dropout_changed","softmax_scale_changed","sdk_files_modified")},"view_examples":[]}


def test_gqa_raw_failed_proof_and_unarchived_recovery_cannot_run():
    from posteriorforge import execution_cache,execution_mixture,execution_hybrid
    kwargs=dict(bindings={},cache_sha256=execution_cache.execution_source_sha256(),mixture_sha256=execution_mixture.execution_source_sha256(),
                hybrid_sha256=execution_hybrid.execution_source_sha256(),mode="component_gqa_hybrid",query_batch_size=1024)
    with pytest.raises(ValueError,match="GQA qualification mismatch"):
        runner.validate_qualification({"schema":"posteriorforge.execution_gqa_hybrid_qualification.v1","status":"conditional_mismatch"},**kwargs)
    with pytest.raises(ValueError,match="source-bound recovered proof file"):
        runner.validate_qualification({"schema":"posteriorforge.execution_gqa_hybrid_recovery.v1"},**kwargs)


def test_actual_gqa_callback_requires_route_alias_fingerprint_and_corrected_cdf():
    cdf={"executed_corrected_cdf_code_sha256":"fixed","corrected_cdf_code_sha256":"fixed","original_cdf_code_sha256":"old"};report={}
    prediction={"gqa_attention_descriptor":gqa_descriptor(),"hybrid_execution_path":"resident_cache"}
    runner.validate_gqa_prediction(prediction,13,6,cdf_descriptor=cdf,report=report)
    fallback={"gqa_attention_descriptor":gqa_descriptor(False),"hybrid_execution_path":"original_fit_preprocessors"}
    runner.validate_gqa_prediction(fallback,10,5,cdf_descriptor=cdf,report=report)
    assert report["verified_calls"]==2 and report["uncached_fallback_calls"]==1 and report["all_aliases_restored"]
    wrong=deepcopy(prediction);wrong["gqa_attention_descriptor"]["executed_original_attention_code_sha256"]="stale"
    with pytest.raises(ValueError,match="fingerprint"):runner.validate_gqa_prediction(wrong,13,6,cdf_descriptor=cdf,report={})
    with pytest.raises((ValueError,RuntimeError)):runner.validate_gqa_prediction(fallback,13,6,cdf_descriptor=cdf,report={})
    cdf["executed_corrected_cdf_code_sha256"]="old"
    with pytest.raises(RuntimeError,match="corrected CDF"):runner.validate_gqa_prediction(prediction,13,6,cdf_descriptor=cdf,report={})


def test_optional_execution_audit_does_not_change_sampler_values_or_rng():
    torch=pytest.importorskip("torch")
    from posteriorforge.execution_mixture import HEAD_MASS_ENVELOPE
    class Criterion:borders=torch.tensor([-2.,0.,3.])
    class Component:
        def predict_component(self,member,query,**kwargs):
            p=torch.tensor([[.25+.04*member,.75-.04*member]],dtype=torch.float32).repeat(len(query),1)
            return {"logits":p.log(),"criterion":Criterion(),"component_raw_mass":torch.full((len(query),),.3+.08*member,dtype=torch.float64),"head_mass_envelope":HEAD_MASS_ENVELOPE}
    def execute(audit):
        generators={name:torch.Generator().manual_seed(seed) for name,seed in zip(runner.STREAM_ROLES,(17,29,53))}
        counters={"stage0_shared_predictions":0,"component_predictions":0,"head_candidates":0,"head_rejections":0};calls=[]
        callback=(lambda prediction,stage,member:calls.append((stage,member))) if audit else None
        draw=runner.component_stage_sampler(Component(),torch,generators,counters,prediction_check=callback)
        values,metadata=draw(None,13,lambda rows:np.zeros((len(rows),46),np.float32),37,8)
        return values,metadata,{name:g.get_state() for name,g in generators.items()},calls
    a,am,ar,_=execute(False);b,bm,br,calls=execute(True)
    np.testing.assert_array_equal(a,b);assert am==bm and all(torch.equal(ar[k],br[k]) for k in ar) and calls


def large_batch_fixture(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from posteriorforge import execution_gqa_hybrid as execution,execution_cache as cache,execution_mixture as mixture,execution_hybrid as hybrid
    from posteriorforge.science.analytical_benchmark_v2 import array_descriptor
    from posteriorforge.compat_cdf import adapter_source_sha256
    helper=runner.gqa_batch_helper();calls=[]
    monkeypatch.setattr(runner,"gqa_batch_helper",lambda:SimpleNamespace(full21_batch_outcomes=helper.full21_batch_outcomes,repeated_pool_indices=helper.repeated_pool_indices,
        shape_evidence=helper.shape_evidence,verify_portable_sources=lambda *a,**k:calls.append("portable_source_bytes")))
    monkeypatch.setattr(runner.gqa_recovery_helper(),"verify_recovery_artifact",lambda *a:(_ for _ in ()).throw(AssertionError("historical ctime replay forbidden for fresh large-batch proof")))
    binding={key:{"value":key} for key in ("checkpoint_sha256","scientific_fingerprint","parameter_order","context_theta","context_obs","query","prefix_manifest_sha256","prefix_samples_sha256")}
    base={**binding,"schema":"posteriorforge.execution_gqa_hybrid_recovery.v1","status":"gqa_hybrid_screen_pass","recovery_source_sha256":runner.GQA_RECOVERY_SHA256,
        "raw_status_preserved":True,"raw_producer_status":"conditional_mismatch","full21_gqa_hybrid_qualified":True,
        "execution_engine":execution.execution_metadata(),"fallback_map":{"10":[5]},"module_sha256":execution.execution_source_sha256(),
        "gqa_module_sha256":execution.PINNED_GQA_SHA256,"hybrid_module_sha256":hybrid.execution_source_sha256(),"cache_module_sha256":cache.execution_source_sha256(),
        "mixture_module_sha256":mixture.execution_source_sha256(),"raw_qualifier_source_sha256":runner.GQA_QUALIFIER_SHA256,"full21_qualified":False,"cache_only_full21_qualified":False}
    base_path=tmp_path/"base.json";base_path.write_text(json.dumps(base))
    metadata=execution.execution_metadata();v={**binding,"schema":"posteriorforge.execution_gqa_batch_qualification.v1","status":"batch_screen_pass","qualified":True,
        "qualifier_source_sha256":runner.GQA_BATCH_QUALIFIER_SHA256,"full21_qualified":False,"cache_only_full21_qualified":False,"posterior_vectors_generated":0,
        "scientific_gate_evaluated":False,"sources_unchanged":True,"requested_batch_sizes":list(runner.GQA_LARGE_BATCHES),"candidate_query_batch_cap":10000,"comparison_block_cap":1024,
        "thresholds":dict(runner.QUALIFICATION_THRESHOLDS),"module_sha256":execution.execution_source_sha256(),"gqa_module_sha256":execution.PINNED_GQA_SHA256,
        "hybrid_module_sha256":hybrid.execution_source_sha256(),"cache_module_sha256":cache.execution_source_sha256(),"mixture_module_sha256":mixture.execution_source_sha256(),
        "execution_engine":metadata,"fallback_map":{"10":[5]},"fallback_policy_sha256":metadata["fallback_policy_sha256"],"base_full21_gqa_hybrid_qualified":True,
        "base_qualification_schema":"posteriorforge.execution_gqa_hybrid_recovery.v1","recovered_producer_sha256":runner.GQA_RECOVERY_SHA256,
        "reference_pool":{"main_rows":1024,"stress_rows":8,"pool_rows":1032,"main_member_count":8,"stress_member_count":8,"original_calls_per_stage":2},
        "stage_coverage":list(range(21)),"stage_results":[],"qualified_batch_sizes":list(runner.GQA_LARGE_BATCHES),"full21_batch_qualified":{str(n):True for n in runner.GQA_LARGE_BATCHES},
        "portable_source_authentication":{"archives":{"base_qualification":{"file":"base.json","sha256":runner.sha256_file(base_path)}},"verification_uses_file_ctime":False,"verification_requires_original_machine_paths":False,"sdk_distribution":"tabpfn==2.0.9","all_wheel_sdk_python_sources_checked":True},
        "base_qualification_manifest_sha256":runner.sha256_file(base_path),
        "compatibility_adapter":{"adapter_sha256":adapter_source_sha256(),**{k:True for k in ("original_alias_restored","sdk_sources_unchanged","sdk_aliases_unchanged","source_guard_passed","original_alias_identity_verified","translator_code_unchanged")},
            "patched_call_count":840,"executed_corrected_cdf_code_sha256":"fixed","corrected_cdf_code_sha256":"fixed","original_cdf_code_sha256":"old"},
        "source_graph_at_end":{"wrapper":execution.execution_source_sha256(),"gqa":execution.PINNED_GQA_SHA256,"hybrid":hybrid.execution_source_sha256(),"cache":cache.execution_source_sha256(),"mixture":mixture.execution_source_sha256(),"recovery":runner.GQA_RECOVERY_SHA256}}
    def metric(n):return {"passed":True,"minimum_probability":0.,"max_normalization_error":0.,"max_cdf_absolute_error":0.,"max_row_total_variation":0.,"rows_compared":n,"comparison_block_cap":1024,"blocks":(n+1023)//1024,"worst_cdf_row":0}
    contract={"input_normalization_train_only_verified":True,"autocast":False,"forced_inference_dtype":"torch.float32","preprocessor_sha256":[str(i) for i in range(8)]}
    for stage in range(21):
        row={"stage_index":stage,"reference_main_rows":1024,"reference_stress_rows":8,"reference_pool_query":{"shape":[1032,33+stage]},
            "prepare_rng_unchanged":True,"normalization_train_only_verified":True,"all8_preprocessor_identity":True,"final_prepared_state_unchanged":True,
            "model_contracts":[deepcopy(contract) for _ in range(3 if stage==10 else 2)],"batch_results":[]}
        for n in runner.GQA_LARGE_BATCHES:
            batch={"query_rows":n,"status":"batch_pass","passed":True,"actual_input_query_shape":[n,33+stage],"explicit_query_batch_cap":10000,
                "reference_row_indices":array_descriptor(helper.repeated_pool_indices(n)),"rng_unchanged":True,"prepared_state_unchanged":True,
                "normalized_mixture_comparison_is_direct_not_error_bound":True,"ideal_raw_mixture_vs_original":metric(n),"actual_mass_weighted_component_mixture_vs_original":metric(n),"members":[]}
            for i in range(8):
                fallback=stage==10 and i==5;d=gqa_descriptor(not fallback);d["expanded_tensor_views"]=0 if fallback else 12
                d["head_shape_histogram"]={str({"q":[2,n,6,32],"qkv":None}):12};d["effective_head_shape_histogram"]=deepcopy(d["head_shape_histogram"])
                identity={"type":"Config","sha256":str(i),"repr":str(i)}
                batch["members"].append({"ensemble_index":i,"direct_raw_probability_comparison":metric(n),
                    **{k:True for k in ("passed","configuration_identity_equal","exact_borders","float32","component_raw_mass_matches_fp64_sum","normalized_component_passed")},
                    "reference_configuration_identity":identity,"candidate_configuration_identity":identity,
                    "actual_execution_route":"original_fit_preprocessors" if fallback else "resident_cache","corrected_cdf_call_delta":1,"rng_before":{"cpu":"same"},"rng_after":{"cpu":"same"},
                    "actual_gqa_descriptor":d,"actual_neural_shape":helper.shape_evidence(d,n),"component_raw_mass_sum_error":0.,"component_raw_mass_fp64_roundoff_bound":float(8*5000*np.finfo(np.float64).eps),
                    "mass_min":1.,"mass_max":1.,"head_mass_envelope":1.00001,
                    "stock_fixed_uniform_replay":{"passed":True,"exact_same_logits_finite_bar_semantics":True,"row_indices":np.linspace(0,n-1,8,dtype=np.int64).tolist()}})
            row["batch_results"].append(batch)
        v["stage_results"].append(row)
    kwargs=dict(bindings=binding,cache_sha256=cache.execution_source_sha256(),mixture_sha256=mixture.execution_source_sha256(),hybrid_sha256=hybrid.execution_source_sha256(),
        mode="component_gqa_hybrid",query_batch_size=4096,qualification_path=tmp_path/"manifest.json",sdk_root=tmp_path/"relocated-sdk")
    return v,kwargs,calls


def test_v3_large_batch_accepts_portable_fresh_proof_without_historical_ctime(tmp_path,monkeypatch):
    value,kwargs,calls=large_batch_fixture(tmp_path,monkeypatch)
    assert runner.validate_qualification(value,**kwargs) and calls==["portable_source_bytes"]
    assert runner.execution_query_cap("component_gqa_hybrid",4096)==10000 and runner.execution_query_cap("component_gqa_hybrid",1024)==1024
    assert runner.execution_query_cap("component_mass",1024)==1024


@pytest.mark.parametrize("mutation",[
    lambda v:v["stage_results"][3]["batch_results"][0]["members"][5]["direct_raw_probability_comparison"].update(max_cdf_absolute_error=2.1e-5),
    lambda v:v["stage_results"][7]["batch_results"][1]["members"][2]["actual_neural_shape"].update(requested_neural_query_rows=1024),
    lambda v:v["stage_results"][10]["batch_results"][0]["members"][5].update(actual_execution_route="resident_cache"),
    lambda v:v["stage_results"][1]["batch_results"][0]["members"][0].update(corrected_cdf_call_delta=0),
    lambda v:v["stage_results"][20].update(final_prepared_state_unchanged=False)])
def test_v3_large_batch_rejects_threshold_shape_route_source_or_state_tampering(tmp_path,monkeypatch,mutation):
    value,kwargs,_=large_batch_fixture(tmp_path,monkeypatch);mutation(value)
    with pytest.raises(ValueError):runner.validate_qualification(value,**kwargs)


def test_v3_never_selects_an_oom_or_failed_neural_size(tmp_path,monkeypatch):
    value,kwargs,_=large_batch_fixture(tmp_path,monkeypatch)
    value["stage_results"][5]["batch_results"][2].update(status="cuda_out_of_memory",passed=False)
    value["full21_batch_qualified"]["10000"]=False;value["qualified_batch_sizes"]=[4096,6250]
    assert runner.validate_qualification(value,**kwargs)
    kwargs["query_batch_size"]=10000
    with pytest.raises(ValueError,match="not full21 qualified"):runner.validate_qualification(value,**kwargs)


def test_v3_query_chunk_domain_keeps_other_modes_at1024_and_rejects_float_or_boolean_caps():
    for n in (1024,4096,6250,10000):assert runner.valid_execution_chunk("component_gqa_hybrid",n)
    for n in (True,4096.,2048,10001):assert not runner.valid_execution_chunk("component_gqa_hybrid",n)
    for mode in ("resident_full","component_mass","component_hybrid"):
        assert runner.valid_execution_chunk(mode,1024) and not runner.valid_execution_chunk(mode,4096)
