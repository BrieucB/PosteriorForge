"""CPU-only partition, authentication, subprocess and failure contracts."""
import importlib.util
import ast
import inspect
import textwrap
import json
from pathlib import Path
import os
import sys
import time
import numpy as np
import pytest
from posteriorforge.artifacts import sha256_file,write_json
from posteriorforge.compat_cdf import adapter_source_sha256,PINNED_SDK_SHA256
from posteriorforge.science.analytical_benchmark_v2 import array_descriptor,build_problem

spec=importlib.util.spec_from_file_location("execution_workers",Path(__file__).parents[1]/"scripts/run_execution_workers.py")
workers=importlib.util.module_from_spec(spec);spec.loader.exec_module(workers)


@pytest.mark.parametrize("count,expected",[(1,[50000]),(2,[25000,25000]),(4,[12500]*4)])
def test_fixed_aggregate_partition(count,expected):
    assert workers.partition_counts(count)==expected
    assert sum(expected)==50000


@pytest.mark.parametrize("value",[0,3,8,True])
def test_invalid_worker_count_cannot_expand_or_truncate_request(value):
    with pytest.raises(ValueError):workers.partition_counts(value)


def test_rank_role_seeds_are_explicit_distinct_and_match_driver():
    from posteriorforge.performance_runner import worker_stream_seeds
    for world in (1,2,4):
        maps=[workers.worker_seed_streams(20261111,0,0,rank,world) for rank in range(world)]
        assert maps==[worker_stream_seeds(20261111,0,0,rank,world) for rank in range(world)]
        assert len({seed for mapping in maps for seed in mapping.values()})==3*world
    assert workers.worker_seed_streams(20261111,0,0,0,1)==workers.worker_seed_streams(20261111,0,0,0,4)


def test_device_partition_rejects_duplicates_missing_and_extra_gpus():
    assert workers.visible_device_tokens("GPU-a,GPU-b",2)==["GPU-a","GPU-b"]
    for value in ("0,0","0","0,1,2",""):
        with pytest.raises(ValueError):workers.visible_device_tokens(value,2)




def actual_driver_execution_engine_header(mode,worker_count,query_batch_size=1024):
    # Evaluate the actual pure metadata expression from the production header,
    # rather than a handwritten fixture that can drift from the driver.
    from posteriorforge import performance_runner,execution_cache,execution_mixture
    tree=ast.parse(textwrap.dedent(inspect.getsource(performance_runner.run_execution)))
    fields=[node.value for node in ast.walk(tree) if isinstance(node,ast.keyword) and node.arg=="execution_engine"]
    assert len(fields)==1
    expression=ast.Expression(fields[0]);ast.fix_missing_locations(expression)
    namespace={**vars(performance_runner),"cache":execution_cache,"mixture":execution_mixture,
               "partition":{"requested_count":worker_count},"query_batch_size":query_batch_size,"mode":mode}
    return eval(compile(expression,"<actual-driver-metadata>","eval"),namespace)


def _payloads(tmp_path,mode="component_mass",query_batch_size=1024,proposal_lookahead=0):
    contracts=[];paths=[]
    problem=build_problem();noise=[problem.parameter_order.index(name) for name in problem.prior.noise_names]
    for rank in range(2):
        root=tmp_path/f"worker-{rank}";root.mkdir()
        contract=workers.build_contract(rank=rank,workers=2,output=root,aggregate_output=tmp_path,
            budget={"job_id":"123","reservation_id":"reservation"},context_seed=20261111,case=0,repetition=0,
            runner_source_sha256="runner",visible_device_token=str(rank))
        contract.update(mode=mode,query_batch_size=query_batch_size,proposal_lookahead=proposal_lookahead)
        path=tmp_path/f"contract-{rank}.json";write_json(path,contract);contracts.append(contract);paths.append(path)
        samples=np.zeros((25000,21),dtype=np.float64);samples[:,noise]=.2;samples[:,0]=rank*.01
        with (root/"samples.npz").open("wb") as stream:np.savez(stream,samples=samples)
        descriptor={"adapter_sha256":adapter_source_sha256(),"variant":"cdf_arithmetic_v1",
            "scope":"finite_ensemble_cdf_parentheses_only","source_guard_passed":True,
            "original_alias_restored":True,"sdk_sources_unchanged":True,"patched_call_count":21,
            "sdk_aliases_unchanged":True,"original_alias_identity_verified":True,"translator_code_unchanged":True,
            "corrected_cdf_code_sha256":"corrected","original_cdf_code_sha256":"original",
            "executed_corrected_cdf_code_sha256":"corrected",
            "original_translator_code_sha256":"same","active_translator_code_sha256":"same",
            "sdk_source_sha256":dict(PINNED_SDK_SHA256),"sdk_source_sha256_after":dict(PINNED_SDK_SHA256)}
        manifest={"schema":"test.worker.v1","status":"complete","run_purpose":"performance_worker",
            "mode":mode,"variant":workers.import_module("posteriorforge.performance_runner").execution_variant(mode,proposal_lookahead),
            "rank":rank,"world_size":2,"benchmark_total_count":50000,"requested_count":25000,
            "worker_contract_sha256":sha256_file(path),"end_to_end":False,
            "completed_stage_indices":list(range(21)),"qualification_manifest_sha256":"qualification",
            "runner_source_sha256":"runner","sources_unchanged":True,"qualification_full21":True,
            "execution_engine":actual_driver_execution_engine_header(mode,25000,query_batch_size),
            "engine_query_batch_cap":workers.import_module("posteriorforge.performance_runner").execution_query_cap(mode,query_batch_size),
            **({"hybrid_execution":workers.expected_hybrid_metadata()} if mode=="component_hybrid" else {}),
            **({"gqa_hybrid_execution":workers.expected_gqa_hybrid_metadata(),"qualification_full21_gqa_hybrid":True,
                "qualification_cache_only_full21":False,
                **({"qualification_family":"large_batch_v1","qualification_large_batch_size":query_batch_size,
                    "qualification_large_batch_producer_sha256":workers.import_module("posteriorforge.performance_runner").GQA_BATCH_QUALIFIER_SHA256,
                    "qualification_portable_source_verified":True,"historical_install_ctime_replayed":False} if query_batch_size!=1024 else {}),
                "gqa_prediction_checks":{"verified_calls":21,"all_aliases_restored":True,"all_sources_unchanged":True},
                "gqa_execution_counters":{"contexts":21,"all_contexts_restored":True,"all_sources_unchanged":True}} if mode=="component_gqa_hybrid" else {}),
            "compatibility_adapter":descriptor,"sampling_policy":{"streams":{key:{"seed":value}
                for key,value in contract["seed_streams"].items()}},
            "samples":array_descriptor(samples),"samples_sha256":sha256_file(root/"samples.npz")}
        if proposal_lookahead:
            from posteriorforge import performance_runner as driver
            manifest.update(context_seed=20261111,case=0,repetition=0,query={"sha256":"query"},context_theta={"sha256":"theta"},
                context_obs={"sha256":"obs"},scientific_fingerprint="science",checkpoint_sha256="checkpoint",proposal_lookahead=proposal_lookahead,
                proposal_buffer_policy=driver.proposal_buffer_policy(proposal_lookahead))
            scope=driver.proposal_buffer_scope(manifest,context_seed=20261111,case=0,repetition=0,rank=rank,world_size=2,seeds=contract["seed_streams"])
            fifo=driver.guarded_proposal_buffer()
            proposal=fifo.ProposalBuffer(lambda n:(np.zeros((n,21),dtype=np.float32),None),scope=scope,
                concatenate=np.concatenate,world_size=2,rank=rank)
            proposal(25000);proposal(200)
            record=proposal.finalize(original_checked_count=25200,accepted_before_trim=25010,returned_count=25000,original_support_calls=2)
            record.update(enabled=True,source_unchanged=True,policy_sha256=manifest["proposal_buffer_policy"]["policy_sha256"])
            manifest.update(proposal_buffer=record,proposal_wave_sizes=[25600],expected_stage_prepare_count=21,generated_proposal_wave_count=1,
                execution_counters={"prepare_count":21},
                rejection={"support_calls":2,"proposed":25200,"accepted_before_final_trim":25010,"returned":25000})
        write_json(root/"manifest.json",manifest)
    return contracts,paths


def test_aggregate_authenticates_two_complete_joint_cohorts_without_resampling(tmp_path):
    contracts,paths=_payloads(tmp_path)
    combined,records=workers.validate_worker_payloads(contracts,paths,expected_bindings={},
             driver_schema="test.worker.v1",qualification_sha256="qualification")
    assert combined.shape==(50000,21)
    np.testing.assert_array_equal(combined[:25000,0],0)
    np.testing.assert_array_equal(combined[25000:,0],.01)
    assert [record["rank"] for record in records]==[0,1]


@pytest.mark.parametrize("field,value",[("requested_count",50000),("completed_stage_indices",list(range(20))),
                                        ("worker_contract_sha256","bad"),("status","failed")])
def test_incomplete_or_misbound_worker_cannot_publish_aggregate(tmp_path,field,value):
    contracts,paths=_payloads(tmp_path);path=Path(contracts[0]["expected_output"])/"manifest.json"
    manifest=json.loads(path.read_text());manifest[field]=value;write_json(path,manifest)
    with pytest.raises(ValueError):workers.validate_worker_payloads(contracts,paths,expected_bindings={},
         driver_schema="test.worker.v1",qualification_sha256="qualification")
    assert not (tmp_path/"samples.npz").exists()


def test_unsupported_noise_and_hash_tampering_fail_without_fill_rows(tmp_path):
    contracts,paths=_payloads(tmp_path)
    root=Path(contracts[1]["expected_output"])
    with (root/"samples.npz").open("wb") as stream:np.savez(stream,samples=np.zeros((24999,21)))
    with pytest.raises(ValueError):workers.validate_worker_payloads(contracts,paths,expected_bindings={},
         driver_schema="test.worker.v1",qualification_sha256="qualification")


def test_successful_spawn_uses_separate_processes_and_explicit_environments(tmp_path):
    outputs=[tmp_path/"rank0",tmp_path/"rank1"]
    commands=[[sys.executable,"-c","import os,pathlib;pathlib.Path(os.environ['RESULT']).write_text(os.environ['CUDA_VISIBLE_DEVICES'])"] for _ in outputs]
    environments=[]
    for rank,path in enumerate(outputs):
        env=os.environ.copy();env.update(RESULT=str(path),CUDA_VISIBLE_DEVICES=f"GPU-{rank}");environments.append(env)
    logs=[(tmp_path/f"out{rank}",tmp_path/f"err{rank}") for rank in range(2)]
    assert workers.execute_workers(commands,environments,logs,timeout_seconds=5)==[0,0]
    assert [path.read_text() for path in outputs]==["GPU-0","GPU-1"]


def test_failed_worker_terminates_other_process_group_and_keeps_logs(tmp_path):
    commands=[[sys.executable,"-c","import time;time.sleep(.2);raise SystemExit(3)"],
              [sys.executable,"-c","import time;print('started',flush=True);time.sleep(20)"]]
    logs=[(tmp_path/f"out{rank}",tmp_path/f"err{rank}") for rank in range(2)]
    start=time.monotonic()
    with pytest.raises(RuntimeError,match="worker failed"):
        workers.execute_workers(commands,[os.environ.copy(),os.environ.copy()],logs,timeout_seconds=5)
    assert time.monotonic()-start<4
    assert all(path.exists() for pair in logs for path in pair)
    assert not (tmp_path/"samples.npz").exists()


def test_existing_aggregate_output_is_never_overwritten(tmp_path,monkeypatch):
    from posteriorforge import reference_grid
    root=tmp_path/"existing";root.mkdir();sentinel=root/"manifest.json";sentinel.write_text("preserve")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","0")
    monkeypatch.setattr(reference_grid,"_require_budgeted_gpu",lambda value:{
        "gpus":1,"job_id":"123","reservation_id":"reservation"})
    with pytest.raises(FileExistsError):
        workers.run(banks="unused",output=root,checkpoint="unused",ledger="unused",
                    reference="unused",reference_grid="unused",qualification="unused")
    assert sentinel.read_text()=="preserve"


def test_actual_driver_header_uses_base_cache_identity_for_all_sampling_modes():
    from posteriorforge.execution_cache import execution_metadata
    for mode in ("resident_full","component_mass","component_hybrid","component_gqa_hybrid"):
        header=actual_driver_execution_engine_header(mode,12500)
        assert header["engine_id"]==execution_metadata()["engine_id"]
        assert header["module_sha256"]==execution_metadata()["module_sha256"]
        assert header["posterior_count"]==12500


def test_default_worker_threads_partition_allocation_without_touching_blas_settings():
    assert workers.worker_thread_plan(4,"64")== (16,64)
    assert workers.worker_thread_plan(2,"32")== (16,32)
    assert workers.worker_thread_plan(1,None)== (16,16)
    assert workers.worker_thread_plan(4,"64",8)== (8,64)
    with pytest.raises(ValueError):workers.worker_thread_plan(4,"64",32)


def test_hybrid_workers_bind_fixed_fallback_map_and_source(tmp_path):
    contracts,paths=_payloads(tmp_path,mode="component_hybrid")
    bindings={"mode":"component_hybrid","variant":"execution_hybrid_v1"}
    values,_=workers.validate_worker_payloads(contracts,paths,expected_bindings=bindings,
                  driver_schema="test.worker.v1",qualification_sha256="qualification")
    assert values.shape==(50000,21)
    target=Path(contracts[1]["expected_output"])/"manifest.json"
    manifest=json.loads(target.read_text());manifest["hybrid_execution"]["fallback_map"]["10"]=[4]
    write_json(target,manifest)
    with pytest.raises(ValueError,match="hybrid source/policy"):
        workers.validate_worker_payloads(contracts,paths,expected_bindings=bindings,
                  driver_schema="test.worker.v1",qualification_sha256="qualification")


def test_gqa_workers_require_new_source_policy_and_actual_restoration_counts(tmp_path):
    contracts,paths=_payloads(tmp_path,mode="component_gqa_hybrid")
    bindings={"mode":"component_gqa_hybrid","variant":"execution_gqa_hybrid_v1"}
    values,_=workers.validate_worker_payloads(contracts,paths,expected_bindings=bindings,driver_schema="test.worker.v1",qualification_sha256="qualification")
    assert values.shape==(50000,21)
    target=Path(contracts[1]["expected_output"])/"manifest.json";original=json.loads(target.read_text())
    for mutation in [lambda m:m["gqa_hybrid_execution"]["fallback_map"].update({"13":[6]}),
                     lambda m:m["gqa_prediction_checks"].update(verified_calls=20),
                     lambda m:m["gqa_execution_counters"].update(all_contexts_restored=False)]:
        broken=json.loads(json.dumps(original));mutation(broken);write_json(target,broken)
        with pytest.raises(ValueError,match="GQA"):
            workers.validate_worker_payloads(contracts,paths,expected_bindings=bindings,driver_schema="test.worker.v1",qualification_sha256="qualification")


def test_large_batch_workers_require_exact_portable_family_and_qualified_cap(tmp_path):
    contracts,paths=_payloads(tmp_path,mode="component_gqa_hybrid",query_batch_size=4096)
    bindings={"mode":"component_gqa_hybrid","variant":"execution_gqa_hybrid_v1"}
    values,_=workers.validate_worker_payloads(contracts,paths,expected_bindings=bindings,driver_schema="test.worker.v1",qualification_sha256="qualification")
    assert values.shape==(50000,21)
    path=Path(contracts[0]["expected_output"])/"manifest.json";manifest=json.loads(path.read_text());manifest["historical_install_ctime_replayed"]=True;write_json(path,manifest)
    with pytest.raises(ValueError,match="portable large-batch"):workers.validate_worker_payloads(contracts,paths,expected_bindings=bindings,driver_schema="test.worker.v1",qualification_sha256="qualification")


def test_fifo_workers_require_policy_conservation_scope_and_actual_stage_coverage(tmp_path):
    contracts,paths=_payloads(tmp_path,mode="component_gqa_hybrid",proposal_lookahead=1200)
    bindings={"mode":"component_gqa_hybrid","variant":"execution_gqa_hybrid_fifo_v1","proposal_lookahead":1200}
    values,records=workers.validate_worker_payloads(contracts,paths,expected_bindings=bindings,
        driver_schema="test.worker.v1",qualification_sha256="qualification")
    assert values.shape==(50000,21)
    assert sum(row["proposal_buffer"]["generated_count"] for row in records)==51200
    target=Path(contracts[0]["expected_output"])/"manifest.json";original=json.loads(target.read_text())
    for mutate in (lambda m:m.update(proposal_lookahead=0),
                   lambda m:m.update(variant="execution_gqa_hybrid_v1"),
                   lambda m:m["proposal_buffer"].update(remaining_unused_buffer_count=0),
                   lambda m:m["proposal_buffer"]["request_scope"].update(case=1),
                   lambda m:m["execution_counters"].update(prepare_count=42),
                   lambda m:m["rejection"].update(proposed=25000),
                   lambda m:m["proposal_buffer_policy"].update(global_lookahead=1201)):
        broken=json.loads(json.dumps(original));mutate(broken);write_json(target,broken)
        with pytest.raises(ValueError):workers.validate_worker_payloads(contracts,paths,expected_bindings=bindings,
            driver_schema="test.worker.v1",qualification_sha256="qualification")


def test_orchestrator_lookahead_cli_stays_opt_in_and_never_exposes_output_count():
    import subprocess
    result=subprocess.run([sys.executable,str(Path(workers.__file__)),"--help"],capture_output=True,text=True,check=True)
    assert "--proposal-lookahead {0,1200}" in result.stdout
    assert "--count" not in result.stdout and "--num-samples" not in result.stdout
