"""CPU-only immutable sampling, strict count and full50k fake-marker ABA tests."""
import hashlib
import inspect
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from posteriorforge import performance_host_session as runner
from posteriorforge.artifacts import sha256_file
from posteriorforge.science.analytical_benchmark_v2 import array_descriptor


def test_import_is_torch_safe_and_case_selection_is_explicit():
    subprocess.run([sys.executable,"-c","import posteriorforge.performance_host_session,sys;assert "+repr("torch")+" not in sys.modules"],check=True)
    assert runner.selected_cases()==tuple(range(8)) and runner.selected_cases("0,1")== (0,1)
    assert runner.selected_cases(isolation_aba=True)==(0,1,0)
    for cases in ("1,0","0,0","8",""):
        with pytest.raises(ValueError):runner.selected_cases(cases)
    with pytest.raises(ValueError):runner.selected_cases("0,1",isolation_aba=True)


def test_sampling_helpers_execute_authenticated_git_blob_not_working_tree(tmp_path):
    ops=runner.load_frozen_sampling(tmp_path/"sampling.py");ops.verify()
    assert sha256_file(tmp_path/"sampling.py")==runner.SAMPLING_SOURCE_SHA256
    assert "run_execution" not in ops.source["exported_helpers"]
    assert ops.stage_major_wave.__code__.co_filename==str((tmp_path/"sampling.py").resolve())
    seeds=ops.worker_stream_seeds(20261111,0,0,0,1)
    assert len(set(seeds.values()))==3


def test_original_sampler_and_support_active_code_are_guarded_cpu_only():
    torch=pytest.importorskip("torch");sampler,verify=runner.guarded_sampler(torch);verify()
    prior,support_verify=runner.guarded_prior(torch)
    from posteriorforge.science.analytical_benchmark_v2 import build_problem
    p=build_problem();values=torch.zeros((2,21),dtype=torch.float32)
    for name in p.prior.noise_names:values[:,p.parameter_order.index(name)]=sum(p.prior.bounds)/2
    assert prior.support.check(values).tolist()==[True,True];support_verify()
    assert callable(inspect.unwrap(sampler)) and not torch.cuda._initialized


def test_full50k_fake_marker_ABA_uses_fresh_fifo_generators_and_original_trim(tmp_path):
    torch=pytest.importorskip("torch");ops=runner.load_frozen_sampling(tmp_path/"sampling.py");sampler,verify=runner.guarded_sampler(torch)
    results=[];buffers=[];kept=[]
    for visit,case in enumerate((0,1,0)):
        seeds=ops.worker_stream_seeds(20261111,case,0,0,1);gens={name:torch.Generator().manual_seed(seed) for name,seed in seeds.items()};kept.append(gens)
        before={name:hashlib.sha256(g.get_state().numpy().tobytes()).hexdigest() for name,g in gens.items()}
        def provider(count,_case=case,_gens=gens):
            values=torch.zeros((count,21),dtype=torch.float32);values[:,0]=_case+torch.rand(count,generator=_gens["component_uniform"]);values[:,1]=torch.arange(count);return values,None
        scope={"case":case,"query":{"marker":case},"seed_streams":seeds}
        fifo=ops.guarded_proposal_buffer().ProposalBuffer(provider,scope=scope,concatenate=lambda parts:torch.cat(parts),world_size=1,rank=0);buffers.append(fifo)
        counts={"checked":0,"accepted":0,"calls":0}
        def support(values):
            mask=torch.ones(len(values),dtype=torch.bool)
            if counts["calls"]==0:mask[:10]=False
            counts["checked"]+=len(values);counts["accepted"]+=int(mask.sum());counts["calls"]+=1;return mask
        samples,_,_=sampler(proposal=fifo,accept_reject_fn=support,num_samples=50000,max_sampling_batch_size=50000,max_iter_rejection=None,show_progress_bars=False)
        record=fifo.finalize(original_checked_count=counts["checked"],accepted_before_trim=counts["accepted"],returned_count=len(samples),original_support_calls=counts["calls"])
        assert samples.shape==(50000,21) and counts["accepted"]==50090
        folder=tmp_path/f"visit-{visit}";folder.mkdir();values=samples.numpy().astype(np.float64);np.savez(folder/"samples.npz",samples=values)
        after={name:hashlib.sha256(g.get_state().numpy().tobytes()).hexdigest() for name,g in gens.items()}
        results.append({"case":case,"seed_streams":seeds,"rejection":counts,"proposal_wave_sizes":record["actual_generated_wave_sizes"],"sampling_counters":{"marker":1},
            "rng_before":before,"rng_after":after,"proposal_buffer":record,"samples":array_descriptor(values),"samples_sha256":sha256_file(folder/"samples.npz"),
            "relative_samples_file":str((folder/"samples.npz").relative_to(tmp_path)),"fifo_instance_identity":id(fifo),"generator_identities":{name:id(g) for name,g in gens.items()}})
    verify();result=runner.isolation_comparison(results,output=tmp_path)
    assert result["passed"] and all(result["predicates"].values())
    results[2]["fifo_instance_identity"]=results[0]["fifo_instance_identity"]
    assert not runner.isolation_comparison(results,output=tmp_path)["passed"]


def test_allocator_observer_preserves_environment_and_returns_detached_settings(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF","max_split_size_mb:512")
    settings={"max_split_size":123,"backend":"native"};calls=[]
    def snapshot():calls.append(1);return {"allocator_settings":settings}
    torch=SimpleNamespace(cuda=SimpleNamespace(memory=SimpleNamespace(_snapshot=snapshot)))
    observed=runner.allocator_observation(torch)
    assert observed["allocator_settings"]==settings and observed["allocator_settings"] is not settings
    assert observed["PYTORCH_CUDA_ALLOC_CONF"]=="max_split_size_mb:512" and calls==[1]
    assert not observed["allocator_policy_changed_by_observer"]
