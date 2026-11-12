"""CPU-only custody, exact-baseline and completed-proof checks; no model calls."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import pytest
from test_cuda_workspace_diagnostic import fake_torch


def helper():
    path=Path(__file__).parents[1]/"scripts"/"qualify_execution_host_cache_warmed.py"
    spec=importlib.util.spec_from_file_location("warmed_host_test",path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


class Rng:
    def __init__(self,value):self.value=value
    def clone(self):return Rng(self.value)


def fake_warm_torch():
    torch,state=fake_torch()
    torch.get_rng_state=lambda:Rng(3);torch.cuda.get_rng_state_all=lambda:[Rng(4)]
    torch.equal=lambda a,b:a.value==b.value
    def forbidden():raise AssertionError("private workspace clear must never execute in warmed proof")
    torch._C._cuda_clearCublasWorkspaces=forbidden
    return torch,state


def completed_proof(h):
    return {"schema":"posteriorforge.execution_host_cache_qualification.v1","status":"host_screen_pass","qualified":True,
        "full21_host_residency_qualified":True,"full21_cache_only_qualified":False,"initial_global_cuda_allocated_bytes":h.WORKSPACE_BYTES,
        "all21_cpu_no_unexplained_cuda_allocations":True,"session_rng_preserved":True,"zero_refits_verified":True,
        "memory_contract_passed":True,"sources_unchanged":True,
        "sources":{"qualifier_source_sha256":h.HOST_QUALIFIER_SHA256,"host_module_sha256":h.HOST_MODULE_SHA256},
        "host_execution":{"counters":{"actual_prepare_miss_count":21,"cache_hit_count":42,"reload_count":42}},
        "cyclic_full_scans":{"evictions":0},"all21_cpu_residency":{"global_cuda_allocated_bytes":h.WORKSPACE_BYTES},
        "stage_results":[{"stage_index":stage,"passed":True} for stage in range(21)],"compatibility_adapter":{"original_alias_restored":True}}


def test_import_and_pinned_sources_are_cpu_safe():
    h=helper();h.load_frozen("diagnose_cuda_workspace.py",h.WORKSPACE_PROBE_SHA256)
    h.load_frozen("qualify_execution_host_cache.py",h.HOST_QUALIFIER_SHA256)
    path=Path(h.__file__)
    code="import runpy,sys;runpy.run_path("+repr(str(path))+",run_name=\"import_only\");assert "+repr("torch")+" not in sys.modules"
    subprocess.run([sys.executable,"-c",code],check=True)


def test_warmup_retains_exact_workspace_without_clear_or_rng_draw():
    h=helper();workspace=h.load_frozen("diagnose_cuda_workspace.py",h.WORKSPACE_PROBE_SHA256)
    torch,state=fake_warm_torch();result=h.warm_runtime(torch,workspace.snapshot)
    assert result["measured_live_workspace_bytes"]==8519680 and result["rng_preserved"]
    assert result["workspace_clear_called"] is False and state["events"]==["mm","empty_cache"]
    assert state["tensor_bytes"]==0 and state["workspace_bytes"]==8519680


def test_warmup_refuses_unexplained_prior_allocation_or_wrong_delta():
    h=helper();workspace=h.load_frozen("diagnose_cuda_workspace.py",h.WORKSPACE_PROBE_SHA256)
    torch,state=fake_warm_torch();state["workspace_bytes"]=1
    with pytest.raises(RuntimeError,match="zero allocated"):h.warm_runtime(torch,workspace.snapshot)
    torch,state=fake_warm_torch();original=torch.mm
    def wrong(left,right):
        output=original(left,right);state["workspace_bytes"]=h.WORKSPACE_BYTES+1;return output
    torch.mm=wrong
    with pytest.raises(RuntimeError,match="exactly authenticated"):h.warm_runtime(torch,workspace.snapshot)


@pytest.mark.parametrize("key,value",[("initial_global_cuda_allocated_bytes",8519681),("full21_host_residency_qualified",False),
    ("full21_cache_only_qualified",True),("all21_cpu_no_unexplained_cuda_allocations",False)])
def test_completed_proof_requires_exact_baseline_and_unchanged_full21_contract(key,value):
    h=helper();proof=completed_proof(h);h.validate_completed_proof(proof);proof[key]=value
    with pytest.raises(RuntimeError):h.validate_completed_proof(proof)


def test_completed_proof_rejects_partial_scan_extra_residual_or_unrestored_alias():
    h=helper()
    for key in ("hits","residual","stage","alias","eviction"):
        proof=completed_proof(h)
        if key=="hits":proof["host_execution"]["counters"]["cache_hit_count"]=21
        if key=="residual":proof["all21_cpu_residency"]["global_cuda_allocated_bytes"]+=1
        if key=="stage":proof["stage_results"].pop()
        if key=="alias":proof["compatibility_adapter"]["original_alias_restored"]=False
        if key=="eviction":proof["cyclic_full_scans"]["evictions"]=1
        with pytest.raises(RuntimeError):h.validate_completed_proof(proof)


def test_authentication_rejects_unapproved_probe_bytes(tmp_path):
    h=helper();probe=tmp_path/"probe.json";failed=tmp_path/"failed.json"
    probe.write_text("{}");failed.write_text("{}")
    with pytest.raises(ValueError,match="diagnostic SHA"):h.authenticate_workspace_evidence(probe,failed)
