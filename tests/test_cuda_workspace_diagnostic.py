"""CPU-only allocator mocks for bounded workspace attribution; no CUDA calls."""
import importlib.util
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import pytest


def helper():
    path=Path(__file__).parents[1]/"scripts"/"diagnose_cuda_workspace.py"
    spec=importlib.util.spec_from_file_location("workspace_probe_test",path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


class FakeTensor:
    def __init__(self,state,shape,checksum=0.):
        self.state=state;self.shape=shape;self.dtype="torch.float32";self.checksum=checksum
        self.bytes=4*shape[0]*shape[1];state["tensor_bytes"]+=self.bytes
    def __del__(self):self.state["tensor_bytes"]-=self.bytes
    def __getitem__(self,index):return SimpleNamespace(item=lambda:self.checksum)


def fake_torch(*,clear_available=True,empty_releases_workspace=False):
    state={"tensor_bytes":0,"workspace_bytes":0,"events":[]}
    def allocated():return state["tensor_bytes"]+state["workspace_bytes"]
    def empty():
        state["events"].append("empty_cache")
        if empty_releases_workspace:state["workspace_bytes"]=0
    def clear():state["events"].append("clear_workspaces");state["workspace_bytes"]=0
    cuda=SimpleNamespace(synchronize=lambda:None,memory_allocated=allocated,memory_reserved=allocated,
        memory_stats=lambda:{"active_bytes.all.current":allocated(),"allocated_bytes.all.current":allocated()},empty_cache=empty)
    def mm(left,right):
        state["events"].append("mm");state["workspace_bytes"]=8519680
        return FakeTensor(state,(left.shape[0],right.shape[1]),float(left.shape[1]))
    result=SimpleNamespace(cuda=cuda,float32="torch.float32",ones=lambda shape,**kwargs:FakeTensor(state,shape),mm=mm,
                           _C=SimpleNamespace(_cuda_clearCublasWorkspaces=clear) if clear_available else SimpleNamespace())
    return result,state


def test_helper_import_is_torch_and_gpu_safe():
    path=Path(__file__).parents[1]/"scripts"/"diagnose_cuda_workspace.py"
    code="import runpy,sys;runpy.run_path("+repr(str(path))+",run_name=\"import_only\");assert "+repr("torch")+" not in sys.modules"
    subprocess.run([sys.executable,"-c",code],check=True)


def test_live_inputs_are_deleted_and_empty_cache_precedes_explicit_clear():
    h=helper();torch,state=fake_torch();result=h.measure_cycle(torch,(2,3,4))
    assert result["workspace_mechanism_reproduced"]
    assert result["post_delete_residual_allocated_bytes"]==8519680
    assert result["empty_cache_allocated_delta_bytes"]==0 and result["workspace_clear_allocated_delta_bytes"]==8519680
    assert state["tensor_bytes"]==0 and state["workspace_bytes"]==0
    assert state["events"]==["mm","empty_cache","clear_workspaces"]
    by={row["label"]:row for row in result["snapshots"]}
    assert by["inputs_live"]["allocated_bytes"]==4*(2*3+3*4)
    assert by["matmul_output_live"]["allocated_bytes"]==4*(2*3+3*4+2*4)+8519680


def test_missing_private_api_is_truthfully_unattributed():
    torch,state=fake_torch(clear_available=False);result=helper().measure_cycle(torch,(2,3,4))
    assert result["clear_api_available"] is False and result["workspace_clear_allocated_delta_bytes"] is None
    assert not result["workspace_mechanism_reproduced"] and state["workspace_bytes"]==8519680


def test_empty_cache_releasing_residual_does_not_misattribute_it_to_clear():
    torch,_=fake_torch(empty_releases_workspace=True);result=helper().measure_cycle(torch,(2,3,4))
    assert result["empty_cache_allocated_delta_bytes"]==8519680
    assert result["workspace_clear_allocated_delta_bytes"]==0 and not result["workspace_mechanism_reproduced"]


def test_two_exact_repeats_bind_failed_host_bytes_but_never_upgrade_qualification():
    h=helper();torch,_=fake_torch();cycles=[h.measure_cycle(torch,shape) for shape in h.FIXED_SHAPES]
    result=h.analyze(cycles,8519680)
    assert result["residual_workspace_mechanism_reproduced"] and result["exactly_matches_preserved_host_residual"]
    assert result["preserved_host_proof_qualified"] is False and result["authorizes_automatic_host_pass"] is False
    assert h.analyze(cycles,1)["exactly_matches_preserved_host_residual"] is False
    cycles[1]["workspace_clear_allocated_delta_bytes"]+=1
    assert not h.analyze(cycles,8519680)["stable_exact_workspace_delta"]


def test_checksum_or_remaining_tensor_allocation_blocks_attribution():
    h=helper();torch,_=fake_torch();cycle=h.measure_cycle(torch,(2,3,4))
    rows=cycle["snapshots"]
    assert not h.summarize_cycle(rows,True,4.,3.)["workspace_mechanism_reproduced"]
    rows[-1]["allocated_bytes"]=4
    assert not h.summarize_cycle(rows,True,3.,3.)["workspace_mechanism_reproduced"]
