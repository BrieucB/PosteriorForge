"""CPU-only allocator replay tests with tiny allocations, no CUDA initialization."""
import ast
from copy import deepcopy
import importlib.util
from pathlib import Path
import subprocess
import sys
import pytest


def helper():
    path=Path(__file__).parents[1]/"scripts"/"diagnose_cpu_allocator.py"
    spec=importlib.util.spec_from_file_location("cpu_allocator_test",path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def source_fixture():
    storages=[{"alias_group":"a","device":"cpu","bytes":2048},{"alias_group":"b","device":"cpu","bytes":1024}]
    return {"status":"failed","all21_cpu_residency":{"stages":[{"stage_index":20,"state":{"storages":storages,"cuda_storage_bytes":0,"cpu_storage_bytes":3072,"unique_storage_bytes":3072}}]}}


def test_source_counts_unique_storages_and_honors_exact_byte_cap():
    h=helper();assert h.storage_sizes(source_fixture(),3072)==[2048,1024]
    with pytest.raises(ValueError,match="cap exceeded"):h.storage_sizes(source_fixture(),3071)
    for kind in ("alias","gpu","total","missing"):
        source=source_fixture();state=source["all21_cpu_residency"]["stages"][0]["state"]
        if kind=="alias":state["storages"][1]["alias_group"]="a"
        if kind=="gpu":state["storages"][0]["device"]="cuda:0"
        if kind=="total":state["cpu_storage_bytes"]+=1
        if kind=="missing":source["all21_cpu_residency"]["stages"]=[]
        with pytest.raises(ValueError):h.storage_sizes(source)


def test_cpu_only_import_and_no_cuda_function_calls_in_source():
    h=helper();path=Path(h.__file__)
    code="import runpy,sys;runpy.run_path("+repr(str(path))+",run_name=\"import_only\");assert "+repr("torch")+" not in sys.modules"
    subprocess.run([sys.executable,"-c",code],check=True)
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
            assert not ast.unparse(node.func).startswith("torch.cuda.")


def memory_sequence(values):
    iterator=iter(values)
    return lambda label:{"label":label,"current_rss_bytes":next(iterator),"process_peak_rss_bytes":999999999,"cgroup":{"readable":False}}


def test_two_tiny_replays_preserve_live_sentinel_storage_views_and_rng():
    torch=pytest.importorskip("torch");h=helper();from posteriorforge import execution_host_cache as host
    sentinel=h.sentinel_state(torch);calls=[];cuda_before=bool(getattr(torch.cuda,"_initialized",False))
    for cycle in range(2):
        result=h.replay_cycle(torch,[2048,1024],sentinel,host,lambda pad:calls.append(pad) or 1,cycle,
                              memory=memory_sequence([1024,600000000,500000000,2048]))
        assert result["allocated_unique_storage_bytes"]==3072 and result["malloc_trim_return"]==1
        assert result["sentinel_full_storage_identity_layout_aliases"]["passed"] and result["rng_preserved"]
        assert result["rss_reclaimed_by_trim_bytes"]==499997952 and result["meaningful_free_page_reclaim"]
    assert calls==[0,0] and bool(getattr(torch.cuda,"_initialized",False))==cuda_before


def test_unsupported_trim_is_recorded_without_claiming_reclaim():
    torch=pytest.importorskip("torch");h=helper();from posteriorforge import execution_host_cache as host
    result=h.replay_cycle(torch,[128],h.sentinel_state(torch),host,None,0,memory=memory_sequence([1,100,50,50]))
    assert result["malloc_trim_return"] is None and not result["meaningful_free_page_reclaim"]


def test_sentinel_mutation_or_rng_draw_is_rejected():
    torch=pytest.importorskip("torch");h=helper();from posteriorforge import execution_host_cache as host
    sentinel=h.sentinel_state(torch)
    def bad_trim(pad):sentinel.base.add_(1);return 1
    with pytest.raises(RuntimeError,match="changed sentinel"):
        h.replay_cycle(torch,[128],sentinel,host,bad_trim,0,memory=memory_sequence([1,100,50,40]))
    def rng_trim(pad):torch.rand(1);return 1
    with pytest.raises(RuntimeError,match="or RNG"):
        h.replay_cycle(torch,[128],sentinel,host,rng_trim,0,memory=memory_sequence([1,100,50,40]))


def test_cgroup_namespace_mapping_labels_current_peak_and_limit(monkeypatch,tmp_path):
    h=helper();directory=tmp_path/"job_123";directory.mkdir()
    for name,value in (("memory.current","12"),("memory.peak","34"),("memory.max","100")):(directory/name).write_text(value)
    original=Path.read_text
    def read(path,*args,**kwargs):
        if str(path)=="/proc/self/cgroup":return "0::/slurm/job_123\n"
        if str(path)=="/proc/self/mountinfo":return "1 0 0:1 /slurm "+str(tmp_path)+" rw - cgroup2 cgroup rw\n"
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,"read_text",read)
    result=h.cgroup_memory();assert result["readable"]
    row=result["records"][0];assert (row["current_bytes"],row["peak_bytes"],row["limit_bytes"])==(12,34,100)
    assert row["membership"]=="/slurm/job_123" and row["directory"]==str(directory)
