"""CPU-only trim-proof pass contract and immutable source custody."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import hashlib
import pytest


def helper():
    path=Path(__file__).parents[1]/"scripts"/"qualify_execution_gqa_host_trim.py"
    spec=importlib.util.spec_from_file_location("trim_proof_test",path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def passing_report():
    result={key:True for key in ("sources_unchanged","all21_cpu_no_unexplained_cuda_allocations","zero_refits_verified","session_rng_preserved",
        "conditional_trim_exercised","all_trim_events_valid","allocator_source_and_evidence_unchanged","all_gqa_lifecycles_restored","memory_contract_passed")}
    result.update(cyclic_full_scans={"passed":True},initial_global_cuda_allocated_bytes=8519680,
        stage_results=[{"stage_index":i,"passed":True} for i in range(21)],
        migrations=[{"comparison":{"passed":True,"full_storage_bytes_equal":True}} for _ in range(105)],
        final_retention={"global_cuda_allocated_bytes":8519680,"retained_stages":21,"stages":[{"state":{"cuda_storage_bytes":0}} for _ in range(21)]})
    return result


def test_strict_final_contract_requires_actual_trim_and_all42_reloads():
    h=helper();report=passing_report();active={"original_alias_restored":True}
    assert all(h.final_predicates(report,active).values())
    for key in ("conditional_trim_exercised","all_trim_events_valid","zero_refits_verified","memory_contract_passed","session_rng_preserved"):
        bad=deepcopy(report);bad[key]=False;assert not all(h.final_predicates(bad,active).values())


def test_extra_cuda_or_missing_migration_cannot_pass():
    h=helper();report=passing_report();active={"original_alias_restored":True}
    report["final_retention"]["global_cuda_allocated_bytes"]+=1
    assert not h.final_predicates(report,active)["all21_final_cpu_no_extra_cuda"]
    report=passing_report();report["migrations"].pop()
    assert not h.final_predicates(report,active)["all105_migration_hashes_aliases_layout"]


def test_new_helper_pins_both_immutable_producer_spines():
    h=helper();h.frozen_helpers()
    for name,expected in (("qualify_execution_gqa_host_cache.py",h.BASE_GQA_HOST_QUALIFIER_SHA256),("qualify_execution_host_cache.py",h.BASE_HOST_QUALIFIER_SHA256)):
        assert hashlib.sha256(Path(h.__file__).with_name(name).read_bytes()).hexdigest()==expected
