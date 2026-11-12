"""CPU-only precision context/source contracts; never claim numerical GPU parity."""
import importlib.util
import inspect
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from posteriorforge import execution_precision_probe as precision


def utility():
    p=Path(__file__).parents[1]/"scripts/boundedprobe_precision.py"
    spec=importlib.util.spec_from_file_location("precision_probe_tests",p)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_dependency_light_metadata_has_distinct_approximate_names_and_sources():
    precision.require_sources()
    assert len(set(precision.MODES))==3
    for mode in precision.MODES:
        meta=precision.metadata(mode)
        assert meta["approximate"] and not meta["qualified_fp32"]
        assert meta["fallback_map"]=={"10":[5]}
    subprocess.run([sys.executable,"-c","import posteriorforge.execution_precision_probe as p,sys;assert 'torch' not in sys.modules;p.require_sources()"],check=True)


@pytest.mark.parametrize("mode",["bf16","fp16","tf32","auto","fp32"])
def test_ambiguous_or_unqualified_mode_names_rejected(mode):
    with pytest.raises(ValueError):precision.PrecisionSession(mode)


def test_private_clone_changes_only_declared_precision_assignment_and_dtype_predicate():
    function,descriptor=precision._resident_prepare_clone()
    assert descriptor["declared_changes"]=={"inference_precision_assignments":1}
    assert descriptor["cache_assumptions"]["declared_changes"]=={"prepared_dtype_predicates":1}
    assert function is not precision.cache.ResidentStageExecutor.prepare
    assert descriptor["original_ast_sha256"]!=descriptor["private_ast_sha256"]


def test_precision_cache_must_have_computed_prefill_witness_not_only_flags_or_storage():
    report={"mode":"amp_bf16_prefill_query_v1","witnesses":[]}
    assert not precision.prefill_witness_passed(report,6)["passed"]
    rows=[]
    for member in range(8):
        rows.append({"phase":"prefill","stage":6,"member":member,"cache_kv":True,"count":12,
                     "autocast_enabled":True,"autocast_dtype":"torch.bfloat16","projection_dtypes":{"k":None,"v":None,"kv":"torch.float32","qkv":None},
                     "supplied_cache_dtypes":[None,None,"torch.float32"]})
    report["witnesses"]=rows
    assert not precision.prefill_witness_passed(report,6)["passed"]
    for row in rows:row["projection_dtypes"]["kv"]="torch.bfloat16"
    assert precision.prefill_witness_passed(report,6)["passed"]


def test_unused_fallback_cached_member_must_also_be_computed_fp32():
    rows=[]
    for member in range(8):
        rows.append({"phase":"prefill","stage":10,"member":member,"cache_kv":True,"count":12,
                     "autocast_enabled":member!=5,"autocast_dtype":"torch.float16","allow_tf32":False,"matmul_precision":"highest","projection_dtypes":{"q":"torch.float32" if member==5 else "torch.float16",
                     "k":None,"v":None,"kv":"torch.float32" if member==5 else "torch.float16","qkv":None},
                     "supplied_cache_dtypes":[None,None,"torch.float32" if member==5 else "torch.float16"]})
    report={"mode":"amp_fp16_prefill_query_v1","witnesses":rows}
    assert precision.prefill_witness_passed(report,10)["passed"]
    rows[5]["autocast_enabled"]=True
    assert not precision.prefill_witness_passed(report,10)["passed"]


def test_same_unchanged_numerical_thresholds_and_bounded_stage_order():
    helper=utility()
    assert helper.STAGES==(6,10,13) and helper.ROWS==(128,1024)
    assert helper.THRESHOLDS=={"max_cdf_absolute":2e-5,"max_row_total_variation":1e-4,
                              "normalization_absolute":1e-5,"quantile_coordinate_span_fraction":2e-6}


def test_actual_pinned_sdk_context_and_prefill_aliases_restore_on_exception_without_cuda_initialization():
    torch=pytest.importorskip("torch")
    from tabpfn.model.multi_head_attention import MultiHeadAttention as cls
    from tabpfn.inference import InferenceEngineCacheKV
    before=(inspect.getattr_static(cls,"compute_attention_heads"),cls.compute_qkv,
            inspect.getattr_static(InferenceEngineCacheKV,"prepare"))
    initialized=torch.cuda.is_initialized()
    with pytest.raises(ArithmeticError):
        with precision.PrecisionSession(precision.CONTROL_MODE) as session:
            assert inspect.getattr_static(cls,"compute_attention_heads") is not before[0]
            assert cls.compute_qkv is not before[1]
            assert inspect.getattr_static(InferenceEngineCacheKV,"prepare") is not before[2]
            assert session.clones["prefill"]["declared_changes"]=={"prefill_autocast_contexts":1}
            with session.forward_scope("query",6,0):
                assert not torch.backends.cuda.matmul.allow_tf32
            raise ArithmeticError("synthetic")
    assert session.report()["aliases_and_flags_restored"]
    assert (inspect.getattr_static(cls,"compute_attention_heads"),cls.compute_qkv,
            inspect.getattr_static(InferenceEngineCacheKV,"prepare"))==before
    assert torch.cuda.is_initialized()==initialized


@pytest.mark.parametrize("mode",precision.MODES+(precision.CONTROL_MODE,))
def test_forward_scope_tf32_amp_default_and_fp32_fallback_restore_cpu_rng_and_flags(mode):
    torch=pytest.importorskip("torch")
    rng=torch.get_rng_state().clone()
    with precision.PrecisionSession(mode) as session:
        before=session._flags()
        with session.forward_scope("query",6,0):
            assert torch.backends.cuda.matmul.allow_tf32==(mode=="tf32_high_forward_v1")
            assert not torch.is_autocast_enabled("cuda")  # SDK must activate it.
            if mode=="amp_bf16_prefill_query_v1":assert torch.get_autocast_dtype("cuda")==torch.bfloat16
        assert session._flags()==before
        with session.forward_scope("prefill",10,5):
            assert torch.get_float32_matmul_precision()=="highest"
            assert not torch.backends.cuda.matmul.allow_tf32
        assert session._flags()==before
    assert torch.equal(rng,torch.get_rng_state())


def test_nested_context_is_rejected_and_original_ownership_survives():
    pytest.importorskip("torch")
    with precision.PrecisionSession(precision.CONTROL_MODE) as first:
        with pytest.raises(RuntimeError,match="another precision"):
            with precision.PrecisionSession(precision.CONTROL_MODE):pass
        assert precision._session() is first


def test_fp32_prediction_byte_replay_distinguishes_nans_payloads_and_values():
    torch=pytest.importorskip("torch");helper=utility()
    member={key:torch.tensor([[1.,float("nan")]]) for key in
            ("raw_logits","translated_probabilities","from_borders","to_borders")}
    left={"logits":torch.tensor([[2.,3.]]),"members":[member]*8}
    import copy
    right=copy.deepcopy(left)
    assert helper.exact_prediction(left,right,torch)
    right["logits"][0,0]+=1
    assert not helper.exact_prediction(left,right,torch)


def test_computed_low_precision_without_actual_buffer_evidence_is_rejected():
    rows=[{"phase":"prefill","stage":6,"member":member,"cache_kv":True,"count":12,
           "autocast_enabled":True,"projection_dtypes":{"q":None,"k":None,"v":None,"kv":"torch.bfloat16","qkv":None}}
          for member in range(8)]
    assert not precision.prefill_witness_passed({"mode":"amp_bf16_prefill_query_v1","witnesses":rows},6)["passed"]


def test_same_original_qkv_call_cpu_tensor_values_are_unmodified_by_observer():
    torch=pytest.importorskip("torch")
    from types import SimpleNamespace
    from tabpfn.model.multi_head_attention import MultiHeadAttention
    module=SimpleNamespace(_w_qkv=torch.arange(3*2*4*8,dtype=torch.float32).reshape(3,2,4,8),has_cached_kv=False)
    x=torch.arange(2*3*8,dtype=torch.float32).reshape(2,3,8)
    kwargs=dict(cache_kv=False,use_cached_kv=False,reuse_first_head_kv=False,use_second_set_of_queries=False)
    original=MultiHeadAttention.compute_qkv(module,x,None,None,None,None,**kwargs)
    with precision.PrecisionSession(precision.CONTROL_MODE) as session:
        with session.forward_scope("query",6,0):
            actual=MultiHeadAttention.compute_qkv(module,x,None,None,None,None,**kwargs)
        assert session.counters["qkv_calls"]==1
    for a,b in zip(original,actual):
        assert a is b is None or torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))


def test_conflicting_attention_alias_is_restored_and_reported_fail_closed():
    pytest.importorskip("torch")
    from tabpfn.model.multi_head_attention import MultiHeadAttention
    before=inspect.getattr_static(MultiHeadAttention,"compute_attention_heads")
    with pytest.raises(RuntimeError,match="restoration"):
        with precision.PrecisionSession(precision.CONTROL_MODE):
            MultiHeadAttention.compute_attention_heads=staticmethod(lambda *args,**kwargs:None)
    assert inspect.getattr_static(MultiHeadAttention,"compute_attention_heads") is before


def test_failed_enter_releases_exclusive_owner_and_preserves_sdk_aliases(monkeypatch):
    pytest.importorskip("torch")
    original=precision._resident_prepare_clone
    monkeypatch.setattr(precision,"_resident_prepare_clone",lambda:(_ for _ in ()).throw(ValueError("synthetic clone guard")))
    with pytest.raises(ValueError,match="synthetic clone guard"):
        with precision.PrecisionSession(precision.CONTROL_MODE):pass
    monkeypatch.setattr(precision,"_resident_prepare_clone",original)
    with precision.PrecisionSession(precision.CONTROL_MODE) as session:
        assert precision._session() is session


def test_tf32_policy_flag_must_be_observed_in_actual_prefill_events():
    rows=[{"phase":"prefill","stage":6,"member":member,"cache_kv":True,"count":12,
           "autocast_enabled":False,"allow_tf32":False,"matmul_precision":"highest",
           "projection_dtypes":{"q":"torch.float32","k":None,"v":None,"kv":"torch.float32","qkv":None},
           "supplied_cache_dtypes":[None,None,"torch.float32"]} for member in range(8)]
    report={"mode":"tf32_high_forward_v1","witnesses":rows}
    assert not precision.prefill_witness_passed(report,6)["passed"]
    for row in rows:row.update(allow_tf32=True,matmul_precision="high")
    assert precision.prefill_witness_passed(report,6)["passed"]


def test_low_query_path_requires_real_low_projection_and_active_inner_amp():
    rows=[{"phase":"query","stage":6,"member":member,"count":12,"autocast_enabled":True,
           "autocast_dtype":"torch.bfloat16","projection_dtypes":{"q":"torch.float32","kv":"torch.float32"}}
          for member in range(8)]
    report={"mode":"amp_bf16_prefill_query_v1","witnesses":rows}
    assert not precision.query_witness_passed(report,6)["passed"]
    for row in rows:row["projection_dtypes"]["q"]="torch.bfloat16"
    assert precision.query_witness_passed(report,6)["passed"]
    rows[0]["autocast_enabled"]=False
    assert not precision.query_witness_passed(report,6)["passed"]


@pytest.mark.parametrize("mode",precision.MODES)
def test_approximate_hybrid_counter_metadata_cannot_claim_entire_forward_is_fp32(mode):
    value=precision._hybrid_counter_metadata(mode)
    assert value["prediction_math_dtype"]!="torch.float32"
    assert value["forward_precision_policy"]==mode and value["postprocessing_math_dtype"]=="torch.float32"
    assert value["module_sha256"]==precision.source_sha256() and value["qualified_fp32"] is False
    assert value["fallback_map"]=={"10":[5]}
    frozen=value["frozen_hybrid_policy_provenance"]
    assert frozen["module_sha256"]==precision.PINNED["hybrid"] and frozen["prediction_math_dtype"]=="torch.float32"
