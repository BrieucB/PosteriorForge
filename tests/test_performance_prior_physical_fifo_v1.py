"""Source-only production boundary/footer guards; no model/GPU or real cohort."""
import hashlib,pickle,random
import numpy as np
import pytest
from types import SimpleNamespace
from posteriorforge import performance_prior_physical_fifo_v1 as p


def rng_state():
    return pickle.dumps((np.random.get_state(),random.getstate()),protocol=5)


def test_runtime_body_reverse_ast_has_only_declared_boundary_footer_changes():
    assert p.derivation_descriptor()["reverse_AST_exact"]
    assert p._RUN_BODY.__globals__["physical_fifo"] is p.physical_fifo
    assert p._RUN_BODY.__globals__["_safe_capture_failure_footer"] is p._safe_capture_failure_footer
    assert p._COMPLETE_REQUEST is p.physical_fifo.complete_request
    p._integrity(require_committed=False)


def test_authorized_entry_still_requires_committed_source_guard_before_runtime(monkeypatch):
    assert p.ACTIVATION_ENABLED is True
    calls=[]
    def uncommitted():raise RuntimeError("exact runtime source closure must be committed")
    monkeypatch.setattr(p,"require_sources",uncommitted)
    monkeypatch.setattr(p,"_RUN_BODY",lambda **kwargs:calls.append(kwargs))
    with pytest.raises(RuntimeError,match="must be committed"):
        p.run(banks="missing",checkpoint="missing",output="missing")
    assert calls==[]


def test_explicit_disabled_entry_cannot_enter_any_runtime(monkeypatch):
    monkeypatch.setattr(p,"ACTIVATION_ENABLED",False)
    with pytest.raises(RuntimeError,match="not activated"):
        p.run(banks="missing",checkpoint="missing",output="missing")


def test_new_private_fifo_namespace_does_not_modify_frozen_sampling():
    original=p._SAMPLER_FACTORY()
    old={name:getattr(original,name) for name in
         ("component_stage_sampler","worker_stream_seeds","proposal_buffer_policy",
          "guarded_proposal_buffer","validate_proposal_buffer_record")}
    namespace=p.physical_sampling_namespace()
    assert namespace.component_stage_sampler is original.component_stage_sampler
    assert namespace.worker_stream_seeds is original.worker_stream_seeds
    assert namespace.guarded_proposal_buffer is p.physical_fifo.guarded_proposal_buffer
    assert namespace.proposal_buffer_policy is p.physical_fifo.proposal_buffer_policy
    assert all(getattr(original,k) is v for k,v in old.items())


def test_source_closure_tampering_rejected_before_any_runtime(monkeypatch):
    monkeypatch.setattr(p,"_capture_failure_footer",lambda *args:None)
    with pytest.raises(RuntimeError,match="closure"):
        p._integrity(require_committed=False)


def test_private_run_global_rebinding_rejected(monkeypatch):
    monkeypatch.setitem(p._RUN_BODY.__globals__,"physical_fifo",SimpleNamespace())
    with pytest.raises(RuntimeError,match="global closure"):
        p._integrity(require_committed=False)


def test_footer_captures_actual_available_objects_without_success_claim():
    class FakeTensor:
        def __init__(self,value):self.value=np.asarray(value,np.uint8)
        def detach(self):return self
        def cpu(self):return self
        def numpy(self):return self.value
    class Generator:
        def get_state(self):return FakeTensor([1,3,5,7])
    engine=SimpleNamespace(metadata=lambda:{"prepare_count":21,"actual_head_contexts":1138})
    buffer=SimpleNamespace(metadata=lambda:{"failed":True,"original_api_delivered_count":0})
    state={"engine":engine,"buffer":buffer,"counters":{"component_predictions":1130},
           "descriptor":{"patched_call_count":1138,"original_alias_restored":True},
           "generators":{"component_uniform":Generator()},
           "prior":SimpleNamespace(support=SimpleNamespace(calls=0,proposed=0,accepted=0))}
    before=rng_state();status={}
    p._capture_failure_footer(status,state)
    assert rng_state()==before
    footer=status["observed_failure_footer"];observed=footer["observations"]
    assert not footer["normal_success_footer_claimed"] and not footer["successful_50000_request_claimed"]
    assert not footer["reconstruction_of_c810_or_prior_run"]
    assert observed["actual_engine_journal"]["value"]["prepare_count"]==21
    assert observed["actual_sampling_counters"]["value"]==state["counters"]
    assert observed["actual_support_counters"]["value"]=={"calls":0,"proposed":0,"accepted":0}
    assert "actual_default_RNG_post_exception" not in observed
    private=observed["actual_private_generator_states"]["value"]["component_uniform"]
    assert private["sha256"]==hashlib.sha256(bytes([1,3,5,7])).hexdigest()


def test_footer_observation_failures_do_not_mask_primary_error():
    def unavailable():raise OSError("missing final observation")
    state={"engine":SimpleNamespace(metadata=unavailable),
           "buffer":SimpleNamespace(metadata=unavailable)}
    primary=ValueError("actual original failure")
    status={}
    with pytest.raises(ValueError) as caught:
        try:raise primary
        except BaseException:
            p._capture_failure_footer(status,state)
            raise
    assert caught.value is primary
    observations=status["observed_failure_footer"]["observations"]
    assert observations["actual_engine_journal"]["status"]=="unavailable"
    assert observations["actual_FIFO_metadata"]["status"]=="unavailable"


def test_original_failed_producer_and_helper_exact_source_hashes():
    assert p.failed.source_sha256()==p.ORIGINAL_PRODUCER_SHA256
    assert p.physical_fifo.source_sha256()==p.PHYSICAL_FIFO_SHA256
    assert p.failed.source_sha256()=="71e5e2906fd663534e56878de1fc4aff55093e3d950a13f348f57417ca7c01d3"


def test_entire_footer_capture_failure_preserves_primary(monkeypatch):
    def unavailable(*args,**kwargs):raise OSError("observer failed before field-level handlers")
    monkeypatch.setattr(p,"_capture_failure_footer",unavailable)
    primary=KeyboardInterrupt("original interruption")
    status={};before=rng_state()
    with pytest.raises(KeyboardInterrupt) as caught:
        try:raise primary
        except BaseException:
            p._safe_capture_failure_footer(status,{})
            raise
    assert caught.value is primary and rng_state()==before
    assert status["failure_footer_capture_error"]=={
        "error_type":"OSError","error":"observer failed before field-level handlers",
        "secondary_to_original_exception":True,"success_footer_fabricated":False}


def test_source_read_and_secondary_storage_failure_cannot_mask_primary(monkeypatch):
    def unavailable():raise OSError("source unreadable")
    monkeypatch.setattr(p,"source_sha256",unavailable)
    class Unwritable(dict):
        def __setitem__(self,key,value):raise RuntimeError("secondary status write failed")
    primary=ValueError("original failure")
    with pytest.raises(ValueError) as caught:
        try:raise primary
        except BaseException:
            p._safe_capture_failure_footer(Unwritable(),{})
            raise
    assert caught.value is primary


def test_pinned_runtime_inventory_has_actual_transitive_ancestry_and_shared_helper():
    result=p.runtime_source_inventory(require_committed=False)
    assert result["current_checkout_exact"] and not result["committed_HEAD_exact"]
    assert result["source_count"]>=100
    assert result["sources"]["src/posteriorforge/execution_prior_native_v1.py"]==p.failed.BACKEND_SHA256
    assert result["sources"]["src/posteriorforge/execution_physical_fifo_v1.py"]==p.PHYSICAL_FIFO_SHA256
    assert result["sources"]["scripts/qualify_prior_native_conditionals_v1.py"]==p.failed.QUALIFIER_SHA256
    assert result["sources"]["src/posteriorforge/prior_coordinate_forensics_v1.py"]==p.failed.FORENSICS_SHA256


def test_runtime_inventory_edit_rejected(monkeypatch):
    monkeypatch.setitem(p.RUNTIME_SOURCE_SHA256,"src/posteriorforge/inference.py","0"*64)
    with pytest.raises(RuntimeError,match="inventory changed"):
        p.runtime_source_inventory(require_committed=False)
