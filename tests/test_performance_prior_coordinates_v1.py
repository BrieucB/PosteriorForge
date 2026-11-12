"""CPU checks of new coordinate orchestration; no model, checkpoint or CUDA."""
from contextlib import ExitStack
from types import SimpleNamespace
import numpy as np
import pytest
from posteriorforge import performance_prior_coordinates_v1 as p


def test_unpinned_backend_prevents_inference(monkeypatch):
    monkeypatch.setattr(p, "BACKEND_SHA256", None)
    with pytest.raises(RuntimeError, match="backend"):
        p.require_sources()


def test_model_bindings_transform_saved_f64_before_single_cast():
    physical = p.codec.decode_ordered(np.zeros((8, 21), dtype=np.float64))
    context = SimpleNamespace(theta=physical, obs=np.arange(8*33, dtype=np.float64).reshape(8,33)/100)
    raw = np.arange(33, dtype=np.float64).reshape(1,33)/100
    theta, obs, query, model, receipt = p.model_bindings(context, raw)
    expected = p.codec.order_latents(p.codec.encode_for_model(physical))
    assert theta.dtype == obs.dtype == query.dtype == np.float32
    assert theta.tobytes() == expected.tobytes()
    assert model == receipt["model_inputs"]
    assert receipt["physical_theta_original_Float64"]["dtype"] == "<f8"
    assert receipt["model_target_coordinate_system"] == "causal_prior_latent_sigma_first_stage_order"
    with pytest.raises(RuntimeError, match="original Float64"):
        p.model_bindings(SimpleNamespace(theta=physical.astype(np.float32), obs=context.obs), raw)


def test_decode_receives_exact_actual_array_without_cast(monkeypatch):
    latent = np.zeros((5,21), dtype=np.float32)
    original = p.codec.decode_ordered
    seen=[]
    def observe(value):
        seen.append(value)
        return original(value)
    monkeypatch.setattr(p.codec, "decode_ordered", observe)
    result = p.decode_complete_wave(latent)
    assert seen == [latent] and seen[0] is latent
    assert result.dtype == np.float64 and result.shape == latent.shape


def test_decode_checks_unused_tail_and_never_repairs():
    latent = np.zeros((51200,21), dtype=np.float32)
    latent[-1,3] = np.nan
    before = latent.tobytes()
    with pytest.raises(ValueError):
        p.decode_complete_wave(latent)
    assert latent.tobytes() == before
    latent[-1,3] = np.inf
    with pytest.raises(ValueError):
        p.decode_complete_wave(latent)
    with pytest.raises(RuntimeError, match="no implicit cast"):
        p.decode_complete_wave(np.zeros((2,21),dtype=np.int32))


def adapter_for_close(order, *, audit_error=False, release_error=False):
    report = {"aliases_and_flags_restored":True,"counters":{"witness_histogram_overflow":0}}
    class Session:
        def __exit__(self,*args): order.append("precision_exit")
        def report(self): return report
    class Engine:
        @property
        def child(self):return self
        @property
        def hybrid_counters(self):
            if audit_error: raise ValueError("counter audit failed")
            return {"resident_component_predict_count":0,"uncached_predict_count":0}
        def close(self):
            order.append("release")
            if release_error: raise OSError("release failed")
    a=p.PriorCoordinateStageAdapter("fake",cdf_descriptor={})
    a.session=Session();a.engine=Engine();a.entered=True
    a.stage_index=0;a.stage_heads=0;a.members=set();a.prefill=a.model=None
    return a


def test_failure_cleanup_releases_before_precision_and_cdf_exit():
    order=[];a=adapter_for_close(order,audit_error=True)
    with pytest.raises(ValueError,match="counter audit failed"):
        with ExitStack() as stack:
            stack.callback(lambda:order.append("CDF_exit"))
            stack.callback(a.close)
    assert order == ["release","precision_exit","CDF_exit"]
    assert a.engine is None and a.session is None and not a.entered
    assert a.stages[-1]["engine_release_succeeded"]
    assert "counter audit failed" in a.stages[-1]["accounting_audit_error"]


def test_combined_cleanup_errors_preserve_primary_and_release_cause():
    order=[];a=adapter_for_close(order,audit_error=True,release_error=True)
    with pytest.raises(ValueError,match="counter audit failed") as info:
        a.close()
    assert isinstance(info.value.__cause__,OSError)
    assert order == ["release","precision_exit"]
    assert a.stages[-1]["engine_release_succeeded"] is False
    assert "release failed" in a.stages[-1]["engine_release_error"]
    assert a.session is None and a.engine is None


def test_cleanup_idempotent_and_routes_not_fabricated():
    order=[];a=adapter_for_close(order)
    a.close();a.close()
    assert order == ["release","precision_exit"]
    assert len(a.stages)==1 and a.stages[0]["actual_head_contexts"]==0


def test_fresh_screen_none_fails_before_helper_or_data(monkeypatch):
    monkeypatch.setattr(p,"QUALIFIER_SHA256",None)
    with pytest.raises(RuntimeError,match="must be pinned"):
        p.qualification_receipt("missing",banks="missing",prefix_run="missing",failed_screen="missing",codec_inventory="missing",codec_reference_inventory="missing",
                                codec_combined_inventory="missing",completed_grid="missing",raw_bindings={},model_inputs={})


def test_invalid_unused_tail_is_preserved_before_decode_failure(tmp_path):
    latent=np.zeros((51200,21),np.float32);latent[-1,3]=np.nan
    waves=[];status={"decode_wave_records":[]}
    with pytest.raises(ValueError):
        p.record_decode_wave(latent,waves,status)
    assert len(waves)==status["actual_complete_generated_wave_count"]==1
    assert waves[0].tobytes()==latent.tobytes()
    assert status["failed_decode_wave_index"]==0
    assert status["decode_wave_records"][0]["strict_decode_passed"] is False
    archive=p.archive_latent_waves(tmp_path,waves)
    with np.load(tmp_path/archive["file"],allow_pickle=False) as saved:
        assert saved.files==["wave_0000"]
        assert saved["wave_0000"].tobytes()==latent.tobytes()
    assert archive["generated_row_count"]==51200
    assert not (tmp_path/"samples.npz").exists()


def test_native_execution_metadata_separates_coordinate_and_route_sources():
    descriptor=p.execution_metadata()
    assert descriptor["backend"]["module_sha256"]==p.BACKEND_SHA256
    assert descriptor["coordinate_policy"]["module_sha256"]==p.backend.CACHED_SHA256
    assert descriptor["native_execution_policy"]["native_stages"]==[6,10]
    assert descriptor["native_all8_stages"]==[6,10]
    assert descriptor["inner88_precision_fallback_map"]=={}
    assert descriptor["mode"]==p.backend.CONTROL_MODE
    assert not descriptor["native_execution_policy"]["FP16_activated"]


@pytest.mark.parametrize("stage",[0,6,10,20])
def test_prepare_uses_native_composite_and_honest_preparation_api(monkeypatch,stage):
    order=[];prepared={"stage_index":stage,"route":p.backend.route(stage)}
    class Session:
        def __enter__(self):order.append("session_enter");return self
        def report(self):return {"mode":p.MODE,"sample":"actual_report"}
        def __exit__(self,*args):order.append("session_exit")
    class Engine:
        def __init__(self,*args,**kwargs):self.regressor=object()
        def prepare(self,theta,obs,j):
            assert theta.dtype==obs.dtype==np.float32 and j==stage
            order.append("prepare");return prepared
        def close(self):order.append("release")
    def witness(report,j,result):
        assert report["sample"]=="actual_report" and j==stage and result is prepared
        return {"passed":True,"stage":j,"native_neural_prefill_calls":0,
                "cached_prefill_claimed":False}
    summary=SimpleNamespace(UTILITY_SHA="fake",model_summary=lambda *args:{"cold":"model"})
    monkeypatch.setattr(p.backend,"PriorNativeSession",Session)
    monkeypatch.setattr(p.backend,"PriorNativeStageExecutor",Engine)
    monkeypatch.setattr(p.backend,"preparation_witness",witness)
    monkeypatch.setattr(p.backend,"immutable_native_snapshot",lambda *args:{"immutable":"all8"})
    monkeypatch.setattr(p,"load_script",lambda *args:summary)
    theta=np.zeros((4096,21),np.float32);obs=np.ones((4096,15),np.float32)
    adapter=p.PriorCoordinateStageAdapter("fake",cdf_descriptor={})
    result=adapter.prepare(theta,obs,stage)
    assert result is prepared and adapter.prepare_count==1
    assert adapter.model_at_prepare=={"cold":"model"}
    assert adapter.prefill["native_neural_prefill_calls"]==0
    if stage in (6,10):assert adapter.immutable_native_before=={"immutable":"all8"}
    assert order==["session_enter","prepare"]


@pytest.mark.parametrize("stage",[3,6,10])
def test_actual_query_validates_declared_policy_and_real_SDK_route(monkeypatch,stage):
    torch=pytest.importorskip("torch")
    descriptor={"patched_call_count":0,"executed_corrected_cdf_code_sha256":"corrected",
                "corrected_cdf_code_sha256":"corrected"}
    class Session:
        def report(self):return {"mode":p.MODE,"actual_full_report":True}
    expected="original_fit_preprocessors" if stage in p.backend.NATIVE_STAGES else "resident_cache"
    wrong=[False];seen=[]
    class Engine:
        def predict_component(self,member,query,**kwargs):
            seen.append(query.copy());descriptor["patched_call_count"]+=1
            proof={"module_sha256":p.BACKEND_SHA256,"policy_sha256":p.backend.policy_metadata()["policy_sha256"],
                "stage":stage,"member":member,"route":p.backend.route(stage),
                "original_query_views_disabled":True,"query_view_setting_restored":True,"native_KV_cache_used":False}
            return {"logits":torch.zeros((len(query),2),dtype=torch.float32),
                "criterion":SimpleNamespace(borders=torch.tensor([-1.,0.,1.],dtype=torch.float32)),
                "component_raw_mass":torch.ones(len(query),dtype=torch.float64),"head_mass_envelope":1.00001,
                "hybrid_execution_path":"wrong" if wrong[0] else expected,"prior_native_execution":proof}
    def witness(before,after,**kwargs):
        assert before["actual_full_report"] and after["actual_full_report"]
        assert kwargs=={"stage":stage,"member":2}
        return {"passed":True,"stage":stage,"member":2}
    monkeypatch.setattr(p.backend,"query_witness_delta",witness)
    adapter=p.PriorCoordinateStageAdapter("fake",cdf_descriptor=descriptor)
    adapter.engine=Engine();adapter.session=Session();adapter.stage_index=stage
    adapter.stage_heads=0;adapter.members=set()
    query=np.ones((4,15+stage),np.float32)
    result=adapter.predict_component(2,query)
    assert result["prior_coordinate_execution_proof"]["mode"]==p.MODE
    assert seen[0].tobytes()==p.backend.mask_query(query,stage).tobytes()
    assert adapter.query_calls[0]["execution_route"]==p.backend.route(stage)
    assert adapter.query_calls[0]["actual_SDK_route"]==expected
    wrong[0]=True
    with pytest.raises(RuntimeError,match="SDK route"):
        adapter.predict_component(2,query)
    assert adapter.failure


def test_native_immutable_audit_failure_still_releases_before_precision(monkeypatch):
    order=[];a=adapter_for_close(order)
    a.stage_index=6;a.stage_heads=4;a.model=None
    a.immutable_native_before={"weights":"original"}
    a.engine.__class__.hybrid_counters=property(lambda self:
        {"resident_component_predict_count":0,"uncached_predict_count":4})
    monkeypatch.setattr(p.backend,"immutable_native_snapshot",lambda *args:{"weights":"changed"})
    with pytest.raises(RuntimeError,match="immutable"):
        a.close()
    assert order==["release","precision_exit"]
    assert a.stages[-1]["engine_release_succeeded"]
    assert a.stages[-1]["execution_route"]=="original_fit_preprocessors_all8"
