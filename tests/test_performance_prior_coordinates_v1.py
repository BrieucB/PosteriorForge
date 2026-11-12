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
        p.qualification_receipt("missing",banks="missing",prefix_run="missing",codec_inventory="missing",codec_reference_inventory="missing",
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
