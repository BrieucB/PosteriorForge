"""Observation-only failure forensics, exact returned buffers, no model or GPU."""
import hashlib,json,pickle,random
import numpy as np
import pytest
from posteriorforge import prior_coordinate_forensics_v1 as f
from posteriorforge import performance_prior_coordinates_v1 as p


def state():
    return pickle.dumps((np.random.get_state(),random.getstate()),protocol=5)


def test_valid_decoded_full_wave_archive_contains_unused_tail(tmp_path):
    latent=np.zeros((51200,21),np.float32);latent[-1,3]=1.
    raw=[];decoded=[];status={"decode_wave_records":[]}
    physical=p.record_decode_wave(latent,raw,status,decoded_waves=decoded)
    info=p.archive_decoded_waves(tmp_path,decoded)
    with np.load(tmp_path/info["file"],allow_pickle=False) as archive:
        assert archive.files==["wave_0000"]
        assert archive["wave_0000"].dtype==np.float64
        assert archive["wave_0000"].tobytes()==physical.tobytes()
    assert info["generated_valid_row_count"]==51200
    assert info["all_rows_including_unused_FIFO_tail"]
    assert info["sha256"]==hashlib.sha256((tmp_path/info["file"]).read_bytes()).hexdigest()


def test_multifault_rows_slots_prefixes_and_default_rng_preserved():
    values=np.zeros((4,21),np.float32)
    values[1,5]=np.nan;values[2,17]=np.inf;values[3,20]=-np.inf
    raw=values.tobytes();before=state()
    report=f.diagnose_prefix(values)
    assert state()==before and values.tobytes()==raw
    assert report["raw_nonfinite_row_count"]==3
    assert report["default_numpy_python_RNG_preserved"]
    failures={r["row_index"]:r for r in report["failures"]}
    for row,stage in [(1,5),(2,17),(3,20)]:
        result=failures[row]
        assert result["first_failing_causal_stage"]==stage
        assert result["physical_slot"]==f.codec.ORDER[stage]
        assert len(result["diagnostic_recomputed_last_valid_physical_prefix"])==stage
        assert result["raw_latent_row_bytes_hex"]==values[row:row+1].tobytes().hex()
    json.dumps(report,allow_nan=False)


def test_finite_saturation_is_localized_without_endpoint_repair():
    values=np.zeros((2,21),np.float32);values[1,3]=100.
    before=values.tobytes()
    report=f.diagnose_prefix(values)
    result=next(r for r in report["failures"] if r["row_index"]==1)
    assert result["first_failing_causal_stage"]==3 and result["physical_slot"]==6
    assert "saturat" in result["reason"] or "endpoint" in result["reason"]
    assert result["latent_value_at_stage"]==100.
    assert len(result["diagnostic_recomputed_last_valid_physical_prefix"])==3
    assert values.tobytes()==before
    assert report["diagnostic_recomputation_not_actual_PFN_or_ICDF_witness"]


def test_bounded_details_do_not_claim_complete_inventory():
    values=np.zeros((10,21),np.float32);values[:,10]=np.nan
    report=f.diagnose_prefix(values,max_failure_rows=2,max_codec_calls=128)
    assert report["raw_nonfinite_row_count"]==10 and len(report["failures"])==2
    assert report["nonfinite_rows_without_detailed_localization"]==8
    assert report["failure_details_truncated"] and not report["complete_failure_inventory"]


def test_exhausted_prefix_budget_reports_unlocalized_details():
    values=np.zeros((1,21),np.float32);values[0,10]=np.nan
    report=f.diagnose_prefix(values,max_failure_rows=2,max_codec_calls=1)
    assert report["codec_replay_calls"]==1
    assert report["nonfinite_rows_without_detailed_localization"]==1
    assert report["emitted_failure_rows_without_stage_localization"]==1
    assert report["failure_details_truncated"] and not report["complete_failure_inventory"]


def test_partial_actual_stage_returns_preserved_no_future_columns(tmp_path):
    result_objects=[];calls=[]
    def draw(engine,stage,query_for,count,chunk):
        calls.append(stage)
        if stage==2:raise RuntimeError("query rejected nonfinite prior prefix")
        values=np.full(count,stage,np.float32)
        if stage==1:values[-1]=np.nan
        result=(values,{"unchanged":True});result_objects.append(result);return result
    observer=f.StageDrawObserver(draw)
    observer.begin_wave(3,7)
    assert observer(None,0,None,7,1024) is result_objects[0]
    assert observer(None,1,None,7,1024) is result_objects[1]
    with pytest.raises(RuntimeError):observer(None,2,None,7,1024)
    info=f.archive_actual_stage_returns(tmp_path,observer,active_stage=2)
    with np.load(tmp_path/info["file"],allow_pickle=False) as archive:
        assert archive["ordered_latent_prefix"].shape==(7,2)
        assert archive["ordered_latent_prefix"][:,-1].tobytes()==result_objects[1][0].tobytes()
        assert archive["filled_stage_mask"].tolist()==[True,True]+[False]*19
        assert set(archive.files)=={"stage_0000","stage_0001","filled_stage_mask","ordered_latent_prefix"}
    metadata=info["observation"]
    assert calls==[0,1,2] and metadata["active_stage_return_incomplete"]
    assert not metadata["active_stage_call_returned"]
    report=f.diagnose_prefix(observer.prefix())
    assert report["failures"][0]["row_index"]==6
    assert report["failures"][0]["first_failing_causal_stage"]==1


def test_raw_return_dtype_preserved_without_forensic_cast(tmp_path):
    vector=np.array([.1,.2,.3],np.float64);result=(vector,{})
    observer=f.StageDrawObserver(lambda *args:result);observer.begin_wave(0,3)
    assert observer(None,0,None,3,1024) is result
    info=f.archive_actual_stage_returns(tmp_path,observer,active_stage=0)
    with np.load(tmp_path/info["file"],allow_pickle=False) as archive:
        assert archive["stage_0000"].dtype==np.float64
        assert archive["stage_0000"].tobytes()==vector.tobytes()
        assert archive["ordered_latent_prefix"].shape==(3,0)
    assert not info["observation"]["future_uninitialized_columns_serialized"]


def test_copy_failure_does_not_claim_sampler_failed_to_return():
    class CopyFailure(np.ndarray):
        def copy(self,*args,**kwargs):raise MemoryError("copy failure")
    vector=np.zeros(3,np.float32).view(CopyFailure);result=(vector,{})
    observer=f.StageDrawObserver(lambda *args:result);observer.begin_wave(0,3)
    assert observer(None,0,None,3,1024) is result
    report=observer.metadata(active_stage=0)
    assert report["active_stage_call_returned"] and not report["active_stage_return_incomplete"]
    assert report["observer_capture_incomplete_after_actual_return"]
    assert observer.raw_returns[0] is vector


def test_raw_archives_precede_diagnostic_and_errors_secondary(tmp_path,monkeypatch):
    latent=np.zeros((8,21),np.float32);latent[-1,3]=np.nan
    raw=[];decoded=[];status={"decode_wave_records":[]}
    primary=None
    try:p.record_decode_wave(latent,raw,status,decoded_waves=decoded)
    except ValueError as error:primary=error
    assert primary is not None
    def failed_diagnostic(values):
        assert (tmp_path/"latent-proposal-waves.npz").exists()
        raise MemoryError("secondary forensic error")
    monkeypatch.setattr(p.forensics,"safe_diagnose_prefix",failed_diagnostic)
    p.preserve_failure_artifacts(tmp_path,status,raw,decoded)
    assert status["failure_forensics_preservation_error"]["type"]=="MemoryError"
    assert not (tmp_path/"samples.npz").exists()
    with np.load(tmp_path/"latent-proposal-waves.npz",allow_pickle=False) as archive:
        assert archive["wave_0000"].tobytes()==latent.tobytes()


def test_private_cpu_generator_trajectory_unchanged_by_observer():
    torch=pytest.importorskip("torch")
    first=torch.Generator(device="cpu").manual_seed(872)
    second=torch.Generator(device="cpu").manual_seed(872)
    def sampler(generator):
        def draw(engine,stage,query_for,count,chunk):
            return torch.rand(count,generator=generator,dtype=torch.float32).numpy(),{}
        return draw
    original=sampler(first);observed_original=sampler(second)
    observer=f.StageDrawObserver(observed_original,rng_snapshot=lambda:
        hashlib.sha256(second.get_state().numpy().tobytes()).hexdigest())
    observer.begin_wave(0,16);default=state();torch_default=torch.get_rng_state().clone()
    for stage in range(4):
        expected=original(None,stage,None,16,1024)
        actual=observer(None,stage,None,16,1024)
        assert actual[0].tobytes()==expected[0].tobytes()
    assert torch.equal(first.get_state(),second.get_state())
    assert torch.equal(torch_default,torch.get_rng_state()) and state()==default
