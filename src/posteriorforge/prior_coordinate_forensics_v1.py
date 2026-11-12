"""Observation-only prior-coordinate failure evidence; never used for acceptance.

Built with PriorLabs-TabPFN. No Torch import, model call, inverse-CDF sampling or
random draw occurs here. Prefix decoding is bounded DIAGNOSTIC recomputation
after failure, not an actual PFN/ICDF intermediate witness.
"""
from __future__ import annotations
import hashlib,json,pickle,random
from pathlib import Path
import numpy as np
from . import prior_coordinates_v1 as codec
from .artifacts import versions,sha256_file
from .science.analytical_benchmark_v2 import array_descriptor

SCHEMA="posteriorforge.prior_coordinate_forensics.v1"
CODEC_SHA256="d8ce6c49e7b65c6ee55c59b7243d4ce943cbea28c92ebed0329798074aaa70c5"
MAX_FAILURE_ROWS=32
MAX_CODEC_CALLS=2048


def source_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def require_sources():
    if codec.source_sha256()!=CODEC_SHA256:raise ValueError("frozen codec changed")
    codec.require_sources()


def default_rng_hash():
    return hashlib.sha256(pickle.dumps((np.random.get_state(),random.getstate()),protocol=5)).hexdigest()


def scalar(value):
    """JSON-safe display plus exact bytes are retained separately."""
    value=float(value)
    if np.isnan(value):return "NaN"
    if np.isposinf(value):return "+Infinity"
    if np.isneginf(value):return "-Infinity"
    return value


class StageDrawObserver:
    """Transparent observer of actual complete stage_draw return vectors only."""
    def __init__(self,draw,*,rng_snapshot=None):
        self.draw=draw;self.draw_code=getattr(draw,"__code__",None)
        self.rng_snapshot=rng_snapshot
        self.wave_index=None;self.count=0;self.returned=[];self.raw_returns={};self.events=[]
        self.active_stage=None;self.active_incomplete=False;self.call_returned=False;self.capture_errors=[]

    def begin_wave(self,wave_index,count):
        self.wave_index=int(wave_index);self.count=int(count)
        self.returned=[];self.raw_returns={};self.events=[];self.active_stage=None;self.active_incomplete=False;self.call_returned=False
        self.capture_errors=[]

    def _snapshot(self):
        if self.rng_snapshot is None:return None
        try:return self.rng_snapshot()
        except BaseException as error:
            self.capture_errors.append({"type":type(error).__name__,"error":str(error),"where":"RNG snapshot"})
            return None

    def __call__(self,engine,stage,query_for,count,chunk_size):
        self.active_stage=int(stage);self.active_incomplete=True;self.call_returned=False
        before=self._snapshot()
        try:
            result=self.draw(engine,stage,query_for,count,chunk_size)
        except BaseException:
            self.events.append({"stage":int(stage),"returned":False,
                "private_streams_before":before,"private_streams_after_or_failure":self._snapshot()})
            raise
        # Keep arguments, return object and original call order unmodified.
        self.call_returned=True
        sample=result[0]
        if isinstance(sample,np.ndarray):self.raw_returns[int(stage)]=sample
        try:
            if not isinstance(sample,np.ndarray) or sample.shape!=(count,):
                raise ValueError("actual stage return is not ndarray[count]")
            if stage!=len(self.returned):raise ValueError("nonsequential actual stage return")
            self.returned.append(sample.copy())
            self.events.append({"stage":int(stage),"returned":True,
                "actual_returned_draws":array_descriptor(sample),
                "returned_dtype_preserved":True,
                "original4037_stored_prefix_bytes_equal_return":sample.dtype==np.float32,
                "private_streams_before":before,"private_streams_after_or_failure":self._snapshot()})
            self.active_incomplete=False
        except BaseException as error:
            self.capture_errors.append({"type":type(error).__name__,"error":str(error),"where":"stage return copy"})
        return result

    def filled_stages(self):
        count=0
        while count in self.raw_returns:
            value=self.raw_returns[count]
            if value.dtype!=np.float32 or value.shape!=(self.count,):break
            count+=1
        return count

    def prefix(self):
        count=self.filled_stages()
        if not count:return np.empty((self.count,0),dtype=np.float32)
        return np.ascontiguousarray(np.column_stack([self.raw_returns[j] for j in range(count)]))

    def metadata(self,*,active_stage=None):
        stage=self.active_stage if active_stage is None else active_stage
        filled=self.filled_stages()
        attempted=stage is not None and stage==self.active_stage
        returned=bool(attempted and self.call_returned)
        return {"schema":SCHEMA+".actual_stage_returns","module_sha256":source_sha256(),
            "wave_index":self.wave_index,"row_count":self.count,"filled_stage_count":filled,
            "filled_stage_mask":[j<filled for j in range(21)],
            "actual_completed_stage_indices":list(range(filled)),
            "physical_slots_of_filled_stages":list(codec.ORDER[:filled]),
            "active_stage":stage,
            "active_stage_call_attempted":attempted,
            "active_stage_call_returned":returned,
            "active_stage_return_incomplete":bool(attempted and not returned),
            "observer_capture_incomplete_after_actual_return":bool(returned and self.capture_errors),
            "active_stage_partial_output_unavailable":bool(stage is not None and int(stage)>=filled),
            "future_uninitialized_columns_serialized":False,
            "current_stage_outputs_fabricated":False,
            "actual_returned_stage_indices":sorted(self.raw_returns),
            "successfully_copied_stage_count":len(self.returned),
            "actual_stage_return_events":self.events,"capture_errors":self.capture_errors,
            "sampler_callable_identity_unchanged":getattr(self.draw,"__code__",None) is self.draw_code,
            "original4037_storage_semantics":"np.asarray(returned,dtype=Float32); values[:,stage]=sample",
            "stored_prefix_reconstruction_scope":"only when actual returned dtype already Float32; no extra cast",
            "observation_changes_sampling":False}


def archive_actual_stage_returns(output,observer,*,active_stage=None):
    """Persist raw returned vectors; only source-proven stored prefix is emitted."""
    path=Path(output)/"latent-stage-returns.npz"
    arrays={f"stage_{j:04d}":v for j,v in sorted(observer.raw_returns.items())}
    metadata=observer.metadata(active_stage=active_stage)
    arrays["filled_stage_mask"]=np.asarray(metadata["filled_stage_mask"],dtype=np.bool_)
    try:
        prefix=observer.prefix()
        arrays["ordered_latent_prefix"]=prefix
        metadata["source_proven_stored_prefix"]=array_descriptor(prefix)
        metadata["stored_prefix_available"]=True
    except BaseException as error:
        metadata["stored_prefix_available"]=False
        metadata["stored_prefix_observation_error"]={"type":type(error).__name__,"error":str(error)}
    with path.open("xb") as stream:np.savez(stream,**arrays)
    return {"file":path.name,"sha256":sha256_file(path),"keys":list(arrays),
            "raw_return_dtype_and_bytes_preserved":True,"observation":metadata}


def diagnose_prefix(values,*,max_failure_rows=MAX_FAILURE_ROWS,max_codec_calls=MAX_CODEC_CALLS):
    """Bounded replay of saved actual latent prefixes, never a repair or retry."""
    before=default_rng_hash();require_sources()
    if not isinstance(values,np.ndarray) or values.dtype not in (np.dtype("float32"),np.dtype("float64")):
        raise ValueError("original Float32/Float64 ndarray required without recast")
    if values.ndim!=2 or not 1<=values.shape[1]<=21 or len(values)<1:
        raise ValueError("nonempty actually-filled ordered prefix required")
    if type(max_failure_rows) is not int or max_failure_rows<1 or type(max_codec_calls) is not int or max_codec_calls<1:
        raise ValueError("positive forensic bounds required")
    original=array_descriptor(values);rows,width=values.shape
    finite=np.isfinite(values);badrows=np.flatnonzero(~finite.all(axis=1))
    report={"schema":SCHEMA,"source_sha256":source_sha256(),"codec_sha256":CODEC_SHA256,
        "runtime_versions":{k:versions()[k] for k in ("python","numpy","scipy")},
        "input":original,"actually_filled_stage_count":width,
        "stage_to_physical_slots":list(codec.ORDER[:width]),
        "diagnostic_recomputation_not_actual_PFN_or_ICDF_witness":True,
        "no_sampling_model_RNG_repair_or_acceptance":True,
        "bounds":{"max_failure_rows":max_failure_rows,"max_codec_calls":max_codec_calls},
        "raw_nonfinite_row_count":int(len(badrows)),
        "raw_nonfinite_cell_count":int((~finite).sum()),"failures":[]}
    calls=0;verified_valid=0;finite_failures=0;finite_unexamined=0
    def decode(batch):
        nonlocal calls
        if calls>=max_codec_calls:raise RuntimeError("forensic codec-call bound exhausted")
        calls+=1
        return codec.decode_ordered_prefix(batch)
    def failure_row(index,observed_reason=None):
        row=values[index:index+1];last=None;first=None;reason=observed_reason
        for stage in range(width):
            if calls>=max_codec_calls:break
            try:last=decode(row[:,:stage+1])
            except codec.CoordinateError as error:
                first=stage;reason=str(error);break
        nonfinite=np.flatnonzero(~np.isfinite(row[0]))
        entry={"row_index":int(index),"ordered_latent_values":[scalar(x) for x in row[0]],
            "raw_latent_row_bytes_hex":row.tobytes().hex(),
            "input_nonfinite_locations":[{"stage_index":int(j),"physical_slot":int(codec.ORDER[j]),
                "latent_value":scalar(row[0,j]),"latent_scalar_bytes_hex":row[0,j:j+1].tobytes().hex()} for j in nonfinite],
            "first_failing_causal_stage":first,"reason":reason,
            "stage_localization_complete":first is not None,
            "diagnostic_recomputed_last_valid_physical_prefix":last[0].tolist() if last is not None else [],
            "last_valid_physical_prefix_slots":list(codec.ORDER[:last.shape[1]]) if last is not None else [],
            "first_failure_is_after_last_valid_prefix":True}
        if first is not None:
            slot=int(codec.ORDER[first]);entry.update(physical_slot=slot,
                physical_parameter=codec.PHYSICAL_ORDER[slot],latent_value_at_stage=scalar(row[0,first]),
                latent_scalar_bytes_hex=row[0,first:first+1].tobytes().hex())
        report["failures"].append(entry)
    # Direct input nonfinites are counted over every actual row; details bounded.
    for index in badrows[:max_failure_rows]:failure_row(int(index),"raw nonfinite latent")
    finite_indices=np.flatnonzero(finite.all(axis=1))
    pending=[finite_indices] if len(finite_indices) else []
    while pending:
        indices=pending.pop()
        if calls>=max_codec_calls or len(report["failures"])>=max_failure_rows:
            finite_unexamined+=len(indices)+sum(map(len,pending));break
        try:
            decode(values[indices])
            verified_valid+=len(indices)
        except codec.CoordinateError as error:
            if len(indices)==1:
                finite_failures+=1;failure_row(int(indices[0]),str(error))
            else:
                split=len(indices)//2
                pending.append(indices[split:]);pending.append(indices[:split])
    localized={r["row_index"] for r in report["failures"] if r["stage_localization_complete"]}
    unlocalized=sum(not r["stage_localization_complete"] for r in report["failures"])
    missing_nonfinite=sum(int(i) not in localized for i in badrows)
    report.update(codec_replay_calls=calls,finite_rows_verified_valid=int(verified_valid),
        finite_failing_rows_identified=int(finite_failures),finite_rows_not_examined=int(finite_unexamined),
        nonfinite_rows_without_detailed_localization=int(missing_nonfinite),
        emitted_failure_rows_without_stage_localization=int(unlocalized),
        failure_details_truncated=bool(finite_unexamined or missing_nonfinite or unlocalized),
        complete_failure_inventory=bool(not finite_unexamined and not missing_nonfinite and not unlocalized),
        input_unchanged=array_descriptor(values)==original)
    after=default_rng_hash()
    report.update(default_numpy_python_RNG_before_sha256=before,
        default_numpy_python_RNG_after_sha256=after,default_numpy_python_RNG_preserved=before==after)
    if before!=after:raise RuntimeError("forensic observer changed default RNG")
    return report


def safe_diagnose_prefix(values,**kwargs):
    """Diagnostic failures are secondary and cannot replace primary exceptions."""
    try:return diagnose_prefix(values,**kwargs)
    except BaseException as error:
        return {"schema":SCHEMA,"status":"forensic_observation_failed",
            "source_sha256":source_sha256(),"error_type":type(error).__name__,"error":str(error),
            "primary_failure_must_be_preserved":True}
