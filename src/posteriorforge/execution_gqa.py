"""Experimental FP32 shared-K/V layout expansion, explicitly scoped.

Built with PriorLabs-TabPFN. SDK files and prefill projections are unchanged.
Only a query with one shared KV head is expanded as a storage-sharing view to
the query head count before the original attention function is called.
"""
from __future__ import annotations
from contextlib import contextmanager
import hashlib
from importlib import import_module
import inspect
import marshal
from pathlib import Path
from threading import Lock,get_ident

from . import execution_cache as cache

ENGINE_ID="expanded_kv_stage_fp32_v1"
ENGINE_VERSION=1
PINNED_CACHE_SHA256="45cd3107c50dcf73d03167e2b6dd045cccdb798337e8d3c47b5210cc30dc1a03"
PINNED_ATTENTION_SHA256="512220bbf253fb856003f7084976a478753644c9963d563ee678eb3be2c28f35"
_LOCK=Lock()


class GQAExecutionError(RuntimeError):pass


def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def _code_sha(function):return hashlib.sha256(marshal.dumps(function.__code__)).hexdigest()
def execution_source_sha256():return _sha(__file__)


def execution_metadata():
    return {"schema":"posteriorforge.execution.gqa.v1","engine_id":ENGINE_ID,"engine_version":ENGINE_VERSION,
        "module_sha256":execution_source_sha256(),"resident_cache_module_sha256":PINNED_CACHE_SHA256,
        "attention_source_sha256":PINNED_ATTENTION_SHA256,
        "scope":"query_single_kv_head_stride_zero_expansion_only","prediction_math_dtype":"torch.float32",
        "prefill_projection_changed":False,"dtype_changed":False,"dropout_changed":False,
        "softmax_scale_changed":False,"sdk_files_modified":False,"qualified":False,
        "attribution":"Built with PriorLabs-TabPFN"}


def _load_guarded_attention():
    module=import_module("tabpfn.model.multi_head_attention")
    cls=module.MultiHeadAttention
    original=cls.compute_attention_heads
    descriptor=inspect.getattr_static(cls,"compute_attention_heads")
    if cache.execution_source_sha256()!=PINNED_CACHE_SHA256 or _sha(module.__file__)!=PINNED_ATTENTION_SHA256:
        raise GQAExecutionError("pinned cache/attention source mismatch")
    if (not isinstance(descriptor,staticmethod) or descriptor.__func__ is not original
        or original.__module__!="tabpfn.model.multi_head_attention"
        or original.__qualname__!="MultiHeadAttention.compute_attention_heads"
        or original.__globals__.get("MultiHeadAttention") is not cls
        or Path(inspect.getfile(original)).resolve()!=Path(module.__file__).resolve()):
        raise GQAExecutionError("original attention alias identity mismatch")
    return cls,original,descriptor,{"attention_source":str(Path(module.__file__).resolve()),
           "attention_source_sha256":_sha(module.__file__),"resident_cache_module_sha256":cache.execution_source_sha256()}


def _head_shape(q,k,v,kv,qkv):
    return {"q":None if q is None else list(q.shape),"k":None if k is None else list(k.shape),
        "v":None if v is None else list(v.shape),"kv":None if kv is None else list(kv.shape),
        "qkv":None if qkv is None else list(qkv.shape),
        "dtype":str(next(t for t in (q,qkv,kv,k,v) if t is not None).dtype)}


def _expand(tensor,heads):
    view=tensor.expand(*tensor.shape[:-2],heads,tensor.shape[-1])
    shared=tensor.untyped_storage().data_ptr()==view.untyped_storage().data_ptr()
    if not shared or view.stride()[-2]!=0:
        raise GQAExecutionError("KV expansion must be a storage-sharing stride-zero view")
    return view,{"before":list(tensor.shape),"after":list(view.shape),
        "before_stride":list(tensor.stride()),"after_stride":list(view.stride()),
        "shares_storage":True,"logical_bytes_before":tensor.numel()*tensor.element_size(),
        "logical_bytes_after":view.numel()*view.element_size(),
        "storage_bytes":tensor.untyped_storage().nbytes()}


@contextmanager
def expanded_kv_attention(*,enabled=True):
    """Temporarily replace one static alias; restore its exact original object.

    enabled=False is a passive shape observer for controlled diagnostic traces.
    The context never draws RNG or changes dtype/scale/dropout. Use is exclusive
    to its owner thread; unadapted concurrent attention calls fail closed.
    """
    if type(enabled) is not bool:raise ValueError("explicit boolean expansion policy required")
    if not _LOCK.acquire(blocking=False):raise GQAExecutionError("GQA context already active")
    try:
        cls,original,original_descriptor,sources=_load_guarded_attention()
        owner=get_ident();original_code=_code_sha(original);original_code_object=original.__code__
        report={**execution_metadata(),**sources,"schema":"posteriorforge.execution.gqa_context.v1",
            "expansion_enabled":enabled,"original_alias_restored":False,"source_guard_passed":True,
            "original_attention_code_sha256":original_code,"attention_call_attempts":0,
            "attention_call_count":0,"expanded_attention_calls":0,"expanded_tensor_views":0,
            "head_shape_histogram":{},"effective_head_shape_histogram":{},"histogram_overflow_calls":0,"view_examples":[]}

        def wrapped(q,k,v,kv,qkv,dropout_p=None,softmax_scale=None):
            if get_ident()!=owner:raise GQAExecutionError("attention called from nonowner thread during scoped override")
            report["attention_call_attempts"]+=1
            shape=_head_shape(q,k,v,kv,qkv);key=str(shape)
            histogram=report["head_shape_histogram"]
            if key in histogram or len(histogram)<128:histogram[key]=histogram.get(key,0)+1
            else:report["histogram_overflow_calls"]+=1
            expanded=[]
            if enabled and q is not None and qkv is None and q.shape[-2]>1:
                heads=q.shape[-2]
                if kv is not None and kv.shape[-2]==1:
                    if str(q.dtype)!="torch.float32" or str(kv.dtype)!="torch.float32":
                        raise GQAExecutionError("only FP32 shared-KV expansion is qualified for this experiment")
                    kv,item=_expand(kv,heads);expanded.append(item)
                elif kv is None and k is not None and v is not None and k.shape[-2]==v.shape[-2]==1:
                    if any(str(t.dtype)!="torch.float32" for t in (q,k,v)):
                        raise GQAExecutionError("only FP32 shared-KV expansion is qualified for this experiment")
                    k,item=_expand(k,heads);expanded.append(item)
                    v,item=_expand(v,heads);expanded.append(item)
            effective_key=str(_head_shape(q,k,v,kv,qkv))
            effective=report["effective_head_shape_histogram"]
            if effective_key in effective or len(effective)<128:effective[effective_key]=effective.get(effective_key,0)+1
            output=original(q,k,v,kv,qkv,dropout_p=dropout_p,softmax_scale=softmax_scale)
            report["attention_call_count"]+=1
            report["executed_original_attention_code_sha256"]=original_code
            if expanded:
                report["expanded_attention_calls"]+=1;report["expanded_tensor_views"]+=len(expanded)
                for item in expanded:
                    if len(report["view_examples"])<16:report["view_examples"].append(item)
            return output

        active_code_object=wrapped.__code__
        active_descriptor=staticmethod(wrapped)
        report["active_attention_code_sha256"]=_code_sha(wrapped)
        cls.compute_attention_heads=active_descriptor
        try:
            yield report
        finally:
            conflict=inspect.getattr_static(cls,"compute_attention_heads") is not active_descriptor
            cls.compute_attention_heads=original_descriptor
            report["original_alias_restored"]=inspect.getattr_static(cls,"compute_attention_heads") is original_descriptor
            report["original_function_restored"]=cls.compute_attention_heads is original
            report["original_code_unchanged"]=original.__code__ is original_code_object
            report["active_code_unchanged"]=wrapped.__code__ is active_code_object
            report["gqa_source_unchanged"]=execution_source_sha256()==report["module_sha256"]
            report["sdk_sources_unchanged"]=(
                _sha(sources["attention_source"])==PINNED_ATTENTION_SHA256
                and cache.execution_source_sha256()==PINNED_CACHE_SHA256)
            if conflict or not report["original_function_restored"] or not report["original_code_unchanged"] or not report["active_code_unchanged"] or not report["gqa_source_unchanged"]:
                raise GQAExecutionError("conflicting attention alias/code change")
            if not report["sdk_sources_unchanged"]:raise GQAExecutionError("GQA context source changed")
    finally:_LOCK.release()


class ExpandedKVStageExecutor(cache.ResidentStageExecutor):
    """Original stage prefill; scoped expanded layout during query prediction."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.last_gqa_descriptor=None
        self.metadata["gqa_execution"]=execution_metadata()
        self.metadata["gqa_contexts"]=0;self.metadata["gqa_expanded_attention_calls"]=0

    def predict(self,*args,**kwargs):
        descriptor=None
        try:
            with expanded_kv_attention() as descriptor:
                return super().predict(*args,**kwargs)
        finally:
            if descriptor is not None:
                self.last_gqa_descriptor=descriptor
                self.metadata["gqa_contexts"]+=1
                self.metadata["gqa_expanded_attention_calls"]+=descriptor["expanded_attention_calls"]
