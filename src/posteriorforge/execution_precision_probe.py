"""Independent approximate-forward precision experiment; never an FP32 qualification.

Built with PriorLabs-TabPFN. No frozen engine or SDK file is modified. Private
source-guarded clones change only precision selection/prefill contexts; every
post-forward temperature/CDF/sampler operation remains the original FP32 path.
"""
from __future__ import annotations
import ast
import __future__
from collections import Counter
from contextlib import contextmanager
import copy
import hashlib
import inspect
import json
from pathlib import Path
from threading import Lock, get_ident
from types import MethodType, SimpleNamespace
import textwrap

from . import execution_cache as cache, execution_hybrid as hybrid
from . import execution_mixture as mixture, execution_gqa as gqa

SCHEMA="posteriorforge.execution.precision_probe.v1"
MODES=("tf32_high_forward_v1","amp_bf16_prefill_query_v1","amp_fp16_prefill_query_v1")
CONTROL_MODE="fp32_transparent_control_v1"
TORCH_AMP_SOURCE_SHA256="37087fa6590d83ba1a8b30e9bb658ec9c310a93a1d0b82713b3c9acb1e5ac1f0"
PINNED={"cache":hybrid.PINNED_CACHE_SHA256,"hybrid":"5dfdefa870489ded594de1db13a55a3da5948c720d7a9fd587ebe3a8f8be00a2",
        "mixture":hybrid.PINNED_MIXTURE_SHA256,"gqa":"8b2cff752855e1b1966d2e374322602fa71368b1926da9d3b0261edd111aee6b"}
_LOCK=Lock()
_ACTIVE=None


def source_sha256():return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
def _sha_ast(node):return hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()
def require_sources():
    for name,module in (("cache",cache),("hybrid",hybrid),("mixture",mixture),("gqa",gqa)):
        if module.execution_source_sha256()!=PINNED[name]:raise RuntimeError("frozen source changed: "+name)


def metadata(mode):
    if mode not in MODES+(CONTROL_MODE,):raise ValueError("explicit precision experiment required")
    return {"schema":SCHEMA,"mode":mode,"module_sha256":source_sha256(),"frozen_sources":dict(PINNED),
            "approximate":mode!=CONTROL_MODE,"qualified_fp32":False,"statistical_candidate":False,
            "fallback_map":{"10":[5]},"fallback_precision":"entire original FP32 uncached path",
            "postprocessing":"original output.float()/0.9, cancellation, corrected CDF, FP32 mixture/log/softmax/ICDF",
            "cache_precision_claim":"computed projection and stored buffer dtypes require actual prefill witnesses; never infer from settings or cast old FP32 KV",
            "attribution":"Built with PriorLabs-TabPFN"}


def _clone(function,transform,namespace):
    tree=ast.parse(textwrap.dedent(inspect.getsource(function)));node=tree.body[0]
    node.decorator_list=[]
    original=copy.deepcopy(node);changes=transform(node)
    ast.fix_missing_locations(tree)
    scope=dict(function.__globals__);scope.update(namespace)
    exec(compile(tree,"<posteriorforge-private-precision-clone>","exec",
                 flags=__future__.annotations.compiler_flag,dont_inherit=True),scope)
    return scope[function.__name__],{"original_ast_sha256":_sha_ast(original),
        "private_ast_sha256":_sha_ast(node),"declared_changes":changes}


def _session():
    if _ACTIVE is None or _ACTIVE.owner!=get_ident():raise RuntimeError("active owner-thread precision session required")
    return _ACTIVE


def _inference_setting():
    session=_session()
    return "autocast" if session.mode.startswith("amp_") else session.torch.float32


def _allowed_prepared_dtype(dtype):
    session=_session()
    allowed={session.torch.float32}
    if session.mode.startswith("amp_"):allowed.add(session.low_dtype)
    return dtype in allowed


def _assumption_clone():
    def transform(node):
        changed=0
        for child in ast.walk(node):
            if isinstance(child,ast.If) and ast.dump(child.test,include_attributes=False)==ast.dump(
                ast.parse("tensor.is_floating_point() and tensor.dtype!=torch.float32",mode="eval").body,include_attributes=False):
                child.test=ast.parse("tensor.is_floating_point() and not _allowed_prepared_dtype(tensor.dtype)",mode="eval").body
                changed+=1
        if changed!=1:raise RuntimeError("one pinned prepared-dtype predicate required")
        return {"prepared_dtype_predicates":changed}
    function,descriptor=_clone(cache._cache_assumptions,transform,{"_allowed_prepared_dtype":_allowed_prepared_dtype})
    def validate(models,torch):
        result=function(models,torch)
        for model in models:
            if any(p.is_floating_point() and p.dtype!=torch.float32 for p in model.parameters()):
                raise RuntimeError("parameters must remain FP32; no post-hoc model/KV casting")
            for module in model.modules():
                if type(module).__name__=="InputNormalizationEncoderStep":
                    for name in ("mean_for_normalization","std_for_normalization","lower_for_outlier_removal","upper_for_outlier_removal"):
                        value=getattr(module,name,None)
                        if torch.is_tensor(value) and value.is_floating_point() and value.dtype!=torch.float32:
                            raise RuntimeError("input normalization statistics changed from FP32")
        return result
    return validate,descriptor


def _resident_prepare_clone():
    assumption,assumption_descriptor=_assumption_clone()
    def transform(node):
        changed=0
        for child in ast.walk(node):
            if (isinstance(child,ast.Assign) and len(child.targets)==1
                and isinstance(child.targets[0],ast.Subscript)
                and isinstance(child.targets[0].value,ast.Name) and child.targets[0].value.id=="settings"
                and isinstance(child.targets[0].slice,ast.Constant) and child.targets[0].slice.value=="inference_precision"):
                if ast.dump(child.value,include_attributes=False)!=ast.dump(ast.parse("torch.float32",mode="eval").body,include_attributes=False):
                    raise RuntimeError("pinned precision assignment changed")
                child.value=ast.parse("_inference_setting()",mode="eval").body;changed+=1
        if changed!=1:raise RuntimeError("one original precision assignment required")
        return {"inference_precision_assignments":changed}
    function,descriptor=_clone(cache.ResidentStageExecutor.prepare,transform,
        {"_inference_setting":_inference_setting,"_cache_assumptions":assumption})
    descriptor["cache_assumptions"]=assumption_descriptor
    return function,descriptor


@contextmanager
def _prefill_scope(member,native_autocast):
    session=_session()
    expected=session.mode.startswith("amp_")
    if bool(native_autocast)!=expected:raise RuntimeError("SDK prefill autocast setting did not select the intended path")
    with session.forward_scope("prefill",session.stage,member):
        with session.torch.autocast("cuda",enabled=expected and not session.strict_fallback(session.stage,member),dtype=session.low_dtype):
            yield


def _prefill_clone(original):
    def transform(node):
        changed=0
        target=ast.dump(ast.parse("torch.autocast(device.type, enabled=autocast)",mode="eval").body,include_attributes=False)
        for child in ast.walk(node):
            if isinstance(child,ast.With):
                for item in child.items:
                    if ast.dump(item.context_expr,include_attributes=False)==target:
                        item.context_expr=ast.parse("_prefill_scope(len(models),autocast)",mode="eval").body;changed+=1
        if changed!=1:raise RuntimeError("one original CacheKV prefill autocast context required")
        return {"prefill_autocast_contexts":changed}
    return _clone(original,transform,{"_prefill_scope":_prefill_scope})


class PrecisionSession:
    """Thread-exclusive source/precision/attention observation, with exact restore."""
    def __init__(self,mode,*,query_views=True):
        require_sources()
        if type(query_views) is not bool:raise ValueError("explicit boolean query-view policy required")
        self.mode=mode;self.descriptor=metadata(mode);self.query_views=query_views
        self.owner=None;self.stage=None;self.current=None;self.hist=Counter();self.examples=[]
        self.counters=Counter();self.clones={};self.restored=False

    def strict_fallback(self,stage,member):return stage==10 and member==5

    def _flags(self):
        torch=self.torch
        return {"matmul_precision":torch.get_float32_matmul_precision(),
                "allow_tf32":torch.backends.cuda.matmul.allow_tf32,
                "autocast_dtype":str(torch.get_autocast_dtype("cuda")),
                "autocast_enabled":torch.is_autocast_enabled("cuda")}

    @contextmanager
    def forward_scope(self,phase,stage,member):
        if self.owner!=get_ident():raise RuntimeError("precision context called by a nonowner")
        if phase not in ("prefill","query") or type(stage) is not int or not 0<=stage<21 or type(member) is not int or not 0<=member<8:
            raise ValueError("explicit logical forward phase/stage/member required")
        torch=self.torch;previous=self._flags();old=self.current
        if previous["autocast_enabled"]:raise RuntimeError("outer AMP must be disabled; SDK owns the forward AMP boundary")
        strict=self.strict_fallback(stage,member)
        try:
            tf32=self.mode=="tf32_high_forward_v1" and not strict
            torch.set_float32_matmul_precision("high" if tf32 else "highest")
            torch.backends.cuda.matmul.allow_tf32=tf32
            torch.set_autocast_dtype("cuda",torch.float16 if strict else self.low_dtype)
            self.current={"phase":phase,"stage":stage,"member":member,"strict_fp32_fallback":strict}
            yield
        finally:
            self.current=old
            torch.set_float32_matmul_precision(previous["matmul_precision"])
            torch.backends.cuda.matmul.allow_tf32=previous["allow_tf32"]
            torch.set_autocast_dtype("cuda",getattr(torch,previous["autocast_dtype"].split(".")[-1]))
            if self._flags()!=previous:raise RuntimeError("precision flags failed exact restoration")

    def _qkv_observe(self,module,args,kwargs,result):
        if self.current is None:raise RuntimeError("model projection outside its declared precision phase")
        event={**self.current,**self._flags(),"cache_kv":bool(kwargs.get("cache_kv")),
               "use_cached_kv":bool(kwargs.get("use_cached_kv")),
               "x_dtype":str(args[0].dtype),
               "projection_dtypes":{name:None if value is None else str(value.dtype)
                                    for name,value in zip(("q","k","v","kv","qkv"),result)},
               "supplied_cache_dtypes":[None if value is None else str(value.dtype) for value in args[2:5]]}
        key=json.dumps(event,sort_keys=True)
        if key in self.hist or len(self.hist)<512:self.hist[key]+=1
        else:self.counters["witness_histogram_overflow"]+=1
        self.counters["qkv_calls"]+=1
        if len(self.examples)<16:self.examples.append(event)

    def __enter__(self):
        global _ACTIVE
        if not _LOCK.acquire(blocking=False):raise RuntimeError("another precision session is active")
        try:
            require_sources()
            self.torch,self.regressor_module,self.inference_module=cache._load_sdk()
            amp_source=Path(self.torch.__file__).resolve().parent/"amp/autocast_mode.py"
            if hashlib.sha256(amp_source.read_bytes()).hexdigest()!=TORCH_AMP_SOURCE_SHA256:
                raise RuntimeError("pinned Torch autocast semantics source changed")
            self.descriptor.update(torch_version=str(self.torch.__version__),torch_autocast_source_sha256=TORCH_AMP_SOURCE_SHA256)
            self.low_dtype=(self.torch.bfloat16 if self.mode=="amp_bf16_prefill_query_v1" else self.torch.float16)
            self.owner=get_ident()
            self.attention_cls,self.original_attention,self.attention_descriptor,_=gqa._load_guarded_attention()
            self.qkv_original=self.attention_cls.compute_qkv
            self.prepare_descriptor=inspect.getattr_static(self.inference_module.InferenceEngineCacheKV,"prepare")
            if not isinstance(self.prepare_descriptor,classmethod):raise RuntimeError("original prefill classmethod required")
            prefill,self.clones["prefill"]=_prefill_clone(self.prepare_descriptor.__func__)
            self.private_prepare_descriptor=classmethod(prefill)
            self.resident_prepare,self.clones["resident_prepare"]=_resident_prepare_clone()
            original_qkv=self.qkv_original
            session=self
            def qkv(module,*args,**kwargs):
                result=original_qkv(module,*args,**kwargs);session._qkv_observe(module,args,kwargs,result);return result
            def attention(q,k,v,kv,qkv,dropout_p=None,softmax_scale=None):
                if session.owner!=get_ident() or session.current is None:raise RuntimeError("attention outside owned forward phase")
                state=session.current;allowed={session.torch.float32}
                if session.mode.startswith("amp_") and not state["strict_fp32_fallback"]:allowed.add(session.low_dtype)
                if any(t is not None and t.dtype not in allowed for t in (q,k,v,kv,qkv)):
                    raise RuntimeError("attention dtype outside declared precision policy")
                expanded=0
                if session.query_views and state["phase"]=="query" and not state["strict_fp32_fallback"] and q is not None and qkv is None and q.shape[-2]>1:
                    if kv is not None and kv.shape[-2]==1:kv,_=gqa._expand(kv,q.shape[-2]);expanded=1
                    elif kv is None and k is not None and v is not None and k.shape[-2]==v.shape[-2]==1:
                        k,_=gqa._expand(k,q.shape[-2]);v,_=gqa._expand(v,q.shape[-2]);expanded=2
                result=session.original_attention(q,k,v,kv,qkv,dropout_p=dropout_p,softmax_scale=softmax_scale)
                session.counters["attention_calls"]+=1;session.counters["expanded_views"]+=expanded
                return result
            self.qkv_private=qkv;self.attention_private=staticmethod(attention)
            self.initial_flags=self._flags()
            if self.initial_flags["autocast_enabled"]:raise RuntimeError("outer AMP must be disabled")
            _ACTIVE=self
            self.attention_cls.compute_qkv=qkv
            self.attention_cls.compute_attention_heads=self.attention_private
            self.inference_module.InferenceEngineCacheKV.prepare=self.private_prepare_descriptor
            return self
        except BaseException:
            if hasattr(self,"qkv_private") and self.attention_cls.compute_qkv is self.qkv_private:
                self.attention_cls.compute_qkv=self.qkv_original
            if hasattr(self,"attention_private") and inspect.getattr_static(self.attention_cls,"compute_attention_heads") is self.attention_private:
                self.attention_cls.compute_attention_heads=self.attention_descriptor
            if hasattr(self,"private_prepare_descriptor") and inspect.getattr_static(self.inference_module.InferenceEngineCacheKV,"prepare") is self.private_prepare_descriptor:
                self.inference_module.InferenceEngineCacheKV.prepare=self.prepare_descriptor
            _ACTIVE=None;self.owner=None;_LOCK.release();raise

    def __exit__(self,exc_type,exc,tb):
        global _ACTIVE
        conflict=(self.attention_cls.compute_qkv is not self.qkv_private
            or inspect.getattr_static(self.attention_cls,"compute_attention_heads") is not self.attention_private
            or inspect.getattr_static(self.inference_module.InferenceEngineCacheKV,"prepare") is not self.private_prepare_descriptor)
        try:
            self.attention_cls.compute_qkv=self.qkv_original
            self.attention_cls.compute_attention_heads=self.attention_descriptor
            self.inference_module.InferenceEngineCacheKV.prepare=self.prepare_descriptor
            self.restored=(self.attention_cls.compute_qkv is self.qkv_original
                and inspect.getattr_static(self.attention_cls,"compute_attention_heads") is self.attention_descriptor
                and inspect.getattr_static(self.inference_module.InferenceEngineCacheKV,"prepare") is self.prepare_descriptor
                and self._flags()==self.initial_flags)
            require_sources()
            if conflict or not self.restored:raise RuntimeError("precision alias/flag restoration failed")
        finally:_ACTIVE=None;self.owner=None;_LOCK.release()
        return False

    def report(self):
        return {**self.descriptor,"private_clones":copy.deepcopy(self.clones),"query_views":self.query_views,
                "counters":dict(self.counters),"witnesses":[{"count":count,**json.loads(key)} for key,count in self.hist.items()],
                "witness_examples":copy.deepcopy(self.examples),"aliases_and_flags_restored":self.restored,
                "computed_kv_not_inferred_from_storage":True}


class PrecisionResident(cache.ResidentStageExecutor):
    def prepare(self,theta_context,obs_context,stage_index):
        session=_session();session.stage=stage_index
        result=session.resident_prepare(self,theta_context,obs_context,stage_index)
        self.metadata["precision_experiment"]=metadata(session.mode)
        self.metadata.setdefault("frozen_constructor_configuration",copy.deepcopy(self.metadata.get("config",{})))
        self.metadata.update(schema=SCHEMA+".resident",engine_id="private_precision_resident_v1",
                             module_sha256=source_sha256(),qualified_fp32=False)
        self.metadata["config"]={"weights_dtype":"torch.float32","forward_policy":session.mode,
            "actual_use_autocast":self.regressor.use_autocast_,
            "actual_forced_inference_dtype":str(self.regressor.forced_inference_dtype_),
            "postprocessing_dtype":"torch.float32","query_batch_size":self.query_batch_size}
        return result


def _hybrid_counter_metadata(mode):
    frozen=hybrid.execution_metadata({10:[5]})
    return {**frozen,"schema":SCHEMA+".hybrid_methods","engine_id":"private_precision_hybrid_methods_v1",
            "module_sha256":source_sha256(),"qualified_fp32":False,
            "forward_precision_policy":mode,"prediction_math_dtype":"policy-dependent forward; require actual QKV witnesses",
            "postprocessing_math_dtype":"torch.float32",
            "frozen_hybrid_policy_provenance":frozen}


class PrecisionStageExecutor(hybrid.HybridStageExecutor):
    """Same original hybrid methods with one private precision-aware base."""
    def __init__(self,checkpoint,*,cdf_descriptor,device="cuda",query_batch_size=1024):
        session=_session();self.session=session
        self.fallback_map=hybrid.canonical_fallback_map({10:[5]})
        self.base=PrecisionResident(checkpoint,cdf_descriptor=cdf_descriptor,device=device,query_batch_size=query_batch_size)
        self.cached_component=mixture.ComponentMixtureExecutor(self.base)
        self.uncached_regressor=None;self._busy=Lock();self._prepared=None
        self.hybrid_counters={**_hybrid_counter_metadata(session.mode),"uncached_fit_count":0,"uncached_predict_count":0,
            "uncached_query_rows":0,"resident_component_predict_count":0,"original_preprocessing_lists_restored":True,
            "original_translator_alias_restored":True,"stages":[]}

    def predict_component(self,member_index,query,*,capture_member=False,with_log_prob=False):
        if _session() is not self.session:raise RuntimeError("precision session rebinding forbidden")
        with self.session.forward_scope("query",self.stage_index,member_index):
            result=super().predict_component(member_index,query,capture_member=capture_member,with_log_prob=with_log_prob)
        if result["logits"].dtype!=self.base._torch.float32:raise RuntimeError("FP32 postprocessing output required")
        return result


class UncachedPrecisionStage:
    """Original all8 preprocessing fit; selected original member per query."""
    def __init__(self,checkpoint,*,cdf_descriptor,stage_index,theta,obs):
        session=_session();torch=session.torch
        self.session=session;self.stage_index=stage_index;self.cdf_descriptor=cdf_descriptor
        self.query_batch_size=1024
        self.base=SimpleNamespace(_torch=torch,device="cuda",metadata={"model_rng_isolation_count":0},
                                  _sdk_regressor=session.regressor_module,cdf_descriptor=cdf_descriptor)
        self.base._model_rng=MethodType(cache.ResidentStageExecutor._model_rng,self.base)
        from .inference import frozen_regressor_settings
        settings=frozen_regressor_settings(checkpoint);settings["inference_precision"]=_inference_setting()
        self.regressor=session.regressor_module.TabPFNRegressor(**settings)
        joint=torch.cat((torch.from_numpy(obs.copy()),torch.from_numpy(theta.copy())),dim=1)
        with self.base._model_rng():self.regressor.fit(joint[:,:33+stage_index],joint[:,33+stage_index])
        self.fp32_regressor=None
        if stage_index==10:
            settings["inference_precision"]=torch.float32
            self.fp32_regressor=session.regressor_module.TabPFNRegressor(**settings)
            with self.base._model_rng():self.fp32_regressor.fit(joint[:,:33+stage_index],joint[:,33+stage_index])
        self.hybrid_counters={**_hybrid_counter_metadata(session.mode),"uncached_predict_count":0,"uncached_query_rows":0,
                              "original_preprocessing_lists_restored":True,"original_translator_alias_restored":True}
        self.uncached_regressor=None

    def predict_component(self,member_index,query,*,capture_member=False):
        if _session() is not self.session:raise RuntimeError("precision session rebinding forbidden")
        self.uncached_regressor=self.fp32_regressor if self.stage_index==10 and member_index==5 else self.regressor
        with self.session.forward_scope("query",self.stage_index,member_index):
            value=hybrid.HybridStageExecutor._predict_uncached(self,member_index,query,capture_member)
        if value["logits"].dtype!=self.session.torch.float32:raise RuntimeError("uncached postprocessing did not remain FP32")
        return value

    def close(self):
        self.regressor=None;self.fp32_regressor=None;self.uncached_regressor=None


def capture_members(engine,query):
    """All8 actual component outputs; FP32 raw-mean reconstruction is diagnostic."""
    torch=__import__("torch");predictions=[engine.predict_component(i,query,capture_member=True) for i in range(8)]
    members=[p["member"] for p in predictions]
    if any(m["translated_probabilities"].dtype!=torch.float32 for m in members):raise RuntimeError("canonical probabilities must remain FP32")
    return {"criterion":predictions[0]["criterion"],"logits":torch.stack([m["translated_probabilities"] for m in members]).mean(0).log(),
            "members":members,"component_predictions":predictions,"full_mixture_is_diagnostic_reconstruction":True}


def prefill_witness_passed(report,stage):
    """Prove the approximate prefill really computed low projections, not a cast."""
    mode=report["mode"]
    expected="torch.bfloat16" if mode=="amp_bf16_prefill_query_v1" else "torch.float16"
    rows=report["witnesses"];result=[]
    for member in range(8):
        selected=[r for r in rows if r["phase"]=="prefill" and r["stage"]==stage and r["member"]==member and r["cache_kv"]]
        strict=stage==10 and member==5
        if mode.startswith("amp_") and not strict:
            passed=bool(selected) and any(expected in [r["projection_dtypes"][key] for key in ("k","v","kv","qkv")] and r["autocast_enabled"] for r in selected)
        else:
            passed=bool(selected) and all(not r["autocast_enabled"] and all(value in (None,"torch.float32") for value in r["projection_dtypes"].values()) for r in selected)
        if mode=="tf32_high_forward_v1" and not strict:
            passed=passed and all(r.get("allow_tf32") is True and r.get("matmul_precision")=="high" for r in selected)
        elif strict or mode==CONTROL_MODE:
            passed=passed and all(r.get("allow_tf32") is False and r.get("matmul_precision")=="highest" for r in selected)
        if mode.startswith("amp_") and not strict:
            passed=passed and all(r.get("autocast_dtype")==expected and r["autocast_enabled"] for r in selected)
        storage={value for r in selected for value in r.get("supplied_cache_dtypes",[]) if value is not None}
        allowed={"torch.float32"} if strict or not mode.startswith("amp_") else {"torch.float32",expected}
        passed=passed and bool(storage) and storage.issubset(allowed)
        result.append({"member":member,"passed":passed,"prefill_event_count":sum(r["count"] for r in selected),
                       "actual_stored_cache_dtypes":sorted(storage)})
    return {"passed":all(r["passed"] for r in result),"members":result,"storage_dtype_is_separate_observation":True}


def query_witness_passed(report,stage):
    mode=report["mode"];expected="torch.bfloat16" if mode=="amp_bf16_prefill_query_v1" else "torch.float16"
    results=[]
    for member in range(8):
        rows=[row for row in report["witnesses"] if row["phase"]=="query" and row["stage"]==stage and row["member"]==member]
        strict=stage==10 and member==5
        if mode.startswith("amp_") and not strict:
            passed=bool(rows) and all(row["autocast_enabled"] and row["autocast_dtype"]==expected for row in rows)
            passed=passed and any(expected in row["projection_dtypes"].values() for row in rows)
        else:
            passed=bool(rows) and all(not row["autocast_enabled"] for row in rows)
            tf32=mode=="tf32_high_forward_v1" and not strict
            passed=passed and all(row["allow_tf32"] is tf32 and row["matmul_precision"]==("high" if tf32 else "highest") for row in rows)
        results.append({"member":member,"passed":passed,"query_event_count":sum(row["count"] for row in rows)})
    return {"passed":all(row["passed"] for row in results),"members":results}
