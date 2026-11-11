"""Explicit FP32 one-stage resident ensemble/KV execution.

Built with PriorLabs-TabPFN. This opt-in executor does not edit SDK files or
activate CDF compatibility. The caller owns one finite_cdf_compatibility context.
No posterior sampler or support-rejection implementation is provided here.
"""
from __future__ import annotations
import ast
import __future__
from contextlib import contextmanager
import gc
import hashlib
from importlib import import_module,metadata
import inspect
import marshal
from pathlib import Path
import pickle
import textwrap
import time
from types import MethodType,MappingProxyType

ENGINE_ID="resident_stage_kv_fp32_v1"
ENGINE_VERSION=1
SCHEMA="posteriorforge.execution.resident_stage.v1"
PINNED_SDK_SHA256=MappingProxyType({
 "tabpfn/inference.py":"62470bbfdd4879dd52db86ea7a6e53ae41fbcc22379bfb2866ad906adc71a0bd",
 "tabpfn/model/multi_head_attention.py":"512220bbf253fb856003f7084976a478753644c9963d563ee678eb3be2c28f35",
 "tabpfn/model/layer.py":"b7c230f917b3e346e1d9cc83eb9e06fd939dbbd3fe3b2c5f7a8e4b7505142754",
 "tabpfn/model/transformer.py":"53471f33e5adff20af64a4730b9caba49cf6490f4e2119397b81ef6b3c0f9751",
 "tabpfn/model/encoders.py":"e16e03c320c69dcd2e9823925d59fa20a0c52008d8cc77ca961c919c7221213e",
 "tabpfn/base.py":"b80fbf83b8f8a7eb69ae77c0a7f0b397a43e32c1c29b8392a0ff9ee525218837",
 "tabpfn/preprocessing.py":"fba9df553f3f7d190972a531c5627f09b88e8b8f537d298df7083f7dd972843d",
 "tabpfn/utils.py":"a0521d68cd2880e041a49ac5acee42a2b8d1297e246e3f7a9c477e49f5e97fc1",
 "tabpfn/regressor.py":"bbb7d6dec265b18a30a54ff28719c957bc05d4164f693f76d998380132d2c46e",
})
CONFIG_INVARIANTS=MappingProxyType({
 "inference_precision":"torch.float32","n_estimators":8,"fit_mode":"fit_with_cache",
 "normalize_on_train_only":True,"fixed_layer_count":True,
 "query_batch_axis":1,"stage_count":21,"observation_features":33,
 "missing_query_features":"NaN passed unchanged to original SDK imputation; infinity rejected",
 "model_rng_isolation":"torch.random.fork_rng CPU and selected CUDA device",
})


class CacheExecutionError(RuntimeError):pass


def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def _code_sha(function):return hashlib.sha256(marshal.dumps(function.__code__)).hexdigest()
def execution_source_sha256():return _sha(__file__)


def execution_metadata():
    """CPU-safe immutable identity; does not import Torch or TabPFN."""
    return {"schema":SCHEMA,"engine_id":ENGINE_ID,"engine_version":ENGINE_VERSION,
            "module_sha256":execution_source_sha256(),"SDKsourcehashes":dict(PINNED_SDK_SHA256),
            "config":dict(CONFIG_INVARIANTS),"attribution":"Built with PriorLabs-TabPFN"}


def _load_sdk():
    torch=import_module("torch");regressor=import_module("tabpfn.regressor")
    inference=import_module("tabpfn.inference")
    if metadata.version("tabpfn")!="2.0.9":raise CacheExecutionError("exact TabPFN2.0.9 required")
    root=Path(regressor.__file__).parent
    hashes={name:_sha(root/Path(name).relative_to("tabpfn")) for name in PINNED_SDK_SHA256}
    if hashes!=dict(PINNED_SDK_SHA256):raise CacheExecutionError("SDK source hash mismatch")
    return torch,regressor,inference


def _require_active_cdf(regressor_module,descriptor):
    if (descriptor.get("variant")!="cdf_arithmetic_v1"
        or descriptor.get("scope")!="finite_ensemble_cdf_parentheses_only"
        or descriptor.get("source_guard_passed") is not True
        or descriptor.get("original_alias_restored") is not False):
        raise CacheExecutionError("caller-owned active CDF compatibility descriptor required")
    active=regressor_module.translate_probs_across_borders
    if _code_sha(active)!=descriptor.get("active_translator_code_sha256"):
        raise CacheExecutionError("active CDF translator fingerprint mismatch")
    helper=active.__globals__.get("_cdf")
    if helper is None or not any(cell.cell_contents is descriptor for cell in (helper.__closure__ or ())):
        raise CacheExecutionError("active CDF descriptor does not own translator helper")
    return active


def _resident_iter(original):
    """Retain stock iter_outputs arithmetic; remove its one final CPU offload."""
    tree=ast.parse(textwrap.dedent(inspect.getsource(original)))
    expected=ast.dump(ast.parse("model.cpu()",mode="eval").body,include_attributes=False)
    matches=[]
    for node in ast.walk(tree):
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):
            for index,statement in enumerate(node.body):
                if (isinstance(statement,ast.Assign)
                    and any(isinstance(t,ast.Name) and t.id=="model" for t in statement.targets)
                    and ast.dump(statement.value,include_attributes=False)==expected):
                    matches.append((node.body,index))
        if isinstance(node,(ast.For,ast.With,ast.If)):
            for index,statement in enumerate(node.body):
                if (isinstance(statement,ast.Assign)
                    and any(isinstance(t,ast.Name) and t.id=="model" for t in statement.targets)
                    and ast.dump(statement.value,include_attributes=False)==expected):
                    matches.append((node.body,index))
    if len(matches)!=1:raise CacheExecutionError("stock cached iter_outputs offload expression mismatch")
    body,index=matches[0];del body[index]
    ast.fix_missing_locations(tree)
    namespace=dict(original.__globals__)
    exec(compile(tree,original.__code__.co_filename,"exec",
                 flags=original.__code__.co_flags & __future__.annotations.compiler_flag,
                 dont_inherit=True),namespace)
    return namespace[original.__name__]


def _numpy_training(theta,obs):
    import numpy as np
    theta=np.asarray(theta,dtype=np.float32);obs=np.asarray(obs,dtype=np.float32)
    if theta.shape!=(4096,21) or obs.shape!=(4096,33) or not np.isfinite(theta).all() or not np.isfinite(obs).all():
        raise ValueError("finite unchanged4096x21 theta and4096x33 raw observations required")
    return theta,obs


def _numpy_query(query,stage,max_rows):
    import numpy as np
    if hasattr(query,"detach"):query=query.detach().cpu().numpy()
    value=np.asarray(query,dtype=np.float32)
    if value.ndim!=2 or value.shape[1]!=33+stage or not 1<=len(value)<=max_rows or np.isinf(value).any():
        raise ValueError("correctly sized stage queries without infinity within explicit query_batch_size required")
    return value


def _configuration_identity(config):
    return {"type":type(config).__module__+"."+type(config).__qualname__,
            "sha256":hashlib.sha256(pickle.dumps(config,protocol=5)).hexdigest(),
            "repr":repr(config)}


def capture_corrected_prediction(regressor,query,*,cdf_descriptor):
    """Qualification-only CPU member captures; return every original output."""
    module=import_module("tabpfn.regressor")
    active=_require_active_cdf(module,cdf_descriptor);captures=[]
    executor=regressor.executor_
    configs=getattr(executor,"configs",getattr(executor,"ensemble_configs",None))
    if configs is None:raise CacheExecutionError("unknown stock ensemble configuration storage")
    def record(logits,*,frm,to):
        output=active(logits,frm=frm,to=to)
        index=len(captures)
        captures.append({"ensemble_index":index,"raw_logits":logits.detach().cpu().clone(),
            "from_borders":frm.detach().cpu().clone(),"to_borders":to.detach().cpu().clone(),
            "translated_probabilities":output.detach().cpu().clone(),
            "configuration_identity":_configuration_identity(configs[index])})
        return output
    module.translate_probs_across_borders=record
    try:
        prediction=regressor.predict(query,output_type="full",quantiles=[])
    finally:
        conflicting=module.translate_probs_across_borders is not record
        module.translate_probs_across_borders=active
        if conflicting:raise CacheExecutionError("conflicting member-capture alias change")
    if len(captures)!=8:raise CacheExecutionError("expected all8member captures")
    prediction["members"]=captures
    return prediction


def _tensor_records(model,torch):
    rows=[];seen_attributes=set()
    for category,iterator in (("parameter",model.named_parameters()),("buffer",model.named_buffers())):
        for name,tensor in iterator:
            rows.append((category,name,tensor));seen_attributes.add(id(tensor))
    for module_name,module in model.named_modules():
        for name,value in vars(module).items():
            if torch.is_tensor(value) and id(value) not in seen_attributes:
                rows.append(("plain_tensor",module_name+"."+name,value));seen_attributes.add(id(value))
    return rows


def _cache_assumptions(models,torch):
    result=[]
    for index,model in enumerate(models):
        normalizers=[];layers=[]
        if model.training:raise CacheExecutionError("cached model must be in eval mode")
        for name,module in model.named_modules():
            if type(module).__name__=="InputNormalizationEncoderStep":
                normalizers.append({"module":name,"normalize_on_train_only":bool(module.normalize_on_train_only)})
                if not module.normalize_on_train_only:raise CacheExecutionError("query-dependent normalization forbids this cache engine")
            if type(module).__name__=="LayerStack":
                layers.append({"module":name,"min":module.min_num_layers_layer_dropout,"max":module.num_layers})
                if module.min_num_layers_layer_dropout!=module.num_layers:raise CacheExecutionError("stochastic layer count forbids this cache engine")
        kv=0
        for _,name,tensor in _tensor_records(model,torch):
            if tensor.is_floating_point() and tensor.dtype!=torch.float32:
                raise CacheExecutionError("FP32-only prepared tensor invariant violated: "+name)
            if any(key in name for key in ("_k_cache","_v_cache","_kv_cache")) and tensor.numel():
                kv+=1
                if tensor.device.type!="cuda":raise CacheExecutionError("KV cache is not residentCUDA")
        if not normalizers or not layers or kv==0:raise CacheExecutionError("required normalization/layer/KV evidence missing")
        result.append({"ensemble_index":index,"normalizers":normalizers,"layer_stacks":layers,
                       "nonempty_kv_tensors":kv,"feature_positional_embedding":model.feature_positional_embedding,
                       "model_seed":model.seed})
    return result


class ResidentStageExecutor:
    """One stage and eight ensemble caches resident at a time.

    The caller chunks proposal waves explicitly. predict rejects larger chunks
    rather than silently regrouping queries. Model operations restore Torch CPU
    and selected CUDA RNG states; the caller owns posterior sampling streams.
    """
    def __init__(self,checkpoint,*,cdf_descriptor,device="cuda",query_batch_size=1024):
        if type(query_batch_size) is not int or query_batch_size<1 or device not in ("cuda","cuda:0"):
            raise ValueError("oneCUDAdevice and a positive explicit query batch size required")
        from .inference import CHECKPOINT_SHA256,frozen_regressor_settings
        if _sha(checkpoint)!=CHECKPOINT_SHA256:raise CacheExecutionError("immutable checkpoint hash mismatch")
        self.checkpoint=str(Path(checkpoint).resolve());self.device=device;self.query_batch_size=query_batch_size
        self.cdf_descriptor=cdf_descriptor;self.regressor=None;self.stage_index=None
        self._torch,self._sdk_regressor,self._sdk_inference=_load_sdk()
        _require_active_cdf(self._sdk_regressor,cdf_descriptor)
        if not self._torch.cuda.is_available():raise CacheExecutionError("reserved CUDA execution required")
        self.settings=frozen_regressor_settings(self.checkpoint);self.settings["fit_mode"]="fit_with_cache"
        self.metadata={**execution_metadata(),"config":{**dict(CONFIG_INVARIANTS),
            "query_batch_size":query_batch_size,"device":device},"prepare_count":0,"predict_count":0,
            "query_rows":0,"model_rng_isolation_count":0,"stages":[]}
        self._prepared=None

    @contextmanager
    def _model_rng(self):
        torch=self._torch
        index=torch.device(self.device).index
        if index is None:index=torch.cuda.current_device()
        with torch.random.fork_rng(devices=[index],enabled=True):
            yield
        self.metadata["model_rng_isolation_count"]+=1

    def prepare(self,theta_context,obs_context,stage_index):
        if type(stage_index) is not int or not 0<=stage_index<21:raise ValueError("stage index0..20 required")
        theta,obs=_numpy_training(theta_context,obs_context)
        self.close();_require_active_cdf(self._sdk_regressor,self.cdf_descriptor)
        torch=self._torch;settings=dict(self.settings);settings["inference_precision"]=torch.float32
        regressor=self._sdk_regressor.TabPFNRegressor(**settings)
        joint=torch.cat((torch.from_numpy(obs.copy()),torch.from_numpy(theta.copy())),dim=1)
        torch.cuda.synchronize();started=time.perf_counter()
        try:
            with self._model_rng():
                regressor.fit(joint[:,:33+stage_index],joint[:,33+stage_index])
            torch.cuda.synchronize();fit_end=time.perf_counter()
            executor=regressor.executor_
            if type(executor) is not self._sdk_inference.InferenceEngineCacheKV or len(executor.models)!=8:
                raise CacheExecutionError("exact stock8member CacheKV engine required")
            self.regressor=regressor;self.stage_index=stage_index
            stock_cache_telemetry=self.inspect_cache()
            original=type(executor).iter_outputs
            if Path(inspect.getfile(original)).resolve()!=Path(self._sdk_inference.__file__).resolve():
                raise CacheExecutionError("cached engine function identity mismatch")
            for model in executor.models:model.to(self.device)
            torch.cuda.synchronize();resident_end=time.perf_counter()
            assumptions=_cache_assumptions(executor.models,torch)
            replacement=_resident_iter(original)
            executor.iter_outputs=MethodType(replacement,executor)
            self.regressor=regressor;self.stage_index=stage_index
            self._prepared={"stage_index":stage_index,"predictor_count":33+stage_index,
                "fit_stock_cache_seconds":fit_end-started,"resident_move_seconds":resident_end-fit_end,
                "original_iter_code_sha256":_code_sha(original),"resident_iter_code_sha256":_code_sha(replacement),
                "removed_operation":"model = model.cpu() after each member prediction",
                "assumptions":assumptions,"stock_prepared_cachetelemetry":stock_cache_telemetry,
                "configuration_identities":[_configuration_identity(config) for config in executor.configs]}
            self.metadata["prepare_count"]+=1;self.metadata["stages"].append(self._prepared)
            self._prepared["cachetelemetry"]=self.inspect_cache()
            return self._prepared
        except BaseException:
            if hasattr(regressor,"executor_") and hasattr(regressor.executor_,"models"):
                regressor.executor_.models.clear()
            self.regressor=None;self.stage_index=None;gc.collect()
            raise

    def predict(self,query,*,output_type="full",capture_members=False):
        if self.regressor is None:raise CacheExecutionError("prepare one stage first")
        if output_type!="full":raise ValueError("this executor preserves the original fullprediction API")
        _require_active_cdf(self._sdk_regressor,self.cdf_descriptor)
        value=_numpy_query(query,self.stage_index,self.query_batch_size)
        with self._model_rng():
            if capture_members:
                prediction=capture_corrected_prediction(self.regressor,value,cdf_descriptor=self.cdf_descriptor)
            else:prediction=self.regressor.predict(value,output_type="full",quantiles=[])
        self.metadata["predict_count"]+=1;self.metadata["query_rows"]+=len(value)
        if capture_members:self._prepared["after_predict_cachetelemetry"]=self.inspect_cache()
        return prediction

    def inspect_cache(self):
        if self.regressor is None:return {"prepared":False,"resident_ensemble_count":0,"unique_storage_bytes":0}
        torch=self._torch;records=[];storage={}
        for index,model in enumerate(self.regressor.executor_.models):
            for category,name,tensor in _tensor_records(model,torch):
                if tensor.numel()==0:continue
                pointer=(str(tensor.device),tensor.untyped_storage().data_ptr())
                storage[pointer]=tensor.untyped_storage().nbytes()
                records.append({"ensemble_index":index,"category":category,"name":name,"shape":list(tensor.shape),
                    "dtype":str(tensor.dtype),"device":str(tensor.device),"logical_bytes":tensor.numel()*tensor.element_size()})
        return {"prepared":True,"stage_index":self.stage_index,"resident_ensemble_count":len(self.regressor.executor_.models),
            "unique_storage_bytes":sum(storage.values()),
            "unique_cuda_storage_bytes":sum(size for (device,_),size in storage.items() if device.startswith("cuda")),
            "tensor_records":records,"cuda_allocated_bytes":torch.cuda.memory_allocated(),
            "cuda_reserved_bytes":torch.cuda.memory_reserved()}

    def close(self):
        if self.regressor is not None:
            executor=getattr(self.regressor,"executor_",None)
            if executor is not None and hasattr(executor,"models"):executor.models.clear()
            self.regressor.executor_=None
        self.regressor=None;self.stage_index=None;self._prepared=None
        gc.collect()

    def __enter__(self):return self
    def __exit__(self,*args):self.close()
