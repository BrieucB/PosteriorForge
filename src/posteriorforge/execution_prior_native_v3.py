"""FP32 0199 neural execution with a private SDK-prefix/FP64 probability seam.

Built with PriorLabs-TabPFN. Existing classes/functions/globals are never patched.
The private predict prefix retains the original validation, fitted transforms,
forward iterator, temperature and cancellation; v3 begins before translation.
"""
from contextlib import ExitStack,contextmanager
from contextvars import ContextVar
import ast
import copy
import __future__
import hashlib
import inspect
import textwrap
import json
from pathlib import Path
from threading import Lock,get_ident
from types import MethodType
from importlib import import_module
from . import execution_prior_native_v2 as neural
from . import execution_probability_v3 as probability

SCHEMA='posteriorforge.execution.prior_native_probability64.v3'
NEURAL_SHA='0199b16e9cc34203fa0d5645d161742fe9ed244633d7beee047fbcbf2528661c'
SDK_REGRESSOR_SHA='bbb7d6dec265b18a30a54ff28719c957bc05d4164f693f76d998380132d2c46e'
PROBABILITY_SHA='7fea13270c38c51eed0eb65cd5026de94e3f7e24cc4734e8906e882fb0bc0494'
NATIVE_STAGES=(6,10,14)
CONTROL_MODE=neural.CONTROL_MODE
cached=neural.cached
cache=neural.cache
codec=neural.codec
stage_policy=neural.stage_policy
mask_training=neural.mask_training
mask_query=neural.mask_query
training_bindings=neural.training_bindings
route=neural.route
_OWNER=ContextVar('posteriorforge_prior_v3_session',default=None)
_PREFIX=None
_PREFIX_DESCRIPTOR=None
_PREFIX_ORIGINAL=None
_PREFIX_SEALS=None


def require(value,message):
    if not value:raise RuntimeError(message)

def source_sha256():return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

def require_sources(*,allow_unfrozen_probability=False):
    require(source_sha256()==_IMPORTED_SHA,'v3 backend changed after import')
    require(neural.source_sha256()==NEURAL_SHA,'0199 neural source changed');neural.require_sources()
    probability.require_sources()
    require((isinstance(PROBABILITY_SHA,str) and probability.source_sha256()==PROBABILITY_SHA)
            or (allow_unfrozen_probability and PROBABILITY_SHA is None),'v3 probability dependency not frozen')
    require(NATIVE_STAGES==(6,10,14),'v3 neural routing changed')
    for name,value,code in _PUBLIC:
        require(globals().get(name) is value and (code is None or value.__code__ is code),'v3 export changed: '+name)
    for owner,name,function,code,namespace,wrapped,wcode in _METHOD_SEALS:
        actual=getattr(owner,name)
        if isinstance(actual,property):actual=actual.fget
        require(actual is function and function.__code__ is code and function.__globals__ is namespace,'v3 body/namespace changed: '+name)
        require(getattr(function,'__wrapped__',None) is wrapped and (wrapped is None or wrapped.__code__ is wcode),'v3 wrapped body changed: '+name)
    for name,value in _CONFIG_SEALS.items():
        require(globals().get(name) is value,'v3 config/dependency changed: '+name)
    for name,value in _GLOBAL_BINDINGS.items():
        require(globals().get(name) is value,'v3 referenced global changed: '+name)
    return True


def policy_metadata():
    require_sources()
    return {'schema':SCHEMA+'.policy','module_sha256':source_sha256(),'neural_source_sha256':NEURAL_SHA,
            'probability_source_sha256':PROBABILITY_SHA,'native_all8_stages':list(NATIVE_STAGES),'cached_stage_count':18,
            'neural_policy':neural.policy_metadata(),'probability_policy':probability.metadata(),
            'old_numerical_pass_inherited':False,'NN_forward_order_mask_settings_changed':False,
            'new_numerical_probability_candidate':True}


def execution_metadata():return {**policy_metadata(),'FP16_activated':False,'full_API_identity_claimed':False,
                                  'forward_dtype':'torch.float32','probability_dtype':'torch.float64','stage_storage_dtype':'numpy.float32'}


def _session():
    session=_OWNER.get();require(session is not None and session.active and session.owner_thread==get_ident(),'active same-thread v3 Session required')
    probability.active_descriptor(session.probability_descriptor)
    return session


def _prefix():
    global _PREFIX,_PREFIX_DESCRIPTOR,_PREFIX_ORIGINAL,_PREFIX_SEALS
    module=import_module('tabpfn.regressor')
    require(hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()==SDK_REGRESSOR_SHA,'exact SDK regressor source required')
    original=inspect.unwrap(module.TabPFNRegressor.predict)
    if _PREFIX is None:
        source=textwrap.dedent(inspect.getsource(original));tree=ast.parse(source);root=tree.body[0]
        cut=[i for i,s in enumerate(root.body) if isinstance(s,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='transformed_logits' for t in s.targets)]
        require(len(cut)==1,'exact SDK probability cut required');index=cut[0]
        private=copy.deepcopy(tree);node=private.body[0];tail=copy.deepcopy(node.body[index:]);decorators=copy.deepcopy(node.decorator_list)
        node.decorator_list=[]
        node.body=node.body[:index]+[ast.Return(value=ast.Tuple(elts=[ast.Name(id='outputs',ctx=ast.Load()),ast.Name(id='borders',ctx=ast.Load()),
            ast.Attribute(value=ast.Attribute(value=ast.Name(id='self',ctx=ast.Load()),attr='bardist_',ctx=ast.Load()),attr='borders',ctx=ast.Load()),
            ast.Attribute(value=ast.Attribute(value=ast.Name(id='self',ctx=ast.Load()),attr='renormalized_criterion_',ctx=ast.Load()),attr='borders',ctx=ast.Load())],ctx=ast.Load()))]
        reverse=copy.deepcopy(private);reverse.body[0].decorator_list=decorators;reverse.body[0].body=reverse.body[0].body[:-1]+tail
        require(ast.dump(reverse,include_attributes=False)==ast.dump(tree,include_attributes=False),'private SDK prefix has undeclared changes')
        namespace=dict(original.__globals__);ast.fix_missing_locations(private)
        exec(compile(private,'<posteriorforge-v3-original-SDK-predict-prefix>','exec',flags=__future__.annotations.compiler_flag,dont_inherit=True),namespace)
        _PREFIX=namespace['predict'];_PREFIX_ORIGINAL=original
        _PREFIX_SEALS=(_PREFIX,_PREFIX.__code__,{k:namespace[k] for k in _PREFIX.__code__.co_names if k in namespace and k!='predict'})
        _PREFIX_DESCRIPTOR={'SDK_regressor_sha256':SDK_REGRESSOR_SHA,'original_function_source_sha256':hashlib.sha256(source.encode()).hexdigest(),
                            'reverse_AST_exact':True,'cut':'immediately before transformed_logits list; original prefix including temperature/cancel/fitted border transform retained',
                            'original_decorator':'inference_mode applied by v3 caller around unchanged prefix','old_predict_or_global_alias_modified':False}
    require(original is _PREFIX_ORIGINAL and _PREFIX is _PREFIX_SEALS[0] and _PREFIX.__code__ is _PREFIX_SEALS[1],'SDK/prefix callable changed')
    require(all(_PREFIX.__globals__.get(k) is v for k,v in _PREFIX_SEALS[2].items()),'private prefix global binding changed')
    return _PREFIX


class PriorNativeSession:
    def __init__(self,mode=CONTROL_MODE,*,probability_descriptor=None):
        require(mode==CONTROL_MODE,'v3 is FP32 neural only');require_sources()
        self.supplied_descriptor=probability_descriptor;self.probability_descriptor=None
        self.active=False;self.owner_thread=None;self.neural=None;self.legacy_guard=None;self._stack=None;self.engines=[]
    def __enter__(self):
        require(_OWNER.get() is None,'nested v3 neural Session unsupported');self._stack=ExitStack()
        try:
            self.probability_descriptor=(self._stack.enter_context(probability.production_scope()) if self.supplied_descriptor is None
                                         else probability.active_descriptor(self.supplied_descriptor))
            from .compat_cdf import finite_cdf_compatibility
            self.legacy_guard=self._stack.enter_context(finite_cdf_compatibility())
            self.neural=self._stack.enter_context(neural.PriorNativeSession())
            self.active=True;self.owner_thread=get_ident();self._token=_OWNER.set(self)
            return self
        except BaseException as primary:
            try:self._stack.close()
            except BaseException as cleanup:raise primary from cleanup
            raise
    def report(self):
        report=self.neural.report() if self.neural is not None else {}
        return {**report,'v3_probability':copy.deepcopy(self.probability_descriptor),
                'legacy_preparation_guard':copy.deepcopy(self.legacy_guard),
                'legacy_guard_is_probability_execution':False,'v3_backend_sha256':source_sha256()}
    def __exit__(self,kind,value,traceback):
        try:
            try:return self._stack.__exit__(kind,value,traceback)
            except BaseException as cleanup:
                if value is not None and cleanup is not value:raise value.with_traceback(traceback) from cleanup
                raise
        finally:
            self.active=False
            if hasattr(self,'_token'):_OWNER.reset(self._token)
            if self.probability_descriptor is not None and self.legacy_guard is not None:
                self.probability_descriptor['legacy_preparation_guards'].append(copy.deepcopy(self.legacy_guard))


@contextmanager
def _original_query_scope(session,stage,member):
    require(session.neural.query_views is True,'unexpected original query-view scope')
    try:
        session.neural.query_views=False
        with session.neural.forward_scope('query',stage,member):yield
    finally:session.neural.query_views=True


@contextmanager
def _cached_member(executor,member,report):
    names=neural.cached.ss15.mixture._MEMBER_FIELDS;original={}
    try:
        for name in names:
            value=getattr(executor,name);require(len(value)==8,'all8 cached lists required');original[name]=value
        report['cached_member_lists_restored']=False
        for name,value in original.items():setattr(executor,name,[value[member]])
        yield
    finally:
        for name,value in original.items():setattr(executor,name,value)
        report['cached_member_lists_restored']=len(original)==len(names) and all(getattr(executor,k) is v for k,v in original.items())


def _component(regressor,query,stage,member,session,base,counters,*,original,capture_member):
    import torch
    require_sources();require(session is _session(),'wrong v3 Session')
    value=neural.cached._numpy_query(query,stage,1024)
    require(type(member) is int and 0<=member<8,'member0..7 required')
    require(not regressor.use_autocast_ and regressor.forced_inference_dtype_==torch.float32
            and regressor.softmax_temperature==.9 and regressor.average_before_softmax is False,'unchanged FP32/temperature/averaging policy required')
    executor=regressor.executor_
    model=regressor.model_ if original else executor.models[member]
    require(model.training is False and not any(m.training for m in model.modules()),'all selected neural modules must remain in eval mode')
    rng_before=base.metadata['model_rng_isolation_count']
    configs=getattr(executor,'ensemble_configs',getattr(executor,'configs',None));configuration=cache._configuration_identity(configs[member])
    selection=(neural.cached.ss15.hybrid.selected_preprocessing_member(executor,member,counters) if original
               else _cached_member(executor,member,counters))
    scope=(_original_query_scope(session,stage,member) if original else session.neural.forward_scope('query',stage,member))
    with selection,scope,base._model_rng(),torch.inference_mode():
        outputs,borders,canonical,api=_prefix()(regressor,value,output_type='full',quantiles=[])
    counters['prefix_query_count']+=1;counters['prefix_query_rows']+=len(value)
    require(base.metadata['model_rng_isolation_count']==rng_before+1,'one actual model RNG isolation scope required')
    require(model.training is False and not any(m.training for m in model.modules()),'selected neural eval flags changed')
    require(len(outputs)==len(borders)==1,'one actual selected forward required')
    adjusted=outputs[0];require(adjusted.dtype==torch.float32,'neural prefix must return adjusted FP32')
    source_border=torch.as_tensor(borders[0]);require(source_border.dtype==torch.float32 and canonical.dtype==api.dtype==torch.float32,'original fitted border bits must remain FP32')
    result=probability.from_adjusted_logits(adjusted.detach().cpu().clone(),source_border.detach().cpu().clone(),
            canonical.detach().cpu().clone(),api.detach().cpu().clone(),descriptor=session.probability_descriptor)
    route_name='original_fit_preprocessors_all8' if original else 'frozen88_resident_cache'
    result['member']={'ensemble_index':member,'configuration_identity':configuration,
        'raw_logits':adjusted.detach().cpu().clone(),'from_borders':source_border.detach().cpu().clone(),
        'to_borders':canonical.detach().cpu().clone(),'translated_probabilities':result['raw_weights64']}
    result['neural_v3']={'stage':stage,'member':member,'route':route_name,'module_sha256':source_sha256(),
        'neural_source_sha256':NEURAL_SHA,'prefix':copy.deepcopy(_PREFIX_DESCRIPTOR),
        'actual_adjusted_dtype':str(adjusted.dtype),'actual_probability_dtype':'torch.float64',
        'original_query_views_disabled':original,'fitted_target_mean':float(regressor.y_train_mean_),
        'fitted_target_sd':float(regressor.y_train_std_),'temperature':regressor.softmax_temperature,
        'average_before_softmax':regressor.average_before_softmax,'NN_forward_order_mask_settings_changed':False,
        'selected_root_training':False,'selected_submodules_training_count':0,'model_rng_isolation_scope_delta':1,
        'selected_member_lists_restored':bool(counters.get('original_preprocessing_lists_restored',True) and counters.get('cached_member_lists_restored',True))}
    if not capture_member:result.pop('member')
    return result


def stock_uncached_predict_component(regressor,query,stage,member,session,*,capture_member=True):
    import torch
    require(regressor.fit_mode=='fit_preprocessors','stock original preprocessing fit required')
    require(type(regressor.executor_) is session.neural.inference_module.InferenceEngineCachePreprocessing,'original all8 preprocessing executor required')
    from types import SimpleNamespace
    base=SimpleNamespace(_torch=torch,device=str(regressor.device_),metadata={'model_rng_isolation_count':0})
    base._model_rng=MethodType(cache.ResidentStageExecutor._model_rng,base)
    counters={'prefix_query_count':0,'prefix_query_rows':0,'original_preprocessing_lists_restored':True}
    return _component(regressor,query,stage,member,session,base,counters,original=True,capture_member=capture_member)


class PriorNativeStageExecutor:
    def __init__(self,checkpoint,*,cdf_descriptor=None,query_batch_size=1024,device='cuda'):
        require_sources();self.session=_session();self.probability_descriptor=probability.active_descriptor(cdf_descriptor or self.session.probability_descriptor)
        require(query_batch_size==1024,'frozen query cap1024 required')
        self.inner=neural.PriorNativeStageExecutor(checkpoint,cdf_descriptor=self.session.legacy_guard,query_batch_size=query_batch_size,device=device)
        self.stage_index=None;self.counters={'prefix_query_count':0,'prefix_query_rows':0,'cached_member_lists_restored':True,'original_preprocessing_lists_restored':True}
        self._busy=Lock();self.session.engines.append(self);self.session._stack.callback(self.close)
    @property
    def regressor(self):return self.inner.regressor
    @property
    def base(self):return self.inner.base
    def prepare(self,theta,obs,stage):
        require_sources();require(self.session is _session(),'wrong prepared v3 session')
        result=self.inner.prepare(theta,obs,stage);self.stage_index=stage
        return {**result,'v3_probability_policy':probability.metadata(),'legacy_CDF_guard_is_ancestry_only':True}
    def predict_component(self,member,query,*,capture_member=False,with_log_prob=False):
        require_sources();require(not with_log_prob,'component is not full-mixture log density')
        require(self.regressor is not None,'prepare v3 stage first');require(self._busy.acquire(blocking=False),'concurrent v3 component unsupported')
        try:return _component(self.regressor,query,self.stage_index,member,self.session,self.base,self.counters,
                              original=self.stage_index in NATIVE_STAGES,capture_member=capture_member)
        finally:self._busy.release()
    def predict(self,query,*,output_type='full',capture_members=False):
        require(self.stage_index==0 and len(query)==1 and output_type=='full','only wave-local shared stage0 full path declared; use capture_members for diagnostics')
        predictions=[self.predict_component(i,query,capture_member=True) for i in range(8)]
        return probability.assemble_shared_raw64(predictions,descriptor=self.probability_descriptor,stage0=True)
    def inspect_cache(self):return {**self.inner.inspect_cache(),'v3_probability':probability.metadata(),'v3_counters':dict(self.counters)}
    def close(self):
        self.inner.close();self.stage_index=None


def capture_members(engine,query):
    values=[engine.predict_component(i,query,capture_member=True) for i in range(8)]
    result=probability.assemble_shared_raw64(values,descriptor=engine.probability_descriptor,stage0=False)
    result['full_mixture_diagnostic']=engine.stage_index!=0
    return result


def preparation_witness(report,stage,prepared):return neural.preparation_witness(report,stage,prepared)
def query_witness_delta(before,after,stage,member):return neural.query_witness_delta(before,after,stage,member)

_PUBLIC=[(name,value,getattr(value,'__code__',None)) for name,value in list(globals().items())
         if (getattr(value,'__module__',None)==__name__ and callable(value)) or name in ['NATIVE_STAGES','CONTROL_MODE','cached','cache','codec','stage_policy','mask_training','mask_query','training_bindings','route']]
_METHOD_SEALS=[]
for _owner in (PriorNativeSession,PriorNativeStageExecutor):
    for _name,_value in vars(_owner).items():
        _function=_value.fget if isinstance(_value,property) else _value
        if callable(_function) and hasattr(_function,'__code__'):
            _METHOD_SEALS.append((_owner,_name,_function,_function.__code__,_function.__globals__,
                                  getattr(_function,'__wrapped__',None),getattr(getattr(_function,'__wrapped__',None),'__code__',None)))
for _name,_function in list(globals().items()):
    if getattr(_function,'__module__',None)==__name__ and callable(_function) and hasattr(_function,'__code__'):
        _METHOD_SEALS.append((__import__('sys').modules[__name__],_name,_function,_function.__code__,_function.__globals__,
                              getattr(_function,'__wrapped__',None),getattr(getattr(_function,'__wrapped__',None),'__code__',None)))
_CONFIG_SEALS={name:globals()[name] for name in ['SCHEMA','NEURAL_SHA','SDK_REGRESSOR_SHA','PROBABILITY_SHA','NATIVE_STAGES','CONTROL_MODE','neural','probability',
    'cached','cache','codec','stage_policy','mask_training','mask_query','training_bindings','route','_OWNER']}
_GLOBAL_BINDINGS={name:globals()[name] for entry in _METHOD_SEALS for name in entry[3].co_names
    if name in globals() and name not in ('_PREFIX','_PREFIX_DESCRIPTOR','_PREFIX_ORIGINAL','_PREFIX_SEALS','_IMPORTED_SHA',
                                          '_PUBLIC','_METHOD_SEALS','_CONFIG_SEALS','_GLOBAL_BINDINGS')}
_IMPORTED_SHA=source_sha256()
