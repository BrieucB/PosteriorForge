"""CPU-only source/dispatch/state tests with a fake stock SDK, never PFN inference."""
from contextlib import contextmanager
from types import FunctionType,SimpleNamespace
import numpy as np
import pytest
from posteriorforge import execution_prior_native_v1 as n
from posteriorforge.inference import CHECKPOINT_SHA256

class Tensor:
 def __init__(self,values,dtype="torch.float32"):
  self.values=np.asarray(values,dtype=np.float32 if dtype=="torch.float32" else np.float64);self.dtype=dtype
 def detach(self):return self
 def cpu(self):return self
 def contiguous(self):return self
 def numpy(self):return self.values
 def clone(self):return Tensor(self.values.copy(),self.dtype)

class Executor:
 def __init__(self,train,target):
  self.preprocessors=[SimpleNamespace(member=i) for i in range(8)]
  self.ensemble_configs=[SimpleNamespace(member=i) for i in range(8)]
  self.X_trains=[train.copy() for _ in range(8)]
  self.y_trains=[target.copy() for _ in range(8)]
  self.cat_ixs=[[] for _ in range(8)]

class Model:
 cache_trainset_representation=False
 def __init__(self):self.weights=Tensor([[1.,2.]]);self.forward_calls=0;self.cold=True
 def named_parameters(self):return [("weight",self.weights)]

@pytest.fixture
def sdk(monkeypatch):
 events=[];module=SimpleNamespace()
 @contextmanager
 def fork_rng(*,devices,enabled=True):
  events.append(("rng_enter",tuple(devices),enabled))
  try:yield
  finally:events.append(("rng_exit",tuple(devices),enabled))
 torch=SimpleNamespace(float32="torch.float32",device=lambda value:SimpleNamespace(index=0),
  cuda=SimpleNamespace(current_device=lambda:0),random=SimpleNamespace(fork_rng=fork_rng))
 session=SimpleNamespace(torch=torch,query_views=True,current=None)
 @contextmanager
 def forward_scope(phase,stage,member):
  previous=session.current;session.current=(phase,stage,member);events.append(("scope_enter",stage,member))
  try:yield
  finally:session.current=previous;events.append(("scope_exit",stage,member))
 session.forward_scope=forward_scope
 class Regressor:
  def __init__(self,**settings):
   self.settings=settings;self.fit_mode=settings["fit_mode"];self.use_autocast_=False
   self.forced_inference_dtype_=settings["inference_precision"]
   self.model_=Model();self.renormalized_criterion_=SimpleNamespace(borders=Tensor([-1.,0.,1.]))
  def fit(self,train,target):
   assert session.current is None
   self.fit_arguments=(train.copy(),target.copy());self.executor_=Executor(train,target)
   self.y_train_mean_=float(target.mean());self.y_train_std_=float(target.std())+1e-20
   events.append(("stock_fit",train.shape,target.shape))
   return self
  def predict(self,value,*,output_type,quantiles):
   assert session.query_views is False and session.current[0]=="query"
   assert all(len(getattr(self.executor_,name))==1 for name in n.cached.ss15.hybrid.PREPROCESSING_FIELDS)
   member=self.executor_.ensemble_configs[0].member
   assert member==session.current[2]
   self.model_.forward_calls+=1;self.model_.cold=False
   if getattr(module,"fail_predict",False):raise RuntimeError("injected original predict failure")
   output=module.translate_probs_across_borders(Tensor(np.zeros((len(value),2))),
       frm=Tensor([-1.,0.,1.]),to=Tensor([-1.,0.,1.]))
   events.append(("stock_predict",member,value.shape))
   return {"logits":Tensor(np.log(output.values)),"criterion":self.renormalized_criterion_}
 module.TabPFNRegressor=Regressor
 session.regressor_module=module
 session.inference_module=SimpleNamespace(InferenceEngineCachePreprocessing=Executor)
 descriptor={"patched_call_count":0,"corrected_cdf_code_sha256":"fixed",
  "original_cdf_code_sha256":"bug","executed_corrected_cdf_code_sha256":None}
 def active(logits,*,frm,to):
  descriptor["patched_call_count"]+=1;descriptor["executed_corrected_cdf_code_sha256"]="fixed"
  return Tensor(np.full((len(logits.values),2),.5))
 module.translate_probs_across_borders=active
 original_cache=n.cache;fake_cache=SimpleNamespace(**vars(original_cache))
 fake_cache._sha=lambda path:CHECKPOINT_SHA256
 fake_cache._require_active_cdf=lambda owner,desc:active
 # Execute the exact original numerical body code in a NEW test dictionary.
 # Neither the frozen function nor its original globals are modified.
 original=n._BODY;namespace=dict(original.__globals__)
 namespace.update(cache=fake_cache,mixture=SimpleNamespace(HEAD_MASS_ENVELOPE=1.00001,
  _component_mass_cpu=lambda p:Tensor(np.sum(p.values,axis=-1,dtype=np.float64),"torch.float64")))
 body=FunctionType(original.__code__,namespace,original.__name__,original.__defaults__,original.__closure__)
 body.__kwdefaults__=original.__kwdefaults__
 monkeypatch.setattr(n,"cache",fake_cache);monkeypatch.setattr(n,"_session",lambda:session)
 monkeypatch.setattr(n,"_BODY",body)
 # Source guards are tested separately. This CPU harness replaces only NEW
 # module aliases, never an original SDK/model/global dictionary.
 monkeypatch.setattr(n,"require_sources",lambda:True)
 return SimpleNamespace(session=session,module=module,descriptor=descriptor,events=events,
  original_body=original,body=body,original_globals=dict(original.__globals__))

def data():
 theta=np.zeros((4096,21),np.float32)
 for j in range(21):theta[:,j]=np.arange(4096,dtype=np.float32)/4096+j
 return theta,np.zeros((4096,15),np.float32)

@pytest.mark.parametrize("stage",[6,10])
def test_original_all8_prepare_labels_no_forward_and_exact_selected_body(sdk,stage):
 theta,obs=data();engine=n.PriorNativeStageExecutor("fake-checkpoint",cdf_descriptor=sdk.descriptor)
 prepared=engine.prepare(theta,obs,stage)
 assert type(engine.child) is n._NativeStage and prepared["original_all8_lists_verified"]
 assert prepared["native_neural_prefill_calls"]==0 and not prepared["native_KV_cache_used"]
 assert prepared["target_codec_coordinate"]["physical_slot"]==n.codec.ORDER[stage]
 regressor=engine.regressor
 np.testing.assert_array_equal(regressor.fit_arguments[0],np.concatenate((obs,theta[:,:stage]),axis=1))
 np.testing.assert_array_equal(regressor.fit_arguments[1],theta[:,stage])
 assert regressor.model_.forward_calls==0 and regressor.model_.cold
 originals={key:getattr(regressor.executor_,key) for key in n.cached.ss15.hybrid.PREPROCESSING_FIELDS}
 query=np.zeros((3,15+stage),np.float32)
 for member in range(8):
  result=engine.predict_component(member,query,capture_member=True)
  assert result["member"]["ensemble_index"]==member
  assert result["member"]["configuration_identity"]==n.cache._configuration_identity(originals["ensemble_configs"][member])
  assert result["prior_native_execution"]["route"]=="original_fit_preprocessors_all8"
  assert sdk.session.query_views is True and sdk.session.current is None
  assert all(getattr(regressor.executor_,key) is value for key,value in originals.items())
 assert sdk.descriptor["patched_call_count"]==8 and regressor.model_.forward_calls==8
 assert engine.child.base._model_rng.__func__ is n._MODEL_RNG
 assert engine.child.base.metadata["model_rng_isolation_count"]==9
 assert sdk.body.__code__ is sdk.original_body.__code__ and sdk.body.__globals__ is not sdk.original_body.__globals__
 assert all(sdk.original_body.__globals__[key] is value for key,value in sdk.original_globals.items())
 assert sdk.module.translate_probs_across_borders is not None
 engine.close();assert engine.child is None

def test_native_exception_restores_view_exact_lists_translator_and_busy_lock(sdk):
 theta,obs=data();engine=n.PriorNativeStageExecutor("fake-checkpoint",cdf_descriptor=sdk.descriptor);engine.prepare(theta,obs,6)
 regressor=engine.regressor;originals={key:getattr(regressor.executor_,key) for key in n.cached.ss15.hybrid.PREPROCESSING_FIELDS}
 translator=sdk.module.translate_probs_across_borders;sdk.module.fail_predict=True
 with pytest.raises(RuntimeError,match="injected"):engine.predict_component(3,np.zeros((1,21),np.float32),capture_member=True)
 assert sdk.session.query_views is True and sdk.session.current is None
 assert sdk.module.translate_probs_across_borders is translator
 assert all(getattr(regressor.executor_,key) is value for key,value in originals.items())
 assert engine.child.hybrid_counters["original_preprocessing_lists_restored"]
 assert engine.child.hybrid_counters["original_translator_alias_restored"]
 assert engine.child.hybrid_counters["native_query_view_setting_restored"]
 sdk.module.fail_predict=False
 assert engine.predict_component(3,np.zeros((1,21),np.float32))["logits"].dtype=="torch.float32"

def test_other19_delegate_exact_class_without_native_calls(sdk,monkeypatch):
 calls=[]
 class Cached:
  def __init__(self,*args,**kwargs):self.metadata={"same":"cached"};self.base=SimpleNamespace();self.regressor=object()
  def prepare(self,*args):calls.append(("prepare",args));return {"same":"original"}
  def predict_component(self,*args,**kwargs):calls.append(("predict",args,kwargs));return {"original":True}
  def predict(self,*args,**kwargs):return ("stock-cached",args,kwargs)
  def inspect_cache(self):return {"original_cache":True}
  def close(self):calls.append(("close",))
 monkeypatch.setattr(n,"_CACHED_CLASS",Cached)
 theta,obs=data();engine=n.PriorNativeStageExecutor("fake",cdf_descriptor=sdk.descriptor)
 for stage in range(21):
  if stage in (6,10):continue
  assert engine.prepare(theta,obs,stage)=={"same":"original"} and type(engine.child) is Cached
  assert calls[-1][1][0] is theta and calls[-1][1][1] is obs and calls[-1][1][2]==stage
  q=np.zeros((1,15+stage),np.float32);result=engine.predict_component(2,q,capture_member=True)
  assert calls[-1][1][1] is q and result["original"] is True
  assert result["prior_native_execution"]["route"]=="frozen88_resident_cache"
  assert engine.predict(q)[0]=="stock-cached"
 assert not sdk.events

def test_session_is_exact88_control_and_FP16_refused():
 assert n.PriorNativeSession.__name__=="PriorNativeSession"
 session=n.PriorNativeSession()
 assert type(session) is n.cached.PriorCoordinateSession and session.mode==n.CONTROL_MODE
 with pytest.raises(RuntimeError,match="FP32 only"):n.PriorNativeSession(n.cached.FP16_MODE)

def test_native_policy_indices_and_unchanged_mask_aliases():
 policy=n.policy_metadata()
 assert policy["native_stages"]==[6,10] and policy["members"]==list(range(8))
 assert [x["physical_slot"] for x in policy["stage_bindings"]]==[5,9]
 assert [x["latent_name"] for x in policy["stage_bindings"]]==["z_d0_m1","z_d0_m2"]
 assert len(policy["other_stages"])==19 and not policy["FP16_activated"]
 for name in ("stage_policy","mask_training","mask_query","training_bindings","codec_module"):
  assert getattr(n,name) is getattr(n.cached,name)

@pytest.mark.parametrize("fault",["source","labels","body"])
def test_wrong_source_labels_or_original_body_refused(monkeypatch,fault):
 if fault=="source":monkeypatch.setattr(n,"CACHED_SHA256","0"*64)
 elif fault=="labels":monkeypatch.setattr(n,"NATIVE_PHYSICAL_SLOTS",(4,9))
 else:monkeypatch.setattr(n,"_BODY",lambda:None)
 with pytest.raises(RuntimeError):n.require_sources()

def test_native_preparation_witness_cannot_invent_forward(sdk):
 theta,obs=data();engine=n.PriorNativeStageExecutor("fake",cdf_descriptor=sdk.descriptor);prepared=engine.prepare(theta,obs,6)
 report={"mode":n.CONTROL_MODE,"module_sha256":n.CACHED_SHA256,"codec_sha256":n.CODEC_SHA256,
  "counters":{"qkv_calls":0,"attention_calls":0},"witnesses":[]}
 assert n.preparation_witness(report,6,prepared)["native_neural_prefill_calls"]==0
 report["counters"]["qkv_calls"]=1
 with pytest.raises(RuntimeError,match="prefill"):n.preparation_witness(report,6,prepared)

def test_immutable_native_snapshot_excludes_only_cold_forward_state(sdk):
 theta,obs=data();engine=n.PriorNativeStageExecutor("fake",cdf_descriptor=sdk.descriptor);engine.prepare(theta,obs,10)
 before=n.immutable_native_snapshot(engine)
 engine.regressor.model_.cold=False;engine.regressor.model_.forward_calls=99
 assert n.immutable_native_snapshot(engine)==before
 engine.regressor.model_.weights.values[0,0]+=1
 assert n.immutable_native_snapshot(engine)["weight_values_sha256"]!=before["weight_values_sha256"]

@pytest.mark.parametrize("field",["X_trains","y_trains"])
def test_fitted_native_context_mutation_changes_immutable_snapshot(sdk,field):
 theta,obs=data();engine=n.PriorNativeStageExecutor("fake",cdf_descriptor=sdk.descriptor);engine.prepare(theta,obs,6)
 before=n.immutable_native_snapshot(engine)
 getattr(engine.regressor.executor_,field)[3].flat[0]+=1
 assert n.immutable_native_snapshot(engine)!=before

def test_native_view_scope_refuses_wrong_stage_or_preexisting_view_change():
 session=SimpleNamespace(query_views=False)
 with pytest.raises(RuntimeError):
  with n.original_native_query_scope(session,6,0):pass
 with pytest.raises(RuntimeError):
  with n.original_native_query_scope(session,5,0):pass

def test_snapshot_configuration_identity_ignores_only_audit_repr(sdk,monkeypatch):
 theta,obs=data();engine=n.PriorNativeStageExecutor("fake",cdf_descriptor=sdk.descriptor);engine.prepare(theta,obs,6)
 identity=n.cache._configuration_identity
 monkeypatch.setattr(n.cache,"_configuration_identity",lambda config:{**identity(config),"repr":"<function stable at 0x111>"})
 before=n.immutable_native_snapshot(engine)
 monkeypatch.setattr(n.cache,"_configuration_identity",lambda config:{**identity(config),"repr":"<function stable at 0x999>"})
 assert n.immutable_native_snapshot(engine)==before
 engine.regressor.executor_.ensemble_configs[0].member=999
 assert n.immutable_native_snapshot(engine)!=before
