"""CPU-only SS15 dimensions, reversible AST scope and original-source isolation."""
from contextlib import nullcontext
import copy
import importlib.util
from pathlib import Path
import subprocess
import sys
from types import FunctionType,SimpleNamespace
import numpy as np
import pytest
from posteriorforge import execution_ss15 as adapter


def test_import_and_clones_are_torch_safe_and_all_edits_reverse_exactly():
 subprocess.run([sys.executable,"-c","import sys;from posteriorforge import execution_ss15 as s;assert 'torch' not in sys.modules;s.require_sources();assert all(v['reverse_AST_exact'] for v in s.clone_descriptors().values())"],check=True)
 descriptors=adapter.clone_descriptors()
 assert {name:row["change_counts"].get("observation_width",0) for name,row in descriptors.items()}=={
  "resident_prepare":3,"resident_predict":0,"hybrid_prepare":2,"uncached_predict_body":0,"uncached_init":4,"uncached_component":0}
 assert all(not row["original_global_dictionary_modified"] for row in descriptors.values())


def test_new_metadata_is_ss15_and_old_raw33_contract_unchanged():
 from posteriorforge import execution_cache as cache
 original=cache.execution_metadata()
 current=adapter.execution_metadata(adapter.CONTROL_MODE)
 assert current["observation_features"]==15 and current["representation"]["module_sha256"]==adapter.REPRESENTATION_SHA256
 assert not current["raw33_numerical_qualification_applies"]
 assert original["config"]["observation_features"]==33 and cache.execution_metadata()==original
 assert current["mode"]==adapter.CONTROL_MODE


@pytest.mark.parametrize("stage",[0,6,20])
def test_training_and_query_width_allow_nan_prefix_only(stage):
 theta=np.zeros((4096,21));ss=np.zeros((4096,15))
 t,x=adapter._numpy_training(theta,ss)
 assert t.dtype==x.dtype==np.float32 and x.shape==(4096,15)
 query=np.zeros((16,15+stage),np.float32)
 if stage:query[0,-1]=np.nan
 assert np.array_equal(adapter._numpy_query(query,stage,1024),query,equal_nan=True)
 with pytest.raises(ValueError):adapter._numpy_training(theta,np.zeros((4096,33)))
 with pytest.raises(ValueError):adapter._numpy_query(np.zeros((16,33+stage)),stage,1024)
 query[0,0]=np.inf
 with pytest.raises(ValueError):adapter._numpy_query(query,stage,1024)
 with pytest.raises(ValueError):adapter._numpy_query(np.zeros((1025,15+stage)),stage,1024)


def test_explicit_fallback_policy_required_and_no_implicit_extension():
 session=adapter.SS15PrecisionSession(adapter.FP16_MODE)
 assert session.strict_fallback(10,5) and not session.strict_fallback(13,6)
 extended=adapter.SS15PrecisionSession(adapter.FP16_MODE,fallback_map={10:[5],13:[6]})
 assert extended.strict_fallback(13,6) and extended.descriptor["fallback_map"]=={"10":[5],"13":[6]}
 assert session.descriptor["fallback_map"]=={"10":[5]}
 assert "UNUSED" in session.descriptor["original_session_resident_prepare"]
 with pytest.raises(ValueError):adapter.SS15PrecisionSession(adapter.FP16_MODE,fallback_map={13:[6]})
 with pytest.raises(ValueError):adapter.SS15PrecisionSession("unapproved-mode")


def fake_sdk(torch):
 seen=[]
 class Model:
  def to(self,device):seen.append(("resident_move",device));return self
 class CacheKV:
  def __init__(self):self.models=[Model() for _ in range(8)];self.configs=list(range(8))
  def iter_outputs(self,*args,**kwargs):
   if False:yield None
 class Regressor:
  def __init__(self,**kwargs):self.kwargs=kwargs;self.executor_=CacheKV()
  def fit(self,x,y):seen.append(("fit",x.clone(),y.clone(),self.kwargs));return self
 fake_torch=SimpleNamespace(cat=torch.cat,from_numpy=torch.from_numpy,float32=torch.float32,cuda=SimpleNamespace(synchronize=lambda:None))
 return Regressor,CacheKV,fake_torch,seen


@pytest.mark.parametrize("stage",[0,6,20])
@pytest.mark.parametrize("inference_setting",["autocast","fp32"])
def test_actual_guarded_resident_prepare_body_fits_exact15_plus_stage_and_physical_target(stage,inference_setting):
 torch=pytest.importorskip("torch")
 Regressor,CacheKV,fake_torch,seen=fake_sdk(torch)
 source=adapter._CLONES["resident_prepare"];namespace=dict(source.__globals__)
 precision_value=torch.float32 if inference_setting=="fp32" else "autocast"
 namespace.update(_ss15_inference_setting=lambda:precision_value,_ss15_cache_assumptions=lambda models,torch:[{}]*8,
                  _resident_iter=lambda original:original,_require_active_cdf=lambda module,descriptor:None)
 function=FunctionType(source.__code__,namespace)
 value=SimpleNamespace(_torch=fake_torch,settings={},cdf_descriptor={},device="cuda",_sdk_regressor=SimpleNamespace(TabPFNRegressor=Regressor),
  _sdk_inference=SimpleNamespace(InferenceEngineCacheKV=CacheKV,__file__=__file__),
  close=lambda:None,_model_rng=lambda:nullcontext(),inspect_cache=lambda:{},
  metadata={"prepare_count":0,"stages":[]})
 theta=np.broadcast_to(np.arange(21,dtype=np.float32),(4096,21)).copy()
 ss=np.broadcast_to(100+np.arange(15,dtype=np.float32),(4096,15)).copy()
 prepared=function(value,theta,ss,stage)
 fit=next(row for row in seen if row[0]=="fit");x,y,kwargs=fit[1:]
 assert x.shape==(4096,15+stage) and torch.equal(x[:,:15],torch.from_numpy(ss))
 assert torch.equal(y,torch.full((4096,),float(stage)))
 assert torch.equal(x[:,15:],torch.from_numpy(theta[:,:stage]))
 assert kwargs["inference_precision"]==precision_value and prepared["predictor_count"]==15+stage
 assert value.metadata["prepare_count"]==1 and len([r for r in seen if r[0]=="resident_move"])==8
 assert not torch.cuda._initialized


@pytest.mark.parametrize("stage",[0,10,20])
def test_actual_guarded_uncached_constructor_fits_same_width_and_all8_fp32_fallback(stage):
 torch=pytest.importorskip("torch");Regressor,CacheKV,fake_torch,seen=fake_sdk(torch)
 source=adapter._CLONES["uncached_init"];namespace=dict(source.__globals__)
 session=SimpleNamespace(torch=fake_torch,mode=adapter.FP16_MODE,regressor_module=SimpleNamespace(TabPFNRegressor=Regressor),fallback_map={10:(5,)})
 namespace.update(_ss15_session=lambda:session,_inference_setting=lambda:"autocast",_ss15_hybrid_metadata=lambda mode:{},
  cache=SimpleNamespace(ResidentStageExecutor=SimpleNamespace(_model_rng=lambda self:nullcontext())))
 function=FunctionType(source.__code__,namespace)
 value=SimpleNamespace();theta=np.broadcast_to(np.arange(21,dtype=np.float32),(4096,21)).copy();ss=np.zeros((4096,15),np.float32)
 function(value,"mock-checkpoint",cdf_descriptor={},stage_index=stage,theta=theta,obs=ss)
 fits=[row for row in seen if row[0]=="fit"]
 assert len(fits)==(2 if stage==10 else 1)
 assert all(row[1].shape==(4096,15+stage) and torch.equal(row[2],torch.full((4096,),float(stage))) for row in fits)
 if stage==10:assert fits[1][3]["inference_precision"]==torch.float32
 assert not torch.cuda._initialized


def test_original_namespaces_and_session_private_methods_are_untouched():
 from posteriorforge import execution_cache as cache,execution_hybrid as hybrid,execution_precision_probe as precision
 functions=[cache.ResidentStageExecutor.prepare,cache.ResidentStageExecutor.predict,hybrid.HybridStageExecutor.prepare,
            hybrid.HybridStageExecutor._predict_uncached,precision.UncachedPrecisionStage.__init__]
 before=[{key:id(value) for key,value in fn.__globals__.items()} for fn in functions]
 code_before=[fn.__code__ for fn in functions]
 private_before=precision._ACTIVE
 clones,descriptors=adapter.build_clones()
 assert all(row["reverse_AST_exact"] for row in descriptors.values())
 assert [fn.__code__ for fn in functions]==code_before and precision._ACTIVE is private_before
 assert [{key:id(value) for key,value in fn.__globals__.items()} for fn in functions]==before
 assert all("_ss15_numpy_training" not in fn.__globals__ for fn in functions)


def test_wrong_ast_edit_count_fails_closed_before_function_execution():
 original=adapter.cache.ResidentStageExecutor.predict
 with pytest.raises(adapter.SS15Error,match="edit count"):
  adapter._clone(original,[("missing","123456789","15",1)],{},"deliberately-wrong-test")


def test_declared_policy_cannot_be_mutated_in_place():
 session=adapter.SS15PrecisionSession(adapter.FP16_MODE);session.require_policy()
 session.fallback_map[13]=(6,)
 with pytest.raises(adapter.SS15Error,match="policy mutated"):session.require_policy()
 session=adapter.SS15PrecisionSession(adapter.CONTROL_MODE);session.mode=adapter.FP16_MODE
 with pytest.raises(adapter.SS15Error,match="policy mutated"):session.require_policy()


def test_uncached_stage_rejects_invalid_target_before_any_model_or_gpu():
 for stage in (-1,21,True):
  with pytest.raises(ValueError,match="stage index"):adapter.SS15UncachedStage("unused",cdf_descriptor={},stage_index=stage,theta=None,obs=None)
