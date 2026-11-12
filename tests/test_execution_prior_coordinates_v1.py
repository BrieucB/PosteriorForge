"""New namespace/mask/source/sampler checks; CPU only, no checkpoint."""
from types import FunctionType,SimpleNamespace
import numpy as np
import pytest
from posteriorforge import execution_prior_coordinates_v1 as b,prior_coordinates_v1 as codec

@pytest.fixture
def pinned(monkeypatch):
 # Tests bind the actual new codec bytes; production remains disabled until its
 # declared final source hash is frozen by the owner.
 monkeypatch.setattr(b,"CODEC_SHA256",codec.source_sha256())
 return b

def data():
 theta=np.arange(4096*21,dtype=np.float32).reshape(4096,21)/10000
 obs=np.arange(4096*15,dtype=np.float32).reshape(4096,15)/10000
 return np.ascontiguousarray(theta),np.ascontiguousarray(obs)

def test_unfrozen_codec_failclosed(monkeypatch):
 monkeypatch.setattr(b,"CODEC_SHA256",None)
 with pytest.raises(b.PriorExecutionError,match="not frozen"):b.execution_metadata()

def test_all21_mask_target_future_and_globals(pinned):
 theta,obs=data();a=theta.copy();o=obs.copy()
 for j in range(21):
  mt,ms=b.mask_training(theta,obs,j);rule=b.stage_policy(j)
  query=np.concatenate((obs[:8],theta[:8,:j]),axis=1);mq=b.mask_query(query,j)
  train=np.concatenate((ms,mt[:,:j]),axis=1)
  assert np.array_equal(train[:8],mq)
  assert mt[:,j:].tobytes()==theta[:,j:].tobytes()
  assert rule["target_original_physical_slot"]==b.ORDER[j] and not rule["target_is_original_physical_value"]
  if j<3:assert mt is theta and ms is obs and mq is query
  else:assert {0,1,2}<=set(rule["ordered_latent_prefix_indices_kept"])
 assert np.array_equal(theta,a) and np.array_equal(obs,o)

def test_mask_removes_crossblock_latents(pinned):
 theta,obs=data()
 for j in range(3,21):
  query=np.concatenate((obs[:8],theta[:8,:j]),axis=1);changed=query.copy()
  changed[:,b.stage_policy(j)["feature_columns_zeroed"]]+=999
  assert b.mask_query(query,j).tobytes()==b.mask_query(changed,j).tobytes()

def test_no_native_policy_inherited(pinned):
 session=b.PriorCoordinateSession(b.CONTROL_MODE)
 assert session.fallback_map=={} and all(not session.strict_fallback(j,m) for j in range(21) for m in range(8))
 md=b.execution_metadata();assert md["fallback_map"]=={} and not md["old_numerical_qualification_inherited"]
 for fallback in ({10:[5]},{14:list(range(8))}):
  with pytest.raises(b.PriorExecutionError):b.PriorCoordinateSession(fallback_map=fallback)

def test_policy_descriptor_and_json_roundtrip(pinned):
 import json
 md=b.policy_metadata();assert json.loads(json.dumps(md))==md
 assert md["codec"]==codec.metadata() and md["stage_to_original_physical_slot"]==list(codec.ORDER)
 assert not md["target_or_prefix_physical_label_identity_claimed"]
 theta,obs=data();receipt=b.training_bindings(theta,obs,8)
 assert receipt["original_latent_target"]==receipt["prepared_latent_target"] and receipt["target_bytes_unchanged"]

def test_original_dictionary_and_code_objects_unchanged(pinned):
 original={name:(f,f.__code__,f.__globals__,dict(f.__globals__)) for name,f in b.ss15._CLONES.items()}
 descriptor=b.clone_descriptors()
 assert all(v["code_object_is_frozen_1732"] for v in descriptor["numerical_private_functions"].values())
 for name,(f,code,glob,snapshot) in original.items():
  assert b.ss15._CLONES[name] is f and f.__code__ is code and f.__globals__ is glob
  assert all(f.__globals__[k] is v for k,v in snapshot.items())
 for name,f in b._CLONES.items():
  assert f.__globals__ is not b.ss15._CLONES[name].__globals__
  assert f.__globals__["_ss15_session"] is b._session

def test_constructor_code_with_new_policy_and_no_SDK(pinned):
 session=SimpleNamespace(fallback_map={},mode=b.CONTROL_MODE)
 class Resident:
  def __init__(self,*args,**kwargs):self.arguments=(args,kwargs)
 original=b._METHODS["stage_init"]
 namespace=dict(original.__globals__);namespace.update(_guard_ss15=lambda:session,SS15Resident=Resident,
  _hybrid_metadata=lambda mode:{"mode":mode},mixture=SimpleNamespace(ComponentMixtureExecutor=lambda base:("component",base)))
 ctor=FunctionType(original.__code__,namespace,original.__name__,original.__defaults__,original.__closure__)
 ctor.__kwdefaults__=original.__kwdefaults__
 target=object.__new__(b.PriorCoordinateStageExecutor);ctor(target,"checkpoint",cdf_descriptor={"probe":True})
 assert target.fallback_map=={} and target.session is session and isinstance(target.base,Resident)
 assert original.__code__ is b.ss15.SS15StageExecutor.__init__.__code__

def test_private_dictionary_tamper_rejected(pinned,monkeypatch):
 monkeypatch.setitem(b._METHODS,"stage_predict",lambda:None)
 with pytest.raises(b.PriorExecutionError,match="dictionary"):b.require_sources()

def test_old_method_tamper_rejected(pinned,monkeypatch):
 monkeypatch.setattr(b.ss15.SS15Resident,"prepare",lambda *args:None)
 with pytest.raises(b.PriorExecutionError,match="alias"):b.require_sources()

def test_training_query_nonfinite_or_implicit_cast_rejected():
 theta,obs=data()
 with pytest.raises(b.PriorExecutionError):b.mask_training(theta.astype(np.float64),obs,4)
 query=np.zeros((1,19),np.float32);query[0,-1]=np.inf
 with pytest.raises(b.PriorExecutionError):b.mask_query(query,4)
 query[0,-1]=np.nan
 with pytest.raises(b.PriorExecutionError):b.mask_query(query,4)

def test_sampler_decodes_latent_draws_not_uniform_physical_bins(pinned):
 torch=pytest.importorskip("torch")
 from posteriorforge.execution_mixture import finite_bar_icdf
 logits=torch.zeros((1,2),dtype=torch.float32);borders=torch.tensor([-2.,0.,2.])
 uniforms=torch.tensor([.1,.3,.6,.9],dtype=torch.float32)
 draws=finite_bar_icdf(logits,borders,uniforms)
 ordered=np.zeros((4,21),np.float32);ordered[:,3]=draws.numpy()
 physical=codec.decode_ordered(ordered)
 assert np.isfinite(physical).all() and np.all((physical[:,6]>.01)&(physical[:,6]<.5))
 # The nonlinear inverse must act on the actual latent draw, not only bin edges.
 edges=np.zeros((3,21),np.float64);edges[:,3]=borders.numpy();decoded_edges=codec.decode_ordered(edges)[:,6]
 bins=np.array([0,0,1,1]);fractions=np.array([.2,.6,.2,.8])
 wrong=decoded_edges[bins]+np.diff(decoded_edges)[bins]*fractions
 assert np.max(np.abs(physical[:,6]-wrong))>1e-4
 assert finite_bar_icdf(logits,borders,uniforms).numpy().tobytes()==draws.numpy().tobytes()

def precision_report(stage,mode,query_member=None,query_count=0):
 strict=mode==b.CONTROL_MODE
 witnesses=[]
 for member in range(8):
  witnesses.append({"phase":"prefill","stage":stage,"member":member,"count":2,"cache_kv":True,
   "strict_fp32_fallback":False,"allow_tf32":False,"matmul_precision":"highest","autocast_enabled":not strict,
   "autocast_dtype":"torch.float16","projection_dtypes":{"q":None,"kv":"torch.float32" if strict else "torch.float16"},
   "supplied_cache_dtypes":["torch.float32" if strict else "torch.float16"]})
 if query_member is not None:
  witnesses.append({**witnesses[query_member],"phase":"query","member":query_member,"count":query_count,"cache_kv":False})
 count=sum(w["count"] for w in witnesses)
 return {"module_sha256":b.source_sha256(),"codec_sha256":b.CODEC_SHA256,"fallback_map":{},"mode":mode,
  "counters":{"qkv_calls":count,"attention_calls":count},"witnesses":witnesses}

@pytest.mark.parametrize("mode",[b.CONTROL_MODE,b.FP16_MODE])
def test_actual_phase_witnesses_and_query_delta_no_old_member5_exception(pinned,mode):
 before=precision_report(10,mode);after=precision_report(10,mode,5,3)
 assert b.prefill_witness_passed(before,10,mode=mode)["passed"]
 assert b.query_witness_delta(before,after,10,5,mode=mode)["qkv_calls"]==3

def test_false_native_fallback_or_histogram_loss_rejected(pinned):
 before=precision_report(10,b.FP16_MODE);after=precision_report(10,b.FP16_MODE,5,3)
 after["witnesses"][-1]["strict_fp32_fallback"]=True
 with pytest.raises(b.PriorExecutionError):b.query_witness_delta(before,after,10,5,mode=b.FP16_MODE)
 after=precision_report(10,b.FP16_MODE,5,3);after["witnesses"][0]["count"]-=1
 after["counters"]["qkv_calls"]-=1;after["counters"]["attention_calls"]-=1
 with pytest.raises(b.PriorExecutionError,match="decreased"):b.query_witness_delta(before,after,10,5,mode=b.FP16_MODE)

def test_phase_witness_dtype_and_member_mismatch_rejected(pinned):
 report=precision_report(4,b.CONTROL_MODE)
 report["witnesses"][2]["projection_dtypes"]["kv"]="torch.float16"
 with pytest.raises(b.PriorExecutionError,match="dtype"):b.prefill_witness_passed(report,4)
 before=precision_report(4,b.CONTROL_MODE);after=precision_report(4,b.CONTROL_MODE,6,3)
 with pytest.raises(b.PriorExecutionError,match="unrelated"):b.query_witness_delta(before,after,4,5)
