"""CPU-only SS15 producer identity, feature custody and failure ownership tests."""
import builtins,hashlib,json,os,subprocess,sys,symtable
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from posteriorforge import performance_ss15 as p
from posteriorforge import representation_ss15 as rep
from posteriorforge.artifacts import sha256_file


def test_import_metadata_help_and_source_guards_are_torch_free():
 subprocess.run([sys.executable,'-c',"import sys;from posteriorforge import performance_ss15 as p;p.require_sources();m=p.execution_metadata();assert 'torch' not in sys.modules;assert m['learned_conditional_input_intentionally_changed'];assert not m['qualified_FP32']"],check=True)
 result=subprocess.run([sys.executable,'-m','posteriorforge.performance_ss15','--help'],check=True,capture_output=True,text=True)
 assert '--qualification' in result.stdout and '--count' not in result.stdout
 assert p.SCHEMA=='posteriorforge.inference.ss15_fp16.v1' and p.COUNT==50000


def test_unpinned_matching_proof_stops_before_budget_or_model(tmp_path,monkeypatch):
 monkeypatch.setattr(p,'CONDITIONAL_QUALIFICATION_SHA256',None)
 with pytest.raises(RuntimeError,match='must be pinned'):
  p.run(banks='none',output=tmp_path/'fresh',checkpoint='none',ledger='none',reference='none',reference_grid='none',qualification='none')
 assert not (tmp_path/'fresh').exists()


def test_model_features_come_from_original_float64_without_target_transform():
 raw=np.linspace(-2,3,4096*33,dtype=np.float64).reshape(4096,33)
 theta=np.linspace(-1,1,4096*21,dtype=np.float64).reshape(4096,21)
 observed=raw[200:201].copy();context=SimpleNamespace(theta=theta,obs=raw)
 t,x,q,m,qualification=p.model_bindings(context,observed)
 assert np.array_equal(t,theta.astype(np.float32))
 assert np.array_equal(x,rep.transform_for_model(raw)) and np.array_equal(q,rep.transform_for_model(observed))
 assert x.shape==(4096,15) and q.shape==(1,15)
 assert np.dtype(m['context_theta']['dtype'])==np.dtype(np.float32) and m['context_obs']['shape']==[4096,15]
 assert np.dtype(qualification['context_obs_SS15_float64']['dtype'])==np.dtype(np.float64)
 assert qualification['query_SS15_model']==m['query']
 assert np.array_equal(context.theta,theta) and np.array_equal(context.obs,raw)


def test_float32_raw_observations_cannot_skip_statistics_precision():
 context=SimpleNamespace(theta=np.zeros((4096,21)),obs=np.zeros((4096,33),np.float32))
 with pytest.raises(ValueError,match='original float64'):p.model_bindings(context,np.zeros((1,33)))


def proof(stage=1,member=2):
 n=8 if member is None else 1
 return {'mode':p.MODE,'SS15_execution_module_sha256':p.SS15_EXECUTION_SHA256,'ancestral_precision_module_sha256':p.PRECISION_SHA,
  'module_sha256':p.source_sha256(),'stage':stage,'member':member,'actual_head_contexts':n,'corrected_CDF_calls':n,'witness':{'passed':True}}


def test_shared_eight_heads_and_callback_counts_are_distinct():
 a={};p.validate_prediction({'ss15_execution_proof':proof(0,None)},0,None,report=a)
 p.validate_prediction({'ss15_execution_proof':proof()},1,2,report=a)
 assert a=={'sampler_callback_calls':2,'actual_head_contexts':9,'shared_densities':1,'original_full_API_identity_claimed':False}

@pytest.mark.parametrize('field,value',[('module_sha256','bad'),('SS15_execution_module_sha256','bad'),('stage',0),
 ('ancestral_precision_module_sha256','bad'),('member',3),('actual_head_contexts',8),('corrected_CDF_calls',0)])
def test_misbound_physical_prediction_is_rejected(field,value):
 item=proof();item[field]=value
 with pytest.raises(RuntimeError):p.validate_prediction({'ss15_execution_proof':item},1,2,report={})

@pytest.mark.parametrize('failure',['enter','constructor','prepare'])
def test_prepare_failures_release_only_entered_precision_scope(monkeypatch,failure):
 from posteriorforge import execution_ss15 as engine
 events=[]
 class Session:
  def __init__(self,mode):pass
  def __enter__(self):
   if failure=='enter':raise RuntimeError('injected enter')
   return self
  def __exit__(self,*args):events.append('precision_exit')
  def report(self):return {'aliases_and_flags_restored':True,'counters':{}}
 class Stage:
  def __init__(self,*args,**kwargs):
   if failure=='constructor':raise RuntimeError('injected constructor')
  def prepare(self,*args):raise RuntimeError('injected prepare')
  def close(self):events.append('stage_close')
 monkeypatch.setattr(engine,'SS15PrecisionSession',Session);monkeypatch.setattr(engine,'SS15StageExecutor',Stage)
 a=p.SS15StageAdapter('unused',cdf_descriptor={})
 with pytest.raises(RuntimeError,match='injected'):a.prepare(None,None,0)
 assert a.session is None and a.engine is None
 assert events==([] if failure=='enter' else ['precision_exit'] if failure=='constructor' else ['stage_close','precision_exit'])
 a.close()

@pytest.mark.parametrize('failure',['inspect','compact'])
def test_pretry_wave_failure_closes_engine_before_CDF_exit(monkeypatch,failure):
 events=[];active={'CDF':False}
 class CDF:
  def __enter__(self):active['CDF']=True;events.append('CDF_enter');return {}
  def __exit__(self,*args):assert events[-1]=='engine_close';events.append('CDF_exit');active['CDF']=False
 class Engine:
  def __init__(self,*args,**kwargs):pass
  def prepare(self,*args):events.append('prepare');return {}
  def inspect_cache(self):
   events.append('inspect')
   if failure=='inspect':raise RuntimeError('injected inspect')
   return {'tensor_records':[{'category':'parameter'}]}
  def close(self):assert active['CDF'];events.append('engine_close')
 monkeypatch.setattr(p,'SS15StageAdapter',Engine)
 with pytest.raises((RuntimeError,KeyError)):
  with CDF() as descriptor,p.engine_scope('unused',cdf_descriptor=descriptor) as engine:
   p.wave.stage_major_wave(engine,None,None,np.zeros(15),1,1,lambda _:None)
 assert events==['CDF_enter','prepare','inspect','engine_close','CDF_exit']


def test_all_module_globals_used_by_run_are_resolved():
 tree=symtable.symtable(Path(p.__file__).read_text(),p.__file__,'exec')
 for name in ('run','qualification_receipt'):
  scope=next(c for c in tree.get_children() if c.get_name()==name)
  assert not [s.get_name() for s in scope.get_symbols() if s.is_global() and s.is_referenced() and s.get_name() not in vars(p) and not hasattr(builtins,s.get_name())]
