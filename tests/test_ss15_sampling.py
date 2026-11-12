"""CPU fake internal waves only; no posterior generation."""
import inspect,textwrap
import numpy as np
import pytest
from posteriorforge import execution_ss15_sampling as s
from posteriorforge import performance_fp16_experiment as frozen

class Engine:
 def __init__(self):self.stage=None;self.queries=[];self.prepared=[];self.closed=[]
 def prepare(self,theta,obs,stage):
  self.stage=stage;self.prepared.append(stage)
  return {'fit_stock_cache_seconds':0.,'resident_move_seconds':0.}
 def inspect_cache(self):return {'tensor_records':[],'scope':'fake SS15 CPU fixture'}
 def predict(self,query,**kwargs):
  self.queries.append(query.copy())
  assert query.shape[1]==15+self.stage
  if self.stage:assert np.array_equal(query[:,15:],np.tile(np.arange(self.stage),(len(query),1)))
  return {'draw':np.full(len(query),self.stage,dtype=np.float32)}
 def close(self):self.closed.append(self.stage)


def test_only_observation_reshape_ast_changes():
 f=frozen.frozen_sampling();source=textwrap.dedent(inspect.getsource(f.stage_major_wave))
 tree,d=s.transform_source(source)
 assert d['reverse_ast_exact'] and d['declared_changes']==[{'expression':'observation reshape','from':33,'to':15}]
 assert s.execution_metadata()['canonical_sampling_source_sha256']==frozen.SAMPLING_SHA
 assert s.private_wave() is not f.stage_major_wave
 assert s.private_wave().__globals__['compact_cache'] is f.compact_cache

@pytest.mark.parametrize('edit',['replace_shape','remove_close','change_stages'])
def test_undeclared_source_change_is_rejected(edit):
 source=textwrap.dedent(inspect.getsource(frozen.frozen_sampling().stage_major_wave))
 if edit=='replace_shape':source=source.replace('reshape(1,33)','reshape(1,34)')
 elif edit=='remove_close':source=source.replace('engine.close()','pass')
 else:source=source.replace('range(21)','range(20)')
 with pytest.raises(RuntimeError,match='source changed'):s.transform_source(source)


def test_all21_stage_order_row_prefixes_and_closes_preserved():
 e=Engine();events=[];query=np.arange(15,dtype=np.float32)
 v=s.stage_major_wave(e,np.zeros((4096,21)),np.zeros((4096,15)),query,7,3,lambda p:p['draw'],progress=events.append)
 assert v.shape==(7,21) and np.array_equal(v,np.tile(np.arange(21),(7,1)))
 assert e.prepared==e.closed==list(range(21))
 assert len(events)==21 and len(e.queries)==63
 assert all(np.array_equal(q[:,:15],np.tile(query,(len(q),1))) for q in e.queries)


def test_stage_draw_receives_unchanged_complete_physical_row_groups():
 e=Engine();seen=[]
 def draw(engine,stage,query_for,count,chunk):
  rows=np.array([count-1,0]);q=query_for(rows)
  assert q.shape==(2,15+stage) and count==7 and chunk==3
  seen.append(stage)
  return np.full(count,stage,dtype=np.float32),{'model_query_chunks':1}
 v=s.stage_major_wave(e,None,None,np.zeros(15),7,3,lambda _:None,stage_draw=draw)
 assert seen==list(range(21)) and v.shape==(7,21) and e.closed==list(range(21))


def test_failure_still_closes_active_stage_without_partial_result():
 e=Engine()
 with pytest.raises(RuntimeError,match='incomplete stage'):
  s.stage_major_wave(e,None,None,np.zeros(15),7,3,lambda _:np.zeros(1))
 assert e.closed==[0]


def test_raw33_observation_cannot_be_silently_reused():
 with pytest.raises(ValueError):s.stage_major_wave(Engine(),None,None,np.zeros(33),7,3,lambda _:None)


def test_component_and_inverse_functions_are_original_authenticated_objects():
 f=frozen.frozen_sampling()
 assert f.component_stage_sampler is frozen.frozen_sampling().component_stage_sampler
 assert s.execution_metadata()['e71_inverse_unchanged']
 assert s.execution_metadata()['raw33_qualification_claimed'] is False
