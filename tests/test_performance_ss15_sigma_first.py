"""No model/posterior calls. Real proposal closure and source-level quota checks."""
import ast,copy,inspect,subprocess,sys,textwrap
from types import SimpleNamespace
import numpy as np
import pytest
from posteriorforge import performance_ss15 as pilot
from posteriorforge import performance_ss15_sigma_first as p,execution_ss15_sigma_first as s


def calls(function,name):
 tree=ast.parse(textwrap.dedent(inspect.getsource(function)))
 return [ast.dump(n,include_attributes=False) for n in ast.walk(tree)
         if isinstance(n,ast.Call) and ast.unparse(n.func)==name]


def test_original_component_seed_rejection_FIFO_and_wave_math_calls_preserved():
 for name in ["frozen.component_stage_sampler","wave.stage_major_wave","accept_reject",
              "buffer.finalize","frozen.worker_stream_seeds","frozen.validate_proposal_buffer_record"]:
  assert calls(p.run,name)==calls(pilot.run,name)
 assert p.COUNT==50000 and p.FALLBACK_MAP==pilot.FALLBACK_MAP=={10:[5],20:[4]}
 assert p.clean is pilot.clean and p.wave is pilot.wave
 assert p.rng_hashes is pilot.rng_hashes and p.qualification_receipt is pilot.qualification_receipt


def test_actual_proposal_closure_restores_before_FIFO_without_filtering_invalid_rows():
 physical=np.zeros((3,21),np.float32);physical[:,[6,10,14,16,18,20]]=.25
 physical[1,6]=np.nan;physical[2,19]=np.inf
 ordered=np.ascontiguousarray(physical[:,s.ORDER]);events=[]
 def fake_wave(*a,**k):
  events.append("complete_ordered_wave")
  assert a[4]==3
  return ordered.copy()
 def from_numpy(value):
  events.append("physical_FIFO_delivery")
  assert value.tobytes()==physical.tobytes()
  return value
 tree=ast.parse(textwrap.dedent(inspect.getsource(p.run)))
 closure=copy.deepcopy(next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=="proposal"))
 scope={"status":{"proposal_wave_sizes":[]},"wave":SimpleNamespace(stage_major_wave=fake_wave),
        "sigma":s,"engine":None,"theta":None,"obs":None,"query":None,"stage_draw":None,
        "progress":None,"torch":SimpleNamespace(from_numpy=from_numpy)}
 code=ast.fix_missing_locations(ast.Module(body=[closure],type_ignores=[]))
 exec(compile(code,"<real_sigma_first_proposal_CPU_fixture>","exec"),scope)
 actual,logprob=scope["proposal"](3)
 assert events==["complete_ordered_wave","physical_FIFO_delivery"]
 assert actual.shape==(3,21) and logprob is None
 assert scope["status"]["proposal_wave_sizes"]==[3]
 from posteriorforge.inference import support_mask
 assert support_mask(actual).tolist()==[True,False,False]


def test_model_target_receipt_source_physical_bank_and_globals_unchanged():
 physical=np.arange(4096*21,dtype=np.float64).reshape(4096,21)/1000
 observed=np.arange(4096*33,dtype=np.float64).reshape(4096,33)/1000
 context=SimpleNamespace(theta=physical,obs=observed)
 query=observed[:1].copy()
 ordered,ss,q,model,qual=p.model_bindings(context,query)
 oldtheta,oldss,oldq,oldmodel,_=pilot.model_bindings(context,query)
 assert ordered.tobytes()==oldtheta[:,s.ORDER].copy().tobytes()
 assert ss.tobytes()==oldss.tobytes() and q.tobytes()==oldq.tobytes()
 receipt=p.target_order_receipt(ordered)
 assert receipt["physical_theta_float32"]==oldmodel["context_theta"]
 assert receipt["ordered_theta_float32"]==model["context_theta"]
 assert receipt["ordered_theta_equals_physical_theta_columns_pi"]
 for j in range(3):
  assert ordered[:,j].tobytes()==oldtheta[:,j].tobytes()
  assert np.concatenate((ss,ordered[:,:j]),axis=1).tobytes()==np.concatenate((oldss,oldtheta[:,:j]),axis=1).tobytes()


def test_anchor_flags_not_relabelled_and_restoration_declared():
 tree=ast.parse(textwrap.dedent(inspect.getsource(p.run)))
 status=next(n.value for n in ast.walk(tree) if isinstance(n,ast.Assign)
             and any(isinstance(t,ast.Name) and t.id=="status" for t in n.targets))
 entries={k.value:v for k,v in zip(status.keys,status.values) if isinstance(k,ast.Constant)}
 assert "matching_FP32_execution_passed_all21" not in entries
 assert entries["actual_sigma_first_fixed_conditional_screen_performed"].value is False
 assert entries["learned_target_order_changed"].value is True
 assert entries["samples_restored_to_physical_order_before_FIFO_support"].value is True
 anchor=entries["conditional_anchor"]
 assert {k.value:v for k,v in zip(anchor.keys,anchor.values)}["qualifies_sigma_first_inputs"].value is False


def test_exact_first_case_count_and_repetition_before_output(tmp_path):
 with pytest.raises(RuntimeError,match="repetition0"):
  p.run(banks="unused",output=tmp_path/"fresh",checkpoint="unused",ledger="unused",
        reference="unused",reference_grid="unused",qualification="unused",repetition=1)
 assert not (tmp_path/"fresh").exists()


def test_close_on_first_failure_before_outer_exit(monkeypatch):
 events=[]
 class Fake:
  def __init__(self,*a,**k):events.append("enter")
  def close(self):events.append("close")
 monkeypatch.setattr(s,"SigmaFirstStageAdapter",Fake)
 with pytest.raises(RuntimeError):
  with p.engine_scope("unused",cdf_descriptor={}):raise RuntimeError("first inspect")
 events.append("outer_exit")
 assert events==["enter","close","outer_exit"]


def test_torch_free_frozen_sources_policy():
 p.require_sources()
 assert p.execution_metadata()["old_unmasked_anchor_qualifies_ordered_inputs"] is False
 subprocess.run([sys.executable,"-c",
  "from posteriorforge import performance_ss15_sigma_first as p;import sys;p.require_sources();p.execution_metadata();assert 'torch' not in sys.modules"],
  check=True)
