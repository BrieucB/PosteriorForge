"""CPU source/plumbing/cleanup tests; no posterior or model evaluation."""
import ast
import copy
import inspect
from pathlib import Path
import subprocess
import sys
import textwrap
import pytest
from posteriorforge import performance_ss15 as pilot
from posteriorforge import performance_ss15_graph as producer
from posteriorforge import execution_ss15_graph as graph


def calls(function, name):
 tree=ast.parse(textwrap.dedent(inspect.getsource(function)))
 return [ast.dump(n,include_attributes=False) for n in ast.walk(tree)
         if isinstance(n,ast.Call) and ast.unparse(n.func)==name]


def test_sampler_inverse_rejection_wave_and_seed_calls_exact_original():
 for name in ["frozen.component_stage_sampler","wave.stage_major_wave","accept_reject",
              "buffer.finalize","frozen.worker_stream_seeds","frozen.validate_proposal_buffer_record"]:
  assert calls(producer.run,name)==calls(pilot.run,name)
 assert calls(producer.run,"accept_reject") and calls(producer.run,"wave.stage_major_wave")
 assert producer.COUNT==50000 and producer.FALLBACK_MAP=={10:[5],20:[4]}
 assert producer.clean is pilot.clean and producer.wave is pilot.wave
 assert producer.model_bindings is pilot.model_bindings
 assert producer.qualification_receipt is pilot.qualification_receipt
 assert producer.rng_hashes is pilot.rng_hashes


def test_unmasked_numerical_flags_are_nested_only_not_graph_claims():
 tree=ast.parse(textwrap.dedent(inspect.getsource(producer.run)))
 status=next(n.value for n in ast.walk(tree) if isinstance(n,ast.Assign)
             and any(isinstance(t,ast.Name) and t.id=="status" for t in n.targets))
 top={k.value:v for k,v in zip(status.keys,status.values) if isinstance(k,ast.Constant)}
 assert "matching_FP32_execution_passed_all21" not in top
 assert "FP16_hard_validity_passed_all21" not in top
 assert top["actual_graph_local_fixed_conditional_screen_performed"].value is False
 anchor=top["conditional_anchor"]
 entries={k.value:v for k,v in zip(anchor.keys,anchor.values) if isinstance(k,ast.Constant)}
 assert entries["qualifies_graph_local_inputs"].value is False
 assert "matching_FP32_execution_passed_all21" in entries
 assert "actual stage-specific masked inputs" in top["model_inputs_scope"].value


def test_full_fixed_case_quota_and_no_repeat_before_any_output(tmp_path):
 with pytest.raises(RuntimeError,match="repetition0"):
  producer.run(banks="unused",output=tmp_path/"fresh",checkpoint="unused",ledger="unused",
               reference="unused",reference_grid="unused",qualification="unused",repetition=1)
 assert not (tmp_path/"fresh").exists()
 assert "20261111" in inspect.getsource(producer.run)
 assert "heldout=load_heldout(banks,0)" in inspect.getsource(producer.run)


def test_engine_close_on_prepare_or_inspect_exception_precedes_outer_exit(monkeypatch):
 events=[]
 class Fake:
  def __init__(self,*a,**k):events.append("construct")
  def close(self):events.append("close")
 monkeypatch.setattr(graph,"GraphLocalStageAdapter",Fake)
 with pytest.raises(RuntimeError,match="fault"):
  with producer.engine_scope("unused",cdf_descriptor={}):
   events.append("prepare_or_inspect")
   raise RuntimeError("fault")
 events.append("outer_exit")
 assert events==["construct","prepare_or_inspect","close","outer_exit"]


def prediction(stage,member):
 count=8 if member is None else 1
 descriptor={"shape":[1,15+stage],"dtype":"<f4","sha256":"a"*64}
 return {"ss15_execution_proof":{"mode":pilot.MODE,
    "SS15_execution_module_sha256":pilot.SS15_EXECUTION_SHA256,
    "ancestral_precision_module_sha256":pilot.PRECISION_SHA,
    "module_sha256":pilot.source_sha256(),"stage":stage,"member":member,
    "actual_head_contexts":count,"corrected_CDF_calls":count,"witness":{"passed":True}},
    "graph_local_input_proof":{"schema":graph.SCHEMA+".prediction",
    "module_sha256":graph.source_sha256(),"policy_sha256":graph.policy_metadata()["policy_sha256"],
    "stage":stage,"member":member,"mask":graph.stage_policy(stage),
    "unmasked_query":descriptor,"actual_masked_query":copy.deepcopy(descriptor),
    "physical_head_contexts":count,"training_query_same_fixed_mask":True,"target_order_unchanged":True}}


def test_actual_backend_and_graph_head_callbacks_both_bound():
 report={}
 producer.validate_prediction(prediction(0,None),0,None,report=report)
 producer.validate_prediction(prediction(12,3),12,3,report=report)
 assert report["actual_head_contexts"]==report["graph_local_physical_head_contexts"]==9
 assert report["sampler_callback_calls"]==2
 for field,bad in [("module_sha256","0"*64),("mask",graph.stage_policy(11)),
                   ("training_query_same_fixed_mask",False)]:
  corrupt=prediction(12,3);corrupt["graph_local_input_proof"][field]=bad
  with pytest.raises(RuntimeError,match="mask proof"):producer.validate_prediction(corrupt,12,3,report={})


def test_global_factor_query_identity_enforced():
 corrupt=prediction(1,2)
 corrupt["graph_local_input_proof"]["actual_masked_query"]["sha256"]="b"*64
 with pytest.raises(RuntimeError,match="global factor input"):producer.validate_prediction(corrupt,1,2,report={})


def test_policy_and_imports_torch_free():
 producer.require_sources()
 metadata=producer.execution_metadata()
 assert metadata["old_unmasked_anchor_qualifies_graph_inputs"] is False
 assert metadata["actual_graph_local_fixed_conditional_screen_performed"] is False
 assert metadata["graph_local_input_policy"]["module_sha256"]==graph.source_sha256()
 command="from posteriorforge import performance_ss15_graph as p;import sys;p.require_sources();p.execution_metadata();assert 'torch' not in sys.modules"
 subprocess.run([sys.executable,"-c",command],check=True)
