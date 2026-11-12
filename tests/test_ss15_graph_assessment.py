"""Actual saved graph-call bindings and frozen sampler policy checks."""
from copy import deepcopy
import importlib.util,json
from pathlib import Path
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location("_graph_assess_tests",ROOT/"scripts/evaluate_ss15_graph.py")
m=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(m)

@pytest.fixture(scope="module")
def actual():
 from posteriorforge.banks import load_context,load_heldout
 root=Path("/scratch/project/eu-26-17/eubrieucb/posteriorforge/results")
 path=root/"accuracy-ss15-v1/graph-local-pilot-01/manifest.json"
 if not path.is_file():pytest.skip("saved Karolina graph pilot unavailable")
 doc=json.loads(path.read_bytes())
 if doc.get("status")!="complete":pytest.skip("saved graph pilot not complete")
 context=load_context(root/"banks-v1",20261111,4096)
 theta,stats,_,_,_=m.producer.model_bindings(context,load_heldout(root/"banks-v1",0).obs.reshape(1,33))
 return doc,theta,stats

def test_actual_training_masks_head_query_counts_and_sampler_policy(actual):
 doc,theta,stats=actual
 assert m.validate_masks(doc,theta,stats)["passed"]
 assert m.validate_sampling(doc)["passed"]

@pytest.mark.parametrize("fault",["target","mask","query_width","global_identity","missing_call","head_count"])
def test_changed_actual_training_query_or_head_binding_is_rejected(actual,fault):
 saved,theta,stats=actual;doc=deepcopy(saved)
 if fault=="target":doc["execution_counters"]["graph_stage_inputs"][4]["prepared_target"]["sha256"]="0"*64
 elif fault=="mask":doc["execution_counters"]["graph_stage_inputs"][4]["mask"]["feature_columns_zeroed"]=[]
 elif fault=="query_width":doc["execution_counters"]["graph_query_calls"][20]["actual_masked_query"]["shape"][1]+=1
 elif fault=="global_identity":doc["execution_counters"]["graph_query_calls"][0]["actual_masked_query"]["sha256"]="0"*64
 elif fault=="missing_call":doc["execution_counters"]["graph_query_calls"].pop()
 else:doc["actual_prediction_checks"]["graph_local_physical_head_contexts"]-=1
 with pytest.raises(ValueError):m.validate_masks(doc,theta,stats)

@pytest.mark.parametrize("fault",["seed","role","uniform","FIFO"])
def test_changed_actual_CPU_stream_or_FIFO_policy_is_rejected(actual,fault):
 saved,_,_=actual;doc=deepcopy(saved)
 if fault=="seed":doc["seed_streams"]["component_uniform"]+=1
 elif fault=="role":doc["sampling_policy"]["streams"]["head_proposal"]["role"]="other"
 elif fault=="uniform":doc["sampling_policy"]["head_retry_preserves_row_uniform"]=False
 else:doc["proposal_buffer"]["request_scope_sha256"]="0"*64
 with pytest.raises((ValueError,RuntimeError)):m.validate_sampling(doc)
