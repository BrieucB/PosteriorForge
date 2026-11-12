"""CPU routing and custody engineering tests; no new case data/model/posterior."""
import ast,builtins,copy,inspect,json,subprocess,symtable,sys,textwrap
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from posteriorforge import performance_ss15_scientific_campaign as c


def toy_inputs(seed,case):
 theta=np.arange(4096*21,dtype=np.float64).reshape(4096,21)/1000+(seed-20261111)
 obs=np.arange(4096*33,dtype=np.float64).reshape(4096,33)/1000
 return SimpleNamespace(theta=theta,obs=obs),obs[case:case+1]+case*.3


def toy_banks(monkeypatch):
 calls=[]
 def context(banks,seed,size):
  assert banks=="engineering-only" and size==4096
  calls.append(("context",seed));return toy_inputs(seed,0)[0]
 def query(banks,case):
  assert banks=="engineering-only"
  calls.append(("query",case));return SimpleNamespace(obs=toy_inputs(20261111,case)[1][0])
 monkeypatch.setattr(c,"load_context",context);monkeypatch.setattr(c,"load_heldout",query)
 return calls


def test_exact23_requests_per_fixed_method_and_no_anchor_rerun():
 assert len(c.PENDING_GRID)==23 and len(set(c.PENDING_GRID))==23
 for method in c.METHODS:
  for seed,case in c.PENDING_GRID:
   spec=c.request_spec(method=method,context_seed=seed,case=case)
   assert spec["requested_count"]==50000 and spec["context_size"]==4096
  with pytest.raises(ValueError):c.request_spec(method=method,context_seed=20261111,case=0)


@pytest.mark.parametrize("change",[{"method":"other"},{"context_seed":20261114},{"case":8},
 {"context_size":1024},{"count":49999},{"repetition":1},{"context_seed":True},{"count":True}])
def test_bad_request_rejected_before_inputs_budget_or_output(tmp_path,change):
 spec={"method":"graph_local","context_seed":20261112,"case":3,**change}
 with pytest.raises(ValueError):
  c.run(banks="unused",output=tmp_path/"fresh",checkpoint="unused",ledger="unused",
        reference="unused",reference_grid="unused",qualification="unused",**spec)
 assert not (tmp_path/"fresh").exists()


@pytest.mark.parametrize("method",list(c.METHODS))
def test_actual_seed_query_physical_vs_ordered_model_bindings(monkeypatch,method):
 calls=toy_banks(monkeypatch)
 bundle=c.input_bundle("engineering-only",method,20261113,7)
 assert calls==[("context",20261113),("query",7)]
 context,rawquery=toy_inputs(20261113,7)
 physical=context.theta.astype(np.float32);order=c.order_metadata(method)["stage_to_physical_order"]
 assert bundle["theta"].tobytes()==physical[:,order].copy().tobytes()
 assert bundle["actual_physical_model_inputs"]["context_theta"]==c.array_descriptor(physical)
 assert bundle["model_inputs"]["context_theta"]==c.array_descriptor(bundle["theta"])
 assert bundle["raw_bindings"]["context_theta"]==c.array_descriptor(context.theta)
 assert bundle["raw_bindings"]["query"]==c.array_descriptor(rawquery.reshape(1,33))
 assert bundle["model_target_descriptors"]["physical_theta_float32"]==c.array_descriptor(physical)


@pytest.mark.parametrize("method",list(c.METHODS))
def test_anchor_own_inputs_never_replaced_by_actual_new_context(monkeypatch,tmp_path,method):
 calls=toy_banks(monkeypatch)
 # This test exercises plumbing with synthetic inputs, never fabricates screen evidence.
 actual=c.input_bundle("engineering-only",method,20261113,7)
 monkeypatch.setattr(c,"require_sources",lambda method:c.method_modules(method))
 fixture=tmp_path/"engineering-only-screen.json"
 fixture.write_text(json.dumps({"matching_FP32_execution_passed_all21":True,
  "FP16_hard_validity_passed_all21":True,"FP16_strict_failed_stages":[1]}))
 monkeypatch.setitem(c.METHODS[method],"qualification_sha256",c.sha256_file(fixture))
 seen={}
 def validate(doc,**kwargs):
  seen.update(kwargs);return {"test_plumbing_only":True}
 helper=SimpleNamespace(source_pins=lambda:{"test":"plumbing"},validate_screen=validate)
 monkeypatch.setattr(c,"load_qualifier",lambda method:helper)
 # Fixed flags here label an engineering stub, never inference evidence.
 anchor=c.conditional_anchor("engineering-only",fixture,method)
 expected=c.input_bundle("engineering-only",method,20261111,0)
 assert anchor["context_seed"]==20261111 and anchor["case"]==0
 assert anchor["raw_scientific_bindings"]==expected["raw_bindings"]!=actual["raw_bindings"]
 assert anchor["physical_model_inputs"]==expected["actual_physical_model_inputs"]
 assert seen["raw_bindings"]==expected["raw_bindings"]
 scope=c.numerical_scope(actual,anchor)
 assert not scope["anchor_qualifies_actual_request"] and not scope["actual_request_fixed_conditional_screen_performed"]
 assert not scope["same_context_as_anchor"] and not scope["same_query_as_anchor"]


def test_no_none_screen_execution_before_actual_case_load(tmp_path,monkeypatch):
 monkeypatch.setitem(c.METHODS["sigma_first"],"qualification_sha256",None)
 monkeypatch.setattr(c,"load_context",lambda *a:pytest.fail("new case loaded before screen pin"))
 with pytest.raises(ValueError,match="must be pinned"):
  c.run(banks="unused",output=tmp_path/"fresh",checkpoint="unused",ledger="unused",
        reference="unused",reference_grid="unused",qualification="unused",
        method="sigma_first",context_seed=20261112,case=0)
 assert not (tmp_path/"fresh").exists()


@pytest.mark.parametrize("method",list(c.METHODS))
def test_proposal_mapping_preserves_invalid_complete_vectors(method):
 physical=np.zeros((3,21),np.float32);physical[0,6]=np.nan;physical[1,19]=np.inf
 order=c.order_metadata(method)["stage_to_physical_order"]
 ordered=np.ascontiguousarray(physical[:,order])
 mapped=c.physical_proposals(method,ordered)
 assert mapped.tobytes()==physical.tobytes()
 if method=="graph_local":assert mapped is ordered
 assert mapped.shape==(3,21)


def test_original_sampling_rejection_and_actual_request_seed_roles():
 import posteriorforge.performance_ss15_campaign as frozen_campaign
 def calls(fn,name):
  return [ast.dump(n,include_attributes=False) for n in ast.walk(ast.parse(textwrap.dedent(inspect.getsource(fn))))
          if isinstance(n,ast.Call) and ast.unparse(n.func)==name]
 for name in ["frozen.component_stage_sampler","accept_reject","buffer.finalize","frozen.worker_stream_seeds"]:
  assert calls(c.run,name)==calls(frozen_campaign.run,name)
 assert calls(c.run,"physical_proposals") and calls(c.run,"reference_binding")


def test_no_unresolved_global_names_in_any_function_or_closure():
 table=symtable.symtable(Path(c.__file__).read_text(),str(c.__file__),"exec")
 def visit(scope):
  for symbol in scope.get_symbols():
   if symbol.is_global() and symbol.is_referenced():
    assert symbol.get_name() in vars(c) or symbol.get_name() in vars(builtins),symbol.get_name()
  for child in scope.get_children():visit(child)
 for child in table.get_children():visit(child)


def test_graph_source_policy_torch_free_and_function_identity_sealed(monkeypatch):
 c.require_sources("graph_local")
 subprocess.run([sys.executable,"-c",
  "from posteriorforge import performance_ss15_scientific_campaign as c;import sys;c.require_sources('graph_local');assert 'torch' not in sys.modules"],
  check=True)
 monkeypatch.setattr(c.graph_base,"engine_scope",lambda *a:None)
 with pytest.raises(ValueError,match="function changed"):c.require_sources("graph_local")


def test_actual_completed_graph_case0_anchor_replays_without_new_case_claim():
 root=Path("/scratch/project/eu-26-17/eubrieucb/posteriorforge/results")
 path=root/"accuracy-ss15-v1/graph-local-conditionals-01/manifest.json"
 if not path.is_file():pytest.skip("existing immutable anchor unavailable")
 anchor=c.conditional_anchor(root/"banks-v1",path,"graph_local")
 assert anchor["replay"]["passed"] and anchor["context_seed"]==20261111 and anchor["case"]==0
 assert anchor["qualification_sha256"]==c.METHODS["graph_local"]["qualification_sha256"]
 assert anchor["qualifies_actual_request"] is False

    
def test_sigma_branch_is_corrected_native_family_not_failed_prototype():
 from posteriorforge import performance_ss15_sigma_native as first
 assert c.sigma_base is first
 assert c.METHODS["sigma_first"]["frozen_source"]==first.source_sha256()
 assert c.METHODS["sigma_first"]["qualifier"]=="qualify_ss15_sigma_native.py"
 assert c.METHODS["sigma_first"]["qualifier_source"]==first.QUALIFIER_SHA256
 assert c.METHODS["sigma_first"]["qualification_sha256"]==first.CONDITIONAL_QUALIFICATION_SHA256
 fields=c.native_execution_fields("sigma_first")
 assert fields["native_execution_adapter_source_sha256"]==first.NATIVE_EXECUTION_SOURCE_SHA256
 assert fields["active_native_execution_policy"]["active_fallback_map"]=={"10":[5],"14":list(range(8)),"20":[4]}
 assert fields["order_mask_source_sha256"]==first.ORDER_MASK_SOURCE_SHA256
 assert c.METHODS["sigma_first"]["variant"]=="accuracy_ss15_sigma_first_native_fp16_fifo_scientific_campaign_v1"


def test_actual_completed_sigma_case0_anchor_replays_without_new_case_claim():
 root=Path("/scratch/project/eu-26-17/eubrieucb/posteriorforge/results")
 path=root/"accuracy-ss15-v1/sigma-native-conditionals-01/manifest.json"
 if not path.is_file():pytest.skip("existing immutable native anchor unavailable")
 anchor=c.conditional_anchor(root/"banks-v1",path,"sigma_first")
 assert anchor["replay"]["passed"] and anchor["context_seed"]==20261111 and anchor["case"]==0
 assert anchor["qualification_sha256"]==c.METHODS["sigma_first"]["qualification_sha256"]
 assert anchor["qualifies_actual_request"] is False
 assert anchor["active_native_execution_policy"]["native_stage_indices"]==[14]
 assert anchor["physical_model_inputs"]["context_theta"]==anchor["model_target_descriptors"]["physical_theta_float32"]
 assert anchor["model_inputs"]["context_theta"]==anchor["model_target_descriptors"]["ordered_theta_float32"]
