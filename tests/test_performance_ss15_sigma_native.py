"""CPU native-producer gates/provenance/cleanup; no model or cohort execution."""
import ast,builtins,inspect,json,symtable,textwrap
from pathlib import Path
import pytest
from posteriorforge import performance_ss15_sigma_native as p
from posteriorforge import performance_ss15_sigma_first as old
from posteriorforge import execution_ss15_sigma_native_policy as n


def calls(function,name):
 return [ast.dump(node,include_attributes=False) for node in ast.walk(ast.parse(textwrap.dedent(inspect.getsource(function))))
  if isinstance(node,ast.Call) and ast.unparse(node.func)==name]


def test_science_inputs_order_draws_inverse_and_rejection_calls_exact_prototype():
 assert p.model_bindings is old.model_bindings and p.target_order_receipt is old.target_order_receipt
 assert p.wave is old.wave and p.clean is old.clean and p.rng_hashes is old.rng_hashes
 assert p.NATIVE_STAGE_INDICES==(14,) and p.FALLBACK_MAP=={10:[5],14:list(range(8)),20:[4]}
 tree=ast.parse(textwrap.dedent(inspect.getsource(p.run)))
 status=next(node.value for node in ast.walk(tree) if isinstance(node,ast.Assign)
  and any(isinstance(t,ast.Name) and t.id=="status" for t in node.targets))
 fields={key.value:value for key,value in zip(status.keys,status.values) if isinstance(key,ast.Constant)}
 assert fields["actual_sigma_first_fixed_conditional_screen_performed"].value is False
 assert fields["fresh_same_policy_all21_screen_replayed"].value is True
 assert "performed separately" in fields["fresh_screen_scope"].value
 for name in ["frozen.worker_stream_seeds","frozen.component_stage_sampler","wave.stage_major_wave",
              "accept_reject","buffer.finalize"]:
  assert calls(p.run,name)==calls(old.run,name)
 def proposal(function):
  return next(node for node in ast.walk(ast.parse(textwrap.dedent(inspect.getsource(function))))
   if isinstance(node,ast.FunctionDef) and node.name=="proposal")
 # The original run also inverted theta only for historical-anchor metadata;
 # compare the actual full proposal closure, including its inverse, directly.
 assert ast.dump(proposal(p.run),include_attributes=False)==ast.dump(proposal(old.run),include_attributes=False)


def test_no_final_proof_no_context_no_model_no_output(tmp_path,monkeypatch):
 monkeypatch.setattr(p,"CONDITIONAL_QUALIFICATION_SHA256",None)
 with pytest.raises(RuntimeError,match="proof must be pinned"):
  p.run(banks="must-not-load",output=tmp_path/"fresh",checkpoint="unused",ledger="unused",
   reference="unused",reference_grid="unused",qualification="missing")
 assert not (tmp_path/"fresh").exists()


def test_active_native_and_ancestral_metadata_remain_distinct_and_json_replayable():
 p.require_sources()
 policy=p.execution_metadata()
 assert policy["active_native_execution_policy"]["native_stage_indices"]==[14]
 assert policy["active_native_execution_policy"]["active_fallback_map"]["14"]==list(range(8))
 assert policy["frozen_scientific_sigma_policy_ancestry"]==old.execution_metadata()
 assert json.loads(json.dumps(policy,sort_keys=True))==policy
 assert policy["new_scientific_order_or_feature_change"] is False


def test_qualification_gate_rejects_missing_pin_before_helper_or_file(monkeypatch):
 monkeypatch.setattr(p,"CONDITIONAL_QUALIFICATION_SHA256",None)
 with pytest.raises(RuntimeError,match="all21 screen must be pinned"):
  p.qualification_receipt("must-not-read",raw_bindings={},transformed_bindings={})


def test_scope_closes_even_when_first_prepare_or_inspect_fails(monkeypatch):
 events=[]
 class Fake:
  def __init__(self,*args,**kwargs):events.append("construct")
  def close(self):events.append("release")
 monkeypatch.setattr(n,"adapter_class",lambda stages:Fake)
 with pytest.raises(RuntimeError):
  with p.engine_scope("no-model",cdf_descriptor={}):
   events.append("failure")
   raise RuntimeError("first prepare or inspect")
 events.append("CDF-exit")
 assert events==["construct","failure","release","CDF-exit"]


def test_native_policy_mutation_fails_closed(monkeypatch):
 monkeypatch.setattr(p,"FALLBACK_MAP",{10:[5],14:[0],20:[4]})
 with pytest.raises(RuntimeError,match="native policy changed"):p.require_sources()


def test_no_unresolved_globals():
 table=symtable.symtable(Path(p.__file__).read_text(),str(p.__file__),"exec")
 def visit(scope):
  for symbol in scope.get_symbols():
   if symbol.is_global() and symbol.is_referenced():
    assert symbol.get_name() in vars(p) or symbol.get_name() in vars(builtins),symbol.get_name()
  for child in scope.get_children():visit(child)
 for child in table.get_children():visit(child)
