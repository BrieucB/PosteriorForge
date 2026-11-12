"""Independent CPU tests for narrowly bounded cross-process repr recovery."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import pytest

path=Path(__file__).resolve().parents[1]/"scripts/recover_gqa_hybrid_qualification.py"
spec=importlib.util.spec_from_file_location("gqa_hybrid_recovery_tests",path)
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)


def identity(address="1234",name="_identity",digest="stable"):
    return {"type":"tabpfn.preprocessing.RegressorEnsembleConfig","sha256":digest,"repr":f"Config(transform=<function {name} at 0x{address}>, shift=17)"}


def golden(value):return {"type":value["type"],"pickle_sha256":value["sha256"],"repr":value["repr"]}


def test_only_pinned_function_addresses_can_differ_across_processes():
    a=identity();b=golden(identity("abcdef"));original=deepcopy((a,b))
    result=r.semantic_identity(a,b)
    assert result["pickle_sha256"]=="stable" and result["repr_equal"] is False
    assert result["recognized_functions"]==["_identity"] and (a,b)==original
    assert r.semantic_identity(identity(name="_inf_to_nan_func"),golden(identity("FFFF",name="_inf_to_nan_func")))


@pytest.mark.parametrize("mutation",[
    lambda v:v.update(pickle_sha256="different"),lambda v:v.update(type="other.Config"),
    lambda v:v.update(repr=v["repr"].replace("shift=17","shift=18")),
    lambda v:v.update(repr=v["repr"].replace("_identity","_inf_to_nan_func")),
    lambda v:v.update(repr=v["repr"].replace("_identity","unknown_callable"))])
def test_state_type_settings_and_function_name_changes_are_not_erased(mutation):
    a=identity();b=golden(identity("abc"));mutation(b)
    with pytest.raises(ValueError):r.semantic_identity(a,b)


def test_arbitrary_hex_addresses_are_never_globally_stripped():
    a=identity(name="not_authorized");b=golden(identity("abcd",name="not_authorized"))
    with pytest.raises(ValueError,match="beyond recognized"):r.semantic_identity(a,b)


def probability():return {"passed":True,"minimum_probability":0.,"max_normalization_error":0.,"max_cdf_absolute_error":0.,"max_row_total_variation":0.}

def prediction():
    return {"passed":False,"eight_members":True,"exact_final_borders":True,"final_float32":True,"final":probability(),
        "members":[{**probability(),"index":i,"configuration_identity_equal":i!=2,"passed":i!=2,"exact_borders":True,"float32":True,
                    "logits":{"max_centered_logit_error_on_common_finite_bins":0.}} for i in range(8)],
        "inverse_cdf":{"passed":True,"icdf_contract_version":2,"independent_fp32_replay_failures":0,"same_logits_and_borders_exact_failures":0,"probability_space_failures":[]}}


def test_derived_identity_predicates_can_be_recomputed_with_numeric_values_unchanged():
    value=prediction();numeric=deepcopy(value["final"]);changes=[]
    r.prediction(value,repair_identity=True,changes=changes,path="/proof",exact_numeric=True)
    assert value["passed"] and value["members"][2]["passed"] and value["final"]==numeric
    assert changes==["/proof/members/2/configuration_identity_equal"]


@pytest.mark.parametrize("mutation",[
    lambda v:v["final"].update(max_cdf_absolute_error=2.1e-5),
    lambda v:v["members"][2].update(max_row_total_variation=1.1e-4),
    lambda v:v["members"][2].update(exact_borders=False),
    lambda v:v["inverse_cdf"].update(independent_fp32_replay_failures=1),
    lambda v:v["members"][0].update(passed=False)])
def test_identity_recovery_never_excuses_numeric_border_replay_or_unexplained_failure(mutation):
    value=prediction();mutation(value)
    with pytest.raises(ValueError):r.prediction(value,repair_identity=True,changes=[],path="/proof")


def test_all_probability_bounds_remain_finite_nonnegative_and_predeclared():
    for key,value in [("max_normalization_error",float("nan")),("max_row_total_variation",-1.),("minimum_probability",-.01)]:
        item=probability();item[key]=value
        with pytest.raises(ValueError):r.probability(item)


def test_unrecognized_producer_cannot_enter_recovery_before_any_source_import():
    with pytest.raises(ValueError,match="exact immutable failed"):
        r.recover_document({"schema":"other"},{},raw_sha="raw",golden_sha="golden",source_evidence={})


def test_import_is_torch_free():
    import subprocess,sys
    code="import importlib.util,sys;s=importlib.util.spec_from_file_location('recovery',"+repr(str(path))+");m=importlib.util.module_from_spec(s);s.loader.exec_module(m);assert 'torch' not in sys.modules"
    subprocess.run([sys.executable,"-c",code],check=True)
