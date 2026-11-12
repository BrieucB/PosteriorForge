"""Pure CPU metadata/domain replay tests, not statistical posterior tests."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from posteriorforge import prior_coordinates_v1 as codec
from posteriorforge.science.analytical_benchmark_v2 import array_descriptor

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("prior_inventory",ROOT/"scripts/qualify_prior_coordinates_v1.py")
inv=importlib.util.module_from_spec(spec);spec.loader.exec_module(inv)

def toy_rows(n):
    # Repeated declared engineering inputs; no simulator or posterior draws.
    values=np.zeros((n,21),np.float64)
    values[:,:3]=[.3,-.2,.1]
    for start in (3,7,11):values[:,start:start+3]=[.4,-.1,.2]
    values[:,[15,17,19]]=.4
    values[:,codec.NOISE_INDICES]=.255
    return values

@pytest.fixture(scope="module")
def valid_document():
    doc=dict(schema=inv.SCHEMA,status="codec_inventory_pass",qualified=True,
        source_pins=inv.source_pins(),codec_metadata=codec.metadata(),
        all_original_training_rows_inventoried=True,banks_unchanged=True,sources_unchanged=True,
        default_RNG_preserved=True,torch_imported=False,RNG_before={"numpy":"same"},RNG_after={"numpy":"same"},
        matching_backend_conditionals_qualified=False,posterior_accuracy_assessed=False,banks=[])
    q=inv.qualify_matrix(toy_rows(4096))
    for seed in (20261111,20261112,20261113):
        doc["banks"].append(dict(context_seed=seed,count=4096,passed=True,domain=q["domain"],qualification=q,
            original_theta=q["physical_Float64"],original_observations=array_descriptor(np.zeros((4096,33),np.float64)),
            observation_array_unchanged=True))
    q=inv.qualify_matrix(toy_rows(1024))
    doc["case2_fixed_prefix"]=dict(count=1024,passed=True,domain=q["domain"],qualification=q,
        custody=dict(rule=inv.PREFIX_RULE,selected_indices=list(range(1024)),
            manifest_sha256=inv.PREFIX_MANIFEST_SHA256,payload_sha256=inv.PREFIX_PAYLOAD_SHA256,
            selection_replaced_or_filtered=False))
    return doc

def test_full_declared_contract_replays_without_Torch(valid_document):
    assert inv.validate_inventory(valid_document)["original_training_rows"]==12288

@pytest.mark.parametrize("change",[
    lambda d:d.update(torch_imported=True),
    lambda d:d.update(matching_backend_conditionals_qualified=True),
    lambda d:d.update(posterior_accuracy_assessed=True),
    lambda d:d.update(banks_unchanged=False),
    lambda d:d["RNG_after"].update(numpy="different"),
    lambda d:d["banks"][2].update(context_seed=20261112),
    lambda d:d["banks"][0].update(count=4095),
    lambda d:d["banks"][0]["domain"]["endpoint_or_outside_counts"].update({"6":1}),
    lambda d:d["banks"][0]["qualification"]["report"]["error_to_declared_bound_max_per_coordinate"]["physical64"].__setitem__(3,1.0001),
    lambda d:d["banks"][0]["qualification"]["ordered_model_Float32"].update(dtype="<f8"),
    lambda d:d["banks"][0]["qualification"]["report"].update(no_rows_removed=False),
    lambda d:d["case2_fixed_prefix"]["custody"]["selected_indices"].__setitem__(0,49999),
    lambda d:d["case2_fixed_prefix"]["custody"].update(selection_replaced_or_filtered=True),
    lambda d:d["source_pins"].update(codec_module_sha256="0"*64),
])
def test_changed_domain_sources_prefix_or_claim_refused(valid_document,change):
    doc=copy.deepcopy(valid_document);change(doc)
    with pytest.raises(ValueError):inv.validate_inventory(doc)

def test_wrong_prefix_bytes_refused_before_array_loading(tmp_path):
    (tmp_path/"manifest.json").write_text("{}")
    with pytest.raises(ValueError,match="exact original"):inv.load_prefix(tmp_path,None,None)

def test_matrix_failure_does_not_remove_endpoint_row():
    values=toy_rows(5);values[3,6]=.01;before=values.tobytes()
    assert inv.domain_counts(values)["endpoint_or_outside_counts"]["6"]==1
    with pytest.raises(codec.CoordinateError):inv.qualify_matrix(values)
    assert values.tobytes()==before

def test_no_Torch_import_on_help():
    result=subprocess.run([sys.executable,str(ROOT/"scripts/qualify_prior_coordinates_v1.py"),"--help"],
        cwd=ROOT,check=True,capture_output=True,text=True)
    assert "CPU-only" in result.stdout

def test_frozen_row_rule_is_first1024_not_old_linspace():
    assert inv.PREFIX_INDICES==tuple(range(1024))
    assert inv.PREFIX_INDICES!=tuple(np.linspace(0,49999,1024,dtype=int))

def test_codec_report_json_has_separate_fixed_bounds():
    report=codec.roundtrip_report(toy_rows(3));json.dumps(report,allow_nan=False)
    assert set(report["error_to_declared_bound_max_per_coordinate"])=={
        "physical64","model32_physical","latent64","model32_latent","final_model_cast"}
