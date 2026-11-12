"""Pure engineering fixtures for the new parent/order wrapper; no real cohort refreshed."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from posteriorforge.artifacts import sha256_file
from posteriorforge.science.analytical_benchmark_v2 import array_descriptor

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("sigma_first_joint_wrapper",
    ROOT/"scripts/refresh_ss15_noise_first_global_block.py")
wrapper=importlib.util.module_from_spec(spec);spec.loader.exec_module(wrapper)


def values():
    data=np.repeat(np.arange(21,dtype=np.float64)[None]/20,50000,axis=0)
    data+=np.linspace(-.1,.1,50000)[:,None]
    data[:,wrapper.kernel.metadata()["noise_indices"]]=.25
    return data


def freeze_fake_parent(monkeypatch):
    # Fake source IDs are confined to unit fixtures; never saved as scientific proof.
    monkeypatch.setattr(wrapper,"PARENT_SOURCE_SHA","a"*64)
    monkeypatch.setattr(wrapper,"ORDER_SOURCE_SHA","b"*64)


def manifest(data):
    meta=wrapper.kernel.metadata()
    raw={"checkpoint_sha256":wrapper.backbone.CHECKPOINT_SHA,
         "scientific_fingerprint":meta["scientific_fingerprint"],
         "parameter_order":meta["physical_parameter_order"],
         "context_theta":{"dtype":"<f8","shape":[4096,21],"sha256":"c"*64},
         "context_obs":{"dtype":"<f8","shape":[4096,33],"sha256":"d"*64},
         "query":{"dtype":"<f8","shape":[1,33],"sha256":"e"*64}}
    physical={"dtype":"<f4","shape":[4096,21],"sha256":"f"*64}
    ordered={"dtype":"<f4","shape":[4096,21],"sha256":"1"*64}
    fallbacks=[{"stage":j,"member":m,"target_physical_index":wrapper.ORDER[j],
                "target_physical_name":meta["physical_parameter_order"][wrapper.ORDER[j]],
                "coordinate_system":"sigma-first stage index"} for j,m in ((10,5),(20,4))]
    return {"schema":wrapper.PARENT_SCHEMA,"variant":wrapper.PARENT_VARIANT,"status":"complete",
        "runner_source_sha256":wrapper.PARENT_SOURCE_SHA,"requested_count":50000,
        "context_seed":20261111,"case":0,"context_size":4096,"repetition":0,
        "sources_unchanged":True,"default_rng_preserved":True,"completed_stage_indices":list(range(21)),
        "physical_parameter_order_unchanged":True,"learned_target_order_changed":True,
        "model_target_order":"sigma_first_stage_order","stage_to_physical_order":list(wrapper.ORDER),
        "physical_to_stage_order":list(wrapper.INVERSE_ORDER),
        "samples_restored_to_physical_order_before_FIFO_support":True,
        "clipping_or_repair_performed":False,"partial_vector_filtering":False,
        "preregistration_sha256":wrapper.CONTRACT_SHA,
        "actual_sigma_first_fixed_conditional_screen_performed":False,
        "raw_scientific_bindings":raw,**raw,"samples":array_descriptor(data),
        "model_inputs":{"context_theta":ordered,"context_obs":{"dtype":"<f4","shape":[4096,15],"sha256":"2"*64},
                        "query":{"dtype":"<f4","shape":[1,15],"sha256":"3"*64}},
        "model_target_descriptors":{"physical_theta_float32":physical,"ordered_theta_float32":ordered,
            "stage_to_physical_order":list(wrapper.ORDER),
            "ordered_theta_equals_physical_theta_columns_pi":True,
            "global_used_predictors_labels_byte_identical":True,
            "target_coordinate_system":"sigma_first_stage_order"},
        "conditional_anchor":{"qualifies_sigma_first_inputs":False,"raw_scientific_bindings":raw,
                              "unmasked_model_inputs":{"context_theta":physical}},
        "sigma_first_input_policy":{"module_sha256":wrapper.ORDER_SOURCE_SHA,"policy_sha256":"4"*64,
            "stage_to_physical_order":list(wrapper.ORDER),"physical_to_stage_order":list(wrapper.INVERSE_ORDER),
            "physical_output_order_restored":True,"fallback_stage_member_bindings":fallbacks},
        "fallback_stage_member_bindings":fallbacks,
        "proposal_buffer":{"finalized":True,"failed":False,"request_scope":{
            "stage_to_physical_order":list(wrapper.ORDER),"physical_to_stage_order":list(wrapper.INVERSE_ORDER),
            "samples_restored_before_FIFO_support":True}}}


def test_no_new_math_or_rng_role_and_backbone_source_function_identity():
    wrapper.require_backbone()
    assert wrapper.refresh_once is wrapper.backbone.refresh_once
    assert wrapper.refresh_once.__code__ is wrapper._SEALS["refresh_once"][1]
    assert wrapper.ROLE=="ss15_global_block_gibbs_once_v1" and wrapper.COUNT==50000
    assert wrapper.kernel.source_sha256()==wrapper.KERNEL_SHA
    assert sha256_file(wrapper.backbone.__file__)==wrapper.BACKBONE_SHA


def test_exact_order_inverse_and_new_parent_metadata_validate_without_bank_reads(monkeypatch):
    freeze_fake_parent(monkeypatch);data=values();parent=manifest(data)
    proof=wrapper.validate_parent(parent,data,wrapper.kernel.metadata())
    assert proof["physical_output_restored_before_original_FIFO_support"]
    assert list(np.array(wrapper.ORDER)[list(wrapper.INVERSE_ORDER)])==list(range(21))
    assert proof["ordered_target_receipt"]==parent["model_target_descriptors"]
    assert "bank arrays are not reloaded" in proof["target_receipt_scope"]


@pytest.mark.parametrize("fault",["oldpure","scalar","oldjoint","source","order","inverse","restore","savedorder",
    "ordered_descriptor","physical_anchor","receipt","fallback","FIFO","policy","raw","case","quota","float32","support"])
def test_wrong_parent_order_restoration_source_or_input_is_rejected(monkeypatch,fault):
    freeze_fake_parent(monkeypatch);data=values();parent=manifest(data)
    if fault=="oldpure":parent["schema"]="posteriorforge.inference.ss15_fp16.v1"
    elif fault=="scalar":parent["schema"]="posteriorforge.ss15_mu_a_refresh.v1"
    elif fault=="oldjoint":parent["schema"]="posteriorforge.ss15_global_block_refresh.v1"
    elif fault=="source":parent["runner_source_sha256"]="0"*64
    elif fault=="order":parent["stage_to_physical_order"]=list(range(21))
    elif fault=="inverse":parent["physical_to_stage_order"]=list(range(21))
    elif fault=="restore":parent["samples_restored_to_physical_order_before_FIFO_support"]=False
    elif fault=="savedorder":parent["physical_parameter_order_unchanged"]=False
    elif fault=="ordered_descriptor":parent["model_target_descriptors"]["ordered_theta_float32"]={"wrong":"descriptor"}
    elif fault=="physical_anchor":parent["conditional_anchor"]["unmasked_model_inputs"]["context_theta"]={"wrong":"descriptor"}
    elif fault=="receipt":parent["model_target_descriptors"]["ordered_theta_equals_physical_theta_columns_pi"]=False
    elif fault=="fallback":
        parent["fallback_stage_member_bindings"][0]["target_physical_index"]=10
    elif fault=="FIFO":parent["proposal_buffer"]["request_scope"]["samples_restored_before_FIFO_support"]=False
    elif fault=="policy":parent["sigma_first_input_policy"]["module_sha256"]="0"*64
    elif fault=="raw":parent["raw_scientific_bindings"]["query"]["sha256"]="not-a-sha"
    elif fault=="case":parent["case"]=1
    elif fault=="quota":data=data[:49999];parent["samples"]=array_descriptor(data)
    elif fault=="float32":data=data.astype(np.float32);parent["samples"]=array_descriptor(data)
    else:data[0,6]=.001;parent["samples"]=array_descriptor(data)
    with pytest.raises(ValueError):wrapper.validate_parent(parent,data,wrapper.kernel.metadata())


def test_missing_source_pins_fail_before_parent_io_or_output_and_backbone_mutation_rejected(tmp_path,monkeypatch):
    monkeypatch.setattr(wrapper,"PARENT_SOURCE_SHA",None)
    monkeypatch.setattr(wrapper,"ORDER_SOURCE_SHA",None)
    with pytest.raises(ValueError,match="not frozen"):
        wrapper.run(tmp_path/"missing","0"*64,tmp_path/"out")
    assert not (tmp_path/"out").exists()
    monkeypatch.setattr(wrapper.backbone,"refresh_once",lambda *args:None)
    with pytest.raises(ValueError,match="callable changed"):wrapper.require_backbone()


def test_expected_sha_is_required_before_payload_access(tmp_path):
    path=tmp_path/"manifest.json";path.write_text("{}")
    for sha in (None,"","bad","0"*64):
        with pytest.raises(ValueError):wrapper.load_parent(path,sha,wrapper.kernel.metadata())


def test_toy_run_calls_the_frozen_joint_refresh_once_with_same_role_and_retains18bytes(tmp_path,monkeypatch):
    freeze_fake_parent(monkeypatch);data=values();before=data.copy();parent=manifest(data)
    parent_path=tmp_path/"parent.json";parent_path.write_text('{"fixture":true}')
    payload=tmp_path/"samples.npz"
    with payload.open("xb") as stream:np.savez(stream,samples=data)
    expected=sha256_file(parent_path)
    lineage={"samples_sha256":sha256_file(payload),"manifest_sha256":expected}
    monkeypatch.setattr(wrapper,"source_guard",lambda:{"fixture_source":"not-scientific-evidence"})
    monkeypatch.setattr(wrapper,"load_parent",lambda *args:(parent,data,lineage))
    calls=[];original=wrapper.refresh_once
    def spy(value,seed):
        calls.append(seed);return original(value,seed)
    monkeypatch.setattr(wrapper,"refresh_once",spy)
    result=wrapper.run(parent_path,expected,tmp_path/"out")
    seed=wrapper.backbone.derive_seed(20261111,wrapper.ROLE,case=0,repetition=0)
    assert calls==[seed] and result["status"]=="complete" and result["requested_count"]==50000
    with np.load(tmp_path/"out/samples.npz",allow_pickle=False) as saved:out=saved["samples"]
    assert out[:,3:].tobytes()==before[:,3:].tobytes() and data.tobytes()==before.tobytes()
    assert result["refresh"]["standard_normal_draw_shape"]==[50000,3]
    assert result["refresh"]["role"]==wrapper.ROLE and result["refresh"]["all18_columns_byte_identical"]
    assert not result["observations_references_truth_or_banks_loaded"]
    assert not result["noise_or_local_likelihood_update_performed"]


def test_import_and_cli_are_torch_free_and_have_only_parent_sha_output_arguments():
    source=ROOT/"scripts/refresh_ss15_noise_first_global_block.py"
    code=("import importlib.util,sys;s=importlib.util.spec_from_file_location('r',"+repr(str(source))+");"
          "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.require_backbone();"
          "assert 'torch' not in sys.modules")
    subprocess.run([sys.executable,"-c",code],check=True)
    result=subprocess.run([sys.executable,str(source),"--help"],check=True,capture_output=True,text=True)
    assert "--parent-manifest" in result.stdout and "--expected-parent-sha256" in result.stdout
    assert "--reference" not in result.stdout and "--observation" not in result.stdout
    assert "--banks" not in result.stdout and "--truth" not in result.stdout
