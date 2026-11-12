#!/usr/bin/env python3
"""One frozen joint-global refresh of a pure sigma-first SS15 physical cohort.

Built with PriorLabs-TabPFN. This small CPU wrapper reuses the exact a995
refresh_once implementation and e346 prior-only kernel, without editing either.
It verifies the new parent/order provenance; it never loads observations,
references, truth, banks, a neural model or a local likelihood update.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import subprocess
import numpy as np

from posteriorforge.artifacts import runtime_identity, sha256_file, write_json
from posteriorforge.science.analytical_benchmark_v2 import array_descriptor

BACKBONE_SHA = "a995ab5e48fab6ddea8355344d9f77fe94bd30357d6451370daf69ed5693f8a4"
KERNEL_SHA = "e346e25bc9825abf016c2bd75addaf2b025d3e3d94e1a0ea6f6e31dd02997a43"
CONTRACT_PATH = "docs/accuracy-ss15-sigma-first-v1.md"
CONTRACT_SHA = "5b7a31237c9a3e35db27f3f9697c7a9293b0738a8ec8b12d63f370d4ad27e3d4"
CONTRACT_ORIGIN = "f9437ee"
PARENT_SOURCE_PATH = "src/posteriorforge/performance_ss15_sigma_first.py"
ORDER_SOURCE_PATH = "src/posteriorforge/execution_ss15_sigma_first.py"
PARENT_SOURCE_SHA = "128c200a82f1252b2ad971ed43f10f534b16b9e02f8882cfd62d222cb88442cd"
ORDER_SOURCE_SHA = "d48d1638ef9e540bdd629d321fba657af4db757700b1092fc58fb9ded1f79ad2"
PARENT_SCHEMA = "posteriorforge.inference.ss15_sigma_first.v1"
PARENT_VARIANT = "accuracy_ss15_sigma_first_fp16_fifo_v1"
SCHEMA = "posteriorforge.ss15_noise_first_global_block_refresh.v1"
VARIANT = "hybrid_ss15_sigma_first_global_block_gibbs_once_v1"
ROLE = "ss15_global_block_gibbs_once_v1"
COUNT = 50000
ORDER = (0,1,2,6,3,4,5,10,7,8,9,14,11,12,13,16,15,18,17,20,19)
INVERSE_ORDER = tuple(int(i) for i in np.argsort(ORDER))


def _load_backbone():
    path = Path(__file__).with_name("refresh_ss15_global_block.py")
    if sha256_file(path) != BACKBONE_SHA:
        raise ValueError("frozen joint-global producer source changed")
    spec = importlib.util.spec_from_file_location("_ss15_frozen_joint_global_refresh", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


backbone = _load_backbone()
require = backbone.require
kernel = backbone.kernel
refresh_once = backbone.refresh_once
_SEALS = {name: (getattr(backbone,name), getattr(backbone,name).__code__)
          for name in ("refresh_once","source_guard","git_bytes","state_sha","json_sha")}


def require_backbone():
    require(sha256_file(backbone.__file__) == BACKBONE_SHA
            and kernel.source_sha256() == KERNEL_SHA
            and backbone.KERNEL_SHA == KERNEL_SHA and backbone.ROLE == ROLE
            and backbone.COUNT == COUNT, "frozen joint kernel/role/source changed")
    for name,(function,code) in _SEALS.items():
        require(getattr(backbone,name) is function and function.__code__ is code,
                "frozen joint producer callable changed: "+name)
    require(refresh_once is backbone.refresh_once, "joint refresh alias replaced")


def source_guard():
    require_backbone()
    require(backbone.digest(PARENT_SOURCE_SHA) and backbone.digest(ORDER_SOURCE_SHA),
            "pure sigma-first parent/order source is not frozen")
    root = Path(__file__).resolve().parents[1]
    old = backbone.source_guard()
    source = Path(__file__).resolve()
    require(sha256_file(root/CONTRACT_PATH) == CONTRACT_SHA, "sigma-first preregistration changed")
    origin = subprocess.run(["git","-C",str(root),"rev-parse",CONTRACT_ORIGIN],
                            check=True,capture_output=True,text=True).stdout.strip()
    require(hashlib.sha256(backbone.git_bytes(root,origin,CONTRACT_PATH)).hexdigest() == CONTRACT_SHA,
            "pre-cohort sigma-first preregistration Git bytes differ")
    require(hashlib.sha256(backbone.git_bytes(root,old["git_revision"],str(source.relative_to(root)))).hexdigest()
            == sha256_file(source), "new wrapper must be committed before cohort execution")
    return {"producer_sha256":sha256_file(source),"git_revision":old["git_revision"],
            "frozen_joint_refresh_sha256":BACKBONE_SHA,"kernel_sha256":KERNEL_SHA,
            "joint_source":old,"sigma_first_contract_sha256":CONTRACT_SHA,
            "sigma_first_contract_origin_commit":origin,
            "producer_backbone_kernel_and_contract_committed_before_run":True}


def _descriptor(value, shape, dtype):
    return (isinstance(value,dict) and value.get("shape")==shape
            and isinstance(value.get("dtype"),str)
            and np.dtype(value["dtype"])==np.dtype(dtype)
            and backbone.digest(value.get("sha256")))


def validate_parent(manifest, values, metadata):
    require(backbone.digest(PARENT_SOURCE_SHA) and backbone.digest(ORDER_SOURCE_SHA),
            "pure sigma-first parent/order source is not frozen")
    expected={"schema":PARENT_SCHEMA,"variant":PARENT_VARIANT,"status":"complete",
        "runner_source_sha256":PARENT_SOURCE_SHA,"requested_count":COUNT,
        "context_seed":20261111,"case":0,"context_size":4096,"repetition":0,
        "sources_unchanged":True,"default_rng_preserved":True,
        "parameter_order":metadata["physical_parameter_order"],
        "scientific_fingerprint":metadata["scientific_fingerprint"],
        "physical_parameter_order_unchanged":True,"learned_target_order_changed":True,
        "model_target_order":"sigma_first_stage_order",
        "stage_to_physical_order":list(ORDER),"physical_to_stage_order":list(INVERSE_ORDER),
        "samples_restored_to_physical_order_before_FIFO_support":True,
        "completed_stage_indices":list(range(21)),
        "clipping_or_repair_performed":False,"partial_vector_filtering":False,
        "preregistration_sha256":CONTRACT_SHA,
        "actual_sigma_first_fixed_conditional_screen_performed":False}
    for name,value in expected.items():
        require(type(manifest.get(name)) is type(value) and manifest[name]==value,
                "pure sigma-first parent binding differs: "+name)
    require(isinstance(values,np.ndarray) and values.dtype==np.dtype("float64")
            and values.shape==(COUNT,21) and array_descriptor(values)==manifest.get("samples"),
            "strict50000 complete original-order Float64 samples required")
    raw=manifest.get("raw_scientific_bindings",{})
    require(set(raw)=={"checkpoint_sha256","scientific_fingerprint","parameter_order",
                       "context_theta","context_obs","query"}
            and all(manifest.get(name)==value for name,value in raw.items()),
            "all six inherited physical scientific bindings required")
    require(raw["checkpoint_sha256"]==backbone.CHECKPOINT_SHA
            and raw["scientific_fingerprint"]==metadata["scientific_fingerprint"]
            and raw["parameter_order"]==metadata["physical_parameter_order"],
            "physical problem/checkpoint/order differs")
    for name,shape in (("context_theta",[4096,21]),("context_obs",[4096,33]),("query",[1,33])):
        require(_descriptor(raw[name],shape,"float64"),"physical raw descriptor differs: "+name)
    model=manifest.get("model_inputs",{})
    for name,shape in (("context_theta",[4096,21]),("context_obs",[4096,15]),("query",[1,15])):
        require(_descriptor(model.get(name),shape,"float32"),"ordered model descriptor differs: "+name)
    receipt=manifest.get("model_target_descriptors",{})
    require(receipt.get("ordered_theta_float32")==model["context_theta"]
            and _descriptor(receipt.get("physical_theta_float32"),[4096,21],"float32")
            and receipt.get("stage_to_physical_order")==list(ORDER)
            and receipt.get("ordered_theta_equals_physical_theta_columns_pi") is True
            and receipt.get("global_used_predictors_labels_byte_identical") is True
            and receipt.get("target_coordinate_system")=="sigma_first_stage_order",
            "source-bound physical/ordered training target receipt differs")
    anchor=manifest.get("conditional_anchor",{})
    require(anchor.get("qualifies_sigma_first_inputs") is False
            and anchor.get("raw_scientific_bindings")==raw
            and anchor.get("unmasked_model_inputs",{}).get("context_theta")==receipt["physical_theta_float32"],
            "original-order finite anchor/physical target receipt was conflated")
    policy=manifest.get("sigma_first_input_policy",{})
    require(policy.get("module_sha256")==ORDER_SOURCE_SHA
            and backbone.digest(policy.get("policy_sha256"))
            and policy.get("stage_to_physical_order")==list(ORDER)
            and policy.get("physical_to_stage_order")==list(INVERSE_ORDER)
            and policy.get("physical_output_order_restored") is True,
            "sigma-first source/order policy differs")
    expected_fallback=[{"stage":stage,"member":member,"target_physical_index":ORDER[stage],
        "target_physical_name":metadata["physical_parameter_order"][ORDER[stage]],
        "coordinate_system":"sigma-first stage index"} for stage,member in ((10,5),(20,4))]
    require(manifest.get("fallback_stage_member_bindings")==expected_fallback
            and policy.get("fallback_stage_member_bindings")==expected_fallback,
            "stage fallback physical-target identity differs")
    fifo=manifest.get("proposal_buffer",{})
    scope=fifo.get("request_scope",{})
    require(fifo.get("finalized") is True and fifo.get("failed") is False
            and scope.get("stage_to_physical_order")==list(ORDER)
            and scope.get("physical_to_stage_order")==list(INVERSE_ORDER)
            and scope.get("samples_restored_before_FIFO_support") is True,
            "physical restoration before original FIFO/support is not recorded")
    require(metadata["target_indices"]==[0,1,2] and metadata["unchanged_indices"]==list(range(3,21)),
            "unchanged joint-global kernel required")
    kernel.conditional_parameters(values)
    return {"stage_to_physical_order":list(ORDER),"physical_to_stage_order":list(INVERSE_ORDER),
            "physical_output_restored_before_original_FIFO_support":True,
            "ordered_target_receipt":receipt,
            "target_receipt_scope":"authenticated parent assertions/source; bank arrays are not reloaded"}


def load_parent(path, expected_sha256, metadata):
    require(backbone.digest(expected_sha256),"explicit expected parent manifest SHA256 required")
    path=Path(path).resolve();raw=path.read_bytes()
    require(hashlib.sha256(raw).hexdigest()==expected_sha256,"parent manifest hash differs")
    manifest=json.loads(raw);require(manifest.get("samples_file")=="samples.npz","fixed samples filename required")
    payload=path.with_name("samples.npz")
    require(backbone.digest(manifest.get("samples_sha256"))
            and sha256_file(payload)==manifest["samples_sha256"],"parent payload hash differs")
    with np.load(payload,allow_pickle=False) as archive:
        require(archive.files==["samples"],"complete physical samples only")
        values=archive["samples"]
    order_receipt=validate_parent(manifest,values,metadata)
    root=Path(__file__).resolve().parents[1];revision=manifest.get("git_revision")
    for filename,expected in ((PARENT_SOURCE_PATH,PARENT_SOURCE_SHA),(ORDER_SOURCE_PATH,ORDER_SOURCE_SHA),
                              (CONTRACT_PATH,CONTRACT_SHA)):
        require(hashlib.sha256(backbone.git_bytes(root,revision,filename)).hexdigest()==expected,
                "executed pure sigma-first source/order/contract Git bytes differ")
    return manifest,values,{"manifest_sha256":expected_sha256,"samples_sha256":manifest["samples_sha256"],
        "samples":array_descriptor(values),"producer_sha256":PARENT_SOURCE_SHA,
        "order_module_sha256":ORDER_SOURCE_SHA,"git_revision":revision,"order_validation":order_receipt}


def run(parent_manifest, expected_parent_sha256, output):
    before=backbone.state_sha((np.random.get_state(),random.getstate()))
    sources=source_guard();metadata=kernel.metadata()
    parent,values,lineage=load_parent(parent_manifest,expected_parent_sha256,metadata)
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    record={"schema":SCHEMA,"variant":VARIANT,"status":"running","attribution":"Built with PriorLabs-TabPFN",
        "method":"pure sigma-first NPE parent plus ONE unchanged correlated3-global Gibbs block",
        "requested_count":COUNT,"context_seed":parent["context_seed"],"case":parent["case"],"repetition":0,
        "parameter_order":metadata["physical_parameter_order"],"scientific_fingerprint":metadata["scientific_fingerprint"],
        "parent":lineage,"source_pins":sources,"conditional_kernel":metadata,"runtime":runtime_identity(),
        "raw_scientific_bindings":parent["raw_scientific_bindings"],
        "raw_scientific_bindings_scope":"inherited provenance; no bound data files loaded",
        "pure_TFM_inference":False,"posterior_accuracy_guaranteed":False,
        "observations_references_truth_or_banks_loaded":False,
        "noise_or_local_likelihood_update_performed":False,
        "support_filtering_clipping_redraw_or_trimming":False,
        "end_to_end_posterior_request_timing_claimed":False}
    write_json(output/"manifest.json",record)
    try:
        seed=backbone.derive_seed(parent["context_seed"],ROLE,case=parent["case"],repetition=0)
        result,receipt=refresh_once(values,seed)
        require(receipt["role"]==ROLE and receipt["standard_normal_draw_shape"]==[COUNT,3],
                "frozen joint refresh policy differs")
        with (output/"samples.npz").open("xb") as stream:np.savez(stream,samples=result)
        require(source_guard()==sources,"source/contract changed during refresh")
        require(sha256_file(parent_manifest)==expected_parent_sha256
                and sha256_file(Path(parent_manifest).with_name("samples.npz"))==lineage["samples_sha256"],
                "pure parent changed during refresh")
        after=backbone.state_sha((np.random.get_state(),random.getstate()))
        require(before==after,"wrapper changed default NumPy/Python RNG")
        record.update(status="complete",samples=array_descriptor(result),samples_file="samples.npz",
            samples_sha256=sha256_file(output/"samples.npz"),refresh=receipt,sources_unchanged=True,
            default_RNG_observation={"before_sha256":before,"after_sha256":after,"preserved":True,
                "observed_generators":["NumPy legacy default","Python random"],
                "torch_CPU_CUDA_RNG_hashes_observed":False})
    except BaseException as error:
        record.update(status="failed",error_type=type(error).__name__,error=str(error))
        write_json(output/"manifest.json",record);raise
    write_json(output/"manifest.json",record)
    write_json(output/"checksums.json",{"manifest.json":sha256_file(output/"manifest.json"),
                                       "samples.npz":record["samples_sha256"]})
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-manifest",type=Path,required=True)
    parser.add_argument("--expected-parent-sha256",required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();record=run(args.parent_manifest,args.expected_parent_sha256,args.output)
    print(json.dumps({"status":record["status"],"returned":record["samples"]["shape"][0],
                      "variant":VARIANT,"samples_sha256":record["samples_sha256"]},sort_keys=True))


if __name__=="__main__":
    main()
