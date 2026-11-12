#!/usr/bin/env python3
"""One frozen joint-global Gibbs block on saved, decoded prior-coordinate rows.

Built with PriorLabs-TabPFN. Loads only parent artifacts, not observations,
references, truth or banks. No model call or coordinate decode after G3.
Run CPU postprocessing in the actual producer inference environment; exact
Python/NumPy/SciPy identity is required before full-wave decode and G3.
"""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,random
from pathlib import Path
import numpy as np
from posteriorforge.artifacts import runtime_identity,sha256_file,write_json,versions
from posteriorforge.science.analytical_benchmark_v2 import array_descriptor
from posteriorforge import prior_coordinates_v1 as codec

SCHEMA="posteriorforge.prior_coordinates_global_block_refresh.v1"
VARIANT="hybrid_prior_coordinates_cached_fp32_fifo_global_block_gibbs_v1"
PARENT_SCHEMA="posteriorforge.inference.prior_coordinates.v1"
PARENT_VARIANT="accuracy_prior_coordinates_cached_fp32_fifo_v1"
PARENT_SOURCE_SHA256=None
BACKEND_SHA256=None
CODEC_SHA256="d8ce6c49e7b65c6ee55c59b7243d4ce943cbea28c92ebed0329798074aaa70c5"
PREREGISTRATION_SHA256="7c4aca04c4571960108c0296ec5d4770940ce0b51fa0a77cdd0c27182b4ed849"
BACKBONE_SHA256="a995ab5e48fab6ddea8355344d9f77fe94bd30357d6451370daf69ed5693f8a4"
KERNEL_SHA256="e346e25bc9825abf016c2bd75addaf2b025d3e3d94e1a0ea6f6e31dd02997a43"
COUNT=50000
ROLE="ss15_global_block_gibbs_once_v1"


def load_backbone():
    path=Path(__file__).with_name("refresh_ss15_global_block.py")
    if sha256_file(path)!=BACKBONE_SHA256:raise ValueError("frozen joint refresh changed")
    spec=importlib.util.spec_from_file_location("_prior_coordinates_joint_backbone",path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


backbone=load_backbone()
kernel=backbone.kernel
require=backbone.require
refresh_once=backbone.refresh_once
_SEALS={name:(getattr(backbone,name),getattr(backbone,name).__code__)
        for name in ("refresh_once","source_guard","git_bytes","state_sha")}


def source_guard():
    require(sha256_file(backbone.__file__)==BACKBONE_SHA256
            and kernel.source_sha256()==KERNEL_SHA256 and backbone.ROLE==ROLE
            and backbone.COUNT==COUNT,"frozen joint source/kernel/ROLE changed")
    for name,(function,code) in _SEALS.items():
        require(getattr(backbone,name) is function and function.__code__ is code,
                "frozen joint callable changed: "+name)
    require(refresh_once is backbone.refresh_once,"joint refresh alias replaced")
    require(backbone.digest(PARENT_SOURCE_SHA256) and backbone.digest(BACKEND_SHA256),
            "new parent/backend source not pinned; refresh disabled")
    root=Path(__file__).resolve().parents[1]
    old=backbone.source_guard()
    sources=[
        (str(Path(__file__).resolve().relative_to(root)),sha256_file(__file__)),
        ("src/posteriorforge/performance_prior_coordinates_v1.py",PARENT_SOURCE_SHA256),
        ("src/posteriorforge/execution_prior_coordinates_v1.py",BACKEND_SHA256),
        ("src/posteriorforge/prior_coordinates_v1.py",CODEC_SHA256),
        ("docs/accuracy-prior-coordinate-pilot-v1.md",PREREGISTRATION_SHA256),
    ]
    for name,digest in sources:
        require(sha256_file(root/name)==digest
                and hashlib.sha256(backbone.git_bytes(root,old["git_revision"],name)).hexdigest()==digest,
                "new source must be unchanged and committed before refresh: "+name)
    return {"producer_sha256":sha256_file(__file__),"git_revision":old["git_revision"],
            "parent_producer_sha256":PARENT_SOURCE_SHA256,"backend_sha256":BACKEND_SHA256,
            "codec_sha256":CODEC_SHA256,"preregistration_sha256":PREREGISTRATION_SHA256,
            "frozen_joint_refresh_sha256":BACKBONE_SHA256,"kernel_sha256":KERNEL_SHA256,
            "joint_source":old,"sources_committed_before_run":True}


def require_replay_environment(document):
    """Exact F64 replay is scoped to the actual producer software versions."""
    keys=("python","numpy","scipy")
    recorded=document.get("versions",{})
    current=versions()
    require(all(isinstance(recorded.get(k),str) and recorded[k] for k in keys),
            "actual producer Python/NumPy/SciPy versions required")
    expected={key:recorded[key] for key in keys}
    actual={key:current.get(key) for key in keys}
    require(actual==expected,
            "exact decode/G3 replay must run in the actual producer inference environment")
    return {"passed":True,"producer_versions":expected,"replay_versions":actual,
            "exact_same_environment_identity":True,
            "global_cross_environment_F64_identity_claimed":False}


def validate_parent(document,values,metadata):
    require(backbone.digest(PARENT_SOURCE_SHA256) and backbone.digest(BACKEND_SHA256),
            "new parent/backend source not pinned")
    expected={
        "schema":PARENT_SCHEMA,"variant":PARENT_VARIANT,"status":"complete",
        "runner_source_sha256":PARENT_SOURCE_SHA256,"context_seed":20261111,"case":2,
        "requested_count":COUNT,"context_size":4096,"repetition":0,"rank":0,"world_size":1,
        "sources_unchanged":True,"default_rng_preserved":True,
        "mode":"fp32_transparent_control_v1","fallback_map":{},
        "FP16_activated":False,"learned_targets_are_original_physical_values":False,
        "samples_in_original_physical_order_before_FIFO_support":True,
        "all_generated_rows_strictly_decoded_including_unused_FIFO_tail":True,
        "model_coordinate_decode_after_global_refresh":False,
        "preregistration_sha256":PREREGISTRATION_SHA256,
        "sampler_math_changed":False,"partial_vector_filtering":False,
        "clipping_or_repair_performed":False,"physical_log_density_claimed":False,
        "completed_stage_indices":list(range(21)),
    }
    for name,value in expected.items():
        require(document.get(name)==value,"new parent field differs: "+name)
    require(values.dtype==np.float64 and values.shape==(COUNT,21)
            and np.isfinite(values).all(),"supported complete Float64[50000,21] required")
    require(document.get("samples")==array_descriptor(values),"parent physical array descriptor differs")
    require(document.get("scientific_fingerprint")==metadata["scientific_fingerprint"]
            and document.get("parameter_order")==metadata["physical_parameter_order"],
            "original physical scientific identity differs")
    require(document.get("codec")==codec.metadata(),"coordinate source/geometry descriptor differs")
    require(document.get("coordinate_policy",{}).get("module_sha256")==BACKEND_SHA256,
            "new backend coordinate policy differs")
    require(document.get("stage_to_physical_order")==list(codec.ORDER)
            and document.get("physical_to_stage_order")==list(codec.INVERSE_ORDER),
            "latent stage/physical order differs")
    execution=document.get("execution_counters",{})
    require(execution.get("all_stage_aliases_restored") is True
            and execution.get("prepare_count")==21
            and len(execution.get("completed_stage_scopes",[]))==21,
            "actual stage ownership/restoration is incomplete")
    compat=document.get("compatibility_adapter",{})
    require(compat.get("original_alias_restored") is True and compat.get("sdk_sources_unchanged") is True,
            "CDF/source restoration differs")
    require(document.get("default_rng_before")==document.get("default_rng_after"),
            "actual default RNG hashes differ")
    require(document.get("matching_cached_FP32_execution_passed_all21") is True
            and backbone.digest(document.get("matching_conditional_qualification_sha256"))
            and document.get("fresh_same_policy_all21_screen_replayed") is True
            and document.get("actual_fixed_conditional_screen_performed_during_request") is False,
            "fresh same-policy finite-screen receipt differs")
    return {"passed":True,"same_physical_order":True,"new_latent_family_only":True,
            "request_count":COUNT,"separate_joint_update":True}


def replay_generated_decode(document,latent_waves,physical_values):
    """Replay strict decode on all generated rows, then recorded checked prefix."""
    from posteriorforge.inference import support_mask
    environment=require_replay_environment(document)
    keys=document["latent_wave_archive"]["keys"]
    require(keys==[f"wave_{j:04d}" for j in range(len(keys))] and len(keys)>0,
            "ordered generated-wave keys required")
    require(set(latent_waves)==set(keys),"latent archive keys differ")
    records=document["decode_wave_records"]
    require(len(records)==len(keys)==len(document["proposal_wave_sizes"]),
            "all generated waves must be journaled")
    decoded=[]
    for index,key in enumerate(keys):
        latent=latent_waves[key]
        require(latent.dtype==np.float32 and latent.ndim==2 and latent.shape[1]==21,
                "actual Float32 latent wave required without a cast")
        require(records[index]["wave_index"]==index
                and records[index]["row_count"]==len(latent)
                and records[index]["actual_latent_draws"]==array_descriptor(latent),
                "actual generated latent descriptor differs")
        physical=codec.decode_ordered(latent)
        require(records[index]["decoded_physical"]==array_descriptor(physical)
                and records[index]["strict_decode_passed"] is True
                and records[index]["includes_unused_FIFO_tail"] is True
                and records[index]["input_dtype_unchanged_before_decode"] is True,
                "full-wave strict physical decode differs")
        require(support_mask(physical).all(),"strict decode produced unsupported physical row")
        decoded.append(physical)
    all_physical=np.concatenate(decoded,axis=0)
    reject=document["rejection"]
    checked=reject["proposed"]
    # Strict decoder permits no invalid physical vector to reach the old support.
    require(reject["support_calls"]==1 and checked==reject["accepted_before_final_trim"]==COUNT
            and reject["returned"]==COUNT and len(decoded)==1
            and len(all_physical)==COUNT+1200,"strict decode/FIFO/support conservation differs")
    require(all_physical[:COUNT].tobytes()==physical_values.tobytes(),
            "saved physical cohort differs from decoded accepted prefix")
    require(document["latent_wave_archive"]["generated_row_count"]==len(all_physical)
            and document["latent_wave_archive"]["all_generated_rows_including_unused_tail"] is True,
            "unused FIFO tail omitted")
    return {"passed":True,"generated_rows_strictly_replayed":len(all_physical),
            "checked_rows":checked,"unused_FIFO_tail_rows":len(all_physical)-COUNT,
            "physical_saved_output_byte_identical":True,"source":CODEC_SHA256,
            "exact_replay_environment":environment}


def load_parent(path,expected_sha256,metadata):
    require(backbone.digest(expected_sha256),"explicit expected parent manifest SHA required")
    path=Path(path).resolve()
    require(sha256_file(path)==expected_sha256,"parent manifest hash differs")
    document=json.loads(path.read_bytes())
    require_replay_environment(document)
    require(document.get("samples_file")=="samples.npz","fixed physical payload filename required")
    payload=path.with_name("samples.npz")
    require(sha256_file(payload)==document["samples_sha256"],"physical payload hash differs")
    with np.load(payload,allow_pickle=False) as archive:
        require(archive.files==["samples"],"physical payload keys differ")
        values=archive["samples"]
    receipt=validate_parent(document,values,metadata)
    archive_info=document["latent_wave_archive"]
    require(archive_info["file"]=="latent-proposal-waves.npz","fixed latent archive filename required")
    latent_path=path.with_name(archive_info["file"])
    require(sha256_file(latent_path)==archive_info["sha256"],"actual latent wave archive hash differs")
    with np.load(latent_path,allow_pickle=False) as archive:
        waves={key:archive[key] for key in archive.files}
    decode=replay_generated_decode(document,waves,values)
    root=Path(__file__).resolve().parents[1]
    for name,digest in [
        ("src/posteriorforge/performance_prior_coordinates_v1.py",PARENT_SOURCE_SHA256),
        ("src/posteriorforge/execution_prior_coordinates_v1.py",BACKEND_SHA256),
        ("src/posteriorforge/prior_coordinates_v1.py",CODEC_SHA256),
        ("docs/accuracy-prior-coordinate-pilot-v1.md",PREREGISTRATION_SHA256)]:
        require(hashlib.sha256(backbone.git_bytes(root,document["git_revision"],name)).hexdigest()==digest,
                "executed parent source/Git identity differs")
    return document,values,{"manifest_sha256":expected_sha256,"samples_sha256":document["samples_sha256"],
        "samples":array_descriptor(values),"producer_sha256":PARENT_SOURCE_SHA256,
        "git_revision":document["git_revision"],"latent_archive_sha256":archive_info["sha256"],
        "input_validation":receipt,"full_generated_decode_replay":decode}


def run(parent_manifest,expected_parent_sha256,output):
    before=backbone.state_sha((np.random.get_state(),random.getstate()))
    sources=source_guard();metadata=kernel.metadata()
    parent,values,lineage=load_parent(parent_manifest,expected_parent_sha256,metadata)
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    record={"schema":SCHEMA,"variant":VARIANT,"status":"running","requested_count":COUNT,
        "context_seed":20261111,"case":2,"repetition":0,"parameter_order":metadata["physical_parameter_order"],
        "scientific_fingerprint":metadata["scientific_fingerprint"],"parent":lineage,"source_pins":sources,
        "conditional_kernel":metadata,"runtime":runtime_identity(),
        "exact_decode_and_G3_replay_environment":lineage["full_generated_decode_replay"]["exact_replay_environment"],
        "raw_scientific_bindings":parent["raw_scientific_bindings"],
        "raw_scientific_bindings_scope":"inherited provenance only; no bound data files loaded",
        "method_description":"prior-coordinate cached-FP32 neural sampler plus ONE frozen joint-global Gibbs block",
        "pure_NPE_claimed":False,"posterior_accuracy_guaranteed":False,
        "observations_references_truth_or_banks_loaded":False,
        "noise_or_local_likelihood_update_performed":False,
        "coordinate_decode_after_global_refresh":False,
        "support_filtering_clipping_redraw_or_trimming":False,
        "end_to_end_posterior_request_timing_claimed":False,"attribution":"Built with PriorLabs-TabPFN"}
    write_json(output/"manifest.json",record)
    try:
        seed=backbone.derive_seed(20261111,ROLE,case=2,repetition=0)
        result,receipt=refresh_once(values,seed)
        require(receipt["role"]==ROLE and receipt["standard_normal_draw_shape"]==[COUNT,3]
                and receipt["all18_columns_byte_identical"] is True,"frozen physical joint rule differs")
        with (output/"samples.npz").open("xb") as stream:np.savez(stream,samples=result)
        require(source_guard()==sources and sha256_file(parent_manifest)==expected_parent_sha256,
                "source or parent changed during single refresh")
        after=backbone.state_sha((np.random.get_state(),random.getstate()))
        require(before==after,"default NumPy/Python RNG changed")
        record.update(status="complete",samples=array_descriptor(result),samples_file="samples.npz",
            samples_sha256=sha256_file(output/"samples.npz"),refresh=receipt,sources_unchanged=True,
            default_RNG_observation={"before_sha256":before,"after_sha256":after,"preserved":True,
                "observed_generators":["NumPy legacy default","Python random"],"torch_CPU_CUDA_RNG_hashes_observed":False})
    except BaseException as error:
        record.update(status="failed",error_type=type(error).__name__,error=str(error))
        write_json(output/"manifest.json",record);raise
    write_json(output/"manifest.json",record)
    write_json(output/"checksums.json",{"manifest.json":sha256_file(output/"manifest.json"),
                                      "samples.npz":record["samples_sha256"]})
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-manifest",required=True)
    parser.add_argument("--expected-parent-sha256",required=True)
    parser.add_argument("--output",required=True)
    record=run(**vars(parser.parse_args()))
    print(json.dumps({k:record[k] for k in ("schema","variant","status","case","context_seed","samples_sha256")},sort_keys=True))


if __name__=="__main__":main()
