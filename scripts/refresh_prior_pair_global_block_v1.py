#!/usr/bin/env python3
"""Pair-bound ONE physical G3 update after a fresh pair-qualified production run.

Built with PriorLabs-TabPFN. New seed/case family only. The frozen case2
sources/proofs are never relabelled. No neural/GPU/reference integration occurs
here; every actual draw requires a closed matching production parent.
"""
from __future__ import annotations
import argparse,copy,hashlib,importlib.util,json,re,subprocess,time
from collections import Counter
from pathlib import Path
import numpy as np
from posteriorforge.artifacts import sha256_file,write_json
from posteriorforge.banks import derive_seed
from posteriorforge.science.analytical_benchmark_v2 import array_descriptor
from posteriorforge import conditional_globals as kernel,prior_coordinates_v1 as codec

SCHEMA="posteriorforge.prior_coordinate_pair_global_refresh.v1"
PRODUCER_SHA256="52eaa638d09358765fcbb1d229265a5679f05a5fe0a5866f513412c7dc572a0b"
PAIR_SOURCE_SHA256="402117cad65af190352cb0b14977fbbdc896f196d069749b0be28608fdef400e"
PAIR_CPU_RECEIPT_SHA256="6f35d9d090a955604186c5fbf3312b82465eff6fdae7e4cd24a23b646dbdf193"
PAIR_QUALIFIER_SHA256="29ca88203b81f240769a6bb38b18b68aff0fe6ed2df36003729b17e0116163a0"
PRODUCER_SCHEMA="posteriorforge.inference.prior_coordinate_pair.v1"
PRODUCER_VARIANT="accuracy_prior_coordinates_pair_native_fp32_physical_f64_fifo_v1"
PROTOCOL_SHA256="9fc6a91568864cc996dafbcfaa41e77280e4825c63f61d455bdadf3703cd0a66"
BASE_WRAPPER_SHA256="4aa226e24ca381df8956d8c8238a9e17092b01f9771b020cae2166c72990d284"
OLD_EVAL_SHA256="6c49e59ccee9bebeecd53990556a6e00a9f41b2fc53a2e8ecf9827014efc2d56"
DRAW_SOURCE_SHA256="c73c4e489946e018399885c981afafc5c47e3eab6e27f6561b377f836f703f8c"
KERNEL_SHA256="e346e25bc9825abf016c2bd75addaf2b025d3e3d94e1a0ea6f6e31dd02997a43"
CODEC_SHA256="d8ce6c49e7b65c6ee55c59b7243d4ce943cbea28c92ebed0329798074aaa70c5"
F64_HELPER_SHA256="f8823ae75e7ae2eea6f1aa667af48ecf5c892450361b5abd490ccda36a28f7e0"
OLD_CASE2_SCREEN_SHA256="5ec69eb7aadbd74c7d96ee534b71d0d8bee2b6dd6f6e3d2f44e74607452f682b"
ROLE="ss15_global_block_gibbs_once_v1"
COUNT=50000

def require(value,message):
 if not value:raise ValueError(message)

def load(filename,digest):
 p=Path(__file__).with_name(filename);require(isinstance(digest,str) and sha256_file(p)==digest,"frozen helper differs: "+filename)
 s=importlib.util.spec_from_file_location("_pair_"+p.stem,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

BASE=load("refresh_prior_corrected_production_v1.py",BASE_WRAPPER_SHA256)
DRAW=load("assess_prior_forensic_replay_v1.py",DRAW_SOURCE_SHA256)
_RECORDING_GENERATOR=DRAW.RecordingGenerator
_RECORDING_CODE=_RECORDING_GENERATOR.standard_normal.__code__
_KERNEL_REFRESH=kernel.refresh
_KERNEL_REFRESH_CODE=_KERNEL_REFRESH.__code__

def source_sha256():return sha256_file(__file__)

def pair_spec(context_seed,case):
 from posteriorforge import prior_coordinate_pair_v1 as pairs
 return pairs.PairSpec(context_seed,case)

def source_guard():
 from posteriorforge import performance_prior_pair_v1 as producer,prior_coordinate_pair_v1 as pairs
 require(isinstance(PRODUCER_SHA256,str) and producer.source_sha256()==PRODUCER_SHA256,"pair production source unpinned/changed")
 require(isinstance(PAIR_SOURCE_SHA256,str) and pairs.source_sha256()==PAIR_SOURCE_SHA256,"pair data source unpinned/changed")
 require(isinstance(PAIR_QUALIFIER_SHA256,str),"pair qualification source not frozen")
 producer.require_sources();pairs.require_sources()
 require(producer.PAIR_CPU_RECEIPT_SHA256==PAIR_CPU_RECEIPT_SHA256,"frozen all23 dualCPU receipt differs")
 require(producer.physical_fifo.source_sha256()==F64_HELPER_SHA256,"frozen physicalF64 completion helper changed")
 require(kernel.source_sha256()==KERNEL_SHA256 and codec.source_sha256()==CODEC_SHA256,"frozen scientific kernel/codec changed")
 require(_RECORDING_GENERATOR is DRAW.RecordingGenerator and _RECORDING_GENERATOR.standard_normal.__code__ is _RECORDING_CODE
  and kernel.refresh is _KERNEL_REFRESH and _KERNEL_REFRESH.__code__ is _KERNEL_REFRESH_CODE,"original draw observer/kernel callable changed")
 root=Path(__file__).resolve().parents[1]
 doc=root/"docs/accuracy-prior-coordinate-fixed-panel-v1.md";require(sha256_file(doc)==PROTOCOL_SHA256,"predeclared fixed-pair protocol changed")
 revision=subprocess.run(["git","-C",str(root),"rev-parse","HEAD"],check=True,capture_output=True,text=True).stdout.strip()
 BASE.verify_git_sources(root,revision,{
  "scripts/"+Path(__file__).name:source_sha256(),"src/posteriorforge/performance_prior_pair_v1.py":PRODUCER_SHA256,
  "src/posteriorforge/prior_coordinate_pair_v1.py":PAIR_SOURCE_SHA256,
  "scripts/qualify_prior_coordinate_pair_v1.py":PAIR_QUALIFIER_SHA256,
  "src/posteriorforge/conditional_globals.py":KERNEL_SHA256,
  "docs/accuracy-prior-coordinate-fixed-panel-v1.md":PROTOCOL_SHA256})
 return producer,pairs,{"wrapper":source_sha256(),"pair_producer":PRODUCER_SHA256,"pair_data":PAIR_SOURCE_SHA256,
  "pair_qualifier":PAIR_QUALIFIER_SHA256,"pair_CPU_receipt":PAIR_CPU_RECEIPT_SHA256,"kernel":KERNEL_SHA256,"codec":CODEC_SHA256,"F64_helper":F64_HELPER_SHA256,
  "guard_primitive_source":BASE_WRAPPER_SHA256,"actual_draw_observer_source":DRAW_SOURCE_SHA256,
  "protocol":PROTOCOL_SHA256,"git_revision":revision}

def validate_header(document,spec,proof_sha256):
 require(not document.get("test_fixture"),"CPU test fixture is not a production request")
 require(isinstance(proof_sha256,str) and re.fullmatch("[0-9a-f]{64}",proof_sha256) and proof_sha256!=OLD_CASE2_SCREEN_SHA256,
  "fresh same-pair proof required; old case2 proof cannot qualify a new pair")
 expected={"schema":PRODUCER_SCHEMA,"variant":PRODUCER_VARIANT,"status":"complete","runner_source_sha256":PRODUCER_SHA256,
  "pair_spec":spec.as_dict(),"context_seed":spec.context_seed,"case":spec.case,"context_size":4096,"requested_count":COUNT,
  "repetition":0,"rank":0,"world_size":1,"mode":"fp32_transparent_control_v1","native_all8_stages":[6,10],"FP16_activated":False,
  "sources_unchanged":True,"default_rng_preserved":True,"completed_stage_indices":list(range(21)),
  "samples_in_original_physical_order_before_FIFO_support":True,"all_generated_rows_strictly_decoded_including_unused_FIFO_tail":True,
  "model_coordinate_decode_after_global_refresh":False,"sampler_math_changed":False,"partial_vector_filtering":False,
  "clipping_or_repair_performed":False,"physical_log_density_claimed":False,"learned_targets_are_original_physical_values":False,
  "fallback_map":{},"fallback_map_scope":"INNER frozen88 precision owner only; outer native all8 stages6/10 are explicit",
  "matching_conditional_qualification_sha256":proof_sha256,"matching_cached_native_FP32_execution_passed_all21":True,
  "fresh_same_policy_all21_screen_replayed":True,"actual_fixed_conditional_screen_performed_during_request":False}
 for k,v in expected.items():require(not BASE.policy_differences(document.get(k),v),"pair-bound production header differs: "+k)
 require(document.get("default_rng_before")==document.get("default_rng_after") and isinstance(document.get("default_rng_after"),dict)
  and set(document["default_rng_after"])=={"numpy","python","torch_CPU","torch_CUDA"},"full default RNG footer missing/changed")
 for k in ("execution_counters","sampling_counters","actual_prediction_checks"):require(isinstance(document.get(k),dict),"normal production footer missing: "+k)
 actual=document["actual_prediction_checks"]
 require(all(type(actual.get(k)) is int and actual[k]>0 for k in ("actual_head_contexts","sampler_callback_calls","shared_densities"))
  and actual["shared_densities"]==1 and actual["sampler_callback_calls"]+7==actual["actual_head_contexts"],"actual callback/head conservation differs")
 timing=document.get("timings",{});memory=document.get("memory",{});cdf=document.get("compatibility_adapter",{})
 require(np.isfinite(timing.get("posterior_request_seconds",np.nan)) and timing["posterior_request_seconds"]>0
  and timing.get("includes_stage_preparation") is True and timing.get("includes_CPU_sampling") is True
  and timing.get("includes_strict_all_generated_wave_decode") is True,"successful request timing scope missing")
 require(type(memory.get("peak_allocated_bytes")) is int and 0<memory["peak_allocated_bytes"]<=memory.get("peak_reserved_bytes",0),"actual memory footer missing")
 require(cdf.get("original_alias_restored") is True and cdf.get("sdk_sources_unchanged") is True and cdf.get("sdk_aliases_unchanged") is True
  and cdf.get("patched_call_count")==document.get("expected_patched_call_count"),"actual corrected-CDF restoration/count differs")
 return {"passed":True,"pair_spec":spec.as_dict(),"fresh_same_pair_proof_sha256":proof_sha256,"old_case2_family_promoted":False}

def refresh_with_timing(values,spec):
 """One unchanged e346 kernel call; actual Z recording, no timing optimization."""
 require(isinstance(values,np.ndarray) and values.dtype==np.float64 and values.shape==(COUNT,21),"strict physical F64[50000,21] required")
 before_descriptor=array_descriptor(values);default_before=DRAW.state_hash()
 seed=derive_seed(spec.context_seed,ROLE,case=spec.case,repetition=0)
 rng=_RECORDING_GENERATOR(np.random.PCG64(seed));initial=copy.deepcopy(rng.bit_generator.state)
 started=time.perf_counter();after=_KERNEL_REFRESH(values,rng);math_seconds=time.perf_counter()-started
 z=rng.actual_draws;final=copy.deepcopy(rng.bit_generator.state)
 require(rng.standard_normal_calls==1 and z.dtype==np.float64 and z.shape==(COUNT,3),"one actual Gaussian matrix required")
 mean,covariance=kernel.conditional_parameters(values)
 require(after[:,:3].tobytes()==(mean+z@np.linalg.cholesky(covariance).T).tobytes()
  and after[:,3:].tobytes()==values[:,3:].tobytes() and array_descriptor(values)==before_descriptor,
  "actual G3 output or physical18/input bytes changed")
 require(DRAW.state_hash()==default_before,"default NumPy/Python RNG changed")
 state_sha=lambda x:hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":")).encode()).hexdigest()
 receipt={"pair_spec":spec.as_dict(),"seed":int(seed),"role":ROLE,"bit_generator":"PCG64",
  "initial_private_state":initial,"final_private_state":final,"initial_private_state_sha256":state_sha(initial),"final_private_state_sha256":state_sha(final),
  "standard_normal_method_calls":1,"standard_normal_draw_shape":[COUNT,3],"standard_normal_scalar_count":COUNT*3,
  "standard_normal_draws":array_descriptor(z),"draws_are_actual_returned_matrix_not_rerolled":True,
  "all18_physical_columns_byte_identical":True,"input_array_unchanged":True,"global_block_refresh_count_per_row":1,
  "coordinate_decode_after_global_refresh":False,"default_numpy_python_RNG_before_sha256":default_before,
  "default_numpy_python_RNG_after_sha256":DRAW.state_hash(),"default_numpy_python_RNG_preserved":True,
  "torch_CPU_CUDA_RNG_observed":False,"G3_kernel_math_seconds":math_seconds,
  "G3_timing_scope":"one unchanged e346.refresh call including its conditional arithmetic, actual draw and output copy; verification/archives excluded"}
 return after,z,receipt


# Source-bound NEW pair-aware copy of frozen6c execution checks. Only signature
# and stream seed pair arguments differ; no old globals or function are changed.
model_guard=load("evaluate_prior_coordinates_pilot_v1.py",OLD_EVAL_SHA256).model_guard
def execution_checks(parent,producer,actual,helper,spec):
    backend=producer.backend
    theta,obs=actual["theta"],actual["obs"]
    execution=parent["execution_counters"];scopes=execution["completed_stage_scopes"]
    training=execution["actual_latent_training_inputs"];calls=execution["actual_latent_query_calls"]
    require(len(scopes)==len(training)==len(parent["stage_events"])==21
        and execution["prepare_count"]==21 and execution["all_stage_aliases_restored"] is True,
        "actual all21 stage scopes differ")
    cursor=0
    for stage,(scope,binding,event) in enumerate(zip(scopes,training,parent["stage_events"])):
        require(scope["stage_index"]==stage and scope["failed"] is False
                and scope["engine_release_succeeded"] is True
                and binding==backend.training_bindings(theta,obs,stage),
                "actual latent training mask/target or owner release differs")
        helper.witness_guard(scope["precision_session"],stage,backend)
        model_guard(scope["model"])
        require(scope["preparation_witness"]["passed"] is True,"actual preparation witness failed")
        if stage in backend.NATIVE_STAGES:
            require(scope["preparation_witness"]["native_neural_prefill_calls"]==0
                and scope["preparation_witness"]["cached_prefill_claimed"] is False
                and scope["actual_original_all8_fit_count"]==1
                and scope["immutable_native_before"]==scope["immutable_native_after"]
                and scope["immutable_native_before"] is not None,
                "native original preparation/immutable state differs")
        observed=Counter()
        subset=calls[cursor:cursor+scope["actual_head_contexts"]]
        for call in subset:
            shape=call["query_unmasked"]["shape"];masked=call["query_masked"]
            require(call["stage"]==stage and call["execution_route"]==backend.route(stage)
                and call["policy_sha256"]==backend.cached.policy_metadata()["policy_sha256"]
                and call["native_execution_policy_sha256"]==backend.policy_metadata()["policy_sha256"]
                and call["mask"]==backend.stage_policy(stage)
                and call["actual_target_coordinate"]==backend.stage_policy(stage)["target_coordinate"]
                and shape==masked["shape"] and shape[1]==15+stage
                and call["query_unmasked"]["dtype"]==masked["dtype"]=="<f4"
                and call["witness"]["passed"] is True
                and call["witness"]["mode"]==producer.MODE,
                "actual query policy/shape/dtype/route witness differs")
            if stage<3:require(call["query_unmasked"]==masked,"global query input unexpectedly masked")
            observed[(call["member"],shape[0])]+=1
        hist=event["sampler"]["neural_query_batch_sizes_by_member"]
        expected=Counter((member,int(rows)) for member in range(8)
                         for rows,count in hist[str(member)].items() for _ in range(count))
        require(observed==expected,"actual per-member query rows/head counts differ")
        routes=scope["route_counters"]
        native_stage=stage in backend.NATIVE_STAGES
        require(routes.get("resident_component_predict_count",0)==(0 if native_stage else sum(expected.values()))
                and routes.get("uncached_predict_count",0)==(sum(expected.values()) if native_stage else 0),
                "cached/native route conservation differs")
        cursor+=scope["actual_head_contexts"]
    require(cursor==len(calls)==execution["actual_head_contexts"],"actual query journal coverage differs")
    frozen=producer.clean.frozen_sampling()
    seeds=frozen.worker_stream_seeds(spec.context_seed,spec.case,0,0,1)
    require(parent["seed_streams"]==seeds and parent["proposal_buffer_policy"]==frozen.proposal_buffer_policy(1200),
            "original seed roles/FIFO policy differs")
    frozen.validate_proposal_buffer_record(parent["proposal_buffer"],world_size=1,rank=0,
        scope=producer.proposal_scope(frozen,actual["raw"],parent["model_inputs"],seeds))
    counters=parent["sampling_counters"]
    heads=8*counters["stage0_shared_predictions"]+counters["component_predictions"]
    require(heads==len(calls)==parent["expected_patched_call_count"]
        ==parent["compatibility_adapter"]["patched_call_count"]
        ==parent["actual_prediction_checks"]["actual_head_contexts"]
        and counters["stage0_shared_predictions"]==1 and execution["shared_densities"]==1,
        "actual physical head/CDF/shared density count differs")
    require(parent["default_rng_preserved"] is True
        and parent["default_rng_before"]==parent["default_rng_after"],"actual default RNG changed")
    return {"passed":True,"actual_training_masks_recomputed":True,"actual_query_policy_shape_dtype_routes_verified":True,
        "query_value_arrays_independently_reconstructed":False,
        "scope":"source-bound actual query descriptors and independent head/shape conservation",
        "stage_count":21,"actual_head_contexts":heads,"inner88_precision_fallback_map":{},
        "native_all8_stages":list(backend.NATIVE_STAGES),"mode":producer.MODE}

def verify_canonical_policy(recorded,current):
 """Exact recursive field/value/TYPE equality; no marshal telemetry is canonical."""
 require(not BASE.policy_differences(recorded,current),"actual pair canonical policy/source/AST/inventory/routing differs")
 require(recorded.get("pair_spec")==current["pair_spec"]
  and recorded.get("old_case2_screen_inherited") is False
  and recorded.get("FP16_activated") is False,"pair policy scope differs")
 require("run_code_sha256" not in recorded.get("derivation",{})
  and "run_code_sha256" not in recorded.get("inherited_numerical_execution",{}).get("derivation",{}),
  "serialized-code telemetry is not a canonical source identity")
 return {"passed":True,"all_canonical_metadata_fields_exact_TYPE_and_value":True,
  "source_identity_basis":"recorded dynamic Git source inventory, source AST and live code/global identity guards",
  "serialized_code_fingerprint_used_as_crossprocess_identity":False}

def _safe_sibling(manifest,filename):
 require(isinstance(filename,str) and Path(filename).name==filename,"artifact must be an explicit sibling file")
 return Path(manifest).with_name(filename)

def load_parent(path,expected_sha,*,scratch,spec,execution_screen,pair_cpu_receipt):
 """Actual same-pair source/proof/codec/FIFO checks before ANY global update."""
 started=time.perf_counter();producer,pairs,pins=source_guard();source_seconds=time.perf_counter()-started
 path=Path(path);require(sha256_file(path)==expected_sha,"explicit pair parent manifest SHA differs")
 d=json.loads(path.read_bytes());proof_sha=producer.qualification_sha256(spec)
 header=validate_header(d,spec,proof_sha)
 require(d.get("pair_cpu_receipt_sha256")==producer.PAIR_CPU_RECEIPT_SHA256
  and sha256_file(pair_cpu_receipt)==producer.PAIR_CPU_RECEIPT_SHA256,"actual pair CPU receipt differs")
 require(d["pair_binding_policy"]==spec.metadata() and d["pair_binding_module_sha256"]==PAIR_SOURCE_SHA256
  and d["fixed_case2_qualification_inherited"] is False
  and d["actual_request_matches_its_own_finite_screen"] is True
  and d["pair_preregistration_sha256"]==PROTOCOL_SHA256,"actual pair binding/qualification policy differs")
 metadata=kernel.metadata()
 require(d["codec"]==codec.metadata() and d["coordinate_policy"]==producer.backend.cached.policy_metadata()
  and d["native_execution_policy"]==producer.backend.policy_metadata()
  and d["stage_to_physical_order"]==list(codec.ORDER) and d["physical_to_stage_order"]==list(codec.INVERSE_ORDER)
  and d["parameter_order"]==metadata["physical_parameter_order"]
  and d["scientific_fingerprint"]==metadata["scientific_fingerprint"],"causal coordinate/physical scientific identity differs")
 old=load("refresh_prior_coordinates_global_block_v1.py",BASE.OLD_WRAPPER_SHA256)
 environment=old.require_replay_environment(d)
 payload=_safe_sibling(path,d["samples_file"]);require(sha256_file(payload)==d["samples_sha256"],"actual physical payload differs")
 with np.load(payload,allow_pickle=False) as archive:
  require(archive.files==["samples"],"strict physical sample key required");values=archive["samples"]
 require(values.dtype==np.float64 and values.shape==(COUNT,21) and array_descriptor(values)==d["samples"],"complete original physical F64 samples required")
 from posteriorforge.science.analytical_benchmark_v2 import build_problem
 require(build_problem().contains(values).all(),"complete physical support required")
 matrices={}
 for key in ("latent_wave_archive","decoded_wave_archive"):
  info=d[key];p=_safe_sibling(path,info["file"]);require(sha256_file(p)==info["sha256"],"actual full proposal archive differs")
  with np.load(p,allow_pickle=False) as archive:matrices[key]={name:archive[name] for name in archive.files}
 decoded=old.replay_generated_decode(d,matrices["latent_wave_archive"],values,matrices["decoded_wave_archive"])
 results=Path(scratch)/"results";banks=results/"banks-v1"
 context,rawquery,raw=pairs.raw_inputs(spec,banks=banks)
 theta,obs,query,model,encoding=producer.model_bindings(context,rawquery)
 require(not BASE.policy_differences(raw,d["raw_scientific_bindings"])
  and not BASE.policy_differences(model,d["model_inputs"]),"actual case/seed raw bank/query/model inputs differ")
 for key in ("physical_theta_original_Float64","latent_physical_slot_model_Float32","ordered_latent_model_Float32","stage_to_physical_order","physical_to_stage_order"):
  require(encoding[key]==d["model_target_descriptors"][key],"actual causal latent training target differs: "+key)
 current_policy=producer.execution_metadata(spec,proof_sha)
 policy=verify_canonical_policy(d["execution_policy"],current_policy)
 expected_files=dict(current_policy["runtime_source_inventory"]["sources"])
 expected_files["src/posteriorforge/performance_prior_pair_v1.py"]=PRODUCER_SHA256
 recorded_sources=BASE.verify_git_sources(Path(__file__).resolve().parents[1],d["git_revision"],expected_files)
 owner=pairs.prefix_parent(spec,completion_receipt=results/"accuracy-ss15-v1/sigma-native-grid-completion-01/completion.json")
 receipt_args=dict(spec=spec,banks=banks,pair_cpu_receipt=pair_cpu_receipt,raw_bindings=raw,model_inputs=model,
  codec_inventory=results/"accuracy-prior-coordinates-v1/codec-inventory-inference-01/manifest.json",
  codec_reference_inventory=results/"accuracy-prior-coordinates-v1/codec-inventory-reference-01/manifest.json",
  codec_combined_inventory=Path(__file__).resolve().parents[1]/"evidence/accuracy-prior-coordinates-v1/codec-inventory-01/CPU-receipt.json",
  completed_grid=results/"accuracy-ss15-v1/sigma-native-grid-completion-01/completion.json",
  prefix_run=Path(owner["manifest_path"]).parent,failed_screen=results/"accuracy-prior-coordinates-v1/conditionals-01/manifest.json")
 _,screen_receipt=producer.qualification_receipt(execution_screen,**receipt_args)
 helper=producer.load_script(producer.QUALIFIER_FILENAME,PAIR_QUALIFIER_SHA256)
 namespace=producer.bound_validation_namespace(spec,proof_sha)
 execution=execution_checks(d,namespace,{"theta":theta,"obs":obs,"raw":raw},helper,spec)
 require(d["expected_stage_prepare_count"]==d["execution_counters"]["prepare_count"]==21
  and d["generated_proposal_wave_count"]==1,"actual stage preparation/full-wave count differs")
 require(sha256_file(path)==expected_sha and sha256_file(execution_screen)==proof_sha
  and sha256_file(pair_cpu_receipt)==producer.PAIR_CPU_RECEIPT_SHA256,"pair parent/proof/CPU evidence changed during validation")
 return d,values,{"parent_manifest_sha256":expected_sha,"samples_sha256":d["samples_sha256"],
  "pair_spec":spec.as_dict(),"header":header,"actual_all_generated_rows_decode_replay":decoded,
  "actual_raw_model_mask_target_inputs_recomputed":True,"fresh_same_pair_screen_replay":screen_receipt,
  "actual_execution_checks":execution,"exact_replay_environment":environment,"source_pins":pins,
  "recorded_execution_source_inventory":recorded_sources,"canonical_execution_policy_authentication":policy,
  "CPU_wall_seconds":{"explicit_initial_source_validation":source_seconds,"total_parent_validation":time.perf_counter()-started},
  "CPU_timing_scope":"wall time; total includes repeated source/proof/data/codec/state checks; source subset is not additive",
  "new_model_GPU_or_reference_integrations":0,"new_global_draws_during_validation":0}

def refresh(*,root,context_seed,case,parent_manifest,parent_manifest_sha256,execution_screen,pair_cpu_receipt,output):
 spec=pair_spec(context_seed,case)
 parent,values,auth=load_parent(parent_manifest,parent_manifest_sha256,scratch=root,spec=spec,
  execution_screen=execution_screen,pair_cpu_receipt=pair_cpu_receipt)
 _,_,pins=source_guard()
 out=Path(output);out.mkdir(parents=True,exist_ok=False)
 record={"schema":SCHEMA,"status":"running","attribution":"Built with PriorLabs-TabPFN",
  "pair_spec":spec.as_dict(),"context_seed":spec.context_seed,"case":spec.case,"repetition":0,"requested_count":COUNT,
  "source_pins":pins,"parent":auth,"conditional_kernel":kernel.metadata(),"original_parent_status":"complete",
  "forensic":False,"pure_NPE_claimed":False,"posterior_accuracy_guaranteed":False,
  "coordinate_decode_after_global_refresh":False,"method":"closed pair-qualified FP32 physical parent plus ONE separate physical G3"}
 write_json(out/"manifest.json",record)
 try:
  after,z,receipt=refresh_with_timing(values,spec)
  with (out/"samples.npz").open("xb") as stream:np.savez(stream,samples=after)
  with (out/"actual-standard-normal-draws.npz").open("xb") as stream:np.savez(stream,standard_normal_draws=z)
  require(BASE.immutable_pins(source_guard()[2])==BASE.immutable_pins(pins)
   and sha256_file(parent_manifest)==parent_manifest_sha256,"pair source/parent changed during ONE G3")
  record.update(status="complete",samples_file="samples.npz",samples=array_descriptor(after),
   samples_sha256=sha256_file(out/"samples.npz"),actual_draws_file="actual-standard-normal-draws.npz",
   actual_draws_sha256=sha256_file(out/"actual-standard-normal-draws.npz"),refresh=receipt,
   CPU_wall_seconds={**auth["CPU_wall_seconds"],"G3_kernel_math":receipt["G3_kernel_math_seconds"]})
 except BaseException as error:
  record.update(status="failed",error_type=type(error).__name__,error=str(error))
  write_json(out/"manifest.json",record);raise
 write_json(out/"manifest.json",record)
 write_json(out/"checksums.json",{p.name:sha256_file(p) for p in out.iterdir() if p.is_file()})
 print(json.dumps({"pair_G3_complete":str(out),"pair_spec":spec.as_dict(),
  "manifest_sha256":sha256_file(out/"manifest.json"),"samples_sha256":record["samples_sha256"]},sort_keys=True),flush=True)
 return record

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ("root","parent-manifest","parent-manifest-sha256","execution-screen","pair-cpu-receipt","output"):p.add_argument("--"+name,required=True)
 p.add_argument("--context-seed",type=int,required=True);p.add_argument("--case",type=int,required=True)
 refresh(**vars(p.parse_args()))
if __name__=="__main__":main()
