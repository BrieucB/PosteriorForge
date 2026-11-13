#!/usr/bin/env python3
"""ONE physical G3 update of a CLOSED corrected production prior-coordinate run.

Built with PriorLabs-TabPFN. Old failed/forensic families are rejected. Every
normal production footer, actual archive, F64 FIFO and finite-screen binding
must be present before any actual global draw. No neural/GPU calls occur here.
"""
from __future__ import annotations
import argparse,copy,hashlib,importlib.util,io,json,re,subprocess,tarfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from posteriorforge.artifacts import sha256_file,write_json,versions
from posteriorforge.banks import derive_seed,load_context,load_heldout
from posteriorforge.science.analytical_benchmark_v2 import array_descriptor
from posteriorforge import conditional_globals as kernel,prior_coordinates_v1 as codec
from posteriorforge.inference import CHECKPOINT_SHA256

SCHEMA="posteriorforge.prior_corrected_production_global_refresh.v1"
PRODUCER_SHA256="a30e0c15513eb0c1a6ed3c68f0781327edef4714a244a119a3a4d4c68f8780bf"
PRODUCER_SCHEMA="posteriorforge.inference.prior_physical_fifo.v1"
PRODUCER_VARIANT="accuracy_prior_coordinates_native_fp32_physical_f64_fifo_v1"
F64_HELPER_SHA256="f8823ae75e7ae2eea6f1aa667af48ecf5c892450361b5abd490ccda36a28f7e0"
KERNEL_SHA256="e346e25bc9825abf016c2bd75addaf2b025d3e3d94e1a0ea6f6e31dd02997a43"
CODEC_SHA256="d8ce6c49e7b65c6ee55c59b7243d4ce943cbea28c92ebed0329798074aaa70c5"
FINITE_SCREEN_SHA256="5ec69eb7aadbd74c7d96ee534b71d0d8bee2b6dd6f6e3d2f44e74607452f682b"
OLD_WRAPPER_SHA256="452280e5597cf2ac987fa10db2386157923c596ac70aff84b8c5e2f1dfa97940"
OLD_EVAL_SHA256="6c49e59ccee9bebeecd53990556a6e00a9f41b2fc53a2e8ecf9827014efc2d56"
DRAW_OBSERVER_SOURCE_SHA256="c73c4e489946e018399885c981afafc5c47e3eab6e27f6561b377f836f703f8c"
ROLE="ss15_global_block_gibbs_once_v1"
COUNT=50000

def require(value,message):
 if not value:raise ValueError(message)

def source_sha256():return sha256_file(__file__)

def immutable_pins(pins):return {k:v for k,v in pins.items() if k!="git_revision"}

def verify_git_sources(root,revision,expected):
 require(isinstance(revision,str) and re.fullmatch("[0-9a-f]{40}",revision) is not None,"recorded full Git revision required")
 raw=subprocess.run(["git","-C",str(root),"archive","--format=tar",revision,"--",*expected],check=True,capture_output=True).stdout
 with tarfile.open(fileobj=io.BytesIO(raw),mode="r:") as stream:
  actual={m.name:hashlib.sha256(stream.extractfile(m).read()).hexdigest() for m in stream.getmembers() if m.isfile()}
 require(actual==expected,"recorded execution Git source inventory differs")
 return {"passed":True,"recorded_revision":revision,"verified_source_count":len(expected)}

def load(filename,digest):
 path=Path(__file__).with_name(filename)
 require(isinstance(digest,str) and sha256_file(path)==digest,"source not frozen: "+filename)
 spec=importlib.util.spec_from_file_location("_production_"+path.stem,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def source_guard():
 from posteriorforge import performance_prior_physical_fifo_v1 as producer
 require(isinstance(PRODUCER_SHA256,str) and producer.source_sha256()==PRODUCER_SHA256,"corrected production source unpinned/changed")
 producer.require_sources()
 require(kernel.source_sha256()==KERNEL_SHA256 and codec.source_sha256()==CODEC_SHA256,"original kernel/codec changed")
 require(producer.physical_fifo.source_sha256()==F64_HELPER_SHA256,"shared physical-F64 helper changed")
 root=Path(__file__).resolve().parents[1]
 rev=subprocess.run(["git","-C",str(root),"rev-parse","HEAD"],check=True,capture_output=True,text=True).stdout.strip()
 files={"scripts/"+Path(__file__).name:source_sha256(),"src/posteriorforge/performance_prior_physical_fifo_v1.py":PRODUCER_SHA256,
  "src/posteriorforge/execution_physical_fifo_v1.py":F64_HELPER_SHA256,"src/posteriorforge/conditional_globals.py":KERNEL_SHA256}
 for name,h in files.items():
  raw=subprocess.run(["git","-C",str(root),"show",rev+":"+name],check=True,capture_output=True).stdout
  require(hashlib.sha256(raw).hexdigest()==h,"corrected production downstream source must be committed: "+name)
 observer=load("assess_prior_forensic_replay_v1.py",DRAW_OBSERVER_SOURCE_SHA256)
 return producer,observer,{"wrapper":source_sha256(),"corrected_producer":PRODUCER_SHA256,
  "F64_postdecode_helper":F64_HELPER_SHA256,"kernel":KERNEL_SHA256,"codec":CODEC_SHA256,
  "draw_observer_function_source":DRAW_OBSERVER_SOURCE_SHA256,
  "draw_function_reuse_scope":"unchanged refresh_with_actual_draws numerical caller only; no forensic cohort/envelope inherited",
  "old_validation_ancestry":OLD_EVAL_SHA256,"git_revision":rev}

def policy_differences(recorded,current):
 """Exact field/value/TYPE differences; diagnostics only, no normalization."""
 differences=[]
 def visit(left,right,path):
  if type(left) is not type(right):
   differences.append({"path":path,"recorded_type":type(left).__name__,"current_type":type(right).__name__,
    "recorded_value":left,"current_value":right});return
  if isinstance(left,dict):
   for key in sorted(set(left)|set(right)):
    if key not in left or key not in right:
     differences.append({"path":path+"."+key,"recorded_present":key in left,"current_present":key in right})
    else:visit(left[key],right[key],path+"."+key)
  elif isinstance(left,list):
   if len(left)!=len(right):differences.append({"path":path,"recorded_length":len(left),"current_length":len(right)})
   else:
    for i,(a,b) in enumerate(zip(left,right)):visit(a,b,path+"["+str(i)+"]")
  elif left!=right:
   differences.append({"path":path,"recorded_type":type(left).__name__,"current_type":type(right).__name__,
    "recorded_value":left,"current_value":right})
 visit(recorded,current,"")
 return differences

def verify_execution_policy(recorded,current):
 """One serialized-code telemetry field is process-local; all other facts exact."""
 left=copy.deepcopy(recorded);right=copy.deepcopy(current)
 old=left.get("derivation",{}).get("run_code_sha256")
 new=right.get("derivation",{}).get("run_code_sha256")
 for value in (old,new):
  require(isinstance(value,str) and re.fullmatch("[0-9a-f]{64}",value) is not None,
   "process-local serialized-code telemetry must be a wellformed SHA256")
 left["derivation"].pop("run_code_sha256");right["derivation"].pop("run_code_sha256")
 require(not policy_differences(left,right),"actual corrected production metadata/source/AST/inventory/routing differs")
 require(left["derivation"]["reverse_AST_exact"] is True
  and left["derivation"]["original_function_globals_unmodified"] is True
  and left["derivation"]["original_neural_engine_codec_stage_draw_objects_used"] is True,
  "source-bound exact AST/global/algorithm ancestry required")
 return {"passed":True,"all_other_metadata_fields_exact":True,
  "recorded_execution_serialized_code_SHA256":old,"current_verifier_serialized_code_SHA256":new,
  "serialized_code_fingerprints_equal":old==new,
  "fingerprint_scope":"process-local, reference-context-dependent serialized-code telemetry; not a cross-process source identity",
  "source_identity_basis":"exact recorded147 Git source blobs/file hashes + reverse AST + original live code/global identity guards",
  "original_producer_manifest_changed":False,"exact_original_metadata_field_value_TYPE_differences":policy_differences(recorded,current)}

def validate_success_header(d,producer_sha=PRODUCER_SHA256):
 require(not d.get("test_fixture"),"CPU test fixture is not an actual production request")
 expected={"schema":PRODUCER_SCHEMA,"variant":PRODUCER_VARIANT,"status":"complete",
  "runner_source_sha256":producer_sha,"context_seed":20261111,"case":2,"context_size":4096,
  "requested_count":COUNT,"repetition":0,"rank":0,"world_size":1,
  "mode":"fp32_transparent_control_v1","native_all8_stages":[6,10],"FP16_activated":False,
  "sources_unchanged":True,"default_rng_preserved":True,"completed_stage_indices":list(range(21)),
  "samples_in_original_physical_order_before_FIFO_support":True,"all_generated_rows_strictly_decoded_including_unused_FIFO_tail":True,
  "model_coordinate_decode_after_global_refresh":False,"sampler_math_changed":False,
  "partial_vector_filtering":False,"clipping_or_repair_performed":False,"physical_log_density_claimed":False,"learned_targets_are_original_physical_values":False,
  "fallback_map":{},"fallback_map_scope":"INNER frozen88 precision owner only; outer native all8 stages6/10 are explicit",
  "matching_cached_native_FP32_execution_passed_all21":True,"fresh_same_policy_all21_screen_replayed":True,
  "actual_fixed_conditional_screen_performed_during_request":False}
 for name,v in expected.items():require(d.get(name)==v,"only CLOSED corrected production parent accepted: "+name)
 require(d.get("default_rng_before")==d.get("default_rng_after") and isinstance(d.get("default_rng_after"),dict)
  and set(d["default_rng_after"])=={"numpy","python","torch_CPU","torch_CUDA"},"complete default RNG footer missing/changed")
 require(d.get("matching_conditional_qualification_sha256")==FINITE_SCREEN_SHA256,"actual fresh matching screen differs")
 require(isinstance(d.get("execution_counters"),dict) and isinstance(d.get("sampling_counters"),dict)
  and isinstance(d.get("actual_prediction_checks"),dict),"full normal engine/sampling/dtype footer missing")
 actual=d["actual_prediction_checks"]
 require(all(type(actual.get(k)) is int and actual[k]>0 for k in ("actual_head_contexts","sampler_callback_calls","shared_densities"))
  and actual["shared_densities"]==1 and actual["sampler_callback_calls"]+7==actual["actual_head_contexts"],
  "actual original callback/head/shared-density conservation differs")
 t=d.get("timings",{});m=d.get("memory",{})
 require(np.isfinite(t.get("posterior_request_seconds",np.nan)) and t["posterior_request_seconds"]>0
  and t.get("includes_stage_preparation") is True and t.get("includes_CPU_sampling") is True
  and t.get("includes_strict_all_generated_wave_decode") is True,"actual successful request timing scope missing")
 require(type(m.get("peak_allocated_bytes")) is int and 0<m["peak_allocated_bytes"]<=m.get("peak_reserved_bytes",0),"actual peak-memory footer missing")
 c=d.get("compatibility_adapter",{})
 require(c.get("original_alias_restored") is True and c.get("sdk_sources_unchanged") is True
  and c.get("sdk_aliases_unchanged") is True and c.get("patched_call_count")==d.get("expected_patched_call_count"),"actual CDF/footer restoration missing")
 return {"passed":True,"closed_successful_production_request":True,"forensic_family_accepted":False}

def load_parent(path,expected_sha,*,scratch):
 """Strict new success family plus actual full-wave/input/execution replay."""
 producer,observer,pins=source_guard();path=Path(path)
 require(sha256_file(path)==expected_sha,"explicit closed production manifest SHA differs")
 d=json.loads(path.read_bytes());header=validate_success_header(d,PRODUCER_SHA256)
 metadata=kernel.metadata()
 require(d["codec"]==codec.metadata() and d["coordinate_policy"]==producer.failed.backend.cached.policy_metadata()
  and d["native_execution_policy"]==producer.failed.backend.policy_metadata()
  and d["stage_to_physical_order"]==list(codec.ORDER) and d["physical_to_stage_order"]==list(codec.INVERSE_ORDER)
  and d["parameter_order"]==metadata["physical_parameter_order"] and d["scientific_fingerprint"]==metadata["scientific_fingerprint"],
  "actual prior-coordinate geometry/order/scientific identity differs")
 old=load("refresh_prior_coordinates_global_block_v1.py",OLD_WRAPPER_SHA256)
 environment=old.require_replay_environment(d)
 payload=path.with_name(d["samples_file"]);require(sha256_file(payload)==d["samples_sha256"],"actual successful physical payload differs")
 with np.load(payload,allow_pickle=False) as archive:
  require(archive.files==["samples"],"strict successful sample key required");values=archive["samples"]
 require(values.shape==(COUNT,21) and values.dtype==np.float64 and array_descriptor(values)==d["samples"],"strict50000 original physical F64 rows required")
 matrices={}
 for key in ("latent_wave_archive","decoded_wave_archive"):
  info=d[key];p=path.with_name(info["file"]);require(sha256_file(p)==info["sha256"],"successful proposal archive differs")
  with np.load(p,allow_pickle=False) as archive:matrices[key]={n:archive[n] for n in archive.files}
 decoded=old.replay_generated_decode(d,matrices["latent_wave_archive"],values,matrices["decoded_wave_archive"])
 results=Path(scratch)/"results";banks=results/"banks-v1"
 context=load_context(banks,20261111,4096);raw_query=load_heldout(banks,2).obs.reshape(1,33)
 theta,obs,query,model,encoding=producer.failed.model_bindings(context,raw_query)
 raw={"checkpoint_sha256":CHECKPOINT_SHA256,"scientific_fingerprint":kernel.metadata()["scientific_fingerprint"],
  "parameter_order":kernel.metadata()["physical_parameter_order"],"context_theta":array_descriptor(context.theta),
  "context_obs":array_descriptor(context.obs),"query":array_descriptor(raw_query)}
 require(raw==d["raw_scientific_bindings"] and model==d["model_inputs"],"actual physical bank/query/latent model inputs differ")
 for key in ("physical_theta_original_Float64","latent_physical_slot_model_Float32","ordered_latent_model_Float32","stage_to_physical_order","physical_to_stage_order"):
  require(encoding[key]==d["model_target_descriptors"][key],"actual causal target encoding differs: "+key)
 execution_policy=producer.execution_metadata()
 policy_authentication=verify_execution_policy(d["execution_policy"],execution_policy)
 expected_files=dict(execution_policy["runtime_source_inventory"]["sources"])
 expected_files["src/posteriorforge/performance_prior_physical_fifo_v1.py"]=PRODUCER_SHA256
 recorded_sources=verify_git_sources(Path(__file__).resolve().parents[1],d["git_revision"],expected_files)
 receipt_args=dict(banks=banks,prefix_run=results/"accuracy-ss15-v1/sigma-native-grid-01/seed-20261111-case-2",
  failed_screen=results/"accuracy-prior-coordinates-v1/conditionals-01/manifest.json",
  codec_inventory=results/"accuracy-prior-coordinates-v1/codec-inventory-inference-01/manifest.json",
  codec_reference_inventory=results/"accuracy-prior-coordinates-v1/codec-inventory-reference-01/manifest.json",
  codec_combined_inventory=Path(__file__).resolve().parents[1]/"evidence/accuracy-prior-coordinates-v1/codec-inventory-01/CPU-receipt.json",
  completed_grid=results/"accuracy-ss15-v1/sigma-native-grid-completion-01/completion.json",raw_bindings=raw,model_inputs=model)
 _,screen_receipt=producer.failed.qualification_receipt(results/"accuracy-prior-coordinates-v1/native-conditionals-01/manifest.json",**receipt_args)
 old_eval=load("evaluate_prior_coordinates_pilot_v1.py",OLD_EVAL_SHA256)
 helper=producer.failed.load_script(producer.failed.QUALIFIER_FILENAME,producer.failed.QUALIFIER_SHA256)
 # Exact old numerical/state witness validator; explicit NEW producer policy
 # namespace is a parameter, never an old module/global dictionary mutation.
 proxy=SimpleNamespace(backend=producer.failed.backend,MODE=producer.failed.MODE,
  clean=producer._CLEAN_PROXY,proposal_scope=producer.proposal_scope)
 execution=old_eval.execution_checks(d,proxy,{"theta":theta,"obs":obs,"raw":raw},helper)
 require(d["expected_stage_prepare_count"]==d["execution_counters"]["prepare_count"]==21
  and d["generated_proposal_wave_count"]==1,"actual preparation/wave counters differ")
 return d,values,{"parent_manifest_sha256":expected_sha,"samples_sha256":d["samples_sha256"],"header":header,
  "actual_all51200_decode_replay":decoded,"actual_raw_model_mask_target_inputs_recomputed":True,
  "fresh_finite_screen_replay":screen_receipt,"actual_execution_checks":execution,
  "exact_replay_environment":environment,"source_pins":pins,"recorded_execution_source_inventory":recorded_sources,
  "execution_policy_authentication":policy_authentication,
  "execution_validation_scope":"exact old6c validation function with explicit new corrected producer/F64 sampling-policy parameter namespace; original globals unchanged"}

def refresh(*,root,parent_manifest,parent_manifest_sha256,output):
 producer,observer,pins=source_guard();parent,values,auth=load_parent(parent_manifest,parent_manifest_sha256,scratch=root)
 out=Path(output);out.mkdir(parents=True,exist_ok=False)
 record={"schema":SCHEMA,"status":"running","attribution":"Built with PriorLabs-TabPFN","source_pins":pins,"parent":auth,
  "context_seed":20261111,"case":2,"repetition":0,"requested_count":COUNT,"conditional_kernel":kernel.metadata(),
  "original_parent_status":"complete","forensic":False,"pure_NPE_claimed":False,"posterior_accuracy_guaranteed":False,
  "coordinate_decode_after_global_refresh":False,"method":"closed corrected FP32 production parent plus ONE separate physical G3"}
 write_json(out/"manifest.json",record)
 try:
  seed=derive_seed(20261111,ROLE,case=2,repetition=0)
  result,normals,receipt=observer.refresh_with_actual_draws(values,seed)
  require(result.shape==(COUNT,21) and result[:,3:].tobytes()==values[:,3:].tobytes(),"actual physical18 bytes changed")
  with (out/"samples.npz").open("xb") as stream:np.savez(stream,samples=result)
  with (out/"actual-standard-normal-draws.npz").open("xb") as stream:np.savez(stream,standard_normal_draws=normals)
  require(immutable_pins(source_guard()[2])==immutable_pins(pins) and sha256_file(parent_manifest)==parent_manifest_sha256,"source/parent changed during ONE G3")
  record.update(status="complete",samples_file="samples.npz",samples=array_descriptor(result),samples_sha256=sha256_file(out/"samples.npz"),
   actual_draws_file="actual-standard-normal-draws.npz",actual_draws_sha256=sha256_file(out/"actual-standard-normal-draws.npz"),refresh=receipt)
 except BaseException as error:
  record.update(status="failed",error_type=type(error).__name__,error=str(error));write_json(out/"manifest.json",record);raise
 write_json(out/"manifest.json",record);write_json(out/"checksums.json",{p.name:sha256_file(p) for p in out.iterdir() if p.is_file()})
 print(json.dumps({"production_ONE_G3_complete":str(out),"manifest_sha256":sha256_file(out/"manifest.json"),"samples_sha256":record["samples_sha256"]}),flush=True)
 return record

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ("root","parent-manifest","parent-manifest-sha256","output"):p.add_argument("--"+name,required=True)
 refresh(**vars(p.parse_args()))
if __name__=="__main__":main()
