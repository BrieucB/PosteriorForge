#!/usr/bin/env python3
"""FORENSIC CPU completion from a failed request's fixed saved proposal stream.

Built with PriorLabs-TabPFN. No new neural proposals, worlds or references.
The original failed request remains failed; its missing success footers stay
missing. Only the actual newly observed CPU buffer/support replay is complete.
"""
from __future__ import annotations
import argparse,gzip,hashlib,importlib.util,json,os,pickle,random,subprocess,sys
from pathlib import Path
import numpy as np
from posteriorforge.artifacts import versions,sha256_file,write_json
from posteriorforge.science.analytical_benchmark_v2 import array_descriptor
from posteriorforge import prior_coordinates_v1 as codec

SCHEMA="posteriorforge.prior_coordinate_forensic_cohort.v1"
VARIANT="forensic_saved_prior_native_FP32_stream_v1"
COUNT=50000
EXPECTED_FIRST50000_ARRAY_SHA256="6b6c36843d6c213d086c6712aacc68dc1cb11148163ebe24bc74119a102b54eb"
FAILED_SHA256="c81001f878bc1cd04d144c4d98a22e2ab5a089698b58978e10773c33654bd488"
FAILED_PRODUCER_SHA256="71e5e2906fd663534e56878de1fc4aff55093e3d950a13f348f57417ca7c01d3"
CODEC_SHA256="d8ce6c49e7b65c6ee55c59b7243d4ce943cbea28c92ebed0329798074aaa70c5"
BUFFER_HELPER_SHA256="f8823ae75e7ae2eea6f1aa667af48ecf5c892450361b5abd490ccda36a28f7e0"
KERNEL_SHA256="e346e25bc9825abf016c2bd75addaf2b025d3e3d94e1a0ea6f6e31dd02997a43"
SCREEN_SHA256="5ec69eb7aadbd74c7d96ee534b71d0d8bee2b6dd6f6e3d2f44e74607452f682b"
MISSING_FOOTERS=("execution_counters","sampling_counters","default_rng_after","default_rng_preserved","sources_unchanged",
 "actual_prediction_checks","expected_patched_call_count","expected_stage_prepare_count","generated_proposal_wave_count")
POSTHOC_FILES={
 "src/posteriorforge/performance_prior_coordinates_v1.py":FAILED_PRODUCER_SHA256,
 "src/posteriorforge/execution_prior_native_v1.py":"e5a2733e767b187e573315547ec9253267e43d1e3700596f61e97116c53f3256",
 "src/posteriorforge/execution_prior_coordinates_v1.py":"88f28c5a1f01c086661243c5d3bfb59631eb3c95fe3567b7f1ede325ddba0aea",
 "src/posteriorforge/execution_proposal_buffer.py":"8fb2ba1fb63cd917378ff9bc8b6f70be7410862c509cd22cb2fb6f454d3d35fa",
 "src/posteriorforge/prior_coordinates_v1.py":CODEC_SHA256,
 "scripts/qualify_prior_native_conditionals_v1.py":"9221da7853d1ecea3606d9001c70589f61bb89a28717bbc333bc8a280ff81bf9"}

def require(value,message):
 if not value:raise ValueError(message)

def source_sha256():return sha256_file(__file__)

def require_committed():
 root=Path(__file__).resolve().parents[1]
 revision=subprocess.run(["git","-C",str(root),"rev-parse","HEAD"],check=True,capture_output=True,text=True).stdout.strip()
 recorded=subprocess.run(["git","-C",str(root),"show",revision+":"+str(Path(__file__).relative_to(root))],check=True,capture_output=True).stdout
 require(hashlib.sha256(recorded).hexdigest()==source_sha256(),"forensic replay source must be committed before execution")
 module=helper();relative=str(Path(module.__file__).resolve().relative_to(root))
 raw=subprocess.run(["git","-C",str(root),"show",revision+":"+relative],check=True,capture_output=True).stdout
 require(hashlib.sha256(raw).hexdigest()==BUFFER_HELPER_SHA256,"new physicalF64 helper Git bytes differ")
 return revision

def helper():
 from posteriorforge import execution_physical_fifo_v1 as module
 require(isinstance(BUFFER_HELPER_SHA256,str) and module.source_sha256()==BUFFER_HELPER_SHA256,"physicalF64 production helper not frozen")
 module.require_sources()
 return module

def posthoc_checks():
 root=Path(__file__).resolve().parents[1]
 for name,expected in POSTHOC_FILES.items():require(sha256_file(root/name)==expected,"posthoc frozen file differs: "+name)
 actual=Path("/scratch/project/eu-26-17/eubrieucb/posteriorforge/results/accuracy-prior-coordinates-v1/native-conditionals-01/manifest.json")
 if actual.is_file():raw=actual.read_bytes()
 else:raw=gzip.decompress((root/"evidence/accuracy-prior-coordinates-v1/native-conditionals-01/manifest.json.gz").read_bytes())
 require(hashlib.sha256(raw).hexdigest()==SCREEN_SHA256,"actual fresh finite execution proof bytes differ")
 return {"passed":True,"frozen_current_files":POSTHOC_FILES.copy(),"actual_finite_screen_sha256":SCREEN_SHA256,
  "scope":"NEW CPU posthoc file/proof-byte checks; not a recovered failed-GPU source or execution footer"}

def source_pins():
 require(codec.source_sha256()==CODEC_SHA256,"frozen codec changed");codec.require_sources()
 module=helper()
 return {"forensic_replay":source_sha256(),"physical_F64_postdecode_helper":BUFFER_HELPER_SHA256,
  "codec":CODEC_SHA256,"failed_neural_producer":FAILED_PRODUCER_SHA256,"failed_manifest":FAILED_SHA256,
  "finite_execution_screen":SCREEN_SHA256,"original_request_promoted_to_complete":False,"new_CPU_posthoc_verification":posthoc_checks()}

def missing_footer_observations(document):
 return {k:{"present":k in document,"recorded_value":document.get(k)} for k in MISSING_FOOTERS}

def authenticate_failed(path):
 """Authenticate all saved rows and their actual same-runtime strict decode."""
 path=Path(path);payload=path.read_bytes()
 require(hashlib.sha256(payload).hexdigest()==FAILED_SHA256,"exact failed original manifest required")
 d=json.loads(payload)
 require(d["status"]=="failed" and d["error"]=="unmodified CPU FP32 vectors required; no implicit casting"
  and d["runner_source_sha256"]==FAILED_PRODUCER_SHA256 and d["case"]==2 and d["context_seed"]==20261111,
  "forensic source must remain the exact failed request")
 require(d["matching_conditional_qualification_sha256"]==SCREEN_SHA256,"separate finite screen binding differs")
 root=Path(__file__).resolve().parents[1]
 for name,expected in POSTHOC_FILES.items():
  raw=subprocess.run(["git","-C",str(root),"show",d["git_revision"]+":"+name],check=True,capture_output=True).stdout
  require(hashlib.sha256(raw).hexdigest()==expected,"failed recorded original Git source differs: "+name)
 posthoc=posthoc_checks()
 require(all(d["versions"][k]==versions()[k] for k in ("python","numpy","scipy")),"same producer Python/NumPy/SciPy required for exact F64 decode")
 require(len(d["stage_events"])==21 and [e["stage_index"] for e in d["stage_events"]]==list(range(21)),"all21 actual stage events required")
 require(d["proposal_buffer"]["failed"] is True and d["proposal_buffer"]["finalized"] is False
  and d["proposal_buffer"]["original_api_delivered_count"]==0,"failed original FIFO was improperly completed")
 require(not path.with_name("samples.npz").exists(),"original failed request must have no fabricated output")
 matrices={};custody={}
 for label in ("latent_wave_archive","decoded_wave_archive","actual_stage_return_archive"):
  info=d[label];p=path.with_name(info["file"]);require(sha256_file(p)==info["sha256"],"saved forensic archive hash differs")
  with np.load(p,allow_pickle=False) as archive:matrices[label]={k:archive[k] for k in archive.files}
  custody[label]=dict(info)
 latent=matrices["latent_wave_archive"]["wave_0000"];physical=matrices["decoded_wave_archive"]["wave_0000"]
 require(latent.dtype==np.float32 and latent.shape==(51200,21) and np.isfinite(latent).all(),"actual full latent wave required")
 require(physical.dtype==np.float64 and physical.shape==(51200,21) and np.isfinite(physical).all(),"actual full decoded F64 wave required")
 returned=matrices["actual_stage_return_archive"]
 require(returned["filled_stage_mask"].tolist()==[True]*21 and returned["ordered_latent_prefix"].tobytes()==latent.tobytes()
  and all(returned[f"stage_{j:04d}"].dtype==np.float32 and returned[f"stage_{j:04d}"].tobytes()==latent[:,j].tobytes() for j in range(21)),
  "actual returned stage bytes differ from archived latent wave")
 require(codec.decode_ordered(latent).tobytes()==physical.tobytes(),"same-runtime complete decode bytes differ")
 require(array_descriptor(physical[:COUNT])["sha256"]==EXPECTED_FIRST50000_ARRAY_SHA256,"fixed first50000 physical raw-byte digest differs")
 from posteriorforge.inference import support_mask
 require(support_mask(physical).all(),"fixed saved wave contains unsupported rows; no replacement stream available")
 r=d["decode_wave_records"]
 require(len(r)==1 and r[0]["strict_decode_passed"] is True
  and r[0]["actual_latent_draws"]==array_descriptor(latent) and r[0]["decoded_physical"]==array_descriptor(physical),"recorded strict decode differs")
 require(d["compatibility_adapter"]["original_alias_restored"] and d["compatibility_adapter"]["sdk_sources_unchanged"],"saved corrected-CDF restoration differs")
 require(codec.source_sha256()==CODEC_SHA256,"codec changed during forensic authentication")
 return d,physical,{"failed_manifest_sha256":FAILED_SHA256,"original_request_status":"failed","saved_archives":custody,
  "same_runtime_full_decode_byte_replay":True,"all_actual51200_stage_returns_verified":True,
  "physical_rows":51200,"fixed_first50000_physical_array_sha256":EXPECTED_FIRST50000_ARRAY_SHA256,
  "missing_failed_request_success_footers":missing_footer_observations(d),
  "missing_original_success_footers_recovered":False,"original_full_request_execution_RNG_qualification_claimed":False,
  "new_CPU_posthoc_verification":posthoc,"recorded_failed_Git_blob_sources_verified":True}

def default_state(torch):
 return {"numpy_python":hashlib.sha256(pickle.dumps((np.random.get_state(),random.getstate()),protocol=5)).hexdigest(),
  "torch_CPU":hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()}

def replay_saved(physical,scope,module):
 """Execute THE new production postdecode/helper; never construct new proposals."""
 import torch
 require(isinstance(physical,np.ndarray) and physical.dtype==np.float64 and physical.shape==(51200,21),"actual51200 F64 physical rows required")
 original=array_descriptor(physical);calls=[]
 def provider(count):
  require(count==51200 and not calls,"only the fixed saved full wave may be provided once")
  calls.append(count)
  return torch.from_numpy(physical),None
 before=default_state(torch)
 selected,record=module.complete_request(provider,scope=scope)
 require(str(selected.dtype)=="torch.float64" and str(selected.device)=="cpu","helper must preserve physical CPU F64")
 values=selected.detach().numpy()
 require(values.shape==(COUNT,21) and values.tobytes()==physical[:COUNT].tobytes(),"original ordered first50000 supported rows required")
 require(array_descriptor(physical)==original,"actual saved physical wave mutated")
 after=default_state(torch);require(before==after,"new CPU replay changed default RNG")
 return values,record,{"provider_calls":calls,"new_CPU_replay_default_RNG_before":before,"new_CPU_replay_default_RNG_after":after,
  "new_CPU_replay_default_RNG_preserved":True,"failed_GPU_request_RNG_footers_recovered":False}

def run(*,failed_manifest,output):
 require(os.environ.get("CUDA_VISIBLE_DEVICES")=="","forensic CPU replay requires CUDA_VISIBLE_DEVICES empty")
 pins=source_pins();revision=require_committed()
 parent,physical,authentication=authenticate_failed(failed_manifest);module=helper()
 out=Path(output);out.mkdir(parents=True,exist_ok=False)
 scope={**parent["proposal_buffer"]["request_scope"],"forensic_from_failed_request":True,
  "failed_manifest_sha256":FAILED_SHA256,"production_request_completion_claimed":False}
 values,accounting,rng=replay_saved(physical,scope,module)
 with (out/"samples.npz").open("xb") as stream:np.savez(stream,samples=values)
 require(source_pins()==pins and sha256_file(failed_manifest)==FAILED_SHA256,"forensic source or failed parent changed")
 record={"schema":SCHEMA,"variant":VARIANT,"status":"forensic_complete","attribution":"Built with PriorLabs-TabPFN",
  "source_pins":pins,"git_revision":revision,"context_seed":20261111,"case":2,"repetition":0,"requested_count":COUNT,
  "parameter_order":parent["parameter_order"],"scientific_fingerprint":parent["scientific_fingerprint"],
  "raw_scientific_bindings":parent["raw_scientific_bindings"],"reference_binding":parent["reference_binding"],
  "authentication":authentication,"failed_parent_manifest_sha256":FAILED_SHA256,"original_request_status":"failed",
  "new_neural_proposals":0,"new_GPU_runs":0,"new_simulator_worlds":0,"new_reference_integrations":0,
  "selection_rule":"first50000 supported rows in original saved51200 proposal order; exact original support/FIFO helper",
  "selected_original_row_indices":{"start":0,"stop_exclusive":COUNT,"step":1},
  "all51200_proposal_archives_preserved_in_failed_parent":True,"production_qualification_claimed":False,
  "original_failed_request_promoted_to_complete":False,"original_missing_footers_recovered":False,
  "proposal_buffer":accounting,"CPU_replay_RNG_observation":rng,
  "samples_file":"samples.npz","samples":array_descriptor(values),"samples_sha256":sha256_file(out/"samples.npz")}
 write_json(out/"manifest.json",record)
 write_json(out/"checksums.json",{p.name:sha256_file(p) for p in out.iterdir() if p.is_file()})
 print(json.dumps({"output":str(out),"status":record["status"],"manifest_sha256":sha256_file(out/"manifest.json"),"samples_sha256":record["samples_sha256"]}),flush=True)
 return record

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--failed-manifest",required=True);p.add_argument("--output",required=True)
 run(**vars(p.parse_args()))
if __name__=="__main__":main()
