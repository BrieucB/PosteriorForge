"""Stdlib-only curation of exact counterfactual capture metadata; no array loading."""
from pathlib import Path
import gzip, hashlib, json, shutil
ROOT=Path("/scratch/project/eu-26-17/eubrieucb/posteriorforge")
PHASE=ROOT/"results/accuracy-prior-coordinates-v1"
CAPTURE=PHASE/"tail-conditionals-capture-01"
LAUNCH=PHASE/"tail-conditionals-root-launch-01"
PREFLIGHT=PHASE/"tail-conditionals-preflight-01/manifest.json"
PUBLIC=Path("/home/it4i-bbenvegnen/workspace/PosteriorForge/evidence/accuracy-prior-coordinates-v1/tail-conditionals-capture-01")
RAW_SHA="cf11be3c542eb11c7ed9423845eb1f80fb046df59a2b85254cb9f20d25663461"
TRACE_SHA="21869db939e39d81326679fd1ac7addd9867a6bd1718acaa087ca3561c0b7629"
COST_SHA="f2779102404a8110599ac5f988d93899268250a7ce5e87344adfe60bd6b24716"
PREFLIGHT_SHA="def6902c0bb03d473587984a304540fde3ff914e5335982cf4f8c82fd511fed9"
def sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as stream:
  for block in iter(lambda:stream.read(1024*1024),b""):h.update(block)
 return h.hexdigest()
def write_json(path,value):
 path.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n")
assert sha(CAPTURE/"manifest.json")==RAW_SHA
assert sha(LAUNCH/"allocation-receipt.json")==COST_SHA
assert sha(PREFLIGHT)==PREFLIGHT_SHA
raw=(CAPTURE/"manifest.json").read_bytes();d=json.loads(raw)
cost=json.loads((LAUNCH/"allocation-receipt.json").read_bytes())
assert d["status"]=="two_stage_conditional_capture_complete"
assert d["counterfactual_reevaluation"] is True and d["historical_grouping_member_recreated"] is False
assert d["posterior_vectors_generated"]==d["G3_updates"]==d["reference_integrations"]==d["uniform_generation_calls"]==0
assert d["sources_unchanged"] is True and d["default_RNG_preserved"] is True
assert d["default_RNG_before"]==d["default_RNG_after"]
assert cost["raw_capture_manifest_sha256"]==RAW_SHA
assert cost["accounting_record"]["JobIDRaw"]=="5411602" and cost["accounting_record"]["ElapsedRaw"]=="53"
assert cost["accounting_record"]["Restarts"]=="0" and cost["accounting_record"]["State"]=="COMPLETED"
assert cost["job"]["actual_gpu_seconds"]==53 and cost["job"]["gpus"]==1
assert cost["budget_after"]["actual_gpu_seconds"]==104974 and cost["budget_after"]["reserved_gpu_seconds"]==0
assert "--no-requeue" in cost["job"]["submitted_argv"]
assert d["profile"]["sha256"]==TRACE_SHA and sha(CAPTURE/d["profile"]["file"])==TRACE_SHA
npz=[];counts={}
for row in d["stages"]:
 stage=row["fixture"]["stage"];counts[str(stage)]={}
 for arm,entries in row["arms"].items():
  labels={}
  for e in entries:
   path=CAPTURE/e["file"]
   assert path.suffix==".npz" and sha(path)==e["sha256"]
   labels[e["label"]]=labels.get(e["label"],0)+1
   npz.append({"stage":stage,"arm":arm,"member":e["member"],"label":e["label"],
               "file":e["file"],"bytes":path.stat().st_size,"sha256":e["sha256"],
               "arrays":e["arrays"],"actual_query_input_descriptor":e["input"],
               "configuration_identity":e["configuration"],
               "scope":"New counterfactual conditional/ICDF observation; not a historical member/bin/ICDF witness."})
  counts[str(stage)][arm]=labels
assert len(npz)==48 and len(set(e["file"] for e in npz))==48
assert set(x.name for x in CAPTURE.glob("*.npz"))==set(e["file"] for e in npz)
assert counts=={"15":{"candidate":{"N1024":8,"N1":8},"original":{"N1024":8,"N1":8}},
               "19":{"candidate":{"N1024":8},"original":{"N1024":8}}}
other=[{"file":x.name,"bytes":x.stat().st_size,"sha256":sha(x),"role":"Fitted configuration/target-transform serialization kept on HPC; not copied into Git."}
       for x in sorted(CAPTURE.glob("*.pkl"))]
PUBLIC.mkdir(parents=True,exist_ok=True)
assert not any(PUBLIC.iterdir()),"fresh evidence directory required"
(PUBLIC/"capture-manifest.json.gz").write_bytes(gzip.compress(raw,mtime=0))
assert gzip.decompress((PUBLIC/"capture-manifest.json.gz").read_bytes())==raw
(PUBLIC/"capture-manifest.raw.sha256").write_text(RAW_SHA+"  manifest.json\n")
write_json(PUBLIC/"capture-manifest-description.json",{"raw_basename":"manifest.json","raw_bytes":len(raw),"raw_sha256":RAW_SHA,"gzip_file":"capture-manifest.json.gz","gzip_sha256":sha(PUBLIC/"capture-manifest.json.gz"),"exact_original_bytes_after_decompression":True})
for source,target in [
 (LAUNCH/"allocation-receipt.json","allocation-receipt.json"),
 (LAUNCH/"allocation-sacct.psv","allocation-sacct.psv"),
 (LAUNCH/"submission.json","submission.json"),
 (LAUNCH/"preflight.json","root-launch-preflight.json"),
 (PREFLIGHT,"input-preflight.json"),
]:
 shutil.copyfile(source,PUBLIC/target)
logs=[]
for suffix in ("out","err"):
 source=ROOT/("logs/pf-tail-cond-pair0-5411602."+suffix)
 entry={"source_path":str(source),"bytes":source.stat().st_size,"sha256":sha(source)}
 if source.stat().st_size<=65536:
  entry["public_file"]="slurm-5411602."+suffix
  shutil.copyfile(source,PUBLIC/entry["public_file"])
 logs.append(entry)
inventory={"schema":"posteriorforge.counterfactual_capture_artifact_descriptors.v1",
 "raw_manifest_sha256":RAW_SHA,
 "raw_npz_files":npz,"raw_npz_file_count":len(npz),
 "raw_fitted_object_files":other,"raw_fitted_object_file_count":len(other),
 "trace":{"file":d["profile"]["file"],"bytes":d["profile"]["bytes"],"sha256":TRACE_SHA,"public_copy_included":False},
 "preflight_inputs":{"file":d["preflight"]["inputs_file"],"sha256":d["preflight"]["inputs_sha256"],"arrays":d["preflight"]["arrays"],"public_copy_included":False},
 "small_raw_Slurm_logs":logs,
 "scope":"File SHA checks and copies only. No NPZ loading, unpickling, numerical re-estimation, or interpretation; dense arrays, fitted pickles and full trace remain on HPC."}
write_json(PUBLIC/"raw-artifact-descriptors.json",inventory)
cdf=d["compatibility_adapter"]
assert cdf["patched_call_count"]==48 and cdf["original_alias_restored"] and cdf["sdk_sources_unchanged"]
assert all(r["candidate_session"]["aliases_and_flags_restored"] for r in d["stages"])
summary={
 "schema":"posteriorforge.tail_conditional_capture_public_scope.v1",
 "status":d["status"],"raw_capture_manifest_sha256":RAW_SHA,
 "job_id":"5411602","source_commit":cost["job"]["metadata"]["candidate_commit"],
 "source_pins":d["sources"],
 "input_failure_manifest_sha256":d["preflight"]["failed_manifest_sha256"],
 "input_latent_wave_sha256":d["preflight"]["actual_wave_sha256"],
 "input_preflight_sha256":PREFLIGHT_SHA,
 "uniform_reconstruction_sha256":d["preflight"]["uniform_reconstruction_sha256"],
 "scope":"Counterfactual FP32 conditional re-evaluation of two saved-prefix windows; no historical member/grouped-batch/ICDF reconstruction claim.",
 "fixtures":[{k:r["fixture"][k] for k in ("stage","row","window","focal_offset","target_coordinate")} for r in d["stages"]],
 "original_arm":"Fresh original TabPFN fit_preprocessors FP32 all8 selected-member controls; outside precision Session.",
 "candidate_arm":"Frozen0199 backend, stages15/19 normal cached FP32 route, all8.",
 "head_prediction_capture_counts":counts,"total_selected_head_query_captures":48,
 "provided_uniform_bits":d["preflight"]["uniform_bits"],"provided_uniform_values":d["preflight"]["uniform_values"],
 "uniform_generation_calls":d["uniform_generation_calls"],
 "reconstructed_U_scope":"Provided component uniforms are independently reconstructed from historical private-stream states; current head/bin/probability/ICDF observations are counterfactual new observations.",
 "actual_footer":{"default_RNG_preserved":d["default_RNG_preserved"],"default_RNG_before":d["default_RNG_before"],"default_RNG_after":d["default_RNG_after"],"sources_unchanged":d["sources_unchanged"],"CDF_calls":cdf["patched_call_count"],"CDF_alias_restored":cdf["original_alias_restored"],"SDK_sources_unchanged":cdf["sdk_sources_unchanged"],"candidate_precision_sessions_restored":[r["candidate_session"]["aliases_and_flags_restored"] for r in d["stages"]],"post_exit_observation_statuses":{k:v["status"] for k,v in d["post_exit_observations"].items()}},
 "RNG_scope":d["RNG_scope"],
 "capture_elapsed_seconds":d["elapsed_seconds"],
 "timing_scope":"Counterfactual capture including tracing/serialization; not a posterior latency or clean performance measurement.",
 "allocation":{"allocation_receipt_sha256":COST_SHA,"job_name":cost["accounting_record"]["JobName"],"state":"COMPLETED","gpus":1,"elapsed_allocated_seconds":53,"actual_GPU_seconds":53,"reserved_GPU_seconds_at_submission":300,"no_requeue":True,"restarts":0,"project_actual_GPU_seconds_after":104974,"project_reserved_GPU_seconds_after":0},
 "profile":d["profile"],
 "raw_NPZ_descriptors_sha256":sha(PUBLIC/"raw-artifact-descriptors.json"),
 "raw_manifest_retains_numeric_comparisons_and_N1_decisions_without_relabeling":True,
 "numerical_interpretation_in_this_summary":False,
 "normalized_reference_analysis":"Separate subsequent offline report; no reference integration was performed by this capture or metadata packaging.",
 "posterior_vectors_generated":0,"G3_updates":0,"reference_integrations":0,"FP16_activated":False,
 "source_policy_changes_or_repairs":False,
 "raw_dense_arrays_pickles_trace_included_in_Git":False,
 "attribution":"Built with PriorLabs-TabPFN",
}
write_json(PUBLIC/"summary.json",summary)
readme="""COUNTERFACTUAL CONDITIONAL CAPTURE — METADATA PACKAGE

This package preserves the exact raw manifest and file descriptors from one
bounded counterfactual FP32 conditional/ICDF capture, job 5411602. It contains
no dense arrays, fitted-object pickles, or full CUDA trace. Built with
PriorLabs-TabPFN.

SCOPE
Two fixed saved-prefix windows were evaluated: ordered stage 15, raw rows
[4096,5120), focal row 4355 (offset 259); and stage 19, rows [40960,41984),
focal row 41765 (offset 805). Both use the authenticated seed20261111/case0
training bank, fixed SS15 inputs, latent labels and masks. The original arm
is fresh stock FP32 fit_preprocessors all8, outside a precision Session;
the candidate arm uses the frozen0199 cached FP32 path at these two stages.

All8 heads have N1024 captures in both arms. The predeclared trigger additionally
caused all8 N1 captures for stage15 in both arms. There are48 selected-head query
capture files total; the actual CDF footer records48 calls. The exact raw
manifest retains every original comparison/trigger/status field. This public
summary performs no numerical interpretation; Science's separate offline
analysis owns that report.

These are NEW COUNTERFACTUAL observations. The contiguous1024 windows are not
authenticated historical member-grouped batches. No historical selected
member, bin, probabilities, logits, or ICDF result is claimed to have been
recreated. Component uniforms were independently reconstructed from historical
private-stream states and supplied as fixed values; no new uniform stream was
drawn by the capture. Additional declared fixed U values and the u=1 guard
stress are diagnostic inputs, not posterior samples.

EXACT BYTES / DESCRIPTORS
capture-manifest.json.gz decompresses byte-for-byte to original manifest SHA:
cf11be3c542eb11c7ed9423845eb1f80fb046df59a2b85254cb9f20d25663461.
The immutable raw SHA sidecar and gzip/raw descriptors are included.
raw-artifact-descriptors.json records each of48 NPZ files' exact file SHA,
disk size, and all actual array dtype/shape/SHA descriptors copied from the raw
manifest. Every file SHA was verified by streaming its bytes; no NPZ was loaded.
Fitted configuration/target-transform pickle file hashes and trace descriptors
are preserved, with those raw files kept on HPC. Small raw Slurm stdout/stderr
are included exactly, together with submission, sacct, allocation receipt,
root launch preflight, and exact input CPU preflight records.

ACTUAL CLOSURE / COST
The raw success footer records source identity unchanged and exact default
NumPy/Python/Torch CPU/CUDA RNG equality; corrected CDF executed48 times and the
alias/SDK source restored. Both candidate precision Sessions restored their
aliases/flags. The report honestly scopes internal preprocessing RNG calls as
not instrumented, rather than claiming zero internal calls.
Capture elapsed time:48.31360623938963s, including tracing/serialization.
This is not a completed posterior request or clean performance latency.

Slurm job5411602 completed with1 GPU,53 allocated GPU-seconds, zero restarts,
and explicit --no-requeue, within a300-second reservation. The exact allocation
receipt is SHA f2779102404a8110599ac5f988d93899268250a7ce5e87344adfe60bd6b24716.
Root reconciled project actual104974 GPU-seconds and zero reserved after this
allocation. Raw submitted and reconciled records are distinct and retained;
the submitted snapshot's zero actual cost is not the final accounting.

NO POSTERIOR / RECOVERY CLAIM
The original failed request92755b... and all earlier diagnostic packages remain
unchanged. This capture produced zero posterior vectors, G3 updates, reference
integrations, or FP16 activations. It changes no production source, codec,
sampler, prior, feature, threshold, or policy and performs no repair, clipping,
filtering, salvage, or retry posterior. The full trace is SHA
21869db939e39d81326679fd1ac7addd9867a6bd1718acaa087ca3561c0b7629
and remains on HPC; its elapsed CUDA spans are not exclusive kernel unions.
Any normalized reference analysis and numerical conclusions belong to the
separate subsequent Science report, not this curation package.
"""
(PUBLIC/"README.txt").write_text(readme)
shutil.copyfile(__file__,PUBLIC/"curate_metadata.py")
files={f.name:sha(f) for f in sorted(PUBLIC.iterdir()) if f.is_file()}
write_json(PUBLIC/"checksums.json",{"schema":"posteriorforge.public_evidence_checksums.v1","files":files})
assert all(sha(PUBLIC/n)==h for n,h in files.items())
assert not any(x.suffix.lower() in (".npz",".npy",".pkl",".pt",".ckpt") for x in PUBLIC.iterdir())
assert sha(CAPTURE/"manifest.json")==RAW_SHA
print(json.dumps({"directory":str(PUBLIC),"checksums_sha256":sha(PUBLIC/"checksums.json"),"summary_sha256":sha(PUBLIC/"summary.json"),"raw_artifact_descriptors_sha256":sha(PUBLIC/"raw-artifact-descriptors.json"),"gzip_sha256":sha(PUBLIC/"capture-manifest.json.gz"),"file_count":len(files)+1,"NPZ_files_described_and_file_hash_verified":len(npz),"raw_manifest_unchanged":True,"stdlib_only_no_array_loads":True},sort_keys=True))
