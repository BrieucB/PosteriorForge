"""Curate a CLOSED corrected pilot's metadata only. Built with PriorLabs-TabPFN.

Requires Root's terminal manifest SHA and reconciled Slurm receipt. Never opens
or copies NPZ/sample arrays, runs model/science code, writes the ledger, stages,
commits or submits jobs. Running manifests are rejected before output creation.
"""
from __future__ import annotations
import argparse,gzip,hashlib,json
from pathlib import Path

PRODUCER_SHA="a30e0c15513eb0c1a6ed3c68f0781327edef4714a244a119a3a4d4c68f8780bf"
FIFO_SHA="f8823ae75e7ae2eea6f1aa667af48ecf5c892450361b5abd490ccda36a28f7e0"
PROOF_SHA="5ec69eb7aadbd74c7d96ee534b71d0d8bee2b6dd6f6e3d2f44e74607452f682b"
SCHEMA="posteriorforge.inference.prior_physical_fifo.v1"
VARIANT="accuracy_prior_coordinates_native_fp32_physical_f64_fifo_v1"
NORMAL_FIELDS=("default_rng_after","default_rng_preserved","sources_unchanged",
 "execution_counters","sampling_counters","actual_prediction_checks","timings",
 "memory","rejection","samples","samples_sha256","completed_stage_indices")


def sha(data):return hashlib.sha256(data).hexdigest()


def authenticated_bytes(path,digest):
    if Path(path).suffix.lower() in (".npz",".npy"):
        raise ValueError("raw array archives are outside curation scope")
    raw=Path(path).read_bytes()
    if sha(raw)!=digest:raise ValueError("closed input SHA mismatch: "+str(path))
    return raw


def validate_closed(document,allocation,manifest_sha):
    if document.get("status") not in ("complete","failed"):
        raise ValueError("closed complete/failed manifest required; running output cannot be curated")
    if (document.get("schema"),document.get("variant"),document.get("runner_source_sha256"))!=(SCHEMA,VARIANT,PRODUCER_SHA):
        raise ValueError("exact corrected production family/source required")
    if (document.get("context_seed"),document.get("case"),document.get("context_size"),document.get("requested_count"))!=(20261111,2,4096,50000):
        raise ValueError("only the predeclared corrected case2 pilot is in scope")
    if document.get("execution_policy",{}).get("physical_fifo_source_sha256")!=FIFO_SHA or document.get("matching_conditional_qualification_sha256")!=PROOF_SHA:
        raise ValueError("shared FIFO or fresh conditional proof binding differs")
    if allocation.get("raw_manifest_sha256")!=manifest_sha or allocation.get("ledger_fully_reconciled") is not True:
        raise ValueError("Root's reconciled allocation must bind this exact closed manifest")
    slurm=allocation.get("Slurm",{})
    terminal=str(slurm.get("State","")).split()[0].rstrip("+")
    if terminal not in ("COMPLETED","FAILED","TIMEOUT","OUT_OF_MEMORY","CANCELLED","NODE_FAIL","PREEMPTED"):
        raise ValueError("terminal Slurm accounting required")
    if allocation.get("GPUs")!=1 or str(allocation.get("job_id"))!=str(document.get("slurm_job_id")):
        raise ValueError("same one-GPU job required")
    if document["status"]=="complete":
        if terminal!="COMPLETED":raise ValueError("complete manifest contradicts terminal scheduler state")
        cdf=document.get("compatibility_adapter",{})
        if not all(document.get(x) is True for x in ("sources_unchanged","default_rng_preserved")) or not all(cdf.get(x) is True for x in ("original_alias_restored","sdk_sources_unchanged")):
            raise ValueError("completed production guards absent or false")
        if document.get("completed_stage_indices")!=list(range(21)) or document.get("rejection",{}).get("returned")!=50000:
            raise ValueError("complete21/50000 footer required")
        if any(x not in document for x in NORMAL_FIELDS):raise ValueError("normal success footer incomplete")
    return True


def curate(*,manifest,manifest_sha256,allocation_receipt,allocation_sha256,output,slurm_raw=None):
    raw=authenticated_bytes(manifest,manifest_sha256)
    allocation_raw=authenticated_bytes(allocation_receipt,allocation_sha256)
    document=json.loads(raw);allocation=json.loads(allocation_raw)
    validate_closed(document,allocation,manifest_sha256)
    output=Path(output)
    if output.exists():raise FileExistsError("fresh evidence directory required")
    inputs=[(Path(manifest),manifest_sha256),(Path(allocation_receipt),allocation_sha256)]
    scheduler_raw=None
    if slurm_raw is not None:
        scheduler_raw=authenticated_bytes(slurm_raw,allocation["Slurm_raw_sha256"])
        inputs.append((Path(slurm_raw),allocation["Slurm_raw_sha256"]))
    output.mkdir(parents=True,exist_ok=False)
    (output/"manifest.json.gz").write_bytes(gzip.compress(raw,mtime=0))
    (output/"allocation-receipt.json").write_bytes(allocation_raw)
    if scheduler_raw is not None:(output/"Slurm-accounting.psv").write_bytes(scheduler_raw)
    def save(name,value):(output/name).write_text(json.dumps(value,indent=2,sort_keys=True)+"\n")
    source_keys=("runner_source_sha256","git_revision","git_dirty","versions","baseline_execution_source",
        "checkpoint_sha256","scientific_fingerprint","parameter_order","raw_scientific_bindings",
        "model_inputs","model_target_descriptors","coordinate_policy","native_execution_policy",
        "execution_policy","matching_conditional_qualification_sha256","matched_execution_screen",
        "reference_binding","preregistration_sha256","native_protocol_sha256")
    save("source-bindings.json",{k:document[k] for k in source_keys if k in document})
    stage_rows=document.get("stage_events",[])
    save("stage-timings.json",{"scope":"original host wall event records; not GPU kernel or CPU compute time",
        "rows":stage_rows,"recorded_stage_event_count":len(stage_rows),
        "prepare_seconds_sum":sum(x.get("prepare_seconds",0) for x in stage_rows),
        "predict_and_sample_seconds_sum":sum(x.get("predict_and_sample_seconds",0) for x in stage_rows),
        "nested_fit_and_move_intervals_may_overlap_prepare":True,
        "request_timings":document.get("timings"),"wrapper_total_seconds":document.get("wrapper_total_seconds"),
        "recorded_memory":document.get("memory")})
    metadata_keys=("schema","variant","status","error_type","error","case","context_seed","context_size",
        "requested_count","proposal_wave_sizes","rejection","proposal_buffer","sampling_counters",
        "actual_prediction_checks","compatibility_adapter","default_rng_before","default_rng_after",
        "default_rng_preserved","sources_unchanged","completed_stage_indices","observed_failure_footer",
        "failure_footer_capture_error","latent_wave_archive","decoded_wave_archive","actual_stage_return_archive",
        "stage_draw_observation","failure_forensics","samples","samples_file","samples_sha256","timings",
        "memory","wrapper_total_seconds","physical_G3_in_this_neural_request")
    summary={k:document[k] for k in metadata_keys if k in document}
    summary.update(original_manifest_sha256=manifest_sha256,original_manifest_bytes=len(raw),
        missing_normal_success_fields=[x for x in NORMAL_FIELDS if x not in document],
        public_raw_NPZ_or_sample_arrays_included=False,posterior_accuracy_claimed=False,
        separate_closed_scientific_assessment_required=True,old_c810_failure_reclassified=False,
        curator_source_sha256=sha(Path(__file__).read_bytes()),
        array_custody_scope="manifest descriptors only; no independent NPZ/array replay in this curator")
    save("summary.json",summary)
    (output/"README.md").write_text("# Corrected physical Float64 FIFO pilot: technical evidence\n\n"
        "Built with PriorLabs-TabPFN.\n\n"
        f"The closed manifest status is **{document['status']}**. This package preserves the exact "
        "manifest bytes as gzip, Root's reconciled one-GPU allocation receipt, source bindings, "
        "original host stage/request timing records and observed memory/guard fields.\n\n"
        "This is technical execution evidence. It makes no posterior-accuracy claim; the separate "
        "scientific assessment and any one physical joint-global update have distinct sources and outputs. "
        "The original c810 failed pilot remains failed, with its missing success guards unchanged.\n\n"
        "NPZ/sample/proposal arrays remain on Karolina and are not copied into Git. Their recorded "
        "hashes and descriptors are retained. This curator does not independently replay those arrays. "
        "Absent failure/success observations remain absent. No model, simulator, reference integration, "
        "new posterior draw, ledger mutation or GPU submission is performed by curation.\n\n"
        "Stage timings are host wall brackets, not CPU-compute or GPU-kernel attribution. Nested "
        "fit/movement timings can overlap preparation and must not be added twice.\n")
    for path,digest in inputs:
        if sha(path.read_bytes())!=digest:raise RuntimeError("closed input changed during curation")
    if gzip.decompress((output/"manifest.json.gz").read_bytes())!=raw:raise RuntimeError("raw manifest round-trip failed")
    checks="".join(sha(p.read_bytes())+"  "+p.name+"\n" for p in sorted(output.iterdir()) if p.is_file())
    (output/"SHA256SUMS").write_text(checks)
    return {"output":str(output),"status":document["status"],"manifest_sha256":manifest_sha256,
            "checksum_file_sha256":sha((output/"SHA256SUMS").read_bytes())}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("manifest","manifest-sha256","allocation-receipt","allocation-sha256","output"):
        p.add_argument("--"+key,required=True)
    p.add_argument("--slurm-raw")
    print(json.dumps(curate(**vars(p.parse_args())),sort_keys=True))


if __name__=="__main__":main()
