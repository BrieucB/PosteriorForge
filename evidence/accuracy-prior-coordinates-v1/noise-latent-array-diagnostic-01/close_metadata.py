"""Close the metadata-only saved-noise diagnostic package; no numerical replay."""
from pathlib import Path
import hashlib, json, shutil

ROOT = Path("/scratch/project/eu-26-17/eubrieucb/posteriorforge")
SCRATCH = ROOT / "results/accuracy-prior-coordinates-v1/noise-latent-array-diagnostic-01"
PUBLIC = Path("/home/it4i-bbenvegnen/workspace/PosteriorForge/evidence/accuracy-prior-coordinates-v1/noise-latent-array-diagnostic-01")
MANIFEST = ROOT / "results/accuracy-prior-coordinates-v1/native-pilot-v2-seed20261111-case0-01/manifest.json"
EXPECTED_DIAGNOSTIC = "c08076c86e7d329f65ddb417224c2430ceb4b3dfdb0e24e00ae8f6000e691f93"
EXPECTED_MANIFEST = "92755b788fc894d4ca16157e573bd1b5c9d8a4468b71cf184fa16303bced7bce"

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def write_json(path, document):
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")

assert sha(SCRATCH / "diagnostic.json") == EXPECTED_DIAGNOSTIC
assert sha(MANIFEST) == EXPECTED_MANIFEST
d = json.loads((SCRATCH / "diagnostic.json").read_bytes())
m = json.loads(MANIFEST.read_bytes())
border_records = []
def walk(value, path=""):
    if isinstance(value, dict):
        for key, child in value.items():
            here = path + "." + key
            if "border" in key.lower():
                record = {"path": here, "type": type(child).__name__}
                if isinstance(child, dict):
                    record["descriptor"] = child
                elif isinstance(child, (str, bool, int, float)) or child is None:
                    record["value"] = child
                elif isinstance(child, list):
                    record["length"] = len(child)
                border_records.append(record)
            walk(child, here)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            walk(child, path + "[" + str(index) + "]")
walk(m)
assert len(border_records) == 165
containers = [r for r in border_records if r["type"] == "dict"]
assert len(containers) == 6
assert all(set(r["descriptor"]) == {"dtype", "shape", "sha256"} for r in containers)
assert all(any("[" + str(j) + "]" in r["path"] for j in (6, 10, 14)) for r in containers)
border_inventory = {
    "schema": "posteriorforge.readonly_saved_border_metadata_inventory.v1",
    "source_manifest_sha256": EXPECTED_MANIFEST,
    "border_named_fields": len(border_records),
    "all_border_named_fields": border_records,
    "numeric_border_arrays_or_numeric_border_minmax_recorded": False,
    "native_stages_6_10_14_criterion_borders_have_only_dtype_shape_hash_descriptors": True,
    "failing_stages_15_19_record_only_border_dtypes": True,
    "selected_head_uniform_logits_bin_cdf_fraction_available": False,
    "scope": "Full existing manifest key scan only; no prediction, array decoding, or source modification.",
}
write_json(SCRATCH / "border-metadata-inventory.json", border_inventory)
rows = []
for r in d["noise_distributions"]:
    g = r["generated_all51200"]
    t = r["training_prior4096_model_Float32"]
    f = r["original_pure_FIRST1024_fixture_model_Float32"]
    rows.append({
        "ordered_stage": r["stage"], "latent_name": r["latent_name"],
        "generated_minimum": g["minimum"], "generated_q99": g["quantiles"]["0.99"],
        "generated_q999": g["quantiles"]["0.999"], "generated_maximum": g["maximum"],
        "training_minimum": t["minimum"], "training_maximum": t["maximum"],
        "original_fixture_minimum": f["minimum"], "original_fixture_maximum": f["maximum"],
    })
summary = {
    "schema": "posteriorforge.noise_latent_failure_diagnostic_summary.v1",
    "status": "diagnosis_complete_request_remains_failed",
    "request_manifest_sha256": EXPECTED_MANIFEST,
    "diagnostic_sha256": EXPECTED_DIAGNOSTIC,
    "diagnostic_code_sha256": d["diagnostic_code_sha256"],
    "latent_archive_sha256": d["latent_archive"]["sha256"],
    "stage_return_archive_sha256": d["stage_return_archive"]["sha256"],
    "raw_latent_array_descriptor": d["latent_archive"]["descriptor"],
    "all_21_stage_return_Float32_vectors_match_ordered_wave_without_extra_cast": True,
    "all_6_noise_coordinates_examined_for_all_51200_rows": True,
    "offending_noise_cells": d["all_offending_noise_cells"],
    "unique_offending_rows": d["unique_offending_rows"],
    "counts": {"raw_rows": 51200, "noise_coordinates": 307200, "offending_rows": 2, "offending_noise_coordinates": 2, "first50000_offending_rows": 2, "nominal_lookahead1200_offending_rows": 0},
    "partitions": d["partitions"],
    "partition_semantics": "The diagnostic raw label unused1200tail denotes only raw-provider row positions [50000,51200). No physical wave reached FIFO, so no actual buffer tail was finalized or accepted.",
    "ordinary_noise_summary": rows,
    "ordinary_quantile_method": "NumPy linear, fixed grid [0,.001,.01,.05,.5,.95,.99,.999,1]; complete table in diagnostic.json",
    "comparison_scope": d["distribution_scope"],
    "scalar_codec_pre_rejection_outputs": [0.5, 0.5],
    "saved_first_failure_records_match_actual_npz": True,
    "strict_failure_source": d["exact_exception_scope"],
    "saved_border_metadata_sha256": sha(SCRATCH / "border-metadata-inventory.json"),
    "historical_prediction_witnesses_missing": ["actual_selected_member", "actual_uniform", "actual_logits_or_probabilities", "actual_prediction_canonical_or_physical_border_values", "actual_selected_bin", "actual_CDF", "actual_interpolation_fraction"],
    "failure_guard_scope": d["available_failure_guard_scope"],
    "normal_success_fields_missing": d["normal_success_fields_missing"],
    "literal_completed_stage_indices": m["completed_stage_indices"],
    "actual_stage_return_count": 21,
    "git_dirty": m["git_dirty"],
    "failed_wrapper_total_seconds": m["wrapper_total_seconds"],
    "no_completed_50000_cohort": True,
    "no_new_model_sampler_world_reference_G3_or_RNG_state_capture": True,
    "no_clipping_filtering_repair_reordering_or_recovery": True,
    "new_generated_full21D_physical_matrix_assembled_or_saved": False,
    "source_hashes_verified_before_and_after": d["sources_before_and_after"],
    "runtime_versions": d["runtime_versions"],
    "independent_math_evidence": {
        "relative_public_folder": "../noise-inverse-math-diagnostic-01",
        "original_report_sha256": "d56e2a006a92a3e377b9f80ecb43e6e42afb809f1081c0a365903cb3bc3f5eb2",
        "reproducible_scalar_report_sha256": "d5a22a8a617023ced6d1b043ecca6185bf384a5dcd33bb2ff862050f430028e5",
        "scope": "Separate scalar mathematical analysis, not a recovered physical cohort or a causal explanation of the generated latent values.",
    },
    "next_evidence_ranking_only": [
        "If separately authorized, a sharply capped conditional/ICDF observation at saved offending prefixes is the most direct way to distinguish wide current border support from rounded-CDF extrapolation. Record all8 current head borders/probability normalization plus selected u/head/bin/fraction. Such a new observation cannot recreate unavailable historical witnesses.",
        "Consider a numerical-policy or coordinate intervention only after that mechanism evidence; current archives do not choose between them.",
        "An additional full posterior request is not needed to obtain that missing conditional/ICDF evidence."
    ],
}
write_json(SCRATCH / "summary.json", summary)
table = "\n".join(
    f"stage {r['ordered_stage']:2d} {r['latent_name']:12s} generated[{r['generated_minimum']:.9g},{r['generated_maximum']:.9g}] q99={r['generated_q99']:.9g} q999={r['generated_q999']:.9g}; training[{r['training_minimum']:.9g},{r['training_maximum']:.9g}]; originalFIRST1024[{r['original_fixture_minimum']:.9g},{r['original_fixture_maximum']:.9g}]"
    for r in rows
)
readme = f"""READ-ONLY SAVED NOISE-LATENT DIAGNOSIS — request remains FAILED

This package authenticates the existing failed seed20261111/case0 v2 request,
manifest {EXPECTED_MANIFEST}. It contains metadata, scalar diagnostic
values, and deterministic diagnostic code only. Raw sample/latent arrays remain
on Karolina and are not included here. Built with PriorLabs-TabPFN.

RESULT
Exactly 2 offending noise coordinates in 2 of 51,200 raw rows (0.00390625% of
rows; 2/307,200 noise scalar coordinates). Both are within raw row positions
[0,50000): 2/50,000 = 0.004% of those rows. None is in positions [50000,51200).
These last 1,200 positions are a NOMINAL raw-provider lookahead tail, not an
actually finalized unused physical FIFO buffer: strict decode failed before
the provider returned any physical wave, support calls=0, delivered=0,
FIFO finalized=False. No completed 50,000-row cohort or accuracy result exists.

Zero-based offending entries:
- row 4355, ordered stage 15, physical slot 16 sigma_a1:
  Float32 latent 13.876729011535645, bytes 15075e41.
- row 41765, ordered stage 19, physical slot 20 sigma_a3:
  Float32 latent 205.2462615966797, bytes 0b3f4d43.
Both actual frozen d8 scalar noise inverses produce 0.5  before rejection.
The saved first-failure row/stage/slot/value/hex/full-row bytes match the NPZ.
No other noise coordinate fails the unchanged component decode/encode and
declared roundtrip checks. All 51,200 rows and all 6 noise slots were examined.

EXACT SOURCE / ARCHIVES
Frozen codec d8ce6c49e7b65c6ee55c59b7243d4ce943cbea28c92ebed0329798074aaa70c5:
prior_coordinates_v1.py:137 calls strict_noise; line110 rejects endpoint,
out-of-support or saturation; line43 raises CoordinateError. Full causal
prefix decode calls the noise inverse at176 via decode_ordered_prefix:234.
F32->F64 widening is the unchanged operation at233. No tolerance was changed.

latent-proposal-waves.npz {d['latent_archive']['sha256']}
contains one Float32 [51200,21] ORDERED LATENT wave, not physical samples.
Its raw array SHA is{d['latent_archive']['descriptor']['sha256']}.
latent-stage-returns.npz {d['stage_return_archive']['sha256']}
contains 21 actual Float32 [51200] stage-return vectors, a bool [21] filled mask,
and a duplicate Float32 ordered prefix. All 21 vectors directly byte-match
their ordered wave columns; no extra cast is needed. This verifies original
performance_runner.py:470,472 storage convention (source 23ccad8b...), preserved
by SS15 wave 4037d488.... Both archives and the failed manifest remain unchanged.

COMPARISON TABLE (all51200 generated latent values)
{table}

Ordinary quantiles use NumPy linear on the fixed grid recorded in diagnostic.json.
The 4096 training values are from the authenticated ORIGINAL seed11 prior bank.
The FIRST 1024 fixture is from the authenticated ORIGINAL pure pre-G3 physical
case0 cohort (manifest 3e0c158f..., payload a9ecadf2...), encoded by the exact
frozen noise map and one declared Float32 cast. Generated values condition on
held-out case 0 and generated prefixes; the bank is unconditional prior data and
the old fixture is an earlier learned-model output. These range/quantile
comparisons are descriptive, not a reference-posterior or calibration test.
Training-range exceedance is not evidence of canonical-border exceedance.

MECHANISM / REPRESENTABILITY
The full saved manifest contains 165 border-named fields: dtype strings and
six dtype/shape/SHA descriptors for native-stage 6/10/14 criterion borders.
For the offending cached stages 15/19 it records border dtypes only, no numeric
prediction-specific border values or min/max. Historical selected member,
uniform, logits/probabilities, bin, CDF, and interpolation fraction are absent.
The archive cannot distinguish wide-border ordinary interpolation from
rounded-CDF extrapolation, or attribute the latent extremes to either.

Separate sibling evidence ../noise-inverse-math-diagnostic-01 reports that the
ideal prior inverse values for these two z are closer to 0.5 than half its F64
predecessor gap: a correctly rounded inverse is endpoint 0.5 for both.
That scalar analysis explains strict representability failure for the SAVED
values; it does not explain why the neural/ICDF path produced them.

If separately authorized, the highest-information next diagnostic is a sharply
capped conditional/ICDF observation at the saved offending prefixes, recording
all 8 current head borders, probability normalization, and u/head/bin/fraction.
New observations cannot recreate missing historical witnesses. Numerical-policy
or coordinate changes should be considered after mechanism evidence; an extra
full posterior request is not needed for this evidence. This package authorizes
no probe, retry, policy change, filtering, repair, or recovery.

OBSERVED GUARDS AND LIMITS
Existing genuine exception observations show all 21 stage returns/aliases restored,
CDF 1141 and restored alias, postexception source closure passed, and observed
default RNG equality. The literal completed_stage_indices remains[] and git_dirty
remains True. Failed wrapper timing 777.6754412082955s is not a successful request
latency. Missing normal-success execution/sampling/prediction/memory/rejection
footers remain missing; see the exact list in summary.json. No failure observations
are relabeled as a normal success footer.

Diagnostic scope: exact per-noise frozen functions on unchanged contiguous
components, recursive subdivision only to locate failures, all rows retained
and accounted. No generated full 21D physical matrix was assembled or saved.
No model, sampler, world, reference integration, G3, random draw/state capture,
clipping, filtering, repair, resampling, coordinate/source/threshold change.
Torch was not imported. Sources and original data SHA verified before/after.
The deterministic diagnostic remains independent of any future experiment.

REPRODUCE (authorized saved HPC artifacts required)
cd /home/it4i-bbenvegnen/workspace/PosteriorForge
PYTHONPATH=src OMP_NUM_THREADS=16 /scratch/project/eu-26-17/eubrieucb/posteriorforge/envs/inference/bin/python \\
  <scratch-folder>/diagnose.py --repo /home/it4i-bbenvegnen/workspace/PosteriorForge \\
  --scratch-root /scratch/project/eu-26-17/eubrieucb/posteriorforge
Runtime versions: Python 3.11.13, NumPy 1.26.4, SciPy 1.12.0.
diagnostic.json SHA {EXPECTED_DIAGNOSTIC}
No code or raw artifact from the failed request is modified.
"""
(SCRATCH / "README.txt").write_text(readme)
PUBLIC.mkdir(parents=True, exist_ok=True)
names = ["diagnose.py", "diagnostic.json", "diagnostic.log", "border-metadata-inventory.json", "summary.json", "README.txt", "close_metadata.py"]
for name in names:
    shutil.copyfile(SCRATCH / name, PUBLIC / name)
checks = {"schema": "posteriorforge.public_evidence_checksums.v1", "files": {name: sha(PUBLIC / name) for name in sorted(names)}}
write_json(PUBLIC / "checksums.json", checks)
shutil.copyfile(PUBLIC / "checksums.json", SCRATCH / "checksums.json")
assert sha(MANIFEST) == EXPECTED_MANIFEST and sha(SCRATCH / "diagnostic.json") == EXPECTED_DIAGNOSTIC
print(json.dumps({"public_directory": str(PUBLIC), "file_count": len(names), "diagnostic_sha256": EXPECTED_DIAGNOSTIC, "summary_sha256": sha(PUBLIC / "summary.json"), "readme_sha256": sha(PUBLIC / "README.txt"), "checksums_sha256": sha(PUBLIC / "checksums.json")}, sort_keys=True))
