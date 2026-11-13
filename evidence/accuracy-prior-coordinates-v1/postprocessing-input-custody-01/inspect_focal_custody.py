"""Saved focal-array/source custody only; no model, transforms, RNG or softmax calls."""
from pathlib import Path
import hashlib, json, platform, sys, subprocess
import numpy as np
ROOT=Path("/scratch/project/eu-26-17/eubrieucb/posteriorforge")
REPO=Path("/home/it4i-bbenvegnen/workspace/PosteriorForge")
CAP=ROOT/"results/accuracy-prior-coordinates-v1/tail-conditionals-capture-01"
OUT=ROOT/"results/accuracy-prior-coordinates-v1/postprocessing-input-custody-01"
EXPECTED="cf11be3c542eb11c7ed9423845eb1f80fb046df59a2b85254cb9f20d25663461"
def sha_file(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for block in iter(lambda:f.read(1048576),b""):h.update(block)
 return h.hexdigest()
def descriptor(a):
 a=np.asarray(a)
 return {"dtype":a.dtype.str,"shape":list(a.shape),"sha256":hashlib.sha256(a.tobytes()).hexdigest()}
def scalar(v):
 return float(v) if np.isfinite(v) else str(float(v))
assert sha_file(CAP/"manifest.json")==EXPECTED
d=json.loads((CAP/"manifest.json").read_bytes())
sdk=ROOT/"envs/inference/lib/python3.11/site-packages/tabpfn"
sources={
 REPO/"scripts/diagnose_prior_tail_conditionals_v1.py":"883fb5840cf3460fa5162b34d0fc24a43b43d70bb3bcff3585d0cfd3e483cc38",
 sdk/"regressor.py":"bbb7d6dec265b18a30a54ff28719c957bc05d4164f693f76d998380132d2c46e",
 sdk/"utils.py":"a0521d68cd2880e041a49ac5acee42a2b8d1297e246e3f7a9c477e49f5e97fc1",
 REPO/"src/posteriorforge/execution_mixture.py":"e71f43b73ae8259c20b5d7b52feccc7c851e9a2fbb86d869bc0a5f7f3af792a1",
 REPO/"src/posteriorforge/compat_cdf.py":"7a86fbbefdb0fa7e850a394bea8af710c2e5f0139c10cdce1fd272fb9709eec0",
}
# First-party callback bytes are also compared with the actual capture commit.
recorded_commit="375426be0041974405d454b14fc4860447d8ccd8"
hybrid_path=REPO/"src/posteriorforge/execution_hybrid.py"
hybrid_git=subprocess.run(["git","-C",str(REPO),"show",recorded_commit+":src/posteriorforge/execution_hybrid.py"],check=True,capture_output=True).stdout
sources[hybrid_path]=hashlib.sha256(hybrid_git).hexdigest()
assert all(sha_file(p)==h for p,h in sources.items())
for path in (REPO/"scripts/diagnose_prior_tail_conditionals_v1.py",REPO/"src/posteriorforge/execution_mixture.py",hybrid_path,REPO/"src/posteriorforge/compat_cdf.py"):
 relative=str(path.relative_to(REPO))
 archived=subprocess.run(["git","-C",str(REPO),"show",recorded_commit+":"+relative],check=True,capture_output=True).stdout
 assert hashlib.sha256(archived).hexdigest()==sources[path]
records=[]
for row in d["stages"]:
 stage=row["fixture"]["stage"];focal=row["fixture"]["focal_offset"]
 wanted_hex=row["fixture"]["reconstructed_uniform"]["float32_bytes_hex"]
 for arm,entries in row["arms"].items():
  for e in entries:
   path=CAP/e["file"];assert sha_file(path)==e["sha256"]
   offset=0 if e["label"]=="N1" else focal
   with np.load(path,allow_pickle=False) as a:
    assert set(a.files)==set(e["arrays"])
    def get(k):
     value=a[k]
     assert value.dtype.str==e["arrays"][k]["dtype"] and list(value.shape)==e["arrays"][k]["shape"]
     return value
    raw=get("raw_model_logits")[offset].copy()
    translated=get("raw_translated_probabilities")[offset].copy()
    api=get("logits")[offset].copy()
    final=get("probabilities")[offset].copy()
    cdf=get("cdf")[offset].copy()
    mass=float(get("component_raw_mass")[offset])
    frm=get("from_borders").copy();to=get("to_borders").copy();borders=get("borders").copy()
    uniforms=get("uniforms").copy()
    uindex=next(i for i,v in enumerate(uniforms) if v.tobytes().hex()==wanted_hex)
    outputs=get("output")[:,offset].copy()
    bins=get("bin")[:,offset].copy()
    rest=get("rest")[:,offset].copy()
    selected=get("selected_probability")[:,offset].copy()
    before=get("cdf_before")[:,offset].copy()
    fraction=get("fraction_diagnostic_F32")[:,offset].copy()
   settings=e["selected_fitted_state"]
   recomputed_raw_mass=float(np.sum(translated,dtype=np.float64))
   records.append({
    "stage":stage,"arm":arm,"member":e["member"],"batch":e["label"],
    "file":e["file"],"file_sha256":e["sha256"],"actual_focal_row_offset":offset,
    "actual_raw_row_zero_based":row["fixture"]["row"],
    "input_descriptor":e["input"],"configuration_identity":e["configuration"],
    "target_transform":settings["actual_target_transform"],
    "target_transform_pickle":settings["target_transform_pickle"],
    "model_training":settings["selected_model_training"],
    "training_module_names":settings["training_module_names"],
    "target_mean":row[arm+"_model_after"]["target_mean"],
    "target_sd":row[arm+"_model_after"]["target_sd"],
    "temperature":row[arm+"_model_after"]["temperature"],
    "pretranslation_logits":{"focal_descriptor":descriptor(raw),"finite":int(np.isfinite(raw).sum()),"negative_infinity":int(np.isneginf(raw).sum()),"NaN":int(np.isnan(raw).sum()),"last_bin":scalar(raw[-1]),"scope":"Actual translator argument after temperature/cancellation, not untouched NN logits."},
    "pretranslation_softmax_probability_array_captured":False,
    "from_borders":{"descriptor":descriptor(frm),"first":scalar(frm[0]),"last":scalar(frm[-1]),"nonfinite":int((~np.isfinite(frm)).sum()),"nonincreasing_steps":int(np.count_nonzero(np.diff(frm)<=0))},
    "to_borders":{"descriptor":descriptor(to),"first":scalar(to[0]),"last":scalar(to[-1])},
    "from_to_exact_equal":bool(np.array_equal(frm,to)),
    "returned_criterion_borders":{"descriptor":descriptor(borders),"first":scalar(borders[0]),"last":scalar(borders[-1])},
    "actual_translated_raw_P_focal_descriptor":descriptor(translated),
    "actual_translated_raw_P_last_bin":scalar(translated[-1]),
    "actual_component_raw_mass_GPU_sum_Float64":mass,
    "CPU_recomputed_translated_raw_P_sum_Float64":recomputed_raw_mass,
    "CPU_minus_actual_GPU_mass":recomputed_raw_mass-mass,
    "actual_API_logits_focal_descriptor":descriptor(api),
    "actual_API_logits_last_bin":scalar(api[-1]),
    "actual_CPU_ICDF_softmax_P_focal_descriptor":descriptor(final),
    "actual_CPU_ICDF_softmax_P_last_bin":scalar(final[-1]),
    "CPU_recomputed_API_P_sum_Float64":float(np.sum(final,dtype=np.float64)),
    "actual_CPU_ICDF_Float32_terminal_CDF":scalar(cdf[-1]),
    "provided_uniform_values":[scalar(v) for v in uniforms],
    "historical_reconstructed_U_grid_index":uindex,
    "actual_new_ICDF_outputs":[scalar(v) for v in outputs],
    "actual_new_selected_bins":[int(v) for v in bins],
    "actual_new_CDF_before":[scalar(v) for v in before],
    "actual_new_rest":[scalar(v) for v in rest],
    "actual_new_selected_probability":[scalar(v) for v in selected],
    "recomputed_diagnostic_fraction_F32":[scalar(v) for v in fraction],
    "historical_U_scalar_case":{"U":scalar(uniforms[uindex]),"bin":int(bins[uindex]),"CDF_before":scalar(before[uindex]),"rest":scalar(rest[uindex]),"P_selected":scalar(selected[uindex]),"fraction_diagnostic_F32":scalar(fraction[uindex]),"output":scalar(outputs[uindex])},
    "scope":"Current counterfactual probe; no historical head/bin/ICDF identification."})
assert len(records)==48 and all(r["model_training"] is False and not r["training_module_names"] for r in records)
recommended=[r for r in records if (r["stage"],r["arm"],r["member"],r["batch"]) in [(19,"original",0,"N1024"),(19,"candidate",0,"N1024"),(15,"original",0,"N1024"),(19,"original",2,"N1024")]]
assert len(recommended)==4
report={
 "schema":"posteriorforge.tail_postprocessing_input_custody.v1",
 "status":"saved_focal_array_and_source_inspection_complete",
 "capture_manifest_sha256":EXPECTED,
 "code_sha256":sha_file(__file__),
 "source_hashes":{str(p):h for p,h in sources.items()},
 "recorded_capture_commit":recorded_commit,
 "first_party_callback_sources_match_recorded_capture_Git_bytes":True,
 "source_paths_lines":{
  "tabpfn/regressor.py":{"temperature":[656,657],"target_inverse_border_and_cancellation":[672,691],"translation_to_common_grid":[695,701],"API_stack_mean_GPU_log_CPU":[703,713]},
  "tabpfn/utils.py":{"pretranslation_softmax":[558,564],"CDF_bucket_interpolation_and_clamp":[567,576],"forced_CDF_endpoints_diff_nonnegative_clamp":[595,599]},
  "src/posteriorforge/execution_hybrid.py":{"translation_callback_argument_and_actual_result_capture":[179,190]},
  "src/posteriorforge/execution_mixture.py":{"actual_CPU_softmax_cumsum_and_finite_inverse":[240,252]},
 },
 "actual_pretranslation_GPU_F32_softmax_result_captured":False,
 "raw_model_logits_semantics":"Actual input to GPU translator after temperature0.9 division and cancellation. Untempered raw neural output and actual GPU pretranslation softmax are absent.",
 "actual_raw_translated_P_semantics":"Actual GPU corrected-CDF translated output, copied without renormalization before API GPU stack/mean/log.",
 "actual_component_mass_semantics":"Actual GPU sum(translated raw P,dtype=Float64), copied to CPU; CPU Float64 reduction checks are recomputations, not that captured operation.",
 "actual_probabilities_semantics":"Actual CPU Float32 softmax of final API logits used in observed finite_bar_icdf. They are not pretranslation softmax probabilities.",
 "boundary_pitfalls":[
 "All8 canonical output bins share the returned target-mean/SD-renormalized criterion; from/to border arrays are in standardized latent target space and are not decoded sigma bounds.",
 "SafePower inverse-border arrays and fitted state are captured. Do not apply the target inverse a second time to final API criterion outputs.",
 "Corrected CDF uses Float32 cumulative-minus-current probability, clamps interpolated CDF, forces first/last target CDF0/1, differences, and clamps negative bin masses. It does not simply return its softmax input even when from==to.",
 "A CPU softmax recomputation from saved pretranslation logits is not bit-evidence of the original GPU softmax; actual GPU pretranslation probabilities were not saved.",
 "API GPU log then CPU softmax can renormalize/round translated masses again; raw translated P, API P, F64 raw mass and Float32 terminal CDF must remain distinct.",
 "The saved fraction rest/p is a posthoc diagnostic temporary; frozen ICDF evaluates left+(right-left)*rest/p, so operation-order rounding may differ from left+width*fraction.",
 "Saved -inf cancellation is an intended API input property; it must not be replaced with epsilon. No clipping, renormalization or repair is activated by this inspection."],
 "record_count":len(records),"all48_focal_records":records,
 "recommended_scalar_cases":recommended,
 "recommended_order":"Stage19 member0 original then candidate: target_transform=None and exact frm==to avoid fitted-transform ambiguity. Stage15 member0 original contrasts the second stage. Stage19 original member2 includes actual SafePower fitted/border state for a transform case.",
 "no_model_constructor_forward_preprocessing_transform_RNG_or_softmax_calls":True,
 "no_G3_posterior_decode_cohort_repair_source_change":True,
 "Torch_imported":("torch" in sys.modules),
 "runtime":{"python":platform.python_version(),"numpy":np.__version__},
 "attribution":"Built with PriorLabs-TabPFN"}
assert "torch" not in sys.modules and sha_file(CAP/"manifest.json")==EXPECTED
assert all(sha_file(p)==h for p,h in sources.items())
OUT.mkdir(parents=True,exist_ok=True)
(OUT/"receipt.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
print(json.dumps({"receipt_sha256":sha_file(OUT/"receipt.json"),"records":len(records),"actual_pretranslation_GPU_softmax_captured":False,"Torch_imported":False,"recommended_cases":[{"file":r["file"],"target_transform_type":r["target_transform"].get("type") if isinstance(r["target_transform"],dict) else None,"from_to_exact_equal":r["from_to_exact_equal"],"actual_mass":r["actual_component_raw_mass_GPU_sum_Float64"],"API_P_F64_sum":r["CPU_recomputed_API_P_sum_Float64"],"terminal_CDF":r["actual_CPU_ICDF_Float32_terminal_CDF"],"historical_U_scalar_case":r["historical_U_scalar_case"]} for r in recommended]},sort_keys=True))
