"""Original prior endpoint contract on a saved Float64 bank row; no draws."""
from pathlib import Path
import hashlib, inspect, json, pickle, platform, random, subprocess, sys
import numpy as np
import scipy
import scipy.stats
import torch
from posteriorforge.banks import load_context
from posteriorforge.inference import _prior, support_mask
from posteriorforge.science.analytical_benchmark_v2 import build_problem, array_descriptor

ROOT=Path("/scratch/project/eu-26-17/eubrieucb/posteriorforge")
REPO=Path("/home/it4i-bbenvegnen/workspace/PosteriorForge")
OUT=ROOT/"results/accuracy-prior-coordinates-v1/original-prior-endpoint-contract-01"
sha=lambda b:hashlib.sha256(b).hexdigest()
file_sha=lambda p:sha(Path(p).read_bytes())
def states():
 return {"numpy":sha(pickle.dumps(np.random.get_state(),protocol=5)),
         "python":sha(pickle.dumps(random.getstate(),protocol=5)),
         "torchCPU":sha(torch.get_rng_state().numpy().tobytes())}
def number(x):
 return float(x) if np.isfinite(x) else str(float(x))
def bits(x):
 a=np.asarray(x,dtype=np.float64)
 return {"dtype":a.dtype.str,"value":float(a),"hex_float":float(a).hex(),"little_endian_bytes_hex":a.tobytes().hex()}
before=states();cuda_before=torch.cuda.is_initialized();assert not cuda_before
problem=build_problem();prior=_prior(torch)
context=load_context(ROOT/"results/banks-v1",20261111,4096)
assert context.theta.dtype==np.float64 and context.theta.shape==(4096,21)
failure=ROOT/"results/accuracy-prior-coordinates-v1/native-pilot-v2-seed20261111-case0-01/manifest.json"
assert file_sha(failure)=="92755b788fc894d4ca16157e573bd1b5c9d8a4468b71cf184fa16303bced7bce"
manifest=json.loads(failure.read_bytes())
assert array_descriptor(context.theta)==manifest["raw_scientific_bindings"]["context_theta"]
baseline=np.ascontiguousarray(context.theta[0:1].copy())
baseline_bytes=baseline.tobytes()
tensor=torch.from_numpy(baseline.copy())
assert tensor.dtype==torch.float64 and tensor.device.type=="cpu"
assert bool(prior.support.check(tensor)[0]) and bool(problem.contains(baseline)[0])
assert np.isfinite(problem.prior.log_prob(baseline)[0]) and bool(torch.isfinite(prior.log_prob(tensor))[0])
closure=inspect.getclosurevars(prior.support.check).nonlocals
low,high=problem.prior.bounds
assert closure["low"]==low and closure["high"]==high
values=[
 ("lower_adjacent_outside",np.nextafter(np.float64(low),-np.inf)),
 ("lower_exact",np.float64(low)),
 ("lower_adjacent_inside",np.nextafter(np.float64(low),np.inf)),
 ("upper_adjacent_inside",np.nextafter(np.float64(high),-np.inf)),
 ("upper_exact",np.float64(high)),
 ("upper_adjacent_outside",np.nextafter(np.float64(high),np.inf)),
]
rows=[]
for noise_index,name in enumerate(problem.prior.noise_names):
 slot=problem.parameter_order.index(name)
 coordinate=problem.coordinates[slot]
 for label,value in values:
  row=baseline.copy();row[0,slot]=value
  unchanged=[i for i in range(21) if i!=slot]
  assert row[:,unchanged].tobytes()==baseline[:,unchanged].tobytes()
  t=torch.from_numpy(row)
  check=bool(prior.support.check(t)[0])
  inference_lp=float(prior.log_prob(t)[0])
  raw_prior_lp=float(problem.prior.log_prob(row)[0])
  problem_lp=float(problem.log_prior(row)[0])
  contains=bool(problem.contains(row)[0])
  isolated=float(problem.prior._noise_log_prob(np.asarray([value],dtype=np.float64),noise_index)[0])
  rows.append({
   "name":name,"physical_slot":slot,"noise_prior_index":noise_index,"case":label,"value":bits(value),
   "Torch_input_dtype":str(t.dtype),"Torch_output_log_prob_dtype":str(prior.log_prob(t).dtype),
   "original_inference_prior_support_check":check,
   "original_inference_prior_log_prob":number(inference_lp),
   "inference_support_vs_finite_log_prob_mismatch":check!=bool(np.isfinite(inference_lp)),
   "scientific_coordinate_contains":bool(coordinate.contains(np.asarray([value],dtype=np.float64))[0]),
   "scientific_problem_contains":contains,"scientific_inference_support_mask":bool(support_mask(row)[0]),
   "scientific_problem_log_prior":number(problem_lp),
   "scientific_raw_prior_log_prob":number(raw_prior_lp),
   "scientific_raw_noise_log_prob":number(isolated),
   "raw_prior_density_finite_vs_coordinate_support_mismatch":contains!=bool(np.isfinite(raw_prior_lp)),
   "scientific_raw_noise_log_prob_dtype":str(problem.prior._noise_log_prob(np.asarray([value],dtype=np.float64),noise_index).dtype),
   "other20_columns_byte_identical":True})
paths=[
 "src/posteriorforge/inference.py","src/posteriorforge/science/analytical_benchmark_v2.py",
 "src/posteriorforge/science/hierarchical_noise.py","src/posteriorforge/science/hierarchical_problem.py",
 "src/posteriorforge/science/hierarchical_exact.py"]
recorded="ff1921e88a3c3436208861ae506bef44871906c9"
sources={}
for name in paths:
 path=REPO/name
 archived=subprocess.run(["git","-C",str(REPO),"show",recorded+":"+name],check=True,capture_output=True).stdout
 assert file_sha(path)==sha(archived)
 sources[str(path)]=file_sha(path)
for path in [Path(scipy.stats._distn_infrastructure.__file__),Path(scipy.stats._continuous_distns.__file__)]:
 sources[str(path)]=file_sha(path)
assert baseline.tobytes()==baseline_bytes and array_descriptor(context.theta)==manifest["raw_scientific_bindings"]["context_theta"]
after=states();assert before==after and not torch.cuda.is_initialized()
assert len(rows)==36
report={
 "schema":"posteriorforge.original_prior_endpoint_contract.v1",
 "status":"original_prior_endpoint_evaluation_complete",
 "code_sha256":file_sha(__file__),
 "source_hashes":sources,"first_party_source_bytes_match_recorded_Git":recorded,
 "source_line_evidence":{"inference.py":{"custom_support_and_counters":[49,63],"log_prob_delegation":[64,72]},
 "hierarchical_problem.py":{"coordinate_contains_inclusive":[96,101],"problem_contains":[396,403],"log_prior_contains_mask":[476,481]},
 "hierarchical_noise.py":{"noise_logpdf":[132,142],"full_prior_log_prob":[149,155]},
 "analytical_benchmark_v2.py":{"frozen_exact_priors":[50,65]}},
 "source_bank":{"context_seed":20261111,"context_size":4096,"row_zero_based":0,"full_theta_descriptor":array_descriptor(context.theta),"saved_row_descriptor":array_descriptor(baseline),"baseline_original_support_check":True,"baseline_log_prob_finite":True},
 "physical_parameter_order":list(problem.parameter_order),
 "actual_bound_source":{"bounds_dtype":"Python float (binary64)","low":bits(low),"high":bits(high),"support_closure_low_type":type(closure["low"]).__name__,"support_closure_high_type":type(closure["high"]).__name__},
 "noise_underlying_gaussian_priors":problem.prior.noise_gaussian_priors,
 "original_custom_support_class":type(prior.support).__qualname__,"original_distribution_class":type(prior).__qualname__,
 "Torch_distribution_validate_args":prior._validate_args,
 "evaluation_dtype_scope":"Saved Float64 physical bank row; Torch CPU float64 inputs and output log_prob; no Float32 casting. Each trial changes exactly one original noise coordinate.",
 "evaluation_rows":rows,
 "inference_support_vs_finite_log_prob_mismatches":[r for r in rows if r["inference_support_vs_finite_log_prob_mismatch"]],
 "scientific_raw_prior_density_vs_physical_contains_mismatches":[r for r in rows if r["raw_prior_density_finite_vs_coordinate_support_mismatch"]],
 "original_support_call_counters":{"calls":prior.support.calls,"proposed":prior.support.proposed,"accepted":prior.support.accepted},
 "default_RNG_scope":"After imports through original prior construction, saved-bank read and deterministic density/support evaluations; no random draw APIs.",
 "default_RNG_before":before,"default_RNG_after":after,"default_RNG_preserved":before==after,
 "CUDA_initialized_before":cuda_before,"CUDA_initialized_after":torch.cuda.is_initialized(),
 "prior_distribution_constructor_calls":1,"neural_model_constructor_or_forward_calls":0,
 "random_world_posterior_G3_reference_draw_or_integration_calls":0,
 "density_and_support_evaluations_only":True,"coordinate_or_source_policy_changes":False,
 "versions":{"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,"torch":torch.__version__},
 "attribution":"Built with PriorLabs-TabPFN"}
OUT.mkdir(parents=True,exist_ok=True)
(OUT/"receipt.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
print(json.dumps({"receipt_sha256":file_sha(OUT/"receipt.json"),"rows":len(rows),"inference_mismatches":len(report["inference_support_vs_finite_log_prob_mismatches"]),"raw_scientific_prior_mismatches":len(report["scientific_raw_prior_density_vs_physical_contains_mismatches"]),"first_noise_cases":[{"case":r["case"],"value":r["value"],"support":r["original_inference_prior_support_check"],"inference_log_prob":r["original_inference_prior_log_prob"],"raw_scientific_prior_log_prob":r["scientific_raw_prior_log_prob"],"raw_noise_log_prob":r["scientific_raw_noise_log_prob"]} for r in rows[:6]],"default_RNG_preserved":True,"CUDA_initialized":False},sort_keys=True))
