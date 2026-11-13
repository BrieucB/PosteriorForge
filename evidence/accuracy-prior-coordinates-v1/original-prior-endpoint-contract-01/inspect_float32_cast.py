"""Original prior Float32 scalar-cast seam; saved row only, no draws or policy change."""
from pathlib import Path
import hashlib,json,pickle,platform,random,subprocess
import numpy as np, scipy, torch
from posteriorforge.banks import load_context
from posteriorforge.inference import _prior,support_mask
from posteriorforge.science.analytical_benchmark_v2 import build_problem,array_descriptor
ROOT=Path("/scratch/project/eu-26-17/eubrieucb/posteriorforge")
REPO=Path("/home/it4i-bbenvegnen/workspace/PosteriorForge")
OUT=ROOT/"results/accuracy-prior-coordinates-v1/original-prior-endpoint-contract-01"
sha=lambda b:hashlib.sha256(b).hexdigest()
file_sha=lambda p:sha(Path(p).read_bytes())
def states():
 return {"numpy":sha(pickle.dumps(np.random.get_state(),protocol=5)),"python":sha(pickle.dumps(random.getstate(),protocol=5)),"torchCPU":sha(torch.get_rng_state().numpy().tobytes())}
def num(v):return float(v) if np.isfinite(v) else str(float(v))
def desc_scalar(v,dtype):
 a=np.asarray(v,dtype=dtype)
 return {"dtype":a.dtype.str,"value":float(a),"hex_float":float(a).hex(),"little_endian_bytes_hex":a.tobytes().hex()}
assert file_sha(OUT/"receipt.json")=="5107b08b474b76f5438cb49e21ac84c3e33be6df5c82c5c7ff5732d197bcfbf5"
before=states();assert not torch.cuda.is_initialized()
problem=build_problem();prior=_prior(torch)
context=load_context(ROOT/"results/banks-v1",20261111,4096)
assert context.theta.dtype==np.float64
baseline=context.theta[0:1].astype(np.float32)
bt=torch.from_numpy(baseline.copy())
assert bool(prior.support.check(bt)[0]) and bool(torch.isfinite(prior.log_prob(bt))[0]) and bool(problem.contains(baseline)[0])
low,high=problem.prior.bounds
low32,high32=np.float32(low),np.float32(high)
cases64=[
 ("lower_adjacent_outside",np.nextafter(np.float64(low),-np.inf)),
 ("lower_exact",np.float64(low)),
 ("lower_adjacent_inside",np.nextafter(np.float64(low),np.inf)),
 ("upper_adjacent_inside",np.nextafter(np.float64(high),-np.inf)),
 ("upper_exact",np.float64(high)),
 ("upper_adjacent_outside",np.nextafter(np.float64(high),np.inf))]
cases32=[
 ("lower_adjacent_outside",np.nextafter(low32,np.float32(-np.inf),dtype=np.float32)),
 ("lower_exact",low32),
 ("lower_adjacent_inside",np.nextafter(low32,np.float32(np.inf),dtype=np.float32)),
 ("upper_adjacent_inside",np.nextafter(high32,np.float32(-np.inf),dtype=np.float32)),
 ("upper_exact",high32),
 ("upper_adjacent_outside",np.nextafter(high32,np.float32(np.inf),dtype=np.float32))]
rows=[]
for name in problem.prior.noise_names:
 slot=problem.parameter_order.index(name)
 for family,cases in [("requested_Float64_endpoint_neighbours_then_whole_row_Float32_cast",cases64),("native_Float32_bound_and_neighbours",cases32)]:
  for label,requested in cases:
   # Actual runtime F32 path: one final whole-row cast from saved source row.
   if family.startswith("requested"):
    original=context.theta[0:1].copy();original[0,slot]=requested;row=original.astype(np.float32)
   else:
    row=baseline.copy();row[0,slot]=requested
   rest=[i for i in range(21) if i!=slot]
   assert row[:,rest].tobytes()==baseline[:,rest].tobytes()
   t=torch.from_numpy(row)
   check=bool(prior.support.check(t)[0]);lp=prior.log_prob(t)
   gate=bool(support_mask(row)[0]);contains=bool(problem.contains(row)[0])
   rows.append({"noise_name":name,"physical_slot":slot,"family":family,"case":label,
    "requested_scalar":desc_scalar(requested,requested.dtype),
    "actual_Torch_Float32_scalar":desc_scalar(row[0,slot],np.float32),
    "actual_Float32_scalar_widened_to_Float64":desc_scalar(np.float64(row[0,slot]),np.float64),
    "Torch_input_dtype":str(t.dtype),"Torch_log_prob_output_dtype":str(lp.dtype),
    "original_support_check":check,"original_log_prob":num(float(lp[0])),
    "scientific_problem_contains":contains,"actual_gate_support_mask":gate,
    "support_vs_finite_log_prob_mismatch":check!=bool(torch.isfinite(lp)[0]),
    "other20_columns_match_Float32_cast_saved_row":True})
assert len(rows)==72
after=states();assert after==before and not torch.cuda.is_initialized()
paths=["src/posteriorforge/inference.py","src/posteriorforge/science/analytical_benchmark_v2.py","src/posteriorforge/science/hierarchical_noise.py","src/posteriorforge/science/hierarchical_problem.py","src/posteriorforge/science/hierarchical_exact.py"]
sources={}
for name in paths:
 raw=subprocess.run(["git","-C",str(REPO),"show","ff1921e88a3c3436208861ae506bef44871906c9:"+name],check=True,capture_output=True).stdout
 assert file_sha(REPO/name)==sha(raw)
 sources[name]=sha(raw)
report={"schema":"posteriorforge.original_prior_Float32_cast_contract.v1","status":"Float32_scalar_cast_seam_observed",
 "code_sha256":file_sha(__file__),"Float64_receipt_sha256":file_sha(OUT/"receipt.json"),"source_hashes":sources,
 "source_bank":{"seed":20261111,"size":4096,"saved_row_zero_based":0,"original_Float64_row_descriptor":array_descriptor(context.theta[0:1]),"Float32_cast_row_descriptor":array_descriptor(baseline),"baseline_support_and_log_prob_valid":True},
 "original_Python_Float64_bounds":{"low":desc_scalar(low,np.float64),"high":desc_scalar(high,np.float64)},
 "explicit_Float32_bound_casts":{"low":desc_scalar(low32,np.float32),"high":desc_scalar(high32,np.float32)},
 "Float32_lower_bound_minus_original_Float64_bound":float(np.float64(low32)-np.float64(low)),
 "row_count":len(rows),"evaluation_rows":rows,
 "support_vs_finite_log_prob_mismatches":[r for r in rows if r["support_vs_finite_log_prob_mismatch"]],
 "interpretation":"Original custom Support.check compares Python scalar bounds in the input tensor's Float32 arithmetic. Its log_prob delegates through NumPy to problem._parameters Float64 and closed physical contains. The stored Float32 representation of0.01 is below the original Float64 physical lower bound; it can pass Support.check while log_prob/gate reject. This is a dtype seam, not an authorization to alter support.",
 "actual_original_prior_constructor_calls":1,"neural_constructor_forward_or_random_draw_calls":0,
 "default_RNG_before":before,"default_RNG_after":after,"default_RNG_preserved":True,
 "CUDA_initialized":False,"no_source_policy_change":True,
 "versions":{"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,"torch":torch.__version__}}
(OUT/"float32-cast-receipt.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
example=[r for r in rows if r["noise_name"]==problem.prior.noise_names[0]]
print(json.dumps({"receipt_sha256":file_sha(OUT/"float32-cast-receipt.json"),"rows":len(rows),"mismatches":len(report["support_vs_finite_log_prob_mismatches"]),"lower_Float32":desc_scalar(low32,np.float32),"lower_widened_Float64":desc_scalar(np.float64(low32),np.float64),"lower_delta":report["Float32_lower_bound_minus_original_Float64_bound"],"first_noise_cases":[{"family":r["family"],"case":r["case"],"actual":r["actual_Torch_Float32_scalar"]["value"],"support":r["original_support_check"],"log_prob":r["original_log_prob"],"gate":r["actual_gate_support_mask"]} for r in example],"RNG_preserved":True,"CUDA_initialized":False},sort_keys=True))
