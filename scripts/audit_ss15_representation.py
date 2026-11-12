#!/usr/bin/env python3
"""Saved-data CPU audit of exactSS15 sufficiency; no model/reference integration.

Built with PriorLabs-TabPFN. All inputs are existing frozen banks/heldout cases.
"""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np
from posteriorforge import representation_ss15 as ss
from posteriorforge.artifacts import runtime_identity,sha256_file,write_json
from posteriorforge.banks import load_context,load_heldout
from posteriorforge.science.analytical_benchmark_v2 import build_problem,unflatten_observations,array_descriptor

def run(*,banks,output):
 started=time.perf_counter();output=Path(output);output.mkdir(parents=True,exist_ok=False)
 p=build_problem();source=ss.source_sha256()
 report={**runtime_identity(),"schema":"posteriorforge.ss15.saved_data_exactness.v1",
   "attribution":"Built with PriorLabs-TabPFN","diagnostic_only":True,"model_calls":0,"new_worlds":0,
   "reference_integrations":0,"original_gate_changed":False,"representation":ss.representation_metadata(),
   "producer_sha256":sha256_file(__file__),"banks":[],"heldout_checks":[],
   "numeric_scope":"exact real-arithmetic sufficient statistics; float64 identity and final float32 feature rounding separately measured"}
 max64=max32=max_raw32=0.
 for seed in (20261111,20261112,20261113):
  bank=load_context(banks,seed,4096);stats=ss.transform_observations(bank.obs);model=ss.transform_for_model(bank.obs)
  rows=np.arange(len(bank.obs))
  chunks=np.concatenate([ss.transform_observations(bank.obs[i:i+73]) for i in range(0,len(bank.obs),73)])
  if chunks.tobytes()!=stats.tobytes():raise RuntimeError("SS15 feature batch invariance failed")
  raw=p.log_likelihood(bank.theta,unflatten_observations(bank.obs));reduced=ss.raw_likelihood_factor(bank.theta,stats)
  error=float(np.max(np.abs(raw-reduced)));max64=max(max64,error)
  if not np.allclose(raw,reduced,rtol=8e-14,atol=2e-9):raise RuntimeError("paired bank likelihood identity failed")
  report["banks"].append({"seed":seed,"rows":len(rows),"source_manifest":bank.manifest,
    "raw_obs":array_descriptor(bank.obs),"theta":array_descriptor(bank.theta),
    "ss15_float64":array_descriptor(stats),"ss15_float32":array_descriptor(model),
    "representation_cache_key":ss.representation_cache_key(bank.obs),"batch73_bitwise_equal":True,
    "paired_raw_vs_ss15_float64_likelihood_max_abs":error,
    "mechanical_residual_energy_min":stats[:,[3,7,11]].min(0).tolist(),
    "acoustic_features_exactly_retained":bool(np.array_equal(stats[:,-3:],bank.obs[:,-3:]))})
  for case in range(8):
   heldout=load_heldout(banks,case);obs=heldout.obs.reshape(1,33);t=ss.transform_observations(obs)
   raw=p.log_likelihood(bank.theta,unflatten_observations(obs))
   reduced=ss.raw_likelihood_factor(bank.theta,t)
   rounded=ss.raw_likelihood_factor(bank.theta,ss.transform_for_model(obs))
   raw32=p.log_likelihood(bank.theta,unflatten_observations(obs.astype(np.float32).astype(np.float64)))
   e64=float(np.max(np.abs(raw-reduced)));e32=float(np.max(np.abs(raw-rounded)));eraw32=float(np.max(np.abs(raw-raw32)))
   max64=max(max64,e64);max32=max(max32,e32);max_raw32=max(max_raw32,eraw32)
   if not np.allclose(raw,reduced,rtol=8e-14,atol=2e-9):raise RuntimeError("heldout likelihood identity failed")
   report["heldout_checks"].append({"seed":seed,"case":case,"heldout_manifest":heldout.manifest,
      "raw_obs":array_descriptor(obs),"ss15_float64":array_descriptor(t),"ss15_float32":array_descriptor(ss.transform_for_model(obs)),
      "raw_vs_ss15_float64_likelihood_max_abs":e64,
      "raw_vs_final_float32_ss15_likelihood_max_abs":e32,
      "raw_vs_old_raw33_float32_likelihood_max_abs":eraw32,
      "normalized_raw_likelihood_range":[float(raw.min()),float(raw.max())],
      "minimum_mechanical_residual_energy":float(t[:,[3,7,11]].min())})
 report.update(status="saved_data_ss15_float64_sufficiency_identity_pass",
   max_float64_likelihood_identity_abs=max64,max_float32_ss15_likelihood_rounding_abs=max32,
   max_old_raw33_float32_likelihood_rounding_abs=max_raw32,all24_heldout_checks_passed=True,
   features_never_use_target_truth=True,sources_unchanged=source==ss.source_sha256(),
   elapsed_seconds=time.perf_counter()-started)
 if not report["sources_unchanged"]:raise RuntimeError("representation source changed during CPU audit")
 write_json(output/"manifest.json",report)
 return report

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument("--banks",required=True);parser.add_argument("--output",required=True)
 result=run(**vars(parser.parse_args()))
 print(json.dumps({k:result[k] for k in ("status","max_float64_likelihood_identity_abs",
 "max_float32_ss15_likelihood_rounding_abs","max_old_raw33_float32_likelihood_rounding_abs","elapsed_seconds")},indent=2))
if __name__=="__main__":main()
