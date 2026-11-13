#!/usr/bin/env python3
"""Saved-focal A/B/C postprocessing arithmetic diagnosis only.

Built with PriorLabs-TabPFN. No GPU/NN/model constructor/world/posterior/G3/RNG/
live sampler call. Existing adjusted F32 logits are already temperature/cancel
processed. No temperature division, refit, border reconstruction or policy edit.
"""
from __future__ import annotations
import argparse,hashlib,json,math,os,pickle,random,time
from bisect import bisect_left,bisect_right
from fractions import Fraction
from pathlib import Path
import numpy as np
from posteriorforge import prior_coordinates_v1 as codec
SCHEMA="posteriorforge.saved_tail_postprocessing_diagnosis.v1"
ANALYSIS_SHA="98eccfb5b6db580f0fa3a2cd64e5765e8312c7ca960cf4c1debe68c37fee2f91"
MATH_SHA="d5a22a8a617023ced6d1b043ecca6185bf384a5dcd33bb2ff862050f430028e5"
CODEC_SHA="d8ce6c49e7b65c6ee55c59b7243d4ce943cbea28c92ebed0329798074aaa70c5"
U_BITS=(0x3f7fffe0,0x3f7ffff6)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(x,m):
 if not x:raise ValueError(m)
def desc(a):
 a=np.asarray(a);return {"shape":list(a.shape),"dtype":a.dtype.str,"sha256":hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()}
def rat(x):return {"numerator":str(x.numerator),"denominator":str(x.denominator)}
def finite(x):return float(x) if np.isfinite(x) else str(float(x))
def units64(a):
 values=[]
 for word in np.ascontiguousarray(a,np.float64).view(np.uint64):
  b=int(word);e=(b>>52)&2047;m=b&((1<<52)-1)
  values.append(m if e==0 else ((1<<52)|m)<<(e-1))
 return values
def exact_summary(weights,borders,uniforms,threshold):
 require(np.isfinite(weights).all() and (weights>=0).all(),"invalid arithmetic-reference weights")
 ints=units64(weights);total=sum(ints);require(total>0,"zero arithmetic-reference mass")
 prefix=[];running=0
 for x in ints:running+=x;prefix.append(running)
 bs=[Fraction.from_float(float(x)) for x in borders];require(all(a<b for a,b in zip(bs,bs[1:])),"finite inverse borders not strictly monotone")
 tau=Fraction(threshold)
 if tau<=bs[0]:tail=Fraction(1)
 elif tau>=bs[-1]:tail=Fraction(0)
 else:
  j=bisect_right(bs,tau)-1
  below=(Fraction(prefix[j]-ints[j])+ints[j]*(tau-bs[j])/(bs[j+1]-bs[j]))/total
  tail=1-below
 points=[]
 for x in uniforms:
  u=Fraction.from_float(float(x));require(0<u<1,"only provided in-contract nonendpointU used")
  j=bisect_left([v*u.denominator for v in prefix],u.numerator*total)
  require(j<len(ints) and ints[j]>0,"exact normalized inverse selected zero bin")
  f=Fraction(u.numerator*total-(prefix[j]-ints[j])*u.denominator,ints[j]*u.denominator)
  value=bs[j]+f*(bs[j+1]-bs[j]);require(0<=f<=1,"exact inverse outside bar")
  points.append({"U":float(x),"U_bits":x.tobytes().hex(),"bin":j,"fraction_exact":rat(f),"value":float(value),"value_exact":rat(value)})
 return {"exact_weight_sum":rat(Fraction(total,1<<1074)),"weight_sum_F64":float(Fraction(total,1<<1074)),
  "last_bin_mass_normalized":float(Fraction(ints[-1],total)),"tail_mass_above_exactmath_endpoint_threshold":float(tail),
  "tail_mass_exact":rat(tail),"points":points}

def softmax64(logits):
 x=np.asarray(logits,np.float64);require(not np.isnan(x).any() and not np.isposinf(x).any(),"invalid captured adjustedlogits")
 maximum=np.max(x);require(np.isfinite(maximum),"all logits cancelled")
 e=np.exp(x-maximum);return e/np.sum(e,dtype=np.float64)
def translation64(p,frm,to,torch):
 # Exact corrected source OPERATION ORDER, but NEW CPU TorchFloat64 arithmetic.
 f=torch.from_numpy(frm.astype(np.float64));y=torch.from_numpy(to.astype(np.float64))
 pr=torch.from_numpy(p.astype(np.float64));width=f[1:]-f[:-1]
 bucket=torch.searchsorted(f,y)-1
 bucket[y==f[0]]=0;bucket[y==f[-1]]=len(f)-2
 ix=bucket.clamp(0,len(pr)-1)
 before=torch.cumsum(pr,dim=-1)-pr
 share=(y-f[ix])/width[ix];share_clipped=share.clamp(0.,1.)
 raw=before[ix]+pr[ix]*share_clipped
 bounded=raw.clone();bounded[y<=f[0]]=0.;bounded[y>=f[-1]]=1.
 clipped=bounded.clip(0.,1.);forced=clipped.clone();forced[0]=0.;forced[-1]=1.
 diff=forced[1:]-forced[:-1];result=diff.clamp_min(0.).numpy().copy()
 widths=width.numpy();raw_cdf=raw.numpy();bounded=bounded.numpy();clipped=clipped.numpy()
 share=share.numpy();diff=diff.numpy()
 require(np.isfinite(result).all(),"nonfinite F64 translation; no replacement")
 stats={"arithmetic":"NEW CPU TorchF64 corrected source order; not captured GPU execution",
  "before_CDF_order":"cumsum(p)-p, not prepended prefix","from_negative_widths":int(np.count_nonzero(widths<0)),
  "from_zero_widths":int(np.count_nonzero(widths==0)),"to_negative_steps":int(np.count_nonzero(np.diff(to)<0)),
  "from_nonfinite":int(np.count_nonzero(~np.isfinite(frm))),"to_nonfinite":int(np.count_nonzero(~np.isfinite(to))),
  "share_below0":int(np.count_nonzero(share<0)),"share_above1":int(np.count_nonzero(share>1)),
  "CDF_before_boundary_monotonicity_violations":int(np.count_nonzero(np.diff(raw_cdf)<0)),
  "CDF_clip_below0_count":int(np.count_nonzero(bounded<0)),"CDF_clip_above1_count":int(np.count_nonzero(bounded>1)),
  "CDF_clip_absolute_change":float(np.sum(np.abs(clipped-bounded),dtype=np.float64)),
  "negative_diff_count_before_source_clamp":int(np.count_nonzero(diff<0)),
  "negative_diff_absolute_mass_before_source_clamp":float(-np.sum(diff[diff<0],dtype=np.float64)),
  "translated_mass_after_source_clamp_F64sum":float(np.sum(result,dtype=np.float64)),
  "translated_last_bin":float(result[-1]),"lastCDF_before_forced_endpoint":float(clipped[-1]),
  "original_source_boundary_overrides_and_clamps_reproduced":True}
 return result,stats

def noise(z,spec):
 result={"input":float(z),"scope":"counterfactual scalar noise only; not a physical21-vector/cohort"}
 try:physical=codec._noise_decode(np.array([z],np.float64),spec)
 except codec.CoordinateError as e:return {**result,"decode_accepted":False,"error":str(e),"RT32_pass":False,"RT64_pass":False}
 result.update(decode_accepted=True,physical=float(physical[0]))
 try:
  back=float(codec._noise_encode(physical,spec)[0]);err=abs(back-z)
  result.update(reencode=back,RT_error=err,
   RT32_bound=float(codec.F32_LATENT_ADMISSIBILITY*(1+abs(z))),RT64_bound=float(codec.F64_LATENT_ADMISSIBILITY*(1+abs(z))),
   RT32_pass=bool(err<=codec.F32_LATENT_ADMISSIBILITY*(1+abs(z))),RT64_pass=bool(err<=codec.F64_LATENT_ADMISSIBILITY*(1+abs(z))))
 except codec.CoordinateError as e:result.update(reencode_error=str(e),RT32_pass=False,RT64_pass=False)
 return result

def post64(raw,torch):
 # average_before_softmax=False; selected one-head mean is identity.
 # NEW F64 arithmetic for log->CPU softmax; never claimed actual GPU values.
 values=torch.from_numpy(raw.astype(np.float64));logits=values.log()
 return logits.softmax(-1).numpy().copy(),logits.numpy().copy()
def run(analysis,output):
 started=time.perf_counter();require(os.environ.get("CUDA_VISIBLE_DEVICES")=="","CPU-only process requires hiddenCUDA")
 analysis=Path(analysis);require(sha(analysis)==ANALYSIS_SHA,"frozen48-file analysis differs")
 m=json.loads(analysis.read_bytes());out=Path(output);require(not out.exists(),"fresh diagnosis folder required");out.mkdir(parents=True);(out/"derived-weights").mkdir()
 mathfile=Path(__file__).resolve().parents[1]/"evidence/accuracy-prior-coordinates-v1/noise-inverse-math-diagnostic-01/reproducible-scalars.json"
 require(sha(mathfile)==MATH_SHA and codec.source_sha256()==CODEC_SHA,"frozen math/codec source differs")
 mr=json.loads(mathfile.read_bytes());spec=mr["source_constants_float"];threshold=mr["exact_prior_z_above_which_nearest_Float64_is_endpoint"]
 import torch
 torch.set_num_threads(1)
 rng_before=hashlib.sha256(pickle.dumps((np.random.get_state(),random.getstate()),protocol=4)+torch.get_rng_state().numpy().tobytes()).hexdigest()
 us=np.asarray(U_BITS,np.uint32).view(np.float32);records=[]
 for entry in m["entries"]:
  stem=Path(entry["source_file"]).stem+"-focal.npz";p=analysis.parent/"focal-densities"/stem
  witness=next(x for x in m["focal_subsets"] if Path(x["file"]).name==stem)
  require(sha(p)==witness["sha256"],"focal NPZ changed")
  with np.load(p,allow_pickle=False) as a:arr={n:a[n] for n in a.files}
  require(all(arr[k].dtype==np.float32 for k in ["raw_model_logits","from_borders","to_borders","raw_translated_probabilities","logits","probabilities","borders"]),"captured dtype changed")
  require(arr["raw_model_logits"].ndim==1 and arr["from_borders"].ndim==arr["to_borders"].ndim==1,"focal/source borders shape changed")
  require(all(desc(v)==witness["derived_arrays"][k]["derived"] for k,v in arr.items()),"focal array bytes changed")
  logits=arr["raw_model_logits"];frm=arr["from_borders"];to=arr["to_borders"];border=arr["borders"]
  pre32=torch.from_numpy(logits.copy()).softmax(-1).numpy().copy()
  pre64=torch.from_numpy(logits.copy()).to(torch.float64).softmax(-1).numpy().copy()
  rawB,statB=translation64(pre32,frm,to,torch);rawC,statC=translation64(pre64,frm,to,torch)
  pB64,logB64=post64(rawB,torch);pC64,logC64=post64(rawC,torch)
  # Source dtype-boundary comparison ONLY: NEW CPU replay of F32 output log
  # and F32 CPU softmax, not captured GPU-log arithmetic.
  def route32(raw):
   x=torch.from_numpy(raw.astype(np.float32));log=x.log();return log.softmax(-1).numpy().copy()
  pB32=route32(rawB);pC32=route32(rawC)
  returned_replay=torch.from_numpy(arr["logits"].copy()).softmax(-1).numpy().copy()
  captured_raw_log_cpu=torch.from_numpy(arr["raw_translated_probabilities"].copy()).log().numpy().copy()
  variants={"A_captured_final_CPU_F32_weights":arr["probabilities"],
   "B_CPU_F32_pre_softmax_replay_then_F64translation_direct":rawB,
   "B_CPU_F32_pre_softmax_replay_F64log_CPUsoftmax64":pB64,
   "B_F32_output_boundary_CPU_log_softmax_replay":pB32,
   "C_softmax64_adjusted_F32logits_F64translation_direct":rawC,
   "C_softmax64_adjusted_F32logits_F64log_CPUsoftmax64":pC64,
   "C_F32_output_boundary_CPU_log_softmax_replay":pC32}
  results={}
  for name,weights in variants.items():
   result=exact_summary(weights,border,us,threshold)
   for point in result["points"]:point["d8_noise"]=noise(point["value"],spec)
   results[name]=result
  # Actual A exact inverse is already independently recorded; this asserts
  # equivalence of the generic F64-dyadic implementation, not a new density.
  for point in results["A_captured_final_CPU_F32_weights"]["points"]:
   old=next(x for x in entry["provided_U_points"] if x["U_bytes_hex"]==point["U_bits"])
   require(float(point["value"]).hex()==float(old["reference_inverse_Float64"]).hex(),"A exact-reference mismatch")
  dtype_compare={}
  for prefix,rawname,postname in [("B","B_CPU_F32_pre_softmax_replay_then_F64translation_direct","B_CPU_F32_pre_softmax_replay_F64log_CPUsoftmax64"),
   ("C","C_softmax64_adjusted_F32logits_F64translation_direct","C_softmax64_adjusted_F32logits_F64log_CPUsoftmax64")]:
   delta=max(abs(a["value"]-b["value"]) for a,b in zip(results[rawname]["points"],results[postname]["points"]))
   tolerance=2e-6*float(border[-1]-border[0])
   dtype_compare[prefix]={"direct_normalized_vs_F64_log_CPUsoftmax_quantile_max_abs":delta,"diagnostic_span_tolerance":tolerance,"within_diagnostic_span_tolerance":delta<=tolerance,
    "scope":"postprocessing arithmetic comparison only; not new conditional qualification"}
  derived={"CPU_F32_pre_softmax_replay":pre32,"CPU_softmax64_adjusted_logits":pre64,"B_raw_translated_F64":rawB,"C_raw_translated_F64":rawC,
   "B_log_F64":logB64,"C_log_F64":logC64,"B_final_post64_weights":pB64,"C_final_post64_weights":pC64,"B_F32_boundary_replay":pB32,"C_F32_boundary_replay":pC32}
  target=out/"derived-weights"/(Path(stem).stem+"-references.npz")
  with target.open("xb") as stream:np.savez(stream,**derived)
  record={"stage":entry["stage"],"arm":entry["arm"],"member":entry["member"],"label":entry["label"],
   "source_focal_file":str(p),"source_focal_sha256":witness["sha256"],"source_array_descriptors":{k:desc(v) for k,v in arr.items()},
   "adjusted_logits_not_raw_unscaled_NN":True,"temperature_reapplied":False,
   "average_before_softmax":False,"averaging_flag_source_bound_not_captured_runtime_boolean":True,
   "one_head_route":"stack.mean0(identity)->GPUlog(dtype of translated output)->CPUcopy->CPUsoftmax; B/C NEW CPU TorchF64 route replay; F32-boundary alternatives separately labeled",
   "A_actual_GPU_pretranslation_softmax":"NOT CAPTURED","A_actual_GPU_preclip_negative_difference_counts":None,"A_actual_GPU_preclip_clipped_mass":None,
   "A_source_translation_clamp_min0_known":True,"A_actual_GPU_raw_mass_F64":float(arr["component_raw_mass"][0]),
   "A_captured_raw_translated_mass_CPU_F64sum":float(np.sum(arr["raw_translated_probabilities"],dtype=np.float64)),
   "A_actual_final_CPU_softmax_sum_CPU_F64sum":float(np.sum(arr["probabilities"],dtype=np.float64)),
   "A_actual_inverse_CDF_last_F32":float(arr["cdf"][-1]),
   "actual_final_CPU_softmax_focal_replay_byte_equal":returned_replay.tobytes()==arr["probabilities"].tobytes(),
   "captured_GPU_log_vs_NEW_CPU_F32_log_byte_equal":captured_raw_log_cpu.tobytes()==arr["logits"].tobytes(),
   "CPU_F32_pre_softmax_replay_sum_F64":float(np.sum(pre32,dtype=np.float64)),"CPU_softmax64_pretranslation_sum":float(np.sum(pre64,dtype=np.float64)),
   "from_to_borders_byte_identical":arr["from_borders"].tobytes()==arr["to_borders"].tobytes(),
   "B_new_F64_translation":statB,"C_new_F64_translation":statC,"variants":results,"postprocessing_route_compare":dtype_compare,
   "derived_reference_weights_file":str(target.relative_to(out)),"derived_reference_weights_sha256":sha(target),
   "derived_reference_weights_are_not_actual_GPU_capture":True,"counterfactual_density_and_grouping_limits_preserved":True}
  records.append(record)
  own_bits="e0ff7f3f" if entry["stage"]==15 else "f6ff7f3f"
  brief={key:next(x for x in results[name]["points"] if x["U_bits"]==own_bits) for key,name in [
   ("A","A_captured_final_CPU_F32_weights"),("B","B_CPU_F32_pre_softmax_replay_F64log_CPUsoftmax64"),("C","C_softmax64_adjusted_F32logits_F64log_CPUsoftmax64")]}
  print(json.dumps({"stage":entry["stage"],"arm":entry["arm"],"member":entry["member"],"label":entry["label"],
   "own_U_values":{k:v["value"] for k,v in brief.items()},"own_U_RT32_pass":{k:v["d8_noise"]["RT32_pass"] for k,v in brief.items()}},sort_keys=True),flush=True)
 rng_after=hashlib.sha256(pickle.dumps((np.random.get_state(),random.getstate()),protocol=4)+torch.get_rng_state().numpy().tobytes()).hexdigest()
 require(rng_before==rng_after and sha(analysis)==ANALYSIS_SHA,"RNG/input changed")
 report={"schema":SCHEMA,"status":"CPU_diagnosis_complete","attribution":"Built with PriorLabs-TabPFN",
  "source_sha256":sha(__file__),"input_analysis_sha256":ANALYSIS_SHA,"math_report_sha256":MATH_SHA,"codec_source_sha256":CODEC_SHA,
  "uniforms_F32":[float(x) for x in us],"uniform_bytes":[x.tobytes().hex() for x in us],
  "threshold":threshold,"threshold_scope":"upper physicalFloat64 correctly-rounded endpoint threshold from180digit math; stricter latentRT domain is also checked separately",
  "no_temperature_double_division":True,"CPU_F32_replay_not_actual_GPU_softmax":True,
  "B_C_float64_promote_existing_captured_F32_borders_no_refit":True,"average_before_softmax_False_source_bound_not_runtime_observation":True,
  "actual_GPU_preclip_counts_and_mass_unobserved":True,"records":records,"default_numpy_python_torchCPU_RNG_before":rng_before,"default_numpy_python_torchCPU_RNG_after":rng_after,
  "new_GPU_neural_model_constructor_world_posterior_G3_live_sampler_or_RNG_draw_calls":0,
  "scope":"new CPU arithmetic references on saved counterfactual focal densities; no policy implementation or recovered physical cohort",
  "wall_seconds":time.perf_counter()-started}
 (out/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
 (out/"checksums.json").write_text(json.dumps({str(p.relative_to(out)):sha(p) for p in sorted(out.rglob("*")) if p.is_file() and p.name!="checksums.json"},indent=2,sort_keys=True)+"\n")
 print(json.dumps({"output":str(out),"report_sha256":sha(out/"report.json"),"records":len(records),"wall_seconds":report["wall_seconds"]}))
if __name__=="__main__":
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--analysis",required=True);p.add_argument("--output",required=True)
 a=p.parse_args();run(a.analysis,a.output)
