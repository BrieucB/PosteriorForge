#!/usr/bin/env python3
"""Offline focal-row capture analysis only. Built with PriorLabs-TabPFN.

No neural fits, model, sampler, RNG, G3, posterior or reference integration.
Full counterfactual capture remains on Karolina; derived focal arrays are
byte-copies of recorded rows. Exact normalization is a diagnostic law only.
"""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,math,time
from decimal import Decimal,localcontext
from fractions import Fraction
from pathlib import Path
import numpy as np
from posteriorforge import prior_coordinates_v1 as codec

REFERENCE_SHA256="8c35b5e10e5b15383b8a2112f25fa555e8d11016a30fef0abcc4167c527e87c1"
CODEC_SHA256="d8ce6c49e7b65c6ee55c59b7243d4ce943cbea28c92ebed0329798074aaa70c5"
CAPTURE_SOURCE_SHA256="883fb5840cf3460fa5162b34d0fc24a43b43d70bb3bcff3585d0cfd3e483cc38"
CAPTURE_SCHEMA="posteriorforge.prior_tail_conditional_capture.v1"
MATH_SHA256="d5a22a8a617023ced6d1b043ecca6185bf384a5dcd33bb2ff862050f430028e5"
SCHEMA="posteriorforge.prior_tail_focal_offline_analysis.v1"
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(value,message):
 if not value:raise ValueError(message)
def descriptor(a):
 a=np.asarray(a);return {"shape":list(a.shape),"dtype":a.dtype.str,"sha256":hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()}
def number(x):
 x=float(x);return x if math.isfinite(x) else str(x)
def ratio(x):return {"numerator":str(x.numerator),"denominator":str(x.denominator)}
def reference_module():
 p=Path(__file__).with_name("prior_tail_icdf_reference_v1.py");require(sha(p)==REFERENCE_SHA256,"independent reference changed")
 s=importlib.util.spec_from_file_location("_offline_tail_exact_reference",p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def noise_diagnostic(z,slot,threshold):
 result={"input_z":number(z),"scope":"single noise-coordinate diagnostic, never a physical21-vector or recovered sample"}
 if not math.isfinite(float(z)):
  return {**result,"finite":False,"frozen_d8_strict_noise_decode_accepted":False}
 result["finite"]=True
 with localcontext() as context:
  context.prec=90;n,d=float(z).as_integer_ratio()
  delta=Decimal(n)/Decimal(d)-Decimal(threshold)
  result["ideal_positive_quantile_correctly_rounds_to_upper_Float64"]=bool(delta>Decimal("1e-60")) if abs(delta)>Decimal("1e-60") else None
 result["ideal_threshold_scope"]="upper endpoint only; threshold from frozen180-digit math report; lower endpoint not excluded by False"
 spec=codec._geometry()["noise"][slot]
 try:
  physical=codec._noise_decode(np.array([z],np.float64),spec)
  back=float(codec._noise_encode(physical,spec)[0]);error=abs(back-float(z))
  result.update(frozen_d8_strict_noise_decode_accepted=True,diagnostic_physical_scalar=float(physical[0]),
   frozen_noise_latent_reencode=back,noise_latent_roundtrip_absolute_error=error,
   current_Float32_latent_roundtrip_bound=float(codec.F32_LATENT_ADMISSIBILITY*(1+abs(float(z)))),
   current_Float64_latent_roundtrip_bound=float(codec.F64_LATENT_ADMISSIBILITY*(1+abs(float(z)))),
   would_pass_noise_only_Float32_latent_roundtrip=bool(error<=codec.F32_LATENT_ADMISSIBILITY*(1+abs(float(z)))),
   would_pass_noise_only_Float64_latent_roundtrip=bool(error<=codec.F64_LATENT_ADMISSIBILITY*(1+abs(float(z)))))
 except codec.CoordinateError as error:
  result.update(frozen_d8_strict_noise_decode_accepted=False,error=str(error),
   would_pass_noise_only_Float32_latent_roundtrip=False,would_pass_noise_only_Float64_latent_roundtrip=False)
 return result

def focal_subset(arrays,row,label):
 n=arrays["probabilities"].shape[0]
 require(label in ("N1024","N1") and n==(1024 if label=="N1024" else 1),"declared captured row count differs")
 require(0<=row<n,"focal row outside capture")
 per_grid={"bin","cdf_before","rest","selected_probability","output","fraction_diagnostic_F32"}
 per_row={"logits","probabilities","cdf","raw_translated_probabilities","component_raw_mass","raw_model_logits"}
 subset={};provenance={}
 for name,a in arrays.items():
  if name in per_grid:
   require(a.ndim==2 and a.shape[1]==n,"grid-row array shape differs:"+name);v=a[:,row].copy();mode="axis1 focal row"
  elif name in per_row:
   require(a.ndim>=1 and a.shape[0]==n,"per-row array shape differs:"+name)
   v=a[row].copy() if a.ndim>1 else a[row:row+1].copy();mode="axis0 focal row"
  else:v=a.copy();mode="whole per-member array"
  subset[name]=v;provenance[name]={"selection":mode,"source_shape":list(a.shape),"focal_row_index":row,"derived":descriptor(v)}
 return subset,provenance

def analyze(capture_manifest,expected_sha,output):
 t=time.perf_counter();capture_manifest=Path(capture_manifest);require(sha(capture_manifest)==expected_sha,"explicit CLOSED capture manifest differs")
 m=json.loads(capture_manifest.read_bytes())
 require(m["schema"]==CAPTURE_SCHEMA and m["status"] in ("two_stage_conditional_capture_complete","failed_or_incomplete_conditional_capture"),
  "closed declared capture required; partial running files are not analyzed")
 require(m["counterfactual_reevaluation"] is True and m["historical_grouping_member_recreated"] is False
  and m["posterior_vectors_generated"]==m["reference_integrations"]==m["G3_updates"]==0,"diagnostic scope differs")
 require(m["sources"]["diagnostic"]==CAPTURE_SOURCE_SHA256 and m["sources"]["codec"]==CODEC_SHA256
  and codec.source_sha256()==CODEC_SHA256,"capture/current codec source differs")
 helper=reference_module();out=Path(output);require(not out.exists(),"fresh offline output required");out.mkdir(parents=True)
 (out/"focal-densities").mkdir()
 mathpath=Path(__file__).resolve().parents[1]/"evidence/accuracy-prior-coordinates-v1/noise-inverse-math-diagnostic-01/reproducible-scalars.json"
 require(sha(mathpath)==MATH_SHA256,"frozen mathematical representability report differs")
 mathreport=json.loads(mathpath.read_bytes());threshold=mathreport["exact_prior_z_above_which_nearest_Float64_is_endpoint"]
 details=[];rows=[];subsets=[]
 for stage_record in m["stages"]:
  fixture=stage_record["fixture"];stage=fixture["stage"];slot=codec.ORDER[stage]
  require((stage,fixture["row"],fixture["window"],fixture["focal_offset"]) in
   [(15,4355,[4096,5120],259),(19,41765,[40960,41984],805)],"undeclared fixed fixture")
  for arm,entries in stage_record["arms"].items():
   require(arm in ("original","candidate"),"unknown arm")
   for entry in entries:
    filename=entry["file"];require(Path(filename).name==filename,"capture file must be sibling")
    p=capture_manifest.parent/filename;require(sha(p)==entry["sha256"],"capture density archive differs")
    with np.load(p,allow_pickle=False) as archive:
     require(set(archive.files)==set(entry["arrays"]),"capture array names differ")
     arrays={name:archive[name] for name in archive.files}
    require({name:descriptor(a) for name,a in arrays.items()}==entry["arrays"],"capture array byte descriptors differ")
    label=entry["label"];focal=fixture["focal_offset"] if label=="N1024" else 0
    subset,provenance=focal_subset(arrays,focal,label)
    dest=out/"focal-densities"/(Path(filename).stem+"-focal.npz")
    with dest.open("xb") as stream:np.savez(stream,**subset)
    subsets.append({"file":str(dest.relative_to(out)),"sha256":sha(dest),"source_capture_file":str(p.resolve()),
     "source_capture_file_sha256":entry["sha256"],"row_index":focal,"logical_row":fixture["row"],
     "derived_arrays":provenance,"byte_copy_no_recompute_or_cast":True})
    u=subset["uniforms"];pweights=subset["probabilities"];borders=subset["borders"];observed=subset["output"]
    require(all(a.dtype==np.float32 for a in (u,pweights,borders,observed)),"captured focal Float32 types differ")
    exact=helper.reference_row(pweights,borders,u,observed_values=observed)
    stress=helper.reference_row(pweights,borders,np.array([1],np.float32))
    detail={"stage":stage,"arm":arm,"member":entry["member"],"label":label,"focal_row_index":focal,
     "source_file_sha256":entry["sha256"],"exact_reference":exact,"reference_u1_stress":stress,
     "stock_u1_out_of_contract":entry["ICDF"]["u1_stress"]}
    details.append(detail)
    historical=fixture["reconstructed_uniform"]["float32_bytes_hex"]
    actual_index=next(i for i,value in enumerate(u) if value.tobytes().hex()==historical)
    points=[]
    for i,ref in enumerate(exact["points"]):
     index=int(subset["bin"][i]);require(0<=index<len(pweights),"captured selected bin invalid")
     value=float(observed[i]);left=float(borders[index]);right=float(borders[index+1]);prob=float(subset["selected_probability"][i])
     before=float(subset["cdf_before"][i]);rest=float(subset["rest"][i])
     stock={"bin":index,"left":left,"right":right,"cdf_before_F32":number(before),"CDF_at_bin_F32":number(subset["cdf"][index]),
      "rest_F32":number(rest),"selected_probability_F32":number(prob),"output_F32":number(value),
      "fraction_diagnostic_F32":number(subset["fraction_diagnostic_F32"][i]),
      "fraction_scope":"capturedrest/p diagnostic; source computes width*rest/p without a fraction temporary",
      "exact_ratio_of_captured_rest_to_probability":ratio(Fraction.from_float(rest)/Fraction.from_float(prob)) if math.isfinite(rest) and prob>0 else None,
      "U_above_last_captured_CDF":bool(float(u[i])>float(subset["cdf"][-1])),
      "outside_selected_bar":not(math.isfinite(value) and left<=value<=right),
      "outside_outer_borders":not(math.isfinite(value) and float(borders[0])<=value<=float(borders[-1]))}
     points.append({"U_Float32":float(u[i]),"U_bytes_hex":u[i].tobytes().hex(),"authenticated_historical_U_index":i==actual_index,
      "stock":stock,"reference_bin":ref["bin"],"reference_fraction_exact":ref["interpolation_fraction_exact"],
      "reference_mass_exact_normalized":ref["bin_mass_exact_normalized"],"reference_inverse_Float64":ref["inverse_nearest_Float64"],
      "reference_inside_outer_borders":ref["inside_outer_borders"],"stock_normalized_reference_CDF_residual":ref["observed_frozen_output"].get("normalized_reference_CDF_residual"),
      "stock_CDF_and_support_detail":ref["observed_frozen_output"],
      "stock_d8_noise_domain":noise_diagnostic(value,slot,threshold),
      "normalized_reference_d8_noise_domain":noise_diagnostic(ref["inverse_nearest_Float64"],slot,threshold)})
    actual=points[actual_index]
    record={"stage":stage,"physical_slot":slot,"arm":arm,"member":entry["member"],"label":label,
     "logical_row":fixture["row"],"focal_index_in_captured_batch":focal,"source_file":filename,"source_file_sha256":entry["sha256"],
     "weight_sum_exact":exact["input_weight_sum_exact"],"weight_sum_Float64":exact["input_weight_sum_nearest_Float64"],
     "last_captured_CDF_F32":float(subset["cdf"][-1]),"outer_borders":exact["outer_borders"],
     "essential_support_endpoints":exact["essential_support_endpoints"],
     "provided_U_points":points,"authenticated_historical_U_counterfactual_result":actual,
     "u1_reference_stress":stress["points"][0],"u1_stock_stress":entry["ICDF"]["u1_stress"]}
    rows.append(record)
    print(json.dumps({"stage":stage,"arm":arm,"member":entry["member"],"label":label,"U":actual["U_Float32"],
     "stock":actual["stock"]["output_F32"],"reference":actual["reference_inverse_Float64"],
     "stock_outside_grid":actual["stock"]["outside_outer_borders"],
     "stock_d8_accept":actual["stock_d8_noise_domain"]["frozen_d8_strict_noise_decode_accepted"],
     "ref_d8_accept":actual["normalized_reference_d8_noise_domain"]["frozen_d8_strict_noise_decode_accepted"]},sort_keys=True),flush=True)
 groups=[]
 for stage in (15,19):
  for label in ("N1024","N1"):
   for arm in ("original","candidate"):
    selected=[r for r in rows if (r["stage"],r["label"],r["arm"])==(stage,label,arm)]
    if not selected:continue
    focus=[r["authenticated_historical_U_counterfactual_result"] for r in selected]
    groups.append({"stage":stage,"label":label,"arm":arm,"members":sorted(r["member"] for r in selected),
     "stock_outside_outer_support_count":sum(p["stock"]["outside_outer_borders"] for p in focus),
     "stock_d8_noise_decode_failure_count":sum(not p["stock_d8_noise_domain"]["frozen_d8_strict_noise_decode_accepted"] for p in focus),
     "normalized_reference_d8_noise_decode_failure_count":sum(not p["normalized_reference_d8_noise_domain"]["frozen_d8_strict_noise_decode_accepted"] for p in focus),
     "normalized_reference_still_hits_strict_physical_endpoint_or_domain_failure":any(not p["normalized_reference_d8_noise_domain"]["frozen_d8_strict_noise_decode_accepted"] for p in focus),
     "stock_range":[min(float(p["stock"]["output_F32"]) for p in focus),max(float(p["stock"]["output_F32"]) for p in focus)],
     "normalized_reference_range":[min(p["reference_inverse_Float64"] for p in focus),max(p["reference_inverse_Float64"] for p in focus)]})
 report={"schema":SCHEMA,"status":"offline_diagnostic_complete","attribution":"Built with PriorLabs-TabPFN",
  "capture_manifest_sha256":expected_sha,"capture_status_preserved":m["status"],"capture_source_sha256":CAPTURE_SOURCE_SHA256,
  "reference_source_sha256":REFERENCE_SHA256,"codec_source_sha256":CODEC_SHA256,"math_report_sha256":MATH_SHA256,
  "counterfactual_reevaluation":True,"historical_grouping_or_member_recreated":False,
  "semantics":"exact normalization of EXISTING capturedF32softmax weights; not softmax64 and not frozenF32cumulative-roundoff law",
  "normalized_inverse_cures_physical_representability_not_assumed":True,"groups":groups,"entries":rows,
  "focal_subsets":subsets,"full_whole_batch_density_archives_remain_on_Karolina":True,
  "posterior_vectors_or_G3_or_models_or_RNG_or_reference_integrations_generated":0,
  "scope_limits":"same reconstructedF32U but counterfactual known1024/N1batches and all8heads; missing historical grouping/member/logits are not inferred; scalar noise checks are not complete physical21vectors",
  "wall_seconds":time.perf_counter()-t}
 require(sha(capture_manifest)==expected_sha,"closed capture changed")
 (out/"exact-reference-details.json").write_text(json.dumps(details,indent=2,sort_keys=True,allow_nan=False)+"\n")
 report["exact_reference_details_sha256"]=sha(out/"exact-reference-details.json")
 (out/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
 (out/"checksums.json").write_text(json.dumps({str(p.relative_to(out)):sha(p) for p in sorted(out.rglob("*")) if p.is_file() and p.name!="checksums.json"},indent=2,sort_keys=True)+"\n")
 print(json.dumps({"output":str(out),"report_sha256":sha(out/"report.json"),"groups":groups},sort_keys=True),flush=True)
 return report

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument("--capture-manifest",required=True);p.add_argument("--capture-sha256",required=True);p.add_argument("--output",required=True)
 a=p.parse_args();analyze(a.capture_manifest,a.capture_sha256,a.output)
if __name__=="__main__":main()
