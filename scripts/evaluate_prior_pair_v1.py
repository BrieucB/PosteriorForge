#!/usr/bin/env python3
"""Pair-bound original physical gates and four-cohort scientific figures.

Built with PriorLabs-TabPFN. This assessment never calls refresh, generates new
normals, invokes a model/GPU, or integrates a reference. Actual saved G3 draws
are checked algebraically. All50000 physical rows and all21+32 gates are retained.
"""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,subprocess,time
from pathlib import Path
import numpy as np
from posteriorforge.artifacts import sha256_file,write_json
from posteriorforge.banks import derive_seed,load_heldout
from posteriorforge.science.analytical_benchmark_v2 import array_descriptor
WRAPPER_SHA256='22e57084351631f36c6989cc01fe8bf4f2827095e587c496cda2e7f368d3cdb9'
SCHEMA="posteriorforge.prior_coordinate_pair_assessment.v1"
COUNT=50000
FOCUS=(0,7,8,10)
COLORS=("#c58a37","#777777","#1479b8","#111111")
LABELS=("OLD matched physical-target neural mixed FP16/FP32 + physical G3",
 "NEW pair-qualified prior coords FP32: 19 cached; all8 original at6/10",
 "NEW same parent + ONE SEPARATE physical G3","qualified same-case reference")

def require(value,message):
 if not value:raise ValueError(message)

def load(filename,digest):
 path=Path(__file__).with_name(filename)
 require(isinstance(digest,str) and sha256_file(path)==digest,"pair assessment source unpinned/changed: "+filename)
 spec=importlib.util.spec_from_file_location("_pair_evaluation_"+path.stem,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def source_guard():
 w=load("refresh_prior_pair_global_block_v1.py",WRAPPER_SHA256)
 producer,pairs,pins=w.source_guard()
 root=Path(__file__).resolve().parents[1]
 w.BASE.verify_git_sources(root,pins["git_revision"],{
  "scripts/"+Path(__file__).name:sha256_file(__file__),
  "scripts/refresh_prior_pair_global_block_v1.py":WRAPPER_SHA256})
 return w,producer,pairs,{**pins,"pair_evaluator":sha256_file(__file__),"pair_G3_wrapper":WRAPPER_SHA256}

def authenticate_hybrid(path,expected_sha,parent_sha,before,w,wrapper_pins,spec):
 require(sha256_file(path)==expected_sha,"explicit actual-pair G3 manifest SHA differs")
 m=json.loads(Path(path).read_bytes())
 require(m["schema"]==w.SCHEMA and m["status"]=="complete" and m["forensic"] is False
  and m["original_parent_status"]=="complete" and m["parent"]["parent_manifest_sha256"]==parent_sha
  and not w.BASE.policy_differences(m["pair_spec"],spec.as_dict())
  and not w.BASE.policy_differences(m["parent"]["pair_spec"],spec.as_dict())
  and m["context_seed"]==spec.context_seed and m["case"]==spec.case and m["repetition"]==0 and m["requested_count"]==COUNT
  and not w.BASE.policy_differences(w.BASE.immutable_pins(m["source_pins"]),w.BASE.immutable_pins(wrapper_pins))
  and m["conditional_kernel"]==w.kernel.metadata() and m["coordinate_decode_after_global_refresh"] is False,
  "only closed same-pair G3 lineage accepted")
 root=Path(__file__).resolve().parents[1]
 w.BASE.verify_git_sources(root,m["source_pins"]["git_revision"],{
  "scripts/refresh_prior_pair_global_block_v1.py":m["source_pins"]["wrapper"],
  "src/posteriorforge/performance_prior_pair_v1.py":w.PRODUCER_SHA256,
  "src/posteriorforge/prior_coordinate_pair_v1.py":w.PAIR_SOURCE_SHA256,
  "scripts/qualify_prior_coordinate_pair_v1.py":w.PAIR_QUALIFIER_SHA256,
  "src/posteriorforge/conditional_globals.py":w.KERNEL_SHA256,
  "src/posteriorforge/execution_physical_fifo_v1.py":w.F64_HELPER_SHA256,
  "scripts/assess_prior_forensic_replay_v1.py":w.DRAW_SOURCE_SHA256})
 p=w._safe_sibling(path,m["samples_file"]);z=w._safe_sibling(path,m["actual_draws_file"])
 require(sha256_file(p)==m["samples_sha256"] and sha256_file(z)==m["actual_draws_sha256"],"physical output/actual Z file differs")
 with np.load(p,allow_pickle=False) as saved:
  require(saved.files==["samples"],"physical sample key differs");after=saved["samples"]
 with np.load(z,allow_pickle=False) as saved:
  require(saved.files==["standard_normal_draws"],"actual Z key differs");normals=saved["standard_normal_draws"]
 require(before.dtype==after.dtype==normals.dtype==np.float64 and before.shape==after.shape==(COUNT,21)
  and normals.shape==(COUNT,3) and array_descriptor(after)==m["samples"],"strict complete physical/draw shapes/dtypes required")
 r=m["refresh"];seed=derive_seed(spec.context_seed,w.ROLE,case=spec.case,repetition=0)
 require(not w.BASE.policy_differences(r["pair_spec"],spec.as_dict())
  and r["seed"]==seed and r["role"]==w.ROLE and r["bit_generator"]=="PCG64"
  and r["initial_private_state"]==np.random.PCG64(seed).state
  and type(r["standard_normal_method_calls"]) is int and r["standard_normal_method_calls"]==1 and r["standard_normal_draw_shape"]==[COUNT,3]
  and type(r["standard_normal_scalar_count"]) is int and r["standard_normal_scalar_count"]==COUNT*3
  and type(r["global_block_refresh_count_per_row"]) is int and r["global_block_refresh_count_per_row"]==1
  and r["draws_are_actual_returned_matrix_not_rerolled"] is True
  and r["default_numpy_python_RNG_preserved"] is True
  and r["default_numpy_python_RNG_before_sha256"]==r["default_numpy_python_RNG_after_sha256"]
  and r["all18_physical_columns_byte_identical"] is True and r["input_array_unchanged"] is True
  and r["coordinate_decode_after_global_refresh"] is False,"same-pair one-call physical G3 seed/control differs")
 require(array_descriptor(normals)==r["standard_normal_draws"],"actual normal matrix descriptor differs")
 for name in ("initial_private_state","final_private_state"):
  require(hashlib.sha256(json.dumps(r[name],sort_keys=True,separators=(",",":")).encode()).hexdigest()==r[name+"_sha256"],"actual private state hash differs")
 mean,cov=w.kernel.conditional_parameters(before)
 require(after[:,:3].tobytes()==(mean+normals@np.linalg.cholesky(cov).T).tobytes()
  and after[:,3:].tobytes()==before[:,3:].tobytes(),"saved actual Z algebra/physical18 identity differs")
 require(np.isfinite(r["G3_kernel_math_seconds"]) and r["G3_kernel_math_seconds"]>=0,"actual G3 math wall time missing")
 return m,after

def validate_reference_custody(parent,custody,query,spec):
 require(parent["context_seed"]==spec.context_seed and parent["case"]==spec.case
  and parent["pair_spec"]==spec.as_dict() and all(type(v) is int for v in parent["pair_spec"].values()),
  "actual reference request PairSpec differs")
 require(type(custody["reference_case"]) is int and custody["reference_case"]==spec.case
  and custody["binding"]["observation_sha256"]==array_descriptor(query)["sha256"]
  and parent["reference_binding"]==custody["runner_reference_binding"],
  "actual case/query qualified reference custody differs")
 return {"passed":True,"reference_case":spec.case,"observation":array_descriptor(query),
  "reference_source":"unchanged original qualified case reference and 53 projection witnesses"}

def load_baseline(frozen,pairs,spec,completed_grid,raw):
 """One predetermined same-seed/case OLD cohort from authenticated613 membership."""
 owner=pairs.prefix_parent(spec,completion_receipt=completed_grid)
 parent_path=Path(owner["point_paths"]["parent"]);hybrid_path=Path(owner["point_paths"]["hybrid"])
 assessment_path=Path(owner["point_paths"]["assessment"])
 require(sha256_file(parent_path)==owner["manifest_sha256"]
  and sha256_file(assessment_path)==owner["old_assessment_sha256"],"matched old display parent/assessment bytes differ")
 old_wrapper=frozen.load("refresh_ss15_campaign_global_block.py",frozen.WRAPPER_SHA)
 parent,hybrid,_,values,auth=frozen.authenticate(old_wrapper,parent_path,hybrid_path,owner["old_hybrid_manifest_sha256"])
 require(parent["context_seed"]==spec.context_seed and parent["case"]==spec.case
  and parent["raw_scientific_bindings"]==raw and values.shape==(COUNT,21),"old display is not identical actual bank/query/seed/case")
 assessment=json.loads(assessment_path.read_bytes())
 require(assessment["context_seed"]==spec.context_seed and assessment["case"]==spec.case
  and assessment["all50000_rows_used"] is True,"old display assessment pair/scope differs")
 return values,assessment,{"pair_spec":spec.as_dict(),"completion_sha256":owner["completion_sha256"],
  "parent_manifest_sha256":owner["manifest_sha256"],"hybrid_manifest_sha256":owner["old_hybrid_manifest_sha256"],
  "hybrid_samples_sha256":hybrid["samples_sha256"],"assessment_sha256":owner["old_assessment_sha256"],
  "old_physical_array":array_descriptor(values),"authentication":auth,
  "selection_rule":"fixed613 same seed/case; no outcome selection or new old-cohort metrics"}

def run(*,root,context_seed,case,parent_manifest,parent_manifest_sha256,hybrid_manifest,hybrid_manifest_sha256,
 execution_screen,pair_cpu_receipt,output):
 started=time.perf_counter();w,producer,pairs,sources=source_guard();source_seconds=time.perf_counter()-started
 spec=pairs.PairSpec(context_seed,case)
 t=time.perf_counter();parent,before,auth=w.load_parent(parent_manifest,parent_manifest_sha256,scratch=root,spec=spec,
  execution_screen=execution_screen,pair_cpu_receipt=pair_cpu_receipt)
 hybrid,after=authenticate_hybrid(hybrid_manifest,hybrid_manifest_sha256,parent_manifest_sha256,before,w,w.source_guard()[2],spec)
 validation_seconds=time.perf_counter()-t
 old=w.load("evaluate_prior_coordinates_pilot_v1.py",w.OLD_EVAL_SHA256)
 frozen=old.load("evaluate_ss15_sigma_native_campaign.py",old.FROZEN_EVALUATOR_SHA256)
 binder=frozen.load("evaluate_cdf_pilot.py",frozen.BINDER_SHA)
 pure=frozen.load("evaluate_ss15_pilot.py",frozen.PURE_HELPER_SHA)
 results=Path(root)/"results";gatepath=results/"gate-final.json"
 require(sha256_file(gatepath)==binder.BASELINE_GATE_SHA256,"unchanged original physical gate required")
 gate=json.loads(gatepath.read_bytes());query=load_heldout(results/"banks-v1",spec.case).obs.reshape(1,33)
 t=time.perf_counter()
 arrays,reference,assessments,controls=frozen.reference_and_assess(results,gate,parent,query,before,after,binder)
 reference_pair_check=validate_reference_custody(parent,reference,query,spec)
 metrics_seconds=time.perf_counter()-t
 t=time.perf_counter();baseline,baseline_assessment,baseline_custody=load_baseline(frozen,pairs,spec,
  results/"accuracy-ss15-v1/sigma-native-grid-completion-01/completion.json",parent["raw_scientific_bindings"])
 baseline_seconds=time.perf_counter()-t
 out=Path(output);out.mkdir(parents=True,exist_ok=False)
 record={"schema":SCHEMA,"status":assessments["after"]["status"],"passed":assessments["after"]["passed"],
  "attribution":"Built with PriorLabs-TabPFN","pair_spec":spec.as_dict(),"context_seed":spec.context_seed,"case":spec.case,
  "source_pins":sources,"parent_manifest_sha256":parent_manifest_sha256,"hybrid_manifest_sha256":hybrid_manifest_sha256,
  "authentication":auth,"closed_pair_production_parent":True,"forensic_cohort_used":False,
  "absolute_physical_assessments":assessments,"summary":{k:pure.summary(v,parent["parameter_order"]) for k,v in assessments.items()},
  "unchanged18_control":controls,"reference":reference,"actual_pair_reference_check":reference_pair_check,"original_gate_sha256":sha256_file(gatepath),
  "all50000_rows_used":True,"scientific_limits_unchanged":True,"focus_indices":list(FOCUS),
  "old_baseline_summary":baseline_assessment["summary"],"old_baseline_custody":baseline_custody,
  "configuration_comparison":{"coordinates_precision_routing_all_changed":True,"isolated_coordinate_effect_claimed":False},
  "physical_G3_policy":"ONE unchanged e346 physical conditional; actual Z saved; decoded physical18 held fixed",
  "CPU_wall_seconds":{"explicit_initial_source_validation":source_seconds,"parent_and_G3_validation":validation_seconds,
   "same_case_reference_custody_and_original_metrics":metrics_seconds,"matched_old_display_validation":baseline_seconds,
   "G3_kernel_math_recorded_during_refresh":hybrid["refresh"]["G3_kernel_math_seconds"]},
  "CPU_timing_scope":"wall time, no optimization; validation includes repeated source/data/proof checks; initial source subset not additive to validation",
  "new_model_GPU_draws_or_reference_integrations_during_assessment":0,"new_global_or_normal_draws_during_assessment":0,"plots_pending":True}
 write_json(out/"assessment.json",record)
 print(json.dumps({"pair_scalar_output":str(out),"pair_spec":spec.as_dict(),"status":record["status"],
  "passed":record["passed"],"summary":record["summary"]},sort_keys=True),flush=True)
 t=time.perf_counter();record["figures"]=plots(out,baseline,before,after,arrays,assessments,parent["parameter_order"],binder,spec)
 record["CPU_wall_seconds"]["plotting"]=time.perf_counter()-t;record["plots_pending"]=False
 require(sha256_file(parent_manifest)==parent_manifest_sha256 and sha256_file(hybrid_manifest)==hybrid_manifest_sha256
  and sha256_file(gatepath)==binder.BASELINE_GATE_SHA256,"closed pair/G3/reference-gate evidence changed")
 write_json(out/"assessment.json",record)
 write_json(out/"checksums.json",{p.name:sha256_file(p) for p in out.iterdir() if p.is_file()})
 return record


def plots(out,baseline,before,after,arrays,assessments,order,binder,spec):
 import matplotlib
 matplotlib.use("Agg")
 import matplotlib.pyplot as plt
 from matplotlib.lines import Line2D
 legend=[Line2D([0],[0],color=c,lw=1.8,label=l) for c,l in zip(COLORS,LABELS)]
 def figure(rows,cols,size,title):
  f,ax=plt.subplots(rows,cols,figsize=size,squeeze=False)
  f.suptitle(title+f"\nseed{spec.context_seed}, case{spec.case}: FP32 / cached19 / original all8 at6+10; physical G3 separate",fontsize=13,y=.995)
  f.legend(handles=legend,loc="upper center",bbox_to_anchor=(.5,.955),ncol=2,fontsize=9)
  f.text(.5,.012,"Built with PriorLabs-TabPFN · all50000 rows normalize each curve · original physical gate unchanged · G3 is separate",ha="center",va="bottom",fontsize=8)
  f.subplots_adjust(left=.07,right=.99,bottom=.07,top=.89,wspace=.28,hspace=.55);return f,ax.flat
 def domain(j):
  x=arrays["queries"]/arrays["directions"][j,j] if j<21 else arrays["queries"];cdf=arrays["cdfs"][j]
  lo=binder.reference_quantile(x,cdf,.001);hi=binder.reference_quantile(x,cdf,.999);pad=.03*(hi-lo);return x,cdf,lo-pad,hi+pad
 def density(ax,j):
  x,cdf,lo,hi=domain(j);bins=np.linspace(lo,hi,41);width=np.diff(bins)
  ax.stairs(np.diff(np.interp(bins,x,cdf))/width,bins,color=COLORS[3],lw=1.4)
  for a,color in ((baseline,COLORS[0]),(before,COLORS[1]),(after,COLORS[2])):
   count,_=np.histogram(a[:,j],bins);ax.stairs(count/(len(a)*width),bins,color=color,lw=1.1)
  old=assessments["before"]["marginal_cdf"]["upper"][j];new=assessments["after"]["marginal_cdf"]["upper"][j]
  ax.set(title=f"{order[j]}: CDF upper {old:.3f} → {new:.3f}",xlim=(lo,hi),xlabel="physical coordinate",ylabel="binned density")
 def save(f,name):
  f.savefig(out/name,dpi=155);plt.close(f);return {"file":name,"sha256":sha256_file(out/name)}
 figs={};f,axes=figure(2,2,(12,8.6),"Pair-qualified prior-coordinate cohorts: four predeclared physical coordinates")
 f.subplots_adjust(top=.81,bottom=.11,hspace=.5,wspace=.27)
 for ax,j in zip(axes,FOCUS):density(ax,j)
 figs["focus_density"]=save(f,"pair-density-focus.png")
 print(json.dumps({"first_focus_density_ready":str(out/"pair-density-focus.png")}),flush=True)
 f,axes=figure(7,3,(14,20),"Pair-qualified prior-coordinate cohorts: all21 physical densities")
 for ax,j in zip(axes,range(21)):density(ax,j)
 figs["density"]=save(f,"pair-density.png")
 precision=assessments["after"]["reference_qualification"]["precision"]
 for label,indices,shape,size,title in (("marginal_cdf",range(21),(7,3),(14,20),"Pair-qualified all21 physical marginal CDFs"),
  ("joint_cdf",range(21,53),(8,4),(17,23),"Pair-qualified all32 fixed physical projection CDFs")):
  f,axes=figure(*shape,size,title)
  for ax,j in zip(axes,indices):
   x,cdf,lo,hi=domain(j);grid=np.linspace(lo,hi,700)
   for a,color in ((baseline,COLORS[0]),(before,COLORS[1]),(after,COLORS[2])):
    values=np.sort(a[:,j] if j<21 else a@arrays["directions"][j]);ax.plot(grid,np.searchsorted(values,grid,side="right")/len(values),color=color,lw=1.1)
   ax.plot(x,cdf,color=COLORS[3],lw=1.3);ax.fill_between(x,np.maximum(0,cdf-precision),np.minimum(1,cdf+precision),color=COLORS[3],alpha=.12)
   key="marginal_cdf" if j<21 else "joint_projection_cdf";index=j if j<21 else j-21
   a=assessments["before"][key]["upper"][index];b=assessments["after"][key]["upper"][index];name=order[j] if j<21 else f"fixed joint {j-20}"
   ax.set(title=f"{name}: {a:.3f} → {b:.3f}",xlim=(lo,hi),ylim=(-.02,1.02),xlabel="physical coordinate" if j<21 else "physical-vector projection",ylabel="CDF")
  figs[label]=save(f,"pair-"+label.replace("_","-")+".png")
 return figs

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ("root","parent-manifest","parent-manifest-sha256","hybrid-manifest","hybrid-manifest-sha256","execution-screen","pair-cpu-receipt","output"):
  p.add_argument("--"+name,required=True)
 p.add_argument("--context-seed",type=int,required=True);p.add_argument("--case",type=int,required=True)
 run(**vars(p.parse_args()))
if __name__=="__main__":main()
