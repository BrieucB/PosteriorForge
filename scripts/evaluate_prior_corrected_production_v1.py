#!/usr/bin/env python3
"""Original physical assessment/plots of CLOSED corrected production + ONE G3.

Built with PriorLabs-TabPFN. No forensic/failed cohort is accepted. Saved actual
normal draws are replayed algebraically, never rerolled. No model, GPU, new
posterior update, simulator or reference integration occurs during assessment.
"""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,subprocess
from pathlib import Path
import numpy as np
from posteriorforge.artifacts import sha256_file,write_json
from posteriorforge.banks import derive_seed,load_heldout
from posteriorforge.science.analytical_benchmark_v2 import array_descriptor

WRAPPER_SHA256="1d62028e28e93f1fe4f4e2abd98ccedd2a63e0379a86f908a05ac64787061e19"
SCHEMA="posteriorforge.prior_corrected_production_assessment.v1"
COUNT=50000
FOCUS=(0,7,8,10)
COLORS=("#c58a37","#777777","#1479b8","#111111")
LABELS=("OLD physical-target neural mixed FP16/FP32 + physical G3",
 "NEW corrected production prior coords FP32: 19 cached; all8 original at6/10",
 "NEW same production parent + ONE SEPARATE physical G3","qualified reference")

def require(value,message):
 if not value:raise ValueError(message)

def load(filename,digest):
 path=Path(__file__).with_name(filename)
 require(isinstance(digest,str) and sha256_file(path)==digest,"source not frozen: "+filename)
 spec=importlib.util.spec_from_file_location("_corrected_eval_"+path.stem,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def source_guard():
 w=load("refresh_prior_corrected_production_v1.py",WRAPPER_SHA256)
 producer,observer,pins=w.source_guard()
 root=Path(__file__).resolve().parents[1];self_path=Path(__file__)
 raw=subprocess.run(["git","-C",str(root),"show","HEAD:"+str(self_path.relative_to(root))],check=True,capture_output=True).stdout
 require(hashlib.sha256(raw).hexdigest()==sha256_file(self_path),"new production evaluator must be committed before assessment")
 return w,producer,{**pins,"production_evaluator":sha256_file(__file__),"production_G3_wrapper":WRAPPER_SHA256}

def authenticate_hybrid(path,expected_sha,parent_sha,before,w,wrapper_pins):
 require(sha256_file(path)==expected_sha,"explicit corrected G3 manifest SHA differs")
 m=json.loads(Path(path).read_bytes())
 require(m["schema"]==w.SCHEMA and m["status"]=="complete" and m["forensic"] is False
  and m["original_parent_status"]=="complete" and m["parent"]["parent_manifest_sha256"]==parent_sha
  and w.immutable_pins(m["source_pins"])==w.immutable_pins(wrapper_pins) and m["conditional_kernel"]==w.kernel.metadata()
  and m["coordinate_decode_after_global_refresh"] is False,"only corrected CLOSED production G3 family accepted")
 root=Path(__file__).resolve().parents[1]
 w.verify_git_sources(root,m["source_pins"]["git_revision"],{
  "scripts/refresh_prior_corrected_production_v1.py":m["source_pins"]["wrapper"],
  "src/posteriorforge/performance_prior_physical_fifo_v1.py":w.PRODUCER_SHA256,
  "src/posteriorforge/execution_physical_fifo_v1.py":w.F64_HELPER_SHA256,
  "src/posteriorforge/conditional_globals.py":w.KERNEL_SHA256,
  "scripts/assess_prior_forensic_replay_v1.py":w.DRAW_OBSERVER_SOURCE_SHA256})
 p=Path(path).with_name(m["samples_file"]);z=Path(path).with_name(m["actual_draws_file"])
 require(sha256_file(p)==m["samples_sha256"] and sha256_file(z)==m["actual_draws_sha256"],"saved production physical/draw matrices differ")
 with np.load(p,allow_pickle=False) as saved:
  require(saved.files==["samples"],"physical sample keys differ");after=saved["samples"]
 with np.load(z,allow_pickle=False) as saved:
  require(saved.files==["standard_normal_draws"],"actual normal matrix keys differ");normals=saved["standard_normal_draws"]
 require(before.dtype==after.dtype==normals.dtype==np.float64 and before.shape==after.shape==(COUNT,21)
  and normals.shape==(COUNT,3) and array_descriptor(after)==m["samples"],"strict50000F64 physical output and150k actual normals required")
 r=m["refresh"];seed=derive_seed(20261111,w.ROLE,case=2,repetition=0)
 require(r["seed"]==seed and r["role"]==w.ROLE and r["bit_generator"]=="PCG64"
  and r["initial_private_state"]==np.random.PCG64(seed).state and r["standard_normal_method_calls"]==1
  and r["standard_normal_draw_shape"]==[COUNT,3] and r["standard_normal_scalar_count"]==COUNT*3
  and r["global_block_refresh_count_per_row"]==1 and r["draws_are_actual_returned_matrix_not_rerolled"]
  and r["default_numpy_python_RNG_preserved"] and r["default_numpy_python_RNG_before_sha256"]==r["default_numpy_python_RNG_after_sha256"]
  and r["all18_physical_columns_byte_identical"] and r["input_array_unchanged"]
  and r["coordinate_decode_after_global_refresh"] is False,"single original physical G3 policy differs")
 require(array_descriptor(normals)==r["standard_normal_draws"],"actual normal matrix descriptor differs")
 for name in ("initial_private_state","final_private_state"):
  require(hashlib.sha256(json.dumps(r[name],sort_keys=True,separators=(",",":")).encode()).hexdigest()==r[name+"_sha256"],"private-state hash differs")
 mean,cov=w.kernel.conditional_parameters(before)
 require(after[:,:3].tobytes()==(mean+normals@np.linalg.cholesky(cov).T).tobytes()
  and after[:,3:].tobytes()==before[:,3:].tobytes(),"saved actual Z/G3 reconstruction or physical18 identity differs")
 return m,after

def run(*,root,parent_manifest,parent_manifest_sha256,hybrid_manifest,hybrid_manifest_sha256,output):
 w,producer,sources=source_guard()
 parent,before,auth=w.load_parent(parent_manifest,parent_manifest_sha256,scratch=root)
 wrapper_pins=w.source_guard()[2]
 hybrid,after=authenticate_hybrid(hybrid_manifest,hybrid_manifest_sha256,parent_manifest_sha256,before,w,wrapper_pins)
 old=w.load("evaluate_prior_coordinates_pilot_v1.py",w.OLD_EVAL_SHA256)
 frozen=old.load("evaluate_ss15_sigma_native_campaign.py",old.FROZEN_EVALUATOR_SHA256)
 binder=frozen.load("evaluate_cdf_pilot.py",frozen.BINDER_SHA);pure=frozen.load("evaluate_ss15_pilot.py",frozen.PURE_HELPER_SHA)
 results=Path(root)/"results";gatepath=results/"gate-final.json"
 require(sha256_file(gatepath)==binder.BASELINE_GATE_SHA256,"original physical gate changed")
 gate=json.loads(gatepath.read_bytes());query=load_heldout(results/"banks-v1",2).obs.reshape(1,33)
 arrays,reference,assessments,controls=frozen.reference_and_assess(results,gate,parent,query,before,after,binder)
 baseline,baseline_assessment,baseline_custody=old.load_baseline(frozen,
  results/"accuracy-ss15-v1/sigma-native-grid-completion-01/completion.json",
  results/"accuracy-ss15-v1/sigma-native-grid-01/seed-20261111-case-2/manifest.json",
  results/"accuracy-ss15-v1/sigma-native-grid-refresh-01/seed-20261111-case-2/manifest.json",
  results/"accuracy-ss15-v1/sigma-native-grid-assessment-01/seed-20261111-case-2/assessment.json",parent["raw_scientific_bindings"])
 out=Path(output);out.mkdir(parents=True,exist_ok=False)
 record={"schema":SCHEMA,"status":assessments["after"]["status"],"passed":assessments["after"]["passed"],
  "attribution":"Built with PriorLabs-TabPFN","context_seed":20261111,"case":2,
  "source_pins":sources,"parent_manifest_sha256":parent_manifest_sha256,"hybrid_manifest_sha256":hybrid_manifest_sha256,
  "authentication":auth,"closed_production_parent":True,"forensic_cohort_used":False,
  "absolute_physical_assessments":assessments,"summary":{k:pure.summary(v,parent["parameter_order"]) for k,v in assessments.items()},
  "unchanged18_control":controls,"reference":reference,"original_gate_sha256":sha256_file(gatepath),
  "all50000_rows_used":True,"scientific_limits_unchanged":True,"focus_indices":list(FOCUS),
  "old_baseline_summary":baseline_assessment["summary"],"old_baseline_custody":baseline_custody,
  "configuration_comparison":{"coordinates_precision_routing_all_changed":True,"isolated_coordinate_effect_claimed":False},
  "physical_G3_policy":"ONE unchanged physical three-global conditional; actualZsaved; physical18bytesunchanged",
  "new_model_GPU_draws_or_reference_integrations_during_assessment":0,"plots_pending":True}
 write_json(out/"assessment.json",record)
 print(json.dumps({"corrected_production_scalar_output":str(out),"status":record["status"],"passed":record["passed"],"summary":record["summary"]},sort_keys=True),flush=True)
 record["figures"]=plots(out,baseline,before,after,arrays,assessments,parent["parameter_order"],binder)
 record["plots_pending"]=False
 require(sha256_file(parent_manifest)==parent_manifest_sha256 and sha256_file(hybrid_manifest)==hybrid_manifest_sha256
  and sha256_file(gatepath)==binder.BASELINE_GATE_SHA256,"closed production/G3/gate inputs changed")
 write_json(out/"assessment.json",record);write_json(out/"checksums.json",{p.name:sha256_file(p) for p in out.iterdir() if p.is_file()})
 return record

def plots(out,baseline,before,after,arrays,assessments,order,binder):
 import matplotlib
 matplotlib.use("Agg")
 import matplotlib.pyplot as plt
 from matplotlib.lines import Line2D
 legend=[Line2D([0],[0],color=c,lw=1.8,label=l) for c,l in zip(COLORS,LABELS)]
 def figure(rows,cols,size,title):
  f,ax=plt.subplots(rows,cols,figsize=size,squeeze=False)
  f.suptitle(title+"\nClosed production parent: FP32 / 19 cached stages / all8 original at6+10; physical G3 is separate",fontsize=13,y=.995)
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
 figs={};f,axes=figure(2,2,(12,8.6),"Corrected production prior-coordinate cohorts: four predeclared physical coordinates")
 f.subplots_adjust(top=.81,bottom=.11,hspace=.5,wspace=.27)
 for ax,j in zip(axes,FOCUS):density(ax,j)
 figs["focus_density"]=save(f,"production-density-focus.png")
 print(json.dumps({"first_focus_density_ready":str(out/"production-density-focus.png")}),flush=True)
 f,axes=figure(7,3,(14,20),"Corrected production prior-coordinate cohorts: all21 physical densities")
 for ax,j in zip(axes,range(21)):density(ax,j)
 figs["density"]=save(f,"production-density.png")
 precision=assessments["after"]["reference_qualification"]["precision"]
 for label,indices,shape,size,title in (("marginal_cdf",range(21),(7,3),(14,20),"Corrected production all21 physical marginal CDFs"),
  ("joint_cdf",range(21,53),(8,4),(17,23),"Corrected production all32 fixed physical projection CDFs")):
  f,axes=figure(*shape,size,title)
  for ax,j in zip(axes,indices):
   x,cdf,lo,hi=domain(j);grid=np.linspace(lo,hi,700)
   for a,color in ((baseline,COLORS[0]),(before,COLORS[1]),(after,COLORS[2])):
    values=np.sort(a[:,j] if j<21 else a@arrays["directions"][j]);ax.plot(grid,np.searchsorted(values,grid,side="right")/len(values),color=color,lw=1.1)
   ax.plot(x,cdf,color=COLORS[3],lw=1.3);ax.fill_between(x,np.maximum(0,cdf-precision),np.minimum(1,cdf+precision),color=COLORS[3],alpha=.12)
   key="marginal_cdf" if j<21 else "joint_projection_cdf";index=j if j<21 else j-21
   a=assessments["before"][key]["upper"][index];b=assessments["after"][key]["upper"][index];name=order[j] if j<21 else f"fixed joint {j-20}"
   ax.set(title=f"{name}: {a:.3f} → {b:.3f}",xlim=(lo,hi),ylim=(-.02,1.02),xlabel="physical coordinate" if j<21 else "physical-vector projection",ylabel="CDF")
  figs[label]=save(f,"production-"+label.replace("_","-")+".png")
 return figs

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ("root","parent-manifest","parent-manifest-sha256","hybrid-manifest","hybrid-manifest-sha256","output"):
  p.add_argument("--"+name,required=True)
 run(**vars(p.parse_args()))
if __name__=="__main__":main()
