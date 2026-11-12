#!/usr/bin/env python3
"""One joint-global hybrid update assessed on saved pure-SS15 cohorts.

Built with PriorLabs-TabPFN. Original reference/gate/all50000 rows; no model,
GPU, reference integration, draw, clipping or row filtering occurs here.
"""
from __future__ import annotations
import argparse,hashlib,importlib.util,json
from pathlib import Path
import numpy as np
from posteriorforge import conditional_globals as kernel
from posteriorforge.artifacts import sha256_file,write_json
from posteriorforge.banks import derive_seed,load_heldout
from posteriorforge.inference import support_mask
from posteriorforge.validation import assess_samples

SCHEMA="posteriorforge.ss15_sigma_first_global_block_assessment.v1"
PRODUCER_SHA="5051ef3f9bb30f456877ad2dcb78181c9a4b7249c90544d6d0bcffbcebe14dc0"
KERNEL_SHA="e346e25bc9825abf016c2bd75addaf2b025d3e3d94e1a0ea6f6e31dd02997a43"
PURE_HELPER_SHA="3156d54b7d29f7aec29c2846782729a8c73da3bfcda16bb2e03cec3084899552"
FOCUS=(0,4,11,12)
COLORS=("#777777","#1479b8","#111111")
CASES={"sigma":{"hybrid_dir":"global-block-sigma-first-01",
 "hybrid_sha":"3bc7da5fb333186764c3312f6ac5b6cd88744f554b9ddd57b3cff9ee9227490e",
 "payload_sha":"9751a1c32f43ebc7b94ac9b3605720777079791aa954e574cb55eebdf5f3f669",
 "parent_dir":"sigma-first-pilot-01","parent_sha":"9f2d79d8c40d1022e0c6508fb111b80dae499337ea79fc1373e5ec7d3291b137",
 "labels":("pure sigma-first SS15","sigma-first SS15 + joint-global Gibbs","qualified reference")}}

def require(value,message):
 if not value:raise ValueError(message)

def load(name,digest):
 p=Path(__file__).with_name(name);require(sha256_file(p)==digest,"frozen source differs: "+name)
 s=importlib.util.spec_from_file_location("_joint_g_assess_"+p.stem,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def local18_equal(before,after):
 return before.dtype==after.dtype==np.float64 and before.shape==after.shape==(50000,21) and before[:,3:].tobytes()==after[:,3:].tobytes()

def unchanged_marginals(before,after):
 for key in ("lower","upper"):require(before["marginal_cdf"][key][3:]==after["marginal_cdf"][key][3:],"18 unchanged marginal CDF bounds differ")
 for key in ("sample_mean","sample_sd","absolute_bias_reference_sd","sd_ratio"):
  require(before["moments"][key][3:]==after["moments"][key][3:],"18 unchanged marginal moments differ")
 return {"passed":True,"all18_local_noise_columns_and_marginal_assessments_unchanged":True}

def authenticate(phase,family):
 spec=CASES[family];path=phase/spec["hybrid_dir"]/"manifest.json"
 require(sha256_file(path)==spec["hybrid_sha"],"exact actual hybrid manifest differs")
 m=json.loads(path.read_bytes());producer=load("refresh_ss15_noise_first_global_block.py",PRODUCER_SHA)
 require(kernel.source_sha256()==KERNEL_SHA and m["conditional_kernel"]==kernel.metadata(),"actual joint conditional source/geometry differs")
 pins=m["source_pins"];repo=Path(__file__).resolve().parents[1]
 require(pins["producer_sha256"]==PRODUCER_SHA and pins["kernel_sha256"]==KERNEL_SHA
  and pins["frozen_joint_refresh_sha256"]==producer.BACKBONE_SHA
  and pins["sigma_first_contract_sha256"]==producer.CONTRACT_SHA
  and pins["producer_backbone_kernel_and_contract_committed_before_run"] is True,
  "committed wrapper/backbone/kernel/preregistered rule differs")
 for revision,name,digest in ((pins["git_revision"],"scripts/refresh_ss15_noise_first_global_block.py",PRODUCER_SHA),
  (pins["git_revision"],"scripts/refresh_ss15_global_block.py",producer.BACKBONE_SHA),
  (pins["git_revision"],"src/posteriorforge/conditional_globals.py",KERNEL_SHA),
  (pins["sigma_first_contract_origin_commit"],producer.CONTRACT_PATH,producer.CONTRACT_SHA),
  (pins["joint_source"]["math_contract_origin_commit"],"docs/accuracy-ss15-graph-local-v1.md",producer.backbone.CONTRACT_SHA)):
  require(hashlib.sha256(producer.backbone.git_bytes(repo,revision,name)).hexdigest()==digest,"recorded committed bytes differ")
 parentpath=phase/spec["parent_dir"]/"manifest.json"
 parent,before,lineage=producer.load_parent(parentpath,spec["parent_sha"],kernel.metadata())
 require(m["schema"]==producer.SCHEMA and m["variant"]==producer.VARIANT and m["status"]=="complete"
  and m["requested_count"]==50000 and m["case"]==0 and m["context_seed"]==20261111 and m["parent"]==lineage
  and m["raw_scientific_bindings"]==parent["raw_scientific_bindings"] and m["sources_unchanged"] is True
  and m["support_filtering_clipping_redraw_or_trimming"] is False,"explicit one-pass hybrid lineage/policy differs")
 payload=path.with_name("samples.npz")
 require(m["samples_file"]=="samples.npz" and sha256_file(payload)==m["samples_sha256"]==spec["payload_sha"],"actual hybrid payload differs")
 with np.load(payload,allow_pickle=False) as a:
  require(a.files==["samples"],"complete physical matrix only");after=a["samples"].copy()
 require(producer.array_descriptor(after)==m["samples"] and local18_equal(before,after)
  and support_mask(after).all(),"50000 supported Float64/all18 byte-control differs")
 r=m["refresh"];seed=derive_seed(20261111,producer.ROLE,case=0,repetition=0)
 rng=np.random.Generator(np.random.PCG64(seed))
 require(r["role"]==producer.ROLE and r["seed"]==seed and r["standard_normal_draw_shape"]==[50000,3]
  and r["standard_normal_draw_count"]==150000 and r["global_block_refresh_count_per_row"]==1
  and rng.bit_generator.state==r["initial_private_state"],"one private correlated-global draw/row policy differs")
 normals=rng.standard_normal((50000,3));mean,covariance=kernel.conditional_parameters(before)
 expected=mean+normals@np.linalg.cholesky(covariance).T
 require(producer.array_descriptor(normals)==r["standard_normal_draws"]
  and rng.bit_generator.state==r["final_private_state"] and after[:,:3].tobytes()==expected.tobytes(),
  "actual whole-matrix private normal/conditional draw replay differs")
 d=m["default_RNG_observation"]
 require(d["preserved"] is True and d["before_sha256"]==d["after_sha256"]
  and d["torch_CPU_CUDA_RNG_hashes_observed"] is False
  and r["default_numpy_python_RNG_preserved"] is True
  and r["default_numpy_python_RNG_before_sha256"]==r["default_numpy_python_RNG_after_sha256"],
  "default NumPy/Python RNG preservation scope differs")
 return m,parent,before,after,{"passed":True,"one_joint_global_draw_per_row":True,"private_matrix_normal_replay":True,
  "all18_columns_byte_identical":True,"posterior_accuracy_guaranteed":False,"pure_NPE":False}

def plots(output,before,after,arrays,assessments,order,binder,labels):
 import matplotlib
 matplotlib.use("Agg")
 import matplotlib.pyplot as plt
 from matplotlib.lines import Line2D
 output=Path(output);legend=[Line2D([0],[0],color=c,lw=1.8,label=l) for c,l in zip(COLORS,labels)]
 footer="Built with PriorLabs-TabPFN\nReference-selected display; all50000 rows normalize each histogram/CDF; unchanged whole-line gate."
 def figure(rows,cols,size,title):
  f,ax=plt.subplots(rows,cols,figsize=size,squeeze=False)
  f.suptitle(title+"\nseed20261111, case0; one predeclared correlated three-global update",fontsize=13,y=.995)
  f.legend(handles=legend,loc="upper center",bbox_to_anchor=(.5,.955),ncol=3,fontsize=9)
  f.text(.5,.012,footer,ha="center",va="bottom",fontsize=8)
  f.subplots_adjust(left=.07,right=.99,bottom=.07,top=.89,wspace=.28,hspace=.55)
  return f,ax.flat
 def domain(j):
  x=arrays["queries"]/arrays["directions"][j,j] if j<21 else arrays["queries"]
  cdf=arrays["cdfs"][j];lo=binder.reference_quantile(x,cdf,.001);hi=binder.reference_quantile(x,cdf,.999)
  pad=.03*(hi-lo);return x,cdf,lo-pad,hi+pad
 def density(ax,j):
  x,cdf,lo,hi=domain(j);bins=np.linspace(lo,hi,41);width=np.diff(bins)
  ax.stairs(np.diff(np.interp(bins,x,cdf))/width,bins,color=COLORS[2],lw=1.4)
  for a,color in ((before,COLORS[0]),(after,COLORS[1])):
   count,_=np.histogram(a[:,j],bins);ax.stairs(count/(len(a)*width),bins,color=color,lw=1.1)
  old=assessments["before"]["marginal_cdf"]["upper"][j];new=assessments["after"]["marginal_cdf"]["upper"][j]
  ax.set(title=f"{order[j]}: CDF upper {old:.3f} → {new:.3f}",xlim=(lo,hi),xlabel="physical coordinate",ylabel="binned density")
 def save(f,name):
  path=output/name;f.savefig(path,dpi=155);plt.close(f)
  return {"file":name,"sha256":sha256_file(path)}
 figs={}
 focus,axes=figure(2,2,(12,8.6),"Joint-global hybrid: four predeclared SS15 coordinates")
 focus.subplots_adjust(top=.81,bottom=.11,hspace=.5,wspace=.27)
 for ax,j in zip(axes,FOCUS):density(ax,j)
 figs["focus_density"]=save(focus,"joint-global-density-focus.png")
 print(json.dumps({"first_focus_density_ready":str(output/figs["focus_density"]["file"])}),flush=True)
 densityfig,axes=figure(7,3,(14,20),"One correlated three-global refresh: all21 densities")
 for ax,j in zip(axes,range(21)):density(ax,j)
 figs["density"]=save(densityfig,"joint-global-density.png")
 precision=assessments["after"]["reference_qualification"]["precision"]
 for label,indices,shape,size,title in (("marginal_cdf",range(21),(7,3),(14,20),"All21 physical marginal CDFs"),
    ("joint_cdf",range(21,53),(8,4),(17,23),"All32 original fixed full-vector projection CDFs")):
  f,axes=figure(*shape,size,title)
  for ax,j in zip(axes,indices):
   x,cdf,lo,hi=domain(j);grid=np.linspace(lo,hi,700)
   for a,color in ((before,COLORS[0]),(after,COLORS[1])):
    values=a[:,j] if j<21 else a@arrays["directions"][j];values=np.sort(values)
    ax.plot(grid,np.searchsorted(values,grid,side="right")/len(values),color=color,lw=1.1)
   ax.plot(x,cdf,color=COLORS[2],lw=1.3);ax.fill_between(x,np.maximum(0,cdf-precision),np.minimum(1,cdf+precision),color=COLORS[2],alpha=.12)
   key="marginal_cdf" if j<21 else "joint_projection_cdf";index=j if j<21 else j-21
   old=assessments["before"][key]["upper"][index];new=assessments["after"][key]["upper"][index]
   name=order[j] if j<21 else f"fixed joint {j-20}"
   ax.set(title=f"{name}: {old:.3f} → {new:.3f}",xlim=(lo,hi),ylim=(-.02,1.02),xlabel="physical coordinate" if j<21 else "physical-vector projection",ylabel="CDF")
  figs[label]=save(f,"joint-global-"+label.replace("_","-")+".png")
 return figs,{"indices":list(FOCUS),"rule":"predeclared original SS15 coordinates; identical focus for both parent families","hybrid_used_for_selection":False,
   "histogram_bins":40,"normalization_rows":50000,"display_range":"saved reference0.001..0.999 plus3%; no assessment truncation"}


def run(*,root,output,family):
 root=Path(root);results=root/"results";phase=results/"accuracy-ss15-v1";spec=CASES[family]
 m,parent,before,after,auth=authenticate(phase,family)
 pure=load("evaluate_ss15_pilot.py",PURE_HELPER_SHA);binder=pure.load("evaluate_cdf_pilot.py",pure.BINDER_SHA)
 gatepath=results/"gate-final.json";require(sha256_file(gatepath)==binder.BASELINE_GATE_SHA256,"immutable original gate differs")
 gate=json.loads(gatepath.read_bytes());query=load_heldout(results/"banks-v1",0).obs.reshape(1,33)
 arrays,_,reference,custody=binder.reference_for_case(results,gate,query,0)
 binder.validate_run_reference_binding(parent,custody)
 assessments={"before":assess_samples(before,reference,support_mask=support_mask(before)),
  "after":assess_samples(after,reference,support_mask=support_mask(after))}
 control=unchanged_marginals(assessments["before"],assessments["after"])
 original=[r for r in gate["runs"] if r["context_seed"]==20261111 and r["case"]==0]
 require(len(original)==1,"original case/seed binding differs")
 for a in assessments.values():require(a["reference_qualified"] and a["integrity"]["passed"]
  and a["limits"]==original[0]["assessment"]["limits"],"original reference/integrity/physical criteria differ")
 summaries={k:pure.summary(v,m["parameter_order"]) for k,v in assessments.items()}
 output=Path(output);output.mkdir(parents=True,exist_ok=False)
 report={"schema":SCHEMA,"attribution":"Built with PriorLabs-TabPFN","family":family,"evaluator_sha256":sha256_file(__file__),
  "status":assessments["after"]["status"],"passed":assessments["after"]["passed"],"pure_NPE_claimed":False,
  "method":"explicit hybrid NPE-PFN/Gibbs: one prior-derived correlated3-global full-conditional block",
  "summary":summaries,"absolute_physical_assessments":assessments,"unchanged18_control":control,
  "authentication":auth,"reference":custody,"hybrid_manifest_sha256":spec["hybrid_sha"],
  "hybrid_samples_sha256":spec["payload_sha"],"parent_manifest_sha256":spec["parent_sha"],
  "parent_samples_sha256":parent["samples_sha256"],"accuracy_gate_required":True,"all50000_rows_used":True,
  "scientific_limits_unchanged":True,"no_model_GPU_or_reference_integrations":True,
  "ordered_matching_comparison_scope":"pending separate fresh all21 screen; hybrid does not itself prove ordered neural execution fidelity",
  "original_gate_sha256":sha256_file(gatepath),"plots_pending":True}
 write_json(output/"assessment.json",report);print(json.dumps({"family":family,"output":str(output),"summary":summaries},indent=2),flush=True)
 report["figures"],report["focus_selection"]=plots(output,before,after,arrays,assessments,m["parameter_order"],binder,spec["labels"])
 report["plots_pending"]=False
 require(sha256_file(phase/spec["hybrid_dir"]/"manifest.json")==spec["hybrid_sha"]
  and sha256_file(gatepath)==binder.BASELINE_GATE_SHA256,"input/gate changed during assessment")
 write_json(output/"assessment.json",report)
 write_json(output/"checksums.json",{p.name:sha256_file(p) for p in output.iterdir() if p.is_file()})
 return report

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 for name in ("root","output"):parser.add_argument("--"+name,required=True)
 parser.add_argument("--family",choices=tuple(CASES),required=True)
 run(**vars(parser.parse_args()))
if __name__=="__main__":main()
