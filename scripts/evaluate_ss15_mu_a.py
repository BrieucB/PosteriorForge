#!/usr/bin/env python3
"""Saved-cohort hybrid evaluation and ordinary physical posterior plots.

Built with PriorLabs-TabPFN. One scalar prior-derived Gibbs update is explicitly
hybrid inference. Original50k rows/all physical gate criteria/reference remain.
"""
from __future__ import annotations
import argparse,copy,hashlib,importlib.util,json,re,subprocess
from pathlib import Path
import numpy as np
from posteriorforge.artifacts import sha256_file,write_json
from posteriorforge.banks import load_heldout,derive_seed
from posteriorforge.inference import support_mask
from posteriorforge.validation import assess_samples
from posteriorforge import conditional_mu_a as kernel

SCHEMA="posteriorforge.ss15_mu_a_assessment.v1"
PRODUCER_SHA="5c1997f2b5e830f88aca1e29a4871c47c7b51940740e866acc7c8801e8a42feb"
KERNEL_SHA="850df8d6498f78c12500ffdf60c4c9126f5c1ab67bc29867002d4c991373d2e8"
PURE_EVALUATOR_SHA="3156d54b7d29f7aec29c2846782729a8c73da3bfcda16bb2e03cec3084899552"
PURE_ASSESSMENT_SHA="e657b4a3bbd0c731ee8d0944fb9468e016ceeb7abd55f4820a3e5a8bb2700b15"
PARENT_MANIFEST_SHA="28da396e0af6cf26009b4eb8feab8b75ed7a1d7a5add30440a35b38539da4f94"
PARENT_PAYLOAD_SHA="f263f712d943d7916205b2f2b77f9243ac4ba3819f5f739c45f00710d5b4be47"
HYBRID_MANIFEST_SHA="0eecafd7775ad1b5f7d3156cdcd66f57b85aecf23a17a4d92bb5ab88b4800db3"
HYBRID_PAYLOAD_SHA="90d4813ba5d1d1b8fa5158b5ba92268cbeda57c4d83e951984f7407134274f28"
FOCUS=(0,4,11,12)
LABELS=("pure SS15 FP16","SS15 + one exact mu_a refresh","qualified reference")
COLORS=("#777777","#1479b8","#111111")

def require(value,message):
 if not value:raise ValueError(message)
def load(name,digest):
 p=Path(__file__).with_name(name);require(sha256_file(p)==digest,"frozen source changed: "+name)
 s=importlib.util.spec_from_file_location("_mu_a_assess_"+p.stem,p)
 m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def other20_equal(before,after):
 return before.dtype==after.dtype==np.float64 and before.shape==after.shape==(50000,21) and before[:,1:].tobytes()==after[:,1:].tobytes()

def authenticate(manifest_path,parent_path):
 require(sha256_file(manifest_path)==HYBRID_MANIFEST_SHA,"actual frozen hybrid manifest changed")
 require(sha256_file(parent_path)==PARENT_MANIFEST_SHA,"actual frozen pure parent changed")
 m=json.loads(Path(manifest_path).read_bytes());p=json.loads(Path(parent_path).read_bytes())
 producer=load("refresh_ss15_mu_a.py",PRODUCER_SHA)
 require(kernel.source_sha256()==KERNEL_SHA and m["conditional_kernel"]==kernel.metadata(),"prior-derived conditional source/geometry differs")
 require(m["schema"]==producer.SCHEMA and m["variant"]==producer.VARIANT and m["status"]=="complete"
   and m["requested_count"]==50000 and m["case"]==0 and m["context_seed"]==20261111
   and m["sources_unchanged"] is True and m["support_filtering_clipping_redraw_or_trimming"] is False,
   "explicit complete single-refresh family required")
 pins=m["source_pins"];repo=Path(__file__).resolve().parents[1]
 require(pins["producer_sha256"]==PRODUCER_SHA and pins["kernel_sha256"]==KERNEL_SHA
   and pins["math_contract_sha256"]==producer.CONTRACT_SHA and pins["producer_committed_before_run"] is True,
   "frozen producer/kernel/predeclared mathematical contract differs")
 for revision,name,digest in ((pins["git_revision"],"scripts/refresh_ss15_mu_a.py",PRODUCER_SHA),
   (pins["math_contract_origin_commit"],"docs/accuracy-ss15-mu-a-gibbs-v1.md",producer.CONTRACT_SHA)):
  require(hashlib.sha256(producer.git_bytes(repo,revision,name)).hexdigest()==digest,"committed source/contract bytes differ")
 parent,values,lineage=producer.load_parent(parent_path,kernel.metadata())
 require(m["parent"]==lineage and m["parent"]["samples_sha256"]==PARENT_PAYLOAD_SHA,
   "exact pure-SS15 parent lineage differs")
 payload=Path(manifest_path).with_name("samples.npz")
 require(m["samples_file"]=="samples.npz" and sha256_file(payload)==m["samples_sha256"]==HYBRID_PAYLOAD_SHA,"actual hybrid payload differs")
 with np.load(payload,allow_pickle=False) as a:
  require(a.files==["samples"],"only complete physical sample matrix expected");hybrid=a["samples"].copy()
 require(producer.array_descriptor(hybrid)==m["samples"] and other20_equal(values,hybrid),"all20 unchanged physical columns must remain byte-identical")
 require(support_mask(hybrid).all() and support_mask(values).all(),"exact50000 complete supported rows required")
 require(m["raw_scientific_bindings"]==parent["raw_scientific_bindings"],"inherited raw data/reference provenance differs")
 r=m["refresh"];seed=derive_seed(20261111,producer.ROLE,case=0,repetition=0)
 require(r["seed"]==seed and r["role"]==producer.ROLE and r["refresh_count_per_row"]==1
   and r["standard_normal_draw_count"]==50000 and r["other20_columns_byte_identical"] is True
   and r["input_array_unchanged"] is True,"single predeclared private draw/row receipt differs")
 rng=np.random.Generator(np.random.PCG64(seed))
 require(rng.bit_generator.state==r["initial_private_state"],"initial private PCG64 state differs")
 z=rng.standard_normal(50000);mean,var=kernel.conditional_parameters(values)
 require(producer.array_descriptor(z)==r["standard_normal_draws"] and rng.bit_generator.state==r["final_private_state"],
   "independent one-pass standard-normal/state replay differs")
 require(np.array_equal(hybrid[:,0],mean+np.sqrt(var)*z),"actual mu_a is not the prescribed prior conditional draw")
 default=m["default_RNG_observation"]
 require(default["preserved"] is True and default["before_sha256"]==default["after_sha256"]
   and default["torch_CPU_CUDA_RNG_hashes_observed"] is False,"default NumPy/Python preservation scope differs")
 return m,parent,values,hybrid,{"passed":True,"other20_byte_identical":True,"one_pass_private_normal_replay":True,
   "recorded_source_commit":pins["git_revision"],"math_contract_origin":pins["math_contract_origin_commit"],
   "no_new_neural_or_GPU_calls":True,"only_changed_coordinate":"mu_a","exactness":"real-arithmetic full conditional; float64 execution"}

def unchanged_marginals(before,after):
 for label in ("marginal_cdf",):
  for key in ("lower","upper"):require(before[label][key][1:]==after[label][key][1:],"unchanged20 marginal CDF bounds differ")
 for key in ("sample_mean","sample_sd","absolute_bias_reference_sd","sd_ratio"):
  require(before["moments"][key][1:]==after["moments"][key][1:],"unchanged20 marginal moments differ")
 return {"passed":True,"unchanged_physical_indices":list(range(1,21)),
   "all_other_marginal_CDF_bounds_and_moments_identical":True}

def plots(output,before,after,arrays,assessments,order,binder):
 import matplotlib
 matplotlib.use("Agg")
 import matplotlib.pyplot as plt
 from matplotlib.lines import Line2D
 output=Path(output);legend=[Line2D([0],[0],color=c,lw=1.8,label=l) for c,l in zip(COLORS,LABELS)]
 footer="Built with PriorLabs-TabPFN\nReference-selected display; all50000 rows normalize each histogram/CDF; unchanged whole-line gate."
 def figure(rows,cols,size,title):
  f,ax=plt.subplots(rows,cols,figsize=size,squeeze=False)
  f.suptitle(title+"\nseed20261111, case0; one predeclared scalar update",fontsize=13,y=.995)
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
 focus,axes=figure(2,2,(12,8.6),"Hybrid check: four predeclared pure-SS15 failure coordinates")
 focus.subplots_adjust(top=.81,bottom=.11,hspace=.5,wspace=.27)
 for ax,j in zip(axes,FOCUS):density(ax,j)
 figs["focus_density"]=save(focus,"mu-a-refresh-density-focus.png")
 print(json.dumps({"first_focus_density_ready":str(output/figs["focus_density"]["file"])}),flush=True)
 densityfig,axes=figure(7,3,(14,20),"Pure SS15 versus one mu_a conditional refresh: all21 densities")
 for ax,j in zip(axes,range(21)):density(ax,j)
 figs["density"]=save(densityfig,"mu-a-refresh-density.png")
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
  figs[label]=save(f,"mu-a-refresh-"+label.replace("_","-")+".png")
 return figs,{"indices":list(FOCUS),"rule":"predeclared current pureSS15 failures before hybrid cohort","hybrid_used_for_selection":False,
   "histogram_bins":40,"normalization_rows":50000,"display_range":"saved reference0.001..0.999 plus3%; no assessment truncation"}

def run(*,root,manifest,output):
 root=Path(root);results=root/"results";phase=results/"accuracy-ss15-v1";manifest=Path(manifest)
 pure=load("evaluate_ss15_pilot.py",PURE_EVALUATOR_SHA);binder=pure.load("evaluate_cdf_pilot.py",pure.BINDER_SHA)
 baseline_path=phase/"pilot-01/seed-20261111-case-0/manifest.json"
 m,parent,before,after,auth=authenticate(manifest,baseline_path)
 pure_path=phase/"pilot-assessment-final/assessment.json"
 require(sha256_file(pure_path)==PURE_ASSESSMENT_SHA,"actual canonical pure assessment changed")
 previous=json.loads(pure_path.read_bytes());gatepath=results/"gate-final.json"
 require(sha256_file(gatepath)==binder.BASELINE_GATE_SHA256,"immutable original gate changed")
 gate=json.loads(gatepath.read_bytes());query=load_heldout(results/"banks-v1",0).obs
 arrays,_,reference,custody=binder.reference_for_case(results,gate,query,0)
 require(custody==previous["reference"],"same saved reference custody required")
 binder.validate_run_reference_binding(parent,custody)
 assessments={"before":assess_samples(before,reference,support_mask=support_mask(before)),
   "after":assess_samples(after,reference,support_mask=support_mask(after))}
 require(assessments["before"]["reference_qualified"] and assessments["after"]["reference_qualified"]
   and assessments["before"]["integrity"]["passed"] and assessments["after"]["integrity"]["passed"],"qualified reference/exact50000 support required")
 control=unchanged_marginals(assessments["before"],assessments["after"])
 require(assessments["after"]["limits"]==previous["absolute_physical_assessment"]["limits"],"original scientific thresholds changed")
 order=m["parameter_order"];summary={k:pure.summary(v,order) for k,v in assessments.items()}
 output=Path(output);output.mkdir(parents=True,exist_ok=False)
 report={"schema":SCHEMA,"attribution":"Built with PriorLabs-TabPFN","status":assessments["after"]["status"],
   "passed":assessments["after"]["passed"],"method":"explicit hybrid NPE-PFN/Gibbs: one mu_a full-conditional draw",
   "pure_NPE_claimed":False,"accuracy_gate_required":True,"all50000_rows_used":True,
   "summary":summary,"absolute_physical_assessments":assessments,"unchanged20_control":control,"authentication":auth,
   "reference":custody,"parent_samples_sha256":PARENT_PAYLOAD_SHA,"hybrid_samples_sha256":HYBRID_PAYLOAD_SHA,
   "hybrid_manifest_sha256":HYBRID_MANIFEST_SHA,"pure_assessment_sha256":PURE_ASSESSMENT_SHA,
   "sources":{"evaluator":sha256_file(__file__),"producer":PRODUCER_SHA,"kernel":KERNEL_SHA,"pure_evaluator":PURE_EVALUATOR_SHA},
   "no_model_GPU_or_reference_integrations":True,"original_gate_sha256":sha256_file(gatepath),"plots_pending":True}
 write_json(output/"assessment.json",report)
 print(json.dumps({"output":str(output),"summary":summary,"unchanged20_control":control},indent=2),flush=True)
 report["figures"],report["focus_selection"]=plots(output,before,after,arrays,assessments,order,binder)
 report["plots_pending"]=False;write_json(output/"assessment.json",report)
 write_json(output/"checksums.json",{p.name:sha256_file(p) for p in output.iterdir() if p.is_file()})
 return report

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ("root","manifest","output"):p.add_argument("--"+name,required=True)
 run(**vars(p.parse_args()))
if __name__=="__main__":main()
