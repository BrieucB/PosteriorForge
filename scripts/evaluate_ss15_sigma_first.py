#!/usr/bin/env python3
"""Saved first sigma-first cohort comparison. Built with PriorLabs-TabPFN.

Original physical gate/reference/all50000 rows; separate masked conditional
comparison may still be pending. This is an intentionally changed learned input.
"""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,subprocess
from collections import Counter
from pathlib import Path
import numpy as np
from posteriorforge import performance_ss15_sigma_first as producer
from posteriorforge.artifacts import sha256_file,write_json
from posteriorforge.banks import load_context,load_heldout
from posteriorforge.inference import support_mask
from posteriorforge.validation import assess_samples

SCHEMA="posteriorforge.ss15_sigma_first_assessment.v1"
PRODUCER_SHA="128c200a82f1252b2ad971ed43f10f534b16b9e02f8882cfd62d222cb88442cd"
PURE_HELPER_SHA="3156d54b7d29f7aec29c2846782729a8c73da3bfcda16bb2e03cec3084899552"
PARENT_SHA="dbebbf30183eee828f6dab3afd21bccc80654ef18b05d8430bda9b3e63a10dfd"
PARENT_PAYLOAD_SHA="7f2290e77e92585850b9856d99675c4f2afe9be7af77a66d87c6bc80da125364"
FOCUS=(0,4,11,12)
LABELS=("pure graph-local SS15 FP16","pure sigma-first SS15 FP16","qualified reference")
COLORS=("#777777","#1479b8","#111111")

def require(value,message):
 if not value:raise ValueError(message)

def load(name,digest):
 p=Path(__file__).with_name(name);require(sha256_file(p)==digest,"frozen helper source changed: "+name)
 s=importlib.util.spec_from_file_location("_graph_assess_"+p.stem,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def validate_masks(m,theta,statistics):
 graph=producer.sigma;scopes=m["execution_counters"]["completed_stage_scopes"]
 inputs=m["execution_counters"]["sigma_first_stage_inputs"];calls=m["execution_counters"]["sigma_first_query_calls"]
 require(m["sigma_first_input_policy"]==graph.policy_metadata() and len(inputs)==len(scopes)
  and len(calls)==m["expected_patched_call_count"]
  and m["actual_prediction_checks"]["sigma_first_physical_head_contexts"]==len(calls),
  "actual graph-local stage/head coverage differs")
 cursor=0
 for binding,scope,event in zip(inputs,scopes,m["stage_events"]):
  stage=scope["stage_index"];mt,ms=graph.mask_training(theta,statistics,stage)
  require(binding==graph.training_bindings(theta,statistics,mt,ms,stage),"actual masked training/target bytes differ")
  n=scope["actual_head_contexts"];observed=Counter()
  for call in calls[cursor:cursor+n]:
   shape=call["unmasked_query"]["shape"];masked=call["actual_masked_query"]
   require(call["module_sha256"]==producer.SIGMA_FIRST_SOURCE_SHA256
    and call["policy_sha256"]==graph.policy_metadata()["policy_sha256"]
    and call["stage"]==stage and call["mask"]==graph.stage_policy(stage)
    and call["physical_head_contexts"]==1 and call["training_query_same_fixed_mask"] is True
    and call["ordered_target_binding_verified"] is True and call["target_physical_index"]==graph.ORDER[stage] and shape[1]==15+stage
    and call["unmasked_query"]["dtype"]==masked["dtype"]=="<f4" and masked["shape"]==shape,
    "actual graph-query policy/width/head descriptor differs")
   if stage<3:require(call["unmasked_query"]==masked,"global factor query changed")
   observed[(call["member"],shape[0])]+=1
  expected=Counter((member,int(rows)) for member in range(8)
   for rows,count in event["sampler"]["neural_query_batch_sizes_by_member"][str(member)].items() for _ in range(count))
  require(observed==expected,"actual masked head/query rows differ from neural event histogram")
  cursor+=n
 require(cursor==len(calls),"unconsumed graph head descriptors")
 return {"passed":True,"stage_training_targets_and_actual_masks_recomputed":True,
  "actual_head_query_descriptors_and_counts_verified":True,"scope":"source-bound actual calls; query arrays themselves not archived"}

def validate_sampling(m):
 frozen=producer.clean.frozen_sampling();seeds=frozen.worker_stream_seeds(20261111,0,0,0,1)
 require(m["seed_streams"]==seeds and m["seed"]==seeds["component_uniform"],"actual component/head seed roles differ")
 expected={"forward_precision":producer.MODE,"streams":{n:{"role":frozen.STREAM_ROLES[n],"seed":v,
  "dtype":frozen.STREAM_DTYPES[n],"device":"cpu","generator":"torch.Generator"} for n,v in seeds.items()},
  "head_policy":"raw_mass_rejection_v1","mass_envelope":1.00001,"uniform_assignment":"preassigned_per_stage_by_physical_row",
  "head_retry_preserves_row_uniform":True,"stage0_density_scope":"one generated proposal wave;8actualrawP heads; no cross-query/request reuse"}
 require(m["sampling_policy"]==expected,"original sampling policy differs")
 scope=frozen.proposal_buffer_scope(m["raw_scientific_bindings"],context_seed=20261111,case=0,repetition=0,rank=0,world_size=1,seeds=seeds)
 scope.update(representation_source_sha256=producer.REPRESENTATION_SHA256,model_inputs=m["model_inputs"],
  SS15_execution_source_sha256=producer.SS15_EXECUTION_SHA256,sigma_first_module_sha256=producer.SIGMA_FIRST_SOURCE_SHA256,
  sigma_first_policy_sha256=producer.sigma.policy_metadata()["policy_sha256"],
  stage_to_physical_order=list(producer.sigma.ORDER),physical_to_stage_order=list(producer.sigma.INVERSE_ORDER),
  samples_restored_before_FIFO_support=True)
 require(m["proposal_buffer_policy"]==frozen.proposal_buffer_policy(1200),"original request-localFIFO policy differs")
 frozen.validate_proposal_buffer_record(m["proposal_buffer"],world_size=1,rank=0,scope=scope)
 return {"passed":True,"actual_seed_streams_and_FIFO_scope_counts_verified":True}

def plots(output,before,after,arrays,assessments,order,binder):
 import matplotlib
 matplotlib.use("Agg")
 import matplotlib.pyplot as plt
 from matplotlib.lines import Line2D
 output=Path(output);legend=[Line2D([0],[0],color=c,lw=1.8,label=l) for c,l in zip(COLORS,LABELS)]
 footer="Built with PriorLabs-TabPFN\nReference-selected display; all50000 rows normalize each histogram/CDF; unchanged whole-line gate."
 def figure(rows,cols,size,title):
  f,ax=plt.subplots(rows,cols,figsize=size,squeeze=False)
  f.suptitle(title+"\nseed20261111, case0; one universal sigma-first order in all6 blocks",fontsize=13,y=.995)
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
 focus,axes=figure(2,2,(12,8.6),"Sigma-first check: four predeclared SS15 coordinates")
 focus.subplots_adjust(top=.81,bottom=.11,hspace=.5,wspace=.27)
 for ax,j in zip(axes,FOCUS):density(ax,j)
 figs["focus_density"]=save(focus,"sigma-first-density-focus.png")
 print(json.dumps({"first_focus_density_ready":str(output/figs["focus_density"]["file"])}),flush=True)
 densityfig,axes=figure(7,3,(14,20),"Graph-local SS15 versus sigma-first order: all21 densities")
 for ax,j in zip(axes,range(21)):density(ax,j)
 figs["density"]=save(densityfig,"sigma-first-density.png")
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
  figs[label]=save(f,"graph-local-"+label.replace("_","-")+".png")
 return figs,{"indices":list(FOCUS),"rule":"predeclared pureSS15 failures before graph cohort","sigma_first_used_for_selection":False,
   "histogram_bins":40,"normalization_rows":50000,"display_range":"saved reference0.001..0.999 plus3%; no assessment truncation"}


def run(*,root,manifest,manifest_sha256,output,plots_enabled=True,masked_screen=None):
 root=Path(root);results=root/"results";phase=results/"accuracy-ss15-v1";manifest=Path(manifest)
 payload=manifest.read_bytes();require(hashlib.sha256(payload).hexdigest()==manifest_sha256,"actual graph manifest custody differs")
 m=json.loads(payload);producer.require_sources()
 require(producer.source_sha256()==PRODUCER_SHA and m["runner_source_sha256"]==PRODUCER_SHA,"actual graph producer source differs")
 repo=Path(__file__).resolve().parents[1]
 blob=subprocess.run(["git","-C",str(repo),"show",m["git_revision"]+":src/posteriorforge/performance_ss15_sigma_first.py"],check=True,capture_output=True).stdout
 require(hashlib.sha256(blob).hexdigest()==PRODUCER_SHA,"recorded producer Git bytes differ")
 pure=load("evaluate_ss15_pilot.py",PURE_HELPER_SHA);binder=pure.load("evaluate_cdf_pilot.py",pure.BINDER_SHA)
 context=load_context(results/"banks-v1",20261111,4096);query=load_heldout(results/"banks-v1",0).obs.reshape(1,33)
 raw,transformed,physical_model=pure.bind_inputs(context,query)
 theta,statistics,_,model,ordered_transform=producer.model_bindings(context,query)
 require(ordered_transform==transformed and producer.sigma.restore_physical(theta).tobytes()==context.theta.astype(np.float32).tobytes(),
  "ordered target bank and inverse differ from original physical source")
 require(m["model_target_descriptors"]==producer.target_order_receipt(theta)
  and m["stage_to_physical_order"]==list(producer.sigma.ORDER)
  and m["physical_to_stage_order"]==list(producer.sigma.INVERSE_ORDER)
  and m["fallback_stage_member_bindings"]==producer.sigma.fallback_bindings()
  and m["learned_target_order_changed"] is True and m["model_target_order"]=="sigma_first_stage_order"
  and m["samples_restored_to_physical_order_before_FIFO_support"] is True,
  "actual ordered target/physical-restoration/fallback bindings differ")
 expected={"schema":producer.SCHEMA,"variant":producer.VARIANT,"status":"complete","case":0,"context_seed":20261111,
  "context_size":4096,"requested_count":50000,"repetition":0,"sources_unchanged":True,"completed_stage_indices":list(range(21)),
  "raw_scientific_bindings":raw,"model_inputs":model,"qualification_transformed_bindings":transformed,
  "execution_policy":producer.execution_metadata(),"preregistration_sha256":producer.PREREGISTRATION_SHA256,
  "actual_sigma_first_fixed_conditional_screen_performed":False,"qualified_FP32":False,"approximate_experiment":True,
  "sampler_math_changed":False,"tail_family_changed":False,"clipping_or_repair_performed":False,
  "partial_vector_filtering":False,"absolute_accuracy_gate_required":True,"original_gate_unchanged":True,**raw}
 for key,value in expected.items():require(type(m.get(key)) is type(value) and m[key]==value,"graph header/binding differs: "+key)
 counts=pure.validate_counts(m,binder);sampling=validate_sampling(m)
 theta,stats,_,_,_=producer.model_bindings(context,query);mask_receipt=validate_masks(m,theta,stats)
 gatepath=results/"gate-final.json";require(sha256_file(gatepath)==binder.BASELINE_GATE_SHA256,"original gate changed")
 gate=json.loads(gatepath.read_bytes());arrays,_,reference,custody=binder.reference_for_case(results,gate,query,0)
 binder.validate_run_reference_binding(m,custody);after=binder.load_samples(manifest,m)
 parentpath=phase/"graph-local-pilot-01/manifest.json";require(sha256_file(parentpath)==PARENT_SHA,"frozen pureSS15 parent differs")
 parent=json.loads(parentpath.read_bytes());require(parent["samples_sha256"]==PARENT_PAYLOAD_SHA,"pureSS15 payload changed")
 pure.validate_raw_scientific_binding(parent,raw);binder.validate_run_reference_binding(parent,custody)
 before=binder.load_samples(parentpath,parent)
 assessments={"before":assess_samples(before,reference,support_mask=support_mask(before)),
  "after":assess_samples(after,reference,support_mask=support_mask(after))}
 original=[r for r in gate["runs"] if r["context_seed"]==20261111 and r["case"]==0]
 require(len(original)==1,"original seed/case binding differs")
 for a in assessments.values():require(a["integrity"]["passed"] and a["reference_qualified"] and a["limits"]==original[0]["assessment"]["limits"],
  "original50000 support/qualifiedreference/physicalcriteria differ")
 screen={"status":"pending","masked_matching_screen_passed":False,"provisional_engine_attribution":True}
 if masked_screen is not None:
  path=Path(masked_screen);d=json.loads(path.read_bytes())
  if d.get("status")=="sigma_first_conditional_screen_pass":
   helper=load("qualify_ss15_sigma_first.py","dde09d9592e8d46c124b14360b8aed78ec699412ba2c5327175e0dc6adda72d8")
   helper.validate_screen(d,expected_sources=helper.source_pins(),raw_bindings=raw,
    transformed_bindings=transformed,order_policy=producer.sigma.policy_metadata())
  screen={"status":d["status"],"manifest_sha256":sha256_file(path),
   "masked_matching_screen_passed":d.get("matching_FP32_execution_passed_all21") is True,
   "provisional_engine_attribution":d.get("matching_FP32_execution_passed_all21") is not True}
 output=Path(output);output.mkdir(parents=True,exist_ok=False)
 report={"schema":SCHEMA,"attribution":"Built with PriorLabs-TabPFN","evaluator_sha256":sha256_file(__file__),
  "status":assessments["after"]["status"],"passed":assessments["after"]["passed"],"pure_NPE_claimed":True,
  "method":"one universal sigma-first order in all6 graph-local blocks; FP16 forward, no exact-global refresh",
  "summary":{k:pure.summary(v,m["parameter_order"]) for k,v in assessments.items()},
  "absolute_physical_assessments":assessments,"reference":custody,"actual_ordered_mask_validation":mask_receipt,
  "counter_validation":counts,"sampling_validation":sampling,"masked_conditional_comparison":screen,
  "accuracy_gate_required":True,"all50000_rows_used":True,"scientific_limits_unchanged":True,
  "sigma_first_manifest_sha256":manifest_sha256,"sigma_first_samples_sha256":m["samples_sha256"],
  "parent_manifest_sha256":PARENT_SHA,"parent_samples_sha256":PARENT_PAYLOAD_SHA,
  "no_model_GPU_or_reference_integrations":True,"original_gate_sha256":sha256_file(gatepath),"plots_pending":plots_enabled}
 write_json(output/"assessment.json",report);print(json.dumps({"output":str(output),"summary":report["summary"],"matching":screen},indent=2),flush=True)
 if plots_enabled:report["figures"],report["focus_selection"]=plots(output,before,after,arrays,assessments,m["parameter_order"],binder)
 report["plots_pending"]=False
 require(sha256_file(manifest)==manifest_sha256 and sha256_file(gatepath)==binder.BASELINE_GATE_SHA256,"input/gate changed during assessment")
 write_json(output/"assessment.json",report)
 write_json(output/"checksums.json",{p.name:sha256_file(p) for p in output.iterdir() if p.is_file()})
 return report

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 for name in ("root","manifest","manifest-sha256","output"):parser.add_argument("--"+name,required=True)
 parser.add_argument("--masked-screen");parser.add_argument("--no-plots",action="store_true")
 args=vars(parser.parse_args());args["plots_enabled"]=not args.pop("no_plots");run(**args)
if __name__=="__main__":main()
