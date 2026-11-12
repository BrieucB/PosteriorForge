#!/usr/bin/env python3
"""Read-only corrected sigma-native FIRST/grid assessment, all50000 rows.

Built with PriorLabs-TabPFN. Parent custody is owned by the joint campaign
wrapper. This evaluator replays saved private normals, checks real request
masks/dtypes/routes/counts and a separate finite execution screen, then uses
the original actual-case reference binder and unchanged physical gate.
No posterior refresh, neural prediction, GPU or reference integration occurs.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import importlib.util
from importlib import import_module
import json
from pathlib import Path
import subprocess
import numpy as np
from posteriorforge.artifacts import sha256_file,write_json
from posteriorforge.banks import load_context,load_heldout
from posteriorforge.inference import support_mask
from posteriorforge.science.analytical_benchmark_v2 import array_descriptor
from posteriorforge.validation import assess_samples

SCHEMA="posteriorforge.ss15_sigma_native_campaign_assessment.v1"
WRAPPER_SHA="fc3c99732c39e96b0e704d28b90323a523337c9c560828656b0bb5972d747a4f"
WRAPPER_NAME="refresh_ss15_campaign_global_block.py"
KERNEL_SHA="e346e25bc9825abf016c2bd75addaf2b025d3e3d94e1a0ea6f6e31dd02997a43"
BINDER_SHA="284d3b061696d4d823666e12d3f79c720f87692b40f45743296cfb8e93865424"
PURE_HELPER_SHA="3156d54b7d29f7aec29c2846782729a8c73da3bfcda16bb2e03cec3084899552"
JOINT_PLOT_HELPER_SHA="80337f6e9b0014a5c4b0086548871b917dc4bec36fb373dc3a016b911bf02386"
WITNESS_HELPER_SHA="0c3dd7274d6741bba2ba0ae2744542865f48b2f887a6831e1d2aa6bce47f78c0"
SCREEN_HELPER_SHA="8c1b7d432888d83eeb4ce07fa8b7f18d3744ee08f00091b1204d1e33ff747334"
NATIVE_SHA="cf135637d073439039cea3e82a98e533cd161b095aac488ea92588aacb9a1461"
ORDER_SHA="d48d1638ef9e540bdd629d321fba657af4db757700b1092fc58fb9ded1f79ad2"
FOCUS=(0,4,11,12)
COLORS=("#777777","#1479b8","#111111")
LABELS=("corrected sigma-first: mixed native FP32 stage14",
        "same parent + one joint-global update","qualified reference")


def require(value,message):
 if not value:raise ValueError(message)


def load(name,digest):
 path=Path(__file__).with_name(name)
 require(isinstance(digest,str) and len(digest)==64 and sha256_file(path)==digest,"source unpinned/changed: "+name)
 spec=importlib.util.spec_from_file_location("_sigma_native_eval_"+path.stem,path)
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def source_guard():
 w=load(WRAPPER_NAME,WRAPPER_SHA);w.require_backbone()
 require(w._ready_method("sigma_first"),"corrected native source/proof is not frozen")
 require(w.kernel.source_sha256()==KERNEL_SHA,"joint kernel source changed")
 root=Path(__file__).resolve().parents[1]
 revision=subprocess.run(["git","-C",str(root),"rev-parse","HEAD"],
  check=True,capture_output=True,text=True).stdout.strip()
 for filename,digest in ((str(Path(__file__).resolve().relative_to(root)),sha256_file(__file__)),
                         ("scripts/"+WRAPPER_NAME,WRAPPER_SHA)):
  require(hashlib.sha256(w.backbone.git_bytes(root,revision,filename)).hexdigest()==digest,
   "evaluator and custody wrapper must be committed before assessment")
 return w,{"evaluator_sha256":sha256_file(__file__),"wrapper_sha256":WRAPPER_SHA,
           "kernel_sha256":KERNEL_SHA,"git_revision":revision}


def replay_joint(w,parent,hybrid,before,after):
 """No refresh is called: recreate only the exact saved [50000,3] normal matrix."""
 require(before.dtype==after.dtype==np.float64 and before.shape==after.shape==(50000,21)
  and before[:,3:].tobytes()==after[:,3:].tobytes(),"18 local/noise bytes or strict cohort shape differ")
 r=hybrid["refresh"];seed=w.backbone.derive_seed(parent["context_seed"],w.ROLE,case=parent["case"],repetition=0)
 rng=np.random.Generator(np.random.PCG64(seed))
 require(r["role"]==w.ROLE and r["seed"]==seed and r["bit_generator"]=="PCG64"
  and r["standard_normal_draw_shape"]==[50000,3] and r["standard_normal_draw_count"]==150000
  and r["global_block_refresh_count_per_row"]==1 and rng.bit_generator.state==r["initial_private_state"],
  "one private correlated-global block policy differs")
 normals=rng.standard_normal((50000,3));mean,cov=w.kernel.conditional_parameters(before)
 require(array_descriptor(normals)==r["standard_normal_draws"] and rng.bit_generator.state==r["final_private_state"]
  and (mean+normals@np.linalg.cholesky(cov).T).tobytes()==after[:,:3].tobytes(),
  "saved whole-matrix joint draw replay differs")
 require(w.backbone.json_sha(r["initial_private_state"])==r["initial_private_state_sha256"]
  and w.backbone.json_sha(r["final_private_state"])==r["final_private_state_sha256"],
  "private state digest differs")
 d=hybrid["default_RNG_observation"]
 require(r["all18_columns_byte_identical"] and r["input_array_unchanged"]
  and r["default_numpy_python_RNG_preserved"]
  and r["default_numpy_python_RNG_before_sha256"]==r["default_numpy_python_RNG_after_sha256"]
  and d["preserved"] and d["before_sha256"]==d["after_sha256"]
  and d["torch_CPU_CUDA_RNG_hashes_observed"] is False,"recorded RNG/control scope differs")
 require(support_mask(after).all(),"hybrid is not a complete supported cohort")
 return {"passed":True,"private_normal_matrix_replayed":True,"private_seed":seed,
         "all18_local_noise_columns_byte_identical":True,"new_refresh_performed":False}


def authenticate(w,parent_path,hybrid_path,expected_hybrid_sha):
 require(w.backbone.digest(expected_hybrid_sha) and sha256_file(hybrid_path)==expected_hybrid_sha,
  "explicit hybrid manifest SHA mismatch")
 hybrid=json.loads(Path(hybrid_path).read_bytes())
 require(hybrid.get("schema")==w.SCHEMA and hybrid.get("status")=="complete"
  and hybrid.get("method")=="sigma_first" and hybrid.get("requested_count")==50000
  and hybrid.get("sources_unchanged") is True,"only completed corrected sigma joint cohort accepted")
 parent,before,lineage=w.load_parent(parent_path,hybrid["parent"]["manifest_sha256"],w.kernel.metadata())
 require(lineage["method"]=="sigma_first" and hybrid["parent"]==lineage
  and hybrid["variant"]==w.METHODS["sigma_first"]["hybrid_variant"]
  and all(hybrid[k]==parent[k] for k in ("context_seed","case","repetition","parameter_order","scientific_fingerprint"))
  and hybrid["raw_scientific_bindings"]==parent["raw_scientific_bindings"]
  and hybrid["conditional_kernel"]==w.kernel.metadata(),"hybrid parent/source/physical lineage differs")
 pins=hybrid["source_pins"];repo=Path(__file__).resolve().parents[1]
 require(pins["producer_sha256"]==WRAPPER_SHA and pins["frozen_joint_refresh_sha256"]==w.BACKBONE_SHA
  and pins["kernel_sha256"]==KERNEL_SHA and pins["grid_contract_sha256"]==w.CONTRACT_SHA
  and pins["sources_committed_before_run"] is True,"hybrid committed source/contract differs")
 for filename,digest in (("scripts/"+WRAPPER_NAME,WRAPPER_SHA),("scripts/refresh_ss15_global_block.py",w.BACKBONE_SHA),
                         ("src/posteriorforge/conditional_globals.py",KERNEL_SHA),(w.CONTRACT_PATH,w.CONTRACT_SHA)):
  require(hashlib.sha256(w.backbone.git_bytes(repo,pins["git_revision"],filename)).hexdigest()==digest,
   "executed hybrid Git bytes differ")
 require(hashlib.sha256(w.backbone.git_bytes(repo,pins["grid_contract_origin_commit"],w.CONTRACT_PATH)).hexdigest()==w.CONTRACT_SHA,
  "preregistered grid contract differs")
 payload=Path(hybrid_path).with_name("samples.npz")
 require(hybrid["samples_file"]=="samples.npz" and sha256_file(payload)==hybrid["samples_sha256"],"hybrid payload differs")
 with np.load(payload,allow_pickle=False) as a:
  require(a.files==["samples"],"physical samples only");after=a["samples"].copy()
 require(array_descriptor(after)==hybrid["samples"],"hybrid array descriptor differs")
 receipt=replay_joint(w,parent,hybrid,before,after)
 return parent,hybrid,before,after,receipt


def actual_inputs(results,seed,case,w):
 """Data permutation helpers only; no original prototype predictor is invoked."""
 from posteriorforge import performance_ss15_sigma_first as order_producer
 require(order_producer.source_sha256()==w.SIGMA_ANCESTRAL_PRODUCER_SHA
  and order_producer.sigma.source_sha256()==ORDER_SHA,"frozen scientific order helper changed")
 context=load_context(Path(results)/"banks-v1",seed,4096)
 query=load_heldout(Path(results)/"banks-v1",case).obs.reshape(1,33)
 theta,statistics,model_query,model,transformed=order_producer.model_bindings(context,query)
 raw={"checkpoint_sha256":w.backbone.CHECKPOINT_SHA,"scientific_fingerprint":w.kernel.metadata()["scientific_fingerprint"],
      "parameter_order":w.kernel.metadata()["physical_parameter_order"],
      "context_theta":array_descriptor(context.theta),"context_obs":array_descriptor(context.obs),"query":array_descriptor(query)}
 physical={**model,"context_theta":array_descriptor(context.theta.astype(np.float32))}
 return {"theta":theta,"statistics":statistics,"query":query,"raw":raw,"model":model,"physical":physical,
         "transformed":transformed,"target":order_producer.target_order_receipt(theta)}


def load_parent_module(w):
 path=Path(__file__).resolve().parents[1]/w.PARENT_SOURCE_PATH
 require(w.backbone.digest(w.PARENT_SOURCE_SHA) and sha256_file(path)==w.PARENT_SOURCE_SHA,
  "shared campaign source is unpinned/changed")
 return import_module("posteriorforge.performance_ss15_scientific_campaign")


def execution_checks(w,parent,actual,anchor,screen_path,binder,pure):
 from posteriorforge import execution_ss15_sigma_native_policy as native
 from posteriorforge import execution_ss15_sigma_first as order
 require(native.source_sha256()==NATIVE_SHA and order.source_sha256()==ORDER_SHA,"native/order source changed")
 policy=native.policy_metadata(w.METHODS["sigma_first"]["native_stage_indices"])
 require(parent["active_native_execution_policy"]==policy,"actual imported native policy differs")
 require(parent["raw_scientific_bindings"]==actual["raw"] and parent["model_inputs"]==actual["model"]
  and parent["actual_physical_model_inputs"]==actual["physical"]
  and parent["qualification_transformed_bindings"]==actual["transformed"]
  and parent["model_target_descriptors"]==actual["target"],"actual saved bank/query/order descriptors differ")
 counts=pure.validate_counts(parent,binder)
 scopes=parent["execution_counters"]["completed_stage_scopes"]
 inputs=parent["execution_counters"]["sigma_first_stage_inputs"]
 calls=parent["execution_counters"]["sigma_first_query_calls"]
 require(len(inputs)==len(scopes) and len(calls)==counts["physical_head_contexts"],"actual mask journals incomplete")
 witness=load("qualify_ss15_conditionals.py",WITNESS_HELPER_SHA);cursor=0
 for binding,scope,event in zip(inputs,scopes,parent["stage_events"]):
  stage=scope["stage_index"];theta,statistics=order.mask_training(actual["theta"],actual["statistics"],stage)
  require(binding==order.training_bindings(actual["theta"],actual["statistics"],theta,statistics,stage),
   "actual masked training predictors or target differ")
  witness.witness_guard(scope["precision_session"],mode="amp_fp16_prefill_query_v1",stage=stage,
                        fallback_map=policy["active_fallback_map"])
  model=scope["model"]
  require(model["canonical_borders_dtype"]==model["physical_borders_dtype"]=="torch.float32"
   and model["temperature"]==.9 and model["contract"]["input_normalization_train_only_verified"],
   "actual borders, temperature or train-only normalization differ")
  for module in model["contract"]["modules"]:
   if module["type"]=="InputNormalizationEncoderStep":
    require(all(t["dtype"]=="torch.float32" for t in module["tensors"].values()),
     "actual normalizer tensors left Float32")
  observed=Counter()
  for call in calls[cursor:cursor+scope["actual_head_contexts"]]:
   shape=call["unmasked_query"]["shape"]
   require(call["module_sha256"]==ORDER_SHA and call["policy_sha256"]==order.policy_metadata()["policy_sha256"] and call["mask"]==order.stage_policy(stage)
    and call["target_physical_index"]==order.ORDER[stage] and call["stage"]==stage
    and call["training_query_same_fixed_mask"] and call["ordered_target_binding_verified"]
    and call["physical_head_contexts"]==1 and shape[1]==15+stage
    and call["actual_masked_query"]["shape"]==shape
    and call["unmasked_query"]["dtype"]==call["actual_masked_query"]["dtype"]=="<f4","actual query mask/shape/dtype differs")
   if stage<3:require(call["unmasked_query"]==call["actual_masked_query"],"global query input changed")
   observed[(call["member"],shape[0])]+=1
  hist=event["sampler"]["neural_query_batch_sizes_by_member"]
  expected=Counter((member,int(rows)) for member in range(8) for rows,count in hist[str(member)].items() for _ in range(count))
  require(observed==expected,"real query rows/head counts differ")
  uncached=sum(count for (member,_),count in expected.items() if member in policy["active_fallback_map"].get(str(stage),[]))
  routes=scope["hybrid_route_counters"]
  require(routes["uncached_predict_count"]==uncached and routes["resident_component_predict_count"]==sum(expected.values())-uncached,
   "actual native/cached routes differ from active policy")
  cursor+=scope["actual_head_contexts"]
 require(cursor==len(calls),"unused query journal")
 from posteriorforge import performance_ss15 as pilot
 sampling=pilot.clean.frozen_sampling();seed,case=parent["context_seed"],parent["case"]
 require(parent["seed_streams"]==sampling.worker_stream_seeds(seed,case,0,0,1),"actual private neural streams differ")
 require(parent["proposal_buffer_policy"]==sampling.proposal_buffer_policy(1200),"FIFO policy differs")
 expected_sampling={"forward_precision":"amp_fp16_prefill_query_v1",
  "streams":{name:{"role":sampling.STREAM_ROLES[name],"seed":value,"dtype":sampling.STREAM_DTYPES[name],
  "device":"cpu","generator":"torch.Generator"} for name,value in parent["seed_streams"].items()},
  "head_policy":"raw_mass_rejection_v1","mass_envelope":1.00001,
  "uniform_assignment":"preassigned_per_stage_by_physical_row","head_retry_preserves_row_uniform":True,
  "stage0_density_scope":"one generated proposal wave;8actualrawP heads; no cross-query/request reuse"}
 require(parent["sampling_policy"]==expected_sampling,"actual frozen sampling policy differs")
 if parent["schema"]==w.NATIVE_FIRST_SCHEMA:
  fs=sampling.proposal_buffer_scope(parent["raw_scientific_bindings"],context_seed=seed,case=case,
    repetition=0,rank=0,world_size=1,seeds=parent["seed_streams"])
  fs.update(representation_source_sha256=pilot.REPRESENTATION_SHA256,model_inputs=parent["model_inputs"],
   SS15_execution_source_sha256=pilot.SS15_EXECUTION_SHA256,sigma_first_module_sha256=ORDER_SHA,
   native_execution_adapter_source_sha256=NATIVE_SHA,native_execution_policy_sha256=policy["policy_sha256"],
   sigma_first_policy_sha256=order.policy_metadata()["policy_sha256"],
   stage_to_physical_order=list(w.SIGMA_ORDER),physical_to_stage_order=[int(i) for i in np.argsort(w.SIGMA_ORDER)],
   samples_restored_before_FIFO_support=True)
 else:
  campaign=load_parent_module(w)
  fs=campaign.proposal_scope(sampling,parent["raw_scientific_bindings"],parent["model_inputs"],
   method="sigma_first",context_seed=seed,case=case,repetition=0,seeds=parent["seed_streams"])
 sampling.validate_proposal_buffer_record(parent["proposal_buffer"],world_size=1,rank=0,scope=fs)
 require(sha256_file(screen_path)==w.METHODS["sigma_first"]["qualification_sha256"],"exact final finite screen differs")
 screen=json.loads(Path(screen_path).read_bytes());helper=load("qualify_ss15_sigma_native.py",SCREEN_HELPER_SHA)
 replay=helper.validate_screen(screen,expected_sources=helper.source_pins(),raw_bindings=anchor["raw"],
  transformed_bindings=anchor["transformed"],order_policy=order.policy_metadata(),native_policy=policy)
 if parent["schema"]==w.NATIVE_FIRST_SCHEMA:
  require(parent["matched_execution_screen"]["replay"]==replay
   and parent["actual_sigma_first_fixed_conditional_screen_performed"] is False
   and parent["fresh_same_policy_all21_screen_replayed"] is True,"separate screen replay scope differs")
 else:
  a=parent["conditional_anchor"]
  require(a["replay"]==replay and a["raw_scientific_bindings"]==anchor["raw"]
   and a["model_inputs"]==anchor["model"] and a["physical_model_inputs"]==anchor["physical"]
   and a["transformed_bindings"]==anchor["transformed"]
   and a["qualifies_actual_request"] is False,"finite anchor was replaced with actual request inputs")
 return {"passed":True,"real_masks_dtype_routes_counts_verified":True,"counts":counts,
  "finite_execution_screen_replay":replay,"screen_scope":"separate case0 finite prefixes; not this50000 request or posterior accuracy"}


def assess_pair(before,after,reference,gate,parent):
 expected=[r for r in gate["runs"] if r["context_seed"]==parent["context_seed"] and r["case"]==parent["case"]]
 require(len(expected)==1,"actual seed/case absent from immutable original grid")
 assessments={"before":assess_samples(before,reference,support_mask=support_mask(before)),
              "after":assess_samples(after,reference,support_mask=support_mask(after))}
 for assessment in assessments.values():
  require(assessment["integrity"]["passed"] and assessment["reference_qualified"]
   and assessment["limits"]==expected[0]["assessment"]["limits"],"original integrity/reference/gate limits differ")
 controls=load("evaluate_ss15_sigma_first_global_block.py",JOINT_PLOT_HELPER_SHA).unchanged_marginals(
  assessments["before"],assessments["after"])
 return assessments,controls


def plots(output,before,after,arrays,assessments,order,binder,labels,*,seed,case,full_cdf=True):
 import matplotlib
 matplotlib.use("Agg")
 import matplotlib.pyplot as plt
 from matplotlib.lines import Line2D
 output=Path(output);legend=[Line2D([0],[0],color=c,lw=1.8,label=l) for c,l in zip(COLORS,labels)]
 footer="Built with PriorLabs-TabPFN\nReference-selected display; all50000 rows normalize each histogram/CDF; unchanged whole-line gate."
 def figure(rows,cols,size,title):
  f,ax=plt.subplots(rows,cols,figsize=size,squeeze=False)
  f.suptitle(title+f"\nseed{seed}, case{case}; mixed FP16/FP32; all8 native stage14; one joint-global update",fontsize=13,y=.995)
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
 if not full_cdf:
  return figs,{"indices":list(FOCUS),"rule":"predeclared fixed coordinates; no new posterior selection",
               "normalization_rows":50000,"histogram_bins":40,"display_range":"saved reference0.001..0.999 plus3%"}
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

def reference_and_assess(results,gate,parent,query,before,after,binder):
 arrays,_,reference,custody=binder.reference_for_case(results,gate,query,parent["case"])
 binder.validate_run_reference_binding(parent,custody)
 assessments,controls=assess_pair(before,after,reference,gate,parent)
 return arrays,custody,assessments,controls


def run(*,root,parent_manifest,hybrid_manifest,hybrid_manifest_sha256,execution_screen,output,plots_enabled=True,full_cdf_plots=False):
 w,sources=source_guard();parent,hybrid,before,after,auth=authenticate(w,parent_manifest,hybrid_manifest,hybrid_manifest_sha256)
 results=Path(root)/"results";pure=load("evaluate_ss15_pilot.py",PURE_HELPER_SHA);binder=load("evaluate_cdf_pilot.py",BINDER_SHA)
 gatepath=results/"gate-final.json";require(sha256_file(gatepath)==binder.BASELINE_GATE_SHA256,"immutable gate changed")
 gate=json.loads(gatepath.read_bytes());seed,case=parent["context_seed"],parent["case"]
 actual=actual_inputs(results,seed,case,w)
 anchor=actual if (seed,case)==w.ANCHOR else actual_inputs(results,*w.ANCHOR,w)
 execution=execution_checks(w,parent,actual,anchor,execution_screen,binder,pure)
 arrays,custody,assessments,controls=reference_and_assess(results,gate,parent,actual["query"],before,after,binder)
 output=Path(output);output.mkdir(parents=True,exist_ok=False)
 record={"schema":SCHEMA,"attribution":"Built with PriorLabs-TabPFN","context_seed":seed,"case":case,
  "status":assessments["after"]["status"],"passed":assessments["after"]["passed"],
  "source_pins":sources,"parent_manifest_sha256":hybrid["parent"]["manifest_sha256"],
  "hybrid_manifest_sha256":hybrid_manifest_sha256,"hybrid_samples_sha256":hybrid["samples_sha256"],
  "authentication":auth,"execution_checks":execution,"reference":custody,
  "absolute_physical_assessments":assessments,"unchanged18_control":controls,
  "summary":{k:pure.summary(v,parent["parameter_order"]) for k,v in assessments.items()},
  "scientific_limits_unchanged":True,"all50000_rows_used":True,"original_gate_sha256":sha256_file(gatepath),
  "source_only_execution_qualification_is_not_posterior_accuracy":True,
  "no_new_posterior_refresh_model_GPU_or_reference_integration":True,"plots_pending":plots_enabled}
 write_json(output/"assessment.json",record)
 print(json.dumps({"output":str(output),"context_seed":seed,"case":case,
                   "status":record["status"],"passed":record["passed"],"summary":record["summary"]},sort_keys=True),flush=True)
 if plots_enabled:
  record["figures"],record["focus_selection"]=plots(output,before,after,arrays,assessments,parent["parameter_order"],binder,LABELS,
   seed=seed,case=case,full_cdf=full_cdf_plots or (seed,case)==w.ANCHOR)
 record["plots_pending"]=False
 require(sha256_file(hybrid_manifest)==hybrid_manifest_sha256
  and sha256_file(parent_manifest)==hybrid["parent"]["manifest_sha256"]
  and sha256_file(gatepath)==binder.BASELINE_GATE_SHA256,"input/gate changed during read-only assessment")
 write_json(output/"assessment.json",record)
 write_json(output/"checksums.json",{p.name:sha256_file(p) for p in output.iterdir() if p.is_file()})
 return record


def main():
 parser=argparse.ArgumentParser(description=__doc__)
 for name in ("root","parent-manifest","hybrid-manifest","hybrid-manifest-sha256","execution-screen","output"):
  parser.add_argument("--"+name,required=True)
 parser.add_argument("--no-plots",action="store_true");parser.add_argument("--full-cdf-plots",action="store_true")
 args=vars(parser.parse_args());args["plots_enabled"]=not args.pop("no_plots");m=run(**args)
 print(json.dumps({k:m.get(k) for k in ("status","passed","context_seed","case","summary")},sort_keys=True))


if __name__=="__main__":
 main()
