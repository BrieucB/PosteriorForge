#!/usr/bin/env python3
"""ONE physical G3 update and original gates/plots for saved-data FORENSIC cohorts.

Built with PriorLabs-TabPFN. Both NEW cohorts come from a FAILED unguarded
neural request. Missing request footers remain missing even if forensic gates
pass. Refresh is a separate one-time command; assessment reuses saved outputs.
"""
from __future__ import annotations
import argparse,copy,hashlib,importlib.util,json,pickle,random,subprocess
from pathlib import Path
import numpy as np
from posteriorforge.artifacts import versions,sha256_file,write_json
from posteriorforge.banks import derive_seed,load_heldout
from posteriorforge.science.analytical_benchmark_v2 import array_descriptor
from posteriorforge import conditional_globals as kernel

REPLAY_SOURCE_SHA256="20bfc8b64eab433de82432f038b53740f706d37f9b43046228c0d59d821071d1"
KERNEL_SHA256="e346e25bc9825abf016c2bd75addaf2b025d3e3d94e1a0ea6f6e31dd02997a43"
OLD_NEW_EVALUATOR_SHA256="6c49e59ccee9bebeecd53990556a6e00a9f41b2fc53a2e8ecf9827014efc2d56"
ROLE="ss15_global_block_gibbs_once_v1"
COUNT=50000
FOCUS=(0,7,8,10)
LABELS=("OLD physical-target neural mixed FP16/FP32 + physical G3",
 "NEW FORENSIC prior coords FP32: 19 cached; all8 original at6/10",
 "NEW same FORENSIC rows + ONE SEPARATE physical G3","qualified reference")
COLORS=("#c58a37","#777777","#1479b8","#111111")
G3_SCHEMA="posteriorforge.prior_forensic_global_refresh.v1"
ASSESS_SCHEMA="posteriorforge.prior_forensic_assessment.v1"

def require(value,message):
 if not value:raise ValueError(message)

def source_sha256():return sha256_file(__file__)

def load(filename,digest):
 path=Path(__file__).with_name(filename)
 require(isinstance(digest,str) and sha256_file(path)==digest,"frozen source differs: "+filename)
 spec=importlib.util.spec_from_file_location("_forensic_"+path.stem,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def source_pins():
 replay=load("replay_prior_forensic_cohort_v1.py",REPLAY_SOURCE_SHA256)
 require(kernel.source_sha256()==KERNEL_SHA256,"unchanged original physical G3 kernel required")
 old=load("evaluate_prior_coordinates_pilot_v1.py",OLD_NEW_EVALUATOR_SHA256)
 return replay,old,{"forensic_refresh_assessment":source_sha256(),"replay":REPLAY_SOURCE_SHA256,
  "original_G3_kernel":KERNEL_SHA256,"original_scientific_reference_helper":old.FROZEN_EVALUATOR_SHA256,
  "prior_display_helper":OLD_NEW_EVALUATOR_SHA256,"neural_request_final_footers_recovered":False}

def require_committed():
 root=Path(__file__).resolve().parents[1]
 revision=subprocess.run(["git","-C",str(root),"rev-parse","HEAD"],check=True,capture_output=True,text=True).stdout.strip()
 raw=subprocess.run(["git","-C",str(root),"show",revision+":"+str(Path(__file__).relative_to(root))],check=True,capture_output=True).stdout
 require(hashlib.sha256(raw).hexdigest()==source_sha256(),"forensic source must be committed before execution")
 return revision

def state_hash():
 return hashlib.sha256(pickle.dumps((np.random.get_state(),random.getstate()),protocol=5)).hexdigest()

class RecordingGenerator(np.random.Generator):
 """Call unchanged base standard_normal once, retain ACTUAL returned Z."""
 def __init__(self,bit_generator):
  super().__init__(bit_generator);self.actual_draws=None;self.standard_normal_calls=0
 def standard_normal(self,*args,**kwargs):
  require(self.standard_normal_calls==0,"single actual G3 normal matrix only")
  values=super().standard_normal(*args,**kwargs)
  self.standard_normal_calls+=1
  self.actual_draws=values.copy()
  return values

def refresh_with_actual_draws(values,seed):
 """Exactly one unchanged e346.refresh call, no replay generator or reroll."""
 require(isinstance(values,np.ndarray) and values.dtype==np.float64 and values.ndim==2 and values.shape[1]==21,"original F64 physical rows required")
 original=array_descriptor(values);before=state_hash()
 rng=RecordingGenerator(np.random.PCG64(seed));initial=copy.deepcopy(rng.bit_generator.state)
 mean,covariance=kernel.conditional_parameters(values)
 result=kernel.refresh(values,rng)
 normals=rng.actual_draws;final=copy.deepcopy(rng.bit_generator.state)
 require(rng.standard_normal_calls==1 and normals.shape==(len(values),3) and normals.dtype==np.float64,"one ACTUAL row-major Gaussian matrix required")
 require(result[:,:3].tobytes()==(mean+normals@np.linalg.cholesky(covariance).T).tobytes(),"actual Z does not reconstruct unmodified kernel output")
 require(result[:,3:].tobytes()==values[:,3:].tobytes() and array_descriptor(values)==original,"decoded physical18 or input changed")
 require(state_hash()==before,"default NumPy/Python RNG changed")
 return result,normals,{"seed":int(seed),"role":ROLE,"bit_generator":"PCG64",
  "standard_normal_method_calls":1,"standard_normal_draw_shape":[len(values),3],"standard_normal_scalar_count":len(values)*3,
  "standard_normal_draws":array_descriptor(normals),"draws_are_actual_returned_matrix_not_rerolled":True,
   "initial_private_state":initial,"final_private_state":final,
  "initial_private_state_sha256":hashlib.sha256(json.dumps(initial,sort_keys=True,separators=(",",":")).encode()).hexdigest(),
  "final_private_state_sha256":hashlib.sha256(json.dumps(final,sort_keys=True,separators=(",",":")).encode()).hexdigest(),
  "all18_physical_columns_byte_identical":True,
  "input_array_unchanged":True,"default_numpy_python_RNG_before_sha256":before,"default_numpy_python_RNG_after_sha256":state_hash(),
  "default_numpy_python_RNG_preserved":True,"torch_CPU_CUDA_RNG_observed":False,
  "global_block_refresh_count_per_row":1,"coordinate_decode_after_global_refresh":False,
  "kernel_source_unchanged":True,"old_wrapper_a995_run_identity_claimed":False}

def load_cohort(replay_manifest,replay_manifest_sha256,failed_manifest,replay):
 require(sha256_file(replay_manifest)==replay_manifest_sha256,"explicit forensic parent SHA differs")
 m=json.loads(Path(replay_manifest).read_bytes())
 require(m["schema"]==replay.SCHEMA and m["status"]=="forensic_complete" and m["original_request_status"]=="failed"
  and m["production_qualification_claimed"] is False and m["original_missing_footers_recovered"] is False
  and m["source_pins"]==replay.source_pins(),"forensic envelope or source differs")
 parent,physical,auth=replay.authenticate_failed(failed_manifest)
 require(m["authentication"]==auth,"saved forensic custody differs")
 path=Path(replay_manifest).with_name(m["samples_file"])
 require(sha256_file(path)==m["samples_sha256"],"saved forensic50000 payload differs")
 with np.load(path,allow_pickle=False) as archive:
  require(archive.files==["samples"],"forensic payload keys differ");values=archive["samples"]
 require(values.dtype==np.float64 and values.shape==(COUNT,21) and array_descriptor(values)==m["samples"]
  and values.tobytes()==physical[:COUNT].tobytes(),"fixed first50000 proposal order differs")
 scope={**parent["proposal_buffer"]["request_scope"],"forensic_from_failed_request":True,
  "failed_manifest_sha256":replay.FAILED_SHA256,"production_request_completion_claimed":False}
 replay.helper().validate_proposal_buffer_record(m["proposal_buffer"],world_size=1,rank=0,scope=scope)
 return m,values,parent

def refresh(*,replay_manifest,replay_manifest_sha256,failed_manifest,output):
 replay,old,pins=source_pins();revision=require_committed()
 parent,values,failed=load_cohort(replay_manifest,replay_manifest_sha256,failed_manifest,replay)
 out=Path(output);out.mkdir(parents=True,exist_ok=False)
 record={"schema":G3_SCHEMA,"status":"running_forensic_G3","attribution":"Built with PriorLabs-TabPFN",
  "source_pins":pins,"git_revision":revision,"parent_manifest_sha256":replay_manifest_sha256,"failed_neural_manifest_sha256":replay.FAILED_SHA256,
  "original_neural_request_status":"failed","original_missing_footers_recovered":False,"production_qualification_claimed":False,
  "context_seed":20261111,"case":2,"repetition":0,"requested_count":COUNT,"physical_parameter_order":parent["parameter_order"],
  "conditional_kernel":kernel.metadata(),"new_neural_proposals":0,"new_GPU_runs":0,"coordinate_decode_after_global_refresh":False,
  "method":"FORENSIC saved failed-request-derived rows plus ONE separate unchanged physical G3 block"}
 write_json(out/"manifest.json",record)
 try:
  seed=derive_seed(20261111,ROLE,case=2,repetition=0)
  result,normals,receipt=refresh_with_actual_draws(values,seed)
  require(result.shape==(COUNT,21),"strict50000 row quota")
  with (out/"samples.npz").open("xb") as stream:np.savez(stream,samples=result)
  with (out/"actual-standard-normal-draws.npz").open("xb") as stream:np.savez(stream,standard_normal_draws=normals)
  require(source_pins()[2]==pins and sha256_file(replay_manifest)==replay_manifest_sha256,"source or forensic parent changed")
  record.update(status="forensic_complete",samples_file="samples.npz",samples=array_descriptor(result),
   samples_sha256=sha256_file(out/"samples.npz"),actual_draws_file="actual-standard-normal-draws.npz",
   actual_draws_sha256=sha256_file(out/"actual-standard-normal-draws.npz"),refresh=receipt)
 except BaseException as error:
  record.update(status="failed_forensic_G3",error_type=type(error).__name__,error=str(error));write_json(out/"manifest.json",record);raise
 write_json(out/"manifest.json",record);write_json(out/"checksums.json",{p.name:sha256_file(p) for p in out.iterdir() if p.is_file()})
 print(json.dumps({"forensic_G3_complete":str(out),"manifest_sha256":sha256_file(out/"manifest.json"),"samples_sha256":record["samples_sha256"]}),flush=True)
 return record

def authenticate_hybrid(path,expected_sha,before,parent_sha,pins):
 require(sha256_file(path)==expected_sha,"explicit saved forensic G3 SHA differs")
 m=json.loads(Path(path).read_bytes())
 require(m["schema"]==G3_SCHEMA and m["status"]=="forensic_complete" and m["source_pins"]==pins
  and m["parent_manifest_sha256"]==parent_sha and m["original_neural_request_status"]=="failed"
  and m["production_qualification_claimed"] is False and m["original_missing_footers_recovered"] is False,
  "FORENSIC G3 envelope/source differs")
 p=Path(path).with_name(m["samples_file"]);z=Path(path).with_name(m["actual_draws_file"])
 require(sha256_file(p)==m["samples_sha256"] and sha256_file(z)==m["actual_draws_sha256"],"forensic output/draw payload differs")
 with np.load(p,allow_pickle=False) as archive:after=archive["samples"]
 with np.load(z,allow_pickle=False) as archive:normals=archive["standard_normal_draws"]
 require(after.shape==before.shape==(COUNT,21) and after.dtype==before.dtype==np.float64 and normals.shape==(COUNT,3)
  and normals.dtype==np.float64 and array_descriptor(after)==m["samples"]
  and array_descriptor(normals)==m["refresh"]["standard_normal_draws"],"actual forensic matrix descriptors differ")
 mean,cov=kernel.conditional_parameters(before)
 require(after[:,:3].tobytes()==(mean+normals@np.linalg.cholesky(cov).T).tobytes()
  and after[:,3:].tobytes()==before[:,3:].tobytes(),"saved actual-draw G3/18column replay differs")
 r=m["refresh"];seed=derive_seed(20261111,ROLE,case=2,repetition=0)
 require(m["conditional_kernel"]==kernel.metadata() and r["seed"]==seed and r["role"]==ROLE
  and r["bit_generator"]=="PCG64" and r["initial_private_state"]==np.random.PCG64(seed).state
  and r["standard_normal_method_calls"]==1 and r["standard_normal_draw_shape"]==[COUNT,3]
  and r["standard_normal_scalar_count"]==COUNT*3 and r["draws_are_actual_returned_matrix_not_rerolled"]
  and r["default_numpy_python_RNG_preserved"] and r["default_numpy_python_RNG_before_sha256"]==r["default_numpy_python_RNG_after_sha256"]
  and r["global_block_refresh_count_per_row"]==1 and r["coordinate_decode_after_global_refresh"] is False,
  "one original physical G3 policy differs")
 for name in ("initial_private_state","final_private_state"):
  require(hashlib.sha256(json.dumps(r[name],sort_keys=True,separators=(",",":")).encode()).hexdigest()==r[name+"_sha256"],
   "recorded private-state hash differs")
 require(r["all18_physical_columns_byte_identical"] and r["input_array_unchanged"],"physical18 or input identity receipt differs")
 return m,after

def assess(*,root,replay_manifest,replay_manifest_sha256,failed_manifest,hybrid_manifest,hybrid_manifest_sha256,output):
 replay,old,pins=source_pins();require_committed()
 parent,before,failed=load_cohort(replay_manifest,replay_manifest_sha256,failed_manifest,replay)
 hybrid,after=authenticate_hybrid(hybrid_manifest,hybrid_manifest_sha256,before,replay_manifest_sha256,pins)
 root=Path(root);results=root/"results"
 frozen=old.load("evaluate_ss15_sigma_native_campaign.py",old.FROZEN_EVALUATOR_SHA256)
 binder=frozen.load("evaluate_cdf_pilot.py",frozen.BINDER_SHA);pure=frozen.load("evaluate_ss15_pilot.py",frozen.PURE_HELPER_SHA)
 gatepath=results/"gate-final.json";require(sha256_file(gatepath)==binder.BASELINE_GATE_SHA256,"unchanged original physical gate required")
 gate=json.loads(gatepath.read_bytes());query=load_heldout(results/"banks-v1",2).obs.reshape(1,33)
 arrays,reference,assessments,controls=frozen.reference_and_assess(results,gate,failed,query,before,after,binder)
 baseline,baseline_assessment,baseline_auth=old.load_baseline(frozen,
  results/"accuracy-ss15-v1/sigma-native-grid-completion-01/completion.json",
  results/"accuracy-ss15-v1/sigma-native-grid-01/seed-20261111-case-2/manifest.json",
  results/"accuracy-ss15-v1/sigma-native-grid-refresh-01/seed-20261111-case-2/manifest.json",
  results/"accuracy-ss15-v1/sigma-native-grid-assessment-01/seed-20261111-case-2/assessment.json",failed["raw_scientific_bindings"])
 out=Path(output);out.mkdir(parents=True,exist_ok=False)
 record={"schema":ASSESS_SCHEMA,"attribution":"Built with PriorLabs-TabPFN","context_seed":20261111,"case":2,
  "status":"forensic_assessed","source_pins":pins,"forensic_parent_sha256":replay_manifest_sha256,
  "forensic_G3_sha256":hybrid_manifest_sha256,"failed_neural_manifest_sha256":replay.FAILED_SHA256,
  "original_request_status":"failed","original_missing_final_footers":parent["authentication"]["missing_failed_request_success_footers"],
  "missing_failed_request_footers_recovered":False,"production_or_posterior_accuracy_qualification_claimed":False,
  "gate_status_for_forensic_arrays_only":assessments["after"]["status"],"gate_passed_for_forensic_arrays_only":assessments["after"]["passed"],
  "absolute_physical_assessments":assessments,"summary":{k:pure.summary(v,failed["parameter_order"]) for k,v in assessments.items()},
  "unchanged18_control":controls,"reference":reference,"original_gate_sha256":sha256_file(gatepath),
  "all50000_rows_used":True,"scientific_limits_unchanged":True,"focus_indices":list(FOCUS),"plots_pending":True,
  "old_baseline_custody":baseline_auth,"old_baseline_summary":baseline_assessment["summary"],
  "configuration_comparison":{"coordinates_precision_routing_all_changed":True,"isolated_coordinate_effect_claimed":False},
  "new_neural_proposals":0,"new_reference_integrations":0,"new_G3_draws_during_assessment":0}
 write_json(out/"assessment.json",record)
 print(json.dumps({"forensic_scalar_output":str(out),"summary":record["summary"],
  "forensic_only":True,"failed_original_request_status":"failed"},sort_keys=True),flush=True)
 record["figures"]=plots(out,baseline,before,after,arrays,assessments,failed["parameter_order"],binder)
 record["plots_pending"]=False
 require(sha256_file(replay_manifest)==replay_manifest_sha256 and sha256_file(hybrid_manifest)==hybrid_manifest_sha256
  and sha256_file(failed_manifest)==replay.FAILED_SHA256 and sha256_file(gatepath)==binder.BASELINE_GATE_SHA256,"saved assessment inputs changed")
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
  f.suptitle(title+"\nFORENSIC saved proposals from FAILED request; missing final neural/RNG guards remain missing",fontsize=13,y=.995)
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
 figs={};f,axes=figure(2,2,(12,8.6),"FORENSIC prior-coordinate cohorts: four predeclared physical coordinates")
 f.subplots_adjust(top=.81,bottom=.11,hspace=.5,wspace=.27)
 for ax,j in zip(axes,FOCUS):density(ax,j)
 figs["focus_density"]=save(f,"forensic-density-focus.png")
 print(json.dumps({"first_focus_density_ready":str(out/"forensic-density-focus.png")}),flush=True)
 f,axes=figure(7,3,(14,20),"FORENSIC prior-coordinate cohorts: all21 physical densities")
 for ax,j in zip(axes,range(21)):density(ax,j)
 figs["density"]=save(f,"forensic-density.png")
 precision=assessments["after"]["reference_qualification"]["precision"]
 for label,indices,shape,size,title in (("marginal_cdf",range(21),(7,3),(14,20),"FORENSIC all21 physical marginal CDFs"),
  ("joint_cdf",range(21,53),(8,4),(17,23),"FORENSIC all32 fixed physical projection CDFs")):
  f,axes=figure(*shape,size,title)
  for ax,j in zip(axes,indices):
   x,cdf,lo,hi=domain(j);grid=np.linspace(lo,hi,700)
   for a,color in ((baseline,COLORS[0]),(before,COLORS[1]),(after,COLORS[2])):
    values=np.sort(a[:,j] if j<21 else a@arrays["directions"][j]);ax.plot(grid,np.searchsorted(values,grid,side="right")/len(values),color=color,lw=1.1)
   ax.plot(x,cdf,color=COLORS[3],lw=1.3);ax.fill_between(x,np.maximum(0,cdf-precision),np.minimum(1,cdf+precision),color=COLORS[3],alpha=.12)
   key="marginal_cdf" if j<21 else "joint_projection_cdf";index=j if j<21 else j-21
   a=assessments["before"][key]["upper"][index];b=assessments["after"][key]["upper"][index];name=order[j] if j<21 else f"fixed joint {j-20}"
   ax.set(title=f"{name}: {a:.3f} → {b:.3f}",xlim=(lo,hi),ylim=(-.02,1.02),xlabel="physical coordinate" if j<21 else "physical-vector projection",ylabel="CDF")
  figs[label]=save(f,"forensic-"+label.replace("_","-")+".png")
 return figs

def main():
 p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest="action",required=True)
 a=sub.add_parser("refresh");b=sub.add_parser("assess")
 for parser in (a,b):
  for name in ("replay-manifest","replay-manifest-sha256","failed-manifest","output"):parser.add_argument("--"+name,required=True)
 for name in ("root","hybrid-manifest","hybrid-manifest-sha256"):b.add_argument("--"+name,required=True)
 args=vars(p.parse_args());action=args.pop("action");(refresh if action=="refresh" else assess)(**args)
if __name__=="__main__":main()
