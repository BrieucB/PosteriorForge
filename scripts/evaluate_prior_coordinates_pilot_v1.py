#!/usr/bin/env python3
"""Read-only physical assessment of the prior-coordinate cached-FP32 pilot.

Built with PriorLabs-TabPFN. Replays archived full-wave decode, physical G3,
actual-case bindings and finite execution screen, then unchanged physical gates.
No model, posterior refresh, GPU or new reference integration.
Run via the producer inference Python for exact same-version decode and G3 replay.
"""
from __future__ import annotations
import argparse,hashlib,importlib.util,json
from collections import Counter
from importlib import import_module
from pathlib import Path
import numpy as np
from posteriorforge.artifacts import sha256_file,write_json
from posteriorforge.banks import load_context,load_heldout
from posteriorforge.science.analytical_benchmark_v2 import array_descriptor

SCHEMA="posteriorforge.prior_coordinates_pilot_assessment.v1"
WRAPPER_FILENAME="refresh_prior_coordinates_global_block_v1.py"
WRAPPER_SHA256=None
PARENT_SOURCE_SHA256=None
FROZEN_EVALUATOR_SHA256="146c2ca35e15b54b7f68134d383574485ed1cc228f38e6fc611624f00b411924"
BASELINE_ASSESSMENT_SHA256="4590a8acff216737d04b8c6144d72df0a14c0246ca5333a330ad365a07a857cb"
COMPLETED_GRID_SHA256="6136487ee78375a802c8a0cb0c225a7f1b207063b4083514614c0d97a35c2425"
DISPLAY_DOC_SHA256="f588c842f6b6e6d2a83afcb46fd839b23d5a2de5014b632dae07796f11dd0974"
FOCUS=(0,7,8,10)
COLORS=("#c58a37","#777777","#1479b8","#111111")
LABELS=("OLD physical-target mixed FP16/FP32 + G3",
        "NEW causal prior coordinates: cached FP32",
        "NEW same physical parent + one G3 block","qualified reference")


def require(value,message):
    if not value:raise ValueError(message)


def load(filename,digest):
    path=Path(__file__).with_name(filename)
    require(isinstance(digest,str) and len(digest)==64 and sha256_file(path)==digest,
            "source not pinned or changed: "+filename)
    spec=importlib.util.spec_from_file_location("_prior_eval_"+path.stem,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def source_guard():
    w=load(WRAPPER_FILENAME,WRAPPER_SHA256)
    require(w.PARENT_SOURCE_SHA256==PARENT_SOURCE_SHA256,"new parent source binding differs")
    pins=w.source_guard()
    display=Path(__file__).resolve().parents[1]/"docs/accuracy-prior-coordinate-pilot-display-v1.md"
    require(sha256_file(display)==DISPLAY_DOC_SHA256,"predeclared display contract changed")
    root=Path(__file__).resolve().parents[1];self_path=Path(__file__).resolve()
    require(hashlib.sha256(w.backbone.git_bytes(root,pins["git_revision"],
            str(self_path.relative_to(root)))).hexdigest()==sha256_file(self_path),
            "new evaluator must be committed before assessment")
    frozen=load("evaluate_ss15_sigma_native_campaign.py",FROZEN_EVALUATOR_SHA256)
    require(hashlib.sha256(w.backbone.git_bytes(root,pins["git_revision"],
        "docs/accuracy-prior-coordinate-pilot-display-v1.md")).hexdigest()==DISPLAY_DOC_SHA256,
        "display contract must be committed before assessment")
    return w,frozen,{"evaluator_sha256":sha256_file(__file__),"wrapper_sha256":WRAPPER_SHA256,
        "parent_producer_sha256":PARENT_SOURCE_SHA256,"git_revision":pins["git_revision"],
        "kernel_sha256":w.KERNEL_SHA256,"frozen_physical_assessment_helper_sha256":FROZEN_EVALUATOR_SHA256,
        "display_contract_sha256":DISPLAY_DOC_SHA256}


def authenticate(w,frozen,parent_path,hybrid_path,expected_hybrid_sha):
    require(w.backbone.digest(expected_hybrid_sha) and sha256_file(hybrid_path)==expected_hybrid_sha,
            "explicit hybrid manifest SHA differs")
    hybrid=json.loads(Path(hybrid_path).read_bytes())
    require(hybrid.get("schema")==w.SCHEMA and hybrid.get("variant")==w.VARIANT
            and hybrid.get("status")=="complete" and hybrid.get("sources_unchanged") is True,
            "only completed new prior-coordinate joint cohort accepted")
    parent,before,lineage=w.load_parent(parent_path,hybrid["parent"]["manifest_sha256"],w.kernel.metadata())
    environment=w.require_replay_environment(parent)
    w.require_replay_environment({"versions":hybrid["runtime"]["versions"]})
    require(hybrid.get("exact_decode_and_G3_replay_environment")==environment,
            "saved G3 exact-replay environment differs from actual producer")
    require(hybrid["parent"]==lineage and all(hybrid[k]==parent[k] for k in
        ("context_seed","case","repetition","parameter_order","scientific_fingerprint"))
        and hybrid["raw_scientific_bindings"]==parent["raw_scientific_bindings"]
        and hybrid["conditional_kernel"]==w.kernel.metadata()
        and hybrid.get("coordinate_decode_after_global_refresh") is False,
        "physical-only joint lineage differs")
    pins=hybrid["source_pins"]
    require(pins["producer_sha256"]==WRAPPER_SHA256
        and pins["parent_producer_sha256"]==PARENT_SOURCE_SHA256
        and pins["kernel_sha256"]==w.KERNEL_SHA256 and pins["sources_committed_before_run"] is True,
        "hybrid source custody differs")
    root=Path(__file__).resolve().parents[1]
    for filename,digest in [("scripts/"+WRAPPER_FILENAME,WRAPPER_SHA256),
        ("scripts/refresh_ss15_global_block.py",w.BACKBONE_SHA256),
        ("src/posteriorforge/conditional_globals.py",w.KERNEL_SHA256)]:
        require(hashlib.sha256(w.backbone.git_bytes(root,pins["git_revision"],filename)).hexdigest()==digest,
                "executed hybrid source/Git bytes differ")
    payload=Path(hybrid_path).with_name("samples.npz")
    require(hybrid["samples_file"]=="samples.npz" and sha256_file(payload)==hybrid["samples_sha256"],
            "hybrid physical payload hash differs")
    with np.load(payload,allow_pickle=False) as archive:
        require(archive.files==["samples"],"hybrid physical sample keys differ");after=archive["samples"]
    require(array_descriptor(after)==hybrid["samples"],"hybrid array descriptor differs")
    auth=frozen.replay_joint(w,parent,hybrid,before,after)
    return parent,hybrid,before,after,auth


def model_guard(model):
    require(model["canonical_borders_dtype"]==model["physical_borders_dtype"]=="torch.float32"
        and model["temperature"]==.9 and model["contract"]["input_normalization_train_only_verified"],
        "actual latent borders/temperature/train-only contract differs")
    for module in model["contract"]["modules"]:
        if module["type"]=="InputNormalizationEncoderStep":
            require(all(t["dtype"]=="torch.float32" for t in module["tensors"].values()),
                    "normalizer tensors left FP32")


def execution_checks(parent,producer,actual,helper):
    backend=producer.backend
    theta,obs=actual["theta"],actual["obs"]
    execution=parent["execution_counters"];scopes=execution["completed_stage_scopes"]
    training=execution["actual_latent_training_inputs"];calls=execution["actual_latent_query_calls"]
    require(len(scopes)==len(training)==len(parent["stage_events"])==21
        and execution["prepare_count"]==21 and execution["all_stage_aliases_restored"] is True,
        "actual all21 stage scopes differ")
    old=helper.load("qualify_ss15_conditionals.py",helper.SS15_QUALIFIER_SHA256)
    cursor=0
    for stage,(scope,binding,event) in enumerate(zip(scopes,training,parent["stage_events"])):
        require(scope["stage_index"]==stage and scope["failed"] is False
                and scope["engine_release_succeeded"] is True
                and binding==backend.training_bindings(theta,obs,stage),
                "actual latent training mask/target or owner release differs")
        helper.witness_guard(scope["precision_session"],producer.MODE,stage,old)
        model_guard(scope["model"])
        require(scope["prefill"]["passed"] is True,"actual prefill witness failed")
        observed=Counter()
        subset=calls[cursor:cursor+scope["actual_head_contexts"]]
        for call in subset:
            shape=call["query_unmasked"]["shape"];masked=call["query_masked"]
            require(call["stage"]==stage and call["execution_route"]=="resident_cache"
                and call["policy_sha256"]==backend.policy_metadata()["policy_sha256"]
                and call["mask"]==backend.stage_policy(stage)
                and call["actual_target_coordinate"]==backend.stage_policy(stage)["target_coordinate"]
                and shape==masked["shape"] and shape[1]==15+stage
                and call["query_unmasked"]["dtype"]==masked["dtype"]=="<f4"
                and call["witness"]["passed"] is True
                and call["witness"]["mode"]==producer.MODE,
                "actual query policy/shape/dtype/route witness differs")
            if stage<3:require(call["query_unmasked"]==masked,"global query input unexpectedly masked")
            observed[(call["member"],shape[0])]+=1
        hist=event["sampler"]["neural_query_batch_sizes_by_member"]
        expected=Counter((member,int(rows)) for member in range(8)
                         for rows,count in hist[str(member)].items() for _ in range(count))
        require(observed==expected,"actual per-member query rows/head counts differ")
        routes=scope["route_counters"]
        require(routes["resident_component_predict_count"]==sum(expected.values())
                and routes["uncached_predict_count"]==0,"cached-only route conservation differs")
        cursor+=scope["actual_head_contexts"]
    require(cursor==len(calls)==execution["actual_head_contexts"],"actual query journal coverage differs")
    frozen=producer.clean.frozen_sampling()
    seeds=frozen.worker_stream_seeds(20261111,2,0,0,1)
    require(parent["seed_streams"]==seeds and parent["proposal_buffer_policy"]==frozen.proposal_buffer_policy(1200),
            "original seed roles/FIFO policy differs")
    frozen.validate_proposal_buffer_record(parent["proposal_buffer"],world_size=1,rank=0,
        scope=producer.proposal_scope(frozen,actual["raw"],parent["model_inputs"],seeds))
    counters=parent["sampling_counters"]
    heads=8*counters["stage0_shared_predictions"]+counters["component_predictions"]
    require(heads==len(calls)==parent["expected_patched_call_count"]
        ==parent["compatibility_adapter"]["patched_call_count"]
        ==parent["actual_prediction_checks"]["actual_head_contexts"]
        and counters["stage0_shared_predictions"]==1 and execution["shared_densities"]==1,
        "actual physical head/CDF/shared density count differs")
    require(parent["default_rng_preserved"] is True
        and parent["default_rng_before"]==parent["default_rng_after"],"actual default RNG changed")
    return {"passed":True,"actual_training_masks_recomputed":True,"actual_query_policy_shape_dtype_routes_verified":True,
        "query_value_arrays_independently_reconstructed":False,
        "scope":"source-bound actual query descriptors and independent head/shape conservation",
        "stage_count":21,"actual_head_contexts":heads,"fallback_map":{},"mode":producer.MODE}


def load_baseline(frozen,completed_grid,parent_path,hybrid_path,assessment_path,raw):
    require(sha256_file(completed_grid)==COMPLETED_GRID_SHA256,"completed old24 receipt differs")
    receipt=json.loads(Path(completed_grid).read_bytes())
    points=[r for r in receipt["points"] if (r["context_seed"],r["case"])==(20261111,2)]
    require(len(points)==1 and points[0]["technical_passed"] is True and points[0]["count"]==50000,
            "fixed old case2 display point absent")
    point=points[0]
    require(sha256_file(parent_path)==point["parent_manifest_sha256"]
            and sha256_file(assessment_path)==point["assessment_sha256"]==BASELINE_ASSESSMENT_SHA256,
            "old display parent/assessment differs from completed grid")
    old_wrapper=frozen.load("refresh_ss15_campaign_global_block.py",frozen.WRAPPER_SHA)
    parent,hybrid,_,values,auth=frozen.authenticate(
        old_wrapper,parent_path,hybrid_path,point["hybrid_manifest_sha256"])
    require(parent["raw_scientific_bindings"]==raw and values.shape==(50000,21),
            "old display is not same physical bank/query/case")
    assessment=json.loads(Path(assessment_path).read_bytes())
    require(assessment["context_seed"]==20261111 and assessment["case"]==2
            and assessment["all50000_rows_used"] is True,
            "old display assessment scope differs")
    return values,assessment,{"completed_grid_sha256":COMPLETED_GRID_SHA256,
        "parent_manifest_sha256":point["parent_manifest_sha256"],
        "hybrid_manifest_sha256":point["hybrid_manifest_sha256"],
        "hybrid_samples_sha256":hybrid["samples_sha256"],
        "assessment_sha256":BASELINE_ASSESSMENT_SHA256,
        "old_physical_array":array_descriptor(values),"authentication":auth,
        "scope":"display and recorded comparison only; no new old-cohort metrics or selection"}


def run(*,root,parent_manifest,hybrid_manifest,hybrid_manifest_sha256,execution_screen,
        codec_inventory,codec_reference_inventory,codec_combined_inventory,completed_grid,prefix_run,
        baseline_assessment,baseline_parent_manifest,baseline_hybrid_manifest,output,plots_enabled=True):
    w,frozen,sources=source_guard()
    parent,hybrid,before,after,auth=authenticate(
        w,frozen,parent_manifest,hybrid_manifest,hybrid_manifest_sha256)
    producer=import_module("posteriorforge.performance_prior_coordinates_v1")
    require(producer.source_sha256()==PARENT_SOURCE_SHA256,"current new producer source differs")
    producer.require_sources()
    results=Path(root)/"results";banks=results/"banks-v1"
    context=load_context(banks,20261111,4096);raw_query=load_heldout(banks,2).obs.reshape(1,33)
    theta,obs,query,model,encoding=producer.model_bindings(context,raw_query)
    raw={"checkpoint_sha256":w.backbone.CHECKPOINT_SHA,
        "scientific_fingerprint":w.kernel.metadata()["scientific_fingerprint"],
        "parameter_order":w.kernel.metadata()["physical_parameter_order"],
        "context_theta":array_descriptor(context.theta),"context_obs":array_descriptor(context.obs),
        "query":array_descriptor(raw_query)}
    require(raw==parent["raw_scientific_bindings"] and model==parent["model_inputs"],
            "actual bank/raw query or exact model-FP32 values differ")
    actual={"theta":theta,"obs":obs,"query":raw_query,"raw":raw}
    for key in ("physical_theta_original_Float64","latent_physical_slot_model_Float32",
                "ordered_latent_model_Float32","stage_to_physical_order","physical_to_stage_order"):
        require(encoding[key]==parent["model_target_descriptors"][key],
                "actual physical/source/model target binding differs: "+key)
    helper=producer.load_script(producer.QUALIFIER_FILENAME,producer.QUALIFIER_SHA256)
    _,screen_replay=producer.qualification_receipt(
        execution_screen,banks=banks,prefix_run=prefix_run,codec_inventory=codec_inventory,
        codec_reference_inventory=codec_reference_inventory,codec_combined_inventory=codec_combined_inventory,
        completed_grid=completed_grid,raw_bindings=raw,model_inputs=model)
    execution=execution_checks(parent,producer,actual,helper)
    binder=frozen.load("evaluate_cdf_pilot.py",frozen.BINDER_SHA)
    pure=frozen.load("evaluate_ss15_pilot.py",frozen.PURE_HELPER_SHA)
    gatepath=results/"gate-final.json"
    require(sha256_file(gatepath)==binder.BASELINE_GATE_SHA256,"immutable physical gate differs")
    gate=json.loads(gatepath.read_bytes())
    arrays,reference,assessments,controls=frozen.reference_and_assess(
        results,gate,parent,raw_query,before,after,binder)
    baseline_values,baseline,baseline_custody=load_baseline(
        frozen,completed_grid,baseline_parent_manifest,baseline_hybrid_manifest,baseline_assessment,raw)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    record={"schema":SCHEMA,"attribution":"Built with PriorLabs-TabPFN","context_seed":20261111,"case":2,
        "status":assessments["after"]["status"],"passed":assessments["after"]["passed"],
        "source_pins":sources,"parent_manifest_sha256":hybrid["parent"]["manifest_sha256"],
        "hybrid_manifest_sha256":hybrid_manifest_sha256,"hybrid_samples_sha256":hybrid["samples_sha256"],
        "authentication":auth,"full_generated_decode_replay":hybrid["parent"]["full_generated_decode_replay"],
        "execution_checks":execution,"fresh_finite_execution_screen_replay":screen_replay,
        "reference":reference,"absolute_physical_assessments":assessments,"unchanged18_control":controls,
        "summary":{k:pure.summary(v,parent["parameter_order"]) for k,v in assessments.items()},
        "configuration_comparison":{"baseline_assessment_sha256":BASELINE_ASSESSMENT_SHA256,
            "baseline_summary":baseline["summary"],"baseline_custody":baseline_custody,"baseline":"physical targets; mixed FP16/FP32 with native14",
            "candidate":"causal prior coordinates; cached FP32 all heads",
            "both_coordinates_and_precision_changed":True,"isolated_coordinate_causal_effect_claimed":False},
        "pure_NPE_claimed":False,"method":"prior-coordinate cached-FP32 NPE-PFN plus one physical G3 block",
        "current_environment_F64_runtime_provenance":encoding,
        "F64_latent_bytes_required_equal_to_producer_environment":False,
        "exact_decode_G3_and_assessment_bound_to_actual_producer_environment":True,
        "global_cross_environment_F64_byte_identity_claimed":False,
        "all50000_rows_used":True,"scientific_limits_unchanged":True,
        "source_execution_qualification_is_not_posterior_accuracy":True,
        "no_new_posterior_refresh_model_GPU_or_reference_integration":True,
        "original_gate_sha256":sha256_file(gatepath),"plots_pending":plots_enabled}
    write_json(output/"assessment.json",record)
    print(json.dumps({"output":str(output),"case":2,"status":record["status"],
                      "passed":record["passed"],"summary":record["summary"]},sort_keys=True),flush=True)
    if plots_enabled:
        record["figures"],record["focus_selection"]=plots(
            output,before,after,arrays,assessments,parent["parameter_order"],binder,LABELS,
            seed=20261111,case=2,full_cdf=True,baseline=baseline_values)
    record["plots_pending"]=False
    require(sha256_file(hybrid_manifest)==hybrid_manifest_sha256
        and sha256_file(parent_manifest)==hybrid["parent"]["manifest_sha256"]
        and sha256_file(gatepath)==binder.BASELINE_GATE_SHA256,"assessment inputs changed")
    write_json(output/"assessment.json",record)
    write_json(output/"checksums.json",{p.name:sha256_file(p) for p in output.iterdir() if p.is_file()})
    return record


def plots(output,before,after,arrays,assessments,order,binder,labels,*,seed,case,full_cdf=True,baseline):
 import matplotlib
 matplotlib.use("Agg")
 import matplotlib.pyplot as plt
 from matplotlib.lines import Line2D
 output=Path(output);legend=[Line2D([0],[0],color=c,lw=1.8,label=l) for c,l in zip(COLORS,labels)]
 footer="Built with PriorLabs-TabPFN\nReference-selected display; all50000 rows normalize each histogram/CDF; unchanged whole-line gate."
 def figure(rows,cols,size,title):
  f,ax=plt.subplots(rows,cols,figsize=size,squeeze=False)
  f.suptitle(title+f"\nseed{seed}, case{case}; NEW causal prior coordinates / cached FP32; OLD physical targets / mixed FP16-FP32 + G3",fontsize=13,y=.995)
  f.legend(handles=legend,loc="upper center",bbox_to_anchor=(.5,.955),ncol=2,fontsize=9)
  f.text(.5,.012,footer,ha="center",va="bottom",fontsize=8)
  f.subplots_adjust(left=.07,right=.99,bottom=.07,top=.89,wspace=.28,hspace=.55)
  return f,ax.flat
 def domain(j):
  x=arrays["queries"]/arrays["directions"][j,j] if j<21 else arrays["queries"]
  cdf=arrays["cdfs"][j];lo=binder.reference_quantile(x,cdf,.001);hi=binder.reference_quantile(x,cdf,.999)
  pad=.03*(hi-lo);return x,cdf,lo-pad,hi+pad
 def density(ax,j):
  x,cdf,lo,hi=domain(j);bins=np.linspace(lo,hi,41);width=np.diff(bins)
  ax.stairs(np.diff(np.interp(bins,x,cdf))/width,bins,color=COLORS[3],lw=1.4)
  for a,color in ((baseline,COLORS[0]),(before,COLORS[1]),(after,COLORS[2])):
   count,_=np.histogram(a[:,j],bins);ax.stairs(count/(len(a)*width),bins,color=color,lw=1.1)
  old=assessments["before"]["marginal_cdf"]["upper"][j];new=assessments["after"]["marginal_cdf"]["upper"][j]
  ax.set(title=f"{order[j]}: CDF upper {old:.3f} → {new:.3f}",xlim=(lo,hi),xlabel="physical coordinate",ylabel="binned density")
 def save(f,name):
  path=output/name;f.savefig(path,dpi=155);plt.close(f)
  return {"file":name,"sha256":sha256_file(path)}
 figs={}
 focus,axes=figure(2,2,(12,8.6),"Prior-coordinate cached-FP32 hybrid: four fixed physical coordinates")
 focus.subplots_adjust(top=.81,bottom=.11,hspace=.5,wspace=.27)
 for ax,j in zip(axes,FOCUS):density(ax,j)
 figs["focus_density"]=save(focus,"joint-global-density-focus.png")
 print(json.dumps({"first_focus_density_ready":str(output/figs["focus_density"]["file"])}),flush=True)
 densityfig,axes=figure(7,3,(14,20),"Prior-coordinate parent and one physical G3 update: all21 densities")
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
   for a,color in ((baseline,COLORS[0]),(before,COLORS[1]),(after,COLORS[2])):
    values=a[:,j] if j<21 else a@arrays["directions"][j];values=np.sort(values)
    ax.plot(grid,np.searchsorted(values,grid,side="right")/len(values),color=color,lw=1.1)
   ax.plot(x,cdf,color=COLORS[3],lw=1.3);ax.fill_between(x,np.maximum(0,cdf-precision),np.minimum(1,cdf+precision),color=COLORS[3],alpha=.12)
   key="marginal_cdf" if j<21 else "joint_projection_cdf";index=j if j<21 else j-21
   old=assessments["before"][key]["upper"][index];new=assessments["after"][key]["upper"][index]
   name=order[j] if j<21 else f"fixed joint {j-20}"
   ax.set(title=f"{name}: {old:.3f} → {new:.3f}",xlim=(lo,hi),ylim=(-.02,1.02),xlabel="physical coordinate" if j<21 else "physical-vector projection",ylabel="CDF")
  figs[label]=save(f,"joint-global-"+label.replace("_","-")+".png")
 return figs,{"indices":list(FOCUS),"rule":"predeclared known-baseline display coordinates from separate source-pinned display document","hybrid_used_for_selection":False,"display_contract_sha256":DISPLAY_DOC_SHA256,
   "histogram_bins":40,"normalization_rows":50000,"display_range":"saved reference0.001..0.999 plus3%; no assessment truncation"}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("root","parent-manifest","hybrid-manifest","hybrid-manifest-sha256","execution-screen",
                 "codec-inventory","codec-reference-inventory","codec-combined-inventory","completed-grid",
                 "prefix-run","baseline-assessment","baseline-parent-manifest","baseline-hybrid-manifest","output"):
        parser.add_argument("--"+name,required=True)
    parser.add_argument("--no-plots",action="store_true")
    options=vars(parser.parse_args());options["plots_enabled"]=not options.pop("no_plots")
    record=run(**options)
    print(json.dumps({"status":record["status"],"passed":record["passed"],"summary":record["summary"]},sort_keys=True))


if __name__=="__main__":main()
