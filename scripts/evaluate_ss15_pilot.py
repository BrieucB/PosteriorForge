#!/usr/bin/env python3
"""Compact SS15 pilot review: unchanged absolute gate first, real plots second.

Built with PriorLabs-TabPFN. Saved50000 complete rows only; no GPU/model calls,
reference integration, clipping, reordering, output repair or original-gate edit.
"""
from __future__ import annotations
import argparse,ast,copy,hashlib,importlib.util,inspect,json,re,subprocess,textwrap
from pathlib import Path
import numpy as np
from posteriorforge.artifacts import sha256_file,write_json
from posteriorforge.banks import load_context,load_heldout
from posteriorforge.inference import support_mask,CHECKPOINT_SHA256
from posteriorforge.science.analytical_benchmark_v2 import array_descriptor,build_problem
from posteriorforge.validation import assess_samples,compare_to_upstream
from posteriorforge import representation_ss15 as representation

SCHEMA="posteriorforge.ss15_pilot_assessment.v1"
RUN_SCHEMA="posteriorforge.inference.ss15_fp16.v1"
VARIANT="accuracy_ss15_fp16_fifo_v1"
RUNNER_SHA="4099da3c19366997ab0800e5430b5bb1c129b4b3e535f678b3d4652233303480"
QUALIFIER_FILENAME="repair_ss15_conditionals.py"
QUALIFIER_SHA="2ecadae34e5173bebb5127f578233964bfde6820ae38f9ed0baaac054d119816"
REPRESENTATION_SHA="46323c0b94301c2f7e678538881d5cbbdd8bc52c62a60021798176bffe81eb85"
BINDER_SHA="284d3b061696d4d823666e12d3f79c720f87692b40f45743296cfb8e93865424"
RAW_FP16_MANIFEST_SHA="3546de7f867d3aeaefbbf4e1dab03d90fe867e8355506629319a6e9eed492133"
RAW_FP16_SAMPLES_SHA="61d0a058233ab2ff6e12d1ad29b17ef9d80c3fe10cf1f474be4d33bd716f1f11"
PLOT_LABELS={
 "original upstream":"raw33 FP16","CDF arithmetic repair":"SS15 FP16",
 "CDF-only repair: four original worst marginal CDFs\n":"SS15: four raw33 worst marginal CDFs\n",
 "Original-only selection; full figures and unchanged gate retain all coordinates.\n":
  "Raw33-only selection; full figures and unchanged gate retain all coordinates.\n",
 "four highest original baseline marginal-CDF upper bounds; original physical-order index breaks ties":
  "four highest raw33 FP16 marginal-CDF upper bounds; physical-order index breaks ties",
 "cdf-pilot-density-focus.png":"ss15-pilot-density-focus.png","cdf-pilot-density.png":"ss15-pilot-density.png",
 "cdf-pilot-marginal-cdf.png":"ss15-pilot-marginal-cdf.png","cdf-pilot-joint-cdf.png":"ss15-pilot-joint-cdf.png"}

def require(value,message):
 if not value:raise ValueError(message)
def load(name,digest):
 require(isinstance(digest,str) and re.fullmatch("[0-9a-f]{64}",digest) is not None,"new source pin unset: "+name)
 path=Path(__file__).with_name(name);require(sha256_file(path)==digest,"pinned helper changed: "+name)
 spec=importlib.util.spec_from_file_location("_ss15_pilot_"+path.stem,path)
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def bind_inputs(context,query):
 p=build_problem();theta32=context.theta.astype(np.float32)
 ss64=representation.transform_observations(context.obs);q64=representation.transform_observations(query)
 ss32=representation.transform_for_model(context.obs);q32=representation.transform_for_model(query)
 raw={"checkpoint_sha256":CHECKPOINT_SHA256,"scientific_fingerprint":p.scientific_fingerprint,
   "parameter_order":list(p.parameter_order),"context_theta":array_descriptor(context.theta),
   "context_obs":array_descriptor(context.obs),"query":array_descriptor(query)}
 transformed={"context_obs_SS15_float64":array_descriptor(ss64),"query_SS15_float64":array_descriptor(q64),
   "context_obs_SS15_model":array_descriptor(ss32),"query_SS15_model":array_descriptor(q32)}
 model={"context_theta":array_descriptor(theta32),"context_obs":array_descriptor(ss32),"query":array_descriptor(q32)}
 return raw,transformed,model

def producer_source(manifest):
 require(isinstance(RUNNER_SHA,str) and len(RUNNER_SHA)==64,"SS15 producer source pin unset; assessment is fail-closed")
 path=Path(__file__).resolve().parents[1]/"src/posteriorforge/performance_ss15.py"
 require(sha256_file(path)==RUNNER_SHA and manifest.get("runner_source_sha256")==RUNNER_SHA,"SS15 producer source changed")
 revision=manifest.get("git_revision")
 require(isinstance(revision,str) and re.fullmatch("[0-9a-f]{40}",revision) is not None,"full executed Git revision required")
 blob=subprocess.run(["git","-C",str(path.parents[2]),"show",revision+":src/posteriorforge/performance_ss15.py"],
   check=True,capture_output=True).stdout
 require(hashlib.sha256(blob).hexdigest()==RUNNER_SHA,"executed SS15 producer Git bytes differ")
 return {"producer_sha256":RUNNER_SHA,"recorded_execution_commit":revision,"recorded_git_blob_verified":True}

def validate_raw_scientific_binding(manifest,raw):
 # The legacy Gate.verify_run_binding additionally enforces its own upstream
 # seed/microbatch/fit-mode. Those are not this source-qualified family.
 for key,value in raw.items():
  require(manifest.get(key)==value,"raw scientific source binding mismatch: "+key)

def validate_header(manifest,raw,model,descriptor):
 expected={"schema":RUN_SCHEMA,"variant":VARIANT,"status":"complete","context_seed":20261111,"case":0,
   "context_size":4096,"requested_count":50000,"repetition":0,"scientific_run":True,
   "max_iter_rejection":None,"partial_vector_filtering":False,"clipping_or_repair_performed":False,
   "tail_family_changed":False,"original_gate_unchanged":True,"model_inputs":model,
   "sources_unchanged":True,"absolute_accuracy_gate_required":True,"approximate_experiment":True,"qualified_FP32":False,
   "physical_parameter_order_unchanged":True,"all_three_globals_condition_local_stages":True,
   "sampler_math_changed":False,"raw33_numerical_qualification_claimed":False,
   "query_batch_size":1024,"engine_query_batch_cap":1024,
   "representation":descriptor,**raw}
 for key,value in expected.items():
  actual=manifest.get(key)
  equal=(actual is value if type(value) is bool else type(actual) is int and actual==value if type(value) is int else actual==value)
  require(equal,"SS15 pilot header/binding mismatch: "+key)
 require(manifest.get("completed_stage_indices")==list(range(21)),"complete21 physical stages required")

def validate_counts(manifest,binder):
 adapter=binder.validate_adapter_descriptor(manifest.get("compatibility_adapter"))
 counts=manifest["sampling_counters"];waves=manifest["proposal_wave_sizes"]
 require(waves and all(type(n) is int and n>0 for n in waves),"actual generated complete-wave sizes required")
 require(manifest["generated_proposal_wave_count"]==len(waves)
   and manifest["expected_stage_prepare_count"]==21*len(waves),"generated-wave stage count differs")
 wanted=8*counts["stage0_shared_predictions"]+counts["component_predictions"]
 require(counts["stage0_shared_predictions"]==len(waves)
   and manifest["expected_patched_call_count"]==adapter["patched_call_count"]==wanted
   and manifest["actual_prediction_checks"]["actual_head_contexts"]==wanted,"actual8head/shared/CDF coverage differs")
 scopes=manifest["execution_counters"]["completed_stage_scopes"]
 require(len(scopes)==21*len(waves) and [s["stage_index"] for s in scopes]==list(range(21))*len(waves),
   "actual stage/session scope coverage differs")
 require(manifest["execution_counters"]["prepare_count"]==len(scopes)
   and manifest["execution_counters"]["actual_head_contexts"]==sum(s["actual_head_contexts"] for s in scopes)==wanted,
   "actual scope/physical head counts differ")
 events=manifest["stage_events"]
 require(len(events)==len(scopes),"actual stage event/scope counts differ")
 for index,(scope,event) in enumerate(zip(scopes,events)):
  stage=index%21;rows=waves[index//21];sample=event["sampler"]
  require(event["stage_index"]==stage and event["proposed_rows"]==rows,"actual physical wave/stage row identity differs")
  hist=sample["neural_query_batch_sizes_by_member"]
  require(set(hist)=={str(j) for j in range(8)},"actual8member query histogram required")
  heads=[];neural_rows=[]
  for member in range(8):
   table=hist[str(member)]
   require(all(str(int(size))==size and 1<=int(size)<=1024 and type(count) is int and count>0 for size,count in table.items()),
     "actual physical head query size/count invalid")
   heads.append(sum(table.values()));neural_rows.append(sum(int(size)*count for size,count in table.items()))
  require(scope["actual_head_contexts"]==sum(heads)
    and scope["queried_members"]==[j for j in range(8) if heads[j]],"actual per-stage physical head/member counts differ")
  if stage==0:require(heads==[1]*8 and neural_rows==[1]*8,"shared stage0 requires8fresh real singleton heads")
  else:
   require(sum(neural_rows)==sample["head_candidates"]
     and neural_rows==sample["head_assignment_counts_by_member"],"actual component assignment/query rows differ")
 require(manifest["execution_counters"]["all_stage_aliases_restored"] is True
   and all(s["failed"] is False
     and s["precision_session"]["aliases_and_flags_restored"] is True
     and s["precision_session"]["counters"].get("witness_histogram_overflow",0)==0 for s in scopes),
   "actual complete source/precision scope restoration failed")
 require(manifest["default_rng_preserved"] is True and manifest["default_rng_before"]==manifest["default_rng_after"],
   "actual default RNG streams changed")
 rejection=manifest["rejection"]
 require(type(rejection["support_calls"]) is int and rejection["support_calls"]>0
   and rejection["proposed"]>=rejection["accepted_before_final_trim"]>=rejection["returned"]==50000,"original complete-vector support/trim counts differ")
 return {"passed":True,"physical_head_contexts":wanted,"generated_complete_waves":len(waves),
   "stage_prepares":21*len(waves),"rejection":rejection,
   "scope":"independent recorded stage/member/head/CDF/defaultRNG/source-restoration checks",
   "FIFO_and_sampling_streams":"bound to exact reviewed producer/source policy; not separately replayed by this compact evaluator"}

def summary(assessment,order):
 moments=assessment["moments"];intervals=moments["uncertainty_intervals"]
 require(intervals is not None,"original conservative reference moment intervals required")
 bias=np.asarray(intervals["absolute_bias_reference_sd"]["upper"])
 lower=np.asarray(intervals["sd_ratio"]["lower"]);upper=np.asarray(intervals["sd_ratio"]["upper"])
 marginal=np.asarray(assessment["marginal_cdf"]["upper"]);joint=np.asarray(assessment["joint_projection_cdf"]["upper"])
 return {"status":assessment["status"],"passed":assessment["passed"],"reference_qualified":assessment["reference_qualified"],
   "marginal_CDF_upper_max":float(marginal.max()),"marginal_CDF_upper_worst_parameter":order[int(marginal.argmax())],
   "joint_CDF_upper_max":float(joint.max()),"joint_CDF_upper_worst_index":int(joint.argmax()),
   "bias_upper_refSD_max":float(bias.max()),"bias_upper_worst_parameter":order[int(bias.argmax())],
   "SD_ratio_interval_min":float(lower.min()),"SD_ratio_interval_max":float(upper.max()),
   "failed_marginal_CDF_indices":np.flatnonzero(marginal>.1).tolist(),
   "failed_joint_CDF_indices":np.flatnonzero(joint>.1).tolist(),
   "failed_bias_upper_indices":np.flatnonzero(bias>.2).tolist(),
   "failed_SD_ratio_interval_indices":np.flatnonzero((lower<.8)|(upper>1.2)).tolist(),
   "all_rows_used":50000,"reference_uncertainty_included":True}

def plotting_function(binder):
 """Reuse unchanged plotting arithmetic; clone literal labels and marginal footer layout."""
 source=textwrap.dedent(inspect.getsource(binder.plot_comparisons));tree=ast.parse(source);original=copy.deepcopy(tree)
 counts={key:0 for key in PLOT_LABELS};reverse={}
 class Labels(ast.NodeTransformer):
  def visit_Constant(self,node):
   if type(node.value) is str and node.value in PLOT_LABELS:
    counts[node.value]+=1;new=ast.copy_location(ast.Constant(PLOT_LABELS[node.value]),node)
    reverse[id(new)]=node;return new
   return node
 changed=Labels().visit(copy.deepcopy(tree))
 require(all(n==1 for n in counts.values()),"expected frozen plot labels/filenames changed")
 class Undo(ast.NodeTransformer):
  def visit(self,node):
   if id(node) in reverse:return copy.deepcopy(reverse[id(node)])
   return super().visit(node)
 restored=Undo().visit(changed)
 require(ast.dump(restored,include_attributes=False)==ast.dump(original,include_attributes=False),"plot has nonlabel changes")
 reverse.clear();counts={key:0 for key in PLOT_LABELS}
 changed=Labels().visit(copy.deepcopy(tree))
 # Pure layout repair for the marginal-CDF figure: reserve more bottom space
 # and place attribution/display text on two lines. No data/bin/math changes.
 layout=[ast.parse('marginal.set_layout_engine("constrained", rect=(0, .055, 1, .88))').body[0],
   ast.parse('marginal.supxlabel(footer, fontsize=9, y=.008)').body[0]]
 layout[1].value.args[0]=ast.Call(func=ast.Attribute(value=ast.Name(id="footer",ctx=ast.Load()),attr="replace",ctx=ast.Load()),
   args=[ast.Constant(" | "),ast.Constant(chr(10))],keywords=[])
 target=ast.dump(ast.parse('marginal.savefig(figs["marginal_cdf"], dpi=145)').body[0],include_attributes=False)
 statements=changed.body[0].body;positions=[i for i,n in enumerate(statements) if ast.dump(n,include_attributes=False)==target]
 require(len(positions)==1,"one marginal-CDF save call required for layout-only repair")
 statements[positions[0]:positions[0]]=layout
 inserted={ast.dump(n,include_attributes=False) for n in layout};inverse_labels={v:k for k,v in PLOT_LABELS.items()}
 class ReverseAll(ast.NodeTransformer):
  def visit_Expr(self,node):
   if ast.dump(node,include_attributes=False) in inserted:return None
   return self.generic_visit(node)
  def visit_Constant(self,node):
   return ast.copy_location(ast.Constant(inverse_labels[node.value]),node) if type(node.value) is str and node.value in inverse_labels else node
 audit=ReverseAll().visit(copy.deepcopy(changed))
 require(ast.dump(audit,include_attributes=False)==ast.dump(original,include_attributes=False),"plot contains nonlabel/nonlayout changes")
 ast.fix_missing_locations(changed);scope=dict(binder.plot_comparisons.__globals__)
 exec(compile(changed,"<ss15-plot-labels-layout-only>","exec",dont_inherit=True),scope)
 return scope["plot_comparisons"],{"source_helper_sha256":BINDER_SHA,"label_changes":dict(PLOT_LABELS),
  "reverse_AST_exact":True,"reference_bins_ranges_all_rows_math_unchanged":True,
  "pilot_used_for_focus_selection":False,
  "layout_only_changes":{"figure":"marginal_cdf","subplot_rectangle":[0,.055,1,.88],
   "footer_two_lines":True,"footer_y":.008}}

def run(*,root,manifest,qualification,output,plots=True):
 binder=load("evaluate_cdf_pilot.py",BINDER_SHA)
 require(representation.source_sha256()==REPRESENTATION_SHA,"exact SS15 representation source changed")
 root=Path(root).resolve();results=root/"results";gatepath=results/"gate-final.json"
 require(sha256_file(gatepath)==binder.BASELINE_GATE_SHA256,"immutable original physical gate changed")
 gate=json.loads(gatepath.read_bytes());manifest=Path(manifest).resolve();m=json.loads(manifest.read_bytes())
 source=producer_source(m)
 from posteriorforge import performance_ss15 as producer
 require(m["execution_policy"]==producer.execution_metadata(),"actual source-qualified SS15 execution policy differs")
 require(m["conditional_qualification_sha256"]==producer.CONDITIONAL_QUALIFICATION_SHA256
   and QUALIFIER_SHA==producer.QUALIFIER_SHA256,"producer/evaluator immutable matching qualification pins differ")
 banks=results/"banks-v1";context=load_context(banks,20261111,4096)
 query=load_heldout(banks,0).obs.reshape(1,33);raw,transformed,model=bind_inputs(context,query)
 descriptor=representation.representation_metadata();validate_header(m,raw,model,descriptor)
 require(m["raw_scientific_bindings"]==raw,"raw scientific binding duplicate differs")
 validate_raw_scientific_binding(m,raw)
 qhelper=load(QUALIFIER_FILENAME,QUALIFIER_SHA);qpath=Path(qualification)
 require(sha256_file(qpath)==m["conditional_qualification_sha256"],"actual matching SS15 proof SHA changed")
 q=json.loads(qpath.read_bytes())
 require(q["fallback_map"]==m["execution_policy"]["SS15_execution"]["fallback_map"],
   "qualified stage-local FP32 fallback policy differs from production")
 replay=qhelper.validate_qualification(q,raw_bindings=raw,
   transformed_bindings=transformed,representation_descriptor=descriptor,expected_sources=qhelper.source_pins(),
   fallback_map=q["fallback_map"],artifact_root=qpath.parent)
 require(m["qualification_transformed_bindings"]==transformed,"production SS15 float64/model feature descriptors differ")
 require(m["FP16_strict_failed_stages"]==replay["FP16_strict_failed_stages"]
   and m["FP16_matching_uncached_strict_failed_stages"]==replay["FP16_matching_uncached_strict_failed_stages"],
   "production relabelled preserved precision approximation flags")
 counters=validate_counts(m,binder)
 arrays,refmanifest,reference,custody=binder.reference_for_case(results,gate,query,0)
 binder.validate_run_reference_binding(m,custody);values=binder.load_samples(manifest,m)
 baseline_path=results/"performance-first-v1/fp16-fifo-50k-01/rep-0/seed-20261111-case-0/manifest.json"
 require(sha256_file(baseline_path)==RAW_FP16_MANIFEST_SHA,"authenticated raw33 FP16 before cohort changed")
 baseline=json.loads(baseline_path.read_bytes())
 require(baseline["samples_sha256"]==RAW_FP16_SAMPLES_SHA,"raw33 FP16 sample payload changed")
 validate_raw_scientific_binding(baseline,raw)
 binder.validate_run_reference_binding(baseline,custody);before=binder.load_samples(baseline_path,baseline)
 physical=assess_samples(values,reference,support_mask=support_mask(values))
 raw_physical=assess_samples(before,reference,support_mask=support_mask(before))
 require(physical["integrity"]["passed"] and raw_physical["integrity"]["passed"]
   and physical["reference_qualified"] and raw_physical["reference_qualified"],"complete supported cohorts/qualified reference required")
 original=[r for r in gate["runs"] if r["context_seed"]==20261111 and r["case"]==0]
 require(len(original)==1 and physical["limits"]==raw_physical["limits"]==original[0]["assessment"]["limits"],
   "scientific physical limits changed from immutable original gate")
 order=list(build_problem().parameter_order)
 relative=compare_to_upstream(values,before,reference_sd=reference["sd"],directions=reference["directions"])
 relative.update(interpretation="descriptive change from intentionally different learned observation representation",
   runtime_fidelity_veto=False,accuracy_veto=False,new_precision_defect_inferred=False,population_equivalence_inferred=False,
   legacy_threshold_flags_scope="descriptive .01 CDF/.05 referenceSD mean/.05 fractionalSD markers only; scientific decision uses original absolute reference gate")
 output=Path(output).resolve()
 for protected in ("banks-v1","upstream-v1","references-v1","reference-grids-v1"):
  require(not output.is_relative_to(results/protected),"review output may not overwrite original artifacts")
 output.mkdir(parents=True,exist_ok=False)
 report={"schema":SCHEMA,"attribution":"Built with PriorLabs-TabPFN","scope":"firstcase0seed20261111 SS15 run, not complete24-run validation",
   "status":physical["status"],"passed":physical["passed"],"accuracy_gate_required":True,
   "scientific_limits_unchanged":True,"summary":{"raw33_FP16":summary(raw_physical,order),"SS15_FP16":summary(physical,order)},
   "absolute_physical_assessment":physical,"raw33_FP16_physical_assessment":raw_physical,
   "relative_samecase_empirical_change":relative,"relative_change_is_not_accuracy_veto":True,
   "qualification_replay":replay,"counter_validation":counters,"representation":descriptor,
   "reference":custody,"source":source,"evaluator_sha256":sha256_file(__file__),
   "pilot":{"manifest_file":str(manifest),"manifest_sha256":sha256_file(manifest),"samples_sha256":m["samples_sha256"]},
   "before":{"manifest_file":str(baseline_path),"manifest_sha256":RAW_FP16_MANIFEST_SHA,"samples_sha256":RAW_FP16_SAMPLES_SHA},
   "original_gate_sha256":sha256_file(gatepath),"no_model_calls":True,"no_GPU_calls":True,
   "no_reference_integrations":True,"all_original_rows_used":True,"plots_pending":bool(plots)}
 write_json(output/"assessment.json",report)
 print(json.dumps({"output":str(output),"summary":report["summary"]},indent=2),flush=True)
 if plots:
  plot,plotdescriptor=plotting_function(binder)
  figures,display,focus=plot(output,before,values,arrays,{"baseline":raw_physical,"pilot":physical},order,seed=20261111,case=0)
  report.update(figures=figures,plotting_source=plotdescriptor,display=display,focus_selection=focus,plots_pending=False)
 require(sha256_file(gatepath)==binder.BASELINE_GATE_SHA256,"original gate changed during review")
 require(sha256_file(manifest)==report["pilot"]["manifest_sha256"] and sha256_file(qpath)==m["conditional_qualification_sha256"],"run/proof changed during review")
 write_json(output/"assessment.json",report)
 write_json(output/"checksums.json",{p.name:sha256_file(p) for p in output.iterdir() if p.is_file()})
 return report

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 for name in ("root","manifest","qualification","output"):parser.add_argument("--"+name,required=True)
 parser.add_argument("--no-plots",action="store_true")
 args=vars(parser.parse_args());args["plots"]=not args.pop("no_plots")
 run(**args)
if __name__=="__main__":main()
