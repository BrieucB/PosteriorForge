"""CPU tests for compact SS15 assessment bindings, criteria and plot labels."""
import ast,copy,importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("ss15_pilot",ROOT/"scripts/evaluate_ss15_pilot.py")
review=importlib.util.module_from_spec(spec);spec.loader.exec_module(review)

def test_new_producer_and_qualification_pins_are_fail_closed_until_frozen(monkeypatch):
 monkeypatch.setattr(review,"RUNNER_SHA",None)
 with pytest.raises(ValueError,match="pin unset"):review.producer_source({})
 with pytest.raises(ValueError,match="pin unset"):review.load("qualify_ss15_conditionals.py",None)

def test_raw_scientific_binding_separate_from_15_feature_model_binding():
 context=SimpleNamespace(theta=np.zeros((4096,21),np.float64),obs=np.zeros((4096,33),np.float64))
 query=np.zeros((1,33),np.float64);raw,transformed,model=review.bind_inputs(context,query)
 assert raw["context_theta"]["dtype"]=="<f8" and raw["context_obs"]["shape"]==[4096,33]
 assert model["context_theta"]["dtype"]==model["context_obs"]["dtype"]=="<f4"
 assert model["context_obs"]["shape"]==[4096,15] and model["query"]["shape"]==[1,15]
 assert transformed["context_obs_SS15_float64"]["shape"]==[4096,15]
 assert transformed["context_obs_SS15_model"]==model["context_obs"]
 assert raw["query"]["sha256"]!=model["query"]["sha256"]

def header():
 raw={"checkpoint_sha256":"checkpoint","scientific_fingerprint":"science","parameter_order":list(range(21)),
      "context_theta":{},"context_obs":{},"query":{}}
 model={"context_theta":{"shape":[4096,21]},"context_obs":{"shape":[4096,15]},"query":{"shape":[1,15]}}
 descriptor={"source":"ss15"}
 m={"schema":review.RUN_SCHEMA,"variant":review.VARIANT,"status":"complete","context_seed":20261111,"case":0,
 "context_size":4096,"requested_count":50000,"repetition":0,"scientific_run":True,
 "max_iter_rejection":None,"partial_vector_filtering":False,"clipping_or_repair_performed":False,
 "tail_family_changed":False,"original_gate_unchanged":True,"model_inputs":model,
 "sources_unchanged":True,"absolute_accuracy_gate_required":True,"approximate_experiment":True,"qualified_FP32":False,
 "physical_parameter_order_unchanged":True,"all_three_globals_condition_local_stages":True,"sampler_math_changed":False,
 "raw33_numerical_qualification_claimed":False,"query_batch_size":1024,"engine_query_batch_cap":1024,
 "representation":descriptor,"completed_stage_indices":list(range(21)),**raw}
 return m,raw,model,descriptor

@pytest.mark.parametrize("key,value",[("case",1),("requested_count",49999),("partial_vector_filtering",True),
 ("context_obs",{"shape":[4096,15]}),("model_inputs",{"context_obs":{"shape":[4096,33]}}),("completed_stage_indices",[0])])
def test_mismatched_case_count_raw_model_order_or_partial_output_rejected(key,value):
 m,raw,model,descriptor=header();review.validate_header(m,raw,model,descriptor)
 m[key]=value
 with pytest.raises(ValueError):review.validate_header(m,raw,model,descriptor)

def test_summary_reports_conservative_upper_bias_and_full_sd_intervals():
 a={"status":"model_failure","passed":False,"reference_qualified":True,
  "marginal_cdf":{"upper":[.08]*20+[.1001]},"joint_projection_cdf":{"upper":[.099]*31+[.1002]},
  "moments":{"uncertainty_intervals":{"absolute_bias_reference_sd":{"upper":[.1]*20+[.2001]},
   "sd_ratio":{"lower":[.81]*20+[.799],"upper":[1.1]*20+[1.201]}}}}
 r=review.summary(a,[str(i) for i in range(21)])
 assert r["bias_upper_refSD_max"]==.2001 and r["failed_bias_upper_indices"]==[20]
 assert r["failed_marginal_CDF_indices"]==[20] and r["failed_joint_CDF_indices"]==[31]
 assert r["failed_SD_ratio_interval_indices"]==[20] and r["reference_uncertainty_included"]
 assert r["all_rows_used"]==50000

def test_plot_clone_changes_labels_only_never_math_display_or_row_selection():
 binder=review.load("evaluate_cdf_pilot.py",review.BINDER_SHA)
 clone,descriptor=review.plotting_function(binder)
 assert descriptor["reverse_AST_exact"] and descriptor["reference_bins_ranges_all_rows_math_unchanged"]
 assert not descriptor["pilot_used_for_focus_selection"]
 constants=[v for item in clone.__code__.co_consts for v in (item if isinstance(item,tuple) else (item,))]
 assert "raw33 FP16" in constants and "SS15 FP16" in constants
 assert "original upstream" not in constants and "CDF arithmetic repair" not in constants
 assert "ss15-pilot-density-focus.png" in constants
 assert clone is not binder.plot_comparisons

def test_actual_original_query_reference_binding_uses_saved_2d_shape():
 import json
 from posteriorforge.banks import load_heldout
 phase=Path("/scratch/project/eu-26-17/eubrieucb/posteriorforge/results")
 if not phase.is_dir():pytest.skip("immutable reference artifacts unavailable")
 binder=review.load("evaluate_cdf_pilot.py",review.BINDER_SHA)
 gate=json.loads((phase/"gate-final.json").read_bytes())
 query=load_heldout(phase/"banks-v1",0).obs
 assert query.shape==(1,33)
 arrays,manifest,reference,custody=binder.reference_for_case(phase,gate,query,0)
 assert arrays["directions"].shape==(53,21) and custody["reference_case"]==0
 assert custody["binding"]["observation_sha256"]==review.array_descriptor(query)["sha256"]

def test_model_binding_agrees_with_actual_root_producer_input_preparation():
 from posteriorforge import performance_ss15 as producer
 context=SimpleNamespace(theta=np.zeros((4096,21),np.float64),obs=np.zeros((4096,33),np.float64))
 query=np.zeros((1,33),np.float64)
 raw,transformed,model=review.bind_inputs(context,query)
 _,_,_,actual_model,actual_qualification=producer.model_bindings(context,query)
 assert actual_model==model and actual_qualification==transformed

def counter_manifest(waves=(50000,)):
 scopes=[];events=[];total=0;components=0
 for n in waves:
  assignments=[n//8+(i<n%8) for i in range(8)]
  for stage in range(21):
   hist={}
   for member in range(8):
    size=1 if stage==0 else assignments[member]
    if stage==0:table={"1":1}
    else:
     blocks,remainder=divmod(size,1024);table={}
     if blocks:table["1024"]=blocks
     if remainder:table[str(remainder)]=table.get(str(remainder),0)+1
    hist[str(member)]=table
   heads=[sum(hist[str(i)].values()) for i in range(8)]
   contexts=sum(heads);total+=contexts
   if stage:components+=contexts
   scopes.append({"stage_index":stage,"actual_head_contexts":contexts,
     "queried_members":[i for i in range(8) if heads[i]],"failed":False,
     "precision_session":{"aliases_and_flags_restored":True,"counters":{}}})
   events.append({"stage_index":stage,"proposed_rows":n,"sampler":{
     "neural_query_batch_sizes_by_member":hist,"head_candidates":0 if stage==0 else n,
     "head_assignment_counts_by_member":[0]*8 if stage==0 else assignments}})
 return {"compatibility_adapter":{"patched_call_count":total},"proposal_wave_sizes":list(waves),
   "sampling_counters":{"stage0_shared_predictions":len(waves),"component_predictions":components},
   "generated_proposal_wave_count":len(waves),"expected_stage_prepare_count":21*len(waves),
   "expected_patched_call_count":total,"actual_prediction_checks":{"actual_head_contexts":total},
   "execution_counters":{"completed_stage_scopes":scopes,"prepare_count":len(scopes),
     "actual_head_contexts":total,"all_stage_aliases_restored":True},"stage_events":events,
   "default_rng_preserved":True,"default_rng_before":{"state":"a"},"default_rng_after":{"state":"a"},
   "rejection":{"support_calls":len(waves),"proposed":sum(waves),"accepted_before_final_trim":50000,"returned":50000}}

def test_independent_counts_accept_real_member_subsets_in_small_refill_metadata():
 binder=SimpleNamespace(validate_adapter_descriptor=lambda descriptor:descriptor)
 m=counter_manifest((50000,7));r=review.validate_counts(m,binder)
 assert r["physical_head_contexts"]==1276 and r["stage_prepares"]==42
 assert m["execution_counters"]["completed_stage_scopes"][-1]["queried_members"]==list(range(7))

@pytest.mark.parametrize("bad",["cdf","head","rows","members","rng","restoration","failed","stage0"])
def test_independent_counter_replay_rejects_mismatched_records(bad):
 binder=SimpleNamespace(validate_adapter_descriptor=lambda descriptor:descriptor);m=counter_manifest()
 if bad=="cdf":m["compatibility_adapter"]["patched_call_count"]-=1
 elif bad=="head":m["execution_counters"]["completed_stage_scopes"][1]["actual_head_contexts"]-=1
 elif bad=="rows":m["stage_events"][1]["sampler"]["neural_query_batch_sizes_by_member"]["0"]["106"]=2
 elif bad=="members":m["execution_counters"]["completed_stage_scopes"][1]["queried_members"]=[1]
 elif bad=="rng":m["default_rng_after"]["state"]="b"
 elif bad=="restoration":m["execution_counters"]["completed_stage_scopes"][1]["precision_session"]["aliases_and_flags_restored"]=False
 elif bad=="failed":m["execution_counters"]["completed_stage_scopes"][1]["failed"]=True
 else:m["stage_events"][0]["sampler"]["neural_query_batch_sizes_by_member"]["0"]={"2":1}
 with pytest.raises(ValueError):review.validate_counts(m,binder)

def test_real_completed_raw_fp16_control_uses_own_source_family_not_old_upstream_gate():
 import json
 from posteriorforge.banks import load_context,load_heldout
 root=Path("/scratch/project/eu-26-17/eubrieucb/posteriorforge/results")
 path=root/"performance-first-v1/fp16-fifo-50k-01/rep-0/seed-20261111-case-0/manifest.json"
 if not path.is_file():pytest.skip("completed actualFP16 artifact unavailable")
 m=json.loads(path.read_bytes())
 assert review.sha256_file(path)==review.RAW_FP16_MANIFEST_SHA
 context=load_context(root/"banks-v1",20261111,4096);query=load_heldout(root/"banks-v1",0).obs
 raw,_,_=review.bind_inputs(context,query)
 review.validate_raw_scientific_binding(m,raw)
 assert m["microbatch"]==50000
 # Actual FP16 path uses its worker-role stream seed; legacy gate's required
 # posterior seed/microbatch are unrelated to matching physical data custody.
 bad=dict(m);bad["query"]={"shape":[1,15]}
 with pytest.raises(ValueError,match="query"):review.validate_raw_scientific_binding(bad,raw)

def test_actual_map2_assembly_consumer_uses_authenticated_archived_inputs():
 import json
 root=Path("/scratch/project/eu-26-17/eubrieucb/posteriorforge/results")
 path=root/"accuracy-ss15-v1/conditional-qualification-map2-assembled-01/manifest.json"
 if not path.is_file():pytest.skip("actual matching SS15 assembly unavailable")
 assert review.sha256_file(path)=="b192d05e68d59a3518ea9df5c6c8c4f36845fa89418b30d404261615a25d5116"
 context=review.load_context(root/"banks-v1",20261111,4096);query=review.load_heldout(root/"banks-v1",0).obs
 raw,transformed,model=review.bind_inputs(context,query)
 helper=review.load(review.QUALIFIER_FILENAME,review.QUALIFIER_SHA)
 q=json.loads(path.read_bytes())
 receipt=helper.validate_qualification(q,raw_bindings=raw,transformed_bindings=transformed,
  representation_descriptor=review.representation.representation_metadata(),expected_sources=helper.source_pins(),
  fallback_map={"10":[5],"20":[4]},artifact_root=path.parent)
 assert receipt["passed"] and receipt["stages"]==21 and receipt["original_status_preserved"]
 assert receipt["distinct_CDF_contexts"]==2 and receipt["FP16_strict_failed_stages"]==list(range(21))
 assert not receipt["FP16_qualified_as_FP32"] and not receipt["model_predictions_reexecuted"]
