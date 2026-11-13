"""Pair-bound source/footer/reference/actual-draw algebra tests, no new worlds.

Synthetic kernel tests use a declared FIXED Z matrix (no normal draws). The
optional existing case2 regression only reads old saved arrays and bank data.
"""
from copy import deepcopy
import ast,hashlib,importlib.util,json,pickle,random
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from posteriorforge import prior_coordinate_pair_v1 as pairs,prior_coordinates_v1 as codec
from posteriorforge.science.analytical_benchmark_v2 import array_descriptor
from posteriorforge.artifacts import sha256_file
ROOT=Path(__file__).resolve().parents[1]

def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
w=load("_pair_v2_downstream_w",ROOT/"scripts/refresh_prior_pair_v2_global_block.py")
e=load("_pair_v2_downstream_e",ROOT/"scripts/evaluate_prior_pair_v2.py")
SPEC=pairs.PairSpec(20261111,0)
PROOF="1"*64

def footer(spec=SPEC):
 states={k:"same" for k in ("numpy","python","torch_CPU","torch_CUDA")}
 return {"schema":w.PRODUCER_SCHEMA,"variant":w.PRODUCER_VARIANT,"status":"complete",
  "runner_source_sha256":w.PRODUCER_SHA256,"pair_spec":spec.as_dict(),"context_seed":spec.context_seed,"case":spec.case,
  "context_size":4096,"requested_count":50000,"repetition":0,"rank":0,"world_size":1,
  "mode":"fp32_transparent_control_v1","native_all8_stages":[6,10,14],"FP16_activated":False,
  "numerical_candidate_changed":True,"execution_routing_changed_relative_v1":True,"fixed_case2_qualification_inherited":False,
  "native_protocol_sha256":w.PROTOCOL_SHA256,
  "sources_unchanged":True,"default_rng_preserved":True,"completed_stage_indices":list(range(21)),
  "samples_in_original_physical_order_before_FIFO_support":True,
  "all_generated_rows_strictly_decoded_including_unused_FIFO_tail":True,
  "model_coordinate_decode_after_global_refresh":False,"sampler_math_changed":False,"partial_vector_filtering":False,
  "clipping_or_repair_performed":False,"physical_log_density_claimed":False,"learned_targets_are_original_physical_values":False,
  "fallback_map":{},"fallback_map_scope":"INNER frozen88 precision owner only; outer native all8 stages6/10/14 are explicit",
  "matching_conditional_qualification_sha256":PROOF,"matching_cached_native_FP32_execution_passed_all21":True,
  "fresh_same_policy_all21_screen_replayed":True,"actual_fixed_conditional_screen_performed_during_request":False,
  "default_rng_before":states.copy(),"default_rng_after":states.copy(),"execution_counters":{},"sampling_counters":{},
  "actual_prediction_checks":{"actual_head_contexts":1138,"sampler_callback_calls":1131,"shared_densities":1},
  "timings":{"posterior_request_seconds":100.,"includes_stage_preparation":True,"includes_CPU_sampling":True,
   "includes_strict_all_generated_wave_decode":True},"memory":{"peak_allocated_bytes":100,"peak_reserved_bytes":200},
  "expected_patched_call_count":1138,"compatibility_adapter":{"original_alias_restored":True,"sdk_sources_unchanged":True,
   "sdk_aliases_unchanged":True,"patched_call_count":1138}}

def test_header_uses_actual_counts_only_footer_and_explicit_pair():
 d=footer();assert "passed" not in d["actual_prediction_checks"]
 assert w.validate_header(d,SPEC,PROOF)["pair_spec"]==SPEC.as_dict()

@pytest.mark.parametrize("fault",["case","seed","pairspec","booltype","family","source","count","oldproof","invalidproof",
 "RNG","missingRNG","CDF","headcount","stage","FP16","fixture","timing","memory","fallback"])
def test_wrong_pair_proof_source_or_normal_footer_rejected(fault):
 d=footer();proof=PROOF
 if fault=="case":d["case"]=1
 elif fault=="seed":d["context_seed"]=20261112
 elif fault=="pairspec":d["pair_spec"]["case"]=1
 elif fault=="booltype":d["pair_spec"]["case"]=False
 elif fault=="family":d["schema"]=w.BASE.PRODUCER_SCHEMA
 elif fault=="source":d["runner_source_sha256"]="0"*64
 elif fault=="count":d["requested_count"]=49999
 elif fault=="oldproof":proof=w.OLD_CASE2_SCREEN_SHA256;d["matching_conditional_qualification_sha256"]=proof
 elif fault=="invalidproof":proof="not a SHA"
 elif fault=="RNG":d["default_rng_after"]["torch_CUDA"]="changed"
 elif fault=="missingRNG":d.pop("default_rng_after")
 elif fault=="CDF":d["compatibility_adapter"]["original_alias_restored"]=False
 elif fault=="headcount":d["actual_prediction_checks"]["actual_head_contexts"]-=1
 elif fault=="stage":d["completed_stage_indices"].pop()
 elif fault=="FP16":d["FP16_activated"]=True
 elif fault=="fixture":d["test_fixture"]={"CPU":"only"}
 elif fault=="timing":d["timings"]["includes_strict_all_generated_wave_decode"]=False
 elif fault=="memory":d["memory"]["peak_allocated_bytes"]=0
 else:d["fallback_map"]={"14":list(range(8))}
 with pytest.raises(ValueError):w.validate_header(d,SPEC,proof)

def canonical():
 return {"pair_spec":SPEC.as_dict(),"old_case2_screen_inherited":False,"FP16_activated":False,
  "numerical_candidate_changed":True,"execution_routing_changed_relative_v1":True,
  "neural_weights_codec_sampler_support_math_changed":False,"inherited_v1_numerical_qualification":False,
  "derivation":{"live_code_identity":True,"source_AST_sha256":"a"*64,
   "top_run_codeobject_identity_with_a30":False,"only_two_run_metadata_literals_changed":True,
   "original_run_controlflow_bytecode_equal":True,"original_run_names_varnames_freevars_equal":True,
   "all_nested_run_codeobjects_identical":True,"metadata_literals_reversible":True,
   "original_adapter_method_codeobjects_identical":True,"private_adapter_backend_source_policy_globals":True},
  "inherited_numerical_execution":{"derivation":{"run_code_telemetry_scope":"excluded"}},
  "runtime_source_inventory":{"sources":{"unchanged":"b"*64}}}

def test_pair_canonical_source_identity_has_no_marshaled_code_guard():
 d=canonical();assert w.verify_canonical_policy(d,deepcopy(d))["all_canonical_metadata_fields_exact_TYPE_and_value"]
 d["derivation"]["run_code_sha256"]="c"*64
 with pytest.raises(ValueError,match="serialized-code"):w.verify_canonical_policy(d,deepcopy(d))

@pytest.mark.parametrize("fault",["pair","source","AST","route","type"])
def test_all_canonical_metadata_fields_value_and_type_remain_strict(fault):
 a=canonical();b=deepcopy(a)
 if fault=="pair":b["pair_spec"]["case"]=1
 elif fault=="source":b["runtime_source_inventory"]["sources"]["unchanged"]="c"*64
 elif fault=="AST":b["derivation"]["source_AST_sha256"]="d"*64
 elif fault=="route":b["native_stages"]=[14]
 else:b["pair_spec"]["case"]=False
 with pytest.raises(ValueError,match="canonical"):w.verify_canonical_policy(a,b)

def test_source_pins_none_block_real_request_before_any_draw(monkeypatch):
 monkeypatch.setattr(w,"PRODUCER_SHA256",None);monkeypatch.setattr(e,"WRAPPER_SHA256",None)
 with pytest.raises(ValueError,match="unpinned"):w.source_guard()
 with pytest.raises(ValueError,match="unpinned"):e.source_guard()

def test_v2_preserves_exact_unmodified_function_codeobjects_and_routing_reverse_AST():
 assert w.execution_checks.__code__ is w._ANCESTOR.execution_checks.__code__
 assert w.refresh_with_timing.__code__ is w._ANCESTOR.refresh_with_timing.__code__
 assert w.refresh.__code__ is w._ANCESTOR.refresh.__code__
 assert e.run.__code__ is e._RUN_CODE and e.run.__code__.co_code==e._ANCESTOR.run.__code__.co_code
 assert e.validate_reference_custody.__code__ is e._ANCESTOR.validate_reference_custody.__code__
 receipt=w.derivation_descriptor()
 assert receipt["unchanged_functions_same_code_objects"] and receipt["old_globals_unmodified"]
 assert receipt["numerical_G3_seed_or_metric_math_changed"] is False
 assert w._ANCESTOR.SCHEMA=="posteriorforge.prior_coordinate_pair_global_refresh.v1"
 assert e._ANCESTOR.SCHEMA=="posteriorforge.prior_coordinate_pair_assessment.v1"

class FixedZGenerator(np.random.Generator):
 """Test-only deterministic provider; NEVER calls a normal/random method."""
 def __init__(self,bit_generator):
  super().__init__(bit_generator);self.standard_normal_calls=0;self.actual_draws=None
 def standard_normal(self,size):
  assert self.standard_normal_calls==0
  self.standard_normal_calls+=1
  self.actual_draws=np.linspace(-1.,1.,np.prod(size)).reshape(size)
  return self.actual_draws

def hybrid_fixture(tmp_path,monkeypatch,spec=SPEC):
 monkeypatch.setattr(w,"COUNT",7);monkeypatch.setattr(e,"COUNT",7)
 monkeypatch.setattr(w,"_RECORDING_GENERATOR",FixedZGenerator)
 # Replace ONLY our new namespace alias; never mutate frozen BASE globals.
 monkeypatch.setattr(w,"BASE",SimpleNamespace(verify_git_sources=lambda *args:{"passed":True},
  policy_differences=w.BASE.policy_differences,immutable_pins=w.BASE.immutable_pins))
 before=codec.decode_ordered(np.zeros((7,21),np.float32));original=before.tobytes()
 default=pickle.dumps((np.random.get_state(),random.getstate()),protocol=4)
 after,z,receipt=w.refresh_with_timing(before,spec)
 assert original==before.tobytes() and default==pickle.dumps((np.random.get_state(),random.getstate()),protocol=4)
 np.savez(tmp_path/"samples.npz",samples=after);np.savez(tmp_path/"Z.npz",standard_normal_draws=z)
 pins={"wrapper":"f"*64,"git_revision":"a"*40}
 d={"schema":w.SCHEMA,"status":"complete","forensic":False,"original_parent_status":"complete",
  "pair_spec":spec.as_dict(),"context_seed":spec.context_seed,"case":spec.case,"repetition":0,"requested_count":7,
  "parent":{"parent_manifest_sha256":"parent","pair_spec":spec.as_dict()},"source_pins":pins,
  "conditional_kernel":w.kernel.metadata(),"coordinate_decode_after_global_refresh":False,
  "samples_file":"samples.npz","samples_sha256":sha256_file(tmp_path/"samples.npz"),"samples":array_descriptor(after),
  "actual_draws_file":"Z.npz","actual_draws_sha256":sha256_file(tmp_path/"Z.npz"),"refresh":receipt}
 return before,d

def authenticate(tmp_path,before,d,spec=SPEC):
 p=tmp_path/"manifest.json";p.write_text(json.dumps(d))
 return e.authenticate_hybrid(p,sha256_file(p),"parent",before,w,{"wrapper":"f"*64,"git_revision":"b"*40},spec)

def test_fixed_Z_G3_pair_seed_algebra_physical18_timing_and_no_assessment_draw(tmp_path,monkeypatch):
 before,d=hybrid_fixture(tmp_path,monkeypatch)
 def forbidden(*args,**kwargs):raise AssertionError("no new normal draw in assessment")
 monkeypatch.setattr(FixedZGenerator,"standard_normal",forbidden)
 _,after=authenticate(tmp_path,before,d)
 assert after[:,3:].tobytes()==before[:,3:].tobytes()
 assert d["refresh"]["seed"]==w.derive_seed(SPEC.context_seed,w.ROLE,case=SPEC.case,repetition=0)
 assert d["refresh"]["standard_normal_method_calls"]==1 and d["refresh"]["G3_kernel_math_seconds"]>=0

@pytest.mark.parametrize("fault",["case","seed","role","parent","state","RNG","physical18","kernel","source","Z","decode","timing","calltype"])
def test_actual_saved_G3_wrong_pair_or_control_fails_without_reroll(tmp_path,monkeypatch,fault):
 before,d=hybrid_fixture(tmp_path,monkeypatch)
 if fault=="case":d["pair_spec"]["case"]=1
 elif fault=="seed":d["refresh"]["seed"]+=1
 elif fault=="role":d["refresh"]["role"]="other"
 elif fault=="parent":d["parent"]["pair_spec"]["case"]=1
 elif fault=="state":d["refresh"]["initial_private_state"]["state"]["state"]+=1
 elif fault=="RNG":d["refresh"]["default_numpy_python_RNG_after_sha256"]="changed"
 elif fault=="physical18":
  with np.load(tmp_path/"samples.npz") as saved:rows=saved["samples"]
  rows[0,3]+=1.;np.savez(tmp_path/"samples.npz",samples=rows)
  d["samples_sha256"]=sha256_file(tmp_path/"samples.npz");d["samples"]=array_descriptor(rows)
 elif fault=="kernel":d["conditional_kernel"]["source_sha256"]="wrong"
 elif fault=="source":d["source_pins"]["wrapper"]="0"*64
 elif fault=="Z":d["refresh"]["standard_normal_draws"]["sha256"]="wrong"
 elif fault=="decode":d["refresh"]["coordinate_decode_after_global_refresh"]=True
 elif fault=="calltype":d["refresh"]["standard_normal_method_calls"]=True
 else:d["refresh"]["G3_kernel_math_seconds"]=float("nan")
 with pytest.raises(ValueError):authenticate(tmp_path,before,d)

def reference_fixture():
 query=np.arange(33,dtype=np.float64).reshape(1,33);binding={"bound":"actual"}
 parent={"context_seed":SPEC.context_seed,"case":SPEC.case,"pair_spec":SPEC.as_dict(),"reference_binding":binding}
 custody={"reference_case":SPEC.case,"binding":{"observation_sha256":array_descriptor(query)["sha256"]},"runner_reference_binding":binding}
 return parent,custody,query

def test_reference_pair_custody_explicit_no_fixed_case2():
 p,c,q=reference_fixture();assert e.validate_reference_custody(p,c,q,SPEC)["reference_case"]==0

@pytest.mark.parametrize("fault",["case","seed","reference","query","binding"])
def test_wrong_seed_case_query_or_reference_is_rejected(fault):
 p,c,q=reference_fixture()
 if fault=="case":p["case"]=2
 elif fault=="seed":p["context_seed"]=20261112
 elif fault=="reference":c["reference_case"]=2
 elif fault=="query":q=q.copy();q[0,0]+=1
 else:c["runner_reference_binding"]={"bound":"other"}
 with pytest.raises(ValueError):e.validate_reference_custody(p,c,q,SPEC)

def test_actual_case2_bank_latent_target_mask_seed_regression_readonly():
 s=Path("/scratch/project/eu-26-17/eubrieucb/posteriorforge")
 manifest=s/"results/accuracy-prior-coordinates-v1/native-pilot-corrected-01/manifest.json"
 if not manifest.exists():pytest.skip("existing read-only Karolina case2 fixture is not attached")
 assert sha256_file(manifest)=="62aa8bf71ddbf922ab9543141c4ada5cb438e5af162ede76cffe60d1dd3819b5"
 parent=json.loads(manifest.read_bytes())
 from posteriorforge import performance_prior_physical_fifo_v1 as old
 context,rawquery,raw=pairs.raw_inputs(pairs.REGRESSION_PAIR,banks=s/"results/banks-v1")
 theta,stats,query,model,encoding=old.failed.model_bindings(context,rawquery)
 assert raw==parent["raw_scientific_bindings"] and model==parent["model_inputs"]
 for key in ("physical_theta_original_Float64","latent_physical_slot_model_Float32","ordered_latent_model_Float32","stage_to_physical_order","physical_to_stage_order"):
  assert encoding[key]==parent["model_target_descriptors"][key]
 for stage,row in enumerate(parent["execution_counters"]["actual_latent_training_inputs"]):
  assert old.failed.backend.training_bindings(theta,stats,stage)==row
 seed=w.derive_seed(20261111,w.ROLE,case=2,repetition=0)
 assert seed==2551793397
 # Actual old case2 family/proof is retained, never relabelled as a new pair.
 with pytest.raises(ValueError):w.validate_header(parent,pairs.REGRESSION_PAIR,w.OLD_CASE2_SCREEN_SHA256)

def test_v1_positive_proof_family_and_v1_G3_cannot_be_promoted_to_v2(tmp_path,monkeypatch):
 d=footer();d["schema"]=w._ANCESTOR.PRODUCER_SCHEMA;d["variant"]=w._ANCESTOR.PRODUCER_VARIANT
 with pytest.raises(ValueError):w.validate_header(d,SPEC,PROOF)
 for proof in w.INELIGIBLE_V1_PROOFS:
  d=footer();d["matching_conditional_qualification_sha256"]=proof
  with pytest.raises(ValueError,match="v1 proof"):w.validate_header(d,SPEC,proof)
 before,m=hybrid_fixture(tmp_path,monkeypatch);m["schema"]=w._ANCESTOR.SCHEMA
 with pytest.raises(ValueError):authenticate(tmp_path,before,m)

def test_v1_two_native_list_is_not_v2_routing():
 d=footer();d["native_all8_stages"]=[6,10]
 with pytest.raises(ValueError,match="native_all8_stages"):w.validate_header(d,SPEC,PROOF)

def test_v2_display_declares_cached18_and_original_all8_6_10_14():
 assert e.FOCUS==(0,7,8,10)
 assert "18 cached" in e.LABELS[1] and "6/10/14" in e.LABELS[1]
 assert "SEPARATE physical G3" in e.LABELS[2]
 assert e.plots.__code__ is not e._ANCESTOR.plots.__code__  # declared display-only AST deltas
 assert w.PROTOCOL_SHA256=="8640a4e157698862823521d3fd14396ecc3ebf121dffffabc987cc937d3ad2fc"
 assert w.PAIR_DATA_PROTOCOL_SHA256=="9fc6a91568864cc996dafbcfaa41e77280e4825c63f61d455bdadf3703cd0a66"

@pytest.mark.parametrize("field",["numerical_candidate_changed","execution_routing_changed_relative_v1",
 "neural_weights_codec_sampler_support_math_changed","inherited_v1_numerical_qualification"])
def test_canonical_v2_requires_explicit_routing_change_not_v1_credit(field):
 d=canonical();d[field]=not d[field]
 with pytest.raises(ValueError,match="active v2"):w.verify_canonical_policy(d,deepcopy(d))

def test_v2_derivation_rejects_invented_original_top_codeobject_identity():
 d=canonical();d["derivation"]["top_run_codeobject_identity_with_a30"]=True
 with pytest.raises(ValueError,match="v2 code/source"):w.verify_canonical_policy(d,deepcopy(d))

def test_active_v2_native_protocol_is_new8640_not_ancestral_v1():
 d=footer();d["native_protocol_sha256"]="82b"+"0"*61
 with pytest.raises(ValueError,match="explicit routing"):w.validate_header(d,SPEC,PROOF)
 d=footer();d["native_protocol_sha256"]=w.PAIR_DATA_PROTOCOL_SHA256
 with pytest.raises(ValueError):w.validate_header(d,SPEC,PROOF)

def test_new_parent_draw_zero_is_scoped_not_old_baseline_rng_method_calls():
 def strings(values):
  for value in values:
   if isinstance(value,str):yield value
   elif isinstance(value,tuple):yield from strings(value)
 constants=set(strings(e.run.__code__.co_consts))
 assert "new_parent_global_or_normal_draws_during_assessment" in constants
 assert "new_global_or_normal_draws_during_assessment" not in constants
 assert "OLD baseline custody replays" in e.__doc__
