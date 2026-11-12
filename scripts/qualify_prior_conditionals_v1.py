#!/usr/bin/env python3
"""Fresh prior-latent fixed-conditional execution screen. Built with PriorLabs-TabPFN.

Zero posterior vectors, no reference integrations. New codec/input/cache identity;
no physical-target/native-stage numerical pass is inherited. Cached FP32 is the
candidate; FP16 is a separately labeled comparison only.
"""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,os,random,re,time
from contextlib import ExitStack
from pathlib import Path
from types import FunctionType
import numpy as np
from posteriorforge.artifacts import sha256_file,write_json
from posteriorforge.science.analytical_benchmark_v2 import array_descriptor

SCHEMA="posteriorforge.prior_coordinates_conditional_screen.v1"
BACKEND_SHA256="88f28c5a1f01c086661243c5d3bfb59631eb3c95fe3567b7f1ede325ddba0aea"
CODEC_SHA256="d8ce6c49e7b65c6ee55c59b7243d4ce943cbea28c92ebed0329798074aaa70c5"
INVENTORY_HELPER_SHA256="ce68e237f15d239f99f091c2310cf15791d1cf7f7cf7442c2db029241834a636"
INVENTORY_SHA256="f3266ab6fd36887c95291020b2cdd2d2a5529f2e6c984b32de22a28b9d31da04"
REFERENCE_INVENTORY_SHA256="93ddc495a3c57c2741608437ff566199fe0b052a297c9f5e9228fb5fb4c25387"
COMBINED_CPU_SHA256="d9facb9256a2c975d745ed6e9a6cb68e4346a875976ebd6e749509113d2650a0"
COMPLETED_GRID_SHA256="6136487ee78375a802c8a0cb0c225a7f1b207063b4083514614c0d97a35c2425"
PREREG_SHA256="7c4aca04c4571960108c0296ec5d4770940ce0b51fa0a77cdd0c27182b4ed849"
SS15_QUALIFIER_SHA256="0c3dd7274d6741bba2ba0ae2744542865f48b2f887a6831e1d2aa6bce47f78c0"
SS15_REPRESENTATION_SHA256="46323c0b94301c2f7e678538881d5cbbdd8bc52c62a60021798176bffe81eb85"
COMPLETION_SCHEMA="posteriorforge.sigma_native_baseline_grid_completion.v1"
BASELINE_SOURCES={
 "campaign":"3965e36206d7f750c66350df54b6d855d6c9f8f4f81a0e069aeeb478492cd453",
 "first":"e322d56492476be1acf5fcdb9afd30b57e2a69a373098417a2bf8944b51ba73b",
 "native":"cf135637d073439039cea3e82a98e533cd161b095aac488ea92588aacb9a1461",
 "refresh":"fc3c99732c39e96b0e704d28b90323a523337c9c560828656b0bb5972d747a4f",
 "evaluator":"146c2ca35e15b54b7f68134d383574485ed1cc228f38e6fc611624f00b411924",
 "qualification":"bedfa67dc8a1e7f41a8536e4c8091643dc05d309b0da0367dc92136c19603705"}
STAGES=tuple(range(21))
NOISE_STAGES=(3,7,11,15,17,19)
ROW_COUNTS=(1,1024)
ARMS=("cached_FP32_vs_original_FP32","cached_FP16_vs_original_FP32")
THRESHOLDS={"normalization_absolute":1e-5,"max_cdf_absolute":2e-5,
 "max_row_total_variation":1e-4,"quantile_coordinate_span_fraction":2e-6}
GUARDS=("sources_unchanged","identical_latent_training_inputs","latent_target_bytes_unchanged",
 "same_train_query_mask","model_RNG_preserved","corrected_CDF_executed","aliases_restored","original_FP32_state_guard")
ARM_GUARDS=("FP32_statistics_normalizers_borders","target_statistics_preserved","cached_contract_preserved",
 "actual_dtype_witnesses","preprocessor_identity_preserved")

def require(value,message):
 if not value:raise ValueError(message)

def digest(value):return isinstance(value,str) and re.fullmatch("[0-9a-f]{64}",value) is not None

def load(name,expected):
 path=Path(__file__).with_name(name);require(digest(expected) and sha256_file(path)==expected,"unfrozen/changed helper: "+name)
 s=importlib.util.spec_from_file_location("_prior_conditions_"+path.stem,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def source_pins():
 from posteriorforge import execution_prior_coordinates_v1 as backend,prior_coordinates_v1 as codec,representation_ss15 as rep
 require(digest(BACKEND_SHA256) and backend.source_sha256()==BACKEND_SHA256,"new backend not frozen")
 require(codec.source_sha256()==backend.CODEC_SHA256==CODEC_SHA256,"new codec binding differs");backend.require_sources()
 require(rep.source_sha256()==SS15_REPRESENTATION_SHA256,"exact SS15 source changed")
 inventory=load("qualify_prior_coordinates_v1.py",INVENTORY_HELPER_SHA256)
 old=load("qualify_ss15_conditionals.py",SS15_QUALIFIER_SHA256)
 require(sha256_file(Path(__file__).resolve().parents[1]/"docs/accuracy-prior-coordinate-pilot-v1.md")==PREREG_SHA256,"preregistered candidate contract changed")
 return {"preregistration":PREREG_SHA256,"qualifier":sha256_file(__file__),"backend":BACKEND_SHA256,"codec":CODEC_SHA256,
  "SS15_representation":SS15_REPRESENTATION_SHA256,"numeric_comparison_helper":SS15_QUALIFIER_SHA256,
  "numeric_utility":old.UTILITY_SHA,"numeric_hard_guards":old.BASE_SHA,"codec_inventory_helper":INVENTORY_HELPER_SHA256,
  "prior_and_order_sources":inventory.source_pins()["prior_and_order_sources"]}

def validate_grid_document(document):
 seeds=[20261111,20261112,20261113];cases=list(range(8));pairs=[[s,c] for s in seeds for c in cases]
 expected={"schema":COMPLETION_SCHEMA,"status":"complete","context_seeds":seeds,"cases":cases,"point_count":24,
  "requested_count_per_point":50000,"completed_pairs":pairs,"all_requests_complete":True,"all_assessments_closed":True,
  "all_integrity_and_reference_checks_passed":True,"all_GPU_allocations_reconciled":True,
  "reserved_GPU_seconds_at_completion":0,"baseline_sources":BASELINE_SOURCES}
 for k,v in expected.items():require(document.get(k)==v,"old fixed24 completion differs: "+k)
 rows=document.get("points",[]);require([[r["context_seed"],r["case"]] for r in rows]==pairs,"exact fixed24 point records required")
 for row in rows:
  require(row.get("count")==50000 and row.get("technical_passed") is True,"closed complete valid50000 required")
  require(all(digest(row.get(k)) for k in ("parent_manifest_sha256","hybrid_manifest_sha256","assessment_sha256")),"point custody hashes required")
  require(isinstance(row.get("physical_status"),str),"retain physical outcome per point")
 require(type(document.get("physical_accuracy_gate_passed")) is bool
  and document["physical_accuracy_gate_passed"]==all(r["physical_status"]=="pass" for r in rows),"physical outcomes must be retained, not required to pass")
 return {"passed":True,"points":24,"physical_accuracy_gate_passed":document["physical_accuracy_gate_passed"]}

def grid_receipt(path):
 require(digest(COMPLETED_GRID_SHA256) and sha256_file(path)==COMPLETED_GRID_SHA256,"old24 completion receipt not pinned/closed; new GPU disabled")
 result=validate_grid_document(json.loads(Path(path).read_bytes()))
 return {**result,"manifest_sha256":COMPLETED_GRID_SHA256}

def inventory_receipt(path,reference_path,combined_path):
 require(digest(INVENTORY_SHA256) and sha256_file(path)==INVENTORY_SHA256,"runtime codec inventory not pinned/passed")
 require(sha256_file(reference_path)==REFERENCE_INVENTORY_SHA256 and sha256_file(combined_path)==COMBINED_CPU_SHA256,"both CPU inventories and combined receipt required")
 helper=load("qualify_prior_coordinates_v1.py",INVENTORY_HELPER_SHA256)
 runtime=json.loads(Path(path).read_bytes());reference=json.loads(Path(reference_path).read_bytes());combined=json.loads(Path(combined_path).read_bytes())
 runtime_replay=helper.validate_inventory(runtime);helper.validate_inventory(reference)
 require(combined.get("schema")=="posteriorforge.prior_coordinates_CPU_receipt.v1"
  and combined.get("status")=="codec_domain_preflight_pass_ONLY"
  and combined["source_pins"]==runtime["source_pins"]==reference["source_pins"]
  and combined["codec_prior_order_tolerance_metadata"]==runtime["codec_metadata"]==reference["codec_metadata"],
  "CPU prior/source/codec receipt differs")
 require(combined["environments"]["inference"]["manifest_raw_sha256"]==INVENTORY_SHA256
  and combined["environments"]["reference"]["manifest_raw_sha256"]==REFERENCE_INVENTORY_SHA256
  and combined["Float32_model_and_ordered_bytes_match_for_all_inventory_groups"] is True
  and combined["Float64_latent_bytes_differ_between_SciPy_versions"] is True,"cross-environment scope differs")
 for a,b in zip(runtime["banks"]+[runtime["case2_fixed_prefix"]],reference["banks"]+[reference["case2_fixed_prefix"]]):
  for key in ("latent_model_Float32","ordered_model_Float32"):
   require(a["qualification"][key]==b["qualification"][key],"both CPU inventories must match exact model Float32 bytes")
 return runtime,{**runtime_replay,"manifest_sha256":INVENTORY_SHA256,"reference_manifest_sha256":REFERENCE_INVENTORY_SHA256,
  "combined_CPU_receipt_sha256":COMBINED_CPU_SHA256,"helper_sha256":INVENTORY_HELPER_SHA256,
  "model_Float32_bytes_match_across_environments":True,"Float64_latent_byte_identity_claimed":False}

def transformed_binding_descriptors(theta,stats,query,ss64,q64,inventory):
 bank=next(r for r in inventory["banks"] if r["context_seed"]==20261111)["qualification"]
 fixed=inventory["case2_fixed_prefix"]["qualification"]
 return {"context_ordered_latent_model_Float32":array_descriptor(theta),
  "context_SS15_Float64":array_descriptor(ss64),"query_SS15_Float64":array_descriptor(q64),
  "context_SS15_model_Float32":array_descriptor(stats),"query_SS15_model_Float32":array_descriptor(query),
  "fixed_prefix_ordered_latent_model_Float32":fixed["ordered_model_Float32"],
  "archived_runtime_inventory_context_latent_Float64":bank["latent_Float64"],
  "archived_runtime_inventory_prefix_latent_Float64":fixed["latent_Float64"]}

def input_bindings(banks,prefix_run,inventory,*,runtime_audit=None):
 from posteriorforge.banks import load_context,load_heldout
 from posteriorforge.inference import CHECKPOINT_SHA256
 from posteriorforge import prior_coordinates_v1 as codec,representation_ss15 as rep
 helper=load("qualify_prior_coordinates_v1.py",INVENTORY_HELPER_SHA256)
 context=load_context(banks,20261111,4096);raw_query=load_heldout(banks,2).obs.reshape(1,33)
 prefix,custody=helper.load_prefix(prefix_run,context,raw_query)
 latent64=codec.encode_physical(context.theta);theta=codec.order_latents(codec.encode_for_model(context.theta))
 prefix64=codec.encode_physical(prefix);prefixes=codec.order_latents(codec.encode_for_model(prefix))
 ss64=rep.transform_observations(context.obs);q64=rep.transform_observations(raw_query)
 stats=rep.transform_for_model(context.obs);query=rep.transform_for_model(raw_query)
 bank=next(row for row in inventory["banks"] if row["context_seed"]==20261111)["qualification"]
 fixed=inventory["case2_fixed_prefix"]["qualification"]
 for values,key in ((context.theta,"physical_Float64"),(theta,"ordered_model_Float32")):
  require(array_descriptor(values)==bank[key],"new training values differ from actual codec inventory: "+key)
 for values,key in ((prefix,"physical_Float64"),(prefixes,"ordered_model_Float32")):
  require(array_descriptor(values)==fixed[key],"fixed FIRST1024 differs from codec inventory: "+key)
 problem=codec._geometry()["problem"]
 raw={"checkpoint_sha256":CHECKPOINT_SHA256,"scientific_fingerprint":problem.scientific_fingerprint,
  "parameter_order":list(problem.parameter_order),"context_theta":array_descriptor(context.theta),
  "context_obs":array_descriptor(context.obs),"query":array_descriptor(raw_query)}
 if runtime_audit is not None:
  import platform,scipy
  require(isinstance(runtime_audit,dict),"explicit runtime input-audit mapping required")
  runtime_audit.update(context_latent_physical_slots_Float64=array_descriptor(latent64),
   fixed_prefix_latent_physical_slots_Float64=array_descriptor(prefix64),
   environment={"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__},
   Float64_latent_cross_environment_byte_identity_claimed=False,
   model_Float32_exactly_matches_runtime_inventory=True,
   scope="actual qualifier environment recomputation; separate from archived runtime-inventory provenance and common Float32 model bindings")
 transformed=transformed_binding_descriptors(theta,stats,query,ss64,q64,inventory)
 return theta,stats,query,prefixes,raw,transformed,custody

def fixed_queries(query,prefixes,theta,stage,backend):
 main=np.ascontiguousarray(np.concatenate((np.repeat(query,1024,axis=0),prefixes[:,:stage]),axis=1))
 stress=np.ascontiguousarray(np.concatenate((np.repeat(query,8,axis=0),theta[np.linspace(0,4095,8,dtype=int),:stage]),axis=1))
 kept=backend.stage_policy(stage)["ordered_latent_prefix_indices_kept"]
 noise=[s for s in NOISE_STAGES if s<stage and s in kept]
 if stage:
  if noise:stress[:4,15+noise[0]]=np.asarray([-3.,-2.,2.,3.],np.float32)
  stress[4,15:]=0;stress[5,15:]=-3;stress[6,15:]=3;stress[7,15:]=theta[:,:stage].mean(0)+3*theta[:,:stage].std(0)
  backend.codec_module().decode_ordered_prefix(stress[:,15:])
 return main,stress,noise

def masked_inputs(theta,stats,main,stress,stage,backend):
 mt,ms=backend.mask_training(theta,stats,stage);mq=backend.mask_query(main,stage);sq=backend.mask_query(stress,stage)
 binding=backend.training_bindings(theta,stats,stage)
 require(mt[:,stage:].tobytes()==theta[:,stage:].tobytes(),"latent labels/future target columns changed")
 return mt,ms,mq,sq,{"training":binding,"main":array_descriptor(mq),"stress":array_descriptor(sq),
  "actual_training":array_descriptor(np.concatenate((ms,mt[:,:stage]),axis=1)),"actual_latent_target":array_descriptor(mt[:,stage])}

def witness_guard(report,mode,stage,old=None):
 if old is None:old=load("qualify_ss15_conditionals.py",SS15_QUALIFIER_SHA256)
 # Exact old numeric witness logic, NEW source identity, explicit EMPTY map.
 namespace=dict(old.witness_guard.__globals__);namespace["ENGINE_SHA"]=BACKEND_SHA256
 check=FunctionType(old.witness_guard.__code__,namespace,old.witness_guard.__name__,old.witness_guard.__defaults__,old.witness_guard.__closure__)
 check.__kwdefaults__=old.witness_guard.__kwdefaults__;check(report,mode=mode,stage=stage,fallback_map={})
 require(report.get("codec_sha256")==CODEC_SHA256 and report.get("fallback_map")=={}
  and all(w["strict_fp32_fallback"] is False for w in report["witnesses"]),"new source/dtype witness inherited a physical-stage fallback")
 return True

def safe_value(value):
 if isinstance(value,dict):return {k:safe_value(v) for k,v in value.items()}
 if isinstance(value,(list,tuple)):return [safe_value(v) for v in value]
 if isinstance(value,(float,np.floating)) and not np.isfinite(value):return {"nonfinite":repr(float(value))}
 return value

def comparison_record(reference,candidate,utility,old):
 # Same original numeric routine; retain the returned failed hard/numeric data
 # instead of throwing away an approximate-arm result at old.compare's guard.
 result=utility.compare_predictions(old.normalize_configuration_metadata(reference),old.normalize_configuration_metadata(candidate))
 result["configuration_identity_basis"]="exact pickle SHA256 and type; repr addresses are audit only"
 json.dumps(result,allow_nan=False)
 return result

def evaluation_error(error,partial=None):
 return {"passed":False,"observed_numeric_comparison":False,
  "evaluation_error":{"type":type(error).__name__,"message":str(error)},
  "partial_evidence":safe_value(partial)}

class ControlHardFailure(RuntimeError):
 """Primary control-integrity failure; approximate arm was not activated."""

def control_hard_decision(row,old,original_state):
 originals=[v for k,v in row["hard_predictions"].items() if k.startswith("original_FP32_")]
 cached=[v for k,v in row["hard_predictions"].items() if k.startswith("cached_FP32_")]
 comparisons=row["comparisons"][ARMS[0]]
 return bool(original_state and all(row["arm_guards"]["FP32"].values())
  and len(originals)==len(cached)==3
  and all(old.probability_hard(v) for v in originals+cached)
  and set(comparisons)=={"1","1024","stress"}
  and all("evaluation_error" not in v and old.comparison_hard(v) for v in comparisons.values()))

def control_hard_abort(row):
 row["failure"]={"kind":"original_or_cached_FP32_hard_failure","FP16_arm_activated":False}
 row["corrected_CDF_calls"]=sum(row["corrected_CDF_call_counts"].values())
 raise ControlHardFailure("original/cachedFP32 hard validity/state failure; FP16 arm not activated")

def stage_decision(row,old):
 require(set(row["comparisons"])==set(ARMS),"two distinct execution arms required")
 common=set(row.get("guards",{}))==set(GUARDS) and all(v is True for v in row["guards"].values())
 original=[v for k,v in row["hard_predictions"].items() if k.startswith("original_FP32_")]
 result={}
 for label,arm in (("FP32",ARMS[0]),("FP16",ARMS[1])):
  predictions=[v for k,v in row["hard_predictions"].items() if k.startswith("cached_"+label+"_")]
  arm_guards=row["arm_guards"][label]
  hard=bool(common and len(original)==3 and all(old.probability_hard(v) for v in original)
   and len(predictions)==3 and all(old.probability_hard(v) for v in predictions)
   and set(arm_guards)==set(ARM_GUARDS) and all(v is True for v in arm_guards.values()))
  require(set(row["comparisons"][arm])=={"1","1024","stress"},"all declared row counts or explicit unevaluated errors required")
  strict=True
  for value in row["comparisons"][arm].values():
   if "evaluation_error" in value:
    require(value["passed"] is False and value["observed_numeric_comparison"] is False,"evaluation error relabelled")
    hard=False;strict=False
   else:
    require(value["passed"] is old.comparison_decision(value),"numeric decision relabelled")
    hard=hard and old.comparison_hard(value);strict=strict and value["passed"]
  result[label]=(bool(hard),bool(hard and strict))
 return {"common_guards_passed":bool(common),"control_hard_validity_passed":result["FP32"][0],
  "matching_FP32_execution_passed":result["FP32"][1],"FP16_hard_validity_passed":result["FP16"][0],
  "FP16_strict_passed":result["FP16"][1]}

def validate_screen(document,*,expected_sources,raw_bindings,transformed_bindings,coordinate_policy,inventory_replay):
 from posteriorforge import execution_prior_coordinates_v1 as backend
 old=load("qualify_ss15_conditionals.py",SS15_QUALIFIER_SHA256)
 expected={"schema":SCHEMA,"status":"prior_coordinates_conditional_screen_pass","sources":expected_sources,
  "raw_bindings":raw_bindings,"transformed_bindings":transformed_bindings,"coordinate_policy":coordinate_policy,"codec_inventory":inventory_replay,
  "thresholds":THRESHOLDS,"row_counts":list(ROW_COUNTS),"stress_rows":8,"fallback_map":{},
  "posterior_vectors_generated":0,"reference_integrations":0,"old_numerical_qualification_inherited":False,
  "sources_unchanged":True,"global_RNG_preserved":True,"FP16_qualified_as_FP32":False,"posterior_equivalence_claimed":False}
 for k,v in expected.items():require(document.get(k)==v,"new coordinate screen binding differs: "+k)
 require(document.get("completed_grid",{}).get("manifest_sha256")==COMPLETED_GRID_SHA256
  and document["completed_grid"].get("passed") is True,"old24 completion custody absent")
 rows=document["stage_results"];require([r["stage_index"] for r in rows]==list(STAGES),"all21 fresh latent stages required")
 for row in rows:
  j=row["stage_index"];rule=backend.stage_policy(j);binding=row["mask_inputs"];training=binding["training"]
  require(row["target_codec_coordinate"]==rule["codec_coordinate"] and training["mask"]==rule
   and training["module_sha256"]==BACKEND_SHA256 and training["codec_sha256"]==CODEC_SHA256
   and training["original_latent_target"]==training["prepared_latent_target"]==binding["actual_latent_target"]
   and training["masked_training"]==binding["actual_training"]
   and training["same_fixed_mask_for_training_query"] is True and training["target_bytes_unchanged"] is True,"latent predictor/label/codec binding differs")
  require(row["RNG_identity_before"]==row["RNG_identity_after"],"observed model RNG changed")
  require(set(row["precision_sessions"])=={"FP32","FP16"} and all(r["aliases_and_flags_restored"] is True for r in row["precision_sessions"].values()),"common alias restoration failed")
  calculated=stage_decision(row,old);require(all(row[k]==v for k,v in calculated.items()) and calculated["matching_FP32_execution_passed"],"fresh latent FP32 structural/hard failure")
  counts=row["corrected_CDF_call_counts"]
  require(counts["original_FP32"]==counts["cached_FP32"]==24 and type(counts["cached_FP16"]) is int
   and 0<=counts["cached_FP16"]<=24 and sum(counts.values())==row["corrected_CDF_calls"],"actual per-arm call coverage differs")
  witness_guard(row["precision_sessions"]["FP32"],backend.CONTROL_MODE,j,old)
  if calculated["FP16_hard_validity_passed"]:witness_guard(row["precision_sessions"]["FP16"],backend.FP16_MODE,j,old)
 descriptor=document["compatibility_adapter"];require(descriptor["patched_call_count"]==sum(r["corrected_CDF_calls"] for r in rows) and descriptor["original_alias_restored"] is True
  and descriptor["sdk_sources_unchanged"] is True and descriptor["executed_corrected_cdf_code_sha256"]==descriptor["corrected_cdf_code_sha256"]
  and descriptor["corrected_cdf_code_sha256"]!=descriptor["original_cdf_code_sha256"],"corrected CDF execution/restoration incomplete")
 failures=[r["stage_index"] for r in rows if not r["FP16_strict_passed"]]
 require(document["FP16_strict_failed_stages"]==failures and document["matching_FP32_execution_passed_all21"] is True
  and document["FP16_hard_validity_passed_all21"] is all(r["FP16_hard_validity_passed"] for r in rows),"coverage/FP16 approximation relabelled")
 return {"passed":True,"fresh_latent_stages":21,"fallback_map":{},"matching_FP32_execution_passed_all21":True,
  "FP16_hard_validity_passed_all21":document["FP16_hard_validity_passed_all21"],"FP16_strict_failed_stages":failures,"FP16_qualified_as_FP32":False,
  "posterior_equivalence_claimed":False,"scope":"fixed transformed prefixes only; no posterior or accuracy proof"}

def run(*,banks,prefix_run,checkpoint,codec_inventory,codec_reference_inventory,codec_cpu_receipt,completed_grid_audit,output,ledger,max_seconds=1050):
 from posteriorforge.banks import derive_seed
 from posteriorforge.inference import CHECKPOINT_SHA256,frozen_regressor_settings
 from posteriorforge.inference_cdf import guard_baseline_source
 from posteriorforge import execution_cache as cache,execution_prior_coordinates_v1 as backend
 # File/source prerequisites are checked before any Torch import/GPU access.
 pins=source_pins();completion=grid_receipt(completed_grid_audit);inventory,inventory_replay=inventory_receipt(codec_inventory,codec_reference_inventory,codec_cpu_receipt)
 runtime_input_audit={}
 theta,stats,query,prefixes,raw,transformed,prefix_custody=input_bindings(banks,prefix_run,inventory,runtime_audit=runtime_input_audit)
 require(type(max_seconds) is int and 600<=max_seconds<=2250,"bounded600..2250s screen required")
 require(sha256_file(checkpoint)==CHECKPOINT_SHA256,"immutable checkpoint differs")
 old=load("qualify_ss15_conditionals.py",SS15_QUALIFIER_SHA256);utility=old.load("qualify_execution_cache.py",old.UTILITY_SHA);base=old.load("characterize_precision_approximation.py",old.BASE_SHA)
 require(utility.THRESHOLDS==old.THRESHOLDS==THRESHOLDS,"original numeric limits changed")
 from posteriorforge.reference_grid import _require_budgeted_gpu
 budget=_require_budgeted_gpu(ledger);require(budget["gpus"]==1,"one reserved GPU required")
 started=time.perf_counter();deadline=started+max_seconds;output=Path(output);output.mkdir(parents=True,exist_ok=False)
 policy=backend.policy_metadata();baseline=guard_baseline_source()
 report={"schema":SCHEMA,"status":"running","attribution":"Built with PriorLabs-TabPFN","sources":pins,"raw_bindings":raw,"transformed_bindings":transformed,
  "coordinate_policy":policy,"codec_inventory":inventory_replay,"completed_grid":completion,"prefix_custody":prefix_custody,"runtime_input_audit":runtime_input_audit,
  "thresholds":THRESHOLDS,"row_counts":list(ROW_COUNTS),"stress_rows":8,"fallback_map":{},"context_seed":20261111,"case":2,
  "numeric_failure_policy":"collect all21 independent fixed prefixes unless common/control hard validity fails; tolerance failures remain failure and block pilot",
  "stress_policy":"finite retained prior-latent noise[-3,-2,2,3]; zero/-3/+3 and trainingmean+3SD prefixes; original observations fixed; decoder-validity checked",
  "stage_results":[],"posterior_vectors_generated":0,"reference_integrations":0,"old_numerical_qualification_inherited":False,
  "matching_FP32_execution_passed_all21":False,"FP16_hard_validity_passed_all21":False,"FP16_hard_failed_stages":[],"FP16_strict_failed_stages":[],
  "FP16_qualified_as_FP32":False,"posterior_equivalence_claimed":False,"budget_binding":budget,"baseline_execution_source":baseline}
 active=None;write_json(output/"manifest.json",report)
 try:
  import torch
  from posteriorforge.compat_cdf import finite_cdf_compatibility
  require(torch.cuda.is_available() and torch.cuda.device_count()==1,"one visible GPU required")
  require(torch.get_default_dtype()==torch.float32 and torch.get_float32_matmul_precision()=="highest" and not torch.backends.cuda.matmul.allow_tf32 and not torch.is_autocast_enabled("cuda"),"strict outer FP32 flags required")
  seed=derive_seed(20261111,"prior-coordinate-fixed-conditionals-v1",case=2);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
  torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS","16")));initial_rng=utility.rng_identity(utility.rng_state(torch));report["seed"]=seed
  with finite_cdf_compatibility() as active:
   for stage in STAGES:
    require(time.perf_counter()+25<deadline,"bounded deadline; partial coverage cannot qualify")
    main,stress,noise=fixed_queries(query,prefixes,theta,stage,backend);mt,ms,mq,sq,binding=masked_inputs(theta,stats,main,stress,stage,backend)
    row={"stage_index":stage,"target_codec_coordinate":backend.stage_policy(stage)["codec_coordinate"],"stress_noise_stage_indices":noise,"mask_inputs":binding,
     "comparisons":{a:{} for a in ARMS},"hard_predictions":{},"guards":{},"precision_sessions":{}};report["stage_results"].append(row);write_json(output/"manifest.json",report)
    before_rng=utility.rng_identity(utility.rng_state(torch));before_cdf=active["patched_call_count"]
    settings=frozen_regressor_settings(checkpoint);settings["inference_precision"]=torch.float32;_,sdk,_=cache._load_sdk();control=sdk.TabPFNRegressor(**settings)
    train=np.concatenate((ms,mt[:,:stage]),axis=1);target=mt[:,stage]
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):control.fit(train,target)
    queries={"1":mq[:1],"1024":mq,"stress":sq};references={}
    for label,features in queries.items():
     with torch.random.fork_rng(devices=[torch.cuda.current_device()]):prediction=cache.capture_corrected_prediction(control,features,cdf_descriptor=active)
     references[label]=prediction;row["hard_predictions"]["original_FP32_"+label]=old.hard_prediction(prediction,base,utility)
    original_model=base.model_summary(control,utility);row["original_FP32_model"]=original_model
    original_state=old.state_guard(original_model);session_reports=row["precision_sessions"];row["arm_guards"]={}
    row["corrected_CDF_call_counts"]={"original_FP32":active["patched_call_count"]-before_cdf,"cached_FP32":0,"cached_FP16":0}
    for mode,label,arm in ((backend.CONTROL_MODE,"FP32",ARMS[0]),(backend.FP16_MODE,"FP16",ARMS[1])):
     arm_before=active["patched_call_count"];session=None
     row["arm_guards"][label]={k:False for k in ARM_GUARDS}
     try:
      with backend.PriorCoordinateSession(mode) as session:
       with ExitStack() as cleanup:
        engine=backend.PriorCoordinateStageExecutor(checkpoint,cdf_descriptor=active,query_batch_size=1024);cleanup.callback(engine.close)
        row[label+"_prepared"]=engine.prepare(mt,ms,stage);model=old.cached_model_summary(engine,base,utility);row[label+"_model_before"]=model
        guards=row["arm_guards"][label]
        guards.update(FP32_statistics_normalizers_borders=old.state_guard(model),
         target_statistics_preserved=all(model[k]==original_model[k] for k in ("target_mean","target_sd","temperature")),
         preprocessor_identity_preserved=model["contract"]["preprocessor_sha256"]==original_model["contract"]["preprocessor_sha256"])
        for key,features in queries.items():
         try:
          candidate=old.capture_components(engine,features,torch)
          row["hard_predictions"]["cached_"+label+"_"+key]=old.hard_prediction(candidate,base,utility)
          row["comparisons"][arm][key]=comparison_record(references[key],candidate,utility,old)
         except Exception as error:
          partial=None
          if "candidate" in locals():
           try:partial=base.hard_prediction_validity(candidate)
           except Exception:pass
          row["hard_predictions"]["cached_"+label+"_"+key]=evaluation_error(error,partial)
          row["comparisons"][arm][key]=evaluation_error(error,partial)
          break
        after=old.cached_model_summary(engine,base,utility);row[label+"_model_after"]=after
        guards["cached_contract_preserved"]=model==after
        guards["FP32_statistics_normalizers_borders"]=guards["FP32_statistics_normalizers_borders"] and old.state_guard(after)
      session_reports[label]=session.report()
      try:row["arm_guards"][label]["actual_dtype_witnesses"]=witness_guard(session_reports[label],mode,stage,old)
      except Exception as error:row.setdefault("arm_errors",{})[label]=evaluation_error(error)
     except Exception as error:
      row.setdefault("arm_errors",{})[label]=evaluation_error(error)
      if session is not None:session_reports[label]=session.report()
     finally:
      row["corrected_CDF_call_counts"]["cached_"+label]=active["patched_call_count"]-arm_before
      for key in queries:
       if key not in row["comparisons"][arm]:
        row["comparisons"][arm][key]=evaluation_error(RuntimeError("not evaluated after an earlier arm failure"))
       if "cached_"+label+"_"+key not in row["hard_predictions"]:
        row["hard_predictions"]["cached_"+label+"_"+key]=evaluation_error(RuntimeError("not evaluated after an earlier arm failure"))
     if label=="FP32":
      require(source_pins()==pins and utility.rng_identity(utility.rng_state(torch))==before_rng,"common source/RNG failure after control arm")
      hard_control=control_hard_decision(row,old,original_state)
      if not hard_control:control_hard_abort(row)
    row["precision_sessions"]=session_reports;row["corrected_CDF_calls"]=active["patched_call_count"]-before_cdf
    require(row["corrected_CDF_call_counts"]["original_FP32"]==24 and row["corrected_CDF_call_counts"]["cached_FP32"]==24,
     "complete original/cachedFP32 3x8call coverage required")
    require(sum(row["corrected_CDF_call_counts"].values())==row["corrected_CDF_calls"],"actual per-arm CDF counts do not sum")
    after_rng=utility.rng_identity(utility.rng_state(torch));row["RNG_identity_before"]=before_rng;row["RNG_identity_after"]=after_rng
    row["guards"]={k:True for k in GUARDS};row["guards"].update(sources_unchanged=source_pins()==pins,
     identical_latent_training_inputs=masked_inputs(theta,stats,main,stress,stage,backend)[4]==binding,
     latent_target_bytes_unchanged=target.tobytes()==theta[:,stage].tobytes(),original_FP32_state_guard=original_state,
     model_RNG_preserved=before_rng==after_rng,corrected_CDF_executed=active.get("executed_corrected_cdf_code_sha256")==active.get("corrected_cdf_code_sha256"),
     aliases_restored=len(session_reports)==2 and all(r["aliases_and_flags_restored"] for r in session_reports.values()))
    row["cached_state_scope"]={"KV_contents_hashed":False,"large_embedding_bytes_hashed":False,"no_retention_claim":True}
    row.update(stage_decision(row,old));write_json(output/"manifest.json",report)
    del control,references,prediction
    if "candidate" in locals():del candidate
    if "engine" in locals():del engine
    torch.cuda.synchronize()
    if not row["control_hard_validity_passed"]:break
  report["compatibility_adapter"]=dict(active);require(active["original_alias_restored"] and active["sdk_sources_unchanged"] and active["patched_call_count"]==sum(r["corrected_CDF_calls"] for r in report["stage_results"]),"actual CDF source/restoration/call coverage differs")
  require(utility.rng_identity(utility.rng_state(torch))==initial_rng,"model RNG changed")
  require(source_pins()==pins and backend.policy_metadata()==policy and guard_baseline_source()==baseline,"source/policy changed")
  require(sha256_file(codec_inventory)==INVENTORY_SHA256 and sha256_file(codec_reference_inventory)==REFERENCE_INVENTORY_SHA256 and sha256_file(codec_cpu_receipt)==COMBINED_CPU_SHA256 and sha256_file(completed_grid_audit)==COMPLETED_GRID_SHA256 and sha256_file(checkpoint)==CHECKPOINT_SHA256,"prerequisite custody changed")
  require(sha256_file(Path(prefix_run)/"manifest.json")==prefix_custody["manifest_sha256"] and sha256_file(Path(prefix_run)/"samples.npz")==prefix_custody["payload_sha256"],"fixed prefix changed")
  rows=report["stage_results"];complete=[r["stage_index"] for r in rows]==list(STAGES)
  report.update(matching_FP32_execution_passed_all21=bool(complete and all(r["matching_FP32_execution_passed"] for r in rows)),
   FP16_hard_validity_passed_all21=bool(complete and all(r["FP16_hard_validity_passed"] for r in rows)),
   FP16_strict_failed_stages=[r["stage_index"] for r in rows if not r["FP16_strict_passed"]],
   FP16_hard_failed_stages=[r["stage_index"] for r in rows if not r["FP16_hard_validity_passed"]],sources_unchanged=True,global_RNG_preserved=True)
  report["status"]="prior_coordinates_conditional_screen_pass" if report["matching_FP32_execution_passed_all21"] else "prior_coordinates_matching_FP32_structural_or_hard_failure"
  if report["status"]=="prior_coordinates_conditional_screen_pass":report["replay"]=validate_screen(report,expected_sources=pins,raw_bindings=raw,transformed_bindings=transformed,coordinate_policy=policy,inventory_replay=inventory_replay)
 except BaseException as error:
  report.update(status="failed_or_incomplete_prior_coordinates_screen",error_type=type(error).__name__,error=str(error))
  if active is not None:report["compatibility_adapter"]=dict(active)
  raise
 finally:report["elapsed_seconds"]=time.perf_counter()-started;write_json(output/"manifest.json",report)
 return report

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for key in ("banks","prefix-run","checkpoint","codec-inventory","codec-reference-inventory","codec-cpu-receipt","completed-grid-audit","output","ledger"):p.add_argument("--"+key,required=True)
 p.add_argument("--max-seconds",type=int,default=1050);report=run(**vars(p.parse_args()))
 print(json.dumps({k:report.get(k) for k in ("status","matching_FP32_execution_passed_all21","FP16_hard_validity_passed_all21","FP16_strict_failed_stages","elapsed_seconds")},sort_keys=True))
if __name__=="__main__":main()
