"""Fresh pair-bound v3 FP32-forward / Policy-C FP64 probability screen.

Built with PriorLabs-TabPFN. CPU source/tests only until all new pins are frozen
and Root separately authorizes a budgeted neural screen. No old pass is inherited.
"""
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from posteriorforge import prior_coordinate_pair_v1 as pairs
from posteriorforge.artifacts import write_json

SCHEMA="posteriorforge.prior_coordinate_pair_conditionals.v3"
PASS_STATUS="prior_coordinate_pair_v3_screen_pass"
PAIR_MODULE_SHA256="402117cad65af190352cb0b14977fbbdc896f196d069749b0be28608fdef400e"
PAIR_CPU_HELPER_SHA256="b69a92b59fa504bce22e34dc31895f9162870425118764293f959c46390110e4"
PAIR_CPU_RECEIPT_SHA256="6f35d9d090a955604186c5fbf3312b82465eff6fdae7e4cd24a23b646dbdf193"
UTILITY_SHA256="6c927a5c278c7fc54b556ac557ecef64f5e6338f5ae13bf466f3b1e6a4640f90"
V2_QUALIFIER_SHA256="a9f4d6818c030ac29587642fa744101c49b9f4a5fd192ab224c0f273f770e875"
V3_BACKEND_SHA256="7c7180e3f161f9fc34684fc492f9ab558660b3b41092b70feea17edbbbf3f19c"
V3_PROBABILITY_SHA256="7fea13270c38c51eed0eb65cd5026de94e3f7e24cc4734e8906e882fb0bc0494"
V3_PROPOSAL_SHA256="9216e76ce335ee52ffb1a5e8fbdd33c72f8c0aa84c4f1803dd90ff06b3e3a6d7"
V3_PROTOCOL_SHA256="7ea6e5c57f8404eb761e07f4011b14966d5885fbbdb8620d463c5283916041d8"
STAGES=tuple(range(21))
QUERIES=("1","1024","stress")
NATIVE_STAGES=(6,10,14)
UNIFORM_BITS=(0x00000000,0x3f000000,0x3f7fffff)
THRESHOLDS={"normalization_absolute":1e-5,"max_cdf_absolute":2e-5,
 "max_row_total_variation":1e-4,"quantile_coordinate_span_fraction":2e-6}
GUARDS=("sources_unchanged","identical_model_inputs","latent_target_bytes_unchanged",
 "FP32_neural_statistics_and_normalizers","target_statistics_preserved",
 "preprocessor_identity_preserved","state_contract_preserved",
 "immutable_native_state_preserved","default_RNG_preserved",
 "actual_neural_dtype_witnesses","actual_FP64_probability_witnesses",
 "probability_source_executed","aliases_and_flags_restored",
 "no_legacy_qualification_inherited")
_utility=pairs.load_script("qualify_execution_cache.py",UTILITY_SHA256)
require=pairs.require
def source_sha256():return pairs.sha(__file__)
def _modules():
 from posteriorforge import execution_prior_native_v3 as backend
 from posteriorforge import execution_probability_v3 as probability
 from posteriorforge import prior_proposals_v3 as proposal
 return backend,probability,proposal

def new_source_files():
 return {"scripts/qualify_prior_coordinate_pair_v3.py":source_sha256(),
  "src/posteriorforge/execution_prior_native_v3.py":V3_BACKEND_SHA256,
  "src/posteriorforge/execution_probability_v3.py":V3_PROBABILITY_SHA256,
  "src/posteriorforge/prior_proposals_v3.py":V3_PROPOSAL_SHA256,
  "docs/accuracy-prior-coordinate-fp64-proposals-v3.md":V3_PROTOCOL_SHA256}
def checkout_source_receipt():
 """Explicit CPU checkout-only evidence; cannot activate a numerical proof."""
 root=Path(__file__).resolve().parents[1];files=new_source_files()
 require(all(isinstance(v,str) and len(v)==64 for v in files.values()),"new v3 source pins incomplete")
 require(all(pairs.sha(root/name)==value for name,value in files.items()),"v3 checkout source differs")
 return {"sources":files,"checkout_bytes_verified":True,"committed_bytes_verified":False,"scope":"CPU checkout only; no proof/activation credit"}
def committed_source_receipt(*,revision=None):
 """Recorded-Git source guard; future evidence commits do not invalidate old bytes."""
 import re,subprocess
 root=Path(__file__).resolve().parents[1];files=checkout_source_receipt()["sources"]
 if revision is None:
  result=subprocess.run(["git","-C",str(root),"rev-parse","HEAD"],check=True,capture_output=True,text=True)
  revision=result.stdout.strip()
 require(type(revision) is str and re.fullmatch("[0-9a-f]{40}",revision) is not None,"valid recorded Git revision required")
 for name,expected in files.items():
  result=subprocess.run(["git","-C",str(root),"show",revision+":"+name],capture_output=True)
  require(result.returncode==0 and hashlib.sha256(result.stdout).hexdigest()==expected,"v3 source not committed at recorded revision: "+name)
 return {"git_revision":revision,"sources":files,"checkout_bytes_verified":True,"committed_bytes_verified":True,
  "scope":"new v3 executed/protocol closure; original dependencies have separate frozen guards"}


def source_pins():
 values=(V3_BACKEND_SHA256,V3_PROBABILITY_SHA256,V3_PROPOSAL_SHA256,V3_PROTOCOL_SHA256)
 require(all(isinstance(x,str) and len(x)==64 for x in values),"v3 sources/protocol not frozen; neural execution disabled")
 require(pairs.source_sha256()==PAIR_MODULE_SHA256,"PairSpec/data source differs")
 root=Path(__file__).resolve().parents[1]
 require(pairs.sha(root/"scripts/preflight_prior_coordinate_pairs_v1.py")==PAIR_CPU_HELPER_SHA256,"original CPU panel helper differs")
 require(pairs.sha(root/"scripts/qualify_execution_cache.py")==UTILITY_SHA256,"immutable numerical math differs")
 require(pairs.sha(root/"scripts/qualify_prior_coordinate_pair_v2.py")==V2_QUALIFIER_SHA256,"v2 data/numeric ancestry differs")
 require(pairs.sha(root/"docs/accuracy-prior-coordinate-fp64-proposals-v3.md")==V3_PROTOCOL_SHA256,"v3 protocol differs")
 backend,probability,proposal=_modules()
 for module,expected in ((backend,V3_BACKEND_SHA256),(probability,V3_PROBABILITY_SHA256),(proposal,V3_PROPOSAL_SHA256)):
  require(module.source_sha256()==expected,"new v3 module source differs");module.require_sources()
 pairs.require_sources()
 return {"qualifier":source_sha256(),"execution_v3":V3_BACKEND_SHA256,"probability_v3":V3_PROBABILITY_SHA256,
  "proposal_v3":V3_PROPOSAL_SHA256,"protocol_v3":V3_PROTOCOL_SHA256,"pair_data":PAIR_MODULE_SHA256,
  "pair_CPU_helper":PAIR_CPU_HELPER_SHA256,"pair_CPU_receipt":PAIR_CPU_RECEIPT_SHA256,
  "unchanged_numerical_limits_math":UTILITY_SHA256,"v2_input_ancestry_only":V2_QUALIFIER_SHA256,
  "old_numerical_qualification_transferred":False}
def cpu_receipt(path,spec):
 require(pairs.sha(path)==PAIR_CPU_RECEIPT_SHA256,"exact original all23 dualCPU data receipt required")
 document=json.loads(Path(path).read_bytes());row=pairs.validate_pair_cpu_receipt(document,spec)
 require(document["producer_source_sha256"]==PAIR_CPU_HELPER_SHA256,"original CPU helper binding differs")
 return document,row
def data_helper():
 legacy=pairs.load_script("qualify_prior_coordinate_pair_v2.py",V2_QUALIFIER_SHA256)
 return legacy.numeric.data_helper()
def bound_inputs(spec,*,banks,completion_receipt,training_inventory,pair_cpu_receipt,prefix_run=None):
 require(type(spec) is pairs.PairSpec,"immutable PairSpec required")
 cpu,row=cpu_receipt(pair_cpu_receipt,spec)
 theta,stats,query,prefixes,raw,transformed,custody=pairs.input_bindings(spec,banks=banks,
  completion_receipt=completion_receipt,training_inventory=training_inventory,pair_inventory=cpu,old_parent=prefix_run)
 require(custody==row["inference"]["prefix_custody"],"actual pair prefix custody differs")
 return SimpleNamespace(theta=theta,stats=stats,query=query,prefixes=prefixes,raw=raw,
  transformed=transformed,custody=custody,cpu=cpu,cpu_row=row)
def _numeric(record,limit):
 return type(record) in (int,float) and np.isfinite(record) and 0<=record<=limit
def strict_probability(record):
 return (type(record) is dict and _numeric(record.get("max_normalization_error"),THRESHOLDS["normalization_absolute"])
  and _numeric(record.get("max_cdf_absolute_error"),THRESHOLDS["max_cdf_absolute"])
  and _numeric(record.get("max_row_total_variation"),THRESHOLDS["max_row_total_variation"])
  and type(record.get("minimum_probability")) in (int,float) and np.isfinite(record["minimum_probability"])
  and record["minimum_probability"]>=0)
def compare_weights(reference,candidate):
 """EXACT immutable probability-metric math; no normalization or repair here."""
 return _utility.compare_probabilities(reference,candidate,thresholds=THRESHOLDS)
def probability_hard(record):
 require(type(record) is dict,"actual probability record required")
 fields=("raw_weights64","probabilities64","sampling_cdf64","sampling_interval_weights64","criterion_borders")
 return bool(record.get("actual_probability_dtypes")=={k:"torch.float64" for k in fields}
  and record.get("all_components_probability_dtype64") is True
  and record.get("all_required_values_finite") is True and record.get("nonnegative_weights") is True
  and record.get("positive_raw_row_mass") is True and record.get("raw_mass_envelope")==1.00001
  and _numeric(record.get("max_raw_mass_deviation"),THRESHOLDS["normalization_absolute"])
  and _numeric(record.get("API_normalization_error"),THRESHOLDS["normalization_absolute"])
  and record.get("CDF_monotonic") is True and record.get("CDF_terminal_exactly_one") is True
  and record.get("all8_actual_components") is True
  and record.get("fitted_F32_border_bits_promoted_unchanged") is True
  and record.get("temperature_reapplied") is False
  and record.get("inverse_same_input_replay",{}).get("passed") is True
  and record.get("inverse_same_input_replay",{}).get("uniform_bits")==list(UNIFORM_BITS)
  and record.get("inverse_same_input_replay",{}).get("u1_rejected") is True
  and record.get("inverse_same_input_replay",{}).get("stored_stage_dtype")=="<f4"
  and record.get("inverse_same_input_replay",{}).get("u0_policy")=="first_positive_CDF_interval_left_border"
  and record.get("independent_postprocessing_replay_passed") is True
  and record.get("proposal_precision",{}).get("hard_passed") is True)
def comparison_hard(record):
 return bool(record.get("same_actual_model_inputs") is True and record.get("exact_F64_criterion_borders") is True
  and record.get("all8_components") is True and len(record.get("members",[]))==8
  and [x.get("member") for x in record["members"]]==list(range(8))
  and all(x.get("configuration_identity_equal") is True and x.get("exact_source_borders") is True for x in record["members"]))
def comparison_decision(record):
 if not comparison_hard(record):return False
 layers=("raw_weights","API_probabilities","sampling_interval_weights")
 return all(strict_probability(record.get(k,{})) for k in layers) and all(
  all(strict_probability(m.get(k,{})) for k in layers) for m in record["members"])
def stage_decision(row):
 require(set(row.get("guards",{}))==set(GUARDS),"complete new v3 guard set required")
 require(set(row.get("comparisons",{}))==set(QUERIES),"N1/N1024/stress all required")
 require(set(row.get("hard_predictions",{}))=={a+"_"+q for a in ("original","candidate") for q in QUERIES},"actual both-arm outputs required")
 hard=all(v is True for v in row["guards"].values())
 hard=hard and all(probability_hard(v) for v in row["hard_predictions"].values())
 for value in row["comparisons"].values():
  require(value.get("passed") is comparison_decision(value),"v3 numerical decision relabelled")
  hard=hard and comparison_hard(value)
 return {"hard_validity_passed":bool(hard),"matching_v3_execution_passed":bool(hard and all(v["passed"] for v in row["comparisons"].values()))}
def validate_screen(document,*,spec,expected_sources,raw_bindings,transformed_bindings,
 coordinate_policy,native_policy,probability_policy,proposal_policy,inventory_replay,pair_cpu_receipt):
 require(type(spec) is pairs.PairSpec,"exact PairSpec required")
 expected={"schema":SCHEMA,"status":PASS_STATUS,"pair":spec.as_dict(),"sources":expected_sources,
  "raw_bindings":raw_bindings,"transformed_bindings":transformed_bindings,
  "coordinate_policy":coordinate_policy,"native_execution_policy":native_policy,
  "probability_policy":probability_policy,"proposal_policy":proposal_policy,
  "codec_inventory":inventory_replay,"thresholds":THRESHOLDS,
  "pair_CPU_receipt_sha256":PAIR_CPU_RECEIPT_SHA256,"v3_preregistration_sha256":V3_PROTOCOL_SHA256,
  "native_all8_stages":[6,10,14],"cached_stage_count":18,"FP16_activated":False,
  "posterior_vectors_generated":0,"reference_integrations":0,
  "old_numerical_qualification_inherited":False,"sources_unchanged":True,"global_RNG_preserved":True}
 for key,value in expected.items():require(document.get(key)==value,"new v3 proof binding differs: "+key)
 closure=document.get("committed_source_closure",{})
 require(closure.get("sources")==new_source_files() and closure.get("git_revision")==document.get("git_revision")
  and closure.get("committed_bytes_verified") is True and closure.get("checkout_bytes_verified") is True,"committed new v3 source closure required")
 cpurow=pairs.validate_pair_cpu_receipt(pair_cpu_receipt,spec)
 require(document.get("prefix_custody")==cpurow["inference"]["prefix_custody"],"original FIRST1024 custody differs")
 rows=document.get("stage_results",[])
 require([r.get("stage_index") for r in rows]==list(STAGES),"fresh all21 coverage required")
 require([r.get("mask_inputs") for r in rows]==cpurow["inference"]["all21_mask_target_query_bindings"],"actual original F32 masks/targets/query bytes differ")
 for row in rows:
  outcome=stage_decision(row)
  require(all(row.get(k) is v for k,v in outcome.items()) and outcome["matching_v3_execution_passed"],"v3 stage hard/numeric failure")
  require(row.get("route")==("original_fit_preprocessors_all8" if row["stage_index"] in NATIVE_STAGES else "frozen88_resident_cache"),"v3 forward route differs")
  require(row.get("probability_translation_counts")=={"original":24,"candidate":24},"actual all8 both-arm probability counts differ")
  require(row.get("RNG_identity_before")==row.get("RNG_identity_after"),"stage default RNG changed")
  if row["stage_index"] in NATIVE_STAGES:
   snapshots=row.get("immutable_native_snapshots",[])
   require(len(snapshots)==4 and all(x==snapshots[0] for x in snapshots)
    and snapshots[0]==row.get("original_control_immutable_state")
    and row.get("state_scope")=="native planned N1 warm baseline; immutable fitted state","native actual state scope differs")
  else:require(row.get("state_scope")=="unchanged88 cached prepared-before/after contract","cached actual state scope differs")
 footer=document.get("probability_scope",{})
 require(footer.get("translation_call_count")==1008 and footer.get("closed") is True
  and footer.get("active") is False and footer.get("source_unchanged") is True
  and footer.get("SDK_global_aliases_modified") is False
  and footer.get("actual_probability_dtype")=="torch.float64","new probability source/closure footer differs")
 require(document.get("matching_v3_execution_passed_all21") is True and document.get("v3_failed_stages")==[],"new all21 pass required")
 return {"passed":True,"matching_v3_execution_passed_all21":True,"fresh_stages":21,
  "pair":spec.as_dict(),"native_all8_stages":[6,10,14],"cached_stages":18,
  "forward_precision":"torch.float32","probability_precision":"torch.float64",
  "old_numerical_qualification_inherited":False,"posterior_equivalence_claimed":False,
  "scope":"Fresh actual-pair finite-prefix v3 screen only; zero population/posterior credit"}
def qualification_receipt(spec,artifact_path,*,banks,completion_receipt,training_inventory,
 reference_training_inventory,combined_training_inventory,pair_cpu_receipt,raw_bindings,
 model_inputs,prefix_run=None,failed_screen=None):
 pins=source_pins();backend,probability,proposal=_modules()
 bound=bound_inputs(spec,banks=banks,completion_receipt=completion_receipt,
  training_inventory=training_inventory,pair_cpu_receipt=pair_cpu_receipt,prefix_run=prefix_run)
 from posteriorforge.science.analytical_benchmark_v2 import array_descriptor
 expected_model={k:array_descriptor(v) for k,v in (("context_theta",bound.theta),("context_obs",bound.stats),("query",bound.query))}
 require(bound.raw==raw_bindings and expected_model==model_inputs,"actual producer pair/raw/model inputs differ")
 data=data_helper();base=data.inventory_receipt(training_inventory,reference_training_inventory,combined_training_inventory)[1]
 inventory={"training_inventory":base,"pair":spec.as_dict(),"pair_CPU_receipt_sha256":PAIR_CPU_RECEIPT_SHA256,
  "COMMON_Float32_model_bytes_match":True,"Float64_cross_environment_byte_identity_claimed":False,"ancestry_scope":"data only; zero v3 neural credit"}
 document=json.loads(Path(artifact_path).read_bytes())
 require(document.get("committed_source_closure")==committed_source_receipt(revision=document.get("git_revision")),"recorded v3 Git/source closure differs")
 replay=validate_screen(document,spec=spec,expected_sources=pins,raw_bindings=bound.raw,
  transformed_bindings=bound.transformed,coordinate_policy=backend.cached.policy_metadata(),
  native_policy=backend.policy_metadata(),probability_policy=probability.metadata(),
  proposal_policy=proposal.contract_metadata(),inventory_replay=inventory,pair_cpu_receipt=bound.cpu)
 return document,{"screen_sha256":pairs.sha(artifact_path),"replay":replay,"pair":spec.as_dict(),
  "raw_scientific_bindings":bound.raw,"transformed_bindings":bound.transformed,"fixed_prefix_custody":bound.custody,
  "pair_CPU_receipt_sha256":PAIR_CPU_RECEIPT_SHA256,"probability_policy":probability.metadata(),
  "proposal_policy":proposal.contract_metadata(),"old_numerical_qualification_inherited":False,
  "scope":"Request only CPU-replays a separately performed same-pair v3 conditional screen"}


def _np(value):
 return value.detach().cpu().numpy() if hasattr(value,"detach") else np.asarray(value)
def _descriptor(value):
 from posteriorforge.science.analytical_benchmark_v2 import array_descriptor
 return array_descriptor(np.asarray(_np(value)))
def reference_translation64(adjusted,frm,to):
 """Independent NumPy F64 replay on ACTUAL adjusted-F32 logits/border bits."""
 x=np.asarray(_np(adjusted),np.float64);f=np.asarray(_np(frm),np.float64);y=np.asarray(_np(to),np.float64)
 require(x.ndim==2 and f.ndim==y.ndim==1 and x.shape[1]+1==len(f),"reference dimensions differ")
 p=np.exp(x-np.max(x,axis=-1,keepdims=True));p/=p.sum(-1,keepdims=True)
 before=np.cumsum(p,-1)-p
 indices=np.searchsorted(f,y,side="left")-1
 indices[y==f[0]]=0;indices[y==f[-1]]=len(f)-2
 indices=np.clip(indices,0,len(f)-2)
 share=np.clip((y-f[indices])/(f[indices+1]-f[indices]),0.,1.)
 cdf=before[:,indices]+p[:,indices]*share
 cdf[:,y<=f[0]]=0.;cdf[:,y>=f[-1]]=1.
 cdf=np.clip(cdf,0.,1.);cdf[:,0]=0.;cdf[:,-1]=1.
 return {"softmax64":p,"raw_weights64":np.maximum(cdf[:,1:]-cdf[:,:-1],0.)}
def independent_inverse64(cdf,borders,uniform):
 """Independent scalar row replay of the DECLARED represented CDF measure."""
 c=np.asarray(cdf,np.float64);b=np.asarray(borders,np.float64);u=float(np.float32(uniform))
 require(c.ndim==2 and b.shape==(c.shape[1]+1,) and 0<=u<1,"reference inverse domain differs")
 intervals=np.diff(np.concatenate((np.zeros((len(c),1)),c),axis=1),axis=1)
 values=[]
 for row,mass in zip(c,intervals):
  index=int(np.flatnonzero(mass>0)[0]) if u==0 else int(np.searchsorted(row,u,side="left"))
  require(index<len(mass) and mass[index]>0,"reference selected empty interval")
  left=0. if index==0 else row[index-1]
  fraction=(u-left)/mass[index]
  require(0<=fraction<=1,"reference fraction invalid")
  values.append(b[index]+(b[index+1]-b[index])*fraction)
 return np.asarray(values,np.float64)
def _components(prediction):
 components=prediction.get("component_predictions")
 require(type(components) is list and len(components)==8,"actual all8 component predictions required")
 return components
def inverse_record(prediction,raw_features,probability,proposal,torch):
 """All-row same-input inverse replay; decoder evidence at predeclared rows only."""
 collection=[prediction]+_components(prediction);values=[];max_error=0.;precision=[];outside=0
 uniforms=np.asarray(UNIFORM_BITS,np.uint32).view(np.float32)
 for index,p in enumerate(collection):
  cdf=_np(p["sampling_cdf64"]);borders=_np(p["criterion"].borders);span=float(borders[-1]-borders[0])
  require(np.isfinite(span) and span>0,"finite criterion span required")
  selected_rows=sorted(set((0,len(cdf)//2,len(cdf)-1)))
  for u in uniforms:
   supplied=torch.full((len(cdf),),float(u),dtype=torch.float32,device="cpu")
   actual=probability.inverse_details(p["logits"],p["criterion"].borders,supplied)
   calculated=_np(actual["values64"])
   replay=independent_inverse64(cdf,borders,u)
   error=float(np.max(np.abs(calculated-replay)))
   max_error=max(max_error,error/span)
   require(np.array_equal(_np(actual["sampling_cdf64"]),cdf),"inverse changed actual same-input sampling measure")
   outside+=int(np.count_nonzero((calculated<borders[0])|(calculated>borders[-1])|~np.isfinite(calculated)))
   # Unmasked original latent prefixes remain the proposal context, never zeroed model-mask columns.
   stored=np.asarray(calculated,dtype=np.float32)
   prefixes=np.asarray(raw_features[selected_rows,15:],np.float32)
   z=np.concatenate((prefixes,stored[selected_rows,None]),axis=1)
   audit=proposal.precision_report(z)
   precision.append({"member":"mixture" if index==0 else index-1,"U_bits":int(np.asarray(u).view(np.uint32)),
    "rows":selected_rows,"stored_stage_dtype":stored.dtype.str,
    "declared_original_stage_storage_replay":_descriptor(stored[selected_rows]),
    "report":audit})
  try:probability.finite_bar_icdf(p["logits"],p["criterion"].borders,torch.ones(len(cdf),dtype=torch.float32,device="cpu"))
  except (ValueError,RuntimeError):pass
  else:raise ValueError("v3 U=1 was not rejected")
 return {"passed":bool(max_error<=THRESHOLDS["quantile_coordinate_span_fraction"] and outside==0),
  "uniform_bits":list(UNIFORM_BITS),"u1_rejected":True,"u0_policy":"first_positive_CDF_interval_left_border",
  "maximum_same_input_quantile_error_span_fraction":max_error,"outside_or_nonfinite_outputs":outside,
  "all8_heads_plus_mixture":True,"stored_stage_dtype":"<f4","inverse_arithmetic_dtype":"torch.float64","decoder_rows":"sorted unique [0,N//2,N-1]; no full-production domain claim",
  "proposal_precision":{"hard_passed":all(p["report"].get("hard_passed") is True for p in precision),
   "scope":"all8 plus mixture, all fixed U, declared scalar rows only; no acceptance/filtering",
   "scalar_reports":precision}}
def prediction_record(prediction,raw_features,probability,proposal,torch):
 fields={"raw_weights64":prediction["raw_weights64"],"probabilities64":prediction["probabilities64"],
  "sampling_cdf64":prediction["sampling_cdf64"],"sampling_interval_weights64":prediction["sampling_interval_weights64"],
  "criterion_borders":prediction["criterion"].borders}
 components=_components(prediction);all_predictions=[prediction]+components
 dtypes={k:str(v.dtype) for k,v in fields.items()}
 finite=all(bool(torch.isfinite(v).all()) for p in all_predictions for v in
  (p["raw_weights64"],p["probabilities64"],p["sampling_cdf64"],p["sampling_interval_weights64"],p["criterion"].borders))
 mass=np.concatenate([_np(p["component_raw_mass"]) for p in all_predictions])
 normalization=max(float(np.max(np.abs(_np(p["probabilities64"]).sum(-1)-1))) for p in all_predictions)
 replay=[]
 for member in components:
  translation=member["probability_capture"]["translation"]
  reference=reference_translation64(translation["adjusted_logits_F32"],translation["from_borders_F32"],translation["to_borders_F32"])
  replay.append(compare_weights(reference["raw_weights64"],_np(member["raw_weights64"])))
 inverse=inverse_record(prediction,raw_features,probability,proposal,torch)
 return {"actual_probability_dtypes":dtypes,"all_required_values_finite":finite,
  "all_components_probability_dtype64":all(all(v.dtype==torch.float64 for v in (p["raw_weights64"],p["probabilities64"],p["sampling_cdf64"],p["sampling_interval_weights64"],p["criterion"].borders)) for p in all_predictions),
  "nonnegative_weights":all(bool((p["raw_weights64"]>=0).all() and (p["probabilities64"]>=0).all()
   and (p["sampling_interval_weights64"]>=0).all()) for p in all_predictions),
  "positive_raw_row_mass":bool(np.isfinite(mass).all() and np.all(mass>0) and np.all(mass<=1.00001)),
  "raw_mass_envelope":1.00001,"max_raw_mass_deviation":float(np.max(np.abs(mass-1))),
  "API_normalization_error":normalization,
  "CDF_monotonic":all(bool((p["sampling_cdf64"][:,1:]>=p["sampling_cdf64"][:,:-1]).all()) for p in all_predictions),
  "CDF_terminal_exactly_one":all(bool((p["sampling_cdf64"][:,-1]==1).all()) for p in all_predictions),
  "all8_actual_components":True,"fitted_F32_border_bits_promoted_unchanged":all(p["probability_v3"]["fitted_border_F32_bits_promoted_unchanged"] is True for p in components),
  "temperature_reapplied":any(p["probability_v3"].get("temperature_reapplied") is not False for p in components),
  "independent_postprocessing_replay":replay,"independent_postprocessing_replay_passed":all(r["passed"] for r in replay),
  "inverse_same_input_replay":inverse,"proposal_precision":inverse["proposal_precision"],
  "actual_tensor_descriptors":{k:_descriptor(v) for k,v in fields.items()},
  "actual_component_descriptors":[{"raw":_descriptor(p["raw_weights64"]),"API":_descriptor(p["probabilities64"]),
   "sampling_intervals":_descriptor(p["sampling_interval_weights64"])} for p in components]}
def compare_predictions(reference,candidate):
 layers={"raw_weights":"raw_weights64","API_probabilities":"probabilities64","sampling_interval_weights":"sampling_interval_weights64"}
 rc,cc=_components(reference),_components(candidate)
 members=[]
 for i,(a,b) in enumerate(zip(rc,cc)):
  ac=a["member"];bc=b["member"]
  record={"member":i,"configuration_identity_equal":all(ac["configuration_identity"][k]==bc["configuration_identity"][k] for k in ("type","sha256")),
   "exact_source_borders":np.array_equal(_np(ac["from_borders"]),_np(bc["from_borders"]))
    and np.array_equal(_np(ac["to_borders"]),_np(bc["to_borders"]))}
  record.update({name:compare_weights(_np(a[key]),_np(b[key])) for name,key in layers.items()});members.append(record)
 result={"same_actual_model_inputs":True,"exact_F64_criterion_borders":np.array_equal(_np(reference["criterion"].borders),_np(candidate["criterion"].borders)),
  "all8_components":True,"members":members}
 result.update({name:compare_weights(_np(reference[key]),_np(candidate[key])) for name,key in layers.items()})
 result["passed"]=comparison_decision(result)
 return result
def legacy_FP32_diagnostic(prediction,torch):
 """CPU same-adjusted-logit legacy replay; NOT captured original GPU softmax."""
 from types import FunctionType
 from posteriorforge import compat_cdf,execution_mixture
 from tabpfn import utils
 helper=compat_cdf._corrected_helper(utils._cdf)
 scope=dict(utils.translate_probs_across_borders.__globals__);scope["_cdf"]=helper
 translator=FunctionType(utils.translate_probs_across_borders.__code__,scope)
 raw=[]
 for p in _components(prediction):
  capture=p["probability_capture"]["translation"]
  raw.append(translator(capture["adjusted_logits_F32"],frm=capture["from_borders_F32"],to=capture["to_borders_F32"]))
 mixture=torch.stack(raw).mean(0);logits=mixture.log()
 borders=prediction["criterion"].borders.to(torch.float32)
 observed=[]
 for u in np.asarray(UNIFORM_BITS,np.uint32).view(np.float32):
  values=execution_mixture.finite_bar_icdf(logits,borders,torch.full((len(logits),),float(u),dtype=torch.float32))
  observed.append({"uniform_bits":int(np.asarray(u).view(np.uint32)),"nonfinite":int((~torch.isfinite(values)).sum()),
   "outside_finite_borders":int(((values<borders[0])|(values>borders[-1])).sum())})
 return {"scope":"CPU-recomputed legacy probability/API path on same actual adjusted F32 neural logits; not historical GPU softmax/extra neural arm",
  "qualifies_v3":False,"forward_calls_added":0,"old_API_inverse_outputs":observed,
  "raw_mixture_comparison_to_v3":compare_weights(_np(mixture),_np(prediction["raw_weights64"]))}


class ControlHardFailure(RuntimeError):
 pass
def make_session(backend,descriptor):
 """Pass the already-active probability owner; never accidentally create a nested scope."""
 return backend.PriorNativeSession(probability_descriptor=descriptor)
def input_fingerprint(mt,ms,mq,sq):
 return {k:_descriptor(v) for k,v in (("theta",mt),("stats",ms),("main_query",mq),("stress_query",sq))}
def control_hard_decision(row):
 expected={"original_"+q for q in QUERIES}
 require(expected.issubset(row.get("hard_predictions",{})),"complete original hard predictions required")
 return all(probability_hard(row["hard_predictions"][k]) for k in expected)

def _capture_original(control,features,stage,session,backend,probability):
 parts=[backend.stock_uncached_predict_component(control,features,stage,m,session,capture_member=True) for m in range(8)]
 return probability.assemble_shared_raw64(parts,stage0=False)
def _capture_candidate(engine,features,backend,probability):
 parts=[engine.predict_component(m,features,capture_member=True) for m in range(8)]
 return probability.assemble_shared_raw64(parts,stage0=False)
def _neural_witness(report,stage,backend,original=False):
 neural=report.get("neural_report",report)
 require(neural.get("aliases_and_flags_restored") is True,"neural aliases/flags not restored")
 require(neural.get("mode")==backend.CONTROL_MODE and neural.get("fallback_map")=={},"actual strict neural mode differs")
 require(neural.get("counters",{}).get("witness_histogram_overflow",0)==0,"neural witness histogram overflow")
 witness=neural.get("witnesses",[])
 require(sum(x["count"] for x in witness)==neural["counters"]["qkv_calls"]==neural["counters"]["attention_calls"],"actual neural QKV/attention counts differ")
 queried={x["member"] for x in witness if x.get("phase")=="query" and x.get("stage")==stage}
 require(queried==set(range(8)),"all8 actual query witnesses required")
 backend.cached._phase_rows(neural,stage,"query",range(8),backend.CONTROL_MODE)
 require(all(x.get("autocast_enabled") is False and x.get("allow_tf32") is False
  and x.get("matmul_precision")=="highest" for x in witness),"strict actual neural flags differ")
 if original or stage in NATIVE_STAGES:
  require(not any(x.get("phase")=="prefill" and x.get("stage")==stage for x in witness),"original preparation cannot claim cached prefill")
 return True
def run(*,spec,banks,prefix_run,checkpoint,codec_inventory,codec_reference_inventory,
 codec_cpu_receipt,completed_grid_audit,pair_cpu_receipt,output,ledger,max_seconds=1050):
 """Future Root-authorized budgeted screen only; current None pins fail before Torch."""
 import gc,os,random,time
 from contextlib import ExitStack
 from posteriorforge.banks import derive_seed
 from posteriorforge.inference import CHECKPOINT_SHA256,frozen_regressor_settings
 from posteriorforge.reference_grid import _require_budgeted_gpu
 require(type(spec) is pairs.PairSpec and spec==pairs.FIRST_PAIR,"only first seed11/case0 screen may run")
 pins=source_pins();committed=committed_source_receipt();backend,probability,proposal=_modules()
 require(type(max_seconds) is int and 60<=max_seconds<=1050,"bounded helper deadline required")
 budget=_require_budgeted_gpu(ledger);require(budget["gpus"]==1,"one budgeted GPU required")
 output=Path(output);require(not output.exists(),"fresh v3 output required")
 require(pairs.sha(checkpoint)==CHECKPOINT_SHA256,"immutable checkpoint differs")
 bound=bound_inputs(spec,banks=banks,completion_receipt=completed_grid_audit,
  training_inventory=codec_inventory,pair_cpu_receipt=pair_cpu_receipt,prefix_run=prefix_run)
 data=data_helper();old=data.load("qualify_ss15_conditionals.py",data.SS15_QUALIFIER_SHA256)
 utility=old.load("qualify_execution_cache.py",old.UTILITY_SHA)
 base=old.load("characterize_precision_approximation.py",old.BASE_SHA)
 inventory=data.inventory_receipt(codec_inventory,codec_reference_inventory,codec_cpu_receipt)[1]
 replay={"training_inventory":inventory,"pair":spec.as_dict(),"pair_CPU_receipt_sha256":PAIR_CPU_RECEIPT_SHA256,
  "COMMON_Float32_model_bytes_match":True,"Float64_cross_environment_byte_identity_claimed":False,"ancestry_scope":"data only; zero v3 neural credit"}
 report={"schema":SCHEMA,"status":"running","pair":spec.as_dict(),"sources":pins,
  "git_revision":committed["git_revision"],"committed_source_closure":committed,
  "raw_bindings":bound.raw,"transformed_bindings":bound.transformed,"prefix_custody":bound.custody,
  "pair_CPU_receipt_sha256":PAIR_CPU_RECEIPT_SHA256,"v3_preregistration_sha256":V3_PROTOCOL_SHA256,
  "coordinate_policy":backend.cached.policy_metadata(),"native_execution_policy":backend.policy_metadata(),
  "probability_policy":probability.metadata(),"proposal_policy":proposal.contract_metadata(),
  "codec_inventory":replay,"thresholds":dict(THRESHOLDS),"native_all8_stages":[6,10,14],"cached_stage_count":18,
  "row_counts":[1,1024],"stress_rows":8,"old_numerical_qualification_inherited":False,
  "FP16_activated":False,"posterior_vectors_generated":0,"reference_integrations":0,"stage_results":[],
  "legacy_FP32_diagnostic_scope":"same-actual-adjusted-logit CPU replay, not captured original GPU softmax; zero extra neural forwards",
  "budget":budget}
 output.mkdir(parents=True);write_json(output/"manifest.json",report)
 started=time.perf_counter();deadline=started+max_seconds;descriptor=None;entry_state=None;torch=None
 try:
  import torch
  from posteriorforge import execution_cache as cache
  require(torch.cuda.is_available() and torch.cuda.device_count()==1,"one visible GPU required")
  require(torch.get_default_dtype()==torch.float32 and torch.get_float32_matmul_precision()=="highest"
   and not torch.backends.cuda.matmul.allow_tf32 and not torch.is_autocast_enabled("cuda"),"strict outer FP32 neural settings required")
  torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS","16")))
  entry_state=utility.rng_state(torch);report["default_RNG_entry"]=utility.rng_identity(entry_state)
  seed=derive_seed(spec.context_seed,"prior-coordinate-fixed-conditionals-v1",case=spec.case)
  random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
  initial=utility.rng_identity(utility.rng_state(torch))
  report["qualification_seed"]=seed;report["default_RNG_after_explicit_setup"]=initial
  with probability.production_scope() as descriptor:
   for stage in STAGES:
    require(time.perf_counter()+25<deadline,"bounded deadline; partial screen cannot qualify")
    main,stress,noise=data.fixed_queries(bound.query,bound.prefixes,bound.theta,stage,backend)
    mt,ms,mq,sq,binding=data.masked_inputs(bound.theta,bound.stats,main,stress,stage,backend)
    require(binding==bound.cpu_row["inference"]["all21_mask_target_query_bindings"][stage],"actual model fixture differs from frozen6f35")
    queries={"1":(mq[:1],main[:1]),"1024":(mq,main),"stress":(sq,stress)}
    actual_inputs=input_fingerprint(mt,ms,mq,sq)
    row={"stage_index":stage,"route":backend.route(stage),"mask_inputs":binding,
     "stress_noise_stage_indices":noise,"comparisons":{},"hard_predictions":{},"legacy_FP32_diagnostics":{}}
    report["stage_results"].append(row)
    before=utility.rng_identity(utility.rng_state(torch));count_before=descriptor["translation_call_count"]
    settings=frozen_regressor_settings(checkpoint);settings["inference_precision"]=torch.float32
    _,sdk,_=cache._load_sdk();control=sdk.TabPFNRegressor(**settings)
    references={}
    with make_session(backend,descriptor) as original_session:
     with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
      control.fit(np.concatenate((ms,mt[:,:stage]),axis=1),mt[:,stage])
     require(control.fit_mode=="fit_preprocessors" and not control.model_.training
      and not control.use_autocast_ and control.forced_inference_dtype_==torch.float32,"actual original FP32 control changed")
     control_immutable=backend.neural.immutable_native_snapshot(SimpleNamespace(regressor=control))
     row["original_control_immutable_state"]=control_immutable
     for key,(features,unmasked) in queries.items():
      p=_capture_original(control,features,stage,original_session,backend,probability)
      references[key]=p
      row["hard_predictions"]["original_"+key]=prediction_record(p,unmasked,probability,proposal,torch)
      row["legacy_FP32_diagnostics"][key]=legacy_FP32_diagnostic(p,torch)
      require(backend.neural.immutable_native_snapshot(SimpleNamespace(regressor=control))==control_immutable,"original weights/fitted context mutated")
    row["original_neural_session"]=original_session.report();_neural_witness(row["original_neural_session"],stage,backend,original=True)
    original_calls=descriptor["translation_call_count"]-count_before
    row["probability_translation_counts"]={"original":original_calls,"candidate":0}
    row["original_model"]=base.model_summary(control,utility)
    row["actual_input_fingerprint_before"]=actual_inputs
    row["original_RNG_after"]=utility.rng_identity(utility.rng_state(torch))
    require(row["original_RNG_after"]==before and source_pins()==pins,"original control changed default RNG/source")
    require(input_fingerprint(mt,ms,mq,sq)==actual_inputs,"original control modified actual F32 inputs")
    require(original_calls==24 and old.state_guard(row["original_model"]),"original control coverage/normalization hard failure")
    # Preserve completed control facts and stop BEFORE any candidate if its hard evidence is invalid.
    if not control_hard_decision(row):
     row["failure_kind"]="original_control_hard_invalid";row["candidate_activated"]=False
     write_json(output/"manifest.json",report);raise ControlHardFailure("original v3 control hard validity failed")
    with make_session(backend,descriptor) as session:
     with ExitStack() as cleanup:
      engine=backend.PriorNativeStageExecutor(checkpoint);cleanup.callback(engine.close)
      prepared=engine.prepare(mt,ms,stage);row["prepared"]=prepared
      row["preparation_witness"]=backend.preparation_witness(session.report(),stage,prepared)
      require(row["preparation_witness"].get("passed") is True,"actual candidate preparation dtype/coverage failed")
      row["state_scope"]="native planned N1 warm baseline; immutable fitted state" if stage in NATIVE_STAGES else "unchanged88 cached prepared-before/after contract"
      cold=base.model_summary(engine.regressor,utility);anchor=cold;immutable_ok=True
      snapshots=[];state_equal=True
      if stage in NATIVE_STAGES:
       snapshots.append(backend.neural.immutable_native_snapshot(engine))
       require(snapshots[0]==control_immutable,"native original all8 weights/targets/context differ")
      for key,(features,unmasked) in queries.items():
       candidate=_capture_candidate(engine,features,backend,probability)
       row["hard_predictions"]["candidate_"+key]=prediction_record(candidate,unmasked,probability,proposal,torch)
       row["comparisons"][key]=compare_predictions(references[key],candidate)
       observed=base.model_summary(engine.regressor,utility)
       if stage in NATIVE_STAGES:
        snapshots.append(backend.neural.immutable_native_snapshot(engine));immutable_ok=immutable_ok and snapshots[-1]==snapshots[0]
        if key=="1":anchor=observed;row["native_model_after_planned_N1"]=observed
        else:state_equal=state_equal and observed==anchor
       else:state_equal=state_equal and observed==cold
       del candidate
      after=base.model_summary(engine.regressor,utility)
      row["model_cold"]=cold;row["model_after"]=after;row["immutable_native_snapshots"]=snapshots
      state_equal=state_equal and after==anchor
      stats_equal=all(cold[k]==row["original_model"][k]==after[k] for k in ("target_mean","target_sd","temperature"))
      pp_equal=cold["contract"]["preprocessor_sha256"]==row["original_model"]["contract"]["preprocessor_sha256"]==after["contract"]["preprocessor_sha256"]
      normalizers=all(old.state_guard(v) for v in (cold,row["original_model"],after))
    row["candidate_neural_session"]=session.report();_neural_witness(row["candidate_neural_session"],stage,backend)
    row["probability_translation_counts"]["candidate"]=descriptor["translation_call_count"]-count_before-original_calls
    row["RNG_identity_before"]=before;row["RNG_identity_after"]=utility.rng_identity(utility.rng_state(torch))
    actual_counts=row["probability_translation_counts"]=={"original":24,"candidate":24}
    row["guards"]={
     "sources_unchanged":source_pins()==pins,
     "identical_model_inputs":input_fingerprint(mt,ms,mq,sq)==actual_inputs and data.masked_inputs(bound.theta,bound.stats,main,stress,stage,backend)[4]==binding,
     "latent_target_bytes_unchanged":mt[:,stage].tobytes()==bound.theta[:,stage].tobytes(),
     "FP32_neural_statistics_and_normalizers":normalizers,
     "target_statistics_preserved":stats_equal,"preprocessor_identity_preserved":pp_equal,
     "state_contract_preserved":state_equal,"immutable_native_state_preserved":immutable_ok,
     "default_RNG_preserved":row["RNG_identity_before"]==row["RNG_identity_after"],
     "actual_neural_dtype_witnesses":True,
     "actual_FP64_probability_witnesses":all(probability_hard(v) for v in row["hard_predictions"].values()),
     "probability_source_executed":actual_counts,"aliases_and_flags_restored":True,
     "no_legacy_qualification_inherited":report["old_numerical_qualification_inherited"] is False}
    row.update(stage_decision(row));write_json(output/"manifest.json",report)
    del control,references,engine;gc.collect();torch.cuda.synchronize()
    if not row["hard_validity_passed"]:break
  report["probability_scope"]=dict(descriptor)
  require(utility.rng_identity(utility.rng_state(torch))==initial,"default RNG changed inside numerical screen")
  require(source_pins()==pins,"v3 source closure changed")
  report["sources_unchanged"]=True
  rows=report["stage_results"];complete=[r["stage_index"] for r in rows]==list(STAGES)
  failed=[r["stage_index"] for r in rows if not r["matching_v3_execution_passed"]]
  report["v3_failed_stages"]=failed;report["matching_v3_execution_passed_all21"]=complete and not failed
  report["status"]=PASS_STATUS if report["matching_v3_execution_passed_all21"] else "prior_coordinate_pair_v3_numeric_or_hard_failure"
 except BaseException as error:
  report.update(status="failed_or_incomplete_prior_coordinate_pair_v3_screen",error_type=type(error).__name__,error=str(error))
  raise
 finally:
  if descriptor is not None:report["probability_scope"]=dict(descriptor)
  if entry_state is not None:
   report["default_RNG_before_restore"]=utility.rng_identity(utility.rng_state(torch))
   utility.restore_rng(torch,entry_state);report["default_RNG_after"]=utility.rng_identity(utility.rng_state(torch))
   report["global_RNG_preserved"]=report["default_RNG_entry"]==report["default_RNG_after"]
  report["elapsed_seconds"]=time.perf_counter()-started;write_json(output/"manifest.json",report)
 if report["status"]==PASS_STATUS:
  report["replay"]=validate_screen(report,spec=spec,expected_sources=pins,raw_bindings=bound.raw,
   transformed_bindings=bound.transformed,coordinate_policy=backend.cached.policy_metadata(),
   native_policy=backend.policy_metadata(),probability_policy=probability.metadata(),
   proposal_policy=proposal.contract_metadata(),inventory_replay=replay,pair_cpu_receipt=bound.cpu)
  write_json(output/"manifest.json",report)
 return report
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--context-seed",type=int,required=True);p.add_argument("--case",type=int,required=True)
 for key in ("banks","prefix-run","checkpoint","codec-inventory","codec-reference-inventory","codec-cpu-receipt","completed-grid-audit","pair-cpu-receipt","output","ledger"):
  p.add_argument("--"+key,required=True)
 p.add_argument("--max-seconds",type=int,default=1050);args=vars(p.parse_args())
 spec=pairs.PairSpec(args.pop("context_seed"),args.pop("case"));report=run(spec=spec,**args)
 print(json.dumps({k:report.get(k) for k in ("schema","status","pair","matching_v3_execution_passed_all21","v3_failed_stages")},sort_keys=True))
if __name__=="__main__":main()
