"""Separate FP16/FIFO full-50k experiment; no exact-FP32 qualification claim.

Built with PriorLabs-TabPFN. Requires an authenticated source checkout/Git history,
PYTHONPATH=<checkout>/src, and an immutable all21 hard-screen artifact. Frozen
sampling, whole-vector rejection, FIFO and e71 inverse arithmetic are reused.
"""
from __future__ import annotations
import argparse,ast,copy,hashlib,importlib.util,json,os,subprocess,sys,time
from contextlib import ExitStack,contextmanager
from importlib import import_module
from pathlib import Path
from types import ModuleType
import numpy as np
from .artifacts import runtime_identity,sha256_file,write_json

SCHEMA="posteriorforge.inference.fp16_fifo_experiment.v1"
VARIANT="execution_fp16_fifo_experiment_v1"
MODE="amp_fp16_prefill_query_v1"
COUNT=50000
SAMPLING_COMMIT="36594f45debeaf1921b10b7bc12a1daa8539ee37"
SAMPLING_SHA="23ccad8b5bbcdd7a48340b349d515ca870763359cfeb71333e02fc2b6a8aca3b"
ALL21_SOURCE_SHA="20e64a64357de7862c20b2fa846c6d128c181ca564c261800b626e6c4bd32208"
PRECISION_SHA="051f477ff4b9ab8bbcfbaf3c1afa4dcf6b111d2f2fbcf9efb9c94a00d2b2ad3d"
BASE_SHA="c55c00a1697f8d4341880af18676c29b4d99551e131c89ff79bea238c693f8bd"
# Immutable all21 approximate hard screen5406017; every strict drift flag stays
# false. This pin authorizes experimental assessment, not posterior equivalence.
ALL21_MANIFEST_SHA256="f01bfddb87dcd16c9bd7c9df15f482ed3ea46756b1d84f3c04f6584644a0687a"
_FROZEN=None
_FROZEN_CODES=None

def require(value,message):
 if not value:raise RuntimeError(message)
def source_sha256():return sha256_file(__file__)
def repository():return Path(__file__).resolve().parents[2]
def load_script(name,expected):
 path=repository()/"scripts"/name
 require(path.is_file() and sha256_file(path)==expected,"required source-checkout helper changed: "+name)
 spec=importlib.util.spec_from_file_location("_fp16_experiment_"+path.stem,path)
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def frozen_sampling():
 """Execute only the authenticated canonical23cc Git blob in a private module."""
 global _FROZEN,_FROZEN_CODES
 root=repository();path=root/"src/posteriorforge/performance_runner.py"
 raw=subprocess.run(["git","-C",str(root),"show",SAMPLING_COMMIT+":src/posteriorforge/performance_runner.py"],
     check=True,capture_output=True).stdout
 require(hashlib.sha256(raw).hexdigest()==SAMPLING_SHA,"canonical frozen sampling Git bytes changed")
 require(path.is_file() and sha256_file(path)==SAMPLING_SHA,"explicit reviewed source checkout required")
 names=("stage_major_wave","component_stage_sampler","worker_stream_seeds","proposal_buffer_policy",
        "proposal_buffer_scope","validate_proposal_buffer_record","guarded_proposal_buffer")
 if _FROZEN is None:
  module=ModuleType("posteriorforge._fp16_authenticated_sampling_v4");module.__package__="posteriorforge"
  module.__file__=str(path);exec(compile(raw,str(path),"exec",dont_inherit=True),module.__dict__)
  _FROZEN=module;_FROZEN_CODES={name:getattr(module,name).__code__ for name in names}
 require(all(getattr(_FROZEN,n).__code__ is _FROZEN_CODES[n] for n in names),"private frozen sampling function changed")
 return _FROZEN

def execution_metadata():
 from . import execution_precision_probe as precision
 require(precision.source_sha256()==PRECISION_SHA,"frozen051 precision source changed");precision.require_sources()
 frozen=frozen_sampling();all21=load_script("characterize_fp16_all_stages.py",ALL21_SOURCE_SHA)
 return {"schema":"posteriorforge.fp16_fifo_execution_policy.v1","module_sha256":source_sha256(),
   "attribution":"Built with PriorLabs-TabPFN","approximate_experiment":True,"qualified_fp32":False,
   "frozen_sampling_commit":SAMPLING_COMMIT,"frozen_sampling_source_sha256":SAMPLING_SHA,
   "source_checkout_required":True,"PYTHONPATH_requirement":"<authenticated checkout>/src",
   "precision_module":precision.metadata(MODE),"all21_characterizer_source_sha256":ALL21_SOURCE_SHA,
   "all21_manifest_sha256":ALL21_MANIFEST_SHA256,"stage0_policy":all21.STAGE0_POLICY,
   "proposal_buffer_policy":frozen.proposal_buffer_policy(1200),"query_batch_size":1024,
   "host_stage_reuse":False,"stage0_density_scope":"one proposal wave; no cross-request/case reuse",
   "stage0_real_head_contexts_per_shared_density":8,"stage0_original_full_API_identity_claimed":False,
   "sampler_math_changed":False,"tail_family_changed":False,"repairs_performed":False}

def _probability(item):
 require(item.get("passed") is True,"hard probability self-check failed")
 for key,limit in (("max_normalization_error",1e-5),("max_cdf_absolute_error",2e-5),("max_row_total_variation",1e-4)):
  value=item.get(key);require(type(value) in (int,float) and np.isfinite(value) and 0<=value<=limit,"hard probability tolerance changed: "+key)
 require(np.isfinite(item.get("minimum_probability",np.nan)) and item["minimum_probability"]>=0,"negative/nonfinite probabilities")

def _raw_validity(item):
 require(item.get("passed") is True and item.get("mass_envelope")==1.00001
     and item.get("normalization_absolute")==1e-5 and 0<=item.get("max_normalization_error",np.inf)<=1e-5,"hard mass/normalization failed")
 require(len(item.get("raw_head_masses",[]))==8,"all8 raw mass records required")
 for mass in item["raw_head_masses"]:require(0<mass["min"]<=mass["max"]<=1.00001,"head mass outside original envelope")
 require(len(item.get("actual_component_mass_checks",[]))==8,"all8 actual FP64 mass checks required")
 for mass in item["actual_component_mass_checks"]:
  require(mass["passed"] is True and 0<=mass["max_fp64_sum_error"]<=mass["fp64_roundoff_bound"]
      and mass["fp64_roundoff_bound"]==8*5000*np.finfo(np.float64).eps,"actual FP64 head mass differs")

def validate_all21(path,bindings):
 require(isinstance(ALL21_MANIFEST_SHA256,str) and len(ALL21_MANIFEST_SHA256)==64,
         "actual all21 immutable manifest SHA is unset; no FP16 posterior experiment permitted")
 raw=Path(path).read_bytes();require(hashlib.sha256(raw).hexdigest()==ALL21_MANIFEST_SHA256,"all21 immutable artifact SHA changed")
 q=json.loads(raw);all21=load_script("characterize_fp16_all_stages.py",ALL21_SOURCE_SHA)
 base=load_script("characterize_precision_approximation.py",BASE_SHA)
 from . import execution_precision_probe as precision
 expected={"schema":all21.SCHEMA,"status":"all21_fp16_approximation_characterized_not_qualified",
   "mode":MODE,"qualified":False,"full21_qualified":False,"all21_hard_validity_passed":True,
   "completed_stages":list(range(21)),"requested_stages":list(range(21)),
   "thresholds":all21.THRESHOLDS,"row_counts":[1024],"stress_rows":8,"sources_unchanged":True,
   "posterior_vectors_generated":0,"reference_integrations":0,"stage0_policy":all21.STAGE0_POLICY,
   "precision_module":precision.metadata(MODE),**bindings}
 for key,value in expected.items():require(q.get(key)==value,"all21 experimental binding mismatch: "+key)
 require(q.get("sources",{}).get("characterizer")==ALL21_SOURCE_SHA and q["sources"].get("precision")==PRECISION_SHA,"all21 producer source changed")
 rows=q.get("stage_results",[]);require([r.get("stage_index") for r in rows]==list(range(21)),"all21 stage records required")
 for row in rows:
  j=row["stage_index"];calculated=all21.outcome(row,base)
  require(calculated["hard_validity_passed"] and all(row.get(k)==v for k,v in calculated.items()),"all21 saved hard/strict predicates differ")
  require(precision.prefill_witness_passed(row["precision_session"],j)==row["computed_prefill_witness"]
      and precision.query_witness_passed(row["precision_session"],j)==row["computed_query_witness"],"all21 actual dtype witnesses differ")
  require(row["precision_session"]["module_sha256"]==PRECISION_SHA,"all21 precision scope source differs")
  for name in ("1024","stress"):
   _raw_validity(row["probability_validity"][name]["Cp"]);_raw_validity(row["probability_validity"][name]["G0"])
   for key,value in row["Cp_self_reconstruction"][name].items():
    if key!="scope":_probability(value)
  if j==0:
   require(row["stage0"]["G0_reference_is_actual_original_full_API"] is True
       and row["stage0"]["Cp_original_full_API_identity_claimed"] is False,"stage0 policy identity changed")
   for key,value in row["stage0"]["shared_self_reconstruction"].items():
    if key!="scope":_probability(value)
 require(q["strict_failed_stages"]==[r["stage_index"] for r in rows if not r["strict_screen_passed"]]
     and q["all21_strict_screen_passed"]==all(r["strict_screen_passed"] for r in rows),"strict failures were relabeled")
 adapter=q["compatibility_adapter"]
 require(adapter["original_alias_restored"] and adapter["sdk_sources_unchanged"]
     and adapter["executed_corrected_cdf_code_sha256"]==adapter["corrected_cdf_code_sha256"]
     and adapter["original_cdf_code_sha256"]!=adapter["corrected_cdf_code_sha256"],"all21 CDF execution/restoration proof failed")
 return q

def query_witness_delta(before,after,*,stage,member):
 rows=[]
 for key,count in after.items():
  delta=count-before.get(key,0)
  if delta:
   event=json.loads(key);require(delta>0 and event["phase"]=="query" and event["stage"]==stage and event["member"]==member,
       "actual precision callback identity changed");rows.append(event)
 strict=stage==10 and member==5;expected="torch.float32" if strict else "torch.float16"
 require(rows,"actual query projection not witnessed")
 for event in rows:
  values=[v for v in event["projection_dtypes"].values() if v is not None]
  require(values and all(v in ("torch.float32",expected) for v in values),"query projection dtype outside policy")
  require(event["autocast_enabled"] is (not strict) and event["allow_tf32"] is False
       and event["matmul_precision"]=="highest","actual query AMP/TF32 flags differ")
  if not strict:require("torch.float16" in values and event["autocast_dtype"]=="torch.float16","FP16 query activation absent")
  else:require(all(v=="torch.float32" for v in values),"fallback query left FP32")
 return {"phase":"query","stage":stage,"member":member,"event_types":len(rows),
     "computed_projection_dtype":expected,"strict_FP32_fallback":strict,"passed":True}

class FP16StageAdapter:
 """One stage/session at a time; every physical head call has its own witness."""
 def __init__(self,checkpoint,*,cdf_descriptor):
  self.checkpoint=checkpoint;self.cdf_descriptor=cdf_descriptor;self.engine=None;self.session=None
  self.stage_index=None;self.actual_heads=0;self.shared_densities=0;self.prepare_count=0;self.stages=[];self.calls=[]
  self.failure=False;self.entered=False
 @property
 def regressor(self):return self.engine.regressor if self.engine is not None else None
 def prepare(self,theta,obs,stage):
  require(self.session is None,"previous precision stage not closed")
  from . import execution_precision_probe as precision
  self.stage_index=stage;self.failure=False;self.stage_heads=0;self.members=set();self.prefill=None;self.model=None;self.entered=False
  self.session=precision.PrecisionSession(MODE)
  try:
   self.session.__enter__();self.entered=True
   self.engine=precision.PrecisionStageExecutor(self.checkpoint,cdf_descriptor=self.cdf_descriptor,query_batch_size=1024)
   result=self.engine.prepare(theta,obs,stage);self.prepare_count+=1
   self.prefill=precision.prefill_witness_passed(self.session.report(),stage)
   require(self.prefill["passed"],"actual FP16 prefill/cache/fallback not witnessed")
   base=load_script("characterize_precision_approximation.py",BASE_SHA)
   self.model=base.model_summary(self.engine.regressor,load_script("qualify_execution_cache.py",base.UTILITY_SHA))
   return result
  except BaseException:
   self.failure=True;self.close();raise
 def predict_component(self,member,query,*,capture_member=False,with_log_prob=False):
  require(self.engine is not None and self.session is not None,"prepared precision stage required")
  import torch
  before=dict(self.session.hist);cdf_before=self.cdf_descriptor["patched_call_count"]
  try:
   prediction=self.engine.predict_component(member,query,capture_member=capture_member,with_log_prob=with_log_prob)
   witness=query_witness_delta(before,dict(self.session.hist),stage=self.stage_index,member=member)
   expected_route="original_fit_preprocessors" if self.stage_index==10 and member==5 else "resident_cache"
   require(prediction.get("hybrid_execution_path")==expected_route,"actual cached/FP32 fallback route differs")
   require(self.cdf_descriptor["patched_call_count"]-cdf_before==1,"one actual corrected CDF call per head required")
   require(self.cdf_descriptor.get("executed_corrected_cdf_code_sha256")==self.cdf_descriptor.get("corrected_cdf_code_sha256"),"corrected CDF not executed")
   logits=prediction["logits"];mass=prediction["component_raw_mass"]
   require(logits.dtype==torch.float32 and logits.device.type=="cpu"
       and bool((torch.isfinite(logits)|torch.isneginf(logits)).all()) and bool(torch.isfinite(logits).any(-1).all()),"invalid postprocessing logits")
   require(mass.dtype==torch.float64 and bool(torch.isfinite(mass).all()) and bool((mass>0).all())
       and bool((mass<=1.00001).all()) and prediction["head_mass_envelope"]==1.00001,"original raw mass guard failed")
   require(prediction["criterion"].borders.dtype==torch.float32 and bool(torch.isfinite(prediction["criterion"].borders).all()),"invalid physical borders")
   self.actual_heads+=1;self.stage_heads+=1;self.members.add(member)
   proof={"schema":"posteriorforge.fp16_actual_prediction.v1","module_sha256":source_sha256(),"precision_module_sha256":PRECISION_SHA,
      "mode":MODE,"stage":self.stage_index,"member":member,"actual_head_contexts":1,"query_rows":len(query),
      "witness":witness,"execution_route":expected_route,"corrected_CDF_calls":1,"original_full_API_identity_claimed":False}
   prediction["fp16_execution_proof"]=proof;return prediction
  except BaseException:self.failure=True;raise
 def predict(self,query,*,output_type="full",capture_members=False):
  require(self.stage_index==0 and len(query)==1 and output_type=="full","only declared stage0 singleton aggregate supported")
  from . import execution_precision_probe as precision
  import torch
  start=self.actual_heads
  components=precision.capture_members(self,query)
  all21=load_script("characterize_fp16_all_stages.py",ALL21_SOURCE_SHA)
  prediction=all21.assemble_shared_stage0(components,torch,self.engine.base.device)
  require(self.actual_heads-start==8,"shared density requires8 real head contexts")
  self.shared_densities+=1
  return {"criterion":prediction["criterion"],"logits":prediction["logits"],
    "fp16_execution_proof":{"schema":"posteriorforge.fp16_actual_prediction.v1","module_sha256":source_sha256(),
       "precision_module_sha256":PRECISION_SHA,"mode":MODE,"stage":0,"member":None,"actual_head_contexts":8,
       "shared_density_index":self.shared_densities,"corrected_CDF_calls":8,"stage0_policy":all21.STAGE0_POLICY,
       "original_full_API_identity_claimed":False}}
 def inspect_cache(self):return self.engine.inspect_cache()
 def close(self):
  if self.session is None:return
  if not self.entered:
   self.engine=None;self.session=None;return
  session=self.session;engine=self.engine;routes={}
  try:
   if engine is not None:
    routes={k:v for k,v in getattr(engine,"hybrid_counters",{}).items() if k!="stages"}
    engine.close()
    if not self.failure:
     require(routes.get("resident_component_predict_count",0)+routes.get("uncached_predict_count",0)==self.stage_heads,
             "actual resident/fallback route counts differ from physical head contexts")
  finally:
   try:session.__exit__(None,None,None)
   finally:
    report=session.report()
    self.stages.append({"stage_index":self.stage_index,"actual_head_contexts":getattr(self,"stage_heads",0),
      "queried_members":sorted(getattr(self,"members",set())),"failed":self.failure,"prefill":getattr(self,"prefill",None),
      "model":getattr(self,"model",None),"hybrid_route_counters":routes,"precision_session":report})
    self.engine=None;self.session=None;self.entered=False
   require(report["aliases_and_flags_restored"] and report["counters"].get("witness_histogram_overflow",0)==0,
      "precision source/alias/flag restoration or witness capacity failed")
 def metadata(self):
  return {"policy":execution_metadata(),"prepare_count":self.prepare_count,"actual_head_contexts":self.actual_heads,
    "shared_densities":self.shared_densities,"completed_stage_scopes":self.stages,
    "all_stage_aliases_restored":all(r["precision_session"]["aliases_and_flags_restored"] for r in self.stages),
    "scope":"actual051 AMP/projection/cache/restoration witnesses; no strict8b2 descriptor claim"}


@contextmanager
def precision_engine_scope(checkpoint,*,cdf_descriptor):
 """Close any entered051 scope before the surrounding corrected-CDF exits.

 The frozen stage-wave implementation prepares/inspects before its own finally;
 this outer ownership guard covers those pre-try failures without changing it.
 """
 with ExitStack() as cleanup:
  engine=FP16StageAdapter(checkpoint,cdf_descriptor=cdf_descriptor)
  cleanup.callback(engine.close)
  yield engine


def validate_prediction(prediction,stage,member,*,report):
 require((member is None)==(stage==0),"only stage0 has a shared density callback")
 proof=prediction.get("fp16_execution_proof",{})
 require(proof.get("mode")==MODE and proof.get("precision_module_sha256")==PRECISION_SHA
   and proof.get("module_sha256")==source_sha256() and proof.get("stage")==stage and proof.get("member")==member,
   "actual FP16 prediction provenance mismatch")
 n=8 if member is None else 1
 require(proof.get("actual_head_contexts")==proof.get("corrected_CDF_calls")==n,"physical head/CDF counts differ")
 if member is not None:require(proof.get("witness",{}).get("passed") is True,"actual per-head witness missing")
 report["sampler_callback_calls"]=report.get("sampler_callback_calls",0)+1
 report["actual_head_contexts"]=report.get("actual_head_contexts",0)+n
 report["shared_densities"]=report.get("shared_densities",0)+int(member is None)
 report["original_full_API_identity_claimed"]=False


def run_experiment(*,banks,output,checkpoint,ledger,reference,reference_grid,approximation,
                   qualification,batch_qualification,sdk_root,repetition=0):
 start=time.perf_counter();output=Path(output);require(not output.exists(),"fresh FP16 experimental output required")
 require(ALL21_MANIFEST_SHA256 is not None,"all21 artifact has not been pinned; posterior launch is disabled")
 require(type(repetition) is int and repetition>=0,"nonnegative repetition required")
 from .inference_cdf import guard_baseline_source,reference_binding
 from .inference import CHECKPOINT_SHA256,frozen_regressor_settings,support_mask,_prior
 from .banks import load_context,load_heldout
 from .science.analytical_benchmark_v2 import array_descriptor,build_problem
 from .reference_grid import _require_budgeted_gpu
 frozen=frozen_sampling();policy=execution_metadata();all21=load_script("characterize_fp16_all_stages.py",ALL21_SOURCE_SHA)
 baseline_source=guard_baseline_source();initial_source=source_sha256()
 require(sha256_file(checkpoint)==CHECKPOINT_SHA256,"immutable checkpoint mismatch")
 context=load_context(banks,20261111,4096);observed=load_heldout(banks,0);query=np.asarray(observed.obs).reshape(1,33)
 problem=build_problem();bindings={"checkpoint_sha256":CHECKPOINT_SHA256,"scientific_fingerprint":problem.scientific_fingerprint,
  "parameter_order":list(problem.parameter_order),"context_theta":array_descriptor(context.theta),
  "context_obs":array_descriptor(context.obs),"query":array_descriptor(query)}
 screen=validate_all21(approximation,bindings)
 _,portable_receipt=all21.portable_qualification(qualification,batch_qualification,sdk_root,bindings)
 require(sha256_file(qualification)==screen["qualification_sha256"] and sha256_file(batch_qualification)==screen["batch_qualification_sha256"],"screen FP32 ancestry differs")
 refs=reference_binding(banks=banks,reference=reference,reference_grid=reference_grid,case=0)
 budget=_require_budgeted_gpu(ledger);require(budget["gpus"]==1,"single GPU FP16 experiment only")
 output.mkdir(parents=True,exist_ok=False);engine=None;descriptor=None;buffer=None
 status={**runtime_identity(),"schema":SCHEMA,"variant":VARIANT,"status":"running","attribution":"Built with PriorLabs-TabPFN",
   "approximate_experiment":True,"qualified_fp32":False,"qualification_full21":False,"qualification_cache_only_full21":False,
   "approximation_all21_hard_screen_passed":True,"all21_strict_screen_passed":screen["all21_strict_screen_passed"],
   "strict_failed_stages":screen["strict_failed_stages"],"approximation_manifest_sha256":ALL21_MANIFEST_SHA256,
   "approximation_producer_sha256":ALL21_SOURCE_SHA,"approximation_scope":screen["scope"],
   "source_checkout_required":True,"runner_source_sha256":initial_source,"execution_policy":policy,
   "context_seed":20261111,"case":0,"repetition":repetition,"context_size":4096,"query_batch_size":1024,"engine_query_batch_cap":1024,
   "requested_count":COUNT,"benchmark_total_count":COUNT,"rank":0,"world_size":1,"end_to_end":True,"scientific_run":True,
   "proposal_lookahead":1200,"proposal_buffer_policy":frozen.proposal_buffer_policy(1200),"proposal_buffer":None,
   "microbatch":COUNT,"max_iter_rejection":None,"log_density":"not_used","context_filter":"no_filtering",
   "observation_preprocessing":"raw; TabPFN frozen preprocessing","performance_first_authorized":True,
   "absolute_accuracy_gate_required_for_performance":False,"known_physical_control_status":"model_failure",
   "candidate_physical_assessment":"pending independent original physical gate; no accuracy pass claim",
   "relative_distribution_review_thresholds":{"CDF_change":.01,"mean_change_refSD":.05,"SD_relative_change":.05},
   "partial_vector_filtering":False,"clipping_or_repair_performed":False,"tail_family_changed":False,"execution_optimized":True,
   "original_gate_unchanged":True,"baseline_execution_source":baseline_source,"budget_binding":budget,
   "reference_binding":refs,"portable_FP32_ancestry_verification":portable_receipt,
   "sampling_rng_policy":"independent_cpu_sampling_streams_model_rng_isolated",
   "timing_boundaries":{"posterior_request":"first proposal through final support/FIFO validation and CUDA synchronization",
     "wrapper":"function entry through NPZ/hash completion; final manifest write/interpreter startup reported by outer job timer"},
   "completed_stage_indices":[],"stage_events":[],"proposal_wave_sizes":[],**bindings}
 write_json(output/"manifest.json",status)
 try:
  import torch
  from . import execution_mixture as mixture
  from .compat_cdf import finite_cdf_compatibility
  from .upstream import load_baseline
  require(torch.cuda.is_available() and torch.cuda.device_count()==1,"one visible CUDA GPU required")
  require(torch.get_default_dtype()==torch.float32 and torch.get_float32_matmul_precision()=="highest"
      and not torch.backends.cuda.matmul.allow_tf32 and not torch.is_autocast_enabled("cuda"),"original strict initial flags required")
  require(np.__version__=="1.26.4","recorded NumPy1.26.4 inference environment required")
  load_baseline();accept_reject=import_module("npe_pfn.accept_reject_sampler").accept_reject_sample
  seeds=frozen.worker_stream_seeds(20261111,0,repetition,0,1)
  torch.manual_seed(seeds["component_uniform"]);torch.cuda.manual_seed_all(seeds["component_uniform"])
  generators={name:torch.Generator(device="cpu").manual_seed(seed) for name,seed in seeds.items()}
  torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS","16")))
  streams={name:{"role":frozen.STREAM_ROLES[name],"seed":seed,"dtype":frozen.STREAM_DTYPES[name],
    "device":"cpu","generator":"torch.Generator"} for name,seed in seeds.items()}
  status.update(seed_streams=seeds,seed=seeds["component_uniform"],gpu=torch.cuda.get_device_name(),torch_cuda=torch.version.cuda,
      regressor_settings={**frozen_regressor_settings(checkpoint),"fit_mode":"fit_with_cache","inference_precision":"autocast"},
      sampling_policy={"schema":"posteriorforge.fp16_sampling_policy.v1","frozen_component_policy_provenance":mixture.execution_metadata(),
       "forward_precision_policy":MODE,"postprocessing_math_dtype":"torch.float32","head_policy":"raw_mass_rejection_v1",
       "uniform_assignment":"preassigned_per_stage_by_physical_row","head_retry_preserves_row_uniform":True,"streams":streams,
       "model_rng_isolation":"CPU and selected CUDA fork_rng in frozen executor","mass_envelope":1.00001,
       "stage0_dedup":{"enabled":True,"scope":"constructed_shared_raw_probability_mixture",
         "policy":all21.STAGE0_POLICY,"component_distribution_dedup":False,"density_scope":"one proposal wave; no cross-request/case reuse"}})
  theta=context.theta.astype(np.float32);obs=context.obs.astype(np.float32);prior=_prior(torch)
  counters={"stage0_shared_predictions":0,"component_predictions":0,"head_candidates":0,"head_rejections":0}
  actual_checks={}
  with finite_cdf_compatibility() as descriptor, precision_engine_scope(checkpoint,cdf_descriptor=descriptor) as engine:
   callback=lambda prediction,stage,member:validate_prediction(prediction,stage,member,report=actual_checks)
   # These two functions are the EXACT authenticated23cc code objects.
   stage_draw=frozen.component_stage_sampler(engine,torch,generators,counters,prediction_check=callback)
   def progress(event):
    status["stage_events"].append({"proposal_round":len(status["proposal_wave_sizes"]),**event})
    status["elapsed_seconds"]=time.perf_counter()-start;write_json(output/"manifest.json",status)
   def proposal(size):
    status["proposal_wave_sizes"].append(int(size))
    values=frozen.stage_major_wave(engine,theta,obs,query,int(size),1024,lambda _:None,stage_draw=stage_draw,progress=progress)
    return torch.from_numpy(values),None
   scope=frozen.proposal_buffer_scope(bindings,context_seed=20261111,case=0,repetition=repetition,rank=0,world_size=1,seeds=seeds)
   fifo=frozen.guarded_proposal_buffer()
   buffer=fifo.ProposalBuffer(proposal,scope=scope,concatenate=lambda parts:torch.cat(parts,dim=0),world_size=1,rank=0)
   torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();request_start=time.perf_counter()
   with torch.no_grad():
    samples,_,_=accept_reject(proposal=buffer,accept_reject_fn=prior.support.check,num_samples=COUNT,
      max_sampling_batch_size=COUNT,max_iter_rejection=None,show_progress_bars=False,proposal_sampling_kwargs={})
   record=buffer.finalize(original_checked_count=prior.support.proposed,accepted_before_trim=prior.support.accepted,
       returned_count=len(samples),original_support_calls=prior.support.calls)
   record.update(enabled=True,source_unchanged=True,policy_sha256=status["proposal_buffer_policy"]["policy_sha256"])
   frozen.validate_proposal_buffer_record(record,world_size=1,rank=0,scope=scope);status["proposal_buffer"]=record
   torch.cuda.synchronize();finished=time.perf_counter()
   values=samples.detach().cpu().numpy().astype(np.float64)
   require(values.shape==(COUNT,21) and support_mask(values).all(),"exact50000 complete finite prior-supported vectors required")
   generated=len(status["proposal_wave_sizes"]);wanted_heads=8*counters["stage0_shared_predictions"]+counters["component_predictions"]
   require(engine.prepare_count==21*generated and len(engine.stages)==21*generated,"actual generated-wave stage coverage differs")
   require(engine.actual_heads==actual_checks["actual_head_contexts"]==wanted_heads==descriptor["patched_call_count"],"actual8-head/shared/CDF counters differ")
   require(engine.shared_densities==counters["stage0_shared_predictions"]==generated,"one shared stage0 density per generated wave required")
   require(record["actual_generated_wave_sizes"]==status["proposal_wave_sizes"],"FIFO/provider wave sizes differ")
   status.update(execution_counters=engine.metadata(),sampling_counters=dict(counters),actual_prediction_checks=actual_checks,
       expected_patched_call_count=wanted_heads,expected_stage_prepare_count=21*generated,generated_proposal_wave_count=generated)
  require(descriptor["original_alias_restored"] and descriptor["sdk_sources_unchanged"],"CDF adapter restoration failed")
  require(source_sha256()==initial_source and frozen_sampling() is frozen and guard_baseline_source()==baseline_source,"source changed during experiment")
  require(sha256_file(approximation)==ALL21_MANIFEST_SHA256,"all21 proof changed during experiment")
  require(sha256_file(qualification)==screen["qualification_sha256"] and sha256_file(batch_qualification)==screen["batch_qualification_sha256"],"FP32 ancestry changed")
  with (output/"samples.npz").open("xb") as stream:np.savez(stream,samples=values)
  status.update(status="complete",samples=array_descriptor(values),samples_file="samples.npz",samples_sha256=sha256_file(output/"samples.npz"),
    completed_stage_indices=list(range(21)),compatibility_adapter=dict(descriptor),sources_unchanged=True,
    rejection={"support_calls":prior.support.calls,"proposed":prior.support.proposed,"accepted_before_final_trim":prior.support.accepted,"returned":len(values)},
    timings={"posterior_request_seconds":finished-request_start,"total_seconds":time.perf_counter()-start,
      "accepted_samples_per_second":COUNT/(finished-request_start),"includes_stage_preparation":True,"includes_CPU_sampling":True},
    memory={"peak_allocated_bytes":torch.cuda.max_memory_allocated(),"peak_reserved_bytes":torch.cuda.max_memory_reserved()})
 except BaseException as error:
  status.update(status="failed",error_type=type(error).__name__,error=str(error))
  if descriptor is not None:status["compatibility_adapter"]=dict(descriptor)
  if buffer is not None:status["proposal_buffer"]=buffer.metadata()
  raise
 finally:
  if engine is not None:engine.close()
  status["wrapper_total_seconds"]=time.perf_counter()-start;write_json(output/"manifest.json",status)
 return status

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 for name in ("banks","output","checkpoint","ledger","reference","reference-grid","approximation","qualification","batch-qualification","sdk-root"):
  parser.add_argument("--"+name,required=True)
 parser.add_argument("--repetition",type=int,default=0)
 result=run_experiment(**vars(parser.parse_args()))
 print(json.dumps({k:result.get(k) for k in ("schema","variant","status","requested_count","timings")},indent=2))

if __name__=="__main__":main()
