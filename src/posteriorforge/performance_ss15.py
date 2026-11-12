"""SS15 accuracy pilot with the reviewed FP16 forward and frozen sampler law.

Built with PriorLabs-TabPFN. Original raw33 banks/queries bind the physical
reference; source-pinned SS15 Float32 inputs bind the learned conditional.
Exactly50,000 complete accepted physical vectors, one GPU, no context filtering.
"""
from __future__ import annotations
import argparse,hashlib,json,os,time
from contextlib import ExitStack,contextmanager
from importlib import import_module
from pathlib import Path
import numpy as np
from . import performance_fp16_experiment as clean
from . import execution_precision_probe as witnesses
from . import execution_ss15_sampling as wave
from . import representation_ss15 as representation
from .artifacts import runtime_identity,sha256_file,write_json

SCHEMA="posteriorforge.inference.ss15_fp16.v1"
VARIANT="accuracy_ss15_fp16_fifo_v1"
MODE=clean.MODE
COUNT=50000
FALLBACK_MAP={10:[5],20:[4]}
CONTROL_MODE=witnesses.CONTROL_MODE
PRECISION_SHA=clean.PRECISION_SHA
ALL21_SOURCE_SHA=clean.ALL21_SOURCE_SHA # frozen raw-P stage0 assembler only, not a raw33 proof.
BASE_SHA=clean.BASE_SHA
REPRESENTATION_SHA256="46323c0b94301c2f7e678538881d5cbbdd8bc52c62a60021798176bffe81eb85"
SS15_EXECUTION_SHA256="1732f7269286f1629f6e09b1b7ea14cfadbaea84cdae9ee4ab5221cf5b78fbbb"
QUALIFIER_SHA256="2ecadae34e5173bebb5127f578233964bfde6820ae38f9ed0baaac054d119816"
CONDITIONAL_QUALIFICATION_SHA256="b192d05e68d59a3518ea9df5c6c8c4f36845fa89418b30d404261615a25d5116"
WAVE_SHA256="4037d48846ef3678ce7b647c1a326606ea967d396b3ce34bb1998241a131f6de"
require=clean.require
load_script=clean.load_script

def source_sha256():return sha256_file(__file__)

def is_fallback(stage,member):return member in FALLBACK_MAP.get(stage,())

# Validation-only copies of the frozen witness checks, with the explicitly
# declared SS15 map replacing the old hardcoded10:5 predicate. No model math.
def prefill_witness_passed(report,stage):
    """Prove the approximate prefill really computed low projections, not a cast."""
    mode=report["mode"]
    expected="torch.bfloat16" if mode=="amp_bf16_prefill_query_v1" else "torch.float16"
    rows=report["witnesses"];result=[]
    for member in range(8):
        selected=[r for r in rows if r["phase"]=="prefill" and r["stage"]==stage and r["member"]==member and r["cache_kv"]]
        strict=is_fallback(stage,member)
        if mode.startswith("amp_") and not strict:
            passed=bool(selected) and any(expected in [r["projection_dtypes"][key] for key in ("k","v","kv","qkv")] and r["autocast_enabled"] for r in selected)
        else:
            passed=bool(selected) and all(not r["autocast_enabled"] and all(value in (None,"torch.float32") for value in r["projection_dtypes"].values()) for r in selected)
        if mode=="tf32_high_forward_v1" and not strict:
            passed=passed and all(r.get("allow_tf32") is True and r.get("matmul_precision")=="high" for r in selected)
        elif strict or mode==CONTROL_MODE:
            passed=passed and all(r.get("allow_tf32") is False and r.get("matmul_precision")=="highest" for r in selected)
        if mode.startswith("amp_") and not strict:
            passed=passed and all(r.get("autocast_dtype")==expected and r["autocast_enabled"] for r in selected)
        storage={value for r in selected for value in r.get("supplied_cache_dtypes",[]) if value is not None}
        allowed={"torch.float32"} if strict or not mode.startswith("amp_") else {"torch.float32",expected}
        passed=passed and bool(storage) and storage.issubset(allowed)
        result.append({"member":member,"passed":passed,"prefill_event_count":sum(r["count"] for r in selected),
                       "actual_stored_cache_dtypes":sorted(storage)})
    return {"passed":all(r["passed"] for r in result),"members":result,"storage_dtype_is_separate_observation":True}

def query_witness_delta(before,after,*,stage,member):
 rows=[]
 for key,count in after.items():
  delta=count-before.get(key,0)
  if delta:
   event=json.loads(key);require(delta>0 and event["phase"]=="query" and event["stage"]==stage and event["member"]==member,
       "actual precision callback identity changed");rows.append(event)
 strict=is_fallback(stage,member);expected="torch.float32" if strict else "torch.float16"
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

def ss15_source_sha():
 from . import execution_ss15
 return execution_ss15.source_sha256()

def require_sources():
 require(clean.source_sha256()=="2280e1ceef807f05a264b5318be18eb8bee7c52c8bab21884d6682f84cafcba9","frozen clean2280 source changed")
 require(representation.source_sha256()==REPRESENTATION_SHA256 and wave.source_sha256()==WAVE_SHA256,"SS15 representation/wave source changed")
 require(isinstance(SS15_EXECUTION_SHA256,str) and ss15_source_sha()==SS15_EXECUTION_SHA256,"matching SS15 execution source is not pinned")


def execution_metadata():
 from . import execution_ss15
 require_sources()
 return {'schema':'posteriorforge.ss15_fp16_execution_policy.v1','module_sha256':source_sha256(),
  'representation':representation.representation_metadata(),'SS15_execution':execution_ss15.execution_metadata(MODE,FALLBACK_MAP),
  'sampling_wave':wave.execution_metadata(),'canonical_component_sampling_source_sha256':clean.SAMPLING_SHA,
  'stage0_rawP_assembler_source_sha256':ALL21_SOURCE_SHA,'postprocessing_dtype':'torch.float32',
  'forward_precision':MODE,'approximate_experiment':True,'qualified_FP32':False,
  'same_physical_likelihood_target':True,'learned_conditional_input_intentionally_changed':True,
  'query_batch_size':1024,'host_stage_reuse':False,'global_proposal_lookahead':1200,
  'whole_vector_rejection':'original unchanged; unlimited iterations; final accepted trim',
  'tail_family_changed':False,'repairs_performed':False,'attribution':'Built with PriorLabs-TabPFN'}

class SS15StageAdapter:
 """One stage/session at a time; every physical head call has its own witness."""
 def __init__(self,checkpoint,*,cdf_descriptor):
  self.checkpoint=checkpoint;self.cdf_descriptor=cdf_descriptor;self.engine=None;self.session=None
  self.stage_index=None;self.actual_heads=0;self.shared_densities=0;self.prepare_count=0;self.stages=[];self.calls=[]
  self.failure=False;self.entered=False
 @property
 def regressor(self):return self.engine.regressor if self.engine is not None else None
 def prepare(self,theta,obs,stage):
  require(self.session is None,"previous precision stage not closed")
  from . import execution_ss15 as precision
  self.stage_index=stage;self.failure=False;self.stage_heads=0;self.members=set();self.prefill=None;self.model=None;self.entered=False
  self.session=precision.SS15PrecisionSession(MODE,fallback_map=FALLBACK_MAP)
  try:
   self.session.__enter__();self.entered=True
   self.engine=precision.SS15StageExecutor(self.checkpoint,cdf_descriptor=self.cdf_descriptor,query_batch_size=1024,fallback_map=FALLBACK_MAP)
   result=self.engine.prepare(theta,obs,stage);self.prepare_count+=1
   self.prefill=prefill_witness_passed(self.session.report(),stage)
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
   expected_route="original_fit_preprocessors" if is_fallback(self.stage_index,member) else "resident_cache"
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
   proof={"schema":"posteriorforge.ss15_actual_prediction.v1","module_sha256":source_sha256(),"SS15_execution_module_sha256":ss15_source_sha(),"ancestral_precision_module_sha256":PRECISION_SHA,
      "mode":MODE,"stage":self.stage_index,"member":member,"actual_head_contexts":1,"query_rows":len(query),
      "witness":witness,"execution_route":expected_route,"corrected_CDF_calls":1,"original_full_API_identity_claimed":False}
   prediction["ss15_execution_proof"]=proof;return prediction
  except BaseException:self.failure=True;raise
 def predict(self,query,*,output_type="full",capture_members=False):
  require(self.stage_index==0 and len(query)==1 and output_type=="full","only declared stage0 singleton aggregate supported")
  from . import execution_ss15 as precision
  import torch
  start=self.actual_heads
  components=precision.capture_members(self,query)
  all21=load_script("characterize_fp16_all_stages.py",ALL21_SOURCE_SHA)
  prediction=all21.assemble_shared_stage0(components,torch,self.engine.base.device)
  require(self.actual_heads-start==8,"shared density requires8 real head contexts")
  self.shared_densities+=1
  return {"criterion":prediction["criterion"],"logits":prediction["logits"],
    "ss15_execution_proof":{"schema":"posteriorforge.ss15_actual_prediction.v1","module_sha256":source_sha256(),
       "SS15_execution_module_sha256":ss15_source_sha(),"ancestral_precision_module_sha256":PRECISION_SHA,"mode":MODE,"stage":0,"member":None,"actual_head_contexts":8,
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
def engine_scope(checkpoint,*,cdf_descriptor):
 # Covers prepare/firstinspect failures before the inherited wave's own try.
 with ExitStack() as cleanup:
  value=SS15StageAdapter(checkpoint,cdf_descriptor=cdf_descriptor)
  cleanup.callback(value.close)
  yield value


def validate_prediction(prediction,stage,member,*,report):
 require((member is None)==(stage==0),'shared callback only at stage0')
 proof=prediction.get('ss15_execution_proof',{})
 require(proof.get('mode')==MODE and proof.get('SS15_execution_module_sha256')==SS15_EXECUTION_SHA256
  and proof.get('ancestral_precision_module_sha256')==PRECISION_SHA and proof.get('module_sha256')==source_sha256()
  and proof.get('stage')==stage and proof.get('member')==member,'actual SS15 prediction provenance differs')
 n=8 if member is None else 1
 require(proof.get('actual_head_contexts')==proof.get('corrected_CDF_calls')==n,'actual SS15 physical head/CDF count differs')
 if member is not None:require(proof.get('witness',{}).get('passed') is True,'actual SS15 head witness missing')
 report['sampler_callback_calls']=report.get('sampler_callback_calls',0)+1
 report['actual_head_contexts']=report.get('actual_head_contexts',0)+n
 report['shared_densities']=report.get('shared_densities',0)+int(member is None)
 report['original_full_API_identity_claimed']=False


def model_bindings(context,raw_query):
 from .science.analytical_benchmark_v2 import array_descriptor
 theta=context.theta.astype(np.float32)
 obs=representation.transform_for_model(context.obs)
 query=representation.transform_for_model(raw_query)
 model={'context_theta':array_descriptor(theta),'context_obs':array_descriptor(obs),'query':array_descriptor(query)}
 qualification={'context_obs_SS15_float64':array_descriptor(representation.transform_observations(context.obs)),
   'query_SS15_float64':array_descriptor(representation.transform_observations(raw_query)),
   'context_obs_SS15_model':array_descriptor(obs),'query_SS15_model':array_descriptor(query)}
 return theta,obs,query,model,qualification


def qualification_receipt(path,*,raw_bindings,transformed_bindings):
 require(isinstance(CONDITIONAL_QUALIFICATION_SHA256,str) and sha256_file(path)==CONDITIONAL_QUALIFICATION_SHA256,
   'successful matching SS15 conditional qualification is not pinned')
 require(isinstance(QUALIFIER_SHA256,str),'matching conditional qualifier source is not pinned')
 qualifier=load_script('repair_ss15_conditionals.py',QUALIFIER_SHA256)
 document=json.loads(Path(path).read_text())
 return document,qualifier.validate_qualification(document,raw_bindings=raw_bindings,transformed_bindings=transformed_bindings,
    representation_descriptor=representation.representation_metadata(),expected_sources=qualifier.source_pins(),fallback_map={str(k):list(v) for k,v in FALLBACK_MAP.items()},artifact_root=Path(path).parent)


def rng_hashes(torch):
 import pickle,random
 return {'torch_CPU':hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
   'torch_CUDA':hashlib.sha256(torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest(),
   'numpy':hashlib.sha256(pickle.dumps(np.random.get_state(),protocol=5)).hexdigest(),
   'python':hashlib.sha256(pickle.dumps(random.getstate(),protocol=5)).hexdigest()}


def run(*,banks,output,checkpoint,ledger,reference,reference_grid,qualification,repetition=0):
 start=time.perf_counter();output=Path(output);require(not output.exists(),'fresh SS15 pilot output required')
 require_sources();require(type(repetition) is int and repetition>=0,'nonnegative repetition required')
 require(isinstance(CONDITIONAL_QUALIFICATION_SHA256,str) and isinstance(QUALIFIER_SHA256,str),'matching successful SS15 conditional qualification must be pinned before launch')
 from .inference_cdf import guard_baseline_source,reference_binding
 from .inference import CHECKPOINT_SHA256,frozen_regressor_settings,support_mask,_prior
 from .banks import load_context,load_heldout
 from .science.analytical_benchmark_v2 import array_descriptor,build_problem
 from .reference_grid import _require_budgeted_gpu
 frozen=clean.frozen_sampling();initial_source=source_sha256();baseline_source=guard_baseline_source()
 require(sha256_file(checkpoint)==CHECKPOINT_SHA256,'immutable checkpoint differs')
 context=load_context(banks,20261111,4096);heldout=load_heldout(banks,0);raw_query=np.asarray(heldout.obs).reshape(1,33)
 problem=build_problem();bindings={'checkpoint_sha256':CHECKPOINT_SHA256,'scientific_fingerprint':problem.scientific_fingerprint,
  'parameter_order':list(problem.parameter_order),'context_theta':array_descriptor(context.theta),
  'context_obs':array_descriptor(context.obs),'query':array_descriptor(raw_query)}
 theta,obs,query,transformed,qualification_inputs=model_bindings(context,raw_query)
 q,receipt=qualification_receipt(qualification,raw_bindings=bindings,transformed_bindings=qualification_inputs)
 refs=reference_binding(banks=banks,reference=reference,reference_grid=reference_grid,case=0)
 budget=_require_budgeted_gpu(ledger);require(budget['gpus']==1,'single GPU SS15 pilot only')
 output.mkdir(parents=True,exist_ok=False);engine=None;descriptor=None;buffer=None
 status={**runtime_identity(),'schema':SCHEMA,'variant':VARIANT,'status':'running','attribution':'Built with PriorLabs-TabPFN',
  'runner_source_sha256':initial_source,'execution_policy':execution_metadata(),'representation':representation.representation_metadata(),
  'model_inputs':transformed,'qualification_transformed_bindings':qualification_inputs,'raw_scientific_bindings':bindings,'raw33_numerical_qualification_claimed':False,
  'conditional_qualification_sha256':CONDITIONAL_QUALIFICATION_SHA256,'conditional_qualification_replay':receipt,
  'matching_FP32_execution_passed_all21':q['matching_FP32_execution_passed_all21'],
  'FP16_hard_validity_passed_all21':q['FP16_hard_validity_passed_all21'],
  'FP16_strict_failed_stages':q['FP16_strict_failed_stages'],
  'FP16_matching_uncached_strict_failed_stages':q['FP16_matching_uncached_strict_failed_stages'],
  'approximate_experiment':True,'qualified_FP32':False,'requested_count':COUNT,'benchmark_total_count':COUNT,
  'context_seed':20261111,'context_size':4096,'case':0,'repetition':repetition,'rank':0,'world_size':1,'end_to_end':True,'scientific_run':True,
  'observation_preprocessing':'source-pinned exact SS15 from original Float64 raw33; one final Float32 cast',
  'context_filter':'no_filtering','physical_parameter_order_unchanged':True,'all_three_globals_condition_local_stages':True,
  'sampler_math_changed':False,'tail_family_changed':False,'clipping_or_repair_performed':False,'partial_vector_filtering':False,
  'original_gate_unchanged':True,'absolute_accuracy_gate_required':True,'candidate_physical_assessment':'pending original reference assessment',
  'proposal_buffer_policy':frozen.proposal_buffer_policy(1200),'proposal_lookahead':1200,'proposal_buffer':None,
  'query_batch_size':1024,'engine_query_batch_cap':1024,'max_iter_rejection':None,'microbatch':COUNT,'log_density':'not_used',
  'budget_binding':budget,'reference_binding':refs,'baseline_execution_source':baseline_source,
  'completed_stage_indices':[],'stage_events':[],'proposal_wave_sizes':[],**bindings}
 write_json(output/'manifest.json',status)
 try:
  import torch
  from .compat_cdf import finite_cdf_compatibility
  from .upstream import load_baseline
  require(torch.cuda.is_available() and torch.cuda.device_count()==1,'one visible GPU required')
  require(torch.get_default_dtype()==torch.float32 and torch.get_float32_matmul_precision()=='highest'
    and not torch.backends.cuda.matmul.allow_tf32 and not torch.is_autocast_enabled('cuda'),'original strict outer flags required')
  load_baseline();accept_reject=import_module('npe_pfn.accept_reject_sampler').accept_reject_sample
  seeds=frozen.worker_stream_seeds(20261111,0,repetition,0,1)
  torch.manual_seed(seeds['component_uniform']);torch.cuda.manual_seed_all(seeds['component_uniform'])
  generators={name:torch.Generator(device='cpu').manual_seed(seed) for name,seed in seeds.items()}
  torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','16')))
  status.update(seed=seeds['component_uniform'],seed_streams=seeds,gpu=torch.cuda.get_device_name(),torch_cuda=torch.version.cuda,
    sampling_policy={'forward_precision':MODE,'streams':{n:{'role':frozen.STREAM_ROLES[n],'seed':v,'dtype':frozen.STREAM_DTYPES[n],
    'device':'cpu','generator':'torch.Generator'} for n,v in seeds.items()},'head_policy':'raw_mass_rejection_v1','mass_envelope':1.00001,
    'uniform_assignment':'preassigned_per_stage_by_physical_row','head_retry_preserves_row_uniform':True,
    'stage0_density_scope':'one generated proposal wave;8actualrawP heads; no cross-query/request reuse'})
  prior=_prior(torch);counters={'stage0_shared_predictions':0,'component_predictions':0,'head_candidates':0,'head_rejections':0};actual={}
  with finite_cdf_compatibility() as descriptor,engine_scope(checkpoint,cdf_descriptor=descriptor) as engine:
   callback=lambda prediction,stage,member:validate_prediction(prediction,stage,member,report=actual)
   stage_draw=frozen.component_stage_sampler(engine,torch,generators,counters,prediction_check=callback)
   def progress(event):status['stage_events'].append({'proposal_round':len(status['proposal_wave_sizes']),**event});write_json(output/'manifest.json',status)
   def proposal(size):
    status['proposal_wave_sizes'].append(int(size))
    values=wave.stage_major_wave(engine,theta,obs,query,int(size),1024,lambda _:None,stage_draw=stage_draw,progress=progress)
    return torch.from_numpy(values),None
   scope=frozen.proposal_buffer_scope(bindings,context_seed=20261111,case=0,repetition=repetition,rank=0,world_size=1,seeds=seeds)
   scope.update(representation_source_sha256=REPRESENTATION_SHA256,model_inputs=transformed,SS15_execution_source_sha256=SS15_EXECUTION_SHA256)
   fifo=frozen.guarded_proposal_buffer();buffer=fifo.ProposalBuffer(proposal,scope=scope,concatenate=lambda parts:torch.cat(parts,dim=0),world_size=1,rank=0)
   before_rng=rng_hashes(torch);status['default_rng_before']=before_rng
   torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();request_start=time.perf_counter()
   with torch.no_grad():samples,_,_=accept_reject(proposal=buffer,accept_reject_fn=prior.support.check,num_samples=COUNT,
      max_sampling_batch_size=COUNT,max_iter_rejection=None,show_progress_bars=False,proposal_sampling_kwargs={})
   record=buffer.finalize(original_checked_count=prior.support.proposed,accepted_before_trim=prior.support.accepted,
      returned_count=len(samples),original_support_calls=prior.support.calls)
   record.update(enabled=True,source_unchanged=True,policy_sha256=status['proposal_buffer_policy']['policy_sha256'])
   frozen.validate_proposal_buffer_record(record,world_size=1,rank=0,scope=scope);status['proposal_buffer']=record
   torch.cuda.synchronize();finished=time.perf_counter();values=samples.detach().cpu().numpy().astype(np.float64)
   require(values.shape==(COUNT,21) and support_mask(values).all(),'exactly50000 finite supported complete vectors required')
   waves=len(status['proposal_wave_sizes']);heads=8*counters['stage0_shared_predictions']+counters['component_predictions']
   require(engine.prepare_count==len(engine.stages)==21*waves,'all21 preparation/scope coverage differs')
   require(engine.actual_heads==actual['actual_head_contexts']==heads==descriptor['patched_call_count'],'physical head/CDF counters differ')
   require(engine.shared_densities==counters['stage0_shared_predictions']==waves,'fresh stage0 density count differs')
   require(record['actual_generated_wave_sizes']==status['proposal_wave_sizes'],'actual proposal/FIFO waves differ')
   status.update(execution_counters=engine.metadata(),sampling_counters=dict(counters),actual_prediction_checks=actual,
      expected_patched_call_count=heads,expected_stage_prepare_count=21*waves,generated_proposal_wave_count=waves)
  after_rng=rng_hashes(torch);status.update(default_rng_after=after_rng,default_rng_preserved=before_rng==after_rng)
  require(before_rng==after_rng,'SS15 model/sampler changed default RNG streams')
  require(descriptor['original_alias_restored'] and descriptor['sdk_sources_unchanged'],'CDF restoration failed')
  require_sources();require(source_sha256()==initial_source and guard_baseline_source()==baseline_source,'execution/science source changed')
  require(sha256_file(qualification)==CONDITIONAL_QUALIFICATION_SHA256,'matching qualification artifact changed')
  with (output/'samples.npz').open('xb') as stream:np.savez(stream,samples=values)
  status.update(status='complete',samples=array_descriptor(values),samples_file='samples.npz',samples_sha256=sha256_file(output/'samples.npz'),
    compatibility_adapter=dict(descriptor),sources_unchanged=True,completed_stage_indices=list(range(21)),
    rejection={'support_calls':prior.support.calls,'proposed':prior.support.proposed,'accepted_before_final_trim':prior.support.accepted,'returned':len(values)},
    timings={'posterior_request_seconds':finished-request_start,'total_seconds':time.perf_counter()-start,
     'accepted_samples_per_second':COUNT/(finished-request_start),'includes_stage_preparation':True,'includes_CPU_sampling':True},
    memory={'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()})
 except BaseException as error:
  status.update(status='failed',error_type=type(error).__name__,error=str(error))
  if descriptor is not None:status['compatibility_adapter']=dict(descriptor)
  if buffer is not None:status['proposal_buffer']=buffer.metadata()
  raise
 finally:
  if engine is not None:engine.close()
  status['wrapper_total_seconds']=time.perf_counter()-start;write_json(output/'manifest.json',status)
 return status


def main():
 parser=argparse.ArgumentParser(description=__doc__)
 for name in ('banks','output','checkpoint','ledger','reference','reference-grid','qualification'):parser.add_argument('--'+name,required=True)
 parser.add_argument('--repetition',type=int,default=0)
 m=run(**vars(parser.parse_args()));print(json.dumps({k:m.get(k) for k in ('schema','variant','status','requested_count','timings')},indent=2))
if __name__=='__main__':main()
