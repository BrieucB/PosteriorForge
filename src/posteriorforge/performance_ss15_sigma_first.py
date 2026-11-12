"""One fixed all6-block sigma-first SS15 request, full50000 complete vectors.

Built with PriorLabs-TabPFN. Frozen4099 model execution/sampling is reused;
The preregistered intervention combines the fixed block mask and sigma-first
conditional order; complete vectors return to physical order before filtering.
The earlier unmasked finite conditional screen is an anchor, not a masked proof.
"""
from __future__ import annotations
import argparse, hashlib, json, os, time
from contextlib import ExitStack, contextmanager
from importlib import import_module
from pathlib import Path
import numpy as np
from . import performance_ss15 as pilot
from . import execution_ss15_sigma_first as sigma
from .artifacts import runtime_identity, sha256_file, write_json

SCHEMA="posteriorforge.inference.ss15_sigma_first.v1"
VARIANT="accuracy_ss15_sigma_first_fp16_fifo_v1"
PILOT_SHA256="4099da3c19366997ab0800e5430b5bb1c129b4b3e535f678b3d4652233303480"
SIGMA_FIRST_SOURCE_SHA256="d48d1638ef9e540bdd629d321fba657af4db757700b1092fc58fb9ded1f79ad2"
PREREGISTRATION_SHA256="5b7a31237c9a3e35db27f3f9697c7a9293b0738a8ec8b12d63f370d4ad27e3d4"
MODE=pilot.MODE
COUNT=pilot.COUNT
FALLBACK_MAP=pilot.FALLBACK_MAP
QUALIFIER_SHA256=pilot.QUALIFIER_SHA256
CONDITIONAL_QUALIFICATION_SHA256=pilot.CONDITIONAL_QUALIFICATION_SHA256
REPRESENTATION_SHA256=pilot.REPRESENTATION_SHA256
SS15_EXECUTION_SHA256=pilot.SS15_EXECUTION_SHA256
clean=pilot.clean
wave=pilot.wave
representation=pilot.representation
require=pilot.require
qualification_receipt=pilot.qualification_receipt
rng_hashes=pilot.rng_hashes
_FROZEN_NAMES=("run","model_bindings","qualification_receipt","rng_hashes","validate_prediction")
_SEALS={n:(getattr(pilot,n),getattr(pilot,n).__code__) for n in _FROZEN_NAMES}


def source_sha256():return sha256_file(__file__)


def require_sources():
 require(pilot.source_sha256()==PILOT_SHA256,"frozen4099 producer changed")
 for name,(function,code) in _SEALS.items():
  require(getattr(pilot,name) is function and function.__code__ is code,"frozen producer function changed: "+name)
 sigma.require_sources()
 require(sigma.source_sha256()==SIGMA_FIRST_SOURCE_SHA256,"sigma-first module not frozen or changed")
 doc=Path(__file__).resolve().parents[2]/"docs/accuracy-ss15-sigma-first-v1.md"
 require(sha256_file(doc)==PREREGISTRATION_SHA256,"pre-cohort sigma-first contract changed")


def execution_metadata():
 require_sources()
 return {"schema":"posteriorforge.ss15_sigma_first_execution_policy.v1",
   "module_sha256":source_sha256(),"frozen_SS15_source_sha256":PILOT_SHA256,
   "backend_policy":pilot.execution_metadata(),"sigma_first_input_policy":sigma.policy_metadata(),
   "preregistration_sha256":PREREGISTRATION_SHA256,
   "sampler_wave_FIFO_probability_math_unchanged":True,
   "actual_sigma_first_fixed_conditional_screen_performed":False,
   "old_unmasked_anchor_qualifies_ordered_inputs":False,"approximate_FP16_forward":True,
   "attribution":"Built with PriorLabs-TabPFN"}


@contextmanager
def engine_scope(checkpoint,*,cdf_descriptor):
 with ExitStack() as cleanup:
  engine=sigma.SigmaFirstStageAdapter(checkpoint,cdf_descriptor=cdf_descriptor)
  cleanup.callback(engine.close)
  yield engine


def validate_prediction(prediction,stage,member,*,report):
 pilot.validate_prediction(prediction,stage,member,report=report)
 proof=prediction.get("sigma_first_input_proof",{})
 require(proof.get("module_sha256")==SIGMA_FIRST_SOURCE_SHA256
    and proof.get("policy_sha256")==sigma.policy_metadata()["policy_sha256"]
    and proof.get("stage")==stage and proof.get("member")==member
    and proof.get("mask")==sigma.stage_policy(stage)
    and proof.get("training_query_same_fixed_mask") is True
    and proof.get("ordered_target_binding_verified") is True
    and proof.get("target_physical_index")==sigma.ORDER[stage],"actual fixed sigma-first input mask proof differs")
 count=8 if member is None else 1
 require(proof.get("physical_head_contexts")==count,"actual sigma-first head count differs")
 if stage<3:require(proof["unmasked_query"]==proof["actual_masked_query"],"global factor input changed")
 report["sigma_first_physical_head_contexts"]=report.get("sigma_first_physical_head_contexts",0)+count



def model_bindings(context,raw_query):
 from .science.analytical_benchmark_v2 import array_descriptor
 physical,obs,query,model,qualification=pilot.model_bindings(context,raw_query)
 ordered=sigma.order_theta(physical)
 require(np.array_equal(ordered,physical[:,sigma.ORDER]),"ordered target bank differs from frozen permutation")
 for stage in range(3):
  require(ordered[:,stage].tobytes()==physical[:,stage].tobytes()
    and np.concatenate((obs,ordered[:,:stage]),axis=1).tobytes()
       ==np.concatenate((obs,physical[:,:stage]),axis=1).tobytes(),
    "global used predictors or target labels changed")
 return ordered,obs,query,{**model,"context_theta":array_descriptor(ordered)},qualification


def target_order_receipt(ordered):
 from .science.analytical_benchmark_v2 import array_descriptor
 physical=sigma.restore_physical(ordered)
 require(np.array_equal(ordered,physical[:,sigma.ORDER]),"target ordering receipt failed")
 return {"physical_theta_float32":array_descriptor(physical),
   "ordered_theta_float32":array_descriptor(ordered),"stage_to_physical_order":list(sigma.ORDER),
   "ordered_theta_equals_physical_theta_columns_pi":True,
   "global_used_predictors_labels_byte_identical":True,
   "target_coordinate_system":"sigma_first_stage_order"}

def run(*,banks,output,checkpoint,ledger,reference,reference_grid,qualification,repetition=0):
 start=time.perf_counter();output=Path(output);require(not output.exists(),'fresh SS15 pilot output required')
 require_sources();require(type(repetition) is int and repetition==0,'fixed sigma-first first request uses repetition0')
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
  'conditional_anchor':{'qualification_sha256':CONDITIONAL_QUALIFICATION_SHA256,'replay':receipt,
    'raw_scientific_bindings':bindings,'unmasked_model_inputs':{**transformed,'context_theta':array_descriptor(sigma.restore_physical(theta))},
    'matching_FP32_execution_passed_all21':q['matching_FP32_execution_passed_all21'],
    'FP16_hard_validity_passed_all21':q['FP16_hard_validity_passed_all21'],
    'FP16_strict_failed_stages':q['FP16_strict_failed_stages'],
    'FP16_matching_uncached_strict_failed_stages':q['FP16_matching_uncached_strict_failed_stages'],
    'scope':'earlier unmasked original-order SS15 finite anchor only; no ordered-input qualification','qualifies_sigma_first_inputs':False},
  'sigma_first_input_policy':sigma.policy_metadata(),'preregistration_sha256':PREREGISTRATION_SHA256,
  'model_inputs_scope':'explicit stage-ORDERED theta and SS15 source; actual masked stage-specific predictors in execution_counters.sigma_first_stage_inputs',
  'actual_sigma_first_fixed_conditional_screen_performed':False,
  'sigma_first_numerical_qualification_status':'pending separate masked matching screen; no inherited pass',
  'approximate_experiment':True,'qualified_FP32':False,'requested_count':COUNT,'benchmark_total_count':COUNT,
  'context_seed':20261111,'context_size':4096,'case':0,'repetition':repetition,'rank':0,'world_size':1,'end_to_end':True,'scientific_run':True,
  'observation_preprocessing':'source-pinned SS15 Float32, then fixed sigma-first zero mask per local stage',
  'context_filter':'no_filtering','physical_parameter_order_unchanged':True,'all_three_globals_condition_local_stages':True,
  'learned_target_order_changed':True,'model_target_order':'sigma_first_stage_order',
  'physical_parameter_order_scope':'raw scientific binding and saved/restored output; learned conditional targets use new order',
  'stage_to_physical_order':list(sigma.ORDER),'physical_to_stage_order':list(sigma.INVERSE_ORDER),
  'samples_restored_to_physical_order_before_FIFO_support':True,
  'fallback_stage_member_bindings':sigma.fallback_bindings(),
  'model_target_descriptors':target_order_receipt(theta),
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
    ordered_values=wave.stage_major_wave(engine,theta,obs,query,int(size),1024,lambda _:None,stage_draw=stage_draw,progress=progress)
    values=sigma.restore_physical(ordered_values)
    return torch.from_numpy(values),None
   scope=frozen.proposal_buffer_scope(bindings,context_seed=20261111,case=0,repetition=repetition,rank=0,world_size=1,seeds=seeds)
   scope.update(representation_source_sha256=REPRESENTATION_SHA256,model_inputs=transformed,SS15_execution_source_sha256=SS15_EXECUTION_SHA256,sigma_first_module_sha256=SIGMA_FIRST_SOURCE_SHA256,sigma_first_policy_sha256=sigma.policy_metadata()['policy_sha256'],stage_to_physical_order=list(sigma.ORDER),physical_to_stage_order=list(sigma.INVERSE_ORDER),samples_restored_before_FIFO_support=True)
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
   require(len(engine.sigma_first_stage_inputs)==21*waves and len(engine.sigma_first_query_calls)==heads
      and actual.get('sigma_first_physical_head_contexts')==heads,'actual masked stage/head coverage differs')
   require([r['stage'] for r in engine.sigma_first_stage_inputs]==list(range(21))*waves,
      'all21 fixed masks required for every complete-vector wave')
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
 for name in ("banks","output","checkpoint","ledger","reference","reference-grid","qualification"):
  parser.add_argument("--"+name,required=True)
 m=run(**vars(parser.parse_args()))
 print(json.dumps({k:m.get(k) for k in ("schema","variant","status","requested_count","timings")},indent=2))
if __name__=="__main__":main()
