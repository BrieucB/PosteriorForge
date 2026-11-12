"""Same frozen sigma-first science under an explicit native-stage repair.

Built with PriorLabs-TabPFN. Native stages are fixed from unchanged FP32 member
checks, never posterior metrics. Original order/masks/FIFO/sampler/seeds remain.
Execution is prohibited until the FINAL same-policy all21 screen is pinned.
"""
from __future__ import annotations
import argparse,hashlib,json,os,time
from contextlib import ExitStack,contextmanager
from importlib import import_module
from pathlib import Path
import numpy as np
from . import performance_ss15_sigma_first as frozen_sigma
from . import execution_ss15_sigma_first as sigma
from . import execution_ss15_sigma_native_policy as native
from .artifacts import runtime_identity,sha256_file,write_json

SCHEMA="posteriorforge.inference.ss15_sigma_native.v1"
VARIANT="accuracy_ss15_sigma_first_native_fp16_fifo_v1"
FROZEN_SIGMA_SOURCE_SHA256="128c200a82f1252b2ad971ed43f10f534b16b9e02f8882cfd62d222cb88442cd"
ORDER_MASK_SOURCE_SHA256="d48d1638ef9e540bdd629d321fba657af4db757700b1092fc58fb9ded1f79ad2"
NATIVE_EXECUTION_SOURCE_SHA256="cf135637d073439039cea3e82a98e533cd161b095aac488ea92588aacb9a1461"
NATIVE_STAGE_INDICES=(14,) # frozen from failed06bd plus completed remaining504494 inventory
PREREGISTRATION_SHA256="ff9c853932b796005db12d2172462a0082a628d116b92ac928a2226497afff6e"
QUALIFIER_FILENAME="qualify_ss15_sigma_native.py"
QUALIFIER_SHA256="8c1b7d432888d83eeb4ce07fa8b7f18d3744ee08f00091b1204d1e33ff747334"
CONDITIONAL_QUALIFICATION_SHA256=None
FAILED_MAP2_SCREEN_SHA256="06bdcc69a189ba90ea121afef8eeb997e760a5c63221ae739859c0ac01094983"
MODE=frozen_sigma.MODE
COUNT=frozen_sigma.COUNT
REPRESENTATION_SHA256=frozen_sigma.REPRESENTATION_SHA256
SS15_EXECUTION_SHA256=frozen_sigma.SS15_EXECUTION_SHA256
clean=frozen_sigma.clean
wave=frozen_sigma.wave
representation=frozen_sigma.representation
require=frozen_sigma.require
model_bindings=frozen_sigma.model_bindings
target_order_receipt=frozen_sigma.target_order_receipt
rng_hashes=frozen_sigma.rng_hashes
_SEALS={name:(getattr(frozen_sigma,name),getattr(frozen_sigma,name).__code__)
 for name in ("model_bindings","target_order_receipt","rng_hashes","require_sources")}
FALLBACK_MAP=native.fallback_map(NATIVE_STAGE_INDICES)


def source_sha256():return sha256_file(__file__)


def require_sources():
 require(frozen_sigma.source_sha256()==FROZEN_SIGMA_SOURCE_SHA256 and sigma.source_sha256()==ORDER_MASK_SOURCE_SHA256,
  "frozen scientific sigma-first source changed")
 for name,(function,code) in _SEALS.items():
  require(getattr(frozen_sigma,name) is function and function.__code__ is code,"frozen scientific function changed: "+name)
 native.require_sources()
 require(isinstance(NATIVE_EXECUTION_SOURCE_SHA256,str) and native.source_sha256()==NATIVE_EXECUTION_SOURCE_SHA256,
  "final native execution source not frozen or changed")
 require(tuple(native.NATIVE_STAGES)==NATIVE_STAGE_INDICES and native.fallback_map(NATIVE_STAGE_INDICES)==FALLBACK_MAP,
  "declared final native policy changed")
 doc=Path(__file__).resolve().parents[2]/"docs/accuracy-ss15-sigma-execution-correction-v1.md"
 require(sha256_file(doc)==PREREGISTRATION_SHA256,"execution-only correction preregistration changed")
 frozen_sigma.require_sources()


def execution_metadata():
 require_sources()
 return {"schema":"posteriorforge.ss15_sigma_native_execution_policy.v1","module_sha256":source_sha256(),
  "frozen_sigma_first_source_sha256":FROZEN_SIGMA_SOURCE_SHA256,
  "order_mask_source_sha256":ORDER_MASK_SOURCE_SHA256,
  "native_execution_adapter_source_sha256":NATIVE_EXECUTION_SOURCE_SHA256,
  "active_native_execution_policy":native.policy_metadata(NATIVE_STAGE_INDICES),
  "frozen_scientific_sigma_policy_ancestry":frozen_sigma.execution_metadata(),
  "ancestral_precision_policy_scope":"old MAP2 is superseded; scientific order/masks remain frozen",
  "private_clone_descriptors":native.clone_descriptors(NATIVE_STAGE_INDICES),
  "preregistration_sha256":PREREGISTRATION_SHA256,
  "new_scientific_order_or_feature_change":False,"attribution":"Built with PriorLabs-TabPFN"}


@contextmanager
def engine_scope(checkpoint,*,cdf_descriptor):
 with ExitStack() as cleanup:
  engine=native.adapter_class(NATIVE_STAGE_INDICES)(checkpoint,cdf_descriptor=cdf_descriptor)
  cleanup.callback(engine.close)
  yield engine


def validate_prediction(prediction,stage,member,*,report):
 native.validate_prediction(prediction,stage,member,report=report,native_stages=NATIVE_STAGE_INDICES)


def qualification_receipt(path,*,raw_bindings,transformed_bindings):
 require(isinstance(CONDITIONAL_QUALIFICATION_SHA256,str) and len(CONDITIONAL_QUALIFICATION_SHA256)==64
  and sha256_file(path)==CONDITIONAL_QUALIFICATION_SHA256,"final successful same-native-policy all21 screen must be pinned")
 require(isinstance(QUALIFIER_SHA256,str) and len(QUALIFIER_SHA256)==64,"native-screen source not pinned")
 helper=frozen_sigma.pilot.load_script(QUALIFIER_FILENAME,QUALIFIER_SHA256)
 document=json.loads(Path(path).read_bytes())
 receipt=helper.validate_screen(document,expected_sources=helper.source_pins(),raw_bindings=raw_bindings,
  transformed_bindings=transformed_bindings,order_policy=sigma.policy_metadata(),
  native_policy=native.policy_metadata(NATIVE_STAGE_INDICES))
 return document,receipt

def run(*,banks,output,checkpoint,ledger,reference,reference_grid,qualification,repetition=0):
 start=time.perf_counter();output=Path(output);require(not output.exists(),'fresh SS15 pilot output required')
 require_sources();require(type(repetition) is int and repetition==0,'fixed sigma-first first request uses repetition0')
 require(isinstance(CONDITIONAL_QUALIFICATION_SHA256,str) and len(CONDITIONAL_QUALIFICATION_SHA256)==64
  and isinstance(QUALIFIER_SHA256,str) and len(QUALIFIER_SHA256)==64,'final same-native-policy all21 proof must be pinned before any inference')
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
  'matching_conditional_qualification_sha256':CONDITIONAL_QUALIFICATION_SHA256,
  'matched_execution_screen':{'qualification_sha256':CONDITIONAL_QUALIFICATION_SHA256,'replay':receipt,
    'raw_scientific_bindings':bindings,'model_inputs':transformed,
    'matching_FP32_execution_passed_all21':q['matching_FP32_execution_passed_all21'],
    'FP16_hard_validity_passed_all21':q['FP16_hard_validity_passed_all21'],
    'FP16_strict_failed_stages':q['FP16_strict_failed_stages'],
    'scope':'same-case final-policy finite fixed-prefix screen only; no posterior-joint equivalence'},
  'active_native_execution_policy':native.policy_metadata(NATIVE_STAGE_INDICES),
  'native_execution_adapter_source_sha256':NATIVE_EXECUTION_SOURCE_SHA256,
  'frozen_sigma_first_source_sha256':FROZEN_SIGMA_SOURCE_SHA256,'order_mask_source_sha256':ORDER_MASK_SOURCE_SHA256,
  'ancestral_order_mask_policy':sigma.policy_metadata(),
  'ancestral_precision_policy_scope':'old d48 MAP2 fallback metadata superseded by active native policy',
  'ancestral_failed_MAP2_screen_sha256':FAILED_MAP2_SCREEN_SHA256,
  'ancestral_failed_MAP2_status':'sigma_first_matching_FP32_structural_or_hard_failure; not promoted',
  'preregistration_sha256':PREREGISTRATION_SHA256,
  'model_inputs_scope':'explicit stage-ORDERED theta and SS15 source; actual masked stage inputs in execution counters',
  'actual_sigma_first_fixed_conditional_screen_performed':False,
  'fresh_same_policy_all21_screen_replayed':True,
  'fresh_screen_scope':'performed separately; same case/policy finite prefixes; this request only CPU-replays its artifact',
  'approximate_experiment':True,'qualified_FP32':False,'requested_count':COUNT,'benchmark_total_count':COUNT,
  'context_seed':20261111,'context_size':4096,'case':0,'repetition':repetition,'rank':0,'world_size':1,'end_to_end':True,'scientific_run':True,
  'observation_preprocessing':'source-pinned SS15 Float32, then fixed sigma-first zero mask per local stage',
  'context_filter':'no_filtering','physical_parameter_order_unchanged':True,'all_three_globals_condition_local_stages':True,
  'learned_target_order_changed':True,'model_target_order':'sigma_first_stage_order',
  'physical_parameter_order_scope':'raw scientific binding and saved/restored output; learned conditional targets use new order',
  'stage_to_physical_order':list(sigma.ORDER),'physical_to_stage_order':list(sigma.INVERSE_ORDER),
  'samples_restored_to_physical_order_before_FIFO_support':True,
  'samples_in_original_physical_order_before_FIFO_support':True,
  'fallback_stage_member_bindings':native.policy_metadata(NATIVE_STAGE_INDICES)['active_fallback_stage_member_bindings'],
  'model_target_descriptors':target_order_receipt(theta),
  'actual_physical_model_inputs':{**transformed,'context_theta':target_order_receipt(theta)['physical_theta_float32']},
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
   scope.update(representation_source_sha256=REPRESENTATION_SHA256,model_inputs=transformed,SS15_execution_source_sha256=SS15_EXECUTION_SHA256,sigma_first_module_sha256=ORDER_MASK_SOURCE_SHA256,native_execution_adapter_source_sha256=NATIVE_EXECUTION_SOURCE_SHA256,native_execution_policy_sha256=native.policy_metadata(NATIVE_STAGE_INDICES)['policy_sha256'],sigma_first_policy_sha256=sigma.policy_metadata()['policy_sha256'],stage_to_physical_order=list(sigma.ORDER),physical_to_stage_order=list(sigma.INVERSE_ORDER),samples_restored_before_FIFO_support=True)
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
