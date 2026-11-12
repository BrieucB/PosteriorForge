"""Frozen graph-local and sigma-first SS15 scientific diagnostic requests.

Built with PriorLabs-TabPFN. Exactly50000 complete physical vectors perrequest,
one GPU, fixed4096-world contexts. Actual inputs/references are case-specific;
the earlier case0 numerical screen remains a separate finite-prefix anchor.
"""
from __future__ import annotations
import argparse,copy,hashlib,importlib.util,json,os,time
from importlib import import_module
from pathlib import Path
import numpy as np
from . import performance_ss15_graph as graph_base,performance_ss15_sigma_native as sigma_base
from .artifacts import runtime_identity,sha256_file,write_json
from .banks import load_context,load_heldout
from .inference import CHECKPOINT_SHA256
from .science.analytical_benchmark_v2 import array_descriptor,build_problem

SCHEMA="posteriorforge.inference.ss15_scientific_campaign.v1"
COUNT=50000
CONTEXT_SIZE=4096
CONTEXT_SEEDS=(20261111,20261112,20261113)
CASES=tuple(range(8))
ANCHOR=(20261111,0)
PENDING_GRID=tuple((s,c) for s in CONTEXT_SEEDS for c in CASES if (s,c)!=ANCHOR)
METHODS={
 "graph_local":{"variant":"accuracy_ss15_graph_local_fp16_fifo_scientific_campaign_v1",
   "frozen_source":"c2f20a8dbf1f4e2bf48294be038962d0c3e4e00ec8b4a7cab29ea259d91d1818",
   "mask_source":"dfa096c314fdc3855d0a45a4bcb3f9858675747edc2403f917e32e19837eb110",
   "qualifier":"qualify_ss15_graph_conditionals.py",
   "qualifier_source":"3f5ee07947418bdc6877500e06f56c5af3dfa9b31d6a6208313aa25eb05c38b2",
   "qualification_sha256":"d90604c99664e6c977ddf59c0f72adbb44fc520594ab36d64d6f25841a3985e3",
   "stage_inputs_attribute":"graph_stage_inputs","query_calls_attribute":"graph_query_calls"},
 "sigma_first":{"variant":"accuracy_ss15_sigma_first_native_fp16_fifo_scientific_campaign_v1",
   "frozen_source":"e322d56492476be1acf5fcdb9afd30b57e2a69a373098417a2bf8944b51ba73b",
   "mask_source":"d48d1638ef9e540bdd629d321fba657af4db757700b1092fc58fb9ded1f79ad2",
   "qualifier":"qualify_ss15_sigma_native.py",
   "qualifier_source":"8c1b7d432888d83eeb4ce07fa8b7f18d3744ee08f00091b1204d1e33ff747334",
   "qualification_sha256":"bedfa67dc8a1e7f41a8536e4c8091643dc05d309b0da0367dc92136c19603705",
   "native_execution_source":"cf135637d073439039cea3e82a98e533cd161b095aac488ea92588aacb9a1461",
   "stage_inputs_attribute":"sigma_first_stage_inputs","query_calls_attribute":"sigma_first_query_calls"}}
_BASES={"graph_local":graph_base,"sigma_first":sigma_base}
_SEALS={method:{name:(getattr(base,name),getattr(base,name).__code__)
               for name in ("model_bindings","engine_scope","validate_prediction","execution_metadata","require_sources","rng_hashes")}
        for method,base in _BASES.items()}


def require(value,message):
 if not value:raise ValueError(message)


def source_sha256():return sha256_file(__file__)


def method_modules(method):
 require(method in METHODS,"explicit frozen method graph_local|sigma_first required")
 base=_BASES[method];mask=base.graph if method=="graph_local" else base.sigma
 require(base.source_sha256()==METHODS[method]["frozen_source"]
  and mask.source_sha256()==METHODS[method]["mask_source"],"frozen scientific method source changed")
 for name,(function,code) in _SEALS[method].items():
  require(getattr(base,name) is function and function.__code__ is code,"frozen method function changed: "+name)
 base.require_sources()
 if method=="sigma_first":
  require(base.native.source_sha256()==METHODS[method]["native_execution_source"]
   and tuple(base.NATIVE_STAGE_INDICES)==(14,),"final sigma native execution policy/source changed")
 return base,mask


def load_qualifier(method):
 spec=METHODS[method];path=Path(__file__).resolve().parents[2]/"scripts"/spec["qualifier"]
 require(sha256_file(path)==spec["qualifier_source"],"fixed numerical-screen source changed")
 module_spec=importlib.util.spec_from_file_location("_scientific_campaign_"+method+"_qualifier",path)
 module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
 return module


def require_sources(method):
 base,mask=method_modules(method)
 digest=METHODS[method]["qualification_sha256"]
 require(isinstance(digest,str) and len(digest)==64,"successful fixed case0 screen must be pinned before any request")
 load_qualifier(method).source_pins()
 return base,mask


def request_spec(*,method,context_seed,case,context_size=CONTEXT_SIZE,count=COUNT,repetition=0):
 require(method in METHODS,"explicit frozen method required")
 require(type(context_seed) is int and type(case) is int and (context_seed,case) in PENDING_GRID,
  "one of the remaining23 predefined points required; completed seed11/case0 is reused")
 require(type(context_size) is int and context_size==4096,"exact4096 context required")
 require(type(count) is int and count==50000,"exact50000 complete accepted vectors required")
 require(type(repetition) is int and repetition==0,"fixed repetition0 only")
 return {"method":method,"context_seed":context_seed,"case":case,"context_size":context_size,
  "requested_count":count,"repetition":repetition}


def raw_bindings(context,query):
 problem=build_problem()
 return {"checkpoint_sha256":CHECKPOINT_SHA256,"scientific_fingerprint":problem.scientific_fingerprint,
  "parameter_order":list(problem.parameter_order),"context_theta":array_descriptor(context.theta),
  "context_obs":array_descriptor(context.obs),"query":array_descriptor(query)}


def order_metadata(method):
 _,mask=method_modules(method)
 order=list(range(21)) if method=="graph_local" else list(mask.ORDER)
 inverse=list(range(21)) if method=="graph_local" else list(mask.INVERSE_ORDER)
 return {"stage_to_physical_order":order,"physical_to_stage_order":inverse,
  "model_target_order":"original_physical_order" if method=="graph_local" else "sigma_first_stage_order",
  "learned_target_order_changed":method=="sigma_first",
  "physical_output_mapping":"identity" if method=="graph_local" else "inverse_stage_permutation",
  "samples_in_original_physical_order_before_FIFO_support":True,
  **({"samples_restored_to_physical_order_before_FIFO_support":True} if method=="sigma_first" else {})}


def input_bundle(banks,method,context_seed,case):
 base,mask=method_modules(method)
 context=load_context(banks,context_seed,4096)
 raw_query=np.asarray(load_heldout(banks,case).obs).reshape(1,33)
 theta,obs,query,model,transformed=base.model_bindings(context,raw_query)
 physical=context.theta.astype(np.float32)
 order=order_metadata(method)["stage_to_physical_order"]
 require(theta.tobytes()==physical[:,order].copy().tobytes(),"actual target-bank permutation differs")
 for j in range(3):
  require(theta[:,j].tobytes()==physical[:,j].tobytes()
   and np.concatenate((obs,theta[:,:j]),axis=1).tobytes()
     ==np.concatenate((obs,physical[:,:j]),axis=1).tobytes(),"global used predictors/labels changed")
 physical_model={**model,"context_theta":array_descriptor(physical)}
 target={"physical_theta_float32":array_descriptor(physical),"ordered_theta_float32":array_descriptor(theta),
  "stage_to_physical_order":order,"ordered_theta_equals_physical_theta_columns_pi":True,
  "global_used_predictors_labels_byte_identical":True,
  "target_coordinate_system":order_metadata(method)["model_target_order"]}
 if method=="sigma_first":
  require(target==base.target_order_receipt(theta),"actual sigma-first target receipt differs")
 return {"context":context,"raw_query":raw_query,"theta":theta,"obs":obs,"query":query,
  "raw_bindings":raw_bindings(context,raw_query),"model_inputs":model,
  "actual_physical_model_inputs":physical_model,"transformed_bindings":transformed,
  "model_target_descriptors":target}


def conditional_anchor(banks,qualification,method):
 require_sources(method)
 anchor=input_bundle(banks,method,*ANCHOR)
 path=Path(qualification);raw=path.read_bytes();expected=METHODS[method]["qualification_sha256"]
 require(hashlib.sha256(raw).hexdigest()==expected,"exact passed case0 numerical-screen artifact required")
 document=json.loads(raw);helper=load_qualifier(method)
 base,mask=method_modules(method)
 arguments={"expected_sources":helper.source_pins(),"raw_bindings":anchor["raw_bindings"],
  "transformed_bindings":anchor["transformed_bindings"],
  "graph_policy" if method=="graph_local" else "order_policy":mask.policy_metadata()}
 if method=="sigma_first":arguments["native_policy"]=base.native.policy_metadata(base.NATIVE_STAGE_INDICES)
 receipt=helper.validate_screen(document,**arguments)
 return {"context_seed":20261111,"case":0,"context_size":4096,"method":method,
  "qualification_sha256":expected,"qualifier_source_sha256":METHODS[method]["qualifier_source"],
  "raw_scientific_bindings":anchor["raw_bindings"],"model_inputs":anchor["model_inputs"],
  "physical_model_inputs":anchor["actual_physical_model_inputs"],
  "transformed_bindings":anchor["transformed_bindings"],
  "model_target_descriptors":anchor["model_target_descriptors"],"replay":receipt,
  "matching_FP32_execution_passed_all21":document["matching_FP32_execution_passed_all21"],
  "FP16_hard_validity_passed_all21":document["FP16_hard_validity_passed_all21"],
  "FP16_strict_failed_stages":document["FP16_strict_failed_stages"],
  **native_execution_fields(method),
  "qualifies_actual_request":False,"scope":"finite fixed-prefix case0 seed11 only; not a screen of another case/context"}


def numerical_scope(actual,anchor):
 return {"actual_request_fixed_conditional_screen_performed":False,
  "anchor_qualifies_actual_request":False,
  "same_context_as_anchor":all(actual["raw_bindings"][k]==anchor["raw_scientific_bindings"][k]
    for k in ("context_theta","context_obs")),
  "same_query_as_anchor":actual["raw_bindings"]["query"]==anchor["raw_scientific_bindings"]["query"],
  "scope":"actual source/dtype/mass/mask/order/RNG guards; distinct finite case0 anchor",
  "posterior_accuracy_claimed":False}



def native_execution_fields(method):
 if method!="sigma_first":return {}
 base,_=method_modules(method)
 return {"active_native_execution_policy":base.native.policy_metadata(base.NATIVE_STAGE_INDICES),
  "native_execution_adapter_source_sha256":base.NATIVE_EXECUTION_SOURCE_SHA256,
  "frozen_sigma_first_source_sha256":base.FROZEN_SIGMA_SOURCE_SHA256,
  "order_mask_source_sha256":base.ORDER_MASK_SOURCE_SHA256}

def execution_metadata(method):
 base,mask=require_sources(method)
 return {"schema":"posteriorforge.ss15_scientific_campaign_policy.v1","module_sha256":source_sha256(),
  "method":method,"frozen_method_source_sha256":METHODS[method]["frozen_source"],
  "masked_adapter_source_sha256":METHODS[method]["mask_source"],"frozen_method_policy":base.execution_metadata(),
  "context_seeds":list(CONTEXT_SEEDS),"cases":list(CASES),"predefined_points_per_method":24,
  "remaining_new_points_per_method":23,"existing_anchor_point":{"context_seed":20261111,"case":0},
  "count_per_request":50000,"context_size":4096,"repetition":0,"one_GPU_per_request":True,
  "new_case_numerical_qualification_claimed":False,"attribution":"Built with PriorLabs-TabPFN",
  **order_metadata(method),**native_execution_fields(method)}


def physical_proposals(method,values):
 if method=="sigma_first":return method_modules(method)[1].restore_physical(values)
 require(method=="graph_local" and isinstance(values,np.ndarray) and values.ndim==2 and values.shape[1]==21,
  "complete original-physical-order graph proposals required")
 return values


def proposal_scope(frozen,bindings,model_inputs,*,method,context_seed,case,repetition,seeds):
 base,mask=method_modules(method)
 scope=frozen.proposal_buffer_scope(bindings,context_seed=context_seed,case=case,repetition=repetition,rank=0,world_size=1,seeds=seeds)
 scope.update(representation_source_sha256=base.REPRESENTATION_SHA256,model_inputs=model_inputs,
  SS15_execution_source_sha256=base.SS15_EXECUTION_SHA256,scientific_campaign_method=method,
  frozen_method_source_sha256=METHODS[method]["frozen_source"],masked_adapter_source_sha256=METHODS[method]["mask_source"],
  mask_policy_sha256=mask.policy_metadata()["policy_sha256"],
  stage_to_physical_order=order_metadata(method)["stage_to_physical_order"],
  physical_to_stage_order=order_metadata(method)["physical_to_stage_order"],
  samples_in_original_physical_order_before_FIFO_support=True)
 if method=="sigma_first":
  scope.update(samples_restored_before_FIFO_support=True,
   native_execution_adapter_source_sha256=base.NATIVE_EXECUTION_SOURCE_SHA256,
   native_execution_policy_sha256=base.native.policy_metadata(base.NATIVE_STAGE_INDICES)["policy_sha256"])
 return scope

def run(*,banks,output,checkpoint,ledger,reference,reference_grid,qualification,method,context_seed,case,context_size=CONTEXT_SIZE,count=COUNT,repetition=0):
 start=time.perf_counter();output=Path(output);require(not output.exists(),'fresh SS15 campaign output required')
 request_spec(method=method,context_seed=context_seed,case=case,context_size=context_size,count=count,repetition=repetition)
 base,mask=require_sources(method);method_spec=METHODS[method]
 clean=base.clean;wave=base.wave;representation=base.representation;MODE=base.MODE
 CONDITIONAL_QUALIFICATION_SHA256=method_spec['qualification_sha256']
 require(isinstance(CONDITIONAL_QUALIFICATION_SHA256,str),'successful fixed conditional anchor must be pinned before launch')
 from .inference_cdf import guard_baseline_source,reference_binding
 from .inference import CHECKPOINT_SHA256,frozen_regressor_settings,support_mask,_prior
 from .banks import load_context,load_heldout
 from .science.analytical_benchmark_v2 import array_descriptor,build_problem
 from .reference_grid import _require_budgeted_gpu
 frozen=clean.frozen_sampling();initial_source=source_sha256();baseline_source=guard_baseline_source()
 require(sha256_file(checkpoint)==CHECKPOINT_SHA256,'immutable checkpoint differs')
 actual_inputs=input_bundle(banks,method,context_seed,case)
 context=actual_inputs['context'];raw_query=actual_inputs['raw_query'];bindings=actual_inputs['raw_bindings']
 theta=actual_inputs['theta'];obs=actual_inputs['obs'];query=actual_inputs['query']
 transformed=actual_inputs['model_inputs'];qualification_inputs=actual_inputs['transformed_bindings']
 anchor=conditional_anchor(banks,qualification,method);scope_claim=numerical_scope(actual_inputs,anchor)
 refs=reference_binding(banks=banks,reference=reference,reference_grid=reference_grid,case=case)
 budget=_require_budgeted_gpu(ledger);require(budget['gpus']==1,'single GPU scientific SS15 campaign request only')
 output.mkdir(parents=True,exist_ok=False);engine=None;descriptor=None;buffer=None
 status={**runtime_identity(),'schema':SCHEMA,'variant':method_spec['variant'],'method':method,'status':'running','attribution':'Built with PriorLabs-TabPFN',
  'runner_source_sha256':initial_source,'execution_policy':execution_metadata(method),'representation':representation.representation_metadata(),'mask_policy':mask.policy_metadata(),
  'model_inputs':transformed,'qualification_transformed_bindings':qualification_inputs,'raw_scientific_bindings':bindings,'raw33_numerical_qualification_claimed':False,
  'conditional_qualification_sha256':CONDITIONAL_QUALIFICATION_SHA256,
  'conditional_anchor':anchor,'actual_request_numerical_scope':scope_claim,
  'frozen_method_source_sha256':method_spec['frozen_source'],'masked_adapter_source_sha256':method_spec['mask_source'],
  'actual_physical_model_inputs':actual_inputs['actual_physical_model_inputs'],
  'model_target_descriptors':actual_inputs['model_target_descriptors'],**order_metadata(method),**native_execution_fields(method),'run_purpose':'predefined24_scientific_diagnostic_grid',
  'actual_request_fixed_conditional_screen_performed':False,
  'approximate_experiment':True,'qualified_FP32':False,'requested_count':COUNT,'benchmark_total_count':COUNT,
  'context_seed':context_seed,'context_size':CONTEXT_SIZE,'case':case,'repetition':repetition,'rank':0,'world_size':1,'end_to_end':True,'scientific_run':True,
  'observation_preprocessing':'frozen SS15 Float32 plus fixed method-specific graph mask and target order',
  'context_filter':'no_filtering','physical_parameter_order_unchanged':True,'all_three_globals_condition_local_stages':True,
  'physical_parameter_order_scope':'raw science and saved output; model target order is separately explicit',
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
  seeds=frozen.worker_stream_seeds(context_seed,case,repetition,0,1)
  torch.manual_seed(seeds['component_uniform']);torch.cuda.manual_seed_all(seeds['component_uniform'])
  generators={name:torch.Generator(device='cpu').manual_seed(seed) for name,seed in seeds.items()}
  torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','16')))
  status.update(seed=seeds['component_uniform'],seed_streams=seeds,gpu=torch.cuda.get_device_name(),torch_cuda=torch.version.cuda,
    sampling_policy={'forward_precision':MODE,'streams':{n:{'role':frozen.STREAM_ROLES[n],'seed':v,'dtype':frozen.STREAM_DTYPES[n],
    'device':'cpu','generator':'torch.Generator'} for n,v in seeds.items()},'head_policy':'raw_mass_rejection_v1','mass_envelope':1.00001,
    'uniform_assignment':'preassigned_per_stage_by_physical_row','head_retry_preserves_row_uniform':True,
    'stage0_density_scope':'one generated proposal wave;8actualrawP heads; no cross-query/request reuse'})
  prior=_prior(torch);counters={'stage0_shared_predictions':0,'component_predictions':0,'head_candidates':0,'head_rejections':0};actual={}
  with finite_cdf_compatibility() as descriptor,base.engine_scope(checkpoint,cdf_descriptor=descriptor) as engine:
   callback=lambda prediction,stage,member:base.validate_prediction(prediction,stage,member,report=actual)
   stage_draw=frozen.component_stage_sampler(engine,torch,generators,counters,prediction_check=callback)
   def progress(event):status['stage_events'].append({'proposal_round':len(status['proposal_wave_sizes']),**event});write_json(output/'manifest.json',status)
   def proposal(size):
    status['proposal_wave_sizes'].append(int(size))
    stage_values=wave.stage_major_wave(engine,theta,obs,query,int(size),1024,lambda _:None,stage_draw=stage_draw,progress=progress)
    values=physical_proposals(method,stage_values)
    return torch.from_numpy(values),None
   scope=proposal_scope(frozen,bindings,transformed,method=method,context_seed=context_seed,case=case,repetition=repetition,seeds=seeds)
   fifo=frozen.guarded_proposal_buffer();buffer=fifo.ProposalBuffer(proposal,scope=scope,concatenate=lambda parts:torch.cat(parts,dim=0),world_size=1,rank=0)
   before_rng=base.rng_hashes(torch);status['default_rng_before']=before_rng
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
   stage_inputs=getattr(engine,method_spec['stage_inputs_attribute']);query_calls=getattr(engine,method_spec['query_calls_attribute'])
   require(len(stage_inputs)==21*waves and len(query_calls)==heads
      and actual.get(method+'_physical_head_contexts')==heads,'actual masked ordered stage/head coverage differs')
   require([r['stage'] for r in stage_inputs]==list(range(21))*waves,'all21 fixed masks required each wave')
   status.update(execution_counters=engine.metadata(),sampling_counters=dict(counters),actual_prediction_checks=actual,
      expected_patched_call_count=heads,expected_stage_prepare_count=21*waves,generated_proposal_wave_count=waves)
  after_rng=base.rng_hashes(torch);status.update(default_rng_after=after_rng,default_rng_preserved=before_rng==after_rng)
  require(before_rng==after_rng,'SS15 model/sampler changed default RNG streams')
  require(descriptor['original_alias_restored'] and descriptor['sdk_sources_unchanged'],'CDF restoration failed')
  require_sources(method);require(source_sha256()==initial_source and guard_baseline_source()==baseline_source,'execution/science source changed')
  require(sha256_file(qualification)==CONDITIONAL_QUALIFICATION_SHA256,'fixed anchor artifact changed')
  require(raw_bindings(context,raw_query)==bindings,'actual physical bank/query input changed')
  require({'context_theta':array_descriptor(theta),'context_obs':array_descriptor(obs),'query':array_descriptor(query)}==transformed,
    'actual transformed training/query input changed')
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
 parser.add_argument("--method",choices=tuple(METHODS),required=True)
 parser.add_argument("--context-seed",type=int,choices=CONTEXT_SEEDS,required=True)
 parser.add_argument("--case",type=int,choices=CASES,required=True)
 m=run(**vars(parser.parse_args()))
 print(json.dumps({k:m.get(k) for k in ("schema","variant","status","method","context_seed","case","requested_count","timings")},indent=2))
if __name__=="__main__":main()
