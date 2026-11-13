"""V3 lineage/finite-support/FIFO and fixed-Z engineering tests; no new draws."""
from copy import deepcopy
import hashlib,importlib.util,json,pickle,random
from pathlib import Path
from types import FunctionType
from unittest.mock import patch
import numpy as np
import pytest
from posteriorforge import prior_coordinate_pair_v1 as pairs,prior_proposals_v3 as proposals
from posteriorforge.artifacts import versions
from posteriorforge.science.analytical_benchmark_v2 import array_descriptor
from posteriorforge.inference import support_mask
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('_test_refresh_v3',ROOT/'scripts/refresh_prior_pair_v3_global_block.py');w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
SPEC=pairs.FIRST_PAIR

def state():return pickle.dumps((np.random.get_state(),random.getstate()),protocol=5)

def footer():
 states={k:'same' for k in ('numpy','python','torch_CPU','torch_CUDA')};heads=100
 return {'schema':w.PRODUCER_SCHEMA,'variant':w.PRODUCER_VARIANT,'status':'complete','runner_source_sha256':w.PRODUCER_SHA256,
 'pair_spec':SPEC.as_dict(),'context_seed':SPEC.context_seed,'case':SPEC.case,'context_size':4096,'requested_count':50000,'benchmark_total_count':50000,
 'repetition':0,'rank':0,'world_size':1,'mode':'fp32_transparent_control_v1','native_all8_stages':[6,10,14],'FP16_activated':False,
 'sources_unchanged':True,'default_rng_preserved':True,'completed_stage_indices':list(range(21)),
 'samples_in_original_physical_order_before_FIFO_support':True,'all_generated_rows_finite_physical_proposals_including_unused_FIFO_tail':True,
 'model_coordinate_decode_after_global_refresh':False,'sampler_math_changed':True,'partial_vector_filtering':False,'clipping_or_repair_performed':False,
 'physical_log_density_claimed':False,'learned_targets_are_original_physical_values':False,'physical_G3_in_this_neural_request':False,
 'fallback_map':{},'fallback_map_scope':'unchanged0199 F32 neural session; probability arithmetic and output contract are separately v3',
 'matching_conditional_qualification_sha256':w.PROOF_SHA256,'matching_cached_native_FP32_execution_passed_all21':True,
 'fresh_same_policy_all21_screen_replayed':True,'actual_fixed_conditional_screen_performed_during_request':False,
 'pair_cpu_receipt_sha256':w.PAIR_CPU_RECEIPT_SHA256,'preregistration_sha256':w.PROTOCOL_SHA256,'native_protocol_sha256':w.PROTOCOL_SHA256,
 'strict_training_encoder':w.codec.metadata(),'default_rng_before':states.copy(),'default_rng_after':states.copy(),
 'execution_counters':{},'generated_proposal_wave_count':1,'actual_prediction_checks':{'actual_head_contexts':heads,'sampler_callback_calls':heads-7,'shared_densities':1},
 'sampling_counters':{},'expected_probability_translation_count':heads,
 'probability_adapter':{'closed':True,'active':False,'source_unchanged':True,'SDK_global_aliases_modified':False,'module_sha256':w.PROBABILITY_SHA256,
 'translation_call_count':heads,'actual_probability_dtype':'torch.float64','inverse_call_count':10,'stage0_assembly_count':1,
 'legacy_preparation_guards':[{'original_alias_restored':True,'sdk_sources_unchanged':True,'patched_call_count':0} for _ in range(21)]},
 'timings':{'posterior_request_seconds':1.,'includes_stage_preparation':True,'includes_CPU_sampling':True,'includes_all_finite_proposal_wave_decode':True},
 'memory':{'peak_allocated_bytes':1,'peak_reserved_bytes':2},'original_gate_unchanged':True}

def test_actual_v3_header_uses_truthful_fields_not_legacy_strict_domain():
 d=footer();assert 'codec' not in d and 'all_generated_rows_strictly_decoded_including_unused_FIFO_tail' not in d
 assert w.validate_header(d,SPEC,w.PROOF_SHA256)['v3_finite_F64_parent_only']

@pytest.mark.parametrize('fault',['case','seed','pairspec','bool','schema1','schema2','failed','partial','source','count','proof','training','finite','G3','sampler','RNG','CDF','heads','inverse','dtype','legacy','stages','timing','memory','fixture','gate'])
def test_failed_old_partial_wrong_source_or_footer_rejected(fault):
 d=footer();proof=w.PROOF_SHA256
 if fault=='case':d['case']=1
 elif fault=='seed':d['context_seed']+=1
 elif fault=='pairspec':d['pair_spec']['case']=1
 elif fault=='bool':d['pair_spec']['case']=False
 elif fault=='schema1':d['schema']='posteriorforge.inference.prior_coordinate_pair.v1'
 elif fault=='schema2':d['schema']='posteriorforge.inference.prior_coordinate_pair.v2'
 elif fault=='failed':d['status']='failed'
 elif fault=='partial':d['status']='running'
 elif fault=='source':d['runner_source_sha256']='0'*64
 elif fault=='count':d['requested_count']=49999
 elif fault=='proof':proof='1'*64
 elif fault=='training':d['strict_training_encoder']['module_sha256']='wrong'
 elif fault=='finite':d['all_generated_rows_finite_physical_proposals_including_unused_FIFO_tail']=False
 elif fault=='G3':d['physical_G3_in_this_neural_request']=True
 elif fault=='sampler':d['sampler_math_changed']=False
 elif fault=='RNG':d['default_rng_after']['torch_CUDA']='changed'
 elif fault=='CDF':d['probability_adapter']['closed']=False
 elif fault=='heads':d['expected_probability_translation_count']-=1
 elif fault=='inverse':d['probability_adapter']['inverse_call_count']=0
 elif fault=='dtype':d['probability_adapter']['actual_probability_dtype']='torch.float32'
 elif fault=='legacy':d['probability_adapter']['legacy_preparation_guards'][0]['patched_call_count']=1
 elif fault=='stages':d['completed_stage_indices'].pop()
 elif fault=='timing':d['timings']['includes_all_finite_proposal_wave_decode']=False
 elif fault=='memory':d['memory']['peak_allocated_bytes']=0
 elif fault=='fixture':d['test_fixture']=True
 else:d['original_gate_unchanged']=False
 with pytest.raises(ValueError):w.validate_header(d,SPEC,proof)

def test_other_pair_cannot_use_registered_case0_proof():
 d=footer();wrong=pairs.PairSpec(20261111,1)
 d.update(pair_spec=wrong.as_dict(),case=1)
 with pytest.raises(ValueError,match='registered first'):w.validate_header(d,wrong,w.PROOF_SHA256)

@pytest.mark.parametrize('sha',w.INELIGIBLE_PARENT_SHA256)
def test_explicit_failed_parent_rejected_before_any_payload_or_decode(sha):
 # Only this test-instance source boundary is supplied; no old/global edits.
 ns=dict(w.load_parent.__globals__);ns['source_guard']=lambda:(None,None,{})
 f=FunctionType(w.load_parent.__code__,ns,w.load_parent.__name__,w.load_parent.__defaults__,w.load_parent.__closure__);f.__kwdefaults__=w.load_parent.__kwdefaults__
 with pytest.raises(ValueError,match='failed earlier parent'):f('never-opened',sha,scratch='unused',spec=SPEC,execution_screen='unused',pair_cpu_receipt='unused')

def engineering_waves(two=False):
 latent=np.zeros((6,21),np.float32);latent[1,15]=np.float32(13.876729011535645);latent[2,19]=np.float32(205.2462615966797)
 latent[3,15]=np.float32(-13.876729011535645);latent[4,3]=np.float32(.25);latent[5,0]=np.float32(1.)
 if two:latent=latent[[0,3,1,2,4,5]]
 waves=[latent[:3],latent[3:]] if two else [latent]
 decoded=[proposals.decode_ordered_proposals(a) for a in waves]
 stream=np.concatenate(decoded);assert support_mask(stream[:5]).sum()==4
 values=stream[:5][support_mask(stream[:5])]
 keys=[f'wave_{i:04d}' for i in range(len(waves))]
 d={'versions':versions(),'latent_wave_archive':{'keys':keys,'generated_row_count':6,'all_generated_rows_including_unused_tail':True},
 'decoded_wave_archive':{'all_finite_precision_valid_proposals_preserved':True,'original_support_selection_not_applied_to_archive':True,'generated_proposal_row_count':6,'all_rows_including_unused_FIFO_tail':True},
 'proposal_wave_sizes':[len(a) for a in waves],'generated_proposal_wave_count':len(waves),
 'rejection':{'proposed':5,'accepted_before_final_trim':4,'returned':4},
 'proposal_buffer':{'generated_count':6,'original_api_delivered_count':5,'remaining_unused_buffer_count':1},'decode_wave_records':[]}
 for i,(a,b) in enumerate(zip(waves,decoded)):
  d['decode_wave_records'].append({'wave_index':i,'row_count':len(a),'actual_latent_draws':array_descriptor(a),'decoded_physical':array_descriptor(b),
   'finite_proposal_decode_passed':True,'physical_proposal_contract_sha256':w.PROPOSAL_SHA256,'support_selection_not_applied_by_decoder':True,
   'includes_unused_FIFO_tail':True,'input_dtype_unchanged_before_decode':True})
 ns=dict(w.replay_generated_decode.__globals__);ns['COUNT']=4
 replay=FunctionType(w.replay_generated_decode.__code__,ns,'explicit_six_row_engineering_replay')
 return d,dict(zip(keys,waves)),values,dict(zip(keys,decoded)),replay

@pytest.mark.parametrize('two',[False,True])
def test_finite_endpoint_and_outside_proposals_replay_original_support_prefix(two):
 before=state();d,latent,values,decoded,replay=engineering_waves(two)
 raw={k:a.tobytes() for k,a in decoded.items()};r=replay(d,latent,values,decoded)
 assert r['passed'] and r['generated_finite_proposals_replayed']==6 and r['checked_rows']==5 and r['accepted_before_trim']==4
 assert r['unused_FIFO_tail_rows']==1 and r['d8_strict_open_or_noise_latent_roundtrip_reimposed'] is False
 assert any(np.any(a[:,[4,8,12,16,18,20]]==.5) for a in decoded.values())
 assert any(np.any(a[:,[4,8,12,16,18,20]]<.01) for a in decoded.values())
 assert all(a.tobytes()==raw[k] for k,a in decoded.items()) and state()==before
 with pytest.raises(w.codec.CoordinateError):w.codec.encode_physical(values)

@pytest.mark.parametrize('fault',['dtype','shape','bytes','finiteflag','source','tail','checked','accepted','returned','FIFO','archive'])
def test_finite_replay_never_repairs_or_promotes_incomplete_lineage(fault):
 d,latent,values,decoded,replay=engineering_waves()
 if fault=='dtype':decoded['wave_0000']=decoded['wave_0000'].astype(np.float32)
 elif fault=='shape':decoded['wave_0000']=decoded['wave_0000'][:5]
 elif fault=='bytes':decoded['wave_0000']=decoded['wave_0000'].copy();decoded['wave_0000'][0,0]+=1
 elif fault=='finiteflag':d['decode_wave_records'][0]['finite_proposal_decode_passed']=False
 elif fault=='source':d['decode_wave_records'][0]['physical_proposal_contract_sha256']='0'*64
 elif fault=='tail':d['latent_wave_archive']['all_generated_rows_including_unused_tail']=False
 elif fault=='checked':d['rejection']['proposed']=7
 elif fault=='accepted':d['rejection']['accepted_before_final_trim']=5
 elif fault=='returned':d['rejection']['returned']=3
 elif fault=='FIFO':d['proposal_buffer']['original_api_delivered_count']=4
 else:d['decoded_wave_archive']['original_support_selection_not_applied_to_archive']=False
 with pytest.raises(ValueError):replay(d,latent,values,decoded)

class FixedZGenerator(np.random.Generator):
 """Declared engineering values only; no random method is called."""
 def __init__(self,bit_generator):super().__init__(bit_generator);self.standard_normal_calls=0;self.actual_draws=None
 def standard_normal(self,size):
  assert self.standard_normal_calls==0;self.standard_normal_calls+=1
  self.actual_draws=np.linspace(-1.,1.,np.prod(size)).reshape(size)
  return self.actual_draws

def test_exact_G3_code_seed_role_and_only_globals_with_closed_endpoints_fixed_Z():
 _,_,values,_,_=engineering_waves();before=values.tobytes();rng=state()
 ns=dict(w.refresh_with_timing.__globals__);ns.update(COUNT=4,_RECORDING_GENERATOR=FixedZGenerator)
 f=FunctionType(w.refresh_with_timing.__code__,ns,'four_row_fixed_Z_engineering_refresh')
 after,z,receipt=f(values,SPEC)
 assert f.__code__ is w._ANCESTOR.refresh_with_timing.__code__
 mean,cov=w.kernel.conditional_parameters(values)
 assert after[:,:3].tobytes()==(mean+z@np.linalg.cholesky(cov).T).tobytes()
 assert after[:,3:].tobytes()==values[:,3:].tobytes() and values.tobytes()==before and state()==rng
 assert receipt['role']==w.ROLE=='ss15_global_block_gibbs_once_v1'
 assert receipt['seed']==w.derive_seed(SPEC.context_seed,w.ROLE,case=SPEC.case,repetition=0)
 assert receipt['standard_normal_method_calls']==1 and receipt['coordinate_decode_after_global_refresh'] is False
 assert receipt['all18_physical_columns_byte_identical'] and support_mask(after).all()

def test_original_sources_globals_code_objects_and_production_COUNT_unchanged():
 before=state();assert w.COUNT==50000 and w.ROLE==w._ANCESTOR.ROLE
 assert w.kernel.source_sha256()==w.KERNEL_SHA256 and w.codec.source_sha256()==w.CODEC_SHA256
 assert w.derivation_descriptor()['unchanged_functions_same_code_objects'] and w.derivation_descriptor()['old_globals_unmodified']
 assert w.source_guard(require_committed=False)[2]['sources_committed'] is False
 assert state()==before
 with patch.object(w,'PROPOSAL_SHA256','0'*64):
  with pytest.raises(ValueError):w.source_guard(require_committed=False)
 with patch.object(w,'refresh_with_timing',lambda *a:None):
  with pytest.raises(ValueError):w.source_guard(require_committed=False)

def query_witness(stage,member,backend):
 result={'passed':True,'stage':stage,'member':member,'qkv_calls':36,'attention_calls':36}
 if stage in backend.NATIVE_STAGES:
  result.update(route=backend.route(stage),original_query_views_disabled=True)
 else:result.update(mode=backend.CONTROL_MODE,fallback_map={},scope='same active session; aliases restored only on later context exit')
 return result


def execution_fixture():
 """Synthetic source-bound journal only, not a parent or numerical qualification."""
 from types import SimpleNamespace
 from posteriorforge import performance_prior_pair_v3 as producer
 backend=producer.backend;theta=np.zeros((4096,21),np.float32);obs=np.zeros((4096,15),np.float32)
 scopes=[];training=[];calls=[];events=[];delegated=[]
 model={'canonical_borders_dtype':'torch.float32','physical_borders_dtype':'torch.float32','temperature':.9,
  'contract':{'input_normalization_train_only_verified':True,'modules':[]}}
 for stage in range(21):
  raw=np.ones((1,15+stage),np.float32);masked=backend.mask_query(raw,stage)
  for member in range(8):
   calls.append({'stage':stage,'member':member,'query_unmasked':array_descriptor(raw),'query_masked':array_descriptor(masked),
    'mask':backend.stage_policy(stage),'execution_route':backend.route(stage),'witness':query_witness(stage,member,backend),
    'probability_policy_source_sha256':w.PROBABILITY_SHA256,'neural_prefix':{
     'module_sha256':w.NATIVE_BACKEND_SHA256,'neural_source_sha256':w.NEURAL_SOURCE_SHA256,'stage':stage,'member':member,
     'actual_adjusted_dtype':'torch.float32','actual_probability_dtype':'torch.float64','model_rng_isolation_scope_delta':1,'selected_member_lists_restored':True}})
  prefill={'passed':True,'native_neural_prefill_calls':0,'cached_prefill_claimed':False}
  scopes.append({'stage_index':stage,'failed':False,'actual_head_contexts':8,'precision_session':{'synthetic_witness_delegation_stage':stage},
   'preparation_witness':prefill,'model_at_prepare':deepcopy(model),'immutable_native_before':{'fixture':'immutable'},'immutable_native_after':{'fixture':'immutable'},
   'v3_prefix_counters':{'prefix_query_count':8,'prefix_query_rows':8,'cached_member_lists_restored':True,'original_preprocessing_lists_restored':True}})
  training.append(backend.training_bindings(theta,obs,stage))
  events.append({'stage_index':stage,'sampler':{'neural_query_batch_sizes_by_member':{str(m):{'1':1} for m in range(8)}}})
 real=producer.physical_sampling_namespace();seeds=real.worker_stream_seeds(SPEC.context_seed,SPEC.case,0,0,1);policy=real.proposal_buffer_policy(1200)
 support={'original_support_check_code_and_closure_verified':True,'original_support_source_sha256':producer.physical_fifo.ORIGINAL_INFERENCE_SHA256}
 def validate_fifo(record,*,world_size,rank,scope):
  assert record==support and world_size==1 and rank==0 and scope=={'declared':'engineering-only'};delegated.append('FIFO')
 frozen=SimpleNamespace(worker_stream_seeds=real.worker_stream_seeds,proposal_buffer_policy=real.proposal_buffer_policy,validate_proposal_buffer_record=validate_fifo)
 proxy=SimpleNamespace(backend=backend,MODE=producer.MODE,physical_fifo=producer.physical_fifo,physical_sampling_namespace=lambda:frozen,
  proposal_scope=lambda *a:{'declared':'engineering-only'})
 def neural(report,stage,actual_backend):
  assert report['synthetic_witness_delegation_stage']==stage and actual_backend is backend;delegated.append(stage)
 helper=SimpleNamespace(_neural_witness=neural)
 d={'generated_proposal_wave_count':1,'execution_counters':{'completed_stage_scopes':scopes,'actual_latent_training_inputs':training,
  'actual_latent_query_calls':calls,'prepare_count':21,'all_stage_aliases_restored':True,'actual_head_contexts':168,'shared_densities':1},
 'stage_events':events,'seed_streams':seeds,'proposal_buffer_policy':policy,'proposal_buffer':support,'model_inputs':{},
 'sampling_counters':{'stage0_shared_predictions':1,'component_predictions':160},'expected_probability_translation_count':168,
 'probability_adapter':{'translation_call_count':168},'actual_prediction_checks':{'actual_head_contexts':168}}
 return d,proxy,{'theta':theta,'obs':obs,'raw':{}},helper,delegated

def test_new_execution_journal_API_and_FIFO_witness_delegation_constructor_free():
 before=state();d,p,a,h,delegated=execution_fixture();r=w.execution_checks(d,p,a,h,SPEC)
 assert r['passed'] and r['stage_count']==21 and r['actual_head_contexts']==168
 assert delegated==list(range(21))+['FIFO'] and state()==before

@pytest.mark.parametrize('fault',['stage','training','dtype','prefix','native','counts','support','Pcount'])
def test_execution_journal_changed_v3_geometry_route_source_count_or_support_rejected(fault):
 d,p,a,h,_=execution_fixture();x=d['execution_counters']
 if fault=='stage':x['completed_stage_scopes'][1]['stage_index']=0
 elif fault=='training':x['actual_latent_training_inputs'][1]['stage']=2
 elif fault=='dtype':x['actual_latent_query_calls'][1]['query_masked']['dtype']='<f8'
 elif fault=='prefix':x['actual_latent_query_calls'][1]['neural_prefix']['module_sha256']='0'*64
 elif fault=='native':x['completed_stage_scopes'][6]['immutable_native_after']={'changed':True}
 elif fault=='counts':x['completed_stage_scopes'][0]['v3_prefix_counters']['prefix_query_rows']=7
 elif fault=='support':d['proposal_buffer']['original_support_check_code_and_closure_verified']=False
 else:d['expected_probability_translation_count']=167
 with pytest.raises(ValueError):w.execution_checks(d,p,a,h,SPEC)

@pytest.mark.parametrize('stage',[0,6,10,14,20])
def test_actual_native_and_cached_witness_schemas_do_not_share_mode_defaults(stage):
 from posteriorforge import execution_prior_native_v3 as backend
 witness=query_witness(stage,0,backend)
 assert w.validate_query_witness(witness,stage,0,backend,backend.CONTROL_MODE)
 assert ('mode' in witness)==(stage not in backend.NATIVE_STAGES)

@pytest.mark.parametrize('fault',['cached_missing_mode','cached_wrong_mode','cached_fallback','native_route','native_view','native_mode_schema','native_missing_schema','stage','member','boolcount','unequalcount','zerocount'])
def test_native_and_cached_witness_schema_route_view_and_counts_are_strict(fault):
 from posteriorforge import execution_prior_native_v3 as backend
 stage=0 if fault.startswith('cached') else 6;witness=query_witness(stage,0,backend)
 if fault=='cached_missing_mode':witness.pop('mode')
 elif fault=='cached_wrong_mode':witness['mode']='fp16'
 elif fault=='cached_fallback':witness['fallback_map']={'6':[0]}
 elif fault=='native_route':witness['route']='frozen88_resident_cache'
 elif fault=='native_view':witness['original_query_views_disabled']=False
 elif fault=='native_mode_schema':witness['mode']=backend.CONTROL_MODE
 elif fault=='native_missing_schema':witness.pop('original_query_views_disabled')
 elif fault=='stage':witness['stage']=5
 elif fault=='member':witness['member']=1
 elif fault=='boolcount':witness['qkv_calls']=True
 elif fault=='unequalcount':witness['attention_calls']-=1
 else:witness.update(qkv_calls=0,attention_calls=0)
 with pytest.raises(ValueError):w.validate_query_witness(witness,stage,0,backend,backend.CONTROL_MODE)

def test_actual_successful_parent_query_journal_native_schema_readonly():
 """Actual saved witness fields only; no parent refresh, samples or RNG call."""
 from posteriorforge import execution_prior_native_v3 as backend
 path=Path('/scratch/project/eu-26-17/eubrieucb/posteriorforge/results/accuracy-prior-coordinates-v3/neural-posterior-seed20261111-case0-01/manifest.json')
 if not path.exists():pytest.skip('authenticated genuine v3 saved journal unavailable')
 assert hashlib.sha256(path.read_bytes()).hexdigest()=='cbe866875983cc3271a22a5862a16fa7e4e3f79e1c28502dad1563c11b96ff32'
 d=json.loads(path.read_bytes());before=state();native=[];cached=[]
 for call in d['execution_counters']['actual_latent_query_calls']:
  stage=call['stage'];assert w.validate_query_witness(call['witness'],stage,call['member'],backend,backend.CONTROL_MODE)
  assert call['query_unmasked']['dtype']==call['query_masked']['dtype']=='<f4'
  assert call['neural_prefix']['actual_adjusted_dtype']=='torch.float32' and call['neural_prefix']['actual_probability_dtype']=='torch.float64'
  if stage in backend.NATIVE_STAGES:
   assert 'mode' not in call['witness'];native.append(stage)
  else:
   assert call['witness']['mode']==backend.CONTROL_MODE;cached.append(stage)
 assert {stage:native.count(stage) for stage in backend.NATIVE_STAGES}=={6:57,10:56,14:56}
 assert len(native)==169 and cached and state()==before
 first=next(c for c in d['execution_counters']['actual_latent_query_calls'] if c['stage']==6)
 corrupted=deepcopy(first['witness']);corrupted['route']='frozen88_resident_cache'
 with pytest.raises(ValueError):w.validate_query_witness(corrupted,6,first['member'],backend,backend.CONTROL_MODE)
 print('ACTUAL_JOURNAL_WITNESS_SCHEMA='+json.dumps({'native_calls':len(native),'cached_calls':len(cached),'native_by_stage':{str(s):native.count(s) for s in backend.NATIVE_STAGES},'all_F32_prefix_and_F64_probability':True,'no_mode_default':True},sort_keys=True))
