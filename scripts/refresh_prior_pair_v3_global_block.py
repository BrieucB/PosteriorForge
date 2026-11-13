#!/usr/bin/env python3
"""ONE unchanged physical G3 after a complete, own-pair v3 physical parent.

Built with PriorLabs-TabPFN. Only lineage/finite-proposal validation is new.
No failed wave recovery, latent-local re-decoding after G3, or changed draw law.
"""
from __future__ import annotations
import hashlib,importlib.util,inspect
from pathlib import Path
from types import FunctionType

V1_WRAPPER_SHA256='22e57084351631f36c6989cc01fe8bf4f2827095e587c496cda2e7f368d3cdb9'
_path=Path(__file__).with_name('refresh_prior_pair_global_block_v1.py')
if hashlib.sha256(_path.read_bytes()).hexdigest()!=V1_WRAPPER_SHA256:raise ValueError('frozen v1 refresh ancestry changed')
_spec=importlib.util.spec_from_file_location('_v3_G3_frozen_v1',_path);_ANCESTOR=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_ANCESTOR)
for _key,_value in vars(_ANCESTOR).items():
 if not _key.startswith('__'):globals()[_key]=_value
for _key,_value in vars(_ANCESTOR).items():
 if isinstance(_value,FunctionType) and _value.__globals__ is vars(_ANCESTOR):
  _clone=FunctionType(_value.__code__,globals(),_value.__name__,_value.__defaults__,_value.__closure__)
  _clone.__kwdefaults__=_value.__kwdefaults__;globals()[_key]=_clone

SCHEMA='posteriorforge.prior_coordinate_pair_global_refresh.v3'
PRODUCER_SCHEMA='posteriorforge.inference.prior_coordinate_pair.v3'
PRODUCER_VARIANT='prior_coordinates_F32_forward_FP64_P_finite_F64_proposals_v3'
PRODUCER_SHA256='28e5c5f6a613b5563a0db7368b30713ee8f7ec251350c99f3112e46a5e462651'
PAIR_QUALIFIER_SHA256='534b48c789bf1168a33b2953d85d747e27c244063dc9fc7cc4b4cb09c4d8324a'
NATIVE_BACKEND_SHA256='7c7180e3f161f9fc34684fc492f9ab558660b3b41092b70feea17edbbbf3f19c'
NEURAL_SOURCE_SHA256='0199b16e9cc34203fa0d5645d161742fe9ed244633d7beee047fbcbf2528661c'
PROBABILITY_SHA256='7fea13270c38c51eed0eb65cd5026de94e3f7e24cc4734e8906e882fb0bc0494'
PROPOSAL_SHA256='9216e76ce335ee52ffb1a5e8fbdd33c72f8c0aa84c4f1803dd90ff06b3e3a6d7'
PROTOCOL_SHA256='7ea6e5c57f8404eb761e07f4011b14966d5885fbbdb8620d463c5283916041d8'
PROTOCOL_PATH='docs/accuracy-prior-coordinate-fp64-proposals-v3.md'
PAIR_DATA_PROTOCOL_SHA256='9fc6a91568864cc996dafbcfaa41e77280e4825c63f61d455bdadf3703cd0a66'
PROOF_SHA256='56ec6cdc81eeab4c755d24fa8ddae47c21f75bce4c624c43908a59f5da21ffd1'
INELIGIBLE_PARENT_SHA256=('92755b788fc894d4ca16157e573bd1b5c9d8a4468b71cf184fa16303bced7bce','25e5e9e001302627ade825a39a29231852f560bcb567b4e6a7504039eb449838')


def derivation_descriptor():
 return {'schema':'posteriorforge.prior_pair_v3_downstream_derivation.v1','v1_wrapper_sha256':V1_WRAPPER_SHA256,
  'unchanged_functions_same_code_objects':all(globals()[n].__code__ is getattr(_ANCESTOR,n).__code__ for n in ('refresh_with_timing','refresh','_safe_sibling')),
  'old_globals_unmodified':_ANCESTOR.SCHEMA=='posteriorforge.prior_coordinate_pair_global_refresh.v1' and _ANCESTOR.PRODUCER_SHA256=='52eaa638d09358765fcbb1d229265a5679f05a5fe0a5866f513412c7dc572a0b',
  'numerical_G3_seed_or_metric_math_changed':False,'parent_validation_contract':'v3 finite F64 proposals; original whole-vector support/FIFO; strict training encoder remains d8',
  'old_failed_wave_recovery':False,'coordinate_decode_after_global_refresh':False}


def source_guard(*,require_committed=True):
 from posteriorforge import performance_prior_pair_v3 as producer,prior_coordinate_pair_v1 as pairs
 require(sha256_file(_ANCESTOR.__file__)==V1_WRAPPER_SHA256 and sha256_file(BASE.__file__)==BASE_WRAPPER_SHA256
  and sha256_file(DRAW.__file__)==DRAW_SOURCE_SHA256,'frozen validation/draw ancestry changed')
 require(producer.source_sha256()==PRODUCER_SHA256 and pairs.source_sha256()==PAIR_SOURCE_SHA256,'v3 producer/data source unpinned or changed')
 producer._integrity(require_committed=require_committed);pairs.require_sources()
 require(producer.qualification_sha256(pairs.FIRST_PAIR)==PROOF_SHA256,'fresh own-pair v3 proof binding differs')
 require(producer.backend.source_sha256()==NATIVE_BACKEND_SHA256 and producer.backend.NEURAL_SHA==NEURAL_SOURCE_SHA256
  and producer.probability.source_sha256()==PROBABILITY_SHA256 and producer.proposals.source_sha256()==PROPOSAL_SHA256,'frozen v3 numerical policy changed')
 require(producer.DEPENDENCY_SHA256['scripts/'+producer.QUALIFIER_FILENAME]==PAIR_QUALIFIER_SHA256
  and producer.PAIR_CPU_RECEIPT_SHA256==PAIR_CPU_RECEIPT_SHA256 and producer.physical_fifo.source_sha256()==F64_HELPER_SHA256,'v3 proof/FIFO ancestry differs')
 require(kernel.source_sha256()==KERNEL_SHA256 and codec.source_sha256()==CODEC_SHA256,'unchanged physical G3 kernel/training codec changed')
 require(_RECORDING_GENERATOR is DRAW.RecordingGenerator and _RECORDING_GENERATOR.standard_normal.__code__ is _RECORDING_CODE
  and kernel.refresh is _KERNEL_REFRESH and _KERNEL_REFRESH.__code__ is _KERNEL_REFRESH_CODE,'original G3/draw callable changed')
 for name,fn,code in _PUBLIC_SEALS:require(globals().get(name) is fn and fn.__code__ is code,'v3 wrapper callable changed: '+name)
 for name,value in _CONFIG_SEALS.items():require(globals().get(name) is value,'v3 wrapper binding changed: '+name)
 derivation=derivation_descriptor();require(derivation['unchanged_functions_same_code_objects'] and derivation['old_globals_unmodified'],'original G3 wrapper modified')
 root=Path(__file__).resolve().parents[1];require(sha256_file(root/PROTOCOL_PATH)==PROTOCOL_SHA256,'v3 protocol changed')
 revision=subprocess.run(['git','-C',str(root),'rev-parse','HEAD'],check=True,capture_output=True,text=True).stdout.strip()
 sources={'scripts/'+Path(__file__).name:source_sha256(),'scripts/refresh_prior_pair_global_block_v1.py':V1_WRAPPER_SHA256,
  'src/posteriorforge/performance_prior_pair_v3.py':PRODUCER_SHA256,'scripts/qualify_prior_coordinate_pair_v3.py':PAIR_QUALIFIER_SHA256,
  'src/posteriorforge/execution_prior_native_v3.py':NATIVE_BACKEND_SHA256,'src/posteriorforge/execution_probability_v3.py':PROBABILITY_SHA256,
  'src/posteriorforge/prior_proposals_v3.py':PROPOSAL_SHA256,'src/posteriorforge/conditional_globals.py':KERNEL_SHA256,
  'src/posteriorforge/prior_coordinates_v1.py':CODEC_SHA256,PROTOCOL_PATH:PROTOCOL_SHA256}
 if require_committed:BASE.verify_git_sources(root,revision,sources)
 return producer,pairs,{'wrapper':source_sha256(),'pair_producer':PRODUCER_SHA256,'pair_data':PAIR_SOURCE_SHA256,
  'pair_qualifier':PAIR_QUALIFIER_SHA256,'native_backend':NATIVE_BACKEND_SHA256,'neural_source':NEURAL_SOURCE_SHA256,
  'probability':PROBABILITY_SHA256,'proposal':PROPOSAL_SHA256,'pair_CPU_receipt':PAIR_CPU_RECEIPT_SHA256,
  'kernel':KERNEL_SHA256,'codec':CODEC_SHA256,'F64_helper':F64_HELPER_SHA256,'protocol':PROTOCOL_SHA256,
  'guard_primitive_source':BASE_WRAPPER_SHA256,'actual_draw_observer_source':DRAW_SOURCE_SHA256,
  'v1_validation_ancestry':V1_WRAPPER_SHA256,'git_revision':revision,'sources_committed':bool(require_committed),'derivation':derivation}


def validate_header(document,spec,proof_sha256):
 from posteriorforge import prior_coordinate_pair_v1 as pairs
 require(type(spec) is pairs.PairSpec and spec==pairs.FIRST_PAIR,'registered first v3 pair required')
 require(not document.get('test_fixture'),'CPU test fixture is not a production request')
 require(proof_sha256==PROOF_SHA256,'fresh registered own-pair v3 proof required')
 expected={'schema':PRODUCER_SCHEMA,'variant':PRODUCER_VARIANT,'status':'complete','runner_source_sha256':PRODUCER_SHA256,
  'pair_spec':spec.as_dict(),'context_seed':spec.context_seed,'case':spec.case,'context_size':4096,'requested_count':COUNT,
  'benchmark_total_count':COUNT,'repetition':0,'rank':0,'world_size':1,'mode':'fp32_transparent_control_v1','native_all8_stages':[6,10,14],
  'FP16_activated':False,'sources_unchanged':True,'default_rng_preserved':True,'completed_stage_indices':list(range(21)),
  'samples_in_original_physical_order_before_FIFO_support':True,'all_generated_rows_finite_physical_proposals_including_unused_FIFO_tail':True,
  'model_coordinate_decode_after_global_refresh':False,'sampler_math_changed':True,'partial_vector_filtering':False,
  'clipping_or_repair_performed':False,'physical_log_density_claimed':False,'learned_targets_are_original_physical_values':False,
  'physical_G3_in_this_neural_request':False,'fallback_map':{},
  'fallback_map_scope':'unchanged0199 F32 neural session; probability arithmetic and output contract are separately v3',
  'matching_conditional_qualification_sha256':proof_sha256,'matching_cached_native_FP32_execution_passed_all21':True,
  'fresh_same_policy_all21_screen_replayed':True,'actual_fixed_conditional_screen_performed_during_request':False,
  'pair_cpu_receipt_sha256':PAIR_CPU_RECEIPT_SHA256,'preregistration_sha256':PROTOCOL_SHA256,'native_protocol_sha256':PROTOCOL_SHA256,
  'strict_training_encoder':codec.metadata()}
 for key,value in expected.items():require(not BASE.policy_differences(document.get(key),value),'v3 physical production header differs: '+key)
 require(document.get('default_rng_before')==document.get('default_rng_after') and isinstance(document.get('default_rng_after'),dict)
  and set(document['default_rng_after'])=={'numpy','python','torch_CPU','torch_CUDA'},'full default RNG footer missing or changed')
 for key in ('execution_counters','sampling_counters','actual_prediction_checks'):require(isinstance(document.get(key),dict),'normal v3 production footer missing: '+key)
 waves=document.get('generated_proposal_wave_count');require(type(waves) is int and waves>0,'positive completed proposal-wave count required')
 actual=document.get('actual_prediction_checks',{});counts=document.get('sampling_counters',{})
 require(all(type(actual.get(k)) is int and actual[k]>0 for k in ('actual_head_contexts','sampler_callback_calls','shared_densities'))
  and actual['shared_densities']==waves and actual['sampler_callback_calls']+7*waves==actual['actual_head_contexts'],'actual v3 callback/head conservation differs')
 scope=document.get('probability_adapter',{});heads=document.get('expected_probability_translation_count')
 require(scope.get('closed') is True and scope.get('active') is False and scope.get('source_unchanged') is True
  and scope.get('SDK_global_aliases_modified') is False and scope.get('module_sha256')==PROBABILITY_SHA256
  and scope.get('translation_call_count')==heads==actual['actual_head_contexts']
  and scope.get('actual_probability_dtype')=='torch.float64' and type(scope.get('inverse_call_count')) is int
  and scope['inverse_call_count']>0 and scope.get('stage0_assembly_count')==waves,'actual PolicyC source/translation/inverse/shared scope differs')
 guards=scope.get('legacy_preparation_guards',[])
 require(len(guards)==21*waves and all(g.get('original_alias_restored') is True and g.get('sdk_sources_unchanged') is True
  and g.get('patched_call_count')==0 for g in guards),'legacy preparation-only source restoration differs')
 timing=document.get('timings',{});memory=document.get('memory',{})
 require(np.isfinite(timing.get('posterior_request_seconds',np.nan)) and timing['posterior_request_seconds']>0
  and timing.get('includes_stage_preparation') is True and timing.get('includes_CPU_sampling') is True
  and timing.get('includes_all_finite_proposal_wave_decode') is True,'successful finite-proposal timing scope missing')
 require(type(memory.get('peak_allocated_bytes')) is int and 0<memory['peak_allocated_bytes']<=memory.get('peak_reserved_bytes',0),'actual memory footer missing')
 require(document.get('original_gate_unchanged') is True,'original whole-vector support gate must be unchanged')
 return {'passed':True,'pair_spec':spec.as_dict(),'fresh_same_pair_proof_sha256':proof_sha256,'v3_finite_F64_parent_only':True,'old_family_or_failed_parent_promoted':False}


def verify_canonical_policy(recorded,current):
 require(not BASE.policy_differences(recorded,current),'actual v3 canonical source/policy/AST/inventory differs')
 require(recorded.get('schema')=='posteriorforge.prior_coordinate_pair_execution.v3'
  and recorded.get('inherited_v1_or_v2_numerical_qualification') is False
  and recorded.get('probability_arithmetic_and_output_precision_contract_changed') is True
  and recorded.get('neural_weights_or_training_representation_changed') is False
  and recorded.get('original_physical_support_and_scientific_gates_changed') is False,'explicit v3 probability/proposal scope required')
 return {'passed':True,'all_canonical_metadata_fields_exact_TYPE_and_value':True,'new_v3_probability_and_physical_contract_explicit':True,'old_numerical_qualification_inherited':False}


def replay_generated_decode(document,latent_waves,physical_values,decoded_waves):
 """Replay every finite proposal, then original support selection of checked rows."""
 from posteriorforge import prior_proposals_v3 as proposals
 from posteriorforge.inference import support_mask
 old=load('refresh_prior_coordinates_global_block_v1.py',BASE.OLD_WRAPPER_SHA256)
 environment=old.require_replay_environment(document)
 require(isinstance(physical_values,np.ndarray) and physical_values.dtype==np.float64 and physical_values.shape==(COUNT,21) and np.isfinite(physical_values).all(),'complete physical F64 selected rows required')
 keys=document['latent_wave_archive']['keys'];records=document['decode_wave_records']
 require(keys==[f'wave_{i:04d}' for i in range(len(keys))] and len(keys)>0 and set(latent_waves)==set(decoded_waves)==set(keys),'complete ordered finite proposal archives required')
 require(len(records)==len(keys)==len(document['proposal_wave_sizes'])==document['generated_proposal_wave_count'],'all generated proposal waves must be journaled')
 all_physical=[]
 for index,key in enumerate(keys):
  latent=latent_waves[key];r=records[index]
  require(latent.dtype==np.float32 and latent.ndim==2 and latent.shape[1]==21 and len(latent)==document['proposal_wave_sizes'][index],'unchanged F32 complete latent wave required')
  require(r['wave_index']==index and r['row_count']==len(latent) and r['actual_latent_draws']==array_descriptor(latent),'actual latent-wave descriptor differs')
  physical=proposals.decode_ordered_proposals(latent);archived=decoded_waves[key]
  require(archived.dtype==np.float64 and archived.shape==physical.shape and archived.tobytes()==physical.tobytes(),'finite physical archive differs from original v3 inverse result')
  require(r.get('finite_proposal_decode_passed') is True and r.get('physical_proposal_contract_sha256')==PROPOSAL_SHA256
   and r.get('support_selection_not_applied_by_decoder') is True and r.get('includes_unused_FIFO_tail') is True
   and r.get('input_dtype_unchanged_before_decode') is True and r['decoded_physical']==array_descriptor(physical),'finite proposal decode journal differs')
  all_physical.append(physical)
 stream=np.concatenate(all_physical,axis=0);reject=document['rejection'];checked=reject['proposed']
 require(type(checked) is int and COUNT<=checked<=len(stream) and reject['returned']==COUNT,'original complete-vector support count differs')
 accepted=stream[:checked][support_mask(stream[:checked])]
 require(len(accepted)==reject['accepted_before_final_trim'] and len(accepted)>=COUNT and accepted[:COUNT].tobytes()==physical_values.tobytes(),'physical output differs from original supported-prefix selection')
 require(document['latent_wave_archive']['generated_row_count']==len(stream)
  and document['latent_wave_archive']['all_generated_rows_including_unused_tail'] is True,'unused latent FIFO tail omitted')
 info=document['decoded_wave_archive']
 require(info.get('all_finite_precision_valid_proposals_preserved') is True and info.get('original_support_selection_not_applied_to_archive') is True
  and info.get('generated_proposal_row_count')==len(stream) and info.get('all_rows_including_unused_FIFO_tail') is True,'complete finite physical proposal archive required')
 buffer=document['proposal_buffer']
 require(buffer.get('generated_count')==len(stream) and buffer.get('original_api_delivered_count')==checked
  and buffer.get('remaining_unused_buffer_count')==len(stream)-checked,'original FIFO delivered/unused proposal counts differ')
 return {'passed':True,'generated_finite_proposals_replayed':len(stream),'checked_rows':checked,
  'accepted_before_trim':len(accepted),'unused_FIFO_tail_rows':len(stream)-checked,'physical_saved_output_byte_identical':True,
  'finite_full_decoded_proposal_archive_byte_identical':True,'source':PROPOSAL_SHA256,'exact_replay_environment':environment,
  'd8_strict_open_or_noise_latent_roundtrip_reimposed':False,'support_selection_owner':'unchanged original whole-vector support'}


def execution_checks(parent,producer,actual,helper,spec):
 from collections import Counter
 backend=producer.backend;theta,obs=actual['theta'],actual['obs'];execution=parent['execution_counters'];waves=parent['generated_proposal_wave_count']
 scopes=execution['completed_stage_scopes'];training=execution['actual_latent_training_inputs'];calls=execution['actual_latent_query_calls'];events=parent['stage_events']
 require(len(scopes)==len(training)==len(events)==execution['prepare_count']==21*waves and execution['all_stage_aliases_restored'] is True,'all actual v3 wave/stage scopes required')
 cursor=0
 for index,(scope,binding,event) in enumerate(zip(scopes,training,events)):
  stage=index%21;report=scope['precision_session'];prefill=scope['preparation_witness']
  require(scope['stage_index']==event['stage_index']==stage and scope['failed'] is False
   and binding==backend.training_bindings(theta,obs,stage),'actual v3 training target/mask or stage status differs')
  helper._neural_witness(report,stage,backend)
  model_guard(scope['model_at_prepare']);require(prefill.get('passed') is True,'actual v3 preparation witness missing')
  if stage in backend.NATIVE_STAGES:
   require(prefill.get('native_neural_prefill_calls')==0 and prefill.get('cached_prefill_claimed') is False
    and scope['immutable_native_before'] is not None and scope['immutable_native_before']==scope['immutable_native_after'],'native fitted immutable state differs')
  subset=calls[cursor:cursor+scope['actual_head_contexts']];observed=Counter();rows=0
  for call in subset:
   shape=call['query_unmasked']['shape'];masked=call['query_masked'];prefix=call['neural_prefix']
   require(call['stage']==stage and call['execution_route']==backend.route(stage) and call['mask']==backend.stage_policy(stage)
    and shape==masked['shape'] and shape[1]==15+stage and call['query_unmasked']['dtype']==masked['dtype']=='<f4'
    and call['witness'].get('passed') is True and call['witness'].get('mode')==producer.MODE
    and call['probability_policy_source_sha256']==PROBABILITY_SHA256,'actual query F32 geometry/mask/route/witness differs')
   require(prefix.get('module_sha256')==NATIVE_BACKEND_SHA256 and prefix.get('neural_source_sha256')==NEURAL_SOURCE_SHA256
    and prefix.get('stage')==stage and prefix.get('member')==call['member'] and prefix.get('actual_adjusted_dtype')=='torch.float32'
    and prefix.get('actual_probability_dtype')=='torch.float64' and prefix.get('model_rng_isolation_scope_delta')==1
    and prefix.get('selected_member_lists_restored') is True,'actual v3 NN-prefix/PolicyC lineage differs')
   if stage<3:require(call['query_unmasked']==masked,'global query unexpectedly masked')
   observed[(call['member'],shape[0])]+=1;rows+=shape[0]
  hist=event['sampler']['neural_query_batch_sizes_by_member']
  expected=Counter((member,int(n)) for member in range(8) for n,count in hist[str(member)].items() for _ in range(count))
  counters=scope['v3_prefix_counters']
  require(observed==expected and counters.get('prefix_query_count')==scope['actual_head_contexts']==len(subset)
   and counters.get('prefix_query_rows')==rows and counters.get('cached_member_lists_restored') is True
   and counters.get('original_preprocessing_lists_restored') is True,'actual selected-head/prefix/member-list counts differ')
  cursor+=scope['actual_head_contexts']
 frozen=producer.physical_sampling_namespace();seeds=frozen.worker_stream_seeds(spec.context_seed,spec.case,0,0,1)
 require(parent['seed_streams']==seeds and parent['proposal_buffer_policy']==frozen.proposal_buffer_policy(1200),'original private stream roles/seeds/FIFO policy differs')
 require(parent['proposal_buffer'].get('original_support_check_code_and_closure_verified') is True
  and parent['proposal_buffer'].get('original_support_source_sha256')==producer.physical_fifo.ORIGINAL_INFERENCE_SHA256,'actual original physical support closure/source witness required')
 frozen.validate_proposal_buffer_record(parent['proposal_buffer'],world_size=1,rank=0,scope=producer.proposal_scope(spec,frozen,actual['raw'],parent['model_inputs'],seeds))
 counters=parent['sampling_counters'];heads=8*counters['stage0_shared_predictions']+counters['component_predictions']
 require(cursor==len(calls)==execution['actual_head_contexts']==heads==parent['expected_probability_translation_count']
  ==parent['probability_adapter']['translation_call_count']==parent['actual_prediction_checks']['actual_head_contexts']
  and counters['stage0_shared_predictions']==execution['shared_densities']==waves,'actual physical head/PolicyC/shared-wave conservation differs')
 return {'passed':True,'stage_count':21*waves,'proposal_waves':waves,'actual_head_contexts':heads,
  'actual_training_masks_recomputed':True,'actual_query_policy_shape_dtype_routes_verified':True,
  'native_all8_stages':list(backend.NATIVE_STAGES),'mode':producer.MODE,'original_private_RNG_roles_and_F64_FIFO_validated':True}


def load_parent(path,expected_sha,*,scratch,spec,execution_screen,pair_cpu_receipt):
 started=time.perf_counter();producer,pairs,pins=source_guard();source_seconds=time.perf_counter()-started
 require(expected_sha not in INELIGIBLE_PARENT_SHA256,'failed earlier parent cannot be recovered or promoted')
 path=Path(path);require(sha256_file(path)==expected_sha,'explicit v3 parent manifest SHA differs')
 d=json.loads(path.read_bytes());proof_sha=producer.qualification_sha256(spec);header=validate_header(d,spec,proof_sha)
 require(sha256_file(pair_cpu_receipt)==PAIR_CPU_RECEIPT_SHA256,'original6f35 CPU receipt differs')
 metadata=kernel.metadata()
 require(d['coordinate_policy']==producer.backend.cached.policy_metadata() and d['native_execution_policy']==producer.backend.policy_metadata()
  and d['stage_to_physical_order']==list(codec.ORDER) and d['physical_to_stage_order']==list(codec.INVERSE_ORDER)
  and d['parameter_order']==metadata['physical_parameter_order'] and d['scientific_fingerprint']==metadata['scientific_fingerprint'], 'original physical science/order or v3 routing differs')
 old=load('refresh_prior_coordinates_global_block_v1.py',BASE.OLD_WRAPPER_SHA256);environment=old.require_replay_environment(d)
 payload=_safe_sibling(path,d['samples_file']);require(sha256_file(payload)==d['samples_sha256'],'physical samples payload hash differs')
 with np.load(payload,allow_pickle=False) as archive:
  require(archive.files==['samples'],'strict physical sample key required');values=archive['samples']
 from posteriorforge.inference import support_mask
 from posteriorforge.science.analytical_benchmark_v2 import build_problem
 require(values.dtype==np.float64 and values.shape==(COUNT,21) and np.isfinite(values).all()
  and array_descriptor(values)==d['samples'] and support_mask(values).all() and build_problem().contains(values).all(),'exactly50000 complete supported physical F64 vectors required')
 matrices={}
 for key in ('latent_wave_archive','decoded_wave_archive'):
  info=d[key];p=_safe_sibling(path,info['file']);require(sha256_file(p)==info['sha256'],'full proposal archive hash differs')
  with np.load(p,allow_pickle=False) as archive:matrices[key]={name:archive[name] for name in archive.files}
 decoded=replay_generated_decode(d,matrices['latent_wave_archive'],values,matrices['decoded_wave_archive'])
 results=Path(scratch)/'results';banks=results/'banks-v1';context,rawquery,raw=pairs.raw_inputs(spec,banks=banks)
 theta,obs,query,model,encoding=producer.model_bindings(context,rawquery)
 require(not BASE.policy_differences(raw,d['raw_scientific_bindings']) and not BASE.policy_differences(model,d['model_inputs']),'actual pair raw bank/query/model inputs differ')
 for key in ('physical_theta_original_Float64','latent_physical_slot_model_Float32','ordered_latent_model_Float32','stage_to_physical_order','physical_to_stage_order'):
  require(encoding[key]==d['model_target_descriptors'][key],'strict original training target differs: '+key)
 current=producer.execution_metadata(spec,proof_sha);policy=verify_canonical_policy(d['execution_policy'],current)
 require(d.get('physical_proposal_contract')==producer.proposals.metadata() and d.get('probability_arithmetic')==producer.probability.metadata(),'active v3 physical/probability contract differs')
 expected_files=dict(current['runtime_source_inventory']['sources']);expected_files['src/posteriorforge/performance_prior_pair_v3.py']=PRODUCER_SHA256
 recorded=BASE.verify_git_sources(Path(__file__).resolve().parents[1],d['git_revision'],expected_files)
 owner=pairs.prefix_parent(spec,completion_receipt=results/'accuracy-ss15-v1/sigma-native-grid-completion-01/completion.json')
 _,screen_receipt=producer.qualification_receipt(execution_screen,spec=spec,banks=banks,pair_cpu_receipt=pair_cpu_receipt,raw_bindings=raw,model_inputs=model,
  codec_inventory=results/'accuracy-prior-coordinates-v1/codec-inventory-inference-01/manifest.json',codec_reference_inventory=results/'accuracy-prior-coordinates-v1/codec-inventory-reference-01/manifest.json',
  codec_combined_inventory=Path(__file__).resolve().parents[1]/'evidence/accuracy-prior-coordinates-v1/codec-inventory-01/CPU-receipt.json',completed_grid=results/'accuracy-ss15-v1/sigma-native-grid-completion-01/completion.json',
  prefix_run=Path(owner['manifest_path']).parent,failed_screen=results/'accuracy-prior-coordinates-v1/conditionals-01/manifest.json')
 helper=producer.load_script(producer.QUALIFIER_FILENAME,PAIR_QUALIFIER_SHA256)
 execution=execution_checks(d,producer,{'theta':theta,'obs':obs,'raw':raw},helper,spec)
 require(d['expected_stage_prepare_count']==d['execution_counters']['prepare_count']==21*d['generated_proposal_wave_count'],'actual all-wave preparation count differs')
 require(sha256_file(path)==expected_sha and sha256_file(execution_screen)==proof_sha and sha256_file(pair_cpu_receipt)==PAIR_CPU_RECEIPT_SHA256,'parent/proof/CPU evidence changed during validation')
 return d,values,{'parent_manifest_sha256':expected_sha,'samples_sha256':d['samples_sha256'],'pair_spec':spec.as_dict(),'header':header,
  'actual_all_generated_rows_decode_replay':decoded,'actual_raw_model_mask_target_inputs_recomputed':True,'fresh_same_pair_screen_replay':screen_receipt,
  'actual_execution_checks':execution,'exact_replay_environment':environment,'source_pins':pins,'recorded_execution_source_inventory':recorded,
  'canonical_execution_policy_authentication':policy,'CPU_wall_seconds':{'explicit_initial_source_validation':source_seconds,'total_parent_validation':time.perf_counter()-started},
  'CPU_timing_scope':'wall time; total includes source/proof/data/finite-decode checks; source subset is not additive','new_model_GPU_or_reference_integrations':0,'new_global_draws_during_validation':0}

_PUBLIC_SEALS=tuple((name,globals()[name],globals()[name].__code__) for name in ('source_sha256','source_guard','validate_header','verify_canonical_policy','replay_generated_decode','execution_checks','load_parent','refresh_with_timing','refresh','_safe_sibling','derivation_descriptor'))
_CONFIG_SEALS={name:globals()[name] for name in ('SCHEMA','PRODUCER_SCHEMA','PRODUCER_VARIANT','PRODUCER_SHA256','PAIR_SOURCE_SHA256','PAIR_CPU_RECEIPT_SHA256','PAIR_QUALIFIER_SHA256','NATIVE_BACKEND_SHA256','NEURAL_SOURCE_SHA256','PROBABILITY_SHA256','PROPOSAL_SHA256','PROTOCOL_SHA256','PROTOCOL_PATH','PROOF_SHA256','INELIGIBLE_PARENT_SHA256','V1_WRAPPER_SHA256','BASE_WRAPPER_SHA256','DRAW_SOURCE_SHA256','COUNT','ROLE','kernel','codec','DRAW','BASE','_KERNEL_REFRESH','_RECORDING_GENERATOR')}
if __name__=='__main__':main()
