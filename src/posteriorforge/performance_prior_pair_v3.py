"""Unqualified v3: frozen FP32 forward, Policy-C FP64 P, finite F64 proposals.

Built with PriorLabs-TabPFN. Old modules/evidence remain immutable. The original
private RNG/stage-buffer/FIFO/support algorithms are retained. No old numerical
pass activates v3; a fresh same-pair all21 proof must be registered first.
"""
from __future__ import annotations
import argparse
import ast
import copy
from contextlib import contextmanager,ExitStack
import hashlib
import inspect
import json
from pathlib import Path
import re
from types import FunctionType,SimpleNamespace
import numpy as np
from . import performance_prior_pair_v2 as v2
from . import execution_prior_native_v3 as backend
from . import execution_probability_v3 as probability
from . import prior_proposals_v3 as proposals

ancestor=v2.ancestor
pairs=v2.pairs
codec=v2.codec  # TRAINING only: exact frozen d8 object and bytes.
model_bindings=v2.model_bindings
physical_fifo=v2.physical_fifo
require=v2.require
sha256_file=v2.sha256_file
load_script=v2.load_script
MODE=v2.MODE
COUNT=50000
SCHEMA='posteriorforge.inference.prior_coordinate_pair.v3'
VARIANT='prior_coordinates_F32_forward_FP64_P_finite_F64_proposals_v3'
V2_PRODUCER_SHA256='311f2cbfb3c3bd977ff9af348cd81a008c3c7711ec70f850fb25b3bc2d120a85'
PAIR_CPU_RECEIPT_SHA256=v2.PAIR_CPU_RECEIPT_SHA256
QUALIFIER_FILENAME='qualify_prior_coordinate_pair_v3.py'
QUALIFICATION_SCHEMA='posteriorforge.prior_coordinate_pair_conditionals.v3'
QUALIFICATION_STATUS='prior_coordinate_pair_v3_screen_pass'
PREREGISTRATION_PATH='docs/accuracy-prior-coordinate-fp64-proposals-v3.md'
# Frozen CPU sources/protocol; no scientific qualification is registered.
DEPENDENCY_SHA256={
 'src/posteriorforge/execution_probability_v3.py':'7fea13270c38c51eed0eb65cd5026de94e3f7e24cc4734e8906e882fb0bc0494',
 'src/posteriorforge/execution_prior_native_v3.py':'7c7180e3f161f9fc34684fc492f9ab558660b3b41092b70feea17edbbbf3f19c',
 'src/posteriorforge/prior_proposals_v3.py':'9216e76ce335ee52ffb1a5e8fbdd33c72f8c0aa84c4f1803dd90ff06b3e3a6d7',
 'scripts/qualify_prior_coordinate_pair_v3.py':'5066e257cebaa84c7dbc66f87976818934c24941c2287a03e6582cf735baac8e',
 PREREGISTRATION_PATH:'7ea6e5c57f8404eb761e07f4011b14966d5885fbbdb8620d463c5283916041d8',
}
QUALIFICATION_SHA256_BY_PAIR={}  # no v3 neural or scientific qualification yet.

def source_sha256():return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
def _sha(value,label):
    require(type(value) is str and re.fullmatch('[0-9a-f]{64}',value) is not None,label+' must be a pinned SHA256')
    return value

def _spec(value):
    require(type(value) is pairs.PairSpec,'explicit immutable PairSpec required')
    return value

def _clone(function,namespace):
    out=FunctionType(function.__code__,namespace,function.__name__,function.__defaults__,function.__closure__)
    out.__kwdefaults__=function.__kwdefaults__;out.__annotations__=dict(function.__annotations__)
    return out

def _same_namespace(namespace,seals,label):
    require(set(namespace)==set(seals),label+' fields changed')
    for key,value in seals.items():require(namespace[key] is value,label+' binding changed: '+key)

def runtime_source_inventory(*,require_committed):
    inherited=v2.runtime_source_inventory(require_committed=require_committed)
    sources={**inherited['sources'],'src/posteriorforge/performance_prior_pair_v2.py':V2_PRODUCER_SHA256}
    sources.update({k:_sha(v,'v3 dependency '+k) for k,v in DEPENDENCY_SHA256.items()})
    root=Path(__file__).resolve().parents[2]
    for name,digest in sources.items():require(sha256_file(root/name)==digest,'v3 runtime source changed: '+name)
    if require_committed:
        require(ancestor._committed_source_hashes(root,list(sources))==sources,'v3 dependencies must match committed HEAD')
        require(hashlib.sha256(ancestor._git_bytes(root,'src/posteriorforge/performance_prior_pair_v3.py')).hexdigest()==source_sha256(),'exact v3 producer must be committed')
    return {'schema':'posteriorforge.prior_coordinate_pair_runtime_sources.v3','sources':sources,'source_count':len(sources),
        'inventory_sha256':hashlib.sha256(json.dumps(sources,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        'current_checkout_exact':True,'committed_HEAD_exact':bool(require_committed),
        'own_producer_source_sha256':source_sha256(),'own_source_checked_separately_to_avoid_recursive_hash':True}

def _configuration():
    return (SCHEMA,VARIANT,COUNT,MODE,tuple(sorted(DEPENDENCY_SHA256.items())),tuple(sorted(QUALIFICATION_SHA256_BY_PAIR.items())))

def _integrity(*,require_committed):
    require(_configuration()==_CONFIGURATION,'v3 configuration changed after import')
    require(source_sha256()==_IMPORTED_SHA,'v3 producer source changed after import')
    require(v2.source_sha256()==V2_PRODUCER_SHA256,'frozen v2 producer changed');v2.require_sources()
    backend.require_sources();probability.require_sources();proposals.require_sources();physical_fifo.require_sources()
    require(model_bindings is v2.model_bindings and codec is v2.codec,'strict training/model bindings changed')
    for name,fn,code in _FUNCTIONS:
        require(globals().get(name) is fn and fn.__code__ is code,'v3 producer callable changed: '+name)
    for name,value in _OBJECTS.items():require(globals().get(name) is value,'v3 dependency object changed: '+name)
    for name,fn,code in _ADAPTER_METHODS:
        current=getattr(StageAdapter,name);current=current.fget if isinstance(current,property) else current
        require(current is fn and fn.__code__ is code,'v3 adapter prototype changed: '+name)
    require(_RUN_TEMPLATE.__code__ is _RUN_CODE and _SAMPLER.__code__ is _SAMPLER_CODE,'v3 private source transform changed')
    _same_namespace(vars(_CLEAN_PROXY),_CLEAN_SEALS,'v3 sampling proxy')
    _same_namespace(_SAMPLER.__globals__,_SAMPLER_GLOBAL_SEALS,'v3 sampler globals')
    _same_namespace(_RUN_TEMPLATE.__globals__,_RUN_GLOBAL_SEALS,'v3 run template globals')
    require(_SAMPLER_DESCRIPTOR==_SAMPLER_DESCRIPTOR_VALUE,'v3 sampler derivation changed')
    require(engine_scope.__wrapped__ is _SCOPE_GENERATOR and _SCOPE_GENERATOR.__code__ is _SCOPE_CODE,
            'v3 context generator changed')
    runtime_source_inventory(require_committed=require_committed)
    return True

def require_sources():return _integrity(require_committed=True)

def qualification_sha256(spec):
    spec=_spec(spec)
    return _sha(QUALIFICATION_SHA256_BY_PAIR.get((spec.context_seed,spec.case)),'fresh own-pair v3 all21 qualification')

def execution_metadata(spec,proof_sha256):
    spec=_spec(spec);_sha(proof_sha256,'v3 all21 proof');require_sources()
    return {'schema':'posteriorforge.prior_coordinate_pair_execution.v3','module_sha256':source_sha256(),
        'pair_spec':spec.as_dict(),'same_pair_qualification_sha256':proof_sha256,
        'same_pair_CPU_panel_receipt_sha256':PAIR_CPU_RECEIPT_SHA256,
        'runtime_source_inventory':runtime_source_inventory(require_committed=True),
        'neural_policy':backend.policy_metadata(),'probability_policy':probability.metadata(),
        'physical_proposal_contract':proposals.metadata(),'strict_training_encoder':codec.metadata(),
        'physical_FIFO_source_sha256':physical_fifo.source_sha256(),'neural_weights_or_training_representation_changed':False,
        'probability_arithmetic_and_output_precision_contract_changed':True,'inherited_v1_or_v2_numerical_qualification':False,
        'scientific_qualification_count_at_preregistration':0,'native_all8_stages':[6,10,14],'cached_stages':18,
        'forward_dtype':'torch.float32','probability_dtype':'torch.float64','stage_storage_dtype':'numpy.float32',
        'physical_proposal_FIFO_output_dtype':'Float64','FP16_activated':False,'RNG_roles_or_assignment_changed':False,
        'original_physical_support_and_scientific_gates_changed':False,'derivation':derivation_descriptor(),
        'preregistration_sha256':DEPENDENCY_SHA256[PREREGISTRATION_PATH],'attribution':'Built with PriorLabs-TabPFN'}

def proposal_scope(spec,frozen,raw,model,seeds):
    scope=v2.proposal_scope(_spec(spec),frozen,raw,model,seeds)
    scope.update(prior_coordinate_producer_sha256=source_sha256(),
        backend_sha256=backend.source_sha256(),probability_source_sha256=probability.source_sha256(),
        physical_proposal_decoder_sha256=proposals.source_sha256(),
        strict_training_codec_sha256=codec.source_sha256(),
        strict_full_wave_decode_before_FIFO_and_support=False,
        finite_full_wave_physical_proposals_before_original_FIFO_support=True,
        numeric_output_contract=proposals.SCHEMA,post_decode_fifo_dtype='Float64 physical proposals; no cast')
    return scope

def qualification_receipt(path,*,spec,banks,pair_cpu_receipt,raw_bindings,model_inputs,
        codec_inventory,codec_reference_inventory,codec_combined_inventory,completed_grid,prefix_run,failed_screen):
    spec=_spec(spec);proof=qualification_sha256(spec)
    require(sha256_file(path)==proof,'fresh v3 own-pair proof SHA differs')
    require(sha256_file(pair_cpu_receipt)==PAIR_CPU_RECEIPT_SHA256,'unchanged6f35 CPU panel required')
    helper=load_script(QUALIFIER_FILENAME,_sha(DEPENDENCY_SHA256['scripts/'+QUALIFIER_FILENAME],'v3 qualifier'))
    document,receipt=helper.qualification_receipt(spec,path,banks=banks,completion_receipt=completed_grid,
        training_inventory=codec_inventory,reference_training_inventory=codec_reference_inventory,
        combined_training_inventory=codec_combined_inventory,pair_cpu_receipt=pair_cpu_receipt,
        raw_bindings=raw_bindings,model_inputs=model_inputs,prefix_run=prefix_run,failed_screen=failed_screen)
    require(document.get('schema')==QUALIFICATION_SCHEMA and document.get('status')==QUALIFICATION_STATUS
        and document.get('matching_v3_execution_passed_all21') is True,'fresh successful v3 all21 proof required; no old-schema credit')
    require(pairs.PairSpec.from_dict(document['pair'])==spec and receipt.get('replay',{}).get('passed') is True,'v3 proof pair/replay differs')
    return document,receipt

class StageAdapter:
    """Actual0199 forward witnesses and separate FP64 probability call journal."""
    def __init__(self,checkpoint,*,cdf_descriptor):
        self.checkpoint=checkpoint;self.cdf_descriptor=cdf_descriptor;self.engine=self.session=None
        self.stage_index=None;self.actual_heads=self.shared_densities=self.prepare_count=0
        self.stages=[];self.training_inputs=[];self.query_calls=[];self.failure=False
    @property
    def regressor(self):return self.engine.regressor if self.engine is not None else None
    def prepare(self,theta,obs,stage):
        require(self.session is None,'previous v3 stage not closed');self.stage_index=stage;self.failure=False;self.stage_heads=0
        self.prefill=self.model_at_prepare=self.native_before=None
        self.session=backend.PriorNativeSession(probability_descriptor=self.cdf_descriptor)
        try:
            self.session.__enter__()
            self.engine=backend.PriorNativeStageExecutor(self.checkpoint,cdf_descriptor=self.cdf_descriptor,query_batch_size=1024)
            entry=backend.training_bindings(theta,obs,stage);mt,ms=backend.mask_training(theta,obs,stage)
            result=self.engine.prepare(mt,ms,stage);self.prepare_count+=1
            self.prefill=backend.preparation_witness(self.session.report(),stage,result);require(self.prefill['passed'],'FP32 preparation witness failed')
            base=load_script('characterize_precision_approximation.py',ancestor.failed.ancestry.BASE_SHA)
            self.model_at_prepare=base.model_summary(self.regressor,load_script('qualify_execution_cache.py',base.UTILITY_SHA))
            self.native_before=backend.neural.immutable_native_snapshot(self.engine.inner) if stage in backend.NATIVE_STAGES else None
            self.training_inputs.append(entry);return result
        except BaseException:
            self.failure=True;self.close();raise
    def predict_component(self,member,query,*,capture_member=False,with_log_prob=False):
        require(self.engine is not None,'prepared v3 stage required')
        from .science.analytical_benchmark_v2 import array_descriptor
        masked=backend.mask_query(query,self.stage_index);before=self.session.report()
        count=self.cdf_descriptor['translation_call_count']
        try:
            p=self.engine.predict_component(member,masked,capture_member=capture_member,with_log_prob=with_log_prob)
            witness=backend.query_witness_delta(before,self.session.report(),self.stage_index,member)
            require(witness['passed'],'actual FP32 neural query witness failed')
            import torch
            require(p['logits'].dtype==torch.float64 and p['logits'].device.type=='cpu'
                and p['criterion'].borders.dtype==torch.float64 and p['component_raw_mass'].dtype==torch.float64,'v3 actual probability dtype differs')
            proof=p['neural_v3'];require(proof['module_sha256']==backend.source_sha256()
                and proof['neural_source_sha256']==backend.NEURAL_SHA and proof['stage']==self.stage_index
                and proof['member']==member and proof['actual_adjusted_dtype']=='torch.float32','actual v3 NN-prefix lineage differs')
            require(self.cdf_descriptor['translation_call_count']-count==1,'one actual PolicyC translation required')
            self.actual_heads+=1;self.stage_heads+=1
            self.query_calls.append({'stage':self.stage_index,'member':member,
                'query_unmasked':array_descriptor(np.asarray(query)),'query_masked':array_descriptor(masked),
                'mask':backend.stage_policy(self.stage_index),'neural_prefix':copy.deepcopy(proof),
                'execution_route':backend.route(self.stage_index),'witness':witness,
                'probability_policy_source_sha256':probability.source_sha256()})
            p['prior_coordinate_execution_proof']={'module_sha256':source_sha256(),'backend_source_sha256':backend.source_sha256(),
                'probability_source_sha256':probability.source_sha256(),'proposal_decoder_sha256':proposals.source_sha256(),
                'mode':MODE,'stage':self.stage_index,'member':member,'actual_head_contexts':1,'probability_translation_calls':1,'witness':witness}
            return p
        except BaseException:self.failure=True;raise
    def predict(self,query,*,output_type='full',capture_members=False):
        require(self.stage_index==0 and len(query)==1 and output_type=='full','only wave-local stage0 shared density allowed')
        p=probability.assemble_shared_raw64([self.predict_component(i,query,capture_member=True) for i in range(8)],
            descriptor=self.cdf_descriptor,stage0=True)
        self.shared_densities+=1
        p['prior_coordinate_execution_proof']={'module_sha256':source_sha256(),'backend_source_sha256':backend.source_sha256(),
            'probability_source_sha256':probability.source_sha256(),'proposal_decoder_sha256':proposals.source_sha256(),
            'mode':MODE,'stage':0,'member':None,'actual_head_contexts':8,'probability_translation_calls':8}
        return p
    def inspect_cache(self):return self.engine.inspect_cache()
    def close(self):
        if self.session is None:return
        session,engine=self.session,self.engine;failure=None;after=None;counters={}
        try:
            if engine is not None:
                counters=copy.deepcopy(engine.counters)
                if not self.failure and self.stage_index in backend.NATIVE_STAGES:
                    after=backend.neural.immutable_native_snapshot(engine.inner)
                    require(after==self.native_before,'all8 native immutable state changed')
                require(self.failure or counters['prefix_query_count']==self.stage_heads,'actual v3 prefix/head count differs')
        except BaseException as e:failure=e
        finally:
            try:
                if engine is not None:engine.close()
            finally:
                try:session.__exit__(None,None,None)
                finally:
                    report=session.report();self.stages.append({'stage_index':self.stage_index,
                        'actual_head_contexts':getattr(self,'stage_heads',0),'failed':bool(self.failure or failure),
                        'preparation_witness':getattr(self,'prefill',None),'model_at_prepare':getattr(self,'model_at_prepare',None),
                        'immutable_native_before':getattr(self,'native_before',None),'immutable_native_after':after,
                        'v3_prefix_counters':counters,'precision_session':report})
                    self.engine=self.session=None
        require(report.get('aliases_and_flags_restored') is True,'actual neural aliases/flags not restored')
        if failure is not None:raise failure
    def metadata(self):
        return {'policy':execution_metadata(),'prepare_count':self.prepare_count,'actual_head_contexts':self.actual_heads,
            'shared_densities':self.shared_densities,'completed_stage_scopes':self.stages,
            'actual_latent_training_inputs':self.training_inputs,'actual_latent_query_calls':self.query_calls,
            'all_stage_aliases_restored':all(s['precision_session'].get('aliases_and_flags_restored') is True for s in self.stages)}

@contextmanager
def engine_scope(checkpoint,*,cdf_descriptor):
    with ExitStack() as stack:
        engine=StageAdapter(checkpoint,cdf_descriptor=cdf_descriptor);stack.callback(engine.close);yield engine

def validate_prediction(prediction,stage,member,*,report):
    p=prediction.get('prior_coordinate_execution_proof',{});n=8 if member is None else 1
    require((member is None)==(stage==0) and p.get('module_sha256')==source_sha256()
        and p.get('backend_source_sha256')==backend.source_sha256()
        and p.get('probability_source_sha256')==probability.source_sha256()
        and p.get('proposal_decoder_sha256')==proposals.source_sha256()
        and p.get('stage')==stage and p.get('member')==member and p.get('mode')==MODE,'actual v3 prediction provenance differs')
    require(p.get('actual_head_contexts')==p.get('probability_translation_calls')==n,'actual v3 head/P counts differ')
    if member is not None:require(p.get('witness',{}).get('passed') is True,'actual FP32 neural witness missing')
    report['sampler_callback_calls']=report.get('sampler_callback_calls',0)+1
    report['actual_head_contexts']=report.get('actual_head_contexts',0)+n
    report['shared_densities']=report.get('shared_densities',0)+int(member is None)

def decode_complete_wave(values):
    return proposals.decode_ordered_proposals(values)

def record_decode_wave(latent,latent_waves,status,*,decoded_waves=None):
    from .science.analytical_benchmark_v2 import array_descriptor
    index=len(latent_waves);latent_waves.append(latent.copy(order='C'))
    r={'wave_index':index,'row_count':len(latent),'actual_latent_draws':array_descriptor(latent),
       'input_dtype_unchanged_before_decode':True,'includes_unused_FIFO_tail':True,'finite_proposal_decode_passed':False}
    status['decode_wave_records'].append(r);status['actual_complete_generated_wave_count']=len(latent_waves)
    try:physical=decode_complete_wave(latent)
    except BaseException as e:
        r.update(decode_error_type=type(e).__name__,decode_error=str(e));status['failed_decode_wave_index']=index;raise
    require(physical.shape==latent.shape and physical.dtype==np.float64 and np.isfinite(physical).all(),'finite complete physical proposal wave required')
    r.update(decoded_physical=array_descriptor(physical),finite_proposal_decode_passed=True,
             physical_proposal_contract_sha256=proposals.source_sha256(),support_selection_not_applied_by_decoder=True)
    if decoded_waves is not None:decoded_waves.append((index,physical.copy(order='C')))
    return physical

def archive_decoded_waves(output,decoded_waves):
    r=ancestor.failed.archive_decoded_waves(output,decoded_waves)
    r['all_finite_precision_valid_proposals_preserved']=r.pop('only_valid_decoded_complete_waves')
    r['generated_proposal_row_count']=r.pop('generated_valid_row_count')
    r['original_support_selection_not_applied_to_archive']=True
    return r

def _preserve_failure(*args,**kwargs):
    ns=dict(ancestor.failed.preserve_failure_artifacts.__globals__);ns['archive_decoded_waves']=archive_decoded_waves
    return _clone(ancestor.failed.preserve_failure_artifacts,ns)(*args,**kwargs)

def _build_sampler():
    original=ancestor.physical_sampling_namespace().component_stage_sampler
    tree=ast.parse(inspect.getsource(original));changed=copy.deepcopy(tree)
    nodes=[n for n in ast.walk(changed) if isinstance(n,ast.ImportFrom) and n.module=='execution_mixture']
    require(len(nodes)==1 and [x.name for x in nodes[0].names]==['finite_bar_icdf','head_acceptance'],'canonical sampler import changed')
    nodes[0].module='execution_probability_v3'
    reverse=copy.deepcopy(changed)
    for n in ast.walk(reverse):
        if isinstance(n,ast.ImportFrom) and n.module=='execution_probability_v3':n.module='execution_mixture'
    require(ast.dump(reverse,include_attributes=False)==ast.dump(tree,include_attributes=False),'undeclared sampler/RNG/stage-buffer change')
    ns=dict(original.__globals__);exec(compile(ast.fix_missing_locations(changed),'<v3-sampler-inverse-binding-only>','exec'),ns)
    return ns['component_stage_sampler'],{'reverse_AST_exact':True,'only_import_module_changed':True,
        'all_RNG_calls_seed_roles_uniform_assignments_head_retry_and_F32_draw_buffer_unchanged':True,
        'head_acceptance_callable_is_exact_e71':probability.head_acceptance is probability.legacy.head_acceptance}

def physical_sampling_namespace():
    original=ancestor.physical_sampling_namespace();ns={k:v for k,v in vars(original).items() if not k.startswith('__')}
    ns['component_stage_sampler']=_SAMPLER
    return SimpleNamespace(**ns)

def _build_run():
    original=ast.parse(inspect.getsource(ancestor.failed.run));candidate=copy.deepcopy(original)
    # Reviewed a30 original-F64 completion and failure-footer deltas.
    blocks=[]
    for n in ast.walk(candidate):
        for _,value in ast.iter_fields(n):
            if isinstance(value,list):
                for i,item in enumerate(value):
                    if isinstance(item,ast.With) and len(item.items)==1 and ast.unparse(item.items[0].context_expr)=='torch.no_grad()':blocks.append((value,i))
    require(len(blocks)==1,'one original support block required');body,i=blocks[0];saved=copy.deepcopy(body[i:i+4])
    require(ast.unparse(saved[1]).startswith('record = buffer.finalize'),'original finalization block changed')
    delegated=ast.parse('with torch.no_grad():\n samples, record = physical_fifo.complete_request(proposal, scope=scope, prior=prior, buffer=buffer, accept_reject_fn=accept_reject)').body[0]
    body[i:i+4]=[delegated]
    handler=[h for n in ast.walk(candidate) if isinstance(n,ast.Try) for h in n.handlers if isinstance(h.type,ast.Name) and h.type.id=='BaseException' and h.name=='error']
    require(len(handler)==1,'one original outer failure handler required')
    handler[0].body.insert(-1,ast.parse('_safe_capture_failure_footer(status, locals())').body[0])
    imports=[n for n in ast.walk(candidate) if isinstance(n,ast.ImportFrom) and n.module=='compat_cdf']
    require(len(imports)==1,'one original CDF context import required')
    imports[0].module='execution_probability_v3';imports[0].names=[ast.alias(name='production_scope',asname='finite_cdf_compatibility')]
    assign=[n for n in ast.walk(candidate) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='status' for t in n.targets) and isinstance(n.value,ast.Dict)]
    require(len(assign)==1,'one original status declaration required');old_status=copy.deepcopy(assign[0].value)
    for key,value in zip(assign[0].value.keys,assign[0].value.values):
        if not isinstance(key,ast.Constant):continue
        if key.value=='codec':key.value='strict_training_encoder'
        elif key.value=='all_generated_rows_strictly_decoded_including_unused_FIFO_tail':key.value='all_generated_rows_finite_physical_proposals_including_unused_FIFO_tail'
        elif key.value=='decode_failure_policy':value.value='FAIL on genuine nonfinite/arithmetic/physical-precision error; retain finite endpoint/outside proposals; original support owns rejection'
        elif key.value=='sampler_math_changed':value.value=True
        elif key.value=='neural_execution_description':value.value='F32 forward native6/10/14/cache18; PolicyC CPU F64 probability; F32 stage storage; finite F64 physical proposals'
        elif key.value=='fallback_map_scope':value.value='unchanged0199 F32 neural session; probability arithmetic and output contract are separately v3'
    mapping={'patched_call_count':'translation_call_count','compatibility_adapter':'probability_adapter','includes_strict_all_generated_wave_decode':'includes_all_finite_proposal_wave_decode'}
    for n in ast.walk(candidate):
        if isinstance(n,ast.Constant) and isinstance(n.value,str) and n.value in mapping:n.value=mapping[n.value]
        elif isinstance(n,ast.keyword) and n.arg in ('expected_patched_call_count','compatibility_adapter','includes_strict_all_generated_wave_decode'):
            n.arg={'expected_patched_call_count':'expected_probability_translation_count','compatibility_adapter':'probability_adapter','includes_strict_all_generated_wave_decode':'includes_all_finite_proposal_wave_decode'}[n.arg]
    guard_old=ast.parse("descriptor['original_alias_restored'] and descriptor['sdk_sources_unchanged']",mode='eval').body
    guards=[n for n in ast.walk(candidate) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='require' and n.args and ast.dump(n.args[0],include_attributes=False)==ast.dump(guard_old,include_attributes=False)]
    require(len(guards)==1,'one original CDF restore check required')
    guard_new=ast.parse("descriptor['closed'] and descriptor['source_unchanged'] and all(g['original_alias_restored'] and g['sdk_sources_unchanged'] for g in descriptor['legacy_preparation_guards'])",mode='eval').body
    guards[0].args[0]=guard_new
    # Reverse every declared v3/a30 change; all scientific/RNG/control flow stays original.
    reverse=copy.deepcopy(candidate)
    reverse_mapping={v:k for k,v in mapping.items()}
    for n in ast.walk(reverse):
        if isinstance(n,ast.Constant) and isinstance(n.value,str) and n.value in reverse_mapping:n.value=reverse_mapping[n.value]
        elif isinstance(n,ast.keyword) and n.arg in ('expected_probability_translation_count','probability_adapter','includes_all_finite_proposal_wave_decode'):
            n.arg={'expected_probability_translation_count':'expected_patched_call_count','probability_adapter':'compatibility_adapter','includes_all_finite_proposal_wave_decode':'includes_strict_all_generated_wave_decode'}[n.arg]
        elif isinstance(n,ast.ImportFrom) and n.module=='execution_probability_v3':n.module='compat_cdf';n.names=[ast.alias(name='finite_cdf_compatibility')]
    for n in ast.walk(reverse):
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='status' for t in n.targets) and isinstance(n.value,ast.Dict):n.value=copy.deepcopy(old_status)
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='require' and n.args and ast.dump(n.args[0],include_attributes=False)==ast.dump(guard_new,include_attributes=False):n.args[0]=copy.deepcopy(guard_old)
        if isinstance(n,ast.ExceptHandler):n.body=[x for x in n.body if not (isinstance(x,ast.Expr) and isinstance(x.value,ast.Call) and isinstance(x.value.func,ast.Name) and x.value.func.id=='_safe_capture_failure_footer')]
    for n in ast.walk(reverse):
        for _,value in ast.iter_fields(n):
            if isinstance(value,list):
                for j,item in enumerate(value):
                    if isinstance(item,ast.With) and ast.dump(item,include_attributes=False)==ast.dump(delegated,include_attributes=False):value[j:j+1]=copy.deepcopy(saved);break
    require(ast.dump(reverse,include_attributes=False)==ast.dump(original,include_attributes=False),'undeclared v3 run control-flow/math changes')
    ns=dict(ancestor._RUN_BODY.__globals__);ns.update(__name__=__name__,__package__=__package__)
    exec(compile(ast.fix_missing_locations(candidate),'<v3-source-derived-physical-proposal-run>','exec'),ns)
    return ns['run']

def derivation_descriptor():
    return {'schema':'posteriorforge.v3_source_derivation.v1','original71_source_sha256':ancestor.ORIGINAL_PRODUCER_SHA256,
        'reverse_run_AST_exact':True,'sampler':copy.deepcopy(_SAMPLER_DESCRIPTOR),
        'declared_run_changes':['same original support/F64FIFO completion delegation','actual failure footer',
            'v3 probability scope import/counters','truthful v3 status metadata and source-restoration check'],
        'numeric_global_bindings':['PolicyC F64 head adapter','finite physical proposal decoder','same original F64 FIFO/support'],
        'old_function_globals_mutated':False,'old_failed_wave_recovered':False}

def _adapter_bindings(spec,proof):
    ns=dict(globals());ns.update(execution_metadata=lambda:execution_metadata(spec,proof))
    attrs={'__module__':__name__}
    for name,value in vars(StageAdapter).items():
        if inspect.isfunction(value):attrs[name]=_clone(value,ns)
        elif isinstance(value,property):attrs[name]=property(_clone(value.fget,ns))
    cls=type('PairV3BoundStageAdapter',(),attrs);ns['StageAdapter']=cls
    return cls,contextmanager(_clone(engine_scope.__wrapped__,ns)),_clone(validate_prediction,ns),ns

def _bind_run(spec,proof,pair_cpu_receipt):
    spec=_spec(spec);_sha(proof,'own-pair v3 proof')
    cls,scope,checker,adapter_ns=_adapter_bindings(spec,proof)
    ns=dict(_RUN_TEMPLATE.__globals__)
    def screen(path,**kwargs):return qualification_receipt(path,spec=spec,pair_cpu_receipt=pair_cpu_receipt,**kwargs)
    def identity():
        return {**ancestor.failed.runtime_identity(),'pair_spec':spec.as_dict(),'pair_cpu_receipt_sha256':PAIR_CPU_RECEIPT_SHA256,
            'preregistration_sha256':DEPENDENCY_SHA256[PREREGISTRATION_PATH],
            'probability_arithmetic':probability.metadata(),'physical_proposal_contract':proposals.metadata(),
            'v3_qualification_inherited':False,'neural_weights_and_training_representation_changed':False}
    observer_ns=dict(ancestor._capture_failure_footer.__globals__);observer_ns.update(source_sha256=source_sha256,require_sources=require_sources)
    capture=_clone(ancestor._capture_failure_footer,observer_ns);observer_ns['_capture_failure_footer']=capture
    safe=_clone(ancestor._safe_capture_failure_footer,observer_ns)
    ns.update(SCHEMA=SCHEMA,VARIANT=VARIANT,CONTEXT_SEED=spec.context_seed,CASE=spec.case,MODE=MODE,
        clean=_CLEAN_PROXY,backend=backend,CODEC_SHA256=codec.source_sha256(),
        BACKEND_SHA256=backend.source_sha256(),codec=codec,model_bindings=model_bindings,
        physical_fifo=physical_fifo,source_sha256=source_sha256,require_sources=require_sources,
        execution_metadata=lambda:execution_metadata(spec,proof),
        proposal_scope=lambda frozen,raw,model,seeds:proposal_scope(spec,frozen,raw,model,seeds),
        qualification_receipt=screen,CONDITIONAL_QUALIFICATION_SHA256=proof,
        PREREGISTRATION_SHA256=DEPENDENCY_SHA256[PREREGISTRATION_PATH],NATIVE_PROTOCOL_SHA256=DEPENDENCY_SHA256[PREREGISTRATION_PATH],
        engine_scope=scope,validate_prediction=checker,runtime_identity=identity,
        record_decode_wave=record_decode_wave,archive_decoded_waves=archive_decoded_waves,
        preserve_failure_artifacts=_preserve_failure,_safe_capture_failure_footer=safe)
    def bound_sources():
        require_sources()
        _same_namespace(vars(cls),class_seals,'private v3 adapter class')
        _same_namespace(adapter_ns,adapter_seals,'private v3 adapter globals')
        _same_namespace(ns,bound_seals,'private v3 run globals')
        _same_namespace(observer_ns,observer_seals,'private v3 failure observer globals')
        for name,method,code in method_seals:
            require(method.__code__ is code and method.__globals__ is adapter_ns,
                    'private v3 adapter code/globals changed: '+name)
        require(scope.__code__ is scope_code and scope.__wrapped__ is scope_generator
                and scope_generator.__code__ is scope_generator_code
                and scope_generator.__globals__ is adapter_ns,'private v3 context changed')
        require(checker.__code__ is checker_code and checker.__globals__ is adapter_ns,
                'private v3 prediction checker changed')
        for function,code in observer_functions:
            require(function.__code__ is code and function.__globals__ is observer_ns,
                    'private v3 failure observer changed')
        require(function_run.__code__ is _RUN_CODE and function_run.__globals__ is ns,
                'private v3 run code/globals changed')
        return True
    ns['require_sources']=adapter_ns['require_sources']=observer_ns['require_sources']=bound_sources
    function_run=_clone(_RUN_TEMPLATE,ns)
    class_seals=dict(vars(cls));adapter_seals=dict(adapter_ns);bound_seals=dict(ns);observer_seals=dict(observer_ns)
    method_seals=[]
    for name,value in class_seals.items():
        methods=(value.fget,value.fset,value.fdel) if isinstance(value,property) else (value,)
        for method in methods:
            if inspect.isfunction(method):method_seals.append((name,method,method.__code__))
    scope_code,scope_generator=scope.__code__,scope.__wrapped__
    scope_generator_code,checker_code=scope_generator.__code__,checker.__code__
    observer_functions=[(f,f.__code__) for f in (capture,safe)]
    return function_run

def run(*,context_seed,case,pair_cpu_receipt,**paths):
    spec=pairs.PairSpec(context_seed,case);require_sources();proof=qualification_sha256(spec)
    require(sha256_file(pair_cpu_receipt)==PAIR_CPU_RECEIPT_SHA256,'unchanged6f35 CPU panel required')
    return _bind_run(spec,proof,pair_cpu_receipt)(**paths)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--context-seed',type=int,required=True);p.add_argument('--case',type=int,required=True)
    for name in ('banks','output','checkpoint','ledger','reference','reference-grid','qualification',
                 'codec-inventory','codec-reference-inventory','codec-combined-inventory',
                 'completed-grid','prefix-run','failed-screen','pair-cpu-receipt'):p.add_argument('--'+name,required=True)
    result=run(**vars(p.parse_args()));print(json.dumps({k:result[k] for k in ('schema','status','case','context_seed','samples_sha256')},sort_keys=True))

_SAMPLER,_SAMPLER_DESCRIPTOR=_build_sampler();_SAMPLER_CODE=_SAMPLER.__code__
_CLEAN_PROXY=SimpleNamespace(**{k:v for k,v in vars(ancestor._CLEAN_PROXY).items() if not k.startswith('__')})
_CLEAN_PROXY.frozen_sampling=physical_sampling_namespace
_RUN_TEMPLATE=_build_run();_RUN_CODE=_RUN_TEMPLATE.__code__
_CLEAN_SEALS=dict(vars(_CLEAN_PROXY));_SAMPLER_GLOBAL_SEALS=dict(_SAMPLER.__globals__)
_RUN_GLOBAL_SEALS=dict(_RUN_TEMPLATE.__globals__);_SAMPLER_DESCRIPTOR_VALUE=copy.deepcopy(_SAMPLER_DESCRIPTOR)
_SCOPE_GENERATOR=engine_scope.__wrapped__;_SCOPE_CODE=_SCOPE_GENERATOR.__code__
_ADAPTER_METHODS=[]
for _name,_value in vars(StageAdapter).items():
    _fn=_value.fget if isinstance(_value,property) else _value
    if inspect.isfunction(_fn):_ADAPTER_METHODS.append((_name,_fn,_fn.__code__))
_FUNCTIONS=tuple((name,value,value.__code__) for name,value in list(globals().items())
    if inspect.isfunction(value) and value.__module__==__name__)
_OBJECTS={name:globals()[name] for name in ('v2','ancestor','pairs','codec','backend','probability','proposals',
    'model_bindings','physical_fifo','_CLEAN_PROXY','_SAMPLER','_RUN_TEMPLATE',
    'require','sha256_file','load_script','ast','copy','contextmanager','ExitStack','hashlib',
    'inspect','json','Path','re','FunctionType','SimpleNamespace','np','PREREGISTRATION_PATH',
    'V2_PRODUCER_SHA256','PAIR_CPU_RECEIPT_SHA256','QUALIFIER_FILENAME','QUALIFICATION_SCHEMA','QUALIFICATION_STATUS')}
_CONFIGURATION=_configuration();_IMPORTED_SHA=source_sha256()
if __name__=='__main__':main()
