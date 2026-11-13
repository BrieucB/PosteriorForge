"""Two counterfactual FP32 conditional windows; no posterior sampling or repair.

Built with PriorLabs-TabPFN. CPU preflight is Torch-free. Root alone may submit
one <=300 GPU-second allocation after source review/commit. All density arrays
stay on Karolina; normalized reference analysis runs separately after capture.
"""
from __future__ import annotations
import argparse
import ast
import copy
from contextlib import ExitStack
import gc
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import pickle
import random
import subprocess
import textwrap
import time
from types import MethodType, SimpleNamespace
import numpy as np

from posteriorforge import performance_prior_pair_v2 as producer
from posteriorforge import execution_mixture as mixture
from posteriorforge.artifacts import write_json, versions
from posteriorforge.science.analytical_benchmark_v2 import array_descriptor

SCHEMA = 'posteriorforge.prior_tail_conditional_capture.v1'
FAILED_SHA = '92755b788fc894d4ca16157e573bd1b5c9d8a4468b71cf184fa16303bced7bce'
WAVE_SHA = '1ba89c1a4e53c5895ad1623b5a396e4086154c3dc830f8036da388f234d90349'
WAVE_ARRAY_SHA = 'e51e00f9e8ec686a09b2280a9f59c417d4f98c306843a20785ed7579c6a1ef7e'
UNIFORM_SHA = 'b6c3e08ce2897d5fead7d129dbbf3e2cae8cd37fa7986903899fb68471d8ac24'
MIXTURE_SHA = 'e71f43b73ae8259c20b5d7b52feccc7c851e9a2fbb86d869bc0a5f7f3af792a1'
FIXTURES = ((15, 4355, 4096, 5120, 'e0ff7f3f'), (19, 41765, 40960, 41984, 'f6ff7f3f'))
UNIFORM_BITS = (0x3f7d70a4, 0x3f7fbe77, 0x3f7ff972, 0x3f7fffe0, 0x3f7ffff6, 0x3f7fffff)
THRESHOLDS = {'normalization_absolute':1e-5, 'max_cdf_absolute':2e-5,
              'max_row_total_variation':1e-4, 'quantile_coordinate_span_fraction':2e-6}
_ICDF = mixture.finite_bar_icdf
_ICDF_CODE = _ICDF.__code__

def require(value, message):
    if not value:
        raise RuntimeError(message)

def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def source_pins(*, committed=False):
    inventory = producer.require_sources()
    require(mixture.execution_source_sha256() == MIXTURE_SHA and mixture.finite_bar_icdf is _ICDF
            and _ICDF.__code__ is _ICDF_CODE, 'frozen inverse source/callable changed')
    if committed:
        root = Path(__file__).resolve().parents[1]
        raw = subprocess.run(['git','-C',str(root),'show','HEAD:scripts/'+Path(__file__).name],
                             check=True,capture_output=True).stdout
        require(hashlib.sha256(raw).hexdigest() == sha_file(__file__), 'diagnostic source must be committed exactly')
    return {'diagnostic':sha_file(__file__), 'producer':producer.source_sha256(),
            'mixture':MIXTURE_SHA, 'backend':producer.backend.source_sha256(),
            'codec':producer.codec.source_sha256(), 'runtime_inventory':inventory}

def fixed_uniforms():
    return np.asarray(UNIFORM_BITS, dtype=np.uint32).view(np.float32).copy()

def window_query(query, latent, stage, row, first, last):
    require((stage,row,first,last) in [x[:4] for x in FIXTURES], 'undeclared window')
    require(query.dtype == latent.dtype == np.float32 and query.shape == (1,15)
            and latent.shape == (51200,21) and np.isfinite(latent).all(), 'actual F32 inputs required')
    require(first <= row < last and last-first == 1024, 'fixed 1024 window required')
    return np.ascontiguousarray(np.concatenate((np.repeat(query,1024,axis=0),latent[first:last,:stage]),axis=1))

def preflight(*, banks, failed_run, uniform_reconstruction, checkpoint, output):
    started=time.perf_counter(); pins=source_pins(); output=Path(output)
    require(not output.exists(), 'fresh CPU preflight output required')
    failed_run=Path(failed_run); require(sha_file(failed_run/'manifest.json')==FAILED_SHA,'failed manifest differs')
    d=json.loads((failed_run/'manifest.json').read_bytes())
    require(d['status']=='failed' and d['runner_source_sha256']==producer.source_sha256()
            and d['pair_spec']==producer.pairs.PairSpec(20261111,0).as_dict(),'wrong failed request')
    require(sha_file(failed_run/'latent-proposal-waves.npz')==WAVE_SHA, 'actual raw wave file differs')
    with np.load(failed_run/'latent-proposal-waves.npz',allow_pickle=False) as saved:
        require(saved.files==['wave_0000'],'exact single actual wave required'); latent=saved['wave_0000'].copy()
    require(array_descriptor(latent)=={'shape':[51200,21],'dtype':'<f4','sha256':WAVE_ARRAY_SHA},'actual raw latent bytes differ')
    require(sha_file(uniform_reconstruction)==UNIFORM_SHA,'authenticated uniform reconstruction differs')
    u=json.loads(Path(uniform_reconstruction).read_bytes())
    require(u['status']=='authenticated' and u['all21_before_after_states_authenticated']
            and u['final_state_matches_actual_failure_footer'] and len(u['stage_state_checks'])==21
            and all(x['before_match'] and x['after_match'] for x in u['stage_state_checks']), 'uniform states incomplete')
    from posteriorforge.inference import CHECKPOINT_SHA256
    require(sha_file(checkpoint)==CHECKPOINT_SHA256==d['checkpoint_sha256'],'checkpoint differs')
    context,rawquery,raw=producer.pairs.raw_inputs(producer.pairs.PairSpec(20261111,0),banks=banks)
    require(all(d['raw_scientific_bindings'][k]==v for k,v in raw.items()),'actual bank/query differs')
    theta=producer.codec.order_latents(producer.codec.encode_for_model(context.theta))
    rep=producer.pairs.representation; stats=rep.transform_for_model(context.obs); query=rep.transform_for_model(rawquery)
    require({k:array_descriptor(v) for k,v in [('context_theta',theta),('context_obs',stats),('query',query)]}==d['model_inputs'],'actual model inputs differ')
    arrays={}; rows=[]
    for stage,row,first,last,bits in FIXTURES:
        event=d['stage_draw_observation']['actual_stage_return_events'][stage]
        require(event['stage']==stage and event['actual_returned_draws']==array_descriptor(np.ascontiguousarray(latent[:,stage])), 'actual returned stage differs')
        scalar=next(x for x in u['historical_uniforms'] if x['stage']==stage and x['row']==row)
        require(scalar['float32_bytes_hex']==bits,'wrong reconstructed row uniform')
        q=window_query(query,latent,stage,row,first,last)
        mt,ms=producer.backend.mask_training(theta,stats,stage); mq=producer.backend.mask_query(q,stage)
        require(mt[:,stage].tobytes()==theta[:,stage].tobytes(),'target changed by mask')
        for key,value in [('theta',mt),('obs',ms),('query_unmasked',q),('query',mq)]: arrays[f'stage{stage}_{key}']=value
        rows.append({'stage':stage,'row':row,'window':[first,last],'focal_offset':row-first,
                     'reconstructed_uniform':scalar,'historical_latent_value':float(latent[row,stage]),
                     'target_coordinate':producer.backend.stage_policy(stage)['codec_coordinate'],
                     'mask':producer.backend.stage_policy(stage)})
    output.mkdir(parents=True)
    with (output/'inputs.npz').open('xb') as stream: np.savez(stream,**arrays)
    report={'schema':SCHEMA+'.preflight','status':'CPU_inputs_authenticated','sources':pins,
            'failed_manifest':str((failed_run/'manifest.json').resolve()),'failed_manifest_sha256':FAILED_SHA,
            'actual_wave_file':str((failed_run/'latent-proposal-waves.npz').resolve()),'actual_wave_sha256':WAVE_SHA,
            'uniform_reconstruction':str(Path(uniform_reconstruction).resolve()),'uniform_reconstruction_sha256':UNIFORM_SHA,
            'banks':str(Path(banks).resolve()),'checkpoint':str(Path(checkpoint).resolve()),'checkpoint_sha256':CHECKPOINT_SHA256,
            'fixtures':rows,'inputs_file':'inputs.npz','inputs_sha256':sha_file(output/'inputs.npz'),
            'arrays':{k:array_descriptor(v) for k,v in arrays.items()},'uniforms':array_descriptor(fixed_uniforms()),
            'uniform_values':fixed_uniforms().astype(float).tolist(),'uniform_bits':list(UNIFORM_BITS),
            'N1_trigger':'any N1024 member matching-limit failure; differing focal finite-endpoint failure masks across arms; or no N1024 focal finite-endpoint reproduction in either arm',
            'counterfactual_reevaluation':True,'historical_member_or_batch_recreated':False,
            'posterior_vectors_generated':0,'reference_integrations':0,'versions':versions(),
            'torch_imported':'torch' in __import__('sys').modules,'wall_seconds':time.perf_counter()-started}
    require(report['torch_imported'] is False,'CPU input preflight must not import Torch')
    write_json(output/'manifest.json',report);return report

def icdf_observer(callback):
    """Only a return observer; reverse AST recovers the exact frozen expression."""
    source=textwrap.dedent(inspect.getsource(_ICDF)); original=ast.parse(source); tree=copy.deepcopy(original)
    node=tree.body[0]; last=node.body[-1];require(isinstance(last,ast.Return),'inverse return changed')
    node.body[-1:]=[ast.Assign(targets=[ast.Name(id='_result',ctx=ast.Store())],value=last.value),
                   ast.Expr(value=ast.Call(func=ast.Name(id='_observe',ctx=ast.Load()),args=[ast.Call(func=ast.Name(id='locals',ctx=ast.Load()),args=[],keywords=[]),ast.Name(id='_result',ctx=ast.Load())],keywords=[])),
                   ast.Return(value=ast.Name(id='_result',ctx=ast.Load()))]
    reverse=copy.deepcopy(tree);reverse.body[0].body[-3:]=[ast.Return(value=reverse.body[0].body[-3].value)]
    require(ast.dump(reverse,include_attributes=False)==ast.dump(original,include_attributes=False),'inverse observer has undeclared AST changes')
    scope=dict(_ICDF.__globals__);scope['_observe']=callback;ast.fix_missing_locations(tree)
    exec(compile(tree,'<prior-tail-private-return-observer>','exec'),scope)
    return scope[_ICDF.__name__], {'reverse_AST_exact':True,'original_function_source_sha256':hashlib.sha256(source.encode()).hexdigest(),
                                  'arithmetic_changed':False,'shared_globals_modified':False}

def observed_inverse(prediction, torch):
    logits=prediction['logits']; borders=prediction['criterion'].borders
    require(logits.dtype==borders.dtype==torch.float32 and logits.device.type==borders.device.type=='cpu','actual CPU F32 output required')
    observations=[]
    def capture(state,result):
        index=state['index'];before=state['before']
        selected=before[index] if len(logits)==1 else before.gather(-1,index[:,None]).squeeze(-1)
        observations.append({k:v.detach().numpy().copy() for k,v in {
            'probabilities':state['probabilities'],'cdf':state['cdf'],'bin':index,'cdf_before':selected,
            'rest':state['rest'],'selected_probability':state['probability'],'output':result}.items()})
    observed,derivation=icdf_observer(capture); output=[]
    for value in fixed_uniforms():
        uniforms=torch.full((len(logits),),float(value),dtype=torch.float32,device='cpu')
        expected=_ICDF(logits,borders,uniforms);actual=observed(logits,borders,uniforms)
        require(actual.numpy().tobytes()==expected.numpy().tobytes(),'observed inverse differs from frozen whole-batch output')
        output.append(actual.numpy().copy())
    first=observations[0]
    require(all(o['probabilities'].tobytes()==first['probabilities'].tobytes() and o['cdf'].tobytes()==first['cdf'].tobytes() for o in observations),'same whole-batch density changed by U')
    arrays={'logits':logits.numpy().copy(),'borders':borders.numpy().copy(),
            'probabilities':first['probabilities'],'cdf':first['cdf'], 'uniforms':fixed_uniforms(),
            **{k:np.stack([o[k] for o in observations]) for k in ['bin','cdf_before','rest','selected_probability','output']}}
    with np.errstate(divide='ignore',invalid='ignore'):
        arrays['fraction_diagnostic_F32']=arrays['rest']/arrays['selected_probability']
    try: _ICDF(logits,borders,torch.ones(len(logits),dtype=torch.float32))
    except ValueError as error: stress={'u':1.,'outside_production_domain':True,'expected_rejection':str(error)}
    else: raise RuntimeError('frozen u=1 guard unexpectedly accepted')
    return arrays, {'observer':derivation,'all_provided_U_whole_batch_byte_parity':True,'u1_stress':stress,
                    'fraction_scope':'diagnostic rest/p division; original evaluates width*rest/p and has no fraction temporary'}

def focal_failure_mask(archive,stage,offset,uniform_index):
    geo=producer.codec._geometry();slot=producer.codec.ORDER[stage];z=float(archive['output'][uniform_index,offset])
    try: producer.codec._noise_decode(np.asarray([z],dtype=np.float64),geo['noise'][slot])
    except producer.codec.CoordinateError as error:
        return bool(np.isfinite(z) and 'endpoint, out-of-support, or saturated noise' in str(error))
    return False

def need_N1(comparisons, original_failures, candidate_failures):
    return (not all(x['passed'] for x in comparisons) or original_failures!=candidate_failures
            or not any(original_failures+candidate_failures))

def _helpers():
    q=producer.pairs.load_script('qualify_prior_coordinate_pair_v2.py',producer.QUALIFIER_SHA256)
    data=q.original_numeric.data_helper();old=data.load('qualify_ss15_conditionals.py',data.SS15_QUALIFIER_SHA256)
    return old.load('qualify_execution_cache.py',old.UTILITY_SHA),old.load('characterize_precision_approximation.py',old.BASE_SHA)

def original_arm(checkpoint,theta,obs,stage,active,torch):
    from posteriorforge import execution_cache as cache
    from posteriorforge.inference import frozen_regressor_settings
    settings=frozen_regressor_settings(checkpoint);settings['inference_precision']=torch.float32
    _,sdk,_=cache._load_sdk();regressor=sdk.TabPFNRegressor(**settings)
    base=SimpleNamespace(_torch=torch,device='cuda',metadata={'model_rng_isolation_count':0},_sdk_regressor=sdk)
    base._model_rng=MethodType(cache.ResidentStageExecutor._model_rng,base)
    with base._model_rng(): regressor.fit(np.concatenate((obs,theta[:,:stage]),axis=1),theta[:,stage])
    require(not regressor.use_autocast_ and regressor.forced_inference_dtype_==torch.float32
            and regressor.fit_mode=='fit_preprocessors','original FP32 settings changed')
    arm=SimpleNamespace(uncached_regressor=regressor,regressor=regressor,base=base,cdf_descriptor=active,
                        stage_index=stage,query_batch_size=1024,hybrid_counters={'uncached_predict_count':0,'uncached_query_rows':0,
                        'original_preprocessing_lists_restored':True,'original_translator_alias_restored':True})
    arm.predict_component=lambda member,query: producer.backend.original._BODY(arm,member,query,True)
    return arm

def parameter_evidence(regressor):
    roots=getattr(regressor.executor_,'models',None) or [regressor.model_]
    models=[]
    for index,model in enumerate(roots):
        digest=hashlib.sha256();normalizers=[]
        for name,value in model.named_parameters():
            digest.update(name.encode());digest.update(str(value.dtype).encode())
            digest.update(str(tuple(value.shape)).encode());digest.update(value.detach().cpu().numpy().tobytes())
        for name,module in model.named_modules():
            if type(module).__name__=='InputNormalizationEncoderStep':
                normalizers.append({'name':name,'state':plain(vars(module))})
        models.append({'index':index,'parameter_sha256':digest.hexdigest(),
                       'training_module_count':sum(bool(m.training) for m in model.modules()),
                       'normalization_parameters':normalizers})
    require(all(m['training_module_count']==0 for m in models),'model must remain eval()')
    return models


def plain(value,seen=None):
    seen=set() if seen is None else seen
    if value is None or isinstance(value,(str,bool,int)):return value
    if isinstance(value,(float,np.floating)):
        return float(value) if np.isfinite(value) else repr(float(value))
    if isinstance(value,np.integer):return int(value)
    if hasattr(value,'detach'):value=value.detach().cpu().numpy()
    if isinstance(value,np.ndarray):return {'array':array_descriptor(value),'values':value.tolist()}
    if isinstance(value,(list,tuple)):return [plain(x,seen) for x in value]
    if isinstance(value,dict):return {str(k):plain(v,seen) for k,v in value.items() if k not in ('_modules','_parameters','_buffers')}
    if id(value) in seen:return {'reference_type':type(value).__qualname__}
    seen.add(id(value))
    return {'type':type(value).__module__+'.'+type(value).__qualname__,
            'state':plain(vars(value),seen) if hasattr(value,'__dict__') else repr(value)}


def selected_parameters(regressor,member,output,stem):
    executor=regressor.executor_;models=getattr(executor,'models',None)
    model=models[member] if models else regressor.model_
    training=[name for name,module in model.named_modules() if module.training]
    require(not training,'selected model must stay eval()')
    configs=getattr(executor,'ensemble_configs',getattr(executor,'configs',None))
    transform=getattr(configs[member],'target_transform',None)
    path=output/(stem+'-target-transform.pkl');raw=pickle.dumps(transform,protocol=5);path.write_bytes(raw)
    normalizers=[]
    for name,module in model.named_modules():
        if type(module).__name__=='InputNormalizationEncoderStep':
            normalizers.append({'name':name,'attributes':plain(vars(module)),
                                'registered_buffers':{k:plain(v) for k,v in module.named_buffers(recurse=False)}})
    return {'selected_model_training':bool(model.training),'training_module_names':training,
            'actual_target_transform':plain(transform),'target_transform_pickle':{
                'file':path.name,'sha256':sha_file(path),'bytes':len(raw)},
            'actual_normalization_parameters':normalizers}


def fitted_parameters(regressor,output,stem):
    executor=regressor.executor_;configs=getattr(executor,'ensemble_configs',getattr(executor,'configs',None))
    require(configs is not None and len(configs)==8,'actual all8 fitted configurations required')
    # Exact trusted local pickle preserves every fitted target-transform parameter;
    # JSON additionally exposes actual arrays/scalars, not only configuration hashes.
    path=output/(stem+'-fitted-configs.pkl');raw=pickle.dumps(configs,protocol=5);path.write_bytes(raw)
    return {'exact_fitted_configurations_pickle':{'file':path.name,'sha256':sha_file(path),'bytes':len(raw)},
            'configurations':plain(configs),'target_mean':regressor.y_train_mean_,
            'target_sd':regressor.y_train_std_,'temperature':regressor.softmax_temperature}


def capture(*,preflight_manifest,output,ledger,max_seconds=240):
    source_pins(committed=True);require(type(max_seconds) is int and 60<=max_seconds<=240,'work deadline must fit <=300s allocation')
    from posteriorforge.reference_grid import _require_budgeted_gpu
    budget=_require_budgeted_gpu(ledger);require(budget['gpus']==1 and budget['reserved_gpu_seconds']<=300,'one GPU <=300 GPU-seconds required')
    p=json.loads(Path(preflight_manifest).read_bytes());require(p['schema']==SCHEMA+'.preflight' and p['status']=='CPU_inputs_authenticated','CPU input preflight required')
    require(p['sources']==source_pins() and sha_file(p['failed_manifest'])==FAILED_SHA and sha_file(p['actual_wave_file'])==WAVE_SHA
            and sha_file(p['uniform_reconstruction'])==UNIFORM_SHA and sha_file(p['checkpoint'])==p['checkpoint_sha256'],'preflight/source/custody changed')
    inp=Path(preflight_manifest).parent/p['inputs_file'];require(sha_file(inp)==p['inputs_sha256'],'input archive changed')
    with np.load(inp,allow_pickle=False) as saved: inputs={k:saved[k].copy() for k in saved.files}
    require({k:array_descriptor(v) for k,v in inputs.items()}==p['arrays'],'input descriptors changed')
    output=Path(output);require(not output.exists(),'fresh GPU capture output required');output.mkdir(parents=True)
    report={'schema':SCHEMA,'status':'running','preflight_sha256':sha_file(preflight_manifest),'preflight':p,'budget':budget,
            'counterfactual_reevaluation':True,'historical_grouping_member_recreated':False,'posterior_vectors_generated':0,
            'reference_integrations':0,'G3_updates':0,'FP16_activated':False,'stages':[],
            'RNG_scope':'fixed SDK random_state; original/candidate Torch fork isolation. Internal preprocessing RNG call counts not instrumented; not claimed zero.',
            'uniform_generation_calls':0,'sources':source_pins()}
    write_json(output/'manifest.json',report);started=time.perf_counter();deadline=started+max_seconds;active=None;prof=None
    try:
        import torch
        from posteriorforge.compat_cdf import finite_cdf_compatibility
        require(torch.cuda.is_available() and torch.cuda.device_count()==1,'one visible GPU required')
        require(torch.get_default_dtype()==torch.float32 and torch.get_float32_matmul_precision()=='highest'
                and not torch.backends.cuda.matmul.allow_tf32 and not torch.is_autocast_enabled('cuda'),'strict FP32 flags required')
        torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','16')))
        utility,base=_helpers();initial=utility.rng_identity(utility.rng_state(torch));report['default_RNG_before']=initial
        prof=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
                                    record_shapes=False,profile_memory=False,with_stack=False)
        with prof, finite_cdf_compatibility() as active:
            for fixture in p['fixtures']:
                require(time.perf_counter()+35<deadline,'bounded work deadline; retain partial capture')
                stage=fixture['stage'];focal=fixture['focal_offset'];q=inputs[f'stage{stage}_query'];mt=inputs[f'stage{stage}_theta'];ms=inputs[f'stage{stage}_obs']
                row={'fixture':fixture,'arms':{},'comparisons':[]};report['stages'].append(row);oldpaths=[];original_masks=[];candidate_masks=[]
                uniform_index=next(i for i,v in enumerate(fixed_uniforms()) if v.tobytes().hex()==fixture['reconstructed_uniform']['float32_bytes_hex'])
                def timed(label,fn):
                    require(time.perf_counter()+8<deadline,'bounded work deadline before neural call')
                    torch.cuda.synchronize();wall=time.perf_counter();a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
                    with torch.profiler.record_function(label): a.record();value=fn();b.record()
                    b.synchronize();return value,{'host_wall_seconds':time.perf_counter()-wall,'CUDA_elapsed_span_ms':a.elapsed_time(b),'CUDA_span_is_not_kernel_union':True}
                def save(arm,member,features,label,prediction,elapsed):
                    arrays,meta=observed_inverse(prediction,torch)
                    arrays.update(raw_translated_probabilities=prediction['member']['translated_probabilities'].numpy().copy(),
                                  component_raw_mass=prediction['component_raw_mass'].numpy().copy(),
                                  raw_model_logits=prediction['member']['raw_logits'].numpy().copy(),
                                  from_borders=prediction['member']['from_borders'].numpy().copy(),
                                  to_borders=prediction['member']['to_borders'].numpy().copy())
                    filename=f'stage{stage}-{arm}-member{member}-{label}.npz'
                    with (output/filename).open('xb') as stream: np.savez(stream,**arrays)
                    entry={'member':member,'label':label,'file':filename,'sha256':sha_file(output/filename),
                           'arrays':{k:array_descriptor(v) for k,v in arrays.items()},'configuration':prediction['member']['configuration_identity'],
                           'input':array_descriptor(features),'ICDF':meta,'timing':elapsed,
                           'selected_fitted_state':selected_parameters(control.regressor if arm=='original' else engine.regressor,member,output,filename[:-4])}
                    row['arms'].setdefault(arm,[]).append(entry);write_json(output/'manifest.json',report)
                    return arrays,entry
                control,fit_time=timed(f'pf-tail/{stage}/original/fit',lambda:original_arm(p['checkpoint'],mt,ms,stage,active,torch))
                row['original_fit_timing']=fit_time;row['original_model_before']=base.model_summary(control.regressor,utility)
                row['original_parameters_before']=parameter_evidence(control.regressor)
                row['original_fitted_parameters']=fitted_parameters(control.regressor,output,f'stage{stage}-original')
                for member in range(8):
                    prediction,t=timed(f'pf-tail/{stage}/original/{member}/N1024',lambda m=member:control.predict_component(m,q))
                    arrays,e=save('original',member,q,'N1024',prediction,t);oldpaths.append(output/e['file'])
                    original_masks.append(focal_failure_mask(arrays,stage,focal,uniform_index));del arrays,prediction
                with producer.backend.PriorNativeSession() as session:
                    with ExitStack() as cleanup:
                        engine=producer.backend.PriorNativeStageExecutor(p['checkpoint'],cdf_descriptor=active);cleanup.callback(engine.close)
                        prepared,t=timed(f'pf-tail/{stage}/candidate/prepare',lambda:engine.prepare(mt,ms,stage));row['candidate_prepare']={'prepared':prepared,'timing':t}
                        row['candidate_model_before']=base.model_summary(engine.regressor,utility)
                        row['candidate_parameters_before']=parameter_evidence(engine.regressor)
                        row['candidate_fitted_parameters']=fitted_parameters(engine.regressor,output,f'stage{stage}-candidate')
                        for member in range(8):
                            prediction,t=timed(f'pf-tail/{stage}/candidate/{member}/N1024',lambda m=member:engine.predict_component(m,q,capture_member=True))
                            arrays,e=save('candidate',member,q,'N1024',prediction,t);candidate_masks.append(focal_failure_mask(arrays,stage,focal,uniform_index))
                            with np.load(oldpaths[member],allow_pickle=False) as previous:
                                require(np.array_equal(previous['borders'],arrays['borders']),'matching canonical borders differ')
                                oc=row['arms']['original'][member]['configuration'];cc=e['configuration']
                                require(all(oc[k]==cc[k] for k in ('type','sha256')),'matching fitted member configuration differs')
                                row['comparisons'].append({'member':member,**utility.compare_probabilities(previous['raw_translated_probabilities'],arrays['raw_translated_probabilities'])})
                            del arrays,prediction
                        trigger=need_N1(row['comparisons'],original_masks,candidate_masks)
                        row['N1_decision']={'triggered':trigger,'declared_rule':p['N1_trigger'],'original_focal_finite_endpoint_failures':original_masks,'candidate_focal_finite_endpoint_failures':candidate_masks}
                        # Preserve at least 180 work seconds for the second mandatory
                        # N1024 window before taking optional first-stage N1 work.
                        allowed=(stage==19 or deadline-time.perf_counter()>=180)
                        row['N1_decision']['budget_allowed']=allowed
                        row['N1_decision']['skipped_to_prioritize_mandatory_N1024']=bool(trigger and not allowed)
                        trigger=trigger and allowed
                        if trigger:
                            for member in range(8):
                                prediction,t=timed(f'pf-tail/{stage}/candidate/{member}/N1',lambda m=member:engine.predict_component(m,q[focal:focal+1],capture_member=True))
                                arrays,_=save('candidate',member,q[focal:focal+1],'N1',prediction,t);del arrays,prediction
                        row['candidate_model_after']=base.model_summary(engine.regressor,utility)
                        row['candidate_parameters_after']=parameter_evidence(engine.regressor)
                        require([x['parameter_sha256'] for x in row['candidate_parameters_before']]==[x['parameter_sha256'] for x in row['candidate_parameters_after']],'candidate parameters changed')
                row['candidate_session']=session.report();require(row['candidate_session']['aliases_and_flags_restored'],'candidate precision aliases not restored')
                if row['N1_decision']['triggered'] and row['N1_decision']['budget_allowed']:
                    for member in range(8):
                        prediction,t=timed(f'pf-tail/{stage}/original/{member}/N1',lambda m=member:control.predict_component(m,q[focal:focal+1]))
                        arrays,_=save('original',member,q[focal:focal+1],'N1',prediction,t);del arrays,prediction
                row['original_model_after']=base.model_summary(control.regressor,utility)
                row['original_parameters_after']=parameter_evidence(control.regressor)
                require([x['parameter_sha256'] for x in row['original_parameters_before']]==[x['parameter_sha256'] for x in row['original_parameters_after']],'original parameters changed')
                require(control.hybrid_counters['original_preprocessing_lists_restored'] and control.hybrid_counters['original_translator_alias_restored'],'original head lists/translator not restored')
                row['original_selected_member_counters']=control.hybrid_counters
                require(all(x['contract']['input_normalization_train_only_verified'] and x['temperature']==.9 and x['canonical_borders_dtype']==x['physical_borders_dtype']=='torch.float32' for x in [row['original_model_before'],row['original_model_after'],row['candidate_model_before'],row['candidate_model_after']]),'normalization/temperature/border contract failed')
                require(utility.rng_identity(utility.rng_state(torch))==initial,'default RNG changed')
                del control,engine;gc.collect();torch.cuda.synchronize();write_json(output/'manifest.json',report)
        report['compatibility_adapter']=dict(active)
        require(active['original_alias_restored'] and active['sdk_sources_unchanged'],'CDF restoration failed')
        report['default_RNG_after']=utility.rng_identity(utility.rng_state(torch));require(report['default_RNG_after']==initial,'default RNG changed')
        require(source_pins()==report['sources'],'source changed during capture')
        trace=output/'cuda-trace.json';export_start=time.perf_counter();prof.export_chrome_trace(str(trace))
        report['profile']={'file':trace.name,'sha256':sha_file(trace),'bytes':trace.stat().st_size,'export_seconds':time.perf_counter()-export_start,
                           'shapes_memory_stacks':False,'scope':'counterfactual capture, not clean performance; actual kernel/copy unions require offline correlation analysis; CUDA elapsed spans are not exclusive kernels'}
        report['status']=('mandatory_N1024_complete_optional_N1_inconclusive' if any(r['N1_decision']['skipped_to_prioritize_mandatory_N1024'] for r in report['stages']) else 'two_stage_conditional_capture_complete');report['sources_unchanged']=True;report['default_RNG_preserved']=True
    except BaseException as error:
        report.update(status='failed_or_incomplete_conditional_capture',error_type=type(error).__name__,error=str(error))
        if active is not None: report['compatibility_adapter']=dict(active)
        raise
    finally:
        # Best-effort observations are actual post-exit state, never a fabricated
        # success footer; secondary observation/export failures cannot mask primary.
        post={}
        def observe(name,fn):
            try:post[name]={'status':'observed','value':fn()}
            except BaseException as error:post[name]={'status':'unavailable','error_type':type(error).__name__,'error':str(error)}
        observe('sources',lambda:source_pins()==report['sources'])
        if 'utility' in locals() and 'initial' in locals():
            observe('default_RNG_matches_before',lambda:utility.rng_identity(utility.rng_state(torch))==initial)
        if active is not None:
            observe('CDF_after_exit',lambda:dict(active))
        if 'session' in locals():observe('precision_session_after_exit',lambda:session.report())
        if 'control' in locals():observe('original_head_lists_and_translator',lambda:dict(control.hybrid_counters))
        if prof is not None and not (output/'cuda-trace.json').exists():
            def partial_trace():
                path=output/'cuda-trace.json';begin=time.perf_counter();prof.export_chrome_trace(str(path))
                return {'file':path.name,'sha256':sha_file(path),'bytes':path.stat().st_size,'export_seconds':time.perf_counter()-begin,'partial':True}
            observe('partial_trace',partial_trace)
        report['post_exit_observations']=post
        report['optional_N1_incomplete']=any(r.get('N1_decision',{}).get('skipped_to_prioritize_mandatory_N1024',False) for r in report['stages'])
        report['elapsed_seconds']=time.perf_counter()-started;write_json(output/'manifest.json',report)
    return report

def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    cpu=sub.add_parser('preflight')
    for name in ('banks','failed-run','uniform-reconstruction','checkpoint','output'): cpu.add_argument('--'+name,required=True)
    gpu=sub.add_parser('capture')
    for name in ('preflight-manifest','output','ledger'): gpu.add_argument('--'+name,required=True)
    gpu.add_argument('--max-seconds',type=int,default=240)
    values=vars(parser.parse_args());action=values.pop('action');result=(preflight if action=='preflight' else capture)(**values)
    print(json.dumps({'status':result['status'],'schema':result['schema']},sort_keys=True))
if __name__=='__main__': main()
