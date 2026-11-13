"""Authenticated CPU component-uniform reconstruction; no model or inverse call."""
from pathlib import Path
import ast
import hashlib
import inspect
import importlib.metadata
import json
import pickle
import platform
import random
import sys
import textwrap
import time

import numpy as np

ROOT = Path('/home/it4i-bbenvegnen/workspace/PosteriorForge')
MANIFEST = Path('/scratch/project/eu-26-17/eubrieucb/posteriorforge/results/accuracy-prior-coordinates-v1/native-pilot-v2-seed20261111-case0-01/manifest.json')
MANIFEST_SHA = '92755b788fc894d4ca16157e573bd1b5c9d8a4468b71cf184fa16303bced7bce'
OUT = Path('/scratch/project/eu-26-17/eubrieucb/posteriorforge/results/accuracy-prior-coordinates-v1/historical-component-uniform-reconstruction-02')
PINS = {
 'src/posteriorforge/performance_prior_pair_v2.py':'311f2cbfb3c3bd977ff9af348cd81a008c3c7711ec70f850fb25b3bc2d120a85',
 'src/posteriorforge/performance_prior_physical_fifo_v1.py':'a30e0c15513eb0c1a6ed3c68f0781327edef4714a244a119a3a4d4c68f8780bf',
 'src/posteriorforge/performance_prior_coordinates_v1.py':'71e5e2906fd663534e56878de1fc4aff55093e3d950a13f348f57417ca7c01d3',
 'src/posteriorforge/performance_fp16_experiment.py':'2280e1ceef807f05a264b5318be18eb8bee7c52c8bab21884d6682f84cafcba9',
 'src/posteriorforge/execution_ss15_sampling.py':'4037d48846ef3678ce7b647c1a326606ea967d396b3ce34bb1998241a131f6de',
 'src/posteriorforge/performance_runner.py':'23ccad8b5bbcdd7a48340b349d515ca870763359cfeb71333e02fc2b6a8aca3b',
 'src/posteriorforge/prior_coordinate_forensics_v1.py':'81399b3beb34760a9b25ac99805e6a500a90f3f505193d4acf00b83b4c3eb5c0',
}
sha = lambda data: hashlib.sha256(data).hexdigest()
def need(value, message):
 if not value: raise RuntimeError(message)
def default_np_python():
 return {'numpy':sha(pickle.dumps(np.random.get_state(),protocol=5)),
         'python':sha(pickle.dumps(random.getstate(),protocol=5))}
def source_hashes():
 return {name:sha((ROOT/name).read_bytes()) for name in PINS}
def syntax(expression):
 return ast.dump(ast.parse(expression,mode='eval').body,include_attributes=False)

need(not OUT.exists(),'fresh diagnostic output required')
OUT.mkdir(parents=True)
start=time.perf_counter()
report={'schema':'posteriorforge.historical_component_uniform_reconstruction.v1',
 'status':'started','historical_uniforms':None,
 'script_sha256':sha(Path(__file__).read_bytes()),'script_path':str(Path(__file__).resolve()),
 'manifest_path':str(MANIFEST),'expected_manifest_sha256':MANIFEST_SHA,
 'scope':'private CPU RNG reconstruction authenticated against historical state hashes, not captured ICDF inputs or historical member/bin/probability reconstruction',
 'no_model_sampler_ICDF_codec_world_reference_posterior_or_G3_execution':True,
 'counts':{'private_Generator_constructions':0,'private_manual_seed_calls':0,
           'replayed_torch_rand_calls':0,'replayed_uniform_values':0,
           'private_get_state_hash_reads':0},'stage_state_checks':[]}
np_python_before=default_np_python()
import torch
cpu_before=sha(torch.get_rng_state().numpy().tobytes())
cuda_before=torch.cuda.is_initialized()
report['default_RNG_scope']={'numpy_python':'before Torch and source-chain imports through end',
 'torch_CPU':'immediately after Torch import, before source-chain imports, through end',
 'restoration_performed':False,'global_manual_seed_calls':0}
report['default_rng_before']={**np_python_before,'torch_CPU':cpu_before}
report['CUDA_initialized_before']=cuda_before
try:
 need(not cuda_before,'CUDA already initialized: reject CPU-only reconstruction')
 raw=MANIFEST.read_bytes(); need(sha(raw)==MANIFEST_SHA,'raw failed manifest SHA mismatch')
 d=json.loads(raw)
 need(d['status']=='failed' and d['pair_spec']=={'context_seed':20261111,'case':0,'context_size':4096,'count':50000,'repetition':0},'wrong failed request identity')
 need(d['versions']['python']==platform.python_version() and d['versions']['numpy']==np.__version__ and d['versions']['torch']==importlib.metadata.version('torch'),'actual producer Python/NumPy/Torch versions differ')
 before=source_hashes(); need(before==PINS,'frozen source file hash mismatch')
 report['source_hashes_before']=before
 from posteriorforge import performance_prior_pair_v2 as producer
 from posteriorforge import execution_ss15_sampling as wave
 producer.require_sources()
 bound=producer._bind_run(producer.pairs.PairSpec(20261111,0),d['matching_conditional_qualification_sha256'],str(Path('/scratch/project/eu-26-17/eubrieucb/posteriorforge/results/accuracy-prior-coordinates-v1/pair-preflight-combined-01/receipt.json')))
 sampler=bound.__globals__['clean'].frozen_sampling()
 canonical=wave.frozen.frozen_sampling()
 need(sampler.component_stage_sampler is canonical.component_stage_sampler,'bound factory differs from authenticated canonical factory')
 need(bound.__globals__['wave'] is wave,'bound run wave namespace differs')
 private=wave.private_wave()
 need(wave.execution_metadata()['source_transform']['reverse_ast_exact'] is True,'wave AST delta not source-sealed')
 factory_source=textwrap.dedent(inspect.getsource(sampler.component_stage_sampler))
 tree=ast.parse(factory_source); nested=next(n for n in tree.body[0].body if isinstance(n,ast.FunctionDef) and n.name=='sample_stage')
 expected=syntax('torch.rand(count,generator=generators["component_uniform"],dtype=torch.float32,device="cpu")')
 calls=[n for n in ast.walk(nested) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='torch' and n.func.attr=='rand' and any(isinstance(k.value,ast.Subscript) and isinstance(k.value.value,ast.Name) and k.value.value.id=='generators' and isinstance(k.value.slice,ast.Constant) and k.value.slice.value=='component_uniform' for k in n.keywords)]
 need(len(calls)==1 and ast.dump(calls[0],include_attributes=False)==expected,'component uniform call shape/dtype/device/generator changed')
 assignment=next(n for n in nested.body if isinstance(n,ast.Assign) and n.value is calls[0])
 need(any(isinstance(t,ast.Name) and t.id=='stage_uniforms' for t in assignment.targets),'uniform vector assignment differs')
 need(nested.body.index(assignment)<next(i for i,n in enumerate(nested.body) if isinstance(n,ast.If)) and nested.body.index(assignment)<next(i for i,n in enumerate(nested.body) if isinstance(n,ast.While)),'uniform draw not once before stage branch/head retries')
 uses=[n for n in ast.walk(nested) if isinstance(n,ast.Subscript) and isinstance(n.value,ast.Name) and n.value.id=='generators' and isinstance(n.slice,ast.Constant) and n.slice.value=='component_uniform']
 need(len(uses)==1,'additional component generator use in stage sampler')
 source_inspect={'component_factory_source_sha256':sha(factory_source.encode()),'canonical_sampling_source_sha256':wave.frozen.SAMPLING_SHA,'canonical_sampling_Git_commit':wave.frozen.SAMPLING_COMMIT,'bound_factory_is_canonical_factory':True,'bound_wave_is_4037':True,'wave_reverse_AST_exact':True,'one_CPU_FP32_count_vector_before_stage_branch_and_retries':True,'component_generator_access_count_in_factory':len(uses),'raw_source_chain_guard_passed':True,'no_original_stage_sampler_invoked':True}
 report['source_chain']=source_inspect
 policy=d['sampling_policy']; stream=policy['streams']['component_uniform']; seed=d['seed_streams']['component_uniform']
 need(seed==3175593578 and stream['seed']==seed and stream['dtype']=='torch.float32' and stream['device']=='cpu' and stream['generator']=='torch.Generator','recorded stream metadata mismatch')
 need(policy['uniform_assignment']=='preassigned_per_stage_by_physical_row' and policy['head_retry_preserves_row_uniform'] is True,'recorded row/U retry semantics differ')
 need(sampler.worker_stream_seeds(20261111,0,0,0,1)==d['seed_streams'],'seed-role derivation differs')
 need(d['actual_complete_generated_wave_count']==1 and d['proposal_wave_sizes']==[51200] and d['failed_decode_wave_index']==0,'more than one or different wave')
 observation=d['stage_draw_observation']; events=observation['actual_stage_return_events']; other=d['actual_stage_return_archive']['observation']['actual_stage_return_events']
 need(events==other and len(events)==21 and observation['capture_errors']==[] and observation['filled_stage_count']==21 and observation['wave_index']==0 and observation['row_count']==51200,'stage observation incomplete or ambiguous')
 need(d['failed_decode_wave_descriptor']['shape']==[51200,21] and d['failed_decode_wave_descriptor']['dtype']=='<f4','full latent wave shape/dtype differs')
 need(len(d['stage_events'])==21 and all(e['stage_index']==i and e['proposed_rows']==51200 and e['sampler']['uniform_draws']==51200 for i,e in enumerate(d['stage_events'])),'actual stage event shape/uniform counts differ')
 for i,e in enumerate(events):
  need(e['stage']==i and e['returned'] is True and e['actual_returned_draws']['shape']==[51200] and e['actual_returned_draws']['dtype']=='<f4','returned stage shape/dtype/order mismatch')
 report['stream']={'name':'component_uniform','seed':seed,'device':'cpu','dtype':'torch.float32','values_per_stage':51200,'stage_count':21,'role':stream['role']}
 footer_record=d['observed_failure_footer']['observations']['actual_private_generator_states']['value']['component_uniform']
 need(isinstance(footer_record,dict) and footer_record['dtype']=='uint8' and footer_record['shape']==[5056] and sha(bytes.fromhex(footer_record['bytes_hex']))==footer_record['sha256'],'actual failure-footer state metadata malformed')
 g=torch.Generator(device='cpu');report['counts']['private_Generator_constructions']+=1
 g.manual_seed(seed);report['counts']['private_manual_seed_calls']+=1
 selected=[]
 def state():
  report['counts']['private_get_state_hash_reads']+=1
  return sha(g.get_state().numpy().tobytes())
 for stage,event in enumerate(events):
  actual_before=state(); expected_before=event['private_streams_before']['component_uniform']
  need(actual_before==expected_before,'stage '+str(stage)+' historical before-state mismatch')
  values=torch.rand(51200,generator=g,dtype=torch.float32,device='cpu')
  report['counts']['replayed_torch_rand_calls']+=1;report['counts']['replayed_uniform_values']+=51200
  need(values.shape==(51200,) and values.dtype==torch.float32 and values.device.type=='cpu','reconstructed vector shape/dtype/device differs')
  actual_after=state();expected_after=event['private_streams_after_or_failure']['component_uniform']
  report['stage_state_checks'].append({'stage':stage,'values':51200,'before_sha256':actual_before,'recorded_before_sha256':expected_before,'before_match':actual_before==expected_before,'after_sha256':actual_after,'recorded_after_sha256':expected_after,'after_match':actual_after==expected_after})
  need(actual_after==expected_after,'stage '+str(stage)+' historical after-state mismatch')
  if stage in (15,19):
   row=4355 if stage==15 else 41765
   scalar=values[row]; b=scalar.numpy().tobytes()
   selected.append({'stage':stage,'row':row,'dtype':'<f4','value':float(scalar.item()),'float32_bytes_hex':b.hex(),'label':'authenticated historical component-uniform reconstruction; not directly captured ICDF-input witness'})
  del values
 final_recorded=footer_record['sha256']
 need(isinstance(final_recorded,str) and actual_after==final_recorded,'actual failure-footer component state differs from stage20 after-state')
 need(source_hashes()==before,'frozen source changed during reconstruction')
 report['source_hashes_after']=source_hashes()
 report['all21_before_after_states_authenticated']=True
 report['final_state_matches_actual_failure_footer']=True
 report['historical_uniforms']=selected
 report['no_member_bin_probability_or_other_generator_reconstruction']=True
 report['status']='authenticated'
except BaseException as error:
 report['status']='failed_historical_uniforms_unknown'; report['historical_uniforms']=None
 report['error_type']=type(error).__name__;report['error']=str(error)
finally:
 after_np_python=default_np_python();after_cpu=sha(torch.get_rng_state().numpy().tobytes())
 report['default_rng_after']={**after_np_python,'torch_CPU':after_cpu}
 report['default_numpy_python_TorchCPU_RNG_preserved']=report['default_rng_after']==report['default_rng_before']
 report['CUDA_initialized_after']=torch.cuda.is_initialized()
 report['CUDA_never_initialized_observed_at_boundaries']=not cuda_before and not report['CUDA_initialized_after']
 if not report['default_numpy_python_TorchCPU_RNG_preserved'] or report['CUDA_initialized_after']:
  report['status']='failed_historical_uniforms_unknown';report['historical_uniforms']=None
 report['versions']={'python':platform.python_version(),'numpy':np.__version__,'torch_metadata':importlib.metadata.version('torch'),'torch_module':torch.__version__}
 report['wall_seconds']=time.perf_counter()-start
 (OUT/'manifest.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
 print(json.dumps({'output':str(OUT),'status':report['status'],'manifest_sha256':sha((OUT/'manifest.json').read_bytes()),'counts':report['counts'],'historical_uniforms':report['historical_uniforms'],'default_RNG_preserved':report['default_numpy_python_TorchCPU_RNG_preserved'],'CUDA_initialized':report['CUDA_initialized_after'],'error':report.get('error')},sort_keys=True))
if report['status']!='authenticated':sys.exit(1)
