import hashlib,importlib.util,json,os,pickle,random,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
s=Path('/scratch/project/eu-26-17/eubrieucb/posteriorforge');phase=s/'results/accuracy-prior-coordinates-v3';out=phase/'posterior-launch-01/parent-validation-01'
parent=phase/'neural-posterior-seed20261111-case0-01/manifest.json';samples=parent.with_name('samples.npz')
expected={'manifest.json':'cbe866875983cc3271a22a5862a16fa7e4e3f79e1c28502dad1563c11b96ff32','samples.npz':'b80a51c250a9155a81df54240b8c6e75bbd3a69dd9f1d750dd6625a227ebe3c0'}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
paths={'manifest.json':parent,'samples.npz':samples};before={k:sha(v) for k,v in paths.items()};assert before==expected
revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();assert revision=='a0d889067b92a2c51d3386f7a9e0a02a41d9ee3d'
assert not subprocess.check_output(['git','diff','--name-only'],text=True).strip()
assert not subprocess.check_output(['git','diff','--cached','--name-only'],text=True).strip()
p=Path('scripts/refresh_prior_pair_v3_global_block.py');module=importlib.util.spec_from_file_location('_actual_v3_parent_readonly',p);w=importlib.util.module_from_spec(module);module.loader.exec_module(w)
assert w.source_sha256()=='0720eb75a10550f2486fd1d77f3a5ba4e20496c3812194c878bec5633fc60ff3'
state_before=pickle.dumps((np.random.get_state(),random.getstate()),protocol=5)
forbidden={w.refresh.__code__:'refresh',w.refresh_with_timing.__code__:'refresh_with_timing',w.kernel.refresh.__code__:'kernel.refresh'};events=[]
def profile(frame,event,arg):
 if event=='call' and frame.f_code in forbidden:
  events.append(forbidden[frame.f_code]);raise AssertionError('unauthorized refresh code entered: '+events[-1])
oldprofile=sys.getprofile();started=time.perf_counter()
receipt={'schema':'posteriorforge.v3_genuine_parent_readonly_validation.v1','worktree_revision':revision,'wrapper_sha256':w.source_sha256(),'raw_before_sha256':before,'expected_raw_sha256':expected,'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),'status':'running'}
try:
 sys.setprofile(profile)
 document,values,auth=w.load_parent(parent,expected['manifest.json'],scratch=s,spec=w.pair_spec(20261111,0),execution_screen=phase/'pair-conditionals-seed20261111-case0-02/manifest.json',pair_cpu_receipt=s/'results/accuracy-prior-coordinates-v1/pair-preflight-combined-01/receipt.json')
 sys.setprofile(oldprofile)
 receipt.update(status='passed',parent_authentication=auth,selected_physical_shape=list(values.shape),selected_physical_dtype=str(values.dtype),selected_physical_descriptor=w.array_descriptor(values))
 print('ACTUAL_V3_PARENT_LOAD_PASS',json.dumps({'shape':list(values.shape),'dtype':str(values.dtype),'header_passed':auth['header']['passed'],'finite_decode_replay':auth['actual_all_generated_rows_decode_replay'],'execution_checks':auth['actual_execution_checks']},sort_keys=True))
except BaseException as error:
 receipt.update(status='failed',error_type=type(error).__name__,error=str(error));traceback.print_exc();raise
finally:
 sys.setprofile(oldprofile)
 after={k:sha(v) for k,v in paths.items()};state_after=pickle.dumps((np.random.get_state(),random.getstate()),protocol=5)
 receipt.update(raw_after_sha256=after,raw_manifest_and_samples_unchanged=after==before,default_numpy_python_RNG_preserved=state_before==state_after,forbidden_refresh_calls=events,production_G3_calls=0,posterior_or_world_or_reference_or_NN_or_GPU_calls=0,elapsed_seconds=time.perf_counter()-started,Torch_imported='torch' in sys.modules)
 assert after==before and state_before==state_after and not events
 (out/'receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
 (out/'source-flags.json').write_text(json.dumps({k:receipt[k] for k in ['worktree_revision','wrapper_sha256','CUDA_VISIBLE_DEVICES','raw_manifest_and_samples_unchanged','default_numpy_python_RNG_preserved','forbidden_refresh_calls','production_G3_calls','posterior_or_world_or_reference_or_NN_or_GPU_calls','Torch_imported']},indent=2,sort_keys=True)+'\n')
