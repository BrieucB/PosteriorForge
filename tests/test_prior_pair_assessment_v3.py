"""CPU-only engineering saved-Z assessment tests; no G3 call or random draw."""
from copy import deepcopy
import hashlib,importlib.util,json,pickle,random,sys
from pathlib import Path
from types import FunctionType,SimpleNamespace
import numpy as np
import pytest
from posteriorforge import conditional_globals as kernel,prior_coordinates_v1 as codec,prior_coordinate_pair_v1 as pairs,validation
from posteriorforge.artifacts import sha256_file
from posteriorforge.science.analytical_benchmark_v2 import array_descriptor
ROOT=Path(__file__).resolve().parents[1]
def load(path,name):
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
e=load(ROOT/'scripts/evaluate_prior_pair_v3.py','_test_v3_assessment')
SPEC=pairs.PairSpec(20261111,0)

def clone(function,namespace):
 f=FunctionType(function.__code__,namespace,function.__name__,function.__defaults__,function.__closure__)
 f.__kwdefaults__=function.__kwdefaults__;return f

def test_original_scientific_run_reference_and_G3_algebra_are_reversible_not_reimplemented():
 r=e.derivation_descriptor()
 assert r['source_display_AST_deltas_reverse_exact'] and r['run_controlflow_bytecode_equal']
 assert r['run_names_varnames_freevars_equal'] and r['run_metadata_key_only_reversible']
 assert r['original_reference_and_baseline_codeobjects_identical'] and r['old_module_globals_unmodified']
 assert not r['metric_G3_seed_or_reference_math_changed']
 assert e.run.__code__.co_code==e._ANCESTOR.run.__code__.co_code
 assert e.validate_reference_custody.__code__ is e._ANCESTOR.validate_reference_custody.__code__
 assert e.load_baseline.__code__ is e._ANCESTOR.load_baseline.__code__
 assert e.COUNT==50000 and e.FOCUS==(0,7,8,10)
 assert r['marginals']==21 and r['joint_projections']==32
 assert (r['CDF_limit'],r['mean_bias_reference_SD_limit'],r['SD_ratio_limits'],r['reference_precision_limit'])==(.10,.20,[.8,1.2],.005)
 assert kernel.source_sha256()=='e346e25bc9825abf016c2bd75addaf2b025d3e3d94e1a0ea6f6e31dd02997a43'
 assert e.SCHEMA=='posteriorforge.prior_coordinate_pair_assessment.v3'
 assert 'Policy C F64 probability' in e.LABELS[1] and 'physical F64' in e.LABELS[1]
 assert 'SEPARATE physical G3' in e.LABELS[2]


def test_missing_or_changed_wrapper_is_fail_closed_before_actual_assessment(monkeypatch):
 monkeypatch.setattr(e,'WRAPPER_SHA256',None)
 with pytest.raises(ValueError):e.source_guard(require_committed=False)


@pytest.mark.parametrize('fault',['count','cdf','bias','sd','precision','gate_callable','assessment_callable','display','schema'])
def test_source_and_original_metric_gates_cannot_be_mutated(monkeypatch,fault):
 if fault=='count':monkeypatch.setattr(e,'COUNT',49999)
 elif fault=='cdf':monkeypatch.setattr(validation,'CDF_LIMIT',.11)
 elif fault=='bias':monkeypatch.setattr(validation,'BIAS_SD_LIMIT',.21)
 elif fault=='sd':monkeypatch.setattr(validation,'SD_RATIO_LIMITS',(.7,1.3))
 elif fault=='precision':monkeypatch.setattr(validation,'REFERENCE_PRECISION_LIMIT',.006)
 elif fault=='gate_callable':monkeypatch.setattr(validation,'assess_samples',lambda *a,**k:None)
 elif fault=='assessment_callable':monkeypatch.setattr(e,'authenticate_hybrid',lambda *a:None)
 elif fault=='display':monkeypatch.setattr(e,'LABELS',('wrong',)*4)
 else:monkeypatch.setattr(e,'SCHEMA',e._ANCESTOR.SCHEMA)
 with pytest.raises(ValueError):e._check_sources()


def fixture(tmp_path):
 """Seven declared physical rows + provided Z; no kernel.refresh/normal call."""
 before=codec.decode_ordered(np.zeros((7,21),np.float32))
 for slot in kernel._geometry()['noise_indices']:
  before[::2,slot]=.01;before[1::2,slot]=.5  # original CLOSED F64 support.
 mean,cov=kernel.conditional_parameters(before)
 z=np.linspace(-1.,1.,21,dtype=np.float64).reshape(7,3)
 after=before.copy();after[:,:3]=mean+z@np.linalg.cholesky(cov).T
 np.savez(tmp_path/'samples.npz',samples=after);np.savez(tmp_path/'Z.npz',standard_normal_draws=z)
 # The real wrapper exports this exact original role; no stream is executed.
 role='prior-coordinate-pair-physical-global-block-v1'
 old=load(ROOT/'scripts/refresh_prior_pair_global_block_v1.py','_test_v3_role_ancestry')
 role=old.ROLE;seed=old.derive_seed(SPEC.context_seed,role,case=SPEC.case,repetition=0)
 state=np.random.PCG64(seed).state
 refresh={'pair_spec':SPEC.as_dict(),'seed':seed,'role':role,'bit_generator':'PCG64',
  'initial_private_state':deepcopy(state),'final_private_state':deepcopy(state),
  'standard_normal_method_calls':1,'standard_normal_draw_shape':[7,3],'standard_normal_scalar_count':21,
  'global_block_refresh_count_per_row':1,'draws_are_actual_returned_matrix_not_rerolled':True,
  'default_numpy_python_RNG_preserved':True,'default_numpy_python_RNG_before_sha256':'same','default_numpy_python_RNG_after_sha256':'same',
  'all18_physical_columns_byte_identical':True,'input_array_unchanged':True,'coordinate_decode_after_global_refresh':False,
  'standard_normal_draws':array_descriptor(z),'G3_kernel_math_seconds':.001}
 for name in ['initial_private_state','final_private_state']:
  refresh[name+'_sha256']=hashlib.sha256(json.dumps(refresh[name],sort_keys=True,separators=(',',':')).encode()).hexdigest()
 pins={'wrapper':'f'*64,'git_revision':'a'*40}
 source_checks=[]
 def verify_recorded(root,revision,sources):source_checks.append((revision,dict(sources)))
 doc={'schema':'posteriorforge.prior_coordinate_pair_global_refresh.v3','status':'complete','forensic':False,
  'original_parent_status':'complete','parent':{'parent_manifest_sha256':'parent','pair_spec':SPEC.as_dict()},
  'pair_spec':SPEC.as_dict(),'context_seed':SPEC.context_seed,'case':SPEC.case,'repetition':0,'requested_count':7,
  'source_pins':pins,'conditional_kernel':kernel.metadata(),'coordinate_decode_after_global_refresh':False,
  'samples_file':'samples.npz','samples_sha256':sha256_file(tmp_path/'samples.npz'),'samples':array_descriptor(after),
  'actual_draws_file':'Z.npz','actual_draws_sha256':sha256_file(tmp_path/'Z.npz'),'refresh':refresh}
 w=SimpleNamespace(SCHEMA=doc['schema'],kernel=kernel,ROLE=role,
  BASE=SimpleNamespace(policy_differences=old.BASE.policy_differences,immutable_pins=old.BASE.immutable_pins,verify_git_sources=verify_recorded),
  _safe_sibling=old._safe_sibling,PRODUCER_SHA256='p'*64,PAIR_SOURCE_SHA256='q'*64,PAIR_QUALIFIER_SHA256='x'*64,
  KERNEL_SHA256=kernel.source_sha256(),F64_HELPER_SHA256='y'*64,DRAW_SOURCE_SHA256='z'*64,source_checks=source_checks)
 for _,attribute in e._RECORDED_GIT_FIELDS:setattr(w,attribute,'v'*64)
 return before,after,z,doc,w

def authenticate(tmp_path,before,doc,w,*,engineering_count=7):
 path=tmp_path/'manifest.json';path.write_text(json.dumps(doc))
 namespace=dict(e.authenticate_hybrid.__globals__);namespace['COUNT']=engineering_count
 check=clone(e.authenticate_hybrid,namespace)
 assert check.__code__ is e.authenticate_hybrid.__code__ and e.COUNT==50000
 return check(path,sha256_file(path),'parent',before,w,{'wrapper':'f'*64,'git_revision':'b'*40},SPEC)


def test_saved_Z_algebra_keeps_all18_bytes_and_closed_endpoints_without_draws(tmp_path,monkeypatch):
 state=pickle.dumps((np.random.get_state(),random.getstate()),protocol=5)
 before,after,z,doc,w=fixture(tmp_path);original=before.tobytes()
 monkeypatch.setattr(kernel,'refresh',lambda *a:(_ for _ in ()).throw(AssertionError('production G3 forbidden')))
 _,actual=authenticate(tmp_path,before,doc,w)
 assert actual.tobytes()==after.tobytes() and before.tobytes()==original
 assert actual[:,:3].tobytes()!=before[:,:3].tobytes()
 assert actual[:,3:].tobytes()==before[:,3:].tobytes()
 assert {float(v) for v in before[:,kernel._geometry()['noise_indices']].flat}=={.01,.5}
 assert pickle.dumps((np.random.get_state(),random.getstate()),protocol=5)==state
 assert e.COUNT==50000
 assert len(w.source_checks)==1 and w.source_checks[0][0]=='a'*40
 recorded=w.source_checks[0][1]
 for path,attribute in e._RECORDED_GIT_FIELDS:assert recorded[path]==getattr(w,attribute)


@pytest.mark.parametrize('fault',['oldschema','failed','partialparent','forensic','parentSHA','failed01parentSHA','wrongcase','seed','role','state','RNG','physical18','kernel','source','Z','decode','redecode','timing','calltype','count','outside'])
def test_old_failed_partial_wrong_pair_G3_or_mutated_physical_rows_rejected(tmp_path,fault):
 before,after,z,doc,w=fixture(tmp_path)
 if fault=='oldschema':doc['schema']='posteriorforge.prior_coordinate_pair_global_refresh.v2'
 elif fault=='failed':doc['status']='failed'
 elif fault=='partialparent':doc['original_parent_status']='failed_or_incomplete_prior_coordinate_pair_v3_screen'
 elif fault=='forensic':doc['forensic']=True
 elif fault=='parentSHA':doc['parent']['parent_manifest_sha256']='92755b788fc894d4ca16157e573bd1b5c9d8a4468b71cf184fa16303bced7bce'
 elif fault=='failed01parentSHA':doc['parent']['parent_manifest_sha256']='25e5e9e001302627ade825a39a29231852f560bcb567b4e6a7504039eb449838'
 elif fault=='wrongcase':doc['pair_spec']['case']=1
 elif fault=='seed':doc['refresh']['seed']+=1
 elif fault=='role':doc['refresh']['role']='other'
 elif fault=='state':doc['refresh']['initial_private_state']['state']['state']+=1
 elif fault=='RNG':doc['refresh']['default_numpy_python_RNG_after_sha256']='changed'
 elif fault=='physical18':
  after[0,3]+=1.;np.savez(tmp_path/'samples.npz',samples=after)
  doc['samples_sha256']=sha256_file(tmp_path/'samples.npz');doc['samples']=array_descriptor(after)
 elif fault=='kernel':doc['conditional_kernel']['module_sha256']='wrong'
 elif fault=='source':doc['source_pins']['wrapper']='0'*64
 elif fault=='Z':doc['refresh']['standard_normal_draws']['sha256']='wrong'
 elif fault=='decode':doc['coordinate_decode_after_global_refresh']=True
 elif fault=='redecode':doc['refresh']['coordinate_decode_after_global_refresh']=True
 elif fault=='timing':doc['refresh']['G3_kernel_math_seconds']=float('nan')
 elif fault=='calltype':doc['refresh']['standard_normal_method_calls']=True
 elif fault=='count':doc['requested_count']=50000
 else:before[0,kernel._geometry()['noise_indices'][0]]=np.nextafter(.01,-np.inf)
 with pytest.raises(ValueError):authenticate(tmp_path,before,doc,w)


def test_production50000_count_is_retained_without_running_a_real_parent(tmp_path):
 before,after,z,doc,w=fixture(tmp_path);doc['requested_count']=50000
 with pytest.raises(ValueError,match='shapes/dtypes'):authenticate(tmp_path,before,doc,w,engineering_count=50000)
 assert not validation.sample_integrity(before,support_mask=np.ones(len(before),bool))['passed']


def reference_fixture():
 query=np.arange(33,dtype=np.float64).reshape(1,33);binding={'bound':'actual'}
 parent={'context_seed':SPEC.context_seed,'case':SPEC.case,'pair_spec':SPEC.as_dict(),'reference_binding':binding}
 custody={'reference_case':SPEC.case,'binding':{'observation_sha256':array_descriptor(query)['sha256']},'runner_reference_binding':binding}
 return parent,custody,query

def test_original_reference_custody_keeps_actual_pair_and_query():
 p,c,q=reference_fixture();assert e.validate_reference_custody(p,c,q,SPEC)['reference_case']==0

@pytest.mark.parametrize('fault',['case','seed','reference','query','binding'])
def test_wrong_reference_seed_case_query_or_binding_is_rejected(fault):
 p,c,q=reference_fixture()
 if fault=='case':p['case']=2
 elif fault=='seed':p['context_seed']=20261112
 elif fault=='reference':c['reference_case']=2
 elif fault=='query':q=q.copy();q[0,0]+=1.
 else:c['runner_reference_binding']={'bound':'other'}
 with pytest.raises(ValueError):e.validate_reference_custody(p,c,q,SPEC)


def test_actual_v3_wrapper_checkout_source_guard_is_constructor_free():
 """Exercise the real pinned wrapper boundary; no parent or G3 is executed."""
 before=pickle.dumps((np.random.get_state(),random.getstate()),protocol=5);modules=set(sys.modules)
 assert e.WRAPPER_SHA256 is not None
 w,producer,pair_module,pins=e.source_guard(require_committed=False)
 assert w.SCHEMA=='posteriorforge.prior_coordinate_pair_global_refresh.v3'
 assert w.COUNT==e.COUNT==50000 and w.kernel.source_sha256()==kernel.source_sha256()
 assert pins['pair_v3_G3_wrapper']==e.WRAPPER_SHA256 and pins['pair_v3_evaluator']==e.source_sha256()
 assert pins['sources_committed'] is False and pins['metric_or_G3_math_changed'] is False
 assert producer.qualification_sha256(pair_module.FIRST_PAIR)==w.PROOF_SHA256
 assert producer.source_sha256()==w.PRODUCER_SHA256
 assert w.kernel.refresh is w._KERNEL_REFRESH and w.codec.source_sha256()==w.CODEC_SHA256
 assert pickle.dumps((np.random.get_state(),random.getstate()),protocol=5)==before
 assert not ({'torch','tabpfn'} & (set(sys.modules)-modules))
