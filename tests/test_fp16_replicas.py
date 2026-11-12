"""CPU fake contracts/cohorts only; no model or posterior generation."""
import ast,copy,importlib.util,inspect,json,os,textwrap
from pathlib import Path
import numpy as np
import pytest
from posteriorforge import performance_fp16_experiment as clean
from posteriorforge import performance_fp16_replica as worker
from posteriorforge.artifacts import sha256_file,write_json
from posteriorforge.compat_cdf import adapter_source_sha256,PINNED_SDK_SHA256
from posteriorforge.science.analytical_benchmark_v2 import array_descriptor,build_problem
spec=importlib.util.spec_from_file_location('fp16_workers',Path(__file__).parents[1]/'scripts/run_fp16_workers.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
BUDGET={'job_id':'123','reservation_id':'test-reservation','gpus':2}

@pytest.mark.parametrize('world',[1,2,4])
def test_contract_partitions_fixed_total_and_distinct_rank_role_seeds(tmp_path,world):
 parts=a.cpu_partitions(world,list(range(16*world)),16*world)
 assert len(set(sum(parts,[])))==16*world
 cs=[a.build_contract(rank=r,world=world,output=tmp_path/f'r{r}',aggregate=tmp_path,budget={**BUDGET,'gpus':world},
   repetition=2,device=str(r),affinity=parts[r]) for r in range(world)]
 assert sum(c['worker_count'] for c in cs)==50000
 assert sum(c['worker_proposal_lookahead'] for c in cs)==1200
 assert len({s for c in cs for s in c['seed_streams'].values()})==3*world
 assert all(c['individual_worker_is_end_to_end'] is False for c in cs)

@pytest.mark.parametrize('world,allowed,allocated',[(3,list(range(64)),48),(2,list(range(31)),32),(2,[0]*32,32),(2,list(range(64)),64),(2,list(range(64)),32),(True,list(range(16)),16)])
def test_affinity_or_world_mismatch_refused(world,allowed,allocated):
 with pytest.raises(RuntimeError):a.cpu_partitions(world,allowed,allocated)


def setup_contract(tmp_path,monkeypatch):
 c=a.build_contract(rank=1,world=2,output=tmp_path/'rank1',aggregate=tmp_path,budget=BUDGET,repetition=0,device='GPU-r1',affinity=list(range(16,32)))
 for k,v in {'PF_EXECUTION_WORKER_RANK':'1','PF_EXECUTION_WORKER_WORLD_SIZE':'2','CUDA_VISIBLE_DEVICES':'GPU-r1','OMP_NUM_THREADS':'16'}.items():monkeypatch.setenv(k,v)
 monkeypatch.setattr(os,'sched_getaffinity',lambda pid:set(range(16,32)))
 return c


def test_worker_contract_binds_budget_source_device_affinity_and_quota(tmp_path,monkeypatch):
 c=setup_contract(tmp_path,monkeypatch)
 assert worker.validate_contract(c,budget=BUDGET,output=tmp_path/'rank1',repetition=0)==(1,2,25000)

@pytest.mark.parametrize('field,value',[('worker_count',50000),('benchmark_total_count',25000),('world_size',True),('rank',2),
 ('budget_reservation_id','other'),('worker_source_sha256','bad'),('orchestrator_source_sha256','bad'),('repetition',1),
 ('worker_proposal_lookahead',1200),('individual_worker_is_end_to_end',True),('cpu_affinity',list(range(16)))])
def test_misbound_worker_contract_cannot_enter_model(tmp_path,monkeypatch,field,value):
 c=setup_contract(tmp_path,monkeypatch);c[field]=value
 with pytest.raises(RuntimeError):worker.validate_contract(c,budget=BUDGET,output=tmp_path/'rank1',repetition=0)


def payloads(tmp_path,world=2):
 budget={**BUDGET,'gpus':world};cs=[];paths=[]
 problem=build_problem();noise=[problem.parameter_order.index(n) for n in problem.prior.noise_names]
 bindings={'context_seed':20261111,'case':0,'repetition':0,'query':{'sha256':'query'},'context_theta':{'sha256':'theta'},
  'context_obs':{'sha256':'obs'},'scientific_fingerprint':'science','checkpoint_sha256':'checkpoint'}
 frozen=clean.frozen_sampling()
 for rank in range(world):
  root=tmp_path/f'worker-{rank}';root.mkdir();quota=50000//world
  c=a.build_contract(rank=rank,world=world,output=root,aggregate=tmp_path,budget=budget,repetition=0,device=str(rank),affinity=list(range(16*rank,16*(rank+1))))
  path=tmp_path/f'contract-{rank}.json';write_json(path,c);cs.append(c);paths.append(path)
  v=np.zeros((quota,21),dtype=np.float64);v[:,noise]=.2;v[:,0]=rank*.01
  with (root/'samples.npz').open('wb') as stream:np.savez(stream,samples=v)
  scope=frozen.proposal_buffer_scope(bindings,context_seed=20261111,case=0,repetition=0,rank=rank,world_size=world,seeds=c['seed_streams'])
  fifo=frozen.guarded_proposal_buffer()
  buffer=fifo.ProposalBuffer(lambda n:(np.zeros((n,21),np.float32),None),scope=scope,concatenate=np.concatenate,world_size=world,rank=rank)
  buffer(quota);buffer(100)
  b=buffer.finalize(original_checked_count=quota+100,accepted_before_trim=quota+50,returned_count=quota,original_support_calls=2)
  b.update(enabled=True,source_unchanged=True,policy_sha256=frozen.proposal_buffer_policy(1200)['policy_sha256'])
  stages=[{'stage_index':j,'actual_head_contexts':8 if j==0 else 1,
    'hybrid_route_counters':{'resident_component_predict_count':8 if j==0 else 1,'uncached_predict_count':0},'failed':False,'prefill':{'passed':True},'precision_session':{'module_sha256':clean.PRECISION_SHA,
    'aliases_and_flags_restored':True,'counters':{'witness_histogram_overflow':0}}} for j in range(21)]
  d={'adapter_sha256':adapter_source_sha256(),'sdk_source_sha256':dict(PINNED_SDK_SHA256),'sdk_source_sha256_after':dict(PINNED_SDK_SHA256),
    'original_alias_restored':True,'sdk_sources_unchanged':True,'sdk_aliases_unchanged':True,'patched_call_count':28,
    'executed_corrected_cdf_code_sha256':'corrected','corrected_cdf_code_sha256':'corrected','original_cdf_code_sha256':'original'}
  m={**bindings,'schema':worker.SCHEMA,'variant':worker.VARIANT,'status':'complete','run_purpose':'performance_worker',
   'end_to_end':False,'scientific_run':False,'requested_count':quota,'benchmark_total_count':50000,'rank':rank,'world_size':world,
   'worker_contract_sha256':sha256_file(path),'runner_source_sha256':worker.source_sha256(),'canonical_clean_producer_sha256':worker.CLEAN_SHA,
   'budget_binding':budget,'completed_stage_indices':list(range(21)),'sources_unchanged':True,'approximate_experiment':True,
   'qualified_fp32':False,'qualification_full21':False,'approximation_all21_hard_screen_passed':True,'all21_strict_screen_passed':False,
   'strict_failed_stages':list(range(21)),'approximation_manifest_sha256':clean.ALL21_MANIFEST_SHA256,'execution_policy':clean.execution_metadata(),
   'proposal_lookahead':1200,'query_batch_size':1024,'engine_query_batch_cap':1024,'max_iter_rejection':None,'partial_vector_filtering':False,
   'clipping_or_repair_performed':False,'tail_family_changed':False,'original_gate_unchanged':True,'seed_streams':c['seed_streams'],
   'sampling_policy':{'streams':{k:{'seed':s} for k,s in c['seed_streams'].items()}},'proposal_buffer':b,
   'rejection':{'support_calls':2,'proposed':quota+100,'accepted_before_final_trim':quota+50,'returned':quota},
   'proposal_wave_sizes':[quota+1200//world],'expected_stage_prepare_count':21,
   'execution_counters':{'prepare_count':21,'completed_stage_scopes':stages,'actual_head_contexts':28,'shared_densities':1,'all_stage_aliases_restored':True},
   'actual_prediction_checks':{'sampler_callback_calls':21,'actual_head_contexts':28,'shared_densities':1,'original_full_API_identity_claimed':False},
   'sampling_counters':{'stage0_shared_predictions':1,'component_predictions':20},'expected_patched_call_count':28,
   'compatibility_adapter':d,'samples_sha256':sha256_file(root/'samples.npz'),'samples':array_descriptor(v),'timings':{},'memory':{}}
  write_json(root/'manifest.json',m)
 return cs,paths,bindings,budget

@pytest.mark.parametrize('world',[1,2,4])
def test_complete_fake_aggregate_is_exactly50000_without_resampling(tmp_path,world):
 cs,paths,b,budget=payloads(tmp_path,world)
 values,records=a.validate_workers(cs,paths,bindings=b,budget=budget)
 assert values.shape==(50000,21) and len(records)==world
 for rank in range(world):np.testing.assert_array_equal(values[rank*50000//world:(rank+1)*50000//world,0],rank*.01)

@pytest.mark.parametrize('field,value',[('status','failed'),('requested_count',50000),('end_to_end',True),('qualified_fp32',True),
 ('all21_strict_screen_passed',True),('runner_source_sha256','bad'),('samples_sha256','bad'),('completed_stage_indices',list(range(20)))])
def test_incomplete_or_mislabelled_rank_prevents_aggregate(tmp_path,field,value):
 cs,paths,b,budget=payloads(tmp_path)
 p=Path(cs[0]['expected_output'])/'manifest.json';m=json.loads(p.read_text());m[field]=value;write_json(p,m)
 with pytest.raises((RuntimeError,ValueError)):a.validate_workers(cs,paths,bindings=b,budget=budget)
 assert not (tmp_path/'samples.npz').exists()

@pytest.mark.parametrize('mutation',['cdf','precision','fifo','quota','unsupported','duplicate_rank','source_contract','scope_heads','callbacks','route_counts'])
def test_actual_counter_and_payload_faults_prevent_aggregate(tmp_path,mutation):
 cs,paths,b,budget=payloads(tmp_path);root=Path(cs[0]['expected_output']);p=root/'manifest.json';m=json.loads(p.read_text())
 if mutation=='cdf':m['compatibility_adapter']['patched_call_count']-=1
 elif mutation=='precision':m['execution_counters']['completed_stage_scopes'][-1]['precision_session']['aliases_and_flags_restored']=False
 elif mutation=='fifo':m['proposal_buffer']['worker_initial_lookahead']=1200
 elif mutation=='scope_heads':m['execution_counters']['completed_stage_scopes'][0]['actual_head_contexts']-=1
 elif mutation=='callbacks':m['actual_prediction_checks']['sampler_callback_calls']+=1
 elif mutation=='route_counts':m['execution_counters']['completed_stage_scopes'][0]['hybrid_route_counters']['resident_component_predict_count']-=1
 elif mutation=='duplicate_rank':cs[1]['rank']=0
 elif mutation=='source_contract':write_json(paths[0],{'changed':True})
 else:
  with np.load(root/'samples.npz') as d:v=d['samples'].copy()
  if mutation=='quota':v=v[:-1]
  else:v[:,-1]=-1
  with (root/'samples.npz').open('wb') as stream:np.savez(stream,samples=v)
  m['samples_sha256']=sha256_file(root/'samples.npz');m['samples']=array_descriptor(v)
 write_json(p,m)
 with pytest.raises((RuntimeError,ValueError)):a.validate_workers(cs,paths,bindings=b,budget=budget)


def test_computational_functions_are_frozen_objects():
 assert worker.frozen_sampling is clean.frozen_sampling
 assert worker.precision_engine_scope is clean.precision_engine_scope
 assert worker.validate_prediction is clean.validate_prediction
 assert a.original_helpers().partition_counts(4)==[12500]*4
 tree=ast.parse(textwrap.dedent(inspect.getsource(worker.run_worker)))
 calls={ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call)}
 assert {'frozen.stage_major_wave','frozen.component_stage_sampler','accept_reject','buffer.finalize'}<=calls
 assert clean.source_sha256()==worker.CLEAN_SHA



def test_actual_worker_sampling_policy_constructs_without_model_or_GPU():
 from types import SimpleNamespace
 from posteriorforge import execution_mixture as mixture
 tree=ast.parse(textwrap.dedent(inspect.getsource(worker.run_worker)))
 fields=[n.value for n in ast.walk(tree) if isinstance(n,ast.keyword) and n.arg=='sampling_policy']
 assert len(fields)==1
 expr=ast.Expression(fields[0]);ast.fix_missing_locations(expr)
 actual_all21=clean.load_script('characterize_fp16_all_stages.py',clean.ALL21_SOURCE_SHA)
 frozen=clean.frozen_sampling()
 seeds=frozen.worker_stream_seeds(20261111,0,0,0,1)
 streams={k:{'role':frozen.STREAM_ROLES[k],'seed':v,'dtype':frozen.STREAM_DTYPES[k],
  'device':'cpu','generator':'torch.Generator'} for k,v in seeds.items()}
 namespace={**vars(worker),'mixture':mixture,'all21':actual_all21,'streams':streams}
 result=eval(compile(expr,'<actual-worker-sampling-policy>','eval'),namespace)
 assert result['forward_precision_policy']==clean.MODE
 assert result['streams']==streams and result['stage0_dedup']['policy']==actual_all21.STAGE0_POLICY


def test_worker_function_has_no_unresolved_module_globals():
 import symtable,builtins
 tree=symtable.symtable(Path(worker.__file__).read_text(),worker.__file__,'exec')
 scope=next(child for child in tree.get_children() if child.get_name()=='run_worker')
 missing=[s.get_name() for s in scope.get_symbols() if s.is_referenced() and s.is_global()
   and s.get_name() not in vars(worker) and not hasattr(builtins,s.get_name())]
 assert missing==[]
