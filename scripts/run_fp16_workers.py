#!/usr/bin/env python3
"""Cold independent FP16 replicas; exactly 50,000 accepted vectors in aggregate.

Built with PriorLabs-TabPFN. Never submits jobs. Every rank receives a distinct
allocated GPU, sixteen disjoint CPUs and 50,000/world complete-vector quota.
The global FIFO lookahead remains 1,200, divided 1,200/600/300 per rank.
"""
from __future__ import annotations
import time
_PROGRAM_START=time.perf_counter()
import argparse,importlib.util,json,os,sys
from pathlib import Path
import numpy as np
from posteriorforge.artifacts import runtime_identity,sha256_file,write_json
from posteriorforge import performance_fp16_experiment as clean
from posteriorforge import performance_fp16_replica as worker

SCHEMA="posteriorforge.inference.fp16_replica_aggregate.v1"
VARIANT="execution_fp16_fifo_replicas_v1"
TOTAL_COUNT=50000
ORCHESTRATION_HELPER_SHA="2136a19ca8d8d081b03dfe0a0069c271f800a0a236fcc76bcd740ba9694365a2"


def require(value,message):
 if not value:raise RuntimeError(message)

def original_helpers():
 path=clean.repository()/"scripts/run_execution_workers.py"
 require(sha256_file(path)==ORCHESTRATION_HELPER_SHA,"frozen orchestration helpers changed")
 spec=importlib.util.spec_from_file_location('_fp16_original_orchestration',path)
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def cpu_partitions(workers,allowed,allocated):
 require(type(workers) is int and workers in (1,2,4),"one/two/four workers required")
 require(type(allocated) is int and allocated==16*workers,"exactly sixteen allocated CPUs per rank required")
 require(isinstance(allowed,list) and len(set(allowed))==len(allowed) and len(allowed)==allocated
     and all(type(i) is int and i>=0 for i in allowed),"allowed CPU affinity must match the allocated count exactly; launch in a Slurm thread-bound step")
 chosen=sorted(allowed)[:allocated]
 return [chosen[16*rank:16*(rank+1)] for rank in range(workers)]


def build_contract(*,rank,world,output,aggregate,budget,repetition,device,affinity):
 require(type(world) is int and world in (1,2,4) and type(rank) is int and 0<=rank<world,"invalid rank/world")
 return {"schema":worker.CONTRACT_SCHEMA,"benchmark_total_count":TOTAL_COUNT,"world_size":world,
  "rank":rank,"worker_count":TOTAL_COUNT//world,"parent_job_id":str(budget['job_id']),
  "budget_reservation_id":budget['reservation_id'],"context_seed":20261111,"case":0,"repetition":repetition,
  "expected_output":str(Path(output).resolve()),"aggregate_output":str(Path(aggregate).resolve()),
  "visible_device_token":device,"cpu_affinity":affinity,"worker_source_sha256":worker.source_sha256(),
  "canonical_clean_producer_sha256":worker.CLEAN_SHA,"orchestrator_source_sha256":sha256_file(__file__),
  "individual_worker_is_end_to_end":False,"query_batch_size":1024,"global_proposal_lookahead":1200,
  "worker_proposal_lookahead":1200//world,
  "seed_streams":clean.frozen_sampling().worker_stream_seeds(20261111,0,repetition,rank,world)}


def validate_workers(contracts,paths,*,bindings,budget):
 from posteriorforge.inference import support_mask
 from posteriorforge.science.analytical_benchmark_v2 import array_descriptor
 from posteriorforge.compat_cdf import adapter_source_sha256,PINNED_SDK_SHA256
 require(len(contracts)==len(paths) and len(contracts) in (1,2,4),"complete contract set required")
 world=len(contracts);arrays=[];records=[];seeds=[];payloads=set()
 require([c.get('rank') for c in contracts]==list(range(world)),"all distinct ordered ranks required")
 frozen=clean.frozen_sampling()
 for c,path in zip(contracts,paths):
  root=Path(c['expected_output']);m=json.loads((root/'manifest.json').read_text())
  expected={**bindings,"schema":worker.SCHEMA,"variant":worker.VARIANT,"status":"complete",
   "run_purpose":"performance_worker","end_to_end":False,"scientific_run":False,
   "requested_count":TOTAL_COUNT//world,"benchmark_total_count":TOTAL_COUNT,"rank":c['rank'],"world_size":world,
   "worker_contract_sha256":sha256_file(path),"runner_source_sha256":worker.source_sha256(),
   "canonical_clean_producer_sha256":worker.CLEAN_SHA,"budget_binding":budget,
   "completed_stage_indices":list(range(21)),"sources_unchanged":True,
   "approximate_experiment":True,"qualified_fp32":False,"qualification_full21":False,
   "approximation_all21_hard_screen_passed":True,"all21_strict_screen_passed":False,
   "strict_failed_stages":list(range(21)),"approximation_manifest_sha256":clean.ALL21_MANIFEST_SHA256,
   "execution_policy":clean.execution_metadata(),"proposal_lookahead":1200,"query_batch_size":1024,
   "engine_query_batch_cap":1024,"max_iter_rejection":None,"partial_vector_filtering":False,
   "clipping_or_repair_performed":False,"tail_family_changed":False,"original_gate_unchanged":True,
   "seed_streams":c['seed_streams']}
  for key,value in expected.items():require(m.get(key)==value,"worker binding differs: "+key)
  seeds.extend(c['seed_streams'].values())
  stream_map={k:v['seed'] for k,v in m['sampling_policy']['streams'].items()}
  require(stream_map==c['seed_streams'],"actual named sampling seeds differ")
  scope=frozen.proposal_buffer_scope(m,context_seed=20261111,case=0,repetition=c['repetition'],
      rank=c['rank'],world_size=world,seeds=c['seed_streams'])
  b=m['proposal_buffer'];frozen.validate_proposal_buffer_record(b,world_size=world,rank=c['rank'],scope=scope)
  require(b['worker_initial_lookahead']==1200//world,"global lookahead partition differs")
  counts=b['original_sampler_counts']
  require(m['rejection']=={'support_calls':counts['original_support_calls'],'proposed':counts['original_checked_count'],
    'accepted_before_final_trim':counts['accepted_before_trim'],'returned':counts['returned_count']},"original support/FIFO counters differ")
  require(counts['returned_count']==c['worker_count'] and m['proposal_wave_sizes']==b['actual_generated_wave_sizes'],"worker quota/waves differ")
  n=b['generated_wave_count'];e=m['execution_counters'];a=m['actual_prediction_checks'];s=m['sampling_counters']
  heads=8*s['stage0_shared_predictions']+s['component_predictions']
  require(m['expected_stage_prepare_count']==e['prepare_count']==21*n and len(e['completed_stage_scopes'])==21*n,
      "actual all21 stage coverage differs")
  require(e['actual_head_contexts']==a['actual_head_contexts']==heads==m['expected_patched_call_count'],"physical head coverage differs")
  require(sum(r['actual_head_contexts'] for r in e['completed_stage_scopes'])==heads,"per-scope physical head conservation differs")
  require(a['sampler_callback_calls']==s['stage0_shared_predictions']+s['component_predictions'],"sampler callback conservation differs")
  require(e['shared_densities']==a['shared_densities']==s['stage0_shared_predictions']==n,"fresh shared-density counts differ")
  require(e['all_stage_aliases_restored'] is True and a['original_full_API_identity_claimed'] is False,"actual FP16 scope restoration differs")
  require([r['stage_index'] for r in e['completed_stage_scopes']]==list(range(21))*n,"actual stage order differs")
  for r in e['completed_stage_scopes']:
   routes=r['hybrid_route_counters']
   require(routes.get('resident_component_predict_count',0)+routes.get('uncached_predict_count',0)==r['actual_head_contexts'],"per-scope actual route counts differ")
   report=r['precision_session']
   require(r['failed'] is False and r['prefill']['passed'] is True and report['module_sha256']==clean.PRECISION_SHA
     and report['aliases_and_flags_restored'] is True and report['counters'].get('witness_histogram_overflow',0)==0,
     "actual precision scope/witness failed")
  d=m['compatibility_adapter']
  require(d['adapter_sha256']==adapter_source_sha256() and d['sdk_source_sha256']==dict(PINNED_SDK_SHA256)
    and d['sdk_source_sha256_after']==dict(PINNED_SDK_SHA256) and d['original_alias_restored'] is True
    and d['sdk_sources_unchanged'] is True and d['sdk_aliases_unchanged'] is True
    and d['patched_call_count']==heads and d['executed_corrected_cdf_code_sha256']==d['corrected_cdf_code_sha256']
    and d['corrected_cdf_code_sha256']!=d['original_cdf_code_sha256'],"actual CDF execution/restoration failed")
  payload=root/'samples.npz';digest=sha256_file(payload)
  require(digest==m['samples_sha256'] and digest not in payloads,"worker payload hash differs or duplicate rank payload")
  payloads.add(digest)
  with np.load(payload,allow_pickle=False) as data:values=data['samples'].copy()
  require(values.shape==(c['worker_count'],21) and support_mask(values).all() and m['samples']==array_descriptor(values),
      "worker quota, physical support or array descriptor differs")
  arrays.append(values.astype(np.float64,copy=False))
  records.append({'rank':c['rank'],'manifest':str(root/'manifest.json'),'manifest_sha256':sha256_file(root/'manifest.json'),
    'samples_sha256':digest,'accepted_count':len(values),'proposal_buffer':b,'timings':m['timings'],'memory':m['memory']})
 require(len(seeds)==len(set(seeds)),"rank or role seed collision")
 result=np.concatenate(arrays,axis=0)
 require(result.shape==(TOTAL_COUNT,21) and support_mask(result).all(),"aggregate must contain exactly50000 supported physical vectors")
 return result,records


def run(*,banks,output,checkpoint,ledger,reference,reference_grid,approximation,qualification,batch_qualification,sdk_root,
        workers=1,repetition=0,timeout_seconds=780):
 from posteriorforge.reference_grid import _require_budgeted_gpu
 from posteriorforge.science.analytical_benchmark_v2 import array_descriptor
 from posteriorforge.inference_cdf import reference_binding
 start=_PROGRAM_START if __name__=='__main__' else time.perf_counter()
 h=original_helpers();counts=h.partition_counts(workers)
 require(type(repetition) is int and repetition>=0 and type(timeout_seconds) is int and timeout_seconds>0,"valid repetition/deadline required")
 require(clean.source_sha256()==worker.CLEAN_SHA,"frozen2280 source changed")
 budget=_require_budgeted_gpu(ledger);require(budget['gpus']==workers,"allocated GPU count must equal worker count")
 tokens=h.visible_device_tokens(os.environ.get('CUDA_VISIBLE_DEVICES'),workers)
 allowed_affinity=sorted(os.sched_getaffinity(0))
 cpus=cpu_partitions(workers,allowed_affinity,int(os.environ.get('SLURM_CPUS_PER_TASK','0')))
 taskset=Path('/usr/bin/taskset');require(taskset.is_file(),"explicit Linux affinity launcher required")
 root=Path(output).resolve();root.mkdir(parents=True,exist_ok=False)
 result={**runtime_identity(),'schema':SCHEMA,'variant':VARIANT,'status':'running','attribution':'Built with PriorLabs-TabPFN',
   'requested_count':TOTAL_COUNT,'worker_counts':counts,'world_size':workers,'end_to_end':True,'run_purpose':'performance_e2e_aggregate',
   'approximate_experiment':True,'qualified_fp32':False,'all21_strict_screen_passed':False,'known_physical_control_status':'model_failure',
   'budget':budget,'threads_per_worker':16,'worker_cpu_affinity':cpus,'visible_device_tokens':tokens,
   'parent_actual_cpu_affinity':allowed_affinity,'slurm_cpu_binding':{k:v for k,v in os.environ.items() if k.startswith('SLURM_CPU_BIND')},
   'affinity_protocol':'exact allocated16W mask; Slurm thread-bound step then disjoint per-rank taskset',
   'orchestrator_source_sha256':sha256_file(__file__),'worker_source_sha256':worker.source_sha256(),
   'canonical_clean_producer_sha256':worker.CLEAN_SHA,'orchestration_helper_sha256':ORCHESTRATION_HELPER_SHA,
   'repetition':repetition,'global_proposal_lookahead':1200,'worker_proposal_lookahead':[1200//workers]*workers,
   'host_stage_reuse':False,'query_batch_size':1024,'approximation_manifest_sha256':sha256_file(approximation),
   'qualification_manifest_sha256':sha256_file(qualification),'batch_qualification_sha256':sha256_file(batch_qualification),
   'aggregate_published':False,'worker_return_codes':[None]*workers}
 write_json(root/'manifest.json',result)
 try:
  bindings=h.scientific_bindings(banks,20261111,0)
  bindings.update(reference_binding=reference_binding(banks=banks,reference=reference,reference_grid=reference_grid,case=0))
  result.update(bindings)
  contracts=[];paths=[];commands=[];environments=[];logs=[]
  common={'banks':banks,'checkpoint':checkpoint,'ledger':ledger,'reference':reference,'reference-grid':reference_grid,
     'approximation':approximation,'qualification':qualification,'batch-qualification':batch_qualification,'sdk-root':sdk_root}
  for rank in range(workers):
   c=build_contract(rank=rank,world=workers,output=root/f'worker-{rank}',aggregate=root,budget=budget,repetition=repetition,
     device=tokens[rank],affinity=cpus[rank])
   path=root/'contracts'/f'worker-{rank}.json';write_json(path,c);contracts.append(c);paths.append(path)
   cmd=[str(taskset),'--cpu-list',','.join(map(str,cpus[rank])),sys.executable,'-m','posteriorforge.performance_fp16_replica']
   for key,value in common.items():cmd.extend(['--'+key,str(Path(value).resolve())])
   cmd.extend(['--output',c['expected_output'],'--worker-contract',str(path),'--repetition',str(repetition)])
   env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=tokens[rank],PF_EXECUTION_WORKER_RANK=str(rank),
       PF_EXECUTION_WORKER_WORLD_SIZE=str(workers),OMP_NUM_THREADS='16')
   commands.append(cmd);environments.append(env);logs.append((root/f'worker-{rank}.stdout.log',root/f'worker-{rank}.stderr.log'))
  result.update(worker_contracts=[{'path':str(p),'sha256':sha256_file(p)} for p in paths],worker_commands=commands,
    worker_environment=[{k:e.get(k) for k in ('CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')} for e in environments])
  write_json(root/'manifest.json',result)
  def progress(codes):result.update(worker_return_codes=codes);write_json(root/'manifest.json',result)
  h.execute_workers(commands,environments,logs,timeout_seconds=timeout_seconds,on_progress=progress,cwd=clean.repository())
  values,records=validate_workers(contracts,paths,bindings=bindings,budget=budget)
  generated=sum(r['proposal_buffer']['generated_count'] for r in records)
  checked=sum(r['proposal_buffer']['original_sampler_counts']['original_checked_count'] for r in records)
  unused=sum(r['proposal_buffer']['remaining_unused_buffer_count'] for r in records)
  require(generated==checked+unused,"aggregate complete-vector FIFO conservation differs")
  require(sha256_file(__file__)==result['orchestrator_source_sha256'] and worker.source_sha256()==result['worker_source_sha256']
    and clean.source_sha256()==worker.CLEAN_SHA and original_helpers() is not None,"execution source changed")
  for path,key in ((approximation,'approximation_manifest_sha256'),(qualification,'qualification_manifest_sha256'),(batch_qualification,'batch_qualification_sha256')):
   require(sha256_file(path)==result[key],"ancestry artifact changed during aggregate")
  with (root/'samples.npz').open('xb') as stream:np.savez(stream,samples=values)
  elapsed=time.perf_counter()-start
  result.update(status='complete',returned_count=len(values),samples=array_descriptor(values),samples_file='samples.npz',
    samples_sha256=sha256_file(root/'samples.npz'),workers=records,aggregate_published=True,
    proposal_buffer={'global_initial_lookahead':1200,'worker_initial_lookahead':[1200//workers]*workers,
      'generated_count':generated,'original_checked_count':checked,'remaining_unused_buffer_count':unused,'returned_count':len(values)},
    timings={'end_to_end_seconds':elapsed,'accepted_samples_per_second':TOTAL_COUNT/elapsed,
      'includes_parent_auth_child_startup_all_workers_aggregation_payload_write_and_hash':True})
  write_json(root/'manifest.json',result);return result
 except BaseException as error:
  result.update(status='failed',error_type=type(error).__name__,error=str(error),elapsed_seconds=time.perf_counter()-start,aggregate_published=False)
  write_json(root/'manifest.json',result);raise


if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__)
 for key in ('banks','output','checkpoint','ledger','reference','reference-grid','approximation','qualification','batch-qualification','sdk-root'):
  parser.add_argument('--'+key,required=True)
 parser.add_argument('--workers',type=int,choices=[1,2,4],default=1)
 parser.add_argument('--repetition',type=int,default=0);parser.add_argument('--timeout-seconds',type=int,default=780)
 m=run(**vars(parser.parse_args()));print(json.dumps({k:m.get(k) for k in ('schema','status','requested_count','world_size','timings')},indent=2))
