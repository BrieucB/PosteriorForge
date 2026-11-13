"""Offline saved-trace summary only; no Torch/model/sampling invocation."""
import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
from types import FunctionType,SimpleNamespace

ROOT=Path('/home/it4i-bbenvegnen/workspace/PosteriorForge')
PINS={'diagnose_runtime.py':'1852980b507649ecf960d0f84b58d15e24073639c7482cb7a433e4e17b51139b',
      'profile_fp16_posterior.py':'328def203b34c72d64c21a8a54bb77d734aaf7b0e8786408e1d79232046d0dfb',
      'diagnose_prefill_allocator.py':'4707b75c3866d31688031da704eeeffe5ab8fed9a2dd85195626028a0c08b021'}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(name):
 p=ROOT/'scripts'/name
 if sha(p)!=PINS[name]:raise ValueError('frozen parser source differs: '+name)
 spec=importlib.util.spec_from_file_location('_tail_offline_'+p.stem,p);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);return m

def operation(name):
 parts=name.split('/')
 if len(parts)==4 and parts[0]=='pf-tail' and parts[-1] in ('fit','prepare'):
  return 'stage'+parts[1]+'_'+parts[2]+'_'+parts[3]
 if len(parts)==5 and parts[0]=='pf-tail' and parts[3].isdigit() and parts[4] in ('N1024','N1'):
  return 'stage'+parts[1]+'_'+parts[2]+'_'+parts[4]
 return None

def run(run_path,expected_sha,output):
 started=time.perf_counter();run_path=Path(run_path);manifest=run_path/'manifest.json'
 if sha(manifest)!=expected_sha:raise ValueError('closed capture manifest differs')
 d=json.loads(manifest.read_bytes());trace=run_path/d['profile']['file']
 if d['status']!='two_stage_conditional_capture_complete' or sha(trace)!=d['profile']['sha256']:raise ValueError('complete capture and exact raw trace required')
 reader=load('diagnose_runtime.py');profile=load('profile_fp16_posterior.py')
 if sha(ROOT/'scripts/diagnose_prefill_allocator.py')!=PINS['diagnose_prefill_allocator.py']:raise ValueError('host-only policy source differs')
 original=reader.iter_trace_events;left=float('inf');right=float('-inf');annotations=[];owner=None
 for e in original(trace):
  if e.get('ph')!='X' or not isinstance(e.get('dur'),(int,float)):continue
  left=min(left,e['ts']);right=max(right,e['ts']+e['dur'])
  op=operation(e.get('name','')) if e.get('cat')=='user_annotation' else None
  if op:
   key=(e.get('pid'),e.get('tid'));owner=owner or key
   if key!=owner:raise ValueError('pf-tail annotations span multiple owner CPU threads')
   annotations.append({'operation':op,'original_name':e['name'],'start_us':e['ts'],'end_us':e['ts']+e['dur'],'host_seconds':e['dur']/1e6})
 if owner is None or not right>left:raise ValueError('owner ranges/trace extent absent')
 boundary={'ph':'X','cat':'user_annotation','name':'pfdiagnostic/profile/request','pid':owner[0],'tid':owner[1],'ts':left,'dur':right-left,'args':{'scope':'virtual entire profiler trace extent; not an actual request annotation'}}
 def events(path):
  for event in original(path):
   name=event.get('name','');op=operation(name)
   if op:
    event={**event,'name':('pfdiagnostic/profile/'+op if event.get('cat')=='user_annotation' else 'tail_gpu_range/'+name)}
   yield event
  yield boundary
 virtual_reader=SimpleNamespace(iter_trace_events=events)
 ns=dict(reader.__dict__);ns['iter_trace_events']=events
 summarize=FunctionType(reader.summarize_trace.__code__,ns)
 overall=summarize(trace)
 ns2=dict(profile.trace_attribution.__globals__);ns2['PRIORITY']=tuple(sorted({a['operation'] for a in annotations}));ns2['BOUNDARY']='pfdiagnostic/profile/request'
 attribute=FunctionType(profile.trace_attribution.__code__,ns2)
 groups=attribute(trace,virtual_reader)
 union=profile.union_seconds
 host_union=union([(a['start_us']/1e6,a['end_us']/1e6) for a in annotations]);extent=(right-left)/1e6
 timings=[]
 for row in d['stages']:
  stage=row['fixture']['stage']
  timings += [{'stage':stage,'arm':'original','kind':'fit',**row['original_fit_timing']},
              {'stage':stage,'arm':'candidate','kind':'prepare',**row['candidate_prepare']['timing']}]
  for arm,entries in row['arms'].items():
   timings.extend({'stage':stage,'arm':arm,'kind':e['label'],'member':e['member'],**e['timing']} for e in entries)
 sums=defaultdict(float)
 for row in timings:sums[row['arm']+'_'+row['kind']]+=row['host_wall_seconds']
 report={'schema':'posteriorforge.prior_tail_saved_trace_timing.v1','capture_manifest_sha256':expected_sha,
  'trace_sha256':sha(trace),'trace_bytes':trace.stat().st_size,'source_pins':PINS,'offline_script_sha256':sha(__file__),
  'virtual_boundary_scope':'min/max complete-event extent of entire saved profiler session, including guard/ICDF/capture work; not an actual timed request or invented raw event',
  'raw_trace_modified':False,'GPU_range_duplicates_not_host_boundaries':True,
  'owner_host_annotations':annotations,'recorded_host_timings':timings,'recorded_host_timings_summed_by_kind':dict(sums),
  'host_neural_annotation_union_seconds':host_union,'profile_extent_seconds':extent,
  'unbracketed_host_extent_seconds':max(0.,extent-host_union),
  'unbracketed_scope':'combined ICDF/observer, weight/normalizer/target guards, serialization/I-O, Python waits and cleanup; individual ICDF versus capture split is unavailable',
  'overall_GPU_sync_summary':overall,'launch_correlation_attribution':groups,
  'capture_elapsed_seconds':d['elapsed_seconds'],'trace_export_seconds':d['profile']['export_seconds'],
  'no_clean_request_speed_or_50k_extrapolation':True,'no_Torch_imported':'torch' not in sys.modules,
  'cautions':['Host neural wall includes waits/copies; it is not CPU compute.','CUDA event spans are not exclusive kernel time.','Kernel/copy/sync unions and per-kind unions may overlap; do not add them.','Unmatched/conflicting correlations are retained, not silently attributed.','SDPA generic/nested operator counts are not exclusive backend-call totals; flags alone do not prove Flash.','Copy bytes missing in args are unknown, not proof of zero bytes.'],
  'offline_wall_seconds':time.perf_counter()-started}
 if not report['no_Torch_imported']:raise ValueError('offline parser imported Torch unexpectedly')
 output=Path(output);output.mkdir(parents=True,exist_ok=False);(output/'summary.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'output':str(output),'summary_sha256':sha(output/'summary.json'),'trace_sha256':report['trace_sha256'],
  'kernel_union_seconds':overall['cuda_kernel_interval_union_seconds'],'copy_union_seconds':overall['cuda_copy_interval_union_seconds'],
  'kernel_copy_union_seconds':overall['cuda_kernel_and_copy_interval_union_seconds'],'sync_host_union_seconds':overall['cuda_runtime_sync_interval_union_seconds'],
  'profile_extent_seconds':extent,'host_neural_annotation_union_seconds':host_union,'unbracketed_host_extent_seconds':report['unbracketed_host_extent_seconds'],
  'recorded_host_timings_summed_by_kind':dict(sums),'correlation_conflicts':groups['correlation_conflicts'],'groups':groups['groups'],
  'no_Torch_imported':report['no_Torch_imported'],'offline_wall_seconds':report['offline_wall_seconds']},sort_keys=True))

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-path',required=True);p.add_argument('--expected-sha',required=True);p.add_argument('--output',required=True);run(**vars(p.parse_args()))
