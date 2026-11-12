"""One pending request in the predefined pure-SS15 scientific diagnostic grid.

Built with PriorLabs-TabPFN. Run from the reviewed source checkout with
PYTHONPATH=src; the frozen pilot's source-checkout helper dependency is retained.
This runner changes request routing and provenance only. The case0/seed20261111
finite conditional screen is an explicitly separate anchor, not qualification
of a new query/context. All real requests return exactly50,000 complete vectors.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from importlib import import_module
from pathlib import Path

import numpy as np

from . import performance_ss15 as pilot
from . import execution_ss15_sampling as wave
from . import representation_ss15 as representation
from .artifacts import runtime_identity, sha256_file, write_json
from .banks import load_context, load_heldout
from .inference import CHECKPOINT_SHA256
from .science.analytical_benchmark_v2 import array_descriptor, build_problem

SCHEMA = "posteriorforge.inference.ss15_campaign.v1"
VARIANT = "accuracy_ss15_fp16_fifo_campaign_v1"
PILOT_SHA256 = "4099da3c19366997ab0800e5430b5bb1c129b4b3e535f678b3d4652233303480"
COUNT = 50000
CONTEXT_SIZE = 4096
CONTEXT_SEEDS = (20261111, 20261112, 20261113)
CASES = tuple(range(8))
ANCHOR = (20261111, 0)
PREDEFINED_GRID = tuple((seed, case) for seed in CONTEXT_SEEDS for case in CASES)
PENDING_GRID = tuple(point for point in PREDEFINED_GRID if point != ANCHOR)
MODE = pilot.MODE
REPRESENTATION_SHA256 = pilot.REPRESENTATION_SHA256
SS15_EXECUTION_SHA256 = pilot.SS15_EXECUTION_SHA256
CONDITIONAL_QUALIFICATION_SHA256 = pilot.CONDITIONAL_QUALIFICATION_SHA256
require = pilot.require

_FUNCTION_NAMES = ("require_sources", "engine_scope", "model_bindings",
                   "validate_prediction", "qualification_receipt",
                   "execution_metadata", "rng_hashes", "is_fallback",
                   "prefill_witness_passed", "query_witness_delta")
_FUNCTION_SEALS = {name: (getattr(pilot, name), getattr(pilot, name).__code__)
                   for name in _FUNCTION_NAMES}
_ADAPTER_CLASS = pilot.SS15StageAdapter
_ADAPTER_SEALS = {name: (getattr(_ADAPTER_CLASS, name), getattr(_ADAPTER_CLASS, name).__code__)
                  for name in ("__init__", "prepare", "predict_component", "predict",
                               "inspect_cache", "close", "metadata")}


def source_sha256():
    return sha256_file(__file__)


def require_sources():
    require(pilot.source_sha256() == PILOT_SHA256, "frozen4099 SS15 pilot source changed")
    require(pilot.SS15StageAdapter is _ADAPTER_CLASS, "frozen SS15 adapter class replaced")
    for name, (function, code) in _FUNCTION_SEALS.items():
        require(getattr(pilot, name) is function and function.__code__ is code,
                "frozen SS15 function changed: " + name)
    for name, (function, code) in _ADAPTER_SEALS.items():
        require(getattr(_ADAPTER_CLASS, name) is function and function.__code__ is code,
                "frozen SS15 adapter method changed: " + name)
    pilot.require_sources()
    require(pilot.FALLBACK_MAP == {10: [5], 20: [4]}
            and pilot.wave is wave and pilot.representation is representation,
            "frozen SS15 fallback/feature/wave policy changed")
    require(pilot.CONDITIONAL_QUALIFICATION_SHA256 == CONDITIONAL_QUALIFICATION_SHA256
            and pilot.COUNT == COUNT and pilot.MODE == MODE,
            "frozen SS15 source/policy constant changed")


def request_spec(*, context_seed, case, context_size=CONTEXT_SIZE, count=COUNT, repetition=0):
    require(type(context_seed) is int and context_seed in CONTEXT_SEEDS,
            "context_seed must be one of the fixed three diagnostic seeds")
    require(type(case) is int and case in CASES, "case must be one of the fixed eight heldout cases")
    require(type(context_size) is int and context_size == CONTEXT_SIZE,
            "campaign context_size is exactly4096")
    require(type(count) is int and count == COUNT, "campaign requests exactly50000 complete accepted vectors")
    require(type(repetition) is int and repetition == 0, "campaign uses repetition0 only")
    require((context_seed, case) in PENDING_GRID,
            "case0/seed20261111 already exists as the frozen pilot; do not rerun it")
    return {"context_seed": context_seed, "case": case, "context_size": context_size,
            "requested_count": count, "repetition": repetition}


def raw_bindings(context, raw_query):
    problem = build_problem()
    return {"checkpoint_sha256": CHECKPOINT_SHA256,
            "scientific_fingerprint": problem.scientific_fingerprint,
            "parameter_order": list(problem.parameter_order),
            "context_theta": array_descriptor(context.theta),
            "context_obs": array_descriptor(context.obs), "query": array_descriptor(raw_query)}


def input_bundle(banks, context_seed, case):
    """Load the actual saved bank/query and use the exact4099 SS15 feature helper."""
    context = load_context(banks, context_seed, CONTEXT_SIZE)
    raw_query = np.asarray(load_heldout(banks, case).obs).reshape(1, 33)
    theta, obs, query, model, transformed = pilot.model_bindings(context, raw_query)
    return {"context": context, "raw_query": raw_query, "theta": theta, "obs": obs,
            "query": query, "raw_bindings": raw_bindings(context, raw_query),
            "model_inputs": model, "transformed_bindings": transformed}


def conditional_anchor(banks, qualification):
    """Replay the fixed case0/seed20261111 artifact against ITS OWN loaded inputs."""
    anchor = input_bundle(banks, *ANCHOR)
    document, receipt = pilot.qualification_receipt(
        qualification, raw_bindings=anchor["raw_bindings"],
        transformed_bindings=anchor["transformed_bindings"])
    return {
        "context_seed": ANCHOR[0], "case": ANCHOR[1], "context_size": CONTEXT_SIZE,
        "qualification_sha256": CONDITIONAL_QUALIFICATION_SHA256,
        "raw_scientific_bindings": anchor["raw_bindings"],
        "model_inputs": anchor["model_inputs"],
        "transformed_bindings": anchor["transformed_bindings"], "replay": receipt,
        "matching_FP32_execution_passed_all21": document["matching_FP32_execution_passed_all21"],
        "FP16_hard_validity_passed_all21": document["FP16_hard_validity_passed_all21"],
        "FP16_strict_failed_stages": document["FP16_strict_failed_stages"],
        "FP16_matching_uncached_strict_failed_stages":
            document["FP16_matching_uncached_strict_failed_stages"],
        "scope": "finite fixed-prefix conditional anchor at case0/seed20261111 only",
        "qualifies_other_queries_or_contexts": False,
    }


def numerical_scope(actual, anchor):
    return {
        "actual_request_fixed_conditional_screen_performed": False,
        "anchor_qualifies_actual_request": False,
        "same_context_as_anchor": all(actual["raw_bindings"][key] ==
                                      anchor["raw_scientific_bindings"][key]
                                      for key in ("context_theta", "context_obs")),
        "same_query_as_anchor": actual["raw_bindings"]["query"] ==
                                anchor["raw_scientific_bindings"]["query"],
        "scope": "actual runtime source/dtype/mass/route/RNG/CDF guards; separate finite anchor",
        "posterior_accuracy_claimed": False,
    }


def execution_metadata():
    require_sources()
    return {"schema": "posteriorforge.ss15_campaign_execution_policy.v1",
            "module_sha256": source_sha256(), "frozen_pilot_source_sha256": PILOT_SHA256,
            "engine_policy": pilot.execution_metadata(),
            "context_seeds": list(CONTEXT_SEEDS), "cases": list(CASES),
            "total_predefined_grid_points": len(PREDEFINED_GRID),
            "pending_new_grid_points": len(PENDING_GRID),
            "existing_pilot_point": {"context_seed": ANCHOR[0], "case": ANCHOR[1]},
            "count_per_request": COUNT, "context_size": CONTEXT_SIZE, "repetition": 0,
            "requires_full_reviewed_source_checkout": True,
            "qualification_scope": "fixed case0/seed20261111 finite conditional anchor only",
            "new_case_numerical_qualification_claimed": False,
            "attribution": "Built with PriorLabs-TabPFN"}

def run(*,banks,output,checkpoint,ledger,reference,reference_grid,qualification,context_seed,case,context_size=CONTEXT_SIZE,count=COUNT,repetition=0):
 start=time.perf_counter();output=Path(output);require(not output.exists(),'fresh SS15 campaign output required')
 require_sources();request_spec(context_seed=context_seed,case=case,context_size=context_size,count=count,repetition=repetition)
 require(isinstance(CONDITIONAL_QUALIFICATION_SHA256,str),'successful fixed conditional anchor must be pinned before launch')
 from .inference_cdf import guard_baseline_source,reference_binding
 from .inference import CHECKPOINT_SHA256,frozen_regressor_settings,support_mask,_prior
 from .banks import load_context,load_heldout
 from .science.analytical_benchmark_v2 import array_descriptor,build_problem
 from .reference_grid import _require_budgeted_gpu
 frozen=pilot.clean.frozen_sampling();initial_source=source_sha256();baseline_source=guard_baseline_source()
 require(sha256_file(checkpoint)==CHECKPOINT_SHA256,'immutable checkpoint differs')
 actual_inputs=input_bundle(banks,context_seed,case)
 context=actual_inputs['context'];raw_query=actual_inputs['raw_query'];bindings=actual_inputs['raw_bindings']
 theta=actual_inputs['theta'];obs=actual_inputs['obs'];query=actual_inputs['query']
 transformed=actual_inputs['model_inputs'];qualification_inputs=actual_inputs['transformed_bindings']
 anchor=conditional_anchor(banks,qualification);scope_claim=numerical_scope(actual_inputs,anchor)
 refs=reference_binding(banks=banks,reference=reference,reference_grid=reference_grid,case=case)
 budget=_require_budgeted_gpu(ledger);require(budget['gpus']==1,'single GPU scientific SS15 campaign request only')
 output.mkdir(parents=True,exist_ok=False);engine=None;descriptor=None;buffer=None
 status={**runtime_identity(),'schema':SCHEMA,'variant':VARIANT,'status':'running','attribution':'Built with PriorLabs-TabPFN',
  'runner_source_sha256':initial_source,'execution_policy':execution_metadata(),'representation':representation.representation_metadata(),
  'model_inputs':transformed,'qualification_transformed_bindings':qualification_inputs,'raw_scientific_bindings':bindings,'raw33_numerical_qualification_claimed':False,
  'conditional_qualification_sha256':CONDITIONAL_QUALIFICATION_SHA256,
  'conditional_anchor':anchor,'actual_request_numerical_scope':scope_claim,
  'frozen_pilot_source_sha256':PILOT_SHA256,'run_purpose':'predefined24_scientific_diagnostic_grid',
  'actual_request_fixed_conditional_screen_performed':False,
  'approximate_experiment':True,'qualified_FP32':False,'requested_count':COUNT,'benchmark_total_count':COUNT,
  'context_seed':context_seed,'context_size':CONTEXT_SIZE,'case':case,'repetition':repetition,'rank':0,'world_size':1,'end_to_end':True,'scientific_run':True,
  'observation_preprocessing':'source-pinned exact SS15 from original Float64 raw33; one final Float32 cast',
  'context_filter':'no_filtering','physical_parameter_order_unchanged':True,'all_three_globals_condition_local_stages':True,
  'sampler_math_changed':False,'tail_family_changed':False,'clipping_or_repair_performed':False,'partial_vector_filtering':False,
  'original_gate_unchanged':True,'absolute_accuracy_gate_required':True,'candidate_physical_assessment':'pending original reference assessment',
  'proposal_buffer_policy':frozen.proposal_buffer_policy(1200),'proposal_lookahead':1200,'proposal_buffer':None,
  'query_batch_size':1024,'engine_query_batch_cap':1024,'max_iter_rejection':None,'microbatch':COUNT,'log_density':'not_used',
  'budget_binding':budget,'reference_binding':refs,'baseline_execution_source':baseline_source,
  'completed_stage_indices':[],'stage_events':[],'proposal_wave_sizes':[],**bindings}
 write_json(output/'manifest.json',status)
 try:
  import torch
  from .compat_cdf import finite_cdf_compatibility
  from .upstream import load_baseline
  require(torch.cuda.is_available() and torch.cuda.device_count()==1,'one visible GPU required')
  require(torch.get_default_dtype()==torch.float32 and torch.get_float32_matmul_precision()=='highest'
    and not torch.backends.cuda.matmul.allow_tf32 and not torch.is_autocast_enabled('cuda'),'original strict outer flags required')
  load_baseline();accept_reject=import_module('npe_pfn.accept_reject_sampler').accept_reject_sample
  seeds=frozen.worker_stream_seeds(context_seed,case,repetition,0,1)
  torch.manual_seed(seeds['component_uniform']);torch.cuda.manual_seed_all(seeds['component_uniform'])
  generators={name:torch.Generator(device='cpu').manual_seed(seed) for name,seed in seeds.items()}
  torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','16')))
  status.update(seed=seeds['component_uniform'],seed_streams=seeds,gpu=torch.cuda.get_device_name(),torch_cuda=torch.version.cuda,
    sampling_policy={'forward_precision':MODE,'streams':{n:{'role':frozen.STREAM_ROLES[n],'seed':v,'dtype':frozen.STREAM_DTYPES[n],
    'device':'cpu','generator':'torch.Generator'} for n,v in seeds.items()},'head_policy':'raw_mass_rejection_v1','mass_envelope':1.00001,
    'uniform_assignment':'preassigned_per_stage_by_physical_row','head_retry_preserves_row_uniform':True,
    'stage0_density_scope':'one generated proposal wave;8actualrawP heads; no cross-query/request reuse'})
  prior=_prior(torch);counters={'stage0_shared_predictions':0,'component_predictions':0,'head_candidates':0,'head_rejections':0};actual={}
  with finite_cdf_compatibility() as descriptor,pilot.engine_scope(checkpoint,cdf_descriptor=descriptor) as engine:
   callback=lambda prediction,stage,member:pilot.validate_prediction(prediction,stage,member,report=actual)
   stage_draw=frozen.component_stage_sampler(engine,torch,generators,counters,prediction_check=callback)
   def progress(event):status['stage_events'].append({'proposal_round':len(status['proposal_wave_sizes']),**event});write_json(output/'manifest.json',status)
   def proposal(size):
    status['proposal_wave_sizes'].append(int(size))
    values=wave.stage_major_wave(engine,theta,obs,query,int(size),1024,lambda _:None,stage_draw=stage_draw,progress=progress)
    return torch.from_numpy(values),None
   scope=frozen.proposal_buffer_scope(bindings,context_seed=context_seed,case=case,repetition=repetition,rank=0,world_size=1,seeds=seeds)
   scope.update(representation_source_sha256=REPRESENTATION_SHA256,model_inputs=transformed,SS15_execution_source_sha256=SS15_EXECUTION_SHA256)
   fifo=frozen.guarded_proposal_buffer();buffer=fifo.ProposalBuffer(proposal,scope=scope,concatenate=lambda parts:torch.cat(parts,dim=0),world_size=1,rank=0)
   before_rng=pilot.rng_hashes(torch);status['default_rng_before']=before_rng
   torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();request_start=time.perf_counter()
   with torch.no_grad():samples,_,_=accept_reject(proposal=buffer,accept_reject_fn=prior.support.check,num_samples=COUNT,
      max_sampling_batch_size=COUNT,max_iter_rejection=None,show_progress_bars=False,proposal_sampling_kwargs={})
   record=buffer.finalize(original_checked_count=prior.support.proposed,accepted_before_trim=prior.support.accepted,
      returned_count=len(samples),original_support_calls=prior.support.calls)
   record.update(enabled=True,source_unchanged=True,policy_sha256=status['proposal_buffer_policy']['policy_sha256'])
   frozen.validate_proposal_buffer_record(record,world_size=1,rank=0,scope=scope);status['proposal_buffer']=record
   torch.cuda.synchronize();finished=time.perf_counter();values=samples.detach().cpu().numpy().astype(np.float64)
   require(values.shape==(COUNT,21) and support_mask(values).all(),'exactly50000 finite supported complete vectors required')
   waves=len(status['proposal_wave_sizes']);heads=8*counters['stage0_shared_predictions']+counters['component_predictions']
   require(engine.prepare_count==len(engine.stages)==21*waves,'all21 preparation/scope coverage differs')
   require(engine.actual_heads==actual['actual_head_contexts']==heads==descriptor['patched_call_count'],'physical head/CDF counters differ')
   require(engine.shared_densities==counters['stage0_shared_predictions']==waves,'fresh stage0 density count differs')
   require(record['actual_generated_wave_sizes']==status['proposal_wave_sizes'],'actual proposal/FIFO waves differ')
   status.update(execution_counters=engine.metadata(),sampling_counters=dict(counters),actual_prediction_checks=actual,
      expected_patched_call_count=heads,expected_stage_prepare_count=21*waves,generated_proposal_wave_count=waves)
  after_rng=pilot.rng_hashes(torch);status.update(default_rng_after=after_rng,default_rng_preserved=before_rng==after_rng)
  require(before_rng==after_rng,'SS15 model/sampler changed default RNG streams')
  require(descriptor['original_alias_restored'] and descriptor['sdk_sources_unchanged'],'CDF restoration failed')
  require_sources();require(source_sha256()==initial_source and guard_baseline_source()==baseline_source,'execution/science source changed')
  require(sha256_file(qualification)==CONDITIONAL_QUALIFICATION_SHA256,'fixed anchor artifact changed')
  require(raw_bindings(context,raw_query)==bindings,'actual physical bank/query input changed')
  require({'context_theta':array_descriptor(theta),'context_obs':array_descriptor(obs),'query':array_descriptor(query)}==transformed,
    'actual transformed training/query input changed')
  with (output/'samples.npz').open('xb') as stream:np.savez(stream,samples=values)
  status.update(status='complete',samples=array_descriptor(values),samples_file='samples.npz',samples_sha256=sha256_file(output/'samples.npz'),
    compatibility_adapter=dict(descriptor),sources_unchanged=True,completed_stage_indices=list(range(21)),
    rejection={'support_calls':prior.support.calls,'proposed':prior.support.proposed,'accepted_before_final_trim':prior.support.accepted,'returned':len(values)},
    timings={'posterior_request_seconds':finished-request_start,'total_seconds':time.perf_counter()-start,
     'accepted_samples_per_second':COUNT/(finished-request_start),'includes_stage_preparation':True,'includes_CPU_sampling':True},
    memory={'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()})
 except BaseException as error:
  status.update(status='failed',error_type=type(error).__name__,error=str(error))
  if descriptor is not None:status['compatibility_adapter']=dict(descriptor)
  if buffer is not None:status['proposal_buffer']=buffer.metadata()
  raise
 finally:
  if engine is not None:engine.close()
  status['wrapper_total_seconds']=time.perf_counter()-start;write_json(output/'manifest.json',status)
 return status

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('banks','output','checkpoint','ledger','reference','reference-grid','qualification'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--context-seed',type=int,choices=CONTEXT_SEEDS,required=True)
    parser.add_argument('--case',type=int,choices=CASES,required=True)
    manifest=run(**vars(parser.parse_args()))
    print(json.dumps({key:manifest.get(key) for key in
                     ('schema','variant','status','context_seed','case','requested_count','timings')},indent=2))

if __name__=='__main__':
    main()
