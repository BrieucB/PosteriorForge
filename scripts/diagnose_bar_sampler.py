#!/usr/bin/env python3
"""Diagnostic collector for pinned criterion semantics on saved accepted prefixes.

No sampling repair, optimization, changed checkpoint or genuine inference run.
Every query prefix is a saved accepted PFN prefix, not truth/reference data.
The prefix population is not the same as the free proposal population. Endpoint
mass at these prefixes cannot establish a rigorous chain/joint error bound.
"""
from __future__ import annotations
import argparse
import inspect
import json
import os
from pathlib import Path
import time
import numpy as np

from posteriorforge.artifacts import runtime_identity,sha256_file,write_json
from posteriorforge.banks import derive_seed,load_context,load_heldout
from posteriorforge.gate import verify_run_binding
from posteriorforge.inference import CHECKPOINT_SHA256,frozen_regressor_settings,support_mask,upstream_source_hashes
from posteriorforge.reference_grid import _require_budgeted_gpu
from posteriorforge.science.analytical_benchmark_v2 import array_descriptor,build_problem


def _quantiles(value):
    a=np.asarray(value,dtype=np.float64)
    return {'mean':float(a.mean()),'max':float(a.max()),'min':float(a.min()),
            'quantile_levels':[0,.01,.1,.5,.9,.99,1],
            'quantiles':np.quantile(a,[0,.01,.1,.5,.9,.99,1]).tolist()}


def collect(*,banks,run,checkpoint,output,ledger=None,rows=128):
    budget=_require_budgeted_gpu(ledger)
    import torch
    from posteriorforge.upstream import load_baseline
    from tabpfn.model.bar_distribution import BarDistribution,FullSupportBarDistribution
    if rows!=128:raise ValueError('this declared probe uses128 accepted-prefix rows')
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter();model_source=Path(inspect.getfile(FullSupportBarDistribution))
    result={**runtime_identity(),'schema':'posteriorforge.diagnostic.bar_sampler.v1',
      'attribution':'Built with PriorLabs-TabPFN','state':'running','diagnostic_only':True,
      'baseline_release':'89404ac9e43bf9fb72d3e51a7113dbd207871221','budget':budget,
      'original_gate_changed':False,'candidate_fix_implemented':False,'sampling_algorithm_changed':False,
      'genuine_inference_run':False,'truth_or_reference_prefixes_used':False,
      'population_limitation':'128 saved accepted PFN prefixes, not the unconditioned free-proposal prefix population; no rigorous chain/joint TV claim is made.',
      'timing_limitation':'Synchronized diagnostic outer fit/predict walls and actual CPU criterion.sample time; not the original full50000request benchmark.',
      'checkpoint_sha256':sha256_file(checkpoint),'criterion_source_sha256':sha256_file(model_source),
      'upstream_files':upstream_source_hashes(),'rows':rows,'stages':[]}
    write_json(root/'manifest.json',result)
    if result['checkpoint_sha256']!=CHECKPOINT_SHA256:raise ValueError('original immutable checkpoint required')
    run=Path(run);path=run/'manifest.json';manifest=json.loads(path.read_text())
    if manifest['context_seed']!=20261111 or manifest['case']!=0 or manifest['context_size']!=4096 or manifest['requested_count']!=50000:
        raise ValueError('probe fixed to original case0 seed20261111 context4096')
    heldout=load_heldout(banks,0);context=load_context(banks,20261111,4096)
    verify_run_binding(manifest,context,heldout.obs,20261111,0)
    if manifest['status']!='complete' or sha256_file(run/'samples.npz')!=manifest['samples_sha256']:
        raise ValueError('authenticated complete original samples required')
    with np.load(run/'samples.npz',allow_pickle=False) as value:samples=value['samples'].copy()
    if samples.shape!=(50000,21) or not support_mask(samples).all():raise ValueError('invalid source samples')
    positions=np.linspace(0,len(samples)-1,rows,dtype=int)
    selected=samples[positions]
    result.update(source_manifest_sha256=sha256_file(path),source_samples_sha256=manifest['samples_sha256'],
        selected_original_row_indices=positions.tolist(),selected_prefix_rows=array_descriptor(selected),
        scientific_fingerprint=manifest['scientific_fingerprint'],parameter_order=manifest['parameter_order'],
        query=manifest['query'],regressor_settings=frozen_regressor_settings(checkpoint),
        criterion_api_owner={name:getattr(FullSupportBarDistribution,name).__qualname__ for name in ('forward','mean','cdf','icdf','sample')},
        sdk_semantics_observed={'density_and_mean':'FullSupportBarDistribution endpoint bins use half-normal tails.',
          'cdf_and_icdf':'Inherited BarDistribution finite uniform endpoint bins.',
          'sampling':'Actual FullSupportBarDistribution.sample row-loops inherited icdf, preserving finite endpoint bars.',
          'unused_pdf_path':'Pinned pdf() exponentiates forward() although forward returns NLL; not called by the baseline NPE sampler, so not asserted as its accuracy cause.'})
    np.random.seed(derive_seed(20261111,'diagnostic-bar-sampler',case=0));torch.manual_seed(derive_seed(20261111,'diagnostic-bar-sampler',case=0))
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','16')))
    settings=frozen_regressor_settings(checkpoint);settings['inference_precision']=torch.float32
    constructor=load_baseline();posterior=constructor(filter_type='no_filtering',regressor_init_kwargs=settings)
    posterior.append_simulations(torch.from_numpy(context.theta.copy()).float(),torch.from_numpy(context.obs.copy()).float())
    joint=torch.cat((posterior._x_train,posterior._theta_train),dim=1)
    selected_t=torch.from_numpy(selected.copy()).float();obs=torch.from_numpy(heldout.obs.copy()).float().repeat(rows,1)
    captures={};problem=build_problem()
    try:
        for stage,name in enumerate(problem.parameter_order):
            query=torch.cat((obs,selected_t[:,:stage]),dim=1)
            assert query.shape==(rows,33+stage)
            torch.cuda.synchronize();t0=time.perf_counter()
            posterior._model.fit(joint[:,:33+stage],joint[:,33+stage])
            torch.cuda.synchronize();t1=time.perf_counter()
            prediction=posterior._model.predict(query,output_type='full',quantiles=[])
            torch.cuda.synchronize();t2=time.perf_counter()
            criterion=prediction['criterion'];logits=prediction['logits']
            if logits.device.type!='cpu':raise RuntimeError('pinned full API unexpectedly returned non-CPU logits')
            # Time the actual unchanged SDK sample implementation; no replacement.
            sample_start=time.perf_counter();draw=criterion.sample(logits);sample_end=time.perf_counter()
            probabilities=logits.softmax(-1)
            left=probabilities[:,0].numpy();right=probabilities[:,-1].numpy();total=left+right
            borders=criterion.borders.detach().cpu()
            uniform_mean=(probabilities*((borders[:-1]+borders[1:])/2)).sum(dim=1)
            declared_mean=criterion.mean(logits)
            mean_delta=(declared_mean-uniform_mean).detach().cpu().numpy()
            source_values=selected_t[:,stage]
            capture_positions=sorted(set([0,rows-1,int(np.argmax(total)),int(np.argmin(total)),*np.linspace(0,rows-1,4,dtype=int).tolist()]))
            capture_positions=np.asarray(capture_positions,dtype=int)
            captures[f'stage{stage}_logits']=logits[capture_positions].detach().numpy()
            captures[f'stage{stage}_borders']=borders.numpy()
            captures[f'stage{stage}_original_row_indices']=positions[capture_positions]
            captures[f'stage{stage}_univariate_diagnostic_draws']=draw.detach().cpu().numpy()
            cache_start=time.perf_counter();torch.cuda.empty_cache();torch.cuda.ipc_collect();cache_end=time.perf_counter()
            row={'stage_index':stage,'parameter':name,'predictor_count':33+stage,'query_rows':rows,
              'criterion_type':type(criterion).__module__+'.'+type(criterion).__qualname__,
              'logits_shape':list(logits.shape),'logits_dtype':str(logits.dtype),'logits_device':str(logits.device),
              'physical_borders':[float(borders[0]),float(borders[1]),float(borders[-2]),float(borders[-1])],
              'left_endpoint_mass':_quantiles(left),'right_endpoint_mass':_quantiles(right),'total_endpoint_mass':_quantiles(total),
              'declared_halfnormal_mean_minus_finite_sampler_mean':_quantiles(mean_delta),
              'saved_target_fraction_first_or_last_finite_bucket':float(((source_values<=borders[1])|(source_values>=borders[-2])).float().mean()),
              'univariate_draw_fraction_outside_finite_borders':float(((draw<borders[0])|(draw>borders[-1])).float().mean()),
              'frozen_prior_bounds':list(problem.prior.bounds) if name in problem.prior.noise_names else 'unbounded',
              'timings_seconds':{'fit_synchronized_wall':t1-t0,'predict_synchronized_wall':t2-t1,'original_criterion_sample_cpu_wall':sample_end-sample_start,'cache_clear_cpu_wall':cache_end-cache_start},
              'captured_logit_original_row_indices':positions[capture_positions].tolist(),
              'conditional_mass_caution':'Endpoint mass bounds only the changed part of one conditional at each recorded prefix; do not sum this probe into a rigorous chain or accepted-joint TV bound.'}
            result['stages'].append(row);write_json(root/'manifest.json',result)
        with (root/'bar-logits-capture.npz').open('xb') as stream:np.savez_compressed(stream,**captures)
        result.update(state='completed_diagnostic',capture_sha256=sha256_file(root/'bar-logits-capture.npz'),elapsed_seconds=time.perf_counter()-start)
        write_json(root/'manifest.json',result)
        return result
    except BaseException as error:
        result.update(state='failed_diagnostic',error_type=type(error).__name__,error=str(error),elapsed_seconds=time.perf_counter()-start)
        write_json(root/'manifest.json',result);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('banks','run','checkpoint','output'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--ledger');parser.add_argument('--rows',type=int,default=128)
    print(json.dumps(collect(**vars(parser.parse_args())),indent=2))
