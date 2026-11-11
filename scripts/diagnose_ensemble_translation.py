#!/usr/bin/env python3
"""Record the original frozen ensemble translation without changing its output.

Diagnostic only: sixteen saved accepted PFN prefixes, raw observations, original
checkpoint/settings. Correct finite-bar interpolation is computed on CPU solely
as a fixed-member counterfactual; it is never returned to TabPFN or sampled.
Built with PriorLabs-TabPFN.
"""
from __future__ import annotations
import argparse
import inspect
import json
import os
from pathlib import Path
import time
import numpy as np

from posteriorforge.artifacts import runtime_identity, sha256_file, write_json
from posteriorforge.banks import derive_seed, load_context, load_heldout
from posteriorforge.gate import verify_run_binding
from posteriorforge.inference import (CHECKPOINT_SHA256, frozen_regressor_settings,
                                    support_mask, upstream_source_hashes)
from posteriorforge.reference_grid import _require_budgeted_gpu
from posteriorforge.science.analytical_benchmark_v2 import array_descriptor, build_problem


def statistics(value):
    a=np.asarray(value,dtype=np.float64)
    if not np.isfinite(a).all() or not a.size:
        raise ValueError('finite nonempty diagnostic values required')
    return {'mean':float(a.mean()),'min':float(a.min()),'max':float(a.max()),
            'quantile_levels':[0,.01,.1,.5,.9,.99,1],
            'quantiles':np.quantile(a,[0,.01,.1,.5,.9,.99,1]).tolist()}


def finite_bar_cdf(probabilities, borders, queries, *, original_precedence=False):
    """Independent NumPy CDF; original_precedence reproduces the source defect."""
    p=np.asarray(probabilities,dtype=np.float64)
    b=np.asarray(borders,dtype=np.float64)
    y=np.asarray(queries,dtype=np.float64)
    if p.ndim!=2 or b.ndim!=1 or y.ndim!=1 or p.shape[1]!=len(b)-1:
        raise ValueError('incompatible finite-bar shapes')
    if not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(-1),1,atol=1e-6,rtol=0):
        raise ValueError('normalized finite probabilities required')
    if not np.isfinite(b).all() or not (np.diff(b)>0).all() or not np.isfinite(y).all():
        raise ValueError('finite increasing borders and finite queries required')
    widths=np.diff(b)
    index=np.clip(np.searchsorted(b,y,side='left')-1,0,len(b)-2)
    share=y-b[index]/widths[index] if original_precedence else (y-b[index])/widths[index]
    share=np.clip(share,0,1)
    before=np.cumsum(p,axis=-1)-p
    result=before[:,index]+p[:,index]*share[None,:]
    result[:,y<=b[0]]=0
    result[:,y>=b[-1]]=1
    return np.clip(result,0,1)


def translate_finite(probabilities,borders,new_borders,*,original_precedence=False):
    cdf=finite_bar_cdf(probabilities,borders,new_borders,
                       original_precedence=original_precedence)
    cdf[:,0]=0;cdf[:,-1]=1
    return np.maximum(np.diff(cdf,axis=-1),0),cdf


def normalize_rows(value):
    a=np.asarray(value,dtype=np.float64)
    total=a.sum(axis=-1,keepdims=True)
    if not np.isfinite(a).all() or (a<0).any() or (total<=0).any():
        raise ValueError('finite nonnegative positive-mass rows required')
    return a/total


def distances(p,q):
    p=normalize_rows(p);q=normalize_rows(q)
    return {'total_variation':statistics(.5*np.abs(p-q).sum(axis=-1)),
            'boundary_cdf_distance':statistics(np.max(np.abs(np.cumsum(p-q,axis=-1)),axis=-1))}


def collect(*,banks,run,checkpoint,output,ledger=None,stages=21,rows=16,max_seconds=500):
    budget=_require_budgeted_gpu(ledger)
    import torch
    import tabpfn.regressor as regressor_module
    from tabpfn.utils import translate_probs_across_borders
    from posteriorforge.upstream import load_baseline
    if rows!=16 or stages not in (3,21) or not 60<=max_seconds<=500:
        raise ValueError('fixed16-prefix diagnostic,3or21stages, maximum500seconds')
    original=regressor_module.translate_probs_across_borders
    if original is not translate_probs_across_borders:
        raise RuntimeError('actual regressor alias is not the audited original helper')
    helper=original.__globals__['_cdf']
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    result={**runtime_identity(),'schema':'posteriorforge.diagnostic.ensemble_path.v1',
      'attribution':'Built with PriorLabs-TabPFN','state':'running','diagnostic_only':True,
      'candidate_fix_implemented':False,'original_gate_changed':False,'corrected_outputs_returned':False,
      'genuine_inference_run':False,'truth_or_reference_prefixes_used':False,'budget':budget,
      'rows':rows,'stages_requested':stages,'max_seconds_before_next_stage':max_seconds,
      'population_limitation':'Saved accepted PFN prefixes, not free proposal prefixes; no chain or accepted-joint error bound.',
      'counterfactual_limitation':'Fixed original member predictions only; correct finite-bar interpolation, not a repaired half-normal family, no propagation or corrected posterior sample.',
      'source_files':{str(Path(inspect.getfile(function))):sha256_file(inspect.getfile(function)) for function in (original,helper)},
      'regressor_source_sha256':sha256_file(inspect.getfile(regressor_module)),
      'helper_first_line':inspect.getsourcelines(helper)[1],
      'regressor_alias_identity_verified':True,'upstream_files':upstream_source_hashes(),
      'checkpoint_sha256':sha256_file(checkpoint),'stages':[]}
    write_json(root/'manifest.json',result)
    if result['checkpoint_sha256']!=CHECKPOINT_SHA256:raise ValueError('immutable original checkpoint required')
    run=Path(run);manifest_path=run/'manifest.json';manifest=json.loads(manifest_path.read_text())
    if any(manifest[k]!=v for k,v in dict(context_seed=20261111,case=0,context_size=4096,requested_count=50000).items()):
        raise ValueError('original case0 seed20261111 context4096 required')
    heldout=load_heldout(banks,0);context=load_context(banks,20261111,4096)
    verify_run_binding(manifest,context,heldout.obs,20261111,0)
    if manifest['status']!='complete' or sha256_file(run/'samples.npz')!=manifest['samples_sha256']:
        raise ValueError('authenticated complete original samples required')
    with np.load(run/'samples.npz',allow_pickle=False) as data:samples=data['samples'].copy()
    if samples.shape!=(50000,21) or not support_mask(samples).all():raise ValueError('invalid original samples')
    positions=np.linspace(0,len(samples)-1,rows,dtype=int);selected=samples[positions]
    result.update(source_manifest_sha256=sha256_file(manifest_path),
        source_samples_sha256=manifest['samples_sha256'],selected_original_row_indices=positions.tolist(),
        selected_prefix_rows=array_descriptor(selected),scientific_fingerprint=manifest['scientific_fingerprint'],
        query=manifest['query'],parameter_order=manifest['parameter_order'],
        regressor_settings=frozen_regressor_settings(checkpoint),
        fixed_finite_cdf_formula='CDF(y)=sum_{k<bucket(y)}p_k+p_bucket*clip((y-b_left)/(b_right-b_left),0,1);0belowfirstborder,1abovelastborder',
        original_precedence_formula='share=clip(y-b_left/(b_right-b_left),0,1)')
    seed=derive_seed(20261111,'diagnostic-ensemble-translation',case=0)
    np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','16')))
    settings=frozen_regressor_settings(checkpoint);settings['inference_precision']=torch.float32
    posterior=load_baseline()(filter_type='no_filtering',regressor_init_kwargs=settings)
    posterior.append_simulations(torch.from_numpy(context.theta.copy()).float(),
                                 torch.from_numpy(context.obs.copy()).float())
    joint=torch.cat((posterior._x_train,posterior._theta_train),dim=1)
    obs=torch.from_numpy(heldout.obs.copy()).float().repeat(rows,1)
    prefix=torch.from_numpy(selected.copy()).float()
    captures={};active_stage=None;active_calls=[]
    def recording_translation(logits,*,frm,to):
        # Original return object is preserved. No corrected data reaches the model.
        original_output=original(logits,frm=frm,to=to)
        probabilities=logits.softmax(-1).detach().cpu().numpy().astype(np.float64)
        b=frm.detach().cpu().numpy();new_b=to.detach().cpu().numpy()
        actual=original_output.detach().cpu().numpy().astype(np.float64)
        corrected,correct_cdf=translate_finite(probabilities,b,new_b)
        replay,buggy_cdf=translate_finite(probabilities,b,new_b,original_precedence=True)
        replay_error=float(np.max(np.abs(actual-replay)))
        call=len(active_calls);key=f'stage{active_stage}_member{call}'
        captures[key+'_logits']=logits.detach().cpu().numpy()
        captures[key+'_from_borders']=b;captures[key+'_to_borders']=new_b
        captures[key+'_original_output']=original_output.detach().cpu().numpy()
        captures[key+'_correct_finite_output']=corrected
        row={'member_index':call,'from_border_count':len(b),'to_border_count':len(new_b),
             'identical_borders':bool(b.shape==new_b.shape and np.array_equal(b,new_b)),
             'from_width_range':[float(np.diff(b).min()),float(np.diff(b).max())],
             'original_probability_sum':statistics(actual.sum(-1)),
             'correct_finite_probability_sum':statistics(corrected.sum(-1)),
             'original_source_formula_cpu_replay_max_abs_error':replay_error,
             'original_cdf_decreasing_boundary_steps':int((np.diff(buggy_cdf,axis=-1)<-1e-7).sum()),
             'correct_cdf_decreasing_boundary_steps':int((np.diff(correct_cdf,axis=-1)<-1e-12).sum()),
             'normalized_one_member_probability_distances':distances(actual,corrected)}
        if replay_error>1e-5:raise RuntimeError('CPU source-formula replay failed qualification')
        active_calls.append((row,actual,corrected))
        return original_output
    regressor_module.translate_probs_across_borders=recording_translation
    try:
        for stage,name in enumerate(build_problem().parameter_order[:stages]):
            if stage and time.perf_counter()-started>max_seconds:break
            active_stage=stage;active_calls.clear()
            query=torch.cat((obs,prefix[:,:stage]),dim=1)
            if query.shape!=(rows,33+stage):raise RuntimeError('wrong original query construction')
            torch.cuda.synchronize();t0=time.perf_counter()
            posterior._model.fit(joint[:,:33+stage],joint[:,33+stage])
            torch.cuda.synchronize();t1=time.perf_counter()
            prediction=posterior._model.predict(query,output_type='full',quantiles=[])
            torch.cuda.synchronize();t2=time.perf_counter()
            if len(active_calls)!=8:raise RuntimeError('expected all8frozen ensemble calls')
            actual_mixture=np.mean([item[1] for item in active_calls],axis=0)
            correct_mixture=np.mean([item[2] for item in active_calls],axis=0)
            final=prediction['logits'].softmax(-1).detach().cpu().numpy()
            reconstruction_error=float(np.max(np.abs(final-normalize_rows(actual_mixture))))
            if reconstruction_error>1e-5:raise RuntimeError('original ensemble reconstruction failed')
            captures[f'stage{stage}_original_final_probabilities']=final
            captures[f'stage{stage}_correct_finite_fixed_member_mixture']=normalize_rows(correct_mixture)
            result['stages'].append({'stage_index':stage,'parameter':name,'predictor_count':33+stage,
                'query_rows':rows,'member_calls':len(active_calls),'members':[item[0] for item in active_calls],
                'fixed_member_mixture_probability_distances':distances(actual_mixture,correct_mixture),
                'final_returned_probability_reconstruction_max_abs_error':reconstruction_error,
                'original_mixture_probability_sum':statistics(actual_mixture.sum(-1)),
                'timings_seconds':{'fit_synchronized_wall':t1-t0,'predict_with_diagnostic_cpu_capture_wall':t2-t1}})
            torch.cuda.empty_cache();torch.cuda.ipc_collect()
            write_json(root/'manifest.json',result)
        with (root/'ensemble-translation-capture.npz').open('xb') as stream:np.savez_compressed(stream,**captures)
        result.update(state='completed_diagnostic',stages_completed=len(result['stages']),
            stopped_before_all_requested_stages=len(result['stages'])<stages,
            capture_sha256=sha256_file(root/'ensemble-translation-capture.npz'),
            capture_bytes=(root/'ensemble-translation-capture.npz').stat().st_size,
            elapsed_seconds=time.perf_counter()-started)
        write_json(root/'manifest.json',result)
    except BaseException as error:
        result.update(state='failed_diagnostic',error_type=type(error).__name__,error=str(error),
                      elapsed_seconds=time.perf_counter()-started)
        write_json(root/'manifest.json',result);raise
    finally:
        regressor_module.translate_probs_across_borders=original
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('banks','run','checkpoint','output'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--ledger');parser.add_argument('--stages',type=int,default=21)
    parser.add_argument('--rows',type=int,default=16);parser.add_argument('--max-seconds',type=int,default=500)
    print(json.dumps(collect(**vars(parser.parse_args())),indent=2))
