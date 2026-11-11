#!/usr/bin/env python3
"""Diagnostic-only true conditional-noise PIT; no model runs or gate changes.

Given physical parameters, independent absolute Gaussian noise implies
p(sigma_i | physics,y) proportional to sigma_i**(-n_i) *
exp(-RSS_i/(2*sigma_i**2)) * exp(-(sigma_i-.255)**2/(2*.1225**2))
on [.01,.5]. The Gaussian truncation normalizer cancels. This necessary
joint-posterior condition does not identify an autoregressive error mechanism.
All real conditional-reference evaluation requires a reserved Slurm GPU job.
"""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import time

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

from posteriorforge.artifacts import runtime_identity, sha256_file, write_json
from posteriorforge.banks import load_context, load_heldout
from posteriorforge.gate import verify_run_binding
from posteriorforge.inference import support_mask
from posteriorforge.reference_grid import _require_budgeted_gpu
from posteriorforge.science.analytical_benchmark_v2 import build_problem, unflatten_observations

LOW, HIGH, PRIOR_MEAN, PRIOR_SD = .01, .5, .255, .1225
TOLERANCE = 1e-5


def _validate(sigma, rss, count):
    sigma, rss = np.asarray(sigma, dtype=np.float64), np.asarray(rss, dtype=np.float64)
    if sigma.ndim != 1 or rss.shape != sigma.shape or not len(sigma):
        raise ValueError('sigma and RSS must be nonempty paired vectors')
    if (not np.isfinite(sigma).all() or not np.isfinite(rss).all()
            or np.any(sigma < LOW) or np.any(sigma > HIGH) or np.any(rss < 0)
            or type(count) is not int or count < 0):
        raise ValueError('finite supported sigma, nonnegative RSS and count required')
    return sigma, rss


def conditional_cdf_numpy(sigma, rss, count, *, nodes=128):
    """Small CPU algebra/unit-test utility; real arrays are GPU-job guarded."""
    sigma, rss = _validate(sigma, rss, count)
    x, w = np.polynomial.legendre.leggauss(nodes)
    def mass(left, right):
        width = right-left
        s = left[:,None] + width[:,None]*(x[None,:]+1)/2
        logf = -count*np.log(s)-rss[:,None]/(2*s*s)-.5*((s-PRIOR_MEAN)/PRIOR_SD)**2
        with np.errstate(divide='ignore'):
            return np.log(width/2)+logsumexp(logf+np.log(w)[None,:],axis=1)
    a=mass(np.full_like(sigma,LOW),sigma)
    b=mass(sigma,np.full_like(sigma,HIGH))
    return np.exp(a-np.logaddexp(a,b))


def conditional_cdf_gpu(sigma, rss, count, *, nodes=128, batch_size=4096):
    import torch
    sigma, rss = _validate(sigma,rss,count)
    if not torch.cuda.is_available():
        raise RuntimeError('conditional diagnostic requires CUDA FP64')
    x,w=np.polynomial.legendre.leggauss(nodes)
    tx=torch.as_tensor(x,device='cuda',dtype=torch.float64)
    tw=torch.log(torch.as_tensor(w,device='cuda',dtype=torch.float64))
    answer=[]
    for start in range(0,len(sigma),batch_size):
        s0=torch.as_tensor(sigma[start:start+batch_size],device='cuda',dtype=torch.float64)
        tr=torch.as_tensor(rss[start:start+batch_size],device='cuda',dtype=torch.float64)
        def mass(left,right):
            width=right-left
            s=left[:,None]+width[:,None]*(tx[None,:]+1)/2
            logf=-count*torch.log(s)-tr[:,None]/(2*s*s)-.5*((s-PRIOR_MEAN)/PRIOR_SD)**2
            return torch.log(width/2)+torch.logsumexp(logf+tw[None,:],dim=1)
        a=mass(torch.full_like(s0,LOW),s0)
        b=mass(s0,torch.full_like(s0,HIGH))
        value=torch.exp(a-torch.logaddexp(a,b))
        if value.dtype!=torch.float64:raise RuntimeError('FP64 unavailable')
        answer.append(value.cpu().numpy())
    return np.concatenate(answer)


def conditional_cdf_quad(sigma, rss, count):
    """Independent adaptive CPU formula, called on real rows only inside GPU job."""
    _validate(np.array([sigma]),np.array([rss]),count)
    def logf(s):return -count*math.log(s)-rss/(2*s*s)-.5*((s-PRIOR_MEAN)/PRIOR_SD)**2
    mode=minimize_scalar(lambda s:-logf(s),bounds=(LOW,HIGH),method='bounded')
    shift=max(logf(LOW),logf(HIGH),logf(mode.x))
    f=lambda s:math.exp(logf(s)-shift)
    denom,error_d=quad(f,LOW,HIGH,epsabs=1e-12,epsrel=1e-12,limit=200)
    numer,error_n=quad(f,LOW,float(sigma),epsabs=1e-12,epsrel=1e-12,limit=200)
    if denom<=0 or not math.isfinite(denom):raise RuntimeError('independent normalizer failed')
    return numer/denom, (error_n+error_d)/denom


def _ks_uniform(values):
    ordered=np.sort(values)
    n=len(ordered)
    return float(max(np.max(np.arange(1,n+1)/n-ordered),np.max(ordered-np.arange(n)/n)))


def _stratified(sigma,rss,limit=32):
    positions={0,len(sigma)-1}
    for v in (sigma,rss):
        order=np.argsort(v,kind='stable')
        positions.update(order[np.linspace(0,len(order)-1,limit//2,dtype=int)].tolist())
    return np.asarray(sorted(positions),dtype=int)


def run(*,banks,runs,output,cases='0',seeds='20261111,20261112,20261113',ledger=None):
    budget=_require_budgeted_gpu(ledger)
    import torch
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter()
    result={**runtime_identity(),'schema':'posteriorforge.diagnostic.noise_conditionals.v1',
        'attribution':'Built with PriorLabs-TabPFN','state':'running','budget':budget,
        'baseline_release':'89404ac9e43bf9fb72d3e51a7113dbd207871221',
        'diagnostic_only':True,'inference_rerun':False,'reference_integration':'one-dimensional conditional CDF only; no marginal reference reintegration',
        'candidate_fix_implemented':False,'original_gate_changed':False,'truth_used':False,
        'meaning':'Conditional PIT uniformity is necessary, not sufficient, for the accepted joint posterior to be correct; it does not identify why the estimator fails or prove autoregressive propagation; uniform PIT can also hide cancelling conditional errors.',
        'formula':'sigma^(-n) exp[-RSS/(2sigma^2)] exp[-.5((sigma-.255)/.1225)^2], .01<=sigma<=.5',
        'qualification_tolerance':TOLERANCE,'rows':[],'qualification':{'passed':False},
        'source_sha256':sha256_file(__file__),'device':torch.cuda.get_device_name(),'torch_float64':True}
    write_json(root/'manifest.json',result)
    problem=build_problem();case_ids=[int(x) for x in cases.split(',')];seed_ids=[int(x) for x in seeds.split(',')]
    observations_counts=[];pit_arrays={};max_refinement=0.;max_quad=0.;max_quad_error=0.;identity_error=0.
    try:
        for case in case_ids:
            observed=load_heldout(banks,case).obs
            observed_blocks=unflatten_observations(observed)
            for seed in seed_ids:
                folder=Path(runs)/'context-4096'/f'seed-{seed}-case-{case}'
                path=folder/'manifest.json';manifest=json.loads(path.read_text())
                verify_run_binding(manifest,load_context(banks,seed,4096),observed,seed,case)
                if manifest['status']!='complete' or sha256_file(folder/'samples.npz')!=manifest['samples_sha256']:
                    raise ValueError('complete authenticated baseline sample payload required')
                with np.load(folder/'samples.npz',allow_pickle=False) as payload:
                    samples=payload['samples'].copy()
                if samples.shape!=(50000,21) or not support_mask(samples).all():raise ValueError('complete valid original samples required')
                predicted=problem.observation_model.predict(samples)
                assembled=np.zeros(len(samples));run_pits=[]
                for block,noise in zip(problem.observation_model.blocks,problem.prior.noise_names):
                    residual=predicted[block.dataset_id]-observed_blocks.values[block.dataset_id]
                    rss=np.square(residual).sum(axis=1);sigma=samples[:,problem.parameter_order.index(noise)];n=block.observation_size
                    assembled += -.5*rss/(sigma*sigma)-n*(np.log(sigma)+.5*math.log(2*math.pi))
                    q128=conditional_cdf_gpu(sigma,rss,n,nodes=128)
                    q256=conditional_cdf_gpu(sigma,rss,n,nodes=256)
                    positions=_stratified(sigma,rss)
                    q64=conditional_cdf_gpu(sigma[positions],rss[positions],n,nodes=64)
                    delta=float(np.max(np.abs(q256-q128)))
                    delta64=float(np.max(np.abs(q64-q128[positions])))
                    quad_records=[]
                    for index in positions:
                        check,quad_error=conditional_cdf_quad(float(sigma[index]),float(rss[index]),n)
                        error=abs(check-float(q256[index]));max_quad=max(max_quad,error);max_quad_error=max(max_quad_error,quad_error)
                        quad_records.append({'row':int(index),'rss':float(rss[index]),'sigma':float(sigma[index]),'quad_cdf':float(check),'gpu_cdf':float(q256[index]),'absolute_error':error,'quad_error_estimate':quad_error})
                    max_refinement=max(max_refinement,delta)
                    if not np.isfinite(q256).all() or np.any(q256<0) or np.any(q256>1):raise RuntimeError('invalid conditional CDF values; no clipping allowed')
                    if delta>TOLERANCE or max_quad>TOLERANCE or max_quad_error>TOLERANCE:
                        raise RuntimeError('conditional CDF precision inconclusive; no diagnostic conclusion')
                    key=f'seed{seed}_case{case}_{noise}';pit_arrays[key]=q256;run_pits.append(q256)
                    result['rows'].append({'case':case,'context_seed':seed,'noise':noise,'dataset':block.dataset_id,
                        'n':n,'count':len(samples),'conditional_pit_ks_uniform':_ks_uniform(q256),
                        'conditional_pit_mean':float(q256.mean()),'conditional_pit_variance':float(q256.var(ddof=1)),
                        'fraction_below_05':float(np.mean(q256<.05)),'fraction_above_95':float(np.mean(q256>.95)),
                        'pit_quantiles':np.quantile(q256,[.01,.05,.25,.5,.75,.95,.99]).tolist(),
                        'refinement128_256_max_abs':delta,'stratified64_128_max_abs':delta64,
                        'independent_quad':quad_records,'manifest_sha256':sha256_file(path),
                        'samples_sha256':manifest['samples_sha256'],'rss_quantiles':np.quantile(rss,[0,.01,.5,.99,1]).tolist()})
                identity_error=max(identity_error,float(np.max(np.abs(assembled-problem.log_likelihood(samples,observed_blocks)))))
                if not np.allclose(assembled,problem.log_likelihood(samples,observed_blocks),rtol=1e-12,atol=1e-9):raise RuntimeError('conditional formula does not reconstruct source likelihood')
                write_json(root/'manifest.json',result)
        with (root/'conditional-pits.npz').open('xb') as stream:np.savez(stream,**pit_arrays)
        result['qualification']={'passed':True,'all_row128_256_max_abs':max_refinement,
            'independent_quad_max_abs':max_quad,'quad_error_estimate_max':max_quad_error,
            'source_likelihood_identity_max_abs':identity_error,
            'contract':'All evaluated PIT points agree128vs256; independent adaptive quad on sigma/RSS strata. This is not a universal conditional-CDF bound.'}
        result.update(state='qualified_diagnostic',pit_payload_sha256=sha256_file(root/'conditional-pits.npz'),
                      elapsed_seconds=time.perf_counter()-start)
        _plot(result,pit_arrays,root)
        write_json(root/'manifest.json',result)
        return result
    except Exception as error:
        result.update(state='failed_or_inconclusive',error_type=type(error).__name__,error=str(error),elapsed_seconds=time.perf_counter()-start)
        write_json(root/'manifest.json',result)
        raise


def _plot(result,pits,root):
    os.environ.setdefault('MPLCONFIGDIR',str(Path(os.environ.get('PF_SCRATCH','/scratch/project/eu-26-17/eubrieucb/posteriorforge'))/'cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cases=sorted({row['case'] for row in result['rows']})
    for case in cases:
        figure,axes=plt.subplots(2,3,figsize=(14,8),sharex=True,sharey=True)
        for axis,noise in zip(axes.flat,build_problem().prior.noise_names):
            axis.plot([0,1],[0,1],color='#25364b',ls='--',label='Uniform reference condition')
            for row in result['rows']:
                if row['case']!=case or row['noise']!=noise:continue
                value=np.sort(pits[f"seed{row['context_seed']}_case{case}_{noise}"])
                axis.step(value,np.arange(1,len(value)+1)/len(value),where='post',lw=1.3,
                    label=f"{row['context_seed']}; D={row['conditional_pit_ks_uniform']:.3f}")
            axis.set(title=noise,xlim=(0,1),ylim=(0,1));axis.grid(alpha=.2);axis.legend(fontsize=8,loc='upper left')
        for a in axes[-1]:a.set_xlabel('True conditional CDF evaluated at the PFN noise draw')
        for a in axes[:,0]:a.set_ylabel('Empirical CDF of conditional PIT')
        figure.suptitle('Built with PriorLabs-TabPFN\nSaved-output conditional-noise calibration diagnostic · case '+str(case),fontsize=15)
        figure.text(.5,.025,'Diagnostic only: necessary joint-posterior condition, not a new acceptance gate or proof of autoregressive causality.\nOriginal samples and gate are unchanged. Uniformity uses the true noise conditional given each PFN physical draw.',ha='center',fontsize=10)
        figure.subplots_adjust(top=.86,bottom=.15,hspace=.25,wspace=.12)
        name=f'model-conditional-noise-pit-case-{case}.png';figure.savefig(root/name,dpi=160);plt.close(figure)
        result.setdefault('figures',{})[name]=sha256_file(root/name)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('banks','runs','output'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--cases',default='0');parser.add_argument('--seeds',default='20261111,20261112,20261113');parser.add_argument('--ledger')
    print(json.dumps(run(**vars(parser.parse_args())),indent=2))
