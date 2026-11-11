#!/usr/bin/env python3
"""CPU-only summaries of saved diagnostic collectors; no model or new inference.
Built with PriorLabs-TabPFN.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.special import erfc
from posteriorforge.artifacts import sha256_file,write_json


def tail_distances(left_mass,right_mass,left_tail=.5,right_tail=.5):
    left=np.asarray(left_mass,dtype=np.float64)*left_tail
    right=np.asarray(right_mass,dtype=np.float64)*right_tail
    if (left<0).any() or (right<0).any() or ((left+right)>1+1e-6).any():
        raise ValueError('invalid endpoint mass')
    return left+right,np.maximum(left,right)


def _softmax(logits):
    value=np.asarray(logits,dtype=np.float64)
    p=np.exp(value-value.max(axis=-1,keepdims=True))
    return p/p.sum(axis=-1,keepdims=True)


def summarize_bar(path):
    path=Path(path);manifest=json.loads((path/'manifest.json').read_text())
    if manifest['state']!='completed_diagnostic':raise ValueError('complete original bar probe required')
    capture=path/'bar-logits-capture.npz'
    if sha256_file(capture)!=manifest['capture_sha256']:raise ValueError('capture hash mismatch')
    # The pinned half-normal's float32 median calibration is reconstructed on CPU.
    import torch
    from tabpfn.model.bar_distribution import FullSupportBarDistribution
    result={'schema':'posteriorforge.diagnostic.bar_summary.v1','attribution':'Built with PriorLabs-TabPFN',
      'diagnostic_only':True,'original_gate_changed':False,'candidate_fix_implemented':False,
      'source_manifest_sha256':sha256_file(path/'manifest.json'),'source_capture_sha256':manifest['capture_sha256'],
      'population_limitation':manifest['population_limitation'],
      'distribution_comparison':'At one fixed prefix: the returned criterion half-normal density family versus the actual inherited finite-uniform sampler family, before support rejection.',
      'ideal_exact_formula':'For ideal median-calibrated tails: TV=(p_left+p_right)/2; KS=max(p_left,p_right)/2.',
      'proof':'Each half-normal puts half its endpoint mass beyond the finite endpoint bar and its density is everywhere lower than the uniform density on that bar. Endpoint discrepancies have disjoint supports. Their TV contributions add; their maximum CDF discrepancies do not add.',
      'finite_precision_formula':'TV=p_left*t_left+p_right*t_right; KS=max(p_left*t_left,p_right*t_right), where t_side=erfc(width/(scale*sqrt(2))). Valid when half-normal density at its anchor is below the overlapping uniform density; checked below.',
      'cdf_translation_caution':'This compares only final returned-criterion families. The separate tabpfn.utils._cdf ensemble interpolation defect can affect interior probabilities before these final logits, and is not bounded by final endpoint mass.',
      'stages':[]}
    with np.load(capture,allow_pickle=False) as data:
        for stage in manifest['stages']:
            j=stage['stage_index'];b=data[f'stage{j}_borders'];w=np.diff(b)
            scales=[float(FullSupportBarDistribution.halfnormal_with_p_weight_before(torch.tensor(width,dtype=torch.from_numpy(w).dtype)).scale) for width in (w[0],w[-1])]
            tail=np.array([erfc(float(width)/(scale*np.sqrt(2))) for width,scale in zip((w[0],w[-1]),scales)])
            overlap_ratio=np.array([float(width)*np.sqrt(2/np.pi)/scale for width,scale in zip((w[0],w[-1]),scales)])
            if not (overlap_ratio<1).all():raise ValueError('tail TV proof assumptions fail')
            left=stage['left_endpoint_mass'];right=stage['right_endpoint_mass']
            exact_mean_tv=tail[0]*left['mean']+tail[1]*right['mean']
            max_ks=max(tail[0]*left['max'],tail[1]*right['max'])
            mean_ks_bounds=[max(tail[0]*left['mean'],tail[1]*right['mean']),exact_mean_tv]
            p=_softmax(data[f'stage{j}_logits'])
            tv,ks=tail_distances(p[:,0],p[:,-1],*tail)
            result['stages'].append({'stage_index':j,'parameter':stage['parameter'],
              'captured_selected_rows':len(p),'all128_endpoint_mass':stage['total_endpoint_mass'],
              'source_float_endpoint_tail_fractions':tail.tolist(),
              'half_normal_peak_density_divided_by_finite_uniform_density':overlap_ratio.tolist(),
              'all128_mean_one_conditional_tv':float(exact_mean_tv),
              'all128_max_one_conditional_ks':float(max_ks),
              'all128_mean_one_conditional_ks_bounds':list(map(float,mean_ks_bounds)),
              'ideal_all128_max_one_conditional_tv':.5*stage['total_endpoint_mass']['max'],
              'ideal_all128_tv_quantile_levels':stage['total_endpoint_mass']['quantile_levels'],
              'ideal_all128_tv_quantiles':(.5*np.asarray(stage['total_endpoint_mass']['quantiles'])).tolist(),
              'ideal_formula_max_abs_precision_error_bound':float(np.max(np.abs(tail-.5))),
              'captured_subset_tv':tv.tolist(),'captured_subset_ks':ks.tolist(),
              'captured_subset_selection_caution':'Selected extremal and evenly spaced capture rows; not a representative independent second128-row population.',
              'timings_seconds':stage['timings_seconds']})
    return result


def summarize_translation(path):
    path=Path(path);manifest=json.loads((path/'manifest.json').read_text())
    if manifest['state']!='completed_diagnostic':raise ValueError('complete translation probe required')
    capture=path/'ensemble-translation-capture.npz'
    if sha256_file(capture)!=manifest['capture_sha256']:raise ValueError('capture hash mismatch')
    return {'schema':'posteriorforge.diagnostic.translation_summary.v1','attribution':'Built with PriorLabs-TabPFN',
      'diagnostic_only':True,'original_gate_changed':False,'candidate_fix_implemented':False,
      'source_manifest_sha256':sha256_file(path/'manifest.json'),'source_capture_sha256':manifest['capture_sha256'],
      'population_limitation':manifest['population_limitation'],
      'counterfactual_limitation':manifest['counterfactual_limitation'],
      'stages_completed':manifest['stages_completed'],
      'interpretation':'Distances compare original final member mixture against correct finite-bar interpolation on the same captured original members. This is an API reconstruction counterfactual, not a corrected posterior, half-normal repair, or demonstrated gate improvement.',
      'maximum_cpu_source_replay_error':max(member['original_source_formula_cpu_replay_max_abs_error'] for stage in manifest['stages'] for member in stage['members']),
      'maximum_final_probability_reconstruction_error':max(stage['final_returned_probability_reconstruction_max_abs_error'] for stage in manifest['stages']),
      'stages':[{'stage_index':stage['stage_index'],'parameter':stage['parameter'],
                 'member_calls':stage['member_calls'],
                 'fixed_member_mixture_probability_distances':stage['fixed_member_mixture_probability_distances'],
                 'same_borders_member_count':sum(member['identical_borders'] for member in stage['members']),
                 'original_member_probability_sum_min':min(member['original_probability_sum']['min'] for member in stage['members']),
                 'original_member_probability_sum_max':max(member['original_probability_sum']['max'] for member in stage['members']),
                 'original_cdf_decreasing_boundary_steps':sum(member['original_cdf_decreasing_boundary_steps'] for member in stage['members']),
                 'timings_seconds':stage['timings_seconds']}
                for stage in manifest['stages']]}


def make_figure(result,path,kind):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    stages=result['stages'];x=np.arange(len(stages));labels=[s['parameter'] for s in stages]
    fig,ax=plt.subplots(figsize=(13,5.5))
    if kind=='bar':
        ax.plot(x,[s['all128_mean_one_conditional_tv'] for s in stages],'o-',label='TV mean,128 saved accepted prefixes')
        ax.plot(x,[s['ideal_all128_max_one_conditional_tv'] for s in stages],'s--',label='TV maximum (ideal median calibration)')
        ax.plot(x,[s['all128_max_one_conditional_ks'] for s in stages],'^:',label='KS maximum')
        ax.set_title('Final-criterion tail-family disagreement at saved accepted prefixes')
    else:
        values=[s['fixed_member_mixture_probability_distances'] for s in stages]
        ax.plot(x,[v['total_variation']['mean'] for v in values],'o-',label='Fixed-member translated-mixture TV mean')
        ax.plot(x,[v['total_variation']['max'] for v in values],'s--',label='Fixed-member translated-mixture TV maximum')
        ax.plot(x,[v['boundary_cdf_distance']['max'] for v in values],'^:',label='Boundary-CDF distance maximum')
        ax.set_title('Original versus correct finite interpolation of the same member predictions')
    ax.set_ylabel('One-conditional distribution discrepancy')
    ax.set_xlabel('Original autoregressive parameter stage')
    ax.set_xticks(x,labels,rotation=55,ha='right')
    ax.set_ylim(bottom=0);ax.grid(alpha=.2);ax.legend(loc='best',fontsize=9)
    fig.text(.01,.015,'Diagnostic only · accepted-prefix population · no corrected posterior or gate change · Built with PriorLabs-TabPFN',fontsize=9)
    fig.tight_layout(rect=(0,.045,1,1));fig.savefig(path,dpi=180);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bar');parser.add_argument('--translation');parser.add_argument('--output',required=True)
    args=parser.parse_args()
    if not args.bar and not args.translation:parser.error('at least one saved diagnostic required')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    for kind,path in (('bar',args.bar),('translation',args.translation)):
        if path:
            result=summarize_bar(path) if kind=='bar' else summarize_translation(path)
            write_json(out/f'model-{kind}-summary.json',result)
            make_figure(result,out/f'model-{kind}-divergence.png',kind)
            print(json.dumps({'kind':kind,'stages':len(result['stages']),'output':str(out)}))
