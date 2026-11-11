#!/usr/bin/env python3
"""Saved-data representation/coverage audit; no PFN calls or candidate pipeline."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy.special import logsumexp
from posteriorforge.artifacts import runtime_identity, sha256_file, write_json
from posteriorforge.banks import load_context, load_heldout
from posteriorforge.reference import load_reference_distribution
from posteriorforge.science.analytical_benchmark_v2 import build_problem, unflatten_observations


def _effective_rank(covariance):
    eigen=np.linalg.eigvalsh(covariance)[::-1]
    eigen=np.maximum(eigen,0)
    cumulative=np.cumsum(eigen)/eigen.sum()
    return {'participation_ratio':float(eigen.sum()**2/np.square(eigen).sum()),
            'components_90pct':int(np.searchsorted(cumulative,.90)+1),
            'components_99pct':int(np.searchsorted(cumulative,.99)+1),
            'eigenvalues':eigen.tolist()}


def run(*,banks,references,output):
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();problem=build_problem()
    result={**runtime_identity(),'schema':'posteriorforge.diagnostic.representation.v1',
      'attribution':'Built with PriorLabs-TabPFN','diagnostic_only':True,
      'candidate_fix_implemented':False,'inference_rerun':False,'original_gate_changed':False,
      'coverage_caution':'Likelihood importance ESS and nearest-world distances diagnose finite-bank geometry only. PFN learns conditional relationships; low ESS does not prove why it fails or that a larger bank fixes it.',
      'reference_summary_distance':'Squared Euclidean distance after marginal-reference-SD scaling; NOT a full-covariance Mahalanobis distance.',
      'sufficiency':{'mechanical_block':'z=y-offset; T=(X.T@z,z.T@z); RSS=z.T@z-2beta.T@X.T@z+beta.T@X.T@X@beta',
        'counts':'Three rank3 ten-point blocks need4 statistics each; three one-point acoustic blocks retain their single observed values. Total33 ->15.',
        'unknown_noise_warning':'OLS coefficients alone omit seven-dimensional residual energy per mechanical block and are not sufficient for unknown sigma.',
        'posterior_preservation':'Fisher-Neyman likelihood factorization for this declared fixed-design Gaussian model; no true parameters required.',
        'scope_limit':'This proof is specific to the current analytical Gaussian likelihood; not a promise for future physical surrogate providers.',
        'pipeline_status':'Only algebra/geometry computed here. No transformed bank, PFN predictions or acceptance result is produced.'},
      'exact_graph_proposal':{'identity':'p(mu,local_blocks|all_y) = p(mu|all_y) product_i p(local_i,sigma_i|mu,y_i)',
        'proof_basis':'Declared local priors independent conditional on globalmu, independent noise priors, and dataset-local likelihood factors.',
        'implication':'Global stages still use all observations; later block stages may use only globalmu, own observations and within-block prefix without changing the population conditional target. Proposed only; estimator accuracy untested.'},
      'banks':[],'coverage':[],'source_audit':{}}
    max_identity=0.
    for seed in (20261111,20261112,20261113):
        bank=load_context(banks,seed,4096);obs=bank.obs
        covariance=np.cov(obs,rowvar=False);scales=obs.std(axis=0,ddof=1)
        standardized=(obs-obs.mean(axis=0))/scales
        corr=np.cov(standardized,rowvar=False)
        summary_columns=[];block_rows=[]
        for block in problem.observation_model.blocks:
            n=block.observation_size
            if n==10:
                lo=10*len(block_rows);values=obs[:,lo:lo+n]-block.offset
                q,r=np.linalg.qr(block.design,mode='reduced')
                signal=values@q
                residual=values-signal@q.T
                energy=np.square(residual).sum(axis=1)
                summary_columns.extend([signal,energy[:,None]])
                block_rows.append({'dataset':block.dataset_id,'n':n,'design_rank':int(np.linalg.matrix_rank(block.design)),
                    'design_condition_number':float(np.linalg.cond(block.design)),
                    'residual_degrees_of_freedom':n-block.design.shape[1],
                    'minimum_residual_energy':float(energy.min()),
                    'mean_abs_offdiag_observation_correlation':float((np.abs(np.corrcoef(values,rowvar=False)).sum()-n)/(n*(n-1)))})
            else:
                index=problem.dataset_ids.index(block.dataset_id)
                offset=sum(problem.observation_model.blocks[j].observation_size for j in range(index))
                summary_columns.append(obs[:,offset:offset+1])
        summaries=np.concatenate(summary_columns,axis=1)
        sample_unique=[len(np.unique(obs[:,j].astype(np.float32))) for j in range(33)]
        target_unique=[len(np.unique(bank.theta[:,j].astype(np.float32))) for j in range(21)]
        result['banks'].append({'seed':seed,'context_count':len(obs),
          'raw_covariance_geometry':_effective_rank(covariance),
          'standardized_observation_geometry':_effective_rank(corr),
          'minimum_float32_unique_observation_values':min(sample_unique),
          'minimum_float32_unique_target_values':min(target_unique),
          'categorical_inference_caution':'All source-continuous features exceed default30-unique heuristic; no obvious discretization into categorical columns.',
          'mechanical_blocks':block_rows,'likelihood_sufficient_dimension':summaries.shape[1]})
        chol=np.linalg.cholesky(corr+1e-10*np.eye(33))
        for case in range(8):
            heldout=load_heldout(banks,case);observations=unflatten_observations(heldout.obs)
            logs=problem.log_likelihood(bank.theta,observations)
            weights=np.exp(logs-logsumexp(logs));ess=float(1/np.square(weights).sum())
            centered=(obs-heldout.obs)/scales
            distances=np.sum(np.linalg.solve(chol,centered.T).T**2,axis=1)
            arrays,manifest=load_reference_distribution(Path(references)/f'case-{case}')
            z=(bank.theta-arrays['mean'])/arrays['sd']
            qdist=np.square(z).sum(axis=1)
            reconstructed=np.zeros(len(bank.theta))
            for block,noise in zip(problem.observation_model.blocks,problem.prior.noise_names):
                zobs=observations.values[block.dataset_id][0]-block.offset
                beta=bank.theta[:,[problem.parameter_order.index(name) for name in block.local_names]]
                xtz=block.design.T@zobs
                sse=float(zobs@zobs)-2*(beta@xtz)+np.einsum('ni,ij,nj->n',beta,block.design.T@block.design,beta)
                sigma=bank.theta[:,problem.parameter_order.index(noise)]
                reconstructed+=-.5*sse/(sigma*sigma)-block.observation_size*(np.log(sigma)+.5*np.log(2*np.pi))
            delta=float(np.max(np.abs(logs-reconstructed)));max_identity=max(max_identity,delta)
            if not np.allclose(logs,reconstructed,rtol=1e-12,atol=1e-9):raise RuntimeError('sufficient-statistic likelihood identity failed')
            result['coverage'].append({'seed':seed,'case':case,'likelihood_importance_ess':ess,
              'largest_normalized_likelihood_weight':float(weights.max()),
              'top10_weight_mass':float(np.sort(weights)[-10:].sum()),
              'observation_mahalanobis_nn_squared':np.sort(distances)[:5].tolist(),
              'minimum_reference_sd_scaled_parameter_squared_distance':float(qdist.min()),
              'worlds_with_all21_within_2_reference_sd':int(np.all(np.abs(z)<=2,axis=1).sum()),
              'worlds_with_all15_physics_within_2_reference_sd':int(np.all(np.abs(z[:,[problem.parameter_order.index(name) for name in problem.prior.base.parameter_order]])<=2,axis=1).sum()),
              'source_likelihood_vs_sufficient_formula_max_abs':delta,
              'context_payload_sha256':bank.manifest.get('sha256'),
              'heldout_payload_sha256':heldout.manifest.get('sha256'),
              'reference_manifest_sha256':sha256_file(Path(references)/f'case-{case}/manifest.json')})
    regressor=Path('/scratch/project/eu-26-17/eubrieucb/posteriorforge/envs/inference/lib/python3.11/site-packages/tabpfn/regressor.py')
    result['source_audit']={'regressor_sha256':sha256_file(regressor),
      'target_scale_reconstruction':'Pinned2.0.9 fit standardizes y and renormalizes criterion borders using y_train_std/mean; predict(full) returns that renormalized criterion plus merged logits in physical target units.',
      'target_transform_reconstruction':'Predict translates target-transformed ensemble probabilities back across borders before ensemble merging. No missing inverse-target transformation is apparent in the sampled API path.',
      'scientific_prior_vs_task_prior':'Banks encode the scientific prior through paired prior-predictive worlds. The frozen checkpoint pretraining task distribution/inductive bias is a separate object and is not changed by supplying the correct scientific prior.',
      'limitations':'Source audit is not a proof of all predictive-distribution internals. No alternative target/observation representation was fed to the PFN.'}
    result['sufficiency']['numeric_likelihood_identity_max_abs']=max_identity
    result['elapsed_seconds']=time.perf_counter()-started
    _plot(result,root)
    write_json(root/'model-representation.json',result)
    return result


def _plot(result,root):
    os.environ.setdefault('MPLCONFIGDIR',str(Path(os.environ.get('PF_SCRATCH','/scratch/project/eu-26-17/eubrieucb/posteriorforge'))/'cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure,axes=plt.subplots(1,2,figsize=(13,4.6))
    seeds=[20261111,20261112,20261113]
    for seed in seeds:
        values=[r for r in result['coverage'] if r['seed']==seed]
        axes[0].plot([r['case'] for r in values],[r['likelihood_importance_ess'] for r in values],marker='o',label=str(seed))
        axes[1].plot([r['case'] for r in values],[r['minimum_reference_sd_scaled_parameter_squared_distance'] for r in values],marker='o',label=str(seed))
    axes[0].set(ylabel='Likelihood-weight importance ESS out of4096',xlabel='Held-out case',title='Finite-bank likelihood concentration')
    axes[0].set_yscale('log')
    axes[1].set(ylabel='Minimum squared distance in reference-SD units',xlabel='Held-out case',title='Nearest complete parameter world (diagonal scale only)')
    for a in axes:a.grid(alpha=.2);a.legend(title='Context seed')
    figure.suptitle('Built with PriorLabs-TabPFN\nSaved-bank coverage diagnostics; no PFN run or inference change',fontsize=14)
    figure.text(.5,.01,'Low importance ESS is not proof of PFN failure: the transformer learns conditional relationships rather than importance-resampling these worlds.',ha='center',fontsize=10)
    figure.tight_layout(rect=(0,.07,1,.88));name='model-context-coverage.png';figure.savefig(root/name,dpi=160);plt.close(figure)
    result['figures']={name:sha256_file(root/name)}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('banks','references','output'):parser.add_argument('--'+name,required=True)
    print(json.dumps(run(**vars(parser.parse_args())),indent=2))
