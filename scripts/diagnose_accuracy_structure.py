#!/usr/bin/env python3
"""Saved-data scientific audit; no inference, reference integration or gate changes."""
from __future__ import annotations
import argparse, csv, hashlib, json, math
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal, norm, spearmanr, truncnorm
from posteriorforge.artifacts import sha256_file, write_json
from posteriorforge.banks import CONTEXT_SEEDS, load_context, load_heldout
from posteriorforge.gate import verify_refined_grid, verify_run_binding
from posteriorforge.inference import support_mask
from posteriorforge.reference import load_reference_distribution
from posteriorforge.science.analytical_benchmark_v2 import (
    DATASET_IDS, OBSERVATION_COUNTS, array_descriptor, build_problem,
    flatten_observations, unflatten_observations,
)

ATTRIBUTION='Built with PriorLabs-TabPFN'
SCHEMA='posteriorforge.failure_science_diagnostic.v1'
QUANTILES=np.asarray([.05,.25,.5,.75,.95])


def jsonable(value):
    if isinstance(value,dict):return {key:jsonable(item) for key,item in value.items()}
    if isinstance(value,(list,tuple)):return [jsonable(item) for item in value]
    if isinstance(value,np.generic):return value.item()
    return value


def group_of(name):
    if name.startswith('mu_'): return 'population globals'
    if name.startswith('sigma_m'): return 'mechanical noise'
    if name.startswith('sigma_a'): return 'acoustic noise'
    if '_m' in name: return 'mechanical coefficients'
    if '_a' in name: return 'acoustic locations'
    raise ValueError(name)


def prior_marginals(problem):
    p=problem.prior.base
    means=dict(zip(p.global_names,p.global_mean))
    variances=dict(zip(p.global_names,np.diag(p.global_covariance)))
    for block in p.conditional_blocks:
        mean=block.population_map@p.global_mean+block.offset
        variance=np.diag(block.population_map@p.global_covariance@block.population_map.T+block.covariance)
        means.update(zip(block.local_names,mean));variances.update(zip(block.local_names,variance))
    for name in problem.prior.noise_names:
        mean,sd=problem.prior.noise_gaussian_priors[problem.prior.noise_names.index(name)]
        lo,hi=problem.prior.bounds
        means[name],variances[name]=truncnorm.stats((lo-mean)/sd,(hi-mean)/sd,loc=mean,scale=sd,moments='mv')
    return np.asarray([means[n] for n in problem.parameter_order]),np.sqrt([variances[n] for n in problem.parameter_order])


def independent_physics(theta,order):
    position={n:i for i,n in enumerate(order)}
    x=np.linspace(0.,1.,10)
    result={}
    for i in range(1,4):
        result[f'mechanical_{i}']=(theta[:,position[f'a_m{i}'],None]*x
            +theta[:,position[f'b_m{i}'],None]*x*x+theta[:,position[f'd0_m{i}'],None])
        result[f'acoustic_{i}']=(1.25+.15*i)*theta[:,position[f'a_a{i}'],None]+(3.+.1*i)
    return result


def independent_likelihood(theta,observed,order):
    means=independent_physics(theta,order)
    position={n:i for i,n in enumerate(order)}
    total=np.zeros(len(theta))
    for dataset,count in zip(DATASET_IDS,OBSERVATION_COUNTS):
        family,index=dataset.split('_')
        name=f'sigma_{"m" if family=="mechanical" else "a"}{index}'
        sigma=theta[:,position[name]]
        residual=observed.values[dataset]-means[dataset]
        total+=np.sum(norm.logpdf(residual,loc=0,scale=sigma[:,None]),axis=1)
    return total


def independent_prior(theta,problem):
    order=problem.parameter_order;position={n:i for i,n in enumerate(order)}
    p=problem.prior.base
    globals_=theta[:,[position[n] for n in p.global_names]]
    total=multivariate_normal.logpdf(globals_,mean=p.global_mean,cov=p.global_covariance)
    for block in p.conditional_blocks:
        local=theta[:,[position[n] for n in block.local_names]]
        residual=local-globals_@block.population_map.T-block.offset
        total+=multivariate_normal.logpdf(residual,mean=np.zeros(len(block.local_names)),cov=block.covariance)
    lo,hi=problem.prior.bounds
    for name,(mean,sd) in zip(problem.prior.noise_names,problem.prior.noise_gaussian_priors):
        total+=truncnorm.logpdf(theta[:,position[name]],(lo-mean)/sd,(hi-mean)/sd,loc=mean,scale=sd)
    return total


def cdf_quantiles(physical_queries,cdf):
    # Numerical inverse of a qualified saved table; diagnostic, never used by gate.
    probabilities=np.maximum.accumulate(np.clip(cdf,0.,1.))
    unique,indices=np.unique(probabilities,return_index=True)
    return np.interp(QUANTILES,unique,physical_queries[indices])


def summarize(values):
    a=np.asarray(values,dtype=float)
    return {'min':float(a.min()),'median':float(np.median(a)),'max':float(a.max())}


def write_csv(path,rows):
    with Path(path).open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def run(root,output):
    root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    problem=build_problem();order=problem.parameter_order
    prior_mean,prior_sd=prior_marginals(problem)
    baseline=root/'results';gate_path=baseline/'gate-final.json'
    gate=json.loads(gate_path.read_text())
    assert len(gate['runs'])==24
    checks={'input_gate_sha256':sha256_file(gate_path),'baseline_unchanged':True,
        'no_model_calls':True,'no_gpu_jobs':True,'no_reference_evaluations':True,
        'scientific_fingerprint':problem.scientific_fingerprint,
        'source_sha256':sha256_file(__file__), 'attribution':ATTRIBUTION}
    all_noise=[];prior_errors=[];likelihood_errors=[];forward_errors=[];noise_error_records=[]
    inputs={};contexts={}
    for seed in CONTEXT_SEEDS:
        bank=load_context(baseline/'banks-v1',seed,4096);contexts[seed]=bank
        theta=bank.theta;pred=independent_physics(theta,order)
        source=problem.observation_model.predict(theta)
        forward_errors.append(max(float(np.max(np.abs(source[n]-pred[n]))) for n in DATASET_IDS))
        prior_errors.append(float(np.max(np.abs(independent_prior(theta[:256],problem)-problem.log_prior(theta[:256])))))
        for dataset,count in zip(DATASET_IDS,OBSERVATION_COUNTS):
            family,index=dataset.split('_');noise=f'sigma_{"m" if family=="mechanical" else "a"}{index}'
            sigma=theta[:,order.index(noise)]
            residual=(bank.obs[:,sum(OBSERVATION_COUNTS[:DATASET_IDS.index(dataset)]):sum(OBSERVATION_COUNTS[:DATASET_IDS.index(dataset)+1])]-pred[dataset])/sigma[:,None]
            all_noise.append((dataset,residual.ravel()))
            noise_error_records.append({'seed':seed,'dataset':dataset,'n':residual.size,'standardized_mean':float(residual.mean()),'standardized_sd':float(residual.std(ddof=1)),
                'maximum_float32_observation_roundoff_in_noise_sd':float(np.max(np.abs(bank.obs.astype(np.float32).astype(np.float64)-bank.obs)[:,sum(OBSERVATION_COUNTS[:DATASET_IDS.index(dataset)]):sum(OBSERVATION_COUNTS[:DATASET_IDS.index(dataset)+1])]/sigma[:,None]))})
        for case in range(8):
            observed=unflatten_observations(load_heldout(baseline/'banks-v1',case).obs)
            likelihood_errors.append(float(np.max(np.abs(independent_likelihood(theta[:128],observed,order)-problem.log_likelihood(theta[:128],observed)))))
        inputs[f'context-{seed}']=bank.manifest['sha256']
    probe=contexts[CONTEXT_SEEDS[0]].theta[:8].copy()
    probe[0,0]=-1e6;probe[1,order.index('sigma_m1')]=.009;probe[2,order.index('sigma_m1')]=.501
    probe[3,order.index('sigma_m1')]=.01;probe[4,order.index('sigma_m1')]=.5;probe[5,0]=np.nan
    manual_support=np.isfinite(probe).all(axis=1)
    manual_support&=((probe[:,[order.index(n) for n in problem.prior.noise_names]]>=.01)&(probe[:,[order.index(n) for n in problem.prior.noise_names]]<=.5)).all(axis=1)
    checks.update(independent_forward_max_abs_error=max(forward_errors),
        independent_prior_max_abs_logdensity_error=max(prior_errors),
        independent_likelihood_max_abs_logdensity_error=max(likelihood_errors),
        exact_finite_six_noise_support_agreement=bool(np.array_equal(manual_support,support_mask(probe))),
        physical_gaussian_unbounded_support_probe_passed=bool(support_mask(probe)[0]),
        noise_boundary_probes={'lower_inclusive':bool(support_mask(probe)[3]),'upper_inclusive':bool(support_mask(probe)[4])})
    assert max(forward_errors)<1e-12 and max(prior_errors)<1e-10 and max(likelihood_errors)<1e-9
    assert checks['exact_finite_six_noise_support_agreement']
    refs={};refparents={};refcovs={};source_cov_precision=[]
    phys_order=problem.prior.base.parameter_order
    for case in range(8):
        parentroot=baseline/f'references-v1/case-{case}';gridroot=baseline/f'reference-grids-v1/case-{case}'
        selected=next(entry for entry in gate['references'] if entry['case']==case)
        assert sha256_file(parentroot/'manifest.json')==selected['manifest_sha256']
        assert sha256_file(gridroot/'manifest.json')==selected['grid_manifest_sha256']
        parent,pm=load_reference_distribution(parentroot);arrays,gm=load_reference_distribution(gridroot)
        assert sha256_file(parentroot/'combined/manifest.json')==pm['combined_node_manifest_sha256']
        for node in gm['retained_node_manifests']:
            assert sha256_file(parentroot/node['directory']/'manifest.json')==node['manifest_sha256']
        verify_refined_grid(arrays,gm,parent,pm,sha256_file(parentroot/'manifest.json'))
        refs[case]=arrays;refparents[case]=gm
        combined=json.loads((parentroot/'combined/manifest.json').read_text())['metadata']
        covariance=np.asarray(combined['physics_covariance']);means=np.asarray(combined['physics_mean'])
        refcovs[case]=covariance
        assert np.allclose(means,arrays['mean'][[order.index(n) for n in phys_order]],rtol=0,atol=1e-12)
        assert np.allclose(np.sqrt(np.diag(covariance)),arrays['sd'][[order.index(n) for n in phys_order]],rtol=0,atol=1e-12)
        final_power=pm['refinement']['history'][-1]['power']
        errors=[]
        scale=np.sqrt(np.diag(covariance))
        for scramble in range(4):
            m=json.loads((parentroot/f'nodes-p{final_power}-s{scramble}/manifest.json').read_text())['metadata']
            errors.append(float(np.max(np.abs(np.asarray(m['physics_covariance'])-covariance)/np.outer(scale,scale))))
        source_cov_precision.append({'case':case,'max_final_scramble_covariance_difference_in_ref_sd_products':max(errors)})
        inputs[f'reference-grid-{case}']=sha256_file(gridroot/'manifest.json')
    rows=[];mechanical=[];independent_cdf_max_error=0.;independent_moment_max_error=0.
    for entry in gate['runs']:
        seed,case=entry['context_seed'],entry['case']
        folder=baseline/'upstream-v1'/Path(entry['manifest_file']).parent
        manifest=json.loads((folder/'manifest.json').read_text());query=load_heldout(baseline/'banks-v1',case).obs
        assert sha256_file(folder/'manifest.json')==entry['manifest_sha256']
        assert sha256_file(folder/'samples.npz')==manifest['samples_sha256']
        verify_run_binding(manifest,contexts[seed],query,seed,case)
        with np.load(folder/'samples.npz',allow_pickle=False) as f: samples=f['samples'].copy()
        assert array_descriptor(samples)==manifest['samples'] and support_mask(samples).all()
        arrays=refs[case];precision=refparents[case]['cdf_precision']
        sample_mean=samples.astype(np.longdouble).mean(axis=0).astype(np.float64)
        sample_sd=np.sqrt(np.square(samples.astype(np.longdouble)-sample_mean).sum(axis=0)/(len(samples)-1)).astype(np.float64)
        independent_moment_max_error=max(independent_moment_max_error,float(np.max(np.abs(sample_mean-np.asarray(entry['assessment']['moments']['sample_mean'])))),float(np.max(np.abs(sample_sd-np.asarray(entry['assessment']['moments']['sample_sd'])))))
        sample_q=np.quantile(samples,QUANTILES,axis=0)
        for j,name in enumerate(order):
            x=arrays['queries']/arrays['directions'][j,j];cdf=arrays['cdfs'][j]
            sorted_samples=np.sort(samples[:,j]);left=np.searchsorted(sorted_samples,x,side='left')/len(samples);right=np.searchsorted(sorted_samples,x,side='right')/len(samples)
            delta=np.maximum(np.abs(left-cdf),np.abs(right-cdf));point=float(delta.max());cdf_lower=max(0.,point-precision)
            independent_cdf_max_error=max(independent_cdf_max_error,abs(cdf_lower-entry['assessment']['marginal_cdf']['lower'][j]))
            reference_q=cdf_quantiles(x,cdf)
            if name in problem.prior.noise_names:
                mean,sd=problem.prior.noise_gaussian_priors[problem.prior.noise_names.index(name)]
                prior_cdf=truncnorm.cdf(x,(.01-mean)/sd,(.5-mean)/sd,loc=mean,scale=sd)
            else: prior_cdf=norm.cdf(x,loc=prior_mean[j],scale=prior_sd[j])
            shift=sample_mean[j]-arrays['mean'][j]
            row={'seed':seed,'case':case,'stage':j+1,'parameter':name,'group':group_of(name),
                'cdf_lower':float(entry['assessment']['marginal_cdf']['lower'][j]),'cdf_upper':float(entry['assessment']['marginal_cdf']['upper'][j]),
                'cdf_point_discrepancy':point,'cdf_signed_gap_at_maximum':float((right-cdf)[int(np.argmax(delta))]),
                'signed_bias_reference_sd':float(shift/arrays['sd'][j]),'absolute_bias_reference_sd':float(abs(shift)/arrays['sd'][j]),
                'sd_ratio':float(sample_sd[j]/arrays['sd'][j]),'sample_mean_mc_se_reference_sd':float(sample_sd[j]/math.sqrt(len(samples))/arrays['sd'][j]),
                'reference_sd_over_prior_sd':float(arrays['sd'][j]/prior_sd[j]),'reference_prior_cdf_witness_difference':float(np.max(np.abs(cdf-prior_cdf))),
                'prior_pull_reference_sd':float((prior_mean[j]-arrays['mean'][j])/arrays['sd'][j]),
                'central90_width_ratio_diagnostic':float((sample_q[-1,j]-sample_q[0,j])/(reference_q[-1]-reference_q[0])),
                'median_bias_reference_sd_diagnostic':float((sample_q[2,j]-reference_q[2])/arrays['sd'][j]),
                'sample_q05':float(sample_q[0,j]),'sample_q50':float(sample_q[2,j]),'sample_q95':float(sample_q[4,j]),
                'reference_q05_diagnostic':float(reference_q[0]),'reference_q50_diagnostic':float(reference_q[2]),'reference_q95_diagnostic':float(reference_q[4])}
            rows.append(row)
        for block in range(1,4):
            names=[f'a_m{block}',f'b_m{block}',f'd0_m{block}'];positions=[order.index(n) for n in names];ppos=[phys_order.index(n) for n in names]
            reference_cov=refcovs[case][np.ix_(ppos,ppos)];sample_cov=np.cov(samples[:,positions],rowvar=False,ddof=1)
            eigenvalues,eigenvectors=np.linalg.eigh(reference_cov);whitener=(eigenvectors/np.sqrt(eigenvalues))@eigenvectors.T
            covariance_ratios=np.linalg.eigvalsh(whitener@sample_cov@whitener)
            x=np.linspace(0.,1.,10);design=np.column_stack((x,x*x,np.ones(10)))
            forward_bias=design@(sample_mean[positions]-arrays['mean'][positions]);noise_mean=arrays['mean'][order.index(f'sigma_m{block}')]
            ref_signal_sd=np.sqrt(np.einsum('ni,ij,nj->n',design,reference_cov,design));pfn_signal_sd=np.sqrt(np.einsum('ni,ij,nj->n',design,sample_cov,design))
            ref_corr=reference_cov/np.sqrt(np.outer(np.diag(reference_cov),np.diag(reference_cov)));pfn_corr=sample_cov/np.sqrt(np.outer(np.diag(sample_cov),np.diag(sample_cov)))
            observed_curve=query[0,(block-1)*10:block*10]
            reference_residual_mean=observed_curve-design@arrays['mean'][positions]
            pfn_residual_mean=observed_curve-design@sample_mean[positions]
            reference_expected_rss=float(np.square(reference_residual_mean).sum()+np.trace(design@reference_cov@design.T))
            pfn_expected_rss=float(np.square(pfn_residual_mean).sum()+np.trace(design@sample_cov@design.T))
            mechanical.append({'seed':seed,'case':case,'block':block,'ref_corr_a_b':float(ref_corr[0,1]),'pfn_corr_a_b':float(pfn_corr[0,1]),
                'ref_corr_a_d0':float(ref_corr[0,2]),'pfn_corr_a_d0':float(pfn_corr[0,2]),'ref_corr_b_d0':float(ref_corr[1,2]),'pfn_corr_b_d0':float(pfn_corr[1,2]),
                'smallest_covariance_variance_ratio':float(covariance_ratios[0]),'largest_covariance_variance_ratio':float(covariance_ratios[-1]),
                'expected_residual_energy_ratio_pfn_over_reference':pfn_expected_rss/reference_expected_rss,
                'mean_curve_rms_bias_over_reference_noise_sd':float(np.sqrt(np.mean(forward_bias**2))/noise_mean),
                'mean_curve_max_bias_over_reference_signal_sd':float(np.max(np.abs(forward_bias)/ref_signal_sd)),
                'signal_sd_ratio_min':float(np.min(pfn_signal_sd/ref_signal_sd)),'signal_sd_ratio_max':float(np.max(pfn_signal_sd/ref_signal_sd)),
                'reference_covariance_condition_number':float(eigenvalues[-1]/eigenvalues[0])})
        inputs[f'run-{seed}-{case}']=sha256_file(folder/'manifest.json')
    checks.update(maximum_float32_context_observation_roundoff_in_noise_sd=max(row['maximum_float32_observation_roundoff_in_noise_sd'] for row in noise_error_records),
                  independent_physical_ecdf_lower_vs_gate_max_abs_difference=independent_cdf_max_error,
                  independent_longdouble_moments_vs_gate_max_abs_difference=independent_moment_max_error,
                  all_24_run_bindings_verified=True,all_8_reference_grid_chains_verified=True,
                  stored_parameter_order=list(order),observation_order=list(DATASET_IDS),observation_counts=list(OBSERVATION_COUNTS),
                  posterior_reconstruction='whole-vector support rejection, no finite timeout, no clipping, no log-density calibration')
    assert independent_cdf_max_error<1e-10 and independent_moment_max_error<1e-10
    parameters=[];decomposition=[]
    for j,name in enumerate(order):
        selected=[r for r in rows if r['parameter']==name]
        b=np.asarray([[next(r['signed_bias_reference_sd'] for r in selected if r['seed']==seed and r['case']==case) for seed in CONTEXT_SEEDS] for case in range(8)])
        se=np.asarray([[next(r['sample_mean_mc_se_reference_sd'] for r in selected if r['seed']==seed and r['case']==case) for seed in CONTEXT_SEEDS] for case in range(8)])
        overall=b.mean();case_effect=b.mean(axis=1)-overall;seed_effect=b.mean(axis=0)-overall;interaction=b-overall-case_effect[:,None]-seed_effect[None,:]
        ss_case=3*float(np.square(case_effect).sum());ss_seed=8*float(np.square(seed_effect).sum());ss_interaction=float(np.square(interaction).sum());total=ss_case+ss_seed+ss_interaction
        ratios=[float(abs(b[c,s]-b[c,t])/np.sqrt(se[c,s]**2+se[c,t]**2)) for c in range(8) for s,t in ((0,1),(0,2),(1,2))]
        decomposition.append({'parameter':name,'stage':j+1,'group':group_of(name),'case_fraction':ss_case/total if total else 0.,'bank_seed_main_fraction':ss_seed/total if total else 0.,'case_bank_interaction_fraction':ss_interaction/total if total else 0.,'median_bank_difference_over_empirical_mc_se':float(np.median(ratios)),'max_bank_difference_over_empirical_mc_se':max(ratios),
            'all_three_biases_same_sign_case_count':int(np.sum(np.all(b>0,axis=1)|np.all(b<0,axis=1)))})
        parameters.append({'parameter':name,'stage':j+1,'group':group_of(name),'definite_marginal_failures':sum(r['cdf_lower']>.1 for r in selected),'inconclusive_marginal_cells':sum(r['cdf_lower']<=.1<r['cdf_upper'] for r in selected),
            'median_cdf_upper':float(np.median([r['cdf_upper'] for r in selected])),'max_cdf_upper':max(r['cdf_upper'] for r in selected),
            'median_signed_bias_reference_sd':float(np.median([r['signed_bias_reference_sd'] for r in selected])),
            'median_absolute_bias_reference_sd':float(np.median([r['absolute_bias_reference_sd'] for r in selected])),
            'median_sd_ratio':float(np.median([r['sd_ratio'] for r in selected])),
            'median_reference_sd_over_prior_sd':float(np.median([r['reference_sd_over_prior_sd'] for r in selected])),
            'median_prior_reference_cdf_witness_difference':float(np.median([r['reference_prior_cdf_witness_difference'] for r in selected])),
            'median_central90_width_ratio_diagnostic':float(np.median([r['central90_width_ratio_diagnostic'] for r in selected]))})
    groups={}
    for group in dict.fromkeys(group_of(n) for n in order):
        selected=[r for r in rows if r['group']==group]
        groups[group]={'cells':len(selected),'definite_cdf_failures':sum(r['cdf_lower']>.1 for r in selected),'inconclusive_cdf_cells':sum(r['cdf_lower']<=.1<r['cdf_upper'] for r in selected),
            'cdf_upper':summarize([r['cdf_upper'] for r in selected]),'absolute_bias_reference_sd':summarize([r['absolute_bias_reference_sd'] for r in selected]),
            'sd_ratio':summarize([r['sd_ratio'] for r in selected]),'reference_sd_over_prior_sd':summarize([r['reference_sd_over_prior_sd'] for r in selected]),
            'prior_reference_cdf_witness_difference':summarize([r['reference_prior_cdf_witness_difference'] for r in selected])}
    bias_failure_cells=[r for r in rows if r['absolute_bias_reference_sd']>.2]
    prior_pull_aligned=sum(np.sign(r['signed_bias_reference_sd'])==np.sign(r['prior_pull_reference_sd']) for r in bias_failure_cells)
    stage_corr=spearmanr([r['stage'] for r in rows],[r['cdf_upper'] for r in rows])
    noise_summary=[]
    for dataset in DATASET_IDS:
        e=np.concatenate([v for n,v in all_noise if n==dataset]);noise_summary.append({'dataset':dataset,'n':len(e),'standardized_mean':float(e.mean()),'standardized_sd':float(e.std(ddof=1))})
    result={'schema':SCHEMA,'attribution':ATTRIBUTION,'label':'post-failure CPU diagnostic, not a new inference variant or gate',
        'checks':checks,'inputs':inputs,'groups':groups,'parameters':parameters,'bank_case_decomposition':decomposition,
        'mechanical_joint_summary':{key:summarize([r[key] for r in mechanical]) for key in ('ref_corr_a_b','pfn_corr_a_b','smallest_covariance_variance_ratio','largest_covariance_variance_ratio','mean_curve_rms_bias_over_reference_noise_sd','mean_curve_max_bias_over_reference_signal_sd','signal_sd_ratio_max','expected_residual_energy_ratio_pfn_over_reference')},
        'reference_covariance_precision_diagnostic':source_cov_precision,
        'prior_pull_alignment_among_point_bias_failure_cells':{'aligned':prior_pull_aligned,'cells':len(bias_failure_cells),'fraction':prior_pull_aligned/len(bias_failure_cells)},
        'stage_rank_correlation_diagnostic':{'spearman_rho':float(stage_corr.statistic),'inference':'descriptive only; repeated case/seed cells are not independent significance-test observations'},
        'noise_residuals':noise_summary,
        'limitations':['No new model call or alternative representation evaluated.','Reference inverse-CDF quantiles and covariance whitening are diagnostic, not additional gate criteria.','Case/bank decomposition includes Monte Carlo sampling and numerical effects; empirical MC SE is conditional on the fitted PFN and assumes independent complete draws.','Simple monotonic stage-error accumulation is testable as a descriptive pattern; this cannot exclude within-block autoregressive propagation.','No root cause follows from correlation or group concentration alone.']}
    result=jsonable(result)
    write_json(output/'science-audit.json',result)
    write_csv(output/'science-per-cell.csv',rows);write_csv(output/'science-per-parameter.csv',parameters);write_csv(output/'science-bank-case.csv',decomposition);write_csv(output/'science-mechanical-joint.csv',mechanical);write_csv(output/'science-noise-residuals.csv',noise_error_records)
    plots(output,parameters,decomposition,mechanical)
    write_json(output/'science-file-checksums.json',{p.name:sha256_file(p) for p in sorted(output.iterdir()) if p.is_file() and p.name!='science-file-checksums.json'})
    return jsonable({'groups':groups,'mechanical_joint_summary':result['mechanical_joint_summary'],'checks':checks,'stage_correlation':result['stage_rank_correlation_diagnostic'],'prior_pull':result['prior_pull_alignment_among_point_bias_failure_cells'],'files':str(output)})


def plots(output,parameters,decomposition,mechanical):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    stages=np.arange(1,22);names=[p['parameter'] for p in parameters]
    figure,axes=plt.subplots(3,1,figsize=(14,10),sharex=True,layout='constrained')
    axes[0].bar(stages,[p['definite_marginal_failures']/24 for p in parameters],color='#b94844')
    axes[0].set(ylabel='Definite marginal CDF\ndiscrepancy failure fraction',ylim=(0,1.08))
    axes[1].plot(stages,[p['median_absolute_bias_reference_sd'] for p in parameters],'o-',color='#2365aa');axes[1].axhline(.2,color='#b94844',ls='--');axes[1].set(ylabel='Median absolute bias / ref SD')
    axes[2].plot(stages,[p['median_sd_ratio'] for p in parameters],'o-',color='#238578');axes[2].axhspan(.8,1.2,color='#dbeee8',alpha=.6);axes[2].set(ylabel='Median PFN / reference SD',xlabel='Autoregressive stage and physical parameter')
    for ax in axes:
        ax.axvline(15.5,color='#8c96a0',lw=1);ax.grid(axis='y',alpha=.2)
    axes[-1].set_xticks(stages,names,rotation=55,ha='right')
    figure.suptitle(ATTRIBUTION+'\nError is concentrated by parameter group; later acoustic stages are not uniformly worse',fontsize=15)
    figure.savefig(output/'science-error-structure.png',dpi=160,bbox_inches='tight');plt.close(figure)
    figure,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    for block,color in ((1,'#2365aa'),(2,'#da8d24'),(3,'#238578')):
        rows=[r for r in mechanical if r['block']==block]
        axes[0].scatter([r['ref_corr_a_b'] for r in rows],[r['pfn_corr_a_b'] for r in rows],label=f'mechanical {block}',color=color,s=26)
        axes[1].scatter([r['reference_covariance_condition_number'] for r in rows],[r['largest_covariance_variance_ratio'] for r in rows],color=color,s=26)
    axes[0].plot([-1,1],[-1,1],'--',color='#85919f');axes[0].set(xlim=(-1,0),ylim=(-1,0),xlabel='Reference posterior corr(a,b)',ylabel='PFN posterior corr(a,b)');axes[0].legend()
    axes[1].axhline(1,color='#85919f',ls='--');axes[1].set(xscale='log',yscale='log',xlabel='Reference 3-coefficient covariance condition number',ylabel='Largest PFN / reference variance ratio\nin reference-whitened coefficient space')
    figure.suptitle(ATTRIBUTION+'\nSaved mechanical posterior dependence diagnostics (not a new acceptance gate)',fontsize=14)
    figure.savefig(output/'science-mechanical-joint.png',dpi=160,bbox_inches='tight');plt.close(figure)
    figure,axes=plt.subplots(figsize=(14,5),layout='constrained')
    lower=np.zeros(21)
    for key,label,color in (('case_fraction','Between-case effect','#2365aa'),('bank_seed_main_fraction','Bank-seed main effect','#da8d24'),('case_bank_interaction_fraction','Case × bank / residual','#238578')):
        values=np.asarray([r[key] for r in decomposition]);axes.bar(stages,values,bottom=lower,label=label,color=color);lower+=values
    axes.set_xticks(stages,names,rotation=55,ha='right');axes.set(ylabel='Fraction of signed standardized-bias variation',xlabel='Physical parameter',ylim=(0,1));axes.legend(loc='upper center',ncol=3,bbox_to_anchor=(.5,1.14))
    figure.suptitle(ATTRIBUTION+'\nBalanced descriptive decomposition: eight cases × three independent context banks',fontsize=14)
    figure.savefig(output/'science-bank-case-decomposition.png',dpi=160,bbox_inches='tight');plt.close(figure)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();print(json.dumps(run(args.root,args.output),indent=2))
