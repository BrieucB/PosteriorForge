#!/usr/bin/env python3
"""Render authenticated saved 50k execution evidence; no model or integration calls.

Built with PriorLabs-TabPFN. Plots do not decide execution qualification or change
the original physical gate. Reference curves use retained CDF values only.
"""
from __future__ import annotations
import argparse
import csv
import importlib.util
import json
from pathlib import Path
import numpy as np
from posteriorforge.artifacts import sha256_file, write_json
from posteriorforge.inference import support_mask
from posteriorforge.science.analytical_benchmark_v2 import array_descriptor
from posteriorforge.validation import assess_samples, compare_to_upstream


def load_run(path):
    path=Path(path);manifest=json.loads((path/'manifest.json').read_text())
    if manifest.get('status')!='complete' or manifest.get('requested_count')!=50000 or manifest.get('performance_worker') is True or manifest.get('end_to_end') is False:
        raise ValueError('only completed 50000-vector end-to-end records may be plotted')
    payload=path/'samples.npz'
    if sha256_file(payload)!=manifest.get('samples_sha256'):raise ValueError('posterior payload checksum mismatch')
    with np.load(payload,allow_pickle=False) as saved:
        if set(saved.files)!={'samples'}:raise ValueError('unexpected posterior payload')
        values=saved['samples'].copy()
    if values.shape!=(50000,21) or array_descriptor(values)!=manifest['samples'] or not support_mask(values).all():
        raise ValueError('incomplete, unsupported or mismatched posterior rows')
    return values,manifest,{'manifest_sha256':sha256_file(path/'manifest.json'),'samples_sha256':sha256_file(payload),'path':str(path)}


def render(*,control,reference,output,runs=(),qualification=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    control_values,control_manifest,control_binding=load_run(control)
    binding_script=Path(__file__).with_name('evaluate_cdf_pilot.py')
    spec=importlib.util.spec_from_file_location('execution_plot_bindings',binding_script)
    bindings=importlib.util.module_from_spec(spec);spec.loader.exec_module(bindings)
    if control_binding['manifest_sha256']!='a0114fd74721cf9bed31e9131045e841f262db123c8f849e5cb3f15814a14f4b':
        raise ValueError('immutable corrected control manifest required')
    results=Path(reference).resolve().parents[1];gate_path=results/'gate-final.json'
    if sha256_file(gate_path)!=bindings.BASELINE_GATE_SHA256:raise ValueError('original gate changed')
    gate=json.loads(gate_path.read_text())
    from posteriorforge.banks import load_heldout
    query=load_heldout(results/'banks-v1',control_manifest['case']).obs
    if array_descriptor(query)!=control_manifest['query']:raise ValueError('heldout query changed')
    arrays,reference_manifest,ref,custody=bindings.reference_for_case(results,gate,query,control_manifest['case'])
    if sha256_file(Path(reference)/'manifest.json')!=custody['runner_reference_binding']['refined_grid']['manifest_sha256']:
        raise ValueError('supplied reference differs from authenticated refined grid')
    if control_manifest.get('reference_binding')!=custody['runner_reference_binding']:
        raise ValueError('control/reference custody mismatch')
    for key in ('scientific_fingerprint','parameter_order'):
        if reference_manifest['binding'][key]!=control_manifest[key]:raise ValueError('reference scientific binding mismatch')
    data=[('Corrected control',control_values,control_manifest,control_binding)]
    for declaration in runs:
        label,path=declaration.split('=',1);values,manifest,binding=load_run(path)
        for key in ('scientific_fingerprint','parameter_order','context_seed','case','context_size','query','checkpoint_sha256','context_theta','context_obs'):
            if manifest[key]!=control_manifest[key]:raise ValueError('incomparable run binding: '+key)
        data.append((label,values,manifest,binding))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    colors=['#596573','#d26c28','#238e8b','#8757ad','#b74367','#3d80bd']
    assessments=[];timing_rows=[];figures={}
    def save(fig,name):
        fig.savefig(output/name,dpi=155,bbox_inches='tight',facecolor='white');plt.close(fig);figures[name]=sha256_file(output/name)
    for label,values,manifest,binding in data:
        absolute=assess_samples(values,ref,support_mask=support_mask(values))
        relative=dict(compare_to_upstream(values,control_values,reference_sd=ref['sd'],directions=ref['directions']))
        relative['original_api_absolute_gate_policy']=relative['absolute_gate_still_required']
        relative['absolute_gate_still_required']=False
        relative['policy_override']='Explicit human performance-first authorization; physical assessment retained separately.'
        assessments.append({'label':label,'binding':binding,'absolute':absolute,'relative':relative,
                            'absolute_physical_gate_required_for_performance':False})
        t=manifest.get('timings',{});m=manifest.get('memory',{});g=manifest.get('world_size',1)
        request=t.get('posterior_request_seconds');total=manifest.get('wrapper_total_seconds',t.get('total_seconds'))
        timing_rows.append({'label':label,'workers':g,'posterior_request_seconds':request,'total_seconds':total,
            'accepted_per_second':50000/request if request else None,'request_wall_gpu_hours':g*request/3600 if request else None,
            'peak_allocated_gib':m['peak_allocated_bytes']/2**30 if m.get('peak_allocated_bytes') is not None else None,'peak_reserved_gib':m['peak_reserved_bytes']/2**30 if m.get('peak_reserved_bytes') is not None else None,
            'proposed':manifest.get('rejection',{}).get('proposed'),'physical_status':absolute['status']})
    windows=[]
    for j in range(21):
        x=arrays['queries']/arrays['directions'][j,j];cdf=arrays['cdfs'][j]
        lo=min(float(v[:,j].min()) for _,v,_,_ in data[:5]);hi=max(float(v[:,j].max()) for _,v,_,_ in data[:5])
        lo=min(lo,float(x[min(np.searchsorted(cdf,.0005),len(x)-1)]));hi=max(hi,float(x[min(np.searchsorted(cdf,.9995),len(x)-1)]))
        pad=.025*(hi-lo);windows.append((lo-pad,hi+pad))
    fig,axes=plt.subplots(7,3,figsize=(15,23))
    for j,ax in enumerate(axes.flat):
        x=arrays['queries']/arrays['directions'][j,j]
        ax.plot(x,arrays['cdfs'][j],color='#225ca7',lw=1.6,label='Retained physical reference')
        for i,(label,values,_,_) in enumerate(data[:5]):
            ordered=np.sort(values[:,j]);ax.step(np.r_[ordered[0],ordered],np.r_[0,np.arange(1,50001)/50000],where='post',lw=1.1,color=colors[i],label=label)
        ax.set(title=control_manifest['parameter_order'][j],ylim=(0,1),xlim=windows[j],xlabel='Physical value');ax.grid(alpha=.15)
        if j%3==0:ax.set_ylabel('Posterior CDF')
    h,l=axes.flat[0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=3,frameon=False)
    fig.suptitle('Built with PriorLabs-TabPFN\n50,000 complete accepted vectors per curve; physical accuracy remains separately assessed',fontsize=17,y=.998)
    fig.tight_layout(rect=(0,.035,1,.975));save(fig,'posterior-all-21-cdfs.png')
    ranked=np.argsort(-np.asarray(assessments[0]['absolute']['marginal_cdf']['upper']))[:4]
    fig,axes=plt.subplots(2,2,figsize=(13,8.5))
    for j,ax in zip(ranked,axes.flat):
        x=arrays['queries']/arrays['directions'][j,j];lo,hi=windows[j];first=max(0,np.searchsorted(x,lo)-1);last=min(len(x)-1,np.searchsorted(x,hi)+1);selection=np.unique(np.linspace(first,last,min(75,last-first+1)).astype(int));edges=x[selection];width=np.diff(edges)
        ax.stairs(np.diff(arrays['cdfs'][j,selection])/width,edges,color='#225ca7',lw=1.8,label='Reference CDF mass / width')
        for i,(label,values,_,_) in enumerate(data[:5]):
            counts,_=np.histogram(values[:,j],bins=edges);ax.stairs(counts/(50000*width),edges,color=colors[i],lw=1.25,label=label)
        ax.set(title=control_manifest['parameter_order'][j],xlabel='Physical value',ylabel='Probability / bin width');ax.grid(alpha=.15)
    h,l=axes.flat[0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=3,frameon=False)
    fig.suptitle('Four largest control marginal errors: full-count normalization, retained reference bins',fontsize=15)
    fig.tight_layout(rect=(0,.075,1,.96));save(fig,'posterior-worst-four-densities.png')
    valid=[r for r in timing_rows if r['posterior_request_seconds'] is not None]
    if valid:
        labels=[r['label'] for r in valid];x=np.arange(len(valid));fig,axes=plt.subplots(1,3,figsize=(15,5))
        axes[0].bar(x,[r['posterior_request_seconds']/60 for r in valid],color='#d26c28');axes[0].set_ylabel('50k request minutes')
        axes[1].bar(x,[r['accepted_per_second'] for r in valid],color='#238e8b');axes[1].set_ylabel('Accepted vectors / second')
        axes[2].bar(x,[r['request_wall_gpu_hours'] for r in valid],color='#8757ad');axes[2].set_ylabel('Nominal request GPU-hours (workers x wall)' )
        for ax in axes:ax.set_xticks(x,labels,rotation=35,ha='right');ax.grid(axis='y',alpha=.15)
        fig.suptitle('Measured 50,000-vector requests; wall-product costs differ from scheduler allocation accounting')
        fig.tight_layout();save(fig,'runtime-and-request-cost.png')
        fig,ax=plt.subplots(figsize=(11,5));w=.36
        ax.bar(x-w/2,[r['peak_allocated_gib'] if r['peak_allocated_gib'] is not None else np.nan for r in valid],w,label='Peak allocated');ax.bar(x+w/2,[r['peak_reserved_gib'] if r['peak_reserved_gib'] is not None else np.nan for r in valid],w,label='Peak reserved')
        ax.set_xticks(x,labels,rotation=35,ha='right');ax.set_ylabel('GiB per reported device');ax.legend();ax.grid(axis='y',alpha=.15)
        ax.set_title('Measured CUDA memory; missing measurements are omitted, allocator reservation is not live storage');fig.tight_layout();save(fig,'cuda-memory.png')
    if qualification:
        q=json.loads(Path(qualification).read_text());stages=sorted(q['stage_results'],key=lambda r:r['stage_index']);x=np.array([r['stage_index'] for r in stages])
        fig,axes=plt.subplots(1,2,figsize=(13,4.5))
        for key,label in [('control_predict1024_capture_free','Corrected unoptimized'),('cache_predict1024_capture_free','Resident full 8-member')]:
            axes[0].plot(x,[r['timings_seconds'][key] for r in stages],marker='o',label=label)
        axes[0].set(xlabel='Autoregressive stage',ylabel='Fixed 1024-query prediction seconds');axes[0].legend()
        axes[1].plot(x,[r['peak_allocated_bytes']/2**30 for r in stages],marker='o',label='Probe peak allocated')
        axes[1].set(xlabel='Autoregressive stage',ylabel='GiB');axes[1].legend()
        for ax in axes:ax.grid(alpha=.15)
        fig.suptitle('Fixed-conditional engineering probe; not an end-to-end posterior benchmark');fig.tight_layout();save(fig,'qualification-stage-time-memory.png')
        profiles=q.get('profiles',{})
        if profiles:
            fig,axes=plt.subplots(1,len(profiles),figsize=(7*len(profiles),5),squeeze=False)
            for ax,(label,p) in zip(axes.flat,profiles.items()):
                rows=p['summary']['top_kernels'][:7];values=[r['summed_duration_seconds'] for r in rows]
                ax.barh(np.arange(len(rows)),values,color='#238e8b');ax.set_yticks(np.arange(len(rows)),[r['name'][:75] for r in rows],fontsize=7);ax.invert_yaxis();ax.set(xlabel='Summed kernel seconds (may overlap)',title=label)
            fig.suptitle('Actual bounded CPU/CUDA trace windows; query counts and capture scope retained in provenance');fig.tight_layout();save(fig,'bounded-trace-kernels.png')
    with (output/'timings.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(timing_rows[0]));writer.writeheader();writer.writerows(timing_rows)
    meta={'schema':'posteriorforge.execution_figures.v1','attribution':'Built with PriorLabs-TabPFN','script_sha256':sha256_file(__file__),
        'reference_manifest_sha256':sha256_file(Path(reference)/'manifest.json'),'reference_recomputed':False,'samples_clipped_or_repaired':False,
        'control':control_binding,'reference_custody':custody,'original_gate_sha256':sha256_file(gate_path),'binding_script_sha256':sha256_file(binding_script),'plotted_posterior_cohorts':[label for label,_,_,_ in data[:5]],'worst_four_parameter_indices':ranked.tolist(),'density_windows':windows,'assessments':assessments,'timings':timing_rows,'figures':figures,
        'qualification_manifest_sha256':sha256_file(qualification) if qualification else None,
        'scope':'Saved finite-cohort visual evidence; not population equivalence, global calibration, or an execution acceptance decision.'}
    write_json(output/'figure-data.json',meta)
    lines=['# PosteriorForge performance evidence','','Built with PriorLabs-TabPFN. Every posterior curve contains 50,000 complete accepted physical vectors. Runtime qualification and absolute physical accuracy are separate decisions. These plots alone do not establish an optimized execution pass.','','| Run | Request seconds | Vectors/s | Physical result |','|---|---:|---:|---|']
    for r in timing_rows:lines.append(f"| {r['label']} | {r['posterior_request_seconds']} | {r['accepted_per_second']} | {r['physical_status']} |")
    lines.extend(['','| Run | Max two-sample CDF | Max mean change / reference SD | Max SD fractional change | Investigation flag |','|---|---:|---:|---:|---|'])
    for item in assessments:
        relative=item['relative']
        lines.append(f"| {item['label']} | {max(relative['cdf_difference']):.6g} | {max(relative['mean_difference_reference_sd']):.6g} | {max(relative['sd_fraction_difference']):.6g} | {relative['investigation_required']} |")
    lines.extend(['','Investigation thresholds remain 0.01 CDF, 0.05 reference SD in mean, and 5% in SD. Repeated controls can also raise flags; every flagged comparison remains an investigation, not an automatic equivalence claim. No reported vector or outlier is removed.'])
    for name in figures:lines.extend(['',f'![{name.removesuffix(".png").replace("-"," ")}]({name})'])
    lines.extend(['','All densities divide by the complete draw count, including mass outside the displayed retained grid. Reference curves use saved values; no new reference integration, clipping, tail repair or physical gate changes were performed. Detailed input/output SHA256 values and absolute assessments are in figure-data.json.'])
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    return {'output':str(output),'figures':figures}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('control','reference','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--run',action='append',dest='runs',default=[]);p.add_argument('--qualification')
    print(json.dumps(render(**vars(p.parse_args())),indent=2))
