"""Render the closed A/B/C CPU diagnosis only; no new numerical evaluation.

Built with PriorLabs-TabPFN. Values and noise-only statuses come directly from
SHA-authenticated primary JSON. No arrays/models/codec/RNG are evaluated.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math

REPORT_SHA='cbc8e44e62780d48e594845bd63a9c72faa76eaf7a1459d9e4dbb2210e989a5a'
VARIANTS=(
 ('A_captured_final_CPU_F32_weights','A: captured final F32 weights; exact-normalized ICDF','#333333',-.13),
 ('B_CPU_F32_pre_softmax_replay_F64log_CPUsoftmax64','B: CPU F32 softmax replay; F64 postprocessing','#d95f02',0.),
 ('C_softmax64_adjusted_F32logits_F64log_CPUsoftmax64','C: same adjusted logits; F64 softmax/postprocessing','#0072b2',.13))
OWN_U={15:'e0ff7f3f',19:'f6ff7f3f'}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def selected_rows(document):
    rows=[r for r in document['records'] if r['label']=='N1024']
    if len(rows)!=32:raise ValueError('exact32 N1024 stage/arm/member rows required')
    result=[]
    for stage in (15,19):
        for arm in ('original','candidate'):
            group=sorted([r for r in rows if r['stage']==stage and r['arm']==arm],key=lambda r:r['member'])
            if [r['member'] for r in group]!=list(range(8)):raise ValueError('all8 fixed member indices required')
            for row in group:
                points=[]
                for key,_,_,_ in VARIANTS:
                    found=[p for p in row['variants'][key]['points'] if p['U_bits']==OWN_U[stage]]
                    if len(found)!=1 or not math.isfinite(found[0]['value']):raise ValueError('one saved finite own-U quantile required')
                    point=found[0]
                    points.append({'variant':key,'value':point['value'],'U':point['U'],'U_bits':point['U_bits'],'noise_status':point['d8_noise']})
                result.append({'stage':stage,'arm':arm,'member':row['member'],'points':points})
    return result


def render(report,output):
    report=Path(report);output=Path(output)
    if sha(report)!=REPORT_SHA:raise ValueError('exact closed primary report required')
    d=json.loads(report.read_bytes())
    if d['status']!='CPU_diagnosis_complete':raise ValueError('closed numerical diagnosis required')
    rows=selected_rows(d)
    if output.exists():raise FileExistsError('fresh figure output required')
    output.mkdir(parents=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11})
    fig,axes=plt.subplots(2,2,figsize=(12.2,8.8),sharex=True,sharey='row')
    for i,stage in enumerate((15,19)):
        for j,arm in enumerate(('original','candidate')):
            ax=axes[i,j];group=[r for r in rows if r['stage']==stage and r['arm']==arm]
            counts32=[];counts64=[]
            for variant_index,(key,_,color,offset) in enumerate(VARIANTS):
                points=[r['points'][variant_index] for r in group];x=[r['member']+offset for r in group];y=[p['value'] for p in points]
                ax.plot(x,y,color=color,lw=1.15,alpha=.72)
                for a,b,p in zip(x,y,points):
                    status=p['noise_status']
                    if not status['decode_accepted']:marker,face='s','none'
                    elif not status['RT32_pass']:marker,face='x',color
                    elif not status['RT64_pass']:marker,face='^',color
                    else:marker,face='o',color
                    ax.plot(a,b,marker=marker,ms=6.3,color=color,markerfacecolor=face,mew=1.25,ls='none')
                counts32.append(sum(bool(p['noise_status']['RT32_pass']) for p in points))
                counts64.append(sum(bool(p['noise_status']['RT64_pass']) for p in points))
            name='sigma_a1' if stage==15 else 'sigma_a3';row=4355 if stage==15 else 41765
            ax.set_title(f'Stage {stage} ({name}), {arm} FP32\nSaved prefix row {row}; own U={points[0]["U"]:.16g}')
            ax.set_xticks(range(8));ax.set_xlim(-.55,7.55);ax.grid(axis='y',alpha=.22)
            ax.text(.02,.97,'Reported RT32 passes A/B/C: '+ '/'.join(map(str,counts32))+' of 8\n'
                    +'Reported RT64 passes A/B/C: '+ '/'.join(map(str,counts64))+' of 8',
                    transform=ax.transAxes,va='top',fontsize=8.7,
                    bbox={'facecolor':'white','alpha':.82,'edgecolor':'none','pad':3})
            ax.margins(y=.23)
            if j==0:ax.set_ylabel('Latent quantile z')
            if i==1:ax.set_xlabel('Fixed ensemble member index')
    fig.suptitle('Saved A/B/C postprocessing diagnosis — seed 20261111, case 0\n'
                 'N1024 counterfactual prefix reevaluations; all eight members in both arms',fontsize=14,y=.985)
    lines=[Line2D([0],[0],color=c,lw=2,label=label) for _,label,c,_ in VARIANTS]
    statuses=[Line2D([0],[0],marker=m,color='#555555',ls='none',markerfacecolor=face,label=label)
              for m,face,label in [('o','#555555','Both noise RT checks pass'),('^','#555555','RT32 passes; RT64 fails'),('x','#555555','Decode accepted; RT32 fails'),('s','none','Noise decode rejected')]]
    fig.legend(handles=lines,loc='lower center',bbox_to_anchor=(.5,.12),frameon=False,ncol=1,fontsize=9)
    fig.legend(handles=statuses,loc='lower center',bbox_to_anchor=(.5,.08),frameon=False,ncol=4,fontsize=8.5)
    fig.text(.5,.021,'B replay p sums above 1; the source CDF clamp removes its far tail; its RT passes do not validate a policy.\n'
             'B replays CPU F32 pretranslation softmax: actual GPU pretranslation weights were not captured.\n'
             'C uses the SAME already temperature-adjusted F32 logits; no second temperature division.\n'
             'Statuses are copied from primary noise-only diagnostics; no physical21 cohort, new gate, repair or posterior-accuracy claim.  Built with PriorLabs-TabPFN.',
             ha='center',va='bottom',fontsize=8.25,color='#444444')
    fig.subplots_adjust(left=.078,right=.985,bottom=.255,top=.88,wspace=.11,hspace=.36)
    image=output/'abc-postprocessing.png';fig.savefig(image,dpi=180,facecolor='white');plt.close(fig)
    receipt={'schema':'posteriorforge.prior_tail_ABC_figure.v1','primary_report':str(report.resolve()),'primary_report_sha256':REPORT_SHA,
             'source_sha256':sha(__file__),'image':image.name,'image_sha256':sha(image),'plotted_rows':rows,
             'scope':'render saved N1024 own-stage U values/statuses only; no numerical diagnosis or new thresholds',
             'N1_shown':False,'new_model_codec_sampling_world_reference_RNG_GPU_G3_calls':0,
             'matplotlib':matplotlib.__version__}
    (output/'figure.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'image':str(image),'image_sha256':sha(image),'receipt_sha256':sha(output/'figure.json'),'source_sha256':sha(__file__)}))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',required=True);p.add_argument('--output',required=True);render(**vars(p.parse_args()))
