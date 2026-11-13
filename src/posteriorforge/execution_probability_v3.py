"""Policy C probability arithmetic: unchanged adjusted FP32 logits, CPU FP64 P.

Built with PriorLabs-TabPFN. No model, RNG, physical parameter clipping, epsilon
mass floor, tail truncation or redraw occurs here. This is a new numerical
policy, including an explicit u=0 extension; legacy bit identity is not claimed.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import json
from pathlib import Path
from threading import Lock,get_ident
from types import SimpleNamespace
from importlib import import_module
from . import execution_mixture as legacy

LEGACY_SHA='e71f43b73ae8259c20b5d7b52feccc7c851e9a2fbb86d869bc0a5f7f3af792a1'
SCHEMA='posteriorforge.execution.probability64.v3'
POLICY='adjusted_F32_logits_CPU_F64_source_CDF_v3'
head_acceptance=legacy.head_acceptance
_HEAD_ACCEPTANCE=head_acceptance
_HEAD_CODE=head_acceptance.__code__
_HEAD_GLOBALS=head_acceptance.__globals__
HEAD_MASS_ENVELOPE=legacy.HEAD_MASS_ENVELOPE
_OWNER=ContextVar('posteriorforge_probability_v3_owner',default=None)
_LOCK=Lock()


def require(value,message):
    if not value:raise RuntimeError(message)

def source_sha256():return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

def require_sources():
    require(source_sha256()==_IMPORTED_SHA,'v3 probability source changed after import')
    require(legacy.execution_source_sha256()==LEGACY_SHA,'frozen e71 source changed')
    require(head_acceptance is _HEAD_ACCEPTANCE and head_acceptance.__code__ is _HEAD_CODE and head_acceptance.__globals__ is _HEAD_GLOBALS,'original head acceptance changed')
    require(legacy.HEAD_MASS_ENVELOPE is HEAD_MASS_ENVELOPE and _HEAD_GLOBALS.get('HEAD_MASS_ENVELOPE') is HEAD_MASS_ENVELOPE,'original head acceptance envelope changed')
    for name,value in _CONFIG_SEALS.items():
        require(globals().get(name) is value,'v3 probability binding changed: '+name)
    for name,value in _GLOBAL_BINDINGS.items():
        require(globals().get(name) is value,'v3 probability referenced global changed: '+name)
    for name,function,code,wrapped,wcode in _FUNCTIONS:
        require(globals().get(name) is function and function.__code__ is code,'v3 probability callable changed: '+name)
        require(getattr(function,'__wrapped__',None) is wrapped and (wrapped is None or wrapped.__code__ is wcode),'v3 probability wrapped body changed: '+name)
    return True


def metadata():
    require_sources()
    return {'schema':SCHEMA,'policy':POLICY,'module_sha256':source_sha256(),
            'neural_input':'actual translator-entry FP32 logits, already temperature/cancel adjusted; no second temperature',
            'device':'cpu','softmax_CDF_translation_probability_dtype':'torch.float64',
            'source_CDF_order':'softmax64; cumsum(p)-p; corrected share; source boundary/share/CDF clamps; diff; negative-source-diff clamp_min0',
            'head_acceptance':'EXACT e71 function; raw translated F64 mass BEFORE normalization',
            'inverse':'softmax64(log RAW weights); cumsum(apiP)/terminal; invert represented positive CDF intervals',
            'zero_uniform':'u=0 selects first represented positive interval left border; NEW finite-support extension, not legacy NaN',
            'positive_uniform_ties':'left search; zero-width represented CDF intervals never selected',
            'uniform_domain':'original CPU torch.float32 [0,1); promoted unchanged to F64; u=1 rejected',
            'CDF_resolution':'positive raw weights can naturally collapse in F64 CDF differences; counts/masses reported, no threshold zeroing',
            'borders':'promote original fitted FP32 bits; do not recompute target scaling in F64',
            'physical_parameter_clipping':False,'RNG_calls':0,'legacy_bit_identity_claimed':False}


@contextmanager
def production_scope():
    require_sources();require(_LOCK.acquire(blocking=False),'nested/concurrent v3 probability scope unsupported')
    descriptor={'schema':SCHEMA+'.scope','policy':POLICY,'module_sha256':source_sha256(),
                'owner_thread':get_ident(),'translation_call_count':0,'inverse_call_count':0,'stage0_assembly_count':0,
                'active':True,'closed':False,'SDK_global_aliases_modified':False,'legacy_preparation_guards':[],
                'source_unchanged':False,'actual_probability_dtype':'torch.float64'}
    token=_OWNER.set(descriptor)
    try:yield descriptor
    finally:
        try:
            descriptor['source_unchanged']=source_sha256()==descriptor['module_sha256']
            descriptor['active']=False;descriptor['closed']=True
        finally:_OWNER.reset(token);_LOCK.release()


def active_descriptor(descriptor=None):
    value=_OWNER.get() if descriptor is None else descriptor
    if value is not None:
        require(value is _OWNER.get() and value['active'] and value['owner_thread']==get_ident(),'wrong/closed v3 probability owner')
    return value


def _torch():return import_module('torch')

def _tensor(value,dtype,label,ndim):
    t=_torch();require(isinstance(value,t.Tensor) and value.dtype==dtype and value.device.type=='cpu' and value.ndim==ndim,label+' exact CPU dtype/shape required')
    return value


def translate_logits(adjusted_logits_F32,from_borders_F32,to_borders_F32,*,descriptor=None):
    require_sources();t=_torch();d=active_descriptor(descriptor)
    x=_tensor(adjusted_logits_F32,t.float32,'adjusted logits',2)
    frm=_tensor(from_borders_F32,t.float32,'source borders',1);to=_tensor(to_borders_F32,t.float32,'canonical borders',1)
    require(x.shape[0]>0 and x.shape[-1]+1==len(frm) and len(to)>=2,'translation dimensions differ')
    require(bool((t.isfinite(x)|t.isneginf(x)).all()) and bool(t.isfinite(x).any(-1).all()),'invalid adjusted F32 logits')
    require(bool(t.isfinite(frm).all()) and bool(t.isfinite(to).all()) and bool((frm[1:]>frm[:-1]).all()) and bool((to[1:]>to[:-1]).all()),'strict finite original F32 border grids required; no repair')
    f=frm.to(t.float64);y=to.to(t.float64);logits=x.to(t.float64)
    p=t.softmax(logits,-1);before=t.cumsum(p,-1)-p
    index=t.searchsorted(f,y.contiguous())-1
    index=t.where(y==f[0],t.zeros_like(index),index)
    index=t.where(y==f[-1],t.full_like(index,len(f)-2),index).clamp(0,len(f)-2)
    share_unclamped=(y-f[index])/(f[1:]-f[:-1])[index]
    share=share_unclamped.clamp(0.,1.)
    # One border geometry is reused across rows; probability arithmetic remains row-wise.
    index=index[None,:].expand(len(x),-1);share=share[None,:]
    cdf_unclamped=before.gather(-1,index)+p.gather(-1,index)*share
    cdf_unclamped=t.where(y<=f[0],t.zeros_like(cdf_unclamped),cdf_unclamped)
    cdf_unclamped=t.where(y>=f[-1],t.ones_like(cdf_unclamped),cdf_unclamped)
    cdf=cdf_unclamped.clamp(0.,1.)
    forced_change=float(cdf[:,0].abs().sum()+(cdf[:,-1]-1.).abs().sum())
    cdf=cdf.clone();cdf[:,0]=0.;cdf[:,-1]=1.
    signed=cdf[:,1:]-cdf[:,:-1];raw=signed.clamp_min(0.)
    mass=raw.sum(-1,dtype=t.float64)
    require(bool(t.isfinite(raw).all()) and bool((mass>0).all()) and bool((mass<=legacy.HEAD_MASS_ENVELOPE).all()),'raw F64 canonical mass outside original envelope')
    if d is not None:d['translation_call_count']+=1
    return {'raw_weights64':raw,'raw_mass64':mass,'pretranslation_probabilities64':p,
            'translation_cdf64':cdf,'translation_signed_differences64':signed,
            'translation_diagnostics':{'negative_difference_count':int((signed<0).sum()),
             'negative_difference_absolute_mass':float((-signed.clamp_max(0.)).sum()),
             'source_CDF_clip_count':int(((cdf_unclamped<0)|(cdf_unclamped>1)).sum()),
             'source_CDF_clip_absolute_change':float((cdf_unclamped-cdf_unclamped.clamp(0.,1.)).abs().sum()),
             'forced_destination_endpoint_absolute_change':forced_change,
             'destination_endpoints':'source-compatible first CDF=0, last CDF=1; NOT physical parameter clipping',
             'parameter_clipping':False,'epsilon_floor_or_tail_truncation':False},
            'adjusted_logits_F32':x,'from_borders_F32':frm,'to_borders_F32':to}


def normalize_weights(raw_weights64):
    require_sources();t=_torch();w=_tensor(raw_weights64,t.float64,'weights',2)
    require(w.shape[0]>0 and w.shape[1]>0 and bool(t.isfinite(w).all()) and bool((w>=0).all()),'finite nonnegative F64 weights required')
    mass=w.sum(-1,dtype=t.float64);require(bool((mass>0).all()),'positive row mass required')
    ordinary=w/mass[:,None]
    cumulative=t.cumsum(w,-1);terminal=cumulative[:,-1]
    cdf=cumulative/terminal[:,None]
    edges=t.cat((t.zeros((len(w),1),dtype=t.float64),cdf),-1);intervals=edges[:,1:]-edges[:,:-1]
    require(bool((intervals>=0).all()) and bool((cdf[:,-1]==1).all()),'represented F64 CDF invalid')
    collapsed=(w>0)&(intervals==0)
    return {'normalized_weights64':ordinary,'sampling_cdf64':cdf,'sampling_interval_weights64':intervals,
            'input_mass64':mass,'cdf_terminal_mass64':terminal,
            'representation_diagnostics':{'collapsed_positive_weight_count':int(collapsed.sum()),
             'collapsed_positive_raw_mass64':float(w[collapsed].sum()),
             'collapsed_positive_normalized_mass64':float(ordinary[collapsed].sum()),
             'normalized_raw_vs_interval_max_abs':float((ordinary-intervals).abs().max()),
             'normalized_raw_vs_interval_max_row_TV':float(.5*(ordinary-intervals).abs().sum(-1).max()),
             'sum_mass_and_cumulative_mass_are_distinct':True,'threshold_zeroing':False}}


def distribution_from_logits(logits_F64):
    require_sources();t=_torch();logits=_tensor(logits_F64,t.float64,'raw-weight log logits',2)
    require(bool((t.isfinite(logits)|t.isneginf(logits)).all()) and bool(t.isfinite(logits).any(-1).all()),'invalid F64 logits')
    api=t.softmax(logits,-1)
    return {'api_probabilities64':api,**normalize_weights(api)}


def inverse_details(logits_F64,borders_F64,uniforms_F32):
    require_sources();t=_torch();d=active_descriptor()
    b=_tensor(borders_F64,t.float64,'fitted promoted borders',1);u=_tensor(uniforms_F32,t.float32,'original uniforms',1)
    require(len(u)>0 and len(logits_F64) in (1,len(u)) and logits_F64.shape[-1]+1==len(b),'inverse dimensions differ')
    require(bool(t.isfinite(b).all()) and bool((b[1:]>b[:-1]).all()),'finite increasing fitted borders required')
    require(bool(t.isfinite(u).all()) and bool((u>=0).all()) and bool((u<1).all()),'original FP32 U must be in [0,1); u1 rejected')
    distribution=distribution_from_logits(logits_F64);cdf=distribution['sampling_cdf64'];masses=distribution['sampling_interval_weights64'];u64=u.to(t.float64)
    if len(cdf)==1:
        index=t.searchsorted(cdf[0],u64,right=False)
        first=t.argmax((masses[0]>0).to(t.int64));index=t.where(u64==0,first.expand_as(index),index)
        before=t.cat((t.zeros(1,dtype=t.float64),cdf[0][:-1]))[index];selected=masses[0,index]
    else:
        index=t.searchsorted(cdf,u64[:,None].contiguous(),right=False).squeeze(-1)
        first=t.argmax((masses>0).to(t.int64),dim=-1);index=t.where(u64==0,first,index)
        before=t.cat((t.zeros((len(cdf),1),dtype=t.float64),cdf[:,:-1]),-1).gather(-1,index[:,None]).squeeze(-1)
        selected=masses.gather(-1,index[:,None]).squeeze(-1)
    require(bool((index<len(b)-1).all()) and bool((selected>0).all()),'invalid represented interval selection; no repair')
    fraction=(u64-before)/selected
    require(bool((fraction>=0).all()) and bool((fraction<=1).all()),'invalid represented fraction; no clipping/extrapolation')
    left=b[index];right=b[index+1];value=left+(right-left)*fraction
    require(bool(t.isfinite(value).all()) and bool((value>=left).all()) and bool((value<=right).all()),'inverse outside selected finite bar; no repair')
    if d is not None:d['inverse_call_count']+=1
    return {'values64':value,'index':index,'fraction64':fraction,'cdf_before64':before,
            'selected_interval_mass64':selected,**distribution}


def finite_bar_icdf(logits_F64,borders_F64,uniforms_F32):
    return inverse_details(logits_F64,borders_F64,uniforms_F32)['values64']


def from_adjusted_logits(adjusted_logits_F32,from_borders_F32,to_borders_F32,api_borders_F32,*,descriptor=None):
    t=_torch();b=_tensor(api_borders_F32,t.float32,'fitted API borders',1)
    translated=translate_logits(adjusted_logits_F32,from_borders_F32,to_borders_F32,descriptor=descriptor)
    raw=translated['raw_weights64'];require(len(b)==raw.shape[-1]+1,'API border dimensions differ')
    logits=raw.log();distribution=distribution_from_logits(logits)
    direct=normalize_weights(raw)
    return {'logits':logits,'criterion':SimpleNamespace(borders=b.to(t.float64)),
            'raw_weights64':raw,'component_raw_mass':translated['raw_mass64'],
            'head_mass_envelope':legacy.HEAD_MASS_ENVELOPE,'probabilities64':distribution['api_probabilities64'],
            'normalized_raw_probabilities64':direct['normalized_weights64'],
            'sampling_cdf64':distribution['sampling_cdf64'],'sampling_interval_weights64':distribution['sampling_interval_weights64'],
            'probability_capture':{'translation':translated,'distribution':distribution},
            'probability_v3':{'policy':POLICY,'module_sha256':source_sha256(),
             'translation_diagnostics':translated['translation_diagnostics'],
             'representation_diagnostics':distribution['representation_diagnostics'],
             'raw_normalized_vs_API_normalized_max_row_TV':float(.5*(direct['normalized_weights64']-distribution['normalized_weights64']).abs().sum(-1).max()),
             'fitted_border_F32_bits_promoted_unchanged':True,'temperature_reapplied':False}}


def assemble_shared_raw64(predictions,*,descriptor=None,stage0=False):
    require_sources();t=_torch();d=active_descriptor(descriptor)
    require(len(predictions)==8,'all8 actual components required')
    borders=predictions[0]['criterion'].borders
    require(all(t.equal(p['criterion'].borders,borders) for p in predictions),'all8 API borders must match')
    raw=t.stack([p['raw_weights64'] for p in predictions]).mean(0)
    logits=raw.log();distribution=distribution_from_logits(logits);direct=normalize_weights(raw)
    require(type(stage0) is bool,'stage0 counter scope must be explicit bool')
    if d is not None and stage0:d['stage0_assembly_count']+=1
    return {'logits':logits,'criterion':SimpleNamespace(borders=borders),'raw_weights64':raw,
            'component_raw_mass':raw.sum(-1,dtype=t.float64),'head_mass_envelope':legacy.HEAD_MASS_ENVELOPE,
            'probabilities64':distribution['api_probabilities64'],'normalized_raw_probabilities64':direct['normalized_weights64'],
            'sampling_cdf64':distribution['sampling_cdf64'],
            'sampling_interval_weights64':distribution['sampling_interval_weights64'],
            'component_predictions':predictions,'members':[p['member'] for p in predictions if 'member' in p],
            'probability_capture':{'distribution':distribution},
            'probability_v3':{'module_sha256':source_sha256(),'raw_mean_before_one_normalization':True,
             'assembly_scope':'production_shared_stage0' if stage0 else 'all8_diagnostic',
             'representation_diagnostics':distribution['representation_diagnostics']}}


_FUNCTIONS=[(name,value,value.__code__,getattr(value,'__wrapped__',None),getattr(getattr(value,'__wrapped__',None),'__code__',None))
            for name,value in list(globals().items())
            if callable(value) and getattr(value,'__module__',None)==__name__ and hasattr(value,'__code__')]
_CONFIG_SEALS={name:globals()[name] for name in ['LEGACY_SHA','SCHEMA','POLICY','legacy','head_acceptance','_HEAD_ACCEPTANCE','_HEAD_CODE','_HEAD_GLOBALS','HEAD_MASS_ENVELOPE','_OWNER','_LOCK']}
_GLOBAL_BINDINGS={name:globals()[name] for entry in _FUNCTIONS for function in (entry[1],entry[3]) if function is not None
    for name in function.__code__.co_names if name in globals() and name not in ('_IMPORTED_SHA','_FUNCTIONS','_CONFIG_SEALS','_GLOBAL_BINDINGS')}
_IMPORTED_SHA=source_sha256()
