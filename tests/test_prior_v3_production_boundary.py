"""Tiny archived-stage engineering fixtures; never a recovered physical cohort.

The FIFO/accept-reject/support code is original. Only an isolated test-instance
policy uses count4/lookahead2; no production cardinality/global/source changes.
"""
from pathlib import Path
from types import FunctionType
import hashlib
import pickle
import random
import numpy as np
import pytest
from posteriorforge import execution_physical_fifo_v1 as fifo
from posteriorforge import execution_proposal_buffer as original_fifo
from posteriorforge import prior_proposals_v3 as proposals
from posteriorforge import prior_coordinates_v1 as training
from posteriorforge.inference import support_mask

PHASE=Path('/scratch/project/eu-26-17/eubrieucb/posteriorforge/results/accuracy-prior-coordinates-v1')
STAGES_SHA='6989b4930fcb8dda2a144811c7988530dec120b8e6c1bfd62b269f47261a14af'
FAILED_SHA='92755b788fc894d4ca16157e573bd1b5c9d8a4468b71cf184fa16303bced7bce'
SCOPE={'case':0,'query':'TEST_ONLY_SIX_ARCHIVED_ROWS_NOT_A_POSTERIOR',
       'context_seed':20261111,'engineering_expected_count':4,'engineering_lookahead':2}

def state():
    return pickle.dumps((np.random.get_state(),random.getstate()),protocol=5)

def archived_six():
    folder=PHASE/'native-pilot-v2-seed20261111-case0-01'
    path=folder/'latent-stage-returns.npz'
    if not path.exists():pytest.skip('Karolina archived-stage engineering fixture unavailable')
    assert hashlib.sha256(path.read_bytes()).hexdigest()==STAGES_SHA
    assert hashlib.sha256((folder/'manifest.json').read_bytes()).hexdigest()==FAILED_SHA
    indices=np.array([0,4355,41765,1,2,3],np.int64)
    with np.load(path,allow_pickle=False) as a:
        assert a['filled_stage_mask'].all()
        rows=np.column_stack([a[f'stage_{j:04d}'][indices] for j in range(21)])
        assert rows.tobytes()==a['ordered_latent_prefix'][indices].copy().tobytes()
    assert rows.dtype==np.float32 and rows.shape==(6,21)
    assert rows[1,15]==np.float32(13.876729011535645)
    assert rows[2,19]==np.float32(205.2462615966797)
    # Explicit engineering perturbation, not a neural draw or edited saved array.
    rows[3,15]=np.float32(-13.876729011535645)
    return rows

def engineering_buffer(provider,concatenate):
    """Original method bodies; isolated constructor policy for a six-row test."""
    original=fifo.ProposalBuffer.__init__
    namespace=dict(original.__globals__)
    def small_policy(world_size,rank):
        assert world_size==1 and rank==0
        p=original_fifo.worker_policy(world_size,rank)
        return {**p,'benchmark_total_count':4,'worker_expected_return_count':4,
                'global_initial_lookahead':2,'worker_initial_lookahead':2}
    namespace['worker_policy']=small_policy
    init=FunctionType(original.__code__,namespace,original.__name__,original.__defaults__,original.__closure__)
    init.__kwdefaults__=dict(original.__kwdefaults__ or {})
    attrs={name:getattr(fifo.ProposalBuffer,name) for name in fifo._NAMES}
    attrs.update(__init__=init,buffer_rows=fifo.ProposalBuffer.buffer_rows,
                 _validate_vectors=staticmethod(fifo._validate_vectors),metadata=fifo._metadata)
    cls=type('ExplicitSixRowEngineeringFIFO',(),attrs)
    for name in fifo._NAMES:
        assert getattr(cls,name).__code__ is getattr(fifo.ProposalBuffer,name).__code__
    return cls(provider,scope=SCOPE,concatenate=concatenate)

def test_archived_stages_new_decoder_numpy_FIFO_and_actual_gate_support():
    before=state();old_decode=training._noise_decode;old_validator=training._strict_noise
    latent=archived_six();source=latent.tobytes()
    physical=proposals.decode_ordered_proposals(latent)
    assert latent.tobytes()==source and physical.dtype==np.float64
    assert physical.shape==(6,21) and np.isfinite(physical).all()
    assert physical[1,16]==.5 and physical[2,20]==.5
    assert physical[3,16]<.01  # unchanged finite arithmetic, never clipped.
    assert support_mask(physical).tolist()==[True,True,True,False,True,True]
    calls=[]
    def provider(n):
        calls.append(n);assert n==6
        return physical.copy(),None
    b=engineering_buffer(provider,np.concatenate)
    first,_=b(4);mask=support_mask(first)
    second,_=b(1);assert support_mask(second).all()
    selected=np.concatenate([first[mask],second])
    assert selected.tobytes()==physical[[0,1,2,4]].copy().tobytes()
    r=b.finalize(original_checked_count=5,accepted_before_trim=4,
                 returned_count=4,original_support_calls=2)
    assert r['generated_count']==6 and r['original_api_delivered_count']==5
    assert r['remaining_unused_buffer_count']==1 and r['finalized']
    assert not r['support_filtering_performed_by_buffer'] and not r['row_repair_or_reordering']
    assert calls==[6] and state()==before
    assert training._noise_decode is old_decode and training._strict_noise is old_validator
    with pytest.raises(training.CoordinateError):training.decode_ordered(latent[:3])

def test_archived_stages_real_original_Torch_accept_reject_and_finalization():
    torch=pytest.importorskip('torch')
    from posteriorforge.inference import _prior
    actual_sampler=fifo._original_sampler()
    latent=archived_six();physical=proposals.decode_ordered_proposals(latent)
    before=state();torch_before=torch.get_rng_state().numpy().tobytes();calls=[]
    def provider(n):
        calls.append(n)
        if len(calls)==1:
            assert n==6
            value=physical.copy()
        else:
            assert n==2
            value=physical[[0,4]].copy()  # deterministic engineering continuation, not new draws.
        return torch.from_numpy(value),None
    b=engineering_buffer(provider,lambda parts:torch.cat(parts,dim=0))
    prior=_prior(torch)
    samples,_,_=actual_sampler(proposal=b,accept_reject_fn=prior.support.check,
        num_samples=4,max_sampling_batch_size=4,max_iter_rejection=None,
        show_progress_bars=False,proposal_sampling_kwargs={})
    r=b.finalize(original_checked_count=prior.support.proposed,
        accepted_before_trim=prior.support.accepted,returned_count=len(samples),
        original_support_calls=prior.support.calls)
    assert samples.dtype==torch.float64 and samples.numpy().tobytes()==physical[[0,1,2,4]].copy().tobytes()
    assert support_mask(samples.numpy()).all() and calls==[6,2]
    assert r['generated_count']==r['original_api_delivered_count']+r['remaining_unused_buffer_count']==8
    assert r['original_sampler_counts']['returned_count']==4
    assert r['finalized'] and not r['support_filtering_performed_by_buffer']
    assert state()==before and torch.get_rng_state().numpy().tobytes()==torch_before
    assert not torch.cuda.is_initialized()


def test_actual_producer_sampler_uses_upstream_C_raw_head_mass_then_F32_stage_storage():
    """Actual backend probability body + sampler, with saved logits replacing NN only."""
    torch=pytest.importorskip('torch')
    import copy,inspect,json,sys
    from contextlib import contextmanager
    from types import SimpleNamespace
    from posteriorforge import execution_prior_native_v3 as backend
    from posteriorforge import execution_probability_v3 as probability
    from posteriorforge import performance_prior_pair_v3 as producer
    path=PHASE/'tail-conditionals-offline-01/focal-densities/stage19-candidate-member2-N1024-focal.npz'
    if not path.exists():pytest.skip('saved PolicyA/C discriminator unavailable')
    report_path=PHASE/'tail-postprocessing-offline-01/report.json'
    assert hashlib.sha256(report_path.read_bytes()).hexdigest()=='cbc8e44e62780d48e594845bd63a9c72faa76eaf7a1459d9e4dbb2210e989a5a'
    report=json.loads(report_path.read_bytes())
    row=next(r for r in report['records'] if(r['stage'],r['arm'],r['member'],r['label'])==(19,'candidate',2,'N1024'))
    point=lambda name:next(p for p in row['variants'][name]['points'] if p['U_bits']=='f6ff7f3f')
    A=point('A_captured_final_CPU_F32_weights')['value']
    C=point('C_softmax64_adjusted_F32logits_F64log_CPUsoftmax64')['value']
    assert A>49 and 8<C<8.3
    with np.load(path,allow_pickle=False) as a:arrays={k:a[k].copy() for k in a.files}
    assert hashlib.sha256(path.read_bytes()).hexdigest()==row['source_focal_sha256']
    snapshots={m:dict(vars(m)) for m in (backend,probability,producer.v2,training)}
    @contextmanager
    def no_neural_scope(*args,**kwargs):yield
    fake_session=SimpleNamespace(neural=SimpleNamespace(forward_scope=no_neural_scope),probability_descriptor=None)
    class ExplicitSavedNNModel:
        # Engineering eval-state fake only: no torch.nn.Module construction,
        # parameter initialization, model forward or RNG call.
        training=False
        def eval(self):
            self.training=False
            return self
        def modules(self):return (self,)
    models=[ExplicitSavedNNModel().eval() for _ in range(8)]
    configs=[SimpleNamespace(engineering_saved_head=i) for i in range(8)]
    executor=SimpleNamespace(configs=configs,models=models,
        preprocessors=[object() for _ in range(8)],cat_ixs=[[] for _ in range(8)],n_train_samples=[4096]*8)
    regressor=SimpleNamespace(use_autocast_=False,forced_inference_dtype_=torch.float32,
        softmax_temperature=.9,average_before_softmax=False,executor_=executor,
        y_train_mean_=-.0055904993787407875,y_train_std_=.9938715696334839)
    base=SimpleNamespace(metadata={'model_rng_isolation_count':0,
        'fixture_only_saved_adjusted_logits':True,'actual_neural_forward_calls':0})
    @contextmanager
    def explicit_observed_rng_scope():
        # Match actual _component bookkeeping without executing a model, seed
        # reset or random draw; the full default states are checked below.
        base.metadata['model_rng_isolation_count']+=1
        yield
    base._model_rng=explicit_observed_rng_scope
    counters={'prefix_query_count':0,'prefix_query_rows':0,'cached_member_lists_restored':True}
    def saved_prefix(reg,q,**kwargs):
        n=len(q)
        return ([torch.from_numpy(np.repeat(arrays['raw_model_logits'][None,:],n,axis=0))],
            [arrays['from_borders'].copy()],torch.from_numpy(arrays['to_borders'].copy()),
            torch.from_numpy(arrays['borders'].copy()))
    def fixture_guard():
        probability.require_sources()
        assert backend._component.__code__ is component_code
        return True
    component_code=backend._component.__code__
    private=dict(backend._component.__globals__)
    private.update(_prefix=lambda:saved_prefix,_PREFIX_DESCRIPTOR={'fixture_only_saved_adjusted_logits':True,'NN_calls':0},
                   _session=lambda:fake_session,require_sources=fixture_guard)
    actual_component=FunctionType(component_code,private,'actual_v3_backend_math_saved_NN_cutpoint',backend._component.__defaults__)
    actual_component.__kwdefaults__=backend._component.__kwdefaults__
    predictions=[]
    class SavedHead:
        def predict_component(self,member,query,**kwargs):
            assert member==2
            p=actual_component(regressor,query,19,member,fake_session,base,counters,original=False,capture_member=True)
            assert p['logits'].dtype==p['raw_weights64'].dtype==p['component_raw_mass'].dtype==torch.float64
            assert p['neural_v3']['actual_adjusted_dtype']=='torch.float32'
            assert p['member']['configuration_identity']==backend.cache._configuration_identity(configs[2])
            assert p['neural_v3']['selected_root_training'] is False
            assert p['neural_v3']['selected_submodules_training_count']==0
            assert p['neural_v3']['model_rng_isolation_scope_delta']==1
            assert torch.equal(p['component_raw_mass'],p['raw_weights64'].sum(-1,dtype=torch.float64))
            predictions.append(p);return p
    supplied=[]
    class FixedProvidedTorch:
        float32=torch.float32;float64=torch.float64;from_numpy=staticmethod(torch.from_numpy)
        @staticmethod
        def rand(n,*,generator,dtype,device):
            supplied.append(('rand',generator,n,dtype,device))
            assert device=='cpu'
            return torch.full((n,),.9999994039535522,dtype=dtype) if generator=='component_U' else torch.zeros(n,dtype=dtype)
        @staticmethod
        def randint(low,high,size,*,generator,device):
            supplied.append(('randint',generator,size,low,high,device))
            assert generator=='head' and(low,high)==(0,8)
            return torch.full(size,2,dtype=torch.int64)
    calls=[];old_profiler=sys.getprofile()
    mass_parameter=next(iter(inspect.signature(probability.head_acceptance).parameters))
    def profile(frame,event,arg):
        if event=='call' and frame.f_code is probability.head_acceptance.__code__:
            calls.append(frame.f_locals[mass_parameter].copy())
    before=state();torch_before=torch.get_rng_state().numpy().tobytes()
    global_counters={'stage0_shared_predictions':0,'component_predictions':0,'head_candidates':0,'head_rejections':0}
    sampler=producer.physical_sampling_namespace().component_stage_sampler(SavedHead(),FixedProvidedTorch,
        {'component_uniform':'component_U','head_proposal':'head','head_acceptance':'accept'},global_counters)
    try:
        sys.setprofile(profile)
        draws,extra=sampler(None,19,lambda rows:np.zeros((len(rows),34),np.float32),6,1024)
    finally:sys.setprofile(old_profiler)
    assert draws.dtype==np.float32 and np.all(draws==np.float32(C))
    assert not np.any(draws==np.float32(A)) and len(predictions)==len(calls)==1
    assert calls[0].tobytes()==predictions[0]['component_raw_mass'].numpy().tobytes()
    assert global_counters['head_candidates']==6 and global_counters['head_rejections']==0
    assert supplied[0]==('rand','component_U',6,torch.float32,'cpu')
    engineering=archived_six();engineering[:,19]=draws  # unchanged ORIGINAL Float32 stage storage.
    physical=producer.decode_complete_wave(engineering)
    assert physical.dtype==np.float64 and np.isfinite(physical).all()
    b=engineering_buffer(lambda n:(physical.copy(),None),np.concatenate)
    first,_=b(4);mask=support_mask(first);rest,_=b(1)
    selected=np.concatenate([first[mask],rest])
    r=b.finalize(original_checked_count=5,accepted_before_trim=4,returned_count=4,original_support_calls=2)
    assert selected.tobytes()==physical[[0,1,2,4]].copy().tobytes() and r['finalized']
    assert counters['prefix_query_count']==1 and counters['prefix_query_rows']==6
    assert counters['cached_member_lists_restored'] and executor.configs is configs
    assert executor.models is models and base.metadata['model_rng_isolation_count']==1
    assert base.metadata['actual_neural_forward_calls']==0
    assert state()==before and torch.get_rng_state().numpy().tobytes()==torch_before
    assert not torch.cuda.is_initialized()
    for module,snapshot in snapshots.items():
        assert all(getattr(module,k) is v for k,v in snapshot.items() if k not in ('_PREFIX', '_PREFIX_DESCRIPTOR','_PREFIX_ORIGINAL','_PREFIX_SEALS'))
