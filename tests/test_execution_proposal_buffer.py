"""CPU engineering tests only; no checkpoint or reduced posterior benchmark."""
import hashlib
import importlib.util
from pathlib import Path
from threading import Thread

import numpy as np
import pytest
from posteriorforge import execution_proposal_buffer as fifo


SCOPE={"case":0,"query":"raw-observation-sha256","context_seed":20261111,"repetition":0}


class OrderedStream:
    """An infinite deterministic complete-vector stream, independent of chunks."""
    def __init__(self,offset=0):self.cursor=offset;self.calls=[];self.last_values=None
    def __call__(self,count):
        self.calls.append(count)
        ids=np.arange(self.cursor,self.cursor+count,dtype=np.float32)
        self.cursor+=count
        self.last_values=ids[:,None]+np.arange(21,dtype=np.float32)[None,:]
        return self.last_values,None


def buffer(provider=None,**kwargs):
    return fifo.ProposalBuffer(provider or OrderedStream(),scope=SCOPE,
        concatenate=lambda parts:np.concatenate(parts,axis=0),**kwargs)


def test_worker_partitions_keep_50000_outputs_and_global_1200_lookahead():
    for world in (1,2,4):
        policies=[fifo.worker_policy(world,rank) for rank in range(world)]
        assert sum(p["worker_expected_return_count"] for p in policies)==50000
        assert sum(p["worker_initial_lookahead"] for p in policies)==1200
        assert [p["worker_expected_return_count"]+p["worker_initial_lookahead"] for p in policies]==[51200//world]*world


@pytest.mark.parametrize("world,rank",[(0,0),(3,0),(True,0),(1,-1),(2,2),(2,False)])
def test_invalid_worker_contract_rejected(world,rank):
    with pytest.raises(ValueError):fifo.worker_policy(world,rank)


def test_fifo_delivers_exact_requests_and_only_later_deficit_generates():
    stream=OrderedStream();proposal=buffer(stream)
    a,_=proposal(10);b,_=proposal(7);c,_=proposal(1200)
    expected=np.arange(1217,dtype=np.float32)[:,None]+np.arange(21,dtype=np.float32)[None,:]
    np.testing.assert_array_equal(np.concatenate([a,b,c]),expected)
    assert stream.calls==[1210,7]
    meta=proposal.metadata()
    assert meta["api_requested_sizes"]==[10,7,1200]
    assert meta["actual_generated_wave_sizes"]==[1210,7]
    assert meta["generated_count"]==meta["original_api_delivered_count"]==1217
    assert meta["generated_wave_count"]==2 and meta["required_stage_preparations_for_generated_waves"]==42
    assert meta["remaining_unused_buffer_count"]==0


def test_retained_tail_is_copied_instead_of_retaining_whole_wave_storage():
    stream=OrderedStream();proposal=buffer(stream)
    proposal(50000)
    assert proposal.buffer_rows==1200
    assert not np.shares_memory(proposal._buffer,stream.last_values)
    old_tail=proposal._buffer
    proposal(7)
    assert proposal.buffer_rows==1193
    assert not np.shares_memory(proposal._buffer,old_tail)


def test_nan_inf_unsupported_rows_and_whole_vector_order_are_delivered_unchanged():
    values=np.arange(1210*21,dtype=np.float32).reshape(1210,21)
    values[1,5]=np.nan;values[2,7]=np.inf;values[3,9]=-np.inf;values[4,0]=-123456.
    proposal=buffer(lambda count:(values.copy(),None))
    a,_=proposal(10);b,_=proposal(1200)
    np.testing.assert_array_equal(np.concatenate([a,b]),values)
    assert proposal.metadata()["support_filtering_performed_by_buffer"] is False
    assert proposal.metadata()["row_repair_or_reordering"] is False
    assert proposal.metadata()["partial_vector_filtering"] is False


def test_no_numpy_rng_consumption_and_scope_metadata_is_copied():
    scope={"case":0,"query":{"sha":"abc"}}
    before=np.random.get_state()
    proposal=fifo.ProposalBuffer(OrderedStream(),scope=scope,concatenate=np.concatenate)
    proposal(7);proposal.metadata()
    after=np.random.get_state()
    assert before[0]==after[0] and before[2:]==after[2:]
    np.testing.assert_array_equal(before[1],after[1])
    scope["query"]["sha"]="changed"
    meta=proposal.metadata();assert meta["request_scope"]["query"]["sha"]=="abc"
    meta["request_scope"]["query"]["sha"]="also changed"
    assert proposal.metadata()["request_scope"]["query"]["sha"]=="abc"
    assert meta["rng_draws_by_buffer"]==0


def test_finalize_records_checked_generated_unused_separately_and_reset_changes_owner_scope():
    proposal=buffer();proposal(50000);proposal(450)
    meta=proposal.finalize(original_checked_count=50450,accepted_before_trim=50100,returned_count=50000,original_support_calls=2)
    assert meta["generated_count"]==51200 and meta["original_api_delivered_count"]==50450
    assert meta["remaining_unused_buffer_count"]==750 and meta["live_buffer_rows"]==0
    assert meta["original_sampler_counts"]=={"original_checked_count":50450,"original_support_calls":2,"accepted_before_trim":50100,"returned_count":50000}
    assert meta["unused_rows_discarded_only_after_original_sampler_completion"] is True
    assert meta["original_sampler_source_sha256"]==fifo.ORIGINAL_SAMPLER_SHA256
    assert meta["module_sha256"]==hashlib.sha256(Path(fifo.__file__).read_bytes()).hexdigest()
    with pytest.raises(RuntimeError):proposal(1)
    with pytest.raises(RuntimeError):proposal.finalize(original_checked_count=50450,accepted_before_trim=50100,returned_count=50000,original_support_calls=2)
    proposal.reset(OrderedStream(offset=100000),scope={"case":1,"query":"different"})
    row,_=proposal(1)
    assert row[0,0]==100000 and proposal.metadata()["buffer_reset_count"]==1
    assert proposal.metadata()["request_scope"]["case"]==1
    assert proposal.metadata()["actual_generated_wave_sizes"]==[1201]


def test_low_count_engineering_toy_cannot_be_finalized_as_50000_benchmark():
    proposal=buffer();proposal(10)
    with pytest.raises(ValueError,match="50k"):
        proposal.finalize(original_checked_count=10,accepted_before_trim=10,returned_count=10,original_support_calls=1)
    with pytest.raises(RuntimeError):proposal.reset(OrderedStream(),scope=SCOPE)


@pytest.mark.parametrize("change",["count","width","dtype","logprob","device"])
def test_invalid_provider_output_fails_closed_without_casting_or_retry(change):
    class WrongDevice(np.ndarray):
        @property
        def device(self):return "cuda:0"
    def invalid(count):
        values=np.zeros((count,21),dtype=np.float32);logprob=None
        if change=="count":values=values[:-1]
        if change=="width":values=values[:,:20]
        if change=="dtype":values=values.astype(np.float64)
        if change=="logprob":logprob=np.zeros(count)
        if change=="device":values=values.view(WrongDevice)
        return values,logprob
    proposal=buffer(invalid)
    with pytest.raises(ValueError):proposal(5)
    meta=proposal.metadata();assert meta["failed"] and meta["actual_generated_wave_sizes"]==[]
    assert meta["generated_wave_attempt_sizes"]==[1205]
    with pytest.raises(RuntimeError):proposal(5)
    proposal.reset(OrderedStream(),scope={"case":1,"query":"fresh"})
    assert proposal(5)[0].shape==(5,21)


def test_concatenation_must_preserve_fp32_shape():
    proposal=fifo.ProposalBuffer(OrderedStream(),scope=SCOPE,
        concatenate=lambda parts:np.concatenate(parts).astype(np.float64))
    proposal(10)
    with pytest.raises(ValueError,match="FP32"):proposal(1205)
    assert proposal.metadata()["failed"]


@pytest.mark.parametrize("bad",[0,-1,True,1.5,np.int64(1)])
def test_invalid_original_request_size_rejected(bad):
    with pytest.raises(ValueError):buffer()(bad)


def test_kwargs_log_density_and_incomplete_scope_not_silently_changed():
    with pytest.raises(ValueError):buffer()(1,with_log_prob=True)
    with pytest.raises(ValueError):fifo.ProposalBuffer(OrderedStream(),scope={"case":0},concatenate=np.concatenate)
    with pytest.raises(ValueError):fifo.ProposalBuffer(OrderedStream(),scope={"case":0,"query":np.nan},concatenate=np.concatenate)


def test_cross_thread_and_provider_identity_changes_rejected():
    proposal=buffer();errors=[]
    def use_elsewhere():
        try:proposal(1)
        except RuntimeError as error:errors.append(str(error))
    thread=Thread(target=use_elsewhere);thread.start();thread.join()
    assert len(errors)==1 and "process/thread" in errors[0]
    proposal._provider=OrderedStream()
    with pytest.raises(RuntimeError,match="provider changed"):proposal(1)


def test_reentrant_provider_fails_closed():
    holder={}
    def provider(count):return holder["buffer"](count)
    holder["buffer"]=buffer(provider)
    with pytest.raises(RuntimeError,match="already active"):holder["buffer"](1)
    assert holder["buffer"].metadata()["failed"]


@pytest.fixture
def original_sampler():
    pytest.importorskip("torch")
    path=Path(fifo.__file__).parent/"upstream/npe_pfn/accept_reject_sampler.py"
    assert hashlib.sha256(path.read_bytes()).hexdigest()==fifo.ORIGINAL_SAMPLER_SHA256
    spec=importlib.util.spec_from_file_location("unchanged_original_rejection_sampler_for_fifo_test",path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.accept_reject_sample


def test_original_cap_51200_alone_keeps_first_request_50000(original_sampler):
    import torch
    calls=[]
    def proposal(count):calls.append(count);return torch.zeros((count,21),dtype=torch.float32),None
    result,logprob,rate=original_sampler(proposal,lambda rows:torch.ones(len(rows),dtype=torch.bool),50000,
        max_sampling_batch_size=51200,max_iter_rejection=None,show_progress_bars=False)
    assert calls==[50000] and result.shape==(50000,21) and logprob is None and rate==1.


@pytest.mark.parametrize("world",[1,2,4])
def test_original_sampler_fifo_matches_same_ordered_stream_and_global_50000(original_sampler,world):
    import torch
    returned_total=0;generated_total=0;unused_total=0
    for rank in range(world):
        count=50000//world;offset=rank*100000
        direct_stream=OrderedStream(offset);buffered_stream=OrderedStream(offset)
        direct_requests=[];support_rows=[]
        def direct(n):
            direct_requests.append(n);values,_=direct_stream(n);return torch.from_numpy(values),None
        def provider(n):
            values,_=buffered_stream(n);return torch.from_numpy(values),None
        proposal=fifo.ProposalBuffer(provider,scope={**SCOPE,"rank":rank},
            concatenate=lambda parts:torch.cat(parts,dim=0),world_size=world,rank=rank)
        def support(rows):
            # Original support stage owns this complete-row decision; the buffer never sees it.
            accepted=rows[:,0].to(torch.int64).remainder(137)!=0
            support_rows.append((len(rows),int(accepted.sum())))
            return accepted
        expected,_,expected_rate=original_sampler(direct,support,count,max_sampling_batch_size=50000,
            max_iter_rejection=None,show_progress_bars=False)
        expected_support=list(support_rows);support_rows.clear()
        actual,logprob,actual_rate=original_sampler(proposal,support,count,max_sampling_batch_size=50000,
            max_iter_rejection=None,show_progress_bars=False)
        assert torch.equal(actual,expected) and actual.shape==(count,21) and logprob is None
        assert actual_rate==expected_rate and support_rows==expected_support
        assert proposal.metadata()["api_requested_sizes"]==direct_requests
        assert buffered_stream.calls==[count+1200//world]
        meta=proposal.finalize(original_checked_count=sum(n for n,_ in support_rows),
            accepted_before_trim=sum(n for _,n in support_rows),returned_count=len(actual),original_support_calls=len(support_rows))
        assert meta["generated_count"]==meta["original_sampler_counts"]["original_checked_count"]+meta["remaining_unused_buffer_count"]
        assert meta["required_stage_preparations_for_generated_waves"]==21
        returned_total+=len(actual);generated_total+=meta["generated_count"];unused_total+=meta["remaining_unused_buffer_count"]
    assert returned_total==50000 and generated_total==51200 and unused_total>0


def test_real_torch_cpu_buffer_draws_no_torch_rng_and_retains_nan(original_sampler):
    import torch
    torch_state=torch.random.get_rng_state().clone()
    def provider(n):
        values=torch.arange(n*21,dtype=torch.float32).reshape(n,21);values[0,5]=float("nan")
        return values,None
    proposal=fifo.ProposalBuffer(provider,scope=SCOPE,concatenate=lambda parts:torch.cat(parts))
    rows,_=proposal(1)
    assert torch.isnan(rows[0,5]) and torch.equal(torch.random.get_rng_state(),torch_state)
    assert proposal._buffer.device.type=="cpu" and proposal._buffer.dtype==torch.float32
