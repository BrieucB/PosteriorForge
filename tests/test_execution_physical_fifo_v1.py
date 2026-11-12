"""Real codec→physical64 FIFO boundary and exact original selection; CPU only."""
import hashlib,pickle,random
from pathlib import Path
import numpy as np
import pytest
from posteriorforge import execution_physical_fifo_v1 as f,execution_proposal_buffer as old,prior_coordinates_v1 as codec
from posteriorforge.inference import support_mask

SCOPE={"case":2,"query":"test-fixed-physical-query","context_seed":20261111,"repetition":0}


def rng_state():
    return pickle.dumps((np.random.get_state(),random.getstate()),protocol=5)


def physical(count=51200):
    z=np.zeros((count,21),np.float32)
    z[:,0]=np.linspace(-1,1,count,dtype=np.float32)
    return codec.decode_ordered(z)


def test_all_fifo_algorithm_functions_are_original_codeobjects():
    f.require_sources()
    for name in f._NAMES:
        assert getattr(f.ProposalBuffer,name) is getattr(old.ProposalBuffer,name)
    assert f.ProposalBuffer.buffer_rows is old.ProposalBuffer.buffer_rows
    assert f.ProposalBuffer._validate_vectors is not old.ProposalBuffer._validate_vectors
    assert old.source_sha256()==f.ORIGINAL_FIFO_SHA256


def test_real_codec_float64_boundary_old_rejects_new_delivers_exactly():
    decoded=physical();before=decoded.tobytes()
    assert decoded.dtype==np.float64 and support_mask(decoded).all()
    original=old.ProposalBuffer(lambda n:(decoded,None),scope=SCOPE,concatenate=np.concatenate)
    with pytest.raises(ValueError,match="FP32"):original(50000)
    assert original.metadata()["original_api_delivered_count"]==0
    proposal=f.ProposalBuffer(lambda n:(decoded,None),scope=SCOPE,concatenate=np.concatenate)
    result,logprob=proposal(50000)
    assert logprob is None and result.dtype==np.float64
    assert result.tobytes()==decoded[:50000].tobytes() and decoded.tobytes()==before
    accepted=int(support_mask(result).sum())
    record=proposal.finalize(original_checked_count=50000,accepted_before_trim=accepted,
                             returned_count=50000,original_support_calls=1)
    record.update(enabled=True,source_unchanged=True,policy_sha256=f.proposal_buffer_policy()["policy_sha256"])
    assert f.validate_proposal_buffer_record(record,scope=SCOPE)
    assert record["generated_count"]==51200 and record["original_api_delivered_count"]==50000
    assert record["remaining_unused_buffer_count"]==1200
    assert record["data_dtype"]==f.DATA_DTYPE and not record["implicit_casting_performed"]


def test_f64_fifo_stream_order_later_deficit_and_nonfinites_unchanged():
    cursor=[0];calls=[]
    def provider(n):
        calls.append(n);value=np.arange(cursor[0]*21,(cursor[0]+n)*21,dtype=np.float64).reshape(n,21)
        cursor[0]+=n
        if len(calls)==1:value[1,3]=np.nan;value[2,5]=np.inf
        return value,None
    before=rng_state()
    buffer=f.ProposalBuffer(provider,scope=SCOPE,concatenate=np.concatenate)
    first,_=buffer(10);second,_=buffer(1205)
    assert calls==[1210,5]
    assert np.isnan(first[1,3]) and np.isposinf(first[2,5])
    assert second[0,0]==210 and second[-1,0]==1214*21
    assert rng_state()==before
    assert buffer.metadata()["rng_draws_by_buffer"]==0


@pytest.mark.parametrize("dtype",[np.float32,np.int64,np.dtype("object")])
def test_no_implicit_conversion_to_float64(dtype):
    buffer=f.ProposalBuffer(lambda n:(np.zeros((n,21),dtype=dtype),None),
                            scope=SCOPE,concatenate=np.concatenate)
    with pytest.raises(ValueError,match="FP64"):buffer(1)
    assert buffer.metadata()["failed"]


def test_policy_validator_rejects_old_dtype_metadata():
    assert f.proposal_buffer_policy()["module_sha256"]==f.source_sha256()
    assert f.proposal_buffer_policy()["policy_id"]!=old.POLICY_ID
    with pytest.raises(ValueError,match="Float64"):
        f.validate_proposal_buffer_record({"data_dtype":"CPU FP32; unchanged"},scope=SCOPE)


def test_real_original_selection_real_codec_50000_float64_nonmutation_rng():
    torch=pytest.importorskip("torch")
    original_sampler=f._original_sampler()
    decoded=physical();source_bytes=decoded.tobytes();calls=[]
    def provider(n):
        calls.append(n);assert n==51200
        return torch.from_numpy(decoded),None
    before=rng_state();torch_before=torch.get_rng_state().clone()
    samples,record=f.complete_request(provider,scope=SCOPE,accept_reject_fn=original_sampler)
    assert samples.dtype==torch.float64 and samples.numpy().tobytes()==decoded[:50000].tobytes()
    assert decoded.tobytes()==source_bytes and calls==[51200]
    assert not np.array_equal(samples.numpy(),samples.numpy().astype(np.float32).astype(np.float64))
    assert record["original_sampler_counts"]=={"original_checked_count":50000,"original_support_calls":1,
                                               "accepted_before_trim":50000,"returned_count":50000}
    assert record["generated_count"]==51200 and record["remaining_unused_buffer_count"]==1200
    assert record["original_support_check_code_and_closure_verified"]
    assert rng_state()==before and torch.equal(torch_before,torch.get_rng_state())


def test_original_support_whole_row_rejection_order_and_counts():
    torch=pytest.importorskip("torch")
    f._original_sampler()
    cursor=[0];calls=[]
    base=physical(1)[0]
    def provider(n):
        start=cursor[0];cursor[0]+=n;calls.append(n)
        ids=np.arange(start,start+n,dtype=np.float64)
        values=np.repeat(base.reshape(1,21),n,axis=0);values[:,0]=ids+.123456789123
        values[ids<3000,6]=1.0  # engineering-only unsupported complete physical rows.
        return torch.from_numpy(values),None
    samples,record=f.complete_request(provider,scope=SCOPE)
    expected=np.arange(3000,53000,dtype=np.float64)+.123456789123
    assert samples[:,0].numpy().tobytes()==expected.tobytes()
    assert samples.dtype==torch.float64 and record["original_sampler_counts"]["returned_count"]==50000
    assert record["original_sampler_counts"]["accepted_before_trim"]>50000
    assert record["original_sampler_counts"]["original_support_calls"]==2
    assert calls[0]==51200 and record["generated_count"]==sum(calls)
    assert record["generated_count"]==record["original_api_delivered_count"]+record["remaining_unused_buffer_count"]
    counts=record["original_sampler_counts"]
    assert counts["accepted_before_trim"]-counts["returned_count"]==1787
    assert not record["support_filtering_performed_by_buffer"] and not record["row_repair_or_reordering"]


def test_wrong_support_or_changed_bounds_rejected_before_provider():
    torch=pytest.importorskip("torch")
    from posteriorforge import inference
    f._original_sampler();calls=[]
    def provider(n):calls.append(n);raise AssertionError("must fail before provider")
    prior=inference._prior(torch)
    prior.support.check=lambda value:torch.ones(len(value),dtype=torch.bool)
    with pytest.raises(RuntimeError,match="support check code"):
        f.complete_request(provider,scope=SCOPE,prior=prior)
    prior=inference._prior(torch)
    method=prior.support.check.__func__
    closure=dict(zip(method.__code__.co_freevars,method.__closure__))
    closure["low"].cell_contents=.02
    with pytest.raises(RuntimeError,match="bound/indices"):
        f.complete_request(provider,scope=SCOPE,prior=prior)
    assert not calls


def test_original_sources_still_exact():
    from posteriorforge.artifacts import sha256_file
    from posteriorforge import performance_prior_coordinates_v1 as failed
    assert failed.source_sha256()=="71e5e2906fd663534e56878de1fc4aff55093e3d950a13f348f57417ca7c01d3"
    assert old.source_sha256()==f.ORIGINAL_FIFO_SHA256
