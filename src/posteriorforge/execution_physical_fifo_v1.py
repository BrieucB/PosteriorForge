"""Physical Float64 dtype bridge over the immutable complete-vector FIFO.

Built with PriorLabs-TabPFN. The FIFO algorithm and original accept/reject are
unchanged. No cast, codec, model, support filter, row reorder or RNG is added.
Both forensic replay and the new production entry call complete_request.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
from types import FunctionType,SimpleNamespace
from . import execution_proposal_buffer as original
from . import performance_runner as runner

SCHEMA="posteriorforge.execution.physical_float64_fifo.v1"
POLICY_ID="complete_physical_float64_fifo_lookahead_v1"
ORIGINAL_FIFO_SHA256="8fb2ba1fb63cd917378ff9bc8b6f70be7410862c509cd22cb2fb6f454d3d35fa"
ORIGINAL_RUNNER_SHA256="23ccad8b5bbcdd7a48340b349d515ca870763359cfeb71333e02fc2b6a8aca3b"
ORIGINAL_SAMPLER_SHA256="e6f862a176eae4cf0269c49655cf2edefc854428461445c74aa2b02f3b144512"
ORIGINAL_INFERENCE_SHA256="95366efc4871b96c3a1a3f19414ed1fd77d058f02135f4afd95a64b5918b3d3e"
BENCHMARK_TOTAL_COUNT=50000
GLOBAL_LOOKAHEAD=1200
VECTOR_WIDTH=21
DATA_DTYPE="CPU FP64; decoded physical values; unchanged"
worker_policy=original.worker_policy
_NAMES=("__init__","_start","_ensure_owner","_generate","__call__","finalize","reset")
_ORIGINAL_METHODS={name:getattr(original.ProposalBuffer,name) for name in _NAMES}
_ORIGINAL_PROPERTY=original.ProposalBuffer.buffer_rows
_ORIGINAL_METADATA=original.ProposalBuffer.metadata
_RUNNER_POLICY=runner.proposal_buffer_policy
_RUNNER_VALIDATE=runner.validate_proposal_buffer_record


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_sha256():
    return _sha(__file__)


def require_sources():
    if original.source_sha256()!=ORIGINAL_FIFO_SHA256 or _sha(runner.__file__)!=ORIGINAL_RUNNER_SHA256:
        raise RuntimeError("immutable original FIFO/runner source changed")
    for name,function in _ORIGINAL_METHODS.items():
        if getattr(original.ProposalBuffer,name) is not function or getattr(ProposalBuffer,name).__code__ is not function.__code__:
            raise RuntimeError("original FIFO algorithm method changed: "+name)
    if original.ProposalBuffer.buffer_rows is not _ORIGINAL_PROPERTY or original.ProposalBuffer.metadata is not _ORIGINAL_METADATA:
        raise RuntimeError("original FIFO property/metadata changed")
    if runner.proposal_buffer_policy is not _RUNNER_POLICY or runner.validate_proposal_buffer_record is not _RUNNER_VALIDATE:
        raise RuntimeError("original FIFO policy validator changed")


def _validate_vectors(values,count):
    if tuple(getattr(values,"shape",()))!=(count,VECTOR_WIDTH):
        raise ValueError("provider must return requested complete physical21-vectors")
    if str(getattr(values,"dtype",None)) not in ("float64","torch.float64"):
        raise ValueError("unmodified CPU FP64 physical vectors required; no implicit casting")
    if str(getattr(values,"device","cpu"))!="cpu":
        raise ValueError("physical FIFO retains CPU complete vectors only")


def _metadata(self):
    require_sources()
    record=_ORIGINAL_METADATA(self)
    record.update(schema=SCHEMA,policy_id=POLICY_ID,module_sha256=source_sha256(),
        data_dtype=DATA_DTYPE,original_fifo_algorithm_sha256=ORIGINAL_FIFO_SHA256,
        original_fifo_algorithm_code_objects_unchanged=True,
        dtype_validator_only_change="Float32 complete vectors -> decoded physical Float64 complete vectors",
        implicit_casting_performed=False)
    return record


# New class namespace; all FIFO algorithm functions/properties are the originals.
_namespace={**_ORIGINAL_METHODS,"buffer_rows":_ORIGINAL_PROPERTY,
    "_validate_vectors":staticmethod(_validate_vectors),"metadata":_metadata,
    "__module__":__name__,"__doc__":"Unmodified original FIFO algorithm, physical CPU Float64 dtype."}
ProposalBuffer=type("ProposalBuffer",(),_namespace)
_PUBLIC=SimpleNamespace(SCHEMA=SCHEMA,POLICY_ID=POLICY_ID,ProposalBuffer=ProposalBuffer,
    ORIGINAL_SAMPLER_SHA256=ORIGINAL_SAMPLER_SHA256,worker_policy=worker_policy,
    source_sha256=source_sha256)


def guarded_proposal_buffer():
    require_sources()
    return _PUBLIC


def _clone(function,namespace):
    copied=dict(function.__globals__);copied.update(namespace)
    clone=FunctionType(function.__code__,copied,function.__name__,function.__defaults__,function.__closure__)
    clone.__kwdefaults__=function.__kwdefaults__
    return clone


def _runner_namespace():
    return {"guarded_proposal_buffer":guarded_proposal_buffer,
        "PROPOSAL_BUFFER_SHA256":source_sha256(),"proposal_buffer_policy":proposal_buffer_policy}


def proposal_buffer_policy(proposal_lookahead=1200):
    require_sources()
    return _clone(_RUNNER_POLICY,_runner_namespace())(proposal_lookahead)


def validate_proposal_buffer_record(record,*,world_size=1,rank=0,scope=None):
    require_sources()
    if record.get("data_dtype")!=DATA_DTYPE or record.get("implicit_casting_performed") is not False:
        raise ValueError("physical Float64 FIFO record required; no casting")
    if record.get("original_fifo_algorithm_sha256")!=ORIGINAL_FIFO_SHA256 or record.get("original_fifo_algorithm_code_objects_unchanged") is not True:
        raise ValueError("immutable original FIFO algorithm identity required")
    return _clone(_RUNNER_VALIDATE,_runner_namespace())(record,world_size=world_size,rank=rank,scope=scope)


def _original_sampler():
    from importlib import import_module
    from .upstream import load_baseline
    load_baseline()
    module=import_module("npe_pfn.accept_reject_sampler")
    if _sha(module.__file__)!=ORIGINAL_SAMPLER_SHA256:
        raise RuntimeError("original accept/reject source changed")
    return module.accept_reject_sample


def _verify_support(prior,reference):
    actual=prior.support.check
    expected=reference.support.check
    actual_function=getattr(actual,"__func__",None)
    expected_function=getattr(expected,"__func__",None)
    if actual_function is None or actual_function.__code__ is not expected_function.__code__:
        raise RuntimeError("actual original physical support check code required")
    if getattr(prior.support,"event_dim",None)!=reference.support.event_dim:
        raise RuntimeError("original physical support event dimension differs")
    def closure(function):
        return {name:cell.cell_contents for name,cell in zip(function.__code__.co_freevars,function.__closure__ or ())}
    left,right=closure(actual_function),closure(expected_function)
    if set(left)!=set(right):
        raise RuntimeError("original physical support closure differs")
    for key in right:
        if key=="torch":
            valid=left[key] is right[key]
        else:
            valid=left[key]==right[key]
        if not valid:raise RuntimeError("original physical support bound/indices differ: "+key)
    for key in ("calls","proposed","accepted"):
        if type(getattr(prior.support,key,None)) is not int or getattr(prior.support,key)!=0:
            raise ValueError("fresh actual original-support counters required")
    return True


def complete_request(provider,*,scope,prior=None,buffer=None,accept_reject_fn=None):
    """Exactly the original50k support selection; physical Float64 throughout.

    Returns (CPU torch.float64 samples[50000,21], finalized accounting). The
    provider supplies complete already-decoded rows; no codec/model is called.
    Existing production buffer/prior may be supplied for honest failure capture.
    """
    require_sources()
    import torch
    from . import inference
    if _sha(inference.__file__)!=ORIGINAL_INFERENCE_SHA256:
        raise RuntimeError("original physical support source changed")
    original_sampler=_original_sampler()
    if accept_reject_fn is not None and accept_reject_fn is not original_sampler:
        raise RuntimeError("actual original accept/reject callable required")
    reference_prior=inference._prior(torch)
    if prior is None:prior=reference_prior
    _verify_support(prior,reference_prior)
    if buffer is None:
        buffer=ProposalBuffer(provider,scope=scope,concatenate=lambda parts:torch.cat(parts,dim=0),world_size=1,rank=0)
    if type(buffer) is not ProposalBuffer or buffer._provider is not provider or buffer.metadata()["request_scope"]!=scope:
        raise ValueError("same request-local physical-Float64 FIFO/provider/scope required")
    with torch.no_grad():
        samples,_,_=original_sampler(
            proposal=buffer,accept_reject_fn=prior.support.check,num_samples=BENCHMARK_TOTAL_COUNT,
            max_sampling_batch_size=BENCHMARK_TOTAL_COUNT,max_iter_rejection=None,
            show_progress_bars=False,proposal_sampling_kwargs={})
    record=buffer.finalize(
        original_checked_count=prior.support.proposed,accepted_before_trim=prior.support.accepted,
        returned_count=len(samples),original_support_calls=prior.support.calls)
    record.update(enabled=True,source_unchanged=True,policy_sha256=proposal_buffer_policy(1200)["policy_sha256"],
        original_support_check_code_and_closure_verified=True,original_support_source_sha256=ORIGINAL_INFERENCE_SHA256)
    validate_proposal_buffer_record(record,world_size=1,rank=0,scope=scope)
    if samples.dtype!=torch.float64 or samples.device.type!="cpu" or tuple(samples.shape)!=(50000,21):
        raise RuntimeError("original selection did not retain exact physical Float64[50000,21]")
    return samples,record
