"""Request-local FIFO lookahead for complete proposal vectors.

Built with PriorLabs-TabPFN. The buffer never inspects support, repairs rows,
reorders candidates or draws RNG. The original rejection sampler receives
EXACTLY the row count it requested. Initial lookahead is 1200 total rows split
across 1/2/4 independent workers; later shortages generate exactly the deficit.
"""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from threading import Lock,get_ident

POLICY_ID="complete_vector_fifo_lookahead_v1"
SCHEMA="posteriorforge.execution.proposal_buffer.v1"
BENCHMARK_TOTAL_COUNT=50000
GLOBAL_LOOKAHEAD=1200
VECTOR_WIDTH=21
ORIGINAL_SAMPLER_SHA256="e6f862a176eae4cf0269c49655cf2edefc854428461445c74aa2b02f3b144512"


def source_sha256():return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def worker_policy(world_size,rank):
    if type(world_size) is not int or world_size not in (1,2,4) or type(rank) is not int or not 0<=rank<world_size:
        raise ValueError("one/two/four workers and a valid original rank required")
    return {"benchmark_total_count":BENCHMARK_TOTAL_COUNT,"world_size":world_size,"rank":rank,
        "worker_expected_return_count":BENCHMARK_TOTAL_COUNT//world_size,
        "global_initial_lookahead":GLOBAL_LOOKAHEAD,"worker_initial_lookahead":GLOBAL_LOOKAHEAD//world_size}


def _scope(value):
    if not isinstance(value,dict) or "case" not in value or "query" not in value:
        raise ValueError("explicit request scope containing case and query required")
    snapshot=deepcopy(value);encoded=json.dumps(snapshot,sort_keys=True,separators=(",",":"),allow_nan=False)
    return snapshot,hashlib.sha256(encoded.encode()).hexdigest()


def _copy(value):return value.clone() if hasattr(value,"clone") else value.copy()


class ProposalBuffer:
    """FIFO adapter for a provider returning (CPU FP32[N,21], None).

    One instance is bound to one posterior request, process, thread and provider.
    finalize requires the actual original-sampler counts and50k/global output
    contract. reset is allowed only after completion or a terminal failure.
    Extra proposal kwargs are deliberately unsupported: bind the scientific
    request and independent streams in the provider, then pass the adapter to
    the unchanged accept_reject_sample API.
    """
    def __init__(self,provider,*,scope,concatenate,world_size=1,rank=0):
        if not callable(provider) or not callable(concatenate):raise TypeError("provider and explicit backend concatenate callable required")
        self._owner_pid=os.getpid();self._owner_thread=get_ident();self._busy=Lock()
        self._policy=worker_policy(world_size,rank);self._concatenate=concatenate
        self._reset_count=0;self._start(provider,scope)

    def _start(self,provider,scope):
        self._scope,self._scope_sha256=_scope(scope)
        self._provider=provider;self._provider_identity=provider;self._provider_code=getattr(provider,"__code__",None)
        self._buffer=None;self._first=True;self._failed=False;self._finalized=False;self._failure=None
        self._api_requests=[];self._wave_attempts=[];self._generated_waves=[];self._generated_count=0;self._issued_count=0
        self._result=None;self._unused_after_finalize=0

    def _ensure_owner(self):
        if os.getpid()!=self._owner_pid or get_ident()!=self._owner_thread:raise RuntimeError("proposal buffer belongs to its original process/thread")
        if self._provider is not self._provider_identity or getattr(self._provider,"__code__",None) is not self._provider_code:
            raise RuntimeError("bound proposal provider changed")

    @staticmethod
    def _validate_vectors(values,count):
        if tuple(getattr(values,"shape",()))!=(count,VECTOR_WIDTH):raise ValueError("provider must return requested complete 21-vectors")
        if str(getattr(values,"dtype",None)) not in ("float32","torch.float32"):raise ValueError("unmodified CPU FP32 vectors required; no implicit casting")
        if str(getattr(values,"device","cpu"))!="cpu":raise ValueError("proposal buffer retains CPU complete vectors only")

    def _generate(self,count):
        self._wave_attempts.append(count)
        values,log_prob=self._provider(count)
        if log_prob is not None:raise ValueError("proposal buffer does not accept or invent log densities")
        self._validate_vectors(values,count)
        self._generated_waves.append(count);self._generated_count+=count
        return values

    def __call__(self,requested_n,**kwargs):
        self._ensure_owner()
        if type(requested_n) is not int or requested_n<1:raise ValueError("positive original API request size required")
        if kwargs:raise ValueError("bind proposal kwargs in the request-local provider")
        if self._failed or self._finalized:raise RuntimeError("failed/finalized proposal buffer requires an explicit reset")
        if not self._busy.acquire(blocking=False):raise RuntimeError("proposal buffer operation already active")
        try:
            self._api_requests.append(requested_n);parts=[];remaining=requested_n
            if self._buffer is not None:
                take=min(remaining,len(self._buffer));parts.append(self._buffer[:take]);remaining-=take
                self._buffer=_copy(self._buffer[take:]) if take<len(self._buffer) else None
            if remaining:
                lookahead=self._policy["worker_initial_lookahead"] if self._first else 0
                self._first=False;values=self._generate(remaining+lookahead)
                parts.append(values[:remaining]);self._buffer=_copy(values[remaining:]) if lookahead else None
            result=parts[0] if len(parts)==1 else self._concatenate(parts)
            self._validate_vectors(result,requested_n)
            self._issued_count+=requested_n
            if self._generated_count!=self._issued_count+self.buffer_rows:raise RuntimeError("complete-vector FIFO conservation failed")
            return result,None
        except BaseException as error:
            self._failed=True;self._failure=type(error).__name__+": "+str(error);self._buffer=None
            raise
        finally:self._busy.release()

    @property
    def buffer_rows(self):return 0 if self._buffer is None else len(self._buffer)

    def finalize(self,*,original_checked_count,accepted_before_trim,returned_count,original_support_calls):
        self._ensure_owner()
        if not self._busy.acquire(blocking=False):raise RuntimeError("proposal buffer operation already active")
        try:
            if self._failed or self._finalized:raise RuntimeError("only one successful request can be finalized")
            for value in (original_checked_count,accepted_before_trim,returned_count,original_support_calls):
                if type(value) is not int or value<0:raise ValueError("actual nonnegative integer original-sampler counts required")
            if original_checked_count!=self._issued_count or original_support_calls!=len(self._api_requests):raise ValueError("original checked counts differ from exact API deliveries")
            if returned_count!=self._policy["worker_expected_return_count"] or not returned_count<=accepted_before_trim<=original_checked_count:
                raise ValueError("exact50k/global accepted output contract required")
            self._unused_after_finalize=self.buffer_rows
            if self._generated_count!=original_checked_count+self._unused_after_finalize:raise RuntimeError("generated/checked/unused conservation failed")
            self._result={"original_checked_count":original_checked_count,"original_support_calls":original_support_calls,
                "accepted_before_trim":accepted_before_trim,"returned_count":returned_count}
            self._buffer=None;self._finalized=True
            return self.metadata()
        finally:self._busy.release()

    def reset(self,provider,*,scope):
        self._ensure_owner()
        if not callable(provider):raise TypeError("explicit next-request provider required")
        if not self._busy.acquire(blocking=False):raise RuntimeError("proposal buffer operation already active")
        try:
            if not (self._finalized or self._failed):raise RuntimeError("finish or fail the current request before resetting its FIFO")
            self._reset_count+=1;self._start(provider,scope)
        finally:self._busy.release()

    def metadata(self):
        self._ensure_owner()
        return {"schema":SCHEMA,"policy_id":POLICY_ID,"module_sha256":source_sha256(),**self._policy,
            "request_scope":deepcopy(self._scope),"request_scope_sha256":self._scope_sha256,"buffer_reset_count":self._reset_count,
            "api_requested_sizes":list(self._api_requests),"actual_generated_wave_sizes":list(self._generated_waves),
            "generated_wave_attempt_sizes":list(self._wave_attempts),"generated_count":self._generated_count,
            "generated_wave_count":len(self._generated_waves),"required_stage_preparations_for_generated_waves":VECTOR_WIDTH*len(self._generated_waves),
            "original_api_delivered_count":self._issued_count,"remaining_unused_buffer_count":self._unused_after_finalize if self._finalized else self.buffer_rows,
            "live_buffer_rows":self.buffer_rows,"finalized":self._finalized,"failed":self._failed,"failure":self._failure,
            "original_sampler_source_sha256":ORIGINAL_SAMPLER_SHA256,"original_sampler_counts":deepcopy(self._result),
            "support_filtering_performed_by_buffer":False,"row_repair_or_reordering":False,"partial_vector_filtering":False,"log_density":"not_used",
            "all_rows_are_complete_vectors":True,"vector_width":VECTOR_WIDTH,"data_dtype":"CPU FP32; unchanged",
            "rng_draws_by_buffer":0,"seed_coupling":"provider generates lookahead rows; role/seed definitions unchanged but seeded vectors need not be bitwise identical",
            "law_scope":"FIFO rechunking preserves an iid complete-vector proposal stream; provider batch law must be qualified separately",
            "unused_rows_discarded_only_after_original_sampler_completion":self._finalized,
            "attribution":"Built with PriorLabs-TabPFN"}
