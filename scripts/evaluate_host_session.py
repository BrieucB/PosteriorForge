#!/usr/bin/env python3
"""Independent CPU review of complete150k ABA host sessions.

Built with PriorLabs-TabPFN. New host facts are authenticated in their own schema.
Metric math is reused unchanged; no model calls, reference integration, row repair,
cross-case comparator, or qualification/accuracy-threshold waiver occurs.
"""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,re,subprocess
from pathlib import Path
from types import ModuleType
import numpy as np
from posteriorforge.artifacts import sha256_file,write_json
from posteriorforge.banks import load_context,load_heldout,derive_seed
from posteriorforge.inference import CHECKPOINT_SHA256,support_mask,frozen_regressor_settings
from posteriorforge.science.analytical_benchmark_v2 import build_problem,array_descriptor
from posteriorforge.validation import sample_integrity,assess_samples

SCHEMA="posteriorforge.host_session_review.v1"
COMMIT="009c01c19f22884918675dd0663983d407ef7db1"
RUNNER_SHA="a578e2d5aba3275bf17e9b080e7cbda4972b17eb2476e180d83ba9facdcd2586"
ADAPTER_SHA="6824d5b874122699ed7dd354b9295469ca4fd9cd4579e4827e2530e11ac1ca9c"
REVIEW_SHA="1c9a185aa5ce28a1d5fa56dbd012522d46c4a2f5c55faf4e0997f63ad0285753"
BINDER_SHA="284d3b061696d4d823666e12d3f79c720f87692b40f45743296cfb8e93865424"
INPUT_SHA={"host_qualification":"18179d2d2161c7713050be99c03d3b8c8a92d82cd1b310fb69134e112a3bf8ba",
 "gqa_qualification":"a83e4cfb925317aea7a4d576deee9e0337fb66bd3d3e80470f7e7102b4c4ea48",
 "portable_batch_qualification":"7bc0e4449abe4aaf0a62b0df55fb25eda423f821524955c9266f8f91761aa77c",
 "allocator_evidence":"0ab24c37e27467b20da531bfea91d1216e46501f925178592957eea9f494b75b",
 "workspace_evidence":"3781229c5930d654eb628b002950eba1a239d50dee2215c088d7048899bb4602",
 "failed_host_proof":"1c849f74784937c69e8d76f769eeb0f7785b1a6bbccfdb7f6e12903e10b6defc"}
PRIOR_FIFO_MANIFEST_SHA="c0937b4b08c23d78829695b1207775cefbd3d161b592cb0139d25241927f1e42"
VARIANT="execution_gqa_host_trim_fifo_v1"
ROLES={"head_proposal":"execution-head-proposal","head_acceptance":"execution-head-acceptance","component_uniform":"execution-component-uniform"}
GLOBALS=("default_torch_cpu","default_torch_cuda","numpy_global","python_global")

def require(value,message):
 if not value:raise ValueError(message)

def within(root,name):
 root=Path(root).resolve();path=(root/name).resolve()
 require(path.is_relative_to(root),"artifact path escapes session")
 return path

def load_script(name,expected):
 path=Path(__file__).with_name(name);require(sha256_file(path)==expected,"reviewed script source changed: "+name)
 spec=importlib.util.spec_from_file_location("_host_review_"+path.stem,path)
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def load_sources():
 repo=Path(__file__).resolve().parents[1]
 for name,digest in (("src/posteriorforge/performance_host_session.py",RUNNER_SHA),("src/posteriorforge/execution_host_session.py",ADAPTER_SHA)):
  data=subprocess.run(["git","-C",str(repo),"show",COMMIT+":"+name],check=True,capture_output=True).stdout
  require(hashlib.sha256(data).hexdigest()==digest,"pinned host producer Git source differs")
 from posteriorforge import execution_gqa_host_trim as host
 host.require_frozen_sources()
 require(host.execution_source_sha256()=="1868d0e22906773fe62baeadbdb3124f9f2086ffdfd8cb918caca6aee3cb3e99","qualified host engine source differs")
 metrics=load_script("evaluate_execution_variant.py",REVIEW_SHA)
 load_script("evaluate_cdf_pilot.py",BINDER_SHA)
 return metrics,metrics.bindings,metrics.reviewed_v4_runner(),host

def counters_add(left,right):
 if isinstance(right,dict):
  require(isinstance(left,dict),"counter structure changed")
  result={}
  for key in set(left)|set(right):
   value=right.get(key,left.get(key));zero={} if isinstance(value,dict) else [0]*len(value) if isinstance(value,list) else 0
   result[key]=counters_add(left.get(key,zero),right.get(key,zero))
  return result
 if isinstance(right,list):
  require(isinstance(left,list) and len(left)==len(right),"counter member-vector shape changed")
  return [counters_add(a,b) for a,b in zip(left,right)]
 require(type(left) is int and type(right) is int and left>=0 and right>=0,"nonnegative integer counters required")
 return left+right

def host_sampling_policy(policy,seeds):
 from posteriorforge import execution_mixture as mixture
 for key,value in mixture.execution_metadata().items():
  require(type(policy.get(key)) is type(value) and policy.get(key)==value,"host sampling primitive source/policy differs: "+key)
 require(policy.get("head_policy")=="raw_mass_rejection_v1" and policy.get("uniform_assignment")=="preassigned_per_stage_by_physical_row"
   and policy.get("head_retry_preserves_row_uniform") is True,"host per-row stage/head retry law differs")
 streams=policy.get("streams",{});require(set(streams)==set(ROLES),"three independent host RNG roles required")
 dtypes={"head_proposal":"torch.int64","head_acceptance":"torch.float64","component_uniform":"torch.float32"}
 for key,role in ROLES.items():
  item=streams[key]
  require(item.get("seed")==seeds[key] and item.get("role")==role and item.get("dtype")==dtypes[key]
    and item.get("device")=="cpu" and item.get("generator")=="torch.Generator","actual host sampling generator policy differs")


def coupled_payload(left,right):
 require(left.shape==right.shape==(50000,21),"coupled diagnostic requires two complete same-case cohorts")
 difference=np.abs(left-right)
 return {"bitwise_array_equal":np.array_equal(left,right),"changed_rows":int(np.count_nonzero(np.any(left!=right,axis=1))),
   "changed_entries":int(np.count_nonzero(left!=right)),"max_absolute":float(difference.max()),
   "scope":"coupled finite-cohort diagnostic; no population equivalence claim"}

def request_counts(request,*,visit,previous,ops):
 waves=request["proposal_wave_sizes"];events=request["stage_events"];nwave=len(waves)
 require(nwave>0 and len(events)==21*nwave,"actual logical stage coverage differs from generated waves")
 expect_routes={route:{kind:[0]*8 for kind in ("calls","rows")} for route in ("cached","uncached")}
 shared=0;component=0;heads=0;rejected=0
 for index,event in enumerate(events):
  wave,stage=divmod(index,21);sample=event["sampler"]
  require(event["proposal_round"]==wave+1 and event["stage_index"]==stage and event["proposed_rows"]==waves[wave] and event["query_batch_size"]==1024,
          "actual wave/stage order/row count changed")
  require(event["precision"]=={"autocast":False,"forced_inference_dtype":"torch.float32"},"actual FP32 stage precision changed")
  cache=event["cache_after"]
  require(len(cache["parameter_count_by_member"])==8 and set(cache["parameter_count_by_member"].values())=={11081864} and cache["parameter_dtypes"]==["torch.float32"],
          "actual all8 frozen member/parameter types changed")
  acquire=event["host_acquire"];cold=visit==0 and wave==0
  require(acquire["case"]==request["case"] and acquire["stage_index"]==stage and acquire["actual_resident_stage_fit_count"]==int(cold)
    and acquire["retained_stage_hit_count"]==int(not cold),"cold fits/retained acquisitions relabeled")
  require(acquire["logical_acquire_index"]==previous.get("host",{}).get("logical_prepare_count",0)+index+1,"global logical-acquire order differs")
  acquisition_delta=acquire["host_counter_delta"]
  for key,expected in {"logical_prepare_count":1,"actual_prepare_miss_count":int(cold),"cache_hit_count":int(not cold),"reload_count":int(not cold),"offload_count":0}.items():
   require(acquisition_delta.get(key,0)==expected,"per-acquisition host counter delta differs: "+key)
  if not cold:
   require(all(value==0 for value in acquire["actual_fit_seconds"].values()) and acquire["hit_provenance_times_are_not_charged_as_new_work"] is True,
           "retained-hit provenance time charged as new fitting")
   require(all(event.get(key)==0 for key in ("fit_stock_cache_seconds","resident_move_seconds","uncached_fit_seconds")),"warm event reuses cold-fit timing")
  require(sample["uniform_draws"]==waves[wave],"per-row stage uniform assignment differs")
  if stage==0:
   require(sample.get("stage0_full_mixture_shared") is True and event["model_query_chunks"]==1
     and sample.get("density_scope")=="proposal-wave-local; no cross-request/case reuse","shared stage0 scope differs")
   shared+=1;continue
  assignments=sample["head_assignment_counts_by_member"];rejects=sample["head_rejection_counts_by_member"]
  require(len(assignments)==len(rejects)==8 and sum(assignments)==sample["head_candidates"]
    and sum(rejects)==sample["head_rejections"] and sum(assignments)-sum(rejects)==waves[wave],"per-row head retries/count conservation differs")
  calls=0
  for member in range(8):
   histogram=sample["neural_query_batch_sizes_by_member"][str(member)]
   require(all(key.isdigit() and 1<=int(key)<=1024 and type(value) is int and value>0 for key,value in histogram.items()),"actual head neural batches invalid")
   ncalls=sum(histogram.values());nrows=sum(int(key)*value for key,value in histogram.items());calls+=ncalls
   require(nrows==assignments[member],"actual member rows differ from assigned head proposals")
   route="uncached" if stage==10 and member==5 else "cached"
   expect_routes[route]["calls"][member]+=ncalls;expect_routes[route]["rows"][member]+=nrows
  require(calls==event["model_query_chunks"],"actual model-query chunks differ from member histogram")
  component+=calls;heads+=sample["head_candidates"];rejected+=sample["head_rejections"]
 sampling={"stage0_shared_predictions":shared,"component_predictions":component,"head_candidates":heads,"head_rejections":rejected}
 require(request["sampling_counters"]==sampling,"request sampler counters differ from stage witnesses")
 accounting=request["host_request_accounting"];delta=accounting["counter_delta"];end=counters_add(previous,delta)
 require(end==accounting["counter_end"]==request["host_session_after_request"]["aggregate_counters"],"cumulative request counter snapshots differ")
 logical=21*nwave;fits=21 if visit==0 else 0;hits=logical-fits
 for key,expected in {"logical_prepare_count":logical,"actual_prepare_miss_count":fits,"cache_hit_count":hits,"reload_count":hits,"offload_count":logical}.items():
  require(delta["host"].get(key,0)==expected,"actual host fit/hit/reload/offload counter differs: "+key)
 require(delta["host"].get("retention_disable_count",0)==0 and delta["host"].get("context_invalidation_count",0)==0,"retention disabled or immutable bank invalidated")
 require(delta["base"]["prepare_count"]==fits,"base fit count differs from actual cold misses")
 for route in ("cached","uncached"):
  for kind in ("calls","rows"):
   require(delta["routes"][route+"_"+kind+"_by_member"]==expect_routes[route][kind],"actual per-member route counter differs")
 cached_calls=sum(expect_routes["cached"]["calls"]);cached_rows=sum(expect_routes["cached"]["rows"])
 uncached_calls=sum(expect_routes["uncached"]["calls"]);uncached_rows=sum(expect_routes["uncached"]["rows"])
 require(delta["base"]["predict_count"]==cached_calls+shared and delta["base"]["query_rows"]==cached_rows+shared,"base resident predictions/rows differ")
 require(delta["hybrid"]["uncached_fit_count"]==int(visit==0)
   and delta["hybrid"]["uncached_predict_count"]==uncached_calls and delta["hybrid"]["uncached_query_rows"]==uncached_rows
   and delta["hybrid"]["resident_component_predict_count"]==cached_calls,"hybrid original/cached member counters differ")
 require(delta["base"]["model_rng_isolation_count"]==delta["base"]["prepare_count"]+delta["base"]["predict_count"]+delta["hybrid"]["uncached_fit_count"]+delta["hybrid"]["uncached_predict_count"],
         "actual model RNG isolation invocation count differs")
 require(delta["component"]["component_predict_count"]==expect_routes["cached"]["calls"]
   and delta["component"]["component_query_rows"]==expect_routes["cached"]["rows"]
   and delta["component"]["mass_observed_rows"]==cached_rows,"all8 actual component/mass observation counters differ")
 checks=request["gqa_prediction_checks"];gqa=delta["gqa"]
 require(checks["verified_calls"]==gqa["contexts"]==shared+component and checks["uncached_fallback_calls"]==uncached_calls
   and checks["all_aliases_restored"] is True and checks["all_sources_unchanged"] is True,"actual GQA execution/restoration coverage differs")
 require(gqa["attention_call_count"]>=gqa["expanded_attention_calls"]>0 and gqa["expanded_tensor_views"]>=gqa["expanded_attention_calls"],"actual positive attention/view counters required")
 fingerprint=checks["attention_fingerprints"];keys=("original_attention_code_sha256","executed_original_attention_code_sha256","active_attention_code_sha256")
 require(set(fingerprint)==set(keys) and all(re.fullmatch("[0-9a-f]{64}",fingerprint[key]) for key in keys)
    and fingerprint[keys[0]]==fingerprint[keys[1]]!=fingerprint[keys[2]],"actual same-process attention fingerprints differ")
 calls=8*shared+component
 require(request["cdf_call_count_delta"]==request["expected_request_cdf_calls"]==calls,"per-request actual CDF counter differs")
 after=request["host_session_after_request"]
 require(after["all_member_cdf_gqa_states_restored"] is True and after["host"]["retained_stages"]==21
   and after["host"]["global_cuda_allocated_bytes"]==8519680
   and sorted(row["stage_index"] for row in after["host"]["stages"])==list(range(21))
   and all(row["state"]["cuda_storage_bytes"]==0 for row in after["host"]["stages"]),
   "request final all21 CPU residency/library workspace differs")
 require(len(accounting["acquisitions"])==logical and accounting["acquisitions"]==[event["host_acquire"] for event in events],"request acquisition archive differs")
 require(request["timings"]["actual_stage_fit_count"]==fits and request["timings"]["retained_stage_hits"]==hits
   and request["timings"]["cold_request"] is bool(fits),"cold/warm request timing attribution differs")
 return end,{"logical_acquires":logical,"actual_fits":fits,"retained_hits":hits,"reloads":hits,"offloads":logical,
    "migration_count":logical+hits,"actual_CDF_calls":calls,"sampler_counts":sampling}

def authenticate_session(path,*,context,queries,metrics,binder,ops,host,reference_custody=None,input_overrides=None,sdk_root=None):
 path=Path(path).resolve();root=path.parent;footer=json.loads(path.read_text())
 expected={"schema":"posteriorforge.inference.host_session.v1","variant":VARIANT,"status":"complete","session_complete":True,
   "all_eight_cases_complete":False,"requested_cases":[0,1,0],"isolation_aba_mode":True,"requested_vectors_per_case":50000,
   "requested_session_total_vectors":150000,"returned_session_total_vectors":150000,"completed_sampling_requests":3,
   "context_seed":20261111,"context_size":4096,"query_batch_size":1024,"direct_host_query_cap":1024,"large_batch_composition_claimed":False,
   "world_size":1,"rank":0,"runner_source_sha256":RUNNER_SHA,"adapter_source_sha256":ADAPTER_SHA,"performance_first_authorized":True,
   "absolute_accuracy_gate_required_for_performance":False,"original_gate_unchanged":True,"sources_unchanged":True,"setup_actual_stage_fits":0}
 for key,value in expected.items():require(type(footer.get(key)) is type(value) and footer.get(key)==value,"complete host session contract differs: "+key)
 problem=build_problem();common={"checkpoint_sha256":CHECKPOINT_SHA256,"scientific_fingerprint":problem.scientific_fingerprint,
   "parameter_order":list(problem.parameter_order),"context_theta":array_descriptor(context.theta),"context_obs":array_descriptor(context.obs)}
 for key,value in common.items():require(footer.get(key)==value,"actual session bank/science binding differs: "+key)
 require(type(footer.get("repetition")) is int and footer["repetition"]>=0,"explicit session repetition required")
 frozen=footer["frozen_sampling_utility"]
 require(frozen.get("sha256")=="23ccad8b5bbcdd7a48340b349d515ca870763359cfeb71333e02fc2b6a8aca3b"
   and frozen.get("git_commit")=="36594f45debeaf1921b10b7bc12a1daa8539ee37"
   and frozen.get("git_blob")=="a910d353032575a3cd0193b9582d771955baf913"
   and frozen.get("requires_reviewed_source_checkout") is True and frozen.get("old_run_execution_not_called") is True
   and frozen.get("relative_runtime_helpers")=={"scripts/recover_gqa_hybrid_qualification.py":"83064f54712014b5c4c66569e854872caff03593cfdc337f2472a7b5723aea02",
      "src/posteriorforge/execution_proposal_buffer.py":"8fb2ba1fb63cd917378ff9bc8b6f70be7410862c509cd22cb2fb6f454d3d35fa"}
   and sha256_file(root/"sources/sampling-v4.py")==frozen["sha256"],"authenticated frozen math archive differs")
 require(footer["host_execution"]==host.execution_metadata()
   and footer.get("host_qualification_manifest_sha256")==INPUT_SHA["host_qualification"]
   and footer.get("host_qualification_producer_sha256")=="9e4043f5ab06b1f5bdfb4d30a91992420f368170a40151bdc9556ae56b80d443",
   "qualified host source/policy differs")
 descriptor=binder.validate_adapter_descriptor(footer["compatibility_adapter"])
 inputs={}
 require(set(footer["source_inputs"])==set(INPUT_SHA),"all original qualified/source inputs required")
 for name,digest in INPUT_SHA.items():
  identity=footer["source_inputs"][name];current=Path((input_overrides or {}).get(name,identity["path"]))
  require(identity["sha256"]==digest and sha256_file(current)==digest,"qualified input/source checksum differs: "+name);inputs[name]=current
 portable=load_script("verify_portable_gqa_recovery.py","3234694dc1d2bb242d0995d465ddc69250cf67fb05e9127e7380fe0eb70ea6ee")
 receipt=portable.verify_portable_recovery(inputs["gqa_qualification"],inputs["portable_batch_qualification"],sdk_root=sdk_root)
 proof=json.loads(inputs["host_qualification"].read_text())
 qualifier=load_script("qualify_execution_gqa_host_trim.py","9e4043f5ab06b1f5bdfb4d30a91992420f368170a40151bdc9556ae56b80d443")
 predicates=qualifier.final_predicates(proof,proof["compatibility_adapter"])
 require(proof["final_predicates"]==predicates and all(predicates.values()) and len(proof["migrations"])==105,
         "all original host qualification/migration predicates required")
 for key,value in common.items():require(proof.get(key)==value,"host proof/session actual context differs: "+key)
 require(proof["query_a"]==array_descriptor(queries[0]) and proof["query_b"]==array_descriptor(queries[1]),"host qualified A/B observations differ")
 paths=footer["request_manifests"];require(paths==[f"visit-{i:02d}-case-{case}/manifest.json" for i,case in enumerate((0,1,0))]
   and set(footer["request_manifest_sha256"])==set(paths),"exact ordered ABA request manifests/hashes required")
 requests=[];arrays=[];summary=[];previous={};cdf=0;all_ids=[];gen_ids=[];fingerprints=[];checkpoint_verified=set()
 for visit,name in enumerate(paths):
  child_path=within(root,name);require(sha256_file(child_path)==footer["request_manifest_sha256"][name],"completed request/footer hash differs")
  child=json.loads(child_path.read_text());case=(0,1,0)[visit]
  required={"schema":"posteriorforge.inference.host_request.v1","variant":VARIANT,"status":"complete","visit_index":visit,"case":case,
    "requested_count":50000,"benchmark_total_count":50000,"end_to_end":True,"scientific_run":True,"world_size":1,"rank":0,
    "context_seed":20261111,"context_size":4096,"query_batch_size":1024,"microbatch":50000,"max_iter_rejection":None,"log_density":"not_used",
    "partial_vector_filtering":False,"clipping_or_repair_performed":False,"sources_unchanged":True,"shutdown_released_all_stages":True,
    "session_id":footer["session_id"],"runner_source_sha256":RUNNER_SHA,"adapter_source_sha256":ADAPTER_SHA,"repetition":footer["repetition"],
    "context_filter":"no_filtering","observation_preprocessing":"raw; TabPFN frozen preprocessing","completed_stage_indices":list(range(21)),
    "frozen_sampling_utility_sha256":frozen["sha256"],"host_qualification_manifest_sha256":INPUT_SHA["host_qualification"],
    "direct_host_query_cap":1024,"large_batch_composition_claimed":False,"performance_first_authorized":True,
    "absolute_accuracy_gate_required_for_performance":False,"original_gate_unchanged":True}
  for key,value in {**common,**required,"query":array_descriptor(queries[case])}.items():require(type(child.get(key)) is type(value) and child.get(key)==value,"host request binding differs: "+key)
  from posteriorforge import execution_cache
  require(child["host_engine"]==footer["host_execution"] and child["execution_engine"]==execution_cache.execution_metadata(),
          "request host/base source identity differs from shared session")
  require((child_path.parent/child["session_footer"]).resolve()==path and child["compatibility_adapter"]==footer["compatibility_adapter"],"request has no exact shared restored footer")
  seeds={key:derive_seed(20261111,role,case=case,rank=0,repetition=footer["repetition"]) for key,role in ROLES.items()}
  require(child["seed_streams"]==seeds and child["global_rng_preserved"] is True,"independent request RNG stream binding differs")
  require(set(child["rng_before"])==set(child["rng_after"])==set(ROLES)|set(GLOBALS)
    and all(isinstance(value,str) and re.fullmatch("[0-9a-f]{64}",value) for state in (child["rng_before"],child["rng_after"]) for value in state.values()),
          "complete recorded request RNG-state hashes required")
  require(all(child["rng_before"][key]==child["rng_after"][key] for key in GLOBALS),"request global RNG was consumed")
  require(set(child["generator_identities"])==set(ROLES),"fresh generator role identities missing")
  host_sampling_policy(child["sampling_policy"],seeds)
  require(child["proposal_buffer_policy"]==ops.proposal_buffer_policy(1200),"fresh FIFO source/policy differs")
  scope=ops.proposal_buffer_scope({**common,"query":array_descriptor(queries[case])},context_seed=20261111,case=case,repetition=footer["repetition"],rank=0,world_size=1,seeds=seeds)
  ops.validate_proposal_buffer_record(child["proposal_buffer"],world_size=1,rank=0,scope=scope)
  record=child["proposal_buffer"];counts=record["original_sampler_counts"]
  require(child["proposal_wave_sizes"]==record["actual_generated_wave_sizes"] and child["rejection"]=={
    "support_calls":counts["original_support_calls"],"proposed":counts["original_checked_count"],"accepted_before_final_trim":counts["accepted_before_trim"],"returned":50000},
    "original support/API/FIFO counts differ")
  if reference_custody is not None:binder.validate_run_reference_binding(child,reference_custody[case])
  settings=child["regressor_settings"];require(settings=={**frozen_regressor_settings(settings["model_path"]),"fit_mode":"fit_with_cache"},"scientific regressor settings changed")
  checkpoint=Path((input_overrides or {}).get("checkpoint",settings["model_path"]))
  if checkpoint not in checkpoint_verified:
   require(sha256_file(checkpoint)==CHECKPOINT_SHA256,"actual immutable checkpoint bytes differ");checkpoint_verified.add(checkpoint)
  values=binder.load_samples(child_path,child);require(sample_integrity(values,expected_count=50000,dimension=21,support_mask=support_mask(values))["passed"],"invalid complete50k host cohort")
  previous,counts_review=request_counts(child,visit=visit,previous=previous,ops=ops);cdf+=counts_review["actual_CDF_calls"]
  require(child["session_cdf_call_count_total_at_footer"]==descriptor["patched_call_count"],"request shared CDF footer differs")
  requests.append(child);arrays.append(values);summary.append(counts_review);all_ids.append(child["fifo_instance_identity"]);gen_ids.extend(child["generator_identities"].values());fingerprints.append(child["gqa_prediction_checks"]["attention_fingerprints"])
 require(all(type(value) is int and value>0 for value in all_ids+gen_ids) and len(set(all_ids))==3 and len(set(gen_ids))==9,
         "FIFO or independent generator object reused across requests")
 require(all(value==fingerprints[0] for value in fingerprints),"same-process GQA source fingerprints differ between requests")
 require(previous==footer["session_counters"]==footer["retention_before_shutdown"]["aggregate_counters"] and cdf==descriptor["patched_call_count"],"cumulative session GQA/CDF/member accounting differs")
 shutdown=footer["shutdown"]
 require(shutdown.get("aggregate_counters")==previous and shutdown.get("accounting_audit_passed") is True
   and shutdown.get("all_member_cdf_gqa_states_restored") is True and shutdown.get("stage_release_call_succeeded") is True
   and shutdown.get("shutdown_released_all_stages") is True and shutdown.get("retained_stages_before_release")==21
   and shutdown.get("retained_stages_after_release")==0 and shutdown.get("active_request_aborted") is False
   and "shutdown_error" not in footer,"retained-state release/CDF footer incomplete")
 equality=coupled_payload(arrays[0],arrays[2])
 require(equality["bitwise_array_equal"] and requests[0]["samples_sha256"]==requests[2]["samples_sha256"],"actual whole50k A/B/A payload isolation failed")
 for key in ("seed_streams","rejection","proposal_wave_sizes","sampling_counters","rng_before","rng_after"):
  require(requests[0][key]==requests[2][key],"actual A/B/A stream/count isolation differs: "+key)
 first,last=requests[0],requests[2]
 aba={key:first[key]==last[key] for key in ("seed_streams","rejection","proposal_wave_sizes","sampling_counters","rng_before","rng_after")}
 fifo_keys=("request_scope_sha256","api_requested_sizes","actual_generated_wave_sizes","generated_count","original_api_delivered_count","remaining_unused_buffer_count","original_sampler_counts")
 aba.update(fifo_scopes_and_counts=all(first["proposal_buffer"][key]==last["proposal_buffer"][key] for key in fifo_keys),
   array_payload_sha256=first["samples"]["sha256"]==last["samples"]["sha256"],npz_payload_sha256=first["samples_sha256"]==last["samples_sha256"],
   fresh_fifo_each_request=len(set(all_ids))==3,fresh_sampling_generators_each_request=len(set(gen_ids))==9)
 require(all(aba.values()) and footer["isolation_aba"]["passed"] is True and footer["isolation_aba"]["predicates"]==aba,
         "producer ABA footer differs from independent full-predicate replay")
 return footer,requests,arrays,{"source_commit":COMMIT,"runner_sha256":RUNNER_SHA,"adapter_sha256":ADAPTER_SHA,
   "portable_recovery":receipt,"host_qualification_105_full_hash_migrations_verified":True,"request_accounting":summary,
   "actual_session_migrations":sum(row["migration_count"] for row in summary),
   "production_copy_validation":"identity/layout guards; no production full-storage byte-hash claim",
   "all150000_integrity_passed":True,"independent_ABA_payload_check":equality,
   "source_input_sha256":dict(INPUT_SHA),"shared_footer_authenticated":True}

def relative_for_case(case,values,*,control,reference,metrics):
 if case==0:
  require(control is not None,"immutable same-case corrected primary required")
  return metrics.comparative_assessment(values,control,reference)
 return {"control":{"status":"not_available","case":case,
   "reason":"No saved unoptimized raw33 CDF-corrected same-case primary; cross-case comparison prohibited."},
   "relative_distribution_review":None,"numeric_review_status":"same_case_unoptimized_control_unavailable"}


def run(*,root,session_manifest,output,sdk_root,input_overrides=None):
 root=Path(root).resolve();results=root/"results";metrics,binder,ops,host=load_sources()
 gate_path=results/"gate-final.json";require(sha256_file(gate_path)==binder.BASELINE_GATE_SHA256,"immutable original gate changed")
 gate=json.loads(gate_path.read_text());context=load_context(results/"banks-v1",20261111,4096)
 queries={case:load_heldout(results/"banks-v1",case).obs.reshape(1,33) for case in (0,1)}
 references={};custody={}
 for case in (0,1):
  _,_,references[case],custody[case]=binder.reference_for_case(results,gate,queries[case],case)
 footer,requests,arrays,authentication=authenticate_session(session_manifest,context=context,queries=queries,metrics=metrics,binder=binder,ops=ops,host=host,
   reference_custody=custody,input_overrides=json.loads(Path(input_overrides).read_text()) if input_overrides else None,sdk_root=sdk_root)
 control_path=results/"cdf-arithmetic-v1/context-4096/seed-20261111-case-0/manifest.json"
 require(sha256_file(control_path)==metrics.CONTROL_MANIFEST_SHA256,"immutable corrected case0 primary changed")
 control_manifest=json.loads(control_path.read_text());binder.validate_pilot_header(control_manifest,seed=20261111,case=0,context_size=4096)
 binder.validate_run_reference_binding(control_manifest,custody[0]);control=binder.load_samples(control_path,control_manifest)
 prior_path=results/"performance-first-v1/gqa-fifo-fp32-batch-1024/rep-0/seed-20261111-case-0/manifest.json"
 require(sha256_file(prior_path)==PRIOR_FIFO_MANIFEST_SHA,"coupled prior FIFO1024 identity changed")
 prior_manifest=json.loads(prior_path.read_text());metrics.validate_execution_header(prior_manifest,context=context,query=queries[0],control=control_manifest)
 prior=binder.load_samples(prior_path,prior_manifest)
 reviews=[]
 for request,values in zip(requests,arrays):
  case=request["case"];absolute=assess_samples(values,references[case],support_mask=support_mask(values),expected_count=50000)
  require(absolute["reference_qualified"],"all saved reference criteria must qualify")
  value={"visit_index":request["visit_index"],"case":case,"samples_sha256":request["samples_sha256"],
    "absolute_physical_assessment":absolute,"sample_integrity":sample_integrity(values,expected_count=50000,dimension=21,support_mask=support_mask(values)),
    "timings":request["timings"],"memory":request["memory"],"rejection":request["rejection"],"proposal_buffer":request["proposal_buffer"]}
  if case==0:
   value.update(relative_for_case(case,values,control=control,reference=references[case],metrics=metrics))
   value["control"]={"status":"available","case":0,"manifest_sha256":sha256_file(control_path)}
   value["coupled_prior_FIFO1024"]={**coupled_payload(values,prior),"prior_manifest_sha256":PRIOR_FIFO_MANIFEST_SHA,
     "relative_distribution_review":metrics.comparative_assessment(values,prior,references[0])["relative_distribution_review"]}
  else:
   value.update(relative_for_case(case,values,control=None,reference=references[case],metrics=metrics))
  reviews.append(value)
 output=Path(output);output.mkdir(parents=True,exist_ok=False)
 report={"schema":SCHEMA,"attribution":"Built with PriorLabs-TabPFN","session_manifest_file":str(Path(session_manifest).resolve()),
   "session_manifest_sha256":sha256_file(session_manifest),"authentication":authentication,"requests":reviews,"references":custody,
   "all150000_integrity_passed":True,"performance_first_authorized":True,"absolute_accuracy_gate_required_for_performance":False,
   "relative_thresholds":{".01":"CDF investigation flag",".05_reference_SD":"mean investigation flag",".05_fraction":"SD investigation flag"},
   "no_model_calls":True,"no_GPU_calls":True,"no_reference_integrations":True,"original_gate_unchanged":True,
   "source":{"reviewer_sha256":sha256_file(__file__),"metric_reviewer_sha256":REVIEW_SHA,"science_binding_helper_sha256":BINDER_SHA},
   "interpretation":"Three complete50k requests, not a new posterior qualification. Relative flags and physical failures stay visible; no earlier permutation p-values are transferred."}
 write_json(output/"host-session-review.json",report)
 write_json(output/"checksums.json",{p.name:sha256_file(p) for p in output.iterdir() if p.is_file()})
 require(sha256_file(gate_path)==binder.BASELINE_GATE_SHA256,"immutable gate changed during CPU review")
 print(json.dumps({"output":str(output),"all150000_integrity_passed":True,"actual_session_migrations":authentication["actual_session_migrations"],
   "requests":[{"visit":r["visit_index"],"case":r["case"],"relative_status":r["numeric_review_status"],"physical_status":r["absolute_physical_assessment"]["status"]} for r in reviews]},indent=2))
 return report

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ("root","session-manifest","output","sdk-root"):p.add_argument("--"+name,required=True)
 p.add_argument("--input-overrides",help="JSON name:path map for relocated artifacts; every recorded SHA remains mandatory")
 run(**vars(p.parse_args()))
if __name__=="__main__":main()
