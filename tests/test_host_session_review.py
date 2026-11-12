"""CPU accounting/routing tests; synthetic fixtures are never posterior evidence."""
from copy import deepcopy
import importlib.util,json,subprocess,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

ROOT=Path(__file__).parents[1]
spec=importlib.util.spec_from_file_location("host_session_review_tests",ROOT/"scripts/evaluate_host_session.py")
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)


def test_import_cli_and_reviewed_sources_are_torch_free():
 subprocess.run([sys.executable,"-c","import importlib.util,sys;s=importlib.util.spec_from_file_location('h','scripts/evaluate_host_session.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);assert 'torch' not in sys.modules;m.load_sources();assert 'torch' not in sys.modules"],cwd=ROOT,check=True)
 text=subprocess.run([sys.executable,str(Path(h.__file__)),"--help"],capture_output=True,text=True,check=True).stdout
 assert "--sdk-root" in text and "--session-manifest" in text


def test_nested_cumulative_counter_union_and_member_vectors_are_exact():
 assert h.counters_add({"a":2,"nested":{"x":[1,2]}},{"b":3,"nested":{"x":[4,5]}})=={"a":2,"b":3,"nested":{"x":[5,7]}}
 for before,delta in (({"a":[1]},{"a":[1,2]}),({"a":1},{"a":-1}),({"a":True},{"a":1})):
  with pytest.raises(ValueError):h.counters_add(before,delta)


def test_cross_case_control_is_never_used_or_fabricated():
 def prohibited(*args):raise AssertionError("cross-case comparison called")
 metrics=SimpleNamespace(comparative_assessment=prohibited)
 result=h.relative_for_case(1,np.zeros((1,21)),control=np.ones((1,21)),reference={},metrics=metrics)
 assert result["relative_distribution_review"] is None
 assert result["control"]["status"]=="not_available" and result["numeric_review_status"]=="same_case_unoptimized_control_unavailable"
 with pytest.raises(ValueError):h.relative_for_case(0,np.zeros((1,21)),control=None,reference={},metrics=metrics)


def test_full50000_coupled_payload_check_records_differences_without_repair():
 a=np.zeros((50000,21));b=a.copy();b[17,3]=2.
 result=h.coupled_payload(a,b)
 assert not result["bitwise_array_equal"] and result["changed_rows"]==result["changed_entries"]==1 and result["max_absolute"]==2
 np.testing.assert_array_equal(a,0)
 with pytest.raises(ValueError):h.coupled_payload(a[:1],b[:1])


def test_new_host_sampling_schema_does_not_require_invented_v4_dedup_metadata():
 from posteriorforge import execution_mixture as mixture
 seeds={"head_proposal":1,"head_acceptance":2,"component_uniform":3}
 types={"head_proposal":"torch.int64","head_acceptance":"torch.float64","component_uniform":"torch.float32"}
 policy={**mixture.execution_metadata(),"head_policy":"raw_mass_rejection_v1","uniform_assignment":"preassigned_per_stage_by_physical_row","head_retry_preserves_row_uniform":True,
  "streams":{name:{"seed":seeds[name],"role":role,"dtype":types[name],"device":"cpu","generator":"torch.Generator"} for name,role in h.ROLES.items()}}
 assert "stage0_dedup" not in policy
 h.host_sampling_policy(policy,seeds)
 policy["streams"]["head_acceptance"]["dtype"]="torch.float32"
 with pytest.raises(ValueError):h.host_sampling_policy(policy,seeds)


@pytest.fixture(scope="module")
def actual_first_request():
 path=Path("/scratch/project/eu-26-17/eubrieucb/posteriorforge/results/performance-first-v1/gqa-host-session-aba-01/visit-00-case-0/manifest.json")
 if not path.is_file():pytest.skip("actual firstA request not available")
 value=json.loads(path.read_text())
 if "host_request_accounting" not in value:pytest.skip("firstA sampling not complete")
 return value


def test_actual_firstA_all_counters_match_stage_and_member_witnesses(actual_first_request):
 end,summary=h.request_counts(actual_first_request,visit=0,previous={},ops=None)
 assert summary["logical_acquires"]==summary["actual_fits"]==summary["offloads"]==21
 assert summary["retained_hits"]==summary["reloads"]==0
 assert summary["actual_CDF_calls"]==actual_first_request["cdf_call_count_delta"]
 assert end==actual_first_request["host_request_accounting"]["counter_end"]


@pytest.mark.parametrize("issue",["precision","members","rows","order","assignment","CDF","RNG","mass","route","fit","acquire","cumulative","cold_time"])
def test_actual_firstA_counter_or_source_witness_tampering_is_rejected(actual_first_request,issue):
 x=deepcopy(actual_first_request)
 if issue=="precision":x["stage_events"][0]["precision"]["autocast"]=True
 elif issue=="members":x["stage_events"][0]["cache_after"]["parameter_count_by_member"]["0"]=1
 elif issue=="rows":x["stage_events"][0]["proposed_rows"]-=1
 elif issue=="order":x["stage_events"][0]["stage_index"]=1
 elif issue=="assignment":x["stage_events"][1]["sampler"]["head_assignment_counts_by_member"][0]+=1
 elif issue=="CDF":x["cdf_call_count_delta"]-=1
 elif issue=="RNG":x["host_request_accounting"]["counter_delta"]["base"]["model_rng_isolation_count"]-=1
 elif issue=="mass":x["host_request_accounting"]["counter_delta"]["component"]["mass_observed_rows"]-=1
 elif issue=="route":x["host_request_accounting"]["counter_delta"]["routes"]["uncached_calls_by_member"][5]-=1
 elif issue=="fit":x["host_request_accounting"]["counter_delta"]["host"]["actual_prepare_miss_count"]=0
 elif issue=="acquire":x["stage_events"][0]["host_acquire"]["host_counter_delta"]["logical_prepare_count"]=2
 elif issue=="cumulative":x["host_request_accounting"]["counter_end"]["base"]["predict_count"]+=1
 else:x["timings"]["actual_stage_fit_count"]=0
 with pytest.raises(ValueError):h.request_counts(x,visit=0,previous={},ops=None)


def test_synthetic_warm_visit_has_zero_new_fits_but_counts_all_logical_migrations(actual_first_request):
 cold=actual_first_request;previous=cold["host_request_accounting"]["counter_end"];x=deepcopy(cold);x["case"]=1
 for event in x["stage_events"]:
  a=event["host_acquire"];a.update(case=1,logical_acquire_index=a["logical_acquire_index"]+21,
    actual_resident_stage_fit_count=0,retained_stage_hit_count=1,hit_provenance_times_are_not_charged_as_new_work=True)
  a["actual_fit_seconds"]={name:0 for name in a["actual_fit_seconds"]}
  a["host_counter_delta"].update(actual_prepare_miss_count=0,cache_hit_count=1,reload_count=1)
  for key in ("fit_stock_cache_seconds","resident_move_seconds","uncached_fit_seconds"):event[key]=0
 delta=x["host_request_accounting"]["counter_delta"]
 delta["host"].update(actual_prepare_miss_count=0,cache_hit_count=21,reload_count=21)
 delta["base"]["prepare_count"]=0;delta["hybrid"]["uncached_fit_count"]=0
 delta["base"]["model_rng_isolation_count"]=delta["base"]["predict_count"]+delta["hybrid"]["uncached_predict_count"]
 end=h.counters_add(previous,delta);x["host_request_accounting"]["counter_end"]=end;x["host_session_after_request"]["aggregate_counters"]=end
 x["host_request_accounting"]["acquisitions"]=[event["host_acquire"] for event in x["stage_events"]]
 x["timings"].update(actual_stage_fit_count=0,retained_stage_hits=21,cold_request=False)
 _,summary=h.request_counts(x,visit=1,previous=previous,ops=None)
 assert summary["actual_fits"]==0 and summary["logical_acquires"]==summary["retained_hits"]==summary["reloads"]==21
 assert summary["migration_count"]==42
 assert 21+summary["migration_count"]*2==105


def test_incomplete_session_footer_never_qualifies_even_with_50000_files(tmp_path):
 path=tmp_path/"manifest.json";path.write_text(json.dumps({"schema":"posteriorforge.inference.host_session.v1","variant":h.VARIANT,"status":"running"}))
 with pytest.raises(ValueError,match="complete host session"):
  h.authenticate_session(path,context=None,queries={},metrics=None,binder=None,ops=None,host=None)


def test_actual_complete150k_shared_footer_and_sources_when_available():
 phase=Path("/scratch/project/eu-26-17/eubrieucb/posteriorforge/results/performance-first-v1")
 path=phase/"gqa-host-session-aba-01/manifest.json"
 if not path.is_file() or json.loads(path.read_text()).get("status")!="complete":pytest.skip("actual ABA footer not complete")
 from posteriorforge.banks import load_context,load_heldout
 metrics,binder,ops,host=h.load_sources();context=load_context(phase.parent/"banks-v1",20261111,4096)
 queries={case:load_heldout(phase.parent/"banks-v1",case).obs.reshape(1,33) for case in (0,1)}
 sdk=phase.parents[1]/"envs/inference/lib/python3.11/site-packages/tabpfn"
 footer,requests,arrays,result=h.authenticate_session(path,context=context,queries=queries,metrics=metrics,binder=binder,ops=ops,host=host,sdk_root=sdk)
 assert len(requests)==len(arrays)==3 and all(a.shape==(50000,21) for a in arrays)
 assert result["all150000_integrity_passed"] and result["independent_ABA_payload_check"]["bitwise_array_equal"]
 assert result["actual_session_migrations"]==sum(row["migration_count"] for row in result["request_accounting"])
