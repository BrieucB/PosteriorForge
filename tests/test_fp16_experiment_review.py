"""Independent CPU reviewer tests; synthetic engineering checks are not inference."""
from copy import deepcopy
import importlib.util,json,subprocess,sys
from pathlib import Path
import numpy as np
import pytest

ROOT=Path(__file__).parents[1]
spec=importlib.util.spec_from_file_location("fp16_experiment_review_tests",ROOT/"scripts/evaluate_fp16_experiment.py")
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
PHASE=Path("/scratch/project/eu-26-17/eubrieucb/posteriorforge/results/performance-first-v1")


def test_import_sources_and_CLI_are_CPU_only_without_torch():
 subprocess.run([sys.executable,"-c","import importlib.util,sys;s=importlib.util.spec_from_file_location('h','scripts/evaluate_fp16_experiment.py');h=importlib.util.module_from_spec(s);s.loader.exec_module(h);h.sources();assert 'torch' not in sys.modules"],cwd=ROOT,check=True)
 text=subprocess.run([sys.executable,str(Path(h.__file__)),"--help"],check=True,capture_output=True,text=True).stdout
 assert "--approximation" in text and "--sdk-root" in text


@pytest.mark.parametrize("key,value",[
 ("passed",False),("max_normalization_error",1.0001e-5),("max_cdf_absolute_error",2.0001e-5),
 ("max_row_total_variation",1.0001e-4),("minimum_probability",-1e-12),("max_cdf_absolute_error",float("nan"))])
def test_hard_self_numeric_bounds_are_checked_even_if_saved_passed(key,value):
 item={"passed":True,"max_normalization_error":0.,"max_cdf_absolute_error":0.,
  "max_row_total_variation":0.,"minimum_probability":0.}
 h.probability(item);item[key]=value
 with pytest.raises(ValueError):h.probability(item)


def parity_fixture():
 return {"passed":True,"same_whole_batch_logits_borders_uniforms":True,"u_one_not_called":True,"production_sampler_changed":False,
  "points":[{"u":u,"rows":2,"bitwise_equal":True,"nonfinite_existing_arithmetic":2 if u>.5 else 0,
   "outside_physical_borders":2 if u>.5 else 0} for u in (0.,.5,float(np.nextafter(np.float32(1),np.float32(0))))]}


def test_inverse_parity_preserves_original_nonfinite_production_endpoint_counts():
 item=parity_fixture();h.inverse_parity(item,2,members=False)
 assert item["points"][-1]["nonfinite_existing_arithmetic"]==2
 for key,value in (("bitwise_equal",False),("rows",1),("u",1.)):
  x=deepcopy(item);x["points"][-1][key]=value
  with pytest.raises(ValueError):h.inverse_parity(x,2,members=False)
 x=deepcopy(item);x["production_sampler_changed"]=True
 with pytest.raises(ValueError):h.inverse_parity(x,2,members=False)


def test_raw_mass_checks_do_not_trust_passed_and_reject_nonfinite_bounds():
 bound=8*5000*np.finfo(np.float64).eps
 item={"passed":True,"mass_envelope":1.00001,"normalization_absolute":1e-5,"max_normalization_error":0.,
  "raw_head_masses":[{"min":.9999999,"max":1.0000001} for _ in range(8)],
  "actual_component_mass_checks":[{"passed":True,"max_fp64_sum_error":0.,"fp64_roundoff_bound":bound} for _ in range(8)]}
 h.raw_probability(item)
 for key,value in (("min",0.),("max",1.00001001)):
  x=deepcopy(item);x["raw_head_masses"][3][key]=value
  with pytest.raises(ValueError):h.raw_probability(x)
 x=deepcopy(item);x["actual_component_mass_checks"][2]["fp64_roundoff_bound"]=float("inf")
 with pytest.raises(ValueError):h.raw_probability(x)


@pytest.fixture(scope="module")
def actual():
 path=PHASE/"fp16-fifo-50k-01/rep-0/seed-20261111-case-0/manifest.json"
 approximation=PHASE/"fp16-all21-approx-characterization-01/manifest.json"
 if not path.is_file() or not approximation.is_file():pytest.skip("authenticated GPU artifacts not available")
 m=json.loads(path.read_text());q=json.loads(approximation.read_text())
 if m["status"]!="complete":pytest.skip("actual full50000 experiment is not complete")
 return path,m,q


def test_actual_all21_scope_counts_bind_callbacks_QKV_and_F32_fallback(actual):
 _,m,q=actual
 _,precision,_,_,_=h.sources()
 rows=[h.scope_review(scope,event,q["stage_results"][j],precision=precision)
  for j,(scope,event) in enumerate(zip(m["execution_counters"]["completed_stage_scopes"],m["stage_events"]))]
 assert len(rows)==21 and sum(r["actual_heads"] for r in rows)==1141
 assert sum(r["sampler_callbacks"] for r in rows)==1134
 assert rows[0]["actual_heads"]==8 and rows[0]["sampler_callbacks"]==1
 assert rows[10]["original_FP32_fallback_heads"]==7
 assert all(r["original_FP32_fallback_heads"]==0 for j,r in enumerate(rows) if j!=10)


@pytest.mark.parametrize("issue",["alias","AMP","fallback","postprocess","train_only","normalizer_dtype","parameters",
 "physical_heads","query_events","overflow","preprocessor","target","assignments","route","rows"])
def test_actual_stage_counter_dtype_route_tampering_fails_closed(actual,issue):
 _,m,q=actual;j=10
 scope=deepcopy(m["execution_counters"]["completed_stage_scopes"][j]);event=deepcopy(m["stage_events"][j])
 report=scope["precision_session"]
 if issue=="alias":report["aliases_and_flags_restored"]=False
 elif issue=="AMP":next(w for w in report["witnesses"] if w["phase"]=="query" and w["member"]==1)["autocast_enabled"]=False
 elif issue=="fallback":next(w for w in report["witnesses"] if w["phase"]=="query" and w["member"]==5)["projection_dtypes"]["q"]="torch.float16"
 elif issue=="postprocess":scope["hybrid_route_counters"]["postprocessing_math_dtype"]="torch.float16"
 elif issue=="train_only":scope["model"]["contract"]["input_normalization_train_only_verified"]=False
 elif issue=="normalizer_dtype":
  normalizer=next(x for x in scope["model"]["contract"]["modules"] if x["type"]=="InputNormalizationEncoderStep")
  next(iter(normalizer["tensors"].values()))["dtype"]="torch.float16"
 elif issue=="parameters":event["cache_after"]["parameter_dtypes"]=["torch.float16"]
 elif issue=="physical_heads":scope["actual_head_contexts"]-=1
 elif issue=="query_events":next(w for w in report["witnesses"] if w["phase"]=="query")["count"]-=1
 elif issue=="overflow":report["counters"]["witness_histogram_overflow"]=1
 elif issue=="preprocessor":scope["model"]["contract"]["preprocessor_sha256"][0]="0"*64
 elif issue=="target":scope["model"]["target_sd"]*=2
 elif issue=="assignments":event["sampler"]["head_assignment_counts_by_member"][0]+=1
 elif issue=="route":scope["hybrid_route_counters"]["uncached_predict_count"]-=1
 elif issue=="rows":event["proposed_rows"]-=1
 _,precision,_,_,_=h.sources()
 with pytest.raises(ValueError):h.scope_review(scope,event,q["stage_results"][j],precision=precision)


@pytest.mark.parametrize("field,value",[("qualified_fp32",True),("all21_strict_screen_passed",True),
 ("requested_count",49999),("runner_source_sha256","0"*64),("variant","execution_gqa_hybrid_fifo_v1"),
 ("partial_vector_filtering",True),("strict_failed_stages",[])])
def test_new_experiment_header_cannot_be_relabelled_or_benchmark_shrunk(actual,tmp_path,field,value):
 _,m,_=actual;x=deepcopy(m);x[field]=value;path=tmp_path/"manifest.json";path.write_text(json.dumps(x))
 with pytest.raises(ValueError,match="FP16 experimental manifest differs"):
  h.authenticate(path,context=None,query=None,approximation=None,qualification=None,batch_qualification=None,sdk_root=None)


def test_actual_f01_hard_and_strict_replay_rejects_relabel_or_self_failure(actual,monkeypatch):
 _,_,q=actual;producer,precision,_,all21,base=h.sources()
 from types import SimpleNamespace
 monkeypatch.setattr(h,"sha256_file",lambda _:h.ALL21_SHA)
 for issue in ("strict","self","mass","inverse"):
  value=deepcopy(q)
  if issue=="strict":value["strict_failed_stages"]=[]
  elif issue=="self":
   check=next(v for k,v in value["stage_results"][5]["Cp_self_reconstruction"]["1024"].items() if k!="scope")
   check["max_cdf_absolute_error"]=3e-5
  elif issue=="mass":value["stage_results"][5]["probability_validity"]["stress"]["Cp"]["raw_head_masses"][2]["max"]=1.1
  else:value["stage_results"][5]["sampler_parity"]["1024"]["members"][2]["points"][-1]["bitwise_equal"]=False
  fake=SimpleNamespace(validate_all21=lambda *_:value)
  with pytest.raises(ValueError):h.all21_replay(None,{},fake,precision,all21,base)


def test_actual_full50000_source_precision_FIFO_and_CDF_authentication(actual):
 path,m,_=actual;root=PHASE.parents[1]
 context=h.load_context(root/"results/banks-v1",20261111,4096)
 query=h.load_heldout(root/"results/banks-v1",0).obs.reshape(1,33)
 _,values,result=h.authenticate(path,context=context,query=query,
  approximation=PHASE/"fp16-all21-approx-characterization-01/manifest.json",
  qualification=PHASE/"gqa-hybrid-qualification-recovered-01/manifest.json",
  batch_qualification=PHASE/"gqa-large-batch-qualification-01/manifest.json",
  sdk_root=root/"envs/inference/lib/python3.11/site-packages/tabpfn")
 assert values.shape==(50000,21) and result["all50000_integrity_passed"]
 assert result["strict_failed_stages_preserved"]==list(range(21))
 assert result["actual_physical_heads"]==result["corrected_CDF_calls"]==1141
 assert result["new_exact_FP32_or_joint_qualification"] is False
 assert result["source_origin_commit"]==h.PRODUCER_COMMIT
 assert result["recorded_execution_commit"]==m["git_revision"]
 evidence=result["RNG_evidence"]
 assert evidence["production_default_CPU_CUDA_state_hashes_recorded"] is False
 assert evidence["production_default_CPU_CUDA_state_hash_equality_observed"] is False
 assert evidence["characterization_default_CPU_CUDA_RNG_preserved"] is True
 assert evidence["characterization_manifest_sha256"]==h.ALL21_SHA


def test_later_commit_with_identical_producer_bytes_passes_but_changed_source_fails(tmp_path):
 """Version-independent source identity uses real Git history, not fake hashes."""
 repo=tmp_path/"checkout";repo.mkdir()
 def git(*args):
  return subprocess.run(["git","-C",str(repo),*args],check=True,capture_output=True,text=True).stdout.strip()
 git("init","--quiet");git("config","user.email","test@example.invalid");git("config","user.name","CPU proof test")
 source=repo/"src/posteriorforge/performance_fp16_experiment.py";source.parent.mkdir(parents=True)
 source.write_bytes((ROOT/"src/posteriorforge/performance_fp16_experiment.py").read_bytes())
 git("add",".");git("commit","-m","original reviewed producer","--quiet");first=git("rev-parse","HEAD")
 (repo/"reviewer-note.txt").write_text("new independent reviewer; producer unchanged\n")
 git("add",".");git("commit","-m","future source-identical execution","--quiet");future=git("rev-parse","HEAD")
 assert future!=first and first!=h.PRODUCER_COMMIT
 for revision in (first,future):
  result=h.producer_revision(revision,repository=repo)
  assert result["recorded_execution_commit"]==revision
  assert result["source_origin_commit"]==h.PRODUCER_COMMIT
  assert result["recorded_producer_blob_sha256"]==h.PRODUCER_SHA
 tree=git("rev-parse",future+"^{tree}")
 with pytest.raises(ValueError,match="not a Git commit"):h.producer_revision(tree,repository=repo)
 source.write_bytes(source.read_bytes()+b"\n# different producer source\n")
 git("add",".");git("commit","-m","modified producer","--quiet");changed=git("rev-parse","HEAD")
 with pytest.raises(ValueError,match="different FP16 producer bytes"):h.producer_revision(changed,repository=repo)
 for invalid in (None,False,"HEAD",future[:8],"0"*40,future+" --"):
  with pytest.raises(ValueError):h.producer_revision(invalid,repository=repo)
