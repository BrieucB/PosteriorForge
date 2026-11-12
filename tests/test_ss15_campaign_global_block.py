"""Synthetic pending-grid fixtures only; no future scientific cohort is loaded/refreshed."""
from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from posteriorforge.artifacts import sha256_file
from posteriorforge.science.analytical_benchmark_v2 import array_descriptor

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("campaign_joint_wrapper",ROOT/"scripts/refresh_ss15_campaign_global_block.py")
w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)


def desc(shape,dtype,tag):
 return {"shape":shape,"dtype":np.dtype(dtype).str,"sha256":hashlib.sha256(tag.encode()).hexdigest()}


def freeze_fixture(monkeypatch):
 monkeypatch.setattr(w,"PARENT_SOURCE_SHA","a"*64)
 monkeypatch.setitem(w.METHODS["sigma_first"],"qualification_sha256","b"*64)
 monkeypatch.setitem(w.METHODS["sigma_first"],"frozen_source","5"*64)
 monkeypatch.setitem(w.METHODS["sigma_first"],"mask_source",w.SIGMA_ORDER_MASK_SHA)
 monkeypatch.setitem(w.METHODS["sigma_first"],"native_source","6"*64)
 monkeypatch.setitem(w.METHODS["sigma_first"],"native_stage_indices",(14,))
 monkeypatch.setitem(w.METHODS["sigma_first"],"qualifier_source","7"*64)


def values(offset=0.):
 result=np.repeat(np.arange(21,dtype=np.float64)[None]/20,50000,axis=0)
 result+=np.linspace(-.1,.1,50000)[:,None]+offset
 result[:,w.kernel.metadata()["noise_indices"]]=.25
 return result


def owner(seed,case,method):
 order=list(range(21)) if method=="graph_local" else list(w.SIGMA_ORDER)
 system="original_physical_order" if method=="graph_local" else "sigma_first_stage_order"
 raw={"checkpoint_sha256":w.backbone.CHECKPOINT_SHA,
      "scientific_fingerprint":w.kernel.metadata()["scientific_fingerprint"],
      "parameter_order":w.kernel.metadata()["physical_parameter_order"],
      "context_theta":desc([4096,21],"float64","rawtheta"+str(seed)),
      "context_obs":desc([4096,33],"float64","rawobs"+str(seed)),
      "query":desc([1,33],"float64","rawquery"+str(case))}
 physical={"context_theta":desc([4096,21],"float32","theta"+str(seed)),
           "context_obs":desc([4096,15],"float32","obs"+str(seed)),
           "query":desc([1,15],"float32","query"+str(case))}
 model=deepcopy(physical)
 if method=="sigma_first":model["context_theta"]=desc([4096,21],"float32","ordered"+str(seed))
 receipt={"physical_theta_float32":physical["context_theta"],"ordered_theta_float32":model["context_theta"],
          "stage_to_physical_order":order,"ordered_theta_equals_physical_theta_columns_pi":True,
          "global_used_predictors_labels_byte_identical":True,"target_coordinate_system":system}
 return raw,physical,model,receipt


def manifest(data,method="sigma_first",seed=20261112,case=0,first_native=False):
 spec=w.METHODS[method];order=list(range(21)) if method=="graph_local" else list(w.SIGMA_ORDER)
 inverse=[int(i) for i in np.argsort(order)]
 raw,physical,model,receipt=owner(seed,case,method)
 ar,ap,am,at=owner(20261111,0,method)
 scope={"context_seed":seed,"case":case,"repetition":0,"scientific_campaign_method":method,
  "frozen_method_source_sha256":spec["frozen_source"],"masked_adapter_source_sha256":spec["mask_source"],
  "mask_policy_sha256":"c"*64,"stage_to_physical_order":order,"physical_to_stage_order":inverse,
  "samples_in_original_physical_order_before_FIFO_support":True}
 if method=="sigma_first":scope["samples_restored_before_FIFO_support"]=True
 result={"schema":w.PARENT_SCHEMA,"variant":spec["variant"],"status":"complete","method":method,
  "runner_source_sha256":w.PARENT_SOURCE_SHA,"frozen_method_source_sha256":spec["frozen_source"],
  "masked_adapter_source_sha256":spec["mask_source"],"requested_count":50000,"context_seed":seed,"case":case,
  "context_size":4096,"repetition":0,"sources_unchanged":True,"default_rng_preserved":True,
  "completed_stage_indices":list(range(21)),"physical_parameter_order_unchanged":True,
  "model_target_order":"original_physical_order" if method=="graph_local" else "sigma_first_stage_order",
  "learned_target_order_changed":method=="sigma_first","stage_to_physical_order":order,
  "physical_to_stage_order":inverse,"physical_output_mapping":"identity" if method=="graph_local" else "inverse_stage_permutation",
  "samples_in_original_physical_order_before_FIFO_support":True,
  "actual_request_fixed_conditional_screen_performed":False,"clipping_or_repair_performed":False,
  "partial_vector_filtering":False,"raw_scientific_bindings":raw,**raw,"samples":array_descriptor(data),
  "model_inputs":model,"actual_physical_model_inputs":physical,"model_target_descriptors":receipt,
  "qualification_transformed_bindings":{"context_obs_SS15_model":physical["context_obs"],"query_SS15_model":physical["query"]},
  "conditional_anchor":{"context_seed":20261111,"case":0,"method":method,"qualification_sha256":spec["qualification_sha256"],
   "qualifier_source_sha256":spec["qualifier_source"],"qualifies_actual_request":False,"replay":{"passed":True},
   "raw_scientific_bindings":ar,"model_inputs":am,"physical_model_inputs":ap,"model_target_descriptors":at},
  "actual_request_numerical_scope":{"actual_request_fixed_conditional_screen_performed":False,
   "anchor_qualifies_actual_request":False,"posterior_accuracy_claimed":False,
   "same_context_as_anchor":seed==20261111,"same_query_as_anchor":case==0},
  "mask_policy":{"module_sha256":spec["mask_source"],"policy_sha256":"c"*64},
  "proposal_buffer":{"finalized":True,"failed":False,"request_scope":scope}}
 if method=="sigma_first":
  result["samples_restored_to_physical_order_before_FIFO_support"]=True
  mapping={"10":[5],"20":[4]}
  for stage in spec["native_stage_indices"]:mapping[str(stage)]=list(range(8))
  policy={"schema":"posteriorforge.execution.ss15_sigma_native_policy.v1",
   "module_sha256":spec["native_source"],"policy_sha256":"8"*64,
   "native_stage_indices":list(spec["native_stage_indices"]),"active_fallback_map":mapping,
   "active_fallback_stage_member_bindings":[{"stage":int(j),"member":m,"target_physical_index":w.SIGMA_ORDER[int(j)],
    "target_physical_name":w.kernel.metadata()["physical_parameter_order"][w.SIGMA_ORDER[int(j)]]}
    for j,members in mapping.items() for m in members],"whole_stage_all8_native_rule":True,
   "scientific_inputs_order_sampler_and_global_kernel_changed":False,
   "ancestral_sources":{"ordered_producer":w.SIGMA_ANCESTRAL_PRODUCER_SHA},
   "order_mask_ancestry":{"module_sha256":w.SIGMA_ORDER_MASK_SHA}}
  result.update(active_native_execution_policy=policy,native_execution_adapter_source_sha256=spec["native_source"],
   frozen_sigma_first_source_sha256=w.SIGMA_ANCESTRAL_PRODUCER_SHA,order_mask_source_sha256=w.SIGMA_ORDER_MASK_SHA)
 if first_native:
  result.update(schema=w.NATIVE_FIRST_SCHEMA,variant=w.NATIVE_FIRST_VARIANT,
   runner_source_sha256=spec["frozen_source"],matching_conditional_qualification_sha256=spec["qualification_sha256"],
   actual_sigma_first_fixed_conditional_screen_performed=False,fresh_same_policy_all21_screen_replayed=True,
   fresh_screen_scope="performed separately; same case/policy finite prefixes; this request only CPU-replays its artifact")
  result.pop("method")
 return result


@pytest.mark.parametrize("method",["graph_local","sigma_first"])
@pytest.mark.parametrize("seed,case",[(20261111,1),(20261112,0),(20261113,7)])
def test_distinct_actual_and_anchor_owners_and_both_fixed_order_families_validate(monkeypatch,method,seed,case):
 freeze_fixture(monkeypatch);data=values();m=manifest(data,method,seed,case)
 receipt=w.validate_parent(m,data,w.kernel.metadata())
 assert receipt["actual_and_anchor_owners_validated_separately"]
 assert receipt["method"]==method and receipt["context_seed"]==seed and receipt["case"]==case
 if seed!=20261111:
  assert m["actual_physical_model_inputs"]["context_theta"]!=m["conditional_anchor"]["physical_model_inputs"]["context_theta"]
 assert len(w.ALLOWED_GRID["graph_local"])==23 and (20261111,0) not in w.ALLOWED_GRID["graph_local"]
 assert len(w.ALLOWED_GRID["sigma_first"])==24 and (20261111,0) in w.ALLOWED_GRID["sigma_first"]


@pytest.mark.parametrize("fault",["seed","case","schema","variant","methodsource","masksource","quota","order",
 "restoration","actualusesanchor","anchorusesactual","scope","transformedusesanchor","claimedqualification","FIFO","float32","support"])
def test_grid_source_order_and_cross_owner_misbindings_are_rejected(monkeypatch,fault):
 freeze_fixture(monkeypatch);data=values();m=manifest(data)
 if fault=="seed":m["context_seed"]=999
 elif fault=="case":m["case"]=8
 elif fault=="schema":m["schema"]="posteriorforge.ss15_global_block_refresh.v1"
 elif fault=="variant":m["variant"]=w.METHODS["graph_local"]["variant"]
 elif fault=="methodsource":m["frozen_method_source_sha256"]="0"*64
 elif fault=="masksource":m["masked_adapter_source_sha256"]="0"*64
 elif fault=="quota":m["requested_count"]=49999
 elif fault=="order":m["stage_to_physical_order"]=list(range(21))
 elif fault=="restoration":m["samples_restored_to_physical_order_before_FIFO_support"]=False
 elif fault=="actualusesanchor":m["actual_physical_model_inputs"]=deepcopy(m["conditional_anchor"]["physical_model_inputs"])
 elif fault=="anchorusesactual":m["conditional_anchor"]["physical_model_inputs"]=deepcopy(m["actual_physical_model_inputs"])
 elif fault=="scope":m["actual_request_numerical_scope"]["same_context_as_anchor"]=True
 elif fault=="transformedusesanchor":m["qualification_transformed_bindings"]["context_obs_SS15_model"]=m["conditional_anchor"]["physical_model_inputs"]["context_obs"]
 elif fault=="claimedqualification":m["conditional_anchor"]["qualifies_actual_request"]=True
 elif fault=="FIFO":m["proposal_buffer"]["request_scope"]["samples_restored_before_FIFO_support"]=False
 elif fault=="float32":data=data.astype(np.float32);m["samples"]=array_descriptor(data)
 else:data[0,6]=.001;m["samples"]=array_descriptor(data)
 with pytest.raises(ValueError):w.validate_parent(m,data,w.kernel.metadata())


def test_graph_original_physical_targets_cannot_be_relabelled_as_ordered(monkeypatch):
 freeze_fixture(monkeypatch);data=values();m=manifest(data,"graph_local")
 altered=desc([4096,21],"float32","not-physical")
 m["model_inputs"]["context_theta"]=altered;m["model_target_descriptors"]["ordered_theta_float32"]=altered
 with pytest.raises(ValueError,match="graph target"):w.validate_parent(m,data,w.kernel.metadata())


def test_new_source_or_missing_method_anchor_prevents_execution_before_parent_or_output(tmp_path,monkeypatch):
 monkeypatch.setattr(w,"PARENT_SOURCE_SHA",None)
 monkeypatch.setitem(w.METHODS["sigma_first"],"frozen_source",None)
 with pytest.raises(ValueError,match="not frozen"):w.run(tmp_path/"missing","0"*64,tmp_path/"out")
 assert not (tmp_path/"out").exists()
 monkeypatch.setattr(w,"PARENT_SOURCE_SHA","a"*64)
 monkeypatch.setitem(w.METHODS["sigma_first"],"qualification_sha256",None)
 with pytest.raises(ValueError,match="not frozen"):w.source_guard()


def test_paired_methods_delegate_exact_frozen_refresh_once_with_actual_seed_case_role(tmp_path,monkeypatch):
 freeze_fixture(monkeypatch);receipts=[];calls=[];original=w.refresh_once
 monkeypatch.setattr(w,"source_guard",lambda **kwargs:{"fixture":"not-scientific-evidence"})
 def spy(data,seed):calls.append(seed);return original(data,seed)
 monkeypatch.setattr(w,"refresh_once",spy)
 for index,method in enumerate(("graph_local","sigma_first")):
  folder=tmp_path/method;folder.mkdir();data=values(index*.03);before=data.copy()
  mp=folder/"parent.json";mp.write_text('{"engineering_fixture":true}')
  with (folder/"samples.npz").open("xb") as stream:np.savez(stream,samples=data)
  expected=sha256_file(mp);parent=manifest(data,method,20261112,3)
  lineage={"samples_sha256":sha256_file(folder/"samples.npz"),"manifest_sha256":expected,"method":method}
  monkeypatch.setattr(w,"load_parent",lambda *args,p=parent,v=data,l=lineage:(p,v,l))
  result=w.run(mp,expected,folder/"out");receipts.append(result["refresh"])
  with np.load(folder/"out/samples.npz",allow_pickle=False) as z:out=z["samples"]
  assert out[:,3:].tobytes()==before[:,3:].tobytes() and data.tobytes()==before.tobytes()
  assert result["method"]==method and result["context_seed"]==20261112 and result["case"]==3
  assert not result["observations_references_truth_or_banks_loaded"]
 assert calls==[w.backbone.derive_seed(20261112,w.ROLE,case=3,repetition=0)]*2
 assert receipts[0]["standard_normal_draws"]==receipts[1]["standard_normal_draws"]
 assert receipts[0]["initial_private_state"]==receipts[1]["initial_private_state"]


def test_math_alias_and_torch_free_import_cli_have_no_case_override_or_data_inputs():
 w.require_backbone();assert w.refresh_once is w.backbone.refresh_once
 assert w.kernel.source_sha256()==w.KERNEL_SHA
 source=ROOT/"scripts/refresh_ss15_campaign_global_block.py"
 code=("import importlib.util,sys;s=importlib.util.spec_from_file_location('r',"+repr(str(source))+");"
       "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.require_backbone();assert 'torch' not in sys.modules")
 subprocess.run([sys.executable,"-c",code],check=True)
 result=subprocess.run([sys.executable,str(source),"--help"],check=True,capture_output=True,text=True)
 assert "--parent-manifest" in result.stdout and "--expected-parent-sha256" in result.stdout
 assert all("--"+name not in result.stdout for name in ("case","seed","reference","observations","banks","truth"))


def test_corrected_sigma_requires_new_case0_but_old_graph_case0_is_reused(monkeypatch):
 freeze_fixture(monkeypatch);data=values()
 sigma=manifest(data,"sigma_first",20261111,0,first_native=True)
 assert w.validate_parent(sigma,data,w.kernel.metadata())["method"]=="sigma_first"
 graph=manifest(data,"graph_local",20261111,0)
 with pytest.raises(ValueError,match="completed case0"):w.validate_parent(graph,data,w.kernel.metadata())


def test_unqualified_sigma_prototype_and_partial_native_overrides_are_rejected(monkeypatch):
 freeze_fixture(monkeypatch);data=values();m=manifest(data,"sigma_first",20261111,0,first_native=True)
 bad=deepcopy(m);bad["schema"]="posteriorforge.inference.ss15_sigma_first.v1"
 with pytest.raises(ValueError):w.validate_parent(bad,data,w.kernel.metadata())
 bad=deepcopy(m);bad["runner_source_sha256"]=w.SIGMA_ANCESTRAL_PRODUCER_SHA
 with pytest.raises(ValueError):w.validate_parent(bad,data,w.kernel.metadata())
 bad=deepcopy(m);bad["active_native_execution_policy"]["active_fallback_map"]["14"]=[1,2,4,5,6]
 with pytest.raises(ValueError,match="all8"):w.validate_parent(bad,data,w.kernel.metadata())
