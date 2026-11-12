#!/usr/bin/env python3
"""One frozen joint-global refresh for a pending fixed scientific-grid cohort.

Built with PriorLabs-TabPFN. Only parent provenance/routing changes here:
a995.refresh_once and the e346 prior-only kernel/ROLE stay unchanged. Actual
training descriptors and the finite case0 anchor are distinct owners. No
observation/reference/truth/bank/model or local-likelihood data is loaded.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import subprocess
import numpy as np
from posteriorforge.artifacts import runtime_identity,sha256_file,write_json
from posteriorforge.science.analytical_benchmark_v2 import array_descriptor

SCHEMA="posteriorforge.ss15_campaign_global_block_refresh.v1"
PARENT_SCHEMA="posteriorforge.inference.ss15_scientific_campaign.v1"
PARENT_SOURCE_PATH="src/posteriorforge/performance_ss15_scientific_campaign.py"
PARENT_SOURCE_SHA="3965e36206d7f750c66350df54b6d855d6c9f8f4f81a0e069aeeb478492cd453"
NATIVE_FIRST_SCHEMA="posteriorforge.inference.ss15_sigma_native.v1"
NATIVE_FIRST_VARIANT="accuracy_ss15_sigma_first_native_fp16_fifo_v1"
SIGMA_ANCESTRAL_PRODUCER_SHA="128c200a82f1252b2ad971ed43f10f534b16b9e02f8882cfd62d222cb88442cd"
SIGMA_ORDER_MASK_SHA="d48d1638ef9e540bdd629d321fba657af4db757700b1092fc58fb9ded1f79ad2"
BACKBONE_SHA="a995ab5e48fab6ddea8355344d9f77fe94bd30357d6451370daf69ed5693f8a4"
KERNEL_SHA="e346e25bc9825abf016c2bd75addaf2b025d3e3d94e1a0ea6f6e31dd02997a43"
CONTRACT_PATH="docs/accuracy-ss15-sigma-first-v1.md"
CONTRACT_SHA="5b7a31237c9a3e35db27f3f9697c7a9293b0738a8ec8b12d63f370d4ad27e3d4"
CONTRACT_ORIGIN="f9437ee"
COUNT=50000
ROLE="ss15_global_block_gibbs_once_v1"
CONTEXT_SEEDS=(20261111,20261112,20261113)
CASES=tuple(range(8))
ANCHOR=(20261111,0)
PREDEFINED_GRID=tuple((seed,case) for seed in CONTEXT_SEEDS for case in CASES)
GRAPH_PENDING_GRID=tuple(point for point in PREDEFINED_GRID if point!=ANCHOR)
ALLOWED_GRID={"graph_local":GRAPH_PENDING_GRID,"sigma_first":PREDEFINED_GRID}
SIGMA_ORDER=(0,1,2,6,3,4,5,10,7,8,9,14,11,12,13,16,15,18,17,20,19)
METHODS={
 "graph_local":{
  "variant":"accuracy_ss15_graph_local_fp16_fifo_scientific_campaign_v1",
  "hybrid_variant":"hybrid_ss15_graph_local_global_block_gibbs_scientific_campaign_v1",
  "frozen_source":"c2f20a8dbf1f4e2bf48294be038962d0c3e4e00ec8b4a7cab29ea259d91d1818",
  "frozen_path":"src/posteriorforge/performance_ss15_graph.py",
  "mask_source":"dfa096c314fdc3855d0a45a4bcb3f9858675747edc2403f917e32e19837eb110",
  "mask_path":"src/posteriorforge/execution_ss15_graph.py",
  "qualification_sha256":"d90604c99664e6c977ddf59c0f72adbb44fc520594ab36d64d6f25841a3985e3",
  "qualifier_source":"3f5ee07947418bdc6877500e06f56c5af3dfa9b31d6a6208313aa25eb05c38b2"},
 "sigma_first":{
  "variant":"accuracy_ss15_sigma_first_native_fp16_fifo_scientific_campaign_v1",
  "hybrid_variant":"hybrid_ss15_sigma_first_native_global_block_gibbs_scientific_campaign_v1",
  "frozen_source":"e322d56492476be1acf5fcdb9afd30b57e2a69a373098417a2bf8944b51ba73b",
  "frozen_path":"src/posteriorforge/performance_ss15_sigma_native.py",
  "mask_source":SIGMA_ORDER_MASK_SHA,
  "mask_path":"src/posteriorforge/execution_ss15_sigma_first.py",
  "native_source":"cf135637d073439039cea3e82a98e533cd161b095aac488ea92588aacb9a1461",
  "native_path":"src/posteriorforge/execution_ss15_sigma_native_policy.py",
  "native_stage_indices":(14,),
  "qualification_sha256":"bedfa67dc8a1e7f41a8536e4c8091643dc05d309b0da0367dc92136c19603705",
  "qualifier_source":"8c1b7d432888d83eeb4ce07fa8b7f18d3744ee08f00091b1204d1e33ff747334"}}


def _load_backbone():
 path=Path(__file__).with_name("refresh_ss15_global_block.py")
 if sha256_file(path)!=BACKBONE_SHA:raise ValueError("frozen a995 refresh producer changed")
 spec=importlib.util.spec_from_file_location("_fixed_campaign_joint_backbone",path)
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


backbone=_load_backbone()
kernel=backbone.kernel
require=backbone.require
refresh_once=backbone.refresh_once
_SEALS={name:(getattr(backbone,name),getattr(backbone,name).__code__)
 for name in ("refresh_once","source_guard","git_bytes","state_sha","json_sha")}


def require_backbone():
 require(sha256_file(backbone.__file__)==BACKBONE_SHA and kernel.source_sha256()==KERNEL_SHA
  and backbone.KERNEL_SHA==KERNEL_SHA and backbone.ROLE==ROLE and backbone.COUNT==COUNT,
  "frozen joint source/kernel/ROLE changed")
 for name,(function,code) in _SEALS.items():
  require(getattr(backbone,name) is function and function.__code__ is code,"frozen callable changed: "+name)
 require(refresh_once is backbone.refresh_once,"joint refresh alias replaced")


def source_guard(*,first_native=False):
 require_backbone()
 require(_ready_method("sigma_first") if first_native else backbone.digest(PARENT_SOURCE_SHA)
  and all(_ready_method(method) for method in METHODS),
  "corrected native/campaign producer or anchor source is not frozen")
 old=backbone.source_guard();root=Path(__file__).resolve().parents[1];source=Path(__file__).resolve()
 require(sha256_file(root/CONTRACT_PATH)==CONTRACT_SHA,"fixed two-method grid contract changed")
 origin=subprocess.run(["git","-C",str(root),"rev-parse",CONTRACT_ORIGIN],
  check=True,capture_output=True,text=True).stdout.strip()
 require(hashlib.sha256(backbone.git_bytes(root,origin,CONTRACT_PATH)).hexdigest()==CONTRACT_SHA,
  "pre-campaign contract Git bytes differ")
 spec=METHODS["sigma_first"]
 code=[(str(source.relative_to(root)),sha256_file(source)),
       (spec["frozen_path"],spec["frozen_source"]),(spec["native_path"],spec["native_source"])]
 if not first_native:code.append((PARENT_SOURCE_PATH,PARENT_SOURCE_SHA))
 for filename,digest in code:
  require(hashlib.sha256(backbone.git_bytes(root,old["git_revision"],filename)).hexdigest()==digest,
   "wrapper/native/campaign sources must be committed before refresh")
 return {"producer_sha256":sha256_file(source),"git_revision":old["git_revision"],
  "frozen_joint_refresh_sha256":BACKBONE_SHA,"kernel_sha256":KERNEL_SHA,"joint_source":old,
  "first_corrected_native_case0":first_native,
  "campaign_parent_producer_sha256":PARENT_SOURCE_SHA if not first_native else None,
  "native_parent_producer_sha256":spec["frozen_source"],"native_execution_source_sha256":spec["native_source"],
  "grid_contract_sha256":CONTRACT_SHA,"grid_contract_origin_commit":origin,"sources_committed_before_run":True}


def _descriptor(value,shape,dtype):
 if not(isinstance(value,dict) and value.get("shape")==shape
        and isinstance(value.get("dtype"),str) and backbone.digest(value.get("sha256"))):return False
 try:return np.dtype(value["dtype"])==np.dtype(dtype)
 except (TypeError,ValueError):return False


def _raw(value,metadata):
 require(set(value)=={"checkpoint_sha256","scientific_fingerprint","parameter_order",
                      "context_theta","context_obs","query"},"all six raw scientific descriptors required")
 require(value["checkpoint_sha256"]==backbone.CHECKPOINT_SHA
  and value["scientific_fingerprint"]==metadata["scientific_fingerprint"]
  and value["parameter_order"]==metadata["physical_parameter_order"],"physical problem/checkpoint/order differs")
 for name,shape in (("context_theta",[4096,21]),("context_obs",[4096,33]),("query",[1,33])):
  require(_descriptor(value[name],shape,"float64"),"raw descriptor differs: "+name)


def _models(model,physical,receipt,order,system):
 for name,shape in (("context_theta",[4096,21]),("context_obs",[4096,15]),("query",[1,15])):
  require(_descriptor(model.get(name),shape,"float32") and _descriptor(physical.get(name),shape,"float32"),
   "actual/anchor model descriptor differs: "+name)
 require(model["context_obs"]==physical["context_obs"] and model["query"]==physical["query"],
  "target permutation must not alter its owner's observation inputs")
 require(receipt.get("physical_theta_float32")==physical["context_theta"]
  and receipt.get("ordered_theta_float32")==model["context_theta"]
  and receipt.get("stage_to_physical_order")==order
  and receipt.get("ordered_theta_equals_physical_theta_columns_pi") is True
  and receipt.get("global_used_predictors_labels_byte_identical") is True
  and receipt.get("target_coordinate_system")==system,"source-bound target-order receipt differs")


def _ready_method(method):
 spec=METHODS[method]
 keys=("frozen_source","mask_source","qualifier_source","qualification_sha256")
 if method=="sigma_first":keys+=("native_source",)
 return (all(backbone.digest(spec.get(k)) for k in keys)
  and (method!="sigma_first" or isinstance(spec.get("native_stage_indices"),(list,tuple))
       and spec["native_stage_indices"] and 14 in spec["native_stage_indices"]
       and len(set(spec["native_stage_indices"]))==len(spec["native_stage_indices"])
       and all(type(j) is int and 3<=j<21 for j in spec["native_stage_indices"])))


def _native_policy(manifest,metadata):
 spec=METHODS["sigma_first"];policy=manifest.get("active_native_execution_policy",{})
 require(_ready_method("sigma_first"),"corrected sigma policy/anchor is not frozen")
 require(policy.get("schema")=="posteriorforge.execution.ss15_sigma_native_policy.v1"
  and policy.get("module_sha256")==spec["native_source"] and backbone.digest(policy.get("policy_sha256"))
  and policy.get("native_stage_indices")==list(spec["native_stage_indices"])
  and policy.get("whole_stage_all8_native_rule") is True
  and policy.get("scientific_inputs_order_sampler_and_global_kernel_changed") is False,
  "qualified whole-stage native policy differs")
 mapping={"10":[5],"20":[4]}
 for stage in spec["native_stage_indices"]:mapping[str(stage)]=list(range(8))
 require(policy.get("active_fallback_map")==mapping,"all8 native stage map differs")
 observed=policy.get("active_fallback_stage_member_bindings",[])
 pairs={(int(stage),member) for stage,members in mapping.items() for member in members}
 require(len(observed)==len(pairs) and {(r["stage"],r["member"]) for r in observed}==pairs
  and all(r.get("target_physical_index")==SIGMA_ORDER[r["stage"]]
      and r.get("target_physical_name")==metadata["physical_parameter_order"][SIGMA_ORDER[r["stage"]]]
      for r in observed),"native stage/member physical identities differ")
 require(policy.get("ancestral_sources",{}).get("ordered_producer")==SIGMA_ANCESTRAL_PRODUCER_SHA
  and policy.get("order_mask_ancestry",{}).get("module_sha256")==SIGMA_ORDER_MASK_SHA
  and manifest.get("native_execution_adapter_source_sha256")==spec["native_source"]
  and manifest.get("frozen_sigma_first_source_sha256")==SIGMA_ANCESTRAL_PRODUCER_SHA
  and manifest.get("order_mask_source_sha256")==SIGMA_ORDER_MASK_SHA,
  "order ancestry was conflated with corrected execution eligibility")
 return policy


def validate_parent(manifest,values,metadata):
 first_native=manifest.get("schema")==NATIVE_FIRST_SCHEMA
 method="sigma_first" if first_native else manifest.get("method")
 require(method in METHODS,"fixed graph_local|corrected sigma_first parent required")
 require(_ready_method(method),"method execution policy/finite anchor is not frozen")
 if not first_native:require(backbone.digest(PARENT_SOURCE_SHA),"campaign producer source is not frozen")
 spec=METHODS[method];seed,case=manifest.get("context_seed"),manifest.get("case")
 allowed=(seed,case)==ANCHOR if first_native else (seed,case) in GRAPH_PENDING_GRID
 require(type(seed) is int and type(case) is int and allowed,
  "new corrected sigma firstcase0 or remaining fixed grid only; completed case0 must be reused")
 order=list(range(21)) if method=="graph_local" else list(SIGMA_ORDER)
 inverse=[int(i) for i in np.argsort(order)]
 system="original_physical_order" if method=="graph_local" else "sigma_first_stage_order"
 expected={"schema":PARENT_SCHEMA,"variant":spec["variant"],"status":"complete",
  "runner_source_sha256":PARENT_SOURCE_SHA,"frozen_method_source_sha256":spec["frozen_source"],
  "masked_adapter_source_sha256":spec["mask_source"],"requested_count":COUNT,
  "context_size":4096,"repetition":0,"sources_unchanged":True,"default_rng_preserved":True,
  "scientific_fingerprint":metadata["scientific_fingerprint"],"parameter_order":metadata["physical_parameter_order"],
  "completed_stage_indices":list(range(21)),"physical_parameter_order_unchanged":True,
  "model_target_order":system,"learned_target_order_changed":method=="sigma_first",
  "stage_to_physical_order":order,"physical_to_stage_order":inverse,
  "physical_output_mapping":"identity" if method=="graph_local" else "inverse_stage_permutation",
  "samples_in_original_physical_order_before_FIFO_support":True,
  "actual_request_fixed_conditional_screen_performed":False,
  "clipping_or_repair_performed":False,"partial_vector_filtering":False}
 if first_native:
  for name in ("frozen_method_source_sha256","masked_adapter_source_sha256","physical_output_mapping",
               "samples_in_original_physical_order_before_FIFO_support","actual_request_fixed_conditional_screen_performed"):
   expected.pop(name)
  expected.update(schema=NATIVE_FIRST_SCHEMA,variant=NATIVE_FIRST_VARIANT,
   runner_source_sha256=spec["frozen_source"],matching_conditional_qualification_sha256=spec["qualification_sha256"],
   actual_sigma_first_fixed_conditional_screen_performed=False,fresh_same_policy_all21_screen_replayed=True,
   fresh_screen_scope="performed separately; same case/policy finite prefixes; this request only CPU-replays its artifact")
 for key,value in expected.items():
  require(type(manifest.get(key)) is type(value) and manifest[key]==value,"campaign parent differs: "+key)
 require(isinstance(values,np.ndarray) and values.dtype==np.dtype("float64")
  and values.shape==(COUNT,21) and array_descriptor(values)==manifest.get("samples"),
  "strict50000 complete physical Float64 cohort required")
 raw=manifest.get("raw_scientific_bindings",{});_raw(raw,metadata)
 require(all(manifest.get(k)==v for k,v in raw.items()),"actual raw bindings differ from top physical provenance")
 _models(manifest.get("model_inputs",{}),manifest.get("actual_physical_model_inputs",{}),
         manifest.get("model_target_descriptors",{}),order,system)
 if method=="graph_local":
  require(manifest["model_inputs"]==manifest["actual_physical_model_inputs"],"graph target inputs must remain physical")
 else:
  require(manifest.get("samples_restored_to_physical_order_before_FIFO_support") is True,"sigma outputs not restored")
  _native_policy(manifest,metadata)
 transformed=manifest.get("qualification_transformed_bindings",{})
 require(transformed.get("context_obs_SS15_model")==manifest["actual_physical_model_inputs"]["context_obs"]
  and transformed.get("query_SS15_model")==manifest["actual_physical_model_inputs"]["query"],
  "actual transformed descriptors were replaced by finite anchor inputs")
 if not first_native:
  anchor=manifest.get("conditional_anchor",{})
  require(anchor.get("context_seed")==ANCHOR[0] and anchor.get("case")==ANCHOR[1] and anchor.get("method")==method
   and anchor.get("qualification_sha256")==spec["qualification_sha256"]
   and anchor.get("qualifier_source_sha256")==spec["qualifier_source"]
   and anchor.get("qualifies_actual_request") is False and anchor.get("replay",{}).get("passed") is True,
   "distinct finite case0 anchor source/scope differs")
  _raw(anchor.get("raw_scientific_bindings",{}),metadata)
  _models(anchor.get("model_inputs",{}),anchor.get("physical_model_inputs",{}),
          anchor.get("model_target_descriptors",{}),order,system)
  scope=manifest.get("actual_request_numerical_scope",{})
  require(scope.get("actual_request_fixed_conditional_screen_performed") is False
   and scope.get("anchor_qualifies_actual_request") is False and scope.get("posterior_accuracy_claimed") is False
   and scope.get("same_context_as_anchor")==all(raw[k]==anchor["raw_scientific_bindings"][k] for k in ("context_theta","context_obs"))
   and scope.get("same_query_as_anchor")==(raw["query"]==anchor["raw_scientific_bindings"]["query"]),
   "actual-request/finite-anchor owners were conflated")
  policy=manifest.get("mask_policy",{})
  require(policy.get("module_sha256")==spec["mask_source"] and backbone.digest(policy.get("policy_sha256")),
   "fixed mask source/policy differs")
 else:policy={"policy_sha256":None}
 fifo=manifest.get("proposal_buffer",{});fs=fifo.get("request_scope",{})
 required={"context_seed":seed,"case":case,"repetition":0,"scientific_campaign_method":method,
  "frozen_method_source_sha256":spec["frozen_source"],"masked_adapter_source_sha256":spec["mask_source"],
  "mask_policy_sha256":policy["policy_sha256"],"stage_to_physical_order":order,
  "physical_to_stage_order":inverse,"samples_in_original_physical_order_before_FIFO_support":True}
 if first_native:
  for name in ("scientific_campaign_method","frozen_method_source_sha256","masked_adapter_source_sha256",
               "mask_policy_sha256","samples_in_original_physical_order_before_FIFO_support"):
   required.pop(name)
 require(fifo.get("finalized") is True and fifo.get("failed") is False
  and all(fs.get(k)==v for k,v in required.items()),"original physical FIFO/support scope differs")
 if method=="sigma_first":require(fs.get("samples_restored_before_FIFO_support") is True,"sigma FIFO restoration not recorded")
 require(metadata["target_indices"]==[0,1,2] and metadata["unchanged_indices"]==list(range(3,21)),
  "unchanged3-global kernel required")
 kernel.conditional_parameters(values)
 return {"method":method,"context_seed":seed,"case":case,"stage_to_physical_order":order,
  "physical_to_stage_order":inverse,"first_corrected_native_case0":first_native,"actual_and_anchor_owners_validated_separately":not first_native,
  "training_target_receipts_scope":"source-bound assertions; no training-bank arrays loaded"}


def load_parent(path,expected_sha256,metadata):
 require(backbone.digest(expected_sha256),"explicit expected parent manifest SHA256 required")
 path=Path(path).resolve();raw=path.read_bytes()
 require(hashlib.sha256(raw).hexdigest()==expected_sha256,"parent manifest hash differs")
 manifest=json.loads(raw);require(manifest.get("samples_file")=="samples.npz","fixed samples filename required")
 payload=path.with_name("samples.npz")
 require(backbone.digest(manifest.get("samples_sha256")) and sha256_file(payload)==manifest["samples_sha256"],
  "parent payload hash differs")
 with np.load(payload,allow_pickle=False) as archive:
  require(archive.files==["samples"],"complete physical samples only");values=archive["samples"]
 receipt=validate_parent(manifest,values,metadata)
 spec=METHODS[receipt["method"]];root=Path(__file__).resolve().parents[1];revision=manifest.get("git_revision")
 code=[(spec["frozen_path"],spec["frozen_source"]),(spec["mask_path"],spec["mask_source"]),(CONTRACT_PATH,CONTRACT_SHA)]
 if not receipt["first_corrected_native_case0"]:code.append((PARENT_SOURCE_PATH,PARENT_SOURCE_SHA))
 if receipt["method"]=="sigma_first":code.append((spec["native_path"],spec["native_source"]))
 for name,checksum in code:
  require(hashlib.sha256(backbone.git_bytes(root,revision,name)).hexdigest()==checksum,
   "executed campaign/frozen family/mask/contract Git bytes differ")
 return manifest,values,{"manifest_sha256":expected_sha256,"samples_sha256":manifest["samples_sha256"],
  "samples":array_descriptor(values),"producer_sha256":manifest["runner_source_sha256"],"git_revision":revision,
  "method":receipt["method"],"frozen_method_source_sha256":spec["frozen_source"],
  "masked_adapter_source_sha256":spec["mask_source"],"input_order_validation":receipt}


def run(parent_manifest,expected_parent_sha256,output):
 before=backbone.state_sha((np.random.get_state(),random.getstate()))
 require(_ready_method("sigma_first"),"corrected native policy/anchor source is not frozen")
 require(backbone.digest(expected_parent_sha256),"explicit expected parent SHA required")
 header_raw=Path(parent_manifest).read_bytes()
 require(hashlib.sha256(header_raw).hexdigest()==expected_parent_sha256,"parent manifest hash differs")
 first_native=json.loads(header_raw).get("schema")==NATIVE_FIRST_SCHEMA
 sources=source_guard(first_native=first_native);metadata=kernel.metadata()
 parent,values,lineage=load_parent(parent_manifest,expected_parent_sha256,metadata)
 method=lineage["method"];output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
 record={"schema":SCHEMA,"variant":METHODS[method]["hybrid_variant"],"method":method,"status":"running",
  "attribution":"Built with PriorLabs-TabPFN","requested_count":COUNT,"context_seed":parent["context_seed"],
  "case":parent["case"],"repetition":0,"parameter_order":metadata["physical_parameter_order"],
  "scientific_fingerprint":metadata["scientific_fingerprint"],"parent":lineage,"source_pins":sources,
  "conditional_kernel":metadata,"runtime":runtime_identity(),"raw_scientific_bindings":parent["raw_scientific_bindings"],
  "raw_scientific_bindings_scope":"inherited provenance only; no bound data files loaded",
  "method_description":"fixed neural method plus ONE unchanged correlated3-global Gibbs block",
  "pure_TFM_inference":False,"posterior_accuracy_guaranteed":False,
  "observations_references_truth_or_banks_loaded":False,"noise_or_local_likelihood_update_performed":False,
  "support_filtering_clipping_redraw_or_trimming":False,"end_to_end_posterior_request_timing_claimed":False}
 write_json(output/"manifest.json",record)
 try:
  seed=backbone.derive_seed(parent["context_seed"],ROLE,case=parent["case"],repetition=0)
  result,receipt=refresh_once(values,seed)
  require(receipt["role"]==ROLE and receipt["standard_normal_draw_shape"]==[COUNT,3],"frozen joint rule changed")
  with (output/"samples.npz").open("xb") as stream:np.savez(stream,samples=result)
  require(source_guard(first_native=first_native)==sources,"source/contract changed during refresh")
  require(sha256_file(parent_manifest)==expected_parent_sha256
   and sha256_file(Path(parent_manifest).with_name("samples.npz"))==lineage["samples_sha256"],"parent changed")
  after=backbone.state_sha((np.random.get_state(),random.getstate()))
  require(before==after,"wrapper changed default NumPy/Python RNG")
  record.update(status="complete",samples=array_descriptor(result),samples_file="samples.npz",
   samples_sha256=sha256_file(output/"samples.npz"),refresh=receipt,sources_unchanged=True,
   default_RNG_observation={"before_sha256":before,"after_sha256":after,"preserved":True,
    "observed_generators":["NumPy legacy default","Python random"],"torch_CPU_CUDA_RNG_hashes_observed":False})
 except BaseException as error:
  record.update(status="failed",error_type=type(error).__name__,error=str(error));write_json(output/"manifest.json",record);raise
 write_json(output/"manifest.json",record)
 write_json(output/"checksums.json",{"manifest.json":sha256_file(output/"manifest.json"),"samples.npz":record["samples_sha256"]})
 return record


def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument("--parent-manifest",type=Path,required=True)
 parser.add_argument("--expected-parent-sha256",required=True)
 parser.add_argument("--output",type=Path,required=True)
 args=parser.parse_args();m=run(args.parent_manifest,args.expected_parent_sha256,args.output)
 print(json.dumps({k:m.get(k) for k in ("schema","variant","status","method","context_seed","case","samples_sha256")},sort_keys=True))


if __name__=="__main__":
 main()
