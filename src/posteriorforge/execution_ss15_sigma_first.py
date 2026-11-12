"""One fixed sigma-first order inside all6 SS15 graph-local blocks.

Stages0..2 keep their original used predictors and target labels. Every later
block orders sigma first, then a,b,d0 (mechanics) or a (acoustics). Complete
stage-ordered proposals must be restored to physical order before support.

Built with PriorLabs-TabPFN. All3 globals remain in every local factor. Given
those globals, six complete local/noise blocks are posterior independent.
This changes learned inputs and conditional target order. The saved output
returns to the original physical order; likelihood and sampler stay unchanged.
No earlier unmasked numerical screen qualifies these newly masked inputs.
"""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from . import performance_ss15 as pilot
from .artifacts import sha256_file
from .science.analytical_benchmark_v2 import array_descriptor, build_problem

SCHEMA = "posteriorforge.execution.ss15_sigma_first.v1"
POLICY_ID = "all6_blocks_sigma_first_all3_globals_own_SS15_own_ordered_prefix_v1"
PILOT_SHA = "4099da3c19366997ab0800e5430b5bb1c129b4b3e535f678b3d4652233303480"
BLOCKS = (
    ("mechanical-1", (3,4,5,6), (0,1,2,3)),
    ("mechanical-2", (7,8,9,10), (4,5,6,7)),
    ("mechanical-3", (11,12,13,14), (8,9,10,11)),
    ("acoustic-1", (15,16), (12,)),
    ("acoustic-2", (17,18), (13,)),
    ("acoustic-3", (19,20), (14,)))
ORDER = (0,1,2,6,3,4,5,10,7,8,9,14,11,12,13,16,15,18,17,20,19)
INVERSE_ORDER = tuple(int(i) for i in np.argsort(ORDER))
FALLBACK_STAGE_MEMBERS = {10:(5,),20:(4,)}
_ORDER = ("mu_a","mu_b","mu_d0","a_m1","b_m1","d0_m1","sigma_m1",
          "a_m2","b_m2","d0_m2","sigma_m2","a_m3","b_m3","d0_m3","sigma_m3",
          "a_a1","sigma_a1","a_a2","sigma_a2","a_a3","sigma_a3")
_BASE_CLASS = pilot.SS15StageAdapter
_SEALS = {name:(getattr(_BASE_CLASS,name),getattr(_BASE_CLASS,name).__code__)
          for name in ("__init__","prepare","predict_component","predict","inspect_cache","close","metadata")}


def require(value, message):
    if not value:
        raise ValueError(message)


def source_sha256():
    return sha256_file(__file__)


def require_sources():
    require(pilot.source_sha256() == PILOT_SHA and pilot.SS15StageAdapter is _BASE_CLASS,
            "frozen4099 adapter source/identity changed")
    for name,(function,code) in _SEALS.items():
        require(getattr(_BASE_CLASS,name) is function and function.__code__ is code,
                "frozen adapter method changed: " + name)
    pilot.require_sources()
    problem = build_problem()
    require(tuple(problem.parameter_order) == _ORDER, "physical parameter order changed")
    prior = problem.prior.base
    require(tuple(prior.global_names) == _ORDER[:3], "all3 original globals required")
    for (name,physical,features),block,noise in zip(BLOCKS,prior.conditional_blocks,problem.prior.noise_names):
        require(tuple(block.local_names)+(noise,) == tuple(_ORDER[i] for i in physical),
                "declared full-block physical graph changed: " + name)


def order_theta(physical):
    require(isinstance(physical,np.ndarray) and physical.dtype == np.dtype("float32")
            and physical.ndim == 2 and physical.shape[1] == 21 and np.isfinite(physical).all(),
            "finite original physical Float32 theta[n,21] required")
    return np.ascontiguousarray(physical[:,ORDER])


def restore_physical(ordered):
    # Preserve invalid proposals too: only the ORIGINAL later support filter
    # decides whether to reject the complete vector.
    require(isinstance(ordered,np.ndarray) and ordered.dtype in (np.dtype("float32"),np.dtype("float64"))
            and ordered.ndim == 2 and ordered.shape[1] == 21,
            "complete stage-ordered Float32/Float64 vectors[n,21] required")
    return np.ascontiguousarray(ordered[:,INVERSE_ORDER])


def fallback_bindings():
    return [{"stage":j,"member":member,"target_physical_index":ORDER[j],
             "target_physical_name":_ORDER[ORDER[j]],"coordinate_system":"sigma-first stage index"}
            for j,members in FALLBACK_STAGE_MEMBERS.items() for member in members]


def stage_policy(stage):
    require(type(stage) is int and 0 <= stage < 21, "ordered stage0..20 required")
    physical_target=ORDER[stage]
    if stage < 3:
        observations=list(range(15));prefix=list(range(stage));block=None
    else:
        block,physical,ss=next(item for item in BLOCKS if physical_target in item[1])
        observations=list(ss)
        prefix=[j for j in range(stage) if ORDER[j]<3 or ORDER[j] in physical]
    kept=observations+[15+j for j in prefix]
    return {"stage":stage,"block":block,"input_width":15+stage,
            "observation_columns_kept":observations,
            "ordered_prefix_indices_kept":prefix,
            "physical_prefix_indices_kept":[ORDER[j] for j in prefix],
            "feature_columns_kept":kept,
            "feature_columns_zeroed":[j for j in range(15+stage) if j not in kept],
            "target_ordered_stage":stage,"target_physical_index":physical_target,
            "target_physical_name":_ORDER[physical_target],
            "global_used_predictors_and_labels_unchanged":stage<3}


def policy_metadata():
    definition = {"policy_id":POLICY_ID,"global_indices":[0,1,2],"stage_to_physical_order":list(ORDER),
                  "physical_to_stage_order":list(INVERSE_ORDER),
                  "stages":[stage_policy(j) for j in range(21)],
                  "zero_value":"positive float32 zero; no fitted mask",
                  "ordered_training_target_and_future_columns_unchanged":True,
                  "all6_blocks_same_fixed_rule":True}
    digest = hashlib.sha256(json.dumps(definition,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    return {"schema":SCHEMA,"module_sha256":source_sha256(),"policy_sha256":digest,
            **definition,"backend_adapter_source_sha256":PILOT_SHA,
            "same_feature_width":True,"learned_target_order_changed":True,"physical_output_order_restored":True,
            "fallback_stage_member_bindings":fallback_bindings(),
            "all3_globals_retained_for_local_factors":True,
            "masked_fixed_conditional_numerical_qualification_performed":False,
            "unmasked_anchor_numerical_qualification_transferred":False,
            "attribution":"Built with PriorLabs-TabPFN"}


def _training(theta, obs):
    require(isinstance(theta,np.ndarray) and theta.dtype == np.dtype("float32")
            and theta.shape == (4096,21) and np.isfinite(theta).all(),
            "finite4096x21 Float32 explicitly ORDERED target bank required")
    require(isinstance(obs,np.ndarray) and obs.dtype == np.dtype("float32")
            and obs.shape == (4096,15) and np.isfinite(obs).all(),
            "original finite4096x15 Float32 SS15 features required")


def mask_training(theta, obs, stage):
    _training(theta,obs)
    policy = stage_policy(stage)
    if stage < 3:
        return theta,obs
    target = np.array(theta,copy=True,order="C")
    features = np.array(obs,copy=True,order="C")
    removed_obs = [i for i in range(15) if i not in policy["observation_columns_kept"]]
    removed_prefix = [i for i in range(stage) if i not in policy["ordered_prefix_indices_kept"]]
    features[:,removed_obs] = np.float32(0)
    target[:,removed_prefix] = np.float32(0)
    return target,features


def mask_query(query, stage):
    policy = stage_policy(stage)
    if hasattr(query,"detach"):
        query = query.detach().cpu().numpy()
    require(isinstance(query,np.ndarray) and query.dtype == np.dtype("float32")
            and query.ndim == 2 and query.shape[1] == 15+stage and len(query)>0
            and not np.isinf(query).any(), "Float32 query[n,15+stage], NaN allowed, infinity rejected")
    if stage < 3:
        return query
    result = np.array(query,copy=True,order="C")
    result[:,policy["feature_columns_zeroed"]] = np.float32(0)
    return result


def training_bindings(theta, obs, masked_theta, masked_obs, stage):
    policy = stage_policy(stage)
    require(theta[:,stage:].tobytes() == masked_theta[:,stage:].tobytes(),
            "ordered training target/future columns changed")
    original = np.concatenate((obs,theta[:,:stage]),axis=1)
    masked = np.concatenate((masked_obs,masked_theta[:,:stage]),axis=1)
    require(mask_query(original,stage).tobytes() == masked.tobytes(), "training/query feature masks differ")
    return {"stage":stage,"policy_sha256":policy_metadata()["policy_sha256"],"mask":policy,
            "unmasked_theta":array_descriptor(theta),"unmasked_SS15":array_descriptor(obs),
            "prepared_theta_features":array_descriptor(masked_theta),"prepared_SS15":array_descriptor(masked_obs),
            "prepared_training_predictors":array_descriptor(masked),
            "original_target":array_descriptor(theta[:,stage]),"prepared_target":array_descriptor(masked_theta[:,stage]),
            "target_bytes_unchanged":True,"future_theta_columns_unchanged":True,
            "target_ordered_stage":stage,"target_physical_index":ORDER[stage],
            "target_physical_name":_ORDER[ORDER[stage]],
            "training_query_same_mask":True,"scope":"feature copy over explicitly ordered bank; current ordered target and future untouched"}


class SigmaFirstStageAdapter(_BASE_CLASS):
    def __init__(self,checkpoint,*,cdf_descriptor):
        require_sources()
        super().__init__(checkpoint,cdf_descriptor=cdf_descriptor)
        self.sigma_first_stage_inputs=[]
        self.sigma_first_query_calls=[]
        self._order_source=source_sha256()

    def prepare(self,theta,obs,stage):
        require(source_sha256()==self._order_source,"sigma-first mask source changed")
        masked_theta,masked_obs=mask_training(theta,obs,stage)
        binding=training_bindings(theta,obs,masked_theta,masked_obs,stage)
        result=super().prepare(masked_theta,masked_obs,stage)
        self.sigma_first_stage_inputs.append(binding)
        self._order_current=binding
        return result

    def predict_component(self,member,query,*,capture_member=False,with_log_prob=False):
        masked=mask_query(query,self.stage_index)
        result=super().predict_component(member,masked,capture_member=capture_member,with_log_prob=with_log_prob)
        proof={"schema":SCHEMA+".prediction","module_sha256":source_sha256(),
               "policy_sha256":policy_metadata()["policy_sha256"],"stage":self.stage_index,"member":member,
               "mask":stage_policy(self.stage_index),"unmasked_query":array_descriptor(query),
               "actual_masked_query":array_descriptor(masked),"physical_head_contexts":1,
               "training_query_same_fixed_mask":True,"ordered_target_binding_verified":True,"target_physical_index":ORDER[self.stage_index]}
        self.sigma_first_query_calls.append(proof)
        result["sigma_first_input_proof"]=proof
        return result

    def predict(self,query,*,output_type="full",capture_members=False):
        start=len(self.sigma_first_query_calls)
        result=super().predict(query,output_type=output_type,capture_members=capture_members)
        require(self.stage_index==0 and len(self.sigma_first_query_calls)-start==8,
                "stage0 original shared density must execute all8 unchanged heads")
        result["sigma_first_input_proof"]={"schema":SCHEMA+".prediction","module_sha256":source_sha256(),
            "policy_sha256":policy_metadata()["policy_sha256"],"stage":0,"member":None,
            "mask":stage_policy(0),"unmasked_query":array_descriptor(query),
            "actual_masked_query":array_descriptor(query),"physical_head_contexts":8,
            "training_query_same_fixed_mask":True,"ordered_target_binding_verified":True,"target_physical_index":ORDER[self.stage_index]}
        return result

    def inspect_cache(self):
        result=super().inspect_cache()
        return {**result,"sigma_first_stage_inputs":copy.deepcopy(getattr(self,"_order_current",None))}

    def metadata(self):
        return {**super().metadata(),"sigma_first_input_policy":policy_metadata(),
                "sigma_first_stage_inputs":copy.deepcopy(self.sigma_first_stage_inputs),
                "sigma_first_query_calls":copy.deepcopy(self.sigma_first_query_calls),
                "sigma_first_actual_head_contexts":len(self.sigma_first_query_calls),
                "unmasked_adapter_policy_scope":"frozen backend only; explicit ordered targets and masked inputs bound separately"}
