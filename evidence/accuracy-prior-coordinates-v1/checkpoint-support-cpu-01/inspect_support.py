"""CPU-only checkpoint/common finite-bar support determination; no neural forward."""
from pathlib import Path
import hashlib, json, pickle, random, sys
import numpy as np
import torch
from posteriorforge import execution_cache as cache

ROOT=Path("/scratch/project/eu-26-17/eubrieucb/posteriorforge")
REPO=Path("/home/it4i-bbenvegnen/workspace/PosteriorForge")
OUT=ROOT/"results/accuracy-prior-coordinates-v1/checkpoint-support-cpu-01"
MANIFEST=ROOT/"results/accuracy-prior-coordinates-v1/native-pilot-v2-seed20261111-case0-01/manifest.json"
CHECKPOINT=ROOT/"cache/model/tabpfn-v2-regressor.ckpt"
sha=lambda b:hashlib.sha256(b).hexdigest()
file_sha=lambda p:sha(Path(p).read_bytes())
EXPECTED_MANIFEST="92755b788fc894d4ca16157e573bd1b5c9d8a4468b71cf184fa16303bced7bce"
EXPECTED_CHECKPOINT="2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736"
assert file_sha(MANIFEST)==EXPECTED_MANIFEST and file_sha(CHECKPOINT)==EXPECTED_CHECKPOINT
m=json.loads(MANIFEST.read_bytes())
sdk=Path(torch.__file__).parent.parent/"tabpfn"
sdk_hashes={name:file_sha(sdk/Path(name).relative_to("tabpfn")) for name in cache.PINNED_SDK_SHA256}
assert sdk_hashes==dict(cache.PINNED_SDK_SHA256)
sources={
 "src/posteriorforge/performance_runner.py":"23ccad8b5bbcdd7a48340b349d515ca870763359cfeb71333e02fc2b6a8aca3b",
 "src/posteriorforge/execution_mixture.py":"e71f43b73ae8259c20b5d7b52feccc7c851e9a2fbb86d869bc0a5f7f3af792a1",
 "src/posteriorforge/performance_prior_pair_v2.py":m["runner_source_sha256"],
}
assert all(file_sha(REPO/n)==h for n,h in sources.items())
def states():
 return {
 "numpy":sha(pickle.dumps(np.random.get_state())),
 "python":sha(pickle.dumps(random.getstate())),
 "torchCPU":sha(torch.random.get_rng_state().numpy().tobytes()),
 }
before=states()
cuda_before=torch.cuda.is_initialized()
assert not cuda_before
checkpoint=torch.load(CHECKPOINT,map_location="cpu",weights_only=False)
assert set(checkpoint)=={"state_dict","config"}
b=checkpoint["state_dict"]["criterion.borders"]
assert b.dtype==torch.float32 and b.device.type=="cpu"
assert bool(torch.isfinite(b).all()) and bool((b[1:]>b[:-1]).all())
scopes=m["observed_failure_footer"]["observations"]["actual_engine_journal"]["value"]["completed_stage_scopes"]
verdicts=[]
for stage,row,z in [(15,4355,13.876729011535645),(19,41765,205.2462615966797)]:
 model=scopes[stage]["model"]
 cold=scopes[stage]["model_at_prepare"]
 assert model["target_mean"]==cold["target_mean"] and model["target_sd"]==cold["target_sd"]
 # EXACT stock regressor.py:528 arithmetic on original CPU Float32 borders.
 physical=(b*model["target_sd"]+model["target_mean"]).float()
 event=m["stage_events"][stage]["sampler"]
 calls=[x for x in m["observed_failure_footer"]["observations"]["actual_engine_journal"]["value"]["actual_latent_query_calls"] if x["stage"]==stage]
 assert all(set(x["query_masked"])=={"dtype","shape","sha256"} and set(x["query_unmasked"])=={"dtype","shape","sha256"} for x in calls)
 verdicts.append({
 "stage":stage,"row_index_zero_based":row,"observed_ordered_latent":z,
 "saved_target_mean":model["target_mean"],"saved_target_sd":model["target_sd"],
 "saved_cold_and_final_target_normalization_equal":True,
 "all8_common_returned_criterion_minimum":physical[0].item(),
 "all8_common_returned_criterion_maximum":physical[-1].item(),
 "common_returned_borders_sha256":sha(physical.numpy().tobytes()),
 "outside_common_finite_bar_envelope":bool(z<physical[0].item() or z>physical[-1].item()),
 "above_common_maximum_by":z-physical[-1].item(),
 "verdict":("OUTSIDE all8 common finite-bar envelope; ordinary within-bin interpolation cannot explain the value" if z>physical[-1].item() else "INSUFFICIENT: inside common envelope, actual positive-mass head support and interpolation details unavailable"),
 "historical_head_rejections":event["head_rejections"],
 "historical_head_rejection_counts_by_member":event["head_rejection_counts_by_member"],
 "query_journal_entries":len(calls),
 "query_journal_has_only_shape_dtype_sha_not_row_indices":True,
 "exact_historical_member_grouped_batch_identified":False,
 "new_contiguous_probe_window":[4096,5120] if stage==15 else [40960,41984],
 "row_offset_in_new_contiguous_probe_window":row-(4096 if stage==15 else 40960),
 })
after=states()
assert after==before and not torch.cuda.is_initialized()
assert file_sha(MANIFEST)==EXPECTED_MANIFEST and file_sha(CHECKPOINT)==EXPECTED_CHECKPOINT
assert all(file_sha(REPO/n)==h for n,h in sources.items())
report={
 "schema":"posteriorforge.checkpoint_common_finite_bar_support_cpu.v1",
 "status":"split_supported_and_insufficient_verdict",
 "manifest_sha256":EXPECTED_MANIFEST,"checkpoint_sha256":EXPECTED_CHECKPOINT,
 "code_sha256":file_sha(__file__),
 "checkpoint_inspection":{"loaded_keys":list(checkpoint),"criterion_border_key":"state_dict.criterion.borders","dtype":str(b.dtype),"shape":list(b.shape),"minimum":b[0].item(),"maximum":b[-1].item(),"sha256":sha(b.contiguous().numpy().tobytes())},
 "source_hashes":sources,"pinned_SDK_source_hashes_verified":sdk_hashes,
 "source_line_evidence":{
 "tabpfn/regressor.py":{"standardize_and_renormalized_criterion":[521,529],"member_target_inverse_borders":[672,685],"all_members_translate_back_to_common_bardist":[695,701],"returned_common_criterion":[747,753]},
 "src/posteriorforge/execution_mixture.py":{"single_member_filtering":[148,161],"finite_bar_icdf":[224,252]},
 "src/posteriorforge/performance_runner.py":{"preassigned_component_uniforms":[486,493],"head_rejection_grouping":[501,523]},
 },
 "all8_support_scope":"All8 returned head distributions are translated to checkpoint canonical bardist bins and sampled using the single target-mean/SD-renormalized criterion. Per-head SafePower/inverse-target fitted parameters are not needed to bound this common output envelope; actual nonzero support within it is not known.",
 "full_support_classname_caveat":"This conclusion concerns the source-bound finite_bar_icdf used by this sampler, not an arbitrary tail-sampling method on the FullSupportBarDistribution class.",
 "per_stage_verdicts":verdicts,
 "historical_authentication_verdict":"Component U can be independently reconstructed because it is preassigned by raw row and retained across retries. Saved query journals identify member/order/shape/SHA but omit raw row indices and row-specific assignment/acceptance decisions. Counts with3/1 rejections at stages15/19 do not identify which row retried or its final member/chunk. No assignment/acceptance stream replay, row-subset brute force, or historical bin/probability inference was performed. New fixed contiguous1024 windows are not authenticated historical member-grouped batches.",
 "target_transform_replay_calls":0,"preprocessing_replay_calls":0,
 "RNG_replay_or_new_draw_calls":0,"model_constructor_or_forward_calls":0,
 "checkpoint_tensor_loading_only":True,
 "default_RNG_preservation_scope":"Snapshots after imports and before/after checkpoint state-dict loading plus CPU Float32 border arithmetic; no initial-import RNG claim.",
 "default_RNG_hashes_before":before,"default_RNG_hashes_after":after,"default_RNG_preserved":True,
 "CUDA_initialized_before":cuda_before,"CUDA_initialized_after":torch.cuda.is_initialized(),
 "no_posterior_G3_decode_cohort_repair_or_source_edit":True,
 "prior_array_diagnostic_unchanged":"c08076c86e7d329f65ddb417224c2430ceb4b3dfdb0e24e00ae8f6000e691f93",
}
OUT.mkdir(parents=True,exist_ok=True)
(OUT/"receipt.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
print(json.dumps({"receipt_sha256":file_sha(OUT/"receipt.json"),"stage_verdicts":verdicts,"default_RNG_preserved":True,"CUDA_initialized":False},sort_keys=True))
