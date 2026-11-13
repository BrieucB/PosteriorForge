"""Actual case0 data regression only; no model/NN/proposal/G3/RNG calls."""
import hashlib
import json
import pickle
import random
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from posteriorforge import prior_coordinate_pair_v1 as pairs
from posteriorforge import prior_coordinates_v1 as training
from posteriorforge import execution_prior_native_v3 as backend
from posteriorforge import performance_prior_pair_v3 as producer
from posteriorforge.science.analytical_benchmark_v2 import array_descriptor

RESULTS=Path("/scratch/project/eu-26-17/eubrieucb/posteriorforge/results")
PHASE=RESULTS/"accuracy-prior-coordinates-v1"
CPU_SHA="6f35d9d090a955604186c5fbf3312b82465eff6fdae7e4cd24a23b646dbdf193"
FAILED_SHA="92755b788fc894d4ca16157e573bd1b5c9d8a4468b71cf184fa16303bced7bce"

def test_actual_case0_training_inputs_masks_queries_targets_match_6f35_and_failed92755():
    cpu_path=PHASE/"pair-preflight-combined-01/receipt.json"
    failed_path=PHASE/"native-pilot-v2-seed20261111-case0-01/manifest.json"
    banks=RESULTS/"banks-v1"
    if not cpu_path.exists() or not failed_path.exists() or not banks.exists():
        pytest.skip("authenticated Karolina actual-case0 bank/receipt fixture unavailable")
    assert hashlib.sha256(cpu_path.read_bytes()).hexdigest()==CPU_SHA
    assert hashlib.sha256(failed_path.read_bytes()).hexdigest()==FAILED_SHA
    cpu=json.loads(cpu_path.read_bytes());failed=json.loads(failed_path.read_bytes())
    spec=pairs.PairSpec(20261111,0)
    before=pickle.dumps((np.random.get_state(),random.getstate()),protocol=5)
    saved_aliases=(training.encode_physical,training.encode_for_model,backend.mask_training,backend.mask_query)
    panel=pairs.validate_pair_cpu_receipt(cpu,spec)
    assert panel["COMMON_Float32_model_bytes_match"] is True
    canonical=panel["inference"]
    theta,stats,query,prefixes,raw,transformed,custody=pairs.input_bindings(spec,
        banks=banks,
        completion_receipt=RESULTS/"accuracy-ss15-v1/sigma-native-grid-completion-01/completion.json",
        training_inventory=PHASE/"codec-inventory-inference-01/manifest.json",
        pair_inventory=cpu)
    assert raw==canonical["raw_bindings"]==failed["raw_scientific_bindings"]
    assert failed["pair_spec"]==spec.as_dict()
    context,raw_query,raw_again=pairs.raw_inputs(spec,banks=banks)
    assert raw_again==raw
    theta2,stats2,query2,model,receipt=producer.model_bindings(context,raw_query)
    assert producer.model_bindings is producer.v2.model_bindings
    assert producer.codec is training
    assert model==failed["model_inputs"]==failed["model_target_descriptors"]["model_inputs"]
    assert theta2.tobytes()==theta.tobytes() and stats2.tobytes()==stats.tobytes() and query2.tobytes()==query.tobytes()
    assert receipt["latent_physical_slot_model_Float32"]==failed["model_target_descriptors"]["latent_physical_slot_model_Float32"]
    assert receipt["ordered_latent_model_Float32"]==failed["model_target_descriptors"]["ordered_latent_model_Float32"]
    assert receipt["model_target_coordinate_system"]==failed["model_target_descriptors"]["model_target_coordinate_system"]
    assert array_descriptor(theta)==transformed["context_ordered_latent_model_Float32"]
    assert array_descriptor(stats)==transformed["context_SS15_model_Float32"]
    assert array_descriptor(query)==transformed["query_SS15_model_Float32"]
    assert array_descriptor(prefixes)==transformed["fixed_prefix_ordered_latent_model_Float32"]
    observed=failed["observed_failure_footer"]["observations"]["actual_engine_journal"]
    assert observed["status"]=="observed"
    old_training=observed["value"]["actual_latent_training_inputs"]
    assert len(old_training)==len(canonical["all21_mask_target_query_bindings"])==21
    data=pairs.load_script("qualify_prior_conditionals_v1.py",pairs.INPUT_HELPER_SHA256)
    # Frozen query fixture validates the original strict training codec; this
    # namespace supplies only the unchanged v3 mask aliases, no neural executor.
    fixed=SimpleNamespace(stage_policy=backend.stage_policy,codec_module=backend.cached.codec_module)
    for stage in range(21):
        main,stress,_=data.fixed_queries(query,prefixes,theta,stage,fixed)
        mt,ms,mq,sq,binding=data.masked_inputs(theta,stats,main,stress,stage,backend)
        assert binding==canonical["all21_mask_target_query_bindings"][stage]
        assert binding["training"]==old_training[stage]
        assert binding["actual_latent_target"]==array_descriptor(theta[:,stage])
        assert binding["actual_latent_target"]==binding["training"]["prepared_latent_target"]==binding["training"]["original_latent_target"]
        assert mt[:,stage:].tobytes()==theta[:,stage:].tobytes()
        assert array_descriptor(np.concatenate((ms,mt[:,:stage]),axis=1))==binding["actual_training"]
        assert array_descriptor(mq)==binding["main"] and array_descriptor(sq)==binding["stress"]
    assert custody["rule"]==pairs.PREFIX_RULE and custody["selected_indices"]==list(range(1024))
    assert custody["selection_replaced_or_filtered"] is False
    assert saved_aliases==(training.encode_physical,training.encode_for_model,backend.mask_training,backend.mask_query)
    assert pickle.dumps((np.random.get_state(),random.getstate()),protocol=5)==before
