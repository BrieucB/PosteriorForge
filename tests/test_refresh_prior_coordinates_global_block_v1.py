"""CPU source/header/decode custody tests; no real cohort, model, or refresh run."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
from posteriorforge.science.analytical_benchmark_v2 import array_descriptor

path=Path(__file__).resolve().parents[1]/"scripts/refresh_prior_coordinates_global_block_v1.py"
spec=importlib.util.spec_from_file_location("test_prior_joint",path)
w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)


def complete_artifacts():
    latent=np.zeros((51200,21),np.float32)
    physical=w.codec.decode_ordered(latent)
    doc={"versions":w.versions(),"latent_wave_archive":{"keys":["wave_0000"],"generated_row_count":51200,
                               "all_generated_rows_including_unused_tail":True},
         "proposal_wave_sizes":[51200],
         "decode_wave_records":[{"wave_index":0,"row_count":51200,"actual_latent_draws":array_descriptor(latent),
             "decoded_physical":array_descriptor(physical),"strict_decode_passed":True,
             "includes_unused_FIFO_tail":True,"input_dtype_unchanged_before_decode":True}],
         "rejection":{"support_calls":1,"proposed":50000,"accepted_before_final_trim":50000,"returned":50000}}
    return doc,{"wave_0000":latent},physical[:50000].copy()


def test_unpinned_new_family_cannot_refresh():
    with pytest.raises(ValueError,match="not pinned"):
        w.source_guard()


def test_frozen_refresh_callable_and_kernel_identity():
    assert w.refresh_once is w.backbone.refresh_once
    assert w.kernel.source_sha256()==w.KERNEL_SHA256
    assert w.ROLE==w.backbone.ROLE=="ss15_global_block_gibbs_once_v1"


def test_full_generated_wave_decode_and_saved_prefix_exact():
    doc,waves,values=complete_artifacts()
    result=w.replay_generated_decode(doc,waves,values)
    assert result["generated_rows_strictly_replayed"]==51200
    assert result["unused_FIFO_tail_rows"]==1200
    assert result["physical_saved_output_byte_identical"]


def test_unused_tail_corruption_rejected():
    doc,waves,values=complete_artifacts()
    waves["wave_0000"][-1,3]=np.nan
    with pytest.raises(ValueError):
        w.replay_generated_decode(doc,waves,values)


def test_decode_has_no_post_G3_reinterpretation():
    doc,waves,values=complete_artifacts()
    changed=values.copy();changed[:,0]+=1.
    with pytest.raises(ValueError,match="accepted prefix"):
        w.replay_generated_decode(doc,waves,changed)
    # The physical local columns are a separately invariant view of the parent.
    assert values[:,3:].tobytes()==changed[:,3:].tobytes()


@pytest.mark.parametrize("key,value",[
    ("support_calls",2),("proposed",50001),("accepted_before_final_trim",49999),("returned",49999)])
def test_strict_decode_cannot_invent_finite_rejection_or_trimming(key,value):
    doc,waves,values=complete_artifacts();doc["rejection"][key]=value
    with pytest.raises(ValueError,match="conservation"):
        w.replay_generated_decode(doc,waves,values)


def test_actual_latent_dtype_cannot_be_silently_recast():
    doc,waves,values=complete_artifacts();waves["wave_0000"]=waves["wave_0000"].astype(np.float64)
    with pytest.raises(ValueError,match="Float32"):
        w.replay_generated_decode(doc,waves,values)


@pytest.mark.parametrize("key",["python","numpy","scipy"])
def test_exact_replay_requires_actual_producer_versions(key):
    document={"versions":dict(w.versions())}
    assert w.require_replay_environment(document)["exact_same_environment_identity"]
    document["versions"][key]="different"
    with pytest.raises(ValueError,match="actual producer inference environment"):
        w.require_replay_environment(document)


def test_environment_mismatch_fails_before_any_decode(monkeypatch):
    doc,waves,values=complete_artifacts()
    doc["versions"]["scipy"]="not-this-version"
    def forbidden_decode(*args):
        raise AssertionError("decode must not run")
    monkeypatch.setattr(w.codec,"decode_ordered",forbidden_decode)
    with pytest.raises(ValueError,match="actual producer inference environment"):
        w.replay_generated_decode(doc,waves,values)
