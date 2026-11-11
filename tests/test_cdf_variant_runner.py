"""CPU-only checks for explicit lineage and metadata isolation."""
from pathlib import Path
import json
import pytest
from posteriorforge import inference_cdf as variant
from posteriorforge.artifacts import write_json


def test_original_sources_remain_exactly_guarded():
    value = variant.guard_baseline_source()
    assert value["baseline_inference_sha256"] == variant.BASELINE_INFERENCE_SHA256
    assert len(value["upstream_files"]) == 6


def test_changed_baseline_runner_fails_closed(monkeypatch):
    monkeypatch.setattr(variant, "sha256_file", lambda _: "0"*64)
    with pytest.raises(RuntimeError, match="runner source hash"):
        variant.guard_baseline_source()


def test_manifest_decoration_is_copy_only_and_waits_for_restoration(tmp_path):
    value = {"schema": "posteriorforge.inference.v1", "variant": "upstream_unmodified", "status": "complete"}
    before = dict(value)
    descriptor = {"original_alias_restored": False}
    with variant.labelled_manifest_writes(tmp_path, descriptor, {"execution_optimized": False}) as state:
        variant.baseline.write_json(tmp_path/"manifest.json", value)
        assert state["owns_output"]
        recorded = json.loads((tmp_path/"manifest.json").read_text())
        assert recorded["status"] == "sampling_complete_pending_adapter_restore"
        assert recorded["variant"] == "cdf_arithmetic_v1"
        assert recorded["schema"] == variant.SCHEMA
    assert state["writer_restored"]
    assert variant.baseline.write_json is write_json
    assert value == before


def test_hook_rejects_other_artifacts_and_restores_on_exception(tmp_path):
    with pytest.raises(RuntimeError, match="unrelated artifact"):
        with variant.labelled_manifest_writes(tmp_path, {}, {}):
            variant.baseline.write_json(tmp_path/"samples.json", {"schema": "posteriorforge.inference.v1"})
    assert variant.baseline.write_json is write_json
    assert not (tmp_path/"samples.json").exists()


def test_existing_output_is_untouched_before_any_gpu_check(tmp_path):
    path = tmp_path/"manifest.json"
    path.write_text('immutable previous run')
    with pytest.raises(FileExistsError):
        variant.run_cdf_variant(banks="none", output=tmp_path, context_seed=20261111,
            case=0, checkpoint="none", ledger="none", reference="none", reference_grid="none")
    assert path.read_text() == 'immutable previous run'


def test_non_scientific_or_changed_batch_contract_rejected_before_gpu(tmp_path):
    for changes in ({"count": 1000}, {"microbatch": 4096}, {"repetition": 1}):
        with pytest.raises(ValueError, match="unoptimized scientific variant"):
            variant.run_cdf_variant(banks="none", output=tmp_path/"fresh", context_seed=20261111,
                case=0, checkpoint="none", ledger="none", reference="none", reference_grid="none", **changes)


@pytest.mark.parametrize("restore_ok", [True, False])
def test_whole_runner_labels_final_state_only_after_restoration(tmp_path, monkeypatch, restore_ok):
    from contextlib import contextmanager
    from types import SimpleNamespace
    import sys
    from posteriorforge import compat_cdf
    output = tmp_path/"new-run"
    descriptor = {"original_alias_restored": False, "patched_call_count": 0}
    expected = 21*8*2
    @contextmanager
    def fake_adapter():
        try:
            yield descriptor
        finally:
            descriptor["original_alias_restored"] = restore_ok
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 1),
        get_float32_matmul_precision=lambda: "highest",
        backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False)),
                                 cudnn=SimpleNamespace(allow_tf32=True)))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(compat_cdf, "finite_cdf_compatibility", fake_adapter)
    monkeypatch.setattr(variant, "_require_budgeted_gpu", lambda _: {"gpus": 1})
    monkeypatch.setattr(variant, "reference_binding", lambda **_: {"test": "metadata-only"})
    seen = {}
    def fake_run(**kwargs):
        seen.update(kwargs)
        output.mkdir()
        state = {"schema": "posteriorforge.inference.v1", "variant": "upstream_unmodified", "status": "running"}
        variant.baseline.write_json(output/"manifest.json", state)
        descriptor["patched_call_count"] = expected
        state.update(status="complete", rejection={"support_calls": 2}, model={"dtype": "torch.float32", "autocast": False})
        variant.baseline.write_json(output/"manifest.json", state)
        assert json.loads((output/"manifest.json").read_text())["status"] == "sampling_complete_pending_adapter_restore"
        return state
    monkeypatch.setattr(variant.baseline, "run_baseline", fake_run)
    kwargs = dict(banks="saved-bank", output=output, context_seed=20261111, case=0,
                  checkpoint="saved-checkpoint", ledger="reserved-ledger", reference="saved-reference", reference_grid="saved-grid")
    if restore_ok:
        result = variant.run_cdf_variant(**kwargs)
        assert result["status"] == "complete"
        assert result["compatibility_adapter"]["original_alias_restored"] is True
        assert result["expected_patched_call_count"] == expected
        assert result["metadata_writer_restored"] is True
    else:
        with pytest.raises(RuntimeError, match="activation/restoration"):
            variant.run_cdf_variant(**kwargs)
    saved = json.loads((output/"manifest.json").read_text())
    assert saved["status"] == ("complete" if restore_ok else "failed")
    assert saved["variant"] == variant.VARIANT
    assert variant.baseline.write_json is write_json
    assert seen == dict(banks="saved-bank", output=output, context_seed=20261111, case=0,
        context_size=4096, count=50000, microbatch=10000, checkpoint="saved-checkpoint",
        checkpoint_sha256=variant.baseline.CHECKPOINT_SHA256, repetition=0, device="cuda")
