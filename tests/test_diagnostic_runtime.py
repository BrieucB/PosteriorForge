"""CPU-only checks of reversible instrumentation and trace accounting."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts/diagnose_runtime.py"
spec = importlib.util.spec_from_file_location("runtime_diagnostic", SCRIPT)
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


def test_overlapping_cuda_intervals_are_unioned_not_summed():
    assert runtime.interval_union([(0, 10), (5, 12), (20, 22), (21, 24)]) == 16
    assert runtime.interval_union([]) == 0
    assert runtime.interval_union([(1, 1)]) == 0


def test_instrumentation_preserves_results_and_restores_bound_methods_on_error():
    calls = []
    class Model:
        def fit(self, x, y):
            calls.append(("fit", x.shape))
            return self
        def predict(self, x, **kwargs):
            calls.append(("predict", x.shape, kwargs))
            return {"same": x}
    class Posterior:
        def __init__(self): self._model = Model()
        def _sample(self, sampling_batch_size, x, **kwargs):
            train = SimpleNamespace(shape=(4096, 33))
            self._model.fit(train, "target")
            result = self._model.predict(x, output_type="full", quantiles=[])
            cuda.empty_cache(); cuda.ipc_collect()
            return result
        def _within_support(self, value):
            calls.append(("support", value)); return True
    cuda = SimpleNamespace(empty_cache=lambda: calls.append(("empty",)),
                           ipc_collect=lambda: calls.append(("ipc",)))
    posterior = Posterior()
    original_fit = posterior._model.fit
    original_empty = cuda.empty_cache
    query = SimpleNamespace(shape=(17, 33))
    recorder = runtime.CallRecorder()
    recorder.request = "test"
    with pytest.raises(RuntimeError, match="deliberate"):
        with recorder.instrument(posterior, cuda):
            assert posterior._sample(sampling_batch_size=17, x=query) == {"same": query}
            assert posterior._within_support("rows") is True
            raise RuntimeError("deliberate")
    assert posterior._model.fit == original_fit
    assert "fit" not in posterior._model.__dict__
    assert "_sample" not in posterior.__dict__
    assert cuda.empty_cache is original_empty
    assert [event["operation"] for event in recorder.events] == ["fit", "predict", "empty_cache", "ipc_collect", "proposal", "support"]
    assert next(event for event in recorder.events if event["operation"] == "proposal")["query_rows"] == 17
    assert next(event for event in recorder.events if event["operation"] == "fit")["stage_index"] == 0
    assert next(event for event in recorder.events if event["operation"] == "predict")["parent_operation"] == "proposal"
    assert next(event for event in recorder.events if event["operation"] == "empty_cache")["parent_operation"] == "proposal"
    assert calls == [("fit", (4096, 33)), ("predict", (17, 33), {"output_type": "full", "quantiles": []}), ("empty",), ("ipc",), ("support", "rows")]


def test_trace_summary_does_not_double_count_kernels_or_add_host_sync(tmp_path):
    trace = {"traceEvents": [
        {"ph": "X", "cat": "user_annotation", "name": "pfdiagnostic/profile/request", "ts": 0, "dur": 10_000_000},
        {"ph": "X", "cat": "kernel", "name": "kernel_a", "ts": 1_000_000, "dur": 4_000_000},
        {"ph": "X", "cat": "kernel", "name": "kernel_b", "ts": 3_000_000, "dur": 4_000_000},
        {"ph": "X", "cat": "gpu_memcpy", "name": "Memcpy HtoD", "ts": 7_000_000, "dur": 1_000_000, "args": {"Bytes": 1024}},
        {"ph": "X", "cat": "cuda_runtime", "name": "cudaStreamSynchronize", "ts": 2_000_000, "dur": 5_000_000},
    ]}
    path = tmp_path / "trace.json"; path.write_text(json.dumps(trace))
    result = runtime.summarize_trace(path)
    assert result["cuda_kernel_summed_seconds"] == 8
    assert result["cuda_kernel_interval_union_seconds"] == 6
    assert result["cuda_kernel_and_copy_interval_union_seconds"] == 7
    assert result["cuda_runtime_sync_host_seconds"] == 5
    assert result["kernel_busy_fraction_of_profile_span"] == .6
    assert result["copies"][0]["bytes"] == 1024
    assert result["cuda_activity_status"] == "recorded"


def test_import_and_help_do_not_import_torch_or_initialize_model():
    code = f"import runpy,sys;runpy.run_path({str(SCRIPT)!r},run_name='diagnostic_import');assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)
    result = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True, check=True)
    assert "summary" in result.stdout and "profile" in result.stdout


def test_profile_refuses_unbudgeted_execution(tmp_path, monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(RuntimeError, match="reserved Slurm"):
        runtime.profile_baseline(banks="none",checkpoint="none",output=tmp_path/"out",ledger=tmp_path/"ledger.json",profile_count=10000)
    assert not (tmp_path/"out").exists()


@pytest.mark.parametrize("chunk_bytes", [16, 23, 73, 1024])
def test_stream_reader_handles_event_boundaries_unicode_and_footer(tmp_path, chunk_bytes):
    events = [
        {"name": "unicode_λ", "ph": "M", "args": {"text": "a,b]c{d}"}},
        {"ph": "X", "cat": "kernel", "name": "k", "ts": 3, "dur": 8},
    ]
    path = tmp_path / "trace.json"
    path.write_text(json.dumps({"traceEvents": events, "tail": "metadata"}, ensure_ascii=False))
    assert list(runtime.iter_trace_events(path, chunk_bytes)) == events


def test_stream_reader_rejects_truncated_trace_array(tmp_path):
    path = tmp_path / "trace.json"
    path.write_text(json.dumps({"traceEvents": [{"ph": "X", "dur": 100}]})[:-5])
    with pytest.raises(ValueError, match="truncated or invalid"):
        list(runtime.iter_trace_events(path, 23))


def test_cpu_sampler_requires_completed_capture_before_library_imports(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"state": "running"}))
    with pytest.raises(ValueError, match="completed authenticated"):
        runtime.benchmark_captured_sampler(capture_dir=tmp_path, output=tmp_path / "out.json")
    assert not (tmp_path / "out.json").exists()
