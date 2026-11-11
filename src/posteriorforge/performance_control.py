"""Contemporaneous 50k corrected-unoptimized timing controls, explicitly opted in.

Built with PriorLabs-TabPFN. The latest human performance-first instruction
supersedes the former optimization stop rule for this phase only. The known
absolute posterior-accuracy failure remains visible and unchanged.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import time

from . import inference as baseline
from .artifacts import sha256_file, write_json
from .compat_cdf import finite_cdf_compatibility
from .inference_cdf import guard_baseline_source, reference_binding
from .reference_grid import _require_budgeted_gpu

SCHEMA = 'posteriorforge.inference.execution.v1'
VARIANT = 'performance_control_v1'
_ACTIVE = False


def decorate(payload, descriptor, common, *, pending=False):
    result = dict(payload)
    result.update(common)
    result.update(schema=SCHEMA, variant=VARIANT, compatibility_adapter=dict(descriptor))
    if pending and result.get('status') == 'complete':
        result['status'] = 'sampling_complete_pending_adapter_restore'
    return result


@contextmanager
def metadata_writes(output, descriptor, common):
    global _ACTIVE
    if _ACTIVE or baseline.write_json is not write_json:
        raise RuntimeError('original writer must be unmodified and non-reentrant')
    original = baseline.write_json
    expected = Path(output).resolve()/'manifest.json'
    state = {'owns_output': False, 'restored': False}
    def recording(path, payload):
        if Path(path).resolve() != expected or payload.get('schema') != 'posteriorforge.inference.v1':
            raise RuntimeError('refused unrelated metadata write')
        state['owns_output'] = True
        original(path, decorate(payload, descriptor, common, pending=True))
    _ACTIVE = True
    baseline.write_json = recording
    try:
        yield state
    finally:
        expected_alias = baseline.write_json is recording
        baseline.write_json = original
        _ACTIVE = False
        state['restored'] = baseline.write_json is original
        if not expected_alias:
            raise RuntimeError('inference writer changed during control')


def run_control(*, banks, output, context_seed, case, checkpoint, ledger,
                reference, reference_grid, repetition=0, context_size=4096):
    output = Path(output)
    if output.exists():
        raise FileExistsError('fresh control output required')
    if type(repetition) is not int or repetition < 0 or context_size != 4096:
        raise ValueError('nonnegative repetition and fixed4096context required')
    start = time.perf_counter()
    source = guard_baseline_source()
    budget = _require_budgeted_gpu(ledger)
    if budget['gpus'] != 1:
        raise RuntimeError('one reserved GPU required for each control')
    refs = reference_binding(banks=banks, reference=reference, reference_grid=reference_grid, case=case)
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('one visible CUDA device required')
    if torch.get_float32_matmul_precision() != 'highest' or torch.backends.cuda.matmul.allow_tf32:
        raise RuntimeError('unchanged highestFP32/TF32False required')
    common = {
        'attribution': 'Built with PriorLabs-TabPFN', 'run_purpose': 'performance_control',
        'performance_first_authorized': True, 'absolute_accuracy_gate_required_for_performance': False,
        'known_physical_control_status': 'model_failure', 'count_contract': '50000completephysicalvectors',
        'budget_binding': budget, 'reference_binding': refs, 'baseline_execution_source': source,
        'tail_family_changed': False, 'execution_optimized': False,
        'metadata_writer_restored': False,
        'execution_engine': {'engine_id': 'corrected_unoptimized', 'engine_version': 1,
            'module_sha256': sha256_file(__file__), 'cache_policy': 'original_fit_preprocessors',
            'sampling_rng_policy': 'original_shared_torch_rng', 'rng_role': 'posterior',
            'model_rng_isolation': False, 'query_batch_size': 10000, 'posterior_count': 50000},
    }
    status = None; descriptor = {}; hook = {'owns_output': False}
    try:
        with finite_cdf_compatibility() as active:
            descriptor = active
            common['wrapper_preexecution_setup_seconds'] = time.perf_counter()-start
            with metadata_writes(output, descriptor, common) as hook:
                status = baseline.run_baseline(banks=banks, output=output, context_seed=context_seed,
                    case=case, context_size=context_size, count=50000, microbatch=10000,
                    checkpoint=checkpoint, checkpoint_sha256=baseline.CHECKPOINT_SHA256,
                    repetition=repetition, device='cuda')
        expected = 21*8*status['rejection']['support_calls']
        if not hook['restored'] or not descriptor.get('original_alias_restored') or descriptor['patched_call_count'] != expected:
            raise RuntimeError('control activation/call coverage/restoration failed')
        if guard_baseline_source() != source:
            raise RuntimeError('original source changed during control')
        common['metadata_writer_restored'] = True
        common['expected_patched_call_count'] = expected
        status = decorate(status, descriptor, common)
        status['wrapper_total_seconds'] = time.perf_counter()-start
        write_json(output/'manifest.json', status)
        return status
    except BaseException as error:
        if hook['owns_output']:
            path = output/'manifest.json'
            status = status or (json.loads(path.read_text()) if path.exists() else {})
            status = decorate(status, descriptor, common)
            status.update(status='failed', error_type=type(error).__name__, error=str(error),
                          wrapper_total_seconds=time.perf_counter()-start,
                          metadata_writer_restored=baseline.write_json is write_json)
            write_json(path, status)
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('banks','output','checkpoint','ledger','reference','reference-grid'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--context-seed', type=int, default=20261111)
    p.add_argument('--case', type=int, default=0)
    p.add_argument('--repetition', type=int, default=0)
    p.add_argument('--context-size', type=int, default=4096)
    print(json.dumps(run_control(**vars(p.parse_args())), indent=2))


if __name__ == '__main__':
    main()
