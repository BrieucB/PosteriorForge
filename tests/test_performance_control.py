"""No model/GPU dependency: controls preserve scientific execution metadata."""
import json
import pytest
from posteriorforge import performance_control as control
from posteriorforge.artifacts import write_json


def test_explicit_phase_metadata_does_not_mutate_payload(tmp_path):
    raw={'schema':'posteriorforge.inference.v1','status':'complete','variant':'upstream_unmodified'}
    old=dict(raw)
    with control.metadata_writes(tmp_path, {'patched_call_count':0}, {'performance_first_authorized':True}) as state:
        control.baseline.write_json(tmp_path/'manifest.json',raw)
        saved=json.loads((tmp_path/'manifest.json').read_text())
        assert saved['variant']=='performance_control_v1'
        assert saved['status']=='sampling_complete_pending_adapter_restore'
        assert saved['performance_first_authorized'] is True
    assert raw==old and state['restored']
    assert control.baseline.write_json is write_json


def test_existing_output_is_never_overwritten(tmp_path):
    p=tmp_path/'manifest.json';p.write_text('existing')
    with pytest.raises(FileExistsError):
        control.run_control(banks='x',output=tmp_path,context_seed=20261111,case=0,
            checkpoint='x',ledger='x',reference='x',reference_grid='x')
    assert p.read_text()=='existing'


def test_50k_control_rejects_changed_context_before_gpu(tmp_path):
    with pytest.raises(ValueError):
        control.run_control(banks='x',output=tmp_path/'new',context_seed=20261111,case=0,
            checkpoint='x',ledger='x',reference='x',reference_grid='x',context_size=1024)


def test_unrelated_write_rejected_and_alias_restored(tmp_path):
    with pytest.raises(RuntimeError,match='unrelated'):
        with control.metadata_writes(tmp_path,{},{}):
            control.baseline.write_json(tmp_path/'other.json',{'schema':'posteriorforge.inference.v1'})
    assert control.baseline.write_json is write_json
