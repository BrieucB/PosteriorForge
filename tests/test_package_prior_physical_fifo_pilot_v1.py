"""Synthetic metadata curation only; no actual samples, model, Torch or GPU."""
import gzip,hashlib,importlib.util,json
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location("physical_fifo_pilot_curator",Path(__file__).resolve().parents[1]/"scripts/package_prior_physical_fifo_pilot_v1.py")
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)


def fixture(tmp_path,status):
    d={"schema":p.SCHEMA,"variant":p.VARIANT,"runner_source_sha256":p.PRODUCER_SHA,
       "context_seed":20261111,"case":2,"context_size":4096,"requested_count":50000,
       "execution_policy":{"physical_fifo_source_sha256":p.FIFO_SHA},
       "matching_conditional_qualification_sha256":p.PROOF_SHA,"slurm_job_id":"unit-fixture",
       "status":status,"stage_events":[{"stage_index":0,"prepare_seconds":1.25,
           "predict_and_sample_seconds":2.5}],"git_dirty":True}
    if status=="complete":
        d.update({key:{} for key in p.NORMAL_FIELDS})
        d.update(default_rng_preserved=True,sources_unchanged=True,
            completed_stage_indices=list(range(21)),rejection={"returned":50000},
            compatibility_adapter={"original_alias_restored":True,"sdk_sources_unchanged":True})
    else:d.update(error_type="ValueError",error="observed synthetic failure")
    m=tmp_path/'manifest.json';m.write_text(json.dumps(d)+"\n");mh=p.sha(m.read_bytes())
    a={"raw_manifest_sha256":mh,"ledger_fully_reconciled":True,"GPUs":1,
       "job_id":"unit-fixture","Slurm":{"State":"COMPLETED" if status=="complete" else "FAILED"}}
    r=tmp_path/'allocation.json';r.write_text(json.dumps(a)+"\n")
    return dict(manifest=m,manifest_sha256=mh,allocation_receipt=r,
                allocation_sha256=p.sha(r.read_bytes()),output=tmp_path/'evidence')


@pytest.mark.parametrize('status',['complete','failed'])
def test_preserve_closed_bytes_and_absent_fields_without_arrays(tmp_path,status):
    args=fixture(tmp_path,status);original=args['manifest'].read_bytes()
    result=p.curate(**args);out=args['output']
    assert result['status']==status
    assert gzip.decompress((out/'manifest.json.gz').read_bytes())==original
    summary=json.loads((out/'summary.json').read_bytes())
    assert not summary['posterior_accuracy_claimed']
    assert not summary['public_raw_NPZ_or_sample_arrays_included']
    assert not list(out.glob('*.npz'))
    if status=='failed':
        assert 'default_rng_after' in summary['missing_normal_success_fields']
        assert 'default_rng_after' not in summary
    for line in (out/'SHA256SUMS').read_text().splitlines():
        h,name=line.split('  ',1);assert p.sha((out/name).read_bytes())==h


def test_running_and_bad_hash_rejected_before_output(tmp_path):
    args=fixture(tmp_path,'running')
    with pytest.raises(ValueError,match='closed'):p.curate(**args)
    assert not args['output'].exists()
    args['manifest_sha256']='0'*64
    with pytest.raises(ValueError,match='SHA mismatch'):p.curate(**args)
    assert not args['output'].exists()


def test_existing_output_never_overwritten(tmp_path):
    args=fixture(tmp_path,'failed');args['output'].mkdir();marker=args['output']/'keep';marker.write_text('unchanged')
    with pytest.raises(FileExistsError):p.curate(**args)
    assert marker.read_text()=='unchanged'


def test_complete_guard_failure_is_not_relabeled_success(tmp_path):
    args=fixture(tmp_path,'complete');d=json.loads(args['manifest'].read_bytes());d['default_rng_preserved']=False
    a=json.loads(args['allocation_receipt'].read_bytes());a['raw_manifest_sha256']=p.sha(json.dumps(d).encode())
    with pytest.raises(ValueError,match='guards'):p.validate_closed(d,a,a['raw_manifest_sha256'])


def test_npz_input_rejected_without_opening(tmp_path):
    with pytest.raises(ValueError,match='raw array archives'):p.authenticated_bytes(tmp_path/'must-not-read.npz','0'*64)
