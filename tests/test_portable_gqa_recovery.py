"""CPU portable replay tests; no model, CUDA or reference integration."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).parents[1]
spec=importlib.util.spec_from_file_location("portable_gqa_recovery_tests",ROOT/"scripts/verify_portable_gqa_recovery.py")
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)


def time_fixture():
    golden={"git_revision":"golden","created_unix":200.}
    raw={"git_revision":"raw","created_unix":300.}
    history={"recorded_source_revisions":[{"git_revision":d["git_revision"],"created_unix":d["created_unix"],
        "wheel_authentication_bytes_equal":True,"inference_lock_bytes_equal":True} for d in (golden,raw)],
        "installed_function_ctime_ns":100_000_000_000,"installed_function_mtime_ns":99_000_000_000}
    return {"historical_authentication":history},raw,golden


def test_import_and_cli_are_torch_free():
    subprocess.run([sys.executable,"-c",
        "import importlib.util,sys;s=importlib.util.spec_from_file_location('p','scripts/verify_portable_gqa_recovery.py');"
        "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);assert 'torch' not in sys.modules"],
        cwd=ROOT,check=True)
    help_text=subprocess.run([sys.executable,str(Path(v.__file__)),"--help"],capture_output=True,text=True,check=True).stdout
    assert "--sdk-root" in help_text and "--recovery-manifest" in help_text


def test_only_archived_timestamps_are_checked_and_missing_RECORD_time_is_explicit(monkeypatch):
    evidence,raw,golden=time_fixture()
    monkeypatch.setattr(Path,"stat",lambda *a,**k:(_ for _ in ()).throw(AssertionError("fresh file timestamps must not be read")))
    report=v.historical_time_check(evidence,raw,golden)
    assert not report["fresh_install_timestamps_checked"]
    assert not report["cached_RECORD_numeric_ctime_archived"]
    assert "no numeric RECORD timestamp was archived" in report["cached_RECORD_historical_check_basis"]


@pytest.mark.parametrize("bad",["late_ctime","late_mtime","missing_time","boolean_time","history","success"])
def test_historical_timing_and_recorded_revision_evidence_is_not_waived(bad):
    evidence,raw,golden=time_fixture();h=evidence["historical_authentication"]
    if bad=="late_ctime":h["installed_function_ctime_ns"]=200_000_000_000
    elif bad=="late_mtime":h["installed_function_mtime_ns"]=300_000_000_000
    elif bad=="missing_time":h.pop("installed_function_ctime_ns")
    elif bad=="boolean_time":h["installed_function_ctime_ns"]=True
    elif bad=="history":h["recorded_source_revisions"][0]["git_revision"]="different"
    else:h["recorded_source_revisions"][0]["wheel_authentication_bytes_equal"]=False
    with pytest.raises(ValueError):v.historical_time_check(evidence,raw,golden)


def test_archived_input_must_be_inside_bundle_and_byte_equal(tmp_path):
    inside=tmp_path/"inside";inside.mkdir();p=inside/"record.csv";p.write_bytes(b"saved RECORD")
    entry={"file":p.name,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()}
    assert v.archived_bytes(inside,entry)==b"saved RECORD"
    p.write_bytes(b"tampered")
    with pytest.raises(ValueError):v.archived_bytes(inside,entry)
    outside=tmp_path/"outside";outside.write_bytes(b"other")
    with pytest.raises(ValueError):v.archived_bytes(inside,{"file":"../outside","sha256":v.sha(outside)})


@pytest.fixture(scope="module")
def actual():
    phase=Path("/scratch/project/eu-26-17/eubrieucb/posteriorforge/results/performance-first-v1")
    recovery=phase/"gqa-hybrid-qualification-recovered-01/manifest.json"
    batch=phase/"gqa-large-batch-qualification-01/manifest.json"
    sdk=Path("/scratch/project/eu-26-17/eubrieucb/posteriorforge/envs/inference/lib/python3.11/site-packages/tabpfn")
    if not recovery.is_file() or not batch.is_file() or not sdk.is_dir():pytest.skip("authenticated HPC CPU artifacts not available")
    document=json.loads(recovery.read_text());producer=json.loads(batch.read_text())
    raw=json.loads(v.archived_bytes(recovery.parent,document["recovery_inputs"]["raw_gpu_proof"]))
    golden=json.loads(v.archived_bytes(recovery.parent,document["recovery_inputs"]["golden_manifest"]))
    helper=v.load_helper("recover_gqa_hybrid_qualification.py",v.RECOVERY_SOURCE_SHA256)
    return recovery,batch,sdk,document,producer,raw,golden,helper


def relocate(actual,tmp_path):
    recovery,batch,sdk,document,producer,*_=actual
    new_sdk=tmp_path/"fresh-sdk"/"tabpfn"
    new_recovery=tmp_path/"downloaded-recovery"/"manifest.json"
    new_batch=tmp_path/"downloaded-batch"/"manifest.json"
    new_recovery.parent.mkdir(parents=True);new_batch.parent.mkdir(parents=True)
    shutil.copyfile(recovery,new_recovery);shutil.copyfile(batch,new_batch)
    for entry in document["recovery_inputs"].values():
        target=new_recovery.parent/entry["file"];target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(recovery.parent/entry["file"],target)
    for entry in producer["portable_source_authentication"]["archives"].values():
        target=new_batch.parent/entry["file"];target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(batch.parent/entry["file"],target)
    for name in producer["portable_source_authentication"]["sdk_source_sha256"]:
        relative=Path(name).relative_to("tabpfn");target=new_sdk/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(sdk/relative,target)
    return new_recovery,new_batch,new_sdk


def test_actual_complete184_decision_replay_with_new_SDK_and_artifact_paths(actual,tmp_path,monkeypatch):
    recovery,batch,sdk=relocate(actual,tmp_path)
    document=actual[3];first=min(actual[5]["created_unix"],actual[6]["created_unix"])
    assert (sdk/"model/preprocessing.py").stat().st_ctime>first
    old_roots=[Path(actual[2]),Path(document["configuration_identity_sdk_evidence"]["sdk_root"])]
    auth_entry=actual[4]["portable_source_authentication"]["archives"]["historical_wheel_authentication"]
    auth=json.loads(v.archived_bytes(actual[1].parent,auth_entry));old_roots.append(Path(auth["cached_extracted_wheel_root"]))
    # Make all historical installed/cached source paths unreadable to this test.
    for name in ("read_bytes","read_text","stat"):
        original=getattr(Path,name)
        def guarded(self,*args,_original=original,**kwargs):
            if any(self.is_relative_to(root) for root in old_roots):
                raise AssertionError("old machine SDK/cache path was read")
            return _original(self,*args,**kwargs)
        monkeypatch.setattr(Path,name,guarded)
    original_load=v.load_helper
    def load(name,expected):
        module=original_load(name,expected)
        if name=="recover_gqa_hybrid_qualification.py":
            module.sdk_evidence=lambda *a,**k:(_ for _ in ()).throw(AssertionError("old machine verifier must not be called"))
        return module
    monkeypatch.setattr(v,"load_helper",load)
    result=v.verify_portable_recovery(recovery,batch,sdk_root=sdk)
    assert result["status"]=="verified_existing_recovery"
    assert result["replay"]["existing_narrow_identity_decisions_replayed"]==184
    assert result["replay"]["entire_recovery_document_equal"] and result["replay"]["raw_FAILED_status_preserved"]
    assert result["all22_current_SDK_source_bytes_and_archived_RECORD_verified"]
    assert not result["current_install_ctime_checked"] and not result["old830_helper_claimed_portable"]
    assert result["model_calls"]==result["gpu_calls"]==result["reference_integrations"]==0


@pytest.mark.parametrize("bad",["decision","numeric","state","RNG","route","status"])
def test_actual_full_original_predicate_replay_rejects_decision_and_other_failures(actual,bad):
    _,_,_,document,_,raw,golden,helper=actual
    doc=deepcopy(document);source=deepcopy(raw)
    if bad=="decision":doc["configuration_identity_corrected_paths"].pop()
    elif bad=="numeric":source["stage_results"][0]["comparisons"][0]["candidate_vs_fresh_control1024_slice"]["final"]["max_cdf_absolute_error"]=2.1e-5
    elif bad=="state":source["stage_results"][0]["prepared_state_unchanged"]=False
    elif bad=="RNG":source["stage_results"][0]["rng"][0]["after"]="different"
    elif bad=="route":source["stage_results"][10]["member_routes"]["5"]="cached"
    else:source["status"]="screen_pass"
    with pytest.raises(ValueError):v.replay_exact_document(doc,source,golden,helper)


def test_actual_historical_Git_lineage_tamper_rejected(actual,monkeypatch):
    recovery,batch,sdk,document,producer,raw,golden,helper=actual
    monkeypatch.setattr(v.subprocess,"run",lambda *a,**k:SimpleNamespace(stdout=b"modified historical Git source"))
    with pytest.raises(ValueError,match="historically recorded Git"):
        v.historical_source_check(document["configuration_identity_sdk_evidence"],raw,golden,
            batch_root=batch.parent,bundle=producer["portable_source_authentication"],sdk_root=sdk,helper=helper)


@pytest.mark.parametrize("bad",["SDK","RECORD","recovery"])
def test_actual_relocated_source_or_historical_archive_tamper_rejected(actual,tmp_path,bad):
    recovery,batch,sdk=relocate(actual,tmp_path)
    if bad=="SDK":target=sdk/"model/preprocessing.py"
    elif bad=="RECORD":target=batch.parent/actual[4]["portable_source_authentication"]["archives"]["wheel_record"]["file"]
    else:target=recovery
    target.write_bytes(target.read_bytes()+b"\n")
    with pytest.raises((ValueError,RuntimeError)):v.verify_portable_recovery(recovery,batch,sdk_root=sdk)
