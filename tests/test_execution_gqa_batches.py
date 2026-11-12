"""CPU-only all-row mapping, raw mass, actual shape and portable source checks."""
from pathlib import Path
import base64
import csv
import hashlib
import importlib.util
import numpy as np
import pytest

path=Path(__file__).resolve().parents[1]/"scripts/qualify_execution_gqa_batches.py"
spec=importlib.util.spec_from_file_location("gqa_batch_tests",path);b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
utilities=b.load_helper("qualify_execution_cache.py",b.BASE_SHA)


@pytest.mark.parametrize("rows",[4096,6250,10000])
def test_repeated_pool_keeps_every_main_and_stress_row_in_exact_order(rows):
    indices=b.repeated_pool_indices(rows)
    np.testing.assert_array_equal(indices[:1032],np.arange(1032))
    np.testing.assert_array_equal(indices[1024:1032],np.arange(1024,1032))
    assert len(indices)==rows and indices[-1]==(rows-1)%1032
    assert np.bincount(indices,minlength=1032).min()>=3


@pytest.mark.parametrize("rows",[1024,5000,True,6250.])
def test_actual_neural_batch_sizes_cannot_be_silently_lowered(rows):
    with pytest.raises(RuntimeError):b.repeated_pool_indices(rows)


def test_direct_comparison_covers_last_row_and_never_exceeds1024_row_blocks():
    p=np.linspace(.1,.9,1032,dtype=np.float32);reference=np.column_stack((p,1-p))
    indices=b.repeated_pool_indices(10000);candidate=reference[indices].copy()
    passed=b.compare_blocks(reference,candidate,indices,utilities)
    assert passed["passed"] and passed["rows_compared"]==10000 and passed["blocks"]==10
    candidate[-1]=[0.,1.]
    failed=b.compare_blocks(reference,candidate,indices,utilities)
    assert not failed["passed"] and failed["rows_compared"]==10000 and failed["worst_cdf_row"]==9999
    with pytest.raises(RuntimeError):b.compare_blocks(reference,candidate,indices,utilities,block_rows=2048)


def test_raw_mass_normalization_is_never_silently_repaired():
    reference=np.array([[.2,.8]],np.float32);candidate=reference*1.1
    result=b.compare_blocks(reference,candidate,np.array([0]),utilities)
    assert not result["passed"] and result["max_normalization_error"]>.09


def test_mass_weighted_actual_component_mixture_is_measured_directly():
    acc=b.probability_accumulators(1,2)
    b.accumulate_component(acc,np.array([[.8,.2]],np.float32),np.array([[.8,.2]],np.float32),np.array([1.]))
    b.accumulate_component(acc,np.array([[.04,.16]],np.float32),np.array([[.2,.8]],np.float32),np.array([.2]))
    ideal=acc["raw_sum"]/acc["mass_sum"][:,None];actual=acc["weighted_output_sum"]/acc["mass_sum"][:,None]
    np.testing.assert_allclose(ideal,[[.7,.3]],rtol=0,atol=2e-8)
    np.testing.assert_allclose(actual,[[.7,.3]],rtol=0,atol=2e-8)
    assert not utilities.compare_probabilities(np.array([[.7,.3]]),np.array([[.5,.5]]))["passed"]


def test_neural_query_shape_is_observed_not_inferred_from_input_cap():
    descriptor={"head_shape_histogram":{str({"q":[50,4096,6,32],"qkv":None}):12,str({"q":[50,6250,6,32],"qkv":None}):12}}
    assert b.shape_evidence(descriptor,6250)["passed"]
    assert not b.shape_evidence(descriptor,10000)["passed"]


def test_reference_pool_concatenation_keeps_all8_members_and_row_order():
    torch=pytest.importorskip("torch")
    class Criterion:borders=torch.tensor([0.,1.,3.])
    def prediction(rows,offset):
        logits=torch.arange(offset,offset+rows*2,dtype=torch.float32).reshape(rows,2)
        return {"logits":logits,"criterion":Criterion(),"members":[{"raw_logits":logits+i,"translated_probabilities":logits+i,
            "from_borders":Criterion.borders,"to_borders":Criterion.borders,"configuration_identity":{"index":i}} for i in range(8)]}
    a=prediction(3,0);s=prediction(2,100);pool=b.concatenate_reference(a,s,torch)
    assert pool["logits"].shape==(5,2) and pool["members"][7]["raw_logits"][3,0]==107
    s["members"][6]["configuration_identity"]={"index":5}
    with pytest.raises(RuntimeError):b.concatenate_reference(a,s,torch)


def test_portable_sdk_source_authentication_accepts_relocated_identical_bytes_without_ctime(tmp_path):
    root=tmp_path/"artifact";root.mkdir();(root/"source-auth").mkdir();sdk=tmp_path/"fresh-install";sdk.mkdir()
    content=b"def same_source(x): return x\n";(sdk/"module.py").write_bytes(content)
    digest=hashlib.sha256(content).hexdigest();encoded=base64.urlsafe_b64encode(bytes.fromhex(digest)).decode().rstrip("=")
    record=root/"source-auth/RECORD.csv"
    with record.open("w") as f:csv.writer(f).writerow(["tabpfn/module.py","sha256="+encoded,str(len(content))])
    bundle={"archives":{"wheel_record":{"file":"source-auth/RECORD.csv","sha256":b.sha(record)}},"sdk_source_sha256":{"tabpfn/module.py":digest}}
    assert b.verify_portable_sources(root,bundle,sdk_root=sdk)
    moved=tmp_path/"another-sdk";moved.mkdir();(moved/"module.py").write_bytes(content)
    assert b.verify_portable_sources(root,bundle,sdk_root=moved)
    (moved/"module.py").write_bytes(content+b"# changed")
    with pytest.raises(RuntimeError,match="SDK source mismatch"):b.verify_portable_sources(root,bundle,sdk_root=moved)


def test_import_and_cli_help_are_torch_free():
    import subprocess,sys
    output=subprocess.run([sys.executable,str(path),"--help"],check=True,capture_output=True,text=True)
    assert "--qualification" in output.stdout and "--max-seconds" in output.stdout


def test_oom_or_numeric_failure_at_one_size_does_not_promote_or_discard_other_sizes():
    rows=[{"stage_index":stage,"normalization_train_only_verified":True,"all8_preprocessor_identity":True,"final_prepared_state_unchanged":True,
        "batch_results":[{"query_rows":n,"passed":True} for n in b.BATCH_SIZES]} for stage in range(21)]
    rows[7]["batch_results"][2].update(status="cuda_out_of_memory",passed=False)
    assert b.full21_batch_outcomes(rows,list(range(21)))=={"4096":True,"6250":True,"10000":False}
    rows[13]["batch_results"][1].update(status="conditional_mismatch",passed=False)
    assert b.full21_batch_outcomes(rows,list(range(21)))=={"4096":True,"6250":False,"10000":False}
    assert not any(b.full21_batch_outcomes(rows[:-1],list(range(20))).values())
