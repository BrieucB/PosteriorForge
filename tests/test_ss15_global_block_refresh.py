"""Strict50000 toy CPU fixtures only; no saved cohort is refreshed by these tests."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import random
import subprocess
import sys
import numpy as np
import pytest
from posteriorforge.artifacts import sha256_file
from posteriorforge.science.analytical_benchmark_v2 import array_descriptor
from posteriorforge.banks import derive_seed

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("joint_refresh_producer",ROOT/"scripts/refresh_ss15_global_block.py")
producer=importlib.util.module_from_spec(spec);spec.loader.exec_module(producer)


def values(count=50000):
    """Distinct supported coordinates; engineering numbers, never inference evidence."""
    result=np.repeat(np.arange(21,dtype=np.float64)[None]/20,count,axis=0)
    result+=np.linspace(-.1,.1,count)[:,None]
    result[:,producer.kernel.metadata()["noise_indices"]]=.25
    result[0,4]=-0.
    return result


def parent_manifest(data,graph=True):
    metadata=producer.kernel.metadata()
    schema="posteriorforge.inference.ss15_graph_local.v1" if graph else "posteriorforge.inference.ss15_fp16.v1"
    variant,source,_,_=producer.PARENT_FAMILIES[schema]
    raw={"checkpoint_sha256":producer.CHECKPOINT_SHA,"scientific_fingerprint":metadata["scientific_fingerprint"],
         "parameter_order":metadata["physical_parameter_order"],
         "context_theta":{"dtype":"<f8","shape":[4096,21],"sha256":"a"*64},
         "context_obs":{"dtype":"<f8","shape":[4096,33],"sha256":"b"*64},
         "query":{"dtype":"<f8","shape":[1,33],"sha256":"c"*64}}
    result={"schema":schema,"variant":variant,"status":"complete","requested_count":50000,
            "context_seed":20261111,"case":0,"context_size":4096,"repetition":0,
            "runner_source_sha256":source,"samples":array_descriptor(data),"samples_file":"samples.npz",
            "samples_sha256":producer.CONTROL_PAYLOAD_SHA if not graph else "d"*64,
            "sources_unchanged":True,"default_rng_preserved":True,"completed_stage_indices":list(range(21)),
            "clipping_or_repair_performed":False,"partial_vector_filtering":False,
            "raw_scientific_bindings":raw,**raw}
    if graph:result.update(preregistration_sha256=producer.CONTRACT_SHA,
                           graph_local_input_policy={"module_sha256":producer.GRAPH_MASK_SHA,"policy_sha256":"e"*64},
                           actual_graph_local_fixed_conditional_screen_performed=False)
    return result


def test_strict_quota_one_correlated_rowmajor_draw_and_all18_columns_defaultRNG_replay():
    data=values();before=data.copy()
    np.random.seed(938);default=np.random.get_state();python=random.getstate()
    seed=derive_seed(20261111,producer.ROLE,case=0,repetition=0)
    output,receipt=producer.refresh_once(data,seed)
    assert output.shape==(50000,21) and output.dtype==np.float64
    assert output[:,3:].tobytes()==before[:,3:].tobytes() and data.tobytes()==before.tobytes()
    assert receipt["standard_normal_draw_shape"]==[50000,3] and receipt["standard_normal_draw_count"]==150000
    assert receipt["independent_whole_matrix_draw_replay_passed"] and receipt["all18_columns_byte_identical"]
    assert receipt["global_block_refresh_count_per_row"]==1 and receipt["default_numpy_python_RNG_preserved"]
    generator=np.random.Generator(np.random.PCG64(seed))
    mean,covariance=producer.kernel.conditional_parameters(data)
    normals=generator.standard_normal((50000,3))
    assert output[:,:3].tobytes()==(mean+normals@np.linalg.cholesky(covariance).T).tobytes()
    assert receipt["standard_normal_draws"]==array_descriptor(normals)
    assert receipt["final_private_state"]==generator.bit_generator.state
    np.testing.assert_array_equal(np.random.get_state()[1],default[1])
    assert np.random.get_state()[2:]==default[2:] and random.getstate()==python


@pytest.mark.parametrize("count",[1,49999,50001])
def test_reduced_or_changed_scientific_quota_cannot_refresh(count):
    with pytest.raises(ValueError,match="quota"):producer.refresh_once(values(count),1)


@pytest.mark.parametrize("fault",["float32","nonfinite","unsupported"])
def test_input_invalidity_fails_instead_of_filtering_or_repair(fault):
    data=values()
    if fault=="float32":data=data.astype(np.float32)
    elif fault=="nonfinite":data[0,1]=np.nan
    else:data[0,6]=.001
    before=data.copy()
    with pytest.raises(ValueError):producer.refresh_once(data,12)
    assert data.tobytes()==before.tobytes()


def test_local_column_corruption_is_detected_even_if_global_draws_are_correct(monkeypatch):
    original=producer.kernel.refresh
    def corrupt(data,rng):
        result=original(data,rng);result[0,3]+=1.;return result
    monkeypatch.setattr(producer.kernel,"refresh",corrupt)
    with pytest.raises(ValueError,match="18 local/noise"):producer.refresh_once(values(),13)


@pytest.mark.parametrize("graph",[False,True])
def test_both_pure_parent_headers_are_supported_without_observation_inputs(graph,monkeypatch):
    data=values();metadata=producer.kernel.metadata();manifest=parent_manifest(data,graph)
    if not graph:monkeypatch.setattr(producer,"CONTROL_ARRAY_SHA",manifest["samples"]["sha256"])
    receipt=producer.validate_parent(manifest,data,metadata)
    assert receipt["family"]==("pure_graph_local" if graph else "pure_ss15_control")


@pytest.mark.parametrize("fault",["scalar","already_joint","source","case","count","raw","graphmask",
                                   "claimedqualification","defaultRNG","refinedscope"])
def test_wrong_or_hybrid_parents_and_mutated_provenance_are_rejected(fault):
    data=values();manifest=parent_manifest(data)
    if fault=="scalar":manifest["schema"]="posteriorforge.ss15_mu_a_refresh.v1"
    elif fault=="already_joint":manifest["schema"]=producer.SCHEMA
    elif fault=="source":manifest["runner_source_sha256"]="0"*64
    elif fault=="case":manifest["case"]=1
    elif fault=="count":manifest["requested_count"]=49999
    elif fault=="raw":manifest["raw_scientific_bindings"]["query"]["sha256"]="bad"
    elif fault=="graphmask":manifest["graph_local_input_policy"]["module_sha256"]="0"*64
    elif fault=="claimedqualification":manifest["actual_graph_local_fixed_conditional_screen_performed"]=True
    elif fault=="defaultRNG":manifest["default_rng_preserved"]=False
    else:manifest["raw_scientific_bindings"].pop("context_obs")
    with pytest.raises(ValueError):producer.validate_parent(manifest,data,producer.kernel.metadata())


def test_explicit_expected_manifest_hash_checked_before_loading_payload(tmp_path):
    path=tmp_path/"manifest.json";path.write_text("{}")
    for expected in (None,"","not-a-sha","0"*64):
        with pytest.raises(ValueError):producer.load_parent(path,expected,producer.kernel.metadata())


def test_unpinned_kernel_stops_before_parent_io_or_output_creation(tmp_path,monkeypatch):
    monkeypatch.setattr(producer,"KERNEL_SHA",None)
    with pytest.raises(ValueError,match="unpinned"):producer.run(tmp_path/"missing.json","0"*64,tmp_path/"output")
    assert not (tmp_path/"output").exists()


def test_real_pure_control_parent_authenticates_without_refresh_or_output():
    root=Path("/scratch/project/eu-26-17/eubrieucb/posteriorforge/results/accuracy-ss15-v1")
    candidate=next((p for p in root.glob("**/manifest.json")
                    if sha256_file(p)==producer.CONTROL_MANIFEST_SHA),None) if root.is_dir() else None
    if candidate is None:pytest.skip("immutable pure control unavailable")
    before=sha256_file(candidate)
    manifest,data,lineage=producer.load_parent(candidate,producer.CONTROL_MANIFEST_SHA,producer.kernel.metadata())
    assert data.shape==(50000,21) and lineage["family"]=="pure_ss15_control"
    assert lineage["samples_sha256"]==producer.CONTROL_PAYLOAD_SHA
    assert sha256_file(candidate)==before and manifest["schema"]=="posteriorforge.inference.ss15_fp16.v1"


def test_import_and_help_are_torch_free_and_expose_no_observation_or_reference_arguments():
    source=ROOT/"scripts/refresh_ss15_global_block.py"
    code=("import importlib.util,sys;s=importlib.util.spec_from_file_location('r',"+repr(str(source))+");"
          "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
          "assert 'torch' not in sys.modules;assert m.ROLE=='ss15_global_block_gibbs_once_v1'")
    subprocess.run([sys.executable,"-c",code],check=True)
    result=subprocess.run([sys.executable,str(source),"--help"],check=True,capture_output=True,text=True)
    assert "--parent-manifest" in result.stdout and "--expected-parent-sha256" in result.stdout
    assert "--reference" not in result.stdout and "--observation" not in result.stdout and "--truth" not in result.stdout
    assert producer.KERNEL_SHA==producer.kernel.source_sha256()
