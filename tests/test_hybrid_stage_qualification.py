"""CPU qualification/provenance checks with synthetic component captures only."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
import numpy as np
import pytest

SCRIPT=Path(__file__).parents[1]/"scripts/qualify_execution_hybrid.py"
spec=importlib.util.spec_from_file_location("hybrid_qualification",SCRIPT)
qual=importlib.util.module_from_spec(spec);spec.loader.exec_module(qual)


def test_explicit_stage_member_cli_policy_and_frozen_utilities():
    assert qual.parse_fallbacks(["10:5","13:6"])=={10:(5,),13:(6,)}
    with pytest.raises(ValueError):qual.parse_fallbacks(["10:5","10:6"])
    utility=qual.frozen_utilities()
    assert utility.THRESHOLDS["max_cdf_absolute"]==2e-5
    assert utility.THRESHOLDS["max_row_total_variation"]==1e-4


def test_import_and_help_do_not_import_torch():
    code="import runpy,sys;runpy.run_path("+repr(str(SCRIPT))+",run_name="+repr("hybrid_import")+");assert "+repr("torch")+" not in sys.modules"
    subprocess.run([sys.executable,"-c",code],check=True)
    value=subprocess.run([sys.executable,str(SCRIPT),"--help"],capture_output=True,text=True,check=True)
    assert "--fallback" in value.stdout


def test_unbudgeted_hybrid_qualification_rejected_without_outputs(tmp_path,monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID",raising=False)
    with pytest.raises(RuntimeError,match="budgeted Slurm"):
        qual.qualify(banks="none",prefix_run="none",checkpoint="none",ledger="none",output=tmp_path/"out",fallback={10:[5],13:[6]})
    assert not (tmp_path/"out").exists()


def fake_capture_engine(*,bad_fingerprint=False,bad_mass=False):
    torch=pytest.importorskip("torch")
    from tabpfn.model.bar_distribution import FullSupportBarDistribution
    class Engine:
        stage_index=10;fallback_map={10:(5,),13:(6,)}
        def __init__(self):
            self.cdf_descriptor={"patched_call_count":0,"corrected_cdf_code_sha256":"corrected","original_cdf_code_sha256":"original"}
            self.hybrid_counters={"resident_component_predict_count":0,"uncached_predict_count":0,"original_preprocessing_lists_restored":True,"original_translator_alias_restored":True}
            self.cached_component=SimpleNamespace(metadata={"full8_member_lists_restored":True})
        def predict_component(self,index,query,**kwargs):
            self.cdf_descriptor["patched_call_count"]+=1
            self.cdf_descriptor["executed_corrected_cdf_code_sha256"]="wrong" if bad_fingerprint else "corrected"
            route="uncached" if index==5 else "cached"
            counter="uncached_predict_count" if index==5 else "resident_component_predict_count"
            self.hybrid_counters[counter]+=1
            p=torch.tensor([[.25,.75]],dtype=torch.float32).repeat(len(query),1)
            borders=torch.tensor([-2.,0.,1.],dtype=torch.float32)
            criterion=FullSupportBarDistribution(borders)
            return {"logits":p.log(),"criterion":criterion,"component_raw_mass":torch.full((len(query),),.999 if bad_mass else 1.,dtype=torch.float64),
                "head_mass_envelope":1.00001,"hybrid_execution_path":"original_fit_preprocessors" if index==5 else "resident_cache",
                "member":{"ensemble_index":index,"raw_logits":p.log(),"from_borders":borders,"to_borders":borders,"translated_probabilities":p,"configuration_identity":{"index":index}}}
    return Engine()


def test_hybrid_capture_records_each_actual_route_and_all8_corrected_callbacks():
    engine=fake_capture_engine();prediction=qual.hybrid_prediction(engine,np.zeros((4,43),np.float32))
    assert len(prediction["members"])==8 and engine.cdf_descriptor["patched_call_count"]==8
    assert prediction["member_routes"]=={str(index):"uncached" if index==5 else "cached" for index in range(8)}
    for item in prediction["component_predictions"]:
        route=item["qualification_route"]
        assert route["route_counter_delta"]==route["corrected_cdf_call_delta"]==1 and route["other_route_counter_delta"]==0
    result=qual.component_report(prediction,prediction,engine,qual.frozen_utilities())
    assert result["passed"] and result["fast_icdf_same_logits_failures"]==0
    for key in ("normalized_mean_captured_raw_vs_full_fp32","normalized_mean_observed_raw_vs_full_fp32","mass_corrected_observed_outputs_vs_full_fp32"):
        assert result[key]["passed"]


@pytest.mark.parametrize("failure",["fingerprint","mass"])
def test_wrong_execution_evidence_cannot_hide_behind_equal_probability_curves(failure):
    engine=fake_capture_engine(bad_fingerprint=failure=="fingerprint",bad_mass=failure=="mass")
    prediction=qual.hybrid_prediction(engine,np.zeros((3,43),np.float32))
    result=qual.component_report(prediction,prediction,engine,qual.frozen_utilities())
    assert not result["passed"]
