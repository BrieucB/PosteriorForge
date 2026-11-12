"""CPU-only composition lineage and scoped routed-capture checks."""
from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest


def helper():
    path=Path(__file__).parents[1]/"scripts"/"qualify_execution_gqa_host_cache.py"
    spec=importlib.util.spec_from_file_location("gqa_host_proof_test",path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def proof():
    return {"schema":"posteriorforge.execution_gqa_hybrid_recovery.v1","status":"gqa_hybrid_screen_pass","qualified":True,
            "full21_gqa_hybrid_qualified":True,"full21_component_qualified":True,"full21_sampler_qualified":True,
            "full21_qualified":False,"cache_only_full21_qualified":False,"icdf_contract_version":2,
            "fallback_map":{"10":[5]},"stage_results":[{"stage_index":stage,"passed":True} for stage in range(21)]}


def test_new_gqa_lineage_cannot_be_replaced_by_old_two_fallback_host_proof():
    h=helper();h.validate_gqa_qualification(proof())
    for key,value in (("schema","posteriorforge.execution_hybrid_qualification.v1"),("fallback_map",{"10":[5],"13":[6]}),
                      ("full21_gqa_hybrid_qualified",False),("cache_only_full21_qualified",True)):
        bad=proof();bad[key]=value
        with pytest.raises(ValueError):h.validate_gqa_qualification(bad)


def test_actual_component_capture_routes_through_wrapper_and_checks_every_alias():
    h=helper();owner=object();seen=[]
    descriptor={key:True for key in ("original_alias_restored","original_function_restored","original_code_unchanged","active_code_unchanged","gqa_source_unchanged","sdk_sources_unchanged")}
    descriptor["expanded_attention_calls"]=12
    components=[{"gqa_attention_descriptor":deepcopy(descriptor)} for _ in range(8)]
    def routed(engine,query):seen.append(engine);return {"component_predictions":components}
    result=h.gqa_prediction(owner,SimpleNamespace(hybrid_prediction=routed),None)
    assert seen==[owner] and result["gqa_execution"]=={"all_predictions_restored":True,"component_predictions":8,"expanded_attention_calls":96}
    components[3]["gqa_attention_descriptor"]["original_alias_restored"]=False
    with pytest.raises(RuntimeError,match="every component"):h.gqa_prediction(owner,SimpleNamespace(hybrid_prediction=routed),None)


def test_frozen_numerical_and_residency_spine_sources_still_authenticate():
    h=helper();base,routed=h.frozen_helpers()
    assert base.THRESHOLDS=={"normalization_absolute":1e-5,"max_cdf_absolute":2e-5,"max_row_total_variation":1e-4,"quantile_coordinate_span_fraction":2e-6}
    import hashlib
    original=Path(h.__file__).with_name("qualify_execution_host_cache.py")
    assert hashlib.sha256(original.read_bytes()).hexdigest()==h.BASE_HOST_QUALIFIER_SHA256
    warm=Path(h.__file__).with_name("qualify_execution_host_cache_warmed.py")
    assert hashlib.sha256(warm.read_bytes()).hexdigest()==h.WARM_HELPER_SHA256


def test_helper_requires_all21_coverage():
    bad=proof();bad["stage_results"].pop()
    with pytest.raises(ValueError,match="all21"):helper().validate_gqa_qualification(bad)
