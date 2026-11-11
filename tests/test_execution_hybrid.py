"""CPU-only route, source, and exact-restoration tests; no real model calls."""
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from threading import Lock
from types import SimpleNamespace
import subprocess
import sys
import numpy as np
import pytest
from posteriorforge import execution_hybrid as hybrid


def executor_fixture():
    return SimpleNamespace(**{name:tuple(SimpleNamespace(member=index) for index in range(8)) for name in hybrid.PREPROCESSING_FIELDS})


def test_selected_preprocessing_member_restores_exact_all8_containers_on_exception():
    executor=executor_fixture();originals={name:getattr(executor,name) for name in hybrid.PREPROCESSING_FIELDS};report={}
    with pytest.raises(ValueError,match="body"):
        with hybrid.selected_preprocessing_member(executor,5,report):
            assert all(len(getattr(executor,name))==1 and getattr(executor,name)[0].member==5 for name in originals)
            raise ValueError("body")
    assert report["original_preprocessing_lists_restored"]
    assert all(getattr(executor,name) is value for name,value in originals.items())


def test_conflicting_field_is_rejected_after_original_identity_is_restored():
    executor=executor_fixture();original=executor.X_trains
    with pytest.raises(hybrid.cache.CacheExecutionError,match="changed"):
        with hybrid.selected_preprocessing_member(executor,6,{}):executor.X_trains=[]
    assert executor.X_trains is original


@pytest.mark.parametrize("mapping",[{}, {10:[]}, {10:[5,5]}, {True:[1]}, {21:[0]}, {13:[8]}, {"10":[5],"010":[6]}])
def test_invalid_undeclared_or_ambiguous_policy_fails_closed(mapping):
    with pytest.raises(ValueError):hybrid.canonical_fallback_map(mapping)


def test_policy_identity_is_canonical_and_declares_only_requested_pairs():
    assert hybrid.policy_metadata({13:[6],10:[5]})==hybrid.policy_metadata({"10":[5],"13":[6]})
    metadata=hybrid.execution_metadata({10:[5],13:[6]})
    assert metadata["fallback_map"]=={"10":[5],"13":[6]}
    assert metadata["fallback_fit_mode"]=="fit_preprocessors" and metadata["fallback_prepared_members"]==8
    assert metadata["full_mixture_predict_at_fallback_stage_supported"] is False


def test_frozen_dependencies_and_import_are_cpu_safe():
    hybrid.require_frozen_sources()
    code="import posteriorforge.execution_hybrid,sys;assert "+repr("torch")+" not in sys.modules"
    subprocess.run([sys.executable,"-c",code],check=True)


def fake_hybrid(monkeypatch,*,stage=10,raise_body=False):
    torch=pytest.importorskip("torch")
    descriptor={"patched_call_count":0,"corrected_cdf_code_sha256":"corrected","original_cdf_code_sha256":"original"}
    module=SimpleNamespace()
    def active(logits,*,frm,to):
        descriptor["patched_call_count"]+=1;descriptor["executed_corrected_cdf_code_sha256"]="corrected"
        return logits.softmax(-1)
    module.translate_probs_across_borders=active
    executor=executor_fixture();seen=[]
    class Regressor:
        executor_=executor
        def predict(self,query,**kwargs):
            seen.append((self.executor_.ensemble_configs[0].member,kwargs,query.copy()))
            logits=torch.log(torch.tensor([[.25,.75]],dtype=torch.float32).repeat(len(query),1));borders=torch.tensor([-2.,0.,1.])
            probabilities=module.translate_probs_across_borders(logits,frm=borders,to=borders)
            if raise_body:raise ValueError("predict body")
            return {"logits":probabilities.log(),"criterion":SimpleNamespace(borders=borders)}
    @contextmanager
    def model_rng():yield
    base=SimpleNamespace(stage_index=stage,query_batch_size=1024,cdf_descriptor=descriptor,_sdk_regressor=module,_model_rng=model_rng,regressor=SimpleNamespace())
    class Cached:
        def __init__(self):self.calls=[]
        def predict_component(self,index,query,**kwargs):self.calls.append(index);return {"cached_index":index}
    value=object.__new__(hybrid.HybridStageExecutor);value.base=base;value.fallback_map={10:(5,),13:(6,)};value._busy=Lock()
    value.uncached_regressor=Regressor();value.cached_component=Cached()
    value.hybrid_counters={"resident_component_predict_count":0,"uncached_predict_count":0,"uncached_query_rows":0,"original_preprocessing_lists_restored":True,"original_translator_alias_restored":True}
    monkeypatch.setattr(hybrid.cache,"_require_active_cdf",lambda module,desc:active)
    return value,module,active,executor,seen


def test_only_declared_member_uses_original_predict_and_active_corrected_translation(monkeypatch):
    value,module,active,executor,seen=fake_hybrid(monkeypatch)
    originals={name:getattr(executor,name) for name in hybrid.PREPROCESSING_FIELDS}
    cached=value.predict_component(4,np.zeros((2,43)),capture_member=True)
    assert cached["hybrid_execution_path"]=="resident_cache" and value.cached_component.calls==[4]
    prediction=value.predict_component(5,np.zeros((2,43)),capture_member=True)
    assert seen[0][0]==5 and seen[0][1]=={"output_type":"full","quantiles":[]}
    assert prediction["member"]["ensemble_index"]==5 and prediction["hybrid_execution_path"]=="original_fit_preprocessors"
    np.testing.assert_allclose(prediction["component_raw_mass"].numpy(),1)
    assert value.cdf_descriptor["patched_call_count"]==1 and module.translate_probs_across_borders is active
    assert all(getattr(executor,name) is original for name,original in originals.items())
    assert value.hybrid_counters["uncached_predict_count"]==1


def test_fallback_predict_error_restores_alias_and_all8_fields(monkeypatch):
    value,module,active,executor,seen=fake_hybrid(monkeypatch,raise_body=True)
    originals={name:getattr(executor,name) for name in hybrid.PREPROCESSING_FIELDS}
    with pytest.raises(ValueError,match="predict body"):value.predict_component(5,np.zeros((1,43)),capture_member=True)
    assert module.translate_probs_across_borders is active
    assert all(getattr(executor,name) is original for name,original in originals.items())
    assert value.hybrid_counters["original_preprocessing_lists_restored"] and value.hybrid_counters["original_translator_alias_restored"]


def test_stage13_routes_member6_only_and_rejects_unqualified_full_cached_mix(monkeypatch):
    value,_,_,_,seen=fake_hybrid(monkeypatch,stage=13)
    value.predict_component(5,np.zeros((1,46)))
    value.predict_component(6,np.zeros((1,46)))
    assert value.cached_component.calls==[5] and seen[0][0]==6
    with pytest.raises(NotImplementedError,match="full cached mixture"):value.predict(np.zeros((1,46)))
    with pytest.raises(NotImplementedError):value.log_prob(None)
