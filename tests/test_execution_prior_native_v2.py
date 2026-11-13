"""Focused CPU route/namespace/source tests; no model calls or GPU."""
from contextlib import contextmanager
from types import SimpleNamespace
import pytest
from posteriorforge import execution_prior_native_v2 as p


def test_exact_new_policy_and_unchanged_v1_namespace():
    assert p.NATIVE_STAGES==(6,10,14) and p.SLOTS==(5,9,13)
    assert p.original.NATIVE_STAGES==(6,10) and p.original.NATIVE_PHYSICAL_SLOTS==(5,9)
    policy=p.policy_metadata();metadata=p.execution_metadata()
    assert policy['native_stages']==[6,10,14] and len(policy['other_stages'])==18
    assert policy['same_execution_candidate_as_v1'] is False
    assert policy['selection_artifacts_are_v2_proof'] is False
    assert metadata['fresh_v2_all21_pair_screen_required'] and not metadata['v1_or_case2_numerical_pass_inherited']
    assert metadata['cached_stage_count']==18 and not metadata['performance_improvement_claimed']
    assert metadata['derivation']['reverse_AST_exact'] and not metadata['derivation']['numerical_method_bodies_changed']
    assert p._PRIVATE is not vars(p.original)
    assert p.PriorNativeStageExecutor.prepare.__globals__ is p._PRIVATE
    assert p.PriorNativeStageExecutor is not p.original.PriorNativeStageExecutor


def test_all21_routes_and_exact_unchanged_numerical_helpers():
    for stage in range(21):assert p.route(stage)==('original_fit_preprocessors_all8' if stage in (6,10,14) else 'frozen88_resident_cache')
    for name in ['stage_policy','mask_training','mask_query','training_bindings','codec_module']:
        assert getattr(p,name) is getattr(p.original,name)
    assert p._PRIVATE['_BODY'] is p.original._BODY
    assert p._PRIVATE['_MODEL_RNG'] is p.original._MODEL_RNG
    p.require_sources()


def test_new_native14_has_no_neural_prefill_claim():
    report={'mode':p.CONTROL_MODE,'module_sha256':p.CACHED_SHA256,'codec_sha256':p.CODEC_SHA256,'counters':{'qkv_calls':0,'attention_calls':0},'witnesses':[]}
    prepared={'route':'original_fit_preprocessors_all8','original_all8_lists_verified':True,'native_neural_prefill_calls':0,'native_KV_cache_used':False}
    witness=p.preparation_witness(report,14,prepared)
    assert witness['passed'] and not witness['cached_prefill_claimed']
    report['counters']['qkv_calls']=1
    with pytest.raises(RuntimeError,match='must not contain a neural prefill'):p.preparation_witness(report,14,prepared)


def test_native14_query_views_restore_on_exception_without_model():
    events=[]
    @contextmanager
    def forward_scope(phase,stage,member):
        events.append((phase,stage,member));yield
    session=SimpleNamespace(query_views=True,forward_scope=forward_scope)
    with pytest.raises(ValueError):
        with p._PRIVATE['original_native_query_scope'](session,14,5):
            assert session.query_views is False
            raise ValueError('body failure')
    assert session.query_views is True and events==[('query',14,5)]


@pytest.mark.parametrize('name,value',[('NATIVE_STAGES',(6,10)),('NATIVE_PHYSICAL_SLOTS',(5,9)),('np',SimpleNamespace())])
def test_private_source_config_mutation_fails_closed(monkeypatch,name,value):
    monkeypatch.setitem(p._PRIVATE,name,value)
    with pytest.raises(RuntimeError,match='private source/config'):p.require_sources()


@pytest.mark.parametrize('name',['preparation_witness','query_witness_delta','mask_training','training_bindings','immutable_native_snapshot'])
def test_public_export_replacement_fails_closed(monkeypatch,name):
    monkeypatch.setattr(p,name,lambda *args:None)
    with pytest.raises(RuntimeError,match='exported binding'):p.require_sources()


def test_private_method_code_mutation_fails_closed_and_v1_unchanged(monkeypatch):
    original_code=p.original.PriorNativeStageExecutor.prepare.__code__
    def changed(self,*args,**kwargs):return None
    monkeypatch.setattr(p.PriorNativeStageExecutor.prepare,'__code__',changed.__code__)
    with pytest.raises(RuntimeError,match='sealed callable'):p.require_sources()
    assert p.original.PriorNativeStageExecutor.prepare.__code__ is original_code


def test_base_policy_alias_and_code_are_sealed(monkeypatch):
    def changed():return {}
    with monkeypatch.context() as m:
        m.setattr(p,'_BASE_POLICY',changed)
        with pytest.raises(RuntimeError,match='exported binding'):p.require_sources()
    with monkeypatch.context() as m:
        m.setattr(p._BASE_POLICY,'__code__',changed.__code__)
        with pytest.raises(RuntimeError,match='sealed callable'):p.require_sources()


def test_no_building_bypass_and_ancestral_seal_contents_protected(monkeypatch):
    monkeypatch.setattr(p,'_BUILDING',True,raising=False)
    monkeypatch.setitem(p._PRIVATE,'NATIVE_STAGES',(6,10))
    with pytest.raises(RuntimeError):p.require_sources()


def test_policy_derivation_mutation_fails_closed(monkeypatch):
    monkeypatch.setitem(p._DERIVATION,'reverse_AST_exact',False)
    with pytest.raises(RuntimeError,match='derivation/policy'):p.require_sources()


def test_native_query_scope_binding_and_wrapped_code_are_sealed(monkeypatch):
    def changed(*args,**kwargs):return None
    with monkeypatch.context() as m:
        m.setitem(p._PRIVATE,'original_native_query_scope',changed)
        with pytest.raises(RuntimeError,match='private source/config'):p.require_sources()
    with monkeypatch.context() as m:
        m.setattr(p._SCOPE_BODY,'__code__',changed.__code__)
        with pytest.raises(RuntimeError,match='sealed callable'):p.require_sources()
