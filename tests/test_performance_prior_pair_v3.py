"""CPU-only source-custody checks; no neural construction, query, or RNG draw."""
import copy
from types import FunctionType
from unittest.mock import patch
import pytest
from posteriorforge import performance_prior_pair_v3 as p


def bound_fixture():
    # Exercise the actual bound guard before commit; only the committed-HEAD
    # precondition is replaced with the same checkout-exact integrity operation.
    ns=dict(p._bind_run.__globals__)
    ns['require_sources']=lambda:p._integrity(require_committed=False)
    bind=FunctionType(p._bind_run.__code__,ns,p._bind_run.__name__,p._bind_run.__defaults__,p._bind_run.__closure__)
    return bind(p.pairs.PairSpec(20261111,0),'a'*64,'fixture-only')


def test_own_pair_registry_authenticates_only_fresh_all21_v3_screen():
    import gzip,hashlib,json
    from pathlib import Path
    proof='56ec6cdc81eeab4c755d24fa8ddae47c21f75bce4c624c43908a59f5da21ffd1'
    assert p.QUALIFICATION_SHA256_BY_PAIR=={(20261111,0):proof}
    assert p.qualification_sha256(p.pairs.PairSpec(20261111,0))==proof
    assert proof not in p.v2.QUALIFICATION_SHA256_BY_PAIR.values()
    path=Path(p.__file__).resolve().parents[2]/'evidence/accuracy-prior-coordinates-v3/screen-pass-5411689/manifest.json.gz'
    raw=gzip.decompress(path.read_bytes());assert hashlib.sha256(raw).hexdigest()==proof
    document=json.loads(raw)
    assert document['schema']==p.QUALIFICATION_SCHEMA and document['status']==p.QUALIFICATION_STATUS
    assert document['matching_v3_execution_passed_all21'] is True and document['v3_failed_stages']==[]
    assert document['replay']['passed'] is True and document['replay']['posterior_equivalence_claimed'] is False
    assert [r['stage_index'] for r in document['stage_results']]==list(range(21))
    assert all(r['hard_validity_passed'] and r['matching_v3_execution_passed'] for r in document['stage_results'])
    assert document['pair']==p.pairs.PairSpec(20261111,0).as_dict()
    assert document['sources']['qualifier']==p.DEPENDENCY_SHA256['scripts/'+p.QUALIFIER_FILENAME]
    assert document['sources']['pair_CPU_receipt']==p.PAIR_CPU_RECEIPT_SHA256
    with pytest.raises(RuntimeError,match='fresh own-pair v3 all21 qualification'):
        p.qualification_sha256(p.pairs.PairSpec(20261111,1))


def test_exact_training_aliases_and_declared_source_transforms():
    assert p.model_bindings is p.v2.model_bindings and p.codec is p.v2.codec
    d=p.derivation_descriptor()
    assert d['reverse_run_AST_exact'] and d['sampler']['reverse_AST_exact']
    assert d['sampler']['only_import_module_changed']
    assert d['sampler']['all_RNG_calls_seed_roles_uniform_assignments_head_retry_and_F32_draw_buffer_unchanged']
    assert d['sampler']['head_acceptance_callable_is_exact_e71']
    assert not d['old_function_globals_mutated'] and not d['old_failed_wave_recovered']


def test_checkout_exact_pins_and_source_seals():
    assert p._integrity(require_committed=False)
    for namespace,key in [(vars(p._CLEAN_PROXY),'frozen_sampling'),(p._SAMPLER.__globals__,'np'),
                           (p._RUN_TEMPLATE.__globals__,'backend')]:
        old=namespace[key]
        try:
            namespace[key]=object()
            with pytest.raises(RuntimeError,match='binding changed'):
                p._integrity(require_committed=False)
        finally:namespace[key]=old
    with patch.object(p.engine_scope,'__wrapped__',lambda:None):
        with pytest.raises(RuntimeError,match='context generator changed'):
            p._integrity(require_committed=False)
    with patch.object(p.StageAdapter,'prepare',lambda *a:None):
        with pytest.raises(RuntimeError,match='adapter prototype changed'):
            p._integrity(require_committed=False)


def test_private_run_adapter_context_and_checker_reject_tampering():
    function=bound_fixture();namespace=function.__globals__;guard=namespace['require_sources']
    assert guard()
    scope=namespace['engine_scope'];adapter_ns=scope.__wrapped__.__globals__
    cls=adapter_ns['StageAdapter']
    mutations=[(namespace,'backend'),(adapter_ns,'probability')]
    for ns,key in mutations:
        old=ns[key]
        try:
            ns[key]=object()
            with pytest.raises(RuntimeError,match='binding changed'):guard()
        finally:ns[key]=old
    with patch.object(cls,'prepare',lambda *a:None):
        with pytest.raises(RuntimeError,match='adapter class binding changed'):guard()
    with patch.object(scope,'__wrapped__',lambda:None):
        with pytest.raises(RuntimeError,match='private v3 context changed'):guard()
    original=namespace['validate_prediction'].__code__
    try:
        namespace['validate_prediction'].__code__=(lambda *a,**k:None).__code__
        with pytest.raises(RuntimeError,match='prediction checker changed'):guard()
    finally:namespace['validate_prediction'].__code__=original
    assert guard()
