"""Isolated FP32 routing v2: original all8 at6/10/14, cached other18.

Built with PriorLabs-TabPFN. Fresh v2 pair proof required. V1 remains immutable.
Only explicit routing/configuration/identity metadata AST deltas are permitted;
all numerical bodies, cached delegation and original-member execution persist.
"""
from __future__ import annotations
import ast,copy,hashlib,json
from pathlib import Path
from . import execution_prior_native_v1 as original

ORIGINAL_SHA256="e5a2733e767b187e573315547ec9253267e43d1e3700596f61e97116c53f3256"
V1_PAIR_FAILURE_SHA256="fbed99d9fb256bb72a49cb03bebb820d805ac2cbafaa4ebac4deb08bf3160734"
PREREGISTRATION_SHA256="8640a4e157698862823521d3fd14396ecc3ebf121dffffabc987cc937d3ad2fc"


def _raw_source_sha256():return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _derive_namespace():
    if original.source_sha256()!=ORIGINAL_SHA256:raise RuntimeError("exact frozen e5 source required")
    tree=ast.parse(Path(original.__file__).read_text());candidate=copy.deepcopy(tree)
    assignments={
        'SCHEMA':('posteriorforge.execution.prior_native_control.v1','posteriorforge.execution.prior_native_control.v2'),
        'ENGINE_ID':('prior_latent_original_stage6_stage10_fp32_v1','prior_latent_original_stage6_stage10_stage14_fp32_v2'),
        'NATIVE_STAGES':((6,10),(6,10,14)),
        'NATIVE_PHYSICAL_SLOTS':((5,9),(5,9,13))}
    labels={original.__doc__:"Explicit FP32 routing v2: original all8 stages6/10/14, cached other18. Numerical bodies unchanged.",
        'new explicit all8 originalFP32 ordered stages6/10 only':'new explicit all8 originalFP32 ordered stages6/10/14 only',
        'other19_cached_numerical_code':'other18_cached_numerical_code'}
    def transform(document,reverse=False):
        counts={k:0 for k in assignments};label_counts={k:0 for k in labels}
        for node in document.body:
            if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in assignments:
                name=node.targets[0].id;before,after=assignments[name]
                if reverse:before,after=after,before
                if ast.literal_eval(node.value)!=before:raise RuntimeError('undeclared v2 config source: '+name)
                node.value=ast.parse(repr(after),mode='eval').body;counts[name]+=1
        mapping={v:k for k,v in labels.items()} if reverse else labels
        for node in ast.walk(document):
            if isinstance(node,ast.Constant) and isinstance(node.value,str) and node.value in mapping:
                old=node.value;node.value=mapping[old];label_counts[mapping[old] if reverse else old]+=1
        if any(v!=1 for v in counts.values()) or any(v!=1 for v in label_counts.values()):raise RuntimeError('v2 exact route/label delta counts differ')
        return counts,label_counts
    counts=transform(candidate);reverse=copy.deepcopy(candidate);transform(reverse,True)
    if ast.dump(reverse,include_attributes=False)!=ast.dump(tree,include_attributes=False):raise RuntimeError('undeclared v2 numerical AST change')
    namespace={'__name__':__name__,'__package__':__package__,'__file__':__file__}
    exec(compile(ast.fix_missing_locations(candidate),__file__+'::isolated_e5_route_config_only','exec'),namespace)
    return namespace,{'original_e5_source_sha256':ORIGINAL_SHA256,'reverse_AST_exact':True,
        'assignment_delta_counts':counts[0],'label_delta_count':sum(counts[1].values()),
        'numerical_method_bodies_changed':False,'old_module_globals_mutated':False}


_PRIVATE,_DERIVATION=_derive_namespace()
_BASE_REQUIRE=_PRIVATE['require_sources'];_BASE_POLICY=_PRIVATE['policy_metadata']
_BASE_EXECUTION=_PRIVATE['execution_metadata'];_BASE_ROUTE=_PRIVATE['route']


def _assert_integrity():
    if _raw_source_sha256()!=_IMPORTED_SOURCE_SHA256:raise RuntimeError('v2 owner source changed after import')
    if original.source_sha256()!=ORIGINAL_SHA256:raise RuntimeError('frozen e5 source changed')
    for name,expected in _CONFIG_SEALS.items():
        if _PRIVATE.get(name) is not expected:raise RuntimeError('v2 private source/config binding changed: '+name)
    for name,expected in _PUBLIC_SEALS.items():
        if globals().get(name) is not expected:raise RuntimeError('v2 exported binding changed: '+name)
    for owner,name,function,code,namespace in _CALLABLE_SEALS:
        actual=getattr(owner,name) if not isinstance(owner,dict) else owner.get(name)
        if isinstance(actual,property):actual=actual.fget
        if actual is not function or function.__code__ is not code or function.__globals__ is not namespace:
            raise RuntimeError('v2 sealed callable/global namespace changed: '+name)
    if tuple(_PRIVATE['_SEALS'])!=_ANCESTRAL_SEAL_CONTENTS:raise RuntimeError('v2 ancestral method seal inventory changed')
    if hashlib.sha256(json.dumps(_DERIVATION,sort_keys=True).encode()).hexdigest()!=_DERIVATION_DIGEST:raise RuntimeError('v2 derivation/policy metadata changed')
    if NATIVE_STAGES!=(6,10,14) or NATIVE_PHYSICAL_SLOTS!=(5,9,13):raise RuntimeError('fixed v2 native tuple changed')


def source_sha256():
    _assert_integrity();return _raw_source_sha256()


def require_sources():
    _assert_integrity()
    doc=Path(__file__).resolve().parents[2]/'docs/accuracy-prior-coordinate-native-control-v2.md'
    if hashlib.sha256(doc.read_bytes()).hexdigest()!=PREREGISTRATION_SHA256:raise RuntimeError('v2 routing preregistration changed')
    original.require_sources()
    return _BASE_REQUIRE()


def policy_metadata():
    require_sources();data=_BASE_POLICY()
    data.update(execution_routing_changed_relative_to_v1=True,same_execution_candidate_as_v1=False,
        v1_execution_source_sha256=ORIGINAL_SHA256,v2_preregistration_sha256=PREREGISTRATION_SHA256,
        selection_artifacts={'ancestral_failed_empty_cache':original.FAILED_SCREEN_SHA256,
            'v1_actual_case0_numerical_failure':V1_PAIR_FAILURE_SHA256},
        selection_artifacts_are_v2_proof=False,v1_or_case2_pass_transferred=False)
    data.pop('policy_sha256',None)
    data['policy_sha256']=hashlib.sha256(json.dumps(data,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return data


def execution_metadata():
    require_sources();data=_BASE_EXECUTION()
    data.update(execution_routing_changed_relative_to_v1=True,same_execution_candidate_as_v1=False,
        native_all8_stages=[6,10,14],cached_stage_count=18,v2_preregistration_sha256=PREREGISTRATION_SHA256,
        v1_or_case2_numerical_pass_inherited=False,fresh_v2_all21_pair_screen_required=True,
        derivation=dict(_DERIVATION),performance_improvement_claimed=False)
    return data


def route(stage):
    require_sources();return _BASE_ROUTE(stage)


_PRIVATE.update(source_sha256=source_sha256,require_sources=require_sources,
                policy_metadata=policy_metadata,execution_metadata=execution_metadata,route=route)
_EXPORT_NAMES=['cached','cache','codec','np','MethodType','SimpleNamespace','Lock','SCHEMA','ENGINE_ID','CACHED_SHA256','CODEC_SHA256','FAILED_SCREEN_SHA256',
              'CONTROL_MODE','MODE','NATIVE_STAGES','NATIVE_PHYSICAL_SLOTS','MEMBERS','PriorNativeSession',
              'PriorNativeStageExecutor','_NativeStage','preparation_witness','query_witness_delta',
              'stage_policy','mask_training','mask_query','training_bindings','codec_module','immutable_native_snapshot']
for _name in _EXPORT_NAMES:
    globals()[_name]=_PRIVATE[_name]
SLOTS=NATIVE_PHYSICAL_SLOTS
_CONFIG_SEALS={name:_PRIVATE[name] for name in ['cached','cache','codec','np','MethodType','SimpleNamespace','Lock','SCHEMA','ENGINE_ID','NATIVE_STAGES','NATIVE_PHYSICAL_SLOTS','MEMBERS','CONTROL_MODE','MODE','_CACHED_CLASS','_SESSION_CLASS','_BODY','_BODY_CODE','_MODEL_RNG','original_native_query_scope','source_sha256','require_sources','policy_metadata','execution_metadata','route','_NativeStage','PriorNativeStageExecutor']}
_PUBLIC_SEALS={name:globals()[name] for name in _EXPORT_NAMES+['source_sha256','require_sources','policy_metadata','execution_metadata','route','SLOTS','ORIGINAL_SHA256','V1_PAIR_FAILURE_SHA256','PREREGISTRATION_SHA256','original','_BASE_REQUIRE','_BASE_POLICY','_BASE_EXECUTION','_BASE_ROUTE']}
_CALLABLE_SEALS=[]
for _name,_function in list(_PRIVATE.items()):
    if callable(_function) and getattr(_function,'__globals__',None) is _PRIVATE:
        _CALLABLE_SEALS.append((_PRIVATE,_name,_function,_function.__code__,_PRIVATE))
for _class in [_NativeStage,PriorNativeStageExecutor]:
    for _name,_value in vars(_class).items():
        _function=_value.fget if isinstance(_value,property) else _value
        if callable(_function) and getattr(_function,'__globals__',None) is _PRIVATE:
            _CALLABLE_SEALS.append((_class,_name,_function,_function.__code__,_PRIVATE))
for _name in ['_BASE_REQUIRE','_BASE_POLICY','_BASE_EXECUTION','_BASE_ROUTE','source_sha256','require_sources','policy_metadata','execution_metadata','route']:
    _function=globals()[_name];_CALLABLE_SEALS.append((globals(),_name,_function,_function.__code__,_function.__globals__))
_SCOPE_WRAPPER=_PRIVATE['original_native_query_scope']
_SCOPE_BODY=_SCOPE_WRAPPER.__wrapped__
_CALLABLE_SEALS.append((_SCOPE_WRAPPER,'__wrapped__',_SCOPE_BODY,_SCOPE_BODY.__code__,_PRIVATE))
_ANCESTRAL_SEAL_CONTENTS=tuple(_PRIVATE['_SEALS'])
_DERIVATION_DIGEST=hashlib.sha256(json.dumps(_DERIVATION,sort_keys=True).encode()).hexdigest()
_IMPORTED_SOURCE_SHA256=_raw_source_sha256()
