"""WIP pair-v2 qualifier source guards; separate from CPU panel bindings."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest
from posteriorforge import prior_coordinate_pair_v1 as p

def load_script(name):
 path=Path(__file__).resolve().parents[1]/'scripts'/name
 s=importlib.util.spec_from_file_location('_test_'+path.stem,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def test_new_qualifier_wrong_oldproof_and_pair_rejected_before_numerics():
    q=load_script('qualify_prior_coordinate_pair_v2.py');spec=p.FIRST_PAIR
    args=dict(spec=spec,expected_sources={},raw_bindings={},transformed_bindings={},coordinate_policy={},native_policy={},inventory_replay={},pair_cpu_receipt={})
    with pytest.raises(ValueError,match='old5ec'):q.validate_screen({'schema':q.numeric.SCHEMA,'status':'prior_native_control_screen_pass'},**args)
    with pytest.raises(ValueError,match='PairSpec'):q.validate_screen({'schema':q.SCHEMA,'status':q.PASS_STATUS,'pair':p.REGRESSION_PAIR.as_dict()},**args)


def test_collector_has_only_declared_seed_delta_and_preserves_old_globals():
    q=load_script('qualify_prior_coordinate_pair_v2.py');old=dict(q.numeric.run.__globals__)
    spec=p.FIRST_PAIR;sentinel=SimpleNamespace()
    collector=q.build_collector(spec,data_proxy=sentinel,source_reader=lambda:{},selection_reader=lambda *args:{},writer=lambda *args:None,validator=lambda *args,**kwargs:{})
    assert collector.__globals__ is not q.numeric.run.__globals__
    assert collector.__globals__['_PAIR_SPEC'] is spec
    assert collector.__globals__['data_helper']() is sentinel
    assert all(q.numeric.run.__globals__[k] is v for k,v in old.items())
    assert q.witness_guard is q.numeric.witness_guard and q.stage_decision is q.numeric.stage_decision
    assert q.THRESHOLDS==q.numeric.THRESHOLDS


def test_qualifier_pins_missing_fail_before_gpu_or_models(monkeypatch):
    q=load_script('qualify_prior_coordinate_pair_v2.py')
    monkeypatch.setattr(q,'PAIR_CPU_RECEIPT_SHA256',None)
    with pytest.raises(ValueError,match='not frozen'):q.source_pins()


def actual_cpu(q):
 path=Path('/scratch/project/eu-26-17/eubrieucb/posteriorforge/results/accuracy-prior-coordinates-v1/pair-preflight-combined-01/receipt.json')
 if not path.exists():pytest.skip('closed Karolina pair CPU receipt is not attached')
 return q._cpu(path,p.FIRST_PAIR)


def binding_document(q,cpu,row):
 return {'schema':q.SCHEMA,'status':q.PASS_STATUS,'pair':p.FIRST_PAIR.as_dict(),
  'fixed_panel_preregistration_sha256':p.PREREGISTRATION_SHA256,
  'pair_CPU_receipt_sha256':q.PAIR_CPU_RECEIPT_SHA256,
  'prefix_custody':row['inference']['prefix_custody'],
  'stage_results':[{'mask_inputs':b} for b in row['inference']['all21_mask_target_query_bindings']],
  'failed_empty_cache_inventory':{'qualifies_actual_pair':False,
   'v1_case0_selection':{'manifest_sha256':q.V1_PAIR_FAILURE_SHA256,'failed_stages':[14],'qualification_passed':False},
   'predeclared_v2_native_all8_stages':[6,10,14]},
  'v2_preregistration_sha256':q.V2_PREREGISTRATION_SHA256,'v2_native_all8_stages':[6,10,14],
  'cached_stage_count':18,'v1_or_case2_pass_transferred':False}


def binding_args(q,cpu,row):
 return dict(spec=p.FIRST_PAIR,expected_sources=q.source_pins(),
  raw_bindings=row['inference']['raw_bindings'],transformed_bindings=row['inference']['transformed_bindings'],
  coordinate_policy={},native_policy={},inventory_replay={},pair_cpu_receipt=cpu)


def test_actual_cpu_receipt_pair_binding_replay_without_numerical_pass_claim(monkeypatch):
 # Numerical validator is explicitly stubbed: this tests CPU custody only, not any NN result.
 q=load_script('qualify_prior_coordinate_pair_v2.py');cpu,row=actual_cpu(q);calls=[]
 monkeypatch.setattr(q,'_numeric_validate',lambda document,**kwargs:(calls.append(document) or {'passed':True}))
 result=q.validate_screen(binding_document(q,cpu,row),**binding_args(q,cpu,row))
 assert result['passed'] and result['pair']==p.FIRST_PAIR.as_dict() and len(calls)==1
 assert calls[0]['status']=='prior_native_control_screen_pass'
 assert row['COMMON_Float32_model_bytes_match'] and not row['Float64_cross_environment_byte_identity_claimed']


@pytest.mark.parametrize('kind',['custody','mask','CPU_hash','selection_scope'])
def test_mutated_actual_pair_binding_rejected_before_numeric_validator(monkeypatch,kind):
 import copy
 q=load_script('qualify_prior_coordinate_pair_v2.py');cpu,row=actual_cpu(q)
 document=copy.deepcopy(binding_document(q,cpu,row));calls=[]
 monkeypatch.setattr(q,'_numeric_validate',lambda *args,**kwargs:calls.append(True))
 if kind=='custody':document['prefix_custody']['payload_sha256']='0'*64
 if kind=='mask':document['stage_results'][0]['mask_inputs']['actual_training']['sha256']='0'*64
 if kind=='CPU_hash':document['pair_CPU_receipt_sha256']='0'*64
 if kind=='selection_scope':document['failed_empty_cache_inventory']['qualifies_actual_pair']=True
 with pytest.raises(ValueError):q.validate_screen(document,**binding_args(q,cpu,row))
 assert calls==[]


def test_changed_pair_source_guard_rejected(monkeypatch):
 q=load_script('qualify_prior_coordinate_pair_v2.py');monkeypatch.setattr(q,'PAIR_MODULE_SHA256','0'*64)
 with pytest.raises(ValueError,match='PairSpec/input source changed'):q.source_pins()


def test_only_first_lexicographic_GPU_pair_allowed_before_any_input_or_model():
 q=load_script('qualify_prior_coordinate_pair_v2.py')
 args={k:'unread' for k in ['banks','prefix_run','checkpoint','codec_inventory','codec_reference_inventory','codec_cpu_receipt','completed_grid_audit','failed_screen','selection_anchor','v1_pair_failure','pair_cpu_receipt','output','ledger']}
 with pytest.raises(ValueError,match='only first lexicographic'):
  q.run(spec=p.PairSpec(20261111,1),**args)


def test_v2_private_backend_imports_and_original550_scope_are_explicit():
 q=load_script('qualify_prior_coordinate_pair_v2.py')
 assert q.numeric.NATIVE_STAGES==(6,10,14)
 assert q.numeric.witness_guard.__globals__['NATIVE_STAGES']==(6,10,14)
 assert 'execution_prior_native_v2' in q.numeric.validate_screen.__code__.co_names
 assert 'execution_prior_native_v1' not in q.numeric.validate_screen.__code__.co_names
 assert q.numeric.failed_screen_receipt is q.original_numeric.failed_screen_receipt
 assert q.numeric.failed_screen_receipt.__globals__['NATIVE_STAGES']==(6,10)
 c=q.build_collector(p.FIRST_PAIR,data_proxy=SimpleNamespace(),source_reader=lambda:{},selection_reader=lambda *args:{},writer=lambda *args:None,validator=lambda *args,**kwargs:{})
 assert 'execution_prior_native_v2' in c.__code__.co_names
 assert 'execution_prior_native_v1' not in c.__code__.co_names
 assert c.__globals__['NATIVE_STAGES']==(6,10,14)


@pytest.mark.parametrize('kind',['v2_doc','native_routes','cached_count','transfer','failure_SHA','failed_stage','failure_pass','selection_routes'])
def test_v2_decorated_lineage_contradictions_rejected_before_numerics(monkeypatch,kind):
 import copy
 q=load_script('qualify_prior_coordinate_pair_v2.py');cpu,row=actual_cpu(q);d=copy.deepcopy(binding_document(q,cpu,row));calls=[]
 monkeypatch.setattr(q,'_numeric_validate',lambda *args,**kwargs:calls.append(True))
 if kind=='v2_doc':d['v2_preregistration_sha256']='0'*64
 if kind=='native_routes':d['v2_native_all8_stages']=[6,10]
 if kind=='cached_count':d['cached_stage_count']=19
 if kind=='transfer':d['v1_or_case2_pass_transferred']=True
 if kind=='failure_SHA':d['failed_empty_cache_inventory']['v1_case0_selection']['manifest_sha256']='0'*64
 if kind=='failed_stage':d['failed_empty_cache_inventory']['v1_case0_selection']['failed_stages']=[13]
 if kind=='failure_pass':d['failed_empty_cache_inventory']['v1_case0_selection']['qualification_passed']=True
 if kind=='selection_routes':d['failed_empty_cache_inventory']['predeclared_v2_native_all8_stages']=[6,10]
 with pytest.raises(ValueError):q.validate_screen(d,**binding_args(q,cpu,row))
 assert calls==[]


def test_v1_pair_passing_schema_cannot_be_v2_proof():
 q=load_script('qualify_prior_coordinate_pair_v2.py');args=dict(spec=p.FIRST_PAIR,expected_sources={},raw_bindings={},transformed_bindings={},coordinate_policy={},native_policy={},inventory_replay={},pair_cpu_receipt={})
 with pytest.raises(ValueError,match='schema/pass status'):
  q.validate_screen({'schema':'posteriorforge.prior_coordinate_pair_conditionals.v1','status':'prior_coordinate_pair_screen_pass'},**args)


def test_actual_v1_failed_selection_is_only_new_route_selection_not_v2_pass():
 q=load_script('qualify_prior_coordinate_pair_v2.py');base=Path('/scratch/project/eu-26-17/eubrieucb/posteriorforge/results/accuracy-prior-coordinates-v1')
 if not (base/'pair-conditionals-seed20261111-case0-01/manifest.json').exists():pytest.skip('closed actual v1 failure is not attached')
 result=q._selection(base/'conditionals-01/manifest.json',base/'native-conditionals-01/manifest.json',base/'pair-conditionals-seed20261111-case0-01/manifest.json')
 assert result['qualifies_actual_pair'] is False and result['predeclared_v2_native_all8_stages']==[6,10,14]
 assert result['v1_case0_selection']['qualification_passed'] is False
 assert result['predeclared_all8_native_stages']==[6,10]
