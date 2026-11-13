"""Pure config/custody/codec tests; no models, reference values or new worlds."""
import importlib.util,json,pickle,random
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from posteriorforge import prior_coordinate_pair_v1 as p


def load_script(name):
    path=Path(__file__).resolve().parents[1]/'scripts'/name
    s=importlib.util.spec_from_file_location('_test_'+path.stem,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


@pytest.mark.parametrize('seed,case',[(20261110,0),(20261111,8),(True,0),(20261111,True),(20261111,-1)])
def test_pair_rejects_outside_fixed_panel(seed,case):
    with pytest.raises(ValueError):p.PairSpec(seed,case)


def test_exact_immutable_pair_and_panel():
    x=p.PairSpec(20261111,0)
    assert x.as_dict()==dict(context_seed=20261111,case=0,context_size=4096,count=50000,repetition=0)
    assert p.PairSpec.from_dict(x.as_dict())==x
    assert len(p.PANEL)==24 and len(p.REMAINING_PAIRS)==23 and p.REGRESSION_PAIR not in p.REMAINING_PAIRS
    assert p.FIRST_PAIR==x
    with pytest.raises(FrozenInstanceError):x.case=3
    wrong=dict(x.as_dict(),count=49999)
    with pytest.raises(ValueError):p.PairSpec.from_dict(wrong)
    with pytest.raises(ValueError):p.PairSpec.from_dict(dict(x.as_dict(),reference='forbidden'))


def test_wrong_pair_parent_header_is_rejected_before_prefix_values(tmp_path,monkeypatch):
    m=tmp_path/'manifest.json';payload=tmp_path/'samples.npz';payload.write_bytes(b'untouched test payload')
    header={'status':'complete','context_seed':20261111,'case':1,'requested_count':50000,'context_size':4096,'repetition':0,'samples_file':'samples.npz','samples_sha256':p.sha(payload)}
    m.write_text(json.dumps(header));point={'context_seed':20261111,'case':0,'technical_passed':True,'count':50000,'parent_manifest_sha256':p.sha(m),'hybrid_manifest_sha256':'a'*64,'assessment_sha256':'b'*64}
    detail={'point_paths':{'parent':str(m),'hybrid':'not_loaded','assessment':'not_loaded'},'parent_payload_sha256':p.sha(payload)}
    monkeypatch.setattr(p,'grid_receipt',lambda path:{'points':[point]+[dict(point,case=7)]*23,'details':[detail]*24})
    with pytest.raises(ValueError,match='actual pair differs'):p.prefix_parent(p.FIRST_PAIR,completion_receipt='fixture')


def cpu_doc():
    return {'schema':p.PAIR_CPU_SCHEMA,'source_pins':p.source_pins(),'all23_processed':True,'COMMON_Float32_model_bytes_match_for_passed_pairs':True,'case2_regression_passed_both_environments':True,'pairs':[{'pair':s.as_dict(),'status':'pair_codec_preflight_pass','COMMON_Float32_model_bytes_match':True,'Float64_cross_environment_byte_identity_claimed':False} for s in p.REMAINING_PAIRS]}


def test_cpu_receipt_requires_actual_pair_all23_and_case2_regression():
    d=cpu_doc();assert p.validate_pair_cpu_receipt(d,p.FIRST_PAIR)['pair']==p.FIRST_PAIR.as_dict()
    d['pairs'][0]['status']='pair_codec_preflight_failure'
    with pytest.raises(ValueError,match='failed or missing'):p.validate_pair_cpu_receipt(d,p.FIRST_PAIR)
    d=cpu_doc();d['pairs'].pop()
    with pytest.raises(ValueError,match='exact23'):p.validate_pair_cpu_receipt(d,p.FIRST_PAIR)
    d=cpu_doc();d['case2_regression_passed_both_environments']=False
    with pytest.raises(ValueError,match='case2'):p.validate_pair_cpu_receipt(d,p.FIRST_PAIR)


def test_real_codec_endpoint_failure_preserves_rows_and_default_rng():
    before=pickle.dumps((np.random.get_state(),random.getstate()),protocol=4)
    values=np.tile(p.codec._geometry()['centers'],(4,1)).astype(np.float64)
    values[1,p.codec.NOISE_INDICES[0]]=.01;original=values.tobytes()
    with pytest.raises(ValueError):p.codec.encode_for_model(values)
    assert values.tobytes()==original
    assert pickle.dumps((np.random.get_state(),random.getstate()),protocol=4)==before




def test_actual_case2_reference_custody_and_wrong_case_or_reference_rejected():
    base=Path('/scratch/project/eu-26-17/eubrieucb/posteriorforge/results')
    banks=base/'banks-v1';parent=base/'accuracy-ss15-v1/sigma-native-grid-01/seed-20261111-case-2/manifest.json'
    if not parent.exists():pytest.skip('existing Karolina case2 reference fixture is not attached')
    preflight=load_script('preflight_prior_coordinate_pairs_v1.py')
    binding=preflight.authenticate_reference(p.REGRESSION_PAIR,banks=banks,parent_manifest=parent)
    assert binding==json.loads(parent.read_bytes())['reference_binding']
    with pytest.raises(ValueError,match='actual-case reference/grid'):
        preflight.authenticate_reference(p.FIRST_PAIR,banks=banks,parent_manifest=parent)
    with pytest.raises(ValueError,match='reference binding mismatch'):
        preflight.authenticate_reference(p.REGRESSION_PAIR,banks=banks,parent_manifest=parent,reference=base/'references-v1/case-0')
