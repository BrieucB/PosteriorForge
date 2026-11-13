"""Focused CPU-only window, observer and fitted-state diagnostic contracts."""
import hashlib
import importlib.util
from pathlib import Path
import pickle
import random
from types import SimpleNamespace
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('tail_capture',ROOT/'scripts/diagnose_prior_tail_conditionals_v1.py')
d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)


def test_fixed_same_FP32_uniforms():
    values=d.fixed_uniforms()
    assert values.dtype==np.float32 and np.all(values<1) and np.all(values>=0)
    assert values[3].tobytes().hex()=='e0ff7f3f'
    assert values[4].tobytes().hex()=='f6ff7f3f'
    assert values[-1]==np.nextafter(np.float32(1),np.float32(0))


@pytest.mark.parametrize('stage,row,first,last,bits',d.FIXTURES)
def test_exact_window_prefix_and_target_exclusion(stage,row,first,last,bits):
    latent=np.arange(51200*21,dtype=np.float32).reshape(51200,21)
    query=np.arange(15,dtype=np.float32).reshape(1,15)
    actual=d.window_query(query,latent,stage,row,first,last)
    assert actual.shape==(1024,15+stage) and actual.dtype==np.float32
    assert actual[:,15:].tobytes()==latent[first:last,:stage].copy().tobytes()
    assert actual[row-first,15:].tobytes()==latent[row,:stage].copy().tobytes()
    assert np.array_equal(actual[:,:15],np.repeat(query,1024,axis=0))


def test_undeclared_window_rejected():
    with pytest.raises(RuntimeError,match='undeclared window'):
        d.window_query(np.zeros((1,15),np.float32),np.zeros((51200,21),np.float32),15,4355,4095,5119)


def test_implicit_query_cast_rejected():
    with pytest.raises(RuntimeError,match='F32 inputs'):
        d.window_query(np.zeros((1,15),np.float64),np.zeros((51200,21),np.float32),15,4355,4096,5120)


@pytest.mark.parametrize('passed,a,b,expected',[
    (True,[True]*8,[True]*8,False),
    (True,[False]*8,[False]*8,True),
    (True,[True]+[False]*7,[False]*8,True),
    (False,[True]*8,[True]*8,True),
])
def test_N1_predeclared_material_or_absent_failure_trigger(passed,a,b,expected):
    assert d.need_N1([{'passed':passed}]*8,a,b) is expected


def test_actual_fitted_transform_numeric_parameters(tmp_path):
    transform=SimpleNamespace(method='yeo-johnson',standardize=True,lambdas_=np.array([.7],dtype=np.float64),
                              revert_indices_=np.array([0]),_scaler=SimpleNamespace(mean_=np.array([.2]),scale_=np.array([.3]),var_=np.array([.09])))
    reg=SimpleNamespace(executor_=SimpleNamespace(configs=[SimpleNamespace(target_transform=transform)]*8),
                        y_train_mean_=.2,y_train_std_=.7,softmax_temperature=.9)
    record=d.fitted_parameters(reg,tmp_path,'fixture')
    assert record['configurations'][0]['state']['target_transform']['state']['lambdas_']['values']==[.7]
    assert record['configurations'][0]['state']['target_transform']['state']['_scaler']['state']['mean_']['values']==[.2]
    assert len(pickle.loads((tmp_path/'fixture-fitted-configs.pkl').read_bytes()))==8


@pytest.mark.parametrize('rows',[1,3])
def test_actual_whole_batch_observer_and_no_default_rng_mutation(rows):
    torch=pytest.importorskip('torch')
    before=(pickle.dumps(np.random.get_state()),pickle.dumps(random.getstate()),torch.get_rng_state().numpy().tobytes())
    logits=torch.tensor([[0.,-3.,-12.],[1.,0.,-8.],[-2.,0.,-15.]],dtype=torch.float32)[:rows]
    prediction={'logits':logits,'criterion':SimpleNamespace(borders=torch.tensor([-93.,-1.,1.,86.],dtype=torch.float32))}
    arrays,meta=d.observed_inverse(prediction,torch)
    assert arrays['probabilities'].tobytes()==logits.softmax(-1).numpy().tobytes()
    assert arrays['cdf'].tobytes()==torch.cumsum(logits.softmax(-1),-1).numpy().tobytes()
    assert arrays['output'].shape==(6,rows) and arrays['bin'].shape==(6,rows)
    assert meta['all_provided_U_whole_batch_byte_parity'] and meta['observer']['reverse_AST_exact']
    assert meta['u1_stress']['outside_production_domain']
    assert d.mixture.finite_bar_icdf is d._ICDF and d._ICDF.__code__ is d._ICDF_CODE
    after=(pickle.dumps(np.random.get_state()),pickle.dumps(random.getstate()),torch.get_rng_state().numpy().tobytes())
    assert before==after and not torch.cuda.is_initialized()


def test_focal_failure_classification_is_not_complete_decode():
    values=np.zeros((6,1),dtype=np.float32)
    values[3,0]=np.float32(13.876729011535645)
    assert d.focal_failure_mask({'output':values},15,0,3) is True
    assert d.focal_failure_mask({'output':values},15,0,0) is False
