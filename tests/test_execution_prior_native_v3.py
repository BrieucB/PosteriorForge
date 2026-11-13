"""No model construction, preprocessing replay, forward, GPU or RNG draw."""
import importlib.util,unittest,copy
from types import FunctionType,SimpleNamespace
from unittest.mock import patch
from posteriorforge import execution_prior_native_v3 as b

class SourceTests(unittest.TestCase):
    def test_unfrozen_probability_is_fail_closed(self):
        if b.PROBABILITY_SHA is None:
            with self.assertRaisesRegex(RuntimeError,'not frozen'):b.require_sources()
        self.assertTrue(b.require_sources(allow_unfrozen_probability=True))
    def test_all_public_alias_and_method_seals(self):
        for name,value in [('NATIVE_STAGES',(6,10)),('route',lambda stage:'fake')]:
            with patch.object(b,name,value):
                with self.assertRaises(RuntimeError):b.require_sources(allow_unfrozen_probability=True)
        with patch.object(b.PriorNativeStageExecutor,'prepare',lambda *a:None):
            with self.assertRaises(RuntimeError):b.require_sources(allow_unfrozen_probability=True)
        with patch.object(b.PriorNativeStageExecutor,'base',property(lambda self:None)):
            with self.assertRaises(RuntimeError):b.require_sources(allow_unfrozen_probability=True)
    def test_referenced_dependency_sealed(self):
        with patch.object(b,'ExitStack',lambda:None):
            with self.assertRaises(RuntimeError):b.require_sources(allow_unfrozen_probability=True)
    def test_failure_closes_all_callbacks_and_preserves_primary(self):
        from contextlib import ExitStack
        s=object.__new__(b.PriorNativeSession);s._stack=ExitStack();events=[]
        s.active=True;s.probability_descriptor=None;s.legacy_guard=None
        def release():events.append('release');raise OSError('cleanup')
        s._stack.callback(lambda:events.append('legacy exit'));s._stack.callback(release)
        primary=ValueError('primary')
        with self.assertRaises(ValueError) as caught:s.__exit__(ValueError,primary,None)
        self.assertIs(caught.exception,primary);self.assertIsInstance(caught.exception.__cause__,OSError)
        self.assertEqual(events,['release','legacy exit']);self.assertFalse(s.active)
    def test_original_method_objects_and_globals_not_modified(self):
        self.assertEqual(b.neural.source_sha256(),b.NEURAL_SHA)
        self.assertEqual(b.NATIVE_STAGES,(6,10,14))
        self.assertIsNot(b.PriorNativeStageExecutor,b.neural.PriorNativeStageExecutor)
        self.assertIs(b.mask_query,b.neural.mask_query)

@unittest.skipUnless(importlib.util.find_spec('torch') is not None,'Torch absent in reference environment')
class PrefixTests(unittest.TestCase):
    def test_prefix_AST_and_exact_temp_mask_cut_no_forward(self):
        import torch,numpy as np
        before=torch.get_rng_state().clone();cuda=torch.cuda.is_initialized()
        original=b.neural.cached.precision.regressor_module if hasattr(b.neural.cached,'precision') and hasattr(b.neural.cached.precision,'regressor_module') else None
        f=b._prefix();self.assertTrue(b._PREFIX_DESCRIPTOR['reverse_AST_exact'])
        namespace=dict(f.__globals__)
        namespace.update(check_is_fitted=lambda self:None,validate_X_predict=lambda X,self:X,
            _fix_dtypes=lambda X,**kw:X,_process_text_na_dataframe=lambda X,**kw:X,
            RegressorEnsembleConfig=SimpleNamespace,
            _transform_borders_one=lambda *a,**kw:(np.asarray([False,True]),False,np.asarray([-1.,0.,2.],dtype=np.float32)))
        invoke=FunctionType(f.__code__,namespace,f.__name__,f.__defaults__,f.__closure__);invoke.__kwdefaults__=f.__kwdefaults__
        x=torch.tensor([[.9,1.8]],dtype=torch.float32);config=SimpleNamespace(target_transform=object())
        state=SimpleNamespace(categorical_features_indices=[],preprocessor_=None,_USABLE_OUTPUT_TYPES=('full',),
            bardist_=SimpleNamespace(borders=torch.tensor([-1.,0.,2.],dtype=torch.float32)),
            renormalized_criterion_=SimpleNamespace(borders=torch.tensor([-3.,0.,6.],dtype=torch.float32)),
            executor_=SimpleNamespace(iter_outputs=lambda *a,**kw:iter([(x,config)])),
            device_='cpu',use_autocast_=False,softmax_temperature=.9,
            interface_config_=SimpleNamespace(FIX_NAN_BORDERS_AFTER_TARGET_TRANSFORM=True))
        with torch.inference_mode():outputs,borders,canonical,api=invoke(state,np.zeros((1,15),np.float32),output_type='full',quantiles=[])
        self.assertEqual(float(outputs[0][0,0]),float((x/.9)[0,0]))
        self.assertTrue(torch.isneginf(outputs[0][0,1]))
        self.assertTrue(torch.equal(x,torch.tensor([[.9,1.8]],dtype=torch.float32)))
        self.assertTrue(torch.equal(torch.get_rng_state(),before));self.assertEqual(torch.cuda.is_initialized(),cuda)
        self.assertIs(b._prefix(),f)
    def test_private_prefix_global_tamper_rejected(self):
        f=b._prefix();name=next(iter(b._PREFIX_SEALS[2]));old=f.__globals__[name]
        try:
            f.__globals__[name]=object()
            with self.assertRaisesRegex(RuntimeError,'global binding'):b._prefix()
        finally:f.__globals__[name]=old
