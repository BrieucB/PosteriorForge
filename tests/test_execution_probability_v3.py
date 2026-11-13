"""CPU-only Policy C proofs; saved focal arrays are not fresh neural evidence."""
import hashlib,json,random,unittest,importlib.util
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch
from posteriorforge import execution_probability_v3 as p

HAS_TORCH=importlib.util.find_spec('torch') is not None
REPORT=Path('/scratch/project/eu-26-17/eubrieucb/posteriorforge/results/accuracy-prior-coordinates-v1/tail-postprocessing-offline-01/report.json')
REPORT_SHA='cbc8e44e62780d48e594845bd63a9c72faa76eaf7a1459d9e4dbb2210e989a5a'

class SourceTests(unittest.TestCase):
    def test_source_policy_and_original_head_identity(self):
        self.assertTrue(p.require_sources());self.assertIs(p.head_acceptance,p.legacy.head_acceptance)
        self.assertFalse(p.metadata()['legacy_bit_identity_claimed'])
        self.assertEqual(p.metadata()['RNG_calls'],0)
    def test_probability_export_and_config_mutation_fail(self):
        with patch.object(p,'POLICY','changed'):
            with self.assertRaisesRegex(RuntimeError,'binding'):p.require_sources()
        with patch.object(p.legacy,'HEAD_MASS_ENVELOPE',2.):
            with self.assertRaisesRegex(RuntimeError,'envelope'):p.require_sources()
        with patch.object(p,'finite_bar_icdf',lambda *a:None):
            with self.assertRaisesRegex(RuntimeError,'callable'):p.require_sources()

@unittest.skipUnless(HAS_TORCH,'Torch is absent in locked reference environment')
class ProbabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch,numpy as np
        cls.t=torch;cls.np=np
    def tensor(self,a,dtype=None):return self.t.tensor(a,dtype=dtype or self.t.float64)
    def test_zero_bins_ties_u0_and_u1(self):
        t=self.t;logits=self.tensor([[0.,0.,2.,0.,2.,0.]]).log()
        b=self.tensor([-3,-2,-1,0,1,2,3]);u=self.tensor([0,.25,.5,.75],t.float32)
        result=p.inverse_details(logits,b,u)
        self.assertEqual(result['index'].tolist(),[2,2,2,4])
        self.assertEqual(result['values64'].tolist(),[-1.,-.5,0.,1.5])
        self.assertTrue(t.equal(result['sampling_cdf64'][:,-1],self.tensor([1.])))
        with self.assertRaisesRegex(RuntimeError,'u1'):p.finite_bar_icdf(logits,b,self.tensor([1.],t.float32))
    def test_uniforms_exact_dtype_domain_and_no_implicit_cast(self):
        t=self.t;b=self.tensor([0,1,2]);x=self.tensor([[0,0]])
        for u in (self.tensor([.5]),self.tensor([-1.],t.float32),self.tensor([float('nan')],t.float32)):
            with self.assertRaises(RuntimeError):p.finite_bar_icdf(x,b,u)
        with self.assertRaises(RuntimeError):p.finite_bar_icdf(x.float(),b,self.tensor([.5],t.float32))
    def test_shared_and_per_row_vectorized_match(self):
        t=self.t;x=self.tensor([[0.,1.,-float('inf')]]);b=self.tensor([-2.,0.,3.,4.]);u=self.tensor([0.,.1,.3,.9],t.float32)
        self.assertTrue(t.equal(p.finite_bar_icdf(x,b,u),p.finite_bar_icdf(x.expand(len(u),-1),b,u)))
    def test_translation_cut_and_original_borders_immutability(self):
        t=self.t;x=self.tensor([[0.,2.,-float('inf')],[2.,0.,1.]],t.float32)
        b=self.tensor([-2.,0.,1.,5.],t.float32);saved=[v.clone() for v in (x,b)]
        result=p.from_adjusted_logits(x,b,b,b)
        self.assertEqual(result['logits'].dtype,t.float64);self.assertTrue(t.equal(result['criterion'].borders,b.double()))
        self.assertTrue(t.equal(result['raw_weights64'].sum(-1),result['component_raw_mass']))
        self.assertTrue(t.equal(x,saved[0]));self.assertTrue(t.equal(b,saved[1]))
        self.assertFalse(result['probability_v3']['temperature_reapplied'])
        json.dumps(result['probability_v3'])
    def test_invalid_grids_logits_and_mass_rejected(self):
        t=self.t;b=self.tensor([0,1,2],t.float32);x=self.tensor([[0,0]],t.float32)
        for bad in (self.tensor([0,0,2],t.float32),self.tensor([0,2,1],t.float32)):
            with self.assertRaises(RuntimeError):p.translate_logits(x,bad,b)
        with self.assertRaises(RuntimeError):p.translate_logits(x.double(),b,b)
        with self.assertRaises(RuntimeError):p.translate_logits(self.tensor([[-float('inf'),-float('inf')]],t.float32),b,b)
        with self.assertRaises(RuntimeError):p.normalize_weights(self.tensor([[0.,0.]]))
    def test_raw_mass_before_normalization_and_stage0_counts(self):
        t=self.t;b=self.tensor([0,1,2]);heads=[]
        for i in range(8):
            raw=self.tensor([[.1 if i==0 else .8,.1 if i==0 else .2]])
            heads.append({'raw_weights64':raw,'criterion':type('Criterion',(),{'borders':b})()})
        self.assertIs(p.head_acceptance,p.legacy.head_acceptance)
        self.assertEqual(p.head_acceptance(self.tensor([.2]),self.tensor([.3],t.float32)).tolist(),[False])
        with p.production_scope() as d:
            diagnostic=p.assemble_shared_raw64(heads)
            self.assertEqual(d['stage0_assembly_count'],0)
            result=p.assemble_shared_raw64(heads,stage0=True)
            self.assertEqual(d['stage0_assembly_count'],1)
            expected=t.stack([h['raw_weights64'] for h in heads]).mean(0)
            self.assertTrue(t.equal(result['raw_weights64'],expected))
            individually=t.stack([h['raw_weights64']/h['raw_weights64'].sum(-1,keepdim=True) for h in heads]).mean(0)
            self.assertFalse(t.equal(result['probabilities64'],individually))
        self.assertTrue(d['closed']);self.assertTrue(d['source_unchanged'])
    def test_natural_CDF_collapse_reported_without_weight_mutation(self):
        t=self.t;w=self.tensor([[1.,2.**-60,0.]]);before=w.clone();d=p.normalize_weights(w)
        self.assertTrue(t.equal(w,before));self.assertEqual(d['representation_diagnostics']['collapsed_positive_weight_count'],1)
        self.assertEqual(d['representation_diagnostics']['collapsed_positive_raw_mass64'],2.**-60)
        self.assertEqual(d['sampling_interval_weights64'].tolist(),[[1.,0.,0.]])
    def test_default_RNG_and_CUDA_not_initialized(self):
        t=self.t;np=self.np
        before=(random.getstate(),np.random.get_state(),t.get_rng_state().clone(),t.cuda.is_initialized())
        x=self.tensor([[1.,2.]],t.float32);b=self.tensor([0.,1.,2.],t.float32)
        result=p.from_adjusted_logits(x,b,b,b);p.finite_bar_icdf(result['logits'],result['criterion'].borders,self.tensor([.7],t.float32))
        self.assertEqual(random.getstate(),before[0]);after=np.random.get_state()
        self.assertEqual(before[1][0],after[0]);self.assertTrue(np.array_equal(before[1][1],after[1]));self.assertEqual(before[1][2:],after[2:])
        self.assertTrue(t.equal(t.get_rng_state(),before[2]));self.assertEqual(t.cuda.is_initialized(),before[3])


    def test_translated_batch_geometry_matches_individual_rows(self):
        t=self.t;x=self.tensor([[0.,2.,1.],[3.,-1.,.5]],t.float32)
        frm=self.tensor([-3.,-1.,1.,4.],t.float32);to=self.tensor([-4.,0.,2.,5.],t.float32)
        batch=p.translate_logits(x,frm,to)
        for i in range(len(x)):
            one=p.translate_logits(x[i:i+1],frm,to)
            for key in ('raw_weights64','translation_cdf64','pretranslation_probabilities64'):
                self.assertTrue(t.equal(batch[key][i:i+1],one[key]))
    def test_scope_failure_closes_and_foreign_descriptor_rejected(self):
        with self.assertRaisesRegex(ValueError,'test failure'):
            with p.production_scope() as descriptor:
                with self.assertRaises(RuntimeError):p.active_descriptor(dict(descriptor))
                raise ValueError('test failure')
        self.assertTrue(descriptor['closed']);self.assertFalse(descriptor['active'])
        self.assertIsNone(p.active_descriptor())

    def test_exact_dyadic_normalized_CDF_oracle(self):
        t=self.t;w=self.tensor([[0.,1.,3.,7.,0.,2.**-40]])
        d=p.normalize_weights(w);cdf=d['sampling_cdf64'][0].tolist()
        exact=[Fraction.from_float(v) for v in w[0].tolist()];total=sum(exact);prefix=Fraction(0)
        for i,weight in enumerate(exact):
            prefix+=weight;self.assertLessEqual(abs(Fraction.from_float(cdf[i])-prefix/total),Fraction(2,2**52))
        x=w.log();u=self.tensor([0.,.1,.4,.8,.99999994],t.float32)
        inverse=p.inverse_details(x,self.tensor([0,1,2,3,4,5,6]),u)
        api=inverse['api_probabilities64'][0].tolist();weights=[Fraction.from_float(v) for v in api];total=sum(weights)
        for ui,bi,frac in zip(u.tolist(),inverse['index'].tolist(),inverse['fraction64'].tolist()):
            exactcdf=(sum(weights[:bi])+weights[bi]*Fraction.from_float(frac))/total
            self.assertLessEqual(abs(exactcdf-Fraction.from_float(ui)),Fraction(16,2**52))

    @unittest.skipUnless(REPORT.exists(),'saved focal arrays are available only on Karolina')
    def test_all48_saved_PolicyC_upstream_and_all_provided_uniforms(self):
        t=self.t;np=self.np
        self.assertEqual(hashlib.sha256(REPORT.read_bytes()).hexdigest(),REPORT_SHA)
        doc=json.loads(REPORT.read_text());self.assertEqual(len(doc['records']),48)
        differences=[];special=None;maximum_exact_CDF_error=0.;maximum_exact_raw_CDF_error=0.;nonfinite_old_outputs=0;bound=None
        for record in doc['records']:
            path=Path(record['source_focal_file']);self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),record['source_focal_sha256'])
            with np.load(path,allow_pickle=False) as z:
                arrays={k:z[k].copy() for k in z.files}
            x=t.from_numpy(arrays['raw_model_logits'][None,:]);frm=t.from_numpy(arrays['from_borders']);to=t.from_numpy(arrays['to_borders']);b=t.from_numpy(arrays['borders'])
            result=p.from_adjusted_logits(x,frm,to,b)
            reference=REPORT.parent/record['derived_reference_weights_file']
            self.assertEqual(hashlib.sha256(reference.read_bytes()).hexdigest(),record['derived_reference_weights_sha256'])
            with np.load(reference,allow_pickle=False) as z:
                expected=z['C_final_post64_weights']
                expected_raw=z['C_raw_translated_F64']
            self.assertTrue(np.array_equal(result['raw_weights64'].numpy().reshape(-1),expected_raw.reshape(-1)))
            self.assertTrue(np.allclose(result['probabilities64'].numpy().reshape(-1),expected.reshape(-1),rtol=0,atol=2e-15))
            u=t.from_numpy(arrays['uniforms']);inv=p.inverse_details(result['logits'],result['criterion'].borders,u)
            self.assertEqual(len(inv['values64']),6)
            self.assertTrue(bool((inv['fraction64']>=0).all() and (inv['fraction64']<=1).all()))
            self.assertTrue(bool((inv['values64']>=float(b[0])).all() and (inv['values64']<=float(b[-1])).all()))
            self.assertTrue(bool((inv['sampling_cdf64'][:,1:]>=inv['sampling_cdf64'][:,:-1]).all()))
            self.assertEqual(float(inv['sampling_cdf64'][0,-1]),1.)
            # Exact dyadic weights define an independent normalized-CDF oracle;
            # represented-CDF error is bounded rather than asserted bit-identical.
            weights=[Fraction.from_float(v) for v in inv['api_probabilities64'][0].tolist()];total=sum(weights)
            bins=inv['index'].tolist()
            prefix={i:sum(weights[:i]) for i in set(bins)}
            for ui,bi,frac in zip(u.tolist(),bins,inv['fraction64'].tolist()):
                exact=(prefix[bi]+weights[bi]*Fraction.from_float(frac))/total
                error=float(abs(exact-Fraction.from_float(ui)));maximum_exact_CDF_error=max(maximum_exact_CDF_error,error)
                gamma=len(weights)*2.**-52/(1-len(weights)*2.**-52);bound=4*gamma
                self.assertLessEqual(error,bound)
            raw_exact=[Fraction.from_float(v) for v in result['raw_weights64'][0].tolist()];raw_total=sum(raw_exact)
            raw_prefix={i:sum(raw_exact[:i]) for i in set(bins)}
            for ui,bi,frac in zip(u.tolist(),bins,inv['fraction64'].tolist()):
                rawcdf=(raw_prefix[bi]+raw_exact[bi]*Fraction.from_float(frac))/raw_total
                error=float(abs(rawcdf-Fraction.from_float(ui)));maximum_exact_raw_CDF_error=max(maximum_exact_raw_CDF_error,error)
                self.assertLessEqual(error,bound+result['probability_v3']['raw_normalized_vs_API_normalized_max_row_TV'])
            finite=np.isfinite(arrays['output']);nonfinite_old_outputs+=int((~finite).sum())
            if finite.any():differences.append(float(np.max(np.abs(inv['values64'].numpy()[finite]-arrays['output'][finite]))))
            if (record['stage'],record['arm'],record['member'],record['label'])==(19,'candidate',2,'N1024'):
                normalized_A=p.finite_bar_icdf(t.from_numpy(arrays['probabilities'][None,:]).double().log(),result['criterion'].borders,u)
                special={'captured_legacy_stock_output':float(arrays['output'][4]),'captured_A_weights_new_normalized64_inverse':float(normalized_A[4]),'new_upstream_C':float(inv['values64'][4])}
        self.assertGreater(max(differences),100.)
        self.assertIsNotNone(special);self.assertGreater(special['captured_A_weights_new_normalized64_inverse'],40.);self.assertLess(special['new_upstream_C'],9.)
        print('CAPTURED48_PROOF '+json.dumps({'focal_records':48,'provided_uniforms_per_record':6,'inverse_values':288,
            'source_report_sha256':REPORT_SHA,'raw_PolicyC_weights_bit_equal_to_independent_saved_C':True,
            'maximum_exact_dyadic_API_weight_CDF_residual':maximum_exact_CDF_error,
            'maximum_exact_dyadic_RAW_weight_CDF_residual':maximum_exact_raw_CDF_error,'legacy_nonfinite_output_count':nonfinite_old_outputs,'conservative_4gamma_n_bound':bound,
            'maximum_finite_legacy_stock_vs_new_C_quantile_difference':max(differences),'stage19_candidate_member2_A_C':special,
            'new_neural_model_preprocessor_GPU_or_RNG_draw_calls':0},sort_keys=True))
