#!/usr/bin/env python3
"""Saved-case2 conditional-mean attribution only; no candidate or new draws.

Built with PriorLabs-TabPFN. All50000 original accepted neural-parent rows
are used before the later joint-global refresh. No gate is changed.
"""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,pickle,random,time
from pathlib import Path
import numpy as np
from posteriorforge.artifacts import sha256_file,write_json
from posteriorforge.banks import load_heldout
from posteriorforge.inference import support_mask
from posteriorforge.science.analytical_benchmark_v2 import build_problem,unflatten_observations,array_descriptor
from posteriorforge.validation import _reference_arrays,_moment_uncertainties

SCHEMA="posteriorforge.case2_conditional_mean_attribution.v1"
PARENT_SHA="28de1d121e5e89a5b55f630625c0f60e86551473c8bbaa3b2840797683f12e6a"
PAYLOAD_SHA="9ce954f75b73c54348733c3068179d542ed050ec1aaae772734b3def67fc752b"
ASSESSMENT_SHA="4590a8acff216737d04b8c6144d72df0a14c0246ca5333a330ad365a07a857cb"
EVALUATOR_SHA="146c2ca35e15b54b7f68134d383574485ed1cc228f38e6fc611624f00b411924"
BINDER_SHA="284d3b061696d4d823666e12d3f79c720f87692b40f45743296cfb8e93865424"
PINS={
 "src/posteriorforge/science/analytical_benchmark_v2.py":"8d5f286f6d8512391dea2ae036313bec05782d28c7803f336490cbcdce3bd99f",
 "src/posteriorforge/science/hierarchical_exact.py":"db001593cc7307b1290b981622705961c5953001a6c1245fae5ec779938e9687",
 "src/posteriorforge/science/hierarchical_noise.py":"cbf0b8c956362a5eaa31923e9b94a3db3e3ede2072603add51e78105b8c0cbf0",
 "src/posteriorforge/inference.py":"95366efc4871b96c3a1a3f19414ed1fd77d058f02135f4afd95a64b5918b3d3e",
 "src/posteriorforge/validation.py":"92ddfd79a1c42cdbc56dfc9ad0f2c987fa70ee69387b0c5decff53c20b9aa694",
 "src/posteriorforge/performance_ss15_scientific_campaign.py":"3965e36206d7f750c66350df54b6d855d6c9f8f4f81a0e069aeeb478492cd453",
 "src/posteriorforge/performance_runner.py":"23ccad8b5bbcdd7a48340b349d515ca870763359cfeb71333e02fc2b6a8aca3b",
 "src/posteriorforge/upstream/npe_pfn/accept_reject_sampler.py":"e6f862a176eae4cf0269c49655cf2edefc854428461445c74aa2b02f3b144512"}
TARGETS=("a_m2","b_m2")
Z95=1.959963984540054

def require(condition,message):
 if not condition:raise ValueError(message)

def load_script(name,digest):
 path=Path(__file__).with_name(name)
 require(sha256_file(path)==digest,"source changed: "+name)
 spec=importlib.util.spec_from_file_location("_case2_attribution_"+path.stem,path)
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def geometry(problem=None):
 p=build_problem() if problem is None else problem
 prior=next(b for b in p.prior.base.conditional_blocks if b.dataset_id=="mechanical_2")
 forward=next(b for b in p.observation_model.blocks if b.dataset_id=="mechanical_2")
 require(prior.local_names==forward.local_names==("a_m2","b_m2","d0_m2"),"matching local coordinate order required")
 require(np.array_equal(forward.covariance,np.eye(forward.observation_size)),"sigma-squared independent-point likelihood required")
 order=tuple(p.parameter_order);sigma=p.prior.noise_name(prior.dataset_id)
 return {"problem":p,"prior":prior,"forward":forward,
  "global_indices":[order.index(k) for k in p.prior.base.global_names],
  "local_indices":[order.index(k) for k in prior.local_names],"sigma_index":order.index(sigma),
  "V0_inverse":np.linalg.solve(prior.covariance,np.eye(len(prior.local_names)))}

def conditional_moments(values,observed,g=None):
 g=geometry() if g is None else g
 x=np.asarray(values);y=np.asarray(observed,dtype=np.float64).reshape(-1)
 require(x.dtype==np.float64 and x.ndim==2 and x.shape[1]==21 and np.isfinite(x).all(),"finite original Float64 physical rows required")
 f=g["forward"];p=g["prior"];sigma=x[:,g["sigma_index"]]
 require(y.shape==(f.observation_size,) and np.isfinite(y).all() and np.all(sigma>0),"finite block observation and positive SD required")
 m0=x[:,g["global_indices"]]@p.population_map.T+p.offset
 precision=g["V0_inverse"][None,:,:]+(f.design.T@f.design)[None,:,:]/sigma[:,None,None]**2
 natural=m0@g["V0_inverse"].T+(f.design.T@(y-f.offset))[None,:]/sigma[:,None]**2
 covariance=np.linalg.solve(precision,np.broadcast_to(np.eye(len(p.local_names)),precision.shape))
 mean=np.linalg.solve(precision,natural[...,None])[...,0]
 require(np.isfinite(mean).all() and np.isfinite(covariance).all(),"nonfinite conditional moments")
 return mean,covariance

def covariance_space(values,observed,g):
 f=g["forward"];p=g["prior"];sigma=values[:,g["sigma_index"]]
 m0=values[:,g["global_indices"]]@p.population_map.T+p.offset
 cross=p.covariance@f.design.T
 s=f.design@cross+sigma[:,None,None]**2*np.eye(f.observation_size)
 residual=np.asarray(observed)-f.offset-m0@f.design.T
 mean=m0+np.einsum("ij,nj->ni",cross,np.linalg.solve(s,residual[...,None])[...,0])
 covariance=p.covariance-np.einsum("ij,njk->nik",cross,np.linalg.solve(s,np.broadcast_to(cross.T,(len(values),*cross.T.shape))))
 return mean,covariance

def signed_numerical_interval(value,mean_error,sd_error):
 require(0<=mean_error<=.005 and 0<=sd_error<=.005,"qualified reference numerical errors required")
 corners=[(value+d)/(1+s) for d in (-mean_error,mean_error) for s in (-sd_error,sd_error)]
 return [float(min(corners)),float(max(corners))]

def paired_decomposition(beta,conditional_mean,reference_mean,reference_sd,mean_error,sd_error):
 beta=np.asarray(beta,dtype=np.float64);m=np.asarray(conditional_mean,dtype=np.float64)
 require(beta.ndim==1 and beta.shape==m.shape and len(beta)>1 and reference_sd>0,"paired rows and positive reference SD required")
 rows=np.column_stack((beta-m,m-reference_mean,beta-reference_mean))/reference_sd
 points=rows.mean(axis=0);cov=np.cov(rows,rowvar=False,ddof=1);mean_cov=cov/len(beta);se=np.sqrt(np.diag(mean_cov))
 names=("accepted_sampler_conditional_residual","conditional_mean_mixing_discrepancy","total_signed_error")
 result={}
 for j,name in enumerate(names):
  err=0. if j==0 else mean_error
  lo,hi=points[j]-Z95*se[j],points[j]+Z95*se[j]
  result[name]={"signed_refSD":float(points[j]),"approximate_iid_row_MCSE_refSD":float(se[j]),
   "approximate_CLT95_MC_only_refSD":[float(lo),float(hi)],
   "declared_reference_numerical_interval_refSD":signed_numerical_interval(points[j],err,sd_error),
   "approximate_CLT95_enlarged_by_declared_reference_errors_refSD":[
    signed_numerical_interval(lo,err,sd_error)[0],signed_numerical_interval(hi,err,sd_error)[1]]}
 return {"components":result,"paired_row_covariance_refSD2":cov.tolist(),
  "paired_mean_covariance_refSD2":mean_cov.tolist(),
  "residual_mixing_correlation":float(cov[0,1]/np.sqrt(cov[0,0]*cov[1,1])),
  "rowwise_decomposition_max_abs_error_refSD":float(np.max(np.abs(rows[:,0]+rows[:,1]-rows[:,2]))),
  "mean_decomposition_abs_error_refSD":float(abs(points[0]+points[1]-points[2])),
  "MCSE_total_variance_identity_abs_error":float(abs(mean_cov[2,2]-(mean_cov[0,0]+mean_cov[1,1]+2*mean_cov[0,1])))}

def state_sha():
 return hashlib.sha256(pickle.dumps((np.random.get_state(),random.getstate()),protocol=5)).hexdigest()

def run(root,output):
 started=time.perf_counter();initial_rng=state_sha();repo=Path(__file__).resolve().parents[1]
 for name,digest in PINS.items():require(sha256_file(repo/name)==digest,"frozen source changed: "+name)
 root=Path(root);results=root/"results";base=results/"accuracy-ss15-v1"
 mp=base/"sigma-native-grid-01/seed-20261111-case-2/manifest.json";payload=mp.with_name("samples.npz")
 ap=base/"sigma-native-grid-assessment-01/seed-20261111-case-2/assessment.json"
 require(sha256_file(mp)==PARENT_SHA and sha256_file(payload)==PAYLOAD_SHA and sha256_file(ap)==ASSESSMENT_SHA,"fixed saved case2 custody differs")
 assessment=json.loads(ap.read_bytes());e=load_script("evaluate_ss15_sigma_native_campaign.py",EVALUATOR_SHA);w,execution_sources=e.source_guard()
 # Parent authentication only: no joint refresh and no replay of random normals.
 parent,values,lineage=w.load_parent(mp,PARENT_SHA,w.kernel.metadata())
 require(parent["schema"]=="posteriorforge.inference.ss15_scientific_campaign.v1" and parent["method"]=="sigma_first"
  and (parent["context_seed"],parent["case"])==(20261111,2),"original neural case2 parent required")
 require(values.shape==(50000,21) and values.dtype==np.float64 and np.isfinite(values).all() and support_mask(values).all(),"complete original accepted rows required")
 require(assessment["parent_manifest_sha256"]==PARENT_SHA and assessment["source_pins"]["evaluator_sha256"]==EVALUATOR_SHA,"audited assessment lineage differs")
 query=np.asarray(load_heldout(results/"banks-v1",2).obs).reshape(1,33)
 require(array_descriptor(query)==parent["raw_scientific_bindings"]["query"],"actual heldout observation binding differs")
 binder=load_script("evaluate_cdf_pilot.py",BINDER_SHA);gatepath=results/"gate-final.json"
 require(sha256_file(gatepath)==binder.BASELINE_GATE_SHA256==assessment["original_gate_sha256"],"immutable original gate changed")
 gate=json.loads(gatepath.read_bytes());arrays,manifest,reference,custody=binder.reference_for_case(results,gate,query,2)
 binder.validate_run_reference_binding(parent,custody)
 require(custody==assessment["reference"],"same qualified case2 reference required")
 _,refmean,refsd,qualification,qualified=_reference_arrays(reference);require(qualified,"qualified numerical moments required")
 mean_error,sd_error=_moment_uncertainties(qualification)
 old=assessment["absolute_physical_assessments"]["before"]["moments"]
 require(np.array_equal(refmean,old["reference_mean"]) and np.array_equal(refsd,old["reference_sd"]),"saved reference moments differ")
 g=geometry();obs=unflatten_observations(query).values["mechanical_2"][0];source_values_hash=array_descriptor(values)
 means,covariances=conditional_moments(values,obs,g)
 indices=np.linspace(0,len(values)-1,33,dtype=int);cm,cc=covariance_space(values[indices],obs,g)
 delta_mean=float(np.max(np.abs(means[indices]-cm)));delta_cov=float(np.max(np.abs(covariances[indices]-cc)))
 require(delta_mean<1e-10 and delta_cov<1e-10,"independent covariance-space conditional identity failed")
 checks=[]
 for row in indices[:3]:
  beta=values[row,g["local_indices"]];alternative=beta+np.asarray([.031,-.027,.019]);h=values[row:row+1,g["global_indices"]];sigma=values[row,g["sigma_index"]]
  def log_target(b):
   local={name:np.asarray([b[j]]) for j,name in enumerate(g["prior"].local_names)}
   local[g["problem"].prior.noise_name("mechanical_2")]=np.asarray([sigma])
   return float(g["problem"].prior.base.conditional_log_prob("mechanical_2",h,b[None,:])[0]+g["problem"].observation_model.local_log_prob("mechanical_2",local,obs[None,:])[0])
  residual=beta-means[row];other=alternative-means[row];p=np.linalg.solve(covariances[row],np.eye(3))
  error=abs((log_target(beta)-log_target(alternative))-(-.5*(residual@p@residual-other@p@other)));checks.append(float(error))
 require(max(checks)<1e-9,"actual prior+likelihood log-ratio conditional identity failed")
 rows=[]
 for name in TARGETS:
  physical=list(g["problem"].parameter_order).index(name);local=g["local_indices"].index(physical)
  data=paired_decomposition(values[:,physical],means[:,local],refmean[physical],refsd[physical],mean_error[physical],sd_error[physical])
  total=data["components"]["total_signed_error"]["signed_refSD"]
  require(abs(abs(total)-old["absolute_bias_reference_sd"][physical])<1e-12,"signed total does not reproduce saved original mean-bias point estimate")
  rows.append({"parameter":name,"physical_index":physical,"neural_parent_mean":float(values[:,physical].mean()),
   "exact_conditional_mean_averaged_over_neural_parent":float(means[:,local].mean()),"reference_mean":float(refmean[physical]),
   "reference_sd":float(refsd[physical]),"reference_mean_error_refSD":float(mean_error[physical]),"reference_sd_error_refSD":float(sd_error[physical]),**data})
 require(array_descriptor(values)==source_values_hash and state_sha()==initial_rng,"input or default NumPy/Python RNG changed")
 rejection=parent["rejection"];rejected=rejection["proposed"]-rejection["accepted_before_final_trim"]
 require(rejection["proposed"]==parent["proposal_buffer"]["original_sampler_counts"]["original_checked_count"],"checked-count consistency differs")
 record={"schema":SCHEMA,"attribution":"Built with PriorLabs-TabPFN","diagnostic_only":True,"candidate_change":False,
  "case":2,"context_seed":20261111,"conditioning_parent":"original neural parent before the later joint-global update",
  "count":50000,"all_rows_used":True,"original_physical_status":assessment["absolute_physical_assessments"]["before"]["status"],
  "hybrid_physical_status_retained":assessment["status"],"gate_or_threshold_waiver":False,
  "source_assumptions":{"scientific_fingerprint":g["problem"].scientific_fingerprint,"prior":g["prior"].manifest(),"forward":g["forward"].manifest(),
   "global_indices":g["global_indices"],"local_indices":g["local_indices"],"sigma_index":g["sigma_index"],
   "likelihood":"y=X beta+forward_offset+Normal(0,sigma_m2^2 I); absolute SD; independent noise prior",
   "conditional":"Vpost=(V0^-1+X^T X/sigma^2)^-1; m=Vpost[V0^-1(population_map G+prior_offset)+X^T(y-forward_offset)/sigma^2]",
   "uses_conditional_prior_covariance_not_global_marginal_covariance":True,
   "other_blocks_cancel_given_globals_and_sigma":True},
  "decomposition":"E_q[beta]-reference_mean = E_q[beta-m(G,sigma,y)] + E_q[m(G,sigma,y)]-reference_mean",
  "interpretation":{"first_term":"accepted-sampler conditional residual; not solely neural-model bias",
   "second_term":"conditional-mean mixing discrepancy under the accepted original joint(G,sigma); not separate global-versus-noise attribution",
   "conditioning_observation":"original scientific raw Float64 heldout y; stored parent globals/noise used unchanged",
   "uncertainty":"Paired row covariance preserves residual/mixing cancellation. MCSE and CLT95 are approximate under iid accepted-row/finite-variance assumptions, not rigorous coverage or posterior accuracy proof. Declared qualified numerical mean/SD error envelopes are propagated separately; no independence or Gaussian law assumed for numerical reference error."},
  "parameters":rows,"independent_math_checks":{"covariance_space_rows":len(indices),"mean_max_abs_difference":delta_mean,"covariance_max_abs_difference":delta_cov,"actual_prior_likelihood_log_ratio_errors":checks},
  "rejection":{"source_sha256":PINS["src/posteriorforge/upstream/npe_pfn/accept_reject_sampler.py"],"whole_vector_support":"all21 finite AND all6 noise bounds [.01,.5]; local Gaussian coordinates unbounded",
   "counters":rejection,"rejected_checked_rows":rejected,"rejected_fraction_of_checked":rejected/rejection["proposed"],
   "surplus_accepted_trimmed":rejection["accepted_before_final_trim"]-50000,"unused_FIFO_rows":parent["proposal_buffer"]["remaining_unused_buffer_count"],
   "saved_parent_directory_files":sorted(p.name for p in mp.parent.iterdir()),"saved_rejected_proposal_arrays_present":False,
   "scope_limit":"Only final accepted/truncated rows are saved. Rejected vectors and their conditional residuals are unavailable; rejection contribution is not identified. A small rejection rate gives no finite mean-error bound for unbounded beta/residuals. No neural-only bias or rejection-causes-bias claim."},
  "custody":{"parent_manifest_sha256":PARENT_SHA,"parent_samples_sha256":PAYLOAD_SHA,"parent_array":source_values_hash,"assessment_sha256":ASSESSMENT_SHA,
   "actual_raw_query":array_descriptor(query),"reference":custody,"reference_moment_qualification":qualification["moment_precision"],
   "original_gate_sha256":sha256_file(gatepath),"script_sha256":sha256_file(__file__),"frozen_sources":PINS,"evaluator_source":execution_sources},
  "default_numpy_python_RNG_preserved":True,"input_unchanged":True,"posterior_draws":0,"model_GPU_calls":0,"reference_integrations":0,
  "wall_seconds":time.perf_counter()-started}
 require(sha256_file(mp)==PARENT_SHA and sha256_file(payload)==PAYLOAD_SHA and sha256_file(ap)==ASSESSMENT_SHA,"saved inputs changed during diagnostic")
 out=Path(output);out.mkdir(parents=True,exist_ok=False);write_json(out/"diagnostic.json",record)
 write_json(out/"checksums.json",{"diagnostic.json":sha256_file(out/"diagnostic.json")})
 print(json.dumps({"output":str(out),"diagnostic_sha256":sha256_file(out/"diagnostic.json"),"wall_seconds":record["wall_seconds"],"parameters":rows,"rejection":record["rejection"]},sort_keys=True),flush=True)
 return record

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--root",required=True);p.add_argument("--output",required=True);a=p.parse_args();run(a.root,a.output)
if __name__=="__main__":main()
