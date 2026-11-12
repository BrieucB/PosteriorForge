"""Exact fixed-design Gaussian sufficient statistics for the frozen raw33 problem.

Built with PriorLabs-TabPFN. This changes observations only, never the21 physical
targets, priors, conditional hierarchy, support, references or rejection gate.

For a ten-point mechanical block z=y-offset=X beta+sigma epsilon, choose fixed
orthonormal Q spanning X. Write u=Q.T z and E=||(I-Q Q.T)z||². Then
  ||z-X beta||² = ||u-Q.T X beta||² + E,
  log p(y|beta,sigma) = -10 log(sigma)-5 log(2 pi)
                       -(||u-Q.T X beta||²+E)/(2 sigma²).
Thus all parameter dependence, including unknown sigma, is preserved by(u,E).
Keeping only the three fitted/signal coordinates is not sufficient for sigma.

An orthogonal completion has unit absolute Jacobian. Integrating the direction
of its seven-dimensional residual gives
  p(u,E|beta,sigma) / p(y|beta,sigma)
    = pi**(7/2) / Gamma(7/2) * E**(5/2),  E>0.
This factor is observation dependent, parameter independent. It cancels from
the posterior, while E/sigma² and sigma**-10 must remain. E/sigma²~chi²_7.
E=0 is a measure-zero boundary with zero summary density, not a reason to clip.

The numerical transform uses original float64 saved observations, a fixed
integer polynomial basis (no data fitting/QR sign choice), explicit residual
squares, then a SINGLE final float32 cast for the PFN. Algebraic exactness is
over real arithmetic; floating-point parity is measured separately.
"""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np
from scipy.special import gammaln
from .science.analytical_benchmark_v2 import build_problem,array_descriptor,DATASET_IDS,OBSERVATION_COUNTS

SCHEMA="posteriorforge.representation.ss15.v1"
REPRESENTATION_ID="raw33_ss15_orthonormal_energy_v1"
RAW_DIMENSION=33
OBSERVATION_DIMENSION=15
RESIDUAL_DOF=7
# Columns constant, linear, quadratic, in observed x-grid order0..9.
BASIS_NUMERATORS=((1,1,1,1,1,1,1,1,1,1),
                  (-9,-7,-5,-3,-1,1,3,5,7,9),
                  (6,2,-1,-3,-4,-4,-3,-1,2,6))
BASIS_SQUARED_NORMS=(10,330,132)
FEATURE_ORDER=tuple(f"mechanical_{i}.{name}" for i in range(1,4)
                    for name in ("signal_constant","signal_linear","signal_quadratic","residual_energy"))+tuple(
                      f"acoustic_{i}.raw" for i in range(1,4))
PINNED_SCIENCE_SHA256={
 "analytical_benchmark_v2.py":"8d5f286f6d8512391dea2ae036313bec05782d28c7803f336490cbcdce3bd99f",
 "hierarchical_noise.py":"cbf0b8c956362a5eaa31923e9b94a3db3e3ede2072603add51e78105b8c0cbf0",
 "hierarchical_exact.py":"db001593cc7307b1290b981622705961c5953001a6c1245fae5ec779938e9687",
 "hierarchical_problem.py":"03aa2d8a3d279783cf6c599f1400f253f3db2e1e7eb027aff5b7cd1b48fe12d0"}

def require(value,message):
 if not value:raise ValueError(message)
def source_sha256():return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
def orthonormal_basis():
 q=np.asarray(BASIS_NUMERATORS,dtype=np.float64).T/np.sqrt(np.asarray(BASIS_SQUARED_NORMS,dtype=np.float64))
 q.setflags(write=False)
 return q

def _problem():
 root=Path(__file__).with_name("science")
 require(all(hashlib.sha256((root/name).read_bytes()).hexdigest()==value for name,value in PINNED_SCIENCE_SHA256.items()),
   "frozen scientific source changed; SS15 not authorized for another forward")
 p=build_problem();q=orthonormal_basis()
 require(tuple(p.dataset_ids)==DATASET_IDS and tuple(b.observation_size for b in p.observation_model.blocks)==OBSERVATION_COUNTS,
   "fixed dataset/observation order changed")
 require(np.max(np.abs(q.T@q-np.eye(3)))<2e-15,"fixed basis lost orthonormality")
 x=np.linspace(0.,1.,10);design=np.column_stack((x,x*x,np.ones(10)))
 for block in p.observation_model.blocks[:3]:
  require(np.array_equal(block.design,design) and np.array_equal(block.offset,np.zeros(10))
    and np.array_equal(block.covariance,np.eye(10)) and len(block.local_names)==3,
    "SS15 requires declared rank3 identity-covariance mechanical blocks")
  require(np.max(np.abs(design-q@(q.T@design)))<2e-15,"fixed basis does not span actual design")
 require(len(p.parameter_order)==21 and len(p.prior.global_names)==3 and len(p.prior.noise_names)==6,
   "physical hierarchy/independent noise order changed")
 return p

def _raw_rows(raw33):
 rows=np.asarray(raw33)
 require(rows.dtype==np.dtype(np.float64),"original float64 observations required BEFORE any model cast")
 require(rows.ndim==2 and rows.shape[1]==33 and len(rows)>0 and np.isfinite(rows).all(),
   "finite original observations[n,33] required")
 return np.ascontiguousarray(rows)

def transform_observations(raw33):
 """Return float64[n,15] from ORIGINAL saved float64[n,33], with no fitted state."""
 raw=_raw_rows(raw33);problem=_problem();q=orthonormal_basis();parts=[]
 for index,block in enumerate(problem.observation_model.blocks[:3]):
  z=raw[:,index*10:(index+1)*10]-block.offset
  # Fixed per-row reductions make feature bytes independent of batch/chunks.
  signal=np.column_stack([np.sum(z*q[:,k],axis=1) for k in range(3)])
  projection=(signal[:,0,None]*q[:,0]+signal[:,1,None]*q[:,1]+signal[:,2,None]*q[:,2])
  residual=z-projection
  energy=np.square(residual).sum(axis=1)
  parts.extend((signal,energy[:,None]))
 parts.append(raw[:,30:33].copy())
 result=np.ascontiguousarray(np.concatenate(parts,axis=1),dtype=np.float64)
 require(result.shape==(len(raw),15) and np.isfinite(result).all(),"nonfinite sufficient statistics")
 return result

def transform_for_model(raw33):
 """Exactly one final cast after the source-bound float64 representation."""
 transformed=transform_observations(raw33).astype(np.float32)
 require(np.isfinite(transformed).all(),"SS15 cannot be represented as finite float32")
 return np.ascontiguousarray(transformed)

def _summary_rows(summary):
 rows=np.asarray(summary,dtype=np.float64)
 require(rows.ndim==2 and rows.shape[1]==15 and len(rows)>0 and np.isfinite(rows).all(),
   "finite summaries[n,15] required")
 require(np.all(rows[:,[3,7,11]]>=0),"residual energies must be nonnegative")
 return rows

def raw_likelihood_factor(parameters,summary):
 """Same normalized raw-data likelihood, evaluated from sufficient statistics.

 This is p(y|theta)'s factor, NOT the density of SS15 with respect to15D Lebesgue
 measure. Its theta dependence is exact, and its sigma normalization is retained.
 """
 p=_problem();theta=np.asarray(parameters,dtype=np.float64);s=_summary_rows(summary)
 require(theta.ndim==2 and theta.shape[1]==21 and len(theta)>0 and np.isfinite(theta).all(),
   "finite parameters[n,21] required")
 require(len(s) in (1,len(theta)),"one shared observation or one per parameter row required")
 q=orthonormal_basis();result=np.zeros(len(theta),dtype=np.float64)
 for index,(block,noise) in enumerate(zip(p.observation_model.blocks,p.prior.noise_names)):
  beta=theta[:,[p.parameter_order.index(name) for name in block.local_names]]
  sigma=theta[:,p.parameter_order.index(noise)];valid=sigma>0;safe=np.where(valid,sigma,1.)
  if index<3:
   difference=s[:,4*index:4*index+3]-beta@(q.T@block.design).T
   sse=np.square(difference).sum(axis=1)+s[:,4*index+3]
  else:
   prediction=beta@block.design.T+block.offset
   sse=np.square(s[:,12+index-3:13+index-3]-prediction).sum(axis=1)
  logs=-.5*sse/np.square(safe)-block.observation_size*(np.log(safe)+.5*np.log(2*np.pi))
  result+=np.where(valid,logs,-np.inf)
 return result

def residual_measure_log_factor(summary):
 """Theta-independent residual-sphere integration/Jacobian factor."""
 s=_summary_rows(summary);energies=s[:,[3,7,11]]
 with np.errstate(divide="ignore"):
  factors=(RESIDUAL_DOF/2*np.log(np.pi)-gammaln(RESIDUAL_DOF/2)
           +(RESIDUAL_DOF/2-1)*np.log(energies))
 return factors.sum(axis=1)

def summary_log_density(parameters,summary):
 """Density of transformed SS15: three Gaussian signals × three radial chi²."""
 return raw_likelihood_factor(parameters,summary)+residual_measure_log_factor(summary)

def representation_metadata():
 p=_problem();q=orthonormal_basis();design=p.observation_model.blocks[0].design
 return {"schema":SCHEMA,"representation_id":REPRESENTATION_ID,"module_sha256":source_sha256(),
   "attribution":"Built with PriorLabs-TabPFN","scientific_fingerprint":p.scientific_fingerprint,
   "scientific_sources_sha256":dict(PINNED_SCIENCE_SHA256),"raw_dimension":33,"observation_dimension":15,
   "raw_dataset_order":list(DATASET_IDS),"raw_counts":list(OBSERVATION_COUNTS),"feature_order":list(FEATURE_ORDER),
   "physical_parameter_order":list(p.parameter_order),"global_parameters_retained":list(p.prior.global_names),
   "noise_parameters_retained":list(p.prior.noise_names),"residual_degrees_of_freedom_per_mechanical_block":7,
   "basis_numerators":[list(row) for row in BASIS_NUMERATORS],"basis_squared_norms":list(BASIS_SQUARED_NORMS),
   "basis_float64":array_descriptor(q),"mechanical_design":array_descriptor(design),
   "projected_design_float64":array_descriptor(q.T@design),
   "maximum_basis_orthonormality_error":float(np.max(np.abs(q.T@q-np.eye(3)))),
   "maximum_design_projection_error":float(np.max(np.abs(design-q@(q.T@design)))),
   "input_dtype":"original saved float64 before any model cast","statistics_dtype":"float64",
   "model_features_dtype":"float32; one final cast","residual_energy":"sum of explicit orthogonal residual squares; no clipping",
   "unknown_noise_likelihood_normalization":"sigma^-10 for each mechanical block, including residual energy",
   "summary_density_measure_factor":"pi^(7/2)/Gamma(7/2)*E^(5/2) per mechanical block; theta independent",
   "posterior_sufficiency":"exact for frozen Gaussian design in real arithmetic; numeric parity audited separately",
   "global_conditioning":"all15 statistics at every target stage; graph-local factorization not implemented",
   "target_transform":False,"new_worlds":False,"prior_changed":False,"sampler_changed":False}

def representation_cache_key(raw33):
 """Bind exact source/geometry, original arrays AND the final model feature bytes."""
 raw=_raw_rows(raw33);features=transform_for_model(raw);metadata=representation_metadata()
 record={"representation":metadata,"raw_observations":array_descriptor(raw),"model_features":array_descriptor(features)}
 return hashlib.sha256(json.dumps(record,sort_keys=True,separators=(",",":")).encode()).hexdigest()
