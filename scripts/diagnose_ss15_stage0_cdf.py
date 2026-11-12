"""Saved stage0 finite-bar CDF diagnostic; zero posterior/model/simulator calls."""
from pathlib import Path
import json
import numpy as np
from posteriorforge.artifacts import sha256_file,write_json
from posteriorforge.science.analytical_benchmark_v2 import array_descriptor

def run(root,output):
 root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=False)
 results=root/'results';qpath=results/'accuracy-ss15-v1/conditional-qualification-02/manifest.json'
 assert sha256_file(qpath)=='5dc422946763156d9f876b2f07d9781e7b350265746ffa1024854d579f819deb'
 q=json.loads(qpath.read_bytes());row=q['stage_results'][0];golden=row['goldens']['1024']
 gpath=qpath.parent/golden['file'];assert sha256_file(gpath)==golden['sha256']
 with np.load(gpath) as z:
  masses=z['full_mixture_probabilities'];borders=z['physical_borders'];queries=z['queries']
 for key,arr in [('full_mixture_probabilities',masses),('physical_borders',borders),('queries',queries)]:
  assert array_descriptor(arr)==golden['arrays'][key]
 assert np.array_equal(queries,np.repeat(queries[:1],1024,axis=0))
 assert masses.shape==(1024,5000) and borders.shape==(5001,)
 assert np.isfinite(masses).all() and (masses>=0).all() and (np.diff(borders)>0).all()
 apath=results/'accuracy-ss15-v1/pilot-assessment-01/assessment.json'
 assert sha256_file(apath)=='1ed0910634845b72fc443114d3a7e34077f207b4688e9884330589bd83bf114b'
 a=json.loads(apath.read_bytes())
 rpath=results/'reference-grids-v1/case-0/distribution.npz'
 assert sha256_file(rpath)==a['reference']['reference_distribution_sha256']
 assert sha256_file(rpath.parent/'manifest.json')==a['reference']['grid_manifest_sha256']
 with np.load(rpath) as z:directions=z['directions'];x=z['queries']/directions[0,0];ref=z['cdfs'][0]
 assert (np.flatnonzero(directions[0])==[0]).all() and directions[0,0]>0
 precision=a['absolute_physical_assessment']['reference_qualification']['precision']
 assert a['absolute_physical_assessment']['reference_qualified'] and precision==a['reference']['cdf_precision']
 # The normalized saved finite-bar measure only: not a production ICDF moment claim.
 weights=masses[0].astype(np.float64);rawsum=float(weights.sum());weights/=rawsum
 edge_cdf=np.r_[0.,weights.cumsum()];edge_cdf[-1]=1.
 actual=np.interp(x,borders.astype(np.float64),edge_cdf,left=0.,right=1.)
 point=float(np.max(np.abs(actual-ref)))
 upper=max(point,float(np.max(ref[1:]-actual[:-1])),float(np.max(actual[1:]-ref[:-1])),
           float(ref[0]),float(actual[0]),float(1-ref[-1]),float(1-actual[-1]))
 result={'scope':'deterministic normalized saved original FP32 finite-bar CDF versus qualified reference; no production-ICDF moment inference',
  'stage':0,'parameter':'mu_a','posterior_vectors_generated':0,'new_model_calls':0,'new_simulator_calls':0,
  'golden_sha256':golden['sha256'],'qualification_manifest_sha256':sha256_file(qpath),
  'reference_distribution_sha256':sha256_file(rpath),'assessment_sha256':sha256_file(apath),
  'source_sha256':sha256_file(__file__),'saved_probability_raw_sum':rawsum,
  'duplicate_queries_exact':True,'saved_row_probability_max_difference':float(np.max(np.abs(masses-masses[:1]))),
  'reference_precision':precision,'grid_witness':point,'whole_line_lower':max(0.,point-precision),
  'whole_line_upper':min(1.,upper+precision),'reference_grid_count':len(x),
  'FP16_saved_singleton_vs_original_FP32_CDF_drift':row['comparisons']['cached_FP16_vs_original_FP32']['1']['final']['max_cdf_absolute_error'],
  'note':'FP16 probabilities were not archived; its measured finite conditional drift is separate from the saved FP32 CDF. No production density is reconstructed from that scalar.'}
 write_json(output/'diagnostic.json',result);print(json.dumps(result,indent=2));return result

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--output',required=True)
 run(**vars(p.parse_args()))
