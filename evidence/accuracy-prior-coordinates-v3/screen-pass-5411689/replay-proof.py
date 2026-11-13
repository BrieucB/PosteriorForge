from pathlib import Path
import hashlib,json,sys
from posteriorforge import performance_prior_pair_v3 as p
from posteriorforge.science.analytical_benchmark_v2 import array_descriptor
s=Path('/scratch/project/eu-26-17/eubrieucb/posteriorforge');phase=s/'results/accuracy-prior-coordinates-v3';launch=phase/'screen-launch-02';plan=json.loads((launch/'plan.json').read_bytes());paths=plan['paths']
q=p.load_script(p.QUALIFIER_FILENAME,p.DEPENDENCY_SHA256['scripts/'+p.QUALIFIER_FILENAME]);spec=q.pairs.FIRST_PAIR
bound=q.bound_inputs(spec,banks=paths['banks'],completion_receipt=paths['completed_grid_audit'],training_inventory=paths['codec_inventory'],pair_cpu_receipt=paths['pair_cpu_receipt'],prefix_run=paths['prefix_run'])
inputs={name:array_descriptor(value) for name,value in [('context_theta',bound.theta),('context_obs',bound.stats),('query',bound.query)]}
artifact=Path(paths['output'])/'manifest.json'
assert hashlib.sha256(artifact.read_bytes()).hexdigest()=='56ec6cdc81eeab4c755d24fa8ddae47c21f75bce4c624c43908a59f5da21ffd1'
document,receipt=q.qualification_receipt(spec,artifact,banks=paths['banks'],completion_receipt=paths['completed_grid_audit'],training_inventory=paths['codec_inventory'],reference_training_inventory=paths['codec_reference_inventory'],combined_training_inventory=paths['codec_cpu_receipt'],pair_cpu_receipt=paths['pair_cpu_receipt'],raw_bindings=bound.raw,model_inputs=inputs,prefix_run=paths['prefix_run'])
assert receipt['replay']['passed'] and receipt['replay']['fresh_stages']==21 and 'torch' not in sys.modules
print(json.dumps({'schema':'posteriorforge.v3_actual_screen_CPU_replay.v1','Python':sys.version,'screen_sha256':hashlib.sha256(artifact.read_bytes()).hexdigest(),'receipt':receipt,'NN_GPU_G3_posterior_or_new_world_calls':0,'Torch_imported':False},indent=2,sort_keys=True))
