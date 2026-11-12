"""Private SS15 wave plumbing; unchanged component, inverse and rejection math.

Built with PriorLabs-TabPFN. Only the authenticated23cc wave's observation
reshape changes33→15. No original function/global/SDK file is replaced.
"""
from __future__ import annotations
import __future__,ast,copy,hashlib,inspect,textwrap
from . import performance_fp16_experiment as frozen
from .artifacts import sha256_file

SCHEMA="posteriorforge.execution.ss15_wave.v1"
ORIGINAL_WAVE_SOURCE_SHA256="1e785186a563f5e28935c6096cd18b57eab042a6961b3e52f01dd61bd0797049"
OBSERVATION_FEATURES=15
_WAVE=None
_WAVE_CODE=None
_DESCRIPTOR=None


def require(value,message):
 if not value:raise RuntimeError(message)

def source_sha256():return sha256_file(__file__)
def ast_sha(node):return hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()


def transform_source(source):
 """One expected shape literal; reverse transformation must recover every AST node."""
 require(hashlib.sha256(source.encode()).hexdigest()==ORIGINAL_WAVE_SOURCE_SHA256,"canonical wave function source changed")
 tree=ast.parse(source);original=copy.deepcopy(tree)
 expected=ast.parse('np.asarray(observation,dtype=np.float32).reshape(1,33)',mode='eval').body
 matches=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and ast.dump(n,include_attributes=False)==ast.dump(expected,include_attributes=False)]
 require(len(matches)==1,"exactly one original33 observation reshape required")
 matches[0].args[1]=ast.Constant(value=15);ast.fix_missing_locations(tree)
 private_sha=ast_sha(tree);reverse=copy.deepcopy(tree)
 changed=ast.parse('np.asarray(observation,dtype=np.float32).reshape(1,15)',mode='eval').body
 candidates=[n for n in ast.walk(reverse) if isinstance(n,ast.Call) and ast.dump(n,include_attributes=False)==ast.dump(changed,include_attributes=False)]
 require(len(candidates)==1,"private15 reshape count differs")
 candidates[0].args[1]=ast.Constant(value=33)
 require(ast.dump(reverse,include_attributes=False)==ast.dump(original,include_attributes=False),"SS15 wave contains undeclared computational changes")
 return tree,{'original_function_source_sha256':ORIGINAL_WAVE_SOURCE_SHA256,'original_ast_sha256':ast_sha(original),
   'private_ast_sha256':private_sha,'reverse_ast_exact':True,'declared_changes':[{'expression':'observation reshape','from':33,'to':15}],
   'unchanged':'all21 stage order, query prefixes, component sampler, RNG streams, inverse arithmetic, close/finally, progress and complete-vector wave'}


def private_wave():
 global _WAVE,_WAVE_CODE,_DESCRIPTOR
 original=frozen.frozen_sampling()
 if _WAVE is None:
  source=textwrap.dedent(inspect.getsource(original.stage_major_wave))
  tree,descriptor=transform_source(source);scope=dict(original.stage_major_wave.__globals__)
  exec(compile(tree,'<posteriorforge-private-ss15-wave>','exec',flags=__future__.annotations.compiler_flag,dont_inherit=True),scope)
  _WAVE=scope['stage_major_wave'];_WAVE_CODE=_WAVE.__code__;_DESCRIPTOR=descriptor
 require(_WAVE.__code__ is _WAVE_CODE,"private SS15 wave code changed")
 require(_WAVE.__globals__['compact_cache'] is original.compact_cache and _WAVE.__globals__['np'] is original.np,
   "private wave dependency identity changed")
 return _WAVE


def execution_metadata():
 private_wave()
 return {'schema':SCHEMA,'module_sha256':source_sha256(),'canonical_sampling_source_sha256':frozen.SAMPLING_SHA,
  'canonical_sampling_commit':frozen.SAMPLING_COMMIT,'observation_features':15,'source_transform':copy.deepcopy(_DESCRIPTOR),
  'component_stage_sampler_unchanged':True,'whole_vector_rejection_unchanged':True,'e71_inverse_unchanged':True,
  'raw33_qualification_claimed':False,'representation_changes_learned_input':True,'attribution':'Built with PriorLabs-TabPFN'}


def stage_major_wave(engine,theta_context,obs_context,observation,proposed_count,query_batch_size,draw,*,stage_draw=None,progress=None):
 return private_wave()(engine,theta_context,obs_context,observation,proposed_count,query_batch_size,draw,stage_draw=stage_draw,progress=progress)
