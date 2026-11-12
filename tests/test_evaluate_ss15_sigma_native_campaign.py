"""Synthetic evaluator tests; no saved scientific cohort or reference is loaded."""
import ast
from copy import deepcopy
import hashlib
import importlib.util
import inspect
from pathlib import Path
import symtable
import builtins
import numpy as np
import pytest
from posteriorforge.artifacts import sha256_file
from posteriorforge.validation import assess_samples as original_assess_samples

ROOT=Path(__file__).resolve().parents[1]
def module(name):
 p=ROOT/"scripts"/name
 s=importlib.util.spec_from_file_location("test_"+p.stem,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
e=module("evaluate_ss15_sigma_native_campaign.py")
w=module("refresh_ss15_campaign_global_block.py")


def toy():
 a=np.repeat(np.arange(21,dtype=np.float64)[None]/20,50000,axis=0)
 a+=np.linspace(-.1,.1,50000)[:,None]
 a[:,w.kernel.metadata()["noise_indices"]]=.25
 return a


def test_unpinned_evaluator_stops_before_input_or_output(tmp_path,monkeypatch):
 monkeypatch.setattr(e,"WRAPPER_SHA",None)
 with pytest.raises(ValueError,match="unpinned"):
  e.run(root=tmp_path,parent_manifest=tmp_path/"missing",hybrid_manifest=tmp_path/"missing",
   hybrid_manifest_sha256="0"*64,execution_screen=tmp_path/"missing",output=tmp_path/"out")
 assert not (tmp_path/"out").exists()


def test_exact_joint_replay_and18_controls_use_actual_case_seed_and_no_refresh(monkeypatch):
 before=toy();parent={"context_seed":20261112,"case":7}
 seed=w.backbone.derive_seed(20261112,w.ROLE,case=7,repetition=0)
 after,receipt=w.refresh_once(before,seed)
 hybrid={"refresh":receipt,"default_RNG_observation":{"preserved":True,"before_sha256":"same","after_sha256":"same",
  "torch_CPU_CUDA_RNG_hashes_observed":False}}
 # Replaying may not call either constructor of a new output cohort.
 monkeypatch.setattr(w,"refresh_once",lambda *args:(_ for _ in ()).throw(AssertionError("refresh called")))
 monkeypatch.setattr(w.kernel,"refresh",lambda *args:(_ for _ in ()).throw(AssertionError("kernel refresh called")))
 proof=e.replay_joint(w,parent,hybrid,before,after)
 assert proof["passed"] and proof["private_seed"]==seed and proof["new_refresh_performed"] is False
 for fault in ("local","global","case","rng"):
  a=after.copy();p=deepcopy(parent);m=deepcopy(hybrid)
  if fault=="local":a[0,4]=np.nextafter(a[0,4],np.inf)
  elif fault=="global":a[0,0]+=1e-6
  elif fault=="case":p["case"]=0
  else:m["refresh"]["seed"]+=1
  with pytest.raises(ValueError):e.replay_joint(w,p,m,before,a)


def test_reference_owner_is_actual_case_not_anchor(monkeypatch,tmp_path):
 events=[];query=np.arange(33,dtype=np.float64)[None]
 parent={"case":7,"context_seed":20261112}
 class Binder:
  def reference_for_case(self,results,gate,q,case):
   assert results==tmp_path and q is query and case==7
   events.append(("reference",case));return "arrays","manifest","actual-case-reference",{"reference_case":case}
  def validate_run_reference_binding(self,p,custody):
   assert p is parent and custody["reference_case"]==7;events.append(("binding",p["case"]))
 def assess(before,after,reference,gate,p):
  assert reference=="actual-case-reference" and p is parent
  events.append(("assessment",p["context_seed"],p["case"]))
  return {"after":{"passed":True}},{"passed":True}
 monkeypatch.setattr(e,"assess_pair",assess)
 result=e.reference_and_assess(tmp_path,{},parent,query,"before","after",Binder())
 assert result[1]["reference_case"]==7
 assert events==[("reference",7),("binding",7),("assessment",20261112,7)]


def test_original_gate_function_and_actual_seed_case_limits_are_used(monkeypatch):
 assert e.assess_samples is original_assess_samples
 template={"integrity":{"passed":True},"reference_qualified":True,"limits":{"cdf":.1},
  "marginal_cdf":{"lower":[.01]*21,"upper":[.02]*21},
  "moments":{name:[1.]*21 for name in ("sample_mean","sample_sd","absolute_bias_reference_sd","sd_ratio")}}
 called=[]
 def fake(values,reference,*,support_mask):
  assert values.shape==(50000,21) and support_mask.all();called.append(reference);return deepcopy(template)
 monkeypatch.setattr(e,"assess_samples",fake)
 gate={"runs":[{"context_seed":20261111,"case":0,"assessment":{"limits":{"cdf":.07}}},
               {"context_seed":20261112,"case":7,"assessment":{"limits":{"cdf":.1}}}]}
 data=toy();assessments,controls=e.assess_pair(data,data.copy(),"same-original-reference",gate,{"context_seed":20261112,"case":7})
 assert controls["passed"] and len(called)==2 and assessments["after"]["limits"]=={"cdf":.1}
 with pytest.raises(ValueError,match="limits"):e.assess_pair(data,data,"ref",gate,{"context_seed":20261111,"case":0})


def test_source_guard_scope_plot_titles_and_no_legacy_gate_or_model_calls():
 tree=ast.parse(Path(e.__file__).read_text())
 calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
 assert not any(isinstance(n.func,ast.Attribute) and n.func.attr in ("verify_run_binding","refresh","refresh_once","fit","predict") for n in calls)
 assert e.FOCUS==(0,4,11,12)
 signature=inspect.signature(e.plots)
 assert "seed" in signature.parameters and "case" in signature.parameters and "full_cdf" in signature.parameters
 plot=inspect.getsource(e.plots)
 assert "seed20261111, case0" not in plot and "mixed FP16/FP32" in plot and "all8 native stage14" in plot
 table=symtable.symtable(Path(e.__file__).read_text(),e.__file__,"exec")
 for name in ("run","authenticate","replay_joint","execution_checks","reference_and_assess"):
  scope=next(c for c in table.get_children() if c.get_name()==name)
  missing=[s.get_name() for s in scope.get_symbols() if s.is_global() and s.is_referenced()
   and s.get_name() not in vars(e) and not hasattr(builtins,s.get_name())]
  assert missing==[]
