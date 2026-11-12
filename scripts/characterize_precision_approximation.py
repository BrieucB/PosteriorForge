#!/usr/bin/env python3
"""Approximate precision characterization; unchanged strict failures remain visible.

Built with PriorLabs-TabPFN. Stages6/10/13 continue after numerical drift flags;
hard validity, source, arithmetic and RNG failures still abort. No qualification.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import importlib.util
import json
import os
import resource
import random
import sys
from pathlib import Path
import time
import numpy as np

SCHEMA="posteriorforge.precision_approximation_characterization.v1"
UTILITY_SHA="6c927a5c278c7fc54b556ac557ecef64f5e6338f5ae13bf466f3b1e6a4640f90"
RECOVERY_SHA="83064f54712014b5c4c66569e854872caff03593cfdc337f2472a7b5723aea02"
STAGES=(6,10,13)
ROWS=(128,1024)
THRESHOLDS={"max_cdf_absolute":2e-5,"max_row_total_variation":1e-4,
            "normalization_absolute":1e-5,"quantile_coordinate_span_fraction":2e-6}


def require(value,message):
    if not value:raise RuntimeError(message)


def load_helper(name,expected):
    path=Path(__file__).with_name(name)
    require(hashlib.sha256(path.read_bytes()).hexdigest()==expected,"pinned helper changed: "+name)
    spec=importlib.util.spec_from_file_location("_precision_"+path.stem,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def exact_prediction(left,right,torch):
    def same(a,b):
        return tuple(a.shape)==tuple(b.shape) and a.dtype==b.dtype and torch.equal(
            a.detach().cpu().contiguous().view(torch.uint8),b.detach().cpu().contiguous().view(torch.uint8))
    return same(left["logits"],right["logits"]) and len(left["members"])==len(right["members"])==8 and all(
        same(a[key],b[key]) for a,b in zip(left["members"],right["members"])
        for key in ("raw_logits","translated_probabilities","from_borders","to_borders"))


def comparison_arms(u0,up,cp,g0,utility):
    return {"uncached_FP32_to_uncached_precision":utility.compare_predictions(u0,up),
            "uncached_precision_to_cached_precision_and_declared_views":utility.compare_predictions(up,cp),
            "qualified_FP32_GQA_hybrid_to_cached_precision":utility.compare_predictions(g0,cp),
            "uncached_FP32_to_qualified_FP32_GQA_hybrid":utility.compare_predictions(u0,g0)}


def reconstruction_checks(components,reference,utility):
    raw=np.stack([utility._numpy(member["translated_probabilities"]) for member in components["members"]]).astype(np.float64)
    masses=raw.sum(-1);weights=masses/masses.sum(0,keepdims=True)
    ideal=raw.mean(0);ideal/=ideal.sum(-1,keepdims=True)
    actual=np.stack([utility._numpy(prediction["logits"].softmax(-1)) for prediction in components["component_predictions"]]).astype(np.float64)
    target=utility._numpy(reference["logits"].softmax(-1))
    return {"normalized_actual_raw_mean_vs_reference":utility.compare_probabilities(ideal,target),
            "mass_corrected_actual_component_outputs_vs_reference":utility.compare_probabilities((actual*weights[:,:,None]).sum(0),target),
            "scope":"direct probability comparisons, not sums of separate CDF error bounds"}


def bound_sources(qualification,context,query,checkpoint):
    from posteriorforge.artifacts import sha256_file
    from posteriorforge.science.analytical_benchmark_v2 import build_problem,array_descriptor
    from posteriorforge import execution_gqa_hybrid as gqa
    recovery=load_helper("recover_gqa_hybrid_qualification.py",RECOVERY_SHA)
    q=recovery.verify_recovery_artifact(qualification)
    expected={"status":"gqa_hybrid_screen_pass","full21_gqa_hybrid_qualified":True,
              "full21_component_qualified":True,"full21_sampler_qualified":True,"full21_qualified":False,
              "cache_only_full21_qualified":False,"thresholds":THRESHOLDS,"fallback_map":{"10":[5]},
              "module_sha256":gqa.execution_source_sha256(),"checkpoint_sha256":sha256_file(checkpoint),
              "context_theta":array_descriptor(context.theta),"context_obs":array_descriptor(context.obs),
              "query":array_descriptor(query),"scientific_fingerprint":build_problem().scientific_fingerprint,
              "parameter_order":list(build_problem().parameter_order)}
    for key,value in expected.items():require(q.get(key)==value,"qualified FP32 comparator binding mismatch: "+key)
    return q



def _model_tensor_bytes(tensor):
    """Read original contiguous storage bytes, including zero-dimensional BF16."""
    import torch
    return tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def _model_tensor_values(tensor):
    """Python scalars preserve BF16 values; no cast or write to model tensors."""
    return tensor.detach().cpu().reshape(-1).tolist()


def private_model_contract(utility):
    """Clone only the two NumPy conversions which cannot represent BF16."""
    import ast,copy,inspect,textwrap
    from types import CodeType
    path=Path(utility.__file__).resolve()
    require(hashlib.sha256(path.read_bytes()).hexdigest()==UTILITY_SHA,"pinned model-contract utility source changed")
    function=utility.model_contract
    compiled=compile(path.read_text(),str(path),"exec",dont_inherit=True)
    expected=next(code for code in compiled.co_consts if isinstance(code,CodeType) and code.co_name=="model_contract")
    require(function.__code__==expected,"active model_contract differs from pinned source")
    node=ast.parse(textwrap.dedent(inspect.getsource(function))).body[0];original=copy.deepcopy(node)
    expressions={"tensor.detach().cpu().contiguous().numpy().tobytes()":"_pf_model_tensor_bytes(tensor)",
                 "tensor.detach().cpu().numpy().reshape(-1)":"_pf_model_tensor_values(tensor)"}
    replacements={ast.dump(ast.parse(a,mode="eval").body,include_attributes=False):
                  ast.parse(b,mode="eval").body for a,b in expressions.items()}
    counts={key:0 for key in replacements}
    class Replace(ast.NodeTransformer):
        def visit_Call(self,child):
            key=ast.dump(child,include_attributes=False)
            if key in replacements:
                counts[key]+=1;return copy.deepcopy(replacements[key])
            return self.generic_visit(child)
    node=Replace().visit(node)
    require(list(counts.values())==[1,1],"exactly one byte-hash and one scalar-values conversion required")
    reverse={ast.dump(ast.parse(b,mode="eval").body,include_attributes=False):
             ast.parse(a,mode="eval").body for a,b in expressions.items()}
    class Undo(ast.NodeTransformer):
        def visit_Call(self,child):
            key=ast.dump(child,include_attributes=False)
            return copy.deepcopy(reverse[key]) if key in reverse else self.generic_visit(child)
    restored=Undo().visit(copy.deepcopy(node))
    require(ast.dump(restored,include_attributes=False)==ast.dump(original,include_attributes=False),
            "metadata clone changed original model-contract arithmetic or fields")
    namespace=dict(function.__globals__);namespace.update(
        _pf_model_tensor_bytes=_model_tensor_bytes,_pf_model_tensor_values=_model_tensor_values)
    module=ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[]))
    exec(compile(module,"<posteriorforge-bf16-safe-model-contract>", "exec",dont_inherit=True),namespace)
    descriptor={"schema":"posteriorforge.precision.model_contract_observer.v1","utility_source_sha256":UTILITY_SHA,
        "original_function_ast_sha256":hashlib.sha256(ast.dump(original,include_attributes=False).encode()).hexdigest(),
        "private_function_ast_sha256":hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest(),
        "ast_restoration_exact":True,"declared_changes":{"byte_hash_conversion":1,"scalar_values_conversion":1},
        "tensor_hash":"original contiguous bytes via reshape(-1).view(torch.uint8).numpy().tobytes()",
        "tensor_values":"original dtype/shape retained; read detached CPU tensor via reshape(-1).tolist()",
        "model_tensor_casts":0,"model_tensor_writes":0,"rng_calls":0,"original_utility_modified":False}
    return namespace["model_contract"],descriptor


def model_summary(regressor,utility):
    observer,descriptor=private_model_contract(utility)
    contract=observer(regressor)
    require(regressor.softmax_temperature==.9
            and str(regressor.bardist_.borders.dtype)=="torch.float32"
            and str(regressor.renormalized_criterion_.borders.dtype)=="torch.float32",
            "frozen FP32 border/temperature postprocessing contract changed")
    require(contract["input_normalization_train_only_verified"],"actual train-only normalization required")
    for module in contract["modules"]:
        if module["type"]=="InputNormalizationEncoderStep":
            for name,item in module["tensors"].items():
                if item.get("dtype")!="torch.float32":
                    raise RuntimeError("input normalization statistics left FP32: "+name)
    # Under AMP, forced_dtype=None is intentional, not an FP32 guarantee.
    return {"contract":contract,"contract_metadata_observer":descriptor,"target_mean":regressor.y_train_mean_,"target_sd":regressor.y_train_std_,
            "canonical_borders_dtype":str(regressor.bardist_.borders.dtype),
            "physical_borders_dtype":str(regressor.renormalized_criterion_.borders.dtype),
            "temperature":regressor.softmax_temperature}



# New characterization-only producer. The original bounded screen remains frozen.
ORIGINAL_PRODUCER_SHA="0ead780c3ca3d3a3789a90a0a0edc650d576ba6623c23831cc5e4a8140de5fc0"
PRECISION_SHA="051f477ff4b9ab8bbcfbaf3c1afa4dcf6b111d2f2fbcf9efb9c94a00d2b2ad3d"
BAR_SOURCE_SHA="3e753389dcb8f3970452e82ff832d40857cf3def29b60f6d4f95c0e1b52937bf"
CENTRAL_U=(.1,.5,.9)
ENDPOINT_U=(0.,float(np.nextafter(np.float32(1),np.float32(0))),1.)
APPROXIMATE_FULL_RUN_REVIEW={"cdf_change_flag":.01,"mean_change_ref_sd_flag":.05,
                           "sd_relative_change_flag":.05,"absolute_physical_status_visible":True,
                           "three_stages_do_not_qualify_all21_or_a_posterior":True}

def characterize_sources():
    from posteriorforge import execution_precision_probe as precision
    require(hashlib.sha256(Path(__file__).with_name("boundedprobe_precision.py").read_bytes()).hexdigest()
            ==ORIGINAL_PRODUCER_SHA,"frozen original strict producer changed")
    require(precision.source_sha256()==PRECISION_SHA,"frozen precision source changed")
    precision.require_sources()
    return {"original_strict_producer_sha256":ORIGINAL_PRODUCER_SHA,
            "precision_module_sha256":PRECISION_SHA,"originals_modified":False}


def timed_all8(engine,sets,torch,*,repetitions=3):
    """Same calls/instrumentation for both arms; preparation excluded."""
    require(type(repetitions) is int and 1<=repetitions<=5,"bounded timing repetitions")
    result={}
    for name in ("128","1024"):
        for member in range(8):engine.predict_component(member,sets[name],capture_member=False)
        torch.cuda.synchronize();times=[]
        for _ in range(repetitions):
            torch.cuda.synchronize();started=time.perf_counter()
            for member in range(8):engine.predict_component(member,sets[name],capture_member=False)
            torch.cuda.synchronize();times.append(time.perf_counter()-started)
        result[name]={"seconds":times,"median_seconds":float(np.median(times)),
                      "warmup_all8_sweeps":1,"timed_all8_sweeps":repetitions}
    return result


def hard_prediction_validity(prediction):
    """Hard validity only; proximity to a comparator is a separate strict flag."""
    tensors=[prediction["logits"]]+[p["logits"] for p in prediction.get("component_predictions",[])]
    raw=[m["translated_probabilities"] for m in prediction["members"]]
    def numpy(x):return x.detach().cpu().numpy()
    normalization=[];masses=[];passed=len(raw)==8
    for logits in tensors:
        x=numpy(logits);p=numpy(logits.softmax(-1)).astype(np.float64)
        passed=passed and str(logits.dtype)=="torch.float32" and not np.isnan(x).any() and not np.isposinf(x).any()
        error=float(np.max(np.abs(p.sum(-1)-1)))
        normalization.append(error);passed=passed and np.isfinite(p).all() and np.min(p)>=0 and error<=1e-5
    for value in raw:
        p=numpy(value).astype(np.float64);mass=p.sum(-1)
        masses.append({"min":float(mass.min()),"max":float(mass.max())})
        passed=passed and str(value.dtype)=="torch.float32" and np.isfinite(p).all() and np.min(p)>=0 and np.all(mass>0) and np.all(mass<=1.00001) and np.max(np.abs(mass-1))<=1e-5
    actual_mass_checks=[]
    components=prediction.get("component_predictions",[])
    passed=passed and len(components) in (0,8)
    for member,component in zip(raw,components):
        expected=numpy(member).astype(np.float64).sum(-1)
        observed=numpy(component["component_raw_mass"]).astype(np.float64)
        tolerance=8*member.shape[-1]*np.finfo(np.float64).eps
        error=float(np.max(np.abs(expected-observed)))
        valid=bool(observed.shape==expected.shape and np.isfinite(observed).all() and np.all(observed>0)
                   and np.all(observed<=1.00001) and component["head_mass_envelope"]==1.00001 and error<=tolerance)
        actual_mass_checks.append({"passed":valid,"max_fp64_sum_error":error,"fp64_roundoff_bound":tolerance})
        passed=passed and valid
    return {"passed":bool(passed),"actual_component_mass_checks":actual_mass_checks,"max_normalization_error":max(normalization),
            "raw_head_masses":masses,"mass_envelope":1.00001,"normalization_absolute":1e-5}


def comparison_hard_validity(comparison):
    return bool(comparison["exact_final_borders"] and comparison["final_float32"] and comparison["eight_members"]
        and comparison["inverse_cdf"]["passed"] and all(m["configuration_identity_equal"]
        and m["exact_borders"] and m["float32"] for m in comparison["members"]))


def strict_and_hard_outcome(row):
    sessions=("private_FP32_session","uncached_FP32_session","uncached_precision_session","cached_precision_session")
    strict=bool(row["rng_preserved"] and row["computed_prefill_witness"]["passed"]
        and all(v["passed"] for c in row["comparisons"].values() for v in c.values())
        and all(v["passed"] for c in row["actual_precision_component_reconstruction"].values()
                for k,v in c.items() if k!="scope")
        and all(row[s]["aliases_and_flags_restored"] for s in sessions))
    hard=bool(row["rng_preserved"] and row["computed_prefill_witness"]["passed"]
        and row["computed_query_witness"]["passed"] and row["target_statistics_and_border_types_preserved"]
        and all(v["passed"] for values in row["hard_probability_validity"].values() for v in values.values())
        and all(comparison_hard_validity(v) for c in row["comparisons"].values() for v in c.values())
        and all(row[s]["aliases_and_flags_restored"] for s in sessions))
    return {"strict_screen_passed":strict,"hard_validity_passed":hard,
            "strict_failure_preserved":not strict,"approximation_accepted":False,
            "scope":"hard-valid characterization only; strict numerical proximity flags are unchanged"}


def _observed_stock(criterion,logits,u):
    """AST clone changes only final return into assign/observe/return."""
    import ast,copy,inspect,textwrap,types
    import torch
    function=type(criterion).icdf
    # FullSupport inherits the same finite BarDistribution.icdf.
    path=Path(inspect.getsourcefile(function))
    require(hashlib.sha256(path.read_bytes()).hexdigest()==BAR_SOURCE_SHA,"exact pinned finite stock ICDF source required")
    node=ast.parse(textwrap.dedent(inspect.getsource(function))).body[0]
    # Bind the ACTIVE method code to the pinned source, not only its filename.
    from types import CodeType
    compiled=compile(path.read_text(),str(path),"exec",dont_inherit=True)
    cls=next(c for c in compiled.co_consts if isinstance(c,CodeType) and c.co_name=="BarDistribution")
    expected=next(c for c in cls.co_consts if isinstance(c,CodeType) and c.co_name=="icdf")
    require(function.__code__==expected,"active stock ICDF code differs from pinned file source")
    original=copy.deepcopy(node);last=node.body[-1]
    require(isinstance(last,ast.Return) and len([n for n in ast.walk(node) if isinstance(n,ast.Return)])==1,
            "one original final ICDF return required")
    assigned=ast.Assign(targets=[ast.Name(id="_pf_value",ctx=ast.Store())],value=last.value)
    names=("idx","cumprobs","rest_prob","probs","left_border","right_border","_pf_value")
    call=ast.Expr(value=ast.Call(func=ast.Name(id="_pf_observe",ctx=ast.Load()),
        args=[ast.Name(id=n,ctx=ast.Load()) for n in names],keywords=[]))
    node.body[-1:]=[assigned,call,ast.Return(value=ast.Name(id="_pf_value",ctx=ast.Load()))]
    restored=copy.deepcopy(node);restored.body[-3:]=[ast.Return(value=restored.body[-3].value)]
    require(ast.dump(restored,include_attributes=False)==ast.dump(original,include_attributes=False),
            "stock observer changed arithmetic")
    captured={}
    def observe(index,before,rest,p,left,right,value):
        selected=p.gather(-1,index[...,None]).squeeze(-1)
        start=before.gather(-1,index[...,None]).squeeze(-1)
        end=before.gather(-1,(index+1)[...,None]).squeeze(-1)
        for key,x in (("index",index),("left_cdf",start),("right_cdf",end),("cdf_terminal",before[...,-1]),
                      ("rest",rest),("mass",selected),("left",left),("right",right),("value",value)):
            captured[key]=x.detach().cpu().numpy().copy()
        captured["probabilities"]=p.detach().cpu().numpy().copy()
        captured["cumulative"]=before[...,1:].detach().cpu().numpy().copy()
    namespace=dict(function.__globals__);namespace["_pf_observe"]=observe
    module=ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[]))
    exec(compile(module,"<posteriorforge-stock-icdf-read-only-observer>","exec",dont_inherit=True),namespace)
    before_rng=torch.get_rng_state().clone()
    value=namespace[function.__name__](criterion,logits,float(np.float32(u)))
    require(torch.equal(before_rng,torch.get_rng_state()),"read-only inverse observer consumed CPU RNG")
    captured["u"]=float(np.float32(u))
    captured["arithmetic_ast_restoration_exact"]=True
    return value,captured


def replay_batch32(capture,borders):
    """Independent NumPy FP32 arithmetic using the SAME batch's actual p/cumsum."""
    p=capture["probabilities"];cdf=capture["cumulative"];b=np.asarray(borders,np.float32);u=np.float32(capture["u"])
    index=np.minimum((cdf<u).sum(-1),p.shape[-1]-1)
    row=np.arange(len(p));previous=np.where(index==0,np.float32(0),cdf[row,np.maximum(index-1,0)])
    with np.errstate(invalid="ignore",divide="ignore",over="ignore",under="ignore"):
        rest=np.subtract(u,previous,dtype=np.float32)
        width=np.subtract(b[index+1],b[index],dtype=np.float32)
        product=np.multiply(width,rest,dtype=np.float32)
        return np.add(b[index],np.divide(product,p[row,index],dtype=np.float32),dtype=np.float32)


def _json_safe(value):
    if isinstance(value,dict):return {k:_json_safe(v) for k,v in value.items()}
    if isinstance(value,list):return [_json_safe(v) for v in value]
    if isinstance(value,float) and not np.isfinite(value):return "nan" if np.isnan(value) else "inf" if value>0 else "-inf"
    return value


def _stats(values):
    finite=np.asarray(values,np.float64);finite=finite[np.isfinite(finite)]
    return {"finite_count":int(len(finite)),"max":float(finite.max()) if len(finite) else None,
            "median":float(np.median(finite)) if len(finite) else None,
            "p95":float(np.quantile(finite,.95)) if len(finite) else None,
            "mean":float(finite.mean()) if len(finite) else None}


class ForensicArchive:
    """At most18 logical row records/stage, no raw full-batch heads stored."""
    def __init__(self,maximum=64):
        require(type(maximum) is int and 0<=maximum<=64,"bounded forensic record cap")
        self.maximum=maximum;self.arrays={};self.records=[];self.logical_rows=None
    def add(self,prediction,distribution,capture,row,*,stage,arm,side,category,other_value):
        if len(self.records)>=self.maximum:return
        key="record_"+str(len(self.records));member=None if distribution=="final" else int(distribution.split("_")[-1])
        logits=prediction["logits"] if member is None else prediction["component_predictions"][member]["logits"]
        criterion=prediction["criterion"] if member is None else prediction["component_predictions"][member]["criterion"]
        self.arrays[key+"_logits"]=logits[row].detach().cpu().numpy().copy()
        self.arrays[key+"_physical_borders"]=criterion.borders.detach().cpu().numpy().copy()
        # These are the ORIGINAL whole-batch CPU softmax/cumsum outputs. Saved
        # rows are selected only afterwards; no isolated-row CDF is recomputed.
        self.arrays[key+"_actual_probabilities"]=capture.get("_actual_probabilities",capture.get("probabilities"))[row].copy()
        self.arrays[key+"_actual_cumulative"]=capture.get("_actual_cumulative",capture.get("cumulative"))[row].copy()
        self.arrays[key+"_native_icdf_value"]=np.asarray(capture["value"][row],dtype=np.float32)
        self.arrays[key+"_uniform"]=np.asarray(capture["u"],dtype=np.float32)
        if capture.get("production_finite_sampler_value") is not None:
            self.arrays[key+"_production_finite_sampler_value"]=np.asarray(capture["production_finite_sampler_value"][row],dtype=np.float32)
        members=prediction["members"] if member is None else [prediction["members"][member]]
        for i,m in enumerate(members):
            for field in ("raw_logits","translated_probabilities","from_borders","to_borders"):
                x=m[field];self.arrays[key+"_member_"+str(i)+"_"+field]=x[row].detach().cpu().numpy().copy() if field in ("raw_logits","translated_probabilities") else x.detach().cpu().numpy().copy()
        record={"key":key,"stage":stage,"arm":arm,"side":side,"distribution":distribution,"row":int(row),
                "category":category,"u":capture["u"],"other_value":float(other_value),
                "border_span":float(criterion.borders[-1].double()-criterion.borders[0].double()),
                "actual_intermediates":{k:float(capture[k][row]) for k in
                    ("index","left_cdf","right_cdf","cdf_terminal","rest","mass","left","right","value")},
                "intermediates_from_same_whole_batch_computation":True,"query_set":"1024","native_inverse":"SDK criterion.icdf (inherited finite BarDistribution method)",
                "production_inverse":"frozen e71 finite_bar_icdf with identical whole-batch logits/borders/CPU FP32 uniforms" if capture["u"]<1 else None,
                "production_finite_sampler_value":None if capture.get("production_finite_sampler_value") is None else float(capture["production_finite_sampler_value"][row]),
                "native_vs_production_whole_batch_bitwise_equal":capture.get("native_vs_finite_sampler_bitwise_equal"),
                "u_one_native_only_stress":capture["u"]==1.,
                "original_prefix_sample_row":None if self.logical_rows is None else int(self.logical_rows[row]),
                "measured_roundoff_tail_band":None if "roundoff_tail_band" not in capture else float(capture["roundoff_tail_band"][row])}
        record["outside_selected_bin"]=record["actual_intermediates"]["value"]<record["actual_intermediates"]["left"] or record["actual_intermediates"]["value"]>record["actual_intermediates"]["right"]
        record["outside_physical_borders"]=record["actual_intermediates"]["value"]<float(criterion.borders[0]) or record["actual_intermediates"]["value"]>float(criterion.borders[-1])
        self.records.append(record)
    def save(self,output):
        from posteriorforge.artifacts import sha256_file
        from posteriorforge.science.analytical_benchmark_v2 import array_descriptor
        path=Path(output)/"quantile-forensics.npz";np.savez_compressed(path,**self.arrays)
        return {"path":path.name,"sha256":sha256_file(path),"records":_json_safe(self.records),"record_count":len(self.records),
                "maximum_records":self.maximum,"arrays":{k:array_descriptor(v) for k,v in self.arrays.items()},
                "no_original_failure_reconstruction_claim":True}


def quantile_characterization(predictions,component_predictions,*,stage,label,target_sd,archive,utility):
    """Whole captured batches, central coordinates and endpoint behavior separated."""
    import torch
    from posteriorforge import execution_mixture as mixture
    names={"uncached_FP32_to_uncached_precision":("U0","Up"),
           "uncached_precision_to_cached_precision_and_declared_views":("Up","Cp"),
           "qualified_FP32_GQA_hybrid_to_cached_precision":("G0","Cp"),
           "uncached_FP32_to_qualified_FP32_GQA_hybrid":("U0","G0")}
    result={arm:{"central":{},"endpoints":{},"finite_uniform_bar_mean":{}} for arm in names}
    chosen={};stress_chosen=None
    for distribution in ["final"]+["member_"+str(i) for i in range(8)]:
        bundles={}
        for arm,pred in predictions.items():
            source=pred if distribution=="final" else component_predictions[arm]["component_predictions"][int(distribution.split("_")[-1])]
            logits=source["logits"];criterion=source["criterion"]
            border=criterion.borders.detach().cpu().numpy().copy()
            require(str(logits.dtype)==str(criterion.borders.dtype)=="torch.float32","quantile postprocessing must stay FP32")
            captures={}
            for u in CENTRAL_U+ENDPOINT_U:
                value,capture=_observed_stock(criterion,logits,u)
                replay=replay_batch32(capture,border);actual=value.detach().cpu().numpy()
                require(np.all((actual==replay)|(np.isnan(actual)&np.isnan(replay))),"stock inverse differs from independent original FP32 replay")
                if u<1:
                    fast=mixture.finite_bar_icdf(logits,criterion.borders,torch.full((len(logits),),float(np.float32(u)),dtype=torch.float32))
                    require(torch.equal(value.view(torch.uint8),fast.view(torch.uint8)),"whole-batch stock and frozen finite sampler differ")
                    capture["production_finite_sampler_value"]=fast.detach().cpu().numpy().copy()
                    capture["native_vs_finite_sampler_bitwise_equal"]=True
                else:
                    capture["production_finite_sampler_value"]=None
                    capture["native_vs_finite_sampler_bitwise_equal"]=None
                capture["_actual_probabilities"]=capture.pop("probabilities")
                capture["_actual_cumulative"]=capture.pop("cumulative")
                p=capture["_actual_probabilities"].astype(np.float64);c=capture["_actual_cumulative"].astype(np.float64)
                ideal=np.cumsum(p/p.sum(-1,keepdims=True),axis=-1)
                capture["roundoff_tail_band"]=np.max(np.abs(ideal-c),axis=-1)
                # Finite-uniform-bin analytic mean is distinct from FP32 inverse arithmetic.
                if "mean" not in bundles.get(arm,{}):
                    centers=(border[:-1].astype(np.float64)+border[1:].astype(np.float64))*.5
                    mean=(p*centers).sum(-1)/p.sum(-1)
                captures[float(np.float32(u))]=capture
            bundles[arm]={"capture":captures,"mean":mean,"borders":border,"source":source}
        for arm,(a,b) in names.items():
            left,right=bundles[a],bundles[b]
            require(np.array_equal(left["borders"],right["borders"]),"cross-arm physical quantile borders differ")
            span=float(np.float64(left["borders"][-1])-np.float64(left["borders"][0]))
            require(np.isfinite(span) and span>0 and np.isfinite(target_sd) and target_sd>0,"positive actual physical span/training target SD required")
            means=np.abs(left["mean"]-right["mean"])
            result[arm]["finite_uniform_bar_mean"][distribution]={"physical_absolute_error":_stats(means),"target_training_sd_absolute_error":_stats(means/target_sd),"scope":"analytic finite-uniform-bin law; not actual rounded inverse mean"}
            central_left=np.stack([left["capture"][float(np.float32(u))]["value"] for u in CENTRAL_U]).astype(np.float64)
            central_right=np.stack([right["capture"][float(np.float32(u))]["value"] for u in CENTRAL_U]).astype(np.float64)
            central_location=np.abs(central_left.mean(0)-central_right.mean(0))
            result[arm].setdefault("central_three_quantile_location",{})[distribution]={
                "definition":"arithmetic mean of physical Q(.1),Q(.5),Q(.9); not a posterior mean",
                "physical_absolute_error":_stats(central_location),"span_absolute_error":_stats(central_location/span),
                "training_target_sd_absolute_error":_stats(central_location/target_sd)}
            for category,uniforms in (("central",CENTRAL_U),("endpoints",ENDPOINT_U)):
                entries=[]
                for u in uniforms:
                    u32=float(np.float32(u));l=left["capture"][u32];r=right["capture"][u32]
                    x,y=l["value"].astype(np.float64),r["value"].astype(np.float64)
                    finite=np.isfinite(x)&np.isfinite(y);difference=np.abs(x-y)
                    entry={"u":u32,"production_lattice_point":bool(0<=u32<1 and float(u32*2**24).is_integer()),"u_one_stress_only":u32==1.,
                           "physical_absolute_error":_stats(difference[finite]),"span_absolute_error":_stats(difference[finite]/span),
                           "training_target_sd_absolute_error":_stats(difference[finite]/target_sd),"exact_shared_physical_border_span":span,
                           "reference_nonfinite":int((~np.isfinite(x)).sum()),"candidate_nonfinite":int((~np.isfinite(y)).sum()),
                           "reference_outside_border":int(((x<left["borders"][0])|(x>left["borders"][-1])).sum()),
                           "candidate_outside_border":int(((y<right["borders"][0])|(y>right["borders"][-1])).sum()),
                           "reference_cdf_terminal_min":float(l["cdf_terminal"].min()),"candidate_cdf_terminal_min":float(r["cdf_terminal"].min()),
                           "reference_roundoff_tail_band_max":float(l["roundoff_tail_band"].max()),"candidate_roundoff_tail_band_max":float(r["roundoff_tail_band"].max())}
                    entries.append(entry)
                    score=np.where(finite,difference,np.inf);idx=int(np.argmax(score))
                    candidate=(float(score[idx]),distribution,idx,l,r,a,b)
                    key=(arm,"central" if category=="central" else "production_endpoint" if u32<1 else "u_one_stress")
                    if key not in chosen or candidate[0]>chosen[key][0]:chosen[key]=candidate
                result[arm][category][distribution]=entries
    if label=="1024":
        for (arm,category),(_,distribution,idx,l,r,a,b) in chosen.items():
            if category=="u_one_stress":
                candidate=(chosen[(arm,category)][0],arm,distribution,idx,l,r,a,b)
                if stress_chosen is None or candidate[0]>stress_chosen[0]:stress_chosen=candidate
                continue
            for side,key,capture,other in (("reference",a,l,r["value"][idx]),("candidate",b,r,l["value"][idx])):
                archive.add(component_predictions[key] if distribution!="final" else predictions[key],distribution,capture,idx,
                            stage=stage,arm=arm,side=side,category=category,other_value=other)
        if stress_chosen:
            _,arm,distribution,idx,l,r,a,b=stress_chosen
            for side,key,capture,other in (("reference",a,l,r["value"][idx]),("candidate",b,r,l["value"][idx])):
                archive.add(component_predictions[key] if distribution!="final" else predictions[key],distribution,capture,idx,
                            stage=stage,arm=arm,side=side,category="u_one_stress",other_value=other)
    return {"arms":result,"central_uniforms":list(CENTRAL_U),"endpoint_uniforms":list(ENDPOINT_U),
            "distributions_per_arm":9,"rows_per_distribution":len(predictions["Cp"]["logits"]),
            "whole_batch_softmax_cumsum_preserved":True,"same_logit_stock_finite_and_independent_replay_exact":True,
            "native_vs_finite_comparison_scope":"identical whole-batch logits, physical borders and CPU FP32 uniforms for every u<1; no isolated-row recomputation",
            "u_one_inverse_scope":"native SDK criterion.icdf stress only; forbidden by unchanged production finite_bar_icdf",
            "old_2_28_span_method":"native SDK criterion.icdf against native SDK criterion.icdf, not an e71-specific measurement",
            "denominator":"shared actual physical border[-1]-border[0] evaluated in float64; training target SD separately labeled",
            "old_2_28_span_failure_location_unavailable_without_saved_old_logits":True}


def uniform_lattice_evidence(torch):
    root=Path(torch.__file__).parent/"include/ATen/core"
    names=("DistributionsHelper.h","TransformationHelper.h")
    hashes={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in names}
    require(hashes=={"DistributionsHelper.h":"81ecfe382101b708307b8abcb05a54bc4c9d8bf2cb5234975a3032a90285c7ee",
        "TransformationHelper.h":"77ba7f2ca028c5258f8cf0cfb26354f71a2a93369e0b82c8f7c09ca1ca1c7b45"},"installed Torch CPU uniform headers changed")
    return {"installed_header_sha256":hashes,"cpu_float_uniform":"(uint32_random & ((1<<24)-1))*2^-24",
            "u_zero_in_production_lattice":True,"production_max":1-2**-24,"u_one_in_production_lattice":False,
            "single_lattice_point_probability_under_uniform_random_word":2**-24,"no_endpoint_waiver":True}


def run(*,banks,prefix_run,checkpoint,qualification,ledger,output,mode,max_seconds=1050,trace=True,timing_repetitions=3):
    from posteriorforge.artifacts import runtime_identity,sha256_file,write_json
    from posteriorforge.reference_grid import _require_budgeted_gpu
    from posteriorforge import execution_precision_probe as precision
    from posteriorforge import execution_gqa_hybrid as gqa,execution_cache as cache
    from posteriorforge.banks import load_context,load_heldout,derive_seed
    from posteriorforge.science.analytical_benchmark_v2 import build_problem,array_descriptor
    from posteriorforge.inference import CHECKPOINT_SHA256,frozen_regressor_settings,support_mask
    from posteriorforge.compat_cdf import finite_cdf_compatibility
    from posteriorforge.gate import verify_run_binding
    require(mode in precision.MODES,"one explicitly named approximate policy required")
    require(type(max_seconds) is int and 300<=max_seconds<=1050,"bounded deadline300..1050s required")
    characterize_sources()
    require(type(timing_repetitions) is int and 1<=timing_repetitions<=5,"bounded timing repetitions1..5 required")
    budget=_require_budgeted_gpu(ledger);require(budget["gpus"]==1,"one registered reserved GPU required")
    output=Path(output);output.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    active=None
    report={**runtime_identity(),"schema":SCHEMA,"status":"running","mode":mode,
        "attribution":"Built with PriorLabs-TabPFN","approximate_experiment":True,"qualified_fp32":False,
        "full21_qualified":False,"full21_component_qualified":False,"full21_sampler_qualified":False,"posterior_vectors_generated":0,"reference_integrations":0,
        "scientific_gate_evaluated":False,"requested_stages":list(STAGES),"completed_stages":[],
        "thresholds":THRESHOLDS,"row_counts":list(ROWS),"stress_rows":8,"stage_results":[],
        "budget_binding":budget,"precision_module":precision.metadata(mode),
        "probe_source_sha256":sha256_file(__file__),"scope":"three-stage approximation characterization only; no exact/all21/posterior qualification or end-to-end speed claim",
        "original_strict_producer_sha256":ORIGINAL_PRODUCER_SHA,
        "approximate_full_run_review_policy":APPROXIMATE_FULL_RUN_REVIEW,
        "strict_failures_do_not_abort_characterization":True,
        "timing_repetitions":timing_repetitions,"timing_order":"private FP32 first; approximate later; no interleaving/randomization",
        "cache_comparison_scope":"cache plus declared query-view layout; do not attribute the composite difference solely to KV reuse"}
    archive=ForensicArchive()
    write_json(output/"manifest.json",report);engine=None
    try:
        import torch
        require(torch.cuda.is_available() and torch.cuda.device_count()==1,"one visible CUDA GPU required")
        require(torch.get_float32_matmul_precision()=="highest" and not torch.backends.cuda.matmul.allow_tf32 and not torch.is_autocast_enabled("cuda"),"start from strict original FP32 flags")
        seed=derive_seed(20261111,"precision-conditional-probe",case=0)
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS","16")))
        report["production_uniform_lattice"]=uniform_lattice_evidence(torch)
        report.update(seed=seed,rng_role="precision-conditional-probe",
                      posterior_sampling_rng_used=False)
        require(sha256_file(checkpoint)==CHECKPOINT_SHA256,"immutable checkpoint required")
        precision.require_sources()
        utility=load_helper("qualify_execution_cache.py",UTILITY_SHA)
        report["model_contract_metadata_observer"]=private_model_contract(utility)[1]
        require(utility.THRESHOLDS==THRESHOLDS,"unchanged strict FP32 thresholds required")
        context=load_context(banks,20261111,4096);case=load_heldout(banks,0);query=case.obs.reshape(1,33)
        qualified=bound_sources(qualification,context,query,checkpoint)
        prefix_run=Path(prefix_run);pm=json.loads((prefix_run/"manifest.json").read_text())
        require(sha256_file(prefix_run/"manifest.json")==qualified["prefix_manifest_sha256"]
                and sha256_file(prefix_run/"samples.npz")==qualified["prefix_samples_sha256"],"fixed-prefix source changed")
        verify_run_binding(pm,context,case.obs,20261111,0)
        with np.load(prefix_run/"samples.npz",allow_pickle=False) as data:values=data["samples"].copy()
        require(values.shape==(50000,21) and support_mask(values).all(),"original complete supported prefix source required")
        positions=np.asarray(qualified["fixed_prefix_row_indices"],dtype=int)
        archive.logical_rows=positions
        prefixes=values[positions];theta=context.theta.astype(np.float32);obs=context.obs.astype(np.float32)
        problem=build_problem();noise=[problem.parameter_order.index(n) for n in problem.prior.noise_names]
        report.update(checkpoint_sha256=CHECKPOINT_SHA256,qualification_sha256=sha256_file(qualification),
            context_theta=array_descriptor(context.theta),context_obs=array_descriptor(context.obs),query=array_descriptor(query),
            parameter_order=list(problem.parameter_order),prefix_samples_sha256=pm["samples_sha256"],fixed_prefix_rows=positions.tolist())
        original_sources={"precision":precision.source_sha256(),"probe":sha256_file(__file__),"qualification":sha256_file(qualification)}
        active=None
        with finite_cdf_compatibility() as active:
            for stage in STAGES:
                require(time.perf_counter()+180<start+max_seconds,"bounded stage deadline; no partial result promoted")
                row={"stage_index":stage,"parameter":problem.parameter_order[stage],"strict_screen_passed":False,"hard_validity_passed":False,"comparisons":{}}
                report["stage_results"].append(row);report["active_stage"]=stage;write_json(output/"manifest.json",report)
                queries,stress=utility.fixed_queries(query,prefixes,theta,stage,noise,problem.prior.bounds)
                sets={"128":queries[:128],"1024":queries,"stress":stress}
                # Each arm is prepared independently. All retained outputs are CPU captures.
                initial_rng=utility.rng_identity(utility.rng_state(torch));t0=time.perf_counter()
                engine=gqa.GQAHybridStageExecutor(checkpoint,cdf_descriptor=active,query_batch_size=1024)
                prepared=engine.prepare(theta,obs,stage)
                g0={name:precision.capture_members(engine,x) for name,x in sets.items()}
                row["qualified_GQA_model"]=model_summary(engine.regressor,utility)
                row["qualified_GQA_prepare"]=prepared;row["qualified_GQA_total_seconds"]=time.perf_counter()-t0
                engine.close();engine=None;gc.collect()
                # Private FP32 arm is a hard prerequisite before lower precision.
                with precision.PrecisionSession(precision.CONTROL_MODE) as transparent:
                    engine=precision.PrecisionStageExecutor(checkpoint,cdf_descriptor=active)
                    torch.cuda.synchronize();prepare_start=time.perf_counter()
                    transparent_prepared=engine.prepare(theta,obs,stage);torch.cuda.synchronize()
                    row["private_FP32_prepare_seconds"]=time.perf_counter()-prepare_start
                    row["private_FP32_prepare"]=transparent_prepared
                    replay={name:precision.capture_members(engine,x) for name,x in sets.items()}
                    row["private_FP32_transparency"]={name:{"byte_exact":exact_prediction(g0[name],replay[name],torch),
                        "comparison":utility.compare_predictions(g0[name],replay[name])} for name in sets}
                    row["private_FP32_capture_free_all8_timings"]=timed_all8(engine,sets,torch,repetitions=timing_repetitions)
                    engine.close();engine=None
                row["private_FP32_session"]=transparent.report()
                require(all(r["byte_exact"] and r["comparison"]["passed"] for r in row["private_FP32_transparency"].values()),
                        "private FP32 adapter did not reproduce qualified FP32 output exactly")
                with precision.PrecisionSession(precision.CONTROL_MODE,query_views=False) as stock:
                    uncached=precision.UncachedPrecisionStage(checkpoint,cdf_descriptor=active,stage_index=stage,theta=theta,obs=obs)
                    u0_components={name:precision.capture_members(uncached,x) for name,x in sets.items()}
                    row["uncached_FP32_model"]=model_summary(uncached.regressor,utility)
                    original_regressor=uncached.regressor;original_base=uncached.base
                row["uncached_FP32_session"]=stock.report()
                u0={}
                for name,x in sets.items():
                    with original_base._model_rng():
                        u0[name]=cache.capture_corrected_prediction(original_regressor,x,cdf_descriptor=active)
                row["FP32_component_reconstruction_vs_actual_original_full_API"]={
                    name:reconstruction_checks(u0_components[name],u0[name],utility) for name in sets}
                require(all(value["passed"] for checks in row["FP32_component_reconstruction_vs_actual_original_full_API"].values()
                            for key,value in checks.items() if key!="scope"),
                        "FP32 component reconstruction differs from actual original full API")
                uncached.close();del original_regressor,original_base

                t0=time.perf_counter()
                with precision.PrecisionSession(mode,query_views=False) as low_stock:
                    uncached=precision.UncachedPrecisionStage(checkpoint,cdf_descriptor=active,stage_index=stage,theta=theta,obs=obs)
                    up={name:precision.capture_members(uncached,x) for name,x in sets.items()}
                    row["uncached_precision_model"]=model_summary(uncached.regressor,utility);uncached.close()
                row["uncached_precision_total_seconds"]=time.perf_counter()-t0;row["uncached_precision_session"]=low_stock.report()
                torch.cuda.reset_peak_memory_stats()
                with precision.PrecisionSession(mode) as candidate:
                    row["candidate_allocator_baseline_bytes"]=torch.cuda.memory_allocated()
                    row["candidate_allocator_reserved_baseline_bytes"]=torch.cuda.memory_reserved()
                    row["reserved_memory_scope"]="allocator pool retained from preceding control arms; not isolated candidate capacity"
                    engine=precision.PrecisionStageExecutor(checkpoint,cdf_descriptor=active)
                    torch.cuda.synchronize();t0=time.perf_counter();prepared=engine.prepare(theta,obs,stage);torch.cuda.synchronize()
                    row["cached_precision_prepare_seconds"]=time.perf_counter()-t0;row["cached_precision_prepare"]=prepared
                    row["computed_prefill_witness"]=precision.prefill_witness_passed(candidate.report(),stage)
                    require(row["computed_prefill_witness"]["passed"],"declared precision prefill not actually witnessed")
                    row["cache_after_computed_prefill"]=engine.inspect_cache()
                    row["cached_precision_model"]=model_summary(engine.regressor,utility)
                    fields=("target_mean","target_sd","canonical_borders_dtype","physical_borders_dtype","temperature")
                    row["target_statistics_and_border_types_preserved"]=all(
                        all(model[field]==row["uncached_FP32_model"][field] for field in fields)
                        for model in (row["qualified_GQA_model"],row["uncached_precision_model"],row["cached_precision_model"]))
                    require(row["target_statistics_and_border_types_preserved"],"target statistics/border types changed")
                    cp={name:precision.capture_members(engine,x) for name,x in sets.items()}
                    row["computed_query_witness"]=precision.query_witness_passed(candidate.report(),stage)
                    require(row["computed_query_witness"]["passed"],"actual query precision activation not witnessed")
                    for name in sets:row["comparisons"][name]=comparison_arms(u0[name],up[name],cp[name],g0[name],utility)
                    row["actual_precision_component_reconstruction"]={
                        name:reconstruction_checks(cp[name],g0[name],utility) for name in sets}
                    # Timing excludes raw member snapshots; dtype observer remains
                    # active and its small instrumentation scope is disclosed.
                    row["approximate_capture_free_all8_timings"]=timed_all8(engine,sets,torch,repetitions=timing_repetitions)
                    row["matched_query_timing_ratio_FP32_over_approximate"]={name:
                        row["private_FP32_capture_free_all8_timings"][name]["median_seconds"]/
                        row["approximate_capture_free_all8_timings"][name]["median_seconds"] for name in ("128","1024")}
                    row["matched_prepare_ratio_FP32_over_approximate"]=row["private_FP32_prepare_seconds"]/row["cached_precision_prepare_seconds"]
                    row["timing_instrumentation"]="no raw snapshots; bounded projection-dtype observer remains active"
                    row["TF32_instruction_precision_claim"]="flags and actual kernels are recorded; no instruction-level precision is inferred from flags alone"
                    if trace:row["bounded_trace"]=utility.profile_prediction(
                        lambda:engine.predict_component(0,queries[:128],capture_member=False),output,
                        "precision-stage-"+str(stage),torch)
                    row["peak_allocated_bytes"]=torch.cuda.max_memory_allocated()
                    row["peak_reserved_bytes"]=torch.cuda.max_memory_reserved()
                    engine.close();engine=None
                row["cached_precision_session"]=candidate.report()
                row["rng_preserved"]=initial_rng==utility.rng_identity(utility.rng_state(torch))
                row["hard_probability_validity"]={name:{arm:hard_prediction_validity(prediction) for arm,prediction in
                    (("U0",u0[name]),("Up",up[name]),("Cp",cp[name]),("G0",g0[name]))} for name in sets}
                row["quantile_characterization"]={name:quantile_characterization(
                    {"U0":u0[name],"Up":up[name],"Cp":cp[name],"G0":g0[name]},
                    {"U0":u0_components[name],"Up":up[name],"Cp":cp[name],"G0":g0[name]},
                    stage=stage,label=name,target_sd=float(row["uncached_FP32_model"]["target_sd"]),
                    archive=archive,utility=utility) for name in sets}
                row.update(strict_and_hard_outcome(row))
                report["completed_stages"].append(stage)
                report["quantile_forensics"]=archive.save(output)
                write_json(output/"manifest.json",report)
                require(row["hard_validity_passed"],"hard validity/source/arithmetic/RNG condition failed; approximate characterization stopped")
                del g0,replay,u0,u0_components,up,cp;gc.collect()
                require(time.perf_counter()<start+max_seconds,"bounded characterization deadline; never promote partial coverage")
            report["status"]="three_stage_approximation_characterized_not_qualified"
            report["strict_failed_stages"]=[row["stage_index"] for row in report["stage_results"] if not row["strict_screen_passed"]]
        report["compatibility_adapter"]=dict(active)
        require(active["original_alias_restored"] is True and active["sdk_sources_unchanged"] is True,"CDF context restoration failed")
        require(original_sources=={"precision":precision.source_sha256(),"probe":sha256_file(__file__),
                                   "qualification":sha256_file(qualification)},"source/proof changed during precision screen")
        characterize_sources()
        report["sources_unchanged"]=True
    except BaseException as error:
        report.update(status="hard_failure_or_incomplete_characterization",error_type=type(error).__name__,error=str(error))
        if active is not None:report["compatibility_adapter"]=dict(active)
        raise
    finally:
        if engine is not None:engine.close()
        report["peak_cpu_rss_bytes"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=="darwin" else 1024)
        report["quantile_forensics"]=archive.save(output)
        report["elapsed_seconds"]=time.perf_counter()-start;write_json(output/"manifest.json",report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("banks","prefix-run","checkpoint","qualification","ledger","output","mode"):parser.add_argument("--"+name,required=True)
    parser.add_argument("--max-seconds",type=int,default=1050);parser.add_argument("--timing-repetitions",type=int,default=3);parser.add_argument("--no-trace",action="store_true")
    args=vars(parser.parse_args());args["trace"]=not args.pop("no_trace")
    result=run(**args);print(json.dumps({k:result[k] for k in ("status","mode","completed_stages","elapsed_seconds")},indent=2))


if __name__=="__main__":main()
