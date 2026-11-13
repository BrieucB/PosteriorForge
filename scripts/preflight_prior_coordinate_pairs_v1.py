"""CPU-only all23 fixed FIRST1024 pair inventories. Built with PriorLabs-TabPFN.

No NN/model, simulator, reference integration, posterior draw, GPU or ledger work.
All fixed rows survive exactly or their pair fails; no easier replacement is used.
"""
from __future__ import annotations
import argparse,hashlib,json,pickle,platform,random,sys,time
from pathlib import Path
import numpy as np
import scipy
from posteriorforge import prior_coordinate_pair_v1 as pairs
from posteriorforge.science.analytical_benchmark_v2 import array_descriptor

SCHEMA_ENV="posteriorforge.prior_coordinate_pair_cpu_environment.v1"
REGRESSION_PROOF_SHA256="5ec69eb7aadbd74c7d96ee534b71d0d8bee2b6dd6f6e3d2f44e74607452f682b"
F32_KEYS=("context_ordered_latent_model_Float32","context_SS15_model_Float32","query_SS15_model_Float32","fixed_prefix_ordered_latent_model_Float32")


def source_sha256():return pairs.sha(__file__)
def rng():return hashlib.sha256(pickle.dumps((np.random.get_state(),random.getstate()),protocol=4)).hexdigest()
def write(path,d):Path(path).write_text(json.dumps(d,indent=2,sort_keys=True,allow_nan=False)+"\n")



def authenticate_reference(spec,*,banks,parent_manifest,reference=None,reference_grid=None):
    """Exact frozen actual-case reference/grid custody only; never integrate."""
    from posteriorforge.inference_cdf import reference_binding
    results=Path(banks).resolve().parent
    reference=results/'references-v1'/('case-'+str(spec.case)) if reference is None else Path(reference)
    reference_grid=results/'reference-grids-v1'/('case-'+str(spec.case)) if reference_grid is None else Path(reference_grid)
    actual=reference_binding(banks=banks,reference=reference,reference_grid=reference_grid,case=spec.case)
    parent=json.loads(Path(parent_manifest).read_bytes())
    pairs.require(actual==parent.get('reference_binding'),'actual-case reference/grid differs from original pure parent')
    return actual


def preflight_one(spec,*,banks,completion_receipt,training_inventory,regression_proof=None):
    started=time.perf_counter();record={"pair":spec.as_dict(),"status":"running","phases_seconds":{},"selection_replaced_or_filtered":False}
    inventory=pairs.load_script("qualify_prior_coordinates_v1.py",pairs.INVENTORY_HELPER_SHA256)
    data=pairs.load_script("qualify_prior_conditionals_v1.py",pairs.INPUT_HELPER_SHA256)
    from posteriorforge import execution_prior_native_v1 as native
    try:
        t=time.perf_counter();record["prefix_owner"]=pairs.prefix_parent(spec,completion_receipt=completion_receipt)
        prefix,custody=pairs.load_prefix(spec,completion_receipt=completion_receipt)
        record.update(prefix_custody=custody,actual_prefix_physical_Float64=array_descriptor(prefix),domain=inventory.domain_counts(prefix))
        record["phases_seconds"]["actual_parent_prefix_authentication"]=time.perf_counter()-t
        t=time.perf_counter();record["reference_binding"]=authenticate_reference(spec,banks=banks,parent_manifest=custody["manifest_path"])
        record["phases_seconds"]["actual_reference_and_grid_authentication"]=time.perf_counter()-t
        t=time.perf_counter();audit={}
        theta,stats,query,prefixes,raw,transformed,custody=pairs.input_bindings(spec,banks=banks,completion_receipt=completion_receipt,training_inventory=training_inventory,runtime_audit=audit)
        record.update(raw_bindings=raw,transformed_bindings=transformed,runtime_input_audit=audit,prefix_custody=custody)
        record["phases_seconds"]["actual_input_codec_and_SS15_bindings"]=time.perf_counter()-t
        t=time.perf_counter();qualification=inventory.qualify_matrix(prefix)
        pairs.require(qualification["causal_all21_prefixes_identical"] is True,"all21 strict causal prefixes required")
        report=qualification["report"]
        pairs.require(all(report.get(k) is True for k in ["Float64_roundtrip_passed","model_Float32_cast_roundtrip_passed","no_rows_removed","no_clipping_or_repair"]),"strict unmodified codec roundtrip flags required")
        pairs.require(qualification["domain"]["nonfinite_count"]==0 and all(v==0 for v in qualification["domain"]["endpoint_or_outside_counts"].values()),"fixed prefix endpoint/nonfinite failure")
        pairs.require(all(np.isfinite(v) and 0<=v<=1 for values in report["error_to_declared_bound_max_per_coordinate"].values() for v in values),"declared codec roundtrip bound failed")
        record["prefix_codec_qualification"]=qualification;record["phases_seconds"]["strict_prefix_codec_causal_roundtrips"]=time.perf_counter()-t
        t=time.perf_counter();bindings=[]
        for stage in range(21):
            main,stress,noise=data.fixed_queries(query,prefixes,theta,stage,native)
            mt,ms,mq,sq,binding=data.masked_inputs(theta,stats,main,stress,stage,native)
            pairs.require(binding["training"]["original_latent_target"]==binding["training"]["prepared_latent_target"]==array_descriptor(theta[:,stage]),"actual masked target bytes changed")
            bindings.append(binding)
        record["all21_mask_target_query_bindings"]=bindings
        record["phases_seconds"]["all21_actual_masks_targets_and_stress_validity"]=time.perf_counter()-t
        if spec==pairs.REGRESSION_PAIR:
            pairs.require(regression_proof is not None and pairs.sha(regression_proof)==REGRESSION_PROOF_SHA256,"authenticated case2 regression proof required")
            old=json.loads(Path(regression_proof).read_bytes())
            pairs.require(raw==old["raw_bindings"] and all(transformed[k]==old["transformed_bindings"][k] for k in F32_KEYS),"case2 original raw/modelF32 data regression differs")
            pairs.require(bindings==[r["mask_inputs"] for r in old["stage_results"]],"case2 all21 actual mask/target/query bytes differ")
            from posteriorforge.banks import derive_seed
            seed=derive_seed(spec.context_seed,"prior-coordinate-fixed-conditionals-v1",case=spec.case)
            pairs.require(seed==old["seed"],"case2 conditional seed derivation changed")
            record["case2_regression"]={"passed":True,"proof_sha256":REGRESSION_PROOF_SHA256,"original_seed":seed,"raw_model_Float32_all21_mask_target_query_bytes_identical":True,"new_model_calls":0}
        record["status"]="pair_codec_preflight_pass"
    except Exception as error:
        record.update(status="pair_codec_preflight_failure",error_type=type(error).__name__,error=str(error),no_row_replacement_or_filtering=True)
    record["elapsed_seconds"]=time.perf_counter()-started
    return record


def run_environment(*,banks,completion_receipt,training_inventory,reference_training_inventory,combined_training_inventory,regression_proof,output,environment):
    pairs.require(environment in ("inference","reference"),"explicit locked environment required")
    versions={"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__}
    expected={"inference":{"python":"3.11.13","numpy":"1.26.4","scipy":"1.12.0"},"reference":{"python":"3.11.13","numpy":"2.2.6","scipy":"1.15.3"}}[environment]
    pairs.require(versions==expected,"locked CPU environment versions differ")
    out=Path(output);pairs.require(not out.exists(),"fresh CPU panel environment output required");out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();before=rng();pins=pairs.source_pins();own=source_sha256()
    data=pairs.load_script("qualify_prior_conditionals_v1.py",pairs.INPUT_HELPER_SHA256)
    t=time.perf_counter();_,training_replay=data.inventory_receipt(training_inventory,reference_training_inventory,combined_training_inventory)
    record={"schema":SCHEMA_ENV,"environment_name":environment,"environment":versions,"source_pins":pins,"producer_source_sha256":own,"status":"running","training_inventory_reused_with_exact_current_model_Float32_check":training_replay,"common_phase_seconds":{"three_bank_dual_CPU_inventory_validation":time.perf_counter()-t},"pairs":[],"case2_regression":None,"no_new_models_worlds_posteriors_or_reference_integrations":True,"Torch_imported":False,"RNG_before":before}
    write(out/"manifest.json",record)
    for spec in pairs.REMAINING_PAIRS:
        row=preflight_one(spec,banks=banks,completion_receipt=completion_receipt,training_inventory=training_inventory)
        record["pairs"].append(row);write(out/"manifest.json",record)
        print(json.dumps({"pair":row["pair"],"status":row["status"],"error":row.get("error"),"elapsed_seconds":row["elapsed_seconds"]},sort_keys=True),flush=True)
    record["case2_regression"]=preflight_one(pairs.REGRESSION_PAIR,banks=banks,completion_receipt=completion_receipt,training_inventory=training_inventory,regression_proof=regression_proof)
    record.update(all23_processed=len(record["pairs"])==23,failed_pairs=[r["pair"] for r in record["pairs"] if r["status"]!="pair_codec_preflight_pass"],RNG_after=rng(),sources_unchanged=pairs.source_pins()==pins and source_sha256()==own,Torch_imported="torch" in sys.modules,elapsed_seconds=time.perf_counter()-started)
    record["default_RNG_preserved"]=record["RNG_before"]==record["RNG_after"]
    pairs.require(record["sources_unchanged"] and record["default_RNG_preserved"] and not record["Torch_imported"],"CPU panel common source/RNG/Torch integrity failed")
    record["status"]="all23_codec_preflight_pass_ONLY" if not record["failed_pairs"] and record["case2_regression"]["status"]=="pair_codec_preflight_pass" else "fixed_panel_codec_failures_retained"
    write(out/"manifest.json",record)
    return record


def combine(*,inference,inference_sha256,reference,reference_sha256,output):
    a=pairs._document(inference,inference_sha256);b=pairs._document(reference,reference_sha256)
    pairs.require(a["schema"]==b["schema"]==SCHEMA_ENV and a["environment_name"]=="inference" and b["environment_name"]=="reference","both named locked CPU inventories required")
    pairs.require(a["source_pins"]==b["source_pins"]==pairs.source_pins() and a["producer_source_sha256"]==b["producer_source_sha256"]==source_sha256(),"both CPU inventories must bind same frozen source")
    pairs.require(all(x["all23_processed"] and x["sources_unchanged"] and x["default_RNG_preserved"] and not x["Torch_imported"] for x in [a,b]),"all23 common CPU integrity required")
    rows=[]
    for ra,rb,spec in zip(a["pairs"],b["pairs"],pairs.REMAINING_PAIRS):
        pairs.require(ra["pair"]==rb["pair"]==spec.as_dict(),"fixed23 actual pair order changed")
        passed=ra["status"]==rb["status"]=="pair_codec_preflight_pass"
        common=passed and ra["raw_bindings"]==rb["raw_bindings"] and ra["reference_binding"]==rb["reference_binding"] and all(ra["transformed_bindings"][k]==rb["transformed_bindings"][k] for k in F32_KEYS) and ra["all21_mask_target_query_bindings"]==rb["all21_mask_target_query_bindings"]
        rows.append({"pair":spec.as_dict(),"status":"pair_codec_preflight_pass" if common else "pair_codec_preflight_failure","COMMON_Float32_model_bytes_match":bool(common),"Float64_cross_environment_byte_identity_claimed":False,"Float64_prefix_latent_bytes_differ":ra.get("runtime_input_audit",{}).get("fixed_prefix_latent_physical_slots_Float64")!=rb.get("runtime_input_audit",{}).get("fixed_prefix_latent_physical_slots_Float64"),"inference":ra,"reference":rb})
    regression=all(x["case2_regression"]["status"]=="pair_codec_preflight_pass" and x["case2_regression"]["case2_regression"]["passed"] for x in [a,b])
    out=Path(output);pairs.require(not out.exists(),"fresh combined CPU receipt required");out.mkdir(parents=True,exist_ok=False)
    record={"schema":pairs.PAIR_CPU_SCHEMA,"status":"all23_dual_CPU_preflight_pass_ONLY" if all(r["COMMON_Float32_model_bytes_match"] for r in rows) and regression else "fixed_panel_CPU_failures_retained","source_pins":pairs.source_pins(),"producer_source_sha256":source_sha256(),"environments":{"inference":{"path":str(Path(inference).resolve()),"sha256":inference_sha256,"versions":a["environment"]},"reference":{"path":str(Path(reference).resolve()),"sha256":reference_sha256,"versions":b["environment"]}},"all23_processed":True,"COMMON_Float32_model_bytes_match_for_passed_pairs":True,"case2_regression_passed_both_environments":regression,"case2_regression":{"inference":a["case2_regression"],"reference":b["case2_regression"]},"pairs":rows,"failed_pairs":[r["pair"] for r in rows if r["status"]!="pair_codec_preflight_pass"],"posterior_or_backend_qualification_claimed":False,"no_row_replacement_filtering_clipping_or_case_switch":True}
    write(out/"receipt.json",record);return record


def main():
    p=argparse.ArgumentParser(description=__doc__);subs=p.add_subparsers(dest="command",required=True)
    r=subs.add_parser("run")
    for key in ["banks","completion-receipt","training-inventory","reference-training-inventory","combined-training-inventory","regression-proof","output","environment"]:r.add_argument("--"+key,required=True)
    c=subs.add_parser("combine")
    for key in ["inference","inference-sha256","reference","reference-sha256","output"]:c.add_argument("--"+key,required=True)
    args=vars(p.parse_args());command=args.pop("command");result=run_environment(**args) if command=="run" else combine(**args)
    print(json.dumps({"status":result["status"],"failed_pairs":result.get("failed_pairs",[])},sort_keys=True))


if __name__=="__main__":main()
