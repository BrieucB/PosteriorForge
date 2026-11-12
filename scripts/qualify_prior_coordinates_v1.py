#!/usr/bin/env python3
"""CPU-only original-bank/first1024 codec qualification, NOT a backend or posterior pass."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import platform
import re
import random
import subprocess
import sys
import time
import numpy as np
import scipy
from posteriorforge import prior_coordinates_v1 as codec
from posteriorforge.banks import CONTEXT_SEEDS,load_context
from posteriorforge.science.analytical_benchmark_v2 import array_descriptor,build_problem

SCHEMA="posteriorforge.prior_coordinates_inventory.v1"
PREFIX_RULE="original_accepted_case2_first1024_in_saved_order_v1"
PREFIX_INDICES=tuple(range(1024))
PREFIX_MANIFEST_SHA256="28de1d121e5e89a5b55f630625c0f60e86551473c8bbaa3b2840797683f12e6a"
PREFIX_PAYLOAD_SHA256="9ce954f75b73c54348733c3068179d542ed050ec1aaae772734b3def67fc752b"
BANK_LOADER_SHA256="9bbbddfec2816a91361bf1bb0ba79b9ed1266dcc46546d78bb616d21e92f9032"

def require(value,message):
    if not value:raise ValueError(message)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def source_sha256():
    return sha(__file__)

def source_pins():
    root=Path(codec.__file__).parent
    require(sha(root/"banks.py")==BANK_LOADER_SHA256,"frozen bank loader changed")
    return dict(codec_module_sha256=codec.source_sha256(),qualifier_source_sha256=source_sha256(),
                bank_loader_sha256=BANK_LOADER_SHA256,prior_and_order_sources=codec.require_sources())

def rng_receipt():
    return dict(numpy_legacy_sha256=hashlib.sha256(pickle.dumps(np.random.get_state(),protocol=4)).hexdigest(),
                python_random_sha256=hashlib.sha256(pickle.dumps(random.getstate(),protocol=4)).hexdigest())

def write_json(path,document):
    path.write_text(json.dumps(document,sort_keys=True,indent=2,allow_nan=False)+"\n")

def domain_counts(values):
    return dict(nonfinite_count=int(np.sum(~np.isfinite(values))),
        endpoint_or_outside_counts={str(k):int(np.sum((values[:,k]<=.01)|(values[:,k]>=.5))) for k in codec.NOISE_INDICES})

def qualify_matrix(values):
    """No row selection or repair: encode/roundtrip the entire supplied matrix."""
    domains=domain_counts(values)
    latent64=codec.encode_physical(values)
    model32=codec.encode_for_model(values)
    ordered32=codec.order_latents(model32)
    for width in range(1,22):
        decoded=codec.decode_ordered_prefix(ordered32[:,:width])
        full=codec.decode_ordered(ordered32)[:,codec.ORDER[:width]]
        require(np.array_equal(decoded,full),"causal prefix differs from full decoder")
    return dict(physical_Float64=array_descriptor(values),latent_Float64=array_descriptor(latent64),
                latent_model_Float32=array_descriptor(model32),ordered_model_Float32=array_descriptor(ordered32),
                domain=domains,report=codec.roundtrip_report(values),causal_all21_prefixes_identical=True)

def load_prefix(prefix_run,context,query):
    root=Path(prefix_run)
    manifest=root/"manifest.json";payload=root/"samples.npz"
    require(sha(manifest)==PREFIX_MANIFEST_SHA256,"exact original case2 prefix manifest required")
    require(sha(payload)==PREFIX_PAYLOAD_SHA256,"exact original case2 prefix payload required")
    header=json.loads(manifest.read_bytes())
    problem=build_problem()
    require(header["status"]=="complete" and header["requested_count"]==50000
            and header["context_seed"]==20261111 and header["case"]==2 and header["repetition"]==0,
            "original full50000 case2 parent identity differs")
    require(header["scientific_fingerprint"]==problem.scientific_fingerprint
            and header["parameter_order"]==list(problem.parameter_order),"prefix science/order differs")
    require(header["context_theta"]==array_descriptor(context.theta)
            and header["context_obs"]==array_descriptor(context.obs)
            and header["query"]==array_descriptor(query),"prefix original raw input custody differs")
    with np.load(payload,allow_pickle=False) as saved:
        require(saved.files==["samples"],"original samples-only payload required")
        values=saved["samples"].copy()
    require(values.dtype==np.float64 and values.shape==(50000,21) and problem.contains(values).all(),
            "exact complete supported original Float64 rows required")
    selected=np.ascontiguousarray(values[:1024])
    return selected,dict(manifest_path=str(manifest.resolve()),manifest_sha256=PREFIX_MANIFEST_SHA256,
        payload_path=str(payload.resolve()),payload_sha256=PREFIX_PAYLOAD_SHA256,
        rule=PREFIX_RULE,selected_indices=list(PREFIX_INDICES),
        entire_payload_shape=list(values.shape),selection_replaced_or_filtered=False)

def validate_inventory(document,expected_sources=None):
    require(document.get("schema")==SCHEMA and document.get("status")=="codec_inventory_pass","codec inventory did not pass")
    require(document.get("qualified") is True and document.get("all_original_training_rows_inventoried") is True,
            "all original training rows must be inventoried")
    require(document.get("source_pins")== (source_pins() if expected_sources is None else expected_sources),"codec source identity differs")
    require(document.get("codec_metadata")==codec.metadata(),"codec/tolerance/prior metadata differs")
    require(document.get("banks_unchanged") is True and document.get("sources_unchanged") is True
            and document.get("default_RNG_preserved") is True and document.get("torch_imported") is False,
            "source/bank/RNG/Torch invariants missing")
    require(document.get("RNG_before")==document.get("RNG_after"),"observed default RNG state changed")
    require(document.get("matching_backend_conditionals_qualified") is False
            and document.get("posterior_accuracy_assessed") is False,"codec evidence cannot be promoted to backend/accuracy")
    banks=document.get("banks",[])
    require([r["context_seed"] for r in banks]==list(CONTEXT_SEEDS),"exact three original context seeds required")
    require(sum(r["count"] for r in banks)==3*4096,"all12288 original training rows required")
    groups=banks+[document["case2_fixed_prefix"]]
    for row in groups:
        require(row.get("passed") is True and row["domain"]["nonfinite_count"]==0
                and all(v==0 for v in row["domain"]["endpoint_or_outside_counts"].values()),"domain failure retained")
        q=row["qualification"]
        require(q["report"]["Float64_roundtrip_passed"] is True
                and q["report"]["model_Float32_cast_roundtrip_passed"] is True
                and q["causal_all21_prefixes_identical"] is True,"roundtrip/causality failure")
        require(q["report"]["no_rows_removed"] is True and q["report"]["no_clipping_or_repair"] is True,
                "training row deletion or repair forbidden")
        require(q["domain"]==row["domain"] and q["report"]["count"]==row["count"],
                "domain/report row count differs")
        require(set(row["domain"]["endpoint_or_outside_counts"])==set(map(str,codec.NOISE_INDICES)),
                "all six source noise dimensions required")
        for name,dtype in (("physical_Float64","<f8"),("latent_Float64","<f8"),
                           ("latent_model_Float32","<f4"),("ordered_model_Float32","<f4")):
            desc=q[name]
            require(desc["shape"]==[row["count"],21] and desc["dtype"]==dtype
                    and re.fullmatch("[0-9a-f]{64}",desc["sha256"]) is not None,
                    "coordinate descriptor shape/dtype/hash differs")
        if "original_theta" in row:
            require(q["physical_Float64"]==row["original_theta"]
                    and row["original_observations"]["shape"]==[4096,33]
                    and row["original_observations"]["dtype"]=="<f8"
                    and row["observation_array_unchanged"] is True,"original paired arrays differ")
        bounds=q["report"]["error_to_declared_bound_max_per_coordinate"]
        require(set(bounds)=={"physical64","model32_physical","latent64","model32_latent","final_model_cast"},
                "all separate roundtrip/cast bounds required")
        for values in bounds.values():
            numbers=np.asarray(values)
            require(numbers.shape==(21,) and np.isfinite(numbers).all()
                    and np.all(numbers>=0) and np.all(numbers<=1),"reported roundtrip exceeds frozen bound")
    require(all(r["count"]==4096 for r in banks),"4096 per unchanged original context required")
    prefix=document["case2_fixed_prefix"]
    require(prefix["count"]==1024 and prefix["custody"]["rule"]==PREFIX_RULE
            and prefix["custody"]["selected_indices"]==list(PREFIX_INDICES)
            and prefix["custody"]["manifest_sha256"]==PREFIX_MANIFEST_SHA256
            and prefix["custody"]["payload_sha256"]==PREFIX_PAYLOAD_SHA256
            and prefix["custody"]["selection_replaced_or_filtered"] is False,"exact first1024 prefix rule differs")
    return dict(passed=True,scope="codec/domain preflight only; no backend numeric or posterior pass",
                original_training_rows=12288,original_case2_prefix_rows=1024)

def run(*,banks,prefix_run,output):
    started=time.perf_counter()
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    banks=Path(banks).resolve()
    paths=[banks/"manifest.json",banks/"heldout.json",banks/"heldout.npz"]
    paths += [banks/(f"context-{s}"+suffix) for s in CONTEXT_SEEDS for suffix in (".json",".npz")]
    entry_hashes={str(p):sha(p) for p in paths}
    pins=source_pins()
    observed_rng=rng_receipt()
    document=dict(schema=SCHEMA,status="running",qualified=False,source_pins=pins,
        codec_metadata=codec.metadata(),prefix_rule_frozen_before_inventory=PREFIX_RULE,
        banks=[],case2_fixed_prefix=None,all_original_training_rows_inventoried=False,
        matching_backend_conditionals_qualified=False,posterior_accuracy_assessed=False,
        no_reference_or_truth_values_used=True,compute=dict(gpu_hours=0,model_calls=0,new_worlds=0,new_posterior_draws=0),
        environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__),
        git_revision=subprocess.run(["git","rev-parse","HEAD"],cwd=Path(codec.__file__).parents[2],check=True,capture_output=True,text=True).stdout.strip(),
        source_identity_scope="exact new working-source SHA256s; Git revision recorded separately",
        original_bank_files_sha256=entry_hashes,RNG_before=observed_rng,errors=[])
    write_json(output/"manifest.json",document)
    try:
        bank_manifest=json.loads((banks/"manifest.json").read_bytes())
        require(bank_manifest["schema"]=="posteriorforge.banks.v1"
                and [r["seed"] for r in bank_manifest["contexts"]]==list(CONTEXT_SEEDS)
                and bank_manifest["shared_cases"]==8,"original fixed banks manifest differs")
        heldout_meta=json.loads((banks/"heldout.json").read_bytes())
        require(sha(banks/"heldout.npz")==heldout_meta["sha256"]==bank_manifest["heldout"]["sha256"],"heldout custody differs")
        with np.load(banks/"heldout.npz",allow_pickle=False) as saved:
            # Only saved observations are read; heldout truth parameters are not used.
            observations=saved["obs"].copy()
        require(observations.dtype==np.float64 and observations.shape==(8,33)
                and array_descriptor(observations)==heldout_meta["obs"],"heldout observation bytes differ")
        document["heldout_observations"]=array_descriptor(observations)
        document["heldout_truth_parameters_read_or_transformed"]=False
        context11=None
        for seed in CONTEXT_SEEDS:
            context=load_context(banks,seed,4096)
            require(context.manifest["sha256"]==next(r["sha256"] for r in bank_manifest["contexts"] if r["seed"]==seed),
                    "context differs from original top manifest")
            row=dict(context_seed=seed,count=4096,domain=domain_counts(context.theta),
                original_theta=array_descriptor(context.theta),original_observations=array_descriptor(context.obs),
                observation_array_unchanged=True,passed=False)
            if seed==20261111:context11=context
            try:
                before=(array_descriptor(context.theta),array_descriptor(context.obs))
                row["qualification"]=qualify_matrix(context.theta)
                require(before==(array_descriptor(context.theta),array_descriptor(context.obs)),"paired bank arrays mutated")
                row["passed"]=True
            except Exception as error:
                row["error"]=type(error).__name__+": "+str(error)
                document["errors"].append(dict(context_seed=seed,error=row["error"]))
            document["banks"].append(row)
            write_json(output/"manifest.json",document)
        document["all_original_training_rows_inventoried"]=len(document["banks"])==3
        prefix,custody=load_prefix(prefix_run,context11,observations[2:3])
        row=dict(count=1024,custody=custody,domain=domain_counts(prefix),passed=False)
        try:
            row["qualification"]=qualify_matrix(prefix);row["passed"]=True
        except Exception as error:
            row["error"]=type(error).__name__+": "+str(error);document["errors"].append(dict(prefix_rule=PREFIX_RULE,error=row["error"]))
        document["case2_fixed_prefix"]=row
    except Exception as error:
        document["errors"].append(dict(fatal=type(error).__name__+": "+str(error)))
    finally:
        document["RNG_after"]=rng_receipt()
        document["default_RNG_preserved"]=document["RNG_after"]==document["RNG_before"]
        document["banks_unchanged"]=all(sha(Path(name))==digest for name,digest in entry_hashes.items())
        try:
            document["sources_unchanged"]=source_pins()==pins
        except Exception as error:
            document["sources_unchanged"]=False
            document["errors"].append(dict(source_recheck=type(error).__name__+": "+str(error)))
        document["torch_imported"]="torch" in sys.modules
        document["elapsed_seconds"]=time.perf_counter()-started
        document["qualified"]=not document["errors"] and document["default_RNG_preserved"] and document["banks_unchanged"] and document["sources_unchanged"] and not document["torch_imported"]
        document["status"]="codec_inventory_pass" if document["qualified"] else "codec_inventory_failed"
        write_json(output/"manifest.json",document)
    if document["qualified"]:
        document["replay"]=validate_inventory(document,expected_sources=pins)
        write_json(output/"manifest.json",document)
    else:
        raise ValueError("CPU codec inventory failed; exact failure manifest preserved")
    return document

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("banks","prefix-run","output"):parser.add_argument("--"+name,required=True)
    args=parser.parse_args()
    result=run(banks=args.banks,prefix_run=args.prefix_run,output=args.output)
    print(json.dumps(dict(status=result["status"],original_training_rows=12288,
        fixed_prefix_rows=1024,manifest_sha256=sha(Path(args.output)/"manifest.json"))))

if __name__=="__main__":main()
