#!/usr/bin/env python3
"""Portable CPU replay of the existing a83 GQA recovery, without relabeling it.

Built with PriorLabs-TabPFN. The old830 verifier remains unchanged and retains
its historical-machine requirements. This NEW verifier authenticates archived
historical evidence and current SDK source bytes separately. It never applies
old creation-time conditions to a fresh SDK install, calls models or integrates
references, or changes any saved numerical predicate/threshold.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

SCHEMA="posteriorforge.portable_gqa_recovery_verification.v1"
RECOVERY_SOURCE_SHA256="83064f54712014b5c4c66569e854872caff03593cfdc337f2472a7b5723aea02"
BATCH_SOURCE_SHA256="31422927077911879b6cd155605a6a668796305ee97ccf6585819d8fe494cfde"
RECOVERY_MANIFEST_SHA256="a83e4cfb925317aea7a4d576deee9e0337fb66bd3d3e80470f7e7102b4c4ea48"
BATCH_MANIFEST_SHA256="7bc0e4449abe4aaf0a62b0df55fb25eda423f821524955c9266f8f91761aa77c"
RAW_MANIFEST_SHA256="152062c486d9273d2531395fbcabf8eb19424151e85090069b14488d64245626"
GOLDEN_MANIFEST_SHA256="7de3549d1ad0c90c11b81ed751b4c9a28dd51441e17ae841f9ff372a8cc63642"
FUNCTION_SOURCE_SHA256="7da6b7c6d060a396b4aa7024fd5ac2c9bfddbc7d62029b113962cecbdaca6787"
DECISION_COUNT=184


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def bytes_sha(value):return hashlib.sha256(value).hexdigest()
def source_sha256():return sha(__file__)


def load_helper(name,expected):
    path=Path(__file__).with_name(name)
    require(sha(path)==expected,"reviewed helper source changed: "+name)
    spec=importlib.util.spec_from_file_location("_portable_replay_"+path.stem,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def archived_bytes(root,entry):
    root=Path(root).resolve();path=(root/entry["file"]).resolve()
    require(path.is_relative_to(root) and path.is_file() and sha(path)==entry["sha256"],
            "archived historical/source input checksum mismatch")
    return path.read_bytes()


def historical_time_check(evidence,raw,golden):
    """Archived old timestamps only. Never stat the fresh SDK or its RECORD."""
    history=evidence.get("historical_authentication",{})
    records=history.get("recorded_source_revisions",[])
    require(len(records)==2,"both archived source revisions/times required")
    expected=[{"git_revision":doc["git_revision"],"created_unix":doc["created_unix"],
               "wheel_authentication_bytes_equal":True,"inference_lock_bytes_equal":True} for doc in (golden,raw)]
    require(records==expected,"archived revision/creation-time lineage differs")
    first=min(doc["created_unix"] for doc in (golden,raw))
    ctime=history.get("installed_function_ctime_ns");mtime=history.get("installed_function_mtime_ns")
    require(type(ctime) is int and type(mtime) is int and ctime<first*1e9 and mtime<first*1e9,
            "archived original function timestamps do not predate the GPU evidence")
    return {"archived_original_function_ctime_ns":ctime,"archived_original_function_mtime_ns":mtime,
            "earliest_original_gpu_evidence_created_unix":first,
            "fresh_install_timestamps_checked":False,
            "cached_RECORD_numeric_ctime_archived":False,
            "cached_RECORD_historical_check_basis":"Exact a83 recovery plus pinned830 source bind its original successful RECORD-ctime check; no numeric RECORD timestamp was archived or is invented."}


def callable_source_audit(sdk_root,evidence,helper):
    path=Path(sdk_root)/"model/preprocessing.py"
    require(sha(path)==FUNCTION_SOURCE_SHA256==evidence.get("function_source_sha256"),
            "relocated callable source bytes differ")
    text=path.read_text();tree=ast.parse(text)
    definitions={node.name:node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in helper.FUNCTIONS}
    require(set(definitions)==set(helper.FUNCTIONS),"both pinned module-level repr callables required")
    audits={};digests={}
    for name,node in definitions.items():
        require(not node.decorator_list and not node.args.defaults and all(value is None for value in node.args.kw_defaults),
                "callable decorators/defaults differ")
        require(not any(isinstance(value,(ast.FunctionDef,ast.Lambda,ast.Nonlocal)) for value in ast.walk(ast.Module(body=node.body,type_ignores=[]))),
                "callable closure/binding differs")
        audits[name]={"module_level_function":True,"defaults":None,"kwdefaults":None,"closure":None,"bound_or_partial":False,
            "source_sha256":bytes_sha(ast.get_source_segment(text,node).encode()),
            "runtime_global_policy":"_identity has no runtime globals; _inf_to_nan_func uses NumPy nan_to_num/nan from recorded1.26.4 hash-locked environment"}
        digests[name]=bytes_sha(ast.dump(node,include_attributes=False).encode())
    require(audits==evidence.get("callable_source_audit") and digests==evidence.get("function_ast_sha256"),
            "archived callable implementation/default/global-policy audit differs")
    return {"function_source_sha256":FUNCTION_SOURCE_SHA256,"function_AST_sha256":digests,
            "defaults_closure_binding_and_recorded_numpy_global_policy_checked":True,
            "SDK_imported_or_callable_executed":False}


def historical_source_check(evidence,raw,golden,*,batch_root,bundle,sdk_root,helper):
    """Bind recorded old lineage to archived files and the current source tree."""
    from posteriorforge import execution_cache as cache
    archives=bundle["archives"]
    auth_bytes=archived_bytes(batch_root,archives["historical_wheel_authentication"])
    record_bytes=archived_bytes(batch_root,archives["wheel_record"])
    lock_bytes=archived_bytes(batch_root,archives["inference_lock"])
    auth=json.loads(auth_bytes);history=evidence["historical_authentication"]
    require(bytes_sha(auth_bytes)==history["auth_manifest_sha256"]
            and bytes_sha(record_bytes)==history["cached_record_sha256"]==auth["cached_record_sha256"]
            and bytes_sha(lock_bytes)==history["inference_lock_sha256"],
            "archived authentication/RECORD/lock lineage mismatch")
    repo=Path(__file__).resolve().parents[1];auth_name="evidence/diagnostics/model-sdk-source-authenticity.json"
    require((repo/auth_name).read_bytes()==auth_bytes and (repo/"environments/inference.lock").read_bytes()==lock_bytes,
            "public recorded authentication/lock differs from archive")
    for doc in (golden,raw):
        require(doc.get("versions",{}).get("numpy")=="1.26.4" and doc["versions"].get("tabpfn")=="2.0.9",
                "recorded callable dependency version differs")
        for name,expected in ((auth_name,auth_bytes),("environments/inference.lock",lock_bytes)):
            actual=subprocess.run(["git","-C",str(repo),"show",doc["git_revision"]+":"+name],check=True,capture_output=True).stdout
            require(actual==expected,"historically recorded Git authentication/lock differs")
    require(evidence.get("sdk_source_sha256")==cache.PINNED_SDK_SHA256,"archived original pinned SDK source map differs")
    for name,digest in evidence["sdk_source_sha256"].items():
        require(bundle["sdk_source_sha256"].get(name)==digest,"original pinned SDK source not in complete portable RECORD")
    require(evidence.get("normalization_scope")=="only <function _inf_to_nan_func|_identity at 0xHEX> process addresses",
            "archived normalization scope expanded")
    return {"historical_time":historical_time_check(evidence,raw,golden),
            "historical_Git_RECORD_lock_and_dependency_lineage_verified":True,
            "callables":callable_source_audit(sdk_root,evidence,helper),
            "historical_absolute_SDK_and_cached_wheel_paths_are_informational":True}


def replay_exact_document(document,raw,golden,helper):
    """Reuse every unchanged830 saved-predicate check and all184 narrow decisions."""
    require(document.get("raw_qualification_manifest_sha256")==RAW_MANIFEST_SHA256
            and document.get("golden_manifest_sha256")==GOLDEN_MANIFEST_SHA256,
            "original failed/golden manifest identity differs")
    require(raw.get("status")=="conditional_mismatch" and raw.get("full21_gqa_hybrid_qualified") is False
            and document.get("raw_status_preserved") is True and document.get("raw_producer_status")==raw["status"],
            "raw FAILED producer status must stay immutable")
    expected=helper.recover_document(raw,golden,raw_sha=RAW_MANIFEST_SHA256,golden_sha=GOLDEN_MANIFEST_SHA256,
        source_evidence=document["configuration_identity_sdk_evidence"])
    require(document==expected,"portable replay differs from complete original recovery document")
    paths=document.get("configuration_identity_corrected_paths",[])
    require(len(paths)==DECISION_COUNT and len(set(paths))==DECISION_COUNT,
            "exact184 existing narrow identity decisions required")
    return {"entire_recovery_document_equal":True,"original_saved_predicates_replayed":True,
            "existing_narrow_identity_decisions_replayed":len(paths),"raw_FAILED_status_preserved":True,
            "new_identity_decisions_or_numeric_relabels":0}


def verify_portable_recovery(recovery_manifest,batch_manifest,*,sdk_root):
    """Verify existing immutable a83 evidence on a fresh/relocated SDK source tree."""
    recovery_manifest=Path(recovery_manifest).resolve();batch_manifest=Path(batch_manifest).resolve();sdk_root=Path(sdk_root).resolve()
    initial_source=source_sha256()
    require(sha(recovery_manifest)==RECOVERY_MANIFEST_SHA256,"exact immutable a83 recovery manifest required")
    require(sha(batch_manifest)==BATCH_MANIFEST_SHA256,"exact immutable7bc portable source bundle producer required")
    helper=load_helper("recover_gqa_hybrid_qualification.py",RECOVERY_SOURCE_SHA256)
    portable=load_helper("qualify_execution_gqa_batches.py",BATCH_SOURCE_SHA256)
    document=json.loads(recovery_manifest.read_text());batch=json.loads(batch_manifest.read_text())
    require(document.get("schema")==helper.SCHEMA and document.get("recovery_source_sha256")==RECOVERY_SOURCE_SHA256,
            "original recovery schema/source changed")
    require(batch.get("status")=="batch_screen_pass" and batch.get("qualified") is True and batch.get("sources_unchanged") is True,
            "original portable bundle producer status/source differs")
    bundle=batch["portable_source_authentication"]
    require(bundle.get("sdk_distribution")=="tabpfn==2.0.9" and len(bundle.get("sdk_source_sha256",{}))==22
            and bundle.get("all_wheel_sdk_python_sources_checked") is True
            and bundle.get("verification_requires_original_machine_paths") is False and bundle.get("verification_uses_file_ctime") is False,
            "complete portable22-source/RECORD contract required")
    portable.verify_portable_sources(batch_manifest.parent,bundle,sdk_root=sdk_root)
    require(archived_bytes(batch_manifest.parent,bundle["archives"]["base_qualification"])==recovery_manifest.read_bytes(),
            "portable bundle base is not exact existing a83 recovery")
    raw_bytes=archived_bytes(recovery_manifest.parent,document["recovery_inputs"]["raw_gpu_proof"])
    golden_bytes=archived_bytes(recovery_manifest.parent,document["recovery_inputs"]["golden_manifest"])
    require(bytes_sha(raw_bytes)==RAW_MANIFEST_SHA256 and bytes_sha(golden_bytes)==GOLDEN_MANIFEST_SHA256,
            "immutable original raw/golden source bytes differ")
    raw,golden=json.loads(raw_bytes),json.loads(golden_bytes)
    historical=historical_source_check(document["configuration_identity_sdk_evidence"],raw,golden,
        batch_root=batch_manifest.parent,bundle=bundle,sdk_root=sdk_root,helper=helper)
    replay=replay_exact_document(document,raw,golden,helper)
    require(sha(recovery_manifest)==RECOVERY_MANIFEST_SHA256 and sha(batch_manifest)==BATCH_MANIFEST_SHA256
            and source_sha256()==initial_source,"artifact/verifier changed during portable replay")
    require(sha(Path(helper.__file__))==RECOVERY_SOURCE_SHA256 and sha(Path(portable.__file__))==BATCH_SOURCE_SHA256,
            "reviewed source changed during replay")
    return {"schema":SCHEMA,"status":"verified_existing_recovery","attribution":"Built with PriorLabs-TabPFN",
        "verifier_source_sha256":initial_source,"recovery_manifest_sha256":RECOVERY_MANIFEST_SHA256,
        "batch_source_bundle_manifest_sha256":BATCH_MANIFEST_SHA256,"raw_failed_manifest_sha256":RAW_MANIFEST_SHA256,
        "golden_manifest_sha256":GOLDEN_MANIFEST_SHA256,"old_recovery_source_sha256":RECOVERY_SOURCE_SHA256,
        "portable_bundle_source_sha256":BATCH_SOURCE_SHA256,"current_sdk_source_directory":str(sdk_root),
        "current_sdk_path_is_identity":False,"all22_current_SDK_source_bytes_and_archived_RECORD_verified":True,
        "current_install_ctime_checked":False,"old830_helper_claimed_portable":False,
        "historical_source_authentication":historical,"replay":replay,"model_calls":0,"gpu_calls":0,"reference_integrations":0,
        "original_artifacts_unchanged":True,"new_qualification_or_numeric_waiver":False,
        "scope":"New portable verifier of the same existing184-decision a83 recovery; all original saved math/state/RNG/route/CDF/ICDF/source predicates retained."}


def load_verified_recovery(recovery_manifest,batch_manifest,*,sdk_root):
    verify_portable_recovery(recovery_manifest,batch_manifest,sdk_root=sdk_root)
    require(sha(recovery_manifest)==RECOVERY_MANIFEST_SHA256,"recovery changed after verification")
    return json.loads(Path(recovery_manifest).read_text())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("recovery-manifest","batch-manifest","sdk-root"):parser.add_argument("--"+name,required=True)
    parser.add_argument("--output",help="fresh JSON verification report; no original artifact is modified")
    args=vars(parser.parse_args());output=args.pop("output")
    result=verify_portable_recovery(**args)
    if output:
        path=Path(output)
        with path.open("x") as handle:json.dump(result,handle,sort_keys=True,indent=2,allow_nan=False);handle.write("\n")
    print(json.dumps({key:result[key] for key in ("schema","status","recovery_manifest_sha256","current_install_ctime_checked","replay")},indent=2))


if __name__=="__main__":main()
