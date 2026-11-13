"""Immutable actual-pair custody for fixed prior-coordinate diagnostics.

Built with PriorLabs-TabPFN. This module neither authorizes GPU work nor changes
codec/model/sampling policy. Every old prefix is pre-G3, exact first1024 rows.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib,json,platform
from pathlib import Path
import numpy as np
import scipy
from . import prior_coordinates_v1 as codec,representation_ss15 as representation

SEEDS=(20261111,20261112,20261113)
CASES=tuple(range(8))
COMPLETION_SHA256="6136487ee78375a802c8a0cb0c225a7f1b207063b4083514614c0d97a35c2425"
TRAINING_INVENTORY_SHA256="f3266ab6fd36887c95291020b2cdd2d2a5529f2e6c984b32de22a28b9d31da04"
CODEC_SHA256="d8ce6c49e7b65c6ee55c59b7243d4ce943cbea28c92ebed0329798074aaa70c5"
BACKEND_SHA256="e5a2733e767b187e573315547ec9253267e43d1e3700596f61e97116c53f3256"
REPRESENTATION_SHA256="46323c0b94301c2f7e678538881d5cbbdd8bc52c62a60021798176bffe81eb85"
INVENTORY_HELPER_SHA256="ce68e237f15d239f99f091c2310cf15791d1cf7f7cf7442c2db029241834a636"
INPUT_HELPER_SHA256="3ae723d4748f23af40b89b9508713e270730985b6a8b5d4ccca1959e60ae17f9"
PREREGISTRATION_SHA256="9fc6a91568864cc996dafbcfaa41e77280e4825c63f61d455bdadf3703cd0a66"
PREFIX_RULE="actual_pair_original_pure_FIRST1024_saved_order_v1"
PAIR_CPU_SCHEMA="posteriorforge.prior_coordinate_pair_cpu_panel.v1"


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def source_sha256():return sha(__file__)


@dataclass(frozen=True)
class PairSpec:
    context_seed:int
    case:int
    def __post_init__(self):
        require(type(self.context_seed) is int and self.context_seed in SEEDS,"fixed context seed required")
        require(type(self.case) is int and self.case in CASES,"fixed case0..7 required")
    @property
    def context_size(self):return 4096
    @property
    def count(self):return 50000
    @property
    def repetition(self):return 0
    def as_dict(self):return {"context_seed":self.context_seed,"case":self.case,"context_size":4096,"count":50000,"repetition":0}
    @classmethod
    def from_dict(cls,value):
        require(isinstance(value,dict) and set(value)=={"context_seed","case","context_size","count","repetition"},"exact PairSpec fields required")
        result=cls(value["context_seed"],value["case"])
        require(value==result.as_dict() and all(type(v) is int for v in value.values()),"fixed4096/50000/rep0 PairSpec required")
        return result
    def metadata(self):return {"schema":"posteriorforge.prior_coordinate_pair_spec.v1","pair":self.as_dict(),"module_sha256":source_sha256(),"preregistration_sha256":PREREGISTRATION_SHA256,"prefix_rule":PREFIX_RULE,"execution_authorization_claimed":False}


PANEL=tuple(PairSpec(s,c) for s in SEEDS for c in CASES)
REGRESSION_PAIR=PairSpec(20261111,2)
REMAINING_PAIRS=tuple(p for p in PANEL if p!=REGRESSION_PAIR)
FIRST_PAIR=REMAINING_PAIRS[0]


def load_script(name,digest):
    from .performance_fp16_experiment import load_script as frozen_loader
    return frozen_loader(name,digest)


def require_sources():
    from . import execution_prior_native_v1 as backend
    require(codec.source_sha256()==CODEC_SHA256 and backend.source_sha256()==BACKEND_SHA256,"frozen pair codec/backend changed")
    codec.require_sources();backend.require_sources()
    require(representation.source_sha256()==REPRESENTATION_SHA256,"fixed SS15 changed")
    doc=Path(__file__).resolve().parents[2]/"docs/accuracy-prior-coordinate-fixed-panel-v1.md"
    require(sha(doc)==PREREGISTRATION_SHA256,"fixed panel preregistration changed")
    return True


def source_pins():
    require_sources()
    return {"pair_module":source_sha256(),"codec":CODEC_SHA256,"execution":BACKEND_SHA256,
        "representation":REPRESENTATION_SHA256,"completion":COMPLETION_SHA256,
        "training_inventory":TRAINING_INVENTORY_SHA256,"inventory_helper":INVENTORY_HELPER_SHA256,
        "input_numeric_ancestry":INPUT_HELPER_SHA256,"preregistration":PREREGISTRATION_SHA256}


def _document(path,digest):
    require(sha(path)==digest,"authenticated document SHA differs: "+str(path))
    return json.loads(Path(path).read_bytes())


def grid_receipt(path):
    d=_document(path,COMPLETION_SHA256)
    require(d.get("status")=="complete" and d.get("point_count")==24 and d.get("all_integrity_and_reference_checks_passed") is True and d.get("all_requests_complete") is True,"closed full24 technical receipt required")
    require(d["completed_pairs"]==[[p.context_seed,p.case] for p in PANEL],"predefined24 panel changed")
    return d


def prefix_parent(spec,*,completion_receipt):
    require(type(spec) is PairSpec,"immutable PairSpec required")
    d=grid_receipt(completion_receipt)
    require(len(d["points"])==len(d["details"])==24,"all24 point/detail bindings required")
    matches=[(point,detail) for point,detail in zip(d["points"],d["details"]) if (point["context_seed"],point["case"])==(spec.context_seed,spec.case)]
    require(len(matches)==1,"actual pair missing/duplicated in original24")
    point,detail=matches[0];manifest=Path(detail["point_paths"]["parent"])
    require(point["technical_passed"] is True and point["count"]==50000,"complete original pure parent required")
    header=_document(manifest,point["parent_manifest_sha256"])
    require(header.get("status")=="complete" and header.get("context_seed")==spec.context_seed and header.get("case")==spec.case and header.get("requested_count")==50000 and header.get("context_size")==4096 and header.get("repetition")==0,"original pure parent actual pair differs")
    require(header.get("physical_G3_in_this_neural_request",False) is False and header.get("samples_file")=="samples.npz","pure pre-G3 samples-only parent required")
    payload=manifest.parent/"samples.npz"
    require(sha(payload)==detail["parent_payload_sha256"]==header["samples_sha256"],"original pure payload SHA differs")
    return {"pair":spec.as_dict(),"manifest_path":str(manifest.resolve()),"manifest_sha256":point["parent_manifest_sha256"],
        "payload_path":str(payload.resolve()),"payload_sha256":detail["parent_payload_sha256"],
        "point_paths":dict(detail["point_paths"]),"old_hybrid_manifest_sha256":point["hybrid_manifest_sha256"],
        "old_assessment_sha256":point["assessment_sha256"],"completion_sha256":COMPLETION_SHA256,
        "prefix_source_is_original_pure_pre_G3":True}


def raw_inputs(spec,*,banks):
    from .banks import load_context,load_heldout
    from .inference import CHECKPOINT_SHA256
    from .science.analytical_benchmark_v2 import array_descriptor,build_problem
    require(type(spec) is PairSpec,"immutable PairSpec required")
    context=load_context(banks,spec.context_seed,4096)
    query=np.asarray(load_heldout(banks,spec.case).obs).reshape(1,33)
    require(context.theta.dtype==context.obs.dtype==query.dtype==np.float64,"original Float64 raw inputs required")
    problem=build_problem()
    raw={"checkpoint_sha256":CHECKPOINT_SHA256,"scientific_fingerprint":problem.scientific_fingerprint,"parameter_order":list(problem.parameter_order),"context_theta":array_descriptor(context.theta),"context_obs":array_descriptor(context.obs),"query":array_descriptor(query)}
    return context,query,raw


def load_prefix(spec,*,completion_receipt,parent=None):
    from .science.analytical_benchmark_v2 import array_descriptor,build_problem
    owner=prefix_parent(spec,completion_receipt=completion_receipt)
    if parent is not None:
        path=Path(parent);path=path/"manifest.json" if path.is_dir() else path
        require(path.resolve()==Path(owner["manifest_path"]),"prefix_run belongs to a different actual pair")
    header=json.loads(Path(owner["manifest_path"]).read_bytes())
    with np.load(owner["payload_path"],allow_pickle=False) as saved:
        require(saved.files==["samples"],"original samples-only NPZ required")
        values=saved["samples"].copy()
    require(values.dtype==np.float64 and values.shape==(50000,21) and build_problem().contains(values).all(),"original complete supported physical Float64 parent required")
    require(array_descriptor(values)==header["samples"],"original physical sample descriptor differs")
    prefix=np.ascontiguousarray(values[:1024])
    custody={**owner,"rule":PREFIX_RULE,"selected_indices":list(range(1024)),"entire_payload_shape":[50000,21],"selection_replaced_or_filtered":False,"prefix_physical_Float64":array_descriptor(prefix)}
    return prefix,custody


def validate_pair_cpu_receipt(document,spec):
    require(document.get("schema")==PAIR_CPU_SCHEMA and document.get("source_pins")==source_pins(),"new pair CPU receipt source/schema differs")
    require(document.get("all23_processed") is True and document.get("COMMON_Float32_model_bytes_match_for_passed_pairs") is True,"all23 dual-env CPU processing required")
    require([r["pair"] for r in document["pairs"]]==[p.as_dict() for p in REMAINING_PAIRS],"exact23 pair order/coverage required")
    require(document.get("case2_regression_passed_both_environments") is True,"case2 input/seed/mask/target regression required")
    rows=[r for r in document["pairs"] if r["pair"]==spec.as_dict()]
    require(len(rows)==1 and rows[0]["status"]=="pair_codec_preflight_pass","actual pair CPU codec/domain preflight failed or missing")
    row=rows[0]
    require(row["COMMON_Float32_model_bytes_match"] is True and row["Float64_cross_environment_byte_identity_claimed"] is False,"commonF32/declaredF64 scope differs")
    return row


def input_bindings(spec,*,banks,completion_receipt,training_inventory,pair_inventory=None,old_parent=None,runtime_audit=None):
    from .science.analytical_benchmark_v2 import array_descriptor
    require_sources();context,rawquery,raw=raw_inputs(spec,banks=banks)
    prefix,custody=load_prefix(spec,completion_receipt=completion_receipt,parent=old_parent)
    header=json.loads(Path(custody["manifest_path"]).read_bytes())
    require(all(header[k]==raw[k] for k in raw),"original prefix and actual raw bank/query/science differ")
    inventory=_document(training_inventory,TRAINING_INVENTORY_SHA256)
    bank=next(r for r in inventory["banks"] if r["context_seed"]==spec.context_seed)
    require(bank["passed"] is True and bank["original_theta"]==raw["context_theta"] and bank["original_observations"]==raw["context_obs"],"actual training bank differs from existing3-bank codec inventory")
    latent64=codec.encode_physical(context.theta);theta=codec.order_latents(codec.encode_for_model(context.theta))
    prefix64=codec.encode_physical(prefix);prefixes=codec.order_latents(codec.encode_for_model(prefix))
    require(array_descriptor(theta)==bank["qualification"]["ordered_model_Float32"],"actual training modelF32 differs from original bank inventory")
    ss64=representation.transform_observations(context.obs);q64=representation.transform_observations(rawquery)
    stats=representation.transform_for_model(context.obs);query=representation.transform_for_model(rawquery)
    transformed={"context_ordered_latent_model_Float32":array_descriptor(theta),"context_SS15_Float64":array_descriptor(ss64),"query_SS15_Float64":array_descriptor(q64),"context_SS15_model_Float32":array_descriptor(stats),"query_SS15_model_Float32":array_descriptor(query),"fixed_prefix_ordered_latent_model_Float32":array_descriptor(prefixes),"archived_runtime_inventory_context_latent_Float64":bank["qualification"]["latent_Float64"],"archived_runtime_inventory_prefix_latent_Float64":array_descriptor(prefix64)}
    if pair_inventory is not None:
        document=json.loads(Path(pair_inventory).read_bytes()) if not isinstance(pair_inventory,dict) else pair_inventory
        row=validate_pair_cpu_receipt(document,spec);canonical=row["inference"]["transformed_bindings"]
        for key in transformed:
            if key.endswith("Float32"):require(transformed[key]==canonical[key],"actual modelF32 bytes differ from dual-env actual-pair receipt: "+key)
        require(raw==row["inference"]["raw_bindings"],"actual raw pair differs from CPU receipt")
        transformed=dict(canonical)
    if runtime_audit is not None:
        runtime_audit.update(environment={"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__},context_latent_physical_slots_Float64=array_descriptor(latent64),fixed_prefix_latent_physical_slots_Float64=array_descriptor(prefix64),Float64_latent_cross_environment_byte_identity_claimed=False,model_Float32_exactly_matches_runtime_inventory=True,scope="actual current runtime provenance; canonical archived inferenceF64 provenance is not a cross-version equality claim")
    return theta,stats,query,prefixes,raw,transformed,custody
