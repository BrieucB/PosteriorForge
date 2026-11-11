#!/usr/bin/env python3
"""Plot one retained predefined full-vector projection; no reference evaluation."""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path
import numpy as np
from posteriorforge.artifacts import sha256_file, write_json
from posteriorforge.banks import load_heldout
from posteriorforge.reference import load_reference_distribution
from posteriorforge.science.analytical_benchmark_v2 import array_descriptor


def render(*, banks, run, reference, gate, output):
    banks, run, reference, gate, output = map(Path, (banks, run, reference, gate, output))
    gate_bytes = gate.read_bytes()
    final_gate = json.loads(gate_bytes)
    if final_gate["schema"] != "posteriorforge.scientific_gate.v1":
        raise ValueError("scientific gate artifact required")
    selected = next(row for row in final_gate["runs"] if row["context_seed"] == 20261111 and row["case"] == 0)
    selected_ref = next(row for row in final_gate["references"] if row["case"] == 0)
    run_path = run / "manifest.json"
    if sha256_file(run_path) != selected["manifest_sha256"]:
        raise ValueError("gate/run manifest hash mismatch")
    manifest = json.loads(run_path.read_text())
    if (manifest["status"] != "complete" or manifest["context_seed"] != 20261111 or manifest["case"] != 0
            or manifest["requested_count"] != 50000 or manifest["context_size"] != 4096):
        raise ValueError("requires complete seed20261111 case0 at context4096")
    if sha256_file(run/"samples.npz") != manifest["samples_sha256"]:
        raise ValueError("sample payload hash mismatch")
    with np.load(run/"samples.npz", allow_pickle=False) as payload:
        samples = payload["samples"].copy()
    if samples.shape != (50000,21) or not np.isfinite(samples).all() or array_descriptor(samples) != manifest["samples"]:
        raise ValueError("sample array descriptor mismatch")
    reference_hash = selected_ref.get("grid_manifest_sha256", selected_ref["manifest_sha256"])
    if sha256_file(reference/"manifest.json") != reference_hash:
        raise ValueError("gate/reference-grid manifest hash mismatch")
    arrays, reference_manifest = load_reference_distribution(reference)
    if reference_manifest["state"] != "verified" or reference_manifest["case_index"] != 0:
        raise ValueError("verified case0 reference required")
    if "grid_manifest_sha256" in selected_ref:
        if reference_manifest["parent_manifest_sha256"] != selected_ref["manifest_sha256"]:
            raise ValueError("retained reference-grid parent hash mismatch")
    heldout = load_heldout(banks, 0)
    binding = reference_manifest["binding"]
    if (manifest["query"] != array_descriptor(heldout.obs)
            or binding["observation_sha256"] != manifest["query"]["sha256"]
            or binding["parameter_order"] != manifest["parameter_order"]
            or binding["scientific_fingerprint"] != manifest["scientific_fingerprint"]):
        raise ValueError("science/order/observation binding mismatch")
    bounds = selected["assessment"]["joint_projection_cdf"]
    upper = np.asarray(bounds["upper"], dtype=np.float64)
    lower = np.asarray(bounds["lower"], dtype=np.float64)
    if upper.shape != (32,) or lower.shape != (32,) or not np.isfinite(upper).all() or not np.isfinite(lower).all():
        raise ValueError("32 retained joint CDF bounds required")
    index = int(np.argmax(upper))
    reference_row = 21 + index
    direction = np.asarray(arrays["directions"][reference_row], dtype=np.float64)
    if direction.shape != (21,) or not np.isfinite(direction).all() or np.any(direction == 0):
        raise ValueError("predefined complete-vector projection must include all21 coordinates")
    values = samples @ direction
    true_projection = float(heldout.theta[0] @ direction)
    queries = arrays["queries"]
    cdf = arrays["cdfs"][reference_row]
    low_ref = queries[min(int(np.searchsorted(cdf,.0005)),len(queries)-1)]
    high_ref = queries[min(int(np.searchsorted(cdf,.9995)),len(queries)-1)]
    low = min(float(values.min()),float(low_ref),true_projection)
    high = max(float(values.max()),float(high_ref),true_projection)
    padding = .025*(high-low) if high>low else 1.
    output.mkdir(parents=True, exist_ok=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(11.5,7.4))
    figure.subplots_adjust(top=.76,bottom=.30,left=.09,right=.97)
    figure.suptitle("Built with PriorLabs-TabPFN\nCase 0: most discrepant predefined full-vector projection",fontsize=18,fontweight="bold",color="#172b43",y=.97)
    ordered = np.sort(values)
    axis.step(np.r_[low-padding,ordered,high+padding], np.r_[0.,np.arange(1,len(ordered)+1)/len(ordered),1.],
              where="post",color="#c56a19",linewidth=1.7,label="PFN empirical CDF · 50,000 complete vectors",zorder=3)
    axis.plot(queries,cdf,color="#2162b8",linewidth=2,label="Qualified retained reference CDF")
    axis.axvline(true_projection,color="#263748",linestyle="--",linewidth=1.4,label="True held-out vector projected after inference")
    axis.set(xlim=(low-padding,high+padding),ylim=(0,1),xlabel="Scalar projection cᵀθ (coefficients include prior scaling)",ylabel="Posterior cumulative probability")
    axis.set_title(f"Projection {index+1}/32 (zero-based index {index}) · CDF discrepancy [{lower[index]:.3f}, {upper[index]:.3f}] · required ≤ 0.10",fontsize=11,pad=12)
    axis.set_yticks([0,.25,.5,.75,1])
    axis.grid(alpha=.18)
    axis.spines[["top","right"]].set_visible(False)
    handles,labels=axis.get_legend_handles_labels()
    figure.legend(handles,labels,loc="lower center",bbox_to_anchor=(.5,.08),frameon=False,fontsize=10,ncol=1)
    figure.text(.5,.042,"One of 32 frozen one-dimensional linear projections using all 21 parameters; diagnostic, not a full-joint or SBC certificate.\nSelected by the largest retained case-0 joint CDF upper bound. No new reference evaluation, altered samples, or changed gate.",ha="center",fontsize=9,color="#425775")
    image_path=output/"case0-worst-predefined-joint-projection.png"
    figure.savefig(image_path,dpi=170,bbox_inches="tight",facecolor="white")
    plt.close(figure)
    print(json.dumps({"png":str(image_path)}),flush=True)
    sidecar=output/"case0-worst-predefined-joint-projection.json"
    metadata={"schema":"posteriorforge.joint_projection_diagnostic.v1","attribution":"Built with PriorLabs-TabPFN",
              "case":0,"context_seed":20261111,"context_size":4096,"sample_count":50000,
              "projection_zero_based_index":index,"reference_cdf_row":reference_row,"predefined_projection_count":32,
              "selection":"argmax retained joint_projection_cdf.upper for fixed seed20261111 case0; first index wins a tie",
              "all_21_parameters_used":True,"parameter_order":manifest["parameter_order"],
              "coefficients":[{"parameter":name,"coefficient":float(coefficient)} for name,coefficient in zip(manifest["parameter_order"],direction)],
              "cdf_distance_lower_bound":float(lower[index]),"cdf_distance_upper_bound":float(upper[index]),
              "true_vector_projection":true_projection,"truth_usage":"post-inference diagnostic marker only",
              "reference_state":reference_manifest["state"],"reference_cdf_precision":reference_manifest["cdf_precision"],
              "gate_status":final_gate["status"],"case_status":selected["status"],
              "not_full_joint_certification":True,"not_SBC":True,"reference_recomputed":False,
              "samples_altered":False,"gate_changed":False,"gate_interpolation":False,
              "inputs":{"gate_sha256":sha256(gate_bytes).hexdigest(),"inference_manifest_sha256":selected["manifest_sha256"],
                        "samples_sha256":manifest["samples_sha256"],"reference_manifest_sha256":reference_hash,
                        "parent_reference_manifest_sha256":selected_ref["manifest_sha256"],
                        "reference_distribution_sha256":reference_manifest["distribution"]["sha256"],
                        "heldout_npz_sha256":sha256_file(banks/"heldout.npz"),"heldout_manifest_sha256":sha256_file(banks/"heldout.json")},
              "figure":{"file":image_path.name,"sha256":sha256_file(image_path)},
              "regeneration":{"script":"scripts/plot_joint_projection.py","arguments":{"banks":str(banks),"run":str(run),"reference":str(reference),"gate":str(gate),"output":"fresh output directory"}}}
    write_json(sidecar,metadata)
    return {"png":str(image_path),"sidecar":str(sidecar),"projection_index":index,"cdf_bounds":[float(lower[index]),float(upper[index])]}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("banks","run","reference","gate","output"):
        parser.add_argument("--"+name,required=True)
    print(json.dumps(render(**vars(parser.parse_args())),indent=2))

