#!/usr/bin/env python3
"""Render frozen grid summaries only. Built with PriorLabs-TabPFN.

No metric/status/reference/sample-array calculations. The root-supplied new
qualification job ID records the authorized display-task execution trigger.
"""
import argparse,hashlib,json,re
from pathlib import Path
INDEX_SHA256="da5c31350734ab97d7a4baf07a264d19e93aaaade0df1a5727fb9893dd1b5016"
SEEDS=(20261111,20261112,20261113)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(value,message):
 if not value:raise ValueError(message)

def run(*,index,output,qualification_job_id):
 require(isinstance(qualification_job_id,str) and re.fullmatch("[0-9]+",qualification_job_id),
         "explicit root qualification-job trigger required")
 index=Path(index);require(sha(index)==INDEX_SHA256,"exact frozen baseline24 index required")
 raw=index.read_bytes();document=json.loads(raw)
 require(document["grid_complete"] and document["assessed_points"]==24 and document["pending_points"]==0,
         "completed original24 required")
 rows=document["rows"]
 require([(r["context_seed"],r["case"]) for r in rows]==[(s,c) for s in SEEDS for c in range(8)],
         "all24 original positions required")
 require(all(r["returned_count"]==50000 and r["reference_qualified"] is True for r in rows),
         "complete qualified-reference cohorts required")
 output=Path(output);output.mkdir(parents=True,exist_ok=False)
 import matplotlib
 matplotlib.use("Agg")
 import matplotlib.pyplot as plt
 from matplotlib.colors import ListedColormap
 from matplotlib.patches import Patch
 import numpy as np
 panels=(
  ("Archived physical gate status",None,None),
  ("Worst marginal CDF upper — limit <= 0.10","marginal_CDF_upper_max","failed_marginal_parameters"),
  ("Worst joint-projection CDF upper — limit <= 0.10","joint_CDF_upper_max","failed_joint_CDF_indices"),
  ("Worst mean-bias upper / reference SD — limit <= 0.20","bias_upper_refSD_max","failed_mean_parameters"),
  ("SD intervals: minimum lower / maximum upper across21 — allowed [0.8,1.2]","SD","failed_SD_ratio_parameters"))
 palette=ListedColormap(["#dcefd8","#f5d5d0"])
 with plt.rc_context({"font.family":"DejaVu Sans","font.size":9}):
  figure,axes=plt.subplots(5,1,figsize=(13.8,12.2))
  figure.subplots_adjust(left=.10,right=.99,bottom=.075,top=.885,hspace=.69)
  for axis,(title,key,flag) in zip(axes,panels):
   states=np.zeros((3,8),dtype=int);texts=[]
   for i,seed in enumerate(SEEDS):
    line=[]
    for case in range(8):
     row=rows[i*8+case]
     if key is None:
      states[i,case]=int(not row["absolute_physical_gate_passed"])
      text="PASS" if row["absolute_physical_gate_passed"] else "FAIL"
     else:
      # Fill follows archived failure flags; no rounded-value decision is made.
      require(isinstance(row[flag],str),"closed archived metric flags required")
      states[i,case]=int(bool(row[flag]))
      text=(f'{row["SD_ratio_interval_min"]:.3f} / {row["SD_ratio_interval_max"]:.3f}'
            if key=="SD" else f"{row[key]:.4f}")
     line.append(text)
    texts.append(line)
   axis.imshow(states,cmap=palette,vmin=0,vmax=1,aspect="auto",interpolation="nearest")
   for i in range(3):
    for case in range(8):
     axis.text(case,i,texts[i][case],ha="center",va="center",fontsize=9,
               color="#273028",fontweight="bold" if key is None else "normal")
   axis.set_title(title,loc="left",fontsize=10,pad=10)
   axis.set_yticks(range(3),labels=[str(s) for s in SEEDS])
   axis.set_xticks(range(8),labels=[f"case{c}" for c in range(8)])
   axis.set_xticks(np.arange(-.5,8,1),minor=True);axis.set_yticks(np.arange(-.5,3,1),minor=True)
   axis.grid(which="minor",color="white",linewidth=2);axis.tick_params(which="both",length=0)
   axis.set_ylabel("context seed")
   for spine in axis.spines.values():spine.set_visible(False)
  figure.suptitle("Frozen sigma-first SS15 baseline + one exact joint-global refresh\n"
                  f'{document["passed_points"]} PASS / {document["failed_points"]} MODEL_FAILURE — 24 cells, 50,000 physical rows each',
                  fontsize=15,y=.976,fontweight="bold")
  figure.text(.10,.927,"Mixed FP16 / native FP32 neural execution; same fixed method at every point",fontsize=10)
  figure.legend(handles=[Patch(facecolor="#dcefd8",label="Archived pass"),
                         Patch(facecolor="#f5d5d0",label="Archived failure")],
                loc="upper right",bbox_to_anchor=(.99,.916),frameon=False,ncol=2,fontsize=9)
  figure.text(.10,.037,"Rounded display only; archived full-precision flags determine color and status. "
              "Shared-case development diagnostic: not SBC or global calibration.",fontsize=9)
  figure.text(.10,.017,"Built with PriorLabs-TabPFN | Frozen grid-index SHA256 "+INDEX_SHA256[:16]+"...",
              fontsize=9,color="#4d4d4d")
  path=output/"sigma-native-grid-overview.png"
  figure.savefig(path,dpi=170,metadata={"Software":"PosteriorForge; Built with PriorLabs-TabPFN"})
  plt.close(figure)
 (output/"input-grid-index.json").write_bytes(raw)
 provenance={"schema":"posteriorforge.sigma_native_grid_overview.v1",
  "source_grid_index_sha256":INDEX_SHA256,"source_plot_helper_sha256":sha(__file__),
  "qualification_job_id_display_trigger":qualification_job_id,"method":document["method"],
  "points":24,"rows_per_point":50000,"passed":document["passed_points"],"model_failures":document["failed_points"],
  "source_flags_not_recomputed":True,"rounded_numbers_are_display_only":True,
  "metrics_recomputed":False,"raw_sample_arrays_loaded":False,"new_model_calls":0,"GPU_hours":0,
  "plot_library":{"matplotlib":matplotlib.__version__},
  "SD_scope":"minimum lower and maximum upper among21 marginal uncertainty intervals; not one coordinate or joint interval",
  "prior_candidate_qualification_claimed":False,"global_calibration_or_SBC_claimed":False}
 (output/"provenance.json").write_text(json.dumps(provenance,sort_keys=True,indent=2)+"\n")
 (output/"README.md").write_text(f"""# Frozen sigma-first grid overview

Built with PriorLabs-TabPFN. **24/24 points: {document["passed_points"]} pass,
{document["failed_points"]} model failures, zero pending.** Each cell uses all
50,000 supported physical vectors and its original qualified reference.

This is the frozen hybrid method: fixed SS15, sigma-first block-local neural
conditionals with mixed FP16/native FP32 execution, followed by one exact
joint-global Gibbs refresh. It is a shared-case development diagnostic,
**not SBC, global calibration, or an untouched confirmation set**.

![All24 archived physical outcomes and existing worst-case metrics](sigma-native-grid-overview.png)

Cell color comes from the archived full-precision failure flags. Numbers are
rounded only for display; a rounded boundary value does not change a gate.
Original limits are CDF upper <=0.10, mean-bias upper <=0.20 reference SD,
and each SD-ratio uncertainty interval contained in [0.8,1.2].
The SD panel shows the minimum lower and maximum upper endpoints over all21
marginal intervals, which need not belong to the same parameter. All reference
precision checks remain those of the saved assessments.

The only data input is the exact frozen grid-index.json, SHA256
{INDEX_SHA256}. Its unchanged bytes are copied here as input-grid-index.json.
No samples, references, metrics, or statuses were recomputed.
The [full frozen evidence](../corrected-sigma-grid-01/README.md) retains original
assessments, all parent/after-refresh outcomes, plots and exact manifests.
This overview does not assess or qualify the new prior-coordinate candidate.

The display ran after the root supplied new-qualification job
{qualification_job_id}. That identifier is a workflow trigger, not evidence
that the new qualification passed.
""")
 closure={p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()}
 (output/"checksums.json").write_text(json.dumps(closure,sort_keys=True,indent=2)+"\n")
 require(index.read_bytes()==raw and sha(index)==INDEX_SHA256,"frozen source index changed")
 return dict(image_sha256=sha(path),checksums_sha256=sha(output/"checksums.json"),points=24,
             passed=document["passed_points"],failed=document["failed_points"])
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ("index","output","qualification-job-id"):p.add_argument("--"+name,required=True)
 args=p.parse_args();print(json.dumps(run(**vars(args)),sort_keys=True))
if __name__=="__main__":main()
