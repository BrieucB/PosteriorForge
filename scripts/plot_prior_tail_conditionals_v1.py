#!/usr/bin/env python3
"""Compact saved-density diagnostic plot. Built with PriorLabs-TabPFN.

CPU display only; never a posterior plot or exact historical density replay.
Predeclared illustration: member0, original/candidate, known N1024 windows.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json
from fractions import Fraction
import numpy as np

REFERENCE_SHA="8c35b5e10e5b15383b8a2112f25fa555e8d11016a30fef0abcc4167c527e87c1"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def draw(report_path,expected_sha,output):
 report_path=Path(report_path);assert sha(report_path)==expected_sha
 report=json.loads(report_path.read_bytes());root=report_path.parent
 p=Path(__file__).with_name("prior_tail_icdf_reference_v1.py");assert sha(p)==REFERENCE_SHA
 s=importlib.util.spec_from_file_location("_tail_plot_exact_mass",p);r=importlib.util.module_from_spec(s);s.loader.exec_module(r)
 import matplotlib
 matplotlib.use("Agg")
 import matplotlib.pyplot as plt
 from matplotlib.lines import Line2D
 fig,axes=plt.subplots(2,3,figsize=(16,10),squeeze=False)
 colors={"original":"#1768ac","candidate":"#d16a12"}
 legend=[Line2D([0],[0],color=colors[a],lw=1.6,label=a+" FP32 conditional evaluation") for a in colors]
 legend += [Line2D([0],[0],color="#333",ls="-",marker="o",label="frozen F32 inverse / captured cumulative"),
            Line2D([0],[0],color="#333",ls="--",marker="x",label="exact-normalized EXISTING F32 weights"),
            Line2D([0],[0],color="#b00020",ls="none",marker="*",ms=10,label="historical scalar only; member/grouping unknown")]
 fig.suptitle("NEW conditional evaluations at saved prefixes: noise tail diagnosis\nMember0, known1024-row layouts; NOT an accepted50k/full posterior or historical density recreation",y=.99,fontsize=15)
 fig.legend(handles=legend,loc="upper center",bbox_to_anchor=(.5,.935),ncol=3,fontsize=9)
 summaries=[]
 for line,stage in enumerate((15,19)):
  density,cdf,quantile=axes[line];selected=[]
  for arm in ("original","candidate"):
   row=next(x for x in report["entries"] if(x["stage"],x["arm"],x["member"],x["label"])==(stage,arm,0,"N1024"))
   name="focal-densities/"+Path(row["source_file"]).stem+"-focal.npz"
   proof=next(x for x in report["focal_subsets"] if x["file"]==name);path=root/name;assert sha(path)==proof["sha256"]
   with np.load(path,allow_pickle=False) as a:arrays={n:a[n] for n in a.files}
   units=r._mass_units(arrays["probabilities"]);total=sum(units)
   weights=np.array([float(Fraction(v,total)) for v in units],np.float64)
   running=0;reference_cdf=[0.]
   for mass in units:running+=mass;reference_cdf.append(float(Fraction(running,total)))
   b=arrays["borders"].astype(np.float64);height=weights/np.diff(b)
   density.stairs(np.where(height>0,height,np.nan),b,color=colors[arm],lw=1.3)
   cdf.plot(b,np.r_[0.,arrays["cdf"].astype(np.float64)]-1,color=colors[arm],lw=1.25)
   cdf.plot(b,np.asarray(reference_cdf)-1,color=colors[arm],ls="--",lw=1.3)
   points=row["provided_U_points"];u=arrays["uniforms"].astype(np.float64);tail=1-u
   stock=np.array([float(x["stock"]["output_F32"]) for x in points])
   normalized=np.array([x["reference_inverse_Float64"] for x in points])
   quantile.plot(tail,stock,color=colors[arm],marker="o",lw=1.2)
   quantile.plot(tail,normalized,color=colors[arm],ls="--",marker="x",lw=1.4)
   focal=row["authenticated_historical_U_counterfactual_result"]
   cdf.axhline(focal["U_Float32"]-1,color="#8551a3",ls=":",lw=.9)
   selected.append((row,b,focal))
  row,b,focal=selected[0]
  stage_name="sigma_a1" if stage==15 else "sigma_a3";historical=13.876729011535645 if stage==15 else 205.2462615966797
  density.set(title=f"stage{stage}: {stage_name} captured bar density",xlabel="latent noise coordinate z",ylabel="normalized density (log scale)")
  density.set_yscale("log");density.set_xlim(float(b[0]),float(b[-1]))
  low=min(x[0]["provided_U_points"][0]["reference_inverse_Float64"] for x in selected)
  cdf.set(title=f"same captured weights: cumulative tail at bar edges",xlabel="latent z (reference0.99 quantile to upper edge)",ylabel="CDF − 1")
  cdf.set_xlim(low,float(b[-1]));distance=1-focal["U_Float32"]
  cdf.set_ylim(-4*distance,3e-7)
  cdf.text(.98,.08,f"reconstructed U={focal['U_Float32']:.10f}",transform=cdf.transAxes,ha="right",fontsize=8)
  quantile.axhline(float(b[-1]),color="#777",ls=":",lw=.8)
  quantile.plot([1-focal["U_Float32"]],[historical],"*",color="#b00020",ms=12)
  quantile.set_xscale("log");quantile.invert_xaxis()
  quantile.set(title="provided-U inverse: frozen vs exact-normalized",xlabel="1 − provided Float32 U (log scale)",ylabel="latent z")
  annotation=[]
  for row,_,f in selected:
   annotation.append(f"{row['arm']}: stock {float(f['stock']['output_F32']):.3f}, norm {f['reference_inverse_Float64']:.3f}")
  quantile.text(.04,.96,"\n".join(annotation)+"\nupper dotted line: actual finite border",transform=quantile.transAxes,va="top",fontsize=8,bbox={"facecolor":"white","alpha":.85,"edgecolor":"none"})
  groups=[g for g in report["groups"] if g["stage"]==stage and g["label"]=="N1024"]
  caption=[]
  for g in groups:
   rows=[x for x in report["entries"] if(x["stage"],x["arm"],x["label"])==(stage,g["arm"],"N1024")]
   noise=[x["authenticated_historical_U_counterfactual_result"]["normalized_reference_d8_noise_domain"] for x in rows]
   rt32=sum(not x["would_pass_noise_only_Float32_latent_roundtrip"] for x in noise)
   rt64=sum(not x["would_pass_noise_only_Float64_latent_roundtrip"] for x in noise)
   caption.append(f"{g['arm']} all8: normalized noise-decode fail {g['normalized_reference_d8_noise_decode_failure_count']}/8; noise-only RT32/64 fail {rt32}/{rt64}")
  summaries.append(f"stage{stage}: "+" · ".join(caption))
 fig.text(.5,.055,"\n".join(summaries),ha="center",va="bottom",fontsize=8)
 fig.text(.5,.012,"Built with PriorLabs-TabPFN · ExistingF32 weights normalized exactly, NOT softmax64 · u1 is separate out-of-contract stress\nAll8 numeric records retained; scalar noise diagnostics are not full d8 qualification. No recovery, new posterior, G3 or live policy change.",ha="center",va="bottom",fontsize=8)
 fig.subplots_adjust(left=.065,right=.985,top=.81,bottom=.15,wspace=.29,hspace=.35)
 output=Path(output);assert not output.exists();fig.savefig(output,dpi=165);plt.close(fig)
 print(json.dumps({"figure":str(output),"sha256":sha(output),"member":0,"layout":"N1024","analysis_sha256":expected_sha}))
if __name__=="__main__":
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--report",required=True);p.add_argument("--report-sha256",required=True);p.add_argument("--output",required=True)
 a=p.parse_args();draw(a.report,a.report_sha256,a.output)
