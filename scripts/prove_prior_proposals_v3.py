#!/usr/bin/env python3
"""Independent scalar precision evidence for v3 proposals; no draws or cohorts."""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,math,pickle,platform,random,sys,time
from pathlib import Path
import numpy as np
import scipy
from posteriorforge import prior_proposals_v3 as decoder
from posteriorforge import prior_coordinates_v1 as training

MATH_SHA="d5a22a8a617023ced6d1b043ecca6185bf384a5dcd33bb2ff862050f430028e5"
CBC_SHA="cbc8e44e62780d48e594845bd63a9c72faa76eaf7a1459d9e4dbb2210e989a5a"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(ok,msg):
    if not ok:raise ValueError(msg)
def load_mp(package):
    try:import mpmath as mp
    except ModuleNotFoundError:
        require(package is not None,"explicit existing mpmath source required")
        path=package.resolve()
        spec=importlib.util.spec_from_file_location("mpmath",path/"__init__.py",submodule_search_locations=[str(path)])
        mp=importlib.util.module_from_spec(spec);sys.modules["mpmath"]=mp;spec.loader.exec_module(mp)
    return mp
def proof(zf,spec,mp):
    def exact(x):
        n,d=float(x).as_integer_ratio();return mp.mpf(n)/d
    def calc(dps):
        mp.mp.dps=dps
        mu,sd,lo,hi=[exact(spec[k]) for k in ("location","scale","low","high")]
        a=(lo-mu)/sd;b=(hi-mu)/sd
        Q=lambda x:mp.erfc(x/mp.sqrt(2))/2
        phi=lambda x:mp.exp(-x*x/2)/mp.sqrt(2*mp.pi)
        z=exact(zf);mass=1-Q(-a)-Q(b)
        positive=z>=0;beta=b if positive else -a
        rhs=mass*Q(abs(z))/phi(beta)
        factor=lambda t:mp.quad(lambda v:mp.exp(beta*t*v-t*t*v*v/2),[0,1])
        if rhs<mp.mpf("0.1"):
            t=rhs
            for _ in range(5):t-=(t*factor(t)-rhs)/mp.exp(beta*t-t*t/2)
        else:
            ideal=mu+sd*mp.sqrt(2)*mp.erfinv(1-2*(Q(b)+mass*Q(z)))
            t=((hi-ideal) if positive else (ideal-lo))/sd
        require(t>=0,"negative boundary gap")
        gap=sd*t;boundary=hi if positive else lo
        return gap,boundary,mu,sd,positive,rhs,beta
    g100=calc(100);g100text=mp.nstr(g100[0],100)
    gap,boundary,mu,sd,positive,rhs,beta=calc(180)
    disagreement=abs(gap-mp.mpf(g100text))
    require(disagreement<=abs(gap)*mp.mpf("1e-80"),"highprecision gap stability check failed")
    # Engineering vector only. No physical posterior sample is constructed/saved.
    z=np.zeros((1,21),np.float64);z[0,19]=zf
    value=decoder.decode_ordered_proposals(z)[0,20]
    report=decoder.precision_report(z)
    require(report["hard_passed"],"new interval precision validation failed")
    # Gap-relative arithmetic remains meaningful when boundary +/- gap cannot
    # itself be resolved at180 digits (e.g. z205 gap~1e-9150).
    offset=exact(value)-boundary
    error=abs(offset+(gap if positive else -gap))
    ideal=boundary-gap if positive else boundary+gap
    standard=(ideal-mu)/sd
    bound=exact(training.F64_PHYSICAL_UNITS)*(1+abs(standard))*sd
    require(error<=bound,"independent physical error exceeds unchanged bound")
    spacing=exact(math.ulp(value))
    nearest=float(ideal)
    inward=math.nextafter(float(boundary),-math.inf if positive else math.inf)
    inward_boundary_spacing=abs(exact(inward)-boundary)
    if gap < inward_boundary_spacing/2:nearest=float(boundary)
    return dict(input_z=zf,input_z_hex=zf.hex(),computed_physical=float(value),computed_hex=float(value).hex(),
        exact_boundary=float(boundary),exact_boundary_gap=mp.nstr(gap,85),
        log10_gap=mp.nstr(mp.log10(gap),85),sign="upper" if positive else "lower",
        highprecision_reference="boundary +/- positive gap; endpoint-subtraction not used for error",
        gap_100_180_relative_disagreement=mp.nstr(disagreement/gap,60) if gap else "0",
        nearest_F64=float(nearest),nearest_hex=float(nearest).hex(),
        physical_absolute_error=mp.nstr(error,85),physical_error_in_ULPs=mp.nstr(error/spacing,60),
        unchanged_physical_F64_absolute_bound=mp.nstr(bound,85),
        physical_error_divided_by_unchanged_bound=mp.nstr(error/bound,60),
        original_closed_support=bool(spec["low"]<=value<=spec["high"]),
        finite_proposal=True,precision_interval_report=report,
        reference_error_scope="100/180dps relative-gap stability and independent integral root; NOT certified interval arithmetic, NOT universal correct rounding",
        proposed_sample_adjustment=False)
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--math-report",type=Path,required=True);p.add_argument("--postprocessing-report",type=Path,required=True)
    p.add_argument("--mpmath-package",type=Path);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    require(not a.output.exists(),"fresh proof output required")
    require(sha(a.math_report)==MATH_SHA and sha(a.postprocessing_report)==CBC_SHA,"archived scalar inputs differ")
    mp=load_mp(a.mpmath_package);start=time.perf_counter()
    spec=json.loads(a.math_report.read_text())["source_constants_float"]
    d=json.loads(a.postprocessing_report.read_text())
    cases=[]
    for r in d["records"]:
        if r["stage"]==19 and r["label"]=="N1024":
            point=next(p0 for p0 in r["variants"]["C_softmax64_adjusted_F32logits_F64log_CPUsoftmax64"]["points"] if p0["U_bits"]=="f6ff7f3f")
            cases.append((dict(group="PolicyC16",arm=r["arm"],member=r["member"]),float(point["value"])))
    require(len(cases)==16,"all16C required")
    cases += [(dict(group="predeclared_endpoint_engineering"),z) for z in
              [-205.2462615966797,-13.876729011535645,-8.,0.,8.,13.876729011535645,205.2462615966797]]
    before=dict(numpy=hashlib.sha256(pickle.dumps(np.random.get_state())).hexdigest(),
                python=hashlib.sha256(pickle.dumps(random.getstate())).hexdigest())
    records=[]
    for label,z in cases:
        record=proof(z,spec,mp);record.update(label);records.append(record)
    after=dict(numpy=hashlib.sha256(pickle.dumps(np.random.get_state())).hexdigest(),
               python=hashlib.sha256(pickle.dumps(random.getstate())).hexdigest())
    require(before==after,"default RNG states changed")
    require("torch" not in sys.modules,"proof does not import Torch")
    package=Path(mp.__file__).parent
    mp_files={str(f.relative_to(package)):sha(f) for f in sorted(package.rglob("*.py"))}
    result=dict(schema="posteriorforge.prior_proposals_precision_proof.v3",status="scalar_CPU_proof_pass",
        source_sha256=sha(__file__),decoder_sha256=decoder.source_sha256(),
        decoder_contract=decoder.contract_metadata(),input_math_report_sha256=MATH_SHA,input_C_report_sha256=CBC_SHA,
        working_dps=[100,180],highprecision_source=dict(version=mp.__version__,package=str(package),source_sha256=mp_files,
            loaded_by_explicit_path=a.mpmath_package is not None),
        versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__),
        source_constants=spec,records=records,RNG_before=before,RNG_after=after,
        new_GPU_NN_model_constructor_world_posterior_G3_or_RNG_draw_calls=0,
        full51200_wave_recovered_or_physical_cohort_created=False,
        physical_gate_or_original_support_changed=False,
        wall_seconds=time.perf_counter()-start)
    a.output.mkdir(parents=True);f=a.output/"report.json";f.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print(json.dumps(dict(report=str(f),sha256=sha(f),decoder_sha256=decoder.source_sha256(),count=len(records),
        physical_bound_passed_all=True,wall_seconds=result["wall_seconds"],
        maximum_error_ULPs=max(float(r["physical_error_in_ULPs"]) for r in records),
        maximum_physical_bound_ratio=max(float(r["physical_error_divided_by_unchanged_bound"]) for r in records)),sort_keys=True))
if __name__=="__main__":main()
