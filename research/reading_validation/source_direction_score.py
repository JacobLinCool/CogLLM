"""Evaluate a common between-study deviation on the same held families as every baseline."""
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
import torch
from research.backbone_transfer.extract import digest,write_json
from research.reading_validation.borrowing_data import OUTPUT,CALIBRATE,EVALUATE,data
from research.reading_validation.borrowing_score import GeometricPool,material_weights
from research.reading_validation.source_directions import OUT,PLAN,fresh
from research.union_training.fit_observations import weights

METHODS=("private","shared","equal_density_pool","calibrated_density_pool","source_direction")
def scores(mode,seed):
    _,tasks,oldsplit=data();split=oldsplit if mode=="historical" else fresh(tasks)
    root=OUT/mode/f"seed-{seed}";control=OUTPUT/f"seed-{seed}" if mode=="historical" else root
    meta=json.loads((root/"fit.json").read_text());old=json.loads((control/"fit.json").read_text())
    if meta["plan"]!=digest(PLAN) or any(digest(root/k)!=v for k,v in meta["files"].items()) or any(digest(control/k)!=v for k,v in old["files"].items()):raise ValueError("Frozen expert or missing source-direction fit")
    parameters={};values={};source=defaultdict(list)
    with np.load(control/"shared-calibration.npz") as a,np.load(control/"private-calibration.npz") as b:
        for t in tasks:
            ids=np.flatnonzero(split[t.name]==CALIBRATE)
            if len(ids):
                pool=GeometricPool(t,ids,a[t.name],b[t.name]);w,n=material_weights(t,ids)
                def objective(p):return w@pool.loss(p)-np.log(p)-np.log1p(-p)
                r=minimize_scalar(objective,bounds=(1e-7,1-1e-7),method="bounded",options={"xatol":2e-8})
                if not r.success:raise ArithmeticError("Density pool did not converge")
                parameters[t.name]=float(r.x)
            else:parameters[t.name]=.5
    with np.load(control/"shared-development.npz") as a,np.load(control/"private-development.npz") as b,np.load(root/"directions-development.npz") as c:
        for t in tasks:
            ids=np.flatnonzero(split[t.name]==EVALUATE)
            if not len(ids):continue
            w=weights(t,ids);n=float(w@t.trials[ids]);pool=GeometricPool(t,ids,a[t.name],b[t.name]);alone=GeometricPool(t,ids,c[t.name],c[t.name])
            l=[float(w@pool.loss(p)/n) for p in (0.,1.,.5,parameters[t.name])]
            l.append(float(w@alone.loss(1.)/n))
            values[t.name]=dict(source=t.source,geometry=t.likelihood,material_families=len(set(t.family[ids])),rows=len(ids),loss=dict(zip(METHODS,l)))
            source[t.source].append(l)
    sm={k:np.mean(v,axis=0) for k,v in sorted(source.items())};scale=np.stack(list(sm.values()));N=len(scale);rng=np.random.default_rng(423)
    boot=rng.multinomial(N,np.ones(N)/N,size=10000)/N;diff=scale[:,-1,None]-scale[:,:-1];paired={}
    for i,k in enumerate(METHODS[:-1]):
        paired[k]={"difference":float(diff[:,i].mean()),"interval95":np.quantile(boot@diff[:,i],[.025,.975]).tolist(),"sources_improved":int((diff[:,i]<0).sum()),"sources":N}
    if mode=="historical":
        official=json.loads((control/"scores.json").read_text())["cross_source_nll"]
        discrepancy=max(abs(official[oldname]-scale[:,i].mean()) for i,oldname in enumerate(("private","shared","midpoint","adaptive")))
        if discrepancy>1e-7:raise ValueError("Historical controls failed independent verification: "+str(discrepancy))
    write_json(root/"comparison.json",{"fit":digest(root/"fit.json"),"control":digest(control/"fit.json"),"source_scales":parameters,
            "held_rows":sum(v["rows"] for v in values.values()),"baseline_parity":float(discrepancy) if mode=="historical" else None,
            "study_mean":dict(zip(METHODS,scale.mean(0).tolist())),"paired":paired,"endpoints":values,
            "studies":{k:dict(zip(METHODS,v.tolist())) for k,v in sm.items()}})
if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("mode",choices=("historical","fresh"));p.add_argument("seed",type=int);v=p.parse_args();scores(v.mode,v.seed)
