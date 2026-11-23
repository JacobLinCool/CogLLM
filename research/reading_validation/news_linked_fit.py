"""Estimate source-specific reading transitions and a marginalized pointer on training prompts."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit,logsumexp

from research.reading_validation.memory_data import sha
from research.reading_validation.news_linked_data import WORK,OUT,PLAN


def load(side):
    stamp=json.loads((OUT/"input-seal.json").read_bytes())
    if sha(PLAN)!=stamp["registration"]:
        raise ValueError("News measurement protocol changed")
    for key in ("sequence","outcomes"):
        name=f"{side}-{key}.json"
        if sha(WORK/name)!=stamp["parts"][name]:
            raise ValueError("Changed raw article geometry or observed response")
    seq=json.loads((WORK/(side+"-sequence.json")).read_bytes())
    outcomes=json.loads((WORK/(side+"-outcomes.json")).read_bytes())
    out=[]
    for row,r in zip(seq,outcomes,strict=True):
        k=row["id"];path=WORK/(k+".npz")
        if r["id"]!=k or sha(path)!=stamp["features"][path.name]:
            raise ValueError("Observation and article feature inventories differ")
        with np.load(path,allow_pickle=False) as f:
            data={n:f[n] for n in f.files}
        out.append({**row,**data,"targets":{int(i):[(g,n) for g,n in v] for i,v in r["targets"].items()}})
    return out


def fit_binary(docs,arm,task):
    x=[];w=[];y=[]
    for row in docs:
        c=row["counts"][:,task]
        possible=(c[:,1]>0)&(row["units"]>0)
        if possible.any():
            tot=c[possible,1].astype(float)
            x.append(row["x0" if arm=="lexical" else "x1"][possible])
            y.append(c[possible,0]/tot);w.append(tot/tot.sum())
    x=np.vstack(x).astype(float);y=np.concatenate(y);w=np.concatenate(w)/len(w)
    mu=w@x;sd=np.sqrt(w@((x-mu)**2));sd[sd<1e-6]=1
    x=(x-mu)/sd
    def objective(v):
        p=x@v[1:]+v[0];r=w*(expit(p)-y)
        return w@(np.logaddexp(0,p)-p*y)+.01*(v[1:]@v[1:]),np.r_[r.sum(),x.T@r+.02*v[1:]]
    v=np.zeros(x.shape[1]+1);base=float(w@y)
    v[0]=np.log(base/(1-base))
    fit=minimize(objective,v,jac=True,method="L-BFGS-B",options={"gtol":1e-8,"ftol":1e-14,"maxiter":400})
    if not fit.success or np.max(abs(fit.jac))>1e-5:
        raise ArithmeticError(f"Binary reading transition failed convergence: {fit.message}")
    return dict(w=fit.x.tolist(),mu=mu.tolist(),sd=sd.tolist(),gradient=float(max(abs(fit.jac))),
                loss=float(fit.fun),iterations=int(fit.nit),weighted_positives=base)


def fit_pointer(docs,arm):
    groups=[];labels=[];weights=[]
    width=10 if arm=="lexical" else 22
    for row in docs:
        ends=np.r_[0,np.cumsum(row["units"])]
        observed={i:cs for i,cs in row["targets"].items() if row["units"][i]>1}
        total=sum(n for vals in observed.values() for g,n in vals)
        for i,options in observed.items():
            x=row["source"][ends[i]:ends[i+1],:width]
            for good,n in options:
                mask=np.isin(np.arange(len(x)),good)
                if not mask.any():
                    raise ValueError("Observed source absent from the causal candidates")
                groups.append(x);labels.append(mask);weights.append(n/total)
    sizes=np.array(list(map(len,groups)));cuts=np.r_[0,np.cumsum(sizes)];ix=cuts[:-1]
    g=np.concatenate(labels);w=np.array(weights);w/=w.sum()
    raw=np.vstack(groups).astype(float)
    per=np.repeat(w/sizes,sizes)
    mu=per@raw;sd=np.sqrt(per@((raw-mu)**2));sd[sd<1e-6]=1
    x=(raw-mu)/sd
    def normal(z):
        peak=np.maximum.reduceat(z,ix);v=np.exp(z-np.repeat(peak,sizes))
        denom=np.add.reduceat(v,ix)
        return peak+np.log(denom),v/np.repeat(denom,sizes)
    ww=np.repeat(w,sizes)
    def objective(v):
        scores=x@v
        allz,p=normal(scores);goodz,q=normal(np.where(g,scores,-np.inf))
        return w@(allz-goodz)+.01*(v@v), x.T@(ww*(p-q))+.02*v
    results=[]
    for start in (False,True):
        v=np.zeros(width)
        if start:
            j=8 if arm=="lexical" else 20
            v[j]=sd[j]
        fit=minimize(objective,v,jac=True,method="L-BFGS-B",options={"gtol":1e-8,"ftol":1e-14,"maxiter":500})
        if not fit.success or np.max(abs(fit.jac))>1e-5:
            raise ArithmeticError(f"Conditional source fit failed convergence: {fit.message}")
        results.append(fit)
    chosen=min(results,key=lambda a:a.fun)
    return dict(w=chosen.x.tolist(),mu=mu.tolist(),sd=sd.tolist(),gradients=[float(max(abs(r.jac))) for r in results],
                losses=[float(r.fun) for r in results],observations=len(labels),candidate_entries=int(cuts[-1]))


def run(dest):
    if dest.exists():raise FileExistsError(dest)
    docs=load("train")
    weights={}
    for arm in ("lexical","semantic"):
        weights[arm]={k:fit_binary(docs,arm,j) for j,k in enumerate(("active","enter","exit"))}
        weights[arm]["source"]=fit_pointer(docs,arm)
    dest.mkdir(parents=True)
    (dest/"weights.json").write_text(json.dumps(weights,allow_nan=False)+"\n")
    (dest/"seal.json").write_text(json.dumps(dict(train_documents=len(docs),inventory=sha(OUT/"input-seal.json"),
                                                 weights=sha(dest/"weights.json")),indent=2)+"\n")
    print("Learned 6 binary and 2 source-specific response heads on",len(docs),"documents")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--dest",type=Path,default=OUT/"fit")
    run(p.parse_args().dest)
