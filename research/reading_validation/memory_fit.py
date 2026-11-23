"""Calibrate new human-cohort forecasts using only other families from Experiment 1."""

import argparse
from collections import Counter
import json
from pathlib import Path
import re

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from research.reading_validation.memory_data import BASE, E1, OUT, PLAN, dump, load_rows, sha

AMEND=Path(__file__).with_name("memory-exposure-amendment.md")
SURFACES=("prior_words","sentence_order","sentence_words","word_overlap")
MODELS=("control","frozen","adapted")


def design(sentences,record=None,curve=None):
    seen=set()
    n=0
    feats=[]
    for pos,text in enumerate(sentences,1):
        words=re.findall(r"[^\W_]+",text.casefold())
        bag=set(words)
        feats.append([np.log1p(n),np.log(pos),np.log1p(len(words)),len(bag&seen)/max(len(bag),1)])
        n+=len(words)
        seen|=bag
    blocks=[np.asarray(feats)]
    if curve is not None:
        signals=np.array(curve)
        if signals.shape!=(len(feats),4):
            raise ValueError("Signals do not align to every original sentence end")
        blocks.append(signals)
    if record is not None:
        z=np.array(record,dtype=float)
        if z.shape!=(len(feats),):
            raise ValueError("Comparison logit does not align to the text")
        before=np.r_[z[0],z[:-1]]
        excess=z-np.maximum.accumulate(before)
        blocks.append(np.c_[z,z-before,np.maximum(excess,0)])
    return np.column_stack(blocks)


def counts(source,metadata):
    result={}
    for r in metadata:
        keys=[trial["keys"] for trial in source if trial["id"]==r["id"]]
        if not keys:
            raise ValueError("An Experiment 1 condition lacks trajectories")
        k=np.asarray(keys)=="d"
        # The initial sentence is a forced-consistent practice response.
        k[:,0]=False
        risk=(np.cumsum(k.astype(int),axis=1)-k)==0
        result[r["id"]]=(risk.sum(0),(risk&k).sum(0),len(k))
    return result


def calibrate(heldout,metadata,inputs,events):
    x=[];rates=[];weights=[]
    for r in metadata:
        if r["story"]==heldout:
            continue
        k=r["id"];at,event,readers=events[k]
        t=r["target"]-1
        keep=at>0
        keep[0]=False
        w=at[keep]/readers
        ix=np.flatnonzero(keep)==t
        if ix.sum()!=1:
            raise ValueError("No readers at risk on original target")
        w[ix]+=1
        x.append(inputs[k][keep])
        rates.append(event[keep]/at[keep])
        weights.append(w)
    X=np.vstack(x)
    w=np.concatenate(weights)/len(x)
    y=np.concatenate(rates)
    mu=np.average(X,axis=0,weights=w)
    sd=np.sqrt(np.average((X-mu)**2,axis=0,weights=w))
    sd=np.where(sd>1e-8,sd,1)
    z=(X-mu)/sd
    def loss(v):
        lin=z@v[1:]+v[0]
        nll=w@(np.logaddexp(0,lin)-y*lin)+.01*(v[1:]@v[1:])
        residual=w*(expit(lin)-y)
        return nll,np.r_[residual.sum(),z.T@residual+.02*v[1:]]
    opt=minimize(loss,np.zeros(1+z.shape[1]),jac=True,method="L-BFGS-B",
                 options=dict(maxiter=700,ftol=2e-14,gtol=2e-8))
    grad=float(np.abs(opt.jac).max())
    if not opt.success or grad>2e-5:
        raise ArithmeticError(f"Unconverged calibration: {grad}, {opt.message}")
    return (mu,sd,opt.x,grad)


def run(extraction, dest):
    if dest.exists():
        raise FileExistsError(dest)
    seal=json.loads((OUT/"input-seal.json").read_text())
    receipt=json.loads((extraction/"receipt.json").read_text())
    if sha(PLAN)!=seal["protocol"] or any(sha(extraction/k)!=v for k,v in receipt["files"].items()):
        raise ValueError("Original data plan or L4 returned features changed")
    for k,v in receipt["source_sha256"].items():
        if sha(BASE/"memory"/k)!=v or v!=seal["files"][k]:
            raise ValueError("The reading source differs from the packet sent to the GPU")
    metadata=json.loads((BASE/"memory/study-map.json").read_text())
    # The only human responses admitted to the estimator are the archived E1 data.
    train=[r for r in metadata if r["study"]==1]
    docs={r["id"]:r["sentences"] for r in load_rows(BASE/"memory/stories.jsonl")}
    events=counts(load_rows(E1/"outcomes.jsonl"),train)
    variants={k:json.loads((extraction/f"stories-{k}.json").read_text()) for k in ("frozen","adapted")}
    output={}
    diagnostics={}
    for cohort in ("strict","historical_curve"):
        for model in MODELS:
            weights_source=variants["frozen" if model=="control" else model]
            feats={k:design(doc,weights_source[k]["logits"] if model!="control" else None,
                            weights_source[k]["signals"] if cohort=="historical_curve" else None)
                   for k,doc in docs.items()}
            for fam in sorted({r["story"] for r in train}):
                mu,sd,v,grad=calibrate(fam,train,feats,events)
                label=f"{cohort}.{model}.{fam}"
                diagnostics[label]=dict(grad=grad,slopes=v.tolist(),mean=mu.tolist(),sd=sd.tolist())
                for r in metadata:
                    k=r["id"]
                    if r["story"]!=fam:
                        continue
                    z=(feats[k]-mu)/sd@v[1:]+v[0]
                    hazard=expit(z)
                    hazard[0]=0.
                    if not np.isfinite(hazard).all():
                        raise ValueError("Prediction must be a sequence of finite conditional hazards")
                    output[f"{cohort}.{model}.{k}"]=hazard
    dest.mkdir(parents=True)
    np.savez_compressed(dest/"forecasts.npz",**output)
    dump(dest/"fits.json",diagnostics)
    dump(dest/"seal.json",dict(inputs=sha(OUT/"input-seal.json"),source=sha(extraction/"receipt.json"),
         amendment=sha(AMEND),original_protocol=sha(PLAN),calibration_trials=sum(x[2] for x in events.values()),
         heldout_trials_seen=0,files={k:sha(dest/k) for k in ("forecasts.npz","fits.json")}))
    print(f"Sealed forecasts from {len(diagnostics)} complete, family-held-out fits")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--extraction",required=True,type=Path)
    p.add_argument("--dest",type=Path,default=OUT/"forecast")
    run(**vars(p.parse_args()))
