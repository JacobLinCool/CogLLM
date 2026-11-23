"""Produce mark-and-source forecasts for every word with all target responses withheld."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import expit,softmax

from research.reading_validation.memory_data import load_rows,sha
from research.reading_validation.news_linked_data import WORK,OUT


def evaluate_linear(spec,x,bias):
    x=(x-np.asarray(spec["mu"]))/spec["sd"];v=np.asarray(spec["w"])
    return x@v[1:]+v[0] if bias else x@v


def curves(units,x,sources,heads,arm):
    n=len(units);width=int(units.max())+1
    probs=[expit(evaluate_linear(heads[k],x,True)) for k in ("active","enter","exit")]
    active,enter,exit_rate=probs
    cursor=0
    current=np.zeros(width);current[0]=1.
    point=np.zeros((n,width));temporal=np.zeros_like(point)
    design=10 if arm=="lexical" else 22
    z=evaluate_linear(heads["source"],sources[:,:design],False)
    for i,m in enumerate(units):
        if m==0:
            point[i,0]=1.;temporal[i,0]=1.
            continue
        p=softmax(z[cursor:cursor+m])
        cursor+=m
        point[i,0]=1-active[i];point[i,1:m+1]=active[i]*p
        vacant=current[0]+(1-current[0])*exit_rate[i]
        current[1:]*=1-exit_rate[i]
        current[1:m+1]+=vacant*enter[i]*p
        current[0]=vacant*(1-enter[i])
        temporal[i]=current
    if cursor!=len(z):
        raise ValueError("Source probabilities did not exhaust exactly the available earlier candidates")
    return point,temporal


def run(dest):
    if dest.exists():
        raise FileExistsError(dest)
    lock=json.loads((OUT/"input-seal.json").read_bytes());fit=json.loads((OUT/"fit/seal.json").read_bytes())
    if sha(OUT/"fit/weights.json")!=fit["weights"] or sha(OUT/"input-seal.json")!=fit["inventory"]:
        raise ValueError("The fitted news response or source inventory changed")
    w=json.loads((OUT/"fit/weights.json").read_bytes());articles=json.loads((WORK/"test-sequence.json").read_bytes())
    prior=Path("research/results/scarecrow-detection-l4/forecasts.jsonl")
    historic={r["id"]:np.array(r["probabilities"]["qwen"]) for r in load_rows(prior)}
    arrays={}
    for doc in articles:
        k=doc["id"];path=WORK/(k+".npz")
        if sha(path)!=lock["features"][path.name]:
            raise ValueError("Changed native evidence or group membership")
        with np.load(path,allow_pickle=False) as f:
            units=f["units"];source=f["source"]
            for arm in ("lexical","semantic"):
                x=f["x0" if arm=="lexical" else "x1"]
                a,b=curves(units,x,source,w[arm],arm)
                arrays[k+"."+arm+".point"]=a;arrays[k+"."+arm+".temporal"]=b
            fixed=f["frozen"];p=1-historic[k][:,0]
            baseline=np.zeros_like(a);cursor=0
            for i,m in enumerate(units):
                if m:
                    baseline[i,1:m+1]=p[i]*fixed[cursor:cursor+m]
                    baseline[i,0]=1-p[i];cursor+=m
                else:
                    baseline[i,0]=1.
            arrays[k+".historical_external"]=baseline
        for v in (arrays[k+"."+name] for name in ("semantic.point","semantic.temporal","lexical.point","lexical.temporal","historical_external")):
            if (not np.isfinite(v).all() or np.min(v)<-1e-10 or not np.allclose(v.sum(1),1,atol=2e-7)
                    or (v[:,1:]*(np.arange(v.shape[1]-1)[None,:]>=units[:,None])).any()):
                raise ArithmeticError("Joint distribution lost mass or assigned a future source")
    dest.mkdir()
    np.savez_compressed(dest/"predictions.npz",**arrays)
    receipt=dict(sealed_fit=sha(OUT/"fit/seal.json"),test_documents=len(articles),historical=sha(prior),
                 predictions=sha(dest/"predictions.npz"))
    (dest/"seal.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print("Uncued joint distributions for",len(articles),"original news continuations")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--dest",type=Path,default=OUT/"forecast")
    run(p.parse_args().dest)
