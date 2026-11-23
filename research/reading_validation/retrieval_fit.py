"""Learn conditional source retrieval and premise compatibility using train-only triples."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from research.reading_validation.memory_data import sha
from research.reading_validation.retrieval_data import DATA, RESULT
from research.reading_validation.retrieval_features import Corpus, SURFACE


def standardize(x):
    center=x.mean(0,dtype=np.float64)
    scale=x.std(0,dtype=np.float64)
    scale[scale<1e-6]=1
    return center,scale


def solve(fun,n):
    result=minimize(fun,np.zeros(n),jac=True,method="L-BFGS-B",
                    options={"ftol":1e-14,"gtol":5e-8,"maxiter":600})
    grad=float(np.abs(result.jac).max())
    if not result.success or grad>3e-5:
        raise ArithmeticError(f"Unconverged source readout ({grad}): {result.message}")
    return result.x, dict(iterations=int(result.nit),gradient=grad,loss=float(result.fun))


def relevance(cases,indices):
    choices=[(v[:,indices],ref) for v,ref,_ in cases if len(v)>1]
    offsets=np.r_[0,np.cumsum([len(v) for v,_ in choices])]
    x=np.vstack([v for v,_ in choices])
    mu,sd=standardize(x)
    z=((x-mu)/sd).astype(np.float64)
    gold=offsets[:-1]+[ref for _,ref in choices]
    sizes=np.diff(offsets)
    n=len(sizes)
    def objective(w):
        score=z@w
        maximum=np.maximum.reduceat(score,offsets[:-1])
        unscaled=np.exp(score-np.repeat(maximum,sizes))
        den=np.add.reduceat(unscaled,offsets[:-1])
        prob=unscaled/np.repeat(den,sizes)
        loss=np.mean(maximum+np.log(den)-score[gold])+.01*(w@w)
        prob[gold]-=1
        return loss, z.T@prob/n+.02*w
    w,diag=solve(objective,len(mu))
    return pack(w,mu,sd,indices,0.,diag)


def conflict(cases,indices):
    x=np.stack([v[ref,indices] for v,ref,_ in cases])
    y=np.array([label for _,_,label in cases])
    mu,sd=standardize(x)
    z=(x-mu)/sd
    groups=np.arange(len(y)).reshape(-1,3)
    if not np.all(y[groups].sum(1)==1):
        raise ValueError("Need exactly one conflicting premise per training triple")
    positive=groups[y[groups]==1]
    negative=groups[y[groups]==0].reshape(-1,2)
    d=(z[positive,None,:]-z[negative]).reshape(-1,z.shape[1])
    def objective(v):
        w,b=v[:-1],v[-1]
        linear=z@w+b
        gap=d@w
        error=expit(linear)-y
        loss=np.mean(np.logaddexp(0,linear)-y*linear)+.5*np.logaddexp(0,-gap).mean()+.01*(w@w)
        grad=z.T@error/len(y)-.5*d.T@expit(-gap)/len(gap)+.02*w
        return loss,np.r_[grad,error.mean()]
    v,diag=solve(objective,len(mu)+1)
    return pack(v[:-1],mu,sd,indices,v[-1],diag)


def pack(w,mu,sd,indices,bias,diag):
    return dict(coefficients=w.tolist(),center=mu.tolist(),scale=sd.tolist(),columns=indices.tolist(),bias=float(bias),
                convergence=diag)


def affine(spec):
    weights=np.asarray(spec["coefficients"])/spec["scale"]
    return np.asarray(spec["columns"]), weights, spec["bias"]-weights@spec["center"]


def scores(x,projection):
    ix,w,b=projection
    return x[...,ix]@w+b


class Readout:
    def __init__(self,weights):
        self.relevance=affine(weights["relevance"])
        self.conflict=affine(weights["conflict"])

    def __call__(self,pairs):
        z=scores(pairs,self.relevance)
        p=np.exp(z-z.max(-1,keepdims=True))
        p/=p.sum(-1,keepdims=True)
        conditional=expit(scores(pairs,self.conflict))
        marginal=np.clip(np.sum(p*conditional,axis=-1),1e-9,1-1e-9)
        source_if_conflict=p*conditional
        source_if_conflict/=np.maximum(source_if_conflict.sum(-1,keepdims=True),1e-20)
        return np.log(marginal)-np.log1p(-marginal),p,source_if_conflict


def run(archive,output):
    if output.exists():
        raise FileExistsError(output)
    corpus=Corpus(archive)
    meta=json.loads((DATA/"training.json").read_bytes())
    cases=[]
    for row in meta:
        if row["split"]!="train":
            continue
        for key,label in zip(row["ids"],row["y"],strict=True):
            page=corpus.page(key)
            v=page.pairs(len(page.sections)-1,terminal=True)[0]
            cases.append((v,row["evidence"],label))
    width=cases[0][0].shape[1]
    result={}
    for name,n in (("lexical",len(SURFACE)),("semantic",width)):
        ix=np.arange(n)
        result[name]={"relevance":relevance(cases,ix),"conflict":conflict(cases,ix)}
    output.mkdir(parents=True)
    (output/"weights.json").write_text(json.dumps(result,allow_nan=False)+"\n")
    seal=dict(method=sha(Path(__file__).with_name("retrieval-estimator.md")),
              source=sha(DATA/"training.json"),archive=sha(archive/"receipt.json"),training_triples=len(cases)//3,
              features=width,weights=sha(output/"weights.json"))
    (output/"seal.json").write_text(json.dumps(seal,indent=2)+"\n")
    print(json.dumps({k:{s:v["convergence"] for s,v in w.items()} for k,w in result.items()}))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive",type=Path)
    parser.add_argument("--output",type=Path,default=RESULT/"fit")
    args=parser.parse_args()
    run(args.archive,args.output)
