"""Learn the probability of a literal copy from developmental click ranges and export calibrated weights."""
from pathlib import Path
from collections import Counter
import json
import numpy as np
from scipy.special import expit,softmax,logsumexp
from research.backbone_transfer.extract import digest,write_json
from research.reading_validation.nominal_data import OUT,ROOT
from research.reading_validation.nominal_estimator import RangeModel,action_predict,Words,DETERMINERS
from research.reading_validation.referent_estimation import optimize
from research.reading_validation.referent_sketch_prepare import material

PLAN=Path("research/reading_validation/copy-reference-protocol.md")

def components(form_word,q,form,native):
    same=form_word[:q]==form_word[q];ix=np.flatnonzero(same)
    other=softmax(native)
    if not len(ix):return other,None,None
    copy=np.zeros(q);copy[ix]=softmax(form[ix])
    lag=q-ix[-1];mass=float(other[ix].sum());surface=float(softmax(form)[ix].sum())
    prev=form_word[q-1]
    f=np.array([np.log1p(len(ix)),np.log1p(lag),float(prev=="the"),float(prev in ("a","an")),
               float(prev in DETERMINERS),mass,surface,other.max(),np.log1p(q)])
    return other,copy,f

def fit():
    base=json.loads((OUT/"fit.json").read_text());initial=OUT/"predictions.npz";source=ROOT/"nominal-source-observations.json"
    if digest(initial)!=base["files"]["predictions.npz"]:raise ValueError("Previously trained pointer changed")
    obs=json.loads(source.read_text());loc=json.loads((OUT/"queried-positions.json").read_text());rows=material()["pointer"]
    examples=[];total=0
    with np.load(initial) as raw:
        for r in rows:
            if r["fold"]!="dev":continue
            key=r["id"];words=np.array([r["text"][a:b].casefold() for a,b in r["words"]])
            if not loc[key]:continue
            logits=raw[key+".r"].astype(float);at=0
            for q in loc[key]:
                form,native=logits[:,at:at+q];at+=q
                clicks=obs[key].get(str(q),{}).get("clicks",[])
                pure,copy,f=components(words,q,form,native)
                if f is None or not clicks:continue
                size=sum(c for _,_,c in clicks);total+=size
                for a,b,c in clicks:examples.append((r["group"],q,key,c/size,f,pure[a:b].sum(),copy[a:b].sum()))
    counts=Counter(g for g,_,_ in {(g,k,q) for g,q,k,*_ in examples})
    x=np.stack([r[4] for r in examples]);w=np.array([wt/len(counts)/counts[g] for g,q,k,wt,*_ in examples])
    if abs(w.sum()-1)>1e-8:raise ValueError("Source click/work weighting failed")
    a,b=np.array([e[-2:] for e in examples]).T;mu=w@x;sd=np.sqrt(w@((x-mu)**2));sd[sd<1e-7]=1.
    X=np.c_[np.ones(len(x)),(x-mu)/sd]
    def fn(v):
        pi=expit(X@v);p=(1-pi)*a+pi*b
        cost=-w@np.log(p)+.025*(v@v)
        grad=X.T@(w*(a-b)/p*pi*(1-pi))+.05*v
        return cost,grad
    v,fit=optimize(fn,np.zeros(X.shape[1]));weights=OUT/"copy-gate.npz"
    np.savez_compressed(weights,mu=mu,sd=sd,beta=v)
    write_json(OUT/"copy-fit.json",dict(source_prior=digest(OUT/"fit.json"),source_clicks=digest(source),weights=digest(weights),
        plan=digest(PLAN),source_works=len(counts),source_queries=sum(counts.values()),source_linked_decisions=total,optimization=fit))

def log_forecasts(words,q,form,native,gate):
    raw,copy,x=components(words,q,form,native)
    if copy is None:return np.stack((form-logsumexp(form),native-logsumexp(native),native-logsumexp(native),native-logsumexp(native)))
    mu,sd,v=gate
    pi=expit(np.r_[1,(x-mu)/sd]@v)
    new=(1-pi)*raw+pi*copy;half=.5*(raw+copy)
    return np.stack((form-logsumexp(form),native-logsumexp(native),np.log(half),np.log(new)))

def model_weights():
    base=json.loads((OUT/"fit.json").read_text());source=OUT/"source-models.npz"
    cp=json.loads((OUT/"copy-fit.json").read_text());gate=OUT/"copy-gate.npz"
    if digest(source)!=base["files"]["source-models.npz"] or digest(gate)!=cp["weights"] or cp["plan"]!=digest(PLAN):
        raise ValueError("The frozen source parameters have changed")
    with np.load(source) as p:
        mu,basis=p["mean"],p["basis"];heads={}
        for name in ("form","native"):
            act=tuple(p[name+".action_"+k] for k in ("mu","sd","coef"))
            index=RangeModel(*[p[name+".range_"+k] for k in ("mu","sd","coef")],name=="native")
            heads[name]=(act,index)
    with np.load(gate) as p:copy=tuple(p[k] for k in ("mu","sd","beta"))
    return mu,basis,heads,copy

def query(doc,q,heads,copy):
    priors=[];links=[]
    for act,index in heads.values():
        priors.append(action_predict(act,doc.action(q,index.context)));links.append(index.predict(doc,q))
    return np.stack([priors[0],priors[1],priors[1],priors[1]]),log_forecasts(doc.forms,q,*links,copy)

if __name__=="__main__":fit()
