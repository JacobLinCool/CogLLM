"""Read a passage through soft location, incompatibility, and source-estimated human hazards."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit

from research.reading_validation.memory_data import BASE,E1,OUT as PRIOR,load_rows,sha
from research.reading_validation.memory_fit import counts,design
from research.reading_validation.retrieval_calibration import word_pool
from research.reading_validation.retrieval_data import RESULT


PLAN=Path(__file__).with_name("memory-mechanism-protocol.md")


def logistic(x,y,weight,contrast=None):
    weight=weight/weight.sum()
    mean=weight@x
    sd=np.sqrt(weight@((x-mean)**2))
    sd=np.where(sd>1e-6,sd,1)
    z=np.column_stack((np.ones(len(x)),(x-mean)/sd))
    def obj(v):
        u=z@v
        loss=weight@(np.logaddexp(0,u)-y*u)
        grad=z.T@(weight*(expit(u)-y))
        if contrast is not None:
            differences=z[contrast[:,0]]-z[contrast[:,1]]
            d=differences@v
            loss=.5*(loss+np.logaddexp(0,-d).mean())
            grad=.5*(grad-differences.T@expit(-d)/len(d))
        loss+=.01*(v[1:]@v[1:])
        grad[1:]+=.02*v[1:]
        return loss,grad
    fitted=minimize(obj,np.zeros(z.shape[1]),jac=True,method="L-BFGS-B",options={"ftol":2e-14,"gtol":1e-8})
    error=max(abs(fitted.jac))
    if not fitted.success or error>1e-5:
        raise ArithmeticError(f"Source location or incompatibility fit did not converge: {error}")
    return dict(coef=fitted.x.tolist(),center=mean.tolist(),scale=sd.tolist(),gradient=error,iterations=fitted.nit)


def activate(model,block,offset=0.):
    v=np.asarray(model["coef"])
    return expit(v[0]+((block-model["center"])/model["scale"])@v[1:]+offset)


def fit(selected,feat,observed):
    forms=[];targets=[];weights=[]
    materials=[];relation=[];pairs=[]
    offset=[]
    rates=[]
    for family in sorted({r["story"] for r in selected}):
        entries=sorted((r for r in selected if r["story"]==family),key=lambda r:r["arm"])
        if [r["arm"] for r in entries]!=["consistent","inconsistent"]:
            raise ValueError("Source arm or matching family is missing")
        indices=[]
        background=[]
        conditional=[]
        for i,r in enumerate(entries):
            at,first,_=observed[r["id"]]
            t=r["target"]-1
            a=feat[r["id"]][1:]
            y=np.arange(len(a))==t-1
            forms.append(a);targets.append(y);offset.append(y.mean())
            weights.append(np.where(y,.5,.5/(len(y)-1)))
            indices.append(len(materials));materials.append(a[t-1]);relation.append(i)
            conditional.append((first[t]+.5)/(at[t]+1))
            background.append((first[1:t].sum()+.5)/(at[1:t].sum()+1))
        pairs.append(indices[::-1])
        rates.append([np.mean(background),*conditional])
    x,y,w=np.vstack(forms),np.concatenate(targets),np.concatenate(weights)
    return dict(location=logistic(x,y,w),offset=float(logit(np.mean(offset))),
                relation=logistic(np.stack(materials),np.array(relation),np.ones(len(relation)),np.array(pairs)),
                rates=np.mean(rates,axis=0).tolist())


def run(archive,output,extra=None):
    if output.exists():
        raise FileExistsError(output)
    src=archive/"stories-frozen.json"
    if json.loads((archive/"receipt.json").read_bytes())["files"][src.name]!=sha(src):
        raise ValueError("Base native forecasts changed")
    original=json.loads(src.read_bytes())
    texts={r["id"]:r["sentences"] for r in load_rows(BASE/"memory/stories.jsonl")}
    meta=json.loads((BASE/"memory/study-map.json").read_bytes())
    train=[r for r in meta if r["study"]==1]
    observed=counts(load_rows(E1/"outcomes.jsonl"),train)
    features={"form":{}, "native":{}}
    if extra:
        q=json.loads(extra.read_bytes());features["retrieval"]={}
    for k,s in texts.items():
        features["form"][k]=design(s)
        features["native"][k]=np.column_stack((design(s,original[k]["logits"]),word_pool(original[k])))
        if extra:
            features["retrieval"][k]=np.column_stack((features["native"][k],q[k]["surprisal"],q[k]["retrieval"]))
    vectors={};fits={}; diagnostics={}
    with np.load(PRIOR/"forecast/forecasts.npz",allow_pickle=False) as past:
        for k in texts:
            vectors["strict.original."+k]=past["strict.frozen."+k]
    for family in sorted({r["story"] for r in train}):
        included=[r for r in train if r["story"]!=family]
        for name,block in features.items():
            m=fit(included,block,observed); fits[f"{family}.{name}"]=m
            base,compatible,incompatible=m["rates"]
            for row in meta:
                if row["story"]!=family:
                    continue
                k=row["id"]
                location=activate(m["location"],block[k],m["offset"])
                conflict=activate(m["relation"],block[k])
                t=row["target"]-1
                target=(np.arange(len(location))==t).astype(float)
                assigned=row["arm"]=="inconsistent"
                oracle_conflict=np.where(target>0,float(assigned),conflict)
                diagnostics[f"{row['study']}.{name}.{k}"]=dict(location=location.tolist(),conflict=conflict.tolist(),
                                                               target=int(t),role=row["arm"],family=family)
                for mode,p,rel in (("article",location,conflict),("target_given",target,conflict),
                                   ("condition_given",location,oracle_conflict),("design_given",target,oracle_conflict)):
                    h=(1-p)*base+p*((1-rel)*compatible+rel*incompatible)
                    h[0]=0
                    vectors["strict."+name+"."+mode+"."+k]=h
    output.mkdir(parents=True)
    np.savez_compressed(output/"forecasts.npz",**vectors)
    for path,contents in (("models.json",fits),("diagnostics.json",diagnostics)):
        (output/path).write_text(json.dumps(contents,allow_nan=False)+"\n")
    spec=dict(protocol=sha(PLAN),e1=sha(E1/"outcomes.jsonl"),native=sha(src),retrieval=sha(extra) if extra else None,
              models=["original"]+[n+"."+v for n in features for v in ("article","target_given","condition_given","design_given")],
              files={s:sha(output/s) for s in ("forecasts.npz","models.json","diagnostics.json")})
    (output/"seal.json").write_text(json.dumps(spec,indent=2)+"\n")
    print("Exported",len(fits),"out-of-family role and human-hazard estimators")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("archive",type=Path);p.add_argument("output",type=Path);p.add_argument("--extra",type=Path)
    a=p.parse_args();run(a.archive,a.output,a.extra)
