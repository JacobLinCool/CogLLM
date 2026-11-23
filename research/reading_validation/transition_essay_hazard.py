"""Share one measured baseline hazard across multiple observed essay revision mechanisms."""
from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from research.reading_validation.transition_essay_reasons import DEST,METRICS,evaluate,fit,infer,mass
from research.reading_validation.transition_external_fit import read_features
from research.reading_validation.transition_external_prepare import csv_rows
from research.reading_validation.transition_score import dump,sha
from research.reading_validation.transition_writer_f1 import threshold


PROTOCOL=Path("research/reading_validation/transition-essay-hazard-protocol.md")


def basis(X,idx,w,arms,*,with_transform=False):
    low=arms["lexical_surprisal"];extra=[i for i in range(10) if i not in low]
    start=10;span=(len(arms["integration"])-start)//3
    groups=[low,extra,*[np.arange(start+i*span,start+(i+1)*span) for i in range(3)]]
    if sorted(int(v) for g in groups for v in g)!=arms["integration"]:raise ValueError("Native feature blocks changed")
    out=[]; coefficients=[]
    for indices in groups:
        v=X[:,indices];mean=w@v[idx];std=np.sqrt(w@((v[idx]-mean)**2))
        valid=std>1e-8
        if valid.any():
            scale=std[valid]*np.sqrt(valid.sum())
            out.append((v[:,valid]-mean[valid])/scale)
            coefficients.extend(zip(np.asarray(indices)[valid],mean[valid],scale,strict=True))
    z=np.column_stack((np.ones(len(X)),*out))
    if not with_transform:return z
    transform=np.zeros((X.shape[1]+1,z.shape[1]));transform[0,0]=1
    for j,(column,center,scale) in enumerate(coefficients,1):
        transform[0,j]=-center/scale;transform[1+column,j]=1/scale
    return z, transform


def mark_loss(h):
    if np.any(h<=0) or np.any(~np.isfinite(h)):raise ArithmeticError("Invalid positive hazard")
    return -np.log(-np.expm1(-h))


def learn(Z,base,r,w):
    complete=(r>=0).all(1);events=r[complete];cc=events*w[complete,None]
    frequency=cc.sum(0)
    if not (frequency>0).all():
        raise ValueError("A source topic fold lacks a reason required by the shared hazard model")
    pi=frequency/frequency.sum();offset=np.log(base[:,None]*pi)
    def objective(v):
        coef=v.reshape(Z.shape[1],7);hazard=np.exp(offset+Z@coef);h=hazard[complete]
        ll=np.empty(len(r));g=np.empty_like(hazard)
        ll[complete]=np.where(events==1,mark_loss(h),h).sum(1)
        g[complete]=np.where(events==1,-h*np.exp(-h)/(-np.expm1(-h)),h)
        if (~complete).any():
            sums=hazard[~complete].sum(1)
            ll[~complete]=mark_loss(sums)
            g[~complete]=-hazard[~complete]*(np.exp(-sums)/(-np.expm1(-sums)))[:,None]
        return float(w@ll+.005*(coef*coef).sum()),(Z.T@(w[:,None]*g)+.01*coef).ravel()
    o=minimize(objective,np.zeros(Z.shape[1]*7),method="L-BFGS-B",jac=True,
               options={"maxiter":400,"ftol":1e-15,"gtol":2e-8,"maxls":50})
    gradient=float(abs(o.jac).max())
    if not np.isfinite(o.fun) or gradient>1e-6:raise ArithmeticError("Unconverged essay hazard")
    return o.x.reshape(-1,7),pi,{"loss":float(o.fun),"max_gradient":gradient,"iterations":int(o.nit)}


def alpha(h,g,y,w):
    def deriv(a):
        value=h+a*(g-h)
        return float(w@((1-y-y*np.exp(-value)/(-np.expm1(-value)))*(g-h)))
    if deriv(0)>=0:return 0.
    if deriv(1)<=0:return 1.
    lo,hi=0.,1.
    for _ in range(58):
        mid=(lo+hi)/2
        if deriv(mid)>0:hi=mid
        else:lo=mid
    return (hi+lo)/2


def main(folder):
    rows,_,F,arms=read_features(folder);index=[i for i,r in enumerate(rows) if r["source"]=="decor"];csv=csv_rows("test_1355_clean.csv")
    data=[csv[rows[i]["source_ref"][1]] for i in index];X=F[index];topic=np.array([r["topic"] for r in data]);essay=np.array([r["essay_id"] for r in data])
    y=np.array([int(r["label"]=="0.0") for r in data]);R=np.array([[0 if not yi else int(float(r["R"+str(j)])) for j in range(1,8)] for r,yi in zip(data,y,strict=True)])
    prior=json.loads((DEST/"scores.json").read_text());order=[x["topic"] for x in prior["folds"]]
    names=("lexical_surprisal","unrestricted_reliability","source_reliability")
    p={m:np.zeros(len(X)) for m in names};ff={m:np.zeros(len(X),bool) for m in p};specific={m:z.copy() for m,z in ff.items()}
    folds=[];surface=X[:,arms["lexical_surprisal"]]
    for i,t in enumerate(order):
        evals=topic==t;dev=topic==order[(i+1)%len(order)];train=~(evals|dev);idx=np.flatnonzero(train)
        w=mass(topic[train],essay[train]);base,_=fit(surface[train],y[train],R[train],w,False);b,_=infer(base,surface)
        h=-np.log1p(-b);Z=basis(X,idx,w,arms)
        cf,pi,con=learn(Z[train],h[train],R[train],w)
        g=h*((np.exp(Z@cf))@pi);a=alpha(h[dev],g[dev],y[dev],mass(topic[dev],essay[dev]))
        cuts={};spec={}
        for m,hz in zip(names,(h,g,h+a*(g-h)),strict=True):
            prob=-np.expm1(-hz);cut=threshold(prob[dev],y[dev])["cutoff"]
            fpr_cut=float(np.quantile(prob[dev & (y==0)],.90,method="higher"))
            p[m][evals]=prob[evals];ff[m][evals]=prob[evals]>=cut;specific[m][evals]=prob[evals]>fpr_cut;cuts[m]=cut;spec[m]=fpr_cut
        folds.append({"topic":t,"source_reliability":a,"residual_convergence":con,"cutoffs":cuts,"source_10pct_fpr":spec})
    _,tp=np.unique(topic,return_inverse=True);_,es=np.unique(essay,return_inverse=True);options={m:(p[m],ff[m]) for m in names}
    rng=np.random.default_rng(31219);point=evaluate(y,options,np.ones((1,len(y))));boot={m:[] for m in names}
    for _ in range(80):
        w=rng.exponential(size=(50,len(set(tp))))[:,tp]*rng.exponential(size=(50,len(set(es))))[:,es]
        scores=evaluate(y,options,w)
        for m in names:boot[m].append(scores[m])
    boot={k:np.concatenate(v) for k,v in boot.items()}
    def ci(p,v):return {k:[float(a),*np.quantile(z,[.025,.975]).tolist()] for k,a,z in zip(METRICS,p[0],v.T,strict=True)}
    result={m:ci(point[m],boot[m]) for m in names};delta={m:ci(point[m]-point[names[0]],boot[m]-boot[names[0]]) for m in names[1:]}
    for metric in METRICS:np.testing.assert_allclose(prior["metrics"][names[0]][metric],result[names[0]][metric],atol=1e-10)
    fixed_specific={m:dict(zip(METRICS,v[0].tolist(),strict=True)) for m,v in evaluate(y,{m:(v,specific[m]) for m,v in p.items()},np.ones((1,len(y)))).items()}
    by_topic={t:{m:dict(zip(METRICS,v[0].tolist(),strict=True)) for m,v in evaluate(y[topic==t],{m:(p[m][topic==t],ff[m][topic==t]) for m in names},
                  np.ones((1,(topic==t).sum()))).items()} for t in order}
    file=DEST/"source-anchored-hazards.npz";np.savez_compressed(file,**p)
    dump(DEST/"source-anchored-hazard-scores.json",{"protocol_sha256":sha(PROTOCOL),"source_essay_result_sha256":sha(DEST/"scores.json"),
         "forecasts_sha256":sha(file),"folds":folds,"metrics":result,"vs_lexical":delta,"all_held_topics":by_topic,"high_specificity":fixed_specific})
    print({m:[v[k][0] for k in ("logloss","auroc","ap","f1")] for m,v in result.items()},[v["source_reliability"] for v in folds])


if __name__=="__main__":
    cli=ArgumentParser(description=__doc__);cli.add_argument("--features",required=True,type=Path);main(cli.parse_args().features)
