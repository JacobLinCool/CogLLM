"""Fit line-indexed births and interval-censored resolutions with source-only wording supervision."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from research.reading_validation.memory_data import sha
from research.reading_validation.narrative_course_data import OUT as ORIGINAL
from research.reading_validation.narrative_course_features import load
from research.reading_validation.narrative_survival_data import geometry
from research.reading_validation.narrative_survival_features import Lines, learn_teacher, pair_design

OUT=Path("research/results/reading-validation-narrative-survival")
ARMS=("duration","lexical","native","wording")


def pages(archive,old):
    receipt=json.loads((archive/"receipt.json").read_bytes())
    if sha(archive/"forecasts.npz")!=receipt["file"] or receipt["content_forwards"]!=4:
        raise ValueError("Original whole-story semantic forecasts or one-pass accounting changed")
    base=load(old)
    meta=json.loads((ORIGINAL/"input-seal.json").read_bytes())
    result={}
    with np.load(archive/"forecasts.npz",allow_pickle=False) as data:
        for name,story in base.items():
            k=meta["articles"][name]["id"]
            for key,expected in (("offsets",story.offsets),("ends",story.last),("lines",story.units),("state",story.state)):
                if not np.array_equal(expected,data[k+"."+key]):
                    raise ValueError(f"Changed exact story coordinates or states: {name}/{key}")
            for key,expected in (("trace",story.trace),("pairs",story.relations)):
                if not np.allclose(expected,data[k+"."+key],rtol=1e-6,atol=1e-6):
                    raise ValueError(f"Numerical change in previously certified study view: {name}/{key}")
            result[name]=Lines(story,data[k+".line_semantic"].astype(float))
    return result


def hazards(pages_,designs):
    # A risk row carries its question's total weight. A closure interval is one event,
    # including when several unclipped lines could have closed the question.
    known=[];kw=[];event=[];ew=[];labels=[];examples=0
    for p,x in zip(pages_,designs,strict=True):
        counts=Counter(s.team for s in p.records)
        for s in p.records:
            w=1/(len(pages_)*len(counts)*counts[s.team])
            rows=p.ind[s.source+1:s.definite+1,s.source]
            if len(rows):
                known.append(x[rows]);kw.extend([w]*len(rows))
            if s.upper is not None:
                end=p.ind[s.definite+1:s.upper+1,s.source]
                event.append(x[end]);ew.append(w);labels.extend([examples]*len(end));examples+=1
    xk=np.vstack(known);xe=np.vstack(event);wk=np.array(kw);we=np.array(ew);group=np.array(labels)
    w=np.r_[wk,we[group]];z=np.vstack((xk,xe));mu=np.average(z,axis=0,weights=w)
    sd=np.sqrt(np.average((z-mu)**2,axis=0,weights=w));sd[sd<1e-6]=1
    def design(v):
        return np.column_stack((np.ones(len(v)),(v-mu)/sd))
    zk,ze=design(xk),design(xe)
    def objective(v):
        lk=zk@v;le=ze@v
        total=np.bincount(group,weights=np.logaddexp(0,le),minlength=examples)
        probability=-np.expm1(-total)
        if np.any(probability<=0):
            raise ArithmeticError("Resolution interval has no representable probability")
        derivative=-we*np.exp(-total)/probability
        loss=wk@np.logaddexp(0,lk)-we@np.log(probability)+.01*(v[1:]@v[1:])
        g=zk.T@(wk*expit(lk))+ze.T@(derivative[group]*expit(le))
        g[1:]+=.02*v[1:]
        return loss,g
    init=np.zeros(zk.shape[1]);init[0]=np.log(we.sum()/wk.sum())
    result=minimize(objective,init,jac=True,method="L-BFGS-B",options={"gtol":1e-8,"ftol":1e-14,"maxiter":900})
    if not result.success or max(abs(result.jac))>1e-5:
        raise ArithmeticError(f"Full-risk hazard failed convergence: {result.message}")
    return dict(center=mu.tolist(),scale=sd.tolist(),coef=result.x.tolist(),loss=float(result.fun),
                gradient=float(max(abs(result.jac))),definite_exposures=len(wk),interval_candidates=len(group),
                definite_and_interval_events=examples)


def birth_design(p,arm):
    if arm=="duration":
        return p.birth_form[:,:1]
    if arm=="lexical":
        return p.birth_form
    return np.column_stack((p.birth_form,p.native))


def births(pages_,arm):
    xx=[];yy=[];ww=[]
    for p in pages_:
        x=birth_design(p,arm)
        for team in p.story.teams:
            y=np.bincount([r.source for r in p.records if r.team==team],minlength=p.n)
            xx.append(x);yy.append(y);ww.extend([1/(len(pages_)*len(p.story.teams)*p.n)]*p.n)
    x=np.vstack(xx);y=np.concatenate(yy);w=np.array(ww)
    mu=w@x;sd=np.sqrt(w@((x-mu)**2));sd[sd<1e-6]=1
    dim=pages_[0].birth_form.shape[1]
    if arm in ("native","wording"):
        sd[dim:]*=np.sqrt(len(sd)-dim)
    z=np.column_stack((np.ones(len(x)),(x-mu)/sd))
    def objective(v):
        lin=z@v
        rate=np.exp(lin)
        g=z.T@(w*(rate-y));g[1:]+=.02*v[1:]
        return w@(rate-y*lin)+.01*(v[1:]@v[1:]),g
    start=np.zeros(z.shape[1]);start[0]=np.log(w@y)
    opt=minimize(objective,start,jac=True,method="L-BFGS-B",options={"gtol":1e-8,"ftol":1e-14,"maxiter":900})
    if not opt.success or max(abs(opt.jac))>1e-5:
        raise ArithmeticError("Question-opening Poisson fit failed its gradient check")
    return dict(center=mu.tolist(),scale=sd.tolist(),coef=opt.x.tolist(),loss=float(opt.fun),
                gradient=float(max(abs(opt.jac))))


def linear(fit,x):
    v=np.array(fit["coef"])
    return (x-fit["center"])/fit["scale"]@v[1:]+v[0]


def propagate(p,m,source,x,arm):
    q=expit(linear(m,x))
    births=np.exp(linear(source,birth_design(p,arm)))
    pending=np.zeros(p.n);total=np.zeros(p.n)
    mass=np.zeros((p.n,p.n));hz=np.zeros_like(mass)
    for t in range(p.n):
        if t:
            rate=q[p.blocks[t-1]]
            hz[t,:t]=rate
            mass[t,:t]=pending[:t]*rate
            pending[:t]-=mass[t,:t]
        pending[t]=births[t]
        total[t]=pending.sum()
    difference=float(np.max(abs(np.diff(total,prepend=0)-births+mass.sum(1))))
    if difference>1e-10 or (mass<0).any() or (np.triu(mass)!=0).any():
        raise ArithmeticError("Question lifetime probabilities violated source or time conservation")
    return dict(hazard=hz,event=mass,stock=total,birth=births),difference


def run(archive,old,dest):
    if dest.exists():
        raise FileExistsError(dest)
    all_stories=pages(archive,old)
    output={};models={};info={}
    for held,p in all_stories.items():
        train=[v for k,v in all_stories.items() if k!=held]
        teachers={s.story.name:learn_teacher([v for v in train if v is not s]) for s in train}
        final_teacher=learn_teacher(train)
        outputs={}
        for arm in ARMS:
            xs=[pair_design(s,arm,teachers[s.story.name] if arm=="wording" else None) for s in train]
            hazard=hazards(train,xs)
            source=births(train,arm)
            xx=pair_design(p,arm,final_teacher if arm=="wording" else None)
            results,err=propagate(p,hazard,source,xx,arm)
            models[held+"."+arm]=dict(hazard=hazard,birth=source,conservation_max_abs=err)
            for k,v in results.items():
                output[held+"."+arm+"."+k]=v
        info[held]=dict(source=[s.story.name for s in train],excluded_question_wordings=held,
                        teacher_shape=[list(m[-1].shape) for m in final_teacher])
    dest.mkdir(parents=True)
    np.savez_compressed(dest/"predictions.npz",**output)
    (dest/"models.json").write_text(json.dumps(models,allow_nan=False)+"\n")
    seal=dict(archive=sha(archive/"receipt.json"),old=sha(old/"receipt.json"),
        protocol=sha(Path(__file__).with_name("narrative-survival-protocol.md")),
        estimator=sha(Path(__file__).with_name("narrative-survival-estimator.md")),
        geometry={k:geometry(p.story) for k,p in all_stories.items()},folds=info,
        files={n:sha(dest/n) for n in ("predictions.npz","models.json")})
    (dest/"seal.json").write_text(json.dumps(seal,indent=2)+"\n")
    print("Sealed probabilistic opening, survival and source-specific closure on",len(info),"stories")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("archive",type=Path);p.add_argument("old",type=Path)
    p.add_argument("--dest",type=Path,default=OUT/"forecast")
    run(**vars(p.parse_args()))
