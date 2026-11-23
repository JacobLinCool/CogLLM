"""Fit each narrative outcome on the other three complete stories."""

from collections import Counter,defaultdict
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from research.reading_validation.memory_data import sha
from research.reading_validation.narrative_course_data import OUT
from research.reading_validation.narrative_course_features import load


def equal_weights(groups):
    n=Counter(groups)
    return np.array([1/(n[g]*len(n)) for g in groups])


def stats(x,w):
    m=np.average(x,axis=0,weights=w)
    sd=np.sqrt(np.average((x-m)**2,axis=0,weights=w))
    return m,np.where(sd>1e-8,sd,1.)


def roles(story):
    options={t:story.pair_matrix(t) for _,t,_ in story.events}
    splits=defaultdict(list)
    for row in story.comparisons:
        team,t,k,_,_=row
        splits[team,t,k].append(row)
    team_levels=defaultdict(set);line_levels=defaultdict(set)
    for team,t,k in splits:
        team_levels[team].add(t);line_levels[team,t].add(k)
    all_rows=[];weights=[]
    for (team,t,k),pairs in splits.items():
        weight=1/(len(team_levels)*len(team_levels[team])*len(line_levels[team,t])*len(pairs))
        for _,_,_,positive,negative in pairs:
            all_rows.append({m:options[t][m][positive]-options[t][m][negative] for m in ("form","semantic")})
            weights.append(weight)
    if not np.isclose(sum(weights),1):
        raise ValueError("No pair of distinguishable active questions in source story")
    return all_rows,weights


def rank(train,name):
    differences=[];weights=[]
    for story in train:
        pairs,w=roles(story)
        differences.extend([r[name] for r in pairs]);weights.extend(w)
    x=np.stack(differences);w=np.asarray(weights)/len(train)
    # Ranking has no intercept. Scale each difference by its source root mean square.
    sd=np.sqrt(w@(x*x));sd[sd<1e-8]=1
    z=x/sd
    def objective(v):
        y=z@v
        return w@np.logaddexp(0,-y)+.01*(v@v),-(w*expit(-y))@z+.02*v
    fit=minimize(objective,np.zeros(len(sd)),jac=True,method="L-BFGS-B",options={"gtol":1e-9,"ftol":1e-14})
    if not fit.success or max(abs(fit.jac))>1e-5:
        raise ArithmeticError("Unconverged source-line competition model")
    return fit.x/sd


def mean_rating(raw):
    count=np.isfinite(raw).sum(0)
    return np.divide(np.nansum(raw,axis=0),count,out=np.full(raw.shape[1],np.nan),where=count>0)


def suspense(train,name):
    data=[]
    for s in train:
        y=mean_rating(s.ratings)
        valid=s.word_ok&np.isfinite(y)
        x=s.feature_blocks[name]
        xline=s.line_mean(x,valid);yline=s.line_mean(y,valid)
        dx=np.diff(xline,axis=0);dy=np.diff(yline)
        paired=np.isfinite(dy)&np.isfinite(dx).all(1)
        data.append((x[valid],y[valid],dx[paired],dy[paired]))
    sizes=[len(r[0]) for r in data]
    w=np.concatenate([np.full(n,1/(len(sizes)*n)) for n in sizes])
    x=np.vstack([r[0] for r in data]);y=np.concatenate([r[1] for r in data]);mu,sd=stats(x,w)
    dx=np.vstack([r[2] for r in data]);dy=np.concatenate([r[3] for r in data])
    wd=np.concatenate([np.full(len(r[3]),1/(len(data)*len(r[3]))) for r in data])
    vx=float(w@((y-w@y)**2));vd=float(wd@((dy-wd@dy)**2))
    if min(vx,vd)<=0:
        raise ValueError("The source target varies on neither the level scale nor the change scale")
    a=np.column_stack((np.ones(len(x)),(x-mu)/sd));d=np.column_stack((np.zeros(len(dx)),dx/sd))
    penalty=np.eye(a.shape[1])*.1;penalty[0,0]=0
    lhs=(a.T*(w/vx))@a+(d.T*(wd/vd))@d+penalty
    rhs=a.T@(w*y/vx)+d.T@(wd*dy/vd)
    coef=np.linalg.solve(lhs,rhs)
    if not np.allclose(lhs@coef,rhs,atol=1e-8):
        raise ArithmeticError("Unsatisfied source suspense normal equations")
    return mu,sd,coef


def count_matrix(story,name):
    if name=="constant":
        return np.empty((story.n,0))
    if name=="original_calibrated":
        return story.trace[story.last,:1]
    return story.line_inputs[name]


def counts(train,name):
    xs=[];ys=[];weights=[]
    for s in train:
        x=count_matrix(s,name)
        for state in s.teams.values():
            xs.append(x);ys.append(np.column_stack(((state==1).sum(1),(state!=0).sum(1))))
            weights.extend([1/(len(train)*len(s.teams)*s.n)]*s.n)
    x=np.vstack(xs);y=np.vstack(ys);w=np.asarray(weights)
    mu,sd=stats(x,w);z=np.column_stack((np.ones(len(x)),(x-mu)/sd))
    def objective(v):
        p=z@v
        r=p-np.clip(p,y[:,0],y[:,1])
        grad=z.T@(w*r);grad[1:]+=.1*v[1:]
        return .5*w@(r*r)+.05*(v[1:]@v[1:]),grad
    fit=minimize(objective,np.zeros(z.shape[1]),jac=True,method="L-BFGS-B",
                 options={"gtol":1e-8,"ftol":1e-14,"maxiter":800})
    if not fit.success or max(abs(fit.jac))>3e-5:
        raise ArithmeticError(f"Unconverged interval counts: {fit.message}, {max(abs(fit.jac))}")
    return mu,sd,fit.x


def score(model,x):
    m,s,v=model
    return ((x-m)/s)@v[1:]+v[0]


def run(archive,dest):
    if dest.exists():
        raise FileExistsError(dest)
    all_stories=load(archive)
    forecasts={};convergence={}
    for key,s in all_stories.items():
        train=[v for k,v in all_stories.items() if k!=key]
        # All t candidate scores are estimated without seeing the evaluation-story question markings.
        coefficients={m:rank(train,m) for m in ("form","semantic")}
        matrices={m:np.full((s.n,s.n),np.nan) for m in ("distance","original","form","semantic")}
        for t in range(1,s.n):
            f=s.pair_matrix(t)
            for m,v in matrices.items():
                v[t,:t]=f[m]@coefficients[m] if m in coefficients else f[m]
        for m,v in matrices.items():
            forecasts[f"{key}.pairs.{m}"]=v
        forecasts[f"{key}.tension.original"]=s.trace[s.word_ends.clip(min=0),0]
        forecasts[f"{key}.tension.word_index"]=np.log1p(np.arange(len(s.words)))
        for m in ("progress","form","semantic"):
            fit=suspense(train,m)
            forecasts[f"{key}.tension.{m}"]=score(fit,s.feature_blocks[m])
        forecasts[f"{key}.count.original"]=s.trace[s.last,0]
        for m in ("constant","original_calibrated","progress","form","semantic"):
            fit=counts(train,m)
            forecasts[f"{key}.count.{m}"]=np.maximum(0,score(fit,count_matrix(s,m)))
        convergence[key]=dict(source=[v.name for v in train],evaluated_current_lines=s.n,
                              earlier_candidates=s.n*(s.n-1)//2)
    dest.mkdir(parents=True)
    np.savez_compressed(dest/"predictions.npz",**forecasts)
    result=dict(native=sha(archive/"receipt.json"),inputs=sha(OUT/"input-seal.json"),fits=convergence,
                estimator=sha(Path(__file__).with_name("narrative_course_estimator.md")),
                predictions=sha(dest/"predictions.npz"))
    (dest/"seal.json").write_text(json.dumps(result,indent=2)+"\n")
    print("Sealed predictions for four source-held-out stories")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("archive",type=Path);p.add_argument("--dest",type=Path,default=OUT/"forecast")
    a=p.parse_args();run(a.archive,a.dest)
