"""Fit reader-response likelihoods on SEAHORSE train and validation, then lock predictions for ungraded test inputs."""
from collections import Counter
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.feature_extraction.text import TfidfVectorizer

from research.reading_validation.memory_data import h
from research.reading_validation.summeval_ballots import forecast
from research.reading_validation.transition_seahorse_data import DEST, PLAN, ROLES, WORK, sha, source_rows
from research.reading_validation.transition_seahorse_features import load, save

LABEL = {"Yes": 0., "No": 1.}
NEWS = Path("research/results/reading-validation-summeval")
SOURCE_STUDY = Path(__file__).with_name("transition-seahorse-text-control.md")


def corpus(story):
    return story.split("-",1)[0].split("_",1)[0]


def key(row):
    return h(json.dumps([row["gem_id"], row["model"], row["summary"]], ensure_ascii=False))


def weights(group):
    _, ids, counts = np.unique(group,return_inverse=True,return_counts=True)
    w = 1/counts[ids]
    return w/w.sum()


def solve(x, y, w, lam, *, positive=False):
    p = x.shape[1]
    initial = np.zeros(p+1)
    initial[0] = np.log(w@y/(1-w@y))
    def criterion(theta):
        z = np.asarray(x@theta[1:]).ravel()+theta[0]
        d = w*(expit(z)-y)
        loss = w@(np.logaddexp(0,z)-y*z)+.5*lam*(theta[1:]@theta[1:])
        return float(loss), np.r_[d.sum(), np.asarray(x.T@d).ravel()+lam*theta[1:]]
    bounds = [(None,None),*([(0,None)]*p)] if positive else None
    opt = minimize(criterion,initial,jac=True,method="L-BFGS-B",bounds=bounds,
                   options={"maxiter":1000,"gtol":1e-8,"ftol":1e-14,"maxls":50})
    grad = opt.jac.copy()
    if positive:
        grad[1:][(opt.x[1:]<1e-9)&(grad[1:]>0)]=0
    residual=float(np.max(abs(grad)))
    if residual>2e-6 or not np.isfinite(opt.fun) or opt.fun>criterion(initial)[0]+1e-8:
        raise ArithmeticError(f"Source optimization did not reach its objective: {opt.message}, gradient {residual}")
    return opt.x, {"initial":criterion(initial)[0], "converged":float(opt.fun),
                   "projected_max_gradient":residual, "iterations":int(opt.nit),"penalty":lam}


def linear(x,theta):
    return np.asarray(x@theta[1:]).ravel()+theta[0]


def threshold(y,p):
    # Every tied score shares one decision. On equal F1 choose the higher threshold.
    ix=np.argsort(-p,kind="stable"); scores=p[ix]; truth=y[ix]
    end=np.r_[scores[:-1]!=scores[1:],True]
    locations=np.flatnonzero(end)+1
    f1=2*np.cumsum(truth)[end]/(locations+truth.sum())
    return float(scores[locations[np.argmax(f1)]-1])


def fit():
    exposure=load(DEST/"exposure-seal.json")
    if exposure["protocol"]!=sha(PLAN):
        raise ValueError("The outcome-blind protocol changed")
    table={}
    for role in exposure["packets"]:
        f=WORK/f"seahorse-{role}-document-features.npz"
        seal=load(DEST/(role+"-feature-seal.json"))
        if sha(f)!=seal["file"]:raise ValueError("Changed native source-feature shard")
        with np.load(f,allow_pickle=False) as a:
            for k,v in zip(a["keys"],a["x"],strict=True):
                if k in table:raise ValueError("Source text encoded twice")
                table[k]=v
    meta=[]; xx=[];y=[]; roles=[]; counts={}
    exposure_audit=load(DEST/"word-exposure-audit.json")
    for role in ("train","validation","test"):
        rejected=set(exposure_audit[role]["exclude_overlap_with_held_roles"])
        details=Counter()
        for r in source_rows(role,grades=role!="test"):
            details["original"]+=1
            if r["gem_id"] in rejected:
                details["excluded_overlap"]+=1
                continue
            k = h(r["summary"])
            if k not in table:
                details["excluded_without_native_coverage"]+=1
                continue
            if role!="test" and r["question1"] not in LABEL:
                details["excluded_indeterminate"]+=1
                continue
            meta.append(r); xx.append(table[k]);roles.append(role)
            y.append(LABEL[r["question1"]] if role!="test" else np.nan)
            details["fit_or_forecast"]+=1
        counts[role]=dict(details)
    roles=np.array(roles); y=np.array(y); x=np.asarray(xx); groups=np.array([r["gem_id"] for r in meta]); text=[r["summary"] for r in meta]
    fit=roles=="train"; dev=roles=="validation"; test=roles=="test";w=weights(groups[fit]);v=weights(groups[dev])
    keys=np.array([key(r) for r in meta])
    if len(np.unique(keys[test]))!=test.sum():raise ValueError("Test row key collision")
    kinds=np.array([corpus(r["gem_id"])+"/"+r["model"] for r in meta])
    cats=sorted(set(kinds[fit]))
    metadata=(kinds[:,None]==np.array(cats)[None,:]).astype(float)
    news=NEWS/"ballot-weights.npz"
    if sha(news)!=load(NEWS/"ballot-forecast-seal.json")["weights_sha256"]:raise ValueError("Earlier frozen news weights changed")
    uncal={};params={};diag={};final={};decisions={}
    with np.load(news,allow_pickle=False) as f:
        for k,z in (("form",x[:,:15]),("native",x)):
            template={a:f[k+"."+a] for a in ("center","scale","slope")}
            uncal["frozen_"+k]=-forecast(template,z)
    for name,basis in (("form",x[:,:15]),("native",x),("metadata",metadata),
                       ("metadata_form",np.column_stack([metadata,x[:,:15]])),
                       ("metadata_native",np.column_stack([metadata,x]))):
        center=w@basis[fit]; std=(w@((basis[fit]-center)**2))**.5
        std=np.where(std>1e-7,std,1.)
        z=(basis-center)/std
        theta,diag[name]=solve(z[fit],y[fit],w,.002)
        uncal[name]=linear(z,theta)
        params.update({name+".mean":center,name+".scale":std,name+".theta":theta})
    vec=TfidfVectorizer(analyzer="char",ngram_range=(3,5),min_df=3,max_features=30000,
                        lowercase=False,sublinear_tf=True,dtype=np.float64)
    gram=vec.fit([s for i,s in enumerate(text) if fit[i]]).transform(text)
    theta,diag["character"]=solve(gram[fit],y[fit],w,1/(10*fit.sum()))
    uncal["character"]=linear(gram,theta)
    params.update({"character.theta":theta,"character.idf":vec.idf_,
                   "character.vocab":vec.get_feature_names_out().astype(str)})
    for name,columns in {**{k:[k] for k in uncal}, "character_form":["character","form"],
                         "character_native":["character","native"]}.items():
        m=np.column_stack([uncal[c] for c in columns])
        t,diag[name+"_val"]=solve(m[dev],y[dev],v,1e-8,positive=True)
        pred=expit(linear(m,t)); final[name]=pred[test]
        decisions[name]=threshold(y[dev],pred[dev])
        params[name+".calibration"]=t
    for k in ("frozen_native","frozen_form"):
        final[k+"_raw"]=uncal[k][test]
    arr=DEST/"forecasts-before-held-responses.npz"
    unit=DEST/"supervised-weights.npz"
    np.savez_compressed(arr,keys=keys[test],**final)
    np.savez_compressed(unit,metadata_categories=np.array(cats),**params)
    notes={"test_Q1_read":False,"source":exposure["source"],"protocol":sha(PLAN),"extra_text_control":sha(SOURCE_STUDY),
           "forecast_sha256":sha(arr),"weights_sha256":sha(unit),"earlier_frozen_weights":sha(news),"optimizer":diag,
           "counts":counts,"thresholds":decisions,"train_and_dev_no_counts":{r:int(y[roles==r].sum()) for r in ("train","validation")},
           "sampling_unit":"article-equal fitting; per-draft proper loss and article-macro directional concordance",
           "threshold_selection":"maximum per-response F1 on validation; largest threshold among ties","char_vocab":len(vec.vocabulary_)}
    save(DEST/"fit-seal.json",notes)
    print("Prediction lock",counts,len(final["native"]),max(d["projected_max_gradient"] for d in diag.values()),flush=True)


if __name__=="__main__":
    fit()
