"""Cross-topic transfer of direct and reason-specific annotation likelihoods in DECOR."""
from argparse import ArgumentParser
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from research.reading_validation.transition_ambicoref_score import rank_order,ranking
from research.reading_validation.transition_external_fit import read_features
from research.reading_validation.transition_external_prepare import OUT,csv_rows
from research.reading_validation.transition_score import dump,sha
from research.reading_validation.transition_writer_f1 import threshold


PROTOCOL=Path("research/reading_validation/transition-essay-reasons-protocol.md")
DEST=OUT/"essay-cross-topic"
METRICS=("logloss","brier","auroc","ap","precision","recall","f1","false_alarms_per_100_unmarked","signed_overprediction")


def mass(topic,essay):
    w=np.zeros(len(topic))
    for t in np.unique(topic):
        idx=np.flatnonzero(topic==t);names,counts=np.unique(essay[idx],return_counts=True)
        for n,num in zip(names,counts,strict=True):
            w[idx[essay[idx]==n]]=1/(len(np.unique(topic))*len(names)*num)
    if not np.isclose(w.sum(),1):raise ValueError("Unequal source topic or essay weights")
    return w


def fit(F,y,R,w,multi):
    mean=w@F;sd=np.maximum(np.sqrt(w@np.square(F-mean)),1e-9);x=np.column_stack([np.ones(len(F)),(F-mean)/sd])
    k=7 if multi else 1;D=x.shape[1];initial=np.zeros((D,k));mask=(R>=0).all(1) if multi else np.ones(len(y),bool)
    event=R if multi else y[:,None];events=event[mask]
    def objective(vec):
        W=vec.reshape(D,k);eta=x@W;p=expit(eta)
        ll=np.empty(len(y));grad=np.empty_like(p);ll[mask]=(np.logaddexp(0.,eta[mask])-events*eta[mask]).sum(1)
        grad[mask]=p[mask]-events
        if (~mask).any():
            code=~mask;log_clean=-np.logaddexp(0.,eta[code]).sum(1);any_r=-np.expm1(log_clean)
            # The location was judged incoherent without a reliable assigned cause.
            if not (y[code]==1).all() or not (any_r>0).all():raise ArithmeticError("Unresolved marked-row likelihood")
            ll[code]=-np.log(any_r);grad[code]=-p[code]*(np.exp(log_clean)/any_r)[:,None]
        reg=W.copy();reg[0]=0
        return float(w@ll+.005*(reg*reg).sum()),(x.T@(w[:,None]*grad)+.01*reg).ravel()
    res=minimize(objective,initial.ravel(),method="L-BFGS-B",jac=True,options={"maxiter":500,"ftol":1e-14,"gtol":3e-8,"maxls":50})
    err=float(abs(res.jac).max())
    if not np.isfinite(res.fun) or err>1e-6:raise ArithmeticError(f"Essay training insufficiently converged: {err}")
    return (mean,sd,res.x.reshape(D,k)),{"max_gradient":err,"iterations":int(res.nit),"loss":float(res.fun),"parameters":int(D*k)}


def infer(model,F):
    mean,sd,W=model;x=np.column_stack((np.ones(len(F)),(F-mean)/sd));eta=x@W
    return -np.expm1(-np.logaddexp(0.,eta).sum(1)),expit(eta)


def evaluate(y,options,w):
    out={}
    for m,(p,flags) in options.items():
        if not ((p>0)&(p<1)).all():raise ValueError("Unusable calibrated probability")
        a=w*y;b=w*(1-y);den=w.sum(1);tp=a@flags;fp=b@flags;n=a.sum(1)
        au,ap=ranking(*rank_order(p),a,b)
        prec=np.divide(tp,tp+fp,out=np.zeros_like(tp),where=tp+fp>0);rec=tp/n
        out[m]=np.column_stack(((-a@np.log(p)-b@np.log1p(-p))/den,(a@(1-p)**2+b@p**2)/den,au,ap,prec,rec,
                                2*tp/(tp+fp+n),100*fp/b.sum(1),w@(p-y)/den))
    return out


def main(folder):
    records,_,F,arms=read_features(folder);csv=csv_rows("test_1355_clean.csv")
    past=pd.read_csv(OUT/"heldout-scored.csv").query("source == 'decor'").set_index("id")
    if not past.index.is_unique:
        raise ValueError("Multiple archived forecasts appeared for one essay stop")
    src=[i for i,r in enumerate(records) if r["source"]=="decor"]
    rows=[records[i] for i in src];X=F[src];Y=[];R=[];topics=[];essays=[]
    for row in rows:
        d=csv[row["source_ref"][1]];v=int(d["label"]=="0.0")
        if v!=past.loc[row["id"],"y"]:raise ValueError("Sign of the reported coherence decision differs")
        Y.append(v);topics.append(d["topic"]);essays.append(d["essay_id"])
        R.append([0 if not v else int(float(d["R"+str(j)])) for j in range(1,8)])
    y=np.array(Y);r=np.array(R);topics=np.array(topics);essays=np.array(essays)
    if len(y)!=1352 or y.sum()!=446 or np.sum((r<0).any(1))!=1:raise ValueError("Unexpected expert annotation support")
    topics_order=sorted(set(topics),key=lambda x:sha256(("cogllm-essay-reason-v1"+x).encode()).hexdigest())
    methods=("lexical_surprisal","essay_binary","essay_reasons")
    scores={m:np.zeros(len(y)) for m in methods};scores["story_frozen"]=past.loc[[row["id"] for row in rows],"integration"].to_numpy()
    flags={m:np.zeros(len(y),bool) for m in scores};cause=np.zeros((len(y),7));folds=[]
    for i,t in enumerate(topics_order):
        d=topics_order[(i+1)%len(topics_order)];test=topics==t;dev=topics==d;train=~(test|dev)
        if (set(essays[train])&set(essays[dev]) or set(essays[train])&set(essays[test]) or set(essays[dev])&set(essays[test])):
            raise ValueError("One essay occurred on two different topics")
        w=mass(topics[train],essays[train]); fits={};cuts={}
        for m in methods:
            x=X[:,arms["lexical_surprisal"] if m=="lexical_surprisal" else arms["integration"]]
            model,stats=fit(x[train],y[train],r[train],w,m=="essay_reasons");p,_=infer(model,x[dev]);cut=threshold(p,y[dev])["cutoff"]
            p,z=infer(model,x[test]);scores[m][test]=p;flags[m][test]=p>=cut;fits[m]=stats;cuts[m]=cut
            if m=="essay_reasons":cause[test]=z
        # The original trained story threshold remains a frozen external benchmark.
        cut=json.loads((OUT/"readouts.json").read_text())["thresholds"]["integration"]
        flags["story_frozen"][test]=scores["story_frozen"][test]>cut;cuts["story_frozen"]=cut
        folds.append({"topic":t,"calibration_topic":d,"train_essays":len(set(essays[train])),"test_essays":len(set(essays[test])),
                      "test_sentences":int(test.sum()),"source_fits":fits,"booked_cutoffs":cuts})
    DEST.mkdir(exist_ok=True,parents=True);forecast=DEST/"topic-disjoint-forecasts.npz";np.savez_compressed(forecast,**scores,fitted_reason_risks=cause)
    options={k:(v,flags[k]) for k,v in scores.items()};_,tp=np.unique(topics,return_inverse=True);_,es=np.unique(essays,return_inverse=True)
    rng=np.random.default_rng(31219);unit=np.ones((1,len(y)));point=evaluate(y,options,unit);boot={m:[] for m in scores}
    for _ in range(80):
        # Crossed essays are nested inside topic. Products preserve each original essay's sentence rows.
        w=rng.exponential(size=(50,len(set(tp))))[:,tp]*rng.exponential(size=(50,len(set(es))))[:,es]
        m=evaluate(y,options,w)
        for k,b in m.items():boot[k].append(b)
    boot={k:np.concatenate(v) for k,v in boot.items()}
    def describe(pt,reps):
        return {k:[float(a),*np.quantile(b,[.025,.975]).tolist()] for k,a,b in zip(METRICS,pt[0],reps.T,strict=True)}
    summary={m:describe(point[m],boot[m]) for m in scores}
    contrasts={m:describe(point["essay_reasons"]-point[m],boot["essay_reasons"]-boot[m]) for m in scores if m!="essay_reasons"}
    sub={t:{m:dict(zip(METRICS,v[0].tolist(),strict=True)) for m,v in evaluate(y[topics==t],{m:(p[topics==t],f[topics==t]) for m,(p,f) in options.items()},np.ones((1,(topics==t).sum()))).items()} for t in topics_order}
    within={}
    for m,(p,f) in options.items():
        vals=[]
        for name in sorted(set(essays)):
            ix=essays==name;val=-(y[ix]*np.log(p[ix])+(1-y[ix])*np.log1p(-p[ix])).mean()
            vals.append((topics[ix][0],val))
        mean=np.array([np.mean([v for k,v in vals if k==t]) for t in topics_order])
        within[m]={"equal_topic_essay_logloss":float(mean.mean()),"per_topic":mean.tolist()}
    types={}
    for i in range(7):
        valid=(r>=0).all(1);response=r[valid,i].astype(float)
        if len(np.unique(response))!=2:continue
        pr=cause[valid,i];au,ap=ranking(*rank_order(pr),response[None,:],(1-response)[None,:])
        types["R"+str(i+1)]={"judged_sentences":len(pr),"events":int(response.sum()),"auroc":float(au[0]),"ap":float(ap[0])}
    dump(DEST/"scores.json",{"protocol_sha256":sha(PROTOCOL),"forecasts_sha256":sha(forecast),"native_receipt_sha256":sha(folder/"receipt.json"),
         "topic_count":len(topics_order),"essay_count":len(set(essays)),"sentence_count":len(y),"human_marked":int(y.sum()),
         "folds":folds,"metrics":summary,"reason_model_minus_baseline":contrasts,"by_topic":sub,"equal_topic_essay":within,"reason_localization":types})
    print({m:[v[k][0] for k in ("logloss","auroc","ap","f1")] for m,v in summary.items()})


if __name__=="__main__":
    cli=ArgumentParser(description=__doc__);cli.add_argument("--features",type=Path,required=True);main(cli.parse_args().features)
