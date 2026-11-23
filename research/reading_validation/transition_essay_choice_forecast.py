"""Freeze topic-disjoint replacement and original-judgment readouts before scoring new ballots."""
from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np
from scipy.special import expit, logit

from research.reading_validation.essay_hazard_inference import EssayHazardInference, WEIGHTS
from research.reading_validation.transition_essay_choice_data import DEST, INPUT, PLAN
from research.reading_validation.transition_essay_hazard import basis
from research.reading_validation.transition_essay_reasons import DEST as SOURCE, mass
from research.reading_validation.transition_essay_rewrite import METHODS, fit, material, reliability, training_design
from research.reading_validation.transition_score import dump, sha


def export(folder):
    X, arms, direct, y, topics, essays, old, new, pt, pe, _ = material(folder)
    metadata=json.loads((SOURCE/"expert-replacement-scores.json").read_text())
    existing=np.load(SOURCE/"expert-replacement-forecasts.npz")
    order=[r["held_topic"] for r in metadata["folds"]]
    native,lexical=EssayHazardInference.load().probabilities(X,per_model=True)
    coefficients=np.zeros((len(order),len(METHODS),X.shape[1]+1));max_error=0.
    for i,topic in enumerate(order):
        other=order[(i+1)%len(order)]
        tr=(topics!=topic)&(topics!=other);edtr=(pt!=topic)&(pt!=other);dev=topics==other;eddev=pt==other
        w=mass(topics[tr],essays[tr]);nx,transform=basis(X,direct[tr],w,arms,with_transform=True)
        cols=np.asarray(arms["lexical_surprisal"]);data=X[:,cols]
        mu=w@data[direct[tr]];sd=np.sqrt(w@(data[direct[tr]]-mu)**2);valid=sd>1e-8
        scale=sd[valid]*np.sqrt(valid.sum())
        lx=np.column_stack((np.ones(len(X)),(data[:,valid]-mu[valid])/scale))
        lt=np.zeros((X.shape[1]+1,lx.shape[1]));lt[0,0]=1
        lt[0,1:]=-mu[valid]/scale;lt[cols[valid]+1,1+np.arange(valid.sum())]=1/scale
        for j,m in enumerate(METHODS):
            kind,method=m.split("_");p=(native if kind=="native" else lexical)[:,i]
            begin=logit(np.clip(p,1e-10,1-1e-10));z=nx if kind=="native" else lx
            if method!="frozen":
                def design(t,ed):return training_design(z,begin,direct,y,topics,essays,old,new,pt,pe,t,ed,method)
                theta,_=fit(design(tr,edtr));alpha=reliability(design(dev,eddev),theta)
                coeff=(transform if kind=="native" else lt)@theta
                coefficients[i,j]=coeff*alpha
                if abs(alpha-metadata["folds"][i]["fits"][m]["source_strength"])>1e-8:raise ValueError("Source strength differed")
            prediction=np.clip(expit(begin+coefficients[i,j,0]+X@coefficients[i,j,1:]),1e-12,1-1e-12)
            for key,sel,idcs in (("risk",topics==topic,direct),("old",pt==topic,old),("new",pt==topic,new)):
                max_error=max(max_error,float(np.max(abs(existing[m+"."+key][sel]-prediction[idcs[sel]]))))
    if max_error>1e-7:raise ArithmeticError("Unmatched activated source fitting pipeline")
    weights=DEST/"topic-residuals.npz";np.savez_compressed(weights,weights=coefficients)
    dump(DEST/"source-forecast-lock.json",{"protocol_sha256":sha(PLAN),"existing_source_score_sha256":sha(SOURCE/"expert-replacement-scores.json"),
        "original_source_weights_sha256":sha(WEIGHTS),"residual_sha256":sha(weights),"methods":METHODS,"topics":order,"original_source_max_error":max_error})
    print("Sources fixed",max_error,sha(weights),flush=True)


def forecast(folder):
    ref=json.loads((DEST/"source-forecast-lock.json").read_text());exposure=json.loads((DEST/"exposure-seal.json").read_text())
    receipt=json.loads((folder/"receipt.json").read_text());f=folder/"sentence-features.jsonl";coeff=DEST/"topic-residuals.npz"
    if (receipt["input_sha256"]!=exposure["packet_sha256"] or receipt["input_sha256"]!=sha(INPUT) or
        receipt["file_sha256"]!=sha(f) or receipt["documents"]!=receipt["forwards"] or receipt["documents"]!=348 or
        receipt["external_outcomes"] or sha(coeff)!=ref["residual_sha256"] or ref["protocol_sha256"]!=sha(PLAN)):
        raise ValueError("The expert-choice held passage exposure or forecast identity changed")
    rows=[json.loads(s) for s in f.read_text().splitlines()]
    matrix=np.stack([np.r_[np.log1p(r["position"]),r["features"],r["representation"]] for r in rows])
    ids=[r["id"] for r in rows];terms=np.load(coeff)["weights"];by_id={}
    for item in exposure["items"].values():
        k=ref["topics"].index(item["topic"])
        for v in item["sources"].values():
            for alt in v["variants"]:
                if by_id.setdefault(alt,k)!=k:raise ValueError("A held alternative is associated with two topics")
    if set(ids)!=set(by_id) or len(ids)!=len(set(ids)):raise ValueError("Missing original expert alternative")
    folds=np.array([by_id[k] for k in ids]);nn,ll=EssayHazardInference.load().probabilities(matrix,per_model=True)
    M=np.column_stack((np.ones(len(matrix)),matrix))
    prediction={}
    for j,k in enumerate(ref["methods"]):
        p=(nn if k.startswith("native") else ll)[np.arange(len(rows)),folds]
        log_odds=logit(np.clip(p,1e-10,1-1e-10))+np.sum(M*terms[folds,j],axis=1)
        prediction[k]=np.clip(expit(log_odds),1e-12,1-1e-12)
    out=DEST/"judgment-blind-forecasts.npz";np.savez_compressed(out,ids=ids,**prediction)
    dump(DEST/"judgment-blind-forecast-lock.json",{"protocol_sha256":sha(PLAN),"source_model_sha256":sha(DEST/"source-forecast-lock.json"),
         "native_receipt":receipt,"predictions_sha256":sha(out),"candidate_count":len(ids)})
    print("Locked expert forecasts",len(ids),sha(out),flush=True)


if __name__=="__main__":
    p=ArgumentParser(description=__doc__);p.add_argument("mode",choices=["export","forecast"]);p.add_argument("folder",type=Path)
    args=p.parse_args();(export if args.mode=="export" else forecast)(args.folder)
