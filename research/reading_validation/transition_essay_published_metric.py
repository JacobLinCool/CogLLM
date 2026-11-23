"""Report the original paper's class-frequency-weighted F1 on source-disjoint topic predictions."""
from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np
from scipy.special import expit, logit
from research.reading_validation.essay_hazard_inference import EssayHazardInference
from research.reading_validation.transition_essay_choice_data import DEST as EXPORT
from research.reading_validation.transition_essay_reasons import DEST
from research.reading_validation.transition_essay_rewrite import material
from research.reading_validation.transition_score import dump,sha


def weighted_f1(y,flags,w):
    pos=w@y;neg=w@(1-y);tp=w@(y*flags);fp=w@((1-y)*flags);tn=neg-fp;fn=pos-tp
    return (pos*2*tp/(2*tp+fp+fn)+neg*2*tn/(2*tn+fp+fn))/(pos+neg)


def best(y,p):
    cutoffs=np.r_[np.nextafter(p.max(),np.inf),np.unique(p)]
    numbers=[weighted_f1(y,p>=cut,np.ones(len(y))) for cut in cutoffs]
    return float(cutoffs[np.flatnonzero(numbers==max(numbers))[-1]])


def main(folder):
    X,_,idx,y,t,e,*_=material(folder);X=X[idx];x=np.column_stack((np.ones(len(y)),X))
    source=EXPORT/"source-forecast-lock.json";lock=json.loads(source.read_text());coef=np.load(EXPORT/"topic-residuals.npz")["weights"]
    native,lexical=EssayHazardInference.load().probabilities(X,per_model=True)
    earlier=np.load(DEST/"expert-replacement-forecasts.npz");flags={};by={}
    for method in lock["methods"]:
        flags[method]={q:np.zeros(len(y)) for q in ("source_break_f1","source_weighted_f1")}
        flags[method]["source_break_f1"]=earlier[method+".flag"];j=lock["methods"].index(method)
        for i,topic in enumerate(lock["topics"]):
            later=lock["topics"][(i+1)%len(lock["topics"])];dev=t==later;held=t==topic
            p=(native if method.startswith("native") else lexical)[:,i]
            p=expit(logit(np.clip(p,1e-10,1-1e-10))+x@coef[i,j])
            cutoff=best(y[dev],p[dev]);flags[method]["source_weighted_f1"][held]=p[held]>=cutoff
            by.setdefault(method,{})[topic]=cutoff
    rng=np.random.default_rng(291131);_,ti=np.unique(t,return_inverse=True);_,ei=np.unique(e,return_inverse=True)
    w=rng.exponential(size=(4000,max(ti)+1))[:,ti]*rng.exponential(size=(4000,max(ei)+1))[:,ei]
    results={}
    for method,values in flags.items():
        results[method]={reg:[float(weighted_f1(y,a,np.ones(len(y)))),*np.quantile(weighted_f1(y,a,w),[.025,.975]).tolist()] for reg,a in values.items()}
    dump(DEST/"published-metric.json",{"source_lock_sha256":sha(source),"human_class_counts":{"coherent":int(sum(1-y)),"incoherent":int(sum(y))},
        "weighted_f1":results,"source_topic_weighted_f1_thresholds":by,"published_weighted_f1":{"GPT4_zero":.6656,"GPT4_16_shot":.6933,"DeBERTa_base_synthetic_DT":.6854,"Llama2_7B_synthetic_DT":.6608},
        "comparison":"Same 1,352 human decisions with different source training and calibration protocols. Source model scores not released for paired inference."})
    print(results["native_joint"])


if __name__=="__main__":
    p=ArgumentParser(description=__doc__);p.add_argument("--features",type=Path,required=True);main(p.parse_args().features)
