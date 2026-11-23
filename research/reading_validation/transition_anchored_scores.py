"""Combine an already learned referent distribution with changes in evidence as text unfolds."""

import numpy as np


def traces(priors,evidence):
    a=np.asarray(priors,dtype=float);d=np.asarray(evidence,dtype=float)
    if a.ndim!=2 or a.shape[0]!=2 or d.ndim!=3 or d.shape[0]!=4 or a.shape[1]!=d.shape[2]:
        raise ValueError("An earlier-word prior and aligned source evidence are required")
    ret={}
    for j,name in enumerate(("lexical","native")):
        ret[name+"_prior"]=np.broadcast_to(a[j],d.shape[1:])
        for k,update in enumerate(("surface","steps","attention","semantic")):
            if k==0:continue
            delta=d[k]-d[k,0];ret[name+"_"+update+"_update"]=a[j]+delta
            if not np.array_equal(ret[name+"_"+update+"_update"][0],a[j]):raise ValueError("The readout changed its measured initial state")
    for k,name in enumerate(("source_surface","source_steps","source_attention","source_semantic")):ret[name]=d[k]
    return ret


def relation_scores(utilities,regions,mean=False):
    one,two=regions
    if not one or not two or set(one)&set(two):raise ValueError("A study requires distinct antecedent choices")
    if mean:return utilities[...,two].mean(-1)-utilities[...,one].mean(-1)
    return np.logaddexp.reduce(utilities[...,two],axis=-1)-np.logaddexp.reduce(utilities[...,one],axis=-1)


def probability(log_odds):
    return np.exp(-np.logaddexp(0.,-np.asarray(log_odds)))
