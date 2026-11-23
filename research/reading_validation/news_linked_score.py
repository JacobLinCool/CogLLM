"""Evaluate word and earlier source jointly, including reviewed passages with no mark."""

from collections import defaultdict
import json

import numpy as np

from research.core_features.readback_score import average_precision
from research.reading_validation.memory_data import sha
from research.reading_validation.news_linked_data import OUT
from research.reading_validation.news_linked_fit import load
from research.reading_validation.retrieval_source_score import summarize


METHODS=("lexical.point","lexical.temporal","semantic.point","semantic.temporal","historical_external")
ACTIVE="semantic.temporal"


def score_one(doc,p):
    units=doc["units"]
    counts=doc["counts"][:,0].copy()
    keep=(units>=2)&(counts[:,1]>0)
    pos=counts[:,0];neg=counts[:,1]-pos
    zero=-(neg[keep]@np.log(p[keep,0].clip(1e-12,1)))
    positive=0.;gold=np.zeros_like(p,dtype=bool)
    for i,observed in doc["targets"].items():
        if keep[i]:
            for allowed,num in observed:
                columns=np.array(allowed)+1
                positive-=num*np.log(np.clip(p[i,columns].sum(),1e-12,1))
                gold[i,columns]=True
    denom=int(counts[keep,1].sum());n=float(pos[keep].sum());m=denom-n
    if not denom:
        return None
    allowed=(np.arange(p.shape[1]-1)[None,:]<units[:,None])&keep[:,None]
    q=p[:,1:][allowed];truth=gold[:,1:][allowed]
    mark=(1-p[keep,0]);marks=pos[keep]>0
    values={"joint_nll":(zero+positive)/denom}
    if m:
        values["unmarked_nll"]=zero/m
    if n:
        values["marked_joint_nll"]=positive/n
        values["all_candidates_top1"]=float(truth[q==q.max()].mean())
        values["all_candidates_ap"]=average_precision(q,truth)
        values["random_expected_top1"]=float(truth.mean())
    if marks.any() and not marks.all():
        values["word_ap"]=average_precision(mark,marks)
    return values,dict(eligible_reader_words=denom,positive=int(n),negative=int(m),
                       eligible_words=int(keep.sum()),simultaneous_candidates=int(allowed.sum()),marked_pairs=int(truth.sum()))


def run():
    forecast=OUT/"forecast";sources=json.loads((forecast/"seal.json").read_bytes())
    if sha(forecast/"predictions.npz")!=sources["predictions"]:
        raise ValueError("Held joint forecasts differ from their lock")
    records=defaultdict(lambda:defaultdict(list));inventory=defaultdict(int)
    weighted=defaultdict(lambda:defaultdict(float))
    with np.load(forecast/"predictions.npz",allow_pickle=False) as pred:
        for doc in load("test"):
            results={}
            for m in METHODS:
                observed=score_one(doc,pred[doc["id"]+"."+m])
                if observed is not None:
                    values,stats=observed
                    results[m]=values
            if not results:
                continue
            for j in results[METHODS[0]]:
                records[j][doc["group"]].append({m:v[j] for m,v in results.items()})
            for m,v in results.items():
                for name,den in [("joint_nll","eligible_reader_words"),("unmarked_nll","negative"),("marked_joint_nll","positive")]:
                    weighted[m][name]+=v.get(name,0)*stats[den]
            inventory["scored_documents"]+=1
            inventory["positive_documents"]+=stats["positive"]>0
            for k,v in stats.items():
                inventory[k]+=v
    result={k:summarize(v,44321,focus=ACTIVE) for k,v in records.items()}
    for stats in weighted.values():
        for name,den in [("joint_nll","eligible_reader_words"),("unmarked_nll","negative"),("marked_joint_nll","positive")]:
            stats[name]/=inventory[den]
    output=dict(seal=sha(forecast/"seal.json"),inventory=inventory,group_equal=result,pooled_likelihood=weighted)
    (OUT/"scores.json").write_text(json.dumps(output,indent=2,allow_nan=False)+"\n")
    print(dict(inventory),{k:{m:round(v["estimate"],4) for m,v in b["methods"].items()} for k,b in result.items()})


if __name__=="__main__":
    run()
