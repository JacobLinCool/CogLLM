"""Measure selective support attribution, source location and calibration on held-out stories."""

import argparse
from collections import defaultdict
from itertools import combinations
import json
from pathlib import Path

import numpy as np
from scipy.special import expit

from research.reading_validation.conflict_fit import classification
from research.reading_validation.event_data import OUTPUT
from research.reading_validation.event_fit import DIRECTIONS, PAIRS, SCHEMES, annotations
from research.reading_validation.joint_data import digest, read


def choice(z,left,right):
    return .5*(1+np.sign(z[left]-z[right]))


def comparisons(tensor,direction,reverse=False):
    # When holding the later (earlier) cue fixed the rivals are earlier (later) actions.
    # When holding the action fixed the competing positions are potential evidence sentences.
    by_fixed=defaultdict(list)
    for i,(a,b) in enumerate(PAIRS):
        anchor,clue=(a,b) if direction==0 else (b,a)
        by_fixed[anchor if reverse else clue].append(i)
    comparisons=defaultdict(list)
    for doc,items in enumerate(tensor):
        for candidates in by_fixed.values():
            positive=[i for i in candidates if items[i,0]>=2 and items[i,0]>items[i,1]/2]
            negative=[i for i in candidates if items[i,0]==0 and items[i,1]>=2]
            comparisons[doc].extend((a,b) for a in positive for b in negative)
    return {doc:choices for doc,choices in comparisons.items() if choices}


def paired(selection,methods):
    if not selection:
        return dict(stories=0,decisions=0)
    names=sorted(methods)
    values=[]
    for doc,items in selection.items():
        left,right=np.asarray(items).T
        values.append([choice(methods[name][doc],left,right).mean() for name in names])
    values=np.array(values)
    groups=np.random.default_rng(91734).integers(len(values),size=(10000,len(values)))
    samples=np.stack([values[:,i][groups].mean(axis=1) for i in range(len(names))],axis=1)
    def summary(v,boot):
        return dict(estimate=float(v.mean()),interval95=np.quantile(boot,[.025,.975]).tolist())
    index=names.index("context")
    return dict(stories=len(values),decisions=sum(map(len,selection.values())),
                methods={name:summary(values[:,i],samples[:,i]) for i,name in enumerate(names)},
                primary_minus={name:summary(values[:,index]-values[:,i],samples[:,index]-samples[:,i])
                               for i,name in enumerate(names) if i!=index})


def two_issues(tensor,z):
    pair_idx={p:i for i,p in enumerate(PAIRS)}
    result=[]
    for doc,items in enumerate(tensor):
        choices=[]
        for a,b in combinations(range(5),2):
            for j,k in combinations(range(b+1,5),2):
                for p,q in ((j,k),(k,j)):
                    match=[pair_idx[t] for t in ((a,p),(b,q))]
                    cross=[pair_idx[t] for t in ((b,p),(a,q))]
                    if all(items[i,0]>=2 and items[i,0]>items[i,1]/2 for i in match) and all(
                            items[i,0]==0 and items[i,1]>=2 for i in cross):
                        choices.append(float(all(z[doc,i]>z[doc,t] for i,t in zip(match,cross,strict=True))))
        if choices:
            result.append(np.mean(choices))
    return dict(stories=len(result),mean_both_correct=float(np.mean(result)) if result else None)


def run(folder,output):
    seal=read(folder/"seal.json")
    if any(digest(folder/k)!=v for k,v in seal["files"].items()):
        raise ValueError("Changed event forecasts")
    with np.load(folder/"forecasts.npz",allow_pickle=False) as bank:
        table={k:bank[k] for k in bank.files}
    raw=annotations("test",table["id"])
    result={}
    for scheme in SCHEMES:
        modes={}
        for d,direction in enumerate(DIRECTIONS):
            tensor=raw[scheme][:,:,d,:]
            methods={k.split(".")[-1]:v for k,v in table.items() if k.startswith(scheme+"."+direction+".")}
            all_scores=methods|dict(overlap=table["overlap"],distance=table["distance"])
            distribution={}
            ok=tensor[:,:,1]>0
            hit,total=tensor[ok].T
            total_votes=total.sum()
            for k,z in methods.items():
                logits=z[ok]
                nll=(total@np.logaddexp(0,logits)-hit@logits)/total_votes
                # Treat each aggregated vote as a distinct Bernoulli judgment for descriptive PR.
                vote_y=np.r_[np.ones(hit.sum()),np.zeros(total_votes-hit.sum())]
                vote_s=np.concatenate([np.repeat(logits,hit),np.repeat(logits,total-hit)])
                scores=classification(vote_y,vote_s)
                distribution[k]=dict(nll=float(nll),mean_probability=float(np.average(expit(logits),weights=total)),
                                     ap=scores["ap"],auc=scores["auc"])
            endpoint=dict(observed_pairs=int(ok.sum()),votes=int(total_votes),positive_votes=int(hit.sum()),
                          observed_rate=float(hit.sum()/total_votes),distribution=distribution,
                          which_action_for_fixed_clue=paired(comparisons(tensor,d),all_scores),
                          which_clue_for_fixed_action=paired(comparisons(tensor,d,True),all_scores))
            if d==0:
                endpoint["simultaneous_selectivity"]=two_issues(tensor,methods["context"])
            modes[direction]=endpoint
        result[scheme]=modes
    doc=dict(coordinate_interpretations=result,n_heldout=len(table["id"]),forecast_seal_sha256=digest(folder/"seal.json"))
    output.write_text(json.dumps(doc,indent=2,allow_nan=False)+"\n")
    print(json.dumps({s:v[DIRECTIONS[0]]["which_action_for_fixed_clue"] for s,v in result.items()}))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--forecast",type=Path,default=OUTPUT/"forecast")
    p.add_argument("--output",type=Path,default=OUTPUT/"scores.json")
    a=p.parse_args()
    run(a.forecast,a.output)
