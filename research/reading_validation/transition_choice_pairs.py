"""Check whether two directional AmbiCoref responses select different referents."""
from argparse import ArgumentParser
from collections import Counter,defaultdict
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from research.reading_validation.transition_ambicoref_prepare import OUT, ROOT
from research.reading_validation.transition_ambicoref_score import PRIMARY,rank_order,ranking
from research.reading_validation.transition_anchored_scores import probability,relation_scores,traces
from research.reading_validation.transition_score import dump,sha


PROTOCOL=Path("research/reading_validation/transition-ambicoref-choice-pairs-protocol.md")
MODELS=(PRIMARY,"native_prior","native_steps_update","native_attention_update","source_semantic")
METRICS=("pair_brier","pair_logloss","auroc","average_precision","outcome_rate")
BOOT=4000


def build(items,geometry,forecasts,definite_only):
    keys=[];pair_sheet=[];opposite=[];pairs_per=[];family=[]
    for k in items:
        obs=geometry[k]["questionnaire_choices"]
        accepted=(0,4) if definite_only else (0,1,3,4)
        eligible=[(s,c) for s,c in obs if c in accepted]
        if len(eligible)<2:continue
        if len(set(s for s,_ in eligible))!=len(eligible):
            raise ValueError("A sentence appears more than once on one questionnaire")
        n=0
        for (a,t),(b,u) in combinations(eligible,2):
            pair_sheet.append((a,b));opposite.append((t<2)!=(u<2));n+=1
        keys.append(k);pairs_per.append(n);family.append(geometry[k]["family"])
    if not keys:return None
    fam,group=np.unique(family,return_inverse=True)
    sheets,ids=np.unique(np.array(pair_sheet).ravel(),return_inverse=True);ids=ids.reshape(-1,2)
    boundaries=np.cumsum(np.r_[0,pairs_per])[:-1]
    opp=np.array(opposite,dtype=int)
    def observation(w):
        pair=w[:,ids[:,0]]*w[:,ids[:,1]]
        return np.add.reduceat(pair*opp,boundaries,axis=1)/np.add.reduceat(pair,boundaries,axis=1)
    y=observation(np.ones((1,len(sheets))))[0]
    q={m:2*(p:=np.array([forecasts[k][m] for k in keys]))*(1-p) for m in MODELS}
    totals=np.bincount(group,weights=y);n=np.bincount(group)
    q["other_families_empirical"]=(y.sum()-totals[group])/(len(y)-n[group])
    if any(not np.all((v>0)&(v<1)) for v in q.values()):
        raise ValueError("External model yielded a zero-probability reader-pair outcome")
    orders={m:rank_order(v) for m,v in q.items()}
    # Different training fold rates carry no within-fold ordering across candidate words.
    orders["other_families_empirical"] = rank_order(np.ones(len(keys)))
    def evaluate(w,y,probs,order):
        a=w*y; b=w*(1-y);norm=w.sum(1)
        au,ap=ranking(*order,a,b)
        return np.column_stack(((a@(1-probs)**2+b@probs**2)/norm,
                    (-a@np.log(probs)-b@np.log1p(-probs))/norm,au,ap,a.sum(1)/norm))
    one={m:evaluate(np.ones((1,len(y))),y[None,:],p,orders[m])[0] for m,p in q.items()}
    rng=np.random.default_rng(24054);reps=defaultdict(list)
    for step in range(0,BOOT,48):
        size=min(48,BOOT-step);sub=observation(rng.exponential(size=(size,len(sheets))))
        material=rng.exponential(size=(size,len(fam)))[:,group]
        for m,p in q.items():reps[m].append(evaluate(material,sub,p,orders[m]))
    reps={m:np.concatenate(x) for m,x in reps.items()}
    def ci(x,v):
        return {m:[float(x[j]),*np.quantile(v[:,j],[.025,.975]).tolist()] for j,m in enumerate(METRICS)}
    return {"sentences":len(keys),"pairs":len(opp),"opposite_pairs":int(opp.sum()),"questionnaires":len(sheets),
            "lexical_skeletons":len(fam),"number_of_pairs_per_sentence":dict(Counter(map(int,pairs_per))),
            "scores":{m:ci(one[m],reps[m]) for m in q},"paired":{m:ci(one[PRIMARY]-one[m],reps[PRIMARY]-reps[m]) for m in q if m!=PRIMARY}}


def main(folder):
    receipt=json.loads((folder/"receipt.json").read_text()); seal=json.loads((OUT/"external-seal.json").read_text())
    f=folder/"prior-words.npz";e=folder/"evidence-words.npz";gold=ROOT/"ambicoref-observed-choices.json"
    if (sha(gold)!=seal["outcomes_sha256"] or receipt["input_sha256"]!=seal["input_sha256"] or
            sha(f)!=receipt["prior_forecast_sha256"] or sha(e)!=receipt["evidence_forecast_sha256"]):
        raise ValueError("Independent external judgments or forecasts changed")
    geo=json.loads(gold.read_text());scores={}
    with np.load(f) as prior,np.load(e) as update:
        for k,r in geo.items():
            raw=traces(prior[k],update[k]);scores[k]={m:float(probability(relation_scores(raw[m],r["names"]))[-1]) for m in MODELS}
    output={}
    panels={"all":list(geo),"following_words":[k for k,v in geo.items() if v["k"]>v["q"]],
                            "terminal":[k for k,v in geo.items() if v["k"]==v["q"]]}
    for name,keys in panels.items():
        for restrict in (False,True):
            output[name+("_definite" if restrict else "_directional")]=build(keys,geo,scores,restrict)
    dump(OUT/"independent-reader-pair-choice.json",{"protocol_sha256":sha(PROTOCOL),"external_seal_sha256":sha(OUT/"external-seal.json"),"panels":output})
    print({k:(v["sentences"],v["scores"][PRIMARY]["pair_brier"]) for k,v in output.items() if v},flush=True)


if __name__=="__main__":
    cli=ArgumentParser(description=__doc__);cli.add_argument("--features",required=True,type=Path)
    main(cli.parse_args().features)
