"""Estimate ordinary pronoun response repeatability directly from independent same-position players."""
from collections import Counter,defaultdict
import json
from pathlib import Path

import numpy as np

from research.reading_validation.transition_external_score import grouped_distribution
from research.reading_validation.transition_score import dump,sha


OUT=Path("research/results/reading-validation-transition-pointer/joint-actions")
ROOT=Path(".cache/cogllm/reading-validation")
ACTIONS=("DO","DN","NR","PR")
SERIES=("same_four_action","loo_four_action_logloss","loo_four_action_brier","loo_best_prior_word_hit",
        "loo_prior_word_offered","both_chose_prior","both_prior_overlap")


def measure(actions,q):
    n=sum(r[3] for r in actions);counts=np.zeros(4,int);cover=np.zeros(q,int)
    if n<4:raise ValueError("Peer recurrence needs at least four independent judgments")
    for r,l,h,k in actions:
        counts[ACTIONS.index(r)]+=k
        if r=="DO":cover[l:h]+=k
    s=np.zeros(len(SERIES))
    for action,l,h,count in actions:
        code=ACTIONS.index(action);rest=counts.copy();rest[code]-=1
        p=(rest+.5)/(n-1+2);vote=count/n;peers=cover.copy()
        if action=="DO":peers[l:h]-=1
        top=peers==peers.max();offer=peers.max()>0
        hit=top[l:h].sum()/top.sum() if action=="DO" and offer else 0.
        second_prior=rest[0]/(n-1) if action=="DO" else 0.
        intersect=sum(k for typ,left,right,k in actions if typ=="DO" and min(h,right)>max(l,left))-1 if action=="DO" else 0.
        s+=vote*np.array([rest[code]/(n-1),-np.log(p[code]),1-2*p[code]+p@p,hit,offer,second_prior,
                          intersect/(n-1)])
    return s


def summarize(rows,label):
    samples=[];owners=[]
    for d,qs in label.items():
        for q,data in qs.items():
            samples.append(measure(data["actions"],int(q)));owners.append(rows[d]["group"])
    all_groups,ids,draws=grouped_distribution(owners);v=np.stack(samples)
    a=np.stack([v[ids==g].mean(0) for g in range(len(all_groups))]);mean=a.mean(0);b=draws@a/len(all_groups)
    def ci(p,z):return [float(p),*np.quantile(z,[.025,.975]).tolist()]
    result={k:ci(mean[i],b[:,i]) for i,k in enumerate(SERIES)}
    result["span_overlap_given_two_prior"]=ci(mean[6]/mean[5],b[:,6]/b[:,5]);result["word_hit_given_loo_offered"]=ci(mean[3]/mean[4],b[:,3]/b[:,4])
    return {"works":len(all_groups),"positions":len(v),"independent_actions":sum(a[3] for q in label.values() for r in q.values() for a in r["actions"]),"scores":result}


def main():
    old=OUT.parent/"action-gate";seal=json.loads((OUT/"source-seal.json").read_text());external=OUT/"independent-article-actions.json"
    packet=ROOT/"null-choice-pointer-stimuli.jsonl";gold=old/"external-actions.json"
    if sha(packet)!=seal["stimuli_sha256"] or sha(external)!=seal["external_actions_sha256"] or sha(gold)!=json.loads((old/"source-seal.json").read_text())["gold_sha256"]:
        raise ValueError("Observation or experiment geometry changed")
    rows={r["id"]:r for r in map(json.loads,packet.read_text().splitlines())};obs=json.loads(external.read_text());a=obs["actions"]
    outcomes={k:summarize(rows,l) for k,l in (("earlier_gold",json.loads(gold.read_text())),("new_articles",a),
              ("new_work_no_older_gold", {d:v for d,v in a.items() if not obs["material_overlap"][d]["shares_work_known_gold"]}))}
    dump(OUT/"human-repeatability.json",{"protocol_sha256":sha(Path("research/reading_validation/transition-pointer-action-agreement.md")),"cohorts":outcomes})
    print({k:v["scores"]["same_four_action"] for k,v in outcomes.items()})


if __name__=="__main__":main()
