"""Score the direct word-inside-click readout without privileging later labels."""
from argparse import ArgumentParser
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.special import expit,logsumexp

from research.reading_validation.transition_external_score import grouped_distribution
from research.reading_validation.transition_null_score import OUT as JOINT,ROOT,OLD,flatten
from research.reading_validation.transition_pointer_selective import calibrate,evaluate
from research.reading_validation.transition_score import dump,sha
from research.reading_validation.transition_word_region_utility import infer,learn,observed


OUT=JOINT.parent/"word-acceptance"
PROTOCOL=Path("research/reading_validation/transition-word-region-utility-protocol.md")


def analysis(keys,all_scores,words,rows,labels,arm,cutoffs):
    methods=("direct","joint_softmax","factorized_softmax");metric_records=defaultdict(list);policy=defaultdict(list);numbers=[0,0]
    work=[];docs=[];people=0
    for k in keys:
        doc,q=k.rsplit("/",1);q=int(q);own=rows[doc]["group"];y,n=observed(labels[k],q);people+=n
        raw=all_scores[k][arm].astype(float);scores=words[k][arm].astype(float)
        p0=np.exp(raw-logsumexp(raw));w=.5*(p0+1/q);p_direct=expit(scores);top0=int(raw.argmax());best=int(scores.argmax())
        numbers[0]+=best!=top0;numbers[1]+=1
        a,b=words["__actions__"][k][arm]
        preds={"direct":p_direct,"joint_softmax":p0*a,"factorized_softmax":p0*b}
        for method,p in preds.items():
            if np.any((p<=0)|(p>=1)):raise ValueError("Word acceptance probability reached a boundary")
            idx=int(p.argmax());z=float(p[idx]);hit=float(y[idx])
            brier=float(w@(y*np.square(1-p)+(1-y)*p*p))
            nll=float(w@(-y*np.log(p)-(1-y)*np.log1p(-p)))
            metric_records[method].append([brier,nll,hit,z,z-hit])
            policy[method].append((own,z,hit,n))
        work.append(own);docs.append(doc)
    groups,ii,draws=grouped_distribution(work);agg={}
    for m,v in metric_records.items():
        x=np.array(v);agg[m]=np.stack([x[ii==j].mean(0) for j in range(len(groups))])
    result={}
    for m,x in {**agg,**{"direct_minus_"+c:agg["direct"]-agg[c] for c in methods if c!="direct"}}.items():
        boot=draws@x/len(groups);result[m]={field:[float(x[:,j].mean()),*np.quantile(boot[:,j],[.025,.975]).tolist()]
                    for j,field in enumerate(("word_brier","word_logloss","offered_seed_hit","estimated_seed_acceptance","seed_signed_overconfidence"))}
    cut=cutoffs.get("selected");thresholds={"always":0.,"fixed_50":.5,"fixed_80":.8}
    if cut:thresholds["source_selected"]=cut["threshold"]
    modes={m:{nm:evaluate(v,t)[0] for nm,t in thresholds.items()} for m,v in policy.items()}
    doc_rows=[(d,*r[1:]) for d,r in zip(docs,policy["direct"],strict=True)]
    return {"work_count":len(groups),"article_count":len(set(docs)),"eligible_pronouns":len(keys),"reader_choices":people,
            "changed_winning_position":numbers,"scores":result,"work_policies":modes,"article_direct":{nm:evaluate(doc_rows,t)[0] for nm,t in thresholds.items()}}


def main(folder):
    seal=json.loads((JOINT/"source-seal.json").read_text());r=json.loads((folder/"receipt.json").read_text())
    file=folder/"all-prior-logits.npz";packet=ROOT/"null-choice-pointer-stimuli.jsonl";sfile=ROOT/"gate-pointer-stimuli.jsonl"
    if sha(file)!=r["forecasts_sha256"] or r["input_sha256"]!=seal["stimuli_sha256"] or sha(packet)!=r["input_sha256"] or sha(sfile)!=seal["original_action_data_sha256"]:
        raise ValueError("Word-level source inputs differ")
    oldf=folder/"factorized-forecasts.json";js=JOINT/"all-four-action-forecasts.npz";jsel=json.loads((JOINT/"source-decision.json").read_text())
    if sha(oldf)!=r["factorized_sha256"] or sha(js)!=jsel["four_action_forecasts_sha256"]:raise ValueError("Mixed-action baseline predictions changed")
    source_prior=np.load(file);previous=json.loads(oldf.read_text());joint=np.load(js)
    old=list(map(json.loads,sfile.read_text().splitlines()));rows=list(map(json.loads,packet.read_text().splitlines()));lookup={v["id"]:v for v in rows}
    training,dev=(flatten({r["id"]:r["actions"] for r in old if r["fold"]==s}) for s in ("train","dev"))
    OUT.mkdir(exist_ok=True);models=[];fits={}
    for j,m in enumerate(("lexical","native")):
        model,health=learn(rows,source_prior,j,training);models.append(model);fits[m]=health
    weights=OUT/"source-model.npz";np.savez_compressed(weights,**{arm+"__"+k:v for arm,model in zip(("lexical","native"),models,strict=True) for k,v in model.items()})
    scores={};forecasts={};doc_scores=defaultdict(list);actions={};dev_policies={"lexical":[],"native":[]}
    for row in rows:
        if row["fold"]=="train":continue
        for i,q in enumerate(row["queries"]):
            key=row["id"]+"/"+str(q);raw=source_prior[key];x=np.stack([infer(row,i,raw[a],m)[0] for a,m in enumerate(models)]).astype(np.float32)
            scores[key]=x;z=joint[key];actions[key]=[(float(z[jsel["forecast_arms"].index(m),0]),old_p) for m,old_p in
                                                      zip(("joint_lexical","joint_native"),previous[key],strict=True)]
            forecasts[key]=[[int(z.argmax()),float(expit(z.max()))] for z in x]
            if key in dev:
                y,n=observed(dev[key],q)
                for a,m in enumerate(dev_policies):
                    top,p=forecasts[key][a];dev_policies[m].append((row["group"],p,float(y[top]),n))
    f=OUT/"top-word-forecasts.json";dump(f,forecasts)
    policy={m:calibrate(v) for m,v in dev_policies.items()};lock=OUT/"source-policy.json"
    dump(lock,{"protocol_sha256":sha(PROTOCOL),"model_sha256":sha(weights),"forecasts_sha256":sha(f),"full_word_score_origin":sha(file),
               "fits":fits,"methods":policy,"source_qualified":policy["native"]["selected"]})
    scores["__actions__"]=actions
    a_file=JOINT/"independent-article-actions.json";p_file=JOINT/"external-reader-actions.json";gfile=OLD/"external-actions.json"
    if sha(a_file)!=seal["external_actions_sha256"] or sha(p_file)!=seal["reader_panel_sha256"] or sha(gfile)!=json.loads((OLD/"source-seal.json").read_text())["gold_sha256"]:
        raise ValueError("A previously registered response databank changed")
    a=json.loads(a_file.read_text());panel=json.loads(p_file.read_text());g=json.loads(gfile.read_text())
    sets={"earlier_gold":g,"new_articles":a["actions"],"new_no_known_gold_work":{d:z for d,z in a["actions"].items() if not a["material_overlap"][d]["shares_work_known_gold"]},
          **{"new_people_"+k:v for k,v in panel.items()}}
    report={}
    for name,s in sets.items():
        y=flatten(s);report[name]={arm:analysis(y,source_prior,scores,lookup,y,i,policy[arm]) for i,arm in enumerate(("lexical","native"))}
    choice=policy["native"]["selected"]
    release=bool(choice and all(report[name]["native"]["article_direct"]["source_selected"]["precision"] and
                report[name]["native"]["article_direct"]["source_selected"]["precision"]["whole_group_95_interval"][0]>=.70 for name in ("earlier_gold","new_articles")))
    dump(OUT/"scores.json",{"source_policy_sha256":sha(lock),"ordinary_automatic_release":release,"cohorts":report})
    print("Qualified:",release,{k:v["native"]["scores"]["direct"]["offered_seed_hit"] for k,v in report.items()},flush=True)


if __name__=="__main__":
    p=ArgumentParser(description=__doc__);p.add_argument("--features",type=Path,required=True);main(p.parse_args().features)
