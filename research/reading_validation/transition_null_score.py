"""Fit competing discourse actions on source works, then open the two external reader panels."""

from argparse import ArgumentParser
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax

from research.reading_validation.antecedent_phrase_inference import PHRASE_MODEL, PHRASE_MODEL_SHA, PhraseInference, PriorWords
from research.reading_validation.transition_external_score import grouped_distribution
from research.reading_validation.transition_gate_features import features
from research.reading_validation.transition_pointer_features import design
from research.reading_validation.transition_pointer_selective import calibrate, evaluate
from research.reading_validation.transition_score import dump, sha


ROOT=Path(".cache/cogllm/reading-validation")
OUT=Path("research/results/reading-validation-transition-pointer/joint-actions")
OLD=OUT.parent/"action-gate"
PROTOCOL=Path("research/reading_validation/transition-pointer-null-protocol.md")
ACTIONS=("DO","DN","NR","PR")


def candidates(row, logits, index, phrase):
    q=row["queries"][index]
    scores=np.asarray(logits,dtype=float)
    if scores.shape!=(q,) or not np.isfinite(scores).all():raise ValueError("Candidate scores do not cover the visible prefix")
    tags=list(row["pos"][:q]);tags[-1]=row["pos_last"][index]
    seed=int(scores.argmax());log_p=scores-logsumexp(scores);p=np.exp(log_p)
    phrase_start,phrase_end=phrase.region(PriorWords.from_prefix(row["text"],row["words"],tags,q),seed)
    best=np.sort(scores)[::-1][:3];mass=float(logsumexp(scores)-np.log(q))
    entropy=float(-p@log_p)
    # When only one earlier word exists its runner-up is the word itself.
    attributes=[mass,best[0],best[min(1,len(best)-1)],best[0]-best[min(1,len(best)-1)],entropy,
                entropy/np.log(max(2,q)),np.sort(p)[-3:].sum(),np.log1p(q-seed)]
    x=np.r_[features(row,q,tags),attributes]
    return {"x":x,"offset":mass,"seed":seed,"point_mass":float(p[seed]),"phrase_mass":float(p[phrase_start:phrase_end].sum()),
            "conditional":log_p}


def action_weights(actions):
    n=np.zeros(4);total=sum(entry[3] for entry in actions)
    if total<1:raise ValueError("A human target has no response")
    for action,left,right,count in actions:
        if action not in ACTIONS or count<1:raise ValueError("Invalid human action")
        n[ACTIONS.index(action)]+=count
    return n/total,int(total)


def equal_works(groups):
    counts=Counter(groups)
    return np.array([1/(len(counts)*counts[g]) for g in groups])


def learn(items, human):
    keys=list(human);X=np.stack([items[k]["x"] for k in keys]);offset=np.array([items[k]["offset"] for k in keys])
    target=np.stack([action_weights(human[k])[0] for k in keys])
    groups=[items[k]["group"] for k in keys];weight=equal_works(groups)
    mu=X.mean(0);sd=X.std(0);sd[sd<1e-10]=1.
    z=(X-mu)/sd;z=np.column_stack((np.ones(len(z)),z))
    def obj(vector):
        beta=vector.reshape(-1,3);s=np.column_stack((offset,z@beta));logp=s-logsumexp(s,axis=1,keepdims=True)
        loss=-np.sum(weight[:,None]*target*logp)+.005*np.square(beta[1:]).sum()
        grad=z.T@(weight[:,None]*(np.exp(logp[:,1:])-target[:,1:]));grad[1:]+=.01*beta[1:]
        return loss,grad.ravel()
    initial=np.zeros(3*z.shape[1]);sol=minimize(obj,initial,jac=True,method="L-BFGS-B",options={"ftol":1e-15,"gtol":1e-9,"maxiter":1500,"maxls":40})
    if not np.isfinite(sol.fun) or np.abs(sol.jac).max()>1e-6:raise RuntimeError(f"Competing null actions failed stationarity: {sol.message} {np.abs(sol.jac).max():.3g}")
    return (mu,sd,sol.x.reshape(-1,3)),{"source_works":len(set(groups)),"source_pronouns":len(keys),
             "source_actions":sum(action_weights(human[k])[1] for k in keys),
             "features":X.shape[1],"free_parameters":len(sol.x),"initial":float(obj(initial)[0]),"objective":float(sol.fun),
             "absolute_max_gradient":float(np.abs(sol.jac).max()),"optimizer_iterations":int(sol.nit)}


def four_prob(item,model):
    mu,sd,beta=model
    return softmax(np.r_[item["offset"],np.r_[1.,(item["x"]-mu)/sd]@beta])


def outcomes(item,action):
    y,n=action_weights(action)
    h=sum(v for method,start,stop,v in action if method=="DO" and start<=item["seed"]<stop)/n
    conditional=0.
    for k,l,r,v in action:
        if k=="DO":
            if not 0<=l<r<=len(item["conditional"]):raise ValueError("True earlier phrase escapes the article")
            conditional-=v/n*logsumexp(item["conditional"][l:r])
    return y,n,h,conditional


def policies(items,labels,prob,calibration=None,group_key="group"):
    modes=("point","phrase")
    by={k:[] for k in modes}
    for key,action in labels.items():
        item=items[key];_,n,h,_=outcomes(item,action)
        for mode in modes:by[mode].append((item[group_key],float(prob[key][0]*item[mode+"_mass"]),h,n))
    if calibration is None:return {m:calibrate(rows) for m,rows in by.items()}
    result={}
    for m,rows in by.items():
        thresholds={"forced":0.,"fixed_50":.5,"fixed_80":.8}
        if choice:=calibration[m]["selected"]:thresholds["source_selected"]=choice["threshold"]
        result[m]={name:evaluate(rows,threshold)[0] for name,threshold in thresholds.items()}
    return result


def score(items, gold, probs, map_method, policies_by_arm):
    keys=list(gold);groups=[items["headwise"][k]["group"] for k in keys]
    works,idx,draws=grouped_distribution(groups)
    metrics=("four_action_logloss","action_plus_click_region_logloss","four_action_brier","prior_choice_signed_calibration",
             "observed_prior_action","predicted_prior_action","conditional_region_loss_per_all_actions")
    matrices={};out={}
    for m,pr in probs.items():
        arm=map_method[m];values=[]
        for key in keys:
            x=items[arm][key];p=pr[key];y,n,h,conditional=outcomes(x,gold[key])
            if (p<=0).any() or not np.isclose(p.sum(),1):raise ValueError("Competing-action probability lost normalization")
            nll=float(-y@np.log(p));brier=float(1-2*y@p+p@p)
            values.append([nll,nll+conditional,brier,p[0]-y[0],y[0],p[0],conditional])
        v=np.array(values);matrices[m]=np.stack([v[idx==i].mean(0) for i in range(len(works))])
    comparisons={**matrices,**{"joint_native_minus_"+m:matrices["joint_native"]-x for m,x in matrices.items() if m!="joint_native"}}
    for m,v in comparisons.items():
        b=draws@v/len(works)
        out[m]={k:[float(v[:,i].mean()),*np.quantile(b[:,i],[.025,.975]).tolist()] for i,k in enumerate(metrics)}
    return {"works":len(works),"whole_articles":len({items["headwise"][k]["doc"] for k in keys}),"eligible_pronouns":len(keys),
            "human_actions":sum(action_weights(gold[k])[1] for k in keys),"scores":out,
            "selection":{method:policies(items[map_method[method]],gold,probs[method],policies_by_arm[method]) for method in policies_by_arm}}


def safety_at_all_sites(items,labels,probs,calibration,documents):
    totals={}
    keys=[k for k,p in items.items() if p["doc"] in documents]
    for mode in ("phrase","point"):
        cuts={"fixed_50":.5,"fixed_80":.8}
        if choice:=calibration[mode]["selected"]:cuts["source_selected"]=choice["threshold"]
        for label,t in cuts.items():
            picks=[k for k in keys if probs[k][0]*items[k][mode+"_mass"]>=t]
            measured=[k for k in picks if k in labels]
            correct=sum(outcomes(items[k],labels[k])[2] for k in measured);unknown=len(picks)-len(measured)
            totals[mode+"/"+label]={"visible_pronouns":len(keys),"offered":len(picks),"without_adjudication":unknown,
               "sharp_precision_bounds":[correct/len(picks),(correct+unknown)/len(picks)] if picks else None,
               "sharp_false_per_100":[100*(len(picks)-correct-unknown)/len(keys),100*(len(picks)-correct)/len(keys)]}
    return totals


def flatten(data):
    return {doc+"/"+str(q):v["actions"] for doc,queries in data.items() for q,v in queries.items()}


def main(folder):
    seal=json.loads((OUT/"source-seal.json").read_text());source=ROOT/"gate-pointer-stimuli.jsonl";stimuli=ROOT/"null-choice-pointer-stimuli.jsonl"
    receipt=json.loads((folder/"receipt.json").read_text());score_file=folder/"all-prior-logits.npz";old_file=folder/"factorized-forecasts.json"
    if (sha(PROTOCOL)!=seal["protocol_sha256"] or sha(stimuli)!=seal["stimuli_sha256"] or sha(source)!=seal["original_action_data_sha256"]
        or receipt["input_sha256"]!=sha(stimuli) or receipt["protocol_sha256"]!=sha(PROTOCOL) or sha(score_file)!=receipt["forecasts_sha256"]
        or sha(old_file)!=receipt["factorized_sha256"] or receipt["held_responses_provided"]!=0):raise ValueError("Source identity differs")
    rows=list(map(json.loads,stimuli.read_text().splitlines()));source_rows=list(map(json.loads,source.read_text().splitlines()))
    if receipt["encoder_forwards"]!=len(rows):raise ValueError("A visible article did not pass through the encoder exactly once")
    forecast=np.load(score_file);old=json.loads(old_file.read_text())
    phrase=PhraseInference.load(PHRASE_MODEL,PHRASE_MODEL_SHA);arms=("lexical","headwise");items={arm:{} for arm in arms}
    for row in rows:
        for i,q in enumerate(row["queries"]):
            key=row["id"]+"/"+str(q)
            for arm,z in zip(arms,forecast[key],strict=True):
                x=candidates(row,z,i,phrase);x.update(group=row["group"],doc=row["id"]);items[arm][key]=x
    if any(len(items[arm])!=receipt["eligible_pronouns"] for arm in arms):raise ValueError("A visible pronoun was omitted")
    train,dev=(flatten({r["id"]:r["actions"] for r in source_rows if r["fold"]==f}) for f in ("train","dev"))
    source_counts=np.stack([action_weights(v)[0] for v in dev.values()])
    constant=equal_works([items["headwise"][k]["group"] for k in dev])@source_counts
    alt=constant[1:]/constant[1:].sum()
    probs={};fits={};saved={};methods={"joint_lexical":"lexical","joint_native":"headwise",
                    "dev_frequency":"headwise","factorized_surface":"headwise","factorized_native":"headwise"}
    for m,arm in list(methods.items())[:2]:
        model,fits[m]=learn(items[arm],train)
        probs[m]={k:four_prob(x,model) for k,x in items[arm].items()}
        for name,arr in zip(("mean","sd","beta"),model,strict=True):saved[m+"__"+name]=arr
    probs["dev_frequency"]={k:constant for k in old}
    for j,m in enumerate(("factorized_surface","factorized_native")):
        probs[m]={k:np.r_[z[j],(1-z[j])*alt] for k,z in old.items()}
    all_actions=OUT/"all-four-action-forecasts.npz"
    np.savez_compressed(all_actions,**{key:np.stack([probs[m][key] for m in methods]).astype(np.float32) for key in old})
    cal={m:policies(items[arm],dev,probs[m]) for m,arm in methods.items() if m.startswith("joint")}
    weights=OUT/"source-model.npz";np.savez_compressed(weights,**saved)
    locked=OUT/"source-decision.json"
    dump(locked,{"protocol_sha256":sha(PROTOCOL),"native_receipt":receipt,"weights_sha256":sha(weights),"four_action_forecasts_sha256":sha(all_actions),"forecast_arms":list(methods),"fits":fits,"dev_four_action_order":ACTIONS,
          "dev_four_action_distribution":constant.tolist(),"policy":cal,"designated_release":"joint_native/phrase",
          "source_qualification":cal["joint_native"]["phrase"]["selected"]})
    old_seal=json.loads((OLD/"source-seal.json").read_text());gold=OLD/"external-actions.json"
    new=OUT/"independent-article-actions.json";panel=OUT/"external-reader-actions.json"
    if sha(gold)!=old_seal["gold_sha256"] or sha(new)!=seal["external_actions_sha256"] or sha(panel)!=seal["reader_panel_sha256"]:
        raise ValueError("Public annotator outcomes changed")
    g=json.loads(new.read_text());a=g["actions"];o=g["material_overlap"];p=json.loads(panel.read_text())
    splits={"earlier_gold":flatten(json.loads(gold.read_text())),"new_articles":flatten(a),
            "new_articles_no_known_gold_phrase":flatten({d:x for d,x in a.items() if not o[d]["shares_20_words_known_gold"]}),
            "new_articles_no_known_gold_work":flatten({d:x for d,x in a.items() if not o[d]["shares_work_known_gold"]}),
            **{"new_people_"+k:flatten(v) for k,v in p.items()}}
    result={k:score(items,labels,probs,methods,cal) for k,labels in splits.items() if labels}
    earlier={r["id"] for r in rows if r["fold"]=="test"}
    for name,documents in (("earlier_gold",earlier),("new_articles",set(a))):
        result[name]["ordinary_article_policy"]=policies(items["headwise"],splits[name],probs["joint_native"],cal["joint_native"],"doc")
        result[name]["all_visible_sites"]=safety_at_all_sites(items["headwise"],splits[name],probs["joint_native"],cal["joint_native"],documents)
    qualified=cal["joint_native"]["phrase"]["selected"]
    release=bool(qualified and all(result[c]["ordinary_article_policy"]["phrase"]["source_selected"]["precision"]
         and result[c]["ordinary_article_policy"]["phrase"]["source_selected"]["precision"]["whole_group_95_interval"][0]>=.70
         for c in ("earlier_gold","new_articles")))
    dump(OUT/"scores.json",{"locked_source_sha256":sha(locked),"ordinary_automatic_release":release,"splits":result})
    print({k:(v["eligible_pronouns"],v["scores"]["joint_native"]["four_action_logloss"],v["scores"]["joint_native_minus_factorized_native"]["four_action_logloss"])
           for k,v in result.items()},flush=True)


if __name__=="__main__":
    parser=ArgumentParser(description=__doc__);parser.add_argument("--features",type=Path,required=True)
    main(parser.parse_args().features)
