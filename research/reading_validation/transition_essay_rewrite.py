"""Learn one coherence score from absolute judgments and expert replacements of the same passage."""
from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit

from research.reading_validation.essay_hazard_inference import EssayHazardInference, WEIGHTS
from research.reading_validation.transition_essay_hazard import basis
from research.reading_validation.transition_essay_reasons import DEST, METRICS, evaluate, mass
from research.reading_validation.transition_external_fit import read_features
from research.reading_validation.transition_external_prepare import ROOT, csv_rows, norm
from research.reading_validation.transition_external_score import edit_quality
from research.reading_validation.transition_score import dump, sha
from research.reading_validation.transition_writer_f1 import threshold

PLAN = Path(__file__).with_name("transition-essay-rewrite-protocol.md")
METHODS = ("lexical_frozen", "native_frozen", "lexical_joint", "native_absolute", "native_replacement", "native_joint")
ENDPOINTS = ("expert_edit_preference", "old_minus_new_risk", "forced_choice_logloss")


def material(folder):
    records, _, raw, arms = read_features(folder)
    content = csv_rows("test_1355_clean.csv")
    editing = csv_rows("test_rewrite_213_no_delete.csv")
    docs = {r["id"]:r for r in map(json.loads,(ROOT/"cohesion-stimuli.jsonl").read_text().splitlines())}
    stops = {v["id"]:v for doc in docs.values() for v in doc["stops"]}
    ix = {r["id"]:i for i,r in enumerate(records) if r["source"] in ("decor", "decor_rewrite_original")}
    direct = np.array([i for i,r in enumerate(records) if r["source"] == "decor"])
    labels = np.array([int(content[records[i]["source_ref"][1]]["label"] == "0.0") for i in direct])
    topics = np.array([content[records[i]["source_ref"][1]]["topic"] for i in direct])
    essays = np.array([records[i]["group"] for i in direct])
    pairs = [r for r in records if r["source"] == "decor_rewrites"]
    quality = edit_quality([r["id"] for r in pairs])
    edit_ids = {r["id"]:i for i,r in enumerate(records) if r["source"] == "decor_rewrites"}
    originals = []; edits = []; edit_topics = []; edit_essays = []; lengths = []
    for r,q in zip(pairs,quality,strict=True):
        a,b = ix[r["counterpart"]],edit_ids[r["id"]]
        first,second = records[a],records[b]; left,right = stops[first["id"]],stops[second["id"]]
        source,revision = docs[first["doc"]]["text"],docs[second["doc"]]["text"]
        if (first["group"]!=second["group"] or first["index"]!=second["index"] or
            norm(source[:left["start"]])!=norm(revision[:right["start"]])):
            raise ValueError("Expert replacement changed an earlier sentence or its index")
        if q != "changed":
            continue
        old = source[left["start"]:left["end"]]; new = revision[right["start"]:right["end"]]
        original_row=editing[r["source_ref"][1]]
        if norm(old)!=norm(original_row["sentence"]) or norm(new)!=norm(original_row["Rewrite"]):
            raise ValueError("The original and replacement passages were not recovered exactly")
        originals.append(a); edits.append(b); edit_topics.append(original_row["topic"]); edit_essays.append(r["group"])
        lengths.append(len(new.split())-len(old.split()))
    if len(labels)!=1352 or sum(labels)!=446 or len(edits)!=203:
        raise ValueError("Unexpected independent essay task denominator")
    return (raw[:,arms["integration"]], arms, direct, labels, topics, essays, np.array(originals), np.array(edits),
            np.array(edit_topics), np.array(edit_essays), np.array(lengths))


def training_design(features, begin, direct, y, t, e, old, new, pt, pe, direct_mask, edit_mask, which):
    blocks=[]; offsets=[]; labels=[]; weights=[]
    if which in ("joint","absolute"):
        idx=direct[direct_mask]
        blocks.append(features[idx]); offsets.append(begin[idx]); labels.append(y[direct_mask])
        weights.append(mass(t[direct_mask],e[direct_mask]))
    if which in ("joint","replacement"):
        a,b=old[edit_mask],new[edit_mask]
        blocks.append(features[a]-features[b]); offsets.append(begin[a]-begin[b]); labels.append(np.ones(len(a)))
        weights.append(mass(pt[edit_mask],pe[edit_mask]))
    k=len(weights)
    return np.vstack(blocks),np.concatenate(offsets),np.concatenate(labels),np.concatenate(weights)/k


def fit(design):
    x,begin,target,w=design
    def objective(theta):
        eta=begin+x@theta
        value=w@(np.logaddexp(0.,eta)-target*eta)+.005*(theta@theta)
        gradient=x.T@(w*(expit(eta)-target))+.01*theta
        return float(value),gradient
    res=minimize(objective,np.zeros(x.shape[1]),jac=True,method="L-BFGS-B",
                 options={"maxiter":400,"gtol":2e-9,"ftol":1e-15,"maxls":50})
    gradient=float(abs(res.jac).max())
    if not np.isfinite(res.fun) or gradient>1e-6:
        raise ArithmeticError(f"The joint essay observer failed stationarity ({gradient})")
    return res.x,{"iterations":int(res.nit),"max_gradient":gradient,"objective":float(res.fun)}


def reliability(design,theta):
    x,begin,y,w=design; adjust=x@theta
    def grad(alpha): return float(w@((expit(begin+alpha*adjust)-y)*adjust))
    if grad(0)>=0:return 0.
    if grad(1)<=0:return 1.
    lo,hi=0.,1.
    for _ in range(60):
        mid=(lo+hi)/2
        if grad(mid)>0:hi=mid
        else:lo=mid
    return (lo+hi)/2


def summarized(held,weights,edit_weights,y):
    probs={k:(v["risk"],v["flag"]) for k,v in held.items()}
    absolute=evaluate(y,probs,weights)
    values={}
    for k,v in held.items():
        old,new=v["old"],v["new"]; diff=old-new
        pair=np.column_stack(((diff>0)+.5*(diff==0),diff,np.logaddexp(0.,logit(new)-logit(old))))
        values[k]=np.column_stack((absolute[k],(edit_weights@pair)/edit_weights.sum(1,keepdims=True)))
    return values


def main(folder):
    X,arms,direct,y,t,e,old,new,pt,pe,lengths=material(folder)
    source=json.loads((DEST/"source-anchored-hazard-scores.json").read_text())
    order=[r["topic"] for r in source["folds"]]
    if set(t)!=set(order) or set(pt)!=set(order):raise ValueError("Every expected source topic needs judgments and edits")
    native,lexical=EssayHazardInference.load().probabilities(X,per_model=True)
    held={k:{field:np.zeros(n) for field,n in (("risk",len(y)),("flag",len(y)),("old",len(old)),("new",len(new)))} for k in METHODS}
    folds=[]
    for i,topic in enumerate(order):
        dev=order[(i+1)%len(order)]
        train=(t!=topic)&(t!=dev); cal=t==dev; test=t==topic
        etrain=(pt!=topic)&(pt!=dev); ecal=pt==dev; etest=pt==topic
        w=mass(t[train],e[train]);native_x=basis(X,direct[train],w,arms)
        lx=X[:,arms["lexical_surprisal"]];mu=w@lx[direct[train]]
        sd=np.sqrt(w@(lx[direct[train]]-mu)**2);consider=sd>1e-8
        lexical_x=np.column_stack((np.ones(len(X)),(lx[:,consider]-mu[consider])/(sd[consider]*np.sqrt(consider.sum()))))
        record={"held_topic":topic,"calibration_topic":dev,"source_edits":int(etrain.sum()),"calibration_edits":int(ecal.sum()),
                "held_edits":int(etest.sum()),"fits":{}}
        for method,storage in held.items():
            base,typ=method.split("_");p=(native if base=="native" else lexical)[:,i]
            begin=logit(np.clip(p,1e-10,1-1e-10));z=native_x if base=="native" else lexical_x
            if typ=="frozen":
                score=expit(begin);fit_info={}
            else:
                def design(d,ed):
                    return training_design(z,begin,direct,y,t,e,old,new,pt,pe,d,ed,typ)
                theta,fit_info=fit(design(train,etrain))
                strength=reliability(design(cal,ecal),theta)
                score=expit(begin+strength*(z@theta))
                fit_info["source_strength"]=strength
            score=np.clip(score,1e-12,1-1e-12)
            cut=threshold(score[direct[cal]],y[cal])["cutoff"]
            storage["risk"][test]=score[direct[test]];storage["flag"][test]=score[direct[test]]>=cut
            storage["old"][etest]=score[old[etest]];storage["new"][etest]=score[new[etest]]
            record["fits"][method]={**fit_info,"cutoff":cut}
        folds.append(record)
    names=list(METRICS)+list(ENDPOINTS)
    pt_mass=np.unique(np.r_[t,pt]);essay_mass=np.unique(np.r_[e,pe])
    ti=np.searchsorted(pt_mass,t);pi=np.searchsorted(pt_mass,pt)
    ei=np.searchsorted(essay_mass,e);pei=np.searchsorted(essay_mass,pe)
    unit=np.ones((1,len(y)));edge=np.ones((1,len(old)));point=summarized(held,unit,edge,y)
    boot={k:[] for k in held};rng=np.random.default_rng(291131)
    for _ in range(80):
        tv=rng.exponential(size=(50,len(pt_mass)));ev=rng.exponential(size=(50,len(essay_mass)))
        values=summarized(held,tv[:,ti]*ev[:,ei],tv[:,pi]*ev[:,pei],y)
        for k,v in values.items():boot[k].append(v)
    boot={k:np.concatenate(v) for k,v in boot.items()}
    def report(pt,dist):
        return {k:[float(x),*np.quantile(v,[.025,.975]).tolist()] for k,x,v in zip(names,pt[0],dist.T,strict=True)}
    results={k:report(point[k],boot[k]) for k in held}
    contrasts={k:report(point["native_joint"]-point[k],boot["native_joint"]-boot[k]) for k in held if k!="native_joint"}
    per={}
    for g in order:
        ix=pt==g
        per[g]={k:float(((a["old"][ix]>a["new"][ix])+.5*(a["old"][ix]==a["new"][ix])).mean()) for k,a in held.items()}
    by_length={}
    for category,ix in (("shorter",lengths<0),("same",lengths==0),("longer",lengths>0)):
        by_length[category]={"pairs":int(ix.sum()),"strict_preference":{k:float((a["old"][ix]>a["new"][ix]).mean()) for k,a in held.items()}}
    path=DEST/"expert-replacement-forecasts.npz";np.savez_compressed(path,**{k+"."+f:v for k,info in held.items() for f,v in info.items()})
    dump(DEST/"expert-replacement-scores.json",{"protocol_sha256":sha(PLAN),"source_readout_sha256":sha(WEIGHTS),"predictions_sha256":sha(path),
          "topics":len(order),"essays":len(set(e)),"changed_revisions":len(new),"edited_essays":len(set(pe)),"judged_originals":len(y),
          "clean_unchanged":9,"editing_instruction":1,"folds":folds,"models":results,"native_joint_minus_matched":contrasts,
          "per_topic_edit_preference":per,"by_word_count_change":by_length})
    for k,v in results.items():print(k,{n:v[n][0] for n in ("expert_edit_preference","logloss","ap","forced_choice_logloss")},flush=True)
    print("paired", {k:(v["expert_edit_preference"],v["logloss"],v["ap"]) for k,v in contrasts.items()},flush=True)


if __name__=="__main__":
    p=ArgumentParser(description=__doc__);p.add_argument("--features",type=Path,required=True);main(p.parse_args().features)
