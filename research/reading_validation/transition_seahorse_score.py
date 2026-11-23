"""Score held SEAHORSE comprehension answers after signed prediction lock, preserving original article families."""
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import average_precision_score, roc_auc_score

from research.reading_validation.transition_seahorse_data import DEST, PLAN, sha, source_rows
from research.reading_validation.transition_seahorse_features import load, save
from research.reading_validation.transition_seahorse_fit import key, corpus, LABEL

B = 4000


def summary(x, boot):
    v = boot[np.isfinite(boot)]
    return {"estimate":float(x),"interval":np.quantile(v,[.025,.975]).tolist(),"bootstrap_draws":len(v)}


def stats(y, pred, threshold):
    if not np.all((pred>=0)&(pred<=1)):
        raise ValueError("Predictions must be unconditional probabilities")
    lab=pred>=threshold; tp=(lab&(y==1)).sum();fp=(lab&(y==0)).sum();fn=(~lab&(y==1)).sum()
    nll=-(y*np.log(np.clip(pred,1e-15,1))+(1-y)*np.log(np.clip(1-pred,1e-15,1)))
    brier=(pred-y)**2
    return {"rows":len(y),"no":int(y.sum()),"prevalence_no":float(y.mean()),"mean_predicted_no":float(pred.mean()),
            "nll":float(nll.mean()),"brier":float(brier.mean()),"AUROC":float(roc_auc_score(y,pred)),
            "average_precision_no":float(average_precision_score(y,pred)),
            "pearson":float(pearsonr(y,pred).statistic) if np.ptp(pred)>0 else None,
            "threshold":float(threshold),"true_mark":int(tp),"false_mark":int(fp),"missed":int(fn),
            "precision":float(tp/max(1,tp+fp)),"recall":float(tp/max(1,tp+fn)),
            "false_alarms_per_100_clear":float(100*fp/((y==0).sum())),
            "F1":float(2*tp/max(1,2*tp+fp+fn))},nll,brier


def group_rows(g):
    v,id=np.unique(g,return_inverse=True)
    return [np.flatnonzero(id==i) for i in range(len(v))],id


def within(grows, y, p):
    estimates=np.full(len(grows),np.nan)
    pairs=np.zeros(len(grows),int)
    for i, ix in enumerate(grows):
        a=p[ix[y[ix]==1]];b=p[ix[y[ix]==0]]
        if len(a) and len(b):
            diff=a[:,None]-b[None,:];pairs[i]=diff.size;estimates[i]=((diff>0)+.5*(diff==0)).mean()
    return estimates,pairs


def drafts(y, pred, g, thresholds, subsets=False):
    methods=list(pred)
    grouped,gid=group_rows(g);G=len(grouped); rng=np.random.default_rng(875143)
    n=np.bincount(gid);C=rng.multinomial(G,np.full(G,1/G),size=B).astype(np.int16)
    denominator=C@n
    allscores={};rep={}
    for name in methods:
        raw=name.endswith("_raw"); v=pred[name]
        if raw:
            if len(np.unique(y))<2:raise ValueError("Both response classes expected")
            info={"AUROC":float(roc_auc_score(y,v)),"average_precision_no":float(average_precision_score(y,v))}
            nll=brier=None
        else:
            info,nll,brier=stats(y,v,thresholds[name])
        indiv,pairs=within(grouped,y,v); ok=np.isfinite(indiv); total=int(pairs.sum())
        info["within_article"]={"informative_articles":int(ok.sum()),"opposite_response_pairs":total}
        dd=(C[:,ok].astype(float)@indiv[ok])/(C[:,ok].sum(1)) if ok.any() else np.full(B,np.nan)
        if ok.any():info["within_article"].update(summary(indiv[ok].mean(),dd))
        local={"within_article":dd}
        for typ,val in [("nll",nll),("brier",brier)]:
            if val is not None:
                z=(C.astype(float)@np.bincount(gid,weights=val))/denominator
                local[typ]=z
                info[typ+"_interval"]=np.quantile(z,[.025,.975]).tolist()
        allscores[name]=info;rep[name]=local
    contrast={}
    targets=[("frozen_native_raw","frozen_form_raw","within_article"),
             ("native","form","nll"),("native","form","within_article"),
             ("metadata_native","metadata_form","nll"),("character_native","character_form","nll"),
             ("native","metadata","nll"),("native","character","nll")]
    for a,b,metric in targets:
        x=allscores[a][metric];z=allscores[b][metric]
        if metric=="within_article":x=x["estimate"];z=z["estimate"]
        contrast[a+" minus "+b+" "+metric]=summary(x-z,rep[a][metric]-rep[b][metric])
    return allscores,contrast


def strata_statistics(y, methods, values):
    out={}
    for name in np.unique(values):
        mask=values==name; primary={}
        for method in ("frozen_native_raw","frozen_form_raw","native","form","metadata","character_native","character_form"):
            v=methods[method][mask];yy=y[mask]
            if len(np.unique(yy))>1:
                primary[method]={"AUROC":float(roc_auc_score(yy,v)),"average_precision_no":float(average_precision_score(yy,v))}
        out[name]={"rows":int(mask.sum()),"no":int(y[mask].sum()),"scores":primary}
    return out


def duplicates(forecast, metadata):
    records=list(source_rows("test",duplicates=True,grades=True))
    groups=defaultdict(list);canonical={}
    for row in records:
        k=key(row);groups[k].append(row["question1"]);canonical[k]=row
    observed=Counter();by_binary=Counter();brier=defaultdict(list);nll=defaultdict(list)
    value_pairs={};held_proba=[]; responses=[]; article=[]
    for k,ballot in groups.items():
        if k not in metadata:continue
        true=[int(LABEL[v]) for v in ballot if v in LABEL]
        if len(true)>=2:
            g=canonical[k]["gem_id"]
            for a,b in combinations(ballot,2):
                observed["total"]+=1;observed["different"]+=a!=b
            for a,b in combinations(true,2):
                by_binary["total"]+=1;by_binary["different"]+=a!=b
            for name,pred in forecast.items():
                if name.endswith("_raw"):continue
                p=pred[metadata[k]]
                yy=np.array(true)
                brier[name].extend(((p-yy)**2).tolist())
                nll[name].extend((-(yy*np.log(np.clip(p,1e-15,1))+(1-yy)*np.log(np.clip(1-p,1e-15,1)))).tolist())
            prior=forecast["native"][metadata[k]]
            held_proba.append(2*prior*(1-prior));responses.append(np.mean([a!=b for a,b in combinations(true,2)]))
            article.append(g)
            value_pairs[k]={"valid":len(true),"no":sum(true)}
    d=np.stack([held_proba,responses],axis=1)
    keep={k:{"rows":len(v),"nll":float(np.mean(nll[k])),"brier":float(np.mean(v))} for k,v in brier.items()}
    return {"original_rows":len(records),"unique_comparisons":len(groups),
            "groups_aligned_to_official_row_and_forecast":sum(k in metadata for k in groups),
            "matching_input_ballots":sum(len(v) for k,v in groups.items() if k in metadata),
            "scored_multi_binary_groups":len(value_pairs),"scored_articles":len(set(article)),
            "published_three_level_disagreement":dict(observed),"definitive_pair_disagreement":dict(by_binary),
            "individual_vote_scores":keep,
            "observed_mean_item_pair_disagreement":float(d[:,1].mean()),
            "model_mean_item_disagreement":float(d[:,0].mean()),
            "disagreement_AUC":float(roc_auc_score(d[:,1]>0,d[:,0])) if len(np.unique(d[:,1]>0))>1 else None,
            "unit":"Same displayed source draft and generator. Additional distributions include the ballots from which one named official grade may have been drawn."}


def score():
    seal=load(DEST/"fit-seal.json");src=DEST/"forecasts-before-held-responses.npz"
    if seal["forecast_sha256"]!=sha(src) or seal["protocol"]!=sha(PLAN):
        raise ValueError("Held forecasts must have a valid pre-response lock")
    with np.load(src,allow_pickle=False) as z:
        forecast={k:z[k] for k in z.files if k!="keys"};ids=list(z["keys"])
    official={key(r):r for r in source_rows("test",grades=True)}
    meta={k:i for i,k in enumerate(ids)}
    grades=Counter(r["question1"] for r in official.values())
    include=np.array([official[k]["question1"] in LABEL for k in ids])
    eligible=[official[k] for i,k in enumerate(ids) if include[i]]
    methods={k:v[include] for k,v in forecast.items()};y=np.array([LABEL[r["question1"]] for r in eligible])
    g=np.array([r["gem_id"] for r in eligible]); models=np.array([r["model"] for r in eligible]); domains=np.array([corpus(v) for v in g])
    main,paired=drafts(y,methods,g,seal["thresholds"])
    historical=load(DEST/"historical-exposure.json")
    contaminated=set(historical["source_articles_with_matches"]) | set().union(*[set(v) for v in load(DEST/"word-exposure-audit.json")["test"]["overlap_with_earlier_sources"].values()])
    good=~np.isin(g,list(contaminated)); reduced,paired_clean=drafts(y[good],{k:v[good] for k,v in methods.items()},g[good],seal["thresholds"])
    corr={}
    for control in ("form","character","metadata"):
        corr[control]=float(np.corrcoef(methods[control],methods["native"])[0,1])
    output={"source":seal["source"],"protocol":sha(PLAN),"recorded_pre_grade_forecast":sha(src),
            "official_rows":len(official),"official_RESPONSE_counts":dict(grades),
            "held_rows_with_observed_forecast":int(len(ids)),"scored_definitive":len(y),"missed_by_native_grade":dict(Counter(r["question1"] for k,r in official.items() if k not in meta)),
            "scored_no":int(y.sum()),"original_articles":len(set(g)),"scores":main,"paired":paired,
            "strict_earlier_source_exposure":{"articles_excluded":sorted(contaminated),"rows_left":int(good.sum()),"articles_left":len(set(g[good])),
                                             "native":reduced["native"],"comparisons":paired_clean},
            "per_dataset":strata_statistics(y,methods,domains),"per_system":strata_statistics(y,methods,models),
            "within_generator_dataset":strata_statistics(y,methods,np.array([a+"/"+b for a,b in zip(domains,models,strict=True)])),
            "forecast_correlation_with_native":corr,"repeat_reader_responses":duplicates(forecast,meta)}
    save(DEST/"scores.json",output)
    print("Native",main["native"],"Controls",[(k,v["estimate"],v["interval"]) for k,v in paired.items()],flush=True)


if __name__=="__main__":
    score()
