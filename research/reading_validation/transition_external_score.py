"""Evaluate locked probabilities against independent incremental human coherence judgments."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.reading_validation.transition_external_outcomes import votes
from research.reading_validation.transition_external_prepare import OUT, csv_rows, norm
from research.reading_validation.transition_score import dump, sha, with_interval

BOOT=10000


def evaluate_one(y, total, score, cutoff, group_number, draws):
    """Rank distinct score thresholds, giving each original rater one vote."""
    if total.min()<1 or np.any(y>total) or np.any(y<0):
        raise ValueError("Invalid denominators")
    sort=np.argsort(-score,kind="mergesort")
    sorted_score=score[sort]
    starts=np.r_[0,np.flatnonzero(np.diff(sorted_score)!=0)+1]
    events=y[sort]; neg=(total-y)[sort]; event_group=group_number[sort]
    alarm=sorted_score>cutoff
    values={k:[] for k in ("ap","auroc","sensitivity","false_alarm_rate","precision","prevalence","weighted_f1")}
    def row_metrics(weights):
        positives=weights[:,event_group]*events
        negatives=weights[:,event_group]*neg
        pos=positives.sum(axis=1); n=negatives.sum(axis=1)
        yp=np.add.reduceat(positives,starts,axis=1)
        yn=np.add.reduceat(negatives,starts,axis=1)
        tp=yp.cumsum(axis=1); fp=yn.cumsum(axis=1)
        alerted_pos=positives[:,alarm].sum(axis=1); alerted_neg=negatives[:,alarm].sum(axis=1)
        with np.errstate(divide="ignore",invalid="ignore"):
            ap=(yp*np.divide(tp,tp+fp,out=np.zeros_like(tp,dtype=float),where=(tp+fp)>0)).sum(axis=1)/pos
            auroc=(yn*(tp-.5*yp)).sum(axis=1)/(pos*n)
            tn=n-alerted_neg
            fpfn=alerted_neg+pos-alerted_pos
            weighted_f1=(pos*2*alerted_pos/(2*alerted_pos+fpfn)+n*2*tn/(2*tn+fpfn))/(pos+n)
            return dict(ap=ap,auroc=auroc,sensitivity=alerted_pos/pos,false_alarm_rate=alerted_neg/n,
                        weighted_f1=weighted_f1,
                        precision=alerted_pos/(alerted_pos+alerted_neg),prevalence=pos/(pos+n))
    base=row_metrics(np.ones((1,draws.shape[1]),dtype=int))
    for batch in range(0,len(draws),64):
        chunk=row_metrics(draws[batch:batch+64])
        for k,v in chunk.items():
            values[k].append(v)
    bootstrap={k:np.concatenate(v) for k,v in values.items()}
    metrics={k:with_interval(base[k][0],bootstrap[k]) for k in base}
    metrics["cutoff"]=float(cutoff)
    metrics["alert_counts"]={"true_positive":int(events[alarm].sum()),"false_positive":int(neg[alarm].sum()),
                             "human_positive":int(events.sum()),"human_negative":int(neg.sum())}
    return metrics,bootstrap


def within_document_order(y, total, score, idx, draws):
    per_doc = np.full(draws.shape[1], np.nan)
    for group in range(len(per_doc)):
        loc = idx == group
        weights = y[loc,None] * (total[loc] - y[loc])[None,:]
        np.fill_diagonal(weights, 0)
        if weights.sum():
            delta = score[loc,None] - score[loc][None,:]
            per_doc[group] = np.sum(weights*((delta>0)+.5*(delta==0)))/weights.sum()
    valid = np.isfinite(per_doc)
    weight = draws[:,valid]
    with np.errstate(divide="ignore", invalid="ignore"):
        samples = weight @ per_doc[valid] / weight.sum(axis=1)
    return {**with_interval(per_doc[valid].mean(),samples), "eligible_documents":int(valid.sum())}


def grouped_distribution(group_labels):
    groups,inv=np.unique(group_labels,return_inverse=True)
    if len(groups)<2:raise ValueError("Independent held-out materials required")
    draws=np.random.default_rng(9271).multinomial(len(groups),np.full(len(groups),1/len(groups)),size=BOOT)
    return groups,inv,draws


def agreement(dataset):
    grouped = dataset[dataset.total>=2]
    positive = grouped[grouped.y>0]
    negative = grouped[grouped.total>grouped.y]
    groups=np.r_[positive.group.to_numpy(),negative.group.to_numpy()]
    y=np.r_[positive.y.to_numpy(),np.zeros(len(negative),dtype=int)]
    n=np.r_[positive.y.to_numpy(),(negative.total-negative.y).to_numpy()]
    score=np.r_[((positive.y-1)/(positive.total-1)).to_numpy(),(negative.y/(negative.total-1)).to_numpy()]
    _,idx,draws=grouped_distribution(groups)
    metrics,_=evaluate_one(y,n,score,.5,idx,draws)
    return {"stories":int(grouped.group.nunique()),"boundaries":len(grouped),"reader_judgments":int(n.sum()),
            "ap":metrics["ap"],"auroc":metrics["auroc"]}


def results(dataset, thresholds, *, consensus=False):
    obs=dataset[dataset.total.gt(0)].copy()
    groups,idx,draws=grouped_distribution(obs.group.to_numpy())
    if consensus:
        y=(2*obs.y>=obs.total).astype(int).to_numpy()
        n=np.ones_like(y)
    else:
        y=obs.y.to_numpy()
        n=obs.total.to_numpy()
    runs={}
    stats={}
    for model,cut in thresholds.items():
        runs[model],stats[model]=evaluate_one(y,n,obs[model].to_numpy(),cut,idx,draws)
        runs[model]["within_document_concordance"]=within_document_order(y,n,obs[model].to_numpy(),idx,draws)
    contrasts={}
    for name in ("integration","directed","student"):
        contrasts[name]={}
        for comparator in ("lexical","display","lexical_surprisal","local_state","static","carter_integration","integration"):
            if comparator==name:
                continue
            contrasts[name][comparator]={}
            for k in ("ap","auroc"):
                p=runs[name][k]["estimate"]-runs[comparator][k]["estimate"]
                contrasts[name][comparator][k]=with_interval(p,stats[name][k]-stats[comparator][k])
    passed=all(contrasts["integration"][name]["ap"]["family_bootstrap_95_interval"][0]>0
               for name in ("lexical","display"))
    passed &= runs["integration"]["auroc"]["family_bootstrap_95_interval"][0]>.5
    return {"families":len(groups),"sentences":len(obs),"reader_judgments":int(n.sum()),
            "positive":int(y.sum()),"negative":int((n-y).sum()),"models":runs,
            "predeclared_raw_head_gate":bool(passed),"comparisons":contrasts}


def edit_quality(rewrites):
    material = {r["id"]:r for r in json.loads((OUT/"stimuli.json").read_text())["material"] if r["source"]=="decor_rewrites"}
    archive = csv_rows("test_rewrite_213_no_delete.csv")
    quality=[]
    for bid in rewrites:
        source=archive[material[bid]["source_ref"][1]]
        new,old=norm(source["Rewrite"]),norm(source["sentence"])
        quality.append("identical" if old==new else "instruction" if new=="Delete second half of sentence" else "changed")
    return np.array(quality)


def score_rewrites(rewrites, pool, thresholds):
    src=pool[pool.source.isin(["decor","decor_rewrite_original"])].set_index("id",verify_integrity=True)
    groups,inv,draws=grouped_distribution(rewrites.group.to_numpy())
    comparisons={}
    w=draws[:,inv]
    quality=edit_quality(rewrites.id.tolist())
    for arm in thresholds:
        if arm == "directed":
            # The separate matched-turn packet makes original and edited exposures identical
            # before the target. Its evaluation is stored in matched-scores.json.
            continue
        old=src.loc[rewrites.counterpart,arm].to_numpy()
        new=rewrites[arm].to_numpy()
        raw_diff = np.abs(new[quality=="identical"]-old[quality=="identical"])
        new=np.where(quality=="identical",old,new)
        delta=new-old
        metrics={}
        for name,v in (("new_minus_old",delta),("strict_improvement",(new<old).astype(float)),
                       ("half_tie_preference",(new<old)+.5*(new==old))):
            boot=(w @ v)/w.sum(axis=1)
            metrics[name]=with_interval(v.mean(),boot)
        metrics["lower_count"]=int((new<old).sum());metrics["tie_count"]=int((new==old).sum())
        metrics["old_alarms"]=int((old>thresholds[arm]).sum());metrics["new_alarms"]=int((new>thresholds[arm]).sum())
        metrics["identical_input_max_abs_difference"]=float(raw_diff.max()) if len(raw_diff) else None
        comparisons[arm]=metrics
    return {"pairs":len(rewrites),"essays":len(groups),"models":comparisons,
            "change_units":"probability for binary trained arms, predicted rating deficit for Carter transfers"}


def score():
    forecast=OUT/"heldout-predictions.csv"; ck=OUT/"readouts.json"
    lock=json.loads((OUT/"prediction-lock.json").read_text())
    if sha(forecast)!=lock["forecast_sha256"] or sha(ck)!=lock["model_sha256"]:
        raise ValueError("Run integrity failed after prediction seal")
    specs=json.loads(ck.read_text()); thresholds=specs["thresholds"]
    meta=json.loads((OUT/"stimuli.json").read_text())["material"]
    pred=pd.read_csv(forecast)
    static=json.loads((OUT/"static-lock.json").read_text())
    if (sha(OUT/"static-predictions.json")!=static["forecast_sha256"] or
            sha(Path("research/reading_validation/transition-static-control.md"))!=static["protocol_sha256"]):
        raise ValueError("Static-control predictions changed after sealing")
    independent=pd.DataFrame(json.loads((OUT/"static-predictions.json").read_text()))
    pred=pred.merge(independent,on=["id","group","index","source","split"],how="outer",validate="1:1")
    if (pred.isna().any().any()
            or np.max(np.abs(pred.lexical-pred.lexical_cpu))>1e-8):
        raise ValueError("Independent CPU/GPU lexical control forecasts differ")
    thresholds["static"]=static["arms"]["static"]["threshold"]
    by_key={(r["id"],r["source"]):r for r in meta if r["split"] in ("external","holdout")}
    if len(by_key)!=len(pred):
        raise ValueError("Duplicate or missing boundary identifiers")
    res={}
    for rec in pred.itertuples():
        source=by_key[rec.id,rec.source]
        if source["source"] in ("decor","cohesentia"):
            t=votes(source)
            entry={"id":rec.id,"source":rec.source,"y":t[0],"total":t[1],"invalid_reader_records":t[3]}
            entry.update({"reason_"+str(k):v for k,v in t[2].items()})
        else:
            entry={"id":rec.id,"source":rec.source,"counterpart":source.get("counterpart")}
        res[rec.id,rec.source]=entry
    joined=pred.merge(pd.DataFrame(res.values()),on=["id","source"],validate="1:1")
    joined.to_csv(OUT/"heldout-scored.csv",index=False,float_format="%.15g")
    co=joined[joined.source.eq("cohesentia")].copy()
    de=joined[joined.source.eq("decor")].copy()
    for df in [co,de]:
        df.y=df.y.astype(int)
        df.total=df.total.astype(int)
    rewritten=joined[joined.source.eq("decor_rewrites")]
    changed=rewritten[edit_quality(rewritten.id.tolist())=="changed"]
    stats={"cohesentia_per_rater":results(co,thresholds), "cohesentia_other_readers":agreement(co),
           "cohesentia_consensus":results(co,thresholds,consensus=True),
           "decor":results(de,thresholds),
           "expert_edit_preference":score_rewrites(rewritten,joined,thresholds),
           "expert_changed_edit_preference":score_rewrites(changed,joined,thresholds),
           "cohesentia_ambiguous_worker_story_records":int(co.groupby("group").invalid_reader_records.max().sum())}
    dump(OUT/"scores.json",stats)
    return stats


if __name__=="__main__":
    all_results=score()
    for name in ["cohesentia_per_rater","decor"]:
        r=all_results[name]
        print(name,r["families"],r["sentences"],r["reader_judgments"],"positive",r["positive"])
        for method in ["lexical","display","integration","directed","carter_integration"]:
            m=r["models"][method]
            print(method,"AP",m["ap"],"AUROC",m["auroc"],"alerts",m["alert_counts"])
