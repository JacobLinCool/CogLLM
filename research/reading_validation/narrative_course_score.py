"""Score complete-story forecasts on human question identity, annotation counts and suspense."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from research.reading_validation.memory_data import sha
from research.reading_validation.narrative_course_data import OUT
from research.reading_validation.narrative_course_features import load
from research.reading_validation.narrative_course_fit import mean_rating


def spearman(a,b):
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
    keep=np.isfinite(a)&np.isfinite(b)
    u=a[keep];v=b[keep]
    if len(u)<3 or u.min()==u.max() or v.min()==v.max():
        return None
    return float(np.corrcoef(rankdata(u),rankdata(v))[0,1])


def hit(a,b):
    return float((a>b)+.5*(a==b))


def identity(story,choices):
    result={}
    for name,table in choices.items():
        grouped=defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
        for team,t,question,positive,negative in story.comparisons:
            grouped[team][t][question].append(hit(table[t,positive],table[t,negative]))
        team_means=[np.mean([np.mean([np.mean(rows) for rows in site.values()]) for site in sites.values()])
                    for sites in grouped.values()]
        all_choices=defaultdict(list)
        for team,t,gold in story.events:
            v=table[t,:t]
            best=np.flatnonzero(v==v.max())
            all_choices[team].append(len(set(gold)&set(best))/len(best))
        result[name]=dict(paired_accuracy=float(np.mean(team_means)) if team_means else None,
                          top1_any_earlier=float(np.mean([np.mean(x) for x in all_choices.values()])) if all_choices else None)
    return dict(**{k:int(v) for k,v in story.totals.items()}, pairs=len(story.comparisons), closing_team_lines=len(story.events),
                competing_team_lines=len({(t,j) for t,j,_,_,_ in story.comparisons}),methods=result)


def count_outcomes(s,predictions):
    result={}
    for name,p in predictions.items():
        values=[]
        for state in s.teams.values():
            lo=(state==1).sum(1);hi=(state!=0).sum(1)
            r=p-np.clip(p,lo,hi)
            sign=np.where(lo[1:]>hi[:-1],1,np.where(hi[1:]<lo[:-1],-1,0))
            observed=sign!=0
            change=sign*np.diff(p)
            arg=change[observed]
            values.append(dict(interval_mse=float(np.mean(r*r)), rank_midpoint=spearman(p,(lo+hi)/2),
                               direction=float(((arg>0)+.5*(arg==0)).mean()) if len(arg) else None,
                               determinate_transitions=int(observed.sum())))
        result[name]={}
        for key in values[0]:
            obs=[v[key] for v in values if v[key] is not None]
            result[name][key]=float(np.mean(obs)) if obs else None
    return result


def reader_agreement(s,seed,draws=250):
    rng=np.random.default_rng(seed)
    word=[];change=[]
    for _ in range(draws):
        order=rng.permutation(len(s.ratings))
        split=len(order)//2
        a,b=mean_rating(s.ratings[order[:split]]),mean_rating(s.ratings[order[split:]])
        ok=s.word_ok&np.isfinite(a)&np.isfinite(b)
        word.append(spearman(a[ok],b[ok]))
        change.append(spearman(np.diff(s.line_mean(a,ok)),np.diff(s.line_mean(b,ok))))
    def report(data):
        data=[v for v in data if v is not None]
        return dict(median=float(np.median(data)),central95=np.quantile(data,[.025,.975]).tolist())
    return dict(word_levels=report(word),consecutive_line_changes=report(change),splits=draws)


def screen_sensitivity(s,measured,predicted,valid,seed):
    """Resample entire visible screens. Spearman ranks are recalculated for each sample."""
    rng=np.random.default_rng(seed)
    screens=sorted(set(s.screens[valid])-{-1})
    if len(screens)<3:
        return None
    parts=[np.flatnonzero(valid&(s.screens==k)) for k in screens]
    estimates=[]
    for sample in rng.integers(len(parts),size=(250,len(parts))):
        ix=np.concatenate([parts[j] for j in sample])
        value=spearman(predicted[ix],measured[ix])
        if value is not None:
            estimates.append(value)
    return np.quantile(estimates,[.025,.975]).tolist() if estimates else None


def suspense_outcomes(s,predictions):
    mean=mean_rating(s.ratings)
    valid=s.word_ok&np.isfinite(mean)
    focus=s.line_mean(mean,valid)
    result={}
    for name,curve in predictions.items():
        line=s.line_mean(curve,valid)
        result[name]=dict(rank_all_words=spearman(mean[valid],curve[valid]),
                          rank_line_changes=spearman(np.diff(line),np.diff(focus)),
                          line_change_rmse=float(np.sqrt(np.nanmean((np.diff(line)-np.diff(focus))**2))),
                          eligible_words=int(valid.sum()),eligible_line_differences=int(np.isfinite(np.diff(focus)).sum()),
                          levels_screen_sensitivity=screen_sensitivity(s,mean,curve,valid,66552))
    return result


def run(archive,forecasts,dest):
    seal=json.loads((forecasts/"seal.json").read_bytes())
    if (sha(archive/"receipt.json")!=seal["native"] or sha(forecasts/"predictions.npz")!=seal["predictions"]):
        raise ValueError("The full-story forecast differs from the lock")
    all_stories=load(archive)
    output={}
    with np.load(forecasts/"predictions.npz",allow_pickle=False) as pred:
        for i,(key,s) in enumerate(all_stories.items()):
            def methods(label):
                prefix=f"{key}.{label}."
                return {k.removeprefix(prefix):pred[k] for k in pred.files if k.startswith(prefix)}
            output[key]=dict(question_identity=identity(s,methods("pairs")),
                             annotation_counts=count_outcomes(s,methods("count")),
                             felt_suspense=suspense_outcomes(s,methods("tension")),reader_half_agreement=reader_agreement(s,9199+i))
    metrics={}
    for endpoint,sub,field in [("which_issue","question_identity","paired_accuracy"),("earlier_line","question_identity","top1_any_earlier"),
                     ("count_mse","annotation_counts","interval_mse"),("count_direction","annotation_counts","direction"),
                     ("suspense_levels","felt_suspense","rank_all_words"),("suspense_changes","felt_suspense","rank_line_changes")]:
        rows=[s[sub]["methods"] if sub=="question_identity" else s[sub] for s in output.values()]
        metrics[endpoint]={m:float(np.mean([v[m][field] for v in rows if v[m][field] is not None]))
                           if any(v[m][field] is not None for v in rows) else None for m in rows[0]}
    wins=sum(s["question_identity"]["methods"]["semantic"]["paired_accuracy"]>
             s["question_identity"]["methods"]["form"]["paired_accuracy"] for s in output.values()
             if s["question_identity"]["methods"]["semantic"]["paired_accuracy"] is not None)
    doc=dict(forecast=sha(forecasts/"seal.json"),stories=output,equal_story=metrics,semantic_pair_wins=int(wins))
    dest.write_text(json.dumps(doc,indent=2,allow_nan=False)+"\n")
    print(json.dumps(metrics,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("archive",type=Path);p.add_argument("--forecasts",type=Path,default=OUT/"forecast")
    p.add_argument("--dest",type=Path,default=OUT/"scores.json")
    args=p.parse_args();run(args.archive,args.forecasts,args.dest)
