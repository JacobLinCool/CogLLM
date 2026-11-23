"""Measure the word-local change in actual readers' completion disagreement."""
from collections import Counter
from pathlib import Path
import re
import numpy as np
from research.backbone_transfer.extract import digest, read_json, write_json
from research.reading_validation.cloze_population_data import OUT, ROOT
from research.reading_validation.physiology_portability_score import fast_ranks, R


def exact_observed(counter):
    n = sum(counter.values())
    if n < 2:
        raise ValueError("Disagreement requires at least two independent responses")
    return 1. - sum(k * (k - 1) for k in counter.values()) / (n * (n - 1))


def interval(a):
    return np.quantile(a, [.025, .975]).tolist()


def orientation(predicted, observed):
    return (1. + np.sign(predicted) * np.sign(observed)) / 2


def study(rows, meta, exact, example=False, discrepancies=None):
    signals = {}
    key = None
    for name, values in rows.items():
        x = np.asarray(values, float)
        x = x[np.argsort(x[:,0])]
        ids = x[:,0].astype(int)
        if key is None:
            key = ids
            coarse = 1. - x[:,4]
        elif not np.array_equal(key,ids) or not np.array_equal(coarse,1.-x[:,4]):
            raise ValueError("Word populations have unmatched observational support")
        lower = 1. - x[:,3]
        upper = lower + x[:,10] ** 2
        signals[name] = (lower, upper)
    if len(set(key)) != len(key):
        raise ValueError("Duplicate original printed endpoint")
    xgroup = meta["articles"][key]
    xpos = meta["pos"][key]
    groups = sorted(set(xgroup))
    lookup = {(a, int(t)):i for i,(a,t) in enumerate(zip(xgroup,xpos,strict=True))}
    if len(lookup) != len(key):
        raise ValueError("Two response observations at the same ordinal")
    pairs=[(i,lookup[a,t+1]) for (a,t),i in lookup.items() if (a,t+1) in lookup]
    a,b = np.asarray(sorted(pairs)).T
    _,group = np.unique(xgroup[a],return_inverse=True)
    K = int(group.max()+1)
    observed=coarse[b]-coarse[a]
    hard=np.abs(observed)>=.20
    fine=np.array([exact[i] for i in key])
    fine_delta=fine[b]-fine[a]
    rng=np.random.default_rng(812951)
    W=rng.multinomial(K,np.full(K,1/K),size=2000)
    counts=np.bincount(group)
    W=np.vstack((np.ones(K),W))
    weights=W@counts
    yscore=fast_ranks(observed)
    measures={}
    draws={}
    for name,(lower,upper) in signals.items():
        p=lower[b]-lower[a]
        allowed=(np.abs(observed)>1e-10)
        rho=None
        if np.ptp(p)>1e-10:
            rank=fast_ranks(p)
            rho=np.array([R(rank(w[group]),yscore(w[group]),w[group]) for w in W])
        err=np.abs(p-observed)
        mae=W@np.bincount(group,weights=err,minlength=K)/weights
        orientation_score=orientation(np.where(np.abs(p)<1e-10,0,p),observed)
        hit=W@np.bincount(group[allowed],weights=orientation_score[allowed],minlength=K)
        directional=hit/(W@np.bincount(group[allowed],minlength=K))
        def signs(mask):
            return {"count":int(mask.sum()),
                    "correct_direction":float(orientation_score[mask].mean()) if np.any(mask) else None}
        lo=lower[b]-upper[a];hi=upper[b]-lower[a]
        sign=(lo>1e-10).astype(int)-(hi < -1e-10).astype(int)
        def selective(mask):
            return {"observed_changes":int(mask.sum()),"identified_directions":int(np.sum(mask&(sign!=0))),
                    "correct_among_identified":float(np.mean(sign[mask&(sign!=0)]==np.sign(fine_delta[mask&(sign!=0)]))) if np.any(mask&(sign!=0)) else None}
        measures[name]={"rho":None if rho is None else float(rho[0]),"rho_interval":None if rho is None else interval(rho[1:]),
                        "mae":float(mae[0]),"mae_interval":interval(mae[1:]),
                        "correct_direction":float(directional[0]),"direction_interval":interval(directional[1:]),
                        "increases":signs(observed>1e-10),"decreases":signs(observed < -1e-10),
                        "large_increases":signs((observed>0)&hard),"large_decreases":signs((observed<0)&hard),
                        "exact_word_sensitivity":{"identified_rate":float(np.mean(sign!=0)),
                             "increases":selective(fine_delta>1e-10),"decreases":selective(fine_delta< -1e-10)}}
        draws[name]=(rho,mae,directional)
    comparisons={}
    for other in ("calibrated","raw"):
        row={}
        for j,k in enumerate(("rho_gain","MAE_reduction","direction_gain")):
            v=draws["context"][j]-draws[other][j]
            if j==1:
                v=-v
            row[k]={"difference":float(v[0]),"paired_passage_interval":interval(v[1:])}
        comparisons["context_over_"+other]=row
    out={"prefixes":len(key),"adjacent_updates":len(a),"original_passages":len(groups),"passages_with_consecutive_updates":K,"measured_no_change":int(np.sum(np.abs(observed)<=1e-10)),
         "large_absolute_changes":int(hard.sum()),"models":measures,"paired":comparisons}
    if example:
        candidates=[(str(g),np.flatnonzero(xgroup==g)) for g in groups if meta["complete"][key[xgroup==g]].all()]
        mid=np.median([len(i) for _,i in candidates])
        name,idx=min(candidates,key=lambda c:(abs(len(c[1])-mid),tuple(map(int,re.findall(r"\d+",c[0]))),c[0]))
        idx=idx[np.argsort(xpos[idx])]
        if np.any(np.diff(xpos[idx])!=1):
            raise ValueError("Selected complete text has a discontinuous response series")
        out["illustrative_article"]={"article":name,"target_word_positions":xpos[idx].tolist(),"coarse_empirical":coarse[idx].tolist(),"unreconciled_reported_denominators":[str(meta["ids"][key[i]]) in discrepancies for i in idx],
            "human_exact":fine[idx].tolist(),"modeled_contextual":signals['context'][0][idx].tolist(),"modeled_calibrated":signals['calibrated'][0][idx].tolist()}
    return out


def evaluate(source,clean,held,meta,details,discrepancies):
    seal=read_json(OUT/"change-registration.json")
    if seal["protocol"]!=digest(Path(__file__).with_name("cloze-change-protocol.md")):
        raise ValueError("Longitudinal protocol differs")
    counts=read_json(ROOT/"source-counts.json")
    exact={i:exact_observed(counts[uid]) for i,(flag,uid) in enumerate(zip(meta["source"],meta["ids"],strict=True)) if flag}
    exact.update({i:v[1] for i,v in details.items()})
    result={"protocol":seal["protocol"],"population_lock":digest(OUT/"forecast-lock.json"),
            "UCL_sentence_heldout":study(source,meta,exact),"Provo_external":study(clean,meta,exact),
            "all_Provo_released_row_mass_sensitivity":study(held,meta,exact,True,discrepancies)}
    target=OUT/"reading-updates.json"
    if target.exists():
        if read_json(target)!=result:
            raise ValueError("Resumed update assay differs")
    else:
        write_json(target,result)
    x=result["Provo_external"]
    print("Native-held adjacent updates",x["adjacent_updates"],x["models"]["context"]["rho"],x["paired"]["context_over_calibrated"])
