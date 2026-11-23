"""Compare locked forecasts with unseen Wikipedia votes and independent essay judgments."""

from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score

from research.reading_validation.discourse_data import OUT, RAW
from research.reading_validation.discourse_fit import METHODS, collect, observations, rates
from research.reading_validation.joint_data import digest, read


def statistics(observed, draws):
    ok=np.isfinite(draws)
    if not ok.any():
        raise ValueError("No estimable grouped bootstraps")
    return {"estimate":float(observed),"interval95":np.quantile(draws[ok],[.025,.975]).tolist(),
            "valid_draws":int(ok.sum())}


def grouped(loss, inverse, weights):
    sizes=np.bincount(inverse,minlength=weights.shape[1])
    means=np.bincount(inverse,weights=loss,minlength=len(sizes))/sizes
    return means.mean(), weights@means/len(sizes)


def score_wikipedia(data, indices, seal):
    rows=[r for r in read(OUT/"materials.json") if r["genre"]=="wikipedia"]
    votes=collect({r["item"] for r in rows})
    counts=observations(rows,votes,seal["classes"])
    truths,disagree,eligible=rates(counts)
    result={}
    for j,m in enumerate(METHODS):
        take=np.flatnonzero(eligible[j])
        idx=[indices[rows[i]["item"]] for i in take]
        groups,inverse=np.unique([rows[i]["group"] for i in take],return_inverse=True)
        rng=np.random.default_rng(731035)
        weights=rng.multinomial(len(groups),np.full(len(groups),1/len(groups)),size=10000)
        y=disagree[j,take]
        truth=truths[j,take]
        methods=("qwen.joint","qwen.single","surface.joint","surface.single","constant","ngrams")
        scores={}
        losses={}
        for name in methods:
            q=data[f"{name}.{m}.disagreement"][idx]
            pred=data[f"{name}.{m}.choice"][idx]
            if not (np.isfinite(q).all() and ((0<=q)&(q<=1)).all()
                    and np.allclose(pred.sum(1),1) and (pred>=0).all()):
                raise ValueError(f"Invalid forecast for {name}.{m}")
            implied=1-(pred**2).sum(1)
            ls={"mse":(q-y)**2,"choice_ce":-(truth*np.log(pred.clip(1e-12))).sum(1),
                "implied_mse":(implied-y)**2,"bias":q-y}
            losses[name]=ls
            stats={metric:statistics(*grouped(v,inverse,weights)) for metric,v in ls.items()}
            stats["item_spearman"]=None if np.std(q)<1e-12 else float(spearmanr(y,q).statistic)
            stats["calibration"]=[]
            by_prediction=np.array_split(np.argsort(q,kind="stable"),5)
            for bin_ in by_prediction:
                stats["calibration"].append({"n":len(bin_),"mean_predicted":float(q[bin_].mean()),
                                             "mean_observed":float(y[bin_].mean())})
            scores[name]=stats
        paired={}
        for other in methods[1:]:
            paired[other]={k:statistics(*grouped(losses["qwen.joint"][k]-losses[other][k],inverse,weights))
                           for k in ("mse","choice_ce","implied_mse")}
        # Agreement reproducibility of two disjoint groups of readers on the same items.
        # With ~10 choices per item, estimate halves over 200 predetermined random partitions.
        split_corr=[]
        for rep in range(200):
            half=[]
            for i in take:
                labs=np.array(list(votes[m][rows[i]["item"]].values()))
                if len(labs)<4:
                    continue
                a,b=np.array_split(rng.permutation(labs),2)
                def u(l):
                    c=np.unique(l,return_counts=True)[1]
                    return 1-(c*(c-1)).sum()/(len(l)*(len(l)-1))
                half.append((u(a),u(b)))
            a,b=np.transpose(half)
            split_corr.append(float(spearmanr(a,b).statistic))
        result[m]=dict(items=len(take),articles=len(groups),votes=int(counts[j,take].sum()),
                       reader_ids=len({k for s in votes[m].values() for k in s}),
                       observed_disagreement=statistics(*grouped(y,inverse,weights)),
                       scores=scores,paired_qwen_joint_minus=paired,
                       split_half_rank_median_and_central_95=np.quantile(split_corr,[.5,.025,.975]).tolist())
    return result


def score_essays(data, indices):
    source=Path("research/results/reading-validation-transition-external/heldout-scored.csv")
    with source.open(newline="") as f:
        rows=[r for r in csv.DictReader(f) if r["source"]=="decor"]
    y=np.array([int(float(r["reason_3"] or 0)) for r in rows])
    idx=[indices[r["id"]] for r in rows]
    group,inv=np.unique([r["group"] for r in rows],return_inverse=True)
    predictors={k:data[k+".forced.disagreement"][idx] for k in ("qwen.joint","surface.joint","ngrams")}
    for k in ("integration","lexical_surprisal"):
        predictors[k]=np.array([float(r[k]) for r in rows])
    rng=np.random.default_rng(731036)
    w=rng.multinomial(len(group),np.full(len(group),1/len(group)),size=10000)
    results={}
    boot={}
    for name,z in predictors.items():
        order=np.argsort(-z,kind="stable")
        starts=np.r_[0,np.where(np.diff(z[order])!=0)[0]+1]
        vals=[]
        for sub in np.array_split(w,25):
            ww=sub[:,inv[order]]
            pos=np.add.reduceat(ww*y[order],starts,axis=1)
            tot=np.add.reduceat(ww,starts,axis=1).cumsum(1)
            prec=pos.cumsum(1)/np.maximum(tot,1)
            denom=pos.sum(1)
            vals.extend(np.divide((pos*prec).sum(1),denom,out=np.full(len(ww),np.nan),where=denom>0))
        boot[name]=np.array(vals)
        results[name]=statistics(average_precision_score(y,z),boot[name])
    paired={name:statistics(results["qwen.joint"]["estimate"]-results[name]["estimate"],boot["qwen.joint"]-b)
            for name,b in boot.items() if name!="qwen.joint"}
    return dict(n=len(y),events=int(y.sum()),essays=len(group),average_precision=results,
                paired_qwen_minus=paired,essay_source_sha256=digest(source))


def run():
    folder=OUT/"forecast"
    seal=read(folder/"seal.json")
    for name,val in seal["files"].items():
        if digest(folder/name)!=val:
            raise ValueError("Prediction changed after sealing")
    if digest(RAW/"item.csv")!=read(OUT/"inputs.json")["source"]["item.csv"]:
        raise ValueError("The public worker votes have changed")
    with np.load(folder/"forecasts.npz",allow_pickle=False) as f:
        ix={k:i for i,k in enumerate(f["item"])}
        if len(ix)!=len(f["item"]):
            raise ValueError("Duplicate study row in forecast")
        a=score_wikipedia(f,ix,seal)
        b=score_essays(f,ix)
    result=dict(wikipedia=a,essay_specificity=b,forecast_seal_sha256=digest(folder/"seal.json"))
    (OUT/"scores.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:v["paired_qwen_joint_minus"]["surface.joint"]["mse"] for k,v in a.items()}))


if __name__=="__main__":
    run()
