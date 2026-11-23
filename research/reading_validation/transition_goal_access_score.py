"""Compare condition-specific old-goal accessibility with reported recognition-speed contrasts."""
from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, t
from research.reading_validation.transition_goal_access_data import DEST,INPUT,PLAN
from research.reading_validation.transition_score import dump,sha

METHODS=("native","uniform","recency","trained_unresolved")
CONDITIONS=("first","rementioned","neutral","new")


def describe(d,draw):
    d=np.asarray(d);boot=d[draw].mean(1);n=int((d!=0).sum());win=int((d>0).sum())
    return {"mean":float(d.mean()),"family_interval":np.quantile(boot,[.025,.975]).tolist(),
            "median":float(np.median(d)),"positive_families":win,"negative_families":int((d<0).sum()),"exact_ties":int((d==0).sum()),
            "two_sided_sign_test":float(binomtest(win,n,.5).pvalue) if n else 1.}


def main(folder):
    seal=json.loads((DEST/"exposure-seal.json").read_text());receipt=json.loads((folder/"receipt.json").read_text());file=folder/"goal-accessibility.jsonl"
    if receipt["source_sha256"]!=sha(INPUT) or seal["input_sha256"]!=sha(INPUT) or seal["protocol_sha256"]!=sha(PLAN) or sha(file)!=receipt["forecast_sha256"] or receipt["native_forwards"]!=96:
        raise ValueError("The original goal narratives, predictions or protocol changed")
    forecasts={r["id"]:r for r in map(json.loads,file.read_text().splitlines())}
    if len(forecasts)!=96:raise ValueError("Three variants must be present in all 32 families")
    res={};max_prefix_error=0.
    data={};families=seal["families"]
    for family,s in families.items():
        names=[s["probe"],*s["control_words"]];records={k:forecasts[v["id"]] for k,v in s["variants"].items()}
        if len({row["prefix_token_sha256"] for row in records.values()})!=1:raise ValueError("Changing a future goal changed the common prefix tokens")
        raw={}
        for subset in ("all","introduction"):
            for method in METHODS:
                initial=np.array([[v["first"][word][subset][method] for word in names] for v in records.values()])
                max_prefix_error=max(max_prefix_error,float(np.max(np.ptp(initial,axis=0))))
                raw[subset,method]=np.stack([initial.mean(0),*[np.array([records[k]["last"][word][subset][method] for word in names]) for k in CONDITIONS[1:]]])
        data[family]=raw
    if max_prefix_error>1e-4:raise ValueError(f"Identical common prefix discrepancy: {max_prefix_error}")
    rng=np.random.default_rng(94301);B=rng.integers(0,len(families),size=(10000,len(families)))
    contrasts={"neutral_minus_new":(2,3),"rementioned_minus_neutral":(1,2),"rementioned_minus_initial":(1,0),"initial_minus_neutral":(0,2)}
    for subset in ("all","introduction"):
        res[subset]={}
        for target in ("probe","probe_minus_control_words"):
            aligned={}
            for name in METHODS:
                cube=np.stack([v[subset,name] for v in data.values()])
                aligned[name]=cube[:,:,0]-(cube[:,:,1:].mean(2) if target=="probe_minus_control_words" else 0)
            results={}
            for name,mat in aligned.items():
                results[name]={"condition_means":dict(zip(CONDITIONS,mat.mean(0).tolist(),strict=True))}
                for label,(a,b) in contrasts.items():results[name][label]=describe(mat[:,a]-mat[:,b],B)
            res[subset][target]=results
            for control in ("uniform","recency"):
                a,b=contrasts["neutral_minus_new"];diff=aligned["native"]-aligned[control]
                results["native_minus_"+control]={"neutral_minus_new":describe(diff[:,a]-diff[:,b],B)}
    rt=seal["aggregate_human_rt_ms"];original={}
    for name,contrast,F1,F2 in (("new_minus_neutral",rt["new"]-rt["neutral"],28.15,20.86),
                                 ("neutral_minus_rementioned",rt["neutral"]-rt["rementioned"],75.68,57.93),
                                 ("new_minus_rementioned",rt["new"]-rt["rementioned"],196.14,148.33)):
        def ci(f,df):
            margin=t.ppf(.975,df)*contrast/np.sqrt(f)
            return [float(contrast-margin),float(contrast+margin)]
        original[name]={"difference_ms":contrast,"participant_F_1_83":F1,"item_F_1_31":F2,"approximate_participant_CI_ms":ci(F1,83),"approximate_item_CI_ms":ci(F2,31)}
    source_counts={s:sum(v["variants"][s]["intervening_primary_mentions"]>0 for v in families.values()) for s in CONDITIONS[1:]}
    fn=DEST/"model-item-scores.npz";np.savez_compressed(fn,family=np.array(list(families)),**{k[0]+"."+k[1]:np.stack([v[k] for v in data.values()]) for k in next(iter(data.values())).keys()})
    result={"protocol_sha256":sha(PLAN),"source_seal_sha256":sha(DEST/"exposure-seal.json"),"receipt":receipt,
            "source_item_forecasts_sha256":sha(fn),"same_prefix_max_absolute_difference":max_prefix_error,
            "eligible_families":len(families),"intervening_exact_probe_rementions_by_family":source_counts,
            "original_reported_human_ms":rt,"reported_human_contrasts":original,"model_family_measurements":res}
    dump(DEST/"scores.json",result)
    for k in ("probe","probe_minus_control_words"):
        print(k,{m:v["neutral_minus_new"] for m,v in res["all"][k].items()},flush=True)


if __name__=="__main__":
    p=ArgumentParser(description=__doc__);p.add_argument("folder",type=Path)
    main(p.parse_args().folder)
