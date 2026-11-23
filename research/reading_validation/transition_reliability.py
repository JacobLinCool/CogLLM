"""Post-primary descriptive analysis of independent halves of public coherence judgments."""
import json

import numpy as np
import pandas as pd

from research.reading_validation.transition_outcomes import load
from research.reading_validation.transition_prepare import OUT
from research.reading_validation.transition_score import dump


def run():
    ratings,_=load()
    trials=ratings[ratings.cond.isin(list("abcd"))].rename(columns={"trial_num":"item","cond":"condition"})
    ids=pd.DataFrame(json.loads((OUT/"stimuli.json").read_text())["rows"])
    joined=ids[ids.split.eq("holdout")].merge(trials,on=["item","condition"],validate="1:1")
    individual=joined.filter(regex=r"^S\d+$").to_numpy()
    rng=np.random.default_rng(4821)
    votes=[row[np.isfinite(row)] for row in individual]
    full=np.array([r.mean() for r in votes])
    if np.max(np.abs(full-joined.trial_mean_rating.to_numpy()))>.00501:
        raise ValueError("Original means do not reproduce published ratings at two decimals")
    random_splits=[]
    for _ in range(1000):
        left=[];right=[]
        for r in votes:
            a=rng.permutation(r)
            n=len(a)//2
            left.append(a[:n].mean());right.append(a[n:].mean())
        left=np.asarray(left);right=np.asarray(right)
        random_splits.append([np.corrcoef(left,right)[0,1],np.sqrt(((left-right)**2).mean())])
    a=np.array(random_splits)
    result={"analysis":"descriptive after primary results",
            "procedure":"split ratings within each original item-condition cell at random, 1,000 repetitions; index columns are not IDs across lists",
            "passages":len(votes),"ratings":int(sum(map(len,votes))),"votes_per_passage_range":[min(map(len,votes)),max(map(len,votes))],
            "median_half_to_half_correlation":float(np.median(a[:,0])),
            "median_half_to_half_rmse_1_to_5":float(np.median(a[:,1])),
            "range_from_rand_split_not_sampling_uncertainty":np.quantile(a,[.025,.975],axis=0).tolist()}
    dump(OUT/"rating-reliability.json",result)
    return result


if __name__=="__main__":
    print(run())
