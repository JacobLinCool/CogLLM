"""Grade complete question lifetimes and unprompted source-and-closing-line proposals."""

import argparse
from collections import defaultdict
import json
from math import log
from pathlib import Path

import numpy as np

from research.core_features.readback_score import average_precision
from research.reading_validation.memory_data import sha
from research.reading_validation.narrative_course_score import identity,count_outcomes
from research.reading_validation.narrative_survival_fit import ARMS,OUT,pages


def observation_loss(page,hz):
    rows=defaultdict(list)
    for s in page.records:
        survival=np.log1p(-hz[s.source+1:s.definite+1,s.source].clip(1e-12,1-1e-12)).sum()
        if s.upper is not None:
            remain=np.log1p(-hz[s.definite+1:s.upper+1,s.source].clip(1e-12,1-1e-12)).sum()
            survival+=log(max(-np.expm1(remain),1e-12))
        rows[s.team].append(-survival)
    return float(np.mean([np.mean(r) for r in rows.values()]))


def uncued(page,score):
    out=[]
    for team in page.story.teams:
        eligible=np.tril(np.ones_like(score,dtype=bool),k=-1)
        gold=np.zeros_like(eligible)
        for spell in page.records:
            if spell.team!=team:
                continue
            if spell.upper==spell.definite+1:
                gold[spell.upper,spell.source]=True
            else:
                upper=page.n if spell.upper is None else spell.upper+1
                eligible[spell.definite+1:upper,spell.source]=False
        eligible|=gold
        pred=score[eligible];y=gold[eligible]
        if not y.any():
            raise ValueError("No observed exact question closure in this source")
        first=pred==pred.max()
        out.append(dict(ap=average_precision(pred,y),prevalence=float(y.mean()),top1=float(y[first].mean()),
                        pairs=int(len(y)),exact_source_closures=int(y.sum())))
    return {m:float(np.mean([r[m] for r in out])) for m in out[0]}


def run(archive,old):
    forecast=OUT/"forecast"
    seal=json.loads((forecast/"seal.json").read_bytes())
    if sha(archive/"receipt.json")!=seal["archive"] or any(sha(forecast/f)!=v for f,v in seal["files"].items()):
        raise ValueError("Question lifetime forecasts changed after being sealed")
    stories=pages(archive,old)
    result={}
    with np.load(forecast/"predictions.npz",allow_pickle=False) as data:
        for name,p in stories.items():
            by_arm={}
            for arm in ARMS:
                key=name+"."+arm
                hz=data[key+".hazard"];mass=data[key+".event"];stock=data[key+".stock"]
                # Known-open candidate comparison uses conditional closing; unprompted
                # earlier-source ranking must additionally predict the origin and its survival.
                both=identity(p.story,{"conditional":hz,"unprompted_source":mass})["methods"]
                counts=count_outcomes(p.story,{"count":stock})["count"]
                by_arm[arm]=dict(which_active_issue=both["conditional"]["paired_accuracy"],
                                 locate_earlier_line_given_closure=both["unprompted_source"]["top1_any_earlier"],
                                 interval_lifetime_nll=observation_loss(p,hz),count_interval_mse=counts["interval_mse"],
                                 count_change_direction=counts["direction"],**uncued(p,mass))
            result[name]=by_arm
    equal={m:{k:float(np.mean([v[m][k] for v in result.values()])) for k in result[next(iter(result))][m]} for m in ARMS}
    out=dict(forecast=sha(forecast/"seal.json"),stories=result,equal_story=equal)
    (OUT/"scores.json").write_text(json.dumps(out,indent=2,allow_nan=False)+"\n")
    print(json.dumps(equal,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("archive",type=Path);p.add_argument("old",type=Path)
    run(**vars(p.parse_args()))
