"""Independently reconstruct crossed intervals from individual lay-reader comparisons."""

from collections import defaultdict
import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from research.reading_validation.local_forecast import digest,rows
from research.reading_validation.relay_data import DATA,OUT
from research.reading_validation.relay_score import forecast,load_features


def ratio(v,w):
    denominator=w.sum(-1)
    return np.divide((np.nan_to_num(v)*w).sum(-1),denominator,out=np.full(len(w),np.nan),where=denominator>0)


def explicit(sn,records,values):
    exclusive=sn.startswith("definition_over_example")
    within=sn.endswith("within_same")
    articles=defaultdict(lambda:defaultdict(list))
    for i,a in enumerate(records):
        for j in range(i):
            b=records[j]
            if (a["article"]!=b["article"] or a["definition"]==b["definition"] or
                (exclusive and (a["definition"]==a["example"] or b["definition"]==b["example"]))):
                continue
            if within:
                if a["reader"]!=b["reader"] or a["familiarity"]!=b["familiarity"]:
                    continue
            elif a["term"]!=b["term"] or a["familiarity"]==b["familiarity"] or a["reader"]==b["reader"]:
                continue
            signed=(values[i]-values[j])*(int(a["definition"])-int(b["definition"]))
            hit=.5 if abs(signed)<=1e-10 else float(signed>0)
            category=a["reader"] if within else a["term"]
            articles[a["article"]][category].append((a["reader"],b["reader"],hit))
    readers=sorted({p for cells in articles.values() for pairs in cells.values() for a,b,_ in pairs for p in (a,b)})
    names=sorted(articles);n=len(readers);m=len(names)
    rng=np.random.default_rng(611290); rw=[np.ones((1,n))];aw=[np.ones((1,m))]
    for _ in range(100):
        rw.append(rng.multinomial(n,np.full(n,1/n),100));aw.append(rng.multinomial(m,np.full(m,1/m),100))
    r,a=np.vstack(rw),np.vstack(aw)
    rs={v:i for i,v in enumerate(readers)}
    result=[]
    for article in names:
        cell_scores=[]
        cell_weights=[]
        for cell,pairs in articles[article].items():
            if within:
                cell_scores.append(np.full(len(r),np.mean([v for _,_,v in pairs])))
                cell_weights.append(r[:,rs[cell]])
            else:
                weights=np.stack([r[:,rs[who]]*r[:,rs[other]] for who,other,_ in pairs],axis=1)
                average=ratio(np.array([v for _,_,v in pairs]),weights)
                cell_scores.append(average);cell_weights.append(np.isfinite(average))
        result.append(ratio(np.stack(cell_scores,axis=1),np.stack(cell_weights,axis=1)))
    all_scores=np.stack(result,axis=1)
    draws=ratio(all_scores,a*np.isfinite(all_scores))
    return np.r_[draws[0],np.nanquantile(draws[1:],[.025,.975])]


def verify(archive):
    matrix=load_features(archive)
    coords=json.loads((DATA/"coordinates.json").read_bytes())
    records=[r for r in rows(DATA/"judgments.jsonl") if r["term"] in matrix and r["regular"] and coords[r["term"]]["tokens"]<=1024]
    score=OUT/"scores.json"; locked=json.loads(score.read_bytes())["subsets"]["native"]["endpoints"]
    errors=[]
    for endpoint in ("definition_over_example.within_same","definition.between_different"):
        methods,values=forecast(matrix,records,endpoint.split(".")[0])
        for m in ("article_prior","lexical_prior"):
            predicted=explicit(endpoint,records,values[:,methods.index(m)])
            s=locked[endpoint]["scores"][m]
            errors.append(abs(predicted-np.r_[s["estimate"],s["interval95"]]).max())
    if max(errors)>1e-10:
        raise ArithmeticError(f"Independent crossed resampling differs: {errors}")
    receipt=dict(primary_and_between_reader_intervals_recomputed=True,estimates_and_quantiles=3*len(errors),
                 max_absolute_error=float(max(errors)),source=digest(score))
    (OUT/"independent-check.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(receipt)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("archive",type=Path)
    verify(p.parse_args().archive)
