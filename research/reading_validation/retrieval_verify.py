"""Certify causal features, registered controls and outcome coordinates without native labels."""

from pathlib import Path
from tempfile import TemporaryDirectory
import json
import re

import numpy as np
from scipy.optimize import check_grad

from research.reading_validation.memory_data import sha
from research.reading_validation.retrieval_calibration import first_log_odds
from research.reading_validation.retrieval_data import DATA,RESULT
from research.reading_validation.retrieval_features import Page
from research.reading_validation.retrieval_source_score import news_results,source_results


def run():
    rng=np.random.default_rng(422701)
    sample="The first river is wide. A steep bank. The water does not freeze."
    spans=[m.span() for m in re.finditer(r"\S+",sample)]
    g=np.repeat([0,1,2],[5,3,5])
    n=len(spans)
    if len(g)!=n:
        raise ValueError("Unexpected fixture coordinates")
    coords=np.column_stack((spans,g,np.arange(n),np.arange(n),np.ones(n,int)))
    state=rng.normal(size=(n,12))
    attention=rng.random((n,3,2))
    signal=rng.normal(size=(n,3))
    signal[:,2]=1
    whole=Page(sample,coords,state,attention,signal)
    check=Page(sample[:spans[-3][1]],coords[:-2],state[:-2],attention[:-2],signal[:-2])
    a=whole.pairs(2)[:-2]
    b=check.pairs(2)
    prefix=float(abs(a-b).max())
    x=rng.normal(size=(5,3))
    w=rng.normal(size=3)*.2
    gradient=check_grad(lambda v:first_log_odds(v,x)[0],lambda v:first_log_odds(v,x)[1],w)
    if prefix>2e-6 or gradient>2e-6:
        raise ArithmeticError("First-report derivative or causal prefix comparison failed")
    materials=json.loads((DATA/"mapping.json").read_bytes())
    docs={r["id"]:r for r in map(json.loads,(DATA/"packet.jsonl").read_text().splitlines())}
    dense={}
    for key,doc in materials["news"].items():
        row=docs[doc]
        table=[]
        for j,(a,b) in enumerate(row["spans"]):
            for m in re.compile(r"\S+").finditer(row["text"],a,b):
                table.append([*m.span(),0.,0.,0.,0.,0.,j,*([j-1]*4)])
        dense[key]=np.array(table)
    tests=[]
    for row in json.loads((DATA/"training.json").read_bytes()):
        if row["split"]=="train":
            continue
        for k in row["ids"]:
            values={"conflict":0.,"source":[1.]*(1+row["n"])}
            tests.append(dict(id=k,methods={m:values for m in ("semantic","lexical")}))
    checked=source_results(tests)
    with TemporaryDirectory() as t:
        p=Path(t)
        np.savez_compressed(p/"news.npz",**dense)
        d=news_results(p)
    out=dict(fixture="Deterministic controls and geometry only. No model performance estimates.",
             source_sha256=sha(DATA/"packet.jsonl"),prefix_max_absolute_error=prefix,
             first_report_max_gradient_error=gradient,news_marks=d["inventory"],
             source_test_triples=checked["test"]["strict_rank"]["groups"])
    (RESULT/"independent-preflight.json").write_text(json.dumps(out,indent=2)+"\n")
    print(out)


if __name__=="__main__":
    run()
