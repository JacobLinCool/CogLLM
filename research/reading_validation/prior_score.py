"""Assess the sample efficiency of a frozen three-study difficulty direction."""

import argparse
import json
from pathlib import Path

import numpy as np

from research.reading_validation.prior_data import DOMAINS, OUT
from research.reading_validation.joint_data import digest, read
from research.reading_validation.prior_fit import grades


def interval(a,b):
    return dict(estimate=float(a),interval95=np.quantile(b,[.025,.975]).tolist())


def run(folder, output):
    seal=read(folder/"seal.json")
    if any(digest(folder/k)!=v for k,v in seal["files"].items()):
        raise ValueError("Forecast archive changed before scoring")
    rows=read(folder/"test-items.json")
    rng=np.random.default_rng(71306)
    bank=np.load(folder/"predictions.npz",allow_pickle=False)
    output_by_domain={}
    diffs={}
    primary={}
    for domain in DOMAINS:
        cases=rows[domain]
        ratings=grades(domain,"test",cases)
        y=ratings[:,:,None]>np.arange(1,6)
        groups,idx=np.unique([r["group"] for r in cases],return_inverse=True)
        draw=rng.multinomial(len(groups),np.full(len(groups),1/len(groups)),size=10000)
        sizes=np.bincount(idx)
        n=draw@sizes
        vals={}
        means={}
        for key in seal["models"]:
            d,budget,design,model=key.split(".")
            if d!=domain:
                continue
            p=bank[key]
            error=((p[:,None,:]-y)**2).mean(axis=(1,2))
            vals[key]=error
            predicted=1+p.sum(axis=1)
            means[key]=dict(rps=float(error.mean()),mae_vs_educator=float(np.abs(predicted[:,None]-ratings).mean()),
                            mean_predicted=float(predicted.mean()))
        comparisons={}
        for key,loss in vals.items():
            d,budget,design,model=key.split(".")
            if design!="qwen" or model!="source_prior":
                continue
            for comp in ("qwen.matched","qwen.standard","qwen.source_only","surface.source_prior"):
                diff=loss-vals[f"{d}.{budget}.{comp}"]
                bootstrap=draw@np.bincount(idx,weights=diff)/n
                comparisons[f"{budget}.{comp}"]=interval(diff.mean(),bootstrap)
                if int(budget)==64 and comp=="qwen.matched":
                    primary[d]=diff.mean()
                    diffs[d]=bootstrap
        output_by_domain[domain]=dict(n_test=len(cases),n_groups=len(groups),mean_human_score=float(ratings.mean()),
                                      educator_mae=float(np.abs(np.diff(ratings,axis=1)).mean()),
                                      agreement=float(np.mean(ratings[:,0]==ratings[:,1])),
                                      methods=means,paired_rps_difference=comparisons)
    result=dict(domains=output_by_domain,primary_equal_corpus_64_sentences=interval(
                np.mean(list(primary.values())),np.mean(list(diffs.values()),axis=0)),
                seal_sha256=digest(folder/"seal.json"))
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps(result["primary_equal_corpus_64_sentences"]))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--folder",type=Path,default=OUT/"forecast")
    p.add_argument("--output",type=Path,default=OUT/"scores.json")
    args=p.parse_args()
    run(args.folder,args.output)
