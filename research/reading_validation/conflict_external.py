"""Transfer the sealed contradiction direction to readers' incremental reason codes."""

import argparse
import json
from pathlib import Path

import numpy as np

from research.reading_validation.conflict_fit import classification, lexical
from research.reading_validation.joint_data import DEFAULT_CACHE, digest, lines, read
from research.reading_validation.joint_fit import normalized_hidden
from research.reading_validation.transition_external_outcomes import votes


SOURCE = Path("research/results/reading-validation-transition-external/stimuli.json")


def forecasts(cache, fitted, extraction, dest):
    if dest.exists():
        raise FileExistsError(dest)
    meta=read(SOURCE)
    receipt=read(extraction/"receipt.json")
    packet=cache/"reading-validation/cohesion-stimuli.jsonl"
    states=extraction/"heldout_sentence_states.npz"
    if (receipt["files"][states.name]!=digest(states) or receipt["input_sha256"]!=digest(packet)
            or meta["packet_sha256"]!=digest(packet)):
        raise ValueError("CoheSentia packet or features changed")
    heads=fitted/"xnli-heads.npz"
    if digest(heads)!=read(fitted/"xnli-seal.json")[heads.name]:
        raise ValueError("Conflict weights changed")
    material={r["id"]:r for r in meta["material"] if r["split"]=="holdout" and r["source"]=="cohesentia"}
    controls={}
    for doc in lines(packet):
        text=doc["text"]
        for stop in doc["stops"]:
            key=stop["id"]
            if key in material:
                controls[key]=lexical(text[:stop["start"]],text[stop["start"]:stop["end"]])
    ids=sorted(material)
    with np.load(states,allow_pickle=False) as actual, np.load(heads,allow_pickle=False) as model:
        if sorted(actual.files)!=ids or sorted(controls)!=ids:
            raise ValueError("CoheSentia input, lexical comparator and semantic projection have different positions")
        x=normalized_hidden(np.stack([actual[k] for k in ids]))
        z=x@model["semantic"]+model["semantic.intercept"]
        surface=np.stack([controls[k] for k in ids])@model["surface"]+model["surface.intercept"]
    dest.mkdir(parents=True)
    np.savez_compressed(dest/"forecasts.npz",id=ids,surface=surface,semantic=z)
    seal=dict(files={"forecasts.npz":digest(dest/"forecasts.npz")},materials_sha256=digest(SOURCE),
              input_sha256=receipt["input_sha256"],weights_sha256=digest(heads))
    (dest/"seal.json").write_text(json.dumps(seal,indent=2)+"\n")
    return seal


def weighted_metrics(y, scores, clusters):
    # Each row represents a document bootstrap sample, with tied scores aggregated together.
    order=np.argsort(-scores,kind="stable")
    starts=np.r_[0,np.flatnonzero(np.diff(scores[order]))+1]
    positives=np.add.reduceat(clusters[:,order]*y[order],starts,axis=1)
    negatives=np.add.reduceat(clusters[:,order]*(1-y[order]),starts,axis=1)
    tp=np.cumsum(positives,axis=1)
    fp=np.cumsum(negatives,axis=1)
    valid=(tp[:,-1]>0)&(fp[:,-1]>0)
    ppv=np.divide(tp,tp+fp,out=np.zeros_like(tp,dtype=float),where=tp+fp>0)
    ap=(positives*ppv).sum(axis=1)/np.maximum(tp[:,-1],1)
    auc=(negatives*(tp-positives/2)).sum(axis=1)/np.maximum(tp[:,-1]*fp[:,-1],1)
    return {k:np.where(valid,v,np.nan) for k,v in (("auc",auc),("ap",ap))}


def score(folder,output):
    seal=read(folder/"seal.json")
    if digest(folder/"forecasts.npz")!=seal["files"]["forecasts.npz"] or digest(SOURCE)!=seal["materials_sha256"]:
        raise ValueError("Sealed CoheSentia predictions or positions changed")
    with np.load(folder/"forecasts.npz",allow_pickle=False) as f:
        ids=f["id"]
        methods={k:f[k] for k in ("surface","semantic")}
    meta={r["id"]:r for r in read(SOURCE)["material"] if r["split"]=="holdout"}
    raw=[votes(meta[k]) for k in ids]
    keep=np.array([n>0 for _,n,_,_ in raw])
    groups,idx=np.unique([meta[k]["group"] for k in ids[keep]],return_inverse=True)
    rng=np.random.default_rng(90732)
    sample=rng.multinomial(len(groups),np.full(len(groups),1/len(groups)),size=10000)
    result={}
    for code in (4,5):
        y=np.array([int(c.get(code,0)>0) for _,_,c,_ in raw])[keep]
        observed={m:classification(y,x[keep]) for m,x in methods.items()}
        draws={m:{} for m in methods}
        for m,x in methods.items():
            chunks=[weighted_metrics(y,x[keep],ss[:,idx]) for ss in np.array_split(sample,25)]
            draws[m]={k:np.concatenate([c[k] for c in chunks]) for k in ("auc","ap")}
        difference={}
        if all(observed["semantic"][k] is not None for k in ("ap","auc")):
            for k in ("ap","auc"):
                delta=draws["semantic"][k]-draws["surface"][k]
                ok=np.isfinite(delta)
                difference[k]=dict(estimate=observed["semantic"][k]-observed["surface"][k],
                                   valid_draws=int(ok.sum()),interval95=np.quantile(delta[ok],[.025,.975]).tolist())
        result[str(code)]=dict(scores=observed,paired_semantic_minus_surface=difference,
                               prevalence=float(y.mean()),positive_reader_votes=sum(c.get(code,0) for _,_,c,_ in raw))
    scored=dict(n_docs=len(groups),boundaries=int(keep.sum()),results=result,seal_sha256=digest(folder/"seal.json"),
                interpretation="At least one eligible incremental rater supplied the specified reason at this boundary")
    output.write_text(json.dumps(scored,indent=2,allow_nan=False)+"\n")
    print(json.dumps(scored))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    commands=p.add_subparsers(dest="mode",required=True)
    f=commands.add_parser("forecast")
    f.add_argument("--cache",type=Path,default=DEFAULT_CACHE)
    for key in ("fitted","extraction","dest"):
        f.add_argument("--"+key,type=Path,required=True)
    s=commands.add_parser("score")
    s.add_argument("--folder",type=Path,required=True)
    s.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.mode=="forecast":
        forecasts(a.cache,a.fitted,a.extraction,a.dest)
    else:
        score(a.folder,a.output)
