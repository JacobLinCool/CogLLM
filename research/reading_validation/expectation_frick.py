"""Fit condition-blind article trajectories from lexical expectation at every observed word."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import expit
from wordfreq import zipf_frequency

from research.reading_validation.expectation_data import OUT,WORK
from research.reading_validation.memory_data import BASE,E1,OUT as PREVIOUS,load_rows,sha
from research.reading_validation.memory_fit import calibrate,counts,design


def pools(sentences,coords,z,text):
    basic=design(sentences)
    groups={"surface":[],"surprise":[],"expectation":[]}
    where=0
    for pos,sentence in enumerate(sentences):
        if text[where:where+len(sentence)]!=sentence:
            raise ValueError("Human sentence boundary and native word coordinates disagree")
        mask=(coords[:,0]>=where)&(coords[:,1]<=where+len(sentence))&(coords[:,4]==1)
        x=z[mask];cs=coords[mask]
        if not len(x):
            raise ValueError("No eligible word for an original sentence")
        frequency=[zipf_frequency(text[a:b],"en") for a,b,*_ in cs]
        f=np.column_stack((x[:,:4],frequency));at=int(np.argmax(x[:,11]+1e3*x[:,14]))
        for name,v in (("surface",f),("surprise",np.column_stack((f,x[:,4]))),
                       ("expectation",np.column_stack((f,x[:,4:15])))):
            # The form comparator cannot use an LM-dependent word to select its own evidence.
            region=at if name=="expectation" else int(np.argmax(x[:,4])) if name=="surprise" else int(np.argmin(f[:,-1]))
            groups[name].append(np.r_[basic[pos],v.mean(0),v.max(0),v[region]])
        where+=len(sentence)+2
    return {m:np.stack(a) for m,a in groups.items()}


def run(archive,dest):
    if dest.exists():
        raise FileExistsError(dest)
    receipt=json.loads((archive/"receipt.json").read_bytes())
    if sha(archive/"features.npz")!=receipt["file"]:
        raise ValueError("Article-only lexical forecasts differ from their lock")
    family=json.loads((BASE/"memory/study-map.json").read_bytes())
    train=[r for r in family if r["study"]==1]
    observed=counts(load_rows(E1/"outcomes.jsonl"),train)
    docs=load_rows(BASE/"memory/stories.jsonl")
    mapping=json.loads((WORK/"opportunities.json").read_bytes())["frick"]
    designs={n:{} for n in ("surface","surprise","expectation")}
    with np.load(archive/"features.npz",allow_pickle=False) as data:
        for row in docs:
            k=row["id"];code=mapping[k];text="\n\n".join(row["sentences"])
            values=pools(row["sentences"],data[code+".word"],data[code+".features"],text)
            for n,design0 in values.items():
                designs[n][k]=design0
    predictions={};fits={}
    with np.load(PREVIOUS/"forecast/forecasts.npz",allow_pickle=False) as stored:
        for k in mapping:
            predictions["strict.original."+k]=stored["strict.frozen."+k]
    for fn in sorted({r["story"] for r in train}):
        for name,inputs in designs.items():
            mu,sd,coef,g=calibrate(fn,train,inputs,observed)
            fits[f"{fn}.{name}"]=dict(mu=mu.tolist(),sd=sd.tolist(),coef=coef.tolist(),gradient=g)
            for r in family:
                if r["story"]==fn:
                    k=r["id"]
                    paths=expit((inputs[k]-mu)/sd@coef[1:]+coef[0])
                    paths[0]=0.;predictions["strict."+name+"."+k]=paths
    dest.mkdir()
    np.savez_compressed(dest/"forecasts.npz",**predictions)
    (dest/"fits.json").write_text(json.dumps(fits)+"\n")
    (dest/"seal.json").write_text(json.dumps(dict(archive=sha(archive/"receipt.json"),
         models=["original",*designs],files={n:sha(dest/n) for n in ("forecasts.npz","fits.json")}),indent=2)+"\n")
    print("Sealed",len(fits),"source-family-blocked first-report fits")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("archive",type=Path);p.add_argument("--dest",type=Path,default=OUT/"frick-forecast")
    run(**vars(p.parse_args()))
