"""Adapt a fixed cross-study direction to a little target-specific ordinal evidence."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from research.reading_validation.external_input import digest_string
from research.reading_validation.joint_data import digest, read, surface
from research.reading_validation.joint_fit import Basis
from research.reading_validation.prior_data import BASE, DOMAINS, OUT, read_source
from research.reading_validation.rewrite_forecast import ROOT, extracted


def fit(x, y, prior, ridge, *, flexible, transfer):
    n,width=x.shape
    d=width if flexible else 0
    shift=d+int(transfer)
    if transfer and np.std(prior) < 1e-8:
        raise ValueError("Source direction is constant on the target training rows")
    scale=float(np.std(prior)) if transfer else 1.
    src=prior/scale
    def objective(v):
        linear=np.zeros(n)
        if flexible:
            linear+=x@v[:d]
        if transfer:
            linear+=src*v[d]
        logits=linear[:,None]+v[shift:]
        # A symmetric pseudo-observation keeps all intercepts finite on a small calibration sample.
        base=v[shift:]
        loss=(np.logaddexp(0,logits)-y*logits).mean()+(np.logaddexp(0,base)-.5*base).mean()/n
        residual=(expit(logits)-y)/(n*5)
        grad=np.zeros_like(v)
        grad[shift:]=residual.sum(axis=0)+(expit(base)-.5)/(n*5)
        total=residual.sum(axis=1)
        if flexible:
            loss+=.5*ridge*(v[:d]@v[:d])
            grad[:d]=x.T@total+ridge*v[:d]
        if transfer:
            loss+=.5*.005*v[d]**2
            grad[d]=total@src+.005*v[d]
        return loss,grad
    initial=np.zeros(shift+5)
    bounds=[(None,None)]*len(initial)
    if transfer:
        bounds[d]=(0,None)
    opt=minimize(objective,initial,jac=True,bounds=bounds,method="L-BFGS-B",
                 options=dict(maxiter=3000,ftol=1e-15,gtol=1e-9,maxls=60))
    _,g=objective(opt.x)
    if transfer and opt.x[d]<=1e-10:
        g[d]=min(g[d],0.)
    if not opt.success or abs(g).max()>1e-6:
        raise ArithmeticError(f"Nonconverged ordinal adaptation: {opt.message}")
    return dict(slope=opt.x[:d].copy() if flexible else np.zeros(width),
                coefficient=float(opt.x[d]/scale) if transfer else 0.,
                standardized_coefficient=float(opt.x[d]) if transfer else 0.,
                cut=opt.x[shift:].copy(),gradient=float(abs(g).max()),iterations=int(opt.nit))


def grades(domain, part, indices):
    file=BASE/f"cefr/{domain}-{part}.tsv"
    expected=read(OUT/"inputs.json")["source"][file.name]
    if digest(file)!=expected:
        raise ValueError("CEFR-SP source changed")
    original={i:(digest_string(t),int(a),int(b)) for i,(t,a,b) in read_source(file)}
    values=[]
    for r in indices:
        uid,a,b=original[r["row"]]
        if r["id"]!=uid or not (1<=a<=6 and 1<=b<=6):
            raise ValueError("Invalid rating or source match")
        values.append((a,b))
    return np.asarray(values)


def run(extraction, dest):
    if dest.exists():
        raise FileExistsError(dest)
    receipt=read(OUT/"inputs.json")
    if digest(OUT/"materials.json")!=receipt["materials_sha256"] or receipt["packet_sha256"]!=digest(BASE/"prior-stimuli.jsonl"):
        raise ValueError("CEFR-SP data seal changed")
    keys,hidden,texts,meta=extracted(extraction,BASE/"prior-stimuli.jsonl")
    idx={k:i for i,k in enumerate(keys)}
    if digest(ROOT/"models.npz")!=read(ROOT/"seal.json")["files"]["models.npz"]:
        raise ValueError("Source direction differs from initial cross-study experiment")
    studies=read(OUT/"materials.json")
    sample=SimpleNamespace(hidden=hidden,surface=np.stack([surface(t,"en") for t in texts]))
    archive={}
    convergence={}
    positions={}
    with np.load(ROOT/"models.npz",allow_pickle=False) as source:
        basis=Basis(**{k:source["zero.basis."+k] for k in Basis.__dataclass_fields__})
        for design in ("qwen","surface"):
            batches=(SimpleNamespace(surface=sample.surface[i:i+256],hidden=sample.hidden[i:i+256])
                     for i in range(0,len(keys),256))
            features=np.vstack([basis.transform(batch,design=="qwen") for batch in batches])
            direction=features@source["zero."+design+".slope"]
            for domain in DOMAINS:
                split=lambda p:[r for r in studies if r["domain"]==domain and r["split"]==p]
                test=split("test")
                positions[domain]=test
                t_idx=[idx[r["id"]] for r in test]
                train=sorted(split("train"),key=lambda r:digest_string("cogllm.cefr.nested.71306."+r["id"]))
                target=grades(domain,"train",train)
                for budget in (64,256,len(train)):
                    t=train[:budget]
                    ids=[idx[r["id"]] for r in t]
                    y=(target[:budget,:,None]>np.arange(1,6)).mean(axis=1)
                    for name,flex,transfer,ridge in (("matched",True,False,.01+4/budget),
                            ("source_prior",True,True,.01+4/budget),("standard",True,False,.01),
                            ("source_only",False,True,.01)):
                        key=f"{domain}.{budget}.{design}.{name}"
                        f=fit(features[ids],y,direction[ids],ridge,flexible=flex,transfer=transfer)
                        z=features[t_idx]@f["slope"]+direction[t_idx]*f["coefficient"]
                        p=expit(z[:,None]+f["cut"])
                        if not np.all(np.isfinite(p)) or np.any(np.diff(p,axis=1)>1e-8):
                            raise ArithmeticError("Invalid cumulative forecast")
                        archive[key]=p
                        convergence[key]={k:v for k,v in f.items() if k not in ("cut","slope")}
                        archive[key+".weight"]=np.r_[f["slope"],f["coefficient"],f["cut"]]
    dest.mkdir()
    np.savez_compressed(dest/"predictions.npz",**archive)
    (dest/"test-items.json").write_text(json.dumps(positions,indent=1)+"\n")
    lock=dict(n_hidden=len(hidden),extraction_sha256=digest(extraction/"extraction.json"),data_sha256=digest(OUT/"inputs.json"),
              revision=meta["revision"],models=convergence,
              files={n:digest(dest/n) for n in ("predictions.npz","test-items.json")})
    (dest/"seal.json").write_text(json.dumps(lock,indent=2)+"\n")
    print("Sealed",len(convergence),"source and comparator fits")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--extraction",type=Path,required=True)
    p.add_argument("--dest",type=Path,required=True)
    a=p.parse_args()
    run(a.extraction,a.dest)
