"""Repeat the sealed few-label comparison over independently drawn calibration material."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.special import expit

from research.reading_validation.joint_data import digest, read, surface
from research.reading_validation.joint_fit import Basis
from research.reading_validation.prior_data import BASE, DOMAINS, OUT
from research.reading_validation.prior_fit import fit, grades
from research.reading_validation.rewrite_forecast import ROOT, extracted


def run(extraction, dest):
    if dest.exists():
        raise FileExistsError(dest)
    keys,hidden,texts,_=extracted(extraction, BASE/"prior-stimuli.jsonl")
    lookup={k:i for i,k in enumerate(keys)}
    s=np.stack([surface(t,"en") for t in texts])
    with np.load(ROOT/"models.npz",allow_pickle=False) as m:
        basis=Basis(**{k:m["zero.basis."+k] for k in Basis.__dataclass_fields__})
        blocks=(SimpleNamespace(surface=s[i:i+256],hidden=hidden[i:i+256]) for i in range(0,len(s),256))
        x=np.vstack([basis.transform(b,True) for b in blocks])
        z=x@m["zero.qwen.slope"]
    material=read(OUT/"materials.json")
    forecasts={}
    samples={}
    domain_rows={}
    for domain in DOMAINS:
        tr=[r for r in material if r["domain"]==domain and r["split"]=="train"]
        te=[r for r in material if r["domain"]==domain and r["split"]=="test"]
        domain_rows[domain]=te
        indices=np.asarray([lookup[r["id"]] for r in tr])
        test=[lookup[r["id"]] for r in te]
        y=(grades(domain,"train",tr)[:,:,None]>np.arange(1,6)).mean(axis=1)
        for seed in range(32):
            rng=np.random.default_rng(1008+seed)
            select=rng.choice(len(tr),64,replace=False)
            a=indices[select]
            samples[f"{domain}.{seed}"]=[tr[i]["id"] for i in select]
            for name,transfer in (("target",False),("prior",True)):
                head=fit(x[a],y[select],z[a],.01+4/64,flexible=True,transfer=transfer)
                linear=x[test]@head["slope"]+z[test]*head["coefficient"]
                forecasts[f"{domain}.{seed}.{name}"]=expit(linear[:,None]+head["cut"])
    dest.mkdir()
    np.savez_compressed(dest/"predictions.npz",**forecasts)
    (dest/"train_splits.json").write_text(json.dumps(samples)+"\n")
    seal={k:digest(dest/k) for k in ("predictions.npz","train_splits.json")}
    (dest/"seal.json").write_text(json.dumps(seal,indent=2)+"\n")
    rng=np.random.default_rng(44373)
    means=[]
    by_domain={}
    draws=[]
    for domain,rows in domain_rows.items():
        truth=grades(domain,"test",rows)[:,:,None]>np.arange(1,6)
        diff=[]
        for seed in range(32):
            errors=[((forecasts[f"{domain}.{seed}.{m}"][:,None,:]-truth)**2).mean(axis=(1,2)) for m in ("prior","target")]
            diff.append(errors[0]-errors[1])
        delta=np.stack(diff);avg=delta.mean(axis=1)
        means.append(delta.mean())
        groups,g=np.unique([r["group"] for r in rows],return_inverse=True)
        vals=np.stack([np.bincount(g,weights=v) for v in delta])
        sizes=np.bincount(g)
        rw=rng.multinomial(32,np.full(32,1/32),size=10000)/32
        ww=rng.multinomial(len(groups),np.full(len(groups),1/len(groups)),size=10000)
        stat=((rw@vals)*ww).sum(axis=1)/(ww@sizes)
        draws.append(stat)
        by_domain[domain]=dict(gains=int((avg<0).sum()),paired_rps_by_draw=avg.tolist(),
             mean=float(delta.mean()),interval95=np.quantile(stat,[.025,.975]).tolist())
    result=dict(corpora=by_domain,equal_corpus_difference=float(np.mean(means)),
                interval95=np.quantile(np.mean(draws,axis=0),[.025,.975]).tolist(),training_draws=32,
                independent_of_original_training_draws=True,test_set_previously_scored=True)
    (dest/"scores.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="corpora"}))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--extraction",type=Path,required=True)
    p.add_argument("--dest",type=Path,default=OUT/"replications")
    a=p.parse_args()
    run(a.extraction,a.dest)
