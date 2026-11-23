"""Fit paired evidence readouts using individual workers' source-sentence choices."""

import argparse
from collections import defaultdict
from itertools import combinations
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from research.reading_validation.event_coordinate_audit import copied
from research.reading_validation.event_data import OUTPUT, align, source
from research.reading_validation.joint_data import digest, lines, read


PAIRS=list(combinations(range(5),2))
SCHEMES=("documented","zero_based","text_anchored")
DIRECTIONS=("later_explains_earlier","earlier_explains_later")


def annotations(role,keys):
    idx={k:i for i,k in enumerate(keys)}
    shape=(len(keys),10,2,2)
    obs={s:np.zeros(shape,dtype=np.int32) for s in SCHEMES}
    idx_pair={p:i for i,p in enumerate(PAIRS)}
    audited=read(OUTPUT/"inputs.json")
    if digest(source(role))!=audited["source_files"][role]:
        raise ValueError("Source release changed after registration")
    for row in lines(source(role)):
        a=align(row)
        if a is None or a[0] not in idx:
            continue
        doc,anchor=idx[a[0]],a[3]
        raw=row["helpful_sentences"]
        if not isinstance(raw,list) or any(type(j) is not int or not 1<=j<=5 for j in raw):
            raise ValueError("Invalid source sentence code")
        if 5 in raw:
            # The text-anchored arm can handle code 5. For the zero-based
            # sensitivity record the vote as unidentified when that code occurs.
            zero=None
        else:
            zero=set(raw) if raw else None
        matched=copied(row,anchor)
        grounded={matched} if matched is not None and raw[0] in (matched,matched+1) else None
        for scheme,support in (("documented",{v-1 for v in raw}),("zero_based",zero),("text_anchored",grounded)):
            if support is None:
                continue
            for j in range(5):
                if j==anchor or (j==0 and scheme=="zero_based"):
                    continue
                p=idx_pair[tuple(sorted((j,anchor)))]
                direction=int(j<anchor)
                obs[scheme][doc,p,direction]+=np.asarray([int(j in support),1])
    return obs


def fit_binary(x,votes):
    x=np.asarray(x,dtype=np.float64)
    hits,total=votes.T
    ok=total>0
    design=x[ok];count=total[ok];hit=hits[ok]
    weight=count/count.sum()
    mean=weight@design
    scale=np.sqrt(weight@((design-mean)**2))
    scale[scale<1e-8]=1.
    design=(design-mean)/scale
    mass=hit/count.sum()
    def loss(v):
        z=design@v[1:]+v[0]
        error=expit(z)*weight-mass
        objective=np.dot(weight,np.logaddexp(0,z))-mass@z+.01*(v[1:]@v[1:])
        return objective,np.r_[error.sum(),design.T@error+.02*v[1:]]
    fitted=minimize(loss,np.zeros(design.shape[1]+1),jac=True,method="L-BFGS-B",
                    options=dict(maxiter=2000,gtol=1e-8,ftol=1e-15,maxls=60))
    objective,grad=loss(fitted.x)
    if not fitted.success or abs(grad).max()>1e-6:
        raise ArithmeticError(f"Event head did not converge: {fitted.message}")
    slope=fitted.x[1:]/scale
    weights=np.r_[fitted.x[0]-mean@slope,slope]
    audit=dict(pairs=int(ok.sum()),votes=int(count.sum()),selected=int(hit.sum()),
               iterations=int(fitted.nit),max_gradient=float(abs(grad).max()),loss=float(objective))
    return weights,audit


def run(extraction,dest):
    if dest.exists():
        raise FileExistsError(dest)
    seal=read(extraction/"receipt.json")
    meta=read(OUTPUT/"inputs.json")
    if any(digest(extraction/k)!=v for k,v in seal["files"].items()) or seal["input_sha256"]!=meta["packet_sha256"]:
        raise ValueError("Event extraction does not match its grade-free input")
    roles=read(OUTPUT/"splits.json")
    ids=read(extraction/"ids.json")
    table={k:i for i,k in enumerate(ids)}
    bank=np.load(extraction/"features.npy",mmap_mode="r")
    baseline={}; fits={}; predictions={}
    tr=roles["train"]; te=roles["test"]
    votes=annotations("train",tr)
    x=bank[[table[k] for k in tr]].reshape(-1,bank.shape[-1])
    z=bank[[table[k] for k in te]].reshape(-1,bank.shape[-1])
    for scheme,tensor in votes.items():
        # Learn the denser prior-support direction first from actual earlier evidence votes.
        source_train=source_test=None
        for d in (1,0):
            y=tensor[:,:,d,:].reshape(-1,2)
            for method,width in (("position",5),("lexical",11),("context",x.shape[1]),("reverse_prior",x.shape[1])):
                if method=="reverse_prior" and d==1:
                    continue
                xx=x[:,:width]; zz=z[:,:width]
                if method=="reverse_prior":
                    xx=np.column_stack((xx,source_train))
                    zz=np.column_stack((zz,source_test))
                head,audit=fit_binary(xx,y)
                pred=zz@head[1:]+head[0]
                if d==1 and method=="context":
                    source_train=xx@head[1:]+head[0]
                    source_test=pred
                key=scheme+"."+DIRECTIONS[d]+"."+method
                fits[key]=audit
                baseline[key]=head
                predictions[key]=pred.reshape(len(te),10)
    predictions["overlap"]=z[:,5].reshape(len(te),10)
    predictions["distance"]=-z[:,2].reshape(len(te),10)
    dest.mkdir(parents=True)
    np.savez_compressed(dest/"heads.npz",**baseline)
    np.savez_compressed(dest/"forecasts.npz",id=te,**predictions)
    doc=dict(extraction_sha256=digest(extraction/"receipt.json"),data_sha256=digest(OUTPUT/"inputs.json"),
             coordinate_amendment_sha256=digest(Path(__file__).with_name("event-coordinate-amendment.md")),
             coordinate_audit_sha256=digest(OUTPUT/"coordinate-audit.json"),fits=fits,
             files={k:digest(dest/k) for k in ("heads.npz","forecasts.npz")})
    (dest/"seal.json").write_text(json.dumps(doc,indent=2)+"\n")
    print("Sealed",len(fits),"fitted relations;",len(te),"held-out stories")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--extraction",type=Path,required=True)
    p.add_argument("--dest",type=Path,default=OUTPUT/"forecast")
    a=p.parse_args()
    run(a.extraction,a.dest)
