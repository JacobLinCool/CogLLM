"""Predict crowd disagreement from causal features, then lock truth-free forecasts."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize
from scipy.special import expit, logsumexp, softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from scipy.sparse import hstack

from research.reading_validation.discourse_data import BASE, OUT, PROTO, RAW, read_table
from research.reading_validation.joint_data import digest, lines, read, surface
from research.reading_validation.joint_fit import normalized_hidden

METHODS = ("forced", "free")


def collect(wanted):
    votes = {m:defaultdict(dict) for m in METHODS}
    for row in read_table("item.csv", "\t"):
        for method, key in (("forced","lev2_forced"), ("free","lev2_free_step2")):
            item = row["itemid_" + method]
            if item not in wanted:
                continue
            label = row[key].strip().lower()
            if not label or label == "na":
                continue
            worker = row["workerid_" + method]
            if not worker:
                raise ValueError("A label lacks a reader")
            old = votes[method][item].setdefault(worker, label)
            if old != label:
                raise ValueError("Incompatible double vote for a worker, item and elicitation")
    return votes


def observations(rows, votes, classes):
    index = {k:i for i,k in enumerate(classes)}
    counts = np.zeros((len(METHODS), len(rows), len(classes)), dtype=np.int32)
    for j,m in enumerate(METHODS):
        for i,r in enumerate(rows):
            for lab in votes[m].get(r["item"],{}).values():
                counts[j,i,index.get(lab,index["__other__"])] += 1
    return counts


def rates(count):
    n = count.sum(-1)
    p = count / n.clip(1)[...,None]
    agreement = (count*(count-1)).sum(-1) / (n*(n-1)).clip(1)
    return p, 1-agreement, n>=2


def lexical(passage, boundary):
    left,right=passage[:boundary].strip(), passage[boundary:].strip()
    a,b=set(left.lower().split()),set(right.lower().split())
    return np.r_[surface(left,"en"),surface(right,"en"),len(a&b)/max(1,len(a|b)),
                 len(a&b)/max(1,len(b)),np.log1p(len(left)),np.log1p(len(right))]


def transform(train_mask, raw):
    mean=raw[train_mask].mean(0)
    sd=raw[train_mask].std(0)
    return (raw-mean)/np.where(sd>1e-6,sd,1),mean,sd


def fit_masks(x, counts, output):
    p,d,mask=rates(counts)
    truth=d[:,:,None] if output=="disagreement" else p
    K=truth.shape[-1]
    Xdim=x.shape[1]
    solutions={}
    for joint,task in ((True,None),(False,0),(False,1)):
        tasks=list(range(2)) if joint else [task]
        T=len(tasks)
        # Each instrument contributes equally. Individual items receive equal weight.
        def unpack(v):
            v=v.reshape(-1,K)
            intercept=v[:T]
            slope=v[T:T+Xdim]
            delta=v[T+Xdim:].reshape(T,Xdim,K) if joint else np.zeros((T,Xdim,K))
            return intercept,slope,delta

        def objective(v):
            b,w,r=unpack(v)
            loss=.01*np.sum(w*w)+.025*np.sum(r*r)
            gw=.02*w.copy()
            gb=np.zeros_like(b)
            gr=.05*r.copy()
            for j,t in enumerate(tasks):
                keep=mask[t]
                features=x[keep]
                y=truth[t,keep]
                z=features@(w+r[j])+b[j]
                if K==1:
                    err=expit(z)-y
                    loss+=np.mean(np.logaddexp(0,z)-y*z)/T
                else:
                    err=softmax(z,axis=1)-y
                    loss+=np.mean(logsumexp(z,axis=1)-(z*y).sum(1))/T
                grad=features.T@err/(T*len(y))
                gb[j]=err.mean(0)/T
                gw+=grad
                if joint:
                    gr[j]+=grad
            # A tiny intercept prior also makes a never-observed category identifiable.
            loss+=.00005*np.sum(b*b)
            gb+=.0001*b
            parts=[gb.ravel(),gw.ravel()]
            if joint:
                parts.append(gr.ravel())
            return loss,np.concatenate(parts)

        size=(T+Xdim+int(joint)*T*Xdim)*K
        opt=minimize(objective,np.zeros(size),jac=True,method="L-BFGS-B",
                     options=dict(maxiter=900,ftol=2e-13,gtol=1e-7,maxls=40))
        maxgrad=float(np.abs(opt.jac).max())
        if not opt.success or maxgrad>2e-5:
            raise ArithmeticError(f"Unconverged discourse model: {opt.message}, {maxgrad}")
        b,w,r=unpack(opt.x)
        for j,t in enumerate(tasks):
            solutions[("joint" if joint else "single",t)]=(b[j],w+r[j],maxgrad)
    return solutions


def ngram_models(texts, train, counts):
    train_text=[texts[i] for i in np.flatnonzero(train)]
    matrices=[]
    for opts in (dict(ngram_range=(1,2),max_features=18000),
                 dict(analyzer="char_wb",ngram_range=(3,5),max_features=18000)):
        vector=TfidfVectorizer(min_df=3,sublinear_tf=True,**opts)
        vector.fit(train_text)
        matrices.append(vector.transform(texts))
    feats=hstack(matrices,format="csr")
    p,d,good=rates(counts)
    preds={}
    for j,m in enumerate(METHODS):
        X=feats[train][good[j]]
        # Each TF-IDF family has unit row norm; alpha=3 gives a finite lexical ridge
        # reference. Multi-output ridge gives an independent distribution comparator.
        rr=Ridge(alpha=3.,solver="lsqr",tol=1e-8).fit(X,np.c_[d[j,good[j]],p[j,good[j]]])
        est=rr.predict(feats)
        prob=np.clip(est[:,1:],.0001,1)
        preds["ngrams."+m+".choice"]=prob/prob.sum(1,keepdims=True)
        preds["ngrams."+m+".disagreement"]=np.clip(est[:,0],0.,1.)
    return preds


def run(extraction, dest):
    if dest.exists():
        raise FileExistsError(dest)
    spec=read(OUT/"inputs.json")
    receipt=read(extraction/"extraction.json")
    packet=BASE/"discourse-stimuli.jsonl"
    if digest(PROTO)!=spec["protocol"] or digest(packet)!=spec["packet_sha256"] or receipt["packet_sha256"]!=digest(packet):
        raise ValueError("Unrecognized stimuli or plan")
    for k,v in receipt["files_sha256"].items():
        if digest(extraction/k)!=v:
            raise ValueError("Feature archive differs from the accelerator receipt")
    for k,v in spec["source"].items():
        if digest(RAW/k)!=v:
            raise ValueError("Public source changed")
    material=read(OUT/"materials.json")
    external=read(OUT/"external.json")
    archive={k:i for i,k in enumerate(read(extraction/"ids.json"))}
    text_lookup={d["id"]:d["text"] for d in lines(packet)}
    extra=[dict(item=k,genre="external",**r) for k,r in sorted(external.items())]
    rows=material+extra
    ids=[r["item"] for r in rows]
    keys=[r["id"] for r in rows]
    train=np.array([r["genre"] in ("novel","europarl") for r in rows])
    train_rows=[r for r,b in zip(rows,train,strict=True) if b]
    votes=collect({r["item"] for r in train_rows})
    classes=sorted({lab for v in votes.values() for pack in v.values() for lab in pack.values()})+["__other__"]
    counts=observations(train_rows,votes,classes)
    texts=[text_lookup[k] for k in keys]
    dense=np.stack([lexical(t,r["left_end"]) for t,r in zip(texts,rows,strict=True)])
    base,sm,sd=transform(train,dense)
    hidden=normalized_hidden(np.load(extraction/"states.npy",allow_pickle=False)[[archive[k] for k in keys]])
    hm=hidden[train].mean(0)
    centered=hidden[train]-hm
    cov=centered.T@centered/len(centered)
    ev,v=eigh(cov,subset_by_index=[len(hm)-64,len(hm)-1])
    del centered,cov
    proj=v/np.sqrt(ev)
    projected=np.column_stack([base,(hidden-hm)@proj])
    params=dict(surface_mean=sm,surface_sd=sd,hidden_mean=hm,projection=proj)
    out=ngram_models(texts,train,counts)
    gradients={}
    for name,X in (("surface",base),("qwen",projected)):
        for target in ("choice","disagreement"):
            for (method,t),(b,w,grad) in fit_masks(X[train],counts,target).items():
                tag=".".join((name,method,METHODS[t],target))
                z=X@w+b
                out[tag]=softmax(z,axis=1) if target=="choice" else expit(z.ravel())
                gradients[tag]=grad
                params[tag+".weights"]=w
                params[tag+".intercept"]=b
    p,d,eligible=rates(counts)
    for j,m in enumerate(METHODS):
        out["constant."+m+".disagreement"]=np.full(len(rows),d[j,eligible[j]].mean())
        cp=(counts[j,eligible[j]].sum(0)+1).astype(float); cp/=cp.sum()
        out["constant."+m+".choice"]=np.tile(cp,(len(rows),1))
    dest.mkdir(parents=True)
    np.savez_compressed(dest/"forecasts.npz",item=ids,**out)
    np.savez_compressed(dest/"weights.npz",**params)
    seal=dict(protocol_sha256=spec["protocol"],packet_sha256=receipt["packet_sha256"],
              source_receipt=digest(OUT/"inputs.json"),extraction_receipt=digest(extraction/"extraction.json"),
              classes=classes,gradient_max_abs=gradients,
              train_items=len(train_rows),train_votes=counts.sum(axis=(1,2)).tolist(),
              test_votes_used_for_training=0,files={k:digest(dest/k) for k in ("forecasts.npz","weights.npz")})
    (dest/"seal.json").write_text(json.dumps(seal,indent=2)+"\n")
    print(json.dumps({k:seal[k] for k in ("train_items","train_votes","files")}))


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--extraction",type=Path,required=True)
    ap.add_argument("--dest",type=Path,default=OUT/"forecast")
    run(**vars(ap.parse_args()))
