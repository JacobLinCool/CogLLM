"""Learn one causal earlier-word utility from clues that reverse authored antecedent choices."""

from argparse import ArgumentParser
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from research.backbone_transfer.extract import digest,write_json
from research.reading_validation.antecedent_inference import REVISION,native_signals
from research.reading_validation.transition_pointer_features import CONTROL,design
from research.reading_validation.transition_pointer_l4 import watch
from research.token_readout.backbone import Backbone


PROTOCOL="ef8809e8c5391c0feefce61d8cd2f5a2cace788cb6fbb040a7eb56fb4469e678"
PACKET="8efcfbb0884c36adf240503b9bd05fa27d1b236dcaaa67791d183f788a095ccc"
ARMS=("surface","steps","attention","semantic")
N=len(CONTROL)


@torch.inference_mode()
def extract(runtime,row):
    bounds=torch.tensor(row["tokens"],device=runtime.device);start,end=bounds.T
    words=torch.arange(len(start),device=runtime.device)
    with watch(runtime,row["token_count"],end,start,end,False) as cap:
        f=runtime.encode(row["text"])
    if len(f.token_ids)!=row["token_count"] or any(f.offsets[j][1]!=b for (_,j),(_,b) in zip(row["tokens"],row["words"],strict=True)):
        raise ValueError("Original annotated word and text-only presentation differ")
    native=native_signals(f,cap,words,start,end)
    states=np.stack([F.layer_norm(h[end],(runtime.width,)).cpu().numpy() for h in (f.mid,f.hidden)])
    q=row["measurement"]["q"]
    C=design(row,q,row["queries"].index(q))
    return {"C":C,"a":native[:,:q],"h":states}


def distance(q):
    return np.log1p(np.arange(q,0,-1))/np.log1p(q)


def source_vector(x,g):
    c=x["C"];a=x["a"];h=x["h"];q=g["q"];end=g["k"]
    one,two=g["names"]
    def difference(values):
        return values[two].mean(0,dtype=np.float64)-values[one].mean(0,dtype=np.float64)
    dx=difference(c);dist=difference(distance(q))
    blocks=[dx,dx*math.log1p(end-q),difference(a[q]),difference(a[end])]
    for layer in h:
        blocks.extend((difference(layer)*layer[end],dist*(layer[end].astype(float)-layer[q])))
    return np.concatenate(blocks)


def fit(X, y, pair_index, width, device):
    d=X.shape[1]
    rms=np.sqrt(np.square(X).mean(0));rms=np.where(rms>1e-6,rms,1.)
    if d>2*N+104:rms[2*N+104:]*=math.sqrt(width/48)
    signed=torch.as_tensor(X*y[:,None]/rms,device=device,dtype=torch.float64)
    ix=torch.as_tensor(pair_index,device=device,dtype=torch.long)
    v=torch.zeros(d,device=device,dtype=torch.float64,requires_grad=True)
    opt=torch.optim.LBFGS([v],max_iter=500,max_eval=1500,line_search_fn="strong_wolfe",history_size=45,
                         tolerance_grad=1e-9,tolerance_change=1e-13)
    history=[]
    def closure():
        opt.zero_grad();s=signed@v
        point=F.softplus(-s).mean(); reversal=F.softplus(-s[ix].sum(1)).mean()
        o=.5*(point+reversal)+.005*v.square().sum()
        o.backward();history.append(float(o.detach()));return o
    opt.step(closure);last=float(closure().detach());grad=float(v.grad.abs().max())
    if not math.isfinite(last+grad) or grad>1e-6 or last>history[0]+1e-10:
        raise ValueError(f"Semantic observer failed source stationarity: {d} coefficients, {last}, {grad}")
    return v.detach().cpu().numpy()/rms,{"coefficients":d,"initial_objective":history[0],"objective":last,"gradient":grad,
                                        "evaluations":len(history),"same_pair_reversal_terms":len(ix)}


def word_logits(x,q,k,beta):
    """Every preceding word receives a causal score, independent of annotated alternatives."""
    c=x["C"].astype(float);a=x["a"];h=x["h"]
    d=h.shape[-1];t=np.arange(q,k+1);size=len(beta)
    z=np.broadcast_to(c@beta[:N],(len(t),q)).copy()
    if size==N:return z
    z+=np.log1p(t-q)[:,None]*(c@beta[N:2*N])
    if size==2*N:return z
    j=2*N
    z+=a[q]@beta[j:j+52];j+=52
    z+=a[t]@beta[j:j+52];j+=52
    if size==j:return z
    geom=distance(q)
    for layer in h:
        z+=(layer[t]*beta[j:j+d])@layer[:q].T;j+=d
        z+=((layer[t].astype(float)-layer[q])@beta[j:j+d])[:,None]*geom;j+=d
    if j!=size or not np.isfinite(z).all():raise ValueError("Invalid native all-word readout matrix")
    return z


def run(model,stimuli,output,input_sha256,protocol_sha256):
    if (digest(stimuli)!=input_sha256 or input_sha256!=PACKET or protocol_sha256!=PROTOCOL):
        raise ValueError("The signed semantic-source protocol or original inputs changed")
    rows=list(map(json.loads,stimuli.read_text().splitlines()))
    if any(("answer" in r)!=(r["fold"]=="train") for r in rows):
        raise ValueError("An external answer was available during source training")
    rt=Backbone.load("qwen",model,native_attention=True)
    if rt.revision!=REVISION:raise ValueError("Different backbone")
    start=perf_counter();calls=0
    def hook(*_):
        nonlocal calls
        calls+=1
    with TemporaryDirectory() as tmp, rt.encoder.register_forward_pre_hook(hook):
        base=Path(tmp);X=[];ys=[];families={}; model_ids=[]
        for j,r in enumerate(rows):
            x=extract(rt,r);g=r["measurement"]
            if r["fold"]=="train":
                ii=len(ys);yy=r["answer"];X.append(source_vector(x,g));ys.append(yy*2-1)
                families.setdefault(r["group"],[]).append((ii,r["identities"][yy],tuple(sorted(r["identities"]))))
                if ii==0:probe=(x,g,X[-1])
            else:
                np.savez(base/(r["id"]+".npz"),**x);model_ids.append(r["id"])
            if calls!=j+1:raise ValueError("Expected one original-text encoder pass per exposure")
            if (j+1)%300==0:print(f"Original exposure {j+1}/{len(rows)}",flush=True)
        matrix=np.stack(X);y=np.asarray(ys)
        pairs=[]
        for fam,entries in families.items():
            if len(entries)!=2 or entries[0][1]==entries[1][1] or entries[0][2]!=entries[1][2]:
                raise ValueError("An authored training twin did not reverse the same two identities")
            pairs.append([entries[0][0],entries[1][0]])
        widths=(N,2*N,2*N+104,matrix.shape[1]); weights={};stats={}
        if matrix.shape[1]!=2*N+104+4*rt.width:raise ValueError("Missing semantic evidence dimensions")
        for arm,width in zip(ARMS,widths,strict=True):
            weights[arm],stats[arm]=fit(matrix[:,:width],y,pairs,rt.width,rt.device)
            px,pg,v=probe
            z=word_logits(px,pg["q"],pg["k"],weights[arm])[-1]
            l,r=pg["names"]
            if abs((z[r].mean()-z[l].mean())-v[:width]@weights[arm])>1e-5:
                raise ValueError("Open-word inference and source region loss differ")
        all_scores={}
        for row in rows:
            if row["fold"]=="train":continue
            g=row["measurement"];q=g["q"];k=g["k"]
            with np.load(base/(row["id"]+".npz")) as saved:
                x={name:saved[name] for name in ("C","a","h")}
            all_scores[row["id"]]=np.stack([word_logits(x,q,k,weights[arm]).astype(np.float32) for arm in ARMS])
        if len(all_scores)!=len(model_ids):raise ValueError("An unlabelled external presentation was omitted")
    output.mkdir(parents=True,exist_ok=True)
    p=output/"all-word-utilities.npz";w=output/"source-semantic-weights.npz"
    np.savez_compressed(p,**all_scores);np.savez_compressed(w,**weights)
    write_json(output/"receipt.json",{"input_sha256":input_sha256,"protocol_sha256":protocol_sha256,"revision":REVISION,
       "encoder_forwards":calls,"source_training_families":len(families),"source_training_sentences":len(y),"arms":ARMS,
       "fits":stats,"weights_sha256":digest(w),"forecast_sha256":digest(p),"external_supervision":0,"seconds":perf_counter()-start})


if __name__=="__main__":
    p=ArgumentParser(description=__doc__)
    for arg in ("model","stimuli","output"):p.add_argument("--"+arg,type=Path,required=True)
    for arg in ("input-sha256","protocol-sha256"):p.add_argument("--"+arg,required=True)
    run(**vars(p.parse_args()))
