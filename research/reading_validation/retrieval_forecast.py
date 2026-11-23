"""Cache word states and native attention to preceding sentences in one causal pass."""

import hashlib
import json
from pathlib import Path
import re
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from research.backbone_transfer.extract import digest, stream_rows, write_json
from research.reading_validation.identity_causal_features import encode_batch, project_basis


WIDTH = 48
HEAD = "463d38528f0d63eb4b7a382eea3022079f701341365f42710adb3f91a765bf1e"
WORD = re.compile(r"\S+")


def geometry(row, encoding):
    offsets = np.asarray(encoding.offsets)
    right = offsets[:,1]
    unit = np.full(len(right),-1,dtype=np.int16)
    words = []
    for j,(lo,hi) in enumerate(row["spans"]):
        indices = np.flatnonzero((offsets[:,0]<hi)&(right>lo))
        for k in indices:
            if row["text"][max(lo,offsets[k,0]):min(hi,right[k])].strip():
                if unit[k] != -1:
                    raise ValueError("Token overlaps two displayed sentences")
                unit[k]=j
        for m in WORD.finditer(row["text"],lo,hi):
            a,b=m.span()
            hits=np.flatnonzero((offsets[:,0]<b)&(right>a))
            if not len(hits):
                raise ValueError("Lost an observed word")
            first,end=int(hits[0]),int(hits[-1])
            certified=right[end]==b and (end+1==len(right) or offsets[end+1,0]>=b)
            words.append((a,b,j,first,end,int(certified)))
    if len(words)==0 or set(unit)-{-1} != set(range(len(row["spans"]))):
        raise ValueError("Text segmentation is incomplete")
    return np.asarray(words,dtype=np.int32),unit


def native(runtime, seqs, layouts, basis, conflict):
    states,_,hidden,mean,peak=encode_batch(runtime,seqs,basis)
    result = []
    for i,(seq,(words,unit)) in enumerate(zip(seqs,layouts,strict=True)):
        length=len(seq)
        end=torch.as_tensor(words[:,4].astype("int64"),device=runtime.device)
        assigned=torch.as_tensor(unit.astype("int64")+1,device=runtime.device)
        groups=F.one_hot(assigned,int(assigned.max())+1)[:,1:].float()
        evidence=torch.stack((mean[i,end,:length]@groups,peak[i,end,:length]@groups),dim=-1)
        # Average headwise peak weights are not normalized over source tokens.
        tensors=[states[i,end].float(),evidence]
        readout=F.layer_norm(hidden[i,:length],(runtime.width,))@conflict["weight"]+conflict["bias"]
        obs=torch.zeros(length,device=runtime.device)
        for start in range(1,length,64):
            stop=min(length,start+64)
            logits=F.linear(hidden[i,start-1:stop-1],runtime.encoder.embed_tokens.weight)
            tokens=torch.as_tensor(seq[start:stop],device=runtime.device)
            obs[start:stop]=F.cross_entropy(logits,tokens,reduction="none")
        cum=F.pad(obs.cumsum(0),(1,0))
        first=torch.as_tensor(words[:,3].astype("int64"),device=runtime.device)
        sizes=(end-first+1).float()
        signals=torch.stack((readout[end],cum[end+1]-cum[first],sizes),dim=1)
        result.append(tuple(t.detach().cpu().numpy() for t in tensors+[signals]))
    return result


@torch.inference_mode()
def run(runtime,reader,checkpoint,source,output):
    source,output=Path(source),Path(output)
    path=source.with_name("frozen.pt")
    if digest(path)!=HEAD or runtime.revision!=checkpoint["revision"]:
        raise ValueError("Counterfactual NLI readout or backbone identity changed")
    head=torch.load(path,map_location=runtime.device,weights_only=True)
    rows=list(stream_rows(source))
    if any(set(r)!={"id","text","spans"} or hashlib.sha256(r["text"].encode()).hexdigest()!=r["id"] for r in rows):
        raise ValueError("Unexpected fields or changed text in reading packet")
    seqs,layouts=[],[]
    for row in rows:
        z=runtime.tokenizer.encode(row["text"],add_special_tokens=False)
        if not 0<len(z.ids)<=2048:
            raise ValueError("A complete passage is outside the verified runtime length")
        seqs.append(z.ids)
        layouts.append(geometry(row,z))
    if len({r["id"] for r in rows})!=len(rows):
        raise ValueError("One forward per unique passage is required")
    output.mkdir(parents=True)
    cuts=np.r_[0,np.cumsum([len(w) for w,_ in layouts])]
    rcuts=np.r_[0,np.cumsum([len(w)*len(r["spans"]) for (w,_),r in zip(layouts,rows,strict=True)])]
    states=np.lib.format.open_memmap(output/"states.npy",mode="w+",dtype=np.float16,shape=(int(cuts[-1]),2*WIDTH))
    relations=np.lib.format.open_memmap(output/"relations.npy",mode="w+",dtype=np.float16,shape=(int(rcuts[-1]),2))
    signals=np.lib.format.open_memmap(output/"signals.npy",mode="w+",dtype=np.float32,shape=(int(cuts[-1]),3))
    np.save(output/"coordinates.npy",np.concatenate([ww for ww,_ in layouts]))
    order=sorted(range(len(rows)),key=lambda i:len(seqs[i]))
    basis=[w[:,:WIDTH].contiguous() for w in project_basis(runtime.width,runtime.device)]
    calls=0
    def count(module,args):
        nonlocal calls
        calls+=1
    start=perf_counter()
    batches=0
    with runtime.encoder.register_forward_pre_hook(count):
        for lo in range(0,len(order),12):
            indices=order[lo:lo+12]
            values=native(runtime,[seqs[j] for j in indices],[layouts[j] for j in indices],basis,head)
            batches+=1
            for j,(z,a,s) in zip(indices,values,strict=True):
                if not all(np.isfinite(v).all() for v in (z,a,s)):
                    raise ArithmeticError("Nonfinite causal evidence")
                ix=slice(cuts[j],cuts[j+1])
                states[ix]=z
                signals[ix]=s
                relations[rcuts[j]:rcuts[j+1]]=a.reshape(-1,2)
            if batches%100==0:
                print(f"Retrieved causal features for {lo+len(indices)}/{len(rows)} passages",flush=True)
    if calls!=batches:
        raise RuntimeError("Expected one native encoder forward per batch")
    checks=[]
    for at in (.35,.8):
        j=order[int(at*len(order))]
        w,u=layouts[j]
        cut=int(w[len(w)//2,4])+1
        kept=w[w[:,4]<cut]
        z,a,s=native(runtime,[seqs[j][:cut]],[(kept,u[:cut])],basis,head)[0]
        old=states[cuts[j]:cuts[j]+len(kept)].astype(np.float32)
        ref=relations[rcuts[j]:rcuts[j+1]].reshape(len(w),-1,2)[:len(kept),:a.shape[1]]
        serr=float(np.abs(signals[cuts[j]:cuts[j]+len(kept)]-s).max())
        checks.append(dict(state_relative=float(np.linalg.norm(z-old)/np.linalg.norm(z)),
                           relation_max=float(np.abs(ref-a).max()),signal_max=serr))
    if any(r["state_relative"]>.002 or r["relation_max"]>.01 or r["signal_max"]>.02 for r in checks):
        raise ArithmeticError(f"Prefix replay failed: {checks}")
    for v in (states,signals,relations):
        v.flush()
    write_json(output/"layout.json",dict(ids=[r["id"] for r in rows],words=cuts.tolist(),relations=rcuts.tolist()))
    names=("states.npy","relations.npy","signals.npy","coordinates.npy","layout.json")
    write_json(output/"receipt.json",dict(packet=digest(source),head=HEAD,revision=runtime.revision,
               code=digest(Path(__file__)),articles=len(rows),words=int(cuts[-1]),tokens=sum(map(len,seqs)),
               model_calls=calls,batches=batches,technical_calls=len(checks),prefix=checks,device=torch.cuda.get_device_name(),
               seconds=perf_counter()-start,files={k:digest(output/k) for k in names}))
