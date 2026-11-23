"""Extract causal states and all word-end-to-token native attention from original textbook windows."""

from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from research.backbone_transfer.extract import digest,stream_rows,write_json
from research.reading_validation.identity_causal_features import encode_batch,project_basis


WIDTH=40


def projection(runtime):
    # Use independent fixed dimensionality reduction of normalized middle and final language states.
    return [p[:,:WIDTH].contiguous() for p in project_basis(runtime.width,runtime.device)]


@torch.inference_mode()
def run(runtime,reader,checkpoint,input_path,output):
    source=Path(input_path);out=Path(output);out.mkdir(parents=True)
    rows=list(stream_rows(source));encoded=[runtime.tokenizer.encode(r['text'],add_special_tokens=False) for r in rows]
    if any(sha256(r['text'].encode()).hexdigest()!=r['id'] for r in rows):raise ValueError('Incorrect text ID')
    n=np.array([len(e.ids) for e in encoded],dtype=np.int64);nw=np.array([len(r['words']) for r in rows],dtype=np.int64)
    if n.min()<=0 or n.max()>1024 or nw.min()<=0:raise ValueError('Word or token range exceeded')
    starts=np.r_[0,n.cumsum()];offsets=np.r_[0,(n*nw).cumsum()]
    states=np.lib.format.open_memmap(out/'states.npy',mode='w+',dtype=np.float16,shape=(int(starts[-1]),2*WIDTH))
    attention=np.lib.format.open_memmap(out/'attention.npy',mode='w+',dtype=np.uint8,shape=(int(offsets[-1]),2))
    for row,e in zip(rows,encoded,strict=True):
        prior=-1
        for a,b,first,end in row['words']:
            if not prior<end<len(e.ids) or not 0<=a<b<=len(row['text']) or e.offsets[end][1]>b or e.offsets[first][1]<=a:
                raise ValueError('Source word coordinate verification failed')
            prior=end
    basis=projection(runtime);batch_size=16;order=sorted(range(len(rows)),key=lambda i:(n[i],rows[i]['id']))
    count=0;start=perf_counter()
    def counted(_m,_a):
        nonlocal count
        count+=1
    with runtime.encoder.register_forward_pre_hook(counted):
        for k in range(0,len(order),batch_size):
            ix=order[k:k+batch_size];X,_,_,mean,peak=encode_batch(runtime,[encoded[i].ids for i in ix],basis)
            for j,i in enumerate(ix):
                T=int(n[i]);ends=torch.as_tensor([v[3] for v in rows[i]['words']],device=runtime.device)
                states[starts[i]:starts[i+1]]=X[j,:T].half().cpu().numpy()
                scores=torch.stack((mean[j,ends,:T],peak[j,ends,:T]),dim=-1)*(ends+1)[:,None,None]
                quant=torch.round(torch.log1p(scores)*32)
                if not torch.isfinite(quant).all() or quant.min()<0 or quant.max()>255:raise ValueError('Native attention quantization out of range')
                attention[offsets[i]:offsets[i+1]]=quant.to(torch.uint8).reshape(-1,2).cpu().numpy()
            if k and k%(batch_size*100)==0:print(f'Textbook windows {k}/{len(rows)}',flush=True)
    if count!=(len(rows)+batch_size-1)//batch_size:raise ValueError('Unexpected backbone forward count')
    # Test two separate trimmed prefixes using original native positions and recomputed attention.
    checks=[];control_forwards=0
    for i in [order[len(order)//3],order[2*len(order)//3]]:
        row=rows[i];stop=row['words'][len(row['words'])//2][3]+1
        q,_,_,m,p=encode_batch(runtime,[encoded[i].ids[:stop]],basis);control_forwards+=1
        saved=states[starts[i]:starts[i]+stop].astype('float32')
        compare=q[0].cpu().numpy();relative=float(np.linalg.norm(saved-compare)/np.linalg.norm(compare))
        all_values=attention[offsets[i]:offsets[i+1]].reshape(len(row['words']),int(n[i]),2)
        ends=[w[3] for w in row['words'] if w[3]<stop]
        rr=torch.stack((m[0,ends],p[0,ends]),dim=-1)*(torch.as_tensor(ends,device=runtime.device)+1)[:,None,None]
        actual=torch.log1p(rr).cpu().numpy();err=float(abs(all_values[:len(ends),:stop].astype('float32')/32-actual).max())
        if relative>.001 or err>.016:raise ValueError('Single-pass semantic or native attention prefix check failed')
        checks.append({'id':row['id'],'tokens':stop,'state_relative_error':relative,'relation_max_error':err})
    states.flush();attention.flush()
    write_json(out/'geometry.json',{'ids':[r['id'] for r in rows],'token_offsets':starts.tolist(),'attention_offsets':offsets.tolist(),
                                    'source':digest(source),'quantization':32,'max_abs_quantization_error':1/64,'dimension':2*WIDTH,
                                    'forward_batches':count,'controls':checks,'control_forwards':control_forwards,'seconds':perf_counter()-start,
                                    'words':int(nw.sum()),'tokens':int(n.sum()),'qwen_revision':runtime.revision})
