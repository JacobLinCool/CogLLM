"""Collect causal word-end states for a conditional human recognition-time study."""

from hashlib import sha256
from pathlib import Path
from time import perf_counter
import numpy as np
import torch

from research.backbone_transfer.extract import digest,stream_rows,write_json
from research.reading_validation.identity_causal_features import encode_batch,project_basis


@torch.inference_mode()
def run(runtime,reader,checkpoint,input_path,output):
    src=Path(input_path);out=Path(output);out.mkdir(parents=True)
    rows=list(stream_rows(src));encoded=[runtime.tokenizer.encode(r['text'],add_special_tokens=False) for r in rows]
    lens=np.array([len(r['word_char_ends']) for r in rows]);offset=np.r_[0,lens.cumsum()];n=np.array([len(e.ids) for e in encoded]);basis=project_basis(runtime.width,runtime.device)
    states=np.zeros((int(offset[-1]),2*basis[0].shape[1]+4),dtype='float16');order=np.argsort(n);count=0;start=perf_counter()
    def see(_m,_a):
        nonlocal count
        count+=1
    def features(X,M,P,ends):
        t=torch.as_tensor(ends,device=X.device);middle=M[t]; log=torch.log(middle.clamp_min(1e-15));peak=P[t]
        s=torch.stack((-(middle*log).sum(-1)/torch.log(t.float()+2),peak.amax(-1),middle[torch.arange(len(t),device=X.device),t],
                       (middle*(torch.arange(M.shape[1],device=X.device)[None,:]<t[:,None]-12)).sum(-1)),1)
        return torch.cat((X[t],s),1)
    for r,e in zip(rows,encoded,strict=True):
        if sha256(r['text'].encode()).hexdigest()!=r['id']:raise ValueError('Text identifier mismatch')
        for t,b in zip(r['word_token_ends'],r['word_char_ends'],strict=True):
            if e.offsets[t][1]>b or t+1<len(e.ids) and e.offsets[t+1][0]<b:raise ValueError('A recognition boundary contains an unrevealed token')
    with runtime.encoder.register_forward_pre_hook(see):
        for s in range(0,len(order),12):
            selected=order[s:s+12];X,_,_,mean,peak=encode_batch(runtime,[encoded[i].ids for i in selected],basis)
            for j,i in enumerate(selected):
                states[offset[i]:offset[i+1]]=features(X[j,:n[i]],mean[j,:n[i],:n[i]],peak[j,:n[i],:n[i]],rows[i]['word_token_ends']).half().cpu().numpy()
    if count!=(len(order)+11)//12:raise ValueError('Unexpected number of encoder calls')
    checks=[]
    for i in [order[len(order)//4],order[len(order)//2],order[3*len(order)//4]]:
        r=rows[i];cut=len(r['word_token_ends'])//2;ends=r['word_token_ends'][:cut];T=ends[-1]+1
        X,_,_,m,p=encode_batch(runtime,[encoded[i].ids[:T]],basis);ind=features(X[0],m[0],p[0],ends).cpu().numpy();old=states[offset[i]:offset[i]+cut].astype('float32')
        err=float(np.linalg.norm(ind-old)/np.linalg.norm(ind));checks.append(err)
    if max(checks)>.001 or not np.isfinite(states).all():raise ValueError('Native prefix features failed')
    np.save(out/'states.npy',states);write_json(out/'geometry.json',{'source':digest(src),'revision':runtime.revision,'seconds':perf_counter()-start,
        'ids':[r['id'] for r in rows],'word_offsets':offset.tolist(),'words':len(states),'encoder_batches':count,'control_calls':len(checks),'prefix_relative_error':checks})
