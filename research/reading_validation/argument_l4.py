"""Capture every completed sentence and its attention to prior sentences in one native pass."""
import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from research.backbone_transfer.extract import digest, stream_rows, write_json
from research.reading_validation.argument_data import h,sentences
from research.reading_validation.identity_causal_features import encode_batch,project_basis
from research.reading_validation.identity_run_l4 import run_phases


def locations(article,offsets):
    coords=sentences(article)
    token_spans=[]
    for a,b in coords:
        hits=[i for i,(s,e) in enumerate(offsets) if s<b and e>a and article[max(s,a):min(e,b)].strip()]
        if not hits:return None
        token_spans.append((hits[0],hits[-1]+1))
    if any(x[1]>y[0] for x,y in zip(token_spans,token_spans[1:])):return None
    return coords,token_spans


def summarize(states,embedding,mean,peak,spans):
    n=len(states);device=states.device
    start,end=torch.as_tensor(spans,device=device).T
    axis=torch.arange(n,device=device)
    mask=(axis>=start[:,None])&(axis<end[:,None])
    counts=mask.sum(1)
    pooling=mask.float()/counts[:,None]
    rep=torch.cat((pooling@states,states[end-1],pooling@embedding),dim=1)
    # Query once at each sentence end. Key-to-sentence reduction never includes a future position.
    avg=torch.log1p(((mean[end-1]@pooling.T)*end[:,None]).clamp_min(0))
    mx=torch.stack([peak[end-1,a:b].amax(1) for a,b in spans],dim=1)
    link=torch.stack((avg,torch.log1p(mx*end[:,None])),dim=-1)
    link=torch.tril(link.permute(2,0,1),diagonal=-1).permute(1,2,0)
    return rep,link


@torch.inference_mode()
def run(runtime,_reader,_checkpoint,source,out):
    out.mkdir()
    rows=list(stream_rows(source));basis=project_basis(runtime.width,runtime.device)
    items=[];skip={};refs={};start=perf_counter()
    for r in rows:
        t=r['text'];key=r['id']
        if set(r)!={'text','id'} or h(t)!=key:raise ValueError('Unmarked whole article did not pass its hash')
        e=runtime.tokenizer.encode(t,add_special_tokens=False)
        if len(e.ids)>1024:skip[key]='more_than_1024_tokens';continue
        loc=locations(t,e.offsets)
        if not loc:skip[key]='nonseparable_sentence_token';continue
        coords,spans=loc
        if len(coords)<2:skip[key]='single_visible_sentence';continue
        items.append((key,e.ids,coords,spans))
    items.sort(key=lambda r:(len(r[1]),r[0]))
    arrays={};calls=0
    def count(_module,_args):
        nonlocal calls
        calls+=1
    def extract(batch):
        seq=[r[1] for r in batch];projected,_,_,avg,peak=encode_batch(runtime,seq,basis);result=[]
        for i,(_,ids,_,spans) in enumerate(batch):
            n=len(ids);emb=runtime.encoder.embed_tokens(torch.tensor(ids,device=runtime.device))
            emb=F.layer_norm(emb,(runtime.width,))@basis[0]
            rep,pair=summarize(projected[i,:n],emb,avg[i,:n,:n],peak[i,:n,:n],spans)
            result.append((rep.cpu().numpy(),pair.cpu().numpy()))
        return result
    with runtime.encoder.register_forward_pre_hook(count):
        for lo in range(0,len(items),8):
            batch=items[lo:lo+8]
            for row,(rep,pair) in zip(batch,extract(batch),strict=True):
                k,ids,coords,_=row
                arrays[k+'.rep']=rep.astype('float16');arrays[k+'.att']=pair.astype('float16')
                refs[k]=dict(tokens=len(ids),sentences=coords)
            if lo and lo%400==0:print(f'{lo}/{len(items)} whole essays',flush=True)
    if calls!=(len(items)+7)//8:raise ArithmeticError('Extra encoder call for a study essay')
    checks=[]
    for ix in (len(items)//3,2*len(items)//3):
        k,ids,c,s=items[ix];limit=max(1,len(s)//2);end=s[limit-1][1]
        rep,att=extract([(k,ids[:end],c[:limit],s[:limit])])[0]
        error=max(float(abs(rep-arrays[k+'.rep'][:limit]).max()),float(abs(att-arrays[k+'.att'][:limit,:limit]).max()))
        if error>.009:raise ArithmeticError(f'Unchanged causal sentences differ by {error}')
        checks.append(dict(id=k,tokens=end,max_abs=error))
    np.savez_compressed(out/'features.npz',**arrays)
    write_json(out/'receipt.json',dict(source=digest(source),file=digest(out/'features.npz'),articles=refs,
               excluded=skip,encoder_batches=calls,suffix_checks=checks,seconds=perf_counter()-start,
               width=rep.shape[1],revision=runtime.revision,device=torch.cuda.get_device_name()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('model','stimuli','output'):p.add_argument('--'+k,type=Path,required=True)
    for k in ('candidate','protocol-sha256'):p.add_argument('--'+k,required=True)
    a=p.parse_args();run_phases(a.model,a.candidate,a.output,a.protocol_sha256,[('argument',run,a.stimuli)])
