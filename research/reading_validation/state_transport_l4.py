"""Archive the causal word and sentence evidence needed to read evolving entity states."""
import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from research.backbone_transfer.extract import digest,stream_rows,write_json
from research.reading_validation.argument_data import h
from research.reading_validation.argument_l4 import locations,summarize
from research.reading_validation.expectation_native import coordinates
from research.reading_validation.identity_causal_features import encode_batch,project_basis
from research.reading_validation.identity_run_l4 import run_phases


def word_values(runtime,ids,words,states,final,mean,peak,spans):
    n=len(ids);tokens=torch.tensor(ids,device=states.device);observed=torch.zeros(n,device=states.device)
    weights=runtime.encoder.embed_tokens.weight
    for lo in range(1,n,64):
        hi=min(n,lo+64)
        observed[lo:hi]=F.cross_entropy(F.linear(final[lo-1:hi-1],weights),tokens[lo:hi],reduction='none')
    c=F.pad(observed.double().cumsum(0),(1,0))
    lo,hi=torch.as_tensor(words[:,2:4].astype(np.int64),device=states.device).T
    sur=(c[hi+1]-c[lo]).float()
    # Keep independent coordinates from both native layers for entity-specific semantic state readout.
    dims=torch.cat((states[:,:32],states[:,96:128]),1)
    running=F.pad(dims.cumsum(0),(0,0,1,0))
    word=(running[hi+1]-running[lo])/(hi-lo+1)[:,None]
    att=torch.stack([torch.stack((mean[hi,a:b].sum(1),peak[hi,a:b].amax(1)),dim=1) for a,b in spans],dim=1)
    complete=torch.tensor([b for _,b in spans],device=states.device)
    att=att.masked_fill((complete[None,:]>lo[:,None])[:,:,None],0.)
    return sur,word,att


@torch.inference_mode()
def run(runtime,_reader,_checkpoint,source,out):
    out.mkdir();basis=project_basis(runtime.width,runtime.device);items=[];skip={};arrays={};info={};start=perf_counter();calls=0
    for r in stream_rows(source):
        k,t=r['id'],r['text']
        if set(r)!={'id','text'} or h(t)!=k:raise ValueError('Expected the unmarked, complete source text only')
        e=runtime.tokenizer.encode(t,add_special_tokens=False);loc=locations(t,e.offsets)
        if len(e.ids)>1024:skip[k]='over_1024_model_tokens';continue
        if loc is None:skip[k]='subword_crosses_two_sentences';continue
        words=coordinates(t,e)
        if not len(words):skip[k]='no_lexical_word';continue
        coords,spans=loc;items.append((k,e.ids,coords,spans,words))
    items.sort(key=lambda r:(len(r[1]),r[0]))
    def extract(batch):
        states,_,final,mean,peak=encode_batch(runtime,[r[1] for r in batch],basis);result=[]
        for i,(_,seq,_,spans,words) in enumerate(batch):
            n=len(seq);st=states[i,:n];m=mean[i,:n,:n];p=peak[i,:n,:n]
            emb=F.layer_norm(runtime.encoder.embed_tokens(torch.tensor(seq,device=runtime.device)),(runtime.width,))@basis[0]
            rep,links=summarize(st,emb,m,p,spans)
            sur,w,rel=word_values(runtime,seq,words,st,final[i,:n],m,p,spans)
            result.append({key:value.cpu().numpy() for key,value in {'rep':rep,'links':links,'surprisal':sur,'words':w,'rel':rel}.items()})
        return result
    def called(_m,_a):
        nonlocal calls
        calls+=1
    with runtime.encoder.register_forward_pre_hook(called):
        for lo in range(0,len(items),12):
            batch=items[lo:lo+12]
            for row,z in zip(batch,extract(batch),strict=True):
                k,seq,original,_,coords=row;info[k]=dict(tokens=len(seq),sentences=original)
                arrays[k+'.char']=coords
                for name,v in z.items():
                    if not np.isfinite(v).all():raise ArithmeticError('Invalid word or sentence feature')
                    arrays[k+'.'+name]=v.astype('float32' if name=='surprisal' else 'float16')
            if lo and lo%600==0:print('Complete article evidence',lo,'of',len(items),flush=True)
    if calls!=(len(items)+11)//12:raise ArithmeticError('Duplicate native call on an original passage')
    checks=[]
    for j in (len(items)//4,len(items)//2,3*len(items)//4):
        k,seq,original,s,w=items[j];limit=max(1,len(s)//2);end=s[limit-1][1];keep=w[:,3]<end
        if not keep.any():raise ValueError('Cannot certify a lexical prefix')
        z=extract([(k,seq[:end],original[:limit],s[:limit],w[keep])])[0];nw=int(keep.sum())
        errors={}
        for name,arr in z.items():
            other=arrays[k+'.'+name]
            if name in ('rep','links'):other=other[:limit]
            else:other=other[:nw]
            if name in ('rel','links'):other=other[:,:limit]
            errors[name]=float(np.max(abs(arr-other)))
            if errors[name]>.015:raise ArithmeticError(f'Prefix precision changed at {name}, {errors[name]}')
        checks.append(dict(id=k,tokens=end,errors=errors))
    np.savez_compressed(out/'features.npz',**arrays)
    write_json(out/'receipt.json',dict(source=digest(source),file=digest(out/'features.npz'),articles=info,
           skipped=skip,encoder_batches=calls,prefix_control_forwards=3,checks=checks,seconds=perf_counter()-start,
           revision=runtime.revision,device=torch.cuda.get_device_name()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('model','stimuli','output'):p.add_argument('--'+key,type=Path,required=True)
    for key in ('candidate','protocol-sha256'):p.add_argument('--'+key,required=True)
    a=p.parse_args();run_phases(a.model,a.candidate,a.output,a.protocol_sha256,[('world-state',run,a.stimuli)])
