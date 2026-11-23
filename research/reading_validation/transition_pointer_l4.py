"""Fit an article-only prior-word pointer to original human clicks with sparse native attention."""
from argparse import ArgumentParser
from collections import Counter
from contextlib import contextmanager
import gc
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from research.backbone_transfer.extract import digest,write_json
from research.reading_validation.transition_pointer_features import CONTROL,POSITION,design
from research.token_readout.backbone import Backbone,_ATTENTION_LOCK,_GlobalRelationships


PROTOCOL='2e6d39f45053dfcf336061d3f07bbebe2bba4f59fbef42d61dfee9c7af9f88af'
REVISION='15852e8c16360a2fea060d615a32b45270f8a8fc'
ARMS=('recency','lexical','pooled','headwise')


class SparseCapture:
    def __init__(self,original,layers,positions,starts,ends,query,technical):
        self.original=original;self.layers=layers;self.seen={};self.positions=positions;self.checked_masks=[]
        self.starts=starts;self.ends=ends;self.query=query;self.technical=technical;self.token_weights={}
    def __call__(self,module,query,key,value,attention_mask,**kwargs):
        if module in self.layers:
            layer=self.layers.index(module);scale=kwargs.get('scaling')
            causal=kwargs.get('is_causal')
            if causal is None:causal=module.is_causal
            if (layer in self.seen or query.shape[0]!=1 or key.shape[0]!=1 or query.shape[-2]!=len(self.positions)
                    or query.dtype!=torch.float32 or key.dtype!=torch.float32 or len(query.shape)!=4
                    or not math.isfinite(scale) or scale<=0 or kwargs.get('dropout',0)!=0
                    or kwargs.get('sliding_window') is not None or kwargs.get('position_bias') is not None
                    or not causal):
                raise ValueError('Selected rows must use actual unpadded FP32 causal global attention')
            _GlobalRelationships.check_mask(self,attention_mask)
            heads=query.shape[1];kh=key.shape[1]
            if heads%kh or module.num_key_value_groups!=heads//kh:raise ValueError('Unexpected grouped-query source')
            q=query[0].unflatten(0,(kh,heads//kh))[:,:,self.query]
            logits=(q@key[0].transpose(-2,-1)[:,None])*scale
            valid=self.positions[None,:]<=self.query[:,None]
            weights=logits.masked_fill(~valid,-torch.inf).softmax(-1).flatten(0,1)
            if self.technical:self.token_weights[layer]=weights
            sums=F.pad(weights.cumsum(-1),(1,0));mass=(sums[:,:,self.ends+1]-sums[:,:,self.starts]).clamp_min(0)
            relative=mass*(self.query+1)[None,:,None]/(self.ends-self.starts+1)[None,None,:]
            self.seen[layer]=relative
        return self.original(module,query,key,value,attention_mask,**kwargs)


@contextmanager
def watch(runtime,tokens,query,starts,ends,technical):
    with _ATTENTION_LOCK:
        module=runtime.attention_module;original=module.ALL_ATTENTION_FUNCTIONS;local=type(original)()
        cap=SparseCapture(original['sdpa'],runtime.global_layers,torch.arange(tokens,device=runtime.device),starts,ends,query,technical)
        local['sdpa']=cap;module.ALL_ATTENTION_FUNCTIONS=local
        try:yield cap
        finally:module.ALL_ATTENTION_FUNCTIONS=original


@torch.inference_mode()
def exposure(runtime,text,positions,anchor,technical=False):
    enc=runtime.tokenizer.encode(text,add_special_tokens=False);ids=torch.tensor([enc.ids],device=runtime.device)
    span=torch.tensor(positions,device=runtime.device,dtype=torch.long);stop=span[:,1];query=stop[anchor]
    mid=[]
    def save(m,i,o):mid.append(o[0])
    with watch(runtime,len(enc.ids),query,span[:,0],stop,technical) as cap,runtime.encoder.layers[runtime.mid_index].register_forward_hook(save):
        last=runtime.encoder(input_ids=ids,use_cache=False,output_attentions=False,output_hidden_states=False).last_hidden_state[0]
    if len(mid)!=1 or len(cap.seen)!=len(runtime.global_layers):raise ValueError('Original forward did not cross every native relation source')
    heads=torch.cat([cap.seen[i] for i in sorted(cap.seen)],0)
    hidden=[F.normalize(z[stop]) for z in [mid[0],last]]
    cos=[(t[anchor]@t.T).clamp(-1,1) for t in hidden]
    p=torch.stack([*cos,torch.log1p(heads.mean(0)),torch.log1p(heads.amax(0))],-1)
    all_pairs=torch.cat((p,torch.log1p(heads).permute(1,2,0)),dim=-1).cpu().numpy()
    raw=torch.cat([cap.token_weights[i] for i in sorted(cap.token_weights)],0) if technical else None
    return all_pairs,raw,len(enc.ids)


def parity(runtime):
    text='Lena met her neighbour. Later she phoned home.'
    feat=runtime.encode(text);ix=np.array([3,6])
    seq=[[k,k] for k in range(len(feat.token_ids))];p,raw,_=exposure(runtime,text,seq,ix,True)
    scale=torch.tensor(ix+1,device=runtime.device)[:,None]
    replay=torch.stack([torch.log1p(raw.mean(0)*scale),torch.log1p(raw.amax(0)*scale)],-1)-math.log(2)
    keep=torch.arange(len(feat.token_ids),device=runtime.device)[None,:]<=torch.tensor(ix,device=runtime.device)[:,None]
    e=float((replay[keep]-feat.relations[ix][keep]).abs().max())
    prefix='Lena met her neighbour.';early=runtime.tokenizer.encode(prefix,add_special_tokens=False)
    if tuple(early.ids)!=feat.token_ids[:len(early.ids)]:raise ValueError('Technical comparison lacks a stable token prefix')
    q,_,_=exposure(runtime,prefix,seq[:len(early.ids)],np.array([3]))
    drift=float(np.max(np.abs(q[0,:4]-p[0,:4])))
    if max(drift,e)>.001:raise ValueError('Headwise native relation or causal future-invariance check failed')
    return {'native_pooled_max_error':e,'changed_suffix_max_error':drift,'native_heads':raw.shape[0],'technical_forwards':3}


def extraction(model,rows,tmp):
    r=Backbone.load('qwen',model,native_attention=True)
    if r.revision!=REVISION:raise ValueError('Wrong frozen Qwen revision')
    tech=parity(r);cols=len(CONTROL)+4+tech['native_heads'];count=sum(sum(s['queries']) for s in rows)
    disk=np.lib.format.open_memmap(tmp/'pairs.npy',mode='w+',dtype='float32',shape=(count,cols))
    targets=[];n=0;calls=0
    def observe(m,a):
        nonlocal calls
        calls+=1
    with r.encoder.register_forward_pre_hook(observe):
        for k,row in enumerate(rows):
            anchor=row['queries']
            if not anchor:
                f=r.encode(row['text'])
                if len(f.token_ids)!=row['token_count'] or calls!=k+1:raise ValueError('Invalid no-pronoun article exposure')
                continue
            blocks,_,t=exposure(r,row['text'],row['tokens'],anchor)
            enc=r.tokenizer.encode(row['text'],add_special_tokens=False)
            if t!=row['token_count'] or any(enc.offsets[j][1]!=b for (_,b),(_,j) in zip(row['words'],row['tokens'],strict=True)):
                raise ValueError('A stored word endpoint changed')
            for i,q in enumerate(anchor):
                stop=n+q;disk[n:stop]=np.column_stack([design(row,q,i),blocks[i,:q]])
                item={'doc':row['id'],'q':q,'range':[n,stop],'group':row['group'],'fold':row['fold']}
                if row['fold']!='test':item['clicks']=row['clicks'][str(q)]
                targets.append(item);n=stop
            if calls!=k+1:raise ValueError('More than one forward for the original article')
            if (k+1)%50==0:print(f'Pointer complete article forwards {k+1}/{len(rows)}',flush=True)
    if n!=count:raise ValueError('Forward cache geometry changed')
    disk.flush();return disk,targets,{'forwards':calls,'pairs':n,'technical':tech,'native_layers':len(r.global_layers),'columns':cols}


def reduction(z,lengths):
    peak=torch.segment_reduce(z.detach(),reduce='max',lengths=lengths)
    ex=(z-peak.repeat_interleave(lengths)).exp()
    return torch.segment_reduce(ex,reduce='sum',lengths=lengths).log()+peak


def train(X,source):
    device=X.device;dtype=X.dtype;lengths=[];voters=[];ixs=[];deltas=[];rank=[]
    groups=Counter(m['group'] for m in source);cursor=0
    for m in source:
        size=m['q'];lengths.append(size);mass=1/(len(groups)*groups[m['group']]);rank.append(mass)
        votes=sum(z[2] for z in m['clicks'])
        for a,b,n in m['clicks']:
            if not 0<=a<b<=size or n<1:raise ValueError('Observed antecedent is not an earlier region')
            ixs.extend(range(cursor+a,cursor+b));deltas.append(b-a);voters.append(mass*n/votes)
        cursor+=size
    if cursor!=len(X) or abs(sum(voters)-1)>1e-8:raise ValueError('Source work weights differ from one')
    loc=torch.tensor(ixs,device=device);q=torch.tensor(lengths,device=device);v=torch.tensor(deltas,device=device)
    wr=torch.tensor(rank,device=device,dtype=dtype);wv=torch.tensor(voters,device=device,dtype=dtype)
    beta=torch.zeros(X.shape[1],device=device,dtype=dtype,requires_grad=True)
    opt=torch.optim.LBFGS([beta],history_size=40,max_iter=250,tolerance_grad=1e-8,tolerance_change=1e-13,line_search_fn='strong_wolfe')
    trace=[]
    def closure():
        opt.zero_grad();s=X@beta
        loss=wr@reduction(s,q)-wv@reduction(s[loc],v)+.005*beta.square().sum()
        loss.backward();trace.append([float(loss.detach()),float(beta.grad.abs().max())]);return loss
    opt.step(closure);closure()
    if not np.isfinite(trace).all() or trace[-1][1]>1e-6:raise ValueError(f'Antecedent region fit did not converge: {trace[-1]}')
    return beta.detach(),{'objective':trace[-1][0],'gradient':trace[-1][1],'evaluations':len(trace),'initial_objective':trace[0][0]}


def run(model,stimuli,output,protocol_sha256,input_sha256):
    if digest(stimuli)!=input_sha256 or protocol_sha256!=PROTOCOL:raise ValueError('Wrong pointer packet or protocol')
    rows=list(map(json.loads,stimuli.read_text().splitlines()))
    if any(('clicks' in r)!=(r['fold']!='test') for r in rows):raise ValueError('Gold response entered feature packet')
    output.mkdir(exist_ok=True,parents=True);start=perf_counter()
    with TemporaryDirectory(dir='/content') as tmp:
        disk,targets,meta=extraction(model,rows,Path(tmp));elapsed=perf_counter()-start
        gc.collect();torch.cuda.empty_cache()
        source=[m for m in targets if m['fold']=='train'];idx=np.concatenate([np.arange(*t['range']) for t in source])
        widths=(len(POSITION),len(CONTROL),len(CONTROL)+4,disk.shape[1]);saved={};forecasts={};fits={}
        for arm,width in zip(ARMS,widths,strict=True):
            X=torch.tensor(disk[:,:width],device='cuda',dtype=torch.float64)
            mu=X[idx].mean(0);sd=X[idx].std(0,unbiased=False);sd=torch.where(sd>1e-8,sd,torch.ones_like(sd))
            X.sub_(mu).div_(sd);b,fit=train(X[idx],source);pred=(X@b).detach().cpu().numpy()
            for t in targets:
                if t['fold']=='train':continue
                a,c=t['range'];k=t['doc']+'/'+str(t['q'])
                if arm==ARMS[0]:forecasts[k]=[]
                forecasts[k].append(pred[a:c].astype(np.float32))
            for name,z in [('mean',mu),('sd',sd),('beta',b)]:saved[arm+'__'+name]=z.cpu().numpy()
            fits[arm]=fit;del X;print('Pointer',arm,fit,flush=True)
        archive=output/'candidate-word-logits.npz';weights=output/'source-weights.npz'
        np.savez_compressed(archive,**{k:np.stack(v) for k,v in forecasts.items()});np.savez_compressed(weights,**saved)
        write_json(output/'receipt.json',{'protocol_sha256':PROTOCOL,'input_sha256':input_sha256,'candidate':'qwen','revision':REVISION,
            'source_fits':fits,'arms':ARMS,'features':meta,'extraction_seconds':elapsed,'elapsed_seconds':perf_counter()-start,
            'predictions_sha256':digest(archive),'weights_sha256':digest(weights)})


if __name__=='__main__':
    p=ArgumentParser(description=__doc__)
    for k in ('model','stimuli','output'):p.add_argument('--'+k,type=Path,required=True)
    for k in ('protocol-sha256','input-sha256'):p.add_argument('--'+k,required=True)
    run(**vars(p.parse_args()))
