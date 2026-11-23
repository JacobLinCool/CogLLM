"""Forecast explanatory links using only the text through the current token."""

from contextlib import contextmanager
from pathlib import Path
import re
import math
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

from research.backbone_transfer.extract import forward


WIDTH=96
SPLIT=re.compile(r'(?<=[.!?])\s+(?=\S)|\s*\n+\s*(?=\S)')


def sentence_tokens(text,offsets):
    """The next visible sentence starts after a completed delimiter."""
    breaks=[0]+[m.end() for m in SPLIT.finditer(text)]+[len(text)]
    result=[]
    for a,b in zip(breaks,breaks[1:]):
        first=next((i for i,(l,r) in enumerate(offsets) if r>a and l<b and text[max(l,a):min(r,b)].strip()),None)
        last=next((i+1 for i in range(len(offsets)-1,-1,-1) if offsets[i][1]>a and offsets[i][0]<b and text[max(a,offsets[i][0]):min(b,offsets[i][1])].strip()),None)
        if first is not None:
            result.append((first,last))
    if not result or any(left[1]>right[0] for left,right in zip(result,result[1:])):
        raise ValueError('Sentence endpoints do not align to disjoint visible token intervals')
    return result


@contextmanager
def capture(runtime, texts):
    module=runtime.attention_module
    original=module.ALL_ATTENTION_FUNCTIONS
    registry=type(original)(); layers=set(runtime.global_layers); seen=set()
    batch, size=len(texts), max(map(len,texts)); dev=runtime.device
    axes=torch.arange(size,device=dev); causal=axes[:,None]>=axes[None,:]
    lengths=torch.tensor([len(ids) for ids in texts],device=dev)
    validq=axes[None,:]<lengths[:,None]
    total=torch.zeros((batch,size,size),device=dev); peak=torch.zeros_like(total); heads=0
    def wrapped(layer,q,k,v,mask,**kwargs):
        nonlocal heads
        if layer in layers:
            if layer in seen or q.dtype!=torch.float32 or k.shape[-2]!=size or q.shape[:1]!=(batch,) or q.shape[1]%k.shape[1]:
                raise ValueError('Unexpected native grouped-query layout')
            seen.add(layer)
            nh,nk=q.shape[1],k.shape[1]
            scores=q.reshape(batch,nk,nh//nk,size,q.shape[-1])@k.transpose(-2,-1).unsqueeze(2)
            scores=scores*kwargs['scaling']
            if mask is None:
                visible=causal[None,None,None,:,:]
            else:
                mask4=mask if mask.dtype==torch.bool else mask==0
                if mask4.ndim!=4 or mask4.shape[-2:]!=(size,size):
                    raise ValueError('Unexpected native causal mask')
                visible=mask4.unsqueeze(2)
                if not torch.equal((visible[:,0,0]&validq[:,:,None]),causal.unsqueeze(0)&validq[:,:,None]):
                    raise ValueError('A nonpadding query saw a future/padded token')
            weight=scores.masked_fill(~visible,torch.finfo(scores.dtype).min).softmax(-1)
            total.add_(weight.sum((1,2)));torch.maximum(peak,weight.amax((1,2)),out=peak);heads+=nh
        return original['sdpa'](layer,q,k,v,mask,**kwargs)
    registry['sdpa']=wrapped
    module.ALL_ATTENTION_FUNCTIONS=registry
    try:
        yield lambda: (total/heads,peak) if seen==layers and heads>0 else (_ for _ in ()).throw(ValueError('Incomplete native attention capture'))
    finally:
        module.ALL_ATTENTION_FUNCTIONS=original


def project_basis(width,device):
    gen=torch.Generator(device='cpu');gen.manual_seed(270653)
    return [(torch.randn(width,WIDTH,generator=gen)/math.sqrt(width)).to(device) for _ in range(2)]


@torch.inference_mode()
def encode_batch(runtime,sequences,basis):
    middle=[]
    def get_middle(_m,_args,value):
        if not isinstance(value,torch.Tensor) or middle:
            raise ValueError('Unexpected intermediate hidden state')
        middle.append(value)
    with runtime.encoder.layers[runtime.mid_index].register_forward_hook(get_middle),capture(runtime,sequences) as read:
        final=forward(runtime.encoder,sequences,248044,torch)
        mean,peak=read()
    states=torch.cat([F.layer_norm(x,x.shape[-1:])@weight for x,weight in zip((middle[0],final),basis,strict=True)],dim=-1)
    return states,middle[0],final,mean,peak


def summarize(states,mean,peak,spans):
    output=[]
    for a,b in spans:
        v=states[a:b]
        output.append(torch.stack((v.mean(0),v[-1])))
    native=torch.zeros((len(spans),len(spans),2),device=states.device)
    for j,(a,b) in enumerate(spans):
        for i,(s,e) in enumerate(spans[:j]):
            # Relative to uniform attention at the sentence-final token.
            native[j,i,0]=torch.log1p(mean[b-1,s:e].mean()*b)
            native[j,i,1]=torch.log1p(peak[b-1,s:e].amax()*b)
    return torch.stack(output),native


def dense_relative(states,mean,peak,spans):
    n=len(states); out=torch.zeros((n,len(spans),2),device=states.device)
    for i,(a,b) in enumerate(spans):
        distance=torch.arange(1,n+1,device=states.device)
        out[b:,i,0]=torch.log1p(mean[b:,a:b].mean(1)*distance[b:])
        out[b:,i,1]=torch.log1p(peak[b:,a:b].amax(1)*distance[b:])
    return out


def original_scores(reader,ck,middle,final,mean,peak,offsets,spans):
    n=len(final); positions=torch.arange(1,n+1,device=final.device)
    rel=torch.log1p(torch.stack((mean,peak),dim=-1)*positions[:,None,None])-math.log(2)
    rel=rel.masked_fill(torch.triu(torch.ones(n,n,dtype=torch.bool,device=final.device),diagonal=1).unsqueeze(-1),0.)
    score=reader(SimpleNamespace(hidden=final,mid=middle,offsets=offsets,relations=rel)).resolution
    res=score[:,[e-1 for _,e in spans]]*ck['scale']['resolution_gain']+ck['scale']['resolution_bias']+ck['priors']['resolution']
    return res.sigmoid().T
