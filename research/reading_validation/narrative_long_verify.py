"""Check blockwise causal attention and the original token stock against dense reference math."""

from math import log
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from research.reading_validation.narrative_long import BLOCK,attention_rows,exact_scores,pool_lines
from research.token_readout.model import question_mass
from research.token_readout.readout import PortableReadout


def dense(reader,packet,features,chars,units,terminal):
    scores=reader(features)
    opening=scores.opening*packet["scale"]["opening_gain"]+packet["priors"]["opening"]
    resolution=(scores.resolution*packet["scale"]["resolution_gain"]+packet["scale"]["resolution_bias"]
                +packet["priors"]["resolution"])
    include=torch.tensor([any(c.isalnum() for c in chars[a:b]) for a,b in features.offsets],device=opening.device)
    opened,stock,releases=question_mass(opening,resolution,include)
    before=torch.cat((torch.zeros_like(stock[:1]),stock[:-1]))
    transitions=before.sum(1)*(.5*(1-scores.continuity)).clamp(0,1)
    trace=torch.stack((stock.sum(1),opened,releases.sum(1),transitions),dim=-1)
    index=torch.as_tensor(units,device=opening.device,dtype=torch.long)
    ends=torch.as_tensor(terminal,device=opening.device,dtype=torch.long)
    raw=(features.relations[...,0]+log(2)).expm1().clamp_min(0)
    raw=torch.tril(raw)
    mass=raw/raw.sum(-1,keepdim=True)
    released=pool_lines(releases,index,len(ends))
    attention=pool_lines(mass,index,len(ends))
    stock=pool_lines(stock,index,len(ends))
    by_source=[]
    for j,t in enumerate(ends):
        visible=(index==j)&(torch.arange(len(index),device=index.device)<=t)
        by_source.append(torch.stack((attention[t],released[t],stock[t],released[visible].sum(0),
                                      attention[visible].mean(0)),dim=-1))
    return trace,torch.stack(by_source)


def fixture():
    torch.manual_seed(55668)
    torch.set_num_threads(1)
    size=119
    gen=lambda *shape:torch.randn(shape)
    inputs=[(gen(1,4,size,16),gen(1,2,size,16),.25) for _ in range(2)]
    grid=torch.arange(size)
    causal=grid[:,None]>=grid[None,:]
    weights=[]
    max_att=0
    for q,k,scale in inputs:
        v=torch.randn_like(k)
        chunks=[]
        for i in range(0,size,BLOCK):
            j=min(i+BLOCK,size)
            p=attention_rows(q,k,scale,i,j)
            chunks.append((p@v[0,:,:j].unsqueeze(1)).flatten(0,1))
        given=F.scaled_dot_product_attention(q,k.repeat_interleave(2,1),v.repeat_interleave(2,1),
                                              is_causal=True,scale=scale)
        max_att=max(max_att,float((torch.cat(chunks,dim=1)-given[0]).abs().max()))
        z=q[0]@k[0].repeat_interleave(2,0).transpose(-1,-2)*scale
        weights.append(z.masked_fill(~causal,-torch.inf).softmax(-1))
    w=torch.cat(weights,dim=0)
    complex_relation=torch.stack((w.mean(0),w.amax(0)),dim=-1)
    native=(complex_relation*(grid[:,None,None]+1)).log1p()-log(2)
    native=native.masked_fill(~causal[:,:,None],0)
    final,middle=gen(size,96),gen(size,96)
    unit=(np.arange(size)//17).clip(0,6)
    ends=np.array([np.flatnonzero(unit==j)[-1] for j in range(7)])
    candidate="".join("A" if i%7 else "." for i in range(size))
    features=SimpleNamespace(hidden=final,mid=middle,offsets=[(i,i+1) for i in range(size)],relations=native)
    reader=PortableReadout(96,variant="span",relations=True).eval().requires_grad_(False)
    reader.opening.weight.normal_(std=.05)
    reader.relation_weight.copy_(torch.tensor([.1,.15]))
    packet=dict(priors=dict(opening=-2.5,resolution=-3.5),scale=dict(opening_gain=1.1,resolution_gain=.88,resolution_bias=.04))
    trace,pairs,_=exact_scores(reader,packet,dict(enumerate(inputs)),final,middle,features.offsets,candidate,
                               unit,ends,torch.zeros(size))
    expected,links=dense(reader,packet,features,candidate,unit,ends)
    errors=dict(attention=max_att, deployed=float((trace[:,:4]-expected).abs().max()),
                links=float((pairs-links).abs().max()))
    if max(errors.values())>8e-6:
        raise ArithmeticError(f"Blockwise CPU parity failed: {errors}")
    return errors


if __name__=="__main__":
    print(fixture())
