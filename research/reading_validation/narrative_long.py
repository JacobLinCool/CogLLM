"""Evaluate the deployed causal readout on long unpadded stories with bounded attention rows."""

from contextlib import contextmanager
from math import log, sqrt
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from research.token_readout.readout import MATCH_WIDTH


BLOCK=48
CHANNELS=("unresolved","opening","resolution","transition","surprisal","current_line_attention",
          "previous_line_attention","earlier_attention_entropy","top_earlier_attention")


def attention_rows(query,key,scale,start,stop):
    """Raw head-specific causal attention, before aggregation or value projection."""
    heads=query.shape[1]
    groups=key.shape[1]
    q=query[0,:,start:stop].unflatten(0,(groups,heads//groups))
    k=key[0,:,:stop].transpose(-2,-1).unsqueeze(1)
    scores=(q@k)*scale
    past=torch.arange(stop,device=q.device)[None,:]<=torch.arange(start,stop,device=q.device)[:,None]
    return scores.masked_fill(~past,-torch.inf).softmax(-1)


@contextmanager
def bounded_global(runtime,length):
    """Apply the native causal dot product using blocks; preserve Q/K for the readout."""
    module=runtime.attention_module
    original=module.ALL_ATTENTION_FUNCTIONS
    registry=type(original)()
    layers=set(runtime.global_layers)
    bank={}
    def bounded(layer,q,k,v,mask,**kwargs):
        if layer not in layers:
            return original["sdpa"](layer,q,k,v,mask,**kwargs)
        if (layer in bank or q.ndim!=4 or q.shape[0]!=1 or q.shape[-2]!=length or q.dtype!=torch.float32
                or k.shape!=v.shape or k.shape[-2]!=length or q.shape[1]%k.shape[1]
                or kwargs.get("dropout",0)!=0 or kwargs.get("sliding_window") is not None
                or kwargs.get("position_bias") is not None or kwargs.get("is_causal",layer.is_causal) is False):
            raise ValueError("Bounded rendering requires the unmodified causal query/key/value geometry")
        scale=kwargs["scaling"]
        if not np.isfinite(scale) or scale<=0:
            raise ValueError("Unverified native attention temperature")
        bank[layer]=(q,k,scale)
        parts=[]
        for start in range(0,length,BLOCK):
            stop=min(start+BLOCK,length)
            if mask is not None:
                if mask.shape!=(1,1,length,length):
                    raise ValueError("Expected a single complete unpadded causal mask")
                raw=mask[0,0,start:stop]
                legal=torch.arange(length,device=q.device)[None,:]<=torch.arange(start,stop,device=q.device)[:,None]
                if raw.dtype==torch.bool:
                    ok=(raw==legal).all()
                else:
                    ok=torch.where(legal,raw==0,raw<=-1e8).all()
                if not bool(ok):
                    raise ValueError("Noncausal attention mask")
            w=attention_rows(q,k,scale,start,stop)
            output=w@v[0,:,:stop].unsqueeze(1)
            parts.append(output.flatten(0,1).transpose(0,1))
        return torch.cat(parts).unsqueeze(0).contiguous(),None
    registry["sdpa"]=bounded
    module.ALL_ATTENTION_FUNCTIONS=registry
    try:
        yield bank
        if bank.keys()!=layers:
            raise ValueError("Some native full-attention layers were not observed")
    finally:
        module.ALL_ATTENTION_FUNCTIONS=original


def geometry(text,bounds,offsets):
    """Certify every final line boundary and attribute separator tokens to the preceding line."""
    off=np.asarray(offsets)
    end=np.array([b for _,b in bounds])
    lines=np.searchsorted(end,off[:,1],side="left").clip(0,len(end)-1)
    for i,(a,b) in enumerate(bounds):
        visible=(off[:,0]<b)&(off[:,1]>a)
        if (not visible.any() or (lines[visible]!=i).any() or off[visible,1][-1]!=b or
            any(text[max(0,l):min(a,r)].strip() for l,r in off[visible])):
            raise ValueError("Uncertified original line ending")
    # Newline-only tokens carry no evidence about the next line. Keep them in the previous span.
    left=np.searchsorted([a for a,_ in bounds],off[:,0],side="right")-1
    for j,(a,b) in enumerate(off):
        if not text[a:b].strip():
            lines[j]=max(0,int(left[j]))
    terminal=[int(np.flatnonzero(off[:,1]==b)[-1]) for b in end]
    if any(lines[q]!=i for i,q in enumerate(terminal)):
        raise ValueError("A terminal word belongs to a different original line")
    return lines,np.asarray(terminal)


def pool_lines(weights,units,number):
    out=weights.new_zeros((len(weights),number))
    return out.scatter_add_(1,units.expand(len(out),-1),weights)


def exact_scores(reader,packet,bank,final,middle,offsets,text,units,terminal,surprisal):
    final0,middle0=reader.project(SimpleNamespace(hidden=final,mid=middle,offsets=offsets))
    state=F.layer_norm(final0+middle0,final0.shape[-1:])
    previous=torch.cat((state[:1],state[:-1]))
    device=state.device
    pos=torch.arange(len(state),device=device)
    units=torch.as_tensor(units,device=device,dtype=torch.long)
    terminal=torch.as_tensor(terminal,device=device,dtype=torch.long)
    nlines=len(terminal)
    anchors=reader._anchor_spans(state,pos)
    key=reader.key(anchors)
    interaction=reader.interaction_key(anchors) if reader.variant=="span" else None
    eligible=torch.tensor([any(c.isalnum() for c in text[a:b]) for a,b in offsets],device=device)
    trace=torch.zeros((len(pos),len(CHANNELS)),device=device)
    # Columns 0–2 are at the certified line end. Columns 3–4 accumulate only
    # reading positions up through that end, and exclude the following separator.
    line_pairs=torch.zeros((nlines,nlines,5),device=device)
    line_lengths=torch.zeros(nlines,device=device)
    last_log_survival=torch.zeros(len(pos),device=device)
    openings=torch.zeros_like(last_log_survival)
    heads=sum(q.shape[1] for q,_,_ in bank.values())
    cal=packet["scale"]
    priors=packet["priors"]
    for start in range(0,len(pos),BLOCK):
        stop=min(start+BLOCK,len(pos))
        p=pos[start:stop]
        prior=p[:,None]>pos[None,:stop]
        aggregate=final.new_zeros((len(p),stop))
        peak=torch.zeros_like(aggregate)
        for q,k,scale in bank.values():
            w=attention_rows(q,k,scale,start,stop).flatten(0,1)
            aggregate+=w.sum(0)
            peak=torch.maximum(peak,w.amax(0))
        mean=aggregate/heads
        native=(torch.stack((mean,peak),dim=-1)*(p+1)[:,None,None]).log1p()-log(2)
        # Match the deployed denominator clamp on weights relative to the uniform prefix.
        relative=(native[:,:,0]+log(2)).expm1().clamp_min(0)*prior
        history=(relative@state[:stop])/relative.sum(-1,keepdim=True).clamp_min(1e-6)
        cue=torch.cat((state[p],state[p]-previous[p],history),dim=-1)
        raw=reader.query(cue)@key[:stop].T/sqrt(MATCH_WIDTH)
        shift=native@reader.relation_weight
        raw+=shift+reader.resolve_bias
        if interaction is not None:
            z=reader.interaction_query(cue)@interaction[:stop].T/sqrt(MATCH_WIDTH)
            raw+=reader.interaction_gain*F.silu(z+shift)
        logit=raw*cal["resolution_gain"]+cal["resolution_bias"]+priors["resolution"]
        opened=(reader.opening(cue).squeeze(-1)*cal["opening_gain"]+priors["opening"]).sigmoid()*eligible[p]
        openings[p]=opened
        keep=F.logsigmoid(-logit).masked_fill(~prior,0)
        log_survival=last_log_survival[:stop]+keep.cumsum(0)
        before=(log_survival-keep).exp()*openings[:stop]*prior
        after=log_survival.exp()*openings[:stop]*(pos[None,:stop]<=p[:,None])
        resolved=before*logit.sigmoid()
        last_log_survival[:stop]=log_survival[-1]
        transition=before.sum(1)*(.5*(1-F.cosine_similarity(state[p],history))).clamp(0,1)
        mass=pool_lines(mean,units[:stop],nlines)
        by_source=pool_lines(resolved,units[:stop],nlines)
        visible=p<=terminal[units[p]]
        line_pairs[:,:,3].index_add_(0,units[p][visible],by_source[visible])
        line_pairs[:,:,4].index_add_(0,units[p][visible],mass[visible])
        line_lengths.index_add_(0,units[p][visible],torch.ones_like(p[visible],dtype=torch.float32))
        old=torch.arange(nlines,device=device)[None,:]<units[p,None]
        oldmass=mass*old
        sumold=oldmass.sum(-1,keepdim=True)
        prob=oldmass/sumold.clamp_min(1e-8)
        entropy=-(prob*prob.clamp_min(1e-12).log()).sum(-1)
        current=mass.gather(1,units[p,None])[:,0]
        nearest=mass.gather(1,(units[p,None]-1).clamp_min(0))[:,0]*(units[p]>0)
        trace[p]=torch.stack((after.sum(1),opened,resolved.sum(1),transition,surprisal[p],current,nearest,
                              entropy,oldmass.max(1).values),dim=-1)
        chosen=(terminal>=start)&(terminal<stop)
        if chosen.any():
            at=terminal[chosen]-start
            line_pairs[chosen,:,:3]=torch.stack((mass[at],by_source[at],pool_lines(after[at],units[:stop],nlines)),dim=-1)
    line_pairs[:,:,4]/=line_lengths[:,None]
    return trace,line_pairs,torch.cat((final0,middle0),dim=-1)


@torch.inference_mode()
def native(runtime,reader,packet,row,*,semantic_lines=False):
    text=row["text"]
    enc=runtime.tokenizer.encode(text,add_special_tokens=False)
    units,ends=geometry(text,row["spans"],enc.offsets)
    ids=torch.tensor([enc.ids],device=runtime.device,dtype=torch.long)
    mid=[]
    def save(_m,_a,y):
        if mid or not isinstance(y,torch.Tensor):
            raise ValueError("Invalid middle-layer activation")
        mid.append(y[0])
    with runtime.encoder.layers[runtime.mid_index].register_forward_hook(save),bounded_global(runtime,len(enc.ids)) as bank:
        final=runtime.encoder(input_ids=ids,use_cache=False,output_attentions=False,output_hidden_states=False).last_hidden_state[0]
    surprise=torch.zeros(len(enc.ids),device=ids.device)
    for start in range(1,len(enc.ids),32):
        stop=min(start+32,len(enc.ids))
        logits=F.linear(final[start-1:stop-1],runtime.encoder.embed_tokens.weight)
        surprise[start:stop]=F.cross_entropy(logits,ids[0,start:stop],reduction="none")
    trace,pairs,state=exact_scores(reader,packet,bank,final,mid[0],enc.offsets,text,units,ends,surprise)
    values=[v.detach().cpu().numpy() for v in (trace,pairs,state)]
    if not all(np.isfinite(v).all() for v in values):
        raise ArithmeticError("Nonfinite narrative output")
    result=dict(trace=values[0],pairs=values[1],state=values[2].astype(np.float16),offsets=np.asarray(enc.offsets),
                lines=units,ends=ends)
    if semantic_lines:
        # Common random projections keep the same units for an opening and a later cue.
        generator=torch.Generator(device="cpu").manual_seed(330107)
        projection=torch.randn(runtime.width,128,generator=generator).to(ids.device)/sqrt(runtime.width)
        views=[]
        for representation in (runtime.encoder.embed_tokens(ids)[0],mid[0],final):
            token_state=F.layer_norm(representation,(runtime.width,))@projection
            pooled=[]
            for j,(a,b) in enumerate(row["spans"]):
                at=[i for i in range(0 if j==0 else int(ends[j-1])+1,int(ends[j])+1)
                    if units[i]==j and any(c.isalnum() for c in text[max(a,enc.offsets[i][0]):min(b,enc.offsets[i][1])])]
                ix=torch.as_tensor(at if at else [int(ends[j])],device=ids.device)
                pooled.append(token_state[ix].mean(0))
            views.append(torch.stack(pooled))
        result["line_semantic"]=torch.cat(views,-1).cpu().numpy()
    return result
