"""Resolve natural Spanish clues, original token traces and a terminal semantic readout in one pass."""
from dataclasses import replace
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from torch.nn import functional as F
from research.backbone_transfer.extract import digest,read_json,stream_rows,write_json
from research.token_readout.runner import trace_rows


@torch.inference_mode()
def run(runtime,reader,checkpoint,input_path,output_dir):
    source=Path(input_path);out=Path(output_dir);out.mkdir();rows=list(stream_rows(source));vfile=source.with_name('vocabulary.json');vocab=read_json(vfile)['onsets'];table=runtime.encoder.embed_tokens.weight
    if sorted(set(vocab))!=vocab or not runtime.encoder.config.tie_word_embeddings:raise ValueError('Unexpected Spanish onset inventory or native head')
    lookup=table[torch.as_tensor(vocab,device=runtime.device)];direction=F.normalize(lookup.float(),dim=-1);start=perf_counter();n=0;signals={};checks=[]
    def count(_m,_a):
        nonlocal n
        n+=1
    def results(r,feat):
        raw=r['article_end'];words=r['words'];span=r['category_tokens'];text=r['text'];ids=torch.as_tensor(feat.token_ids,device=runtime.device)
        if len(words)!=21 or len(ids)<raw or not 0<span[0]<span[1]<=raw or not all(feat.offsets[t][0]<b and feat.offsets[t][1]<=b for a,b,s,t in words):raise ValueError('Study-specific word coordinate mismatch')
        scores=reader(feat);scale=checkpoint['scale'];scores=replace(scores,opening=scores.opening*scale['opening_gain'],resolution=scores.resolution*scale['resolution_gain']+scale['resolution_bias'])
        tr=trace_rows(text,feat.offsets,scores,checkpoint['priors'],reader.anchor_tokens);ends=torch.as_tensor([w[3] for w in words],device=runtime.device)
        # Each row comes from its actual native causal prefix. The last row reads a fixed suffix after the unmodified original text.
        sel=torch.cat([ends,torch.tensor([span[0]-1,len(ids)-1],device=runtime.device)]);onsets=F.linear(feat.hidden[sel],lookup).float()
        ctx=F.normalize(table[ids[:raw]].float(),dim=-1).cumsum(0);sd=F.normalize(ctx[torch.cat([ends,torch.tensor([span[0]-1,raw-1],device=runtime.device)])],dim=-1)@direction.T
        logits=F.linear(feat.hidden[span[0]-1:span[1]-1],table).float();target=ids[span[0]:span[1]];nll=F.cross_entropy(logits,target,reduction='none');trace=np.asarray([[tr[t][q] for q in ('unresolved','opened','resolved','transition')] for _,_,_,t in words],np.float32)
        if not np.all(np.isfinite(trace)) or not all(k in tr[0] for k in ('unresolved','opened','resolved','transition')):raise ValueError('Published trace schema mismatch')
        wordnll=torch.full((raw,),float('nan'),device=runtime.device)
        for lo in range(0,raw-1,8):
            hi=min(raw-1,lo+8);wordnll[lo+1:hi+1]=F.cross_entropy(F.linear(feat.hidden[lo:hi],table).float(),ids[lo+1:hi+1],reduction='none')
        perword=torch.stack([wordnll[a:b+1].sum() for _,_,a,b in words]);return tuple(v.detach().cpu().numpy() for v in (onsets,sd,nll))+ (trace,perword.cpu().numpy())
    with runtime.encoder.register_forward_pre_hook(count):
        for j,r in enumerate(rows):
            feat=runtime.encode(r['text']);z,s,nll,tr,wordnll=results(r,feat);key=r['id'];signals[key+'_onset']=z;signals[key+'_static']=s;signals[key+'_nll']=nll;signals[key+'_trace']=tr;signals[key+'_wordnll']=wordnll
            if j in (0,len(rows)//2):
                raw=r['text'][:r['words'][-1][1]];f=runtime.encode(raw);z0,s0,n0,t0,w0=results(dict(r,text=raw),f);err=float(np.abs(z[:-1]-z0[:-1]).max());terr=float(np.abs(tr-t0).max())
                if err>.02 or terr>.003 or not np.allclose(nll,n0,atol=.005):raise ArithmeticError('Terminal query affected an original Spanish word')
                checks.append(dict(original=r['id'],onsets_max=err,trace_max=terr))
            if (j+1)%100==0:print('Spanish scientific cue triplets',j+1,flush=True)
    if n!=len(rows)+len(checks):raise ValueError('Unexpected number of Spanish native forwards')
    p=out/'signals.npz';np.savez_compressed(p,**signals);write_json(out/'receipt.json',{'input':digest(source),'vocabulary':digest(vfile),'rows':len(rows),'forward_calls':n,'production_forwards':len(rows),'checks':checks,'revision':runtime.revision,'signals':digest(p),'word_points':21,'category_position_one_based':18,'seconds':perf_counter()-start})
