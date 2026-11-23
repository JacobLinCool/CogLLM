"""Outcome-blind standard and directed one-pass discourse forecasts on whole articles."""
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
import torch
from torch.nn import functional as F

from research.reading_validation.bridge_forecast import STOP, FEATURES, contiguous_overlap, norm_word
from research.token_readout.runner import trace_rows

PROMPT = ("Assess how each new sentence fits the preceding writing, including the topic or title when present. "
          "Answer Yes if the sentence is understandable and connected, even when it introduces a new idea. "
          "Answer No if the connection is missing, a reference is unclear, or the new claim is inconsistent. "
          "Earlier answers may be blank. Respond with Yes or No only.")
ASSISTANT = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def form(row):
    prefix="<|im_start|>system\n"+PROMPT+"<|im_end|>\n"
    pos=0; cuts=[]
    for stop in row["stops"]:
        if not pos <= stop["start"] < stop["end"]:
            raise ValueError("Directed marker order must follow the original text")
        background=row["text"][pos:stop["start"]].strip()
        sentence=row["text"][stop["start"]:stop["end"]].strip()
        msg=(("Previously read:\n"+background+"\n") if background else "")+"New sentence:\n"+sentence
        prefix += "<|im_start|>user\n"+msg+"\nDoes it fit?"+ASSISTANT
        cuts.append((stop["id"],len(prefix)))
        prefix += "<|im_end|>\n"
        pos=stop["end"]
    return prefix,cuts


def directed(runtime,row):
    text,cuts=form(row)
    enc=runtime.tokenizer.encode(text,add_special_tokens=False)
    if len(enc.ids)>2048:
        raise ValueError("Directed document exceeds the verified context limit")
    ends={e:i for i,(_,e) in enumerate(enc.offsets)}
    idx=[ends[c] for _,c in cuts]
    if any(enc.offsets[i][1]!=c or (i+1 < len(enc.ids) and enc.offsets[i+1][0]<c) for i,(_,c) in zip(idx,cuts,strict=True)):
        raise ValueError("Directed word-boundary representation requires an exact completed marker")
    answers=[runtime.tokenizer.encode(a,add_special_tokens=False).ids for a in ("No","Yes")]
    if [len(a) for a in answers]!=[1,1] or answers[0]==answers[1]:
        raise ValueError("Directed forecast requires one distinct token per answer")
    h=runtime.encoder(input_ids=torch.tensor([enc.ids],device=runtime.device),use_cache=False,
                      output_attentions=False,output_hidden_states=False).last_hidden_state[0,idx]
    directions=runtime.encoder.embed_tokens.weight[[a[0] for a in answers]]
    logits=F.linear(h,directions)
    return dict(zip((bid for bid,_ in cuts),(logits[:,0]-logits[:,1]).tolist(),strict=True))


def project(runtime,reader,checkpoint,row):
    f=runtime.encode(row["text"])
    final,middle=reader.project(f)
    sc=reader.score_projected(final,middle,f.relations)
    scale=checkpoint["scale"]
    sc=replace(sc,opening=sc.opening*scale["opening_gain"],
               resolution=sc.resolution*scale["resolution_gain"]+scale["resolution_bias"])
    traces=trace_rows(row["text"],f.offsets,sc,checkpoint["priors"],reader.anchor_tokens)
    return pooled_boundary_features(runtime,f,middle,final,traces,row)


def token_surprisals(runtime,features):
    """Apply the tied vocabulary head in bounded blocks to cached causal token states."""
    ids=torch.tensor(features.token_ids[1:],device=runtime.device)
    losses=np.zeros(len(features.token_ids),dtype=np.float64)
    for start in range(0,len(ids),32):
        target=ids[start:start+32]
        logits=F.linear(features.hidden[start:start+len(target)],runtime.encoder.embed_tokens.weight)
        losses[start+1:start+len(target)+1]=F.cross_entropy(logits,target,reduction="none").cpu().numpy()
    return losses


def pooled_boundary_features(runtime,f,middle,final,traces,row,token_surprisal=None):
    """Recover released sentence features from cached encoder output and calibrated token scores."""
    transition=np.asarray([s["transition"] for s in traces])
    offsets=np.asarray(f.offsets)
    real=offsets[:,1]>offsets[:,0]
    tmask=np.array([any(c.isalnum() for c in row["text"][s:e]) for s,e in f.offsets])
    word_ranges=list(re.finditer(r"\S+",row["text"]))
    words=np.array([norm_word(m.group()) for m in word_ranges])
    word_idx=np.full(len(real),-1)
    for i,m in enumerate(word_ranges):
        aligned=(offsets[:,0]<m.end()) & (offsets[:,1]>m.start()) & real
        if np.any(word_idx[aligned]>=0):
            raise ValueError("Token straddles orthographic words")
        word_idx[aligned]=i
    content_words=np.array([w!="" and w not in STOP for w in words])
    content=(word_idx>=0) & content_words[word_idx] & tmask
    wstart=np.array([m.start() for m in word_ranges]); wend=np.array([m.end() for m in word_ranges])
    surprisal=token_surprisals(runtime,f) if token_surprisal is None else token_surprisal
    if np.shape(surprisal)!=(len(f.token_ids),) or not np.isfinite(surprisal).all():
        raise ValueError("Source token surprisal is missing or misaligned")
    n=len(f.offsets); rel=(f.relations[...,0]+math.log(2)).expm1()/torch.arange(1,n+1,device=runtime.device)[:,None]
    rel=rel.cumsum(dim=1).cpu().numpy()
    normalized=[F.normalize(a.float(),dim=-1).cpu().numpy() for a in (f.mid,f.hidden)]
    projected=[x.float().cpu().numpy() for x in (middle,final)]
    def mean(states,mask):
        return states[mask].mean(axis=0) if mask.any() else np.zeros(states.shape[1],dtype=states.dtype)
    rows=[]; raw={}
    for stop in row["stops"]:
        s,e=stop["start"],stop["end"]
        if row["text"][s:e]!=row["text"][s:e].strip():
            raise ValueError("Untrimmed sentence span")
        ctx=(offsets[:,1]<=s) & real
        current=(offsets[:,0]<e)&(offsets[:,1]>s)&real
        if not current.any() or not ctx.any() or (offsets[:,1][current]>e).any() or (ctx&current).any():
            raise ValueError("Unaligned target or preamble")
        lexical=current & (word_idx>=0)
        wc=words[wend<=s].tolist(); wt=words[(wstart<e)&(wend>s)].tolist()
        cs={w for w in wc if w and w not in STOP};ts={w for w in wt if w and w not in STOP}
        cctx=ctx&content; ccur=current&content
        cosine=[]
        for layer in normalized:
            a,b=mean(layer,cctx),mean(layer,ccur)
            denom=np.linalg.norm(a)*np.linalg.norm(b)
            cosine.append(float(np.dot(a,b)/denom) if denom>0 else 0.)
        feats=[math.log1p(len(wc)),math.log1p(len(wt)),len(cs&ts)/max(1,len(ts)),
               contiguous_overlap(wc,wt),float(transition[lexical].mean()),float(surprisal[lexical].sum()/len(wt)),
               float(rel[lexical,np.where(ctx)[0][-1]].mean()),*cosine]
        current_proj=np.concatenate([mean(x,current) for x in projected])
        prev_proj=np.concatenate([mean(x,ctx) for x in projected])
        rep=np.concatenate([current_proj,prev_proj,current_proj*prev_proj])
        rows.append({"id":stop["id"],"features":feats,"representation":rep.tolist(),"position":stop["index"]})
        if stop["archive_final"]:
            raw[stop["id"]]=f.hidden[np.where(current)[0][-1]].cpu().numpy().copy()
    return rows,raw


@torch.inference_mode()
def run(runtime,reader,checkpoint,input_path,output_dir):
    path=Path(input_path); raw=path.read_bytes()
    data=[json.loads(x) for x in raw.splitlines()]
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    if runtime.candidate!="qwen" or not runtime.encoder.config.tie_word_embeddings:
        raise ValueError("This study requires the pinned tied Qwen 2B model")
    count=[0]
    def observe(_m,_args):
        count[0]+=1
    raw_vectors={}
    with runtime.encoder.register_forward_pre_hook(observe), (out/"features.jsonl").open("w") as of:
        for i,row in enumerate(data):
            before=count[0]
            rows,raw_row=project(runtime,reader,checkpoint,row)
            if count[0]-before!=1:
                raise RuntimeError("Raw article readout must use one forward")
            dirs=directed(runtime,row)
            if count[0]-before!=2:
                raise RuntimeError("Directed comparator must use one independent forward")
            raw_vectors.update(raw_row)
            for r in rows:
                r["directed"]=dirs[r["id"]]
                of.write(json.dumps(r,allow_nan=False)+"\n")
            if i%32==31:
                of.flush()
                print(f"Coherence articles {i+1}/{len(data)}",flush=True)
    np.savez_compressed(out/"heldout_sentence_states.npz",**raw_vectors)
    files={s.name:hashlib.sha256(s.read_bytes()).hexdigest() for s in out.iterdir() if s.is_file() and s.name!="receipt.json"}
    (out/"receipt.json").write_text(json.dumps({"candidate":runtime.candidate,"revision":runtime.revision,
       "input_sha256":hashlib.sha256(raw).hexdigest(),"documents":len(data),"raw_forward_calls":len(data),
       "directed_forward_calls":len(data),"total_encoder_calls":count[0],"archived_final_states":len(raw_vectors),
       "features":FEATURES,"files":files},indent=2))
