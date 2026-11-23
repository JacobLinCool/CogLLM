"""Measure unprompted native access to the same original word under three narrative continuations."""
from hashlib import sha256
import json
from math import log
from pathlib import Path
import re
from time import perf_counter

import numpy as np
import torch
from research.token_readout.model import question_mass
from research.reading_validation.transition_inference import REVISION

WORD=re.compile(r"[^\W\d_]+(?:[’'][^\W\d_]+)*",re.UNICODE)


def word_pieces(text,offsets):
    spans=[]
    for m in WORD.finditer(text):
        parts=np.flatnonzero((offsets[:,0]<m.end())&(offsets[:,1]>m.start()))
        if not len(parts):raise ValueError("Missing source lexical word")
        spans.append((m.group().casefold(),m.start(),m.end(),parts))
    return spans


def retrieve(r,word_spans,query_start,query_end,attention,unresolved):
    prior=[w for w in word_spans if w[2]<=query_start]
    query=[w for w in word_spans if w[1]>=query_start and w[2]<=query_end]
    if not prior or not query:raise ValueError("No query or first-paragraph antecedent position")
    q=np.array([w[3][-1] for w in query],int);n=len(prior);word_index=np.arange(n)
    # Lexical recency has all preceding printed words as its competitor set.
    rec=np.stack([1/(1+n+i-word_index) for i in range(len(query))]);rec/=rec.sum(1,keepdims=True)
    out={}
    for target in r["targets"]:
        stats={}
        for name,cap in (("all",query_start),("introduction",r["prior_end"])):
            match=[(i,w[3]) for i,w in enumerate(prior) if w[0]==target and w[2]<=cap]
            if not match:raise ValueError("Published target does not match an original word")
            mass=sum(attention[np.ix_(q,parts)].mean(1) for _,parts in match)
            raw=sum(unresolved[np.ix_(q,parts)].mean(1) for _,parts in match)
            vals={"native":log(max(1e-30,float(mass.mean())*n)),"recency":log(float(n*rec[:,[i for i,_ in match]].sum(1).mean())),
                  "uniform":log(len(match)),"trained_unresolved":float(raw.mean()),"occurrences":len(match)}
            if not all(np.isfinite(v) for v in vals.values()):raise ArithmeticError("Nonfinite goal accessibility")
            stats[name]=vals
        out[target]=stats
    return out


@torch.inference_mode()
def run(runtime,reader,checkpoint,input_path,output_dir):
    source=Path(input_path);out=Path(output_dir);out.mkdir(exist_ok=True,parents=True);started=perf_counter()
    if runtime.candidate!="qwen" or runtime.revision!=REVISION:raise ValueError("Unexpected original native encoder")
    n=[0];seen=set();fout=out/"goal-accessibility.jsonl"
    def count(*_):n[0]+=1
    with runtime.encoder.register_forward_pre_hook(count),fout.open("w") as f:
        for r in map(json.loads,source.read_text().splitlines()):
            text=r["text"];uid=sha256(text.encode()).hexdigest()
            if uid!=r["id"] or uid in seen or r["last_end"]!=len(text):raise ValueError("Unexpected original narrative")
            seen.add(uid);before=n[0];features=runtime.encode(text)
            if n[0]!=before+1:raise ValueError("Goal observation requires exactly one original text pass")
            offsets=np.asarray(features.offsets);words=word_pieces(text,offsets)
            prefix=runtime.tokenizer.encode(text[:r["first_end"]],add_special_tokens=False).ids
            if tuple(prefix)!=features.token_ids[:len(prefix)] or offsets[len(prefix)-1,1]!=r["first_end"]:
                raise ValueError("Shared first paragraph tokenization changes with a continuation")
            final,middle=reader.project(features);scores=reader.score_projected(final,middle,features.relations)
            sc=checkpoint["scale"];priors=checkpoint["priors"]
            candidate=torch.tensor([any(c.isalnum() for c in text[a:b]) for a,b in offsets],device=runtime.device)
            _,unresolved,_=question_mass(scores.opening*sc["opening_gain"]+priors["opening"],
                scores.resolution*sc["resolution_gain"]+sc["resolution_bias"]+priors["resolution"],candidate)
            pos=torch.arange(1,len(offsets)+1,device=runtime.device)
            att=(features.relations[...,0]+log(2)).expm1()/pos[:,None]
            att=att.cpu().numpy();unresolved=unresolved.cpu().numpy()
            report={k:retrieve(r,words,r[k+"_start"],r[k+"_end"],att,unresolved) for k in ("first","last")}
            report.update(id=uid,prefix_token_sha256=sha256(np.asarray(prefix,dtype="<i8").tobytes()).hexdigest(),
                          words=len(words),tokens=len(offsets))
            f.write(json.dumps(report,allow_nan=False)+"\n")
    if n[0]!=96:raise ValueError("Expected the complete public factorial exposure")
    (out/"receipt.json").write_text(json.dumps({"source_sha256":sha256(source.read_bytes()).hexdigest(),
        "forecast_sha256":sha256(fout.read_bytes()).hexdigest(),"native_forwards":n[0],"revision":runtime.revision,
        "participant_outcomes_in_input":0,"seconds":perf_counter()-started},indent=2)+"\n")
