"""Evaluate original noun probes, preserving the complete hyphenated object from the same article pass."""
from hashlib import sha256
import json
from math import log
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from research.reading_validation.transition_goal_access_l4 import retrieve, word_pieces
from research.reading_validation.transition_inference import REVISION
from research.token_readout.model import question_mass


def object_spans(text, offsets, target):
    ordinary=word_pieces(text,offsets)
    if target!="t-shirt":return ordinary
    out=[];i=0
    while i<len(ordinary):
        w=ordinary[i]
        if (w[0]=="t" and i+1<len(ordinary) and ordinary[i+1][0]=="shirt"
                and text[w[1]:ordinary[i+1][2]].casefold()=="t-shirt"):
            end=ordinary[i+1][2]
            pieces=np.flatnonzero((offsets[:,0]<end)&(offsets[:,1]>w[1]))
            out.append((target,w[1],end,pieces));i+=2
        else:
            out.append(w);i+=1
    return out


@torch.inference_mode()
def run(runtime, reader, checkpoint, input_path, output_dir):
    source=Path(input_path);out=Path(output_dir);out.mkdir(exist_ok=True,parents=True)
    if runtime.candidate!="qwen" or runtime.revision!=REVISION:raise ValueError("Unregistered native encoder")
    start=perf_counter();calls=[0];seen=set();file=out/"goal-accessibility.jsonl"
    def counter(*_):calls[0]+=1
    with runtime.encoder.register_forward_pre_hook(counter),file.open("w") as f:
        for r in map(json.loads,source.read_text().splitlines()):
            text=r["text"];uid=sha256(text.encode()).hexdigest()
            if uid!=r["id"] or uid in seen or r["last_end"]!=len(text):raise ValueError("Original text mismatch")
            seen.add(uid);before=calls[0];features=runtime.encode(text)
            if before+1!=calls[0]:raise ValueError("Unexpected number of causal encodings")
            off=np.asarray(features.offsets);words=object_spans(text,off,r["targets"][0])
            ids=runtime.tokenizer.encode(text[:r["first_end"]],add_special_tokens=False).ids
            if tuple(ids)!=features.token_ids[:len(ids)] or off[len(ids)-1,1]!=r["first_end"]:
                raise ValueError("Future paragraph altered initial processing")
            final,middle=reader.project(features)
            scores=reader.score_projected(final,middle,features.relations)
            prior,scale=checkpoint["priors"],checkpoint["scale"]
            eligible=torch.tensor([any(c.isalnum() for c in text[a:b]) for a,b in off],device=runtime.device)
            _,mass,_=question_mass(scores.opening*scale["opening_gain"]+prior["opening"],
                         scores.resolution*scale["resolution_gain"]+scale["resolution_bias"]+prior["resolution"],eligible)
            pos=torch.arange(1,len(off)+1,device=runtime.device)
            att=((features.relations[...,0]+log(2)).expm1()/pos[:,None]).cpu().numpy()
            mass=mass.cpu().numpy()
            result={k:retrieve(r,words,r[k+"_start"],r[k+"_end"],att,mass) for k in ("first","last")}
            result.update(id=uid,prefix_token_sha256=sha256(np.asarray(ids,dtype="<i8").tobytes()).hexdigest(),words=len(words),tokens=len(off))
            f.write(json.dumps(result,allow_nan=False)+"\n")
    if calls[0]!=96:raise ValueError("Missing original object condition")
    (out/"receipt.json").write_text(json.dumps(dict(source_sha256=sha256(source.read_bytes()).hexdigest(),
         forecast_sha256=sha256(file.read_bytes()).hexdigest(),native_forwards=calls[0],revision=runtime.revision,
         participant_outcomes_in_input=0,seconds=perf_counter()-start),indent=2)+"\n")
