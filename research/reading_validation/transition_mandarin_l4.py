"""One original-presentation Qwen forward per distinct human Mandarin prompt."""

from argparse import ArgumentParser
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from research.backbone_transfer.extract import digest,write_json
from research.reading_validation.antecedent_inference import REVISION
from research.token_readout.backbone import Backbone


@torch.inference_mode()
def run(model,stimuli,output,input_sha256,protocol_sha256):
    if digest(stimuli)!=input_sha256:raise ValueError("Original Mandarin presentations changed")
    rows=list(map(json.loads,stimuli.read_text().splitlines()))
    if len(rows)!=220 or any(set(r)!={"id","text","context_end_token","tokens"} for r in rows):
        raise ValueError("An extraneous stimulus field or a human outcome was uploaded")
    rt=Backbone.load("qwen",model,native_attention=True)
    if rt.revision!=REVISION:raise ValueError("Inconsistent native observer")
    calls=0;results={};lasts={};started=perf_counter()
    def hook(*_):
        nonlocal calls
        calls+=1
    with rt.encoder.register_forward_pre_hook(hook):
        for i,r in enumerate(rows):
            f=rt.encode(r["text"])
            if len(f.token_ids)!=r["tokens"]:raise ValueError("Chinese string and declared tokenizer differ")
            end=r["context_end_token"]
            if not 0<=end<len(f.token_ids):raise ValueError("Chinese context end outside the original article")
            states=torch.stack([v for h in (f.mid,f.hidden) for v in (h[end],h[:end+1].mean(0))]).cpu().numpy()
            if calls!=i+1 or not np.isfinite(states).all():raise ValueError("Failed single original Mandarin presentation")
            results[r["id"]]=states;lasts[r["id"]]=tuple(f.token_ids[:end+1])
    groups={}
    for key,ids in lasts.items():groups.setdefault(ids,[]).append(key)
    if any(len(ids)!=2 for ids in groups.values()):raise ValueError("Every free prompt must match exactly one overt prompt")
    maxdiff=max(float(np.max(np.abs(results[a]-results[b]))) for a,b in groups.values())
    maxrms=max(float(np.linalg.norm(results[a]-results[b])/max(np.linalg.norm(results[a]),1e-12)) for a,b in groups.values())
    if maxrms>.001:raise ValueError(f"Shared Mandarin context changed with supplied pronoun: {maxrms}")
    output.mkdir(exist_ok=True,parents=True);p=output/"unprompted-states.npz";np.savez_compressed(p,**results)
    write_json(output/"receipt.json",{"input_sha256":input_sha256,"protocol_sha256":protocol_sha256,"revision":REVISION,
         "encoder_forwards":calls,"original_presentation_count":len(rows),"prefix_pairs":len(groups),"maximum_tokens":max(r["tokens"] for r in rows),
         "changed_suffix_rms":maxrms,"changed_suffix_absolute":maxdiff,"features_sha256":digest(p),"seconds":perf_counter()-started,
         "outcomes_on_accelerator":0,"native_width":rt.width})


if __name__=="__main__":
    p=ArgumentParser(description=__doc__)
    for arg in ("model","stimuli","output"):p.add_argument("--"+arg,type=Path,required=True)
    for arg in ("input-sha256","protocol-sha256"):p.add_argument("--"+arg,required=True)
    run(**vars(p.parse_args()))
