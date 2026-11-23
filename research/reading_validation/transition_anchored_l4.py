"""Extract frozen earlier-word priors and incremental evidence at every measured word in one causal pass."""
from argparse import ArgumentParser
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from research.backbone_transfer.extract import digest,write_json
from research.reading_validation.antecedent_inference import REVISION
from research.reading_validation.transition_dpr_l4 import ARMS,extract,word_logits
from research.token_readout.backbone import Backbone


PRIOR_SHA="9162700762a7574f45532bac88f8c018b1f2922118ac033bd1206a1edaecfaf5"
UPDATE_SHA="ddda847cbaf7e0280ff74fbe13a7632d83f0f3dc26dda6a9ee74561403d9284f"


@torch.inference_mode()
def run(model,source_model,update_model,stimuli,output,input_sha256,protocol_sha256):
    pfile=source_model/"source-weights.npz";ufile=update_model/"source-semantic-weights.npz"
    if digest(stimuli)!=input_sha256 or digest(pfile)!=PRIOR_SHA or digest(ufile)!=UPDATE_SHA:raise ValueError("Source priors or evidence changed")
    rows=list(map(json.loads,stimuli.read_text().splitlines()));prior=np.load(pfile);updates=np.load(ufile)
    rt=Backbone.load("qwen",model,native_attention=True)
    if rt.revision!=REVISION:raise ValueError("Unexpected Qwen revision")
    t0=perf_counter();calls=0;words={};states={}
    def count(*_):
        nonlocal calls
        calls+=1
    with rt.encoder.register_forward_pre_hook(count):
        for i,row in enumerate(rows):
            if set(row["measurement"])!={"q","k"} or set(row)&{"choices","actions","names","answer","condition","question","annotation"}:
                raise ValueError("Native extraction accepts only the original exposure and measurement positions")
            q=row["measurement"]["q"];k=row["measurement"]["k"]
            if not 0<q<=k<len(row["words"]):raise ValueError("Invalid measurement positions")
            x=extract(rt,row);control=x["C"];full=np.column_stack((control,x["a"][q]))
            words[row["id"]]=np.stack([((z-prior[a+"__mean"])/prior[a+"__sd"])@prior[a+"__beta"]
                                      for a,z in (("lexical",control),("headwise",full))]).astype(np.float32)
            states[row["id"]]=np.stack([word_logits(x,q,k,updates[a]) for a in ARMS]).astype(np.float32)
            if calls!=i+1:raise ValueError("A sentence did not make exactly one encoder pass")
    output.mkdir(parents=True,exist_ok=True)
    pp=output/"prior-words.npz";up=output/"evidence-words.npz"
    np.savez_compressed(pp,**words);np.savez_compressed(up,**states)
    write_json(output/"receipt.json",{"input_sha256":input_sha256,"protocol_sha256":protocol_sha256,"prior_weights_sha256":PRIOR_SHA,
         "evidence_weights_sha256":UPDATE_SHA,"prior_forecast_sha256":digest(pp),"evidence_forecast_sha256":digest(up),
         "encoder_forwards":calls,"training_records":0,"external_response_fields":0,"revision":REVISION,"seconds":perf_counter()-t0})


if __name__=="__main__":
    parser=ArgumentParser(description=__doc__)
    for k in ("model","source-model","update-model","stimuli","output"):parser.add_argument("--"+k,type=Path,required=True)
    for k in ("input-sha256","protocol-sha256"):parser.add_argument("--"+k,required=True)
    run(**vars(parser.parse_args()))
