"""Make outcome-blind earlier-word and previous all-action forecasts for complete English texts."""

from argparse import ArgumentParser
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from research.backbone_transfer.extract import digest, write_json
from research.reading_validation.antecedent_inference import REVISION, native_signals
from research.reading_validation.transition_gate_features import features
from research.reading_validation.transition_pointer_features import design
from research.reading_validation.transition_pointer_l4 import watch
from research.token_readout.backbone import Backbone


@torch.inference_mode()
def run(model,source_model,gate_model,stimuli,output,input_sha256,protocol_sha256):
    start=perf_counter()
    if digest(stimuli)!=input_sha256:
        raise ValueError("The sealed mixed-action archive has changed")
    prior=source_model/"source-weights.npz"
    gate=gate_model/"gate-weights.npz"
    if digest(prior)!="9162700762a7574f45532bac88f8c018b1f2922118ac033bd1206a1edaecfaf5":
        raise ValueError("Frozen prior-word weights differ")
    if digest(gate)!=json.loads((gate_model/"receipt.json").read_text())["weights_sha256"]:
        raise ValueError("Original factorized comparison weights differ")
    rows=list(map(json.loads,stimuli.read_text().splitlines()))
    if any(set(r)&{"actions","clicks","responses","voters"} for r in rows):
        raise ValueError("The native worker accepts words only")
    p,g=np.load(prior),np.load(gate)
    runtime=Backbone.load("qwen",model,native_attention=True)
    if runtime.revision!=REVISION:raise ValueError("Qwen revision differs")
    logits={};base_scores={};calls=0
    def count(*args):
        nonlocal calls
        calls+=1
    with runtime.encoder.register_forward_pre_hook(count):
        for i,row in enumerate(rows):
            qs=row["queries"]
            if not qs:
                runtime.encode(row["text"])
                continue
            ids=torch.tensor(qs,device=runtime.device)
            locations=torch.tensor(row["tokens"],device=runtime.device)
            starts,ends=locations[:,0],locations[:,1]
            with watch(runtime,row["token_count"],ends[ids],starts,ends,False) as sparse:
                base=runtime.encode(row["text"])
            if len(base.token_ids)!=row["token_count"] or any(base.offsets[j][1]!=b for (_,j),(_,b) in zip(row["tokens"],row["words"],strict=True)):
                raise ValueError("The displayed words and native BPE did not align")
            states=torch.cat([F.layer_norm(z[ends[ids]],(runtime.width,)) for z in (base.mid,base.hidden)],1).cpu().numpy()
            signals=native_signals(base,sparse,ids,starts,ends)
            for k,q in enumerate(qs):
                key=row["id"]+"/"+str(q);surface=design(row,q,k);native=np.column_stack((surface,signals[k,:q]))
                logits[key]=np.stack([((x-p[m+"__mean"])/p[m+"__sd"])@p[m+"__beta"]
                                      for m,x in (("lexical",surface),("headwise",native))]).astype(np.float32)
                tags=list(row["pos"][:q]);tags[-1]=row["pos_last"][k]
                control=features(row,q,tags);both=np.r_[control,states[k]]
                base_scores[key]=[float(1/(1+np.exp(-((x-g[m+"__mean"])/g[m+"__sd"])@g[m+"__beta"]-g[m+"__bias"])))
                                  for m,x in (("surface",control),("native",both))]
            if calls!=i+1:raise ValueError("The article required an extra encoder pass")
    if calls!=len(rows) or len(logits)!=sum(len(r["queries"]) for r in rows):raise ValueError("Source rows missing")
    output.mkdir(parents=True,exist_ok=True)
    result=output/"all-prior-logits.npz";np.savez_compressed(result,**logits)
    previous=output/"factorized-forecasts.json";write_json(previous,base_scores)
    write_json(output/"receipt.json",{"input_sha256":input_sha256,"protocol_sha256":protocol_sha256,"revision":REVISION,
       "encoder_forwards":calls,"eligible_pronouns":len(logits),"candidate_words":sum(v.shape[1] for v in logits.values()),
       "source_position_sha256":digest(prior),"source_existence_sha256":digest(gate),"forecasts_sha256":digest(result),
       "factorized_sha256":digest(previous),"held_responses_provided":0,"seconds":perf_counter()-start})


if __name__=="__main__":
    parser=ArgumentParser(description=__doc__)
    for key in ("model","source-model","gate-model","stimuli","output"):parser.add_argument("--"+key,type=Path,required=True)
    for key in ("input-sha256","protocol-sha256"):parser.add_argument("--"+key,required=True)
    run(**vars(parser.parse_args()))
