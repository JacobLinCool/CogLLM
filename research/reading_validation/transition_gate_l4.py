"""Fit the missing human choice of whether to point to an earlier word."""

from argparse import ArgumentParser
from collections import Counter
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from research.backbone_transfer.extract import digest, write_json
from research.reading_validation.antecedent_inference import REVISION, native_signals
from research.reading_validation.transition_direct_l4 import learn
from research.reading_validation.transition_gate_features import NAMES, features
from research.reading_validation.transition_pointer_features import design
from research.reading_validation.transition_pointer_l4 import watch
from research.token_readout.backbone import Backbone


@torch.inference_mode()
def collect(runtime, rows, pointer):
    matrices, index, conditional = [], [], {}
    passes = 0
    def count(*_):
        nonlocal passes
        passes += 1
    with runtime.encoder.register_forward_pre_hook(count):
        for i,row in enumerate(rows):
            qs=row["queries"]
            if not qs:
                runtime.encode(row["text"])
                continue
            device=runtime.device
            ids=torch.tensor(qs,device=device)
            locations=torch.tensor(row["tokens"],device=device)
            starts,ends=locations[:,0],locations[:,1]
            with watch(runtime,row["token_count"],ends[ids],starts,ends,False) as sparse:
                base=runtime.encode(row["text"])
            if len(base.token_ids)!=row["token_count"] or any(base.offsets[j][1]!=b for (_,j),(_,b) in zip(row["tokens"],row["words"],strict=True)):
                raise ValueError("Native discourse-action features and original offsets differ")
            state=torch.cat(tuple(F.layer_norm(x[ends[ids]],(runtime.width,)) for x in (base.mid,base.hidden)),1).cpu().numpy()
            blocks=native_signals(base,sparse,ids,starts,ends) if row["fold"]!="train" else None
            for k,q in enumerate(qs):
                tags=list(row["pos"][:q]);tags[-1]=row["pos_last"][k]
                control=features(row,q,tags)
                matrices.append(np.concatenate((control,state[k])))
                key=row["id"]+"/"+str(q)
                action=row.get("actions",{}).get(str(q))
                prior=(sum(t[3] for t in action["actions"] if t[0]=="DO")/sum(t[3] for t in action["actions"])) if action else -1.
                index.append((key,row["group"],row["fold"],prior))
                if blocks is not None:
                    z=(np.column_stack((design(row,q,k),blocks[k,:q])).astype(float)-pointer["headwise__mean"])/pointer["headwise__sd"]
                    logits=z@pointer["headwise__beta"]
                    conditional[key]=[int(logits.argmax()),float(1/np.exp(logits-logits.max()).sum())]
            if passes!=i+1:
                raise ValueError("The complete discourse-action article was not processed in one forward")
    if passes!=len(rows):raise ValueError("An original exposure was omitted")
    return np.stack(matrices),index,conditional,passes


def run(model,source_model,stimuli,output,input_sha256,protocol_sha256):
    if digest(stimuli)!=input_sha256:raise ValueError("Gate input differs")
    prior=source_model/"source-weights.npz"
    if digest(prior)!="9162700762a7574f45532bac88f8c018b1f2922118ac033bd1206a1edaecfaf5":
        raise ValueError("Frozen positioning coefficients changed")
    rows=list(map(json.loads,stimuli.read_text().splitlines()))
    if any(("actions" in r)!=(r["fold"]!="test") for r in rows):raise ValueError("Gold response in a gate forward")
    start=perf_counter();r=Backbone.load("qwen",model,native_attention=True)
    if r.revision!=REVISION:raise ValueError("Unexpected Qwen revision")
    matrix,index,confidence,calls=collect(r,rows,np.load(prior))
    d=r.width
    select=np.array([fold=="train" and y>=0 for key,group,fold,y in index]);owners=[g for (_,g,_,_),m in zip(index,select,strict=True) if m]
    counts=Counter(owners);mass=np.array([1/(len(counts)*counts[g]) for g in owners])
    truth=np.array([y for _,_,_,y in index])[select]
    if not np.isclose(mass.sum(),1):raise ValueError("Training works were not equally weighted")
    y=torch.tensor(truth,device=r.device,dtype=torch.float64)[:,None]
    w=torch.tensor(mass,device=r.device,dtype=torch.float64)[:,None]
    fits={};saved={};estimates={}
    for name,width in (("surface",len(NAMES)),("native",len(NAMES)+2*d)):
        X=torch.as_tensor(matrix[:,:width],device=r.device,dtype=torch.float64);training=X[select]
        mu=training.mean(0);sd=training.std(0,unbiased=False);sd=torch.where(sd>1e-10,sd,torch.ones_like(sd))
        if name=="native":sd[-2*d:]*=math.sqrt(d/48)
        X=(X-mu)/sd
        beta,bias,fit=learn(X[select],y,w,0,False)
        if fit["max_abs_gradient"]>1e-6:raise ValueError("Discourse-action gate did not converge")
        scores=(X@beta+bias).sigmoid().squeeze(1).cpu().numpy();fits[name]=fit
        for key,z in (("mean",mu),("sd",sd),("beta",beta[:,0]),("bias",bias[0])):saved[name+"__"+key]=z.cpu().numpy()
        for k,(key,_,fold,_) in enumerate(index):
            if fold!="train":estimates.setdefault(key,{})[name]=float(scores[k])
    dest=output;dest.mkdir(parents=True,exist_ok=True)
    for key,v in estimates.items():
        v["top"],v["conditional_confidence"]=confidence[key]
    write_json(dest/"word-forecasts.json",estimates);np.savez_compressed(dest/"gate-weights.npz",**saved)
    write_json(dest/"receipt.json",{"input_sha256":input_sha256,"protocol_sha256":protocol_sha256,"candidate":"qwen",
         "revision":REVISION,"source_pointer_sha256":digest(prior),"encoder_forwards":calls,"eligible_training_pronouns":int(select.sum()),
         "source_training_works":len(counts),"source_work_weighted_rate":float(mass@truth),"surface_columns":list(NAMES),
         "fits":fits,"weights_sha256":digest(dest/"gate-weights.npz"),"forecasts_sha256":digest(dest/"word-forecasts.json"),
         "seconds":perf_counter()-start})


if __name__=="__main__":
    p=ArgumentParser(description=__doc__)
    for arg in ("model","source-model","stimuli","output"):p.add_argument("--"+arg,type=Path,required=True)
    for arg in ("input-sha256","protocol-sha256"):p.add_argument("--"+arg,required=True)
    run(**vars(p.parse_args()))
