"""Capture exactly the source word sketch on extra articles with one original forward each."""
from pathlib import Path
import json
import numpy as np
import torch
from torch.nn import functional as F
from research.backbone_transfer.extract import digest,stream_rows,write_json
from research.reading_validation.referent_sketch_l4 import DIM,SEED,REVISION

@torch.inference_mode()
def run(runtime,reader,packet,source,out):
    if runtime.revision!=REVISION:raise ValueError("Different reading backbone")
    out=Path(out);out.mkdir();rows=list(stream_rows(source));rng=np.random.default_rng(SEED)
    proj=[torch.as_tensor(rng.choice([-1.,1.],(runtime.width,DIM)).astype("float32")/np.float32(runtime.width**.5),device=runtime.device) for _ in range(2)]
    calls=0;data={};tokens=0
    def count(*_):
        nonlocal calls
        calls+=1
    with runtime.encoder.register_forward_pre_hook(count):
        for i,r in enumerate(rows):
            if set(r)!={"id","text","words","tokens"}:raise ValueError("Outcomes entered new exposure")
            enc=runtime.tokenizer.encode(r["text"],add_special_tokens=False);ids=torch.tensor([enc.ids],device=runtime.device)
            end=torch.tensor([t[1] for t in r["tokens"]],device=runtime.device)
            if any(enc.offsets[j][1]!=b for (_,j),(_,b) in zip(r["tokens"],r["words"],strict=True)):raise ValueError("Uncertified word endpoint")
            mid=[]
            def remember(module,args,v):mid.append(v[0])
            with runtime.encoder.layers[runtime.mid_index].register_forward_hook(remember):
                last=runtime.encoder(input_ids=ids,use_cache=False,output_attentions=False,output_hidden_states=False).last_hidden_state[0]
            if calls!=i+1 or len(mid)!=1:raise ValueError("Article was exposed more than once")
            h=torch.cat([F.layer_norm(v[end],(runtime.width,))@m for v,m in zip((mid[0],last),proj,strict=True)],dim=1)
            data[r["id"]+"__h"]=h.half().cpu().numpy();tokens+=len(enc.ids)
    file=out/"word-states.npz";np.savez_compressed(file,**data)
    write_json(out/"receipt.json",{"input_sha256":digest(source),"archive_sha256":digest(file),"forwards":calls,"articles":len(rows),
        "words":sum(map(len,data.values())),"tokens":tokens,"external_responses":0,"model":runtime.revision,"seed":SEED})
