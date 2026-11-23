"""Capture random-projected causal word states and source-trained attention priors in one pass."""
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from research.backbone_transfer.extract import digest, write_json
from research.reading_validation.transition_pointer_features import FORMS
from research.reading_validation.transition_pointer_l4 import REVISION, watch
from research.token_readout.backbone import Backbone

DIM = 64
SEED = 61392

def anchors(row):
    return [i for i,(a,b) in enumerate(row["words"]) if i and row["text"][a:b].casefold() in FORMS]

@torch.inference_mode()
def extract(runtime, row, proj, beta):
    text = row["text"]
    indices = anchors(row)
    enc = runtime.tokenizer.encode(text,add_special_tokens=False)
    if any(enc.offsets[j][1]!=b for (_,j),(_,b) in zip(row["tokens"],row["words"],strict=True)):
        raise ValueError("Word endpoint does not match tokenizer")
    spans = torch.tensor(row["tokens"],device=runtime.device)
    start,end = spans.T
    ix = torch.tensor(indices or [0], device=runtime.device)
    ids = torch.tensor([enc.ids],device=runtime.device)
    middle = []
    def capture(module,args,result):
        middle.append(result[0])
    with watch(runtime,len(enc.ids),end[ix],start,end,False) as cap, runtime.encoder.layers[runtime.mid_index].register_forward_hook(capture):
        final = runtime.encoder(input_ids=ids,use_cache=False,output_attentions=False,output_hidden_states=False).last_hidden_state[0]
    if len(middle)!=1 or len(cap.seen)!=len(runtime.global_layers):
        raise ValueError("Missing native layer capture")
    words = [x[end] for x in (middle[0],final)]
    sketch = torch.cat([F.layer_norm(x,(runtime.width,)) @ m for x,m in zip(words,proj,strict=True)],dim=1)
    normalized = [F.normalize(x) for x in words]
    heads = torch.cat([cap.seen[i] for i in sorted(cap.seen)],0)
    if heads.shape[0]!=48:
        raise ValueError("Different global head geometry")
    prior = sum(b*(x[ix]@x.T).clamp(-1,1) for b,x in zip(beta[:2],normalized,strict=True))
    prior += beta[2]*torch.log1p(heads.mean(0))+beta[3]*torch.log1p(heads.amax(0))
    prior += torch.einsum("h,hij->ij",beta[4:],torch.log1p(heads))
    selected = torch.cat([prior[i,:q] for i,q in enumerate(indices)]) if indices else prior.new_empty(0)
    return sketch.to(torch.float16).cpu().numpy(),selected.cpu().numpy(),len(enc.ids)

def run(model,stimuli,output,input_sha256,protocol_sha256,seal):
    output = Path(output)
    receipt = json.loads(Path(seal).read_text())
    if digest(stimuli)!=input_sha256 or input_sha256!=receipt["input_sha256"] or protocol_sha256!=receipt["protocol_sha256"]:
        raise ValueError("Sealed exposure differs")
    rows = list(map(json.loads,Path(stimuli).read_text().splitlines()))
    if len(rows)!=receipt["articles"] or any(set(r)!={"id","text","words","tokens"} for r in rows):
        raise ValueError("Original-text-only worker received extra inputs")
    rt = Backbone.load("qwen",model,native_attention=True)
    if rt.revision!=REVISION:
        raise ValueError("Wrong frozen backbone")
    rng = np.random.default_rng(SEED)
    proj = [torch.as_tensor(rng.choice([-1.,1.],(rt.width,DIM)).astype("float32")/np.float32(rt.width**.5),device=rt.device) for _ in range(2)]
    beta = torch.tensor(receipt["native_weights"],device=rt.device,dtype=torch.float32)
    output.mkdir(parents=True,exist_ok=True)
    calls = 0
    def counted(*args):
        nonlocal calls
        calls += 1
    start = perf_counter()
    cache = {};wc=0;pc=0;tc=0
    with rt.encoder.register_forward_pre_hook(counted):
        for j,r in enumerate(rows):
            h,z,n = extract(rt,r,proj,beta)
            if h.shape!=(len(r["words"]),2*DIM) or not np.isfinite(h).all() or len(z)!=sum(anchors(r)) or not np.isfinite(z).all():
                raise ValueError("Malformed native score or word state")
            cache[r["id"]+"__h"]=h;cache[r["id"]+"__b"]=z
            wc+=len(h);pc+=len(z);tc+=n
            if calls!=j+1:
                raise ValueError("Original article requires one exposure")
            if calls%400==0:
                print(f"Shared-reference word states {calls}/{len(rows)}",flush=True)
    f=output/"word-states.npz"
    np.savez_compressed(f,**cache)
    write_json(output/"receipt.json",{"input_sha256":input_sha256,"protocol_sha256":protocol_sha256,"archive_sha256":digest(f),
               "source_weights_sha256":receipt["source_weights_sha256"],"model_revision":REVISION,"model":"Qwen3.5-2B",
               "forwards":calls,"articles":len(rows),"words":wc,"candidate_prior_scores":pc,"tokens":tc,"sketch_dimensions":2*DIM,
               "seed":SEED,"external_outcomes":0,"seconds":perf_counter()-start})
