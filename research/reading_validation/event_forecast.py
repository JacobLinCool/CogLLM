"""Extract causal event-pair features for every sentence pair in each five-sentence story."""

from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
import re
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from research.reading_validation.identity_features import STOP


PAIRS=list(combinations(range(5),2))
WORD=re.compile(r"[^\W_]+")
SIZE=32
N_POSITION=5
N_LEXICAL=11


def projection(device,width):
    rng=np.random.default_rng(24119)
    q,_=np.linalg.qr(rng.standard_normal((width,SIZE)))
    return torch.from_numpy(q.astype(np.float32)).to(device)


def design(runtime,reader,row,proj):
    sent=row["spans"]
    if len(sent)!=5:
        raise ValueError("Expected five fixed source sentence spans")
    text=row["text"]
    ft=runtime.encode(text)
    if ft.relations is None:
        raise ValueError("Native attention is required")
    rez=reader(ft).resolution
    embedding=runtime.encoder.embed_tokens(torch.tensor(ft.token_ids,device=ft.hidden.device)).float()
    reps=[F.normalize(m.float(),dim=1) for m in (embedding,ft.mid,ft.hidden)]
    lex=[];positions=[];reduced=[];state=[]
    for a,b in sent:
        tok=runtime.tokenizer.encode(text[:b],add_special_tokens=False).ids
        all_pos=[i for i,(l,h) in enumerate(ft.offsets) if l<b and h>a]
        if not all_pos or tuple(tok)!=tuple(ft.token_ids[:all_pos[-1]+1]):
            raise ValueError("A story sentence endpoint is not prefix-certified")
        words=WORD.findall(text[a:b].casefold())
        content=[i for i in all_pos if any(w not in STOP and len(w)>1 for w in
                 WORD.findall(text[max(ft.offsets[i][0],a):min(ft.offsets[i][1],b)].casefold()))]
        # Sentences without a content word still have a causal sentence endpoint.
        tokens=torch.tensor(content or [all_pos[-1]],device=ft.hidden.device)
        positions.append(tokens)
        lex.append((words,set(words)-STOP))
        reduced.append(torch.cat([F.normalize(rep[tokens].mean(dim=0),dim=0)@proj for rep in reps[1:]]))
        state.append(torch.cat([rep[all_pos[-1]]@proj for rep in reps[1:]]))
    features=[]
    for a,b in PAIRS:
        wa,sa=lex[a]; wb,sb=lex[b]
        overlap=sa&sb
        scalars=[float(a),float(b),float(b-a),np.log1p(len(wa)),np.log1p(len(wb)),
                 len(overlap)/max(len(sa|sb),1),len(overlap)/max(len(sa),1),len(overlap)/max(len(sb),1),
                 float(bool(overlap)),len(set(wa)&set(wb))/max(len(set(wa)|set(wb)),1),
                 float(any(w in {"because","therefore","since","so","thus"} for w in wb))]
        left,right=positions[a],positions[b]
        context=[]
        for rep in reps:
            sim=rep[left]@rep[right].T
            context.extend((sim.amax(),sim.amax(0).mean(),sim.amax(1).mean(),sim.mean()))
        rel=ft.relations[right[:,None],left[None,:]]
        context.extend((rel[:,:,0].max(),rel[:,:,0].mean(),rel[:,:,1].max(),rel[:,:,1].mean()))
        mass=rez[right[:,None],left[None,:]]
        context.extend((mass.max(),mass.mean()))
        block=torch.cat([torch.as_tensor(scalars,dtype=torch.float32,device=ft.hidden.device),
                         torch.stack(context),reduced[a],reduced[b],SIZE*reduced[a]*reduced[b],
                         state[b]-state[a]])
        features.append(block)
    matrix=torch.stack(features).cpu().numpy()
    if matrix.shape!=(10,N_LEXICAL+18+8*SIZE) or not np.isfinite(matrix).all():
        raise ValueError("Nonfinite or misaligned sentence relations")
    return matrix


@torch.inference_mode()
def run(runtime,reader,packet,input_path,output):
    if runtime.candidate!="qwen" or runtime.revision!=packet["revision"]:
        raise ValueError("Expected frozen matching Qwen backbone")
    path=Path(input_path)
    cases=[json.loads(s) for s in path.read_text().splitlines()]
    out=Path(output)
    out.mkdir(parents=True)
    proj=projection(runtime.device,runtime.width)
    data=np.lib.format.open_memmap(out/"features.npy",mode="w+",dtype="float32",
                                  shape=(len(cases),10,N_LEXICAL+18+8*SIZE))
    ids=[]
    count=0
    def observe(m,v):
        nonlocal count
        count+=1
    started=perf_counter()
    with runtime.encoder.register_forward_pre_hook(observe):
        for i,row in enumerate(cases):
            if set(row)!={"id","text","spans"} or sha256(row["text"].encode()).hexdigest()!=row["id"]:
                raise ValueError("Unexpected grade-free event packet")
            data[i]=design(runtime,reader,row,proj)
            ids.append(row["id"])
            if count!=i+1:
                raise AssertionError("Counting encoder hook detected extra passes")
            if (i+1)%512==0:
                data.flush()
                print(f"Event traces {i+1}/{len(cases)}",flush=True)
    data.flush()
    # Replace text strictly after the third sentence and compare every fully read pair.
    row=cases[0]
    alternative=row["text"][:row["spans"][2][1]]+" It was raining. She looked outside."
    segments=[*row["spans"][:3]]
    for part in ("It was raining.","She looked outside."):
        a=segments[-1][1]+1
        segments.append([a,a+len(part)])
    check=design(runtime,reader,dict(text=alternative,spans=segments),proj)
    eligible=[i for i,(_,b) in enumerate(PAIRS) if b<3]
    delta=float(np.max(np.abs(check[eligible]-data[0,eligible])))
    if delta>2e-3:
        raise ArithmeticError(f"Unseen continuation affected earlier relations by {delta}")
    if len(set(ids))!=len(ids):
        raise ValueError("Duplicate model exposures")
    (out/"ids.json").write_text(json.dumps(ids)+"\n")
    receipt=dict(revision=runtime.revision,documents=len(ids),model_calls=count,validation_extra_forward=1,
                 prefix_max_abs_difference=delta,elapsed_seconds=perf_counter()-started,
                 input_sha256=sha256(path.read_bytes()).hexdigest(),
                 files={k:sha256((out/k).read_bytes()).hexdigest() for k in ("ids.json","features.npy")})
    (out/"receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
