"""Compute authentic next-word probabilities once for each public nine-word sentence context."""
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from research.backbone_transfer.extract import digest,stream_rows,write_json,forward
from research.reading_validation.population_inference import DEFAULT,PopulationWordInference
from research.reading_validation.transition_architecture_l4 import native_logits


@torch.inference_mode()
def run(runtime,reader,packet,source,out):
    start=perf_counter();out=Path(out);out.mkdir()
    population=PopulationWordInference.load()
    if runtime.revision!=population.revision:raise ValueError("Wrong native model")
    words=torch.as_tensor(population.weights["unique"],device=runtime.device)
    others=torch.ones(len(runtime.encoder.embed_tokens.weight),dtype=torch.bool,device=runtime.device);others[words]=False
    rows=list(stream_rows(source))
    if len(rows)!=3085 or any(set(r)!={'id','text'} for r in rows):raise ValueError("Unexpected label-free reading material")
    seq=[runtime.tokenizer.encode(r["text"],add_special_tokens=False).ids for r in rows]
    pad=runtime.tokenizer.token_to_id("<|endoftext|>")
    if pad is None:raise ValueError("Canonical padding token unavailable")
    def probs(last):
        logits=native_logits(runtime,last)
        return torch.cat([logits[:,words],torch.logsumexp(logits[:,others],1)[:,None]],1)-torch.logsumexp(logits,1)[:,None]
    ordered=sorted(range(len(rows)),key=lambda i:len(seq[i]));arr=np.empty((len(seq),len(words)+1),dtype=np.float16)
    calls=0;backups={}
    def count(*_):
        nonlocal calls
        calls+=1
    with runtime.encoder.register_forward_pre_hook(count):
        for i in range(0,len(seq),32):
            idx=ordered[i:i+32];ss=[seq[k] for k in idx]
            y=forward(runtime.encoder,ss,pad,torch)
            p=probs(y[range(len(ss)),[len(q)-1 for q in ss]])
            if abs(torch.logsumexp(p,dim=1)).max().item()>0.0001:raise ValueError("Unnormalized reading distribution")
            for j,k in enumerate(idx):
                if k in (17,137,2741):backups[k]=(p[j].clone(),y[j,len(ss[j])//2].clone())
            arr[idx]=p.half().cpu().numpy()
        B=calls;checks=[]
        for k,(final,prefix) in backups.items():
            ii=seq[k];a=forward(runtime.encoder,[ii],pad,torch)[0];q=probs(a[-1:])[0]
            b=forward(runtime.encoder,[ii[:len(ii)//2+1]],pad,torch)[0,-1]
            err=max(float(abs(q-final).max()),float(abs(prefix-b).max()))
            if err>0.002:raise ValueError("Future or padding changed authentic input")
            checks.append(err)
    f=out/'logs.npz';np.savez_compressed(f,ids=np.array([r['id'] for r in rows]),native=arr)
    write_json(out/'receipt.json',dict(output=digest(f),input=digest(source),texts=len(rows),batch_forwards=B,
               study_text_exposures=len(rows),technical_forwards=calls-B,suffix_errors=checks,seconds=perf_counter()-start))
