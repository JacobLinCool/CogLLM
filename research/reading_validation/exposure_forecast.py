"""Produce all meaning-onset trajectories from the single causal pass over each original Dutch stimulus."""
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F
from research.backbone_transfer.extract import digest,read_json,stream_rows,write_json


def run(runtime,reader,checkpoint,input_path,output_dir):
    source=Path(input_path);out=Path(output_dir);out.mkdir(parents=True)
    vocab=source.with_name('vocabulary.json');codes=read_json(vocab)['onsets']
    if codes!=sorted(set(codes)):raise ValueError('Meaning onsets must be unique')
    table=runtime.encoder.embed_tokens.weight
    if not runtime.encoder.config.tie_word_embeddings:raise ValueError('Native LM head is untied')
    projection=table[torch.as_tensor(codes,device=runtime.device)];start=perf_counter();values={};count=0
    def hook(_m,_a):
        nonlocal count
        count+=1
    with runtime.encoder.register_forward_pre_hook(hook),torch.inference_mode():
        for r in stream_rows(source):
            f=runtime.encode(r['text'])
            y=F.linear(f.hidden,projection).float()
            if not torch.isfinite(y).all():raise ValueError('Nonfinite native logits')
            values[r['id']]=y.cpu().numpy()
            if count!=len(values):raise ValueError('Each new article must use one encoder call')
    archive=out/'onsets.npz';np.savez_compressed(archive,**values)
    write_json(out/'receipt.json',{'input':digest(source),'vocabulary':digest(vocab),'archive':digest(archive),'forwards':count,
        'tokens':sum(len(v) for v in values.values()),'hypotheses':len(codes),'revision':runtime.revision,'seconds':perf_counter()-start})
