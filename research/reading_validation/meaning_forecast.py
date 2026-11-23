"""Extract original causal reading states and lexical expectations for Dutch judgment sentences."""
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from research.backbone_transfer.extract import digest, read_json, stream_rows, write_json
from research.reading_validation.definition_forecast import projection


@torch.inference_mode()
def run(runtime,reader,checkpoint,input_path,output_dir):
    source=Path(input_path);output=Path(output_dir);output.mkdir()
    vocab=source.with_name('vocabulary.json');codes=read_json(vocab)['onsets']
    table=runtime.encoder.embed_tokens.weight
    if not runtime.encoder.config.tie_word_embeddings or codes!=sorted(set(codes)):
        raise ValueError('Expected a tied native language head and unique registered onsets')
    onset_weight=table[torch.as_tensor(codes,device=runtime.device)]
    basis=projection(runtime);rows=list(stream_rows(source))
    items={};onsets={};surprisal={};forwards=0;tokens=0;started=perf_counter()
    def count(_module,_args):
        nonlocal forwards
        forwards+=1
    with runtime.encoder.register_forward_pre_hook(count):
        for i,row in enumerate(rows,1):
            f=runtime.encode(row['text']);ids=torch.as_tensor(f.token_ids,device=runtime.device)
            states=torch.cat([F.layer_norm(x,x.shape[-1:])@p for x,p in zip((f.mid,f.hidden),basis,strict=True)],dim=1)
            logprobs=[]
            for j in range(0,len(ids)-1,12):
                stop=min(len(ids)-1,j+12)
                logits=F.linear(f.hidden[j:stop],table).float()
                obs=logits.gather(1,ids[j+1:stop+1,None]).flatten()
                logprobs.append(logits.logsumexp(-1)-obs)
            nll=torch.cat([torch.zeros(1,device=runtime.device),*logprobs])
            local=F.linear(f.hidden,onset_weight).float()
            if not (states.isfinite().all() and nll.isfinite().all() and local.isfinite().all()):
                raise ValueError('Unusable native token values')
            key=row['id'];items[key]=states.half().cpu().numpy();onsets[key]=local.cpu().numpy()
            surprisal[key]=nll.cpu().numpy();tokens+=len(ids)
            if forwards!=i:raise ValueError('Each original article requires a single backbone call')
    archives={}
    for name,data in [('states',items),('onsets',onsets),('surprisal',surprisal)]:
        p=output/(name+'.npz');np.savez_compressed(p,**data);archives[name]=digest(p)
    write_json(output/'receipt.json',{'stimuli':digest(source),'onset_vocabulary':digest(vocab),'forwards':forwards,
        'tokens':tokens,'native_final_and_middle_width':80,'sha256':archives,'revision':runtime.revision,'seconds':perf_counter()-started})
