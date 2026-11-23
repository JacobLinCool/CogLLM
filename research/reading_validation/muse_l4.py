"""One native causal German prefix per original stimulus, without any reader outcome inputs."""
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from research.backbone_transfer.extract import digest,read_json,write_json,stream_rows,forward
from research.reading_validation.transition_architecture_l4 import native_logits


@torch.inference_mode()
def run(runtime,reader,packet,source,out):
    start=perf_counter();out=Path(out);out.mkdir()
    rows=list(stream_rows(source));dic=Path('german-word-inventory.json');lex=read_json(dic)
    if len(rows)!=619 or len(lex)!=4096 or any(set(v)!={'id','text'} for v in rows):
        raise ValueError('Wrong German stimulus or independent lexicon')
    seq=[runtime.tokenizer.encode(r['text'],add_special_tokens=False).ids for r in rows]
    pad=runtime.tokenizer.token_to_id('<|endoftext|>')
    tokenids=[t for e in lex for t in e['tokens']]
    if len(tokenids)!=len(set(tokenids)):raise ValueError('Duplicate German spelling buckets')
    refs=torch.tensor(tokenids,device=runtime.device)
    route=torch.tensor([i for i,e in enumerate(lex) for _ in e['tokens']],device=runtime.device)
    others=torch.ones(len(runtime.encoder.embed_tokens.weight),device=runtime.device,dtype=torch.bool);others[refs]=False
    def probabilities(last):
        logits=native_logits(runtime,last)
        denom=torch.logsumexp(logits,1,keepdim=True)
        values=torch.exp(logits[:,refs]-denom)
        named=torch.zeros((len(last),len(lex)),device=runtime.device).scatter_add_(1,route.expand(len(last),-1),values)
        return torch.cat((torch.log(named),torch.logsumexp(logits[:,others],1,keepdim=True)-denom),1)
    dest=np.empty((len(rows),len(lex)+1),np.float32)
    ids=sorted(range(len(seq)),key=lambda j:len(seq[j]));checks=0
    for batch in range(0,len(ids),32):
        given=ids[batch:batch+32];token=[seq[k] for k in given];h=forward(runtime.encoder,token,pad,torch)
        final=torch.stack([h[j,len(z)-1] for j,z in enumerate(token)])
        dest[given]=probabilities(final).cpu().numpy()
    err=[]
    for j in [ids[0],ids[len(rows)//2],ids[-1]]:
        h=forward(runtime.encoder,[seq[j]],pad,torch)
        obj=probabilities(h[0,-1:]).cpu().numpy()
        err.append(float(np.max(abs(obj-dest[j]))));checks+=1
    if max(err)>.002 or not np.isfinite(dest).all() or abs(np.exp(dest.astype(float)).sum(1)-1).max()>2e-5:
        raise ValueError('German probabilities, causal prefix or padding certification failed')
    f=out/'native.npz';np.savez_compressed(f,ids=np.array([r['id'] for r in rows]),logp=dest)
    write_json(out/'receipt.json',dict(stimuli=digest(source),lexicon=digest(dic),file=digest(f),texts=len(rows),cohort_fields=0,
            named_probability_classes=len(lex),native_grouped_tokens=len(refs),study_batches=(len(seq)+31)//32,
            technical_forwards=checks,max_padding_error=max(err),revision=runtime.revision,seconds=perf_counter()-start))
