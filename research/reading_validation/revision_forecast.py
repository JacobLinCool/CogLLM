"""Extract source-independent word and passage features for revision judgments."""

import hashlib
import json
from pathlib import Path
import re
from time import perf_counter

import numpy as np

from research.backbone_transfer.extract import forward


WORD = re.compile(r"[^\W_]+(?:[’'\-][^\W_]+)*")
DIM = 96
SEED = 611934
BATCH = 16


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def run(runtime, reader, checkpoint, input_path, output_dir):
    import torch

    if runtime.candidate!='qwen' or runtime.revision!=checkpoint['revision'] or runtime.width!=2048:
        raise ValueError('Expected pinned causal Qwen3.5-2B features')
    packet=Path(input_path)
    lexfile=packet.with_name('lexical.npz')
    if file_hash(lexfile)!='caef04fb8b90154e208e528fee39fc2ccba96cd34a6945fac836771427784254':
        raise ValueError('Unexpected lexical head')
    with np.load(lexfile,allow_pickle=False) as l:
        w=torch.from_numpy(l['w']).to(runtime.device)
        b=float(l['b'])
    rows=[json.loads(l) for l in packet.read_text().splitlines()]
    ids,seqs,spans,word_indexes=[],[],[],[]
    for r in rows:
        if set(r)!={'id','text'} or hashlib.sha256(r['text'].encode()).hexdigest()!=r['id']:
            raise ValueError('Revision extraction accepts only hashed, grade-free passages')
        t=r['text']
        z=runtime.tokenizer.encode(t,add_special_tokens=False)
        if not 0<len(z.ids)<=2048:
            raise ValueError('Unsupported passage length')
        stops={}
        for i,(left,right) in enumerate(z.offsets):
            if left<right: stops[right]=i
        words=[m.span() for m in WORD.finditer(t)]
        native=[stops.get(right,-1) for left,right in words]
        ids.append(r['id']);seqs.append(z.ids);spans.append(words);word_indexes.append(native)
    if len(set(ids))!=len(rows):
        raise ValueError('Duplicate passage forwards are prohibited')
    offsets=np.r_[0,np.cumsum([len(s) for s in spans])].astype(np.int64)
    imax=int(offsets[-1])
    words=np.asarray([ab for s in spans for ab in s],dtype=np.int32).reshape(-1,2)
    certified=np.asarray([j>=0 for ns in word_indexes for j in ns],dtype=bool)
    contextual=np.zeros((imax,DIM),dtype=np.float16)
    lexical=np.zeros(imax,dtype=np.float32)
    states=np.empty((len(ids),runtime.width),dtype=np.float32)
    projection=np.linalg.qr(np.random.default_rng(SEED).standard_normal((runtime.width,DIM)))[0].astype('float32')
    proj=torch.tensor(projection,device=runtime.device)
    order=sorted(range(len(ids)),key=lambda i:(len(seqs[i]),ids[i]))
    counter=0
    def count(_, __):
        nonlocal counter
        counter+=1
    start=perf_counter()
    with runtime.encoder.register_forward_pre_hook(count),torch.inference_mode():
        for first in range(0,len(order),BATCH):
            indices=order[first:first+BATCH]
            matrix=forward(runtime.encoder,[seqs[j] for j in indices],248044,torch)
            last=matrix[torch.arange(len(indices),device=runtime.device),[len(seqs[j])-1 for j in indices]]
            states[indices]=last.cpu().numpy()
            where=[(k,i,p) for k,j in enumerate(indices) for i,p in enumerate(word_indexes[j]) if p>=0]
            if not where: continue
            head=[k for k,i,p in where]
            pos=[p for k,i,p in where]
            native=matrix[head,pos].double()
            native=torch.nn.functional.normalize(native,dim=1)
            packed=np.array([offsets[indices[k]]+i for k,i,p in where])
            contextual[packed]=(native.float()@proj).half().cpu().numpy()
            lexical[packed]=(native@w+b).clamp(0,1).float().cpu().numpy()
    if counter!=(len(order)+BATCH-1)//BATCH or not all(np.isfinite(a).all() for a in (contextual,lexical,states)):
        raise ValueError('Invalid single-forward extraction')
    checks={}
    for j in [order[len(order)//4],order[len(order)//2],order[-1]]:
        seq=seqs[j]
        single=forward(runtime.encoder,[seq],248044,torch)[0]
        stop=max(1,len(seq)//2)
        prefix=forward(runtime.encoder,[seq[:stop]],248044,torch)[0,-1]
        def error(a,b):
            return float((a-b).norm()/b.norm().clamp_min(1e-12))
        checks[ids[j]]=[error(torch.tensor(states[j],device=runtime.device),single[-1]),error(prefix,single[stop-1])]
    if max(v for row in checks.values() for v in row)>1e-3:
        raise ArithmeticError('Batch or prefix equivalence failed')
    out=Path(output_dir)
    out.mkdir(exist_ok=True,parents=True)
    (out/'ids.json').write_text(json.dumps(ids)+'\n')
    np.save(out/'states.npy',states,allow_pickle=False)
    np.savez_compressed(out/'words.npz',range_offsets=offsets,word_spans=words,observed=certified,contextual=contextual,lexical=lexical)
    files={k:file_hash(out/k) for k in ('ids.json','states.npy','words.npz')}
    meta=dict(revision=runtime.revision,packet_sha256=file_hash(packet),lexical_sha256=file_hash(lexfile),
              extraction_code_sha256=file_hash(Path(__file__)),files_sha256=files,unique_passages=len(ids),
              words=imax,words_with_exact_token_end=int(certified.sum()),tokens=sum(map(len,seqs)),word_projection=DIM,
              projection_seed=SEED,inference_batches=counter,one_forward_for_each_passage=True,validation_forwards=6,
              checks=checks,elapsed_seconds=perf_counter()-start,device=torch.cuda.get_device_name())
    (out/'extraction.json').write_text(json.dumps(meta,indent=2)+'\n')
    return meta
