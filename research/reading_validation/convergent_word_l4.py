"""Return single-forward article-only native states and word-local integration signals."""
from dataclasses import replace
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from research.backbone_transfer.extract import digest,stream_rows,write_json
from research.reading_validation.transition_external_forecast import token_surprisals
from research.token_readout.runner import trace_rows


@torch.inference_mode()
def run(runtime,reader,packet,input_path,output_dir):
    source=Path(input_path);dest=Path(output_dir);dest.mkdir();start=perf_counter();rows=list(stream_rows(source));counter=0;data={}
    def count(*args):
        nonlocal counter
        counter+=1
    def emit(row):
        f=runtime.encode(row['text']);A,B=reader.project(f);p=reader.score_projected(A,B,f.relations);scale=packet['scale']
        p=replace(p,opening=p.opening*scale['opening_gain'],resolution=p.resolution*scale['resolution_gain']+scale['resolution_bias'])
        traces=trace_rows(row['text'],f.offsets,p,packet['priors'],reader.anchor_tokens);surp=token_surprisals(runtime,f)
        A=A.float().cpu().numpy();B=B.float().cpu().numpy();states=np.c_[A,B]
        feat=[];sur=[];previous=np.zeros(states.shape[1],dtype='float32')
        for a,b,lo,hi in row['words']:
            if f.offsets[hi][1]!=b or not all(f.offsets[t][0]<b and f.offsets[t][1]>a for t in range(lo,hi+1)):raise ValueError('Printed word coordinate changed in forward')
            state=states[hi];delta=state-previous;word=row['text'][a:b];stock=[traces[hi][k] for k in ('unresolved','opened','resolved','transition')]
            feat.append(np.r_[np.log1p(len(word)),np.log1p(hi-lo+1),np.log1p(len(feat)),float(word[:1].isupper()),float(word[-1:] in ',.;:!?'),np.sum(surp[lo:hi+1]) if lo else 0.,stock,np.linalg.norm(delta),state,delta]);previous=state
        return np.asarray(feat,'float32')
    checks=[]
    with runtime.encoder.register_forward_pre_hook(count):
        for i,r in enumerate(rows):
            key=r['id'];q=emit(r);data[key]=q
            if i in (15,310):
                n=max(len(r['words'])//2,1);short={'text':r['text'][:r['words'][n-1][1]],'words':r['words'][:n]};v=emit(short);d=np.max(abs(v-q[:n]),axis=0)
                if max(d[6:])>.008 or d[5]>.04:raise ArithmeticError('A later original word changed a committed reading feature')
                checks.append({'words':n,'context_difference':float(max(d[6:])),'surprisal_difference':float(d[5])})
    if counter!=len(rows)+len(checks):raise ValueError('Extra encoder calls during article analysis')
    np.savez_compressed(dest/'word-states.npz',**data);write_json(dest/'receipt.json',dict(input=digest(source),word_states=digest(dest/'word-states.npz'),texts=len(rows),words=sum(map(len,data.values())),dimensions=q.shape[1],production_forwards=len(rows),checks=checks,seconds=perf_counter()-start,revision=runtime.revision,device=torch.cuda.get_device_name()))

