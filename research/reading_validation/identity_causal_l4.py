"""One batched Qwen forward per causal passage, with native attention and prefix controls."""

from argparse import ArgumentParser
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from research.backbone_transfer.extract import digest,read_json,stream_rows,write_json
from research.reading_validation.identity_causal_features import dense_relative,encode_batch,original_scores,project_basis,sentence_tokens,summarize
from research.reading_validation.identity_run_l4 import run_phases


def validate(runtime,basis):
    units=['A recorded voltage fell. A signal was delayed.\nA loose connector reduced the supply voltage.',
           'A sensor was noisy.\nThe crew reported the symptom.']
    tokens=[runtime.tokenizer.encode(t,add_special_tokens=False) for t in units];seq=[q.ids for q in tokens]
    batch,mid,fin,mean,peak=encode_batch(runtime,seq,basis)
    prefix=seq[0][:len(seq[0])//2]; cut=encode_batch(runtime,[prefix],basis)
    error={};native=runtime.encode(units[0]);n=len(seq[0]);width=len(prefix)
    for name,aa,bb in [('truncated_projected',batch[0,:width],cut[0][0]),('native_middle',mid[0,:n],native.mid),('native_final',fin[0,:n],native.hidden)]:
        error[name]=float((aa-bb).norm()/bb.norm().clamp_min(1e-12))
    # Independently reconstruct attention through the original single-document collector.
    scale=torch.arange(1,n+1,device=mean.device)[:,None,None];original=(native.relations+np.log(2.)).expm1()/scale
    mask=torch.tril(torch.ones(n,n,device=mean.device,dtype=torch.bool))
    error['native_attention']=float((original-torch.stack((mean[0,:n,:n],peak[0,:n,:n]),dim=-1)).abs()[mask].max())
    if not all(np.isfinite(list(error.values()))) or max(error.values())>.001:
        raise ArithmeticError(f'Batched/causal feature disagreement: {error}')
    return error


@torch.inference_mode()
def extract(runtime,reader,checkpoint,inputs,output):
    output.mkdir(parents=True,exist_ok=True)
    meta=read_json(Path('causal-subsets.json')); dense=set(meta['dense']);baseline=set(meta['baseline'])
    rows=list(stream_rows(inputs)); ids=[r['id'] for r in rows]
    if not dense<=set(ids) or not baseline<=set(ids) or len(set(ids))!=len(ids):
        raise ValueError('Dense or comparison queues mismatch the stimulus bank')
    basis=project_basis(runtime.width,runtime.device)
    checks=validate(runtime,basis)
    enc=[runtime.tokenizer.encode(row['text'],add_special_tokens=False) for row in rows]
    geometry=[sentence_tokens(row['text'],ei.offsets) for row,ei in zip(rows,enc,strict=True)]
    if max(len(q.ids) for q in enc)>2048:raise ValueError('Overlong document')
    n_sent=np.array(list(map(len,geometry)),dtype=np.int64);n_sent_max=max(n_sent)
    starts=np.r_[0,np.cumsum(n_sent,dtype=np.int64)]; width=2*basis[0].shape[1];S=len(ids)
    bank=np.empty((int(starts[-1]),2,width),dtype=np.float16)
    relations=np.zeros((S,n_sent_max,n_sent_max,2),dtype=np.float16)
    old=np.zeros((S,n_sent_max,n_sent_max),dtype=np.float32)
    dense_arrays={};dense_attn={};before=perf_counter();count=0;expected=0
    def count_forward(_module,_args):
        nonlocal count
        count+=1
    order=sorted(range(S),key=lambda j:(len(enc[j].ids),ids[j])); batch_size=24
    with runtime.encoder.register_forward_pre_hook(count_forward):
        for i in range(0,S,batch_size):
            indices=order[i:i+batch_size];seq=[enc[j].ids for j in indices]
            states,mid,final,attn,peak=encode_batch(runtime,seq,basis);expected+=1
            for d,j in enumerate(indices):
                key=ids[j];n=len(seq[d]);sp=geometry[j];s=len(sp)
                state=states[d,:n];mean=attn[d,:n,:n]; top=peak[d,:n,:n]
                features,prior=summarize(state,mean,top,sp)
                bank[starts[j]:starts[j+1]]=features.cpu().numpy();relations[j,:s,:s]=prior.cpu().numpy()
                if key in baseline:
                    p=original_scores(reader,checkpoint,mid[d,:n],final[d,:n],mean,top,enc[j].offsets,sp)
                    for t,(a,b) in enumerate(sp):
                        old[j,t,:s]=-torch.log1p(-p[:,a:b].clamp_max(1-1e-7)).sum(1).cpu().numpy()
                if key in dense:
                    dense_arrays[key]=state.half().cpu().numpy()
                    if key in meta['historical']:
                        rel=torch.stack((mean,top),dim=-1).half().cpu().numpy()
                    else:rel=dense_relative(state,mean,top,sp).half().cpu().numpy()
                    dense_attn[key]=rel
            if i and i%(batch_size*100)==0:print(f'Causal passage bank {i}/{S}',flush=True)
    if count!=expected or not np.isfinite(bank).all() or not np.isfinite(relations).all():
        raise ValueError('Invalid features or forward count')
    np.savez_compressed(output/'sentence-states.npz',states=bank,relations=relations,old=old)
    np.savez_compressed(output/'token-states.npz',**dense_arrays);np.savez_compressed(output/'token-attention.npz',**dense_attn)
    write_json(output/'geometry.json', {'ids':ids,'sentence_tokens':geometry,'tokens':[len(x.ids) for x in enc],
                                         'sentence_offsets':starts.tolist(),'dense':sorted(dense),'baseline':sorted(baseline),
                                         'source':digest(inputs),'basis_seed':270653,'width':width,'validation':checks,
                                         'seconds':perf_counter()-before,'batched_forwards':count,'passages':S,
                                         'validation_forwards':3,'total_tokens':sum(len(x.ids) for x in enc)})


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__)
    for name in ('model','output','stimuli'):parser.add_argument('--'+name,type=Path,required=True)
    for name in ('candidate','protocol-sha256'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    from research.reading_validation.transition_matched_forecast import run as matched
    from research.reading_validation.external_forecast import run as prior
    from research.reading_validation.writer_verify import run as writer
    jobs=[('writer',writer,Path('prior-stimuli.jsonl')),('cohesion-matched',matched,Path('cohesion-matched-stimuli.jsonl')),
          ('prior',prior,Path('prior-stimuli.jsonl')),('causal',extract,args.stimuli)]
    run_phases(args.model,args.candidate,args.output,args.protocol_sha256,jobs)
