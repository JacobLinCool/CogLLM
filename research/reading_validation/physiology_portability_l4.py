"""Archive label-free native word states and validated next-word probabilities."""
from argparse import ArgumentParser
import gc
from pathlib import Path
from time import perf_counter
import json
import traceback
import numpy as np
import torch
from research.backbone_transfer.extract import digest,stream_rows,write_json
from research.reading_validation.identity_run_l4 import run_phases
from research.reading_validation import state_transport_l4,transition_anchored_l4,transition_balanced_verify
from research.reading_validation.transition_architecture_l4 import native_losses,verify_logits
from research.token_readout.backbone import Backbone


@torch.inference_mode()
def study(model,candidate,stimuli,dest):
    runtime=Backbone.load(candidate,model,native_attention=False);native=verify_logits(runtime)
    docs=list(stream_rows(stimuli));dest.mkdir();count=0;geom={};arrays={};checks=[];began=perf_counter()
    def counted(*args):
        nonlocal count
        count+=1
    def capture(row,upto=None):
        n=len(row['words']) if upto is None else upto;b=row['words'][n-1][1];text=row['text'][:b]
        f=runtime.encode(text);bounds=row['native'][candidate][:n]
        if (upto is None and row['tokens'][candidate]!=len(f.token_ids)) or len(f.token_ids)!=bounds[-1][1]+1:raise ValueError('Pinned whole-word positions differ')
        for (a,e),(u,v) in zip(row['words'][:n],bounds):
            if f.offsets[v][1]!=e or any(not (f.offsets[i][0]<e and f.offsets[i][1]>a) for i in range(u,v+1)):raise ValueError('Real word no longer aligns to the native token')
        idx=torch.tensor([v for u,v in bounds],device=runtime.device);h=torch.cat((f.mid[idx],f.hidden[idx]),dim=1)
        states=h.float().cpu().numpy();neg=native_losses(runtime,f)
        sur=np.asarray([neg[u:v+1].sum() if u else 0. for u,v in bounds],np.float32)
        return states,sur
    with runtime.encoder.register_forward_pre_hook(counted):
        for i,r in enumerate(docs):
            a,s=capture(r);arrays[r['id']]=a.astype(np.float16)
            if not np.isfinite(arrays[r['id']]).all():raise ArithmeticError('Nonfinite archival hidden state')
            geom[r['id']]=s
            if i in (0,350,600):
                j=max(2,len(s)//2);short,ss=capture(r,j);rms=float(np.linalg.norm(a[:j]-short)/max(np.linalg.norm(a[:j]),1e-8));lp=float(abs(ss-s[:j]).max())
                if rms>.001 or lp>.04:raise ValueError('Unseen suffix changed a committed native score')
                checks.append(dict(words=j,relative_rms=rms,surprisal_abs=lp))
            if (i+1)%200==0:print(candidate,i+1,len(docs),flush=True)
    if count!=len(docs)+len(checks):raise ValueError('Required one production call per source document')
    p=dest/'native-states.npz';q=dest/'word-surprisal.npz';np.savez_compressed(p,**arrays);np.savez_compressed(q,**geom)
    write_json(dest/'receipt.json',dict(candidate=candidate,revision=runtime.revision,device=torch.cuda.get_device_name(),stimuli=digest(stimuli),files={p.name:digest(p),q.name:digest(q)},native_projection=native,texts=len(docs),production_forwards=len(docs),checks=checks,stored_words=sum(map(len,geom.values())),seconds=perf_counter()-began,dimensions=2*runtime.width))


def run(model,candidate,stimuli,output,protocol_sha256):
    if torch.cuda.device_count()!=1 or torch.cuda.get_device_name()!='NVIDIA L4':raise ValueError('Single L4 required')
    if candidate=='qwen':
        jobs=[('balanced-narrative',transition_balanced_verify.run,Path('narrative-parity.jsonl')),('danish-coherence',state_transport_l4.run,Path('danish.jsonl')),('italian-order-control',state_transport_l4.run,Path('italian.jsonl')),('italian-order-source',state_transport_l4.run,Path('italian-source.jsonl'))];run_phases(model,candidate,output,protocol_sha256,jobs)
        (output/'causality.json').rename(output/'danish-status.json');(output/'seal.json').rename(output/'danish-seal.json')
        status=json.loads((output/'danish-status.json').read_text())['studies'];gc.collect();torch.cuda.empty_cache()
        try:
            transition_anchored_l4.run(model,Path('source-pointer'),Path('source-meaning'),Path('ambicoref.jsonl'),output/'ambicoref-forecast','73da8cb603efacafd8b7516a87def1e33219d6398c9ebf2a47566c4b7aec9d58','a7094264d505f1465bab8b84cbd762dc0c5d7ddea1067cfe95f7ebc357fcee5f');status['ambicoref']={'completed':True}
        except Exception as exc:
            traceback.print_exc();status['ambicoref']={'completed':False,'exception':type(exc).__name__,'message':str(exc)}
        gc.collect();torch.cuda.empty_cache()
    else:output.mkdir();status={}
    try:
        study(model,candidate,stimuli,output/'physiology-forecast');status['physiology']={'completed':True}
    except Exception as exc:
        traceback.print_exc();status['physiology']={'completed':False,'exception':type(exc).__name__,'message':str(exc)}
    write_json(output/'causality.json',dict(candidate=candidate,studies=status))
    write_json(output/'seal.json',dict(protocol_sha256=protocol_sha256,all_completed=all(v['completed'] for v in status.values()),files_sha256={str(p.relative_to(output)):digest(p) for p in sorted(output.rglob('*')) if p.is_file()}))


if __name__=='__main__':
    p=ArgumentParser(description=__doc__)
    for name in ('candidate','protocol-sha256'):p.add_argument('--'+name,required=True)
    for name in ('model','stimuli','output'):p.add_argument('--'+name,type=Path,required=True)
    run(**vars(p.parse_args()))
