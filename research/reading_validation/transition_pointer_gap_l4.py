"""Apply the frozen reading-game pointer to untouched Wikipedia test passages."""
from argparse import ArgumentParser
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np

from research.backbone_transfer.extract import digest,write_json
from research.reading_validation.transition_pointer_l4 import ARMS,PROTOCOL as TRAIN_PROTOCOL,extraction


def run(model,stimuli,source_model,output,protocol_sha256,input_sha256):
    if digest(stimuli)!=input_sha256:raise ValueError('External article packet changed')
    trained=json.loads((source_model/'receipt.json').read_text());fp=source_model/'source-weights.npz'
    if (trained['protocol_sha256']!=TRAIN_PROTOCOL or trained['input_sha256']!='2dee71b79d698976c0ab2bb6fa100f3fe085f45fa47f08e01aac6770711d94b0'
            or digest(fp)!=trained['weights_sha256'] or trained['arms']!=list(ARMS)):
        raise ValueError('External pointer weights did not come from the registered prior reader experiment')
    rows=list(map(json.loads,stimuli.read_text().splitlines()))
    if any(r['fold']!='test' or 'clicks' in r or any(k in r for k in ('A','B','target','article')) for r in rows):
        raise ValueError('External model cannot see an annotated candidate')
    output.mkdir(exist_ok=True,parents=True);start=perf_counter()
    with TemporaryDirectory(dir='/content') as tmp:
        disk,items,meta=extraction(model,rows,Path(tmp));X=np.asarray(disk);bundle=np.load(fp);all_logits={}
        for arm in ARMS:
            mu=bundle[arm+'__mean'];sd=bundle[arm+'__sd'];b=bundle[arm+'__beta']
            if len(mu)!=len(b) or len(sd)!=len(b) or not (sd>0).all():raise ValueError('Source model shape differs')
            # Score by article/query blocks to keep additional per-model memory bounded.
            for q in items:
                a,t=q['range'];idx=q['doc']+'/'+str(q['q']);z=(X[a:t,:len(b)].astype(float)-mu)/sd
                if arm==ARMS[0]:all_logits[idx]=[]
                values=z@b
                if not np.isfinite(values).all():raise ValueError('Nonfinite transferred word pointer')
                all_logits[idx].append(values.astype(np.float32))
        pred=output/'candidate-word-logits.npz';np.savez_compressed(pred,**{k:np.stack(v) for k,v in all_logits.items()})
        write_json(output/'receipt.json',{'protocol_sha256':protocol_sha256,'input_sha256':input_sha256,'source_protocol_sha256':TRAIN_PROTOCOL,
            'trained_weights_sha256':digest(fp),'predictions_sha256':digest(pred),'features':meta,'seconds':perf_counter()-start,
            'arms':ARMS,'masked_candidate_locations':True,'external_weight_updates':0})


if __name__=='__main__':
    p=ArgumentParser(description=__doc__)
    for k in ('model','stimuli','source-model','output'):p.add_argument('--'+k,type=Path,required=True)
    for k in ('protocol-sha256','input-sha256'):p.add_argument('--'+k,required=True)
    run(**vars(p.parse_args()))
