"""Run the released word-response readout on nine source texts and on two truncation controls."""
from pathlib import Path
from time import perf_counter
import json
import numpy as np
import torch
from research.backbone_transfer.extract import digest,stream_rows,write_json
from research.reading_validation.word_response_inference import WordResponseInference


@torch.inference_mode()
def run(runtime,reader,checkpoint,input_path,output_dir):
    dest=Path(output_dir);dest.mkdir();start=perf_counter();rows=list(stream_rows(input_path));head=WordResponseInference.load();calls=0;points=0;out={};drifts=[]
    if head.meta['verification_packet']!=digest(input_path) or len(rows)!=9:raise ValueError('Expected the preregistered original English sentence replay')
    def counter(*args):
        nonlocal calls
        calls+=1
    with runtime.encoder.register_forward_pre_hook(counter):
        for i,r in enumerate(rows):
            text=r['text'];data=head.from_features(text,runtime,reader,checkpoint,runtime.encode(text));words=data['words'];points+=len(words);out[r['id']]=words
            if not data['available'] or len(words)!=len(text.split())-1:raise ValueError('Printed word did not yield exactly one observable decision')
            if i in (1,5):
                j=len(words)//2;prefix=text[:words[j]['end']];truncated=head.from_features(prefix,runtime,reader,checkpoint,runtime.encode(prefix))['words'];a=words[:j+1]
                if [(v['start'],v['end'],v['commit_token']) for v in truncated]!=[(v['start'],v['end'],v['commit_token']) for v in a]:raise ValueError('Suffix controls changed earlier word identity')
                drift=max(abs(v[k]-b[k]) for v,b in zip(a,truncated) for k in ('expected_N400_negativity','cloze_target_probability','raw_fitted_rating'));drifts.append(drift)
                if drift>.004:raise ArithmeticError('Unseen suffix changed released reading population traces')
    if calls!=11:raise ValueError('Expected one backbone call per text and two additional technical controls')
    path=dest/'responses.json';write_json(path,out);write_json(dest/'receipt.json',dict(stimuli=digest(input_path),weights=head.meta['weights_sha256'],output=digest(path),texts=len(rows),ordinary_forwards=len(rows),technical_calls=calls-len(rows),total_committed_human_units=points,max_suffix_error=max(drifts),gpu=torch.cuda.get_device_name(),seconds=perf_counter()-start))
