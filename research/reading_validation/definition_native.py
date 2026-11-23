"""Extract the original curve and replay the deployable terminology ledger on one causal encoding."""
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from research.backbone_transfer.extract import digest, read_json, stream_rows, write_json
from research.reading_validation.definition_inference import ConceptLedger
from research.token_readout.runner import trace_rows


def events(trace):
    return [[m['chars'], m['opening_token'], m['closing_token']] for m in trace['mentions']]


def verify_api(runtime, refs, features, row, ledger):
    reference=refs[row['id']]
    trace=ledger.from_features(row['text'],features,runtime)
    curve=[t['open_mentions'] for t in trace['tokens']]
    probs=[w['explanatory_evidence'] for w in trace['words']]
    max_error=float(np.max(np.abs(np.asarray(probs)-reference['evidence'])))
    exact=events(trace)==reference['mentions'] and curve==reference['curve']
    if not exact or max_error>.003:raise ValueError(f'Offline-to-serving replay differs: {exact}, {max_error}')
    stop=reference['stop_char'];prefix=ledger.analyze(row['text'][:stop],runtime)
    num=len(prefix['tokens'])
    a=[(t['chars'],t['opened_mention'],t['closed_mention'],t['open_mentions']) for t in prefix['tokens']]
    b=[(t['chars'],t['opened_mention'],t['closed_mention'],t['open_mentions']) for t in trace['tokens'][:num]]
    if a!=b:raise ValueError('A later word changed an earlier terminology ledger event')
    pa=np.asarray([w['explanatory_evidence'] for w in prefix['words']]);delta=float(np.max(np.abs(pa-np.asarray(probs[:len(pa)]))))
    if delta>.003:raise ValueError(f'Prefix evidence drifted by {delta}')
    return {'tokens':len(curve),'mentions':len(trace['mentions']),'closes':sum(m['closing_token'] is not None for m in trace['mentions']),
            'verified_prefix_tokens':num,'prefix_forward_count':prefix['encoder_forwards'],'replay_max_error':max_error,'prefix_max_error':delta}


def run(runtime,reader,checkpoint,input_path,output_dir):
    output=Path(output_dir);output.mkdir(parents=True)
    refs=read_json(Path('stock-serving.json'));ledger=ConceptLedger.load()
    scores={};found={};count=0;start=perf_counter()
    def on_forward(_module,_args):
        nonlocal count
        count+=1
    scale=checkpoint['scale'];rows=list(stream_rows(input_path))
    with runtime.encoder.register_forward_pre_hook(on_forward),torch.inference_mode():
        for i,r in enumerate(rows):
            before=count;f=runtime.encode(r['text']);s=reader(f)
            s=replace(s,opening=s.opening*scale['opening_gain'],resolution=s.resolution*scale['resolution_gain']+scale['resolution_bias'])
            trace=trace_rows(r['text'],f.offsets,s,checkpoint['priors'],reader.anchor_tokens)
            data=[]
            for t,v in enumerate(trace):
                raw=f.offsets[max(0,t+1-reader.anchor_tokens)][0]
                data.append([v['start'],v['end'],raw,v['pointer_start'] if v['pointer_start'] is not None else -1,
                             v['pointer_end'] if v['pointer_end'] is not None else -1,v['unresolved'],v['opened'],v['resolved'],v['transition']])
            scores[r['id']]=np.asarray(data,dtype='float32')
            if count-before!=1:raise ValueError('A fresh source made multiple forward calls')
            if r['id'] in refs:found[r['id']]=verify_api(runtime,refs,f,r,ledger)
            if (i+1)%200==0:print(f'Original curve: {i+1}/{len(rows)} complete passages',flush=True)
    if set(refs)!=set(found) or count!=len(rows)+len(found):raise ValueError('Incomplete replay/forecast collection')
    out=output/'original.npz';np.savez_compressed(out,**scores)
    write_json(output/'receipt.json',{'original_sha256':digest(out),'source':digest(input_path),'model_revision':runtime.revision,
        'reader_sha256':digest(Path('curve.pt')),'serving_ref_sha256':digest(Path('stock-serving.json')),
        'columns':['start','end','anchor_start','pointer_start','pointer_end','unresolved','opening','resolution','transition'],
        'full_article_calls':len(rows),'independent_prefix_calls':len(refs),'total_encoder_calls':count,'runtime_seconds':perf_counter()-start,
        'anchor_tokens':reader.anchor_tokens,'serving_replays':found})
