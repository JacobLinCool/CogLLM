"""Export the source-normalized, ambiguity-aware Qwen reader and an outcome-free serving replay."""
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from research.reading_validation.narrative_inference import CHECKPOINT,REVISION
from research.reading_validation.transition_partial_loss import OUT,PROTOCOL
from research.reading_validation.transition_score import dump,sha
from research.reading_validation.transition_snac_fit import inputs
from research.reading_validation.transition_snac_prepare import ROOT


def export():
    meta=json.loads((OUT/'scores.json').read_text());co=OUT/'readouts.npz';pred=OUT/'locked-forecasts.npz'
    if (sha(co)!=meta['readout_sha256'] or sha(pred)!=meta['forecasts_sha256'] or meta['protocol_sha256']!=sha(PROTOCOL)
            or meta.get('candidate','original_qwen')!='original_qwen'):
        raise ValueError('Serving requires the verified original Qwen features and source-only fit')
    w=np.load(co);vector=w['set_marginal']
    if len(vector)!=129 or len(w['mean'])!=128 or len(w['sd'])!=128:raise ValueError('Wrong projected reader dimensions')
    p={'schema':'cogllm-narrative-character-en-marginal-v1','backbone':'Qwen/Qwen3.5-2B','revision':REVISION,
       'reader_checkpoint_sha256':CHECKPOINT,'source_model_sha256':sha(co),'source_protocol_sha256':sha(PROTOCOL),
       'feature_archive_sha256':meta['feature_sha256'],'training_words':51309,'training_titles':44,
       'intended_observation':'A word in an English narrative summary overlaps a character reference selected by a reader as missing an introduction.',
       'threshold':meta['fits']['set_marginal']['cutoff'],'threshold_source':'95th percentile of guaranteed-negative words in 11 independent book titles',
       'weights':{'mean':w['mean'].tolist(),'std':w['sd'].tolist(),'weights':vector[1:].tolist(),'intercept':float(vector[0])}}
    bundle=OUT/'serving'/'character-word.json';bundle.parent.mkdir(exist_ok=True);dump(bundle,p)
    docs,material=inputs();preds=np.load(pred);eta=preds['set_marginal'];n=0
    packet=ROOT/'narrative-marginal-serving-stimuli.jsonl'
    with packet.open('w') as out:
        for m in material['documents']:
            if m['split']!='test':continue
            d=m['doc'];raw=docs[d];words=raw['words'];end=n+len(words);scores=np.exp(-np.logaddexp(0.,-eta[n:end]))
            obj={'text':raw['text'],'segments':[[s['start'],s['end']] for s in m['segments']],
                 'expected':[[a,b,float(v)] for (a,b,*_),v in zip(words,scores,strict=True)]}
            print(json.dumps(obj,separators=(',',':')),file=out);n=end
    if n!=len(eta) or n!=26417:raise ValueError('The complete external movie universe is required for serving')
    dump(OUT/'serving'/'export.json',{'weights_sha256':sha(bundle),'packet_sha256':sha(packet),'source_train_fit_sha256':sha(co),
         'cutoff':p['threshold'],'external_words':n,'above_threshold':int((np.exp(-np.logaddexp(0.,-eta))>p['threshold']).sum())})
    print(sha(bundle),sha(packet),p['threshold'],n)


if __name__=='__main__':export()
