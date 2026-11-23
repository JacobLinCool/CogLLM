"""Fit on book data, calibrate on disjoint books, and seal all movie forecasts."""
from argparse import ArgumentParser
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from research.reading_validation.transition_external_fit import binomial, predict
from research.reading_validation.transition_snac_prepare import RESULTS, ROOT, source
from research.reading_validation.transition_snac_outcomes import outcomes
from research.reading_validation.transition_score import dump, sha

PROTOCOL='d517e5dc6cae17076afeb73663873af029fbf8aba30bcce57b022aece1debaec'
ARMS=('surface','surprisal','local','antecedents')
TARGETS=('CharE','RefE')


def inputs():
    meta=json.loads((RESULTS/'stimuli.json').read_text());text=ROOT/'snac-stimuli.jsonl'
    if sha(text)!=meta['packet_sha256'] or meta['protocol_sha256']!=PROTOCOL:raise ValueError('SNaC source seal changed')
    docs={r['id']:r for r in [json.loads(l) for l in text.read_text().splitlines()]}
    if docs.keys()!={s['doc'] for s in meta['documents']}:raise ValueError('SNaC documents do not match')
    return docs,meta


def fit(directory):
    docs,meta=inputs();material=meta['documents'];receipt=json.loads((directory/'receipt.json').read_text());archive=directory/'word-features.npz'
    if (receipt['input_sha256']!=meta['packet_sha256'] or len(docs)!=receipt['forwards'] or
        receipt['feature_sha256']!=sha(archive) or receipt['documents']!=len(docs)):
        raise ValueError('SNaC source forecasts do not match the L4 receipt')
    n_surf=len(receipt['surface_features']); local_end=n_surf+1+receipt['local_width'];width=local_end+len(receipt['relation_features'])
    columns=dict(surface=n_surf,surprisal=n_surf+1,local=local_end,antecedents=width)
    files=np.load(archive,allow_pickle=False);selected={};raw=source();records={};predictions={};audits={}
    for m in material:
        d=m['doc'];v=files[d]
        if v.shape!=(m['words'],width) or not np.isfinite(v).all():raise ValueError('Malformed word features')
        selected[d]=v
    for target in TARGETS:
        train_x=[];train_y=[];dev_x=[];dev_y=[];audits[target]={};records[target]={}
        for split,x,y in [('train',train_x,train_y),('dev',dev_x,dev_y)]:
            summary=Counter()
            for m in material:
                if m['split']!=split:continue
                d=m['doc'];valid,votes,audit=outcomes(docs[d],m,raw[d],target);summary.update(audit)
                x.append(selected[d][valid]);y.append((votes[valid]>0).astype(int))
            audits[target][split]=dict(summary)
        x=np.concatenate(train_x).astype(np.float64); y=np.concatenate(train_y)
        dx=np.concatenate(dev_x).astype(np.float64);dy=np.concatenate(dev_y)
        for arm,stop in columns.items():
            model=binomial(x[:,:stop],y,np.ones(len(y),dtype=int));prob=predict(dx[:,:stop],model)
            cutoff=float(np.quantile(prob[dy==0],.95,method='higher'))
            records[target][arm]={'weights':model,'threshold':cutoff}
        for m in material:
            if m['split']!='test':continue
            d=m['doc'];predictions[target+'__'+d]=np.column_stack([predict(selected[d][:,:stop],records[target][arm]['weights']) for arm,stop in columns.items()])
    model_path=RESULTS/'readouts.json';forecast=RESULTS/'heldout-probabilities.npz'
    dump(model_path,{'readouts':records,'columns':columns,'book_alignment':audits});np.savez_compressed(forecast,**predictions)
    dump(RESULTS/'prediction-lock.json',{'protocol_sha256':PROTOCOL,'features_receipt_sha256':sha(directory/'receipt.json'),
        'model_sha256':sha(model_path),'forecast_sha256':sha(forecast),'movie_documents':sum(m['split']=='test' for m in material),
        'training_observations':'word-level human marked-span membership, CharE and RefE separately'})
    print({t:{s:audits[t][s]['eligible_words'] for s in ('train','dev')} for t in TARGETS})


if __name__=='__main__':
    parser=ArgumentParser();parser.add_argument('--features',required=True,type=Path);fit(parser.parse_args().features)
