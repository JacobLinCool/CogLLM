"""Evaluate presealed correct-buzz trajectories at their actual word or clue resolution."""

from bisect import bisect_left
from collections import defaultdict
from pathlib import Path
import numpy as np

from research.backbone_transfer.extract import digest,read_json,stream_rows,write_json
from research.reading_validation.identity_causal_score import confidence
from research.reading_validation.recognition_fit import MODELS
from research.reading_validation.recognition_prepare import OUT,ROOT,EXTERNAL,key


def event_loss(log_survival,left,right):
    decay=log_survival[right]-log_survival[left]
    return float(-log_survival[left]-np.log(-np.expm1(decay)))


def stats(logits,row,checkpoints):
    h=np.logaddexp(0.,logits.astype('float64'));logsurv=np.r_[0.,-np.cumsum(h)]
    n=len(logits);stops=np.asarray(checkpoints,dtype=int)
    if not len(stops) or np.any(np.diff(stops)<=0) or stops[-1]!=n:raise ValueError('Invalid response-time grid')
    observed=np.zeros(len(stops));nll=0.
    for a,b,num in row['intervals']:
        scale=num/row['n'];nll+=scale*event_loss(logsurv,a,b);observed[stops>=b]+=scale
    model=-np.expm1(logsurv[stops]);ib=float(np.mean(observed*(1-model)**2+(1-observed)*model**2))
    med=int(np.searchsorted(model,.5));truth=int(np.searchsorted(observed,.5));mederr=abs(med-truth)
    return {'nll':nll,'integrated_brier':ib,'median_absolute_error_in_observed_units':mederr,'terminal_probability':float(model[-1]),
            'calibration_rmse':float(np.sqrt(np.mean((model-observed)**2)))}


def summary(rows):
    clusters=defaultdict(list)
    for r in rows:clusters[r['group']].append(r)
    if not rows:return {'texts':0,'families':0,'events':0}
    ms=list(rows[0]['scores']['semantic']);means={g:{m:{j:float(np.mean([v['scores'][m][j] for v in q])) for j in ms} for m in MODELS} for g,q in clusters.items()}
    scores={a:{m:confidence([z[a][m] for z in means.values()]) for m in ms} for a in MODELS}
    gains={a:{m:confidence([z[a][m]-z['semantic'][m] for z in means.values()]) for m in ('nll','integrated_brier','median_absolute_error_in_observed_units')} for a in MODELS if a!='semantic'}
    return {'texts':len(rows),'families':len(means),'events':sum(r['n'] for r in rows),'models':scores,'reduction_semantic_vs_comparator':gains,
            'nll_superiority':all(gains[m]['nll']['paired_item_bootstrap_95'][0]>0 for m in ('lexical','position'))}


def main():
    seal=read_json(OUT/'input-seal.json');forecast=read_json(OUT/'forecast-seal.json');p=ROOT/'forecasts.npz'
    if digest(p)!=forecast['forecasts'] or digest(OUT/'weights.json')!=forecast['model_sha256']:raise ValueError('Forecast or model differs from pre-score lock')
    docs={r['id']:r for r in stream_rows(ROOT/'stimuli.jsonl')};qanta={key(r['question']):r for r in stream_rows(EXTERNAL/'tossup_questions.jsonl')};report={};details={}
    train_lens=[len(docs[r['id']]['word_char_ends']) for r in stream_rows(ROOT/'train.jsonl')];median=float(np.median(train_lens))
    with np.load(p,allow_pickle=False) as preds:
        for split in ('test','external'):
            path=ROOT/(split+'.jsonl')
            if digest(path)!=seal['outcome_sha256'][split]:raise ValueError('Held-out responses changed')
            items=[]
            for r in stream_rows(path):
                k=r['id'];v=preds[k];n=v.shape[1];ends=docs[k]['word_char_ends'];stop=list(range(1,n+1))
                if split=='external':stop=[bisect_left(ends,b)+1 for _,b in qanta[k]['clue_spans']]
                if split=='external' and any(x not in stop for _,x,_ in r['intervals']):raise ValueError('Human response maps to unknown QANTA clue')
                vals={m:stats(v[i],r,stop) for i,m in enumerate(MODELS)}
                items.append({'id':k,'group':r['group'],'n':r['n'],'subject':r['subject'],'length':n,'scores':vals})
            report[split]={'all':summary(items),'science':summary([r for r in items if 'science'==r['subject'].casefold()]),'shorter':summary([r for r in items if r['length']<=median]),'longer':summary([r for r in items if r['length']>median]),
                           'training_median_words':median,'observation_resolution':'clue' if split=='external' else 'numbered word'};details[split]=items
            print(split,report[split]['all']['events'],report[split]['all']['reduction_semantic_vs_comparator']['lexical']['nll'],flush=True)
    write_json(OUT/'scores.json',report);write_json(OUT/'item-scores.json',details)


if __name__=='__main__':main()
