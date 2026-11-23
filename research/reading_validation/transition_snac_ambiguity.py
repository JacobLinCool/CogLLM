"""Bound alert coverage of every released span without inferring the selected occurrence."""
import json
import re

import numpy as np

from research.reading_validation.transition_external_score import grouped_distribution
from research.reading_validation.transition_score import dump, sha, with_interval
from research.reading_validation.transition_snac_fit import inputs, TARGETS, ARMS
from research.reading_validation.transition_snac_prepare import RESULTS, source


def run():
    docs,m=inputs();raw=source();path=RESULTS/'heldout-probabilities.npz';lock=json.loads((RESULTS/'prediction-lock.json').read_text())
    if sha(path)!=lock['forecast_sha256']:raise ValueError('Word forecasts changed after sealing')
    weights=json.loads((RESULTS/'readouts.json').read_text())['readouts'];pred=np.load(path,allow_pickle=False);out={}
    for target in TARGETS:
        groups=[];all_bounds=[];ambiguity=[]
        for rec in m['documents']:
            if rec['split']!='test':continue
            doc=rec['doc'];ws=np.array(docs[doc]['words']);scores=pred[target+'__'+doc]
            thresholds=np.array([weights[target][arm]['threshold'] for arm in ARMS]);alarm=scores>thresholds
            for seg in rec['segments']:
                index=seg['index'];block=raw[doc][str(index)];local=ws[:,4]==index;base=seg['start']
                text=docs[doc]['text'][base:seg['end']]
                for e in block['errors']:
                    if e['error_type']!=target:continue
                    needle=' '.join(e['span'].split());matches=list(re.finditer(re.escape(needle),text)) if needle else []
                    detections=[]
                    for hit in matches:
                        mask=local&(ws[:,0]<base+hit.end())&(ws[:,1]>base+hit.start())
                        if mask.any():detections.append(alarm[mask].any(axis=0))
                    if len(detections)!=len(matches) or not matches:
                        guaranteed=np.zeros(len(ARMS),dtype=bool);possible=np.ones(len(ARMS),dtype=bool)
                    else:
                        guaranteed=np.all(detections,axis=0);possible=np.any(detections,axis=0)
                    all_bounds.append(np.stack([guaranteed,possible]));groups.append(rec['group']);ambiguity.append(len(detections)!=1)
        all_bounds=np.stack(all_bounds);groups=np.array(groups);amb=np.array(ambiguity);table={}
        for subset,mask in [('all',np.ones(len(groups),dtype=bool)),('ambiguous',amb),('single_match',~amb)]:
            if not mask.any():continue
            group_ids,ids=np.unique(groups[mask],return_inverse=True);n=len(group_ids)
            if n==1:
                draws=np.ones((1,1),dtype=int)
            else:
                _,_,draws=grouped_distribution(groups[mask]);
            denom=np.bincount(ids,minlength=n);methods={}
            for i,arm in enumerate(ARMS):
                metrics={}
                for j,k in enumerate(['guaranteed','possible']):
                    hits=all_bounds[mask,j,i];counts=np.bincount(ids,weights=hits,minlength=n)
                    with np.errstate(invalid='ignore',divide='ignore'):
                        metric=with_interval(hits.mean(),draws@counts/(draws@denom))
                    if n==1:metric['family_bootstrap_95_interval']=None
                    metrics[k]={**metric,'hits':int(hits.sum())}
                methods[arm]=metrics
            table[subset]={'source_span_records':int(mask.sum()),'movies':n,'models':methods}
        out[target]=table
    dump(RESULTS/'span-ambiguity-bounds.json',{'analysis':'post-primary sensitivity at the frozen alert thresholds',
        'observation':'one released human-selected phrase record within one source segment',
        'guaranteed':'every matching occurrence overlaps at least one alerted word',
        'possible':'at least one matching occurrence overlaps at least one alerted word','results':out})
    return out


if __name__=='__main__':
    for target,sets in run().items():
        print(target,[(key,d['source_span_records'],d['models']['antecedents']) for key,d in sets.items()])
