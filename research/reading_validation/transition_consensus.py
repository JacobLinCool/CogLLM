"""Measure within-segment agreement ordering among words already marked by a reader."""
from collections import Counter,defaultdict
import json
from pathlib import Path

import numpy as np

from research.reading_validation.transition_external_score import grouped_distribution
from research.reading_validation.transition_score import dump,sha,with_interval
from research.reading_validation.transition_snac_fit import inputs
from research.reading_validation.transition_snac_outcomes import outcomes
from research.reading_validation.transition_snac_prepare import RESULTS,source

PROTOCOL=Path('research/reading_validation/transition-consensus-protocol.md')


def movie_mean(values):
    x=np.array(list(values.values()))
    point=x.mean(axis=0)
    if len(x)<2:
        return {'estimates':point.tolist(),'cluster_intervals':None}
    _,_,draws=grouped_distribution(np.array(sorted(values)))
    # Multiplicities are exchangeable, but keep exact identities aligned in every comparison.
    x=np.array([values[k] for k in sorted(values)])
    sim=(draws@x)/len(x)
    return {'qwen':with_interval(point[0],sim[:,0]),'ner':with_interval(point[1],sim[:,1]),
            'qwen_minus_ner':with_interval(point[0]-point[1],sim[:,0]-sim[:,1])}


def audit():
    docs,folds=inputs();raw=source()
    orig=RESULTS/'heldout-probabilities.npz';npath=RESULTS/'entity-control-probabilities.npz'
    if sha(orig)!=json.loads((RESULTS/'prediction-lock.json').read_text())['forecast_sha256'] or sha(npath)!=json.loads((RESULTS/'entity-control-lock.json').read_text())['forecast_sha256']:
        raise ValueError('Word forecasts changed since the original movie evaluation')
    qp=np.load(orig);np0=np.load(npath)
    groups={k:defaultdict(list) for k in ('all_extreme','all_ordered','first_person_extreme','first_person_ordered')}
    pairs=Counter();distribution=Counter();pooled=[];segments_n4=0
    for m in folds['documents']:
        if m['split']!='test':continue
        d=m['doc'];words=np.asarray(docs[d]['words']);valid,v,_=outcomes(docs[d],m,raw[d],'CharE')
        q=qp['CharE__'+d][:,3];ner=np0[d][:,0];first=np0[d][:,2]>0
        selected=valid&(v>0)
        distribution.update(v[selected].tolist())
        for loc in np.flatnonzero(selected):
            pooled.append([v[loc],q[loc],ner[loc],m['group'],words[loc,4]])
        for s in m['segments']:
            region=selected&(words[:,4]==s['index'])
            if np.any(v[region]>3):segments_n4+=1
            for area,mask in [('all',region),('first_person',region&first)]:
                scores=np.stack((q[mask],ner[mask]),axis=1)
                votes=v[mask]
                if len(votes)<2:continue
                for method,cond in [('extreme',(votes[:,None]>=3)&(votes[None,:]==1)),
                                    ('ordered',votes[:,None]>votes[None,:])]:
                    high,low=np.nonzero(cond)
                    if not len(high):continue
                    key=area+'_'+method
                    dif=scores[high]-scores[low];arr=(dif>0)+.5*(dif==0)
                    groups[key][m['group']].append(arr.mean(axis=0));pairs[key]+=len(dif)
    ret={}
    for k,rows in groups.items():
        points={g:np.mean(v,axis=0) for g,v in rows.items()}
        ret[k]={'movies':len(points),'segments':sum(len(v) for v in rows.values()),'word_pairs':pairs[k],
                'ordering':movie_mean(points) if points else None}
    v=np.array([r[0] for r in pooled]);x=np.array([[r[1],r[2]] for r in pooled]);high=x[v>=3];low=x[v==1]
    d=high[:,None,:]-low[None,:,:]
    desc=((d>0)+.5*(d==0)).mean(axis=(0,1))
    dump(RESULTS/'within-marked-consensus.json',{'protocol_sha256':sha(PROTOCOL),'known_marked_movie_words':len(v),
         'movie_word_max_votes':dict(sorted(distribution.items())),'movie_segments_with_votes_above_three':segments_n4,
         'within_segment':ret,'pooled_descriptive_rank_probability':{'qwen':desc[0],'ner':desc[1]},
         'source_has_rater_ids_or_registered_denominators':False})
    print('all',ret['all_extreme'],'first',ret['first_person_extreme'],'votes',distribution,'extra',segments_n4,'pooled',desc)


if __name__=='__main__':audit()
