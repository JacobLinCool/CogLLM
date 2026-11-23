"""Set a source-group precision target and evaluate transferred editorial coverage."""
from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np

from research.reading_validation.transition_external_fit import predict
from research.reading_validation.transition_external_score import grouped_distribution
from research.reading_validation.transition_score import dump, sha, with_interval
from research.reading_validation.transition_snac_fit import inputs
from research.reading_validation.transition_snac_outcomes import outcomes
from research.reading_validation.transition_snac_prepare import RESULTS, source

GRID=(.80,.85,.90,.925,.95,.96,.97,.98,.985,.99,.9925,.995,.9975,.999)
PROTOCOL=Path('research/reading_validation/transition-author-precision-protocol.md')


def counts(y, predicted, group):
    size=int(group.max())+1
    return np.stack([np.bincount(group,weights=v,minlength=size) for v in
                     [predicted & y,predicted & ~y,~predicted & y,~predicted & ~y]],axis=1)


def measures(total):
    tp,fp,fn,tn=total.T
    def ratio(a,b):return np.divide(a,b,out=np.zeros_like(a,dtype=float),where=b>0)
    return {'precision':ratio(tp,tp+fp),'recall':ratio(tp,tp+fn),'f1':ratio(2*tp,2*tp+fp+fn),
            'false_alarms_per_100_negatives':100*ratio(fp,fp+tn),
            'flags_per_1000_words':1000*ratio(tp+fp,tp+fp+fn+tn)}


def run(directory):
    archive=directory/'word-features.npz'
    if sha(archive)!=json.loads((directory/'receipt.json').read_text())['feature_sha256']:
        raise ValueError('Expected the original serving representation receipt')
    weights=json.loads((RESULTS/'readouts.json').read_text())['readouts']['CharE']['antecedents']
    docs,meta=inputs();raw=source();values=np.load(archive)
    training=[];dev_p=[];dev_y=[];dev_group=[]
    for m in meta['documents']:
        key=m['doc']
        if m['split']=='train':
            training.extend(predict(values[key],weights['weights']))
        elif m['split']=='dev':
            valid,votes,_=outcomes(docs[key],m,raw[key],'CharE')
            dev_p.extend(predict(values[key][valid],weights['weights']))
            dev_y.extend(votes[valid]>0);dev_group.extend([m['group']]*int(valid.sum()))
    scores=np.array(dev_p);y=np.array(dev_y);grp,g,draws=grouped_distribution(dev_group)
    criteria=[]
    for q,cut in zip(GRID,np.quantile(training,GRID,method='higher'),strict=True):
        flags=scores>cut
        c=counts(y,flags,g)
        m=measures(c.sum(axis=0,keepdims=True));docs_with_alerts=int(np.count_nonzero(c[:,:2].sum(axis=1)))
        eligible=bool(flags.sum()>=20 and docs_with_alerts>=5)
        boot=measures(draws@c)['precision']
        criteria.append(({'quantile':q,'cutoff':float(cut),'dev_alerts':int(flags.sum()),
                         'dev_titles_with_alerts':docs_with_alerts, 'precision':float(m['precision'][0]),
                         'recall':float(m['recall'][0]),'eligible':eligible},boot))
    eligible=[(k,b) for k,b in criteria if k['eligible']]
    correction=float(np.quantile(np.stack([k['precision']-b for k,b in eligible]).max(axis=0),.95))
    for k,b in criteria:
        k['simultaneous_precision_lower']=max(0.,k['precision']-correction) if k['eligible'] else None
    accepted=[k for k,b in eligible if k['simultaneous_precision_lower']>=.70]
    chosen=max(accepted,key=lambda k:(k['recall'],k['simultaneous_precision_lower'],k['cutoff'])) if accepted else None
    lock={'protocol_sha256':sha(PROTOCOL),'primary_readout_sha256':sha(RESULTS/'readouts.json'),
          'source_titles':len(grp),'eligible_dev_words':len(y),'eligible_dev_positive':int(y.sum()),
          'candidate_cutoffs':[k for k,_ in criteria],'simultaneous_correction':correction,'chosen':chosen}
    dump(RESULTS/'author-precision-lock.json',lock)
    original=RESULTS/'heldout-probabilities.npz'
    if sha(original)!=json.loads((RESULTS/'prediction-lock.json').read_text())['forecast_sha256']:
        raise ValueError('Released original movie forecast differs')
    p0=np.load(original);movie_y=[];movie_p=[];movie_g=[]
    for m in meta['documents']:
        if m['split']!='test':continue
        d=m['doc'];valid,votes,_=outcomes(docs[d],m,raw[d],'CharE')
        movie_y.extend(votes[valid]>0);movie_p.extend(p0['CharE__'+d][valid,3]);movie_g.extend([m['group']]*int(valid.sum()))
    y=np.asarray(movie_y);p=np.asarray(movie_p);grp,g,draws=grouped_distribution(movie_g);result={}
    thresholds={'default':weights['threshold']}
    if chosen:thresholds['source_precision_target']=chosen['cutoff']
    for name,cut in thresholds.items():
        c=counts(y,p>cut,g);tot=c.sum(axis=0,keepdims=True);m=measures(tot);boot=measures(draws@c)
        result[name]={'threshold':cut,'movies_with_alerts':int(np.count_nonzero(c[:,:2].sum(axis=1))),
                      'movies':len(grp),'tp_fp_fn_tn':tot.astype(int).ravel().tolist(),
                      **{k:with_interval(v[0],boot[k]) for k,v in m.items()}}
    dump(RESULTS/'author-precision-scores.json',{'threshold_lock_sha256':sha(RESULTS/'author-precision-lock.json'),
         'known_external_words':len(y),'known_external_positive':int(y.sum()),'policies':result})
    print('selected',chosen,'peak lower bound',max(k['simultaneous_precision_lower'] for k,_ in eligible))
    print({n:(v['tp_fp_fn_tn'],v['precision'],v['recall']) for n,v in result.items()})


if __name__ == '__main__':
    parser=ArgumentParser(description=__doc__);parser.add_argument('--features',type=Path,required=True)
    run(parser.parse_args().features)
