"""Evaluate sealed predictions of independent movie reader markings by full movie cluster."""
from collections import Counter
import json

import numpy as np

from research.reading_validation.transition_calibration import direct_rank
from research.reading_validation.transition_external_score import evaluate_one, grouped_distribution
from research.reading_validation.transition_score import dump, sha, with_interval
from research.reading_validation.transition_snac_fit import inputs,TARGETS,ARMS
from research.reading_validation.transition_snac_outcomes import outcomes
from research.reading_validation.transition_snac_prepare import RESULTS,source


def top_one(y,p,segment,group,draws):
    n=draws.shape[1];sums=np.zeros(n);counts=np.zeros(n);base=np.zeros(n)
    for s in np.unique(segment):
        selected=segment==s;actual=y[selected];score=p[selected];g=np.unique(group[selected])
        if len(g)!=1:raise ValueError('One source segment appeared in multiple movie clusters')
        if actual.any() and (1-actual).any():
            maxima=score==score.max();sums[g[0]]+=actual[maxima].mean();base[g[0]]+=actual.mean();counts[g[0]]+=1
    value=with_interval(sums.sum()/counts.sum(),draws@sums/(draws@counts))
    value.update(eligible_segments=int(sum(counts)),within_segment_random=float(base.sum()/counts.sum()))
    return value


def score():
    docs,meta=inputs();material=meta['documents'];raw=source();lock=json.loads((RESULTS/'prediction-lock.json').read_text());output={}
    model_path=RESULTS/'readouts.json';forecast=RESULTS/'heldout-probabilities.npz';
    if sha(forecast)!=lock['forecast_sha256'] or sha(model_path)!=lock['model_sha256']:raise ValueError('SNaC forecast seal differs')
    data=np.load(forecast,allow_pickle=False);spec=json.loads(model_path.read_text())['readouts']
    for target in TARGETS:
        labels=[];capitalized=[];probs=[];groups=[];segments=[];rows=[];audit=Counter()
        for m in material:
            if m['split']!='test':continue
            did=m['doc'];doc=docs[did];mask,votes,record=outcomes(doc,m,raw[did],target);audit.update(record)
            matrix=data[target+'__'+did];ws=np.array(doc['words']);
            if matrix.shape!=(len(mask),len(ARMS)):raise ValueError('Incomplete movie forecasts')
            labels.extend(votes[mask]);probs.extend(matrix[mask]);groups.extend([m['group']]*int(mask.sum()))
            for word,s in zip(ws[mask],votes[mask],strict=True):
                a,b,_,_,seg,_=word;capitalized.append(doc['text'][a:b][0].isupper());segments.append(did+':'+str(seg))
                rows.append([did,int(a),int(b),int(seg),int(s)])
        y=np.array(labels);matrix=np.array(probs);capitalized=np.array(capitalized,dtype=bool);segments=np.array(segments)
        unique,ids,draws=grouped_distribution(groups);truth=(y>0).astype(int);n=np.ones(len(y),dtype=int)
        methods={};bootstrap={}
        for i,arm in enumerate(ARMS):
            p=matrix[:,i];cut=spec[target][arm]['threshold'];m,boot=evaluate_one(truth,n,p,cut,ids,draws)
            m['top_word_in_marked_segment']=top_one(truth,p,segments,ids,draws)
            for k,subset,positive in [('capitalized_word',capitalized,truth),('noncapitalized_word',~capitalized,truth),('two_or_more_votes',n.astype(bool),(y>=2).astype(int))]:
                yy=positive[subset];nn=n[subset];ap,auc=direct_rank(yy,nn,p[subset])
                m[k]={'words':int(subset.sum()),'positive':int(yy.sum()),'ap':ap,'auroc':auc}
            methods[arm]=m;bootstrap[arm]=boot
        effects={}
        for control in ('surface','surprisal','local'):
            effects[control]={}
            for metric in ('ap','auroc'):
                delta=methods['antecedents'][metric]['estimate']-methods[control][metric]['estimate']
                effects[control][metric]=with_interval(delta,bootstrap['antecedents'][metric]-bootstrap[control][metric])
        output[target]={'heldout_movies':len(unique),'word_sites':len(y),'human_marked':int(truth.sum()),
            'prevalence':float(truth.mean()),'spans':dict(audit),'models':methods,'antecedents_minus_control':effects,
            'increment_beyond_both_strong_controls':all(effects[c]['ap']['family_bootstrap_95_interval'][0]>0 for c in ('surprisal','local'))}
        dump(RESULTS/(target+'-aligned-outcomes.json'),{'columns':['summary_id','start','end','segment','maximum_span_votes'],'rows':rows})
        print(target, 'books-to-movies',len(unique),'marked',int(truth.sum()),'site count',len(y), flush=True)
        for arm in ARMS:print(arm,methods[arm]['ap'],methods[arm]['alert_counts'],flush=True)
    dump(RESULTS/'scores.json',output)
    return output


if __name__=='__main__': score()
