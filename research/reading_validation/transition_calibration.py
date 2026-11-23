"""Independently audit outcome orientation, grouped evaluation and calibration error."""
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from research.reading_validation.transition_external_prepare import OUT, csv_rows
from research.reading_validation.transition_external_source import open_verified
from research.reading_validation.transition_score import dump, sha, with_interval


ARMS = ('lexical','display','lexical_surprisal','local_state','integration','student','directed','static')


def calibrate(y,n,p,group,draws):
    n_groups=draws.shape[1]
    # Per-group numerators support cluster draws without treating judgments as independent.
    count=lambda v: np.bincount(group,weights=v,minlength=n_groups)
    nby=count(n); yby=count(y); pby=count(n*p)
    brier=count(y*(1-p)**2 + (n-y)*p**2)
    bin_count=[]; obs=[]; forecast=[]
    which=np.minimum((p*10).astype(int),9)
    for decile in range(10):
        valid=which==decile
        bin_count.append(count(valid*n));obs.append(count(valid*y));forecast.append(count(valid*n*p))
    stats={}; weights=np.vstack([np.ones(n_groups,dtype=int),draws]); denom=weights@nby
    bs=weights@brier/denom; avg=weights@pby/denom; human=weights@yby/denom
    ece=np.abs(weights @ (np.array(obs)-np.array(forecast)).T).sum(axis=1)/denom
    for k,v in [('brier',bs),('mean_prediction',avg),('observed_rate',human),('predicted_minus_observed',avg-human),('ece_ten_fixed_bins',ece)]:
        stats[k]=with_interval(v[0],v[1:])
    stats['bins']=[{'lower':k/10,'upper':(k+1)/10,'judgments':int(v.sum()),'marks':int(obs[k].sum()),
                    'expected_marks':float(forecast[k].sum())} for k,v in enumerate(bin_count)]
    return stats


def direct_rank(y,n,p):
    by_score=Counter();neg_by_score=Counter()
    for a,b,s in zip(y,n,p,strict=True):
        by_score[s]+=a;neg_by_score[s]+=b-a
    pos=int(y.sum());neg=int(n.sum()-pos)
    tp=fp=0;ap=0.;auc=0.
    for s,count in sorted(by_score.items(),reverse=True):
        false=neg_by_score[s]
        auc+=false*(tp+.5*count)
        tp+=count;fp+=false
        ap+=count*tp/(tp+fp)
    return ap/pos,auc/(pos*neg)


def audit():
    outcomes=json.loads((OUT/'scores.json').read_text())
    base=pd.read_csv(OUT/'heldout-scored.csv')
    meta=json.loads((OUT/'stimuli.json').read_text())['material']
    raw_co=json.loads(open_verified('TestData.json'));raw_de=csv_rows('test_1355_clean.csv')
    checked=Counter();models={};reasons={};max_error=0.
    missing=0
    for source,result_key in [('cohesentia','cohesentia_per_rater'),('decor','decor')]:
        independent={};reasons[source]=Counter()
        for stop in meta:
            if stop['source']!=source or stop['split'] not in ('external','holdout'):
                continue
            if source=='cohesentia':
                _,story,idx=stop['source_ref'];raw=raw_co[story]['IncrementalData']
                sets=[]
                for who,marks in raw['reasons'].items():
                    if any(not k.isdecimal() or int(k)>=len(raw['sentences']) for k in marks):
                        raise ValueError('New ambiguous local annotator in official test')
                    one=set(marks.get(str(idx),[]));sets.append(bool(one));reasons[source].update(one)
                pair=sum(sets),len(sets)
            else:
                raw=raw_de[stop['source_ref'][1]]
                # The authors define blank rewrite and all missing reasons for coherent examples.
                marked=bool(raw['Rewrite'].strip()); pair=(int(marked),1)
                assert marked==(float(raw['label'])==0.)
                reasons[source].update(i for i in range(1,8) if raw['R'+str(i)]=='1.0')
                missing+=sum(raw['R'+str(i)]=='-1.0' for i in range(1,8))
            independent[stop['id']]=pair;checked[source]+=1
        rows=base[base.source.eq(source)].copy(); y=rows.y.to_numpy(dtype=int);n=rows.total.to_numpy(dtype=int)
        assert [independent[k] for k in rows.id]==list(zip(y,n))
        if source=='decor' and (y.sum()!=446 or (n-y).sum()!=906):
            raise ValueError('Marginal detection prevalence differs from original paper, section 5.1.2')
        group,ids=np.unique(rows.group.to_numpy(),return_inverse=True);g=len(group)
        draws=np.random.default_rng(9271).multinomial(g,[1/g]*g,size=10000)
        assert np.all(draws.sum(axis=1)==g)
        models[source]={}
        for arm in ARMS:
            p=rows[arm].to_numpy(dtype=float);s=outcomes[result_key]['models'][arm]
            ap,auc=direct_rank(y,n,p)
            max_error=max(max_error,abs(ap-s['ap']['estimate']),abs(auc-s['auroc']['estimate']))
            alarm=p>s['cutoff']
            counts=s['alert_counts']
            assert int(y[alarm].sum())==counts['true_positive'] and int((n[alarm]-y[alarm]).sum())==counts['false_positive']
            models[source][arm]=calibrate(y,n,p,ids,draws)
    if max_error>1e-10:raise ValueError('Independent AP or AUROC arithmetic differs')
    out={'analysis':'calibration and audit after primary test','forecast_sha256':sha(OUT/'heldout-predictions.csv'),
         'scored_data_sha256':sha(OUT/'heldout-scored.csv'),'independently_aligned_boundaries':checked,
         'reasons_positive_votes':reasons,'decor_missing_reason_cells':int(missing),'max_abs_ranking_replay_error':max_error,
         'bootstrap':'each draw resamples 49 whole stories or 100 whole essays with replacement; fixed local readers',
         'metrics':models}
    dump(OUT/'calibration-audit.json',out)
    return out


if __name__=='__main__':
    result=audit();print('independent ranking residual',result['max_abs_ranking_replay_error'])
    for source in result['metrics']:
        for arm in ['integration','student','lexical_surprisal','directed']:
            m=result['metrics'][source][arm]
            print(source,arm,[(k,round(m[k]['estimate'],4)) for k in ['brier','mean_prediction','observed_rate','ece_ten_fixed_bins']])
