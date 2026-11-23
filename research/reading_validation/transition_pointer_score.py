"""Evaluate prior-word probabilities against each reader's original antecedent click."""
from argparse import ArgumentParser
from collections import Counter,defaultdict
import json
from pathlib import Path

import numpy as np

from research.reading_validation.transition_external_score import grouped_distribution
from research.reading_validation.transition_pointer_data import OUT,PROTOCOL,ROOT
from research.reading_validation.transition_score import dump,sha,with_interval


ARMS=('uniform','recency','lexical','pooled','headwise')
METRICS=('region_nll','top1_inside','top3_inside','reciprocal_rank','clicked_probability',
         'entity_top1_inside','entity_probability','region_words','words_back','entity_word_coverage_fraction')


def hit_k(scores,match,k):
    k=min(len(scores),k)
    if not match.any():return 0.
    threshold=np.partition(scores,-k)[-k];certain=scores>threshold
    if np.any(certain&match):return 1.
    tie=scores==threshold;total=int(tie.sum());ok=int((tie&match).sum());slots=k-int(certain.sum())
    if slots>total-ok:return 1.
    missed=1.
    for draw in range(slots):missed*=(total-ok-draw)/(total-draw)
    return 1.-missed


def mrr(scores,match):
    cutoff=scores[match].max();better=int((scores>cutoff).sum());tied=scores==cutoff;n=int(tied.sum());m=int((tied&match).sum())
    prob=1.;out=0.
    for j in range(1,n-m+2):
        at=prob*m/(n-j+1);out+=at/(better+j);prob-=at
    return out


def forecast_one(logit,scope,equivalent):
    peak=logit.max();shifted=logit-peak;normal=np.log(np.exp(shifted).sum())
    selected=shifted[scope];s=selected.max()
    logmass=float(s+np.log(np.exp(selected-s).sum())-normal)
    distribution=np.exp(shifted-normal)
    return np.array([-logmass,hit_k(logit,scope,1),hit_k(logit,scope,3),mrr(logit,scope),np.exp(logmass),
                     hit_k(logit,equivalent,1),distribution[equivalent].sum(),scope.sum(),len(scope)-np.flatnonzero(scope)[-1],
                     equivalent.mean()])


def summarize(mentions):
    if not mentions:return {'articles':0,'queries':0,'clicks':0}
    bound=defaultdict(list)
    for doc,_,_,vals in mentions:bound[doc].append(vals)
    ids=sorted(bound);per=np.stack([np.stack(bound[d]).mean(0) for d in ids])
    if len(ids)>1:
        _,_,draw=grouped_distribution(ids);boot=(draw@per.reshape(len(ids),-1)).reshape(-1,len(ARMS),len(METRICS))/len(ids)
    else:boot=np.full((1,len(ARMS),len(METRICS)),np.nan)
    center=per.mean(0);values={}
    for a,key in enumerate(ARMS):
        values[key]={m:with_interval(center[a,j],boot[:,a,j]) for j,m in enumerate(METRICS)}
    deltas={}
    for left,right in [('lexical','recency'),('headwise','lexical'),('headwise','pooled'),('pooled','lexical'),('headwise','uniform')]:
        a=ARMS.index(left);b=ARMS.index(right);deltas[left+'_minus_'+right]={}
        for j,m in enumerate(METRICS[:7]):
            d=with_interval(center[a,j]-center[b,j],boot[:,a,j]-boot[:,b,j])
            pos=per[:,a,j]-per[:,b,j];d['articles_improved']=int((pos<0 if m=='region_nll' else pos>0).sum())
            deltas[left+'_minus_'+right][m]=d
    return {'articles':len(ids),'queries':len(mentions),'clicks':sum(v[2] for v in mentions),'methods':values,'paired':deltas}


def human_anchor(spans,size):
    total=sum(s[2] for s in spans);votes=np.zeros(size);mantissa=0.;r_same=0;e_same=0
    for i,(a,b,n,e,_) in enumerate(spans):
        votes[a:b]+=n
        for j,(c,d,m,f,_) in enumerate(spans):
            pairs=n*(m-int(i==j))
            mantissa+=pairs*max(0,min(b,d)-max(a,c))/(d-c)
            r_same+=pairs*((a,b)==(c,d));e_same+=pairs*(e==f)
    denom=total*(total-1)
    return [mantissa/denom,r_same/denom,e_same/denom,votes.max()/total]


def verify():
    x=np.array([0.,0.,0.,0.]);hit=np.array([0,1,1,0],bool)
    np.testing.assert_allclose(hit_k(x,hit,1),.5);np.testing.assert_allclose(hit_k(x,hit,3),1)
    np.testing.assert_allclose(hit_k(x,np.zeros_like(hit),1),0)
    y=forecast_one(x,hit,hit);np.testing.assert_allclose(y[0],-np.log(.5))
    z=human_anchor([[0,2,2,0,0],[2,3,2,1,0]],3)
    np.testing.assert_allclose(z,[1/3,1/3,1/3,.5])


def run(source):
    verify()
    receipt=json.loads((source/'receipt.json').read_text());seal=json.loads((OUT/'source-seal.json').read_text())
    goldfile=OUT/'external-clicks.json';archive=source/'candidate-word-logits.npz';weights=source/'source-weights.npz'
    if (sha(archive)!=receipt['predictions_sha256'] or sha(weights)!=receipt['weights_sha256'] or receipt['protocol_sha256']!=sha(PROTOCOL)
            or seal['input_sha256']!=receipt['input_sha256'] or seal['gold_sha256']!=sha(goldfile) or receipt['arms']!=list(ARMS[1:])):
        raise ValueError('Antecedent pointer, registered packet or blinded human outcomes changed')
    gold=json.loads(goldfile.read_text())['articles'];material={r['id']:r for r in map(json.loads,(ROOT/'pointer-stimuli.jsonl').read_text().splitlines())}
    pred=np.load(archive);all_queries={d+'/'+str(q) for d,r in material.items() if r['fold']!='train' for q in r['queries']}
    if set(pred.files)!=all_queries:raise ValueError('Not all original prefix-visible test pronouns were forecast')
    held=defaultdict(list);human=defaultdict(list);intake=Counter()
    for d,annotations in gold.items():
        r=material[d];genre=d.split('-',2)[1]
        for key,label in annotations.items():
            q=int(key);spans=label['spans'];logits=np.vstack([np.zeros(q),pred[d+'/'+key]]).astype(float)
            if logits.shape!=(len(ARMS),q):raise ValueError('An earlier text word is absent from the candidate space')
            human[d].append(human_anchor(spans,q))
            doc_values=defaultdict(list);doc_weights=defaultdict(list);disputed=len(set(s[3] for s in spans))>1
            for a,b,n,e,unseen in spans:
                if not 0<=a<b<=q or not 0<=unseen<=n:raise ValueError('A literal human click has an invalid extent')
                mask=np.zeros(q,bool);mask[a:b]=True;equiv=np.zeros(q,bool)
                for l,t in label['equivalent'][e]:equiv[l:t]=True
                if np.any(mask&~equiv):raise ValueError('Expert identity did not include its original clicked mention')
                v=np.stack([forecast_one(s,mask,equiv) for s in logits])
                for typ,weight in [('all',n),('unseen_reader',unseen),('nonlocal_10words',n if q-(b-1)>=10 else 0),
                                   ('single_word_region',n if b-a==1 else 0),('disputed_entity',n if disputed else 0),
                                   ('genre_'+genre,n)]:
                    if weight:doc_values[typ].append(v);doc_weights[typ].append(weight)
                intake['clicks']+=n;intake['single_word_clicks']+=n*(b-a==1)
            for typ,vals in doc_values.items():
                ws=doc_weights[typ];held[typ].append((d,q,sum(ws),np.average(np.stack(vals),axis=0,weights=ws)))
                if typ=='all':
                    held['all_distinct_works'].append((r['group'],q,sum(ws),np.average(np.stack(vals),axis=0,weights=ws)))
    if intake['clicks']!=seal['target_intake']['test']['reader_clicks']:raise ValueError('Missing original reader response')
    output={k:summarize(v) for k,v in held.items()}
    ids=sorted(human);_,_,draw=grouped_distribution(ids);h=np.stack([np.average(human[d],0) for d in ids]);summary={}
    for i,name in enumerate(('word_from_another_click_in_held_click','same_exact_region','same_expert_entity','empirical_best_word_fraction')):
        summary[name]=with_interval(h[:,i].mean(),draw@h[:,i]/len(ids))
    for f in ('receipt.json','source-weights.npz','candidate-word-logits.npz'):
        target=OUT/f;target.write_bytes((source/f).read_bytes())
        if sha(target)!=sha(source/f):raise ValueError('Output copy changed')
    dump(OUT/'scores.json',{'protocol_sha256':sha(PROTOCOL),'source_forecast_sha256':sha(archive),'external_human_sha256':sha(goldfile),
                            'distinct_eligible_clickers':seal['distinct_eligible_clickers'],'unseen_eligible_gold_clickers':seal['unseen_eligible_gold_clickers'],
                            'intake':dict(intake),'reader_region_consistency':summary,'evaluations':output})
    print(output['all']['methods']['headwise'],output['all']['paired']['headwise_minus_lexical'],flush=True)


if __name__=='__main__':
    cli=ArgumentParser(description=__doc__);cli.add_argument('--features',type=Path,required=True)
    run(cli.parse_args().features)
