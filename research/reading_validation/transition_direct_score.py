"""Compare source-only observers to movie labels and independent entity-choice disagreement."""
from argparse import ArgumentParser
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.reading_validation.transition_direct_data import OUT,PRON,PROTOCOL,ROOT,TASKS
from research.reading_validation.transition_external_score import evaluate_one,grouped_distribution
from research.reading_validation.transition_score import dump,sha,with_interval
from research.reading_validation.transition_snac_fit import inputs
from research.reading_validation.transition_snac_outcomes import outcomes
from research.reading_validation.transition_snac_prepare import source


def movie_scores(probs,arms,cutoffs):
    docs,meta=inputs();raw=source();out={}
    for t,task in enumerate(TASKS):
        X=[];y=[];gs=[];forms=[];alignment=defaultdict(int)
        for m in meta['documents']:
            if m['split']!='test':continue
            d=m['doc'];r=docs[d];ok,v,c=outcomes(r,m,raw[d],task);idx=np.flatnonzero(ok)
            for k,n in c.items():alignment[k]+=n
            X.extend(probs[d][ok,:,t]);y.extend(v[ok]>0);gs.extend([d]*len(idx))
            forms.extend(r['text'][a:b].casefold() for a,b,*_ in np.asarray(r['words'])[ok])
        X=np.array(X);y=np.array(y,dtype=int);gs=np.array(gs);forms=np.array(forms)
        masks={'all':np.ones(len(y),dtype=bool)}
        if task=='CorefE':masks['personal_pronouns']=np.isin(forms,sorted(PRON))
        out[task]={'source_alignment':dict(alignment)}
        for name,m in masks.items():
            yy=y[m];groups,idx,draws=grouped_distribution(gs[m])
            metrics={};boots={}
            for a,arm in enumerate(arms):
                p=X[m,a];metrics[arm],samples=evaluate_one(yy,np.ones_like(yy),p,cutoffs[arm][t],idx,draws)
                loss=(yy-p)**2;sum_error=np.bincount(idx,weights=loss);tot=np.bincount(idx)
                metrics[arm]['brier']=with_interval(loss.mean(),(draws@sum_error)/(draws@tot))
                boots[arm]=samples['ap']
            differences={}
            for a,b in [('direct','historic'),('shared','direct'),('shared','ner'),('direct','ner'),
                        ('historic','ner'),('shared','surface')]:
                differences[a+'_minus_'+b]=with_interval(metrics[a]['ap']['estimate']-metrics[b]['ap']['estimate'],boots[a]-boots[b])
            out[task][name]={'movies':len(groups),'words':len(yy),'human_marked':int(yy.sum()),'methods':metrics,'paired_ap':differences}
            print(task,name,len(yy),int(yy.sum()),{k:round(i['ap']['estimate'],3) for k,i in metrics.items()},flush=True)
    return out


def concordance(records,arm,key,same_form):
    by_doc=defaultdict(list)
    for r in records:
        if r[key] is not None:by_doc[r['doc']].append(r)
    vals={};pairs={}
    for d,items in sorted(by_doc.items()):
        h=np.asarray([r[key] for r in items]);s=np.asarray([r[arm] for r in items])
        valid=h[:,None]>h[None,:]+1e-12
        if same_form:
            forms=np.array([r['form'] for r in items]);valid&=forms[:,None]==forms[None,:]
        high,low=np.nonzero(valid)
        if not len(high):continue
        diff=s[high]-s[low]
        vals[d]=float(((diff>0)+.5*(diff==0)).mean());pairs[d]=len(diff)
    return vals,pairs


def game_scores(probs,arms,material):
    for r in material:
        for a,arm in enumerate(arms):
            if arm!='ner':
                r[arm]=float(probs[r['doc']][r['word'],a,TASKS.index('CorefE')])
        if not 0<=r['word']<len(probs[r['doc']]):raise ValueError('Reader choice location outside forecasts')
    result={}
    for key in ('antecedent_disagreement','all_disagreement'):
        observed=[r for r in material if r[key] is not None]
        resp=np.array([r[key] for r in observed])
        endpoint={'mentions':len(observed),'documents':len({r['doc'] for r in observed}),
                  'mentions_with_disagreement':int((resp>0).sum()),
                  'pooled_spearman':{a:float(pd.Series(resp).rank().corr(pd.Series([r[a] for r in observed]).rank())) for a in arms if a!='ner'}}
        for condition,rows,same in [('within_document',observed,False),('within_same_pronoun_form_and_document',observed,True)]+[
                ('genre_'+g,[r for r in observed if r['genre']==g],False) for g in sorted({r['genre'] for r in observed})]:
            vs={};samples={};ids=None;pair_counts=None
            for arm in arms:
                if arm=='ner':continue
                estimates,counts=concordance(rows,arm,key,same)
                if ids is None:
                    ids=sorted(estimates);pair_counts=counts
                    if len(ids)<2:break
                    _,_,draws=grouped_distribution(np.asarray(ids))
                if sorted(estimates)!=ids:raise ValueError('Unmatched external document comparisons')
                x=np.asarray([estimates[d] for d in ids]);b=draws@x/len(ids)
                vs[arm]=with_interval(x.mean(),b);samples[arm]=b
            entry={'qualifying_documents':len(ids or []),'ordered_pronoun_pairs':int(sum((pair_counts or {}).values())),'methods':vs}
            if len(vs)>1:
                entry['paired_differences']={a+'_minus_'+b:with_interval(vs[a]['estimate']-vs[b]['estimate'],samples[a]-samples[b])
                     for a,b in [('direct','historic'),('shared','direct'),('shared','surface'),('direct','surface')]}
            endpoint[condition]=entry
        result[key]=endpoint
        print(key,endpoint['within_document'],flush=True)
    return result


def coefficient_geometry(weights,width):
    result={}
    for arm in ('direct','shared'):
        w=weights[arm+'__weights'][-2*width:]
        sv=np.linalg.svd(w,compute_uv=False);p=sv/sv.sum()
        norm=np.linalg.norm(w,axis=0)
        result[arm]={'singular_values':sv.tolist(),'entropy_effective_rank':float(np.exp(-(p*np.log(p)).sum())),
                     'task_cosines':((w.T@w)/(norm[:,None]*norm[None,:])).tolist()}
    return result


def score(path):
    meta=json.loads((path/'receipt.json').read_text());source_seal=json.loads((OUT/'input-seal.json').read_text())
    archive=path/'word-probabilities.npz'; external=OUT/'external-reader-outcomes.json'
    actual=json.loads(external.read_text())
    if (meta['protocol_sha256']!=sha(PROTOCOL) or source_seal['input_sha256']!=meta['input_sha256']
            or actual['packet_sha256']!=meta['input_sha256'] or sha(archive)!=meta['predictions_sha256']
            or sha(path/'weights.npz')!=meta['weights_sha256'] or tuple(meta['tasks'])!=TASKS):
        raise ValueError('Native projections or human packet did not match the sealed source')
    probs=dict(np.load(archive));arms=meta['arms']
    # Score independent interpretations first, before any outcome-driven dependence on movies.
    game=game_scores(probs,arms,actual['rows'])
    movie=movie_scores(probs,arms,meta['cutoffs'])
    out=OUT/meta['candidate'];out.mkdir(exist_ok=True)
    for name in ('receipt.json','weights.npz','word-probabilities.npz'):
        dst=out/name;dst.write_bytes((path/name).read_bytes())
        if sha(dst)!=sha(path/name):raise ValueError('Artifact copy changed')
    dump(out/'scores.json',{'forecast_sha256':meta['predictions_sha256'],'external_response_sha256':sha(external),
         'primary_protocol_sha256':sha(PROTOCOL),'independent_game':game,'movie_words':movie,
         'native_coefficient_geometry':coefficient_geometry(np.load(path/'weights.npz'),meta['feature_metadata']['native_width'])})


if __name__=='__main__':
    cmd=ArgumentParser(description=__doc__);cmd.add_argument('--features',required=True,type=Path)
    score(cmd.parse_args().features)
