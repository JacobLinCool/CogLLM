"""Evaluate unmarked name-region recovery and two-name preference on independent articles."""
from argparse import ArgumentParser
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from research.reading_validation.transition_external_score import grouped_distribution
from research.reading_validation.transition_pointer_gap_data import OUT,PROTOCOL,ROOT
from research.reading_validation.transition_pointer_score import ARMS,hit_k
from research.reading_validation.transition_score import dump,sha,with_interval


ENDPOINTS=('open_top1','open_top3','name_nll','two_name_correct','length_adjusted_two_name_correct',
           'two_name_conditional_nll','named_mass','other_name_mass','both_named_mass')


def reduce(rows):
    if not rows:return {'articles':0,'examples':0}
    groups=defaultdict(list)
    for r,v in rows:groups[r['article']].append(v)
    ids=sorted(groups);matrix=np.stack([np.stack(groups[i]).mean(0) for i in ids]);mean=matrix.mean(0)
    if len(ids)>1:
        _,_,draws=grouped_distribution(ids)
        boot=(draws@matrix.reshape(len(ids),-1)).reshape(len(draws),*mean.shape)/len(ids)
    else:boot=np.full((1,*mean.shape),np.nan)
    results={a:{k:with_interval(mean[i,j],boot[:,i,j]) for j,k in enumerate(ENDPOINTS)} for i,a in enumerate(ARMS)}
    paired={}
    for a,b in [('headwise','lexical'),('headwise','pooled'),('lexical','recency'),('pooled','lexical'),('headwise','uniform')]:
        i=ARMS.index(a);o=ARMS.index(b)
        paired[a+'_minus_'+b]={k:with_interval(mean[i,j]-mean[o,j],boot[:,i,j]-boot[:,o,j]) for j,k in enumerate(ENDPOINTS)}
    return {'articles':len(ids),'examples':len(rows),'methods':results,'paired':paired}


def run(path):
    seal=json.loads((OUT/'source-seal.json').read_text());r=json.loads((path/'receipt.json').read_text())
    pfile=path/'candidate-word-logits.npz';gfile=OUT/'external-regions.json'
    if (sha(pfile)!=r['predictions_sha256'] or r['input_sha256']!=seal['input_sha256'] or r['protocol_sha256']!=sha(PROTOCOL)
            or sha(gfile)!=seal['gold_sha256'] or r['external_weight_updates']!=0 or r['arms']!=list(ARMS[1:])):
        raise ValueError('GAP outcome-free forecast or its frozen source changed')
    gold=json.loads(gfile.read_text())['rows'];data=list(map(json.loads,(ROOT/'gap-pointer-stimuli.jsonl').read_text().splitlines()))
    p=np.load(pfile)
    if set(p.files)!={row['id']+'/'+str(q) for row in data for q in row['queries']}:raise ValueError('The external pointer did not score every visible pronoun')
    held=defaultdict(list);nearest=defaultdict(list);counts=defaultdict(int)
    for item in gold:
        q=item['q'];a,b=tuple([item[k] for k in ('A','B')][::-1] if item['target'] else [item['A'],item['B']])
        if max(a+b)>=q or set(a)&set(b):raise ValueError('Two regions do not mark distinct earlier words')
        outcome=[]
        mask=np.zeros(q,bool);mask[a]=True
        for logits in np.vstack([np.zeros(q),p[item['id']+'/'+str(q)]]):
            logits=np.asarray(logits,dtype=float);weights=np.exp(logits-logits.max());norm=weights.sum()
            ya,yb=weights[a].sum(),weights[b].sum()
            if min(ya,yb)<=0:raise ValueError('An original candidate received zero mass')
            correct=float(ya>yb)+.5*(ya==yb);la=ya/len(a);lb=yb/len(b);size_fix=float(la>lb)+.5*(la==lb)
            outcome.append([hit_k(logits,mask,1),hit_k(logits,mask,3),-np.log(ya/norm),correct,size_fix,
                            -np.log(ya/(ya+yb)),ya/norm,yb/norm,(ya+yb)/norm])
        near=int(max(a)>max(b));gender='masculine' if item['form'] in ('he','his','him','himself') else 'feminine'
        labels=['all',gender]
        if near==0:labels.append('farther_correct_name')
        if len(a)==len(b):labels.append('equal_length_names')
        for cat in labels:held[cat].append((item,np.array(outcome)));nearest[cat].append((item['article'],near))
    if len(held['all'])!=seal['eligible_examples']:raise ValueError('Different human evaluation sample')
    result={k:reduce(v) for k,v in held.items()}
    for k,v in nearest.items():
        ids,ix,draws=grouped_distribution([x[0] for x in v]);z=np.bincount(ix,weights=[x[1] for x in v])/np.bincount(ix)
        result[k]['choose_nearer_annotated_name']=with_interval(z.mean(),draws@z/len(z))
    for name in ('receipt.json','candidate-word-logits.npz'):
        dest=OUT/name;dest.write_bytes((path/name).read_bytes())
        if sha(dest)!=sha(path/name):raise ValueError('External audit copy differs')
    dump(OUT/'scores.json',{'protocol_sha256':sha(PROTOCOL),'public_outcome_sha256':sha(gfile),'forecast_sha256':sha(pfile),
                            'masked_input_sha256':seal['input_sha256'],'evaluations':result})
    print({k:[v['open_top1']['estimate'],v['name_nll']['estimate'],v['two_name_correct']['estimate']]
           for k,v in result['all']['methods'].items()},flush=True)


if __name__=='__main__':
    p=ArgumentParser(description=__doc__);p.add_argument('--features',type=Path,required=True)
    run(p.parse_args().features)
