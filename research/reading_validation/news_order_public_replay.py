"""Replay primary news findings directly from numeric forecasts, actual author releases and SHA identities."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


RES=Path('research/results')
NEWS=RES/'reading-validation-summeval'
ORDER=RES/'reading-validation-columbia-order'
RATINGS=Path('.cache/cogllm/reading-validation/summeval/original-model-annotations.jsonl')


def read(p):
    return json.loads(p.read_bytes())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def val(xs):
    a=np.asarray(xs,float)
    return float(a.mean())


def truth(scores):
    a=np.asarray(scores,float);return (np.sign(a)+1)/2


def news():
    if sha(RATINGS)!=read(NEWS/'inputs.json')['raw']:raise ValueError('Unexpected public grade file')
    design=read(NEWS/'story-geometry.json');verification={};v=defaultdict(list)
    for l in RATINGS.read_text().splitlines():
        row=json.loads(l);k=row['id']
        if design[k]['role']!='confirm':continue
        key=design[k]['systems'][row['model_id']];v[k,key].extend(r['coherence'] for r in row['expert_annotations'])
    methods={};complete=read(NEWS/'scores.json');conditional=read(NEWS/'ballot-scores.json')
    for path,names in [('frozen-predictions.npz',['adaptive','surprisal','essay_hazard','essay_lexical','geval']),('calibrated-predictions.npz',['independent_native','shared_native']),('ballot-forecasts.npz',['native_choice','native_absolute','form_choice','mixture_choice'])]:
        with np.load(NEWS/path,allow_pickle=False) as d:
            for n in names: methods[n]=dict(zip(d['keys'],d[n],strict=True))
    stories=sorted({k for k,_ in v});external={};paird={}
    for name,method in methods.items():
        bydoc={};loss=[]
        for story in stories:
            ks=sorted(k for g,k in v if g==story and k in method and np.isfinite(method[k]));pairs=[];ns=[];unmasked=[]
            for r,k in enumerate(ks):
                for t in ks[r+1:]:
                    a=np.asarray(v[story,k]);b=np.asarray(v[story,t]);signed=a.mean()-b.mean();eta=method[k]-method[t]
                    if abs(signed)>1e-10:pairs.append(float(truth((0. if abs(eta)<=1e-10 else eta)*signed)))
                    if name.endswith('_choice'):
                        q=truth(a[:,None]-b[None,:]).mean();ns.append(np.logaddexp(0.,eta)-q*eta)
            if pairs:bydoc[story]=val(pairs)
            if ns:loss.append(val(ns))
        score=val(list(bydoc.values()));external[name]=score;target=(conditional['ordering'][name] if name in conditional['ordering'] else complete['arms']['expert'][name])['pair_order']['estimate'];verification[name]=abs(score-target)
        if loss: verification[name+'.ballot_loss']=abs(val(loss)-conditional['pair_grade_prob_logloss'][name]['estimate'])
    return verification


def columbia():
    orders=read(ORDER/'observed-orders.json')['orders'];base=read(ORDER/'scores.json');learned=read(ORDER/'crossfit-scores.json');methods={};bounds={}
    with np.load(ORDER/'forecasts.npz',allow_pickle=False) as f:
        for n in ['adaptive.centered','augmented.centered','mean_surprisal.centered']:
            methods[n]=dict(zip(f['keys'],f[n],strict=True))
    with np.load(ORDER/'crossfit-predictions.npz',allow_pickle=False) as f:
        groups=sorted({k[:-5] for k in f.files if k.endswith('.keys')})
        for n in ['surface_and_probability','frozen_readouts','native_choices','native_three_observations']:
            methods[n]={k:val for g in groups for k,val in zip(f[g+'.keys'],f[n+'.'+g],strict=True)}
    rows=defaultdict(lambda:defaultdict(list));loss=defaultdict(lambda:defaultdict(list))
    for event,label in sorted(orders.items()):
        if event in base['omitted']:continue
        ref=int(hashlib.sha256(event.encode()).hexdigest()[:16],16)%len(label)
        if ref!=base['reference_row_0based'][event]:raise ValueError('A different human served as reference')
        for r,order in enumerate(label):
            if r==ref:continue
            for ix in range(1,len(order)-1):
                other=sorted(set(order[ix:])-{order[ix]});winner=order[ix];prev=order[ix-1]
                for n,vals in methods.items():
                    s=vals[f'{event}/{prev}/{winner}'];v=np.array([vals[f'{event}/{prev}/{k}'] for k in other]);d=s-v;d[np.abs(d)<1e-10]=0
                    rows[n][event,r].append(float(truth(d).mean()))
                    if n in learned['scores']:
                        loss[n][event,r].append(np.logaddexp.reduce(np.r_[0.,v-s])/np.log(1+len(v)))
    errors={}
    def macro(items):
        g=defaultdict(list)
        for (event,person),r in items.items():g[event].append(val(r))
        return val([val(r) for r in g.values()])
    for n,ds in rows.items():
        out=learned['scores'].get(n,base['methods'].get(n,{}));errors[n]=abs(macro(ds)-out['pairwise_choice']['estimate'])
        if n in loss:errors[n+'.normalized_logloss']=abs(macro(loss[n])-out['normalized_logloss']['estimate'])
    return errors


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('study',choices=['summeval','columbia']);arg=p.parse_args();rec=news() if arg.study=='summeval' else columbia()
    # Counts and sensitivity endpoints have their own archived scripts. This independent replay checks point estimates.
    if not rec or max(rec.values())>5e-12:raise ArithmeticError(f'Independent replay disagrees: {rec}')
    dest=NEWS if arg.study=='summeval' else ORDER;out={'replicated_terms':len(rec),'largest_absolute_difference':max(rec.values()),'absolute_differences':rec,'code':sha(Path(__file__))};(dest/'public-replay.json').write_text(json.dumps(out,indent=2)+'\n');print(out['replicated_terms'],out['largest_absolute_difference'])

if __name__=='__main__':main()
