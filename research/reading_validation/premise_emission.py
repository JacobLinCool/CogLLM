"""Marginalize genuine first-report hazards over the inferred state of the current premise."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.special import expit, logsumexp, softmax

from research.reading_validation.memory_data import BASE,E1,load_rows,sha
from research.reading_validation.memory_fit import counts
from research.reading_validation.memory_score import reveal,summarize
from research.reading_validation.referent_features import setup
from research.reading_validation.referent_monitoring import optimize
from research.reading_validation.taxonomic_monitor import Route,category_features,weighted_scale
from research.reading_validation.taxonomic_norm import OUT as TAX
from research.reading_validation.taxonomic_pairs import Ontology,form,native_words,opportunities,source_links

OUT=Path('research/results/reading-validation-premise-emission')
PLAN=Path(__file__).with_name('premise-emission-protocol.md')
STATES=('ordinary','consistent','neutral','inconsistent')
ARMS=('location','symmetric','directional','native','oracle')


def design(j,sentence):
    w=len(sentence)
    return np.array([np.log1p(j),np.log1p(w),j/(j+5),float(w<=8)])


def mix(values,p):
    mean=p@values
    return np.r_[mean, np.sqrt(np.maximum(0.,p@((values-mean)**2)))]


def measurements(route,arm,key,j,doc,bags):
    b=bags[key][j];base=design(j,doc[key][j])
    role=route.target(j+1,len(doc[key][j]));base=np.r_[base,role,np.log(max(role,1e-8)/max(1-role,1e-8))]
    if arm=='location':return base
    cols={'symmetric':5,'directional':11,'native':25}[arm]
    if not len(b.rank): return np.r_[base,np.zeros(6+2*cols)]
    x=(b.rank-route.mu)/route.sd;prior=route.offset[b.origin]
    lp=x@route.v+prior;p=softmax(lp)
    def concentration(coord):return np.bincount(coord,weights=p).max()
    details=[logsumexp(lp)-logsumexp(prior),p.max(),np.sum(p*p),concentration(b.origin),concentration(b.word),np.log1p(len(p))]
    return np.r_[base,details,mix(category_features(b,arm),p)]


def fit_state(rows,features,arm):
    # Binary target presence is estimated on every original opportunity.
    by=Counter(r['study'] for r in rows);xx=[];event=[];weights=[]
    for r in rows:
        k=r['id'];matrix=features[k][1:];ix=np.arange(2,len(matrix)+2)
        xx.extend(matrix);event.extend(ix==r['target'])
        weights.extend([1/(len(by)*by[r['study']]*len(ix))]*len(ix))
    X=np.array(xx);w=np.array(weights);mu,sd=weighted_scale(X,w);x=np.column_stack((np.ones(len(X)),(X-mu)/sd));y=np.array(event)
    def presence(v):
        e=x@v;err=w*(expit(e)-y);g=x.T@err;g[1:]+=.02*v[1:]
        return w@(np.logaddexp(0,e)-y*e)+.01*(v[1:]@v[1:]),g
    gate,eg=optimize(presence,np.zeros(x.shape[1]))
    # Condition discrimination uses only the simultaneous alternative interpretations of a target.
    # Fit a two-class conditional model for neutral versus inconsistent on E2. E1 observed true-repeat
    # material supplies the independent pair selector but its easier repeat contrast is a different task.
    pairs=defaultdict(dict)
    for r in rows:
        if r['study']==2:pairs[r['story']][r['arm']]=features[r['id']][r['target']-1]
    x0=np.stack([v['neutral'] for v in pairs.values()]);x1=np.stack([v['inconsistent'] for v in pairs.values()])
    xx=np.vstack((x0,x1));m=xx.mean(0);sig=xx.std(0);sig[sig<1e-7]=1;z=(xx-m)/sig
    y=np.repeat([0.,1.],len(pairs));difference=(x1-x0)/sig;n=len(z)
    def direction(v):
        score=z@v[1:]+v[0];d=difference@v[1:];e=(expit(score)-y)/n
        loss=.5*np.mean(np.logaddexp(0,score)-y*score)+.5*np.mean(np.logaddexp(0,-d))+.01*(v[1:]@v[1:])
        g=.5*(z.T@e+difference.T@(expit(d)-1)/len(d))+.02*v[1:]
        return loss,np.r_[.5*e.sum(),g]
    cl,ec=optimize(direction,np.zeros(z.shape[1]+1))
    def state(features):
        p=expit((features-mu)/sd@gate[1:]+gate[0])
        q=expit((features-m)/sig@cl[1:]+cl[0]) if arm!='location' else np.full(len(features),.5)
        return np.column_stack((1-p,p*(1-q),p*q))
    return state,dict(gate_gradient=eg,condition_gradient=ec)


def emissions(rows,observations,docs):
    # The neutral/incompatible experiment defines the deployment population.
    # Estimate background at each position using training stories' at-risk responses.
    group=defaultdict(lambda:np.array([0.,0.]));ordinary=[]
    for r in rows:
        if r['study']!=2:continue
        k=r['id'];at,events,_=observations[2,k]
        for j in range(1,len(at)):
            if not at[j]:continue
            if j==r['target']-1:group[r['arm']]+=[events[j],at[j]]
            else:ordinary.append((design(j,docs[k][j]),events[j],at[j]))
    x=np.stack([r[0] for r in ordinary]);num=np.array([r[1] for r in ordinary]);risk=np.array([r[2] for r in ordinary])
    w=risk/risk.sum();mu,sd=weighted_scale(x,w);z=np.column_stack((np.ones(len(x)),(x-mu)/sd));y=num/risk
    def likelihood(v):
        e=z@v;g=z.T@(w*(expit(e)-y));g[1:]+=.01*v[1:]
        return w@(np.logaddexp(0,e)-y*e)+.005*(v[1:]@v[1:]),g
    v,err=optimize(likelihood,np.zeros(z.shape[1]));rates={a:(c[0]+.5)/(c[1]+1.) for a,c in group.items()}
    def hazards(sentences):
        raw=np.stack([design(j,s) for j,s in enumerate(sentences)])
        h0=expit((raw-mu)/sd@v[1:]+v[0])
        return np.column_stack((h0,np.full(len(h0),rates['neutral']),np.full(len(h0),rates['inconsistent'])))
    return hazards,{**rates,'ordinary_gradient':err}


def run(archive):
    setup();OUT.mkdir(parents=True,exist_ok=True);out=OUT/'forecast';out.mkdir(exist_ok=False)
    meta=json.loads((BASE/'memory/study-map.json').read_bytes())
    raw={r['id']:r['sentences'] for r in load_rows(BASE/'memory/stories.jsonl')};doc={k:list(map(form,s)) for k,s in raw.items()}
    native=native_words(archive,raw);ontology=Ontology();bags={k:opportunities(v,native[k],ontology) for k,v in raw.items()}
    training=[r for r in meta if r['study'] in (1,2)];links=source_links(training,raw)
    obs={(2,k):v for k,v in counts(reveal(2,meta),[r for r in training if r['study']==2]).items()}
    families=sorted({r['story'] for r in meta if r['study']==3});models={};all_preds={};diag={}
    def router(exclusions):
        key=tuple(sorted(exclusions))
        if key not in models:models[key]=Route([r for r in training if r['story'] not in exclusions],doc,bags,links)
        return models[key]
    for held in families:
        source=[r for r in training if r['story']!=held];test={r['id']:r for r in meta if r['study']==3 and r['story']==held}
        hazards,info=emissions(source,obs,doc);diag[held]=dict(human_emissions=info)
        for arm in ARMS:
            if arm=='oracle':
                for k,r in test.items():
                    state=np.zeros((len(doc[k]),3));state[:,0]=1;state[r['target']-1]=[0,int(r['arm']=='neutral'),int(r['arm']=='inconsistent')]
                    val=(state*hazards(doc[k])).sum(1);val[0]=0;all_preds['strict.oracle.'+k]=val
                continue
            def project(rows,retriever):
                return {r['id']:np.stack([measurements(retriever,arm,r['id'],j,doc,bags) for j in range(len(doc[r['id']]))]) for r in rows}
            cross={}
            for f in {r['story'] for r in source}:
                cross.update(project([r for r in source if r['story']==f],router([f,held])))
            state,diagnostics=fit_state(source,cross,arm);diag[held][arm]=diagnostics
            feats=project(test.values(),router([held]))
            for k,r in test.items():
                probs=state(feats[k]);val=(probs*hazards(doc[k])).sum(1);val[0]=0
                all_preds['strict.'+arm+'.'+k]=val
                all_preds['occupancy.'+arm+'.'+k]=probs
        print('Finished risk-preserving model for held family',held,flush=True)
    with np.load(TAX/'story-forecast/forecasts.npz') as previous:
        for k in raw:
            if 'strict.original.'+k in previous:all_preds['strict.original.'+k]=previous['strict.original.'+k]
    np.savez_compressed(out/'hazards.npz',**all_preds);(out/'fits.json').write_text(json.dumps(diag,indent=2)+'\n')
    (out/'seal.json').write_text(json.dumps(dict(protocol=sha(PLAN),archive=sha(archive/'receipt.json'),
                  files={f:sha(out/f) for f in ('fits.json','hazards.npz')},cohort3_responses_used=False),indent=2)+'\n')


def score():
    out=OUT/'forecast';seal=json.loads((out/'seal.json').read_bytes())
    if any(sha(out/f)!=v for f,v in seal['files'].items()):raise ValueError('Risk forecasts changed')
    rows=json.loads((BASE/'memory/study-map.json').read_bytes())
    with np.load(out/'hazards.npz') as f:
        scores={str(s):summarize(f,reveal(s,rows),373+s,cohorts=('strict',),methods=('original',*ARMS),focus='native',
                    controls=('location','symmetric','directional','original','oracle')) for s in (2,3)}
    (OUT/'scores.json').write_text(json.dumps(dict(forecast=sha(out/'seal.json'),cohorts=scores),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('fit','score'));p.add_argument('archive',type=Path,nargs='?')
    a=p.parse_args();run(a.archive) if a.action=='fit' else score()
