"""Fit family-blocked word links and premise compatibility before evaluating the reader's first report."""

import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.special import expit,logsumexp,softmax

from research.reading_validation.memory_data import BASE,E1,OUT as ORIGINAL,load_rows,sha
from research.reading_validation.memory_fit import counts
from research.reading_validation.memory_score import reveal
from research.reading_validation.referent_features import setup
from research.reading_validation.referent_monitoring import calibrate,location,optimize
from research.reading_validation.taxonomic_norm import OUT,PLAN
from research.reading_validation.taxonomic_pairs import Ontology,form,native_words,opportunities,source_links

ARMS=('symmetric','directional','native')


def weighted_scale(x,w):
    mu=w@x;sd=np.sqrt(w@((x-mu)**2));sd[sd<1e-6]=1
    return mu,sd


def normalized(x,mu,sd):
    return np.column_stack((np.ones(len(x)),(x-mu)/sd))


def category_features(bag,arm):
    if arm=='symmetric':return bag.symmetric
    if arm=='directional':return np.column_stack((bag.symmetric,bag.directional))
    return np.column_stack((bag.symmetric,bag.directional,bag.neural))


class Route:
    def __init__(self,rows,docs,bags,links):
        self.target=location(rows,docs)
        annotated=[r for r in rows if r['id'] in links]
        bank=[bags[r['id']][r['target']-1] for r in annotated]
        positive=[b.match(links[r['id']]) for r,b in zip(annotated,bank,strict=True)]
        if any(not g.any() for g in positive):raise ValueError('No causal candidate can represent a source link')
        n=len(bank);self.source=Counter(links[r['id']][0] for r in annotated)
        self.offset=np.log(np.array([.5+self.source[j] for j in range(max(map(len,docs.values())))]))
        xx=np.vstack([b.rank for b in bank]).astype(float)
        w=np.concatenate([np.full(len(b.rank),1/(n*len(b.rank))) for b in bank]);self.mu,self.sd=weighted_scale(xx,w)
        matrices=[(b.rank-self.mu)/self.sd for b in bank];base=[self.offset[b.origin] for b in bank]
        def objective(v):
            loss=0.;g=np.zeros_like(v)
            for z,prior,pos in zip(matrices,base,positive,strict=True):
                score=z@v+prior;sub=score[pos]
                loss+=logsumexp(score)-logsumexp(sub);g+=softmax(score)@z-softmax(sub)@z[pos]
            return loss/n+.01*(v@v),g/n+.02*v
        self.v,self.residual=optimize(objective,np.zeros(len(self.mu)))
        self.models={}
        by_study=Counter(r['study'] for r in annotated)
        weight=np.array([1/(len(by_study)*by_study[r['study']]) for r in annotated])
        y=np.array([r['arm']=='inconsistent' for r in annotated],dtype=float)
        for arm in ARMS:
            z=np.stack([category_features(b,arm)[p].mean(0) for b,p in zip(bank,positive,strict=True)])
            mu,sd=weighted_scale(z,weight);x=normalized(z,mu,sd)
            def objective(v):
                score=x@v;q=expit(score)
                g=x.T@(weight*(q-y));g[1:]+=.02*v[1:]
                return weight@(np.logaddexp(0,score)-y*score)+.01*(v[1:]@v[1:]),g
            v,err=optimize(objective,np.zeros(x.shape[1]));self.models[arm]=(mu,sd,v,err)
        self.cache={}

    def weights(self,b):
        return softmax((b.rank-self.mu)/self.sd@self.v+self.offset[b.origin])

    def forecast(self,key,docs,bags):
        if key in self.cache:return self.cache[key]
        scores=defaultdict(list)
        for j,b in enumerate(bags[key]):
            target=self.target(j+1,len(docs[key][j]));scores['role'].append([target]);p=self.weights(b) if len(b.rank) else None
            for arm,(mu,sd,v,_) in self.models.items():
                x=category_features(b,arm)
                conditional=float(p@expit(normalized(x,mu,sd)@v)) if p is not None else .5
                scores[arm].append([target,target*conditional])
        self.cache[key]={m:np.array(z) for m,z in scores.items()};return self.cache[key]


def run(archive):
    setup();rows=json.loads((BASE/'memory/study-map.json').read_bytes())
    doc={r['id']:r['sentences'] for r in load_rows(BASE/'memory/stories.jsonl')}
    words={k:list(map(form,v)) for k,v in doc.items()}
    native=native_words(archive,doc);ontology=Ontology();bags={k:opportunities(v,native[k],ontology) for k,v in doc.items()}
    source=[r for r in rows if r['study'] in (1,2)];links=source_links(source,doc)
    outcomes={}
    for s,observations in ((1,load_rows(E1/'outcomes.jsonl')),(2,reveal(2,rows))):
        outcomes.update({(s,k):v for k,v in counts(observations,[r for r in source if r['study']==s]).items()})
    families=sorted({r['story'] for r in source});trained={};prediction={};info={};link_diagnostics={}
    def get(excluded):
        excluded=tuple(sorted(set(excluded)))
        if excluded not in trained:
            learning=[r for r in source if r['story'] not in excluded]
            trained[excluded]=Route(learning,words,bags,links)
        return trained[excluded]
    out=OUT/'story-forecast';out.mkdir(exist_ok=False);arms=['role',*ARMS]
    with np.load(ORIGINAL/'forecast/forecasts.npz') as old:
        for k in doc:prediction['strict.original.'+k]=old['strict.frozen.'+k]
    for held in sorted({r['story'] for r in rows if r['study']==3}):
        learning=[r for r in source if r['story']!=held];folds={}
        for other in families:
            if held==other:continue
            model=get([held,other]);keys=[r['id'] for r in learning if r['story']==other]
            folds.update({k:model.forecast(k,words,bags) for k in keys})
        individual=get([held]);tests=[r for r in rows if r['study']>=2 and r['story']==held]
        for m in arms:
            z={k:s[m] for k,s in folds.items()};mu,sd,v,err=calibrate(learning,z,outcomes)
            info[f'{held}.{m}']=dict(gradient=err,source_families=len(families)-1,coefficients=v.tolist())
            for r in tests:
                k=r['id'];hz=expit((individual.forecast(k,words,bags)[m]-mu)/sd@v[1:]+v[0]);hz[0]=0
                prediction['strict.'+m+'.'+k]=hz
        for r in tests:
            k=r['id']
            if k not in links or k in link_diagnostics:continue
            b=bags[k][r['target']-1];p=individual.weights(b);s,_,i=links[k]
            mask=b.match(links[k]);best=p==p.max()
            link_diagnostics[k]=dict(family=held,arm=r['arm'],candidates=len(p),joint_mass=float(p[mask].sum()),
                exact_top=float(mask[best].mean()),source_mass=float(p[b.origin==s].sum()),word_mass=float(p[b.word==i].sum()),
                nearest_prior_sentence_correct=int(s==r['target']-2))
        print('Completed held story',held,flush=True)
    np.savez_compressed(out/'forecasts.npz',**prediction)
    (out/'fits.json').write_text(json.dumps(dict(outer_fits=info,joint_source_diagnostic=link_diagnostics),indent=2)+'\n')
    (out/'seal.json').write_text(json.dumps(dict(protocol=sha(PLAN),lexical_fit=sha(OUT/'lexical-source.json'),
          native=sha(archive/'receipt.json'),no_experiment3_response_in_fit=True,source_link_articles=len(links),
          largest_training_gradient=max(max(z.residual,*[v[3] for v in z.models.values()]) for z in trained.values()),
          models=['original',*arms],files={n:sha(out/n) for n in ('fits.json','forecasts.npz')}),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('archive',type=Path);run(**vars(p.parse_args()))
