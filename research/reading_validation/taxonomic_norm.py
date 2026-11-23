"""Learn whether one noun is a type of another from original 0–6 human votes."""

from collections import deque
from functools import lru_cache
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import spearmanr
from wordfreq import zipf_frequency

from research.reading_validation.memory_data import BASE,sha
from research.reading_validation.referent_features import senses,setup

OUT=Path('research/results/reading-validation-taxonomic-premise')
STUDY=BASE/'taxonomy/hyperlex-data.zip'
SIGNATURE='27ef061e2f711d8e8a53b9fe204eaf06c596ef53d95bbed81ae52c6eebe5713b'
PLAN=Path(__file__).with_name('taxonomic-premise-protocol.md')


@lru_cache(None)
def ancestors(sense):
    queue=deque([(sense,0)]);seen={}
    while queue:
        current,d=queue.popleft()
        if current not in seen:
            seen[current]=d
            queue.extend((p,d+1) for p in (*current.hypernyms(),*current.instance_hypernyms()))
    return seen


@lru_cache(None)
def paths(a,b):
    aa,bb=ancestors(a),ancestors(b)
    common=aa.keys() & bb.keys()
    da,db=a.max_depth(),b.max_depth()
    close=max((z.max_depth() for z in common),default=0)
    shared=(2.*(close+1))/(da+db+2)
    distance=min((aa[z]+bb[z] for z in common),default=40)
    return [shared,1./(1+distance),close/20.,float(a==b),
            float(b in aa),1./(1+aa[b]) if b in aa else 0.,float(a in bb),1./(1+bb[a]) if a in bb else 0.]


@lru_cache(100000)
def metrics(a,b):
    left,right=senses(a),senses(b)
    overlap=[];weights=[]
    for i,x in enumerate(left):
        for j,y in enumerate(right):
            overlap.append(paths(x,y));weights.append(1/((i+1)*(j+1)))
    if overlap:
        z=np.array(overlap);mx=z.max(0);av=np.average(z,weights=weights,axis=0)
    else:
        mx=av=np.zeros(8)
    dl=np.mean([s.max_depth() for s in left]) if left else 0.
    dr=np.mean([s.max_depth() for s in right]) if right else 0.
    fl,fr=zipf_frequency(a,'en'),zipf_frequency(b,'en')
    sym=np.r_[a==b,bool(left and right),mx[:4],av[:4],abs(dl-dr),min(fl,fr),max(fl,fr)]
    directional=np.r_[mx[4:],av[4:],dl-dr,fl-fr,len(left)-len(right)]
    return sym,directional


def matrix(pairs,arm):
    features=[metrics(a,b) for a,b in pairs]
    return np.stack([s if arm=='symmetric' else np.r_[s,d] for s,d in features])


def human(subset):
    if sha(STUDY)!=SIGNATURE:raise ValueError('Incorrect HyperLex author archive')
    with ZipFile(STUDY) as book:
        text=book.read('splits/lexical/hyperlex_'+subset+'_all_lexical.txt').decode().splitlines()
    pairs=[];counts=[]
    for row in text[1:]:
        a,b,pos,_,average,*other=row.split()
        if pos!='N':continue
        votes=list(map(int,other[2:]));v=np.bincount(votes,minlength=7)
        if len(v)!=7 or abs(np.mean(votes)-float(average))>.006:
            raise ValueError('Individual votes do not reconstruct the original rating')
        pairs.append((a,b));counts.append(v)
    return pairs,np.stack(counts)


def probability(x,v):
    cut=np.r_[v[x.shape[1]],v[x.shape[1]]+np.cumsum(np.logaddexp(0,v[x.shape[1]+1:]))]
    c=expit(cut[None,:]-(x@v[:x.shape[1]])[:,None]);p=np.diff(np.pad(c,((0,0),(1,0)),constant_values=0),axis=1)
    return c,np.column_stack((p,1-c[:,-1])).clip(1e-15,1)


def fit(x,y):
    mean=x.mean(0);sd=x.std(0);sd[sd<1e-7]=1;z=(x-mean)/sd;d=z.shape[1];w=y/y.sum(1,keepdims=True)/len(y)
    def objective(v):
        c,p=probability(z,v);ratio=w/p
        gcut=(ratio[:,1:]-ratio[:,:-1])*c*(1-c)
        loss=-np.sum(w*np.log(p))+.01*(v[:d]@v[:d]);g=-z.T@gcut.sum(1)+.02*v[:d]
        csum=gcut.sum(0);h=np.r_[csum.sum(),np.cumsum(csum[::-1])[::-1][1:]*expit(v[d+1:])]
        return loss,np.r_[g,h]
    initial=np.zeros(d+6);initial[d]=-2.
    opt=minimize(objective,initial,jac=True,method='L-BFGS-B',options={'ftol':1e-14,'gtol':1e-8,'maxiter':800})
    if not opt.success or np.max(abs(opt.jac))>1e-5:raise ArithmeticError(opt.message)
    return dict(mean=mean.tolist(),sd=sd.tolist(),v=opt.x.tolist(),gradient=float(max(abs(opt.jac))))


def predictions(spec,x):
    z=(x-np.asarray(spec['mean']))/np.asarray(spec['sd'])
    return probability(z,np.asarray(spec['v']))[1]


def run():
    setup();OUT.mkdir(parents=True,exist_ok=True);source,votes=human('training');fits={}
    for arm in ('symmetric','directional'):fits[arm]=fit(matrix(source,arm),votes)
    dest=OUT/'lexical-source.json';dest.write_text(json.dumps(dict(plan=sha(PLAN),archive=SIGNATURE,training_pairs=len(source),
         individual_votes=int(votes.sum()),models=fits),indent=2,allow_nan=False)+'\n')
    held,observed=human('test');ref=observed@np.arange(7)/observed.sum(1)
    overlap={w for pair in source for w in pair}&{w for pair in held for w in pair}
    if overlap:raise ValueError('Word-disjoint original split is not disjoint')
    methods={};losses={};ordering={}
    for arm,spec in fits.items():
        p=predictions(spec,matrix(held,arm));losses[arm]=-(observed*np.log(p)).sum(1)/observed.sum(1)
        ordering[arm]=p@np.arange(7)
        methods[arm]=dict(nll=float(losses[arm].mean()),rho=float(spearmanr(ref,ordering[arm]).statistic),
                          mean_absolute_error=float(np.abs(ref-ordering[arm]).mean()))
    rng=np.random.default_rng(86901);d=losses['directional']-losses['symmetric'];boot=rng.choice(d,(6000,len(d))).mean(1)
    report=dict(fit=sha(dest),held_noun_pairs=len(held),votes=int(observed.sum()),overlap_words=len(overlap),
                measures=methods,paired_pair_bootstrap=dict(estimate=float(d.mean()),interval=np.quantile(boot,[.025,.975]).tolist()))
    (OUT/'lexical-test.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(report))


if __name__=='__main__':run()
