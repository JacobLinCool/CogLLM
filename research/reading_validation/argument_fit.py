"""Fit article-balanced measurement likelihoods and seal forecasts on both entire essay panels."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from research.reading_validation.argument_data import OUT
from research.reading_validation.argument_features import Corpus
from research.reading_validation.argument_labels import bank
from research.reading_validation.memory_data import sha

EST=Path(__file__).with_name('argument-estimator.md')


def normalize(x,w):
    mu=w@x;sd=np.sqrt(w@((x-mu)**2));sd[sd<1e-7]=1
    return (x-mu)/sd,mu,sd


def optimize(obj,size):
    opt=minimize(obj,np.zeros(size),jac=True,method='L-BFGS-B',
                 options={'maxiter':650,'ftol':1e-13,'gtol':1e-7,'maxcor':16})
    err=float(np.max(abs(opt.jac)))
    if not opt.success or err>3e-5:raise ArithmeticError(f'Argument source likelihood did not converge: {opt.message}, {err}')
    return opt.x,dict(iterations=int(opt.nit),maximum_gradient=err,objective=float(opt.fun))


def ordinal(z,v):
    a=v[-2]-z@v[:-2];b=a+np.logaddexp(0,v[-1])+.0001
    u=expit(a);v2=expit(b)
    return np.column_stack((u,v2-u,1-v2)),u*(1-u),v2*(1-v2)


def position_model(rows,kind):
    x=np.stack([t[1] for t in rows]);y=np.array([t[2] for t in rows],int)
    counts=Counter(t[0] for t in rows);w=np.array([1/(len(counts)*counts[t[0]]) for t in rows])
    z,mu,sd=normalize(x,w);reg=.01
    if kind=='event':
        def obj(v):
            e=z@v[:-1]+v[-1];g=w*(expit(e)-y)
            return w@(np.logaddexp(0,e)-y*e)+.5*reg*(v[:-1]@v[:-1]),np.r_[z.T@g+reg*v[:-1],g.sum()]
        size=z.shape[1]+1
    else:
        idx=np.arange(len(y))
        def obj(v):
            p,da,db=ordinal(z,v);py=p[idx,y].clip(1e-14)
            a=w*np.where(y==0,-da,np.where(y==1,da,0.))/py
            b=w*np.where(y==1,-db,np.where(y==2,db,0.))/py
            g=a+b
            return -w@np.log(py)+.5*reg*(v[:-2]@v[:-2]),np.r_[-z.T@g+reg*v[:-2],g.sum(),b.sum()*expit(v[-1])]
        size=z.shape[1]+2
    v,diag=optimize(obj,size)
    return dict(mu=mu,sd=sd,v=v,diag=diag)


def choice_model(rows):
    lengths=np.array([len(t[1]) for t in rows]);starts=np.r_[0,lengths.cumsum()[:-1]]
    groups=np.repeat(np.arange(len(rows)),lengths)
    counts=Counter(t[0] for t in rows);w=np.array([1/(len(counts)*counts[t[0]]) for t in rows]);gold=np.concatenate([t[2] for t in rows])
    x=np.vstack([t[1] for t in rows]);z,mu,sd=normalize(x,(w/lengths)[groups]);reg=.01
    def obj(v):
        logits=z@v
        def partition(score):
            peak=np.maximum.reduceat(score,starts);ez=np.exp(score-peak[groups])
            total=np.add.reduceat(ez,starts)
            return peak+np.log(total),ez/total[groups]
        total,p=partition(logits);part,q=partition(np.where(gold,logits,-np.inf))
        return w@(total-part)+.5*reg*(v@v),z.T@(w[groups]*(p-q))+reg*v
    v,diag=optimize(obj,z.shape[1])
    return dict(mu=mu,sd=sd,v=v,diag=diag)


def train(corpus,annotations,arm):
    rows={n:[] for n in ('choice','event','grade')}
    for k,r in annotations.items():
        if not corpus.eligible(k):continue
        pos,pairs=corpus.design(k)
        for j,(x,p) in enumerate(zip(pos,pairs,strict=True)):
            if j:
                truth=r['links'].get(j,[])
                if truth:rows['choice'].append((k,p[arm],np.isin(np.arange(j),truth)))
                if r['complete'] and j not in r['exclude']:rows['event'].append((k,x[arm],int(bool(truth))))
            for y in r['grades'].get(j,[]):rows['grade'].append((k,x[arm],y))
    fitted={}
    for name,rr in rows.items():
        model=choice_model(rr) if name=='choice' else position_model(rr,name)
        model['diag'].update(observations=len(rr),essays=len({t[0] for t in rr}));fitted[name]=model
        print(f'{arm} {name} {model["diag"]}',flush=True)
    return fitted


def predictions(pos,pair,model,arm):
    x=np.stack([r[arm] for r in pos]);output={}
    for name in ('event','grade'):
        spec=model[name];z=(x-spec['mu'])/spec['sd'];v=spec['v']
        prob=expit(z@v[:-1]+v[-1]) if name=='event' else ordinal(z,v)[0]
        if not np.isfinite(prob).all():raise ArithmeticError('Invalid essay probability')
        output[name]=prob
    spec=model['choice'];source=[]
    for j,p in enumerate(pair):
        if j:
            logits=((p[arm]-spec['mu'])/spec['sd'])@spec['v'];value=np.exp(logits-logits.max());source.extend(value/value.sum())
    output['choice']=np.array(source)
    return output


def run(archive,mode):
    if mode=='native' and archive is None:raise ValueError('Native arm requires original L4 features')
    out=OUT/('forecast-'+mode);out.mkdir(exist_ok=False)
    corpus=Corpus(archive);annotations=bank('train');arms=('form','native') if mode=='native' else ('form',)
    models={arm:train(corpus,annotations,arm) for arm in arms};arrays={};keys={}
    for k,meta in corpus.meta['articles'].items():
        if meta['role'] in ('held','external','dev') and corpus.eligible(k):
            pos,links=corpus.design(k);keys[k]=len(pos)
            for arm,m in models.items():
                for name,v in predictions(pos,links,m,arm).items():arrays[k+'.'+arm+'.'+name]=v.astype('float32')
    np.savez_compressed(out/'forecasts.npz',**arrays)
    def encode(o):
        if isinstance(o,np.ndarray):return o.tolist()
        raise TypeError(type(o).__name__)
    (out/'model.json').write_text(json.dumps(models,default=encode,allow_nan=False)+'\n')
    receipt=dict(keys=keys,arms=arms,source=sha(OUT/'inputs.json'),estimator=sha(EST),
        archive=sha(archive/'receipt.json') if archive else None,files={n:sha(out/n) for n in ('model.json','forecasts.npz')})
    (out/'seal.json').write_text(json.dumps(receipt,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',type=Path);p.add_argument('--mode',choices=('form','native'),required=True)
    run(**vars(p.parse_args()))
