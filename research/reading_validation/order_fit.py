"""Fit study-unit and reader-conditioned measurement likelihoods on the source-only order cohort."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from research.reading_validation.memory_data import sha
from research.reading_validation.order_data import OUT
from research.reading_validation.order_measure import ARMS,TASKS,Study,features


def weights(rows):
    counts=Counter((g,c) for g,c,_,_ in rows);by=Counter(g for g,c in counts)
    return np.array([1/(len(by)*by[g]*counts[g,c]) for g,c,_,_ in rows])


def standardize(x,w):
    mu=w@x;sd=np.sqrt(w@((x-mu)**2));sd[sd<1e-9]=1.;return mu,sd


def matrix(rows,arm,study,design,task):
    x=np.stack([design[task,(g,c)][arm] for g,c,_,_ in rows]);r=np.array([study.readers[p] for g,c,p,y in rows])
    return x,r


def augment(z,profile,arm):
    if arm=='profile':z=np.column_stack((z,profile,z*profile[:,0,None],z*profile[:,1,None]))
    return np.column_stack((np.ones(len(z)),z))


def fit(rows,task,arm,study,design):
    x,r=matrix(rows,arm,study,design,task);y=np.array([t[3] for t in rows]);w=weights(rows)
    mu,sd=standardize(x,w);rm,rs=standardize(r,w);z=augment((x-mu)/sd,(r-rm)/rs,arm)
    if task=='question':
        def objective(v):
            e=z@v;p=expit(e);g=z.T@(w*(p-y));g[1:]+=.02*v[1:]
            return w@(np.logaddexp(0,e)-y*e)+.01*(v[1:]@v[1:]),g
        opt=minimize(objective,np.zeros(z.shape[1]),jac=True,method='L-BFGS-B',options={'gtol':1e-8,'ftol':1e-14,'maxiter':500})
        v=opt.x;err=float(max(abs(opt.jac)));sigma,smear=1.,1.
        if not opt.success or err>1e-5:raise ArithmeticError('Source correctness likelihood did not converge')
    else:
        penalty=np.diag([0.,*[.02]*(z.shape[1]-1)]);gram=z.T@(w[:,None]*z)+penalty
        target=z.T@(w*y);v=np.linalg.solve(gram,target);err=float(np.max(abs(gram@v-target)))
        residual=y-z@v;sigma=float(np.sqrt(w@(residual**2)));smear=float(w@np.exp(residual))
    return dict(mean=mu.tolist(),scale=sd.tolist(),reader_mean=rm.tolist(),reader_scale=rs.tolist(),coef=v.tolist(),
                residual_sd=sigma,source_smearing=smear,gradient=err)


def predict(spec,x,r,arm,task):
    z=augment((x-spec['mean'])/spec['scale'],(r-spec['reader_mean'])/spec['reader_scale'],arm)
    f=z@spec['coef'];return expit(f) if task=='question' else f


def identity(row):return '|'.join(row[:3])


def run(archive):
    out=OUT/'forecast';out.mkdir(exist_ok=False);study=Study();design=features(archive,study);fits={};forecasts={}
    for task in TASKS:
        source=study.panel(task,'train');test=study.panel(task,'held_material')
        if any(r[0] in {t[0] for t in source} for r in test) or len({identity(r) for r in test})!=len(test):
            raise ValueError('Overlapping materials or an unidentified repeated reader encounter')
        decisions={};model={}
        for arm in ARMS:
            spec=fit(source,task,arm,study,design);x,r=matrix(test,arm,study,design,task);y=predict(spec,x,r,arm,task)
            if not np.isfinite(y).all():raise ArithmeticError('Invalid held article forecast')
            decisions[arm]=y.tolist();model[arm]=spec
        fits[task]=model;forecasts[task]=dict(keys=[identity(r) for r in test],predictions=decisions)
    for name,record in (('fits.json',fits),('predictions.json',forecasts)):
        (out/name).write_text(json.dumps(record,allow_nan=False)+'\n')
    stamp=dict(native=sha(archive/'receipt.json'),input=sha(OUT/'inputs.json'),estimator=sha(Path(__file__).with_name('order-estimator.md')),
               files={name:sha(out/name) for name in ('fits.json','predictions.json')})
    (out/'seal.json').write_text(json.dumps(stamp,indent=2)+'\n')
    print('Sealed original sentence, second-clause and question decisions before reading model grades')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('archive',type=Path);run(**vars(p.parse_args()))
