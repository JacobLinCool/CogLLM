"""Doubly held-out estimates with and without a reader's measured prereading familiarity."""

from hashlib import sha256
import json

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from research.reading_validation.local_features import OUT
from research.reading_validation.local_forecast import digest, rows
from research.reading_validation.local_jargon import JARGON, outcomes, two_way


ARMS=('source_distribution','surface','text','prior','text_with_prior')
RIDGE=.015


def estimate(train, y, x, classes):
    """Ordered log odds across rating categories with category-specific baseline frequencies."""
    levels=np.arange(classes,dtype=float)
    ht=np.eye(classes)[y[train]]
    a=x[train]
    center=a.mean(axis=0)
    scale=np.maximum(a.std(axis=0),1e-6)
    x=(x-center)/scale
    a=x[train]
    n,d=a.shape
    def obj(w):
        z=(a@w[:d])[:,None]*levels+np.r_[w[d:],0.]
        logp=z-logsumexp(z,axis=1,keepdims=True)
        diff=(np.exp(logp)-ht)/n
        grad=np.r_[a.T@(diff@levels)+RIDGE*w[:d],diff.sum(axis=0)[:-1]]
        return float(-(ht*logp).sum()/n+.5*RIDGE*np.sum(w[:d]**2)),grad
    mean=(ht.sum(axis=0)+.5)/(n+.5*classes)
    if d==0:
        return np.tile(mean,(len(x),1)),0.
    initial=np.r_[np.zeros(d),np.log(mean[:-1]/mean[-1])]
    fit=minimize(obj,initial,method='L-BFGS-B',jac=True,options={'ftol':1e-13,'gtol':3e-7,'maxiter':250})
    gradient=float(abs(fit.jac).max())
    if not fit.success or gradient>2e-5:
        raise ValueError(f'Rating calibration failed to converge: {fit.message}')
    logits=(x@fit.x[:d])[:,None]*levels+np.r_[fit.x[d:],0.]
    return np.exp(logits-logsumexp(logits,axis=1,keepdims=True)),gradient


def one_endpoint(records,field):
    data=[r for r in records if r[field] is not None and r['before'] is not None]
    papers=sorted({r['paper'] for r in data})
    readers=sorted({r['reader'] for r in data})
    ix=np.array([papers.index(r['paper']) for r in data])
    jx=np.array([readers.index(r['reader']) for r in data])
    group=np.array([int(sha256(p.encode()).hexdigest(),16)%5 for p in papers])[ix]
    n=len(data)
    cat=5 if field=='after' else 2
    # Familiarity was encoded as 6-rating for the external ranking. Restore the original survey response here.
    y=np.array([int(5-r[field] if cat==5 else r[field]) for r in data])
    prior=np.array([int(5-r['before']) for r in data])
    sources=np.array([[r['scores'][k] for k in ('form','external_lexical','local')] for r in data])
    onehot=np.eye(5)[prior][:,1:]
    features=(np.empty((n,0)),sources[:,:1],sources,onehot,np.column_stack((sources,onehot)))
    forecasts=np.full((n,len(ARMS),cat),np.nan)
    convergence=[]
    for reader in range(len(readers)):
        for f in range(5):
            train=(jx!=reader)&(group!=f)
            test=(jx==reader)&(group==f)
            if not test.any(): continue
            if set(ix[train])&set(ix[test]) or set(jx[train])&set(jx[test]):
                raise ValueError('A held-out person or article entered the training split')
            for m,x in enumerate(features):
                probs,g=estimate(train,y,x,cat)
                forecasts[test,m]=probs[test]
                convergence.append(g)
    if not np.isfinite(forecasts).all() or not np.allclose(forecasts.sum(-1),1):
        raise ValueError('Unfilled forecast cell')
    exp=forecasts@np.arange(cat)
    loss=[-np.log(forecasts[np.arange(n),:,y])]
    loss.append(np.abs(exp-y[:,None]) if cat==5 else (exp-y[:,None])**2)
    metric={}
    for name,values in zip(('nll_nats','rating_mae' if cat==5 else 'brier'),loss):
        sums=np.zeros((len(papers),len(readers),len(ARMS)))
        count=np.zeros((len(papers),len(readers)))
        np.add.at(sums,(ix,jx),values)
        np.add.at(count,(ix,jx),1)
        means=sums/np.maximum(count[:,:,None],1)
        point,boot,fixed=two_way(means,count>0)
        def summary(p,b,t):
            return dict(mean=float(p),crossed_95=np.quantile(b,[.025,.975]).tolist(),
                        conditional_on_readers_95=np.quantile(t,[.025,.975]).tolist())
        result={m:summary(point[k],boot[:,k],fixed[:,k]) for k,m in enumerate(ARMS)}
        for a,b in [('text','surface'),('text','source_distribution'),('prior','text'),('text_with_prior','prior')]:
            i,j=ARMS.index(a),ARMS.index(b)
            result[a+' minus '+b]=summary(point[i]-point[j],boot[:,i]-boot[:,j],fixed[:,i]-fixed[:,j])
        metric[name]=result
    return dict(rows=n,papers=len(papers),readers=len(readers),paper_folds=5,reader_folds=len(readers),
                max_abs_fit_gradient=max(convergence),metrics=metric),forecasts,[
                    sha256(json.dumps([r['paper'],r['reader'],r['term']]).encode()).hexdigest() for r in data]


def run():
    target=OUT/'jargon'
    lock=json.loads((target/'prediction-lock.json').read_bytes())
    forecast=target/'predictions.jsonl'
    if lock['files_sha256'][forecast.name]!=digest(forecast):
        raise ValueError('Expected frozen zero-shot predictions')
    preds={(r['paper'],r['term']):r['scores'] for r in rows(forecast)}
    records,duplicates=outcomes(JARGON/'source.csv',preds)
    result,arrays={},{}
    for field in ('after','definition','background'):
        result[field],arrays[field],keys=one_endpoint(records,field)
        arrays[field+'_keys']=np.array(keys)
    np.savez_compressed(target/'crossfit-forecasts.npz',**arrays)
    result.update(prediction_lock=digest(target/'prediction-lock.json'),ridge=RIDGE,
                  excluded_conflicting_duplicates=duplicates,forecasts_sha256=digest(target/'crossfit-forecasts.npz'),
                  protocol_sha256=digest(__import__('pathlib').Path(__file__).with_name('local-knowledge-protocol.md')))
    (target/'crossfit-scores.json').write_text(json.dumps(result,indent=2)+'\n')
    print({k:result[k]['metrics']['nll_nats']['text_with_prior minus prior'] for k in ('after','definition','background')})


if __name__=='__main__':
    run()
