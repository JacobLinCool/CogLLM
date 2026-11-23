"""Estimate borrowing on the calibration families and grade all eligible human endpoints."""

import argparse
from collections import defaultdict
import json

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
import torch

from research.backbone_transfer.extract import digest
from research.reading_validation.borrowing_data import OUTPUT, CALIBRATE, EVALUATE, data
from research.union_training.fit_observations import COUNT_MODELS, row_nll, weights


class GeometricPool:
    """Log likelihood of a normalized density pool with exact grouped Gaussian sufficient statistics."""

    def __init__(self,task,rows,shared,private):
        self.kind=task.likelihood
        self.y=task.response[rows].double().numpy()
        self.left,self.right=(np.asarray(x,dtype=np.float64) for x in (private,shared))
        if self.left.shape!=self.right.shape or len(self.left)!=len(rows):
            raise ValueError(f'Incompatible forecasts for {task.name}')
        if self.kind in COUNT_MODELS:
            self.valid=np.isfinite(self.left)
            if not np.array_equal(self.valid,np.isfinite(self.right)) or np.any((self.y>0)&~self.valid):
                raise ValueError('Changed event support between two predictive distributions')
            self.delta=np.where(self.valid,self.right,0.)-np.where(self.valid,self.left,0.)
        elif self.kind=='paired_gaussian':
            def natural(x):
                a,b,r=x[:,2],x[:,3],x[:,4]
                cov=np.stack([a*a,a*b*r,a*b*r,b*b],axis=-1).reshape(-1,2,2)
                precision=np.linalg.inv(cov)
                return precision,(precision@x[:,:2,None]).squeeze(2)
            self.p0,self.eta0=natural(self.left)
            self.p1,self.eta1=natural(self.right)
        else:
            sampling_var=self.y[:,1]**2 if self.kind=='known_error_gaussian' else 0.
            self.p0=1/(self.left[:,1]**2+sampling_var);self.p1=1/(self.right[:,1]**2+sampling_var)
            self.eta0=self.p0*self.left[:,0];self.eta1=self.p1*self.right[:,0]
        for a,raw in ((0.,private),(1.,shared)):
            expected=row_nll(task,torch.from_numpy(raw).double(),task.response[rows].double()).numpy()
            # Each expert stored float32 log probabilities. Bound renormalization error per observed trial.
            n=task.trials[rows]
            if not np.allclose(self.loss(a)/n,expected/n,rtol=3e-6,atol=2e-6):
                raise ArithmeticError(f'Distribution endpoint identity failed: {task.name}, {a}')

    def loss(self,a):
        if self.kind in COUNT_MODELS:
            logp=self.left+a*self.delta
            centered=logp-logsumexp(logp,axis=1,keepdims=True)
            return -(self.y*np.where(self.y>0,centered,0)).sum(axis=1)
        p=(1-a)*self.p0+a*self.p1
        eta=(1-a)*self.eta0+a*self.eta1
        if self.kind=='paired_gaussian':
            mu=np.linalg.solve(p,eta[:,:,None]).squeeze(2)
            diff=self.y-mu
            return np.log(2*np.pi)-.5*np.linalg.slogdet(p)[1]+.5*np.einsum('bi,bij,bj->b',diff,p,diff)
        if self.kind=='gaussian':
            count,target,within=self.y.T
        else:
            count,target,within=1,self.y[:,0],0.
        return count*.5*(np.log(2*np.pi)-np.log(p)+p*((target-eta/p)**2+within))


def material_weights(task,ids):
    """Equal calibration importance per material, averaged over its actual observed responses."""
    families,inverse=np.unique(task.family[ids],return_inverse=True)
    denominators=np.bincount(inverse,weights=task.trials[ids])
    return 1./denominators[inverse],len(families)


def fits(seed):
    out=OUTPUT/f'seed-{seed}'
    receipt=json.loads((out/'fit.json').read_bytes())
    if any(digest(out/k)!=v for k,v in receipt['files'].items()):
        raise ValueError('An expert forecast file changed')
    _,tasks,parts=data()
    methods=['private','shared','midpoint','adaptive']
    cal=[np.load(out/f'{key}-calibration.npz',allow_pickle=False) for key in ('shared','private')]
    coefficients={}
    for t in tasks:
        ids=np.flatnonzero(parts[t.name]==CALIBRATE)
        if not len(ids):
            coefficients[t.name]=dict(weight=.5,calibration_families=0)
            continue
        pool=GeometricPool(t,ids,*(f[t.name] for f in cal))
        mass,n=material_weights(t,ids)
        def objective(a):
            return np.dot(mass,pool.loss(a))-np.log(a)-np.log1p(-a)
        solution=minimize_scalar(objective,method='bounded',bounds=(1e-7,1-1e-7),options={'xatol':2e-8})
        if not solution.success:
            raise ArithmeticError('Endpoint calibration optimizer failed')
        coefficients[t.name]=dict(weight=float(solution.x),calibration_families=n)
    for f in cal:f.close()
    lock=dict(seed=seed,expert_receipt=digest(out/'fit.json'),coefficients=coefficients)
    (out/'borrowing-lock.json').write_text(json.dumps(lock,indent=2,allow_nan=False)+'\n')
    dev=[np.load(out/f'{key}-development.npz',allow_pickle=False) for key in ('shared','private')]
    blocks={};sources=defaultdict(list)
    for t in tasks:
        ids=np.flatnonzero(parts[t.name]==EVALUATE)
        if not len(ids):continue
        pool=GeometricPool(t,ids,*(f[t.name] for f in dev))
        w=weights(t,ids);denom=float(np.dot(w,t.trials[ids]))
        scores={k:float(np.dot(w,pool.loss(a))/denom) for k,a in
                zip(methods,[0.,1.,.5,coefficients[t.name]['weight']],strict=True)}
        blocks[t.name]=dict(source=t.source,likelihood=t.likelihood,rows=len(ids),families=len(set(t.family[ids])),
                            actual_response_denominator=float(np.sum(t.trials[ids])),nll=scores)
        sources[t.source].append([scores[m] for m in methods])
    for f in dev:f.close()
    source={s:dict(zip(methods,np.mean(v,axis=0),strict=True)) for s,v in sorted(sources.items())}
    matrix=np.array([list(s.values()) for s in source.values()]);rng=np.random.default_rng(4173)
    sample=rng.integers(0,len(matrix),(4000,len(matrix)))
    contrasts={}
    for i,key in enumerate(methods[:-1]):
        delta=matrix[:,-1]-matrix[:,i]
        contrasts[key]=dict(adaptive_minus_control=float(delta.mean()),interval=np.quantile(delta[sample].mean(axis=1),[.025,.975]).tolist(),
                            improved=int(sum(delta<0)),total=len(delta))
    report=dict(cross_source_nll=dict(zip(methods,matrix.mean(axis=0),strict=True)),contrasts=contrasts,sources=source,
                blocks=blocks,coefficient_lock=digest(out/'borrowing-lock.json'))
    (out/'scores.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report[k] for k in ('cross_source_nll','contrasts')},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--seed',type=int,default=1729)
    fits(**vars(parser.parse_args()))
