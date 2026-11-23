"""Likelihood of reader effectiveness grades marginalized over admissible earlier claims."""
from collections import Counter
from dataclasses import dataclass

import numpy as np
from scipy.special import expit

from research.reading_validation.argument_fit import predictions,optimize


def grouped(scores,starts,group):
    top=np.maximum.reduceat(scores,starts)
    value=np.exp(scores-top[group]);total=np.add.reduceat(value,starts)
    return value/total[group],top+np.log(total)


def scaled(x,mean=None,sd=None):
    if mean is None:
        mean=x.mean(0);sd=x.std(0);sd[sd<1e-8]=1.
    return (x-mean)/sd,mean,sd


def essay_weight(keys,n):
    counts=Counter()
    for k,v in zip(keys,n,strict=True):counts[k]+=v
    eligible=sum(v>0 for v in counts.values())
    return np.array([1/(eligible*counts[k]) if counts[k] else 0. for k in keys])


@dataclass
class Evidence:
    keys:list
    endpoints:np.ndarray
    ptr:np.ndarray
    event:np.ndarray
    grade:np.ndarray
    gate_x:np.ndarray
    pair_x:np.ndarray
    nc:np.ndarray
    forms:list
    choices:list
    exists:np.ndarray
    known:np.ndarray
    counts:np.ndarray


def features(corpus,keys,arm,expert,labels=None):
    which=[];endpoints=[];ptr=[];event=[];grade=[];gx=[];px=[];lengths=[];choices=[];exists=[];known=[];counts=[]
    for k in keys:
        if not corpus.eligible(k):continue
        pos,links=corpus.design(k);base=predictions(pos,links,expert,arm)
        rec=labels[k] if labels is not None else None
        for j,(x,pairs) in enumerate(zip(pos,links,strict=True)):
            which.append(k);endpoints.append(j);q=base['event'][j];event.append(np.log(max(q,1e-12)/max(1-q,1e-12)))
            cumulative=np.cumsum(base['grade'][j])[:2].clip(1e-9,1-1e-9)
            grade.append(np.log(cumulative/(1-cumulative)))
            xg=x['form']
            if arm=='native':xg=np.r_[xg,x['native'][-14:]]
            gx.append(xg);lengths.append(j+1)
            old=base['choice'][j*(j-1)//2:(j+1)*j//2]
            ptr.extend(np.r_[0.,np.log(old.clip(1e-15))])
            dim=29+(7 if arm=='native' else 0)
            if j:
                f=pairs['form'];v=f[:,:28]
                if arm=='native':v=np.column_stack((v,pairs['native'][:,f.shape[1]:f.shape[1]+7]))
                px.append(np.vstack((np.r_[1.,np.zeros(dim-1)],np.column_stack((np.zeros(j),v)))))
            else:px.append(np.r_[1.,np.zeros(dim-1)][None,:])
            truth=rec['links'].get(j,[]) if rec else []
            exists.append(bool(truth));known.append(bool(rec and rec['complete'] and j and j not in rec['exclude']))
            choices.append(np.isin(np.arange(-1,j),truth) if truth else np.ones(j+1,bool))
            counts.append(np.bincount(rec['grades'].get(j,[]),minlength=3) if rec else np.zeros(3))
    return Evidence(which,np.array(endpoints),np.array(ptr),np.array(event),np.array(grade),np.array(gx),np.vstack(px),
                   np.array(lengths),[],choices,np.array(exists),np.array(known),np.stack(counts))


class Distribution:
    def __init__(self,data,scales=None):
        self.data=data
        x,xm,xs=scaled(data.gate_x,*(scales[:2] if scales else (None,None)))
        v,vm,vs=scaled(data.pair_x,*(scales[2:] if scales else (None,None)))
        self.scales=(xm,xs,vm,vs)
        self.x=np.column_stack((np.ones(len(x)),x))
        self.v=v
        self.group=np.repeat(np.arange(len(x)),data.nc);self.starts=np.r_[0,data.nc.cumsum()[:-1]]
        self.null=np.zeros(len(v),bool);self.null[self.starts]=True
        self.has=data.nc>1;self.valid=np.concatenate(data.choices)
        self.events=essay_weight(data.keys,data.known)*data.known
        self.parents=essay_weight(data.keys,data.exists)*data.exists
        self.grades=essay_weight(data.keys,data.counts.sum(1))[:,None]*data.counts
        self.split=(self.x.shape[1],self.x.shape[1]+v.shape[1])
        self.size=self.split[-1]+v.shape[1]+1

    def evaluate(self,param,transfer=True,traces=False):
        a,b,c=np.split(param,self.split)
        d=self.data;g=self.group;s=self.starts;v=self.v
        e=d.event+(self.x@a if transfer else 0);gate=expit(e)*self.has
        logits=d.ptr+(v@b if transfer else 0)
        # Sentences without a previous sentence have a null-only, deterministic state.
        logits=np.where(self.null&self.has[g],-np.inf,logits)
        parent,logz=grouped(logits,s,g)
        prior=np.where(self.null,1-gate[g],parent*gate[g])
        shift=v@c[:-1]+c[-1];lo=expit(d.grade[g,0]-shift);hi=expit(d.grade[g,1]-shift)
        quality=np.column_stack((lo,hi-lo,1-hi))
        pq=np.add.reduceat(prior[:,None]*quality,s,axis=0)
        if traces:return gate,parent,pq,prior
        # The two fitted marginal likelihoods only use their observed intervals/locations.
        subset,zsub=grouped(np.where(self.valid,logits,-np.inf),s,g)
        event_loss=self.events@(np.logaddexp(0,e)-d.exists*e)
        choice_loss=self.parents@(logz-zsub)
        grade_loss=-np.sum(self.grades*np.log(pq.clip(1e-15)))
        inv=-self.grades/pq.clip(1e-15)
        cost=np.sum(inv[g]*quality,axis=1)
        center=np.add.reduceat(prior*cost,s)
        grad_phi=prior*(cost-center[g])
        source_grad=np.add.reduceat(np.where(self.null,0.,grad_phi),s)
        grad_event=self.x.T@(self.events*(gate-d.exists)+source_grad)
        slope=self.parents[g]*(parent-subset)+np.where(self.null,0.,grad_phi-parent*source_grad[g])
        grad_choice=v.T@slope
        grad_shift=prior*((inv[g,1]-inv[g,0])*lo*(1-lo)+(inv[g,2]-inv[g,1])*hi*(1-hi))
        grad=np.r_[grad_event if transfer else np.zeros_like(a),grad_choice if transfer else np.zeros_like(b),
                   v.T@grad_shift,grad_shift.sum()]/3
        reg=.005
        return (event_loss+choice_loss+grade_loss)/3+.5*reg*(param@param),grad+reg*param

    def fit(self,transfer):
        zero=np.zeros(self.size);direction=np.random.default_rng(93114).normal(size=self.size);direction/=np.linalg.norm(direction)
        err=abs((self.evaluate(zero+1e-5*direction,transfer)[0]-self.evaluate(zero-1e-5*direction,transfer)[0])/2e-5
                  -self.evaluate(zero,transfer)[1]@direction)
        if err>2e-6:raise ArithmeticError(f'Incorrect shared marginal gradient: {err}')
        param,diag=optimize(lambda z:self.evaluate(z,transfer),self.size);diag['gradient_check_error']=err
        return param,diag

    def unpack(self,param,transfer):
        gate,parent,grade,_=self.evaluate(param,transfer,traces=True)
        if max(abs(grade.sum(1)-1))>2e-6 or min(grade.flatten())<0:raise ArithmeticError('Invalid marginal ordinal probabilities')
        result={};d=self.data
        for j,(k,n) in enumerate(zip(d.keys,d.endpoints,strict=True)):
            if k not in result:result[k]=dict(event=[],choice=[],grade=[])
            rec=result[k];rec['event'].append(gate[j]);rec['grade'].append(grade[j])
            rec['choice'].extend(parent[self.starts[j]+1:self.starts[j]+n+1])
        return {k:{name:np.array(value) for name,value in r.items()} for k,r in result.items()}
