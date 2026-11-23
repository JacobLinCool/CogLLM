"""Estimate an outstanding incompatibility with a cause-specific, causal first-event observation model."""
from collections import Counter,defaultdict
from dataclasses import dataclass
import json

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp, softmax

from research.reading_validation.memory_data import h
from research.reading_validation.state_words import ATTR,TRIP, norm,standard


def materials(archive,split):
    design=archive.spec['groups']['physical'][split];source=json.loads((TRIP/'www.json').read_bytes())[split];items={};skip=Counter()
    for ident,spec in design.items():
        key=spec['key'];r=source[ident]
        if not archive.eligible(key) or not archive.eligible(spec['original']):skip['native_coordinates']+=1;continue
        doc=archive.document(key)
        if list(map(list,doc['regions']))!=spec['spans']:skip['display_boundary']+=1;continue
        items[ident]=(r,spec)
    return items,dict(skip)


def state_signals(doc,i,j,pred):
    a,b=doc['starts'][i:i+2];c,d=doc['starts'][j:j+2];prev=dict(zip(doc['names'][a:b],range(a,b)));now=dict(zip(doc['names'][c:d],range(c,d)))
    shared=prev.keys()&now.keys()
    if not shared:return np.zeros(2*len(ATTR)+5)
    earlier,late=zip(*[(prev[k],now[k]) for k in sorted(shared)],strict=True)
    before=pred[list(earlier),:,1];after=pred[list(late),:,0]
    # Human labels are selectively specified. For ungrounded attributes integrate out the comparison.
    available=before[:,:,1]*after[:,:,1];q=before[:,:,0];r=after[:,:,0];mismatch=q*(1-r)+(1-q)*r
    evidence=mismatch*available;change=(r-q)*available
    return np.r_[evidence.max(0),change[np.abs(change).argmax(0),np.arange(len(ATTR))],evidence.max(),evidence.sum(),available.max(),available.sum(),len(shared)]


def pair_rows(doc,state=None):
    rows=doc['rows'];out=[]
    for j,late in enumerate(rows[1:],1):
        parts=defaultdict(list)
        for i,earlier in enumerate(rows[:j]):
            overlap=late['content']&earlier['content'];union=late['content']|earlier['content'];distance=j-i
            form=np.r_[1.,np.log1p(distance),1/distance,int(distance==1),earlier['surface'],late['surface'],
                      len(overlap)/max(len(union),1),len(overlap)/max(len(late['content']),1),len(overlap),
                      earlier['bag'],late['bag'],earlier['bag']*late['bag']]
            parts['form'].append(form);parent=form
            if 'surp' in late:
                z=np.r_[form,earlier['surp'],late['surp'],late['surp']-earlier['surp']];parts['surprisal'].append(z)
                now=late['reps'];old=earlier['reps'];product=now*old
                neural=np.r_[z,late['att'][i],np.sum(product,1),product[:2,:48].ravel(),
                              (now[1,:48]-old[1,:48]),now[1,:48],old[1,:48]]
                parts['context'].append(neural);parent=neural
            if state is not None:parts['state'].append(np.r_[parent,state_signals(doc,i,j,state)])
        out.append({k:np.stack(v) for k,v in parts.items()})
    return out


def build_pairs(items,archive,models,folds,exclude_training=False):
    bank={}
    for r,s in items.values():
        k=s['key'];g=s['family'];model=models[folds[g]] if exclude_training else models
        doc=archive.document(k);bank[k]=pair_rows(doc,model.predict(doc['x']))
    return bank


def balance(items):
    per=Counter((s['family'],r['plausible']) for r,s in items.values());nf=len({s['family'] for r,s in items.values()});return {k:.5/nf/per[s['family'],r['plausible']] for k,(r,s) in items.items()}


def accepted(r):
    j=r['breakpoint'];return set(i for i,b in r['confl_pairs'] if b==j)


@dataclass
class Options:
    x: np.ndarray
    start: np.ndarray
    ix: np.ndarray
    offsets: np.ndarray
    positive: np.ndarray
    weight: np.ndarray

    def event(self,v):
        logits=self.x@v+self.offsets;hm=np.maximum.reduceat(logits,self.start)
        val=np.exp(logits-hm[self.ix]);den=np.add.reduceat(val,self.start);p=val/den[self.ix]
        logz=hm+np.log(den);correct=np.where(self.positive,logits,-np.inf);best=np.maximum.reduceat(correct,self.start)
        e=np.exp(correct-best[self.ix]);mass=np.add.reduceat(e,self.start);q=e/mass[self.ix]
        return (self.weight@(logz-best-np.log(mass)),self.x.T@((p-q)*self.weight[self.ix]))


def packed(items,bank,arm):
    x=[];positive=[];w=[];offsets=[];st=[];pairx=[];pw=[];weights=balance(items)
    for ident,(r,s) in items.items():
        k=s['key'];table=bank[k];last=r['breakpoint'] if not r['plausible'] else len(table);gold=accepted(r)
        for j in range(1,last+1):
            mx=table[j-1][arm];st.append(len(x));x.extend(np.vstack((np.zeros(mx.shape[1]),mx)))
            offsets.extend(np.r_[0.,np.full(j,-np.log(j))]);w.append(weights[ident]);target=np.zeros(j+1,dtype=bool)
            if not r['plausible'] and j==last:
                target[1+np.array(sorted(gold))]=True
                if s['original'] in bank:
                    # Match exactly the pair coordinates that a person changed in the source design.
                    # The mean of accepted logits defines a convex paired ranking term.
                    ix=np.array(sorted(gold));pairx.append(mx[ix].mean(0)-bank[s['original']][j-1][arm][ix].mean(0));pw.append(weights[ident])
            else:target[0]=True
            positive.extend(target)
    st=np.asarray(st);n=len(x);index=np.repeat(np.arange(len(st)),np.diff(np.r_[st,n]));return np.stack(x),st,index,np.array(offsets),np.array(positive),np.array(w),np.stack(pairx),np.array(pw)


class EventEstimator:
    def __init__(self,items,bank,arm):
        raw,st,ix,off,positive,weight,match,pw=packed(items,bank,arm);ones=np.ones(len(raw),dtype=bool);ones[st]=False
        w=weight[ix[ones]];self.mu,self.sd=standard(raw[ones],w);self.mu[0]=0;self.sd[0]=1
        z=(raw-self.mu)/self.sd;z[st]=0;match=match/self.sd;pw=pw/pw.sum();measure=Options(z,st,ix,off,positive,weight)
        def total(v):
            event,gradient=measure.event(v);d=match@v;pair=pw@np.logaddexp(0,-d);g=match.T@(pw*(expit(d)-1))
            val=.75*event+.25*pair+.025*np.square(v[1:]).sum();gg=.75*gradient+.25*g;gg[1:]+=.05*v[1:];return val,gg
        fit=minimize(total,np.zeros(z.shape[1]),jac=True,method='L-BFGS-B',options={'maxiter':500,'gtol':1e-7,'ftol':1e-13})
        if not fit.success or max(abs(fit.jac))>1e-5:raise ArithmeticError(f'First-event likelihood failed to converge {fit.message}')
        self.v=fit.x;self.diag=dict(likelihood=float(fit.fun),grad=float(max(abs(fit.jac))),iterations=fit.nit,opportunities=len(st),pairs=len(pw))

    def predict(self,table,arm):
        return event_probabilities(table,arm,self.mu,self.sd,self.v)


def event_probabilities(table,arm,mean,scale,coefficient):
    return [softmax(np.r_[0.,((r[arm]-mean)/scale)@coefficient-np.log(j)])
            for j,r in enumerate(table,1)]


def path(r,probs):
    gold=accepted(r);bp=r['breakpoint'];v=0.
    for j,x in enumerate(probs,1):
        if bp==j:return v-np.log(np.clip(sum(x[1+k] for k in gold),1e-15,1))
        v-=np.log(np.clip(x[0],1e-15,1))
    return v


def mixture(items,forecasts,arms):
    mass=balance(items);pa=[];weight=[]
    for ident,(r,s) in items.items():
        gold=accepted(r);bp=r['breakpoint'];arr=[forecasts[m][s['key']] for m in arms]
        for j,x in enumerate(zip(*arr,strict=True),1):
            pa.append([sum(p[i+1] for i in gold) if j==bp else p[0] for p in x]);weight.append(mass[ident]);
            if j==bp:break
    vals=np.array(pa);w=np.array(weight);n=len({s['family'] for r,s in items.values()})
    def objective(a):
        p=vals@a;return -(np.log(p)@w)-np.log(a).sum()/n, -(vals.T@(w/p))-1/(a*n)
    fit=minimize(objective,np.ones(len(arms))/len(arms),jac=True,method='SLSQP',bounds=[(1e-6,1)]*len(arms),
                constraints=[dict(type='eq',fun=lambda a:a.sum()-1,jac=lambda a:np.ones(len(a)))],options={'maxiter':400,'ftol':1e-12})
    if not fit.success:raise ArithmeticError('Development probability mixture failed to converge')
    return fit.x


def combine(all_preds,arms,weights):
    return {k:[np.average(np.stack(rows),axis=0,weights=weights) for rows in zip(*[all_preds[a][k] for a in arms],strict=True)] for k in all_preds[arms[0]]}
