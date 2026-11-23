"""Fit prefix-only conditional time-to-response distributions on primary quiz families."""

import argparse
from collections import Counter
from functools import lru_cache
from pathlib import Path
import re

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from research.backbone_transfer.extract import digest,read_json,stream_rows,write_json
from research.reading_validation.recognition_prepare import ROOT,OUT,key

MODELS=('position','relative_length_oracle','lexical','semantic')
RIDGE=(.001,.01,.1,1.)
BUCKETS=192


class Source:
    def __init__(self,root):
        self.geometry=read_json(root/'geometry.json');self.states=np.load(root/'states.npy',mmap_mode='r');self.docs={r['id']:r for r in stream_rows(ROOT/'stimuli.jsonl')}
        self.index={k:i for i,k in enumerate(self.geometry['ids'])}
        if self.geometry['source']!=digest(ROOT/'stimuli.jsonl') or set(self.index)!=set(self.docs):raise ValueError('Grade-free stimuli mismatch')

    @lru_cache(maxsize=150)
    def raw(self,k):
        r=self.docs[k];i=self.index[k];a,b=self.geometry['word_offsets'][i:i+2];S=self.states[a:b].astype('float32');text=r['text'];ends=r['word_char_ends'];n=len(ends)
        words=[text[left:right].strip() for left,right in zip([0,*ends[:-1]],ends,strict=True)];forms=[re.sub(r'[^\w]','',w).casefold() for w in words]
        current=np.arange(n)+1;sent=np.cumsum([0,*[bool(re.search(r'[.!?][)\]”"\']*$',w)) for w in words[:-1]]]);age=np.zeros(n);rep=[];seen=Counter();cues=[]
        for j,w in enumerate(forms):
            if j and sent[j]==sent[j-1]:age[j]=age[j-1]+1
            rep.append(seen[w]);seen[w]+=1
            cues.append([w in {'this','these','which','who','name','identify'},w in {'ten','points','ftp'},w in {'for','a','the'},words[j][0].isupper(),
                         bool(re.search(r'[\d]',w)),bool(re.search(r'[.?!]$',words[j])),np.log1p(len(w))])
        relative=current/n
        x=np.column_stack((np.ones(n),np.log1p(current),current/100,np.log1p(sent),np.log1p(age),(age==0),
                           *[np.maximum(current-stop,0)/60 for stop in (10,25,50,75,100,125,150,175,220)])).astype('float32')
        oracle=np.column_stack((x,relative,*[np.maximum(0.,relative-f) for f in (.1,.25,.5,.65,.75,.85,.95)])).astype('float32')
        hashed=np.zeros((n,BUCKETS),dtype='float32')
        for j,w in enumerate(forms):
            left=forms[j-1] if j else ''
            for cat in ['u.'+w,'p.'+left,'b.'+left+'_'+w,'s.'+w[-3:]]:
                h=int(key(cat)[:8],16);hashed[j,h%BUCKETS]+=(1. if h>=2**31 else -1.)
        lexical=np.column_stack((x,np.log1p(rep),cues,hashed)).astype('float32')
        z=S[:,:-4];nz=z/np.maximum(np.sqrt((z*z).mean(1,keepdims=True)),.001);cs=np.vstack((np.zeros((1,z.shape[1]),dtype='float32'),nz.cumsum(0)))
        lo=np.maximum(0,current-5);local=(cs[current]-cs[lo])/(current-lo)[:,None];prior=cs[current-1]/np.maximum(1,current-1)[:,None]
        semantic=np.column_stack((lexical,S[:,-4:],nz,local,nz-local,nz*prior)).astype('float32')
        return dict(position=x,relative_length_oracle=oracle,lexical=lexical,semantic=semantic)


def arrays(bank,rows,arm):
    clusters=Counter(r['group'] for r in rows);xx=[];yy=[];ww=[]
    for r in rows:
        x=bank.raw(r['id'])[arm];d=np.zeros(len(x))
        for lo,hi,n in r['intervals']:
            if hi!=lo+1:raise ValueError('Only exact word observations enter model training')
            d[lo]+=n
        risk=d[::-1].cumsum()[::-1];ok=risk>0;xx.append(x[ok]);yy.append((d[ok]/risk[ok]).astype('float64'))
        ww.append(risk[ok]/r['n']/clusters[r['group']]/len(clusters))
    return np.vstack(xx).astype('float64'),np.concatenate(yy),np.concatenate(ww)


def objective(v,X,Y,W,penalty):
    z=X@v;reg=v.copy();reg[0]=0.; loss=float(W@(np.logaddexp(0,z)-Y*z))+.5*penalty*(reg@reg)
    return loss,X.T@(W*(expit(z)-Y))+penalty*reg


def finish_newton(v,x,y,w,penalty):
    """Polish a stalled quasi-Newton solution of this convex, regularized likelihood."""
    v=v.copy()
    regularizer=np.eye(len(v))*penalty;regularizer[0,0]=0.
    for step in range(15):
        value,gradient=objective(v,x,y,w,penalty)
        if np.max(np.abs(gradient))<4e-6:return v,value,gradient,step
        p=expit(x@v);hessian=x.T@((w*p*(1.-p))[:,None]*x)+regularizer
        direction=np.linalg.solve(hessian,gradient);descent=float(gradient@direction)
        scale=1.
        while scale>=2**-20:
            proposal=v-scale*direction
            if objective(proposal,x,y,w,penalty)[0]<=value-1e-4*scale*descent:break
            scale*=.5
        else:raise ArithmeticError('Recognition Newton line search failed')
        v=proposal
    value,gradient=objective(v,x,y,w,penalty)
    return v,value,gradient,15


def fit(root):
    seal=read_json(OUT/'input-seal.json');bank=Source(root);models={}
    for split in ('train','validation'):
        if digest(ROOT/(split+'.jsonl'))!=seal['outcome_sha256'][split]:raise ValueError('Training data changed')
    train=list(stream_rows(ROOT/'train.jsonl'));val=list(stream_rows(ROOT/'validation.jsonl'))
    for arm in MODELS:
        x,y,w=arrays(bank,train,arm);a,b,c=arrays(bank,val,arm);scale=np.sqrt(np.average(x*x,axis=0,weights=w));scale=np.maximum(.05,scale);scale[0]=1.
        x/=scale;a/=scale;results=[];weights=[]
        for lam in RIDGE:
            v=np.zeros(x.shape[1]);v[0]=-4.5
            sol=minimize(objective,v,args=(x,y,w,lam),method='L-BFGS-B',jac=True,options={'maxiter':700,'ftol':1e-13,'gtol':4e-6,'maxls':60})
            v,value,gradient,extra=finish_newton(sol.x,x,y,w,lam)
            loss=objective(v,a,b,c,0.)[0];grad=float(np.abs(gradient).max())
            if not np.isfinite(loss) or grad>0.00015:raise ArithmeticError(f'{arm} gradient {grad}')
            weights.append(v/scale);results.append({'penalty':lam,'training_nll_penalised':float(value),'development_nll':loss,'gradient':grad,'iterations':int(sol.nit),'newton_steps':extra})
        ix=int(np.argmin([t['development_nll'] for t in results]));models[arm]={'weights':weights[ix].tolist(),'trials':results,'chosen':ix};print(arm,results[ix],flush=True)
        bank.raw.cache_clear()
    write_json(OUT/'weights.json',{'stimuli_sha256':seal['stimuli_sha256'],'arms':models,'heldout_grades_loaded':False})
    result={}
    for k in sorted(bank.docs):
        raw=bank.raw(k);result[k]=np.stack([(raw[m]@np.asarray(models[m]['weights'])).astype('float32') for m in MODELS])
    np.savez_compressed(ROOT/'forecasts.npz',**result)
    write_json(OUT/'forecast-seal.json',{'model_sha256':digest(OUT/'weights.json'),'forecasts':digest(ROOT/'forecasts.npz'),'geometry':digest(root/'geometry.json'),'test_outcomes_loaded':False})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);fit(p.parse_args().root)
