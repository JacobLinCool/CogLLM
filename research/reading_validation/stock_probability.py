"""Calibrate occurrence and first-explanation hazards without creating a unit of load at every proposal."""
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from collections import Counter

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from research.backbone_transfer.extract import digest,read_json,stream_rows,write_json
from research.reading_validation.definition_data import OUT,ROOT
from research.reading_validation.definition_features import Corpus
from research.reading_validation.definition_fit import all_spans
from research.reading_validation.definition_predict import feature_series
from research.reading_validation.definition_stock_fit import gold,load_models

ARMS=('position','surface','semantic')
PENALTIES=(.001,.01,.1)


def proposals(page,model,arm):
    left,end,logits=all_spans(page,model,arm)
    ix=[]
    for lo,hi in zip(np.r_[0,np.flatnonzero(np.diff(end))+1],np.r_[np.flatnonzero(np.diff(end))+1,len(end)],strict=True):
        ix.append(lo+np.argmax(logits[lo:hi]))
    left=left[ix];end=end[ix]
    if not np.array_equal(end,np.arange(page.n)):raise ValueError('Expected one possible source per visible complete word')
    features=np.column_stack((np.ones(page.n),logits[ix],np.log1p(end-left+1),np.log1p(page.since),
                              np.log1p(end),np.asarray(page.since)==0))
    return left,end,features


def hazard_features(page,span,model,arm):
    a,b=span;ts=np.arange(b+1,page.n);delay=ts-b;age=np.log1p(delay);v,_=feature_series(page,span,model,arm)
    v=v[ts].astype('float64');prev=np.r_[v[:1],v[:-1]] if len(v) else v
    return np.column_stack((np.ones(len(v)),age,age*age,np.log1p(np.maximum(0,delay-4)),v,v-prev,
                            np.maximum.accumulate(v) if len(v) else v,page.cross[ts]-page.cross[b]))


def cases(bank,rows,arm):
    term,link=load_models()[arm];opens=[];labs=[];source=[];risk=[];starts=[];entries=[];ends=[];files=[];at=0
    for row in rows:
        pg=bank.page(row['id']);left,right,X=proposals(pg,term,arm);g=gold(pg,row)
        truth={(int(a),int(b)) for a,b in g}
        y=[(int(pg.words[a,0]),int(pg.words[b,1])) in truth for a,b in zip(left,right,strict=True)]
        opens.append(X);labs.extend(y);source.extend([row['source']]*pg.n)
        for ab,(_,bands) in g.items():
            span=pg.word_span(ab)
            if span is None:continue
            lo,hi=bands[0];b=span[1]
            H=hazard_features(pg,span,link,arm)[:hi-b-1]
            if not len(H) or not b<lo<hi or hi>pg.n:raise ValueError('Invalid explanation censoring window')
            starts.append(at);entries.append(at+lo-b-1);at+=len(H);ends.append(at);risk.append(H);files.append(row['source'])
    def balance(v):
        c=Counter(v);return np.asarray([1./len(c)/c[k] for k in v])
    return ((np.vstack(opens),np.asarray(labs,dtype=float),balance(source)),(np.vstack(risk),np.asarray(starts),
            np.asarray(entries),np.asarray(ends),balance(files)))


def binary(v,X,y,weights,penalty):
    z=X@v;r=v.copy();r[0]=0.
    return float(weights@(np.logaddexp(0,z)-y*z)+.5*penalty*(r@r)),X.T@(weights*(expit(z)-y))+penalty*r


def interval(v,X,starts,entries,ends,weights,penalty):
    z=X@v;exposure=np.logaddexp(0,z);cs=np.r_[0.,np.cumsum(exposure)];inside=cs[ends]-cs[entries]
    if np.any(inside<=1e-15):return 1e20,np.zeros_like(v)
    loss=cs[entries]-cs[starts]-np.log(-np.expm1(-inside))
    # Difference arrays carry group weights to arbitrary interval lengths.
    dw=np.zeros(len(z)+1);inner=-weights/np.expm1(np.minimum(inside,700.))
    np.add.at(dw,starts,weights);np.add.at(dw,entries,inner-weights);np.add.at(dw,ends,-inner)
    r=v.copy();r[0]=0.
    return float(weights@loss+.5*penalty*(r@r)),X.T@(np.cumsum(dw)[:-1]*expit(z))+penalty*r


def train_model(train,dev,func):
    x,*tr=train;y,*va=dev
    scale=np.sqrt(np.mean(x*x,axis=0));scale=np.maximum(scale,.1);scale[0]=1.;x=x/scale;y=y/scale
    trials=[];models=[]
    for lam in PENALTIES:
        v=np.zeros(x.shape[1]);v[0]=-4. if func==binary else -1.
        sol=minimize(func,v,args=(x,*tr,lam),method='L-BFGS-B',jac=True,options={'maxiter':500,'gtol':1e-7,'ftol':1e-13})
        grad=float(np.abs(sol.jac).max());val=func(sol.x,y,*va,0.)[0]
        if not np.isfinite(val) or grad>3e-5:raise ArithmeticError(f'Unconverged prevalence/hazard model: {grad}')
        models.append((sol.x/scale).tolist());trials.append({'penalty':lam,'dev_nll':val,'train_penalized_nll':sol.fun,'gradient':grad,'steps':int(sol.nit)})
    best=int(np.argmin([r['dev_nll'] for r in trials]))
    return {'weights':models[best],'trials':trials,'chosen':best}


def fit(bank_path):
    bank=Corpus(bank_path);tr=list(stream_rows(ROOT/'train.jsonl'));val=list(stream_rows(ROOT/'calibration.jsonl'));out={}
    for arm in ARMS:
        (a,b),(c,d)=cases(bank,tr,arm),cases(bank,val,arm)
        out[arm]={'occurrence':train_model(a,c,binary),'hazard':train_model(b,d,interval),'proposed_training_word_ends':len(a[0]),
                  'exact_source_positives':int(a[1].sum()),'observed_forward_intervals':len(b[1])}
        print(arm,out[arm]['exact_source_positives'],out[arm]['observed_forward_intervals'],flush=True)
    write_json(OUT/'stock-probability-weights.json',{'training_data':digest(ROOT/'train.jsonl'),'validation_data':digest(ROOT/'calibration.jsonl'),
        'base_weights':digest(OUT/'weights.json'),'protocol':digest(Path(__file__).with_name('stock-probability-protocol.md')),'fits':out})


@dataclass
class ProbabilityTrace:
    starts: np.ndarray
    occurrence: np.ndarray
    unresolved: np.ndarray
    resolved: np.ndarray

    def median(self):
        # Distribution of the sum of independent, source-specific Bernoulli risks.
        out=[]
        for t,p in enumerate(self.unresolved):
            coeff=np.array([1.])
            for rate in p[:t+1]:coeff=np.convolve(coeff,[1.-rate,rate])
            out.append(np.searchsorted(np.cumsum(coeff),.5))
        return np.array(out)


def predict(page,arm,fit,base):
    term,link=base;left,end,X=proposals(page,term,arm);occ=expit(X@fit['occurrence']['weights'])
    unresolved=np.zeros((page.n,page.n));resolved=np.zeros_like(unresolved);unresolved[end,end]=occ
    for b,a in enumerate(left[:-1]):
        h=hazard_features(page,(a,b),link,arm)@fit['hazard']['weights'];survival=np.cumprod(expit(-h))
        unresolved[b+1:,b]=occ[b]*survival;resolved[b+1:,b]=occ[b]*np.r_[1.,survival[:-1]]*expit(h)
    balance=np.cumsum(occ-resolved.sum(1))
    if not np.allclose(balance,unresolved.sum(1),atol=1e-8):raise ArithmeticError('Mismatched opening/closure flow')
    return ProbabilityTrace(left,occ,unresolved,resolved)


if __name__=='__main__':
    from argparse import ArgumentParser
    p=ArgumentParser(description=__doc__);p.add_argument('bank',type=Path);fit(p.parse_args().bank)
