"""Calibrate cause-specific physical conflict evidence on the independent reader's first-report process."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from research.reading_validation.argument_data import sentences
from research.reading_validation.memory_data import BASE,load_rows,sha
from research.reading_validation.memory_fit import counts
from research.reading_validation.memory_score import reveal,summarize
from research.reading_validation.state_score import unpack
from research.reading_validation.state_transport_data import OUT

PLAN=Path(__file__).with_name('state-reports-estimator.md')

def signal(ss,values):
    text='\n\n'.join(ss);regions=sentences(text);pp=unpack(values,len(regions));where=[];start=0;orig=[]
    for s in ss:
        orig.append((start,start+len(s)));start+=len(s)+2
    for a,b in regions:
        found=[i for i,(u,v) in enumerate(orig) if u<=a<b<=v]
        if len(found)!=1:raise ValueError('Physical event location crosses one of the reader displays')
        where.append(found[0])
    if set(where)!=set(range(len(ss))):raise ValueError('The original reader display lacks a completed region')
    parts=defaultdict(list)
    for j,p in enumerate(pp,1):parts[where[j]].append(p)
    estimates=[];last=0
    for j,s in enumerate(ss):
        f=parts[j];hz=1-np.prod([p[0] for p in f]);sources=np.zeros(j+1);prior=1.
        for p in f:
            np.add.at(sources,np.asarray(where[:len(p)-1],dtype=int),prior*p[1:]);prior*=p[0]
        if abs(sum(sources)-hz)>1e-5:raise ArithmeticError('Report probability does not conserve possible causes')
        conditional=sources/max(hz,1e-8)
        ff=np.array([np.log1p(j),np.log1p(len(s.split())),np.log1p(len(s)),j/(5+j)])
        evidence=np.array([hz, sources.max(initial=0),conditional.max(initial=0),conditional@conditional,hz-last,max(0.,hz-last)])
        estimates.append((ff,evidence));last=hz
    return estimates

def fit_model(rows,observations,inputs):
    X=[];e=[];n=[];w=[]
    for r in rows:
        k=r['id'];risk,first,readers=observations[k];take=risk>0;take[0]=False
        X.extend(inputs[k][take]);e.extend(first[take]);n.extend(risk[take]);w.extend([1/readers]*take.sum())
    x=np.array(X);e=np.array(e);n=np.array(n);w=np.array(w);w/=w@n;wt=w*n;mu=wt@x;sd=np.sqrt(wt@((x-mu)**2));sd[sd<1e-6]=1
    z=np.column_stack((np.ones(len(x)),(x-mu)/sd))
    def loss(v):
        val=z@v;g=z.T@(w*(n*expit(val)-e));g[1:]+=.015*v[1:]
        return w@(n*np.logaddexp(0,val)-e*val)+.0075*(v[1:]@v[1:]),g
    fit=minimize(loss,np.zeros(z.shape[1]),jac=True,method='L-BFGS-B',options={'maxiter':500,'gtol':1e-9,'ftol':1e-14})
    if not fit.success or abs(fit.jac).max()>1e-6:raise ValueError('Human risk calibration did not converge')
    def apply(x):
        pp=expit((x-mu)/sd@fit.x[1:]+fit.x[0]);pp[0]=0;return pp
    return apply,float(abs(fit.jac).max())

def forecast(arm):
    origin=OUT/('forecast-'+arm);seal=json.loads((origin/'seal.json').read_bytes())
    if sha(origin/'events.npz')!=seal['files']['events.npz']:raise ValueError('Physical evidence differs from trained state')
    dest=OUT/('reports-'+arm);dest.mkdir(exist_ok=False);design=json.loads((OUT/'inputs.json').read_bytes())['groups']['frick']
    meta=json.loads((BASE/'memory/study-map.json').read_bytes());fit=[r for r in meta if r['study']==2];test=[r for r in meta if r['study']==3]
    obs=counts(reveal(2,meta),fit);texts={r['id']:r['sentences'] for r in load_rows(BASE/'memory/stories.jsonl')};inputs=defaultdict(dict);raw={}
    modes=json.loads((origin/'model.json').read_bytes())['modes']+['mixture']
    with np.load(origin/'events.npz') as f:
        for k in set(r['id'] for r in fit+test):
            for m in modes:
                data=signal(texts[k],f[design[k]+'.'+m]);a=np.stack([x for x,y in data]);b=np.stack([y for x,y in data]);inputs['position'][k]=a;inputs[m][k]=np.column_stack((a,b))
                if m=='mixture':
                    h=b[:,0].copy();h[0]=0;raw[k]=h
    results={};errors=[]
    for family in sorted({r['story'] for r in test}):
        train=[r for r in fit if r['story']!=family];held=[r for r in test if r['story']==family]
        for m,data in inputs.items():
            apply,err=fit_model(train,obs,data);errors.append(err)
            for r in held:results['strict.'+m+'.'+r['id']]=apply(data[r['id']])
    for r in test:results['strict.uncalibrated.'+r['id']]=raw[r['id']]
    old=Path('research/results/reading-validation-premise-emission/forecast')
    with np.load(old/'hazards.npz') as f:
        for r in test:results['strict.prior.'+r['id']]=f['strict.directional.'+r['id']]
    np.savez_compressed(dest/'forecasts.npz',**results)
    (dest/'seal.json').write_text(json.dumps(dict(protocol=sha(PLAN),physical=sha(origin/'seal.json'),previous=sha(old/'seal.json'),data=sha(dest/'forecasts.npz'),
                    readers_in_calibration=276,largest_gradient=max(errors),arms=['position',*modes,'uncalibrated','prior']),indent=2)+'\n')

def score(arm):
    dest=OUT/('reports-'+arm);s=json.loads((dest/'seal.json').read_bytes())
    if sha(dest/'forecasts.npz')!=s['data']:raise ValueError('Reported probabilities changed')
    meta=json.loads((BASE/'memory/study-map.json').read_bytes())
    with np.load(dest/'forecasts.npz') as f:q=summarize(f,reveal(3,meta),20311,cohorts=('strict',),methods=s['arms'],focus='mixture',controls=('position','form','prior'))
    (dest/'scores.json').write_text(json.dumps(dict(seal=sha(dest/'seal.json'),cohort3=q),indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('fit','score'));p.add_argument('arm',choices=('form','native'));a=p.parse_args()
    (forecast if a.action=='fit' else score)(a.arm)
