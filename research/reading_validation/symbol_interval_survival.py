"""Estimate the time of a source's first explanation under interval and right censoring."""
from argparse import ArgumentParser
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from research.backbone_transfer.extract import digest,read_json,write_json
from research.reading_validation.symbol_factorized import event_design,eligibility_design,readout,conditional_design,available,quality
from research.reading_validation.symbol_evaluate import weights,score_events,metric
from research.reading_validation.symbol_features import canonical
from research.reading_validation.symbol_model import Bank,case_geometry
from research.reading_validation.symbol_geometry import OUT

PROTOCOL=Path(__file__).with_name('symbol-survival-protocol.md');ARMS=('surface','native')
CUT=np.r_[.002,.005,.01,.02,.035,.05,.075,.1,.14,.2,.27,.35,.45,.55,.65,.75,.85,.92,.96,.98,.99,.995,.999,.9999]


def designs(page,arm,model,old):
    t=np.arange(page.n);names=np.array([canonical(page.text[m.start:m.end]) for m in page.mentions]);at=available(page)
    if not len(at):return np.zeros((0,page.n,0),dtype='float32'),at,names
    ev=readout(event_design(page)[0],model['event']);elig=readout(eligibility_design(page)[0],model['eligibility'])[:,1]
    cl=conditional_design(page,t,arm,old)[0];b=np.array(model['attribution']['beta']);r=cl[:,:,:len(b)]@b
    visible=t[None,:]>=at[:,None];pr=r+np.log(np.maximum(elig,1e-12))[:,None];pr[~visible]=-np.inf
    # Compete across names at the same original token. Other aliases of one name do not steal its mass.
    groups,ix=np.unique(names,return_inverse=True);name_scores=np.stack([np.max(pr[ix==j],axis=0) for j in range(len(groups))])
    ordered=np.sort(name_scores,axis=0);max1=ordered[-1];max2=ordered[-2] if len(groups)>1 else np.full(page.n,-np.inf)
    A=np.zeros((len(at),page.n,15),'float32');log_odds=np.log(np.maximum(ev[:,1:],1e-12)/np.maximum(ev[:,:1],1e-12))
    for j,m in enumerate(page.mentions):
        ok=visible[j];nearest=np.where(name_scores[ix[j]]==max1,max2,max1);rival=np.where(np.isfinite(nearest),nearest,pr[j])
        loc=np.flatnonzero(ok);context=cl[j,ok];q=log_odds[ok];h=np.log(np.maximum(elig[j],1e-12));own=pr[j,ok]
        A[j,ok]=np.column_stack((context[:,0],r[j,ok],own-rival[ok],h+np.zeros(len(loc)),np.log1p(loc-at[j]),context[:,2],context[:,3],q,q[:,0]-q[:,1],np.log1p(np.isfinite(name_scores[:,ok]).sum(0)),np.maximum(r[j,ok],0)*expit(q[:,1]),expit(q[:,0])*context[:,0],context[:,6],np.log1p(loc)))
    return A,at,names


def collect(bank,arm,model,old):
    X=[];mask=[];is_event=[];dur=[];papers=[];importance=[]
    for page,doc in bank.pages('fit'):
        A,at,names=designs(page,arm,model,old);geom={(m.start,m.end):ts for m,d,ts in case_geometry(page,doc)};indices={(m.start,m.end):k for k,m in enumerate(page.mentions)}
        ids=[j for k,j in indices.items() if k in geom];neg=[j for k,j in indices.items() if k not in geom];rng=np.random.default_rng(int(doc['id'][:8],16))
        chosen=list(rng.choice(neg,min(6,len(neg)),replace=False))
        for j in ids+chosen:
            k=page.mentions[j];ts=geom.get((k.start,k.end));start=at[j];end=int(ts[-1])+1 if ts is not None else page.n
            if end<=start:continue
            v=A[j,start:end];m=np.zeros(end-start,bool)
            if ts is not None:m[np.asarray(ts)-start]=True
            X.append(v);mask.append(m);is_event.append(ts is not None);dur.append(len(v));papers.append(doc['paper']);importance.append(1. if ts is not None else len(neg)/len(chosen))
    return np.vstack(X),np.concatenate(mask),np.asarray(is_event),np.asarray(dur),papers,np.asarray(importance)


def fit(data):
    X,wzone,positive,lens,ps,factor=data;density=Counter()
    for p,w in zip(ps,factor):density[p]+=w
    weights=np.array([w/density[p]/len(density) for p,w in zip(ps,factor)]);idn=np.repeat(np.arange(len(lens)),lens)
    mean=X.mean(0,dtype=float);sd=np.maximum(X.std(0,dtype=float),.02);Z=np.asarray((X-mean)/sd,dtype=float);Z=np.c_[np.ones(len(Z)),Z];mass=weights[idn]
    loc=np.where(wzone)[0];shape=Z.shape[1];ridge=.012
    def objective(v):
        z=Z@v;lam=np.logaddexp(0,z);A=np.bincount(idn[loc],weights=lam[loc],minlength=len(lens));Apos=np.maximum(A[positive],1e-12)
        penalty=np.zeros(len(lens));penalty[positive]=-np.log(-np.expm1(-Apos))
        cost=mass[~wzone]@lam[~wzone]+weights@penalty+.5*ridge*(v[1:]@v[1:])
        deriv=np.ones(len(lens));deriv[positive]=-1/np.expm1(np.minimum(Apos,650));dl=mass*expit(z);dl[loc]*=deriv[idn[loc]];g=Z.T@dl;g[1:]+=ridge*v[1:]
        return float(cost),g
    init=np.zeros(shape);init[0]=-5
    from scipy.optimize import check_grad
    # Verify the full analytic per-interval gradient along a reproducible direction.
    direction=np.random.default_rng(331).normal(size=shape);h=1e-5;numeric=(objective(init+h*direction)[0]-objective(init-h*direction)[0])/(2*h)
    if abs(numeric-objective(init)[1]@direction)>1e-5:raise ValueError('Marginal interval-likelihood gradient is incorrect')
    op=minimize(objective,init,method='L-BFGS-B',jac=True,options={'maxiter':400,'ftol':1e-14,'gtol':3e-7})
    m=float(abs(op.jac).max())
    if not op.success or m>1e-5:raise ArithmeticError('Interval survival head failed to converge')
    return dict(beta=(op.x[1:]/sd).tolist(),bias=float(op.x[0]-mean@(op.x[1:]/sd)),intervals=int(positive.sum()),censored=int(sum(~positive)),observed_at_risk_tokens=len(X),papers=len(set(ps)),steps=int(op.nit),max_gradient=m,initial_nll=objective(init)[0],final_nll=float(op.fun),directional_gradient_error=float(abs(numeric-objective(init)[1]@direction)))


def survival(A,at,names,model):
    if not len(at):return []
    visible=np.arange(A.shape[1])[None,:]>=at[:,None];beta=np.array(model['beta']);raw=A@beta+model['bias'];intensity=np.where(visible,np.logaddexp(0,raw),0);events=[]
    for name in np.unique(names):
        idx=np.flatnonzero(names==name);arg=idx[np.argmax(intensity[idx],axis=0)];local=intensity[arg,np.arange(len(arg))];F=-np.expm1(-np.cumsum(local))
        events.append((F,arg))
    return events


def decode(events,cut):
    out=[]
    for F,keys in events:
        t=int(np.searchsorted(F,cut))
        if t<len(F):out.append((int(keys[t]),t,float(F[t])))
    return sorted(out,key=lambda a:a[1])


def report(row):
    ps=sorted(row);arr=np.stack([row[k] for k in ps]);s=arr.sum(0);rng=np.random.default_rng(7355);b=arr[rng.integers(len(ps),size=(5000,len(ps)))].sum(1);v=metric(s);bv=metric(b)
    results={}
    for i,arm in enumerate(ARMS):
        results[arm]=dict(counts=s[i].tolist())
        for name in v:
            bb=bv[name][:,i];bb=bb[np.isfinite(bb)];results[arm][name]={'value':float(v[name][i]) if np.isfinite(v[name][i]) else None,'paper_ci':np.quantile(bb,[.025,.975]).tolist() if len(bb) else None}
    gain={}
    for name in ('F1','both_sources_and_timings','retain_other_at_correct_first'):
        d=bv[name][:,1]-bv[name][:,0];d=d[np.isfinite(d)];gain[name]=[float(v[name][1]-v[name][0]),np.quantile(d,[.025,.975]).tolist() if len(d) else None]
    return {'papers':len(ps),'with_competing_sources':int(np.sum(arr[:,0,7]>0)),'arms':results,'paired_native_minus_surface':gain}


def main(args):
    bank=Bank(args.folder);old=weights(read_json(OUT/'coefficients.json')['parameters']);factors=read_json(OUT/'factorized-heads.json')['models'];seal=read_json(OUT/'survival-registration.json');assert digest(PROTOCOL)==seal['plan']
    if args.stage=='fit':
        heads={}
        for a in ARMS:
            dataset=collect(bank,a,factors[a],old);heads[a]=fit(dataset);del dataset
        write_json(OUT/'survival-heads.json',dict(factors=digest(OUT/'factorized-heads.json'),plan=digest(PROTOCOL),code=digest(Path(__file__)),models=heads));return
    models=read_json(OUT/'survival-heads.json')['models'];calibrating=args.stage=='calibrate'
    if not calibrating:cuts=read_json(OUT/'survival-thresholds.json')['thresholds']
    for split in (('calibration',) if calibrating else ('held','published_dev','external')):
        scores=defaultdict(lambda:np.zeros((len(ARMS),8),int));sweep=np.zeros((len(ARMS),len(CUT),8),int)
        for page,doc in bank.pages(split):
            for i,a in enumerate(ARMS):
                A,at,names=designs(page,a,factors[a],old);ev=survival(A,at,names,models[a])
                if calibrating:
                    for k,c in enumerate(CUT):sweep[i,k]+=score_events(page,doc,decode(ev,c))
                else:scores[doc['paper']][i]+=score_events(page,doc,decode(ev,cuts[a]))
        if calibrating:
            cuts={a:float(CUT[max(range(len(CUT)),key=lambda j:quality(sweep[i,j]))]) for i,a in enumerate(ARMS)}
            write_json(OUT/'survival-thresholds.json',dict(weights=digest(OUT/'survival-heads.json'),thresholds=cuts,grid=CUT.tolist(),development_counts=sweep.tolist()))
        else:
            write_json(OUT/('survival-'+split+'.json'),dict(thresholds=digest(OUT/'survival-thresholds.json'),**report(scores)))
        print('Finished proper interval-likelihood evaluation',split,flush=True)


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('fit','calibrate','score'));parser.add_argument('folder',type=Path);main(parser.parse_args())
