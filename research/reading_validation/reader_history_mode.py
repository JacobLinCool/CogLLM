"""Learn between-reader concentration from within-reader pairs and retain within-reader variability."""
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp

from research.backbone_transfer.extract import digest, read_json, write_json
from research.reading_validation.reader_history_external import base, DIR, OUT
from research.reading_validation.reader_history_peer import BETA, binary, conditionals, histories, population_weights
from research.reading_validation.reader_history_sequence import timeline, probabilities, mixture
from research.reading_validation.reader_history_model import report
from research.reading_validation.reader_history_decompose import decompose

PLAN=Path(__file__).with_name('reader-history-mode-mixture.md')


def pair_concentration(loglik, readers, prior):
    """Exact one-dimensional maximum of the observed-pair composite likelihood."""
    logp = logsumexp(loglik + np.log(prior),axis=1)
    ratio = np.exp(loglik-logp[:,None]) * np.sqrt(prior)
    association=[]
    for person in np.unique(readers):
        x=ratio[readers==person]
        i,j=np.triu_indices(len(x),1)
        association.extend(np.sum(x[i]*x[j],axis=1)-1.)
    u=np.asarray(association)
    if not len(u) or not np.all(u>-1):
        raise ValueError('Invalid observed pair support')
    def derivative(x):return np.mean(u/(1+x*u))
    if derivative(0)<=0:
        lam=0.
    elif derivative(1)>=0:
        lam=1.
    else:
        lam=brentq(derivative,0.,1.,xtol=1e-12)
    return float(lam), dict(pairs=len(u),derivative=float(derivative(lam)))


def fit(native):
    reg=read_json(OUT/'mode-registration.json')
    if reg['protocol']!=digest(PLAN) or reg['previous']!=digest(OUT/'sequential-scores.json'):
        raise ValueError('Changed registered result or materials')
    lp,idx,_=base(native);odds,zn=conditionals(lp)
    hist,_=histories(idx);r,i,w=hist.T;resid=w==lp.shape[1]-1
    source=DIR/'peer-fit-parameters.json'
    with np.load(DIR/'peer-future-geometry.npz') as f:
        sf=f['stem'];rf=dict(zip(f['subjects'],f['reader'],strict=True))
    out=[]
    for h in read_json(source):
        a,b=h['reader_fold'],h['stem_fold']
        tr=(np.array([rf[k] for k in r])!=a)&(sf[i]!=b)
        logyes,logno=binary(odds,np.asarray(h['class_coeff']))
        classes=np.where(resid[:,None],logyes[i],logno[i])
        names=lp[i,w,None].astype(float)*BETA-zn[i]
        row=dict(reader_fold=a,stem_fold=b,class_coeff=h['class_coeff'])
        for key,obs,keep in [('class',classes,tr),('name',names,tr&~resid)]:
            mat=obs[keep]
            prior,fitinfo=population_weights(mat,np.arange(len(mat)))
            lam,detect=pair_concentration(mat,r[keep],prior)
            row[key]=dict(prior=prior.tolist(),intraclass=lam,effective_prior_n=(1/lam-1 if lam>0 else None),
                          fit=fitinfo,calibration=detect,peer_responses=int(keep.sum()))
        out.append(row)
    path=OUT/'mode-priors.json';write_json(path,out)
    write_json(OUT/'mode-prior-lock.json',dict(registration=digest(OUT/'mode-registration.json'),parameters=digest(path),
         fit_only_earlier_peer_answers=True,calibration_choice_total=len(hist),held_future_fit_choices=0,
         max_gradient=max(m[k]['fit']['stationarity'] for m in out for k in ('class','name')),
         full_intraclass_range={k:np.quantile([m[k]['intraclass'] for m in out],[0,.5,1]).tolist() for k in ('class','name')}))


def online_priors(observations, update, included, prior, lam):
    """Run a chronological moment-matched Dirichlet filter for each observed participant."""
    readers,steps=update.shape
    values=np.empty((readers,steps+1,len(prior)),np.float32)
    state=np.tile(prior,(readers,1));mass=np.zeros(readers)
    values[:,0]=state
    if lam==0:
        values[:,1:,:]=prior
        return values
    strength=max(1e-8,1/lam-1)
    for k in range(steps):
        live=update[:,k]&included[:,k]
        if live.any():
            active=state[live]
            responsibility=observations[live,k]*active
            responsibility/=responsibility.sum(1,keepdims=True)
            mass[live]+=1
            state[live] += (responsibility-active)/(strength+mass[live,None])
        values[:,k+1]=state
    if abs(values.sum(2)-1).max()>2e-6 or values.min()<0:
        raise ValueError('Response-mode distribution lost normalization')
    return values


def observed_paths(table,people,sf,b):
    times=[];items=[];words=[]
    for r in people:
        t,i,w=np.array(table[r]).T;i=i.astype(int);w=w.astype(int);ok=sf[i]!=b
        times.append(t[ok]);items.append(i[ok]);words.append(w[ok])
    length=np.array([len(t) for t in times]);lim=int(length.max())
    inputs=np.zeros((len(people),lim),np.int32);answers=np.zeros_like(inputs)
    for j,(i,w) in enumerate(zip(items,words,strict=True)):
        inputs[j,:len(i)]=i;answers[j,:len(w)]=w
    return times,length,inputs,answers


def score(native):
    lock=read_json(OUT/'mode-prior-lock.json')
    if lock['parameters']!=digest(OUT/'mode-priors.json'):
        raise ValueError('Changed mode priors')
    lp,idx,_=base(native);odds,zn=conditionals(lp)
    all_rows,table,yoked=timeline(idx)
    people=sorted(table);mapping={r:j for j,r in enumerate(people)}
    by_time={(int(r),int(i)):start for r,i,_,start,_ in all_rows}
    original=read_json(OUT/'sequential-scores.json')
    sample=DIR/'sequential-forecast-scores.npz'
    if digest(sample)!=original['scored_arrays']:
        raise ValueError('Original time-matched observations changed')
    # Only the registered event geometry and history sizes are needed from the previous release.
    with np.load(sample) as g:
        E=g['events'];support=g['support']
    r,i,w=E.T
    start=np.array([by_time[p,z] for p,z in zip(r,i,strict=True)])
    with np.load(DIR/'peer-future-geometry.npz') as f:
        sf=f['stem'];rf=dict(zip(f['subjects'],f['reader'],strict=True))
    rs=np.array([rf[p] for p in r]);L=np.full((len(r),4),np.nan);O=np.full_like(L,np.nan)
    outside=lp.shape[1]-1;flag=w==outside
    for h in read_json(OUT/'mode-priors.json'):
        a,b=h['reader_fold'],h['stem_fold']
        rows=np.flatnonzero((rs==a)&(sf[i]==b));times,length,inp,ans=observed_paths(table,people,sf,b)
        active=np.arange(inp.shape[1])[None,:]<length[:,None];outcome=ans==outside
        yes,no=binary(odds,np.asarray(h['class_coeff']))
        pc=np.asarray(h['class']['prior']);pn=np.asarray(h['name']['prior'])
        probs=np.exp(np.where(outcome[:,:,None],yes[inp],no[inp]))
        C=online_priors(probs,active,np.ones_like(outcome),pc,h['class']['intraclass']);del probs
        probs=np.exp(lp[inp,ans,None]*BETA-zn[inp])
        Z=online_priors(probs,active,~outcome,pn,h['name']['intraclass']);del probs
        groups=defaultdict(list)
        for j in rows:groups[r[j],support[j]].append(j)
        cache={}
        for (owner,n),indices in groups.items():
            indices=np.asarray(indices);u=mapping[owner];d=mapping[yoked[owner]]
            if length[u]<n or length[d]<n or times[u][n-1]>min(start[indices]):
                raise ValueError('Same-person profile sees a pending assignment')
            if n not in cache:
                valid=length>=n
                c=C[valid,n].astype(float);z=Z[valid,n].astype(float)
                cache[n]=c.sum(0),c.T@z,int(valid.sum())-1
            c,joint,k=cache[n];own=mixture(C[u,n].astype(float),Z[u,n].astype(float))
            mean=(c-own[0])/k,(joint-own[1])/k
            yoke=mixture(C[d,n].astype(float),Z[d,n].astype(float));unknown=mixture(pc,pn)
            pos=i[indices];word=w[indices]
            conditional=np.exp(lp[pos,word,None].astype(float)*BETA-zn[pos])
            names,out=probabilities(np.exp(yes[pos]),np.exp(no[pos]),conditional,[own,mean,yoke,unknown])
            L[indices]=-np.log(np.where(flag[indices,None],out,names));O[indices]=out
    if not np.isfinite(L).all() or np.any((O<=0)|(O>=1)):
        raise ValueError('Nonfinite normalized mode prediction')
    result=report(E,L,O);result['choice_content']=decompose(E,L,O)
    by_band={}
    for lo,hi in ((20,49),(50,199),(200,799),(800,10**6)):
        m=(support>=lo)&(support<=hi)
        by_band[str(lo)]=report(E[m],L[m],O[m],draws=1000)['paired']['same_over_unknown_reader']
    out=DIR/'mode-sequential-forecasts.npz';np.savez_compressed(out,events=E,loss=L,other=O,support=support)
    write_json(OUT/'mode-scores.json',dict(results=result,registered=lock,score_sha256=digest(out),
        comparison_keys=original['comparison_keys'],history_size_percentiles=original['history_size_percentiles'],
        support_bands_over_experience_matched=by_band,human_events=len(E),prior_history_geometry_identical=True))
    print(result['paired'],flush=True)


if __name__=='__main__':
    p=ArgumentParser(description=__doc__);p.add_argument('action',choices=['fit','score']);p.add_argument('native',type=Path)
    args=p.parse_args();(fit if args.action=='fit' else score)(args.native)
