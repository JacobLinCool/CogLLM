"""Fit on auditory responders and score separate German readers and written-only material."""
from argparse import ArgumentParser
from collections import defaultdict
from hashlib import sha256
from pathlib import Path

import numpy as np
from scipy.optimize import brentq,minimize
from scipy.special import logsumexp,softmax
from research.backbone_transfer.extract import digest,read_json,write_json
from research.reading_validation.muse_materials import WORK,OUT,PLAN
from research.reading_validation.reader_history_peer import BETA,binary,calibrate,conditionals,population_weights
from research.reading_validation.reader_history_mode import online_priors
from research.reading_validation.reader_history_model import report,match
from research.reading_validation.reader_history_decompose import decompose
from research.reading_validation.reader_history_sequence import mixture,probabilities

OTHER=4096


def native(folder):
    seal=read_json(OUT/'exposure-seal.json');src=folder/'native.npz';receipt=read_json(folder/'receipt.json')
    if (digest(src)!=receipt['file'] or digest(PLAN)!=seal['protocol'] or receipt['stimuli']!=seal['materials'] or
        receipt['lexicon']!=seal['lexicon'] or digest(WORK/'dictionary.json')!=seal['lexicon']):
        raise ValueError('Published materials, German lexicon or frozen native array changed')
    with np.load(src) as f:
        keys=f['ids'].tolist();logp=f['logp'].astype(float)
    if len(keys)!=619 or logp.shape!=(619,OTHER+1):raise ValueError('Unexpected German probability support')
    return logp,{k:i for i,k in enumerate(keys)}


def categorical(logp,theta):
    d=read_json(WORK/'dictionary.json')
    f=np.log([a['freq'] for a in d]);l=np.log([len(a['surface'][0]) for a in d])
    f=np.r_[f-f.mean(),0.];l=np.r_[l-l.mean(),0.];o=np.r_[np.zeros(len(d)),1.]
    def evaluate(parameters):
        g=parameters[0]*logp+f*parameters[1]+l*parameters[2]+o*parameters[3]
        return g-logsumexp(g,axis=1,keepdims=True)
    return evaluate(theta) if theta is not None else (evaluate,f,l,o)


def fit_base(logp,ev):
    R,I,W=ev.T
    n=np.bincount(I,minlength=len(logp))
    counts=np.zeros_like(logp);np.add.at(counts,(I,W),1)
    evaluate,f,l,o=categorical(logp,None)
    ctr=np.array([1.,0.,0.,0.])
    def loss(p):
        z=evaluate(p);reg=p-ctr
        err=np.exp(z)*n[:,None]-counts
        df=np.array([np.einsum('ij,ij->',err,logp),err.sum(0)@f,err.sum(0)@l,err.sum(0)@o])
        return float(-(counts*z).sum()+0.5*(reg@reg)),df+reg
    res=minimize(loss,ctr,jac=True,method='L-BFGS-B',options={'gtol':1e-8,'ftol':1e-15,'maxiter':200})
    grad=float(np.max(abs(loss(res.x)[1]))/len(ev))
    if not res.success or grad>1e-6:raise ValueError(f'German source calibration failed, {res.message}: {grad}')
    return res.x,grad


def event_rows(which,idx):
    return np.asarray([[r,idx[k],w] for r,k,w in read_json(WORK/(which+'-observations.json'))],int)


def correlation_fit(loglik,participant,prior):
    q=np.exp(loglik-logsumexp(loglik+np.log(prior),axis=1,keepdims=True))*np.sqrt(prior)
    pieces=[]
    for who in np.unique(participant):
        points=q[participant==who]
        a,b=np.triu_indices(len(points),1)
        pieces.append(np.einsum('ij,ij->i',points[a],points[b]).astype(np.float32)-1)
    values=np.concatenate(pieces).astype(float)
    if np.any(values<=-1):raise ValueError('Invalid within-person pair')
    def grad(lam):return np.mean(values/(1+lam*values))
    lam=0. if grad(0)<=0 else (1. if grad(1)>=0 else brentq(grad,0.,1.,xtol=1e-12))
    return float(lam),int(len(values))


def fit(folder):
    l0,idx=native(folder)
    src=WORK/'training-observations.json';events=event_rows('training',idx)
    r,i,w=events.T;theta,grad=fit_base(l0,events)
    lp=categorical(l0,theta);odds,zn=conditionals(lp)
    flag=w==OTHER
    coeff,cgrad=calibrate(odds[i],flag);yes,no=binary(odds,coeff)
    cls=np.where(flag[:,None],yes[i],no[i]);nm=lp[i,w,None]*BETA-zn[i]
    modes={}
    for k,arr,mask in (('class',cls,np.ones(len(w),bool)),('name',nm,~flag)):
        prior,resp=population_weights(arr[mask],np.arange(mask.sum()))
        lam,pairs=correlation_fit(arr[mask],r[mask],prior)
        modes[k]=dict(prior=prior.tolist(),intraclass=lam,pair_trials=pairs,prior_kkt=resp['stationarity'])
    write_json(OUT/'auditory-fit.json',dict(protocol=digest(PLAN),human_source=digest(src),source_participants=int(len(np.unique(r))),
        source_exposures=int(len(r)),source_stems=int(len(np.unique(i))),source_named=int((~flag).sum()),base=theta.tolist(),
        base_kkt=grad,class_coeff=coeff.tolist(),class_kkt=cgrad,modes=modes,gpu=digest(folder/'receipt.json')))


def lock(folder):
    raw,idx=native(folder);m=read_json(OUT/'auditory-fit.json')
    if m['gpu']!=digest(folder/'receipt.json'):raise ValueError('Wrong German native source')
    lp=categorical(raw,np.asarray(m['base']));odds,zn=conditionals(lp)
    yes,no=binary(odds,np.asarray(m['class_coeff']))
    panel=read_json(WORK/'cohorts.json')
    vis={int(r) for r,mode in panel['readers'].items() if mode=='v'}
    shared=set(panel['stems']['a'])&set(panel['stems']['v']);targets=set(panel['stems']['v'])-shared
    test={idx[k] for k in targets}
    hist=event_rows('history',idx);r,i,w=hist.T
    if test&set(i) or set(r)-vis:raise ValueError('Held person/material leaked into calibration')
    geo=read_json(WORK/'geometry.json')
    allowed=defaultdict(set)
    for who in vis:
        for s in geo[str(who)]:
            k=sha256(s.encode()).hexdigest()
            if k in shared:allowed[who].add(idx[k])
    people=sorted(k for k in vis if (r==k).sum()>=20)
    other,overlap=match(people,allowed)
    orderkey={j:sha256(('German shared calibration '+k).encode()).digest() for k,j in idx.items()}
    groups=[sorted(hist[r==k].tolist(),key=lambda z:orderkey[z[1]]) for k in people]
    lengths=np.array([len(x) for x in groups]);shape=(len(people),max(lengths));items=np.zeros(shape,int);words=np.zeros(shape,int)
    for a,ev in enumerate(groups):
        for b,(_,item,word) in enumerate(ev):items[a,b]=item;words[a,b]=word
    live=np.arange(shape[1])<lengths[:,None];outside=words==OTHER
    pc=np.array(m['modes']['class']['prior']);pn=np.array(m['modes']['name']['prior'])
    c=online_priors(np.exp(np.where(outside[:,:,None],yes[items],no[items])),live,np.ones_like(live),pc,m['modes']['class']['intraclass'])
    z=online_priors(np.exp(lp[items,words,None]*BETA-zn[items]),live,~outside,pn,m['modes']['name']['intraclass'])
    file=WORK/'visual-conditional-forecasts.npz';op={k:j for j,k in enumerate(people)}
    arms=[];support=[]
    for limit in (None,20):
        cs=[];js=[];its=[];nn=[];peer_counts=[]
        for j,k in enumerate(people):
            d=op[other[k]];n=min(lengths[j],lengths[d]) if limit is None else limit
            use=(lengths>=n);use[j]=False
            if use.sum()<2:continue
            own=mixture(c[j,n].astype(float),z[j,n].astype(float));null=mixture(c[d,n].astype(float),z[d,n].astype(float))
            cp=c[use,n].astype(float);np_=z[use,n].astype(float);pooled=(cp.mean(0),cp.T@np_/len(cp))
            anchors=[own,pooled,null,mixture(pc,pn)]
            its.append(k);nn.append(int(n));cs.append([u for u,v in anchors]);js.append([v for u,v in anchors]);peer_counts.append(int(use.sum()))
        arms.append((its,cs,js));support.append(dict(people=its,history_sizes=nn,other_peers=peer_counts))
    np.savez_compressed(file,**{f'{k}_{a}':v for k,arm in enumerate(arms) for a,v in zip(('people','class','joint'),arm)},held=sorted(test))
    write_json(OUT/'human-forecast-lock.json',dict(weights=digest(OUT/'auditory-fit.json'),forecasts=digest(file),
         history=digest(WORK/'history-observations.json'),original_geometry=digest(WORK/'geometry.json'),exposure=digest(OUT/'exposure-seal.json'),
         studies=support,offered_shared_min=float(np.min(overlap)),offered_shared_avg=float(np.mean(overlap)),
         held_visual_stems=len(test),visual_target_answers_consumed=False,eligible_history_participants=len(people)))


def score(folder):
    raw,idx=native(folder)
    sealed=read_json(OUT/'human-forecast-lock.json');file=WORK/'visual-conditional-forecasts.npz'
    if digest(file)!=sealed['forecasts'] or sealed['weights']!=digest(OUT/'auditory-fit.json'):raise ValueError('Forecast file changed')
    m=read_json(OUT/'auditory-fit.json');lp=categorical(raw,m['base']);odds,zn=conditionals(lp)
    yes,no=binary(odds,np.asarray(m['class_coeff']))
    target=event_rows('target',idx);R,I,W=target.T
    studies={}
    with np.load(file) as f:
        if set(I)-set(f['held']):raise ValueError('Unexpected German held stimulus')
        for k,name in enumerate(('all_available','fixed_twenty')):
            E=[];L=[];O=[];native_nll=[];cal_nll=[]
            people=f[f'{k}_people'];C=f[f'{k}_class'];J=f[f'{k}_joint']
            for j,r in enumerate(people):
                sel=R==r
                if not np.any(sel):continue
                is_=I[sel];ws=W[sel];named=np.exp(lp[is_,ws,None]*BETA-zn[is_])
                vals,qout=probabilities(np.exp(yes[is_]),np.exp(no[is_]),named,zip(C[j],J[j],strict=True))
                loss=-np.log(np.where(ws[:,None]==OTHER,qout,vals))
                E.extend(target[sel]);L.extend(loss);O.extend(qout)
                native_nll.extend(-raw[is_,ws]);cal_nll.extend(-lp[is_,ws])
            e=np.asarray(E);l=np.asarray(L);o=np.asarray(O);marked=e.copy();marked[e[:,2]==OTHER,2]=8192
            res=report(marked,l,o);res['choice_content']=decompose(marked,l,o)
            res['raw_native_NLL']=float(np.mean(native_nll));res['source_calibrated_native_NLL']=float(np.mean(cal_nll))
            res['median_personal_history']=float(np.median(sealed['studies'][k]['history_sizes']))
            studies[name]=res
    write_json(OUT/'independent-reading-scores.json',dict(forecast=digest(OUT/'human-forecast-lock.json'),
       source=digest(OUT/'auditory-fit.json'),external_file=digest(WORK/'target-observations.json'),external=studies,
       introduced_person_profiles=0,usage='True cross-item conditioning across independently held visually presented stems and readers. Within-page answer chronology is unavailable.',
       keys={'same_history':'Own observed response-mode mixture','unknown_reader':'Mixture averaged across all equally supported other visual participants',
             'other_history':'One matched other visual reader','single_head':'Frozen German auditory-cohort mode law'}))
    for k,v in studies.items():print(k,v['readers'],v['scored_responses'],v['NLL'],v['paired'],flush=True)


if __name__=='__main__':
    p=ArgumentParser(description=__doc__);p.add_argument('action',choices=['fit','lock','score']);p.add_argument('native',type=Path)
    a=p.parse_args();{'fit':fit,'lock':lock,'score':score}[a.action](a.native)
