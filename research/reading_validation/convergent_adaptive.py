"""Fit endpoint-specific reading likelihoods with an empirical covariance among word readouts."""
from argparse import ArgumentParser
from pathlib import Path
import json
import numpy as np
from scipy.linalg import cholesky,cho_solve
from scipy.optimize import minimize
from scipy.special import betaln,digamma,expit,gammaln,logit
from scipy.stats import spearmanr
from research.backbone_transfer.extract import digest,read_json,stream_rows,write_json
from research.reading_validation.convergent_counts import study
from research.reading_validation.convergent_rating_counts import counts
from research.reading_validation.convergent_word_score import OUT as BASE,ROOT,assemble
from research.reading_validation.meaning_score import folds

OUT=BASE/'adaptive';PLAN=Path(__file__).with_name('convergent-adaptive-protocol.md');ARMS=('fixed','diagonal','shared');TARGETS=('N400_negativity','cloze_miss_count','rated_unpredictability')


def transform(X,train,context):
    surface=X[train,:6];mu=surface.mean(0);sd=np.maximum(surface.std(0),.02);parts=[np.ones((len(X),1)),(X[:,:6]-mu)/sd];rec={'mean':mu.tolist(),'sd':sd.tolist(),'context':context}
    if context:
        A=X[train,6:];mean=A.mean(0);s=np.maximum(A.std(0),.025);A=(A-mean)/s;v,U=np.linalg.eigh(A.T@A/len(A));i=np.argsort(v)[-24:][::-1];basis=U[:,i]/np.sqrt(np.maximum(v[i],.1));parts.append(((X[:,6:]-mean)/s)@basis);rec.update({'context_mean':mean.tolist(),'context_scale':s.tolist(),'basis':basis.tolist()})
    return np.column_stack(parts),rec


def labels():
    cloze,coh=study();ratings,rs=counts()
    if len(cloze)!=1726 or set(cloze)!=set(ratings):raise ValueError('Incompatible originally measured response counts')
    return {key:(cloze[key][0],cloze[key][0]-cloze[key][1],ratings[key][2]**2/ratings[key][0]) for key in cloze}


def bb(k,n,eta,log_precision,gradient=False):
    q=expit(np.clip(eta,-32.,32.));precision=np.exp(log_precision);a=precision*q;b=precision*(1.-q)
    y=gammaln(n+1)-gammaln(k+1)-gammaln(n-k+1)+betaln(k+a,n-k+b)-betaln(a,b)
    if not gradient:return y
    common=digamma(precision)-digamma(n+precision);h=digamma(k+a)-digamma(a)+common;j=digamma(n-k+b)-digamma(b)+common
    return y,precision*q*(1.-q)*(h-j),a*h+b*j


def curvature(eta,n,theta):
    # Exact model Fisher information. Each row sums every possible choice count for that observed list size.
    answer=np.zeros(len(n))
    for size in np.unique(n):
        pick=n==size;k=np.arange(int(size)+1)[None,:];lp,d,_=bb(k,size,eta[pick][:,None],theta,True);prob=np.exp(lp)
        if abs(prob.sum(1)-1).max()>1e-7:raise ArithmeticError('Discrete count distribution does not normalize')
        answer[pick]=(prob*d*d).sum(1)
    return answer


def likelihood(x,y,n,se2,B,h,with_gradient):
    eta=x@B;v0=np.exp(2*h[0]);s0=v0;v2=np.exp(2*h[2]);s2=v2+se2
    err0=eta[:,0]-y[:,0];err2=eta[:,2]-y[:,2];a=.5*(np.log(2*np.pi*s0)+err0**2/s0);c=.5*(np.log(2*np.pi*s2)+err2**2/s2)
    if not with_gradient:return np.c_[a,-bb(y[:,1],n,eta[:,1],h[1]),c]
    ll,d,dc=bb(y[:,1],n,eta[:,1],h[1],True);g=x.T@np.c_[err0/s0,-d,err2/s2];gh=np.r_[(1-err0**2/s0).sum(),-dc.sum(),((1-err2**2/s2)*v2/s2).sum()]
    return (a-ll+c).sum(),g,gh


def mapfit(x,y,n,se2,prior,warm):
    d=x.shape[1];inv=np.linalg.inv(prior)
    if warm is None:
        B=np.zeros((d,3));B[0,1]=logit((y[:,1].sum()+.5)/(n.sum()+1));warm=np.r_[B.ravel(),0.,np.log(16.),0.]
    bounds=[(None,None)]*(d*3)+[(-4.,2.3),(np.log(.6),np.log(2000.)),(-4.,2.3)]
    def objective(vec):
        B=vec[:d*3].reshape(d,3);h=vec[d*3:];val,g,gh=likelihood(x,y,n,se2,B,h,True);q=B[1:]@inv;val+=.5*(B[1:]*q).sum();g[1:]+=q
        return val/len(x),np.r_[g.ravel(),gh]/len(x)
    opt=minimize(objective,warm,jac=True,method='L-BFGS-B',bounds=bounds,options=dict(maxiter=450,ftol=2e-13,gtol=2e-7,maxls=40))
    gr=objective(opt.x)[1];pg=gr.copy()
    for j,(lo,hi) in enumerate(bounds):
        if lo is not None and opt.x[j]<=lo+1e-8 and gr[j]>0:pg[j]=0
        if hi is not None and opt.x[j]>=hi-1e-8 and gr[j]<0:pg[j]=0
    station=float(abs(pg).max())
    if not opt.success or station>9e-6:raise ArithmeticError(f'MAP stationary criterion failed: {station}, {opt.message}')
    B=opt.x[:3*d].reshape(d,3);h=opt.x[3*d:];etas=x@B;curv=[np.full(len(x),np.exp(-2*h[0])),curvature(etas[:,1],n,h[1]),1/(np.exp(2*h[2])+se2)]
    info=np.zeros((3*d,3*d));info[3:,3:]+=np.kron(np.eye(d-1),inv)
    for j,v in enumerate(curv):info[j::3,j::3]+=(x*v[:,None]).T@x
    L=cholesky(info,lower=True);V=cho_solve((L,True),np.eye(3*d));
    expected=B[1:].T@B[1:]+sum(V[3*k:3*k+3,3*k:3*k+3] for k in range(1,d))
    return opt.x,expected,dict(objective=float(opt.fun),stationarity=station,iterations=opt.nit)


def fit(x,raw,n,sampling,arm):
    y=raw.copy();mu=np.array([y[:,0].mean(),0.,y[:,2].mean()]);scale=np.array([y[:,0].std(),1.,y[:,2].std()]);y=(y-mu)/scale;se2=sampling/scale[2]**2
    # Shrink towards three independent N(0, .35^2) coefficient margins, with eight equivalent coefficient rows.
    base=.35**2*np.eye(3);prior=base.copy();warm=None;steps=[];strength=8
    for outer in range(130):
        warm,em,rec=mapfit(x,y,n,se2,prior,warm);steps.append(rec)
        if arm=='fixed':break
        update=(em+strength*base)/(x.shape[1]-1+strength)
        if arm=='diagonal':update=np.diag(np.diag(update))
        delta=np.linalg.norm(update-prior)/(np.linalg.norm(prior)+1e-8);steps[-1]['relative_covariance_update']=float(delta)
        if delta<.0008:break
        prior=update
    else:raise ArithmeticError(f'Empirical covariance did not settle: {delta}')
    return dict(B=warm[:-3].reshape(x.shape[1],3).tolist(),h=warm[-3:].tolist(),mu=mu.tolist(),scale=scale.tolist(),prior=prior.tolist(),iterations=len(steps),max_gradient=max(z['stationarity'] for z in steps),last=steps[-1],task='fixed' if arm=='fixed' else 'empirical '+arm)


def forecast(x,y,n,sampling,m):
    B=np.array(m['B']);h=np.array(m['h']);mu=np.array(m['mu']);scale=np.array(m['scale']);pred=x@B;pred[:,1]=expit(pred[:,1]);pred=pred*scale+mu
    terms=likelihood(x,(y-mu)/scale,n,sampling/scale[2]**2,B,h,False)+np.log(scale)
    if not np.isfinite(terms).all():raise ArithmeticError('Non-finite held response likelihood')
    return pred,terms


def train(folder):
    if digest(PLAN)!=read_json(OUT/'registration.json')['plan']:raise ValueError('Study registration changed')
    (_,_,_,X,H,c,_,_),_=assemble(folder);table=labels();o=np.array([table[int(i)] for i in c[:,1]]);n=o[:,0];y=H.copy();y[:,1]=o[:,1];sampling=o[:,2];ids=c[:,0];split=folds(ids,'UCL 2026 reading sentence:');fold=np.array([split[s] for s in ids]);source=[]
    for ctype in ('surface','context'):
        for f in range(6):
            cache=OUT/(ctype+f'-fold-{f}.json')
            if cache.exists():source.append(digest(cache));continue
            tr=np.ones(len(ids),dtype=bool) if f==5 else fold!=f;te=~tr;z,prep=transform(X,tr,ctype=='context');fits={};P={};L={}
            for arm in ARMS:
                m=fit(z[tr],y[tr],n[tr],sampling[tr],arm);fits[arm]=m
                if te.any():
                    p,l=forecast(z[te],y[te],n[te],sampling[te],m);P[arm]=p.tolist();L[arm]=l.tolist()
            write_json(cache,dict(plan=digest(PLAN),training_sentences=len(set(ids[tr])),held_articles=sorted(set(ids[te])),held_items=c[te,1].astype(int).tolist(),raw_y=y[te].tolist(),trial_n=n[te].astype(int).tolist(),projection=prep,models=fits,predictions=P,held_likelihood=L));print('Converged',ctype,f,[(arm,m['iterations']) for arm,m in fits.items()],flush=True);source.append(digest(cache))
    write_json(OUT/'fit-seal.json',dict(plan=digest(PLAN),file_sha=source,program=digest(Path(__file__)),sentences=len(set(ids)),words=len(ids),cloze_trials=int(n.sum())))


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('folder',type=Path);a=p.parse_args();train(a.folder)
