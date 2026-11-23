"""Fit partially observed word-response laws using only measured EEG likelihood and information."""
from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
import numpy as np
from scipy.linalg import cho_factor,cho_solve
from scipy.optimize import minimize
from scipy.special import logit
from scipy.stats import spearmanr
from research.backbone_transfer.extract import digest,read_json,write_json
from research.reading_validation.convergent_adaptive import ARMS,BASE,assemble,bb,curvature,folds,forecast,labels,transform

OUT=BASE/'sparse'
PLAN=Path(__file__).with_name('convergent-sparse-protocol.md')


def map_fit(x,y,n,se2,observed,prior,warm):
    d=x.shape[1];inv=np.linalg.inv(prior);eeg_x=x[observed];eeg_y=y[observed,0]
    bounds=[(None,None)]*(d*3)+[(-4.,2.3),(np.log(.6),np.log(2000.)),(-4.,2.3)]
    if warm is None:
        B=np.zeros((d,3));B[0,1]=logit((y[:,1].sum()+.5)/(n.sum()+1));warm=np.r_[B.ravel(),0.,np.log(16.),0.]
    def objective(vec):
        B=vec[:-3].reshape(d,3);h=vec[-3:];eta=x@B;v0=np.exp(2*h[0]);v2=np.exp(2*h[2]);s2=v2+se2
        e0=eta[observed,0]-eeg_y;e2=eta[:,2]-y[:,2];lc,deta,dlp=bb(y[:,1],n,eta[:,1],h[1],True)
        oe=.5*(np.log(2*np.pi*v0)+e0**2/v0)
        orating=.5*(np.log(2*np.pi*s2)+e2**2/s2)
        mapped=B[1:]@inv;cost=oe.sum()+orating.sum()-lc.sum()+.5*(B[1:]*mapped).sum()
        g=np.column_stack([eeg_x.T@(e0/v0),x.T@(-deta),x.T@(e2/s2)]);g[1:]+=mapped
        gh=[(1-e0**2/v0).sum(),-dlp.sum(),((1-e2**2/s2)*v2/s2).sum()]
        return cost/len(x),np.r_[g.ravel(),gh]/len(x)
    opt=minimize(objective,warm,jac=True,method='L-BFGS-B',bounds=bounds,options=dict(maxiter=650,maxls=40,gtol=1e-7,ftol=5e-14))
    g=opt.jac.copy()
    for i,(lo,hi) in enumerate(bounds):
        if (lo is not None and opt.x[i]<lo+1e-8 and g[i]>0) or (hi is not None and opt.x[i]>hi-1e-8 and g[i]<0):g[i]=0
    station=float(abs(g).max())
    if not opt.success or station>8e-6:raise ArithmeticError(f'Sparse observation fit did not converge: {station}')
    B=opt.x[:-3].reshape(d,3);h=opt.x[-3:];eta=x@B
    cur=[observed*np.exp(-2*h[0]),curvature(eta[:,1],n,h[1]),1/(np.exp(2*h[2])+se2)]
    H=np.zeros((3*d,3*d));H[3:,3:]=np.kron(np.eye(d-1),inv)
    for j,v in enumerate(cur):H[j::3,j::3]+=(x*v[:,None]).T@x
    V=cho_solve(cho_factor(H,lower=True),np.eye(3*d))
    moment=B[1:].T@B[1:]+sum(V[3*j:3*j+3,3*j:3*j+3] for j in range(1,d))
    return opt.x,moment,station


def fit(x,raw,n,sampling,obs,arm):
    if np.isfinite(raw[~obs,0]).any() or not np.isfinite(raw[obs,0]).all():raise ValueError('EEG holdout or missing values were not correctly applied')
    mu=np.array([raw[obs,0].mean(),0.,raw[:,2].mean()]);scale=np.array([raw[obs,0].std(),1.,raw[:,2].std()])
    y=(raw-mu)/scale;se2=sampling/scale[2]**2
    base=np.eye(3)*.35**2;prior=base.copy();warm=None;errors=[]
    for j in range(130):
        warm,moment,station=map_fit(x,y,n,se2,obs,prior,warm);errors.append(station)
        if arm=='fixed':break
        update=(moment+8*base)/(x.shape[1]-1+8)
        if arm=='diagonal':update=np.diag(np.diag(update))
        if np.linalg.norm(update-prior)/(np.linalg.norm(prior)+1e-8)<.0008:break
        prior=update
    else:raise ArithmeticError('Sparse shared covariance did not settle')
    return dict(B=warm[:-3].reshape(x.shape[1],3).tolist(),h=warm[-3:].tolist(),mu=mu.tolist(),scale=scale.tolist(),prior=prior.tolist(),iterations=len(errors),gradient=max(errors))


def main(folder):
    if read_json(OUT/'registration.json')['protocol']!=digest(PLAN):raise ValueError('Sparse registration changed')
    (_,_,_,X,H,c,_,_),_=assemble(folder);table=labels();o=np.asarray([table[int(i)] for i in c[:,1]]);n=o[:,0];Y=H.copy();Y[:,1]=o[:,1];ids=c[:,0];partition=folds(ids,'UCL 2026 reading sentence:');fold=np.array([partition[k] for k in ids])
    P={};L={};books=[]
    for typ in ('surface','context'):
        for arm in ARMS:
            name=typ+'_'+arm;P[name]=np.full_like(Y,np.nan);L[name]=np.full_like(Y,np.nan)
        for f in range(5):
            tr=fold!=f;te=~tr;selected=sorted(set(ids[tr]),key=lambda s:sha256(('CogLLM EEG small assay '+s).encode()).digest())[:10]
            observed=np.isin(ids[tr],selected);sparse=Y[tr].copy();sparse[~observed,0]=np.nan;z,transformer=transform(X,tr,typ=='context');fits={}
            path=OUT/f'{typ}-fold-{f}.json'
            for arm in ARMS:
                m=fit(z[tr],sparse,n[tr],o[tr,2],observed,arm);fits[arm]=m
                P[typ+'_'+arm][te],L[typ+'_'+arm][te]=forecast(z[te],Y[te],n[te],o[te,2],m)
            write_json(path,dict(selected_sentences=selected,observed_EEG_words=int(observed.sum()),held_sentences=sorted(set(ids[te])),models=fits,transform=transformer,held_items=c[te,1].astype(int).tolist(),forecasts={a:P[typ+'_'+a][te].tolist() for a in ARMS}))
            books.append(digest(path));print('Converged',typ,f,'EEG words',observed.sum(),flush=True)
    gain={};stats={};truth=Y.copy();truth[:,1]/=n;groups,g=np.unique(ids,return_inverse=True);count=np.bincount(g)
    for name,p in P.items():
        stats[name]=dict(rho=[float(spearmanr(v,y).statistic) for v,y in zip(p.T,truth.T)],rmse=np.sqrt(((p-truth)**2).mean(0)).tolist(),nll=L[name].mean(0).tolist())
    generator=np.random.default_rng(43817);w=generator.multinomial(len(groups),np.ones(len(groups))/len(groups),size=10000)
    for a,b in [('context_shared','context_diagonal'),('context_shared','context_fixed'),('context_shared','surface_shared'),('surface_shared','surface_diagonal')]:
        d=L[b]-L[a];summed=np.vstack([d[g==i].sum(0) for i in range(len(groups))]);v=(w@summed)/(w@count)[:,None]
        gain[a+'__over__'+b]=dict(mean=d.mean(0).tolist(),interval=np.quantile(v,[.025,.975],axis=0).T.tolist())
    write_json(OUT/'scores.json',dict(registration=digest(OUT/'registration.json'),program=digest(Path(__file__)),fold_checkpoints=books,primary='context_shared__over__context_diagonal',observed_units=len(g),held_sentences=len(groups),models=stats,gains=gain))
    print(gain['context_shared__over__context_diagonal'])


if __name__=='__main__':
    p=ArgumentParser(description=__doc__);p.add_argument('folder',type=Path);main(p.parse_args().folder)
