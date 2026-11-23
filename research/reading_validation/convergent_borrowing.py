"""Selective borrowing with restricted marginal likelihood on sparsely measured EEG words."""
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
from scipy.linalg import cho_factor,cho_solve
from scipy.optimize import minimize
from scipy.stats import spearmanr
from research.backbone_transfer.extract import digest,read_json,write_json
from research.reading_validation.convergent_adaptive import BASE,assemble,transform
from research.reading_validation.convergent_word_score import predict
from research.reading_validation.meaning_score import folds

OUT=BASE/'borrowing'
PLAN=Path(__file__).with_name('convergent-borrowing-protocol.md')
ARMS={'surface':(), 'native':('native',), 'behavior':('native','behavior'), 'physiology_behavior':('native','behavior','physiology'), 'early_control':('native','behavior','early')}


def restrict(fixed,components,y,observed,test):
    # A constant property in a small panel cannot identify its slope. Preserve
    # the intercept and estimable fixed effects in their original, signed order.
    keep=[]
    for j in range(fixed.shape[1]):
        if np.linalg.matrix_rank(fixed[observed][:,keep+[j]])>len(keep):keep.append(j)
    x=fixed[observed][:,keep];raw=y[observed];untested=fixed[test][:,keep];n,p=x.shape
    matrices=np.stack([np.eye(n),*[a[observed]@a[observed].T for a in components]])
    def evaluate(variance,save=False):
        V=np.tensordot(variance,matrices,axes=(0,0));fac=cho_factor(V,lower=True)
        vi=cho_solve(fac,np.eye(n));vix=vi@x;normal=cho_factor(x.T@vix,lower=True);s=cho_solve(normal,np.eye(p))
        b=s@(vix.T@raw);alpha=vi@(raw-x@b);P=vi-vix@s@vix.T
        val=.5*((n-p)*np.log(2*np.pi)+2*np.log(np.diag(fac[0])).sum()+2*np.log(np.diag(normal[0])).sum()+raw@alpha)
        grad=.5*np.einsum('ijk,jk->i',matrices,P-np.outer(alpha,alpha))
        return (val/n,grad/n) if not save else (vi,b,alpha,s,vix)
    initial=np.r_[max(raw.var(),.01),np.full(len(components),.05*max(raw.var(),.01))]
    bounds=[(1e-6,None)]+[(0.,None)]*len(components)
    opt=minimize(evaluate,initial,jac=True,method='L-BFGS-B',bounds=bounds,options=dict(gtol=1e-8,ftol=1e-14,maxiter=400,maxls=50))
    grad=opt.jac.copy()
    for j,(low,_) in enumerate(bounds):
        if opt.x[j]<=low+1e-8 and grad[j]>0:grad[j]=0
    if not opt.success or abs(grad).max()>2e-5:raise ArithmeticError(f'Restricted covariance fit failed: {opt.message} {grad}')
    vi,b,alpha,s,vix=evaluate(opt.x,True);cov=np.zeros((int(test.sum()),n));diagonal=np.full(int(test.sum()),opt.x[0])
    for a,var in zip(components,opt.x[1:]):
        cov+=var*(a[test]@a[observed].T);diagonal+=var*(a[test]**2).sum(1)
    offset=untested-cov@vix;mean=untested@b+cov@alpha
    variance=diagonal-(cov@vi*cov).sum(1)+(offset@s*offset).sum(1)
    if np.any(variance<=0):raise ArithmeticError('Gaussian predictive variance failed')
    return mean,variance,dict(estimable_fixed_columns=keep,variance_components=opt.x.tolist(),restricted_objective=float(opt.fun),gradient=float(abs(grad).max()),iterations=int(opt.nit),fixed_coefficients=b.tolist(),random_coefficients=[(v*(a[observed].T@alpha)).tolist() for v,a in zip(opt.x[1:],components)])


def main(folder):
    if digest(PLAN)!=read_json(OUT/'registration.json')['protocol']:raise ValueError('Study registered material changed')
    (_,_,_,X,H,c,_,_),_=assemble(folder);Y=H[:,0];ids=c[:,0];part=folds(ids,'UCL 2026 reading sentence:');fold=np.array([part[k] for k in ids])
    m=read_json(BASE/'scores.json')['external_spanish_no_updates']['spanish_coefficients']['context'];span=predict(X,m)
    P={k:np.full(len(X),np.nan) for k in ARMS};V={k:v.copy() for k,v in P.items()}
    for f in range(5):
        tr=fold!=f;te=~tr;saved=read_json(BASE/f'sparse/context-fold-{f}.json')
        fit=np.isin(ids,saved['selected_sentences'])
        if np.any(te&fit):raise ValueError('Held outcome leakage')
        z,prep=transform(X,tr,True);s=z[:,:7];b=np.array(saved['models']['diagonal']['B'])
        derived={'native':z[:,7:],'behavior':(z@b)[:,1:],'physiology':span[:,:1],'early':span[:,1:]}
        residual={};residualization={}
        for k,a in derived.items():
            surface=np.linalg.lstsq(s[tr],a[tr],rcond=None)[0];v=a-s@surface
            scale=np.sqrt((v[tr]**2).sum(1).mean())
            if scale<=1e-7:raise ValueError('Variance component has no independent directions')
            residual[k]=v/scale;residualization[k]=dict(surface=surface.tolist(),scale=float(scale))
        models={}
        for k,comps in ARMS.items():
            p,v,model=restrict(s,[residual[q] for q in comps],Y,fit,te);P[k][te]=p;V[k][te]=v;models[k]=model
        write_json(OUT/f'fold-{f}.json',dict(held_ids=c[te,1].astype(int).tolist(),eeg_sentences=saved['selected_sentences'],models=models,projection=prep,residualization=residualization,behavior_source=digest(BASE/f'sparse/context-fold-{f}.json'),forecasts={k:np.column_stack((P[k][te],V[k][te])).tolist() for k in P}))
    nll={k:.5*(np.log(2*np.pi*V[k])+(P[k]-Y)**2/V[k]) for k in P}
    group,gi=np.unique(ids,return_inverse=True);size=np.bincount(gi);rng=np.random.default_rng(1027);w=rng.multinomial(len(group),np.ones(len(group))/len(group),size=10000);comparison={}
    for k,arr in nll.items():
        delta=arr-nll['physiology_behavior'];summed=np.bincount(gi,weights=delta);boot=w@summed/(w@size)
        comparison[k]=dict(gain=float(delta.mean()),interval=np.quantile(boot,[.025,.975]).tolist())
    scores={k:dict(nll=float(nll[k].mean()),rho=float(spearmanr(P[k],Y).statistic),rmse=float(np.mean((Y-P[k])**2)**.5),coverage_95=float(np.mean(abs(Y-P[k])<=1.96*np.sqrt(V[k])))) for k in P}
    write_json(OUT/'scores.json',dict(registration=digest(OUT/'registration.json'),folds=[digest(OUT/f'fold-{f}.json') for f in range(5)],primary='physiology_behavior over surface',units=len(ids),sentences=len(group),full_over=comparison,scores=scores));print(scores,comparison)


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('folder',type=Path);main(p.parse_args().folder)
