"""Fit evidence adequacy to genuine individual ballots and pairwise judgments of comparable arguments."""
import argparse
from collections import Counter,defaultdict
import csv
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize,minimize_scalar
from scipy.special import expit,logsumexp,softmax
from research.reading_validation.memory_data import h,sha
from research.reading_validation.reason_quality_data import DATA,OUT,ukp,experts,DIMENSIONS
from research.reading_validation.reason_quality_features import Material
from research.reading_validation.state_words import standard

EST=Path(__file__).with_name('reason-quality-estimator.md')

def observations(material,partition):
    use=material.index;index=material.parts;binary=[];links=defaultdict(set)
    for r in ukp(True):
        k=h(r['text'])
        if k in use and index['ukp'].get(k)==partition:binary.append([k,r['family'],sum(r['votes']),len(r['votes'])])
    by={tuple(r['id']):r for r in material.meta['webis'] if index['webis'].get(r['key'])==partition and r['key'] in use}
    comparisons=defaultdict(lambda:np.zeros(3,int))
    with (DATA/'webis-choice.csv').open(newline='') as f:
        for r in csv.DictReader(f):
            a=by.get((r['Discussion ID A'],r['Argument ID A']));b=by.get((r['Discussion ID B'],r['Argument ID B']))
            if a is None or b is None or a['key']==b['key']:continue
            if a['topic']!=b['topic']:raise ValueError('A retained vote compares different debates')
            key=(a['key'],b['key'],a['topic']);comparisons[key][('A','B','Tie').index(r['Comparison'])]+=1
    return binary,[[a,b,g,counts] for (a,b,g),counts in comparisons.items()]

def weights(group,amount):
    tot=defaultdict(float)
    for g,v in zip(group,amount,strict=True):tot[g]+=v
    return np.array([1/(len(tot)*tot[g]) for g in group])

class Measures:
    def __init__(self,obs,z,ix):
        one,two=obs
        if not one or not two:raise ValueError('Each study needs real training and calibration examples')
        self.z1=z[[ix[k] for k,_,_,_ in one]];self.y=np.array([r[2] for r in one]);self.n=np.array([r[3] for r in one]);self.w1=weights([r[1] for r in one],self.n)
        self.d=.5*np.stack([z[ix[a]]-z[ix[b]] for a,b,_,_ in two]);self.votes=np.stack([r[3] for r in two]);self.w2=weights([r[2] for r in two],self.votes.sum(1))

    def one(self,v):
        x=self.z1@v;return self.w1@(self.n*np.logaddexp(0,x)-self.y*x), self.z1.T@(self.w1*(self.n*expit(x)-self.y))

    def two(self,v,t):
        s=self.d@v;z=np.column_stack((s,-s,np.full(len(s),t)));lp=z-logsumexp(z,axis=1,keepdims=True)
        deriv=self.w2[:,None]*(np.exp(lp)*self.votes.sum(1)[:,None]-self.votes)
        return -self.w2@(self.votes*lp).sum(1),self.d.T@(deriv[:,0]-deriv[:,1]),deriv[:,2].sum()

    def fit(self,mode):
        size=self.z1.shape[1]
        # Separate coefficients and joint shrinkage represent the same two response distributions.
        def objective(v):
            a=v[:size];b=v[size:-1];t=v[-1];l1,g1=self.one(a);l2,g2,dt=self.two(b,t)
            reg=.0125*(np.sum(a[1:]**2)+np.sum(b[1:]**2));g1=.5*g1;g2=.5*g2;g1[1:]+=.025*a[1:];g2[1:]+=.025*b[1:]
            # A paired comparison cannot estimate a common intercept.
            reg+=.5*b[0]**2;g2[0]+=b[0]
            if mode=='joint':
                delta=a[1:]-b[1:];reg+=.075*(delta@delta);g1[1:]+=.15*delta;g2[1:]-=.15*delta
            return .5*(l1+l2)+reg,np.r_[g1,g2,.5*dt]
        op=minimize(objective,np.zeros(2*size+1),jac=True,method='L-BFGS-B',options={'maxiter':900,'gtol':1e-8,'ftol':1e-13,'maxcor':16})
        if not op.success or abs(op.jac).max()>1e-5:raise ArithmeticError('Joint logical and binary ballots did not converge')
        return op.x[:size],op.x[size:-1],float(op.x[-1]),dict(iterations=int(op.nit),gradient=float(abs(op.jac).max()),objective=float(op.fun))

def expert_folds(records):
    topics=sorted({r['topic'] for r in records},key=lambda t:h('dagstuhl-four-fold-'+t));return {g:i%4 for i,g in enumerate(topics)}

def ordinal(z,v):
    score=z@v[:-1];gap=np.logaddexp(0.,v[-1])+.05;low=expit(-score);high=expit(gap-score)
    p=np.column_stack((low,high-low,1-high));g0=low*(1-low);g1=high*(1-high)
    return p.clip(1e-14),np.column_stack((-g0,g0-g1,g1)),np.column_stack((np.zeros(len(g1)),g1,-g1))*expit(v[-1])

def adapt(z,y,wt):
    def objective(v):
        p,ds,dg=ordinal(z,v);cd=-wt[:,None]*y/p;grad=z.T@(cd*ds).sum(1);grad[1:]+=.04*v[1:-1]
        return -(wt@(y*np.log(p)).sum(1))+.02*(v[1:-1]@v[1:-1]), np.r_[grad,(cd*dg).sum()]
    fit=minimize(objective,np.zeros(z.shape[1]+1),jac=True,method='L-BFGS-B',options={'maxiter':650,'gtol':1e-8,'ftol':1e-13})
    if not fit.success or abs(fit.jac).max()>2e-5:raise ArithmeticError('Expert ordinal fit did not converge')
    return fit.x,float(abs(fit.jac).max())

def run(mode,archive):
    dest=OUT/('forecast-'+mode);dest.mkdir(exist_ok=True)
    if (dest/'seal.json').exists():raise FileExistsError('Judgment forecasts already sealed')
    material=Material(archive);data={p:observations(material,p) for p in ('train','cal')};arrays={};fitlog={};params={};kix=material.index
    pool={k for kk in material.parts.values() for k,split in kk.items() if split=='train' and k in kix};src=[kix[k] for k in sorted(pool)]
    expert_rows=material.expert;ek=[r['key'] for r in expert_rows];exposures=np.array([kix[k] for k in ek]);topics=[r['topic'] for r in expert_rows]
    mapping=expert_folds(expert_rows);material_key={r['record']:r['key'] for r in expert_rows}
    for arm,xx in material.x.items():
        mu,sd=standard(xx[src],np.ones(len(src)));z=np.column_stack((np.ones(len(xx)),(xx-mu)/sd));fitter=Measures(data['train'],z,kix);tune=Measures(data['cal'],z,kix)
        fitted={m:fitter.fit(m) for m in ('isolated','joint')};fitlog[arm]={m:v[-1] for m,v in fitted.items()}
        params[arm]=dict(mu=mu.tolist(),sd=sd.tolist(),fits={m:dict(binary=v[0].tolist(),paired=v[1].tolist(),tie=v[2]) for m,v in fitted.items()})
        u={m:z@v[0] for m,v in fitted.items()};paired={m:z@v[1] for m,v in fitted.items()}
        q=expit(tune.z1@fitted['isolated'][0]);w=expit(tune.z1@fitted['joint'][0]);nd=len({row[1] for row in data['cal'][0]})
        def calibration(alpha):
            p=q*(1-alpha)+w*alpha
            return -tune.w1@(tune.y*np.log(p)+(tune.n-tune.y)*np.log1p(-p))-(np.log(alpha)+np.log1p(-alpha))/nd
        op=minimize_scalar(calibration,bounds=(1e-6,1-1e-6),method='bounded');a=float(op.x)
        pp=(1-a)*expit(u['isolated'])+a*expit(u['joint']);score=np.log(pp/(1-pp));fitlog[arm]['binary_shared_fraction']=a
        for m,v in u.items():arrays[arm+'.'+m]=v.astype('float32')
        for m,v in paired.items():arrays[arm+'.paired-'+m]=v.astype('float32')
        arrays[arm+'.adaptive']=score.astype('float32')
        for dim in DIMENSIONS:
            for approach in ('source_only','expert_only','expert_plus_source'):
                arrays[arm+'.'+dim+'.'+approach]=np.full((len(ek),3),np.nan)
        # For each adaptation fold, access scores only for other debates.
        for fold in range(4):
            train=np.array([mapping[g]!=fold for g in topics]);valid=~train;train_ids={r['record'] for r,keep in zip(expert_rows,train,strict=True) if keep}
            labels=expert_votes(train_ids);mdev=xx[exposures][train];sx=score[exposures,None]
            use_mu,use_sd=standard(mdev,np.ones(len(mdev)));feat=(xx[exposures]-use_mu)/use_sd
            sm=sx[train].mean();ss=max(1e-5,sx[train].std());sx=(sx-sm)/ss
            for dim in DIMENSIONS:
                yy=np.stack([np.bincount(np.array(labels[rec['record']][dim])-1,minlength=3) for rec,ok in zip(expert_rows,train,strict=True) if ok])
                wt=weights([t for t,keep in zip(topics,train,strict=True) if keep],yy.sum(1))
                for approach,matrix in [('source_only',sx),('expert_only',feat),('expert_plus_source',np.column_stack((sx,feat)))]:
                    x=np.column_stack((np.ones(len(ek)),matrix));v,err=adapt(x[train],yy,wt)
                    arrays[arm+'.'+dim+'.'+approach][valid]=ordinal(x[valid],v)[0]
                    fitlog[arm]['max_expert_gradient']=max(fitlog[arm].get('max_expert_gradient',0),err)
    if any(not np.isfinite(a).all() for a in arrays.values()):raise ValueError('Incomplete whole-argument predictions')
    np.savez_compressed(dest/'forecasts.npz',**arrays)
    (dest/'fits.json').write_text(json.dumps(dict(fits=fitlog,models=params,group_folds=mapping,expert=ek,keys=material.keys),separators=(',',':'))+'\n')
    (dest/'seal.json').write_text(json.dumps(dict(protocol=sha(EST),material=sha(OUT/'material.json'),native=material.certificate,
          files={f:sha(dest/f) for f in ('forecasts.npz','fits.json')},train_and_cal={p:dict(direct=len(v[0]),pair_sets=len(v[1]),comparative_votes=int(sum(row[3].sum() for row in v[1]))) for p,v in data.items()}),indent=2)+'\n')
    print('Sealed forecasts for ordinary passages and topic-held experts',len(ek),fitlog,flush=True)

def expert_votes(ids):
    return {r['record']:r['votes'] for r in experts(True,ids)}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('native','form'));p.add_argument('--archive',type=Path);run(**vars(p.parse_args()))
