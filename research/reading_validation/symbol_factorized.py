"""Fit separate source-eligibility, evidence-boundary, and competing-identity likelihoods."""
from argparse import ArgumentParser
from bisect import bisect_left
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit,logsumexp,softmax
from research.backbone_transfer.extract import digest,read_json,write_json
from research.reading_validation.symbol_model import Bank,case_geometry
from research.reading_validation.symbol_features import canonical
from research.reading_validation.symbol_evaluate import ARMS,THRESHOLDS,weights,score_events,decode,probabilities,metric
from research.reading_validation.symbol_geometry import OUT

METHODS=('surface','native')
LIMITS=np.r_[0.,np.geomspace(.00005,.8,22)]
PLAN=Path(__file__).with_name('symbol-factorization-protocol.md')


def event_design(page):
    t=np.arange(page.n);prev=np.maximum(t-1,0);base=page.base;state=page.state
    x=np.column_stack((base,base[prev],np.log1p(np.arange(page.n)),state,page.window,state-state[prev]))
    return x.astype('float32'),2*base.shape[1]+1


def available(page):
    return np.array([bisect_left(page.end,m.committed) for m in page.mentions],int)


def eligibility_design(page):
    abspos=available(page)
    if not len(abspos):return np.zeros((0,0)),0
    rows=[]
    for t,m in zip(abspos,page.mentions):
        s,q=page.features(m,[int(t)]);a,b=page.token_span(m.start,m.end)
        rows.append(np.r_[s[0],page.state[a:b].mean(0),page.state[t]])
    return np.asarray(rows,'float32'),s.shape[1]


def conditional_design(page,t,method,old):
    """Source-specific features at identical visible evidence, including the trained first-study score."""
    out=[];n=len(t);seen=available(page)
    for m,at in zip(page.mentions,seen):
        valid=t>=at;f=np.zeros((n,11),dtype='float32')
        if valid.any():
            s,h=page.features(m,t[valid]);a,b=page.token_span(m.start,m.end);k=ARMS.index(method)
            raw=np.column_stack((s,h))@old[0][:,k]+old[1][k]
            f[valid,:7]=np.column_stack([raw,s[:,9:11],s[:,12:14],s[:,17],s[:,19]])
            if method=='native':f[valid,7:]=h[:,[0,1,5,7]]
        out.append(f)
    return np.asarray(out),seen


def collect(bank,old,limit=None):
    A=[];Ay=[];E=[];Ey=[];papers_a=[];papers_e=[];ranks={s:[] for s in METHODS}
    for page,doc in bank.pages('fit',limit):
        true=case_geometry(page,doc);zone=np.zeros(page.n,dtype=int);g={(m.start,m.end):t for m,_,t in true}
        for m,_,t in true:zone[t]=2
        for m,_,t in true:zone[t[0]]=1
        X,e_cut=event_design(page);E.append(X);Ey.extend(zone);papers_e.extend([doc['paper']]*page.n)
        if not page.mentions:continue
        X,a_cut=eligibility_design(page);A.append(X);marks=[(m.start,m.end) for m in page.mentions];Ay.extend([k in g for k in marks]);papers_a.extend([doc['paper']]*len(X))
        ix={k:i for i,k in enumerate(marks)};times=sorted({int(tt) for _,_,t in true for tt in np.unique([t[0],t[len(t)//2],t[-1]])})
        if not times:continue
        times=np.array(times);candidates=available(page)
        for method in METHODS:
            F,_=conditional_design(page,times,method,old)
            for m,_,ts in true:
                gold=ix.get((m.start,m.end))
                if gold is None:continue
                for tt in np.unique([ts[0],ts[len(ts)//2],ts[-1]]):
                    active=[j for j,k in enumerate(marks) if candidates[j]<=tt and not(k in g and g[k][-1]<tt)]
                    if gold not in active or len(active)<2:continue
                    rows=F[active,np.searchsorted(times,tt)];ranks[method].append((rows,active.index(gold),doc['paper']))
    return (np.vstack(A),np.asarray(Ay,int),papers_a,a_cut),(np.vstack(E),np.asarray(Ey),papers_e,e_cut),ranks


def classfit(tr,method,classes):
    X,y,papers,cut=tr;X=X[:,:cut] if method=='surface' else X
    # Use the actual class frequencies in each paper; normalization preserves calibrated conditional event rates.
    count=Counter(papers);w=np.array([1/len(count)/count[p] for p in papers]);mu=w@X;sd=np.maximum(np.sqrt(w@((X-mu)**2)),.03)
    Z=np.asarray((X-mu)/sd,dtype='float32');d=Z.shape[1];k=classes-1;lam=.002
    def fn(a):
        a=a.reshape(d+1,k);loss=.5*lam*np.sum(a[1:]**2);grad=np.vstack([np.zeros(k),lam*a[1:]])
        for left in range(0,len(y),8192):
            sl=slice(left,left+8192);z=Z[sl].astype(float);v=np.c_[np.zeros(len(z)),a[0]+z@a[1:]];pr=softmax(v,axis=1)
            loss+=w[sl]@(logsumexp(v,axis=1)-v[np.arange(len(z)),y[sl]])
            pr[np.arange(len(z)),y[sl]]-=1;e=w[sl,None]*pr[:,1:];grad[0]+=e.sum(0);grad[1:]+=z.T@e
        return loss,grad.ravel()
    op=minimize(fn,np.zeros((d+1)*k),jac=True,method='L-BFGS-B',options={'maxiter':450,'ftol':1e-13,'gtol':3e-7})
    if not op.success or abs(op.jac).max()>1e-5:raise ArithmeticError(f'Multiclass factor failed: {abs(op.jac).max()}')
    coef=op.x.reshape(d+1,k);beta=coef[1:]/sd[:,None]
    return dict(beta=beta.tolist(),bias=(coef[0]-mu@beta).tolist(),n=len(y),counts=np.bincount(y,minlength=classes).tolist(),papers=len(count),max_gradient=float(abs(op.jac).max()),iterations=int(op.nit),loss=float(op.fun))


def rankfit(samples,method):
    a=np.vstack([s for s,y,p in samples]);sizes=np.array([len(s) for s,y,p in samples]);off=np.r_[0,np.cumsum(sizes)[:-1]];idn=np.repeat(np.arange(len(samples)),sizes)
    if method=='surface':a=a[:,:7]
    scale=np.maximum(a.std(0),.03);a=np.asarray(a/scale,float);truth=np.asarray([y for s,y,p in samples])+off;cnt=Counter(p for s,y,p in samples);w=np.array([1/len(cnt)/cnt[p] for s,y,p in samples])
    def fn(v):
        z=a@v;mx=np.maximum.reduceat(z,off);ex=np.exp(z-mx[idn]);den=np.add.reduceat(ex,off);e=ex/den[idn];e[truth]-=1
        return w@(mx+np.log(den)-z[truth])+.015*(v@v),a.T@(w[idn]*e)+.03*v
    op=minimize(fn,np.zeros(a.shape[1]),jac=True,method='L-BFGS-B',options={'maxiter':300,'gtol':2e-7,'ftol':1e-13})
    if not op.success or abs(op.jac).max()>1e-5:raise ArithmeticError('Conditional source likelihood did not converge')
    return dict(beta=(op.x/scale).tolist(),opportunities=len(samples),candidate_rows=len(a),loss=float(op.fun),iterations=int(op.nit),max_gradient=float(abs(op.jac).max()))


def readout(x,model):
    beta=np.array(model['beta']);z=x[:,:len(beta)]@beta+np.array(model['bias']);return softmax(np.c_[np.zeros(len(z)),z],axis=1)


def contents(page,old,models):
    # Compute candidate features once per article and reuse them across calibration limits.
    event_x,_=event_design(page);elig_x,_=eligibility_design(page);t=np.arange(page.n);out={}
    names=np.array([canonical(page.text[m.start:m.end]) for m in page.mentions]);at=available(page)
    for method in METHODS:
        model=models[method];event=readout(event_x,model['event']);elig=readout(elig_x,model['eligibility'])[:,1] if len(elig_x) else np.empty(0)
        if len(elig):
            f,_=conditional_design(page,t,method,old);b=np.array(model['attribution']['beta']);q=f[:,:,:len(b)]@b+np.log(np.maximum(elig,1e-12))[:,None];q[t[None,:]<at[:,None]]=-np.inf
        else:q=np.empty((0,page.n))
        out[method]=(event,q,names)
    return out


def release(data,cut):
    e,q,names=data;out=[];available=np.ones(len(names),dtype=bool);region=False
    for t,p in enumerate(e):
        if p[0]>p[1]+p[2]:region=False
        if region:continue
        eligible=available&np.isfinite(q[:,t]);indices=np.flatnonzero(eligible)
        if not len(indices):continue
        v=q[indices,t];j=int(v.argmax());gain=p[1]*np.exp(v[j]-logsumexp(v))
        if gain>=cut:
            k=int(indices[j]);out.append((k,t,gain));available[names==names[k]]=False;region=True
    return out


def quality(v):
    m=metric(v);p=np.nan_to_num(m['both_sources_and_timings']);f=np.nan_to_num(m['F1'])
    return (float(2*p*f/(p+f)) if p+f>0 else 0.,float(f))


def bootstrap(pages):
    ps=sorted(pages);arr=np.stack([pages[k] for k in ps]);pooled=arr.sum(0);rng=np.random.default_rng(43201);draw=arr[rng.integers(len(ps),size=(5000,len(ps)))].sum(1)
    vals=metric(pooled);bs=metric(draw);methods=list(METHODS)+['prior_native_dual_calibration','prior_native_original_calibration'];summary={}
    def point(x,b):
        b=b[np.isfinite(b)];return [float(x) if np.isfinite(x) else None,np.quantile(b,[.025,.975]).tolist() if len(b) else None]
    for i,arm in enumerate(methods):
        summary[arm]={'counts':pooled[i].tolist(),**{k:point(v[i],bs[k][:,i]) for k,v in vals.items()}}
    paired={a:{k:point(vals[k][1]-vals[k][j],bs[k][:,1]-bs[k][:,j]) for k in ('F1','both_sources_and_timings','retain_other_at_correct_first')} for j,a in enumerate(methods) if j!=1}
    return dict(papers=len(ps),papers_with_dual=int((arr[:,1,-1]>0).sum()),models=summary,paired=paired)


def run(args):
    bank=Bank(args.folder);base=read_json(OUT/'coefficients.json');w=weights(base['parameters']);lock=digest(PLAN)
    if read_json(OUT/'factorization-registration.json')['plan']!=lock:raise ValueError('Changed conditional-likelihood protocol')
    if args.action=='fit':
        A,E,R=collect(bank,w);fit={a:{'event':classfit(E,a,3),'eligibility':classfit(A,a,2),'attribution':rankfit(R[a],a)} for a in METHODS}
        write_json(OUT/'factorized-heads.json',dict(protocol=lock,previous_checkpoint=digest(OUT/'coefficients.json'),features=digest(Path(__file__)),models=fit));return
    fitted=read_json(OUT/'factorized-heads.json');model=fitted['models'];historic=read_json(OUT/'thresholds.json')
    old_grid=np.asarray(historic['development_counts'])[2];old_dual=float(THRESHOLDS[max(range(len(THRESHOLDS)),key=lambda i:quality(old_grid[i]))])
    if args.action=='calibrate':
        tallies=np.zeros((2,len(LIMITS),8),dtype=int)
        for page,doc in bank.pages('calibration'):
            scans=contents(page,w,model)
            for i,a in enumerate(METHODS):
                for j,k in enumerate(LIMITS):tallies[i,j]+=score_events(page,doc,release(scans[a],k))
        chosen={a:float(LIMITS[max(range(len(LIMITS)),key=lambda k:quality(tallies[i,k]))]) for i,a in enumerate(METHODS)}
        write_json(OUT/'factorized-calibration.json',dict(heads=digest(OUT/'factorized-heads.json'),old_dual=old_dual,thresholds=chosen,grid=LIMITS.tolist(),counts=tallies.tolist()));return
    calibration=read_json(OUT/'factorized-calibration.json')
    for split in ('held','published_dev','external'):
        pages=defaultdict(lambda:np.zeros((4,8),int))
        for page,doc in bank.pages(split):
            scans=contents(page,w,model);names=[canonical(page.text[m.start:m.end]) for m in page.mentions]
            for i,a in enumerate(METHODS):pages[doc['paper']][i]+=score_events(page,doc,release(scans[a],calibration['thresholds'][a]))
            f=np.full((len(names),page.n),-1.,'float32')
            for i,(m,t) in enumerate(zip(page.mentions,available(page))):
                t=np.arange(t,page.n)
                if len(t):f[i,t]=probabilities(page,m,t,w)[:,2]
            for i,k in enumerate([old_dual,historic['thresholds']['native']]):
                pages[doc['paper']][2+i]+=score_events(page,doc,decode(f,names,k))
        write_json(OUT/('factorized-'+split+'.json'),dict(calibration=digest(OUT/'factorized-calibration.json'),**bootstrap(pages)))
        print('Completed source and timing holdout',split,flush=True)


if __name__=='__main__':
    p=ArgumentParser(description=__doc__);p.add_argument('action',choices=('fit','calibrate','score'));p.add_argument('folder',type=Path);run(p.parse_args())
