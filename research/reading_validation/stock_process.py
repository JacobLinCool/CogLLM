"""Differentiate the observed marginal count through a conserved, two-timescale issue memory."""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit,gammaln,logsumexp
from research.reading_validation.narrative_course_fit import mean_rating

def norm(x):
    mu=x.mean(0);sd=x.std(0);sd[sd<1e-6]=1;return mu,sd

def word_features(p):
    a=p.story.line_mean(p.story.feature_blocks['form'],p.story.word_ok)
    return np.nan_to_num(a)

class Design:
    def __init__(self,training):
        all_native=np.vstack([p.native for p in training]);self.center=all_native.mean(0)
        _,_,v=np.linalg.svd(all_native-self.center,full_matrices=False);self.axis=v[:8].T
        self.bmode=norm(np.vstack([self.local(p) for p in training]))
        self.hmode=norm(np.vstack([self.pair(p) for p in training]))
        self.response_scale=np.sqrt(np.nanmean(np.concatenate([np.diff(suspense(p))**2 for p in training])))
        self.birth_scale=np.mean([len(p.records)/len(p.story.teams)/p.n for p in training])
    def local(self,p):
        return np.column_stack((p.birth_form,word_features(p)[:,-2:],(p.native-self.center)@self.axis))
    def pair(self,p):
        n=p.form.shape[1];return np.column_stack((p.age,p.lex[:,-2:],p.semantic,p.lex[:,n:n+3]))
    def page(self,p):
        x=(self.local(p)-self.bmode[0])/self.bmode[1];y=(self.pair(p)-self.hmode[0])/self.hmode[1]
        return Page(p,x,y,self.response_scale)

def suspense(p):
    avg=mean_rating(p.story.ratings);ok=p.story.word_ok&np.isfinite(avg)
    return p.story.line_mean(avg,ok)

class Page:
    def __init__(self,p,x,y,scale):
        self.x=np.column_stack((np.ones(p.n),x));self.y=np.column_stack((np.ones(len(y)),y));self.short=self.x[:,:min(self.x.shape[1],10)]
        self.n=p.n;self.row,self.col=np.tril_indices(p.n,-1);self.support=np.tri(p.n,dtype=bool);self.regional=p
        self.mu_obs=np.nan_to_num(np.diff(suspense(p),prepend=np.nan)/scale)
        self.mu_ok=np.isfinite(np.diff(suspense(p),prepend=np.nan))
        self.birth=np.mean([np.bincount([r.source for r in p.records if r.team==t],minlength=p.n) for t in p.story.teams],axis=0)
        self.terms=[]
        for team,s in p.story.teams.items():
            low=(s==1).sum(1);high=(s!=0).sum(1);k=np.arange(high.max()+1)[None,:]
            self.terms.append((k,(k>=low[:,None])&(k<=high[:,None])))
        events=p.records;self.src=np.array([r.source for r in events]);self.end=np.array([r.definite for r in events]);self.closed=np.array([r.upper is not None for r in events]);self.upper=np.array([0 if r.upper is None else r.upper for r in events])
        self.ew=np.array([1/(len(p.story.teams)*sum(s.team==r.team for s in events)) for r in events])

    def process(self,v):
        b,h,g,s,delta,affect,c=v
        lograte=self.x@b
        if np.max(lograte)>70:raise FloatingPointError('Opening rate outside supported finite scale')
        rate=np.exp(lograte);pi=expit(self.short@s);relative=self.y@h;diff=np.logaddexp(0,g)
        x=np.stack([relative+.5*diff,relative-.5*diff]);q=expit(x)
        logstep=np.zeros((2,self.n,self.n));logstep[:,self.row,self.col]=-np.logaddexp(0,x)
        survivor=np.exp(np.cumsum(logstep,axis=1))*self.support;mix=np.stack((1-pi,pi))[:,None,:]*survivor
        total=mix.sum(0);count=total@rate;change=np.diff(count,prepend=0.);pred=self.short@affect+c*change
        return rate,pi,q,survivor,mix,total,count,change,pred

    def loss(self,v,mode,bscale):
        b,h,g,s,_,affect,c=v
        rate,pi,q,surv,mix,total,count,change,pred=self.process(v)
        val=(rate-self.birth*np.log(rate)).mean()/bscale;dlograte=(rate-self.birth)/(self.n*bscale)
        # The joint source occurrence likelihood identifies the hazard on its own source span.
        src,t=self.src,self.end;obs=total[t,src].copy()
        active=self.closed;u=self.upper[active];sp=src[active];obs[active]-=total[u,sp]
        obs=np.maximum(obs,1e-30);val-=self.ew@np.log(obs)/4
        dm=np.zeros_like(total);slope=self.ew/obs/4;np.add.at(dm,(t,src),-slope);np.add.at(dm,(u,sp),slope[active])
        dc=np.zeros(self.n);da=np.zeros(len(affect));dscale=0
        if mode in ('count','joint'):
            for k,allowed in self.terms:
                lp=k*np.log(count[:,None])-count[:,None]-gammaln(k+1);lp[~allowed]=-np.inf
                z=logsumexp(lp,axis=1);expected=(np.exp(lp-z[:,None])*k).sum(1)
                factor=.5/(len(self.terms)*self.n);val-=factor*z.sum();dc+=factor*(1-expected/count)
        if mode=='joint':
            mask=self.mu_ok;residual=(pred-self.mu_obs)*mask;factor=.5/max(1,mask.sum());val+=.5*factor*(residual@residual)
            ds=factor*residual;da=self.short.T@ds;dscale=ds@change;dc+=c*ds;dc[:-1]-=c*ds[1:]
        dm+=dc[:,None]*rate[None,:];dlograte+=rate*(dc@total)
        # A later loss contributes to every earlier survival factor for the same source only.
        dlog=mix*dm[None,:,:];rev=dlog[:,::-1].cumsum(1)[:,::-1];dz=-rev[:,self.row,self.col]*q
        dp=((surv[1]-surv[0])*dm).sum(0)*pi*(1-pi)
        grads=(self.x.T@dlograte,self.y.T@dz.sum(0),.5*expit(g)*(dz[0].sum()-dz[1].sum()),self.short.T@dp,0.,da,dscale)
        return val,grads

    def forecast(self,v):
        rate,pi,q,surv,mix,total,count,change,pred=self.process(v);before=np.zeros_like(total);before[1:]=total[:-1]
        resolution=(before-total)*np.tri(self.n,k=-1);hz=np.divide(resolution,before,out=np.zeros_like(before),where=before>1e-30)
        mass=resolution*rate[None,:];difference=np.max(abs(np.diff(count,prepend=0)-rate+mass.sum(1)))
        if difference>1e-8 or np.min(mass)<-1e-10 or np.max(hz)>1+1e-8:raise ArithmeticError('Double survivor mass is not conserved')
        return dict(birth=rate,stock=count,hazard=hz,event=mass,suspense_change=pred,conservation=float(difference))

class Fit:
    def __init__(self,pages):
        b=pages[0].x.shape[1];h=pages[0].y.shape[1];s=pages[0].short.shape[1]
        self.lens=[b,h,1,s,0,s,1];self.ends=np.cumsum(self.lens)[:-1]
    def unpack(self,v):
        b,h,g,s,_,a,c=np.split(v,self.ends)
        return b,h,float(g[0]),s,None,a,float(c[0])
    def flatten(self,gr):
        b,h,g,s,_,a,c=gr;return np.r_[b,h,g,s,a,c]
    def fit(self,pages,mode,bscale,init=None,emphasis=1.):
        sizes=self.lens;v=np.zeros(sum(sizes));v[0]=np.log(bscale);v[sizes[0]]=-3.;v[sum(sizes[:2])]=2.
        if init is not None:v=init.copy()
        penalty=np.full(len(v),.055);penalty[[0,sizes[0]]]=.001;penalty[sum(sizes[:2])]=.008
        center=v*0.;center[0]=np.log(bscale);center[sizes[0]]=-3.;center[sum(sizes[:2])]=2.
        def objective(x):
            flat=self.unpack(x);items=[p.loss(flat,mode,bscale) for p in pages]
            val=sum(z[0] for z in items)/len(items);g=np.mean([self.flatten(z[1]) for z in items],axis=0);offset=x-center
            return val+.5*(penalty@np.square(offset)),g+penalty*offset
        direction=np.linspace(-.2,.2,len(v));direction/=np.linalg.norm(direction)
        numeric=(objective(v+1e-5*direction)[0]-objective(v-1e-5*direction)[0])/2e-5;analytic=objective(v)[1]@direction
        if abs(numeric-analytic)>1e-4:raise ArithmeticError(f'Composite count and survival gradient mismatch {numeric} {analytic}')
        opt=minimize(objective,v,jac=True,method='L-BFGS-B',options={'maxiter':700,'gtol':1e-7,'ftol':1e-13})
        if not opt.success or np.max(np.abs(opt.jac))>1e-5:raise ArithmeticError(f'Conserved dynamics did not converge {opt.message} {abs(opt.jac).max()}')
        return opt.x,dict(gradient=float(abs(opt.jac).max()),iterations=int(opt.nit),objective=float(opt.fun),check=float(abs(numeric-analytic)))

def fit_affect(pages,v,fit):
    x=[];y=[];w=[]
    for page in pages:
        count=page.process(fit.unpack(v))[6];z=np.column_stack((page.short,np.diff(count,prepend=0)))
        ok=page.mu_ok;x.extend(z[ok]);y.extend(page.mu_obs[ok]);w.extend([1./(len(pages)*ok.sum())]*ok.sum())
    x=np.array(x);y=np.array(y);w=np.array(w);v2=np.linalg.solve(x.T@(x*w[:,None])+.11*np.eye(x.shape[1]),x.T@(y*w))
    v=v.copy();v[-len(v2):]=v2;return v
