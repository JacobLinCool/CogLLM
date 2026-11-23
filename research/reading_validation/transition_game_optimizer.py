"""Optimize the preregistered smooth squared nuclear norm near low-rank solutions."""
import math
import numpy as np


EPS=1e-12


def spectral_prox(values,c):
    """Proximity operator of c/2 (sum s² + (sum sqrt(s²+eps))²)."""
    raw=np.asarray(values,dtype=np.float64)
    if np.any(raw<0) or c<=0:raise ValueError('Singular values and proximal step are nonnegative')
    A=1+c
    # Start from the exactly solvable eps=0 operator, then solve the strictly convex smooth problem.
    s=np.sort(raw)[::-1];cum=np.cumsum(s)
    theta=max(c*cum[j]/(A+c*(j+1)) for j in range(len(s)))
    z=np.maximum((raw-theta)/A,0.)
    def entries(x):
        h=np.sqrt(x*x+EPS);T=h.sum();v=x/h
        obj=.5*np.sum((x-raw)**2)+.5*c*(x@x+T*T)
        grad=A*x-raw+c*T*v
        jac=A*np.eye(len(x))+c*(np.outer(v,v)+np.diag(T*EPS/h**3))
        return obj,grad,jac
    for _ in range(65):
        obj,g,h=entries(z)
        if np.max(np.abs(g))<=2e-12*(1+raw.max()):return z
        step=np.linalg.solve(h,g);rate=1.
        for _ in range(55):
            nxt=np.maximum(0.,z-rate*step)
            if entries(nxt)[0]<=obj-1e-4*rate*(g@step)+4*np.finfo(float).eps*(1+abs(obj)):
                z=nxt;break
            rate*=.5
        else:break
    # All s_i are monotone in a shared T; bounded scalar roots avoid any nearly singular Newton behavior.
    low=len(raw)*math.sqrt(EPS);high=np.sqrt((raw/A)**2+EPS).sum()
    for _ in range(60):
        T=(low+high)/2;lo=np.zeros_like(raw);hi=raw/A
        for _ in range(65):
            mid=(lo+hi)/2;left=A*mid+c*T*mid/np.sqrt(mid*mid+EPS)<raw
            lo=np.where(left,mid,lo);hi=np.where(left,hi,mid)
        z=(lo+hi)/2
        if np.sqrt(z*z+EPS).sum()>T:low=T
        else:high=T
    if np.max(abs(entries(z)[1]))>2e-10*(1+raw.max()):raise ArithmeticError('Uncertified exact singular-value proximal solution')
    return z


def learn_shared(X,labels,mass,semantic):
    import torch
    from torch.nn import functional as F

    d,k=X.shape[1],labels.shape[1];lam=.01;cut=d-semantic
    def smooth(w,b,gradient=True):
        logit=X@w+b;val=(mass*(F.softplus(logit)-labels*logit)).sum()
        if not gradient:return val
        e=(logit.sigmoid()-labels)*mass
        return val,X.T@e,e.sum(0)
    def regular(w):
        u,s,v=torch.linalg.svd(w[cut:],full_matrices=False);h=torch.sqrt(s*s+EPS);T=h.sum()
        val=lam/(2*k)*w[:cut].square().sum()+lam/(4*k)*(s.square().sum()+T*T)
        native=(u*(lam/(2*k)*(s+T*s/h)))@v
        return val,torch.cat((lam/k*w[:cut],native))
    def prox(w,tau):
        u,s,v=torch.linalg.svd(w[cut:],full_matrices=False)
        z=s.new_tensor(spectral_prox(s.cpu().numpy(),tau*lam/(2*k)))
        return torch.cat((w[:cut]/(1+tau*lam/k),(u*z)@v))
    w=torch.zeros((d,k),device=X.device,dtype=X.dtype,requires_grad=True)
    b=torch.special.logit(((labels*mass).sum(0)/mass.sum(0)).clamp(1e-9,1-1e-9)).detach().requires_grad_()
    # Stable analytical gradient avoids differentiating nearly repeated or tiny Gram eigenvalues.
    optimizer=torch.optim.LBFGS((w,b),max_iter=320,history_size=40,tolerance_grad=2e-8,tolerance_change=1e-14,line_search_fn='strong_wolfe')
    calls=0;initial=None
    @torch.no_grad()
    def closure():
        nonlocal calls,initial
        data,v,g=smooth(w,b);r,gr=regular(w);val=data+r
        w.grad=v+gr;b.grad=g;calls+=1
        if initial is None:initial=float(val)
        return val
    optimizer.step(closure);w=w.detach();b=b.detach()
    history=[];step=2.;moment=1.;u=w.clone();t=b.clone();last=None
    for itr in range(2501):
        val,gr,gb=smooth(w,b);r,rg=regular(w);grad=max(float((gr+rg).abs().max()),float(gb.abs().max()))
        obj=float(val+r);last=obj
        if itr in (0,1,2,5,10,25,50,100,250,500,1000,2000) or grad<=5e-7:
            history.append([itr,obj,grad])
        if grad<=5e-7:
            return w,b,{'initial_objective':initial,'objective':obj,'max_abs_gradient':grad,'evaluations':calls,
                        'proximal_iterations':itr,'analytic_svd_gradient':True,'proximal_trace':history}
        if itr==2500:break
        yu,gy,gt=smooth(u,t)
        tau=min(1.07*step,100.)
        for _ in range(35):
            cand=prox(u-tau*gy,tau);cb=t-tau*gt;du=cand-u;dt=cb-t
            f=smooth(cand,cb,False)
            major=yu+(gy*du).sum()+(gt*dt).sum()+(du.square().sum()+dt.square().sum())/(2*tau)
            if float(f-major)<=2e-14:break
            tau*=.5
        else:raise ArithmeticError('No valid logistic majorization step')
        # FISTA restart when momentum no longer points down the objective.
        new_moment=(1+math.sqrt(1+4*moment*moment))/2
        momentum=(moment-1)/new_moment
        if float(((u-cand)*(cand-w)).sum()+((t-cb)*(cb-b)).sum())>0:
            new_moment=1.;momentum=0.
        u=cand+momentum*(cand-w);t=cb+momentum*(cb-b)
        w,b,step,moment=cand,cb,tau,new_moment
    raise RuntimeError(f'Four-task signed penalty failed true stationarity: {last}, {grad}, {history}')


def self_test():
    rng=np.random.default_rng(103)
    for val in ([1,.02,1e-9,0],[2,2,.3,.3],[1e-6,1e-7,1e-8,1e-9],[0,0,0,0],rng.random(4)):
        s=np.array(val)
        for c in (.0001,.1,3.):
            z=spectral_prox(s,c);h=np.sqrt(z*z+EPS)
            np.testing.assert_allclose((1+c)*z+c*h.sum()*z/h,s,atol=1e-10)
    return True


if __name__=='__main__':print(self_test())
