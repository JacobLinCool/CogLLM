"""Verify the multi-view objective, threshold behavior, and causal coordinate conventions on synthetic data."""

from types import SimpleNamespace

import numpy as np
from scipy.optimize import minimize
from scipy.optimize._numdiff import approx_derivative

from research.backbone_transfer.extract import write_json
from research.reading_validation.identity_causal_fit import OUT,design,loss_and_grad,trace


def run():
    rng=np.random.default_rng(82171)
    ntrain=160;d=5;X=rng.normal(size=(ntrain,d+1)).astype('float32');X[:100,-1]=0
    truth=np.array([1.5,-1.2,.4,1.8,-.6,0]);X*=np.sign(X@truth)[:,None];w=np.ones(ntrain,dtype='float32')/ntrain
    theta=rng.normal(size=d+1); delta=.002
    numerical=approx_derivative(lambda a:np.array([loss_and_grad(a,X,w,.1)[0]]),theta,method='3-point',abs_step=delta).ravel()
    grad=loss_and_grad(theta,X,w,.1)[1];diff=float(abs(numerical-grad).max())
    fit=minimize(loss_and_grad,np.zeros(d+1),args=(X,w,.1),jac=True,method='L-BFGS-B',options={'gtol':1e-7})
    held=rng.normal(size=(300,d+1)).astype('float32');score=float(np.mean(np.sign(held@fit.x)==np.sign(held@truth)))
    width=4;n=11;arr=rng.normal(size=(n,width)).astype('float32');base=arr[0:3]
    direct=design(np.stack([base.mean(0),base[-1]]),np.stack([base.mean(0),base[-1]]),[0,0],1,'x','x')
    weights={k:{'bias':-.2,'weights':(rng.normal(size=len(v))*.04).tolist()} for k,v in direct.items()}
    def document(stop):
        s=['a','b','c','d','e','f','g','h','i','j','k'][:stop];text=' '.join(s)
        seg=[(0,3),(3,7)]+([(7,stop)] if stop>7 else [])
        return SimpleNamespace(spans={'q':seg},text={'q':text},offsets={'q':[(2*i,2*i+1) for i in range(stop)]})
    attention=np.ones((n,n,2),dtype='float32')/n
    levels,raw=trace(document(n),weights,'q',arr,attention)
    cut,short=trace(document(7),weights,'q',arr[:7],attention[:7,:7])
    err=max(np.max(abs(cut[k]-levels[k][:7])) for k in cut)
    if diff>1e-4 or score<.89 or err>1e-8 or not all(np.all(np.diff(p)>=0) for p in levels.values()):
        raise AssertionError('Objective or prefix consistency check failed')
    out={'analytic_gradient_max_error':diff,'withheld_synthetic_classification':score,'n':300,
         'synthetic_prefix_error':float(err),'numerical_objective_precision':'float32 matrix, 0.002 finite difference step'}
    write_json(OUT/'simulation-check.json',out);print(out)


if __name__=='__main__':run()
