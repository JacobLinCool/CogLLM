"""Conditional likelihoods for actual clicks and paired interpretation changes."""
from dataclasses import dataclass
from collections import Counter
import hashlib
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

DIM = 24
RIDGE = .04
COUPLING = .04

def normalize(v, starts):
    """Log-partitions and probabilities for adjacent nonempty groups."""
    n = np.diff(np.r_[starts,len(v)])
    peak = np.maximum.reduceat(v,starts)
    ex = np.exp(v-np.repeat(peak,n))
    total = np.add.reduceat(ex,starts)
    return peak+np.log(total), ex/np.repeat(total,n)

def is_development(group):
    return int(hashlib.sha256(("referent-source-v1/"+group).encode()).hexdigest()[:8],16)%5==0

@dataclass
class Clicks:
    keys: np.ndarray
    query: np.ndarray
    base: np.ndarray
    offsets: np.ndarray
    repeat: np.ndarray
    item_weights: np.ndarray
    click_words: np.ndarray
    click_offsets: np.ndarray
    click_weights: np.ndarray
    click_repeat: np.ndarray

    @classmethod
    def make(cls, rows, features):
        examples=[(r,q) for r in rows for q in r["queries"]]
        groups=Counter(r["group"] for r,q in examples)
        keys=[];queries=[];base=[];weights=[];lengths=[]
        sx=[];sl=[];sw=[];c=0
        for r,q in examples:
            f=features[r["id"]];clicks=r["clicks"][str(q)]
            n=sum(v for _,_,v in clicks);w=1/(len(groups)*groups[r["group"]])
            keys.append(f["h"][:q]);queries.append(f["h"][q]);base.append(f["b"][q]);weights.append(w);lengths.append(q)
            for a,b,v in clicks:
                if not 0<=a<b<=q or v<1:raise ValueError("Invalid source click")
                sx.extend(range(c+a,c+b));sl.append(b-a);sw.append(w*v/n)
            c+=q
        return cls(np.concatenate(keys),np.stack(queries),np.concatenate(base),
                   np.cumsum([0]+lengths[:-1]),np.repeat(np.arange(len(lengths)),lengths),
                   np.array(weights),np.array(sx),np.cumsum([0]+sl[:-1]),np.array(sw),
                   np.repeat(np.arange(len(sl)),sl))

    def __call__(self,W):
        scores=self.base+np.einsum("ij,ij->i",self.keys,(self.query@W.T)[self.repeat])
        zn,p=normalize(scores,self.offsets);zp,pc=normalize(scores[self.click_words],self.click_offsets)
        loss=self.item_weights@zn-self.click_weights@zp
        residual=p*self.item_weights[self.repeat]
        np.add.at(residual,self.click_words,-pc*self.click_weights[self.click_repeat])
        dk=np.add.reduceat(residual[:,None]*self.keys,self.offsets,axis=0)
        return loss,dk.T@self.query

@dataclass
class Twins:
    keys: np.ndarray
    query: np.ndarray
    change: np.ndarray
    base: np.ndarray
    groups: np.ndarray
    side: np.ndarray
    item: np.ndarray
    y: np.ndarray
    twins: np.ndarray

    @classmethod
    def make(cls,rows,features):
        keys=[];queries=[];changes=[];base=[];lengths=[];ys=[];families={}
        for i,r in enumerate(rows):
            f=features[r["id"]];g=r["measurement"];q=g["q"];k=g["k"];y=r["answer"]
            for region in g["names"]:
                if not region or any(s<0 or s>=q for s in region):raise ValueError("Invalid candidate")
                keys.append(f["h"][region]);base.append(f["b"][q][region]);lengths.append(len(region))
            queries.append(f["h"][q]);changes.append(f["h"][k]-f["h"][q]);ys.append(2*y-1)
            families.setdefault(r["group"],[]).append((i,r["identities"][y]))
        if any(len(f)!=2 or f[0][1]==f[1][1] for f in families.values()):
            raise ValueError("Training pair does not reverse the same underlying choice")
        side=np.repeat(np.arange(len(lengths)),lengths)
        return cls(np.concatenate(keys),np.stack(queries),np.stack(changes),np.concatenate(base),
                   np.cumsum([0]+lengths[:-1]),side,side//2,np.array(ys),np.array([[a for a,_ in f] for f in families.values()]))

    def __call__(self,W,H):
        act=self.query@W.T+self.change@H.T
        z=self.base+np.einsum("ij,ij->i",self.keys,act[self.item])
        mass,p=normalize(z,self.groups)
        signed=self.y*(mass[1::2]-mass[::2]);combined=signed[self.twins].sum(1)
        o=.5*(np.logaddexp(0,-signed).mean()+np.logaddexp(0,-combined).mean())
        grad=-.5*expit(-signed)/len(signed)
        np.add.at(grad,self.twins.ravel(),np.repeat(-.5*expit(-combined)/len(combined),2))
        delta=self.y*grad
        pairkeys=np.add.reduceat(self.keys*p[:,None],self.groups,axis=0)
        dk=delta[:,None]*(pairkeys[1::2]-pairkeys[::2])
        return o,dk.T@self.query,dk.T@self.change

def optimize(fun,start):
    calls=[]
    initial=fun(start)[0]
    def obj(x):
        f,g=fun(x.reshape(start.shape))
        calls.append(float(f))
        return f,g.ravel()
    fitted=minimize(obj,start.ravel(),method="L-BFGS-B",jac=True,
                    options={"maxiter":700,"gtol":4e-7,"ftol":1e-14,"maxls":60,"maxcor":18})
    weights=fitted.x.reshape(start.shape);last,grad=fun(weights)
    if not fitted.success or np.abs(grad).max()>3e-6 or last>initial+1e-10:
        raise ValueError(f"Source-only fit failed: {fitted.message}, gradient {np.abs(grad).max()}")
    return weights,{"initial":float(initial),"objective":float(last),"gradient":float(np.abs(grad).max()),"iterations":fitted.nit,
                    "evaluations":len(calls),"criterion":str(fitted.message)}

def fit(clicks,twins):
    z=np.zeros((DIM,DIM))
    def click(W):
        l,g=clicks(W)
        return l+.5*RIDGE*np.sum(W*W),g+RIDGE*W
    W,wfit=optimize(click,z)
    def revision(H):
        l,_,g=twins(W,H)
        return l+.5*RIDGE*np.sum(H*H),g+RIDGE*H
    H,hfit=optimize(revision,z)
    def joint(block):
        w,h=block
        l0,g0=clicks(w);l1,g1,g2=twins(w,h)
        loss=.5*(l0+l1)+.25*RIDGE*(np.sum(w*w)+np.sum(h*h))+.25*COUPLING*np.sum((w-h)**2)
        g=np.stack([g0+g1+RIDGE*w+COUPLING*(w-h),g2+RIDGE*h+COUPLING*(h-w)])*.5
        return loss,g
    J,jfit=optimize(joint,np.stack([W,H]))
    return {"prior":np.stack([z,z]),"private":np.stack([W,H]),"shared":J},{"private_click":wfit,"private_revision":hfit,"shared":jfit}

def gradient_audit(clicks,twins):
    rng=np.random.default_rng(918);shape=(DIM,DIM)
    W,H=[rng.normal(0,.01,shape) for _ in range(2)]
    u,v=[rng.normal(0,1/DIM,shape) for _ in range(2)]
    step=1e-5
    _,g=clicks(W)
    ck=(clicks(W+step*u)[0]-clicks(W-step*u)[0])/(2*step)
    _,ga,gb=twins(W,H)
    tw=(twins(W+step*u,H+step*v)[0]-twins(W-step*u,H-step*v)[0])/(2*step)
    errors=[abs(ck-np.sum(g*u)),abs(tw-np.sum(ga*u)-np.sum(gb*v))]
    if max(errors)>2e-7:
        raise ValueError(f"Analytic likelihood gradient error: {errors}")
    return list(map(float,errors))
