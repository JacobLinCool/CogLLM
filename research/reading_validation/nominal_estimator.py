"""Estimate action frequencies and literal earlier ranges for ordinary noun phrases."""
from collections import Counter
from dataclasses import dataclass
import hashlib
import numpy as np
from scipy.special import logsumexp,softmax
from nltk.stem.snowball import EnglishStemmer
from research.reading_validation.nominal_data import DETERMINERS,REJECT
from research.reading_validation.referent_estimation import normalize,optimize

STEM=EnglishStemmer()
ACTION_RIDGE=.028
RANGE_RIDGE=.028
DIM=24

class Words:
    def __init__(self,row,states):
        self.h=states
        self.forms=np.array([row["text"][a:b].casefold() for a,b in row["words"]])
        self.stems=np.array([STEM.stem(x) for x in self.forms])
        self.grams=[set(zip(w,w[1:])) for w in self.forms]
        self.sent=np.array(row["sentences"])
        self.det=np.r_[False,np.isin(self.forms[:-1],list(DETERMINERS))]
        self.reject=np.isin(self.forms,list(REJECT))
        self.letter=np.array([w.isalpha() for w in self.forms])
        self.caps=np.array([row["text"][a:b].istitle() for a,b in row["words"]])
        self.length=np.log1p([len(w) for w in self.forms])
        self.unit=self.h/np.maximum(np.linalg.norm(self.h,axis=1,keepdims=True),1e-8)
    def surface(self,q):
        x=self.forms;word=x[q];repeat=x[:q]==word;stem=self.stems[:q]==self.stems[q]
        gap=np.arange(q,0,-1);diff=self.sent[q]-self.sent[:q]
        old=Counter(x[:q]);g=self.grams[q]
        sim=[len(g&v)/max(1,len(g|v)) for v in self.grams[:q]]
        return np.column_stack((np.log1p(gap),1/gap,np.exp(-gap/20),diff==0,np.log1p(diff),self.length[:q],
            self.caps[:q],self.letter[:q],self.reject[:q],self.det[:q],repeat,stem,
            np.log1p([old[a] for a in x[:q]]),sim,repeat/np.sqrt(gap),stem/np.sqrt(gap),
            self.caps[q]==self.caps[:q])).astype("float32")
    def action(self,q,context):
        x=self.forms;last=x[q-1];same=x[:q]==x[q];root=self.stems[:q]==self.stems[q];ix=np.flatnonzero(same)
        surf=np.array([np.log1p(q),np.log1p(self.sent[q]),self.length[q],self.caps[q],self.det[q],last=="the",
            last in ("a","an"),last in ("this","that","these","those"),last in ("my","your","our","her","his","its","their"),
            np.log1p(same.sum()),np.log1p(root.sum()),same.any(),root.any(),1/(q-ix[-1]) if len(ix) else 0.,
            same[self.sent[:q]==self.sent[q]].sum(),len(set(x[:q]))/q])
        if not context:return surf
        rel=self.unit[:q]@self.unit[q]
        order=np.argsort(-rel);peak=order[0]
        stats=np.array([rel[peak],np.mean(rel[order[:3]]),np.mean(rel),np.std(rel),np.log1p(q-peak),same[peak],
                  rel[same].max() if same.any() else 0.,rel[~same].max() if (~same).any() else 0.])
        return np.r_[surf,stats,self.h[q],self.h[q]-self.h[q-1]]

def action_model(exposures,context):
    keys=[(r,q,o) for r,items in exposures for q,o in items.items()]
    counts=Counter(r["group"] for r,q,o in keys)
    w=np.array([1/(len(counts)*counts[r["group"]]) for r,q,o in keys])
    x=np.stack([r["features"].action(int(q),context) for r,q,o in keys])
    y=np.array([o["action"] for r,q,o in keys],float);y/=y.sum(1,keepdims=True)
    mu=w@x;sd=np.sqrt(w@np.square(x-mu));sd[sd<1e-7]=1.
    x=np.column_stack((np.ones(len(x)),(x-mu)/sd))
    start=np.zeros((x.shape[1],4));start[0]=np.log(w@y);start[0]-=start[0].mean()
    def fn(b):
        l=x@b
        value=w@(logsumexp(l,axis=1)-np.sum(l*y,axis=1))+.5*ACTION_RIDGE*np.sum(b[1:]**2)
        grad=x.T@((softmax(l,axis=1)-y)*w[:,None]);grad[1:]+=ACTION_RIDGE*b[1:]
        return value,grad
    b,fitted=optimize(fn,start)
    return (mu,sd,b),fitted

@dataclass
class RangeModel:
    form_mu: np.ndarray
    form_sd: np.ndarray
    coef: np.ndarray
    context: bool
    def predict(self,doc,q):
        x=(doc.surface(q)-self.form_mu)/self.form_sd
        y=x@self.coef[:x.shape[1]]
        if self.context:
            W=self.coef[x.shape[1]:].reshape(DIM,DIM)
            y+=doc.h[:q]@(W@doc.h[q])
        return y

def certain_sample(r,items,limit):
    strata=[[],[]];f=r["features"]
    for key,o in items.items():
        if sum(e[2] for e in o["clicks"])==0:continue
        q=int(key);repeats=(f.forms[:q]==f.forms[q]).any()
        val=hashlib.sha256(f"nominal-v1/{r['id']}/{key}".encode()).digest()
        strata[int(repeats)].append((val,q,o))
    if sum(map(len,strata))<=limit:return [(q,o) for s in strata for _,q,o in s]
    order=[sorted(s) for s in strata]
    n=[min(limit//2,len(s)) for s in order]
    for i in range(2):n[i]+=min(limit-sum(n),len(order[i])-n[i])
    return [(q,o) for ix,s in enumerate(order) for _,q,o in s[:n[ix]]]

def range_inputs(exposures):
    qs=[];group=[];h=[];qu=[];x=[];n=[];si=[];sl=[];votes=[];cursor=0;qi=[]
    for r,items in exposures:
        doc=r["features"]
        for q,o in certain_sample(r,items,40):
            cs=o["clicks"];total=sum(e[2] for e in cs)
            n.append(q);qs.append(q);group.append(r["group"])
            h.append(doc.h[:q]);qu.append(doc.h[q]);x.append(doc.surface(q))
            for a,b,c in cs:
                si.extend(range(cursor+a,cursor+b));sl.append(b-a);votes.append(c/total);qi.append(len(qs)-1)
            cursor+=q
    counts=Counter(group);ww=np.array([1/(len(counts)*counts[g]) for g in group]);K=np.concatenate(h).astype("float32")
    raw=np.concatenate(x);mu=raw.mean(0,dtype=float);sd=raw.std(0,dtype=float);sd[sd<1e-7]=1.
    raw-=mu;raw/=sd
    return (raw,K,np.stack(qu),np.cumsum([0]+n[:-1]),np.repeat(np.arange(len(n)),n),np.array(si),
            np.cumsum([0]+sl[:-1]),np.repeat(np.arange(len(sl)),sl),ww,ww[qi]*np.array(votes),mu,sd)

def range_model(data,context,initial=None):
    x,k,q,off,iq,ix,so,sid,w,ws,mu,sd=data
    d=x.shape[1];extra=DIM*DIM if context else 0
    start=np.zeros(d+extra)
    if initial is not None:start[:d]=initial.coef
    def fn(b):
        z=x@b[:d]
        if context:
            v=q@b[d:].reshape(DIM,DIM).T
            z+=np.einsum("ij,ij->i",k,v[iq])
        norm,p=normalize(z,off);inside,pc=normalize(z[ix],so)
        value=w@norm-ws@inside+.5*RANGE_RIDGE*(b@b)
        dz=p*w[iq];np.add.at(dz,ix,-pc*ws[sid]);g=x.T@dz
        if context:
            ds=np.add.reduceat(dz[:,None]*k,off,axis=0)
            g=np.r_[g,(ds.T@q).ravel()]
        return value,g+RANGE_RIDGE*b
    b,details=optimize(fn,start)
    return RangeModel(mu,sd,b,context),details

def action_predict(model,x):
    mu,sd,b=model
    return softmax(np.r_[1.,(x-mu)/sd]@b)
