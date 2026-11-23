"""Read a visible action and estimate entity properties from masked, human-coded state changes."""
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import json
import re

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from research.reading_validation.argument_data import sentences
from research.reading_validation.memory_data import load_rows, sha
from research.reading_validation.state_transport_data import OUT, TRIP

WORD = re.compile(r"[^\W\d_]+(?:['’\-][^\W\d_]+)*|\d+(?:[.,]\d+)*")
STOP = frozenset('a an the in of some for on from to out with and or into by that at his her its their my as then very was were is are has had have he she it him them they who there this'.split())
ATTR = ('conscious','wearing','h_wet','hygiene','exist','clean','power','functional','pieces','wet','open','temperature','solid','contain','running','moveable','mixed','edible')
# Position zero and one are the released proposition before and after the action.
CODES = ((-1,-1),(0,0),(1,1),(1,0),(0,1),(-1,0),(-1,1),(0,-1),(1,-1))


@lru_cache(maxsize=32000)
def coordinate(text, width):
    digest = hashlib.blake2b(text.encode(),digest_size=8).digest()
    return int.from_bytes(digest[:4],'little')%width, 1 if digest[-1]&1 else -1


def sketch(tokens, width):
    value=np.zeros(width)
    for t in tokens:
        i,s=coordinate(t,width);value[i]+=s
    return value/max(1.,np.sqrt(value@value))


def norm(x):
    return x/max(1e-7,np.linalg.norm(x))


class Archive:
    def __init__(self,path=None):
        self.spec=json.loads((OUT/'inputs.json').read_bytes())
        self.text={r['id']:r['text'] for r in load_rows(TRIP/'stimuli.jsonl')}
        self.native=None;self.info={};self.certificate=None
        if path:
            r=json.loads((path/'receipt.json').read_bytes());self.certificate=sha(path/'receipt.json')
            if r['source']!=self.spec['source'] or r['file']!=sha(path/'features.npz'):
                raise ValueError('Unverified native feature archive')
            self.native=np.load(path/'features.npz',allow_pickle=False);self.info=r['articles']

    def eligible(self,k):return self.native is None or k in self.info

    @lru_cache(maxsize=2300)
    def document(self,k):
        native = None
        if self.native is not None:
            native = {name:self.native[k+'.'+name] for name in ('char','rep','links','surprisal','words')}
            if list(map(list,sentences(self.text[k]))) != self.info[k]['sentences']:
                raise ValueError('Native sentence mismatch')
        return build_document(self.text[k], native)


def build_document(text, native=None):
    regions = sentences(text)
    ws = list(WORD.finditer(text))
    tokens = [m.group().casefold() for m in ws]
    if native is not None:
        if not np.array_equal(native['char'][:,:2],np.array([m.span() for m in ws])):
            raise ValueError('Native lexical offsets mismatch')
        r = native['rep'].astype('float32')
        link = native['links'].astype('float32')
        sur = native['surprisal']
        wrs = native['words'].astype('float32')
    history=Counter();index=0;rows=[];names=[];x=[];sentence=[];starts=[]
    for j,(a,b) in enumerate(regions):
        lo=index
        while index<len(ws) and ws[index].end()<=b:index+=1
        ts=tokens[lo:index];original=[m.group() for m in ws[lo:index]]
        if not ts:raise ValueError('No words in sentence')
        meaningful=[v for v in ts if v not in STOP];bag=sketch(meaningful+[u+' '+v for u,v in zip(ts,ts[1:])],48)
        row=dict(tokens=tuple(ts),content=set(meaningful),bag=bag.astype('float32'),character=b-a,
                 surface=np.array([np.log1p(j),np.log1p(len(ts)),np.log1p(b-a),np.mean([v in history for v in ts]),
                    float(any(v in {'not','never','no','without'} for v in ts)),float(any(v in {'again','still','already','instead'} for v in ts))]))
        if native is not None:
            value=sur[lo:index];row.update(surp=np.array([value.mean(),value.max(),value[0],value[-1],np.quantile(value,.75)]),
                       reps=np.stack([norm(q) for q in np.split(r[j],5)]),att=link[j])
        starts.append(len(x));endpoints={}
        for m,v in enumerate(ts):
            if v in STOP:continue
            # Last visible occurrence represents a repeated lexical entity at this completed action.
            for width in range(1,min(3,m+1)+1):
                span=ts[m-width+1:m+1]
                if span[0] not in STOP:endpoints[' '.join(span)]=m
        for v,m in endpoints.items():
            wi=lo+m
            est=np.r_[np.log1p(m),np.log1p(len(ts)-m),np.log1p(len(v)),np.log1p(history[v]),
               float(original[m][0].isupper()),float(m==0),row['surface'],bag,
               sketch((v,ts[max(m-1,0)]+'__', '__'+ts[min(m+1,len(ts)-1)],v[-3:]),24)]
            if native is not None:
                est=np.r_[est,wrs[wi],r[j,0:32],r[j,96:128],sur[wi]]
            x.append(est.astype('float32'));names.append(v);sentence.append(j)
        rows.append(row);history.update(ts)
    starts.append(len(x));return dict(rows=rows,names=names,regions=regions,x=np.stack(x),sentence=np.array(sentence),starts=starts)


def align(record,doc):
    labels=np.full((len(doc['x']),len(ATTR),2),-2,dtype=np.int8);counts=Counter()
    for j,states in enumerate(record['states']):
        ts=doc['rows'][j]['tokens'];lookup={doc['names'][i]:i for i in range(*doc['starts'][j:j+2])}
        for a,name in enumerate(ATTR):
            for entity,code in states.get(name,[]):
                counts['source_attribute_entries']+=1
                target=tuple(m.group().casefold() for m in WORD.finditer(entity))
                text=' '.join(target)
                if text not in lookup:continue
                i=lookup[text];expected=CODES[code];present=labels[i,a]
                if min(present)>-2 and tuple(present)!=expected:
                    raise ValueError('Two distinct source entities map to incompatible states at the same mention')
                labels[i,a]=expected;counts['aligned_attribute_entries']+=1
    return labels,counts


def standard(x,w):
    w=w/w.sum();mu=w@x;sd=np.sqrt(w@((x-mu)**2));sd[sd<1e-5]=1
    return mu,sd


class StateEstimator:
    def __init__(self,x,y,w):
        self.mu,self.sd=standard(x,w);z=np.column_stack((np.ones(len(x)),(x-self.mu)/self.sd))
        flat=y.reshape(len(y),-1);known=flat>=0;offered=flat>=-1
        # One conditional truth and one observation likelihood per source-coded property and side.
        positives=np.column_stack((flat==1,known));mask=np.column_stack((known,offered));mw=w[:,None]*mask
        mass=mw.sum(0);mw=mw/np.where(mass>0,mass,1);pos=(mw*positives).sum(0)
        active=(mass>0)&(pos>1e-8)&(pos<1-1e-8);self.active=active
        # With a single observed class, Jeffreys smoothing supplies an intercept estimated only on source judgments.
        raw=(positives*mask).sum(0);n=mask.sum(0);p=(raw+.5)/(n+1.);self.v=np.zeros((z.shape[1],len(pos)));self.v[0]=np.log(p/(1-p))
        a=positives[:,active].astype(float);weights=mw[:,active];n_out=a.shape[1]
        def objective(raw):
            v=raw.reshape(z.shape[1],n_out);linear=z@v;loss=(weights*(np.logaddexp(0,linear)-a*linear)).sum()+.0125*np.square(v[1:]).sum()
            grad=z.T@(weights*(expit(linear)-a));grad[1:]+=.025*v[1:];return loss,grad.ravel()
        fit=minimize(objective,self.v[:,active].ravel(),jac=True,method='L-BFGS-B',options={'maxiter':450,'gtol':1e-6,'ftol':1e-12,'maxcor':8})
        if not fit.success or abs(fit.jac).max()>5e-5:raise ArithmeticError(f'Entity-state likelihood failed to converge {fit.message} {abs(fit.jac).max()}')
        self.v[:,active]=fit.x.reshape(z.shape[1],n_out);self.diag=dict(active=int(active.sum()),measurements=len(y),gradient=float(abs(fit.jac).max()),iterations=int(fit.nit))

    def predict(self,x):
        return predict_attributes(x,self.mu,self.sd,self.v)


def predict_attributes(x,mean,scale,coefficients):
    p=expit(((x-mean)/scale)@coefficients[1:]+coefficients[0])
    value,known=np.split(p,2,axis=1)
    return np.stack([a.reshape(len(x),len(ATTR),2) for a in (value,known)],axis=-1)


def state_training(records,archive):
    all_x=[];all_y=[];w=[];n_by=Counter(r['story_id'] for r,k in records);coverage=Counter()
    for r,k in records:
        doc=archive.document(k);y,c=align(r,doc);coverage.update(c);mask=y.reshape(len(y),-1).max(1)>=-1
        if not mask.any():continue
        all_x.append(doc['x'][mask]);all_y.append(y[mask]);w.extend([1/(n_by[r['story_id']]*mask.sum())]*mask.sum())
    model=StateEstimator(np.vstack(all_x),np.concatenate(all_y),np.array(w));model.diag.update(coverage);return model
