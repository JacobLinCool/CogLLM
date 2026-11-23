"""Fit convex, view-specific term and definition objectives on the registered textbook domains."""

import argparse
from collections import Counter,defaultdict
from hashlib import sha256
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from research.backbone_transfer.extract import digest,read_json,stream_rows,write_json
from research.reading_validation.definition_data import OUT,ROOT
from research.reading_validation.definition_features import Corpus
from research.reading_validation.identity_causal_fit import loss_and_grad


ARMS=('position','surface','semantic')
GRID=(.03,.15,.75)
TASKS={'term':{'term':.5,'other':.5},'link':{'identity':.30,'interval':.35,'definition_word':.175,'other_word':.175}}


def random_stream(key,suffix):
    return np.random.default_rng(int.from_bytes(sha256((key+suffix).encode()).digest()[:8],'big'))


def term_rows(page,row,arm):
    left,right=page.grid();truth={v for coords in row['terms'].values() if (v:=page.word_span(coords)) is not None}
    pos=np.array([tuple(z) in truth for z in zip(left,right)],dtype=bool)
    rng=random_stream(row['id'],'terms');negative=np.flatnonzero(~pos)
    ordinary=rng.choice(negative,size=min(len(negative),max(8,12*len(truth))),replace=False)
    nearby=np.flatnonzero(~pos & (np.isin(left,[a for a,b in truth])|np.isin(right,[b for a,b in truth])))
    focus=rng.choice(nearby,size=min(len(nearby),8*len(truth)),replace=False)
    ids=np.unique(np.r_[np.flatnonzero(pos),ordinary,focus]);X=page.span_features(left[ids],right[ids],'surface' if arm=='position' else arm)
    if arm=='position':X=X[:,[0,16,17,18]]
    signs=np.where(pos[ids],1.,-1.)
    features=np.column_stack((X,np.ones(len(ids))))*signs[:,None]
    kinds=['term' if value else 'other' for value in pos[ids]]
    return features.astype('float32'),kinds


def overlap(a,b):
    return a[0]<b[1] and b[0]<a[1]


def groups(page,row):
    gold={i:a for i,c in row['terms'].items() if (a:=page.word_span(c)) is not None}
    original=[(x['term'],page.word_interval(x['definition'])) for x in row['links']]
    all_bands=defaultdict(list)
    for q,band in original:
        if band:all_bands[q].append(band)
    for j,(name,band) in enumerate(original):
        if name not in gold or band is None:continue
        span=gold[name];s,e=band;b=span[1]
        if b>=s:continue
        choices={o:v for o,v in gold.items() if o!=name and v[1]<s and not any(overlap(band,g) for g in all_bands[o])}
        rng=random_stream(row['id'],f'link{j}')
        length=e-s
        starts=[t for t in range(b+1,page.n-length+1)
                if not any(overlap((t,t+length),p) for p in all_bands[name])]
        if starts:
            # Stable ties alternate direction by item without selecting an observed model outcome.
            sides=rng.random(len(starts));best=min(zip(starts,sides),key=lambda p:(min(abs(p[0]+length-s),abs(p[0]-e)),p[1]))[0]
            other=(best,best+length)
        else:other=None
        zeros=[k for k in range(b+1,page.n) if not any(a<=k<f for a,f in all_bands[name])]
        negword=int(rng.choice(zeros)) if zeros else None
        yield name,span,band,choices,other,int(rng.integers(s,e)),negword


def link_rows(page,row,arm):
    records=[];kinds=[]
    for name,span,(s,e),choices,negative,positive_word,negative_word in groups(page,row):
        pos=page.link_features(span,np.arange(s,e),arm);mean=pos.mean(0)
        if choices:
            # One competing human-marked term per event avoids overweighting dense windows.
            rival=min(choices,key=lambda k:sha256((row['id']+name+k).encode()).digest());other=page.link_features(choices[rival],np.arange(s,e),arm).mean(0)
            records.append(np.r_[mean-other,0.]);kinds.append('identity')
        if negative is not None:
            a,b=negative;other=page.link_features(span,np.arange(a,b),arm).mean(0)
            records.append(np.r_[mean-other,0.]);kinds.append('interval')
        if negative_word is not None:
            records.append(np.r_[pos[positive_word-s],1.]);kinds.append('definition_word')
            other=page.link_features(span,[negative_word],arm)[0];records.append(-np.r_[other,1.]);kinds.append('other_word')
    return np.asarray(records,dtype='float32'),kinds


def data(bank,rows,arm,task):
    X=[];labels=[]
    for r in rows:
        v,g=(term_rows if task=='term' else link_rows)(bank.page(r['id']),r,arm)
        if len(g):
            X.append(v);labels.extend((k,r['source']) for k in g)
    counts=Counter(labels);participating=Counter(k for k,source in counts);expected=TASKS[task]
    if set(participating)!=set(expected):raise ValueError(f'One training objective missing {participating}')
    w=np.array([expected[k]/participating[k]/counts[k,s] for k,s in labels],dtype='float32')
    return np.concatenate(X),w,dict(Counter(k for k,s in labels))


def fitted(bank,train,validation,arm,task):
    X,weights,count=data(bank,train,arm,task);Y,vw,vc=data(bank,validation,arm,task)
    scale=np.sqrt(np.einsum('ij,ij->j',X,X,dtype='float64')/len(X)).astype('float32')
    scale[scale<.012]=1;scale[-1]=1;X/=scale;Y/=scale;candidates=[]
    for penalty in GRID:
        res=minimize(loss_and_grad,np.zeros(X.shape[1]),args=(X,weights,penalty),method='L-BFGS-B',jac=True,
                     options={'maxiter':600,'gtol':3e-6,'ftol':1e-12,'maxls':50})
        grad=float(max(abs(res.jac)));val=float(vw@np.logaddexp(0.,-(Y@res.x.astype('float32')).astype('float64')))
        if not np.isfinite(res.fun) or grad>1.5e-4:raise ArithmeticError(f'Unconverged {task}/{arm}: {res.message} gradient={grad}')
        candidates.append({'penalty':penalty,'calibration':val,'objective':float(res.fun),'gradient':grad,
                           'iterations':int(res.nit),'weights':(res.x[:-1]/scale[:-1]).tolist(),'bias':float(res.x[-1])})
    m=min(candidates,key=lambda c:c['calibration']);m['trials']=[{k:v for k,v in u.items() if k not in ('weights','bias','trials')} for u in candidates]
    m['train_counts']=count;m['calibration_counts']=vc
    return m


def score(m,X):
    return np.asarray(X,dtype='float32')@np.asarray(m['weights'],dtype='float32')+m['bias']


def all_spans(page,model,arm):
    l,r=page.grid();X=page.span_features(l,r,'surface' if arm=='position' else arm)
    if arm=='position':X=X[:,[0,16,17,18]]
    return l,r,score(model,X)


def threshold(bank,rows,m,arm):
    p=[];y=[]
    for r in rows:
        page=bank.page(r['id']);l,b,x=all_spans(page,m,arm);yes={page.word_span(v) for v in r['terms'].values()}
        targets=np.array([(a,j) in yes for a,j in zip(l,b,strict=True)],dtype=bool)
        p.extend(x);y.extend(targets)
    p=np.asarray(p);y=np.asarray(y);o=np.argsort(-p,kind='stable');v=p[o];a=y[o].cumsum();n=np.arange(1,len(p)+1)
    stops=np.r_[v[:-1]!=v[1:],True];f=2*a/(n+int(y.sum()));f[~stops]=-1;i=int(f.argmax())
    return {'threshold':float(v[i]),'calibration_f1':float(f[i]),'num_candidates':len(y),'num_true':int(y.sum())}


def train(root):
    bank=Corpus(root);seal=read_json(OUT/'input-seal.json')
    for split in ('train','calibration'):
        if digest(ROOT/(split+'.jsonl'))!=seal['grades_sha256'][split]:raise ValueError('Source or calibration labels changed')
    train=list(stream_rows(ROOT/'train.jsonl'));val=list(stream_rows(ROOT/'calibration.jsonl'));fit={}
    for task in ('term','link'):
        fit[task]={}
        for arm in ARMS:
            m=fitted(bank,train,val,arm,task)
            if task=='term':m.update(threshold(bank,val,m,arm))
            fit[task][arm]=m
            print(task,arm,m['calibration'],m['gradient'],flush=True)
    fit['source_sha256']=seal['inputs_sha256'];fit['version']=1;write_json(OUT/'weights.json',fit)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);train(p.parse_args().root)
