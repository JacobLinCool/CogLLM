"""Jointly tune term and link readouts with the same probability used to rank antecedents at deployment."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit,log_expit

from research.backbone_transfer.extract import digest,read_json,stream_rows,write_json
from research.reading_validation.definition_data import OUT,ROOT
from research.reading_validation.definition_features import Corpus
from research.reading_validation.definition_fit import GRID,all_spans,data,groups,score,threshold


def pairs(bank,rows,models):
    out=[];labels=[]
    for row in rows:
        pg=bank.page(row['id']);layouts=list(groups(pg,row))
        if not layouts:continue
        l,r,prop=all_spans(pg,models['term']['semantic'],'semantic');order=np.lexsort((l,r,-prop));grid={(a,b):i for i,(a,b) in enumerate(zip(l,r,strict=True))}
        z=pg.span_features(l,r,'semantic');true_by_stop={}
        for name,span,(s,e),*_ in layouts:true_by_stop.setdefault(e,set()).add(span)
        for name,true,(s,e),*_ in layouts:
            t=e-1; occupied=np.zeros(t,dtype=bool);proposals=[]
            for j in order[r[order]<t]:
                if occupied[l[j]:r[j]+1].any():continue
                proposals.append(j);occupied[l[j]:r[j]+1]=True
                if len(proposals)==8:break
            indices=[j for j in proposals if (l[j],r[j]) not in true_by_stop[e]]
            if not indices:continue
            def features(j):
                a,b=l[j],r[j];ts=np.arange(max(b+1,t-3),t+1)
                return pg.link_features((a,b),ts,'semantic').mean(0)
            rel=np.stack([features(j) for j in indices]);target=grid[true];good=features(target)
            auto=log_expit(prop[indices])+log_expit(score(models['link']['semantic'],rel));ii=int(np.argmax(auto));bad=indices[ii]
            out.append([z[target],z[bad],good,rel[ii]]);labels.append(row['source'])
    if not out:raise ValueError('No nontrivial automatic candidates')
    counts=Counter(labels);weights=np.array([1/(len(counts)*counts[x]) for x in labels],dtype='float32')
    return [np.column_stack((x,np.ones(len(x)))).astype('float32') for x in zip(*out,strict=True)],weights


def full_loss(delta,old,scale,dataset,penalty):
    tx,tw,lx,lw,pair,pw=dataset;nt=tx.shape[1];v=old+delta;ti,li=v[:nt].astype('float32'),v[nt:].astype('float32')
    zt=(tx@ti).astype(float);zl=(lx@li).astype(float)
    a,b,c,d=pair;ta=(a@ti).astype(float);tb=(b@ti).astype(float);lc=(c@li).astype(float);ld=(d@li).astype(float)
    diff=log_expit(ta)-log_expit(tb)+log_expit(lc)-log_expit(ld)
    obj=.25*(tw@np.logaddexp(0.,-zt)+lw@np.logaddexp(0.,-zl))+.5*pw@np.logaddexp(0.,-diff)
    common=(-.5*pw*expit(-diff))
    gt=tx.T@(-.25*tw*expit(-zt)).astype('float32')+a.T@(common*expit(-ta)).astype('float32')-b.T@(common*expit(-tb)).astype('float32')
    gl=lx.T@(-.25*lw*expit(-zl)).astype('float32')+c.T@(common*expit(-lc)).astype('float32')-d.T@(common*expit(-ld)).astype('float32')
    return float(obj+penalty*.5*(delta@delta)),np.r_[gt,gl].astype(float)+penalty*delta


def build(bank,rows,base):
    T,tw,_=data(bank,rows,'semantic','term');L,lw,_=data(bank,rows,'semantic','link');P,rw=pairs(bank,rows,base)
    return T,tw,L,lw,P,rw


def fit(root):
    bank=Corpus(root);models=read_json(OUT/'weights.json');rows=list(stream_rows(ROOT/'train.jsonl'));val=list(stream_rows(ROOT/'calibration.jsonl'))
    training=build(bank,rows,models);validation=build(bank,val,models)
    t,l=training[0],training[2];st=np.sqrt(np.einsum('ij,ij->j',t,t,dtype=float)/len(t)).astype('float32')
    sl=np.sqrt(np.einsum('ij,ij->j',l,l,dtype=float)/len(l)).astype('float32')
    for ss in (st,sl):ss[ss<.012]=1;ss[-1]=1.
    for tx,tw,lx,lw,px,pw in (training,validation):
        tx/=st;lx/=sl
        for i,arr in enumerate(px):arr/=st if i<2 else sl
    old=np.r_[np.r_[models['term']['semantic']['weights'],models['term']['semantic']['bias']]*st,
              np.r_[models['link']['semantic']['weights'],models['link']['semantic']['bias']]*sl]
    rec=[];deltas=[]
    for lam in GRID:
        res=minimize(full_loss,np.zeros_like(old),args=(old,None,training,lam),method='L-BFGS-B',jac=True,
                     options={'maxiter':500,'ftol':1e-12,'gtol':3e-6,'maxls':50})
        grad=float(abs(res.jac).max());loss=full_loss(res.x,old,None,validation,0.)[0]
        if not np.isfinite(loss) or grad>.00015:raise ArithmeticError(f'Joint stationarity {grad}')
        rec.append({'penalty':lam,'val':loss,'objective':float(res.fun),'gradient':grad,'iterations':int(res.nit)});deltas.append(res.x)
    ix=int(np.argmin([r['val'] for r in rec]));w=old+deltas[ix];a=w[:len(st)]/st;b=w[len(st):]/sl
    result={'base_weights_sha256':digest(OUT/'weights.json'),'trials':rec,'selection_index':ix,'train_pairs':len(training[-1]),
            'calibration_pairs':len(validation[-1]),'term':{'weights':a[:-1].tolist(),'bias':float(a[-1])},'link':{'weights':b[:-1].tolist(),'bias':float(b[-1])}}
    result['term'].update(threshold(bank,val,result['term'],'semantic'));write_json(OUT/'joint-weights.json',result)
    print('Jointly optimized source/pair loss',len(training[-1]),rec[ix],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);fit(p.parse_args().root)
