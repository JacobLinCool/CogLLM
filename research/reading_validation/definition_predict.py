"""Freeze article-only top antecedents and supplied-term explanatory trajectories before loading outcomes."""

import argparse
from hashlib import sha256
from pathlib import Path

import numpy as np
from scipy.special import expit, log_expit

from research.backbone_transfer.extract import digest,read_json,stream_rows,write_json
from research.reading_validation.definition_data import OUT, ROOT
from research.reading_validation.definition_features import Corpus
from research.reading_validation.definition_fit import ARMS,all_spans,score


def best_candidates(page,logits):
    l,r=page.grid()
    order=np.lexsort((l,r,-logits));inventory=np.full((page.n,8),-1,dtype=np.int32)
    for t in range(1,page.n):
        keep=[];occupied=np.zeros(t,dtype=bool)
        for i in order[r[order]<t]:
            if occupied[l[i]:r[i]+1].any():continue
            keep.append(i);occupied[l[i]:r[i]+1]=True
            if len(keep)==8:break
        inventory[t,:len(keep)]=keep
    return inventory


def feature_series(page,span,weights,arm):
    n=page.n;a,b=span;values=np.full(n,np.nan,dtype='float32');prob=np.zeros(n,dtype='float32')
    if b+1<n:
        ts=np.arange(b+1,n);features=page.link_features(span,ts,arm);v=score(weights,features)
        before=np.r_[0.,v.astype('float64').cumsum()];i=np.arange(1,len(v)+1);lo=np.maximum(0,i-4)
        values[ts]=((before[i]-before[lo])/(i-lo)).astype('float32')
        prob[ts]=expit(values[ts])
    return values,prob


def get_forecasts(page,models,queries):
    l,r=page.grid();idx={(int(a),int(b)):i for i,(a,b) in enumerate(zip(l,r,strict=True))}
    q=[page.word_span(ab) for _,ab in sorted(queries.items())];qidx=np.array([idx.get(row,-1) for row in q],dtype='int32')
    data={'left':l.astype('int16'),'right':r.astype('int16'),'qidx':qidx}
    for arm in (*ARMS,'joint'):
        design='semantic' if arm=='joint' else arm
        _,_,lp=all_spans(page,models['term'][arm],design)
        bag=best_candidates(page,lp)
        ids=np.unique(np.r_[bag[bag>=0],qidx[qidx>=0]])
        lookup=np.full(len(lp),-1,dtype=np.int32);lookup[ids]=np.arange(len(ids));full=np.zeros((len(ids),page.n),dtype='float32')
        p=np.zeros_like(full)
        for i,si in enumerate(ids):
            full[i],p[i]=feature_series(page,(l[si],r[si]),models['link'][arm],design)
        auto=np.full(page.n,-1,dtype=np.int32)
        for t,props in enumerate(bag):
            avail=props[props>=0]
            if len(avail):
                logjoint=log_expit(lp[avail])+log_expit(full[lookup[avail],t]);auto[t]=avail[np.argmax(logjoint)]
        qprob=np.zeros((len(qidx),page.n),dtype='float32');gate=np.zeros(len(qidx),dtype=bool)
        for j,si in enumerate(qidx):
            if si<0:continue
            qprob[j]=p[lookup[si]];stop=r[si];ending=np.flatnonzero(r==stop);best=ending[np.argmax(lp[ending])]
            gate[j]=bool(best==si and lp[si]>=models['term'][arm]['threshold'])
        data.update({arm+'_term':lp.astype('float32'),arm+'_proposals':bag,arm+'_top':auto,arm+'_qprob':qprob,
                     arm+'_detected_query':gate,arm+'_pool':ids.astype('int32'),arm+'_pool_prob':p})
    return data


def forecast(root,output):
    models=read_json(OUT/'weights.json');lock=read_json(OUT/'input-seal.json');bank=Corpus(root)
    if models['source_sha256']!=lock['inputs_sha256']:raise ValueError('Features or fitted coefficients changed')
    j=read_json(OUT/'joint-weights.json')
    if j['base_weights_sha256']!=digest(OUT/'weights.json'):raise ValueError('Joint model base mismatch')
    for task in ('term','link'):models[task]['joint']=j[task]
    layout=list(stream_rows(ROOT/'probe-layout-v2.jsonl'));output.mkdir(parents=True,exist_ok=True)
    hashes={}; counts={}
    for i,q in enumerate(layout):
        case=q['id'];path=output/(case+'.npz');P=get_forecasts(bank.page(case),models,q['query_terms'])
        np.savez_compressed(path,**P);hashes[case]=digest(path);counts[q['kind']]=counts.get(q['kind'],0)+1
        if (i+1)%500==0:print(f'Frozen term predictions {i+1}/{len(layout)}',flush=True)
    write_json(OUT/'forecast-seal.json',{'inputs_sha256':lock['inputs_sha256'],'weights_sha256':digest(OUT/'weights.json'),
               'query_geometry_sha256':digest(ROOT/'probe-layout-v2.jsonl'),'protocol_sha256':lock['protocol_sha256'],
               'joint_weights_sha256':digest(OUT/'joint-weights.json'),'models':[*ARMS,'joint'],'counts':counts,'predictions_sha256':hashes,'definition_relations_or_reader_ratings_loaded':False})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--output',type=Path,default=ROOT/'forecasts')
    a=p.parse_args();forecast(a.root,a.output)
