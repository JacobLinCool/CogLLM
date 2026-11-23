"""Evaluate an expected count of unresolved *annotated* terms on familiar and independent research prose."""
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from research.backbone_transfer.extract import digest,read_json,stream_rows,write_json
from research.reading_validation.definition_data import OUT,ROOT
from research.reading_validation.definition_features import Corpus
from research.reading_validation.definition_stock import evidence,events
from research.reading_validation.definition_stock_fit import gold,load_models
from research.reading_validation.identity_causal_score import confidence
from research.reading_validation.stock_probability import ARMS,predict


def bounds(n,g):
    lower=np.zeros(n);upper=np.zeros(n);event=np.zeros(n,dtype=bool)
    for b,regions in g.values():
        lo,hi=regions[0];lower[b:lo]+=1.;upper[b:hi-1]+=1.;event[lo:hi]=True
    return lower,upper,event


def metrics(v,lower,upper):
    d=np.maximum(lower-v,0.)+np.maximum(v-upper,0.);known=upper==lower
    pos=lower>=1;neg=upper==0
    return np.array([len(v),abs(d).sum(),(d*d).sum(),v.sum(),pos.sum(),v[pos].sum(),neg.sum(),v[neg].sum(),known.sum(),
                     ((v[known]-lower[known])**2).sum()])


def values(row):
    n,ab,sq,level,p,predp,q,predq,k,knownsq=row
    return dict(nwords=int(n),positive_words=int(p),certain_empty_words=int(q),certain_words=int(k),mean_count=level/n,
                interval_mae=ab/n,interval_mse=sq/n,mean_on_certain_positive=predp/p if p else None,
                mean_on_certain_empty=predq/q if q else None,certain_mse=knownsq/k if k else None)


def average_precision(rows):
    x=np.concatenate(rows);target=x[:,0].astype(bool);score=x[:,1];order=np.argsort(-score,kind='stable')
    score=score[order];y=target[order]
    if not y.sum():return None
    stops=np.r_[np.flatnonzero(np.diff(score)!=0),len(y)-1];t=np.cumsum(y)[stops];rec=t/y.sum();pr=t/(stops+1)
    return float(np.diff(np.r_[0.,rec])@pr)


def summarize(cases,continuous,primary='semantic'):
    arms=list(next(iter(cases.values())));all_metrics={a:values(sum(r[a] for r in cases.values())) for a in arms}
    by={s:{a:values(row[a]) for a in arms} for s,row in cases.items()}
    gain={a:{m:confidence([v[a][m]-v[primary][m] for v in by.values() if v[a][m] is not None and v[primary][m] is not None])
             for m in ('interval_mse','interval_mae','certain_mse')} for a in arms if a!=primary}
    rank={a:average_precision(rows) for a,rows in continuous.items()}
    return {'pooled':all_metrics,'groups':len(cases),'group_weighted_error_reduction_semantic_vs':gain,'certain_pending_average_precision':rank}


def score(bank_path,external,scaled=False):
    fits=read_json(OUT/'stock-probability-weights.json');models=load_models();cal=read_json(OUT/'stock-thresholds.json')['thresholds']
    if fits['base_weights']!=digest(OUT/'weights.json'):raise ValueError('Unexpected underlying link head')
    if external:
        lock=read_json(OUT/'stock-external-lock.json');src=ROOT/'w00-stimuli.jsonl'
        if digest(ROOT/'w00-structural.jsonl')!=lock['grades']:raise ValueError('External source has changed')
        truth=[]
        for r in stream_rows(ROOT/'w00-structural.jsonl'):
            if r['kind'] not in ('negative','single_forward'):continue
            q=r['terms'];defs=r['definitions']
            truth.append({'id':r['id'],'subject':'w00','source':r['source_section'],'terms':{'t':q[0]} if q else {},
                          'links':[{'term':'t','definition':defs[0]}] if q else []})
    else:
        truth=list(stream_rows(ROOT/'test.jsonl'));src=ROOT/'stimuli.jsonl'
    calibration=read_json(OUT/'stock-count-scales.json') if scaled else None
    if scaled and calibration['weights']!=digest(OUT/'stock-probability-weights.json'):raise ValueError('Unexpected count calibration')
    bank=Corpus(bank_path,src);grouped=defaultdict(dict);items=defaultdict(dict);ranks=defaultdict(list);count=0
    for r in truth:
        pg=bank.page(r['id']);g=gold(pg,r)
        lo,hi,_=bounds(pg.n,g);certain=(lo>=1)|(hi==0)
        curves={'zero':np.zeros(pg.n)}
        for a in ARMS:
            out=predict(pg,a,fits['fits'][a],models[a]);curves[a]=out.unresolved.sum(1);curves[a+'_median']=out.median()
            if scaled:curves[a+'_scaled']=curves[a]*calibration['scales'][a]['scale']
        if scaled:curves['constant']=np.full(pg.n,calibration['scales']['constant']['scale'])
        prev=evidence(pg,*models['semantic'],'semantic');curves['hard_ledger']=events(prev,cal['semantic']['threshold'])[2]
        for a,curve in curves.items():
            v=metrics(curve,lo,hi);group=r['source']
            grouped[group][a]=grouped[group].get(a,0.)+v;items[r['id']][a]=v
            ranks[a].append(np.column_stack((lo[certain]>=1,curve[certain])))
        count+=1
        if count%500==0:print(('W00' if external else 'DEFT'),count,len(truth),flush=True)
    report={'fit':digest(OUT/'stock-probability-weights.json'),'source':digest(src),'evaluated_documents':count,
        'source_group':summarize(grouped,ranks,'semantic_scaled' if scaled else 'semantic')}
    if external:report['sentence']=summarize(items,ranks,'semantic_scaled' if scaled else 'semantic')
    if scaled:report['scale_hash']=digest(OUT/'stock-count-scales.json')
    name=('w00' if external else 'deft')+('-scaled' if scaled else '');write_json(OUT/('stock-probability-'+name+'.json'),report)
    write_json(OUT/('stock-probability-'+name+'-source.json'),{k:{m:values(v) for m,v in item.items()} for k,item in grouped.items()})
    print({a:round(v['interval_mse'],4) for a,v in report['source_group']['pooled'].items()},flush=True)


if __name__=='__main__':
    p=ArgumentParser(description=__doc__);p.add_argument('bank',type=Path);p.add_argument('--external',action='store_true')
    p.add_argument('--scaled',action='store_true')
    a=p.parse_args();score(a.bank,a.external,a.scaled)
