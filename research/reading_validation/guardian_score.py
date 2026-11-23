"""Join frozen article estimates to directly reported human appraisals with article-level uncertainty."""
from collections import Counter,defaultdict
from argparse import ArgumentParser
from pathlib import Path
import csv
import numpy as np
from scipy.stats import rankdata
from research.backbone_transfer.extract import digest,read_json,write_json
from research.reading_validation.guardian_sources import OUT,ROOT


def correlation(x,y):
    a=rankdata(x,axis=-1);b=rankdata(y,axis=-1);a-=a.mean(axis=-1,keepdims=True);b-=b.mean(axis=-1,keepdims=True)
    den=np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1)
    return np.divide((a*b).sum(axis=-1),den,out=np.full(np.shape(den),np.nan),where=den>0)


def interval(v):
    v=np.asarray(v);v=v[np.isfinite(v)];return np.quantile(v,[.025,.975]).tolist() if len(v) else None


def analyse(predictions,article_ids,responses,mode):
    yy=[];xx=[];samples=0
    for i,row in article_ids.items():
        valid=responses[i];f=np.array([x['familiarity'] for x in valid]);median=np.median(f)
        if mode=='lower':valid=[x for x in valid if x['familiarity']<median]
        if mode=='higher':valid=[x for x in valid if x['familiarity']>median]
        if not valid:continue
        yy.append([np.mean([r[k] for r in valid]) for k in ('difficulty','coherence','reading')]);xx.append(predictions[row]);samples+=len(valid)
    y=np.array(yy);n=len(y);rg=np.random.default_rng(85356)
    bootstrap=rg.integers(0,n,(10000,n));orders=np.argsort(rg.uniform(size=(10000,n)),axis=1);result={};tests=[]
    for j,endpoint in enumerate(('difficulty','coherence','reading')):
        names=('difficulty','coherence','ARI','coleman_liau','surprisal','ease_difficulty','peak_transition','mean_unresolved','peak_unresolved')
        stats={};bootstats={};truth=y[:,j]
        for name in names:
            x=np.array([v[name] for v in xx]);rho=float(correlation(x,truth));boot=correlation(x[bootstrap],truth[bootstrap]);bootstats[name]=boot
            pp=correlation(np.broadcast_to(x,orders.shape),truth[orders]);pv=(1+np.sum(abs(pp)>=abs(rho)))/(1+len(pp))
            stats[name]={'rho':rho,'article_bootstrap_95':interval(boot),'permutation_p':float(pv)}
        prime={'difficulty':'difficulty','coherence':'coherence','reading':'ease_difficulty'}[endpoint]
        for comp in ('ARI','coleman_liau'):
            stats[prime]['gain_vs_'+comp]={'difference':stats[prime]['rho']-stats[comp]['rho'],'paired_bootstrap_95':interval(bootstats[prime]-bootstats[comp])}
        result[endpoint]=stats
        if endpoint in ('difficulty','coherence'):tests.append((stats[prime]['permutation_p'],endpoint))
    largest=0.
    for j,(pv,endpoint) in enumerate(sorted(tests)):
        largest=max(largest,(len(tests)-j)*pv);result[endpoint][endpoint]['holm_p']=min(1.,largest)
    return {'articles':n,'reading_sessions':samples,'correlations':result}


def main(forecasts):
    seal=read_json(OUT/'input-seal.json');receipt=read_json(forecasts/'receipt.json')
    if (seal['stimuli']!=receipt['input'] or receipt['forecasts']!=digest(forecasts/'predictions.json') or
            digest(ROOT/'guardian2.csv')!=seal['outcomes'] or receipt['protocol']!=seal['protocol']):raise ValueError('Article, model or human-data provenance changed')
    source=read_json(OUT/'source-recovery.json');ids={r['id']:r for r in source['articles'] if r['status']=='both_anchors'}
    pp=read_json(forecasts/'predictions.json');responses=defaultdict(list);dx=[];official=[]
    with (ROOT/'guardian2.csv').open(newline='',encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            k=int(r['stimulus'])
            if k not in ids:continue
            def item(n):return float(r[f'novelty_comprehension_scale_{n}'])
            difficulty=float(np.mean([item(n) for n in (60,61,62)]));dx.append(difficulty);official.append(float(r['appraised_comprehensibility']))
            responses[k].append({'difficulty':difficulty,'coherence':item(62),'reading':item(59),'familiarity':float(r['topical_familiarity'])})
    direction=float(correlation(np.array(dx),np.array(official)))
    if not np.isfinite(direction) or direction>-.95:raise ValueError(f'Released comprehensibility orientation disagrees with registered raw item direction: {direction}')
    results={version:{mode:analyse(pp,{i:r[version+'_id'] for i,r in ids.items()},responses,mode) for mode in ('all','lower','higher')}
             for version in ('body','current_title')}
    write_json(OUT/'scores.json',{'forecasts':digest(forecasts/'predictions.json'),'source':seal,
        'raw_difficulty_vs_released_comprehensibility_correlation':direction,'article_effective_sample':len(ids),'results':results})
    write_json(OUT/'article-scores.json',{str(i):{'sessions':len(responses[i]),'observed':{k:float(np.mean([x[k] for x in responses[i]])) for k in responses[i][0]},
            'primary_model':pp[r['body_id']],'current_title_sensitivity':pp[r['current_title_id']]} for i,r in ids.items()})


if __name__=='__main__':
    p=ArgumentParser(description=__doc__);p.add_argument('forecasts',type=Path);main(p.parse_args().forecasts)
