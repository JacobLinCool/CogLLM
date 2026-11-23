"""Score original fixed-width pointers and curve at every actual Qwen token."""
from argparse import ArgumentParser
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import rankdata

from research.backbone_transfer.extract import digest,read_json,stream_rows,write_json
from research.reading_validation.definition_data import OUT,ROOT
from research.reading_validation.definition_features import Corpus
from research.reading_validation.definition_stock import evidence,events
from research.reading_validation.definition_stock_fit import gold,load_models,score_one,summarize
from research.reading_validation.identity_causal_score import confidence
from research.reading_validation.stock_probability import predict
from research.reading_validation.stock_probability_score import bounds,metrics,values

Q=(0.,.10,.25,.50,.70,.85,.92,.96,.98,.99,1.)


def trim(text,a,b):
    if a<0 or b<=a:return None
    while a<b and text[a].isspace():a+=1
    while b>a and text[b-1].isspace():b-=1
    return (a,b) if a<b else None


def geometry(page,tensor,r):
    text=page.text;start,end,anchor,left,right=tensor[:,:5].astype(int).T
    src=[trim(text,a,b) if any(c.isalnum() for c in text[s:e]) else None for a,b,s,e in zip(anchor,end,start,end,strict=True)]
    ref=[trim(text,a,b) for a,b in zip(left,right,strict=True)];g=gold(page,r);by=defaultdict(list)
    for link in r['links']:
        ab=tuple(r['terms'][link['term']])
        if ab in g:by[ab].append(tuple(link['definition']))
    return src,ref,end,by


def native_count(tensor,geom,op,res):
    src,refs,ends,g=geom;sources=set();closed=set();hits=set();overlap=set();c=Counter(gold=len(g),n_words=len(ends))
    for n,(s,ref,end) in enumerate(zip(src,refs,ends,strict=True)):
        if ref in sources and ref not in closed and tensor[n,7]>=res and tensor[n,7]>0:
            closed.add(ref);c['emitted']+=1
            active={a for a,v in g.items() if any(lo<end<=hi for lo,hi in v)}
            if ref in active:hits.add(ref)
            choices=[a for a in active-overlap if ref[0]<a[1] and a[0]<ref[1]]
            if choices:overlap.add(max(choices,key=lambda ab:(min(ref[1],ab[1])-max(ref[0],ab[0]),ab)))
        if s is not None and tensor[n,6]>=op and tensor[n,6]>0:sources.add(s)
    c.update({'hit':len(hits),'detected':len(set(g)&sources),'overlap_hit':len(overlap),'native_source_representable':len(set(src)&set(g))})
    return c


def score(raw_dir,bank_dir):
    bank=Corpus(bank_dir);data=np.load(raw_dir/'original.npz',allow_pickle=False);ids=read_json(ROOT/'stock-native-keys.json');lock=read_json(OUT/'stock-lock.json')
    receipt=read_json(raw_dir/'receipt.json')
    if receipt['original_sha256']!=digest(raw_dir/'original.npz') or receipt['source']!=lock['native_stimuli']:raise ValueError('Unexpected native predictions')
    rows={a:{r['id']:r for r in stream_rows(ROOT/(a+'.jsonl'))} for a in ('calibration','test')}
    cal=[];nums=[];sett=read_json(OUT/'stock-thresholds.json')['thresholds'];base=load_models()
    for k in ids['calibration']:
        t=data[k];r=rows['calibration'][k];pg=bank.page(k);ge=geometry(pg,t,r);cal.append((t,ge));lo,hi,_=bounds(pg.n,gold(pg,r))
        nums.append((t[pg.words[:,3],5].astype(float),lo,hi))
    def choices(i):
        z=np.concatenate([t[t[:,i]>0,i] for t,_ in cal]);return np.r_[np.quantile(z,Q),np.nextafter(z.max(),np.inf,dtype=np.float32)]
    trials=[]
    for op in choices(6):
        for res in choices(7):
            total=Counter()
            for t,ge in cal:total.update(native_count(t,ge,op,res))
            trials.append((2*total['hit']/max(1,total['gold']+total['emitted']),-total['emitted'],op,res))
    selected=max(trials)
    def loss(scale):
        return np.mean(np.concatenate([(np.maximum(lo-scale*x,0.)+np.maximum(scale*x-hi,0.))**2 for x,lo,hi in nums]))
    opt=minimize_scalar(loss,bounds=(0.,1000.),method='bounded',options={'xatol':1e-10});unit=min([0.,opt.x,1000.],key=loss)
    w=read_json(OUT/'stock-probability-weights.json')['fits']['semantic'];scale=read_json(OUT/'stock-count-scales.json')['scales']['semantic']['scale']
    counts=defaultdict(lambda:defaultdict(Counter));errors=defaultdict(dict);correls=defaultdict(lambda:defaultdict(list))
    for k in ids['test']:
        t=data[k];r=rows['test'][k];pg=bank.page(k);g=gold(pg,r);lo,hi,_=bounds(pg.n,g);key=r['source'];ge=geometry(pg,t,r)
        counts[key]['native'].update(native_count(t,ge,*selected[2:]));curves={'native':t[pg.words[:,3],5]*unit,'zero':np.zeros(pg.n)}
        for a in ('position','surface','semantic','joint'):
            e=evidence(pg,*base[a],a);tau=sett[a]['threshold'];counts[key][a].update(score_one(pg,e,tau,g));curves[a]=events(e,tau)[2]
        curves['semantic_probability']=scale*predict(pg,'semantic',w,base['semantic']).unresolved.sum(1)
        for a,curve in curves.items():
            errors[key][a]=errors[key].get(a,0.)+metrics(curve,lo,hi)
            truth=lo+hi
            if np.ptp(curve)>1e-12 and np.ptp(truth)>0:
                corr=np.corrcoef(rankdata(curve),rankdata(truth))[0,1];correls[key][a].append(float(corr))
    total=defaultdict(Counter)
    for v in counts.values():
        for a,c in v.items():total[a].update(c)
    macro={a:confidence([summarize(c[a])['f1'] for c in counts.values() if summarize(c[a])['f1'] is not None]) for a in total}
    paired=confidence([summarize(c['semantic'])['f1']-summarize(c['native'])['f1'] for c in counts.values() if all(summarize(c[a])['f1'] is not None for a in ('native','semantic'))])
    err={a:values(sum(c[a] for c in errors.values())) for a in next(iter(errors.values()))}
    cor={a:{'passages':sum(len(v[a]) for v in correls.values()),'mean':confidence([np.mean(v[a]) for v in correls.values() if v[a]])} for a in err}
    columns=('gold','hit','emitted','detected','overlap_hit','precision','recall','target_overlap_precision','f1','native_source_representable')
    out={'protocol':digest(Path(__file__).with_name('stock-native-analysis.md')),'forecast':digest(raw_dir/'original.npz'),
         'calibration_passages':len(cal),'test_passages':len(ids['test']),'test_files':len(counts),'native_span_width':receipt['anchor_tokens'],
         'psychology':{'f1':selected[0],'opening_threshold':selected[2],'resolution_threshold':selected[3],'native_mass_to_count':unit},
         'exact_source_interval_pooled':{a:{k:v for k,v in summarize(c).items() if k in columns} for a,c in total.items()},'file_macro_f1':macro,
         'file_macro_f1_gain_semantic_vs_native':paired,'calibrated_stock_errors':err,'within_document_rank_correlation':cor}
    write_json(OUT/'stock-native-scores.json',out)


if __name__=='__main__':
    p=ArgumentParser(description=__doc__);p.add_argument('raw_dir',type=Path);p.add_argument('bank_dir',type=Path);a=p.parse_args();score(a.raw_dir,a.bank_dir)
