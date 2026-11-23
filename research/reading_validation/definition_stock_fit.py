"""Calibrate concept closure on psychology and score every word of held-out textbook articles."""

import argparse
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np

from research.backbone_transfer.extract import digest,read_json,stream_rows,write_json
from research.reading_validation.definition_data import OUT,ROOT
from research.reading_validation.definition_features import Corpus
from research.reading_validation.definition_stock import evidence,events
from research.reading_validation.identity_causal_score import confidence

ARMS=('position','surface','name','semantic','joint')
GRID=np.arange(1,100,dtype=int)/100


def load_models():
    m=read_json(OUT/'weights.json');j=read_json(OUT/'joint-weights.json')
    return {a:(j['term'],j['link']) if a=='joint' else (m['term']['surface' if a=='name' else a],m['link']['surface' if a=='name' else a]) for a in ARMS}


def gold(page,row):
    links=defaultdict(list)
    for entry in row['links']:
        source=tuple(row['terms'][entry['term']]);sp=page.word_interval(source);region=page.word_interval(entry['definition'])
        if sp and region and sp[1]<=region[0]: links[source].append(region)
    return {a:(page.word_interval(a)[1]-1,sorted(set(bands))) for a,bands in links.items()}


def map_spans(page,e):
    return [(int(page.words[a,0]),int(page.words[b,1])) for a,b in e.spans]


def counts_at_thresholds(e,refs,g):
    # Each trajectory is shared by all threshold candidates. The first hit is source-specific.
    n=np.zeros((len(GRID),3),dtype=int);n[:,2]=len(g)
    for j,ref in enumerate(refs):
        positions=np.flatnonzero(e.winners==j)
        if not len(positions):continue
        decisions=e.scores[positions,None]>=GRID;has=decisions.any(0);ts=positions[decisions.argmax(0)];n[:,1]+=has
        if ref in g:
            valid=np.zeros(len(GRID),dtype=bool)
            for lo,hi in g[ref][1]:valid|=(ts>=lo)&(ts<hi)
            n[:,0]+=has&valid
    return n


def calibrate(bank):
    models=load_models();rows=list(stream_rows(ROOT/'calibration.jsonl'));out={}
    for a in ARMS:
        if a=='name':out[a]={'threshold':1.,'method':'exact lexical recurrence'};continue
        counts=np.zeros((len(GRID),3),dtype=int)
        for r in rows:
            pg=bank.page(r['id']);e=evidence(pg,*models[a],a);refs=map_spans(pg,e);counts+=counts_at_thresholds(e,refs,gold(pg,r))
        tp,p,g=counts.T;f=2*tp/np.maximum(1,p+g);k=int(np.flatnonzero(f==f.max())[-1]);out[a]={'threshold':float(GRID[k]),'calibration_f1':float(f[k]),
           'true_positive':int(tp[k]),'emitted':int(p[k]),'gold':int(g[k]),'grid_counts':counts.tolist()}
    w={m:digest(OUT/('joint-weights.json' if m=='joint' else 'weights.json')) for m in ARMS}
    write_json(OUT/'stock-thresholds.json',{'protocol':digest(Path(__file__).with_name('definition-stock-protocol.md')),'revision':bank.geometry['qwen_revision'],
        'thresholds':out,'weights_sha256':w,'calibration_source':digest(ROOT/'calibration.jsonl'),'used_test':False});return out


def summarize(c):
    d={k:int(v) if isinstance(v,(int,np.integer)) else float(v) for k,v in c.items()}
    for metric,num,den in [('precision','hit','emitted'),('recall','hit','gold'),('opening_recall','detected','gold'),('closure_given_opened','hit','detected'),
            ('competition','both_ok','comp_pairs'),('competition_given_both_opened','both_ok','both_opened'),('wrong_source_on_def','wrong_source_during_definition','emitted'),
            ('distance_to_gold_count_bounds','stock_distance','n_words'),('zero_distance_to_gold_count_bounds','null_distance','n_words'),('target_overlap_precision','overlap_hit','emitted')]:
        d[metric]=float(c[num]/c[den]) if c[den] else None
    d['f1']=2*c['hit']/(c['gold']+c['emitted']) if c['gold']+c['emitted'] else None
    return d


def score_one(pg,e,tau,g):
    closed,at,stock=events(e,tau);refs=map_spans(pg,e);by={ref:i for i,ref in enumerate(refs)};cc=Counter();cc.update({'gold':len(g),'emitted':int(np.sum(at>=0)),
       'detected':sum(ref in by for ref in g),'n_words':pg.n,'opened':len(refs)});hits=set();overlap_hits=set()
    for t in np.flatnonzero(at>=0):
        ref=refs[at[t]];active=[a for a,(_,bands) in g.items() if any(lo<=t<hi for lo,hi in bands)]
        if ref in active:cc['hit']+=1;hits.add(ref)
        elif active:cc['wrong_source_during_definition']+=1
        elif ref in g:cc['wrong_time_for_true_term']+=1
        else:cc['neither_known_source_nor_time']+=1
        options=[s for s in active if ref[0]<s[1] and s[0]<ref[1] and s not in overlap_hits]
        if options:
            target=max(options,key=lambda s:min(ref[1],s[1])-max(ref[0],s[0]));overlap_hits.add(target)
    cc['overlap_hit']=len(overlap_hits)
    lower=np.zeros(pg.n);upper=np.zeros(pg.n)
    for a,(opening,regions) in g.items():
        lo,hi=regions[0];lower[opening:lo]+=1;upper[opening:hi-1]+=1
        for other,(other_open,other_regions) in g.items():
            if other==a or other_open>=lo or other_regions[0][0]<hi:continue
            cc['comp_pairs']+=1
            if a not in by or other not in by:continue
            cc['both_opened']+=1
            if lo<=closed[by[a]]<hi and (closed[by[other]]<0 or closed[by[other]]>=hi):cc['both_ok']+=1
    cc['stock_distance']=float((np.maximum(lower-stock,0.)+np.maximum(stock-upper,0.)).sum());cc['null_distance']=float(lower.sum())
    return cc


def results(cases):
    by=defaultdict(lambda:defaultdict(Counter));total=defaultdict(Counter)
    for src,rows in cases.items():
        for a,v in rows.items():by[src][a]=summarize(v);total[a].update(v)
    aggregate={a:summarize(row) for a,row in total.items()};metrics=('precision','recall','f1','opening_recall','closure_given_opened','competition','competition_given_both_opened','distance_to_gold_count_bounds','target_overlap_precision')
    means={a:{m:confidence([rows[a][m] for rows in by.values() if rows[a][m] is not None]) for m in metrics} for a in total}
    diff={a:{m:confidence([rows['semantic'][m]-rows[a][m] for rows in by.values() if rows['semantic'][m] is not None and rows[a][m] is not None]) for m in metrics} for a in total if a!='semantic'}
    return {'pooled':aggregate,'macro_source':means,'semantic_minus_comparator':diff,'source_count':len(by)}


def score(root):
    bank=Corpus(root);models=load_models();seal=read_json(OUT/'input-seal.json')
    if digest(ROOT/'calibration.jsonl')!=seal['grades_sha256']['calibration']:raise ValueError('Calibration corpus changed')
    tau=calibrate(bank);test=list(stream_rows(ROOT/'test.jsonl'))
    if digest(ROOT/'test.jsonl')!=seal['grades_sha256']['test']:raise ValueError('Test annotations changed')
    cases=defaultdict(lambda:defaultdict(Counter));subjects={}
    for r in test:
        page=bank.page(r['id']);g=gold(page,r);subjects[r['source']]=r['subject']
        for a in ARMS:
            e=evidence(page,*models[a],a);cases[r['source']][a].update(score_one(page,e,tau[a]['threshold'],g))
    subsets={s:results({k:c for k,c in cases.items() if s=='both' or subjects[k]==s}) for s in ('both','biology','physics')}
    write_json(OUT/'stock-scores.json',{'protocol':digest(Path(__file__).with_name('definition-stock-protocol.md')),'calibration':digest(OUT/'stock-thresholds.json'),
               'heldout_passages':len(test),'subjects':subsets})
    write_json(OUT/'stock-source-scores.json',{k:{a:summarize(c) for a,c in rows.items()} for k,rows in cases.items()})
    print({a:round(x['f1'],4) for a,x in subsets['both']['pooled'].items()},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);score(p.parse_args().root)
