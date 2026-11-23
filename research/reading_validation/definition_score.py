"""Score registered late-term evidence on withheld textbooks and human pre/post familiarity."""

import argparse
from collections import Counter,defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

from research.backbone_transfer.extract import digest,read_json,stream_rows,write_json
from research.reading_validation.definition_data import ROOT,OUT
from research.reading_validation.definition_features import Corpus
from research.reading_validation.definition_fit import ARMS as SEPARATE, groups

ARMS=(*SEPARATE,'joint')
from research.reading_validation.identity_causal_score import confidence
from research.reading_validation.identity_prepare import DATA


def average_precision(y,scores):
    y=np.asarray(y,dtype=bool);scores=np.asarray(scores);order=np.argsort(-scores,kind='stable')
    if not y.any():return None
    vals=scores[order];hit=y[order].cumsum();ends=np.flatnonzero(np.r_[vals[:-1]!=vals[1:],True])
    return float(np.sum(np.diff(np.r_[0,hit[ends]])*hit[ends]/(ends+1))/y.sum())


def win(x):
    return float(x>1e-8)+.5*float(abs(x)<=1e-8)


def rank(values,index):
    top=max(values);ids=np.flatnonzero(abs(np.array(values)-top)<=1e-8)
    return float(index in ids)/len(ids)


def aggregate(rows,models):
    clustered=defaultdict(list)
    for r in rows:clustered[r['source']].append(r)
    means={key:{m:np.mean([v['values'][m] for v in rec]) for m in models} for key,rec in clustered.items()}
    main='semantic' if 'semantic' in models else models[-1]
    output={'n':len(rows),'source_files':len(means),'arms':{m:confidence([v[m] for v in means.values()]) for m in models},
            'paired_primary_minus_comparator':{m:confidence([v[main]-v[m] for v in means.values()]) for m in models if m!=main}}
    if 'joint' in models and 'semantic' in models:
        output['paired_joint_minus_separate']=confidence([v['joint']-v['semantic'] for v in means.values()])
    if 'automatic_joint' in models:
        output['paired_automatic_joint_minus_separate']=confidence([v['automatic_joint']-v['automatic_semantic'] for v in means.values()])
    return output


def evaluate_page(row,page,x):
    ids=sorted(row['terms']);who={name:i for i,name in enumerate(ids)};n=page.n;outs=defaultdict(list)
    left=x['left'];right=x['right'];qidx=x['qidx'];gids={(a,b):k for k,(a,b) in enumerate(zip(left,right,strict=True))}
    positives={gids[w] for t in row['terms'].values() if (w:=page.word_span(t)) is not None}
    y=np.zeros(len(left),dtype=bool);y[list(positives)]=True
    count=Counter(source_terms=len(row['terms']),covered_terms=sum(qidx>=0))
    for name,sp,band,alternatives,other,_,_ in groups(page,row):
        if other is None:continue
        vals={};i=who[name];a,b=band;s,e=other
        for m in ARMS:
            v=x[m+'_qprob'][i];vals[m]=win(float(v[a:b].mean()-v[s:e].mean()))
        outs['same_term_definition_vs_other_interval'].append(vals)
    intervals={x['term']:[] for x in row['links']}
    for g in row['links']:
        s=page.word_interval(g['definition'])
        if s:intervals[g['term']].append(s)
    for g in row['links']:
        name=g['term'];band=page.word_interval(g['definition'])
        if not band or row['terms'][name][1]>g['definition'][0]:continue
        start,end=band;stop=end-1
        rivals=[o for o in ids if o!=name and row['terms'][o][1]<=g['definition'][0] and
                not any(s<end and start<e for s,e in intervals.get(o,[]))]
        if not rivals:continue
        count['earlier_rival_links']+=1;q=who[name];desired=qidx[q];auto={};coverage={};given={};baseline={}
        candnames=[name,*rivals];usable=[who[z] for z in candnames if qidx[who[z]]>=0]
        for m in ARMS:
            selected=x[m+'_top'][stop];inventory=x[m+'_proposals'][stop]
            auto[m]=float(desired>=0 and selected==desired)
            coverage[m]=float(desired>=0 and desired in inventory)
            if desired>=0 and len(usable)>=2:
                if q not in usable:raise ValueError('Missing expected supplied term')
                given[m]=rank(x[m+'_qprob'][usable,stop],usable.index(q))
        outs['article_only_top1'].append(auto);outs['article_only_proposal_recall_at8'].append(coverage)
        if given:outs['supplied_top1'].append(given)
    return y,count,outs


def textbook(bank,forecast_dir,lock):
    labels=list(stream_rows(ROOT/'test.jsonl'));rows=defaultdict(list);tags=defaultdict(list);counts=Counter()
    models=read_json(OUT/'weights.json')
    joint=read_json(OUT/'joint-weights.json')
    for task in ('term','link'):models[task]['joint']=joint[task]
    for r in labels:
        path=forecast_dir/(r['id']+'.npz')
        if digest(path)!=lock['predictions_sha256'][r['id']]:raise ValueError('Changed held-out predictions')
        with np.load(path,allow_pickle=False) as packed:x={k:packed[k] for k in packed.files}
        p=bank.page(r['id']);y,c,rec=evaluate_page(r,p,x);counts.update(c)
        tags[r['source']].append((y,{m:x[m+'_term'] for m in ARMS},r['subject']))
        for endpoint,values in rec.items():
            rows[endpoint].extend({'source':r['source'],'subject':r['subject'],'id':r['id'],'values':v} for v in values)
    ap=[];f1=[];flat=[];source_records={}
    for src,items in tags.items():
        y=np.concatenate([a for a,_,_ in items]);v={m:np.concatenate([b[m] for _,b,_ in items]) for m in ARMS}
        subj=items[0][2];scores={m:average_precision(y,v[m]) for m in ARMS}
        if scores['semantic'] is None:continue
        f={m:float(2*np.sum(y&(v[m]>=models['term'][m]['threshold']))/max(1,int(y.sum())+np.sum(v[m]>=models['term'][m]['threshold']))) for m in ARMS}
        ap.append({'source':src,'subject':subj,'values':scores});f1.append({'source':src,'subject':subj,'values':f})
        source_records[src]={'subject':subj,'candidate_spans':len(y),'positive_spans':int(y.sum()),'average_precision':scores,'f1':f};flat.append((y,v))
    rows['term_exact_average_precision_by_source']=ap;rows['term_exact_f1_by_source']=f1
    report={'coverage':{k:int(v) for k,v in counts.items()},'by_domain':{}}
    for subset in ('combined','biology','physics'):
        report['by_domain'][subset]={k:aggregate([r for r in rows[k] if subset=='combined' or r['subject']==subset],list(ARMS)) for k in rows}
    ys=np.concatenate([y for y,_ in flat]);report['all_covered_candidate_spans']={'n':len(ys),'positives':int(ys.sum()),
             'micro_average_precision':{m:average_precision(ys,np.concatenate([v[m] for _,v in flat])) for m in ARMS}}
    return report,source_records,rows


def guo(bank,forecast_dir,lock):
    from research.reading_validation.local_jargon import outcomes
    ext=read_json(ROOT/'external-keys.json')['jargon'];queries={r['id']:r['query_terms'] for r in stream_rows(ROOT/'probe-layout-v2.jsonl') if r['kind']=='jargon'}
    preds={};coverage=Counter();detail=[]
    for orig,k in ext.items():
        path=forecast_dir/(k+'.npz')
        if digest(path)!=lock['predictions_sha256'][k]:raise ValueError('Guo predictions changed')
        page=bank.page(k)
        with np.load(path,allow_pickle=False) as a:x={k:a[k] for k in a.files}
        for i,(term,coords) in enumerate(sorted(queries[k].items())):
            j=x['qidx'][i];sp=page.word_span(coords)
            if j<0 or sp is None:continue
            values={'remaining':float(page.n-sp[1]-1),'at_mention':float(x['semantic_term'][j])}
            for m in ARMS:
                later=x[m+'_qprob'][i,sp[1]+1:];raw=float(max(later,default=0.))
                values[m]=raw;values['automatic_'+m]=raw*bool(x[m+'_detected_query'][i])
                coverage['detected_'+m]+=int(x[m+'_detected_query'][i])
            preds[orig,term]=values;coverage['terms']+=1
    records,conflicts=outcomes(DATA/'jargon/source.csv',preds); methods=['remaining','at_mention',*ARMS,*['automatic_'+m for m in ARMS]]
    blocks=defaultdict(list)
    for r in records:
        if r['before'] is not None and r['after'] is not None:blocks[(r['paper'],r['reader'],r['before'])].append(r)
    for (paper,reader,before),values in blocks.items():
        for left,right in combinations(values,2):
            # Both have the same initial rating; a lower reverse-coded post score means greater familiarity gain.
            dy=right['after']-left['after']
            if not dy:continue
            vv={m:win(np.sign(dy)*(left['scores'][m]-right['scores'][m])) for m in methods}
            detail.append({'source':paper,'reader':reader,'baseline':before,'values':vv})
    def summarise(group):
        return {'informative_term_pairs':len(group),'papers':len({v['source'] for v in group}),'readers':len({v['reader'] for v in group}),
                'concordance_paper_clustered':aggregate(group,methods),
                'concordance_reader_clustered':aggregate([{**v,'source':v['reader']} for v in group],methods)}
    return {'coverage':{k:int(v) for k,v in coverage.items()},'conflicting_duplicates':conflicts,'available_reader_term_ratings':len(records),
            'all_same_baseline':summarise(detail),'initial_partial':summarise([r for r in detail if 1<r['baseline']<5])},detail


def main(root,forecast_dir):
    bank=Corpus(root);s=read_json(OUT/'input-seal.json');lock=read_json(OUT/'forecast-seal.json')
    if lock['joint_weights_sha256']!=digest(OUT/'joint-weights.json') or lock['weights_sha256']!=digest(OUT/'weights.json') or digest(ROOT/'test.jsonl')!=s['grades_sha256']['test']:raise ValueError('Lock mismatch')
    out,by,dec=textbook(bank,forecast_dir,lock);write_json(OUT/'textbook-scores.json',out);write_json(OUT/'source-scores.json',by)
    write_json(OUT/'term-link-item-scores.json',dict(dec))
    human,raw=guo(bank,forecast_dir,lock);write_json(OUT/'reader-scores.json',human)
    print(out['coverage'],human['all_same_baseline']['informative_term_pairs'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--forecasts',type=Path,default=ROOT/'forecasts')
    a=p.parse_args();main(a.root,a.forecasts)
