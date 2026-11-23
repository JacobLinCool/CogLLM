"""Lock predictive scores on Guo's published terms, then join anonymous human responses."""

import argparse
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
import csv
import json

import numpy as np

from research.reading_validation.local_features import OUT, CACHE, read_gpu, row_grams, sentence_features, source_lexicon, verified_rows
from research.reading_validation.local_forecast import WORD, digest, rows
from research.reading_validation.local_jargon_prepare import JARGON, sentence_bounds
from research.reading_validation.local_rank import SEED


def find_overlap(papers, segments):
    references = {'plaba': [r['text'] for r in verified_rows('plaba','stimuli.jsonl')]}
    with (CACHE/'medreadme-audit/raw/readability.csv').open(newline='') as stream:
        references['medreadme']=[r['Sentence'] for r in csv.DictReader(stream) if r['split']=='train']
    references['complex']=[r['text'] for r in rows(CACHE/'stage5/complex-l4-input/stimuli.jsonl') if r['split'] in ('train','trial')]
    excluded=defaultdict(set)
    forms=lambda s:tuple(w.group().casefold() for w in WORD.finditer(s))
    for study,texts in references.items():
        index=defaultdict(set)
        known=set()
        for i,text in enumerate(texts):
            sequence=forms(text)
            known.add(sequence)
            for gram in row_grams(sequence):
                index[gram].add(i)
        for paper,text in papers.items():
            target=[text] if study=='plaba' else [text[a:b] for a,b in segments[paper]]
            for sample in target:
                sequence=forms(sample)
                hits=Counter(i for gram in row_grams(sequence) for i in index.get(gram,()))
                if sequence in known or (hits and max(hits.values())*4>=len(row_grams(sequence))):
                    excluded[paper].add(study)
                    break
    return {k:sorted(v) for k,v in excluded.items()}


def restore_predictor(data, node):
    vals=[(node.features[k]-np.asarray(data['center'][k]))/np.asarray(data['scale'][k]) for k in data['features']]
    return np.column_stack(vals)@np.asarray(data['coef'])


def forecast(gpu, fitted, out):
    source_seal=OUT/'jargon/source-seal.json'
    source=json.loads(source_seal.read_bytes())
    xs=JARGON/'local-stimuli.jsonl'
    gs=JARGON/'geometry.jsonl'
    if digest(xs)!=source['stimuli_sha256'] or digest(gs)!=source['geometry_sha256']:
        raise ValueError('Grade-free new corpus differs from source seal')
    cache=read_gpu(gpu,xs,source_seal)
    papers={r['id']:r['text'] for r in rows(xs)}
    geometry={r['id']:r for r in rows(gs)}
    exclusions=find_overlap(papers,{k:r['segments'] for k,r in geometry.items()})
    lex,lex_audit=source_lexicon()
    trained=json.loads(fitted.read_text())
    if set(trained)!= {'form','medical_form','local_lexical','local','local_shared','local_and_persistent'}:
        raise ValueError('All prespecified PLABA heads are required before opening Guo annotations')
    forecasts=[]
    for paper,text in papers.items():
        if paper in exclusions:
            continue
        geo=geometry[paper]
        if [tuple(ab) for ab in geo['segments']]!=sentence_bounds(text):
            raise ValueError('Unstable sentence segmentation')
        for i,(a,b) in enumerate(geo['segments']):
            terms=[t for t in geo['terms'] if t['sentence']==i]
            if not terms:
                continue
            node=sentence_features(f'{paper}@{a}',paper,text,a,b,lex,cache[paper],i,0.,sentence_endpoint=False)
            by_coord={(left,right):j for j,(left,right) in enumerate(node.coords)}
            signals={name:restore_predictor(model,node) for name,model in trained.items()}
            signals.update(node.raw)
            for term in terms:
                j=by_coord[term['start'],term['end']]
                forecasts.append(dict(paper=paper,term=term['term'],start=term['start'],end=term['end'],
                                     scores={m:float(z[j]) for m,z in signals.items()}))
    out.mkdir(exist_ok=True,parents=True)
    path=out/'predictions.jsonl'
    with path.open('w') as stream:
        for row in forecasts:
            stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    seal=dict(hosted_study='Guo et al. NAACL 2024; scored supplied terms in unseen study',
              source_seal_sha256=digest(source_seal), protocol_sha256=source['protocol_sha256'],
              probe_weights_sha256=digest(fitted), gpu_sha256=digest(gpu/'seal.json'), code_sha256=digest(Path(__file__)),
              excluded_papers=exclusions,medreadme_fit=lex_audit,
              papers=len({r['paper'] for r in forecasts}),scored_terms=len(forecasts),files_sha256={path.name:digest(path)})
    (out/'prediction-lock.json').write_text(json.dumps(seal,indent=2)+'\n')
    print(json.dumps({k:seal[k] for k in ('papers','scored_terms','excluded_papers')},indent=2))


def two_way(cube, mask, draws=10000):
    articles, readers, n_methods=cube.shape
    missing=(mask.sum(axis=1)==0)
    cub=cube[~missing]
    val=mask[~missing].astype(float)
    articles=len(cub)
    point=(cub.sum(axis=1)/val.sum(axis=1,keepdims=True)).mean(axis=0)
    arng,rrng=[np.random.default_rng(s) for s in np.random.SeedSequence(SEED).spawn(2)]
    crossed,conditional=[],[]
    for start in range(0,draws,100):
        batch=min(draws-start,100)
        aw=arng.multinomial(articles,np.ones(articles)/articles,batch)
        rw=rrng.multinomial(readers,np.ones(readers)/readers,batch)
        denom=rw@val.T
        # A resampled reader pool can contain no raters with unequal responses on a given abstract.
        aw_cross=aw*(denom>0)
        if (aw_cross.sum(axis=1)==0).any():
            raise ValueError('No eligible resampled abstract')
        nums=np.einsum('br,arm->bam',rw,cub)
        m=nums/np.maximum(denom[:,:,None],1)
        crossed.append(np.einsum('ba,bam->bm',aw_cross,m)/aw_cross.sum(axis=1,keepdims=True))
        conditional.append(aw@(cub.sum(axis=1)/val.sum(axis=1,keepdims=True))/articles)
    return point,np.vstack(crossed),np.vstack(conditional)


def rank_stats(records, field, base_methods, use_prior=False):
    methods=base_methods+(['prereading_rating'] if use_prior else [])
    papers=sorted({r['paper'] for r in records})
    people=sorted({r['reader'] for r in records})
    cube=np.zeros((len(papers),len(people),len(methods)))
    valid=np.zeros((len(papers),len(people)),dtype=bool)
    raw=defaultdict(list)
    for r in records:
        if r[field] is not None and (not use_prior or r['before'] is not None):
            raw[r['paper'],r['reader']].append(r)
    n_pairs=0
    for (paper,reader),group in raw.items():
        observations=[]
        for a,b in combinations(group,2):
            delta=a[field]-b[field]
            if delta==0:
                continue
            diff=np.array([a['scores'][k]-b['scores'][k] for k in base_methods])
            if use_prior:
                diff=np.r_[diff,a['before']-b['before']]
            observations.append(.5*(np.sign(diff)*np.sign(delta)+1))
        if observations:
            i,j=papers.index(paper),people.index(reader)
            cube[i,j]=np.mean(observations,axis=0)
            valid[i,j]=True
            n_pairs+=len(observations)
    point,draws,fixed=two_way(cube,valid)
    report={}
    def report_item(p,b,f):
        return dict(mean=float(p),crossed_95=np.quantile(b,[.025,.975]).tolist(),
                    conditional_on_readers_95=np.quantile(f,[.025,.975]).tolist())
    for i,m in enumerate(methods):
        report[m]=report_item(point[i],draws[:,i],fixed[:,i])
    for m,b in [('local','form'),('local','displayed_mass'),('local','medical_form'),
                ('local','external_lexical'),('local_and_persistent','local'),
                ('external_lexical','form'),('prereading_rating','local')]:
        if m not in methods or b not in methods:
            continue
        i,j=methods.index(m),methods.index(b)
        report[m+' minus '+b]=report_item(point[i]-point[j],draws[:,i]-draws[:,j],fixed[:,i]-fixed[:,j])
    return dict(papers=int(valid.any(axis=1).sum()),readers=int(valid.any(axis=0).sum()),reader_paper_cells=int(valid.sum()),
                unequal_pairs=n_pairs,estimates=report)


def outcomes(source, preds):
    expected=json.loads((OUT/'jargon/source-seal.json').read_bytes())
    if digest(source)!=expected['source_sha256']:
        raise ValueError('Guo source changed')
    def familiarity(value):
        if value in ('','nan','NA'):
            return None
        f=float(value)
        if f not in (0,1,2,3,4):
            raise ValueError(f'Unrecognized familiarity response {value!r}')
        # The distributed CSV stores the five ordered survey ratings as integer indexes 0..4.
        return 5.-f
    def binary(value):
        if value in ('','nan','NA'):
            return None
        if value.lower() in ('yes','true','1','1.0'):
            return 1.
        if value.lower() in ('no','false','0','0.0'):
            return 0.
        raise ValueError('Unrecognized information-need coding')
    matched=defaultdict(set)
    with source.open(newline='') as stream:
        for r in csv.DictReader(stream):
            key='jargon/'+r['paper_sha'],r['entity']
            if key not in preds:
                continue
            matched[key,r['annotator']].add((familiarity(r['familiarity']),familiarity(r['familiarity_before']),
                                              binary(r['additional_definition']),binary(r['additional_background'])))
    records=[]
    for ((paper,term),reader),values in matched.items():
        if len(values)!=1:
            continue
        post,pre,definition,background=next(iter(values))
        records.append(dict(paper=paper,reader=reader,term=term,after=post,before=pre,definition=definition,
                            background=background,scores=preds[paper,term]))
    return records,sum(len(v)!=1 for v in matched.values())


def score(out):
    lock=json.loads((out/'prediction-lock.json').read_bytes())
    path=out/'predictions.jsonl'
    if digest(path)!=lock['files_sha256'][path.name]:
        raise ValueError('Forecasts changed after sealing')
    preds={(r['paper'],r['term']):r['scores'] for r in rows(path)}
    records,conflicts=outcomes(JARGON/'source.csv',preds)
    methods=list(next(iter(preds.values())))
    report={k:rank_stats(records,k,methods) for k in ('after','before','definition','background')}
    report['after_with_prereading_comparator']=rank_stats(records,'after',methods,True)
    valid=[r for r in records if r['after'] is not None and r['before'] is not None]
    changes=np.array([r['before']-r['after'] for r in valid])
    grouped=defaultdict(list)
    for r in records:
        if r['after'] is not None:
            grouped[r['paper'],r['term']].append(r['after'])
    group_mean=[dict(paper=paper,reader='group',term=term,after=float(np.mean(vals)),scores=preds[paper,term])
                for (paper,term),vals in grouped.items()]
    report['mean_reader_after']=rank_stats(group_mean,'after',methods)
    report.update(eligible_reader_terms=len(records),distinct_eligible_terms=len(grouped),excluded_conflicting_duplicates=conflicts,
                  prediction_lock=digest(out/'prediction-lock.json'),score_code_sha256=digest(Path(__file__)),
                  familiarity_encoding='The author CSV indexes the 1–5 questionnaire choices as 0–4. Difficulty = 5 − stored index.',
                  primary='after, local minus form',
                  before_after=dict(paired=len(changes),same=int(np.sum(changes==0)),increased_familiarity=int(np.sum(changes>0)),
                                    decreased_familiarity=int(np.sum(changes<0)),mean_change=float(changes.mean())))
    (out/'scores.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k]['estimates']['local minus form'] for k in ('after','before','definition','background')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('forecast','score'))
    p.add_argument('--gpu',type=Path)
    p.add_argument('--fitted',type=Path,default=OUT/'localization/deployment-weights.json')
    p.add_argument('--out',type=Path,default=OUT/'jargon')
    a=p.parse_args()
    if a.action=='forecast':
        forecast(a.gpu,a.fitted,a.out)
    else:
        score(a.out)
