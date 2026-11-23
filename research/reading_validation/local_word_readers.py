"""Transfer the archived Qwen lexical head to two individual-rater complexity lexicons."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import csv
import json
import re

import numpy as np

from research.reading_validation.local_features import CACHE, OUT, RESULTS
from research.reading_validation.local_forecast import digest, rows

SOURCE = CACHE / 'reading-validation/word-readers'
TARGET = OUT / 'word-readers'
FEATURES = CACHE / 'stage5/runs/cogllm-stage5-complex-20261120T172049Z-1c86fcd8/outputs/features/features.npz'
INPUT = CACHE / 'stage5/complex-l4-input'
WORD = re.compile('[a-z]+', re.ASCII)
SOURCES = {'l2':'85788879c1a234e8cdf11b9425f3e65d8e3fe43e', 'dhh':'71ef53d851a9c2ded9ec989b3308afdca0d137a5'}
SEED = 425991
DRAWS = 2000


def source_rows(population):
    with (SOURCE / (population + '.tsv')).open(newline='', encoding='utf-8-sig') as stream:
        table = csv.reader(stream, delimiter='\t')
        if population == 'dhh':
            header = next(table)
            if header[0] != 'word' or len(header) != 13:
                raise ValueError('Unexpected DHH lexicon schema')
        yield from table


def word_features(word, frequency):
    letters = word.casefold()
    freq = np.mean([np.log1p(frequency.get(term,0)) for term in letters.split()])
    return [len(word), np.log1p(len(word)), sum(c in 'aeiouy' for c in letters), freq, freq**2]


def prepare():
    root = RESULTS / 'gradio-candidate-checkpoint/lexical'
    meta = json.loads((root/'fitting.json').read_bytes())
    if digest(FEATURES) != meta['inputs']['features']['sha256'] or digest(INPUT/'design.jsonl') != meta['inputs']['design']['sha256']:
        raise ValueError('Archived Qwen lexical features or original design changed')
    if digest(root/'heads.npz') != meta['artifacts']['heads.npz']['sha256']:
        raise ValueError('The released CompLex head differs')
    design = {r['id']:r for r in rows(INPUT/'design.jsonl')}
    known = {w for r in design.values() if r['split'] in ('train','trial') for w in re.findall(r'[a-z]+',r['target_text'].casefold())}
    counts = {p:Counter(row[0].strip().casefold() for row in source_rows(p)) for p in SOURCES}
    common = set.intersection(*[{w for w,n in c.items() if n==1 and WORD.fullmatch(w)} for c in counts.values()])
    frequencies = {word:int(value) for word,value in (line.rstrip().split(' ') for line in (SOURCE/'frequency.txt').open())}
    with np.load(root/'heads.npz',allow_pickle=False) as fit:
        frac = fit['fractions'].tolist().index(100)
        meth = fit['methods'].tolist().index('target_only')
        coef, bias = fit['weights_target_only'][frac], fit['intercepts'][meth,frac]
        if not np.all(coef==coef[0]) or not np.all(bias==bias[0]):
            raise ValueError('Nominally identical full-data lexical heads disagree')
        weights, intercept = coef[0], bias[0]
    variants = defaultdict(set)
    with np.load(FEATURES,allow_pickle=False) as features:
        ids = features['ids'].tolist()
        vectors = features['target_only'].astype(np.float64)
        if set(ids)!=set(design):
            raise ValueError('Original lexical target identities differ')
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        lexical = np.clip(vectors@weights + intercept,0,1)
        for key,index in zip(ids,features['target_only_index'],strict=True):
            r=design[key]
            target=r['target_text']
            if r['kind']=='single' and target.casefold() in common:
                variants[target.casefold()].add((target,int(index)))
    training = [r for r in design.values() if r['split']=='train']
    x=np.array([word_features(r['target_text'],frequencies) for r in training])
    y=np.array([r['value'] for r in training])
    mu,sd=x.mean(0),x.std(0)
    z=(x-mu)/sd
    form=np.linalg.solve(z.T@z+np.eye(z.shape[1]),z.T@(y-y.mean()))
    eligible=[]
    for word, options in sorted(variants.items()):
        chosen=sorted(options,key=lambda t:(t[0]!=word,t[0],t[1]))[0]
        if len({i for t,i in options if t==chosen[0]})!=1:
            raise ValueError('Same native lexical prompt returned multiple feature rows')
        x=np.array(word_features(chosen[0],frequencies))
        eligible.append(dict(word=word,native=chosen[0],novel=word not in known,qwen=float(lexical[chosen[1]]),
                             surface=float(np.clip((x-mu)/sd@form+y.mean(),0,1)),frequency=float(-x[3])))
    TARGET.mkdir(exist_ok=True,parents=True)
    predicted=TARGET/'predictions.jsonl'
    predicted.write_text(''.join(json.dumps(r)+'\n' for r in eligible))
    seal=dict(protocol_sha256=digest(Path(__file__).with_name('local-word-reader-protocol.md')),code_sha256=digest(Path(__file__)),
              source_revisions=SOURCES,source_sha256={f:digest(SOURCE/f) for f in ('dhh.tsv','l2.tsv','frequency.txt')},
              features_sha256=digest(FEATURES),head_sha256=digest(root/'heads.npz'),design_sha256=digest(INPUT/'design.jsonl'),
              prediction_sha256=digest(predicted),population_forms={p:sum(c.values()) for p,c in counts.items()},
              common_unique_forms=len(common),matched=len(eligible),strict_novel=sum(r['novel'] for r in eligible))
    (TARGET/'prediction-lock.json').write_text(json.dumps(seal,indent=2)+'\n')
    print({k:seal[k] for k in ('common_unique_forms','matched','strict_novel')})


def weighted_midranks(x, counts):
    order=np.argsort(x,kind='stable')
    starts=np.r_[0,np.flatnonzero(np.diff(x[order])!=0)+1]
    tie_counts=np.add.reduceat(counts[:,order],starts,axis=1)
    ranks=np.cumsum(tie_counts,axis=1)-.5*tie_counts
    groups=np.empty(len(x),dtype=int)
    groups[order]=np.repeat(np.arange(len(starts)),np.diff(np.r_[starts,len(x)]))
    return ranks[:,groups]


def rank_correlations(y,x,weights):
    valid=np.isfinite(y) & np.isfinite(x)
    if np.unique(y[valid]).size<2 or np.unique(x[valid]).size<2:
        return np.full(len(weights),np.nan)
    counts=weights[:,valid]
    total=counts.sum(axis=1,keepdims=True)
    midpoint=total*.5
    a=weighted_midranks(x[valid],counts)-midpoint
    b=weighted_midranks(y[valid],counts)-midpoint
    num=np.einsum('bw,bw,bw->b',counts,a,b)
    denom=np.sqrt(np.einsum('bw,bw,bw->b',counts,a,a)*np.einsum('bw,bw,bw->b',counts,b,b))
    return np.divide(num,denom,out=np.full_like(num,np.nan),where=denom>0)


def report_population(y,scores):
    n,people=y.shape
    seen=np.isfinite(y)
    counts=seen.sum(axis=1,keepdims=True)-seen
    other=np.divide(np.nansum(y,axis=1,keepdims=True)-np.nan_to_num(y),counts,
                    out=np.full_like(y,np.nan),where=counts>0)
    names=list(scores)+['other_readers']
    wrng,rrng=[np.random.default_rng(s) for s in np.random.SeedSequence(SEED).spawn(2)]
    def average(w,r):
        individual=np.stack([np.column_stack([rank_correlations(y[:,j], x if name!='other_readers' else other[:,j],w)
                              for name,x in [*scores.items(),('other_readers',None)]]) for j in range(people)],axis=1)
        valid=np.isfinite(individual)
        denom=np.einsum('br,brm->bm',r,valid)
        if (denom==0).any():
            raise ValueError('No eligible participant in bootstrap sample')
        return np.einsum('br,brm->bm',r,np.nan_to_num(individual))/denom,np.nanmean(individual,axis=1)
    point=average(np.ones((1,n)),np.ones((1,people)))[0][0]
    samples,fixed=[],[]
    for start in range(0,DRAWS,25):
        k=min(25,DRAWS-start)
        w=wrng.multinomial(n,np.full(n,1/n),k)
        r=rrng.multinomial(people,np.full(people,1/people),k)
        a,b=average(w,r)
        samples.append(a)
        fixed.append(b)
    samples,fixed=np.vstack(samples),np.vstack(fixed)
    def summary(v,boot,conditional):
        return dict(mean=float(v),crossed_95=np.quantile(boot,[.025,.975]).tolist(),fixed_readers_95=np.quantile(conditional,[.025,.975]).tolist())
    results={name:summary(point[i],samples[:,i],fixed[:,i]) for i,name in enumerate(names)}
    q=names.index('qwen')
    for name in ('frequency','surface','other_readers'):
        i=names.index(name)
        results['qwen minus '+name]=summary(point[q]-point[i],samples[:,q]-samples[:,i],fixed[:,q]-fixed[:,i])
    return dict(words=n,readers=people,individual_ratings=int(seen.sum()),ratings_per_reader=seen.sum(0).tolist(),
                per_reader_words=[int(seen[:,j].sum()) for j in range(people)],results=results)


def score():
    lock=json.loads((TARGET/'prediction-lock.json').read_bytes())
    predicted=TARGET/'predictions.jsonl'
    if digest(predicted)!=lock['prediction_sha256'] or any(digest(SOURCE/f)!=s for f,s in lock['source_sha256'].items()):
        raise ValueError('Word scores or source files changed after forecast lock')
    forecasts=rows(predicted)
    ix={r['word']:i for i,r in enumerate(forecasts)}
    report={}
    for pop in SOURCES:
        observed=np.full((len(ix),11),np.nan)
        invalid=Counter()
        for row in source_rows(pop):
            word=row[0].strip().casefold()
            if word not in ix: continue
            if len(row[1:12])!=11:
                raise ValueError('Expected 11 individual rating columns')
            for j,raw in enumerate(row[1:12]):
                if raw in ('1','2','3','4','5','6'):
                    observed[ix[word],j]=int(raw)
                elif raw!='-1':
                    invalid[raw]+=1
        report[pop]={'invalid_response_values_in_matched_words':dict(invalid)}
        for name,mask in [('strict_novel',np.array([r['novel'] for r in forecasts])),('all_matched',np.ones(len(forecasts),dtype=bool))]:
            arm={m:np.array([r[m] for r in forecasts])[mask] for m in ('qwen','surface','frequency')}
            report[pop][name]=report_population(observed[mask],arm)
        print(pop,report[pop]['strict_novel']['results']['qwen minus surface'],flush=True)
    report.update(prediction_lock=digest(TARGET/'prediction-lock.json'),draws=DRAWS,seed=SEED,score_code_sha256=digest(Path(__file__)))
    (TARGET/'scores.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    args=argparse.ArgumentParser(description=__doc__)
    args.add_argument('action',choices=('forecast','score'))
    if args.parse_args().action=='forecast':
        prepare()
    else:
        score()
