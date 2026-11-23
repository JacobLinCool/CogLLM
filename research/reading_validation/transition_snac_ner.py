"""Strong pretrained POS/NER baseline and evaluation restricted to person names."""
from argparse import ArgumentParser
from collections import Counter
import json
from pathlib import Path

import numpy as np

from research.reading_validation.transition_external_fit import binomial,predict
from research.reading_validation.transition_external_score import grouped_distribution,evaluate_one
from research.reading_validation.transition_score import dump,sha,with_interval
from research.reading_validation.transition_snac_fit import inputs
from research.reading_validation.transition_snac_outcomes import outcomes
from research.reading_validation.transition_snac_prepare import RESULTS,ROOT,source

FILES={'maxent_ne_chunker_tab.zip':'1370234c7770045d0c50f41e08bc627ec92450324a946de14b93cd7d5e362a86',
       'averaged_perceptron_tagger_eng.zip':'6025f530624335c67d6547d44757b357b4e79bae030a0383e9887a92c1718f0b',
       'words.zip':'54ed02917d6771dcc3e8141218960d020947f7f2ccfd9ac9b320979349746015'}
POS=('NNP','NNPS','NN','NNS','PRP','PRP$','DT','VBD','IN','JJ','CD','other')
TYPES=('PERSON','ORGANIZATION','GPE','LOCATION','FACILITY','GSP','other','O')


def taggers():
    import nltk
    from nltk.tag import PerceptronTagger
    from nltk.chunk import Maxent_NE_Chunker
    from nltk.tokenize import TreebankWordTokenizer
    base=ROOT/'nltk';base.mkdir(exist_ok=True,parents=True)
    for filename,expected in FILES.items():
        paths=list(base.rglob(filename))
        if not paths:
            nltk.download(filename.removesuffix('.zip'),download_dir=str(base),raise_on_error=True,quiet=True)
            paths=list(base.rglob(filename))
        if len(paths)!=1 or sha(paths[0])!=expected:raise ValueError('Unknown English NLTK model resource')
    nltk.data.path.insert(0,str(base));tagger=PerceptronTagger();ner=Maxent_NE_Chunker();tokenizer=TreebankWordTokenizer()
    return tagger,ner,tokenizer


def annotate_document(doc,segments,tagger,ner,tokenizer):
    from nltk.chunk import tree2conlltags
    words=np.asarray(doc['words']);x=np.zeros((len(words),len(POS)+len(TYPES)+4));seen={}
    for seg in segments:
        basechar=seg['start'];s=doc['text'][basechar:seg['end']];spans=list(tokenizer.span_tokenize(s));tokens=[s[a:b] for a,b in spans]
        tagged=tree2conlltags(ner.parse(tagger.tag(tokens)));bio=[t[2] for t in tagged];types={};k=0
        while k<len(tagged):
            typ=bio[k][2:] if bio[k]!='O' else 'O';stop=k+1
            if typ!='O':
                while stop<len(tagged) and bio[stop]=='I-'+typ:stop+=1
            name=' '.join(tokens[k:stop]).casefold();absolute=basechar+spans[k][0];previous=seen.get(name)
            for q in range(k,stop):
                types[q]=(typ,typ=='PERSON' and previous is None,typ=='PERSON' and previous is not None,
                          np.log1p(absolute-previous) if typ=='PERSON' and previous is not None else 0.,stop-k if typ=='PERSON' else 0)
            if typ=='PERSON':seen[name]=absolute
            k=stop
        bounds=np.array(spans)+basechar
        for row in np.flatnonzero(words[:,4]==seg['index']):
            a,b=words[row,:2];hit=np.flatnonzero((bounds[:,0]<b)&(bounds[:,1]>a))
            if not hit.size:raise ValueError('Pretrained tagger did not cover a certified word')
            best=next((q for q in hit if types[q][0]=='PERSON'),hit[0]);pos=tagged[best][1];typ,*controls=types[best]
            x[row,POS.index(pos if pos in POS else 'other')]=1
            x[row,len(POS)+TYPES.index(typ if typ in TYPES else 'other')]=1;x[row,-4:]=controls
    return x


def annotate():
    tagger,ner,tokenizer=taggers()
    docs,meta=inputs();features={}
    for material in meta["documents"]:
        name=material["doc"]
        features[name]=annotate_document(docs[name],material["segments"],tagger,ner,tokenizer)
    np.savez_compressed(ROOT/'snac-ner.npz',**features)


def fit(directory):
    docs,meta=inputs();raw=source();ner=np.load(ROOT/'snac-ner.npz');gpu=np.load(directory/'word-features.npz');x={};y={};
    if sha(directory/'word-features.npz')!=json.loads((directory/'receipt.json').read_text())['feature_sha256']:raise ValueError('SNaC features changed')
    for split in ('train','dev'):
        xx=[];yy=[]
        for m in meta['documents']:
            if m['split']!=split:continue
            d=m['doc'];ok,v,_=outcomes(docs[d],m,raw[d],'CharE');xx.append(np.column_stack((gpu[d][:,:17],ner[d]))[ok]);yy.append((v[ok]>0).astype(int))
        x[split]=np.concatenate(xx);y[split]=np.concatenate(yy)
    model=binomial(x['train'],y['train'],np.ones(len(y['train']),dtype=int));dev=predict(x['dev'],model)
    cutoff=float(np.quantile(dev[y['dev']==0],.95,method='higher'));forecasts={}
    for m in meta['documents']:
        if m['split']!='test':continue
        d=m['doc'];tag=ner[d];p=predict(np.column_stack((gpu[d][:,:17],tag)),model)
        forecasts[d]=np.column_stack((p,tag[:,len(POS)+TYPES.index('PERSON')],tag[:,-4]))
    path=RESULTS/'entity-control-probabilities.npz';np.savez_compressed(path,**forecasts)
    dump(RESULTS/'entity-control-lock.json',{'model':model,'cutoff':cutoff,'forecast_sha256':sha(path),
        'local_ner_sha256':sha(ROOT/'snac-ner.npz'),'nltk_resources':FILES,'protocol_sha256':sha(Path('research/reading_validation/transition-snac-entity-control.md'))})


def score():
    docs,meta=inputs();raw=source();path=RESULTS/'entity-control-probabilities.npz';lock=json.loads((RESULTS/'entity-control-lock.json').read_text());p0=RESULTS/'heldout-probabilities.npz'
    if sha(path)!=lock['forecast_sha256'] or sha(p0)!=json.loads((RESULTS/'prediction-lock.json').read_text())['forecast_sha256']:raise ValueError('Movie probabilities changed')
    ner=np.load(path);original=np.load(p0);groups=[];labels=[];features=[]
    for m in meta['documents']:
        if m['split']!='test':continue
        d=m['doc'];ok,v,_=outcomes(docs[d],m,raw[d],'CharE');groups.extend([m['group']]*int(ok.sum()));labels.extend(v[ok]>0)
        features.append(np.column_stack((ner[d],original['CharE__'+d][:,3]))[ok])
    X=np.concatenate(features);y=np.array(labels,dtype=int);n=np.ones(len(y),dtype=int);groups=np.array(groups)
    threshold=json.loads((RESULTS/'readouts.json').read_text())['readouts']['CharE']['antecedents']['threshold'];out={}
    for region,mask in [('all',n.astype(bool)),('recognized_person',X[:,1]>0),('first_person_mention',X[:,2]>0)]:
        group,idx,draws=grouped_distribution(groups[mask]);methods={};boot={}
        for arm,col,cut in [('ner_pos_surprisal',0,lock['cutoff']),('contextual_model',3,threshold)]:
            methods[arm],boot[arm]=evaluate_one(y[mask],n[mask],X[mask,col],cut,idx,draws)
        delta=methods['contextual_model']['ap']['estimate']-methods['ner_pos_surprisal']['ap']['estimate']
        out[region]={'movies':len(group),'words':int(mask.sum()),'positive':int(y[mask].sum()),'models':methods,
            'contextual_minus_ner_ap':with_interval(delta,boot['contextual_model']['ap']-boot['ner_pos_surprisal']['ap'])}
    dump(RESULTS/'entity-control-scores.json',out)
    for key,val in out.items(): print(key,val['words'],val['positive'],val['models']['ner_pos_surprisal']['ap'],val['contextual_minus_ner_ap'])


if __name__=='__main__':
    parser=ArgumentParser();parser.add_argument('phase',choices=['annotate','fit','score']);parser.add_argument('--features',type=Path);args=parser.parse_args()
    if args.phase=='annotate':annotate()
    elif args.phase=='fit':fit(args.features)
    else:score()
