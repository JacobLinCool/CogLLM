"""Prepare a frozen independent MultiRC transfer and evaluate human sentence identities."""

import argparse
from collections import Counter,defaultdict
from hashlib import sha256
from html import unescape
from itertools import combinations
import json
from pathlib import Path
import re
import zipfile

import numpy as np
from tokenizers import Tokenizer

from research.backbone_transfer.extract import digest, read_json, stream_rows, write_json
from research.reading_validation.identity_fit import ARMS, DATA, RESULT, predict
from research.reading_validation.identity_prepare import order, sealed_rows,to_tokens,TOKENIZER
from research.reading_validation.identity_score import Keys, intervals, qasper_metrics


ARCHIVE='35fc7ba1a703a0dc499c8de174b051721ce560e93775a4954493344bbf3b570e'
PROTOCOL=Path(__file__).with_name('identity-transfer-protocol.md')
OUT=Path('research/results/reading-validation-identity-transfer')
MARKER=re.compile(r'<b>\s*Sent\s+(\d+):\s*</b>',re.I)
TAGS=re.compile(r'<[^>]+>'); WORD=re.compile(r'\b\w+\b')
IGNORE=frozenset('a an the of to and or that they she he it what was in is are were for has have as about by who which how their on at do did this with'.split())


def parse_body(markup):
    parts=MARKER.split(markup)
    if parts[0].strip():
        raise ValueError('Unrecognized MultiRC display prefix')
    positions=list(map(int,parts[1::2]))
    if positions!=list(range(1,len(positions)+1)):
        raise ValueError('Expected sequential displayed sentence numbers')
    sentence=[unescape(TAGS.sub('',value)).strip() for value in parts[2::2]]
    if not sentence or any(not value for value in sentence):
        raise ValueError('Empty or unmapped original sentence')
    return sentence


def ngrams(text,n=12):
    words=WORD.findall(text.lower())
    return {tuple(words[i:i+n]) for i in range(len(words)-n+1)}


def prepare():
    OUT.mkdir(exist_ok=True,parents=True)
    source=DATA/'multirc-v2.zip'
    if digest(source)!=ARCHIVE:
        raise ValueError('MultiRC public archive changed')
    tokenizer=Tokenizer.from_file(str(TOKENIZER)); pool=[]; seen=set(); seen_ids=set(); totals=Counter()
    training={r['id'] for r in stream_rows(DATA/'identity-train.jsonl')}; previous=set()
    for row in stream_rows(DATA/'identity-stimuli.jsonl'):
        if row['id'] in training:
            previous.update(ngrams(row['text']))
    with zipfile.ZipFile(source) as archive:
        for name in sorted(archive.namelist()):
            docs=json.load(archive.open(name))['data']
            for item in sorted(docs,key=lambda i:order(i['id'])):
                totals['raw']+=1
                identity=item['id']; content=item['paragraph']; s=parse_body(content['text']); body='\n'.join(s)
                if identity in seen_ids:
                    raise ValueError('Repeated MultiRC passage identifier')
                seen_ids.add(identity)
                fingerprint=' '.join(body.split()).casefold()
                if fingerprint in seen:
                    totals['duplicate_passage']+=1;continue
                seen.add(fingerprint)
                qs=sorted([q for q in content['questions'] if q['multisent'] and len(set(q['sentences_used']))>=2],
                          key=lambda q:order(f"{identity}|{q['idx']}"))
                pair=next(((a,b) for a,b in combinations(qs,2) if
                           max(a['sentences_used']) not in b['sentences_used'] and
                           max(b['sentences_used']) not in a['sentences_used'] and
                           a['question'].casefold().strip()!=b['question'].casefold().strip()),None)
                if not pair:
                    totals['no_unambiguous_pair']+=1;continue
                key='multirc-'+sha256(identity.encode()).hexdigest()[:32]
                text=''; anchors={}; truth={}; ranges={}; questions={}
                for j,question in enumerate(pair,1):
                    label=str(question['idx']); wording=question['question'].strip()
                    text+=f'Question {j}: '; anchors[label]=[len(text),len(text)+len(wording)]
                    text+=wording+'\n'; truth[label]=max(question['sentences_used']);questions[label]=wording
                    if any(not 0<=idx<len(s) for idx in question['sentences_used']):
                        raise ValueError('Invalid zero-indexed evidence offset')
                for i,sentence in enumerate(s):
                    text+='\n';ranges[i]=[len(text),len(text)+len(sentence)];text+=sentence+'\n'
                enc=tokenizer.encode(text,add_special_tokens=False)
                if len(enc.ids)>1536:
                    totals['beyond_token_cap']+=1;continue
                anchors={k:to_tokens(enc.offsets,span) for k,span in anchors.items()}
                stops={k:to_tokens(enc.offsets,span) for k,span in ranges.items()}
                overlap=bool(ngrams(body)&previous)
                totals['any_train_12gram']+=overlap
                pool.append(({'id':key,'text':text,'anchor_tokens':anchors},
                             {'id':key,'group_id':key,'source':'multirc','source_file':name,'original_id':identity,
                              'domain':identity.split('/')[0],'risk_start':stops[0][0],
                              'window_tokens':{k:stops[i] for k,i in truth.items()},'support_sets':{str(q['idx']):q['sentences_used'] for q in pair},
                              'annotators':{k:0 for k in anchors},'overlap':overlap,'questions':questions,'sentences':s}))
    inputs=DATA/'multirc-stimuli.jsonl';outcomes=DATA/'multirc-outcomes.jsonl';pool.sort(key=lambda x:x[0]['id'])
    sealed_rows(inputs,[p[0] for p in pool]);sealed_rows(outcomes,[p[1] for p in pool]);totals['eligible']=len(pool)
    marker={'protocol_sha256':digest(PROTOCOL),'source_sha256':ARCHIVE,'inputs_sha256':digest(inputs),
            'outcomes_sha256':digest(outcomes),'coverage':dict(totals),'domains':dict(Counter(p[1]['domain'] for p in pool)),
            'model_features_include_outcomes':False}
    if (OUT/'input-seal.json').exists() and read_json(OUT/'input-seal.json')==marker:
        return
    write_json(OUT/'input-seal.json',marker)
    print(marker['coverage'])


def bm25(doc, rows):
    words=lambda text:[w for w in WORD.findall(text.lower()) if w not in IGNORE]
    sentences=[words(s) for s in rows['sentences']]; size=len(sentences); avg=np.mean([len(s) for s in sentences]);
    freq=Counter(w for s in sentences for w in set(s)); results=[]
    for target,other in ((doc[0],doc[1]),(doc[1],doc[0])):
        index=max(rows['support_sets'][target]); terms=sentences[index]; tf=Counter(terms); scores=[]
        for name in (target,other):
            value=0.
            for word in set(words(rows['questions'][name])):
                n=tf[word]
                if n:
                    value+=np.log1p((size-freq[word]+.5)/(freq[word]+.5))*n*2.2/(n+1.2*(.25+.75*len(terms)/max(avg,1.)))
            scores.append(value)
        diff=scores[0]-scores[1];results.append(.5 if abs(diff)<1e-10 else float(diff>0))
    return results[0]*results[1]


def forecast(extraction):
    marker=read_json(OUT/'input-seal.json'); weights=read_json(RESULT/'readout.json'); saved=read_json(RESULT/'forecast-seal.json')
    if digest(RESULT/'readout.json')!=saved['weights_sha256'] or read_json(extraction/'geometry.json')['input_sha256']!=marker['inputs_sha256']:
        raise ValueError('Incompatible frozen weights or source inputs')
    predictions={}
    for row in stream_rows(DATA/'multirc-stimuli.jsonl'):
        with np.load(extraction/(row['id']+'.npz'),allow_pickle=False) as archive:
            predictions[row['id']]=predict(weights['models']['official'],archive['pair'])
    out=DATA/'multirc-forecast.npz';np.savez_compressed(out,**predictions)
    write_json(OUT/'forecast-seal.json',{'source_input_sha256':marker['inputs_sha256'],'weights_sha256':saved['weights_sha256'],
                                        'predictions_sha256':digest(out),'outcomes_opened_during_inference':False})


def score():
    lock=read_json(OUT/'forecast-seal.json');source=read_json(OUT/'input-seal.json')
    if digest(DATA/'multirc-outcomes.jsonl')!=source['outcomes_sha256'] or digest(DATA/'multirc-forecast.npz')!=lock['predictions_sha256']:
        raise ValueError('Data or forecasts changed after sealing')
    with np.load(DATA/'multirc-forecast.npz',allow_pickle=False) as arr:
        predictions={k:arr[k] for k in arr.files}
    rows=list(stream_rows(DATA/'multirc-outcomes.jsonl'));keys=Keys(list(stream_rows(DATA/'multirc-stimuli.jsonl')))
    output,details=qasper_metrics(rows,keys,predictions)
    output.pop('both_questions_multiple_annotators')
    by_id={r['id']:r for r in rows}
    for row in details:
        key=row['paper'];row.pop('all_multiple');row['bm25_both']=bm25(keys[key],by_id[key]);row['domain']=by_id[key]['domain'];row['overlap']=by_id[key]['overlap']
    output['bm25_both']=intervals([r['bm25_both'] for r in details]);output['candidate_minus_bm25']=intervals([r['arms']['new_readout']['both']-r['bm25_both'] for r in details]);
    output['by_domain']={d:{'passages':len(group:=[r for r in details if r['domain']==d]),
                              'both':{a:intervals([r['arms'][a]['both'] for r in group]) for a in ARMS}}
                        for d in sorted({r['domain'] for r in details})}
    unique=[r for r in details if not r['overlap']];output['no_12word_overlap']={'n':len(unique),'both':{a:intervals([r['arms'][a]['both'] for r in unique]) for a in ARMS}}
    write_json(OUT/'scores.json',output);write_json(OUT/'passage-scores.json',details)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('action',choices=('prepare','forecast','score'))
    parser.add_argument('--features',type=Path);opts=parser.parse_args()
    if opts.action=='prepare':prepare()
    elif opts.action=='forecast':forecast(opts.features)
    else:score()
