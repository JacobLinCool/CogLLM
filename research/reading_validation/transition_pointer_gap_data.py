"""Seal the unmarked official GAP passages and assemble independent named-region labels."""
from collections import Counter
import csv
from hashlib import sha256
import json
from pathlib import Path
import re
from urllib.parse import unquote,urlparse

import numpy as np

from research.reading_validation.transition_architecture_prepare import load_tokenizer,word_indices
from research.reading_validation.transition_direct_data import PRON
from research.reading_validation.transition_game_prepare import shingles
from research.reading_validation.transition_pointer_features import sentence_ids
from research.reading_validation.transition_score import dump,sha
from research.reading_validation.transition_snac_ner import FILES
from research.reading_validation.transition_snac_prepare import ROOT,WORD


OUT=Path('research/results/reading-validation-transition-pointer-gap')
PROTOCOL=Path('research/reading_validation/transition-pointer-gap-protocol.md')
SOURCE='1c35e36d5b14f6313ec3f6cd67b275de282595dd59e59390e00cfff9897a6819'


def title(s):return ' '.join(re.findall(r'[a-z0-9]+',unquote(s).casefold()))


def prepare():
    import nltk
    from nltk.tag import PerceptronTagger
    direct=ROOT/'game-native-stimuli.jsonl';raw=ROOT/'gap-official-test.tsv'
    if sha(raw)!=SOURCE or sha(direct)!='3b711a247e6b7372ad8772aa2e877efc20f35d5ee38f3cac9186e6cf200b410a':raise ValueError('Unexpected external source or original game split')
    with raw.open(newline='') as f:official=list(csv.DictReader(f,delimiter='\t'))
    known=set();prior=set()
    for r in map(json.loads,direct.read_text().splitlines()):
        key=r['id']
        if '-wikipedia-' in key:known.add(title(key.split('-wikipedia-',1)[1]))
        prior.update(shingles(r['text']))
    material=[];exclude=Counter();excluded={};seen={}
    for r in official:
        u=title(urlparse(r['URL']).path.split('/wiki/')[-1]);key=r['ID'];text=r['Text'];t=title(text)
        reason='shared_article' if u in known else 'shared_20_words' if (prior&shingles(text)) else None
        if reason:exclude[reason]+=1;excluded[key]=reason;continue
        if t in seen:exclude['repeated_test_passage']+=1
        seen[t]=u
        material.append((key,u,text,r))
    spec=json.loads(Path('research/backbone_transfer/models.json').read_text())['models']['qwen'];tok=load_tokenizer('qwen',spec)
    posdir=ROOT/'nltk';file=next(posdir.rglob('averaged_perceptron_tagger_eng.zip'))
    if sha(file)!=FILES[file.name]:raise ValueError('POS software resources changed')
    nltk.data.path.insert(0,str(posdir));tagger=PerceptronTagger();prepared=[];labels=[];bounds=Counter()
    for key,u,text,r in material:
        enc=tok.encode(text,add_special_tokens=False);all_words=[[w.start(),w.end()] for w in WORD.finditer(text)]
        if len(enc.ids)>2048:exclude['complete_context_limit']+=1;continue
        positions=word_indices(text,all_words,tok,enc);words=[w for w,t in zip(all_words,positions,strict=True) if t];tokens=[t for t in positions if t]
        forms=[text[a:b] for a,b in words];pos=[p for _,p in tagger.tag(forms)];anchors=[i for i,f in enumerate(forms) if i and f.casefold() in PRON]
        prev=[]
        for j in anchors:
            pp=[p for _,p in tagger.tag(forms[:j+1])]
            if pp[:j-1]!=pos[:j-1]:raise ValueError('Future POS features in external article')
            prev.append(pp[j-1])
        prepared.append({'id':key,'group':key,'fold':'test','text':text,'token_count':len(enc.ids),'words':words,'tokens':tokens,
                         'sentences':sentence_ids(text,words),'pos':pos,'queries':anchors,'pos_last':prev})
        if not anchors:exclude['no_visible_pronoun_in_snippet']+=1
        q=int(r['Pronoun-offset']);end=q+len(r['Pronoun'])
        if text[q:end]!=r['Pronoun']:raise ValueError('GAP pronoun offset changed')
        matches=[i for i,(a,b) in enumerate(words) if a==q and b==end]
        if len(matches)!=1 or matches[0] not in anchors:bounds['unavailable_anchor_word']+=1;continue
        index=matches[0];regions=[]
        for name in ('A','B'):
            a=int(r[name+'-offset']);b=a+len(r[name])
            if text[a:b]!=r[name]:raise ValueError('Reference span does not match original UTF-8 source')
            before=b<=q;word=[i for i,(l,h) in enumerate(words) if a<h and l<b]
            geometry=bool(word) and words[word[0]][0]>=a and words[word[-1]][1]<=b
            if r[name+'-coref'] not in ('TRUE','FALSE'):raise ValueError('Unexpected human referent label')
            regions.append({'original':[a,b],'words':word,'before':before,'mapped':geometry,'coref':r[name+'-coref']=='TRUE'})
        if not all(z['before'] for z in regions):bounds['at_least_one_name_not_entirely_earlier']+=1;continue
        if not all(z['mapped'] and max(z['words'])<index for z in regions):bounds['unavailable_certified_name_span']+=1;continue
        if set(regions[0]['words'])&set(regions[1]['words']):bounds['overlapping_named_regions']+=1;continue
        n=sum(z['coref'] for z in regions)
        if n!=1:bounds['neither' if n==0 else 'both']+=1;continue
        labels.append({'id':key,'article':u,'q':index,'form':r['Pronoun'].casefold(),'A':regions[0]['words'],'B':regions[1]['words'],'target':0 if regions[0]['coref'] else 1})
    packet=ROOT/'gap-pointer-stimuli.jsonl';packet.write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in prepared))
    OUT.mkdir(exist_ok=True,parents=True);gold=OUT/'external-regions.json';dump(gold,{'source_sha256':SOURCE,'rows':labels})
    result={'protocol_sha256':sha(PROTOCOL),'input_sha256':sha(packet),'source_sha256':SOURCE,'gold_sha256':sha(gold),
            'upstream_test_passages':len(official),'all_passages_scored':len(prepared),'all_visible_pronouns':sum(len(s['queries']) for s in prepared),
            'eligible_examples':len(labels),'eligible_article_groups':len({s['article'] for s in labels}),
            'eligible_forms':dict(Counter(s['form'] for s in labels)),'external_tagged_data_exclusions':dict(bounds),
            'article_overlap_screen':dict(exclude),'overlap_ids':excluded,'all_scored_earlier_word_comparisons':sum(sum(s['queries']) for s in prepared)}
    dump(OUT/'source-seal.json',result);print(result)


if __name__=='__main__':prepare()
