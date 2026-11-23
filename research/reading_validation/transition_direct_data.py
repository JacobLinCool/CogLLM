"""Prepare book labels, common original word coordinates, and blind reader-game articles."""
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re
from unicodedata import normalize
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import numpy as np

from research.reading_validation.transition_architecture_prepare import CANDIDATES,load_tokenizer,word_indices
from research.reading_validation.transition_score import dump,sha
from research.reading_validation.transition_snac_fit import inputs
from research.reading_validation.transition_snac_outcomes import outcomes
from research.reading_validation.transition_snac_prepare import ROOT,WORD,source

OUT=Path('research/results/reading-validation-transition-direct')
PROTOCOL=Path('research/reading_validation/transition-direct-protocol.md')
PD=ROOT/'phrase-detectives-corpus.zip'
PDFINGER='3741448ec8bdde51b4ce1e7d61960dea166ca18a1f5ee4e3f147b92f5b7948ab'
TASKS=('CharE','RefE','CorefE')
PRON=set('he him his himself she her hers herself they them their theirs themself themselves it its itself'.split())


def gold_docs():
    if sha(PD)!=PDFINGER:raise ValueError('Unknown human antecedent source')
    with ZipFile(PD) as archive:
        for f in sorted(archive.namelist()):
            if not (f.startswith('gold/') and f.endswith('-game.xml')):continue
            form=f.replace('masxml/','CRAC/').replace('-game.xml','.CRAC')
            root=ET.fromstring(archive.read(f))
            crac=[r.split('\t') for r in archive.read(form).decode().splitlines()[1:] if r.strip()]
            wels=list(root.iter('W'))
            tokens=[''.join(w.itertext()).strip() for w in wels]
            if tokens!=[p[0] for p in crac]:raise ValueError('Expert annotation and player text disagree')
            genre=f.split('/')[-3];name=Path(f).name.removesuffix('-game.xml')
            yield 'pd-'+genre+'-'+name,genre,root,crac,wels,tokens


def display(root,wels,tokens):
    lookup={w:i for i,w in enumerate(wels)}
    sentence={lookup[w]:j for j,s in enumerate(root.iter('s')) for w in s.iter('W')}
    if len(sentence)!=len(wels):raise ValueError('Missing source sentence boundaries')
    chunks=[];coords=[];loc=0;prev=''
    for t in tokens:
        left=not chunks or t[0] in '.,?!:;%)]}’' or t in ("'s","'re","'d","'ll","'ve","n't") or prev in ('(','[','{','‘','“')
        if not left:chunks.append(' ');loc+=1
        chunks.append(t);coords.append((loc,loc+len(t)));loc+=len(t);prev=t
    text=''.join(chunks)
    ends=np.array([b for a,b in coords]);per=Counter();words=[]
    for w in WORD.finditer(text):
        token=int(np.searchsorted(ends,w.start(),side='right'));block=sentence[token]//3
        words.append([w.start(),w.end(),block,per[block]]);per[block]+=1
    return text,coords,words


def prepare():
    original,meta=inputs();groups={s['doc']:s for s in meta['documents']}
    common=ROOT/'cross-snac-stimuli.jsonl'
    if sha(common)!='3643a827d22636eba092274865003ee4739d77b86baacc5681142362f5f895ac':raise ValueError('Unknown basis of comparison')
    raw=source();out=ROOT/'snac-direct-stimuli.jsonl';stats={};ex=[];pdocs={}
    specs=json.loads(Path('research/backbone_transfer/models.json').read_text())['models']
    ts={c:load_tokenizer(c,specs[c]) for c in CANDIDATES}
    with out.open('w') as handle:
        for row in map(json.loads,common.read_text().splitlines()):
            d=row['id'];m=groups[d];row.update(split=m['split'],group=m['group'])
            if m['split']!='test':
                labels=[]
                for t in TASKS:
                    valid,y,_=outcomes(original[d],m,raw[d],t)
                    labels.append(np.where(valid,(y>0).astype(int),-1).tolist())
                row['labels']=np.array(labels,dtype=int).T.tolist()
            print(json.dumps(row,separators=(',',':')),file=handle)
        for d,genre,tree,crac,wels,tokens in gold_docs():
            text,coords,words=display(tree,wels,tokens)
            encoded={c:t.encode(text,add_special_tokens=False) for c,t in ts.items()}
            ns={c:len(v.ids) for c,v in encoded.items()}
            if max(ns.values())>2048:
                ex.append({'doc':d,'tokens':ns,'reason':'whole_document_token_limit'});continue
            maps={c:word_indices(text,[w[:2] for w in words],t,encoded[c]) for c,t in ts.items()}
            take=[i for i in range(len(words)) if all(maps[c][i] for c in CANDIDATES)]
            row={'id':d,'group':d,'split':'independent','text':text,'indices':take,'words':[words[i] for i in take],
                 'token_ranges':{c:[ar[i] for i in take] for c,ar in maps.items()},'token_counts':ns}
            print(json.dumps(row,separators=(',',':')),file=handle)
            pdocs[d]={'genre':genre,'canonical_words':len(words),'common_words':len(take),'tokens':ns,'text_sha256':sha256(text.encode()).hexdigest()}
    OUT.mkdir(parents=True,exist_ok=True)
    dump(OUT/'input-seal.json',{'protocol_sha256':sha(PROTOCOL),'input_sha256':sha(out),'original_snac_sha256':sha(common),
         'game_sha256':PDFINGER,'independent_articles':pdocs,'excluded_game_articles':ex,'tasks':TASKS})
    print(sha(out),len(pdocs),len(ex),out.stat().st_size)


if __name__=='__main__':prepare()
