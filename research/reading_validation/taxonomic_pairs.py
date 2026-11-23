"""Enumerate visible noun pairs and source-authored partial referent alignments in complete stories."""

from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
import json

import numpy as np
from wordfreq import zipf_frequency

from research.reading_validation.expectation_data import WORK
from research.reading_validation.memory_data import sha
from research.reading_validation.referent_features import DETERMINERS,WORD,senses,source_carriers
from research.reading_validation.taxonomic_norm import OUT,metrics,predictions


@dataclass
class Pairs:
    origin: np.ndarray
    earlier: np.ndarray
    word: np.ndarray
    rank: np.ndarray
    symmetric: np.ndarray
    directional: np.ndarray
    neural: np.ndarray

    def match(self,link):
        s,old,word=link
        return (self.origin==s)&np.isin(self.earlier,old)&(self.word==word)


def form(sentence):
    return [m.group().casefold() for m in WORD.finditer(sentence)]


def source_links(rows,docs):
    by=defaultdict(dict)
    for r in rows:by[r['study'],r['story']][r['arm']]=r
    carriers=source_carriers([r for r in rows if r['study']==1],docs);mapped={}
    for (_,family),pair in by.items():
        if family not in carriers:continue
        alt=pair['inconsistent'];other=pair.get('consistent',pair.get('neutral'));t=alt['target']-1
        if form(docs[alt['id']][t])!=form(docs[other['id']][t]):
            raise ValueError('The target sentence changed between a source study pair')
        changed=[i for i,(u,v) in enumerate(zip(docs[other['id']][:t],docs[alt['id']][:t],strict=True)) if u!=v]
        if not changed:continue
        s=changed[0];a=form(docs[other['id']][s]);b=form(docs[alt['id']][s]);term=carriers[family]
        if not senses(form(docs[alt['id']][t])[term]):continue
        aa=[];bb=[]
        for tag,a0,a1,b0,b1 in SequenceMatcher(a=a,b=b,autojunk=False).get_opcodes():
            if tag!='equal':
                aa.extend(k for k in range(a0,a1) if senses(a[k]));bb.extend(k for k in range(b0,b1) if senses(b[k]))
        for r,possible in ((other,aa),(alt,bb)):
            if possible:
                v=(s,possible,term);key=r['id']
                if key in mapped and mapped[key]!=v:
                    raise ValueError('Reused article has different original lexical clues')
                mapped[key]=v
    return mapped


def native_words(archive,docs):
    seal=json.loads((archive/'receipt.json').read_bytes());keys=json.loads((WORK/'opportunities.json').read_bytes())['frick']
    if sha(archive/'features.npz')!=seal['file']:raise ValueError('Changed native expectations')
    result={}
    with np.load(archive/'features.npz') as data:
        for doc,ss in docs.items():
            raw=data[keys[doc]+'.word'];z=data[keys[doc]+'.features'];coords={(a,b):v for (a,b,*_),v in zip(raw,z,strict=True)}
            loc=0;result[doc]=[]
            for sentence in ss:
                values=[]
                for m in WORD.finditer(sentence):
                    x=coords[loc+m.start(),loc+m.end()]
                    values.append(x[[4,6,8,10,11,13,15]] if x[-1] else np.zeros(7))
                result[doc].append(np.array(values));loc+=len(sentence)+2
    return result


class Ontology:
    def __init__(self):
        self.spec=json.loads((OUT/'lexical-source.json').read_bytes())['models']
        self.look=lru_cache(maxsize=80000)(self.pair)

    def pair(self,a,b):
        sx,dx=metrics(a,b);sz,dz=metrics(b,a)
        if not np.allclose(sx,sz):raise ValueError('Asymmetric symmetric WordNet features')
        s=predictions(self.spec['symmetric'],sx[None,:])[0]@np.arange(7)/6
        ab=predictions(self.spec['directional'],np.r_[sx,dx][None,:])[0]@np.arange(7)/6
        ba=predictions(self.spec['directional'],np.r_[sz,dz][None,:])[0]@np.arange(7)/6
        ancestor=max(ab,ba)
        return [sx[2],sx[3],sx[4],s], [ab,ba,ancestor,abs(ab-ba),1-ancestor,(1-ancestor)*sx[2]]


def opportunities(sentences,neural,ontology):
    earlier=[];bags=[]
    for t,s in enumerate(sentences):
        words=form(s);nouns=[(i,w) for i,w in enumerate(words) if senses(w)]
        origin=[];old=[];current=[];features=[];symmetric=[];directional=[];native=[]
        for i,b in nouns:
            for j,other,a,previous,left in earlier:
                sym,direct=ontology.look(a,b)
                features.append([np.log1p(t-j), np.log1p(i), i/max(1,len(words)-1), np.log1p(other),
                                 np.log1p(len(b)), zipf_frequency(b,'en'),zipf_frequency(a,'en'),
                                 bool(i and words[i-1] in DETERMINERS),left,a==b,*sym[:2]])
                origin.append(j);old.append(other);current.append(i);symmetric.append([a==b,*sym]);directional.append(direct)
                native.append(np.r_[previous,neural[t][i]])
        bags.append(Pairs(*[np.asarray(x,dtype=np.int16) for x in (origin,old,current)],
                          *[np.asarray(x,dtype=np.float32).reshape(-1,n) for x,n in
                            ((features,12),(symmetric,5),(directional,6),(native,14))]))
        earlier.extend((t,i,w,neural[t][i],bool(i and words[i-1] in DETERMINERS)) for i,w in nouns)
    return bags
