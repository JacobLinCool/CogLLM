"""Find evidence for a previously established noun at each visible word of an article."""

from collections import defaultdict
from functools import lru_cache
import re

import numpy as np
from nltk.corpus import wordnet
from wordfreq import zipf_frequency

from research.reading_validation.memory_data import BASE,sha

WORD=re.compile(r"[^\W\d_]+(?:['’\-][^\W\d_]+)*|\d+(?:[.,]\d+)*")
FUNCTION=frozenset("a an the of at in on to from by for with into about as and or so if then when while but than he she "
                   "it they we you i me him her them us my our their his its your this these that those "
                   "is are were was be being been do does did have has had will would should can could "
                   "may might there here also very not who which what how where".split())
DETERMINERS=frozenset("the a an my your his her its our their this that these those some".split())


def setup():
    import nltk
    file=BASE/"nltk/corpora/wordnet.zip"
    if sha(file)!="cbda5ea6eef7f36a97a43d4a75f85e07fccbb4f23657d27b4ccbc93e2646ab59":
        raise ValueError("Lexical ontology differs from its release")
    nltk.data.path.insert(0,str(BASE/"nltk"))
    wordnet.ensure_loaded()


@lru_cache(None)
def senses(word):
    return tuple(wordnet.synsets(word,pos=wordnet.NOUN)[:3]) if word not in FUNCTION else ()


@lru_cache(None)
def relation(a,b):
    old,now=senses(a),senses(b)
    if not old or not now:
        return 0.
    return max(sa.wup_similarity(sb) or 0. for sa in old for sb in now)


def lexical_features(sentences,native=None):
    seen={};forms=[];lex=[];neural=[];spans=[];start=0
    if native is not None:
        coords,z=native
        lookup={(int(a),int(b)):i for i,(a,b,*_) in enumerate(coords)}
    for j,sentence in enumerate(sentences):
        text=[];a0=[];a1=[];a2=[];xs=[]
        introduced={}
        for pos,match in enumerate(WORD.finditer(sentence)):
            word=match.group().casefold();a=start+match.start();b=start+match.end()
            past=seen.get(word)
            noun=senses(word);prev=text[-1] if text else ""
            before=seen.items()
            competitors=[]
            for alternative,(sent,idx,preceded) in before:
                if alternative!=word and sent<j and senses(alternative) and noun:
                    competitors.append((relation(alternative,word),j-sent,preceded))
            related=max(competitors,key=lambda c:(c[0],-c[1]),default=(0.,0,False))
            siblings=[c for c in competitors if c[0]>.8]
            count=len(noun)
            basic=[np.log1p(len(word)),zipf_frequency(word,"en"),np.log1p(pos),match.group()[:1].isupper(),
                   word in FUNCTION,prev in DETERMINERS,np.log1p(count)]
            link=[past is not None,np.log1p(j-past[0]) if past else 0,related[0],np.log1p(related[1]),
                  related[2],np.log1p(len(siblings)),max((c[0] for c in competitors if c[2]),default=0.),
                  related[0]*(1-float(past is not None))]
            text.append(word);a0.append(basic);a1.append(link);xs.append((a,b))
            if native is not None:
                i=lookup.get((a,b))
                if i is None:
                    raise ValueError("Complete story word has no signed native geometry")
                # Initial article word has no preceding logit; encode its explicit missingness.
                a2.append(z[i,4:] if z[i,-1] else np.zeros(z.shape[1]-4))
            introduced[word]=(j,pos,prev in DETERMINERS)
        if not text:
            raise ValueError("Empty displayed sentence")
        forms.append(np.array(a0,dtype=float));lex.append(np.array(a1,dtype=float))
        neural.append(np.array(a2,dtype=float));spans.append(xs);start+=len(sentence)+2
        seen.update(introduced)
    return dict(forms=forms,lex=lex,neural=neural,spans=spans)


def source_carriers(train,documents):
    """Identify a source word by comparing released earlier premise edits, when unique."""
    families=defaultdict(dict)
    for row in train:
        families[row["story"]][row["arm"]]=row
    result={}
    for family,pair in families.items():
        a=pair["consistent"];b=pair["inconsistent"];t=a["target"]-1
        left=documents[a["id"]];right=documents[b["id"]]
        get=lambda x:{m.group().casefold() for m in WORD.finditer(x)}-FUNCTION
        differences=set().union(*(get(u)-get(v) for u,v in zip(left[:t],right[:t],strict=True) if u!=v))
        words=[w.group().casefold() for w in WORD.finditer(left[t])]
        ix=[i for i,w in enumerate(words) if w in differences]
        if len(ix)==1:
            result[family]=ix[0]
    return result
