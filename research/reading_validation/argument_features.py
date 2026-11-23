"""Causal sentence features for a writer's evidence, candidate claims and argument strength."""
from collections import Counter
from functools import lru_cache
import json
import math
import re

import numpy as np

from research.reading_validation.argument_data import OUT,WORK,sentences
from research.reading_validation.memory_data import load_rows,sha

LEX=re.compile(r"[a-z]+(?:'[a-z]+)?|\d+",re.I)
STOP=set("a an the to be for by as of from at in as is are this those these their they that what who how it its was were have be into on and has had or with can will would could should".split())
MARKERS=(
 r'\b(?:i (?:think|believe|agree)|in my (?:opinion|view)|my position)\b',
 r'\b(?:because|since|as a result|due to|reason)\b',
 r'\b(?:for example|for instance|according to|evidence|study|studies|research)\b',
 r'\b(?:however|although|whereas|on the other hand|despite|yet|but)\b',
 r'\b(?:therefore|in conclusion|in summary|thus|finally|overall)\b',
 r'\b(?:first|second|third|furthermore|moreover|also|in addition)\b',
 r'\b(?:should|must|may|can|could|would)\b',r'\b(?:i|my|me|we|our)\b',
 r'\b(?:school|teacher|student|people)\b',
)
PATTERNS=tuple(map(re.compile,MARKERS))


def unit_rows(a):
    return a/np.maximum(np.linalg.norm(a,axis=1,keepdims=True),1e-9)


class Corpus:
    def __init__(self,archive=None):
        self.meta=json.loads((OUT/'inputs.json').read_bytes())
        self.text={r['id']:r['text'] for r in load_rows(WORK/'stimuli.jsonl')}
        self.native=None;self.info=None
        if archive is not None:
            receipt=json.loads((archive/'receipt.json').read_bytes())
            if receipt['source']!=self.meta['stimuli'] or receipt['file']!=sha(archive/'features.npz'):
                raise ValueError('Changed unmarked native essay bank')
            self.native=np.load(archive/'features.npz',allow_pickle=False);self.info=receipt['articles']
        docs=Counter();n=0
        for k,r in self.meta['articles'].items():
            if r['role']=='train' and (self.info is None or k in self.info):
                n+=1;docs.update(set(LEX.findall(self.text[k].casefold()))-STOP)
        self.idf={t:math.log1p((n-f+.5)/(f+.5)) for t,f in docs.items()};self.unseen=math.log1p((n+.5)/.5)
        self.design=lru_cache(maxsize=32)(self.features)

    def eligible(self,k):return self.info is None or k in self.info

    def features(self,k):
        t=self.text[k];coords=sentences(t)
        if self.info is not None and [list(s) for s in coords]!=self.info[k]['sentences']:
            raise ValueError('Grading text and native character coordinates disagree')
        paragraphs=[];j=-1;last=0;words=[];norm=[];surface=[];sets=[];ends=[]
        for a,b in coords:
            if not words or '\n' in t[last:a] or '\r' in t[last:a]:j+=1
            paragraphs.append(j);last=b
            part=t[a:b].casefold();tokens=LEX.findall(part);ct=Counter(w for w in tokens if w not in STOP)
            words.append(ct);v={w:(1+math.log(c))*self.idf.get(w,self.unseen) for w,c in ct.items()}
            norm.append(math.sqrt(sum(z*z for z in v.values())));sets.append(v)
            indicators=[float(p.search(part) is not None) for p in PATTERNS]
            surface.append(np.r_[np.log1p(len(tokens)),np.log1p(len(part)),len(ct)/max(1,len(tokens)),np.mean([self.idf.get(w,self.unseen) for w in ct]) if ct else 0,
                                part.count(','),part.count('"'),part.count('?'),sum(bool(re.match(r'\d',w)) for w in tokens),indicators])
            ends.append(float(len(paragraphs)==1 or paragraphs[-2]!=j))
        surface=np.array(surface);n=len(coords)
        if self.native is not None:
            rep=self.native[k+'.rep'].astype(float);att=self.native[k+'.att'].astype(float)
            if rep.shape!=(n,480) or att.shape!=(n,n,2):raise ValueError('Unexpected released sentence feature layout')
            units=[unit_rows(z) for z in np.split(rep,5,axis=1)]
        pairs=[];positions=[]
        for now in range(n):
            earlier=np.arange(now);target=sets[now]
            cosine=[];jaccard=[];informativeness=[]
            for i in earlier:
                prior=sets[i];overlap=target.keys()&prior.keys()
                dot=sum(target[w]*prior[w] for w in overlap)
                cosine.append(dot/max(1e-9,norm[now]*norm[i]))
                jaccard.append(len(overlap)/max(1,len(target.keys()|prior.keys())))
                informativeness.append(max((self.idf.get(w,self.unseen) for w in overlap),default=0.))
            sim=np.array(cosine);preceding=np.array(paragraphs[:now]);p=paragraphs[now]
            local=np.r_[np.log1p(now),np.log1p(p),ends[now],surface[now],sim[-1] if now else 0,sim.max(initial=0)]
            if now:
                form=np.column_stack((np.log1p(now-earlier),1/(now-earlier),earlier==now-1,
                       np.log1p(earlier),np.log1p(p-preceding),p==preceding,np.array(ends[:now]),
                       sim,jaccard,informativeness,surface[:now],np.log1p(now-earlier)*sim))
                form=np.column_stack((form,form[:,2,None]*surface[now],(p==preceding)[:,None]*surface[now]))
            else:form=np.empty((0,0))
            pos={'form':local};pair={'form':form}
            if self.native is not None:
                cos=[v[:now]@v[now] for v in units]
                if now:
                    pair['native']=np.column_stack((form,att[now,:now],*cos,
                        units[0][:now]*units[0][now],units[1][:now]*units[1][now],
                        units[4][:now]*units[4][now]))
                    close=np.r_[[v.max() for v in cos],[v[-1] for v in cos],att[now,:now].max(0),att[now,now-1]]
                else:
                    pair['native']=np.empty((0,0));close=np.zeros(14)
                pos['native']=np.r_[local,*[v[now] for v in units],close]
            positions.append(pos);pairs.append(pair)
        return positions,pairs
