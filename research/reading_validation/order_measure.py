"""Align authored changes in information order with the original reading and comprehension responses."""

from collections import Counter,defaultdict
import json
from math import log
import re

import numpy as np

from research.reading_validation.memory_data import h,load_rows,sha
from research.reading_validation.order_data import OUT,WORK,table

TASKS=('sentence_total','clause_first_pass','clause_total','question')
ARMS=('constant','form','surprisal','context','profile','design')
FUNCTION=set('a the an is was were had were he she it they their her his its and of in to at into on or before after for by i we some with'.split())


class Study:
    def __init__(self):
        meta=json.loads((OUT/'inputs.json').read_bytes())
        if any(sha(WORK/k)!=v for k,v in meta['original'].items()) or sha(WORK/'stimuli.jsonl')!=meta['text']:
            raise ValueError('The original clause experiments or unmarked article text changed')
        self.meta=meta;self.conditions={(g,r['condition']):r for g,arms in meta['eligible'].items() for r in arms}
        self.texts={r['id']:r['text'] for r in load_rows(WORK/'stimuli.jsonl')}
        self.observations=defaultdict(list);self.readers={};counts=Counter()
        for file in ('full.tsv','regions.tsv'):
            for row in table(file):
                group,condition=row['itemid'],row['condition'];item=group,condition
                if item not in self.conditions:continue
                person=h(row['participant']);v=(float(row['TOWRE_scaled']),float(row['WM_span_credit']))
                if self.readers.setdefault(person,v)!=v:
                    raise ValueError('The same participant has different standardized reading or memory scores')
                if file=='full.tsv':
                    values={'sentence_total':row['totfixdur']}
                else:
                    values={'question':row['QUESTION_ACCURACY']}
                    if row['outlier']=='0':values.update(clause_first_pass=row['fpdur'],clause_total=row['totfixdur'])
                for task,value in values.items():
                    if value in ('NA','.',''):continue
                    z=float(value)
                    if task=='question':
                        if z==-1:continue
                        if z not in (0,1):raise ValueError('Unrecognized question response')
                    else:
                        if z<=0:continue
                        z=log(z)
                    self.observations[task].append((group,condition,person,z));counts[task]+=1
        if len(self.readers)!=75:raise ValueError('Original full participant set not recovered')

    def panel(self,task,panel):
        def keep(row):
            g,c,p,_=row;m=self.meta['material_split'][g];who=self.meta['reader_split'][p]
            if panel=='train':return m=='train' and who=='train'
            if panel=='held_both':return m=='test' and who=='test'
            if panel=='held_material':return m=='test'
            raise ValueError(f'Unknown study panel {panel}')
        return list(filter(keep,self.observations[task]))


def lexical(text):
    return [w for w in re.findall(r"[^\W\d_]+",text.casefold()) if w not in FUNCTION]


def aggregate(z,indices):
    if not len(z):raise ValueError('A measured reading unit has no certified original words')
    v=z[:,indices]
    return np.r_[v.mean(0),v.max(0),v[0],v[-1]]


def features(archive,study):
    receipt=json.loads((archive/'receipt.json').read_bytes())
    if sha(archive/'features.npz')!=receipt['file'] or receipt['source']!=study.meta['text']:
        raise ValueError('Single-pass clause-order archive does not match the original texts')
    design={}
    with np.load(archive/'features.npz') as f:
        for item,r in study.conditions.items():
            text=study.texts[r['key']];coords=f[r['key']+'.word'];z=f[r['key']+'.features'].astype(float)
            a,e=r['target'];b,_=r['second'];u,v=r['connective'];right=coords[:,1];left=coords[:,0]
            eligible=(coords[:,-1]==1);cl1=(left>=a)&(right<=b)&eligible;cl2=(left>=b)&(right<=e)&eligible
            old=set(lexical(text[:a]));first=lexical(text[a:b]);second=lexical(text[b:e])
            def overlap(tokens):return sum(t in old for t in tokens)/max(1,len(tokens))
            main=r['structure']=='m-s';given=r['given']=='given-new';chrono=r['order']=='e1-e2'
            oracle=np.array([main,given,chrono,main*given,main*chrono,given*chrono,main*given*chrono],dtype=float)
            form=np.array([np.log1p(a),np.log1p(e-a),np.log1p(b-a),np.log1p(e-b),cl1.sum(),cl2.sum(),
                       text[u:v].casefold()=='before',u==a,overlap(first),overlap(second),
                       np.mean(z[cl1,3]),np.mean(z[cl2,3]),len(set(first)&set(second))/max(len(set(second)),1)],dtype=float)
            for task in TASKS:
                sections=[cl1,cl2]
                extra=np.empty(0)
                if task=='question':
                    post=(left>=e)&eligible
                    sections.append(post);extra=np.r_[np.log1p(len(text)-e),overlap(lexical(text[e:]))]
                base=np.r_[form,extra];surp=np.r_[base,*[aggregate(z[s],(4,)) for s in sections]]
                context=np.r_[surp,*[aggregate(z[s],(6,8,9,10,11,12,13)) for s in sections]]
                design[task,item]=dict(constant=np.empty(0),form=base,surprisal=surp,context=context,profile=context,design=oracle)
    return design
