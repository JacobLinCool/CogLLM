"""Represent what a person saw before making one whole-argument quality judgment."""
from collections import Counter,defaultdict
import json
from pathlib import Path
import re
import numpy as np
from research.reading_validation.argument_data import sentences,chunks
from research.reading_validation.memory_data import h,load_rows,sha
from research.reading_validation.reason_quality_data import DATA,OUT
from research.reading_validation.state_words import STOP,WORD,sketch

MARKERS=[r'\b(because|since|as a result)\b',r'\b(therefore|so|thus|hence|consequently)\b',r'\b(however|although|but|despite|whereas)\b',
         r'\b(example|instance|illustrate|evidence|experiment|study|research|survey)\b',r'\b(first|second|finally|furthermore|moreover)\b',
         r'\b(every|all|always|never|none|only)\b',r'\b(some|may|might|perhaps|often|sometimes)\b',r'\b(i|me|my|believe|think)\b',
         r'\b(if|unless|provided)\b',r'\b(that|this|these|those|it|they)\b']
MARKERS=list(map(re.compile,MARKERS))

def partition(records):
    panel={}
    for source,n_dev,n_held,group in [('webis',3,4,'topic'),('ukp',.15,.20,'family')]:
        rr=records[source];groups=sorted({r[group] for r in rr},key=lambda k:h('reason-scope-20261122-'+str(k)))
        a=int(np.ceil(len(groups)*n_held)) if n_held<1 else n_held;b=int(np.ceil(len(groups)*n_dev)) if n_dev<1 else n_dev
        slots={g:('test' if j<a else ('cal' if j<a+b else 'train')) for j,g in enumerate(groups)}
        if source=='webis':
            ids=defaultdict(set)
            for r in rr:ids[r['id'][0]].add(r[group])
            bad={r['key'] for r in rr if len(ids[r['id'][0]])>1}
        else:bad=set()
        by=defaultdict(set)
        for r in rr:by[r['key']].add(slots[r[group]])
        panel[source]={k:next(iter(parts)) for k,parts in by.items() if len(parts)==1 and k not in bad}
    return panel

def magnitude(array):
    # Distributional summaries have meaning even when one argument has only one sentence.
    x=np.asarray(array);return np.r_[x.mean(0),x[-1],x.max(0),x.min(0)]

class Material:
    def __init__(self,archive):
        meta=json.loads((OUT/'material.json').read_bytes());self.meta=meta
        self.texts={r['id']:r['text'] for r in load_rows(DATA/'stimuli.jsonl')};self.parts=partition(meta);self.archive=None;self.certificate=None;self.native={}
        if archive:
            receipt=json.loads((archive/'receipt.json').read_bytes())
            if meta['source']!=receipt['source'] or sha(archive/'features.npz')!=receipt['file']:raise ValueError('Unverified whole-reason native archive')
            self.archive=np.load(archive/'features.npz',allow_pickle=False);self.native=receipt['articles'];self.certificate=sha(archive/'receipt.json')
        # Keep training material intact; withhold exact or 20-word repeated held material from scoring and source calibration.
        windows=set();train=set()
        for source in self.parts:
            train|={k for k,p in self.parts[source].items() if p=='train'}
        for k in train:windows.update(chunks(self.texts[k]))
        self.near={k for k,t in self.texts.items() if k not in train and chunks(t)&windows}
        for source,mapping in self.parts.items():self.parts[source]={k:v for k,v in mapping.items() if k not in self.near}
        self.expert=[r for r in meta['expert'] if r['eligible'] and r['key'] not in self.near and self.eligible(r['key'])]
        self.keys=sorted({k for groups in self.parts.values() for k in groups if self.eligible(k)}|{r['key'] for r in self.expert})
        self.index={k:i for i,k in enumerate(self.keys)};raw=[self.extract(k) for k in self.keys];self.x={m:np.stack([v[m] for v in raw]) for m in raw[0]}

    def eligible(self,k):return self.archive is None or k in self.native

    def extract(self,k):
        t=self.texts[k];regions=sentences(t);prev=set();last=set();vectors=[];all_terms=[];surprise=[];deeper=[];bag_terms=[];coord=0
        if self.archive is not None:
            rays=self.archive[k+'.char'];sur=self.archive[k+'.surprisal'].astype(float);reps=self.archive[k+'.rep'].astype(float);links=self.archive[k+'.links'].astype(float)
            if self.native[k]['sentences']!=list(map(list,regions)):raise ValueError('Unmarked original sentence split mismatch')
            values=reps.reshape(len(reps),5,-1);unit=values/np.maximum(1e-7,np.linalg.norm(values,axis=2,keepdims=True))
        for j,(a,b) in enumerate(regions):
            part=t[a:b];matches=list(WORD.finditer(part));words=[m.group().casefold() for m in matches];parts=set(words)-STOP
            form=np.array([np.log1p(len(words)),np.log1p(b-a),len(parts)/max(len(words),1),len(parts&prev)/max(len(parts),1),len(parts&last)/max(len(parts|last),1),
                           len(parts-prev)/max(len(parts|prev),1),part.count('?'),part.count('!'),part.count(','),sum(w.isdigit() for w in words)] + [len(m.findall(part.casefold())) for m in MARKERS])
            vectors.append(form);all_terms.extend(words);bag_terms.extend(parts);prev|=parts;last=parts
            if self.archive is None:continue
            ids=slice(coord,coord+len(words));off=np.array([[a+m.start(),a+m.end()] for m in matches],dtype=int).reshape(-1,2)
            if not np.array_equal(rays[ids,:2],off):raise ValueError('Original printed units differ from complete word evidence')
            coord+=len(words);q=sur[ids];surprise.append([q.mean(),np.max(q),q[0],q[-1],np.median(q)] if len(q) else np.zeros(5))
            # Similarity only to preceding complete sentences. Current meaning at the last word and average state have separate effects.
            relations=(unit[:j]*unit[j]).sum(2)
            interaction=np.r_[relations.max(0),relations[-1],relations.mean(0),links[j,:j].max(0),links[j,j-1]] if j else np.zeros(19)
            deeper.append(np.r_[values[j,0,:32],values[j,1,:32],values[j,3,:32],interaction])
        f=np.r_[np.log1p(len(t)),np.log1p(len(regions)),np.log1p(len(all_terms)),len(set(all_terms))/max(1,len(all_terms)),magnitude(vectors),
                sketch(bag_terms,96),sketch([u+' '+v for u,v in zip(all_terms,all_terms[1:]) if u not in STOP or v not in STOP],64)]
        o={'form':f}
        if self.archive is not None:
            o['surprisal']=np.r_[f,magnitude(surprise)];deep=np.stack(deeper)
            o['native']=np.r_[o['surprisal'],deep.mean(0),deep[-1],deep[:,-19:].max(0),deep[:,-19:].min(0)]
        return o
