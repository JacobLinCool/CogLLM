"""Align frozen native token evidence to original words, expert lines, and competing sources."""

from collections import Counter, defaultdict
import json
import re

import numpy as np

from research.reading_validation.memory_data import sha
from research.reading_validation.narrative_course_data import OUT,WORK,STORIES,fold


def unit_vector(x):
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-9)


class Story:
    def __init__(self,name,row,native,outcomes):
        self.name=name
        self.text=row["text"]
        self.bounds=np.array(row["spans"])
        self.offsets=native["offsets"]
        self.last=native["ends"]
        self.units=native["lines"]
        self.trace=native["trace"]
        self.relations=native["pairs"]
        self.state=native["state"].astype(float)
        self.words=outcomes[name+".words"]
        self.ratings=outcomes[name+".rating"]
        self.screens=outcomes[name+".screen"]
        keys=[k for k in outcomes.files if k.startswith(name+".") and k.split(".")[1] in ("AA","JE")]
        self.teams={k.split(".")[1]:outcomes[k] for k in keys}
        self.n=len(self.bounds)
        if any(len(s)!=self.n for s in self.teams.values()) or self.ratings.shape[1]!=len(self.words):
            raise ValueError("Independent reader and annotation geometries do not match")
        self.wordform=[fold(self.text[a:b]) for a,b,_ in self.words]
        # Use the last native piece of the original whitespace-delimited word, including its punctuation.
        stops={int(b):i for i,(_,b) in enumerate(self.offsets)}
        self.word_ends=np.array([stops.get(int(b),-1) for _,b,_ in self.words])
        good=self.word_ends>=0
        if np.any(self.units[self.word_ends[good]]!=self.words[good,2]) or good.mean()<.98:
            raise ValueError("The native word-end representation is not certified against the input")
        self.word_ok=good
        self.feature_blocks=self.build_words()
        self.line_inputs={k:self.line_mean(v,good) for k,v in self.feature_blocks.items()}
        self.line_terms=[set() for _ in range(self.n)]
        self.line_texts=[self.text[a:b] for a,b in self.bounds]
        for w,(_,_,j) in zip(self.wordform,self.words,strict=True):
            self.line_terms[j].add(w)
        self.line_state=np.stack([self.state[(self.units==j)&(np.arange(len(self.units))<=i)].mean(0)
                                  for j,i in enumerate(self.last)])
        self.comparisons,self.events,self.totals=self.identities()

    def line_mean(self,values,valid=None):
        w=self.words[:,2]
        if valid is None:
            valid=np.ones(len(w),bool)
        out=[]
        for j in range(self.n):
            take=(w==j)&valid
            out.append(values[take].mean(0) if take.any() else np.full(values.shape[1:],np.nan))
        return np.stack(out)

    def build_words(self):
        past=Counter()
        line_terms=defaultdict(set); accum=Counter(); line_words=Counter()
        forms=[]; flow=[]
        for i,((a,b,j),w,t) in enumerate(zip(self.words,self.wordform,self.word_ends,strict=True)):
            # Determine subword interval by source offsets, including multi-piece Unicode words.
            lo=np.searchsorted(self.offsets[:,1],a,side="right")
            x=self.trace[lo:t+1] if t>=lo else np.empty((0,self.trace.shape[1]))
            prev=line_terms[j-1]
            v=float(x[:,4].mean()) if len(x) else 0.
            accum[j]+=v
            count=line_words[j]
            forms.append([np.log1p(i),np.log1p(j),np.log1p(len(w)),self.text[a:b][:1].isupper(),
                          np.log1p(past[w]),float(w in prev),np.log1p(count),v,accum[j]/(1+count)])
            if t>=0 and len(x):
                flow.append([x[-1,0],x[:,1].sum(),x[:,2].sum(),x[:,3].max(),*x[-1,5:]])
            else:
                flow.append(np.zeros(8))
            line_terms[j].add(w);past[w]+=1;line_words[j]+=1
        forms=np.array(forms)
        native=np.column_stack((forms,np.array(flow),self.state[self.word_ends.clip(min=0)]))
        return {"progress":forms[:,:2],"form":forms,"semantic":native}

    def identities(self):
        pairs=[];events=[];stats=Counter()
        for team,states in self.teams.items():
            starts=[np.flatnonzero(a==1) for a in states.T]
            start=np.array([x[0] if len(x) else -1 for x in starts])
            for t in range(1,self.n):
                closed=np.flatnonzero((states[t-1]==1)&(states[t]==0))
                doubtful=np.flatnonzero((states[t-1]==2)&(states[t]==0))
                stats["imprecise_closures"]+=len(doubtful)
                rivals=np.flatnonzero((states[t-1]==1)&(states[t]==1))
                gold=sorted({int(start[k]) for k in closed if 0<=start[k]<t})
                if gold:
                    events.append((team,t,gold))
                for k in closed:
                    positive=int(start[k])
                    if not 0<=positive<t:
                        raise ValueError("A closed question lacks a prior open line")
                    different=[int(start[q]) for q in rivals if 0<=start[q]<t and start[q]!=positive]
                    stats["certain_closures"]+=1
                    stats["same_start_competitors"]+=sum(start[q]==positive for q in rivals)
                    stats["closures_with_rivals"]+=bool(different)
                    for negative in different:
                        pairs.append((team,t,int(k),positive,negative))
        return pairs,events,dict(stats)

    def pair_matrix(self,t):
        a=np.arange(t)
        if not t:
            raise ValueError("No earlier candidate")
        sets=self.line_terms
        words=len(sets[t])
        overlaps=np.array([len(s&sets[t]) for s in sets[:t]])
        quoted=lambda s:any(c in s for c in '„“»«"')
        forms=np.column_stack((np.log1p(t-a),np.log1p(a),np.log1p([len(s) for s in sets[:t]]),
                               overlaps/max(words,1),overlaps/np.maximum(1,[len(s|sets[t]) for s in sets[:t]]),
                               [quoted(s)==quoted(self.line_texts[t]) for s in self.line_texts[:t]]))
        prior=self.line_state[:t]
        v=unit_vector(prior)@unit_vector(self.line_state[t])
        e=unit_vector(self.state[self.last[:t]])@unit_vector(self.state[self.last[t]])
        signal=self.relations[t,:t]
        return dict(form=forms,semantic=np.column_stack((forms,v,e,np.log1p(signal*100))),
                    distance=-(t-a),original=signal[:,3])


def load(archive):
    seal=json.loads((OUT/"input-seal.json").read_bytes())
    receipt=json.loads((archive/"receipt.json").read_bytes())
    if (sha(WORK/"packet.jsonl")!=receipt["packet"] or receipt["packet"]!=seal["packet"] or
        sha(WORK/"response.npz")!=seal["outcomes"] or any(sha(archive/k)!=v for k,v in receipt["files"].items())):
        raise ValueError("Narrative outcomes or native inputs differ from a hash lock")
    docs={r['id']:r for r in map(json.loads,(WORK/"packet.jsonl").read_text().splitlines())}
    data={}
    with np.load(WORK/"response.npz",allow_pickle=False) as outcomes:
        for story in STORIES:
            k=seal['articles'][story]['id']
            with np.load(archive/(k+".npz"),allow_pickle=False) as native:
                data[story]=Story(story,docs[k],native,outcomes)
    return data
