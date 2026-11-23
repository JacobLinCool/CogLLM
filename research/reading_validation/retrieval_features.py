"""Recover causal word-to-earlier-sentence features from a sealed native archive."""

import json
from pathlib import Path
import re

import numpy as np

from research.reading_validation.memory_data import sha
from research.reading_validation.retrieval_data import DATA


STOP = frozenset("a an the and in on of to as or it its their at by for is was were be has have that his her he she they we with from had did".split())
SURFACE = ("log_distance","log_prior_length","current_seen","same_word","shared_fraction","shared_content_fraction",
           "negation_difference","numeric_difference")
SCALAR = SURFACE + ("attention_mean","attention_peak","attention_history","word_best_similarity",
                    "mean_best_similarity","last_similarity","mean_similarity","logit_terminal","logit_average","logit_peak")


def normalize(x):
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-9)


class Corpus:
    def __init__(self,archive):
        archive=Path(archive)
        receipt=json.loads((archive/"receipt.json").read_bytes())
        if sha(DATA/"packet.jsonl")!=receipt["packet"] or any(sha(archive/k)!=v for k,v in receipt["files"].items()):
            raise ValueError("Native source differs from signed receipt")
        self.metadata=json.loads((archive/"layout.json").read_bytes())
        self.rows={r["id"]:r for r in map(json.loads,(DATA/"packet.jsonl").read_text().splitlines())}
        self.index={k:i for i,k in enumerate(self.metadata["ids"])}
        if self.index.keys()!=self.rows.keys():
            raise ValueError("Archive geometry differs from source articles")
        self.arrays={k:np.load(archive/(k+".npy"),mmap_mode="r",allow_pickle=False)
                     for k in ("coordinates","states","signals","relations")}

    def page(self,key):
        i=self.index[key]
        word=slice(*self.metadata["words"][i:i+2])
        link=slice(*self.metadata["relations"][i:i+2])
        row=self.rows[key]
        ww=self.arrays["coordinates"][word]
        n=len(row["spans"])
        att=self.arrays["relations"][link].reshape(len(ww),n,2)
        return Page(row["text"],ww,self.arrays["states"][word],att,self.arrays["signals"][word])


class Page:
    def __init__(self,text,words,states,attention,signals):
        self.text=text
        self.coords=words[:,:2]
        self.unit=words[:,2]
        self.certified=words[:,5].astype(bool)
        self.tokens=words[:,4]+1
        self.state=normalize(states.astype(np.float32))
        self.attention=attention.astype(np.float32)
        self.signals=signals
        self.forms=[re.sub(r"[^\w'-]+","",text[a:b].casefold()) for a,b in self.coords]
        self.sections=[np.flatnonzero(self.unit==j) for j in range(int(self.unit.max())+1)]
        if not self.certified.all() or any(not len(x) for x in self.sections):
            raise ValueError("One or more source words lack a causal certificate")

    def pairs(self,section,terminal=False):
        current=self.sections[section]
        n=len(current)
        if section==0:
            raise ValueError("No earlier sentence to retrieve")
        word=self.state[current]
        steps=np.arange(1,n+1,dtype=np.float32)
        mean=normalize(np.cumsum(word,axis=0)/steps[:,None])
        forms=np.array([self.forms[j] for j in current],dtype=str)
        cnt=np.array([bool(s and s not in STOP) for s in forms])
        negative=np.maximum.accumulate(np.isin(forms,["not","never","no","without","neither"]))
        number=np.maximum.accumulate(np.array([any(c.isdigit() for c in v) for v in forms]))
        logits=self.signals[current,0]
        values=[]
        for previous,ix in enumerate(self.sections[:section]):
            bank=self.state[ix]
            earlier=normalize(bank.mean(0))
            last=bank[-1]
            vocab={self.forms[j] for j in ix}
            shared=np.isin(forms,list(vocab))
            m=word@bank.T
            best=m.max(1)
            att=self.attention[current,previous]
            base=np.maximum(1,self.signals[ix,2].sum())
            rel=np.log1p(att*self.tokens[current,None]/base)
            scalars=np.column_stack((np.full(n,np.log1p(section-previous)),np.full(n,np.log1p(len(ix))),
                                     np.log1p(steps),shared,np.cumsum(shared)/steps,
                                     np.cumsum(shared&cnt)/np.maximum(1,np.cumsum(cnt)),
                                     np.logical_xor(negative,bool(vocab&{"not","never","no","without","neither"})),
                                     np.logical_xor(number,any(any(c.isdigit() for c in v) for v in vocab)),
                                     rel[:,0],rel[:,1],np.cumsum(rel[:,0])/steps,best,np.cumsum(best)/steps,
                                     word@last,mean@earlier,logits,np.cumsum(logits)/steps,np.maximum.accumulate(logits)))
            if scalars.shape[1]!=len(SCALAR):
                raise ValueError("Pair feature columns changed")
            # The two sentence vectors and their interactions preserve directional information.
            raw=np.column_stack((scalars,mean,np.broadcast_to(earlier,mean.shape),mean*earlier,np.abs(mean-earlier)))
            values.append(raw[-1:] if terminal else raw)
        return np.stack(values,axis=1).astype(np.float32)
