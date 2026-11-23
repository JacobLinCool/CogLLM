"""Prefix-only span proposals and term-specific lexical, contextual, and attention evidence."""

from collections import Counter
from pathlib import Path
import re

import numpy as np

from research.backbone_transfer.extract import digest, read_json, stream_rows
from research.reading_validation.definition_data import ROOT


WIDTH=8
FUNCTION=set('a an the this that these those another his her my our their your each such as to at in on and or but is are was were of for by from into be been being some most many it which who when how why also has have had with than its can may not more any two he she they we i other very before after first however while there then if will through its no because'.split())
DEFINITION=set('is are was were means mean meaning denotes called known termed defined defines refers refers-to describes denotes consists consists-of process type kind form refers implies occurs consists occurs happens i.e namely example examples involve involves represents represents'.split())
PUNCT=re.compile(r'[.!?;:\n]')


def standard(v):
    return v/np.maximum(np.sqrt(np.mean(v*v,axis=-1,keepdims=True)),.001)


class Corpus:
    def __init__(self,directory,source=ROOT/'stimuli.jsonl'):
        self.geometry=read_json(directory/'geometry.json')
        if self.geometry['source']!=digest(source):raise ValueError('Unmatched textbook and model receipts')
        self.rows={x['id']:x for x in stream_rows(source)};self.index={k:i for i,k in enumerate(self.geometry['ids'])}
        if self.rows.keys()!=self.index.keys():raise ValueError('Missing input')
        self.states=np.load(directory/'states.npy',mmap_mode='r');self.attention=np.load(directory/'attention.npy',mmap_mode='r')
        self._ids=None;self._page=None

    def page(self,key):
        if key!=self._ids:
            i=self.index[key];a,b=self.geometry['token_offsets'][i:i+2];c,d=self.geometry['attention_offsets'][i:i+2];r=self.rows[key]
            self._page=Page(r['text'],r['words'],self.states[a:b],self.attention[c:d].reshape(len(r['words']),b-a,2));self._ids=key
        return self._page


class Page:
    def __init__(self,text,geometry,tokens,attention):
        self.text=text;self.words=np.asarray(geometry,dtype=np.int64);self.tokens=np.asarray(tokens,dtype='float32')
        self.attention=attention
        self.raw=[text[a:b] for a,b,_,_ in geometry];self.forms=[w.casefold() for w in self.raw];self.n=len(self.raw)
        self.function=np.array([w in FUNCTION for w in self.forms]);self.cop=np.array([w in DEFINITION for w in self.forms])
        self.cap=np.array([w[0].isupper() for w in self.raw]);self.length=np.array(list(map(len,self.raw)))
        self.digit=np.array([any(c.isdigit() for c in w) for w in self.raw]);self.upper=np.array([w.isupper() and len(w)>1 for w in self.raw])
        self.cross=np.zeros(self.n,dtype=int);self.since=np.zeros(self.n);seen=Counter();self.freq=[]
        for j in range(self.n):
            self.freq.append(seen[self.forms[j]]);seen[self.forms[j]]+=1
            if j:
                self.cross[j]=self.cross[j-1]+bool(PUNCT.search(text[self.words[j-1,1]:self.words[j,0]]))
                self.since[j]=self.since[j-1]+1 if self.cross[j]==self.cross[j-1] else 0
        self.freq=np.asarray(self.freq);self.end=standard(self.tokens[self.words[:,3]]);self.start=standard(self.tokens[self.words[:,2]])
        self.prefix=np.concatenate((np.zeros((1,self.end.shape[1]),dtype='float32'),self.end.cumsum(0)))
        self.starts={a:i for i,(a,b,_,_) in enumerate(geometry)};self.ends={b:i for i,(a,b,_,_) in enumerate(geometry)}
        self._grid=None

    def word_span(self,coords):
        a,b=coords
        if a not in self.starts or b not in self.ends:return None
        i,j=self.starts[a],self.ends[b]
        return (i,j) if i<=j and j-i<WIDTH and self.cross[i]==self.cross[j] else None

    def word_interval(self,coords):
        a,b=coords;grid=np.where((self.words[:,1]>a)&(self.words[:,0]<b))[0]
        return (int(grid[0]),int(grid[-1])+1) if len(grid) else None

    def grid(self):
        if self._grid is None:
            end=np.repeat(np.arange(self.n),WIDTH);length=np.tile(np.arange(1,WIDTH+1),self.n);start=end-length+1
            good=(start>=0)&(self.cross[end]==self.cross[np.maximum(start,0)])
            self._grid=(start[good],end[good])
        return self._grid

    def span_features(self,left,right,arm):
        l=np.asarray(left,dtype=np.int64);r=np.asarray(right,dtype=np.int64);size=r-l+1;den=np.maximum(size,1)
        def avg(v):
            cs=np.r_[0.,v.cumsum()];return (cs[r+1]-cs[l])/den
        prev=np.maximum(0,l-1)
        surf=np.column_stack((np.log1p(size),np.log1p(self.words[r,1]-self.words[l,0]),np.log1p(self.length[r]),np.log1p(self.length[l]),
            self.function[l],self.function[r],avg(self.function),avg(self.digit),avg(self.cap),avg(self.upper),self.cap[l],self.upper[l],
            np.log1p(self.freq[r]),np.log1p(self.freq[l]),(l>0)*self.function[prev],(l>0)*self.cop[prev],
            np.log1p(self.since[r]),np.log1p(r),np.log1p(self.cross[r]),avg(self.cop),
            [self.forms[j].endswith(('tion','ity','ism','ology','ics','ment','ance','ness')) for j in r],
            [self.forms[j].endswith(('ing','ed','ly')) for j in r])).astype('float32')
        if arm=='surface':return surf
        mean=standard((self.prefix[r+1]-self.prefix[l])/den[:,None]);a=self.start[l];b=self.end[r]
        return np.column_stack((surf,a,b,mean,a*b)).astype('float32')

    def link_features(self,span,targets,arm):
        a,b=span;t=np.asarray(targets,dtype=np.int64)
        if not 0<=a<=b or np.any(t<=b):raise ValueError('A definition must follow the observed term')
        captures=set(self.forms[a:b+1])-FUNCTION
        match=np.array([f in captures for f in self.forms],dtype='float32')
        def recent(v,width):
            cs=np.r_[0.,v.cumsum()];lo=np.maximum(t-width+1,b+1)
            return (cs[t+1]-cs[lo])/np.maximum(1,t-lo+1)
        td=t-b
        surface=np.column_stack((np.log1p(td),np.log1p(t),np.log1p(b-a+1)*np.ones(len(t)),self.cross[t]-self.cross[b],self.cop[t],
            recent(self.cop,4),recent(self.cop,8),match[t],recent(match,5),recent(match,12),self.function[t],self.cap[t],
            np.log1p(self.length[t]),np.log1p(self.since[t]),np.log1p(self.freq[t]),np.array([np.max(match[b+1:k+1]) for k in t]),
            np.log1p(np.max(self.length[a:b+1]))*np.ones(len(t)))).astype('float32')
        if arm=='position':return surface[:,:4].copy()
        if arm=='surface':return surface
        anchor=standard((self.prefix[b+1]-self.prefix[a])/(b-a+1));anchor_last=self.end[b];cu=self.end[t]
        lo=np.maximum(t-3,b+1);window=standard((self.prefix[t+1]-self.prefix[lo])/(t-lo+1)[:,None]);last=np.broadcast_to(anchor,cu.shape)
        raw=self.attention[t,self.words[a,2]:self.words[b,3]+1,:].astype('float32')/32.
        rel=np.column_stack((raw[:,:,0].mean(1),raw[:,:,0].max(1),raw[:,:,1].max(1)))
        return np.column_stack((surface,rel,cu,window,last,cu*anchor_last,window*anchor,cu*anchor)).astype('float32')
