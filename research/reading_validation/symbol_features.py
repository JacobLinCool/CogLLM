"""Causal notation anchors and evidence features for a candidate symbol at each visible token."""
from bisect import bisect_left,bisect_right
from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
import re
import numpy as np
from research.reading_validation.symbol_syntax import Mention,proposals

CUE=re.compile(r'\b(?:where|here|denote[sd]?|represent(?:s|ed)?|mean(?:s|ing)?|define[sd]?|stand(?:s)? for|correspond(?:s|ing)? to|is|are|the|respectively|given by|called)\b',re.I)
DISTINCT=re.compile(r'\b(?:denote[sd]?|represent(?:s|ed)?|mean(?:s|ing)?|define[sd]?|stand(?:s)? for|correspond(?:s|ing)? to|is|are|given by|called)\b',re.I)
RESET=re.compile(r'[.;!?\n]')
WORD=re.compile(r"[^\W_]+(?:[-’'][^\W_]+)*",re.UNICODE)


def canonical(s):
    return re.sub(r'\s+','',s.strip().strip('$').removeprefix('[[').removesuffix(']]').strip())


def cosine(a,b):
    return (a*b).sum(-1)/(np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1)+1e-7)


@dataclass
class Page:
    row:dict
    words:object
    states:np.ndarray
    relations:np.ndarray

    def __post_init__(self):
        self.text=self.row['text'];o=self.words.offsets;self.start=np.array([v[0] for v in o]);self.end=np.array([v[1] for v in o]);self.n=len(o);self.keys=np.asarray(self.row['keys'],dtype=int)
        self.native_positions={k:i for i,k in enumerate(self.keys)}
        self.mentions=proposals(self.text);self.by_name=defaultdict(list)
        for m in self.mentions:self.by_name[canonical(self.text[m.start:m.end])].append(m)
        t=np.arange(self.n);last_distinct=[];last_clause=[];lastcue=[];kind=[]
        visible=[self.text[a:b] for a,b in o]
        # Scan the textual prefix only. A complete word is committed when a following delimiter is already visible.
        for regex,dest in [(DISTINCT,last_distinct),(CUE,lastcue),(RESET,last_clause)]:
            by=np.full(self.n,-1,dtype=int)
            for match in regex.finditer(self.text):
                commit=match.end() if regex is RESET else match.end()+1;k=bisect_left(self.end,commit)
                if k<self.n:by[k]=k
            dest.extend(np.maximum.accumulate(by))
        self.last_distinct=np.asarray(last_distinct);self.last_clause=np.asarray(last_clause);self.lastcue=np.asarray(lastcue)
        self.latest_ordinal=np.zeros(self.n,dtype=np.float32)
        for pat,sign in [('first|former',1.),('second|latter',-1.),('respectively',.5)]:
            for m in re.finditer(r'\b(?:'+pat+r')\b',self.text,re.I):
                k=bisect_left(self.end,m.end()+1);self.latest_ordinal[k:min(k+12,self.n)]=sign
        prev=np.maximum(t-3,0);self.state=self.states.astype(np.float32);c=np.vstack([np.zeros(self.state.shape[1],np.float32),self.state.cumsum(0)]);self.window=(c[t+1]-c[prev])/(t-prev+1)[:,None]
        self.base=np.c_[np.log1p(t),np.log1p(t-self.last_distinct),np.log1p(t-self.lastcue),np.log1p(t-self.last_clause),self.last_distinct>self.last_clause,self.latest_ordinal,
                        [float(bool(re.search(r'[A-Za-z]',s))) for s in visible],[float(bool(re.search(r'[.,;:]',s))) for s in visible],[float(bool(re.search(r'[$\\=]',s))) for s in visible]]

    def token_span(self,a,b):
        lo=bisect_right(self.end,a);hi=bisect_left(self.start,b)
        if lo>=hi or lo>=self.n or hi>self.n:raise ValueError('Unaligned source span')
        return lo,hi

    def interval(self,span):
        a,b=self.token_span(*span);return np.arange(a,b)

    def features(self,anchor,t):
        t=np.asarray(t,dtype=int);a,b=self.token_span(anchor.start,anchor.end);commit=bisect_left(self.end,anchor.committed)
        if t.min()<commit:raise ValueError('Attempted to use a source before its proposal')
        name=canonical(self.text[anchor.start:anchor.end]);mentions=self.by_name[name];later=[];seen=set()
        for m in mentions:
            if m.start<anchor.start or (m.start<anchor.end and m.end>anchor.start):continue
            mlo,mhi=self.token_span(m.start,m.end);j=bisect_left(self.end,m.committed)
            if (mlo,mhi) not in seen:later.append((j,mlo,mhi));seen.add((mlo,mhi))
        later=sorted(later);arr=np.array([commit]+[j for j,_,_ in later]);hist=np.searchsorted(arr,t,side='right')-1
        lo=np.asarray([a]+[l for _,l,_ in later])[hist];hi=np.asarray([b]+[u for _,_,u in later])[hist];age=t-b+1;reage=t-hi+1
        before=sum(m.end<=anchor.start and m.committed<=anchor.committed for m in mentions);ds=t-self.last_distinct[t]
        surface=np.c_[self.base[t],np.log1p(age),np.log1p(reage),np.minimum(reage,5),np.log1p(hist),np.full(len(t),np.log1p(before)),np.full(len(t),np.log1p(b-a)),np.full(len(t),np.log1p(len(name))),
                      hist>0,hi-1>=self.last_clause[t],hi-1<=self.last_distinct[t],(hi-1<=self.last_distinct[t])&(ds<=6),reage<4,reage<9,np.exp(-reage/8),
                      np.full(len(t),name.startswith('\\')),np.full(len(t),'_' in name),np.full(len(t),'^' in name),np.full(len(t),anchor.kind=='delimited')]
        src=self.state[a:b].mean(0);last=self.state[hi-1];cue=self.state[t];ctx=self.window[t];groups={}
        for start,stop in set(zip(lo,hi)):
            groups[(start,stop)]=[self.native_positions[k] for k in range(start,stop) if k in self.native_positions]
        r=np.zeros((len(t),4),dtype='float32');rawkeys=[self.native_positions[k] for k in range(a,b) if k in self.native_positions]
        if rawkeys:
            z=self.relations[np.ix_(t,rawkeys)];r[:,0]=z[:,:,0].mean(1)/32;r[:,1]=z[:,:,1].max(1)/32
        for span,kk in groups.items():
            if not kk:continue
            m=(lo==span[0])&(hi==span[1]);z=self.relations[np.ix_(t[m],kk)];r[m,2]=z[:,:,0].mean(1)/32;r[m,3]=z[:,:,1].max(1)/32
        semantic=np.column_stack([r,cosine(src,cue),cosine(src,ctx),cosine(last,cue),cosine(last,ctx),cue,ctx,src[None,:]*ctx,last*ctx])
        return surface.astype('float32'),semantic.astype('float32')
