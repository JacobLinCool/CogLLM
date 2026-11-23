"""Causal literal span proposals and observation types in symbolic scientific explanations."""
from bisect import bisect_left,bisect_right
from collections import defaultdict
from hashlib import sha256
from functools import lru_cache
import re
import numpy as np
from research.backbone_transfer.extract import read_json
from research.reading_validation.symbol_geometry import ROOT
from research.reading_validation.symbol_features import canonical
from research.reading_validation.symbol_syntax import DELIM,Mention,proposals

ATOM=re.compile(r"\\[A-Za-z]+|[A-Za-z]+|[0-9]+|[^\s]")
WORD=re.compile(r"\\[A-Za-z]+|[A-Za-z]+|[0-9]+|[^\s]")
CUE={'denote','denotes','where','respectively','define','defined','denoted','symbol','called','let','be','is','are','represent','represents','corresponds','correspond','for','of','and','the','with','we','as'}
CHARS='\\$^_{}()[]|=,+-.:;<>/'
MAX_CHAR=96


@lru_cache(maxsize=1)
def annotations():
    source={}
    for phase in ('train','dev'):
        for path in sorted((ROOT/'symlink'/phase).glob('*.json')):
            for row in read_json(path).values():
                text=row['text'];key=sha256(text.encode()).hexdigest()
                if key in source:continue
                raw=row['entity'];symbols=set();description=set()
                for v in raw.values():
                    a,b=v['start'],v['end']
                    if text[a:b]!=v['text']:break
                    if v['label']=='SYMBOL':symbols.add((a,b))
                    elif v['label'] in ('PRIMARY','ORDERED'):description.add((a,b))
                else:source[key]=(symbols,description)
    return source


def gold(doc):
    if doc['id'] in annotations():return annotations()[doc['id']]
    return ({tuple(span) for j in doc['issues'] for span in j['sources']},
            {tuple(span) for j in doc['issues'] for span in j['definitions']})


def math_position(text):
    # Mark the state attained as each character arrives. Seeing a later delimiter never changes prior entries.
    v=np.zeros(len(text)+1,dtype=bool);d=None;tokens={}
    for m in DELIM.finditer(text):
        marker=m.group();end=m.end()
        if marker.startswith('$'):
            if end>=len(text) or text[end]=='$':continue
            end+=1
        tokens[end]=marker
    for j in range(len(text)):
        x=tokens.get(j+1)
        if x:
            if d:
                if x==d:d=None
            elif x.startswith('$'):d=x
            elif x.startswith(r'\begin'):d=x.replace('begin','end',1)
            elif x in (r'\[',r'\(','[['):d={r'\[':r'\]',r'\(':r'\)','[[':']]'}[x]
        v[j+1]=d is not None
    return v


def mentions(text):
    annotations_by_span={(m.start,m.end):m for m in proposals(text)};math=math_position(text);atoms=list(ATOM.finditer(text))
    def put(a,b,c,kind):
        if b<=a or b-a>MAX_CHAR or c>len(text):return
        k=(a,b)
        if k not in annotations_by_span or c<annotations_by_span[k].committed:annotations_by_span[k]=Mention(a,b,c,kind)
    for j,m in enumerate(atoms):
        start,end=m.span();tok=m.group();inner=math[start] or math[end];special=inner or bool(re.match(r'[\\$A-Z0-9_^{}]',tok))
        if not special and not (len(tok)==1 and tok.isalpha()):continue
        # A terminal delimiter can be recognized immediately; general lexical endpoints require a visible following character.
        commit=end+int(tok[-1].isalnum() or tok[-1] in ("'", "_", "\\"))
        if math[start] and tok.isalpha():
            for q in range(start,min(end,start+8)):put(q,q+1,q+2,'atom_character')
        if commit>len(text):continue
        put(start,end,commit,'token')
        if inner or tok[0] in '$}])' or tok[0].isupper() or tok[0].isdigit():
            for u in range(max(0,j-(15 if inner else 3)),j+1):
                a=atoms[u].start();first=atoms[u].group()
                if first[0] in ';,.=+*/)' or (not math[a] and not math[start] and not any(c in text[a:end] for c in '$\\_0123456789') and not first[0].isupper()):continue
                if not inner and any(s in text[a:end] for s in ('\n','. ', '; ','? ','! ')):continue
                put(a,end,commit,'notation_span')
    return sorted(annotations_by_span.values(),key=lambda m:(m.committed,m.end,m.start))


def shape(t):
    if not t:return 0
    c=t[0]
    if c in CHARS:return CHARS.index(c)+5
    if c.isdigit():return 1
    if c.isupper():return 2
    if c.islower():return 3
    return 4


def clipped_id(s):return int.from_bytes(sha256(s.encode()).digest()[:2],'little')%2048


def unit_features(page,ms):
    """Only the source, its committed terminal delimiter, and earlier material participate."""
    text=page.text;math=math_position(text);out=[]
    for m in ms:
        a,b,upto=m.start,m.end,m.committed;v=text[a:b];n=len(v);p=text[a-1:a] if a else '';q=text[b:upto];l,r=page.token_span(a,b);t=bisect_left(page.end,upto);s=(v.strip('$').strip());c=''.join(v.split())
        row=[np.log1p(n),np.log1p(r-l),np.log1p(t),math[a],math[b],v[0]=='$',v[-1]=='$',p=='$',q.startswith('$'),p.isspace(),q[:1].isspace(),v.startswith('\\'),v.endswith('}'),v.count('{')-v.count('}'),v.count('(')-v.count(')'),v.count('[')-v.count(']'),sum(x.isdigit() for x in v)/n,sum(x.isupper() for x in v)/n,sum(x.isalpha() for x in v)/n,sum(x.isspace() for x in v)/n,v.count('\\'),v.count('$'),v.count('^'),v.count('_'),v.count('='),v.count(','),v.count('~'),shape(v[0]),shape(v[-1]),shape(p),shape(q),m.kind in ('delimited','expression','atom','decorated'),len(s)==1,s.isalpha(),s.isupper(),l==r-1,page.start[l]==a,page.end[r-1]==b,min(a-page.start[l],6),min(page.end[r-1]-b,6)]
        out.append(np.r_[row,page.state[l,:20],page.state[r-1,:20]])
    return np.asarray(out,dtype='float32')


def observation(page,doc):
    all_symbols,all_desc=gold(doc);forward={tuple(j['definitions'][0]) for j in doc['issues'] if j['kind']=='forward'};y=np.zeros(page.n,dtype=int)
    for regions,begin,inside in ((all_desc,1,2),(forward,3,4)):
        spans=[page.interval(r) for r in regions]
        for ix in spans:y[ix]=inside
        for ix in spans:y[ix[0]]=begin
    return y


def tokens(page):
    text=page.text;atoms=list(WORD.finditer(text));ends=np.array([m.end() for m in atoms]);starts=np.array([m.start() for m in atoms]);state=math_position(text);cats=[];raw=[]
    for t,end in enumerate(page.end):
        j=bisect_left(starts,end)-1;info=[]
        for k in range(j,max(-1,j-6),-1):
            s=atoms[k].group()[:min(end,ends[k])-starts[k]];info.append(s.casefold())
        info+=['-']*max(0,6-len(info))
        q=info[0];hist=' '.join(info[:3]);prior=list(reversed(info[1:]));n=[np.log1p(len(q)),q.isalpha(),q.isdigit(),q in CUE,any(w in CUE for w in prior[:3]),shape(text[page.start[t]:end]),shape(text[end-1:end]),int(state[end]),np.log1p(t),page.base[t,1],page.base[t,2],page.base[t,3]]
        cats.append((q,info[1],info[2],q[-3:],q[:3],info[1]+'_'+q,info[2]+'_'+info[1],hist))
        raw.append(np.r_[n,page.state[t,:32],page.window[t,:24],page.state[t,:24]-page.state[max(0,t-1),:24]])
    return cats,np.asarray(raw,dtype='float32')
