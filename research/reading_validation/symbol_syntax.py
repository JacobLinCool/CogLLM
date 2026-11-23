"""Incremental TeX mention proposals and exact causal commit points, without annotations."""
from dataclasses import dataclass
import re

# Literal notation from the two released scientific-writing corpora.
DELIM=re.compile(r'\\begin\{(?:equation\*?|align\*?|gather\*?|multline\*?|displaymath|eqnarray\*?)\}|\\end\{(?:equation\*?|align\*?|gather\*?|multline\*?|displaymath|eqnarray\*?)\}|(?<!\\)\$\$?|\[\[|\]\]|\\[\[\]()]')
IDENT=re.compile(r"\\[A-Za-z]+|[A-Za-z]{1,12}|\d+(?:\.\d+)?")
EXCLUDE=set('frac dfrac tfrac begin end mathrm mathit mathbf mathcal mathbb text operatorname hbar left right sum prod int in to times cdot eq le ge label ref nonumber quad qquad Big big Bigg bigg! mid mathop ldots dots sqrt pm neq rightarrow leftrightarrow rm bf underline overline'.split())


@dataclass(frozen=True,order=True)
class Mention:
    start:int
    end:int
    committed:int
    kind:str


def _group(text,i,limit):
    if i>=limit:return i
    if text[i]!='{':
        m=IDENT.match(text,i,limit);return m.end() if m else i+1
    depth=0
    for t in range(i,limit):
        if text[t]=='{':depth+=1
        if text[t]=='}':
            depth-=1
            if not depth:return t+1
    return i


def proposals(text):
    out={};cursor=0;mode=None;body=None;opener=None
    def add(a,b,commit,kind):
        if a<b and commit<=len(text):
            v=Mention(a,b,commit,kind)
            if (a,b) not in out or commit<out[(a,b)].committed:out[(a,b)]=v
    def content(start,stop,closed,closing):
        left=start;right=stop
        while left<right and text[left].isspace():left+=1
        while right>left and text[right-1].isspace():right-=1
        if closed and right-left<=110:
            add(left,right,closing,'expression');add(opener,closing,closing,'delimited')
        for m in IDENT.finditer(text,start,stop):
            a,b=m.span();v=m.group().lstrip('\\');k=b
            if v in EXCLUDE:continue
            while k<stop and text[k] in "'_^{":
                if text[k]=="'":k+=1;continue
                if text[k]=='{' and (not text[a:b].startswith('\\') or k!=b):break
                j=_group(text,k if text[k]=='{' else k+1,stop)
                if j<=k or (text[k] in '_^' and j==k+1):break
                k=j
            # A lexical boundary or a completed TeX group must be visible before emitting an identifier.
            if b<stop or closed:add(a,b,min(b+1,closing),'atom')
            if k>b and (k<stop or closed) and (k>=stop or text[k] not in "'_^{"):
                add(a,k,min(k+1,closing),'decorated')
    for m in DELIM.finditer(text):
        d=m.group();closing=d.startswith('\\end{') or d in (']]',r'\]',r'\)')
        if mode:
            matches=(d==mode if mode.startswith('$') else d==(r'\end'+mode[6:] if mode.startswith(r'\begin') else {'[[':']]',r'\[':r'\]',r'\(':r'\)'}.get(mode)))
            if matches:content(body,m.start(),True,m.end());mode=None
        elif not closing and d not in (r'\ ',):mode=d;body=m.end();opener=m.start()
    if mode:content(body,len(text),False,len(text))
    return sorted(out.values(),key=lambda m:(m.committed,m.end,m.start))
