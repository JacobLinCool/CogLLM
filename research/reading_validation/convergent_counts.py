"""Recover only integer cloze response counts identifiable from the investigators' sufficient summaries."""
from collections import Counter,defaultdict
from math import log
from csv import DictReader
from research.backbone_transfer.extract import digest,write_json
from research.reading_validation.convergent_word_prepare import ROOT,OUT


def admissible(p,other,h,n):
    answers=set()
    for k in range(2,n+2):
        d=n+k;sa=p*d;sb=other*d;a=round(sa)-1;b=round(sb)-1
        if max(abs(sa-a-1),abs(sb-b-1))>1e-7 or min(a,b)<0 or a+b>n:continue
        m=k-1-int(a>0)-int(b>0);left=n-a-b
        if m<0 or left<m or left>m*b:continue
        # Entropy is Schur concave. Extremes allocate remaining integer responses as evenly/unevenly as the largest competitor permits.
        uniform=[];concentrated=[]
        if m:
            base=left//m;rem=left%m;uniform=[base]*(m-rem)+[base+1]*rem;remaining=left-m
            for j in range(m):
                extra=min(max(b-1,0),remaining);concentrated.append(1+extra);remaining-=extra
        counts=([a] if a else [])+([b] if b else [])+[0]
        def entropy(v):return -sum((c+1)/d*log((c+1)/d) for c in v+counts)
        if entropy(concentrated)-1e-7<=h<=entropy(uniform)+1e-7:answers.add(a)
    return answers


def study():
    groups=defaultdict(list);rows=list(DictReader((ROOT/'outcomes.csv').open()));out={};stable={};N=[79,76,79,78,78,80,78,79]
    for r in rows:
        p=float(r['cloze_p_smoothed']);alt=p-float(r['competition']);h=float(r['entropy']);groups[int(r['list'])].append((int(r['item_id']),p,alt,h))
    for group,X in sorted(groups.items()):
        allowed={};dist={}
        for n in range(65,89):
            a=[admissible(p,v,h,n) for _,p,v,h in X];ok=sum(bool(z) for z in a);dist[n]=ok
            if ok==len(a):allowed[n]=a
        possible=sorted(allowed);stable[group]=dict(rows=len(X),published_nominal=N[group-1],all_admissible_N=possible,items_supported_by_N=dist)
        if len(possible)==1:
            n=possible[0]
            for (item,_,_,_),answers in zip(X,allowed[n]):
                if len(answers)==1:out[item]=(n,answers.pop())
    return out,stable


if __name__=='__main__':
    out,s=study();print('certified',len(out),[(g,q['rows'],q['all_admissible_N'],sorted(q['items_supported_by_N'].items(),key=lambda k:k[1],reverse=True)[:2]) for g,q in s.items()]);write_json(OUT/'cloze-count-identification.json',dict(source=digest(ROOT/'outcomes.csv'),preprocessing=digest(ROOT/'preprocess.py'),allowed_totals='65–88, identical retained respondents for all included positions within a list',recovered=len(out),cohorts=s))
