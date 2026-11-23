"""Recover the published GPT-4 reference and compute grade-independent format and human-reference baselines."""

import json
import re
import unicodedata
from collections import Counter

import numpy as np

from research.reading_validation.argument_data import sentences
from research.reading_validation.local_forecast import digest
from research.reading_validation.summeval_data import OUT,WORK,material

G=WORK/'g-eval-coherence.json';G_SHA='bec62a6a8408b7f0c7a55a665d8c9a44b97eecf671bf0b4319284b192d218508';TOKEN=re.compile(r'[^\W_]+',re.UNICODE)


def tokens(t):return TOKEN.findall(unicodedata.normalize('NFKC',t).casefold())


def print_identity(t):
    s=unicodedata.normalize('NFKC',t).casefold();s=re.sub(r'-(lrb|rrb|lsb|rsb|lcb|rcb)-',' ',s);return ''.join(TOKEN.findall(s))


def matched(original,other):
    if original==other:return 'exact'
    norm=print_identity(original)
    if norm==print_identity(other):return 'punctuation_whitespace_case'
    for encoding in ('latin1','cp1252'):
        try:repaired=other.encode(encoding).decode('utf8')
        except UnicodeError:continue
        if norm==print_identity(repaired):return 'reversible_'+encoding
    return None


def original_geval_parse(output):
    match=re.search(r'^ ?([\d\.]+)',output)
    if match is None:return 0.
    try:return float(match.group(1))
    except ValueError:return 0.


def score_geval(data):
    if digest(G)!=G_SHA:raise ValueError('Different external G-Eval release')
    lookup={(k,m):t for k,m,t,_ in data};out={};states=Counter();counts=Counter();parse_zero=0
    for v in json.loads(G.read_bytes()):
        k=(v['doc_id'],v['system_id']);grade=matched(lookup[k],v['system_output'])
        if grade is None:raise ValueError('G-Eval identity or content mismatch')
        raw=v['all_responses'];scores=[original_geval_parse(s) for s in raw];states[grade]+=1;counts[len(scores)]+=1;parse_zero+=sum(v==0 for v in scores)
        if not scores or k in out:raise ValueError('Missing repeated GPT-4 answers or duplicated method')
        out[k]=sum(scores)/len(scores)
    if set(out)!=set(lookup):raise ValueError('Different expert-rated drafts and GPT-4 outputs')
    return out,{'public_sha':G_SHA,'text_matches':dict(states),'repeated_responses':dict(counts),'source_parser_fallback_zero':parse_zero,'human_scores_field_used':False}


def ngrams(t,n):return Counter(tuple(t[j:j+n]) for j in range(len(t)-n+1))


def lexical(t,refs):
    a=tokens(t);spans=sentences(t);parts=[tokens(t[i:j]) for i,j in spans];v=[];prev=set()
    for part in parts:
        if prev:v.append(sum(x in prev for x in part)/max(1,len(part)))
        prev.update(part)
    count=lambda char:t.count(char)/max(1,len(a))
    forms=[np.log1p(len(a)),np.log1p(len(parts)),len(a)/max(1,len(parts)),sum(map(len,a))/max(1,len(a)),len(set(a))/max(1,len(a))]
    olap=[]
    for n in (1,2,3):
        base=ngrams(a,n);nall=sum(base.values());rep=1-len(base)/max(1,nall) if nall else 0
        if n>1:forms.append(rep)
        if n<3:
            f=[]
            for ref in refs:
                b=ngrams(tokens(ref),n);f.append(2*sum((base&b).values())/max(1,nall+sum(b.values())))
            olap.extend((max(f),np.mean(f)))
    forms.extend((1-len(set(map(tuple,parts)))/max(1,len(parts)),np.mean(v) if v else 0,count(','),count(':'),count("'"),count('?'),count('!')))
    return list(map(float,forms)),list(map(float,olap))


def run():
    data=material();g,checks=score_geval(data);out={}
    for topic,sys,t,ref in data:
        k=topic+'/'+sys;v,extra=lexical(t,ref);out[k]={'form':v,'reference_overlap':extra,'source_aware_gpt4':g[topic,sys]}
    path=OUT/'control-forecasts.json';path.write_text(json.dumps(out,separators=(',',':'),sort_keys=True)+'\n');checks.update({'forecasts':digest(path),'references_per_system':sorted({len(v[3]) for v in data}),'human_evaluations_accessed':False,'wording':'Original SummEval raw strings for all CogLLM estimates. Original G-Eval uses the source article plus a separately detokenized and cased version.'});(OUT/'control-seal.json').write_text(json.dumps(checks,indent=2)+'\n');print(checks)


if __name__=='__main__':run()
