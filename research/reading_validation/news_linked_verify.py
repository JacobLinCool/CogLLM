"""Reconstruct human numerator/denominator and scores independently of the training loader."""

from collections import defaultdict
from itertools import product
import ast
import csv
import json
from math import log

import numpy as np

from research.reading_validation.memory_data import load_rows, sha
from research.reading_validation.news_linked_data import OUT,RAW,TRACK
from research.reading_validation.retrieval_data import DATA


def convert(text,a,b):
    # Independently implement the published UTF-16 unit convention using the codec.
    blob=text.encode("utf-16-le")
    if not (isinstance(a,int) and isinstance(b,int) and 0<=a<b<=len(blob)//2):
        return None
    try:
        left=len(blob[:2*a].decode("utf-16-le"))
        piece=blob[2*a:2*b].decode("utf-16-le")
    except UnicodeDecodeError:
        return None
    if not piece.strip():
        return None
    return left+len(piece)-len(piece.lstrip()),left+len(piece.rstrip())


def overlap(a,b):
    return max(a[0],b[0])<min(a[1],b[1])


def decisions(text,words,sections,submission):
    events=[]
    for kind,_,_,start,end,source in submission:
        if kind!="Self-contradiction":
            continue
        mark=convert(text,start,end)
        if mark is None:
            return
        ranges=[convert(text,a,b) for a,b in zip(source[::2],source[1::2])] if source else []
        allowed=None
        if source and len(source)%2==0 and all(v is not None and v[1]<=mark[0] for v in ranges):
            allowed={j for j,r in enumerate(sections) if any(overlap(s,r) for s in ranges)}
        events.append((mark,allowed))
    for i,w in enumerate(words):
        j=next((k for k,r in enumerate(sections) if overlap(r,w)),None)
        if j is None or j<2:
            continue
        slots=[None if old is None else {k for k in old if k<j} for span,old in events if overlap(span,w)]
        if any(not v for v in slots):
            continue
        yield i,set().union(*slots)


def check_recursion():
    from research.reading_validation.news_linked_forecast import curves
    units=np.array([0,1,1,2,2],dtype=int)
    x=np.array([[0],[.4],[.1],[-.9],[1.2]])
    data=np.zeros((units.sum(),10));data[:,0]=np.linspace(-2,1,len(data))
    def binary(b,s):
        return dict(mu=[0.],sd=[1.],w=[b,s])
    weights={k:binary(b,s) for k,b,s in (("active",-.3,.9),("enter",-.1,.8),("exit",.6,-.5))}
    weights["source"]=dict(mu=[0.]*10,sd=[1.]*10,w=[1.]+[0.]*9)
    _,temporal=curves(units,x,data,weights,"lexical")
    known=np.array([1.,0.,0.]);err=0.;pos=0
    for t,m in enumerate(units[1:],1):
        enter=1/(1+np.exp(.1-.8*x[t,0]))
        leave=1/(1+np.exp(-.6+.5*x[t,0]))
        als=np.exp(data[pos:pos+m,0]);als/=als.sum();pos+=m
        trans=np.zeros((3,3))
        for last,next_state in product(range(3),repeat=2):
            released=1 if not last else leave
            new=(1-enter) if next_state==0 else (enter*als[next_state-1] if next_state<=m else 0)
            trans[last,next_state]=released*new+(1-released)*(last==next_state)
        known=known@trans
        err=max(err,float(abs(known-temporal[t]).max()))
    return err


def run():
    source={f"scarecrow-{r['id']}":r for r in csv.DictReader(RAW.open(encoding="utf8",newline=""))}
    ids=json.loads((DATA/"mapping.json").read_bytes())["news"]
    passages={r["id"]:r for r in load_rows(DATA/"packet.jsonl")}
    scores=json.loads((OUT/"scores.json").read_bytes())
    likelihood=defaultdict(list);total=positive=0
    with np.load(OUT/"forecast/predictions.npz",allow_pickle=False) as stored:
        for row in load_rows(TRACK/"scarecrow-detection/prepared/layout.jsonl"):
            if row["split"]!="test":
                continue
            k=row["id"];r=source[k];p=stored[k+".semantic.temporal"]
            section=passages[ids[k]]["spans"]
            observations=[(i,g) for reader in ast.literal_eval(r["responses"])
                          for i,g in decisions(r["generation"],row["words"],section,reader)]
            if observations:
                total+=len(observations);positive+=sum(bool(g) for _,g in observations)
                logs=[-log(max(1e-12,sum(p[i,j+1] for j in g) if g else p[i,0])) for i,g in observations]
                likelihood[row["group_id"]].append(float(np.mean(logs)))
    estimate=np.mean([np.mean(items) for items in likelihood.values()])
    reference=scores["group_equal"]["joint_nll"]["methods"]["semantic.temporal"]["estimate"]
    delta=float(abs(estimate-reference));recursion=check_recursion()
    if (total!=scores["inventory"]["eligible_reader_words"] or positive!=scores["inventory"]["positive"]
            or delta>1e-10 or recursion>1e-10):
        raise ArithmeticError(f"Human counting, published joint score or filter disagree: {total}, {positive}, {delta}, {recursion}")
    result=dict(independent_utf16_reconstruction=True,reader_words=total,marked=positive,
                contexts=len(likelihood),joint_nll_abs_difference=delta,explicit_transition_error=recursion,
                input=sha(RAW),published=sha(OUT/"scores.json"))
    (OUT/"independent-check.json").write_text(json.dumps(result,indent=2)+"\n")
    print(result)


if __name__=="__main__":
    run()
