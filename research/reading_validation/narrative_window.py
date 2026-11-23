"""Locate co-occurring plot-question changes and evaluate the tolerance actually needed."""
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
from research.backbone_transfer.extract import digest, write_json
from research.reading_validation.narrative_agreement import OUT, changes, direction, read_stories

PLAN=Path(__file__).with_name("narrative-window-protocol.md")


def aligned(left,right,tolerance):
    @lru_cache(None)
    def go(i,j):
        if i==len(left) or j==len(right):return (0,0,())
        if left[i]<right[j]-tolerance:return go(i+1,j)
        if right[j]<left[i]-tolerance:return go(i,j+1)
        n,c,pairs=go(i+1,j+1)
        match=(n+1,c-abs(int(left[i]-right[j])),((int(left[i]),int(right[j])),)+pairs)
        return max((match,go(i+1,j),go(i,j+1)),key=lambda v:v[:2])
    return list(go(0,0)[2])


def cardinality(a,b,tolerance):
    i=j=n=0
    while i<len(a) and j<len(b):
        if a[i]<b[j]-tolerance:i+=1
        elif b[j]<a[i]-tolerance:j+=1
        else:n+=1;i+=1;j+=1
    return n


def main():
    stories=read_stories();rng=np.random.default_rng(943355)
    observed={};chance={};peaks=[]
    for title,(teams,forecast) in stories.items():
        dirs={team:direction(changes(b)) for team,b in teams.items()}
        n=len(dirs["AA"]);f=np.diff(forecast["count"])
        shifts=rng.integers(10,n-9,size=4000)
        num_blocks=(n+29)//30
        rotation=np.stack([rng.integers(0,min(30,n-30*k),size=4000) for k in range(num_blocks)],axis=1)
        for name,sign in (("rise",1),("decline",-1)):
            a=np.flatnonzero(dirs["AA"]==sign);b=np.flatnonzero(dirs["JE"]==sign)
            for tolerance in (0,1,2):
                pairs=aligned(a,b,tolerance)
                if len(pairs)!=cardinality(a,b,tolerance):raise ArithmeticError("Matching inconsistent")
                label=f"{title}.{name}.{tolerance}"
                observed[label]=dict(AA=len(a),JE=len(b),matched=len(pairs),F1=2*len(pairs)/(len(a)+len(b)),
                        original_lines=[[x+2,y+2] for x,y in pairs])
                counts=np.empty((4000,2),int)
                block=b//30; widths=np.minimum(30,n-30*block)
                for i,shift in enumerate(shifts):
                    whole=np.sort((b+shift)%n)
                    local=np.sort(30*block+(b%30+rotation[i,block])%widths)
                    counts[i]=[cardinality(a,whole,tolerance),cardinality(a,local,tolerance)]
                chance[label]=counts
        for team,b in teams.items():
            d=changes(b);k=int(np.argmin(d[:,1]))
            peaks.append(dict(story=title,team=team,line=k+2,observed=d[k].tolist(),
                best_change=[float(f[max(0,k-tol):min(n,k+tol+1)].min()) for tol in (0,1,2)]))
    totals={}
    for name in ("rise","decline"):
        for t in (0,1,2):
            keys=[k for k in observed if k.endswith(f".{name}.{t}")]
            count=sum(observed[k]["matched"] for k in keys)
            opportunity=sum(observed[k]["AA"]+observed[k]["JE"] for k in keys)
            sims=sum(chance[k] for k in keys)
            totals[f"{name}.{t}"]=dict(pairs=count,event_labels=opportunity,F1=2*count/opportunity,
                nulls={k:dict(expected_matches=float(sims[:,j].mean()),p=float((1+(sims[:,j]>=count).sum())/(1+len(sims))))
                       for j,k in enumerate(("whole_story","within_blocks"))})
    write_json(OUT/"window-scores.json",dict(protocol=digest(PLAN),references=digest(OUT/"scores.json"),
       results_by_story=observed,pooled=totals,peak_windows=peaks))
    print(json.dumps(totals))


if __name__=="__main__":
    main()
