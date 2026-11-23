"""Grade all context-held word forecasts, fitting no parameters on test readers or items."""

from collections import defaultdict
from itertools import combinations
import json

import numpy as np

from research.reading_validation.expectation_data import OUT
from research.reading_validation.expectation_measure import Materials,Observations
from research.reading_validation.memory_data import sha
from research.reading_validation.retrieval_source_score import summarize


def evaluate(held,observed,assignment,study,conditions,methods,*,unexpected=False):
    scores=defaultdict(lambda:defaultdict(list)); by_group=defaultdict(list)
    for k,y in observed.items():
        if k not in held or (unexpected and conditions[study,k]=="BC"):
            continue
        family,split=assignment[study,k]
        if split!="test":
            raise ValueError("Source material reached the held grade")
        by_group[family].append((y,np.array(held[k]),conditions.get((study,k))))
        for metric,z in (("mae",abs(np.array(held[k])-y)),("mse",(np.array(held[k])-y)**2)):
            scores[metric][family].append(dict(zip(methods,z,strict=True)))
    for family,rows in by_group.items():
        for (y1,p1,_),(y2,p2,_) in combinations(rows,2):
            if y1==y2:
                continue
            z=np.sign(y1-y2)*(p1-p2)
            scores["within_context_order"][family].append(dict(zip(methods,(z>0).astype(float)+.5*(z==0),strict=True)))
    result={k:summarize(v,59101,focus="expectation") for k,v in scores.items() if v}
    return result


def run():
    dest=OUT/"forecast"
    receipt=json.loads((dest/"seal.json").read_bytes())
    if any(sha(dest/n)!=v for n,v in receipt["files"].items()):
        raise ValueError("A prediction or fit has changed after lock")
    forecasts=json.loads((dest/"predictions.json").read_bytes());mat=Materials()
    selection=defaultdict(set)
    for name,r in forecasts.items():
        study,task,mode=name.split("|")
        selection[study].update(r["held"])
    observed=Observations(mat,selection)
    results={}
    for name,record in forecasts.items():
        study,task,mode=name.split("|")
        y=observed.responses(study,task,record["beta"])
        methods=record["methods"];held=record["held"]
        row=evaluate(held,y,mat.assignment,study,observed.condition,methods)
        if study=="michaelov":
            row["unexpected"]=evaluate(held,y,mat.assignment,study,observed.condition,methods,unexpected=True)
        results[name]=row
    readers={}
    for study,keys in selection.items():
        participants={u for k in keys for u in observed.readers.get((study,k),{})}
        readers[study]=dict(people=len(participants),trials=sum(len(observed.readers.get((study,k),{})) for k in keys),
                            source_records=observed.raw[study],articles=len(keys))
    output=dict(receipt=sha(dest/"seal.json"),observed=readers,methods=results)
    (OUT/"scores.json").write_text(json.dumps(output,indent=2,allow_nan=False)+"\n")
    for k,v in results.items():
        if "within_context_order" in v:
            print(k,{n:round(x["estimate"],4) for n,x in v["within_context_order"]["methods"].items()})


if __name__=="__main__":
    run()
