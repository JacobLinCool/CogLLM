"""Preserve the measurement unit, reading context, and split of every published observation."""

from collections import Counter,defaultdict
import csv
from functools import lru_cache
import json
from math import isfinite, log
import re

import numpy as np

from research.reading_validation.expectation_data import WORK, OUT, PUBLIC
from research.reading_validation.memory_data import h,load_rows,sha


def stem(text, start):
    return " ".join(re.findall(r"[^\W_]+",text[:start].casefold()))


class Materials:
    def __init__(self):
        self.seal=json.loads((OUT/"input-seal.json").read_bytes())
        if sha(WORK/"opportunities.json")!=self.seal["opportunities"] or sha(WORK/"stimuli.jsonl")!=self.seal["text"]:
            raise ValueError("Expectation materials changed")
        self.text={row["id"]:row["text"] for row in load_rows(WORK/"stimuli.jsonl")}
        groups=json.loads((WORK/"opportunities.json").read_bytes())
        self.rows=defaultdict(dict);self.released={}
        for collection in PUBLIC:
            for row in groups[collection]:
                study=row["study"];key=row["id"];family=row["family"]
                if key in self.rows[study] and self.rows[study][key]["family"]!=family:
                    raise ValueError("One sentence is attributed to multiple source families")
                self.rows[study][key]={**row,"stem":stem(self.text[key],row["span"][0]),"collection":collection}
                self.released[(collection,study,row["source_id"])]=key
        self.assignment={}
        for study,rows in self.rows.items():
            family={r["family"]:r["family"] for r in rows.values()}
            def root(i):
                while i!=family[i]:
                    family[i]=family[family[i]]
                    i=family[i]
                return i
            used={}
            for key,row in sorted(rows.items()):
                prefix=row["stem"]
                if len(prefix.split())<3:
                    continue
                a=root(row["family"]);b=root(used.setdefault(prefix,a))
                family[max(a,b)]=min(a,b)
            clusters=sorted({root(i) for i in family},key=lambda g:h("cogllm-expectation-20261122."+study+"."+g))
            order={v:i for i,v in enumerate(clusters)}
            for key,row in rows.items():
                data_group=root(row["family"])
                i=order[data_group]
                role="train" if i<int(.6*len(clusters)) else "dev" if i<int(.75*len(clusters)) else "test"
                self.assignment[(study,key)]=(data_group,role)

    def other_materials(self,study,case="test"):
        """Prevent leakage even when different publications reuse an original item."""
        target=self.rows[study]
        ids={k for k in target}
        stems={r["stem"] for r in target.values() if len(r["stem"].split())>=3}
        sources={}
        for name,rows in self.rows.items():
            if name==study or name=="brothers":
                continue
            hit={row["family"] for k,row in rows.items() if k in ids or row["stem"] in stems}
            keep={k for k,row in rows.items() if row["family"] not in hit and self.assignment[name,k][1]=="train"}
            if keep:
                sources[name]=keep
        return sources


def available(value):
    try:
        result=float(value)
    except (ValueError,TypeError):
        return None
    return result if isfinite(result) else None


class Observations:
    def __init__(self,mat,selection):
        self.ratings={};self.condition={};self.readers=defaultdict(dict);self.raw=Counter()
        self.selection=selection
        for collection,(_,name,_,_) in PUBLIC.items():
            if sha(WORK/name)!=mat.seal["files"][name]:
                raise ValueError("Public participant source changed")
            if not any(mat.rows[s][next(iter(v))]["collection"]==collection for s,v in selection.items() if v):
                continue
            data=defaultdict(list)
            with (WORK/name).open() as f:
                for r in csv.DictReader(f,delimiter="\t"):
                    study=r.get("dataset",collection)
                    if study not in selection:
                        continue
                    k=mat.released.get((collection,study,h(r["FullSentence"])))
                    if k not in selection[study]:
                        continue
                    if collection=="michaelov":
                        rating=available(r["PlausibilityJudgement"])
                        if rating is None or not 1<=rating<=5 or self.ratings.setdefault((study,k),rating)!=rating:
                            raise ValueError("Plausibility norm varies across repeated copies of the same material")
                        self.condition[study,k]=r["Condition"]
                        eeg=available(r["N400"])
                        if eeg is not None:
                            data[(study,k,r["Subject"])].append((eeg,0.,r["Electrode"]))
                    elif collection=="brothers":
                        rt=available(r["SUM_3RT_trimmed"])
                        self.condition[study,k]=r["condition"]
                        if rt is not None and rt>0:
                            data[(study,k,r["SUB"])].append((log(rt),0.,"trial"))
                    else:
                        val,bline=available(r["n400"]),available(r["bline"])
                        if val is not None and bline is not None:
                            data[(study,k,r["Subject"])].append((val,bline,"trial"))
            for (study,k,subject),obs in data.items():
                if collection=="michaelov" and len({x[2] for x in obs})!=len(obs):
                    raise ValueError("A reader's same electrode occurs twice at the same word")
                self.raw[study]+=len(obs)
                self.readers[study,k][subject]=np.mean([r[:2] for r in obs],axis=0)
        self.beta={}

    def slopes(self):
        """Estimate a source-only physiological baseline using variation within each stimulus."""
        xx=Counter();xy=Counter()
        for (s,k),rows in self.readers.items():
            data=np.vstack(list(rows.values()))
            z=data-data.mean(0)
            xx[s]+=float(z[:,1]@z[:,1])
            xy[s]+=float(z[:,1]@z[:,0])
        return {s:xy[s]/v if v else 0. for s,v in xx.items()}

    def responses(self,study,task,beta):
        if task=="plausibility":
            return {k:v for (s,k),v in self.ratings.items() if s==study}
        result={}
        for (s,k),rows in self.readers.items():
            if s==study:
                values=np.stack(list(rows.values()))
                result[k]=float(np.mean(values[:,0]-beta*values[:,1]))
        return result
