"""Recover source sets, masking ambiguous reading opportunities at the respondent level."""

from collections import Counter,defaultdict
import ast
import csv
import json
from pathlib import Path

import numpy as np

from research.reading_validation.memory_data import load_rows,sha
from research.reading_validation.retrieval_data import DATA,TRACK,RESULT
from research.reading_validation.retrieval_features import Corpus,SURFACE
from research.reading_validation.retrieval_fit import Readout
from research.semantic_tracking import source_prepare as original
from research.semantic_tracking.scarecrow_detection_prepare import overlaps
from research.semantic_tracking.scarecrow_detection_score import word_features


OUT=Path("research/results/reading-validation-news-linked")
WORK=DATA.parent/"news-linked"
PLAN=Path(__file__).with_name("news-linked-protocol.md")
RAW=TRACK/"scarecrow/grouped_data.pinned.csv"


def reports(text,spans,words,units,responses):
    mapping=original.utf16_to_characters(text)
    result=[]
    for submission in responses:
        uncertain=[];known=[]
        unknowable=False
        for category,_,severity,a,b,source in submission:
            if category!="Self-contradiction":
                continue
            target=original.convert_span(text,mapping,a,b)
            if target is None:
                unknowable=True
                continue
            if not source or len(source)%2:
                uncertain.append(target)
                continue
            earlier=[original.convert_span(text,mapping,l,r) for l,r in zip(source[::2],source[1::2],strict=True)]
            if any(s is None or s[1]>target[0] for s in earlier):
                uncertain.append(target)
            else:
                source_units={j for j,span in enumerate(spans) if any(overlaps(span,s) for s in earlier)}
                known.append((target,source_units))
        labels=[]
        for w,t in zip(words,units,strict=True):
            if unknowable or any(overlaps(w,s) for s in uncertain):
                labels.append(None)
                continue
            choices=set()
            unavailable=False
            for s,older in known:
                if overlaps(s,w):
                    legal={j for j in older if j<t}
                    if not legal:
                        unavailable=True
                    choices.update(legal)
            labels.append(None if unavailable else sorted(choices))
        result.append(labels)
    return result


def transition_counts(labels):
    length=len(labels[0]);counts=np.zeros((length,3,2),dtype=np.int16)
    targets=defaultdict(Counter)
    for reader in labels:
        if len(reader)!=length:
            raise ValueError("Rater trajectory has unequal geometry")
        for i,now in enumerate(reader):
            if now is None:
                continue
            active=bool(now)
            counts[i,0]+=(active,1)
            if active:
                targets[i][tuple(now)]+=1
            previous=reader[i-1] if i else []
            if previous is None:
                continue
            was=bool(previous)
            closed=was and (not active or set(previous).isdisjoint(now))
            if was:
                counts[i,2]+=(closed,1)
            if not was or closed:
                counts[i,1]+=(active,1)
    return counts,targets


def features(page,layout,heads):
    indices={tuple(span):i for i,span in enumerate(page.coords.tolist())}
    ix=np.array([indices[tuple(s)] for s in layout])
    units=page.unit[ix]
    shape=word_features(page.text,layout)
    state=page.state[ix];delta=np.vstack((np.zeros_like(state[:1]),np.diff(state,axis=0)))
    signal=page.signals[ix]
    shape=np.column_stack((shape,np.log1p(units),[np.searchsorted(page.sections[j],i)
                                               for i,j in zip(ix,units,strict=True)]))
    native=np.column_stack((shape,signal[:,0],signal[:,1]/signal[:,2],state,delta)).astype("f")
    bank=[]
    relprior=[]
    for j,section in enumerate(page.sections):
        if not j:
            for i in ix[units==0]:
                bank.append(np.zeros((0,22),dtype="f"));relprior.append(np.empty(0))
            continue
        cube=page.pairs(j)
        _,lexical_r,lexical=heads["lexical"](cube)
        _,semantic_r,semantic=heads["semantic"](cube)
        # Columns through 9 depend only on text form and the pretrained lexical source scorer.
        block=np.concatenate((cube[:,:,:len(SURFACE)],np.log(lexical.clip(1e-12))[:,:,None],
                              np.log(lexical_r.clip(1e-12))[:,:,None],cube[:,:,8:18],
                              np.log(semantic.clip(1e-12))[:,:,None],np.log(semantic_r.clip(1e-12))[:,:,None]),axis=2)
        for i in ix[units==j]:
            site=int(np.searchsorted(section,i))
            bank.append(block[site].astype("f"))
            relprior.append(semantic[site].astype("f"))
    if len(bank)!=len(shape) or any(arr.shape[0]!=j for arr,j in zip(bank,units,strict=True)):
        raise ValueError("A word's candidate set includes an invisible source sentence")
    return ix,units,shape.astype("f"),native,bank,relprior


def prepare(archive):
    if sha(RAW)!=original.CSV_SHA256:
        raise ValueError("Public annotation release changed")
    geometry=load_rows(TRACK/"scarecrow-detection/prepared/layout.jsonl")
    mapping=json.loads((DATA/"mapping.json").read_bytes())["news"]
    weights=json.loads((RESULT/"fit/weights.json").read_bytes())
    heads={k:Readout(v) for k,v in weights.items()}
    corpus=Corpus(archive)
    OUT.mkdir(exist_ok=True);WORK.mkdir(exist_ok=True)
    raw={f"scarecrow-{r['id']}":r for r in csv.DictReader(RAW.open(encoding="utf-8",newline="")) if r["model"]!="human"}
    totals=defaultdict(Counter)
    files={}
    for side in ("train","test"):
        seq=[];obs=[]
        for item in geometry:
            if item["split"]!=side:
                continue
            k=item["id"];code=mapping[k];row=corpus.rows[code];page=corpus.page(code)
            if raw[k]["generation"]!=row["text"]:
                raise ValueError("The response source differs from the encoded continuation")
            ix,units,x0,x1,opts,prior=features(page,item["words"],heads)
            label=reports(page.text,row["spans"],item["words"],units,ast.literal_eval(raw[k]["responses"]))
            counts,targets=transition_counts(label)
            summary=totals[side]
            summary["documents"]+=1;summary["valid_reader_words"]+=int(counts[units>=2,0,1].sum())
            summary["positive_reader_words"]+=int(counts[units>=2,0,0].sum())
            summary["total_reviewers_on_continuations"]+=len(label)
            # Concatenation preserves a strictly varying number of earlier alternatives.
            block=np.vstack([a for a in opts if len(a)]) if units.max()>0 else np.empty((0,22),dtype="f")
            seq.append(dict(id=k,group=item["group_id"],start=len(seq),words=ix.tolist()))
            pointer={str(i):[[list(g),v] for g,v in cs.items()] for i,cs in targets.items()}
            obs.append(dict(id=k,targets=pointer))
            np.savez_compressed(WORK/(k+".npz"),units=units,x0=x0,x1=x1,source=block,counts=counts,
                                frozen=np.concatenate(prior))
        for label,obj in (("sequence",seq),("outcomes",obs)):
            path=WORK/(side+"-"+label+".json")
            path.write_text(json.dumps(obj,separators=(",",":"))+"\n");files[path.name]=sha(path)
    disks={p.name:sha(p) for p in WORK.glob("scarecrow-*.npz")}
    seal=dict(raw=sha(RAW),registration=sha(PLAN),archive=sha(archive/"receipt.json"),
              source_weights=sha(RESULT/"fit/weights.json"),parts=files,features=disks,inventory=totals)
    (OUT/"input-seal.json").write_text(json.dumps(seal,indent=2,default=dict)+"\n")
    print(json.dumps(totals,default=dict))


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("archive",type=Path);prepare(p.parse_args().archive)
