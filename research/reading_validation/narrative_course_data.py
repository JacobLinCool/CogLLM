"""Recover original German stories and separately certify human ratings against their wording."""

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import re
import unicodedata

import numpy as np

from research.reading_validation.memory_data import BASE, h, sha
from research.semantic_tracking.resolution_prepare import public_source_manifest, read_sheet


AUTHOR=BASE.parent/"semantic-tracking/resolution-osf"
WORK=BASE/"narrative-course"
OUT=Path("research/results/reading-validation-narrative-course")
PLAN=Path(__file__).with_name("narrative-course-protocol.md")
STORIES=("Auftrag","Katze","Thomas","Verkehrsunfall")


def fold(word):
    return "".join(c for c in unicodedata.normalize("NFKC",word).casefold() if c.isalnum())


def original_path(story,team):
    name=f"PITQ_{story}_{team}_{team}.xlsx"
    path=BASE/f"onea-unfall-{team}.xlsx" if story=="Verkehrsunfall" else AUTHOR/name
    return name,path


def book(path):
    rows=read_sheet(path)
    questions=[column for column,value in rows[1].items() if column!="A" and value.isdecimal()]
    if sorted(int(rows[1][c]) for c in questions)!=list(range(1,len(questions)+1)):
        raise ValueError("Question identifiers are incomplete")
    text=[r for n,r in sorted(rows.items()) if n>2 and r.get("A") and not r["A"].startswith("Wichtigkeit (")]
    labels={"":0,"offen":1,"?offen":2}
    states=np.array([[labels[r.get(c,"")] for c in questions] for r in text],dtype=np.uint8)
    return [r["A"] for r in text],states


def align(reference,target):
    """Map target lexical words only through exact, unique contiguous contexts."""
    matches=defaultdict(set)
    for length in (7,5):
        def index(words):
            result=defaultdict(list)
            for j in range(len(words)-length+1):
                result[tuple(words[j:j+length])].append(j)
            return result
        a,b=index(reference),index(target)
        for key,starts in a.items():
            if len(starts)==1 and len(b.get(key,()))==1:
                for d in range(length):
                    matches[b[key][0]+d].add(starts[0]+d)
    accepted={i:next(iter(ref)) for i,ref in matches.items() if len(ref)==1}
    ordered=[accepted[i] for i in sorted(accepted)]
    if ordered!=sorted(set(ordered)) or any(reference[j]!=target[i] for i,j in accepted.items()):
        raise ValueError("Word certificate disagrees with the monotone original")
    return accepted


def ratings(file,original,word_coords):
    words={}
    records={}
    participants={}
    for r in csv.DictReader(file.open(encoding="utf-8-sig",newline="")):
        # Ignore all published demographics, free text and the expanded expert annotation columns.
        j=int(r["number"])
        item=(fold(r["text"]),int(r["screen_number"]),int(r["paragraph"]))
        if words.setdefault(j,item)!=item:
            raise ValueError("Original visible word or screen disagrees across readers")
        key=participants.setdefault(r["participant"],len(participants)),j
        if key in records:
            raise ValueError("Duplicated reader/word rating")
        value=r["z_suspense"]
        if value not in ("","NA"):
            records[key]=float(value)
    order=sorted(k for k,v in words.items() if v[0])
    forms=[fold(original[a:b]) for a,b,_ in word_coords]
    matched=align(forms,[words[k][0] for k in order])
    index={order[i]:j for i,j in matched.items()}
    measurements=np.full((len(participants),len(forms)),np.nan)
    screens=np.full(len(forms),-1,dtype=np.int16)
    for k,j in index.items():
        screens[j]=words[k][1]
    for (person,k),value in records.items():
        if k in index:
            measurements[person,index[k]]=value
    if len(index)<.85*len(order) or not np.isfinite(measurements).any(axis=1).all():
        raise ValueError(f"Insufficient alignment: {len(index)}/{len(order)}")
    return measurements,screens,dict(readers=len(participants),published_lexical_positions=len(order),
            certified_words=len(index),source_words=len(forms),screens=len(set(screens)-{-1}),
            observed_reader_word_values=int(np.isfinite(measurements).sum()))


def prepare():
    from tokenizers import Tokenizer
    from research.reading_validation.relay_data import REV
    tokenizer=Tokenizer.from_file(str(Path.home()/".cache/huggingface/hub/models--Qwen--Qwen3.5-2B/snapshots"/REV/"tokenizer.json"))
    WORK.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)
    manifest=public_source_manifest(AUTHOR)
    publications,documents,inventory,tables,cache={},[],{}, {},{}
    for story in STORIES:
        teams=("AA",) if story=="Katze" else ("AA","JE")
        lines=None
        for team in teams:
            name,path=original_path(story,team)
            if sha(path)!=manifest[name]["sha256"]:
                raise ValueError(f"Unverified author workbook {name}")
            publications[name]=manifest[name]["sha256"]
            text,states=book(path)
            if lines is not None and text!=lines:
                raise ValueError(f"Two original texts differ for {story}")
            lines=text
            tables[f"{story}.{team}"]=states
        article="\n".join(lines)
        bounds=[]
        acc=0
        for line in lines:
            bounds.append((acc,acc+len(line)))
            acc+=len(line)+1
        coords=[(*m.span(),j) for j,(a,b) in enumerate(bounds) for m in re.compile(r"\S+").finditer(article,a,b) if fold(m.group())]
        alias="Unfall" if story=="Verkehrsunfall" else story
        name=f"{alias}_suspense_data.csv"
        path=AUTHOR/name if alias=="Unfall" else BASE/f"onea-{alias}.csv"
        if sha(path)!=manifest[name]["sha256"]:
            raise ValueError(f"Unverified participant release {name}")
        publications[name]=manifest[name]["sha256"]
        measurements,screen,details=ratings(path,article,coords)
        tables[f"{story}.rating"]=measurements
        tables[f"{story}.screen"]=screen
        tables[f"{story}.words"]=np.array(coords)
        code=h(article)
        documents.append(dict(id=code,text=article,spans=bounds))
        native=tokenizer.encode(article,add_special_tokens=False)
        inventory[story]=dict(**details,id=code,lines=len(lines),tokens=len(native.ids),
                              questions={t:tables[f"{story}.{t}"].shape[1] for t in teams})
    packet=WORK/"packet.jsonl"; outcomes=WORK/"response.npz"
    packet.write_text("".join(json.dumps(v,ensure_ascii=False)+"\n" for v in documents))
    np.savez_compressed(outcomes,**tables)
    seal=dict(protocol=sha(PLAN),source="https://osf.io/m524v/",license="CC BY-NC-SA 4.0",
              files=publications,packet=sha(packet),outcomes=sha(outcomes),tokenizer_revision=REV,articles=inventory)
    (OUT/"input-seal.json").write_text(json.dumps(seal,indent=2)+"\n")
    print(json.dumps(inventory,indent=2))


if __name__=="__main__":
    prepare()
