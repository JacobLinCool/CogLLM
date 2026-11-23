"""Reconstruct the published three-arm narrative interventions without model-written passages."""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re

from research.backbone_transfer.extract import digest,write_json
from research.reading_validation.transition_goal_access_data import STOP,WORD

ROOT=Path(".cache/cogllm/reading-validation")
OUT=Path("research/results/reading-validation-character-access")
PLAN=Path(__file__).with_name("character-access-protocol.md")
SOURCE=ROOT/"character-names-official-lines.txt"
INPUT=ROOT/"character-names-presentations.jsonl"


def words(text):
    return [m.group().casefold() for m in WORD.finditer(text)]


def controls(target, intro, first, last, new, neutral):
    original=words(intro)
    if target not in original:
        raise ValueError("Published probe absent from the original introduction.")
    n=Counter(original)
    position={w:i for i,w in enumerate(original)}
    banned=set(words(" ".join((first,last,new,neutral))))|STOP|{target}
    eligible=set(original)-banned
    ranking=sorted(eligible,key=lambda w:(abs(n[target]-n[w]),abs(position[target]-position[w]),abs(len(target)-len(w)),w))
    if len(ranking)<3:
        raise ValueError("Could not identify the three original control words.")
    return ranking[:3]


def name_stories():
    if digest(SOURCE)!="4b2818d6da1195e8eb9e58bc92addef26e84c51ec3d16eb6c4c2af0240f48c31":
        raise ValueError("The author's word and probe source has changed.")
    rows=[r.split("\t",1) for r in SOURCE.read_text().splitlines()]
    if [int(i) for i,_ in rows]!=list(range(1529)):
        raise ValueError("Original source pages are incomplete.")
    lines=[v.strip() for _,v in rows]
    lines=lines[lines.index("Experimental stories")+1:lines.index("Fillers")]
    starts=[i for i,v in enumerate(lines) if re.match(r"^\d+\.(?: |$)",v)]
    if [int(lines[i].split(".")[0]) for i in starts]!=list(range(1,33)):
        raise ValueError("A complete factorial story is missing.")
    stories={}
    for ident,(begin,end) in enumerate(zip(starts,[*starts[1:],len(lines)],strict=True),1):
        parts=[v for v in lines[begin+1:end] if v and not re.fullmatch(r"Page \d+",v)]
        parts_lower=[v.lower() for v in parts]
        def between(left,right):
            a,b=parts_lower.index(left),parts_lower.index(right)
            if a>=b:raise ValueError("Experimental landmarks are out of order.")
            return " ".join(parts[a+1:b])
        target,=[v.split(":")[1].strip().lower() for v in parts if v.startswith("TEST WORD:")]
        stories[ident]=dict(intro=between("introduction","transition statement 1"),
             first=between("transition statement 1","test 1"),last=between("transition statement 2","test 2"),
             rementioned=between("old","new"),new=between("new","neither"),neutral=between("neither","transition statement 2"),
             target=target)
    return stories


def package(stories, source, input_path, sealed):
    rows=[];families={};old={}
    for r in map(json.loads,(ROOT/"goal-access-stimuli.jsonl").read_text().splitlines()):
        w=words(r["text"])
        for i in range(len(w)-19):old.setdefault(tuple(w[i:i+20]),set()).add(r["id"])
    for ident,parts in stories.items():
        intro,first,last,target=[parts[k] for k in ("intro","first","last","target")]
        control=controls(target,intro,first,last,parts["new"],parts["neutral"])
        common=intro+" "+first
        meta=dict(probe=target,control_words=control,variants={})
        for condition in ("rementioned","new","neutral"):
            middle=parts[condition]
            text=common+"\n\n"+middle+" "+last
            uid=sha256(text.encode()).hexdigest()
            tokens=words(text)
            overlap=any(tuple(tokens[i:i+20]) in old for i in range(len(tokens)-19))
            rows.append(dict(id=uid,text=text,prior_end=len(intro),first_start=len(intro)+1,first_end=len(common),
                             last_start=len(text)-len(last),last_end=len(text),targets=[target,*control]))
            meta["variants"][condition]=dict(id=uid,intervening_primary_mentions=words(middle).count(target),
                       manipulation_word_count=len(words(middle)),shared_20_lexical_words_with_old_goal_study=overlap)
        families[str(ident)]=meta
    if len({r["id"] for r in rows})!=len(rows):
        raise ValueError("Two expected article presentations have identical bytes.")
    OUT.mkdir(parents=True,exist_ok=True)
    input_path.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows))
    write_json(sealed,dict(stimulus_source=digest(source),protocol=digest(PLAN),input=digest(input_path),
               families=families,conditions=["rementioned","new","neutral"],presentations=len(rows)))


if __name__=="__main__":
    package(name_stories(),SOURCE,INPUT,OUT/"name-input.json")
