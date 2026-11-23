"""Recover the original 32 wearable/handled object probes and the three intact character manipulations."""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re

from research.reading_validation.character_access_data import ROOT, OUT, PLAN, controls, words
from research.reading_validation.transition_goal_access_data import STOP, WORD
from research.reading_validation.transition_score import dump,sha

SOURCE=ROOT/"character-objects-official-lines.txt"
ARM=Path(__file__).with_name("character-object-access-addendum.md")
INPUT=ROOT/"character-objects-presentations.jsonl"
CLOTHING=re.compile(r"t-shirt|"+WORD.pattern,re.IGNORECASE|re.UNICODE)


def read_stories():
    if sha(SOURCE)!="f551b67d043a5573b9e8fe23888a95e31c2d750ca3f617fa16bb3bab65a60601":
        raise ValueError("The complete PDF source changed")
    raw=[r.split("\t",1) for r in SOURCE.read_text().splitlines()]
    if [int(k) for k,_ in raw]!=list(range(1326)):raise ValueError("Incomplete original PDF source")
    lines=[v.strip() for _,v in raw]
    lines=lines[lines.index("Experimental stories")+1:lines.index("Fillers")]
    starts=[i for i,v in enumerate(lines) if re.match(r"^\d+\.(?: |$)",v)]
    if [int(lines[i].split(".")[0]) for i in starts]!=list(range(1,33)):raise ValueError("Missing object item")
    out={}
    for i,(start,stop) in enumerate(zip(starts,[*starts[1:],len(lines)],strict=True),1):
        fields={};cur=None
        for v in lines[start+1:stop]:
            if not v or re.fullmatch(r"Page \d+",v):continue
            label=v.casefold().rstrip(":").replace("transition statement ","transition ")
            if label in ("introduction","transition 1","transition 2","test 1","test 2","old","new","neither","conclusion","test 3"):
                if label in fields:raise ValueError((i,label,"repeated"))
                cur=label;fields[cur]=""
            elif v.casefold().startswith("test word:"):
                fields["target"]=v.split(":",1)[1].strip().lower()
            elif cur:
                fields[cur]=fields[cur][:-1]+v if fields[cur].endswith("-") else (fields[cur]+" "+v).strip()
        out[i]={k:fields[v] for k,v in (("intro","introduction"),("first","transition 1"),("last","transition 2"),
                                           ("rementioned","old"),("new","new"),("neutral","neither"),("target","target"))}
    return out


def object_words(text,token):
    return [m.group().casefold() for m in (CLOTHING if token=="t-shirt" else WORD).finditer(text)]


def main():
    variants=[];families={};old=set()
    for r in map(json.loads,(ROOT/"goal-access-stimuli.jsonl").read_text().splitlines()):
        words0=words(r["text"])
        old.update(tuple(words0[i:i+20]) for i in range(len(words0)-19))
    for num,f in read_stories().items():
        target=f["target"];intro=f["intro"];first=f["first"]; last=f["last"];common=intro+" "+first
        raw=object_words(intro,target)
        if target not in raw:raise ValueError((num,"Missing entire exact probe"))
        if target=="t-shirt":
            cnt=Counter(raw);pos={w:i for i,w in enumerate(raw)}
            forbidden=set(object_words(" ".join((first,last,f["new"],f["neutral"])),target))|STOP|{target}
            matched=sorted(set(raw)-forbidden,key=lambda w:(abs(cnt[w]-cnt[target]),abs(pos[w]-pos[target]),abs(len(w)-len(target)),w))[:3]
        else:
            matched=controls(target,intro,first,last,f["new"],f["neutral"])
        if len(matched)!=3:raise ValueError("Missing original word controls")
        fam={"probe":target,"control_words":matched,"variants":{}}
        for role in ("rementioned","new","neutral"):
            middle=f[role];text=common+"\n\n"+middle+" "+last;uid=sha256(text.encode()).hexdigest()
            variants.append(dict(id=uid,text=text,prior_end=len(intro),first_start=len(intro)+1,first_end=len(common),
                                 last_start=len(text)-len(last),last_end=len(text),targets=[target,*matched]))
            u=words(text); overlap=any(tuple(u[i:i+20]) in old for i in range(len(u)-19))
            fam["variants"][role]=dict(id=uid,intervening_primary_mentions=object_words(middle,target).count(target),
                        manipulation_word_count=len(object_words(middle,target)),shared_20_lexical_words_with_old_goal_study=overlap)
        families[str(num)]=fam
    if len({r["id"] for r in variants})!=96:raise ValueError("Expected 96 distinct original authored presentations")
    INPUT.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in variants))
    OUT.mkdir(parents=True,exist_ok=True)
    dump(OUT/"object-input.json",dict(stimulus_source=sha(SOURCE),protocol=sha(PLAN),addendum=sha(ARM),
             input=sha(INPUT),families=families,conditions=["rementioned","new","neutral"],presentations=len(variants)))
    print(sha(INPUT),len(variants),dict(Counter(r["targets"][0] for r in variants).most_common(3)))


if __name__=="__main__":
    main()
