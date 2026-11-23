"""Recover 32 complete original manipulations and the prior-word probes from their authors' materials."""
from hashlib import sha256
import json
from pathlib import Path
import re

from research.reading_validation.transition_score import dump,sha

ROOT=Path(".cache/cogllm/reading-validation")
DEST=Path("research/results/reading-validation-transition-goal-access")
PLAN=Path(__file__).with_name("transition-goal-access-protocol.md")
SOURCE=ROOT/"linderholm-archived-source.txt"
INPUT=ROOT/"goal-access-stimuli.jsonl"
WORD=re.compile(r"[^\W\d_]+(?:[’'][^\W\d_]+)*",re.UNICODE)
STOP=set("a an the am is are was were be been being has have had do did does can could would should will shall may might must i he she it we you they me him her us them my your our their his hers its and or for to of at on in from that this with as by not so some very when while other after before during into out than what where why how who which all any both there then because like her herself himself".split())


def recover():
    if sha(SOURCE)!="08b61cdd6bce81a806108db931bc6229cbeb802abede411e07710f6e939c646b":raise ValueError("Unknown official source-line snapshot")
    source=[r.rstrip("\n").split("\t",1) for r in SOURCE.open()]
    if [int(i) for i,_ in source]!=list(range(941)):raise ValueError("Missing original PDF text lines")
    lines=[s for _,s in source];starts=[i for i in range(15,684) if re.fullmatch(r"\d+\.",lines[i])]
    if [int(lines[i][:-1]) for i in starts]!=list(range(1,33))+[1]:raise ValueError("A main family was missing or repeated")
    key=re.compile(r"^(INTRODUCTION|TRANSITION [12]|TEST [123]|GR|NG|NEUT|CONCLUSION)\s*(?=[:(])",re.I)
    items=[]
    for num,(a,b) in enumerate(zip(starts,starts[1:]),1):
        fields={};seen=None
        for line in lines[a+1:b]:
            if re.fullmatch(r"Page \d+",line):continue
            if match:=key.match(line):
                seen=match[1].upper()
                if seen in fields:raise ValueError(f"Duplicate original field {num} {seen}")
                fields[seen]=line[match.end():].strip().lstrip(":").strip()
            elif seen:fields[seen]=fields[seen][:-1]+line if fields[seen].endswith("-") else fields[seen]+" "+line
        if not set(("INTRODUCTION","TRANSITION 1","GR","NEUT","NG","TRANSITION 2","TEST 1","TEST 2"))<=fields.keys():raise ValueError(f"Missing source condition {num}")
        def probe(field):
            m=re.search(r"([A-Z]+)\s*/\s*[A-Z]+",fields[field])
            if not m:raise ValueError(f"Missing published {field} in {num}")
            return m[1].lower()
        target=probe("TEST 2")
        if target!=probe("TEST 1"):raise ValueError("Old-goal target switched between probe opportunities")
        items.append((num,fields,target))
    return items


def main():
    variants=[];families={};failed=[]
    for number,f,target in recover():
        intro=f["INTRODUCTION"];first=f["TRANSITION 1"];last=f["TRANSITION 2"];ids={}
        parts={k:f[v] for k,v in (("rementioned","GR"),("new","NG"),("neutral","NEUT"))}
        words=lambda t:[m.group().casefold() for m in WORD.finditer(t)]
        prior=words(intro);forms=set(prior)
        if target not in forms:failed.append({"story":number,"probe":target,"reason":"no_exact_introduction_occurrence"});continue
        target_last=max(i for i,v in enumerate(prior) if v==target)
        exclude=set(words(first+" "+last+" "+parts["new"]+" "+parts["neutral"]))|STOP|{target}
        def distance(w):
            latest=max(i for i,v in enumerate(prior) if v==w)
            return abs(prior.count(w)-prior.count(target)),abs(latest-target_last),abs(len(w)-len(target)),w
        controls=sorted(forms-exclude,key=distance)[:3]
        if len(controls)!=3:raise ValueError("Fewer than three distinct source controls")
        p1=intro+" "+first;meta={"probe":target,"control_words":controls,"variants":{}}
        for label,p2 in parts.items():
            txt=p1+"\n\n"+p2+" "+last;fid=sha256(txt.encode()).hexdigest()
            variants.append({"id":fid,"text":txt,"prior_end":len(intro),"first_start":len(intro)+1,"first_end":len(p1),
                             "last_start":len(txt)-len(last),"last_end":len(txt),"targets":[target,*controls]})
            meta["variants"][label]={"id":fid,"intervening_primary_mentions":words(p2).count(target),"manipulation_word_count":len(words(p2))}
        families[str(number)]=meta
    DEST.mkdir(exist_ok=True,parents=True)
    INPUT.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in variants))
    dump(DEST/"exposure-seal.json",{"protocol_sha256":sha(PLAN),"source_rendered_text_sha256":sha(SOURCE),"source_rendered_lines":941,
         "input_sha256":sha(INPUT),"families":families,"excluded":failed,"presentations":len(variants),
         "aggregate_human_rt_ms":{"initial":714,"rementioned":713,"neutral":834,"new":908},"original_participants_retained":84,
         "trial_or_item_level_human_times_available":False})
    print(len(families),len(variants),failed,sha(INPUT),flush=True)


if __name__=="__main__":main()
