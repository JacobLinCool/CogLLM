"""Preserve exact Mandarin source prompts while separating anonymous human measurements."""

import csv
from collections import Counter,defaultdict
from hashlib import sha256
import json
from pathlib import Path

from research.reading_validation.transition_architecture_prepare import load_tokenizer
from research.reading_validation.transition_score import dump,sha
from research.reading_validation.transition_snac_prepare import ROOT


OUT=Path("research/results/reading-validation-transition-mandarin")
PROTOCOL=Path("research/reading_validation/transition-mandarin-protocol.md")
EXPS=("baseline","bei","ba")
SOURCES=("cd04e554aea959fd075564ea432a6ccf3bbd6e88bf2fbdabc3f8abeb6b3ff4ef","3b1f8de102c796d9b32b614a5507f5c9f468e9f7de23ccc1c1895cb46df18acb","9c71ee1da6decbc2afb0bcc4d8e4d972c8dfd4a00174a72198c668a4ff4efc32")


def read(name):
    with (ROOT/("mandarin-"+name+".csv")).open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))


def key(text):return sha256(text.encode()).hexdigest()[:20]


def main():
    tok=load_tokenizer("qwen",json.loads(Path("research/backbone_transfer/models.json").read_text())["models"]["qwen"])
    stimuli={}; trials=[];parity=set();counts={}; seen=defaultdict(set);folds=defaultdict(dict);original_files={}
    for exp,ss,npeople,nrows in zip(EXPS,SOURCES,(50,71,87),(1651,2334,2741),strict=True):
        f=ROOT/("mandarin-"+exp+".csv");original_files[f.name]=sha(f)
        fc=ROOT/("mandarin-"+exp+"-counts.csv");original_files[fc.name]=sha(fc)
        if sha(f)!=ss:raise ValueError("Original Mandarin response sheet changed")
        rows=read(exp);pids=sorted({r["participantID"] for r in rows})
        if len(pids)!=npeople or len(rows)!=nrows:raise ValueError("Original published cohort size differs")
        subjects={s:i for i,s in enumerate(pids)};rate=Counter()
        for r in rows:
            t=r["prompt"];prompted=r["promptType"]=="pronoun";bare=t[:-1] if prompted else t
            if not bare.endswith("。") or (prompted and t[-1] not in ("她","他")):raise ValueError("Unexpected Chinese exposure")
            if r["referenceForm"] not in ("overtPro","name","nullPro") or r["reMention"] not in ("np1","np2"):
                raise ValueError("Unknown eligible referent or expression")
            if prompted and r["referenceForm"]!="overtPro":raise ValueError("Expected overt pronoun in prompted response")
            verb=r["verb"].split("_")[-1]
            family={"xiaokan":"anwei","ciji":"jili"}.get(verb,verb)
            structure=r.get("structure","active")
            structure="active" if structure=="canonical" else structure
            seen[(exp,family,structure,r["promptType"])].add(t)
            form=folds[family].setdefault(exp,set());form.add(bare)
            y=int(r["reMention"]=="np1");o=int(r["referenceForm"]=="overtPro")
            trials.append({"exp":exp,"reader":subjects[r["participantID"]],"family":family,"source_verb":verb,"structure":structure,"class":r["verbType"],
                           "bare":key(bare),"shown":key(t),"condition":r["promptType"],"first":y,"overt":o})
            e=tok.encode(t,add_special_tokens=False);prefix=tok.encode(bare,add_special_tokens=False)
            if prompted and e.ids!=prefix.ids+e.ids[-1:]:raise ValueError("Overt pronoun input changes the original bare token sequence")
            stimuli[key(t)]={"id":key(t),"text":t,"context_end_token":len(prefix.ids)-1,"tokens":len(e.ids)}
            if prompted:parity.add((key(bare),key(t)))
            rate[(r["promptType"],r["referenceForm"])]+=1
        counts[exp]={"people":len(pids),"eligible_responses":len(rows),"modes":{"|".join(k):v for k,v in rate.items()}}
    if len(folds)!=36 or any(len(v)!=1 for v in seen.values()):raise ValueError("Unexpected item grouping")
    overlap={}
    for family,v in folds.items():
        source=set().union(*(ps[x] for f,ps in folds.items() if f!=family for x in EXPS[:2]))
        if source&v["ba"]:raise ValueError("Held Chinese item text reached source")
        overlap[family]=len(source&v["ba"])
    if any(b not in stimuli or p not in stimuli for b,p in parity):raise ValueError("Unobserved free prompt")
    packet=ROOT/"mandarin-reading-stimuli.jsonl";packet.write_text("".join(json.dumps(stimuli[k],ensure_ascii=False,separators=(",",":"))+"\n" for k in sorted(stimuli)))
    obs=ROOT/"mandarin-anonymous-choices.json";dump(obs,trials)
    OUT.mkdir(exist_ok=True,parents=True);dump(OUT/"context-pairs.json",sorted(parity))
    dump(OUT/"source-seal.json",{"protocol_sha256":sha(PROTOCOL),"input_sha256":sha(packet),"external_measurements_sha256":sha(obs),
         "intake":counts,"source_files_sha256":original_files,"canonical_materials_pdf_sha256":sha(ROOT/"mandarin-stimuli.pdf"),
         "paired_name_families":len(folds),"distinct_source_verbs":len({r["source_verb"] for r in trials}),
         "paired_name_replacements":{"anwei_xiaokan":"孟璇 / 李月: 安慰 → 小看","jili_ciji":"淑贞 / 香惠: 激励 → 刺激"},
         "original_exposures":len(stimuli),"shared_context_pairs":len(parity),"max_native_tokens":max(r["tokens"] for r in stimuli.values()),
         "external_overlap_by_held_predicate":overlap,"original_author_analysis_sha256":sha(ROOT/"mandarin-models.R")})
    print(len(stimuli),len(parity),counts,flush=True)


if __name__=="__main__":main()
