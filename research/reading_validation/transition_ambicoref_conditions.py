"""Recover exact documented stimulus-design labels from the authors' four source verb tables."""
import csv
from collections import Counter,defaultdict
import io
import json
import tarfile

from research.reading_validation.transition_ambicoref_prepare import OUT,ROOT,tokens
from research.reading_validation.transition_score import dump,sha


def source_tables():
    tables={}
    with tarfile.open(ROOT/"ambicoref-source.tgz") as t:
        for m in t:
            if "/Data/Verb_phrases/" in m.name and m.isfile():
                tables[m.name.rsplit("/",1)[1][:-4]]=list(csv.DictReader(io.TextIOWrapper(t.extractfile(m),encoding="utf-8-sig")))
    expected={"ECO","ECS","TOP","IC"}
    if set(tables)!=expected:raise ValueError("Published verb category tables changed")
    return tables


def recover(g,tables):
    words=[a.casefold() for a in g["visible_words"]];q=g["q"];suffix=tuple(words[q+1:])
    target=words[q]
    def choose(source,cols,mode):
        hits=[]
        for cls,col in zip(("ambiguous","unambiguous"),cols,strict=True):
            for r in tables[source]:
                v=tokens(r.get(col,"") or "")
                if not v:continue
                if source=="TOP":v=tuple("she" if x=="he" and target=="she" else "her" if x in ("him","his") and target=="she" else x for x in v)
                if mode=="suffix":match=suffix==v
                elif mode=="starts":match=suffix[:len(v)]==v
                else:match=tuple(words[:q])[-len(v):]==v
                if match:hits.append((len(v),cls))
        if not hits:return None
        size=max(t[0] for t in hits);cats={s for n,s in hits if n==size}
        return source+"_"+next(iter(cats)) if len(cats)==1 else None
    if "told" in words[:q] and "that" in words[:q]:
        if words[q-1]=="that":return choose("ECO",("bored","saw"),"starts")
        if q==len(words)-1:return choose("ECS",("liked","met-with"),"ends")
        return None
    if q and words[q-1]=="because":return choose("IC",("reason_ambig","reason_unambig"),"suffix")
    return choose("TOP",("event_ambig","event_unambig"),"suffix")


def main():
    seal=json.loads((OUT/"external-seal.json").read_text());gold=ROOT/"ambicoref-observed-choices.json"
    if sha(gold)!=seal["outcomes_sha256"]:raise ValueError("Frozen experimental spreadsheet changed")
    labels=json.loads(gold.read_text());tables=source_tables();audit=Counter();final={}
    for k,g in labels.items():
        existing=g["condition"];detected=recover(g,tables)
        if existing!="unmatched":
            construct,cond=existing.split("_");check=construct.split("-")[0]+"_"+cond
            if detected and detected!=check:raise ValueError(f"Original full corpus disagrees with its source verb table: {existing} {detected}")
            final[k]=check;audit["exact_complete_corpus"]+=1;audit["exact_also_matched_table"]+=bool(detected)
        elif detected:final[k]=detected;audit["independently_matched_source_verb_class"]+=1
        else:final[k]="unmatched";audit["no_exact_support"]+=1
    dump(OUT/"design-conditions.json",{"external_seal_sha256":sha(OUT/"external-seal.json"),"method":"Original complete stimuli and exact source verb phrases; outcome-independent.",
          "audit":dict(audit),"conditions":dict(Counter(final.values())),"assignments":final})
    print(audit,Counter(final.values()))


if __name__=="__main__":main()
