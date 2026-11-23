"""Separate all eligible source-game actions from held-out forward inputs."""

from collections import Counter,defaultdict
from itertools import chain
import json
from pathlib import Path

from research.reading_validation.antecedent_inference import BUNDLE,load_tagger
from research.reading_validation.transition_direct_data import gold_docs
from research.reading_validation.transition_game_prepare import silver_docs
from research.reading_validation.transition_pointer_data import OUT,ROOT
from research.reading_validation.transition_pointer_features import FORMS
from research.reading_validation.transition_pointer_unconditional import labels
from research.reading_validation.transition_score import dump,sha


DEST=OUT/"action-gate"
PROTOCOL=Path("research/reading_validation/transition-pointer-abstention-protocol.md")


def run():
    old=ROOT/"pointer-stimuli.jsonl"
    if sha(old)!="2dee71b79d698976c0ab2bb6fa100f3fe085f45fa47f08e01aac6770711d94b0":
        raise ValueError("Source game study changed")
    rows={r["id"]:r for r in map(json.loads,old.read_text().splitlines())}
    tagger=load_tagger(BUNDLE/"averaged_perceptron_tagger_eng.zip")
    data=[];gold={};counts=defaultdict(Counter)
    catalog=chain(silver_docs(),((d,g,"test",tr,cr,el,ts) for d,g,tr,cr,el,ts in gold_docs()))
    for doc,_,fold,tr,cr,el,ts in catalog:
        if doc not in rows:
            continue
        row=rows[doc].copy();row.pop("clicks",None)
        forms=[row["text"][a:b] for a,b in row["words"]]
        qs=[q for q,w in enumerate(forms) if q and w.casefold() in FORMS]
        row["queries"]=qs;last=[]
        for q in qs:
            t=[v for _,v in tagger.tag(forms[:q+1])]
            if t[:q-1]!=row["pos"][:q-1]:raise ValueError("Unexpected future POS influence")
            last.append(t[q-1])
        row["pos_last"]=last
        answers,_=labels(row,tr,cr,el,ts)
        summary={}
        for q,votes in answers.items():
            origins=Counter(tuple(v) for v in votes.values())
            summary[str(q)]={"actions":[[kind,a,b,n] for (kind,a,b),n in sorted(origins.items())]}
            counts[fold]["responses"]+=sum(origins.values())
            counts[fold]["prior_responses"]+=sum(n for (t,_,_),n in origins.items() if t=="DO")
        counts[fold]["texts"]+=1;counts[fold]["measured_positions"]+=len(answers);counts[fold]["offered_positions"]+=len(qs)
        if fold=="test":gold[doc]=summary
        else:row["actions"]=summary
        data.append(row)
    if len(data)!=len(rows) or counts["test"]["responses"]!=4880:
        raise ValueError("Full game corpus differs from the applicability audit")
    DEST.mkdir(parents=True,exist_ok=True);packet=ROOT/"gate-pointer-stimuli.jsonl"
    packet.write_text("".join(json.dumps(r,separators=(",",":"))+"\n" for r in data))
    dump(DEST/"external-actions.json",gold)
    dump(DEST/"source-seal.json",{"protocol_sha256":sha(PROTOCOL),"input_sha256":sha(packet),
         "gold_sha256":sha(DEST/"external-actions.json"),"fold_counts":{k:dict(v) for k,v in counts.items()}})
    print(counts,flush=True)


if __name__=="__main__":
    run()
