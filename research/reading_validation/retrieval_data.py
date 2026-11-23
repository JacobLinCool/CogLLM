"""Randomize premise locations without giving source or target annotations to the encoder."""

from collections import Counter
import hashlib
import json
from pathlib import Path

from research.reading_validation.local_inference import english_sentences
from research.reading_validation.memory_data import BASE, OUT as MEMORY, h, load_rows, sha


DATA = BASE / "retrieval"
RESULT = Path("research/results/reading-validation-retrieval")
TRACK = BASE.parent / "semantic-tracking"


def prepare():
    DATA.mkdir(exist_ok=True)
    RESULT.mkdir(exist_ok=True)
    plan = Path(__file__).with_name("retrieval-protocol.md")
    sealed = json.loads((MEMORY/"input-seal.json").read_bytes())
    if sha(BASE/"memory"/"cad.jsonl")!=sealed["files"]["cad.jsonl"]:
        raise ValueError("Revised premise labels differ from the source experiment")
    documents, roles = {}, {}
    meta, coverage = [], Counter()

    def add(text, spans, domain):
        key = h(text)
        if any(not (0<=a<b<=len(text)) or (i and spans[i-1][1]>a) for i,(a,b) in enumerate(spans)):
            raise ValueError("Overlapping or invalid displayed units")
        row = dict(id=key,text=text,spans=spans)
        if documents.setdefault(key,row)!=row:
            raise ValueError("One stimulus has multiple segmentations")
        roles.setdefault(key,set()).add(domain)
        return key

    def paragraphs(text):
        sections = text.split("\n\n")
        spans, start = [],0
        for s in sections:
            spans.append((start,start+len(s)))
            start += len(s)+2
        return spans

    for row in load_rows(BASE/"memory/cad.jsonl"):
        n = row["n"]
        position = int(h("retrieval-order-v1/"+row["id"]),16)%(n+1)
        ids = []
        for text in row["texts"]:
            units = text.split("\n\n")
            if len(units)!=n+2:
                raise ValueError("Unexpected number of premise and distractor units")
            premise = units.pop(0)
            units.insert(position,premise)
            article = "\n\n".join(units)
            ids.append(add(article,paragraphs(article),"cad"))
        meta.append(dict(family=row["id"],split=row["split"],evidence=position,n=n,ids=ids,y=row["y"]))
    mapping = {"frick":{}, "currie":{}, "news":{}}
    for row in load_rows(BASE/"memory/stories.jsonl"):
        text = "\n\n".join(row["sentences"])
        mapping["frick"][row["id"]] = add(text,paragraphs(text),"frick")
    for row in load_rows(TRACK/"larrc-consistency/prepared/stimuli.jsonl"):
        mapping["currie"][row["id"]] = add(row["text"],paragraphs(row["text"]),"currie")
    for row in load_rows(TRACK/"scarecrow-detection/prepared/stimuli.jsonl"):
        spans = english_sentences(row["text"])
        mapping["news"][row["id"]] = add(row["text"],spans,"news")
    with (DATA/"packet.jsonl").open("w") as f:
        for key,row in sorted(documents.items()):
            f.write(json.dumps(row,ensure_ascii=False)+"\n")
            coverage.update(roles[key])
    for name,data in (("training.json",meta),("mapping.json",mapping)):
        (DATA/name).write_text(json.dumps(data,sort_keys=True,ensure_ascii=False)+"\n")
    seal=dict(roles=coverage,unique_documents=len(documents),protocol=sha(plan),source=sha(BASE/"memory/cad.jsonl"),
              files={n:sha(DATA/n) for n in ("packet.jsonl","training.json","mapping.json")})
    (RESULT/"input-seal.json").write_text(json.dumps(seal,indent=2)+"\n")
    print(json.dumps(seal))


if __name__=="__main__":
    prepare()
