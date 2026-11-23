"""Build a grade-free CEFR-SP corpus and prevent near-duplicate train/test matches."""

from collections import Counter, defaultdict
import json
from pathlib import Path

from research.reading_validation.external_input import digest_string, material_key
from research.reading_validation.joint_data import DEFAULT_CACHE, TASKS, digest, lines
from research.reading_validation.rewrite_input import ngrams


BASE = DEFAULT_CACHE / "reading-validation"
OUT = Path("research/results/reading-validation-prior")
DOMAINS = ("SCoRE", "Wikiauto")


def read_source(path):
    with path.open(encoding="utf-8-sig") as f:
        for i,line in enumerate(f):
            parts=line.rstrip("\n\r").split("\t")
            if len(parts)!=3 or not parts[0].strip():
                raise ValueError(f"Invalid source at {path.name}:{i+1}")
            yield i,parts


def auxiliary_texts():
    source={r["id"]:r["text"] for r in lines(DEFAULT_CACHE/"union-training/observations-2/stimuli.jsonl")}
    parts={digest_string(r["text"]):{p["end"]:p["start"] for p in r["segments"]}
           for r in lines(DEFAULT_CACHE/"stage2/plaba/stimuli.jsonl")}
    result=set()
    for study,task in TASKS.items():
        for r in lines(DEFAULT_CACHE/f"adaptive-joint/inputs/observations/{task}.jsonl"):
            key,stop=r["coordinates"][0]
            start=parts[key][stop] if study=="plaba" else 0
            result.add(source[key][start:stop])
    return result


def components(texts):
    parents={k:k for k in texts}
    def root(k):
        while parents[k]!=k:
            parents[k]=parents[parents[k]]
            k=parents[k]
        return k
    def unite(a,b):
        a,b=sorted((root(a),root(b)))
        parents[b]=a
    exact={}
    grams={}
    inv=defaultdict(list)
    for k,t in sorted(texts.items()):
        key=material_key(t)
        if key in exact:
            unite(k,exact[key])
        exact[key]=k
        s=ngrams(t)
        counts=Counter(i for g in s for i in inv[g])
        for i,hits in counts.items():
            if hits>=2 and hits>=.8*min(len(s),len(grams[i])):
                unite(i,k)
        for g in s:
            inv[g].append(k)
        grams[k]=s
    return {k:root(k) for k in texts}


def run():
    aux=auxiliary_texts()
    texts={digest_string(t):t for t in aux}
    rows=[]
    files={}
    for d in DOMAINS:
        for split in ("train","dev","test"):
            file=BASE/f"cefr/{d}-{split}.tsv"
            files[file.name]=digest(file)
            for i,(t,_,_) in read_source(file):
                uid=digest_string(t)
                texts[uid]=t
                rows.append(dict(id=uid,domain=d,split=split,row=i))
    comp=components(texts)
    aux_groups={comp[digest_string(t)] for t in aux}
    test={comp[r["id"]] for r in rows if r["split"]=="test"}
    seen=set()
    keep=[]
    omitted=Counter()
    for r in rows:
        group=comp[r["id"]]
        code=(r["domain"],r["split"],material_key(texts[r["id"]]))
        if group in aux_groups:
            omitted["auxiliary_overlap"]+=1
        elif r["split"]!="test" and group in test:
            omitted["test_relative"]+=1
        elif code in seen:
            omitted["within_partition_exact"]+=1
        else:
            seen.add(code)
            keep.append(r|dict(group=group))
    packet=BASE/"prior-stimuli.jsonl"
    keys=sorted({r["id"] for r in keep})
    packet.write_text("".join(json.dumps(dict(id=k,text=texts[k]),ensure_ascii=False)+"\n" for k in keys))
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"materials.json").write_text(json.dumps(keep,indent=1)+"\n")
    receipt=dict(revision="2ba4a5005fbff68ab0f863891cf977c50fdcda29",source=files,
                 packet_sha256=digest(packet),materials_sha256=digest(OUT/"materials.json"),
                 protocol_sha256=digest(Path(__file__).with_name("transfer-prior-protocol.md")),
                 counts=dict(Counter(r["domain"]+"."+r["split"] for r in keep)),omitted=dict(omitted),
                 auxiliary_materials=len(aux),unique_texts=len(keys))
    (OUT/"inputs.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(receipt))


if __name__=="__main__":
    run()
