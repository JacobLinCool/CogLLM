"""Build a response-free ASSET input packet and check overlap with source studies."""
import csv
from collections import defaultdict
from pathlib import Path
import json
import re

from research.reading_validation.joint_data import DEFAULT_CACHE, digest, lines, read
from research.reading_validation.external_input import digest_string, material_key


def get_texts(cache):
    base=cache/"reading-validation"
    inputs=set()
    groups=defaultdict(set)
    files=sorted(base.glob("asset-*.csv"))
    if len(files)!=2:
        raise ValueError("Expected both official ASSET files")
    for file in files:
        with file.open() as f:
            for row in csv.DictReader(f):
                if "source" in row:
                    orig=row["source"]
                    alternatives=[row["asset_simplification"],row["other_simplification"]]
                else:
                    orig=row["original"]
                    alternatives=[row["simplification"]]
                if not all(isinstance(t,str) and t.strip() for t in [orig,*alternatives]):
                    raise ValueError("Missing source text or rewrite")
                groups[orig].update([orig,*alternatives])
                inputs.update([orig,*alternatives])
    return inputs,groups,files


def ngrams(t):
    words=re.findall(r"\w+",t.casefold())
    return {tuple(words[i:i+5]) for i in range(len(words)-4)}


def prepare(cache=DEFAULT_CACHE):
    base=cache/"reading-validation"
    inputs,groups,files=get_texts(cache)
    materials=read(base/"readme/materials.json")["rows"]
    texts={r["id"]:r["text"] for r in lines(base/"external-stimuli.jsonl")}
    training={texts[r["id"]] for r in materials if r["split"]!="test"}
    union={r["id"]:r["text"] for r in lines(cache/"union-training/observations-2/stimuli.jsonl")}
    for task in ("clear_excerpt_easiness","plaba_processing_ease","studentannotations_comprehension_difficulty"):
        for r in lines(cache/f"adaptive-joint/inputs/observations/{task}.jsonl"):
            key,stop=r["coordinates"][0]
            training.add(union[key][:stop])
    for r in lines(cache/"stage2/plaba/stimuli.jsonl"):
        for part in r["segments"]:
            training.add(r["text"][part["start"]:part["end"]])
    exact={material_key(t) for t in training}
    inv=defaultdict(list)
    grams=[ngrams(t) for t in sorted(training)]
    for i,ng in enumerate(grams):
        for g in ng:
            inv[g].append(i)
    exclusions={}
    for orig,versions in groups.items():
        for v in sorted(versions):
            if material_key(v) in exact:
                exclusions[digest_string(orig)]="exact"
                break
            ng=ngrams(v)
            candidates={i for g in ng for i in inv.get(g,())}
            if any(len(ng & grams[i])>=2 and len(ng & grams[i])>=.8*min(len(ng),len(grams[i])) for i in candidates):
                exclusions[digest_string(orig)]="near"
                break
    packet=base/"rewrite-stimuli.jsonl"
    packet.write_text("".join(json.dumps(dict(id=digest_string(t),text=t),ensure_ascii=False)+"\n"
                              for t in sorted(inputs,key=digest_string)))
    dest=Path("research/results/reading-validation-rewrite")
    dest.mkdir(parents=True,exist_ok=True)
    result=dict(files={p.name:digest(p) for p in files},n_sources=len(groups),n_unique_texts=len(inputs),
                source_exclusions=exclusions,n_training_texts=len(training),
                packet_sha256=digest(packet),model_seal_sha256=digest(Path("research/results/reading-validation-joint/forecast/seal.json")),
                protocol_sha256=digest(Path(__file__).with_name("rewrite-protocol.md")))
    (dest/"inputs.json").write_text(json.dumps(result,indent=2)+"\n")
    return result

if __name__=="__main__":
    print(json.dumps(prepare()))
