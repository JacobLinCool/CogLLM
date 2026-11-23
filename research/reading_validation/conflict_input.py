"""Prepare text-only XNLI inputs, keeping language translations in the same partition."""

from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZipFile

from research.reading_validation.external_input import digest_string, material_key


def prepare(cache):
    source = cache / "xnli.zip"
    pairs = {}
    with ZipFile(source) as bundle:
        for split in ("dev","test"):
            by_id = {}
            with bundle.open(f"XNLI-1.0/xnli.{split}.jsonl") as f:
                for line in f:
                    row = json.loads(line)
                    lang = row["language"]
                    if lang not in ("en","de"):
                        continue
                    entry = by_id.setdefault(row["pairID"],{})
                    if lang in entry:
                        raise ValueError("Duplicated XNLI translation")
                    entry[lang] = row
            if any(set(value)!={"en","de"} for value in by_id.values()):
                raise ValueError("Incomplete translation pair")
            ids = sorted(by_id,key=lambda k:digest_string("cogllm-frozen-nli-v1/"+k))
            pairs[split] = [by_id[k] for k in (ids[:512] if split=="test" else ids)]
    test_premises = {material_key(row["sentence1"]) for pair in pairs["test"] for row in pair.values()}
    export, design = {}, []
    for partition, groups in pairs.items():
        for pair in groups:
            if partition=="dev" and any(material_key(row["sentence1"]) in test_premises for row in pair.values()):
                continue
            for lang,row in pair.items():
                text = row["sentence1"]+"\n\n"+row["sentence2"]
                uid = digest_string(text)
                export[uid]=text
                design.append(dict(id=uid,pair=row["pairID"],lang=lang,role=partition,
                                   premise=row["sentence1"],hypothesis=row["sentence2"],label=row["gold_label"]))
    path = cache/"conflict-stimuli.jsonl"
    path.write_text("".join(json.dumps(dict(id=k,text=t),ensure_ascii=False)+"\n" for k,t in sorted(export.items())))
    (cache/"conflict-design.json").write_text(json.dumps(design,ensure_ascii=False))
    receipt = dict(n_pairs={split:len({r["pair"] for r in design if r["role"]==split}) for split in pairs},
                   source_sha256=sha256(source.read_bytes()).hexdigest(), stimuli_sha256=sha256(path.read_bytes()).hexdigest(),
                   unique_sentences=len(export))
    (cache/"conflict-input.json").write_text(json.dumps(receipt,indent=2)+"\n")
    return receipt


if __name__=="__main__":
    base = Path(".cache/cogllm/reading-validation")
    print(json.dumps(prepare(base)))
