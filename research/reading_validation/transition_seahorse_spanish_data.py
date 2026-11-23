"""Seal grade-free Spanish presentations and document their overlap with English learning texts."""

from collections import Counter,defaultdict
import csv
from io import TextIOWrapper
import json
from pathlib import Path
from zipfile import ZipFile

from research.reading_validation.memory_data import h
from research.reading_validation.transition_seahorse_data import DEST, EXPECTED, WORK, ZIP, sha, source_rows
from research.reading_validation.transition_seahorse_features import load, save, signatures

PLAN=Path(__file__).with_name("transition-seahorse-spanish-protocol.md")
INPUT=WORK/"seahorse-spanish-stimuli.jsonl"


def spanish(*,grades=False):
    with ZipFile(ZIP) as archived, archived.open("seahorse_data/test.tsv") as buf:
        reader=csv.DictReader(TextIOWrapper(buf,encoding="utf8",newline=""),delimiter="\t")
        for row in reader:
            if row["worker_lang"]=="es-ES":
                keys=("gem_id","model","summary",*(("question1","question2","question3") if grades else ()))
                yield {k:row[k] for k in keys}


def main():
    if sha(ZIP)!=EXPECTED:
        raise ValueError("Unrecognized original Spanish measurement archive")
    src=DEST/"gated-weights.npz";english=load(DEST/"gated-fit-seal.json")
    if sha(src)!=english["weights_sha256"]:
        raise ValueError("English task-specific coefficients changed")
    rows=list(spanish());unique={h(r["summary"]):r["summary"] for r in rows}
    with INPUT.open("w",encoding="utf8",newline="\n") as fp:
        for k,t in sorted(unique.items()):
            fp.write(json.dumps({"id":k,"text":t},ensure_ascii=False)+"\n")
    candidates={}
    for k,t in unique.items():
        for sig in signatures(t):candidates.setdefault(sig,set()).add(k)
    match=set();exact=set();similar=defaultdict(set)
    for part in ("train","validation"):
        for r in source_rows(part):
            source=r["summary"]
            if h(source) in unique:exact.add(h(source))
            for token in signatures(source):
                hit=candidates.get(token,())
                match.update(hit)
                for word in hit:similar[word].add(r["gem_id"])
    articles=sorted({r["gem_id"] for r in rows if h(r["summary"]) in match})
    corpora=Counter(r["gem_id"].split("-")[0].split("_")[0] for r in rows)
    save(DEST/"spanish-exposure-seal.json",{
        "text_source_sha256":EXPECTED,"protocol_sha256":sha(PLAN),"native_weights":sha(src),"input_sha256":sha(INPUT),
        "official_original_test_rows":len(rows),"unique_strings":len(unique),"distinct_source_articles":len({r["gem_id"] for r in rows}),
        "corpus_rows":dict(corpora),"english_train_or_validation_exact_strings":len(exact),
        "english_train_or_validation_normalized_full_or_shared_twelve_words":len(match),
        "earlier_english_articles_with_text_match":sorted(set().union(*similar.values())),
        "affected_spanish_articles":articles,"input_fields":["id","text"],"new_human_grades_read":False})
    print(sha(INPUT),len(rows),len(unique),len(articles),flush=True)


if __name__=="__main__":
    main()
