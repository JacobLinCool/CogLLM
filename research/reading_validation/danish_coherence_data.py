"""Prepare the official DDisCo test texts without reading their coherence grades."""

import csv
import json
from collections import Counter, defaultdict
from io import TextIOWrapper
from pathlib import Path
from zipfile import ZipFile

from research.reading_validation.coherence_order_data import WORD
from research.reading_validation.local_forecast import digest
from research.reading_validation.memory_data import BASE, h

WORK = BASE / "danish-order"
OUT = Path("research/results/reading-validation-danish-coherence")
SOURCE = WORK / "ddisco.zip"
SHA = "f8c8d7bab8a9a065f6fb28b3d9abecc28a0b2aca1dd6321ab4bf2c9e5fcdff90"
PLAN = Path(__file__).with_name("danish-coherence-protocol.md")


def entries(split, *, labels=False):
    if digest(SOURCE) != SHA:
        raise ValueError("DDisCo original archive does not match the frozen release")
    if split not in ("train", "test"):
        raise ValueError("Unknown split")
    with (
        ZipFile(SOURCE) as archive,
        archive.open(f"ddisco.{split}.tsv") as byte_stream,
        TextIOWrapper(byte_stream, encoding="utf-8", newline="") as stream,
    ):
        table = csv.DictReader(stream, delimiter="\t")
        if table.fieldnames != ["text", "domain", "rating"]:
            raise ValueError("Unexpected DDisCo schema")
        for row in table:
            t = row["text"]
            g = row["domain"]
            if not t.strip() or g not in ("wikipedia", "reddit"):
                raise ValueError("Missing source material")
            item = {"id": h(t), "text": t, "domain": g}
            if labels:
                item["rating"] = int(row["rating"])
            yield item


def words(text):
    return tuple(m.group().casefold() for m in WORD.finditer(text))


def grams(text):
    w = words(text)
    return {w[i : i + 12] for i in range(len(w) - 11)}


def components(train, test):
    parent = {r["id"]: r["id"] for r in [*train, *test]}
    index = {}

    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for row in [*train, *test]:
        k = row["id"]
        for shingle in grams(row["text"]):
            other = index.setdefault(shingle, k)
            a, b = sorted((root(other), root(k)))
            parent[b] = a
    return {k: root(k) for k in parent}


def run():
    train = list(entries("train"))
    test = list(entries("test"))
    grp = components(train, test)
    held = {grp[r["id"]] for r in test}
    seen_train = {grp[r["id"]] for r in train}
    test_ids = {r["id"] for r in test}
    meta = {r["id"]: {"domain": r["domain"], "group": grp[r["id"]]} for r in test}
    if len(meta) != len(test):
        raise ValueError("A held passage has two records")
    order = sorted(test, key=lambda r: r["id"])
    target = WORK / "stimuli.jsonl"
    target.write_text(
        "".join(
            json.dumps(
                {"id": r["id"], "text": r["text"]},
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
            for r in order
        )
    )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "geometry.json").write_text(
        json.dumps(meta, sort_keys=True, indent=2) + "\n"
    )
    overlap = defaultdict(set)
    for r in train:
        if grp[r["id"]] in held:
            overlap[grp[r["id"]]].add(r["id"])
    note = {
        "source": digest(SOURCE),
        "packet": digest(target),
        "official_geometry": digest(OUT / "geometry.json"),
        "protocol": digest(PLAN),
        "source_url": "https://danlp-downloads.alexandra.dk/datasets/ddisco.zip",
        "source_training_rows": len(train),
        "distinct_source_training_texts": len({r["id"] for r in train}),
        "official_held_rows": len(test),
        "held_domains": dict(Counter(r["domain"] for r in test)),
        "held_groups": len(held),
        "source_overlap_in_held": sum(grp[k] in seen_train for k in test_ids),
        "source_train_excluded_rows": sum(grp[r["id"]] in held for r in train),
        "source_train_excluded_unique": sum(map(len, overlap.values())),
        "test_grades_opened": False,
    }
    (OUT / "inputs.json").write_text(json.dumps(note, indent=2) + "\n")
    print(note)


if __name__ == "__main__":
    run()
