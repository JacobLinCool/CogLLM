"""Create outcome-free sentence queries and source-family splits in both languages."""

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from research.reading_validation.argument_data import sentences
from research.reading_validation.local_forecast import digest, rows
from research.reading_validation.memory_data import BASE, h
from research.reading_validation.nns_exposure import grams

OUT = Path("research/results/reading-validation-coherence-order")
STATE = Path("research/results/reading-validation-state-transport/inputs.json")
HIST = Path("research/results/reading-validation-transition-external")
ITALIAN = BASE / "italian-order"
PLAN = Path(__file__).with_name("coherence_order_protocol.md")
WORD = re.compile(r"[^\W\d_]+(?:['’\-][^\W\d_]+)*|\d+(?:[.,]\d+)*")


def read(path):
    return json.loads(path.read_bytes())


def csv_rows(path, separator="\t"):
    with path.open(encoding="utf-8-sig", newline="") as f:
        yield from csv.DictReader(f, delimiter=separator)


def groups(articles):
    parents = {key: key for key in articles}

    def root(a):
        while parents[a] != a:
            parents[a] = parents[parents[a]]
            a = parents[a]
        return a

    def join(a, b):
        x, y = sorted((root(a), root(b)))
        parents[y] = x

    known_doc, shingles = {}, {}
    for k, r in sorted(articles.items()):
        if r["role"] == "train":
            doc = r["source_id"].split("#")[0]
            join(k, known_doc.setdefault(doc, k))
        words = tuple(m.group().casefold() for m in WORD.finditer(r["text"]))
        for gram in grams(words, 12):
            join(k, shingles.setdefault(gram, k))
    return {k: root(k) for k in articles}


def prepare():
    src = read(STATE)
    italian = {}
    for role in ("train", "held"):
        data = ITALIAN / (role + ".tsv")
        if digest(data) != src["raw"]["italian-order/" + role + ".tsv"]:
            raise ValueError("Unexpected Italian material")
        expected = {r["id"]: r["key"] for r in src["groups"]["italian"][role]}
        for r in csv_rows(data):
            key = h(r["TEXT"])
            if key != expected[r["ID"]] or key in italian:
                raise ValueError("Ambiguous original passage")
            italian[key] = {"role": role, "text": r["TEXT"], "source_id": r["ID"]}
    cluster = groups(italian)
    held_groups = {g for k, g in cluster.items() if italian[k]["role"] == "held"}
    train_groups = {
        g for k, g in cluster.items() if italian[k]["role"] == "train"
    } - held_groups
    ids = sorted(train_groups, key=lambda g: h("coherence-order-source-v1:" + g))
    fit = set(ids[: int(0.8 * len(ids))])
    bank = {}
    for k, r in italian.items():
        g = cluster[k]
        role = (
            "held"
            if r["role"] == "held"
            else "omit"
            if g in held_groups
            else "fit"
            if g in fit
            else "dev"
        )
        pts = sentences(r["text"])
        choices = pts[1:] if len(pts) > 1 else pts
        if not choices:
            raise ValueError("Empty Italian passage")
        bank[k] = {
            "id": k,
            "group": g,
            "role": role,
            "source": "italian",
            "source_id": r["source_id"],
            "segments": pts,
            "boundaries": [
                {"id": h(f"{k}/{a}/{b}"), "start": a, "end": b} for a, b in choices
            ],
        }
    m = read(HIST / "stimuli.json")
    geometry = {r["id"]: r for r in rows(BASE / "cohesion-stimuli.jsonl")}
    source_groups = defaultdict(list)
    for r in m["material"]:
        if r["source"] in ("cohesentia", "decor"):
            source_groups[r["doc"]].append(r)
    for k, records in source_groups.items():
        record = records[0]
        s = geometry[k]
        if k != h(s["text"]) or src["groups"]["english_coherence"][k] != k:
            raise ValueError("English source identity changed")
        stops = {v["id"]: v for v in s["stops"]}
        mapped = [
            {a: stops[q["id"]][a] for a in ("id", "start", "end")} for q in records
        ]
        bank[k] = {
            "id": k,
            "group": record["group"],
            "role": {
                "train": "fit",
                "dev": "dev",
                "holdout": "held",
                "external": "held",
            }[record["split"]],
            "source": record["source"],
            "source_id": record["group"],
            "segments": sentences(s["text"]),
            "boundaries": sorted(mapped, key=lambda u: u["end"]),
        }
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "geometry.json"
    target.write_text(json.dumps(bank, separators=(",", ":")) + "\n")
    evidence = {
        "protocol": digest(PLAN),
        "parent": digest(STATE),
        "english_source": digest(HIST / "stimuli.json"),
        "geometry": digest(target),
        "role_documents": dict(
            Counter(r["source"] + "/" + r["role"] for r in bank.values())
        ),
        "role_boundaries": dict(
            sum(
                (
                    Counter({r["source"] + "/" + r["role"]: len(r["boundaries"])})
                    for r in bank.values()
                ),
                Counter(),
            )
        ),
        "italian": {
            role: {
                "materials": len(
                    v := [
                        r
                        for r in bank.values()
                        if r["source"] == "italian" and r["role"] == role
                    ]
                ),
                "connected_groups": len({r["group"] for r in v}),
                "single_segment_pooled": sum(len(r["segments"]) == 1 for r in v),
            }
            for role in ("fit", "dev", "held", "omit")
        },
    }
    (OUT / "geometry-seal.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(evidence["role_documents"])
    print(evidence["italian"])


def texts():
    return {r["id"]: r["text"] for r in rows(BASE / "world-states/stimuli.jsonl")}


def historical_exposure():
    from research.reading_validation.emtec_material import exposures

    material = read(OUT / "geometry.json")
    content = texts()
    held = [
        k
        for k, r in material.items()
        if r["source"] == "italian" and r["role"] == "held"
    ]
    bad, details = exposures([content[k] for k in held])
    output = {
        "official_test_passages": len(held),
        "overlapping_test_ids": [material[held[i]]["source_id"] for i in sorted(bad)],
        "source_matches": details,
        "inference_model": "The new coherence readout uses frozen Qwen and source coherence ratings only.",
    }
    (OUT / "historical-italian-exposure.json").write_text(
        json.dumps(output, indent=2) + "\n"
    )
    print("Any historical target overlap", len(bad))


if __name__ == "__main__":
    import argparse

    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("--audit-history", action="store_true")
    if a.parse_args().audit_history:
        historical_exposure()
    else:
        prepare()
