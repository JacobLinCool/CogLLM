"""Seal source and variant geometry for independent human rewrite evaluation."""

from collections import defaultdict
from pathlib import Path
import csv
import json
import shutil

from research.reading_validation.external_input import digest_string, material_key
from research.reading_validation.joint_data import DEFAULT_CACHE, digest, lines, read
from research.reading_validation.rewrite_input import ngrams

BASE = DEFAULT_CACHE / "reading-validation"
SOURCE = BASE / "revision"
OUT = Path("research/results/reading-validation-revision")
COLS = {
    "past": ("original", "generation"),
    "wikida": ("orig_sent", "simp_sent"),
    "simpda2022": ("Input.original", "Input.simplified"),
}


def records(study):
    if study == "asset_pair":
        path = next(BASE.glob("asset-asset_pair*.csv"))
    elif study == "asset_rating":
        path = BASE / "asset-human_ratings.csv"
    else:
        path = SOURCE / (study + ".csv")
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            if study == "past" and not row["generation"].strip():
                continue
            yield row


def geometry():
    materials = defaultdict(set)
    for study in (*COLS, "asset_pair", "asset_rating"):
        for r in records(study):
            if study == "asset_pair":
                source, versions = r["source"], [r["asset_simplification"], r["other_simplification"]]
            elif study == "asset_rating":
                source, versions = r["original"], [r["simplification"]]
            else:
                a, b = COLS[study]
                source, versions = r[a], [r[b]]
            group = r["Input.id"] if study == "simpda2022" else material_key(source)
            if not source.strip() or any(not v.strip() for v in versions):
                raise ValueError(f"Missing text for {study}")
            materials["train" if study != "simpda2022" else "test", group].update([source, *versions])
    return materials


def earlier_training_texts():
    materials = read(BASE / "readme/materials.json")["rows"]
    texts = {r["id"]: r["text"] for r in lines(BASE / "external-stimuli.jsonl")}
    known = {texts[r["id"]] for r in materials if r["split"] != "test"}
    union = {
        r["id"]: r["text"] for r in lines(DEFAULT_CACHE / "union-training/observations-2/stimuli.jsonl")
    }
    for task in (
        "clear_excerpt_easiness",
        "plaba_processing_ease",
        "studentannotations_comprehension_difficulty",
    ):
        for r in lines(DEFAULT_CACHE / f"adaptive-joint/inputs/observations/{task}.jsonl"):
            key, stop = r["coordinates"][0]
            known.add(union[key][:stop])
    for r in lines(DEFAULT_CACHE / "stage2/plaba/stimuli.jsonl"):
        known.update(r["text"][part["start"] : part["end"]] for part in r["segments"])
    return known


def near_terms(texts):
    inv = defaultdict(set)
    terms = [ngrams(t) for t in texts]
    for i, ng in enumerate(terms):
        for g in ng:
            inv[g].add(i)
    return terms, inv


def matches(text, terms, inverted):
    ng = ngrams(text)
    ix = {i for g in ng for i in inverted.get(g, ())}
    return {
        i for i in ix if len(ng & terms[i]) >= 2 and len(ng & terms[i]) >= 0.8 * min(len(ng), len(terms[i]))
    }


def components():
    texts = {}
    for study in ("wikida", "past", "asset_rating", "asset_pair"):
        col = "source" if study == "asset_pair" else "original" if study != "wikida" else "orig_sent"
        for r in records(study):
            texts.setdefault(material_key(r[col]), r[col])
    keys = sorted(texts)
    v = [texts[k] for k in keys]
    parent = list(range(len(v)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    terms, inv = near_terms(v)
    for i, s in enumerate(v):
        for j in matches(s, terms, inv):
            a, b = root(i), root(j)
            parent[max(a, b)] = min(a, b)
    return {k: digest_string(keys[root(i)]) for i, k in enumerate(keys)}


def run():
    groups = geometry()
    known = earlier_training_texts()
    for (tag, _), items in groups.items():
        if tag == "train":
            known.update(items)
    known = {material_key(v): v for v in sorted(known)}
    grams, inv = near_terms(list(known.values()))
    excluded = {}
    for (tag, key), values in groups.items():
        if tag != "test":
            continue
        for t in sorted(values):
            if material_key(t) in known or matches(t, grams, inv):
                excluded[key] = "exact or near text overlap with fitting materials"
                break
    unique = sorted(
        {t for (phase, g), vs in groups.items() if phase == "train" or g not in excluded for t in vs},
        key=digest_string,
    )
    packet = SOURCE / "stimuli.jsonl"
    packet.write_text(
        "".join(json.dumps({"id": digest_string(t), "text": t}, ensure_ascii=False) + "\n" for t in unique)
    )
    mass = SOURCE / "lexical.npz"
    shutil.copyfile(BASE / "local-lexical.npz", mass)
    foldmap = components()
    for name, data in [("foldmap.json", foldmap)]:
        (SOURCE / name).write_text(json.dumps(data, sort_keys=True, separators=(",", ":")) + "\n")
    OUT.mkdir(exist_ok=True, parents=True)
    with (SOURCE / "past.csv").open(newline="", encoding="utf-8-sig") as source_stream:
        empty_past_outputs = sum(not row["generation"].strip() for row in csv.DictReader(source_stream))
    result = dict(
        excluded_empty_past_outputs=empty_past_outputs,
        source_revisions={
            "LENS": "8fa5b149f099f0c75563ee8344a2411b0bffbc37",
            "WikiDA": "7703c461858cb2557ff1d8ce0a20c63d8a3d280f",
            "ASSET": "9d659040d0d8942dbc4cd65cf357563b43fd9ab4",
        },
        protocol_sha256=digest(Path(__file__).with_name("revision-protocol.md")),
        stimuli_sha256=digest(packet),
        lexical_sha256=digest(mass),
        material_files={p.name: digest(p) for p in [*(SOURCE / (study + ".csv") for study in COLS), *BASE.glob("asset-*.csv")]},
        train_family_map=digest(SOURCE / "foldmap.json"),
        training_source_strings=len(foldmap),
        training_components=len(set(foldmap.values())),
        test_source_ids=sum(t == "test" for t, g in groups),
        excluded_test_source_ids=excluded,
        unique_texts=len(unique),
        word_count=sum(len(t.split()) for t in unique),
        max_words=max(len(t.split()) for t in unique),
        prepare_sha256=digest(Path(__file__)),
    )
    (OUT / "inputs.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        {
            k: result[k]
            for k in (
                "unique_texts",
                "word_count",
                "training_components",
                "excluded_test_source_ids",
                "stimuli_sha256",
                "lexical_sha256",
            )
        }
    )


if __name__ == "__main__":
    run()
