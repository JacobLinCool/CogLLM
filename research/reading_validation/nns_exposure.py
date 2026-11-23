"""Audit candidate geometry and supervised material without using word judgments."""

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
from zipfile import ZipFile

from research.reading_validation.local_forecast import CACHE, digest, rows
from research.reading_validation.nns_material import DATA, OUT
from research.reading_validation.release_overlap import learning_material, words
from research.reading_validation.revision_prepare import (
    earlier_training_texts,
    geometry,
    matches,
    near_terms,
)


def grams(sequence, size):
    return {sequence[i : i + size] for i in range(len(sequence) - size + 1)}


def sources():
    staged, _, receipt = learning_material()
    med = CACHE / "medreadme-audit/raw/readability.csv"
    with med.open(newline="") as stream:
        medical = {
            r["Sentence"]
            for r in csv.DictReader(stream)
            if r["split"] in {"train", "dev"}
        }
    cwi = CACHE / "core-features/cwi-source/cwishareddataset.zip"
    with ZipFile(cwi) as archive:
        news = {
            line.decode("utf-8-sig").split("\t")[1]
            for name in archive.namelist()
            if "/english/" in name
            and name.endswith(".tsv")
            and not name.startswith("__")
            for line in archive.open(name)
        }
    revision = {
        t for (role, _), values in geometry().items() if role == "train" for t in values
    }
    collections = {
        "24_study_training": staged["train"],
        "24_study_development": staged["dev"],
        "readme_clear_students_plaba": earlier_training_texts(),
        "medreadme_train_dev": medical,
        "cwi2018_all_english_sources": news,
        "revision_sources": revision,
        "complex_all_splits": {
            r["text"] for r in rows(CACHE / "stage5/complex-l4-input/stimuli.jsonl")
        },
    }
    return collections, receipt


def run():
    sparse, dense = [rows(DATA / (name + ".jsonl")) for name in ("sparse", "dense")]
    docs = [*dense, *sparse]
    ids = [x["id"] for x in docs]
    strings = [x["text"] for x in docs]
    units = [words(t) for t in strings]
    # Form material components before reading any target observations.
    parent = list(range(len(docs)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    ngrams, index = near_terms(strings)
    whole = {}
    twelve = {}
    for i, text in enumerate(strings):
        edges = matches(text, ngrams, index)
        edges.add(whole.setdefault(units[i], i))
        for w in grams(units[i], 12):
            edges.add(twelve.setdefault(w, i))
        for j in edges:
            a, b = root(i), root(j)
            parent[max(a, b)] = min(a, b)
    families = {key: ids[root(i)] for i, key in enumerate(ids)}
    dense_families = {families[r["id"]] for r in dense}
    target = [r["text"] for r in dense]
    named, receipt = sources()
    gsmall, ismall = near_terms(target)
    exact = defaultdict(set)
    windows = defaultdict(set)
    for i, t in enumerate(target):
        u = words(t)
        exact[u].add(i)
        for w in grams(u, 12):
            windows[w].add(i)
    by_source = {}
    excluded_families = set()
    for name, available in named.items():
        counts = defaultdict(set)
        for t in available:
            u = words(t)
            counts["exact"].update(exact[u] if u in exact else ())
            for k in grams(u, 12):
                if k in windows:
                    counts["shared_12"].update(windows[k])
            counts["near"].update(matches(t, gsmall, ismall))
        matched = {i for vs in counts.values() for i in vs}
        excluded_families.update(families[ids[i]] for i in matched)
        by_source[name] = {
            "source_texts": len(available),
            **{k: len(v) for k, v in counts.items()},
            "any_dense": len(matched),
        }
    split = {
        name: sorted(
            r["id"]
            for r in group
            if (
                families[r["id"]]
                not in (excluded_families if name == "dense" else dense_families)
            )
        )
        for name, group in (("sparse", sparse), ("dense", dense))
    }
    if {families[k] for k in split["dense"]} & {families[k] for k in split["sparse"]}:
        raise ArithmeticError("Held material leaked into supervised sparse rows")
    out = DATA / "eligible-groups.json"
    out.write_text(
        json.dumps(dict(split=split, family=families), sort_keys=True) + "\n"
    )
    counts = Counter(families[r["id"]] for r in dense)
    record = dict(
        earlier_sources=by_source,
        prior_source_receipt=receipt,
        original_dense=len(dense),
        original_sparse=len(sparse),
        eligible={k: len(v) for k, v in split.items()},
        dense_families=len(dense_families),
        dense_family_sizes=dict(Counter(counts.values())),
        excluded_sparse_related_to_dense=len(sparse) - len(split["sparse"]),
        eligible_dense_families=len({families[k] for k in split["dense"]}),
        material_sha256=digest(out),
        program_sha256=digest(Path(__file__)),
    )
    (OUT / "material-audit.json").write_text(json.dumps(record, indent=2) + "\n")
    print(record["eligible"], record["eligible_dense_families"], by_source)


if __name__ == "__main__":
    run()
