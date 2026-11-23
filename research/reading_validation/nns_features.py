"""Attach outcome-independent measures to original word locations in English sentences."""

import csv
import json
import re
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
from nltk.stem import SnowballStemmer
from wordfreq import zipf_frequency

from research.reading_validation.local_features import sentence_features
from research.reading_validation.local_forecast import WORD, digest, rows
from research.reading_validation.local_inference import LocalInference, _matrix_blocks
from research.reading_validation.nns_material import DATA, LEVELS, OUT, observations

STEM = SnowballStemmer("english")
SOURCE_WEIGHTS = "2f72b3b28ec5bc4f47bfe084ae76ccf2f9fdc01e4f4d61bfd4a7a38bda99bfa3"
CEFR = {
    "cefrj": "b0dd3c635f1c9a4fdf1490c7e5b7c48e8bbe55b652ad0c9860a95f98e10ae498",
    "octanove": "18c33a407f2f89f7b8de9671c6d45fe3ea0bce45e7d2d7dcaab48d73e0f7b380",
}


@lru_cache(maxsize=60000)
def stem(word):
    return STEM.stem(word)


@lru_cache(maxsize=1)
def curriculum():
    by_surface, by_stem = {}, {}
    for name, expected in CEFR.items():
        path = DATA / (name + ".csv")
        if digest(path) != expected:
            raise ValueError("Unexpected CEFR-J or Octanove reference")
        with path.open(newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                grade = LEVELS.index(row["CEFR"]) + 1
                for alias in row["headword"].split("/"):
                    w = alias.strip().casefold()
                    if WORD.fullmatch(w):
                        for store, key in ((by_surface, w), (by_stem, stem(w))):
                            store[key] = min(grade, store.get(key, grade))
    return by_surface, by_stem


@lru_cache(maxsize=60000)
def lexical(w):
    surfaces, stems = curriculum()
    freq = zipf_frequency(w, "en")
    both = surfaces.get(w, stems.get(stem(w)))
    grade = 3.5 if both is None else float(both)
    return [
        len(w),
        np.log1p(len(w)),
        freq,
        max(0, 3 - freq),
        max(0, freq - 5),
        len(re.findall(r"[aeiouy]+", w)),
        grade,
        float(both is None),
        float(w not in surfaces),
        float("-" in w),
        float(any(x.isdigit() for x in w)),
    ]


def geometry(role):
    audit = json.loads((OUT / "material-audit.json").read_bytes())
    groups = DATA / "eligible-groups.json"
    if digest(groups) != audit["material_sha256"]:
        raise ValueError("Material group assignments changed")
    info = json.loads(groups.read_bytes())
    eligible = set(info["split"][role])
    docs = {
        q["id"]: q["text"]
        for q in rows(DATA / (role + ".jsonl"))
        if q["id"] in eligible
    }
    return docs, info["family"]


def single_word_rank(version, t, key, positions, ok, prob, projected):
    """Evaluate the frozen PLABA ranker on every single-word candidate."""
    arr = {
        "words": np.column_stack(
            [positions, np.where(ok, np.arange(len(positions)), -1)]
        ),
        "lexical": prob,
        "projection": projected,
    }
    node = sentence_features(
        key,
        "article",
        t,
        0,
        len(t),
        version.lexicon,
        arr,
        0,
        0,
        sentence_endpoint=False,
    )
    score = (
        _matrix_blocks(
            version.rank["parts"],
            version.rank["center"],
            version.rank["scale"],
            node.features,
        )
        @ version.rank["coef"]
    )
    return {tuple(span): float(z) for span, z in zip(node.coords, score, strict=True)}


def build(role, extraction):
    docs, families = geometry(role)
    meta = json.loads((extraction / "extraction.json").read_bytes())
    if (
        meta["packet_sha256"] != digest(DATA / (role + ".jsonl"))
        or meta["worker_sha256"] != digest(Path(__file__).with_name("nns_l4.py"))
        or meta["source_weight_sha256"] != SOURCE_WEIGHTS
    ):
        raise ValueError("Unexpected extraction identity")
    for name, sha in meta["files_sha256"].items():
        if digest(extraction / name) != sha:
            raise ValueError("Corrupt native word features")
    all_ids = json.loads((extraction / "ids.json").read_bytes())
    version = LocalInference.load()
    bank = defaultdict(list)
    with np.load(extraction / "signals.npz", allow_pickle=False) as enc:
        offsets, coords, native, p, c, prob = [
            enc[k]
            for k in ("offsets", "word_spans", "native", "plaba", "context", "lexical")
        ]
        for i, key in enumerate(all_ids):
            if key not in docs:
                continue
            t = docs[key]
            matches = list(WORD.finditer(t))
            sl = slice(offsets[i], offsets[i + 1])
            positions = coords[sl]
            if not np.array_equal(positions, np.array([m.span() for m in matches])):
                raise ValueError("Word offsets disagree")
            ok = native[sl] >= 0
            rank = single_word_rank(version, t, key, positions, ok, prob[sl], p[sl])
            for j, m in enumerate(matches):
                if not ok[j]:
                    continue
                word = m.group()
                w = word.casefold()
                bank["doc"].append(key)
                bank["family"].append(families[key])
                bank["type"].append(w)
                bank["stem"].append(stem(w))
                bank["coords"].append(m.span())
                bank["form"].append(
                    [
                        *lexical(w),
                        float(word[:1].isupper()),
                        float(word.isupper()),
                        np.log1p(j),
                    ]
                )
                bank["source"].append([float(prob[sl][j]), rank[m.span()]])
                bank["context"].append(np.r_[p[sl][j], c[sl][j]])
    out = DATA / (role + "-features.npz")
    np.savez_compressed(out, **{k: np.asarray(v) for k, v in bank.items()})
    return out


def selected_counts(role, bank):
    docs, _ = geometry(role)
    # One released string can occur at several positions and the authors supply no token index.
    idx = {}
    multiple = set()
    for key, t in docs.items():
        c = Counter(m.group().casefold() for m in WORD.finditer(t))
        multiple.update((key, w) for w, z in c.items() if z > 1)
    for i, (k, w) in enumerate(zip(bank["doc"], bank["type"], strict=True)):
        if (k, w) not in multiple:
            idx[k, w] = i
    total = np.zeros((len(bank["doc"]), 6), np.int16)
    positives = np.zeros_like(total)
    coverage = Counter()
    ids = {t: key for key, t in docs.items()}
    for level, text, term, mark in observations():
        key = ids.get(text)
        if key is None:
            continue
        k = (key, term.casefold())
        row = idx.get(k)
        if row is None:
            coverage[
                "ambiguous_repeated"
                if k in multiple
                else "not_unique_certified_printed_word"
            ] += 1
            continue
        if mark not in ("Simple", "Complex"):
            raise ValueError("Unknown reader response")
        j = LEVELS.index(level)
        total[row, j] += 1
        positives[row, j] += mark == "Complex"
        coverage["included"] += 1
    return positives, total, dict(coverage)


def pack_model_input(role):
    p = DATA / (role + "-features.npz")
    with np.load(p, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}
