"""Source-relative lexical edits and bidirectional contextual coverage of human revisions."""

from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
import json
import re

import numpy as np
from nltk.stem.snowball import EnglishStemmer
from wordfreq import zipf_frequency

from research.reading_validation.external_input import digest_string
from research.reading_validation.joint_data import digest
from research.reading_validation.joint_fit import normalized_hidden
from research.reading_validation.local_inference import english_sentences
from research.reading_validation.revision_forecast import WORD
from research.reading_validation.revision_prepare import OUT
from research.reading_validation.rewrite_forecast import ROOT, predictions


STEM = EnglishStemmer()
NEGATION = {"no", "not", "never", "without", "none", "neither", "nor", "cannot", "n't"}
NUMBER = re.compile(r"(?<!\w)[+-]?\d+(?:[.,]\d+)*(?:%|\b)")
OPERATOR = re.compile(r"[,:;()\[\]—–]")
EMBEDDED_DIMS = 24


@lru_cache(maxsize=40000)
def word_properties(w):
    return STEM.stem(w), zipf_frequency(w, "en")


@dataclass
class Passage:
    text: str
    words: list
    stems: list
    freq: np.ndarray
    idf: np.ndarray
    count: dict
    surface: np.ndarray
    numbers: set
    names: set


def passage(text):
    tokens = [m.group() for m in WORD.finditer(text)]
    words = [t.casefold() for t in tokens]
    stems = [word_properties(w)[0] for w in words]
    freq = np.array([word_properties(w)[1] for w in words])
    sizes = np.array([len(t) for t in tokens])
    sentences = english_sentences(text)
    ln = np.log1p(len(words))
    idf = 1 + np.maximum(5.5 - freq, 0)

    def stat(x, i):
        return float(np.quantile(x, i)) if len(x) else 0.0

    sf = np.array(
        [
            ln,
            np.log1p(len(text)),
            np.log1p(len(sentences)),
            len(words) / max(len(sentences), 1),
            sizes.mean() if len(sizes) else 0,
            stat(sizes, 0.9),
            freq.mean() if len(freq) else 0,
            stat(freq, 0.15),
            np.mean(sizes >= 7) if len(sizes) else 0,
            np.mean(freq < 3) if len(freq) else 0,
            len(OPERATOR.findall(text)),
            len(set(words)) / max(1, len(words)),
            len(NUMBER.findall(text)),
            sum(w in NEGATION for w in words),
        ]
    )
    return Passage(
        text,
        words,
        stems,
        freq,
        idf,
        Counter(stems),
        sf,
        set(NUMBER.findall(text)),
        {w.casefold() for w in tokens[1:] if w[0].isupper()},
    )


def frac(a, b):
    return a / b if b else 1.0


def set_recall(a, b):
    return frac(len(a & b), len(a))


def comparison(a, b):
    denom = max(1, len(a.words) + len(b.words))
    common = a.count & b.count
    left_missing = np.array([w not in b.count for w in a.stems], dtype=bool)
    right_missing = np.array([w not in a.count for w in b.stems], dtype=bool)

    def weighted_recall(p, other):
        if not len(p.words):
            return 1.0
        match = np.array([w in other.count for w in p.stems])
        return float(np.mean(match)), float(np.average(match, weights=p.idf))

    coverage_a = weighted_recall(a, b) if a.words else (1.0, 1.0)
    coverage_b = weighted_recall(b, a) if b.words else (1.0, 1.0)
    coverage_f = 2 * coverage_a[1] * coverage_b[1] / max(coverage_a[1] + coverage_b[1], 1e-12)
    sequence = SequenceMatcher(None, a.stems, b.stems, autojunk=False)
    ops = sequence.get_opcodes()
    ops_info = []
    for name in ("replace", "delete", "insert", "equal"):
        changes = [(j - i, q - p) for label, i, j, p, q in ops if label == name]
        ops_info.extend(
            (
                len(changes) / denom,
                sum(l for l, r in changes) / max(len(a.words), 1),
                sum(r for l, r in changes) / max(len(b.words), 1),
            )
        )
    ng = []
    for k in (2, 3):
        left = {tuple(a.stems[j : j + k]) for j in range(len(a.words) - k + 1)}
        right = {tuple(b.stems[j : j + k]) for j in range(len(b.words) - k + 1)}
        ng.extend((set_recall(left, right), set_recall(right, left)))
    ext = np.r_[
        *coverage_a,
        *coverage_b,
        coverage_f,
        sum(common.values()) / denom,
        set_recall(a.numbers, b.numbers),
        set_recall(b.numbers, a.numbers),
        set_recall(a.names, b.names),
        set_recall(b.names, a.names),
        set_recall(set(a.words) & NEGATION, set(b.words) & NEGATION),
        set_recall(set(b.words) & NEGATION, set(a.words) & NEGATION),
        ng,
        ops_info,
        sequence.ratio(),
        a.freq[left_missing].mean() if left_missing.any() else 0,
        b.freq[right_missing].mean() if right_missing.any() else 0,
    ]
    relative = b.surface - a.surface
    surface = np.r_[a.surface, b.surface, relative, np.abs(relative), ext]
    return surface, dict(length=-float(len(b.text.split())), word_coverage=coverage_f), left_missing, right_missing


def neural_info(extraction, texts):
    lock = json.loads((OUT / "inputs.json").read_bytes())
    meta = json.loads((extraction / "extraction.json").read_bytes())
    if meta["packet_sha256"] != lock["stimuli_sha256"] or meta["lexical_sha256"] != lock["lexical_sha256"]:
        raise ValueError("Revision feature packet differs from locked geometry")
    for name, h in meta["files_sha256"].items():
        if digest(extraction / name) != h:
            raise ValueError(f"Changed extracted feature: {name}")
    ids = json.loads((extraction / "ids.json").read_bytes())
    material = {digest_string(t): t for t in texts}
    if set(material) != set(ids):
        raise ValueError("Feature archive and revision passage inventory differ")
    strings = [material[k] for k in ids]
    states = np.load(extraction / "states.npy", mmap_mode="r")
    source_model = ROOT / "models.npz"
    if digest(source_model) != json.loads((ROOT / "seal.json").read_bytes())["files"]["models.npz"]:
        raise ValueError("Earlier CEFR coefficients changed")
    with np.load(source_model, allow_pickle=False) as model:
        pred = predictions(model, states, strings)
        grades = (1 + pred["adaptive.qwen"].sum(axis=1)).astype(float)
        basis = model["fit.basis.projection"][:, -EMBEDDED_DIMS:]
        terminal = (normalized_hidden(states) - model["fit.basis.hidden_mean"]) @ basis
    vectors = np.asarray(states, dtype=np.float64).copy()
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors /= norms
    with np.load(extraction / "words.npz", allow_pickle=False) as arrays:
        bounds = arrays["range_offsets"]
        observed = arrays["observed"]
        context = arrays["contextual"].astype(np.float32)
        lexical = arrays["lexical"]
        spans = arrays["word_spans"]
    # Context cosines compare the original and revision's matched prefix states.
    context /= np.maximum(np.linalg.norm(context, axis=1, keepdims=True), 1e-9)
    result = {}
    for i, t in enumerate(strings):
        span = slice(bounds[i], bounds[i + 1])
        if not np.array_equal(
            spans[span], np.array([m.span() for m in WORD.finditer(t)], dtype=int).reshape(-1, 2)
        ):
            raise ValueError("Native token projections do not align with the original text")
        result[t] = (
            float(grades[i]),
            vectors[i],
            terminal[i],
            observed[span],
            context[span],
            lexical[span],
        )
    return result


def word_readouts(a, b, ma, mb, na, nb):
    ax, ah, at, ai, aw, al = na
    bx, bh, bt, bi, bw, bl = nb
    cross = aw[ai] @ bw[bi].T
    pa = np.flatnonzero(ai)
    pb = np.flatnonzero(bi)

    def coverage(direction, ps, pt, ss, ts, weights, lex, lost, all_valid):
        valid = any(all_valid) and len(pt) > 0
        if not valid:
            return np.zeros(11)
        sim = direction.max(axis=1)
        comparable = {ts[i] for i in pt}
        same = np.array([ss[i] in comparable for i in ps])
        hybrid = np.where(same, 1.0, sim)
        miss = lost[ps]
        lv = lex[ps]
        return np.r_[
            sim.mean(),
            np.average(sim, weights=weights[ps]),
            np.quantile(sim, 0.1),
            hybrid.mean(),
            np.average(hybrid, weights=weights[ps]),
            sim[miss].mean() if miss.any() else 1.0,
            lv.mean(),
            np.quantile(lv, 0.9),
            lv[miss].mean() if miss.any() else 0.0,
            lv[~miss].mean() if (~miss).any() else 0.0,
            np.mean(all_valid),
        ]

    ca = coverage(cross, pa, pb, a.stems, b.stems, a.idf, al, ma, ai)
    cb = coverage(cross.T, pb, pa, b.stems, a.stems, b.idf, bl, mb, bi)
    # Absolute terminal distances and distinct directions retain insertions and omissions.
    sem = np.r_[
        ax,
        bx,
        ax - bx,
        ah @ bh,
        ca,
        cb,
        cb - ca,
        np.abs(cb - ca),
        np.linalg.norm(at - bt),
        np.mean(np.abs(at - bt)),
    ]
    high = np.r_[bt, bt - at, np.abs(bt - at)]
    return sem, high


def build(task, extraction=None):
    text = sorted({raw for pair in task.passages for raw in pair})
    source = {r: passage(r) for r in text}
    neural = neural_info(extraction, text) if extraction else None
    vals = defaultdict(list)
    raw = defaultdict(list)
    for src, draft in task.passages:
        a, b = source[src], source[draft]
        x, base, ma, mb = comparison(a, b)
        vals["edit"].append(x)
        if neural:
            sem, high = word_readouts(a, b, ma, mb, neural[src], neural[draft])
            vals["neural"].append(sem)
            vals["directions"].append(high)
            base["readme"] = neural[src][0] - neural[draft][0]
            base["contextual_coverage"] = float((sem[4] + sem[15]) / 2)
        for k, v in base.items():
            raw[k].append(v)
    return {k: np.stack(r) for k, r in vals.items()}, {k: np.array(r) for k, r in raw.items()}
