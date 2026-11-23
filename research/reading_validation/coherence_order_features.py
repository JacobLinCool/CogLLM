"""Extract boundary-local statistics from printed source coordinates and the shared L4 feature bank."""

import argparse
import itertools
import json
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from research.reading_validation.coherence_order_data import (
    OUT,
    STATE,
    WORD,
    read,
    texts,
)
from research.reading_validation.local_forecast import digest

BASE_COLUMNS = (
    "prefix_log_words",
    "completed_unit",
    "local_log_words",
    "prior_log_words",
    "mean_letters",
    "long_word_fraction",
    "case_fraction",
    "numeric_fraction",
    "type_token_ratio",
    "comma_rate",
    "question",
    "discourse_initial",
    "previous_jaccard",
    "prefix_jaccard",
    "previous_match_fraction",
    "prefix_match_fraction",
    "previous_bigram_fraction",
    "prefix_type_recurrence",
)
PRED_COLUMNS = (
    "word_surprisal",
    "word_surprisal_sd",
    "upper_surprisal",
    "first_surprisal",
    "first_three_surprisal",
    "closing_surprisal",
    "surprisal_minus_previous",
    "initial_surprisal_minus_previous",
)
SEM_SUMMARY = (
    "cosine_previous",
    "cosine_prefix",
    "cosine_initial_previous",
    "cosine_closing_previous",
    "strongest_earlier_cosine",
    "recent_cosine_advantage",
    "mean_previous_attention",
    "initial_previous_attention",
    "peak_previous_attention",
    "attention_concentration",
    "attention_distance",
    "previous_available",
)
PROJ = np.r_[0:8, 32:40]
SEM_COLUMNS = (
    *[f"{kind}_{j}" for kind in ("mean", "last", "previous_mismatch") for j in PROJ],
    *SEM_SUMMARY,
)
CONNECT = frozenset(
    [
        "and",
        "but",
        "so",
        "then",
        "however",
        "meanwhile",
        "therefore",
        "nevertheless",
        "consequently",
        "instead",
        "therefore",
        "because",
        "although",
        "yet",
        "dunque",
        "ma",
        "poi",
        "però",
        "tuttavia",
        "quindi",
        "inoltre",
        "invece",
        "perciò",
        "e",
        "mentre",
        "sebbene",
    ]
)


def ratio(k, n):
    return k / max(1, n)


def jaccard(a, b):
    return ratio(len(a & b), len(a | b))


def cosine(a, b):
    scale = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / scale) if scale > 1e-9 else 0.0


def form(text, words, coords, a, b, atoms):
    current = np.flatnonzero((coords[:, 0] >= a) & (coords[:, 1] <= b))
    if not len(current):
        return None
    before = np.flatnonzero(coords[:, 1] <= a)
    prev = None
    for s, e in reversed(atoms):
        if s >= a:
            continue
        hit = np.flatnonzero((coords[:, 0] >= s) & (coords[:, 1] <= min(a, e)))
        if len(hit):
            prev = hit
            break
    if prev is None:
        prev = np.empty(0, dtype=int)
    current_words = [words[i] for i in current]
    near = [words[i] for i in prev]
    history = [words[i] for i in before]
    c, p, all_prev = map(set, (current_words, near, history))
    n = len(current_words)
    nt = len(before)
    bigram = set(itertools.pairwise(near))
    seg = text[a:b]
    position = sum(1 for s, e in atoms if e <= a)
    first = current_words[0]
    x = np.array(
        [
            np.log1p(nt),
            np.log1p(position),
            np.log1p(n),
            np.log1p(len(prev)),
            np.mean([len(w) for w in current_words]),
            np.mean([len(w) >= 8 for w in current_words]),
            np.mean([text[coords[i, 0]].isupper() for i in current]),
            np.mean([w.isnumeric() for w in current_words]),
            len(c) / n,
            seg.count(",") / n,
            "?" in seg,
            first in CONNECT,
            jaccard(c, p),
            jaccard(c, all_prev),
            np.mean([w in p for w in current_words]),
            np.mean([w in all_prev for w in current_words]),
            ratio(sum(v in bigram for v in itertools.pairwise(current_words)), n - 1),
            ratio(len(c & all_prev), len(c)),
        ],
        float,
    )
    return x, current, prev, before


def neural(u, c, p, before, atoms, a):
    coords, valid, surprisal, word, attention = u
    c, p, before = (q[valid[q]] for q in (c, p, before))
    if not len(c):
        return None
    S = surprisal[c]
    pc = surprisal[p] if len(p) else np.empty(0)
    prevmean = pc.mean() if len(pc) else 0.0
    q = np.array(
        [
            S.mean(),
            S.std(),
            np.quantile(S, 0.9),
            S[0],
            S[:3].mean(),
            S[-1],
            S.mean() - prevmean,
            S[:3].mean() - prevmean,
        ]
    )
    v = word[c].mean(0)
    previous = word[p].mean(0) if len(p) else np.zeros_like(v)
    overall = word[before].mean(0) if len(before) else np.zeros_like(v)
    visible = [j for j, (s, e) in enumerate(atoms) if e <= a]
    relevance = []
    for j in visible:
        b, e = atoms[j]
        ix = before[(coords[before, 0] >= b) & (coords[before, 1] <= e)]
        if len(ix):
            relevance.append(cosine(v, word[ix].mean(0)))
    strong = max(relevance, default=0.0)
    recent = cosine(v, previous)
    links = np.zeros(5)
    if visible:
        observed = attention[c][:, visible, :]
        m = observed[:, :, 0].mean(0)
        den = m.sum()
        links[:3] = [m[-1], observed[0, -1, 0], observed[:, -1, 1].max()]
        if den > 1e-9:
            dist = m / den
            links[3:] = [
                dist @ dist,
                dist @ np.arange(len(m) - 1, -1, -1) / max(1, len(m) - 1),
            ]
    sem = np.r_[
        v[PROJ],
        word[c[-1], PROJ],
        np.abs(v[PROJ] - previous[PROJ]),
        recent,
        cosine(v, overall),
        cosine(word[c[0]], previous),
        cosine(word[c[-1]], previous),
        strong,
        recent - strong,
        links,
        bool(visible),
    ]
    return np.r_[q, sem]


def run(archive=None):
    geo = read(OUT / "geometry.json")
    if digest(OUT / "geometry.json") != read(OUT / "geometry-seal.json")["geometry"]:
        raise ValueError("Changed study geometry")
    strings = texts()
    features, meta, missing = [], [], Counter()
    types = (
        BASE_COLUMNS
        if archive is None
        else (*BASE_COLUMNS, *PRED_COLUMNS, *SEM_COLUMNS)
    )
    dest = OUT / ("form-features.npz" if archive is None else "native-features.npz")
    rec = None
    with ExitStack() as stack:
        if archive is not None:
            rec = read(archive / "receipt.json")
            if (
                rec["source"] != read(STATE)["source"]
                or digest(archive / "features.npz") != rec["file"]
            ):
                raise ValueError("L4 source or extraction digest mismatch")
            z = stack.enter_context(
                np.load(archive / "features.npz", allow_pickle=False)
            )
        for key, r in sorted(geo.items()):
            if r["role"] == "omit":
                continue
            t = strings[key]
            atoms = r["segments"]
            matches = list(WORD.finditer(t))
            coordinates = np.array([m.span() for m in matches], int)
            words = [m.group().casefold() for m in matches]
            native = None
            if rec is not None:
                if key not in rec["articles"]:
                    reason = rec["skipped"][key]
                    missing[r["source"] + "/" + r["role"] + "/" + reason] += len(
                        r["boundaries"]
                    )
                    continue
                char = z[key + ".char"].astype(int)
                if (
                    not np.array_equal(coordinates, char[:, :2])
                    or rec["articles"][key]["sentences"] != atoms
                ):
                    raise ValueError(
                        "Native coordinates differ from the original words or segments"
                    )
                sur = z[key + ".surprisal"].astype(float)
                word = z[key + ".words"].astype(float)
                att = z[key + ".rel"].astype(float)
                if att.shape != (len(words), len(atoms), 2) or word.shape != (
                    len(words),
                    64,
                ):
                    raise ValueError("Unexpected native word representation")
                native = (coordinates, char[:, 4].astype(bool), sur, word, att)
            for q in r["boundaries"]:
                raw = form(t, words, coordinates, q["start"], q["end"], atoms)
                if raw is None:
                    missing[r["source"] + "/" + r["role"] + "/no_printed_words"] += 1
                    continue
                x, *idx = raw
                if native is not None:
                    vec = neural(native, *idx, atoms, q["start"])
                    if vec is None:
                        missing[
                            r["source"] + "/" + r["role"] + "/no_certified_word"
                        ] += 1
                        continue
                    x = np.r_[x, vec]
                if x.shape != (len(types),) or not np.isfinite(x).all():
                    raise ValueError("Invalid boundary features")
                features.append(x)
                meta.append(
                    (q["id"], key, r["group"], r["source"], r["role"], r["source_id"])
                )
    ids = np.asarray(meta)
    if len(set(ids[:, 0])) != len(meta):
        raise ValueError("Duplicate native boundary ID")
    np.savez_compressed(
        dest, ids=ids, x=np.asarray(features), features=np.asarray(types)
    )
    note = {
        "native": archive is not None,
        "archive": None if rec is None else rec["file"],
        "features": digest(dest),
        "columns": len(types),
        "boundaries": len(meta),
        "documents": len(set(ids[:, 1])),
        "omitted_boundaries": dict(missing),
    }
    dest.with_suffix(".json").write_text(json.dumps(note, indent=2) + "\n")
    print(note)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path)
    run(p.parse_args().archive)
