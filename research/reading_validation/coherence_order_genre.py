"""Recover identifiable genre evidence and estimate passage genre without Task 2 test ratings."""

import json
from collections import Counter, defaultdict
from itertools import pairwise

import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import expit

from research.reading_validation.coherence_order_data import (
    ITALIAN,
    OUT,
    WORD,
    csv_rows,
    read,
    texts,
)
from research.reading_validation.coherence_order_estimator import equal_group_weights
from research.reading_validation.local_forecast import digest


def lex(text):
    return tuple(w.group().casefold() for w in WORD.finditer(text))


def terms(tokens):
    return (
        *tokens,
        *(a + " " + b for a, b in pairwise(tokens)),
    )


def material():
    rows = []
    for genre in ("ted", "wiki"):
        for split in ("train", "test"):
            for r in csv_rows(ITALIAN / f"task1-{genre}-{split}.tsv"):
                rows.append(
                    {
                        "id": r["ID"],
                        "group": genre + "/" + r["ID"].split("#")[0],
                        "genre": genre,
                        "split": split,
                        "tokens": lex(r["PROMPT"]),
                    }
                )
    return rows


def match(train_rows, t2):
    queries = {}
    roles = {}
    for k, t in t2.items():
        queries[k] = t["tokens"]
        roles[k] = "t2-" + t["role"]
    for t in train_rows:
        if t["split"] == "test":
            queries[t["id"]] = t["tokens"]
            roles[t["id"]] = "t1-test"
    index = defaultdict(list)
    for k, words in queries.items():
        for i in range(len(words) - 11):
            index[hash(words[i : i + 12])].append((k, i))
    links = defaultdict(set)
    exclude = set()
    for r in train_rows:
        words = r["tokens"]
        overlap = defaultdict(int)
        for j in range(len(words) - 11):
            gram = words[j : j + 12]
            for k, i in index.get(hash(gram), ()):
                if gram == queries[k][i : i + 12]:
                    if r["split"] == "train" and roles[k] in ("t1-test", "t2-held"):
                        exclude.add(r["group"])
                    if k in t2:
                        overlap[k] |= ((1 << 12) - 1) << i
        for k, count in overlap.items():
            if count.bit_count() >= 24:
                links[k].add((r["genre"], r["group"] if r["split"] == "train" else ""))
    return links, exclude


def matrices(train, tests):
    df = Counter()
    for r in train:
        df.update(set(terms(r["tokens"])))
    vocab = {k: i for i, k in enumerate(sorted(w for w, c in df.items() if c >= 3))}
    scale = np.asarray([1 + np.log((len(train) + 1) / (df[k] + 1)) for k in vocab])
    banks = []
    for pool in [train, *tests]:
        indices = []
        vals = []
        rowptr = [0]
        for r in pool:
            counts = Counter(vocab[k] for k in terms(r["tokens"]) if k in vocab)
            j = np.fromiter(counts.keys(), dtype=int)
            v = np.log1p(np.fromiter(counts.values(), dtype=float)) * scale[j]
            denom = np.linalg.norm(v)
            if denom > 0:
                v /= denom
            indices.extend(j)
            vals.extend(v)
            rowptr.append(len(indices))
        banks.append(
            sparse.csr_matrix((vals, indices, rowptr), shape=(len(pool), len(vocab)))
        )
    return banks


def compute():
    geo = read(OUT / "geometry.json")
    content = texts()
    t2 = {
        k: {"tokens": lex(content[k]), "role": r["role"], "id": r["source_id"]}
        for k, r in geo.items()
        if r["source"] == "italian"
    }
    data = material()
    links, exclude = match(data, t2)
    train = [r for r in data if r["split"] == "train" and r["group"] not in exclude]
    valid = [r for r in data if r["split"] == "test"]
    target = sorted(t2)
    (X, V, T) = matrices(train, [valid, [t2[k] for k in target]])
    y = np.array([r["genre"] == "wiki" for r in train], float)
    w = equal_group_weights([r["group"] for r in train])
    for c in (0.0, 1.0):
        w[y == c] *= 0.5 / w[y == c].sum()
    ridge = 0.0003

    def obj(q):
        z = X @ q[:-1] + q[-1]
        a = w * (expit(z) - y)
        return w @ (np.logaddexp(0, z) - y * z) + 0.5 * ridge * q[:-1] @ q[:-1], np.r_[
            X.T @ a + ridge * q[:-1], a.sum()
        ]

    fit = minimize(
        obj,
        np.zeros(X.shape[1] + 1),
        jac=True,
        method="L-BFGS-B",
        options={"ftol": 1e-13, "gtol": 1e-8, "maxiter": 400, "maxcor": 8},
    )
    grad = float(np.max(abs(obj(fit.x)[1])))
    if not fit.success or grad > 2e-6:
        raise ValueError("Independent genre estimator failed to converge")
    pred = expit(V @ fit.x[:-1] + fit.x[-1])
    actual = np.array([r["genre"] == "wiki" for r in valid])
    tp = np.sum((pred >= 0.5) & actual)
    tn = np.sum((pred < 0.5) & ~actual)
    tprobs = expit(T @ fit.x[:-1] + fit.x[-1])
    out = {}
    for k, p in zip(target, tprobs, strict=True):
        matchset = links[k]
        genres = {g for g, _ in matchset}
        ids = {src for _, src in matchset if src}
        if len(genres) > 1:
            genre = None
        else:
            genre = next(iter(genres)) if genres else None
        out[t2[k]["id"]] = {
            "wiki_probability": float(p),
            "matched_genre": genre,
            "matched_source": next(iter(ids)) if len(ids) == 1 and genre else None,
            "role": t2[k]["role"],
        }
    held = [r for r in out.values() if r["role"] == "held"]
    n = sum(r["matched_genre"] is not None for r in held)
    ack = sum(
        r["matched_genre"] == ("wiki" if r["wiki_probability"] >= 0.5 else "ted")
        for r in held
        if r["matched_genre"]
    )
    path = OUT / "genre-forecasts.json"
    path.write_text(json.dumps(out, separators=(",", ":")) + "\n")
    rec = {
        "outcome_blind": True,
        "files": {p.name: digest(p) for p in ITALIAN.glob("task1-*.tsv")},
        "source": digest(path),
        "held_task1": {
            "prompts": len(valid),
            "wiki": int(actual.sum()),
            "true_wiki": int(tp),
            "true_ted": int(tn),
            "balanced_accuracy": float(
                0.5 * (tp / actual.sum() + tn / (~actual).sum())
            ),
            "accuracy": float((tp + tn) / len(valid)),
            "logloss": float(np.mean(-np.where(actual, np.log(pred), np.log1p(-pred)))),
        },
        "task1_training_prompts_after_overlap_exclusion": len(train),
        "removed_task1_document_groups": len(exclude),
        "task2_test_physical_genres": dict(
            Counter(r["matched_genre"] or "unidentified" for r in held)
        ),
        "task2_test_physical_source_ids": sum(
            r["matched_source"] is not None for r in held
        ),
        "task2_test_genre_correct_on_matched": ack,
        "task2_test_genre_matched": n,
        "fit_gradient": grad,
        "vocabulary": X.shape[1],
        "ridge": ridge,
        "iterations": fit.nit,
    }
    (OUT / "genre-seal.json").write_text(json.dumps(rec, indent=2) + "\n")
    print(rec)


if __name__ == "__main__":
    compute()
