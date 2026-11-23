"""Fit strictly Danish-source references and audit each untouched release test passage."""

import json
from collections import defaultdict
from itertools import pairwise

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsmr

from research.reading_validation.argument_data import sentences
from research.reading_validation.coherence_order_data import OUT as IT
from research.reading_validation.coherence_order_data import read, texts
from research.reading_validation.coherence_order_estimator import equal_group_weights
from research.reading_validation.coherence_order_genre import matrices
from research.reading_validation.danish_coherence_data import (
    OUT,
    PLAN,
    components,
    entries,
    grams,
    words,
)
from research.reading_validation.emtec_material import exposures
from research.reading_validation.local_forecast import digest

COORD = frozenset(
    ["og", "eller", "men", "for", "samt", "så", "enten", "hverken", "både"]
)
SUBORD = frozenset(
    [
        "at",
        "hvis",
        "fordi",
        "når",
        "da",
        "inden",
        "før",
        "end",
        "som",
        "mens",
        "selvom",
        "eftersom",
        "hvorimod",
        "hvorvidt",
        "medmindre",
        "skønt",
        "såfremt",
        "idet",
        "om",
        "hvor",
    ]
)


def xform(r):
    t = r["text"]
    atoms = sentences(t)
    w = words(t)
    groups = [set(words(t[a:b])) for a, b in atoms]
    jac = [len(a & b) / max(1, len(a | b)) for a, b in pairwise(groups)]
    n = len(w)
    m = len(atoms)
    long = sum(len(x) > 6 for x in w) / n
    return [
        r["domain"] == "wikipedia",
        np.log1p(n),
        np.log1p(m),
        long,
        n / m + 100 * long,
        sum(x in COORD for x in w) / n,
        sum(x in SUBORD for x in w) / n,
        np.mean(jac) if jac else 0.0,
    ]


def ridge(train, test, y, w, reg):
    m = sparse.hstack([np.ones((len(y), 1)), train], format="csr")
    z = sparse.hstack([np.ones((test.shape[0], 1)), test], format="csr")
    pen = np.r_[0.0, reg]
    design = sparse.vstack(
        [m.multiply(np.sqrt(w)[:, None]), sparse.diags(np.sqrt(pen), format="csr")],
        format="csr",
    )
    goal = np.r_[np.sqrt(w) * y, np.zeros(len(pen))]
    sol = lsmr(design, goal, atol=1e-12, btol=1e-12, maxiter=1400)
    grad = float(abs(design.T @ (design @ sol[0] - goal)).max())
    if grad > 1e-7 or sol[1] not in (1, 2):
        raise ValueError(f"Target-supervised reference failed convergence at {grad}")
    return np.clip(z @ sol[0], 1.0, 3.0), {
        "normal_residual": grad,
        "iterations": sol[2],
        "features": len(pen) - 1,
    }


def fit_baselines():
    src = list(entries("train", labels=True))
    test = list(entries("test"))
    group = components(src, test)
    held = {group[r["id"]] for r in test}
    obs = defaultdict(list)
    unique = {}
    for r in src:
        key = r["id"]
        if group[key] in held:
            continue
        if not 1 <= r["rating"] <= 3:
            raise ValueError("Unexpected Danish training scale")
        unique[key] = r
        obs[key].append(r["rating"])
    train = [unique[k] for k in sorted(unique)]
    test = sorted(test, key=lambda r: r["id"])
    keys = [r["id"] for r in test]
    y = np.array([np.mean(obs[r["id"]]) for r in train])
    w = equal_group_weights([r["domain"] for r in train])
    xt = np.array([xform(r) for r in train])
    xs = np.array([xform(r) for r in test])
    mu = w @ xt
    sd = np.sqrt(w @ ((xt - mu) ** 2))
    sd = np.where(sd > 1e-6, sd, 1.0)
    u, v = sparse.csr_matrix((xt - mu) / sd), sparse.csr_matrix((xs - mu) / sd)
    bas = {}
    record = {}
    bas["danish_form"], record["danish_form"] = ridge(
        u, v, y, w, np.full(u.shape[1], 0.01)
    )
    for r in [*train, *test]:
        r["tokens"] = words(r["text"])
    X, T = matrices(train, [test])
    U = sparse.hstack([u, X], format="csr")
    V = sparse.hstack([v, T], format="csr")
    bas["danish_lexical"], record["danish_lexical"] = ridge(
        U, V, y, w, np.r_[np.full(u.shape[1], 0.01), np.full(X.shape[1], 0.001)]
    )
    print(record)
    out = OUT / "local-controls.npz"
    np.savez_compressed(out, keys=keys, **bas)
    rec = {
        "control": digest(out),
        "protocol": digest(PLAN),
        "source_training_unique": len(train),
        "merged_duplicate_rows": sum(len(v) - 1 for v in obs.values()),
        "source_with_conflicting_repeats": sum(len(set(v)) > 1 for v in obs.values()),
        "official_test": len(test),
        "test_grades_used": False,
        "fits": record,
    }
    (OUT / "control-seal.json").write_text(json.dumps(rec, indent=2) + "\n")


def exposure():
    test = list(entries("test"))
    bad, note = exposures([r["text"] for r in test])
    target = defaultdict(set)
    for r in test:
        for sh in grams(r["text"]):
            target[sh].add(r["id"])
    content = texts()
    bank = read(IT / "geometry.json")
    found = defaultdict(set)
    for k, meta in bank.items():
        if meta["role"] not in ("fit", "dev"):
            continue
        for sh in grams(content[k]):
            found[meta["source"]] |= target.get(sh, set())
    (OUT / "exposure.json").write_text(
        json.dumps(
            {
                "historical_source_panels": note,
                "matching_danish_test_ids": [test[i]["id"] for i in sorted(bad)],
                "italian_english_readout_source_overlap": {
                    k: sorted(v) for k, v in found.items()
                },
                "backbone_pretraining": "unknown",
            },
            indent=2,
        )
        + "\n"
    )
    print("Prior overlap", len(bad), {k: len(v) for k, v in found.items()})


if __name__ == "__main__":
    fit_baselines()
    exposure()
