"""Cross-fit source-balanced pair losses for scientific information needs."""

import csv
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit

from research.reading_validation.joint_data import digest, lines
from research.reading_validation.local_help_inference import DESIGNS, design
from research.reading_validation.local_help_prepare import CACHE, HERE, OUT, code

OUTCOMES = ("definition_over_example", "definition", "example")
RIDGE = 0.05


def data():
    src = CACHE / "jargon/source.csv"
    seal = json.loads((OUT / "geometry.json").read_bytes())
    if (
        digest(src) != seal["source_sha256"]
        or digest(OUT / "features.jsonl") != seal["features_sha256"]
    ):
        raise ValueError("Guo geometry or registered survey changed")
    features = {(r["paper"], r["term_key"]): r for r in lines(OUT / "features.jsonl")}
    reserved = set(seal["reserved"])
    all_ = defaultdict(set)
    with src.open(newline="", encoding="utf8") as stream:
        for r in csv.DictReader(stream):
            paper = "jargon/" + r["paper_sha"]
            if paper in reserved:
                continue
            key = (paper, code(r["entity"]))
            if key not in features:
                continue
            pre = int(float(r["familiarity_before"]))
            if not 0 <= pre <= 4:
                raise ValueError("Prior scale differs")
            d, e = (
                {"Yes": 1, "No": 0}[r[k]]
                for k in ("additional_definition", "additional_example")
            )
            all_[paper, key[1], code(r["annotator"])].add((pre, d, e))
    rows = []
    for (paper, term, reader), observations in sorted(all_.items()):
        if len(observations) != 1:
            continue
        ((prior, def_, example),) = observations
        f = features[paper, term]
        rows.append(
            {
                "paper": paper,
                "term": term,
                "reader": reader,
                "prior": prior,
                "fold": f["article_fold"],
                "features": f["features"],
                "targets": (def_ if def_ != example else -1, def_, example),
            }
        )
    return rows, sum(len(v) != 1 for v in all_.values())


def pairs(rows, y):
    cells = defaultdict(list)
    for i, r in enumerate(rows):
        if y[i] >= 0:
            cells[r["paper"], r["reader"]].append(i)
    pos, neg = [], []
    for items in cells.values():
        for i, j in combinations(items, 2):
            if y[i] != y[j]:
                a, b = (i, j) if y[i] else (j, i)
                pos.append(a)
                neg.append(b)
    return np.array(pos), np.array(neg)


def observation_weights(paper, reader):
    cells = Counter(zip(paper, reader))
    n_readers = Counter(a for a, _ in cells)
    return np.array(
        [
            1 / (len(n_readers) * n_readers[a] * cells[a, p])
            for a, p in zip(paper, reader)
        ]
    )


def fit(x, y, paper, reader, subset, pos, neg):
    point = subset & (y >= 0)
    pweights = observation_weights(paper[point], reader[point])
    mean = np.average(x[point], weights=pweights, axis=0)
    scale = np.sqrt(
        np.maximum(np.average((x[point] - mean) ** 2, weights=pweights, axis=0), 1e-6)
    )
    z = (x - mean) / scale
    keep = subset[pos] & subset[neg]
    u, v = pos[keep], neg[keep]
    if not len(u):
        raise ValueError("No training pairs")
    d = z[u] - z[v]
    w = observation_weights(paper[u], reader[u])
    coef = np.zeros(x.shape[1])
    eye = np.eye(len(coef))
    for iteration in range(30):
        odds = expit(-d @ coef)
        grad = -(w * odds) @ d + RIDGE * coef
        if np.max(np.abs(grad)) <= 1e-7:
            break
        hess = d.T @ ((w * odds * (1 - odds))[:, None] * d) + RIDGE * eye
        step = np.linalg.solve(hess, grad)
        before = w @ np.logaddexp(0, -d @ coef) + 0.5 * RIDGE * coef @ coef
        for n in range(20):
            new = coef - (0.5**n) * step
            obj = w @ np.logaddexp(0, -d @ new) + 0.5 * RIDGE * new @ new
            if obj <= before + 1e-13:
                coef = new
                break
        else:
            raise ArithmeticError("Pairwise loss line search failed")
    else:
        raise ArithmeticError("Pairwise estimator did not converge")
    rank_score = z @ coef
    target = pweights @ y[point]
    intercept = brentq(
        lambda b: pweights @ expit(rank_score[point] + b) - target, -150, 150
    )
    return rank_score + intercept, {
        "center": mean.tolist(),
        "scale": scale.tolist(),
        "coef": coef.tolist(),
        "intercept": float(intercept),
        "max_gradient": float(np.max(np.abs(grad))),
        "updates": iteration,
        "points": int(point.sum()),
        "pairs": len(u),
    }


def run():
    records, conflicts = data()
    x = np.array([r["features"] for r in records])
    paper, reader, fold, prior = (
        np.array([r[k] for r in records]) for k in ("paper", "reader", "fold", "prior")
    )
    matrices = {arm: design(x, prior, arm) for arm in DESIGNS}
    tests = [(person, i) for person in sorted(set(reader)) for i in sorted(set(fold))]
    fitted, forecasts, checks = {}, {}, []
    for target, endpoint in enumerate(OUTCOMES):
        y = np.array([r["targets"][target] for r in records])
        pos, neg = pairs(records, y)
        fitted[endpoint] = {}
        for arm, xx in matrices.items():
            predictions = np.full(len(y), np.nan)
            for person, group in tests:
                subset = (reader != person) & (fold != group)
                test = (reader == person) & (fold == group)
                if not test.any():
                    continue
                pred, model = fit(xx, y, paper, reader, subset, pos, neg)
                predictions[test] = pred[test]
                checks.append((model["max_gradient"], model["updates"]))
            if not np.isfinite(predictions).all():
                raise ArithmeticError("Omitted a held-out reader or article")
            _full, model = fit(xx, y, paper, reader, np.ones(len(y), bool), pos, neg)
            checks.append((model["max_gradient"], model["updates"]))
            fitted[endpoint][arm] = model
            forecasts[endpoint + "." + arm] = predictions
        forecasts[endpoint + ".fixed_lexical"] = x[:, 0]
    p = OUT / "forecasts.npz"
    np.savez_compressed(
        p,
        key=[code("/".join((r["paper"], r["term"], r["reader"]))) for r in records],
        **forecasts,
    )
    weights = OUT / "weights.json"
    weights.write_text(json.dumps(fitted, indent=2, allow_nan=False) + "\n")
    receipt = {
        "protocol_sha256": digest(HERE / "local-help-protocol.md"),
        "data_sha256": digest(OUT / "geometry.json"),
        "forecast_sha256": digest(p),
        "weights_sha256": digest(weights),
        "observation_rows": len(records),
        "conflicting_duplicate_rows_excluded": conflicts,
        "fits": len(checks),
        "max_absolute_gradient": max(a for a, _ in checks),
        "max_iterations": max(b for _, b in checks),
        "fit_sha256": digest(Path(__file__)),
        "score_sha256": digest(HERE / "local_help_score.py"),
        "serving_sha256": digest(HERE / "local_help_inference.py"),
    }
    (OUT / "forecast-seal.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        {k: receipt[k] for k in ("fits", "max_absolute_gradient", "observation_rows")}
    )


if __name__ == "__main__":
    run()
