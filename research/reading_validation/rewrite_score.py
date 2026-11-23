"""Evaluate frozen revision preferences against the ASSET workers' judgments."""

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from research.reading_validation.external_input import digest_string
from research.reading_validation.joint_data import DEFAULT_CACHE, digest, read


METHODS = ("adaptive.qwen", "adaptive.surface", "zero.qwen", "zero.surface", "length", "always_asset")


def interval(estimate, draws):
    return dict(estimate=float(estimate), interval95=np.quantile(draws, [.025, .975]).tolist())


def preference(method, a, b, values):
    if method == "always_asset":
        return 1.0
    return float((1 + np.sign(values[method][b] - values[method][a])) / 2)


def evaluate_votes(rows, idx, values):
    totals, votes, tie_count, represented = {}, defaultdict(list), Counter(), set()
    n = 0
    for r in rows:
        group = r["sentence_id"]
        represented.add(group)
        a,b = (idx[digest_string(r[k])] for k in ("asset_simplification","other_simplification"))
        winner = r["winner"].strip().lower()
        n += 1
        if winner in {"similar", "tie", "equal", "both", "same"}:
            tie_count[group] += 1
            continue
        if winner not in {"asset", r["other_dataset"].strip().lower()}:
            raise ValueError(f"Unknown ASSET winner code {winner!r}")
        decision = [preference(m, a, b, values) for m in METHODS]
        votes[group].append(decision if winner == "asset" else 1 - np.asarray(decision))
    if not votes:
        raise ValueError("No directional votes in selected stratum")
    group_mean = np.stack([np.mean(votes[g],axis=0) for g in sorted(votes)])
    rng = np.random.default_rng(40617)
    selection = rng.integers(len(votes), size=(10000,len(votes)))
    draws = np.stack([group_mean[:,i][selection].mean(axis=1) for i in range(len(METHODS))],axis=1)
    for i, m in enumerate(METHODS):
        totals[m] = interval(group_mean[:,i].mean(), draws[:,i])
    diffs = {m:interval((group_mean[:,0]-group_mean[:,i]).mean(), draws[:,0]-draws[:,i])
             for i,m in enumerate(METHODS[1:], 1)}
    return dict(rows=n, ties=sum(tie_count.values()), families_all=len(represented),
                families_directional=len(votes), agreement=totals, qwen_minus=diffs)


def absolute_ratings(rows, idx, values):
    by_pair = defaultdict(list)
    for r in rows:
        a,b = (idx[digest_string(r[k])] for k in ("original","simplification"))
        rating = float(r["rating"])
        if not 0 <= rating <= 100:
            raise ValueError("Unknown ASSET rating scale")
        by_pair[(a,b)].append(rating)
    pairs = sorted(by_pair)
    y = np.array([np.mean(by_pair[k]) for k in pairs])
    a,b = np.asarray(pairs).T
    methods = {}
    for method in METHODS[:-1]:
        z = values[method]
        methods[method] = dict(difficulty_reduction=float(spearmanr(z[a]-z[b],y).statistic),
                               final_ease=float(spearmanr(-z[b],y).statistic))
    return dict(pairs=len(pairs), rows=len(rows), methods=methods)


def run(forecast, cache, output):
    inputs = read(forecast.parent/"inputs.json")
    seal = read(forecast/"seal.json")
    if digest(forecast/"scores.npz") != seal["scores_sha256"] or digest(forecast.parent/"inputs.json") != seal["data_sha256"]:
        raise ValueError("Prejudgment seal changed")
    loaded = {}
    for name, expected in inputs["files"].items():
        file = cache/"reading-validation"/name
        if digest(file) != expected:
            raise ValueError("Human judgment release changed")
        with file.open(newline="") as f:
            data = list(csv.DictReader(f))
        loaded["pair" if "other_dataset" in data[0] else "rating"] = data
    with np.load(forecast/"scores.npz", allow_pickle=False) as arrays:
        vals = {k:arrays[k] for k in arrays.files}
    idx = {k:i for i,k in enumerate(vals["id"])}
    exclude = inputs["source_exclusions"]
    pair = [r for r in loaded["pair"] if digest_string(r["source"]) not in exclude]
    rating = [r for r in loaded["rating"] if digest_string(r["original"]) not in exclude]
    evaluations, absolute = {}, {}
    for aspect in sorted({r["aspect"] for r in pair}):
        subset = [r for r in pair if r["aspect"] == aspect]
        sources = sorted({r["other_dataset"] for r in subset})
        both_single = [r for r in subset if all(vals["sentence_count"][idx[digest_string(r[k])]] == 1
                       for k in ("other_simplification","asset_simplification"))]
        sets = {"pooled": subset,"both_one_sentence": both_single} | {
            s: [r for r in subset if r["other_dataset"] == s] for s in sources}
        evaluations[aspect] = {s:evaluate_votes(rows,idx,vals) for s,rows in sets.items() if rows}
    for aspect in sorted({r["aspect"] for r in rating}):
        absolute[aspect] = absolute_ratings([r for r in rating if r["aspect"]==aspect],idx,vals)
    result = dict(pairwise=evaluations,absolute=absolute,forecast_sha256=digest(forecast/"seal.json"),
                  excluded_source_strings=len(exclude))
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({a:e["pooled"]["agreement"]["adaptive.qwen"] for a,e in evaluations.items()}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--forecast", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    run(a.forecast,a.cache,a.output)
