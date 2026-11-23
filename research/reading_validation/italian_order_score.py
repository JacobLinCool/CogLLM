"""Grade actual Italian human passage ratings and their separately measured structural sensitivity."""

import argparse
import json
from collections import Counter

import numpy as np

from research.reading_validation.coherence_order_data import OUT as FIT
from research.reading_validation.coherence_order_data import read
from research.reading_validation.coherence_order_fit import source_targets
from research.reading_validation.coherence_order_genre_score import (
    group_sample,
    metrics,
)
from research.reading_validation.coherence_order_score import point
from research.reading_validation.italian_order_data import OUT
from research.reading_validation.local_forecast import digest


def average(x, c):
    den = c.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (c @ x) / den


def describe(u):
    return point(u[0], u[1:])


def observed():
    if read(FIT / "scores.json")["forecast_seal"] != digest(FIT / "forecast/seal.json"):
        raise ValueError(
            "The original independent human confirmation must be graded first"
        )
    data = {
        k: v for k, v in read(OUT / "provenance.json").items() if v["split"] == "held"
    }
    human = source_targets(held=True)["italian"]
    with np.load(FIT / "forecast/predictions.npz", allow_pickle=False) as m:
        idx = {k: i for i, k in enumerate(m["italian.keys"])}
        ks = sorted(set(idx) & set(data))
        arms = {
            k: m["italian." + k][[idx[i] for i in ks]]
            for k in ("adaptive", "representation", "shared", "surprisal", "form")
        }
    arms["human"] = np.array([human[k] for k in ks])
    groups = np.array([data[k]["source"] for k in ks])
    types = np.array([data[k]["type"] for k in ks])
    c = group_sample(groups)
    out = {}
    for arm, v in arms.items():
        real = types == "original"
        alter = ~real
        diff = average(v[real], c[:, real]) - average(v[alter], c[:, alter])
        srcs = [
            k
            for k in sorted(set(groups))
            if np.any((groups == k) & real) and np.any((groups == k) & alter)
        ]
        by = np.array(
            [
                v[(groups == k) & real].mean() - v[(groups == k) & alter].mean()
                for k in srcs
            ]
        )
        cdoc = group_sample(np.array(srcs))
        score = {
            "mean_original_minus_altered": describe(diff),
            "mean_of_within_source_differences": describe(average(by, cdoc)),
            "comparable_sources": len(srcs),
            "by_authentic_type": {
                g: {
                    "mean": float(v[types == g].mean()),
                    "passages": int(sum(types == g)),
                }
                for g in sorted(set(types))
            },
        }
        if arm != "human":
            metric = metrics(arms["human"], v, c)
            score["spearman_vs_human"] = point(*metric["spearman"])
        out[arm] = score
    out = {
        "passages": len(ks),
        "sources": len(set(groups)),
        "genre_count": dict(Counter(data[k]["domain"] for k in ks)),
        "models": out,
        "public_document_pair_count": len(srcs),
        "registered_inputs": digest(OUT / "inputs.json"),
    }
    (OUT / "authentic-scores.json").write_text(
        json.dumps(out, indent=2, allow_nan=False) + "\n"
    )
    print("Authentic human contrast", out["models"]["human"])


def intervention():
    path = OUT / "order-differences.json"
    seal = read(OUT / "forecast-seal.json")
    if digest(path) != seal["estimates"]:
        raise ValueError("Unknown intervention predictions")
    records = read(path)

    def grade(rows, measure):
        records = [r for r in rows if r[measure] is not None]
        if not records:
            return {"pairs": 0}
        source = np.array([r["group"] for r in records])
        all_arms = set().union(*(r[measure] for r in records))
        results = {}
        for arm in sorted(all_arms):
            use = [
                (r["group"], r[measure][arm], r[measure]["adaptive"])
                for r in records
                if arm in r[measure]
            ]
            g, v, a = zip(*use, strict=True)
            v = np.array(v)
            a = np.array(a)
            w = group_sample(np.asarray(g))
            # Computational precision at the native affine output must not break mathematically tied permutations.
            score = lambda d: np.where(d > 1e-10, 1.0, np.where(d < -1e-10, 0.0, 0.5))
            results[arm] = {
                "pairs": len(v),
                "primary_correct": describe(average(score(v), w)),
                "paired_mean_shift": describe(average(v, w)),
                "adaptive_gain_over_arm_on_matched_pairs": describe(
                    average(score(a) - score(v), w)
                ),
            }
        return {"pairs": len(records), "sources": len(set(source)), "arms": results}

    panels = {}
    for tag, rs in (
        ("all", records),
        (
            "genuine_restorations",
            [r for r in records if r["observed_type"] == "transposed"],
        ),
        ("designed_swaps", [r for r in records if r["observed_type"] == "original"]),
    ):
        panels[tag] = {
            "passage": grade(rs, "canonical_minus_permuted_passage"),
            "first_disrupted_link": grade(
                rs, "permuted_minus_canonical_break_probability"
            ),
            "both_noninitial": grade(
                [r for r in rs if r["canonical_has_predecessor"]],
                "permuted_minus_canonical_break_probability",
            ),
        }
    (OUT / "structural-scores.json").write_text(
        json.dumps(panels, indent=2, allow_nan=False) + "\n"
    )
    print("Paired order", panels["all"]["passage"]["arms"]["adaptive"])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--authentic", action="store_true")
    p.add_argument("--structural", action="store_true")
    a = p.parse_args()
    if a.authentic:
        observed()
    if a.structural:
        intervention()
