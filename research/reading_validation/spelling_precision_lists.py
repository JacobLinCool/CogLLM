"""Recompute precision intervals by resampling the original intact Peelle assignment lists."""

import csv
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np

from research.backbone_transfer.extract import digest, read_json, write_json
from research.reading_validation.reader_history_external import get_geometry, sources
from research.reading_validation.spelling_human_precision import METHODS, OUT

PLAN = Path(__file__).with_name("spelling-human-list-protocol.md")


def main():
    stem_file = OUT / "stem-precision.csv"
    primary = read_json(OUT / "scores.json")
    if read_json(OUT / "input-lock.json")["source_files"] != sources():
        raise ValueError("Peelle source files changed")
    if primary["per_stem_sha256"] != digest(stem_file):
        raise ValueError("Location-specific peer errors changed")
    geometry, _ = get_geometry()
    lists = sorted(
        {tuple(sorted(s for s, _, _ in info[3])) for info in geometry.values()}
    )
    sizes = np.array([len(s) for s in lists])
    if Counter(sizes) != {50: 61, 35: 1}:
        raise ValueError(
            "Recovered lists differ from the original 61 large and one small assignment"
        )
    membership = {s: i for i, row in enumerate(lists) for s in row}
    if len(membership) != sizes.sum() or len(membership) != 3085:
        raise ValueError("At least one printed stem spans multiple lists")
    with stem_file.open() as f:
        rows = list(csv.DictReader(f))
    if set(membership) != {r["stem_sha256"] for r in rows} or len(rows) != len(
        membership
    ):
        raise ValueError(
            "The previous reader-split measurements do not cover the original lists"
        )
    totals = np.zeros((len(lists), len(METHODS)))
    for r in rows:
        totals[membership[r["stem_sha256"]]] += [
            float(r[name + "_half_mae"]) for name in METHODS
        ]
    mean = totals.sum(0) / sizes.sum()
    if not np.allclose(
        mean,
        [primary["half_results"][m]["mae"]["estimate"] for m in METHODS],
        rtol=0,
        atol=1e-12,
    ):
        raise ValueError("Clustering changed the point estimate")
    rng = np.random.default_rng(715515)
    draws = rng.multinomial(len(lists), np.full(len(lists), 1 / len(lists)), size=10000)
    sample = (draws @ totals) / (draws @ sizes)[:, None]
    results = {}
    for i, name in enumerate(METHODS):
        results[name] = {
            "mae": mean[i],
            "list_ci95": np.quantile(sample[:, i], [0.025, 0.975]).tolist(),
        }
    contrasts = {}
    for i, j in combinations(range(len(METHODS)), 2):
        contrasts[f"{METHODS[j]}_minus_{METHODS[i]}"] = {
            "mae_difference": mean[j] - mean[i],
            "list_ci95": np.quantile(
                sample[:, j] - sample[:, i], [0.025, 0.975]
            ).tolist(),
        }
    write_json(
        OUT / "assignment-list-intervals.json",
        {
            "original_disjoint_lists": len(lists),
            "fifty_stem_lists": 61,
            "thirty_five_stem_lists": 1,
            "union_of_retained_stems": len(membership),
            "list_bootstrap_draws": len(draws),
            "per_method": results,
            "paired_contrasts": contrasts,
            "source_lock": digest(OUT / "input-lock.json"),
            "stem_errors": digest(stem_file),
            "original_scores": digest(OUT / "scores.json"),
            "protocol": digest(PLAN),
            "scorer": digest(Path(__file__)),
        },
    )
    print(results)
    print(contrasts["conditional_minus_peer"])


if __name__ == "__main__":
    main()
