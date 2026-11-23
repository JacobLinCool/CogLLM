"""Grade unseen Danish ordinal coherence within each authentic domain."""

import json
from collections import Counter
from pathlib import Path

import numpy as np

from research.reading_validation.coherence_order_genre_score import (
    metrics,
)
from research.reading_validation.coherence_order_score import point
from research.reading_validation.danish_coherence_data import OUT, PLAN, entries
from research.reading_validation.local_forecast import digest


def concordance(y, p, counts):
    """Each pair from distinct ordinal classes is counted once, ties in predictions receive half."""
    if not set(np.unique(y)) <= {1, 2, 3}:
        raise ValueError("Unexpected Danish ordered scale")
    _, inv = np.unique(p, return_inverse=True)
    order = np.argsort(inv)
    start = np.r_[0, 1 + np.flatnonzero(np.diff(inv[order]))]
    values = []
    for start_row in range(0, len(counts), 160):
        batch = counts[start_row : start_row + 160]
        mass = [
            np.add.reduceat(batch[:, order] * (y[order] == g), start, axis=1)
            for g in (1, 2, 3)
        ]
        denom = np.zeros(len(batch))
        num = np.zeros(len(batch))
        for hi in range(1, 3):
            for lo in range(hi):
                below = np.cumsum(mass[lo], axis=1) - 0.5 * mass[lo]
                num += np.sum(mass[hi] * below, axis=1)
                denom += mass[lo].sum(1) * mass[hi].sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            values.append(num / denom)
    return np.concatenate(values)


def verify_concordance():
    rng = np.random.default_rng(773)
    y = np.array([1, 1, 2, 2, 3, 3])
    p = np.array([2, 1, 1, 2, 4, 1])
    counts = rng.multinomial(6, np.full(6, 1 / 6), size=150)
    by_rank = concordance(y, p, counts)
    errors = []
    for count, estimate in zip(counts, by_rank, strict=True):
        yy, pp = np.repeat(y, count), np.repeat(p, count)
        valid = yy[:, None] > yy[None, :]
        d = pp[:, None] - pp[None, :]
        correct = np.mean(((d > 0) + 0.5 * (d == 0))[valid]) if valid.any() else np.nan
        if not np.isnan(correct):
            errors.append(abs(correct - estimate))
    err = max(errors)
    if err > 1e-12:
        raise ArithmeticError("Ordinal ordering differs from literal pair expansion")
    return float(err)


def draw_domains(groups, domains):
    # Independent bootstrap per original genre. Seed the second genre its own stream.
    base = np.random.SeedSequence(378283)
    result = {}
    for g, seed in zip(
        sorted(set(domains)), base.spawn(len(set(domains))), strict=True
    ):
        idx = np.flatnonzero(domains == g)
        unique, inv = np.unique(groups[idx], return_inverse=True)
        n = len(unique)
        counts = np.random.default_rng(seed).multinomial(n, np.full(n, 1 / n), 10000)
        result[g] = (idx, np.vstack([np.ones(len(idx), int), counts[:, inv]]))
    return result


def run():
    proof = verify_concordance()
    f = OUT / "forecasts.npz"
    seal = json.loads((OUT / "prediction-seal.json").read_text())
    if (
        digest(f) != seal["predictions"]
        or digest(PLAN) != seal["protocol"]
        or any(
            digest(Path(__file__).with_name(k)) != v
            for k, v in seal["science_code"].items()
        )
    ):
        raise ValueError(
            "Danish ratings must follow the frozen estimates and original analysis"
        )
    with np.load(f, allow_pickle=False) as raw:
        z = dict(raw)
    yref = {r["id"]: r["rating"] for r in entries("test", labels=True)}
    y = np.array([yref[k] for k in z["keys"]])
    domains = z["domain"]
    groups = z["group"]
    draws = draw_domains(groups, domains)
    arms = [k for k in z if k not in ("keys", "group", "domain")]
    panels = {g: {} for g in draws}
    macro = {}
    for k in arms:
        regional = []
        for g, (ix, c) in draws.items():
            ys, p = y[ix], z[k][ix]
            v = concordance(ys, p, c)
            # Transported estimates retain their 1–5 scores for order. The fixed map only affects absolute error.
            calibrated = (
                p if k.startswith("danish_") or k == "domain_only" else 1 + (p - 1) / 2
            )
            residual = metrics(ys, calibrated, c)
            d = {
                "concordance": v,
                **{
                    q: np.r_[residual[q][0], residual[q][1]]
                    for q in ("spearman", "mse")
                },
            }
            regional.append(d)
            panels[g][k] = d
        macro[k] = {q: np.mean([r[q] for r in regional], axis=0) for q in regional[0]}

    def aggregate(data):
        out = {
            k: {q: point(v[0], v[1:]) for q, v in measures.items()}
            for k, measures in data.items()
        }
        contrast = {}
        for k in data:
            if k == "adaptive":
                continue
            contrast[k] = {}
            for q, v in data["adaptive"].items():
                delta = (v - data[k][q]) * (-1 if q == "mse" else 1)
                contrast[k][q] = point(delta[0], delta[1:])
        return {"by_model": out, "adaptive_gain_over": contrast}

    output = {
        "domain_equal": aggregate(macro),
        "domains": {
            g: {
                **aggregate(rows),
                "n": int(sum(domains == g)),
                "human_grades": dict(
                    sorted(Counter(map(int, y[domains == g])).items())
                ),
            }
            for g, rows in panels.items()
        },
        "published_rating_field": "rating",
        "distinct_passages": len(y),
        "forecast_seal": digest(OUT / "prediction-seal.json"),
        "ordinal_pair_verification_max_error": proof,
        "draws": 10000,
        "seed": 378283,
    }
    (OUT / "scores.json").write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n"
    )
    print({k: v["concordance"] for k, v in output["domain_equal"]["by_model"].items()})


if __name__ == "__main__":
    run()
