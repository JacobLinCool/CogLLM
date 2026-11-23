"""Assess whether two trained experts' input-level variability predicts where to borrow."""

from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from research.backbone_transfer.extract import digest
from research.reading_validation.borrowing_data import EVALUATE
from research.reading_validation.borrowing_prior import (
    MODES,
    SEEDS,
    OUT as PREVIOUS,
    experts,
    partition,
    read,
    write,
)
from research.reading_validation.borrowing_reliability import ARMS, OUT, PLAN
from research.reading_validation.borrowing_score import GeometricPool
from research.union_training.fit_observations import COUNT_MODELS, weights


def variable_nll(pool, coeff):
    a = np.asarray(coeff)
    if a.ndim != 1 or len(a) != len(pool.y) or np.any((a < 0) | (a > 1)):
        raise ValueError("Per-observation coefficient shape is inconsistent")
    if pool.kind in COUNT_MODELS:
        logp = pool.left + a[:, None] * pool.delta
        center = logp - logsumexp(logp, axis=1, keepdims=True)
        return -(pool.y * np.where(pool.y > 0, center, 0)).sum(axis=1)
    if pool.kind == "paired_gaussian":
        r = a[:, None, None]
        p = (1 - r) * pool.p0 + r * pool.p1
        eta = (1 - a[:, None]) * pool.eta0 + a[:, None] * pool.eta1
        d = pool.y - np.linalg.solve(p, eta[:, :, None])[:, :, 0]
        return (
            np.log(2 * np.pi)
            - 0.5 * np.linalg.slogdet(p)[1]
            + 0.5 * np.einsum("bi,bij,bj->b", d, p, d)
        )
    return pool.loss(a)


def evaluate(mode, tasks, split):
    path = OUT / mode
    lock = read(path / "inputs-and-coefficients.json")
    if (
        lock["weights"] != digest(path / "coefficients.npz")
        or lock["protocol"] != digest(PLAN)
        or lock["code"] != digest(Path(__file__).with_name("borrowing_reliability.py"))
    ):
        raise ValueError("Material-level gate changed")
    folds = {}
    blocks = {}
    with np.load(path / "coefficients.npz", allow_pickle=False) as coef:
        for seed in SEEDS:
            old = read(PREVIOUS / mode / f"seed-{seed}" / "scores.json")
            sources = defaultdict(list)
            per = {}
            referr = 0.0
            nllerr = 0.0
            with ExitStack() as stream:
                forecasts = experts(stream, mode, seed, "development", tasks, split)
                for t in tasks:
                    ix = np.flatnonzero(split[t.name] == EVALUATE)
                    if not len(ix):
                        continue
                    pool = GeometricPool(t, ix, *(q[t.name] for q in forecasts))
                    a = coef[f"{t.name}.{seed}"]
                    w = weights(t, ix)
                    den = w @ t.trials[ix]
                    loss = np.column_stack([variable_nll(pool, v) for v in a.T])
                    vals = w @ loss / den
                    # Confirm equal coefficient predictions with the exact previously validated scalar likelihood on every row.
                    referr = max(
                        referr,
                        float(np.max(abs(loss[:, 0] - pool.loss(a[0, 0])))),
                        float(np.max(abs(loss[:, 3] - pool.loss(a[0, 3])))),
                    )
                    # Reconstruct actual varying coefficients one row at a time on a deterministic sample.
                    for k in (0, len(ix) // 2, len(ix) - 1):
                        nllerr = max(
                            nllerr, abs(float(pool.loss(a[k, 1])[k] - loss[k, 1]))
                        )
                    supplied = old["blocks"][t.name]["scores"]
                    if (
                        max(
                            abs(vals[0] - supplied["independent_mode"]),
                            abs(vals[3] - supplied["supported"]),
                        )
                        > 1e-8
                    ):
                        raise ArithmeticError(
                            "Matched endpoint control disagrees with earlier scorers"
                        )
                    sources[t.source].append(vals)
                    per[t.name] = {
                        "source": t.source,
                        "rows": len(ix),
                        "nll": dict(zip(ARMS, vals, strict=True)),
                    }
            if referr >= 1e-8 or nllerr >= 1e-8:
                raise ArithmeticError(
                    "Original per-human observation likelihood changed"
                )
            folds[seed] = {
                "groups": {
                    s: dict(zip(ARMS, np.mean(x, axis=0), strict=True))
                    for s, x in sorted(sources.items())
                },
                "constant_row_nll_max_error": referr,
                "variable_row_nll_max_error": nllerr,
                "endpoints": per,
            }
    write(
        path / "scores.json",
        {
            "folds": folds,
            "coefficient_seal": digest(path / "inputs-and-coefficients.json"),
        },
    )
    print("scored material-level", mode, flush=True)
    return folds


def aggregate(result):
    src = sorted(result[MODES[0]][SEEDS[0]]["groups"])
    n = len(src)
    D = np.random.default_rng(95078).integers(n, size=(10000, n))
    flip = np.random.default_rng(273896).choice(np.int8([-1, 1]), size=(100000, n))
    comparisons = [
        ("aligned", "endpoint"),
        ("aligned", "permuted"),
        ("prior_aligned", "prior_supported"),
    ]

    def summarize(matrix):
        out = {}
        for j, (a, b) in enumerate(comparisons):
            d = matrix[:, ARMS.index(b)] - matrix[:, ARMS.index(a)]
            draw = d[D].mean(1)
            num = 2 if j < 2 else 1
            pv = (1 + np.count_nonzero(flip @ d / n >= d.mean() - 1e-15)) / (
                len(flip) + 1
            )
            out[f"{a}_over_{b}"] = {
                "gain_nats_per_observed_trial": float(d.mean()),
                "ci95": np.quantile(draw, [0.025, 0.975]).tolist(),
                "ci95_simultaneous": np.quantile(
                    draw, [0.025 / num, 1 - 0.025 / num]
                ).tolist(),
                "improved_sources": int((d > 0).sum()),
                "declined_sources": int((d < 0).sum()),
                "signflip_p": float(pv),
                "bonferroni_p": float(min(1, num * pv)),
                "paired_by_source": dict(zip(src, d, strict=True)),
            }
        return {
            "mean_source_nll": dict(zip(ARMS, matrix.mean(0), strict=True)),
            "contrasts": out,
        }

    stats = {}
    allx = []
    for m in MODES:
        z = []
        for s in SEEDS:
            g = result[m][s]["groups"]
            p = np.array([[g[k][a] for a in ARMS] for k in src])
            stats[m + "/" + str(s)] = summarize(p)
            z.append(p)
        allx += z
        stats[m] = summarize(np.mean(z, axis=0))
    overall = summarize(np.mean(allx, axis=0))
    weight_counts = {m: {} for m in MODES}
    for m in MODES:
        d = read(OUT / m / "inputs-and-coefficients.json")["inputs"]
        weight_counts[m]["scope"] = {
            "endpoints": len(d),
            "active": sum(v["input_only_active"] for v in d.values()),
            "human_rows": sum(v["evaluation_rows"] for v in d.values()),
            "permuted_rows": sum(v["shuffled_rows_moved"] for v in d.values()),
        }
        for seed in SEEDS:
            weight_counts[m][str(seed)] = sum(
                v["seeds"][str(seed)]["above_point_one"] for v in d.values()
            )
    out = {
        "sources": n,
        "overall": overall,
        "split_and_seed": stats,
        "movement": weight_counts,
        "protocol": digest(PLAN),
        "frozen_scoring": digest(OUT / "forecast-freeze.json"),
    }
    write(OUT / "scores.json", out)
    print(
        {k: v["gain_nats_per_observed_trial"] for k, v in overall["contrasts"].items()}
    )


def main():
    import torch

    torch.set_num_threads(1)
    frozen = read(OUT / "forecast-freeze.json")
    result = {}
    if frozen["scoring_program"] != digest(Path(__file__)):
        raise ValueError("Changed registered observer")
    for mode in MODES:
        if frozen["modes"][mode] != digest(OUT / mode / "inputs-and-coefficients.json"):
            raise ValueError("An input-gated prediction changed")
    for mode in MODES:
        tasks, split = partition(mode)
        result[mode] = evaluate(mode, tasks, split)
    aggregate(result)


if __name__ == "__main__":
    main()
