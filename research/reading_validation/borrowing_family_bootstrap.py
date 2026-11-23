"""Estimate an observation-specific sharing weight by resampling directly measured material families."""

import argparse
from contextlib import ExitStack
from hashlib import sha256
from pathlib import Path

import numpy as np

from research.backbone_transfer.extract import digest
from research.reading_validation.borrowing_prior import (
    MODES,
    SEEDS,
    experts,
    partition,
    read,
    write,
)
from research.reading_validation.borrowing_data import CALIBRATE, EVALUATE
from research.reading_validation.borrowing_score import GeometricPool, material_weights
from research.reading_validation.borrowing_replication import states
from research.reading_validation.borrowing_replication_score import get
from research.union_training.fit_observations import weights

OUT = Path("research/results/reading-validation-family-bootstrap")
PLAN = Path(__file__).with_name("borrowing-family-bootstrap-protocol.md")
ALL_MODES = (*MODES, "third")
N_PAIR = 2048


def setup(mode):
    if mode == "third":
        cache, tasks, split, _ = states()
        del cache
        return tasks, split
    return partition(mode)


def files(stream, mode, seed, phase, tasks, split):
    return (
        get(stream, seed, phase)
        if mode == "third"
        else experts(stream, mode, seed, phase, tasks, split)
    )


def minimum(values):
    """Convex equal-grid minima, with local curvature evaluated at both true endpoints."""
    n = values.shape[1] - 1
    h = 1.0 / n
    index = np.argmin(values, axis=1)
    r = np.arange(len(values))
    j = np.clip(index, 1, n - 1)
    left, center, right = values[r, j - 1], values[r, j], values[r, j + 1]
    denom = left - 2 * center + right
    step = np.divide(
        0.5 * (left - right), denom, out=np.zeros(len(r)), where=denom > 1e-15
    )
    ans = np.clip((j + np.clip(step, -1, 1)) * h, 0, 1)
    # A convex function with a nonnegative derivative at the left endpoint, or nonpositive at the right, has its minimum exactly there.
    lo = (-3 * values[:, 0] + 4 * values[:, 1] - values[:, 2]) / (2 * h)
    hi = (3 * values[:, -1] - 4 * values[:, -2] + values[:, -3]) / (2 * h)
    ans[(index == 0) & (lo >= 0)] = 0.0
    ans[(index == n) & (hi <= 0)] = 1.0
    ans[np.ptp(values, axis=1) < 1e-13] = 0.5
    return ans


def boot(curves, generator_seed):
    rng = np.random.default_rng(generator_seed)
    items = []
    coarse = []
    antipairs = []
    for batch in range(0, N_PAIR, 64):
        n = min(64, N_PAIR - batch)
        u = rng.uniform(size=(n, len(curves)))
        u = np.clip(u, np.finfo(float).tiny, 1 - np.finfo(float).eps)
        g = np.concatenate((-np.log(u), -np.log1p(-u)))
        loss = g @ curves
        fine = minimum(loss)
        rough = minimum(loss[:, ::2])
        items.extend(fine)
        coarse.extend(rough)
        antipairs.extend(0.5 * (fine[:n] + fine[n:]))
    a = np.array(items)
    b = np.array(antipairs)
    return float(a.mean()), {
        "draws": len(a),
        "mass_interval": np.quantile(a, [0.025, 0.975]).tolist(),
        "optimum_zero_rate": float(np.mean(a == 0)),
        "optimum_one_rate": float(np.mean(a == 1)),
        "coarse_mean_error": float(abs(a.mean() - np.mean(coarse))),
        "mc_standard_error_antithetic": float(np.std(b, ddof=1) / np.sqrt(len(b))),
        "independent_half_mean_difference": float(
            abs(b[: len(b) // 2].mean() - b[len(b) // 2 :].mean())
        ),
    }


def estimate(pool, task, ids, mode):
    group, owner = np.unique(task.family[ids], return_inverse=True)
    w, n = material_weights(task, ids)
    f = lambda a: np.bincount(owner, weights=w * pool.loss(a), minlength=n)
    seed = int(
        sha256(
            ("cogllm-calibration-family-replica-v1/" + mode + "/" + task.name).encode()
        ).hexdigest()[:16],
        16,
    )
    mid = f(0.5)
    slope = (f(0.50001) - f(0.49999)) / 0.00002
    s2 = (f(0.500005) - f(0.499995)) / 0.00001
    fd = float(np.max(np.abs(slope - s2)))
    if fd > 3e-5:
        raise ArithmeticError("Family loss derivative unstable at the midpoint")
    nodes = np.linspace(0, 1, 129)
    curves = np.stack([f(float(a)) - mid for a in nodes], axis=1)

    def construct(v, grid):
        neutral = (v - slope[:, None] * (grid - 0.5)).mean(0)
        conv = float(np.min(np.diff(v, n=2, axis=1)))
        small = float(np.min(neutral))
        if conv < -1e-7 or small < -1e-7:
            raise ArithmeticError(
                "Base log-pooling risk is not convex or its material KL became negative"
            )
        return np.vstack([v, np.maximum(0.0, neutral)]), conv, small

    allv, c, k = construct(curves, nodes)
    alpha, diag = boot(allv, seed)
    version = 129
    err = diag["coarse_mean_error"]
    if err > 0.0005:
        fine = np.linspace(0, 1, 257)
        v = np.empty((n, 257))
        v[:, ::2] = curves
        v[:, 1::2] = np.stack([f(float(a)) - mid for a in fine[1::2]], axis=1)
        other, c, k = construct(v, fine)
        alpha, diag = boot(other, seed)
        version = 257
        if diag["coarse_mean_error"] > 0.0005:
            raise ArithmeticError(
                "Conditional sharing weight is unresolved numerically"
            )
    return alpha, {
        **diag,
        "grid": version,
        "coarse_65_vs129_check": err,
        "midpoint_derivative_error": fd,
        "minimum_KL": k,
        "smallest_measured_second_difference": c,
        "families": n,
        "calibration_rows": len(ids),
    }


def fit(mode, tasks, split):
    params = {}
    out = OUT / mode
    out.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        v = {}
        with ExitStack() as st:
            raw = files(st, mode, seed, "calibration", tasks, split)
            for t in tasks:
                ix = np.flatnonzero(split[t.name] == CALIBRATE)
                if not len(ix):
                    v[t.name] = {"weight": 0.5, "families": 0, "calibration_rows": 0}
                    continue
                pool = GeometricPool(t, ix, *(z[t.name] for z in raw))
                a, b = estimate(pool, t, ix, mode)
                v[t.name] = {"weight": a, **b}
        params[str(seed)] = v
        print(mode, "bootstrap fit", seed, flush=True)
    write(
        out / "forecast-freeze.json",
        {
            "program": digest(Path(__file__)),
            "protocol": digest(PLAN),
            "evaluation_response_opened_in_this_new_fit": False,
            "seeds": params,
        },
    )


def baseline_map(mode, seed):
    if mode == "third":
        from research.reading_validation.borrowing_replication import OUT as p

        return read(p / "forecast-freeze.json")["coefficients"][str(seed)]["weights"]
    from research.reading_validation.borrowing_prior import OUT as p

    return read(p / mode / f"seed-{seed}" / "calibration-lock.json")["coefficients"]


def summary(matrix):
    n = len(matrix)
    rng = np.random.default_rng(931788)
    D = rng.integers(n, size=(10000, n))
    sign = np.random.default_rng(382401).choice(np.int8([-1, 1]), size=(100000, n))
    out = {}
    for j, name in enumerate(["endpoint_map", "midpoint"]):
        z = matrix[:, j] - matrix[:, 2]
        values = z[D].mean(1)
        pv = (1 + np.count_nonzero(sign @ z / n >= z.mean() - 1e-15)) / (1 + len(sign))
        out[name] = {
            "gain": float(z.mean()),
            "ci95": np.quantile(values, [0.025, 0.975]).tolist(),
            "ci95_simultaneous": np.quantile(values, [0.0125, 0.9875]).tolist(),
            "signflip_p": float(pv),
            "bonferroni_p": min(1.0, float(2 * pv)),
            "improving_sources": int((z > 0).sum()),
            "losing_sources": int((z < 0).sum()),
        }
    return {
        "source_macro_nll": dict(
            zip(
                ("endpoint_map", "midpoint", "family_bootstrap"),
                matrix.mean(0),
                strict=True,
            )
        ),
        "comparisons": out,
    }


def score(mode, tasks, split):
    out = OUT / mode
    f = read(out / "forecast-freeze.json")
    if f["program"] != digest(Path(__file__)) or f["protocol"] != digest(PLAN):
        raise ValueError("The new observer changed after fitting")
    results = {}
    matrices = []
    endpoints = {}
    units = {}
    for seed in SEEDS:
        baseline = baseline_map(mode, seed)
        by = {}
        data = {}
        with ExitStack() as st:
            raw = files(st, mode, seed, "development", tasks, split)
            for t in tasks:
                ix = np.flatnonzero(split[t.name] == EVALUATE)
                if not len(ix):
                    continue
                pool = GeometricPool(t, ix, *(z[t.name] for z in raw))
                w = weights(t, ix)
                a = f["seeds"][str(seed)][t.name]["weight"]
                g, ident = np.unique(t.family[ix], return_inverse=True)
                d = np.bincount(ident, weights=w * t.trials[ix], minlength=len(g))
                risk = np.column_stack(
                    [
                        np.bincount(ident, weights=w * pool.loss(v), minlength=len(g))
                        for v in (baseline[t.name]["independent_mode"], 0.5, a)
                    ]
                )
                vals = risk.sum(0) / d.sum()
                by.setdefault(t.source, []).append(vals)
                data[t.name] = {
                    "source": t.source,
                    "rows": len(ix),
                    "families": len(g),
                    "nll": vals.tolist(),
                }
                units.setdefault(t.name, []).append((g, d, risk))
        names = sorted(by)
        x = np.array([np.mean(by[k], axis=0) for k in names])
        matrices.append(x)
        results[str(seed)] = summary(x)
        endpoints[str(seed)] = data
    m = np.mean(matrices, axis=0)
    sourcem = {k: m[i].tolist() for i, k in enumerate(names)}
    family_ci = {}
    for src in names:
        keys = [k for k, v in endpoints[str(SEEDS[0])].items() if v["source"] == src]
        allg = sorted({g for k in keys for g in units[k][0][0]})
        n = len(allg)
        if n < 2:
            family_ci[src] = None
            continue
        idx = {g: i for i, g in enumerate(allg)}
        num = np.zeros((n, len(keys), 3))
        den = np.zeros((n, len(keys)))
        for j, k in enumerate(keys):
            (a, d, p), (b, d2, q) = units[k]
            if not np.array_equal(a, b) or not np.array_equal(d, d2):
                raise ValueError("Seed changes human trials")
            pos = [idx[z] for z in a]
            num[pos, j] = 0.5 * (p + q)
            den[pos, j] = d
        samples = []
        rng = np.random.default_rng(73130)
        for j in range(0, 4000, 100):
            counts = rng.multinomial(n, np.full(n, 1 / n), size=100)
            tr = counts @ den
            ok = (tr > 0).all(1)
            if ok.any():
                samples.append(
                    (
                        (counts[ok] @ num.reshape(n, -1)).reshape(-1, len(keys), 3)
                        / tr[ok, :, None]
                    ).mean(1)
                )
        s = np.vstack(samples)
        family_ci[src] = {
            key: np.quantile(s[:, j] - s[:, 2], [0.025, 0.975]).tolist()
            for j, key in enumerate(["over_map", "over_midpoint"])
        }
    res = {
        "sources": len(names),
        "endpoints": len(units),
        "roles": mode,
        "seed_mean": summary(m),
        "separate_seeds": results,
        "per_source_nll_order_map_half_bootstrap": sourcem,
        "source_ci": family_ci,
        "endpoints_by_seed": endpoints,
        "fit": digest(out / "forecast-freeze.json"),
    }
    write(out / "scores.json", res)
    print("family bootstrap scores", mode, res["seed_mean"], flush=True)


def main():
    import torch

    torch.set_num_threads(1)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("fit", "score"))
    p.add_argument("--mode", choices=ALL_MODES, nargs="+", default=list(ALL_MODES))
    args = p.parse_args()
    for m in args.mode:
        tasks, split = setup(m)
        (fit if args.stage == "fit" else score)(m, tasks, split)


if __name__ == "__main__":
    main()
