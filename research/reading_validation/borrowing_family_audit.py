"""Audit family-calibrated geometric pools against analytic KL and nested material resampling."""

from __future__ import annotations

from collections import defaultdict
from contextlib import ExitStack
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from research.backbone_transfer.extract import digest
from research.reading_validation.borrowing_data import CALIBRATE, EVALUATE
from research.reading_validation.borrowing_family_bootstrap import (
    ALL_MODES,
    OUT,
    baseline_map,
    boot,
    files,
    setup,
)
from research.reading_validation.borrowing_prior import SEEDS, read, write
from research.reading_validation.borrowing_prior_score import group_id
from research.reading_validation.borrowing_score import GeometricPool
from research.union_training.fit_observations import COUNT_MODELS, weights

ARMS = ("endpoint_map", "midpoint", "family_bootstrap")
COMPARISONS = ("over_map", "over_midpoint")
WITHIN_DRAWS = 4000
NESTED_DRAWS = 20000


@dataclass
class FamilyRisks:
    groups: list[str]
    numerator: np.ndarray
    denominator: np.ndarray


def native_kl(pool, alpha):
    """Compute KL from the normalized half pool using each channel's native distribution."""
    if pool.kind in COUNT_MODELS:

        def category_logp(a):
            lp = pool.left + a * pool.delta
            return lp - logsumexp(lp, axis=1, keepdims=True)

        x, y = category_logp(0.5), category_logp(alpha)
        ratio = np.subtract(x, y, out=np.zeros_like(x), where=pool.valid)
        return (np.exp(x) * ratio).sum(axis=1)
    p = (1 - alpha) * pool.p0 + alpha * pool.p1
    m = 0.5 * (pool.p0 + pool.p1)
    eta = (1 - alpha) * pool.eta0 + alpha * pool.eta1
    center = 0.5 * (pool.eta0 + pool.eta1)
    if pool.kind == "paired_gaussian":
        diff = np.linalg.solve(p, eta[:, :, None]) - np.linalg.solve(
            m, center[:, :, None]
        )
        det = np.linalg.slogdet(m)[1] - np.linalg.slogdet(p)[1]
        trace = np.einsum("bij,bji->b", p, np.linalg.inv(m))
        ma = np.einsum("bi,bij,bj->b", diff[:, :, 0], p, diff[:, :, 0])
        return 0.5 * (trace + ma - 2 + det)
    return 0.5 * (np.log(m / p) + p / m - 1 + p * (eta / p - center / m) ** 2)


def check_kl(tasks, split, raw):
    check = defaultdict(lambda: {"rows": 0, "max_abs_error": 0.0, "min_direct_kl": 1.0})
    for t in tasks:
        rows = np.flatnonzero(split[t.name] == CALIBRATE)
        if not len(rows):
            continue
        k = min(17, len(rows))
        position = np.linspace(0, len(rows) - 1, k).astype(int)
        rows = rows[position]
        pool = GeometricPool(t, rows, *(z[t.name][position] for z in raw))
        h = 0.000005
        slope = (pool.loss(0.5 + h) - pool.loss(0.5 - h)) / (2 * h)
        ans = check[pool.kind]
        for a in (0.18, 0.83):
            measured = (pool.loss(a) - pool.loss(0.5) - (a - 0.5) * slope) / t.trials[
                rows
            ]
            true = native_kl(pool, a)
            ans["max_abs_error"] = max(
                ans["max_abs_error"], float(np.max(np.abs(true - measured)))
            )
            ans["min_direct_kl"] = min(ans["min_direct_kl"], float(true.min()))
        ans["rows"] += k
    if len(check) != 10 or max(v["max_abs_error"] for v in check.values()) > 1e-5:
        raise ArithmeticError(
            "The per-trial tangent residual failed the native analytic divergence check"
        )
    return dict(check)


def pack(mode, tasks, split, root):
    freeze = read(root / "forecast-freeze.json")
    bysource = defaultdict(dict)
    saved = {}
    raw_counts = defaultdict(int)
    with ExitStack() as stream:
        dev = [files(stream, mode, s, "development", tasks, split) for s in SEEDS]
        cal = files(stream, mode, SEEDS[0], "calibration", tasks, split)
        analytic = check_kl(tasks, split, cal)
        other = [baseline_map(mode, seed) for seed in SEEDS]
        for t in tasks:
            rows = np.flatnonzero(split[t.name] == EVALUATE)
            if not len(rows):
                continue
            groups, index = np.unique(t.family[rows], return_inverse=True)
            raw_counts[t.source] += int(np.sum(t.trials[rows]))
            g = [group_id(t.source, s) for s in groups]
            w = weights(t, rows)
            den = np.bincount(index, weights=w * t.trials[rows], minlength=len(g))
            loss = []
            for j, seed in enumerate(SEEDS):
                pool = GeometricPool(t, rows, *(z[t.name] for z in dev[j]))
                candidates = (
                    other[j][t.name]["independent_mode"],
                    0.5,
                    freeze["seeds"][str(seed)][t.name]["weight"],
                )
                loss.append(
                    np.column_stack(
                        [
                            np.bincount(
                                index, weights=w * pool.loss(a), minlength=len(g)
                            )
                            for a in candidates
                        ]
                    )
                )
            family = FamilyRisks(g, np.mean(loss, axis=0), den)
            bysource[t.source][t.name] = family
            saved[t.name + ".group_sha256"] = g
            saved[t.name + ".numerator"] = np.stack(loss)
            saved[t.name + ".denominator"] = den
    np.savez_compressed(root / "material-statistics.npz", **saved)
    return bysource, analytic, raw_counts


def contrasts(x):
    return x[..., :2] - x[..., [2]]


def within(key, endpoint_families):
    labels = sorted({g for e in endpoint_families.values() for g in e.groups})
    jdx = {g: i for i, g in enumerate(labels)}
    n, k = len(labels), len(endpoint_families)
    den = np.zeros((n, k))
    num = np.zeros((n, k, len(ARMS)))
    for j, e in enumerate(endpoint_families.values()):
        coords = [jdx[g] for g in e.groups]
        num[coords, j] = e.numerator
        den[coords, j] = e.denominator
    exact = (num.sum(axis=0) / den.sum(axis=0)[:, None]).mean(axis=0)
    if len(labels) == 1:
        return (
            1,
            exact,
            np.broadcast_to(contrasts(exact), (WITHIN_DRAWS, 2)),
            WITHIN_DRAWS,
        )
    rng = np.random.default_rng(int(sha256(key.encode()).hexdigest()[:16], 16))
    draws = []
    valid = 0
    tried = 0
    while valid < WITHIN_DRAWS:
        rep = rng.multinomial(n, np.full(n, 1 / n), size=100)
        mass = rep @ den
        okay = np.all(mass > 0, axis=1)
        count = int(okay.sum())
        if count:
            v = (rep[okay] @ num.reshape(n, -1)).reshape(-1, k, len(ARMS))
            draws.append(contrasts((v / mass[okay, :, None]).mean(axis=1)))
        valid += count
        tried += len(rep)
        if tried > 20 * WITHIN_DRAWS:
            raise ValueError("Too few resamples preserve every observation channel")
    return n, exact, np.concatenate(draws)[:WITHIN_DRAWS], tried


def exact_signflip(delta):
    """Count all sign configurations with an equal or greater signed sum, including zero ties."""

    def signs(z):
        x = np.zeros(1)
        for v in z:
            x = np.concatenate((x + v, x - v))
        return x

    cut = len(delta) // 2
    a, b = signs(delta[:cut]), np.sort(signs(delta[cut:]))
    counts = len(b) - np.searchsorted(b, delta.sum() - a - 1e-13)
    return float(counts.sum() / (len(a) * len(b)))


def weight_diagnostics(seal):
    vals = [v for s in seal["seeds"].values() for v in s.values() if v["families"]]
    w = np.array([v["weight"] for v in vals])
    return {
        "calibrated_endpoint_seed_pairs": len(vals),
        "coefficient_quantiles_0_25_50_75_100": np.quantile(
            w, [0, 0.25, 0.5, 0.75, 1.0]
        ).tolist(),
        "median_calibration_families": float(np.median([v["families"] for v in vals])),
        "grid_refinements": sum(v["grid"] == 257 for v in vals),
        "max_final_coarse_mean_error": max(v["coarse_mean_error"] for v in vals),
        "max_mc_standard_error": max(v["mc_standard_error_antithetic"] for v in vals),
        "max_independent_half_mean_difference": max(
            v["independent_half_mean_difference"] for v in vals
        ),
        "max_midpoint_derivative_step_error": max(
            v["midpoint_derivative_error"] for v in vals
        ),
        "min_calibration_kl": min(v["minimum_KL"] for v in vals),
    }


def audit(mode):
    root = OUT / mode
    tasks, split = setup(mode)
    bysource, analytic, raw = pack(mode, tasks, split, root)
    score = read(root / "scores.json")
    names = sorted(bysource)
    meta, arrays, points = {}, [], []
    diff = []
    for s in names:
        n, observed, samples, tried = within(mode + "/" + s, bysource[s])
        target = score["per_source_nll_order_map_half_bootstrap"][s]
        diff.append(np.max(np.abs(observed - target)))
        meta[s] = {
            "families": n,
            "endpoints": len(bysource[s]),
            "raw_observations": raw[s],
            "observed_nll": dict(zip(ARMS, observed, strict=True)),
            "material_resamples_attempted": tried if n > 1 else 0,
            "within_material_intervals": None
            if n == 1
            else dict(
                zip(
                    COMPARISONS,
                    np.quantile(samples, [0.025, 0.975], axis=0).T.tolist(),
                    strict=True,
                )
            ),
        }
        arrays.append(samples)
        points.append(contrasts(observed))
    n = len(names)
    points, arrays = np.array(points), np.array(arrays)
    rr = np.random.default_rng(8176213)
    sid = rr.integers(n, size=(NESTED_DRAWS, n))
    idx = rr.integers(WITHIN_DRAWS, size=sid.shape)
    nested = arrays[sid, idx].mean(axis=1)
    result = {}
    for j, key in enumerate(COMPARISONS):
        x = points[:, j]
        leave_out = (x.sum() - x) / (len(x) - 1)
        exact_p = exact_signflip(x)
        result[key] = {
            "gain": float(x.mean()),
            "two_level_nominal_ci": np.quantile(nested[:, j], [0.025, 0.975]).tolist(),
            "two_level_simultaneous_ci": np.quantile(
                nested[:, j], [0.0125, 0.9875]
            ).tolist(),
            "exact_source_signflip_p": exact_p,
            "two_comparison_adjusted_exact_p": min(1.0, 2 * exact_p),
            "leave_one_source_out_range": [
                float(leave_out.min()),
                float(leave_out.max()),
            ],
            "minimum_after_removing": names[np.argmin(leave_out)],
            "maximum_after_removing": names[np.argmax(leave_out)],
            "influences": {s: float(p) for s, p in zip(names, leave_out, strict=True)},
            "median_source_gain": float(np.median(x)),
        }
    if max(diff) > 1e-10:
        raise ArithmeticError(
            "Grouped numerator reconstruction changed the original study average"
        )
    write(
        root / "audit.json",
        {
            "mode": mode,
            "two_level_design": "Draw study with replacement and paired materials with replacement within each draw; single-family effects conditional. Endpoints preserved, duplicate material coordinates resampled jointly across endpoints. Resample two seeds together.",
            "source_n": n,
            "endpoint_n": sum(map(len, bysource.values())),
            "raw_observation_count": sum(raw.values()),
            "usable_families": sum(v["families"] for v in meta.values()),
            "single_material_sources": sum(v["families"] == 1 for v in meta.values()),
            "median_held_families_per_source": float(
                np.median([v["families"] for v in meta.values()])
            ),
            "resamples": {
                "source_by_material": NESTED_DRAWS,
                "material_per_source": WITHIN_DRAWS,
            },
            "max_independent_replay_error": float(max(diff)),
            "analytic_tangent_checks": analytic,
            "coefficient_numerics": weight_diagnostics(
                read(root / "forecast-freeze.json")
            ),
            "effects": result,
            "sources": meta,
            "statistics_sha256": digest(root / "material-statistics.npz"),
            "original_score_sha256": digest(root / "scores.json"),
            "audit_code_sha256": digest(Path(__file__)),
        },
    )
    print(
        mode,
        {
            k: (
                v["gain"],
                v["two_level_simultaneous_ci"],
                v["leave_one_source_out_range"],
            )
            for k, v in result.items()
        },
        flush=True,
    )


def self_test():
    nodes = np.linspace(0, 1, 129)
    constant, _ = boot(np.ones((2, 129)), 19)
    observed, _ = boot(np.stack(((nodes - 0.9) ** 2, (nodes - 0.5) ** 2)), 19)
    # Opposite uniform draws give exactly complementary normalized weights only if the two coordinates
    # are swapped. The Dirichlet(1,1) population mean still provides a check within Monte Carlo error.
    if constant != 0.5 or abs(observed - 0.7) > 0.003:
        raise ArithmeticError(
            "Synthetic family bootstrap missed the known symmetric optimum"
        )
    return {
        "flat_returned": constant,
        "quadratic_estimate": observed,
        "quadratic_exact": 0.7,
    }


def main():
    import torch

    torch.set_num_threads(1)
    write(OUT / "numerical-test.json", self_test())
    for mode in ("third", *ALL_MODES[:-1]):
        audit(mode)


if __name__ == "__main__":
    main()
