"""Score presealed study-excluded coefficients with the original observation likelihoods and held materials."""

from collections import defaultdict
from contextlib import ExitStack
from hashlib import sha256
from pathlib import Path

import numpy as np

from research.backbone_transfer.extract import digest
from research.reading_validation.borrowing_data import EVALUATE
from research.reading_validation.borrowing_prior import (
    METHODS,
    MODES,
    OUT,
    PLAN,
    SEEDS,
    expert_dir,
    experts,
    partition,
    read,
    write,
)
from research.reading_validation.borrowing_score import GeometricPool
from research.union_training.fit_observations import weights

PRIMARY = (
    ("supported", "independent_mode"),
    ("supported", "midpoint"),
    ("transported", "midpoint"),
)
COMPARISONS = (
    *PRIMARY,
    ("supported", "private"),
    ("supported", "shared"),
    ("supported", "independent_mean"),
    ("supported", "transported"),
    ("transported", "private"),
    ("transported", "shared"),
    ("independent_mode", "midpoint"),
    ("independent_mean", "midpoint"),
)


def group_id(s, g):
    return sha256((str(s) + "\0" + str(g)).encode()).hexdigest()


def contrasts(scores):
    return {
        f"{a}_over_{b}": scores[..., METHODS.index(b)] - scores[..., METHODS.index(a)]
        for a, b in COMPARISONS
    }


def evaluate(mode, seed, tasks, splits):
    folder = OUT / mode / f"seed-{seed}"
    cal = read(folder / "calibration-lock.json")
    original = expert_dir(mode, seed)
    if (
        cal["protocol"] != digest(PLAN)
        or cal["code"] != digest(Path(__file__).with_name("borrowing_prior.py"))
        or digest(original / "fit.json") != cal["experts"]
    ):
        raise ValueError("An estimator or earlier base forecast changed")
    coef = cal["coefficients"]
    data = {}
    source = defaultdict(list)
    results = {}
    with ExitStack() as opened:
        targets = experts(opened, mode, seed, "development", tasks, splits)
        for task in tasks:
            selected = np.flatnonzero(splits[task.name] == EVALUATE)
            if not len(selected):
                continue
            w = weights(task, selected)
            pool = GeometricPool(task, selected, *(q[task.name] for q in targets))
            s = task.source
            family, ix = np.unique(task.family[selected], return_inverse=True)
            den = np.bincount(
                ix, weights=w * task.trials[selected], minlength=len(family)
            )
            if np.any(den <= 0):
                raise ValueError("A held family has no observed human response")
            row = np.column_stack([pool.loss(coef[task.name][k]) for k in METHODS])
            if not np.all(np.isfinite(row)):
                raise ValueError("Invalid held observation likelihood")
            num = np.column_stack(
                [
                    np.bincount(ix, weights=w * col, minlength=len(family))
                    for col in row.T
                ]
            )
            scores = num.sum(0) / den.sum()
            expected = (w @ row) / (w @ task.trials[selected])
            if not np.allclose(scores, expected, atol=1e-9, rtol=1e-10):
                raise ArithmeticError(
                    "Summing the original family statistics changed the response denominator"
                )
            results[task.name] = {
                "source": s,
                "observation": task.likelihood,
                "rows": len(selected),
                "trials": float(task.trials[selected].sum()),
                "source_families": len(family),
                "scores": dict(zip(METHODS, scores, strict=True)),
            }
            source[s].append(scores)
            f = np.array([group_id(s, k) for k in family])
            data[task.name] = f, num, den
    source = {k: np.mean(v, axis=0) for k, v in sorted(source.items())}
    agg = np.stack(list(source.values())).mean(0)
    if mode == "historical":
        published = read(original / "scores.json")["cross_source_nll"]
        names = {
            "private": "private",
            "shared": "shared",
            "midpoint": "midpoint",
            "independent_mode": "adaptive",
        }
    else:
        published = read(original / "comparison.json")["study_mean"]
        names = {
            "private": "private",
            "shared": "shared",
            "midpoint": "equal_density_pool",
            "independent_mode": "calibrated_density_pool",
        }
    diff = max(abs(agg[METHODS.index(k)] - published[t]) for k, t in names.items())
    if diff is not None and diff > 1e-8:
        raise ArithmeticError(
            "Original scored private/shared/midpoint or individual coefficient differs"
        )
    export = folder / "family-statistics.npz"
    np.savez_compressed(
        export,
        **{
            n + "." + c: v[j]
            for n, v in data.items()
            for j, c in enumerate(("ids", "numerator", "denominator"))
        },
    )
    d = {
        "calibration_lock": digest(folder / "calibration-lock.json"),
        "family_statistics": digest(export),
        "evaluated_sources": len(source),
        "evaluated_endpoints": len(results),
        "methods": list(METHODS),
        "published_four_control_scores_replay_max_error": diff,
        "macro_source_nll": dict(zip(METHODS, agg, strict=True)),
        "blocks": results,
        "sources": {s: dict(zip(METHODS, a, strict=True)) for s, a in source.items()},
    }
    write(folder / "scores.json", d)
    print(mode, seed, "completed", len(source), len(results), flush=True)


def source_interval(num, den, bootstrap_size=4000):
    """Columns retain each endpoint even if its material family is absent from a particular resample."""
    n = num.shape[0]
    ep = num.shape[1]
    m = num.shape[2]
    if n < 2:
        return {"observed_families": n, "contributing_draws": 0, "intervals": None}
    # Different sources have different held family sets, so resample all endpoints jointly at their original family level.
    rng = np.random.default_rng(81732)
    acc = []
    for k in range(0, bootstrap_size, 80):
        counts = rng.multinomial(n, np.full(n, 1 / n), size=min(80, bootstrap_size - k))
        total = counts @ den
        good = (total > 0).all(axis=1)
        if not np.any(good):
            continue
        trial = (counts[good] @ num.reshape(n, ep * m)).reshape(-1, ep, m) / total[
            good, :, None
        ]
        acc.append(trial.mean(axis=1))
    if not acc:
        return {"observed_families": n, "contributing_draws": 0, "intervals": None}
    samples = np.vstack(acc)
    return {
        "observed_families": n,
        "contributing_draws": len(samples),
        "intervals": {
            k: np.quantile(v, [0.025, 0.975]).tolist()
            for k, v in contrasts(samples).items()
        },
        "method_nll_intervals": {
            k: np.quantile(samples[:, j], [0.025, 0.975]).tolist()
            for j, k in enumerate(METHODS)
        },
    }


def fold_aggregate(mode):
    folders = [OUT / mode / f"seed-{s}" for s in SEEDS]
    stats = [read(p / "scores.json") for p in folders]
    cal = [read(p / "calibration-lock.json") for p in folders]
    sources = sorted(stats[0]["sources"])
    if set(sources) != set(stats[1]["sources"]):
        raise ValueError("Different held source rosters by seed")
    result = {}
    with ExitStack() as opened:
        family = [
            opened.enter_context(
                np.load(p / "family-statistics.npz", allow_pickle=False)
            )
            for p in folders
        ]
        for source in sources:
            ep = [k for k, v in stats[0]["blocks"].items() if v["source"] == source]
            src = sorted({k for name in ep for k in family[0][name + ".ids"]})
            index = {s: i for i, s in enumerate(src)}
            num = np.zeros((len(src), len(ep), len(METHODS)))
            den = np.zeros((len(src), len(ep)))
            for j, k in enumerate(ep):
                key = k + ".ids"
                denkey = k + ".denominator"
                nkey = k + ".numerator"
                if not np.array_equal(
                    family[0][key], family[1][key]
                ) or not np.array_equal(family[0][denkey], family[1][denkey]):
                    raise ValueError("Seed changed tested humans or materials")
                loc = [index[s] for s in family[0][key]]
                den[loc, j] = family[0][denkey]
                num[loc, j] = 0.5 * (family[0][nkey] + family[1][nkey])
            means = (num.sum(0) / den.sum(0)[:, None]).mean(0)
            old = np.mean(
                [[s["sources"][source][k] for k in METHODS] for s in stats], axis=0
            )
            if not np.allclose(means, old, rtol=1e-10, atol=1e-9):
                raise ArithmeticError("Seed averaging changed estimate")
            uncertainty = source_interval(num, den)
            coeff = {
                name: [np.mean([r["coefficients"][k][name] for k in ep]) for r in cal]
                for name in (
                    "supported",
                    "transported",
                    "independent_mode",
                    "independent_mean",
                )
            }
            ncal = {
                k: next(
                    r["families"]
                    for r in cal[0]["calibration_rows_and_families"]
                    if r["endpoint"] == k
                )
                for k in ep
            }
            result[source] = {
                "endpoints": len(ep),
                "calibration_families_per_endpoint": ncal,
                "held_families": len(src),
                "mean_coefficients_by_seed": coeff,
                "nll_by_method": dict(zip(METHODS, means, strict=True)),
                "paired_improvement": contrasts(means),
                "held_material_uncertainty": uncertainty,
            }
    return result


def aggregate():
    split = {mode: fold_aggregate(mode) for mode in MODES}
    sources = sorted(set().union(*(v.keys() for v in split.values())))
    if any(set(sources) != s.keys() for s in split.values()):
        raise ValueError(
            "Some study is observed only in one split: separate split inference required"
        )
    n = len(sources)
    draw = np.random.default_rng(21752).integers(n, size=(10000, n))
    sign = np.random.default_rng(95261).choice(np.int8([-1, 1]), size=(100000, n))
    allrecords = {}

    def summary(p):
        out = {}
        point = p.mean(0)
        for (a, b), (name, d) in zip(COMPARISONS, contrasts(p).items(), strict=True):
            raw = d[draw].mean(1)
            sz = 3 if (a, b) in PRIMARY else 1
            flip = (sign @ d) / n
            pval = (1 + np.sum(flip >= d.mean() - 1e-15)) / (len(flip) + 1)
            out[name] = {
                "gain_nats_per_observed_trial": float(d.mean()),
                "ci95": np.quantile(raw, [0.025, 0.975]).tolist(),
                "ci95_simultaneous": np.quantile(
                    raw, [0.025 / sz, 1 - 0.025 / sz]
                ).tolist(),
                "source_improvements": int(np.sum(d > 0)),
                "source_losses": int(np.sum(d < 0)),
                "one_sided_signflip_p": float(pval),
                "bonferroni_p": float(min(1.0, sz * pval)),
            }
        return {
            "source_mean_nll": dict(zip(METHODS, point, strict=True)),
            "contrasts": out,
        }

    matrices = []
    for mode, rec in split.items():
        p = np.array([[rec[s]["nll_by_method"][k] for k in METHODS] for s in sources])
        matrices.append(p)
        allrecords[mode] = summary(p)
        for seed in SEEDS:
            srec = read(OUT / mode / f"seed-{seed}" / "scores.json")["sources"]
            allrecords[f"{mode}/seed-{seed}"] = summary(
                np.asarray([[srec[s][k] for k in METHODS] for s in sources])
            )
    full = summary(np.mean(matrices, axis=0))
    write(OUT / "source-scores-and-intervals.json", split)
    verdict = {
        "sources": sources,
        "source_count": n,
        "macro_over_both_splits_and_seeds": full,
        "separate_split_and_seed": allrecords,
        "code": digest(Path(__file__)),
        "design": digest(PLAN),
        "source_stats": digest(OUT / "source-scores-and-intervals.json"),
    }
    write(OUT / "scores.json", verdict)
    print(
        "Aggregate complete",
        n,
        {k: full["contrasts"][k] for k in [f"{a}_over_{b}" for a, b in PRIMARY]},
        flush=True,
    )


def main():
    import torch

    torch.set_num_threads(1)
    locks = read(OUT / "forecast-freeze.json")
    for mode in MODES:
        for seed in SEEDS:
            k = f"{mode}/seed-{seed}"
            folder = OUT / k
            if (
                digest(folder / "calibration-lock.json") != locks["coefficients"][k]
                or digest(Path(__file__)) != locks["scoring_program"]
            ):
                raise ValueError(
                    "Fit all four coefficient sets and freeze scoring before evaluation"
                )
    for mode in MODES:
        tasks, split = partition(mode)
        for seed in SEEDS:
            evaluate(mode, seed, tasks, split)
    aggregate()


if __name__ == "__main__":
    main()
