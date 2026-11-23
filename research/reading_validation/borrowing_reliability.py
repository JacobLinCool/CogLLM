"""Use disagreement of two independently trained heads to set input-specific observation weights."""

from collections import defaultdict
from contextlib import ExitStack
from hashlib import sha256
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from research.backbone_transfer.extract import digest
from research.reading_validation.borrowing_prior import (
    MODES,
    SEEDS,
    OUT as PREVIOUS,
    experts,
    partition,
    read,
    write,
)
from research.reading_validation.borrowing_data import CALIBRATE, EVALUATE
from research.union_training.fit_observations import COUNT_MODELS

OUT = Path("research/results/reading-validation-borrowing-reliability")
PLAN = Path(__file__).with_name("borrowing-reliability-protocol.md")
ARMS = ("endpoint", "aligned", "permuted", "prior_supported", "prior_aligned")


def latent_cov(v):
    sd = v[:, 2:4]
    rho = v[:, 4]
    cv = sd[:, 0] * sd[:, 1] * rho
    return np.column_stack([sd[:, 0] ** 2, cv, cv, sd[:, 1] ** 2]).reshape(-1, 2, 2)


def disagreement(kind, x, y):
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    if a.shape != b.shape:
        raise ValueError("Head replicas do not share an output support")
    if kind in COUNT_MODELS:
        if not np.array_equal(np.isfinite(a), np.isfinite(b)):
            raise ValueError("Incompatible category masks")
        a = a - logsumexp(a, axis=1, keepdims=True)
        b = b - logsumexp(b, axis=1, keepdims=True)
        logaff = logsumexp(0.5 * (a + b), axis=1)
    elif kind == "paired_gaussian":
        mu = a[:, :2] - b[:, :2]
        sa, sb = latent_cov(a), latent_cov(b)
        cov = 0.5 * (sa + sb)
        logaff = (
            0.25 * (np.linalg.slogdet(sa)[1] + np.linalg.slogdet(sb)[1])
            - 0.5 * np.linalg.slogdet(cov)[1]
            - 0.125
            * np.einsum("ij,ij->i", mu, np.linalg.solve(cov, mu[:, :, None])[:, :, 0])
        )
    else:
        var = 0.5 * (a[:, 1] ** 2 + b[:, 1] ** 2)
        if np.any(var <= 0) or np.any((a[:, 1] <= 0) | (b[:, 1] <= 0)):
            raise ValueError("Nonpositive latent response scale")
        logaff = (
            0.5 * (np.log(a[:, 1]) + np.log(b[:, 1]) - np.log(var))
            - 0.125 * (a[:, 0] - b[:, 0]) ** 2 / var
        )
    value = -np.expm1(np.minimum(0.0, logaff))
    if (
        np.any(logaff > 1e-8)
        or np.any(~np.isfinite(value))
        or np.any((value < 0) | (value > 1))
    ):
        raise ArithmeticError("Invalid squared Hellinger distance")
    return value


def family_mean(v, ids):
    _, group = np.unique(ids, return_inverse=True)
    return float(np.mean(np.bincount(group, weights=v) / np.bincount(group)))


def adjusted(a, ratio):
    return a * ratio / (1 - a + a * ratio)


def permute(task, ix, classes, ratio):
    groups = defaultdict(list)
    coords = task.coordinates[ix].numpy()
    for j, valid in enumerate(classes):
        val = (
            "cogllm-disagreement-permutation/"
            + task.name
            + "/"
            + str(task.family[ix[j]])
            + "/"
            + ",".join(map(str, coords[j]))
            + "/"
            + str(ix[j])
        )
        groups[int(valid)].append((sha256(val.encode()).digest(), j))
    ixmap = np.arange(len(ix))
    for group in groups.values():
        o = np.array([v for _, v in sorted(group)])
        ixmap[o] = np.roll(o, len(o) // 2)
    if sorted(ixmap) != list(range(len(ix))) or np.any(classes != classes[ixmap]):
        raise ArithmeticError("Permutation changed the expert support")
    return ratio[ixmap], int(np.sum(ixmap != np.arange(len(ix))))


def generate(mode, tasks, split):
    destination = OUT / mode
    destination.mkdir(parents=True, exist_ok=True)
    locks = [
        read(PREVIOUS / mode / f"seed-{s}" / "calibration-lock.json") for s in SEEDS
    ]
    meta = {}
    arrays = {}
    with ExitStack() as stream:
        # Both sets of frozen model predictions are read without looking up a human evaluation response.
        cal = [experts(stream, mode, s, "calibration", tasks, split) for s in SEEDS]
        dev = [experts(stream, mode, s, "development", tasks, split) for s in SEEDS]
        for t in tasks:
            ci = np.flatnonzero(split[t.name] == CALIBRATE)
            ev = np.flatnonzero(split[t.name] == EVALUATE)
            if not len(ev):
                continue

            def features(models):
                return tuple(
                    disagreement(
                        t.likelihood, models[0][j][t.name], models[1][j][t.name]
                    )
                    for j in (0, 1)
                )

            es, ep = features(dev)
            ms = mp = None
            if len(ci):
                cs, cp = features(cal)
                ms, mp = (family_mean(v, t.family[ci]) for v in (cs, cp))
            active = bool(
                ms is not None and mp is not None and ms > 1e-12 and mp > 1e-12
            )
            r = (1 + ep / mp) / (1 + es / ms) if active else np.ones(len(ev))
            if np.any(~np.isfinite(r)) or not np.all(r > 0):
                raise ArithmeticError("Invalid trust odds")
            valid = (
                np.isfinite(dev[0][0][t.name]).sum(axis=1)
                if t.likelihood in COUNT_MODELS
                else np.ones(len(ev))
            )
            shuffled, moved = permute(t, ev, valid, r)
            rec = {
                "source": t.source,
                "likelihood": t.likelihood,
                "calibration_rows": len(ci),
                "evaluation_rows": len(ev),
                "input_only_active": active,
                "mean_calibration_shared_H2": ms,
                "mean_calibration_private_H2": mp,
                "held_shared_H2_mean": float(es.mean()),
                "held_private_H2_mean": float(ep.mean()),
                "shuffled_rows_moved": moved,
                "odds_quantiles": np.quantile(r, [0, 0.1, 0.5, 0.9, 1]).tolist(),
                "seeds": {},
            }
            for j, s in enumerate(SEEDS):
                c = locks[j]["coefficients"][t.name]
                a = c["independent_mode"]
                u = c["supported"]
                values = np.column_stack(
                    [
                        np.full(len(ev), a),
                        adjusted(a, r),
                        adjusted(a, shuffled),
                        np.full(len(ev), u),
                        adjusted(u, r),
                    ]
                )
                arrays[f"{t.name}.{s}"] = values
                diff = np.abs(values[:, 0] - values[:, 1])
                rec["seeds"][str(s)] = {
                    "typical_absolute_change": float(np.median(diff)),
                    "mean_absolute_change": float(diff.mean()),
                    "above_point_one": int((diff > 0.10).sum()),
                }
            meta[t.name] = rec
    coef = destination / "coefficients.npz"
    np.savez_compressed(coef, **arrays)
    write(
        destination / "inputs-and-coefficients.json",
        {
            "arms": ARMS,
            "protocol": digest(PLAN),
            "code": digest(Path(__file__)),
            "inputs": meta,
            "weights": digest(coef),
            "expert_splits": {
                str(s): digest(PREVIOUS / mode / f"seed-{s}" / "calibration-lock.json")
                for s in SEEDS
            },
            "target_row_responses_used_for_gate": False,
        },
    )
    print("frozen material-level", mode, len(meta), flush=True)


def main():
    import torch

    torch.set_num_threads(1)
    for mode in MODES:
        tasks, parts = partition(mode)
        generate(mode, tasks, parts)


if __name__ == "__main__":
    main()
