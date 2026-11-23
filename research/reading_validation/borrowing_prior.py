"""Estimate an excluded-study hyperprior for normalized observation distributions."""

import argparse
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
import json

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import logsumexp

from research.backbone_transfer.extract import digest
from research.reading_validation.borrowing_data import (
    CALIBRATE,
    EVALUATE,
    OUTPUT as HISTORICAL,
    data,
)
from research.reading_validation.borrowing_score import GeometricPool, material_weights
from research.reading_validation.source_directions import OUT as DIRECTIONS, fresh

OUT = Path("research/results/reading-validation-borrowing-prior")
PLAN = Path(__file__).with_name("borrowing-prior-protocol.md")
MODES = ("historical", "fresh")
SEEDS = (1729, 2718)
METHODS = (
    "private",
    "shared",
    "midpoint",
    "independent_mode",
    "independent_mean",
    "transported",
    "supported",
)
LOG2 = np.log(2.0)
MAX_SHAPE = np.log(32.0)
SIGMA = 1.5


def read(path):
    return json.loads(path.read_bytes())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def partition(mode):
    cache, tasks, p = data()
    del cache
    if mode == "fresh":
        p = fresh(tasks)
    if mode not in MODES:
        raise ValueError("Unknown material split")
    return tasks, p


def expert_dir(mode, seed):
    return (
        HISTORICAL if mode == "historical" else DIRECTIONS / "fresh"
    ) / f"seed-{seed}"


def experts(opened, mode, seed, stage, tasks, roles):
    directory = expert_dir(mode, seed)
    receipt = read(directory / "fit.json")
    out = []
    for name in ("shared", "private"):
        path = directory / (name + "-" + stage + ".npz")
        if digest(path) != receipt["files"][path.name]:
            raise ValueError("A trained expert or partition changed")
        v = opened.enter_context(np.load(path, allow_pickle=False))
        out.append(v)
    counts = {t.name: np.bincount(roles[t.name], minlength=4).tolist() for t in tasks}
    if counts != receipt["roles"]:
        raise ValueError(
            "Material source fold counts differ from the original training run"
        )
    return tuple(out)


def grid(n):
    a, w = np.polynomial.legendre.leggauss(n)
    return (1 + a) / 2, w / 2


class Marginal:
    def __init__(self, nodes, base, loss, endpoint_mass):
        self.nodes = nodes
        self.lognodes = np.column_stack([np.log(nodes), np.log1p(-nodes)])
        self.base = np.log(base)
        self.loss = loss
        self.endpoint_mass = endpoint_mass

    def probabilities(self, shapes):
        lp = self.base + self.lognodes @ (shapes - 1)
        lp -= logsumexp(lp)
        return lp

    def value(self, z):
        shapes = np.exp(z)
        lp = self.probabilities(shapes)
        postlogs = lp - self.loss
        evidence = logsumexp(postlogs, axis=1)
        post = np.exp(postlogs - evidence[:, None])
        prior = np.exp(lp)
        weight = self.endpoint_mass
        objective = -weight @ evidence + 0.5 * np.sum(((z - LOG2) / SIGMA) ** 2)
        gradient = (
            -(weight @ (post - prior[None, :])) @ self.lognodes * shapes
            + (z - LOG2) / SIGMA**2
        )
        return float(objective), gradient

    def fit(self):
        starts = [(2.0, 2.0), (1.0, 8.0), (8.0, 1.0), (16.0, 16.0)]
        sol = []
        for q in starts:
            p = minimize(
                self.value,
                np.log(q),
                jac=True,
                method="L-BFGS-B",
                bounds=[(0.0, MAX_SHAPE)] * 2,
                options={"maxiter": 250, "ftol": 1e-14, "gtol": 5e-7, "maxls": 50},
            )
            v, g = self.value(p.x)
            grad = g.copy()
            grad[(p.x <= 1e-8) & (g > 0)] = 0
            grad[(p.x >= MAX_SHAPE - 1e-8) & (g < 0)] = 0
            if not p.success or np.abs(grad).max() > 1e-5:
                raise ArithmeticError(
                    f"Unconverged prior optimization {p.message}, projected gradient {grad}"
                )
            sol.append((v, p.x, float(np.abs(grad).max()), p.nit))
        value, x, gradient, steps = min(sol, key=lambda v: v[0])
        test = np.log([2.3, 3.4])
        analytic = self.value(test)[1]
        num = []
        for unit in np.eye(2):
            eps = 2e-6
            num.append(
                (self.value(test + eps * unit)[0] - self.value(test - eps * unit)[0])
                / (2 * eps)
            )
        error = float(np.max(abs(analytic - num)))
        if error > 2e-5:
            raise ArithmeticError("Beta-evidence gradient failed")
        return np.exp(x), {
            "objective": value,
            "at_uniform_weak_prior": self.value(np.full(2, LOG2))[0],
            "projected_gradient": gradient,
            "finite_difference_error": error,
            "iterations": int(steps),
            "start_objective_range": float(
                max(v[0] for v in sol) - min(v[0] for v in sol)
            ),
        }


def post(nodes, logprior, loss):
    weight = logprior - loss
    return float(np.exp(weight - logsumexp(weight)) @ nodes)


def calibration_curves(tasks, parts, blocks, grids):
    matrices = {k: [] for k in grids}
    info = []
    for task in tasks:
        ix = np.flatnonzero(parts[task.name] == CALIBRATE)
        source = task.source
        if not len(ix):
            for k, (u, w) in grids.items():
                matrices[k].append(np.zeros(len(u)))
            info.append(
                {
                    "endpoint": task.name,
                    "source": source,
                    "likelihood": task.likelihood,
                    "rows": 0,
                    "families": 0,
                    "independent_mode": 0.5,
                }
            )
            continue
        pool = GeometricPool(task, ix, *(r[task.name] for r in blocks))
        w, f = material_weights(task, ix)

        def loss(alpha):
            return float(w @ pool.loss(alpha))

        anchor = loss(0.5)
        for k, (u, _) in grids.items():
            matrices[k].append(np.array([loss(a) - anchor for a in u]))
        h = minimize_scalar(
            lambda a: loss(a) - np.log(a) - np.log1p(-a),
            method="bounded",
            bounds=(1e-7, 1 - 1e-7),
            options={"xatol": 2e-8},
        )
        if not h.success:
            raise ArithmeticError("Independent reference optimizer failed")
        info.append(
            {
                "endpoint": task.name,
                "source": source,
                "likelihood": task.likelihood,
                "rows": len(ix),
                "families": f,
                "independent_mode": float(h.x),
            }
        )
    return {k: np.asarray(v) for k, v in matrices.items()}, info


def calibrate_grid(gridspec, loss, info):
    nodes, w = gridspec
    sources = np.array([r["source"] for r in info])
    n = np.array([r["families"] for r in info])
    records = {}
    coef = {}
    for held in sorted(set(sources)):
        train = (sources != held) & (n > 0)
        names = sources[train]
        freq = Counter(names)
        mass = np.asarray([1 / freq[s] for s in names])
        m = Marginal(nodes, w, loss[train], mass)
        if held in freq or abs(mass.sum() - len(freq)) > 1e-10:
            raise ValueError("The target study entered its own hyperprior")
        shape, diag = m.fit()
        lp = m.probabilities(shape)
        weak = m.probabilities(np.array([2.0, 2.0]))
        mu = float(shape[0] / shape.sum())
        records[held] = {
            "alpha_shape": float(shape[0]),
            "beta_shape": float(shape[1]),
            "mean": mu,
            "training_sources": len(freq),
            "training_endpoints": int(train.sum()),
            "training_source_equal_mass_max_error": max(
                abs(mass[names == s].sum() - 1) for s in freq
            ),
            "convergence": diag,
        }
        for idx in np.flatnonzero(sources == held):
            v = info[idx]
            s = post(nodes, lp, loss[idx]) if n[idx] else mu
            coef[v["endpoint"]] = {
                "private": 0.0,
                "shared": 1.0,
                "midpoint": 0.5,
                "independent_mode": v["independent_mode"],
                "independent_mean": post(nodes, weak, loss[idx]) if n[idx] else 0.5,
                "transported": mu,
                "supported": s,
            }
    return records, coef


def calibrate(mode, seed, tasks, parts):
    out = OUT / mode / f"seed-{seed}"
    out.mkdir(parents=True, exist_ok=True)
    grids = {n: grid(n) for n in (97, 193)}
    with ExitStack() as opened:
        fs = experts(opened, mode, seed, "calibration", tasks, parts)
        curves, info = calibration_curves(tasks, parts, fs, grids)
    low, high = [calibrate_grid(grids[n], curves[n], info) for n in grids]
    lh, lc = low
    hh, hc = high
    err = max(abs(lc[t][m] - hc[t][m]) for t in lc for m in METHODS)
    objective = max(
        abs(lh[s]["convergence"]["objective"] - hh[s]["convergence"]["objective"])
        / hh[s]["training_sources"]
        for s in hh
    )
    chosen = 193 if err > 0.0005 or objective > 0.005 else 97
    source, c = low if chosen == 97 else high
    if mode == "historical":
        old = read(expert_dir(mode, seed) / "borrowing-lock.json")["coefficients"]
        delta = max(abs(c[k]["independent_mode"] - v["weight"]) for k, v in old.items())
        if delta > 1e-7:
            raise ArithmeticError(
                "Historical independently calibrated coefficient changed"
            )
    else:
        delta = None
    curvepath = out / "calibration-curves.npz"
    np.savez_compressed(
        curvepath,
        **{f"grid{n}": grids[n][0] for n in grids},
        **{f"weights{n}": grids[n][1] for n in grids},
        **{f"loss{n}": x for n, x in curves.items()},
        endpoints=np.array([v["endpoint"] for v in info]),
    )
    record = {
        "mode": mode,
        "seed": seed,
        "protocol": digest(PLAN),
        "code": digest(Path(__file__)),
        "experts": digest(expert_dir(mode, seed) / "fit.json"),
        "human_training_endpoint_count": len(info),
        "human_source_count": len(source),
        "any_evaluation_prediction_opened": False,
        "historical_mode_max_error": delta,
        "source_hyperpriors": source,
        "coefficients": c,
        "calibration_rows_and_families": info,
        "quadrature": {
            "selected": chosen,
            "max_abs_coefficient_difference": err,
            "max_abs_objective_difference_per_training_source": objective,
        },
        "calibration_curves": digest(curvepath),
    }
    write(out / "calibration-lock.json", record)
    print(mode, seed, len(source), len(info), record["quadrature"], flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=MODES, nargs="+", default=list(MODES))
    p.add_argument("--seed", type=int, choices=SEEDS, nargs="+", default=list(SEEDS))
    args = p.parse_args()
    import torch

    torch.set_num_threads(1)
    for mode in args.mode:
        tasks, parts = partition(mode)
        for seed in args.seed:
            calibrate(mode, seed, tasks, parts)


if __name__ == "__main__":
    main()
