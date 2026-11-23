"""Jointly fit the probability of actual word responses and changes in population spread."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax

from research.backbone_transfer.extract import digest, read_json, write_json
from research.reading_validation.cloze_population_data import OUT as POP
from research.reading_validation.cloze_population_data import ROOT
from research.reading_validation.cloze_population_fit import assign, data
from research.reading_validation.cloze_population_model import distribution, project

OUT = Path("research/results/reading-validation-word-dispersion")
PLAN = Path(__file__).with_name("word-population-dispersion-protocol.md")
LOG_PRIOR = np.log([0.1, 0.8, 0.1])
EXPO = np.array([2 / 3, 1, 1.5])


def components(logp):
    z = logp[:, :, None] * EXPO
    z -= logsumexp(z, axis=1, keepdims=True)
    return np.exp(z)


def measurement(choices, indices):
    rows = []
    cols = []
    counts = []
    observed = []
    for k, i in enumerate(indices):
        c = choices.counts[i]
        n = c.sum()
        if n < 2:
            raise ValueError("Two distinct responses are required")
        rows.extend([k] * len(c))
        cols.extend(choices.indices[i])
        counts.extend(c)
        observed.append(1.0 - c @ (c - 1) / (n * (n - 1.0)))
    return tuple(np.array(v) for v in (rows, cols, counts, observed))


def adjacent(stories, positions):
    lookup = {(s, t): j for j, (s, t) in enumerate(zip(stories, positions))}
    if len(lookup) != len(stories):
        raise ValueError("Duplicate word locations")
    return np.array(
        [(j, lookup[s, t + 1]) for (s, t), j in lookup.items() if (s, t + 1) in lookup]
    ).T


def gates(x, w):
    z = x @ w.reshape(x.shape[1], 2)
    return softmax(
        np.column_stack((z[:, 0], np.zeros(len(x)), z[:, 1])) + LOG_PRIOR, axis=1
    )


def fit(X, gram, choices, response, step, with_change):
    row, p, hits = choices
    a, b = step
    observed_up = response[b] * (1 - response[a])
    observed_down = response[a] * (1 - response[b])
    norm = (observed_up + observed_down).sum()

    def objective(co):
        g = gates(X, co)
        likelihood = np.einsum("ij,ij->i", g[row], p)
        freq = hits / hits.sum()
        val = -(freq @ np.log(likelihood))
        grad_g = np.zeros_like(g)
        np.add.at(grad_g, row, -(freq / likelihood)[:, None] * p)
        if with_change:
            gq = np.einsum("ij,ijk->ik", g, gram)
            d = np.clip(1.0 - np.einsum("ij,ij->i", gq, g), 1e-11, 1.0 - 1e-11)
            logodds = np.log(d) - np.log1p(-d)
            update = logodds[b] - logodds[a]
            val += (
                observed_up @ np.logaddexp(0, -update)
                + observed_down @ np.logaddexp(0, update)
            ) / norm
            q = 1.0 / (1.0 + np.exp(-update))
            err = (q * (observed_up + observed_down) - observed_up) / norm
            grad_l = np.bincount(b, weights=err, minlength=len(X)) - np.bincount(
                a, weights=err, minlength=len(X)
            )
            grad_g -= 2 * gq * (grad_l / (d * (1.0 - d)))[:, None]
        grad_z = g * (grad_g - (g * grad_g).sum(1, keepdims=True))
        return val + 0.5 * (co @ co) / len(X), (
            X.T @ grad_z[:, [0, 2]]
        ).ravel() + co / len(X)

    w = np.zeros(X.shape[1] * 2)
    direction = np.random.default_rng(7104).normal(size=len(w))
    discrepancy = abs(
        (objective(w + 1e-5 * direction)[0] - objective(w - 1e-5 * direction)[0]) / 2e-5
        - objective(w)[1] @ direction
    )
    if discrepancy > 1e-6:
        raise ArithmeticError(
            "Composite likelihood gradient disagrees with finite differences"
        )
    sol = minimize(
        objective,
        w,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 500, "gtol": 1e-9, "ftol": 1e-15},
    )
    station = float(np.max(np.abs(objective(sol.x)[1])))
    if not sol.success or station > 2e-7:
        raise ArithmeticError(
            f"Observation model failed to converge: {sol.message} ({station})"
        )
    return {
        "coefficients": sol.x.tolist(),
        "loss": float(sol.fun),
        "gradient_inf": station,
        "iterations": int(sol.nit),
        "composite": with_change,
        "gradient_check": discrepancy,
    }


def train(folder):
    OUT.mkdir(exist_ok=True)
    raw, S, U, V, hum, meta = data(folder)
    train_ids = np.flatnonzero(meta["source"])
    ext_ids = np.flatnonzero(~meta["source"])
    article = meta["articles"]
    fold = assign(article[train_ids])
    trmap = {i: j for j, i in enumerate(train_ids)}
    exmap = {i: j for j, i in enumerate(ext_ids)}
    row, col, hits, observed = measurement(hum, train_ids)
    offsets = np.searchsorted(row, np.arange(len(train_ids) + 1))
    chosen = np.full((len(row), 3), np.nan)
    X = np.full((len(raw), 4), np.nan)
    gram = np.full((len(raw), 3, 3), np.nan)
    ext_file = ROOT / "dispersion-external-logp.npy"
    logp_out = np.lib.format.open_memmap(
        ext_file, mode="w+", dtype="float64", shape=(len(ext_ids), raw.shape[1])
    )
    for f in range(6):
        idx = (
            np.array([i for i in train_ids if fold[article[i]] == f])
            if f < 5
            else ext_ids
        )
        spec = read_json(POP / f"fold-{f}.json")
        if f < 5 and set(meta["ids"][idx]) & set(spec["training_ids"]):
            raise ValueError("Source sentence leaked into its base predictions")
        for k in range(0, len(idx), 32):
            ii = idx[k : k + 32]
            logp = distribution(
                raw[ii],
                project(S[ii], spec["geometry"]),
                spec["models"]["context"],
                U,
                V,
            )
            p = components(logp)
            base = p[:, :, 1]
            X[ii] = np.column_stack(
                [
                    np.ones(len(ii)),
                    base.max(1),
                    base[:, -1],
                    -(base * logp).sum(1) / np.log(base.shape[1]),
                ]
            )
            gram[ii] = np.einsum("ijk,ijl->ikl", p, p)
            for n, i in enumerate(ii):
                if f < 5:
                    j = trmap[i]
                    sl = slice(offsets[j], offsets[j + 1])
                    chosen[sl] = p[n, col[sl]]
                else:
                    logp_out[exmap[i]] = logp[n]
    logp_out.flush()
    if not all(np.isfinite(v).all() for v in (chosen, gram, X)):
        raise ValueError("Missing original responses or distribution geometry")
    center = X[train_ids, 1:].mean(0)
    scale = X[train_ids, 1:].std(0)
    X[:, 1:] = (X[:, 1:] - center) / scale
    step = adjacent(article[train_ids], meta["pos"][train_ids])
    models = {
        k: fit(
            X[train_ids],
            gram[train_ids],
            (row, chosen, hits),
            observed,
            step,
            k == "joint",
        )
        for k in ("choices", "joint")
    }
    gate = {
        k: gates(X[ext_ids], np.array(m["coefficients"])) for k, m in models.items()
    }
    # Sealing these gates and all base probabilities fixes the full categorical forecast.
    filename = OUT / "held-gates.npz"
    np.savez_compressed(
        filename,
        ids=meta["ids"][ext_ids],
        article=article[ext_ids],
        pos=meta["pos"][ext_ids],
        **gate,
    )
    write_json(
        OUT / "forecast-lock.json",
        {
            "protocol": digest(PLAN),
            "native": digest(folder / "receipt.json"),
            "frozen_base_fits": [digest(POP / f"fold-{f}.json") for f in range(6)],
            "held_logp": digest(ext_file),
            "held_gates": digest(filename),
            "source_prefixes": len(train_ids),
            "source_choices": int(hits.sum()),
            "source_adjacent_steps": len(step[0]),
            "source_center": center.tolist(),
            "source_scale": scale.tolist(),
            "models": models,
            "forecast_articles": len(set(article[ext_ids])),
            "external_outcomes_opened_in_training": False,
        },
    )
    print(json.dumps(models), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("native", type=Path)
    train(p.parse_args().native)
