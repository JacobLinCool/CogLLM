"""Fit shared ordinal readouts for sentence ratings and aligned rewrite effects."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from research.reading_validation.local_features import CACHE, RESULTS, build, verified_rows
from research.reading_validation.local_forecast import digest, rows


LEVELS = np.arange(1, 6, dtype=float)
RIDGE = 0.025
SEED = 83103
TARGETS = ("ease", "understanding")
ARMS = {
    "registered_form": ("condition",),
    "text_form": ("surface",),
    "medical": ("surface", "medical"),
    "sentence_state": ("surface", "sentence_state"),
    "sentence_state_local": ("surface", "medical", "sentence_state"),
    "local": ("surface", "medical", "lexical", "representation"),
    "local_and_persistent": ("surface", "medical", "lexical", "representation", "shared", "persistent"),
    "full_state_local": ("surface", "medical", "lexical", "sentence_state"),
}


def full_states():
    root = CACHE / "backbone-transfer/l4/cogllm-backbone-qwen-20261121T103619Z-58de9ae9/extraction"
    seal = json.loads((root / "seal.json").read_bytes())
    for name in ("states.npy", "coordinates.jsonl"):
        if digest(root / name) != seal["file_sha256"][name]:
            raise ValueError("Qwen transfer archive failed its checksum")
    keys = {
        (doc["id"], node["stop"]): node["anchor_id"]
        for doc in verified_rows("plaba", "ratings.jsonl")
        for node in doc["ratings"]
    }
    source = np.load(root / "states.npy", mmap_mode="r", allow_pickle=False)
    result = {}
    n = 0
    for item in rows(root / "coordinates.jsonl"):
        for stop in item["stops"]:
            key = keys.get((item["id"], stop))
            if key is not None:
                v = source[n].astype(float)
                result[key] = v / np.linalg.norm(v)
            n += 1
    if n != len(source) or set(result) != set(keys.values()):
        raise ValueError("The transfer archive does not cover all PLABA sentence ends")
    return result


def geometry(nodes):
    kset, y, folds, groups = [], [], [], []
    positions = {}
    for doc in verified_rows("plaba", "ratings.jsonl"):
        for r in doc["ratings"]:
            positions[doc["id"], r["stop"]] = len(kset)
            kset.append(r["anchor_id"])
            y.append([r[target] for target in TARGETS])
            folds.append(doc["fold"])
            groups.append(doc["group_id"])
    n = len(kset)
    pairs, pgroups = [], []
    for row in verified_rows("plaba", "pairs.jsonl"):
        weights = np.zeros(n)
        for col, sign in [("plain", 1.0), ("original", -1.0)]:
            refs = [positions[tuple(k)] for k in row[col]]
            if any(folds[idx] != row["fold"] for idx in refs):
                raise ValueError("Paired versions must share the original fold")
            weights[refs] = sign / len(refs)
        pairs.append(weights)
        pgroups.append(row["group_id"])
    if len(kset) != 609 or len(set(groups)) != 25 or len(pairs) != 265:
        raise ValueError("PLABA original component folds differ")
    return kset, np.array(y), np.array(folds), np.array(groups), np.array(pairs), np.array(pgroups)


def prepare_design(nodes, keys, columns, train):
    design, params = [], {}
    for c in columns:
        data = np.stack([nodes[k].sentence[c] for k in keys])
        mu = data[train].mean(axis=0)
        sd = np.maximum(data[train].std(axis=0), 1e-7)
        if c in ("representation", "sentence_state", "shared"):
            sd *= np.sqrt(data.shape[1] / 4)
        design.append((data - mu) / sd)
        params[c] = dict(mean=mu.tolist(), scale=sd.tolist())
    return np.column_stack(design), params


def estimate(x, y, pairs):
    n, d = x.shape
    q = y / y.sum(axis=-1, keepdims=True)
    human = q @ LEVELS
    centered = LEVELS - 3.0

    def function(raw):
        slope = raw[: d * 2].reshape(d, 2)
        offset = raw[d * 2 :].reshape(2, 5)
        z = (x @ slope)[:, :, None] * centered[None, None, :] + offset
        logp = z - logsumexp(z, axis=2, keepdims=True)
        p = np.exp(logp)
        loss = -(q * logp).sum() / (2 * n)
        dz = (p - q) / (2 * n)
        # Pair loss balances matched original/plain versions and their mean effect.
        diff = pairs @ ((p @ LEVELS) - human)
        loss += 0.075 * (diff**2).mean() + 0.075 * (diff.mean(axis=0) ** 2).mean()
        dmean = pairs.T @ (0.075 * (diff + diff.mean(axis=0)) / len(diff))
        dz += dmean[:, :, None] * p * (LEVELS - (p @ LEVELS)[:, :, None])
        loss += 0.5 * RIDGE * np.sum(slope * slope) + 0.125 * RIDGE * np.sum(
            (slope[:, 0] - slope[:, 1]) ** 2
        )
        slopes = x.T @ (dz @ centered) + RIDGE * slope
        delta = 0.25 * RIDGE * (slope[:, 0] - slope[:, 1])
        slopes[:, 0] += delta
        slopes[:, 1] -= delta
        return float(loss), np.r_[slopes.ravel(), dz.sum(axis=0).ravel()]

    initial = np.r_[np.zeros(d * 2), np.log(q.mean(axis=0)).ravel()]
    fit = minimize(
        function,
        initial,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 350, "ftol": 1e-12, "gtol": 1e-7, "maxls": 45},
    )
    gm = float(np.max(np.abs(fit.jac)))
    if not fit.success or gm > 2e-5:
        raise ValueError(f"Ordinal readout failed to converge: {fit.message}, {gm}")
    w = fit.x[: d * 2].reshape(d, 2)
    intercept = fit.x[d * 2 :].reshape(2, 5)

    def predictor(x):
        z = (x @ w)[:, :, None] * centered + intercept
        return np.exp(z - logsumexp(z, axis=-1, keepdims=True))

    return predictor, dict(
        coef=w.tolist(), offset=intercept.tolist(), max_abs_gradient=gm, updates=int(fit.nit)
    )


def intervals(data, group_ids):
    groups = sorted(set(group_ids))
    mapping = np.array([groups.index(g) for g in group_ids])
    draws = np.random.default_rng(SEED).multinomial(
        len(groups), np.ones(len(groups)) / len(groups), size=10000
    )
    denom = draws @ np.bincount(mapping, minlength=len(groups))
    s = np.zeros((len(groups), data.shape[1]))
    np.add.at(s, mapping, data)
    sampled = (draws @ s) / denom[:, None]
    return data.mean(axis=0), sampled


def scores(pred, y, groups, pairs, pgroups, arms):
    observed = y / y.sum(axis=-1, keepdims=True)
    human = observed @ LEVELS
    by_arm = {}
    for name in arms:
        p = pred[name]
        mean = p @ LEVELS
        by_arm[name] = dict(
            nll=-(observed * np.log(p)).sum(axis=-1),
            mae=np.abs(mean - human),
            rewrite_mae=np.abs(pairs @ (mean - human)),
            rewrite_effect=pairs @ mean,
            human_effect=pairs @ human,
        )
    report = {}
    for endpoint, j in zip(TARGETS, range(2)):
        report[endpoint] = {}
        for metric in by_arm[arms[0]]:
            vals = np.column_stack([by_arm[a][metric][:, j] for a in arms])
            point, boot = intervals(
                vals, pgroups if metric in ("rewrite_mae", "rewrite_effect", "human_effect") else groups
            )

            def summarize(x, b):
                return dict(mean=float(x), interval=np.quantile(b, [0.025, 0.975]).tolist())

            out = {a: summarize(point[i], boot[:, i]) for i, a in enumerate(arms)}
            for aa, bb in [
                ("local", "registered_form"),
                ("local", "text_form"),
                ("full_state_local", "historical_frozen"),
                ("sentence_state", "registered_form"),
                ("sentence_state_local", "historical_frozen"),
                ("local_and_persistent", "local"),
                ("medical", "registered_form"),
                ("sentence_state_local", "sentence_state"),
                ("full_state_local", "sentence_state_local"),
            ]:
                if aa in arms and bb in arms:
                    a, b = arms.index(aa), arms.index(bb)
                    out[aa + " minus " + bb] = summarize(point[a] - point[b], boot[:, a] - boot[:, b])
            report[endpoint][metric] = out
    return report


def assay(out, gpu=None):
    nodes, source = build(gpu)
    hidden = full_states()
    for key, node in nodes.items():
        surface = node.sentence["surface"]
        node.sentence["condition"] = surface[[0, 1, 2, 3, 4, 5, 6]]
        node.sentence["surface"] = surface[[0, 1, 2, 4, 5, 6]]
        if key in hidden:
            node.sentence["sentence_state"] = hidden[key]
    keys, y, folds, groups, pairs, pgroups = geometry(nodes)
    arms = {k: v for k, v in ARMS.items() if all(f in nodes[keys[0]].sentence for f in v)}
    pred, weights, deploy = {}, {}, {}
    for arm, cols in arms.items():
        p = np.full(y.shape, np.nan, dtype=float)
        weights[arm] = {}
        for f in [0, 1, 2, 3, 4, "all"]:
            train = np.ones(len(y), dtype=bool) if f == "all" else folds != f
            if f != "all" and set(groups[train]) & set(groups[~train]):
                raise ValueError("Training and held-out reader components overlap")
            x, norm = prepare_design(nodes, keys, cols, train)
            chosen = np.abs(pairs[:, ~train]).sum(axis=1) == 0
            if (pairs[chosen][:, train] @ np.ones(train.sum()) > 1e-10).any():
                raise ValueError("Unbalanced version weights")
            predict, params = estimate(x[train], y[train], pairs[chosen][:, train])
            if f == "all":
                deploy[arm] = dict(features=cols, normalization=norm, **params)
            else:
                p[~train] = predict(x[~train])
                weights[arm][str(f)] = {
                    k: v for k, v in params.items() if k in ("max_abs_gradient", "updates")
                }
        if not np.isfinite(p).all() or not np.allclose(p.sum(axis=-1), 1):
            raise ValueError("Unfilled fold or unnormalized prediction")
        pred[arm] = p
    historical = rows(RESULTS / "expository-lora-l4/predictions.jsonl")
    historical = {
        (r["id"], stop): p
        for r in historical
        if r["dataset"] == "plaba" and r["model"] == "frozen"
        for stop, p in zip(r["stops"], r["probabilities"], strict=True)
    }
    lookup = [(r["id"], s["stop"]) for r in verified_rows("plaba", "ratings.jsonl") for s in r["ratings"]]
    pred["historical_frozen"] = np.array([historical[k] for k in lookup])
    out.mkdir(exist_ok=True, parents=True)
    np.savez_compressed(out / "predictions.npz", keys=keys, folds=folds, **pred)
    (out / "deployment-weights.json").write_text(json.dumps(deploy, separators=(",", ":")) + "\n")
    result = dict(
        metrics=scores(pred, y, groups, pairs, pgroups, list(pred)),
        source=source,
        convergence=weights,
        bootstrap_seed=SEED,
        draws=10000,
        reader_connected_groups=len(set(groups)),
        folds=5,
        sentences=len(keys),
        rewrite_pairs=len(pairs),
        gpu_sha256=digest(gpu / "seal.json") if gpu else None,
        predictions_sha256=digest(out / "predictions.npz"),
        historical_exposure=True,
    )
    (out / "scores.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        {
            endpoint: {k: round(v["mean"], 4) for k, v in result["metrics"][endpoint]["nll"].items()}
            for endpoint in TARGETS
        }
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu", type=Path)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    assay(a.out, a.gpu)
