"""Score independent Italian passage ratings and English boundary reports on matched material."""

import json
from pathlib import Path

import numpy as np
from scipy.special import xlogy
from scipy.stats import rankdata

from research.reading_validation.coherence_order_data import OUT, read
from research.reading_validation.coherence_order_fit import source_targets
from research.reading_validation.local_forecast import digest

DRAWS = 10000
SEED = 378283


def point(center, draws):
    valid = np.isfinite(draws)
    if not np.isfinite(center) or not np.any(valid):
        return {"estimate": None, "ci95": None}
    return {
        "estimate": float(center),
        "ci95": np.quantile(draws[valid], [0.025, 0.975]).tolist(),
    }


def pearson(x, y):
    xc = x - x.mean(axis=-1, keepdims=True)
    yc = y - y.mean(axis=-1, keepdims=True)
    v = np.linalg.norm(xc, axis=-1) * np.linalg.norm(yc, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(
            (np.ptp(x, axis=-1) > 0) & (np.ptp(y, axis=-1) > 0),
            np.sum(xc * yc, axis=-1) / v,
            np.nan,
        )


def it_metrics(y, p, ix):
    squared = (y - p) ** 2
    absolute = abs(y - p)

    def correlations(t, m):
        return pearson(rankdata(t, axis=-1), rankdata(m, axis=-1)), pearson(t, m)

    sr, pr = correlations(y, p)
    result = {
        "mse": (squared.mean(), squared[ix].mean(1)),
        "mae": (absolute.mean(), absolute[ix].mean(1)),
        "mean_prediction": (p.mean(), p[ix].mean(1)),
    }
    s, pair = [], []
    for lo in range(0, len(ix), 160):
        sel = ix[lo : lo + 160]
        v, w = correlations(y[sel], p[sel])
        s.append(v)
        pair.append(w)
    result["spearman"] = sr, np.concatenate(s)
    result["pearson"] = pr, np.concatenate(pair)
    return result


def summary(raw, comparators, target="adaptive"):
    output = {"scores": {}, "paired_gain": {}}
    for name, items in raw.items():
        output["scores"][name] = {k: point(*v) for k, v in items.items()}
    for ctrl in comparators:
        if ctrl not in raw:
            continue
        gain = {}
        for metric, (m, b) in raw[target].items():
            if metric == "mean_prediction":
                continue
            ref, br = raw[ctrl][metric]
            sign = -1 if metric in ("mse", "mae", "logloss", "brier") else 1
            gain[metric] = point(sign * (m - ref), sign * (b - br))
        output["paired_gain"][target + "_over_" + ctrl] = gain
    return output


def italian(preds, obs, official):
    ids = preds["keys"]
    good = np.ones(len(ids), bool)
    if official:
        good = np.isfinite(preds["official_scaled"])
    y = np.array([obs[k] for k in ids[good]])
    ix = np.random.default_rng(SEED).integers(len(y), size=(DRAWS, len(y)))
    result = {}
    for name, arr in preds.items():
        if name in ("keys", "group", "document", "official_rank"):
            continue
        if name == "official_scaled" and not official:
            continue
        measures = it_metrics(y, arr[good], ix)
        if name == "official_scaled":
            ranks = it_metrics(y, preds["official_rank"][good], ix)
            measures["pearson"], measures["spearman"] = (
                ranks["pearson"],
                ranks["spearman"],
            )
        result[name] = measures
    out = summary(
        result,
        [
            "representation",
            "shared",
            "surprisal",
            "form",
            "official_scaled",
            "constant",
        ],
    )
    out.update(
        passages=len(y),
        mean_observed=float(y.mean()),
        rating_range=[float(y.min()), float(y.max())],
    )
    return out


def english_arrays(s, n, p, groups, draw):
    outcomes = []
    for k in np.unique(groups):
        q = groups == k
        pk = np.clip(p[q], 1e-12, 1 - 1e-12)
        pos = s[q]
        neg = n[q] - pos
        w = n[q].sum()
        joint = pos[:, None] * neg[None, :]
        np.fill_diagonal(joint, 0)
        direction = pk[:, None] - pk[None, :]
        concordance = (
            (joint * ((direction > 0) + 0.5 * (direction == 0))).sum() / joint.sum()
            if joint.sum()
            else np.nan
        )
        outcomes.append(
            [
                (-xlogy(pos, pk) - xlogy(neg, 1 - pk)).sum() / w,
                (pos * (1 - pk) ** 2 + neg * pk**2).sum() / w,
                n[q] @ pk / w,
                concordance,
            ]
        )
    values = np.asarray(outcomes)
    output = {}
    for j, k in enumerate(
        ("logloss", "brier", "mean_prediction", "within_article_order")
    ):
        valid = np.isfinite(values[:, j])
        sample = draw[:, valid]
        v = values[valid, j]
        with np.errstate(invalid="ignore", divide="ignore"):
            boot = sample @ v / sample.sum(1)
        output[k] = v.mean(), boot
    return output


def english(preds, obs, source):
    s, n = np.asarray([obs[k] for k in preds["keys"]]).T
    q = (n > 0) & (preds["study"] == source)
    s, n = s[q], n[q]
    groups = preds["group"][q]
    nd = len(np.unique(groups))
    draw = np.random.default_rng(SEED).multinomial(nd, np.ones(nd) / nd, size=DRAWS)
    results = {
        k: english_arrays(s, n, v[q], groups, draw)
        for k, v in preds.items()
        if k not in ("group", "document", "study", "keys")
    }
    doc_rate = [s[groups == g].sum() / n[groups == g].sum() for g in np.unique(groups)]
    out = summary(
        results,
        [
            "representation",
            "shared",
            "surprisal",
            "form",
            "constant",
            "previous_integration",
            "previous_lexical_surprisal",
        ],
    )
    out.update(
        documents=nd,
        boundaries=len(s),
        positive_marks=int(s.sum()),
        individual_judgments=int(n.sum()),
        material_weighted_human_rate=float(np.mean(doc_rate)),
        pooled_human_rate=float(s.sum() / n.sum()),
    )
    return out


def run():
    source = OUT / "forecast"
    seal = read(source / "seal.json")
    if not seal["native"] or seal["predictions"] != digest(source / "predictions.npz"):
        raise ValueError("Missing sealed native held estimates")
    if any(digest(Path(__file__).parent / k) != v for k, v in seal["code"].items()):
        raise ValueError("A registered estimator changed before assessment")
    with np.load(source / "predictions.npz", allow_pickle=False) as f:
        panels = {
            l: {k.split(".", 1)[1]: f[k] for k in f if k.startswith(l + ".")}
            for l in ("english", "italian")
        }
    native_coverage = read(OUT / "native-features.json")
    if any(
        panels["italian"]["group"][i] == panels["italian"]["group"][j]
        for i in range(len(panels["italian"]["keys"]))
        for j in range(i)
    ):
        raise ValueError(
            "Correlated anonymous held passages require a separate cluster sensitivity"
        )
    y = source_targets(held=True)
    result = {
        "italian_all": italian(panels["italian"], y["italian"], False),
        "italian_official_matched": italian(panels["italian"], y["italian"], True),
    }
    for lang in ("cohesentia", "decor"):
        result[lang] = english(panels["english"], y["english"], lang)
    evidence = "adaptive_over_representation"
    result["joint_benefit_satisfies_both"] = bool(
        result["italian_all"]["paired_gain"][evidence]["mse"]["ci95"][0] > 0
        and result["cohesentia"]["paired_gain"][evidence]["logloss"]["ci95"][0] > 0
    )
    result["forecast_seal"] = digest(source / "seal.json")
    result["bootstrap"] = {"draws": DRAWS, "seed": SEED}
    result["ineligible_boundaries"] = native_coverage["omitted_boundaries"]
    (OUT / "scores.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(
        {
            k: v["scores"]["adaptive"]
            for k, v in result.items()
            if isinstance(v, dict) and "scores" in v
        }
    )


if __name__ == "__main__":
    run()
