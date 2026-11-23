"""Use distributions of real expert grades to fit a source-story-conditional ordering model."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, logsumexp

from research.reading_validation.coherence_order_data import read
from research.reading_validation.coherence_order_estimator import equal_group_weights
from research.reading_validation.local_forecast import digest
from research.reading_validation.memory_data import h
from research.reading_validation.summeval_data import OUT
from research.reading_validation.summeval_forecast import (
    archive_data,
    judgments,
    open_old,
    ridge,
)
from research.reading_validation.summeval_score import evaluate, paired, result


def distributions(keys, story, geo, visits):
    scores = []
    for k, g in zip(keys, story, strict=True):
        systems = [s for s, v in geo[g]["systems"].items() if v == k]
        numbers = [val for s in systems for val in visits[g, s][0]]
        if not numbers:
            raise ValueError("No expert vote for a registered draft")
        scores.append(np.bincount(numbers, minlength=6)[1:] / len(numbers))
    return np.array(scores)


def pair_data(vote, groups):
    pair = []
    source = []
    q = []
    n = len(groups)
    if vote.shape != (n, 5) or not np.allclose(vote.sum(1), 1):
        raise ValueError("Malformed expert-vote distribution")
    credit = np.tril(np.ones((5, 5)), -1) + 0.5 * np.eye(5)
    for k in np.unique(groups):
        ind = np.flatnonzero(groups == k)
        a, b = np.triu_indices(len(ind), 1)
        left = ind[a]
        right = ind[b]
        q.extend(np.einsum("ij,jk,ik->i", vote[left], credit, vote[right]))
        pair.extend(zip(left, right, strict=True))
        source.extend([k] * len(a))
    return np.array(pair).T, np.asarray(q), np.asarray(source)


def fit_order(x, vote, groups):
    scale_weights = equal_group_weights(groups)
    center = scale_weights @ x
    scale = np.sqrt(scale_weights @ ((x - center) ** 2))
    scale = np.where(scale > 1e-8, scale, 1.0)
    z = (x - center) / scale
    (a, b), q, source = pair_data(vote, groups)
    delta = z[a] - z[b]
    weights = equal_group_weights(source)
    lam = 0.01

    def objective(beta):
        logits = delta @ beta
        loss = weights @ (np.logaddexp(0, logits) - q * logits) + 0.5 * lam * (
            beta @ beta
        )
        return loss, (weights * (expit(logits) - q)) @ delta + lam * beta

    x0 = np.zeros(x.shape[1])
    r = np.random.default_rng(7191)
    probe = 0.03 * r.normal(size=len(x0))
    grad = objective(probe)[1]
    err = []
    for vec in r.normal(size=(3, len(x0))):
        vec /= np.linalg.norm(vec)
        eps = 1e-6
        err.append(
            abs(
                (objective(probe + eps * vec)[0] - objective(probe - eps * vec)[0])
                / (2 * eps)
                - grad @ vec
            )
        )
    opt = minimize(
        objective,
        x0,
        jac=True,
        method="L-BFGS-B",
        options={"ftol": 1e-14, "gtol": 1e-9, "maxiter": 600, "maxls": 50},
    )
    diff = float(np.abs(objective(opt.x)[1]).max())
    if not opt.success or diff > 5e-6 or max(err) > 2e-6:
        raise ArithmeticError("Within-story expert ballots failed to converge")
    return {"center": center, "scale": scale, "slope": opt.x}, {
        "target_source_articles": len(set(groups)),
        "total_distinct_draft_pairs": len(source),
        "native_dim": len(opt.x),
        "start": float(objective(x0)[0]),
        "finish": float(opt.fun),
        "max_gradient": diff,
        "max_fd_error": float(max(err)),
        "iterations": int(opt.nit),
    }


def forecast(p, x):
    return ((x - p["center"]) / p["scale"]) @ p["slope"]


def risk(groups, probs, eta):
    (a, b), q, src = pair_data(probs, groups)
    logits = eta[a] - eta[b]
    obs = np.logaddexp(0, logits) - q * logits
    return np.array([obs[src == k].mean() for k in np.unique(groups)])


def calibrate(archive):
    before = read(OUT / "instrument-lock.json")
    for filename, sha in before["files"].items():
        if digest(Path(__file__).with_name(filename)) != sha:
            raise ValueError("The locked source artifact changed")
    ids, x, _, aux, _, rec = archive_data(archive)
    p, _ = open_old()
    old = read(OUT / "prediction-seal.json")
    if digest(OUT / "frozen-predictions.npz") != old["frozen"]:
        raise ValueError("Fixed predictions changed")
    with np.load(OUT / "frozen-predictions.npz", allow_pickle=False) as a:
        keys = a["keys"]
        story = a["stories"]
        roles = a["roles"]
    idx = roles == "calibration"
    g = story[idx]
    geo = read(OUT / "story-geometry.json")
    visits = judgments("calibration")
    v = distributions(keys[idx], g, geo, visits)
    y = v @ np.arange(1, 6)
    z = np.clip((x - p["english.center"]) / p["english.scale"], -8, 8)
    form = np.asarray([aux[k]["form"] for k in keys])
    m = []
    for k in keys:
        region = z[ids[:, 1] == k]
        if len(region) == 0:
            raise ValueError("No native boundary for a rated draft")
        m.append(np.r_[region.mean(0), logsumexp(region, axis=0) - np.log(len(region))])
    full = np.column_stack([form, m])
    test_groups = sorted(
        set(g), key=lambda a: h("cogllm-summeval-pair-mixture-v1/" + a)
    )[:8]
    inner = np.isin(g, test_groups)
    logits = []
    pred = {}
    notes = {}
    weights = {}
    for label, basis in [("form", form), ("native", full)]:
        xt = basis[idx]
        prior, first = fit_order(xt[~inner], v[~inner], g[~inner])
        logits.append(forecast(prior, xt[inner]))
        allp, rep = fit_order(xt, v, g)
        pred[label + "_choice"] = forecast(allp, basis)
        pred[label + "_absolute"] = ridge(xt, y, g)(basis)
        notes[label] = {"source_calibration": rep, "inner": first}
        for name, matrix in allp.items():
            weights[label + "." + name] = matrix

    def metric(a):
        return float(
            risk(g[inner], v[inner], (1 - a) * logits[0] + a * logits[1]).mean()
        )

    o = minimize_scalar(
        metric, bounds=(0, 1), method="bounded", options={"xatol": 1e-12}
    )
    if not o.success:
        raise ArithmeticError(
            "Failed to select a study weight from calibration-only stories"
        )
    alpha = float(min((0.0, o.x, 1.0), key=metric))
    pred["mixture_choice"] = (1 - alpha) * pred["form_choice"] + alpha * pred[
        "native_choice"
    ]
    pref = OUT / "ballot-forecasts.npz"
    params = OUT / "ballot-weights.npz"
    np.savez_compressed(pref, keys=keys, **pred)
    np.savez_compressed(params, alpha=alpha, **weights)
    info = {
        "registered_instrument": digest(OUT / "instrument-lock.json"),
        "public_capture": rec["file"],
        "before_confirm": True,
        "source_story_mixture": {
            "n": len(test_groups),
            "groups": test_groups,
            "native_weight": alpha,
            "form_loss": metric(0.0),
            "native_loss": metric(1.0),
            "selected_loss": metric(alpha),
        },
        "optimizers": notes,
        "forecast_sha256": digest(pref),
        "weights_sha256": digest(params),
    }
    (OUT / "ballot-forecast-seal.json").write_text(json.dumps(info, indent=2) + "\n")
    print("within-story source mixture", alpha)


def score():
    before = read(OUT / "ballot-forecast-seal.json")
    prior = read(OUT / "prediction-seal.json")
    for name, key, obj in [
        ("ballot-forecasts.npz", "forecast_sha256", before),
        ("frozen-predictions.npz", "frozen", prior),
        ("calibrated-predictions.npz", "calibrated", prior),
    ]:
        if digest(OUT / name) != obj[key]:
            raise ValueError("Frozen picks changed before confirmation")
    with (
        np.load(OUT / "ballot-forecasts.npz", allow_pickle=False) as b,
        np.load(OUT / "frozen-predictions.npz", allow_pickle=False) as f,
        np.load(OUT / "calibrated-predictions.npz", allow_pickle=False) as n,
    ):
        if not np.array_equal(b["keys"], f["keys"]) or not np.array_equal(
            f["keys"], n["keys"]
        ):
            raise ValueError("Different eligible drafts")
        idx = f["roles"] == "confirm"
        g = f["stories"][idx]
        keys = f["keys"][idx]
        methods = {k: v[idx] for k, v in b.items() if k != "keys"}
        controls = {
            "geval": f["geval"][idx],
            "shared_native": n["shared_native"][idx],
            "adaptive": f["adaptive"][idx],
            "form_calibrated": n["form_calibrated"][idx],
            "system_mean": n["system_mean"][idx],
        }
    v = distributions(keys, g, read(OUT / "story-geometry.json"), judgments("confirm"))
    y = v @ np.arange(1, 6)
    groups = np.unique(g)
    r = {name: risk(g, v, e) for name, e in methods.items() if name.endswith("_choice")}
    r["uniform"] = risk(g, v, np.zeros(len(g)))
    overall = {n: evaluate(y, vec, groups, g)[0] for n, vec in methods.items()}
    primary = {
        "ballot_native_minus_absolute_grade_native_order": paired(
            y, methods["native_choice"], methods["native_absolute"], groups, g, 2
        ),
        "adaptive_over_form_ballot_logloss": result(
            r["form_choice"] - r["mixture_choice"], 2
        ),
    }
    secondary = {
        name: paired(y, methods["mixture_choice"], vec, groups, g, 1)
        for name, vec in {**controls, "native_choice": methods["native_choice"]}.items()
    }
    out = {
        "groups": len(groups),
        "drafts": len(g),
        "true_eligible_judgment_pairs": len(pair_data(v, g)[1]),
        "pair_grade_prob_logloss": {name: result(s) for name, s in r.items()},
        "ordering": overall,
        "primary": primary,
        "secondary_adaptive_minus_other_ordering": secondary,
        "forecasts": digest(OUT / "ballot-forecast-seal.json"),
    }
    (OUT / "ballot-scores.json").write_text(
        json.dumps(out, indent=2, allow_nan=False) + "\n"
    )
    print(primary)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fit", type=Path)
    p.add_argument("--score", action="store_true")
    args = p.parse_args()
    if bool(args.fit) == args.score:
        p.error("Select either --fit ARCHIVE or --score")
    if args.fit:
        calibrate(args.fit)
    else:
        score()
