"""Fit three observation-specific likelihoods, holding each cluster of real ordering topics out in full."""

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from research.reading_validation.coherence_order_data import OUT as SOURCE, read
from research.reading_validation.coherence_order_estimator import (
    LOG_SCALE_RIDGE,
    RIDGE,
    prepare,
)
from research.reading_validation.coherence_order_fit import source_targets
from research.reading_validation.columbia_order_data import OUT
from research.reading_validation.columbia_order_forecast import center
from research.reading_validation.columbia_order_score import (
    decision,
    difference,
    estimate,
    event_average,
    load_design,
    sampling,
    source_group,
)
from research.reading_validation.italian_contrast_estimator import make_objective
from research.reading_validation.local_forecast import digest

ARMS = (
    "uniform",
    "surface_and_probability",
    "frozen_readouts",
    "native_choices",
    "native_three_observations",
)
BASE = ("mean_surprisal", "tfidf", "native_cosine", "form")


def choice_design(trials):
    max_size = max(len(t.available) for t in trials)
    ix = np.zeros((len(trials), max_size), int)
    valid = np.zeros_like(ix, bool)
    chosen = np.zeros(len(ix), int)
    topic_people = defaultdict(set)
    for t in trials:
        topic_people[t.topic].add(t.person)
    parts = Counter((t.topic, t.person) for t in trials)
    w = []
    for i, t in enumerate(trials):
        n = len(t.available)
        ix[i, :n] = t.pair_indices
        valid[i, :n] = True
        chosen[i] = t.selected
        w.append(
            1
            / (
                len(topic_people)
                * len(topic_people[t.topic])
                * parts[t.topic, t.person]
                * np.log(n)
            )
        )
    return ix, valid, chosen, np.array(w)


def conditional(logits, design, *, gradient):
    ix, valid, best, w = design
    scores = np.where(valid, logits[ix], -np.inf)
    logden = logsumexp(scores, axis=1)
    correct = scores[np.arange(len(ix)), best]
    unscaled = logden - correct
    prob = np.exp(scores - logden[:, None])
    ptrue = prob[np.arange(len(ix)), best].copy()
    if not gradient:
        return unscaled / np.log(valid.sum(1)), ptrue
    prob[np.arange(len(ix)), best] -= 1.0
    contribution = np.bincount(
        ix[valid], weights=(prob * w[:, None])[valid], minlength=len(logits)
    )
    return float(w @ unscaled), contribution


def human_source():
    labels = source_targets()
    wp = SOURCE / "forecast/weights.npz"
    with (
        np.load(wp, allow_pickle=False) as w,
        np.load(SOURCE / "native-features.npz", allow_pickle=False) as f,
    ):
        ids = f["ids"]
        x = f["x"]
        data = []
        for g in ("english", "italian"):
            ix = (ids[:, 3] == ("cohesentia" if g == "english" else "italian")) & (
                ids[:, 4] == "fit"
            )
            z = (x[ix] - w[g + ".center"]) / w[g + ".scale"]
            data.append(prepare(ids[ix], z, g, labels[g]))
        beta = w["representation.shared"]
        normal = (w["english.center"], w["english.scale"])
    model = make_objective(data, None, 0.0)
    original = read(SOURCE / "forecast/seal.json")["fit"]["representation.shared"]
    a, g = model(beta)
    if abs(a - original["objective_final"]) > 1e-10 or np.max(abs(g)) > 5e-6:
        raise ArithmeticError("Registered bilingual objective cannot be reconstructed")
    return model, beta, normal


def scaled_features(q, training_mask, register):
    cols = [z for k in register for z in (k, k + ".centered")]
    a = np.column_stack([q[k] for k in cols])
    mean = a[training_mask].mean(0)
    scale = a[training_mask].std(0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    return (a - mean) / scale


def optimize(x, cases, arm, human, original, nform):
    joined = arm == "native_three_observations"
    native = arm in ("native_choices", "native_three_observations")
    ns = x.shape[1]
    p0 = np.r_[np.zeros(nform), original] if joined else np.zeros(ns)

    def objective(p):
        coef = p[:ns]
        loss, dscore = conditional(x @ coef, cases, gradient=True)
        g = x.T @ dscore
        if not joined:
            return loss + 0.5 * RIDGE * (coef @ coef), g + RIDGE * coef
        theta = p[:nform]
        old = p[nform:]
        beta = old[:-3]
        j, hg = human(old)
        old_pen = 0.5 * RIDGE * (beta @ beta) + 0.5 * LOG_SCALE_RIDGE * old[-1] ** 2
        pen_grad = np.r_[RIDGE * beta, 0.0, 0.0, LOG_SCALE_RIDGE * old[-1]]
        loss = loss / 3 + 2 * j / 3 + old_pen / 3 + 0.5 * RIDGE * (theta @ theta)
        grad = np.r_[g[:nform] / 3 + RIDGE * theta, 2 * hg / 3 + pen_grad / 3]
        grad[nform:ns] += g[nform:] / 3
        return loss, grad

    rng = np.random.default_rng(451307)
    probe = p0 + 0.025 * rng.normal(size=len(p0))
    analytic = objective(probe)[1]
    delta = []
    for direction in rng.normal(size=(2, len(p0))):
        direction /= np.linalg.norm(direction)
        v = (
            objective(probe + 1e-6 * direction)[0]
            - objective(probe - 1e-6 * direction)[0]
        ) / (2e-6)
        delta.append(float(abs(v - analytic @ direction)))
    if max(delta) > 2e-6:
        raise ArithmeticError(
            "Conditional categorical/shared gradient differs from finite differences"
        )
    bounds = (
        None
        if not joined
        else [(None, None)] * (len(p0) - 1) + [(-np.log(4), np.log(4))]
    )
    solution = minimize(
        objective,
        p0,
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        options={
            "maxiter": 650,
            "maxls": 64,
            "ftol": 1e-14,
            "gtol": 1e-8,
            "maxcor": 22,
        },
    )
    grad = objective(solution.x)[1].copy()
    if joined and (
        solution.x[-1] <= -np.log(4) + 1e-9
        and grad[-1] > 0
        or solution.x[-1] >= np.log(4) - 1e-9
        and grad[-1] < 0
    ):
        grad[-1] = 0.0
    norm = float(np.max(abs(grad)))
    if not solution.success or norm > 5e-6:
        raise ArithmeticError(
            f"Human-choice fit {arm} did not converge: {solution.message}; {norm}"
        )
    record = {
        "start": float(objective(p0)[0]),
        "finish": float(solution.fun),
        "steps": int(solution.nit),
        "projected_gradient": norm,
        "fd_max": max(delta),
        "coefficient_norm": float(np.linalg.norm(solution.x)),
        "conditional_choice_slopes": ns,
        "source_scale": float(np.exp(solution.x[-1])) if joined else None,
    }
    if not native and ns != nform:
        raise ValueError("Changed reference parameterization")
    return solution.x[:ns], record


def run_crossfit(q, keys, x, trials, human, old, normal):
    topics = sorted({t.topic for t in trials})
    group, draw = sampling(topics)
    families = sorted(set(group))
    observation = {s: [] for s in ARMS}
    alltrials = []
    fits = {}
    z = (x - normal[0]) / normal[1]
    context = np.column_stack([center(z[:, i], keys) for i in range(z.shape[1])])
    pairs = []
    for cluster in families:
        fit = [t for t in trials if source_group(t.topic) != cluster]
        held = [t for t in trials if source_group(t.topic) == cluster]
        df = choice_design(fit)
        dt = choice_design(held)
        topics_fit = {t.topic for t in fit}
        training = np.array([k.split("/")[0] in topics_fit for k in keys])
        ref = scaled_features(q, training, BASE)
        latent = scaled_features(q, training, (*BASE, "adaptive", "augmented"))
        native = np.column_stack([ref, -context])
        notes = {}
        alltrials.extend(held)
        if set(t.topic for t in held) & topics_fit:
            raise ValueError("Whole topic appeared on both sides")
        for arm, features in [
            ("uniform", ref),
            ("surface_and_probability", ref),
            ("frozen_readouts", latent),
            ("native_choices", native),
            ("native_three_observations", native),
        ]:
            if arm == "uniform":
                score = np.zeros(len(keys))
                notes[arm] = {"conditional_choice_slopes": 0}
            else:
                nsource = (
                    ref.shape[1] if arm.startswith("native") else features.shape[1]
                )
                coefs, notes[arm] = optimize(features, df, arm, human, old, nsource)
                score = features @ coefs
            nll, pr = conditional(score, dt, gradient=False)
            row = []
            for j, t in enumerate(held):
                rank, exact = decision(score[list(t.pair_indices)], t.selected)
                row.append([nll[j], rank, exact, pr[j]])
            observation[arm].extend(row)
            held_keys = np.array(
                [k.split("/")[0] in {t.topic for t in held} for k in keys]
            )
            pairs.append(
                (arm, cluster, score[held_keys].copy(), keys[held_keys].copy())
            )
        fits[cluster] = notes
    scores = {}
    by_topic = {}
    metric = (
        "normalized_logloss",
        "pairwise_choice",
        "top_one",
        "actual_choice_probability",
    )
    for arm, values in observation.items():
        res = event_average(alltrials, values, topics)
        by_topic[arm] = res
        scores[arm] = {name: estimate(res[:, i], draw) for i, name in enumerate(metric)}
    contrasts = {
        k: difference(by_topic[a][:, 0] - by_topic[b][:, 0], draw, group, 2)
        for k, a, b in [
            (
                "reading_features_over_reference",
                "surface_and_probability",
                "frozen_readouts",
            ),
            (
                "joint_over_matched_native",
                "native_choices",
                "native_three_observations",
            ),
        ]
    }
    return {
        "topic_sets": len(topics),
        "families": len(families),
        "human_choices": len(trials),
        "scores": scores,
        "two_primary": contrasts,
        "by_topic": {
            k: {t: v.tolist() for t, v in zip(topics, m, strict=True)}
            for k, m in by_topic.items()
        },
        "by_topic_metrics": list(metric),
        "source_only_fit_and_convergence": fits,
    }, pairs


def main():
    if not (OUT / "scores.json").is_file():
        raise ValueError(
            "Keep the frozen external ordering result independent and grade it first"
        )
    forecasts, trials, _, _ = load_design()
    src = OUT / "native-features.npz"
    if digest(src) != read(OUT / "forecast-seal.json")["native_features"]:
        raise ValueError("Unknown paired native features")
    with np.load(src, allow_pickle=False) as z:
        keys, x = z["keys"], z["x"]
    human, old, normal = human_source()
    d, pairs = run_crossfit(forecasts, keys, x, trials, human, old, normal)
    file = OUT / "crossfit-predictions.npz"
    np.savez_compressed(
        file,
        **{f"{arm}.{g}": v for arm, g, v, _ in pairs},
        **{f"{g}.keys": k for _, g, _, k in pairs},
    )
    d.update(
        {
            "predictions": digest(file),
            "protocol": digest(Path(__file__).with_name("columbia-choice-training.md")),
        }
    )
    (OUT / "crossfit-scores.json").write_text(
        json.dumps(d, indent=2, allow_nan=False) + "\n"
    )
    print(d["two_primary"])


if __name__ == "__main__":
    main()
