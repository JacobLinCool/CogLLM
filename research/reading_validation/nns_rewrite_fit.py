"""Fit independent and adaptively shared unknown-word and revision-preference likelihoods."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.special import expit, logit, logsumexp, softmax, xlogy

from research.reading_validation.local_forecast import digest
from research.reading_validation.nns_features import pack_model_input, selected_counts
from research.reading_validation.nns_fit import (
    all_features,
    family_weights,
    prior,
    solve_mix,
)
from research.reading_validation.nns_material import OUT as SELF
from research.reading_validation.nns_material import identifier
from research.reading_validation.nns_rewrite_features import comparisons, word_features
from research.reading_validation.nns_rewrite_material import DATA, OUT

RIDGE = 0.003
ARMS = ("lexical", "contextual")


def folds(key):
    groups = sorted(set(key), key=lambda s: identifier("nns-rewrite/crossfit/" + s))
    mapping = {s: i % 4 for i, s in enumerate(groups)}
    return np.array([mapping[k] for k in key])


def weights(doc, groups):
    _, d = np.unique(doc, return_inverse=True)
    _, g = np.unique(groups, return_inverse=True)
    return family_weights(np.ones(len(doc)), d, g)


def multiclass_mix(a, b, y, w):
    diff = b - a

    def gradient(alpha):
        return -float(w @ np.sum(y * diff / np.maximum(1e-12, a + alpha * diff), 1))

    if gradient(0) >= 0:
        return 0.0
    if gradient(1) <= 0:
        return 1.0
    return float(brentq(gradient, 0, 1))


def ternary(d, t):
    return np.column_stack([d, -d, np.full(len(d), t)])


def standardize(unknown, w, a, b, r):
    mean = (w @ unknown) / 2 + (r @ a + r @ b) / 4
    second = (w @ (unknown**2)) / 2 + (r @ (a**2) + r @ (b**2)) / 4
    sd = np.sqrt(np.maximum(0, second - mean**2))
    sd[sd < 1e-7] = 1
    return mean, sd


def train(x, y, w, contrast, vote, r, *, coupling, initial=None):
    p = x.shape[1]
    hu0 = w @ y
    ht = r @ vote[:, 2]
    hprob = np.array([(1 - ht) / 2, (1 - ht) / 2, ht])
    hu = -np.sum(xlogy([hu0, 1 - hu0], [hu0, 1 - hu0]))
    hr = -np.sum(xlogy(hprob, hprob))
    if min(hu, hr) <= 0:
        raise ValueError("No task variation on training set")
    if initial is None:
        initial = np.r_[np.zeros(2 * p), logit(hu0), np.log(2 * ht / (1 - ht))]

    def f(q):
        u = q[:p]
        s = q[p : 2 * p]
        off, t = q[-2:]
        z = x @ u + off
        d = contrast @ s
        eta = ternary(d, t)
        lu = w @ (np.logaddexp(0, z) - y * z)
        lr = r @ (logsumexp(eta, axis=1) - np.sum(eta * vote, axis=1))
        du = (w * (expit(z) - y)) * (0.5 / hu)
        dr = (r[:, None] * (softmax(eta, axis=1) - vote)) * (0.5 / hr)
        delta = u - s
        cost = (
            0.5 * (lu / hu + lr / hr)
            + RIDGE / 4 * (u @ u + s @ s)
            + coupling / 2 * (delta @ delta)
        )
        grad = np.r_[
            x.T @ du + RIDGE / 2 * u + coupling * delta,
            contrast.T @ (dr[:, 0] - dr[:, 1]) + RIDGE / 2 * s - coupling * delta,
            du.sum(),
            dr[:, 2].sum(),
        ]
        return float(cost), grad

    result = minimize(
        f,
        initial,
        jac=True,
        method="L-BFGS-B",
        options={"ftol": 1e-12, "gtol": 1e-7, "maxiter": 240, "maxls": 40},
    )
    g = float(np.max(abs(result.jac)))
    if not result.success or g > 7e-6:
        raise ArithmeticError(
            f"Joint readout convergence failed, gradient {g}: {result.message}"
        )
    return result.x, {
        "maximum_absolute_gradient": g,
        "iterations": int(result.nit),
        "objective": float(result.fun),
    }


def predict(theta, u, a, b):
    p = u.shape[1]
    v = expit(u @ theta[:p] + theta[-2])
    d = (b - a) @ theta[p : 2 * p] / 2
    return v, softmax(ternary(d, theta[-1]), axis=1)


def calibrated_surprisal(bank, lexical, training, testing, output, params):
    """Fit the same three-way preference likelihood using form and LM probabilities."""
    nll = bank["surprisal"]
    measured = bank["measured_surprisal"].all(1)
    covered = measured[training["pair"]].all(1)
    chosen = training["pair"][covered]
    w = weights(training["doc"][covered], training["family"][covered])
    counts = training["counts"][covered]
    y = counts / counts.sum(1, keepdims=True)
    # The second column reflects the continuation following the replacement.
    x = np.column_stack(
        [lexical, np.log1p(nll[:, 0]), np.log1p(np.maximum(0, nll[:, 1] - nll[:, 0]))]
    )
    a, b = x[chosen[:, 0]], x[chosen[:, 1]]
    center = (w @ a + w @ b) / 2
    sg = np.sqrt((w @ (a - center) ** 2 + w @ (b - center) ** 2) / 2)
    sg = np.where(sg > 1e-7, sg, 1.0)
    z = (b - a) / (2 * sg)
    tie = float(w @ y[:, 2])
    base = np.array([(1 - tie) / 2, (1 - tie) / 2, tie])
    entropy = -xlogy(base, base).sum()

    def objective(theta):
        eta = ternary(z @ theta[:-1], theta[-1])
        nll = w @ (logsumexp(eta, axis=1) - np.sum(y * eta, axis=1))
        e = w[:, None] * (softmax(eta, axis=1) - y) / entropy
        grad = np.r_[z.T @ (e[:, 0] - e[:, 1]) + RIDGE * theta[:-1], e[:, 2].sum()]
        return float(nll / entropy + RIDGE * (theta[:-1] @ theta[:-1]) / 2), grad

    fit = minimize(
        objective,
        np.r_[np.zeros(z.shape[1]), np.log(2 * tie / (1 - tie))],
        jac=True,
        method="L-BFGS-B",
        options={"ftol": 1e-12, "gtol": 1e-7, "maxiter": 240},
    )
    g = float(np.max(abs(fit.jac)))
    if not fit.success or g > 7e-6:
        raise ArithmeticError("Unconverged matched language-model baseline")
    ia, ib = testing["pair"].T
    output["reference.common"] = measured[testing["pair"]].all(1)
    output["reference.target"] = nll[ib, 0] - nll[ia, 0]
    output["reference.continuation"] = nll[ib, 1] - nll[ia, 1]
    output["reference.learned"] = softmax(
        ternary((x[ib] - x[ia]) / (2 * sg) @ fit.x[:-1], fit.x[-1]), axis=1
    )
    params["reference.theta"] = fit.x
    params["reference.scale"] = sg
    return {
        "source_pairs": int(covered.sum()),
        "held_pairs": int(output["reference.common"].sum()),
        "maximum_absolute_gradient": g,
    }


def run(extracted):
    tick = perf_counter()
    a, b = [pack_model_input(s) for s in ("sparse", "dense")]
    c = word_features(extracted)
    np.savez_compressed(DATA / "word-features.npz", **c)
    reports, counts, coverage = selected_counts("sparse", a)
    observed = counts.sum(1) > 0
    a = {
        k: a[k][observed]
        for k in ("doc", "family", "stem", "form", "source", "context")
    }
    reports, counts = reports[observed], counts[observed]
    wu0 = np.arange(len(reports))
    train_r = comparisons(c, "train", outcomes=True)
    test_r = comparisons(c, "held", outcomes=False)
    na = len(a["doc"])
    nb = len(b["doc"])
    nc = len(c["doc"])
    off = na + nb
    ib = np.arange(na, off)
    ic = np.arange(off, off + nc)
    # Earlier original word forecasts have been sealed independently of the held responses.
    original = json.loads((SELF / "forecast-seal.json").read_bytes())
    wp = SELF / "weights.npz"
    if digest(wp) != original["weights_sha256"]:
        raise ValueError("Original word-study estimator changed")
    base = {
        k: np.concatenate([o[k] for o in (a, b, c)])
        for k in ("form", "source", "context")
    }
    docs = np.concatenate([o["doc"] for o in (a, b, c)])
    group = np.concatenate([o["family"] for o in (a, b, c)])
    _, fidx = np.unique(group, return_inverse=True)
    _, types = np.unique(
        np.concatenate([o["stem"] for o in (a, b, c)]), return_inverse=True
    )
    del a
    y = reports[wu0].sum(1) / counts[wu0].sum(1)
    yf = train_r["counts"] / train_r["counts"].sum(1, keepdims=True)
    training_r = train_r["pair"] + off
    held_r = test_r["pair"] + off
    uf = folds(group[wu0])
    rf = folds(train_r["family"])
    output = {
        "u_doc": b["doc"],
        "u_family": b["family"],
        "u_coords": b["coords"],
        "r_keys": test_r["keys"],
        "r_doc": test_r["doc"],
        "r_family": test_r["family"],
    }
    params = {}
    diag = {}
    raw_x = {}
    mix_audit = {}
    ww = weights(docs[wu0], group[wu0])
    rw = weights(train_r["doc"], train_r["family"])
    for arm in ARMS:
        cv_u = {key: np.full(len(y), np.nan) for key in ("independent", "shared")}
        cv_r = {key: np.full_like(yf, np.nan) for key in cv_u}
        pred = {}
        fits = []
        for f in range(5):
            final = f == 4
            use_u = np.arange(len(wu0)) if final else np.flatnonzero(uf != f)
            use_r = np.arange(len(rf)) if final else np.flatnonzero(rf != f)
            it = wu0[use_u]
            eval_u = ib if final else wu0[uf == f]
            eval_r = held_r if final else training_r[rf == f]
            w = weights(docs[it], group[it])
            r = weights(train_r["doc"][use_r], train_r["family"][use_r])
            obs = y[use_u]
            vr = yf[use_r]

            # Every source word prior excludes the material family that supplied its own label.
            def x(index, subtract=False, source=it, features=arm):
                index = np.asarray(index)
                rates = prior(source, index, reports, counts, types, fidx, subtract)
                return all_features(
                    base,
                    np.column_stack([index, np.zeros(len(index), int)]),
                    rates,
                    features,
                    False,
                )

            xu = x(it, True)
            ur = x(eval_u)
            both = np.unique(
                np.r_[
                    training_r[use_r].ravel(),
                    eval_r.ravel(),
                    ic if final else np.empty(0, int),
                ]
            )
            zz = x(both)
            work = zz[np.searchsorted(both, training_r[use_r])]
            rr = zz[np.searchsorted(both, eval_r)]
            m, s = standardize(xu, w, work[:, 0, :], work[:, 1, :], r)
            xu = (xu - m) / s
            work = (work - m) / s
            ur = (ur - m) / s
            rr = (rr - m) / s
            contrast = (work[:, 1, :] - work[:, 0, :]) / 2
            theta = None
            for mode, lam in (("independent", 0.0), ("shared", RIDGE)):
                theta, info = train(
                    xu, obs, w, contrast, vr, r, coupling=lam, initial=theta
                )
                gu, gr = predict(theta, ur, rr[:, 0, :], rr[:, 1, :])
                info.update(fold=f, mode=mode)
                fits.append(info)
                if final:
                    pred[mode] = (gu, gr)
                    params[arm + "." + mode + ".theta"] = theta
                else:
                    cv_u[mode][uf == f] = gu
                    cv_r[mode][rf == f] = gr
            if final:
                params[arm + ".mean"] = m
                params[arm + ".scale"] = s
                raw_x[arm] = zz[np.searchsorted(both, ic)]
        if not all(np.isfinite(v).all() for d in (cv_u, cv_r) for v in d.values()):
            raise ValueError("Incomplete source predictions")
        au = solve_mix(cv_u["independent"], cv_u["shared"], y, ww)
        ar = multiclass_mix(cv_r["independent"], cv_r["shared"], yf, rw)
        pred["adaptive"] = tuple(
            (1 - alpha) * q + alpha * z
            for alpha, q, z in zip(
                (au, ar), pred["independent"], pred["shared"], strict=True
            )
        )
        for kind, (gu, gr) in pred.items():
            output["u." + arm + "." + kind] = gu
            output["r." + arm + "." + kind] = gr
        for name, value in (("u", au), ("r", ar)):
            params[arm + "." + name + ".fraction"] = value
        mix_audit[arm] = {"word_fraction": au, "revision_fraction": ar}
        diag[arm] = fits
    # Apply the original word study, frozen before the substitution study was graded, directly to the held choices.
    with np.load(wp, allow_pickle=False) as frozen:
        pp = {}
        for arm in ARMS:
            nm = "text." + arm
            zz = raw_x[arm]
            th = frozen[nm + ".theta"]
            pp[arm] = expit(
                (zz - frozen[nm + ".mean"]) / frozen[nm + ".scale"] @ th[:-1] + th[-1]
            )
        alpha = float(frozen["text.mixture"])
        pp["adaptive"] = (1 - alpha) * pp["lexical"] + alpha * pp["contextual"]
    raw = {
        "length": c["form"][:, 0],
        "rarity": -c["form"][:, 2],
        "curriculum": np.where(c["form"][:, 7] > 0, 7, c["form"][:, 6]),
        "plaba": c["source"][:, 1],
        "complex": c["source"][:, 0],
        **pp,
    }
    for nm, v in raw.items():
        output["zero." + nm] = v[test_r["pair"]]
    reference = calibrated_surprisal(
        c, raw_x["lexical"], train_r, test_r, output, params
    )
    outfile = OUT / "forecasts.npz"
    model = OUT / "weights.npz"
    np.savez_compressed(outfile, **output)
    np.savez_compressed(model, **params)
    seal = {
        "forecasts_sha256": digest(outfile),
        "weights_sha256": digest(model),
        "definition_sha256": digest(Path(__file__)),
        "sources_sha256": digest(Path(__file__).with_name("nns_rewrite_features.py")),
        "material_sha256": digest(OUT / "material.json"),
        "original_word_study_forecasts_sha256": digest(SELF / "forecast-seal.json"),
        "elapsed_seconds": perf_counter() - tick,
        "unknown_word_training_rows": int(counts.sum()),
        "unknown_word_training_words": len(y),
        "unknown_word_training_coverage": coverage,
        "revision_training_annotations": int(train_r["counts"].sum()),
        "revision_training_pairs": len(yf),
        "revision_held_pairs_without_judgments": len(test_r["pair"]),
        "revision_training_coverage": train_r["coverage"],
        "fractions": mix_audit,
        "convergence": diag,
        "language_model_reference": reference,
        "unknown_word_held_labels_inspected": False,
        "revision_held_labels_inspected": False,
        "extraction": json.loads((extracted / "extraction.json").read_bytes()),
    }
    (OUT / "forecast-seal.json").write_text(
        json.dumps(seal, indent=2, allow_nan=False) + "\n"
    )
    print(mix_audit, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extraction", type=Path, required=True)
    run(parser.parse_args().extraction)
