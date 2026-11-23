"""Nested, crossed reader/article prediction using study-specific ordinal and binomial losses."""

import json
from collections import Counter
from hashlib import sha256
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit

from research.reading_validation.local_forecast import digest, rows
from research.reading_validation.relay_summary_material import OUT, PRIVATE, SCORES

TARGETS = (*SCORES, "correct")
ARMS = (
    "constant",
    "independent",
    "shared",
    "adaptive",
    "independent_prior",
    "shared_prior",
    "adaptive_prior",
)
RIDGE = 0.1


def matrix(records, stage0, prior):
    xx = np.asarray([r["features"] for r in records], float)
    neutral = [v for v in [xx] if v.size]
    if stage0:
        neutral.append(
            np.array(
                [
                    [
                        r["scores"][k]
                        for k in ("plaba_ease", "readme", "noun_ease", "noun_correct")
                    ]
                    for r in records
                ]
            )
        )
    if prior:
        neutral.append(np.asarray([r["prior"] for r in records])[:, None])
    return np.column_stack(neutral)


def fold_ids(keys, salt, n):
    unique = sorted(set(keys), key=lambda k: sha256((salt + str(k)).encode()).digest())
    lookup = {k: j % n for j, k in enumerate(unique)}
    return np.array([lookup[k] for k in keys], int)


def splits(records, n, salt):
    pid = fold_ids([r["reader"] for r in records], salt + "/person/", n)
    gid = fold_ids([r["family"] for r in records], salt + "/article/", n)
    for p in range(n):
        for g in range(n):
            train = (pid != p) & (gid != g)
            test = (pid == p) & (gid == g)
            if test.any():
                yield train, test


def weights(readers):
    counts = Counter(readers)
    return np.asarray([1 / (len(counts) * counts[r]) for r in readers])


def observations(y, binomial):
    return y[:, None] / 4 if binomial else (y[:, None] > np.arange(1, 5)).astype(float)


def loss(pred, y, binomial):
    if binomial:
        v = pred[:, 0]
        p = y / 4
        return p * (1 - v) ** 2 + (1 - p) * v**2
    return np.mean((observations(y, False) - pred) ** 2, axis=1)


def fit_predict(xx, y, readers, train, test, binomial):
    x = xx[train]
    w = weights(readers[train])
    n = len(x)
    gold = y[train]
    mu = np.average(x, axis=0, weights=w)
    sd = np.sqrt(np.average((x - mu) ** 2, axis=0, weights=w))
    sd = np.where(sd > 1e-7, sd, 1)
    z = (x - mu) / sd
    held = (xx[test] - mu) / sd
    width = z.shape[1]
    if binomial:
        # One half success and one half failure at the centered input keep the intercept finite.
        obs = np.r_[gold / 4, 0.5]
        ww = np.r_[w * 4 * n, 1]
        ww /= ww.sum()
        z = np.r_[z, np.zeros((1, width))]
        init = np.r_[np.zeros(width), logit(np.clip(ww @ obs, 1e-6, 1 - 1e-6))]

        def fun(v):
            q = z @ v[:width] + v[-1]
            d = ww * (expit(q) - obs)
            return float(
                ww @ (np.logaddexp(0, q) - obs * q)
                + RIDGE * (v[:width] @ v[:width]) / 2
            ), np.r_[z.T @ d + RIDGE * v[:width], d.sum()]

        bds = None
    else:
        # P(Y=k) = F(cut_k - input) - F(cut_(k-1) - input).
        obs = np.r_[gold, np.arange(1, 6)].astype(int)
        ww = np.r_[w * n, np.full(5, 0.5)]
        ww /= ww.sum()
        z = np.r_[z, np.zeros((5, width))]
        cut = logit(np.cumsum(np.bincount(obs, weights=ww, minlength=6)[1:])[:-1])
        init = np.r_[np.zeros(width), cut[:1], np.diff(cut)]
        bds = [(None, None)] * (width + 1) + [(0.001, None)] * 3
        idx = np.arange(len(obs))

        def fun(v):
            cut = np.cumsum(v[width:])
            p = expit(cut[None, :] - (z @ v[:width])[:, None])
            bounded = np.column_stack([np.zeros(len(obs)), p, np.ones(len(obs))])
            cat = bounded[idx, obs] - bounded[idx, obs - 1]
            if np.any(cat <= 0):
                raise ArithmeticError("Degenerate ordinal category interval")
            der = np.zeros_like(bounded)
            der[idx, obs] += -ww / cat
            der[idx, obs - 1] += ww / cat
            d = der[:, 1:-1] * p * (1 - p)
            dc = d.sum(0)
            return -float(ww @ np.log(cat)) + RIDGE * (
                v[:width] @ v[:width]
            ) / 2, np.r_[-z.T @ d.sum(1) + RIDGE * v[:width], np.cumsum(dc[::-1])[::-1]]

    fitted = minimize(
        fun,
        init,
        jac=True,
        method="L-BFGS-B",
        bounds=bds,
        options={"maxiter": 250, "ftol": 1e-13, "gtol": 1e-8, "maxls": 40},
    )
    err = np.max(abs(fitted.jac))
    if not fitted.success or err > 2e-5:
        raise ArithmeticError(
            f"Response likelihood did not converge: {fitted.message}, {err}"
        )
    v = fitted.x
    score = held @ v[:width]
    ans = (
        expit(score + v[-1])[:, None]
        if binomial
        else expit(score[:, None] - np.cumsum(v[width:])[None, :])
    )
    return ans, float(err)


def joined(include_six=False):
    p = PRIVATE / "static-visits.jsonl"
    q = OUT / "text-scores.jsonl"
    if (
        digest(p)
        != json.loads((OUT / "inputs.json").read_bytes())["private_response_sha256"]
        or digest(q)
        != json.loads((OUT / "forecast-seal.json").read_bytes())["text_score_sha256"]
    ):
        raise ValueError("Human observations or frozen text-only forecasts changed")
    store = {r["article"]: r for r in rows(q)}
    return [
        {**row, **store[row["article"]]}
        for row in rows(p)
        if (row["regular"] or include_six)
        and row["eligible"]
        and store[row["article"]]["included"]
    ]


def adapt(records):
    count = len(records)
    readers = np.array([r["reader"] for r in records])
    out = {}
    evidence = {}
    largest = 0
    fits = 0
    designs = {
        (share, prior): matrix(records, share, prior)
        for prior in (False, True)
        for share in (False, True)
    }
    xv = np.empty((count, 0))
    for endpoint in TARGETS:
        y = np.asarray([r[endpoint] for r in records])
        binomial = endpoint == "correct"
        d = 1 if binomial else 4
        bank = {arm: np.full((count, d), np.nan) for arm in ARMS}
        alphas = {False: [], True: []}
        for train, test in splits(records, 5, "relay-outer-v1"):
            tr = np.flatnonzero(train)
            sub = [records[i] for i in tr]
            pc, err = fit_predict(xv, y, readers, train, test, binomial)
            bank["constant"][test] = pc
            largest = max(largest, err)
            fits += 1
            for prior in (False, True):
                suffix = "_prior" if prior else ""
                nested = {
                    shared: np.full((len(tr), d), np.nan) for shared in (False, True)
                }
                outer = {}
                for shared in (False, True):
                    x = designs[shared, prior]
                    for t, v in splits(sub, 2, "relay-inner-v1"):
                        forecast, err = fit_predict(
                            x[tr], y[tr], readers[tr], t, v, binomial
                        )
                        nested[shared][v] = forecast
                        largest = max(largest, err)
                        fits += 1
                    outer[shared], err = fit_predict(
                        x, y, readers, train, test, binomial
                    )
                    largest = max(largest, err)
                    fits += 1
                if not all(np.isfinite(z).all() for z in nested.values()):
                    raise ArithmeticError("Missing crossed calibration observations")
                w = weights(readers[tr])
                delta = nested[True] - nested[False]
                human = observations(y[tr], binomial)
                den = w @ (delta**2).mean(1)
                raw = (
                    (w @ ((human - nested[False]) * delta).mean(1)) / den
                    if den > 1e-14
                    else 0.0
                )
                beta = float(np.clip(raw, 0, 1))
                alphas[prior].append(beta)
                bank["independent" + suffix][test] = outer[False]
                bank["shared" + suffix][test] = outer[True]
                bank["adaptive" + suffix][test] = outer[False] + beta * (
                    outer[True] - outer[False]
                )
        for name, pred in bank.items():
            if (
                not np.isfinite(pred).all()
                or np.any(pred < 0)
                or np.any(pred > 1)
                or (not binomial and np.any(np.diff(pred, axis=1) > 1e-12))
            ):
                raise ArithmeticError("Invalid out-of-sample probability predictions")
            out[endpoint + "." + name] = pred
        evidence[endpoint] = {
            str(k): {
                "min": min(a),
                "median": float(np.median(a)),
                "max": max(a),
                "positive": sum(i > 0 for i in a),
                "folds": len(a),
            }
            for k, a in alphas.items()
        }
    return out, {
        "calibrations": fits,
        "max_abs_gradient": largest,
        "source_mixture": evidence,
    }


def main():
    sets = {}
    info = {}
    for label, use_six in (("primary", False), ("six_visit_sensitivity", True)):
        rec = joined(use_six)
        out, diagnostic = adapt(rec)
        p = PRIVATE / (label + "-predictions.npz")
        np.savez_compressed(p, **out)
        sets[label] = digest(p)
        info[label] = diagnostic
    (OUT / "trained-seal.json").write_text(
        json.dumps(
            {
                "private_predictions_sha256": sets,
                "diagnostics": info,
                "training_script_sha256": digest(Path(__file__)),
                "estimator_sha256": digest(
                    Path(__file__).with_name("relay-summary-estimator.md")
                ),
                "published_row_level_responses": False,
            },
            indent=2,
        )
        + "\n"
    )
    print({k: q["calibrations"] for k, q in info.items()})


if __name__ == "__main__":
    main()
