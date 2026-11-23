"""Test locked foot-of-word stock and post-reading ratings in crossed EMTeC folds."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import rankdata

from research.reading_validation.emtec_data import native, observation_geometry
from research.reading_validation.emtec_material import DATA, OUT
from research.reading_validation.local_forecast import digest

METHODS = ("constant", "form_and_surprisal", "strong", "strong_plus_curve")
SEED = 919841
N_BOOT = 10000


def fit(x, y, w):
    """Cumulative logits with one slope for each predictor, free cut intercepts."""
    w = np.asarray(w, np.float64)
    w /= w.sum()
    y = np.asarray(y, np.float64)
    ncut = y.shape[1]
    mean = w @ x
    std = np.sqrt(w @ ((x - mean) ** 2))
    std[std < 1e-8] = 1
    z = (x - mean) / std
    base = w @ y
    if (base <= 0).any() or (base >= 1).any():
        raise ValueError("At least one calibration threshold has no variation")
    initial = np.r_[np.zeros(z.shape[1]), logit(base)]

    def objective(p):
        slope = p[:-ncut]
        h = (z @ slope)[:, None] + p[-ncut:]
        error = (expit(h) - y) * w[:, None] / ncut
        value = w @ np.mean(np.logaddexp(0, h) - h * y, axis=1)
        return float(value + 0.005 * (slope @ slope)), np.r_[
            z.T @ error.sum(axis=1) + 0.01 * slope, error.sum(axis=0)
        ]

    result = minimize(
        objective,
        initial,
        jac=True,
        method="L-BFGS-B",
        options={"gtol": 1e-8, "ftol": 1e-13, "maxiter": 300, "maxls": 40},
    )
    gradient = float(np.max(np.abs(result.jac)))
    if not result.success or gradient > 2e-6:
        raise ArithmeticError(
            f"Invalid likelihood solution, gradient = {gradient}: {result.message}"
        )
    slope = result.x[:-ncut]
    bias = result.x[-ncut:]
    if np.any(np.diff(bias) > 1e-7):
        raise ArithmeticError("Increasing predicted threshold probabilities")
    return (
        mean,
        std,
        slope,
        bias,
        {"iterations": int(result.nit), "gradient": gradient, "slopes": slope.tolist()},
    )


def forecast(params, x):
    m, s, p, q, _ = params
    return expit((((x - m) / s) @ p)[:, None] + q)


def bce(y, p):
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log1p(-p))


def model_trials(n, g):
    sid, group = g["reader_fold"], g["prompt_fold"]
    zd = n["article"]
    observed_words = n["word"]
    visits = g["v"]
    rr = g["rr"]
    areas = g["w"]
    keys = areas * 2 + g["line"]
    word_grid = np.repeat(observed_words, 2, axis=0)
    # The original corrected eye-tracking file supplies final-line position for this display.
    all_words = np.column_stack(
        [word_grid[:, :12], np.arange(len(word_grid)) % 2, word_grid[:, 12:]]
    )
    dim_word = (0, 13, 15, len(all_words.T))
    dim_article = (0, 7, 9, len(zd.T))
    rows = g["docs"]
    rating = g["difficulty"]
    levels = (rating[:, None] > np.arange(1, 5)).astype(float)
    means = np.full((len(rows), 4, 4), np.nan)
    wp = np.full((len(areas), 4), np.nan)
    exposure = np.bincount(visits, minlength=len(rows))
    folds = []
    _, prompt = np.unique(g["prompts"], return_inverse=True)
    for u in range(4):
        for v in range(4):
            tr = (sid != u) & (group != v)
            te = (sid == u) & (group == v)
            if np.any(np.isfinite(means[te])) or np.any(np.isfinite(wp[te[visits]])):
                raise ValueError("Duplicated held forecasts")
            nv = np.bincount(prompt[tr], minlength=prompt.max() + 1)
            visit_weight = np.zeros(len(rows))
            visit_weight[tr] = 1 / nv[prompt[tr]]
            eye_tr = tr[visits]
            eye_te = te[visits]
            ev = np.bincount(prompt[tr & (exposure > 0)], minlength=prompt.max() + 1)
            ww = 1 / (ev[prompt[visits[eye_tr]]] * exposure[visits[eye_tr]])
            tot = np.bincount(keys[eye_tr], weights=ww, minlength=len(all_words))
            yes = np.bincount(
                keys[eye_tr], weights=ww * rr[eye_tr], minlength=len(all_words)
            )
            keep = tot > 0
            d = {
                "reader_fold": u,
                "prompt_fold": v,
                "test_visits": int(te.sum()),
                "test_words": int(eye_te.sum()),
                "training_visits": int(tr.sum()),
                "models": {},
            }
            for m, name in enumerate(METHODS):
                dd, wd = dim_article[m], dim_word[m]
                pf = fit(zd[rows[tr], :dd], levels[tr], visit_weight[tr])
                wf = fit(
                    all_words[keep, :wd], (yes[keep] / tot[keep])[:, None], tot[keep]
                )
                means[te, m, :] = forecast(pf, zd[rows[te], :dd])
                wp[eye_te, m] = forecast(wf, all_words[keys[eye_te], :wd])[:, 0]
                d["models"][name] = {"passage": pf[-1], "reread": wf[-1]}
            folds.append(d)
    if not np.isfinite(means).all() or not np.isfinite(wp).all():
        raise ValueError("Missing held reader/prompt predictions")
    return levels, means, wp, exposure, folds


def summarize_grid(values, subject, prompt):
    ok = np.isfinite(values).all(axis=1)
    nsub, nmat = subject.max() + 1, prompt.max() + 1
    grid = np.zeros((nsub, nmat, values.shape[1]))
    seen = np.zeros((nsub, nmat))
    if len(np.unique(subject * nmat + prompt)) != len(subject):
        raise ValueError("Reader revisited the same prompt")
    grid[subject[ok], prompt[ok]] = values[ok]
    seen[subject[ok], prompt[ok]] = 1
    base = seen.sum(axis=0)
    eligible = base > 0
    center = np.mean(grid.sum(axis=0)[eligible] / base[eligible, None], axis=0)
    rng = np.random.default_rng(SEED)
    sr = rng.multinomial(nsub, np.full(nsub, 1 / nsub), size=N_BOOT)
    pr = rng.multinomial(nmat, np.full(nmat, 1 / nmat), size=N_BOOT)
    boot = np.zeros((N_BOOT, values.shape[1]))
    flat = grid.reshape(nsub, -1)
    for start in range(0, N_BOOT, 256):
        stop = min(N_BOOT, start + 256)
        ms = sr[start:stop]
        mp = pr[start:stop]
        tot = (ms @ flat).reshape(len(ms), nmat, -1)
        den = ms @ seen
        okd = den > 0
        matw = mp * okd
        q = np.divide(
            tot, den[:, :, None], out=np.zeros_like(tot), where=okd[:, :, None]
        )
        boot[start:stop] = (matw[:, :, None] * q).sum(axis=1) / matw.sum(axis=1)[
            :, None
        ]
    return center, boot


def ci(est, draws):
    return {"estimate": float(est), "ci95": np.quantile(draws, [0.025, 0.975]).tolist()}


def report(v, s, p, *, less):
    mx, boot = summarize_grid(v, s, p)
    sign = -1.0 if less else 1.0
    return {
        "visits": int(np.isfinite(v).all(axis=1).sum()),
        "methods": {nm: ci(mx[i], boot[:, i]) for i, nm in enumerate(METHODS)},
        "gain_over_strong": ci(
            sign * (mx[-1] - mx[-2]), sign * (boot[:, -1] - boot[:, -2])
        ),
    }


def raw_corr(y, x, s, p):
    vals = np.column_stack([y, x]).astype(float)
    for _ in range(100):
        old = vals.copy()
        for gr in (s, p):
            den = np.bincount(gr)
            vals -= np.column_stack([np.bincount(gr, weights=z) / den for z in vals.T])[
                gr
            ]
        if np.max(abs(old - vals)) < 1e-10:
            break
    return np.corrcoef(vals.T)[0, 1:].tolist()


def run(extraction):
    before = json.loads((OUT / "analysis-lock.json").read_bytes())
    src = Path(__file__).parent
    if any(digest(src / k) != v for k, v in before["registered_code_sha256"].items()):
        raise ValueError("An analysis differs from the outcome-blind seal")
    n = native(extraction)
    g = observation_geometry(n, outcomes=True)
    levels, post, word, exposure, folds = model_trials(n, g)
    flags = g["rr"]
    at = g["v"]
    tot = len(g["docs"])
    loss = bce(flags[:, None], word)
    local = np.full((tot, len(METHODS)), np.nan)
    for i in range(len(METHODS)):
        local[:, i] = np.divide(
            np.bincount(at, weights=loss[:, i], minlength=tot),
            exposure,
            out=np.full(tot, np.nan),
            where=exposure > 0,
        )
    auc = np.full_like(local, np.nan)
    ix = np.argsort(at, kind="stable")
    for sl in np.split(ix, np.flatnonzero(np.diff(at[ix])) + 1):
        pos = flags[sl].astype(bool)
        if pos.all() or not pos.any():
            continue
        ranks = rankdata(word[sl], axis=0, method="average")
        k = pos.sum()
        m = len(pos) - k
        auc[at[sl[0]]] = (ranks[pos].sum(0) - k * (k + 1) / 2) / (k * m)
    _, s = np.unique(g["subjects"], return_inverse=True)
    _, p = np.unique(g["prompts"], return_inverse=True)
    results = {
        "passage_rps": report(
            ((levels[:, None, :] - post) ** 2).mean(axis=2), s, p, less=True
        ),
        "passage_threshold_logloss": report(
            bce(levels[:, None, :], post).mean(axis=2), s, p, less=True
        ),
        "word_reread_logloss": report(local, s, p, less=True),
        "word_reread_auroc_within_visit": report(auc, s, p, less=False),
        "within_reader_and_prompt_raw_numeric_rating_correlation": dict(
            zip(
                (
                    "stock",
                    "five_minus_plaba_ease",
                    "surprisal",
                    "word_length",
                    "words_per_sentence",
                ),
                raw_corr(g["difficulty"], n["raw_refs"][g["docs"]], s, p),
                strict=True,
            )
        ),
        "response_distribution": {
            "numeric_difficulty_counts": np.bincount(g["difficulty"], minlength=6)[
                1:
            ].tolist(),
            "attention_check_accuracy": float(g["attention"].mean()),
            "reread_word_count": int(flags.sum()),
            "reread_word_fraction": float(flags.mean()),
        },
        "observations": g["audit"],
        "folds": folds,
        "bootstrap_replicates": N_BOOT,
        "native_extraction": n["receipt"],
        "analysis_lock_sha256": digest(OUT / "analysis-lock.json"),
    }
    results["curve_adds_to_both_held_outcomes"] = all(
        results[k]["gain_over_strong"]["ci95"][0] > 0
        for k in ("passage_rps", "word_reread_logloss")
    )
    # Save only material-masked forecasts separately from public aggregate results.
    file = DATA / "crossed-predictions.npz"
    np.savez_compressed(
        file,
        article=post,
        word=word,
        reader_fold=g["reader_fold"],
        prompt_fold=g["prompt_fold"],
        visits=at,
        area=g["w"],
    )
    results["private_prediction_sha256"] = digest(file)
    (OUT / "scores.json").write_text(
        json.dumps(results, indent=2, allow_nan=False) + "\n"
    )
    print(
        {
            k: results[k]["gain_over_strong"]
            for k in (
                "passage_rps",
                "word_reread_logloss",
                "word_reread_auroc_within_visit",
            )
        },
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extraction", type=Path, required=True)
    run(parser.parse_args().extraction)
