"""Fit direct word-flag probabilities on sparse sentences and seal dense forecasts."""

import argparse
import json
import resource
import sys
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.special import expit, logit

from research.reading_validation.local_forecast import digest
from research.reading_validation.nns_features import (
    build,
    pack_model_input,
    selected_counts,
)
from research.reading_validation.nns_material import OUT

ARMS = ("lexical", "human_scores", "contextual")
RIDGE = 0.003


def family_weights(n, doc, family):
    per_doc = np.bincount(doc, minlength=doc.max() + 1)
    fam_doc = np.unique(np.stack([family, doc], axis=1), axis=0)[:, 0]
    docs_per_fam = np.bincount(fam_doc, minlength=family.max() + 1)
    return n / (per_doc[doc] * docs_per_fam[family] * len(np.unique(family)))


def prior(train, query, human, total, types, families, subtract):
    tt, kk = types[train], families[train]
    yy, nn = human[train].sum(1), total[train].sum(1)
    lookup = {}
    for grp, weight in (("y", yy), ("n", nn)):
        type_sum = np.bincount(tt, weights=weight, minlength=types.max() + 1)
        type_sum = type_sum[types[query]]
        pop = np.full(len(query), weight.sum(), float)
        if subtract:
            family_sum = np.bincount(kk, weights=weight, minlength=families.max() + 1)
            pop -= family_sum[families[query]]
            stride = int(types.max()) + 1
            train_code = kk.astype(np.int64) * stride + tt
            unique, inverse = np.unique(train_code, return_inverse=True)
            both = np.bincount(inverse, weights=weight)
            code = families[query].astype(np.int64) * stride + types[query]
            pos = np.searchsorted(unique, code)
            hit = (pos < len(unique)) & (
                unique[np.minimum(pos, len(unique) - 1)] == code
            )
            type_sum[hit] -= both[pos[hit]]
        lookup[grp] = (type_sum, pop)
    (y, py), (n, pn) = lookup["y"], lookup["n"]
    if np.min(n) < 0 or np.any(y > n) or np.min(pn) <= 0:
        raise ArithmeticError("Invalid disjoint word prior")
    return np.column_stack(
        [logit(np.clip((y + 20 * py / pn) / (n + 20), 1e-7, 1 - 1e-7)), np.log1p(n)]
    )


def all_features(base, pairs, rates, arm, aware):
    words, levels = pairs.T
    form = base["form"][words]
    columns = [form, rates]
    if aware:
        l = levels - 2.5
        columns.extend(
            [
                (levels[:, None] > np.arange(5)[None, :]),
                l[:, None] * np.column_stack([form[:, 2], form[:, 6], rates[:, 0]]),
            ]
        )
    if arm != "lexical":
        src = base["source"][words]
        columns.append(src)
        if aware:
            columns.append((levels - 2.5)[:, None] * src[:, :1])
    if arm == "contextual":
        columns.append(base["context"][words])
    return np.column_stack(columns)


def make_rows(ix, lab, total, doc, family, aware):
    if aware:
        sub, level = np.nonzero(total[ix])
        words = ix[sub]
        pairs = np.column_stack([words, level])
        obs = lab[words, level] / total[words, level]
        dist = total[words, level] / total[words].sum(1)
    else:
        words = ix
        pairs = np.column_stack([words, np.zeros(len(ix), int)])
        obs = lab[ix].sum(1) / total[ix].sum(1)
        dist = np.ones(len(words))
        sub = np.arange(len(ix))
    dw = family_weights(np.ones(len(ix)), doc[ix], family[ix])
    w = dw[sub] * dist
    if not np.isclose(w.sum(), 1):
        raise ArithmeticError("Incorrect material normalization")
    return pairs, obs, w, sub


def ridge(x, obs, w):
    mean = w @ x
    x -= mean
    sd = np.sqrt(np.einsum("i,ij,ij->j", w, x, x))
    sd = np.where(sd > 1e-7, sd, 1.0)
    x /= sd
    z = x

    def f(par):
        s = z @ par[:-1] + par[-1]
        delta = w * (expit(s) - obs)
        return float(
            w @ (np.logaddexp(0, s) - obs * s) + RIDGE * (par[:-1] @ par[:-1]) / 2
        ), np.r_[z.T @ delta + RIDGE * par[:-1], delta.sum()]

    fitted = minimize(
        f,
        np.r_[np.zeros(z.shape[1]), logit(w @ obs)],
        jac=True,
        method="L-BFGS-B",
        options={"ftol": 2e-12, "gtol": 1e-7, "maxiter": 200, "maxls": 35},
    )
    g = np.max(abs(fitted.jac))
    if not fitted.success or g > 5e-6:
        raise ArithmeticError(f"Unconverged NNS fit {g}: {fitted.message}")
    return (mean, sd, fitted.x), float(g)


def solve_mix(a, b, y, w):
    diff = b - a

    def der(alpha):
        p = np.clip(a + alpha * diff, 1e-9, 1 - 1e-9)
        return float(w @ ((p - y) * diff / (p * (1 - p))))

    if der(0) >= 0:
        return 0.0
    if der(1) <= 0:
        return 1.0
    return float(brentq(der, 0, 1, xtol=1e-11))


def run(sparse_extraction, dense_extraction):
    start = perf_counter()
    for role, path in (("sparse", sparse_extraction), ("dense", dense_extraction)):
        build(role, path)
    a, b = [pack_model_input(k) for k in ("sparse", "dense")]
    h, n, coverage = selected_counts("sparse", a)
    n_sparse = len(n)
    sizes = {k: len(x["doc"]) for k, x in (("sparse", a), ("dense", b))}
    base = {k: np.concatenate([a[k], b[k]]) for k in ("form", "source", "context")}

    def coded(k):
        return np.unique(np.r_[a[k], b[k]], return_inverse=True)

    type_names, types = coded("stem")
    families, fam = coded("family")
    _, doc = coded("doc")
    train = np.flatnonzero(n.sum(1) > 0)
    held = np.arange(n_sparse, n_sparse + len(b["doc"]))
    ordering = sorted(
        set(fam[train]),
        key=lambda i: sha256(("nns/sparse/v1/" + families[i]).encode()).digest(),
    )
    folds = {j: i % 5 for i, j in enumerate(ordering)}
    fold = np.array([folds[q] for q in fam[train]])
    result = {k: b[k] for k in ("doc", "family", "coords", "type", "form", "source")}
    result["seen_stem"] = np.isin(types[held], np.unique(types[train]))
    fitted_parameters = {}
    diag = {}
    inferr = {}
    for aware in (False, True):
        key = "cefr" if aware else "text"
        allrows, gold, wt, _ = make_rows(train, h, n, doc, fam, aware)
        sparse_forecast = {arm: np.full(len(gold), np.nan) for arm in ARMS}
        out = {}
        grad = 0
        for f in range(6):
            whole = f == 5
            it = train if whole else train[fold != f]
            iv = held if whole else train[fold == f]
            tr, obs, w, reps = make_rows(it, h, n, doc, fam, aware)
            p = prior(it, it, h, n, types, fam, True)[reps]
            if whole:
                if aware:
                    rix = np.repeat(np.arange(len(iv)), 6)
                    va = np.column_stack((iv[rix], np.tile(np.arange(6), len(iv))))
                else:
                    rix = np.arange(len(iv))
                    va = np.column_stack((iv, np.zeros(len(iv), int)))
            else:
                mask = np.isin(allrows[:, 0], iv)
                va = allrows[mask]
                rix = np.searchsorted(iv, va[:, 0])
            pv = prior(it, iv, h, n, types, fam, False)[rix]
            for arm in ARMS:
                (m, s, theta), g = ridge(all_features(base, tr, p, arm, aware), obs, w)
                yhat = expit(
                    (all_features(base, va, pv, arm, aware) - m) / s @ theta[:-1]
                    + theta[-1]
                )
                grad = max(grad, g)
                if whole:
                    out[arm] = yhat.reshape(len(iv), 6 if aware else 1)
                    for name, array in (("mean", m), ("scale", s), ("theta", theta)):
                        fitted_parameters[key + "." + arm + "." + name] = array
                else:
                    sparse_forecast[arm][mask] = yhat
        if not all(np.isfinite(v).all() for v in sparse_forecast.values()):
            raise ArithmeticError("Missing source-family forecasts")
        alpha = solve_mix(
            sparse_forecast["lexical"], sparse_forecast["contextual"], gold, wt
        )
        out["adaptive"] = (1 - alpha) * out["lexical"] + alpha * out["contextual"]
        for name, yhat in out.items():
            if not (
                np.isfinite(yhat).all() and np.min(yhat) >= 0 and np.max(yhat) <= 1
            ):
                raise ArithmeticError("Invalid forecast")
            result[key + "." + name] = yhat
        diag[key] = {
            "alpha": alpha,
            "max_abs_gradient": grad,
            "crossfit_logloss": {
                arm: float(
                    -wt
                    @ (
                        gold * np.log(np.clip(z, 1e-12, 1))
                        + (1 - gold) * np.log(np.clip(1 - z, 1e-12, 1))
                    )
                )
                for arm, z in sparse_forecast.items()
            },
            "fits": 6 * len(ARMS),
            "training_rows": len(gold),
        }
        inferr[key] = alpha
    # Store only public word geometry, aggregate training priors and fitted coefficients.
    truestem = np.bincount(
        types[train], weights=h[train].sum(1), minlength=len(type_names)
    )
    trials = np.bincount(
        types[train], weights=n[train].sum(1), minlength=len(type_names)
    )
    used = trials > 0
    np.savez_compressed(
        OUT / "weights.npz",
        **fitted_parameters,
        stem=type_names[used],
        stem_flags=truestem[used],
        stem_answers=trials[used],
        **{k + ".mixture": v for k, v in inferr.items()},
    )
    np.savez_compressed(OUT / "word-forecasts.npz", **result)
    seal = {
        "code_sha256": digest(Path(__file__)),
        "dense_labels_joined": False,
        "material_sha256": digest(OUT / "material-audit.json"),
        "protocol_sha256": digest(OUT / "protocol-lock.json"),
        "source": coverage,
        "certified_words": sizes,
        "fits": diag,
        "feature_code_sha256": digest(Path(__file__).with_name("nns_features.py")),
        "outcome_free_forecast_sha256": digest(OUT / "word-forecasts.npz"),
        "weights_sha256": digest(OUT / "weights.npz"),
        "source_flags": int(h.sum()),
        "source_role": "sparse",
        "source_words": len(train),
        "extractions": {
            role: json.loads((p / "extraction.json").read_bytes())
            for role, p in (("dense", dense_extraction), ("sparse", sparse_extraction))
        },
        "runtime_seconds": perf_counter() - start,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / (2**20 if sys.platform == "darwin" else 1024),
    }
    (OUT / "forecast-seal.json").write_text(
        json.dumps(seal, indent=2, allow_nan=False) + "\n"
    )
    print(diag, coverage)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sparse", type=Path, required=True)
    parser.add_argument("--dense", type=Path, required=True)
    args = parser.parse_args()
    run(args.sparse, args.dense)
