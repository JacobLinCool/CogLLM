"""Transfer a source-balanced human-response curvature into held-reader, held-summary likelihoods."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, log_expit, logit

from research.reading_validation.joint_data import surface
from research.reading_validation.local_forecast import digest, rows
from research.reading_validation.noun_material import MODEL
from research.reading_validation.relay_summary_fit import (
    joined,
    loss,
    observations,
    splits,
    weights,
)
from research.reading_validation.relay_summary_material import OUT, PRIVATE, STIM
from research.reading_validation.relay_summary_score import Crossed, interval
from research.reading_validation.relay_summary_source import coordinates

RIDGE = 0.1
ARMS = (
    "independent",
    "geometry_only",
    "isotropic_centered",
    "transferred",
    "adaptive",
    "prior_direction",
)


def predict(x, y, r, train, test, center, precision, binomial):
    tr = x[train]
    weight = weights(r[train])
    zbar = weight @ tr
    z = tr - zbar
    hold = x[test] - zbar
    width = z.shape[1]
    n = len(tr)
    yy = y[train]
    if binomial:
        observed = np.r_[yy / 4, 0.5]
        w = np.r_[weight * 4 * n, 1]
        w /= w.sum()
        z = np.vstack([z, np.zeros((1, width))])
        initial = np.r_[center, logit(w @ observed)]

        def nll(v):
            t = z @ v[:width] + v[-1]
            d = w * (expit(t) - observed)
            return float(w @ (np.logaddexp(0, t) - observed * t)), np.r_[
                z.T @ d, d.sum()
            ]

        bounds = None
    else:
        observed = np.r_[yy, np.arange(1, 6)].astype(int)
        w = np.r_[weight * n, np.full(5, 0.5)]
        w /= w.sum()
        z = np.vstack([z, np.zeros((5, width))])
        init = logit(np.cumsum(np.bincount(observed, weights=w, minlength=6)[1:])[:-1])
        initial = np.r_[center, init[0], np.diff(init)]
        bounds = [(None, None)] * (width + 1) + [(0.001, None)] * 3

        def nll(v):
            knots = np.r_[-np.inf, np.cumsum(v[width:]), np.inf]
            s = z @ v[:width]
            low, hi = knots[observed - 1], knots[observed]
            widths = hi - low
            denom = -np.expm1(-widths)
            mass = log_expit(hi - s) + log_expit(s - low) + np.log(denom)
            flo, fhi = expit(low - s), expit(hi - s)
            dz = w * (1 - fhi - flo)
            reciprocal = np.exp(-widths) / denom
            base = np.zeros(6)
            np.add.at(base, observed, -w * (1 - fhi + reciprocal))
            np.add.at(base, observed - 1, w * (flo + reciprocal))
            return -float(w @ mass), np.r_[z.T @ dz, np.cumsum(base[1:-1][::-1])[::-1]]

    def objective(v):
        val, grad = nll(v)
        d = v[:width] - center
        p = RIDGE * (precision @ d)
        grad[:width] += p
        return val + float(d @ p) / 2, grad

    sol = minimize(
        objective,
        initial,
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 350, "maxls": 45, "gtol": 1e-8, "ftol": 1e-13},
    )
    error = float(np.max(abs(sol.jac)))
    if not sol.success or error > 2e-5:
        raise ArithmeticError(
            f"Joint posterior did not converge {sol.message}, {error}"
        )
    v = sol.x
    score = hold @ v[:width]
    return (
        expit(score + v[-1])[:, None]
        if binomial
        else expit(score[:, None] - np.cumsum(v[width:])[None, :])
    ), error


def run(extraction):
    start = perf_counter()
    predseal = json.loads((OUT / "forecast-seal.json").read_bytes())
    if (
        digest(extraction / "document.npy")
        != predseal["remote"]["files_sha256"]["document.npy"]
    ):
        raise ValueError("Native public summary states changed")
    prior_receipt = json.loads((OUT / "collective-source.json").read_bytes())
    if digest(OUT / "collective-prior.npz") != prior_receipt["source_prior_sha256"]:
        raise ValueError("Human source Hessian changed")
    with np.load(OUT / "collective-prior.npz", allow_pickle=False) as f:
        mu, precision = f["center"], f["precision"]
    geometry = json.loads((extraction / "geometry.json").read_bytes())
    texts = {q["id"]: q["text"] for q in rows(STIM)}
    ids = {q["id"].removeprefix("relay-summary/"): i for i, q in enumerate(geometry)}
    with np.load(MODEL, allow_pickle=False) as f:
        feature = coordinates(
            np.asarray([surface(texts[q["id"]], "en") for q in geometry]),
            np.load(extraction / "document.npy", mmap_mode="r"),
            f,
        )
    rec = joined(False)
    r = np.asarray([q["reader"] for q in rec])
    xx = feature[[ids[q["article"]] for q in rec]]
    original = np.asarray([-q["scores"]["zero_qwen"] for q in rec])[:, None]
    ident = np.eye(len(mu))
    zero = np.zeros_like(mu)
    design = {
        "independent": (xx, zero, ident),
        "geometry_only": (xx, zero, precision),
        "isotropic_centered": (xx, mu, ident),
        "transferred": (xx, mu, precision),
        "prior_direction": (original, np.zeros(1), np.eye(1)),
    }
    nfits = 0
    worst = 0
    receipt = {}
    out = {}
    cross = Crossed(rec)
    for end, binomial in (("simplicity", False), ("correct", True)):
        y = np.asarray([4 - q["correct"] if binomial else 6 - q[end] for q in rec])
        d = 1 if binomial else 4
        forecasts = {arm: np.full((len(y), d), np.nan) for arm in ARMS}
        alphas = []
        for train, test in splits(rec, 5, "relay-outer-v1"):
            idx = np.flatnonzero(train)
            sample = [rec[i] for i in idx]
            calib = {
                k: np.full((len(idx), d), np.nan)
                for k in ("geometry_only", "transferred")
            }
            for arm, (x, center, p) in design.items():
                z, err = predict(x, y, r, train, test, center, p, binomial)
                forecasts[arm][test] = z
                nfits += 1
                worst = max(worst, err)
                if arm in calib:
                    for t, v in splits(sample, 2, "relay-inner-v1"):
                        z, err = predict(
                            x[idx], y[idx], r[idx], t, v, center, p, binomial
                        )
                        calib[arm][v] = z
                        nfits += 1
                        worst = max(worst, err)
            w = weights(r[idx])
            u = calib["geometry_only"]
            delta = calib["transferred"] - u
            if not np.isfinite(delta).all():
                raise ArithmeticError("Missing prior calibration")
            den = w @ (delta**2).mean(1)
            a = (
                float(
                    np.clip(
                        (w @ ((observations(y[idx], binomial) - u) * delta).mean(1))
                        / den,
                        0,
                        1,
                    )
                )
                if den > 1e-14
                else 0.0
            )
            alphas.append(a)
            forecasts["adaptive"][test] = forecasts["geometry_only"][test] + a * (
                forecasts["transferred"][test] - forecasts["geometry_only"][test]
            )
        if not all(np.isfinite(s).all() for s in forecasts.values()):
            raise ValueError("Omitted held summary")
        out.update({end + "." + k: v for k, v in forecasts.items()})
        ls = np.column_stack([loss(forecasts[k], y, binomial) for k in ARMS])
        p, b = cross.visit(ls)
        effects = {}
        for a, bas in (
            ("adaptive", "independent"),
            ("adaptive", "geometry_only"),
            ("transferred", "independent"),
            ("transferred", "geometry_only"),
            ("transferred", "isotropic_centered"),
            ("geometry_only", "independent"),
            ("isotropic_centered", "independent"),
            ("adaptive", "prior_direction"),
            ("prior_direction", "independent"),
        ):
            i, j = ARMS.index(a), ARMS.index(bas)
            effects[a + "_over_" + bas] = interval(p[j] - p[i], b[:, j] - b[:, i])
        receipt[end] = {
            "scores": {k: interval(p[i], b[:, i]) for i, k in enumerate(ARMS)},
            "paired_improvement": effects,
            "mixture_weight": {
                "median": float(np.median(alphas)),
                "min": min(alphas),
                "max": max(alphas),
                "positive_folds": sum(t > 0 for t in alphas),
                "folds": len(alphas),
            },
        }
    p = PRIVATE / "joint-forecasts.npz"
    np.savez_compressed(p, **out)
    effects = receipt["simplicity"]["paired_improvement"]
    payload = {
        "endpoints": receipt,
        "outcome_supervision_gate_passed": all(
            effects["adaptive_over_" + z]["ci95"][0] > 0
            for z in ("independent", "geometry_only")
        ),
        "human_source_sha256": digest(OUT / "collective-source.json"),
        "private_prediction_sha256": digest(p),
        "max_likelihood_gradient": worst,
        "fit_count": nfits,
        "runtime_seconds": perf_counter() - start,
        "script_sha256": digest(Path(__file__)),
        "immutable_plan_sha256": digest(OUT / "joint-plan.json"),
    }
    (OUT / "collective-scores.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n"
    )
    print(
        payload["outcome_supervision_gate_passed"],
        effects["adaptive_over_independent"],
        effects["adaptive_over_geometry_only"],
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--extraction", required=True, type=Path)
    run(p.parse_args().extraction)
