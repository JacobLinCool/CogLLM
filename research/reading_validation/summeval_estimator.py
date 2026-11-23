"""Joint or independent fit of the observed expert passage mean with source boundary and passage likelihoods."""

import numpy as np
from scipy.optimize import minimize
from scipy.special import logit

from research.reading_validation.coherence_order_estimator import (
    LOG_SCALE_RIDGE,
    RIDGE,
    terms,
)
from research.reading_validation.italian_contrast_estimator import make_objective


def third_task(source, initial, news):
    old = make_objective(source, None, 0.0)
    y = news["y"]
    w = news["w"]
    avg = w @ y
    null = w @ ((y - avg) ** 2)
    if null <= 0:
        raise ValueError("The calibration group has no coherence variation")
    original = initial.copy()
    p0 = np.r_[original, logit(np.clip((avg - 1) / 4, 0.001, 0.999)), 0.0]

    def objective(p):
        prior = p[:-2]
        beta = prior[:-3]
        b_news, g_news = p[-2:]
        g_it = prior[-1]
        s1 = np.exp(g_news)
        v, g = old(prior)
        regular = 0.5 * (RIDGE * (beta @ beta) + LOG_SCALE_RIDGE * g_it * g_it)
        pen_grad = np.r_[RIDGE * beta, 0.0, 0.0, LOG_SCALE_RIDGE * g_it]
        b = np.r_[beta, 0.0, b_news, g_news]
        estimated, (pool, coeff, frac) = terms(news["x"] @ beta, b, news)
        err = estimated - y
        nw = w / (3 * null)
        wave = 2 * nw * err * 4 * frac * (1 - frac)
        loss = (
            2 * v / 3
            + regular / 3
            + nw @ (err**2)
            + 0.5 * LOG_SCALE_RIDGE * g_news * g_news
        )
        g = np.r_[
            2 * g / 3 + pen_grad / 3,
            wave.sum(),
            -s1 * (wave @ pool) + LOG_SCALE_RIDGE * g_news,
        ]
        g[: len(beta)] -= s1 * (wave[news["owner"]] * coeff) @ news["x"]
        return float(loss), g

    rng = np.random.default_rng(178539)
    probe = p0 + 0.05 * rng.normal(size=len(p0))
    an = objective(probe)[1]
    errors = []
    for vec in rng.normal(size=(4, len(p0))):
        vec /= np.linalg.norm(vec)
        num = (objective(probe + 1e-6 * vec)[0] - objective(probe - 1e-6 * vec)[0]) / (
            2e-6
        )
        errors.append(float(abs(num - an @ vec)))
    if max(errors) > 2e-6:
        raise ArithmeticError(
            "The third response likelihood failed a finite difference check"
        )
    bounds = [(None, None)] * len(p0)
    bounds[-1] = bounds[-3] = (-np.log(4), np.log(4))
    f = minimize(
        objective,
        p0,
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        options={
            "maxiter": 800,
            "maxls": 64,
            "maxcor": 22,
            "ftol": 1e-14,
            "gtol": 1e-8,
        },
    )
    grad = objective(f.x)[1].copy()
    for j in (-3, -1):
        if (f.x[j] <= -np.log(4) + 1e-9 and grad[j] > 0) or (
            f.x[j] >= np.log(4) - 1e-9 and grad[j] < 0
        ):
            grad[j] = 0
    projected = float(abs(grad).max())
    if not f.success or projected > 5e-6:
        raise ArithmeticError(
            f"Joint observations failed to converge, gradient {projected}"
        )
    return np.r_[f.x[:-5], 0.0, f.x[-2:]], {
        "start": objective(p0)[0],
        "final": float(f.fun),
        "max_analytic_fd": max(errors),
        "projected_gradient": projected,
        "iterations": int(f.nit),
        "new_article_scale": float(np.exp(f.x[-1])),
        "source_italian_scale": float(np.exp(f.x[-3])),
        "source_null": float(null),
        "full_parameters": f.x,
    }
