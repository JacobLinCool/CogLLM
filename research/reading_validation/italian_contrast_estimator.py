"""Joint human likelihood with one fixed, source-only sentence restoration loss."""

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from research.reading_validation.coherence_order_estimator import (
    LOG_SCALE_RIDGE,
    RIDGE,
    equal_group_weights,
    point_loss,
    pool,
    terms,
)

LAMBDA = 0.25


def compression(rows):
    owners = np.concatenate([np.full(len(x), i, int) for i, x in enumerate(rows)])
    return np.concatenate(rows), owners, np.array(list(map(len, rows)))


def layout(pairs):
    usable = [p for p in pairs if p["local"] is not None]
    if not pairs or not usable:
        raise ValueError("Unable to fit both registered source constraints")
    return {
        "canonical": compression([p["canonical"] for p in pairs]),
        "swapped": compression([p["permuted"] for p in pairs]),
        "w": equal_group_weights([p["group"] for p in pairs]),
        "local": np.array([p["local"] for p in usable]),
        "wu": equal_group_weights([p["group"] for p in usable]),
        "pairs": len(pairs),
        "localized": len(usable),
        "groups": len({p["group"] for p in pairs}),
        "localized_groups": len({p["group"] for p in usable}),
    }


def auxiliary(beta, source):
    def global_values(group):
        x, owners, counts = source[group]
        values, jac = pool(x @ beta, owners, counts)
        return values, jac, owners, x

    canon, ca, oi, xc = global_values("canonical")
    sw, sa, si, xs = global_values("swapped")
    d = sw - canon
    w = source["w"]
    observed = source["local"]
    wu = source["wu"]
    delta = observed @ beta
    normalizer = 2 * np.log(2)
    loss = (w @ np.logaddexp(0, -d) + wu @ np.logaddexp(0, -delta)) / normalizer
    du = -w * expit(-d)
    grad = (du[si] * sa) @ xs - (du[oi] * ca) @ xc - (wu * expit(-delta)) @ observed
    return loss, grad / normalizer


def make_objective(blocks, structural, aux_weight):
    norm = {
        b["language"]: b["w"] @ point_loss(b, np.full(len(b["y"]), b["w"] @ b["y"]))
        for b in blocks
    }

    def objective(p):
        beta = p[:-3]
        scale = np.exp(p[-1])
        loss = 0.5 * RIDGE * (beta @ beta) + 0.5 * LOG_SCALE_RIDGE * p[-1] ** 2
        grad = np.r_[RIDGE * beta, 0.0, 0.0, LOG_SCALE_RIDGE * p[-1]]
        for obs in blocks:
            x = obs["x"]
            w = obs["w"] / (len(blocks) * norm[obs["language"]])
            y = obs["y"]
            z = x @ beta
            if obs["language"] == "english":
                logits = p[-3] + z
                werr = w * (expit(logits) - y)
                loss += w @ (np.logaddexp(0, logits) - y * logits)
                grad[:-3] += werr @ x
                grad[-3] += werr.sum()
            else:
                pred, (val, jac, r) = terms(z, p, obs)
                e = pred - y
                loss += w @ (e**2)
                t = 2 * w * e * 4 * r * (1 - r)
                grad[-2] += t.sum()
                grad[-1] -= scale * (t @ val)
                grad[:-3] -= scale * (t[obs["owner"]] * jac) @ x
        if aux_weight:
            loss_pair, g = auxiliary(beta, structural)
            loss += aux_weight * loss_pair
            grad[:-3] += aux_weight * g
        return float(loss), grad

    return objective


def projected(p, gradient):
    g = gradient.copy()
    bound = np.log(4)
    if (p[-1] <= -bound + 1e-9 and g[-1] > 0) or (p[-1] >= bound - 1e-9 and g[-1] < 0):
        g[-1] = 0.0
    return float(np.max(abs(g)))


def fit(blocks, structure, original, expected):
    base = make_objective(blocks, structure, 0.0)
    bval, bgrad = base(original)
    if (
        abs(bval - expected["objective_final"]) > 1e-9
        or projected(original, bgrad) > 5e-6
    ):
        raise ValueError(
            "The independent recreation of the original joint human objective failed"
        )
    opt = make_objective(blocks, structure, LAMBDA)
    rng = np.random.default_rng(7319)
    probe = original + 0.05 * rng.normal(size=original.shape)
    g = opt(probe)[1]
    errors = []
    for v in rng.normal(size=(4, len(original))):
        v /= np.linalg.norm(v)
        num = (opt(probe + 1e-6 * v)[0] - opt(probe - 1e-6 * v)[0]) / 2e-6
        errors.append(abs(num - g @ v))
    if max(errors) > 2e-6:
        raise ValueError(
            "Gradient of human and source restoration likelihood fails finite differences"
        )
    bounds = [(None, None)] * (len(original) - 1) + [(-np.log(4), np.log(4))]
    solution = minimize(
        opt,
        original,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={
            "maxiter": 650,
            "maxls": 64,
            "maxcor": 22,
            "ftol": 1e-14,
            "gtol": 1e-8,
        },
    )
    final = solution.x
    grad = projected(final, opt(final)[1])
    if not solution.success or grad > 5e-6:
        raise ValueError("Source restoration and human model has not converged")
    val = lambda q: float(auxiliary(q[:-3], structure)[0])
    return final, {
        "global_and_local_lambda": LAMBDA,
        "delta_old_human_objective": bval - expected["objective_final"],
        "old_projected_grad": projected(original, bgrad),
        "starting_objective": opt(original)[0],
        "final_objective": solution.fun,
        "initial_auxiliary": val(original),
        "final_auxiliary": val(final),
        "initial_human_plus_ridge": bval,
        "final_human_plus_ridge": base(final)[0],
        "projected_gradient": grad,
        "finite_difference_max": float(max(errors)),
        "steps": solution.nit,
    }
