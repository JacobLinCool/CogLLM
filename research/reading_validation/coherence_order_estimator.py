"""Fit and calibrate shared causal boundary slopes using two human observation scales."""

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, logit, xlogy

RIDGE = 0.01
LOG_SCALE_RIDGE = 0.1


def equal_group_weights(groups, freq=None):
    freq = np.ones(len(groups)) if freq is None else np.asarray(freq, float)
    levels, inverse = np.unique(groups, return_inverse=True)
    denom = np.bincount(inverse, weights=freq)
    if np.any(denom <= 0):
        raise ValueError("Training includes a group without observations")
    return freq / denom[inverse] / len(levels)


def pool(z, owners, counts):
    mx = np.full(len(counts), -np.inf)
    np.maximum.at(mx, owners, z)
    val = np.exp(z - mx[owners])
    den = np.bincount(owners, weights=val, minlength=len(counts))
    return mx + np.log(den / counts), val / den[owners]


def prepare(ids, x, language, target=None):
    if len(ids) != len(x) or not len(ids):
        raise ValueError("Missing inference material")
    if language == "english":
        use = (
            np.ones(len(x), bool)
            if target is None
            else np.array([target[k][1] > 0 for k in ids[:, 0]])
        )
        ids, x = ids[use], x[use]
        ids_out = ids[:, 0]
        data = {
            "x": x,
            "keys": ids_out,
            "language": language,
            "group": ids[:, 2],
            "doc": ids[:, 1],
        }
        if target is not None:
            y = np.array([target[k] for k in ids_out])
            data["y"] = y[:, 0] / y[:, 1]
            data["w"] = equal_group_weights(ids[:, 2], y[:, 1])
    else:
        unique, first, owner, counts = np.unique(
            ids[:, 1], return_index=True, return_inverse=True, return_counts=True
        )
        ids_out = ids[first, 5]
        data = {
            "x": x,
            "keys": ids_out,
            "language": language,
            "group": ids[first, 2],
            "doc": unique,
            "owner": owner,
            "counts": counts,
        }
        if target is not None:
            data["y"] = np.array([target[k] for k in ids_out])
            data["w"] = equal_group_weights(data["group"])
    return data


def terms(z, p, block):
    if block["language"] == "english":
        return expit(p[-3] + z), None
    values, weight = pool(z, block["owner"], block["counts"])
    fraction = expit(p[-2] - np.exp(p[-1]) * values)
    return 1 + 4 * fraction, (values, weight, fraction)


def predict(p, block):
    return terms(block["x"] @ p[:-3], p, block)[0]


def point_loss(u, prediction):
    if u["language"] == "italian":
        return (prediction - u["y"]) ** 2
    p = np.clip(prediction, 1e-12, 1 - 1e-12)
    y = u["y"]
    return -xlogy(y, p) - xlogy(1 - y, 1 - p)


def fit(blocks):
    width = blocks[0]["x"].shape[1]
    p0 = np.zeros(width + 3)
    scales = {}
    for t in blocks:
        ybar = t["w"] @ t["y"]
        is_en = t["language"] == "english"
        p0[-3 if is_en else -2] = logit(
            np.clip(ybar if is_en else (ybar - 1) / 4, 0.001, 0.999)
        )
        scales[t["language"]] = float(
            t["w"] @ point_loss(t, np.full(len(t["y"]), ybar))
        )

    def obj(p):
        b = p[:-3]
        s = np.exp(p[-1])
        value = 0.5 * RIDGE * (b @ b) + 0.5 * LOG_SCALE_RIDGE * p[-1] ** 2
        grad = np.r_[RIDGE * b, 0.0, 0.0, LOG_SCALE_RIDGE * p[-1]]
        for t in blocks:
            x = t["x"]
            y = t["y"]
            w = t["w"] / (len(blocks) * scales[t["language"]])
            z = x @ b
            if t["language"] == "english":
                q = z + p[-3]
                value += w @ (np.logaddexp(0, q) - y * q)
                a = w * (expit(q) - y)
                grad[:-3] += a @ x
                grad[-3] += a.sum()
            else:
                out, (v, weight, frac) = terms(z, p, t)
                loss = out - y
                value += w @ (loss**2)
                c = 2 * w * loss * 4 * frac * (1 - frac)
                grad[-2] += c.sum()
                grad[-1] -= s * (c @ v)
                grad[:-3] -= s * (c[t["owner"]] * weight) @ x
        return value, grad

    # Validate the gradient of both observation models at a nonconstant boundary function.
    rng = np.random.default_rng(608)
    probe = p0 + 0.08 * rng.normal(size=p0.shape)
    derivative = obj(probe)[1]
    diff = []
    for direction in rng.normal(size=(3, len(probe))):
        direction /= np.linalg.norm(direction)
        numeric = (
            obj(probe + 1e-6 * direction)[0] - obj(probe - 1e-6 * direction)[0]
        ) / 2e-6
        diff.append(abs(numeric - derivative @ direction))
    if max(diff) > 2e-6:
        raise ArithmeticError(
            "Observed-data likelihood gradient failed its finite-difference check"
        )
    bounds = [(None, None)] * (len(p0) - 1) + [(-np.log(4), np.log(4))]
    record = minimize(
        obj,
        p0,
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
    grad = obj(record.x)[1]
    if (
        record.x[-1] <= -np.log(4) + 1e-9
        and grad[-1] > 0
        or record.x[-1] >= np.log(4) - 1e-9
        and grad[-1] < 0
    ):
        grad[-1] = 0.0
    tol = float(np.max(np.abs(grad)))
    if not record.success or tol > 5e-6:
        raise ValueError(f"Readout did not converge: {record.message}; gradient {tol}")
    return record.x, {
        "objective_start": float(obj(p0)[0]),
        "objective_final": float(record.fun),
        "iterations": record.nit,
        "max_projected_gradient": tol,
        "null_loss": scales,
        "gradient_check_max_abs": max(diff),
        "italian_scale": float(np.exp(record.x[-1])),
    }


def mixture(t, independent, shared):
    def value(a):
        return t["w"] @ point_loss(t, (1 - a) * independent + a * shared)

    mid = minimize_scalar(
        value, method="bounded", bounds=(0.0, 1.0), options={"xatol": 1e-12}
    )
    if not mid.success:
        raise ValueError("Development mixing weight failed")
    return float(min((0.0, mid.x, 1.0), key=value))


def normalize(ids, x):
    out = np.empty_like(x)
    stats = {}
    for lang, fit, test in (
        ("italian", "italian", ids[:, 3] == "italian"),
        ("english", "cohesentia", ids[:, 3] != "italian"),
    ):
        mask = (ids[:, 4] == "fit") & (ids[:, 3] == fit)
        w = equal_group_weights(ids[mask, 2])
        center = w @ x[mask]
        v = np.sqrt(w @ ((x[mask] - center) ** 2))
        scale = np.where(v > 1e-6, v, 1.0)
        out[test] = (x[test] - center) / scale
        stats[lang + ".center"] = center
        stats[lang + ".scale"] = scale
    return out, stats
