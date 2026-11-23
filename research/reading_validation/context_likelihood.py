"""Conditional likelihood for changes in measured human effort at the same printed word."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln, logsumexp

PRECISION = 1 / 0.35**2


def equal_type_weights(types):
    groups, ix, counts = np.unique(types, return_inverse=True, return_counts=True)
    return 1 / counts[ix]


def logchoose(n, k):
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)


def conditional_counts(n1, n2, k1, k2):
    total = k1 + k2
    low = np.maximum(0, total - n2)
    high = np.minimum(total, n1)
    width = int(np.max(high - low) + 1)
    values = low[:, None] + np.arange(width)
    valid = values <= high[:, None]
    a = np.minimum(values, high[:, None])
    b = total[:, None] - a
    base = np.where(
        valid, logchoose(n1[:, None], a) + logchoose(n2[:, None], b), -np.inf
    )
    return (
        values - k1[:, None],
        base - logchoose(n1, k1)[:, None] - logchoose(n2, k2)[:, None],
    )


def count_slope(d, types, n1, n2, k1, k2):
    used = (k1 + k2 > 0) & (k1 + k2 < n1 + n2)
    d = d[used]
    n1, n2, k1, k2 = [arr[used].astype(int) for arr in (n1, n2, k1, k2)]
    weights = equal_type_weights(types[used])
    support, base = conditional_counts(n1, n2, k1, k2)

    def objective(b):
        shift = (d @ b)[:, None] * support
        z = base + shift
        ln = logsumexp(z, axis=1)
        mean = np.sum(np.exp(z - ln[:, None]) * support, axis=1)
        return (
            weights @ ln + 0.5 * PRECISION * (b @ b),
            d.T @ (weights * mean) + PRECISION * b,
        )

    opt = minimize(
        objective,
        np.zeros(d.shape[1]),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 400, "gtol": 3e-7, "ftol": 1e-15, "maxcor": 20},
    )
    val, g = objective(opt.x)
    error = float(abs(g).max())
    if not np.isfinite(val) or error > 5e-5:
        raise ArithmeticError(
            f"Conditional count likelihood failed stationarity: {error}"
        )
    return opt.x, dict(
        objective=float(val),
        gradient=error,
        iterations=int(opt.nit),
        pairs=len(d),
        word_types=len(set(types[used])),
        support_max=int(support.shape[1]),
    )


def fit(x, y, n, sampling, pair, types, scale):
    a, b = pair
    delta = x[a, 1:] - x[b, 1:]
    wp = equal_type_weights(types[a])
    B = np.zeros((delta.shape[1], 3))
    info = {}
    for j in (0, 2):
        d = (y[a, j] - y[b, j]) / scale[j]
        w = wp / (2 + (sampling[a] + sampling[b]) / scale[j] ** 2) if j == 2 else wp / 2
        A = (delta * w[:, None]).T @ delta + PRECISION * np.eye(delta.shape[1])
        cross = delta.T @ (w * d)
        z = np.linalg.solve(A, cross)
        err = float(np.max(abs(A @ z - cross)))
        if err > 1e-7:
            raise ArithmeticError(f"Conditional Gaussian solve failed: {err}")
        B[:, j] = z * scale[j]
        info[str(j)] = dict(gradient=err, pairs=len(a), types=len(set(types[a])))
    B[:, 1], info["1"] = count_slope(delta, types[a], n[a], n[b], y[a, 1], y[b, 1])
    return dict(slope=B.tolist(), intercept=[0, 0, 0], diagnostics=info)
