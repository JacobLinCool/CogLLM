"""Fit a normalized population lexicon to genuine grouped human word choices."""
from dataclasses import dataclass
from hashlib import sha256
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

BATCH = 64


@dataclass
class Choices:
    """Sparse observed reader responses in a common lexicon, one list per visible prefix."""

    indices: list
    counts: list

    def subset(self, ix):
        return Choices([self.indices[i] for i in ix], [self.counts[i] for i in ix])

    def total(self):
        return np.array([sum(c) for c in self.counts], float)


def source_basis(states, fit):
    x = states[fit].astype(float)
    mean = x.mean(0)
    sd = np.maximum(x.std(0), 0.02)
    a = (x - mean) / sd
    e, v = np.linalg.eigh(a.T @ a / len(a))
    e = e[-16:]
    V = v[:, -16:] / np.sqrt(e)
    if min(e) < 0.0001:
        raise ValueError("Source words fail to identify sixteen contextual directions")
    return dict(mean=mean.tolist(), scale=sd.tolist(), vectors=V.tolist())


def project(states, model):
    return ((states - np.array(model["mean"])) / np.array(model["scale"])) @ np.array(
        model["vectors"]
    )


def packed_words(coords, frequencies):
    f = np.log(frequencies)
    f = (f - f.mean()) / f.std()
    w = np.pad(coords.astype(float), ((0, 1), (0, 0)))
    u = np.zeros((len(w), 18))
    u[:-1, 0] = f
    u[-1, 1] = 1.0
    u[:, 2:] = w
    return u, w


def unpack(coeff, c):
    return coeff[0], coeff[1:19], coeff[19:].reshape(c.shape[1], 16)


def logits(raw, ctx, U, V, co):
    a, b, A = unpack(co, ctx)
    return a * raw + U @ b + (ctx @ A) @ V.T


def loss_grad(co, raw, ctx, counts, U, V, penalty=1.0):
    a, b, A = unpack(co, ctx)
    actual = counts.total()
    nt = actual.sum()
    val = 0.0
    da = 0.0
    db = np.zeros(len(b))
    dA = np.zeros_like(A)
    base = U @ b
    for lo in range(0, len(raw), BATCH):
        hi = min(lo + BATCH, len(raw))
        h = ctx[lo:hi]
        r = raw[lo:hi]
        z = a * r + base + (h @ A) @ V.T
        l = logsumexp(z, axis=1)
        num = actual[lo:hi]
        val += num @ l
        err = np.exp(z - l[:, None]) * num[:, None]
        for j, (cols, hits) in enumerate(
            zip(counts.indices[lo:hi], counts.counts[lo:hi])
        ):
            val -= z[j, cols] @ hits
            err[j, cols] -= hits
        da += np.sum(err * r)
        db += err.sum(0) @ U
        if dA.size:
            dA += h.T @ (err @ V)
    bias = co.copy()
    bias[0] -= 1.0
    val += 0.5 * penalty * (bias @ bias)
    grad = np.r_[da, db, dA.ravel()] + penalty * bias
    return val / nt, grad / nt


def fit(raw, ctx, counts, U, V):
    width = 19 + ctx.shape[1] * 16
    start = np.zeros(width)
    start[0] = 1.0
    fun = lambda w: loss_grad(w, raw, ctx, counts, U, V)
    z = minimize(
        fun,
        start,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 240, "ftol": 1e-13, "gtol": 3e-7, "maxcor": 16},
    )
    val, grad = fun(z.x)
    station = float(abs(grad).max())
    if not np.isfinite(val) or station > 1e-5:
        raise ArithmeticError(
            f"Full-source multinomial score did not converge: {station:.5g} after {z.nit} iterations"
        )
    return dict(
        coefficients=z.x.tolist(),
        logloss=float(val),
        stationarity=station,
        iterations=int(z.nit),
        choices=int(counts.total().sum()),
        prefixes=len(raw),
    )


def material(cols, logs, route, share):
    p = np.empty((len(cols), len(route) + 1), np.float32)
    weights = share.astype("float32")
    for lo in range(0, len(cols), BATCH):
        at = cols[lo : lo + BATCH]
        row = p[lo : lo + len(at)]
        row[:, :-1] = logs[at, :-1][:, route] + weights
        row[:, -1] = logs[at, -1]
        row -= logsumexp(row, axis=1)[:, None]
    return p


def distribution(raw, ctx, model, U, V):
    if model["kind"] == "contextless":
        return np.broadcast_to(np.asarray(model["logp"]), raw.shape)
    if model["kind"] == "raw":
        return raw.astype(float)
    x = ctx if model["kind"] == "context" else ctx[:, :0]
    z = logits(raw, x, U, V, np.array(model["coefficients"]))
    return z - logsumexp(z, axis=1)[:, None]


def batches(ix, raw, states, projection, models, U, V):
    for lo in range(0, len(ix), BATCH):
        index = ix[lo : lo + BATCH]
        c = project(states[index], projection)
        p = raw[index]
        yield index, {k: distribution(p, c, m, U, V) for k, m in models.items()}


def predictions(ix, raw, states, projection, models, U, V, observed=None, targets=None):
    # Every probability enters the byte seal before target-study answer tables are read.
    dig = {k: sha256() for k in models}
    scores = {k: [] for k in models}
    for ids, pred in batches(ix, raw, states, projection, models, U, V):
        for k, lp in pred.items():
            lp = np.asarray(lp, dtype="float32", order="C")
            dig[k].update(lp.tobytes())
            if observed is None:
                continue
            for j, i in enumerate(ids):
                c = observed.counts[i]
                v = observed.indices[i]
                n = int(c.sum())
                p = np.exp(lp[j].astype(float))
                g = (c * (c - 1)).sum() / (n * (n - 1)) if n > 1 else np.nan
                winner = int(np.argmax(p))
                tops = v[c == c.max()]
                scores[k].append(
                    [
                        int(i),
                        n,
                        float(-(c @ lp[j, v])),
                        float(np.sum(p * p)),
                        float(g),
                        float(winner in tops),
                        winner,
                        int(v[np.argmax(c)]),
                        float(p[targets[i]]) if targets[i] >= 0 else -1.0,
                        float(c[v == targets[i]].sum() / n)
                        if targets[i] >= 0
                        else -1.0,
                        float(p[-1]),
                    ]
                )
    return {k: v.hexdigest() for k, v in dig.items()}, scores
