"""Joint graded and pairwise human supervision with grouped out-of-fold shrinkage."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import json

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit
from research.reading_validation.joint_data import digest
from research.reading_validation.revision_data import ASPECTS, STUDIES, build
from research.reading_validation.revision_features import build as features
from research.reading_validation.revision_prepare import OUT, SOURCE

RIDGE = 0.012
SHARED = 0.006
CONFIG = {
    "edit": ("edit",),
    "aligned": ("edit", "neural"),
    "full": ("edit", "neural", "directions"),
    "independent_full": ("edit", "neural", "directions"),
}


def fractional_loss(z, y):
    return np.logaddexp(0.0, z) - y * z


def obs_arrays(task, mask):
    chosen = [r for r in task.observations if mask[r.first]]
    if any(r.second >= 0 and not mask[r.second] for r in chosen):
        raise ValueError("Training pair crosses source split")
    base = Counter()
    cells = defaultdict(set)
    for r in chosen:
        arm = r.study, r.aspect, r.second >= 0
        base[arm, r.group] += r.weight
        cells[arm].add(r.group)
    nv = Counter((a, b) for a, b, c in cells)
    scale = {arm: (0.35 if arm[0] == "past" else 1.0) / nv[arm[:2]] for arm in cells}
    weights = []
    for r in chosen:
        arm = r.study, r.aspect, r.second >= 0
        weights.append(scale[arm] * r.weight / base[arm, r.group] / len(cells[arm]))
    c = np.array([ASPECTS.index(r.aspect) for r in chosen], dtype=int)
    study = np.array([STUDIES.index(r.study) for r in chosen], dtype=int)
    i = np.array([r.first for r in chosen], dtype=int)
    j = np.array([r.second for r in chosen], dtype=int)
    y = np.array([r.value for r in chosen], dtype=float)
    wt = np.array(weights, dtype=float)
    wt /= wt.sum()
    return i, j, c, study, y, wt


def fit(task, blocks, eligible, labels, shared=SHARED):
    arrays = obs_arrays(task, eligible)
    ix, jx, channel, survey, y, wt = arrays
    mu, sd, cols, values = {}, {}, {}, []
    for k in labels:
        v = blocks[k]
        mu[k] = v[eligible].mean(axis=0)
        sd[k] = np.maximum(v[eligible].std(axis=0), 1e-5)
        sd[k] *= np.sqrt(max(v.shape[1], 24) / 24)
        z = np.clip((v - mu[k]) / sd[k], -8, 8)
        values.append(z)
        cols[k] = z.shape[1]
    x = np.column_stack(values)
    p = x.shape[1]
    n = len(x)
    num_channels = len(ASPECTS)
    num_studies = len(STUDIES)
    scalar = jx < 0
    left = ix * num_channels + channel
    right = jx[~scalar] * num_channels + channel[~scalar]
    strata = survey[scalar] * num_channels + channel[scalar]

    def objective(theta):
        beta = theta[: p * num_channels].reshape(p, num_channels)
        offsets = theta[p * num_channels :].reshape(num_studies, num_channels)
        s = x @ beta
        z = s.ravel()[left].copy()
        z[~scalar] -= s.ravel()[right]
        z[scalar] += offsets.ravel()[strata]
        loss = wt @ fractional_loss(z, y)
        dz = wt * (expit(z) - y)
        g = np.bincount(left, weights=dz, minlength=n * num_channels) - np.bincount(
            right, weights=dz[~scalar], minlength=n * num_channels
        )
        gb = x.T @ g.reshape(n, num_channels)
        go = np.bincount(strata, weights=dz[scalar], minlength=num_studies * num_channels)
        diff = beta - beta.mean(axis=1, keepdims=True)
        loss += 0.5 * RIDGE * np.sum(beta**2) + 0.5 * shared * np.sum(diff**2)
        gb += RIDGE * beta + shared * diff
        return float(loss), np.r_[gb.ravel(), go]

    start = np.zeros((p + num_studies) * num_channels)
    opt = minimize(
        objective,
        start,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 400, "ftol": 1e-12, "gtol": 3e-7, "maxls": 40},
    )
    grad = float(abs(opt.jac).max())
    if not opt.success or grad > 2e-5:
        raise RuntimeError(f"Unconverged revision head: {opt.message}, gradient {grad}")
    b = opt.x[: p * num_channels].reshape(p, num_channels)
    off = opt.x[p * num_channels :].reshape(num_studies, num_channels)
    return (
        x @ b,
        off,
        dict(coef=b, offset=off, gradient=grad, iterations=int(opt.nit), cols=cols, center=mu, scale=sd),
    )


def weights_for_mixture(task, oof, mask, arms):
    ix, jx, c, _, y, wt = obs_arrays(task, mask)
    result = []
    for k in range(len(ASPECTS)):
        sel = (c == k) & (jx >= 0)
        # Same-source differences cancel any study-specific rating scale intercept.
        base = oof[arms[0]][ix[sel], k] - oof[arms[0]][jx[sel], k]
        full = oof[arms[1]][ix[sel], k] - oof[arms[1]][jx[sel], k]

        def obj(alpha):
            return float(np.average(fractional_loss(base + alpha * (full - base), y[sel]), weights=wt[sel]))

        opt = minimize_scalar(obj, bounds=(0.0, 1.0), method="bounded", options={"xatol": 1e-7})
        chosen = min((0.0, float(opt.x), 1.0), key=obj)
        result.append(chosen)
    return np.array(result)


def anchored_calibration(task, raw, oof_score, final_score, eligible):
    """Preserve a known ordering cue when observed, held-family judgments warrant it."""
    ix, jx, chan, _, y, weights = obs_arrays(task, eligible)
    result = final_score.copy()
    metadata = {}
    shortening = np.array(
        [np.log1p(len(src.split())) - np.log1p(len(draft.split())) for src, draft in task.passages]
    )
    for aspect, cue in [("simplicity", shortening), ("meaning", raw["word_coverage"])]:
        k = ASPECTS.index(aspect)
        selected = (chan == k) & (jx >= 0)
        base = np.column_stack((oof_score[:, k], cue))
        mu = base[eligible].mean(axis=0)
        scale = np.maximum(base[eligible].std(axis=0), 1e-7)
        delta = (base[ix[selected]] - base[jx[selected]]) / scale
        wt = weights[selected] / weights[selected].sum()
        target = y[selected]
        prior = np.array([1.0, 0.0])

        def objective(coef):
            z = delta @ coef
            d = coef - prior
            return float(wt @ fractional_loss(z, target) + 0.01 * (d @ d)), delta.T @ (
                wt * (expit(z) - target)
            ) + 0.02 * d

        opt = minimize(
            objective,
            prior,
            jac=True,
            method="L-BFGS-B",
            bounds=[(0.0, None)] * 2,
            options={"gtol": 1e-8, "ftol": 1e-14, "maxiter": 150},
        )
        effective = np.where((opt.x < 1e-8) & (opt.jac > 0), 0, opt.jac)
        if not opt.success or max(abs(effective)) > 1e-5:
            raise RuntimeError("Source-coverage calibration failed to converge")
        result[:, k] = (np.column_stack((final_score[:, k], cue)) - mu) / scale @ opt.x
        metadata[aspect] = dict(
            coef=opt.x.tolist(),
            center=mu.tolist(),
            scale=scale.tolist(),
            gradient=float(max(abs(effective))),
            iterations=int(opt.nit),
        )
    return result, metadata


def run(gpu, out):
    out.mkdir(exist_ok=True, parents=True)
    lock = json.loads((OUT / "inputs.json").read_bytes())
    if any(
        digest((SOURCE if (SOURCE / k).exists() else SOURCE.parent) / k) != h
        for k, h in lock["material_files"].items()
    ):
        raise ValueError("One of the original study inputs has changed")
    task = build()
    vals, raw = features(task, gpu)
    families = np.array(task.groups)
    eligible = np.array([s.startswith("train/") for s in families])
    fold = np.array(
        [
            int(g.split("/", 1)[1], 16) % 5 if keep else -1
            for g, keep in zip(families, eligible, strict=True)
        ]
    )
    methods = {k: v for k, v in CONFIG.items() if set(v) <= set(vals)}
    oof = {}
    deploy = {}
    params = {}
    diagnostic = {}
    for name, cols in methods.items():
        oof[name] = np.full((len(fold), len(ASPECTS)), np.nan)
        receipt = []
        for f in [0, 1, 2, 3, 4, "full"]:
            train = eligible if f == "full" else eligible & (fold != f)
            pred, _, bundle = fit(
                task, vals, train, cols, shared=0.0 if name == "independent_full" else SHARED
            )
            receipt.append(dict(fold=f, gradient=bundle["gradient"], iterations=bundle["iterations"]))
            if f == "full":
                deploy[name] = pred
                for k, v in bundle.items():
                    if k in ("coef", "offset"):
                        params[f"{name}.{k}"] = v
                    elif k in ("center", "scale"):
                        params.update({f"{name}.{k}.{g}": a for g, a in v.items()})
            else:
                oof[name][fold == f] = pred[fold == f]
        if not np.isfinite(oof[name][eligible]).all():
            raise ValueError("Incomplete development forecasts")
        diagnostic[name] = receipt
    if gpu:
        mix = weights_for_mixture(task, oof, eligible, ("edit", "full"))
        params["mix"] = mix
        deploy["adaptive"] = deploy["edit"] + mix * (deploy["full"] - deploy["edit"])
        adaptive_oof = oof["edit"] + mix * (oof["full"] - oof["edit"])
        deploy["anchored"], anchored = anchored_calibration(
            task, raw, adaptive_oof, deploy["adaptive"], eligible
        )
        for asp, cal in anchored.items():
            for key in ("coef", "center", "scale"):
                params[f"anchored.{asp}.{key}"] = cal[key]
    else:
        mix = None
        anchored = None
    forecast = []
    for group, draft, i in task.heldout:
        cols = {k: v[i].tolist() for k, v in deploy.items()}
        cols.update({k: float(v[i]) for k, v in raw.items()})
        forecast.append(
            dict(
                group=group,
                source_id=__import__("hashlib").sha256(task.passages[i][0].encode()).hexdigest(),
                draft_id=draft,
                scores=cols,
            )
        )
    name = "forecasts.jsonl"
    path = out / name
    path.write_text("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in forecast))
    np.savez_compressed(out / "coefficients.npz", **params)
    ids = np.flatnonzero(eligible)
    np.savez_compressed(
        out / "development.npz", case=ids, fold=fold[eligible], **{k: v[eligible] for k, v in oof.items()}
    )
    receipt = dict(
        data_sha256=digest(OUT / "inputs.json"),
        source_sha256=digest(Path(__file__)),
        feature_code_sha256=digest(Path(__file__).with_name("revision_features.py")),
        training_design_sha256=digest(Path(__file__).with_name("revision_data.py")),
        extraction=digest(gpu / "extraction.json") if gpu else None,
        models=methods,
        mixing=mix.tolist() if gpu else None,
        anchored=anchored,
        sharing_protocol_sha256=digest(Path(__file__).with_name("revision-sharing-protocol.md")),
        prior_protocol_sha256=digest(Path(__file__).with_name("revision-prior-protocol.md")),
        aspects=ASPECTS,
        train_cases=int(eligible.sum()),
        source_families=len(set(families[eligible])),
        target_cases=len(forecast),
        train_observations=len(task.observations),
        convergence=diagnostic,
        files_sha256={k: digest(out / k) for k in (name, "coefficients.npz", "development.npz")},
    )
    (out / "forecast-seal.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print({k: receipt[k] for k in ("train_cases", "source_families", "target_cases", "mixing")}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=Path)
    parser.add_argument("--out", type=Path, default=OUT / "forecast")
    opts = parser.parse_args()
    run(opts.gpu, opts.out)
