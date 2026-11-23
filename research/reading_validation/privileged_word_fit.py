"""Use alternative causal encoders only on source texts to select reproducible Qwen directions."""
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
from scipy.linalg import eigh
from research.backbone_transfer.extract import digest, read_json, write_json
from research.reading_validation.convergent_adaptive import fit, forecast
from research.reading_validation.convergent_sparse import fit as sparse_fit
from research.reading_validation.meaning_score import folds
from research.reading_validation.physiology_portability_data import CANDIDATES, OUT
from research.reading_validation.physiology_portability_fit import (
    index_artifact,
    get_rows,
    evaluate_logits,
)
from research.reading_validation.convergent_adaptive import BASE

RESULT = OUT / "shared"
PLAN = Path(__file__).with_name("privileged-word-protocol.md")
ARMS = ("residual_pca", "shared_directions")


def training_view(native, surface, train):
    raw = native[train].astype("float32")
    sf = surface[train].astype(float)
    mu = sf.mean(0)
    sd = np.maximum(sf.std(0), 0.02)
    fixed = np.column_stack((np.ones(len(train)), (sf - mu) / sd))
    beta = np.linalg.lstsq(fixed, raw, rcond=None)[0]
    a = raw - (fixed @ beta).astype("float32")
    s = np.maximum(a.std(0, dtype=np.float64), 0.025).astype("float32")
    a /= s
    n = len(train)
    gram = np.array(a @ a.T / n, dtype=float)
    eig, U = eigh(gram, subset_by_index=[n - 96, n - 1])
    eig = eig[::-1]
    U = U[:, ::-1]
    if eig.min() <= 0.001:
        raise ValueError("Could not identify 96 reliable source coordinates")
    basis = ((a.T @ U) / np.sqrt(n) / eig).astype("float32")
    u = a @ basis
    if np.max(abs(u.T @ u / n - np.eye(96))) > 0.002:
        raise ValueError("Source covariance decomposition failed")
    matrix = basis / s[:, None]
    return u, dict(
        hidden=matrix, surface_correction=beta @ matrix, surface_mean=mu, surface_sd=sd
    )


def project(native, surface, q, rotation):
    hidden = (q["hidden"] @ rotation).astype("float32")
    correction = q["surface_correction"] @ rotation
    fixed = np.column_stack(
        (np.ones(len(surface)), (surface - q["surface_mean"]) / q["surface_sd"])
    )
    z = np.zeros((len(native), 24), dtype=float)
    for i in range(0, len(native), 256):
        sl = slice(i, i + 256)
        z[sl] = native[sl].astype("float32") @ hidden - fixed[sl] @ correction
    return np.column_stack((fixed, z)), dict(
        hidden=hidden,
        surface_correction=correction,
        surface_mean=q["surface_mean"],
        surface_sd=q["surface_sd"],
    )


def main(paths, historical):
    seat = read_json(RESULT / "registration.json")
    if digest(PLAN) != seat["protocol"] or seat["nine_lab_scores_opened"]:
        raise ValueError("Invalid prospective study registration")
    inputs = {}
    native = {}
    surface = {}
    locations = None
    for path in paths:
        n, s, off, c = index_artifact(path)
        if c in native or (locations is not None and locations != off):
            raise ValueError("Common word support changed across backbones")
        native[c] = n
        surface[c] = s
        locations = off
        inputs[c] = digest(path / "receipt.json")
    if set(native) != set(CANDIDATES):
        raise ValueError("All source views required")
    idx, ids, items, Y, o, es_idx, es_y, es_family, nine, nine_idx = get_rows(
        historical, locations
    )
    part = folds(ids, "UCL 2026 reading sentence:")
    F = np.array([part[s] for s in ids])
    P = {}
    L = {}
    E = {}
    M = {}
    books = []
    for f in range(6):
        tr = (F != f) if f < 5 else np.ones(len(ids), bool)
        te = ~tr
        u = {}
        q = None
        for c in CANDIDATES:
            u[c], pack = training_view(native[c], surface[c], idx[tr])
            if c == "qwen":
                q = pack
        cross = (
            sum(
                (
                    (u["qwen"].T @ u[c] / len(u[c])) @ (u[c].T @ u["qwen"] / len(u[c]))
                ).astype(float)
                for c in CANDIDATES
                if c != "qwen"
            )
            / 2.0
        )
        eig, R = eigh(cross)
        R = R[:, ::-1][:, :24]
        if eig[-1] > 1.002 or eig[0] < -0.001:
            raise ValueError("Cross-view canonical predictability lies outside [0,1]")
        ck = {}
        fits = {}
        checks = {}
        if f < 5:
            sets = read_json(BASE / f"sparse/context-fold-{f}.json")[
                "selected_sentences"
            ]
            obs = np.isin(ids[tr], sets)
            masked = Y[tr].copy()
            masked[~obs, 0] = np.nan
        for arm, rot in [("residual_pca", np.eye(96, 24)), ("shared_directions", R)]:
            x, geometry = project(native["qwen"], surface["qwen"], q, rot)
            parity = float(abs(x[idx[tr], 7:] - u["qwen"] @ rot).max())
            if parity > 0.004:
                raise ValueError(
                    f"Deployed linear Qwen projection is inconsistent with training: {parity}"
                )
            p = RESULT / (arm + f"-basis-{f}.npz")
            np.savez_compressed(p, **geometry)
            ck[arm] = digest(p)
            checks[arm] = parity
            fits[arm] = fit(x[idx[tr]], Y[tr], o[tr, 0], o[tr, 2], "shared")
            if f < 5:
                fits[arm + "_sparse"] = sparse_fit(
                    x[idx[tr]], masked, o[tr, 0], o[tr, 2], obs, "shared"
                )
                for key in [arm, arm + "_sparse"]:
                    if key not in P:
                        P[key] = np.full_like(Y, np.nan)
                        L[key] = np.full_like(Y, np.nan)
                    P[key][te], L[key][te] = forecast(
                        x[idx[te]], Y[te], o[te, 0], o[te, 2], fits[key]
                    )
            else:
                E[arm] = evaluate_logits(x[es_idx], fits[arm])[:, 0]
                M[arm] = evaluate_logits(x[nine_idx.ravel()], fits[arm]).reshape(
                    -1, 2, 3
                )
        path = RESULT / f"fold-{f}.json"
        write_json(
            path,
            dict(
                protocol=digest(PLAN),
                source_archives=inputs,
                held_items=items[te].tolist(),
                basis=ck,
                encoding_numerical_error=checks,
                models=fits,
                canonical_spectrum=eig[::-1].tolist(),
                principal_angle_cosines=np.linalg.svd(
                    R[:24], compute_uv=False
                ).tolist(),
            ),
        )
        books.append(digest(path))
        print("Single-Qwen aligned fit", f, flush=True)
    out = RESULT / "forecasts.npz"
    np.savez_compressed(
        out,
        items=items,
        sentence=ids,
        Y=Y,
        n=o[:, 0],
        es_y=es_y,
        es_family=es_family,
        **{"P_" + k: v for k, v in P.items()},
        **{"L_" + k: v for k, v in L.items()},
        **{"es_" + k: v for k, v in E.items()},
        **{"multi_" + k: v for k, v in M.items()},
    )
    write_json(
        RESULT / "forecast-lock.json",
        dict(
            protocol=digest(PLAN),
            program=digest(Path(__file__)),
            archive=digest(out),
            labs_order=[r["id"] for r in nine],
            folds=books,
            source_encoders=list(CANDIDATES),
            deployment_encoder="qwen",
        ),
    )


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("historical", type=Path)
    p.add_argument("archives", nargs=3, type=Path)
    a = p.parse_args()
    main(a.archives, a.historical)
