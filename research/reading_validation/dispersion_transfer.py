"""Lock the UCL-only forecasts at real held stems, then evaluate chronological reader responses."""

import argparse
from collections import defaultdict
from hashlib import sha256
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix

from research.backbone_transfer.extract import digest, read_json, write_json
from research.reading_validation.cloze_population_data import OUT as POP
from research.reading_validation.cloze_population_model import (
    distribution,
    material,
    project,
)
from research.reading_validation.reader_history_external import (
    answer_normalizer,
    records,
    sources,
)
from research.reading_validation.reader_history_materials import DIR
from research.reading_validation.reader_history_materials import OUT as HISTORY
from research.reading_validation.word_population_dispersion import (
    OUT,
    components,
    gates,
)

PLAN = Path(__file__).with_name("dispersion-transfer-protocol.md")
BASE = DIR / "dispersion-transfer-base.npy"
METHODS = ("original", "choices", "joint")


def lock(native, contextual):
    ext = read_json(HISTORY / "external-lock.json")
    nr = read_json(native / "receipt.json")
    r = read_json(contextual / "receipt.json")
    if (
        digest(native / "receipt.json") != ext["native"]
        or digest(native / "logs.npz") != nr["output"]
        or r["source"] != nr["input"]
        or r["states"] != digest(contextual / "states.npz")
        or r["batch_forwards"] != 97
        or r["observed_responses_in_input"]
    ):
        raise ValueError(
            "The two original native measurements do not match the registered stem corpus"
        )
    with np.load(native / "logs.npz") as d, np.load(contextual / "states.npz") as z:
        native_logs = d["native"]
        native_ids = d["ids"]
        states = z["state"]
        if sha256(native_logs.tobytes()).hexdigest() != r[
            "native_array_sha256"
        ] or not np.array_equal(native_ids, z["ids"]):
            raise ValueError(
                "An actual one-pass population forecast must reproduce native probabilities and IDs"
            )
    trained = read_json(OUT / "forecast-lock.json")
    fit = read_json(POP / "fold-5.json")
    if trained["frozen_base_fits"][-1] != digest(POP / "fold-5.json"):
        raise ValueError("Base forecast changed after source training")
    with np.load(POP / "serving/weights.npz") as f:
        U, V, route, share = [f[k] for k in ("U", "V", "route", "share")]
    buf = np.lib.format.open_memmap(
        BASE, mode="w+", dtype=np.float32, shape=(len(native_ids), len(route) + 1)
    )
    X = np.empty((len(native_ids), 4))
    for k in range(0, len(states), 36):
        ii = np.arange(k, min(len(states), k + 36))
        lp = material(ii, native_logs, route, share)
        lp = distribution(
            lp,
            project(states[ii].astype(float), fit["geometry"]),
            fit["models"]["context"],
            U,
            V,
        )
        buf[ii] = lp
        p = np.exp(lp)
        X[ii] = np.column_stack(
            [
                np.ones(len(ii)),
                p.max(1),
                p[:, -1],
                -(p * lp).sum(1) / np.log(p.shape[1]),
            ]
        )
    buf.flush()
    X[:, 1:] = (X[:, 1:] - trained["source_center"]) / trained["source_scale"]
    weights = {
        k: gates(X, np.array(m["coefficients"])) for k, m in trained["models"].items()
    }
    path = OUT / "transfer-gates.npz"
    np.savez_compressed(path, ids=native_ids, **weights)
    write_json(
        OUT / "transfer-lock.json",
        {
            "protocol": digest(PLAN),
            "native": digest(native / "receipt.json"),
            "contextual": digest(contextual / "receipt.json"),
            "source": digest(OUT / "forecast-lock.json"),
            "gates": digest(path),
            "probabilities": digest(BASE),
            "original_responses_opened": False,
            "matched_native_logits": True,
            "outcome_selection": ext["future_item_assignments"],
            "stems": len(states),
            "batch_forwards": r["batch_forwards"],
        },
    )


def grade():
    l = read_json(OUT / "transfer-lock.json")
    future = DIR / "independent-future-items.json"
    if (
        digest(BASE) != l["probabilities"]
        or digest(OUT / "transfer-gates.npz") != l["gates"]
        or digest(future) != l["outcome_selection"]
        or read_json(HISTORY / "external-lock.json")["inputs"] != sources()
    ):
        raise ValueError("Forecast or original study score sheet changed")
    lookup = defaultdict(list)
    for rid, f, k, field, no, stem in read_json(future)["chronological"]:
        lookup[f, k].append((rid, field, no, stem))
    seen = set()
    cases = []
    missing = 0
    normal = answer_normalizer()
    for f, k, row, fields in records():
        for rid, field, no, stem in lookup.get((f, k), ()):
            val, _ = normal(row[field], no)
            if val is None:
                missing += 1
                continue
            if (rid, stem) in seen:
                raise ValueError(
                    "Duplicate response by same reader to same original stem"
                )
            cases.append((rid, stem, val))
            seen.add((rid, stem))
    cases = np.array(sorted(cases, key=lambda r: (r[1], r[0])))
    r, stem, choices = cases.T
    _, ri = np.unique(r, return_inverse=True)
    rows = np.load(BASE, mmap_mode="r")
    N = len(rows)
    per = np.empty((len(cases), 3))
    outside = np.empty_like(per)
    dis = np.empty((N, 3))
    obs = np.full(N, np.nan)
    count = np.bincount(stem, minlength=N)
    offset = np.searchsorted(stem, np.arange(N + 1))
    with np.load(OUT / "transfer-gates.npz") as gw:
        gates0 = {k: gw[k] for k in METHODS[1:]}
    for k in range(0, N, 32):
        idx = np.arange(k, min(N, k + 32))
        p = components(rows[idx].astype(float))
        forecasts = [
            p[:, :, 1],
            *[np.einsum("ijk,ik->ij", p, gates0[m][idx]) for m in METHODS[1:]],
        ]
        for pos, original in enumerate(idx):
            sl = slice(offset[original], offset[original + 1])
            c = choices[sl]
            n = len(c)
            if n >= 2:
                v = np.unique(c, return_counts=True)[1]
                obs[original] = 1.0 - v @ (v - 1) / (n * (n - 1.0))
            for j, allp in enumerate(forecasts):
                prob = allp[pos]
                per[sl, j] = -np.log(prob[c])
                outside[sl, j] = -np.where(
                    c == len(prob) - 1, np.log(prob[-1]), np.log1p(-prob[-1])
                )
                dis[original, j] = 1 - prob @ prob
    R = max(ri) + 1
    G = N
    rng = np.random.default_rng(221601)
    B = 2000
    strengths = rng.multinomial(R, np.full(R, 1 / R), B)
    by_word = rng.multinomial(G, np.full(G, 1 / G), B)

    def draws(v):
        mat = coo_matrix((v, (ri, stem)), shape=(R, G)).tocsr()
        ans = []
        for i in range(0, B, 32):
            ans.extend(
                np.sum(strengths[i : i + 32] * (mat @ by_word[i : i + 32].T).T, axis=1)
            )
        return np.array(ans)

    D = draws(np.ones(len(r)))
    named = choices != 8192
    diffs = per[:, [0]] - per[:, 1:]
    est = diffs.mean(0)
    boots = np.stack([draws(diffs[:, i]) / D for i in range(2)])
    sd = boots.std(1)
    crit = np.quantile(np.max(abs((boots - est[:, None]) / sd[:, None]), axis=0), 0.95)
    measured = np.isfinite(obs)
    summary = {
        k: {
            "nll": float(per[:, j].mean()),
            "word_class_nll": float(outside[:, j].mean()),
            "named_conditional_nll": float((per[named, j] - outside[named, j]).mean()),
            "pair_disagreement_mae": float(np.mean(abs(obs[measured] - dis[measured, j]))),
        }
        for j, k in enumerate(METHODS)
    }
    gains = {
        k: {
            "nll_reduction": float(est[j]),
            "simultaneous": (est[j] + np.array([-1, 1]) * sd[j] * crit).tolist(),
        }
        for j, k in enumerate(METHODS[1:])
    }
    file = OUT / "transfer-word-results.npz"
    np.savez_compressed(
        file, observed=obs, counts=count, probability=dis, method=np.array(METHODS)
    )
    write_json(
        OUT / "transfer-results.json",
        {
            "forecast": digest(OUT / "transfer-lock.json"),
            "material_scores": digest(file),
            "readers": int(R),
            "stems": N,
            "with_two_readers": int(measured.sum()),
            "responses": len(r),
            "outside": int((~named).sum()),
            "omitted_empty": missing,
            "repeated_reader_word": 0,
            "estimates": summary,
            "primary_contrasts": gains,
        },
    )
    print(summary, gains)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="action", required=True)
    s = sub.add_parser("lock")
    s.add_argument("native", type=Path)
    s.add_argument("contextual", type=Path)
    sub.add_parser("score")
    args = p.parse_args()
    if args.action == "lock":
        lock(args.native, args.contextual)
    else:
        grade()
