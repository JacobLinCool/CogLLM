"""Calibrate and score the independent materials of the third internal study-family partition."""

import argparse
import json
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from research.backbone_transfer.extract import digest
from research.reading_validation.borrowing_data import CALIBRATE, EVALUATE
from research.reading_validation.borrowing_prior import (
    METHODS,
    SEEDS,
    calibrate_grid,
    calibration_curves,
    grid,
    read,
    write,
)
from research.reading_validation.borrowing_prior_score import group_id, source_interval
from research.reading_validation.borrowing_reliability import (
    adjusted,
    disagreement,
    family_mean,
    permute,
)
from research.reading_validation.borrowing_reliability_score import variable_nll
from research.reading_validation.borrowing_replication import OUT, PLAN, states
from research.reading_validation.borrowing_score import GeometricPool
from research.union_training.fit_observations import COUNT_MODELS, weights

ARMS = (*METHODS, "material_gate", "misaligned_gate")


def get(stream, seed, side):
    p = OUT / f"seed-{seed}"
    r = read(p / "fit.json")
    if r["geometry"] != digest(OUT / "material-freeze.json"):
        raise ValueError("The third expert did not use the prospective held families")
    res = []
    for label in ("shared", "private"):
        f = p / f"{label}-{side}.npz"
        if digest(f) != r["files"][f.name]:
            raise ValueError("An independently trained forecast changed")
        res.append(stream.enter_context(np.load(f, allow_pickle=False)))
    return tuple(res)


def calibrate(tasks, parts):
    nums = {n: grid(n) for n in (97, 193)}
    params = {}
    summary = {}
    meta = {}
    alphas = {}
    with ExitStack() as st:
        cal = [get(st, s, "calibration") for s in SEEDS]
        held = [get(st, s, "development") for s in SEEDS]
        for j, seed in enumerate(SEEDS):
            curves, info = calibration_curves(tasks, parts, cal[j], nums)
            low, high = [calibrate_grid(nums[n], curves[n], info) for n in nums]
            lh, lc = low
            hh, hc = high
            err = max(abs(lc[t][m] - hc[t][m]) for t in lc for m in METHODS)
            od = max(
                abs(
                    lh[s]["convergence"]["objective"]
                    - hh[s]["convergence"]["objective"]
                )
                / hh[s]["training_sources"]
                for s in hh
            )
            chosen = 193 if err > 0.0005 or od > 0.005 else 97
            h, c = low if chosen == 97 else high
            params[seed] = c
            summary[str(seed)] = {
                "source_priors": h,
                "weights": c,
                "calibration_geometry": info,
                "quadrature_choice": chosen,
                "quadrature_max_coefficient_error": err,
                "quadrature_mean_objective_error": od,
                "source_fit": digest(OUT / f"seed-{seed}" / "fit.json"),
            }
        for t in tasks:
            ev = np.flatnonzero(parts[t.name] == EVALUATE)
            ci = np.flatnonzero(parts[t.name] == CALIBRATE)
            if not len(ev):
                continue
            s, p = (
                disagreement(t.likelihood, held[0][k][t.name], held[1][k][t.name])
                for k in (0, 1)
            )
            r = np.ones(len(ev))
            moved = 0
            ms = mp = None
            if len(ci):
                cs, cp = (
                    disagreement(t.likelihood, cal[0][k][t.name], cal[1][k][t.name])
                    for k in (0, 1)
                )
                ms = family_mean(cs, t.family[ci])
                mp = family_mean(cp, t.family[ci])
                if ms > 1e-12 and mp > 1e-12:
                    r = (1 + p / mp) / (1 + s / ms)
            valid = (
                np.isfinite(held[0][0][t.name]).sum(axis=1)
                if t.likelihood in COUNT_MODELS
                else np.ones(len(ev))
            )
            shuffled, moved = permute(t, ev, valid, r)
            meta[t.name] = {
                "rows": len(ev),
                "moved": moved,
                "cal_shared_H2": ms,
                "cal_private_H2": mp,
            }
            for seed in SEEDS:
                a = params[seed][t.name]["independent_mode"]
                alphas[t.name + "." + str(seed)] = np.stack(
                    (adjusted(a, r), adjusted(a, shuffled)), axis=1
                )
    out = OUT / "material-coefficients.npz"
    np.savez_compressed(out, **alphas)
    write(
        OUT / "forecast-freeze.json",
        {
            "code": digest(Path(__file__)),
            "protocol": digest(PLAN),
            "geometry": digest(OUT / "material-freeze.json"),
            "coefficients": summary,
            "input_reliability": meta,
            "material_weight_file": digest(out),
            "responses_scored_before_freeze": False,
        },
    )
    print("Registered third coefficient sets", len(meta), flush=True)


def report(matrix, names):
    n = len(matrix)
    draw = np.random.default_rng(806721).integers(n, size=(10000, n))
    flip = np.random.default_rng(22279).choice(np.int8([-1, 1]), size=(100000, n))
    pairs = [
        ("independent_mode", "midpoint"),
        ("independent_mode", "private"),
        ("independent_mode", "shared"),
        ("supported", "independent_mode"),
        ("supported", "midpoint"),
        ("transported", "midpoint"),
        ("material_gate", "independent_mode"),
        ("material_gate", "misaligned_gate"),
    ]
    allout = {}
    for a, b in pairs:
        delta = matrix[:, names.index(b)] - matrix[:, names.index(a)]
        v = float(delta.mean())
        pv = float(
            (1 + np.count_nonzero(flip @ delta / n >= v - 1e-15)) / (len(flip) + 1)
        )
        allout[a + "_over_" + b] = {
            "gain_nats_per_trial": v,
            "source_confidence_interval95": np.quantile(
                delta[draw].mean(1), [0.025, 0.975]
            ).tolist(),
            "one_sided_signflip_p": pv,
            "improved_sources": int((delta > 0).sum()),
            "declined_sources": int((delta < 0).sum()),
        }
    return {
        "source_mean_nll": dict(zip(names, matrix.mean(0), strict=True)),
        "paired": allout,
    }


def score(tasks, parts):
    seal = read(OUT / "forecast-freeze.json")
    path = OUT / "material-coefficients.npz"
    geo = read(OUT / "material-freeze.json")
    if (
        seal["protocol"] != digest(PLAN)
        or seal["code"] != digest(Path(__file__))
        or seal["material_weight_file"] != digest(path)
        or seal["geometry"] != digest(OUT / "material-freeze.json")
    ):
        raise ValueError("Newly held outcome selection or model changed")
    src = defaultdict(list)
    endpoint = {}
    family = {}
    npz = {}
    with np.load(path, allow_pickle=False) as bank, ExitStack() as stack:
        trained = [get(stack, s, "development") for s in SEEDS]
        for t in tasks:
            ix = np.flatnonzero(parts[t.name] == EVALUATE)
            if not len(ix):
                continue
            w = weights(t, ix)
            g, ig = np.unique(t.family[ix], return_inverse=True)
            den = np.bincount(ig, weights=w * t.trials[ix], minlength=len(g))
            sums = []
            seen = {}
            for j, seed in enumerate(SEEDS):
                pool = GeometricPool(t, ix, *(z[t.name] for z in trained[j]))
                co = seal["coefficients"][str(seed)]["weights"][t.name]
                var = bank[t.name + "." + str(seed)]
                row = np.column_stack(
                    [
                        *(pool.loss(co[nm]) for nm in METHODS),
                        *(variable_nll(pool, v) for v in var.T),
                    ]
                )
                num = np.column_stack(
                    [np.bincount(ig, weights=w * v, minlength=len(g)) for v in row.T]
                )
                sums.append(num)
                seen[str(seed)] = dict(zip(ARMS, num.sum(0) / den.sum(), strict=True))
            fid = np.array([group_id(t.source, z) for z in g])
            family[t.name] = (fid, sums, den)
            endpoint[t.name] = {
                "source": t.source,
                "likelihood": t.likelihood,
                "held_rows": len(ix),
                "held_families": len(g),
                "actual_response_count": float(np.sum(t.trials[ix])),
                "outcomes_by_seed": seen,
            }
            src[t.source].append(t.name)
            npz[t.name + ".groups"] = fid
            npz[t.name + ".numerator"] = np.stack(sums)
            npz[t.name + ".denominator"] = den
    sums = {}
    allmodels = []
    sorted_src = sorted(src)
    seed_out = {}
    for j, seed in enumerate(SEEDS):
        p = np.array(
            [
                np.mean(
                    [
                        [endpoint[k]["outcomes_by_seed"][str(seed)][m] for m in ARMS]
                        for k in src[s]
                    ],
                    axis=0,
                )
                for s in sorted_src
            ]
        )
        allmodels.append(p)
        seed_out[str(seed)] = report(p, ARMS)
    matrix = np.mean(allmodels, axis=0)
    for i, s in enumerate(sorted_src):
        es = src[s]
        items = sorted({v for k in es for v in family[k][0]})
        lookup = {a: j for j, a in enumerate(items)}
        num = np.zeros((len(items), len(es), len(METHODS)))
        den = np.zeros((len(items), len(es)))
        for j, k in enumerate(es):
            labels, params, n = family[k]
            ix = [lookup[v] for v in labels]
            num[ix, j] = np.mean(params, axis=0)[:, : len(METHODS)]
            den[ix, j] = n
        stats = source_interval(num, den)
        sums[s] = {
            "observed_endpoints": len(es),
            "new_held_families": len(items),
            "nll": dict(zip(ARMS, matrix[i], strict=True)),
            "paired_map_over_half": float(
                matrix[i, ARMS.index("midpoint")]
                - matrix[i, ARMS.index("independent_mode")]
            ),
            "paired_map_over_half_ci95": None
            if stats["intervals"] is None
            else stats["intervals"]["independent_mode_over_midpoint"],
            "valid_material_resamples": stats["contributing_draws"],
        }
    file = OUT / "family-nll-statistics.npz"
    np.savez_compressed(file, **npz)
    d = {
        "protocol": digest(PLAN),
        "frozen_coefficients": digest(OUT / "forecast-freeze.json"),
        "source_families": sum(v["new_held_families"] for v in sums.values()),
        "new_held_rows": sum(
            len(np.flatnonzero(parts[t.name] == EVALUATE)) for t in tasks
        ),
        "registered_sources": len(geo["assignments"]),
        "sources_evaluated": len(sums),
        "endpoints_evaluated": len(endpoint),
        "joint_two_seed_result": report(matrix, ARMS),
        "single_seed_results": seed_out,
        "sources": sums,
        "endpoints": endpoint,
        "family_nll_statistics": digest(file),
        "benefit_in_nats": "control minus candidate, lower held loss is preferred",
    }
    write(OUT / "scores.json", d)
    print(
        "Third family material",
        d["sources_evaluated"],
        d["source_families"],
        d["joint_two_seed_result"]["paired"]["independent_mode_over_midpoint"],
        flush=True,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("calibrate", "score"))
    args = p.parse_args()
    import torch

    torch.set_num_threads(1)
    cache, tasks, parts, geom = states()
    del cache
    if read(OUT / "material-freeze.json") != geom:
        raise ValueError("Reconstructed geometry changed")
    (calibrate if args.stage == "calibrate" else score)(tasks, parts)


if __name__ == "__main__":
    main()
