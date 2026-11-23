"""Refit and compare reading observation experts after whole-archive article quarantine."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize_scalar

from research.backbone_transfer.extract import digest, stream_rows
from research.reading_validation.borrowing_data import (
    CALIBRATE,
    EVALUATE,
    FIT,
    PACKET,
)
from research.reading_validation.borrowing_family_audit import (
    ARMS,
    COMPARISONS,
    FamilyRisks,
    contrasts,
    exact_signflip,
)
from research.reading_validation.borrowing_family_bootstrap import estimate, summary
from research.reading_validation.borrowing_fit import Private, train
from research.reading_validation.borrowing_prior import SEEDS, read, write
from research.reading_validation.borrowing_replication import states
from research.reading_validation.borrowing_score import GeometricPool, material_weights
from research.union_training.fit_observations import (
    COUNT_MODELS,
    Readouts,
    observation_scales,
    task_responses,
    weights,
)

OUT = Path("research/results/reading-validation-family-bootstrap/global")
PLAN = Path(__file__).with_name("borrowing-global-families-protocol.md")
LINKED = {"race", "onestop", "onestop-rereading"}
QUARANTINE = 4


def protocol():
    return digest(PLAN)


def load():
    h, tasks, original, old = states()
    per_family = defaultdict(dict)
    for t in tasks:
        for uid, r in zip(t.family, original[t.name], strict=True):
            if per_family[uid].setdefault(t.source, int(r)) != r:
                raise ValueError("Original partition divided a source family")
    parts = {}
    records = {}
    for t in tasks:
        oldr = original[t.name]
        role = np.array(
            [
                r if r == max(per_family[g].values()) else QUARANTINE
                for r, g in zip(oldr, t.family, strict=True)
            ]
        )
        if not np.array_equal(oldr == EVALUATE, role == EVALUATE) or not np.any(
            role == FIT
        ):
            raise ValueError("Affected original evaluation or fitting support")
        parts[t.name] = role
        if t.likelihood not in COUNT_MODELS:
            values = task_responses(
                t.likelihood,
                list(stream_rows(PACKET / (t.name + ".jsonl"))),
                t.coordinates.shape[1],
            )
            t.trials, t.center, t.spread = observation_scales(
                t.likelihood, values, role
            )
            t.response = torch.as_tensor(values, dtype=torch.float32)
        records[t.name] = {
            "source": t.source,
            "registered_roles": np.bincount(role, minlength=5).tolist(),
            "quarantined_by_original_role": np.bincount(
                oldr[role == QUARANTINE], minlength=4
            ).tolist(),
        }
    attended = {}
    for t in tasks:
        for g, r in zip(t.family, parts[t.name], strict=True):
            if r != QUARANTINE and attended.setdefault(g, r) != r:
                raise ValueError("A material entered more than one active split")
    shared = {str(k): m for k, m in per_family.items() if len(m) > 1}
    meta = {
        "protocol": protocol(),
        "program": digest(Path(__file__)),
        "original_geometry": old["code"],
        "linked_article_original_roles": shared,
        "registered_by_endpoint": records,
        "remaining_globally_unique_families": len(attended),
        "original_and_corrected_held_rows_identical": True,
        "held_fit_global_family_intersection": 0,
    }
    return h, tasks, parts, meta


def fit():
    OUT.mkdir(parents=True, exist_ok=True)
    h, tasks, parts, meta = load()
    write(OUT / "geometry.json", meta)
    for s in SEEDS:
        torch.manual_seed(s)
        common = Readouts(h.shape[1], tasks)
        private = Private(common, tasks)
        for t in tasks:
            rows = torch.from_numpy(np.flatnonzero(parts[t.name] == FIT)[:3])
            if not torch.equal(common(t, h, rows), private(t, h, rows)):
                raise AssertionError("Unmatched initial predictions")
        path = OUT / f"seed-{s}"
        path.mkdir(exist_ok=True)
        results = {
            key: train(model, key, tasks, h, parts, s, path)
            for key, model in (("shared", common), ("private", private))
        }
        if any(v["stop"] != "inner_likelihood_plateau" for v in results.values()):
            raise ArithmeticError("An expert failed its convergence guard")
        write(
            path / "fit.json",
            {
                "geometry": digest(OUT / "geometry.json"),
                "seed": s,
                "fits": results,
                "files": {p.name: digest(p) for p in sorted(path.glob("*.npz"))},
            },
        )
        print("completed linked-article fit", s, flush=True)


def read_experts(stream, seed, stage):
    root = OUT / f"seed-{seed}"
    receipt = read(root / "fit.json")
    if receipt["geometry"] != digest(OUT / "geometry.json"):
        raise ValueError("Training and calibration material disagree")
    archive = []
    for arm in ("shared", "private"):
        file = root / f"{arm}-{stage}.npz"
        if digest(file) != receipt["files"][file.name]:
            raise ValueError("Changed expert forecasts")
        archive.append(stream.enter_context(np.load(file, allow_pickle=False)))
    return archive


def calibrate():
    _, tasks, split, geo = load()
    if geo != read(OUT / "geometry.json"):
        raise ValueError("Changed global article roles")
    frozen = {}
    for s in SEEDS:
        table = {}
        with ExitStack() as st:
            predictions = read_experts(st, s, "calibration")
            for t in tasks:
                rows = np.flatnonzero(split[t.name] == CALIBRATE)
                if not len(rows):
                    table[t.name] = {
                        "endpoint_map": 0.5,
                        "family_bootstrap": 0.5,
                        "families": 0,
                    }
                    continue
                pool = GeometricPool(t, rows, *(p[t.name] for p in predictions))
                w, _ = material_weights(t, rows)

                def objective(a, w=w, pool=pool):
                    return np.dot(w, pool.loss(a)) - np.log(a) - np.log1p(-a)

                ans = minimize_scalar(
                    objective,
                    method="bounded",
                    bounds=(1e-7, 1 - 1e-7),
                    options={"xatol": 2e-8},
                )
                if not ans.success:
                    raise ArithmeticError("Failed original MAP fit")
                # Hold the old pseudorandom bootstrap draws fixed under the correction.
                b, diag = estimate(pool, t, rows, "third")
                table[t.name] = {
                    "endpoint_map": float(ans.x),
                    "family_bootstrap": b,
                    **diag,
                }
        frozen[str(s)] = table
    write(
        OUT / "forecast-freeze.json",
        {
            "geometry": digest(OUT / "geometry.json"),
            "program": digest(Path(__file__)),
            "estimator": digest(Path(estimate.__code__.co_filename)),
            "protocol": protocol(),
            "seeds": frozen,
        },
    )
    print("froze coefficients after whole-archive article quarantine", flush=True)


def ci_table(matrix, draws):
    paired = contrasts(matrix).mean(axis=0)
    return {
        name: {
            "gain": float(paired[j]),
            "joint_source_material_simultaneous_ci": np.quantile(
                draws[:, j], [0.0125, 0.9875]
            ).tolist(),
            "joint_source_material_nominal_ci": np.quantile(
                draws[:, j], [0.025, 0.975]
            ).tolist(),
            "exact_signflip_p": exact_signflip(contrasts(matrix)[:, j]),
            "leave_one_range": [
                float(
                    np.min(
                        (len(matrix) * paired[j] - contrasts(matrix)[:, j])
                        / (len(matrix) - 1)
                    )
                ),
                float(
                    np.max(
                        (len(matrix) * paired[j] - contrasts(matrix)[:, j])
                        / (len(matrix) - 1)
                    )
                ),
            ],
        }
        for j, name in enumerate(COMPARISONS)
    }


def grouped_samples(items, store, mode):
    """Jointly resample underlying article strings, averaging endpoints within source and sources within group."""
    keys = [(s, n) for s in items for n in store[s]]
    groups = sorted({g for s, k in keys for g in store[s][k].groups})
    pos = {k: i for i, k in enumerate(groups)}
    den = np.zeros((len(groups), len(keys)))
    num = np.zeros((len(groups), len(keys), 3))
    for j, (s, k) in enumerate(keys):
        x = store[s][k]
        loc = [pos[a] for a in x.groups]
        den[loc, j] = x.denominator
        num[loc, j] = x.numerator[:, :3]

    def summarize(x):
        return np.stack(
            [
                x[:, [j for j, (ss, _) in enumerate(keys) if ss == s]].mean(axis=1)
                for s in items
            ],
            axis=1,
        ).mean(axis=1)

    exact = summarize((num.sum(axis=0) / den.sum(axis=0)[:, None])[None])[0]
    G = len(groups)
    if G == 1:
        return exact, np.broadcast_to(contrasts(exact), (4000, 2))
    stream = np.random.default_rng(
        int(sha256((mode + "/" + " ".join(items)).encode()).hexdigest()[:16], 16)
    )
    out = []
    n = 0
    while n < 4000:
        selected = stream.multinomial(G, np.full(G, 1 / G), size=100)
        mass = selected @ den
        good = (mass > 0).all(axis=1)
        if good.any():
            q = (selected[good] @ num.reshape(G, -1)).reshape(-1, len(keys), 3) / mass[
                good, :, None
            ]
            arr = contrasts(summarize(q))
            out.append(arr)
            n += len(arr)
    return exact, np.concatenate(out)[:4000]


def grouped_summary(stores, blocknames, mode):
    point, draws = zip(
        *(grouped_samples(bl, stores, mode) for bl in blocknames), strict=True
    )
    point, draws = np.array(point), np.array(draws)
    n = len(point)
    rng = np.random.default_rng(8176213)
    ix = rng.integers(n, size=(20000, n))
    m = rng.integers(4000, size=ix.shape)
    return {
        "units": len(blocknames),
        "source_and_material": ci_table(point, draws[ix, m].mean(axis=1)),
        "registered_source_only": summary(point),
    }


def score():
    _, tasks, parts, geo = load()
    f = read(OUT / "forecast-freeze.json")
    if (
        geo != read(OUT / "geometry.json")
        or f["geometry"] != digest(OUT / "geometry.json")
        or f["estimator"] != digest(Path(estimate.__code__.co_filename))
        or f["program"] != digest(Path(__file__))
    ):
        raise ValueError("Changed material or forecast after freeze")
    rows_by = defaultdict(dict)
    source_risks = defaultdict(list)
    packed = {}
    counts = Counter()
    endpoints = {}
    with ExitStack() as st:
        files = [read_experts(st, s, "development") for s in SEEDS]
        for t in tasks:
            ix = np.flatnonzero(parts[t.name] == EVALUATE)
            if not len(ix):
                continue
            g, grp = np.unique(t.family[ix], return_inverse=True)
            hashed = [sha256(k.encode()).hexdigest() for k in g]
            w = weights(t, ix)
            den = np.bincount(grp, weights=w * t.trials[ix], minlength=len(g))
            two = []
            for seed, bank in zip(SEEDS, files, strict=True):
                p = GeometricPool(t, ix, *(file[t.name] for file in bank))
                params = f["seeds"][str(seed)][t.name]
                two.append(
                    np.column_stack(
                        [
                            np.bincount(grp, weights=w * p.loss(a), minlength=len(g))
                            for a in (
                                params["endpoint_map"],
                                0.5,
                                params["family_bootstrap"],
                                0.0,
                                1.0,
                            )
                        ]
                    )
                )
            two = np.stack(two)
            values = two.sum(axis=1) / den.sum()
            source_risks[t.source].append(values)
            rows_by[t.source][t.name] = FamilyRisks(hashed, two.mean(axis=0), den)
            endpoints[t.name] = {
                "source": t.source,
                "rows": len(ix),
                "raw_observations": int(t.trials[ix].sum()),
                "families": len(g),
                "likelihood": t.likelihood,
                "per_seed_nll": {
                    str(s): dict(zip((*ARMS, "private", "shared"), row, strict=True))
                    for s, row in zip(SEEDS, values, strict=True)
                },
            }
            counts["held_rows"] += len(ix)
            counts["raw_observations"] += int(t.trials[ix].sum())
            packed[t.name + ".numerator"] = two
            packed[t.name + ".denominator"] = den
            packed[t.name + ".global_group_sha256"] = hashed
    names = sorted(source_risks)
    matrix = np.stack([np.mean(source_risks[s], axis=0) for s in names], axis=1)
    point = matrix.mean(axis=0)
    singles = [[s] for s in names]
    independent = [[s] for s in names if s not in LINKED]
    connected = [*independent, sorted(LINKED)]
    all_arms = (*ARMS, "private", "shared")
    per_source = {}
    for name, row in zip(names, point, strict=True):
        x, bd = grouped_samples([name], rows_by, "corrected")
        if not np.allclose(x, row[:3], rtol=1e-13, atol=1e-13):
            raise ArithmeticError("Source NLL replay mismatch")
        ng = len({g for e in rows_by[name].values() for g in e.groups})
        per_source[name] = {
            "families": ng,
            "seed_mean_nll": dict(zip(all_arms, row, strict=True)),
            "material_intervals": dict(
                zip(
                    COMPARISONS,
                    np.quantile(bd, [0.025, 0.975], axis=0).T.tolist(),
                    strict=True,
                )
            )
            if ng >= 2
            else None,
        }
    packed_path = OUT / "material-statistics.npz"
    np.savez_compressed(packed_path, **packed)
    results = {
        "observations": counts,
        "eligible_sources": len(names),
        "eligible_endpoints": len(endpoints),
        "source_macro_nll": dict(zip(all_arms, point.mean(axis=0), strict=True)),
        "seeds": {
            str(s): summary(risk[:, :3]) for s, risk in zip(SEEDS, matrix, strict=True)
        },
        "by_source": per_source,
        "by_endpoint": endpoints,
        "equal_sources": grouped_summary(rows_by, singles, "corrected"),
        "equal_linked_components": grouped_summary(rows_by, connected, "corrected"),
        "remove_three_linked_sources": grouped_summary(
            rows_by, independent, "corrected"
        ),
        "linked_sources": sorted(LINKED),
        "frozen_forecast_sha256": digest(OUT / "forecast-freeze.json"),
        "paired_statistics_sha256": digest(packed_path),
        "geometry_sha256": digest(OUT / "geometry.json"),
    }
    write(OUT / "scores.json", results)
    for k in (
        "equal_sources",
        "equal_linked_components",
        "remove_three_linked_sources",
    ):
        print(k, results[k]["source_and_material"], flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=("fit", "calibrate", "score"))
    torch.set_num_threads(1)
    {"fit": fit, "calibrate": calibrate, "score": score}[ap.parse_args().stage]()


if __name__ == "__main__":
    main()
