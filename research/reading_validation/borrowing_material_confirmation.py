"""Grade fixed observation-specific borrowing rules on previously ungraded global parent families."""

from __future__ import annotations

import argparse
from collections import defaultdict
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
    data,
)
from research.reading_validation.borrowing_family_audit import ARMS, FamilyRisks
from research.reading_validation.borrowing_family_bootstrap import estimate, summary
from research.reading_validation.borrowing_fit import Private, train
from research.reading_validation.borrowing_global_families import (
    LINKED,
    QUARANTINE,
    grouped_samples,
    grouped_summary,
)
from research.reading_validation.borrowing_global_families import (
    load as global_third,
)
from research.reading_validation.borrowing_prior import SEEDS, read, write
from research.reading_validation.borrowing_score import GeometricPool, material_weights
from research.reading_validation.source_directions import fresh
from research.union_training.fit_observations import (
    COUNT_MODELS,
    Readouts,
    observation_scales,
    task_responses,
    weights,
)

OUT = Path("research/results/reading-validation-family-bootstrap/confirmation")
PLAN = Path(__file__).with_name("borrowing-material-confirmation-protocol.md")


def order(family):
    return sha256(
        ("cogllm-whole-material-confirmation-v1/" + family).encode()
    ).hexdigest()


def geometry():
    _, old_tasks, p1 = data()
    p2 = fresh(old_tasks)
    del old_tasks
    h, tasks, p3, previous = global_third()
    eligibility = {}
    fit_by_source = defaultdict(set)
    all_names = defaultdict(set)
    for t in tasks:
        for family, a, b, c in zip(
            t.family, p1[t.name], p2[t.name], p3[t.name], strict=True
        ):
            eligibility[family] = eligibility.get(family, True) and a == b == c == FIT
            all_names[t.source].add(family)
            if c == FIT:
                fit_by_source[t.source].add(family)
    candidates = {}
    held = set()
    for s, groups in sorted(all_names.items()):
        eligible = sorted((g for g in groups if eligibility[g]), key=order)
        count = min(
            len(eligible),
            max(1, round(len(eligible) / 3)),
            max(0, len(fit_by_source[s]) - 1),
        )
        candidates[s] = len(eligible)
        held.update(eligible[:count])
    safeguards = []
    for t in sorted(tasks, key=lambda t: t.name):
        fit = set(t.family[p3[t.name] == FIT])
        if not (fit - held):
            restore = min(fit, key=order)
            held.remove(restore)
            safeguards.append({"endpoint": t.name, "restored": order(restore)})
    if any(not eligibility[g] for g in held):
        raise ValueError("A previous grade or calibration entered the confirmation")
    parts = {}
    meta = {}
    active = {}
    for t in tasks:
        roles = np.array(
            [
                EVALUATE if g in held else QUARANTINE if r == EVALUATE else r
                for r, g in zip(p3[t.name], t.family, strict=True)
            ]
        )
        if not np.any(roles == FIT):
            raise ValueError("An endpoint lost all fit support")
        for g, r in zip(t.family, roles, strict=True):
            if r != QUARANTINE and active.setdefault(g, int(r)) != r:
                raise ValueError("A global family crossed two roles")
        if t.likelihood not in COUNT_MODELS:
            v = task_responses(
                t.likelihood,
                list(stream_rows(PACKET / (t.name + ".jsonl"))),
                t.coordinates.shape[1],
            )
            t.trials, t.center, t.spread = observation_scales(t.likelihood, v, roles)
            t.response = torch.as_tensor(v, dtype=torch.float32)
        parts[t.name] = roles
        meta[t.name] = {
            "source": t.source,
            "rows": np.bincount(roles, minlength=5).tolist(),
            "families": [len(set(t.family[roles == i])) for i in range(5)],
        }
    manifest = {
        "protocol": digest(PLAN),
        "geometry_code": digest(Path(__file__)),
        "input": digest(PACKET / "receipt.json"),
        "global_previous_protocol": previous["protocol"],
        "globally_eligible_family_counts": candidates,
        "globally_held_sha256": sorted(order(g) for g in held),
        "exclusions_and_support": meta,
        "geometry_safeguards": safeguards,
        "held_role_in_all_three_previous_partitions": "fit",
        "previous_third_grades_used_for_new_fitting": False,
        "prior_three_graded_family_intersection": 0,
        "within_partition_global_role_overlap": 0,
    }
    return h, tasks, parts, manifest


def stage_fit():
    OUT.mkdir(exist_ok=True)
    h, tasks, parts, manifest = geometry()
    dst = OUT / "geometry.json"
    if dst.exists() and read(dst) != manifest:
        raise ValueError("Previously frozen confirmation geometry changed")
    write(dst, manifest)
    for seed in SEEDS:
        path = OUT / f"seed-{seed}"
        path.mkdir(exist_ok=True)
        torch.manual_seed(seed)
        common = Readouts(h.shape[1], tasks)
        source = Private(common, tasks)
        for t in tasks:
            ii = torch.from_numpy(np.flatnonzero(parts[t.name] == FIT)[:3])
            if not torch.equal(common(t, h, ii), source(t, h, ii)):
                raise ValueError("Seeds differ at initialization")
        res = {}
        for label, model in (("shared", common), ("private", source)):
            res[label] = train(model, label, tasks, h, parts, seed, path)
            if res[label]["stop"] != "inner_likelihood_plateau":
                raise ArithmeticError("Early stopping unresolved")
        write(
            path / "fit.json",
            {
                "fits": res,
                "geometry": digest(dst),
                "members": {p.name: digest(p) for p in sorted(path.glob("*.npz"))},
            },
        )
        print("completed new ungraded-article experts", seed, flush=True)


def expert_files(stack, seed, stage):
    path = OUT / f"seed-{seed}"
    rec = read(path / "fit.json")
    if rec["geometry"] != digest(OUT / "geometry.json"):
        raise ValueError("Fitted materials changed")
    arr = []
    for key in ("shared", "private"):
        p = path / f"{key}-{stage}.npz"
        if digest(p) != rec["members"][p.name]:
            raise ValueError("Forecast hash mismatch")
        arr.append(stack.enter_context(np.load(p, allow_pickle=False)))
    return arr


def stage_calibrate():
    _, tasks, parts, manifest = geometry()
    if read(OUT / "geometry.json") != manifest:
        raise ValueError("Changed confirmation geometry")
    frozen = {}
    for seed in SEEDS:
        ps = {}
        with ExitStack() as stack:
            bank = expert_files(stack, seed, "calibration")
            for t in tasks:
                ix = np.flatnonzero(parts[t.name] == CALIBRATE)
                if not len(ix):
                    ps[t.name] = {
                        "endpoint_map": 0.5,
                        "family_bootstrap": 0.5,
                        "families": 0,
                    }
                    continue
                pool = GeometricPool(t, ix, *(f[t.name] for f in bank))
                mass, _ = material_weights(t, ix)

                def nll(a, p=pool, w=mass):
                    return float(w @ p.loss(a) - np.log(a) - np.log1p(-a))

                choice = minimize_scalar(
                    nll,
                    method="bounded",
                    bounds=(1e-7, 1 - 1e-7),
                    options={"xatol": 2e-8},
                )
                if not choice.success:
                    raise ArithmeticError("Unresolved fixed MAP coefficient")
                a, diag = estimate(pool, t, ix, "confirmation")
                ps[t.name] = {
                    "endpoint_map": float(choice.x),
                    "family_bootstrap": a,
                    **diag,
                }
        frozen[str(seed)] = ps
    write(
        OUT / "forecast-freeze.json",
        {
            "geometry": digest(OUT / "geometry.json"),
            "code": digest(Path(__file__)),
            "estimator": digest(Path(estimate.__code__.co_filename)),
            "protocol": digest(PLAN),
            "seeds": frozen,
            "opened_evaluation": False,
        },
    )
    print("froze fourth material forecasts", flush=True)


def stage_score():
    _, tasks, parts, manifest = geometry()
    if read(OUT / "geometry.json") != manifest:
        raise ValueError("Changed confirmation geometry")
    f = read(OUT / "forecast-freeze.json")
    if (
        f["estimator"] != digest(Path(estimate.__code__.co_filename))
        or f["geometry"] != digest(OUT / "geometry.json")
        or f["protocol"] != digest(PLAN)
    ):
        raise ValueError("A registered forecast has changed")
    arm_names = (*ARMS, "private", "shared")
    endpoints = {}
    archive = {}
    rows_by = defaultdict(dict)
    summary_by = defaultdict(list)
    with ExitStack() as stack:
        banks = [expert_files(stack, s, "development") for s in SEEDS]
        for t in tasks:
            ix = np.flatnonzero(parts[t.name] == EVALUATE)
            if not len(ix):
                continue
            g, ids = np.unique(t.family[ix], return_inverse=True)
            names = [sha256(str(k).encode()).hexdigest() for k in g]
            mass = weights(t, ix)
            den = np.bincount(ids, weights=t.trials[ix] * mass, minlength=len(g))
            v = []
            for s, bank in zip(SEEDS, banks, strict=True):
                pool = GeometricPool(t, ix, *(x[t.name] for x in bank))
                par = f["seeds"][str(s)][t.name]
                v.append(
                    np.column_stack(
                        [
                            np.bincount(
                                ids, weights=mass * pool.loss(a), minlength=len(g)
                            )
                            for a in (
                                par["endpoint_map"],
                                0.5,
                                par["family_bootstrap"],
                                0.0,
                                1.0,
                            )
                        ]
                    )
                )
            v = np.stack(v)
            scores = v.sum(axis=1) / den.sum()
            rows_by[t.source][t.name] = FamilyRisks(names, v.mean(axis=0), den)
            summary_by[t.source].append(scores)
            endpoints[t.name] = {
                "source": t.source,
                "families": len(g),
                "rows": len(ix),
                "raw_observations": int(t.trials[ix].sum()),
                "likelihood": t.likelihood,
                "seed_nll": {
                    str(s): dict(zip(arm_names, row, strict=True))
                    for s, row in zip(SEEDS, scores, strict=True)
                },
            }
            archive[t.name + ".global_family_sha256"] = names
            archive[t.name + ".numerator"] = v
            archive[t.name + ".denominator"] = den
    names = sorted(rows_by)
    all_groups = [[s] for s in names]
    unlinked = [[s] for s in names if s not in LINKED]
    linked = sorted(set(names) & LINKED)
    blocks = [*unlinked, linked] if linked else unlinked
    cube = np.stack([np.mean(summary_by[s], axis=0) for s in names], axis=1)
    x = cube.mean(axis=0)
    out = OUT / "material-statistics.npz"
    np.savez_compressed(out, **archive)
    per_source = {}
    for s, mean in zip(names, x, strict=True):
        val, material = grouped_samples([s], rows_by, "confirmation")
        if np.max(np.abs(val - mean[:3])) > 1e-10:
            raise ArithmeticError("Inconsistent grouped score")
        ng = len({g for e in rows_by[s].values() for g in e.groups})
        per_source[s] = {
            "families": ng,
            "nll": dict(zip(arm_names, mean, strict=True)),
            "paired_material_ci": np.quantile(
                material, [0.025, 0.975], axis=0
            ).T.tolist()
            if ng > 1
            else None,
        }
    result = {
        "observations": {
            "held_rows": sum(e["rows"] for e in endpoints.values()),
            "raw_observations": sum(e["raw_observations"] for e in endpoints.values()),
            "global_held_articles": len(manifest["globally_held_sha256"]),
        },
        "eligible_sources": len(names),
        "eligible_endpoints": len(endpoints),
        "excluded_sources": sorted(
            set(manifest["globally_eligible_family_counts"]) - set(names)
        ),
        "equal_linked_components": grouped_summary(rows_by, blocks, "confirmation"),
        "equal_sources": grouped_summary(rows_by, all_groups, "confirmation"),
        "source_macro_nll": dict(zip(arm_names, x.mean(axis=0), strict=True)),
        "by_source": per_source,
        "by_endpoint": endpoints,
        "seeds": {str(s): summary(p[:, :3]) for s, p in zip(SEEDS, cube, strict=True)},
        "statistics": digest(out),
        "forecast": digest(OUT / "forecast-freeze.json"),
    }
    write(OUT / "scores.json", result)
    print(
        "new material comparison",
        result["observations"],
        result["equal_linked_components"]["source_and_material"],
        flush=True,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("fit", "calibrate", "score"))
    torch.set_num_threads(1)
    {"fit": stage_fit, "calibrate": stage_calibrate, "score": stage_score}[
        p.parse_args().stage
    ]()


if __name__ == "__main__":
    main()
