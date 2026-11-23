"""Refit the two previously published readers with a third, previously unassessed material split."""

import argparse
from collections import defaultdict
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch

from research.backbone_transfer.extract import digest, stream_rows
from research.reading_validation.borrowing_data import (
    CALIBRATE,
    EVALUATE,
    FIT,
    TUNE,
    PACKET,
    data,
)
from research.reading_validation.borrowing_fit import Private, train
from research.reading_validation.borrowing_prior import read, write, SEEDS
from research.reading_validation.source_directions import fresh
from research.union_training.fit_observations import (
    COUNT_MODELS,
    Readouts,
    observation_scales,
    task_responses,
)

OUT = Path("research/results/reading-validation-borrowing-replication")
PLAN = Path(__file__).with_name("borrowing-replication-protocol.md")


def salted(stage, s, g):
    return sha256(
        ("cogllm-borrowing-third-" + stage + "-v1/" + s + "/" + str(g)).encode()
    ).hexdigest()


def states():
    cache, tasks, historical = data()
    other = fresh(tasks)
    src = defaultdict(dict)
    for t in tasks:
        for idx, g in enumerate(t.family):
            v = int(historical[t.name][idx]), int(other[t.name][idx])
            old = src[t.source].setdefault(g, v)
            if old != v:
                raise ValueError(
                    "One material family changed role across its measured responses"
                )
    sets = {}
    eligible = {}
    safeguards = []
    for s, groups in src.items():
        options = sorted(
            (g for g, v in groups.items() if v == (FIT, FIT)),
            key=lambda g: salted("holdout", s, g),
        )
        n = min(max(1, round(0.25 * len(options))), len(options), len(groups) - 1)
        held = set(options[:n])
        rest = sorted(set(groups) - held, key=lambda g: salted("inner", s, g))
        k = min(max(1, round(0.15 * len(rest))), (len(rest) - 1) // 2)
        for i, g in enumerate(rest):
            sets[s, g] = TUNE if i < k else CALIBRATE if i < 2 * k else FIT
        for g in held:
            sets[s, g] = EVALUATE
        eligible[s] = len(options)
    for t in sorted(tasks, key=lambda t: t.name):
        available = sorted(
            set(t.family), key=lambda g: salted("holdout", t.source, g), reverse=True
        )
        if not any(sets[t.source, k] == FIT for k in available):
            k = next(
                (g for g in available if sets[t.source, g] == EVALUATE), available[0]
            )
            safeguards.append(
                {
                    "endpoint": t.name,
                    "original_role": int(sets[t.source, k]),
                    "family_hash": salted("public", t.source, k),
                }
            )
            sets[t.source, k] = FIT
    actual = {}
    docs = {}
    all_counts = {}
    for t in tasks:
        roles = np.array([sets[t.source, g] for g in t.family])
        actual[t.name] = roles
        ix = roles == EVALUATE
        if np.any((historical[t.name][ix] != FIT) | (other[t.name][ix] != FIT)):
            raise ArithmeticError(
                "A previously tuned, calibrated or evaluated family entered replication"
            )
        records = list(stream_rows(PACKET / (t.name + ".jsonl")))
        for r, part in zip(records, roles, strict=True):
            for doc, _ in r["coordinates"]:
                val = (t.source, r["family"], int(part))
                if docs.setdefault(doc, val) != val:
                    raise ValueError(
                        "One document has multiple source families or roles"
                    )
        if t.likelihood not in COUNT_MODELS:
            values = task_responses(t.likelihood, records, t.coordinates.shape[1])
            t.trials, t.center, t.spread = observation_scales(
                t.likelihood, values, roles
            )
            t.response = torch.as_tensor(values, dtype=torch.float32)
        all_counts[t.name] = {
            "source": t.source,
            "rows": np.bincount(roles, minlength=4).tolist(),
            "family_counts": [len(set(t.family[roles == p])) for p in range(4)],
        }
    geom = {
        "prior_folds_common_fit_family_counts": eligible,
        "assignments": {
            s: {
                salted("public", s, k): v
                for (sink, k), v in sorted(sets.items())
                if sink == s
            }
            for s in sorted(src)
        },
        "rows_and_groups_by_endpoint": all_counts,
        "geometry_safeguards": safeguards,
        "document_exposures": len(docs),
        "held_was_fit_in_both_earlier_models": True,
        "newly_fitted_expert_exposure_to_held_families": False,
        "protocol": digest(PLAN),
        "archive": digest(PACKET / "receipt.json"),
        "code": digest(Path(__file__)),
    }
    return cache, tasks, actual, geom


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("--train", action="store_true")
    args = a.parse_args()
    torch.set_num_threads(1)
    cache, tasks, g, geom = states()
    OUT.mkdir(parents=True, exist_ok=True)
    file = OUT / "material-freeze.json"
    if not file.exists():
        write(file, geom)
    elif read(file) != geom:
        raise ValueError("Third-batch geometry changed")
    if not args.train:
        print(
            sum(r["rows"][3] for r in geom["rows_and_groups_by_endpoint"].values()),
            "held observations",
            sum(r["rows"][3] > 0 for r in geom["rows_and_groups_by_endpoint"].values()),
            "observed endpoints",
            flush=True,
        )
        return
    for seed in SEEDS:
        torch.manual_seed(seed)
        p = OUT / f"seed-{seed}"
        p.mkdir(exist_ok=True)
        base = Readouts(cache.shape[1], tasks)
        private = Private(base, tasks)
        with torch.no_grad():
            for t in tasks:
                ids = torch.as_tensor(np.flatnonzero(g[t.name] == FIT)[:3])
                if not torch.equal(base(t, cache, ids), private(t, cache, ids)):
                    raise ValueError("Initial experts differ")
        reports = {}
        for label, net in [("shared", base), ("private", private)]:
            reports[label] = train(net, label, tasks, cache, g, seed, p)
        if not all(v["stop"] == "inner_likelihood_plateau" for v in reports.values()):
            raise ArithmeticError(
                "A fresh fitting run exhausted its convergence budget"
            )
        members = {
            f.name: digest(f)
            for f in sorted(p.glob("*.npz"))
            if f.name.startswith(("shared-", "private-"))
        }
        write(
            p / "fit.json",
            {
                "seed": seed,
                "geometry": digest(file),
                "cache_shape": list(cache.shape),
                "roles": {
                    t.name: np.bincount(g[t.name], minlength=4).tolist() for t in tasks
                },
                "fits": reports,
                "files": members,
            },
        )
        print("Finished blinded experts", seed, flush=True)


if __name__ == "__main__":
    main()
