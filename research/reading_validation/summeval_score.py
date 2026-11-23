"""Compare held human coherence judgments for distinct drafts of the same source news story."""

import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from research.reading_validation.coherence_order_data import read
from research.reading_validation.local_forecast import digest
from research.reading_validation.summeval_data import OUT
from research.reading_validation.summeval_forecast import judgments


PRIMARY = [
    ("adaptive", "surprisal"),
    ("essay_hazard", "essay_lexical"),
    ("shared_native", "independent_native"),
]


def result(values, family_size=1):
    arr = np.asarray(values, float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"stories": 0}
    draws = np.random.default_rng(486246).integers(len(arr), size=(10000, len(arr)))
    samples = arr[draws].mean(axis=1)
    return {
        "estimate": float(arr.mean()),
        "stories": len(arr),
        "ci95": np.quantile(samples, [0.025, 0.975]).tolist(),
        "ci95_simultaneous": np.quantile(
            samples, [0.025 / family_size, 1 - 0.025 / family_size]
        ).tolist(),
    }


def compare(difference, truth):
    s = np.where(np.isclose(difference, 0, atol=1e-10, rtol=0), 0, np.sign(difference))
    return (1 + s * np.sign(truth)) * 0.5


def ranks(outcomes, values):
    return (
        float(np.corrcoef(rankdata(outcomes), rankdata(values))[0, 1])
        if len(set(outcomes)) > 1 and len(set(values)) > 1
        else float("nan")
    )


def evaluate(target, score, groups, stories, other=None):
    reports = []
    pair_count = 0
    covered = 0
    for g in groups:
        can = (stories == g) & np.isfinite(score) & np.isfinite(target)
        if other is not None:
            can &= np.isfinite(other)
        i, j = np.triu_indices(int(can.sum()), 1)
        ys = target[can]
        ps = score[can]
        valid = np.abs(ys[i] - ys[j]) > 1e-10
        i, j = i[valid], j[valid]
        if not len(i):
            continue
        test = compare(ps[i] - ps[j], ys[i] - ys[j])
        if other is not None:
            control = other[can]
            test -= compare(control[i] - control[j], ys[i] - ys[j])
        reports.append((g, float(test.mean()), ranks(ys, ps)))
        pair_count += len(i)
        covered += int(can.sum())
    return {
        "pair_order": result([x[1] for x in reports]),
        "within_story_spearman": result([x[2] for x in reports]),
        "graded_pairs": pair_count,
        "distinct_eligible_drafts": covered,
    }, {g: v for g, v, _ in reports}


def paired(target, p, q, groups, stories, size):
    report, by_story = evaluate(target, p, groups, stories, q)
    return {
        **result(list(by_story.values()), size),
        "compared_pairs": report["graded_pairs"],
        "compared_drafts": report["distinct_eligible_drafts"],
    }


def peer(visits, geo, keys, story, methods):
    ref = {}
    for g in np.unique(story):
        rows = geo[g]["systems"]
        idx = np.flatnonzero(story == g)
        ballot = []
        for i in idx:
            sys = min(k for k, v in rows.items() if v == keys[i])
            ballot.append(np.asarray(visits[g, sys][0], float))
        i, j = np.triu_indices(len(idx), 1)
        a = np.asarray(ballot)[i]
        b = np.asarray(ballot)[j]
        signed = (
            a.sum(1)[:, None, None]
            - a[:, :, None]
            - b.sum(1)[:, None, None]
            + b[:, None, :]
        ) / 2
        withheld = a[:, :, None] - b[:, None, :]
        valid = abs(signed) > 1e-10
        p = compare(withheld, signed)
        ref[g] = {
            "included_draws": int(valid.sum()),
            "distinct_draft_pairs": int(valid.any(axis=(1, 2)).sum()),
            "held_human": float(p[valid].mean()),
        }
        for name, vals in methods.items():
            d = (vals[idx[i]] - vals[idx[j]])[:, None, None]
            ok = valid & np.isfinite(d)
            if np.any(ok):
                agreement = compare(d, signed)
                ref[g][name] = {
                    "held_human": float(p[ok].mean()),
                    "model": float(agreement[ok].mean()),
                    "draws": int(ok.sum()),
                }
    numeric = {
        "human": result([v["held_human"] for v in ref.values()]),
        "informative_withheld_draws": sum(v["included_draws"] for v in ref.values()),
    }
    for name in methods:
        s = [r[name] for r in ref.values() if name in r]
        numeric[name] = {
            "forecast": result([v["model"] for v in s]),
            "human_same_comparisons": result([v["held_human"] for v in s]),
            "forecast_minus_human": result([v["model"] - v["held_human"] for v in s]),
            "informative_draws": sum(v["draws"] for v in s),
        }
    return numeric


def run():
    seal = read(OUT / "prediction-seal.json")
    pre = read(OUT / "instrument-lock.json")
    if (
        digest(OUT / "frozen-predictions.npz") != seal["frozen"]
        or digest(OUT / "calibrated-predictions.npz") != seal["calibrated"]
    ):
        raise ValueError("Predictions changed after calibrating")
    for name, sha in {**pre["files"], **seal["code"]}.items():
        if digest(Path(__file__).with_name(name)) != sha:
            raise ValueError(f"The registered instrument changed: {name}")
    with (
        np.load(OUT / "frozen-predictions.npz", allow_pickle=False) as a,
        np.load(OUT / "calibrated-predictions.npz", allow_pickle=False) as b,
    ):
        if not np.array_equal(a["keys"], b["keys"]):
            raise ValueError("Mismatched draft geometry")
        ix = a["roles"] == "confirm"
        keys = a["keys"][ix]
        story = a["stories"][ix]
        nstops = a["eligible_essay_stops"][ix]
        methods = {
            k: v[ix]
            for k, v in a.items()
            if k not in ("keys", "roles", "stories", "eligible_essay_stops")
        }
        methods.update({k: v[ix] for k, v in b.items() if k != "keys"})
    visits = judgments("confirm")
    geo = read(OUT / "story-geometry.json")
    groups = np.unique(story)
    all_outcomes = {}
    for label, pos in [("expert", 0), ("crowd", 1)]:
        y = []
        for k, g in zip(keys, story, strict=True):
            assigned = [m for m, v in geo[g]["systems"].items() if v == k]
            y.append(np.mean([visits[g, sys][pos] for sys in assigned]))
        all_outcomes[label] = np.array(y)
    stats = {}
    deltas = {}
    for label, target in all_outcomes.items():
        stats[label] = {
            name: evaluate(target, p, groups, story)[0] for name, p in methods.items()
        }
        deltas[label] = {
            f"{aa}_minus_{bb}": paired(
                target,
                methods[aa],
                methods[bb],
                groups,
                story,
                3 if (aa, bb) in PRIMARY else 1,
            )
            for aa, bb in [
                *PRIMARY,
                ("frozen_plus_form", "form_calibrated"),
                ("both_heads_plus_form", "matched_form"),
                ("shared_native", "form_calibrated"),
                ("shared_native", "references_calibrated"),
                ("shared_native", "geval"),
                ("shared_native", "system_mean"),
                ("adaptive", "geval"),
                ("adaptive", "system_mean"),
                ("adaptive", "form_calibrated"),
            ]
        }
    crowd = all_outcomes["crowd"]
    expert = all_outcomes["expert"]
    grades = {
        "expert_mean_range": [float(expert.min()), float(expert.max())],
        "crowd_mean_range": [float(crowd.min()), float(crowd.max())],
        "expert_instrument_histogram": {
            str(i): sum(vals[0].count(i) for vals in visits.values())
            for i in range(1, 6)
        },
        "crowd_instrument_histogram": {
            str(i): sum(vals[1].count(i) for vals in visits.values())
            for i in range(1, 6)
        },
    }
    mses = {
        name: result(
            [
                float(
                    np.mean(
                        (
                            p[(story == g) & np.isfinite(p)]
                            - expert[(story == g) & np.isfinite(p)]
                        )
                        ** 2
                    )
                )
                for g in groups
                if np.any((story == g) & np.isfinite(p))
            ]
        )
        for name, p in methods.items()
        if not name.startswith("essay")
    }
    numeric = {
        "source_stories": len(groups),
        "graded_official_drafts": len(visits),
        "unique_graded_forecasts": len(keys),
        "eligible_essay_drafts": int((nstops > 0).sum()),
        "noninitial_sentence_stops": int(nstops.sum()),
        "confirmed_only_ballot_values_accessed": True,
        "future_story_outcomes_accessed": False,
        "true_ratings": grades,
        "arms": stats,
        "differences": deltas,
        "calibration_mse_1_to_5": mses,
        "crowd_as_expert_predictor": evaluate(expert, crowd, groups, story)[0],
        "held_expert": peer(
            visits,
            geo,
            keys,
            story,
            {
                k: methods[k]
                for k in [
                    "adaptive",
                    "essay_hazard",
                    "shared_native",
                    "system_mean",
                    "geval",
                ]
            },
        ),
        "forecast_seal": digest(OUT / "prediction-seal.json"),
    }
    (OUT / "scores.json").write_text(
        json.dumps(numeric, indent=2, allow_nan=False) + "\n"
    )
    np.savez_compressed(
        OUT / "confirmation-outcomes.npz",
        keys=keys,
        story=story,
        expert=expert,
        crowd=crowd,
    )
    print(
        numeric["source_stories"],
        numeric["unique_graded_forecasts"],
        {f"{a}-{b}": deltas["expert"][f"{a}_minus_{b}"] for a, b in PRIMARY},
    )


if __name__ == "__main__":
    run()
