"""Check source-specific human preference changes with fixed pairs of systems and stories."""

import json
from itertools import product
from pathlib import Path

import numpy as np

from research.reading_validation.coherence_order_data import read
from research.reading_validation.local_forecast import digest
from research.reading_validation.summeval_data import OUT
from research.reading_validation.summeval_forecast import judgments
from research.reading_validation.summeval_score import compare


def estimate(values, multiplicity, i, j, arms=1):
    ok = np.isfinite(values)
    samples = []
    for start in range(0, len(multiplicity), 256):
        m = multiplicity[start : start + 256]
        w = m[:, i[ok]] * m[:, j[ok]]
        den = w.sum(axis=1)
        valid = den > 0
        samples.extend((w[valid] @ values[ok]) / den[valid])
    boot = np.asarray(samples)
    return {
        "mean": float(values[ok].mean()),
        "story_pairs": int(ok.sum()),
        "ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
        "ci95_simultaneous": np.quantile(
            boot, [0.025 / arms, 1 - 0.025 / arms]
        ).tolist(),
    }


def table(keys, stories, geo, visits, forecasts):
    groups = sorted(set(stories))
    systems = sorted(geo[groups[0]]["systems"])
    ind = {k: i for i, k in enumerate(keys)}
    votes = np.full((len(groups), len(systems), 3), np.nan)
    crowd = np.full((len(groups), len(systems)), np.nan)
    rowid = np.full(crowd.shape, -1, int)
    mat = {name: np.full_like(crowd, np.nan) for name in forecasts}
    for i, g in enumerate(groups):
        if set(geo[g]["systems"]) != set(systems):
            raise ValueError("Each story must have the same original system roster")
        for j, m in enumerate(systems):
            k = geo[g]["systems"][m]
            idx = ind.get(k)
            if idx is None:
                continue
            rowid[i, j] = idx
            human, panel = visits[g, m]
            votes[i, j] = human
            crowd[i, j] = np.mean(panel)
            for name, p in forecasts.items():
                mat[name][i, j] = p[idx]
    mat["independent_crowd"] = crowd
    mat["fixed_system_identity"] = np.broadcast_to(
        np.arange(len(systems)), (len(groups), len(systems))
    )
    return groups, systems, rowid, votes, mat


def compute(rowid, votes, raw, *, nboot=10000):
    n, ns = rowid.shape
    sa, sb = np.triu_indices(ns, 1)
    ii, jj = np.triu_indices(n, 1)
    rating = votes.mean(-1)
    delta = rating[:, sa] - rating[:, sb]
    valid = (rowid[:, sa] != rowid[:, sb]) & (rowid[:, sa] >= 0) & (rowid[:, sb] >= 0)
    diff = {k: v[:, sa] - v[:, sb] for k, v in raw.items()}
    overall = {k: np.full(len(ii), np.nan) for k in diff}
    hold = {
        k: np.full(len(ii), np.nan)
        for k in [
            "held_human",
            *(
                "matched_" + name
                for name in ["mixture_choice", "geval", "shared_native"]
            ),
        ]
    }
    target_ct = np.zeros(len(ii), int)
    draw_count = 0
    layout_count = 0
    contributing = np.zeros(len(sa), int)
    four = np.array(list(product(range(3), repeat=4))).T
    for ix, (i, j) in enumerate(zip(ii, jj, strict=True)):
        can = valid[i] & valid[j]
        a, b = delta[i], delta[j]
        rev = can & (a * b < -1e-10)
        target_ct[ix] = rev.sum()
        contributing += rev
        truth = a - b
        for name, mat in diff.items():
            eta = mat[i] - mat[j]
            ok = rev & np.isfinite(eta)
            if ok.any():
                overall[name][ix] = compare(eta[ok], truth[ok]).mean()
        if not can.any():
            continue
        units = (
            votes[i, sa[can]],
            votes[i, sb[can]],
            votes[j, sa[can]],
            votes[j, sb[can]],
        )
        held = [x[:, f] for x, f in zip(units, four, strict=True)]
        left = [
            (x.sum(axis=1)[:, None] - u) / 2 for x, u in zip(units, held, strict=True)
        ]
        di = left[0] - left[1]
        dj = left[2] - left[3]
        cross = di * dj < -1e-10
        each = cross.sum(axis=1)
        use = each > 0
        if not use.any():
            continue
        draw_count += int(each.sum())
        layout_count += int(use.sum())
        hd = held[0] - held[1] - held[2] + held[3]
        dec = compare(hd, di - dj)
        hold["held_human"][ix] = np.mean((dec * cross).sum(axis=1)[use] / each[use])
        for name in ("mixture_choice", "geval", "shared_native"):
            model = (diff[name][i, can] - diff[name][j, can])[:, None]
            ok = use & np.isfinite(model[:, 0])
            if ok.any():
                hold["matched_" + name][ix] = np.mean(
                    (compare(model, di - dj) * cross).sum(axis=1)[ok] / each[ok]
                )
    rng = np.random.default_rng(47935)
    m = rng.multinomial(n, np.full(n, 1 / n), size=nboot).astype(float)
    methods = {k: estimate(v, m, ii, jj) for k, v in overall.items()}
    peer = {k: estimate(v, m, ii, jj) for k, v in hold.items()}
    tests = {
        "adaptive_over_system_only_chance": estimate(
            overall["mixture_choice"] - 0.5, m, ii, jj, 2
        ),
        "adaptive_over_form_probability": estimate(
            overall["mixture_choice"] - overall["form_choice"], m, ii, jj, 2
        ),
    }
    return {
        "source_system_pairs": len(sa),
        "contributing_system_pairs": int((contributing > 0).sum()),
        "source_story_pairs": len(ii),
        "pairs_with_reversed_order": int((target_ct > 0).sum()),
        "observed_preference_reversals": int(target_ct.sum()),
        "weighted_draws_for_reference": draw_count,
        "layouts_for_human_reference": layout_count,
        "actual_grades": methods,
        "registered_tests": tests,
        "matched_held_expert": peer,
    }


def main():
    seal = read(OUT / "crossover-lock.json")
    here = Path(__file__)
    if (
        digest(here) != seal["code"]
        or digest(here.with_name("summeval-crossover.md")) != seal["protocol"]
    ):
        raise ValueError("Changed registered crossover instrument")
    original = read(OUT / "prediction-seal.json")
    study = read(OUT / "ballot-forecast-seal.json")
    for f, v in [
        ("frozen-predictions.npz", original["frozen"]),
        ("calibrated-predictions.npz", original["calibrated"]),
        ("ballot-forecasts.npz", study["forecast_sha256"]),
    ]:
        if digest(OUT / f) != v:
            raise ValueError("Frozen outcomes changed")
    with (
        np.load(OUT / "frozen-predictions.npz", allow_pickle=False) as a,
        np.load(OUT / "calibrated-predictions.npz", allow_pickle=False) as b,
        np.load(OUT / "ballot-forecasts.npz", allow_pickle=False) as c,
    ):
        if not np.array_equal(a["keys"], b["keys"]) or not np.array_equal(
            a["keys"], c["keys"]
        ):
            raise ValueError("Different drafts")
        ok = a["roles"] == "confirm"
        keys = a["keys"][ok]
        stories = a["stories"][ok]
        methods = {
            k: s[k][ok]
            for s, ns in [
                (a, ["adaptive", "essay_hazard", "surprisal", "geval"]),
                (b, ["shared_native", "system_mean"]),
                (c, ["form_choice", "native_choice", "mixture_choice"]),
            ]
            for k in ns
        }
    groups, systems, ids, votes, raw = table(
        keys, stories, read(OUT / "story-geometry.json"), judgments("confirm"), methods
    )
    out = compute(ids, votes, raw)
    out.update(
        {
            "stories": len(groups),
            "systems": len(systems),
            "instrument": digest(OUT / "crossover-lock.json"),
        }
    )
    (OUT / "crossover-scores.json").write_text(
        json.dumps(out, indent=2, allow_nan=False) + "\n"
    )
    print(out["registered_tests"])


if __name__ == "__main__":
    main()
