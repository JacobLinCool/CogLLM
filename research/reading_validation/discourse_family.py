"""Recompute paired intervals at fixed simultaneous confidence limits for ten prespecified contrasts."""

import json
from pathlib import Path

import numpy as np

from research.reading_validation.coherence_order_data import OUT as COH
from research.reading_validation.coherence_order_data import read
from research.reading_validation.coherence_order_fit import source_targets
from research.reading_validation.coherence_order_genre_score import group_sample
from research.reading_validation.coherence_order_score import (
    DRAWS,
    SEED,
    english_arrays,
)
from research.reading_validation.danish_coherence_data import (
    OUT as DANISH,
)
from research.reading_validation.danish_coherence_data import (
    entries as danish_rows,
)
from research.reading_validation.danish_coherence_score import concordance, draw_domains
from research.reading_validation.italian_order_data import OUT as PAIRS
from research.reading_validation.italian_order_score import average
from research.reading_validation.local_forecast import digest

PLAN = Path(__file__).with_name("discourse-family-protocol.md")
OUT = COH / "family.json"


def report(v):
    f = np.isfinite(v[1:])
    x = v[1:][f]
    return {
        "estimate": float(v[0]),
        "ci95": np.quantile(x, [0.025, 0.975]).tolist(),
        "ci95_simultaneous": np.quantile(x, [0.0025, 0.9975]).tolist(),
        "finite_draws": int(f.sum()),
    }


def human():
    for fname in ("scores.json", "genre-scores.json"):
        if not (COH / fname).is_file():
            raise FileNotFoundError(
                "Official Italian and English tables must already be open"
            )
    truth = source_targets(held=True)
    with (
        np.load(COH / "forecast/predictions.npz", allow_pickle=False) as p,
        np.load(COH / "genre-controls.npz", allow_pickle=False) as st,
    ):
        k = p["italian.keys"]
        y = np.array([truth["italian"][j] for j in k])
        provenance = read(COH / "genre-forecasts.json")
        group = [provenance[j]["matched_source"] or "unknown/" + j for j in k]
        mult = group_sample(group)
        a = (p["italian.adaptive"] - y) ** 2
        sel = {j: i for i, j in enumerate(st["keys"])}
        base = {
            "italian_joint": p["italian.representation"],
            "italian_context": p["italian.surprisal"],
            "italian_style": st["genre_plus_form"][[sel[j] for j in k]],
        }
        vals = {
            name: report(average((v - y) ** 2 - a, mult)) for name, v in base.items()
        }
        k = p["english.keys"]
        s, n = np.array([truth["english"][j] for j in k]).T
        for study in ("cohesentia", "decor"):
            q = (n > 0) & (p["english.study"] == study)
            grp = p["english.group"][q]
            ng = len(np.unique(grp))
            count = np.random.default_rng(SEED).multinomial(
                ng, np.ones(ng) / ng, size=DRAWS
            )
            est = {}
            for arm in ("adaptive", "representation"):
                m = english_arrays(s[q], n[q], p["english." + arm][q], grp, count)[
                    "logloss"
                ]
                est[arm] = np.r_[m[0], m[1]]
            vals[study + "_joint"] = report(est["representation"] - est["adaptive"])
    return vals


def external():
    if not (DANISH / "scores.json").is_file():
        raise FileNotFoundError("Danish original scores must already be open")
    truth = {r["id"]: r["rating"] for r in danish_rows("test", labels=True)}
    with np.load(DANISH / "forecasts.npz", allow_pickle=False) as p:
        y = np.array([truth[k] for k in p["keys"]])
        blocks = draw_domains(p["group"], p["domain"])
        avg = {
            k: np.mean(
                [concordance(y[ix], p[k][ix], w) for ix, w in blocks.values()], axis=0
            )
            for k in ("adaptive", "surprisal", "danish_lexical")
        }
    return {
        "danish_transfer": report(avg["adaptive"] - 0.5),
        "danish_context": report(avg["adaptive"] - avg["surprisal"]),
        "danish_target_lexical": report(avg["adaptive"] - avg["danish_lexical"]),
    }


def structural():
    if not (PAIRS / "structural-scores.json").is_file():
        raise FileNotFoundError("The exact-word intervention must already be graded")
    data = read(PAIRS / "order-differences.json")
    result = {}
    for name, col in (
        ("italian_same_words", "canonical_minus_permuted_passage"),
        (
            "italian_first_disrupted_same_sentence",
            "permuted_minus_canonical_break_probability",
        ),
    ):
        r = [
            v
            for v in data
            if v[col] is not None
            and (col.startswith("canonical") or v["canonical_has_predecessor"])
        ]
        w = group_sample([x["group"] for x in r])
        d = np.array([x[col]["adaptive"] for x in r])
        res = np.where(d > 1e-10, 1.0, np.where(d < -1e-10, 0.0, 0.5))
        result[name] = report(average(res, w) - 0.5)
    return result


def main():
    estimates = {**human(), **external(), **structural()}
    if len(estimates) != 10:
        raise ValueError("Different multiple-testing family")
    rec = {
        "tests": estimates,
        "family_count": 10,
        "supported_unadjusted": sum(r["ci95"][0] > 0 for r in estimates.values()),
        "supported_simultaneous": sum(
            r["ci95_simultaneous"][0] > 0 for r in estimates.values()
        ),
        "design_seal": digest(PLAN),
        "draws": DRAWS,
    }
    OUT.write_text(json.dumps(rec, indent=2) + "\n")
    print(rec)


if __name__ == "__main__":
    main()
