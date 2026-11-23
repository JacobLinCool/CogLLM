"""Score order-aware training on fixed human measurements and separately on true-order sensitivity."""

import json
from collections import defaultdict
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
from research.reading_validation.danish_coherence_data import OUT as DAN
from research.reading_validation.danish_coherence_data import entries
from research.reading_validation.danish_coherence_score import concordance, draw_domains
from research.reading_validation.italian_contrast_data import OUT
from research.reading_validation.italian_order_data import OUT as PRE
from research.reading_validation.italian_order_score import average
from research.reading_validation.local_forecast import digest

ARMS = {
    "old_adaptive": "adaptive",
    "new_adaptive": "augmented_adaptive",
    "old_shared": "shared",
    "new_shared": "augmented_shared",
    "independent": "representation",
    "surprisal": "surprisal",
}


def summarize(arr):
    ok = np.isfinite(arr[1:])
    vals = arr[1:][ok]
    if not np.isfinite(arr[0]) or not len(vals):
        return {"estimate": None, "ci95": None, "ci95_four_primary": None}
    return {
        "estimate": float(arr[0]),
        "ci95": np.quantile(vals, [0.025, 0.975]).tolist(),
        "ci95_four_primary": np.quantile(vals, [0.00625, 0.99375]).tolist(),
    }


def reduce(values, sign=1.0):
    return {
        "arms": {k: summarize(v) for k, v in values.items()},
        "new_adaptive_over": {
            k: summarize(sign * (values["new_adaptive"] - v))
            for k, v in values.items()
            if k != "new_adaptive"
        },
        "new_shared_over_old_shared": summarize(
            sign * (values["new_shared"] - values["old_shared"])
        ),
    }


def read_arrays(study, base, new):
    if not np.array_equal(
        base[
            (
                "italian."
                if study == "italian"
                else "english."
                if study == "english"
                else ""
            )
            + "keys"
        ],
        new[study + ".keys"],
    ):
        raise ValueError("Held material alignment changed")
    scores = {}
    for label, field in ARMS.items():
        if label.startswith("new_"):
            scores[label] = new[study + "." + field]
        else:
            scores[label] = base[("" if study == "danish" else study + ".") + field]
    return scores


def authentic(new):
    targets = source_targets(held=True)
    out = {}
    with np.load(COH / "forecast/predictions.npz", allow_pickle=False) as old:
        ids = old["italian.keys"]
        y = np.array([targets["italian"][k] for k in ids])
        source = read(COH / "genre-forecasts.json")
        w = group_sample([source[k]["matched_source"] or "unknown/" + k for k in ids])
        pred = read_arrays("italian", old, new)
        out["italian"] = reduce(
            {k: average((v - y) ** 2, w) for k, v in pred.items()}, -1
        )
        out["italian"]["n"] = len(ids)
        s, n = np.array([targets["english"][k] for k in old["english.keys"]]).T
        pred = read_arrays("english", old, new)
        for study in ("cohesentia", "decor"):
            ix = (n > 0) & (old["english.study"] == study)
            grp = old["english.group"][ix]
            nd = len(np.unique(grp))
            draw = np.random.default_rng(SEED).multinomial(
                nd, np.ones(nd) / nd, size=DRAWS
            )
            vals = {}
            for k, v in pred.items():
                m = english_arrays(s[ix], n[ix], v[ix], grp, draw)["logloss"]
                vals[k] = np.r_[m[0], m[1]]
            out[study] = reduce(vals, -1)
            out[study]["n"] = nd
            out[study]["boundaries"] = int(ix.sum())
    with np.load(DAN / "forecasts.npz", allow_pickle=False) as old:
        labels = {r["id"]: r["rating"] for r in entries("test", labels=True)}
        y = np.array([labels[k] for k in old["keys"]])
        draw = draw_domains(old["group"], old["domain"])
        pred = read_arrays("danish", old, new)
        pred["danish_supervised_lexical"] = old["danish_lexical"]
        vals = {}
        regional = defaultdict(dict)
        for k, p in pred.items():
            for g, (ix, w) in draw.items():
                regional[g][k] = concordance(y[ix], p[ix], w)
            vals[k] = np.mean([d[k] for d in regional.values()], axis=0)
        out["danish"] = reduce(vals)
        out["danish"]["n"] = len(y)
        out["danish"]["within_domain"] = {g: reduce(d) for g, d in regional.items()}
    return out


def structural():
    old = {r["id"]: r for r in read(PRE / "order-differences.json")}
    new = {r["id"]: r for r in read(OUT / "order-differences.json")}
    if set(old) != set(new):
        raise ValueError("Changed order groups")
    controls = {k: v for k, v in ARMS.items() if k != "surprisal"}
    out = {}
    score = lambda v: np.where(v > 1e-10, 1.0, np.where(v < -1e-10, 0.0, 0.5))
    for tag, field, kind, seen in (
        ("all", "canonical_minus_permuted_passage", None, False),
        (
            "genuine_restorations",
            "canonical_minus_permuted_passage",
            "transposed",
            False,
        ),
        ("synthetic_swaps", "canonical_minus_permuted_passage", "original", False),
        ("first_break", "permuted_minus_canonical_break_probability", None, False),
        (
            "first_break_both_predecessors",
            "permuted_minus_canonical_break_probability",
            None,
            True,
        ),
    ):
        ids = [
            k
            for k, r in new.items()
            if r[field] is not None
            and old[k][field] is not None
            and (not kind or r["observed_type"] == kind)
            and (not seen or r["canonical_has_predecessor"])
        ]
        w = group_sample([new[k]["group"] for k in ids])
        values = {}
        deltas = {}
        for arm, key in controls.items():
            f = new if arm.startswith("new_") else old
            name = key.removeprefix("augmented_")
            d = np.array([f[k][field][name] for k in ids])
            values[arm] = average(score(d), w)
            deltas[arm] = average(d, w)
        out[tag] = {
            "rights": reduce(values),
            "mean_direction": reduce(deltas),
            "pairs": len(ids),
            "sources": len({new[k]["group"] for k in ids}),
        }
    return out


def run():
    receipt = read(OUT / "prediction-seal.json")
    if (
        digest(OUT / "predictions.npz") != receipt["predictions"]
        or digest(OUT / "order-differences.json")
        != receipt["paired_structural_predictions"]
        or any(
            digest(Path(__file__).with_name(k)) != v
            for k, v in receipt["study_implementation"].items()
        )
    ):
        raise ValueError("Unsealed or modified predictive method")
    if not all(
        p.is_file()
        for p in (
            COH / "scores.json",
            DAN / "scores.json",
            PRE / "structural-scores.json",
        )
    ):
        raise ValueError("The original nominal benchmark tables must be graded first")
    with np.load(OUT / "predictions.npz", allow_pickle=False) as new:
        o = authentic(new)
    s = structural()
    four = {
        **{
            k: o[k]["new_adaptive_over"]["old_adaptive"]
            for k in ("italian", "cohesentia", "danish")
        },
        "italian_order": s["all"]["rights"]["new_adaptive_over"]["old_adaptive"],
    }
    result = {
        "authentic_human_ratings": o,
        "source_order": s,
        "four_primary_contrasts": four,
        "simultaneous_advantages": sum(
            r["ci95_four_primary"][0] > 0 for r in four.values()
        ),
        "receipt": digest(OUT / "prediction-seal.json"),
    }
    (OUT / "scores.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(four)


if __name__ == "__main__":
    run()
