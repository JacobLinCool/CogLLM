"""Evaluate large complete-spelling disagreement marks on real Provo prefix pairs."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np

from research.backbone_transfer.extract import digest, read_json, stream_rows
from research.reading_validation.cloze_population_data import normalized

ROOT = Path(".cache/cogllm/reading-validation/cloze-population")
SOURCE = ROOT.parent / "provo-predictability.csv"
OUT = Path("research/results/reading-validation-word-marks")
THRESHOLD = 0.2
ARMS = ("context_certified", "context_grouped", "raw_certified", "raw_grouped")


def original_words():
    measures = defaultdict(Counter)
    totals = defaultdict(set)
    blanks = Counter()
    with SOURCE.open(encoding="mac_roman") as stream:
        for r in csv.DictReader(stream):
            i = r["Word_Unique_ID"]
            form = normalized(r["Response"])
            n = int(r["Response_Count"])
            if form:
                measures[i][form] += n
            else:
                blanks[i] += n
            totals[i].add(int(r["Total_Response_Count"]))
    result = {}
    for i, c in measures.items():
        n = sum(c.values())
        if len(totals[i]) != 1:
            raise ValueError("Conflicting reported response denominator")
        if n < 2:
            continue
        d = 1 - Fraction(sum(x * (x - 1) for x in c.values()), n * (n - 1))
        result[i] = (d, n, n + blanks[i] == next(iter(totals[i])))
    return result


def bounds(d, r):
    if not 0 <= d <= 1 or not 0 <= r <= 1 or d + r * r > 1 + 1e-8:
        raise ValueError("Model group disagreement or residual mass is impossible")
    return d, min(1.0, d + r * r)


def classify(d0, r0, d1, r1):
    lo0, hi0 = bounds(d0, r0)
    lo1, hi1 = bounds(d1, r1)
    lower, upper = lo1 - hi0, hi1 - lo0
    step = d1 - d0
    point = int(step >= THRESHOLD) - int(step <= -THRESHOLD)
    mark = int(lower >= THRESHOLD) - int(upper <= -THRESHOLD)
    spans = lo1 <= hi0 and lo0 <= hi1
    if lower > upper + 1e-12 or (mark and mark != point) or (spans and mark):
        raise ValueError("Impossible positive certificate or overlap")
    return mark, point, lower, upper, spans


def check_identity():
    largest = 0.0
    for r in (0.0, 0.05, 0.27, 0.8, 0.96):
        named = np.array([0.3, 0.7]) * (1 - r)
        d = 1 - (named @ named + r * r)
        l, h = bounds(d, r)
        for k in (1, 2, 8, 10000):
            other = np.full(k, r / k)
            full = 1 - np.r_[named, other] @ np.r_[named, other]
            expected = h - r * r / k
            largest = max(largest, abs(full - expected))
            if not l - 1e-12 <= full <= h + 1e-12 or abs(full - expected) > 1e-12:
                raise ArithmeticError(
                    "Explicit spelling distributions violate certified bounds"
                )
    return {
        "max_explicit_distribution_error": largest,
        "residual_split_counts": [1, 2, 8, 10000],
        "largest_verified_residual": 0.96,
    }


def observe():
    official = original_words()
    text = {r["id"]: r["text"] for r in stream_rows(ROOT / "semantic-stimuli.jsonl")}
    native = {
        r["id"]: r for r in stream_rows(ROOT / "native-verified-word-surprisal.jsonl")
    }
    raw = list(stream_rows(ROOT / "external-scored-word-rows.jsonl"))
    full = {}
    exact_error = 0.0
    count = 0
    for d in raw:
        q, n, ok = official[d["id"]]
        if (
            bool(d["normative_denominator_verified"]) != ok
            or d["released_nonempty_response_mass"] != n
        ):
            raise ValueError("Released word denominator failed original reconstruction")
        exact_error = max(exact_error, abs(float(q) - d["human_spelling_disagreement"]))
        if ok:
            if (
                not d["adjacent_tokens"]
                or native[d["id"]]["target_word"] != d["target_ordinal_zero_based"]
            ):
                raise ValueError(
                    "Verified norming row failed its native token alignment"
                )
            k = (d["entity"], d["target_ordinal_zero_based"])
            if k in full:
                raise ValueError("Duplicate original prefix")
            full[k] = d
            count += n
    if exact_error > 5e-14:
        raise ArithmeticError(
            "Exact spelling U-statistic disagrees with original counts"
        )
    pairs = []
    wrong_space = 0
    overlap = defaultdict(int)
    relative = Counter()
    delta_err = 0
    for (paragraph, idx), d0 in sorted(full.items()):
        d1 = full.get((paragraph, idx + 1))
        if d1 is None:
            continue
        if (
            d1["preceding_chars"] != d0["target_chars"]
            or d1["preceding_print"] != d0["shown_target"]
            or d1["Text_ID"] != d0["Text_ID"]
        ):
            raise ValueError(
                "Two adjacent target sites do not refer to the same consumed form"
            )
        s, e = d0["target_chars"]
        if text[paragraph][s:e] != d0["shown_target"]:
            raise ValueError("Original target character offsets differ")
        if text[paragraph][d0["preceding_chars"][1] : s] != " ":
            wrong_space += 1
            continue
        eq = official[d1["id"]][0] - official[d0["id"]][0]
        y = int(eq > 0) - int(eq < 0)
        hard = int(eq >= Fraction(1, 5)) - int(eq <= -Fraction(1, 5))
        relative["exact_human_threshold_ties"] += int(abs(eq) == Fraction(1, 5))
        rec = {
            "paragraph": d0["Text_ID"],
            "word_ordinal": idx,
            "start": s,
            "end": e,
            "id_before": d0["id"],
            "id_after": d1["id"],
            "human_change": float(eq),
            "human_sign": y,
            "human_large_sign": hard,
        }
        for arm in ("context", "raw"):
            g0, g1 = (
                d0["model_category_disagreement"][arm],
                d1["model_category_disagreement"][arm],
            )
            r0, r1 = d0["other_mass"][arm], d1["other_mass"][arm]
            fit = classify(g0, r0, g1, r1)
            if arm == "context":
                a, b = native[d0["id"]], native[d1["id"]]
                live = classify(
                    a["before_disagreement"],
                    a["before_other"],
                    b["before_disagreement"],
                    b["before_other"],
                )
                delta_err = max(delta_err, abs(live[2] - fit[2]), abs(live[3] - fit[3]))
                relative["native_changes_in_certified_label"] += int(live[0] != fit[0])
                relative["native_changes_in_grouped_label"] += int(live[1] != fit[1])
                fit = live
                g0, r0, g1, r1 = (
                    a["before_disagreement"],
                    a["before_other"],
                    b["before_disagreement"],
                    b["before_other"],
                )
            mark, point, lo, hi, spans = fit
            rec.update(
                {
                    arm + "_certified": mark,
                    arm + "_grouped": point,
                    arm + "_lower": lo,
                    arm + "_upper": hi,
                    arm + "_step": g1 - g0,
                    arm + "_other_before": r0,
                    arm + "_other_after": r1,
                    arm + "_level_intervals_overlap": spans,
                }
            )
            overlap[arm] += int(spans)
        pairs.append(rec)
    return pairs, {
        "aligned_prefixes": len(raw),
        "verified_prefixes": len(full),
        "nonempty_responses_at_verified_prefixes": count,
        "verified_prefixes_in_paragraphs": len({k[0] for k in full}),
        "exact_human_probability_max_error": exact_error,
        "excluded_non_single_space_adjacent_updates": wrong_space,
        "native_vs_locked_max_update_bound_error": delta_err,
        "overlapping_adjacent_level_ranges": dict(overlap),
        **relative,
    }


def denominators(mark, y, large, side):
    if side == 0:
        called = mark != 0
        actual = large != 0
        right = called & (mark == y)
        big = called & (mark == large)
        opposite = called & (y == -mark)
        zero = called & (y == 0)
    else:
        called = mark == side
        actual = large == side
        right = called & (y == side)
        big = called & (large == side)
        opposite = called & (y == -side)
        zero = called & (y == 0)
    return dict(
        eligible=np.ones(len(y), dtype=bool),
        marked=called,
        observed_large=actual,
        right_sign=right,
        right_large=big,
        opposite_sign=opposite,
        zero_change=zero,
        unmarked_true_large=actual & ~big,
        wrong_size=right & ~big,
    )


def evaluate(pairs):
    q = np.array([r["human_sign"] for r in pairs])
    z = np.array([r["human_large_sign"] for r in pairs])
    ids, group = np.unique([r["paragraph"] for r in pairs], return_inverse=True)
    n = len(ids)
    w = np.random.default_rng(853749).multinomial(n, np.full(n, 1 / n), size=10000)
    W = np.vstack([np.ones(n), w])
    ratios = {
        "coverage": ("marked", "eligible"),
        "directional_precision": ("right_sign", "marked"),
        "large_same_sign_precision": ("right_large", "marked"),
        "large_same_sign_recall": ("right_large", "observed_large"),
    }
    out = {}
    dl = {}
    for arm in ARMS:
        vals = {}
        marks = np.array([r[arm] for r in pairs])
        for name, side in (("both", 0), ("increase", 1), ("decrease", -1)):
            tests = denominators(marks, q, z, side)
            hist = {
                k: np.bincount(group, weights=a, minlength=n) for k, a in tests.items()
            }
            counts = {k: W @ a for k, a in hist.items()}
            res = {"counts": {k: int(v[0]) for k, v in counts.items()}, "ratios": {}}
            for metric, (numer, denom) in ratios.items():
                a, b = counts[numer], counts[denom]
                s = np.divide(a, b, out=np.full(len(a), np.nan), where=b > 0)
                sample = s[1:][np.isfinite(s[1:])]
                res["ratios"][metric] = {
                    "estimate": float(s[0]) if np.isfinite(s[0]) else None,
                    "paragraph_ci95": np.quantile(sample, [0.025, 0.975]).tolist()
                    if len(sample)
                    else None,
                    "valid_draws": len(sample),
                }
                dl[arm, name, metric] = s
            vals[name] = res
        vals["abstentions"] = int(np.sum(marks == 0))
        out[arm] = vals
    contrasts = {}
    for comp in ("context_grouped", "raw_certified"):
        v = {}
        for sub in ("both", "increase", "decrease"):
            hh = {}
            for metric in ratios:
                a = dl["context_certified", sub, metric] - dl[comp, sub, metric]
                ok = np.isfinite(a[1:])
                hh[metric] = {
                    "difference": float(a[0]) if np.isfinite(a[0]) else None,
                    "paragraph_ci95": np.quantile(a[1:][ok], [0.025, 0.975]).tolist()
                    if ok.any()
                    else None,
                    "valid_draws": int(ok.sum()),
                }
            v[sub] = hh
        contrasts["context_certified_minus_" + comp] = v
    return {
        "paragraphs": n,
        "adjacent_words": len(q),
        "observed_rises": int((q > 0).sum()),
        "observed_falls": int((q < 0).sum()),
        "observed_zeros": int((q == 0).sum()),
        "large_spelling_rises": int((z > 0).sum()),
        "large_spelling_falls": int((z < 0).sum()),
        "both_adjacent_decisions_share_one_prefix": True,
        "arms": out,
        "paired": contrasts,
    }


def main():
    lock = read_json(OUT / "forecast-lock.json")
    if any(digest(Path(k)) != v for k, v in lock["files"].items()):
        raise ValueError("A preregistered file changed")
    pairs, metadata = observe()
    if len({(r["paragraph"], r["word_ordinal"]) for r in pairs}) != len(pairs):
        raise ValueError("Duplicate adjacent comparison")
    results = evaluate(pairs)
    packed = OUT / "word-updates.csv"
    with packed.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(pairs[0]))
        writer.writeheader()
        writer.writerows(pairs)
    results.update(
        {
            "design_sha256": digest(OUT / "forecast-lock.json"),
            "estimator_sha256": digest(Path(__file__)),
            "word_outcomes_sha256": digest(packed),
            "geometry_and_parity": metadata,
            "distribution_identity": check_identity(),
            "resampling": "10,000 paired whole paragraphs",
        }
    )
    (OUT / "scores.json").write_text(
        json.dumps(results, indent=2, allow_nan=False) + "\n"
    )
    for arm, rec in results["arms"].items():
        print(
            arm,
            rec["increase"]["counts"]["marked"],
            rec["decrease"]["counts"]["marked"],
            rec["abstentions"],
            rec["both"]["ratios"],
            flush=True,
        )


if __name__ == "__main__":
    main()
