"""Evaluate sealed predictions against repeated direct reports of lexical non-understanding."""

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from research.reading_validation.local_forecast import digest
from research.reading_validation.nns_features import selected_counts
from research.reading_validation.nns_fit import ARMS
from research.reading_validation.nns_material import LEVELS, OUT

SEED = 631402
B = 10000
METHODS = ("length", "rarity", "curriculum", "plaba", "complex", *ARMS, "adaptive")


def group_index(keys):
    out = defaultdict(list)
    for i, key in enumerate(keys):
        out[key].append(i)
    return {k: np.asarray(v, int) for k, v in out.items()}


def ci(x, draws):
    finite = draws[np.isfinite(draws)]
    return {
        "estimate": float(x) if np.isfinite(x) else None,
        "ci95": np.quantile(finite, [0.025, 0.975]).tolist()
        if len(finite) >= 0.95 * len(draws)
        else [None, None],
    }


def aggregate(values, mult):
    mask = np.isfinite(values).all(1)
    v = values[mask]
    if not v.size:
        return np.full(values.shape[1], np.nan), np.full(
            (len(mult), values.shape[1]), np.nan
        )
    w = mult[:, mask]
    d = w.sum(1)
    return v.mean(0), np.divide(
        w @ v,
        d[:, None],
        where=d[:, None] > 0,
        out=np.full((len(d), v.shape[1]), np.nan),
    )


def contrasts(center, boot, names, pairs):
    s = {k: ci(center[i], boot[:, i]) for i, k in enumerate(names)}
    d = {
        a + "_minus_" + b: ci(
            center[names.index(a)] - center[names.index(b)],
            boot[:, names.index(a)] - boot[:, names.index(b)],
        )
        for a, b in pairs
    }
    return {"scores": s, "paired_difference": d}


def ap(y, score):
    if not np.any(y):
        return np.nan
    vals = np.unique(score)[::-1]
    total = hits = 0
    out = 0.0
    for v in vals:
        selected = score == v
        h = int(y[selected].sum())
        hits += h
        total += int(selected.sum())
        out += h * hits / total
    return out / y.sum()


def pairs(i, flags, counts):
    if len(i) < 2:
        return np.empty(0, int), np.empty(0, int), np.empty(0)
    a, b = i[np.triu_indices(len(i), 1)[0]], i[np.triu_indices(len(i), 1)[1]]
    diff = flags[a].astype(int) * counts[b] - flags[b].astype(int) * counts[a]
    return a[diff != 0], b[diff != 0], np.sign(diff[diff != 0])


def rank_groups(ix, raw, flags, total, form, doc, all_fams, govern, mult):
    rank = np.full((len(all_fams), raw.shape[1]), np.nan)
    selective = rank.copy()
    precision = rank.copy()
    table = group_index(doc[ix])
    gains = defaultdict(list)
    close = defaultdict(list)
    aps = defaultdict(list)
    n = matched = 0
    for k, v in table.items():
        i = ix[v]
        a, b, d = pairs(i, flags, total)
        f = govern[k]
        if len(a):
            correct = (1 + np.sign(raw[a] - raw[b]) * d[:, None]) / 2
            gains[f].append(correct.mean(0))
            n += len(a)
            fine = abs(form[a, 2] - form[b, 2]) <= 0.25
            if fine.any():
                close[f].append(correct[fine].mean(0))
                matched += fine.sum()
        if (flags[i] > 0).any():
            aps[f].append([ap(flags[i] > 0, raw[i, j]) for j in range(raw.shape[1])])
    fc = {f: j for j, f in enumerate(all_fams)}
    for d, v in ((gains, rank), (close, selective), (aps, precision)):
        for f, arr in d.items():
            v[fc[f]] = np.asarray(arr).mean(0)
    compared = [
        ("adaptive", "lexical"),
        ("contextual", "lexical"),
        ("human_scores", "lexical"),
        ("adaptive", "human_scores"),
        ("complex", "rarity"),
        ("plaba", "rarity"),
        ("complex", "length"),
        ("plaba", "curriculum"),
    ]
    out = {}
    for name, vc in (
        ("rank", rank),
        ("frequency_matched", selective),
        ("average_precision_any", precision),
    ):
        p, b = aggregate(vc, mult)
        out[name] = {
            **contrasts(p, b, METHODS, compared),
            "families": int(np.isfinite(vc[:, 0]).sum()),
        }
    out["rank"]["pairs"] = int(n)
    out["frequency_matched"]["pairs"] = int(matched)
    return out


def across_context(ix, bank, raw, flags, total, famnum, mult):
    by_type = group_index(bank["type"][ix])
    chunks = []
    for v in by_type.values():
        i = ix[v]
        a, b, d = pairs(i, flags, total)
        keep = famnum[a] != famnum[b]
        if keep.any():
            chunks.append((a[keep], b[keep], d[keep]))
    if not chunks:
        return {"pairs": 0}
    points = []
    numerator = np.zeros((len(mult), raw.shape[1]))
    den = np.zeros(len(mult))
    for a, b, d in chunks:
        scores = (1 + np.sign(raw[a] - raw[b]) * d[:, None]) / 2
        w = mult[:, famnum[a]] * mult[:, famnum[b]]
        s = w.sum(1)
        present = s > 0
        numerator[present] += (w[present] @ scores) / s[present, None]
        den += present
        points.append(scores.mean(0))
    draws = np.divide(
        numerator,
        den[:, None],
        where=den[:, None] > 0,
        out=np.full_like(numerator, np.nan),
    )
    return {
        **contrasts(
            np.mean(points, axis=0),
            draws,
            METHODS,
            [("adaptive", "lexical"), ("contextual", "lexical"), ("complex", "rarity")],
        ),
        "pairs": sum(len(c[0]) for c in chunks),
        "types": len(chunks),
        "qualification": "Different readers may contribute across contexts.",
    }


def open_word_check(bank, raw, flags, counts, families, mult):
    all_docs = group_index(bank["doc"])
    grouped = defaultdict(list)
    bounds = defaultdict(list)
    for at in all_docs.values():
        family = bank["family"][at[0]]
        judged = counts[at] > 0
        if not judged.any() or family not in families:
            continue
        fraction = np.divide(flags[at], counts[at], out=np.zeros(len(at)), where=judged)
        if len(np.unique(counts[at][judged])) != 1:
            raise ValueError(
                "Word marginals within this text have different denominators"
            )
        bounds[family].append([fraction.max(), min(1, fraction.sum())])
        v = raw[at]
        winners = v == v.max(0)
        masses = winners / winners.sum(0)
        coverage = judged @ masses
        lower = fraction @ masses
        any_mark = (flags[at] > 0) @ masses
        grouped[family].append(
            np.stack([coverage, lower, lower + 1 - coverage, any_mark])
        )
    table = np.stack([np.mean(grouped[f], axis=0) for f in families])
    labels = (
        "judgment_coverage",
        "flag_rate_lower",
        "flag_rate_upper",
        "at_least_one_flag_lower",
    )
    out = {}
    for i, label in enumerate(labels):
        mean, boot = aggregate(table[:, i, :], mult)
        out[label] = {k: ci(mean[j], boot[:, j]) for j, k in enumerate(METHODS)}
    positive = np.stack([np.mean(bounds[f], axis=0) for f in families])
    mean, boot = aggregate(positive, mult)
    out["sentence_any_flag_marginal_bounds"] = {
        name: ci(mean[j], boot[:, j]) for j, name in enumerate(("lower", "upper"))
    }
    ok = table[:, 0, :].mean(0)
    any_mass = table[:, 3, :].mean(0)
    out["any_flag_among_judged_top_choices"] = {
        name: float(any_mass[i] / ok[i]) if ok[i] > 0 else None
        for i, name in enumerate(METHODS)
    }
    return out


def run():
    seal = json.loads((OUT / "forecast-seal.json").read_bytes())
    p = OUT / "word-forecasts.npz"
    if digest(p) != seal["outcome_free_forecast_sha256"]:
        raise ValueError("Unsealed dense probabilities")
    with np.load(p, allow_pickle=False) as z:
        bank = {k: z[k] for k in z.files}
    flags, num, covered = selected_counts("dense", bank)
    y, n = flags.sum(1), num.sum(1)
    ix = np.flatnonzero(n > 0)
    doc = bank["doc"]
    fam = bank["family"]
    all_fams = sorted(set(fam[ix]))
    lk = {f: i for i, f in enumerate(all_fams)}
    famnum = np.asarray([lk.get(f, -1) for f in fam])
    govern = dict(zip(doc, fam))
    mult = np.random.default_rng(SEED).multinomial(
        len(all_fams), np.ones(len(all_fams)) / len(all_fams), size=B
    )
    form = bank["form"]
    raw = np.column_stack(
        [
            form[:, 0],
            -form[:, 2],
            np.where(form[:, 7] > 0, 7, form[:, 6]),
            bank["source"][:, 1],
            bank["source"][:, 0],
            *[bank["text." + k][:, 0] for k in (*ARMS, "adaptive")],
        ]
    )
    result = rank_groups(ix, raw, y, n, form, doc, all_fams, govern, mult)
    result["same_printed_word"] = across_context(ix, bank, raw, y, n, famnum, mult)
    result["open_word"] = open_word_check(bank, raw, y, n, all_fams, mult)
    counts = Counter()
    doc_groups = group_index(doc[ix])
    proper_names = [s + "." + m for s in ("text", "cefr") for m in (*ARMS, "adaptive")]
    per = {name: defaultdict(list) for name in ("logloss", "brier")}
    human = defaultdict(list)
    difficult = y[ix]
    N = n[ix]
    for key, indices in doc_groups.items():
        k = ix[indices]
        family = govern[key]
        logloss = []
        brier = []
        for name in proper_names:
            pp = np.clip(bank[name][k], 1e-10, 1 - 1e-10)
            logloss.append(
                np.mean(
                    (-flags[k] * np.log(pp) - (num[k] - flags[k]) * np.log1p(-pp)).sum(
                        1
                    )
                    / n[k]
                )
            )
            brier.append(
                np.mean(
                    (flags[k] * (1 - pp) ** 2 + (num[k] - flags[k]) * pp**2).sum(1)
                    / n[k]
                )
            )
        per["logloss"][family].append(logloss)
        per["brier"][family].append(brier)
        a, b = y[k], n[k]
        dev = 2 * a * (b - a) / (b * (b - 1))
        # Leave the evaluated reader's response out of the item-majority comparison.
        pos = np.sign(2 * (a - 1) - (b - 1))
        neg = np.sign((b - 1) - 2 * a)
        human[family].append(
            [
                dev.mean(),
                np.mean((a * (pos + 1) / 2 + (b - a) * (neg + 1) / 2) / b),
                np.mean(a / b),
                np.mean((a > 0) & (a < b)),
                0.5 * dev.mean(),
                np.mean(1 - a / b),
                np.mean(
                    (a * (1 - (a - 1) / (b - 1)) ** 2 + (b - a) * (a / (b - 1)) ** 2)
                    / b
                ),
            ]
        )
        counts["judged_words"] += len(k)
        counts["mixed_words"] += int(np.sum((a > 0) & (a < b)))
    for score, values in per.items():
        arr = np.stack([np.mean(values[f], axis=0) for f in all_fams])
        pt, bb = aggregate(arr, mult)
        comparisons = [
            ("text.lexical", "text.adaptive"),
            ("text.lexical", "text.contextual"),
            ("text.lexical", "text.human_scores"),
            ("text.adaptive", "cefr.adaptive"),
            ("cefr.lexical", "cefr.adaptive"),
        ]
        result[score] = contrasts(pt, bb, proper_names, comparisons)
    hnames = (
        "pair_disagreement",
        "leave_one_voter_out_majority_accuracy",
        "observed_flag_rate",
        "mixed_word_fraction",
        "binomial_brier_noise",
        "always_known_accuracy",
        "other_voters_fraction_leave_one_out_brier",
    )
    h, draws = aggregate(np.stack([np.mean(human[f], axis=0) for f in all_fams]), mult)
    result["rater_heterogeneity"] = {
        k: ci(h[i], draws[:, i]) for i, k in enumerate(hnames)
    }
    allbands = []
    for l, code in enumerate(LEVELS):
        enough = np.flatnonzero(num[:, l] >= 2)
        dd = flags[enough, l]
        nn = num[enough, l]
        allbands.append(
            {
                "level": code,
                "annotations": int(num[:, l].sum()),
                "flagged": int(flags[:, l].sum()),
                "words_with_two_or_more": len(enough),
                "within_level_pair_disagreement": float(
                    np.mean(2 * dd * (nn - dd) / (nn * (nn - 1)))
                )
                if len(enough)
                else None,
            }
        )
    result["by_proficiency"] = allbands
    for label, mask in (
        ("known_stem", bank["seen_stem"][ix]),
        ("unseen_stem", ~bank["seen_stem"][ix]),
        ("zipf_below_three", form[ix, 2] < 3),
    ):
        result[label] = rank_groups(
            ix[mask], raw, y, n, form, doc, all_fams, govern, mult
        )["rank"]
    pwin = result["rank"]["paired_difference"]["adaptive_minus_lexical"]["ci95"][0] > 0
    calibration = (
        result["logloss"]["paired_difference"]["text.lexical_minus_text.adaptive"][
            "ci95"
        ][0]
        > 0
    )
    q = {
        "reported": result,
        "coverage": covered,
        "word_counts": dict(counts),
        "human_answers": int(N.sum()),
        "unknown_word_flags": int(difficult.sum()),
        "families": len(all_fams),
        "sentences": len(doc_groups),
        "primary_discrimination_passed": bool(pwin),
        "primary_discrimination_and_calibration": bool(pwin and calibration),
        "score_code": digest(Path(__file__)),
        "forecast_seal": digest(OUT / "forecast-seal.json"),
        "bootstrap": B,
    }
    (OUT / "scores.json").write_text(json.dumps(q, indent=2, allow_nan=False) + "\n")
    print(
        q["primary_discrimination_passed"],
        result["rank"]["paired_difference"]["adaptive_minus_lexical"],
    )


if __name__ == "__main__":
    run()
