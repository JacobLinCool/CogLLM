"""Grade frozen joint predictions against direct lexical reports and held rewrite choices."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.special import xlogy

from research.reading_validation.local_forecast import digest
from research.reading_validation.nns_features import pack_model_input, selected_counts
from research.reading_validation.nns_rewrite_features import comparisons
from research.reading_validation.nns_rewrite_material import DATA, OUT
from research.reading_validation.nns_score import aggregate, ci, group_index, pairs

SEED = 231907
B = 10000
LEARNED = [
    a + "." + j
    for a in ("lexical", "contextual")
    for j in ("independent", "shared", "adaptive")
]
ZERO = [
    "length",
    "rarity",
    "curriculum",
    "plaba",
    "complex",
    "lexical",
    "contextual",
    "adaptive",
]


def family_matrix(values, docs, fam):
    groups = group_index(docs)
    families = sorted(set(fam))
    by = defaultdict(list)
    for idx in groups.values():
        f = fam[idx[0]]
        if not np.all(fam[idx] == f):
            raise ValueError("A context spans two material families")
        z = values[idx]
        ok = np.isfinite(z).all(1)
        if ok.any():
            by[f].append(z[ok].mean(0))
    out = np.full((len(families), values.shape[1]), np.nan)
    for i, f in enumerate(families):
        if f in by:
            out[i] = np.mean(by[f], 0)
    return out


def reported(z, doc, fam, names, contrasts, *, lower=False):
    m = family_matrix(z, doc, fam)
    n = len(m)
    w = np.random.default_rng(SEED).multinomial(n, np.ones(n) / n, size=B)
    avg, boot = aggregate(m, w)
    delta = {}
    for extended, base in contrasts:
        a = names.index(extended)
        b = names.index(base)
        sign = -1 if lower else 1
        item = ci(sign * (avg[a] - avg[b]), sign * (boot[:, a] - boot[:, b]))
        delta[extended + "_gain_vs_" + base] = item
    return {
        "scores": {name: ci(avg[i], boot[:, i]) for i, name in enumerate(names)},
        "oriented_gains": delta,
        "families": int(np.isfinite(m).all(1).sum()),
    }


def directions(diff, k):
    n = k[:, :2].sum(1)
    d = np.sign(diff)
    h = np.sign(k[:, 0] - k[:, 1])
    hit = (1 + d) / 2
    humans = np.full_like(diff, np.nan)
    good = n > 0
    humans[good] = (
        k[good, 0, None] * hit[good] + k[good, 1, None] * (1 - hit[good])
    ) / n[good, None]
    pt = (1 + h[:, None] * d) / 2
    pt[h == 0] = np.nan
    return humans, pt


def run():
    seal = json.loads((OUT / "forecast-seal.json").read_bytes())
    pred = OUT / "forecasts.npz"
    if digest(pred) != seal["forecasts_sha256"]:
        raise ValueError("Unsealed predictive output")
    with np.load(pred, allow_pickle=False) as z:
        forecast = {k: z[k] for k in z.files}
    with np.load(DATA / "word-features.npz", allow_pickle=False) as z:
        rbank = {k: z[k] for k in z.files}
    u = pack_model_input("dense")
    labels, totals, ucov = selected_counts("dense", u)
    k = labels.sum(1)
    n = totals.sum(1)
    ix = np.flatnonzero(n > 0)
    y = k[ix] / n[ix]
    if not np.array_equal(u["doc"], forecast["u_doc"]) or not np.array_equal(
        u["coords"], forecast["u_coords"]
    ):
        raise ValueError("Held word geometry changed")
    pairs_r = comparisons(rbank, "held", outcomes=True)
    pr = pairs_r["counts"]
    den = pr.sum(1)
    true = pr / den[:, None]
    if not np.array_equal(pairs_r["keys"], forecast["r_keys"]) or not np.array_equal(
        pairs_r["family"], forecast["r_family"]
    ):
        raise ValueError("Held rewrite geometry changed")
    focus = [
        ("contextual.adaptive", "contextual.independent"),
        ("contextual.adaptive", "lexical.adaptive"),
        ("lexical.adaptive", "lexical.independent"),
        ("contextual.shared", "contextual.independent"),
        ("contextual.independent", "lexical.independent"),
    ]
    uq = np.clip(
        np.column_stack([forecast["u." + a] for a in LEARNED])[ix], 1e-12, 1 - 1e-12
    )
    ud = u["doc"][ix]
    uf = u["family"][ix]
    bern = -(y[:, None] * np.log(uq) + (1 - y[:, None]) * np.log1p(-uq))
    brier = (1 - y[:, None]) * uq**2 + y[:, None] * (1 - uq) ** 2
    uout = {
        "logloss": reported(bern, ud, uf, LEARNED, focus, lower=True),
        "brier": reported(brier, ud, uf, LEARNED, focus, lower=True),
        "judged_words": len(ix),
        "actual_judgments": int(n.sum()),
        "coverage": ucov,
    }
    scores = []
    dg = []
    fg = []
    num = 0
    for d, v in group_index(ud).items():
        a, b, sg = pairs(v, k[ix], n[ix])
        if not len(a):
            continue
        scores.append(((1 + np.sign(uq[a] - uq[b]) * sg[:, None]) / 2).mean(0))
        dg.append(d)
        fg.append(uf[v[0]])
        num += len(a)
    uout["concordance"] = reported(
        np.stack(scores), np.array(dg), np.array(fg), LEARNED, focus
    )
    uout["unequal_pairs"] = int(num)
    rq = np.stack([forecast["r." + a] for a in LEARNED], axis=1)
    rd = pairs_r["doc"]
    rf = pairs_r["family"]
    log3 = -np.sum(xlogy(true[:, None, :], np.clip(rq, 1e-12, 1)), axis=2)
    score3 = np.sum(rq**2, axis=2) - 2 * np.sum(rq * true[:, None, :], axis=2) + 1
    rout = {
        "logloss": reported(log3, rd, rf, LEARNED, focus, lower=True),
        "brier": reported(score3, rd, rf, LEARNED, focus, lower=True),
        "pairs": len(rd),
        "actual_votes": int(pr.sum()),
        "tie_votes": int(pr[:, 2].sum()),
        "contexts": len(set(rd)),
        "families": len(set(rf)),
        "coverage": pairs_r["coverage"],
    }
    learned_diff = rq[:, :, 0] - rq[:, :, 1]
    zero_diff = np.column_stack(
        [forecast["zero." + k][:, 1] - forecast["zero." + k][:, 0] for k in ZERO]
    )
    diffs = np.column_stack([learned_diff, zero_diff])
    names = LEARNED + ["zero." + s for s in ZERO]
    compared = focus + [
        ("zero.adaptive", "zero.lexical"),
        ("zero.adaptive", "zero.rarity"),
        ("zero.complex", "zero.rarity"),
        ("zero.plaba", "zero.rarity"),
        ("contextual.adaptive", "zero.adaptive"),
    ]
    votes, major = directions(diffs, pr)
    for label, matrix in (
        ("direction_per_non_tie_response", votes),
        ("direction_panel_majority_among_non_ties", major),
    ):
        rout[label] = reported(matrix, rd, rf, names, compared)
        rout[label]["pairs"] = int(np.isfinite(matrix).all(1).sum())
    freq = diffs[:, names.index("zero.rarity")]
    self0 = diffs[:, names.index("zero.adaptive")]
    disagree = (freq * self0) < 0
    # This subset depends only on fixed alternative forecasts, never on human winners.
    selected_votes = votes.copy()
    selected_votes[~disagree] = np.nan
    rout["frequency_discordant_non_tie_response"] = reported(
        selected_votes, rd, rf, names, compared
    )
    rout["frequency_discordant_non_tie_response"]["pairs"] = int(
        np.isfinite(selected_votes[:, 0]).sum()
    )
    common = forecast["reference.common"]
    ref = forecast["reference.learned"]
    refnames = [
        "contextual.adaptive",
        "lexical.adaptive",
        "zero.adaptive",
        "causal_word",
        "word_through_three",
        "lexical_plus_surprisal",
    ]
    refdiff = np.column_stack(
        [
            diffs[:, names.index("contextual.adaptive")],
            diffs[:, names.index("lexical.adaptive")],
            diffs[:, names.index("zero.adaptive")],
            forecast["reference.target"],
            forecast["reference.continuation"],
            ref[:, 0] - ref[:, 1],
        ]
    )
    refvotes, refpanel = directions(refdiff, pr)
    refcontrast = [
        ("contextual.adaptive", name)
        for name in refnames
        if name != "contextual.adaptive"
    ]
    sub = {"pairs": int(common.sum())}
    for endpoint, data in (("non_tie_vote", refvotes), ("non_tie_majority", refpanel)):
        m = data.copy()
        m[~common] = np.nan
        sub[endpoint] = reported(m, rd, rf, refnames, refcontrast)
    rn = ["contextual.adaptive", "lexical.adaptive", "lexical_plus_surprisal"]
    rr = np.column_stack(
        [
            log3[:, LEARNED.index(rn[0])],
            log3[:, LEARNED.index(rn[1])],
            -np.sum(xlogy(true, np.clip(ref, 1e-12, 1)), axis=1),
        ]
    )
    rr[~common] = np.nan
    sub["logloss"] = reported(
        rr, rd, rf, rn, [(rn[0], rn[1]), (rn[0], rn[2])], lower=True
    )
    rout["conditional_language_model_reference"] = sub
    flo = (pr * (pr - 1)).sum(1) / (den * (den - 1))
    rout["pointwise_same_choice_two_respondents"] = reported(
        flo[:, None], rd, rf, ["agreement"], []
    )
    delta = "contextual.adaptive_gain_vs_contextual.independent"
    wp = uout["logloss"]["oriented_gains"][delta]["ci95"][0]
    rp = rout["logloss"]["oriented_gains"][delta]["ci95"][0]
    report = {
        "unknown_word": uout,
        "revision_preference": rout,
        "improved_on_both_held_behavioral_likelihoods": bool(wp > 0 and rp > 0),
        "bootstrap_family_draws": B,
        "seed": SEED,
        "forecast_seal_sha256": digest(OUT / "forecast-seal.json"),
        "code_sha256": digest(Path(__file__)),
    }
    (OUT / "scores.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(
        "Behavioral gains",
        uout["logloss"]["oriented_gains"][delta],
        rout["logloss"]["oriented_gains"][delta],
    )


if __name__ == "__main__":
    run()
