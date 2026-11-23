"""Evaluate source-frozen rankings and nested reliability-weighted forecasts on static summaries."""

import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

from research.reading_validation.local_forecast import digest
from research.reading_validation.relay_summary_fit import ARMS, TARGETS, joined, loss
from research.reading_validation.relay_summary_material import OUT, PRIVATE

B = 10000
SEED = 693701


def interval(point, boot):
    finite = np.isfinite(boot)
    return {
        "estimate": float(point),
        "ci95": np.quantile(boot[finite], [0.025, 0.975]).tolist()
        if finite.mean() >= 0.95
        else [None, None],
        "valid_bootstrap_fraction": float(finite.mean()),
    }


class Crossed:
    def __init__(self, records):
        readers = sorted({r["reader"] for r in records})
        families = sorted({r["family"] for r in records})
        self.r = np.asarray([readers.index(v["reader"]) for v in records], int)
        self.a = np.asarray([families.index(v["family"]) for v in records], int)
        rng = np.random.default_rng(SEED)
        self.people = rng.multinomial(
            len(readers), np.ones(len(readers)) / len(readers), size=B
        )
        self.material = rng.multinomial(
            len(families), np.ones(len(families)) / len(families), size=B
        )

    def aggregate(self, rows, values, clusters, weights):
        """Average pairwise terms inside the focal cluster, then average informative clusters."""
        g = values.shape[1]
        add = np.zeros((B, g))
        den = np.zeros(B)
        point = []
        for j, sel in sorted(rows.items()):
            v = values[sel]
            w = weights(sel)
            d = w.sum(axis=1)
            inform = d > 0
            focal = clusters[:, j] * inform
            a = np.divide(
                w @ v, d[:, None], where=inform[:, None], out=np.zeros((B, g))
            )
            add += focal[:, None] * a
            den += focal
            point.append(v.mean(0))
        if not point:
            return np.full(g, np.nan), np.full((B, g), np.nan)
        return np.mean(point, axis=0), np.divide(
            add, den[:, None], where=den[:, None] > 0, out=np.full_like(add, np.nan)
        )

    def rank(self, observed, score, *, same_article=False):
        group = self.a if same_article else self.r
        other = self.r if same_article else self.a
        lookup = defaultdict(list)
        for i, k in enumerate(group):
            lookup[k].append(i)
        pairs = []
        rows = defaultdict(list)
        for gid, ids in lookup.items():
            for i, j in combinations(ids, 2):
                if other[i] != other[j] and observed[i] != observed[j]:
                    rows[gid].append(len(pairs))
                    pairs.append((i, j))
        if not pairs:
            return {"count": 0}
        iz, jz = np.asarray(pairs).T
        values = (
            1
            + np.sign(observed[iz] - observed[jz])[:, None]
            * np.sign(score[iz] - score[jz])
        ) / 2
        inner = self.people if same_article else self.material

        def w(sel):
            return inner[:, other[iz[sel]]] * inner[:, other[jz[sel]]]

        center, boot = self.aggregate(
            rows, values, self.material if same_article else self.people, w
        )
        return {
            "count": len(pairs),
            "clusters": len(rows),
            "reader_count": len(set(self.r[np.r_[iz, jz]])),
            "article_family_count": len(set(self.a[np.r_[iz, jz]])),
            "center": center,
            "boot": boot,
        }

    def visit(self, v):
        lookup = defaultdict(list)
        for i, k in enumerate(self.r):
            lookup[k].append(i)
        return self.aggregate(
            lookup, v, self.people, lambda sel: self.material[:, self.a[sel]]
        )


def compare_rank(cross, y, vals, names, same_article=False):
    q = cross.rank(y, vals, same_article=same_article)
    if not q["count"]:
        return q
    x, b = q.pop("center"), q.pop("boot")
    q["scores"] = {k: interval(x[i], b[:, i]) for i, k in enumerate(names)}
    pairs = (
        [("prior", "constant")]
        if same_article
        else [
            ("plaba_ease", "constant"),
            ("plaba_ease", "authors_formula"),
            ("plaba_ease", "flesch_kincaid"),
            ("plaba_ease", "readme_surface"),
            ("plaba_understanding", "authors_formula"),
            ("readme", "authors_formula"),
        ]
    )
    q["differences"] = {}
    for m, n in pairs:
        i, j = names.index(m), names.index(n)
        q["differences"][m + "_minus_" + n] = interval(x[i] - x[j], b[:, i] - b[:, j])
    return q


def human_readout(rec):
    group = defaultdict(list)
    for r in rec:
        group[r["article"]].append(r)
    counts = {k: dict(sorted(Counter(q[k] for q in rec).items())) for k in TARGETS}
    ranges = {}
    for k in TARGETS:
        grouped = [
            max(r[k] for r in vals) - min(r[k] for r in vals)
            for vals in group.values()
            if len(vals) > 1
        ]
        ranges[k] = {
            "multiple_reader_articles": len(grouped),
            "observed_range_width_distribution": dict(sorted(Counter(grouped).items())),
        }
    unique = sorted({r["reader"] for r in rec})
    return {
        "visits": len(rec),
        "readers": len(unique),
        "articles": len(group),
        "observed_distributions": counts,
        "within_article_range": ranges,
        "exact_content_correct": sum(q["correct"] for q in rec),
        "possible_content": 4 * len(rec),
        "binary_option_accuracy": sum(q["option_decisions"] for q in rec)
        / (12 * len(rec)),
        "attention_checks_passed": sum(q["check"] for q in rec),
        "study_scaled_to_three_visits": float(
            np.mean(
                [
                    3 * np.mean([r["correct"] for r in rec if r["reader"] == j])
                    for j in unique
                ]
            )
        ),
        "mean_likert": {
            k: float(np.mean([q[k] for q in rec])) for k in TARGETS if k != "correct"
        },
    }


def run():
    trained = json.loads((OUT / "trained-seal.json").read_bytes())
    by_sample = {}
    for label, all_rows in (("primary", False), ("six_visit_sensitivity", True)):
        rec = joined(all_rows)
        cross = Crossed(rec)
        out = {
            "human": human_readout(rec),
            "frozen": {},
            "within_same_summary": {},
            "adaptive": {},
        }
        names = list(rec[0]["scores"]) + ["prior"]
        scores = np.array(
            [[v["scores"][k] for k in names[:-1]] + [v["prior"]] for v in rec]
        )
        path = PRIVATE / (label + "-predictions.npz")
        if digest(path) != trained["private_predictions_sha256"][label]:
            raise ValueError("Unrecognized held-out probabilities")
        with np.load(path, allow_pickle=False) as bank:
            for target in TARGETS:
                y = np.asarray([v[target] for v in rec])
                out["frozen"][target] = compare_rank(cross, y, scores, names)
                out["within_same_summary"][target] = compare_rank(
                    cross,
                    y,
                    scores[:, [names.index("constant"), names.index("prior")]],
                    ["constant", "prior"],
                    True,
                )
                losses = np.column_stack(
                    [loss(bank[target + "." + k], y, target == "correct") for k in ARMS]
                )
                x, b = cross.visit(losses)
                q = {
                    "proper_score": {
                        k: interval(x[i], b[:, i]) for i, k in enumerate(ARMS)
                    },
                    "paired_improvement": {},
                }
                for base, ext in [
                    ("constant", "independent"),
                    ("constant", "adaptive"),
                    ("independent", "shared"),
                    ("independent", "adaptive"),
                    ("independent_prior", "adaptive_prior"),
                    ("adaptive", "adaptive_prior"),
                    ("independent", "independent_prior"),
                ]:
                    i, j = ARMS.index(base), ARMS.index(ext)
                    q["paired_improvement"][ext + "_over_" + base] = interval(
                        x[i] - x[j], b[:, i] - b[:, j]
                    )
                out["adaptive"][target] = q
        primary = out["frozen"]["simplicity"]["differences"]
        rank_bounds = [
            primary["plaba_ease_minus_" + control]["ci95"][0]
            for control in ("constant", "authors_formula")
        ]
        adaptive = out["adaptive"]["simplicity"]["paired_improvement"][
            "adaptive_over_independent"
        ]["ci95"][0]
        out["frozen_primary_passed"] = all(v is not None and v > 0 for v in rank_bounds)
        out["adaptive_primary_passed"] = adaptive is not None and adaptive > 0
        by_sample[label] = out
    info = {
        "datasets": by_sample,
        "score_program_sha256": digest(Path(__file__)),
        "bootstrap_draws": B,
        "seed": SEED,
        "input_sha256": digest(OUT / "inputs.json"),
        "target_forecast_sha256": digest(OUT / "trained-seal.json"),
    }
    (OUT / "scores.json").write_text(json.dumps(info, indent=2, allow_nan=False) + "\n")
    print(
        by_sample["primary"]["frozen_primary_passed"],
        by_sample["primary"]["adaptive_primary_passed"],
    )


if __name__ == "__main__":
    run()
