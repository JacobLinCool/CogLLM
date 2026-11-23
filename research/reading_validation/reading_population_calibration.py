"""Adapt an exact-spelling forecast to a study with disjoint reference readers and stems."""

from argparse import ArgumentParser
from collections import defaultdict
from hashlib import sha256
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.special import expit, logit
from scipy.stats import spearmanr

from research.backbone_transfer.extract import digest, read_json, write_json
from research.reading_validation.dispersion_transfer import HISTORY
from research.reading_validation.reader_history_external import (
    answer_normalizer,
    records,
    sources,
)
from research.reading_validation.reader_history_materials import DIR
from research.reading_validation.spelling_agreement import OUT as SPELL

OUT = SPELL.parent / "reading-validation-population-calibration"
PLAN = Path(__file__).with_name("reading-population-calibration-protocol.md")


def inputs():
    future = DIR / "independent-future-items.json"
    source = read_json(HISTORY / "external-lock.json")
    path = SPELL / "peelle-forecasts.npz"
    if (
        digest(future) != source["future_item_assignments"]
        or source["inputs"] != sources()
        or digest(path)
        != read_json(SPELL / "fit-lock.json")["studies"]["peelle"]["forecast"]
    ):
        raise ValueError(
            "Role geometry, original outcomes or source predictions changed"
        )
    with np.load(path) as z:
        ids = z["ids"]
        p = z["forecast"][:, list(z["methods"]).index("conditional")]
    events = read_json(future)["chronological"]
    people = sorted({r[0] for r in events})
    rank = lambda tag, v: sha256(
        f"CogLLM task reference panel v1|{tag}|{v}".encode()
    ).digest()
    readers = sorted(people, key=lambda i: rank("person", i))
    stimuli = sorted(range(len(ids)), key=lambda i: rank("stem", ids[i]))
    if len(readers) != 233 or len(stimuli) != 3085:
        raise ValueError("Registered role sizes do not match the study")
    roles = (
        {"people": sorted(readers[:32]), "stems": sorted(stimuli[:64])},
        {"people": sorted(readers[32:]), "stems": sorted(stimuli[64:])},
    )
    return p, events, roles


class Panel:
    def __init__(self, events, roles, minimum):
        people = {rid: i for i, rid in enumerate(roles["people"])}
        stems = set(roles["stems"])
        needed = defaultdict(list)
        for rid, f, k, field, question, s in events:
            if s in stems and rid in people:
                needed[f, k].append((people[rid], field, question, s))
        normalize = answer_normalizer()
        by_stem = defaultdict(dict)
        for f, k, row, _ in records():
            for rid, field, question, s in needed.get((f, k), ()):
                _, word = normalize(row[field], question)
                if word is None:
                    continue
                if rid in by_stem[s]:
                    raise ValueError(
                        "Repeated response from one person to one stimulus"
                    )
                by_stem[s][rid] = word
        self.stems = np.array(
            sorted(s for s, words in by_stem.items() if len(words) >= minimum)
        )
        self.people = len(people)
        self.items = len(self.stems)
        cells, cols, cuts = [], [], []
        stem_rows, stem_cols = [], []
        next_cell = 0
        for j, stem in enumerate(self.stems):
            words = by_stem[stem]
            lookup = {
                w: next_cell + i for i, w in enumerate(sorted(set(words.values())))
            }
            cuts.append(next_cell)
            next_cell += len(lookup)
            for rid, word in words.items():
                cells.append(lookup[word])
                cols.append(rid)
                stem_rows.append(j)
                stem_cols.append(rid)
        self.word_counts = csr_matrix(
            (np.ones(len(cells)), (cells, cols)), shape=(next_cell, self.people)
        )
        self.coverage = csr_matrix(
            (np.ones(len(stem_rows)), (stem_rows, stem_cols)),
            shape=(self.items, self.people),
        )
        self.cuts = np.array(cuts)
        self.responses = len(cells)

    def disagreement(self, weights):
        """Reweight people while preserving both their repeated measurements and their identity."""
        people = np.atleast_2d(weights)
        total = self.coverage @ people.T
        same_person = self.coverage @ (people.T**2)
        same_word = np.add.reduceat(
            (self.word_counts @ people.T) ** 2, self.cuts, axis=0
        )
        opportunities = total**2 - same_person
        values = np.divide(
            total**2 - same_word,
            opportunities,
            where=opportunities > 0,
            out=np.full_like(total, np.nan, dtype=float),
        )
        return values.T


def adjust(p, y, weights):
    y = np.atleast_2d(y)
    w = np.atleast_2d(weights) * np.isfinite(y)
    mean = (np.nan_to_num(y) * w).sum(1) / w.sum(1)
    intercept = np.zeros(len(w))
    l = logit(p)
    for _ in range(30):
        q = expit(l + intercept[:, None])
        error = (q * w).sum(1) - mean * w.sum(1)
        if np.max(abs(error) / w.sum(1)) < 1e-11:
            return intercept, mean
        derivative = (q * (1 - q) * w).sum(1)
        intercept -= np.clip(error / derivative, -2, 2)
    raise ArithmeticError("Study offset failed to converge")


def register():
    p, events, roles = inputs()
    OUT.mkdir(exist_ok=True)
    panel = Panel(events, roles[0], 5)
    y = panel.disagreement(np.ones(panel.people))
    bias, mean = adjust(p[panel.stems], y, np.ones(panel.items))
    dest = OUT / "forecast.npz"
    all_held = np.array(roles[1]["stems"])
    np.savez_compressed(
        dest,
        stem=all_held,
        source=p[all_held],
        calibrated=expit(logit(p[all_held]) + bias[0]),
        constant=mean,
        intercept=bias,
    )
    write_json(
        OUT / "forecast-lock.json",
        {
            "protocol": digest(PLAN),
            "source": digest(SPELL / "fit-lock.json"),
            "roles": roles,
            "calibration_people": panel.people,
            "calibration_stems": panel.stems.tolist(),
            "calibration_responses": panel.responses,
            "target_constant": float(mean[0]),
            "offset": float(bias[0]),
            "held_respondents_used_in_fit": 0,
            "held_stems_used_in_fit": 0,
            "forecast": digest(dest),
        },
    )


def scores():
    p, events, roles = inputs()
    seal = read_json(OUT / "forecast-lock.json")
    if (
        seal["roles"] != list(roles)
        or digest(PLAN) != seal["protocol"]
        or seal["forecast"] != digest(OUT / "forecast.npz")
    ):
        raise ValueError("Forecast, protocol or reader/material holdout changed")
    calibration, held = Panel(events, roles[0], 5), Panel(events, roles[1], 2)
    y = held.disagreement(np.ones(held.people))[0]
    pc = p[calibration.stems]
    pt = p[held.stems]
    source_c = read_json(SPELL / "fit-lock.json")["weights"]["constant"]
    q = expit(logit(pt) + seal["offset"])
    estimates = np.array(
        [q, pt, np.full(len(y), seal["target_constant"]), np.full(len(y), source_c)]
    )
    with np.load(OUT / "forecast.npz") as f:
        if not np.array_equal(
            f["calibrated"][np.searchsorted(f["stem"], held.stems)], q
        ):
            raise ValueError("A graded probability differs from its seal")
    errs = abs(estimates - y).mean(1)
    rng = np.random.default_rng(662504)

    def counts(n, batch):
        return rng.multinomial(n, np.full(n, 1 / n), batch)

    boot = []
    for _ in range(80):
        yc = calibration.disagreement(counts(calibration.people, 10))
        yh = held.disagreement(counts(held.people, 10))
        offset, adapted = adjust(pc, yc, counts(calibration.items, 10))
        w = counts(held.items, 10) * np.isfinite(yh)
        tested = (expit(logit(pt) + offset[:, None]), pt, adapted[:, None], source_c)
        boot.append(
            np.column_stack(
                [(np.nan_to_num(abs(f - yh)) * w).sum(1) / w.sum(1) for f in tested]
            )
        )
    draws = np.vstack(boot)
    gains = errs[1:3] - errs[0]
    contrasts = draws[:, 1:3] - draws[:, [0]]
    sd = contrasts.std(0)
    critical = np.quantile(np.max(abs((contrasts - gains) / sd), axis=1), 0.95)
    names = ("calibrated", "source_only", "same_data_constant", "source_constant")
    results = {
        k: {
            "mae": float(errs[j]),
            "mean_forecast": float(estimates[j].mean()),
            "mae_interval": np.quantile(draws[:, j], [0.025, 0.975]).tolist(),
            "spearman": float(spearmanr(estimates[j], y).statistic) if j < 2 else None,
        }
        for j, k in enumerate(names)
    }
    write_json(
        OUT / "scores.json",
        {
            "seal": digest(OUT / "forecast-lock.json"),
            "held_readers": held.people,
            "held_stems": held.items,
            "held_responses": held.responses,
            "measured_mean": float(y.mean()),
            "results": results,
            "bootstrap_draws": len(draws),
            "gain": {
                k: {
                    "difference": float(gains[j]),
                    "simultaneous": (
                        gains[j] + np.array([-1, 1]) * critical * sd[j]
                    ).tolist(),
                }
                for j, k in enumerate(names[1:3])
            },
        },
    )
    print(results, gains)


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("action", choices=("lock", "score"))
    (register if args.parse_args().action == "lock" else scores)()
