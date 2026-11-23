"""Evaluate fixed full-spelling forecasts on independently recorded reader spellings."""

import csv
from collections import Counter, defaultdict

import numpy as np

from research.backbone_transfer.extract import digest, read_json, write_json
from research.reading_validation.dispersion_transfer import HISTORY
from research.reading_validation.reader_history_external import (
    answer_normalizer,
    records,
    sources,
)
from research.reading_validation.reader_history_materials import DIR
from research.reading_validation.spelling_agreement import METHODS, OUT, PLAN
from research.reading_validation.word_mark_certification import original_words


def peelle_counts():
    original = read_json(HISTORY / "external-lock.json")
    future = DIR / "independent-future-items.json"
    if (
        original["future_item_assignments"] != digest(future)
        or original["inputs"] != sources()
    ):
        raise ValueError("Human responses or chronological stem assignments changed")
    events = defaultdict(list)
    for rid, f, k, field, question, s in read_json(future)["chronological"]:
        events[f, k].append((rid, field, question, s))
    forms = defaultdict(Counter)
    readers = set()
    cleaned = answer_normalizer()
    for f, k, entry, _ in records():
        for rid, field, question, s in events.get((f, k), ()):
            _, spelling = cleaned(entry[field], question)
            if spelling is None:
                continue
            if (rid, s) in readers:
                raise ValueError("A participant answered the same stem twice")
            readers.add((rid, s))
            forms[s][spelling] += 1
    if len(readers) != 297927 or set(forms) != set(range(3085)):
        raise ValueError("Exact spellings and previous study denominators differ")
    obs = []
    for i in range(len(forms)):
        c = forms[i]
        n = c.total()
        obs.append(1 - sum(v * (v - 1) for v in c.values()) / (n * (n - 1)))
    return np.array(obs)


def summarize(estimates, observed, material, seed, paired_update):
    if estimates.shape != (len(observed), len(METHODS)):
        raise ValueError("Outcome positions differ from registered forecasts")
    _, group = np.unique(material, return_inverse=True)
    n = int(group.max()) + 1
    rng = np.random.default_rng(seed)
    # Chunking retains paired material draws without a draws-by-word dense matrix.
    size = np.bincount(group, minlength=n)
    error = abs(estimates - observed[:, None])
    contribution = np.stack(
        [np.bincount(group, weights=v, minlength=n) for v in error.T]
    )
    draws = []
    for _ in range(40):
        w = rng.multinomial(n, np.full(n, 1 / n), size=100)
        draws.append((w @ contribution.T) / (w @ size)[:, None])
    draws = np.vstack(draws)
    means = error.mean(0)
    comparisons = ("grouped", "direct")
    differences = np.array([means[METHODS.index(k)] - means[-1] for k in comparisons])
    paired = np.column_stack(
        [draws[:, METHODS.index(k)] - draws[:, -1] for k in comparisons]
    )
    sd = paired.std(axis=0)
    critical = np.quantile(np.max(abs((paired - differences) / sd), axis=1), 0.95)
    scores = {}
    for j, name in enumerate(METHODS):
        row = {
            "mae": float(means[j]),
            "mean_prediction": float(estimates[:, j].mean()),
            "interval": np.quantile(draws[:, j], [0.025, 0.975]).tolist(),
        }
        if paired_update:
            active = abs(observed) > 1e-12
            row["direction_with_half_for_tie"] = float(
                np.mean(
                    (1 + np.sign(estimates[active, j]) * np.sign(observed[active])) / 2
                )
            )
        scores[name] = row
    return {
        "sites": len(observed),
        "materials": n,
        "mean_observed": float(observed.mean()),
        "methods": scores,
        "conditional_mae_reduction": {
            k: {
                "mean": float(differences[j]),
                "simultaneous": (
                    differences[j] + np.array([-1, 1]) * critical * sd[j]
                ).tolist(),
            }
            for j, k in enumerate(comparisons)
        },
    }


def grade():
    lock = read_json(OUT / "fit-lock.json")
    if lock["protocol"] != digest(PLAN):
        raise ValueError("Scoring protocol differs from source fit")
    scores = {}
    for name in ("provo", "peelle"):
        path = OUT / f"{name}-forecasts.npz"
        if digest(path) != lock["studies"][name]["forecast"]:
            raise ValueError("Material forecast changed after sealing")
        with np.load(path) as archive:
            ids, predictions = archive["ids"], archive["forecast"]
            if tuple(archive["methods"]) != METHODS or len(set(ids)) != len(ids):
                raise ValueError("Unregistered method or duplicated material")
        if name == "peelle":
            observed = peelle_counts()
            scores[name] = summarize(
                predictions, observed, np.arange(len(ids)), 12127, False
            )
        else:
            source = original_words()
            with open(
                "research/results/reading-validation-word-marks/word-updates.csv"
            ) as f:
                pairs = list(csv.DictReader(f))
            lookup = dict(zip(ids, predictions, strict=True))
            changes, y, group = [], [], []
            for p in pairs:
                a, b = p["id_before"], p["id_after"]
                measured = float(source[b][0] - source[a][0])
                if (
                    not source[a][2]
                    or not source[b][2]
                    or abs(measured - float(p["human_change"])) > 1e-12
                ):
                    raise ValueError(
                        "Human spelling counts did not yield the verified change"
                    )
                changes.append(lookup[b] - lookup[a])
                y.append(measured)
                group.append(int(p["paragraph"]))
            scores[name] = summarize(np.stack(changes), np.array(y), group, 47385, True)
            # Word-end levels use all denominator-verified sites, including isolated words.
            ok = np.array([source[i][2] for i in ids])
            true = np.array([float(source[i][0]) for i in ids[ok]])
            scores[name]["level_mae"] = dict(
                zip(METHODS, abs(predictions[ok] - true[:, None]).mean(0).tolist())
            )
            scores[name]["level_sites"] = int(ok.sum())
    write_json(
        OUT / "scores.json",
        {"fit_lock": digest(OUT / "fit-lock.json"), "studies": scores},
    )
    print({k: v["conditional_mae_reduction"] for k, v in scores.items()}, flush=True)


if __name__ == "__main__":
    grade()
