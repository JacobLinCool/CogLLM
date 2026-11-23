"""Test the ordering of reader observations when two original contexts end in the identical word."""
from argparse import ArgumentParser
from collections import defaultdict
import csv
from itertools import combinations
from pathlib import Path

import numpy as np
from research.backbone_transfer.extract import digest, read_json, write_json
from research.reading_validation.convergent_word_prepare import ROOT
from research.reading_validation.convergent_adaptive import BASE, labels
from research.reading_validation.physiology_portability_data import OUT, CANDIDATES
from research.reading_validation.observed_cloze import observe
from research.reading_validation.meaning_score import folds


def held(candidate):
    if candidate == "original_qwen":
        base = BASE / "adaptive"
        lock = read_json(base / "fit-seal.json")
        arr = {}
        ids = []
        for key in ("surface", "context"):
            pred = []
            order = []
            for f in range(5):
                p = base / f"{key}-fold-{f}.json"
                assert digest(p) in lock["file_sha"]
                d = read_json(p)
                pred.extend(d["predictions"]["shared"])
                order.extend(d["held_items"])
            if ids and ids != order:
                raise ValueError("Unequal original score support")
            ids = order
            arr[key] = np.asarray(pred, float)
        return np.asarray(ids), arr, digest(base / "fit-seal.json")
    root = OUT / candidate
    lock = read_json(root / "forecast-lock.json")
    file = root / "forecasts.npz"
    if lock["forecast"] != digest(file):
        raise ValueError("Missing candidate lock")
    with np.load(file) as z:
        return (
            z["eeg_items"],
            {k: z[k + "_P"] for k in ("surface", "context")},
            digest(root / "forecast-lock.json"),
        )


def pairs(words, sent, index, part, restriction):
    within = defaultdict(list)
    for j, w in enumerate(words):
        within[w].append(j)
    i = []
    j = []
    for occurrences in within.values():
        for a, b in combinations(occurrences, 2):
            if (
                sent[a] == sent[b]
                or (restriction == "same_index" and index[a] != index[b])
                or (restriction == "same_fold" and part[a] != part[b])
            ):
                continue
            i.append(a)
            j.append(b)
    return np.asarray(i, int), np.asarray(j, int)


def study(candidate):
    lock = read_json(OUT / "within-word-registration.json")
    protocol = Path(__file__).with_name("context-within-word-protocol.md")
    if digest(protocol) != lock["protocol"]:
        raise ValueError("Scientific registration changed")
    items, pred, source = held(candidate)
    gold = {
        int(r["item_id"]): r for r in csv.DictReader((ROOT / "outcomes.csv").open())
    }
    first = read_json(BASE / "input-seal.json")
    assert digest(ROOT / "outcomes.csv") == first["english_endpoint_table"]
    rows = [gold[i] for i in items]
    obs = observe(labels())
    y = np.array(
        [
            [-float(r["N400"]), obs[k][1] / obs[k][0], -float(r["rating_mean"])]
            for k, r in zip(items, rows)
        ]
    )
    words = np.array([r["word"] for r in rows])
    un, types = np.unique(words, return_inverse=True)
    sents = np.array([r["sent_id"] for r in rows])
    sn, s = np.unique(sents, return_inverse=True)
    index = np.array([int(r["context_length"]) for r in rows])
    foldby = folds(sents, "UCL 2026 reading sentence:")
    part = np.array([foldby[k] for k in sents])
    assert len(set(items)) == 1546
    rng = np.random.default_rng(19219)
    sw = rng.exponential(size=(2000, len(sn)))
    tw = rng.exponential(size=(2000, len(un)))
    result = {}
    for restriction in ["all", "same_index", "same_fold"]:
        i, j = pairs(words, s, index, part, restriction)
        result[restriction] = {}
        samples = []
        for k, name in enumerate(
            ["N400_negativity", "cloze_miss", "rated_unpredictability"]
        ):
            sign = np.sign(y[i, k] - y[j, k])
            mask = sign != 0
            a = i[mask]
            b = j[mask]
            direction = sign[mask]
            t = types[a]
            eligible = np.unique(t)
            grouping = np.unique(
                np.sort(np.stack([s[a], s[b]], axis=1), axis=1), axis=0
            )
            votes = np.column_stack(
                [
                    0.5 * (1 + direction * np.sign(v[a, k] - v[b, k]))
                    for v in pred.values()
                ]
            )
            data = []

            def macro(ws, wt):
                pair = ws[s[a]] * ws[s[b]]
                normal = np.bincount(t, weights=pair, minlength=len(un))[t]
                coefficients = pair / normal * wt[t] / sum(wt[eligible])
                return coefficients @ votes

            for ws, wt in zip(sw, tw):
                data.append(macro(ws, wt))
            center = macro(np.ones(len(sn)), np.ones(len(un)))
            boots = np.array(data)
            delta = boots[:, 1] - boots[:, 0]
            delta0 = center[1] - center[0]
            samples.append((delta, delta0))
            result[restriction][name] = {
                "types": len(eligible),
                "positions": len(np.union1d(a, b)),
                "sentences": len(np.unique(np.r_[s[a], s[b]])),
                "occurrence_pairs": len(a),
                "sentence_pairs": len(grouping),
                "tied_human_pairs_excluded": int((~mask).sum()),
                "scores": {
                    h: {
                        "concordance": v,
                        "crossed_interval": np.quantile(
                            boots[:, q], [0.025, 0.975]
                        ).tolist(),
                    }
                    for q, (h, v) in enumerate(zip(pred.keys(), center))
                },
                "gain": delta0,
                "crossed_interval": np.quantile(delta, [0.025, 0.975]).tolist(),
            }
        changes = np.stack([a - b for a, b in samples], axis=1)
        rad = np.quantile(np.max(abs(changes), axis=1), 0.95)
        for v in result[restriction].values():
            v["simultaneous_gain_interval"] = [v["gain"] - rad, v["gain"] + rad]
    dest = OUT / "within-word"
    dest.mkdir(exist_ok=True)
    write_json(
        dest / (candidate + ".json"),
        dict(
            protocol=digest(protocol),
            source=source,
            candidate=candidate,
            results=result,
        ),
    )
    print(
        candidate,
        [
            (k, v["types"], v["gain"], v["crossed_interval"])
            for k, v in result["all"].items()
        ],
    )


if __name__ == "__main__":
    p = ArgumentParser(description=__doc__)
    p.add_argument("candidate", choices=("original_qwen", *CANDIDATES))
    study(p.parse_args().candidate)
