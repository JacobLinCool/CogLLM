"""Check the component selectivity of frozen Spanish-reading coefficients on English displays."""
from argparse import ArgumentParser
import csv
from itertools import chain
from pathlib import Path

import numpy as np
from research.backbone_transfer.extract import (
    digest,
    read_json,
    stream_rows,
    write_json,
)
from research.reading_validation.convergent_word_score import (
    OUT,
    ROOT,
    assemble,
    predict,
)
from research.reading_validation.physiology_portability_score import fast_ranks, R

EEG = {"ELAN": -1, "LAN": -1, "N400": -1, "EPNP": 1, "P600": 1, "PNP": 1}
READING = (
    "RTfirstfix",
    "RTfirstpass",
    "RTrightbound",
    "RTgopast",
    "self_paced_reading_time",
)


def inputs(original, replay):
    source = read_json(OUT / "selectivity-registration.json")
    for key, filename in [
        ("frozen_source", "scores.json"),
        ("previous_N400", "original-exposure-scores.json"),
    ]:
        if digest(OUT / filename) != source[key]:
            raise ValueError("Published endpoint changed")
    if (
        digest(Path(__file__).with_name("word-selectivity-protocol.md"))
        != source["protocol"]
    ):
        raise ValueError("Wrong scientific protocol")
    old_receipt = read_json(OUT / "original-exposure-scores.json")
    if (
        digest(replay / "receipt.json") != old_receipt["replay"]
        or digest(replay / "word-states.npz")
        != read_json(replay / "receipt.json")["word_states"]
    ):
        raise ValueError("Unverified original human exposure")
    (_, _, _, state, _, coords, _, _), _ = assemble(original)
    rows = {int(v[1]): x for x, v in zip(state, coords)}
    with np.load(replay / "word-states.npz") as states:
        for doc in stream_rows(ROOT / "erp-map.jsonl"):
            for item, offset, same in doc["items"]:
                if not same:
                    if item in rows:
                        raise ValueError("Duplicate printed word")
                    rows[item] = states[doc["id"]][offset].astype(float)
    gold = {
        int(row["item_id"]): row
        for row in csv.DictReader((ROOT / "outcomes.csv").open())
    }
    if len(gold) != 1726 or set(gold) != set(rows):
        raise ValueError("Lost an original word")
    ids = sorted(gold)
    X = np.stack([rows[k] for k in ids])
    models = read_json(OUT / "scores.json")["external_spanish_no_updates"][
        "spanish_coefficients"
    ]
    c = predict(X, models["context"])
    predictions = {
        "spanish_N400": c[:, 0],
        "spanish_early": c[:, 1],
        "trained_surface": predict(X[:, :6], models["qwen_surface"])[:, 0],
        "raw_surprisal": X[:, 5],
    }
    truth = {
        name: np.asarray([sign * float(gold[k][name] or "nan") for k in ids])
        for name, sign in {**EEG, **dict.fromkeys(READING, 1.0)}.items()
    }
    return np.asarray([gold[k]["sent_id"] for k in ids]), predictions, truth


def score(original, replay):
    group, pred, truth = inputs(original, replay)
    groups, ix = np.unique(group, return_inverse=True)
    if len(groups) != 205:
        raise ValueError("Changed English article coverage")
    rng = np.random.default_rng(738)
    counts = rng.multinomial(
        len(groups), np.full(len(groups), 1 / len(groups)), size=2000
    )
    result = {}
    tested = {}
    target = "N400"
    for name, y in truth.items():
        use = np.isfinite(y)
        j = ix[use]
        gold = fast_ranks(y[use])
        funcs = {k: fast_ranks(z[use]) for k, z in pred.items()}
        stats = {k: [] for k in funcs}
        for w in chain([np.ones(len(j))], (v[j].astype(float) for v in counts)):
            rr = gold(w)
            for k, calc in funcs.items():
                stats[k].append(R(calc(w), rr, w))
        tested[name] = {k: np.asarray(v) for k, v in stats.items()}
        result[name] = {
            "words": int(use.sum()),
            "sentences": len(set(group[use])),
            "association": {
                k: {
                    "rho": v[0],
                    "sentence_interval": np.quantile(v[1:], [0.025, 0.975]).tolist(),
                }
                for k, v in stats.items()
            },
        }
    report = {}
    for model in pred:
        scored = tested[target][model]
        alternatives = [k for k in EEG if k != target]
        competitors = np.column_stack([tested[a][model] for a in alternatives])
        d = scored[:, None] - competitors
        expansion = np.quantile(np.max(abs(d[1:] - d[0]), axis=1), 0.95)
        individual = {
            name: {
                "rho_gain": float(d[0, i]),
                "simultaneous_interval": [
                    float(d[0, i] - expansion),
                    float(d[0, i] + expansion),
                ],
            }
            for i, name in enumerate(alternatives)
        }
        signed = d.min(1)
        absolute = abs(scored) - np.max(abs(competitors), axis=1)

        def effect(a):
            return {
                "rho_gain": float(a[0]),
                "sentence_interval": np.quantile(a[1:], [0.025, 0.975]).tolist(),
            }

        report[model] = {
            "largest_directional_competitor": alternatives[competitors[0].argmax()],
            "against_max_directional": effect(signed),
            "against_max_absolute": effect(absolute),
            "individual_comparisons": individual,
        }
    expected = read_json(OUT / "original-exposure-scores.json")["observations"][
        "entire_1726"
    ]["rho"]["context"]["rho"]
    if abs(result[target]["association"]["spanish_N400"]["rho"] - expected) > 1e-8:
        raise ValueError("N400 does not reproduce previous scoring")
    write_json(
        OUT / "word-selectivity.json",
        {
            "source": digest(OUT / "selectivity-registration.json"),
            "program": digest(Path(__file__)),
            "grouped_draws": len(counts),
            "words": len(group),
            "sentences": len(groups),
            "associations": result,
            "specificity": report,
        },
    )
    print(report["spanish_N400"]["against_max_directional"])


if __name__ == "__main__":
    p = ArgumentParser(description=__doc__)
    p.add_argument("original", type=Path)
    p.add_argument("replay", type=Path)
    x = p.parse_args()
    score(x.original, x.replay)
