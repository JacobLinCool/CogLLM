"""Audit the magnitudes shown in the deployed single-sentence English learner view."""

import csv
import json
from pathlib import Path

import numpy as np

from research.reading_validation.local_forecast import digest
from research.reading_validation.nns_features import selected_counts
from research.reading_validation.nns_material import LEVELS, OUT
from research.reading_validation.nns_score import ci

EDGES = np.array([0.01, 0.05, 0.10, 0.20, 0.40])
BINS = ("<1%", "1–5%", "5–10%", "10–20%", "20–40%", "40–100%")
DRAW_COUNT = 10000
SEED = 215817


def assess(p, flags, total, family, draws):
    idx = np.searchsorted(EDGES, p, side="right")
    result = {}
    for j, name in enumerate(("overall", *BINS), -1):
        chosen = np.ones(len(p), bool) if j < 0 else idx == j
        judged = chosen & (total > 0)
        z = family[judged]
        n = total[judged]
        count = int(n.sum())
        if count == 0:
            result[name] = {"responses": 0, "eligible_word_cells": int(chosen.sum())}
            continue
        cells = np.c_[n, flags[judged], n * p[judged]]
        fam = np.column_stack(
            [np.bincount(z, weights=a, minlength=draws.shape[1]) for a in cells.T]
        )
        boot = draws @ fam
        valid = boot[:, 0] > 0
        rates = np.full((len(boot), 2), np.nan)
        rates[valid] = boot[valid, 1:] / boot[valid, :1]
        obs, pred = cells[:, 1:].sum(0) / count
        n_families = len(np.unique(z))
        result[name] = {
            "families": n_families,
            "eligible_word_cells": int(chosen.sum()),
            "judged_word_cells": int(judged.sum()),
            "with_any_mark": int(np.sum(flags[judged] > 0)),
            "responses": count,
            "marks": int(flags[judged].sum()),
            "mean_prediction": ci(pred, rates[:, 1]),
            "realized_rate": ci(obs, rates[:, 0]),
            "realized_minus_prediction": ci(obs - pred, rates[:, 0] - rates[:, 1]),
            "descriptive_sparse_bin": bool(n_families < 10 or valid.mean() < 0.95),
        }
    return result


def run():
    seal = json.loads((OUT / "forecast-seal.json").read_bytes())
    path = OUT / "word-forecasts.npz"
    if digest(path) != seal["outcome_free_forecast_sha256"]:
        raise ValueError("Unexpected deployed source predictions")
    with np.load(path, allow_pickle=False) as f:
        bank = {k: f[k] for k in f.files}
    flags, n, _ = selected_counts("dense", bank)
    fam, inv = np.unique(bank["family"], return_inverse=True)
    draws = np.random.default_rng(SEED).multinomial(
        len(fam), np.ones(len(fam)) / len(fam), size=DRAW_COUNT
    )
    pp = bank["text.adaptive"].ravel()
    cc = bank["cefr.adaptive"]
    if cc.shape != n.shape or pp.shape != n[:, 0].shape:
        raise ValueError(
            "The released word locations no longer match the survey counts"
        )
    outcome = {
        "pooled": assess(pp, flags.sum(1), n.sum(1), inv, draws),
        "measured": assess(
            cc.ravel(), flags.ravel(), n.ravel(), np.repeat(inv, len(LEVELS)), draws
        ),
    }
    for i, label in enumerate(LEVELS):
        outcome[label] = assess(cc[:, i], flags[:, i], n[:, i], inv, draws)
    out = OUT / "calibration"
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "forecast_sha256": digest(path),
        "serving_weights_sha256": digest(OUT / "weights.npz"),
        "source_study": "Paetzold and Specia (2016), Experiment 1, multiply rated items",
        "role": "retrospective fixed-holdout probability calibration",
        "seed": SEED,
        "draws": DRAW_COUNT,
        "family_count": len(fam),
        "code_sha256": digest(Path(__file__)),
        "protocol_sha256": digest(
            Path(__file__).with_name("nns-calibration-protocol.md")
        ),
        "scores": outcome,
    }
    (out / "scores.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    with (out / "source-data.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "group",
                "risk_bin",
                "families",
                "scorable_word_level_cells",
                "judged_cells",
                "responses",
                "marks",
                "predicted",
                "observed",
                "observed_lo",
                "observed_hi",
                "observed_minus_predicted",
                "difference_lo",
                "difference_hi",
            ]
        )
        for arm, outcomes in outcome.items():
            for name, v in outcomes.items():
                if v["responses"] == 0:
                    writer.writerow(
                        [
                            arm,
                            name,
                            0,
                            v["eligible_word_cells"],
                            0,
                            0,
                            0,
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                        ]
                    )
                    continue
                r, delta = v["realized_rate"], v["realized_minus_prediction"]
                writer.writerow(
                    [
                        arm,
                        name,
                        v["families"],
                        v["eligible_word_cells"],
                        v["judged_word_cells"],
                        v["responses"],
                        v["marks"],
                        v["mean_prediction"]["estimate"],
                        r["estimate"],
                        *r["ci95"],
                        delta["estimate"],
                        *delta["ci95"],
                    ]
                )
    print(out / "scores.json", flush=True)


if __name__ == "__main__":
    run()
