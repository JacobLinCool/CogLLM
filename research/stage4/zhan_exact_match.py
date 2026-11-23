"""Evaluate existing Exp1 fold-held-out predictions on identical Exp2/3 prompts."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from ..stage3.zhan_landmark_probe import CONDITIONS, VERB_TYPES
from .zhan_weak import ARMS
from .zhan_weak_score import _load_predictions, _paired_intervals


def evaluate(stimuli, heads, exp2, exp3):
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    rows = [json.loads(line) for line in stimuli.read_text(encoding="utf-8").splitlines()]
    by_text = {row["text"]: row for row in rows}
    verbs = {row["family_id"]: row["metadata"]["verb_type"] for row in rows}
    families = sorted(verbs)
    probabilities = _load_predictions(heads, families)
    if len(rows) != 72 or len(by_text) != 72 or len(families) != 36:
        raise ValueError("Expected the 36 Stage 4 Exp1 families and three sealed readout arms")
    index = {family: i for i, family in enumerate(families)}
    report = {
        "exp1_stimuli_sha256": digest(stimuli),
        "stage4_heads_sha256": digest(heads),
        "metric_aggregation": "Calculate metrics separately on each label draw, then average scores across draws.",
        "scope": "Exact canonical prompt matching; separate reader cohorts; preexisting Exp1 out-of-fold predictions with matched Exp1 family held out; prompts and experimental family overlap.",
        "experiments": {},
    }
    for experiment, path, structure in [(2, exp2, "active"), (3, exp3, "canonical")]:
        counts = np.zeros((len(families), 2, 2), dtype=np.int64)
        readers = set()
        with path.open(encoding="utf-8-sig", newline="") as source:
            for row in csv.DictReader(source):
                if row["structure"] != structure or row["prompt"] not in by_text:
                    continue
                stimulus = by_text[row["prompt"]]
                meta = stimulus["metadata"]
                if row["verbType"] != meta["verb_type"] or row["promptType"] != meta["prompt_type"]:
                    raise ValueError(f"Inconsistent labels for matched prompt {stimulus['id']}")
                i = index[stimulus["family_id"]]
                j = CONDITIONS.index(row["promptType"])
                y = ("np2", "np1").index(row["reMention"])
                counts[i, j, y] += 1
                readers.add(row["participantID"])
        included = counts.sum(axis=(1, 2)) > 0
        counts = counts[included]
        totals = counts.sum(axis=-1)
        if (totals == 0).any():
            raise ValueError(f"A matched item lacks one prompt condition in experiment {experiment}")
        p = probabilities[:, :, included, :]
        observed = counts[:, :, 1] / totals
        human_shift = observed[:, 1] - observed[:, 0]
        brier_numerator = (counts[:, :, 0] * p**2 + counts[:, :, 1] * (1 - p)**2).sum(axis=-1)
        model_shift = p[..., 1] - p[..., 0]
        tags = np.array([verbs[family] for family, use in zip(families, included, strict=True) if use])
        strata = [np.flatnonzero(tags == verb) for verb in VERB_TYPES]
        model_groups = [model_shift[..., group].mean(axis=-1) for group in strata]
        human_groups = [human_shift[group].mean().item() for group in strata]
        model_interaction = model_groups[1] - model_groups[0]
        human_interaction = human_groups[1] - human_groups[0]
        difference_ci, effect_error_ci = _paired_intervals(
            brier_numerator[1] - brier_numerator[0], totals.sum(axis=1), (model_shift - human_shift)[:2], strata,
        )
        report["experiments"][str(experiment)] = {
            "source_sha256": digest(path),
            "trials": int(totals.sum()),
            "readers": len(readers),
            "families": int(included.sum()),
            "human_interaction": human_interaction,
            "human_shifts": human_groups,
            "arms": {
                arm: {
                    "brier": float(brier_numerator[i].sum(axis=-1).mean() / totals.sum()),
                    "interaction": model_interaction[i].mean().item(),
                    "absolute_interaction_error": np.abs(model_interaction[i] - human_interaction).mean().item(),
                }
                for i, arm in enumerate(ARMS)
            },
            "weak_minus_limited_brier_95_ci": np.quantile(difference_ci, [.025, .975]).tolist(),
            "weak_minus_limited_abs_interaction_error_95_ci": np.quantile(effect_error_ci, [.025, .975]).tolist(),
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("stimuli", "heads", "exp2", "exp3", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(args.stimuli, args.heads, args.exp2, args.exp3)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
