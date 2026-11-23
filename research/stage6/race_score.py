"""Evaluate sealed RACE predictions against held-out human answer choices."""

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from ..joint.onestop_probe import METHODS, row_losses


TRIALS = 1940
ARTICLES = 216
READERS = 970
LEVELS = ("Middle", "High")
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 43
BOOTSTRAP_BLOCK = 200


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_outcomes(path, ids):
    outcomes = {}
    fields = {"id", "article_id", "reader_id", "selected", "correct", "level"}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.keys() != fields or row["id"] in outcomes:
                raise ValueError("Unexpected outcome schema or duplicate trial")
            if any(type(row[key]) is not int or not 0 <= row[key] < 4 for key in ("selected", "correct")):
                raise ValueError("Outcome indices must refer to the four displayed options")
            if row["level"] not in LEVELS or any(
                not isinstance(row[key], str) or not row[key] for key in ("article_id", "reader_id")
            ):
                raise ValueError("The source article, reader, and school level must be identified")
            outcomes[row["id"]] = row
    if set(outcomes) != set(ids):
        raise ValueError("Held-out outcomes and sealed prediction IDs do not match")
    if len({row["article_id"] for row in outcomes.values()}) != ARTICLES:
        raise ValueError("Unexpected number of held-out article families")
    readers = {row["reader_id"] for row in outcomes.values()}
    reader_levels = {(row["reader_id"], row["level"]) for row in outcomes.values()}
    if len(readers) != READERS or len(reader_levels) != TRIALS:
        raise ValueError("Expected one trial per school level for each released reader")
    return [outcomes[key] for key in ids]


def bootstrap_row_means(rows_by_loss, articles, readers):
    """Pair models within each resample and cross article and reader weights."""
    _, article_index = np.unique(articles, return_inverse=True)
    _, reader_index = np.unique(readers, return_inverse=True)
    n_articles, n_readers = article_index.max() + 1, reader_index.max() + 1
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    article_weights = rng.multinomial(n_articles, np.full(n_articles, 1 / n_articles), size=BOOTSTRAP_DRAWS)
    reader_weights = rng.multinomial(n_readers, np.full(n_readers, 1 / n_readers), size=BOOTSTRAP_DRAWS)
    means = np.empty((BOOTSTRAP_DRAWS, rows_by_loss.shape[0]), dtype=np.float64)
    for start in range(0, BOOTSTRAP_DRAWS, BOOTSTRAP_BLOCK):
        block = slice(start, min(start + BOOTSTRAP_BLOCK, BOOTSTRAP_DRAWS))
        weights = article_weights[block][:, article_index] * reader_weights[block][:, reader_index]
        totals = weights.sum(axis=1)
        if (totals == 0).any():
            raise ArithmeticError("A crossed group replicate has no released trial")
        means[block] = (weights @ rows_by_loss.T) / totals[:, None]
    return means


def evaluate(outcomes_path, predictions_path, seal_path, output):
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    prediction_sha = digest(predictions_path)
    if prediction_sha != seal["predictions_sha256"]:
        raise ValueError("The predicted probabilities must match the pre-scoring seal")
    with np.load(predictions_path, allow_pickle=False) as archive:
        ids = archive["ids"].tolist()
        values = archive["predictions"]
        if (
            len(ids) != TRIALS or len(set(ids)) != TRIALS
            or archive["methods"].tolist() != list(METHODS)
            or values.shape != (len(METHODS), TRIALS, 4)
            or not np.issubdtype(values.dtype, np.floating)
            or not np.all(np.isfinite(values) & (values > 0) & (values <= 1))
            or not np.allclose(values.sum(axis=-1), 1, atol=1e-7, rtol=0)
        ):
            raise ValueError("Expected the complete sealed display-position probabilities")

    rows = load_outcomes(outcomes_path, ids)
    selected = np.array([row["selected"] for row in rows])
    correct = np.array([row["correct"] for row in rows])
    levels = np.array([row["level"] for row in rows])
    losses = [row_losses(probability, np.eye(4, dtype=np.int64)[selected], correct) for probability in values]
    metric_keys = tuple(losses[0])
    matrix = np.stack([loss[metric] for metric in metric_keys for loss in losses])
    resampled = bootstrap_row_means(matrix, [row["article_id"] for row in rows], [row["reader_id"] for row in rows])
    resampled = resampled.reshape(BOOTSTRAP_DRAWS, len(metric_keys), len(METHODS))
    overall = matrix.mean(axis=1).reshape(len(metric_keys), len(METHODS))
    methods = {
        method: dict(zip(metric_keys, overall[:, i].tolist(), strict=True)) for i, method in enumerate(METHODS)
    }
    contrasts = {
        f"{METHODS[right]}_minus_{METHODS[left]}": {
            metric: {
                "difference": float(overall[m, right] - overall[m, left]),
                "crossed_bootstrap_95_ci": np.quantile(
                    resampled[:, m, right] - resampled[:, m, left], [0.025, 0.975],
                ).tolist(),
            }
            for m, metric in enumerate(metric_keys)
        }
        for left, right in itertools.combinations(range(len(METHODS)), 2)
    }
    by_level = {}
    for level in LEVELS:
        eligible = levels == level
        if eligible.sum() != TRIALS // len(LEVELS):
            raise ValueError("The study has one released RACE trial per school level per reader")
        by_level[level] = {
            "trials": int(eligible.sum()), "human_correct": int(((selected == correct) & eligible).sum()),
            "methods": {
                name: {metric: float(values[eligible].mean()) for metric, values in loss.items()}
                for name, loss in zip(METHODS, losses, strict=True)
            },
        }
    by_response_correctness = {}
    for key, eligible in (("correct", selected == correct), ("incorrect", selected != correct)):
        if not eligible.any():
            raise ValueError("Expected both correct and incorrect human choices in the evaluation")
        by_response_correctness[key] = {
            "trials": int(eligible.sum()),
            "choice_nll": {
                method: float(loss["choice_nll"][eligible].mean())
                for method, loss in zip(METHODS, losses, strict=True)
            },
        }
    result = {
        "scope": "RACE materials in the same Prolific QA experiment and cohort as the OneStopQA training trials.",
        "trials": TRIALS, "articles": ARTICLES, "readers": READERS,
        "human_correct": int((selected == correct).sum()),
        "primary_metric": "choice_nll",
        "primary_contrast": "passage_question_options_minus_question_options",
        "methods": methods, "paired_contrasts": contrasts, "by_school_level": by_level,
        "by_human_response_correctness": by_response_correctness,
        "bootstrap": {
            "draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED,
            "independent_cluster_weights": ["article", "reader"],
            "trial_weight": "article multiplicity times reader multiplicity",
            "fitted_readouts_fixed": True,
        },
        "input_sha256": {
            "outcomes": digest(outcomes_path), "predictions": prediction_sha, "seal": digest(seal_path),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("outcomes", "predictions", "seal", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    evaluate(args.outcomes, args.predictions, args.seal, args.output)
