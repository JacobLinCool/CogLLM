"""Fit sentence-rating heads on frozen Qwen3.5-2B features from PLABA-EVAL."""

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
import numpy as np


TARGETS = ("processing_ease", "perceived_understanding")
PROMPT = (
    "Read the following abstract in full. Assess each focal sentence in its "
    "original context for ease of processing and understanding."
)


def read_csv(directory, filename):
    with (directory / filename).open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def load_data(directory):
    sentences = read_csv(directory, "sentences.csv")
    by_document = defaultdict(list)
    for row in sentences:
        by_document[row["stimulus_id"]].append(row)
    for rows in by_document.values():
        rows.sort(key=lambda row: int(row["sentence_index"]))

    ratings = defaultdict(list)
    for row in read_csv(directory, "annotations.csv"):
        if row["included_in_paper_sentence_analysis"] == "True":
            ratings[row["sentence_uid"]].append([int(row[name]) for name in TARGETS])

    ids = [row["sentence_uid"] for row in sentences]
    if len(set(ids)) != len(ids) or set(ids) != set(ratings):
        raise ValueError("The sentence inventory and included rating IDs differ")
    targets = np.array([np.mean(ratings[uid], axis=0) for uid in ids])
    if not np.all((1 <= targets) & (targets <= 5)):
        raise ValueError("Unexpected rating range")
    return sentences, by_document, targets, sum(map(len, ratings.values()))


def extract_features(model_path, by_document):
    model, tokenizer = load(str(model_path))
    encoder = model.language_model.model
    features = {}
    for doc_index, (doc_id, rows) in enumerate(sorted(by_document.items()), start=1):
        article = "\n".join(
            f"[{row['sentence_index']}] {row['sentence_text']}" for row in rows
        )
        tokens = tokenizer.encode(f"{PROMPT}\n\n{article}\n", add_special_tokens=False)
        positions = []
        for row in rows:
            focus = f"\nFocal sentence [{row['sentence_index']}]: {row['sentence_text']}\nAssessment:"
            tokens.extend(tokenizer.encode(focus, add_special_tokens=False))
            positions.append(len(tokens) - 1)
        hidden = encoder(mx.array([tokens]))[0, mx.array(positions), :]
        features.update(
            zip(
                (row["sentence_uid"] for row in rows),
                np.array(hidden.astype(mx.float32), copy=True),
                strict=True,
            )
        )
        if doc_index % 15 == 0 or doc_index == len(by_document):
            print(f"Encoded {doc_index}/{len(by_document)} documents", flush=True)
    return features


def fit_ridge(features, targets, penalty):
    feature_mean = features.mean(axis=0)
    target_mean = targets.mean(axis=0)
    centered = features - feature_mean
    covariance = centered @ centered.T
    covariance.flat[:: len(covariance) + 1] += penalty
    weights = centered.T @ np.linalg.solve(covariance, targets - target_mean)
    return weights, target_mean - feature_mean @ weights


def predict(features, head):
    weights, intercept = head
    return np.clip(features @ weights + intercept, 1, 5)


def length_features(lexical, mean, scale):
    return (lexical - mean) / scale


def combine_features(features, normalized_length):
    return np.column_stack((features, normalized_length / np.sqrt(normalized_length.shape[1])))


def evaluate(groups, targets, lexical, features, is_original):
    pair_ids = np.unique(groups)
    shuffled = np.random.default_rng(42).permutation(pair_ids)
    predictions = {
        name: np.empty_like(targets)
        for name in ("mean", "length", "qwen", "qwen_plus_length", "length_plus_variant")
    }
    splits = np.array_split(shuffled, 5)
    for test_pairs in splits:
        test = np.isin(groups, test_pairs)
        train = ~test
        predictions["mean"][test] = targets[train].mean(axis=0)
        length_train = lexical[train]
        normalized_length = length_features(lexical, length_train.mean(axis=0), length_train.std(axis=0))
        for name, matrix in (
            ("length", normalized_length),
            ("qwen", features),
            ("qwen_plus_length", combine_features(features, normalized_length)),
            ("length_plus_variant", np.column_stack((normalized_length, is_original))),
        ):
            predictions[name][test] = predict(
                matrix[test], fit_ridge(matrix[train], targets[train], penalty=1.0)
            )

    metrics = {}
    for name, predicted in predictions.items():
        metrics[name] = {
            target: {
                "mae": float(np.abs(predicted[:, index] - targets[:, index]).mean()),
                "pearson": (
                    float(np.corrcoef(predicted[:, index], targets[:, index])[0, 1])
                    if name != "mean" else None
                ),
            }
            for index, target in enumerate(TARGETS)
        }
    return metrics, predictions, [fold.tolist() for fold in splits]


def rank_metrics(scores, incorrect):
    order = np.argsort(-scores, kind="stable")
    cutoffs = np.r_[np.flatnonzero(np.diff(scores[order])), len(order) - 1]
    hits = np.cumsum(incorrect[order])[cutoffs]
    positives = hits[-1]
    negatives = len(order) - positives
    recall = hits / positives
    false_positive_rate = (cutoffs + 1 - hits) / negatives
    return {
        "auroc": float(np.trapezoid(np.r_[0, recall], np.r_[0, false_positive_rate])),
        "average_precision": float(np.sum(np.diff(np.r_[0, recall]) * hits / (cutoffs + 1))),
    }


def evaluate_open_book(directory, sentences, predictions):
    positions = {}
    by_document = defaultdict(list)
    for i, row in enumerate(sentences):
        key = (row["stimulus_id"], int(row["sentence_index"]))
        if key in positions:
            raise ValueError("Duplicate sentence index in an abstract")
        positions[key] = i
        by_document[row["stimulus_id"]].append(i)
    trials = [
        row for row in read_csv(directory, "mcq_responses.csv")
        if row["included_in_final_paper_analysis"] == "True"
    ]
    indices, incorrect, pairs, variants = [], [], [], []
    missing_evidence = 0
    for row in trials:
        key = (row["stimulus_id"], int(row["critical_evidence_sentence_index_this_variant"]))
        if key not in positions:
            missing_evidence += 1
            continue
        if row["is_correct"] not in {"True", "False"}:
            raise ValueError("Unexpected MCQ correctness value")
        position = positions[key]
        sentence = sentences[position]
        if (row["doc_pair_id"], row["variant"]) != (sentence["doc_pair_id"], sentence["variant"]):
            raise ValueError("MCQ evidence and its sentence disagree on document pair or variant")
        indices.append(position)
        incorrect.append(row["is_correct"] == "False")
        pairs.append(int(row["doc_pair_id"]))
        variants.append(row["variant"])
    incorrect = np.array(incorrect)
    pairs = np.array(pairs)
    variants = np.array(variants)
    if set(variants) != {"orig", "simp"}:
        raise ValueError("Unexpected set of document variants")
    unique_pairs = np.unique(pairs)
    trials_by_pair = {pair: np.flatnonzero(pairs == pair) for pair in unique_pairs}
    draws = np.random.default_rng(43).choice(unique_pairs, size=(1000, len(unique_pairs)))
    bootstrap_samples = [
        np.concatenate([trials_by_pair[pair] for pair in draw]) for draw in draws
    ]

    understanding_index = TARGETS.index("perceived_understanding")
    scores = {
        name: -ratings[indices, understanding_index]
        for name, ratings in predictions.items()
        if name != "mean"
    }
    qwen_understanding = predictions["qwen"][:, understanding_index]
    document_means = {
        doc: qwen_understanding[rows].mean() for doc, rows in by_document.items()
    }
    document_risk = np.array([-document_means[sentences[i]["stimulus_id"]] for i in indices])
    scores["qwen_document_mean"] = document_risk
    scores["qwen_centered_within_document"] = scores["qwen"] - document_risk
    scores["original_variant_only"] = (variants == "orig").astype(float)
    variant_masks = {variant: variants == variant for variant in np.unique(variants)}
    results = {}
    for name, risk in scores.items():
        point = rank_metrics(risk, incorrect)
        bootstrap_auroc = [
            rank_metrics(risk[sample], incorrect[sample])["auroc"]
            for sample in bootstrap_samples
            if np.any(incorrect[sample]) and not np.all(incorrect[sample])
        ]
        point["auroc_cluster_bootstrap_95ci"] = np.quantile(
            bootstrap_auroc, [0.025, 0.975]
        ).tolist()
        point["by_variant"] = {
            variant: rank_metrics(risk[mask], incorrect[mask])
            for variant, mask in variant_masks.items()
        }
        results[name] = point
    return {
        "trial_count": len(indices),
        "excluded_missing_variant_evidence": missing_evidence,
        "incorrect_count": int(incorrect.sum()),
        "trials_by_variant": {variant: int(mask.sum()) for variant, mask in variant_masks.items()},
        "no_skill_auroc": 0.5,
        "random_ranking_average_precision": float(incorrect.mean()),
        "score": "Negative out-of-fold perceived understanding at the designated evidence sentence. "
        "The document controls average scores across all sentences or subtract that average risk. "
        "The original-variant control assigns higher risk to every original abstract.",
        "metrics": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Local Qwen3.5-2B snapshot")
    parser.add_argument("--data", type=Path, required=True, help="PLABA-EVAL dataset directory")
    parser.add_argument("--output", type=Path, required=True, help="Directory for metrics and head")
    args = parser.parse_args()
    if not (args.model / "config.json").is_file():
        parser.error("--model must point to a local model snapshot")

    sentences, by_document, targets, rating_count = load_data(args.data)
    extracted = extract_features(args.model, by_document)
    features = np.array([extracted[row["sentence_uid"]] for row in sentences], dtype=np.float64)
    features /= np.linalg.norm(features, axis=1, keepdims=True)
    groups = np.array([int(row["doc_pair_id"]) for row in sentences])
    lexical = np.array([
        [np.log1p(len(row["sentence_text"])), np.log1p(len(row["sentence_text"].split()))]
        for row in sentences
    ])
    is_original = np.array([row["variant"] == "orig" for row in sentences])
    metrics, predictions, splits = evaluate(groups, targets, lexical, features, is_original)
    open_book = evaluate_open_book(args.data, sentences, predictions)
    weights, intercept = fit_ridge(features, targets, penalty=1.0)
    length_mean = lexical.mean(axis=0)
    length_scale = lexical.std(axis=0)
    combined = combine_features(features, length_features(lexical, length_mean, length_scale))
    combined_weights, combined_intercept = fit_ridge(combined, targets, penalty=1.0)

    args.output.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output / "plaba-eval-head.npz",
        weights=weights,
        intercept=intercept,
        combined_weights=combined_weights,
        combined_intercept=combined_intercept,
        length_mean=length_mean,
        length_scale=length_scale,
        targets=TARGETS,
    )
    results = {
        "base_checkpoint": "Qwen/Qwen3.5-2B",
        "model_snapshot": args.model.name,
        "dataset_sha256": {
            name: hashlib.sha256((args.data / name).read_bytes()).hexdigest()
            for name in ("annotations.csv", "sentences.csv", "mcq_responses.csv")
        },
        "documents": len(by_document),
        "document_pairs": int(len(np.unique(groups))),
        "sentences": len(sentences),
        "included_ratings": rating_count,
        "rating_scale": [1, 5],
        "test_doc_pair_ids_per_fold": splits,
        "metrics": metrics,
        "open_book_mcq": open_book,
    }
    (args.output / "plaba-eval-metrics.json").write_text(
        json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"sentence_ratings": metrics, "open_book_mcq": open_book}, indent=2), flush=True)


if __name__ == "__main__":
    main()
