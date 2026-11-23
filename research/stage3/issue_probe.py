"""Evaluate PLABA-EVAL checklist endorsements on locked reader-connected folds."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from .probe import _jsonl, _length_standardization, _load_plaba


ENDPOINTS = ("any_issue", "unclear_term")
EXTRA_BLOCKS = ("surprisal", "intermediate", "attention")
BASELINE = "sentence_hidden_plus_length"
ADDITIONS = {
    BASELINE: (),
    f"{BASELINE}_plus_surprisal": ("surprisal",),
    f"{BASELINE}_plus_intermediate": ("intermediate",),
    f"{BASELINE}_plus_surprisal_intermediate": ("surprisal", "intermediate"),
    f"{BASELINE}_plus_attention": ("attention",),
    f"{BASELINE}_plus_all": EXTRA_BLOCKS,
}
METHODS = ("prevalence", "length", "length_plus_variant", *ADDITIONS)
RIDGE_PENALTY = 1.0
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 43


def _reader_groups(reader_families):
    neighbors = defaultdict(set)
    for families in reader_families.values():
        for family in families:
            neighbors[family].update(families - {family})
    representatives = {}
    for family in sorted(neighbors):
        if family in representatives:
            continue
        queue = [family]
        representatives[family] = family
        while queue:
            attached = neighbors[queue.pop()] - representatives.keys()
            representatives.update(dict.fromkeys(attached, family))
            queue.extend(attached)
    return representatives


def _checklist(data_dir, stimuli):
    reader_families = defaultdict(set)
    votes = defaultdict(list)
    participants = set()
    for trial in _jsonl(data_dir / "trials.jsonl"):
        reader = trial["participant_id"]
        reader_families[reader].add(stimuli[trial["stimulus_id"]]["family_id"])
        if trial["exposure"]["task"] != "sentence_rating":
            continue
        tags = trial["observations"]["issue_tags"]
        no_issue = trial["observations"]["no_issue"]
        if tags["status"] == "unobserved" and no_issue["status"] == "unobserved":
            continue
        if tags["status"] == "missing" and no_issue["status"] == "missing":
            continue
        if (
            tags["status"] != "observed"
            or no_issue["status"] != "observed"
            or not isinstance(tags["value"], list)
            or type(no_issue["value"]) is not bool
            or no_issue["value"] == bool(tags["value"])
        ):
            raise ValueError(f"Inconsistent PLABA checklist observation: {trial['id']}")
        tags = tags["value"]
        votes[trial["target_segment_id"]].append((bool(tags), "unclear_term" in tags))
        participants.add(reader)
    return votes, _reader_groups(reader_families), len(participants)


def _feature_blocks(features_path, sentences, included_ids):
    blocks = {}
    with np.load(features_path, allow_pickle=False) as archive:
        ids = archive["ids"].tolist()
        if len(set(ids)) != len(ids) or set(ids) != set(sentences):
            raise ValueError("Feature IDs must identify every PLABA sentence exactly once")
        positions = {sentence_id: row for row, sentence_id in enumerate(ids)}
        order = [positions[sentence_id] for sentence_id in included_ids]
        for name in ("sentence_hidden", *EXTRA_BLOCKS):
            values = np.asarray(archive[name], dtype=np.float64)
            if values.ndim == 1 and name != "sentence_hidden":
                values = values[:, None]
            if values.ndim != 2 or len(values) != len(ids) or values.shape[1] == 0:
                raise ValueError(f"Expected a nonempty feature matrix indexed by sentence IDs: {name}")
            if not np.isfinite(values).all():
                raise ValueError(f"Non-finite feature values in {name}")
            blocks[name] = values[order]
    hidden = blocks["sentence_hidden"]
    if hidden.shape[1] != 2048:
        raise ValueError("Expected 2,048 Qwen sentence-end hidden dimensions")
    norm = np.linalg.norm(hidden, axis=1, keepdims=True)
    if (norm == 0).any() or not np.isfinite(norm).all():
        raise ValueError("Sentence hidden state norms must be finite and positive")
    blocks["sentence_hidden"] = hidden / norm
    return blocks


def _standardize_block(values, train):
    center = values[train].mean(axis=0)
    scale = values[train].std(axis=0)
    variable = scale > 0
    return np.divide(
        values - center, scale, out=np.zeros_like(values), where=variable,
    ) / np.sqrt(values.shape[1])


def _sigmoid(logits):
    return np.exp(-np.logaddexp(0, -logits))


def _ridge_design(train, test):
    if train.shape[1] <= len(train):
        return train, test
    # A positive L2 penalty puts the optimum in the training row space.
    eigenvalues, vectors = np.linalg.eigh(train @ train.T)
    keep = eigenvalues > 1e-10
    vectors = vectors[:, keep]
    singular = np.sqrt(eigenvalues[keep])
    return vectors * singular, (test @ train.T) @ (vectors / singular)


def _fit_logistic(train, test, positives, totals):
    design = np.column_stack((np.ones(len(train)), train))
    penalty = np.full(design.shape[1], RIDGE_PENALTY)
    penalty[0] = 0
    coefficients = np.zeros(design.shape[1])
    positive_count = positives.sum()
    negative_count = totals.sum() - positive_count
    if not positive_count or not negative_count:
        raise ValueError("Each checklist training fold needs both positive and negative responses")
    coefficients[0] = np.log(positive_count / negative_count)

    def objective(weights):
        logits = design @ weights
        nll = totals * np.logaddexp(0, logits) - positives * logits
        return float(nll.sum() + 0.5 * (penalty * weights**2).sum())

    for _ in range(100):
        probabilities = _sigmoid(design @ coefficients)
        gradient = design.T @ (totals * probabilities - positives) + penalty * coefficients
        if np.max(np.abs(gradient)) < 1e-8:
            break
        curvature = totals * probabilities * (1 - probabilities)
        hessian = (design.T * curvature) @ design + np.diag(penalty)
        step = np.linalg.solve(hessian, gradient)
        if np.linalg.norm(step) < 1e-10:
            break
        previous = objective(coefficients)
        scale = 1.0
        while objective(coefficients - scale * step) > previous + 1e-12:
            scale /= 2
            if scale < 1e-9:
                raise RuntimeError("Checklist logistic line search did not converge")
        coefficients -= scale * step
    else:
        raise RuntimeError("Checklist logistic ridge did not converge")
    return _sigmoid(coefficients[0] + test @ coefficients[1:])


def _auroc(values, labels):
    positive = values[labels, None]
    negative = values[None, ~labels]
    if not positive.size or not negative.size:
        raise ValueError("Each checklist test fold needs positive and negative responses")
    return float(((positive > negative) + 0.5 * (positive == negative)).mean())


def _within_document_pairs(document_ids):
    by_document = defaultdict(list)
    for position, document_id in enumerate(document_ids):
        by_document[document_id].append(position)
    pairs = []
    for indexes in by_document.values():
        indexes = np.asarray(indexes)
        pairs.append(indexes[np.asarray(np.triu_indices(len(indexes), k=1))])
    return np.concatenate(pairs, axis=1)


def _brier_comparisons(loss, row_cluster):
    clusters, inverse, sizes = np.unique(row_cluster, return_inverse=True, return_counts=True)
    draws = np.random.default_rng(BOOTSTRAP_SEED).integers(
        0, len(clusters), size=(BOOTSTRAP_DRAWS, len(clusters)),
    )
    totals = sizes[draws].sum(axis=1)
    results = {}
    for method in METHODS:
        if method == BASELINE:
            continue
        differences = loss[method] - loss[BASELINE]
        sums = np.bincount(inverse, weights=differences)
        samples = sums[draws].sum(axis=1) / totals
        results[method] = {
            "difference": float(differences.mean()),
            "95_ci": np.quantile(samples, [0.025, 0.975]).tolist(),
        }
    return results


def _score(predictions, label_matrix, totals, test_folds, document_ids, sentence_cluster):
    row_to_sentence = np.repeat(np.arange(len(totals)), totals)
    row_fold = test_folds[row_to_sentence]
    row_cluster = sentence_cluster[row_to_sentence]
    left, right = _within_document_pairs(document_ids)

    result = {}
    for index, name in enumerate(ENDPOINTS):
        labels = label_matrix[:, index]
        positives = np.bincount(row_to_sentence, weights=labels)
        observed_diff = positives[left] / totals[left] - positives[right] / totals[right]
        different = observed_diff != 0
        loss = {}
        methods = {}
        for method, values in predictions.items():
            scores = values[:, index]
            row_scores = scores[row_to_sentence]
            loss[method] = (row_scores - labels)**2
            folds = []
            for fold in range(5):
                keep = row_fold == fold
                folds.append({
                    "fold": fold,
                    "auroc": _auroc(row_scores[keep], labels[keep]),
                    "brier": float(loss[method][keep].mean()),
                })
            order = (scores[left] - scores[right])[different] * np.sign(observed_diff[different])
            methods[method] = {
                "pooled_brier": float(loss[method].mean()),
                "within_fold_macro_auroc": float(np.mean([fold["auroc"] for fold in folds])),
                "within_document_concordance": (
                    float(np.mean((order > 0) + 0.5 * (order == 0))) if order.size else None
                ),
                "folds": folds,
            }
        result[name] = {
            "observed_trials": len(labels),
            "positives": int(labels.sum()),
            "within_document_comparable_sentence_pairs": int(different.sum()),
            "methods": methods,
            "paired_brier_differences_minus_sentence_hidden_plus_length": _brier_comparisons(loss, row_cluster),
        }
    return result


def evaluate(data_dir: Path, features_path: Path, output_dir: Path) -> dict:
    """Fit fixed checklist models and save aggregated, out-of-fold sentence predictions."""
    stimuli, sentences, _, family_fold, _ = _load_plaba(data_dir)
    checklist, representatives, participant_count = _checklist(data_dir, stimuli)
    ids = sorted(checklist)
    blocks = _feature_blocks(features_path, sentences, ids)
    document_ids = [sentences[sentence_id][0] for sentence_id in ids]
    families = [stimuli[document_id]["family_id"] for document_id in document_ids]
    sentence_cluster = np.array([representatives[family] for family in families])
    test_folds = np.array([family_fold[family] for family in families])
    variants = [stimuli[document_id]["metadata"]["variant"] for document_id in document_ids]
    lexical = np.log1p([
        (len(sentences[sentence_id][1]), len(sentences[sentence_id][1].split()))
        for sentence_id in ids
    ])
    totals = np.array([len(checklist[sentence_id]) for sentence_id in ids])
    positives = np.array([np.sum(checklist[sentence_id], axis=0) for sentence_id in ids])
    label_matrix = np.concatenate([checklist[sentence_id] for sentence_id in ids])
    predictions = {method: np.empty(positives.shape, dtype=np.float64) for method in METHODS}

    folds = []
    for fold in range(5):
        test = test_folds == fold
        train = ~test
        if not test.any():
            raise ValueError(f"No checklist ratings in fold {fold}")
        length_mean, length_scale = _length_standardization(lexical[train])
        length = (lexical - length_mean) / length_scale
        extra = {name: _standardize_block(blocks[name], train) for name in EXTRA_BLOCKS}
        baseline = np.column_stack((blocks["sentence_hidden"], length / np.sqrt(length.shape[1])))
        designs = {"length": length, "length_plus_variant": np.column_stack((length, np.equal(variants, "orig")))}
        designs.update({
            method: np.column_stack((baseline, *(extra[name] for name in additions))) if additions else baseline
            for method, additions in ADDITIONS.items()
        })
        predictions["prevalence"][test] = positives[train].sum(axis=0) / totals[train].sum()
        for method, features in designs.items():
            train_design, test_design = _ridge_design(features[train], features[test])
            for endpoint in range(len(ENDPOINTS)):
                predictions[method][test, endpoint] = _fit_logistic(
                    train_design, test_design, positives[train, endpoint], totals[train],
                )
        folds.append({
            "fold": fold,
            "observed_trials": int(totals[test].sum()),
            "sentences": int(test.sum()),
            "positives": dict(zip(ENDPOINTS, positives[test].sum(axis=0).tolist(), strict=True)),
        })

    metrics = {
        "task": "full_text_sentence_checklist",
        "sentences": len(ids),
        "article_families": len(set(families)),
        "participants": participant_count,
        "ridge_penalty": RIDGE_PENALTY,
        "intercept_penalized": False,
        "feature_widths": {name: block.shape[1] for name, block in blocks.items()},
        "feature_scaling": "Sentence-end hidden state L2-normalized per sentence; extra blocks train-fold "
        "z-scored and divided by square root of block width. Length uses log1p character and "
        "whitespace-word counts, standardized in each train fold, then divided by square root of two "
        "in the hidden-state baselines only. Known variant is supplied only to its named control.",
        "label_definition": "Checklist endorsements only; an unchecked category is no endorsement.",
        "concordance": "Pooled within-document sentence pairs with differing empirical reader-positive rates; "
        "prediction ties receive half credit.",
        "folds": folds,
        "bootstrap": {
            "unit": "reader_connected_article_family_group",
            "groups_with_observations": len(set(sentence_cluster)),
            "draws": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED,
            "reference": BASELINE,
            "scope": "Paired Brier-loss differences on fixed out-of-fold predictions, without refitting.",
        },
        "endpoints": _score(predictions, label_matrix, totals, test_folds, document_ids, sentence_cluster),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8", newline="\n") as output:
        for row, sentence_id in enumerate(ids):
            record = {
                "sentence_id": sentence_id,
                "stimulus_id": document_ids[row],
                "family_id": families[row],
                "variant": variants[row],
                "fold": int(test_folds[row]),
                "reader_connected_group": str(sentence_cluster[row]),
                "observed_trials": int(totals[row]),
                "positives": dict(zip(ENDPOINTS, positives[row].tolist(), strict=True)),
                "predictions": {
                    name: {method: float(scores[row, index]) for method, scores in predictions.items()}
                    for index, name in enumerate(ENDPOINTS)
                },
            }
            output.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    evaluate(args.data, args.features, args.output)
