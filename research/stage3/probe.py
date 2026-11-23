"""Evaluate a frozen PLABA-EVAL sentence readout on the locked development folds."""

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


TARGETS = ("processing_ease", "perceived_understanding")
METHODS = ("mean", "length", "length_plus_variant", "qwen", "qwen_plus_length")
RIDGE_PENALTY = 1.0
BOOTSTRAP_DRAWS = 1000
BOOTSTRAP_SEED = 43


def _jsonl(path):
    with path.open(encoding="utf-8") as source:
        for line in source:
            yield json.loads(line)


def _load_plaba(data_dir):
    stimuli = {}
    sentences = {}
    for stimulus in _jsonl(data_dir / "stimuli.jsonl"):
        stimulus_id = stimulus["id"]
        if stimulus_id in stimuli:
            raise ValueError(f"Duplicate PLABA stimulus {stimulus_id}")
        if stimulus["metadata"]["variant"] not in {"orig", "simp"}:
            raise ValueError(f"Unknown PLABA document variant for {stimulus_id}")
        stimuli[stimulus_id] = stimulus
        for segment in stimulus["segments"]:
            segment_id = segment["id"]
            if segment_id in sentences:
                raise ValueError(f"Duplicate PLABA segment {segment_id}")
            start, end, text = segment["start"], segment["end"], stimulus["text"]
            if not 0 <= start < end <= len(text):
                raise ValueError(f"Invalid PLABA segment offsets for {segment_id}")
            sentences[segment_id] = (stimulus_id, text[start:end])

    split = json.loads((data_dir / "splits.json").read_text(encoding="utf-8"))
    folds = split["folds"]
    if (
        split["grouping"] != "reader_connected_stimulus_families"
        or sorted(row["fold"] for row in folds) != list(range(5))
    ):
        raise ValueError("Expected the five Stage 2 PLABA reader-connected folds")
    family_fold = {family: row["fold"] for row in folds for family in row["families"]}
    if (
        len(family_fold) != sum(len(row["families"]) for row in folds)
        or set(family_fold) != {stimulus["family_id"] for stimulus in stimuli.values()}
        or set(family_fold.values()) != set(range(5))
    ):
        raise ValueError("PLABA article families must be covered once by nonempty test folds")

    participant_folds = defaultdict(set)
    fold_trial_counts = Counter()
    ratings = defaultdict(list)
    for trial in _jsonl(data_dir / "trials.jsonl"):
        stimulus = stimuli[trial["stimulus_id"]]
        fold = family_fold[stimulus["family_id"]]
        fold_trial_counts[fold] += 1
        participant_id = trial["participant_id"]
        if participant_id is None:
            raise ValueError("Cannot verify PLABA reader separation for a trial without a reader ID")
        participant_folds[participant_id].add(fold)

        if trial["exposure"]["task"] != "sentence_rating":
            continue
        if trial["exposure"]["mode"] != "full_text" or trial["exposure"]["visible_end"] != len(stimulus["text"]):
            raise ValueError(f"Expected full document exposure for sentence rating {trial['id']}")
        segment_id = trial["target_segment_id"]
        if segment_id not in sentences or sentences[segment_id][0] != stimulus["id"]:
            raise ValueError(f"Rating references a segment outside its stimulus: {trial['id']}")
        observations = [trial["observations"][name] for name in TARGETS]
        if any(
            row["status"] != "observed" or type(row["value"]) is not int or not 1 <= row["value"] <= 5
            for row in observations
        ):
            raise ValueError(f"Expected observed PLABA ratings of 1–5: {trial['id']}")
        ratings[segment_id].append([row["value"] for row in observations])

    crossing_readers = sum(len(folds) > 1 for folds in participant_folds.values())
    if crossing_readers:
        raise ValueError(f"{crossing_readers} PLABA readers occur in multiple development folds")
    if any(row["trials"] != fold_trial_counts[row["fold"]] for row in folds):
        raise ValueError("PLABA split trial counts disagree with the Stage 2 records")
    if set(ratings) != set(sentences):
        raise ValueError("The PLABA sentence inventory and observed rating coverage differ")

    return stimuli, sentences, ratings, family_fold, participant_folds


def _fit_ridge(features, targets):
    feature_mean = features.mean(axis=0)
    target_mean = targets.mean(axis=0)
    centered = features - feature_mean
    residual = targets - target_mean
    if centered.shape[1] <= centered.shape[0]:
        gram = centered.T @ centered
        gram.flat[::len(gram) + 1] += RIDGE_PENALTY
        weights = np.linalg.solve(gram, centered.T @ residual)
    else:
        gram = centered @ centered.T
        gram.flat[::len(gram) + 1] += RIDGE_PENALTY
        weights = centered.T @ np.linalg.solve(gram, residual)
    return weights, target_mean - feature_mean @ weights


def _length_standardization(lexical_train):
    mean = lexical_train.mean(axis=0)
    scale = lexical_train.std(axis=0)
    return mean, np.where(scale > 0, scale, 1.0)


def _target_values(values):
    return {name: float(value) for name, value in zip(TARGETS, values, strict=True)}


def _bootstrap_differences(losses, families):
    unique_families, by_row, family_sizes = np.unique(families, return_inverse=True, return_counts=True)
    draws = np.random.default_rng(BOOTSTRAP_SEED).integers(
        0, len(unique_families), size=(BOOTSTRAP_DRAWS, len(unique_families)),
    )
    sample_sizes = family_sizes[draws].sum(axis=1, keepdims=True)
    results = {}
    for candidate, reference in (
        ("qwen", "length"),
        ("qwen_plus_length", "length"),
        ("qwen_plus_length", "length_plus_variant"),
    ):
        delta = losses[candidate] - losses[reference]
        by_family = np.zeros((len(unique_families), len(TARGETS)))
        np.add.at(by_family, by_row, delta)
        intervals = np.quantile(by_family[draws].sum(axis=1) / sample_sizes, [0.025, 0.975], axis=0)
        results[f"{candidate}_minus_{reference}"] = {
            name: {"mae_difference": float(point), "95_ci": [float(low), float(high)]}
            for name, point, low, high in zip(TARGETS, delta.mean(axis=0), *intervals, strict=True)
        }
    return results


def evaluate(data_dir: Path, feature_ids: list[str], features: np.ndarray, output_dir: Path) -> dict:
    """Fit fixed ridge heads and persist sentence-level out-of-fold predictions."""
    stimuli, sentences, ratings, family_fold, participant_folds = _load_plaba(data_dir)
    feature_ids = list(feature_ids)
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or len(features) != len(feature_ids) or features.shape[1] == 0:
        raise ValueError("Expected one nonempty feature vector per PLABA sentence ID")
    if len(set(feature_ids)) != len(feature_ids) or set(feature_ids) != set(sentences):
        raise ValueError("Feature IDs must cover the PLABA sentence inventory exactly once")
    feature_index = {sentence_id: row for row, sentence_id in enumerate(feature_ids)}
    sentence_ids = list(sentences)
    features = features[[feature_index[sentence_id] for sentence_id in sentence_ids]]
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    if not np.isfinite(features).all() or not np.isfinite(norms).all() or (norms == 0).any():
        raise ValueError("Frozen features must be finite with positive finite L2 norms")
    features = features / norms

    stimulus_ids = [sentences[sentence_id][0] for sentence_id in sentence_ids]
    families = np.array([stimuli[stimulus_id]["family_id"] for stimulus_id in stimulus_ids])
    is_original = np.array([stimuli[stimulus_id]["metadata"]["variant"] == "orig" for stimulus_id in stimulus_ids])
    test_folds = np.array([family_fold[family] for family in families])
    lexical = np.log1p([
        (len(sentence_text), len(sentence_text.split()))
        for _, sentence_text in sentences.values()
    ])
    targets = np.array([np.mean(ratings[sentence_id], axis=0) for sentence_id in sentence_ids])
    predictions = {name: np.empty_like(targets) for name in METHODS}

    report_folds = []
    for fold in range(5):
        test = test_folds == fold
        train = ~test
        if not test.any():
            raise ValueError(f"PLABA fold {fold} has no held-out ratings")
        length_mean, length_scale = _length_standardization(lexical[train])
        lengths = (lexical - length_mean) / length_scale
        predictions["mean"][test] = targets[train].mean(axis=0)
        for name, matrix in (
            ("length", lengths),
            ("length_plus_variant", np.column_stack((lengths, is_original))),
            ("qwen", features),
            ("qwen_plus_length", np.column_stack((features, lengths / np.sqrt(lengths.shape[1])))),
        ):
            weights, intercept = _fit_ridge(matrix[train], targets[train])
            predictions[name][test] = np.clip(matrix[test] @ weights + intercept, 1, 5)
        report_folds.append({
            "fold": fold,
            "test_families": sorted(family for family, value in family_fold.items() if value == fold),
            "test_participants": sum(fold in reader_folds for reader_folds in participant_folds.values()),
            "test_sentences": int(test.sum()),
        })

    losses = {name: np.abs(values - targets) for name, values in predictions.items()}
    metrics = {
        "task": "full_text_sentence_rating",
        "sentences": len(sentence_ids),
        "sentence_rating_trials": sum(map(len, ratings.values())),
        "article_families": len(family_fold),
        "participants": len(participant_folds),
        "article_families_spanning_folds": 0,
        "participants_spanning_folds": 0,
        "ridge_penalty": RIDGE_PENALTY,
        "feature_normalization": "per_sentence_l2",
        "feature_width": int(features.shape[1]),
        "folds": report_folds,
        "pooled_mae": {name: _target_values(loss.mean(axis=0)) for name, loss in losses.items()},
        "paired_mae_differences": _bootstrap_differences(losses, families),
        "bootstrap": {
            "unit": "source_article_family",
            "draws": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED,
            "scope": "Fixed out-of-fold predictions; does not resample overlapping readers or refit heads.",
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8", newline="\n") as output:
        for row, (sentence_id, stimulus_id) in enumerate(zip(sentence_ids, stimulus_ids, strict=True)):
            record = {
                "sentence_id": sentence_id,
                "stimulus_id": stimulus_id,
                "family_id": str(families[row]),
                "variant": stimuli[stimulus_id]["metadata"]["variant"],
                "fold": int(test_folds[row]),
                "rating_count": len(ratings[sentence_id]),
                "targets": _target_values(targets[row]),
                "predictions": {name: _target_values(values[row]) for name, values in predictions.items()},
            }
            output.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")

    # The exported head is for subsequent use. The held-out metrics use only fold-specific heads.
    length_mean, length_scale = _length_standardization(lexical)
    lengths = (lexical - length_mean) / length_scale
    weights, intercept = _fit_ridge(
        np.column_stack((features, lengths / np.sqrt(lengths.shape[1]))), targets,
    )
    np.savez(
        output_dir / "head.npz",
        combined_weights=weights,
        combined_intercept=intercept,
        length_mean=length_mean,
        length_scale=length_scale,
        targets=np.array(TARGETS),
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    return metrics
