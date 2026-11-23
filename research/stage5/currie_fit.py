"""Fit family-held-out readouts for recorded Currie et al. sense judgments."""

import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np


PRIMARY_METHODS = ("prevalence", "last_only", "full")
SECONDARY_METHODS = ("critical_only", "critical")
CONDITIONS = ("near", "far")
FOLD_SEED = 0
BOOTSTRAP_SEED = 43
BOOTSTRAP_DRAWS = 10000
PROBABILITY_FLOOR = 1e-6
HIDDEN_SIZE = 2048
RIDGE_PENALTY = 1.0


def read_jsonl(path, id_field):
    rows = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            identifier = row[id_field]
            if not isinstance(identifier, str) or not identifier or identifier in rows:
                raise ValueError(f"Invalid or duplicate key in {path.name}: {identifier!r}")
            rows[identifier] = row
    return rows


def load_inputs(stimuli_path, observations_path, features_path):
    stimuli = read_jsonl(stimuli_path, "id")
    observations = read_jsonl(observations_path, "stimulus_id")
    if stimuli.keys() != observations.keys() or len(stimuli) != 48:
        raise ValueError("Expected 48 matching near/far stimulus and response-count IDs")
    ids = sorted(stimuli)
    rows = [stimuli[identifier] for identifier in ids]
    votes = []
    for identifier in ids:
        response = observations[identifier]["responses_made_sense"]
        yes, no = response["yes"], response["no"]
        if type(yes) is not int or type(no) is not int or min(yes, no) < 0 or yes + no <= 0:
            raise ValueError(f"Invalid original sense judgment counts for {identifier}")
        votes.append((yes, no))
    with np.load(features_path, allow_pickle=False) as archive:
        if archive["ids"].tolist() != ids:
            raise ValueError("Feature rows must match the sorted stimulus IDs")
        features = {}
        for key in (*PRIMARY_METHODS[1:], *SECONDARY_METHODS):
            matrix = archive[key]
            if matrix.shape != (len(ids), HIDDEN_SIZE) or matrix.dtype.kind != "f" or not np.isfinite(matrix).all():
                raise ValueError(f"Invalid Qwen feature matrix for {key}")
            matrix = matrix.astype(np.float64)
            norms = np.linalg.norm(matrix, axis=1)
            if not np.all((norms > 0) & np.isfinite(norms)):
                raise ValueError(f"Unexpected hidden-state row norm in {key}")
            features[key] = matrix / norms[:, None]
    return rows, np.array(votes, dtype=np.float64), features


def families_of(rows):
    families = defaultdict(dict)
    genres = {}
    for i, row in enumerate(rows):
        family, genre, condition = row["family_id"], row["genre"], row["condition"]
        if not isinstance(family, str) or not family or not isinstance(genre, str) or not genre:
            raise ValueError("Family and genre are required for fold assignment")
        if condition not in CONDITIONS or condition in families[family]:
            raise ValueError(f"Invalid or duplicate condition in family {family}")
        if genres.setdefault(family, genre) != genre:
            raise ValueError(f"Genre changes within family {family}")
        families[family][condition] = i
    for family, variants in families.items():
        if set(variants) != set(CONDITIONS):
            raise ValueError(f"Incomplete family {family}")
    if Counter(genres.values()) != {"narrative": 12, "expository": 12}:
        raise ValueError("Expected 12 narrative and 12 expository near/far pairs")
    return families, genres


def fold_assignments(genres, names, n_folds):
    strata = defaultdict(list)
    for name in sorted(names):
        strata[genres[name]].append(name)
    rng = np.random.default_rng(FOLD_SEED)
    assignment = {}
    start = 0
    for _, members in sorted(strata.items()):
        assignment.update({name: (start + i) % n_folds for i, name in enumerate(rng.permutation(members).tolist())})
        start += len(members)
    return assignment


def ridge(train, vote_counts, test, device):
    if device == "cuda":
        import torch

        train, vote_counts, test = (torch.as_tensor(array, device=device) for array in (train, vote_counts, test))
        identity = torch.eye(len(train), dtype=torch.float64, device=device)
        linalg = torch.linalg
    else:
        identity = np.eye(len(train))
        linalg = np.linalg
    totals = vote_counts.sum(axis=1)
    weights = totals / totals.mean()
    target = vote_counts[:, 0] / totals
    x_mean = (train * weights[:, None]).sum(axis=0) / weights.sum()
    y_mean = (target * weights).sum() / weights.sum()
    root = weights ** 0.5
    x = (train - x_mean) * root[:, None]
    y = (target - y_mean) * root
    gram = x @ x.T + RIDGE_PENALTY * identity
    slope = x.T @ linalg.solve(gram, y)
    residual = x.T @ (x @ slope - y) + RIDGE_PENALTY * slope
    if float(linalg.norm(residual)) / max(float(linalg.norm(x.T @ y)), 1e-12) > 1e-8:
        raise ArithmeticError("Weighted ridge solution did not satisfy its normal equation")
    estimates = (test - x_mean) @ slope + y_mean
    if device == "cuda":
        estimates = estimates.cpu().numpy()
    return estimates.clip(0, 1)


def cross_fit(rows, votes, features, methods, assignment, n_folds, device):
    folds = np.array([assignment.get(row["family_id"], -1) for row in rows])
    predictions = np.full((len(methods), len(rows)), np.nan)
    if set(folds[folds >= 0]) != set(range(n_folds)):
        raise ValueError("Every planned fold must contain eligible families")
    for fold in range(n_folds):
        training = (folds != fold) & (folds >= 0)
        testing = folds == fold
        for i, method in enumerate(methods):
            if method == "prevalence":
                predictions[i, testing] = votes[training, 0].sum() / votes[training].sum()
            else:
                x = features[method]
                predictions[i, testing] = ridge(x[training], votes[training], x[testing], device)
    if not np.all(np.isfinite(predictions[:, folds >= 0])):
        raise ValueError("Missing predictions for an eligible row")
    return predictions, folds


def metric_summary(votes, predictions, groups, methods):
    totals = votes.sum(axis=1)
    yes, no = votes.T
    bounded = predictions.clip(PROBABILITY_FLOOR, 1 - PROBABILITY_FLOOR)
    losses = {
        "brier": yes * (1 - predictions) ** 2 + no * predictions ** 2,
        "cross_entropy_nats": -(yes * np.log(bounded) + no * np.log1p(-bounded)),
    }
    group_index = np.unique(groups, return_inverse=True)[1]
    group_totals = np.bincount(group_index, weights=totals)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(len(group_totals), size=(BOOTSTRAP_DRAWS, len(group_totals)))
    denominators = group_totals[draws].sum(axis=1)
    result = {}
    for metric, matrix in losses.items():
        errors = matrix.sum(axis=1) / totals.sum()
        grouped_errors = np.array([np.bincount(group_index, weights=row) for row in matrix])
        resampled = grouped_errors[:, draws].sum(axis=2) / denominators
        result[metric] = {
            "by_method": dict(zip(methods, errors.tolist(), strict=True)),
            "paired_family_differences": {
                f"{methods[j]}_minus_{methods[i]}": {
                    "difference": float(errors[j] - errors[i]),
                    "family_bootstrap_95_ci": np.quantile(resampled[j] - resampled[i], [0.025, 0.975]).tolist(),
                }
                for i, j in itertools.combinations(range(len(methods)), 2)
            },
        }
    return result


def condition_summary(votes, predictions, conditions, methods):
    results = {}
    for condition in sorted(set(conditions)):
        include = conditions == condition
        totals = votes[include].sum(axis=1)
        probability = predictions[:, include] @ totals / totals.sum()
        results[condition] = {
            "stimuli": int(include.sum()), "human_votes": int(totals.sum()),
            "human_p_yes": float(votes[include, 0].sum() / totals.sum()),
            "predicted_p_yes": dict(zip(methods, probability.tolist(), strict=True)),
        }
    return results


def pair_summary(votes, predictions, families, methods):
    paired = [(row["near"], row["far"]) for row in families.values()]
    near, far = np.array(paired).T
    votes_yes = votes[:, 0] / votes.sum(axis=1)
    human = votes_yes[near] - votes_yes[far]
    predicted = predictions[:, near] - predictions[:, far]
    nonzero = human != 0
    ranked = {}
    if nonzero.any():
        compared = predicted[:, nonzero]
        concordance = np.where(np.abs(compared) <= 1e-10, 0.5, compared * human[nonzero] > 0)
        ranked = dict(zip(methods, concordance.mean(axis=1).tolist(), strict=True))
    return {
        "contrast": "near minus far, equal weight per material family",
        "families": len(paired), "non_tied_human_families": int(nonzero.sum()),
        "human_difference": float(human.mean()),
        "predicted_difference": dict(zip(methods, predicted.mean(axis=1).tolist(), strict=True)),
        "direction_concordance_model_ties_half_credit": ranked,
    }


def sentence_texts(row):
    text = row["text"]
    sentences = []
    cursor = 0
    for segment in row["segments"]:
        start, end = segment["start"], segment["end"]
        if not cursor <= start < end <= len(text) or text[cursor:start].strip():
            raise ValueError(f"Invalid sentence boundaries for {row['id']}")
        sentences.append(text[start:end])
        cursor = end
    if not sentences or text[cursor:].strip():
        raise ValueError(f"Sentence list does not cover the passage for {row['id']}")
    return sentences


def critical_reorder_families(rows, features, families):
    selected, exclusions = {}, {}
    for name, pair in families.items():
        parts = []
        for condition in ("near", "far"):
            row = rows[pair[condition]]
            sentences = sentence_texts(row)
            critical, support = row["critical_index"], row["support_index"]
            if type(critical) is not int or type(support) is not int or not 0 <= support < critical < len(sentences):
                raise ValueError(f"Incorrect critical and support order in {row['id']}")
            parts.append((sentences, critical, support))
        (near, nc, ns), (far, fc, fs) = parts
        reasons = []
        if near == far or Counter(near) != Counter(far):
            reasons.append("sentence_multiset_not_exact_reorder")
        if near[nc] != far[fc]:
            reasons.append("critical_sentence_differs")
        if near[ns] != far[fs] or nc - ns >= fc - fs:
            reasons.append("support_differs_or_distance_not_near_before_far")
        if reasons:
            exclusions[name] = reasons
            continue
        if not np.array_equal(features["critical_only"][pair["near"]], features["critical_only"][pair["far"]]):
            raise ValueError(f"Identical isolated critical sentences require identical features in {name}")
        selected[name] = pair
    return selected, exclusions


def file_digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def evaluate(stimuli_path, observations_path, features_path, output, device):
    hardware = "CPU"
    if device == "cuda":
        import torch

        if torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != "NVIDIA L4":
            raise RuntimeError("CUDA fitting requires exactly one NVIDIA L4")
        hardware = torch.cuda.get_device_name()
    rows, votes, features = load_inputs(stimuli_path, observations_path, features_path)
    families, genres = families_of(rows)
    group_by_row = np.array([row["family_id"] for row in rows])
    conditions = np.array([row["condition"] for row in rows])
    primary_assignment = fold_assignments(genres, families, 6)
    for fold in range(6):
        names = [name for name, assigned in primary_assignment.items() if assigned == fold]
        if Counter(genres[name] for name in names) != {"narrative": 2, "expository": 2}:
            raise ValueError("Each primary fold requires two narrative and two expository pairs")
    predicted, primary_fold = cross_fit(rows, votes, features, PRIMARY_METHODS, primary_assignment, 6, device)
    selected, exclusions = critical_reorder_families(rows, features, families)
    secondary = {
        "scope": "Critical-sentence end readouts of the eventual whole-passage sense judgment in exact sentence reorder pairs.",
        "eligible_families": len(selected), "excluded_families": exclusions,
        "status": "insufficient_exact_reorders_for_five_folds",
    }
    if len(selected) >= 5:
        secondary_assignment = fold_assignments(genres, selected, 5)
        critical_pred, critical_fold = cross_fit(rows, votes, features, SECONDARY_METHODS, secondary_assignment, 5, device)
        included = critical_fold >= 0
        secondary.update({
            "status": "evaluated", "family_to_fold": secondary_assignment,
            "metrics": metric_summary(votes[included], critical_pred[:, included], group_by_row[included], SECONDARY_METHODS),
            "near_far": pair_summary(votes, critical_pred, selected, SECONDARY_METHODS),
        })
        critical_pred, critical_fold = critical_pred[:, included], critical_fold[included]
    else:
        included = np.zeros(len(rows), dtype=bool)
        critical_pred, critical_fold = np.empty((len(SECONDARY_METHODS), 0)), np.empty(0, dtype=int)
    result = {
        "input_sha256": {
            key: file_digest(path) for key, path in (
                ("stimuli", stimuli_path), ("observations", observations_path), ("features", features_path),
            )
        },
        "primary": {
            "scope": "Estimation of a recorded yes vote after a complete passage for source-ID-aligned near/far materials.",
            "human_votes": int(votes.sum()), "stimuli": len(rows), "family_to_fold": primary_assignment,
            "metrics": metric_summary(votes, predicted, group_by_row, PRIMARY_METHODS),
            "condition_probabilities": condition_summary(votes, predicted, conditions, PRIMARY_METHODS),
            "near_far": pair_summary(votes, predicted, families, PRIMARY_METHODS),
        },
        "secondary": secondary,
        "fitting": {
            "hardware": hardware, "dtype": "float64", "ridge_penalty": RIDGE_PENALTY, "feature_rows": "L2 normalized",
            "training_weights": "Each cell's vote count divided by mean count of training cells.",
            "probability_floor_for_log_loss_only": PROBABILITY_FLOOR,
            "fold_seed": FOLD_SEED,
            "bootstrap": {"unit": "material family", "paired": True, "seed": BOOTSTRAP_SEED, "draws": BOOTSTRAP_DRAWS},
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "predictions.npz", ids=[row["id"] for row in rows],
        primary_methods=PRIMARY_METHODS, primary=predicted, primary_fold=primary_fold,
        secondary_methods=SECONDARY_METHODS, secondary_ids=[row["id"] for row, keep in zip(rows, included, strict=True) if keep],
        secondary=critical_pred, secondary_fold=critical_fold,
    )
    (output / "scores.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("stimuli", "observations", "features", "output"):
        parser.add_argument(f"--{key}", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    evaluate(args.stimuli, args.observations, args.features, args.output, args.device)
