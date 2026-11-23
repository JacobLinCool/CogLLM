"""Replay the archived OneStop heads and seal outcome-free Stage 6 forecasts."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from ..joint.onestop_probe import METHODS, RIDGE, align_features, fit_weights, row_losses


ONESTOP_DIGESTS = {
    "design": "05e41dbfac748869dd696ebd671bdb17f8da7726a18c42b01d3b48c32b5ef680",
    "features": "5fc0632af94a60f1be672910a81b1378c26ffab9764d5ab86d9cb48c3982ecaa",
    "reference_predictions": "8ccd7af935dd26e8838f5ca1d932112c00b99b19a7b071c2e6383f3cbb129f08",
}
N_FOLDS = 5
N_OPTIONS = 4
N_RACE_TRIALS = 1940
HIDDEN_SIZE = 2048
REPLAY_TOLERANCE = 1e-5
RACE_DESIGN_KEYS = {"id", "full_id", "question_id", "displayed_order"}


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, record):
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_archive(path, **arrays):
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def rebuild(design, features, reference, output):
    # The original fit ran on L4. Only the numerical replay/verification runs here.
    import torch

    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    actual_digests = {
        name: digest(path) for name, path in (
            ("design", design), ("features", features), ("reference_predictions", reference),
        )
    }
    if actual_digests != ONESTOP_DIGESTS:
        raise ValueError("Inputs differ from the recorded OneStopQA baseline")
    with np.load(design, allow_pickle=False) as archive:
        ids = archive["onestop_ids"]
        question_ids = archive["onestop_question_only_ids"]
        folds = archive["onestop_folds"]
        counts = archive["onestop_counts"]
        correct = archive["onestop_correct"]
    with np.load(features, allow_pickle=False) as archive:
        feature_ids, hidden = archive["ids"], archive["hidden"]
        x = {
            METHODS[1]: align_features(question_ids, feature_ids, hidden),
            METHODS[2]: align_features(ids, feature_ids, hidden),
        }
    with np.load(reference, allow_pickle=False) as archive:
        if not np.array_equal(archive["ids"], ids) or tuple(archive["methods"]) != METHODS:
            raise ValueError("Reference IDs or methods differ from the original OneStopQA design")
        l4 = archive["predictions"]

    rebuilt = np.empty_like(l4)
    weights = {method: [] for method in METHODS}
    fits = []
    for fold in range(N_FOLDS):
        train = folds != fold
        prevalence = counts[train].sum(axis=0)
        prior = prevalence / prevalence.sum()
        rebuilt[0, ~train] = prior
        weights[METHODS[0]].append(prior)
        for method_index, method in enumerate(METHODS[1:], start=1):
            weight, record = fit_weights(torch, x[method], counts, train, fold, "cpu")
            test = torch.as_tensor(x[method][~train], dtype=torch.float64)
            rebuilt[method_index, ~train] = (test @ weight[:-1] + weight[-1]).softmax(dim=-1).numpy()
            weights[method].append(weight.numpy())
            fits.append({"fold": fold, "method": method, **record})
    difference = np.abs(rebuilt - l4)
    max_error = float(difference.max())
    if not np.isfinite(difference).all() or max_error > REPLAY_TOLERANCE:
        raise ArithmeticError(f"OneStopQA CPU replay exceeded {REPLAY_TOLERANCE}: {max_error}")

    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "weights.npz"
    oof = output / "onestop-replayed-oof.npz"
    write_archive(checkpoint, folds=np.arange(N_FOLDS), **{k: np.stack(v) for k, v in weights.items()})
    write_archive(oof, ids=ids, methods=np.asarray(METHODS), predictions=rebuilt)
    write_json(output / "rebuild.json", {
        "scope": "OneStopQA five-fold replay with archived inputs, before any RACE prediction or scoring.",
        "input_sha256": actual_digests,
        "weights_sha256": digest(checkpoint),
        "oof_replay_sha256": digest(oof),
        "torch_version": torch.__version__,
        "fit_dtype": "float64",
        "ridge_penalty": RIDGE,
        "fold_count": N_FOLDS,
        "cpu_vs_l4_max_absolute_probability_error": max_error,
        "cpu_vs_l4_max_error_by_method": dict(zip(METHODS, difference.max(axis=(1, 2)).tolist(), strict=True)),
        "replay_tolerance": REPLAY_TOLERANCE,
        "oof_replay_choice_nll": {
            method: float(row_losses(p, counts, correct)["choice_nll"].sum() / counts.sum())
            for method, p in zip(METHODS, rebuilt, strict=True)
        },
        "diagnostics": fits,
    })
    print(f"Replayed all {len(ids)} OneStopQA out-of-fold estimates; CPU/L4 max deviation {max_error:.3g}")


def race_rows(path):
    with path.open(encoding="utf-8") as source:
        rows = [json.loads(line) for line in source]
    if len(rows) != N_RACE_TRIALS or any(row.keys() != RACE_DESIGN_KEYS for row in rows):
        raise ValueError("Expected the outcome-free RACE trial inventory")
    if any(
        not isinstance(row[key], str) or not row[key]
        for row in rows for key in ("id", "full_id", "question_id")
    ) or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("RACE trial and prompt identifiers must be valid and trials unique")
    if any(
        not isinstance(row["displayed_order"], list)
        or any(type(index) is not int for index in row["displayed_order"])
        or sorted(row["displayed_order"]) != list(range(N_OPTIONS))
        for row in rows
    ):
        raise ValueError("RACE inputs must preserve valid displayed option permutations")
    return sorted(rows, key=lambda row: row["id"])


def softmax(logits):
    numerator = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return numerator / numerator.sum(axis=-1, keepdims=True)


def predict(design, features, weights_path, rebuild_path, output):
    sealed = json.loads(rebuild_path.read_text(encoding="utf-8"))
    weight_digest = digest(weights_path)
    if (
        sealed["input_sha256"] != ONESTOP_DIGESTS or sealed["weights_sha256"] != weight_digest
        or not 0 <= sealed["cpu_vs_l4_max_absolute_probability_error"] <= REPLAY_TOLERANCE
    ):
        raise ValueError("RACE application requires the verified OneStopQA reconstruction")

    rows = race_rows(design)
    full = [row["full_id"] for row in rows]
    questions = [row["question_id"] for row in rows]
    with np.load(features, allow_pickle=False) as archive:
        ids, hidden = archive["ids"], archive["hidden"]
        full_ids, question_ids = set(full), set(questions)
        if full_ids & question_ids or set(ids) != full_ids | question_ids:
            raise ValueError("Feature inventory differs from the two authorized RACE exposures")
        if hidden.dtype != np.float32:
            raise ValueError("Expected original FP32 Qwen feature archive")
        x = {
            METHODS[1]: align_features(questions, ids, hidden).astype(np.float64),
            METHODS[2]: align_features(full, ids, hidden).astype(np.float64),
        }
    with np.load(weights_path, allow_pickle=False) as archive:
        if not np.array_equal(archive["folds"], np.arange(N_FOLDS)):
            raise ValueError("The weight archive must contain the original five folds")
        prior = archive[METHODS[0]]
        if prior.dtype != np.float64 or prior.shape != (N_FOLDS, N_OPTIONS):
            raise ValueError("Unexpected position prior in the OneStopQA checkpoint")
        forecasts = [np.broadcast_to(prior.mean(axis=0), (len(rows), N_OPTIONS))]
        for method in METHODS[1:]:
            weights = archive[method]
            if weights.dtype != np.float64 or weights.shape != (N_FOLDS, HIDDEN_SIZE + 1, N_OPTIONS):
                raise ValueError(f"Unexpected coefficient dimensions for {method}")
            forecasts.append(softmax(x[method] @ weights[:, :-1] + weights[:, -1, None]).mean(axis=0))
    forecasts = np.stack(forecasts)
    if not np.all(np.isfinite(forecasts) & (forecasts > 0)) or not np.allclose(
        forecasts.sum(axis=-1), 1, rtol=0, atol=1e-12,
    ):
        raise ArithmeticError("External application produced invalid option probabilities")

    output.mkdir(parents=True, exist_ok=True)
    predictions = output / "predictions.npz"
    write_archive(
        predictions, ids=np.array([row["id"] for row in rows]), methods=np.asarray(METHODS), predictions=forecasts,
    )
    write_json(output / "verification.json", {
        "scope": "RACE outcomes excluded. Predict in displayed order and average five original OneStopQA folds in probability space.",
        "predictions_sha256": digest(predictions),
        "input_sha256": {
            "outcome_free_design": digest(design), "race_l4_features": digest(features),
            "onestop_weights": weight_digest, "onestop_rebuild_verification": digest(rebuild_path),
        },
        "prediction_trials": len(rows),
        "feature_count_by_method": {METHODS[1]: len(question_ids), METHODS[2]: len(full_ids)},
        "one_stop_cpu_vs_l4_max_absolute_probability_error": sealed["cpu_vs_l4_max_absolute_probability_error"],
    })
    print(f"Sealed outcome-free RACE forecasts for {len(rows)} trials")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    rebuild_parser = subparsers.add_parser("rebuild")
    predict_parser = subparsers.add_parser("predict")
    for area, fields in (
        (rebuild_parser, ("design", "features", "reference", "output")),
        (predict_parser, ("design", "features", "weights", "rebuild-verification", "output")),
    ):
        for field in fields:
            area.add_argument(f"--{field}", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "rebuild":
        rebuild(args.design, args.features, args.reference, args.output)
    else:
        predict(args.design, args.features, args.weights, args.rebuild_verification, args.output)
