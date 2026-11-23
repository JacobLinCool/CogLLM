"""Compare independent and shared adapters on frozen CogLLM features."""

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


TASKS = ("plaba", "zhan", "onestop")
METHODS = ("linear", "independent_adapter", "shared_adapter")
RATINGS = ("processing_ease", "perceived_understanding")
ISSUES = ("any_issue", "unclear_term")
HIDDEN_WIDTH = 2048
ADAPTER_RANK = 32
N_FOLDS = 5
UPDATES = 160
SEED = 1729
LR = 0.006
WEIGHT_DECAY = 0.01


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def save_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def align_features(path, ids, key, anchors, allow_extra):
    with np.load(path, allow_pickle=False) as cache:
        cached_ids = cache["ids"].tolist()
        raw = cache[key]
    indexed = {identifier: position for position, identifier in enumerate(cached_ids)}
    if (
        len(indexed) != len(cached_ids)
        or raw.shape != (len(cached_ids), anchors * HIDDEN_WIDTH)
        or (not allow_extra and len(cached_ids) != len(ids))
        or any(identifier not in indexed for identifier in ids)
    ):
        raise ValueError(f"Feature IDs or dimensions do not match the frozen design: {path.name}")
    values = raw[[indexed[identifier] for identifier in ids]].astype(np.float32)
    values = values.reshape(len(ids), anchors, HIDDEN_WIDTH)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    if not np.isfinite(norms).all() or (norms == 0).any():
        raise ValueError(f"Features must be finite and nonzero: {path.name}")
    return values / norms


def load_counts(arrays, key, shape, folds, require_each_row=True):
    counts = arrays[key]
    if counts.shape != shape or not np.issubdtype(counts.dtype, np.integer) or (counts < 0).any():
        raise ValueError(f"Invalid observed count tensor {key}. Expected {shape}")
    totals = counts.sum(axis=-1)
    if require_each_row and (totals <= 0).any():
        raise ValueError(f"An observation in {key} is missing its required counts")
    if any((totals[folds == fold].sum(axis=0) <= 0).any() for fold in range(N_FOLDS)):
        raise ValueError(f"Every fold needs observations of each {key} endpoint")
    return counts


def read_data(design, feature_paths):
    datasets = {}
    with np.load(design, allow_pickle=False) as arrays:
        for task in TASKS:
            ids = arrays[f"{task}_ids"]
            folds = arrays[f"{task}_folds"]
            if (
                ids.ndim != 1 or ids.dtype.kind != "U" or len(set(ids.tolist())) != len(ids)
                or folds.shape != ids.shape or not np.issubdtype(folds.dtype, np.integer)
                or set(folds.tolist()) != set(range(N_FOLDS))
            ):
                raise ValueError(f"Invalid {task} IDs or fixed five-fold assignments")
            n = len(ids)
            plaba = task == "plaba"
            states = align_features(
                feature_paths[task], ids, "features" if plaba else "hidden",
                2 if plaba else 1, task == "onestop",
            )
            datasets[task] = {
                "ids": ids,
                "folds": folds,
                "states": states,
            }
            data = datasets[task]
            if plaba:
                length = arrays["plaba_length"]
                if length.shape != (n, 2) or not np.isfinite(length).all():
                    raise ValueError("Expected two finite PLABA length features per sentence")
                data["length"] = length.astype(np.float32)
                data["ratings"] = load_counts(arrays, "plaba_rating_counts", (n, 2, 5), folds)
                data["issues"] = load_counts(arrays, "plaba_issue_counts", (n, 2, 2), folds, require_each_row=False)
            elif task == "zhan":
                data["choice"] = load_counts(arrays, "zhan_counts", (n, 2), folds)
            else:
                data["choice"] = load_counts(arrays, "onestop_counts", (n, 4), folds)
                correct = arrays["onestop_correct"]
                if correct.shape != (n,) or not np.issubdtype(correct.dtype, np.integer) or not np.all((0 <= correct) & (correct < 4)):
                    raise ValueError("Expected the displayed gold option index of each OneStop input")
                data["correct"] = correct
    return datasets


class ResidualAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.down = nn.Linear(HIDDEN_WIDTH, ADAPTER_RANK)
        self.up = nn.Linear(ADAPTER_RANK, HIDDEN_WIDTH, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, states):
        scale = math.sqrt(HIDDEN_WIDTH)
        return states + self.up(F.gelu(self.down(states * scale))) / scale


class Readout(nn.Module):
    def __init__(self, method):
        super().__init__()
        self.heads = nn.ModuleDict({
            "ratings": nn.Linear(2 * HIDDEN_WIDTH + 2, 2),
            "issues": nn.Linear(HIDDEN_WIDTH + 2, 2),
            "zhan": nn.Linear(HIDDEN_WIDTH, 1),
            "onestop": nn.Linear(HIDDEN_WIDTH, 4),
        })
        for head in self.heads.values():
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        self.adapters = nn.ModuleDict()
        if method == "independent_adapter":
            template = ResidualAdapter()
            self.adapters.update({task: copy.deepcopy(template) for task in TASKS})
        elif method == "shared_adapter":
            self.adapters["shared"] = ResidualAdapter()

    def forward(self, task, states, length=None):
        if len(self.adapters):
            adapter = self.adapters["shared" if "shared" in self.adapters else task]
            states = adapter(states)
        if task == "plaba":
            rating_logits = self.heads["ratings"](torch.cat((states.flatten(start_dim=1), length), dim=-1))
            issue_logits = self.heads["issues"](torch.cat((states[:, 0], length), dim=-1))
            return rating_logits, issue_logits
        return self.heads[task](states[:, 0])


def binary_nll(logits, counts):
    losses = counts[..., 0] * F.softplus(logits) + counts[..., 1] * F.softplus(-logits)
    return (losses.sum(dim=0) / counts.sum(dim=(0, -1))).mean()


def task_loss(model, task, batch):
    result = model(task, batch["states"], batch.get("length"))
    if task == "plaba":
        rating, issue = result
        scale = torch.linspace(0, 1, 5, dtype=rating.dtype, device=rating.device)
        squared_error = (rating.sigmoid()[..., None] - scale).square()
        ratings = batch["ratings"]
        rating_loss = ((squared_error * ratings).sum(dim=(0, 2)) / ratings.sum(dim=(0, 2))).mean()
        return (rating_loss + binary_nll(issue, batch["issues"])) / 2
    if task == "zhan":
        return binary_nll(result[:, 0], batch["choice"])
    counts = batch["choice"]
    return -(counts * result.log_softmax(dim=-1)).sum() / counts.sum()


def make_fold(datasets, fold, device):
    training, test, scaling = {}, {}, {}
    for task, data in datasets.items():
        mask = data["folds"] == fold
        states = data["states"]
        training[task] = {"states": torch.as_tensor(states[~mask], device=device)}
        test[task] = {"states": torch.as_tensor(states[mask], device=device)}
        if task == "plaba":
            length = data["length"]
            center = length[~mask].mean(axis=0)
            spread = length[~mask].std(axis=0)
            spread[spread == 0] = 1
            normalized = (length - center) / spread / math.sqrt(length.shape[1])
            training[task]["length"] = torch.as_tensor(normalized[~mask], device=device)
            test[task]["length"] = torch.as_tensor(normalized[mask], device=device)
            scaling = {"length_center": center, "length_scale": spread}
            keys = ("ratings", "issues")
        else:
            keys = ("choice",)
        for key in keys:
            training[task][key] = torch.as_tensor(data[key][~mask], dtype=torch.float32, device=device)
    return training, test, scaling


def predict(model, task, batch):
    result = model(task, batch["states"], batch.get("length"))
    if task == "plaba":
        rating, issue = result
        return torch.cat((1 + 4 * rating.sigmoid(), issue.sigmoid()), dim=-1)
    if task == "zhan":
        return result[:, 0].sigmoid()
    return result.softmax(dim=-1)


def train_fold(training, test, method, fold, device):
    torch.manual_seed(SEED + fold)
    model = Readout(method).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    started = perf_counter()
    first = None
    for step in range(UPDATES):
        optimizer.zero_grad(set_to_none=True)
        by_task = torch.stack([task_loss(model, task, training[task]) for task in TASKS])
        if step == 0:
            first = by_task.detach().cpu().tolist()
        loss = by_task.mean()
        loss.backward()
        optimizer.step()
    with torch.inference_mode():
        last = torch.stack([task_loss(model, task, training[task]) for task in TASKS]).cpu().tolist()
        predictions = {task: predict(model, task, batch).cpu().numpy() for task, batch in test.items()}
    if not np.isfinite(last).all() or not all(np.isfinite(v).all() for v in predictions.values()):
        raise ValueError(f"Nonfinite objective or prediction for {method} fold {fold}")
    checkpoint = {key: value.detach().cpu().numpy() for key, value in model.state_dict().items()}
    diagnostics = {
        "fold": fold,
        "method": method,
        "updates": UPDATES,
        "training_trials_or_counts": {
            "plaba_ratings_by_endpoint": training["plaba"]["ratings"].sum(dim=(0, 2)).cpu().int().tolist(),
            "plaba_checklist_by_endpoint": training["plaba"]["issues"].sum(dim=(0, 2)).cpu().int().tolist(),
            "zhan": int(training["zhan"]["choice"].sum().item()),
            "onestop": int(training["onestop"]["choice"].sum().item()),
        },
        "initial_task_loss": dict(zip(TASKS, first, strict=True)),
        "final_task_loss": dict(zip(TASKS, last, strict=True)),
        "seconds": perf_counter() - started,
    }
    return checkpoint, predictions, diagnostics


def binary_scores(p, counts):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    no, yes = counts.T
    n = int(counts.sum())
    return {
        "trials": n,
        "brier": float((no * p**2 + yes * (1 - p)**2).sum() / n),
        "nll_nats": float(-(no * np.log1p(-p) + yes * np.log(p)).sum() / n),
    }


def score(task, data, rows, prediction):
    if task == "plaba":
        scale = np.arange(1, 6)
        metrics = {}
        for index, name in enumerate(RATINGS):
            counts = data["ratings"][rows, index]
            pred = prediction[:, index]
            distance = np.abs(pred[:, None] - scale)
            n = int(counts.sum())
            metrics[name] = {
                "trials": n,
                "mean_sentence_mae": float(np.abs(pred - counts @ scale / counts.sum(axis=1)).mean()),
                "trial_mae": float((counts * distance).sum() / n),
                "trial_mse": float((counts * distance**2).sum() / n),
            }
        for index, name in enumerate(ISSUES):
            metrics[name] = binary_scores(prediction[:, 2 + index], data["issues"][rows, index])
        return metrics
    counts = data["choice"][rows]
    if task == "zhan":
        return binary_scores(prediction, counts)
    n = int(counts.sum())
    selected = np.argmax(prediction, axis=-1)
    totals = counts.sum(axis=-1)
    gold = data["correct"][rows]
    positions = np.arange(len(gold))
    correct_counts = counts[positions, gold]
    gold_probabilities = prediction[positions, gold]
    correct_binary_counts = np.column_stack((totals - correct_counts, correct_counts))
    return {
        "trials": n,
        "selection_nll_nats": float(-(counts * np.log(np.clip(prediction, 1e-12, 1))).sum() / n),
        "selection_brier": float((totals @ (prediction**2).sum(axis=-1) - 2 * (counts * prediction).sum() + n) / n),
        "modal_choice_agreement": float(counts[positions, selected].sum() / n),
        "correctness": binary_scores(gold_probabilities, correct_binary_counts),
    }


def run(args):
    started = perf_counter()
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or not torch.cuda.get_device_name().endswith(" L4"):
            raise RuntimeError("Expected training on one NVIDIA L4")
        torch.cuda.reset_peak_memory_stats()
    torch.set_float32_matmul_precision("highest")
    paths = {task: getattr(args, f"{task}_features") for task in TASKS}
    datasets = read_data(args.design, paths)
    args.output.mkdir(parents=True, exist_ok=True)
    shapes = {"plaba": (4,), "zhan": (), "onestop": (4,)}
    stored = {task: np.full((len(METHODS), len(datasets[task]["ids"]), *shapes[task]), np.nan) for task in TASKS}
    metrics, heads, fits = {}, {}, []
    for fold in range(N_FOLDS):
        training, test, scaling = make_fold(datasets, fold, device)
        heads.update({f"fold{fold}.{name}": value for name, value in scaling.items()})
        for index, method in enumerate(METHODS):
            checkpoint, predictions, detail = train_fold(training, test, method, fold, device)
            heads.update({f"{method}.fold{fold}.{name}": value for name, value in checkpoint.items()})
            for task in TASKS:
                stored[task][index, datasets[task]["folds"] == fold] = predictions[task]
            fits.append(detail)
            print(f"Fold {fold + 1}/{N_FOLDS}, {method}: {detail['seconds']:.2f} s", flush=True)
    if not all(np.isfinite(x).all() for x in stored.values()):
        raise RuntimeError("Incomplete or nonfinite out-of-fold predictions")
    for index, method in enumerate(METHODS):
        metrics[method] = {}
        for task, data in datasets.items():
            pred = stored[task][index]
            metrics[method][task] = {
                "pooled": score(task, data, slice(None), pred),
                "folds": [
                    {"fold": fold, **score(task, data, mask, pred[mask])}
                    for fold in range(N_FOLDS) if (mask := data["folds"] == fold).any()
                ],
            }
    predictions = {}
    for task in TASKS:
        predictions[f"{task}_ids"] = datasets[task]["ids"]
        predictions[f"{task}_predictions"] = stored[task]
    np.savez_compressed(args.output / "heads.npz", **heads)
    np.savez_compressed(args.output / "predictions.npz", methods=np.asarray(METHODS), **predictions)
    save_json(args.output / "metrics.json", {
        "scope": "Five fixed source-family folds; the same cached states are used for every condition.",
        "methods": metrics,
    })
    save_json(args.output / "fitting.json", {
        "device": torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
        "torch": torch.__version__,
        "dtype": "float32",
        "optimizer": {"name": "AdamW", "lr": LR, "weight_decay": WEIGHT_DECAY, "updates_per_fold": UPDATES},
        "adapter_rank": ADAPTER_RANK,
        "seed_base": SEED,
        "loss": {
            "task_weights": "Equal mean of PLABA, Zhan, OneStop; one complete training batch per task and optimizer update.",
            "plaba": "Average of rating MSE on a [0,1] scale and checklist Bernoulli NLL. Endpoint means use observed counts.",
            "zhan": "Bernoulli NLL on observed NP1 choices.",
            "onestop": "Categorical NLL on observed displayed choices; no gold option given to model.",
        },
        "input_sha256": {"design": digest(args.design), **{task: digest(path) for task, path in paths.items()}},
        "output_sha256": {name: digest(args.output / name) for name in ("heads.npz", "predictions.npz", "metrics.json")},
        "total_s": perf_counter() - started,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
        "fits": fits,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, required=True)
    for name in TASKS:
        parser.add_argument(f"--{name}-features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), required=True)
    run(parser.parse_args())
