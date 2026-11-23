"""Fit matched low-label readouts with and without two disjoint study effects."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .train import (
    ADAPTER_RANK,
    LR,
    N_FOLDS,
    SEED,
    TASKS,
    UPDATES,
    WEIGHT_DECAY,
    Readout,
    digest,
    make_fold,
    predict,
    read_data,
    save_json,
    task_loss,
)


METHODS = ("linear_trial", "linear_effect", "shared_trial", "shared_effect")
SEEDS = (0, 1, 2, 3, 4)
EFFECTS = ("plaba", "zhan")


def _remaining_means(counts, levels):
    totals = counts.sum(axis=-1)
    means = np.divide(counts @ levels, totals, out=np.zeros_like(totals, dtype=np.float64), where=totals > 0)
    return means, totals > 0


def read_effect_design(path, datasets):
    """Validate split isolation and recompute targets from disjoint rating counts."""
    with np.load(path, allow_pickle=False) as archive:
        selected = {key: archive[key] for key in archive.files}
    if selected["seeds"].tolist() != list(SEEDS):
        raise ValueError("Expected the five fixed label draws")

    prefix = (N_FOLDS, len(SEEDS))
    direct = {
        "plaba_rating_counts": datasets["plaba"]["ratings"],
        "plaba_issue_counts": datasets["plaba"]["issues"],
        "zhan_counts": datasets["zhan"]["choice"],
    }
    for key, full in direct.items():
        count = selected[key]
        if count.shape != (*prefix, *full.shape) or not np.issubdtype(count.dtype, np.integer):
            raise ValueError(f"Invalid count inventory for {key}")
        if (count < 0).any() or (count > full[None, None]).any():
            raise ValueError(f"{key} includes counts outside the full observation inventory")
    for corpus in EFFECTS:
        n_items = len(datasets[corpus]["ids"])
        coef = selected[f"{corpus}_effect_coefficients"]
        target = selected[f"{corpus}_effect_targets"]
        if coef.shape != (*prefix, n_items) or target.shape != prefix or not (np.isfinite(coef).all() and np.isfinite(target).all()):
            raise ValueError(f"Invalid effect inventory for {corpus}")

    scales = {"plaba": (np.arange(1, 6), 1.0), "zhan": (np.arange(2), 2.0)}
    for fold in range(N_FOLDS):
        plaba_training = datasets["plaba"]["folds"] != fold
        zhan_training = datasets["zhan"]["folds"] != fold
        for seed_index in range(len(SEEDS)):
            ratings = selected["plaba_rating_counts"][fold, seed_index]
            issues = selected["plaba_issue_counts"][fold, seed_index]
            zhan = selected["zhan_counts"][fold, seed_index]
            if ratings[~plaba_training].any() or issues[~plaba_training].any() or zhan[~zhan_training].any():
                raise ValueError(f"Direct labels include held-out items in fold {fold}")
            if (
                not np.all(ratings[plaba_training].sum(axis=-1) == 1)
                or (issues.sum(axis=-1) > 1).any()
                or not np.array_equal(issues[:, 0].sum(axis=-1), issues[:, 1].sum(axis=-1))
                or (issues.sum(axis=(0, 2)) <= 0).any()
                or not np.all(zhan[zhan_training].sum(axis=-1) == 4)
            ):
                raise ValueError(f"Unexpected direct observation counts in fold {fold}")
            residual = {
                "plaba": datasets["plaba"]["ratings"][:, 1] - ratings[:, 1],
                "zhan": datasets["zhan"]["choice"] - zhan,
            }
            for corpus, mask in (("plaba", plaba_training), ("zhan", zhan_training)):
                coef = selected[f"{corpus}_effect_coefficients"][fold, seed_index]
                target = selected[f"{corpus}_effect_targets"][fold, seed_index]
                values, mass = scales[corpus]
                mean, observed = _remaining_means(residual[corpus], values)
                if (
                    np.any(coef[~mask | ~observed] != 0)
                    or not np.isclose(coef.clip(min=0).sum(), mass, atol=1e-10, rtol=0)
                    or not np.isclose(coef.clip(max=0).sum(), -mass, atol=1e-10, rtol=0)
                    or not np.isclose(coef @ mean, target, atol=1e-10, rtol=0)
                ):
                    raise ValueError(f"{corpus} contrast disagrees with disjoint training observations in fold {fold}")
    return selected


def effect_blocks(training, datasets, selected, fold, seed_index, device):
    """Bind the same sampled direct counts to both arms of a comparison."""
    current = {task: dict(batch) for task, batch in training.items()}
    for task, field, count_key in (
        ("plaba", "ratings", "plaba_rating_counts"),
        ("plaba", "issues", "plaba_issue_counts"),
        ("zhan", "choice", "zhan_counts"),
    ):
        mask = datasets[task]["folds"] != fold
        current[task][field] = torch.as_tensor(selected[count_key][fold, seed_index, mask], dtype=torch.float32, device=device)
    coefs = {
        task: torch.as_tensor(
            selected[f"{task}_effect_coefficients"][fold, seed_index, datasets[task]["folds"] != fold],
            dtype=torch.float32, device=device,
        )
        for task in EFFECTS
    }
    targets = [float(selected[f"{task}_effect_targets"][fold, seed_index]) for task in EFFECTS]
    return current, coefs, torch.tensor(targets, dtype=torch.float32, device=device)


def predicted_effects(model, training, coefs):
    plaba = predict(model, "plaba", training["plaba"])[:, 1] @ coefs["plaba"]
    zhan = predict(model, "zhan", training["zhan"]) @ coefs["zhan"]
    return torch.stack((plaba, zhan))


def fit_readout(training, test, coefs, target, method, fold, seed, device):
    torch.manual_seed(SEED + len(SEEDS) * fold + seed)
    readout_type = "shared_adapter" if method.startswith("shared_") else "linear"
    model = Readout(readout_type).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    widths = target.new_tensor([4.0, 1.0])
    use_effects = method.endswith("_effect")
    start = perf_counter()
    for _ in range(UPDATES):
        optimizer.zero_grad(set_to_none=True)
        objective = torch.stack([task_loss(model, task, training[task]) for task in TASKS]).mean()
        if use_effects:
            objective = objective + ((predicted_effects(model, training, coefs) - target) / widths).square().mean()
        objective.backward()
        optimizer.step()
    with torch.inference_mode():
        direct_losses = [float(task_loss(model, task, training[task]).item()) for task in TASKS]
        estimates = predicted_effects(model, training, coefs)
        effect_error = float(((estimates - target) / widths).square().mean().item())
        predictions = {task: predict(model, task, batch).cpu().numpy() for task, batch in test.items()}
    if not np.isfinite([*direct_losses, effect_error]).all() or not all(np.isfinite(p).all() for p in predictions.values()):
        raise RuntimeError(f"Nonfinite weak-fit result for {method}, fold {fold}, draw {seed}")
    checkpoint = {name: value.detach().cpu().numpy() for name, value in model.state_dict().items()}
    report = {
        "fold": fold,
        "seed": seed,
        "method": method,
        "updates": UPDATES,
        "seconds": perf_counter() - start,
        "final_direct_task_losses": dict(zip(TASKS, direct_losses, strict=True)),
        "final_effect_error_normalized": effect_error,
        "train_effect_predictions": dict(zip(EFFECTS, estimates.cpu().tolist(), strict=True)),
        "train_effect_targets": dict(zip(EFFECTS, target.cpu().tolist(), strict=True)),
    }
    return checkpoint, predictions, report


def publish(output, datasets, saved, heads, report):
    arrays = {"methods": np.asarray(METHODS), "seeds": np.asarray(SEEDS)}
    for task in TASKS:
        arrays[f"{task}_ids"] = datasets[task]["ids"]
        arrays[f"{task}_predictions"] = saved[task]
    for name, values in (("heads.npz", heads), ("predictions.npz", arrays)):
        temporary = output / f"{name}.partial.npz"
        np.savez_compressed(temporary, **values)
        temporary.replace(output / name)
    report["output_sha256"] = {name: digest(output / name) for name in ("heads.npz", "predictions.npz")}
    temporary = output / "fitting.json.partial"
    save_json(temporary, report)
    temporary.replace(output / "fitting.json")


def run(args):
    start = perf_counter()
    device = torch.device(args.device)
    gpu = device.type == "cuda"
    if gpu:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or not torch.cuda.get_device_name().endswith(" L4"):
            raise RuntimeError("Expected training on one NVIDIA L4")
        torch.cuda.reset_peak_memory_stats()
    torch.set_float32_matmul_precision("highest")
    paths = {task: getattr(args, f"{task}_features") for task in TASKS}
    datasets = read_data(args.design, paths)
    selected = read_effect_design(args.effect_design, datasets)
    shapes = {"plaba": (4,), "zhan": (), "onestop": (4,)}
    stored = {
        task: np.full((len(METHODS), len(SEEDS), len(datasets[task]["ids"]), *shapes[task]), np.nan)
        for task in TASKS
    }
    metadata = {"design": args.design, "effect_design": args.effect_design, **paths}
    report = {
        "device": torch.cuda.get_device_name() if gpu else "cpu",
        "torch": torch.__version__,
        "dtype": "float32",
        "optimizer": {"name": "AdamW", "lr": LR, "weight_decay": WEIGHT_DECAY, "updates_per_fold": UPDATES},
        "adapter_rank": ADAPTER_RANK,
        "initialization_seed_rule": "1729 + 5 × fold + label draw",
        "methods": list(METHODS),
        "seeds": list(SEEDS),
        "objective": "Equal mean of three direct task losses; effect arms add the mean squared PLABA/4 and Zhan/1 effect errors.",
        "input_sha256": {key: digest(path) for key, path in metadata.items()},
        "completed_folds": [],
        "fits": [],
        "completed": False,
    }
    heads = {}
    args.output.mkdir(parents=True, exist_ok=True)
    for fold in range(N_FOLDS):
        training, test, scaling = make_fold(datasets, fold, device)
        heads.update({f"fold{fold}.{key}": value for key, value in scaling.items()})
        for seed_index, seed in enumerate(SEEDS):
            low, coefs, target = effect_blocks(training, datasets, selected, fold, seed_index, device)
            for method_index, method in enumerate(METHODS):
                weights, predictions, fit = fit_readout(low, test, coefs, target, method, fold, seed, device)
                heads.update({f"{method}.seed{seed}.fold{fold}.{name}": weight for name, weight in weights.items()})
                for task in TASKS:
                    stored[task][method_index, seed_index, datasets[task]["folds"] == fold] = predictions[task]
                report["fits"].append(fit)
        report["completed_folds"].append(fold)
        if fold == N_FOLDS - 1:
            if any(not np.isfinite(value).all() for value in stored.values()):
                raise RuntimeError("Incomplete out-of-fold predictions")
            if not np.array_equal(stored["onestop"][0], stored["onestop"][1]):
                raise RuntimeError("The linear control leaked effect gradients into OneStop")
            report["completed"] = True
        report["total_s"] = perf_counter() - start
        report["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated() if gpu else None
        publish(args.output, datasets, stored, heads, report)
        print(f"Completed fold {fold + 1}/{N_FOLDS} with {len(SEEDS) * len(METHODS)} matched fits", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--effect-design", type=Path, required=True)
    for task in TASKS:
        parser.add_argument(f"--{task}-features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), required=True)
    run(parser.parse_args())
