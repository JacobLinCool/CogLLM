"""Fit independent and shared endpoint readouts on sealed, frozen Qwen states."""

import argparse
import copy
from dataclasses import dataclass
from hashlib import file_digest
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from research.joint.train import ResidualAdapter


WIDTH = 2048
METHODS = ('independent', 'shared', 'shared_effect')
EPOCHS = 6
BATCH_SIZE = 256
BATCHES_PER_UPDATE = 4
SEED = 1729
LEARNING_RATE = 0.002
WEIGHT_DECAY = 0.01
EFFECT_WEIGHT = 0.5
GRADIENT_CLIP = 5.0


def digest(path):
    with path.open('rb') as source:
        return file_digest(source, 'sha256').hexdigest()


def write_json(path, value):
    with path.open('x', encoding='utf-8') as output:
        json.dump(value, output, allow_nan=False, sort_keys=True, indent=2)
        output.write('\n')


@dataclass
class Effect:
    indices: torch.Tensor
    coefficients: torch.Tensor
    target: float
    scale: float
    weight: float
    category: int | None


@dataclass
class Task:
    name: str
    kind: str
    path: Path
    labels: list
    ids: np.ndarray
    groups: np.ndarray | None
    splits: np.ndarray
    features: torch.Tensor
    outcomes: torch.Tensor
    counts: np.ndarray
    train: np.ndarray
    development: np.ndarray
    center: float
    spread: float
    effects: list[Effect]


def read_effects(declarations, splits, kind, classes, device):
    effects = []
    for declaration in declarations:
        if not isinstance(declaration, dict):
            raise ValueError('Expected a numeric effect declaration')
        indices = np.asarray(declaration['indices'])
        coefficients = np.asarray(declaration['coefficients'], dtype=np.float64)
        target, scale, weight = (float(declaration[key]) for key in ('target', 'scale', 'weight'))
        if (
            indices.ndim != 1 or indices.dtype.kind not in 'iu' or not len(indices)
            or coefficients.shape != indices.shape or len(np.unique(indices)) != len(indices)
            or np.any(indices < 0) or np.any(indices >= len(splits)) or np.any(splits[indices] != 0)
            or not np.isfinite(coefficients).all() or not np.any(coefficients)
            or not math.isfinite(target) or not math.isfinite(scale) or not math.isfinite(weight)
            or scale <= 0.0 or weight <= 0.0
        ):
            raise ValueError('Effect must have a finite target and positive scale/weight on unique training rows')
        metric = declaration['metric']
        category = declaration.get('category')
        if metric == 'category':
            if kind != 'categorical' or type(category) is not int or not 0 <= category < classes:
                raise ValueError('Category effects require a valid categorical output index')
        elif metric != 'mean' or category is not None:
            raise ValueError('Mean effects require an unspecified category')
        effects.append(Effect(
            torch.as_tensor(indices.astype(np.int64), device=device),
            torch.as_tensor(coefficients, dtype=torch.float32, device=device),
            target, scale, weight, category,
        ))
    return effects


def read_task(root, row, device):
    path = root / row['file']
    name, kind, labels = (row[key] for key in ('name', 'kind', 'labels'))
    if (
        type(name) is not str or not name or kind not in ('categorical', 'gaussian')
        or type(labels) is not list or Path(row['file']).name != row['file'] or path.suffix != '.npz'
        or not isinstance(row['effects'], list)
    ):
        raise ValueError('Invalid endpoint declaration')
    with np.load(path, allow_pickle=False) as source:
        features, outcomes, splits, ids = (source[key] for key in ('features', 'outcomes', 'split', 'id'))
        groups = source['group'] if 'group' in source else None
    if (
        ids.ndim != 1 or ids.dtype.kind != 'U' or len(ids) == 0 or len(set(ids.tolist())) != len(ids)
        or '' in ids or splits.shape != ids.shape or splits.dtype.kind not in 'iu'
        or not set(splits.tolist()) <= {0, 1, 2}
        or (groups is not None and (groups.shape != ids.shape or groups.dtype.kind != 'U' or '' in groups))
        or features.shape != (len(ids), WIDTH) or features.dtype.kind != 'f' or not np.isfinite(features).all()
        or np.any(np.max(np.abs(features), axis=1) == 0)
        or outcomes.ndim != 2 or len(outcomes) != len(ids) or outcomes.dtype.kind not in 'fiu'
        or not np.isfinite(outcomes).all() or np.any(outcomes[splits == 2] != 0)
    ):
        raise ValueError(f'Invalid IDs, splits, frozen states, or blinded outcomes for {name}')
    if kind == 'categorical':
        if outcomes.shape[1] < 2 or len(labels) != outcomes.shape[1] or np.any(outcomes < 0) or np.any(outcomes != np.floor(outcomes)):
            raise ValueError(f'{name} requires one nonnegative whole count per response category')
        counts = outcomes.sum(axis=1, dtype=np.float64)
    else:
        if (
            outcomes.shape[1] != 3 or len(labels) != 1
            or np.any(outcomes[:, 0] < 0) or np.any(outcomes[:, 0] != np.floor(outcomes[:, 0]))
            or np.any(outcomes[:, 2] < 0) or np.any(outcomes[outcomes[:, 0] == 0] != 0)
        ):
            raise ValueError(f'{name} requires [count, observed mean, population variance] with empty rows all zero')
        counts = outcomes[:, 0].astype(np.float64)
    train = np.flatnonzero((splits == 0) & (counts > 0))
    development = np.flatnonzero((splits == 1) & (counts > 0))
    effects = read_effects(row['effects'], splits, kind, outcomes.shape[1], device)
    if len(train) == 0 and not effects:
        raise ValueError(f'{name} has neither supervised training rows nor quantitative training effects')
    center, spread = 0.0, 1.0
    outcomes = outcomes.astype(np.float32)
    if kind == 'gaussian':
        if len(train):
            train_outcomes = outcomes[train].astype(np.float64)
            mean = train_outcomes[:, 1]
            center = float(np.average(mean, weights=counts[train]))
            variance = np.average(train_outcomes[:, 2] + (mean - center) ** 2, weights=counts[train])
            spread = float(np.sqrt(variance)) if variance > 0.0 else 1.0
        else:
            spread = float(np.median([effect.scale for effect in effects]))
        present = counts > 0
        outcomes[present, 1] = (outcomes[present, 1] - center) / spread
        outcomes[present, 2] /= spread ** 2
        if not np.isfinite(outcomes).all():
            raise ValueError(f'{name} contains values outside Float32 observation scaling')
    return Task(
        name, kind, path, labels, ids, groups, splits,
        F.normalize(torch.as_tensor(features, dtype=torch.float32, device=device), dim=1),
        torch.as_tensor(outcomes, device=device), counts, train, development, center, spread, effects,
    )


def read_packet(root, device):
    with (root / 'index.json').open(encoding='utf-8') as source:
        packet = json.load(source)
    tasks = [read_task(root, row, device) for row in packet['tasks']]
    names = [task.name for task in tasks]
    task_files = {task.path for task in tasks}
    if not names or len(set(names)) != len(names) or len(task_files) != len(names):
        raise ValueError('Expected distinct endpoints and one registered packet file per endpoint')
    if not any(len(task.train) for task in tasks) or not any(len(task.development) for task in tasks):
        raise ValueError('Matched fitting requires direct human training and grouped development observations')
    if not any(task.effects for task in tasks):
        raise ValueError('A matched effect arm needs at least one training-only effect')
    groups = {}
    for task in tasks:
        if task.groups is None:
            continue
        for group, role in zip(task.groups.tolist(), task.splits.tolist(), strict=True):
            if groups.setdefault(group, role) != role:
                raise ValueError(f'Material group spans split roles: {group}')
    return tasks


class Readouts(nn.Module):
    def __init__(self, tasks, independent):
        super().__init__()
        self.independent = independent
        adapter = ResidualAdapter()
        self.adapters = nn.ModuleList(
            [copy.deepcopy(adapter) for _ in tasks] if independent else [adapter]
        )
        self.heads = nn.ModuleList(
            nn.Linear(WIDTH, task.outcomes.shape[1] if task.kind == 'categorical' else 2)
            for task in tasks
        )
        for head in self.heads:
            nn.init.zeros_(head.bias)
            nn.init.zeros_(head.weight)

    def forward(self, task_no, states):
        return self.heads[task_no](self.adapters[task_no if self.independent else 0](states) * math.sqrt(WIDTH))


def row_nll(task, raw, observed):
    if task.kind == 'categorical':
        return -(observed * raw.log_softmax(dim=1)).sum(dim=1)
    offset, log_sigma = raw[:, 0], raw[:, 1].clamp(-5, 4)
    squared_error = (offset - observed[:, 1]).square() + observed[:, 2]
    return observed[:, 0] * (0.5 * (math.log(2 * math.pi) + torch.exp(-2 * log_sigma) * squared_error) + log_sigma)


def effect_prediction(model, task_no, task, effect):
    states = task.features[effect.indices]
    raw = model(task_no, states)
    if task.kind == 'gaussian':
        observed_scale = raw[:, 0] * task.spread + task.center
    else:
        probabilities = raw.softmax(dim=1)
        if effect.category is not None:
            observed_scale = probabilities[:, effect.category]
        else:
            observed_scale = probabilities @ torch.arange(probabilities.shape[1], dtype=raw.dtype, device=raw.device)
    return observed_scale @ effect.coefficients


def development_metrics(task, raw, rows):
    observed = task.outcomes[rows]
    vote_count = observed.sum(dim=1) if task.kind == 'categorical' else observed[:, 0]
    total = vote_count.sum()
    nll = float(row_nll(task, raw, observed).sum() / total)
    metrics = {'observations': int(total), 'rows': len(rows), 'nll_nats': nll}
    if task.kind == 'gaussian':
        mean = raw[:, 0]
        error = mean - observed[:, 1]
        metrics['nll_nats'] += math.log(task.spread)
        metrics['standardized_nll_nats'] = nll
        metrics['mean_calibration_gap'] = float((vote_count * error).sum() / total) * task.spread
        metrics['rmse'] = float((vote_count * (error.square() + observed[:, 2])).sum().div(total).sqrt()) * task.spread
        return metrics, nll

    p = raw.softmax(dim=1)
    brier = (vote_count * (1 + p.square().sum(dim=1)) - 2 * (observed * p).sum(dim=1)).sum() / total
    confidence, predicted = p.max(dim=1)
    correct = observed.gather(1, predicted[:, None])[:, 0]
    if p.shape[1] == 2:
        confidence, selected = p[:, 1], observed[:, 1]
    else:
        selected = correct
    bins = (confidence * 10).floor().clamp(0, 9).long()
    difference = torch.zeros(10, dtype=p.dtype, device=p.device)
    difference.scatter_add_(0, bins, vote_count * confidence - selected)
    metrics.update({
        'brier': float(brier), 'accuracy': float(correct.sum() / total),
        'calibration_error_10_bins': float(difference.abs().sum() / total),
        'mean_calibration_gap': float((vote_count * confidence - selected).sum() / total),
    })
    return metrics, nll


@torch.inference_mode()
def evaluate(model, tasks, batch_size):
    scores, objectives = {}, []
    for task_no, task in enumerate(tasks):
        rows = task.development
        if not len(rows):
            continue
        raw = torch.cat([
            model(task_no, task.features[rows[start:start + batch_size]])
            for start in range(0, len(rows), batch_size)
        ], dim=0)
        scores[task.name], objective = development_metrics(task, raw, rows)
        objectives.append(objective)
    objective = float(np.mean(objectives))
    if not math.isfinite(objective):
        raise ArithmeticError('Nonfinite development objective')
    return scores, objective


def epoch_batches(tasks, direct, generators, schedule_rng, batch_size):
    batches = []
    for index in direct:
        rows = generators[index].permutation(tasks[index].train)
        batches.extend((index, rows[start:start + batch_size]) for start in range(0, len(rows), batch_size))
    return [batches[index] for index in schedule_rng.permutation(len(batches))]


def fit(tasks, method, epochs, batch_size, seed, device):
    torch.manual_seed(seed)
    model = Readouts(tasks, independent=method == 'independent').to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    direct = [i for i, task in enumerate(tasks) if len(task.train)]
    effect_tasks = [i for i, task in enumerate(tasks) if task.effects]
    batch_counts = {i: math.ceil(len(tasks[i].train) / batch_size) for i in direct}
    n_batches = sum(batch_counts.values())
    steps = math.ceil(n_batches / BATCHES_PER_UPDATE)
    generators = {i: np.random.default_rng(np.random.SeedSequence((seed, i))) for i in direct}
    schedule_rng = np.random.default_rng(np.random.SeedSequence((seed, len(tasks))))
    effect_rng = np.random.default_rng(np.random.SeedSequence((seed, len(tasks) + 1)))
    effects_per_update = math.ceil(len(effect_tasks) / steps)
    scale = n_batches / sum(math.sqrt(count) for count in batch_counts.values())
    batch_weights = {i: scale / math.sqrt(batch_counts[i]) for i in direct}
    # The mean turnout per training row fixes the outcome weight across minibatches.
    train_row_count = {i: float(tasks[i].counts[tasks[i].train].mean()) for i in direct}
    best, best_loss, best_epoch, history = None, math.inf, None, []
    start = perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        batches = epoch_batches(tasks, direct, generators, schedule_rng, batch_size)
        effect_order = effect_rng.permutation(effect_tasks)
        training_losses, gradient_norms = [], []
        for update, start_batch in enumerate(range(0, n_batches, BATCHES_PER_UPDATE)):
            optimizer.zero_grad(set_to_none=True)
            endpoints = []
            for i, rows in batches[start_batch:start_batch + BATCHES_PER_UPDATE]:
                task = tasks[i]
                raw = model(i, task.features[rows])
                endpoints.append(row_nll(task, raw, task.outcomes[rows]).mean() * (batch_weights[i] / train_row_count[i]))
            loss = torch.stack(endpoints).sum() / BATCHES_PER_UPDATE
            if method == 'shared_effect':
                terms = []
                offset = update * effects_per_update
                for position in range(offset, offset + effects_per_update):
                    i = effect_order[position % len(effect_order)]
                    task = tasks[i]
                    squared, weights = [], []
                    for effect in task.effects:
                        difference = effect_prediction(model, i, task, effect) - effect.target
                        squared.append((difference / effect.scale).square() * effect.weight)
                        weights.append(effect.weight)
                    terms.append(torch.stack(squared).sum() / sum(weights))
                loss = loss + EFFECT_WEIGHT * torch.stack(terms).mean()
            if not torch.isfinite(loss):
                raise ArithmeticError(f'Nonfinite training objective for {method}')
            loss.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP, error_if_nonfinite=True)
            training_losses.append(loss.detach())
            gradient_norms.append(norm.detach())
            optimizer.step()
        model.eval()
        dev_metrics, dev_loss = evaluate(model, tasks, batch_size)
        step_record = {
            'epoch': epoch, 'cumulative_updates': epoch * steps,
            'train_batch_objective_mean': float(torch.stack(training_losses).mean()),
            'max_gradient_norm_before_clipping': float(torch.stack(gradient_norms).max()),
            'development_objective': dev_loss, 'development': dev_metrics,
        }
        history.append(step_record)
        if dev_loss < best_loss:
            best, best_loss, best_epoch = copy.deepcopy(model.state_dict()), dev_loss, epoch
        print(json.dumps({
            'event': 'epoch', 'method': method, 'epoch': epoch, 'dev_nll': dev_loss,
            'train_objective': step_record['train_batch_objective_mean'],
        }, allow_nan=False), flush=True)
    model.load_state_dict(best)
    model.eval()
    return model, {
        'selected_epoch': best_epoch, 'steps_per_epoch': steps, 'updates': epochs * steps,
        'task_batches_each_epoch': {tasks[i].name: count for i, count in batch_counts.items()},
        'task_batch_weights': {tasks[i].name: weight for i, weight in batch_weights.items()},
        'observed_training_row_coverage_each_epoch': 'exactly_once',
        'seconds': perf_counter() - start, 'epochs': history,
    }


@torch.inference_mode()
def predict(model, tasks, method, batch_size):
    out = {}
    for i, task in enumerate(tasks):
        raw = torch.cat([
            model(i, task.features[start:start + batch_size])
            for start in range(0, len(task.ids), batch_size)
        ], dim=0)
        if task.kind == 'categorical':
            prediction = raw.softmax(dim=1)
        else:
            prediction = torch.stack((
                raw[:, 0] * task.spread + task.center, raw[:, 1].clamp(-5, 4).exp() * task.spread,
            ), dim=1)
        if not torch.isfinite(prediction).all():
            raise ArithmeticError(f'Nonfinite {method} predictions for {task.name}')
        out[f'{method}.{task.name}.predictions'] = prediction.cpu().numpy()
    return out


@torch.inference_mode()
def summarize_effects(model, tasks):
    return {
        task.name: [
            {'target': effect.target, 'predicted': float(effect_prediction(model, i, task, effect)),
             'original_scale': effect.scale, 'weight': effect.weight}
            for effect in task.effects
        ]
        for i, task in enumerate(tasks) if task.effects
    }


def run(input_path, output, device_name, epochs, batch_size, seed):
    if epochs < 1 or batch_size < 2 or seed < 0:
        raise ValueError('Expected a positive training budget and a nonnegative seed')
    device = torch.device(device_name)
    if device.type == 'cuda':
        if torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != 'NVIDIA L4':
            raise RuntimeError('Expected one Colab NVIDIA L4')
        torch.cuda.reset_peak_memory_stats()
    torch.set_float32_matmul_precision('highest')
    torch.set_num_threads(2)
    tasks = read_packet(input_path, device)
    output.mkdir(parents=True, exist_ok=False)
    start = perf_counter()
    report = {
        'seed': seed, 'epochs': epochs, 'batch_size': batch_size,
        'batches_per_update': BATCHES_PER_UPDATE,
        'batch_weighting': 'Inverse square root task batch counts, normalized to mean batch weight one.',
        'effect_schedule': 'All tasks have their effect contrasts applied at least once per epoch, in a matched deterministic rotation.',
        'learning_rate': LEARNING_RATE, 'weight_decay': WEIGHT_DECAY, 'effect_weight': EFFECT_WEIGHT,
        'device': torch.cuda.get_device_name() if device.type == 'cuda' else 'cpu',
        'torch': str(torch.__version__), 'backbone_updates': 0,
        'normalization': 'Unit-L2 Qwen states; Gaussian labels standardized with observed training trials only.',
        'selection': 'Minimum equal-endpoint development NLL, in standardized units for Gaussian endpoints.',
        'tasks': {
            task.name: {
                'kind': task.kind, 'labels': task.labels, 'file': task.path.name, 'sha256': digest(task.path),
                'train_rows': len(task.train), 'train_observations': int(task.counts[task.train].sum()),
                'development_rows': len(task.development),
                'development_observations': int(task.counts[task.development].sum()),
                'formal_unlabeled_rows': int(np.sum(task.splits == 2)),
                'train_effects': len(task.effects), 'center': task.center, 'spread': task.spread,
            }
            for task in tasks
        },
        'methods': {},
        'index_sha256': digest(input_path / 'index.json'),
    }
    predictions = {}
    for task in tasks:
        predictions[f'{task.name}.ids'] = task.ids
        predictions[f'{task.name}.split'] = task.splits
        if task.groups is not None:
            predictions[f'{task.name}.group'] = task.groups
    heads = {}
    for method in METHODS:
        model, diagnostics = fit(tasks, method, epochs, batch_size, seed, device)
        scores, _ = evaluate(model, tasks, batch_size)
        diagnostics['selected_development'] = scores
        diagnostics['training_effects'] = summarize_effects(model, tasks)
        heads.update({f'{method}.{key}': value.detach().cpu().numpy() for key, value in model.state_dict().items()})
        predictions.update(predict(model, tasks, method, batch_size))
        report['methods'][method] = diagnostics
    np.savez_compressed(output / 'heads.npz', **heads)
    np.savez_compressed(output / 'predictions.npz', **predictions)
    report['seconds'] = perf_counter() - start
    if device.type == 'cuda':
        report['peak_gpu_allocated_bytes'] = torch.cuda.max_memory_allocated()
        report['peak_gpu_reserved_bytes'] = torch.cuda.max_memory_reserved()
    write_json(output / 'fitting.json', report)
    write_json(output / 'seal.json', {
        'index_sha256': report['index_sha256'], 'fit_concluded': True, 'backbone_updates': 0,
        'device': report['device'], 'fit_code_sha256': digest(Path(__file__)),
        'files_sha256': {name: digest(output / name) for name in ('heads.npz', 'predictions.npz', 'fitting.json')},
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, dest='input_path')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cuda', 'cpu'), required=True, dest='device_name')
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--seed', type=int, default=SEED)
    run(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
