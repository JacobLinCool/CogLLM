"""Jointly fit observation-specific models on one cache of causal Qwen states."""

import argparse
from collections import Counter
from dataclasses import dataclass
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from research.backbone_transfer.extract import digest, read_json, stream_rows, write_json
from research.union_training.observation_losses import (
    categorical_nll, first_event_log_probs, ordered_cutpoints, ordered_log_probs,
)
from research.union_training.observation_registry import contracts


STATE_WIDTH = 12
MIN_SCALE = 0.02
COUNT_MODELS = frozenset({
    'categorical', 'choice', 'ordinal', 'ordinal_binary', 'hurdle_ordinal', 'ordered_choice', 'first_event',
})


@dataclass
class Task:
    name: str
    source: str
    likelihood: str
    states: tuple
    coordinates: torch.Tensor
    response: torch.Tensor
    role: np.ndarray
    trials: np.ndarray
    family: np.ndarray
    correct: np.ndarray | None
    center: float
    spread: float

    @property
    def output_width(self):
        return self.response.shape[1]


def load_projection(projection, device):
    matrix = np.load(projection / 'states.npy', allow_pickle=False)
    coordinates = {}
    for row in stream_rows(projection / 'coordinates.jsonl'):
        for stop in row['stops']:
            index = len(coordinates)
            if coordinates.setdefault((row['id'], stop), index) != index:
                raise ValueError('Duplicate state coordinate')
    if matrix.ndim != 2 or len(matrix) != len(coordinates) or not np.isfinite(matrix).all():
        raise ValueError('Invalid extracted states')
    values = torch.from_numpy(matrix.astype(np.float32, copy=False)).to(device)
    return F.normalize(values, dim=-1) * values.shape[-1] ** 0.5, coordinates


def task_responses(kind, rows, width):
    if kind != 'first_event':
        return np.asarray([row['outcome'] for row in rows], dtype=np.float64)
    observed = np.zeros((len(rows), width + 1), dtype=np.float64)
    for i, row in enumerate(rows):
        counts = row['outcome']
        n_stops = len(row['coordinates'])
        if len(counts) != n_stops + 1:
            raise ValueError('First-event outcomes require one count per visited stop and one censored count')
        observed[i, :n_stops] = counts[:-1]
        observed[i, -1] = counts[-1]
    return observed


def observation_scales(kind, observed, role):
    if kind in COUNT_MODELS:
        if (np.any(observed < 0) or np.any(observed != np.floor(observed))
                or np.any(observed.sum(axis=1) == 0)):
            raise ValueError('Expected positive response denominators')
        return observed.sum(axis=1), 0.0, 1.0

    train = observed[role == 0]
    if kind == 'gaussian':
        if observed.shape[1] != 3 or np.any(observed[:, 0] <= 0) or np.any(observed[:, 2] < 0):
            raise ValueError('Expected count, mean and population variance')
        counts = observed[:, 0].copy()
        center = np.average(train[:, 1], weights=train[:, 0])
        variance = np.average(train[:, 2] + (train[:, 1] - center) ** 2, weights=train[:, 0])
        spread = np.sqrt(variance) if variance > 0 else 1.0
        observed[:, 1] = (observed[:, 1] - center) / spread
        observed[:, 2] /= spread ** 2
        return counts, float(center), float(spread)

    if observed.shape[1] != 2:
        raise ValueError('Expected two continuous observed values')
    if kind == 'known_error_gaussian':
        if np.any(observed[:, 1] < 0):
            raise ValueError('The sampling error must be nonnegative')
        center = train[:, 0].mean()
        spread = train[:, 0].std() or 1.0
        observed[:, 0] = (observed[:, 0] - center) / spread
        observed[:, 1] /= spread
    elif kind == 'paired_gaussian':
        center, spread = train.mean(), train.std() or 1.0
        observed[:] = (observed - center) / spread
    else:
        raise ValueError(f'Unregistered likelihood: {kind}')
    return np.ones(len(observed)), float(center), float(spread)


def load_tasks(packet, lookup, device):
    index = read_json(packet / 'index.json')
    registry = contracts()
    if index['schema_version'] != 2 or {item['name'] for item in index['tasks']} != set(registry):
        raise ValueError('Unexpected registered observation blocks')
    tasks = []
    for registered in index['tasks']:
        name = registered['name']
        contract = registry[name]
        if (registered['likelihood'], registered['coordinates'], tuple(registered['states'])) != (
                contract.likelihood, contract.coordinates, contract.states):
            raise ValueError(f'Changed observation contract: {name}')
        rows = list(stream_rows(packet / registered['file']))
        n_stops = [len(row['coordinates']) for row in rows]
        width = max(n_stops)
        if ((contract.coordinates == 'point' and set(n_stops) != {1})
                or (contract.coordinates in {'pair', 'target_and_document_end'} and set(n_stops) != {2})
                or min(n_stops) < 1):
            raise ValueError(f'Wrong number of observation coordinates: {name}')
        coordinates = np.full((len(rows), width), -1, dtype=np.int64)
        for i, row in enumerate(rows):
            coordinates[i, :n_stops[i]] = [lookup[tuple(value)] for value in row['coordinates']]
            if contract.coordinates == 'trajectory':
                doc = row['coordinates'][0][0]
                stops = [stop for _, stop in row['coordinates']]
                if any(key != doc for key, _ in row['coordinates']) or stops != sorted(set(stops)):
                    raise ValueError('A trajectory must follow one strictly ordered exposure')
        role = np.asarray([row['role'] for row in rows])
        if not set(role) <= {0, 1} or not np.any(role == 0):
            raise ValueError(f'No legitimate training split for {name}')
        response = task_responses(contract.likelihood, rows, width)
        if response.ndim != 2 or not np.isfinite(response).all():
            raise ValueError(f'Invalid response for {name}')
        if contract.likelihood in COUNT_MODELS - {'first_event'} and len(registered['labels']) != response.shape[1]:
            raise ValueError(f'Wrong response categories for {name}')
        trials, center, spread = observation_scales(contract.likelihood, response, role)
        correct = None
        if contract.likelihood == 'choice':
            correct = np.asarray([row['correct_index'] for row in rows])
            if correct.dtype.kind not in 'iu' or np.any(correct < 0) or np.any(correct >= response.shape[1]):
                raise ValueError('An answer identity is outside the observed alternatives')
        tasks.append(Task(name, registered['source'], contract.likelihood, contract.states,
                          torch.as_tensor(coordinates, device=device),
                          torch.as_tensor(response, dtype=torch.float32, device=device),
                          role, trials, np.asarray([row['family'] for row in rows]), correct, center, spread))
    return tasks


class Measurement(nn.Module):
    def __init__(self, task):
        super().__init__()
        self.kind = task.likelihood
        width = STATE_WIDTH * len(task.states)
        if self.kind not in {'first_event', 'ordered_choice', 'paired_gaussian'}:
            width *= task.coordinates.shape[1]
        categories = task.output_width
        outputs = {
            'ordinal': 1, 'ordinal_binary': 1 + categories // 2, 'hurdle_ordinal': 2,
            'first_event': 1, 'ordered_choice': 1, 'gaussian': 2, 'known_error_gaussian': 2,
            'paired_gaussian': 2,
        }.get(self.kind, categories)
        self.linear = nn.Linear(width, outputs, bias=self.kind != 'ordered_choice')
        nn.init.normal_(self.linear.weight, std=0.025)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)
        ordinal_levels = {'ordinal': categories, 'ordinal_binary': categories // 2,
                          'hurdle_ordinal': categories - 1}.get(self.kind)
        if ordinal_levels is not None:
            self.gaps = nn.Parameter(torch.zeros(ordinal_levels - 2))
        if self.kind == 'ordered_choice':
            self.order_bias = nn.Parameter(torch.zeros(()))
        if self.kind == 'paired_gaussian':
            self.post_offset = nn.Parameter(torch.zeros(()))
            self.correlation = nn.Parameter(torch.zeros(()))

    def forward(self, representation, available):
        if self.kind == 'first_event':
            logits = self.linear(representation).squeeze(-1)
            mass, survival = first_event_log_probs(logits, available)
            return torch.cat((mass, survival[:, -1:]), dim=1)
        if self.kind == 'ordered_choice':
            first, second = self.linear(representation).squeeze(-1).unbind(dim=1)
            logit = first - second + self.order_bias
            return torch.stack((F.logsigmoid(-logit), F.logsigmoid(logit)), dim=1)
        if self.kind == 'paired_gaussian':
            raw = self.linear(representation)
            first, second = raw[:, :, 0].unbind(dim=1)
            means = torch.stack((first, second + self.post_offset), dim=1)
            scales = F.softplus(raw[:, :, 1]) + MIN_SCALE
            rho = self.correlation.tanh().expand(len(raw)) * 0.99
            return torch.cat((means, scales, rho[:, None]), dim=1)
        raw = self.linear(representation.flatten(1))
        if self.kind in {'categorical', 'choice'}:
            return F.log_softmax(raw, dim=1)
        if self.kind == 'ordinal':
            return ordered_log_probs(raw[:, 0], ordered_cutpoints(self.gaps))
        if self.kind == 'ordinal_binary':
            rating = ordered_log_probs(raw[:, 0], ordered_cutpoints(self.gaps))
            flag = torch.stack((F.logsigmoid(-raw[:, 1:]), F.logsigmoid(raw[:, 1:])), dim=2)
            return (rating[:, :, None] + flag).flatten(1)
        if self.kind == 'hurdle_ordinal':
            ordinal = ordered_log_probs(raw[:, 1], ordered_cutpoints(self.gaps))
            return torch.cat((F.logsigmoid(-raw[:, :1]), F.logsigmoid(raw[:, :1]) + ordinal), dim=1)
        return torch.stack((raw[:, 0], F.softplus(raw[:, 1]) + MIN_SCALE), dim=1)


class Readouts(nn.Module):
    def __init__(self, width, tasks):
        super().__init__()
        self.shared = nn.ModuleDict({
            state: nn.Linear(width, STATE_WIDTH)
            for state in sorted({state for task in tasks for state in task.states})
        })
        self.measurements = nn.ModuleDict({task.name: Measurement(task) for task in tasks})

    def measure(self, task, hidden, available):
        state = torch.cat([self.shared[name](hidden).tanh() for name in task.states], dim=-1)
        return self.measurements[task.name](state, available)

    def forward(self, task, state_cache, rows):
        positions = task.coordinates[rows]
        return self.measure(task, state_cache[positions.clamp_min(0)], positions >= 0)


def row_nll(task, forecast, response):
    if task.likelihood in COUNT_MODELS:
        return categorical_nll(forecast, response)
    log_two_pi = math.log(2 * math.pi)
    if task.likelihood == 'gaussian':
        count, mean, variance = response.unbind(dim=1)
        location, scale = forecast.unbind(dim=1)
        return count * (0.5 * (log_two_pi + ((location - mean).square() + variance) / scale.square()) + scale.log())
    if task.likelihood == 'known_error_gaussian':
        location, scale = forecast.unbind(dim=1)
        point, standard_error = response.unbind(dim=1)
        variance = scale.square() + standard_error.square()
        return 0.5 * (log_two_pi + variance.log() + (point - location).square() / variance)
    location, scale, rho = forecast[:, :2], forecast[:, 2:4], forecast[:, 4]
    first, second = ((response - location) / scale).unbind(dim=1)
    det = 1 - rho.square()
    standardized = (first.square() - 2 * rho * first * second + second.square()) / det
    return log_two_pi + scale.log().sum(dim=1) + 0.5 * (det.log() + standardized)


def weights(task, allowed):
    if task.likelihood == 'hurdle_ordinal':
        totals = Counter()
        for index in allowed:
            totals[task.family[index]] += task.trials[index]
        return np.asarray([1 / totals[task.family[index]] for index in allowed])
    return np.ones(len(allowed))


@torch.no_grad()
def evaluate(model, cache, tasks, batch_size, prediction_path=None):
    metrics, predictions = {}, {}
    for task in tasks:
        role = (0, 1) if prediction_path else (1,)
        for side in role:
            indices = np.flatnonzero(task.role == side)
            if not len(indices):
                continue
            aggregate = 0.0
            observed = 0.0
            values = []
            extra = weights(task, indices)
            for start in range(0, len(indices), batch_size):
                batch = indices[start:start + batch_size]
                rows = torch.as_tensor(batch, device=cache.device)
                forecast = model(task, cache, rows)
                mass = extra[start:start + len(batch)]
                multipliers = torch.as_tensor(mass, dtype=cache.dtype, device=cache.device)
                aggregate += float((row_nll(task, forecast, task.response[rows]) * multipliers).sum())
                observed += float(np.dot(task.trials[batch], mass))
                if prediction_path:
                    values.append(forecast.cpu())
            average = aggregate / observed
            if not np.isfinite(average):
                raise ArithmeticError(f'Nonfinite evaluation likelihood for {task.name}')
            if side == 1:
                metrics[task.name] = {'source': task.source, 'nll': average, 'rows': len(indices)}
            if prediction_path:
                key = f'{task.name}.{"train" if side == 0 else "development"}'
                forecast = torch.cat(values).numpy()
                predictions[f'{key}.rows'] = indices
                if task.likelihood in COUNT_MODELS:
                    probability = np.exp(forecast)
                    predictions[f'{key}.probabilities'] = probability
                    if task.correct is not None:
                        predictions[f'{key}.correct_probability'] = probability[np.arange(len(indices)), task.correct[indices]]
                elif task.likelihood == 'paired_gaussian':
                    means = forecast[:, :2] * task.spread + task.center
                    predictions[f'{key}.means'] = means
                    predictions[f'{key}.difference'] = means[:, 1] - means[:, 0]
                    predictions[f'{key}.scales'] = forecast[:, 2:4] * task.spread
                    predictions[f'{key}.correlation'] = forecast[:, 4]
                else:
                    predictions[f'{key}.mean'] = forecast[:, 0] * task.spread + task.center
                    predictions[f'{key}.scale'] = forecast[:, 1] * task.spread
    if prediction_path:
        np.savez_compressed(prediction_path, **predictions)
    return metrics


def source_average(metrics):
    sources = Counter(record['source'] for record in metrics.values())
    return sum(record['nll'] / sources[record['source']] for record in metrics.values()) / len(sources)


@dataclass
class Plateau:
    rates: tuple
    patience: int
    minimum_gain: float
    reference: float
    stage: int = 0
    stale: int = 0

    def observe(self, score):
        if score <= self.reference - self.minimum_gain:
            self.reference = score
            self.stale = 0
        else:
            self.stale += 1
        if self.stale < self.patience:
            return 'continue'
        if self.stage == len(self.rates) - 1:
            return 'plateau'
        self.stage += 1
        self.stale = 0
        return 'reduce_rate'


def continuation_start(directory, packet, model):
    policy_file = directory / 'policy.json'
    policy = read_json(policy_file)
    seal = read_json(directory / 'seal.json')
    prior = read_json(directory / 'fitting.json')
    weights_file = directory / 'heads.npz'
    if (policy['packet_receipt_sha256'] != digest(packet / 'receipt.json')
            or seal['packet_receipt_sha256'] != policy['packet_receipt_sha256']
            or seal['backbone_updates'] != 0
            or any(digest(directory / name) != seal['files_sha256'][name]
                   for name in ('heads.npz', 'fitting.json'))
            or policy['origin_heads_sha256'] != digest(weights_file)
            or prior['selected_epoch'] != policy['origin_epoch']):
        raise ValueError('Continuation does not match the verified prior fit and training data')
    rates = tuple(policy['learning_rates'])
    if (not rates or any(not math.isfinite(rate) or rate <= 0 for rate in rates)
            or sorted(set(rates), reverse=True) != list(rates)
            or type(policy['patience']) is not int or policy['patience'] < 1
            or not 0 < policy['minimum_gain'] < 1
            or type(policy['max_additional_epochs']) is not int or policy['max_additional_epochs'] < 1):
        raise ValueError('Invalid predeclared stopping policy')
    template = model.state_dict()
    with np.load(weights_file, allow_pickle=False) as archive:
        if len(archive.files) != len(template) or set(archive.files) != set(template):
            raise ValueError('Prior weights use a different readout inventory')
        state = {}
        for key, tensor in template.items():
            matrix = archive[key]
            if matrix.shape != tuple(tensor.shape) or matrix.dtype != np.float32 or not np.isfinite(matrix).all():
                raise ValueError(f'Invalid prior readout parameter {key}')
            state[key] = torch.from_numpy(matrix.copy())
    model.load_state_dict(state)
    return policy, digest(policy_file), rates


def run(packet, projection, output, device_name='cuda', epochs=12, batch_size=256, seed=1729,
        continuation=None, progress=None):
    if epochs < 1 or batch_size < 1:
        raise ValueError('Positive epochs and batch size required')
    if device_name == 'cuda' and (torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != 'NVIDIA L4'):
        raise RuntimeError('The configured GPU run requires one NVIDIA L4')
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    fit_started_at = perf_counter()
    receipt = read_json(packet / 'receipt.json')
    for name, expected in receipt['packet_sha256'].items():
        if digest(packet / name) != expected:
            raise ValueError(f'Observation packet changed: {name}')
    device = torch.device(device_name)
    cache, lookup = load_projection(projection, device)
    if len(lookup) != receipt['positions']:
        raise ValueError('Observation packet and extracted positions differ')
    tasks = load_tasks(packet, lookup, device)
    model = Readouts(cache.shape[1], tasks).to(device)
    history, best, best_epoch, best_score = [], None, None, float('inf')
    plateau, policy, policy_hash = None, None, None
    origin_epoch, learning_rate = 0, .001
    if continuation is not None:
        policy, policy_hash, rates = continuation_start(continuation, packet, model)
        origin_epoch, learning_rate = policy['origin_epoch'], rates[0]
        epochs = policy['max_additional_epochs']
        model.eval()
        development = evaluate(model, cache, tasks, batch_size)
        best_score = source_average(development)
        if not math.isclose(best_score, policy['origin_development_nll'], abs_tol=2e-5, rel_tol=0):
            raise ValueError(f'Archived baseline does not replay on L4: {best_score}')
        best_epoch, best = origin_epoch, {key: value.detach().cpu().clone()
                                          for key, value in model.state_dict().items()}
        history.append({'epoch': best_epoch, 'source_balanced_development_nll': best_score,
                        'updates': 0, 'development': development})
        plateau = Plateau(rates, policy['patience'], policy['minimum_gain'], best_score)
        print(f'Replayed archived epoch {origin_epoch}: development loss {best_score:.6f}', flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    source_counts = Counter(task.source for task in tasks)
    training = [np.flatnonzero(task.role == 0) for task in tasks]
    task_weights = [weights(task, rows) for task, rows in zip(tasks, training, strict=True)]
    denominators = [np.dot(task.trials[rows], weight)
                    for task, rows, weight in zip(tasks, training, task_weights, strict=True)]
    rng = np.random.default_rng(seed)
    output.mkdir(parents=True, exist_ok=False)
    reason = 'epoch_budget'
    for epoch in range(origin_epoch + 1, origin_epoch + epochs + 1):
        model.train()
        batches = []
        for i, rows in enumerate(training):
            shuffled = rng.permutation(len(rows))
            batches.extend((i, shuffled[start:start + batch_size]) for start in range(0, len(rows), batch_size))
        order = rng.permutation(len(batches))
        steps = (len(order) + 3) // 4
        norm_sum, clipped = 0.0, 0
        for start in range(0, len(order), 4):
            optimizer.zero_grad(set_to_none=True)
            loss = cache.new_zeros(())
            for index in order[start:start + 4]:
                i, indices = batches[index]
                task = tasks[i]
                rows = torch.as_tensor(training[i][indices], device=device)
                weights_batch = torch.as_tensor(task_weights[i][indices], dtype=cache.dtype, device=device)
                raw = model(task, cache, rows)
                scale = steps / (len(source_counts) * source_counts[task.source] * denominators[i])
                loss = loss + row_nll(task, raw, task.response[rows]).dot(weights_batch) * scale
            loss.backward()
            norm = float(nn.utils.clip_grad_norm_(model.parameters(), max_norm=5, error_if_nonfinite=True))
            norm_sum += norm
            clipped += norm > 5
            optimizer.step()
        model.eval()
        development = evaluate(model, cache, tasks, batch_size)
        score = source_average(development)
        row = {'epoch': epoch, 'source_balanced_development_nll': score, 'development': development,
               'updates': steps, 'mean_preclip_gradient_norm': norm_sum / steps, 'clipped_updates': clipped}
        history.append(row)
        print(f'Observation model epoch {epoch}: development loss {score:.6f}, updates {steps}', flush=True)
        if score < best_score:
            best, best_epoch, best_score = {key: value.detach().cpu().clone()
                                           for key, value in model.state_dict().items()}, epoch, score
        if progress is not None:
            progress(epoch, score, best_epoch, best_score)
        if plateau is not None:
            action = plateau.observe(score)
            row.update({'learning_rate': optimizer.param_groups[0]['lr'], 'plateau_action': action})
            if action == 'plateau':
                reason = 'development_plateau'
                break
            if action == 'reduce_rate':
                for group in optimizer.param_groups:
                    group['lr'] = plateau.rates[plateau.stage]
                print(f'Development plateau; reducing LR to {plateau.rates[plateau.stage]}', flush=True)
    model.load_state_dict(best)
    development = evaluate(model, cache, tasks, batch_size, output / 'predictions.npz')
    np.savez_compressed(output / 'heads.npz', **{key: value.numpy() for key, value in best.items()})
    write_json(output / 'fitting.json', {
        'seed': seed, 'seconds': perf_counter() - fit_started_at, 'selected_epoch': best_epoch,
        'checkpoint_at_budget_edge': best_epoch == origin_epoch + epochs, 'device': str(device),
        'stop_reason': reason,
        **({'continuation': policy, 'continuation_policy_sha256': policy_hash,
            'optimizer_start': 'Fresh AdamW moments for the archived weight checkpoint'} if policy else {}),
        'sources': sorted(source_counts), 'history': history, 'development': development,
        'weighting': 'mean over study sources, mean over their measurement blocks; retrospective word votes balanced by material family',
        'effect_term': 'within-study derived effects routed to diagnostics; independent published CLEAR standard errors included in its marginal observation likelihood',
        'observation_blocks': len(tasks), 'shared_states': sorted(model.shared),
        'training_rows': sum(len(rows) for rows in training),
    })
    write_json(output / 'seal.json', {
        'packet_receipt_sha256': digest(packet / 'receipt.json'),
        'backbone_updates': 0,
        'fit_code_sha256': digest(Path(__file__)),
        **({'continuation_policy_sha256': policy_hash,
            'origin_heads_sha256': policy['origin_heads_sha256']} if policy else {}),
        'files_sha256': {path.name: digest(path) for path in sorted(output.iterdir())},
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('packet', 'projection', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--device-name', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--continuation', type=Path)
    run(**vars(parser.parse_args()))
