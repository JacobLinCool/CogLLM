"""Fit fixed and adaptive multi-observation readouts on the same frozen features."""

import argparse
from collections import Counter
from dataclasses import dataclass
import gc
from math import ceil, isfinite
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn

from research.adaptive_joint.policy import SourceWeights
from research.backbone_transfer.extract import digest, read_json, stream_rows, write_json
from research.token_readout.backbone import Backbone
from research.token_readout.readout import PortableReadout
from research.token_readout.transfer_fit import (
    forecasts, observations, report_loss, report_values, support_loss, support_values,
    validated_losses, validation_values,
)
from research.union_training.fit_observations import Readouts, evaluate, load_tasks, row_nll, weights


@dataclass(frozen=True)
class Projected:
    offsets: tuple
    hidden: torch.Tensor
    middle: torch.Tensor
    relations: torch.Tensor


@dataclass(frozen=True)
class Block:
    name: str
    source: str
    records: object
    mass: np.ndarray
    multiplier: np.ndarray | None = None
    task: object | None = None


@dataclass(frozen=True)
class Batch:
    block: Block
    indices: np.ndarray
    exposure: float
    fraction: float


class JointReadout(nn.Module):
    def __init__(self, seed, tasks):
        super().__init__()
        table = seed['reader']
        self.curve = PortableReadout(table['final.weight'].shape[1], seed['variant'], seed['relations'])
        self.curve.load_state_dict(table, strict=True)
        self.curve.final.requires_grad_(False)
        self.curve.middle.requires_grad_(False)
        width = self.curve.final.out_features
        self.adapter = nn.Linear(width, width, bias=False)
        nn.init.zeros_(self.adapter.weight)
        self.auxiliary = Readouts(width, tasks)

    def adapt(self, hidden):
        return hidden + self.adapter(hidden)

    def by_observation(self, task, cache, rows):
        positions = task.coordinates[rows]
        return self.auxiliary.measure(task, self.adapt(cache[positions.clamp_min(0)]), positions >= 0)

    def by_token(self, document):
        return self.curve.score_projected(self.adapt(document.hidden), document.middle, document.relations)

    @torch.no_grad()
    def export(self, seed, destination):
        residual = self.adapter.weight.detach()
        curve = {name: value.detach().cpu() for name, value in self.curve.state_dict().items()}
        old = self.curve.final.weight
        curve['final.weight'] = (old + residual @ old).cpu()
        torch.save({**seed, 'reader': curve}, destination / 'curve.pt')
        np.save(destination / 'adapter_weight.npy', residual.cpu().numpy(), allow_pickle=False)
        auxiliary = {name: value.detach().cpu().numpy() for name, value in self.auxiliary.state_dict().items()}
        np.savez_compressed(destination / 'heads.npz', **auxiliary)


def load_observations(root, device):
    coordinates = {(row['id'], stop): index for index, (row, stop) in enumerate(
        (row, stop) for row in stream_rows(root / 'coordinates.jsonl') for stop in row['stops'])}
    matrix = np.load(root / 'states.npy', allow_pickle=False)
    if (matrix.dtype != np.float32 or matrix.ndim != 2 or len(coordinates) != len(matrix)
            or sorted(coordinates.values()) != list(range(len(matrix))) or not np.isfinite(matrix).all()):
        raise ValueError('Projected cache and original causal stops differ')
    return torch.from_numpy(matrix).to(device), load_tasks(root, coordinates, device)


@torch.no_grad()
def extract_token(root, candidate, path, seed):
    start = perf_counter()
    runtime = Backbone.load(candidate, path)
    if runtime.revision != seed['revision']:
        raise ValueError('Token observation and cached Qwen states use distinct revisions')
    curve = PortableReadout(runtime.width, seed['variant'], seed['relations']).cuda()
    curve.load_state_dict(seed['reader'], strict=True)
    curve.eval()
    layout = read_json(root / 'layout.json')
    if layout['model'] != candidate:
        raise ValueError('Layout belongs to another tokenizer')
    rows = layout['rows']
    checked = runtime.validate('Knowledge about the first part of the experiment becomes available during the next section.')
    cache = {}
    forwards = 0
    sample = None

    def counted(_module, _inputs):
        nonlocal forwards
        forwards += 1

    with runtime.encoder.register_forward_pre_hook(counted):
        for row in stream_rows(root / 'stimuli.jsonl'):
            key = row['id']
            last = forwards
            features = runtime.encode(row['text'])
            if key in cache or forwards != last + 1 or features.offsets != tuple(map(tuple, rows[key]['offsets'])):
                raise ValueError('Token pass or original coordinates differ from the certified layout')
            first, middle = curve.project(features)
            cache[key] = Projected(features.offsets, first, middle, features.relations)
            if rows[key]['kind'] == 'controls' and sample is None:
                sample = key, features.hidden
    if cache.keys() != rows.keys() or sample is None:
        raise ValueError('The token stimulus inventory is incomplete')
    return cache, rows, sample, {'forwards': forwards, 'documents': len(cache),
                                 'tokens': sum(len(document.offsets) for document in cache.values()),
                                 'causality': checked, 'seconds': perf_counter() - start}


def make_blocks(tasks, token_train, linked):
    blocks = []
    for task in tasks:
        indices = np.flatnonzero(task.role == 0)
        correction = weights(task, indices)
        blocks.append(Block(task.name, task.source, indices, task.trials[indices] * correction,
                            multiplier=correction, task=task))
    for name, source, rows in (
        ('token_first_report', 'frick', token_train['frick']),
        ('evidence_window', 'expmrc', token_train['evidence']),
        ('linked_support_window', 'iirc', linked),
    ):
        if not rows:
            raise ValueError(f'The joint experiment requires nonempty training rows for {name}')
        blocks.append(Block(name, source, rows, np.ones(len(rows), dtype=np.float64)))
    return blocks


def epoch_batches(blocks, rng, settings):
    batches = []
    for block in blocks:
        shuffled = rng.permutation(len(block.records))
        size = settings['observation_batch'] if block.task is not None else settings['token_batch']
        total = float(block.mass.sum())
        for start in range(0, len(shuffled), size):
            indices = shuffled[start:start + size]
            exposure = float(block.mass[indices].sum())
            if exposure <= 0:
                raise ValueError(f'Observation block has no exposure: {block.name}')
            batches.append(Batch(block, indices, exposure, exposure / total))
    order = rng.permutation(len(batches))
    return [batches[int(index)] for index in order]


def microbatch_loss(model, batch, matrix, documents, seed):
    block = batch.block
    if block.task is not None:
        batch_rows = torch.as_tensor(block.records[batch.indices], device=matrix.device)
        result = model.by_observation(block.task, matrix, batch_rows)
        observed = block.task.response[batch_rows]
        sampling_weight = torch.as_tensor(block.multiplier[batch.indices], device=matrix.device, dtype=matrix.dtype)
        return row_nll(block.task, result, observed).dot(sampling_weight) / batch.exposure
    prior, scale = seed['priors'], seed['scale']

    def predict(key):
        return model.by_token(documents[key])

    selected = [block.records[int(index)] for index in batch.indices]
    if block.name == 'token_first_report':
        terms = [report_loss(report_values(group, predict), prior, scale['opening_gain'], paired=True)
                 for group in selected]
    else:
        terms = [support_loss(support_values(row, predict), prior, scale['resolution_gain'],
                              scale['resolution_bias'], paired=True) for row in selected]
    return torch.stack(terms).mean()


def fit_update(model, optimizer, policy, group, base, matrix, documents, seed, radius=None):
    sources = [batch.block.source for batch in group]
    endpoints = [batch.block.name for batch in group]
    factors = policy.weights(sources, endpoints=endpoints, base=base)
    optimizer.zero_grad(set_to_none=True)
    measurements = []
    for batch, coefficient, weight in zip(group, base, factors, strict=True):
        loss = microbatch_loss(model, batch, matrix, documents, seed)
        shared = []
        with model.adapter.weight.register_hook(lambda gradient: shared.append(gradient.detach().norm())):
            (loss * (coefficient * weight)).backward()
        if len(shared) != 1:
            raise RuntimeError(f'Task did not update the shared residual: {batch.block.name}')
        measurements.append((loss.detach(), shared[0] / (coefficient * weight)))
    clipped = nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
    values = torch.stack([value for item in measurements for value in item] + [clipped]).cpu().tolist()
    if not all(isfinite(value) for value in values):
        raise ArithmeticError('Nonfinite observation loss or shared gradient')
    optimizer.step()
    projection = None
    if radius is not None:
        with torch.no_grad():
            amplitude = model.adapter.weight.norm()
            model.adapter.weight.mul_(radius / amplitude.clamp_min(radius))
            projection = amplitude > radius
    for i, (source, endpoint) in enumerate(zip(sources, endpoints, strict=True)):
        policy.observe(source, values[2 * i], values[2 * i + 1], endpoint=endpoint)
    return values[-1], factors, values[:-1], projection


@torch.no_grad()
def development(model, matrix, tasks, documents, dev, seed, batch_size):
    model.eval()
    old = evaluate(model.by_observation, matrix, tasks, batch_size)
    values = validation_values(dev, lambda key: model.by_token(documents[key]))
    token = validated_losses(values, seed['priors'], seed['scale'])
    by_source = {source: [] for source in {block['source'] for block in old.values()} | {'expmrc'}}
    for item in old.values():
        by_source[item['source']].append(item['nll'])
    by_source['frick'].append(token['frick'])
    by_source['expmrc'].append(token['evidence'])
    average = {source: float(np.mean(scores)) for source, scores in sorted(by_source.items())}
    return {'source_balanced_nll': float(np.mean(list(average.values()))), 'by_source_nll': average,
            'token_nll': token, 'observation_nll': old}


@torch.no_grad()
def check_export(model, seed, sample, documents, destination):
    key, raw_hidden = sample
    reference = model.by_token(documents[key])
    exported = torch.load(destination / 'curve.pt', weights_only=True, map_location='cuda')
    rebuilt = PortableReadout(model.curve.input_width, seed['variant'], seed['relations']).cuda()
    rebuilt.load_state_dict(exported['reader'], strict=True)
    projected = rebuilt.final(nn.functional.layer_norm(raw_hidden, raw_hidden.shape[-1:]))
    actual = rebuilt.score_projected(projected, documents[key].middle, documents[key].relations)
    differences = {name: float((getattr(actual, name) - getattr(reference, name)).abs().max())
                   for name in ('opening', 'resolution', 'continuity')}
    if not all(isfinite(value) and value < 5e-4 for value in differences.values()):
        raise AssertionError(f'Exported portable checkpoint does not reproduce the joint scores: {differences}')
    return differences


def run_arm(mode, trial_seed, seed, matrix, tasks, documents, geometry, sample, training, dev, linked, settings, out):
    torch.manual_seed(trial_seed)
    order = np.random.default_rng(trial_seed)
    model = JointReadout(seed, tasks).cuda()
    guarded = mode in ('guarded', 'guarded-fixed')
    guard = settings['retention'] if guarded else None
    if guarded:
        model.curve.requires_grad_(False)
        token_reference = validation_values(dev, lambda key: model.by_token(documents[key]))
        before = validated_losses(token_reference, seed['priors'], seed['scale'])
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=settings['lr'], weight_decay=settings['weight_decay'])
    policy = SourceWeights(mode='fixed' if mode == 'guarded-fixed' else 'adaptive' if guarded else mode,
                           **settings['adaptive'])
    blocks = make_blocks(tasks, training, linked)
    source_counts = Counter(block.source for block in blocks)
    start = perf_counter()
    best_epoch, best_value, best = 0, float('inf'), None
    best_violation = float('inf')
    reason = 'epoch_budget'
    history = []
    out.mkdir()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, settings['epochs'] + 1):
        model.train()
        batches = epoch_batches(blocks, order, settings)
        group_size = settings['microbatches_per_step']
        steps = ceil(len(batches) / group_size)
        norms = []
        projected = []
        contributions = {source: [] for source in source_counts}
        losses = {source: [] for source in source_counts}
        for begin in range(0, len(batches), group_size):
            group = batches[begin:begin + group_size]
            base = [steps * b.fraction / (len(source_counts) * source_counts[b.block.source]) for b in group]
            norm, factors, measurement, projection = fit_update(
                model, optimizer, policy, group, base, matrix, documents, seed, guard['radius'] if guarded else None)
            norms.append(norm)
            if projection is not None:
                projected.append(projection)
            for i, (batch, factor) in enumerate(zip(group, factors, strict=True)):
                contributions[batch.block.source].append(factor)
                losses[batch.block.source].append(measurement[2 * i])
        evidence = development(model, matrix, tasks, documents, dev, seed, settings['observation_batch'])
        objective = evidence['source_balanced_nll']
        violation = sum(max(0.0, evidence['token_nll'][key] - before[key] - guard[key])
                        for key in ('frick', 'evidence')) if guarded else 0.0
        if violation < best_violation or (violation == best_violation and objective < best_value - settings['min_dev_gain']):
            best_epoch, best_value, best_violation = epoch, objective, violation
            best = {name: value.detach().clone() for name, value in model.state_dict().items()}
        elapsed = perf_counter() - start
        history.append({'epoch': epoch, 'elapsed_seconds': elapsed, 'updates': steps, 'ordinary_backward_calls': len(batches),
                        'source_balanced_nll': objective, 'by_source_dev': evidence['by_source_nll'],
                        'token_dev': evidence['token_nll'], 'mean_gradient_norm': float(np.mean(norms)),
                        'p95_gradient_norm': float(np.percentile(norms, 95)),
                        'clipped_fraction': sum(norm > 5 for norm in norms) / len(norms),
                        'mean_source_weight': {source: float(np.mean(w)) for source, w in contributions.items()},
                        'source_weight_range': [min(min(w) for w in contributions.values()),
                                                max(max(w) for w in contributions.values())],
                        'source_microbatches': {source: len(w) for source, w in contributions.items()},
                        'median_source_train_nll': {source: float(np.median(w)) for source, w in losses.items()}})
        if guarded:
            history[-1]['retention_violation_nll'] = violation
            history[-1]['adapter_projection_fraction'] = float(torch.stack(projected).float().mean())
        print(mode, trial_seed, 'epoch', epoch, 'updates', steps, 'dev NLL', round(objective, 5), 'elapsed',
              round(elapsed, 2), flush=True)
        if epoch >= settings['min_epochs'] and epoch - best_epoch >= settings['patience']:
            reason = 'development_plateau'
            break
    if best is None:
        raise ArithmeticError('Training completed without a finite development checkpoint')
    model.load_state_dict(best)
    model.eval()
    model.export(seed, out)
    fold = check_export(model, seed, sample, documents, out)
    evaluate(model.by_observation, matrix, tasks, settings['observation_batch'], out / 'predictions.npz')
    write_json(out / 'token-forecast.json', forecasts(geometry, documents, seed['priors'], model.by_token, seed['scale']))
    write_json(out / 'fitting.json', {
        'mode': mode, 'seed': trial_seed, 'selected_epoch': best_epoch, 'stop_reason': reason, 'history': history,
        'seconds': perf_counter() - start, 'selected_dev_nll': best_value,
        'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved(), 'source_names': sorted(source_counts),
        'blocks': len(blocks), 'observation_training_rows': sum(int((task.role == 0).sum()) for task in tasks),
        'forward_checkpoint_fold_max_abs': fold, 'adaptive_policy': policy.summary(),
        'trainable_parameters': sum(value.numel() for value in model.parameters() if value.requires_grad),
        **({'retention': {'seed_token_dev_nll': before, 'guard': guard, 'selected_violation': best_violation,
                         'selected_adapter_norm': float(model.adapter.weight.detach().norm())}} if guarded else {}),
    })


def run(candidate, model, stimuli, output, protocol_sha256, settings):
    if candidate != 'qwen' or torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != 'NVIDIA L4':
        raise RuntimeError('The joint comparison requires one verified Colab L4 and the cached Qwen revision')
    root = stimuli.parent.parent
    receipt = read_json(root / 'receipt.json')
    settings = read_json(settings)
    seed = torch.load(root / 'seed.pt', map_location='cpu', weights_only=True)
    if seed['candidate'] != candidate:
        raise ValueError('Portable seed checkpoint belongs to another candidate')
    output.mkdir(exist_ok=False)
    matrix, tasks = load_observations(root / 'observations', torch.device('cuda'))
    if len(tasks) != 62:
        raise ValueError('Expected all 62 registered observation blocks')
    print('Loaded compact states for', len(matrix), 'causal stops, and', len(tasks), 'observation blocks', flush=True)
    documents, geometry, sample, extraction = extract_token(root / 'token', candidate, model, seed)
    print('Captured', extraction['forwards'], 'single-pass token documents', flush=True)
    gc.collect()
    torch.cuda.empty_cache()
    learning = read_json(root / 'token' / 'learning.json')
    train, dev = (observations(learning[split]) for split in ('train', 'dev'))
    linked = learning['train']['linked']
    if any(geometry[row['id']]['split'] != split for split in ('train', 'dev')
           for task_rows in learning[split].values() for row in task_rows):
        raise ValueError('Final token labels must stay outside the training worker')
    write_json(output / 'causality.json', extraction)
    for trial_seed in settings['seeds']:
        for mode in settings['modes']:
            run_arm(mode, trial_seed, seed, matrix, tasks, documents, geometry, sample, train, dev, linked, settings,
                    output / f'{mode}-{trial_seed}')
    write_json(output / 'seal.json', {'protocol_sha256': protocol_sha256, 'input_receipt_sha256': digest(root / 'receipt.json'),
                                     'seed_sha256': digest(root / 'seed.pt'), 'revision': seed['revision'],
                                     'registered_observations': list(task.name for task in tasks),
                                     'formal_labels_in_worker': False, 'seed_settings': settings,
                                     'files_sha256': {path.relative_to(output).as_posix(): digest(path)
                                                      for path in sorted(output.rglob('*')) if path.is_file()}})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', choices=('qwen',), required=True)
    for name in ('model', 'stimuli', 'output', 'settings'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--protocol-sha256', required=True)
    run(**vars(parser.parse_args()))
