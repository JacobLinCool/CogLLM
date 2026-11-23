"""Fit a calibrated frozen readout and a matched query/key adapter on Colab L4."""

import argparse
from collections import defaultdict
from dataclasses import replace
from math import log
from pathlib import Path
import random
from time import monotonic

import numpy as np
import torch
from torch.nn import functional as F

from research.backbone_transfer.extract import digest, read_json, stream_rows, write_json
from research.token_readout.extraction import QwenRuntime
from research.token_readout.model import (
    Scores, TokenReadout, censored_nll, first_report_logits, first_report_nll,
    interval_nll, intervals, matched_effect, phase_first_event, question_mass,
)


def placement(features, device):
    return replace(features, **{name: getattr(features, name).to(device)
                                for name in ('hidden', 'mid', 'queries', 'keys', 'layer_input')})


def forward(readout, runtime, features, adapter):
    if adapter is not None:
        hidden, q, k = runtime.replay_last(features.layer_input, adapter)
        features = replace(features, hidden=hidden,
                           queries=torch.cat((features.queries[:-1], q[None])),
                           keys=torch.cat((features.keys[:-1], k[None])))
    return readout(features)


def event_spans(row, offsets):
    ends = row['event_ends']
    if [end for _, end in row['sentence_spans'][1:]] != ends:
        raise ValueError('Frick exposure ends disagree with the observed risk sequence')
    return [intervals(offsets, start, end) for start, end in row['sentence_spans'][1:]]


def evidence_coordinates(row, offsets):
    beginning = next(i for i, (_, end) in enumerate(offsets) if end > row['article_start'])
    anchors = {q['id']: intervals(offsets, *q['span'])[1] - 1 for q in row['queries']}
    if not anchors or len(anchors) != len(row['queries']) or max(anchors.values()) >= beginning:
        raise ValueError('Each evidence question needs a unique anchor before the article')
    return beginning, anchors


def prior_probability(events, exposure):
    return (events + 0.5) / (exposure + 1)


def logit(probability):
    return log(probability) - log(1 - probability)


def calibrate(fricks, evidence, cache):
    positional = defaultdict(lambda: [0, 0])
    word_events = 0
    word_exposure = 0
    for row in fricks:
        counts = row['event_counts']
        spans = event_spans(row, cache[row['id']].offsets)
        remaining = sum(counts) + row['censor_count']
        for position, (count, (start, end)) in enumerate(zip(counts, spans, strict=True)):
            totals = positional[position]
            totals[0] += count
            totals[1] += remaining
            word_exposure += remaining * (end - start)
            word_events += count
            remaining -= count
        if remaining != row['censor_count']:
            raise ValueError('Frick risk set mismatch')
    resolution_events = 0
    resolution_exposure = 0
    for row in evidence:
        offsets = cache[row['id']].offsets
        beginning, _ = evidence_coordinates(row, offsets)
        for query in row['queries']:
            if query['evidence_window'] is not None:
                end = intervals(offsets, *query['evidence_window'])[1]
                resolution_events += 1
            else:
                end = len(offsets)
            if end <= beginning:
                raise ValueError('Evidence window ends before the article starts')
            resolution_exposure += end - beginning
    if not word_events or not resolution_events:
        raise ValueError('Both training endpoints require observed events')
    return {
        'sentence': [logit(prior_probability(*positional[i])) for i in range(len(positional))],
        'sentence_default': logit(prior_probability(sum(x[0] for x in positional.values()),
                                                    sum(x[1] for x in positional.values()))),
        'opening': logit(prior_probability(word_events, word_exposure)),
        'resolution': logit(prior_probability(resolution_events, resolution_exposure)),
    }


def sentence_prior(priors, n, device):
    values = priors['sentence'][:n]
    return torch.tensor(values + [priors['sentence_default']] * (n - len(values)), device=device)


def prepare_train(train, cache):
    pairs = defaultdict(list)
    for row in train['frick']:
        spans = event_spans(row, cache[row['id']].offsets)
        pairs[row['group_id']].append((row, spans))
    if any(len(group) != 2 for group in pairs.values()):
        raise ValueError('Frick training requires both paired conditions')
    evidence = []
    for row in train['evidence']:
        offsets = cache[row['id']].offsets
        beginning, anchors = evidence_coordinates(row, offsets)
        windows = {q['id']: None if q['evidence_window'] is None else intervals(offsets, *q['evidence_window'])
                   for q in row['queries']}
        if any(window is not None and window[0] < beginning for window in windows.values()):
            raise ValueError('Observed answer precedes article')
        early_id, late_id = row['earlier_query'], row['later_query']
        sham = intervals(offsets, *row['earlier_sham']) if row.get('earlier_sham') else None
        evidence.append((row['id'], beginning, anchors, windows, early_id, late_id, sham))
    return list(pairs.values()), evidence


def response_likelihood(group, predict, priors):
    predictions, targets, losses = [], [], []
    for row, spans in sorted(group, key=lambda item: item[0]['condition']):
        values = predict(row['id']).opening
        logits = first_report_logits(values, spans, sentence_prior(priors, len(spans), values.device))
        counts = values.new_tensor([*row['event_counts'], row['censor_count']])
        target = row['target_sentence_index'] - 1
        if not 0 <= target < len(spans) or row['event_ends'][target] != row['target_span'][1]:
            raise ValueError('Corrected Frick target does not match the evaluated window')
        losses.append(first_report_nll(logits, counts))
        predictions.append(logits[target])
        targets.append(counts[target] / counts[target:].sum().clamp_min(1))
    return torch.stack(losses).mean() + 0.25 * matched_effect(predictions, targets)


def evidence_likelihood(record, predict, priors):
    identity, beginning, anchors, windows, early, late, sham = record
    prediction = predict(identity)
    logits = prediction.resolution + priors['resolution']
    losses = []
    for name, anchor in anchors.items():
        window = windows[name]
        head = logits[:, anchor]
        losses.append(interval_nll(head, beginning, *window) if window is not None
                      else censored_nll(head, beginning, len(head)))
    if early in anchors and late in anchors and windows[early] is not None:
        start, end = windows[early]
        positive = torch.logsumexp(logits[start:end, anchors[early]], dim=0)
        competing = torch.logsumexp(logits[start:end, anchors[late]], dim=0)
        losses.append(0.5 * F.softplus(competing - positive))
        if sham is not None and sham[0] >= beginning:
            negative = torch.logsumexp(logits[sham[0]:sham[1], anchors[early]], dim=0) - log(sham[1] - sham[0])
            losses.append(0.5 * F.softplus(negative - positive + log(end - start)))
    return torch.stack(losses).mean()


def next_batches(groups, generator):
    """Interleave equally frequent task updates with replacement only across complete sweeps."""
    size = max(map(len, groups))
    for i in range(size):
        for group in groups:
            if i % len(group) == 0:
                generator.shuffle(group)
            yield group[i % len(group)]


def replace_json(path, value):
    pending = path.with_suffix('.pending')
    write_json(pending, value)
    pending.replace(path)


def stable_losses(history, tolerance):
    if len(history) < 9:
        return False
    for task in history[-1]['losses']:
        old = min(row['losses'][task] for row in history[:-8])
        recent = [row['losses'][task] for row in history[-8:]]
        if old - min(recent) >= tolerance or max(recent) - min(recent) >= 5 * tolerance:
            return False
    return True


def train_arm(runtime, cache, train, priors, settings, mode, output):
    torch.manual_seed(settings['seed'])
    reader = TokenReadout().to('cuda')
    adapter = runtime.new_adapter(rank=4).to('cuda') if mode == 'lora' else None
    pairs, evidence = prepare_train(train, cache)
    generator = random.Random(settings['seed'])
    optim_groups = [{'params': reader.parameters(), 'lr': settings['head_lr']}]
    if adapter is not None:
        optim_groups.append({'params': adapter.parameters(), 'lr': settings['adapter_lr']})
    optimizer = torch.optim.AdamW(optim_groups, weight_decay=settings['weight_decay'])
    trace = []
    started = monotonic()
    deadline = started + settings[f'{mode}_seconds']
    best_value, best_epoch = float('inf'), None

    def predict(identity):
        return forward(reader, runtime, placement(cache[identity], 'cuda'), adapter)

    def checkpoint():
        path = output / f'{mode}.pt'
        torch.save({'reader': reader.state_dict(), 'adapter': adapter.state_dict() if adapter is not None else None,
                    'priors': priors, 'mode': mode}, path.with_suffix('.pending'))
        path.with_suffix('.pending').replace(path)

    for epoch in range(1, settings['epochs'] + 1):
        if epoch > 1 and monotonic() >= deadline:
            break
        totals = {'first_report': [], 'support': []}
        gradients = []
        for i, item in enumerate(next_batches((pairs, evidence), generator)):
            optimizer.zero_grad(set_to_none=True)
            task = 'first_report' if i % 2 == 0 else 'support'
            loss = response_likelihood(item, predict, priors) if task == 'first_report' else evidence_likelihood(item, predict, priors)
            if not torch.isfinite(loss):
                raise ArithmeticError(f'Nonfinite {mode} {task} loss')
            loss.backward()
            gradients.append(float(nn_clip(reader, adapter)))
            optimizer.step()
            totals[task].append(float(loss.detach()))
        with torch.no_grad():
            losses = {'first_report': float(np.mean([float(response_likelihood(item, predict, priors)) for item in pairs])),
                      'support': float(np.mean([float(evidence_likelihood(item, predict, priors)) for item in evidence]))}
        mean_loss = float(np.mean(list(losses.values())))
        if mean_loss < best_value:
            best_epoch, best_value = epoch, mean_loss
            checkpoint()
        row = {'epoch': epoch, 'updates': len(gradients), 'losses': losses,
               'task_updates': {task: len(values) for task, values in totals.items()},
               'gradient_median': float(np.median(gradients)), 'elapsed_seconds': monotonic() - started}
        trace.append(row)
        print(mode, row, flush=True)
        replace_json(output / f'{mode}-history.json', trace)
        if stable_losses(trace, settings['minimum_epoch_gain']):
            break
    if best_epoch is None:
        raise RuntimeError(f'{mode} did not complete an epoch')
    value = torch.load(output / f'{mode}.pt', map_location='cuda', weights_only=True)
    reader.load_state_dict(value['reader'])
    if adapter is not None:
        adapter.load_state_dict(value['adapter'])
    return reader.eval(), adapter, {'mode': mode, 'selected_epoch': best_epoch,
                                    'per_task_training_plateau': stable_losses(trace, settings['minimum_epoch_gain']),
                                    'epochs': len(trace),
                                    'seconds': monotonic() - started}


def nn_clip(reader, adapter):
    parameters = list(reader.parameters())
    if adapter is not None:
        parameters.extend(adapter.parameters())
    return torch.nn.utils.clip_grad_norm_(parameters, 5.0, error_if_nonfinite=True)


def unlocked_scores(features, mode, runtime, readout, adapter):
    if mode != 'position':
        return forward(readout, runtime, features, adapter)
    t = len(features.token_ids)
    z = features.hidden.new_zeros(t)
    return Scores(z, z.expand(t, t), z)


@torch.no_grad()
def forecasts(runtime, cache, layout, readout, adapter, priors, mode):
    result = {'frick': {}, 'evidence': {}, 'controls': {}}
    for assay in ('frick', 'evidence', 'controls'):
        for row in layout[assay]:
            identity = row['id']
            features = placement(cache[identity], 'cuda')
            scores = unlocked_scores(features, mode, runtime, readout, adapter)
            if assay == 'frick':
                spans = event_spans(row, features.offsets)
                logits = first_report_logits(scores.opening, spans, sentence_prior(priors, len(spans), 'cuda'))
                predicted = {'first_report': phase_first_event(logits).exp().tolist(), 'hazard': logits.sigmoid().tolist()}
            elif assay == 'evidence':
                beginning, anchors = evidence_coordinates(row, features.offsets)
                predicted = {'offsets': features.offsets, 'risk_start': beginning,
                             'hazard': {query: (scores.resolution[:, anchor] + priors['resolution']).sigmoid().tolist()
                                        for query, anchor in anchors.items()}}
            else:
                anchor = intervals(features.offsets, *row['anchor_span'])[1] - 1
                opening, unresolved, resolved = question_mass(scores.opening + priors['opening'],
                                                              scores.resolution + priors['resolution'])
                predicted = {'offsets': features.offsets, 'anchor': anchor,
                             'anchor_unresolved': unresolved[:, anchor].tolist(),
                             'anchor_resolved': resolved[:, anchor].tolist(),
                             'opening': opening.tolist(), 'total_unresolved': unresolved.sum(1).tolist()}
            result[assay][identity] = predicted
    return result


def run(candidate, model, stimuli, output, protocol_sha256, training, layout, settings):
    if candidate != 'qwen' or not torch.cuda.is_available() or torch.cuda.get_device_name() != 'NVIDIA L4':
        raise RuntimeError('This experiment requires the pinned Qwen on one NVIDIA L4')
    settings = read_json(settings)
    train, layout = read_json(training), read_json(layout)
    output.mkdir(exist_ok=False)
    runtime = QwenRuntime.load(model)
    inputs = list(stream_rows(stimuli))
    if not inputs or len({row['id'] for row in inputs}) != len(inputs) or any(set(row) != {'id', 'text'} for row in inputs):
        raise ValueError('The stimulus packet must contain distinct labels-free texts')
    controls = [runtime.validate(text) for text in settings['technical_controls']]
    write_json(output / 'causality.json', {'checks': controls})
    if settings.get('smoke'):
        write_json(output / 'seal.json', {'protocol_sha256': protocol_sha256, 'smoke': True,
                                        'causality_sha256': digest(output / 'causality.json')})
        return
    cache = {}
    started = monotonic()
    with torch.no_grad():
        for row in inputs:
            cache[row['id']] = placement(runtime.encode(row['text']), 'cpu')
    print(f'Encoded {len(cache)} stimuli in {monotonic() - started:.1f}s', flush=True)
    priors = calibrate(train['frick'], train['evidence'], cache)
    write_json(output / 'forecast-position.json', forecasts(runtime, cache, layout, None, None, priors, 'position'))
    fitting = []
    for mode in ('frozen', 'lora'):
        try:
            reader, adapter, record = train_arm(runtime, cache, train, priors, settings, mode, output)
            write_json(output / f'forecast-{mode}.json', forecasts(runtime, cache, layout, reader, adapter, priors, mode))
            fitting.append(record)
        except Exception as error:
            if mode != 'lora':
                raise
            fitting.append({'mode': mode, 'error': str(error)})
            print(f'LoRA failed: {error}', flush=True)
    write_json(output / 'fitting.json', {'fits': fitting, 'documents': len(cache),
                                        'tokens': sum(len(row.token_ids) for row in cache.values()),
                                        'peak_reserved_bytes': torch.cuda.max_memory_reserved()})
    write_json(output / 'seal.json', {'protocol_sha256': protocol_sha256, 'smoke': False,
                                    'files_sha256': {p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()}})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--protocol-sha256', required=True)
    for name in ('model', 'stimuli', 'output', 'training', 'layout', 'settings'):
        parser.add_argument('--' + name, type=Path, required=True)
    run(**vars(parser.parse_args()))
