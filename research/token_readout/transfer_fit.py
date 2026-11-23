"""Fit matched portable readouts on one frozen Colab backbone."""

import argparse
from collections import defaultdict
from dataclasses import replace
import gc
from math import ceil, log
from pathlib import Path
import random
from time import monotonic

import torch
from torch.nn import functional as F

from research.backbone_transfer.extract import digest, read_json, stream_rows, write_json
from research.token_readout.backbone import Backbone
from research.token_readout.model import (
    Scores, censored_nll, first_report_logits, first_report_nll, interval_nll, matched_effect, phase_first_event,
)
from research.token_readout.readout import PortableReadout


ENDPOINTS = ('frick', 'evidence')
VARIANTS = ('point', 'span')
DEFAULT_SCALE = {'opening_gain': 1.0, 'resolution_gain': 1.0, 'resolution_bias': 0.0}
CHECKPOINT_SCHEMA = 2


def smoothed_log_odds(events, exposure):
    if not 0 <= events <= exposure:
        raise ValueError('Observed events must be inside the risk set')
    return log((events + 0.5) / (exposure - events + 0.5))


def observations(packet):
    stories = defaultdict(list)
    for row in packet['frick']:
        stories[row['group_id']].append(row)
    frick = [sorted(group, key=lambda row: row['condition']) for group in stories.values()]
    if any(len(group) != 2 for group in frick) or not all((frick, packet['evidence'])):
        raise ValueError('Both observation endpoints require data and complete story pairs')
    return {'frick': frick, 'evidence': packet['evidence']}


def priors_for(training, cache):
    by_sentence = defaultdict(lambda: [0, 0])
    openings, opening_exposure, supports, support_exposure = 0, 0, 0, 0
    for pair in training['frick']:
        for row in pair:
            events = row['event_counts']
            risk = sum(events) + row['censor_count']
            for i, (count, (begin, end)) in enumerate(zip(events, row['sentence_tokens'], strict=True)):
                by_sentence[i][0] += count
                by_sentence[i][1] += risk
                openings += count
                opening_exposure += risk * (end - begin)
                risk -= count
            if risk != row['censor_count']:
                raise ValueError('Frick exposure differs from the observed censor count')
    for row in training['evidence']:
        for window in row['window_tokens'].values():
            stop = len(cache[row['id']].offsets) if window is None else window[1]
            supports += window is not None
            support_exposure += stop - row['risk_start']
    if not openings or not supports:
        raise ValueError('Training requires observed events on both endpoints')
    return {
        'sentence': [smoothed_log_odds(*by_sentence[index]) for index in range(len(by_sentence))],
        'sentence_default': smoothed_log_odds(sum(value[0] for value in by_sentence.values()),
                                             sum(value[1] for value in by_sentence.values())),
        'opening': smoothed_log_odds(openings, opening_exposure),
        'resolution': smoothed_log_odds(supports, support_exposure),
    }


def sentence_logits(residuals, spans, priors, gain=1.0):
    values = priors['sentence'][:len(spans)]
    values = values + [priors['sentence_default']] * (len(spans) - len(values))
    return first_report_logits(residuals * gain, spans, residuals.new_tensor(values))


def report_values(rows, predict):
    return [(row, predict(row['id']).opening) for row in rows]


def report_loss(values, priors, gain=1.0, paired=False):
    logits, targets, losses = [], [], []
    for row, residue in values:
        hazard = sentence_logits(residue, row['sentence_tokens'], priors, gain)
        counts = residue.new_tensor([*row['event_counts'], row['censor_count']])
        losses.append(first_report_nll(hazard, counts))
        index = row['target_sentence_index'] - 1
        if not 0 <= index < len(hazard):
            raise ValueError('Frick target is outside its risk sequence')
        logits.append(hazard[index])
        targets.append(counts[index] / counts[index:].sum().clamp_min(1))
    base = torch.stack(losses).mean()
    return base + 0.25 * matched_effect(logits, targets) if paired else base


def support_values(row, predict):
    pair_logits = predict(row['id']).resolution
    return row, {name: pair_logits[:, span[1] - 1] for name, span in row['anchor_tokens'].items()}


def support_loss(item, priors, gain=1.0, bias=0.0, paired=False):
    row, residue = item
    begin = row['risk_start']
    logits = {name: values * gain + priors['resolution'] + bias for name, values in residue.items()}
    losses = []
    for name, window in row['window_tokens'].items():
        scores = logits[name]
        loss = censored_nll(scores, begin, len(scores)) if window is None else interval_nll(scores, begin, *window)
        losses.append(loss)
    if not paired:
        return torch.stack(losses).mean()
    early, late = row['earlier_query'], row['later_query']
    if early in logits and late in logits and row['window_tokens'][early] is not None:
        start, end = row['window_tokens'][early]
        positive = torch.logsumexp(logits[early][start:end], dim=0)
        competing = torch.logsumexp(logits[late][start:end], dim=0)
        losses.append(0.5 * F.softplus(competing - positive))
        if row['sham_tokens'] is not None:
            sham_start, sham_end = row['sham_tokens']
            negative = torch.logsumexp(logits[early][sham_start:sham_end], dim=0)
            negative = negative + log((end - start) / (sham_end - sham_start))
            losses.append(0.5 * F.softplus(negative - positive))
    return torch.stack(losses).mean()


def validation_values(groups, predict):
    with torch.no_grad():
        return {
            'frick': [report_values(rows, predict) for rows in groups['frick']],
            'evidence': [support_values(row, predict) for row in groups['evidence']],
        }


def validated_losses(data, priors, scale):
    report = [report_loss(group, priors, scale['opening_gain']) for group in data['frick']]
    support = [support_loss(pair, priors, scale['resolution_gain'], scale['resolution_bias'])
               for pair in data['evidence']]
    return {'frick': float(torch.stack(report).mean()), 'evidence': float(torch.stack(support).mean())}


def balanced_updates(groups, order):
    packets = {key: list(groups[key]) for key in ENDPOINTS}
    for step in range(max(map(len, packets.values()))):
        for source, rows in packets.items():
            if step % len(rows) == 0:
                order.shuffle(rows)
            yield source, rows[step % len(rows)]


def linked_stream(rows, seed):
    order = random.Random(seed)
    remaining = list(rows)
    while remaining:
        order.shuffle(remaining)
        yield from remaining


def checkpoint(reader, candidate, revision, variant, priors, scale, path):
    payload = {'schema': CHECKPOINT_SCHEMA, 'candidate': candidate, 'revision': revision, 'variant': variant,
               'relations': True, 'reader': reader.state_dict(), 'priors': priors, 'scale': scale}
    pending = path.with_suffix('.pending')
    torch.save(payload, pending)
    pending.replace(path)


def train_variant(candidate, revision, variant, cache, width, training, dev, linked, priors, settings, output):
    torch.manual_seed(settings['seed'])
    order = random.Random(settings['seed'])
    reader = PortableReadout(width, variant=variant, relations=True).cuda()
    optimizer = torch.optim.AdamW(reader.parameters(), lr=settings['lr'], weight_decay=settings['weight_decay'])
    additional = linked_stream(linked, settings['seed'])
    linked_batch = ceil(len(linked) / max(map(len, training.values())))
    history = []
    started = monotonic()
    best_score, best_epoch, stop = float('inf'), 0, 'epoch_limit'
    path = output / f'{variant}.pt'

    def predict(identity):
        return reader(cache[identity])

    for epoch in range(1, settings['epochs'] + 1):
        reader.train()
        losses = {source: [] for source in ENDPOINTS}
        auxiliary = []
        for source, row in balanced_updates(training, order):
            optimizer.zero_grad(set_to_none=True)
            if source == 'frick':
                loss = report_loss(report_values(row, predict), priors, paired=True)
            else:
                loss = support_loss(support_values(row, predict), priors, paired=True)
                if linked_batch:
                    interval_losses = [support_loss(support_values(next(additional), predict), priors, paired=True)
                                       for _ in range(linked_batch)]
                    auxiliary_loss = torch.stack(interval_losses).mean()
                    auxiliary.append(auxiliary_loss.detach())
                    loss = loss + 0.5 * auxiliary_loss
            if not torch.isfinite(loss):
                raise ArithmeticError(f'{candidate} {variant} {source} loss is nonfinite')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(reader.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
            losses[source].append(loss.detach())
        reader.eval()
        study = validated_losses(validation_values(dev, predict), priors, DEFAULT_SCALE)
        row = {'epoch': epoch, 'training_composite': {key: float(torch.stack(value).mean()) for key, value in losses.items()},
               'dev_nll': study, 'updates': {key: len(value) for key, value in losses.items()},
               'elapsed_seconds': monotonic() - started}
        if auxiliary:
            row['linked_auxiliary_nll'] = float(torch.stack(auxiliary).mean())
            row['linked_examples_used'] = len(auxiliary) * linked_batch
        score = sum(study.values()) / len(study)
        if not all(logit < float('inf') for logit in study.values()):
            raise ArithmeticError('Development objective is nonfinite')
        if score < best_score - settings['min_dev_gain']:
            best_score, best_epoch = score, epoch
            checkpoint(reader, candidate, revision, variant, priors, DEFAULT_SCALE, path)
        history.append(row)
        print(candidate, variant, row, flush=True)
        if epoch >= settings['minimum_epochs'] and epoch - best_epoch >= settings['patience']:
            stop = 'development_plateau'
            break
        if row['elapsed_seconds'] >= settings['arm_seconds']:
            stop = 'arm_time_limit'
            break
    write_json(output / f'{variant}-history.json', history)
    if best_epoch == 0:
        raise RuntimeError('No valid dev checkpoint was saved')
    packet = torch.load(path, map_location='cuda', weights_only=True)
    reader.load_state_dict(packet['reader'], strict=True)
    return reader.eval(), {'selected_epoch': best_epoch, 'epochs': len(history), 'stop': stop,
                           'dev_nll': history[best_epoch - 1]['dev_nll'],
                           'parameters': sum(value.numel() for value in reader.parameters()),
                           'seconds': monotonic() - started}


def choose_calibration(reader, cache, dev, priors):
    """Shrink extreme residuals and fit a support intercept using only development observations."""
    values = validation_values(dev, lambda identity: reader(cache[identity]))
    gains = (1.0, 0.75, 0.5, 0.3, 0.2, 0.12, 0.06)
    biases = (0.0, -0.75, 0.75, -1.5, 1.5, -2.25, 2.25, -3.0, 3.0)

    def frick(gain):
        losses = [report_loss(group, priors, gain) for group in values['frick']]
        return float(torch.stack(losses).mean())

    def evidence(gain, bias):
        losses = [support_loss(item, priors, gain, bias) for item in values['evidence']]
        return float(torch.stack(losses).mean())

    first = min(gains, key=frick)
    gain, bias = min(((gain, bias) for gain in gains for bias in biases),
                     key=lambda pair: evidence(*pair))
    scale = {'opening_gain': first, 'resolution_gain': gain, 'resolution_bias': bias}
    return scale, validated_losses(values, priors, scale)


def probabilities(logits):
    return logits.sigmoid().clamp(1e-8, 1 - 1e-7).tolist()


def forecasts(geometry, cache, priors, reader=None, scale=DEFAULT_SCALE):
    result = {kind: {} for kind in ('frick', 'evidence', 'controls', 'ab')}
    with torch.no_grad():
        for identity, row in geometry.items():
            if row['split'] != 'test':
                continue
            features = cache[identity]
            length = len(features.offsets)
            scores = reader(features) if reader is not None else None
            if row['kind'] == 'frick':
                residuals = features.hidden.new_zeros(length) if scores is None else scores.opening
                logits = sentence_logits(residuals, row['sentence_tokens'], priors, scale['opening_gain'])
                prediction = {'hazard': probabilities(logits), 'first_report': phase_first_event(logits).exp().tolist()}
            else:
                prediction = {'hazard': {}}
                for name, span in row['anchor_tokens'].items():
                    residuals = features.hidden.new_zeros(length) if scores is None else scores.resolution[:, span[1] - 1]
                    prediction['hazard'][name] = probabilities(residuals * scale['resolution_gain']
                                                               + priors['resolution'] + scale['resolution_bias'])
            result[row['kind']][identity] = prediction
    return result


def run(candidate, model, stimuli, output, protocol_sha256, learning, layout, settings):
    runtime = Backbone.load(candidate, model)
    revision, width = runtime.revision, runtime.width
    device = next(runtime.encoder.parameters()).device
    if device.type != 'cuda' or torch.cuda.get_device_name(device) != 'NVIDIA L4':
        raise RuntimeError('The portable experiment requires a verified Colab L4')
    settings, learning, layout = read_json(settings), read_json(learning), read_json(layout)
    output.mkdir(parents=True, exist_ok=False)
    geometry = layout['rows']
    texts = {row['id']: row['text'] for row in stream_rows(stimuli)}
    if layout['model'] != candidate or set(texts) != geometry.keys():
        raise ValueError('Stimuli and tokenizer geometry differ')
    controls = [runtime.validate(text) for text in settings['technical_controls']]
    cache, counted = {}, 0
    start = monotonic()

    def count(_module, _inputs):
        nonlocal counted
        counted += 1

    with runtime.encoder.register_forward_pre_hook(count):
        for identity, text in texts.items():
            prior_count = counted
            features = runtime.encode(text)
            if counted != prior_count + 1 or features.offsets != tuple(map(tuple, geometry[identity]['offsets'])):
                raise ValueError(f'Token coordinates or forward count changed for {identity}')
            if features.relations is None:
                raise ValueError('Comparative arms require observed native attention relations')
            cache[identity] = replace(features, hidden=features.hidden.clone(), mid=features.mid.clone(),
                                      relations=features.relations.clone())
    extraction_seconds = monotonic() - start
    tokens = sum(len(features.offsets) for features in cache.values())
    extraction_peaks = torch.cuda.max_memory_reserved()
    del runtime
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print(candidate, 'extracted', counted, 'documents and', tokens, 'native tokens', flush=True)
    write_json(output / 'causality.json', {'checks': controls, 'model': candidate, 'revision': revision,
                                          'document_forwards': counted, 'every_document_single_forward': True})
    training, dev = (observations(learning[split]) for split in ('train', 'dev'))
    linked = learning['train'].get('linked', [])
    for split, rows in (('train', learning['train']), ('dev', learning['dev'])):
        if any(geometry[row['id']]['split'] != split or geometry[row['id']]['kind'] != source
               for source, observations_for_source in rows.items() for row in observations_for_source):
            raise ValueError('Training or development payload contains a row from another split')
    if learning['dev'].get('linked') or len(linked) != sum(row['kind'] == 'linked' for row in geometry.values()):
        raise ValueError('IIRC supervision is restricted to the supplemental train split')
    priors = priors_for(training, cache)
    write_json(output / 'forecast-position.json', forecasts(geometry, cache, priors))
    fits = {}
    for variant in VARIANTS:
        reader, fits[variant] = train_variant(candidate, revision, variant, cache, width,
                                             training, dev, linked, priors, settings, output)
        write_json(output / f'forecast-{variant}.json', forecasts(geometry, cache, priors, reader))
    scale, dev_nll = choose_calibration(reader, cache, dev, priors)
    fits['span-calibrated'] = {'dev_nll': dev_nll, 'scale': scale, 'parameters': fits['span']['parameters'],
                               'selected_epoch': fits['span']['selected_epoch']}
    checkpoint(reader, candidate, revision, 'span', priors, scale, output / 'span-calibrated.pt')
    write_json(output / 'forecast-span-calibrated.json', forecasts(geometry, cache, priors, reader, scale))
    write_json(output / 'fitting.json', {
        'candidate': candidate, 'revision': revision, 'documents': counted, 'tokens': tokens, 'fits': fits,
        'linked_training_examples': len(linked),
        'selection_score': 'equal mean Frick and ExpMRC development negative log likelihood',
        'dev_position_nll': validated_losses(validation_values(dev, lambda identity: zero_scores(cache[identity])),
                                            priors, DEFAULT_SCALE),
        'peak_extraction_reserved_bytes': extraction_peaks, 'peak_fitting_reserved_bytes': torch.cuda.max_memory_reserved(),
        'extraction_seconds': extraction_seconds, 'device': torch.cuda.get_device_name(device),
    })
    write_json(output / 'seal.json', {'protocol_sha256': protocol_sha256, 'candidate': candidate, 'revision': revision,
                                     'files_sha256': {path.name: digest(path) for path in sorted(output.iterdir())}})


def zero_scores(features):
    length = len(features.offsets)
    opening = features.hidden.new_zeros(length)
    return Scores(opening, features.hidden.new_zeros(length, length), opening)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', required=True, choices=('qwen', 'lfm3', 'gemma4'))
    for name in ('model', 'stimuli', 'output', 'learning', 'layout', 'settings'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--protocol-sha256', required=True)
    run(**vars(parser.parse_args()))
