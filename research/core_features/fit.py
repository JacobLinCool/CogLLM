"""Fit frozen-Qwen readouts for lexical judgments and deliberate rereading."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from time import perf_counter

import numpy as np

from research.stage3.probe import _jsonl
from research.stage3.run import save_json


PENALTY = 0.002
STEPS = 110
WORD = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*", re.UNICODE)
CWI_METRICS = ('length_frequency', 'qwen_pooled', 'background_only', 'qwen_group_offset', 'qwen_group_interaction')


def train_score(design, y, weights, device):
    import torch

    x = torch.as_tensor(design, dtype=torch.float64, device=device)
    target = torch.as_tensor(y, dtype=torch.float64, device=device)
    mass = torch.as_tensor(weights, dtype=torch.float64, device=device)
    theta = torch.zeros(x.shape[-1], dtype=x.dtype, device=device, requires_grad=True)
    optimizer = torch.optim.LBFGS([theta], max_iter=STEPS, line_search_fn='strong_wolfe',
                                  tolerance_grad=1e-8, tolerance_change=1e-10)
    if x.ndim == 2:
        def objective(logits):
            return torch.nn.functional.binary_cross_entropy_with_logits(logits, target, weight=mass, reduction='sum')
    else:
        eligible = torch.as_tensor(design[..., 0] == 1, device=device)

        def objective(logits):
            log_probability = torch.log_softmax(logits.masked_fill(~eligible, -torch.inf), dim=1)
            return -(target * log_probability.masked_fill(~eligible, 0)).sum()

    denominator = mass.sum()
    if denominator <= 0:
        raise ValueError('No labeled training observations')

    def closure():
        optimizer.zero_grad()
        error = objective(x @ theta) / denominator + (PENALTY / 2) * theta[1:].square().sum()
        error.backward()
        return error

    optimizer.step(closure)
    final = closure()
    if not torch.isfinite(final).item():
        raise ArithmeticError('Nonfinite fitted objective')
    return theta.detach().cpu().numpy(), {'objective': final.item(), 'gradient_max': theta.grad.abs().max().item()}


def sigmoid(logit):
    return np.exp(-np.logaddexp(0, -logit))


def canonical_stops(path):
    rows = list(_jsonl(path))
    documents = {row['id']: row for row in rows}
    if len(documents) != len(rows):
        raise ValueError('Duplicate stimulus IDs')
    return documents


def read_features(path):
    with np.load(path, allow_pickle=False) as result:
        state = result['states']
        norms = np.linalg.norm(state, axis=1, keepdims=True)
        if not np.all(np.isfinite(norms) & (norms > 0)):
            raise ValueError('Expected finite nonzero frozen states')
        states = {key: vector for key, vector in zip(
            zip(result['ids'].tolist(), result['stops'].tolist(), strict=True), state / norms, strict=True,
        )}
        if len(states) != len(state):
            raise ValueError('Duplicate encoded character stops')
    return states


def fit_scaler(train_values):
    center = np.mean(train_values, axis=0)
    scale = np.maximum(np.std(train_values, axis=0), 0.1)
    return center, scale


def apply_scaler(values, center, scale):
    return (values - center) / scale / np.sqrt(values.shape[-1])


def standardized(train_values, values):
    return apply_scaler(values, *fit_scaler(train_values))


def cwi_base(text, row, corpus):
    span = text[row['start']:row['stop']]
    words = [word.casefold() for word in WORD.findall(span)]
    counts = [corpus[word] for word in words] or [0]
    return [np.log1p(len(span)), np.log1p(len(words)), np.log1p(row['start']),
            np.log1p(min(counts)), np.mean(np.log1p(counts)),
            float(span.isupper()), float(any(char.isdigit() for char in span))]


def cwi_design(method, base, states):
    n = len(base)
    intercept = np.ones((n, 2, 1))
    group = np.broadcast_to(np.array([-1.0, 1.0])[None, :, None], (n, 2, 1))
    local = np.broadcast_to(base[:, None, :], (n, 2, base.shape[1]))
    hidden = np.broadcast_to(states[:, None, :], (n, 2, states.shape[1]))
    if method == 'length_frequency':
        return np.concatenate((intercept, local), axis=2)
    if method == 'background_only':
        return np.concatenate((intercept, group), axis=2)
    if method == 'qwen_pooled':
        return np.concatenate((intercept, local, hidden), axis=2)
    if method == 'qwen_group_offset':
        return np.concatenate((intercept, group, local, hidden), axis=2)
    if method == 'qwen_group_interaction':
        return np.concatenate((intercept, group, local, hidden, group * local, group * hidden), axis=2)
    raise ValueError(f'Unknown scoring method {method}')


def cwi_corpus(documents, train):
    return Counter(word.casefold() for key in {row['id'] for row in train}
                   for word in WORD.findall(documents[key]['text']))


def cwi_features(documents, states, rows, corpus):
    accepted = [row for row in rows if (row['id'], row['stop']) in states]
    if not accepted:
        raise ValueError('No eligible CWI spans in split')
    lexical, hidden = [], []
    keys = set()
    for row in accepted:
        key = (row['id'], row['start'], row['stop'])
        text = documents[row['id']]['text']
        if not 0 <= row['start'] < row['stop'] <= len(text) or key in keys:
            raise ValueError(f'Invalid or duplicate CWI span {key}')
        lexical.append(cwi_base(text, row, corpus))
        hidden.append(states[(row['id'], row['stop'])])
        keys.add(key)
    return accepted, np.asarray(lexical), np.asarray(hidden)


def cwi_targets(rows):
    totals = np.array([[row['native_total'], row['nonnative_total']] for row in rows])
    marked = np.array([[row['native_marked'], row['nonnative_marked']] for row in rows])
    if not np.all((totals > 0) & (marked >= 0) & (marked <= totals)):
        raise ValueError('Invalid CWI training counts')
    return marked / totals, totals


def cwi(documents, states, train, evaluation, output, device):
    corpus = cwi_corpus(documents, train)
    train_rows, train_base, h_train = cwi_features(documents, states, train, corpus)
    eval_rows, eval_base, h_eval = cwi_features(documents, states, evaluation, corpus)
    center, scale = fit_scaler(train_base)
    x_train = apply_scaler(train_base, center, scale)
    x_eval = apply_scaler(eval_base, center, scale)
    values, totals = cwi_targets(train_rows)
    result = [{key: row[key] for key in ('id', 'start', 'stop')} for row in eval_rows]
    fit_stats = {}
    for method in CWI_METRICS:
        design = cwi_design(method, x_train, h_train)
        weights, fit_stats[method] = train_score(
            design.reshape(-1, design.shape[-1]), values.ravel(), totals.ravel(), device,
        )
        estimated = sigmoid(cwi_design(method, x_eval, h_eval) @ weights)
        for row, p in zip(result, estimated, strict=True):
            row[method] = p.tolist()
        print(f'Fit CWI {method}', flush=True)
    write_rows(output / 'cwi-predictions.jsonl', result)
    return {'rows_train': len(train_rows), 'rows_evaluated': len(eval_rows),
            'train_char_boundary_exclusions': len(train) - len(train_rows),
            'evaluation_char_boundary_exclusions': len(evaluation) - len(eval_rows), 'methods': fit_stats}


def readback_design(documents, states, keys):
    base, hidden, destinations = [], [], []
    maximum = max(len(doc['stops']) for doc in documents.values())
    candidate = np.zeros((len(keys), maximum, 6), dtype=np.float32)
    pair = np.zeros((len(keys), maximum, 2048), dtype=np.float32)
    for i, (key, stop) in enumerate(keys):
        doc = documents[key]
        prior = [end for end in doc['stops'] if end < stop]
        current = states[(key, stop)]
        if not prior:
            raise ValueError(f'No prior region available at {key}:{stop}')
        text = doc['text'][prior[-1]:stop].strip()
        base.append([np.log1p(len(prior)), np.log1p(stop), np.log1p(len(text)), np.log1p(len(WORD.findall(text)))])
        hidden.append(current)
        destinations.append(prior)
        previous_end = 0
        for source, end in enumerate(prior):
            distance = len(prior) - source
            candidate[i, source] = [1, np.log1p(distance), float(distance == 1), float(source == 0),
                                    np.log1p(stop - end), np.log1p(end - previous_end)]
            # Elementwise source/trigger interactions vary across candidate destinations.
            interaction = states[(key, end)] * current
            pair[i, source] = interaction / (np.linalg.norm(interaction) + 1e-12)
            previous_end = end
    return np.asarray(base), np.asarray(hidden), candidate, pair, destinations


def readback(documents, states, train, evaluation, output, device):
    def position(row):
        return row['id'], row['stop']

    complete = {key for key, doc in documents.items() if all((key, end) in states for end in doc['stops'])}
    train_rows = [row for row in train if row['id'] in complete]
    train_keys = sorted({position(row) for row in train_rows})
    eval_keys = sorted({position(row) for row in evaluation if row['id'] in complete})
    if not train_keys or not eval_keys:
        raise ValueError('Both readback splits need fully certified stimuli')
    tr_base, tr_h, tr_z, tr_pair, tr_d = readback_design(documents, states, train_keys)
    ev_base, ev_h, ev_z, ev_pair, ev_d = readback_design(documents, states, eval_keys)
    tr_base, ev_base = standardized(tr_base, tr_base), standardized(tr_base, ev_base)
    by_position = {key: i for i, key in enumerate(train_keys)}
    observed = np.zeros(len(train_keys))
    regressions = np.zeros(len(train_keys))
    destination_counts = np.zeros(tr_z.shape[:2])
    for row in train_rows:
        i = by_position[position(row)]
        observed[i] += 1
        if row['readback']:
            regressions[i] += 1
            destination_counts[i, tr_d[i].index(row['source_stop'])] += 1
    if not np.all(destination_counts.sum(axis=1) == regressions):
        raise ValueError('Each first backward event needs one furthest-back destination')
    predictions = [{'id': key, 'stop': end, 'source_stops': possible}
                   for (key, end), possible in zip(eval_keys, ev_d, strict=True)]
    fit_stats = {}
    for name, fit, target in (
        ('event_prior', np.ones((len(tr_base), 1)), np.ones((len(ev_base), 1))),
        ('event_position', np.column_stack((np.ones(len(tr_base)), tr_base)),
         np.column_stack((np.ones(len(ev_base)), ev_base))),
        ('event_qwen', np.column_stack((np.ones(len(tr_base)), tr_base, tr_h)),
         np.column_stack((np.ones(len(ev_base)), ev_base, ev_h))),
    ):
        weight, fit_stats[name] = train_score(fit, regressions / observed, observed, device)
        for row, p in zip(predictions, sigmoid(target @ weight), strict=True):
            row[name] = float(p)
    for name, fit, target in (
        ('source_distance', tr_z, ev_z),
        ('source_qwen', np.concatenate((tr_z, tr_pair), axis=2), np.concatenate((ev_z, ev_pair), axis=2)),
    ):
        # A position with no observed regression contributes no destination loss.
        eligible = regressions > 0
        weight, fit_stats[name] = train_score(fit[eligible], destination_counts[eligible], regressions[eligible], device)
        for row, logits in zip(predictions, target @ weight, strict=True):
            logits = logits[:len(row['source_stops'])]
            probability = np.exp(logits - logits.max())
            row[name] = (probability / probability.sum()).tolist()
    write_rows(output / 'readback-predictions.jsonl', predictions)
    return {'fit_rows': len(train_rows), 'fit_events': int(observed.sum()), 'fit_backwards': int(regressions.sum()),
            'test_landmarks': len(eval_keys), 'excluded_stimuli': sorted(set(documents) - complete), 'methods': fit_stats}


def write_rows(path, rows):
    with path.open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False, separators=(',', ':')) + '\n')


def run(stimuli_path, feature_dir, cwi_train, cwi_eval, readback_train, readback_eval, output):
    import torch

    timer = perf_counter()
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name(0) != 'NVIDIA L4':
        raise RuntimeError('Expected one NVIDIA L4 for both fits')
    torch.set_num_threads(2)
    device = torch.device('cuda')
    output.mkdir(parents=True, exist_ok=False)
    feature_path = feature_dir / 'features.npz'
    extraction = json.loads((feature_dir / 'extraction.json').read_text())
    if extraction['input_sha256'] != hashlib.sha256(stimuli_path.read_bytes()).hexdigest():
        raise ValueError('Extracted representations refer to a different stimulus inventory')
    documents = canonical_stops(stimuli_path)
    states = read_features(feature_path)
    result = {}
    for name, method, train_path, eval_path in (
        ('cwi', cwi, cwi_train, cwi_eval), ('readback', readback, readback_train, readback_eval),
    ):
        train, evaluation = list(_jsonl(train_path)), list(_jsonl(eval_path))
        if any(row['split'] != 'train' for row in train) or {row['group_id'] for row in train} & {
            row['group_id'] for row in evaluation
        }:
            raise ValueError(f'Unexpected {name} training role or contaminated split')
        fields = {'id', 'start', 'stop', 'group_id', 'split'} if name == 'cwi' else {'id', 'stop', 'group_id', 'split'}
        if any(set(row) != fields or row['split'] not in {'test', 'dev'} for row in evaluation):
            raise ValueError(f'Unexpected information in {name} evaluation input')
        docs = {row['id'] for row in train + evaluation}
        result[name] = method({key: documents[key] for key in docs}, states, train, evaluation, output, device)
    result['gpu'] = torch.cuda.get_device_name(0)
    result['fit_wall_s'] = perf_counter() - timer
    result['penalty'] = PENALTY
    result['max_lbfgs_iterations'] = STEPS
    result['forecast_sha256'] = {
        name: hashlib.sha256((output / name).read_bytes()).hexdigest()
        for name in ('cwi-predictions.jsonl', 'readback-predictions.jsonl')
    }
    save_json(output / 'fitting.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('stimuli', 'features', 'cwi-train', 'cwi-eval', 'readback-train', 'readback-eval', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    args = parser.parse_args()
    run(args.stimuli, args.features, args.cwi_train, args.cwi_eval, args.readback_train, args.readback_eval, args.output)
