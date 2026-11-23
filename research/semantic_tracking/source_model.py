"""Rank earlier words given the causal stop of a marked discourse inconsistency."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from time import perf_counter

import numpy as np
import torch

from research.core_features.fit import PENALTY, STEPS, canonical_stops, read_features, write_rows
from research.stage3.probe import _jsonl
from research.stage3.run import save_json


METHODS = ('uniform', 'distance_word', 'qwen')
WORD = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", re.UNICODE)


def end_word(text, end):
    # Preserve the candidate's identity even when it starts immediately after punctuation.
    previous_space = text.rfind(' ', 0, end)
    matches = list(WORD.finditer(text, max(previous_space, 0), end))
    return matches[-1].group().casefold() if matches and matches[-1].end() == end else ''


def design(rows, documents, states):
    indices = []
    lexical = []
    hidden = []
    rejections = []
    accepted = []
    for row in rows:
        doc, stop, candidates = row['id'], row['stop'], row['source_candidates']
        if not candidates or candidates != sorted(set(candidates)) or not 0 < candidates[0] <= candidates[-1] < stop:
            raise ValueError(f'Invalid causal choice set for {doc}:{stop}')
        text = documents[doc]['text']
        if stop > len(text):
            raise ValueError(f'Out-of-text stop for {doc}')
        if (doc, stop) not in states:
            rejections.append({'row_id': row['row_id'], 'reason': 'uncertified_current_stop'})
            continue
        # Exclude a whole comparison if it would lose an unverified candidate.
        if any((doc, end) not in states for end in candidates):
            rejections.append({'row_id': row['row_id'], 'reason': 'uncertified_earlier_candidate'})
            continue
        query = states[(doc, stop)]
        context = Counter(token.group().casefold() for token in WORD.finditer(text, max(0, stop - 60), stop))
        for i, end in enumerate(candidates):
            word = end_word(text, end)
            earlier = states[(doc, end)]
            interaction = earlier * query
            lexical.append([np.log1p(stop - end), np.log1p(len(candidates) - i),
                            np.log1p(len(word)), np.log1p(context[word]),
                            float(i == len(candidates) - 1), float(i == 0)])
            hidden.append(interaction / (np.linalg.norm(interaction) + 1e-12))
            indices.append(len(accepted))
        accepted.append(row)
    if not accepted:
        raise ValueError('No candidate sets survived causal alignment')
    return accepted, np.asarray(indices, np.int64), np.asarray(lexical), np.asarray(hidden), rejections


def normalize(lexical_train, matrix):
    return (matrix - lexical_train.mean(axis=0)) / np.maximum(lexical_train.std(axis=0), 0.1) / np.sqrt(matrix.shape[1])


def gold_labels(row):
    # Source endpoints are validated against the exact earlier words at preparation.
    return [row['source_start'] < end <= row['source_stop'] for end in row['source_candidates']]


def fit(features, indices, labels, device):
    x = torch.as_tensor(features, dtype=torch.float64, device=device)
    group = torch.as_tensor(indices, device=device)
    gold = torch.as_tensor(labels, device=device)
    size = int(indices[-1]) + 1
    theta = torch.zeros(x.shape[1], device=device, dtype=x.dtype, requires_grad=True)
    opt = torch.optim.LBFGS([theta], max_iter=STEPS, line_search_fn='strong_wolfe',
                            tolerance_grad=1e-8, tolerance_change=1e-10)

    def closure():
        opt.zero_grad()
        z = x @ theta
        maximum = torch.full((size,), -torch.inf, dtype=x.dtype, device=device)
        maximum.scatter_reduce_(0, group, z.detach(), reduce='amax', include_self=True)
        evidence = torch.exp(z - maximum[group])
        total = torch.zeros_like(maximum).scatter_add_(0, group, evidence)
        positive = torch.zeros_like(maximum).scatter_add_(0, group[gold], evidence[gold])
        loss = (torch.log(total) - torch.log(positive)).mean() + (PENALTY / 2) * theta.square().sum()
        loss.backward()
        return loss

    opt.step(closure)
    loss = closure()
    if not torch.isfinite(loss).item():
        raise RuntimeError('Source pointer objective did not remain finite')
    return theta.detach().cpu().numpy(), {'objective': loss.item(), 'gradient_max': theta.grad.abs().max().item()}


def run(stimuli, features, train, evaluation, output):
    timer = perf_counter()
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name(0) != 'NVIDIA L4':
        raise RuntimeError('This experiment expects a single NVIDIA L4')
    torch.set_num_threads(2)
    output.mkdir(parents=True, exist_ok=False)
    documents = canonical_stops(stimuli)
    states = read_features(features)
    training = list(_jsonl(train))
    tests = list(_jsonl(evaluation))
    keys = ('row_id', 'id', 'group_id', 'stop', 'source_candidates')
    if not training or not tests or any(row.keys() != set(keys) for row in tests) or (
        {row['group_id'] for row in training} & {row['group_id'] for row in tests}
    ):
        raise ValueError('Expected source labels confined to disjoint training families')
    if len(training + tests) != len({row['row_id'] for row in training + tests}):
        raise ValueError('Source row identifiers must be unique')
    training, tr_index, tr_lex, tr_h, rejected_train = design(training, documents, states)
    tests, _, test_lex, test_h, rejected_test = design(tests, documents, states)
    labels = [gold_labels(row) for row in training]
    if not all(any(bits) for bits in labels):
        raise ValueError('Each pointer training row needs an earlier matched gold word')
    labels = np.concatenate(labels)
    tr_x, test_x = normalize(tr_lex, tr_lex), normalize(tr_lex, test_lex)
    result = [{key: row[key] for key in keys} for row in tests]
    stats = {}
    weights = {}
    lens = np.asarray([len(row['source_candidates']) for row in result])
    breaks = np.cumsum(lens)[:-1]
    for name, fit_x, eval_x in (
        ('distance_word', tr_x, test_x),
        ('qwen', np.concatenate((tr_x, tr_h), axis=1), np.concatenate((test_x, test_h), axis=1)),
    ):
        weight, stats[name] = fit(fit_x, tr_index, labels, torch.device('cuda'))
        weights[name] = weight
        scores = eval_x @ weight
        for row, row_scores in zip(result, np.split(scores, breaks), strict=True):
            evidence = np.exp(row_scores - row_scores.max())
            row[name] = (evidence / evidence.sum()).tolist()
            row['uniform'] = [1 / len(evidence)] * len(evidence)
    predictions = output / 'source-predictions.jsonl'
    write_rows(predictions, result)
    np.savez(output / 'source-heads.npz', **weights, lexical_mean=tr_lex.mean(axis=0),
             lexical_scale=np.maximum(tr_lex.std(axis=0), 0.1))
    save_json(output / 'source-fit.json', {
        'predictions_sha256': hashlib.sha256(predictions.read_bytes()).hexdigest(),
        'train_rows': len(training), 'evaluation_rows': len(tests),
        'excluded_train': rejected_train, 'excluded_test': rejected_test,
        'gpu': torch.cuda.get_device_name(0), 'p': PENALTY, 'seconds': perf_counter() - timer, 'methods': stats,
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('stimuli', 'features', 'train', 'evaluation', 'output'):
        parser.add_argument(f'--{key}', type=Path, required=True)
    run(**vars(parser.parse_args()))
