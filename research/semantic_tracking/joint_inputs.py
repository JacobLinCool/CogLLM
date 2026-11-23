"""Build matched, outcome-safe inputs for all semantic readout conditions."""

from collections import defaultdict
from dataclasses import dataclass
import re

import numpy as np
import torch

from research.stage3.probe import _jsonl
from .joint_model import INPUT_SIZE, projection, readout_features


TOKEN_WORD = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", re.UNICODE)


@dataclass
class Rows:
    keys: list
    inputs: torch.Tensor
    mask: torch.Tensor
    labels: torch.Tensor | None


def tensor_rows(rows, features, targets, device):
    size = max(1, max(map(len, features)))
    x = np.zeros((len(rows), size, INPUT_SIZE), dtype=np.float32)
    valid = np.zeros((len(rows), size), dtype=bool)
    for i, row in enumerate(features):
        x[i, :len(row)] = row
        valid[i, :len(row)] = True
    return Rows(rows, torch.as_tensor(x, device=device), torch.as_tensor(valid, device=device),
                None if targets is None else torch.as_tensor(targets, device=device))


def load_projected(path, expected):
    with np.load(path, allow_pickle=False) as data:
        ids = data['ids'].tolist()
        stops = data['stops'].tolist()
        raw = data['states']
        keys = list(zip(ids, stops, strict=True))
        if len(keys) != len(expected) or set(keys) != expected or raw.shape != (len(ids), 2048):
            raise ValueError('Expected exactly the frozen features certified for the previous studies')
        values = np.empty((len(keys), 48), dtype=np.float32)
        matrix = projection()
        for start in range(0, len(keys), 1024):
            source = raw[start:start + 1024]
            norms = np.linalg.norm(source, axis=1, keepdims=True)
            if not np.all(np.isfinite(norms) & (norms > 0)):
                raise ValueError('Expected finite nonzero hidden states')
            values[start:start + len(source)] = (source / norms) @ matrix
    return dict(zip(keys, values, strict=True))


def onset_rows(rows, texts, states, device, training):
    observations = defaultdict(list)
    for row in rows:
        stop = row['id'], row['stop']
        if not training and stop in observations:
            raise ValueError('Repeated outcome-free onset anchor')
        observations[stop].append(row)
    feature = []
    outcome = []
    keys = []
    for (key, end), trials in observations.items():
        doc = texts[key]
        index = doc['stops'].index(end)
        left = doc['stops'][index - 1] if index else 0
        width = len(doc['text'][left:end].strip())
        position = index + 1
        value = readout_features(end, 0, width, 0, position, position, 0, 1, states[(key, end)])
        feature.append(value[None, :])
        keys.append({'id': key, 'stop': end})
        if training:
            outcome.append([sum(row['readback'] for row in trials), len(trials)])
    return tensor_rows(keys, feature, np.asarray(outcome, np.float32) if training else None, device)


def source_rows(rows, texts, states, device, training):
    words = {key: {word.end(): word.end() - word.start() for word in TOKEN_WORD.finditer(texts[key]['text'])}
             for key in {row['id'] for row in rows}}
    features = []
    gold = []
    keys = []
    maximum = max(len(row['source_candidates']) for row in rows)
    for row in rows:
        key, stop = row['id'], row['stop']
        ends = np.asarray(row['source_candidates'])
        if not 0 < ends[0] <= ends[-1] < stop or np.any(np.diff(ends) <= 0):
            raise ValueError('Only ordered, strictly earlier source candidates are legal')
        lex = words[key]
        width = np.asarray([lex.get(end, 1) for end in ends])
        features.append(readout_features(stop, ends, lex.get(stop, 1), width, np.arange(1, len(ends) + 1),
                                         len(ends) - np.arange(len(ends)), 0, 1, states[key, stop],
                                         np.stack([states[key, end] for end in ends])))
        keys.append({field: row[field] for field in ('row_id', 'id', 'group_id', 'stop', 'source_candidates')})
        if training:
            labels = (row['source_start'] < ends) & (ends <= row['source_stop'])
            if not labels.any():
                raise ValueError('No gold word in the causal source set')
            gold.append(np.pad(labels, (0, maximum - len(ends))))
    return tensor_rows(keys, features, np.asarray(gold) if training else None, device)


def closure_rows(rows, device):
    features = []
    bounds = []
    keys = []
    for row in rows:
        origin, lower, upper, last = (row[key] for key in ('open_line', 'lower_line', 'upper_line', 'last_line'))
        if (not 0 < origin <= lower <= last or (upper is not None and not lower < upper <= last)
                or (upper if upper is not None else lower) - origin != row['line_opportunities']):
            raise ValueError(f'Invalid closure interval {row["id"]}')
        now = np.arange(origin + 1, last + 1)
        features.append(readout_features(now, origin, 0, 0, now, now - origin, 1, 0))
        bounds.append([lower - origin, -1 if upper is None else upper - origin])
        keys.append({key: row[key] for key in ('id', 'story', 'open_line', 'last_line')})
    return tensor_rows(keys, features, np.asarray(bounds, np.int64), device)


def load(stimuli, features, folder, device):
    docs = list(_jsonl(stimuli))
    texts = {row['id']: row for row in docs}
    if len(texts) != len(docs):
        raise ValueError('Repeated stimulus IDs')
    expected = {(row['id'], end) for row in docs for end in row['stops']}
    states = load_projected(features, expected)
    training, heldout = {}, {}
    for task, builder in (('onset', onset_rows), ('source', source_rows)):
        train = list(_jsonl(folder / f'{task}-train.jsonl'))
        evaluation = list(_jsonl(folder / f'{task}-eval.jsonl'))
        anchors = {'id', 'group_id', 'stop'}
        if task == 'source':
            anchors |= {'row_id', 'source_candidates'}
        if (any(row['split'] != 'train' for row in train) or any(row.keys() != anchors for row in evaluation)
                or {row['group_id'] for row in train} & {row['group_id'] for row in evaluation}):
            raise ValueError(f'Incompatible frozen data splits for {task}')
        training[task] = builder(train, texts, states, device, True)
        heldout[task] = builder(evaluation, texts, states, device, False)
    intervals = list(_jsonl(folder / 'closure-intervals.jsonl'))
    if len(intervals) != 196 or len({row['id'] for row in intervals}) != len(intervals):
        raise ValueError('Expected the 196 sealed Onea question trajectories')
    stories = sorted({row['story'] for row in intervals})
    if len(stories) != 3:
        raise ValueError('Exactly three directly comparable held-out stories are required')
    closure = {story: (closure_rows([row for row in intervals if row['story'] != story], device),
                       closure_rows([row for row in intervals if row['story'] == story], device)) for story in stories}
    return training, heldout, closure
