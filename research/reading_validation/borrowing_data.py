"""Create group-disjoint partitions for the observation-specific borrowing experiment."""

from collections import defaultdict
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch

from research.backbone_transfer.extract import digest, read_json, stream_rows
from research.union_training.fit_observations import (COUNT_MODELS, load_projection, load_tasks,
                                                      observation_scales, task_responses)

PACKET = Path('.cache/cogllm/adaptive-joint/inputs/observations')
OUTPUT = Path('research/results/reading-validation-borrowing')
FIT, TUNE, CALIBRATE, EVALUATE = range(4)


def partitions(tasks):
    families = defaultdict(set)
    for task in tasks:
        families[task.source].update(task.family[task.role == 0])
    plan = {}
    for source, groups in families.items():
        def key(family):
            return sha256(f'cogllm-reading-borrowing-20261122|{source}|{family}'.encode()).digest()
        ordered = sorted(groups, key=key)
        n = min(max(1, round(.15 * len(ordered))), (len(ordered) - 1) // 2)
        for i, family in enumerate(ordered):
            plan[source, family] = TUNE if i < n else CALIBRATE if i < 2 * n else FIT
    return plan


def data():
    receipt = read_json(PACKET / 'receipt.json')
    for name, signature in receipt['packet_sha256'].items():
        if digest(PACKET / name) != signature:
            raise ValueError(f'Observation archive changed: {name}')
    cache, coordinates = load_projection(PACKET, 'cpu')
    tasks = load_tasks(PACKET, coordinates, 'cpu')
    plan = partitions(tasks)
    splits, documents = {}, {}
    for task in tasks:
        role = np.array([EVALUATE if old else plan[task.source, group]
                         for old, group in zip(task.role, task.family, strict=True)])
        if not np.any(role == FIT):
            raise ValueError(f'No remaining fit observations for {task.name}')
        splits[task.name] = role
        records = list(stream_rows(PACKET / f'{task.name}.jsonl'))
        for row, split in zip(records, role, strict=True):
            assignment = (task.source, row['family'], int(split))
            for key, _ in row['coordinates']:
                if documents.setdefault(key, assignment) != assignment:
                    raise ValueError('An exposure occurs in two source, family or partition assignments')
        if task.likelihood not in COUNT_MODELS:
            values = task_responses(task.likelihood, records, task.coordinates.shape[1])
            task.trials, task.center, task.spread = observation_scales(task.likelihood, values, role)
            task.response = torch.as_tensor(values, dtype=torch.float32)
    return cache, tasks, splits
