"""Fit explicit first-visit reverse-button readouts on the reanalysis corpus."""

import argparse
import hashlib
from pathlib import Path
from time import perf_counter

import torch

from research.core_features.fit import canonical_stops, read_features, readback
from research.stage3.probe import _jsonl
from research.stage3.run import save_json


def run(stimuli, features, train, evaluation, output):
    started = perf_counter()
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name(0) != 'NVIDIA L4':
        raise RuntimeError('Expected one NVIDIA L4')
    torch.set_num_threads(2)
    documents = canonical_stops(stimuli)
    states = read_features(features)
    training, tests = list(_jsonl(train)), list(_jsonl(evaluation))
    if any(row['split'] != 'train' for row in training) or any(set(row) != {'id', 'group_id', 'stop'} for row in tests):
        raise ValueError('Only training rows may carry labels')
    if {row['group_id'] for row in training} & {row['group_id'] for row in tests}:
        raise ValueError('The reanalysis split must hold out complete material families')
    docs = {row['id'] for row in training + tests}
    output.mkdir(parents=True, exist_ok=False)
    result = readback({key: documents[key] for key in docs}, states, training, tests, output, torch.device('cuda'))
    predictions = output / 'readback-predictions.jsonl'
    save_json(output / 'onset-fit.json', {
        **result, 'gpu': torch.cuda.get_device_name(0), 'seconds': perf_counter() - started,
        'predictions_sha256': hashlib.sha256(predictions.read_bytes()).hexdigest(),
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('stimuli', 'features', 'train', 'evaluation', 'output'):
        parser.add_argument(f'--{key}', type=Path, required=True)
    run(**vars(parser.parse_args()))
