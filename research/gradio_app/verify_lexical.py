"""Replay the published CompLex target-end fit without loading Qwen."""

import argparse
import json
from pathlib import Path

import numpy as np

from research.stage3.run import save_json
from research.stage5.complex_score import _paired_delta
from research.stage3.probe import _jsonl

from .causal_plaba import digest
from .lexical import CHECKPOINT, LexicalReadout


ABSOLUTE_TOLERANCE = 2e-12


def run(features, design, observations, output):
    readout = LexicalReadout()
    manifest = json.loads((CHECKPOINT / 'fitting.json').read_text(encoding='utf-8'))
    prediction_path = CHECKPOINT / 'predictions.npz'
    scored = json.loads((CHECKPOINT.parent / 'scores.json').read_text(encoding='utf-8'))
    if (manifest['inputs']['features']['sha256'] != digest(features)
            or manifest['inputs']['design']['sha256'] != digest(design)
            or manifest['artifacts']['predictions.npz']['sha256'] != digest(prediction_path)
            or scored['input_sha256']['observations.jsonl'] != digest(observations)):
        raise ValueError('CompLex replay needs the immutable fitting inputs and predictions')
    held_out = {row['id']: row for row in _jsonl(design) if row['split'] == 'test'}
    if any(row['value'] is not None for row in held_out.values()):
        raise ValueError('Design must exclude the held-out lexical ratings')
    with np.load(prediction_path, allow_pickle=False) as prediction, np.load(features, allow_pickle=False) as extracted:
        ids = prediction['ids'].tolist()
        method = prediction['methods'].tolist().index('target_end')
        fraction = prediction['fractions'].tolist().index(100)
        expected = prediction['predictions'][method, fraction]
        positions = {identifier: row for row, identifier in enumerate(extracted['ids'].tolist())}
        if len(ids) != 1094 or len(set(ids)) != 1094 or set(ids) != set(held_out):
            raise ValueError('Incomplete CompLex held-out prediction inventory')
        replay = readout.score_states(extracted['target_end'][[positions[key] for key in ids]])
        np.testing.assert_allclose(expected, np.broadcast_to(replay, expected.shape), atol=ABSOLUTE_TOLERANCE, rtol=0)

    truth = {row['id']: row['observations']['lexical_difficulty']['value'] for row in _jsonl(observations)
             if row['split'] == 'test' and row['id'] in held_out}
    if set(truth) != set(held_out) or any(not 0 <= value <= 1 for value in truth.values()):
        raise ValueError('CompLex held-out lexical ratings differ from the frozen inventory')
    labels = np.asarray([truth[key] for key in ids])
    groups = np.asarray([held_out[key]['context_group'] for key in ids])
    errors = np.abs(replay - labels)
    differences = _paired_delta(errors[None, :] - np.abs(expected - labels), groups)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_json(output, {
        'scope': 'Existing held-out designated CompLex spans, cached L4 target-end states, one fixed head.',
        'head_sha256': readout.head_sha256, 'source_forecast_sha256': digest(prediction_path),
        'cached_features_sha256': digest(features), 'outcomes_sha256': digest(observations),
        'replay_count': len(ids), 'sentence_groups': len(set(groups)),
        'mae_0_to_1': float(errors.mean()), 'maximum_frozen_prediction_difference': float(np.max(np.abs(expected - replay))),
        'score_tolerance': ABSOLUTE_TOLERANCE, 'paired_absolute_error_replay_minus_source': differences,
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('features', 'design', 'observations', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    run(**vars(parser.parse_args()))
