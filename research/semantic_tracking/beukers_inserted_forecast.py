"""Forecast sealed grade-free inserted-block trials from the original Beukers readouts."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory

import numpy as np
from scipy.special import expit

from research.semantic_tracking import beukers_background as base
from research.semantic_tracking.beukers_factorized_replay import centroid, one_key, tag
from research.semantic_tracking.beukers_state_diagnostic import OPTIONS, examples_from_original


PROTOCOL = Path(__file__).with_name('beukers-inserted-protocol.md')
FIELDS = {'id', 'cohort', 'arm', 'reader_id', 'article_id', 'left_id', 'right_id', 'story', 'depth'}
ARMS = {'early': 1, 'middle': -1, 'late': -1}
COHORTS = {'original', 'replication'}
ID = re.compile(r'[0-9a-f]{32}')


def grade_free_trials(assay):
    assay_path = assay / 'assay.json'
    test_path = assay / 'test.jsonl'
    receipt = json.loads(assay_path.read_text(encoding='utf-8'))
    test_sha = base.digest(test_path)
    if receipt['file_sha256']['test.jsonl'] != test_sha:
        raise ValueError('Inserted-block geometry differs from the prepared assay')
    rows = base.read_rows(test_path)
    seen, reader_arms, conditions = set(), {}, defaultdict(set)
    for row in rows:
        if (not isinstance(row, dict) or row.keys() != FIELDS or row['cohort'] not in COHORTS or row['arm'] not in ARMS
                or any(type(row[key]) is not str or ID.fullmatch(row[key]) is None
                       for key in ('id', 'reader_id', 'article_id', 'left_id', 'right_id'))
                or type(row['story']) is not int or not 160 <= row['story'] < 200
                or type(row['depth']) is not int or row['depth'] not in (1, 2)
                or row['id'] in seen or len(set(one_key(row))) != 3):
            raise ValueError('Invalid or repeated grade-free inserted-block trial')
        cohort, arm = row['cohort'], row['arm']
        reader = cohort, row['reader_id']
        if reader_arms.setdefault(reader, arm) != arm:
            raise ValueError('The same reader appears in different inserted-block arms')
        seen.add(row['id'])
        conditions[cohort, row['depth'], *one_key(row)].add(arm)
    for cohort in COHORTS:
        for comparator in ('middle', 'late'):
            if not any(cell[0] == cohort and {'early', comparator} <= arms for cell, arms in conditions.items()):
                raise ValueError('Every registered early-versus-comparator pair needs shared ordered layout support')
    return rows, base.digest(assay_path), test_sha


def original_readout(assay, extraction, fit):
    receipt = json.loads(fit.read_text(encoding='utf-8'))
    seal_path = extraction / 'seal.json'
    seal = json.loads(seal_path.read_text(encoding='utf-8'))
    assay_hash = base.digest(assay / 'assay.json')
    extraction_hash = base.digest(seal_path)
    if (not seal['assay_sha256'] == receipt['assay_sha256'] == assay_hash
            or not seal['protocol_sha256'] == receipt['protocol_sha256'] == base.digest(base.PROTOCOL)
            or receipt['extraction_sha256'] != extraction_hash or seal['candidate_id'] != base.CHECKPOINT
            or receipt['fit_mapping_sha256'] != base.digest(assay / 'fit.jsonl')
            or seal['stimuli_sha256'] != base.digest(assay / 'model_input/stimuli.jsonl')
            or any(seal['file_sha256'][name] != base.digest(extraction / name) for name in ('coordinates.jsonl', 'vectors.npy'))):
        raise ValueError('Expected the original fit, authored layouts and seal-bound app projection')
    indices, states = base.read_vectors(extraction)
    original = base.read_rows(assay / 'fit.jsonl')
    ids = {key for row in original for key in one_key(row)}
    layouts = {row['id']: row for row in base.read_rows(assay / 'model_input/stimuli.jsonl') if row['id'] in ids}
    examples = examples_from_original(original, layouts, indices, states)
    cue = centroid(examples['cafe_cue'])
    current = {depth: centroid(examples[f'current_event_depth_{depth}']) for depth in OPTIONS}
    candidates = {depth: centroid([row for row in examples['candidate_event'] if row[1] in nodes])
                  for depth, nodes in OPTIONS.items()}
    params = receipt['params']
    basis = {tuple(entry): index for index, entry in enumerate(params['basis'])}
    structure = np.asarray(params['coefficients']['structure'], dtype=np.float64)
    text = np.asarray(params['coefficients']['text'], dtype=np.float64)
    if (not basis or len(basis) != len(params['basis']) or structure.shape != (2 + 2 * len(basis),)
            or text.shape != (1 + 3 * base.WIDTH + 2,) or not np.isfinite(structure).all() or not np.isfinite(text).all()):
        raise ValueError('Frozen structural and text coefficients have inconsistent dimensions')
    return indices, states, cue, current, candidates, basis, structure, text


def predict_layout(key, depth, indices, states, cue, current, candidates, basis, structure, text):
    story, left, right = (base.vectors_for(indices, states, identifier) for identifier in key)
    if (len(story), len(left), len(right)) != (depth + 2, 1, 1):
        raise ValueError('Expected depth-specific story vectors and independent one-stop options')
    nodes = tuple(tag(current[step], story[step + 1]) for step in range(1, depth + 1))
    family = '/'.join((tag(cue, story[1]), *nodes))
    left_node, right_node = (tag(candidates[depth], vector[0]) for vector in (left, right))
    cell = (family, *sorted((left_node, right_node)))
    if left_node == right_node or cell not in basis:
        raise ValueError('A new ordered layout has no identifiable original structural cell')
    indicator = np.zeros(len(basis))
    indicator[basis[cell]] = (1 if left_node < right_node else -1) * np.sqrt(len(basis))
    logits = {group: float(np.concatenate(([1], indicator, [group], group * indicator)) @ structure) for group in (-1, 1)}
    text_logit = float(np.concatenate(([1], base.feature_vector(key, indices, states))) @ text)
    if not np.isfinite([*logits.values(), text_logit]).all():
        raise ValueError('Expected finite frozen forecast logits')
    probabilities = {group: float(expit(logit)) for group, logit in logits.items()}
    marginal = 0.5 * (probabilities[-1] + probabilities[1])
    text_probability = float(expit(text_logit))
    if not all(0 < value < 1 for value in (*probabilities.values(), marginal, text_probability)):
        raise ValueError('Expected strict finite probabilities of choosing the right-hand event')
    return probabilities, marginal, text_probability


def run(assay, base_assay, base_extraction, base_fit, output):
    if output.exists():
        raise FileExistsError(output)
    rows, assay_sha, test_sha = grade_free_trials(assay)
    readout = original_readout(base_assay, base_extraction, base_fit)
    keys = {(one_key(row), row['depth']) for row in rows}
    if any(identifier not in readout[0] for key, _ in keys for identifier in key):
        raise ValueError('Forecast requires exclusively previously certified coordinate IDs')
    layouts = {(key, depth): predict_layout(key, depth, *readout) for key, depth in sorted(keys)}
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix='.beukers-inserted-forecast-') as name:
        stage = Path(name) / 'forecasts'
        stage.mkdir()
        path = stage / 'forecasts.jsonl'
        with path.open('x', encoding='utf-8') as stream:
            for row in rows:
                probabilities, marginal, text_probability = layouts[one_key(row), row['depth']]
                result = {'id': row['id'], 'methods': {'initial_block': probabilities[ARMS[row['arm']]],
                                                        'empty_context': marginal, 'text': text_probability}}
                stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True) + '\n')
        seal = {'assay_sha256': assay_sha, 'test_sha256': test_sha, 'forecasts_sha256': base.digest(path),
                'protocol_sha256': base.digest(PROTOCOL), 'base_fit_sha256': base.digest(base_fit),
                'base_extraction_sha256': base.digest(base_extraction / 'seal.json'),
                'forecast_code_sha256': base.digest(Path(__file__))}
        (stage / 'forecast-seal.json').write_text(json.dumps(seal, sort_keys=True, indent=2, allow_nan=False) + '\n',
                                                 encoding='utf-8')
        stage.rename(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('assay', 'base-assay', 'base-extraction', 'base-fit', 'output'):
        parser.add_argument(f'--{flag}', type=Path, required=True)
    run(**vars(parser.parse_args()))
