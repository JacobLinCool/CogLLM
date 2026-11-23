"""Seal TextComplexityDE forecasts from certified states and score them locally."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import tarfile

import numpy as np
from scipy.special import log_softmax

from research.core_features.readback_score import BOOTSTRAP_DRAWS, BOOTSTRAP_SEED, paired_material_interval
from research.semantic_tracking.iirc_linked_pack import write_archive
from research.semantic_tracking.iirc_prepare import digest
from research.stage2.records import write_jsonl
from research.stage3.probe import TARGETS, _jsonl
from research.stage3.run import MODEL, REVISION, TRANSFORMERS_VERSION, save_json

from .textcomplexityde_transfer import HEAD, LEVELS, PROTOCOL, RATINGS, WIDTH


AMENDMENT = Path(__file__).with_name('textcomplexityde-transfer-amendment.md')
PARENT_ASSAY = '2294f015450fe398517ef52cadd6ae99a132bf229951ec643e01df4fa2fbe481'
MEMBERS = ('LICENSE', 'attribution.json', 'heads.npz', 'stimuli.jsonl')
PRIMARY = 'plaba_ordinal'
BASELINE = 'plaba_length'
SECONDARY = 'plaba_effect_calibrated'
METHODS = (BASELINE, PRIMARY, SECONDARY)
COVERAGE = 0.95


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def archive_hashes(source, members):
    with tarfile.open(source, 'r:gz') as archive:
        result = {}
        for member in archive:
            if not member.isfile() or member.name not in members or member.name in result:
                raise ValueError(f'Unexpected TextComplexityDE payload member: {member.name}')
            with archive.extractfile(member) as stream:
                result[member.name] = hashlib.file_digest(stream, 'sha256').hexdigest()
    return result


def amend(base, output, report):
    base_assay = base / 'assay.json'
    if digest(base_assay) != PARENT_ASSAY:
        raise ValueError('The primary TextComplexityDE assay differs from its first seal')
    original = read_json(base_assay)
    original_archive = base / 'model-inputs.tar.gz'
    if archive_hashes(original_archive, {*MEMBERS, 'assay.json'}) != {
            'assay.json': PARENT_ASSAY, **original['model_input_hashes']}:
        raise ValueError('Grade-free primary input archive differs from its first seal')

    source_head = HEAD / 'head.npz'
    if digest(source_head) != original['plaba_ordinal_head_sha256']:
        raise ValueError('The two PLABA heads must come from the frozen PLABA checkpoint')
    with np.load(source_head, allow_pickle=False) as source, np.load(base / 'heads.npz', allow_pickle=False) as prior:
        weights = {key: prior[key] for key in prior.files}
        secondary = source['effect_calibrated_weights']
        if (digest(base / 'heads.npz') != original['model_input_hashes']['heads.npz']
                or not np.array_equal(weights['weights'], source['weights'])
                or secondary.shape != weights['weights'].shape or not np.isfinite(secondary).all()):
            raise ValueError('The primary coefficient block or fixed secondary shape differs')

    output.mkdir(parents=True, exist_ok=False)
    with tarfile.open(original_archive, 'r:gz') as archive:
        for name in ('LICENSE', 'attribution.json', 'stimuli.jsonl'):
            with archive.extractfile(name) as stream:
                (output / name).write_bytes(stream.read())
    np.savez_compressed(output / 'heads.npz', **weights, effect_calibrated_weights=secondary)
    paths = {name: output / name for name in MEMBERS}
    assay_path = output / 'assay.json'
    amended = {**original, 'version': 2, 'parent_assay_sha256': PARENT_ASSAY,
               'amendment_sha256': digest(AMENDMENT), 'model_input_hashes': {name: digest(path) for name, path in paths.items()},
               'forecast_methods': list(METHODS), 'primary_method': PRIMARY, 'length_baseline_method': BASELINE,
               'secondary_method': SECONDARY, 'secondary_checkpoint_array': 'effect_calibrated_weights'}
    save_json(assay_path, amended)
    paths['assay.json'] = assay_path
    payload = output / 'model-inputs.tar.gz'
    write_archive(payload, paths)
    verified = archive_hashes(payload, paths)
    if verified != {**amended['model_input_hashes'], 'assay.json': digest(assay_path)}:
        raise ValueError('Amended archive member checksum mismatch')
    save_json(report, {**amended, 'archive_members_sha256': verified, 'archive_sha256': digest(payload)})


def load_inputs(inputs):
    assay_path = inputs / 'assay.json'
    assay = read_json(assay_path)
    if (assay['version'] != 2 or assay['parent_assay_sha256'] != PARENT_ASSAY
            or assay['model'] != MODEL or assay['revision'] != REVISION or assay['forecast_methods'] != list(METHODS)
            or assay['primary_method'] != PRIMARY or assay['length_baseline_method'] != BASELINE
            or assay['secondary_method'] != SECONDARY or assay['protocol_sha256'] != digest(PROTOCOL)
            or assay['amendment_sha256'] != digest(AMENDMENT) or set(assay['model_input_hashes']) != set(MEMBERS)):
        raise ValueError('Unknown external diagnostic protocol or archive format')
    for name in MEMBERS:
        if digest(inputs / name) != assay['model_input_hashes'][name]:
            raise ValueError(f'TextComplexityDE input differs from the amended assay: {name}')
    documents = {}
    for row in _jsonl(inputs / 'stimuli.jsonl'):
        case, text = row['id'], row['text']
        if (row.keys() != {'id', 'text', 'stops'} or case in documents or not text.strip()
                or row['stops'] != [len(text)]):
            raise ValueError(f'Invalid isolated test sentence or exact stop: {case}')
        documents[case] = text
    attribution = read_json(inputs / 'attribution.json')
    groups = attribution['article_by_stimulus']
    if (documents.keys() != groups.keys() or len(documents) != assay['cases']
            or set(groups.values()) != attribution['article_attribution'].keys()
            or len(set(groups.values())) != assay['article_groups']):
        raise ValueError('Article attribution differs from the sealed document inventory')
    return assay, digest(assay_path), documents, groups


def certified(readout, assay, documents):
    manifest = read_json(readout)
    if (manifest['model'] != MODEL or manifest['revision'] != REVISION or manifest['transformers'] != TRANSFORMERS_VERSION
            or manifest['gpu'] != 'NVIDIA L4' or manifest['dtype'] != 'float32' or manifest['segment_means'] is not False
            or manifest['input_sha256'] != assay['model_input_hashes']['stimuli.jsonl']
            or manifest['input_texts'] != len(documents)):
        raise ValueError('Expected the pinned sparse L4 extraction of the sealed sentences')
    excluded = {}
    for row in manifest['excluded_stops']:
        case = row['id']
        if case in excluded or case not in documents or row['stop'] != len(documents[case]):
            raise ValueError('Unexpected excluded sentence-end stop')
        excluded[case] = row['reason']
    included = set(documents) - excluded.keys()
    if not included or len(included) != manifest['unique_certified_stops'] or len(included) != manifest['encoded_texts']:
        raise ValueError('Incomplete isolated certified sentence inventory')
    return included, excluded


def load_states(features, documents, included):
    with np.load(features, allow_pickle=False) as archive:
        if set(archive.files) != {'ids', 'stops', 'states'}:
            raise ValueError('Expected a sparse causal feature archive')
        ids, stops, values = archive['ids'], archive['stops'], archive['states']
    if (ids.shape != (len(included),) or ids.dtype.kind != 'U' or set(ids.tolist()) != included
            or stops.dtype.kind not in 'iu' or not np.array_equal(stops, [len(documents[case]) for case in ids])
            or values.shape != (len(included), WIDTH) or values.dtype != np.float32):
        raise ValueError('Unexpected TextComplexityDE state coordinates, dimensions, or dtype')
    normalized = values.astype(np.float64)
    norms = np.linalg.norm(normalized, axis=1, keepdims=True)
    if not np.all(np.isfinite(norms) & (norms > 0)):
        raise ValueError('Expected finite nonzero state at each certified position')
    return ids.tolist(), normalized / norms


def predict(heads, texts, hidden):
    expected = {'length_only_weights': (len(TARGETS), 3, len(LEVELS)),
                'weights': (len(TARGETS), WIDTH + 3, len(LEVELS)),
                'effect_calibrated_weights': (len(TARGETS), WIDTH + 3, len(LEVELS)),
                'length_mean': (2,), 'length_scale': (2,)}
    with np.load(heads, allow_pickle=False) as archive:
        if (set(archive.files) != {*expected, 'targets', 'ratings'} or tuple(archive['targets']) != TARGETS
                or not np.array_equal(archive['ratings'], LEVELS)):
            raise ValueError('Unexpected PLABA weight or target schema')
        blocks = {name: archive[name] for name in expected}
    if (any(value.shape != expected[name] or value.dtype != np.float64 or not np.isfinite(value).all()
            for name, value in blocks.items()) or np.any(blocks['length_scale'] <= 0)):
        raise ValueError('Unexpected PLABA coefficient dimensions, dtype, or scales')
    length = np.log1p([(len(text), len(text.split())) for text in texts])
    scaled = (length - blocks['length_mean']) / blocks['length_scale'] / np.sqrt(2)
    first = np.ones(len(texts))
    basic = np.column_stack((first, scaled))
    full = np.column_stack((first, hidden, scaled))
    return {method: (np.exp(log_softmax(np.einsum('nf,tfr->ntr', design, blocks[weights]), axis=-1)) @ LEVELS)
            for method, design, weights in ((BASELINE, basic, 'length_only_weights'), (PRIMARY, full, 'weights'),
                                            (SECONDARY, full, 'effect_calibrated_weights'))}


def forecast(inputs, features, output):
    assay, assay_hash, documents, _ = load_inputs(inputs)
    extraction = features / 'extraction.json'
    readouts = features / 'features.npz'
    included, _ = certified(extraction, assay, documents)
    ids, hidden = load_states(readouts, documents, included)
    scores = predict(inputs / 'heads.npz', [documents[case] for case in ids], hidden)
    output.mkdir(parents=True, exist_ok=False)
    path = output / 'predictions.jsonl'
    write_jsonl(path, ({'id': case, 'stop': len(documents[case]),
                        'methods': {name: scores[name][index].tolist() for name in METHODS}}
                       for index, case in enumerate(ids)))
    save_json(output / 'seal.json', {
        'model': MODEL, 'revision': REVISION, 'assay_sha256': assay_hash, 'forecasts_sha256': digest(path),
        'extraction_sha256': digest(extraction), 'features_sha256': digest(readouts),
        'heads_sha256': assay['model_input_hashes']['heads.npz'], 'evaluation_code_sha256': digest(Path(__file__)),
        'forecasts': len(ids), 'target_order': list(TARGETS), 'methods': list(METHODS),
    })


def read_forecasts(path, documents, included):
    predictions = {}
    for row in _jsonl(path):
        case = row['id']
        if (row.keys() != {'id', 'stop', 'methods'} or case not in included or case in predictions
                or row['stop'] != len(documents[case]) or set(row['methods']) != set(METHODS)):
            raise ValueError('Unexpected text, position, or method in the sealed forecast')
        vector = np.asarray([row['methods'][name] for name in METHODS], dtype=np.float64)
        if vector.shape != (len(METHODS), len(TARGETS)) or not np.all(np.isfinite(vector) & (1 <= vector) & (vector <= 5)):
            raise ValueError('Expected finite PLABA scores between one and five')
        predictions[case] = vector
    if predictions.keys() != included:
        raise ValueError('Forecasts omit or add certified positions')
    return predictions


def read_outcomes(path, groups):
    result = {}
    for row in _jsonl(path):
        case = row['id']
        if (row.keys() != {'id', 'group_id', *RATINGS} or case in result or case not in groups
                or row['group_id'] != groups[case]):
            raise ValueError('Unexpected sentence or article in the external ratings')
        for key in RATINGS:
            observation = row[key]
            if (observation.keys() != {'mean', 'votes'} or type(observation['votes']) is not int
                    or observation['votes'] < 1 or not 1 <= observation['mean'] <= 7):
                raise ValueError('Unexpected external aggregate rating or denominator')
        result[case] = [row[key]['mean'] for key in RATINGS]
    if result.keys() != groups.keys():
        raise ValueError('External ratings differ from the exact source inventory')
    return result


def rank_diagnostic(predictions, outcomes, groups):
    article_cases = defaultdict(list)
    for case in sorted(predictions):
        article_cases[groups[case]].append(case)
    metrics = {}
    for index, target in enumerate(TARGETS):
        articles, agreements, counts = [], [], []
        for group, ids in article_cases.items():
            human = np.array([outcomes[case][index] for case in ids])
            values = np.stack([predictions[case][:, index] for case in ids])
            left, right = np.triu_indices(len(ids), 1)
            signs = np.sign(human[left] - human[right])
            observed = signs != 0
            if not observed.any():
                continue
            # Higher PLABA scores indicate easier sentences, opposite to the human difficulty direction.
            delta = values[right[observed]] - values[left[observed]]
            pairs = np.where(np.abs(delta) <= 1e-12, 0.5, np.sign(delta) == signs[observed, None])
            agreements.append(pairs.mean(axis=0))
            articles.append({'group_id': group})
            counts.append(int(observed.sum()))
        aggregates = np.asarray(agreements).reshape(-1, len(METHODS))
        deltas = {}
        for model, baseline in ((PRIMARY, BASELINE), (SECONDARY, BASELINE), (SECONDARY, PRIMARY)):
            delta = aggregates[:, METHODS.index(model)] - aggregates[:, METHODS.index(baseline)]
            deltas[f'{model}_minus_{baseline}'] = paired_material_interval(delta, articles) if len(articles) >= 2 else None
        comparison = deltas[f'{PRIMARY}_minus_{BASELINE}']
        metrics[target] = {
            'eligible_articles': len(articles), 'eligible_pairs': sum(counts),
            'mean_article_concordance': {name: float(aggregates[:, index].mean()) if len(articles) else None
                                         for index, name in enumerate(METHODS)},
            'paired_concordance_differences': deltas,
            'primary_gate_passed': comparison is not None and comparison['descriptive_paired_material_95_ci'][0] > 0,
        }
    return metrics


def score(inputs, run, extraction, outcomes, output):
    assay, assay_hash, documents, groups = load_inputs(inputs)
    path = run / 'predictions.jsonl'
    seal = read_json(run / 'seal.json')
    if (seal['model'] != MODEL or seal['revision'] != REVISION or seal['assay_sha256'] != assay_hash
            or seal['forecasts_sha256'] != digest(path) or seal['extraction_sha256'] != digest(extraction)
            or seal['heads_sha256'] != assay['model_input_hashes']['heads.npz']
            or seal['evaluation_code_sha256'] != digest(Path(__file__)) or seal['target_order'] != list(TARGETS)
            or seal['methods'] != list(METHODS)):
        raise ValueError('Forecaster artifacts differ from their pre-scoring seal')
    included, excluded = certified(extraction, assay, documents)
    predictions = read_forecasts(path, documents, included)
    if seal['forecasts'] != len(predictions):
        raise ValueError('Sealed forecast count differs from the certified stop inventory')
    if digest(outcomes) != assay['heldout_outcomes_sha256']:
        raise ValueError('Local human rating file differs from the locked outcome seal')
    measured = read_outcomes(outcomes, groups)
    metrics = rank_diagnostic(predictions, measured, groups)
    coverage = len(predictions) / len(documents)
    save_json(output, {
        'scope': assay['scope'], 'assay_sha256': assay_hash, 'forecasts_sha256': seal['forecasts_sha256'],
        'heldout_outcomes_sha256': assay['heldout_outcomes_sha256'], 'certified_sentences': len(predictions),
        'coverage': coverage, 'coverage_gate_passed': coverage >= COVERAGE, 'excluded_stops': excluded,
        'bootstrap': {'unit': 'source article', 'draws': BOOTSTRAP_DRAWS, 'seed': BOOTSTRAP_SEED},
        'targets': metrics, 'primary_conditions_passed': coverage >= COVERAGE
        and all(value['primary_gate_passed'] for value in metrics.values()),
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest='command', required=True)
    for name, fields in (('amend', ('base', 'output', 'report')), ('forecast', ('inputs', 'features', 'output')),
                         ('score', ('inputs', 'run', 'extraction', 'outcomes', 'output'))):
        command = subcommands.add_parser(name)
        for field in fields:
            command.add_argument(f'--{field}', type=Path, required=True)
    options = vars(parser.parse_args())
    {'amend': amend, 'forecast': forecast, 'score': score}[options.pop('command')](**options)
