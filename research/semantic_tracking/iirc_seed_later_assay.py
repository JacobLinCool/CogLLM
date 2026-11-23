"""Prepare, fit and score the IIRC same-paragraph later-support assay."""

import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from research.semantic_tracking.iirc_linked_pack import METHODS, write_archive
from research.semantic_tracking.iirc_prepare import digest, normalized
from research.semantic_tracking.iirc_seed_later import TOKEN_STRIDE, certified_stops
from research.stage2.records import write_jsonl
from research.stage3.probe import _jsonl
from research.stage3.run import MODEL, REVISION, save_json


SPLITS = ('train', 'dev')
PREFIX_CONTROL_TOKENS = 64
TRAIN_TARGETS = 'train/intervals.jsonl'
DEV_TARGETS = 'dev/locked-intervals.jsonl'
STIMULI = tuple(f'{split}/stimuli.jsonl' for split in SPLITS)
PUBLIC = (*STIMULI, *(f'{split}/queries.jsonl' for split in SPLITS), TRAIN_TARGETS)


@dataclass(frozen=True)
class EncodedParagraph:
    text: str
    tokens: tuple[int, ...]
    offsets: tuple[tuple[int, int], ...]


def outcome(stops, support_end):
    index = bisect_left([end for _, end in stops], support_end)
    return {'lower': index, 'upper': index + 1 if index < len(stops) else None}


def prepare_split(stimuli_path, outcomes_path, tokenizer, expected):
    by_paragraph = defaultdict(list)
    for row in _jsonl(outcomes_path):
        if row['direction'] == 'later':
            by_paragraph[row['paragraph_id']].append(row)
    if sum(map(len, by_paragraph.values())) != expected:
        raise ValueError('Later-support cohort differs from the independent author audit')

    exclusions = Counter()
    stimuli, queries, locked = [], [], []
    token_count = 0
    for row in _jsonl(stimuli_path):
        paragraph = row['id']
        if paragraph not in by_paragraph:
            continue
        if row.keys() != {'id', 'text', 'stops'} or row['stops'] != [len(row['text'])]:
            raise ValueError(f'Expected exactly the original full paragraph: {paragraph}')
        source = tokenizer.encode(row['text'], add_special_tokens=False)
        encoded = EncodedParagraph(row['text'], tuple(source.ids), tuple(source.offsets))
        coordinates = set()
        for label in by_paragraph.pop(paragraph):
            start, end = (label['anchor'][key] for key in ('start', 'end'))
            support_start, support_end = (label[key] for key in ('support_start', 'support_end'))
            if not 0 <= start < end <= support_start < support_end <= len(encoded.text):
                raise ValueError(f'Invalid later support coordinates: {label["question_id"]}')
            try:
                anchor, stops = certified_stops(encoded, end, tokenizer)
            except ValueError as error:
                if str(error) not in {'Cannot certify the anchor stop', 'Cannot certify any stop after the anchor'}:
                    raise
                exclusions[str(error)] += 1
                continue
            case = label['question_id']
            queries.append({'case_id': case, 'paragraph_id': paragraph, 'anchor_start': start, 'stops': (anchor, *stops)})
            locked.append({'case_id': case, 'article_cluster': label['article_family'], 'support_end': support_end,
                           **outcome(stops, support_end)})
            coordinates.update(char for _, char in (anchor, *stops))
        if coordinates:
            stimuli.append({'id': paragraph, 'text': encoded.text, 'stops': sorted(coordinates)})
            token_count += len(encoded.tokens)

    if by_paragraph or not queries or len(queries) + sum(exclusions.values()) != expected:
        raise ValueError('Could not account for all source paragraphs and author observations')
    if len({row['case_id'] for row in queries}) != len(queries):
        raise ValueError('Repeated question ID in later-support cohort')
    same_anchor = Counter((row['paragraph_id'], row['stops'][0]) for row in queries)
    coverage = {'questions': len(queries), 'paragraphs': len(stimuli), 'tokens': token_count,
                'stops': sum(len(row['stops']) for row in stimuli), 'exclusions': dict(exclusions),
                'article_families': len({row['article_cluster'] for row in locked}),
                'questions_at_shared_anchors': sum(count for count in same_anchor.values() if count > 1),
                'observed': sum(row['upper'] is not None for row in locked),
                'right_censored': sum(row['upper'] is None for row in locked)}
    return stimuli, queries, locked, coverage


def prepare(source, seed_audit, later_audit, snapshot, output):
    from research.semantic_tracking.iirc_linked_prepare import load_tokenizer

    previous = json.loads(seed_audit.read_text(encoding='utf-8'))
    audit = json.loads(later_audit.read_text(encoding='utf-8'))
    if (audit['author_archive_sha256'] != previous['author_archive']['sha256']
            or audit['independent_seed_evidence_audit_sha256'] != digest(seed_audit)
            or not audit['independent_author_release_reconstruction']):
        raise ValueError('Expected the independently reconstructed IIRC release inventory')
    tokenizer = load_tokenizer(snapshot).backend_tokenizer
    splits = {}
    for split in SPLITS:
        names = (f'{split}/stimuli.jsonl', f'{split}/locked-evidence.jsonl')
        if any(digest(source / name) != previous['files_sha256'][name] for name in names):
            raise ValueError(f'{split} source data changed after its published-release audit')
        splits[split] = prepare_split(*(source / name for name in names), tokenizer,
                                      audit['cohorts'][split]['questions'])
    for key, group, function in (('family', 2, lambda row: row['article_cluster']),
                                 ('paragraph', 0, lambda row: row['id']),
                                 ('normalized text', 0, lambda row: normalized(row['text'])),
                                 ('question', 1, lambda row: row['case_id'])):
        if {function(row) for row in splits['train'][group]} & {function(row) for row in splits['dev'][group]}:
            raise ValueError(f'Official train and dev overlap by {key}')

    output.mkdir(parents=True, exist_ok=False)
    for split, (stimuli, queries, locked, _) in splits.items():
        location = output / split
        location.mkdir()
        write_jsonl(location / 'stimuli.jsonl', stimuli)
        write_jsonl(location / 'queries.jsonl', queries)
        if split == 'train':
            write_jsonl(output / TRAIN_TARGETS, ({key: row[key] for key in ('case_id', 'lower', 'upper')}
                                                  for row in locked))
        else:
            write_jsonl(output / DEV_TARGETS, locked)

    assay = output / 'assay.json'
    save_json(assay, {
        'kind': 'iirc-seed-later-v1', 'model': MODEL, 'revision': REVISION,
        'author_later_audit_sha256': digest(later_audit), 'nominal_token_stride': TOKEN_STRIDE,
        'fixed_prefix_control_tokens_since_anchor': PREFIX_CONTROL_TOKENS,
        'scope': 'First certified stop with complete coverage of a retrospective later support within the same paragraph, '
                 'conditional on a supplied anchor and the presence of an eventual annotated support.',
        'methods': list(METHODS), 'cases': {split: parts[3]['paragraphs'] for split, parts in splits.items()},
        'tokens': {split: parts[3]['tokens'] for split, parts in splits.items()},
        'inputs_sha256': {name: digest(output / name) for name in PUBLIC},
        'held_out_targets_sha256': digest(output / DEV_TARGETS),
        'coverage': {split: parts[3] for split, parts in splits.items()},
        'strict_same_paragraph_swap_pairs_before_alignment': {
            split: audit['cohorts'][split]['same_paragraph_identity_swap_pairs'] for split in SPLITS},
    })
    archive = output / 'model-inputs.tar.gz'
    write_archive(archive, {name: output / name for name in (*PUBLIC, 'assay.json')})
    save_json(output / 'package.json', {'archive_sha256': digest(archive), 'public_members': sorted((*PUBLIC, 'assay.json'))})


def read_inputs(inputs):
    assay = inputs / 'assay.json'
    seal = json.loads(assay.read_text(encoding='utf-8'))
    if (seal['kind'] != 'iirc-seed-later-v1' or seal['model'] != MODEL or seal['revision'] != REVISION
            or seal['nominal_token_stride'] != TOKEN_STRIDE or seal['methods'] != list(METHODS)
            or seal['fixed_prefix_control_tokens_since_anchor'] != PREFIX_CONTROL_TOKENS
            or set(seal['inputs_sha256']) != set(PUBLIC)
            or any(digest(inputs / name) != seal['inputs_sha256'][name] for name in PUBLIC)):
        raise ValueError('Unexpected or changed same-paragraph assay inputs')
    queries, paragraphs = {}, {}
    for split in SPLITS:
        paragraphs[split] = list(_jsonl(inputs / f'{split}/stimuli.jsonl'))
        planned = {row['id']: set(row['stops']) for row in paragraphs[split]}
        remaining = {key: stops.copy() for key, stops in planned.items()}
        queries[split] = {}
        for row in _jsonl(inputs / f'{split}/queries.jsonl'):
            key, paragraph, stops = row['case_id'], row['paragraph_id'], row['stops']
            if set(row) != {'case_id', 'paragraph_id', 'anchor_start', 'stops'} or key in queries[split] or len(stops) < 2:
                raise ValueError(f'Unexpected query input: {key}')
            for left, right in zip(stops, stops[1:]):
                if not left[0] < right[0] or not left[1] < right[1]:
                    raise ValueError(f'Causal cutpoints must increase: {key}')
            ends = {end for _, end in stops}
            if not 0 <= row['anchor_start'] < stops[0][1] or not ends <= planned[paragraph]:
                raise ValueError(f'Query cuts do not match the supplied paragraph: {key}')
            remaining[paragraph].difference_update(ends)
            queries[split][key] = row
        if any(remaining.values()) or len(planned) != seal['cases'][split] or len(queries[split]) != seal['coverage'][split]['questions']:
            raise ValueError(f'Paragraph and query inventories disagree in {split}')
    return seal, digest(assay), queries, paragraphs


def make_paths(queries, features, feature_index, *, prefix=False):
    from research.semantic_tracking.iirc_linked_fit import Timeline

    paths, indices = [], []
    for key, query in queries.items():
        stops = query['stops']
        anchor = stops[0][0]
        if prefix:
            stops = [stop for stop in stops if stop[0] - anchor <= PREFIX_CONTROL_TOKENS]
            if len(stops) == 1:
                continue
        paths.append(Timeline(key, tuple(token for token, _ in stops), anchor))
        indices.extend(feature_index[query['paragraph_id'], end] for _, end in stops)
    return paths, features[indices]


def fit(inputs, features, extraction, output):
    from scipy.special import expit
    from research.semantic_tracking.iirc_linked_fit import (
        GEOMETRY_WIDTH, PENALTY, Timeline, design, fit as fit_head, read_features, risk_rows,
    )

    if output.exists():
        raise FileExistsError(output)
    seal, assay_hash, queries, paragraphs = read_inputs(inputs)
    features_hash = digest(features)
    receipt = json.loads(extraction.read_text(encoding='utf-8'))
    expected = {'assay': assay_hash, **{name: seal['inputs_sha256'][name] for name in STIMULI}}
    if (receipt['features_sha256'] != features_hash or receipt['assay_sha256'] != assay_hash
            or receipt['model'] != MODEL or receipt['revision'] != REVISION or receipt['inputs_sha256'] != expected):
        raise ValueError('Same-paragraph features do not match the sealed plain-text assay')
    paths = {split: [Timeline(row['id'], tuple(row['stops']), 0) for row in paragraphs[split]] for split in SPLITS}
    states, projection = read_features(features, paths)
    feature_index = {key: index for index, key in enumerate((row.case, end) for split in SPLITS
                                                           for row in paths[split] for end in row.stops)}
    timelines = {split: make_paths(queries[split], states, feature_index) for split in SPLITS}
    train_targets = {}
    for row in _jsonl(inputs / TRAIN_TARGETS):
        key, lower, upper = (row[name] for name in ('case_id', 'lower', 'upper'))
        maximum = len(queries['train'][key]['stops']) - 1
        if (set(row) != {'case_id', 'lower', 'upper'} or key in train_targets or type(lower) is not int
                or not 0 <= lower <= maximum or (upper is None and lower != maximum)
                or (upper is not None and (type(upper) is not int or upper != lower + 1 or upper > maximum))):
            raise ValueError(f'Invalid training interval: {key}')
        train_targets[key] = lower, upper
    if train_targets.keys() != queries['train'].keys():
        raise ValueError('Incomplete same-paragraph training targets')

    train, dev = (risk_rows(timelines[split][0], 0, train_targets if split == 'train' else None) for split in SPLITS)
    prefix_paths, prefix_states = make_paths(queries['dev'], states, feature_index, prefix=True)
    if not prefix_paths:
        raise ValueError('No certified opportunities for the fixed-horizon control')
    prefix = risk_rows(prefix_paths, 0)
    center = train.geometry.mean(axis=0)
    scale = np.maximum(train.geometry.std(axis=0), 0.1)

    def features_for(rows, embeddings, method):
        geometry = (rows.geometry - center) / (scale * np.sqrt(GEOMETRY_WIDTH))
        return design(rows, embeddings, geometry, method)

    weights, diagnostics, all_hazards = {}, {}, {}
    for method in METHODS:
        weights[method], diagnostics[method] = fit_head(features_for(train, timelines['train'][1], method), train)
        hazards = expit(features_for(dev, timelines['dev'][1], method) @ weights[method])
        if not np.all(np.isfinite(hazards) & (0 < hazards) & (hazards < 1)):
            raise ValueError(f'Invalid held-out hazards for {method}')
        all_hazards[method] = dict(zip(queries['dev'], np.split(hazards, dev.case_ends[:-1]), strict=True))
        truncated = expit(features_for(prefix, prefix_states, method) @ weights[method])
        expected_prefix = np.concatenate([all_hazards[method][path.case][:len(path.stops) - 1] for path in prefix_paths])
        if not np.allclose(truncated, expected_prefix, rtol=0, atol=1e-12):
            raise ValueError(f'Later features changed earlier probabilities for {method}')

    output.mkdir(parents=True, exist_ok=False)
    heads = output / 'heads.npz'
    forecasts = output / 'predictions.jsonl'
    np.savez_compressed(heads, **weights, projection=projection, geometry_center=center, geometry_scale=scale)
    write_jsonl(forecasts, ({'case_id': key, 'methods': {method: all_hazards[method][key].tolist() for method in METHODS}}
                            for key in queries['dev']))
    save_json(output / 'fit-manifest.json', {
        'model': MODEL, 'revision': REVISION, 'assay_sha256': assay_hash, 'features_sha256': features_hash,
        'extraction_sha256': digest(extraction), 'heads_sha256': digest(heads), 'predictions_sha256': digest(forecasts),
        'train_cases': len(train.case_ends), 'dev_cases': len(dev.case_ends), 'prefix_control_cases': len(prefix.case_ends),
        'matched_ridge_penalty': PENALTY, 'fits': diagnostics,
    })


def score(inputs, outcomes, forecasts, fit_seal, output):
    from research.semantic_tracking.iirc_linked_score import score_case, summarize
    from research.semantic_tracking.joint_score import rows_by_key

    manifest, assay_hash, queries, _ = read_inputs(inputs)
    receipt = json.loads(fit_seal.read_text(encoding='utf-8'))
    if (receipt['model'] != MODEL or receipt['revision'] != REVISION or receipt['assay_sha256'] != assay_hash
            or receipt['predictions_sha256'] != digest(forecasts)
            or manifest['held_out_targets_sha256'] != digest(outcomes)):
        raise ValueError('Held-out sources or predictions fail their pre-scoring seal')
    targets = rows_by_key(outcomes, lambda row: row['case_id'])
    predictions = rows_by_key(forecasts, lambda row: row['case_id'])
    if targets.keys() != predictions.keys() or targets.keys() != queries['dev'].keys():
        raise ValueError('Held-out case inventory is incomplete')

    def compare(prefix):
        results, labels = [], []
        for key, query in queries['dev'].items():
            anchor, *stops = query['stops']
            if prefix:
                stops = [stop for stop in stops if stop[0] - anchor[0] <= PREFIX_CONTROL_TOKENS]
                if not stops:
                    continue
            label = targets[key]
            intervals = outcome(stops, label['support_end'])
            forecast = predictions[key]
            if prefix:
                forecast = {'case_id': key, 'methods': {k: v[:len(stops)] for k, v in forecast['methods'].items()}}
            elif any(label[name] != value for name, value in intervals.items()):
                raise ValueError(f'Locked targets do not match certified cuts: {key}')
            labels.append({**label, **intervals})
            results.append(score_case(forecast, intervals, len(stops)))
        observed = sum(label['upper'] is not None for label in labels)
        summary = summarize(results, labels)
        if prefix:
            del summary['all_conditions_passed'], summary['conditions']
        return {'cases': len(labels), 'observed': observed, 'right_censored': len(labels) - observed, **summary}

    save_json(output, {'scope': manifest['scope'], 'assay_sha256': assay_hash, 'fit_sha256': digest(fit_seal),
                       'primary': compare(False), 'fixed_64_token_prefix_control_descriptive': compare(True)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='command', required=True)
    for function, names in ((prepare, ('source', 'seed-audit', 'later-audit', 'snapshot', 'output')),
                            (fit, ('inputs', 'features', 'extraction', 'output')),
                            (score, ('inputs', 'outcomes', 'forecasts', 'fit-seal', 'output'))):
        command = actions.add_parser(function.__name__)
        command.set_defaults(callback=function)
        for name in names:
            command.add_argument(f'--{name}', required=True, type=Path)
    arguments = vars(parser.parse_args())
    arguments.pop('command')
    arguments.pop('callback')(**arguments)
